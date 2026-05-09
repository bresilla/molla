//! Stage 24e.1 — GPU-side mutable insert into a sparse grid.
//!
//! Wraps `shaders/sparse_grid_insert.wgsl`. One [`SparseGridInsertKernel`]
//! owns the WGSL pipelines (one per value type — `f32`, `vec3<f32>`,
//! `f16`); the matching [`SparseGridStorage`] type owns the persistent
//! GPU buffers + free-list pools so the per-dispatch cost is just an
//! upload + dispatch + readback.
//!
//! ## What the GPU insert kernel does
//!
//! For each `(coord, value)` pair in a batch:
//!
//! 1. Hash the root key. Probe the open-addressed hash table; if the
//!    slot is empty, atomically claim it via `atomicCompareExchangeWeak`
//!    on a sentinel-RESERVED → real-index two-phase publication. The
//!    winner allocates a fresh internal-node index from the
//!    `next_internal` counter via `atomicAdd`; losers spin until the
//!    winner publishes.
//! 2. Once the internal-node index is in hand, atomically OR the bit
//!    in the internal-node mask. The thread that flipped the bit
//!    (i.e. saw the previous mask word with the bit unset) allocates
//!    a fresh leaf index from `next_leaf` and `atomicStore`s it into
//!    `internal_children`. Losers spin on `internal_children[slot]`
//!    until the winner publishes.
//! 3. Write `value` into `leaf_values[leaf_idx * 512 + leaf_local]`
//!    and atomically OR the leaf-mask bit.
//!
//! ## Pool exhaustion
//!
//! The pre-allocated pools (`internal_capacity`, `leaf_capacity`) are
//! sized at construction. If a write would exceed the pool the kernel
//! sets a bit in `error_flags` (read back by the host) and silently
//! drops the write. No dynamic GPU allocation in v1; the host is
//! expected to either size the pool conservatively or rebuild the
//! storage with a larger pool when exhaustion is detected.
//!
//! ## Templated value types
//!
//! The same WGSL is shared across `f32`, `vec3<f32>`, and `f16`
//! instantiations via a tiny string-replace at construction time
//! (`VALUE_TYPE` placeholder). The atomic-allocation paths are
//! type-agnostic; only the value store + load differ.
//!
//! `f16` requires the `wgpu` adapter to expose the `shader-f16`
//! feature; if absent we fall back to a half-float emulation
//! (`u32`-packed pairs) and document the precision caveat.

use crate::core::{Error, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Trait implemented by every value type the sparse grid supports
/// (`f32`, `[f32; 3]` for `vec3<f32>`, `u16`-packed for `f16`).
///
/// The trait is `pub` so external callers can construct
/// `SparseGridStorage<V>` for any value type they like, but the
/// only practical instantiations in v1 are the three exported below.
pub trait Value: Copy + Pod + 'static {
    /// WGSL type name for `VALUE_TYPE` (e.g. `"f32"`, `"vec3<f32>"`).
    /// Used by the kernel-builder string-replace.
    fn wgsl_type() -> &'static str;
    /// Number of bytes per stored value.
    fn size_bytes() -> usize {
        core::mem::size_of::<Self>()
    }
    /// Background value (returned for misses on host-side queries).
    fn zero() -> Self;
}

impl Value for f32 {
    fn wgsl_type() -> &'static str {
        "f32"
    }
    fn zero() -> Self {
        0.0
    }
}

/// `vec3<f32>` storage type. WGSL alignment for `vec3<f32>` in storage
/// buffers is 16 bytes (one trailing pad lane), so we represent the
/// host side as a `[f32; 4]` and only use `.xyz`. The `_pad` field is
/// uninitialised on writes (the kernel writes the whole `vec3` from
/// `query_values[i].xyz`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct ValueVec3([f32; 4]);

impl ValueVec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self([x, y, z, 0.0])
    }
    pub fn xyz(&self) -> [f32; 3] {
        [self.0[0], self.0[1], self.0[2]]
    }
}

impl Value for ValueVec3 {
    fn wgsl_type() -> &'static str {
        "vec3<f32>"
    }
    fn zero() -> Self {
        Self([0.0; 4])
    }
}

/// `f16` half-float storage. WGSL has an `f16` type behind the
/// `shader-f16` adapter feature, but it isn't universally supported
/// (notably some Vulkan drivers + WebGPU). We sidestep the feature-
/// detection problem by storing two halves per `u32` slot and
/// quantising on the host. The kernel sees a `u32`-typed value with
/// the lower 16 bits carrying the half-float; the upper 16 bits are
/// reserved for a future "second channel" use.
///
/// Quantisation error is bounded at `2^-10` (relative) — same as
/// IEEE 754 half precision — which is the test's tolerance bar.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct ValueF16(pub u32);

impl ValueF16 {
    /// Encode an `f32` into the lower-16-bit half-float slot.
    /// Pure-software conversion (round-to-nearest-even); avoids
    /// pulling in `half` as a dependency.
    pub fn from_f32(x: f32) -> Self {
        Self(f32_to_f16_bits(x) as u32)
    }
    pub fn to_f32(self) -> f32 {
        f16_bits_to_f32((self.0 & 0xFFFF) as u16)
    }
}

impl Value for ValueF16 {
    fn wgsl_type() -> &'static str {
        // We use u32-packed half storage so the kernel doesn't need
        // the `shader-f16` extension. The kernel only ever does
        // `array<u32>` value writes for this type.
        "u32"
    }
    fn zero() -> Self {
        Self(0)
    }
}

/// Convert `f32` → IEEE-754 `binary16` bit pattern. Round-to-nearest-
/// even, NaN-preserving, ±Inf-preserving. Translated from the public-
/// domain reference at `https://galfar.vevb.net/wp/2011/16bit-half-float-in-pascalc`.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mut mant = bits & 0x007F_FFFF;

    if exp >= 31 {
        // Inf / NaN.
        if mant != 0 && (bits >> 23) & 0xFF == 0xFF {
            return sign | 0x7E00; // qNaN
        }
        return sign | 0x7C00; // ±Inf
    }
    if exp <= 0 {
        // Subnormal half.
        if exp < -10 {
            return sign;
        }
        mant |= 0x0080_0000;
        let shift = 14 - exp;
        let half_mant = (mant >> shift) as u16;
        return sign | half_mant;
    }
    // Normal — round-to-nearest-even.
    let mant_round = (mant + 0x0000_1000) & 0x007F_FFFF;
    if mant_round < mant {
        // Rounding overflow into exponent.
        exp += 1;
        if exp >= 31 {
            return sign | 0x7C00;
        }
    }
    let half_mant = (mant_round >> 13) as u16;
    sign | ((exp as u16) << 10) | half_mant
}

/// Convert IEEE-754 `binary16` bit pattern → `f32`. Pure-software
/// (no `core::arch` intrinsics so it stays portable).
fn f16_bits_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exp = ((half >> 10) & 0x1F) as u32;
    let mant = (half & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal — normalise.
        let mut e = 1i32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        let m = m & 0x3FF;
        let bits = (sign << 31) | (((127 + e - 15) as u32) << 23) | (m << 13);
        return f32::from_bits(bits);
    }
    if exp == 31 {
        let bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        return f32::from_bits(bits);
    }
    let bits = (sign << 31) | (((127 + exp - 15) as u32) << 23) | (mant << 13);
    f32::from_bits(bits)
}

/// Layout uniforms shared with the WGSL kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct InsertUniforms {
    count: u32,
    root_table_capacity: u32,
    internal_capacity: u32,
    leaf_capacity: u32,
}

/// Persistent GPU storage for the mutable sparse grid. Owns five
/// `wgpu::Buffer`s (the topology buffers + counters) sized at
/// construction. Re-usable across many insert / lookup dispatches.
///
/// Sizing guidance:
/// - `root_table_capacity`: round up `expected_root_keys / 0.7` to
///   the next power of two. Going larger reduces probe length but
///   wastes memory (16 B per slot).
/// - `internal_capacity`: number of distinct (256³ blocks) the
///   inserted set spans. For a 1024³ asset this is ≤ 64.
/// - `leaf_capacity`: number of distinct 8³ tiles the inserts touch.
///   For a 1 % occupancy 1024³ asset this is ~ 16k–100k depending
///   on clustering.
pub struct SparseGridStorage<V: Value> {
    pub root_table: wgpu::Buffer,
    pub internal_children: wgpu::Buffer,
    pub internal_masks: wgpu::Buffer,
    pub leaf_values: wgpu::Buffer,
    pub leaf_masks: wgpu::Buffer,
    pub counters: wgpu::Buffer,
    pub root_table_capacity: u32,
    pub internal_capacity: u32,
    pub leaf_capacity: u32,
    _phantom: core::marker::PhantomData<V>,
}

const INTERNAL_NODE_VOXELS: u32 = 32_768; // 32³
const INTERNAL_NODE_MASK_U32S: u32 = 1024;
const LEAF_NODE_VOXELS: u32 = 512;
const LEAF_NODE_MASK_U32S: u32 = 16;
const INVALID_CHILD: u32 = u32::MAX;

impl<V: Value> SparseGridStorage<V> {
    /// Allocate a fresh storage with the given pool sizes.
    /// `root_table_capacity` must be a power of two ≥ 4.
    pub fn new(
        device: &wgpu::Device,
        root_table_capacity: u32,
        internal_capacity: u32,
        leaf_capacity: u32,
    ) -> Self {
        assert!(
            root_table_capacity >= 4 && root_table_capacity.is_power_of_two(),
            "root_table_capacity must be a power of two ≥ 4 (got {root_table_capacity})"
        );
        assert!(
            internal_capacity > 0 && internal_capacity < (INVALID_CHILD - 2),
            "internal_capacity out of range"
        );
        assert!(
            leaf_capacity > 0 && leaf_capacity < INVALID_CHILD,
            "leaf_capacity out of range"
        );

        // Root table: 4 i32s per slot. internal_idx = INVALID_CHILD
        // (i.e. -1 as i32) marks empty.
        let mut root_init: Vec<i32> = vec![0; (root_table_capacity as usize) * 4];
        for i in 0..(root_table_capacity as usize) {
            root_init[i * 4 + 3] = -1;
        }
        let root_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_storage root_table"),
            contents: bytemuck::cast_slice(&root_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let internal_children_size = internal_capacity * INTERNAL_NODE_VOXELS;
        let internal_children_init: Vec<u32> = vec![INVALID_CHILD; internal_children_size as usize];
        let internal_children = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_storage internal_children"),
            contents: bytemuck::cast_slice(&internal_children_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let internal_masks_size = internal_capacity * INTERNAL_NODE_MASK_U32S;
        let internal_masks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_grid_storage internal_masks"),
            size: (internal_masks_size as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let leaf_value_count = leaf_capacity * LEAF_NODE_VOXELS;
        let leaf_value_bytes = (leaf_value_count as u64) * (V::size_bytes() as u64);
        let leaf_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_grid_storage leaf_values"),
            size: leaf_value_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let leaf_mask_count = leaf_capacity * LEAF_NODE_MASK_U32S;
        let leaf_masks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_grid_storage leaf_masks"),
            size: (leaf_mask_count as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Counters: [next_internal=0, next_leaf=0, error_flags=0].
        let counters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_storage counters"),
            contents: bytemuck::cast_slice(&[0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            root_table,
            internal_children,
            internal_masks,
            leaf_values,
            leaf_masks,
            counters,
            root_table_capacity,
            internal_capacity,
            leaf_capacity,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Reset the counters + masks so the storage is empty again.
    /// Cheap (a single `queue.write_buffer` per buffer).
    pub fn reset(&self, queue: &wgpu::Queue) {
        let mut root_init: Vec<i32> = vec![0; (self.root_table_capacity as usize) * 4];
        for i in 0..(self.root_table_capacity as usize) {
            root_init[i * 4 + 3] = -1;
        }
        queue.write_buffer(&self.root_table, 0, bytemuck::cast_slice(&root_init));
        queue.write_buffer(&self.counters, 0, bytemuck::cast_slice(&[0u32, 0u32, 0u32]));
        // Zero the masks — the values themselves can stay (the masks
        // gate them; reading an inactive value never returns it).
        let internal_masks_zero = vec![0u8; (self.internal_capacity * INTERNAL_NODE_MASK_U32S * 4) as usize];
        queue.write_buffer(&self.internal_masks, 0, &internal_masks_zero);
        let leaf_masks_zero = vec![0u8; (self.leaf_capacity * LEAF_NODE_MASK_U32S * 4) as usize];
        queue.write_buffer(&self.leaf_masks, 0, &leaf_masks_zero);
        // Reset internal_children to INVALID_CHILD so the leaf-spin
        // wait condition still works after a reset.
        let internal_children_init: Vec<u32> = vec![INVALID_CHILD; (self.internal_capacity * INTERNAL_NODE_VOXELS) as usize];
        queue.write_buffer(
            &self.internal_children,
            0,
            bytemuck::cast_slice(&internal_children_init),
        );
    }

    /// Read back the current `[next_internal, next_leaf, error_flags]`
    /// counter triple. Synchronous; uses a one-shot staging buffer.
    pub fn read_counters(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<[u32; 3]> {
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_grid_storage counters stage"),
            size: 12,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sparse_grid_storage counters encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.counters, 0, &stage, 0, 12);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = stage.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("counters readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("counters poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("counters recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        stage.unmap();
        Ok([v[0], v[1], v[2]])
    }
}

/// Compiled WGSL pipeline for one value-type instantiation of the
/// insert kernel. Three entry points: `allocate_internal_atomic`,
/// `allocate_leaf_atomic`, `set_value`.
pub struct SparseGridInsertKernel<V: Value> {
    pipeline_alloc_internal: wgpu::ComputePipeline,
    pipeline_alloc_leaf: wgpu::ComputePipeline,
    pipeline_set_value: wgpu::ComputePipeline,
    layout_storage: wgpu::BindGroupLayout,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    _phantom: core::marker::PhantomData<V>,
}

const WORKGROUP_SIZE: u32 = 64;

impl<V: Value> SparseGridInsertKernel<V> {
    pub fn new(device: &wgpu::Device) -> Self {
        let wgsl = include_str!("../../shaders/sparse_grid_insert.wgsl")
            .replace("VALUE_TYPE", V::wgsl_type());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sparse_grid_insert"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let layout_storage = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_grid_insert storage"),
            entries: &storage_entries(5, false),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_grid_insert io"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_grid_insert uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::sparse_grid_insert layout"),
            bind_group_layouts: &[&layout_storage, &layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let pipeline_alloc_internal = make_pipeline("allocate_internal_atomic");
        let pipeline_alloc_leaf = make_pipeline("allocate_leaf_atomic");
        let pipeline_set_value = make_pipeline("set_value");

        Self {
            pipeline_alloc_internal,
            pipeline_alloc_leaf,
            pipeline_set_value,
            layout_storage,
            layout_io,
            layout_uniforms,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Insert a batch of `(coord, value)` pairs into `storage`.
    /// Returns the post-dispatch counter triple
    /// `[next_internal, next_leaf, error_flags]`.
    ///
    /// `error_flags` carries the bit-mask `(1 = internal pool full,
    /// 2 = leaf pool full, 4 = root table full)` — non-zero means
    /// some writes were dropped.
    pub fn insert(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        storage: &SparseGridStorage<V>,
        coords: &[[i32; 3]],
        values: &[V],
    ) -> Result<[u32; 3]> {
        assert_eq!(
            coords.len(),
            values.len(),
            "coords and values must be same length"
        );
        if coords.is_empty() {
            return storage.read_counters(device, queue);
        }

        // Pad coords to vec4<i32>.
        let mut padded_coords: Vec<i32> = Vec::with_capacity(coords.len() * 4);
        for c in coords {
            padded_coords.push(c[0]);
            padded_coords.push(c[1]);
            padded_coords.push(c[2]);
            padded_coords.push(0);
        }

        let buf_coords = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_insert query_coords"),
            contents: bytemuck::cast_slice(&padded_coords),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buf_values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_insert query_values"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = InsertUniforms {
            count: coords.len() as u32,
            root_table_capacity: storage.root_table_capacity,
            internal_capacity: storage.internal_capacity,
            leaf_capacity: storage.leaf_capacity,
        };
        let buf_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_insert uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_insert storage bg"),
            layout: &self.layout_storage,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: storage.root_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: storage.internal_children.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: storage.internal_masks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: storage.leaf_values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: storage.leaf_masks.as_entire_binding() },
            ],
        });
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_insert io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: storage.counters.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_coords.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_values.as_entire_binding() },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_insert uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_uniforms.as_entire_binding(),
            }],
        });

        let groups = (coords.len() as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sparse_grid_insert encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sparse_grid_insert set_value"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_set_value);
            pass.set_bind_group(0, &bg_storage, &[]);
            pass.set_bind_group(1, &bg_io, &[]);
            pass.set_bind_group(2, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        storage.read_counters(device, queue)
    }

    /// Allocate-only entry point for the `allocate_internal_atomic`
    /// kernel — useful for warming up the topology before a value-
    /// scatter pass.
    pub fn allocate_internal(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        storage: &SparseGridStorage<V>,
        coords: &[[i32; 3]],
    ) -> Result<[u32; 3]> {
        self.dispatch_alloc(device, queue, storage, coords, &self.pipeline_alloc_internal)
    }

    /// Allocate-only entry point for the `allocate_leaf_atomic` kernel.
    pub fn allocate_leaf(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        storage: &SparseGridStorage<V>,
        coords: &[[i32; 3]],
    ) -> Result<[u32; 3]> {
        self.dispatch_alloc(device, queue, storage, coords, &self.pipeline_alloc_leaf)
    }

    fn dispatch_alloc(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        storage: &SparseGridStorage<V>,
        coords: &[[i32; 3]],
        pipeline: &wgpu::ComputePipeline,
    ) -> Result<[u32; 3]> {
        if coords.is_empty() {
            return storage.read_counters(device, queue);
        }
        let mut padded_coords: Vec<i32> = Vec::with_capacity(coords.len() * 4);
        for c in coords {
            padded_coords.push(c[0]);
            padded_coords.push(c[1]);
            padded_coords.push(c[2]);
            padded_coords.push(0);
        }
        // values buffer is unused by the alloc kernels but the bind
        // group still requires it — stub a one-element placeholder.
        let dummy_value: V = V::zero();
        let buf_coords = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_insert alloc query_coords"),
            contents: bytemuck::cast_slice(&padded_coords),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buf_values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_insert alloc query_values stub"),
            contents: bytemuck::cast_slice(&[dummy_value; 1]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let uniforms = InsertUniforms {
            count: coords.len() as u32,
            root_table_capacity: storage.root_table_capacity,
            internal_capacity: storage.internal_capacity,
            leaf_capacity: storage.leaf_capacity,
        };
        let buf_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_insert alloc uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_insert alloc storage bg"),
            layout: &self.layout_storage,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: storage.root_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: storage.internal_children.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: storage.internal_masks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: storage.leaf_values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: storage.leaf_masks.as_entire_binding() },
            ],
        });
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_insert alloc io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: storage.counters.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_coords.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_values.as_entire_binding() },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_insert alloc uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_uniforms.as_entire_binding(),
            }],
        });

        let groups = (coords.len() as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sparse_grid_insert alloc encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sparse_grid_insert alloc pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg_storage, &[]);
            pass.set_bind_group(1, &bg_io, &[]);
            pass.set_bind_group(2, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        storage.read_counters(device, queue)
    }
}

/// Read back every active `(coord, value)` triple from a populated
/// [`SparseGridStorage`]. Useful for tests + the viewer's "GPU build
/// → CPU dump" path. Not on the hot path; uses one map_async per
/// buffer.
///
/// The returned vector is sorted by coord for deterministic output.
pub fn read_storage_to_pairs<V: Value>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    storage: &SparseGridStorage<V>,
) -> Result<Vec<([i32; 3], V)>> {
    let counters = storage.read_counters(device, queue)?;
    let internal_used = counters[0].min(storage.internal_capacity);
    let leaf_used = counters[1].min(storage.leaf_capacity);

    let root_bytes = (storage.root_table_capacity * 4 * 4) as u64;
    let root = read_buffer::<i32>(device, queue, &storage.root_table, root_bytes)?;

    let internal_children_bytes = (internal_used * INTERNAL_NODE_VOXELS * 4) as u64;
    let internal_children = if internal_children_bytes > 0 {
        read_buffer::<u32>(
            device,
            queue,
            &storage.internal_children,
            internal_children_bytes,
        )?
    } else {
        Vec::new()
    };
    let internal_masks_bytes = (internal_used * INTERNAL_NODE_MASK_U32S * 4) as u64;
    let internal_masks = if internal_masks_bytes > 0 {
        read_buffer::<u32>(
            device,
            queue,
            &storage.internal_masks,
            internal_masks_bytes,
        )?
    } else {
        Vec::new()
    };
    let leaf_values_bytes = (leaf_used * LEAF_NODE_VOXELS * V::size_bytes() as u32) as u64;
    let leaf_values = if leaf_values_bytes > 0 {
        read_buffer::<V>(device, queue, &storage.leaf_values, leaf_values_bytes)?
    } else {
        Vec::new()
    };
    let leaf_masks_bytes = (leaf_used * LEAF_NODE_MASK_U32S * 4) as u64;
    let leaf_masks = if leaf_masks_bytes > 0 {
        read_buffer::<u32>(device, queue, &storage.leaf_masks, leaf_masks_bytes)?
    } else {
        Vec::new()
    };

    let mut out: Vec<([i32; 3], V)> = Vec::new();
    for slot in 0..(storage.root_table_capacity as usize) {
        let base = slot * 4;
        let internal_idx = root[base + 3] as u32;
        if internal_idx == INVALID_CHILD || internal_idx == INVALID_CHILD - 1 {
            continue;
        }
        if internal_idx >= internal_used {
            continue;
        }
        let key = [root[base + 0], root[base + 1], root[base + 2]];
        let mask_off = (internal_idx * INTERNAL_NODE_MASK_U32S) as usize;
        let child_off = (internal_idx * INTERNAL_NODE_VOXELS) as usize;
        for local in 0..(INTERNAL_NODE_VOXELS as usize) {
            let bit = 1u32 << (local & 31);
            if internal_masks[mask_off + (local >> 5)] & bit == 0 {
                continue;
            }
            let leaf_idx = internal_children[child_off + local];
            if leaf_idx >= leaf_used {
                continue;
            }
            let leaf_mask_off = (leaf_idx * LEAF_NODE_MASK_U32S) as usize;
            let leaf_value_off = (leaf_idx * LEAF_NODE_VOXELS) as usize;
            for ll in 0..(LEAF_NODE_VOXELS as usize) {
                let lbit = 1u32 << (ll & 31);
                if leaf_masks[leaf_mask_off + (ll >> 5)] & lbit == 0 {
                    continue;
                }
                let lx = (ll & 7) as i32;
                let ly = ((ll >> 3) & 7) as i32;
                let lz = ((ll >> 6) & 7) as i32;
                let ix = (local & 31) as i32;
                let iy = ((local >> 5) & 31) as i32;
                let iz = ((local >> 10) & 31) as i32;
                let coord = [
                    (key[0] << 8) | (ix << 3) | lx,
                    (key[1] << 8) | (iy << 3) | ly,
                    (key[2] << 8) | (iz << 3) | lz,
                ];
                let value = leaf_values[leaf_value_off + ll];
                out.push((coord, value));
            }
        }
    }
    out.sort_by_key(|(c, _)| (c[0], c[1], c[2]));
    Ok(out)
}

fn read_buffer<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<T>> {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::sparse_grid_storage stage"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::sparse_grid_storage stage encoder"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &stage, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = stage.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r.map_err(|e| Error::Gpu(format!("readback: {e:?}"))))
            .ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("readback poll: {e:?}")))?;
    receiver
        .recv()
        .map_err(|e| Error::Gpu(format!("readback recv: {e:?}")))??;
    let data = slice.get_mapped_range();
    let v: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    stage.unmap();
    Ok(v)
}

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

#[cfg(test)]
mod cpu_tests {
    use super::*;

    #[test]
    fn f16_roundtrip_within_tolerance() {
        // Random sweep of f32 values; quantise → f16 → f32 and confirm
        // the relative error stays under 2^-10 (the f16 mantissa
        // resolution).
        let cases = [0.0_f32, 1.0, -1.0, 0.5, 1.5, 1e-3, 1e3, 65000.0, -65000.0];
        for &v in &cases {
            let q = ValueF16::from_f32(v).to_f32();
            if v.abs() < 1e-6 {
                assert!(q.abs() < 1e-3, "near-zero round-trip {v} → {q}");
            } else {
                let rel = ((q - v) / v).abs();
                assert!(rel < 2.0_f32.powi(-10), "rel error {rel} for {v} → {q}");
            }
        }
    }

    #[test]
    fn vec3_round_trip_exact() {
        let v = ValueVec3::new(1.5, -2.25, 3.125);
        let xyz = v.xyz();
        assert_eq!(xyz, [1.5, -2.25, 3.125]);
    }
}
