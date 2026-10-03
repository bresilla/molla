//! GPU radix sort over `(key=u32, value=u32)` pairs.
//!
//! 4 passes of 8-bit radix → 32-bit keys. Each pass uses three
//! dispatches (`count_block_bins`, `scan_global_bins`, `scatter`) and
//! a double-buffer ping-pong: pass 0 reads from caller-supplied
//! `(keys, values)` and writes to a pair of internal scratch buffers;
//! pass 1 reads from scratch and writes back to the caller buffers;
//! and so on. After 4 passes the final sorted result lands back in
//! the caller's buffers.
//!
//! ### Algorithm choice — radix vs bitonic vs merge
//!
//! See the WGSL header in `shaders/radix_sort.wgsl` for the full
//! rationale. TL;DR: 4-pass 8-bit radix is the textbook GPU LBVH
//! choice (Karras 2012, OptiX, Embree, NVIDIA Warp / cuBQL all do
//! this). Bitonic is O(n log² n) which loses badly past n ~ 10k;
//! merge sort hits memory bandwidth before radix does.
//!
//! Stage 19.1 — replaces the CPU-sort stop-gap that Stage 19 shipped.
//!
//! ### Stable sort
//!
//! The scatter kernel processes each block's elements in input-index
//! order from a single thread, so the i-th occurrence of digit `d` in
//! block `b` always lands at offset `block_bins[d, b] + i`. Across
//! blocks, the global scan visits (digit, block) in row-major order
//! so earlier blocks always get lower output offsets. Stability is
//! load-bearing for LBVH: ties on Morton codes need to break
//! deterministically so the WGSL `delta` function in `lbvh.wgsl`
//! sees the same ordering the build pass expects.

use std::sync::Arc;

use molla_core::{Error, Result};

const BLOCK_SIZE: u32 = 256;
const NUM_BINS: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    shift: u32,
    num_blocks: u32,
    _pad: u32,
}

/// Reusable GPU radix-sort dispatcher. Holds the three compute
/// pipelines + two layouts and grows scratch buffers as needed.
pub struct RadixSortKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipeline_count: wgpu::ComputePipeline,
    pipeline_scan: wgpu::ComputePipeline,
    pipeline_scatter: wgpu::ComputePipeline,

    layout_data: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,

    /// Scratch buffers for ping-pong + per-pass uniforms + block_bins.
    scratch_keys: wgpu::Buffer,
    scratch_values: wgpu::Buffer,
    block_bins: wgpu::Buffer,
    /// Four uniform buffers — one per pass — so encode-then-submit
    /// in one go without `write_buffer` between passes.
    uniforms: [wgpu::Buffer; 4],

    /// Capacity (in elements) of `scratch_keys` / `scratch_values`.
    capacity: usize,
    /// Capacity (in u32s) of `block_bins`.
    bins_capacity: usize,
}

/// Fixed-length stable sort with independent retained scratch and bindings.
pub struct RadixSortDispatch {
    pipelines: [wgpu::ComputePipeline; 3],
    data: [wgpu::BindGroup; 2],
    uniforms: [wgpu::BindGroup; 4],
    blocks: u32,
}

impl RadixSortDispatch {
    /// Record twelve ordered passes without allocation, upload or submission.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for digit in 0..4 {
            for stage in 0..3 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipelines[stage]);
                pass.set_bind_group(0, &self.data[digit % 2], &[]);
                pass.set_bind_group(1, &self.uniforms[digit], &[]);
                pass.dispatch_workgroups(if stage == 1 { 1 } else { self.blocks }, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
        }
    }

    pub fn resource_counts(&self) -> [usize; 3] {
        [7, 6, 3]
    }
}

impl RadixSortKernel {
    /// Build pipelines + initial scratch buffers sized for
    /// `initial_capacity` keys.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        initial_capacity: usize,
    ) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::radix_sort"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/radix_sort.wgsl").into()),
        });

        let layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::radix_sort data"),
            entries: &[
                ro_storage(0),
                ro_storage(1),
                rw_storage(2),
                rw_storage(3),
                rw_storage(4),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::radix_sort uniforms"),
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
            label: Some("molla::radix_sort layout"),
            bind_group_layouts: &[Some(&layout_data), Some(&layout_uniforms)],
            immediate_size: 0,
        });

        let pipeline_count = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::radix_sort::count_block_bins"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("count_block_bins"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_scan = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::radix_sort::scan_global_bins"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scan_global_bins"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_scatter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::radix_sort::scatter"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cap = initial_capacity.max(BLOCK_SIZE as usize);
        let scratch_keys = alloc_storage(
            &device,
            "molla::radix_sort scratch_keys",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let scratch_values = alloc_storage(
            &device,
            "molla::radix_sort scratch_values",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let bins_cap = bins_capacity_for(cap);
        let block_bins = alloc_storage(
            &device,
            "molla::radix_sort block_bins",
            (bins_cap * core::mem::size_of::<u32>()) as u64,
        );

        let uniforms = std::array::from_fn(|i| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(match i {
                        0 => "molla::radix_sort uniforms[0]",
                        1 => "molla::radix_sort uniforms[1]",
                        2 => "molla::radix_sort uniforms[2]",
                        _ => "molla::radix_sort uniforms[3]",
                    }),
                    size: core::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        });

        Ok(Self {
            device,
            queue,
            pipeline_count,
            pipeline_scan,
            pipeline_scatter,
            layout_data,
            layout_uniforms,
            scratch_keys,
            scratch_values,
            block_bins,
            uniforms,
            capacity: cap,
            bins_capacity: bins_cap,
        })
    }

    /// Element capacity of the scratch ping-pong buffers.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Prepare an independent fixed-length sort; padding keys remain caller-owned.
    pub fn prepare(
        &self,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        len: usize,
    ) -> Result<RadixSortDispatch> {
        let limits = self.device.limits();
        let bytes = (len as u64).saturating_mul(4);
        let blocks = (len as u64).div_ceil(u64::from(BLOCK_SIZE));
        let bins_bytes = blocks.saturating_mul(256 * 4);
        if len == 0
            || len > u32::MAX as usize
            || keys == values
            || blocks > u64::from(limits.max_compute_workgroups_per_dimension)
            || bins_bytes > limits.max_storage_buffer_binding_size
            || [keys, values].iter().any(|b| {
                b.size() < bytes
                    || b.size() > limits.max_storage_buffer_binding_size
                    || !b.usage().contains(wgpu::BufferUsages::STORAGE)
            })
        {
            return Err(Error::Gpu(
                "invalid retained radix-sort capacity or buffers".into(),
            ));
        }
        let scratch_keys = alloc_storage(&self.device, "retained sort keys", bytes);
        let scratch_values = alloc_storage(&self.device, "retained sort values", bytes);
        let bins = alloc_storage(&self.device, "retained sort bins", bins_bytes);
        let data = std::array::from_fn(|digit| {
            let buffers = if digit == 0 {
                [keys, values, &scratch_keys, &scratch_values, &bins]
            } else {
                [&scratch_keys, &scratch_values, keys, values, &bins]
            };
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("retained radix sort data"),
                layout: &self.layout_data,
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, b)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        let uniforms = std::array::from_fn(|digit| {
            let buffer = molla_core::gpu_create_buffer!(
                self.device,
                &wgpu::BufferDescriptor {
                    label: Some("retained radix sort uniform"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: true,
                }
            );
            molla_core::gpu_mapped_upload!(buffer.slice(..)).copy_from_slice(bytemuck::bytes_of(
                &Uniforms {
                    n: len as u32,
                    shift: digit as u32 * 8,
                    num_blocks: blocks as u32,
                    _pad: 0,
                },
            ));
            buffer.unmap();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("retained radix sort uniform"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        });
        Ok(RadixSortDispatch {
            pipelines: [
                self.pipeline_count.clone(),
                self.pipeline_scan.clone(),
                self.pipeline_scatter.clone(),
            ],
            data,
            uniforms,
            blocks: blocks as u32,
        })
    }

    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity {
            return;
        }
        let cap = (n * 2).max(self.capacity * 2).max(BLOCK_SIZE as usize);
        self.scratch_keys = alloc_storage(
            &self.device,
            "molla::radix_sort scratch_keys",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        self.scratch_values = alloc_storage(
            &self.device,
            "molla::radix_sort scratch_values",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let bins_cap = bins_capacity_for(cap);
        if bins_cap > self.bins_capacity {
            self.block_bins = alloc_storage(
                &self.device,
                "molla::radix_sort block_bins",
                (bins_cap * core::mem::size_of::<u32>()) as u64,
            );
            self.bins_capacity = bins_cap;
        }
        self.capacity = cap;
    }

    /// Sort `len` elements of `(keys, values)` in place, ascending by
    /// key. `keys` and `values` must each be at least
    /// `len * size_of::<u32>()` bytes and have `STORAGE | COPY_SRC |
    /// COPY_DST` usage. Stable.
    pub fn dispatch(
        &mut self,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        len: usize,
    ) -> Result<()> {
        if len < 2 {
            return Ok(());
        }
        self.ensure_capacity(len);

        let n = len as u32;
        let num_blocks = n.div_ceil(BLOCK_SIZE);

        // Stage uniform buffers — one per pass (shifts 0, 8, 16, 24).
        for pass in 0..4 {
            let u = Uniforms {
                n,
                shift: (pass * 8) as u32,
                num_blocks,
                _pad: 0,
            };
            molla_core::gpu_write_buffer!(
                self.queue,
                &self.uniforms[pass],
                0,
                bytemuck::bytes_of(&u)
            );
        }

        // Encode all 4 passes in one command encoder.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::radix_sort encoder"),
            });

        // Pass alternates the (in, out) pair. We always read from
        // `caller`-supplied buffers on pass 0 and end with the result
        // back in caller buffers after 4 passes.
        // pass 0: caller → scratch
        // pass 1: scratch → caller
        // pass 2: caller → scratch
        // pass 3: scratch → caller
        for pass in 0..4 {
            let (in_keys, in_values, out_keys, out_values) = if pass % 2 == 0 {
                (keys, values, &self.scratch_keys, &self.scratch_values)
            } else {
                (&self.scratch_keys, &self.scratch_values, keys, values)
            };
            let bg_data = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::radix_sort data bg"),
                layout: &self.layout_data,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: in_values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.block_bins.as_entire_binding(),
                    },
                ],
            });
            let bg_uniforms = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::radix_sort uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms[pass].as_entire_binding(),
                }],
            });

            // Count.
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::radix_sort::count"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.pipeline_count);
                p.set_bind_group(0, &bg_data, &[]);
                p.set_bind_group(1, &bg_uniforms, &[]);
                p.dispatch_workgroups(num_blocks, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
            // Scan.
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::radix_sort::scan"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.pipeline_scan);
                p.set_bind_group(0, &bg_data, &[]);
                p.set_bind_group(1, &bg_uniforms, &[]);
                p.dispatch_workgroups(1, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
            // Scatter.
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::radix_sort::scatter"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.pipeline_scatter);
                p.set_bind_group(0, &bg_data, &[]);
                p.set_bind_group(1, &bg_uniforms, &[]);
                p.dispatch_workgroups(num_blocks, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

/// Convenience free function — builds a transient kernel and runs one
/// sort. Prefer creating a [`RadixSortKernel`] and reusing it across
/// frames so the pipelines + scratch buffers don't get re-allocated.
pub fn radix_sort_pairs(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    keys: &wgpu::Buffer,
    values: &wgpu::Buffer,
    len: usize,
) -> Result<()> {
    let mut k = RadixSortKernel::new(device, queue, len)?;
    k.dispatch(keys, values, len)
}

fn bins_capacity_for(n_keys: usize) -> usize {
    // ceil(n / BLOCK) blocks × NUM_BINS bins each. Always at least
    // NUM_BINS (1 block).
    let blocks = n_keys.div_ceil(BLOCK_SIZE as usize).max(1);
    blocks * NUM_BINS as usize
}

fn alloc_storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    let size = size.max(16);
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// Quiet unused-import warning when only Error is referenced indirectly.
#[allow(dead_code)]
fn _touch_error(_: Error) {}

// =====================================================================
// 64-bit variant — `RadixSortKernel64`.
// =====================================================================
//
// Same multi-pass 8-bit-radix pattern as `RadixSortKernel`, but with
// 8 passes instead of 4. The 64-bit key is stored as two parallel
// `u32` arrays (`keys_lo`, `keys_hi`) on the GPU.
//
// For segmented sort we use the **prefix-tag** approach: each key is
// host-XOR'd with `(segment_id << TAG_SHIFT)` in the high word, so
// sorting in normal 64-bit order keeps segments contiguous AND sorts
// within each segment. After the sort the tag bits are masked out.
// This means the user's keys cannot use the top `TAG_BITS` bits if
// they want segmented sort — documented on the public API.

const TAG_BITS_64: u32 = 16;
const TAG_SHIFT_64: u32 = 32 - TAG_BITS_64; // shift inside `keys_hi`
const TAG_MASK_HI_64: u32 = ((1u32 << TAG_BITS_64) - 1) << TAG_SHIFT_64;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms64 {
    n: u32,
    shift: u32,
    num_blocks: u32,
    use_high: u32,
}

/// 64-bit-key GPU radix sort dispatcher. Reusable across many sorts;
/// scratch buffers grow as needed. Stable.
pub struct RadixSortKernel64 {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipeline_count: wgpu::ComputePipeline,
    pipeline_scan: wgpu::ComputePipeline,
    pipeline_scatter: wgpu::ComputePipeline,

    layout_data: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,

    scratch_lo: wgpu::Buffer,
    scratch_hi: wgpu::Buffer,
    scratch_values: wgpu::Buffer,
    block_bins: wgpu::Buffer,
    /// 8 uniform buffers — one per pass.
    uniforms: [wgpu::Buffer; 8],

    capacity: usize,
    bins_capacity: usize,
}

impl RadixSortKernel64 {
    /// Build pipelines + initial scratch buffers.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        initial_capacity: usize,
    ) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::radix_sort_64"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/radix_sort_64.wgsl").into()),
        });

        let layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::radix_sort_64 data"),
            entries: &[
                ro_storage(0),
                ro_storage(1),
                ro_storage(2),
                rw_storage(3),
                rw_storage(4),
                rw_storage(5),
                rw_storage(6),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::radix_sort_64 uniforms"),
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
            label: Some("molla::radix_sort_64 layout"),
            bind_group_layouts: &[Some(&layout_data), Some(&layout_uniforms)],
            immediate_size: 0,
        });

        let pipeline_count = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::radix_sort_64::count_block_bins"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("count_block_bins"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_scan = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::radix_sort_64::scan_global_bins"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scan_global_bins"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_scatter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::radix_sort_64::scatter"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cap = initial_capacity.max(BLOCK_SIZE as usize);
        let scratch_lo = alloc_storage(
            &device,
            "molla::radix_sort_64 scratch_lo",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let scratch_hi = alloc_storage(
            &device,
            "molla::radix_sort_64 scratch_hi",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let scratch_values = alloc_storage(
            &device,
            "molla::radix_sort_64 scratch_values",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let bins_cap = bins_capacity_for(cap);
        let block_bins = alloc_storage(
            &device,
            "molla::radix_sort_64 block_bins",
            (bins_cap * core::mem::size_of::<u32>()) as u64,
        );

        let uniforms = std::array::from_fn(|_| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::radix_sort_64 uniforms[i]"),
                    size: core::mem::size_of::<Uniforms64>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        });

        Ok(Self {
            device,
            queue,
            pipeline_count,
            pipeline_scan,
            pipeline_scatter,
            layout_data,
            layout_uniforms,
            scratch_lo,
            scratch_hi,
            scratch_values,
            block_bins,
            uniforms,
            capacity: cap,
            bins_capacity: bins_cap,
        })
    }

    /// Maximum supported segment count (number of distinct tag values).
    /// Segmented sort encodes the segment id in the top 16 bits of the
    /// high word — see the module docs.
    pub const MAX_SEGMENTS: u32 = 1u32 << TAG_BITS_64;

    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity {
            return;
        }
        let cap = (n * 2).max(self.capacity * 2).max(BLOCK_SIZE as usize);
        self.scratch_lo = alloc_storage(
            &self.device,
            "molla::radix_sort_64 scratch_lo",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        self.scratch_hi = alloc_storage(
            &self.device,
            "molla::radix_sort_64 scratch_hi",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        self.scratch_values = alloc_storage(
            &self.device,
            "molla::radix_sort_64 scratch_values",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let bins_cap = bins_capacity_for(cap);
        if bins_cap > self.bins_capacity {
            self.block_bins = alloc_storage(
                &self.device,
                "molla::radix_sort_64 block_bins",
                (bins_cap * core::mem::size_of::<u32>()) as u64,
            );
            self.bins_capacity = bins_cap;
        }
        self.capacity = cap;
    }

    /// Sort `(keys_lo, keys_hi, values)` in place — `n` elements,
    /// ascending by 64-bit key (`(keys_hi << 32) | keys_lo`). Stable.
    ///
    /// `keys_lo`, `keys_hi`, `values` must be `STORAGE | COPY_SRC |
    /// COPY_DST` u32 buffers of at least `n` u32s each.
    pub fn sort_pairs_u64(
        &mut self,
        keys_lo: &wgpu::Buffer,
        keys_hi: &wgpu::Buffer,
        values: &wgpu::Buffer,
        n: u32,
    ) -> Result<()> {
        if n < 2 {
            return Ok(());
        }
        self.ensure_capacity(n as usize);
        let num_blocks = n.div_ceil(BLOCK_SIZE);

        // Stage the 8 per-pass uniform buffers.
        // pass 0..3 → low word, shifts 0/8/16/24
        // pass 4..7 → high word, shifts 0/8/16/24
        for pass in 0..8usize {
            let use_high = if pass >= 4 { 1u32 } else { 0u32 };
            let shift = ((pass as u32) % 4) * 8;
            let uni = Uniforms64 {
                n,
                shift,
                num_blocks,
                use_high,
            };
            molla_core::gpu_write_buffer!(
                self.queue,
                &self.uniforms[pass],
                0,
                bytemuck::bytes_of(&uni)
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::radix_sort_64 encoder"),
            });

        // 8-pass ping-pong: even pass → caller → scratch, odd pass →
        // scratch → caller. Ends back in caller buffers.
        for pass in 0..8usize {
            let (in_lo, in_hi, in_v, out_lo, out_hi, out_v) = if pass % 2 == 0 {
                (
                    keys_lo,
                    keys_hi,
                    values,
                    &self.scratch_lo,
                    &self.scratch_hi,
                    &self.scratch_values,
                )
            } else {
                (
                    &self.scratch_lo,
                    &self.scratch_hi,
                    &self.scratch_values,
                    keys_lo,
                    keys_hi,
                    values,
                )
            };
            let bg_data = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::radix_sort_64 data bg"),
                layout: &self.layout_data,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_lo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: in_hi.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: in_v.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_lo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_hi.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out_v.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.block_bins.as_entire_binding(),
                    },
                ],
            });
            let bg_uniforms = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::radix_sort_64 uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms[pass].as_entire_binding(),
                }],
            });

            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::radix_sort_64::count"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.pipeline_count);
                p.set_bind_group(0, &bg_data, &[]);
                p.set_bind_group(1, &bg_uniforms, &[]);
                p.dispatch_workgroups(num_blocks, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::radix_sort_64::scan"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.pipeline_scan);
                p.set_bind_group(0, &bg_data, &[]);
                p.set_bind_group(1, &bg_uniforms, &[]);
                p.dispatch_workgroups(1, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::radix_sort_64::scatter"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.pipeline_scatter);
                p.set_bind_group(0, &bg_data, &[]);
                p.set_bind_group(1, &bg_uniforms, &[]);
                p.dispatch_workgroups(num_blocks, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Segmented sort: sort each segment `segment_offsets[i] ..
    /// segment_offsets[i+1]` independently, leaving cross-segment
    /// order unchanged.
    ///
    /// Implementation: tag each key's `keys_hi` with the segment id in
    /// the top 16 bits (`segment_id << 16`), run a single 64-bit
    /// global sort (so the tag dominates and segments stay grouped),
    /// then strip the tag.
    ///
    /// **Caveat:** keys must NOT use the top 16 bits of `keys_hi` for
    /// segmented sort. With `n_segments <= 65 536` this is safe; an
    /// assertion is hoisted. The straight `sort_pairs_u64` has no such
    /// restriction.
    pub fn segmented_sort_pairs_u64(
        &mut self,
        keys_lo: &wgpu::Buffer,
        keys_hi: &wgpu::Buffer,
        values: &wgpu::Buffer,
        segment_offsets: &wgpu::Buffer,
        n_segments: u32,
        n_total: u32,
    ) -> Result<()> {
        if n_total < 2 || n_segments == 0 {
            return Ok(());
        }
        if n_segments > Self::MAX_SEGMENTS {
            return Err(Error::Gpu(format!(
                "radix_sort_64: n_segments={n_segments} exceeds MAX_SEGMENTS={}",
                Self::MAX_SEGMENTS
            )));
        }

        // Pull the offsets host-side so we can tag keys per-segment.
        let offsets_host = read_u32(&self.device, &self.queue, segment_offsets, n_segments + 1)?;
        // Validate offset structure BEFORE indexing into keys_hi —
        // otherwise a malformed `segment_offsets` would panic deep
        // inside a `&mut hi_host[start..end]` slice with no
        // actionable error.
        for seg in 0..n_segments as usize {
            if offsets_host[seg + 1] < offsets_host[seg] {
                return Err(Error::Gpu(format!(
                    "segmented_sort_pairs_u64: non-monotonic segment_offsets — \
                     offsets[{seg}]={} > offsets[{}]={}",
                    offsets_host[seg],
                    seg + 1,
                    offsets_host[seg + 1]
                )));
            }
        }
        let total_from_offsets = offsets_host[n_segments as usize];
        if total_from_offsets != n_total {
            return Err(Error::Gpu(format!(
                "segmented_sort_pairs_u64: segment_offsets[{n_segments}] = \
                 {total_from_offsets} but n_total = {n_total}; segment offsets must \
                 cover exactly the input range",
            )));
        }
        // Pull keys_hi host-side, tag with segment id, write back.
        let mut hi_host = read_u32(&self.device, &self.queue, keys_hi, n_total)?;
        for seg in 0..n_segments as usize {
            let start = offsets_host[seg] as usize;
            let end = offsets_host[seg + 1] as usize;
            let tag = ((seg as u32) << TAG_SHIFT_64) & TAG_MASK_HI_64;
            for hi in &mut hi_host[start..end] {
                // Forbid users from passing pre-tagged data — the top
                // 16 bits must be zero.
                if *hi & TAG_MASK_HI_64 != 0 {
                    return Err(Error::Gpu(format!(
                        "radix_sort_64: keys_hi top {TAG_BITS_64} bits must be zero for segmented sort (at offset {seg})"
                    )));
                }
                *hi |= tag;
            }
        }
        molla_core::gpu_write_buffer!(self.queue, keys_hi, 0, bytemuck::cast_slice(&hi_host));

        // Run the regular 64-bit sort.
        self.sort_pairs_u64(keys_lo, keys_hi, values, n_total)?;

        // Strip the tag from keys_hi.
        let mut hi_host = read_u32(&self.device, &self.queue, keys_hi, n_total)?;
        for hi in &mut hi_host {
            *hi &= !TAG_MASK_HI_64;
        }
        molla_core::gpu_write_buffer!(self.queue, keys_hi, 0, bytemuck::cast_slice(&hi_host));

        Ok(())
    }
}

/// Convenience free function — builds a transient kernel + runs one
/// 64-bit sort.
pub fn radix_sort_pairs_u64(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    keys_lo: &wgpu::Buffer,
    keys_hi: &wgpu::Buffer,
    values: &wgpu::Buffer,
    n: u32,
) -> Result<()> {
    let mut k = RadixSortKernel64::new(device, queue, n as usize)?;
    k.sort_pairs_u64(keys_lo, keys_hi, values, n)
}

/// Read `n` u32s from `buf` back to the host. Sync — waits on the GPU.
fn read_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    n: u32,
) -> Result<Vec<u32>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let bytes = (n as u64) * core::mem::size_of::<u32>() as u64;
    let staging = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("molla::radix_sort_64 read_u32 staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::radix_sort_64 read_u32 enc"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("radix_sort_64 poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("radix_sort_64 recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("radix_sort_64 map: {e:?}")))?;
    let mapped = molla_core::gpu_readback!(slice);
    let out = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    Ok(out)
}
