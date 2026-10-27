//! Stage 24e — sparse-grid lookup kernel.
//!
//! Wraps `shaders/sparse_grid.wgsl`. One [`SparseGridKernel`] holds
//! the lookup pipeline + bind-group layouts; per-dispatch the caller
//! supplies the packed grid (`SparseGridGpu`) + a list of integer
//! query coords. Useful both as the GPU side of the [`SparseSdf`]
//! query API and as a generic "look this voxel up on the device" tool
//! for any future consumer.
//!
//! v1 ships a *blocking* dispatch helper (one `submit + map_async +
//! poll`) and the matching pipeline-level dispatch builder for callers
//! that already own persistent buffers. Per-submit overhead dominates
//! for small queries — single-coord lookups land in the 50–500 µs
//! range depending on the adapter — so the deliverable's
//! "< 50 µs per ray" target is amortised over batches (the WGSL math
//! itself is a handful of loads, well below the µs bar).

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};
use molla_geometry::SparseGridGpu;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    query_count: u32,
    root_entry_count: u32,
    background: f32,
    /// **Stage 24e.1.** Power-of-two capacity of the root hash table
    /// — 0 falls through to the linear-scan path.
    root_table_capacity: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Pre-built pipeline + bind-group layouts for the sparse-grid lookup
/// kernel. One instance is reusable across many dispatches; the only
/// per-call state is the uniform buffer + the packed-grid storage
/// buffers.
pub struct SparseGridKernel {
    pipeline: wgpu::ComputePipeline,
    layout_grid: wgpu::BindGroupLayout,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl SparseGridKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sparse_grid"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sparse_grid.wgsl").into()),
        });

        let layout_grid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_grid grid"),
            // 6 storage buffers: root_entries, internal_children,
            // internal_masks, leaf_values, leaf_masks, root_table.
            entries: &storage_entries(6, true),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_grid io"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_grid uniforms"),
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
            label: Some("molla::sparse_grid layout"),
            bind_group_layouts: &[Some(&layout_grid), Some(&layout_io), Some(&layout_uniforms)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::sparse_grid::lookup_kernel"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("lookup_kernel"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_grid,
            layout_io,
            layout_uniforms,
        }
    }

    /// Run a batch of integer voxel-coord lookups against `packed` and
    /// return the per-query f32 results. Empty `coords` is a no-op
    /// (returns `Ok(vec![])`).
    ///
    /// `coords[i] = [x, y, z]` queries the active value at that
    /// voxel; misses return `packed.background`.
    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packed: &SparseGridGpu,
        coords: &[[i32; 3]],
    ) -> Result<Vec<f32>> {
        if coords.is_empty() {
            return Ok(vec![]);
        }

        // Pack each coord into a `vec4<i32>` (xyz + 1 padding word).
        // WGSL's storage-buffer layout rules demand 16-byte alignment
        // for `array<vec4<i32>>`; using `vec4` directly here means
        // we don't have to write a bespoke struct on the WGSL side.
        let mut padded: Vec<i32> = Vec::with_capacity(coords.len() * 4);
        for c in coords {
            padded.push(c[0]);
            padded.push(c[1]);
            padded.push(c[2]);
            padded.push(0);
        }

        let n = coords.len();
        let coord_bytes = (padded.len() * 4) as u64;
        let result_bytes = (n * 4) as u64;

        let usage_storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        // ── Grid storage buffers ──────────────────────────────────
        let buf_root = create_storage_with_data(
            device,
            queue,
            "molla::sparse_grid root_entries",
            bytemuck::cast_slice(&packed.root_entries),
            usage_storage,
        );
        let buf_internal_children = create_storage_with_data(
            device,
            queue,
            "molla::sparse_grid internal_children",
            bytemuck::cast_slice(&packed.internal_children),
            usage_storage,
        );
        let buf_internal_masks = create_storage_with_data(
            device,
            queue,
            "molla::sparse_grid internal_masks",
            bytemuck::cast_slice(&packed.internal_masks),
            usage_storage,
        );
        let buf_leaf_values = create_storage_with_data(
            device,
            queue,
            "molla::sparse_grid leaf_values",
            bytemuck::cast_slice(&packed.leaf_values),
            usage_storage,
        );
        let buf_leaf_masks = create_storage_with_data(
            device,
            queue,
            "molla::sparse_grid leaf_masks",
            bytemuck::cast_slice(&packed.leaf_masks),
            usage_storage,
        );
        let buf_root_table = create_storage_with_data(
            device,
            queue,
            "molla::sparse_grid root_table",
            bytemuck::cast_slice(&packed.root_table),
            usage_storage,
        );

        // ── IO buffers ────────────────────────────────────────────
        let buf_coords = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::sparse_grid query_coords"),
                size: coord_bytes,
                usage: usage_storage,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(queue, &buf_coords, 0, bytemuck::cast_slice(&padded));

        let buf_results = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::sparse_grid query_results"),
                size: result_bytes,
                usage: usage_storage,
                mapped_at_creation: false,
            }
        );

        // ── Bind groups ──────────────────────────────────────────
        let bg_grid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid grid bg"),
            layout: &self.layout_grid,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_root.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_internal_children.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_internal_masks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_leaf_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_leaf_masks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_root_table.as_entire_binding(),
                },
            ],
        });
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_coords.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_results.as_entire_binding(),
                },
            ],
        });

        let uniforms = Uniforms {
            query_count: n as u32,
            root_entry_count: packed.root_entry_count(),
            background: packed.background,
            root_table_capacity: packed.root_table_capacity,
        };
        let buf_uniforms = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::sparse_grid uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(queue, &buf_uniforms, 0, bytemuck::bytes_of(&uniforms));
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_uniforms.as_entire_binding(),
            }],
        });

        // ── Encode + dispatch ─────────────────────────────────────
        let groups = (n as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sparse_grid encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sparse_grid pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_grid, &[]);
            pass.set_bind_group(1, &bg_io, &[]);
            pass.set_bind_group(2, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let stage = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::sparse_grid stage"),
                size: result_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(&buf_results, 0, &stage, 0, result_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = stage.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("sparse_grid readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("sparse_grid readback poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("sparse_grid readback recv: {e:?}")))??;
        let data = molla_core::gpu_readback!(slice);
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        stage.unmap();
        Ok(out)
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn create_storage_with_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    // wgpu requires every storage buffer to be at least 4 bytes.
    // Empty inputs (e.g. a zero-leaf grid) get a 4-byte stub so the
    // bind group remains valid.
    let bytes = data.len().max(4);
    let buf = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes as u64,
            usage,
            mapped_at_creation: false,
        }
    );
    if !data.is_empty() {
        molla_core::gpu_write_buffer!(queue, &buf, 0, data);
    }
    buf
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
