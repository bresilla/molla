//! Stage 37.8 — stream compaction (gather).
//!
//! Public API:
//! - [`compact_cpu`] — reference impl. Returns `Vec<f32>` containing
//!   `input[i]` for every `i` where `mask[i] == true`. Handy outside
//!   tests too — it's the obvious way to filter a sparse-mask buffer
//!   on the host side.
//! - [`TileStreamCompactionKernel`] (gpu only) — single-workgroup
//!   Hillis-Steele scan + scatter. Limited to `n ≤ MAX_N` (1024)
//!   in v1; multi-workgroup decoupled-look-back is a v2.

use molla_core::{Error, Result};

/// Maximum input length supported by the GPU kernel. 1024 keeps the
/// shader inside one workgroup of 256 threads × stride 4.
pub const MAX_N: usize = 1024;

/// CPU reference compaction. Walks `input`/`mask`, keeps every entry
/// whose mask is true, returns the resulting dense vector.
pub fn compact_cpu(input: &[f32], mask: &[bool]) -> Result<Vec<f32>> {
    if input.len() != mask.len() {
        return Err(Error::Internal(format!(
            "tile_stream_compaction.compact_cpu: input.len() = {} but mask.len() = {}",
            input.len(),
            mask.len()
        )));
    }
    let mut out = Vec::with_capacity(input.len());
    for (v, &keep) in input.iter().zip(mask.iter()) {
        if keep {
            out.push(*v);
        }
    }
    Ok(out)
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    struct Uniforms {
        n: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }

    pub struct TileStreamCompactionKernel {
        pipeline: wgpu::ComputePipeline,
        layout_io: wgpu::BindGroupLayout,
        layout_uniforms: wgpu::BindGroupLayout,
    }

    impl TileStreamCompactionKernel {
        pub fn new(device: &wgpu::Device) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("molla::tile_stream_compaction"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/tile_stream_compaction.wgsl").into(),
                ),
            });

            let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tile_stream_compaction io"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    storage_entry(3, false),
                ],
            });
            let layout_uniforms =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("molla::tile_stream_compaction uniforms"),
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
                label: Some("molla::tile_stream_compaction layout"),
                bind_group_layouts: &[Some(&layout_io), Some(&layout_uniforms)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tile_stream_compaction pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("tile_stream_compaction"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            Self {
                pipeline,
                layout_io,
                layout_uniforms,
            }
        }

        /// Synchronous gather. Returns the dense output vector. The
        /// mask is taken as `&[u32]` (1 = keep, 0 = drop) to match the
        /// WGSL kernel's storage type — pass a converted slice if you
        /// have `&[bool]`.
        pub fn dispatch_blocking(
            &self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            input: &[f32],
            mask: &[u32],
        ) -> Result<Vec<f32>> {
            if input.len() != mask.len() {
                return Err(Error::Gpu(format!(
                    "tile_stream_compaction: input/mask length mismatch {}/{}",
                    input.len(),
                    mask.len()
                )));
            }
            let n = input.len();
            if n == 0 {
                return Ok(Vec::new());
            }
            if n > MAX_N {
                return Err(Error::Gpu(format!(
                    "tile_stream_compaction: n = {n} exceeds single-workgroup limit {MAX_N}"
                )));
            }

            let bytes_in = (n * 4) as u64;
            let bytes_mask = (n * 4) as u64;
            let bytes_out = (n * 4) as u64;
            let bytes_count = 4u64;

            let storage_usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            let buf_in = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_stream_compaction in"),
                size: bytes_in,
                usage: storage_usage,
                mapped_at_creation: false,
            });
            let buf_mask = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_stream_compaction mask"),
                size: bytes_mask,
                usage: storage_usage,
                mapped_at_creation: false,
            });
            let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_stream_compaction out"),
                size: bytes_out,
                usage: storage_usage,
                mapped_at_creation: false,
            });
            let buf_count = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_stream_compaction count"),
                size: bytes_count,
                usage: storage_usage,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf_in, 0, bytemuck::cast_slice(input));
            queue.write_buffer(&buf_mask, 0, bytemuck::cast_slice(mask));

            let uniforms = Uniforms {
                n: n as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let buf_u = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_stream_compaction uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf_u, 0, bytemuck::bytes_of(&uniforms));

            let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::tile_stream_compaction io bg"),
                layout: &self.layout_io,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_mask.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf_count.as_entire_binding(),
                    },
                ],
            });
            let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::tile_stream_compaction uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_u.as_entire_binding(),
                }],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::tile_stream_compaction encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::tile_stream_compaction pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg_io, &[]);
                pass.set_bind_group(1, &bg_u, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            let stage_out = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_stream_compaction stage out"),
                size: bytes_out,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let stage_count = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_stream_compaction stage count"),
                size: bytes_count,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&buf_out, 0, &stage_out, 0, bytes_out);
            encoder.copy_buffer_to_buffer(&buf_count, 0, &stage_count, 0, bytes_count);
            queue.submit(std::iter::once(encoder.finish()));

            // Read back count first.
            let read = |buf: &wgpu::Buffer, n_bytes: u64| -> Result<Vec<u8>> {
                let slice = buf.slice(..n_bytes);
                let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    sender
                        .send(r.map_err(|e| Error::Gpu(format!("compaction readback: {e:?}"))))
                        .ok();
                });
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| Error::Gpu(format!("compaction poll: {e:?}")))?;
                receiver
                    .recv()
                    .map_err(|e| Error::Gpu(format!("compaction recv: {e:?}")))??;
                let data = slice.get_mapped_range();
                let v = data.to_vec();
                drop(data);
                buf.unmap();
                Ok(v)
            };
            let count_bytes = read(&stage_count, bytes_count)?;
            let count = u32::from_le_bytes(count_bytes[..4].try_into().unwrap()) as usize;
            let out_bytes = read(&stage_out, bytes_out)?;
            let out_floats: &[f32] = bytemuck::cast_slice(&out_bytes);
            Ok(out_floats[..count].to_vec())
        }
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu::TileStreamCompactionKernel;

/// Convenience: convert a `&[bool]` mask to the `Vec<u32>` form the GPU
/// kernel expects (1 for true, 0 for false).
pub fn mask_to_u32(mask: &[bool]) -> Vec<u32> {
    mask.iter().map(|&b| if b { 1 } else { 0 }).collect()
}
