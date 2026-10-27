//! Stage 37.8 — workgroup-tiled GEMM `C = A · B`.
//!
//! Public API:
//! - [`matmul_cpu`] — row-major reference implementation; used by tests
//!   and as the fallback when the `gpu` feature is off or no [`Engine`]
//!   is available.
//! - [`TileMatmulKernel`] (gpu only) — built once per [`wgpu::Device`],
//!   then [`TileMatmulKernel::dispatch_blocking`] uploads `A` + `B`,
//!   runs the WGSL kernel with 16×16 workgroup tiles, and reads `C`
//!   back into the supplied buffer.
//!
//! Shapes are row-major and the dimensions name the natural matrix
//! sizes:
//! - `A` is `M × K`,
//! - `B` is `K × N`,
//! - `C` is `M × N`.

use molla_core::{Error, Result};

/// CPU reference GEMM. Used by tests + as a non-GPU fallback. Ignores
/// any tile structure — three nested loops, row-major addressing.
///
/// Returns `Err(Error::Internal)` on shape mismatch (so callers don't
/// have to validate ahead of time).
pub fn matmul_cpu(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
    if a.len() != m * k {
        return Err(Error::Internal(format!(
            "tile_matmul.matmul_cpu: a.len() = {} but m*k = {}",
            a.len(),
            m * k
        )));
    }
    if b.len() != k * n {
        return Err(Error::Internal(format!(
            "tile_matmul.matmul_cpu: b.len() = {} but k*n = {}",
            b.len(),
            k * n
        )));
    }
    if c.len() != m * n {
        return Err(Error::Internal(format!(
            "tile_matmul.matmul_cpu: c.len() = {} but m*n = {}",
            c.len(),
            m * n
        )));
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    const TILE: u32 = 16;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    struct Uniforms {
        m: u32,
        n: u32,
        k: u32,
        _pad: u32,
    }

    /// Built once per [`wgpu::Device`]; cheap to dispatch many times.
    pub struct TileMatmulKernel {
        pipeline: wgpu::ComputePipeline,
        layout_io: wgpu::BindGroupLayout,
        layout_uniforms: wgpu::BindGroupLayout,
    }

    impl TileMatmulKernel {
        pub fn new(device: &wgpu::Device) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("molla::tile_matmul"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/tile_matmul.wgsl").into(),
                ),
            });

            let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tile_matmul io"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
            let layout_uniforms =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("molla::tile_matmul uniforms"),
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
                label: Some("molla::tile_matmul layout"),
                bind_group_layouts: &[Some(&layout_io), Some(&layout_uniforms)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tile_matmul pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("tile_matmul"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            Self {
                pipeline,
                layout_io,
                layout_uniforms,
            }
        }

        /// Synchronous GEMM: upload `A` + `B`, dispatch, read back `C`.
        /// `c.len()` must equal `m * n`. Inputs are row-major.
        pub fn dispatch_blocking(
            &self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<()> {
            if a.len() != m * k || b.len() != k * n || c.len() != m * n {
                return Err(Error::Gpu(format!(
                    "tile_matmul: shape mismatch a={} b={} c={} for m={} n={} k={}",
                    a.len(),
                    b.len(),
                    c.len(),
                    m,
                    n,
                    k
                )));
            }
            if m == 0 || n == 0 || k == 0 {
                for v in c.iter_mut() {
                    *v = 0.0;
                }
                return Ok(());
            }

            let bytes_a = (a.len() * 4) as u64;
            let bytes_b = (b.len() * 4) as u64;
            let bytes_c = (c.len() * 4) as u64;

            let storage_usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;

            let buf_a = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_matmul A"),
                    size: bytes_a,
                    usage: storage_usage,
                    mapped_at_creation: false,
                }
            );
            let buf_b = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_matmul B"),
                    size: bytes_b,
                    usage: storage_usage,
                    mapped_at_creation: false,
                }
            );
            let buf_c = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_matmul C"),
                    size: bytes_c,
                    usage: storage_usage,
                    mapped_at_creation: false,
                }
            );
            molla_core::gpu_write_buffer!(queue, &buf_a, 0, bytemuck::cast_slice(a));
            molla_core::gpu_write_buffer!(queue, &buf_b, 0, bytemuck::cast_slice(b));

            let uniforms = Uniforms {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                _pad: 0,
            };
            let buf_u = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_matmul uniforms"),
                    size: core::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );
            molla_core::gpu_write_buffer!(queue, &buf_u, 0, bytemuck::bytes_of(&uniforms));

            let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::tile_matmul io bg"),
                layout: &self.layout_io,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_c.as_entire_binding(),
                    },
                ],
            });
            let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::tile_matmul uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_u.as_entire_binding(),
                }],
            });

            let groups_x = (n as u32).div_ceil(TILE);
            let groups_y = (m as u32).div_ceil(TILE);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::tile_matmul encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::tile_matmul pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg_io, &[]);
                pass.set_bind_group(1, &bg_u, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            let stage = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_matmul stage"),
                    size: bytes_c,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }
            );
            encoder.copy_buffer_to_buffer(&buf_c, 0, &stage, 0, bytes_c);
            queue.submit(std::iter::once(encoder.finish()));

            let slice = stage.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                sender
                    .send(r.map_err(|e| Error::Gpu(format!("tile_matmul readback: {e:?}"))))
                    .ok();
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("tile_matmul poll: {e:?}")))?;
            receiver
                .recv()
                .map_err(|e| Error::Gpu(format!("tile_matmul recv: {e:?}")))??;
            let data = molla_core::gpu_readback!(slice);
            let read: &[f32] = bytemuck::cast_slice(&data);
            c.copy_from_slice(read);
            drop(data);
            stage.unmap();
            Ok(())
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
pub use gpu::TileMatmulKernel;
