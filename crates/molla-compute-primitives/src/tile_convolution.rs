//! Stage 37.8 — separable 2D convolution.
//!
//! Public API:
//! - [`Kernel5`] — 5-tap weight vector applied along x and y in turn.
//! - [`convolve_separable_cpu`] — CPU oracle. Two passes: row-convolve
//!   into a temporary, then column-convolve into the output. Boundary
//!   mode is clamp-to-edge.
//! - [`TileConvolutionKernel`] (gpu only) — pipeline pair (`conv_horiz`,
//!   `conv_vert`); the host wrapper allocates one scratch buffer and
//!   chains the two passes.

use molla_core::{Error, Result};

/// 5-tap separable kernel. Applied along x first, then along y.
/// `coeffs[2]` is the centre tap.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Kernel5 {
    pub coeffs: [f32; 5],
}

impl Kernel5 {
    /// Box filter: equal weights summing to 1.
    pub const fn box5() -> Self {
        Self {
            coeffs: [0.2, 0.2, 0.2, 0.2, 0.2],
        }
    }

    /// Approximate Gaussian kernel `(1, 4, 6, 4, 1) / 16`. Standard
    /// 5-tap binomial filter.
    pub const fn gaussian5() -> Self {
        Self {
            coeffs: [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0],
        }
    }

    /// Identity kernel (δ-function at the centre tap).
    pub const fn identity() -> Self {
        Self {
            coeffs: [0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    /// Discrete second-derivative ("1-D Laplacian") on the 5-tap stencil.
    /// Used by the filtering helpers (Stage 37.8 deliverable D).
    pub const fn laplacian1d() -> Self {
        Self {
            coeffs: [0.0, 1.0, -2.0, 1.0, 0.0],
        }
    }
}

const RADIUS: i32 = 2;

/// CPU separable convolution. `input.len()` and `output.len()` must
/// equal `width * height`. Boundary mode is clamp-to-edge ("replicate").
pub fn convolve_separable_cpu(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    kernel: &Kernel5,
) -> Result<()> {
    let total = width * height;
    if input.len() != total || output.len() != total {
        return Err(Error::Internal(format!(
            "tile_convolution: shape mismatch in/out = {}/{} for {} × {}",
            input.len(),
            output.len(),
            width,
            height
        )));
    }
    if total == 0 {
        return Ok(());
    }

    // Pass 1: row convolve into a temporary.
    let mut tmp = vec![0.0_f32; total];
    let w = width as i32;
    let h = height as i32;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            for t in -RADIUS..=RADIUS {
                let xi = (x + t).clamp(0, w - 1);
                acc += input[(y * w + xi) as usize] * kernel.coeffs[(t + RADIUS) as usize];
            }
            tmp[(y * w + x) as usize] = acc;
        }
    }
    // Pass 2: column convolve from temp into output.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            for t in -RADIUS..=RADIUS {
                let yi = (y + t).clamp(0, h - 1);
                acc += tmp[(yi * w + x) as usize] * kernel.coeffs[(t + RADIUS) as usize];
            }
            output[(y * w + x) as usize] = acc;
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    const WG: u32 = 16;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    struct Uniforms {
        width: u32,
        height: u32,
        _pad0: u32,
        _pad1: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    struct KernelUniform {
        coeffs: [f32; 4],
        coeff4: f32,
        _pad0: f32,
        _pad1: f32,
        _pad2: f32,
    }

    pub struct TileConvolutionKernel {
        pipeline_horiz: wgpu::ComputePipeline,
        pipeline_vert: wgpu::ComputePipeline,
        layout_io: wgpu::BindGroupLayout,
        layout_uniforms: wgpu::BindGroupLayout,
    }

    impl TileConvolutionKernel {
        pub fn new(device: &wgpu::Device) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("molla::tile_convolution"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/tile_convolution.wgsl").into(),
                ),
            });

            let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tile_convolution io"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });
            let layout_uniforms =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("molla::tile_convolution uniforms"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tile_convolution layout"),
                bind_group_layouts: &[Some(&layout_io), Some(&layout_uniforms)],
                immediate_size: 0,
            });

            let mk = |entry: &str, label: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            };
            Self {
                pipeline_horiz: mk("conv_horiz", "molla::tile_convolution::horiz"),
                pipeline_vert: mk("conv_vert", "molla::tile_convolution::vert"),
                layout_io,
                layout_uniforms,
            }
        }

        pub fn dispatch_blocking(
            &self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            input: &[f32],
            output: &mut [f32],
            width: usize,
            height: usize,
            kernel: &Kernel5,
        ) -> Result<()> {
            let total = width * height;
            if input.len() != total || output.len() != total {
                return Err(Error::Gpu(format!(
                    "tile_convolution: shape mismatch in/out = {}/{} for {} × {}",
                    input.len(),
                    output.len(),
                    width,
                    height
                )));
            }
            if total == 0 {
                return Ok(());
            }

            let bytes = (total * 4) as u64;
            let storage_usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            let buf_in = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_convolution in"),
                    size: bytes,
                    usage: storage_usage,
                    mapped_at_creation: false,
                }
            );
            let buf_tmp = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_convolution tmp"),
                    size: bytes,
                    usage: storage_usage,
                    mapped_at_creation: false,
                }
            );
            let buf_out = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_convolution out"),
                    size: bytes,
                    usage: storage_usage,
                    mapped_at_creation: false,
                }
            );
            molla_core::gpu_write_buffer!(queue, &buf_in, 0, bytemuck::cast_slice(input));

            let uniforms = Uniforms {
                width: width as u32,
                height: height as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let kernel_u = KernelUniform {
                coeffs: [
                    kernel.coeffs[0],
                    kernel.coeffs[1],
                    kernel.coeffs[2],
                    kernel.coeffs[3],
                ],
                coeff4: kernel.coeffs[4],
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            };
            let buf_u = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_convolution uniforms"),
                    size: core::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );
            molla_core::gpu_write_buffer!(queue, &buf_u, 0, bytemuck::bytes_of(&uniforms));
            let buf_k = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_convolution kernel"),
                    size: core::mem::size_of::<KernelUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );
            molla_core::gpu_write_buffer!(queue, &buf_k, 0, bytemuck::bytes_of(&kernel_u));

            let bg_horiz = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::tile_convolution horiz io"),
                layout: &self.layout_io,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_tmp.as_entire_binding(),
                    },
                ],
            });
            let bg_vert = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::tile_convolution vert io"),
                layout: &self.layout_io,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_tmp.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_out.as_entire_binding(),
                    },
                ],
            });
            let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::tile_convolution uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_u.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_k.as_entire_binding(),
                    },
                ],
            });

            let groups_x = (width as u32).div_ceil(WG);
            let groups_y = (height as u32).div_ceil(WG);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::tile_convolution encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::tile_convolution horiz pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_horiz);
                pass.set_bind_group(0, &bg_horiz, &[]);
                pass.set_bind_group(1, &bg_u, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::tile_convolution vert pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_vert);
                pass.set_bind_group(0, &bg_vert, &[]);
                pass.set_bind_group(1, &bg_u, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }

            let stage = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::tile_convolution stage"),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }
            );
            encoder.copy_buffer_to_buffer(&buf_out, 0, &stage, 0, bytes);
            queue.submit(std::iter::once(encoder.finish()));

            let slice = stage.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                sender
                    .send(r.map_err(|e| Error::Gpu(format!("conv readback: {e:?}"))))
                    .ok();
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("conv poll: {e:?}")))?;
            receiver
                .recv()
                .map_err(|e| Error::Gpu(format!("conv recv: {e:?}")))??;
            let data = molla_core::gpu_readback!(slice);
            let read: &[f32] = bytemuck::cast_slice(&data);
            output.copy_from_slice(read);
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
pub use gpu::TileConvolutionKernel;
