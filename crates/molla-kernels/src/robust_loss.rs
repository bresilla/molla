//! Stage 24b — batched robust-loss evaluation kernel.
//!
//! Wraps `shaders/robust_loss.wgsl`. One [`RobustLossKernel`] holds
//! the pipelines for all six entry points (L1 / smooth-L2 / Huber /
//! pseudo-Huber / Tukey / Cauchy); pick which one to dispatch via
//! [`RobustLoss`]. Output layout: `[ρ_0, ρ'_0, ρ_1, ρ'_1, …]` —
//! convenient for parity tests against the CPU reference, since the
//! WGSL math runs in `f32` on the GPU and we want to verify both the
//! loss value and its gradient land within the `1e-5` budget.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};
use molla_math::robust_loss::RobustLoss;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    sample_count: u32,
    param0: f32,
    param1: f32,
    _pad: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Pair `(ρ(r_i), ρ'(r_i))` returned per input sample.
#[derive(Copy, Clone, Debug, Default)]
pub struct LossPair {
    pub value: f32,
    pub grad: f32,
}

pub struct RobustLossKernel {
    pipeline_l1: wgpu::ComputePipeline,
    pipeline_smooth_l2: wgpu::ComputePipeline,
    pipeline_huber: wgpu::ComputePipeline,
    pipeline_pseudo_huber: wgpu::ComputePipeline,
    pipeline_tukey: wgpu::ComputePipeline,
    pipeline_cauchy: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl RobustLossKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::robust_loss"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/robust_loss.wgsl").into()),
        });

        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::robust_loss io"),
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
            label: Some("molla::robust_loss uniforms"),
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
            label: Some("molla::robust_loss layout"),
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
            pipeline_l1: mk("l1_kernel", "molla::robust_loss::l1"),
            pipeline_smooth_l2: mk("smooth_l2_kernel", "molla::robust_loss::smooth_l2"),
            pipeline_huber: mk("huber_kernel", "molla::robust_loss::huber"),
            pipeline_pseudo_huber: mk("pseudo_huber_kernel", "molla::robust_loss::pseudo_huber"),
            pipeline_tukey: mk("tukey_kernel", "molla::robust_loss::tukey"),
            pipeline_cauchy: mk("cauchy_kernel", "molla::robust_loss::cauchy"),
            layout_io,
            layout_uniforms,
        }
    }

    /// Run the chosen loss over `residuals` and read the pair vector
    /// back to host. L2 isn't on the GPU — it's a one-line CPU op
    /// nobody needs to amortise — and asking for L2 here returns
    /// `Err(Error::Build)` so callers don't accidentally pay for an
    /// unnecessary dispatch.
    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        loss: RobustLoss,
        residuals: &[f32],
    ) -> Result<Vec<LossPair>> {
        if residuals.is_empty() {
            return Ok(vec![]);
        }
        let n = residuals.len();
        let in_bytes = (n * 4) as u64;
        let out_bytes = (n * 2 * 4) as u64;

        let usage_storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::robust_loss input"),
            size: in_bytes,
            usage: usage_storage,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::robust_loss output"),
            size: out_bytes,
            usage: usage_storage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&in_buf, 0, bytemuck::cast_slice(residuals));

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::robust_loss io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let (param0, pipeline) = match loss {
            RobustLoss::L2 => {
                return Err(Error::Build(
                    "RobustLossKernel: L2 has no GPU path — use the CPU formula".into(),
                ));
            }
            RobustLoss::L1 => (0.0_f32, &self.pipeline_l1),
            RobustLoss::SmoothL2 { eps } => (eps as f32, &self.pipeline_smooth_l2),
            RobustLoss::Huber { delta } => (delta as f32, &self.pipeline_huber),
            RobustLoss::PseudoHuber { delta } => (delta as f32, &self.pipeline_pseudo_huber),
            RobustLoss::Tukey { c } => (c as f32, &self.pipeline_tukey),
            RobustLoss::Cauchy { c } => (c as f32, &self.pipeline_cauchy),
        };
        let uniforms = Uniforms {
            sample_count: n as u32,
            param0,
            param1: 0.0,
            _pad: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::robust_loss uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::robust_loss uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = (n as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::robust_loss encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::robust_loss pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::robust_loss stage"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = stage.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("robust_loss readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("robust_loss readback poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("robust_loss readback recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let raw: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        stage.unmap();

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(LossPair {
                value: raw[i * 2],
                grad: raw[i * 2 + 1],
            });
        }
        Ok(out)
    }
}
