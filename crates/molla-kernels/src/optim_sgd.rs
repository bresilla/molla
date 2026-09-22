//! Stage 38c — GPU dispatcher for the fused SGD parameter-update
//! kernel (`shaders/optim_sgd.wgsl`).
//!
//! Mirrors [`super::optim_adam::AdamGpuKernel`] — one pipeline, one
//! IO bind group built per call, one cached uniform bind group.

use std::sync::Arc;

use molla_core::Result;
use molla_optim::SgdConfig;

use super::optim_adam::read_storage_buffer;

const WG_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SgdUniforms {
    n: u32,
    nesterov: u32,
    step: u32,
    _pad0: u32,
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
}

/// GPU dispatcher for the fused SGD step.
pub struct SgdGpuKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    layout_u: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    bg_u: wgpu::BindGroup,
}

impl SgdGpuKernel {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::optim_sgd"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/optim_sgd.wgsl").into()),
        });

        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::optim_sgd io"),
            entries: &[
                rw_storage(0), // theta
                ro_storage(1), // grad
                rw_storage(2), // buf
            ],
        });
        let layout_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::optim_sgd uniforms"),
            entries: &[uniform_entry(0)],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::optim_sgd layout"),
            bind_group_layouts: &[Some(&layout_io), Some(&layout_u)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::optim_sgd::sgd_step"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sgd_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::optim_sgd uniforms buf"),
            size: std::mem::size_of::<SgdUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::optim_sgd uniforms bg"),
            layout: &layout_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        Self {
            device,
            queue,
            pipeline,
            layout_io,
            layout_u,
            uniform_buf,
            bg_u,
        }
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Dispatch one SGD update. `theta` and `buf` (momentum buffer)
    /// are mutated in place; `grad` is read-only. `step` is the
    /// **0-based** index of the current step (i.e. number of prior
    /// updates) — matches the host-side `SgdState::step` counter
    /// before it gets incremented.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        theta: &wgpu::Buffer,
        grad: &wgpu::Buffer,
        buf: &wgpu::Buffer,
        n: u32,
        cfg: &SgdConfig,
        step: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let uniforms = SgdUniforms {
            n,
            nesterov: cfg.nesterov as u32,
            step,
            _pad0: 0,
            lr: cfg.lr as f32,
            momentum: cfg.momentum as f32,
            dampening: cfg.dampening as f32,
            weight_decay: cfg.weight_decay as f32,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::optim_sgd io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: theta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf.as_entire_binding(),
                },
            ],
        });

        let groups = n.div_ceil(WG_SIZE);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::optim_sgd encoder"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::optim_sgd::sgd_step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &self.bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    /// Test helper — read back an `n`-element f32 storage buffer.
    pub fn read_buffer(&self, buf: &wgpu::Buffer, n: u32) -> Result<Vec<f32>> {
        read_storage_buffer(&self.device, &self.queue, buf, n)
    }
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

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
