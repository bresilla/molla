//! GPU batched Dieterich-Ruina rate-and-state friction advance. Mirrors
//! the CPU [`advance_rate_state_friction`](crate::signorini::advance_rate_state_friction)
//! over many contacts at once (one thread per contact).

use crate::signorini::{RateStateFrictionParams, RateStateFrictionState};
use molla_core::{Error, Result};
use molla_math::Real;
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    p0: [f32; 4],
    p1: [f32; 4],
}

const WORKGROUP_SIZE: u32 = 64;

/// Advance a batch of rate-and-state contacts one Euler step on the GPU.
/// Returns `(updated states, coefficients)`, matching the CPU
/// [`advance_rate_state_friction`](crate::signorini::advance_rate_state_friction)
/// within f32 tolerance.
pub fn advance_rate_state_friction_batch_gpu(
    engine: &molla_compute::Engine,
    params: RateStateFrictionParams,
    states: &[RateStateFrictionState],
    slip_rates: &[Real],
    dt: Real,
) -> Result<(Vec<RateStateFrictionState>, Vec<Real>)> {
    let n = states.len();
    if slip_rates.len() != n {
        return Err(Error::Fem(format!(
            "rate_state_gpu: states.len() ({n}) != slip_rates.len() ({})",
            slip_rates.len()
        )));
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("signorini_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("signorini_gpu: engine has no queue".into()))?
        .clone();

    let mut slip = Buffer::from_vec(slip_rates.to_vec(), BufferUsage::Storage);
    slip.ensure_device(&device)?;
    slip.upload(&queue)?;
    let theta_host: Vec<Real> = states.iter().map(|s| s.theta).collect();
    let mut theta_in = Buffer::from_vec(theta_host, BufferUsage::Storage);
    theta_in.ensure_device(&device)?;
    theta_in.upload(&queue)?;

    let mut theta_out = Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);
    theta_out.ensure_device(&device)?;
    theta_out.upload(&queue)?;
    let mut mu_out = Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);
    mu_out.ensure_device(&device)?;
    mu_out.upload(&queue)?;

    let uniforms = Uniforms {
        p0: [
            params.mu0 as f32,
            params.a as f32,
            params.b as f32,
            params.v0 as f32,
        ],
        p1: [params.dc as f32, dt as f32, n as f32, 0.0],
    };

    let kernel = RateStateKernel::new(&device);
    let bg_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::rsf in bg"),
        layout: &kernel.layout_in,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: slip
                    .device_buffer()
                    .ok_or_else(dev_err)?
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: theta_in
                    .device_buffer()
                    .ok_or_else(dev_err)?
                    .as_entire_binding(),
            },
        ],
    });
    let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::rsf out bg"),
        layout: &kernel.layout_out,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: theta_out
                    .device_buffer()
                    .ok_or_else(dev_err)?
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mu_out
                    .device_buffer()
                    .ok_or_else(dev_err)?
                    .as_entire_binding(),
            },
        ],
    });
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::rsf uniforms"),
        size: core::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    let bg_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::rsf uniform bg"),
        layout: &kernel.layout_uniform,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let groups = (n as u32).div_ceil(WORKGROUP_SIZE).max(1);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::rsf encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::rsf pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg_in, &[]);
        pass.set_bind_group(1, &bg_out, &[]);
        pass.set_bind_group(2, &bg_uniform, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    theta_out.mark_device_dirty();
    mu_out.mark_device_dirty();
    theta_out.download(&device, &queue)?;
    mu_out.download(&device, &queue)?;

    let next = theta_out
        .host()?
        .iter()
        .map(|&theta| RateStateFrictionState { theta })
        .collect();
    let mu = mu_out.host()?.to_vec();
    Ok((next, mu))
}

fn dev_err() -> Error {
    Error::Gpu("signorini_gpu: no device mirror".into())
}

struct RateStateKernel {
    pipeline: wgpu::ComputePipeline,
    layout_in: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniform: wgpu::BindGroupLayout,
}

impl RateStateKernel {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::rate_state_friction"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/rate_state_friction.wgsl").into(),
            ),
        });
        let layout_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::rsf in"),
            entries: &storage_entries(2, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::rsf out"),
            entries: &storage_entries(2, false),
        });
        let layout_uniform = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::rsf uniform"),
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
            label: Some("molla::rsf layout"),
            bind_group_layouts: &[&layout_in, &layout_out, &layout_uniform],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::rsf pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("advance"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_in,
            layout_out,
            layout_uniform,
        }
    }
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
