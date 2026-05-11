//! Stage 38c — GPU dispatcher for the fused Adam parameter-update
//! kernel (`shaders/optim_adam.wgsl`).
//!
//! Mirrors the bind-group + uniform pattern from
//! [`super::FemCgGpuKernel`]'s `vec_axpy` / `vec_axpby` plumbing:
//! one shader module, one pipeline, two bind-group layouts (IO +
//! uniform). The caller owns the four storage buffers (`theta`,
//! `grad`, `m`, `v`) and we own one uniform buffer that gets refreshed
//! per step via `queue.write_buffer`.

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::optim::AdamConfig;

const WG_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamUniforms {
    n: u32,
    step: u32,
    _pad0: u32,
    _pad1: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

/// GPU dispatcher for the fused Adam step. One instance is reusable
/// across an arbitrary number of `step` calls (the caller's buffers
/// hold all per-parameter state).
pub struct AdamGpuKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    // Layout is only consumed during construction of `bg_u`; keep the
    // owning handle on the struct to keep wgpu's internal refs alive
    // for the lifetime of the bind group.
    #[allow(dead_code)]
    layout_u: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    bg_u: wgpu::BindGroup,
}

impl AdamGpuKernel {
    /// Compile the kernel against `device` + `queue`. Allocates the
    /// uniform buffer + its bind group up-front so per-step dispatch
    /// only writes the 64-byte uniform payload.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::optim_adam"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/optim_adam.wgsl").into(),
            ),
        });

        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::optim_adam io"),
            entries: &[
                rw_storage(0), // theta
                ro_storage(1), // grad
                rw_storage(2), // m
                rw_storage(3), // v
            ],
        });
        let layout_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::optim_adam uniforms"),
            entries: &[uniform_entry(0)],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::optim_adam layout"),
            bind_group_layouts: &[&layout_io, &layout_u],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::optim_adam::adam_step"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("adam_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::optim_adam uniforms buf"),
            size: std::mem::size_of::<AdamUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::optim_adam uniforms bg"),
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

    /// Borrow the device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Borrow the queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Dispatch one Adam update. `theta`, `m`, and `v` are storage
    /// buffers of length `n` (`f32`) that get mutated in place. `grad`
    /// is read-only. `step` is the 1-based current step index (host
    /// counter that's incremented before each call). The kernel uses
    /// it for the bias-correction factors `1 − βⁿ`.
    ///
    /// All four buffers must have at least `n * 4` bytes capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        theta: &wgpu::Buffer,
        grad: &wgpu::Buffer,
        m: &wgpu::Buffer,
        v: &wgpu::Buffer,
        n: u32,
        cfg: &AdamConfig,
        step: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        // Adam's bias correction factors are `1 − βᵗ`; at t = 0 they
        // are exactly 0 and the per-element update divides by them.
        // The kernel clamps to a tiny ε to avoid INF but the result
        // is still meaningless. Reject step = 0 explicitly so callers
        // can't shoot themselves in the foot with an off-by-one in
        // the host-side step counter.
        if step == 0 {
            return Err(Error::Build(
                "AdamGpuKernel::step: step must be >= 1 (1-based step index per \
                 Kingma & Ba; the bias correction is undefined at step 0)"
                    .into(),
            ));
        }
        let uniforms = AdamUniforms {
            n,
            step,
            _pad0: 0,
            _pad1: 0,
            lr: cfg.lr as f32,
            beta1: cfg.beta1 as f32,
            beta2: cfg.beta2 as f32,
            eps: cfg.eps as f32,
            weight_decay: cfg.weight_decay as f32,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::optim_adam io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: theta.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: m.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            ],
        });

        let groups = n.div_ceil(WG_SIZE);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::optim_adam encoder"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::optim_adam::adam_step"),
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

    /// Read an `n`-element f32 storage buffer back into a host vec.
    /// Test helper — molla's main code paths keep buffers resident.
    pub fn read_buffer(&self, buf: &wgpu::Buffer, n: u32) -> Result<Vec<f32>> {
        read_storage_buffer(&self.device, &self.queue, buf, n)
    }
}

/// Read-back helper, factored so both [`AdamGpuKernel`] and
/// [`super::optim_sgd::SgdGpuKernel`] share one implementation.
pub(crate) fn read_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    n: u32,
) -> Result<Vec<f32>> {
    let bytes = (n as u64).max(4) * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::optim readback staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::optim readback encoder"),
    });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("optim readback poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("optim readback recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("optim readback map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let mut out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    out.truncate(n as usize);
    Ok(out)
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
