//! Stage 36.5 Slice H — Rust dispatchers for the GPU-resident
//! FEM step kernels (predictor / RHS / corrector).
//!
//! Two pipelines per scheme:
//!   * [`WaveStepOpsKernel`] — `wave_predictor`, `wave_rhs`,
//!     `wave_corrector`, `wave_copy_u` (driving `WaveStepperGpu`).
//!   * [`TransientStepOpsKernel`] — `transient_rhs`, `transient_copy_u`
//!     (driving `TransientStepperGpu`).
//!
//! Each dispatcher owns its compute pipelines + bind-group layouts.
//! Bind groups are constructed per dispatch — they're cheap relative
//! to the kernel work and the per-step buffer set is small (≤ 5 IO
//! buffers + 1 uniform). The pipelines are shared across many steps
//! and many `WaveStepperGpu` instances.
//!
//! Style mirrors [`super::fem_cg::FemCgGpuKernel`] — same
//! `ro_storage` / `rw_storage` / `uniform_entry` helpers, same
//! workgroup size (256), same precision (`f32` GPU side).

use std::sync::Arc;

use crate::core::Result;

const WG_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveStepUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    dt: f32,
    half_minus_beta_dt2: f32,
    bdt2_inv: f32,
    gamma: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TransientStepUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    inv_dt: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

// ─── Wave step ops ───────────────────────────────────────────────────

/// GPU dispatcher for the Newmark-β predictor / RHS / corrector +
/// `u` copy kernels. Compiles `shaders/fem_wave_ops.wgsl` once and
/// holds four pipelines + their bind-group layouts.
pub struct WaveStepOpsKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_predictor: wgpu::ComputePipeline,
    pipe_rhs: wgpu::ComputePipeline,
    pipe_corrector: wgpu::ComputePipeline,
    pipe_copy_u: wgpu::ComputePipeline,

    layout_pred_io: wgpu::BindGroupLayout,
    layout_rhs_io: wgpu::BindGroupLayout,
    layout_corr_io: wgpu::BindGroupLayout,
    layout_copy_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl WaveStepOpsKernel {
    /// Build the kernel against `device` + `queue`. Compiles the
    /// shader module and creates four pipelines.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem_step_ops::wave"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_wave_ops.wgsl").into(),
            ),
        });

        let layout_pred_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_step_ops::wave_pred io"),
            entries: &[
                ro_storage(0), // u
                ro_storage(1), // v
                ro_storage(2), // a
                rw_storage(3), // u_pred (out)
            ],
        });
        let layout_rhs_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_step_ops::wave_rhs io"),
            entries: &[
                ro_storage(0), // m_upred
                ro_storage(1), // bc_shift
                ro_storage(2), // is_pinned (u32)
                ro_storage(3), // pinned_value
                rw_storage(4), // b (out)
            ],
        });
        let layout_corr_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_step_ops::wave_corr io"),
            entries: &[
                ro_storage(0), // u_new
                ro_storage(1), // u_pred
                rw_storage(2), // v (in: v_old, out: v_new)
                rw_storage(3), // a (in: a_old, out: a_new)
                ro_storage(4), // is_pinned
            ],
        });
        let layout_copy_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_step_ops::wave_copy io"),
            entries: &[
                ro_storage(0), // src
                rw_storage(1), // dst
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_step_ops::wave uniforms"),
            entries: &[uniform_entry(0)],
        });

        let pipe_predictor = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem_step_ops::wave_predictor"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::fem_step_ops::wave_predictor layout"),
                    bind_group_layouts: &[&layout_pred_io, &layout_uniforms],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("wave_predictor"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_rhs = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem_step_ops::wave_rhs"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::fem_step_ops::wave_rhs layout"),
                    bind_group_layouts: &[&layout_rhs_io, &layout_uniforms],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("wave_rhs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_corrector = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem_step_ops::wave_corrector"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::fem_step_ops::wave_corrector layout"),
                    bind_group_layouts: &[&layout_corr_io, &layout_uniforms],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("wave_corrector"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_copy_u = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem_step_ops::wave_copy_u"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::fem_step_ops::wave_copy_u layout"),
                    bind_group_layouts: &[&layout_copy_io, &layout_uniforms],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("wave_copy_u"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipe_predictor,
            pipe_rhs,
            pipe_corrector,
            pipe_copy_u,
            layout_pred_io,
            layout_rhs_io,
            layout_corr_io,
            layout_copy_io,
            layout_uniforms,
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

    fn make_uniform_buf(&self, label: &str, uniforms: &WaveStepUniforms) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: core::mem::size_of::<WaveStepUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buf, 0, bytemuck::bytes_of(uniforms));
        buf
    }

    fn n_groups(n: u32) -> u32 {
        n.div_ceil(WG_SIZE).max(1)
    }

    /// Run `u_pred = u + dt·v + (½−β)·dt²·a` element-wise.
    pub fn predictor(
        &self,
        n: u32,
        dt: f32,
        half_minus_beta_dt2: f32,
        u: &wgpu::Buffer,
        v: &wgpu::Buffer,
        a: &wgpu::Buffer,
        u_pred: &wgpu::Buffer,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let uniforms = WaveStepUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            dt,
            half_minus_beta_dt2,
            bdt2_inv: 0.0,
            gamma: 0.0,
        };
        let u_buf = self.make_uniform_buf("molla::fem_step_ops::wave_pred uniforms", &uniforms);
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::wave_pred io bg"),
            layout: &self.layout_pred_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: u_pred.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::wave_pred u bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: u_buf.as_entire_binding(),
            }],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_step_ops::wave_pred enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_step_ops::wave_pred"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_predictor);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(Self::n_groups(n), 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    /// Run the RHS construction:
    /// `b[i] = bdt2_inv · m_upred[i] + bc_shift[i]` for unpinned i,
    /// `b[i] = pinned_value[i]` for pinned i.
    #[allow(clippy::too_many_arguments)]
    pub fn rhs(
        &self,
        n: u32,
        bdt2_inv: f32,
        m_upred: &wgpu::Buffer,
        bc_shift: &wgpu::Buffer,
        is_pinned: &wgpu::Buffer,
        pinned_value: &wgpu::Buffer,
        b: &wgpu::Buffer,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let uniforms = WaveStepUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            dt: 0.0,
            half_minus_beta_dt2: 0.0,
            bdt2_inv,
            gamma: 0.0,
        };
        let u_buf = self.make_uniform_buf("molla::fem_step_ops::wave_rhs uniforms", &uniforms);
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::wave_rhs io bg"),
            layout: &self.layout_rhs_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: m_upred.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bc_shift.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: is_pinned.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pinned_value.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::wave_rhs u bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: u_buf.as_entire_binding(),
            }],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_step_ops::wave_rhs enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_step_ops::wave_rhs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_rhs);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(Self::n_groups(n), 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    /// Run the Newmark corrector:
    /// `a_new = (u_new − u_pred) · bdt2_inv`,
    /// `v_new = v_old + dt · ((1−γ)·a_old + γ·a_new)`,
    /// then `v, a := 0` on every pinned DoF.
    /// `v` and `a` are read in-place as the old values and overwritten.
    #[allow(clippy::too_many_arguments)]
    pub fn corrector(
        &self,
        n: u32,
        dt: f32,
        bdt2_inv: f32,
        gamma: f32,
        u_new: &wgpu::Buffer,
        u_pred: &wgpu::Buffer,
        v: &wgpu::Buffer,
        a: &wgpu::Buffer,
        is_pinned: &wgpu::Buffer,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let uniforms = WaveStepUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            dt,
            half_minus_beta_dt2: 0.0,
            bdt2_inv,
            gamma,
        };
        let u_buf = self.make_uniform_buf("molla::fem_step_ops::wave_corr uniforms", &uniforms);
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::wave_corr io bg"),
            layout: &self.layout_corr_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_new.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_pred.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: is_pinned.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::wave_corr u bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: u_buf.as_entire_binding(),
            }],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_step_ops::wave_corr enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_step_ops::wave_corr"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_corrector);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(Self::n_groups(n), 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    /// Copy `src[..n] -> dst[..n]` element-wise. Used to pull the
    /// final CG iterate (which lives in a fresh buffer per solve)
    /// into the persistent `u` storage.
    pub fn copy_u(&self, n: u32, src: &wgpu::Buffer, dst: &wgpu::Buffer) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let uniforms = WaveStepUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            dt: 0.0,
            half_minus_beta_dt2: 0.0,
            bdt2_inv: 0.0,
            gamma: 0.0,
        };
        let u_buf = self.make_uniform_buf("molla::fem_step_ops::wave_copy uniforms", &uniforms);
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::wave_copy io bg"),
            layout: &self.layout_copy_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::wave_copy u bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: u_buf.as_entire_binding(),
            }],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_step_ops::wave_copy enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_step_ops::wave_copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_copy_u);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(Self::n_groups(n), 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }
}

// ─── Transient step ops ──────────────────────────────────────────────

/// GPU dispatcher for the backward-Euler transient RHS construction +
/// `u` copy. Compiles `shaders/fem_transient_ops.wgsl` once.
pub struct TransientStepOpsKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_rhs: wgpu::ComputePipeline,
    pipe_copy_u: wgpu::ComputePipeline,

    layout_rhs_io: wgpu::BindGroupLayout,
    layout_copy_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl TransientStepOpsKernel {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem_step_ops::transient"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_transient_ops.wgsl").into(),
            ),
        });

        let layout_rhs_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_step_ops::transient_rhs io"),
            entries: &[
                ro_storage(0), // m_uold
                ro_storage(1), // f
                ro_storage(2), // bc_shift
                ro_storage(3), // is_pinned (u32)
                ro_storage(4), // pinned_value
                rw_storage(5), // b (out)
            ],
        });
        let layout_copy_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_step_ops::transient_copy io"),
            entries: &[
                ro_storage(0), // src
                rw_storage(1), // dst
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_step_ops::transient uniforms"),
            entries: &[uniform_entry(0)],
        });

        let pipe_rhs = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem_step_ops::transient_rhs"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::fem_step_ops::transient_rhs layout"),
                    bind_group_layouts: &[&layout_rhs_io, &layout_uniforms],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("transient_rhs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_copy_u = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem_step_ops::transient_copy_u"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::fem_step_ops::transient_copy_u layout"),
                    bind_group_layouts: &[&layout_copy_io, &layout_uniforms],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("transient_copy_u"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipe_rhs,
            pipe_copy_u,
            layout_rhs_io,
            layout_copy_io,
            layout_uniforms,
        }
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    fn make_uniform_buf(&self, label: &str, uniforms: &TransientStepUniforms) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: core::mem::size_of::<TransientStepUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buf, 0, bytemuck::bytes_of(uniforms));
        buf
    }

    fn n_groups(n: u32) -> u32 {
        n.div_ceil(WG_SIZE).max(1)
    }

    /// Run the transient RHS construction.
    #[allow(clippy::too_many_arguments)]
    pub fn rhs(
        &self,
        n: u32,
        inv_dt: f32,
        m_uold: &wgpu::Buffer,
        f: &wgpu::Buffer,
        bc_shift: &wgpu::Buffer,
        is_pinned: &wgpu::Buffer,
        pinned_value: &wgpu::Buffer,
        b: &wgpu::Buffer,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let uniforms = TransientStepUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            inv_dt,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        let u_buf = self.make_uniform_buf("molla::fem_step_ops::transient_rhs uniforms", &uniforms);
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::transient_rhs io bg"),
            layout: &self.layout_rhs_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: m_uold.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: f.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bc_shift.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: is_pinned.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pinned_value.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::transient_rhs u bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: u_buf.as_entire_binding(),
            }],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_step_ops::transient_rhs enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_step_ops::transient_rhs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_rhs);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(Self::n_groups(n), 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    /// Copy `src[..n] -> dst[..n]` element-wise. Same role as
    /// [`WaveStepOpsKernel::copy_u`].
    pub fn copy_u(&self, n: u32, src: &wgpu::Buffer, dst: &wgpu::Buffer) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let uniforms = TransientStepUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            inv_dt: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        let u_buf =
            self.make_uniform_buf("molla::fem_step_ops::transient_copy uniforms", &uniforms);
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::transient_copy io bg"),
            layout: &self.layout_copy_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_step_ops::transient_copy u bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: u_buf.as_entire_binding(),
            }],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_step_ops::transient_copy enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_step_ops::transient_copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_copy_u);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(Self::n_groups(n), 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
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
