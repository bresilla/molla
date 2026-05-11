//! Stage 41 v4 — dispatcher for the fused NS-IMEX RHS assembly kernel
//! (`fem_ns_rhs_assemble.wgsl`).
//!
//! Replaces the host axpy + readback path from Stage 41 v3. The kernel
//! has two entry points sharing the same bind-group layout — one for
//! the per-step IMEX RHS (`(M/dt)·u_old + f_ext − f_conv − bc_delta`)
//! and one for the post-Schur recovery RHS (`rhs_step − Bᵀ p`). Pinned
//! velocity DoFs are overridden inline so the host never touches the
//! RHS buffer in the per-step inner loop.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::core::Result;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RhsUniforms {
    n_v: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    inv_dt: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Which fused RHS to assemble — selects the shader entry point.
#[derive(Copy, Clone, Debug)]
pub enum NsRhsMode {
    /// `rhs[i] = inv_dt * m_u[i] + f_ext[i] − f_conv[i] − bc_delta[i]`
    /// with the pin-override applied to pinned-velocity DoFs.
    Step,
    /// `rhs[i] = rhs_old[i] − bt_p[i]` with the pin-override applied
    /// to pinned-velocity DoFs.
    Recovery,
    /// `rhs[i] = 0` for `pin_mask[i] != 0`, identity otherwise. Replaces
    /// the v3 host `apply_zero_mask_f32` which issued one
    /// `queue.write_buffer` per pinned DoF.
    ZeroMask,
}

/// GPU dispatcher for the fused NS RHS assembly.
pub struct FemNsRhsAssembleKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_step: wgpu::ComputePipeline,
    pipe_recovery: wgpu::ComputePipeline,
    pipe_zero_mask: wgpu::ComputePipeline,

    layout_src: wgpu::BindGroupLayout,
    layout_pin: wgpu::BindGroupLayout,
    layout_dst: wgpu::BindGroupLayout,

    // Dummy buffer reused as `src_*` bindings for the recovery /
    // zero-mask paths (which don't read those slots). Sized once at
    // construction — the bind-group layout requires four `src_*`
    // bindings.
    dummy_src: Arc<wgpu::Buffer>,
}

impl FemNsRhsAssembleKernel {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem::ns_rhs_assemble"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_ns_rhs_assemble.wgsl").into(),
            ),
        });

        let layout_src = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::ns_rhs src"),
            entries: &[ro_storage(0), ro_storage(1), ro_storage(2), ro_storage(3)],
        });
        let layout_pin = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::ns_rhs pin"),
            entries: &[ro_storage(0), ro_storage(1)],
        });
        let layout_dst = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem::ns_rhs dst"),
            entries: &[rw_storage(0), uniform_entry(1)],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::fem::ns_rhs layout"),
            bind_group_layouts: &[&layout_src, &layout_pin, &layout_dst],
            push_constant_ranges: &[],
        });

        let pipe_step = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem::ns_rhs step pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("assemble_step_rhs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_recovery = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem::ns_rhs recovery pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("assemble_recovery_rhs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_zero_mask = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem::ns_rhs zero-mask pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_zero_mask"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let dummy_src = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::ns_rhs dummy_src"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipe_step,
            pipe_recovery,
            pipe_zero_mask,
            layout_src,
            layout_pin,
            layout_dst,
            dummy_src: Arc::new(dummy_src),
        }
    }

    /// Dummy buffer the dispatcher uses as a placeholder for unused
    /// `src_*` bindings (zero-mask path). Exposed so the host can also
    /// supply it for the recovery path's unused `src_c` / `src_d` slots
    /// if convenient — any well-formed storage buffer works there.
    pub fn dummy_src(&self) -> &Arc<wgpu::Buffer> {
        &self.dummy_src
    }

    /// Dispatch the fused RHS assembly.
    ///
    /// `src_a`, `src_b`, `src_c`, `src_d` are the four group(0) buffers
    /// — see the shader source for the per-entry-point semantics. The
    /// recovery entry uses only `src_a` and `src_b`; the host can bind
    /// any well-formed buffer to `src_c`/`src_d` for that path (they
    /// are never read).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        mode: NsRhsMode,
        src_a: &wgpu::Buffer,
        src_b: &wgpu::Buffer,
        src_c: &wgpu::Buffer,
        src_d: &wgpu::Buffer,
        pin_mask: &wgpu::Buffer,
        pin_val: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        n_v: u32,
        inv_dt: f32,
    ) -> Result<()> {
        if n_v == 0 {
            return Ok(());
        }

        let uniforms = RhsUniforms {
            n_v,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            inv_dt,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::ns_rhs uniforms"),
            size: core::mem::size_of::<RhsUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_src = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::ns_rhs src bg"),
            layout: &self.layout_src,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: src_d.as_entire_binding(),
                },
            ],
        });
        let bg_pin = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::ns_rhs pin bg"),
            layout: &self.layout_pin,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pin_mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pin_val.as_entire_binding(),
                },
            ],
        });
        let bg_dst = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem::ns_rhs dst bg"),
            layout: &self.layout_dst,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline = match mode {
            NsRhsMode::Step => &self.pipe_step,
            NsRhsMode::Recovery => &self.pipe_recovery,
            NsRhsMode::ZeroMask => &self.pipe_zero_mask,
        };

        let groups = n_v.div_ceil(WORKGROUP_SIZE).max(1);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem::ns_rhs encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem::ns_rhs pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg_src, &[]);
            pass.set_bind_group(1, &bg_pin, &[]);
            pass.set_bind_group(2, &bg_dst, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Convenience wrapper for the zero-mask path — binds the dispatcher's
    /// dummy buffer to the four unused `src_*` slots so the call site stays
    /// readable. Equivalent to
    /// `dispatch(NsRhsMode::ZeroMask, dummy, dummy, dummy, dummy, pin_mask,
    /// pin_val, rhs, n, 0.0)`.
    pub fn zero_mask(
        &self,
        pin_mask: &wgpu::Buffer,
        pin_val: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        n: u32,
    ) -> Result<()> {
        let dummy = Arc::clone(&self.dummy_src);
        self.dispatch(
            NsRhsMode::ZeroMask,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            pin_mask,
            pin_val,
            rhs,
            n,
            0.0,
        )
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
