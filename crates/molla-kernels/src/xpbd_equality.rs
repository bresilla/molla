//! Rust dispatch wrapper for `shaders/xpbd_equality.wgsl`.
//!
//! Single-threaded GPU port of the equality-constraint projection
//! pass. Replaces the host-side `apply_equality_post_gpu` shim — no
//! more body_q/body_qd download/upload around each XPBD substep.
//!
//! Velocity recovery is intentionally **not** done here; the existing
//! `recover_velocity` pass in `XpbdKernel` already runs after this
//! kernel and reads the post-equality `body_q_out`, so the corrected
//! pose flows through naturally.

use molla_core::{Error, Result};
use molla_sim::{Model, State};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    body_count: u32,
    equality_count: u32,
    iterations: u32,
    dt: f32,
}

const RECOVER_WORKGROUP_SIZE: u32 = 64;

pub struct XpbdEqualityKernel {
    pipeline_project: wgpu::ComputePipeline,
    pipeline_recover: wgpu::ComputePipeline,
    /// Joint-mimic projection (Stage 32). Dispatched after the body
    /// equality pass so chained mimics see the post-equality joint
    /// state. Uses an extended pipeline layout that adds bind group
    /// 3 (joint_q / joint_qd / coord & dof offsets / mimic side-table).
    pipeline_project_joint_mimic: wgpu::ComputePipeline,
    layout_body: wgpu::BindGroupLayout,
    layout_eq: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    /// Bind group 3 layout for the joint-mimic pipeline.
    layout_joint_mimic: wgpu::BindGroupLayout,
    /// Scratch buffer holding `body_q` snapshotted immediately BEFORE
    /// the equality projection runs. Sized for `body_q_capacity`
    /// transforms (32 bytes each); regrown lazily when a larger model
    /// shows up. Used by `recover_velocity_after_equality` to compute
    /// the equality-only displacement so the recovery pass ADDS to the
    /// main kernel's contact-friction velocity instead of overwriting
    /// it. See `shaders/xpbd_equality.wgsl::recover_velocity_after_equality`
    /// for the full bug-fix rationale.
    body_q_pre_eq: Option<wgpu::Buffer>,
    body_q_capacity: u32,
}

impl XpbdEqualityKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::xpbd_equality"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/xpbd_equality.wgsl").into()),
        });

        // Body group: body_q (RW), inv_mass, inv_inertia, com, flags,
        // body_q_in (read-only, pre-step pose), body_qd (RW, written
        // by the recovery pass).
        let layout_body = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_equality body"),
            entries: &[
                rw_storage_entry(0),
                ro_storage_entry(1),
                ro_storage_entry(2),
                ro_storage_entry(3),
                ro_storage_entry(4),
                ro_storage_entry(5),
                rw_storage_entry(6),
            ],
        });
        // Equality group: 7 read-only arrays.
        let layout_eq = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_equality eq"),
            entries: &(0..7).map(ro_storage_entry).collect::<Vec<_>>(),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_equality uniforms"),
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
            label: Some("molla::xpbd_equality layout"),
            bind_group_layouts: &[Some(&layout_body), Some(&layout_eq), Some(&layout_uniforms)],
            immediate_size: 0,
        });

        let pipeline_project = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::xpbd_equality::project_equality"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("project_equality"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_recover = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::xpbd_equality::recover_velocity_after_equality"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("recover_velocity_after_equality"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Joint-mimic group (Stage 32): joint_q (RW), joint_qd (RW),
        // joint_coord_offset (R), joint_dof_offset (R), mimic
        // driver / driven joint indices (R), and the 5-coeff
        // polynomial buffer (R). 7 entries total.
        let layout_joint_mimic =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::xpbd_equality joint_mimic"),
                entries: &[
                    rw_storage_entry(0),
                    rw_storage_entry(1),
                    ro_storage_entry(2),
                    ro_storage_entry(3),
                    ro_storage_entry(4),
                    ro_storage_entry(5),
                    ro_storage_entry(6),
                ],
            });
        let pipeline_layout_jm = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::xpbd_equality::project_joint_mimic layout"),
            bind_group_layouts: &[
                Some(&layout_body),
                Some(&layout_eq),
                Some(&layout_uniforms),
                Some(&layout_joint_mimic),
            ],
            immediate_size: 0,
        });
        let pipeline_project_joint_mimic =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::xpbd_equality::project_joint_mimic"),
                layout: Some(&pipeline_layout_jm),
                module: &shader,
                entry_point: Some("project_joint_mimic"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            pipeline_project,
            pipeline_recover,
            pipeline_project_joint_mimic,
            layout_body,
            layout_eq,
            layout_uniforms,
            layout_joint_mimic,
            body_q_pre_eq: None,
            body_q_capacity: 0,
        }
    }

    /// Ensure the `body_q_pre_eq` scratch buffer exists and is at
    /// least `body_count` Transforms wide (32 bytes each — matches the
    /// `Transform` GPU mirror layout used by `XpbdKernel`).
    fn ensure_pre_eq_buffer(&mut self, device: &wgpu::Device, body_count: u32) {
        let needed = body_count.max(1);
        if needed <= self.body_q_capacity && self.body_q_pre_eq.is_some() {
            return;
        }
        self.body_q_pre_eq = Some(molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::xpbd_equality body_q_pre_eq"),
                size: 32u64 * needed as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        ));
        self.body_q_capacity = needed;
    }

    /// Run the equality-projection pass on `state_out.body_q`
    /// followed by a velocity-recovery pass that recomputes
    /// `state_out.body_qd` from `(state_out.body_q − state_in.body_q) / dt`.
    /// Replaces the host-side `apply_equality_post_gpu` shim.
    /// **Stage 32:** also dispatches the JOINT-mimic projection pass
    /// (`project_joint_mimic`) when the model carries any
    /// `EqualityType::JointMimic` entries; the pass mutates
    /// `state_out.joint_q` / `state_out.joint_qd` in place to enforce
    /// `q_driven = poly(q_driver)` (driver-authoritative).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        iterations: u32,
        dt: f32,
    ) -> Result<()> {
        if model.equality_count == 0 {
            return Ok(());
        }

        // `state_in` is unused now that the recovery pass diffs against
        // a snapshot of the post-main-kernel body_q (taken below) rather
        // than the pre-step pose. The parameter is kept on the public
        // signature for backwards compatibility with the dispatcher.
        let _ = state_in;

        // Make sure the snapshot scratch buffer is sized for this model.
        self.ensure_pre_eq_buffer(device, model.body_count as u32);
        let body_q_pre_eq_buf = self
            .body_q_pre_eq
            .as_ref()
            .expect("ensure_pre_eq_buffer just allocated it");

        let body_q_buf = state_out
            .body_q
            .device_buffer()
            .ok_or_else(missing_buf("state_out.body_q"))?;
        let body_qd_buf = state_out
            .body_qd
            .device_buffer()
            .ok_or_else(missing_buf("state_out.body_qd"))?;

        let bg_body = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_equality body bg"),
            layout: &self.layout_body,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: body_q_buf.as_entire_binding(),
                },
                buffer_entry(1, model.body_inv_mass.device_buffer())?,
                buffer_entry(2, model.body_inv_inertia.device_buffer())?,
                buffer_entry(3, model.body_com.device_buffer())?,
                buffer_entry(4, model.body_flags.device_buffer())?,
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: body_q_pre_eq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: body_qd_buf.as_entire_binding(),
                },
            ],
        });
        let bg_eq = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_equality eq bg"),
            layout: &self.layout_eq,
            entries: &[
                buffer_entry(0, model.equality_type.device_buffer())?,
                buffer_entry(1, model.equality_body_a.device_buffer())?,
                buffer_entry(2, model.equality_body_b.device_buffer())?,
                buffer_entry(3, model.equality_anchor_a.device_buffer())?,
                buffer_entry(4, model.equality_anchor_b.device_buffer())?,
                buffer_entry(5, model.equality_relative_rotation.device_buffer())?,
                buffer_entry(6, model.equality_enabled.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            body_count: model.body_count as u32,
            equality_count: model.equality_count as u32,
            iterations,
            dt,
        };
        let uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::xpbd_equality uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_equality uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // Joint-mimic bind group (Stage 32). Built unconditionally
        // when `equality_count > 0`; the WGSL kernel branches on
        // `equality_type` so non-mimic models incur a single skip
        // per slot and no work.
        let bg_jm = self.try_build_joint_mimic_bg(device, model, state_out)?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::xpbd_equality encoder"),
        });
        // 0. Snapshot body_q → body_q_pre_eq BEFORE the projection
        //    runs. The recovery pass below reads this snapshot to
        //    compute the equality-only displacement and ADDS it to
        //    the existing `body_qd` (which already carries the
        //    contact / friction / restitution velocity from the main
        //    XPBD kernel). Without this, the recovery would overwrite
        //    `body_qd` with `(body_q − body_q_pre_step) / dt`,
        //    destroying the friction velocity for every body that has
        //    an equality — symptom: rope tunnels through the floor,
        //    FourBar bars lose friction, arm-mimic chains jitter.
        encoder.copy_buffer_to_buffer(
            body_q_buf,
            0,
            body_q_pre_eq_buf,
            0,
            32u64 * model.body_count as u64,
        );
        // 1. Project equality (single-threaded Gauss-Seidel).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd_equality::project_equality"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_project);
            pass.set_bind_group(0, &bg_body, &[]);
            pass.set_bind_group(1, &bg_eq, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // 2. Velocity recovery — one thread per body.
        {
            let groups = (model.body_count as u32).div_ceil(RECOVER_WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd_equality::recover_velocity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_recover);
            pass.set_bind_group(0, &bg_body, &[]);
            pass.set_bind_group(1, &bg_eq, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        // 3. Joint-mimic projection (Stage 32). Skipped if the
        // joint state has no device-side mirror (model has no joints).
        if let Some(bg_jm) = bg_jm.as_ref() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd_equality::project_joint_mimic"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_project_joint_mimic);
            pass.set_bind_group(0, &bg_body, &[]);
            pass.set_bind_group(1, &bg_eq, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.set_bind_group(3, bg_jm, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        state_out.body_q.mark_device_dirty();
        state_out.body_qd.mark_device_dirty();
        if bg_jm.is_some() {
            state_out.joint_q.mark_device_dirty();
            state_out.joint_qd.mark_device_dirty();
        }
        Ok(())
    }

    /// Build the joint-mimic bind group, or return `None` if any
    /// required device buffer is missing (model has no joints, or the
    /// joint state hasn't been uploaded). Returning `None` skips the
    /// joint-mimic pipeline cleanly — non-jointed XPBD scenes (e.g.
    /// box stacks) shouldn't pay any cost for Stage 32.
    fn try_build_joint_mimic_bg(
        &self,
        device: &wgpu::Device,
        model: &Model,
        state_out: &State,
    ) -> Result<Option<wgpu::BindGroup>> {
        if model.joint_count == 0 {
            return Ok(None);
        }
        let Some(joint_q_buf) = state_out.joint_q.device_buffer() else {
            return Ok(None);
        };
        let Some(joint_qd_buf) = state_out.joint_qd.device_buffer() else {
            return Ok(None);
        };
        let Some(coord_off_buf) = model.joint_coord_offset.device_buffer() else {
            return Ok(None);
        };
        let Some(dof_off_buf) = model.joint_dof_offset.device_buffer() else {
            return Ok(None);
        };
        let Some(driver_buf) = model.equality_mimic_driver_joint.device_buffer() else {
            return Ok(None);
        };
        let Some(driven_buf) = model.equality_mimic_driven_joint.device_buffer() else {
            return Ok(None);
        };
        let Some(poly_buf) = model.equality_mimic_poly.device_buffer() else {
            return Ok(None);
        };
        Ok(Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_equality joint_mimic bg"),
            layout: &self.layout_joint_mimic,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: joint_q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: joint_qd_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: coord_off_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dof_off_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: driver_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: driven_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: poly_buf.as_entire_binding(),
                },
            ],
        })))
    }
}

fn missing_buf(name: &'static str) -> impl FnOnce() -> Error {
    move || Error::Gpu(format!("xpbd_equality: missing device buffer for {name}"))
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf
        .ok_or_else(|| Error::Gpu(format!("xpbd_equality binding {binding}: no device mirror")))?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn ro_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
fn rw_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
