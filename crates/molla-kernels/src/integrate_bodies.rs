//! Rust dispatch wrapper around `shaders/integrate_bodies.wgsl`.
//!
//! See PLAN.md §9 for the bind-group convention.

use molla_core::{Error, Result};
use molla_sim::{Control, Model, State};

/// Uniform block matching the `Uniforms` struct in WGSL.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dt: f32,
    angular_damping: f32,
    body_count: u32,
    has_external_force: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Compute pipeline + bind-group layouts for the rigid-body integrator.
pub struct IntegrateBodiesKernel {
    pipeline: wgpu::ComputePipeline,
    layout_model: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_control: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl IntegrateBodiesKernel {
    /// Compile the WGSL and lay out the pipeline. Cheap to call once per
    /// solver instance; expensive enough to want to cache.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::integrate_bodies"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/integrate_bodies.wgsl").into(),
            ),
        });

        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::integrate_bodies model"),
            entries: &storage_bgl_entries(7, true),
        });
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::integrate_bodies state"),
            entries: &state_bgl_entries(),
        });
        let layout_control = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::integrate_bodies control"),
            entries: &storage_bgl_entries(1, true),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::integrate_bodies uniforms"),
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
            label: Some("molla::integrate_bodies layout"),
            bind_group_layouts: &[
                &layout_model,
                &layout_state,
                &layout_control,
                &layout_uniforms,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::integrate_bodies"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("integrate_bodies"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_model,
            layout_state,
            layout_control,
            layout_uniforms,
        }
    }

    /// Dispatch one integration step. The caller must have already
    /// uploaded the model, both states, and (if used) control to the
    /// device — see [`Model::upload_to_device`] and friends.
    ///
    /// On return, `state_out`'s body buffers are marked device-dirty;
    /// call [`State::download_from_device`] before reading them on host.
    #[allow(
        clippy::too_many_arguments,
        reason = "explicit bind group sources are clearer than a struct here"
    )]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        control: &Control,
        external_f_dummy: &wgpu::Buffer,
        state_out: &mut State,
        dt: f64,
    ) -> Result<()> {
        if model.body_count == 0 {
            return Ok(());
        }

        // ── Model bind group ─────────────────────────────────────────
        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::integrate_bodies model bg"),
            layout: &self.layout_model,
            entries: &[
                buffer_entry(0, model.body_inv_mass.device_buffer())?,
                buffer_entry(1, model.body_inertia.device_buffer())?,
                buffer_entry(2, model.body_inv_inertia.device_buffer())?,
                buffer_entry(3, model.body_com.device_buffer())?,
                buffer_entry(4, model.body_flags.device_buffer())?,
                buffer_entry(5, model.body_world.device_buffer())?,
                buffer_entry(6, model.gravity.device_buffer())?,
            ],
        });

        // ── State bind group ─────────────────────────────────────────
        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::integrate_bodies state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, state_in.body_q.device_buffer())?,
                buffer_entry(1, state_in.body_qd.device_buffer())?,
                buffer_entry(2, state_in.body_f.device_buffer())?,
                buffer_entry(3, state_out.body_q.device_buffer())?,
                buffer_entry(4, state_out.body_qd.device_buffer())?,
            ],
        });

        // ── Control bind group ───────────────────────────────────────
        // If control supplies external forces sized to body_count, use it;
        // otherwise bind the dummy and tell the kernel to ignore it.
        let (external_buf, has_external) = if control.body_external_f.len() == model.body_count {
            (
                control.body_external_f.device_buffer().ok_or_else(|| {
                    Error::Gpu(
                        "control.body_external_f has no device buffer; \
                                 call control.upload_to_device() first"
                            .into(),
                    )
                })?,
                1u32,
            )
        } else {
            (external_f_dummy, 0u32)
        };
        let bg_control = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::integrate_bodies control bg"),
            layout: &self.layout_control,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: external_buf.as_entire_binding(),
            }],
        });

        // ── Uniforms bind group ──────────────────────────────────────
        let uniforms = Uniforms {
            dt: dt as f32,
            angular_damping: model.angular_damping as f32,
            body_count: model.body_count as u32,
            has_external_force: has_external,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::integrate_bodies uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::integrate_bodies uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ── Encode + dispatch ────────────────────────────────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::integrate_bodies encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::integrate_bodies pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_state, &[]);
            pass.set_bind_group(2, &bg_control, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            let groups = (model.body_count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        state_out.body_q.mark_device_dirty();
        state_out.body_qd.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "binding {binding}: buffer has no device mirror — call upload_to_device first"
        ))
    })?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn storage_bgl_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
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

fn state_bgl_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let rw = wgpu::BufferBindingType::Storage { read_only: false };
    (0..5)
        .map(|i| {
            let ty = if i < 3 { read } else { rw };
            wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect()
}
