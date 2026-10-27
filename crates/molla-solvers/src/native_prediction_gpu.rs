//! Experimental admitted control, native prediction and paired Featherstone drift.

use crate::generalized_prediction_gpu::{
    GeneralizedPredictionOutputs, PreparedGeneralizedPrediction,
};
use crate::{
    FeatherstoneSolver,
    compute::Engine,
    core::{Error, Result},
    sim::{Control, Model, State},
};
use molla_kernels::ExperimentalBiasBindings;

/// One retained input/output state direction for force-only or direct-effort control.
///
/// Requires an already prepared paired-rate Featherstone solver, one articulation
/// and one solver substep. PD/acceleration motors, implicit brakes, muscles,
/// friction, caps on velocity and compliance are rejected. Contact corrections,
/// momentum transport and world-level rollback remain the caller's responsibility.
pub struct PreparedNativePrediction {
    device: wgpu::Device,
    queue: wgpu::Queue,
    graph: PreparedGeneralizedPrediction,
    controls: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    references: Vec<wgpu::Buffer>,
    drive: wgpu::Buffer,
    dt: f64,
}

fn control_source(bodies: usize, dofs: usize) -> String {
    include_str!("../shaders/native_prediction_controls.wgsl")
        .replace("__BODIES__", &bodies.to_string())
        .replace("__DOFS__", &dofs.to_string())
}

fn validate(model: &Model, control: &Control, solver: &FeatherstoneSolver) -> Result<()> {
    let d = model.joint_dof_total;
    if !solver.gpu_uses_paired_rates()
        || solver.substeps != 1
        || !solver.rigorous_d_solve
        || model.muscle_count != 0
        || model.joint_effort_limit.len() != d
        || model.joint_compliance.len() != d
        || model
            .joint_compliance
            .host()?
            .iter()
            .any(|v| v.enabled != 0)
        || [
            control.joint_target_mode.len(),
            control.joint_target_pos.len(),
            control.joint_target_vel.len(),
            control.joint_target_kp.len(),
            control.joint_target_kd.len(),
            control.joint_motor_model.len(),
            control.joint_motor_max_force.len(),
            control.joint_force.len(),
            control.joint_damping.len(),
        ]
        .iter()
        .any(|&n| n != d)
        || control.body_external_f.len() != model.body_count
    {
        return Err(Error::Gpu(
            "unsupported native prediction solver, model or control layout".into(),
        ));
    }
    Ok(())
}

fn control_buffers<'a>(
    model: &'a Model,
    control: &'a Control,
    input: &'a State,
) -> Result<Vec<&'a wgpu::Buffer>> {
    Ok(vec![
        control.joint_target_mode.device_buffer_current()?,
        control.joint_target_pos.device_buffer_current()?,
        control.joint_motor_model.device_buffer_current()?,
        control.joint_motor_max_force.device_buffer_current()?,
        control.joint_force.device_buffer_current()?,
        control.joint_damping.device_buffer_current()?,
        model.joint_effort_limit.device_buffer_current()?,
        input.body_f.device_buffer_current()?,
        control.body_external_f.device_buffer_current()?,
    ])
}

fn references(
    model: &Model,
    control: &Control,
    solver: &FeatherstoneSolver,
    input: &State,
    output: &State,
) -> Result<Vec<wgpu::Buffer>> {
    let mut r: Vec<_> = control_buffers(model, control, input)?
        .into_iter()
        .cloned()
        .collect();
    macro_rules! add { ($($b:expr),* $(,)?) => { $(r.push($b.clone());)* }; }
    add!(
        solver
            .gpu_status_buffer()
            .ok_or_else(|| Error::Gpu("missing native prediction status".into()))?,
        solver.gpu_active_articulations_buffer()?,
        input.body_q.device_buffer_current()?,
        input.body_qd.device_buffer_current()?,
        input.joint_q.device_buffer_current()?,
        input.joint_qd.device_buffer_current()?,
        output.body_q.device_buffer_current()?,
        output.body_qd.device_buffer_current()?,
        output.joint_q.device_buffer_current()?,
        output.joint_qd.device_buffer_current()?,
        output.body_f.device_buffer_current()?,
        solver.gpu_paired_rates_buffer(input)?,
        solver.gpu_paired_rates_buffer(output)?,
        solver.gpu_position_compensation_buffer(input)?,
        solver.gpu_position_compensation_buffer(output)?,
        solver.gpu_coordinate_compensation_buffer(input)?,
        solver.gpu_coordinate_compensation_buffer(output)?,
        solver.gpu_body_velocity_compensation_buffer(input)?,
        solver.gpu_body_velocity_compensation_buffer(output)?,
        solver
            .gpu_paired_velocity_delta_buffer()
            .ok_or_else(|| Error::Gpu("missing paired predictor delta".into()))?,
        solver.gpu_velocity_delta_buffer().unwrap()
    );
    macro_rules! model_buffers { ($($field:ident),*) => { $(r.push(model.$field.device_buffer_current()?.clone());)* }; }
    model_buffers!(
        joint_type,
        joint_parent,
        joint_child,
        joint_x_p,
        joint_x_c,
        joint_coord_offset,
        joint_dof_offset,
        joint_axis,
        joint_reversed,
        joint_motion_flags,
        joint_limit_lower,
        joint_limit_upper,
        joint_friction,
        joint_velocity_limit,
        joint_compliance,
        joint_limit_references,
        joint_frame_references,
        body_mass,
        body_inertia,
        body_com,
        body_flags,
        body_world,
        gravity
    );
    add!(
        control.joint_target_vel.device_buffer_current()?,
        control.joint_target_kp.device_buffer_current()?,
        control.joint_target_kd.device_buffer_current()?
    );
    Ok(r)
}

impl PreparedNativePrediction {
    pub fn required_bytes(model: &Model) -> Result<u64> {
        Ok(
            PreparedGeneralizedPrediction::required_bytes(model.body_count, model.joint_dof_total)?
                + 8 * model.joint_dof_total as u64
                + 32 * model.body_count as u64,
        )
    }

    /// Retains one parity direction; prepare the reverse direction separately.
    pub fn new(
        engine: &Engine,
        solver: &FeatherstoneSolver,
        model: &Model,
        control: &Control,
        input: &State,
        output: &State,
        dt: f64,
        budget: u64,
    ) -> Result<Self> {
        validate(model, control, solver)?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("native prediction requires GPU".into()))?;
        let queue = engine.queue().unwrap();
        solver.validate_gpu_prediction_device(device)?;
        let required = Self::required_bytes(model)?;
        let n = model.body_count as u64;
        let d = model.joint_dof_total as u64;
        if budget < required
            || device.limits().max_storage_buffers_per_shader_stage < 13
            || device.limits().max_bindings_per_bind_group < 13
        {
            return Err(Error::Gpu(
                "native prediction budget or device limits exceeded".into(),
            ));
        }
        let references = references(model, control, solver, input, output)?;
        let create = |size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("native prediction inputs"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false
                }
            )
        };
        let drive = create(8 * d);
        let loads = create(32 * n);
        let status = solver.gpu_status_buffer().unwrap();
        let active = solver.gpu_active_articulations_buffer()?;
        let graph = PreparedGeneralizedPrediction::new_with_active(
            device,
            queue,
            model,
            ExperimentalBiasBindings {
                poses: input.body_q.device_buffer_current()?,
                position_low: solver.gpu_position_compensation_buffer(input)?,
                paired_rates: solver.gpu_paired_rates_buffer(input)?,
                external_com_wrenches: &loads,
            },
            &drive,
            GeneralizedPredictionOutputs {
                paired_rates: solver.gpu_paired_rates_buffer(output)?,
                scalar_rates: output.joint_qd.device_buffer_current()?,
                paired_delta: solver.gpu_paired_velocity_delta_buffer().unwrap(),
                scalar_delta: solver.gpu_velocity_delta_buffer().unwrap(),
                status,
            },
            dt,
            solver.joint_damping as f32,
            required - 8 * d - 32 * n,
            active,
        )?;
        let sizes = [
            4 * d,
            4 * d,
            4 * d,
            4 * d,
            4 * d,
            4 * d,
            4 * d,
            32 * n,
            32 * n,
            4,
            4,
            8 * d,
            32 * n,
        ];
        let entries: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: i < 9 || i == 10,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(size),
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native control"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("native control"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native control"),
            source: wgpu::ShaderSource::Wgsl(control_source(n as usize, d as usize).into()),
        });
        let controls = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("native control"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut buffers = control_buffers(model, control, input)?;
        buffers.extend([status, active, &drive, &loads]);
        let entries: Vec<_> = buffers
            .iter()
            .zip(sizes)
            .enumerate()
            .map(|(i, (buffer, size))| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(size),
                }),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native control"),
            layout: &layout,
            entries: &entries,
        });
        Ok(Self {
            device: device.as_ref().clone(),
            queue: queue.as_ref().clone(),
            graph,
            controls,
            group,
            references,
            drive,
            dt,
        })
    }

    /// Predicts one substep then drifts coordinates and reconstructs body state.
    pub fn step(
        &self,
        solver: &mut FeatherstoneSolver,
        model: &Model,
        control: &Control,
        input: &State,
        output: &mut State,
        dt: f64,
    ) -> Result<()> {
        self.step_profiled(solver, model, control, input, output, dt, None)
    }

    pub(crate) fn step_profiled(
        &self,
        solver: &mut FeatherstoneSolver,
        model: &Model,
        control: &Control,
        input: &State,
        output: &mut State,
        dt: f64,
        timestamps: Option<(wgpu::QuerySet, [u32; 2])>,
    ) -> Result<()> {
        self.predict_profiled(solver, model, control, input, output, dt, timestamps, true)
    }

    /// Writes rates and deltas only; the caller must finish drift before publishing poses.
    pub(crate) fn rates_profiled(
        &self,
        solver: &mut FeatherstoneSolver,
        model: &Model,
        control: &Control,
        input: &State,
        output: &mut State,
        dt: f64,
        timestamps: Option<(wgpu::QuerySet, [u32; 2])>,
    ) -> Result<()> {
        self.predict_profiled(solver, model, control, input, output, dt, timestamps, false)
    }

    fn predict_profiled(
        &self,
        solver: &mut FeatherstoneSolver,
        model: &Model,
        control: &Control,
        input: &State,
        output: &mut State,
        dt: f64,
        timestamps: Option<(wgpu::QuerySet, [u32; 2])>,
        drift: bool,
    ) -> Result<()> {
        validate(model, control, solver)?;
        solver.validate_gpu_prediction_device(&self.device)?;
        self.graph
            .validate_model(model, dt, solver.joint_damping as f32)?;
        if dt.to_bits() != self.dt.to_bits()
            || references(model, control, solver, input, output)? != self.references
        {
            return Err(Error::Gpu(
                "native prediction owners or timestep changed; prepare again".into(),
            ));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let count = model.joint_dof_total as u64;
        encoder.copy_buffer_to_buffer(
            input.joint_qd.device_buffer_current()?,
            0,
            output.joint_qd.device_buffer_current()?,
            0,
            4 * count,
        );
        encoder.copy_buffer_to_buffer(
            solver.gpu_paired_rates_buffer(input)?,
            0,
            solver.gpu_paired_rates_buffer(output)?,
            0,
            8 * count,
        );
        encoder.clear_buffer(solver.gpu_velocity_delta_buffer().unwrap(), 0, None);
        encoder.clear_buffer(solver.gpu_paired_velocity_delta_buffer().unwrap(), 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("native prediction controls"),
                timestamp_writes: timestamps.as_ref().map(|(query_set, indices)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(indices[0]),
                        end_of_pass_write_index: None,
                    }
                }),
            });
            pass.set_pipeline(&self.controls);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        self.graph.encode(&mut encoder);
        if !drift && let Some((query, indices)) = &timestamps {
            encoder.write_timestamp(query, indices[1]);
        }
        self.queue.submit([encoder.finish()]);
        output.joint_qd.mark_device_dirty();
        if drift {
            if let Some((query, indices)) = timestamps {
                solver.set_gpu_dispatch_end_timestamp(query, indices[1]);
            }
            solver.drift_gpu(model, input, control, dt, output)?;
            output.body_f.mark_device_dirty();
        }
        Ok(())
    }

    /// Clears internal predictor faults; shared status and state recovery remain caller-owned.
    pub fn reset_internal(&self, encoder: &mut wgpu::CommandEncoder) {
        self.graph.reset_internal(encoder);
    }
    pub fn mass_matrix(&self) -> &wgpu::Buffer {
        self.graph.mass_matrix()
    }

    /// COM Jacobians evaluated with the retained input-configuration mass matrix.
    pub fn com_jacobians(&self) -> &wgpu::Buffer {
        self.graph.com_jacobians()
    }
    pub fn applied_efforts(&self) -> &wgpu::Buffer {
        &self.drive
    }
}

#[cfg(test)]
#[path = "native_prediction_tests.rs"]
mod tests;
