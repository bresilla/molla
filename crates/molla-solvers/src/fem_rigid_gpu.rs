//! GPU-resident FEM/rigid substeps for one articulated contact island.

use crate::{FeatherstoneSolver, Solver, VbdSolver, fem_rigid::FemRigidConfig};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::momentum::{GeneralizedMomentumBindings, MomentumBindings, MomentumMonitor};
use molla_kernels::soft_rigid_contact::{
    SoftRigidContactBindings, SoftRigidContactKernel, SoftRigidSampleGpu, SoftRigidShapeGpu,
};
use molla_kernels::soft_rigid_solve::{
    SoftParticleDriftKernel, SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use molla_kernels::tet_volume::{TetVolumeBindings, TetVolumeMonitor};
use molla_kernels::{JacobianKernel, JacobianRequest, MassMatrixKernel};
use molla_math::Real;
use molla_sim::{Contacts, Control, Model, State};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use wgpu::util::DeviceExt;

fn device_queue(engine: &Engine) -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("FEM requires a GPU device".into()))?;
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("FEM requires a GPU queue".into()))?;
    Ok((device.clone(), queue.clone()))
}

fn validate_device_limits(limits: &wgpu::Limits) -> Result<()> {
    if limits.max_bind_groups < 6
        || limits.max_bindings_per_bind_group < 17
        || limits.max_storage_buffers_per_shader_stage < 42
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_storage_size < 2048
    {
        return Err(Error::Gpu(
            "FEM device requires 6 bind groups, 17 bindings per group, 42 storage buffers, \
             1 uniform buffer, 64 compute invocations in X, and 2048 workgroup bytes"
                .into(),
        ));
    }
    Ok(())
}

/// Raise a device request to the FEM kernel minimums without lowering other limits.
/// Unsupported adapters return an error without changing the request.
pub fn request_fem_device_limits(
    requested: &mut wgpu::Limits,
    supported: &wgpu::Limits,
) -> Result<()> {
    validate_device_limits(supported)?;
    requested.max_bind_groups = requested.max_bind_groups.max(6);
    requested.max_bindings_per_bind_group = requested.max_bindings_per_bind_group.max(17);
    requested.max_storage_buffers_per_shader_stage =
        requested.max_storage_buffers_per_shader_stage.max(42);
    requested.max_uniform_buffers_per_shader_stage =
        requested.max_uniform_buffers_per_shader_stage.max(1);
    requested.max_compute_invocations_per_workgroup =
        requested.max_compute_invocations_per_workgroup.max(64);
    requested.max_compute_workgroup_size_x = requested.max_compute_workgroup_size_x.max(64);
    requested.max_compute_workgroup_storage_size =
        requested.max_compute_workgroup_storage_size.max(2048);
    Ok(())
}

pub struct FemRigidGpuScene {
    pub soft_model: Model,
    pub soft_state: State,
    pub rigid_model: Model,
    pub rigid_state: State,
    pub control: Control,
    pub shapes: Vec<SoftRigidShapeGpu>,
}

#[cfg(test)]
mod device_limits_tests {
    use super::{request_fem_device_limits, validate_device_limits};

    #[test]
    fn external_device_requires_all_kernel_limits() {
        assert!(validate_device_limits(&wgpu::Limits::default()).is_err());
        let limits = wgpu::Limits {
            max_bind_groups: 6,
            max_bindings_per_bind_group: 17,
            max_storage_buffers_per_shader_stage: 42,
            max_uniform_buffers_per_shader_stage: 1,
            max_compute_invocations_per_workgroup: 64,
            max_compute_workgroup_size_x: 64,
            max_compute_workgroup_storage_size: 2048,
            ..Default::default()
        };
        assert!(validate_device_limits(&limits).is_ok());
        let mut requested = wgpu::Limits::default();
        let original = requested.clone();
        request_fem_device_limits(&mut requested, &limits).unwrap();
        assert!(validate_device_limits(&requested).is_ok());
        assert_eq!(requested.max_bind_groups, 6);
        assert_eq!(requested.max_storage_buffers_per_shader_stage, 42);
        let mut expected = original.clone();
        expected.max_bind_groups = 6;
        expected.max_storage_buffers_per_shader_stage = 42;
        assert_eq!(requested, expected);
        let mut larger = requested.clone();
        larger.max_bind_groups = 8;
        larger.max_storage_buffers_per_shader_stage = 64;
        larger.max_texture_array_layers = 2048;
        let preserved = larger.clone();
        request_fem_device_limits(&mut larger, &preserved).unwrap();
        assert_eq!(larger, preserved);
        let reductions: [fn(&mut wgpu::Limits); 7] = [
            |v| v.max_bind_groups -= 1,
            |v| v.max_bindings_per_bind_group -= 1,
            |v| v.max_storage_buffers_per_shader_stage -= 1,
            |v| v.max_uniform_buffers_per_shader_stage -= 1,
            |v| v.max_compute_invocations_per_workgroup -= 1,
            |v| v.max_compute_workgroup_size_x -= 1,
            |v| v.max_compute_workgroup_storage_size -= 1,
        ];
        for reduce in reductions {
            let mut insufficient = limits.clone();
            reduce(&mut insufficient);
            assert!(validate_device_limits(&insufficient).is_err());
            let mut unchanged = original.clone();
            assert!(request_fem_device_limits(&mut unchanged, &insufficient).is_err());
            assert_eq!(unchanged, original);
        }
    }
}

pub struct FemRigidGpuSystem {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    soft_model: Model,
    soft_state: State,
    soft_next: State,
    rigid_model: Model,
    rigid_state: State,
    rigid_next: State,
    control: Control,
    fem: VbdSolver,
    rigid: FeatherstoneSolver,
    mass: MassMatrixKernel,
    jacobian: JacobianKernel,
    requests: Vec<JacobianRequest>,
    mass_buffer: wgpu::Buffer,
    jacobian_buffer: wgpu::Buffer,
    surface: wgpu::Buffer,
    shapes: wgpu::Buffer,
    gather: SoftRigidContactKernel,
    solve: SoftRigidSolveKernel,
    drift: SoftParticleDriftKernel,
    volume: TetVolumeMonitor,
    momentum: Option<MomentumMonitor>,
    config: FemRigidConfig,
    queued_substeps: u32,
    completion: Option<Arc<AtomicBool>>,
}

impl FemRigidGpuSystem {
    pub fn new(
        engine: Arc<Engine>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
    ) -> Result<Self> {
        let (device, queue) = device_queue(&engine)?;
        Self::new_on_device(device, queue, scene, config)
    }

    /// Use vertices, edge midpoints and triangle centroids for primitive contacts.
    pub fn new_surface_sampled(
        engine: Arc<Engine>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
    ) -> Result<Self> {
        let (device, queue) = device_queue(&engine)?;
        Self::new_surface_sampled_on_device(device, queue, scene, config)
    }

    /// Construct on a matching externally managed device and queue.
    /// The caller retains device-loss handling and queue scheduling; this system
    /// neither replaces callbacks nor destroys the device. State buffers remain
    /// on that device and can be bound directly by its renderer.
    pub fn new_on_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
    ) -> Result<Self> {
        Self::build(device, queue, scene, config, false)
    }

    /// Construct sampled surface contacts on an externally managed device/queue.
    pub fn new_surface_sampled_on_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
    ) -> Result<Self> {
        Self::build(device, queue, scene, config, true)
    }

    fn build(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        mut scene: FemRigidGpuScene,
        config: FemRigidConfig,
        sample_surface: bool,
    ) -> Result<Self> {
        validate_device_limits(&device.limits())?;
        let device = &device;
        let queue = &queue;
        if scene.soft_state.particle_q.len() != scene.soft_model.particle_count
            || scene.soft_state.particle_qd.len() != scene.soft_model.particle_count
            || scene.rigid_state.body_q.len() != scene.rigid_model.body_count
            || scene.rigid_state.body_qd.len() != scene.rigid_model.body_count
            || scene.rigid_state.joint_q.len() != scene.rigid_model.joint_coord_total
            || scene.rigid_state.joint_qd.len() != scene.rigid_model.joint_dof_total
            || scene.control.joint_target_vel.len() != scene.rigid_model.joint_dof_total
            || scene
                .soft_state
                .particle_q
                .host()?
                .iter()
                .chain(scene.soft_state.particle_qd.host()?)
                .any(|v| !v.is_finite())
            || scene
                .rigid_state
                .joint_q
                .host()?
                .iter()
                .chain(scene.rigid_state.joint_qd.host()?)
                .any(|v| !v.is_finite())
        {
            return Err(Error::Build(
                "GPU FEM scene state does not match its models".into(),
            ));
        }
        if scene.soft_model.tet_count == 0
            || scene.soft_model.body_count != 0
            || scene.soft_model.shape_count != 0
            || scene.rigid_model.articulation_count > 1
            || scene.rigid_model.shape_count != 0
            || scene.rigid_model.joint_count > 32
            || !(1..=64).contains(&scene.rigid_model.joint_dof_total)
            || scene
                .rigid_model
                .joint_reversed
                .host()?
                .iter()
                .any(|&v| v != 0)
            || scene
                .rigid_model
                .joint_enabled
                .host()?
                .iter()
                .any(|&v| v == 0)
            || scene
                .rigid_model
                .body_flags
                .host()?
                .iter()
                .any(|&v| v & 3 != 0)
            || scene
                .control
                .joint_motor_model
                .host()?
                .iter()
                .any(|&v| v != 0)
            || scene
                .control
                .joint_motor_max_force
                .host()?
                .iter()
                .any(|v| v.is_finite())
            || !config.max_substep.is_finite()
            || config.max_substep <= 0.0
            || config.elastic_iterations == 0
            || !(1..=4096).contains(&config.contact.iterations)
            || !config.contact_margin.is_finite()
            || config.contact_margin < 0.0
            || !config.contact.penetration_recovery.is_finite()
            || !(0.0..=1.0).contains(&config.contact.penetration_recovery)
            || !config.contact.max_recovery_speed.is_finite()
            || config.contact.max_recovery_speed < 0.0
        {
            return Err(Error::Build(
                "unsupported GPU FEM island model or configuration".into(),
            ));
        }
        for shape in &scene.shapes {
            let quaternion_norm: f32 = shape.rotation.iter().map(|v| v * v).sum();
            if shape.ids[0] > 2
                || (shape.ids[1] != u32::MAX
                    && shape.ids[1] as usize >= scene.rigid_model.body_count)
                || shape
                    .position
                    .iter()
                    .chain(&shape.rotation)
                    .chain(&shape.data)
                    .any(|v| !v.is_finite())
                || (quaternion_norm - 1.0).abs() > 1e-5
                || shape.data[3] < 0.0
                || shape.data[0] <= 0.0
                || (shape.ids[0] > 0 && shape.data[1] <= 0.0)
                || (shape.ids[0] == 1 && shape.data[2] <= 0.0)
            {
                return Err(Error::Build("invalid GPU FEM primitive descriptor".into()));
            }
        }
        let tets: Vec<[u32; 4]> = scene
            .soft_model
            .tet_indices
            .host()?
            .chunks_exact(4)
            .map(|t| [t[0], t[1], t[2], t[3]])
            .collect();
        let triangles = molla_sim::compute_tet_surface_triangles(&tets);
        let mut nodes: Vec<_> = triangles.iter().flatten().copied().collect();
        nodes.sort_unstable();
        nodes.dedup();
        let mut samples = Vec::new();
        if sample_surface {
            samples.extend(nodes.iter().map(|&node| SoftRigidSampleGpu {
                nodes: [node, 0, 0, 1],
                weights: [1.0, 0.0, 0.0, 0.0],
            }));
            let mut edges = std::collections::BTreeSet::new();
            for &[a, b, c] in &triangles {
                for (i, j) in [(a, b), (b, c), (c, a)] {
                    edges.insert((i.min(j), i.max(j)));
                }
            }
            samples.extend(edges.into_iter().map(|(a, b)| SoftRigidSampleGpu {
                nodes: [a, b, 0, 2],
                weights: [0.5, 0.5, 0.0, 0.0],
            }));
            samples.extend(triangles.iter().map(|&[a, b, c]| SoftRigidSampleGpu {
                nodes: [a, b, c, 3],
                weights: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
            }));
        }
        let surface = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FEM surface features"),
            contents: if sample_surface {
                bytemuck::cast_slice(&samples)
            } else {
                bytemuck::cast_slice(&nodes)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let shape_count = scene.shapes.len();
        if scene.shapes.is_empty() {
            scene.shapes.push(bytemuck::Zeroable::zeroed());
        }
        let shapes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FEM rigid primitives"),
            contents: bytemuck::cast_slice(&scene.shapes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        scene.soft_model.upload_to_device(device, queue)?;
        scene.rigid_model.upload_to_device(device, queue)?;
        scene.soft_state.upload_to_device(device, queue)?;
        scene.rigid_state.upload_to_device(device, queue)?;
        scene.control.upload_to_device(device, queue)?;
        let mut soft_next = scene.soft_model.state()?;
        let mut rigid_next = scene.rigid_model.state()?;
        soft_next.upload_to_device(device, queue)?;
        rigid_next.upload_to_device(device, queue)?;
        let mass = MassMatrixKernel::new_on_device(device, queue, &scene.rigid_model)?;
        let jacobian = JacobianKernel::new_on_device(device, queue, &scene.rigid_model)?;
        let n = scene.rigid_model.joint_dof_total;
        let mass_buffer = mass.create_output_buffer(device);
        let jacobian_buffer =
            jacobian.create_output_buffer(device, scene.rigid_model.body_count * 6 * n);
        let requests = scene
            .rigid_model
            .body_com
            .host()?
            .iter()
            .enumerate()
            .map(|(body, p)| JacobianRequest {
                articulation: 0,
                target_body: body as u32,
                body_q_offset: 0,
                output_offset: (body * 6 * n) as u32,
                point_local: [p.x as f32, p.y as f32, p.z as f32, 0.0],
            })
            .collect();
        let gather = if sample_surface {
            SoftRigidContactKernel::new_sampled(device, samples.len() as u32, shape_count as u32)?
        } else {
            SoftRigidContactKernel::new(device, nodes.len() as u32, shape_count as u32)?
        };
        let solve = SoftRigidSolveKernel::new(device, n as u32, gather.row_count())?;
        let drift = SoftParticleDriftKernel::new(device);
        let volume = TetVolumeMonitor::new(device);
        let mut encoder = device.create_command_encoder(&Default::default());
        volume.encode(
            device,
            &mut encoder,
            TetVolumeBindings {
                positions: scene.soft_state.particle_q.device_buffer().unwrap(),
                reference: scene.soft_model.initial_particle_q.device_buffer().unwrap(),
                indices: scene.soft_model.tet_indices.device_buffer().unwrap(),
                inverse_rest: scene.soft_model.tet_dm_inv.device_buffer().unwrap(),
                status: solve.status(),
                particle_count: scene.soft_model.particle_count as u32,
                tet_count: scene.soft_model.tet_count as u32,
            },
        )?;
        queue.submit([encoder.finish()]);
        let fem = VbdSolver::new_gpu_on_device(device.clone(), queue.clone(), &scene.soft_model)?
            .with_iterations(config.elastic_iterations)
            .with_contact_stiffness(0.0);
        let rigid = FeatherstoneSolver::new_gpu_on_device(
            device.clone(),
            queue.clone(),
            &scene.rigid_model,
        )?
        .with_substeps(1)
        .with_rigorous_d_solve(true);
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            soft_model: scene.soft_model,
            soft_state: scene.soft_state,
            soft_next,
            rigid_model: scene.rigid_model,
            rigid_state: scene.rigid_state,
            rigid_next,
            control: scene.control,
            fem,
            rigid,
            mass,
            jacobian,
            requests,
            mass_buffer,
            jacobian_buffer,
            surface,
            shapes,
            gather,
            solve,
            drift,
            volume,
            momentum: None,
            config,
            queued_substeps: 0,
            completion: None,
        })
    }

    pub fn soft_state(&self) -> &State {
        &self.soft_state
    }
    pub fn rigid_state(&self) -> &State {
        &self.rigid_state
    }
    /// Sticky flags: 1 mass, 2 contact, 4 nonfinite, 8 invalid or nonpositive tetrahedron volume.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        self.solve.status()
    }

    /// Lifetime minimum J and smallest invalid tet index, including initial and every accepted substep.
    pub fn volume_diagnostics_buffer(&self) -> &wgpu::Buffer {
        self.volume.diagnostics()
    }

    /// Enable cumulative device-only momentum diagnostics without changing simulated state.
    pub fn enable_momentum_diagnostics(&mut self) {
        if self.momentum.is_none() {
            self.momentum = Some(MomentumMonitor::new(&self.device));
        }
    }

    /// Input, FEM prediction, rigid prediction, contact and final-drift momentum pairs, then deltas.
    pub fn momentum_diagnostics_buffer(&self) -> Option<&wgpu::Buffer> {
        self.momentum.as_ref().map(MomentumMonitor::diagnostics)
    }

    fn record_momentum(&self, phase: u32) -> Result<()> {
        let Some(monitor) = &self.momentum else {
            return Ok(());
        };
        let device = &self.device;
        let soft = if phase == 0 {
            &self.soft_state
        } else {
            &self.soft_next
        };
        let rigid = if phase < 2 {
            &self.rigid_state
        } else {
            &self.rigid_next
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        monitor.encode(
            device,
            &mut encoder,
            MomentumBindings {
                particle_mass: self.soft_model.particle_mass.device_buffer().unwrap(),
                particle_velocity: soft.particle_qd.device_buffer().unwrap(),
                body_mass: self.rigid_model.body_mass.device_buffer().unwrap(),
                body_velocity: rigid.body_qd.device_buffer().unwrap(),
                particles: self.soft_model.particle_count as u32,
                bodies: self.rigid_model.body_count as u32,
                generalized: (phase == 2).then_some(GeneralizedMomentumBindings {
                    joint_velocity: rigid.joint_qd.device_buffer().unwrap(),
                    jacobians: &self.jacobian_buffer,
                    dofs: self.rigid_model.joint_dof_total as u32,
                }),
            },
            phase,
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Update a velocity motor command without transferring simulated state to the host.
    pub fn set_drive_velocity(&mut self, dof: usize, target: Real) -> Result<()> {
        if !target.is_finite()
            || !(target as f32).is_finite()
            || self.control.joint_target_mode.host()?.get(dof).copied()
                != Some(molla_sim::JointTargetMode::Velocity as u32)
        {
            return Err(Error::Build(
                "invalid GPU FEM velocity motor command".into(),
            ));
        }
        self.control.joint_target_vel.host_mut()?[dof] = target;
        self.control.joint_target_vel.upload(&self.queue)
    }

    /// Select a direct generalized motor effort, subject to the model's joint effort limit.
    pub fn set_drive_effort(&mut self, dof: usize, effort: Real) -> Result<()> {
        if !effort.is_finite()
            || !(effort as f32).is_finite()
            || dof >= self.rigid_model.joint_dof_total
        {
            return Err(Error::Build("invalid GPU FEM effort command".into()));
        }
        self.control.joint_target_pos.host_mut()?[dof] = effort;
        self.control.joint_target_mode.host_mut()?[dof] = molla_sim::JointTargetMode::Effort as u32;
        let queue = &self.queue;
        self.control.joint_target_pos.upload(queue)?;
        self.control.joint_target_mode.upload(queue)
    }

    /// Upper bound on substeps submitted since the last completion fence.
    pub fn queued_substeps(&self) -> u32 {
        self.queued_substeps
    }

    fn arm_completion(&mut self) {
        if self.completion.is_none() && self.queued_substeps != 0 {
            let complete = Arc::new(AtomicBool::new(false));
            let signal = complete.clone();
            self.queue.on_submitted_work_done(move || {
                signal.store(true, Ordering::Release);
            });
            self.completion = Some(complete);
        }
    }

    /// Poll queue completion without waiting or transferring simulation state.
    /// Completion is not a device-health or physical-validity check.
    pub fn poll_completion(&mut self) -> Result<bool> {
        self.arm_completion();
        let Some(complete) = &self.completion else {
            return Ok(true);
        };
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| Error::Gpu(format!("FEM completion poll failed: {error}")))?;
        if !complete.load(Ordering::Acquire) {
            return Ok(false);
        }
        self.completion = None;
        self.queued_substeps = 0;
        Ok(true)
    }

    /// Submit at most one substep, returning false while earlier work is pending.
    /// True means submitted, not completed. No explicit GPU completion wait or
    /// state readback is performed; callers must handle device loss and status.
    pub fn try_step_substep(&mut self, dt: Real) -> Result<bool> {
        if !dt.is_finite() || dt <= 0.0 || dt > self.config.max_substep {
            return Err(Error::Solver("invalid GPU FEM single timestep".into()));
        }
        if !self.poll_completion()? {
            return Ok(false);
        }
        self.step_inner(dt, false)?;
        self.arm_completion();
        Ok(true)
    }

    /// Submit substeps in windows of 32; callers must inspect the sticky device status.
    pub fn step(&mut self, dt: Real) -> Result<()> {
        self.step_inner(dt, true)
    }

    fn step_inner(&mut self, dt: Real, wait: bool) -> Result<()> {
        let count = (dt / self.config.max_substep).ceil();
        if !dt.is_finite() || !(1.0..=4096.0).contains(&count) {
            return Err(Error::Solver("invalid GPU FEM timestep".into()));
        }
        let h = dt / count;
        if wait && self.completion.is_some() {
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| Error::Gpu(format!("FEM submission fence failed: {error}")))?;
            self.completion = None;
            self.queued_substeps = 0;
        }
        let device = &self.device;
        let queue = &self.queue;
        for _ in 0..count as usize {
            self.record_momentum(0)?;
            self.fem.step(
                &self.soft_model,
                &self.soft_state,
                &self.soft_model.control(),
                &Contacts::with_capacity(0),
                h,
                &mut self.soft_next,
            )?;
            self.record_momentum(1)?;
            self.rigid.step(
                &self.rigid_model,
                &self.rigid_state,
                &self.control,
                &Contacts::with_capacity(0),
                h,
                &mut self.rigid_next,
            )?;
            self.mass.dispatch_on_device(
                device,
                queue,
                &self.rigid_model,
                &self.rigid_state,
                &self.mass_buffer,
            )?;
            self.jacobian.dispatch_with_body_q_on_device(
                device,
                queue,
                &self.rigid_model,
                self.rigid_state.body_q.device_buffer().unwrap(),
                &self.requests,
                &self.jacobian_buffer,
            )?;
            self.record_momentum(2)?;
            let mut encoder = device.create_command_encoder(&Default::default());
            self.gather.encode(
                device,
                &mut encoder,
                SoftRigidContactBindings {
                    positions: self.soft_state.particle_q.device_buffer().unwrap(),
                    radii: self.soft_model.particle_radius.device_buffer().unwrap(),
                    surface: &self.surface,
                    shapes: &self.shapes,
                    body_poses: self.rigid_state.body_q.device_buffer().unwrap(),
                    particle_count: self.soft_model.particle_count as u32,
                    body_count: self.rigid_model.body_count as u32,
                },
                self.config.contact_margin as f32,
            )?;
            self.solve.encode(
                device,
                &mut encoder,
                SoftRigidSolveBindings {
                    positions: self.soft_state.particle_q.device_buffer().unwrap(),
                    velocities: self.soft_next.particle_qd.device_buffer().unwrap(),
                    inverse_mass: self.soft_model.particle_inv_mass.device_buffer().unwrap(),
                    contacts: self.gather.rows(),
                    body_poses: self.rigid_state.body_q.device_buffer().unwrap(),
                    body_com: self.rigid_model.body_com.device_buffer().unwrap(),
                    jacobians: &self.jacobian_buffer,
                    mass_matrix: &self.mass_buffer,
                    rigid_velocities: self.rigid_next.joint_qd.device_buffer().unwrap(),
                    body_velocities: self.rigid_next.body_qd.device_buffer().unwrap(),
                    particle_count: self.soft_model.particle_count as u32,
                    body_count: self.rigid_model.body_count as u32,
                },
                SoftRigidSolveSettings {
                    iterations: self.config.contact.iterations as u32,
                    dt: h as f32,
                    penetration_recovery: self.config.contact.penetration_recovery as f32,
                    max_recovery_speed: self.config.contact.max_recovery_speed as f32,
                },
            )?;
            self.drift.encode(
                device,
                &mut encoder,
                self.soft_state.particle_q.device_buffer().unwrap(),
                self.soft_next.particle_qd.device_buffer().unwrap(),
                self.soft_next.particle_q.device_buffer().unwrap(),
                self.solve.status(),
                self.soft_model.particle_count as u32,
                h as f32,
            )?;
            self.volume.encode(
                device,
                &mut encoder,
                TetVolumeBindings {
                    positions: self.soft_next.particle_q.device_buffer().unwrap(),
                    reference: self.soft_model.initial_particle_q.device_buffer().unwrap(),
                    indices: self.soft_model.tet_indices.device_buffer().unwrap(),
                    inverse_rest: self.soft_model.tet_dm_inv.device_buffer().unwrap(),
                    status: self.solve.status(),
                    particle_count: self.soft_model.particle_count as u32,
                    tet_count: self.soft_model.tet_count as u32,
                },
            )?;
            queue.submit([encoder.finish()]);
            self.record_momentum(3)?;
            self.rigid.drift_gpu(
                &self.rigid_model,
                &self.rigid_state,
                &self.control,
                h,
                &mut self.rigid_next,
            )?;
            self.jacobian.dispatch_with_body_q_on_device(
                device,
                queue,
                &self.rigid_model,
                self.rigid_next.body_q.device_buffer().unwrap(),
                &self.requests,
                &self.jacobian_buffer,
            )?;
            let mut encoder = device.create_command_encoder(&Default::default());
            self.solve.encode_velocity_recovery(
                device,
                &mut encoder,
                SoftRigidSolveBindings {
                    positions: self.soft_next.particle_q.device_buffer().unwrap(),
                    velocities: self.soft_next.particle_qd.device_buffer().unwrap(),
                    inverse_mass: self.soft_model.particle_inv_mass.device_buffer().unwrap(),
                    contacts: self.gather.rows(),
                    body_poses: self.rigid_next.body_q.device_buffer().unwrap(),
                    body_com: self.rigid_model.body_com.device_buffer().unwrap(),
                    jacobians: &self.jacobian_buffer,
                    mass_matrix: &self.mass_buffer,
                    rigid_velocities: self.rigid_next.joint_qd.device_buffer().unwrap(),
                    body_velocities: self.rigid_next.body_qd.device_buffer().unwrap(),
                    particle_count: self.soft_model.particle_count as u32,
                    body_count: self.rigid_model.body_count as u32,
                },
                SoftRigidSolveSettings {
                    iterations: self.config.contact.iterations as u32,
                    dt: h as f32,
                    penetration_recovery: self.config.contact.penetration_recovery as f32,
                    max_recovery_speed: self.config.contact.max_recovery_speed as f32,
                },
            )?;
            queue.submit([encoder.finish()]);
            self.record_momentum(4)?;
            self.soft_next.particle_qd.mark_device_dirty();
            self.rigid_next.joint_qd.mark_device_dirty();
            self.rigid_next.body_qd.mark_device_dirty();
            std::mem::swap(&mut self.soft_state, &mut self.soft_next);
            std::mem::swap(&mut self.rigid_state, &mut self.rigid_next);
            self.queued_substeps += 1;
            if wait && self.queued_substeps >= 32 {
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|error| Error::Gpu(format!("FEM submission fence failed: {error}")))?;
                self.queued_substeps = 0;
            }
        }
        Ok(())
    }
}
