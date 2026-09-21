//! GPU-resident FEM/rigid substeps for one articulated contact island.

use crate::{FeatherstoneSolver, Solver, VbdSolver, fem_rigid::FemRigidConfig};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::momentum::{GeneralizedMomentumBindings, MomentumBindings, MomentumMonitor};
use molla_kernels::soft_rigid_contact::{
    SoftRigidContactBindings, SoftRigidContactKernel, SoftRigidShapeGpu,
};
use molla_kernels::soft_rigid_solve::{
    SoftParticleDriftKernel, SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use molla_kernels::tet_volume::{TetVolumeBindings, TetVolumeMonitor};
use molla_kernels::{JacobianKernel, JacobianRequest, MassMatrixKernel};
use molla_math::Real;
use molla_sim::{Contacts, Control, Model, State};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct FemRigidGpuScene {
    pub soft_model: Model,
    pub soft_state: State,
    pub rigid_model: Model,
    pub rigid_state: State,
    pub control: Control,
    pub shapes: Vec<SoftRigidShapeGpu>,
}

pub struct FemRigidGpuSystem {
    engine: Arc<Engine>,
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
}

impl FemRigidGpuSystem {
    pub fn new(
        engine: Arc<Engine>,
        mut scene: FemRigidGpuScene,
        config: FemRigidConfig,
    ) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("FEM requires a GPU device".into()))?;
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("FEM requires a GPU queue".into()))?;
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
        let mut nodes: Vec<_> = molla_sim::compute_tet_surface_triangles(&tets)
            .into_iter()
            .flatten()
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        let surface = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FEM surface nodes"),
            contents: bytemuck::cast_slice(&nodes),
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
        let mass = MassMatrixKernel::new(&engine, &scene.rigid_model)?;
        let jacobian = JacobianKernel::new(&engine, &scene.rigid_model)?;
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
        let gather = SoftRigidContactKernel::new(device, nodes.len() as u32, shape_count as u32)?;
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
        let fem = VbdSolver::new_gpu(&engine, &scene.soft_model)?
            .with_iterations(config.elastic_iterations)
            .with_contact_stiffness(0.0);
        let rigid = FeatherstoneSolver::new_gpu(&engine, &scene.rigid_model)?
            .with_substeps(1)
            .with_rigorous_d_solve(true);
        Ok(Self {
            engine,
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
            self.momentum = Some(MomentumMonitor::new(self.engine.device().unwrap()));
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
        let device = self.engine.device().unwrap();
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
        self.engine.queue().unwrap().submit([encoder.finish()]);
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
        self.control
            .joint_target_vel
            .upload(self.engine.queue().unwrap())
    }

    /// Submit GPU substeps; callers must inspect the sticky device status before accepting results.
    pub fn step(&mut self, dt: Real) -> Result<()> {
        let count = (dt / self.config.max_substep).ceil();
        if !dt.is_finite() || !(1.0..=4096.0).contains(&count) {
            return Err(Error::Solver("invalid GPU FEM timestep".into()));
        }
        let h = dt / count;
        let device = self.engine.device().unwrap();
        let queue = self.engine.queue().unwrap();
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
            self.mass.dispatch(
                &self.engine,
                &self.rigid_model,
                &self.rigid_state,
                &self.mass_buffer,
            )?;
            self.jacobian.dispatch(
                &self.engine,
                &self.rigid_model,
                &self.rigid_state,
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
            self.jacobian.dispatch(
                &self.engine,
                &self.rigid_model,
                &self.rigid_next,
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
        }
        Ok(())
    }
}
