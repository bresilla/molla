//! GPU-resident FEM/rigid substeps for one articulated contact island.

use crate::{FeatherstoneSolver, Solver, VbdSolver, fem_rigid::FemRigidConfig};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::momentum::{GeneralizedMomentumBindings, MomentumBindings, MomentumMonitor};
pub use molla_kernels::soft_rigid_contact::{SoftRigidBallJoint, SoftRigidShapeGpu};
use molla_kernels::soft_rigid_contact::{
    SoftRigidContactBindings, SoftRigidContactKernel, SoftRigidSampleGpu,
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
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use wgpu::util::DeviceExt;

mod bounds;
mod contact_snapshot;
mod coupling;
mod global_newton;
mod timing;
pub use contact_snapshot::ContactSolveSnapshot;
pub use coupling::MaterialContactConfig;
use coupling::MaterialContactState;
pub use global_newton::GlobalNewtonConfig;
use global_newton::GlobalNewtonState;

fn device_queue(engine: &Engine) -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("FEM requires a GPU device".into()))?;
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("FEM requires a GPU queue".into()))?;
    Ok((device.clone(), queue.clone()))
}

fn sampled_contact_capacity(
    limits: &wgpu::Limits,
    samples: u32,
    shapes: u32,
    dofs: usize,
    loops: usize,
) -> Result<u32> {
    let desired = u64::from(samples) * u64::from(shapes.min(4));
    let row_bytes = (12 * dofs as u64)
        .max(std::mem::size_of::<molla_kernels::soft_rigid_contact::SoftRigidContactGpu>() as u64);
    let bytes = (128 * 1024 * 1024_u64)
        .min(u64::from(limits.max_storage_buffer_binding_size))
        .min(limits.max_buffer_size);
    let rows = (bytes / row_bytes).min(u64::from(limits.max_compute_workgroups_per_dimension) * 64);
    let capacity = rows
        .checked_sub(loops as u64)
        .ok_or_else(|| Error::Build("FEM loop rows exceed contact budget".into()))?
        .min(desired);
    if desired != 0 && capacity == 0 {
        return Err(Error::Build("FEM contact budget has no sample rows".into()));
    }
    u32::try_from(capacity).map_err(|_| Error::Build("FEM contact capacity overflow".into()))
}

fn validate_device_limits(limits: &wgpu::Limits) -> Result<()> {
    if limits.max_bind_groups < 6
        || limits.max_bindings_per_bind_group < 17
        || limits.max_storage_buffers_per_shader_stage < 43
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_storage_size < 2060
    {
        return Err(Error::Gpu(
            "FEM device requires 6 bind groups, 17 bindings per group, 43 storage buffers, \
             1 uniform buffer, 64 compute invocations in X, and 2060 workgroup bytes"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod capacity_tests {
    use super::sampled_contact_capacity;

    #[test]
    fn sampled_rows_respect_response_budget_and_loop_tail() {
        let mut limits = wgpu::Limits::default();
        let old = sampled_contact_capacity(&limits, 53760, 70, 29, 2).unwrap();
        assert_eq!(old, 53760 * 4);
        let large = sampled_contact_capacity(&limits, 160000, 70, 29, 2).unwrap();
        assert_eq!(large, 128 * 1024 * 1024 / (12 * 29) - 2);
        assert!(u64::from(large + 2) * 12 * 29 <= 128 * 1024 * 1024);
        limits.max_storage_buffer_binding_size = 4096;
        assert_eq!(
            sampled_contact_capacity(&limits, 160000, 70, 29, 2).unwrap(),
            4096 / (12 * 29) - 2
        );
        limits.max_storage_buffer_binding_size = 1024 * 1024;
        limits.max_buffer_size = 2048;
        assert_eq!(
            sampled_contact_capacity(&limits, 160000, 70, 29, 2).unwrap(),
            2048 / (12 * 29) - 2
        );
        limits.max_compute_workgroups_per_dimension = 1;
        limits.max_buffer_size = 1024 * 1024;
        assert_eq!(
            sampled_contact_capacity(&limits, 160000, 70, 6, 2).unwrap(),
            62
        );
        assert!(sampled_contact_capacity(&limits, 1, 1, 6, 64).is_err());
        assert!(sampled_contact_capacity(&limits, 1, 1, 6, 65).is_err());
        assert_eq!(sampled_contact_capacity(&limits, 1000, 0, 6, 2).unwrap(), 0);
    }
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
        requested.max_storage_buffers_per_shader_stage.max(43);
    requested.max_uniform_buffers_per_shader_stage =
        requested.max_uniform_buffers_per_shader_stage.max(1);
    requested.max_compute_invocations_per_workgroup =
        requested.max_compute_invocations_per_workgroup.max(64);
    requested.max_compute_workgroup_size_x = requested.max_compute_workgroup_size_x.max(64);
    requested.max_compute_workgroup_storage_size =
        requested.max_compute_workgroup_storage_size.max(2060);
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

/// Raises a device request to the coupled material patch limits without lowering other limits.
pub fn request_material_contact_patch_device_limits(
    requested: &mut wgpu::Limits,
    supported: &wgpu::Limits,
) -> Result<()> {
    if supported.max_bind_groups < 7 || supported.max_storage_buffers_per_shader_stage < 51 {
        return Err(Error::Gpu(
            "material contact patch requires 7 groups and 51 storage buffers".into(),
        ));
    }
    request_fem_device_limits(requested, supported)?;
    requested.max_bind_groups = requested.max_bind_groups.max(7);
    requested.max_storage_buffers_per_shader_stage =
        requested.max_storage_buffers_per_shader_stage.max(51);
    Ok(())
}

#[cfg(test)]
mod device_limits_tests {
    use super::{
        request_fem_device_limits, request_material_contact_patch_device_limits,
        validate_device_limits,
    };

    #[test]
    fn patch_limits_preserve_requests_and_reject_unsupported_devices() {
        let mut supported = wgpu::Limits::default();
        supported.max_bind_groups = 7;
        supported.max_storage_buffers_per_shader_stage = 51;
        let original = wgpu::Limits::default();
        let mut request = original.clone();
        request_material_contact_patch_device_limits(&mut request, &supported).unwrap();
        assert_eq!(request.max_bind_groups, 7);
        assert_eq!(request.max_storage_buffers_per_shader_stage, 51);
        for (groups, buffers) in [(6, 51), (7, 50)] {
            let mut small = supported.clone();
            small.max_bind_groups = groups;
            small.max_storage_buffers_per_shader_stage = buffers;
            let mut unchanged = original.clone();
            assert!(request_material_contact_patch_device_limits(&mut unchanged, &small).is_err());
            assert_eq!(unchanged, original);
        }
        let mut large = supported.clone();
        large.max_bind_groups = 8;
        large.max_storage_buffers_per_shader_stage = 64;
        let unchanged = large.clone();
        request_material_contact_patch_device_limits(&mut large, &supported).unwrap();
        assert_eq!(large, unchanged);
    }

    #[test]
    fn external_device_requires_all_kernel_limits() {
        assert!(validate_device_limits(&wgpu::Limits::default()).is_err());
        let limits = wgpu::Limits {
            max_bind_groups: 6,
            max_bindings_per_bind_group: 17,
            max_storage_buffers_per_shader_stage: 43,
            max_uniform_buffers_per_shader_stage: 1,
            max_compute_invocations_per_workgroup: 64,
            max_compute_workgroup_size_x: 64,
            max_compute_workgroup_storage_size: 2060,
            ..Default::default()
        };
        assert!(validate_device_limits(&limits).is_ok());
        let mut requested = wgpu::Limits::default();
        let original = requested.clone();
        request_fem_device_limits(&mut requested, &limits).unwrap();
        assert!(validate_device_limits(&requested).is_ok());
        assert_eq!(requested.max_bind_groups, 6);
        assert_eq!(requested.max_storage_buffers_per_shader_stage, 43);
        let mut expected = original.clone();
        expected.max_bind_groups = 6;
        expected.max_storage_buffers_per_shader_stage = 43;
        assert_eq!(requested, expected);
        let mut small = original.clone();
        small.max_compute_workgroup_storage_size = 1024;
        request_fem_device_limits(&mut small, &limits).unwrap();
        assert_eq!(small.max_compute_workgroup_storage_size, 2060);
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
    bounds: Option<molla_kernels::soft_rigid_bounds::SoftRigidBoundsKernel>,
    solve: SoftRigidSolveKernel,
    drift: SoftParticleDriftKernel,
    volume: TetVolumeMonitor,
    momentum: Option<MomentumMonitor>,
    momentum_fences: Option<Arc<[AtomicU64; 5]>>,
    contact_fences: Option<Arc<[AtomicU64; 7]>>,
    material_contact: Option<MaterialContactState>,
    global_newton: Option<GlobalNewtonState>,
    timing: Option<timing::SubstepTiming>,
    contact_snapshot: Option<ContactSolveSnapshot>,
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
    /// GPU compaction reserves up to four rows per feature, capped at 128 MiB per buffer.
    /// Exhausting that pool latches status bit 16; no contacts are silently dropped.
    pub fn new_surface_sampled(
        engine: Arc<Engine>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
    ) -> Result<Self> {
        let (device, queue) = device_queue(&engine)?;
        Self::new_surface_sampled_on_device(device, queue, scene, config)
    }

    /// Construct on a matching externally managed device and queue, initializing
    /// rigid body transforms and velocities from generalized coordinates.
    /// The caller retains device-loss handling and queue scheduling; this system
    /// neither replaces callbacks nor destroys the device. State buffers remain
    /// on that device and can be bound directly by its renderer.
    pub fn new_on_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
    ) -> Result<Self> {
        Self::build(device, queue, scene, config, false, &[], None, false)
    }

    /// Construct sampled surface contacts on an externally managed device/queue.
    pub fn new_surface_sampled_on_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
    ) -> Result<Self> {
        Self::build(device, queue, scene, config, true, &[], None, false)
    }

    /// Sampled surface contacts and bilateral body-local spherical closures.
    pub fn new_surface_sampled_with_ball_joints_on_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
        joints: &[SoftRigidBallJoint],
    ) -> Result<Self> {
        Self::build(device, queue, scene, config, true, joints, None, false)
    }

    /// Sampled contacts with an explicit shared contact capacity, excluding loop rows.
    /// Overflow latches device status bit 16 and prevents contact impulses.
    pub fn new_surface_sampled_with_contact_capacity_on_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
        joints: &[SoftRigidBallJoint],
        contact_capacity: u32,
    ) -> Result<Self> {
        Self::build(
            device,
            queue,
            scene,
            config,
            true,
            joints,
            Some(contact_capacity),
            false,
        )
    }

    /// Sampled surface contacts with GPU coplanar point/edge blocks and optional capacity.
    pub fn new_surface_sampled_with_contact_blocks_on_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        scene: FemRigidGpuScene,
        config: FemRigidConfig,
        joints: &[SoftRigidBallJoint],
        contact_capacity: Option<u32>,
    ) -> Result<Self> {
        Self::build(
            device,
            queue,
            scene,
            config,
            true,
            joints,
            contact_capacity,
            true,
        )
    }

    fn build(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        mut scene: FemRigidGpuScene,
        config: FemRigidConfig,
        sample_surface: bool,
        joints: &[SoftRigidBallJoint],
        contact_capacity: Option<u32>,
        contact_blocks: bool,
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
                .any(|v| v.is_nan() || *v < 0.0)
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
            shape.validate(scene.rigid_model.body_count)?;
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
        molla_sim::eval_fk(&scene.rigid_model, &mut scene.rigid_state)?;
        molla_sim::eval_body_velocities(&scene.rigid_model, &mut scene.rigid_state)?;
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
        let (bounds, bound_rows) = bounds::build(device, &scene.rigid_model)?;
        let mut constraint_rows = joints
            .iter()
            .map(|j| j.row(scene.rigid_model.body_count as u32))
            .collect::<Result<Vec<_>>>()?;
        constraint_rows.extend(bound_rows);
        let gather = if sample_surface {
            let sample_count = u32::try_from(samples.len())
                .map_err(|_| Error::Build("too many FEM surface features".into()))?;
            let shape_count = u32::try_from(shape_count)
                .map_err(|_| Error::Build("too many FEM contact shapes".into()))?;
            let capacity = match contact_capacity {
                Some(capacity) => capacity,
                None => sampled_contact_capacity(
                    &device.limits(),
                    sample_count,
                    shape_count,
                    n,
                    constraint_rows.len(),
                )?,
            };
            SoftRigidContactKernel::new_with_constraint_rows(
                device,
                queue,
                sample_count,
                shape_count,
                true,
                Some(capacity),
                &constraint_rows,
            )?
        } else {
            SoftRigidContactKernel::new_with_constraint_rows(
                device,
                queue,
                nodes.len() as u32,
                shape_count as u32,
                false,
                None,
                &constraint_rows,
            )?
        };
        let solve = if contact_blocks {
            SoftRigidSolveKernel::new_blocked_contacts(
                device,
                n as u32,
                gather.row_count(),
                scene.soft_model.particle_count as u32,
            )
        } else {
            SoftRigidSolveKernel::new_separated_static(
                device,
                n as u32,
                gather.row_count(),
                scene.soft_model.particle_count as u32,
            )
        }?;
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
        let mut rigid = FeatherstoneSolver::new_gpu_on_device(
            device.clone(),
            queue.clone(),
            &scene.rigid_model,
        )?
        .with_substeps(1)
        .with_rigorous_d_solve(true);
        rigid.set_gpu_status_buffer(solve.status())?;
        rigid.set_external_scalar_bounds(true)?;
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
            bounds,
            solve,
            drift,
            volume,
            momentum: None,
            momentum_fences: None,
            contact_fences: None,
            config,
            queued_substeps: 0,
            completion: None,
            material_contact: None,
            global_newton: None,
            timing: None,
            contact_snapshot: None,
        })
    }

    /// Enable effective-inertia stabilization of scalar rigid feedback drives.
    #[must_use]
    pub fn with_stable_motor_feedback(mut self, enabled: bool) -> Self {
        self.rigid.stable_motor_feedback = enabled;
        self
    }

    pub fn soft_state(&self) -> &State {
        &self.soft_state
    }

    /// Enable device-only first-contact input and pre-drift velocity copies while idle.
    pub fn configure_contact_snapshot(&mut self, enabled: bool) -> Result<()> {
        if self.queued_substeps != 0 || self.completion.is_some() {
            return Err(Error::Build(
                "contact snapshot configuration requires an idle island".into(),
            ));
        }
        self.contact_snapshot = enabled.then(|| {
            ContactSolveSnapshot::new(
                &self.device,
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
            )
        });
        Ok(())
    }

    /// Last substep's contact boundary snapshots; wait for completion before reading.
    pub fn contact_snapshot(&self) -> Option<&ContactSolveSnapshot> {
        self.contact_snapshot
            .as_ref()
            .filter(|snapshot| snapshot.settings.is_some())
    }
    /// Configure material/contact re-solves while the submission queue is idle.
    pub fn configure_material_contact(
        &mut self,
        config: Option<MaterialContactConfig>,
    ) -> Result<()> {
        if self.queued_substeps != 0 || self.completion.is_some() {
            return Err(Error::Build(
                "material contact configuration requires an idle island".into(),
            ));
        }
        let state = config
            .map(|config| {
                MaterialContactState::new(
                    &self.device,
                    &self.queue,
                    &self.soft_model,
                    self.rigid_model.body_count,
                    config,
                )
            })
            .transpose()?;
        self.material_contact = state;
        self.global_newton = None;
        self.fem.clear_material_contact_patch_gpu();
        self.timing = None;
        Ok(())
    }

    /// Selects the simultaneous GPU solve; None keeps the alternating material/contact path.
    /// Requires an idle elastic island with material contact configured and no patch/timing trace.
    pub fn configure_global_newton(&mut self, config: Option<GlobalNewtonConfig>) -> Result<()> {
        if self.queued_substeps != 0 || self.completion.is_some() {
            return Err(Error::Build(
                "global Newton configuration requires an idle island".into(),
            ));
        }
        let state = if let Some(config) = config {
            let coupling = self
                .material_contact
                .as_ref()
                .ok_or_else(|| Error::Build("global Newton requires material contact".into()))?;
            if coupling.coupled_patch
                || self.contact_fences.is_some()
                || config.solve.relative_tolerance != 0.0
                || config.solve.absolute_tolerance
                    > coupling
                        .config
                        .linear_tolerance
                        .min(coupling.config.angular_tolerance)
            {
                return Err(Error::Build(
                    "global Newton requires matching absolute gates and no patch/contact trace"
                        .into(),
                ));
            }
            Some(GlobalNewtonState::new(
                &self.device,
                &self.queue,
                &self.soft_model,
                self.rigid_model.body_count,
                self.rigid_model.joint_dof_total,
                self.gather.row_count(),
                config,
            )?)
        } else {
            None
        };
        self.global_newton = state;
        self.timing = None;
        self.fem.clear_material_contact_patch_gpu();
        Ok(())
    }

    /// Last global solve's CoupledNewtonDiagnostics; consume after device completion.
    pub fn global_newton_diagnostics(&self) -> Option<&wgpu::Buffer> {
        self.global_newton
            .as_ref()
            .and_then(GlobalNewtonState::diagnostics)
    }

    /// Final negative mixed physical residual at the last accepted Newton candidate.
    pub fn global_newton_residual_rhs(&self) -> Option<&wgpu::Buffer> {
        self.global_newton
            .as_ref()
            .and_then(GlobalNewtonState::residual_rhs)
    }

    /// Up to 64 global Newton iteration records; consume after device completion.
    pub fn global_newton_iteration_history(&self) -> Option<&wgpu::Buffer> {
        self.global_newton
            .as_ref()
            .and_then(GlobalNewtonState::iteration_history)
    }

    /// Sticky status used by the last global candidate/impulse commit.
    pub fn global_newton_commit_status(&self) -> Option<&wgpu::Buffer> {
        self.global_newton
            .as_ref()
            .map(GlobalNewtonState::commit_status)
    }

    /// Enable GPU phase timestamps while idle; material/contact/global reconfiguration disables timing.
    pub fn configure_gpu_timing(&mut self, enabled: bool) -> Result<()> {
        if self.queued_substeps != 0 || self.completion.is_some() {
            return Err(Error::Build(
                "FEM timing configuration requires an idle island".into(),
            ));
        }
        let iterations = if self.global_newton.is_some() {
            1
        } else {
            self.material_contact
                .as_ref()
                .map_or(1, |c| c.config.iterations)
        };
        self.timing = enabled
            .then(|| timing::SubstepTiming::new(&self.device, iterations))
            .transpose()?;
        Ok(())
    }

    /// Last enqueued substep's 2*N+3 u64 timestamps; wait for completion before reading.
    /// Boundaries: start, initial material, rigid/gather, first contact, alternating material/contact, final drift.
    /// Differences include queue idle time; multiply by the queue timestamp period for nanoseconds.
    pub fn gpu_timing_ticks(&self) -> Option<&wgpu::Buffer> {
        self.timing.as_ref().and_then(|t| t.ticks())
    }

    /// Last enqueued substep's N pairs of u64 contact timestamps, after predictor copies and after recovery.
    /// The first pair includes contact preparation; later pairs include only continued solve/recovery.
    /// Wait for completion before reading and multiply differences by the queue timestamp period.
    pub fn gpu_contact_timing_ticks(&self) -> Option<&wgpu::Buffer> {
        self.timing.as_ref().and_then(|t| t.contact_ticks())
    }

    /// Active count, ordered candidate row indices and compaction scratch; read after completion.
    pub fn active_contact_rows(&self) -> &wgpu::Buffer {
        self.solve.active_rows()
    }

    /// Select retained material iterates and their per-correction sweep count; None uses full restarts.
    pub fn configure_material_contact_reuse(&mut self, iterations: Option<u32>) -> Result<()> {
        if self.queued_substeps != 0
            || self.completion.is_some()
            || iterations.is_some_and(|n| !(1..=4096).contains(&n))
        {
            return Err(Error::Build(
                "invalid or pending material reuse configuration".into(),
            ));
        }
        let coupling = self.material_contact.as_mut().ok_or_else(|| {
            Error::Build("material reuse requires configured material contact".into())
        })?;
        if iterations.is_none() && coupling.polish_iterations.is_some() {
            return Err(Error::Build(
                "disable material polish before retained iterates".into(),
            ));
        }
        coupling.reuse_iterations = iterations;
        Ok(())
    }

    /// Select contact sweeps after each material correction; None uses the initial solve count.
    pub fn configure_material_contact_sweeps(&mut self, sweeps: Option<u32>) -> Result<()> {
        if self.queued_substeps != 0
            || self.completion.is_some()
            || sweeps.is_some_and(|n| !(1..=4096).contains(&n))
        {
            return Err(Error::Build(
                "invalid or pending material contact sweep configuration".into(),
            ));
        }
        let coupling = self.material_contact.as_mut().ok_or_else(|| {
            Error::Build("contact sweep configuration requires material contact".into())
        })?;
        coupling.contact_sweeps = sweeps;
        Ok(())
    }

    /// Select final retained material sweeps with fresh contact, force and velocity-change checks.
    pub fn configure_material_contact_polish(&mut self, iterations: Option<u32>) -> Result<()> {
        if self.queued_substeps != 0
            || self.completion.is_some()
            || iterations.is_some_and(|n| !(1..=4096).contains(&n))
        {
            return Err(Error::Build(
                "invalid or pending material polish configuration".into(),
            ));
        }
        let coupling = self.material_contact.as_mut().ok_or_else(|| {
            Error::Build("material polish requires configured material contact".into())
        })?;
        if iterations.is_some() && coupling.reuse_iterations.is_none() {
            return Err(Error::Build(
                "material polish requires retained iterates".into(),
            ));
        }
        coupling.polish_iterations = iterations;
        coupling.polish_cycles = 1;
        coupling.polish_contact_sweeps = None;
        coupling.coupled_patch = false;
        self.fem.clear_material_contact_patch_gpu();
        Ok(())
    }

    /// Enables one GPU-selected coupled point/edge patch during final material polish.
    pub fn configure_material_contact_patch(&mut self, enabled: bool) -> Result<()> {
        if self.queued_substeps != 0
            || self.completion.is_some()
            || (enabled && self.global_newton.is_some())
        {
            return Err(Error::Build(
                "material contact patch configuration requires an idle island".into(),
            ));
        }
        let coupling = self.material_contact.as_mut().ok_or_else(|| {
            Error::Build("material contact patch requires material coupling".into())
        })?;
        if enabled
            && (coupling.polish_iterations.is_none()
                || coupling.polish_cycles < 2
                || coupling.reuse_iterations.is_none()
                || !self.solve.has_contact_blocks()
                || self.soft_model.shape_count != 0
                || !self.soft_model.vertex_cable_bending_entries.is_empty()
                || self.device.limits().max_bind_groups < 7
                || self.device.limits().max_storage_buffers_per_shader_stage < 51)
        {
            return Err(Error::Build("material contact patch requires retained elastic polish, blocked contacts and 7 groups/51 buffers".into()));
        }
        coupling.coupled_patch = enabled;
        self.fem.clear_material_contact_patch_gpu();
        Ok(())
    }

    /// Last selected local patch's solution, residual, counters and application flag.
    pub fn material_contact_patch_diagnostics(&self) -> Option<&wgpu::Buffer> {
        if !self.material_contact.as_ref()?.coupled_patch {
            return None;
        }
        self.fem.material_patch_diagnostics_gpu()
    }

    /// Select contact sweeps between final material passes; None inherits the continuation count.
    pub fn configure_material_contact_polish_contact_sweeps(
        &mut self,
        sweeps: Option<u32>,
    ) -> Result<()> {
        if self.queued_substeps != 0
            || self.completion.is_some()
            || sweeps.is_some_and(|count| !(1..=4096).contains(&count))
        {
            return Err(Error::Build(
                "invalid or pending polish contact sweep configuration".into(),
            ));
        }
        let coupling = self.material_contact.as_mut().ok_or_else(|| {
            Error::Build("polish contact sweeps require configured material contact".into())
        })?;
        if coupling.polish_iterations.is_none() {
            return Err(Error::Build(
                "enable material polish before its contact sweeps".into(),
            ));
        }
        coupling.polish_contact_sweeps = sweeps;
        Ok(())
    }

    /// Select final material passes interleaved with retained contact corrections; default one.
    pub fn configure_material_contact_polish_cycles(&mut self, cycles: u32) -> Result<()> {
        if self.queued_substeps != 0 || self.completion.is_some() || !(1..=4096).contains(&cycles) {
            return Err(Error::Build(
                "invalid or pending material polish cycles".into(),
            ));
        }
        let coupling = self.material_contact.as_mut().ok_or_else(|| {
            Error::Build("material polish cycles require configured contact".into())
        })?;
        if coupling.polish_iterations.is_none() {
            return Err(Error::Build(
                "enable material polish before its cycles".into(),
            ));
        }
        if coupling.coupled_patch && cycles < 2 {
            return Err(Error::Build(
                "coupled material patch requires at least two polish cycles".into(),
            ));
        }
        coupling.polish_cycles = cycles;
        Ok(())
    }

    /// Last soft linear, rigid linear and rigid angular contact or material-polish corrections.
    pub fn material_contact_diagnostics(&self) -> Option<&wgpu::Buffer> {
        if self.global_newton.is_some() {
            return None;
        }
        self.material_contact
            .as_ref()
            .map(|state| state.kernel.diagnostics())
    }

    /// Pre-polish sticky status u32 followed by three contact-correction f32 values.
    /// Cleared at substep start; consume after device completion with polish enabled.
    pub fn material_pre_polish_diagnostics(&self) -> Option<&wgpu::Buffer> {
        if self.global_newton.is_some() {
            return None;
        }
        let state = self.material_contact.as_ref()?;
        state.polish_iterations?;
        Some(&state.pre_polish)
    }

    /// Eight f32 values per contact: normal inverse mass, tangent bound, target, residual,
    /// three impulses and padding. Ball rows start with three effective inverse masses.
    pub fn contact_row_diagnostics(&self) -> &wgpu::Buffer {
        self.solve.row_state()
    }

    /// Maximum h M^-1 force residual at the final coupled candidate, as one f32.
    pub fn material_force_diagnostics(&self) -> Option<&wgpu::Buffer> {
        self.material_contact
            .as_ref()
            .map(|state| state.kernel.force_diagnostics())
    }

    /// Last enqueued coupled force samples: inertial target xyz and velocity-equivalent residual w.
    /// Device completion must be checked before consuming the buffer.
    pub fn material_force_samples(&self) -> Option<&wgpu::Buffer> {
        self.material_contact.as_ref()?;
        self.fem.material_force_samples_gpu()
    }

    /// Device stiffness table used by the configured material/contact solver.
    pub fn material_spring_stiffness(&self) -> Option<&wgpu::Buffer> {
        self.material_contact.as_ref()?;
        self.fem.spring_stiffness_gpu()
    }
    pub fn rigid_state(&self) -> &State {
        &self.rigid_state
    }
    /// Allocated contact capacity plus persistent bilateral loop rows.
    pub fn contact_row_count(&self) -> u32 {
        self.gather.row_count()
    }

    /// Sticky flags: 1 mass, 2 contact, 4 nonfinite, 8 volume, 16 capacity, 32 scalar, 64 material/contact convergence.
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
            self.momentum_fences = Some(Arc::new(std::array::from_fn(|_| AtomicU64::new(0))));
            self.contact_fences = Some(Arc::new(std::array::from_fn(|_| AtomicU64::new(0))));
        }
    }

    /// Queue-completed counts for diagnostic phases; not a physical-validity check or state readback.
    pub fn completed_momentum_phases(&self) -> Option<[u64; 5]> {
        self.momentum_fences
            .as_ref()
            .map(|counts| std::array::from_fn(|phase| counts[phase].load(Ordering::Acquire)))
    }

    /// Completed factor, prepare, count, offset, scatter, solve and recovery passes in diagnostic mode.
    pub fn completed_contact_passes(&self) -> Option<[u64; 7]> {
        self.contact_fences
            .as_ref()
            .map(|counts| std::array::from_fn(|pass| counts[pass].load(Ordering::Acquire)))
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
        let counts = self.momentum_fences.as_ref().unwrap().clone();
        self.queue.on_submitted_work_done(move || {
            counts[phase as usize].fetch_add(1, Ordering::Release);
        });
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
            if let Some(snapshot) = &mut self.contact_snapshot {
                snapshot.settings = None;
            }
            self.record_momentum(0)?;
            if self.material_contact.is_some() || self.timing.is_some() {
                let mut encoder = device.create_command_encoder(&Default::default());
                if let Some(coupling) = &mut self.material_contact {
                    coupling.begin(&mut encoder, &self.soft_state);
                }
                if let Some(timing) = &mut self.timing {
                    timing.ready = false;
                    timing.mark(&mut encoder, 0);
                }
                queue.submit([encoder.finish()]);
            }
            if let Some(coupling) = self
                .material_contact
                .as_ref()
                .filter(|c| c.reuse_iterations.is_some())
            {
                self.fem.material_iteration_gpu(
                    &self.soft_model,
                    &coupling.input,
                    &mut self.soft_next,
                    h,
                    self.config.elastic_iterations as u32,
                    true,
                )?;
            } else {
                self.fem.step(
                    &self.soft_model,
                    &self.soft_state,
                    &self.soft_model.control(),
                    &Contacts::with_capacity(0),
                    h,
                    &mut self.soft_next,
                )?;
            }
            self.record_momentum(1)?;
            if let Some(timing) = &self.timing {
                let mut encoder = device.create_command_encoder(&Default::default());
                timing.mark(&mut encoder, 1);
                queue.submit([encoder.finish()]);
            }
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
            if let Some(bounds) = &self.bounds {
                bounds.encode(
                    device,
                    &mut encoder,
                    self.rigid_state.joint_q.device_buffer().unwrap(),
                    self.gather.rows(),
                    self.solve.status(),
                    self.gather.row_count() - bounds.row_count(),
                    h as f32,
                )?;
            }
            let soft_velocity_buffer = self.soft_next.particle_qd.device_buffer().unwrap().clone();
            let rigid_velocity_buffer = self.rigid_next.joint_qd.device_buffer().unwrap().clone();
            let body_velocity_buffer = self.rigid_next.body_qd.device_buffer().unwrap().clone();
            let solve_bindings = SoftRigidSolveBindings {
                positions: self.soft_state.particle_q.device_buffer().unwrap(),
                velocities: &soft_velocity_buffer,
                inverse_mass: self.soft_model.particle_inv_mass.device_buffer().unwrap(),
                contacts: self.gather.rows(),
                body_poses: self.rigid_state.body_q.device_buffer().unwrap(),
                body_com: self.rigid_model.body_com.device_buffer().unwrap(),
                jacobians: &self.jacobian_buffer,
                mass_matrix: &self.mass_buffer,
                rigid_velocities: &rigid_velocity_buffer,
                body_velocities: &body_velocity_buffer,
                particle_count: self.soft_model.particle_count as u32,
                body_count: self.rigid_model.body_count as u32,
            };
            let solve_settings = SoftRigidSolveSettings {
                iterations: self.config.contact.iterations as u32,
                dt: h as f32,
                penetration_recovery: self.config.contact.penetration_recovery as f32,
                max_recovery_speed: self.config.contact.max_recovery_speed as f32,
            };
            if let Some(timing) = &self.timing {
                timing.mark(&mut encoder, 2);
            }
            if let Some(coupling) = &self.material_contact {
                coupling.capture(&mut encoder, &self.soft_next, &self.rigid_next);
            }
            if let Some(snapshot) = &mut self.contact_snapshot {
                snapshot.capture(&mut encoder, solve_bindings, solve_settings);
            }
            if let Some(timing) = &self.timing {
                timing.mark_contact(&mut encoder, 0);
            }
            if let Some(global) = &mut self.global_newton {
                let coupling = self.material_contact.as_mut().unwrap();
                global.run(
                    device,
                    queue,
                    &mut encoder,
                    &mut self.fem,
                    &self.soft_model,
                    &mut coupling.input,
                    &self.soft_next,
                    &self.solve,
                    solve_bindings,
                    solve_settings,
                    h,
                )?;
                self.solve.encode_velocity_recovery(
                    device,
                    &mut encoder,
                    solve_bindings,
                    solve_settings,
                )?;
                self.solve.encode_residual_check(
                    device,
                    &mut encoder,
                    solve_bindings,
                    solve_settings,
                    coupling.config.linear_tolerance,
                )?;
                queue.submit([encoder.finish()]);
                let residual = self.fem.material_force_residual_gpu(
                    &self.soft_model,
                    &coupling.input,
                    &mut self.soft_next,
                    h,
                )?;
                encoder = device.create_command_encoder(&Default::default());
                coupling.kernel.encode_force_check(
                    device,
                    &mut encoder,
                    residual,
                    self.solve.status(),
                    self.soft_model.particle_count as u32,
                    coupling.config.linear_tolerance,
                )?;
                if let Some(timing) = &self.timing {
                    timing.mark_contact(&mut encoder, 1);
                    timing.mark(&mut encoder, 3);
                }
            } else {
                if let Some(counts) = &self.contact_fences {
                    let counts = counts.clone();
                    self.solve.encode_with_completion(
                        device,
                        queue,
                        &mut encoder,
                        solve_bindings,
                        solve_settings,
                        Arc::new(move |pass| {
                            counts[pass].fetch_add(1, Ordering::Release);
                        }),
                    )?;
                } else {
                    self.solve
                        .encode(device, &mut encoder, solve_bindings, solve_settings)?;
                }
                if let Some(timing) = &self.timing {
                    timing.mark_contact(&mut encoder, 1);
                }
                if self.bounds.is_some() && self.material_contact.is_none() {
                    self.solve.encode_residual_check(
                        device,
                        &mut encoder,
                        solve_bindings,
                        solve_settings,
                        1e-5,
                    )?;
                }
                if let Some(coupling) = &mut self.material_contact {
                    coupling.accumulate(
                        device,
                        &mut encoder,
                        &self.soft_next,
                        &self.rigid_next,
                        self.solve.status(),
                        false,
                    )?;
                    if let Some(timing) = &self.timing {
                        timing.mark(&mut encoder, 3);
                    }
                    queue.submit([encoder.finish()]);
                    for iteration in 1..coupling.config.iterations {
                        if let Some(iterations) = coupling.reuse_iterations {
                            self.fem.material_iteration_gpu(
                                &self.soft_model,
                                &coupling.input,
                                &mut self.soft_next,
                                h,
                                iterations,
                                false,
                            )?;
                        } else {
                            self.fem.step(
                                &self.soft_model,
                                &coupling.input,
                                &self.soft_model.control(),
                                &Contacts::with_capacity(0),
                                h,
                                &mut self.soft_next,
                            )?;
                        }
                        let mut pass = device.create_command_encoder(&Default::default());
                        if let Some(timing) = &self.timing {
                            timing.mark(&mut pass, 2 * iteration + 2);
                        }
                        coupling.capture(&mut pass, &self.soft_next, &self.rigid_next);
                        if let Some(timing) = &self.timing {
                            timing.mark_contact(&mut pass, 2 * iteration);
                        }
                        let bindings = SoftRigidSolveBindings {
                            positions: self.soft_state.particle_q.device_buffer().unwrap(),
                            velocities: self.soft_next.particle_qd.device_buffer().unwrap(),
                            inverse_mass: self
                                .soft_model
                                .particle_inv_mass
                                .device_buffer()
                                .unwrap(),
                            contacts: self.gather.rows(),
                            body_poses: self.rigid_state.body_q.device_buffer().unwrap(),
                            body_com: self.rigid_model.body_com.device_buffer().unwrap(),
                            jacobians: &self.jacobian_buffer,
                            mass_matrix: &self.mass_buffer,
                            rigid_velocities: self.rigid_next.joint_qd.device_buffer().unwrap(),
                            body_velocities: self.rigid_next.body_qd.device_buffer().unwrap(),
                            particle_count: self.soft_model.particle_count as u32,
                            body_count: self.rigid_model.body_count as u32,
                        };
                        let continuation_settings = SoftRigidSolveSettings {
                            iterations: coupling
                                .contact_sweeps
                                .unwrap_or(solve_settings.iterations),
                            ..solve_settings
                        };
                        self.solve.encode_continuation(
                            device,
                            &mut pass,
                            bindings,
                            continuation_settings,
                        )?;
                        if let Some(timing) = &self.timing {
                            timing.mark_contact(&mut pass, 2 * iteration + 1);
                        }
                        let final_correction = iteration + 1 == coupling.config.iterations
                            && !(coupling.polish_iterations.is_some()
                                && coupling.polish_cycles > 1);
                        if final_correction {
                            self.solve.encode_residual_check(
                                device,
                                &mut pass,
                                bindings,
                                continuation_settings,
                                coupling.config.linear_tolerance,
                            )?;
                        }
                        coupling.accumulate(
                            device,
                            &mut pass,
                            &self.soft_next,
                            &self.rigid_next,
                            self.solve.status(),
                            final_correction,
                        )?;
                        if let Some(timing) = &self.timing {
                            timing.mark(&mut pass, 2 * iteration + 3);
                        }
                        queue.submit([pass.finish()]);
                    }
                    encoder = device.create_command_encoder(&Default::default());
                    if let Some(iterations) = coupling.polish_iterations {
                        if coupling.coupled_patch {
                            self.solve.encode_residual_check(
                                device,
                                &mut encoder,
                                solve_bindings,
                                solve_settings,
                                f32::MAX,
                            )?;
                            self.fem.prepare_material_contact_patch_gpu(
                                molla_kernels::MaterialContactPatchBindings {
                                    rows: self.solve.row_state(),
                                    projected: self.solve.projected_rows(),
                                    response: self.solve.row_responses(),
                                    rigid_velocity: self
                                        .rigid_next
                                        .joint_qd
                                        .device_buffer()
                                        .unwrap(),
                                    contacts: self.gather.rows(),
                                    schedule: self.solve.active_rows(),
                                    input_velocity: coupling
                                        .input
                                        .particle_qd
                                        .device_buffer()
                                        .unwrap(),
                                    inverse_mass: self
                                        .soft_model
                                        .particle_inv_mass
                                        .device_buffer()
                                        .unwrap(),
                                },
                            )?;
                        }
                        for cycle in 0..coupling.polish_cycles {
                            let last_cycle = cycle + 1 == coupling.polish_cycles;
                            if cycle > 0 {
                                coupling.capture(&mut encoder, &self.soft_next, &self.rigid_next);
                                let bindings = SoftRigidSolveBindings {
                                    positions: self.soft_state.particle_q.device_buffer().unwrap(),
                                    velocities: self.soft_next.particle_qd.device_buffer().unwrap(),
                                    inverse_mass: self
                                        .soft_model
                                        .particle_inv_mass
                                        .device_buffer()
                                        .unwrap(),
                                    contacts: self.gather.rows(),
                                    body_poses: self.rigid_state.body_q.device_buffer().unwrap(),
                                    body_com: self.rigid_model.body_com.device_buffer().unwrap(),
                                    jacobians: &self.jacobian_buffer,
                                    mass_matrix: &self.mass_buffer,
                                    rigid_velocities: self
                                        .rigid_next
                                        .joint_qd
                                        .device_buffer()
                                        .unwrap(),
                                    body_velocities: self
                                        .rigid_next
                                        .body_qd
                                        .device_buffer()
                                        .unwrap(),
                                    particle_count: self.soft_model.particle_count as u32,
                                    body_count: self.rigid_model.body_count as u32,
                                };
                                let settings = SoftRigidSolveSettings {
                                    iterations: coupling
                                        .polish_contact_sweeps
                                        .or(coupling.contact_sweeps)
                                        .unwrap_or(solve_settings.iterations),
                                    ..solve_settings
                                };
                                if coupling.coupled_patch {
                                    self.fem
                                        .encode_material_patch_contacts_gpu(&mut encoder, false);
                                }
                                self.solve.encode_continuation(
                                    device,
                                    &mut encoder,
                                    bindings,
                                    settings,
                                )?;
                                if coupling.coupled_patch {
                                    self.fem
                                        .encode_material_patch_contacts_gpu(&mut encoder, true);
                                }
                                if last_cycle {
                                    self.solve.encode_residual_check(
                                        device,
                                        &mut encoder,
                                        bindings,
                                        settings,
                                        coupling.config.linear_tolerance,
                                    )?;
                                }
                                coupling.accumulate(
                                    device,
                                    &mut encoder,
                                    &self.soft_next,
                                    &self.rigid_next,
                                    self.solve.status(),
                                    last_cycle,
                                )?;
                            }
                            encoder.copy_buffer_to_buffer(
                                self.solve.status(),
                                0,
                                &coupling.pre_polish,
                                0,
                                4,
                            );
                            encoder.copy_buffer_to_buffer(
                                coupling.kernel.diagnostics(),
                                0,
                                &coupling.pre_polish,
                                4,
                                12,
                            );
                            coupling.capture(&mut encoder, &self.soft_next, &self.rigid_next);
                            queue.submit([encoder.finish()]);
                            self.fem.material_polish_gpu(
                                &self.soft_model,
                                &coupling.input,
                                &mut self.soft_next,
                                h,
                                iterations,
                            )?;
                            encoder = device.create_command_encoder(&Default::default());
                            if coupling.coupled_patch {
                                coupling.input.particle_qd.mark_device_dirty();
                                self.rigid_next.joint_qd.mark_device_dirty();
                                self.solve.encode_velocity_recovery(
                                    device,
                                    &mut encoder,
                                    solve_bindings,
                                    solve_settings,
                                )?;
                            }
                            if last_cycle {
                                coupling.check_polish(
                                    device,
                                    &mut encoder,
                                    &self.soft_next,
                                    &self.rigid_next,
                                    self.solve.status(),
                                )?;
                                self.solve.encode_residual_check(
                                    device,
                                    &mut encoder,
                                    SoftRigidSolveBindings {
                                        positions: self
                                            .soft_state
                                            .particle_q
                                            .device_buffer()
                                            .unwrap(),
                                        velocities: self
                                            .soft_next
                                            .particle_qd
                                            .device_buffer()
                                            .unwrap(),
                                        inverse_mass: self
                                            .soft_model
                                            .particle_inv_mass
                                            .device_buffer()
                                            .unwrap(),
                                        contacts: self.gather.rows(),
                                        body_poses: self
                                            .rigid_state
                                            .body_q
                                            .device_buffer()
                                            .unwrap(),
                                        body_com: self
                                            .rigid_model
                                            .body_com
                                            .device_buffer()
                                            .unwrap(),
                                        jacobians: &self.jacobian_buffer,
                                        mass_matrix: &self.mass_buffer,
                                        rigid_velocities: self
                                            .rigid_next
                                            .joint_qd
                                            .device_buffer()
                                            .unwrap(),
                                        body_velocities: self
                                            .rigid_next
                                            .body_qd
                                            .device_buffer()
                                            .unwrap(),
                                        particle_count: self.soft_model.particle_count as u32,
                                        body_count: self.rigid_model.body_count as u32,
                                    },
                                    solve_settings,
                                    coupling.config.linear_tolerance,
                                )?;
                            }
                            queue.submit([encoder.finish()]);
                            encoder = device.create_command_encoder(&Default::default());
                        }
                    }
                    let residual = self.fem.material_force_residual_gpu(
                        &self.soft_model,
                        &coupling.input,
                        &mut self.soft_next,
                        h,
                    )?;
                    coupling.kernel.encode_force_check(
                        device,
                        &mut encoder,
                        residual,
                        self.solve.status(),
                        self.soft_model.particle_count as u32,
                        coupling.config.linear_tolerance,
                    )?;
                } else if let Some(timing) = &self.timing {
                    timing.mark(&mut encoder, 3);
                }
            }
            if let Some(snapshot) = &self.contact_snapshot {
                snapshot.capture_solved(
                    &mut encoder,
                    self.soft_next.particle_qd.device_buffer().unwrap(),
                    self.rigid_next.joint_qd.device_buffer().unwrap(),
                );
            }
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
            if let Some(timing) = &mut self.timing {
                timing.mark(&mut encoder, 2 * timing.iterations + 2);
                timing.resolve(&mut encoder);
            }
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
