#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder};
use molla_solvers::fem_rigid::FemRigidConfig;
use molla_solvers::fem_rigid_gpu::{FemRigidGpuScene, FemRigidGpuSystem};
use std::sync::Arc;

#[path = "common/gpu_readback.rs"]
mod gpu_readback;

fn system(engine: Arc<Engine>) -> FemRigidGpuSystem {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.set_linear_damping(0.0);
    for point in [Vec3::ZERO, Vec3::X * 0.1, Vec3::Y * 0.1, Vec3::Z * 0.1] {
        builder.add_particle(point, 0.1, 0.0, WorldId(0));
    }
    builder
        .add_tet_mesh(&[[0, 1, 2, 3]], 2e6, 0.45, None)
        .unwrap();
    let soft_model = builder.build().unwrap();
    let soft_state = soft_model.state().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.begin_articulation();
    let rotor = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: Vec3::ZERO,
        inertia: Mat3::from_diagonal(Vec3::splat(0.25)),
    }));
    builder
        .add_joint_revolute(
            BodyId::NONE,
            rotor,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::X,
        )
        .unwrap();
    let mut rigid_model = builder.build().unwrap();
    rigid_model.joint_effort_limit.host_mut().unwrap()[0] = 1.0;
    let rigid_state = rigid_model.state().unwrap();
    let control = rigid_model.control();
    let mut system = FemRigidGpuSystem::new(
        engine,
        FemRigidGpuScene {
            soft_model,
            soft_state,
            rigid_model,
            rigid_state,
            control,
            shapes: Vec::new(),
        },
        FemRigidConfig {
            max_substep: 1.0 / 4096.0,
            elastic_iterations: 4,
            ..Default::default()
        },
    )
    .unwrap();
    system.set_drive_effort(0, 0.5).unwrap();
    system
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn submission_windows_preserve_state_across_step_calls() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut one_call = system(engine.clone());
    let mut split = system(engine.clone());
    let h = 1.0 / 4096.0;
    assert_eq!(one_call.queued_substeps(), 0);
    one_call.step(65.0 * h).unwrap();
    assert_eq!(one_call.queued_substeps(), 1);
    split.step(31.0 * h).unwrap();
    assert_eq!(split.queued_substeps(), 31);
    split.step(h).unwrap();
    assert_eq!(split.queued_substeps(), 0);
    split.step(33.0 * h).unwrap();
    assert_eq!(split.queued_substeps(), 1);
    assert!(split.step(0.0).is_err());
    assert_eq!(split.queued_substeps(), 1);
    for system in [&one_call, &split] {
        assert!(system.soft_state().particle_q.host().is_err());
        assert!(system.soft_state().particle_qd.host().is_err());
        assert_eq!(
            gpu_readback::read::<u32>(&engine, system.status_buffer(), 1),
            [0]
        );
    }
    let snapshot = |system: &FemRigidGpuSystem| {
        (
            gpu_readback::read::<[f32; 4]>(
                &engine,
                system.soft_state().particle_q.device_buffer().unwrap(),
                4,
            ),
            gpu_readback::read::<[f32; 4]>(
                &engine,
                system.soft_state().particle_qd.device_buffer().unwrap(),
                4,
            ),
            gpu_readback::read::<f32>(
                &engine,
                system.rigid_state().joint_q.device_buffer().unwrap(),
                1,
            ),
            gpu_readback::read::<f32>(
                &engine,
                system.rigid_state().joint_qd.device_buffer().unwrap(),
                1,
            ),
            gpu_readback::body_poses(&engine, system, 1),
        )
    };
    let a = snapshot(&one_call);
    let b = snapshot(&split);
    assert_eq!(a, b);
    assert!((a.3[0] as f64 - 2.0 * 65.0 * h).abs() < 1e-6);
}
