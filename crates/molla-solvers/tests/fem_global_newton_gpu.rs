#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_math::Vec3;
use molla_sim::{MaterialNeoHookean, ModelBuilder, TetViscosity};
use molla_solvers::fem_rigid::FemRigidConfig;
use molla_solvers::fem_rigid_gpu::{
    FemRigidGpuScene, FemRigidGpuSystem, GlobalNewtonConfig, MaterialContactConfig,
    SoftRigidShapeGpu,
};

#[path = "common/fem_rotor.rs"]
#[allow(dead_code)]
mod fem_rotor;
#[path = "common/material_patch_oracle.rs"]
mod oracle;
#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn scene() -> FemRigidGpuScene {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.set_linear_damping(0.0);
    for p in [
        Vec3::new(-0.01, -1e-5, 0.1),
        Vec3::new(0.01, 1e-5, 0.1),
        Vec3::new(0.0, 0.01, 0.12),
        Vec3::new(0.0, -0.01, 0.12),
    ] {
        builder.add_particle(p, 1.0 / 3000.0, 0.0, WorldId(0));
    }
    let mut material = MaterialNeoHookean::elastic(2e6, 0.45);
    material.viscosity = TetViscosity {
        shear: 1379.3103448275863,
        bulk: 13333.333333333336,
    };
    builder
        .add_tet_mesh_per_material(&[[0, 1, 2, 3]], &[material], &[0], None)
        .unwrap();
    let mut scene = fem_rotor::scene();
    scene.soft_model = builder.build().unwrap();
    scene.soft_state = scene.soft_model.state().unwrap();
    scene
        .soft_state
        .particle_qd
        .host_mut()
        .unwrap()
        .fill(Vec3::new(0.01, -0.02, 0.01));
    scene.shapes.push(SoftRigidShapeGpu {
        position: [0.0, -0.1, 0.1, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        data: [0.02, 0.1, 0.002, 0.6],
        ids: [1, 0, 0, 0],
    });
    scene
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn native_global_newton_preserves_material_and_contact_gates() {
    let engine = Engine::new_gpu().unwrap();
    let h = 1.0 / 9600.0;
    let scene = scene();
    let reference = oracle::Reference::new(&scene.soft_model);
    let mut system = FemRigidGpuSystem::new_surface_sampled_with_contact_blocks_on_device(
        engine.device().unwrap().clone(),
        engine.queue().unwrap().clone(),
        scene,
        FemRigidConfig {
            max_substep: h,
            elastic_iterations: 128,
            ..Default::default()
        },
        &[],
        None,
    )
    .unwrap();
    system.configure_contact_snapshot(true).unwrap();
    system
        .configure_material_contact(Some(MaterialContactConfig {
            iterations: 2,
            linear_tolerance: 1e-4,
            angular_tolerance: 1e-4,
        }))
        .unwrap();
    system.configure_material_contact_reuse(Some(8)).unwrap();
    system
        .configure_global_newton(Some(GlobalNewtonConfig {
            contact_capacity: 128,
            ..Default::default()
        }))
        .unwrap();
    for step in 1..=3 {
        let prior_velocity = readback::read::<[f32; 4]>(
            &engine,
            system.soft_state().particle_qd.device_buffer().unwrap(),
            4,
        );
        assert!(system.try_step_substep(h).unwrap());
        assert!(system.configure_global_newton(None).is_err());
        let status = readback::read::<u32>(&engine, system.status_buffer(), 1)[0];
        let diagnostics = readback::read::<molla_kernels::soft_rigid_solve::CoupledNewtonDiagnostics>(
            &engine,
            system.global_newton_diagnostics().unwrap(),
            1,
        )[0];
        let commit =
            readback::read::<u32>(&engine, system.global_newton_commit_status().unwrap(), 1)[0];
        let material =
            readback::read::<f32>(&engine, system.material_force_diagnostics().unwrap(), 1)[0];
        let contacts = readback::read::<[f32; 8]>(
            &engine,
            system.contact_row_diagnostics(),
            system.contact_row_count() as usize,
        );
        let max_contact = contacts.iter().map(|row| row[3]).fold(0.0_f32, f32::max);
        println!(
            "NATIVE GLOBAL step={step} status={status} commit={commit} material={material} contact={max_contact} newton={diagnostics:?}"
        );
        assert_eq!(status, 0);
        assert_eq!(commit, 0);
        assert_eq!(diagnostics.faults, 0);
        assert_eq!(diagnostics.converged, 1);
        assert!(material <= 1e-4 && max_contact <= 1e-4);
        reference.check(&engine, &system, &prior_velocity);
        assert!(system.poll_completion().unwrap());
    }
    system.configure_global_newton(None).unwrap();
    assert!(system.global_newton_diagnostics().is_none());
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn native_global_rejects_capacity_and_newton_exhaustion() {
    let engine = Engine::new_gpu().unwrap();
    let h = 1.0 / 9600.0;
    for overflow in [false, true] {
        let mut system = FemRigidGpuSystem::new_surface_sampled_with_contact_blocks_on_device(
            engine.device().unwrap().clone(),
            engine.queue().unwrap().clone(),
            scene(),
            FemRigidConfig {
                max_substep: h,
                elastic_iterations: 128,
                ..Default::default()
            },
            &[],
            None,
        )
        .unwrap();
        system.configure_contact_snapshot(true).unwrap();
        system
            .configure_material_contact(Some(MaterialContactConfig {
                iterations: 2,
                linear_tolerance: 1e-4,
                angular_tolerance: 1e-4,
            }))
            .unwrap();
        system.configure_material_contact_reuse(Some(8)).unwrap();
        let mut config = GlobalNewtonConfig::default();
        config.contact_capacity = if overflow { 1 } else { 128 };
        config.restart = 16;
        config.solve.iterations = 1;
        system.configure_global_newton(Some(config)).unwrap();
        assert!(system.try_step_substep(h).unwrap());
        let status = readback::read::<u32>(&engine, system.status_buffer(), 1)[0];
        let commit =
            readback::read::<u32>(&engine, system.global_newton_commit_status().unwrap(), 1)[0];
        let required = if overflow { 16 } else { 64 };
        println!("GLOBAL REJECT overflow={overflow} status={status} commit={commit}");
        assert_ne!(status & required, 0);
        assert_ne!(commit & required, 0);
        let snapshot = system.contact_snapshot().unwrap();
        let buffer = |name| {
            snapshot
                .buffers()
                .chain(snapshot.solved_buffers())
                .find(|(key, _)| *key == name)
                .unwrap()
                .1
        };
        assert_eq!(
            readback::read::<[f32; 4]>(&engine, buffer("velocities"), 4),
            readback::read::<[f32; 4]>(&engine, buffer("solved_velocities"), 4),
        );
        assert_eq!(
            readback::read::<f32>(&engine, buffer("rigid_velocities"), 1),
            readback::read::<f32>(&engine, buffer("solved_rigid_velocities"), 1),
        );
        let rows = readback::read::<[f32; 8]>(
            &engine,
            system.contact_row_diagnostics(),
            system.contact_row_count() as usize,
        );
        assert!(rows.iter().all(|row| row[4..7] == [0.0; 3]));
        assert!(system.poll_completion().unwrap());
    }
}
