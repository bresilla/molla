#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_math::Vec3;
use molla_sim::{MaterialNeoHookean, ModelBuilder, TetViscosity};
use molla_solvers::fem_rigid::FemRigidConfig;
use molla_solvers::fem_rigid_gpu::{
    FemRigidGpuScene, FemRigidGpuSystem, MaterialContactConfig, SoftRigidShapeGpu,
};
use std::sync::Arc;

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
fn native_patch_retargets_material_and_preserves_rejection_gates() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
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
    assert!(system.configure_material_contact_patch(true).is_err());
    system
        .configure_material_contact(Some(MaterialContactConfig {
            iterations: 2,
            linear_tolerance: 1e-4,
            angular_tolerance: 1e-4,
        }))
        .unwrap();
    assert!(system.configure_material_contact_patch(true).is_err());
    system.configure_material_contact_reuse(Some(8)).unwrap();
    system.configure_material_contact_sweeps(Some(16)).unwrap();
    system.configure_material_contact_polish(Some(8)).unwrap();
    assert!(system.configure_material_contact_patch(true).is_err());
    system.configure_material_contact_polish_cycles(65).unwrap();
    system.configure_material_contact_patch(true).unwrap();
    assert!(system.configure_material_contact_polish_cycles(1).is_err());
    for step in 1..=3 {
        let prior_velocity = readback::read::<[f32; 4]>(
            &engine,
            system.soft_state().particle_qd.device_buffer().unwrap(),
            4,
        );
        assert!(system.try_step_substep(h).unwrap());
        assert!(system.configure_material_contact_patch(false).is_err());
        let status = readback::read::<u32>(&engine, system.status_buffer(), 1)[0];
        let patch = readback::read::<f32>(
            &engine,
            system.material_contact_patch_diagnostics().unwrap(),
            37,
        );
        let material =
            readback::read::<f32>(&engine, system.material_force_diagnostics().unwrap(), 1)[0];
        let contacts = readback::read::<[f32; 8]>(
            &engine,
            system.contact_row_diagnostics(),
            system.contact_row_count() as usize,
        );
        let max_contact = contacts.iter().map(|row| row[3]).fold(0.0_f32, f32::max);
        println!(
            "NATIVE PATCH step={step} status={status} material={material} contact={max_contact} attempts={} accepted={} last={}",
            patch[34], patch[35], patch[33]
        );
        assert_eq!(status, 0);
        assert!(patch.iter().all(|value| value.is_finite()));
        assert!(patch[34] > 0.0 && patch[35] > 0.0 && patch[36] == 1.0);
        assert!(material <= 1e-4 && max_contact <= 1e-4);
        reference.check(&engine, &system, &prior_velocity);
        assert!(system.poll_completion().unwrap());
    }
    system.configure_material_contact_patch(false).unwrap();
    assert!(system.material_contact_patch_diagnostics().is_none());
    for enabled in [false, true] {
        system.configure_material_contact_patch(enabled).unwrap();
        let prior_velocity = readback::read::<[f32; 4]>(
            &engine,
            system.soft_state().particle_qd.device_buffer().unwrap(),
            4,
        );
        assert!(system.try_step_substep(h).unwrap());
        assert_eq!(
            readback::read::<u32>(&engine, system.status_buffer(), 1),
            [0]
        );
        reference.check(&engine, &system, &prior_velocity);
        assert!(system.poll_completion().unwrap());
    }
    system.configure_material_contact(None).unwrap();
    assert!(system.configure_material_contact_patch(true).is_err());
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn native_patch_rejects_exhaustion_and_skips_absent_contacts() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let h = 1.0 / 9600.0;
    let mut absent_baseline = None;
    for (absent, enabled) in [(false, true), (true, false), (true, true)] {
        let mut scene = scene();
        if absent {
            scene.shapes[0].position[2] = -10.0;
        }
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
        system
            .configure_material_contact(Some(MaterialContactConfig {
                iterations: 2,
                linear_tolerance: 1e-4,
                angular_tolerance: 1e-4,
            }))
            .unwrap();
        system.configure_material_contact_reuse(Some(8)).unwrap();
        system.configure_material_contact_sweeps(Some(16)).unwrap();
        system.configure_material_contact_polish(Some(8)).unwrap();
        system.configure_material_contact_polish_cycles(2).unwrap();
        system.configure_material_contact_patch(enabled).unwrap();
        system.step(h).unwrap();
        let status = readback::read::<u32>(&engine, system.status_buffer(), 1)[0];
        if !absent {
            let patch = readback::read::<f32>(
                &engine,
                system.material_contact_patch_diagnostics().unwrap(),
                37,
            );
            println!(
                "EXHAUSTED native patch status={status} attempts={} accepted={}",
                patch[34], patch[35]
            );
            assert_eq!(status, 64);
            assert!(patch[34] > 0.0);
        } else {
            assert_eq!(status, 0);
            let velocity = readback::read::<[f32; 4]>(
                &engine,
                system.soft_state().particle_qd.device_buffer().unwrap(),
                4,
            );
            if enabled {
                let patch = readback::read::<f32>(
                    &engine,
                    system.material_contact_patch_diagnostics().unwrap(),
                    37,
                );
                assert_eq!(patch[34], 0.0);
                assert_eq!(patch[36], 0.0);
                assert_eq!(velocity, absent_baseline.take().unwrap());
            } else {
                absent_baseline = Some(velocity);
            }
        }
    }
}
