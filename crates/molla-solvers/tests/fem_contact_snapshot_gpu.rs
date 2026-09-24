#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_solve::{SoftRigidSolveBindings, SoftRigidSolveKernel};
use molla_math::Vec3;
use molla_solvers::fem_rigid_gpu::{FemRigidGpuSystem, SoftRigidShapeGpu};
use std::sync::Arc;

#[path = "common/fem_rotor.rs"]
mod fem_rotor;
#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn bytes(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<u8> {
    readback::read(engine, buffer, buffer.size() as usize)
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn contact_snapshot_preserves_state_replaces_inputs_and_replays_first_solve() {
    check_snapshot(false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn contact_snapshot_separates_solved_velocities_from_stop_drift() {
    check_snapshot(true);
}

fn check_snapshot(stopped: bool) {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let make = || {
        let mut scene = fem_rotor::scene();
        scene
            .soft_state
            .particle_qd
            .host_mut()
            .unwrap()
            .fill(Vec3::NEG_Y * 0.01);
        scene.shapes.push(SoftRigidShapeGpu {
            position: [0.0, -0.1, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            data: [10.0, 0.1, 10.0, 0.85],
            ids: [1, if stopped { 0 } else { u32::MAX }, 0, 0],
        });
        if stopped {
            scene.rigid_model.joint_limit_upper.host_mut().unwrap()[0] = 0.0;
        }
        let mut system =
            FemRigidGpuSystem::new(engine.clone(), scene, fem_rotor::config()).unwrap();
        system
            .set_drive_effort(0, if stopped { 0.0 } else { 0.25 })
            .unwrap();
        system
    };
    let mut plain = make();
    let mut captured = make();
    assert!(captured.contact_snapshot().is_none());
    captured.configure_contact_snapshot(true).unwrap();
    assert!(captured.contact_snapshot().is_none());
    for _ in 0..2 {
        let initial_positions = bytes(
            &engine,
            captured.soft_state().particle_q.device_buffer().unwrap(),
        );
        let initial_poses = bytes(
            &engine,
            captured.rigid_state().body_q.device_buffer().unwrap(),
        );
        plain.step(fem_rotor::config().max_substep).unwrap();
        assert!(
            captured
                .try_step_substep(fem_rotor::config().max_substep)
                .unwrap()
        );
        assert!(captured.configure_contact_snapshot(false).is_err());
        assert!(captured.configure_contact_snapshot(true).is_err());
        while !captured.poll_completion().unwrap() {
            std::thread::yield_now();
        }
        assert!(plain.contact_snapshot().is_none());
        for (a, b) in [
            (
                plain.soft_state().particle_q.device_buffer().unwrap(),
                captured.soft_state().particle_q.device_buffer().unwrap(),
            ),
            (
                plain.soft_state().particle_qd.device_buffer().unwrap(),
                captured.soft_state().particle_qd.device_buffer().unwrap(),
            ),
            (
                plain.rigid_state().body_q.device_buffer().unwrap(),
                captured.rigid_state().body_q.device_buffer().unwrap(),
            ),
            (
                plain.rigid_state().joint_qd.device_buffer().unwrap(),
                captured.rigid_state().joint_qd.device_buffer().unwrap(),
            ),
            (
                plain.rigid_state().body_qd.device_buffer().unwrap(),
                captured.rigid_state().body_qd.device_buffer().unwrap(),
            ),
            (plain.active_contact_rows(), captured.active_contact_rows()),
            (plain.status_buffer(), captured.status_buffer()),
        ] {
            assert_eq!(bytes(&engine, a), bytes(&engine, b));
        }
        assert_eq!(bytes(&engine, captured.status_buffer()), 0u32.to_le_bytes());
        let snapshot = captured.contact_snapshot().unwrap();
        let solved = snapshot.solved_buffers().collect::<Vec<_>>();
        assert_eq!(
            solved.iter().map(|v| v.0).collect::<Vec<_>>(),
            ["solved_velocities", "solved_rigid_velocities"]
        );
        assert_eq!(
            bytes(&engine, solved[0].1),
            bytes(
                &engine,
                captured.soft_state().particle_qd.device_buffer().unwrap()
            )
        );
        if stopped {
            let before = readback::read::<f32>(&engine, solved[1].1, 1)[0];
            let after = readback::read::<f32>(
                &engine,
                captured.rigid_state().joint_qd.device_buffer().unwrap(),
                1,
            )[0];
            assert!(
                before > 1e-6,
                "stop fixture did not receive contact torque: {before}"
            );
            assert_eq!(after, 0.0);
        }
        let named = snapshot.buffers().collect::<Vec<_>>();
        assert_eq!(
            named.iter().map(|v| v.0).collect::<Vec<_>>(),
            [
                "positions",
                "velocities",
                "inverse_mass",
                "contacts",
                "body_poses",
                "body_com",
                "jacobians",
                "mass_matrix",
                "rigid_velocities",
                "body_velocities",
            ]
        );
        assert_eq!(bytes(&engine, named[0].1), initial_positions);
        assert_eq!(bytes(&engine, named[4].1), initial_poses);
        let settings = snapshot.settings().unwrap();
        assert_eq!(settings.dt, fem_rotor::config().max_substep as f32);
        assert_eq!(
            settings.iterations,
            fem_rotor::config().contact.iterations as u32
        );
        assert_eq!(
            settings.penetration_recovery,
            fem_rotor::config().contact.penetration_recovery as f32
        );
        assert_eq!(
            settings.max_recovery_speed,
            fem_rotor::config().contact.max_recovery_speed as f32
        );
        let kernel = SoftRigidSolveKernel::new_serial(
            device,
            (named[8].1.size() / 4) as u32,
            (named[3].1.size() / 80) as u32,
        )
        .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode(
                device,
                &mut encoder,
                SoftRigidSolveBindings {
                    positions: named[0].1,
                    velocities: named[1].1,
                    inverse_mass: named[2].1,
                    contacts: named[3].1,
                    body_poses: named[4].1,
                    body_com: named[5].1,
                    jacobians: named[6].1,
                    mass_matrix: named[7].1,
                    rigid_velocities: named[8].1,
                    body_velocities: named[9].1,
                    particle_count: (named[0].1.size() / 16) as u32,
                    body_count: (named[4].1.size() / 32) as u32,
                },
                settings,
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(bytes(&engine, kernel.status()), 0u32.to_le_bytes());
        assert!(readback::read::<u32>(&engine, kernel.active_rows(), 1)[0] > 0);
        for (actual, expected) in [(named[1].1, solved[0].1), (named[8].1, solved[1].1)] {
            assert_eq!(bytes(&engine, actual), bytes(&engine, expected));
        }
        if !stopped {
            assert_eq!(
                bytes(&engine, named[9].1),
                bytes(
                    &engine,
                    captured.rigid_state().body_qd.device_buffer().unwrap()
                )
            );
        }
    }
    captured.configure_contact_snapshot(true).unwrap();
    assert!(captured.contact_snapshot().is_none());
    captured.configure_contact_snapshot(false).unwrap();
    captured.step(fem_rotor::config().max_substep).unwrap();
    assert!(captured.contact_snapshot().is_none());
}
