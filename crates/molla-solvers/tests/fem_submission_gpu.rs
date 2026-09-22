#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_solvers::fem_rigid_gpu::FemRigidGpuSystem;
use std::sync::Arc;

#[path = "common/gpu_readback.rs"]
mod gpu_readback;

#[path = "common/fem_rotor.rs"]
mod fem_rotor;

fn system(engine: Arc<Engine>) -> FemRigidGpuSystem {
    let mut system =
        FemRigidGpuSystem::new(engine, fem_rotor::scene(), fem_rotor::config()).unwrap();
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

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn asynchronous_substeps_match_blocking_without_host_state() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut blocking = system(engine.clone());
    let mut asynchronous = system(engine.clone());
    let h = 1.0 / 4096.0;
    blocking.step(65.0 * h).unwrap();
    assert!(asynchronous.poll_completion().unwrap());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let mut submitted = 0;
    while submitted < 65 {
        assert!(std::time::Instant::now() < deadline);
        if asynchronous.try_step_substep(h).unwrap() {
            submitted += 1;
        }
        assert!(asynchronous.queued_substeps() <= 1);
        std::thread::yield_now();
    }
    for invalid in [0.0, -h, 2.0 * h, f64::NAN, f64::INFINITY] {
        assert!(asynchronous.try_step_substep(invalid).is_err());
        assert_eq!(asynchronous.queued_substeps(), 1);
    }
    while !asynchronous.poll_completion().unwrap() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(asynchronous.queued_substeps(), 0);
    assert!(asynchronous.soft_state().particle_q.host().is_err());
    assert!(asynchronous.rigid_state().joint_q.host().is_err());
    for state in [&blocking, &asynchronous] {
        assert_eq!(
            gpu_readback::read::<u32>(&engine, state.status_buffer(), 1),
            [0]
        );
    }
    assert_eq!(
        gpu_readback::read::<[f32; 4]>(
            &engine,
            blocking.soft_state().particle_q.device_buffer().unwrap(),
            4,
        ),
        gpu_readback::read::<[f32; 4]>(
            &engine,
            asynchronous
                .soft_state()
                .particle_q
                .device_buffer()
                .unwrap(),
            4,
        ),
    );
    let velocity = |state: &FemRigidGpuSystem| {
        gpu_readback::read::<f32>(
            &engine,
            state.rigid_state().joint_qd.device_buffer().unwrap(),
            1,
        )
    };
    assert_eq!(velocity(&blocking), velocity(&asynchronous));
    assert!((velocity(&asynchronous)[0] as f64 - 2.0 * 65.0 * h).abs() < 1e-6);
    assert_eq!(
        gpu_readback::body_poses(&engine, &blocking, 1),
        gpu_readback::body_poses(&engine, &asynchronous, 1),
    );
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn asynchronous_and_blocking_submission_modes_can_alternate() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut mixed = system(engine.clone());
    let mut reference = system(engine.clone());
    let h = 1.0 / 4096.0;
    mixed.step(31.0 * h).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while !mixed.try_step_substep(h).unwrap() {
        assert!(std::time::Instant::now() < deadline);
        assert_eq!(mixed.queued_substeps(), 31);
        std::thread::yield_now();
    }
    assert_eq!(mixed.queued_substeps(), 1);
    mixed.step(33.0 * h).unwrap();
    assert_eq!(mixed.queued_substeps(), 1);
    reference.step(65.0 * h).unwrap();
    for state in [&mixed, &reference] {
        assert_eq!(
            gpu_readback::read::<u32>(&engine, state.status_buffer(), 1),
            [0]
        );
    }
    assert_eq!(
        gpu_readback::body_poses(&engine, &mixed, 1),
        gpu_readback::body_poses(&engine, &reference, 1),
    );
    assert_eq!(
        gpu_readback::read::<f32>(
            &engine,
            mixed.rigid_state().joint_qd.device_buffer().unwrap(),
            1,
        ),
        gpu_readback::read::<f32>(
            &engine,
            reference.rigid_state().joint_qd.device_buffer().unwrap(),
            1,
        ),
    );
}
