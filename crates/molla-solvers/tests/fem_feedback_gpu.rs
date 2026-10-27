#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_sim::JointTargetMode;
use molla_solvers::fem_rigid_gpu::FemRigidGpuSystem;
use std::sync::Arc;

#[path = "common/fem_rotor.rs"]
mod fem_rotor;
#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn fem_feedback_applies_once_before_contact_and_not_again_during_drift() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let h = 1.0 / 19200.0;
    let kp = 572957.795;
    let kd = 8594.367;
    for stable in [false, true] {
        let mut scene = fem_rotor::scene();
        scene.rigid_model.joint_effort_limit.host_mut().unwrap()[0] = 20000.0;
        scene.control.joint_target_mode.host_mut().unwrap()[0] =
            JointTargetMode::PositionVelocity as u32;
        scene.control.joint_target_pos.host_mut().unwrap()[0] = 1e-4;
        scene.control.joint_target_kp.host_mut().unwrap()[0] = kp;
        scene.control.joint_target_kd.host_mut().unwrap()[0] = kd;
        let mut config = fem_rotor::config();
        config.max_substep = h;
        let mut system = FemRigidGpuSystem::new(engine.clone(), scene, config)
            .unwrap()
            .with_stable_motor_feedback(stable);
        let mut q = 0.0;
        let mut qd = 0.0;
        for step in 1..=32 {
            let drive = kp * (1e-4 - q) - kd * qd;
            let torque = if stable {
                (drive - kp * h * qd) / (1.0 + (kd + kp * h) * h / 0.25)
            } else {
                drive
            };
            qd += h * torque / 0.25;
            q += h * qd;
            system.step(h).unwrap();
            let actual_q = gpu_readback::read::<f32>(
                &engine,
                system.rigid_state().joint_q.device_buffer().unwrap(),
                1,
            )[0] as f64;
            let actual_qd = gpu_readback::read::<f32>(
                &engine,
                system.rigid_state().joint_qd.device_buffer().unwrap(),
                1,
            )[0] as f64;
            assert!((actual_q - q).abs() < 2e-9, "stable={stable} step={step}");
            assert!((actual_qd - qd).abs() < 2e-7, "stable={stable} step={step}");
            assert_eq!(
                gpu_readback::read::<u32>(&engine, system.status_buffer(), 1)[0],
                0
            );
            assert!(system.rigid_state().joint_q.host().is_err());
            assert!(system.soft_state().particle_q.host().is_err());
        }
    }
}
