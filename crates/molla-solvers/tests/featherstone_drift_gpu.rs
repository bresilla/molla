#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Mat3, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder, eval_fk};
use molla_solvers::FeatherstoneSolver;

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn corrected_velocity_drift_matches_cpu_without_reapplying_forces() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.begin_articulation();
    let mut parent = BodyId::NONE;
    for kind in 0..8 {
        let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 2.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        let xp = Transform::from_translation(Vec3::new(0.03, 0.04, 0.05));
        let xc = Transform::IDENTITY;
        let joint = match kind {
            0 => builder.add_joint_prismatic(parent, body, xp, xc, Vec3::Y),
            1 => builder.add_joint_revolute(parent, body, xp, xc, Vec3::X),
            2 => builder.add_joint_ball(parent, body, xp, xc),
            3 => builder.add_joint_fixed(parent, body, xp, xc),
            4 => builder.add_joint_free(parent, body, xp, xc),
            5 => builder.add_joint_distance(parent, body, xp, xc),
            6 => builder.add_joint_d6(parent, body, xp, xc),
            _ => builder.add_joint_cable(parent, body, xp, xc, 50.0, 20.0, 2.0, 2.0),
        }
        .unwrap();
        if kind == 6 {
            builder
                .set_joint_limits(joint, &[0.0; 6], &[0.0; 6])
                .unwrap();
        }
        parent = body;
    }
    let mut model = builder.build().unwrap();
    let mut start = model.state().unwrap();
    start.joint_qd.host_mut().unwrap().fill(-9.0);
    eval_fk(&model, &mut start).unwrap();
    let mut cpu = start.clone_host().unwrap();
    for (i, v) in cpu.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
        *v = (i as f64 + 1.0) * 0.03;
    }
    let mut gpu = cpu.clone_host().unwrap();
    let mut control = model.control();
    control.joint_force.host_mut().unwrap().fill(1000.0);
    FeatherstoneSolver::new_cpu()
        .drift_cpu(&model, &start, 0.01, &mut cpu)
        .unwrap();
    model.upload_to_device(device, queue).unwrap();
    start.upload_to_device(device, queue).unwrap();
    gpu.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    FeatherstoneSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_damping(20.0)
        .drift_gpu(&model, &start, &control, 0.01, &mut gpu)
        .unwrap();
    assert!(gpu.joint_q.host().is_err());
    gpu.joint_q.download(device, queue).unwrap();
    gpu.joint_qd.download(device, queue).unwrap();
    gpu.body_q.download(device, queue).unwrap();
    for (a, b) in gpu
        .joint_q
        .host()
        .unwrap()
        .iter()
        .zip(cpu.joint_q.host().unwrap())
    {
        assert!((a - b).abs() < 2e-6, "coordinate {a} vs {b}");
    }
    for (a, b) in gpu
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .zip(cpu.joint_qd.host().unwrap())
    {
        assert!((a - b).abs() < 2e-6, "velocity {a} vs {b}");
    }
    for (a, b) in gpu
        .body_q
        .host()
        .unwrap()
        .iter()
        .zip(cpu.body_q.host().unwrap())
    {
        assert!((a.position - b.position).length() < 2e-6);
        assert!(a.rotation.dot(b.rotation).abs() > 1.0 - 2e-6);
    }
}
