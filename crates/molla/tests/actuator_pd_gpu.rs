//! GPU PD / PID actuator — golden-value tests against hand-computed
//! expectations. `#[ignore]` because they need a real wgpu adapter;
//! run with `make test-actuator-gpu`.

#![cfg(feature = "gpu")]

use molla::actuators::{ControllerPd, ControllerPdConfig, ControllerPid, ControllerPidConfig};
use molla::compute::Engine;
use molla::math::{Real, Transform, Vec3};
use molla::prelude::*;

/// Build a 1-revolute-joint articulation so we get a valid
/// `joint_q[0]` / `joint_qd[0]` slot, upload everything, and return
/// the bits the test needs.
fn setup_one_joint() -> Option<(Engine, Model, State, Control)> {
    let mut b = ModelBuilder::new();
    b.begin_articulation();
    let parent = BodyId::NONE;
    let child = b.add_body(BodyParams::new());
    let _ = b
        .add_joint_revolute(
            parent,
            child,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::Y,
        )
        .ok()?;
    let model = b.build().ok()?;

    let engine = Engine::new_gpu().ok()?;
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut model = model;
    model.upload_to_device(&device, &queue).ok()?;
    let mut state = model.state().ok()?;
    state.upload_to_device(&device, &queue).ok()?;
    let mut control = model.control();
    control.upload_to_device(&device, &queue).ok()?;
    Some((engine, model, state, control))
}

/// Set `joint_q[0]`, `joint_qd[0]`, target_pos, target_vel via host
/// mirrors and re-upload so the kernel sees the new values.
fn set_state(
    engine: &Engine,
    state: &mut State,
    control: &mut Control,
    q: Real,
    qd: Real,
    target_q: Real,
    target_qd: Real,
) {
    let queue = engine.queue().unwrap().clone();
    state.joint_q.host_mut().unwrap()[0] = q;
    state.joint_qd.host_mut().unwrap()[0] = qd;
    control.joint_target_pos.host_mut().unwrap()[0] = target_q;
    control.joint_target_vel.host_mut().unwrap()[0] = target_qd;
    for v in control.joint_force.host_mut().unwrap() {
        *v = 0.0;
    }
    state.joint_q.upload(&queue).unwrap();
    state.joint_qd.upload(&queue).unwrap();
    control.joint_target_pos.upload(&queue).unwrap();
    control.joint_target_vel.upload(&queue).unwrap();
    control.joint_force.upload(&queue).unwrap();
}

fn read_force(engine: &Engine, control: &mut Control) -> Real {
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    control.joint_force.download(&device, &queue).unwrap();
    control.joint_force.host().unwrap()[0]
}

/// PD: effort = kp·(target_q - q) + kd·(target_qd - qd).
/// q=0.0, qd=0.0, target=1.0 → effort = kp·1.0 = 50.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-gpu"]
fn pd_basic_position_error() {
    let (engine, model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");
    let mut pd = ControllerPd::new(
        &engine,
        &model,
        &[ControllerPdConfig {
            dof: 0,
            kp: 50.0,
            kd: 5.0,
            max_effort: 0.0,
        }],
    )
    .unwrap();
    set_state(&engine, &mut state, &mut control, 0.0, 0.0, 1.0, 0.0);
    pd.step(&engine, &state, &mut control).unwrap();
    let f = read_force(&engine, &mut control);
    eprintln!("PD effort: {f}");
    assert!((f - 50.0).abs() < 1.0e-3, "expected 50, got {f}");
}

/// PD: derivative term — q=0.0, qd=2.0, target=0.0, target_qd=0.0
/// effort = 0 + kd·(0 - 2) = -10.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-gpu"]
fn pd_velocity_error() {
    let (engine, model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");
    let mut pd = ControllerPd::new(
        &engine,
        &model,
        &[ControllerPdConfig {
            dof: 0,
            kp: 0.0,
            kd: 5.0,
            max_effort: 0.0,
        }],
    )
    .unwrap();
    set_state(&engine, &mut state, &mut control, 0.0, 2.0, 0.0, 0.0);
    pd.step(&engine, &state, &mut control).unwrap();
    let f = read_force(&engine, &mut control);
    assert!((f - (-10.0)).abs() < 1.0e-3, "expected -10, got {f}");
}

/// PD: effort cap clamps a huge error.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-gpu"]
fn pd_effort_cap_clamps() {
    let (engine, model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");
    let mut pd = ControllerPd::new(
        &engine,
        &model,
        &[ControllerPdConfig {
            dof: 0,
            kp: 1000.0,
            kd: 0.0,
            max_effort: 25.0, // raw effort would be 1000, cap to 25
        }],
    )
    .unwrap();
    set_state(&engine, &mut state, &mut control, 0.0, 0.0, 1.0, 0.0);
    pd.step(&engine, &state, &mut control).unwrap();
    let f = read_force(&engine, &mut control);
    assert!(
        (f - 25.0).abs() < 1.0e-3,
        "expected cap at 25, got {f}"
    );

    // Negative direction also clamped.
    set_state(&engine, &mut state, &mut control, 1.0, 0.0, 0.0, 0.0);
    pd.step(&engine, &state, &mut control).unwrap();
    let f = read_force(&engine, &mut control);
    assert!(
        (f - (-25.0)).abs() < 1.0e-3,
        "expected -25, got {f}"
    );
}

/// PID: integral term grows with constant position error. Over 4
/// steps with dt=0.1, error=1.0, integral = clamp(0 + 4·0.1 = 0.4,
/// integral_max). With ki=10 → integral contributes 10·0.4 = 4.0.
/// Plus kp=20·1.0 = 20. Total effort = 24.0 (no cap, no kd term).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-gpu"]
fn pid_integral_accumulates() {
    let (engine, model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");
    let mut pid = ControllerPid::new(
        &engine,
        &model,
        &[ControllerPidConfig {
            dof: 0,
            kp: 20.0,
            ki: 10.0,
            kd: 0.0,
            integral_max: 100.0,
            max_effort: 0.0,
        }],
    )
    .unwrap();
    let dt: Real = 0.1;
    set_state(&engine, &mut state, &mut control, 0.0, 0.0, 1.0, 0.0);
    for _ in 0..4 {
        pid.step(&engine, &state, &mut control, dt).unwrap();
    }
    let f = read_force(&engine, &mut control);
    // Expected: kp·error + ki·integral = 20·1 + 10·(4·0.1·1) = 20 + 4 = 24.
    eprintln!("PID effort after 4 steps: {f}");
    assert!((f - 24.0).abs() < 1.0e-2, "expected 24, got {f}");
}

/// PID anti-windup clamp limits integral growth even when error
/// persists. With integral_max=0.2, after 10 steps · dt=0.1 · err=1
/// the raw integral would be 1.0, but clamp pins it at 0.2.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-gpu"]
fn pid_anti_windup_clamp() {
    let (engine, model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");
    let mut pid = ControllerPid::new(
        &engine,
        &model,
        &[ControllerPidConfig {
            dof: 0,
            kp: 0.0,
            ki: 50.0,
            kd: 0.0,
            integral_max: 0.2,
            max_effort: 0.0,
        }],
    )
    .unwrap();
    let dt: Real = 0.1;
    set_state(&engine, &mut state, &mut control, 0.0, 0.0, 1.0, 0.0);
    for _ in 0..10 {
        pid.step(&engine, &state, &mut control, dt).unwrap();
    }
    let f = read_force(&engine, &mut control);
    // Capped: ki · integral_max = 50 · 0.2 = 10.
    eprintln!("PID anti-windup effort: {f}");
    assert!(
        (f - 10.0).abs() < 1.0e-2,
        "expected 10 (capped integral), got {f}"
    );
}
