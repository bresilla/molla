//! Stage-18 composable actuator pipeline — GPU parity + clamp tests.
//! `#[ignore]` because they need a real wgpu adapter; run with
//! `make test-actuator-pipeline-gpu` (or the equivalent
//! `cargo test -p molla --features gpu --test actuator_pipeline_gpu -- --ignored`).

#![cfg(feature = "gpu")]

use molla::actuators::{
    Actuator, ClampingDcMotor, ClampingDcMotorConfig, ClampingMaxEffort,
    ClampingMaxEffortConfig, ClampingStage, ControllerPd, ControllerPdConfig,
    ControllerPipelinePd, ControllerPipelinePdConfig, ControllerStage, Delay, DelayConfig,
};
use molla::compute::Engine;
use molla::math::{Real, Transform, Vec3};
use molla::prelude::*;

/// Build a 1-revolute-joint articulation so `joint_q[0] / joint_qd[0]`
/// exist; identical to `actuator_pd_gpu.rs`'s helper.
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

/// Parity: a standalone `ControllerPd { kp, kd, max_effort=cap }`
/// produces the same effort as
/// `Actuator { delay=0, ctrl=PipelinePd, clamp=MaxEffort(cap) }`
/// for the same `(q, qd, target_q, target_qd)`.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-pipeline-gpu"]
fn pipeline_matches_standalone_pd() {
    let Some((engine, model, mut state, mut control)) = setup_one_joint() else {
        return;
    };

    let kp: Real = 75.0;
    let kd: Real = 8.0;
    let cap: Real = 30.0;

    // Standalone PD (effort cap baked in).
    let mut pd = ControllerPd::new(
        &engine,
        &model,
        &[ControllerPdConfig {
            dof: 0,
            kp,
            kd,
            max_effort: cap,
        }],
    )
    .unwrap();

    set_state(&engine, &mut state, &mut control, 0.1, 0.2, 0.5, 0.0);
    pd.step(&engine, &state, &mut control).unwrap();
    let f_standalone = read_force(&engine, &mut control);

    // Composed pipeline: zero-lag delay → pipeline-PD → MaxEffort.
    let delay = Delay::new(
        &engine,
        &[DelayConfig {
            dof: 0,
            delay_steps: 0,
        }],
        1, // max_delay must be ≥ 1 even for zero-lag
    )
    .unwrap();
    let pipe_pd = ControllerPipelinePd::new(
        &engine,
        &[ControllerPipelinePdConfig { dof: 0, kp, kd }],
    )
    .unwrap();
    let clamp = ClampingMaxEffort::new(
        &engine,
        &[ClampingMaxEffortConfig {
            dof: 0,
            max_effort: cap,
        }],
    )
    .unwrap();
    let mut act = Actuator::new(
        delay,
        ControllerStage::Pd(pipe_pd),
        ClampingStage::MaxEffort(clamp),
    );

    set_state(&engine, &mut state, &mut control, 0.1, 0.2, 0.5, 0.0);
    act.step(&engine, &state, &mut control).unwrap();
    let f_pipeline = read_force(&engine, &mut control);

    eprintln!("standalone PD: {f_standalone}, pipeline: {f_pipeline}");
    assert!(
        (f_standalone - f_pipeline).abs() < 1.0e-3,
        "expected pipeline to match standalone PD: standalone={f_standalone}, pipeline={f_pipeline}"
    );

    // Sanity: in the saturating regime they must both clip identically.
    set_state(&engine, &mut state, &mut control, 0.0, 0.0, 5.0, 0.0); // huge error
    pd.step(&engine, &state, &mut control).unwrap();
    let f_standalone_cap = read_force(&engine, &mut control);
    set_state(&engine, &mut state, &mut control, 0.0, 0.0, 5.0, 0.0);
    act.step(&engine, &state, &mut control).unwrap();
    let f_pipeline_cap = read_force(&engine, &mut control);
    assert!(
        (f_standalone_cap - cap).abs() < 1.0e-3 && (f_pipeline_cap - cap).abs() < 1.0e-3,
        "both must hit cap: standalone={f_standalone_cap}, pipeline={f_pipeline_cap}, cap={cap}"
    );
}

/// DC-motor clamp: at qd=0 the clamp passes a small effort untouched;
/// at qd=no_load_speed the clamp must drop the effort (in the
/// direction of motion) to zero. Sweeps a few velocities and checks
/// the clamp follows `effort_max(qd) = stall · (1 − qd / no_load)`.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-pipeline-gpu"]
fn dc_motor_clamp_torque_speed_curve() {
    let Some((engine, model, mut state, mut control)) = setup_one_joint() else {
        return;
    };

    let stall: Real = 10.0; // N·m at zero velocity
    let no_load: Real = 5.0; // rad/s (zero-torque crossing)
    let cap: Real = 20.0; // current limit ≫ stall (no flat-top in test sweep)

    // Run controller with kp=1000 so the requested effort is far above
    // anything the motor can actually deliver — this isolates the
    // clamp's behaviour from the controller.
    let _ = ControllerPd::new(
        &engine,
        &model,
        &[ControllerPdConfig {
            dof: 0,
            kp: 1000.0,
            kd: 0.0,
            max_effort: 0.0,
        }],
    )
    .unwrap();

    let mut clamp = ClampingDcMotor::new(
        &engine,
        &[ClampingDcMotorConfig {
            dof: 0,
            stall_torque: stall,
            no_load_speed: no_load,
            max_motor_effort: cap,
        }],
    )
    .unwrap();

    // For each qd, write a "huge positive" raw effort into joint_force
    // and then run the clamp; expected output is the upper rail of
    // the linear envelope.
    let test_velocities: [Real; 5] = [0.0, 1.0, 2.5, 4.0, 5.0];
    for &qd in &test_velocities {
        // Manually pre-load joint_force with a huge positive request.
        let queue = engine.queue().unwrap().clone();
        state.joint_qd.host_mut().unwrap()[0] = qd;
        state.joint_qd.upload(&queue).unwrap();
        control.joint_force.host_mut().unwrap()[0] = 1.0e3;
        control.joint_force.upload(&queue).unwrap();

        use molla::actuators::Clamping;
        clamp.step(&engine, &state, &mut control).unwrap();
        let f = read_force(&engine, &mut control);

        let expected = (stall * (1.0 - qd / no_load)).clamp(-cap, cap);
        eprintln!("qd={qd}: clamp out={f}, expected={expected}");
        assert!(
            (f - expected).abs() < 1.0e-3,
            "qd={qd}: expected {expected}, got {f}"
        );
    }

    // Negative direction sanity check.
    let queue = engine.queue().unwrap().clone();
    state.joint_qd.host_mut().unwrap()[0] = -2.0;
    state.joint_qd.upload(&queue).unwrap();
    control.joint_force.host_mut().unwrap()[0] = -1.0e3;
    control.joint_force.upload(&queue).unwrap();
    use molla::actuators::Clamping;
    clamp.step(&engine, &state, &mut control).unwrap();
    let f = read_force(&engine, &mut control);
    let expected = (stall * (-1.0 - (-2.0) / no_load)).clamp(-cap, cap);
    eprintln!("qd=-2.0: clamp out={f}, expected={expected}");
    assert!((f - expected).abs() < 1.0e-3);
}

/// Delay stage: with a constant input target, after exactly `delay`
/// substeps the controller's "delayed target" reads back the
/// most-recently-pushed value. Before that, the read clamps to the
/// oldest available sample (Newton semantics).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-pipeline-gpu"]
fn delay_holds_step_response_for_n_substeps() {
    let Some((engine, _model, mut state, mut control)) = setup_one_joint() else {
        return;
    };
    let _ = &mut state; // unused-mut suppression

    let delay_n: u32 = 5;
    let mut delay = Delay::new(
        &engine,
        &[DelayConfig {
            dof: 0,
            delay_steps: delay_n,
        }],
        delay_n + 2, // a bit of headroom on the buffer depth
    )
    .unwrap();

    // Helper: set the current target, push it through the delay, and
    // download `out_pos[0]` so we can compare.
    let push_and_read =
        |delay: &mut Delay, control: &mut Control, target: Real| -> Real {
            let queue = engine.queue().unwrap().clone();
            let device = engine.device().unwrap().clone();
            control.joint_target_pos.host_mut().unwrap()[0] = target;
            control.joint_target_vel.host_mut().unwrap()[0] = 0.0;
            control.joint_target_pos.upload(&queue).unwrap();
            control.joint_target_vel.upload(&queue).unwrap();
            delay.step(&engine, control).unwrap();
            let out = delay.out_pos_mut();
            out.download(&device, &queue).unwrap();
            out.host().unwrap()[0]
        };

    // Frame 1: push target=1.0. With delay=5 but only 1 sample
    // available, the read clamps to the oldest available sample —
    // which is also the most recent push (1.0). Expect 1.0.
    let v1 = push_and_read(&mut delay, &mut control, 1.0);
    eprintln!("after push 1: {v1}");
    assert!((v1 - 1.0).abs() < 1.0e-5);

    // Frames 2..5: push 2.0 each frame. With delay=5 and
    // num_pushes=k, lag is clamped to k-1 → we always read the
    // oldest sample, which is still 1.0.
    for i in 2..=5u32 {
        let v = push_and_read(&mut delay, &mut control, 2.0);
        eprintln!("after push {i}: {v} (expect 1.0)");
        assert!(
            (v - 1.0).abs() < 1.0e-5,
            "frame {i}: expected oldest sample 1.0, got {v}"
        );
    }

    // Frame 6: now the buffer is full (5 pushes covering frames 2..5
    // plus the initial 1.0 push from frame 1, so the oldest is still
    // 1.0). Push 3.0 — buffer rotates, oldest entry becomes 2.0.
    let v6 = push_and_read(&mut delay, &mut control, 3.0);
    eprintln!("after push 6: {v6} (expect 2.0 — first 1.0 has aged out)");
    assert!(
        (v6 - 2.0).abs() < 1.0e-5,
        "frame 6: expected 2.0, got {v6}"
    );
}

/// Zero-lag passthrough: with `delay_steps == 0`, the delay must
/// always emit the current target (matches the "no delay" Newton
/// case).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-pipeline-gpu"]
fn delay_zero_steps_is_passthrough() {
    let Some((engine, _model, _state, mut control)) = setup_one_joint() else {
        return;
    };
    let mut delay = Delay::new(
        &engine,
        &[DelayConfig {
            dof: 0,
            delay_steps: 0,
        }],
        1,
    )
    .unwrap();

    let queue = engine.queue().unwrap().clone();
    let device = engine.device().unwrap().clone();

    for &target in &[0.0_f64, 1.0, -2.0, 0.42] {
        control.joint_target_pos.host_mut().unwrap()[0] = target;
        control.joint_target_pos.upload(&queue).unwrap();
        delay.step(&engine, &control).unwrap();
        let out = delay.out_pos_mut();
        out.download(&device, &queue).unwrap();
        let v = out.host().unwrap()[0];
        assert!((v - target).abs() < 1.0e-5, "passthrough: expected {target}, got {v}");
    }
}
