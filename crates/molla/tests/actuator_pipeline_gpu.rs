//! Stage-18 composable actuator pipeline — GPU parity + clamp tests.
//! `#[ignore]` because they need a real wgpu adapter; run with
//! `make test-actuator-pipeline-gpu` (or the equivalent
//! `cargo test -p molla --features gpu --test actuator_pipeline_gpu -- --ignored`).

#![cfg(feature = "gpu")]

use molla::actuators::{
    Actuator, ClampingDcMotor, ClampingDcMotorConfig, ClampingMaxEffort,
    ClampingMaxEffortConfig, ClampingPositionBased, ClampingStage,
    ControllerNeuralMlp, ControllerNeuralMlpConfig, ControllerPd, ControllerPdConfig,
    ControllerPid, ControllerPidConfig, ControllerPipelinePd, ControllerPipelinePdConfig,
    ControllerPipelinePid, ControllerPipelinePidConfig, ControllerStage, Delay, DelayConfig,
    MlpWeights,
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
    let (engine, model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");

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
    act.step(&engine, &state, &mut control, 1.0e-3).unwrap();
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
    act.step(&engine, &state, &mut control, 1.0e-3).unwrap();
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
    let (engine, model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");

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
    let (engine, _model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");
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
    let (engine, _model, _state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");
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

// ── Stage 18.1 — PID pipeline parity ────────────────────────────────
//
// Standalone `ControllerPid { kp,ki,kd, integral_max, max_effort=cap }`
// must produce the same effort sequence as
// `Actuator { delay=0, ctrl=PipelinePid, clamp=MaxEffort(cap) }`
// across a step-response history. Both controllers carry an integral
// term so the test runs ≥ a few substeps to give the integral a
// chance to wind up — single-shot parity is too easy.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-pipeline-gpu"]
fn pid_pipeline_matches_standalone_pid() {
    let (engine, model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");

    let kp: Real = 50.0;
    let ki: Real = 30.0;
    let kd: Real = 6.0;
    let imax: Real = 2.0;
    let cap: Real = 25.0;
    let dt: Real = 1.0e-3;

    let mut pid = ControllerPid::new(
        &engine,
        &model,
        &[ControllerPidConfig {
            dof: 0,
            kp,
            ki,
            kd,
            integral_max: imax,
            max_effort: cap,
        }],
    )
    .unwrap();

    let delay = Delay::new(
        &engine,
        &[DelayConfig {
            dof: 0,
            delay_steps: 0,
        }],
        1,
    )
    .unwrap();
    let pipe_pid = ControllerPipelinePid::new(
        &engine,
        &[ControllerPipelinePidConfig {
            dof: 0,
            kp,
            ki,
            kd,
            integral_max: imax,
        }],
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
        ControllerStage::Pid(pipe_pid),
        ClampingStage::MaxEffort(clamp),
    );

    // Step-response history: target ramp from 0.5 to 1.5, current
    // joint state stays at zero (controller-only test, no integration).
    let targets: [Real; 8] = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5];
    for (i, &target) in targets.iter().enumerate() {
        // Standalone — joint state stays zero, only target moves.
        set_state(&engine, &mut state, &mut control, 0.0, 0.0, target, 0.0);
        pid.step(&engine, &state, &mut control, dt).unwrap();
        let f_standalone = read_force(&engine, &mut control);

        // Pipeline.
        set_state(&engine, &mut state, &mut control, 0.0, 0.0, target, 0.0);
        act.step(&engine, &state, &mut control, dt).unwrap();
        let f_pipeline = read_force(&engine, &mut control);

        eprintln!(
            "step {i}: target={target}, standalone={f_standalone}, pipeline={f_pipeline}"
        );
        assert!(
            (f_standalone - f_pipeline).abs() < 1.0e-3,
            "step {i}: standalone={f_standalone}, pipeline={f_pipeline}"
        );
    }
}

// ── Stage 18.1 — ClampingPositionBased smoothness + parity ──────────
//
// The kernel matches Newton's `_position_based_clamp_kernel`
// line-for-line per the Stage-18 commit message but no test exercises
// it. This test asserts:
//   1. The clamp interpolates linearly between adjacent table points
//      (smoothness — no hard cliff).
//   2. Outside the table the clamp pins to the boundary.
//
// Setup: a 5-point lookup that drops effort from 10 N·m → 0 N·m over
// the last 5° (≈ 0.0873 rad) before the joint limit at 1.0 rad. The
// test sweeps `joint_q` across 0 → 1.1 rad and checks the per-step
// torque envelope matches the analytic linear envelope.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-pipeline-gpu"]
fn position_based_clamp_smooth_envelope() {
    let (engine, _model, mut state, mut control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");

    // Lookup: full effort at q ≤ 0.913 rad, ramps to zero at 1.0 rad.
    let positions: Vec<Real> = vec![-1.0, 0.913, 1.0, 1.5];
    let efforts: Vec<Real> = vec![10.0, 10.0, 0.0, 0.0];
    use molla::actuators::Clamping;
    let mut clamp =
        ClampingPositionBased::new(&engine, &[0u32], &positions, &efforts).unwrap();

    let queue = engine.queue().unwrap().clone();
    // Sample joint angles bracketing the soft limit.
    let samples: Vec<Real> = vec![
        0.0,    // far below limit — full effort
        0.5,    // still well below — full effort
        0.913,  // exactly at the start of ramp — still 10
        0.9565, // halfway through the ramp — analytic 5.0
        0.9783, // 75% through — analytic 2.5
        1.0,    // at the limit — 0.0
        1.05,   // past limit — 0.0
    ];

    let mut last = f32::INFINITY;
    for &q in &samples {
        state.joint_q.host_mut().unwrap()[0] = q;
        state.joint_q.upload(&queue).unwrap();
        // Pre-load joint_force with a huge request — clamp drives it
        // down to the envelope.
        control.joint_force.host_mut().unwrap()[0] = 1.0e3;
        control.joint_force.upload(&queue).unwrap();
        clamp.step(&engine, &state, &mut control).unwrap();
        let f = read_force(&engine, &mut control) as f32;

        // Analytic envelope: linear interp between table points.
        let expected = if q <= 0.913 {
            10.0
        } else if q >= 1.0 {
            0.0
        } else {
            let t = (q - 0.913) / (1.0 - 0.913);
            10.0 * (1.0 - t)
        } as f32;
        eprintln!("q={q}: clamped={f}, expected={expected}");
        assert!(
            (f - expected).abs() < 1.0e-2,
            "q={q}: expected {expected}, got {f}"
        );
        // Smoothness: monotone non-increasing as q increases past the
        // soft limit (no cliff — the ramp samples should produce a
        // sequence of decreasing values).
        if q >= 0.913 {
            assert!(
                f <= last + 1.0e-3,
                "q={q}: torque jumped up from {last} to {f}"
            );
            last = f;
        }
    }
}

// ── Stage 18.1 — MLP forward-pass correctness ───────────────────────
//
// Build a small 3→4→4→2 net with hand-rolled weights, run the kernel,
// and compare the output against the analytic CPU reference. Tests:
//   • bias is added
//   • tanh non-linearity on hidden layers
//   • linear output
//   • input → hidden → output dimension routing all match
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-pipeline-gpu"]
fn mlp_forward_matches_cpu_reference() {
    let (engine, _model, _state, _control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");

    let cfg = ControllerNeuralMlpConfig {
        input_dim: 3,
        hidden_dim: 4,
        output_dim: 2,
    };
    // dof_indices length must equal output_dim. For this pure-forward
    // test we never call `step()` (which would scatter into
    // `joint_force`), so the actual indices don't matter — they just
    // have to be unique. Using {0,1} would race against
    // setup_one_joint's 1-DOF model if scatter were called; that's
    // safe here because we only call `forward()`.
    let mut mlp =
        ControllerNeuralMlp::new(&engine, cfg, &[0u32, 1u32]).expect("mlp ctor");

    // Hand-rolled weights: simple identity-ish so the result is easy
    // to predict.
    // w1 = 4×3, b1 = 4
    let w1: Vec<Real> = vec![
        1.0, 0.0, 0.0,  // h1 = i0
        0.0, 1.0, 0.0,  // h2 = i1
        0.0, 0.0, 1.0,  // h3 = i2
        1.0, 1.0, 1.0,  // h4 = i0+i1+i2
    ];
    let b1: Vec<Real> = vec![0.0, 0.0, 0.0, -0.5];
    // w2 = 4×4, b2 = 4 — hidden→hidden, identity.
    let w2: Vec<Real> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let b2: Vec<Real> = vec![0.0; 4];
    // w3 = 2×4, b3 = 2 — first output sums hidden, second is h4.
    let w3: Vec<Real> = vec![
        1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let b3: Vec<Real> = vec![0.1, -0.2];

    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    mlp.set_weights(
        &device,
        &queue,
        &MlpWeights {
            w1: &w1,
            b1: &b1,
            w2: &w2,
            b2: &b2,
            w3: &w3,
            b3: &b3,
        },
    )
    .unwrap();

    let inputs: Vec<Real> = vec![0.3, -0.1, 0.5];
    mlp.set_inputs(&queue, &inputs).unwrap();
    mlp.forward(&engine).unwrap();
    let out = mlp.download_outputs(&device, &queue).unwrap();

    // CPU reference.
    let h1: Vec<Real> = vec![
        (inputs[0] + b1[0]).tanh(),
        (inputs[1] + b1[1]).tanh(),
        (inputs[2] + b1[2]).tanh(),
        (inputs[0] + inputs[1] + inputs[2] + b1[3]).tanh(),
    ];
    let h2: Vec<Real> = (0..4).map(|i| (h1[i] + b2[i]).tanh()).collect();
    let expected: Vec<Real> = vec![
        h2[0] + h2[1] + h2[2] + b3[0],
        h2[3] + b3[1],
    ];
    eprintln!("mlp out: {out:?}, expected: {expected:?}");
    for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1.0e-4,
            "out[{i}]: got {a}, expected {b}"
        );
    }
}

// ── Stage 18.1 — MLP forward-pass timing on the 32→64→64→8 net ──────
//
// Log-only timing test. Repeated dispatches; we report the average
// per-call wall-clock plus the GPU-queue submit. Don't fail on slow
// machines — adapters vary 10× across the test fleet.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-actuator-pipeline-gpu"]
fn mlp_forward_timing_log_only() {
    use std::time::Instant;
    let (engine, _model, _state, _control) = setup_one_joint().expect("GPU adapter required for this test (gated by #[ignore])");
    let cfg = ControllerNeuralMlpConfig {
        input_dim: 32,
        hidden_dim: 64,
        output_dim: 8,
    };
    let mut mlp = ControllerNeuralMlp::new(&engine, cfg, &(0..8).collect::<Vec<_>>())
        .expect("mlp ctor failed (DOF range exceeds 1-joint setup, but only the kernel is exercised below)");

    // Random-ish weights via deterministic LCG (no extra deps).
    let mut rng_state: u32 = 0xC0FFEEu32;
    let mut next_real = || -> Real {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng_state >> 16) as Real) / 32768.0 - 1.0
    };
    let (w1_n, b1_n, w2_n, b2_n, w3_n, b3_n) = mlp.expected_weight_dims();
    let w1: Vec<Real> = (0..w1_n).map(|_| next_real() * 0.1).collect();
    let b1: Vec<Real> = (0..b1_n).map(|_| next_real() * 0.05).collect();
    let w2: Vec<Real> = (0..w2_n).map(|_| next_real() * 0.1).collect();
    let b2: Vec<Real> = (0..b2_n).map(|_| next_real() * 0.05).collect();
    let w3: Vec<Real> = (0..w3_n).map(|_| next_real() * 0.1).collect();
    let b3: Vec<Real> = (0..b3_n).map(|_| next_real() * 0.05).collect();

    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    mlp.set_weights(
        &device,
        &queue,
        &MlpWeights {
            w1: &w1,
            b1: &b1,
            w2: &w2,
            b2: &b2,
            w3: &w3,
            b3: &b3,
        },
    )
    .unwrap();
    let inputs: Vec<Real> = (0..32).map(|_| next_real()).collect();
    mlp.set_inputs(&queue, &inputs).unwrap();

    // Warmup so pipeline caches are hot.
    for _ in 0..8 {
        mlp.forward(&engine).unwrap();
    }
    // Force a sync via a host download so the wall-clock includes
    // GPU completion — without this we only measure queue submit.
    let _ = mlp.download_outputs(&device, &queue);

    let n_iters = 64;
    let start = Instant::now();
    for _ in 0..n_iters {
        mlp.forward(&engine).unwrap();
    }
    // Sync for an honest end-of-loop time stamp.
    let _ = mlp.download_outputs(&device, &queue);
    let elapsed = start.elapsed();
    let per_call_us = (elapsed.as_secs_f64() * 1.0e6) / (n_iters as f64);
    eprintln!(
        "mlp forward (32→64→64→8) avg over {n_iters} dispatches incl. final sync: {per_call_us:.2} µs"
    );
    // Log-only — don't assert.
}

