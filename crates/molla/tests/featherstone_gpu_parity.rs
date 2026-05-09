//! Featherstone CPU vs GPU parity test. `#[ignore]` because the GPU
//! backend requires a real wgpu adapter — run via:
//!
//! ```bash
//! make test-featherstone-gpu
//! # or:
//! cargo test -p molla --test featherstone_gpu_parity --features gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use molla::actuators::{ControllerPd, ControllerPdConfig};
use molla::compute::{Backend, Engine};
use molla::core::BodyId;
use molla::math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;

fn build_pendulum() -> Model {
    let mut b = ModelBuilder::new();
    let link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    b.add_joint_revolute(
        BodyId::NONE,
        link,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
        V3::Z,
    )
    .unwrap();
    b.build().unwrap()
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn pendulum_cpu_gpu_match_within_1e_3_rad() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    eprintln!("GPU adapter: {:?}", engine.adapter_info());

    let dt = 0.005_f64;
    let n_steps = 200usize;
    let initial_angle = 0.5_f64; // ~28°

    // ── CPU run ──
    let model_cpu = build_pendulum();
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    s_a.joint_q.host_mut().unwrap()[0] = initial_angle;
    let control = model_cpu.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = FeatherstoneSolver::new_cpu();
    let mut cpu_q = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        solver_cpu
            .step(&model_cpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        cpu_q.push(s_a.joint_q.host().unwrap()[0]);
    }

    // ── GPU run ──
    let mut model_gpu = build_pendulum();
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    s_a.joint_q.host_mut().unwrap()[0] = initial_angle;
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model_gpu.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);

    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    let mut gpu_q = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        // GPU buffers chain across iterations: kernel's `copy_buffer`
        // moves state_in.GPU → state_out.GPU then mutates state_out.
        // Re-uploading from host would clobber the chain with stale
        // (post-swap host = pre-step host) data.
        solver_gpu
            .step(&model_gpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);

        let _ = s_a.joint_q.download(&device, &queue);
        gpu_q.push(s_a.joint_q.host().unwrap()[0]);
    }

    // Print first 10 steps for diagnosis.
    eprintln!("step | cpu_q     | gpu_q     | |Δ|");
    for i in 0..10.min(cpu_q.len()) {
        let c = cpu_q[i];
        let g = gpu_q[i];
        eprintln!("{:4} | {:.6} | {:.6} | {:.3e}", i, c, g, (c - g).abs());
    }

    let mut max_diff = 0.0_f64;
    for (i, (c, g)) in cpu_q.iter().zip(gpu_q.iter()).enumerate() {
        let diff = (c - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff >= 1e-2 {
            panic!(
                "step {i}: CPU q = {c:.6}, GPU q = {g:.6}, |Δ| = {diff:.6e}"
            );
        }
    }
    eprintln!(
        "ok: pendulum CPU/GPU match over {n_steps} steps; max |Δq| = {max_diff:.6e}"
    );
}

fn build_pd_pendulum_with_limits() -> (Model, molla::core::JointId) {
    let mut b = ModelBuilder::new();
    let link = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
    );
    let joint = b
        .add_joint_revolute(
            BodyId::NONE,
            link,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
    // Limit + friction.
    b.set_joint_limits(joint, &[-1.0], &[1.0]).unwrap();
    b.set_joint_friction(joint, &[0.5]).unwrap();
    (b.build().unwrap(), joint)
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn pd_pendulum_with_limits_cpu_gpu_match() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    let dt = 0.005_f64;
    let n_steps = 400usize;

    // Same scene on both backends; a PD drive holds the pendulum at
    // q=0.7 (well inside the [-1, 1] limit), with friction damping.
    let configure_control = |control: &mut Control| {
        control.joint_target_mode.host_mut().unwrap()[0] =
            JointTargetMode::Position as u32;
        control.joint_target_pos.host_mut().unwrap()[0] = 0.7;
        control.joint_target_kp.host_mut().unwrap()[0] = 80.0;
        control.joint_target_kd.host_mut().unwrap()[0] = 8.0;
    };

    let (model_cpu, _) = build_pd_pendulum_with_limits();
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    let mut control = model_cpu.control();
    configure_control(&mut control);
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = FeatherstoneSolver::new_cpu();
    for _ in 0..n_steps {
        solver_cpu
            .step(&model_cpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_q = s_a.joint_q.host().unwrap()[0];

    let (mut model_gpu, _) = build_pd_pendulum_with_limits();
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model_gpu.control();
    configure_control(&mut control);
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    for _ in 0..n_steps {
        solver_gpu
            .step(&model_gpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.joint_q.download(&device, &queue);
    let gpu_q = s_a.joint_q.host().unwrap()[0];

    eprintln!("PD+limit pendulum: cpu_q={cpu_q:.6} gpu_q={gpu_q:.6}");
    let diff = (cpu_q - gpu_q).abs();
    assert!(
        diff < 5e-3,
        "CPU/GPU diverged: cpu={cpu_q} gpu={gpu_q} |Δ|={diff}"
    );
    // And both should sit close to the target.
    assert!(
        (cpu_q - 0.7).abs() < 0.1,
        "CPU PD drive missed target: cpu_q={cpu_q}"
    );
}

/// Two-link revolute chain (double pendulum). Exercises ABA's
/// backward articulated-inertia accumulation + Coriolis bias
/// propagation. If parent→child force coupling is wrong on either
/// backend, the two solvers diverge fast.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn double_pendulum_cpu_gpu_match() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    fn build() -> Model {
        let mut b = ModelBuilder::new();
        let l0 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.083,
        }));
        let l1 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.083,
        }));
        b.add_joint_revolute(
            BodyId::NONE,
            l0,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
        b.add_joint_revolute(
            l0,
            l1,
            Transform::from_translation(V3::new(0.0, -0.5, 0.0)),
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
        b.build().unwrap()
    }

    let dt = 0.005;
    let n_steps = 400;
    let q0 = [0.4, -0.3];

    let model_cpu = build();
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    s_a.joint_q.host_mut().unwrap().copy_from_slice(&q0);
    let ctl = model_cpu.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = FeatherstoneSolver::new_cpu();
    for _ in 0..n_steps {
        solver_cpu.step(&model_cpu, &s_a, &ctl, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_q = s_a.joint_q.host().unwrap().to_vec();

    let mut model_gpu = build();
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    s_a.joint_q.host_mut().unwrap().copy_from_slice(&q0);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut ctl = model_gpu.control();
    ctl.upload_to_device(&device, &queue).unwrap();
    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    for _ in 0..n_steps {
        solver_gpu.step(&model_gpu, &s_a, &ctl, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.joint_q.download(&device, &queue);
    let gpu_q = s_a.joint_q.host().unwrap().to_vec();

    let dq0 = (cpu_q[0] - gpu_q[0]).abs();
    let dq1 = (cpu_q[1] - gpu_q[1]).abs();
    eprintln!(
        "double pendulum: cpu=({:.4}, {:.4}) gpu=({:.4}, {:.4}) Δ=({dq0:.2e}, {dq1:.2e})",
        cpu_q[0], cpu_q[1], gpu_q[0], gpu_q[1]
    );
    assert!(dq0 < 1e-2, "joint 0 diverged: cpu={} gpu={}", cpu_q[0], gpu_q[0]);
    assert!(dq1 < 1e-2, "joint 1 diverged: cpu={} gpu={}", cpu_q[1], gpu_q[1]);
}

/// 16 independent single-joint pendulums in one scene. Exercises the
/// per-articulation parallelism of the GPU kernel — each chain runs
/// on its own workgroup invocation.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn many_articulations_cpu_gpu_match() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    fn build(n: usize) -> Model {
        let mut b = ModelBuilder::new();
        for i in 0..n {
            if i > 0 {
                b.begin_articulation();
            }
            let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: Mat3::IDENTITY,
            }));
            b.add_joint_revolute(
                BodyId::NONE,
                link,
                Transform::IDENTITY,
                Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
                V3::Z,
            )
            .unwrap();
        }
        b.build().unwrap()
    }

    let n = 16;
    let dt = 0.005;
    let n_steps = 200;

    let model_cpu = build(n);
    assert_eq!(model_cpu.articulation_count, n);
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    {
        let q = s_a.joint_q.host_mut().unwrap();
        for (i, qi) in q.iter_mut().enumerate() {
            *qi = 0.1 + 0.05 * i as Real; // distinct starts so divergence is visible
        }
    }
    let ctl = model_cpu.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = FeatherstoneSolver::new_cpu();
    for _ in 0..n_steps {
        solver_cpu.step(&model_cpu, &s_a, &ctl, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_q = s_a.joint_q.host().unwrap().to_vec();

    let mut model_gpu = build(n);
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    {
        let q = s_a.joint_q.host_mut().unwrap();
        for (i, qi) in q.iter_mut().enumerate() {
            *qi = 0.1 + 0.05 * i as Real;
        }
    }
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut ctl = model_gpu.control();
    ctl.upload_to_device(&device, &queue).unwrap();
    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    for _ in 0..n_steps {
        solver_gpu.step(&model_gpu, &s_a, &ctl, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.joint_q.download(&device, &queue);
    let gpu_q = s_a.joint_q.host().unwrap().to_vec();

    let mut max_diff = 0.0_f64;
    for (i, (c, g)) in cpu_q.iter().zip(gpu_q.iter()).enumerate() {
        let d = (c - g).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(d < 1e-2, "art {i}: cpu={c} gpu={g} Δ={d}");
    }
    eprintln!("{n} articulations: max |Δq| = {max_diff:.2e}");
}

/// Prismatic-joint single-step parity (we mostly tested revolute up
/// to here). Sliding mass with PD drive.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn prismatic_pd_cpu_gpu_match() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    fn build() -> Model {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 5.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        b.add_joint_prismatic(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            V3::Y,
        )
        .unwrap();
        let mut m = b.build().unwrap();
        m.gravity.host_mut().unwrap()[0] = V3::ZERO; // pure PD test
        m
    }

    let dt = 0.005;
    let n_steps = 300;
    let target = 1.5_f64;

    let configure = |c: &mut Control| {
        c.joint_target_mode.host_mut().unwrap()[0] =
            JointTargetMode::PositionVelocity as u32;
        c.joint_target_pos.host_mut().unwrap()[0] = target;
        c.joint_target_kp.host_mut().unwrap()[0] = 200.0;
        c.joint_target_kd.host_mut().unwrap()[0] = 30.0;
    };

    let model_cpu = build();
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    let mut ctl = model_cpu.control();
    configure(&mut ctl);
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = FeatherstoneSolver::new_cpu();
    for _ in 0..n_steps {
        solver_cpu.step(&model_cpu, &s_a, &ctl, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_q = s_a.joint_q.host().unwrap()[0];

    let mut model_gpu = build();
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model_gpu.state().unwrap();
    let mut s_b = model_gpu.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut ctl = model_gpu.control();
    configure(&mut ctl);
    ctl.upload_to_device(&device, &queue).unwrap();
    let mut solver_gpu = FeatherstoneSolver::new_gpu(&engine, &model_gpu).unwrap();
    for _ in 0..n_steps {
        solver_gpu.step(&model_gpu, &s_a, &ctl, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let _ = s_a.joint_q.download(&device, &queue);
    let gpu_q = s_a.joint_q.host().unwrap()[0];

    eprintln!("prismatic PD: cpu={cpu_q:.6} gpu={gpu_q:.6}");
    assert!(
        (cpu_q - gpu_q).abs() < 1e-3,
        "cpu/gpu prismatic diverged: cpu={cpu_q} gpu={gpu_q}"
    );
    // Target tracking: should be near 1.5 m.
    assert!(
        (cpu_q - target).abs() < 0.05,
        "PD didn't track target on CPU: q={cpu_q}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn joint_limits_clamp_on_gpu() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let (mut model, _) = build_pd_pendulum_with_limits();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    // Start at the upper limit with a velocity that would otherwise
    // push past it.
    s_a.joint_q.host_mut().unwrap()[0] = 1.0;
    s_a.joint_qd.host_mut().unwrap()[0] = 5.0;
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    let mut max_q = -10.0_f64;
    for _ in 0..1000 {
        solver
            .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let _ = s_a.joint_q.download(&device, &queue);
        let q = s_a.joint_q.host().unwrap()[0];
        if q > max_q {
            max_q = q;
        }
    }
    assert!(
        max_q <= 1.0 + 1e-3,
        "GPU joint exceeded upper limit: max_q={max_q}"
    );
}

/// Six-link revolute arm anchored at the world frame with `+X` joint
/// axis — the viewer's `Scene::Arm` setup exactly. ControllerPd writes
/// `joint_force` from a constant non-zero target and Featherstone
/// integrates the chain. We assert the joint coordinates AND the
/// downloaded `body_q` poses both move noticeably away from initial
/// values, which catches the bug class "kernel runs but FK doesn't
/// reach `body_q.host()`".
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-featherstone-gpu`"]
fn arm_six_link_pd_drives_motion_on_gpu() {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    const N: usize = 6;
    const HALF_H: f64 = 0.3;
    const RADIUS: f64 = 0.06;
    const BASE_HEIGHT: f64 = 5.0;
    const MASS: f64 = 0.5;

    fn capsule_inertia(mass: f64, half_h: f64, r: f64) -> Mat3 {
        let r2 = r * r;
        let h2 = (half_h * 2.0).powi(2);
        let i_xx = mass * (r2 * 0.25 + h2 / 12.0);
        let i_yy = mass * (r2 * 0.5);
        Mat3::from_diagonal(V3::new(i_xx, i_yy, i_xx))
    }
    let inertia = capsule_inertia(MASS, HALF_H, RADIUS);

    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.1);
    b.begin_articulation();
    let mut prev_body = BodyId::NONE;
    let mut next_pos = V3::new(0.0, BASE_HEIGHT - HALF_H, 0.0);
    for i in 0..N {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia { mass: MASS, com: V3::ZERO, inertia })
                .with_position(next_pos),
        );
        let x_p = if i == 0 {
            // Flip joint-local +Y to world -Y so the rest pose hangs down.
            let flip = molla::math::Quat::from_axis_angle(V3::X, std::f64::consts::PI);
            Transform::new(V3::new(0.0, BASE_HEIGHT, 0.0), flip)
        } else {
            Transform::from_translation(V3::new(0.0, HALF_H, 0.0))
        };
        let x_c = Transform::from_translation(V3::new(0.0, -HALF_H, 0.0));
        let axis = if i == 0 { V3::Z } else { V3::X };
        b.add_joint_revolute(prev_body, body, x_p, x_c, axis).unwrap();
        prev_body = body;
        next_pos -= V3::new(0.0, HALF_H * 2.0, 0.0);
    }
    let mut model = b.build().unwrap();
    model.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let initial_body_q = s_a.body_q.host().unwrap().to_vec();
    let initial_joint_q = s_a.joint_q.host().unwrap().to_vec();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();

    let mut control = model.control();
    // Constant non-zero target on every joint — drives the arm to bend.
    let target = 0.5_f64;
    for v in control.joint_target_pos.host_mut().unwrap() {
        *v = target;
    }
    control.upload_to_device(&device, &queue).unwrap();

    let configs: Vec<ControllerPdConfig> = (0..N)
        .map(|i| ControllerPdConfig {
            dof: i as u32,
            kp: 120.0,
            kd: 12.0,
            max_effort: 80.0,
        })
        .collect();
    let mut controller = ControllerPd::new(&engine, &model, &configs).unwrap();

    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    let dt = 1.0_f64 / 60.0 / 4.0; // viewer FIXED_DT / SUBSTEPS_PER_FRAME ≈ 0.0042 s

    // 60 frames × 4 substeps = ~1 s of simulated time.
    for _ in 0..240 {
        controller.step(&engine, &s_a, &mut control).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let _ = s_a.body_q.download(&device, &queue);
        let _ = s_a.joint_q.download(&device, &queue);
    }

    let final_body_q = s_a.body_q.host().unwrap();
    let final_joint_q = s_a.joint_q.host().unwrap();

    // Joint coords should have moved measurably toward the target.
    let mut max_dq = 0.0_f64;
    for (qf, qi) in final_joint_q.iter().zip(initial_joint_q.iter()) {
        max_dq = max_dq.max((qf - qi).abs());
    }
    eprintln!(
        "arm: joint_q final = {final_joint_q:?}  max |Δq| = {max_dq:.4}"
    );
    assert!(
        max_dq > 0.05,
        "controller didn't move any joint after 1 s: max |Δq|={max_dq}"
    );

    // Body world poses should differ from the initial poses too —
    // catches FK-not-reaching-host bug.
    let mut max_dp = 0.0_f64;
    for (i, (pf, pi)) in final_body_q.iter().zip(initial_body_q.iter()).enumerate() {
        let d = (pf.position - pi.position).length();
        eprintln!(
            "  body {i}: init y={:.3} → final y={:.3}  |Δ|={:.4}",
            pi.position.y, pf.position.y, d
        );
        max_dp = max_dp.max(d);
    }
    assert!(
        max_dp > 0.05,
        "FK never updated body_q: max |Δpos|={max_dp}"
    );
}
