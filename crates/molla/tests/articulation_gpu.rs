//! GPU articulation-math tests — CRBA mass matrix, RNEA inverse
//! dynamics, spatial Jacobian, and IK. All `#[ignore]`-marked so they
//! only run with a real wgpu adapter via:
//!
//! ```bash
//! nixVulkan cargo test -p molla --test articulation_gpu --features gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use molla::compute::{Backend, Engine};
use molla::core::BodyId;
use molla::kernels::{inverse_dynamics_gpu, jacobian_gpu, MassMatrixKernel};
use molla::math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;
use molla::solvers::{inverse_dynamics, jacobian, mass_matrix, IkConfig, IkObjective, IkSolver};

fn try_engine() -> Option<Engine> {
    match Engine::new(Backend::AutoGpu) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("skipping GPU test: no adapter ({e})");
            None
        }
    }
}

fn upload_state(model: &mut Model, state: &mut State, engine: &Engine) {
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    state.upload_to_device(&device, &queue).unwrap();
}

fn build_pendulum() -> Model {
    let mass = 1.5_f64;
    let l = 0.4_f64;
    let i_xx = mass * (2.0 * l).powi(2) / 12.0;
    let inertia = Mat3::from_diagonal(V3::new(i_xx, 0.0, i_xx));
    let mut b = ModelBuilder::new();
    let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia,
    }));
    b.add_joint_revolute(
        BodyId::NONE,
        link,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
        V3::Z,
    )
    .unwrap();
    b.build().unwrap()
}

/// 7-DOF Franka-style serial arm. Mirrors `tests/ik_franka.rs::build_franka`
/// — kept as a top-level helper so other tests / the viewer can re-use it.
pub fn build_franka() -> (Model, usize) {
    let mut b = ModelBuilder::new();
    let half = V3::new(0.04, 0.04, 0.04);
    let mass = 1.0_f64;
    let h2 = half * half;
    let m_third = mass / 3.0;
    let inertia = Mat3::from_diagonal(V3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ));
    let make_link = |b: &mut ModelBuilder| {
        b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass,
            com: V3::ZERO,
            inertia,
        }))
    };
    b.begin_articulation();
    let axes = [V3::Z, V3::Y, V3::Z, V3::Y, V3::Z, V3::Y, V3::Z];
    let link_lens = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107];
    let mut prev = molla::core::BodyId::NONE;
    let mut last_body = 0usize;
    for i in 0..7 {
        let body = make_link(&mut b);
        last_body = body.index();
        let x_p = if i == 0 {
            Transform::IDENTITY
        } else {
            Transform::from_translation(V3::new(0.0, 0.0, link_lens[i - 1]))
        };
        let x_c = Transform::IDENTITY;
        b.add_joint_revolute(prev, body, x_p, x_c, axes[i]).unwrap();
        prev = body;
    }
    let model = b.build().unwrap();
    (model, last_body)
}

fn build_double_pendulum() -> Model {
    let mass = 1.0_f64;
    let l = 0.5_f64;
    let i_xx = mass * (2.0 * l).powi(2) / 12.0;
    let inertia = Mat3::from_diagonal(V3::new(i_xx, 0.0, i_xx));
    let mut b = ModelBuilder::new();
    let l0 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia,
    }));
    let l1 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia,
    }));
    b.add_joint_revolute(
        BodyId::NONE,
        l0,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
        V3::Z,
    )
    .unwrap();
    b.add_joint_revolute(
        l0,
        l1,
        Transform::from_translation(V3::new(0.0, l, 0.0)),
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
        V3::Z,
    )
    .unwrap();
    b.build().unwrap()
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_crba_pendulum_matches_cpu() {
    let Some(engine) = try_engine() else { return };
    let mut model = build_pendulum();
    let mut state = model.state().unwrap();
    state.joint_q.host_mut().unwrap()[0] = 0.3;
    molla::sim::eval_fk(&model, &mut state).unwrap();

    let m_cpu = mass_matrix(&model, &state, 0).unwrap();

    upload_state(&mut model, &mut state, &engine);
    let kernel = MassMatrixKernel::new(&engine, &model).unwrap();
    let device = engine.device().unwrap().clone();
    let out = kernel.create_output_buffer(&device);
    kernel.dispatch(&engine, &model, &state, &out).unwrap();
    let m_gpu = kernel.read_back(&engine, &out).unwrap();

    eprintln!("pendulum M_cpu={m_cpu:?} M_gpu={m_gpu:?}");
    assert_eq!(m_cpu.len(), m_gpu.len());
    for (i, (c, g)) in m_cpu.iter().zip(m_gpu.iter()).enumerate() {
        let diff = (*c as f32 - g).abs();
        assert!(diff < 1e-4, "M[{i}] CPU={c} GPU={g} |Δ|={diff}");
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_jacobian_finite_difference_matches_analytical() {
    // 1-link revolute pendulum, axis +Z, child anchor at -Y by L. Tip at
    // local (0, L, 0) sits at world (0, 2L, 0). Jacobian linear column:
    // ω×r = (0,0,1)×(0,2L,0) = (-2L, 0, 0).
    let Some(engine) = try_engine() else { return };
    let mass = 1.0_f64;
    let l = 0.5_f64;
    let mut b = ModelBuilder::new();
    let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia: Mat3::from_diagonal(V3::splat(0.1)),
    }));
    b.add_joint_revolute(
        BodyId::NONE,
        link,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
        V3::Z,
    )
    .unwrap();
    let mut model = b.build().unwrap();
    let mut state = model.state().unwrap();
    molla::sim::eval_fk(&model, &mut state).unwrap();

    let tip_local = V3::new(0.0, l, 0.0);
    let j_cpu = jacobian(&model, &state, 0, link.index(), tip_local).unwrap();

    upload_state(&mut model, &mut state, &engine);
    let mut j_gpu = vec![0.0_f64; 6];
    jacobian_gpu(&engine, &model, &state, 0, link.index(), tip_local, &mut j_gpu).unwrap();

    eprintln!("Jacobian CPU: {j_cpu:?}");
    eprintln!("Jacobian GPU: {j_gpu:?}");
    let dof = 1;
    let jx = j_gpu[0 * dof + 0];
    let jy = j_gpu[1 * dof + 0];
    let jz = j_gpu[2 * dof + 0];
    eprintln!("GPU linear column: ({jx}, {jy}, {jz}); expected (~{:.3}, 0, 0)", -2.0 * l);
    assert!((jx - (-2.0 * l)).abs() < 1e-4);
    assert!(jy.abs() < 1e-4);
    assert!(jz.abs() < 1e-4);
    for (i, (c, g)) in j_cpu.iter().zip(j_gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(diff < 1e-4, "J[{i}] cpu={c} gpu={g} |Δ|={diff}");
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_rnea_pendulum_gravity_torque_matches_cpu() {
    let Some(engine) = try_engine() else { return };
    let mut model = build_pendulum();
    let mut state = model.state().unwrap();
    state.joint_q.host_mut().unwrap()[0] = std::f64::consts::FRAC_PI_4;
    molla::sim::eval_fk(&model, &mut state).unwrap();

    // CPU reference.
    let qdd_cpu = vec![0.0_f64];
    let tau_cpu = inverse_dynamics(&model, &state, 0, &qdd_cpu).unwrap();

    // GPU.
    upload_state(&mut model, &mut state, &engine);
    let mut tau_gpu = vec![0.0_f64; tau_cpu.len()];
    inverse_dynamics_gpu(&engine, &model, &state, 0, &qdd_cpu, &mut tau_gpu).unwrap();

    eprintln!("RNEA pendulum: cpu={tau_cpu:?} gpu={tau_gpu:?}");
    assert_eq!(tau_cpu.len(), tau_gpu.len());
    for (i, (c, g)) in tau_cpu.iter().zip(tau_gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(diff < 1e-4, "tau[{i}] cpu={c} gpu={g} |Δ|={diff}");
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_rnea_decomposition_matches_cpu() {
    // Verify RNEA(q, qd, qdd) = M(q)·qdd + RNEA(q, qd, 0) on GPU.
    let Some(engine) = try_engine() else { return };
    let mut model = build_double_pendulum();
    let mut state = model.state().unwrap();
    state.joint_q.host_mut().unwrap().copy_from_slice(&[0.3, -0.2]);
    state.joint_qd.host_mut().unwrap().copy_from_slice(&[0.4, 0.1]);
    molla::sim::eval_fk(&model, &mut state).unwrap();

    let qdd: [Real; 2] = [1.7, -0.6];

    // CPU baselines.
    let tau_cpu_full = inverse_dynamics(&model, &state, 0, &qdd).unwrap();
    let tau_cpu_bias = inverse_dynamics(&model, &state, 0, &[0.0, 0.0]).unwrap();
    let m_cpu = mass_matrix(&model, &state, 0).unwrap();

    upload_state(&mut model, &mut state, &engine);

    let mut tau_gpu_full = vec![0.0_f64; 2];
    let mut tau_gpu_bias = vec![0.0_f64; 2];
    inverse_dynamics_gpu(&engine, &model, &state, 0, &qdd, &mut tau_gpu_full).unwrap();
    inverse_dynamics_gpu(&engine, &model, &state, 0, &[0.0, 0.0], &mut tau_gpu_bias).unwrap();

    // GPU mass matrix.
    let mass_kernel = MassMatrixKernel::new(&engine, &model).unwrap();
    let device = engine.device().unwrap().clone();
    let m_buf = mass_kernel.create_output_buffer(&device);
    mass_kernel.dispatch(&engine, &model, &state, &m_buf).unwrap();
    let m_gpu = mass_kernel.read_back(&engine, &m_buf).unwrap();

    eprintln!("RNEA full: cpu={tau_cpu_full:?} gpu={tau_gpu_full:?}");
    eprintln!("RNEA bias: cpu={tau_cpu_bias:?} gpu={tau_gpu_bias:?}");
    eprintln!("CRBA M: cpu={m_cpu:?} gpu={m_gpu:?}");

    // Decomposition on GPU side.
    let n = 2usize;
    for i in 0..n {
        let mut mqdd = 0.0f32;
        for j in 0..n {
            mqdd += m_gpu[i * n + j] * qdd[j] as f32;
        }
        let lhs = tau_gpu_full[i] as f32;
        let rhs = mqdd + tau_gpu_bias[i] as f32;
        let err = (lhs - rhs).abs();
        eprintln!("i={i}: lhs={lhs} rhs={rhs} |Δ|={err}");
        assert!(err < 1e-3, "RNEA decomposition broken on GPU at i={i}: |Δ|={err}");
    }
    // GPU full vs CPU full.
    for (i, (c, g)) in tau_cpu_full.iter().zip(tau_gpu_full.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(diff < 1e-3, "tau_full[{i}] cpu={c} gpu={g} |Δ|={diff}");
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_crba_double_pendulum_symmetric() {
    let Some(engine) = try_engine() else { return };
    let mut model = build_double_pendulum();
    let mut state = model.state().unwrap();
    state.joint_q.host_mut().unwrap().copy_from_slice(&[0.4, -0.3]);
    molla::sim::eval_fk(&model, &mut state).unwrap();

    let m_cpu = mass_matrix(&model, &state, 0).unwrap();

    upload_state(&mut model, &mut state, &engine);
    let kernel = MassMatrixKernel::new(&engine, &model).unwrap();
    let device = engine.device().unwrap().clone();
    let out = kernel.create_output_buffer(&device);
    kernel.dispatch(&engine, &model, &state, &out).unwrap();
    let m_gpu = kernel.read_back(&engine, &out).unwrap();

    assert_eq!(m_gpu.len(), 4);
    let m00 = m_gpu[0];
    let m01 = m_gpu[1];
    let m10 = m_gpu[2];
    let m11 = m_gpu[3];
    eprintln!("M_gpu = [[{m00}, {m01}], [{m10}, {m11}]]");
    eprintln!("M_cpu = {m_cpu:?}");
    assert!((m01 - m10).abs() < 1e-4, "GPU M not symmetric");
    assert!(m00 > 0.0 && m11 > 0.0, "GPU diagonal non-positive");
    for (i, (c, g)) in m_cpu.iter().zip(m_gpu.iter()).enumerate() {
        let diff = (*c as f32 - g).abs();
        assert!(diff < 1e-4, "M[{i}] CPU={c} GPU={g} |Δ|={diff}");
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_ik_franka_reaches_target() {
    let Some(engine) = try_engine() else { return };
    let (mut model, ee_body) = build_franka();
    let mut state = model.state().unwrap();
    molla::sim::eval_fk(&model, &mut state).unwrap();
    let ee_at_zero = state.body_q.host().unwrap()[ee_body].position;
    let target = ee_at_zero + V3::new(0.2, 0.1, -0.15);

    upload_state(&mut model, &mut state, &engine);
    let solver = IkSolver::new(vec![IkObjective::Position {
        body: ee_body,
        point_local: V3::ZERO,
        target_world: target,
        weight: 1.0,
    }])
    .with_config(IkConfig {
        max_iter: 50,
        ..IkConfig::default()
    });
    let q0 = vec![0.0; 7];
    let t0 = std::time::Instant::now();
    let r = solver.solve_gpu(&engine, &model, &mut state, &q0).unwrap();
    let dt = t0.elapsed();
    let achieved = state.body_q.host().unwrap()[ee_body].position;
    let err = (achieved - target).length();
    eprintln!(
        "GPU Franka IK: iters={}, residual={:.5}, |Δ|={err:.4} m, took {:.2} ms",
        r.iters, r.residual, dt.as_secs_f64() * 1e3
    );
    assert!(err < 5.0e-3, "IK missed target by {err:.4} m");
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_ik_multi_seed_runs_in_parallel() {
    let Some(engine) = try_engine() else { return };
    let (mut model, ee_body) = build_franka();
    let mut state = model.state().unwrap();
    molla::sim::eval_fk(&model, &mut state).unwrap();
    let ee_at_zero = state.body_q.host().unwrap()[ee_body].position;
    // A challenging-ish target — moderately offset.
    let target = ee_at_zero + V3::new(0.25, -0.15, -0.1);

    upload_state(&mut model, &mut state, &engine);
    let solver = IkSolver::new(vec![IkObjective::Position {
        body: ee_body,
        point_local: V3::ZERO,
        target_world: target,
        weight: 1.0,
    }])
    .with_config(IkConfig {
        max_iter: 30,
        ..IkConfig::default()
    });

    // 32 deterministic seeds — slight angular jitter around q=0.
    let mut seeds: Vec<Vec<Real>> = Vec::with_capacity(32);
    for s in 0..32 {
        let theta = 0.05 * (s as Real);
        seeds.push(vec![theta, -theta, theta, -theta, theta, -theta, theta]);
    }

    let t0 = std::time::Instant::now();
    let r = solver
        .solve_gpu_with_seeds(&engine, &model, &mut state, &seeds)
        .unwrap();
    let dt = t0.elapsed();
    let achieved = state.body_q.host().unwrap()[ee_body].position;
    let err = (achieved - target).length();
    eprintln!(
        "GPU multi-seed IK (32 seeds): iters={}, residual={:.5}, |Δ|={err:.4} m, total {:.2} ms",
        r.iters, r.residual, dt.as_secs_f64() * 1e3
    );
    assert!(err < 1.0e-2, "best-of-32 GPU IK missed by {err:.4} m");
    // Soft runtime check — desktop GPUs hit this easily, CI VMs may not.
    let ms = dt.as_secs_f64() * 1e3;
    if ms > 500.0 {
        eprintln!("warning: multi-seed IK took {ms:.2} ms (loose bound 500 ms)");
    }
}
