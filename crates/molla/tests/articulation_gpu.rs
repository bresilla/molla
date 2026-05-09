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
use molla::kernels::MassMatrixKernel;
use molla::math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;
use molla::solvers::mass_matrix;

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
