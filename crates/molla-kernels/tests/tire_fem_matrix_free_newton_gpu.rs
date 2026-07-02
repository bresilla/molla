//! Hardware integration test for the matrix-free Newton tire-FEM debug
//! timestep WGSL kernel.
//!
//! Ignored by default because it needs a real wgpu adapter:
//! `cargo test -p molla-kernels --features gpu --test tire_fem_matrix_free_newton_gpu -- --ignored`.
//!
//! Test strategy: build a trivial 4-node / 1-tetra mesh with zero
//! Neo-Hookean stiffness (μ = λ = 0). With internal forces zeroed, the
//! per-node residual collapses to
//!
//!     R(node) = m · ((x − x₀) / dt − v₀) − dt · (external − damping · v)
//!
//! which is straightforward to oracle on the CPU. The GPU dispatch is
//! exercised end-to-end: `residual_eval`, `jv_probe`, `diagonal_extract`,
//! `apply_correction` all run through one `dispatch_blocking` round-trip,
//! and every output buffer is compared against the closed-form CPU answer.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireFemMatrixFreeNewtonGpuInput, TireFemMatrixFreeNewtonGpuKernel,
    tire_fem_matrix_free_newton_gpu_contract,
};

fn unbalanced_input() -> TireFemMatrixFreeNewtonGpuInput {
    // Same mesh as tiny_input but with a nonzero v0 to give the residual a
    // real magnitude before Newton runs.
    let mut input = tiny_input();
    input.v0_vel = (0..4)
        .map(|i| [0.05 * i as f32, -0.04, 0.02 * (i as f32 + 1.0), 0.0])
        .collect();
    // Bump x0 slightly off rest so the candidate (= rest) is not exactly
    // x0; the residual then has both inertia and external contributions.
    for (i, p) in input.x0_pos.iter_mut().enumerate() {
        p[0] -= 0.001 * (i as f32 + 1.0);
        p[1] += 0.002 * (i as f32 + 1.0);
        p[2] -= 0.0015 * (i as f32 + 1.0);
    }
    input
}

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn flat_external() -> [f32; 4] {
    [0.0, 0.0, 0.0, 0.0]
}

/// Build the simplest valid input the WGSL contract accepts: four nodes that
/// form one unit tetra with zero stiffness, zero initial velocity, and a
/// per-node external force pattern that gives a clean oracle.
fn tiny_input() -> TireFemMatrixFreeNewtonGpuInput {
    // Rest positions of a regular tetra at the origin. The four nodes are
    // placed so the rest tet has positive volume.
    let rest = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    // Shape-function gradients ∇N_i for the regular unit tet match
    // (−1,−1,−1), (1,0,0), (0,1,0), (0,0,1). Volume = 1/6.
    let grads = [
        [-1.0_f32, -1.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let external: Vec<[f32; 4]> = (0..4)
        .map(|i| {
            let f = (i + 1) as f32;
            [0.7 * f, -0.4 * f, 0.5 * f, 0.0]
        })
        .collect();
    let direction: Vec<[f32; 4]> = (0..4)
        .map(|i| [0.02 * (i as f32 + 1.0), -0.015, 0.01, 0.0])
        .collect();
    TireFemMatrixFreeNewtonGpuInput {
        dt: 0.01,
        node_damping: 0.0,
        jv_epsilon: 1.0e-4,
        line_search_alpha: 0.5,
        rest_pos: rest.to_vec(),
        x0_pos: rest.to_vec(),
        v0_vel: vec![flat_external(); 4],
        mass: vec![1.0; 4],
        external,
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: grads.to_vec(),
        tet_volume: vec![1.0 / 6.0],
        // μ = λ = 0 zeros out the Neo-Hookean tetra force so the residual
        // collapses to the trivial closed-form below.
        tet_mu: vec![0.0],
        tet_lambda: vec![0.0],
        candidate_pos: rest.to_vec(),
        direction,
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_matrix_free_newton_gpu_residual_matches_cpu_oracle() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping matrix-free Newton GPU dispatch: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemMatrixFreeNewtonGpuKernel::new(&device);
    let input = tiny_input();
    let contract = tire_fem_matrix_free_newton_gpu_contract(input.node_count() as u32, 16);
    assert_eq!(contract.node_count, 4);
    assert_eq!(contract.coupled_dof_count, 12);

    let output = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("dispatch matrix-free Newton kernel");
    assert_eq!(output.residual.len(), 4);
    assert_eq!(output.jv.len(), 4);
    assert_eq!(output.diagonal.len(), 4);
    assert_eq!(output.pos.len(), 4);

    // CPU oracle: with x = x0, μ = λ = 0, damping = 0, v₀ = 0,
    //   v_new = (x − x0)/dt = 0
    //   R     = m·(0 − v0) − dt·(internal + external − damping·v_new)
    //         = −dt · external  (since internal = 0 and damping·v_new = 0)
    let dt = input.dt;
    for (i, &expected_external) in input.external.iter().enumerate() {
        for axis in 0..3 {
            let want = -dt * expected_external[axis];
            let got = output.residual[i][axis];
            assert!(
                (got - want).abs() < 1.0e-4,
                "residual mismatch at node {i} axis {axis}: gpu={got} cpu_oracle={want}"
            );
        }
    }

    // CPU oracle for jv_probe: with internal = 0 and damping = 0,
    //   R(x) = m · (x − x0) / dt − m · v0 − dt · external
    // The Jacobian wrt x is m/dt · I. Hence J·v[axis] = m / dt · direction[axis].
    let m_over_dt = 1.0_f32 / dt;
    for (i, dir) in input.direction.iter().enumerate() {
        for axis in 0..3 {
            let want = m_over_dt * dir[axis];
            let got = output.jv[i][axis];
            assert!(
                (got - want).abs() < 0.1,
                "jv_probe mismatch at node {i} axis {axis}: gpu={got} cpu_oracle={want}"
            );
        }
    }

    // CPU oracle for diagonal_extract: per-DoF diagonal of J = m/dt for the
    // trivial system above.
    for (i, diag) in output.diagonal.iter().enumerate() {
        for axis in 0..3 {
            let got = diag[axis];
            assert!(
                (got - m_over_dt).abs() < 0.1,
                "diagonal mismatch at node {i} axis {axis}: gpu={got} cpu_oracle={m_over_dt}"
            );
        }
    }

    // apply_correction: pos_out = candidate + alpha * direction.
    let alpha = input.line_search_alpha;
    for (i, ((candidate, dir), pos)) in input
        .candidate_pos
        .iter()
        .zip(input.direction.iter())
        .zip(output.pos.iter())
        .enumerate()
    {
        for axis in 0..3 {
            let want = candidate[axis] + alpha * dir[axis];
            let got = pos[axis];
            assert!(
                (got - want).abs() < 1.0e-5,
                "apply_correction mismatch at node {i} axis {axis}: gpu={got} cpu_oracle={want}"
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_matrix_free_newton_gpu_outer_iteration_reduces_residual() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping matrix-free Newton outer iteration: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemMatrixFreeNewtonGpuKernel::new(&device);
    let input = unbalanced_input();
    let outcome = kernel
        .run_newton_outer_blocking(&device, &queue, &input, 8)
        .expect("orchestrated Newton outer iteration");
    assert!(
        outcome.initial_residual_norm.is_finite()
            && outcome.final_residual_norm.is_finite()
            && outcome.final_residual_norm <= outcome.initial_residual_norm,
        "host-orchestrated Newton outer iteration must not increase residual: \
         initial={} final={} alpha={} iters={}",
        outcome.initial_residual_norm,
        outcome.final_residual_norm,
        outcome.accepted_alpha,
        outcome.krylov_iteration_count
    );
    assert!(
        outcome
            .final_pos
            .iter()
            .all(|p| { p[0].is_finite() && p[1].is_finite() && p[2].is_finite() }),
        "final position must be finite after Newton outer iteration"
    );
    assert!(
        outcome
            .diagonal
            .iter()
            .all(|d| d[0].is_finite() && d[1].is_finite() && d[2].is_finite()),
        "Jacobi diagonal sample must be finite"
    );
    assert!(
        outcome.krylov_iteration_count > 0,
        "expected at least one GMRES inner iteration"
    );
    // Accept the case where the diagonal-only solve already drives the
    // residual to zero (`accepted_alpha = 0`); otherwise the line search
    // must have picked one of the allowed alphas.
    let alpha = outcome.accepted_alpha;
    assert!(
        alpha == 0.0
            || (alpha - 1.0).abs() < 1.0e-6
            || (alpha - 0.5).abs() < 1.0e-6
            || (alpha - 0.25).abs() < 1.0e-6
            || (alpha - 0.125).abs() < 1.0e-6
            || (alpha - 0.0625).abs() < 1.0e-6,
        "accepted line-search alpha must be from the CPU helper sweep: got {alpha}"
    );
}
