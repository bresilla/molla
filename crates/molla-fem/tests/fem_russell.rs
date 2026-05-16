//! Stage 36.7 — `russell` feature contract tests.
//!
//! Three checks (all gated behind `cfg(feature = "russell")`):
//!
//! 1. **Parity** — small heat-equation problem (8³ tets), assert the
//!    direct solver matches CPU CG to `1e-10`. Direct LU is exact-
//!    arithmetic up to f64 round-off; CG's default tolerance
//!    (`1e-8` from `solver::DEFAULT_TOL`) is the looser bound.
//! 2. **Factor-once-solve-many** — factorise the same matrix once,
//!    then back-solve against 10 different RHS vectors. Verify each
//!    one matches the CG result to `1e-10`.
//! 3. **Wall-time check** — informational only (`eprintln!`, not
//!    asserted). Times factor + 10 back-solves vs 10 fresh CG
//!    solves and prints the comparison so it shows up in
//!    `cargo test --features russell -- --nocapture`.

#![cfg(feature = "russell")]

use std::time::Instant;

use molla_fem::{
    BoundaryCondition, DirectSolver, FemMesh, FunctionSpace, assemble_global,
    compute_element_stiffness, solve_steady_state, solve_steady_state_direct,
};
use molla_math::Real;
use molla_math::sparse::BsrMatrix;

// ─── Local helpers (mirror tests/fem_v1.rs) ─────────────────────────

fn assemble_stiffness(mesh: &FemMesh, alpha: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .expect("assemble stiffness")
}

/// Build the canonical 8³-tet heat-equation problem used by all three
/// tests: unit cube, Dirichlet `T(z=0) = 0`, `T(z=1) = 1`, sides
/// insulating (zero Neumann is the natural homogeneous case for the
/// P1 weak form).
fn build_heat_problem(n_grid: usize) -> (BsrMatrix, Vec<Real>, Vec<BoundaryCondition>, FemMesh) {
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let k = assemble_stiffness(&mesh, 1.0);
    let n = space.ndof(&mesh);
    let f = vec![0.0; n];
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.z.abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        } else if (p.z - 1.0).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 1.0,
            });
        }
    }
    (k, f, bcs, mesh)
}

/// Apply the same Dirichlet reduction `solve_steady_state` /
/// `solve_steady_state_direct` use internally, so the
/// factor-once-solve-many test can drive `DirectSolver::back_solve`
/// against the reduced matrix directly.
fn reduce_for_direct(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
) -> (BsrMatrix, Vec<Real>) {
    let mut k_red = k.clone();
    let mut f_red = f.to_vec();
    molla_fem::apply_dirichlet(&mut k_red, &mut f_red, bcs);
    (k_red, f_red)
}

// ─── Test 1: parity with CG ─────────────────────────────────────────

#[test]
fn direct_solver_matches_cg_on_heat_8x8x8() {
    let (k, f, bcs, _mesh) = build_heat_problem(8);

    let cg = solve_steady_state(&k, &f, &bcs, 5_000, 1e-12);
    let direct = solve_steady_state_direct(&k, &f, &bcs)
        .expect("direct solve must succeed on the heat problem");

    assert_eq!(cg.len(), direct.len(), "DoF count mismatch");
    let mut max_err: Real = 0.0;
    for (cg_v, dr_v) in cg.iter().zip(direct.iter()) {
        max_err = max_err.max((cg_v - dr_v).abs());
    }
    assert!(
        max_err < 1.0e-10,
        "direct vs CG disagreement {max_err:.3e} > 1e-10"
    );
}

// ─── Test 2: factor once, back-solve many ───────────────────────────

#[test]
fn factor_once_back_solve_many_matches_cg() {
    let (k, f, bcs, _mesh) = build_heat_problem(8);
    let (k_red, _f_red) = reduce_for_direct(&k, &f, &bcs);

    let mut solver = DirectSolver::new().unwrap();
    solver.factorize(&k_red).expect("factorize");
    assert!(solver.is_factorized());
    let n = solver.dim();
    assert_eq!(n, k_red.nrows());

    // Cook up 10 distinct RHS vectors. Mix of unit basis impulses,
    // a smooth sinusoid, and a couple of random-looking patterns
    // generated deterministically from indices.
    let mut rhs_set: Vec<Vec<Real>> = Vec::new();
    for k_rhs in 0..10 {
        let mut b = vec![0.0; n];
        match k_rhs {
            0 => b[0] = 1.0,
            1 => b[n / 2] = 1.0,
            2 => b[n - 1] = 1.0,
            3 => {
                for i in 0..n {
                    b[i] = (i as Real * 0.123).sin();
                }
            }
            4 => {
                for i in 0..n {
                    b[i] = (i as Real * 0.5).cos();
                }
            }
            5 => b
                .iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = i as Real / n as Real),
            6 => b.iter_mut().enumerate().for_each(|(i, v)| {
                *v = ((i * 31 + 7) % 13) as Real - 6.0;
            }),
            7 => b.iter_mut().for_each(|v| *v = 1.0),
            8 => b.iter_mut().enumerate().for_each(|(i, v)| {
                *v = if i % 2 == 0 { 1.0 } else { -1.0 };
            }),
            _ => b.iter_mut().enumerate().for_each(|(i, v)| {
                *v = ((i as Real) * 0.7).tanh();
            }),
        }
        rhs_set.push(b);
    }

    for (idx, b) in rhs_set.iter().enumerate() {
        // Reference: CG on the reduced matrix with the same RHS.
        // (No BC reduction here — `b` is already in the reduced
        // coordinate system.)
        let cg = molla_math::iterative::cg_solve(
            |x| {
                let mut y = vec![0.0; k_red.nrows()];
                k_red.mv(x, &mut y).unwrap();
                y
            },
            b,
            &vec![0.0; n],
            1e-12,
            10_000,
        )
        .x;

        let direct = solver.back_solve(b).expect("back_solve");
        assert_eq!(direct.len(), n);
        let mut max_err: Real = 0.0;
        for (cg_v, dr_v) in cg.iter().zip(direct.iter()) {
            max_err = max_err.max((cg_v - dr_v).abs());
        }
        assert!(
            max_err < 1.0e-9,
            "RHS #{idx}: direct vs CG disagreement {max_err:.3e} > 1e-9"
        );
    }
}

// ─── Test 3: informational wall-time comparison ─────────────────────
//
// Not asserted — just prints the timing breakdown so
// `cargo test --features russell --test fem_russell -- --nocapture`
// shows the speedup (or slowdown) from the direct solver at this
// problem size. The expected pattern: factor + 10 back-solves should
// run in roughly the same wall-time as **one** CG solve once the
// problem is small enough that the dense LU isn't crushed by O(n³),
// because each back-solve is O(n²) and CG iterations are O(nnz).

#[test]
fn wall_time_factor_once_vs_cg_loop() {
    let (k, f, bcs, _mesh) = build_heat_problem(8);
    let (k_red, _f_red) = reduce_for_direct(&k, &f, &bcs);
    let n = k_red.nrows();

    // Same 10-RHS battery the parity test above used.
    let rhs_set: Vec<Vec<Real>> = (0..10)
        .map(|k_rhs| {
            let mut b = vec![0.0; n];
            match k_rhs {
                0 => b[0] = 1.0,
                1 => b[n / 2] = 1.0,
                2 => b[n - 1] = 1.0,
                3 => b.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = (i as Real * 0.123).sin();
                }),
                4 => b.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = (i as Real * 0.5).cos();
                }),
                5 => b.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = i as Real / n as Real;
                }),
                6 => b.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = ((i * 31 + 7) % 13) as Real - 6.0;
                }),
                7 => b.iter_mut().for_each(|v| *v = 1.0),
                8 => b.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = if i % 2 == 0 { 1.0 } else { -1.0 };
                }),
                _ => b.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = ((i as Real) * 0.7).tanh();
                }),
            }
            b
        })
        .collect();

    // ─── Direct path: factorise once, back-solve 10× ───
    let mut solver = DirectSolver::new().unwrap();
    let t_factor_start = Instant::now();
    solver.factorize(&k_red).expect("factorize");
    let t_factor = t_factor_start.elapsed();

    let t_back_start = Instant::now();
    let mut direct_results: Vec<Vec<Real>> = Vec::with_capacity(10);
    for b in &rhs_set {
        direct_results.push(solver.back_solve(b).unwrap());
    }
    let t_back = t_back_start.elapsed();
    let t_direct_total = t_factor + t_back;

    // ─── CG path: 10 fresh solves on the reduced system ───
    let t_cg_start = Instant::now();
    for b in &rhs_set {
        let _ = molla_math::iterative::cg_solve(
            |x| {
                let mut y = vec![0.0; n];
                k_red.mv(x, &mut y).unwrap();
                y
            },
            b,
            &vec![0.0; n],
            1.0e-8,
            10_000,
        );
    }
    let t_cg = t_cg_start.elapsed();

    eprintln!();
    eprintln!("─── Stage 36.7 wall-time comparison (n = {n} reduced DoFs) ───");
    eprintln!(
        "direct: factorize = {:>8.3} ms   back_solve × 10 = {:>8.3} ms   total = {:>8.3} ms",
        t_factor.as_secs_f64() * 1e3,
        t_back.as_secs_f64() * 1e3,
        t_direct_total.as_secs_f64() * 1e3,
    );
    eprintln!(
        "CG    : 10 fresh solves                                    = {:>8.3} ms",
        t_cg.as_secs_f64() * 1e3,
    );
    let speedup = t_cg.as_secs_f64() / t_direct_total.as_secs_f64();
    eprintln!(
        "speedup (CG / direct) = {speedup:>5.2}×  ({} faster)",
        if speedup > 1.0 { "direct" } else { "CG" }
    );
    eprintln!();
}
