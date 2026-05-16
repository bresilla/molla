//! Stage 49 — Signorini hard non-penetration contact via Lagrange
//! multipliers (fixed active set) — contract tests.
//!
//! Setup shared by every test: a small 3D tet block, clamped on the
//! top face, loaded with a downward body force pushing it against a
//! rigid floor placed at `z = 0`. The bottom-face nodes are the
//! Signorini slave nodes; their master plane is `z = 0` with outward
//! normal `+z`, target gap `0`.
//!
//! Coverage:
//! - `signorini_block_on_plane_no_penetration` — every active slave's
//!   vertical displacement is non-negative within numerical noise
//!   (contrast: the unconstrained problem produces negative values).
//! - `signorini_reproduces_penalty_at_high_stiffness` — the Lagrange
//!   solution matches a stiff penalty solve (`k = 1e8`) to within
//!   ~`1e-3` relative error.
//! - `signorini_lambda_non_negative` — the dual multipliers stay
//!   `λ ≥ −1e-9` (no spurious negative pressures).

use molla_fem::{
    BoundaryCondition, FemMesh, FunctionSpace, apply_bcs, assemble_constraint_matrix,
    assemble_global, compute_element_elasticity_stiffness, solve_augmented_system,
    target_gap_vector,
};
use molla_math::iterative::cg_solve;
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// ─── Shared helpers ─────────────────────────────────────────────────

fn lame(youngs: Real, poisson: Real) -> (Real, Real) {
    let mu = youngs / (2.0 * (1.0 + poisson));
    let lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    (lam, mu)
}

fn matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    let mut y = vec![0.0; a.nrows()];
    a.mv(x, &mut y).unwrap();
    y
}

/// Build a small tet block of dimensions `[0, lx] × [0, ly] × [0, lz]`,
/// fully clamp every vertex on the **top face** (`z = lz`), and apply
/// a uniform downward body force `f_z = −body_force / n_dofs` to every
/// free node. Returns `(K, f, bottom_node_indices, lz)` ready for the
/// Signorini solve. The `K` and `f` have already been routed through
/// `apply_bcs`, so they're the reduced system.
struct BlockSetup {
    /// Reduced (Dirichlet-applied) stiffness matrix.
    k_red: BsrMatrix,
    /// Reduced RHS.
    f_red: Vec<Real>,
    /// Vertex indices of the bottom face (`z = 0`).
    bottom_verts: Vec<u32>,
}

fn build_block_setup(
    nx: usize,
    ny: usize,
    nz: usize,
    body_force_total: Real,
    youngs: Real,
    poisson: Real,
) -> BlockSetup {
    let lx: Real = 1.0;
    let ly: Real = 1.0;
    let lz: Real = 1.0;
    let dim = 3usize;
    let mesh = FemMesh::tet_grid_3d(nx, ny, nz, lx, ly, lz);
    let space = FunctionSpace::P1Vector { dim };

    let (lam, mu) = lame(youngs, poisson);
    let k = assemble_global(&mesh, space, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam, mu)
    })
    .unwrap();

    // Top-face Dirichlet: pin every vertex on z = lz to u = 0.
    // (Cantilever-style: the block hangs from its top; gravity pulls
    // the rest down into the floor.)
    let mut bcs: Vec<BoundaryCondition> = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if (p.z - lz).abs() < 1.0e-9 {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0, 0.0],
            });
        }
    }

    // Body force: distribute `body_force_total` uniformly across all
    // free DoFs as a downward force on the z-component.
    let n = space.ndof(&mesh);
    let mut f_ext = vec![0.0; n];
    // Lump the body force at each node — same per-vertex weight,
    // which is good enough for the contract tests since we only
    // care about the *qualitative* "node pushed into the floor"
    // behaviour.
    let per_vertex = -body_force_total / mesh.vertices.len() as Real;
    for i in 0..mesh.vertices.len() {
        f_ext[i * dim + 2] += per_vertex;
    }

    // Reduce.
    let mut k_red = k;
    let mut f_red = f_ext;
    apply_bcs(&mut k_red, &mut f_red, &bcs, &mesh.vertices, dim);

    // Collect bottom-face vertices.
    let bottom_verts: Vec<u32> = mesh
        .vertices
        .iter()
        .enumerate()
        .filter(|(_, p)| p.z.abs() < 1.0e-9)
        .map(|(i, _)| i as u32)
        .collect();

    BlockSetup {
        k_red,
        f_red,
        bottom_verts,
    }
}

/// Build the Signorini active pairs: every bottom vertex is a slave
/// with master normal `+z` and target gap `0`.
fn active_pairs_from_bottom(bottom_verts: &[u32], dim: usize) -> Vec<(u32, Vec3, Real)> {
    bottom_verts
        .iter()
        .map(|&v| (v * dim as u32, Vec3::new(0.0, 0.0, 1.0), 0.0))
        .collect()
}

// ─── Test 1: no penetration on the active set ───────────────────────

#[test]
fn signorini_block_on_plane_no_penetration() {
    // Small block — coarse mesh keeps the test fast; the contract is
    // about the *constraint*, not the elasticity discretisation error.
    let setup = build_block_setup(
        /*nx*/ 3, /*ny*/ 3, /*nz*/ 3, /*body_force_total*/ 1.0e3,
        /*E*/ 1.0e6, /*ν*/ 0.3,
    );

    // ── Sanity: unconstrained solve produces penetration ────────────
    let n_dofs = setup.f_red.len();
    let dim = 3usize;
    let u_free = cg_solve(
        |x| matvec(&setup.k_red, x),
        &setup.f_red,
        &vec![0.0; n_dofs],
        1.0e-11,
        5_000,
    )
    .x;
    let mut min_uz_free: Real = 0.0;
    for &v in &setup.bottom_verts {
        let uz = u_free[v as usize * dim + 2];
        if uz < min_uz_free {
            min_uz_free = uz;
        }
    }
    assert!(
        min_uz_free < -1.0e-6,
        "Sanity check failed: unconstrained block should penetrate floor — \
         got min u_z = {min_uz_free} (expected < −1e-6). The test is \
         degenerate without a real load."
    );

    // ── Signorini solve ────────────────────────────────────────────
    let pairs = active_pairs_from_bottom(&setup.bottom_verts, dim);
    let c = assemble_constraint_matrix(&pairs, n_dofs).unwrap();
    let g = target_gap_vector(&pairs);
    let res = solve_augmented_system(
        &setup.k_red,
        &c,
        &setup.f_red,
        &g,
        /*max_iters*/ 500,
        /*tol*/ 1.0e-10,
    )
    .unwrap();

    eprintln!(
        "signorini_block_on_plane: outer_iters = {}, primal_res = {:.3e}, dual_res = {:.3e}",
        res.iters, res.primal_residual, res.dual_residual
    );

    // ── Contract: no penetration on the active set ────────────────
    let mut max_penetration: Real = 0.0;
    for &v in &setup.bottom_verts {
        let uz = res.u[v as usize * dim + 2];
        if uz < -max_penetration {
            max_penetration = -uz;
        }
    }
    assert!(
        max_penetration < 1.0e-9,
        "Active slave nodes should not penetrate: max −u_z = {max_penetration} > 1e-9"
    );
}

// ─── Test 2: hard contact agrees with stiff penalty ────────────────

#[test]
fn signorini_reproduces_penalty_at_high_stiffness() {
    let setup = build_block_setup(
        3, 3, 3, /*body_force_total*/ 1.0e3, /*E*/ 1.0e4, /*ν*/ 0.3,
    );
    let n_dofs = setup.f_red.len();
    let dim = 3usize;

    let pairs = active_pairs_from_bottom(&setup.bottom_verts, dim);
    let c = assemble_constraint_matrix(&pairs, n_dofs).unwrap();
    let g = target_gap_vector(&pairs);

    // ── Hard Lagrange solve ────────────────────────────────────────
    let res_hard =
        solve_augmented_system(&setup.k_red, &c, &setup.f_red, &g, 500, 1.0e-10).unwrap();
    eprintln!(
        "signorini_vs_penalty: hard outer_iters = {}, primal_res = {:.3e}, dual_res = {:.3e}",
        res_hard.iters, res_hard.primal_residual, res_hard.dual_residual
    );

    // ── Penalty solve: K_pen = K + k_pen · C^T C, f_pen = f + k_pen · C^T g ─
    // k = 1e8 is the headline number for the contract test (matches
    // the Stage 47 v2 penalty solver's default). The residual
    // penetration is `λ/k`; the bound on relative displacement
    // difference is approximately `‖λ‖ · compliance / k`. For this
    // problem `compliance ≈ 3e-6` and `λ_total ≈ body_force`, so
    // `1e-3` relative tolerance is comfortably reachable at k = 1e8.
    let k_pen: Real = 1.0e8;
    // Build the augmented operator implicitly via a closure:
    //   K_pen · u = K · u + k_pen · C^T · (C · u)
    let k_for_op = setup.k_red.clone();
    let c_for_op = c.clone();
    let op = |x: &[Real]| -> Vec<Real> {
        let mut y = matvec(&k_for_op, x);
        let cx = matvec(&c_for_op, x);
        // C^T · cx using row walk of C.
        for r in 0..c_for_op.block_rows {
            let lo = c_for_op.row_offsets[r];
            let hi = c_for_op.row_offsets[r + 1];
            let xr = cx[r];
            for slot in lo..hi {
                let col = c_for_op.col_idx[slot];
                y[col] += k_pen * c_for_op.values[slot] * xr;
            }
        }
        y
    };
    // RHS: f + k_pen · C^T · g.
    let mut f_pen = setup.f_red.clone();
    for r in 0..c.block_rows {
        let lo = c.row_offsets[r];
        let hi = c.row_offsets[r + 1];
        let gr = g[r];
        for slot in lo..hi {
            let col = c.col_idx[slot];
            f_pen[col] += k_pen * c.values[slot] * gr;
        }
    }
    let pen_cg = cg_solve(op, &f_pen, &vec![0.0; n_dofs], 1.0e-12, 50_000);
    let u_pen = pen_cg.x;
    eprintln!(
        "penalty CG: iters = {}, residual = {:.3e}",
        pen_cg.iters, pen_cg.residual
    );

    // ── Compare ────────────────────────────────────────────────────
    // The penalty solution allows a tiny residual penetration of order
    // 1/k_pen, so we compare the L²-norm of the difference relative to
    // the L²-norm of the hard solution and accept up to 1e-3.
    let n = res_hard.u.len();
    let mut diff_sq: Real = 0.0;
    let mut ref_sq: Real = 0.0;
    for i in 0..n {
        let d = res_hard.u[i] - u_pen[i];
        diff_sq += d * d;
        ref_sq += res_hard.u[i] * res_hard.u[i];
    }
    let rel = (diff_sq / ref_sq.max(1.0e-30)).sqrt();
    eprintln!(
        "signorini_vs_penalty: rel_diff = {:.3e}, ‖u_hard‖ = {:.3e}",
        rel,
        ref_sq.sqrt()
    );
    assert!(
        rel < 1.0e-3,
        "Hard Lagrange differs from k=1e8 penalty by {rel} > 1e-3 relative"
    );
}

// ─── Test 3: dual multipliers are non-negative ──────────────────────

#[test]
fn signorini_lambda_non_negative() {
    let setup = build_block_setup(
        3, 3, 3, /*body_force_total*/ 1.0e3, /*E*/ 1.0e4, /*ν*/ 0.3,
    );
    let n_dofs = setup.f_red.len();
    let dim = 3usize;

    let pairs = active_pairs_from_bottom(&setup.bottom_verts, dim);
    let c = assemble_constraint_matrix(&pairs, n_dofs).unwrap();
    let g = target_gap_vector(&pairs);

    let res = solve_augmented_system(&setup.k_red, &c, &setup.f_red, &g, 500, 1.0e-10).unwrap();

    let mut min_lambda: Real = Real::INFINITY;
    for &l in &res.lambda {
        if l < min_lambda {
            min_lambda = l;
        }
    }
    eprintln!(
        "signorini_lambda: min λ = {:.3e}, max λ = {:.3e}, sum λ = {:.3e}",
        min_lambda,
        res.lambda.iter().fold(Real::NEG_INFINITY, |a, &b| a.max(b)),
        res.lambda.iter().sum::<Real>(),
    );
    assert!(
        min_lambda >= -1.0e-9,
        "Signorini dual must satisfy λ ≥ 0 (up to 1e-9 numerical noise) — \
         got min λ = {min_lambda}"
    );
}
