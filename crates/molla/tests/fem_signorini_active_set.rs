//! Stage 51 — Signorini active-set update loop — contract tests.
//!
//! Stage 49 (`fem_signorini.rs`) tests the fixed-active-set Uzawa
//! solver with a caller-supplied list of slaves. This test file
//! exercises the **v2** surface: the caller passes *candidate* slaves
//! plus a gap function, and the PDAS Newton loop figures out which
//! ones are actually in contact, releasing pairs that separate.
//!
//! Coverage:
//! - `active_set_converges_for_block_on_plane` — every bottom-face
//!   vertex is a candidate; the loop converges to the same answer as
//!   Stage 49's manual active set in ≤ 5 outer iters.
//! - `active_set_releases_lifted_corner` — block with a translation
//!   that lifts one corner above the plane; that pair should be
//!   released (λ = 0, gap > 0), others stay active.
//! - `active_set_handles_zero_initial_penetration` — block hovering
//!   above the plane; the active set converges to empty and `u`
//!   matches the unconstrained solution.

use molla::fem::{
    apply_bcs, assemble_global, compute_element_elasticity_stiffness,
    solve_active_set_signorini, solve_augmented_system, assemble_constraint_matrix,
    target_gap_vector, BoundaryCondition, FemMesh, FunctionSpace,
    SignoriniContactCandidates,
};
use molla::math::iterative::cg_solve;
use molla::math::sparse::BsrMatrix;
use molla::math::{Real, Vec3};

// ─── Shared helpers (copied from fem_signorini.rs) ──────────────────

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

struct BlockSetup {
    k_red: BsrMatrix,
    f_red: Vec<Real>,
    /// Rest positions of every vertex (used by the gap closure).
    vertex_positions: Vec<Vec3>,
    /// Vertex indices of the bottom face (`z = 0`).
    bottom_verts: Vec<u32>,
    /// Total DoF count after BC reduction (= 3 × n_verts here, since
    /// `apply_bcs` keeps the indexing but zeroes the Dirichlet rows).
    n_dofs: usize,
}

/// Same as `fem_signorini.rs::build_block_setup` but also returns the
/// vertex positions so the gap closure can compute current-position
/// distance to the master plane.
fn build_block_setup(
    nx: usize,
    ny: usize,
    nz: usize,
    body_force_total: Real,
    youngs: Real,
    poisson: Real,
    z_offset: Real,
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
    let mut bcs: Vec<BoundaryCondition> = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if (p.z - lz).abs() < 1.0e-9 {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0, 0.0],
            });
        }
    }

    let n = space.ndof(&mesh);
    let mut f_ext = vec![0.0; n];
    let per_vertex = -body_force_total / mesh.vertices.len() as Real;
    for i in 0..mesh.vertices.len() {
        f_ext[i * dim + 2] += per_vertex;
    }

    let mut k_red = k;
    let mut f_red = f_ext;
    apply_bcs(&mut k_red, &mut f_red, &bcs, &mesh.vertices, dim);

    // Bottom-face vertices.
    let bottom_verts: Vec<u32> = mesh
        .vertices
        .iter()
        .enumerate()
        .filter(|(_, p)| p.z.abs() < 1.0e-9)
        .map(|(i, _)| i as u32)
        .collect();

    // Apply the optional z-shift to the rest positions (used by the
    // "hovering block" test to lift the whole mesh above the floor).
    let vertex_positions: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|p| Vec3::new(p.x, p.y, p.z + z_offset))
        .collect();

    BlockSetup {
        k_red,
        f_red,
        vertex_positions,
        bottom_verts,
        n_dofs: n,
    }
}

/// Build the candidate set: every bottom-face vertex is a candidate
/// (NOT pre-selected — the active-set loop discovers contacts).
fn candidates_from_bottom(
    bottom_verts: &[u32],
    dim: usize,
    vertex_positions: &[Vec3],
    plane_point: Vec3,
    plane_normal: Vec3,
) -> SignoriniContactCandidates {
    let slave_dofs: Vec<u32> = bottom_verts.iter().map(|&v| v * dim as u32).collect();
    SignoriniContactCandidates::from_plane(
        slave_dofs,
        vertex_positions.to_vec(),
        dim,
        plane_point,
        plane_normal,
    )
}

// ─── Test 1: full-floor contact converges ───────────────────────────

#[test]
fn active_set_converges_for_block_on_plane() {
    let setup = build_block_setup(
        /*nx*/ 3, /*ny*/ 3, /*nz*/ 3,
        /*body_force_total*/ 1.0e3,
        /*E*/ 1.0e6,
        /*ν*/ 0.3,
        /*z_offset*/ 0.0,
    );
    let dim = 3usize;

    let candidates = candidates_from_bottom(
        &setup.bottom_verts,
        dim,
        &setup.vertex_positions,
        /*plane_point*/ Vec3::new(0.0, 0.0, 0.0),
        /*plane_normal*/ Vec3::new(0.0, 0.0, 1.0),
    );

    let result = solve_active_set_signorini(
        &setup.k_red,
        &candidates,
        &setup.f_red,
        /*max_outer_iters*/ 10,
        /*gap_tol*/ 1.0e-10,
        /*inner_max_iters*/ 500,
        /*inner_tol*/ 1.0e-10,
    )
    .unwrap();

    eprintln!(
        "active_set_block_on_plane: outer_iters = {}, converged = {}, final_active = {} of {}",
        result.trace.outer_iters,
        result.trace.converged,
        result.trace.final_active().len(),
        setup.bottom_verts.len(),
    );
    for it in &result.trace.iterations {
        eprintln!(
            "  iter {}: |A| = {}, +{} / -{}, inner = {} (primal {:.2e}, dual {:.2e})",
            it.iter,
            it.active_slaves.len(),
            it.added.len(),
            it.released.len(),
            it.inner_iters,
            it.primal_residual,
            it.dual_residual,
        );
    }

    // ── Contract: converged within budget ──────────────────────────
    assert!(
        result.trace.converged,
        "PDAS Newton must converge — outer_iters = {}",
        result.trace.outer_iters
    );
    assert!(
        result.trace.outer_iters <= 5,
        "Expected ≤ 5 outer iters, got {}",
        result.trace.outer_iters
    );

    // ── Contract: no penetration on the active set ────────────────
    let mut max_penetration: Real = 0.0;
    for &v in &setup.bottom_verts {
        let uz = result.u[v as usize * dim + 2];
        if uz < -max_penetration {
            max_penetration = -uz;
        }
    }
    assert!(
        max_penetration < 1.0e-8,
        "Bottom slaves should not penetrate: max -u_z = {max_penetration} > 1e-8"
    );

    // ── Contract: same answer as v1's manual active set ───────────
    // Build the same fixed active set and run Stage 49 solver; the
    // displacement field should match within a tight tolerance.
    let pairs: Vec<(u32, Vec3, Real)> = setup
        .bottom_verts
        .iter()
        .map(|&v| (v * dim as u32, Vec3::new(0.0, 0.0, 1.0), 0.0))
        .collect();
    let c = assemble_constraint_matrix(&pairs, setup.n_dofs).unwrap();
    let g = target_gap_vector(&pairs);
    let res_v1 = solve_augmented_system(
        &setup.k_red,
        &c,
        &setup.f_red,
        &g,
        500,
        1.0e-10,
    )
    .unwrap();

    let mut diff_sq: Real = 0.0;
    let mut ref_sq: Real = 0.0;
    for i in 0..setup.n_dofs {
        let d = result.u[i] - res_v1.u[i];
        diff_sq += d * d;
        ref_sq += res_v1.u[i] * res_v1.u[i];
    }
    let rel = (diff_sq / ref_sq.max(1.0e-30)).sqrt();
    eprintln!(
        "active_set vs v1 manual: rel_diff = {:.3e}, ‖u_v1‖ = {:.3e}",
        rel,
        ref_sq.sqrt()
    );
    assert!(
        rel < 1.0e-4,
        "Active-set solution should match v1's manual selection: rel_diff = {rel} > 1e-4"
    );
}

// ─── Test 2: tilted block releases lifted corner ────────────────────

#[test]
fn active_set_releases_lifted_corner() {
    // Strategy: build a block, then artificially lift the rest
    // positions of one bottom corner ABOVE the floor — that corner
    // should never enter the active set (its gap is always positive).
    // Meanwhile, the rest of the bottom face is pushed into the floor
    // by the body force and *does* hit contact. After convergence the
    // lifted corner must NOT be in the active set, and its multiplier
    // must be zero.
    //
    // Note: we modify only the gap closure's view of the geometry —
    // not the FEM mesh itself — to keep the stiffness matrix clean.
    // This is the right model anyway: the "lifted corner" is a
    // contact-detection decision, separate from the elastic
    // discretisation.
    let mut setup = build_block_setup(
        /*nx*/ 3, /*ny*/ 3, /*nz*/ 3,
        /*body_force_total*/ 1.0e3,
        /*E*/ 1.0e6,
        /*ν*/ 0.3,
        /*z_offset*/ 0.0,
    );
    let dim = 3usize;

    // Pick the first bottom vertex as the "lifted corner" — bump its
    // z-coordinate up by 0.1, so the slave starts 0.1 above the
    // floor. The body force won't push it down anywhere near far
    // enough to make contact.
    let lifted_idx = setup.bottom_verts[0] as usize;
    setup.vertex_positions[lifted_idx].z += 0.1;

    let candidates = candidates_from_bottom(
        &setup.bottom_verts,
        dim,
        &setup.vertex_positions,
        /*plane_point*/ Vec3::new(0.0, 0.0, 0.0),
        /*plane_normal*/ Vec3::new(0.0, 0.0, 1.0),
    );

    let result = solve_active_set_signorini(
        &setup.k_red,
        &candidates,
        &setup.f_red,
        /*max_outer_iters*/ 10,
        /*gap_tol*/ 1.0e-10,
        /*inner_max_iters*/ 500,
        /*inner_tol*/ 1.0e-10,
    )
    .unwrap();

    eprintln!(
        "active_set_lifted_corner: outer_iters = {}, converged = {}, final_active = {} of {}",
        result.trace.outer_iters,
        result.trace.converged,
        result.trace.final_active().len(),
        setup.bottom_verts.len(),
    );
    for it in &result.trace.iterations {
        eprintln!(
            "  iter {}: |A| = {}, +{} / -{}, inner = {}",
            it.iter,
            it.active_slaves.len(),
            it.added.len(),
            it.released.len(),
            it.inner_iters,
        );
    }

    assert!(result.trace.converged, "PDAS Newton must converge");

    // ── Contract: the lifted corner is NOT in the active set ──────
    let lifted_dof = setup.bottom_verts[0] * dim as u32;
    let final_active = result.trace.final_active();
    assert!(
        !final_active.contains(&lifted_dof),
        "Lifted corner (DoF {}) should be released, final active = {:?}",
        lifted_dof,
        final_active
    );

    // ── Contract: lifted corner has λ = 0 in the per-candidate vec ─
    // The candidate-index of the lifted corner is 0 (we picked the
    // first bottom vertex).
    let lifted_lambda = result.lambda[0];
    assert!(
        lifted_lambda.abs() < 1.0e-9,
        "Lifted corner λ should be 0, got {lifted_lambda}"
    );

    // ── Contract: lifted corner has gap > 0 at the converged u ────
    let (_, gap_lifted) = (candidates.master_gap_fn)(lifted_dof, &result.u);
    assert!(
        gap_lifted > 1.0e-6,
        "Lifted corner should be separating (gap > 0), got gap = {gap_lifted}"
    );

    // ── Contract: λ ≥ 0 everywhere (no spurious negative pressures) ─
    let min_lambda = result
        .lambda
        .iter()
        .fold(Real::INFINITY, |a, &b| a.min(b));
    assert!(
        min_lambda >= -1.0e-9,
        "All multipliers must be ≥ 0, got min λ = {min_lambda}"
    );

    // ── Contract: at least one OTHER slave is still active ───────
    // (Otherwise the test is degenerate — no contact happening.)
    assert!(
        !final_active.is_empty(),
        "Body force should produce non-empty contact set"
    );
}

// ─── Test 3: hovering block — empty active set, matches unconstrained ─

#[test]
fn active_set_handles_zero_initial_penetration() {
    // Setup: lift the WHOLE mesh by `z_offset = 0.5` so every node
    // starts above the floor (`z ≥ 0.5`). With a moderate body force,
    // the block hangs from the top Dirichlet face — bottom nodes
    // displace slightly downward but never reach z = 0. The active
    // set should converge to empty, and `u` should match the
    // unconstrained CG solve.
    let setup = build_block_setup(
        /*nx*/ 3, /*ny*/ 3, /*nz*/ 3,
        /*body_force_total*/ 1.0e1, // small force — won't bridge the gap.
        /*E*/ 1.0e6,
        /*ν*/ 0.3,
        /*z_offset*/ 0.5, // entire block sits at z ∈ [0.5, 1.5].
    );
    let dim = 3usize;

    let candidates = candidates_from_bottom(
        &setup.bottom_verts,
        dim,
        &setup.vertex_positions,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );

    let result = solve_active_set_signorini(
        &setup.k_red,
        &candidates,
        &setup.f_red,
        10,
        1.0e-10,
        500,
        1.0e-10,
    )
    .unwrap();

    eprintln!(
        "active_set_hovering: outer_iters = {}, converged = {}, final_active = {}",
        result.trace.outer_iters,
        result.trace.converged,
        result.trace.final_active().len(),
    );

    assert!(result.trace.converged, "PDAS Newton must converge");
    assert!(
        result.trace.final_active().is_empty(),
        "Hovering block should produce empty active set, got {:?}",
        result.trace.final_active()
    );

    // ── λ vector must be all zeros ────────────────────────────────
    for &l in &result.lambda {
        assert!(l.abs() < 1.0e-12, "All λ must be zero, got {l}");
    }

    // ── u must match the unconstrained CG solve ────────────────────
    let u_free = cg_solve(
        |x| matvec(&setup.k_red, x),
        &setup.f_red,
        &vec![0.0; setup.n_dofs],
        1.0e-12,
        5_000,
    )
    .x;

    let mut diff_sq: Real = 0.0;
    let mut ref_sq: Real = 0.0;
    for i in 0..setup.n_dofs {
        let d = result.u[i] - u_free[i];
        diff_sq += d * d;
        ref_sq += u_free[i] * u_free[i];
    }
    let rel = (diff_sq / ref_sq.max(1.0e-30)).sqrt();
    eprintln!(
        "active_set_hovering: rel_diff vs unconstrained = {:.3e}",
        rel
    );
    assert!(
        rel < 1.0e-8,
        "Hovering block u should match unconstrained, rel_diff = {rel}"
    );

    // ── No slave should have gap < 0 ──────────────────────────────
    for &v in &setup.bottom_verts {
        let dof = v * dim as u32;
        let (_, gap) = (candidates.master_gap_fn)(dof, &result.u);
        assert!(
            gap > -1.0e-9,
            "Slave DoF {dof} has gap {gap} < 0 — should be separated"
        );
    }
}
