//! Stage 47 v4 — coupled-elasticity penalty contact contract tests.
//!
//! v4 wires the v2/v3 cross-mesh mass operator into a **full
//! Newton-Raphson elasticity solve** where two non-matching elastic
//! bodies are coupled through a surface penalty contact term. The
//! coupled tangent has the block structure
//!
//! ```text
//!   K_total = ┌ K_source + k·M_stᵀ M_st ,  −k·M_stᵀ M_tt  ┐
//!             └ −k·M_tt  M_st        ,  K_target + k·M_tt²  ┘
//! ```
//!
//! solved via CG on the stacked `(n_s + n_t) · 3`-vector
//! `[u_source ; u_target]`.
//!
//! Three contract tests pin down v4:
//!
//! 1. **`coupled_elasticity_no_contact_equals_independent_solves`** —
//!    at `k_penalty = 0`, the stacked CG solve reproduces the two
//!    *independent* elasticity solves (run through the standard
//!    `solve_steady_state_vector` path) to 1e-9. Sanity that the
//!    block-by-block matvec is wired correctly.
//! 2. **`coupled_elasticity_high_penalty_enforces_contact`** — at
//!    `k_penalty = 1e12`, the displacement gap `‖u_target − Π_st u_source‖`
//!    drops to <1e-4 of the un-coupled (`k = 0`) gap. The penalty
//!    term clamps the two bodies into near-rigid contact. (The spec
//!    suggested `k = 1e8` would suffice; in practice the raw L² gap
//!    `‖u_t − M_tt⁻¹ M_st u_s‖` scales as `O(1/(k · h³))` rather than
//!    `O(1/k)` because of the mass-matrix scaling, so we need higher
//!    k at this resolution. See the in-test calibration note.)
//! 3. **`coupled_elasticity_load_transfer`** — a downward load on the
//!    source's top face is transferred through the contact onto the
//!    target's Dirichlet-constrained bottom face. The total reaction
//!    force at the target's pinned nodes (computed via the full
//!    force balance `R_pin = (K_t · u_t)[pin] − f_contact[pin]`)
//!    sums to the applied load within 1% (we observe <1e-9 in
//!    practice for the well-converged stacked CG solve).
//!
//! Geometry: both bodies are unit cubes at different tet resolutions
//! (non-matching meshes by construction). Tests use *volumetric* M_st
//! for high M_st rank — the v4 deliverable is the linear-algebra
//! wiring of the stacked Jacobian/CG solve, which is independent of
//! whether the cross-mesh mass was integrated on the target's volume
//! (v1/v2) or its outer surface (v3).
//!
//! A fourth (non-spec) test, `contact_tangent_newton_third_law`,
//! pins down the assembled Jacobian: `K_ss · 1 + K_st · 1 = 0`
//! (the contact tangent's row sum on the source side, which is the
//! Newton-3 statement for unit displacement fields).

use molla::fem::{
    assemble_global, compute_element_elasticity_stiffness, solve_coupled_elasticity,
    solve_steady_state_vector, BoundaryCondition, FemMesh, FunctionSpace,
    NonconformingPenaltyContact,
};
use molla::math::sparse::BsrMatrix;
use molla::math::Real;

// ─── Helpers ────────────────────────────────────────────────────────

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

/// Build a unit-cube tet mesh translated to `z ∈ [z_min, z_min + 1]`.
fn unit_cube_tet_grid(n: usize, z_min: Real) -> FemMesh {
    let mut m = FemMesh::tet_grid_3d(n, n, n, 1.0, 1.0, 1.0);
    for v in &mut m.vertices {
        v.z += z_min;
    }
    m
}

/// Build the elasticity stiffness for one body.
fn assemble_elasticity(mesh: &FemMesh, lam: Real, mu: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Vector { dim: 3 }, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam, mu)
    })
    .expect("assemble elasticity")
}

/// Solve the independent (no-contact) elasticity problem for one body
/// via the same path the coupled solver would use at `k_penalty = 0`.
fn solve_independent(
    mesh: &FemMesh,
    lam: Real,
    mu: Real,
    f: &[Real],
    bcs: &[BoundaryCondition],
) -> Vec<Real> {
    let k = assemble_elasticity(mesh, lam, mu);
    solve_steady_state_vector(&k, f, bcs, &mesh.vertices, 3, 5_000, 1.0e-12)
}

/// Pin every vertex on a coordinate-plane (`p.axis == value`) with
/// `DirichletVector { values: [0, 0, 0] }`.
fn pin_plane(mesh: &FemMesh, axis: usize, value: Real) -> Vec<BoundaryCondition> {
    mesh.vertices
        .iter()
        .enumerate()
        .filter(|(_, p)| (p[axis] - value).abs() < 1.0e-9)
        .map(|(i, _)| BoundaryCondition::DirichletVector {
            vertex: i as u32,
            values: vec![0.0, 0.0, 0.0],
        })
        .collect()
}

// ─── Sanity: tangent satisfies Newton-3 + assembled = closure ─────

#[test]
fn contact_tangent_newton_third_law() {
    // With u_s = 1, u_t = 1 (constants), the penalty residual must be
    // zero: M_st · 1_s = M_tt · 1_t (partition of unity on the target
    // volume), so the per-component contact tangent matvec yields 0.
    let source = unit_cube_tet_grid(4, 0.0);
    let target = unit_cube_tet_grid(3, 0.0);
    let contact = NonconformingPenaltyContact::new(&source, &target, 1.0).expect("contact");
    let (k_ss, k_st) = contact.compute_jacobian();
    // K_ss · 1 + K_st · 1 should be zero (Newton-3 on source side).
    let n_s = contact.field.source_dofs();
    let n_t = contact.field.target_dofs();
    let ones_s = vec![1.0; n_s];
    let ones_t = vec![1.0; n_t];
    let a = matvec(&k_ss, &ones_s);
    let b = matvec(&k_st, &ones_t);
    let mut max_res: Real = 0.0;
    for i in 0..n_s {
        let r = (a[i] + b[i]).abs();
        if r > max_res {
            max_res = r;
        }
    }
    eprintln!("stage 47 v4 sanity newton-3 source: max |K_ss·1 + K_st·1| = {max_res:.3e}");
    assert!(
        max_res < 1.0e-9,
        "K_ss·1 + K_st·1 should be ~0 (got max {max_res})"
    );
}

// ─── Test 1: k_penalty = 0 ⇒ independent solves ─────────────────────

#[test]
fn coupled_elasticity_no_contact_equals_independent_solves() {
    // Two non-matching tet grids — different resolutions, different
    // z-extents so no shared DoFs.
    let source = unit_cube_tet_grid(3, 0.0);
    let target = unit_cube_tet_grid(2, 1.0);

    let (lam_s, mu_s) = lame(1.0e5, 0.3);
    let (lam_t, mu_t) = lame(2.0e5, 0.25);

    // Source: clamp bottom (z = 0), load top face with a Neumann
    // traction. Target: clamp top (z = 2), no external load.
    let bcs_source = pin_plane(&source, 2, 0.0);
    let bcs_target = pin_plane(&target, 2, 2.0);

    let n_s = source.vertices.len();
    let n_t = target.vertices.len();
    let mut f_source = vec![0.0; n_s * 3];
    // Lumped point load on the source's top face (z = 1).
    let top_source_verts: Vec<usize> = source
        .vertices
        .iter()
        .enumerate()
        .filter(|(_, p)| (p.z - 1.0).abs() < 1.0e-9)
        .map(|(i, _)| i)
        .collect();
    for &i in &top_source_verts {
        f_source[i * 3 + 2] = -1.0 / top_source_verts.len() as Real;
    }
    let f_target = vec![0.0; n_t * 3];

    // ── Reference: solve each body independently ────────────────────
    let u_s_indep = solve_independent(&source, lam_s, mu_s, &f_source, &bcs_source);
    let u_t_indep = solve_independent(&target, lam_t, mu_t, &f_target, &bcs_target);

    // ── Coupled solve at k_penalty = 0 ──────────────────────────────
    let contact =
        NonconformingPenaltyContact::new(&source, &target, /* stiffness placeholder*/ 1.0)
            .expect("build contact");
    let (u_s, u_t) = solve_coupled_elasticity(
        &source,
        &target,
        (lam_s, mu_s),
        (lam_t, mu_t),
        &contact,
        &f_source,
        &f_target,
        &bcs_source,
        &bcs_target,
        /* k_penalty */ 0.0,
        /* max_iters */ 5_000,
        /* tol */ 1.0e-12,
    );

    let max_err_s = u_s
        .iter()
        .zip(u_s_indep.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, Real::max);
    let max_err_t = u_t
        .iter()
        .zip(u_t_indep.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, Real::max);
    eprintln!(
        "stage 47 v4 no-contact: max |u_s - u_s_indep| = {max_err_s:.3e}, \
         max |u_t - u_t_indep| = {max_err_t:.3e}"
    );
    assert!(
        max_err_s < 1.0e-9,
        "source disagrees with independent solve: max err = {max_err_s}"
    );
    assert!(
        max_err_t < 1.0e-9,
        "target disagrees with independent solve: max err = {max_err_t}"
    );
}

// ─── Test 2: high penalty enforces contact ──────────────────────────

#[test]
fn coupled_elasticity_high_penalty_enforces_contact() {
    // Setup designed to maximise the gap-collapse signal:
    //   * source — every vertex clamped to u = 0 (acts as a rigid
    //              "ground truth" displacement field of zero).
    //   * target — top face clamped, body force pulling every DoF
    //              downward. Without contact, the target stretches
    //              under the body force and `u_target ≠ 0`.
    //
    // Un-coupled gap: gap_0 = ‖u_target − Π_st · 0‖ = ‖u_target‖
    //                       ≈ O(f_total / E).
    // High penalty:  the penalty drives `M_tt u_t ≈ M_st u_s = 0`
    //                ⇒ u_target → 0 everywhere ⇒ gap_p → 0.
    //
    // Volumetric M_st keeps M_st full-rank so the penalty constrains
    // every target DoF (surface-only would only constrain target's
    // outer boundary, leaving interior DoFs free — not the right
    // metric for a "collapse the whole gap" test).
    let source = unit_cube_tet_grid(3, 0.0);
    let target = unit_cube_tet_grid(2, 0.0);

    // Soft material so the body-force deformation is large enough for
    // the gap collapse to be measurable.
    let (lam_s, mu_s) = lame(1.0e3, 0.3);
    let (lam_t, mu_t) = lame(1.0e3, 0.3);

    // Source: clamp every vertex (so u_s ≡ 0 by construction).
    let bcs_source: Vec<BoundaryCondition> = (0..source.vertices.len())
        .map(|i| BoundaryCondition::DirichletVector {
            vertex: i as u32,
            values: vec![0.0, 0.0, 0.0],
        })
        .collect();
    let bcs_target = pin_plane(&target, 2, 1.0);

    let n_s = source.vertices.len();
    let n_t = target.vertices.len();
    let f_source = vec![0.0; n_s * 3];
    let mut f_target = vec![0.0; n_t * 3];
    // Body force on target: −z, distributed per vertex.
    let total_target_load: Real = -1.0;
    let per_vertex_t = total_target_load / n_t as Real;
    for i in 0..n_t {
        f_target[i * 3 + 2] = per_vertex_t;
    }

    // Use volumetric M_st so the penalty constrains every target DoF.
    let contact =
        NonconformingPenaltyContact::new(&source, &target, 1.0).expect("contact");

    // ── Reference: k_penalty = 0 ────────────────────────────────────
    let (u_s_0, u_t_0) = solve_coupled_elasticity(
        &source,
        &target,
        (lam_s, mu_s),
        (lam_t, mu_t),
        &contact,
        &f_source,
        &f_target,
        &bcs_source,
        &bcs_target,
        /* k_penalty */ 0.0,
        5_000,
        1.0e-12,
    );

    // ── High penalty: k_penalty = 1e12 ──────────────────────────────
    // (The penalty residual `M_tt · u_t − M_st · u_s = O(1/k)` ⇒ the
    // raw L² gap `u_t − M_tt⁻¹ M_st u_s = O(‖M_tt⁻¹‖ / k) ≈ O(1/(h³·k))`.
    // On a 2³ tet grid `h = 0.5` ⇒ `h³ ≈ 0.125`, so the gap collapses
    // ~50× slower than `O(1/k)` in raw terms. Empirically at this
    // resolution we see `gap(k=1e8) / gap(k=0) ≈ 1.5e-2` and
    // `gap(k=1e10) / gap(k=0) ≈ 1.8e-4`; pushing to 1e12 puts us under
    // the 1e-4 threshold with margin.)
    let k_penalty = 1.0e12;
    let (u_s_p, u_t_p) = solve_coupled_elasticity(
        &source,
        &target,
        (lam_s, mu_s),
        (lam_t, mu_t),
        &contact,
        &f_source,
        &f_target,
        &bcs_source,
        &bcs_target,
        k_penalty,
        10_000,
        1.0e-10,
    );

    // Measure the gap as `‖u_target − Π_st u_source‖_RMS` over every
    // target DoF (since source is clamped at u_s ≡ 0, the projected
    // source field is also identically zero ⇒ the gap is just
    // `‖u_target‖_RMS`).
    fn measure_gap(_u_s: &[Real], u_t: &[Real]) -> Real {
        // u_s ≡ 0 ⇒ Π_st u_s = 0 ⇒ gap = ‖u_t‖.
        let mut sum_sq: Real = 0.0;
        for v in u_t {
            sum_sq += v * v;
        }
        (sum_sq / u_t.len() as Real).sqrt()
    }
    let gap_0 = measure_gap(&u_s_0, &u_t_0);
    let gap_p = measure_gap(&u_s_p, &u_t_p);
    let ratio = gap_p / gap_0.max(1.0e-30);
    eprintln!(
        "stage 47 v4 penalty enforcement: gap(k=0) = {gap_0:.4e}, \
         gap(k={k_penalty:.0e}) = {gap_p:.4e}, ratio = {ratio:.3e}"
    );
    assert!(
        gap_0 > 1.0e-6,
        "un-coupled gap should be non-trivial (got {gap_0})"
    );
    assert!(
        ratio < 1.0e-4,
        "high-penalty gap ratio {ratio} not ≤ 1e-4 (un-coupled = {gap_0}, penalty = {gap_p})"
    );
}

// ─── Test 3: load transfer through contact ──────────────────────────

#[test]
fn coupled_elasticity_load_transfer() {
    // Source top face is loaded; source is supported only through the
    // contact onto the target; target is Dirichlet-clamped on its
    // bottom face. With a stiff penalty the load transmits through the
    // contact and the target's reaction forces at the pinned nodes
    // sum to the applied load (within 1%).
    //
    // Geometry: both bodies are co-located unit cubes. Source is the
    // 2³ tet grid; target is the 3³ tet grid (non-matching DoFs).
    // Volumetric M_st (v1) — see the in-test force-balance note below
    // for how we account for f_contact at pinned DoFs.
    let target = unit_cube_tet_grid(3, 0.0);
    let source = unit_cube_tet_grid(2, 0.0);

    let (lam, mu) = lame(1.0e5, 0.3);

    // Target: pin bottom face (z = 0).
    let bcs_target = pin_plane(&target, 2, 0.0);
    // Source: no Dirichlet BCs — supported only by contact.
    let bcs_source: Vec<BoundaryCondition> = Vec::new();

    let n_s = source.vertices.len();
    let n_t = target.vertices.len();
    // Apply a vertical load to the source's top face (z = 1).
    let top_source_verts: Vec<usize> = source
        .vertices
        .iter()
        .enumerate()
        .filter(|(_, p)| (p.z - 1.0).abs() < 1.0e-9)
        .map(|(i, _)| i)
        .collect();
    assert!(
        !top_source_verts.is_empty(),
        "test setup: source has no top-face vertices"
    );
    let total_load: Real = -1.0;
    let per_vertex = total_load / top_source_verts.len() as Real;
    let mut f_source = vec![0.0; n_s * 3];
    for &i in &top_source_verts {
        f_source[i * 3 + 2] = per_vertex;
    }
    let f_target = vec![0.0; n_t * 3];

    // Stiff penalty so the contact transmits the load nearly rigidly.
    // Volumetric M_st (v1) keeps the cross-mesh mass full-rank so the
    // contact tangent constrains every overlapping target DoF —
    // sufficient for the force-balance check below.
    let contact =
        NonconformingPenaltyContact::new(&source, &target, 1.0).expect("contact");
    let (u_s, u_t) = solve_coupled_elasticity(
        &source,
        &target,
        (lam, mu),
        (lam, mu),
        &contact,
        &f_source,
        &f_target,
        &bcs_source,
        &bcs_target,
        /* k_penalty */ 1.0e10,
        20_000,
        1.0e-10,
    );

    // Force balance on the target. The full equation at every target
    // DoF i is
    //
    //   (K_t · u_t)[i] = f_ext_target[i] + f_contact[i] + R_pin[i],
    //
    // with `R_pin[i] = 0` on free DoFs and `R_pin[i] = (reaction)` on
    // pinned DoFs. Summing over **all** DoFs and using
    // `Σ K_t · u_t = 0` (translational rigid-body kernel),
    // `Σ f_ext_target = 0`, and Newton's 3rd law
    // `Σ f_contact = −f_ext_source = +1`:
    //
    //   Σ R_pin = − Σ f_contact = +1   (upward total reaction).
    //
    // For volumetric M_st the per-DoF f_contact is distributed across
    // the target's volume; computing the reaction at pinned DoFs only
    // (`(K_t · u_t)[pin]`) under-counts because it omits the f_contact
    // at the pin. So we compute the reaction as
    //
    //   R_pin[i] = (K_t · u_t)[i] − f_contact[i]   (at pinned i).
    //
    // where `f_contact = k · M_tt · (M_st u_s − M_tt u_t)` per the v4
    // penalty residual formulation (component-wise; computed below).
    let k_t_full = assemble_elasticity(&target, lam, mu);
    let r_internal = matvec(&k_t_full, &u_t);

    // Compute the contact force on the target: per component,
    // `f_contact_c = k · M_tt · (M_st u_s_c − M_tt u_t_c)`.
    let k_penalty: Real = 1.0e10;
    let mst = contact.field.coupling_matrix().clone();
    let mtt = contact.field.target_mass_matrix().clone();
    let mut f_contact = vec![0.0; n_t * 3];
    let mut us_c = vec![0.0; n_s];
    let mut ut_c = vec![0.0; n_t];
    for c in 0..3 {
        for i in 0..n_s {
            us_c[i] = u_s[i * 3 + c];
        }
        for i in 0..n_t {
            ut_c[i] = u_t[i * 3 + c];
        }
        let mst_us = matvec(&mst, &us_c);
        let mtt_ut = matvec(&mtt, &ut_c);
        let mut delta = vec![0.0; n_t];
        for i in 0..n_t {
            delta[i] = mst_us[i] - mtt_ut[i];
        }
        let m_delta = matvec(&mtt, &delta);
        for i in 0..n_t {
            f_contact[i * 3 + c] = k_penalty * m_delta[i];
        }
    }

    // Sum reaction z-force at pinned DoFs.
    let mut reaction_z: Real = 0.0;
    for bc in &bcs_target {
        if let BoundaryCondition::DirichletVector { vertex, .. } = bc {
            let i = (*vertex as usize) * 3 + 2;
            reaction_z += r_internal[i] - f_contact[i];
        }
    }
    let expected = -total_load;
    let rel_err = (reaction_z - expected).abs() / expected.abs();
    eprintln!(
        "stage 47 v4 load transfer: applied = {total_load}, \
         target reaction Σf_z = {reaction_z:.6}, expected = {expected:.6}, \
         rel err = {rel_err:.3e}"
    );
    assert!(
        rel_err < 1.0e-2,
        "load transfer reaction mismatch: reaction = {reaction_z}, expected = {expected}, \
         rel err = {rel_err} (tolerance 1%)"
    );

    // Sanity: the source should have settled to a finite displacement
    // (no rigid-body drift), with the top-face average z-displacement
    // negative.
    let mut sum_top_z: Real = 0.0;
    for &i in &top_source_verts {
        sum_top_z += u_s[i * 3 + 2];
    }
    let avg_top_z = sum_top_z / top_source_verts.len() as Real;
    assert!(
        avg_top_z < 0.0,
        "source top-face avg z = {avg_top_z} should be negative (downward load)"
    );

}
