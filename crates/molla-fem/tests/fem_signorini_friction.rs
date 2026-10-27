//! Stage 50 — Coulomb friction at the Signorini contact interface —
//! contract tests.
//!
//! Setup shared by every test: a small 3D tet block, clamped on the
//! top face, loaded with a downward body force pushing it against a
//! rigid floor placed at `z = 0`. The bottom-face nodes are the
//! Signorini slave nodes; their master plane is `z = 0` with outward
//! normal `+z`, target gap `0`. On top of the Stage 49 normal load
//! we add a uniform *lateral* body force along `+x` that creates a
//! tangential shear at the contact patch — the Coulomb cap then
//! decides whether the bottom face sticks (`u_x ≈ 0` at slaves) or
//! slips (`|λ_t| ≈ μ · λ_n`).
//!
//! Coverage:
//! - `coulomb_friction_zero_mu_matches_frictionless` — at μ = 0 the
//!   friction disc collapses to a point so the tangential multipliers
//!   are forced to zero; both `u` and `λ_n` must match the Stage 49
//!   frictionless solver to within `1e-9`.
//! - `coulomb_friction_high_mu_sticks` — block + small lateral force +
//!   μ = 1.0 → tangential displacement at the slave nodes near zero
//!   (stick regime); tangential multiplier magnitude strictly below
//!   `μ · λ_n` (interior of the cone).
//! - `coulomb_friction_low_mu_slips` — same problem with μ = 0.1 →
//!   non-zero tangential displacement and `|λ_t| ≈ μ · λ_n` per active
//!   slave (slip regime — at the cone boundary).

use molla_fem::{
    BoundaryCondition, ContactPair, FemMesh, FrictionState, FunctionSpace, RateStateFrictionParams,
    RateStateFrictionState, SignoriniContact, SignoriniContactCandidates,
    advance_rate_state_friction, apply_bcs, assemble_constraint_matrix, assemble_global,
    compute_element_elasticity_stiffness, rate_state_friction_coeff,
    solve_active_set_signorini_with_friction, solve_augmented_system,
    solve_augmented_system_with_friction, target_gap_vector,
};
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// ─── Shared helpers ─────────────────────────────────────────────────

fn lame(youngs: Real, poisson: Real) -> (Real, Real) {
    let mu = youngs / (2.0 * (1.0 + poisson));
    let lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    (lam, mu)
}

struct BlockSetup {
    k_red: BsrMatrix,
    f_red: Vec<Real>,
    bottom_verts: Vec<u32>,
    vertex_positions: Vec<Vec3>,
}

/// Build a small tet block, clamp the top face, and apply a uniform
/// body force `(f_lat_x, 0, -body_force_z) / n_verts` per vertex.
fn build_block_with_lateral_force(
    nx: usize,
    ny: usize,
    nz: usize,
    body_force_z: Real,
    body_force_x: Real,
    youngs: Real,
    poisson: Real,
) -> BlockSetup {
    let lx: Real = 1.0;
    let ly: Real = 1.0;
    let lz: Real = 1.0;
    let dim = 3usize;
    let mesh = FemMesh::tet_grid_3d(nx, ny, nz, lx, ly, lz);
    let space = FunctionSpace::P1Vector { dim };

    let (lam, mu_lame) = lame(youngs, poisson);
    let k = assemble_global(&mesh, space, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam, mu_lame)
    })
    .unwrap();

    // Top-face Dirichlet.
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
    let per_vertex_z = -body_force_z / mesh.vertices.len() as Real;
    let per_vertex_x = body_force_x / mesh.vertices.len() as Real;
    for i in 0..mesh.vertices.len() {
        f_ext[i * dim] += per_vertex_x;
        f_ext[i * dim + 2] += per_vertex_z;
    }

    let mut k_red = k;
    let mut f_red = f_ext;
    apply_bcs(&mut k_red, &mut f_red, &bcs, &mesh.vertices, dim);

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
        vertex_positions: mesh.vertices,
    }
}

/// Build the Signorini active pairs with a given friction coefficient.
/// Normal = +z, target_gap = 0; tangents auto-derived (so t_u ≈ +x,
/// t_v ≈ +y).
fn frictional_contact_from_bottom(
    bottom_verts: &[u32],
    dim: usize,
    friction_coeff: Real,
) -> SignoriniContact {
    let mut contact = SignoriniContact::with_friction(friction_coeff);
    for &v in bottom_verts {
        let pair = ContactPair::from_normal(v * dim as u32, Vec3::new(0.0, 0.0, 1.0), 0.0);
        contact.active_pairs.push(pair);
    }
    contact
}

fn candidates_from_bottom(
    bottom_verts: &[u32],
    dim: usize,
    vertex_positions: &[Vec3],
) -> SignoriniContactCandidates {
    let slave_dofs: Vec<u32> = bottom_verts.iter().map(|&v| v * dim as u32).collect();
    SignoriniContactCandidates::from_plane(
        slave_dofs,
        vertex_positions.to_vec(),
        dim,
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0),
    )
}

// ─── Test 1: μ = 0 matches the Stage 49 frictionless solver ────────

#[test]
fn coulomb_friction_zero_mu_matches_frictionless() {
    // Exact same body load as Stage 49 test 1 — no lateral force.
    let setup = build_block_with_lateral_force(
        /*nx*/ 3, /*ny*/ 3, /*nz*/ 3, /*body_force_z*/ 1.0e3,
        /*body_force_x*/ 0.0, /*E*/ 1.0e6, /*ν*/ 0.3,
    );
    let n_dofs = setup.f_red.len();
    let dim = 3usize;

    // ── Stage 49 reference ────────────────────────────────────────
    let pairs_tuple: Vec<(u32, Vec3, Real)> = setup
        .bottom_verts
        .iter()
        .map(|&v| (v * dim as u32, Vec3::new(0.0, 0.0, 1.0), 0.0))
        .collect();
    let c_ref = assemble_constraint_matrix(&pairs_tuple, n_dofs).unwrap();
    let g_ref = target_gap_vector(&pairs_tuple);
    let res_ref = solve_augmented_system(
        &setup.k_red,
        &c_ref,
        &setup.f_red,
        &g_ref,
        /*max_iters*/ 500,
        /*tol*/ 1.0e-11,
    )
    .unwrap();

    // ── Stage 50 with μ = 0 ───────────────────────────────────────
    let contact = frictional_contact_from_bottom(&setup.bottom_verts, dim, /*μ*/ 0.0);
    // Use a larger iteration cap: the stacked-C friction solver has
    // 3× the dual dimension and its `ρ` estimate is correspondingly
    // smaller (Schur spectrum bound covers the tangent rows too), so
    // it needs more outer iterations to drive `λ_n` to the same
    // tolerance as the normal-only Uzawa.
    let res_fric =
        solve_augmented_system_with_friction(&setup.k_red, &contact, &setup.f_red, 4000, 1.0e-11)
            .unwrap();

    eprintln!(
        "zero_mu: ref iters = {}, friction iters = {}",
        res_ref.iters, res_fric.iters,
    );

    // ── Contract: displacements agree to 1e-9 absolute, λ_n agrees ─
    let mut max_du: Real = 0.0;
    for i in 0..n_dofs {
        let d = (res_ref.u[i] - res_fric.u[i]).abs();
        if d > max_du {
            max_du = d;
        }
    }
    assert!(
        max_du < 1.0e-9,
        "μ=0 friction solver should match frictionless within 1e-9; \
         max |Δu| = {max_du:.3e}"
    );

    let mut max_dlam: Real = 0.0;
    for k in 0..res_ref.lambda.len() {
        let d = (res_ref.lambda[k] - res_fric.lambda_normal[k]).abs();
        if d > max_dlam {
            max_dlam = d;
        }
    }
    assert!(
        max_dlam < 1.0e-9,
        "μ=0 λ_n should match frictionless within 1e-9; max Δλ_n = {max_dlam:.3e}"
    );

    // Tangential multipliers must be identically zero at μ = 0.
    let max_lam_t: Real = res_fric
        .lambda_tangent
        .iter()
        .map(|v| v.abs())
        .fold(0.0, Real::max);
    assert!(
        max_lam_t < 1.0e-12,
        "μ=0 must force λ_t to zero; got max |λ_t| = {max_lam_t:.3e}"
    );
}

// ─── Test 2: high μ → stick ────────────────────────────────────────

#[test]
fn coulomb_friction_high_mu_sticks() {
    // Heavy normal load, *small* lateral load. With μ = 1.0 the
    // Coulomb cap is huge, so the contact should be in the stick
    // regime: tangential displacement at the bottom slaves stays
    // tiny compared to the unconstrained (no-friction) baseline.
    let body_z: Real = 1.0e3;
    let body_x: Real = 50.0; // small fraction of body_z
    let setup =
        build_block_with_lateral_force(3, 3, 3, body_z, body_x, /*E*/ 1.0e6, /*ν*/ 0.3);
    let dim = 3usize;

    // Baseline: no friction → bottom face shears freely.
    let contact_free = frictional_contact_from_bottom(&setup.bottom_verts, dim, 0.0);
    let res_free = solve_augmented_system_with_friction(
        &setup.k_red,
        &contact_free,
        &setup.f_red,
        500,
        1.0e-11,
    )
    .unwrap();
    let max_ux_free: Real = setup
        .bottom_verts
        .iter()
        .map(|&v| res_free.u[v as usize * dim].abs())
        .fold(0.0, Real::max);

    // Friction-on solve: μ = 1.0.
    let contact = frictional_contact_from_bottom(&setup.bottom_verts, dim, /*μ*/ 1.0);
    let res =
        solve_augmented_system_with_friction(&setup.k_red, &contact, &setup.f_red, 2000, 1.0e-10)
            .unwrap();

    eprintln!(
        "high_mu_sticks: iters = {}, primal_res = {:.3e}, dual_res = {:.3e}",
        res.iters, res.primal_residual, res.dual_residual,
    );

    // ── Tangential displacement at slaves should be tiny ─────────
    let max_ux: Real = setup
        .bottom_verts
        .iter()
        .map(|&v| res.u[v as usize * dim].abs())
        .fold(0.0, Real::max);
    eprintln!(
        "high_mu_sticks: max |u_x| = {max_ux:.3e}, baseline (free) = {max_ux_free:.3e}, \
         ratio = {:.3e}",
        max_ux / max_ux_free.max(1.0e-30),
    );
    // The friction-on solution must be far closer to "stuck" than to
    // the free-slip baseline. We allow at most 10 % of the free-slip
    // sliding amplitude.
    assert!(
        max_ux < 0.1 * max_ux_free,
        "stick regime should suppress lateral displacement: \
         max |u_x| = {max_ux:.3e} not << free-slip {max_ux_free:.3e}"
    );

    // ── Contract: every pair inside the cone (stick), and λ_t aligned
    //    against the applied tangential force ─────────────────────
    let mu = contact.friction_coeff;
    let mut max_ratio: Real = 0.0;
    for k in 0..contact.n_active() {
        let lam_n = res.lambda_normal[k];
        let lam_tu = res.lambda_tangent[2 * k];
        let lam_tv = res.lambda_tangent[2 * k + 1];
        let mag = (lam_tu * lam_tu + lam_tv * lam_tv).sqrt();
        let cap = mu * lam_n;
        let ratio = mag / cap.max(1.0e-30);
        if ratio > max_ratio {
            max_ratio = ratio;
        }
    }
    eprintln!("high_mu_sticks: max |λ_t| / (μ λ_n) = {max_ratio:.3e}");
    assert!(
        res.friction_state
            .iter()
            .all(|s| matches!(s, FrictionState::Stick | FrictionState::Inactive)),
        "high-mu regime should not classify contacts as slipping: {:?}",
        res.friction_state
    );
    // Allow a tiny over-shoot from the projected-Uzawa increment
    // (the projection is applied *after* the gradient step so the
    // sup-norm can drift by O(tol)).
    assert!(
        max_ratio < 1.0 + 1.0e-6,
        "stick regime: λ_t must stay inside friction disc; max ratio = {max_ratio:.3e}"
    );
    // And the cone is *not* tight everywhere — strict stick means
    // average ratio well below 1.
    let avg_ratio: Real = (0..contact.n_active())
        .map(|k| {
            let lam_n = res.lambda_normal[k];
            let lam_t = (res.lambda_tangent[2 * k] * res.lambda_tangent[2 * k]
                + res.lambda_tangent[2 * k + 1] * res.lambda_tangent[2 * k + 1])
                .sqrt();
            lam_t / (mu * lam_n).max(1.0e-30)
        })
        .sum::<Real>()
        / contact.n_active() as Real;
    eprintln!("high_mu_sticks: avg |λ_t|/(μ λ_n) = {avg_ratio:.3e}");
    assert!(
        avg_ratio < 0.5,
        "stick regime average should be well inside cone; got {avg_ratio:.3e}"
    );
}

// ─── Test 3: low μ → slip ──────────────────────────────────────────

#[test]
fn coulomb_friction_low_mu_slips() {
    // Same geometry and loads as the stick test — only μ changes.
    let body_z: Real = 1.0e3;
    let body_x: Real = 50.0;
    let setup = build_block_with_lateral_force(3, 3, 3, body_z, body_x, 1.0e6, 0.3);
    let dim = 3usize;

    let contact = frictional_contact_from_bottom(&setup.bottom_verts, dim, /*μ*/ 0.1);
    let res =
        solve_augmented_system_with_friction(&setup.k_red, &contact, &setup.f_red, 5000, 1.0e-10)
            .unwrap();

    eprintln!(
        "low_mu_slips: iters = {}, primal_res = {:.3e}, dual_res = {:.3e}",
        res.iters, res.primal_residual, res.dual_residual,
    );

    // ── Tangential displacement must be non-trivial ──────────────
    let max_ux: Real = setup
        .bottom_verts
        .iter()
        .map(|&v| res.u[v as usize * dim].abs())
        .fold(0.0, Real::max);
    eprintln!("low_mu_slips: max |u_x| at slaves = {max_ux:.3e}");
    assert!(
        max_ux > 1.0e-6,
        "slip regime should produce tangential motion; got max |u_x| = {max_ux:.3e}"
    );

    // ── At slip, |λ_t| ≈ μ · λ_n on the active set ───────────────
    // Skip pairs whose λ_n is itself tiny (numerical zero — those
    // slaves separated from the floor and the cone is degenerate).
    let mu = contact.friction_coeff;
    let active_threshold = res.lambda_normal.iter().fold(0.0_f64, |a, &b| a.max(b)) * 1.0e-3;
    let mut n_at_boundary = 0usize;
    let mut n_active_pairs = 0usize;
    let mut worst_dev: Real = 0.0;
    for k in 0..contact.n_active() {
        let lam_n = res.lambda_normal[k];
        if lam_n < active_threshold {
            continue;
        }
        n_active_pairs += 1;
        let lam_t = (res.lambda_tangent[2 * k] * res.lambda_tangent[2 * k]
            + res.lambda_tangent[2 * k + 1] * res.lambda_tangent[2 * k + 1])
            .sqrt();
        let cap = mu * lam_n;
        let dev = (lam_t - cap).abs() / cap.max(1.0e-30);
        if dev > worst_dev {
            worst_dev = dev;
        }
        if dev < 5.0e-2 {
            n_at_boundary += 1;
        }
    }
    eprintln!(
        "low_mu_slips: {}/{} active pairs at cone boundary; worst dev = {:.3e}",
        n_at_boundary, n_active_pairs, worst_dev,
    );
    let n_slip = res
        .friction_state
        .iter()
        .filter(|s| matches!(s, FrictionState::Slip))
        .count();
    assert!(
        n_slip * 2 >= n_active_pairs,
        "low-mu regime should classify a majority of active contacts as slipping; \
         {n_slip}/{n_active_pairs} slipped"
    );
    // In the slip regime essentially every active pair should sit on
    // the cone boundary `|λ_t| = μ λ_n`. Demand a clear majority.
    assert!(
        n_at_boundary * 2 >= n_active_pairs,
        "slip regime: most active pairs should be on the cone boundary; \
         {n_at_boundary}/{n_active_pairs} within 5 %"
    );

    // And no pair should exceed the cone (modulo Uzawa projection
    // residual).
    let mut max_ratio: Real = 0.0;
    for k in 0..contact.n_active() {
        let lam_n = res.lambda_normal[k];
        let lam_t = (res.lambda_tangent[2 * k] * res.lambda_tangent[2 * k]
            + res.lambda_tangent[2 * k + 1] * res.lambda_tangent[2 * k + 1])
            .sqrt();
        let cap = mu * lam_n;
        let ratio = lam_t / cap.max(1.0e-30);
        if ratio > max_ratio {
            max_ratio = ratio;
        }
    }
    eprintln!("low_mu_slips: max |λ_t|/(μ λ_n) = {max_ratio:.3e}");
    assert!(
        max_ratio < 1.0 + 1.0e-6,
        "slip cap violated: max |λ_t|/(μ λ_n) = {max_ratio:.3e}"
    );
}

#[test]
fn active_set_friction_couples_contact_activation_and_coulomb_projection() {
    let setup = build_block_with_lateral_force(
        /*nx*/ 2, /*ny*/ 2, /*nz*/ 2, /*body_force_z*/ 5.0e2,
        /*body_force_x*/ 25.0, /*E*/ 1.0e6, /*ν*/ 0.3,
    );
    let dim = 3usize;
    let candidates = candidates_from_bottom(&setup.bottom_verts, dim, &setup.vertex_positions);
    let res = solve_active_set_signorini_with_friction(
        &setup.k_red,
        &candidates,
        &setup.f_red,
        /*mu_s*/ 0.2,
        /*mu_k*/ 0.1,
        /*max_outer_iters*/ 10,
        /*gap_tol*/ 1.0e-10,
        /*inner_max_iters*/ 5000,
        /*inner_tol*/ 1.0e-10,
    )
    .unwrap();

    assert!(res.trace.converged, "active-set friction should converge");
    assert!(
        !res.trace.final_active().is_empty(),
        "loaded block should activate at least one contact"
    );

    for (i, state) in res.friction_state.iter().enumerate() {
        if res.lambda_normal[i] <= 1.0e-12 {
            assert_eq!(*state, FrictionState::Inactive);
            assert_eq!(res.lambda_tangent[2 * i], 0.0);
            assert_eq!(res.lambda_tangent[2 * i + 1], 0.0);
            continue;
        }
        let lam_t = (res.lambda_tangent[2 * i] * res.lambda_tangent[2 * i]
            + res.lambda_tangent[2 * i + 1] * res.lambda_tangent[2 * i + 1])
            .sqrt();
        let cap = 0.2 * res.lambda_normal[i];
        assert!(
            lam_t <= cap + 1.0e-6,
            "candidate {i}: tangent multiplier {lam_t:.3e} exceeds static cap {cap:.3e}"
        );
        assert!(
            matches!(state, FrictionState::Stick | FrictionState::Slip),
            "active normal contact should have stick/slip friction state"
        );
    }
}

#[test]
fn rate_state_friction_steady_state_and_aging_law() {
    let params = RateStateFrictionParams {
        mu0: 0.6,
        a: 0.01,
        b: 0.015,
        v0: 1.0e-6,
        dc: 1.0e-3,
    };
    let v = 2.0e-6;
    let steady = RateStateFrictionState::steady(params, v);
    let mu = rate_state_friction_coeff(params, steady, v);
    let expected = params.mu0 + (params.a - params.b) * (v / params.v0).ln();
    assert!(
        (mu - expected).abs() < 1.0e-12,
        "steady-state rate-state coefficient mismatch: {mu} vs {expected}"
    );

    let old = RateStateFrictionState {
        theta: params.dc / params.v0,
    };
    let (next, next_mu) = advance_rate_state_friction(params, old, v, 0.1);
    assert!(
        next.theta < old.theta,
        "sliding faster than reference should age the state downward"
    );
    assert!(next_mu.is_finite());
}
