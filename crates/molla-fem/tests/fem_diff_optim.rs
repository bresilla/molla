//! Stage 38.2 — differentiable FEM via Adam.
//!
//! Mirror of `tests/mpm_diff_optim.rs` (Stage 38.1) but on the FEM
//! elasticity side: build a cantilever beam, define a tip-displacement
//! loss, run reverse-mode through the FEM solve via
//! `fem::adjoint::{solve_adjoint, compute_grad_youngs}`, and use Adam
//! to fit Young's modulus from a wrong initial guess back to the
//! truth.
//!
//! The adjoint chain is the textbook variational one:
//!   1. Forward: K(E) · u = f.
//!   2. J(u) — here, `J = ½ (u_tip_z − u_target)²`.
//!   3. dJ/du — single non-zero entry at the tip's z-DoF.
//!   4. Adjoint solve: Kᵀ · λ = dJ/du (`solve_adjoint`).
//!   5. dJ/dE = `compute_grad_youngs(mesh, verts, u, λ, ν, E)`.
//!   6. Adam step on E.
//!
//! Stage 38 shipped `optim::AdamState` (Stage 38c); the FEM adjoint
//! itself shipped earlier in Stage 36.4. Stage 38.2 wires them
//! together end-to-end.

use molla_fem::adjoint::{compute_grad_youngs, solve_adjoint};
use molla_fem::{
    BoundaryCondition, FemMesh, FunctionSpace, assemble_global,
    compute_element_elasticity_stiffness, solve_steady_state_vector,
};
use molla_math::Real;
use molla_math::sparse::BsrMatrix;
use molla_optim::{AdamConfig, AdamState};

const POISSON: Real = 0.3;
const DIM: usize = 3;
const TIP_FORCE_Z: Real = -1.0;

fn lame_params(youngs: Real, poisson: Real) -> (Real, Real) {
    let lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = youngs / (2.0 * (1.0 + poisson));
    (lam, mu)
}

/// Build the cantilever-beam test problem: a 4×2×2 tet grid pinned on
/// the `x = 0` face, with a downward point load on the highest-x
/// centroid vertex (the "tip"). Returns the assembled stiffness
/// pattern + the load vector + the BC list + the tip DoF index.
fn build_cantilever(
    youngs: Real,
) -> (FemMesh, BsrMatrix, Vec<Real>, Vec<BoundaryCondition>, usize) {
    let mesh = FemMesh::tet_grid_3d(4, 2, 2, 4.0, 1.0, 1.0);
    let n_verts = mesh.vert_count();
    let n_dof = n_verts * DIM;

    let (lam, mu) = lame_params(youngs, POISSON);
    let k = assemble_global(&mesh, FunctionSpace::P1Vector { dim: DIM }, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam, mu)
    })
    .expect("assemble elasticity stiffness");

    // Pin the x = 0 face: all 3 components clamped to zero.
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0; DIM],
            });
        }
    }

    // Pick the "tip" vertex — highest x, closest to (y, z) = (0.5, 0.5).
    let mut tip_idx = 0usize;
    let mut best = Real::NEG_INFINITY;
    for (i, p) in mesh.vertices.iter().enumerate() {
        let yz_dist = (p.y - 0.5).powi(2) + (p.z - 0.5).powi(2);
        let score = p.x - 0.1 * yz_dist;
        if score > best {
            best = score;
            tip_idx = i;
        }
    }
    let tip_dof_z = tip_idx * DIM + 2;

    let mut f = vec![0.0_f64; n_dof];
    f[tip_dof_z] = TIP_FORCE_Z;

    (mesh, k, f, bcs, tip_dof_z)
}

/// Forward solve at a given Young's modulus — re-assembles K(E) (since
/// the local elasticity stiffness depends on E linearly) and runs the
/// CG steady-state solver. Returns `(u, K_red, K_orig)`. `K_red` is
/// what the adjoint solve will use; `K_orig` is needed by
/// `compute_grad_youngs` (which walks the *original* element matrices).
fn forward_at_e(
    mesh: &FemMesh,
    bcs: &[BoundaryCondition],
    f: &[Real],
    youngs: Real,
) -> (Vec<Real>, BsrMatrix) {
    let (lam, mu) = lame_params(youngs, POISSON);
    let k = assemble_global(mesh, FunctionSpace::P1Vector { dim: DIM }, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam, mu)
    })
    .expect("assemble");
    let u = solve_steady_state_vector(&k, f, bcs, &mesh.vertices, DIM, 5000, 1.0e-10);
    (u, k)
}

fn point_tip_load(n_dof: usize, tip_dof_z: usize, force_z: Real) -> Vec<Real> {
    let mut f = vec![0.0_f64; n_dof];
    f[tip_dof_z] = force_z;
    f
}

#[test]
fn forward_tip_displacement_scales_inversely_with_e() {
    // Sanity check: for linear elasticity at fixed load, u ∝ 1/E.
    // Doubling E should ~halve the tip deflection. This guards against
    // an assembly-side bug before we run the gradient test.
    let (mesh, _k, f, bcs, tip_dof_z) = build_cantilever(1.0e4);
    let (u1, _) = forward_at_e(&mesh, &bcs, &f, 1.0e4);
    let (u2, _) = forward_at_e(&mesh, &bcs, &f, 2.0e4);
    let tip1 = u1[tip_dof_z];
    let tip2 = u2[tip_dof_z];
    // Linear elasticity → exact 2× scaling on tip deflection.
    let ratio = tip1 / tip2;
    assert!(
        (ratio - 2.0).abs() < 0.01,
        "expected u(E)/u(2E) ≈ 2 for linear elasticity, got {ratio:.4}"
    );
}

#[test]
fn adam_fits_youngs_modulus_via_fem_adjoint() {
    // Cantilever beam with a "true" Young's modulus E_true; record the
    // tip deflection it produces; pick a wrong E_init and run Adam on
    // the FEM-adjoint gradient until E recovers.
    let e_true: Real = 1.5e4;
    let (mesh, _k0, f, bcs, tip_dof_z) = build_cantilever(e_true);
    let (u_true, _) = forward_at_e(&mesh, &bcs, &f, e_true);
    let tip_target = u_true[tip_dof_z];

    // Start 4× too stiff so the initial gradient is well-conditioned.
    // We optimize in **log space** (`ξ = ln(E)`) so the Adam step size
    // is scale-invariant — `E` lives in [1e3, 1e5], a fixed-lr Adam on
    // raw E either oscillates or stalls. Chain rule: `dJ/dξ = dJ/dE · E`.
    let e_init: Real = 4.0 * e_true;
    let mut xi = e_init.ln();
    let mut adam = AdamState::new(1);
    let cfg = AdamConfig {
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };

    let mut j_init: Real = 0.0;
    let mut j_best: Real = Real::INFINITY;
    let mut e_best: Real = e_init;
    let mut iter_best: usize = 0;
    let mut iters_taken: usize = 0;

    for it in 0..200 {
        let e: Real = xi.exp();
        // Forward + loss.
        let (u, k) = forward_at_e(&mesh, &bcs, &f, e);
        let tip = u[tip_dof_z];
        let resid = tip - tip_target;
        let j: Real = 0.5 * resid * resid;
        if it == 0 {
            j_init = j;
        }
        if j < j_best {
            j_best = j;
            e_best = e;
            iter_best = it;
        }
        if j < 1.0e-8 * j_init.max(1.0e-12) {
            iters_taken = it + 1;
            break;
        }

        // dJ/du: single non-zero entry at the tip DoF.
        let mut dj_du = vec![0.0; u.len()];
        dj_du[tip_dof_z] = resid;

        // Adjoint solve: Kᵀ · λ = dJ/du.
        let lambda = solve_adjoint(&k, &dj_du, &bcs, &mesh.vertices, DIM);

        // dJ/dE via the variational identity.
        let dj_de = compute_grad_youngs(&mesh, &mesh.vertices, &u, &lambda, POISSON, e);
        // Chain rule into log-space: dJ/dξ = dJ/dE · E.
        let dj_dxi = dj_de * e;

        let mut params = vec![xi];
        adam.step(&mut params, &[dj_dxi], &cfg);
        xi = params[0];
        iters_taken = it + 1;
    }

    let rel_err = (e_best - e_true).abs() / e_true;
    let loss_ratio = j_best / j_init.max(1.0e-30);
    eprintln!(
        "fem_diff_optim: E_init = {:.3e}, E_true = {:.3e}, E_best = {:.3e} (iter {}), \
         rel_err = {:.2}%, J_init = {:.3e}, J_best = {:.3e} ({:.2e}× reduction), \
         iters = {}",
        e_init,
        e_true,
        e_best,
        iter_best,
        rel_err * 100.0,
        j_init,
        j_best,
        loss_ratio,
        iters_taken,
    );

    assert!(
        rel_err < 0.10 || loss_ratio < 0.01,
        "Adam failed to recover E within 10% (rel_err = {:.4}) AND failed to reduce J by \
         99% (loss_ratio = {:.4})",
        rel_err,
        loss_ratio,
    );
}

#[test]
fn adam_fits_tip_force_via_fem_adjoint() {
    // Fixed-material cantilever with a known downward tip force. We
    // observe its target tip displacement, then start from the wrong
    // load magnitude and optimize the force parameter through the same
    // FEM adjoint used for material fitting. This is a load/actuation
    // optimization gate, not another Young's-modulus gate.
    let e_fixed: Real = 1.5e4;
    let (mesh, _k0, _f0, bcs, tip_dof_z) = build_cantilever(e_fixed);
    let n_dof = mesh.vert_count() * DIM;

    let force_true: Real = 1.0;
    let f_true = point_tip_load(n_dof, tip_dof_z, -force_true);
    let (u_true, _) = forward_at_e(&mesh, &bcs, &f_true, e_fixed);
    let tip_target = u_true[tip_dof_z];

    // Keep the load magnitude positive by optimizing ξ = ln(force).
    let force_init: Real = 0.25 * force_true;
    let mut xi = force_init.ln();
    let mut adam = AdamState::new(1);
    let cfg = AdamConfig {
        lr: 0.08,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };

    let mut j_init: Real = 0.0;
    let mut j_best: Real = Real::INFINITY;
    let mut force_best: Real = force_init;
    let mut iter_best = 0usize;

    for it in 0..180 {
        let force = xi.exp();
        let f = point_tip_load(n_dof, tip_dof_z, -force);
        let (u, k) = forward_at_e(&mesh, &bcs, &f, e_fixed);
        let resid = u[tip_dof_z] - tip_target;
        let j = 0.5 * resid * resid;
        if it == 0 {
            j_init = j;
        }
        if j < j_best {
            j_best = j;
            force_best = force;
            iter_best = it;
        }
        if j < 1.0e-10 * j_init.max(1.0e-30) {
            break;
        }

        let mut dj_du = vec![0.0; u.len()];
        dj_du[tip_dof_z] = resid;
        let lambda = solve_adjoint(&k, &dj_du, &bcs, &mesh.vertices, DIM);

        // f_tip_z = -force, so dJ/dforce = λᵀ df/dforce = -λ_tip_z.
        let dj_dforce = -lambda[tip_dof_z];

        // One early finite-difference guard keeps the load-gradient
        // sign and scale honest without making this a separate test.
        if it == 0 {
            let h = (force * 1.0e-4).max(1.0e-6);
            let f_p = point_tip_load(n_dof, tip_dof_z, -(force + h));
            let f_m = point_tip_load(n_dof, tip_dof_z, -(force - h));
            let (u_p, _) = forward_at_e(&mesh, &bcs, &f_p, e_fixed);
            let (u_m, _) = forward_at_e(&mesh, &bcs, &f_m, e_fixed);
            let r_p = u_p[tip_dof_z] - tip_target;
            let r_m = u_m[tip_dof_z] - tip_target;
            let j_p = 0.5 * r_p * r_p;
            let j_m = 0.5 * r_m * r_m;
            let dj_fd = (j_p - j_m) / (2.0 * h);
            let rel = (dj_dforce - dj_fd).abs() / dj_fd.abs().max(1.0e-30);
            assert!(
                rel < 5.0e-3,
                "tip-force adjoint gradient disagrees with FD: analytic = {dj_dforce:.6e}, \
                 fd = {dj_fd:.6e}, rel = {rel:.3e}",
            );
        }

        let dj_dxi = dj_dforce * force;
        let mut params = vec![xi];
        adam.step(&mut params, &[dj_dxi], &cfg);
        xi = params[0];
    }

    let rel_force_err = (force_best - force_true).abs() / force_true;
    let loss_ratio = j_best / j_init.max(1.0e-30);
    eprintln!(
        "fem_diff_optim load fit: F_init = {:.3e}, F_true = {:.3e}, F_best = {:.3e} \
         (iter {}), rel_err = {:.2}%, J_init = {:.3e}, J_best = {:.3e} \
         ({:.2e}× reduction)",
        force_init,
        force_true,
        force_best,
        iter_best,
        rel_force_err * 100.0,
        j_init,
        j_best,
        loss_ratio,
    );

    assert!(
        rel_force_err < 0.05 || loss_ratio < 1.0e-4,
        "Adam failed to recover tip force within 5% (rel_err = {:.4}) and failed to \
         reduce J by 1e4× (loss_ratio = {:.4e})",
        rel_force_err,
        loss_ratio,
    );
}

#[test]
fn analytic_grad_youngs_matches_central_difference() {
    // Sanity check on the gradient pipeline: at a fixed E, the analytic
    // dJ/dE from the adjoint method must match a central-difference
    // oracle to within FEM-CG-tolerance + finite-difference noise.
    let e0: Real = 1.5e4;
    let (mesh, _, f, bcs, tip_dof_z) = build_cantilever(e0);
    let (u_target, _) = forward_at_e(&mesh, &bcs, &f, 2.0 * e0);
    let tip_target = u_target[tip_dof_z];

    // Loss + analytic gradient at e0.
    let (u0, k0) = forward_at_e(&mesh, &bcs, &f, e0);
    let resid0 = u0[tip_dof_z] - tip_target;
    let j0 = 0.5 * resid0 * resid0;
    let mut dj_du = vec![0.0; u0.len()];
    dj_du[tip_dof_z] = resid0;
    let lambda = solve_adjoint(&k0, &dj_du, &bcs, &mesh.vertices, DIM);
    let dj_de_analytic = compute_grad_youngs(&mesh, &mesh.vertices, &u0, &lambda, POISSON, e0);

    // Central difference: J(E + h) − J(E − h) / (2h).
    let h: Real = 1.0; // E ~ 1e4, h ~ 1.0 is 1e-4 relative
    let (u_p, _) = forward_at_e(&mesh, &bcs, &f, e0 + h);
    let (u_m, _) = forward_at_e(&mesh, &bcs, &f, e0 - h);
    let resid_p = u_p[tip_dof_z] - tip_target;
    let resid_m = u_m[tip_dof_z] - tip_target;
    let j_p = 0.5 * resid_p * resid_p;
    let j_m = 0.5 * resid_m * resid_m;
    let dj_de_fd = (j_p - j_m) / (2.0 * h);

    let rel_err = ((dj_de_analytic - dj_de_fd).abs() / dj_de_fd.abs().max(1.0e-30)).abs();
    eprintln!(
        "fem_diff_optim FD check: J0 = {:.4e}, dJ/dE analytic = {:.6e}, FD = {:.6e}, rel err = {:.3e}",
        j0, dj_de_analytic, dj_de_fd, rel_err,
    );
    // CG tolerance is 1e-10 on each forward solve → propagated through
    // the adjoint identity we expect agreement to ~1e-3 relative. The
    // tip force is tiny so the displacement scale is also tiny — be
    // gentle on the threshold.
    assert!(
        rel_err < 0.05,
        "FEM adjoint dJ/dE disagrees with FD: analytic = {dj_de_analytic:.6e}, FD = \
         {dj_de_fd:.6e}, rel err = {rel_err:.3e}",
    );
}
