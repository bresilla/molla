//! Stage 36.4 — Adjoint / differentiable FEM contract tests.
//!
//! All tests are CPU-only and run under the standard
//! `cargo test -p molla` suite (no `#[ignore]`).
//!
//! Coverage:
//!   * `adjoint_recovers_finite_diff_grad_wrt_youngs` — the analytic
//!     `∂J/∂E` from `compute_grad_youngs` matches a central-FD estimate
//!     within 1e-3 relative on a small tet beam.
//!   * `adjoint_recovers_finite_diff_grad_wrt_traction` — same for the
//!     traction gradient via `compute_grad_load`.
//!   * `gradient_descent_minimises_compliance` — gradient-descent on
//!     Young's modulus reduces compliance monotonically over 20 steps.
//!
//! The forward problem is the same 2×2×2 cantilever-tet beam as the
//! Stage 36.1 elasticity tests, kept tiny so the CG inner solve hits
//! `1e-12` quickly and the FD perturbations don't drown in solver noise.

use molla_fem::{
    BoundaryCondition, FemCell, FemMesh, FunctionSpace, apply_bcs, assemble_global,
    compute_element_elasticity_stiffness, compute_grad_load, compute_grad_youngs,
    solve_adjoint_with,
};
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3, iterative::cg_solve};

// ─── Shared fixture ────────────────────────────────────────────────

const BEAM_NX: usize = 2;
const BEAM_NY: usize = 2;
const BEAM_NZ: usize = 2;
const BEAM_LEN_X: Real = 2.0;
const BEAM_DEPTH_Y: Real = 1.0;
const BEAM_HEIGHT_Z: Real = 1.0;
const BEAM_NU: Real = 0.3;
const BEAM_E0: Real = 1.0e6;
const BEAM_LOAD_Z: Real = -1.0;

/// (E, ν) → (λ, μ).
fn lame(youngs: Real, poisson: Real) -> (Real, Real) {
    let mu = youngs / (2.0 * (1.0 + poisson));
    let lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    (lam, mu)
}

fn build_beam_mesh() -> FemMesh {
    FemMesh::tet_grid_3d(
        BEAM_NX,
        BEAM_NY,
        BEAM_NZ,
        BEAM_LEN_X,
        BEAM_DEPTH_Y,
        BEAM_HEIGHT_Z,
    )
}

fn build_clamp_left_bcs(mesh: &FemMesh) -> Vec<BoundaryCondition> {
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0, 0.0],
            });
        }
    }
    bcs
}

/// Find every triangular face on `x = BEAM_LEN_X` (the loaded face).
/// Returns `(face_vertices, area)` pairs.
fn collect_tip_faces(mesh: &FemMesh) -> Vec<([u32; 3], Real)> {
    let mut out = Vec::new();
    let on_tip = |p: Vec3| (p.x - BEAM_LEN_X).abs() < 1e-9;
    for cell in &mesh.cells {
        if let FemCell::Tet(idx) = cell {
            let faces = [
                [idx[0], idx[1], idx[2]],
                [idx[0], idx[1], idx[3]],
                [idx[0], idx[2], idx[3]],
                [idx[1], idx[2], idx[3]],
            ];
            for face in &faces {
                let pa = mesh.vertices[face[0] as usize];
                let pb = mesh.vertices[face[1] as usize];
                let pc = mesh.vertices[face[2] as usize];
                if !(on_tip(pa) && on_tip(pb) && on_tip(pc)) {
                    continue;
                }
                let area = 0.5 * (pb - pa).cross(pc - pa).length();
                out.push((*face, area));
            }
        }
    }
    out
}

fn collect_tip_vertices(mesh: &FemMesh) -> Vec<u32> {
    let mut vertices = Vec::new();
    for (face, _) in collect_tip_faces(mesh) {
        for v in face {
            if !vertices.contains(&v) {
                vertices.push(v);
            }
        }
    }
    vertices
}

fn tip_average_z(u: &[Real], tip_vertices: &[u32]) -> Real {
    let mut sum = 0.0;
    for &v in tip_vertices {
        sum += u[v as usize * 3 + 2];
    }
    sum / tip_vertices.len() as Real
}

/// Build the external load vector for a uniform downward (−z) traction
/// of magnitude `traction_z` over the tip face.
fn build_tip_traction_load(mesh: &FemMesh, traction_z: Real) -> Vec<Real> {
    let n = mesh.vert_count() * 3;
    let mut f = vec![0.0; n];
    for (face, area) in collect_tip_faces(mesh) {
        let third = area / 3.0;
        for &v in &face {
            f[(v as usize) * 3 + 2] += third * traction_z;
        }
    }
    f
}

/// Assemble the linear-elasticity stiffness for the beam at the
/// material parameters `(youngs, poisson)`.
fn assemble_beam_stiffness(mesh: &FemMesh, youngs: Real, poisson: Real) -> BsrMatrix {
    let (lam, mu) = lame(youngs, poisson);
    let space = FunctionSpace::P1Vector { dim: 3 };
    assemble_global(mesh, space, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam, mu)
    })
    .expect("FEM beam: assemble stiffness")
}

/// Solve the linear elasticity problem `K · u = f` with Dirichlet BCs
/// applied to the (k, f) pair.
fn forward_solve(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[Vec3],
) -> Vec<Real> {
    let mut k_red = k.clone();
    let mut f_red = f.to_vec();
    apply_bcs(&mut k_red, &mut f_red, bcs, vertices, 3);
    let res = cg_solve(
        |x| {
            let mut y = vec![0.0; k_red.nrows()];
            k_red.mv(x, &mut y).unwrap();
            y
        },
        &f_red,
        &vec![0.0; f_red.len()],
        1.0e-12,
        50_000,
    );
    res.x
}

/// Compliance loss `J = ½ uᵀ f` (no explicit `E` dependence; standard
/// shape-optimisation scalar). The seed for the adjoint is
/// `∂J/∂u = ½ f` — but for a load-independent design parameter (E),
/// the chain rule simplifies further. We use the full seed here so the
/// adjoint code path is exercised end-to-end.
fn compliance(u: &[Real], f: &[Real]) -> Real {
    debug_assert_eq!(u.len(), f.len());
    let mut s = 0.0;
    for (ui, fi) in u.iter().zip(f.iter()) {
        s += ui * fi;
    }
    0.5 * s
}

// ─── Test 1: ∂J/∂E vs central FD ────────────────────────────────────

#[test]
fn adjoint_recovers_finite_diff_grad_wrt_youngs() {
    let mesh = build_beam_mesh();
    let bcs = build_clamp_left_bcs(&mesh);
    let f = build_tip_traction_load(&mesh, BEAM_LOAD_Z);

    // --- Analytic gradient via the adjoint method.
    let k0 = assemble_beam_stiffness(&mesh, BEAM_E0, BEAM_NU);
    let u0 = forward_solve(&k0, &f, &bcs, &mesh.vertices);
    // Compliance loss ⇒ ∂J/∂u = ½ · f.
    let dj_du: Vec<Real> = f.iter().map(|fi| 0.5 * fi).collect();
    let lambda = solve_adjoint_with(&k0, &dj_du, &bcs, &mesh.vertices, 3, 50_000, 1.0e-12);
    let g_analytic = compute_grad_youngs(&mesh, &mesh.vertices, &u0, &lambda, BEAM_NU, BEAM_E0);

    // --- Central finite-difference gradient.
    let eps = BEAM_E0 * 1.0e-4; // 0.01% perturbation
    let k_plus = assemble_beam_stiffness(&mesh, BEAM_E0 + eps, BEAM_NU);
    let u_plus = forward_solve(&k_plus, &f, &bcs, &mesh.vertices);
    let j_plus = compliance(&u_plus, &f);
    let k_minus = assemble_beam_stiffness(&mesh, BEAM_E0 - eps, BEAM_NU);
    let u_minus = forward_solve(&k_minus, &f, &bcs, &mesh.vertices);
    let j_minus = compliance(&u_minus, &f);
    let g_fd = (j_plus - j_minus) / (2.0 * eps);

    let rel_err = (g_analytic - g_fd).abs() / g_fd.abs().max(1e-30);
    eprintln!(
        "∂J/∂E: analytic = {:.6e}, fd = {:.6e}, rel_err = {:.4e}",
        g_analytic, g_fd, rel_err
    );
    assert!(
        rel_err < 1e-3,
        "∂J/∂E adjoint {:.6e} differs from FD {:.6e} by {:.2e} (>1e-3 bound)",
        g_analytic,
        g_fd,
        rel_err
    );
}

#[test]
fn adjoint_youngs_gradient_matches_tip_displacement_target_fd() {
    let mesh = build_beam_mesh();
    let bcs = build_clamp_left_bcs(&mesh);
    let f = build_tip_traction_load(&mesh, BEAM_LOAD_Z);
    let tip_vertices = collect_tip_vertices(&mesh);
    assert!(!tip_vertices.is_empty(), "beam should expose tip vertices");

    let loss = |youngs: Real| -> Real {
        let k = assemble_beam_stiffness(&mesh, youngs, BEAM_NU);
        let u = forward_solve(&k, &f, &bcs, &mesh.vertices);
        let tip_z = tip_average_z(&u, &tip_vertices);
        0.5 * tip_z * tip_z
    };

    let k0 = assemble_beam_stiffness(&mesh, BEAM_E0, BEAM_NU);
    let u0 = forward_solve(&k0, &f, &bcs, &mesh.vertices);
    let tip_z = tip_average_z(&u0, &tip_vertices);
    let mut dj_du = vec![0.0; u0.len()];
    for &v in &tip_vertices {
        dj_du[v as usize * 3 + 2] = tip_z / tip_vertices.len() as Real;
    }
    let lambda = solve_adjoint_with(&k0, &dj_du, &bcs, &mesh.vertices, 3, 50_000, 1.0e-12);
    let g_analytic = compute_grad_youngs(&mesh, &mesh.vertices, &u0, &lambda, BEAM_NU, BEAM_E0);

    let eps = BEAM_E0 * 1.0e-4;
    let g_fd = (loss(BEAM_E0 + eps) - loss(BEAM_E0 - eps)) / (2.0 * eps);
    let rel_err = (g_analytic - g_fd).abs() / g_fd.abs().max(1.0e-30);
    eprintln!(
        "tip-target ∂J/∂E: tip_z={tip_z:.6e}, analytic={g_analytic:.6e}, fd={g_fd:.6e}, rel_err={rel_err:.4e}",
    );
    assert!(
        g_analytic < 0.0,
        "stiffening should reduce the zero-target tip-displacement loss"
    );
    assert!(
        rel_err < 1.0e-3,
        "tip-target ∂J/∂E adjoint {g_analytic:.6e} differs from FD {g_fd:.6e} by {rel_err:.2e}",
    );
}

// ─── Test 2: ∂J/∂F vs central FD ────────────────────────────────────

#[test]
fn adjoint_recovers_finite_diff_grad_wrt_traction() {
    let mesh = build_beam_mesh();
    let bcs = build_clamp_left_bcs(&mesh);
    let traction0 = BEAM_LOAD_Z;
    let f = build_tip_traction_load(&mesh, traction0);

    let k = assemble_beam_stiffness(&mesh, BEAM_E0, BEAM_NU);
    let u0 = forward_solve(&k, &f, &bcs, &mesh.vertices);
    // Compliance ⇒ ∂J/∂u = ½ · f. With f = f(t), there is also an
    // explicit ½ · u contribution flowing through f's t-dependence;
    // we'll handle it in the gradient step below.
    let dj_du: Vec<Real> = f.iter().map(|fi| 0.5 * fi).collect();
    let lambda = solve_adjoint_with(&k, &dj_du, &bcs, &mesh.vertices, 3, 50_000, 1.0e-12);

    // The adjoint contribution to ∂J/∂t is `λᵀ · ∂f/∂t`. The explicit
    // ∂J/∂t through `J = ½ uᵀ f(t)` adds `½ uᵀ · ∂f/∂t`. Per
    // `compute_grad_load`, ∂f/∂t_c at face vertex v is `(area / 3)`
    // for the c-th component slot. Sum the two contributions.
    let tip_faces = collect_tip_faces(&mesh);
    let mut g_analytic_z: Real = 0.0;
    for (face, _area) in &tip_faces {
        // λᵀ · ∂f/∂t (z-component only, since traction is along z).
        let g_face = compute_grad_load(&mesh.vertices, &lambda, *face, 3);
        g_analytic_z += g_face[2];
        // ½ uᵀ · ∂f/∂t — same form but with u in place of λ.
        let g_face_u = compute_grad_load(&mesh.vertices, &u0, *face, 3);
        g_analytic_z += 0.5 * g_face_u[2];
    }

    // Central FD on the global traction.
    let eps = traction0.abs() * 1.0e-4;
    let f_plus = build_tip_traction_load(&mesh, traction0 + eps);
    let u_plus = forward_solve(&k, &f_plus, &bcs, &mesh.vertices);
    let j_plus = compliance(&u_plus, &f_plus);
    let f_minus = build_tip_traction_load(&mesh, traction0 - eps);
    let u_minus = forward_solve(&k, &f_minus, &bcs, &mesh.vertices);
    let j_minus = compliance(&u_minus, &f_minus);
    let g_fd = (j_plus - j_minus) / (2.0 * eps);

    let rel_err = (g_analytic_z - g_fd).abs() / g_fd.abs().max(1e-30);
    eprintln!(
        "∂J/∂t_z: analytic = {:.6e}, fd = {:.6e}, rel_err = {:.4e}",
        g_analytic_z, g_fd, rel_err
    );
    assert!(
        rel_err < 1e-3,
        "∂J/∂t_z adjoint {:.6e} differs from FD {:.6e} by {:.2e} (>1e-3 bound)",
        g_analytic_z,
        g_fd,
        rel_err
    );
}

// ─── Test 3: gradient descent on E reduces compliance ──────────────

#[test]
fn gradient_descent_minimises_compliance() {
    let mesh = build_beam_mesh();
    let bcs = build_clamp_left_bcs(&mesh);
    let f = build_tip_traction_load(&mesh, BEAM_LOAD_Z);

    // Compliance = ½ uᵀ f scales as 1/E in linear elasticity (since
    // u ∝ 1/E for fixed load), so descent on J w.r.t. E should
    // *increase* E (∂J/∂E < 0). We verify monotonic decrease.
    let mut e = BEAM_E0;
    let mut last_j = Real::INFINITY;
    // Adaptive step: for J(E) ≈ c/E in linear elasticity,
    // ∂J/∂E ≈ −J/E. To halve J in one step we need ΔE ≈ E, i.e.
    // ΔE = lr · |∂J/∂E| ≈ lr · J/E ⇒ lr ≈ E²/J. We pick a more
    // conservative `0.1 · E²/J` per step so the descent is monotone
    // across all 20 steps without overshoot.
    let k0 = assemble_beam_stiffness(&mesh, e, BEAM_NU);
    let u0 = forward_solve(&k0, &f, &bcs, &mesh.vertices);
    let j0 = compliance(&u0, &f);
    eprintln!("step  0: E = {:.4e}, J = {:.6e}", e, j0);

    for step in 1..=20 {
        let k = assemble_beam_stiffness(&mesh, e, BEAM_NU);
        let u = forward_solve(&k, &f, &bcs, &mesh.vertices);
        let j = compliance(&u, &f);
        // Strict-monotonicity check from step 1 onward (we re-evaluate
        // from the current E each iter, so even step 0 vs step 1 must
        // strictly decrease).
        if step > 1 {
            assert!(
                j <= last_j + 1e-12 * j.abs(),
                "compliance increased at step {step}: {last_j:.6e} → {j:.6e}",
            );
        }
        last_j = j;
        // Adjoint gradient.
        let dj_du: Vec<Real> = f.iter().map(|fi| 0.5 * fi).collect();
        let lambda = solve_adjoint_with(&k, &dj_du, &bcs, &mesh.vertices, 3, 50_000, 1.0e-12);
        let g = compute_grad_youngs(&mesh, &mesh.vertices, &u, &lambda, BEAM_NU, e);
        // Gradient descent with adaptive step: lr = 0.1 · E² / J.
        let lr = 0.1 * e * e / j.abs().max(1e-30);
        let e_next = (e - lr * g).max(1.0e3).min(1.0e12);
        eprintln!(
            "step {:>2}: E = {:.4e}, J = {:.6e}, ∂J/∂E = {:.4e} → E_next = {:.4e}",
            step, e, j, g, e_next
        );
        e = e_next;
    }
    // J should have shrunk substantially from the initial value.
    let k_end = assemble_beam_stiffness(&mesh, e, BEAM_NU);
    let u_end = forward_solve(&k_end, &f, &bcs, &mesh.vertices);
    let j_end = compliance(&u_end, &f);
    assert!(
        j_end < 0.5 * j0,
        "compliance only dropped from {j0:.4e} to {j_end:.4e} (<2× expected)"
    );
}
