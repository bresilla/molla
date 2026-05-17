//! Stage 37.1 — Wave PDE element matrix + Newmark-β transient
//! solver contract tests. CPU-only; no `#[ignore]`.
//!
//! Spec lives in `PLAN.md` "### Stage 37.1". Three checks:
//!
//! 1. The wave element matrix equals the Laplacian element matrix
//!    when `c² = α`.
//! 2. A 1D standing-wave eigenmode evolved with Newmark-β
//!    (constant-average-acceleration) matches the analytic
//!    `(1/π) sin(πx) sin(πt)` solution to 5% at `t = 0.5 s`,
//!    confirming the period `T = 2π / (c·π) = 2 s`.
//! 3. Pinned endpoints stay pinned across many time steps.

use molla_fem::{
    BoundaryCondition, FemCell, FemMesh, FunctionSpace, NEWMARK_BETA_DEFAULT,
    NEWMARK_GAMMA_DEFAULT, assemble_global, compute_element_mass, compute_element_stiffness,
    compute_element_wave_stiffness, solve_wave_transient,
};
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

fn approx_eq(a: Real, b: Real, eps: Real) -> bool {
    (a - b).abs() <= eps
}

// ─── Test 1 — wave stiffness == Laplacian stiffness for c² = α ──────

#[test]
fn wave_element_stiffness_proportional_to_laplacian() {
    // Single triangle.
    let cell = FemCell::Tri([0, 1, 2]);
    let v = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.3, 0.1, 0.0),
        Vec3::new(0.2, 1.7, 0.0),
    ];
    let lap = compute_element_stiffness(&cell, &v, 1.0);
    let wave = compute_element_wave_stiffness(&cell, &v, 1.0);
    assert_eq!(lap.len(), wave.len());
    for (rl, rw) in lap.iter().zip(wave.iter()) {
        assert_eq!(rl.len(), rw.len());
        for (a, b) in rl.iter().zip(rw.iter()) {
            assert!(approx_eq(*a, *b, 1e-15), "tri: lap {a} ≠ wave {b}");
        }
    }

    // Single tet.
    let cell = FemCell::Tet([0, 1, 2, 3]);
    let v = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.2, 0.1),
        Vec3::new(0.1, 1.0, 0.3),
        Vec3::new(0.2, 0.1, 1.0),
    ];
    let lap = compute_element_stiffness(&cell, &v, 1.0);
    let wave = compute_element_wave_stiffness(&cell, &v, 1.0);
    for (rl, rw) in lap.iter().zip(wave.iter()) {
        for (a, b) in rl.iter().zip(rw.iter()) {
            assert!(approx_eq(*a, *b, 1e-15), "tet: lap {a} ≠ wave {b}");
        }
    }

    // Sanity: doubling c² doubles every entry.
    let wave_2 = compute_element_wave_stiffness(&cell, &v, 2.0);
    for (rw, rw2) in wave.iter().zip(wave_2.iter()) {
        for (a, b) in rw.iter().zip(rw2.iter()) {
            assert!(
                approx_eq(2.0 * a, *b, 1e-12),
                "tet: 2·wave(c²=1)={} ≠ wave(c²=2)={}",
                2.0 * a,
                b
            );
        }
    }
}

// ─── Test 2 — standing-wave eigenmode, Newmark-β ────────────────────
//
// 1D bar `x ∈ [0, L]` with `L = 1`, pinned at both ends. Wave eq:
//   `u_tt = c² u_xx`, `c² = 1`.
// IC: `u(x, 0) = 0`, `u_t(x, 0) = sin(π x)`.
// Analytic solution (separation of variables; first eigenmode has
// frequency `ω₁ = π c / L = π`, period `T = 2π / ω₁ = 2 s`):
//   `u(x, t) = (1 / (π c)) · sin(π x) · sin(π c t)`.
// At `c = 1`, `t = 0.5 s`: `sin(π/2) = 1` ⇒
//   `u(x, 0.5) = (1/π) · sin(π x)`.
//
// We model the 1D bar as a thin 2D strip (`tri_grid_2d`) and pin only
// the left + right vertical edges; top + bottom edges are free, so the
// Newmark step reduces to the 1D eigenmode in the x direction.

#[test]
fn wave_standing_mode_decays_correctly() {
    let nx = 32usize;
    // Thin strip — 2 cells in y so the y-gradient is non-degenerate.
    let ny = 2usize;
    let l: Real = 1.0;
    let h: Real = 0.1;
    let mesh = FemMesh::tri_grid_2d(nx, ny, l, h);
    let space = FunctionSpace::P1Nodal;
    let c2: Real = 1.0;
    let m = assemble_global(&mesh, space, compute_element_mass).expect("mass");
    let k = assemble_global(&mesh, space, |c, vts| {
        compute_element_wave_stiffness(c, vts, c2)
    })
    .expect("stiff");

    // Initial state: u₀ = 0, v₀ = sin(π x), a₀ = 0.
    let n_dof = space.ndof(&mesh);
    let u = vec![0.0_f64; n_dof];
    let mut v: Vec<f64> = mesh
        .vertices
        .iter()
        .map(|p| (std::f64::consts::PI * p.x).sin())
        .collect();
    let a = vec![0.0_f64; n_dof];

    // Pin only the x = 0 and x = L vertical edges.
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 || (p.x - l).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        }
    }
    // Make sure the IC honours the BCs.
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, .. } = bc {
            v[*vertex as usize] = 0.0;
        }
    }

    let dt: Real = 0.005;
    let total_t: Real = 0.5;
    let n_steps = (total_t / dt).round() as usize;

    let mut u_cur = u;
    let mut v_cur = v;
    let mut a_cur = a;
    for _ in 0..n_steps {
        let step = solve_wave_transient(
            &m,
            &k,
            &u_cur,
            &v_cur,
            &a_cur,
            dt,
            NEWMARK_BETA_DEFAULT,
            NEWMARK_GAMMA_DEFAULT,
            &bcs,
        );
        u_cur = step.u_new;
        v_cur = step.v_new;
        a_cur = step.a_new;
    }

    // Analytic at t = 0.5 s: u(x, 0.5) = sin(π x) / π.
    // We compare on the centerline (any y row). The strip ny = 2 means
    // each y-row has nx + 1 vertices; pick the middle row j = 1.
    let row_stride = nx + 1;
    let amp_analytic = 1.0 / std::f64::consts::PI;
    let mut max_rel_err: f64 = 0.0;
    let mut max_abs_err: f64 = 0.0;
    for i in 0..=nx {
        let p = mesh.vertices[1 * row_stride + i];
        let want = amp_analytic * (std::f64::consts::PI * p.x).sin();
        let got = u_cur[1 * row_stride + i];
        let abs_err = (got - want).abs();
        max_abs_err = max_abs_err.max(abs_err);
        // Skip the relative-error denominator near zero.
        if want.abs() > 0.05 {
            max_rel_err = max_rel_err.max(abs_err / want.abs());
        }
    }
    // 5% relative error tolerance per spec (eigenmode period within
    // 5%, equivalent to the displacement amplitude landing within 5%
    // of the analytic value at t = T/4).
    assert!(
        max_rel_err < 0.05,
        "standing-wave mode mismatch: max_rel_err = {max_rel_err:.4}, max_abs_err = {max_abs_err:.4e}"
    );

    // Sanity check on the temporal phase: the velocity at t = 0.5 s
    // (= T/4) should be ~0 in magnitude relative to the IC, since
    // `u_t = cos(π t) sin(π x)` ⇒ at t = 0.5 the cosine vanishes.
    let v_max: f64 = v_cur
        .iter()
        .enumerate()
        .skip(row_stride) // middle row
        .take(row_stride)
        .map(|(_, vv)| vv.abs())
        .fold(0.0_f64, f64::max);
    // The IC velocity is `sin(π x)` with peak 1.0 — by t = T/4 the
    // theoretical envelope is 0. Numerical Newmark-β leaks a few % of
    // that energy into higher modes; allow a generous bound.
    assert!(
        v_max < 0.2,
        "velocity at t = T/4 should be ≈ 0; got v_max = {v_max:.3}"
    );
}

// ─── Test 3 — pinned endpoints survive across time steps ────────────

#[test]
fn wave_pinned_endpoints_stay_pinned() {
    let nx = 16usize;
    let ny = 1usize;
    let l: Real = 1.0;
    let h: Real = 0.05;
    let mesh = FemMesh::tri_grid_2d(nx, ny, l, h);
    let space = FunctionSpace::P1Nodal;
    let c2: Real = 1.0;
    let m: BsrMatrix = assemble_global(&mesh, space, compute_element_mass).expect("mass");
    let k: BsrMatrix = assemble_global(&mesh, space, |c, vts| {
        compute_element_wave_stiffness(c, vts, c2)
    })
    .expect("stiff");

    let n_dof = space.ndof(&mesh);
    let mut u = vec![0.0_f64; n_dof];
    // Excite a non-trivial initial displacement in the interior.
    for (i, p) in mesh.vertices.iter().enumerate() {
        u[i] = (std::f64::consts::PI * p.x).sin() * 0.3;
    }
    let mut v = vec![0.0_f64; n_dof];
    let mut a = vec![0.0_f64; n_dof];

    // Pin both ends at value 0.
    let mut pinned: Vec<usize> = Vec::new();
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 || (p.x - l).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
            pinned.push(i);
            // Ensure IC is consistent with BC.
            u[i] = 0.0;
        }
    }

    let dt: Real = 0.01;
    for _ in 0..200 {
        let step = solve_wave_transient(
            &m,
            &k,
            &u,
            &v,
            &a,
            dt,
            NEWMARK_BETA_DEFAULT,
            NEWMARK_GAMMA_DEFAULT,
            &bcs,
        );
        u = step.u_new;
        v = step.v_new;
        a = step.a_new;
        for &p in &pinned {
            assert!(
                u[p].abs() < 1e-10,
                "pinned vertex {p} drifted: u = {} after a step",
                u[p]
            );
        }
    }
}
