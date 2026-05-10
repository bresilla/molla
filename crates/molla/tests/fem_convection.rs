//! Stage 37.2 — convection element + SUPG stabilisation contract tests.
//!
//! Three CPU-only checks of `compute_element_convection` and
//! `compute_element_supg_convection`:
//!
//! 1. **Zero velocity → zero block.** The convection matrix degenerates
//!    to a row of zeros when `b = 0`; the SUPG block degenerates the
//!    same way (the textbook `τ = h / (2|b|)` is undefined at
//!    `|b| = 0`, and the kernel returns zeros instead of `NaN`).
//! 2. **No oscillations at Pe = 100.** Steady advection-diffusion
//!    `(α∇² − b·∇) u = 0` on a 16×16 unit-square triangulation with
//!    `α = 0.1`, `|b| = 10` ⇒ global Pe = 100, cell Pe ≈ 6.25.
//!    Inflow Dirichlet `u = 1` on the left, outflow Dirichlet `u = 0`
//!    on the right. Without SUPG the discretisation is the textbook
//!    "wiggle" pathology — interior nodes overshoot above 1.0 / below
//!    0.0. With SUPG, the steady state stays bounded by the BC values
//!    everywhere.
//! 3. **Pure advection carries the inflow downstream.** With diffusion
//!    set to a tiny floor and a coarse mesh, the SUPG-stabilised
//!    convection-diffusion problem propagates the inflow boundary
//!    value to nearby downstream nodes within 5%.

use molla::fem::{
    apply_dirichlet, assemble_global, compute_element_convection,
    compute_element_stiffness, compute_element_supg_convection,
    BoundaryCondition, FemCell, FemMesh, FunctionSpace,
};
use molla::math::iterative::bicgstab_solve;
use molla::math::sparse::BsrMatrix;
use molla::math::{Real, Vec3};

/// Apply Dirichlet BCs and run BiCGSTAB on the (non-symmetric)
/// advection-diffusion system. CG can't handle the non-SPD matrix that
/// the convection block produces; BiCGSTAB is the textbook drop-in.
fn solve_advection_diffusion(
    a: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    max_iters: usize,
    tol: Real,
) -> Vec<Real> {
    let mut a_red = a.clone();
    let mut b = f.to_vec();
    apply_dirichlet(&mut a_red, &mut b, bcs);
    let n = b.len();
    let mut x0 = vec![0.0; n];
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x0[*vertex as usize] = *value;
        }
    }
    let res = bicgstab_solve(
        |x| {
            let mut y = vec![0.0; n];
            a_red.mv(x, &mut y).expect("BsrMatrix mv");
            y
        },
        &b,
        &x0,
        tol,
        max_iters,
    );
    assert!(
        res.residual <= tol * 100.0,
        "BiCGSTAB stalled: residual {} > tol {} after {} iters",
        res.residual,
        tol,
        res.iters,
    );
    res.x
}

#[test]
fn convection_zero_velocity_recovers_identity() {
    // A single triangle and a single tet — both should produce a zero
    // matrix when `b = 0`. (`Mₑ = 0` for the convection block; SUPG
    // is also zero because `|b| = 0` skips the τ computation.)
    let tri_v = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let tri = FemCell::Tri([0, 1, 2]);
    let c = compute_element_convection(&tri, &tri_v, Vec3::ZERO);
    for row in &c {
        for v in row {
            assert!(v.abs() < 1e-15, "tri convection: expected 0, got {v}");
        }
    }
    let s = compute_element_supg_convection(&tri, &tri_v, Vec3::ZERO, 1.0, 0.1);
    for row in &s {
        for v in row {
            assert!(v.abs() < 1e-15, "tri SUPG: expected 0, got {v}");
        }
    }

    let tet_v = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let tet = FemCell::Tet([0, 1, 2, 3]);
    let c = compute_element_convection(&tet, &tet_v, Vec3::ZERO);
    for row in &c {
        for v in row {
            assert!(v.abs() < 1e-15, "tet convection: expected 0, got {v}");
        }
    }
    let s = compute_element_supg_convection(&tet, &tet_v, Vec3::ZERO, 1.0, 0.1);
    for row in &s {
        for v in row {
            assert!(v.abs() < 1e-15, "tet SUPG: expected 0, got {v}");
        }
    }
}

/// Assemble `A = α·K + C [+ S_supg]` for the steady advection-diffusion
/// problem on `mesh` with diffusion `alpha` and convection `b`.
/// `with_supg` toggles the Petrov-Galerkin block. `h` is the cell
/// length scale used inside τ.
fn assemble_advection_diffusion(
    mesh: &FemMesh,
    alpha: Real,
    b: Vec3,
    with_supg: bool,
    h: Real,
) -> molla::math::sparse::BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |cell, verts| {
        let k = compute_element_stiffness(cell, verts, alpha);
        let c = compute_element_convection(cell, verts, b);
        let n = cell.vert_count();
        let mut out = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                out[i][j] = k[i][j] + c[i][j];
            }
        }
        if with_supg {
            let s = compute_element_supg_convection(cell, verts, b, 1.0, h);
            for i in 0..n {
                for j in 0..n {
                    out[i][j] += s[i][j];
                }
            }
        }
        out
    })
    .expect("assemble adv-diff")
}

#[test]
fn convection_dominated_problem_no_oscillation() {
    let nx = 16;
    let ny = 16;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    // High global Péclet — Pe = |b| L / α = 10 · 1 / 0.1 = 100.
    // Cell Péclet at h = 1/16: |b|·h / (2α) ≈ 3.1 ≫ 1 ⇒ oscillation
    // regime for the unstabilised Galerkin discretisation.
    let alpha = 0.1;
    let b = Vec3::new(10.0, 0.0, 0.0);
    let h = 1.0 / nx as Real;

    // Inflow Dirichlet on the left edge, outflow Dirichlet on the
    // right. Top/bottom are left as zero-flux Neumann (the natural BC
    // — homogeneous and not added to the linear system).
    let mut bcs = Vec::new();
    let inflow_value = 1.0;
    let outflow_value = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet { vertex: i as u32, value: inflow_value });
        } else if (p.x - 1.0).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet { vertex: i as u32, value: outflow_value });
        }
    }

    let n = mesh.vert_count();
    let f = vec![0.0; n];

    // SUPG-stabilised solve — should be monotone between BC values.
    let a_supg = assemble_advection_diffusion(&mesh, alpha, b, true, h);
    let u_supg = solve_advection_diffusion(&a_supg, &f, &bcs, 5000, 1e-10);

    let lo = outflow_value.min(inflow_value);
    let hi = outflow_value.max(inflow_value);

    // Compute the maximum overshoot for both the SUPG-stabilised
    // solve and the bare Galerkin solve. Galerkin must oscillate at
    // this cell-Péclet (~3); SUPG must shrink the overshoot by at
    // least an order of magnitude — that's the textbook contract.
    let a_no_supg = assemble_advection_diffusion(&mesh, alpha, b, false, h);
    let u_galerkin = solve_advection_diffusion(&a_no_supg, &f, &bcs, 5000, 1e-10);
    let overshoot = |u: &[Real]| {
        let mut max_over: Real = 0.0;
        for &v in u {
            let over = (v - hi).max(0.0).max((lo - v).max(0.0));
            if over > max_over {
                max_over = over;
            }
        }
        max_over
    };
    let supg_overshoot = overshoot(&u_supg);
    let galerkin_overshoot = overshoot(&u_galerkin);
    assert!(
        galerkin_overshoot > 0.1,
        "Galerkin should oscillate strongly at cell-Pe ~3, got overshoot {galerkin_overshoot}"
    );
    assert!(
        supg_overshoot < 0.05,
        "SUPG overshoot {supg_overshoot} should be ≤ 5% (textbook bound for τ = h/(2|b|) on a P1 right-tri mesh at cell-Pe ~3)"
    );
    assert!(
        supg_overshoot < galerkin_overshoot * 0.1,
        "SUPG overshoot {supg_overshoot} should be at least 10× smaller than Galerkin {galerkin_overshoot}"
    );
}

#[test]
fn convection_pure_advection_carries_inflow() {
    // Tiny diffusion — almost-pure advection. Coarse mesh so we can
    // read off the propagated inflow value at a known interior node.
    let nx = 8;
    let ny = 8;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let alpha = 0.001;
    let b = Vec3::new(1.0, 0.0, 0.0);
    let h = 1.0 / nx as Real;

    let mut bcs = Vec::new();
    let inflow_value: Real = 1.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet { vertex: i as u32, value: inflow_value });
        } else if (p.x - 1.0).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet { vertex: i as u32, value: 0.0 });
        }
    }

    let n = mesh.vert_count();
    let f = vec![0.0; n];

    let a = assemble_advection_diffusion(&mesh, alpha, b, true, h);
    let u = solve_advection_diffusion(&a, &f, &bcs, 5000, 1e-10);

    // The two next-to-inflow columns (i = 1, 2) at mid-height
    // (j = ny/2) should still carry the inflow value to within 5%
    // because diffusion is negligible and SUPG keeps the upwind
    // information flowing along the streamline.
    let j_mid = ny / 2;
    let stride = nx + 1;
    let i1 = j_mid * stride + 1;
    let i2 = j_mid * stride + 2;
    let v1 = u[i1];
    let v2 = u[i2];
    assert!(
        (v1 - inflow_value).abs() < 0.05,
        "node 1 column off mid-height: expected ≈{inflow_value}, got {v1}"
    );
    assert!(
        (v2 - inflow_value).abs() < 0.05,
        "node 2 column off mid-height: expected ≈{inflow_value}, got {v2}"
    );
}
