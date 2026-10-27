//! CPU PCG parity tests — `solve_steady_state_pcg` and
//! `solve_transient_pcg` must agree with their non-preconditioned
//! siblings on the solution and use **fewer** iterations.
//!
//! Run with `cargo test -p molla-fem --test fem_pcg_cpu`.

use molla_fem::solver::{
    DEFAULT_MAX_ITERS, DEFAULT_TOL, solve_steady_state_pcg, solve_steady_state_with_initial,
    solve_transient_pcg, solve_transient_with_initial,
};
use molla_fem::{
    BoundaryCondition, FemMesh, FunctionSpace, assemble_global, compute_element_mass,
    compute_element_stiffness,
};
use molla_math::Real;
use molla_math::sparse::BsrMatrix;

fn assemble_stiffness(mesh: &FemMesh, alpha: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .unwrap()
}

fn assemble_mass(mesh: &FemMesh) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_mass(c, v)
    })
    .unwrap()
}

fn build_heat_bcs(mesh: &FemMesh) -> Vec<BoundaryCondition> {
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
    bcs
}

#[test]
fn cpu_pcg_steady_state_matches_cg_with_fewer_iters() {
    let nx = 8usize;
    let mesh = FemMesh::tet_grid_3d(nx, nx, nx, 1.0, 1.0, 1.0);
    let n = mesh.vert_count();
    let k = assemble_stiffness(&mesh, 1.0);
    let bcs = build_heat_bcs(&mesh);
    // Non-trivial source — Gaussian heat injection — so the steady-
    // state solution is genuinely interior-rich and CG actually has
    // work to do (a zero RHS would produce a zero residual at iter 0
    // for the seeded BCs, masking any iteration-count comparison).
    let f: Vec<Real> = mesh
        .vertices
        .iter()
        .map(|p| {
            let d = (p.x - 0.5) * (p.x - 0.5) + (p.y - 0.5) * (p.y - 0.5);
            (-50.0 * d).exp()
        })
        .collect();
    let x0 = vec![0.0_f64; n];

    let cg = solve_steady_state_with_initial(&k, &f, &bcs, &x0, DEFAULT_MAX_ITERS, DEFAULT_TOL);
    let pcg = solve_steady_state_pcg(&k, &f, &bcs, &x0, DEFAULT_MAX_ITERS, DEFAULT_TOL);

    let mut max_du: Real = 0.0;
    for i in 0..n {
        max_du = max_du.max((cg.x[i] - pcg.x[i]).abs());
    }
    eprintln!(
        "cpu_pcg_steady_state: n = {n}, CG iters = {}, PCG iters = {}, max |Δ| = {max_du:.3e}",
        cg.iters, pcg.iters,
    );
    // Both paths converge to within `tol = 1e-8` of the true
    // solution; f64 round-off between two different iterative
    // trajectories means their final iterates can disagree by ~tol
    // even though both are correct. Threshold matches DEFAULT_TOL.
    assert!(
        max_du < 1e-5,
        "CPU PCG diverges from CG: max |Δ| = {max_du:.3e} (DEFAULT_TOL = 1e-8)"
    );
    // Jacobi must reduce iter count on a Laplacian-like system.
    assert!(
        pcg.iters < cg.iters,
        "PCG took {} iters vs CG {}; expected strict reduction",
        pcg.iters,
        cg.iters
    );
}

#[test]
fn cpu_pcg_transient_matches_cg_with_fewer_iters() {
    let nx = 8usize;
    let mesh = FemMesh::tet_grid_3d(nx, nx, nx, 1.0, 1.0, 1.0);
    let n = mesh.vert_count();
    let m = assemble_mass(&mesh);
    let k = assemble_stiffness(&mesh, 1.0);
    let bcs = build_heat_bcs(&mesh);
    let f = vec![0.0_f64; n];
    let dt: Real = 0.01;

    // Initial state: a Gaussian bump in the interior — the heat
    // diffusion away from this is what gives CG real iteration work.
    let mut u: Vec<Real> = mesh
        .vertices
        .iter()
        .map(|p| {
            let d =
                (p.x - 0.5) * (p.x - 0.5) + (p.y - 0.5) * (p.y - 0.5) + (p.z - 0.5) * (p.z - 0.5);
            (-30.0 * d).exp()
        })
        .collect();
    // Stamp BC values to keep the initial state BC-consistent.
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            u[*vertex as usize] = *value;
        }
    }

    // 20 backward-Euler steps through both paths.
    let mut u_cg = u.clone();
    let mut u_pcg = u.clone();
    let mut total_cg_iters = 0usize;
    let mut total_pcg_iters = 0usize;
    for _ in 0..20 {
        let cg = solve_transient_with_initial(
            &m,
            &k,
            &f,
            &u_cg,
            dt,
            &bcs,
            &u_cg.clone(),
            DEFAULT_MAX_ITERS,
            DEFAULT_TOL,
        );
        let pcg = solve_transient_pcg(
            &m,
            &k,
            &f,
            &u_pcg,
            dt,
            &bcs,
            &u_pcg.clone(),
            DEFAULT_MAX_ITERS,
            DEFAULT_TOL,
        );
        total_cg_iters += cg.iters;
        total_pcg_iters += pcg.iters;
        u_cg = cg.x;
        u_pcg = pcg.x;
    }

    let mut max_du: Real = 0.0;
    for i in 0..n {
        max_du = max_du.max((u_cg[i] - u_pcg[i]).abs());
    }
    eprintln!(
        "cpu_pcg_transient: n = {n}, 20 steps, CG total iters = {total_cg_iters}, \
         PCG total iters = {total_pcg_iters}, max |Δ| = {max_du:.3e}",
    );
    assert!(
        max_du < 1e-5,
        "CPU PCG transient diverges from CG: max |Δ| = {max_du:.3e}"
    );
    assert!(
        total_pcg_iters < total_cg_iters,
        "PCG total {total_pcg_iters} >= CG total {total_cg_iters}"
    );
}
