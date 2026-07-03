//! INSPECTION.md F15 / WarpFemMixedElasticity — nearly-incompressible
//! mixed (u, p) Taylor-Hood elasticity (`solve_mixed_elasticity_steady`).
//!
//! Oracles:
//! 1. CONSTANT-STRAIN PATCH TEST (exact): the linear field
//!    `u = (ε·x, −ν·ε·y, −ν·ε·z)` solves homogeneous elasticity with
//!    zero body force, so with the exact field imposed on every
//!    boundary P2 node the interior must reproduce it to solver
//!    tolerance, and the pressure must equal `λ·div u` (constant).
//! 2. ANTI-LOCKING: a bending 2D cantilever's deflection must survive
//!    ν → ½ (a volumetrically locking element collapses it by 10–100×).
//! 3. Incompressibility: div u shrinks as ν → ½.

use molla_fem::assembly::matvec;
use molla_fem::compute_edges;
use molla_fem::{
    FemMesh, MixedTaylorHood, StokesBc, StokesSimpleConfig, assemble_velocity_mass,
    solve_mixed_elasticity_steady,
};
use molla_math::{Real, Vec3};

fn lame(e: Real, nu: Real) -> (Real, Real) {
    let mu = e / (2.0 * (1.0 + nu));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    (mu, lambda)
}

/// Every P2 velocity node (vertices then edge midpoints) with its
/// position.
fn p2_nodes(mesh: &FemMesh) -> Vec<Vec3> {
    let (edges, _) = compute_edges(mesh);
    let mut out: Vec<Vec3> = mesh.vertices.clone();
    for e in &edges {
        out.push((mesh.vertices[e[0] as usize] + mesh.vertices[e[1] as usize]) * 0.5);
    }
    out
}

#[test]
fn mixed_elasticity_constant_strain_patch_test() {
    let mesh = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 3);
    let (mu, lambda) = lame(1.0e4, 0.3);
    let eps = 0.01_f64;
    let nu = 0.3_f64;
    let exact = |p: Vec3| Vec3::new(eps * p.x, -nu * eps * p.y, -nu * eps * p.z);

    let nodes = p2_nodes(&mesh);
    assert_eq!(nodes.len() * 3, th.n_velocity);
    let on_boundary = |p: Vec3| {
        p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9
            || p.z.abs() < 1e-9
            || (p.z - 1.0).abs() < 1e-9
    };
    let mut bcs = Vec::new();
    let mut interior = Vec::new();
    for (k, p) in nodes.iter().enumerate() {
        if on_boundary(*p) {
            let u = exact(*p);
            bcs.push(StokesBc::LidVelocity {
                node: k as u32,
                velocity: vec![u.x, u.y, u.z],
            });
        } else {
            interior.push(k);
        }
    }
    assert!(interior.len() > 10, "mesh must have interior P2 nodes");

    let f = vec![0.0; th.n_velocity];
    let cfg = StokesSimpleConfig {
        tol_inner: 1.0e-12,
        tol_outer: 1.0e-10,
        ..Default::default()
    };
    let res =
        solve_mixed_elasticity_steady(&mesh, &th, mu, lambda, &f, &bcs, &cfg).expect("mixed solve");

    let mut max_err: Real = 0.0;
    for &k in &interior {
        let want = exact(nodes[k]);
        let got = Vec3::new(res.u[3 * k], res.u[3 * k + 1], res.u[3 * k + 2]);
        max_err = max_err.max((got - want).length());
    }
    let p_expect = lambda * eps * (1.0 - 2.0 * nu);
    let mut p_err: Real = 0.0;
    for &pv in &res.p {
        p_err = p_err.max((pv - p_expect).abs());
    }
    eprintln!(
        "patch test: max interior |Δu| = {max_err:.3e} (scale {:.1e}), max |p − λ·divu| = {p_err:.3e} (p = {p_expect:.3e})",
        eps
    );
    // Taylor-Hood reproduces linear fields exactly — machine noise
    // only (measured 2.8e-14 on the 1e-2 field).
    assert!(
        max_err < 1.0e-10,
        "patch test displacement error too large: {max_err:.3e}"
    );
    assert!(
        p_err < 1.0e-6 * p_expect.abs(),
        "patch test pressure error too large: {p_err:.3e} vs {p_expect:.3e}"
    );
}

fn cantilever_tip_deflection(nu: Real) -> (Real, Real) {
    // Beam [0,4]×[0,1], clamped at x = 0, uniform downward body load.
    let mesh = FemMesh::tri_grid_2d(16, 4, 4.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 2);
    let (mu, lambda) = lame(1.0e4, nu);
    let nodes = p2_nodes(&mesh);
    assert_eq!(nodes.len() * 2, th.n_velocity);

    let mut bcs = Vec::new();
    for (k, p) in nodes.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(StokesBc::NoSlip { node: k as u32 });
        }
    }
    // f = M_v · (0, −g) per node — a consistent uniform body load.
    let m_v = assemble_velocity_mass(&mesh, &th).expect("mass");
    let mut b_field = vec![0.0; th.n_velocity];
    for k in 0..nodes.len() {
        b_field[2 * k + 1] = -100.0;
    }
    let f = matvec(&m_v, &b_field);

    let cfg = StokesSimpleConfig {
        tol_inner: 1.0e-11,
        tol_outer: 1.0e-9,
        ..Default::default()
    };
    let res =
        solve_mixed_elasticity_steady(&mesh, &th, mu, lambda, &f, &bcs, &cfg).expect("mixed solve");
    // Tip deflection: mean u_y over nodes at x = 4.
    let mut tip = 0.0;
    let mut count = 0;
    for (k, p) in nodes.iter().enumerate() {
        if (p.x - 4.0).abs() < 1e-9 {
            tip += res.u[2 * k + 1];
            count += 1;
        }
    }
    (tip / count as Real, res.div_residual)
}

#[test]
fn mixed_elasticity_does_not_lock_near_incompressible() {
    let (tip_c, _div_c) = cantilever_tip_deflection(0.3);
    let (tip_i, _div_i) = cantilever_tip_deflection(0.4999);
    eprintln!("cantilever tip: ν=0.3 → {tip_c:.5}, ν=0.4999 → {tip_i:.5}");
    assert!(tip_c < -1.0e-3, "compressible beam must deflect: {tip_c}");
    // Bending stiffness scales ≈ E/(1−ν²) in plane strain: going from
    // ν = 0.3 to 0.4999 changes the deflection by ~20%, not orders of
    // magnitude. A locking element collapses the ratio to ≪ 0.5.
    let ratio = tip_i / tip_c;
    assert!(
        ratio > 0.5 && ratio < 1.5,
        "near-incompressible deflection collapsed (locking?): ratio = {ratio:.3}"
    );
}
