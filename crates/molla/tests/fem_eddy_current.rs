//! Stage 48 v1 — block-real GMRES eddy-current solver (Jacobi-PC).
//!
//! Two contract tests:
//!
//! 1. **Magnetostatics limit** — with `σ = 0`, the eddy-current solver
//!    should reproduce a magnetostatics-style solution to within
//!    floating-point precision.
//! 2. **Skin-depth decay** — solve `curl(ν curl A) + iωσA = 0` on a
//!    slab with imposed tangential `A_y` on the top face, free
//!    `n × A = 0` on the bottom + sides. The interior field magnitude
//!    `|A_y(z)|` must decay with a characteristic length that matches
//!    the analytic skin depth `δ = √(2/(ωμσ))` to coarse-mesh
//!    tolerance.

use molla::fem::{
    compute_curl_per_cell, solve_eddy_current_steady, solve_magnetostatics_steady,
    BoundaryCondition, EddyCurrentConfig, FemCell, FemMesh, NedelecDofMap,
};
use molla::math::{Real, Vec3};

// ─── Test 1: magnetostatics limit ────────────────────────────────────

#[test]
fn eddy_with_zero_sigma_matches_magnetostatics() {
    let mesh = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let n_cells = mesh.cells.len();
    // Pick a non-trivial source.
    let mut j = vec![Vec3::ZERO; n_cells];
    for (i, cell) in mesh.cells.iter().enumerate() {
        let centroid = cell_centroid(cell, &mesh.vertices);
        if centroid.z > 0.3 && centroid.z < 0.7 {
            j[i] = Vec3::new(0.0, 1.0, 0.0);
        }
    }
    let nu = vec![1.0; n_cells];
    let sigma = vec![0.0; n_cells];
    let j_zero = vec![Vec3::ZERO; n_cells];

    // Reference magnetostatics solve.
    let ref_res =
        solve_magnetostatics_steady(&mesh, &j, &nu, &[]).expect("magneto solve");

    // Eddy-current solve with σ = 0, ω arbitrary (off-diagonal vanishes).
    let cfg = EddyCurrentConfig {
        omega: 1.5,
        ..Default::default()
    };
    let eddy_res = solve_eddy_current_steady(
        &mesh, &j, &j_zero, &nu, &sigma, &[], &cfg,
    )
    .expect("eddy solve");

    // Compare the **curl** of the two solutions per cell (gauge-
    // invariant — the gauge ambiguity in the un-Coulomb-gauged
    // curl-curl problem means the two Krylov solvers may converge to
    // potentials differing by a gradient, but their curls must agree
    // to convergence-tolerance accuracy).
    let map = NedelecDofMap::build(&mesh);
    let b_eddy = compute_curl_per_cell(&mesh, &map, &eddy_res.a_re);
    let b_ref = compute_curl_per_cell(&mesh, &map, &ref_res.a);
    let mut max_err: Real = 0.0;
    let mut max_ref: Real = 0.0;
    for i in 0..b_eddy.len() {
        let e = (b_eddy[i] - b_ref[i]).length();
        max_err = max_err.max(e);
        max_ref = max_ref.max(b_ref[i].length());
    }
    eprintln!(
        "eddy(σ=0) vs magneto: curl max abs err = {max_err:.3e}, ref scale = {max_ref:.3e}, \
         eddy iters = {}, magneto iters = {}",
        eddy_res.iters, ref_res.iters
    );
    assert!(
        max_err / max_ref.max(1e-30) < 1e-5,
        "eddy curl deviates from magnetostatics by {max_err:.3e} (rel = {:.3e})",
        max_err / max_ref.max(1e-30)
    );
    // Imaginary part should be machine-zero — the equations decouple
    // entirely when ω σ = 0 and the imaginary source is zero.
    let max_im = eddy_res
        .a_im
        .iter()
        .map(|v| v.abs())
        .fold(0.0, Real::max);
    assert!(max_im < 1e-6, "a_im = {max_im} not negligible");
}

// ─── Test 2: skin-depth decay ────────────────────────────────────────

/// Imposed-tangential-field skin-depth test. Geometry: a thin slab of
/// conductor occupying `(0, L) × (0, L) × (0, L_z)` (with `L_z ≫ δ`).
/// Boundary conditions:
/// - Top face `z = L_z`: tangential `A_y = 1` (imposed real
///   Dirichlet on every edge that lies on the top face and points
///   primarily in the y-direction).
/// - Other faces: `n × A = 0` (zero Dirichlet — natural).
///
/// The analytic infinite-half-space skin-depth solution gives
/// `A_y(z) ~ exp(-(L_z - z) / δ)` (real part decays into the
/// conductor with characteristic length `δ = √(2 / (ωμσ))`).
///
/// We verify the decay rate by fitting `log|A_y(z)|` vs `(L_z - z)` to
/// a line and checking the slope matches `-1/δ` to ~50% (the coarse
/// mesh + finite slab dominate the error; we only rule out
/// wrong-scaling bugs here).
#[test]
fn skin_depth_decay_matches_analytic() {
    // Pick parameters so δ is well-resolved by the mesh AND the slab
    // is at least a few skin-depths thick. For dimensionless σ ≈ 1,
    // ω ≈ 50, μ = 1 ⇒ δ = √(2 / 50) ≈ 0.2. A slab of `L_z = 1.0` with
    // `nz = 12` gives ~5 skin-depths of thickness and `~2.5` cells per
    // δ in z — coarse but the test only checks the order of magnitude.
    let nx = 4usize;
    let ny = 4usize;
    let nz = 12usize;
    let l_x: Real = 0.4;
    let l_y: Real = 0.4;
    let l_z: Real = 1.0;
    let mesh = FemMesh::tet_grid_3d(nx, ny, nz, l_x, l_y, l_z);
    let n_cells = mesh.cells.len();

    let omega: Real = 50.0;
    let sigma: Real = 1.0;
    let mu: Real = 1.0;
    let nu: Real = 1.0 / mu;
    let delta = (2.0 / (omega * mu * sigma)).sqrt();
    eprintln!("Analytic skin depth δ = {delta:.4} (slab L_z = {l_z}, nz = {nz})");

    let nu_per_cell = vec![nu; n_cells];
    let sigma_per_cell = vec![sigma; n_cells];
    let j_zero = vec![Vec3::ZERO; n_cells];

    // Build BC list: on the top face (z = L_z), impose A_y = 1 for
    // every edge that points primarily in y. On the bottom (z = 0),
    // impose A_y = 0 (so it doesn't interfere with the eigen-mode).
    let map = NedelecDofMap::build(&mesh);
    let z_top = l_z;
    let z_bot = 0.0;
    let eps_geom: Real = 1.0e-6;
    let mut bcs: Vec<BoundaryCondition> = Vec::new();
    for ([a, b], dof) in map.edge_id.iter().map(|(k, v)| (*k, *v)) {
        let pa = mesh.vertices[a as usize];
        let pb = mesh.vertices[b as usize];
        let edge_dir = pb - pa;
        let on_top = (pa.z - z_top).abs() < eps_geom && (pb.z - z_top).abs() < eps_geom;
        let on_bot = (pa.z - z_bot).abs() < eps_geom && (pb.z - z_bot).abs() < eps_geom;
        // Project onto y-direction; "primarily y" means |dy| > |dx|, |dz|.
        let dx = edge_dir.x.abs();
        let dy = edge_dir.y.abs();
        let dz = edge_dir.z.abs();
        let mainly_y = dy > dx && dy > dz;
        if on_top && mainly_y {
            // Tangential integral of A_y = 1 along an edge parallel to
            // y of length L is just dy = edge_dir.y. The Nedelec DoF
            // value equals that integral.
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: dof,
                value: edge_dir.y,
            });
        } else if on_bot && mainly_y {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: dof,
                value: 0.0,
            });
        }
    }
    eprintln!("Skin-depth test: applied {} BC pins on top + bottom", bcs.len());

    let cfg = EddyCurrentConfig {
        omega,
        tol: 1.0e-9,
        max_iter: 5000,
        restart: 50,
        ..Default::default()
    };
    let r = solve_eddy_current_steady(
        &mesh, &j_zero, &j_zero, &nu_per_cell, &sigma_per_cell, &bcs, &cfg,
    )
    .expect("eddy solve");
    eprintln!(
        "Skin-depth solve: iters = {}, residual = {:.3e}",
        r.iters, r.residual
    );
    assert!(r.residual < 1e-6, "did not converge: residual = {}", r.residual);

    // Sample |A_y(z)| at the midplane line x = L_x/2, y in the middle.
    // We do this by averaging the magnitude `sqrt(a_re² + a_im²)` of
    // every y-edge whose mid-z falls within a thin z-slab.
    let n_z_samples = 8usize;
    let mut samples_z: Vec<Real> = vec![0.0; n_z_samples];
    let mut samples_abs: Vec<Real> = vec![0.0; n_z_samples];
    let mut samples_count: Vec<usize> = vec![0; n_z_samples];
    for ([a, b], dof) in map.edge_id.iter().map(|(k, v)| (*k, *v)) {
        let pa = mesh.vertices[a as usize];
        let pb = mesh.vertices[b as usize];
        let edge_dir = pb - pa;
        let dx = edge_dir.x.abs();
        let dy = edge_dir.y.abs();
        let dz = edge_dir.z.abs();
        let mainly_y = dy > dx && dy > dz;
        if !mainly_y {
            continue;
        }
        let mid = 0.5 * (pa + pb);
        // Restrict to a vertical strip near the middle of the slab.
        if (mid.x - l_x * 0.5).abs() > 0.1 * l_x {
            continue;
        }
        if (mid.y - l_y * 0.5).abs() > 0.4 * l_y {
            continue;
        }
        let bin = ((mid.z / l_z) * n_z_samples as Real).floor() as usize;
        let bin = bin.min(n_z_samples - 1);
        let dof = dof as usize;
        let a_re = r.a_re[dof];
        let a_im = r.a_im[dof];
        // Divide by `edge_dir.y` to undo the tangential-integral scale.
        let scale = edge_dir.y;
        if scale.abs() < 1.0e-9 {
            continue;
        }
        let mag = ((a_re / scale).powi(2) + (a_im / scale).powi(2)).sqrt();
        samples_z[bin] += mid.z;
        samples_abs[bin] += mag;
        samples_count[bin] += 1;
    }
    for k in 0..n_z_samples {
        if samples_count[k] > 0 {
            samples_z[k] /= samples_count[k] as Real;
            samples_abs[k] /= samples_count[k] as Real;
        }
    }
    eprintln!("Skin-depth samples (z, |A_y|):");
    for k in 0..n_z_samples {
        eprintln!(
            "  bin {k}: z = {:.3}, |A_y| = {:.3e}, n = {}",
            samples_z[k], samples_abs[k], samples_count[k]
        );
    }

    // Fit `log|A_y(z)|` vs `(L_z - z)` to a line, ignoring near-zero
    // samples (deep interior where numerics dominates).
    let mut xs: Vec<Real> = Vec::new();
    let mut ys: Vec<Real> = Vec::new();
    for k in 0..n_z_samples {
        if samples_count[k] == 0 || samples_abs[k] < 1.0e-4 {
            continue;
        }
        xs.push(l_z - samples_z[k]);
        ys.push(samples_abs[k].ln());
    }
    assert!(xs.len() >= 3, "not enough samples to fit");
    let n_fit = xs.len() as Real;
    let sx: Real = xs.iter().sum();
    let sy: Real = ys.iter().sum();
    let sxx: Real = xs.iter().map(|v| v * v).sum();
    let sxy: Real = xs.iter().zip(ys.iter()).map(|(a, b)| a * b).sum();
    let slope = (n_fit * sxy - sx * sy) / (n_fit * sxx - sx * sx);
    let measured_delta = -1.0 / slope;
    eprintln!(
        "Slope of log|A_y| vs (L_z - z): {slope:.4}, measured δ = {measured_delta:.4}, \
         analytic δ = {delta:.4}",
    );
    // Rule out wrong-sign or wrong-order-of-magnitude bugs. The slope
    // must be negative (decay into the conductor) and the recovered δ
    // must lie within a factor of 2 of the analytic value — the
    // coarse mesh (nz = 12, ~2.5 cells per δ) is the limiting factor.
    assert!(slope < 0.0, "slope = {slope} should be negative (decay)");
    assert!(
        measured_delta > 0.5 * delta && measured_delta < 2.0 * delta,
        "measured δ = {measured_delta} far from analytic δ = {delta}"
    );
}

// ─── Helpers ─────────────────────────────────────────────────────────

fn cell_centroid(cell: &FemCell, vertices: &[Vec3]) -> Vec3 {
    match cell {
        FemCell::Tet(v) => {
            let p = vertices[v[0] as usize]
                + vertices[v[1] as usize]
                + vertices[v[2] as usize]
                + vertices[v[3] as usize];
            p / 4.0
        }
        FemCell::Tri(v) => {
            let p = vertices[v[0] as usize]
                + vertices[v[1] as usize]
                + vertices[v[2] as usize];
            p / 3.0
        }
    }
}
