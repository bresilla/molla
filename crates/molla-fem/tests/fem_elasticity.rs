//! Stage 36.1 — FEM elasticity contract tests.
//!
//! All tests are CPU-only and run as part of the standard
//! `cargo test -p molla` suite (no `#[ignore]`).
//!
//! Coverage:
//!   * Rigid-body motion → zero stress (kernel of the linear stiffness).
//!   * 2D plane-strain cantilever beam → tip deflection within 5% of
//!     analytic Euler-Bernoulli `δ = P L³ / (3 E I)`.
//!   * 3D tet cantilever beam → same 5% bound.
//!   * SVK → linear elasticity in the small-strain limit (<1%).

use molla_fem::{
    BoundaryCondition, FemCell, FemMesh, FunctionSpace, apply_bcs, assemble_global,
    compute_element_elasticity_stiffness, compute_element_mooney_rivlin_force,
    compute_element_mooney_rivlin_tangent, compute_element_neo_hookean_force,
    compute_element_neo_hookean_tangent, compute_element_st_venant_kirchhoff_stress,
    solve_nonlinear_static, solve_steady_state_vector,
};
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// ─── Helpers ────────────────────────────────────────────────────────

/// Convert (E, ν) → Lamé parameters (λ, μ).
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

/// `nx × ny` quad grid spanning `[0, w] × [0, h]`, each quad split
/// into 4 tris around a centre vertex (the "Union-Jack" / "criss-
/// cross" triangulation). This breaks the diagonal-direction bias
/// of [`FemMesh::tri_grid_2d`] and substantially mitigates P1 shear
/// locking on bending problems.
fn build_criss_cross_tri_grid(nx: usize, ny: usize, w: Real, h: Real) -> FemMesh {
    let mut vertices: Vec<Vec3> = Vec::with_capacity((nx + 1) * (ny + 1) + nx * ny);
    let dx = w / nx as Real;
    let dy = h / ny as Real;
    // Corner vertices first (so x = 0 / x = w detection works on
    // exact equality).
    for j in 0..=ny {
        for i in 0..=nx {
            vertices.push(Vec3::new(i as Real * dx, j as Real * dy, 0.0));
        }
    }
    let centre_offset = vertices.len();
    // Centre vertices.
    for j in 0..ny {
        for i in 0..nx {
            vertices.push(Vec3::new(
                (i as Real + 0.5) * dx,
                (j as Real + 0.5) * dy,
                0.0,
            ));
        }
    }
    let corner = |i: usize, j: usize| -> u32 { (j * (nx + 1) + i) as u32 };
    let centre = |i: usize, j: usize| -> u32 { (centre_offset + j * nx + i) as u32 };

    let mut cells: Vec<FemCell> = Vec::with_capacity(4 * nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let v00 = corner(i, j);
            let v10 = corner(i + 1, j);
            let v11 = corner(i + 1, j + 1);
            let v01 = corner(i, j + 1);
            let vc = centre(i, j);
            // 4 tris: south, east, north, west — all CCW.
            cells.push(FemCell::Tri([v00, v10, vc]));
            cells.push(FemCell::Tri([v10, v11, vc]));
            cells.push(FemCell::Tri([v11, v01, vc]));
            cells.push(FemCell::Tri([v01, v00, vc]));
        }
    }
    FemMesh { vertices, cells }
}

fn assemble_elasticity(mesh: &FemMesh, dim: usize, lam: Real, mu: Real) -> BsrMatrix {
    let space = FunctionSpace::P1Vector { dim };
    assemble_global(mesh, space, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam, mu)
    })
    .expect("assemble elasticity")
}

// ─── Test 1: rigid-body motion → zero stress ────────────────────────
//
// Apply a uniform displacement u = (0.1, 0, 0) to every vertex of a
// tet cube. The linear stiffness should be in the kernel of pure
// translation: K · u = 0 (rotation is also in the kernel for the
// linearised model — but checking translation is sufficient for the
// "zero in the null space" contract).

#[test]
fn tet_elasticity_constant_displacement_zero_stress() {
    let mesh = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let (lam, mu) = lame(1.0e6, 0.3);
    let k = assemble_elasticity(&mesh, 3, lam, mu);

    // u = (0.1, 0, 0) for every vertex.
    let n = mesh.vert_count();
    let mut u = vec![0.0; 3 * n];
    for i in 0..n {
        u[3 * i] = 0.1;
        // y, z components stay zero.
    }
    let ku = matvec(&k, &u);
    let max_residual: Real = ku.iter().fold(0.0, |a: Real, &b| a.max(b.abs()));
    assert!(
        max_residual < 1e-6,
        "K · u_translation should be ~0 — got max |Ku| = {max_residual}"
    );
}

// ─── Test 2: 2D plane-strain cantilever beam ────────────────────────
//
// Beam dimensions: L (along x) × h (along y), modelled as a
// `nx × ny` quad grid → 2·nx·ny tris. Left edge (x = 0) is fully
// clamped (Dirichlet u = 0). Right edge (x = L) carries a downward
// traction loading → resulting tip deflection compared to the
// Euler-Bernoulli formula δ = P L³ / (3 E I).
//
// I (second moment) for a unit-thickness rectangular cross-section
// of height h is I = h³ / 12. Per unit thickness, the line load is
// w (N/m); total tip load P = w · h_distributed_along_load. We push
// a constant traction over the right edge.
//
// Plane-strain → effective E_ps = E / (1 − ν²). Use that in the
// analytic formula so the Bernoulli prediction is consistent with
// the FEM model.

#[test]
fn cantilever_beam_2d_tip_deflection() {
    // Beam dimensions — slender (L/h = 20) so Euler-Bernoulli is a
    // good prediction (the shear-deformation term is O((h/L)²) and
    // <0.5% at this aspect ratio).
    let len_x: Real = 20.0;
    let h: Real = 1.0;
    // P1 plane-strain tris notoriously shear-lock under bending. The
    // standard mitigation without bumping to P2 is the criss-cross
    // (Union-Jack) triangulation: each quad is split into 4 tris
    // sharing a centre vertex. Symmetry across both diagonals breaks
    // the directional bias of the regular mesh and the converged
    // tip deflection drops within ~3% of analytic.
    let nx = 80usize;
    let ny = 8usize;
    let mesh = build_criss_cross_tri_grid(nx, ny, len_x, h);
    let dim = 2usize;
    let space = FunctionSpace::P1Vector { dim };

    // Material — linear elasticity. Use Poisson = 0 here so the
    // plane-strain D-matrix has no axial-axial coupling and the
    // analytic Bernoulli formula δ = P L³ / (3 E I) applies
    // directly (no 1/(1−ν²) correction). This decouples the test
    // from shear-locking-amplification effects of axial-shear
    // coupling at non-zero ν, isolating the bending-element
    // accuracy.
    let youngs: Real = 1.0e6;
    let poisson: Real = 0.0;
    let (lam, mu) = lame(youngs, poisson);
    let k = assemble_elasticity(&mesh, dim, lam, mu);

    // BCs — left edge clamped, right edge loaded with a downward
    // traction. Total load P = traction_y · h (per unit thickness).
    // We pick a load small enough that the linear theory is accurate.
    let traction_y: Real = -1.0; // N / m of edge length.
    let total_p = traction_y * h; // total tip load magnitude (signed).

    let mut bcs: Vec<BoundaryCondition> = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0],
            });
        }
    }

    // Apply traction over the edges along x = L. We need to find the
    // line segments (pairs of consecutive vertices) on that edge.
    // Easier path: visit each tri cell, look for an edge with both
    // endpoints on x = L, and integrate the traction over it.
    let n = space.ndof(&mesh);
    let mut f_ext = vec![0.0; n];
    let on_right = |p: Vec3| (p.x - len_x).abs() < 1e-9;
    for cell in &mesh.cells {
        if let FemCell::Tri(idx) = cell {
            let mut on_edge: Vec<u32> = Vec::new();
            for &v in idx {
                if on_right(mesh.vertices[v as usize]) {
                    on_edge.push(v);
                }
            }
            if on_edge.len() == 2 {
                // Integrate constant traction over a line segment of
                // length L_e. Distributed load on the two endpoints:
                // f_a = f_b = (L_e / 2) · t.
                let pa = mesh.vertices[on_edge[0] as usize];
                let pb = mesh.vertices[on_edge[1] as usize];
                let len = (pb - pa).length();
                let half = 0.5 * len;
                let va = on_edge[0] as usize;
                let vb = on_edge[1] as usize;
                f_ext[va * dim + 1] += half * traction_y;
                f_ext[vb * dim + 1] += half * traction_y;
            }
        }
    }

    // Solve.
    let u = solve_steady_state_vector(&k, &f_ext, &bcs, &mesh.vertices, dim, 20_000, 1.0e-10);

    // Read tip deflection — average of all vertices on x = L.
    let mut sum_uy = 0.0;
    let mut cnt = 0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        if (p.x - len_x).abs() < 1e-9 {
            sum_uy += u[i * dim + 1];
            cnt += 1;
        }
    }
    let tip_uy = sum_uy / cnt as Real;

    // Analytic Euler-Bernoulli for plane-strain: δ = P L³ / (3 E_ps I)
    // with E_ps = E / (1 − ν²) and I = h³ / 12 (unit thickness).
    let i_inertia = h * h * h / 12.0;
    let e_ps = youngs / (1.0 - poisson * poisson);
    let delta_analytic = total_p * len_x.powi(3) / (3.0 * e_ps * i_inertia);

    let rel_err = (tip_uy - delta_analytic).abs() / delta_analytic.abs();
    eprintln!(
        "cantilever_2d: tip_uy = {:.6e}, analytic = {:.6e}, rel_err = {:.4}",
        tip_uy, delta_analytic, rel_err
    );
    assert!(
        rel_err < 0.05,
        "2D cantilever tip deflection {:.6e} differs from analytic {:.6e} by {:.2}% (>5% bound)",
        tip_uy,
        delta_analytic,
        100.0 * rel_err
    );
}

// ─── Test 3: 3D tet cantilever beam ────────────────────────────────
//
// Same setup as above, but a 3D tet beam of dimensions L × b × h
// (along x × y × z). Left face (x = 0) clamped, right face (x = L)
// carries a downward (−z) traction.
//
// Analytic: δ = P L³ / (3 E I) with I = b · h³ / 12.

#[test]
fn cantilever_beam_3d_tip_deflection() {
    // Beam dimensions — moderate slenderness (L/h = 6). P1 tets
    // (Kuhn 6-tet decomposition of each hex) shear-lock badly at
    // higher slenderness; a Stage 36.3 P2 element would be the
    // textbook fix, but for v1 we limit the test to a regime where
    // P1 can hit 5% with a workable mesh size (~14k tets).
    let len_x: Real = 6.0;
    let b: Real = 1.0;
    let h: Real = 1.0;
    let nx = 40usize;
    let ny = 10usize;
    let nz = 10usize;
    let mesh = FemMesh::tet_grid_3d(nx, ny, nz, len_x, b, h);
    let dim = 3usize;
    let space = FunctionSpace::P1Vector { dim };

    let youngs: Real = 1.0e6;
    // Poisson = 0 removes the plane-strain-style axial-axial
    // coupling that amplifies P1 shear locking. Stage 36.3 P2
    // elements lift this restriction.
    let poisson: Real = 0.0;
    let (lam, mu) = lame(youngs, poisson);
    let k = assemble_elasticity(&mesh, dim, lam, mu);

    // BCs — clamp left face.
    let mut bcs: Vec<BoundaryCondition> = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0, 0.0],
            });
        }
    }

    // Traction on the right face: total area = b · h. Downward
    // (−z) traction per unit area. Total tip load:
    // P = traction_z · b · h.
    let traction_z: Real = -1.0;
    let total_p = traction_z * b * h;

    let n = space.ndof(&mesh);
    let mut f_ext = vec![0.0; n];
    // For each tet on the right face, find the triangular face whose
    // 3 vertices are all on x = L, and add the surface integral.
    // Each tet has 4 faces; iterate them.
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
                let on_x_l = (pa.x - len_x).abs() < 1e-9
                    && (pb.x - len_x).abs() < 1e-9
                    && (pc.x - len_x).abs() < 1e-9;
                if !on_x_l {
                    continue;
                }
                let area = 0.5 * (pb - pa).cross(pc - pa).length();
                let third = area / 3.0;
                for &v in face {
                    let v = v as usize;
                    f_ext[v * dim + 2] += third * traction_z;
                }
            }
        }
    }

    let u = solve_steady_state_vector(&k, &f_ext, &bcs, &mesh.vertices, dim, 50_000, 1.0e-10);

    let mut sum_uz = 0.0;
    let mut cnt = 0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        if (p.x - len_x).abs() < 1e-9 {
            sum_uz += u[i * dim + 2];
            cnt += 1;
        }
    }
    let tip_uz = sum_uz / cnt as Real;

    let i_inertia = b * h * h * h / 12.0;
    let delta_analytic = total_p * len_x.powi(3) / (3.0 * youngs * i_inertia);

    let rel_err = (tip_uz - delta_analytic).abs() / delta_analytic.abs();
    eprintln!(
        "cantilever_3d: tip_uz = {:.6e}, analytic = {:.6e}, rel_err = {:.4}",
        tip_uz, delta_analytic, rel_err
    );
    assert!(
        rel_err < 0.05,
        "3D cantilever tip deflection {:.6e} differs from analytic {:.6e} by {:.2}% (>5% bound)",
        tip_uz,
        delta_analytic,
        100.0 * rel_err
    );
}

// ─── Test 4: SVK → linear in the small-strain limit ─────────────────
//
// Pick a SMALL load on a tet beam, solve once with the linear
// elasticity stiffness (closed-form `K · u = f`) and once with the
// SVK Newton driver. The two solutions should agree to <1%.
//
// We use a coarser, shorter beam to keep the Newton solve cheap (5
// iters on each of ~10² Newton steps in a CG inner loop).

#[test]
fn svk_large_deformation_consistent_with_linear() {
    let len_x: Real = 4.0;
    let b: Real = 1.0;
    let h: Real = 1.0;
    let nx = 8usize;
    let ny = 2usize;
    let nz = 2usize;
    let mesh = FemMesh::tet_grid_3d(nx, ny, nz, len_x, b, h);
    let dim = 3usize;
    let space = FunctionSpace::P1Vector { dim };

    let youngs: Real = 1.0e7;
    let poisson: Real = 0.3;
    let (lam, mu) = lame(youngs, poisson);
    let k = assemble_elasticity(&mesh, dim, lam, mu);

    // Tiny traction so SVK is in its small-strain regime.
    let traction_z: Real = -1.0;

    // BCs — clamp left face.
    let mut bcs: Vec<BoundaryCondition> = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1e-9 {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0, 0.0],
            });
        }
    }

    // External load (traction on right face).
    let n = space.ndof(&mesh);
    let mut f_ext = vec![0.0; n];
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
                let on_x_l = (pa.x - len_x).abs() < 1e-9
                    && (pb.x - len_x).abs() < 1e-9
                    && (pc.x - len_x).abs() < 1e-9;
                if !on_x_l {
                    continue;
                }
                let area = 0.5 * (pb - pa).cross(pc - pa).length();
                let third = area / 3.0;
                for &v in face {
                    let v = v as usize;
                    f_ext[v * dim + 2] += third * traction_z;
                }
            }
        }
    }

    // Linear baseline.
    let u_lin = solve_steady_state_vector(&k, &f_ext, &bcs, &mesh.vertices, dim, 50_000, 1.0e-10);

    // SVK Newton-Raphson — tangent uses the linear stiffness (a valid
    // approximation in the small-strain limit; a proper full tangent
    // would also include geometric-stiffness terms but they're O(u²)
    // and vanish in the limit), residual uses the SVK internal force.
    let mesh_v = mesh.vertices.clone();
    let cells = mesh.cells.clone();
    let tangent = |_u: &[Real]| -> BsrMatrix { k.clone() };
    let residual = |u: &[Real]| -> Vec<Real> {
        let mut r = vec![0.0; n];
        for cell in &cells {
            if let FemCell::Tet(idx) = cell {
                // Gather local u.
                let mut u_local = [0.0; 12];
                for (i, &v) in idx.iter().enumerate() {
                    let base = (v as usize) * dim;
                    u_local[3 * i] = u[base];
                    u_local[3 * i + 1] = u[base + 1];
                    u_local[3 * i + 2] = u[base + 2];
                }
                let f_int =
                    compute_element_st_venant_kirchhoff_stress(cell, &mesh_v, &u_local, lam, mu);
                for (i, &v) in idx.iter().enumerate() {
                    let base = (v as usize) * dim;
                    r[base] += f_int[3 * i];
                    r[base + 1] += f_int[3 * i + 1];
                    r[base + 2] += f_int[3 * i + 2];
                }
            }
        }
        r
    };

    let u_svk = solve_nonlinear_static(
        tangent,
        residual,
        &f_ext,
        &bcs,
        &mesh.vertices,
        dim,
        10,
        1.0e-9,
    );

    // Compare tip deflection.
    let mut sum_lin = 0.0;
    let mut sum_svk = 0.0;
    let mut cnt = 0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        if (p.x - len_x).abs() < 1e-9 {
            sum_lin += u_lin[i * dim + 2];
            sum_svk += u_svk[i * dim + 2];
            cnt += 1;
        }
    }
    let tip_lin = sum_lin / cnt as Real;
    let tip_svk = sum_svk / cnt as Real;
    let rel_err = (tip_svk - tip_lin).abs() / tip_lin.abs().max(1e-30);
    eprintln!(
        "svk_vs_linear: tip_lin = {:.6e}, tip_svk = {:.6e}, rel_err = {:.4}",
        tip_lin, tip_svk, rel_err
    );
    assert!(
        rel_err < 0.01,
        "SVK tip deflection {:.6e} differs from linear {:.6e} by {:.2}% (>1% bound)",
        tip_svk,
        tip_lin,
        100.0 * rel_err
    );
}

// ─── Smoke: vector apply_bcs sanity check ──────────────────────────

#[test]
fn vector_apply_bcs_pins_all_components() {
    // 2-vertex, 1-cell mock. Use a 3D vector field.
    let mesh = FemMesh::tet_grid_3d(1, 1, 1, 1.0, 1.0, 1.0);
    let dim = 3usize;
    let (lam, mu) = lame(1.0, 0.3);
    let k = assemble_elasticity(&mesh, dim, lam, mu);
    let n = k.nrows();
    let mut k_red = k.clone();
    let mut f = vec![0.0; n];
    let bc = BoundaryCondition::DirichletVector {
        vertex: 0,
        values: vec![1.0, 2.0, 3.0],
    };
    apply_bcs(&mut k_red, &mut f, &[bc], &mesh.vertices, dim);
    // f at vertex 0 components 0,1,2 should equal the prescribed
    // values; the row has diag = 1, off-diag = 0.
    assert!((f[0] - 1.0).abs() < 1e-12);
    assert!((f[1] - 2.0).abs() < 1e-12);
    assert!((f[2] - 3.0).abs() < 1e-12);
}

#[test]
fn neo_hookean_tangent_matches_linear_elasticity_at_identity() {
    let mesh = FemMesh::tet_grid_3d(1, 1, 1, 1.0, 1.0, 1.0);
    let cell = mesh
        .cells
        .iter()
        .find(|c| matches!(c, FemCell::Tet(_)))
        .unwrap();
    let (lam, mu) = lame(10.0, 0.25);
    let u = vec![0.0; 12];
    let f = compute_element_neo_hookean_force(cell, &mesh.vertices, &u, lam, mu);
    let max_f = f.iter().map(|v| v.abs()).fold(0.0, Real::max);
    assert!(
        max_f < 1.0e-8,
        "identity Neo-Hookean force should vanish, max {max_f:.3e}"
    );

    let k_nh = compute_element_neo_hookean_tangent(cell, &mesh.vertices, &u, lam, mu);
    let k_lin = compute_element_elasticity_stiffness(cell, &mesh.vertices, lam, mu);
    let mut max_err: Real = 0.0;
    let mut max_ref: Real = 0.0;
    for i in 0..12 {
        for j in 0..12 {
            max_err = max_err.max((k_nh[i][j] - k_lin[i][j]).abs());
            max_ref = max_ref.max(k_lin[i][j].abs());
        }
    }
    assert!(
        max_err / max_ref.max(1.0e-30) < 1.0e-3,
        "Neo-Hookean identity tangent should match linear elasticity: rel {:.3e}",
        max_err / max_ref.max(1.0e-30)
    );
}

#[test]
fn mooney_rivlin_tangent_is_symmetric_and_identity_force_free() {
    let mesh = FemMesh::tet_grid_3d(1, 1, 1, 1.0, 1.0, 1.0);
    let cell = mesh
        .cells
        .iter()
        .find(|c| matches!(c, FemCell::Tet(_)))
        .unwrap();
    let u = vec![0.0; 12];
    let f = compute_element_mooney_rivlin_force(cell, &mesh.vertices, &u, 0.4, 0.1, 10.0);
    let max_f = f.iter().map(|v| v.abs()).fold(0.0, Real::max);
    assert!(
        max_f < 1.0e-8,
        "identity Mooney-Rivlin force should vanish, max {max_f:.3e}"
    );

    let k = compute_element_mooney_rivlin_tangent(cell, &mesh.vertices, &u, 0.4, 0.1, 10.0);
    let mut max_skew: Real = 0.0;
    let mut max_ref: Real = 0.0;
    for i in 0..12 {
        for j in 0..12 {
            max_skew = max_skew.max((k[i][j] - k[j][i]).abs());
            max_ref = max_ref.max(k[i][j].abs());
        }
    }
    assert!(
        max_skew / max_ref.max(1.0e-30) < 5.0e-3,
        "Mooney-Rivlin tangent should be symmetric: rel skew {:.3e}",
        max_skew / max_ref.max(1.0e-30)
    );
}
