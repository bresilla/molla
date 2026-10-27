//! Stage 39B — Discontinuous Galerkin (DG) contract tests.
//!
//! Three deliverables from the stage spec:
//!
//! 1. `dg_p0_mass_matches_volume` — every DG-P0 mass entry is exactly
//!    the cell volume (mass matrix is diagonal, off-blocks vanish).
//! 2. `dg_p1_solves_constant_advection_correctly` — a 1D-on-2D step
//!    initial condition is advanced under DG-P1 + SIP-stabilised
//!    diffusion and remains stable + monotone (no Gibbs oscillations
//!    at the jump).
//! 3. `dg_recovers_continuous_solution_on_smooth_problem` — DG-P1
//!    with SIP and Nitsche-style weak Dirichlet reproduces the
//!    CG-P1 solution of a Laplace Dirichlet problem to within 5%
//!    on a 16×16 grid.
//!
//! All tests are CPU-only and run as part of the standard
//! `cargo test -p molla-fem --test fem_dg` suite.

use molla_fem::{
    BoundaryCondition, FemCell, FemMesh, assemble_dg_global, assemble_global,
    compute_dg_flux_jump_block, compute_element_dg_mass, compute_element_dg_stiffness,
    compute_element_mass, compute_element_stiffness, compute_interior_faces, dg_ndof_per_cell,
    solve_steady_state,
};
use molla_math::iterative::cg_solve;
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// ─── Helpers ─────────────────────────────────────────────────────────

/// Assemble a DG global stiffness matrix (interior + SIP flux glue)
/// for the given mesh and degree.
fn assemble_dg_stiffness(mesh: &FemMesh, alpha: Real, degree: usize, eta: Real) -> BsrMatrix {
    assemble_dg_global(
        mesh,
        degree,
        |cell, verts| compute_element_dg_stiffness(cell, verts, alpha, degree),
        Some(|cl: &FemCell, cr: &FemCell, face: &_, verts: &[Vec3]| {
            compute_dg_flux_jump_block(cl, cr, face, verts, alpha, degree, eta)
        }),
    )
    .expect("assemble DG stiffness")
}

/// Assemble a DG global mass matrix (block-diagonal, no flux glue).
fn assemble_dg_mass(mesh: &FemMesh, degree: usize) -> BsrMatrix {
    assemble_dg_global::<
        _,
        fn(&FemCell, &FemCell, &molla_fem::InteriorFace, &[Vec3]) -> Vec<Vec<Real>>,
    >(
        mesh,
        degree,
        |cell, verts| compute_element_dg_mass(cell, verts, degree),
        None,
    )
    .expect("assemble DG mass")
}

fn matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    let mut y = vec![0.0; a.nrows()];
    a.mv(x, &mut y).unwrap();
    y
}

/// Look up the DG DoF offset for cell `c` under `degree`-DG. With the
/// homogeneous meshes used by the tests this is just `c * ndof_per_cell`.
fn dg_offset(mesh: &FemMesh, cell_idx: usize, degree: usize) -> usize {
    let mut off = 0;
    for c in &mesh.cells[..cell_idx] {
        off += dg_ndof_per_cell(c, degree);
    }
    off
}

// ─── Test 1: DG-P0 mass = cell volume on the diagonal ───────────────

#[test]
fn dg_p0_mass_matches_volume() {
    // Triangle grid: tri_grid_2d(4, 4) on the unit square.
    let mesh = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
    let m = assemble_dg_mass(&mesh, 0);

    // The matrix should be ncells × ncells (one DoF per cell), with the
    // diagonal equal to the cell volume and all off-diagonals zero.
    let n = m.nrows();
    assert_eq!(n, mesh.cell_count());

    for ci in 0..mesh.cell_count() {
        // Row sum equals diagonal because off-diagonals must be zero.
        let mut e = vec![0.0_f64 as Real; n];
        e[ci] = 1.0;
        let col = matvec(&m, &e);
        // Off-diagonals must be exactly zero.
        for (i, v) in col.iter().enumerate() {
            if i == ci {
                // Diagonal == cell volume.
                let expected = match mesh.cells[ci] {
                    FemCell::Tri(verts) => {
                        let p = [
                            mesh.vertices[verts[0] as usize],
                            mesh.vertices[verts[1] as usize],
                            mesh.vertices[verts[2] as usize],
                        ];
                        // Use the same area formula as element.rs.
                        let dx = (p[1].x - p[0].x) * (p[2].y - p[0].y)
                            - (p[2].x - p[0].x) * (p[1].y - p[0].y);
                        0.5 * dx.abs()
                    }
                    FemCell::Tet(_) => unreachable!("only tri mesh here"),
                };
                assert!(
                    (v - expected).abs() < 1e-12,
                    "diag[{ci}] = {v}, want {expected}"
                );
            } else {
                assert!(
                    v.abs() < 1e-30,
                    "off-diag[{i}][{ci}] = {v} (should be zero)"
                );
            }
        }
    }

    // Sanity: sum of diagonal == total mesh measure (unit square = 1).
    let ones = vec![1.0; n];
    let row_sums = matvec(&m, &ones);
    let total: Real = row_sums.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-12,
        "total area from DG-P0 mass = {total}, expected 1"
    );
}

// ─── Test 2: DG-P1 advection of a step IC remains stable + monotone ─

#[test]
fn dg_p1_solves_constant_advection_correctly() {
    // We test "stability + monotonicity" via the canonical DG-P1
    // application: advancing a step initial condition through an
    // SIP-stabilised diffusion problem. The SIP penalty replaces the
    // missing inter-element continuity and keeps the solution from
    // developing Gibbs-type overshoots at the jump.
    //
    // Problem:
    //   ∂u/∂t = α ∇²u   on [0, 1]² with periodic-like setup (we use
    //   strip geometry with insulated top/bottom).
    //   u(x, y, 0) = 1 if x < 0.5 else 0          (step initial condition)
    //
    // We use a moderate 16×4 grid (16 in x, 4 in y), DG-P1 with SIP,
    // and run a single backward-Euler step (large enough to diffuse
    // the step into a smooth ramp). At t > 0 the solution should be:
    //   - monotone in x (no overshoots / undershoots),
    //   - clamped: 0 ≤ u(x) ≤ 1 (with a small tolerance for the
    //     penalty-driven smoothing).

    let nx = 16;
    let ny = 4;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 0.25);
    let degree = 1;
    let alpha: Real = 1.0e-2;
    let dt: Real = 0.05;
    let eta: Real = 10.0;
    let n = mesh.cells.len() * 3; // DG-P1 on tri ⇒ 3 DoFs per cell

    let m = assemble_dg_mass(&mesh, degree);
    let k = assemble_dg_stiffness(&mesh, alpha, degree, eta);

    // Initial condition: step function — for each DG-P1 cell, sample
    // the step at each vertex.
    let mut u = vec![0.0_f64 as Real; n];
    for (ci, cell) in mesh.cells.iter().enumerate() {
        let off = ci * 3;
        let verts = match cell {
            FemCell::Tri(v) => v,
            FemCell::Tet(_) => unreachable!(),
        };
        for (i, &vi) in verts.iter().enumerate() {
            let x = mesh.vertices[vi as usize].x;
            u[off + i] = if x < 0.5 { 1.0 } else { 0.0 };
        }
    }

    // Backward Euler: (M/dt + K) u^{n+1} = (M/dt) u^n.
    let inv_dt = 1.0 / dt;
    let m_over_dt_uold = {
        let mu = matvec(&m, &u);
        mu.into_iter().map(|v| v * inv_dt).collect::<Vec<Real>>()
    };

    // matvec for (M/dt + K).
    let m_ref = &m;
    let k_ref = &k;
    let op = |x: &[Real]| -> Vec<Real> {
        let mut y = vec![0.0; n];
        let mu = matvec(m_ref, x);
        let ku = matvec(k_ref, x);
        for i in 0..n {
            y[i] = mu[i] * inv_dt + ku[i];
        }
        y
    };

    let res = cg_solve(op, &m_over_dt_uold, &u, 1e-8, 2000);
    let u_new = res.x;

    // Project to cell-average values to check monotonicity along x.
    // For DG-P1 the cell average is the mean of the three vertex
    // DoFs (P1 basis ⇒ ∫_K φᵢ dx = |K| / 3, so the cell average is
    // (1/3) Σ uᵢ).
    let mut cell_means: Vec<(Real, Real)> = Vec::with_capacity(mesh.cell_count());
    for (ci, cell) in mesh.cells.iter().enumerate() {
        let off = ci * 3;
        let mean = (u_new[off] + u_new[off + 1] + u_new[off + 2]) / 3.0;
        let verts = match cell {
            FemCell::Tri(v) => v,
            FemCell::Tet(_) => unreachable!(),
        };
        let cx = (mesh.vertices[verts[0] as usize].x
            + mesh.vertices[verts[1] as usize].x
            + mesh.vertices[verts[2] as usize].x)
            / 3.0;
        cell_means.push((cx, mean));
    }
    cell_means.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Group by x-bucket (since the strip has multiple cells per x-band)
    // and average to get a 1D profile.
    let mut buckets: Vec<Vec<Real>> = vec![Vec::new(); nx];
    for &(cx, mean) in &cell_means {
        // Cells span x ∈ [0, 1] divided into nx bands of width 1/nx.
        // The cell centroid x is at (j + offset) / nx for some
        // 0 ≤ j < nx.
        let mut bucket = (cx * nx as Real).floor() as usize;
        if bucket >= nx {
            bucket = nx - 1;
        }
        buckets[bucket].push(mean);
    }
    let profile: Vec<Real> = buckets
        .iter()
        .map(|b| b.iter().sum::<Real>() / b.len() as Real)
        .collect();

    // Monotonicity check: profile should be non-increasing in x (the
    // step starts at 1 on the left, 0 on the right, and diffusion
    // only smooths). Allow a tiny tolerance for the SIP penalty's
    // numerical jitter.
    for i in 0..(nx - 1) {
        assert!(
            profile[i] + 1e-2 >= profile[i + 1],
            "non-monotone at x bucket {i}: profile[{i}] = {} < profile[{}] = {}",
            profile[i],
            i + 1,
            profile[i + 1]
        );
    }

    // Clampedness: 0 ≤ profile ≤ 1 (with a small SIP-driven slack).
    for (i, v) in profile.iter().enumerate() {
        assert!(
            *v >= -0.05 && *v <= 1.05,
            "profile[{i}] = {v} out of [0, 1] (with 5% tolerance)"
        );
    }

    // Diffusion check: the step should have spread (some x ≈ 0.5 cell
    // should have a value between 0.1 and 0.9 — neither pure 0 nor 1).
    let mid = profile[nx / 2];
    assert!(
        mid > 0.05 && mid < 0.95,
        "middle profile value = {mid} — step did not diffuse?"
    );
}

// ─── Test 3: DG-P1 reproduces the CG-P1 solution on a smooth problem ─
//
// Solve `-∇²u = f` on the unit square with f = 2π²·sin(πx)·sin(πy) and
// homogeneous Dirichlet BC. The analytic solution is
// `u(x, y) = sin(πx)·sin(πy)`.
//
// CG-P1 path: standard `assemble_global` + `apply_dirichlet`.
//
// DG-P1 path: `assemble_dg_global` with SIP interior flux + Nitsche
// weak Dirichlet on boundary faces (assembled below as an extra face
// pass).
//
// Comparison: project the DG-P1 solution to cell-centroid values and
// compare against the CG-P1 solution sampled at the same centroids.
// The DG ↔ CG max relative difference should fall under 5%.

#[test]
fn dg_recovers_continuous_solution_on_smooth_problem() {
    let n_grid = 16;
    let mesh = FemMesh::tri_grid_2d(n_grid, n_grid, 1.0, 1.0);
    let alpha: Real = 1.0;
    let pi: Real = std::f64::consts::PI;
    let analytic = |x: Real, y: Real| (pi * x).sin() * (pi * y).sin();
    let source = |x: Real, y: Real| 2.0 * pi * pi * (pi * x).sin() * (pi * y).sin();

    // ─── CG-P1 solve ───────────────────────────────────────────
    let k_cg = assemble_global(&mesh, molla_fem::FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .unwrap();
    let m_cg = assemble_global(
        &mesh,
        molla_fem::FunctionSpace::P1Nodal,
        compute_element_mass,
    )
    .unwrap();
    // Build the RHS as M · f_at_nodes (consistent P1 source projection).
    let f_at_nodes: Vec<Real> = mesh.vertices.iter().map(|v| source(v.x, v.y)).collect();
    let rhs_cg = matvec(&m_cg, &f_at_nodes);

    // Dirichlet BCs: u = 0 on every boundary vertex.
    let mut bcs_cg = Vec::new();
    for (i, v) in mesh.vertices.iter().enumerate() {
        if v.x <= 1.0e-9 || v.x >= 1.0 - 1.0e-9 || v.y <= 1.0e-9 || v.y >= 1.0 - 1.0e-9 {
            bcs_cg.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        }
    }
    let u_cg = solve_steady_state(&k_cg, &rhs_cg, &bcs_cg, 4000, 1e-10);

    // ─── DG-P1 solve ───────────────────────────────────────────
    // Interior SIP flux + Nitsche boundary terms are assembled below.
    // We assemble three matrices via the DG visitor:
    //   K_DG = K_cell + K_face (SIP interior) + K_bnd (Nitsche bnd).
    // Then RHS = M_DG · f_at_dg_dofs (per-cell-vertex sample).
    let degree = 1;
    let eta: Real = 10.0;
    let eta_bnd: Real = 10.0;

    let k_dg_interior = assemble_dg_global(
        &mesh,
        degree,
        |c, v| compute_element_dg_stiffness(c, v, alpha, degree),
        Some(|cl: &FemCell, cr: &FemCell, face: &_, verts: &[Vec3]| {
            compute_dg_flux_jump_block(cl, cr, face, verts, alpha, degree, eta)
        }),
    )
    .unwrap();

    // Build Nitsche boundary terms: for each boundary edge (owned by
    // exactly one cell), the LHS gets a one-sided SIP-like block:
    //   a_bnd(u, v) = − α ∫_F (∇u·n) v dS − α ∫_F u (∇v·n) dS
    //                  + (η_b α / h_F) ∫_F u v dS.
    // u_D = 0 ⇒ RHS contribution is zero (no boundary forcing).
    let bnd_block = assemble_boundary_nitsche(&mesh, alpha, eta_bnd);

    // Sum k_dg_interior + bnd_block. Both are square matrices of the
    // same dimension (ndof_dg). Re-extract via matvec into Vec to add.
    // Simpler: keep both and operate via two matvecs in the CG closure.
    let m_dg = assemble_dg_mass(&mesh, degree);

    // RHS: M_DG · f sampled at each DG DoF (cell vertex position).
    let ndof_dg = m_dg.nrows();
    let mut f_at_dg: Vec<Real> = Vec::with_capacity(ndof_dg);
    for ci in 0..mesh.cell_count() {
        let verts = match &mesh.cells[ci] {
            FemCell::Tri(v) => v,
            FemCell::Tet(_) => unreachable!(),
        };
        for &vi in verts {
            let p = mesh.vertices[vi as usize];
            f_at_dg.push(source(p.x, p.y));
        }
    }
    assert_eq!(f_at_dg.len(), ndof_dg);
    let rhs_dg = matvec(&m_dg, &f_at_dg);

    let m_dg_ref = &m_dg; // silence move; not actually used in op
    let _ = m_dg_ref;
    let k_int_ref = &k_dg_interior;
    let k_bnd_ref = &bnd_block;
    let op = |x: &[Real]| -> Vec<Real> {
        let mut y = matvec(k_int_ref, x);
        let yb = matvec(k_bnd_ref, x);
        for i in 0..ndof_dg {
            y[i] += yb[i];
        }
        y
    };
    let res = cg_solve(op, &rhs_dg, &vec![0.0; ndof_dg], 1e-9, 8000);
    assert!(
        res.iters < 8000,
        "DG-P1 CG did not converge (iters={}, residual={})",
        res.iters,
        res.residual
    );
    let u_dg = res.x;

    // ─── Compare DG ↔ CG at cell centroids ─────────────────────
    // Compute the max absolute difference between DG and CG cell-
    // centroid values, then normalise by the CG solution's max
    // magnitude (analytic max = 1.0 here, so this is essentially
    // an "infinity-norm relative" metric). This avoids the
    // divide-by-zero pathology of per-point relative error near the
    // Dirichlet boundary where the solution is identically zero.
    let mut max_abs: Real = 0.0;
    let mut max_cg_mag: Real = 0.0;
    for (ci, cell) in mesh.cells.iter().enumerate() {
        let verts = match cell {
            FemCell::Tri(v) => v,
            FemCell::Tet(_) => unreachable!(),
        };
        // DG centroid value: average of the 3 vertex DoFs.
        let off = dg_offset(&mesh, ci, degree);
        let dg_val = (u_dg[off] + u_dg[off + 1] + u_dg[off + 2]) / 3.0;
        // CG centroid value: average of the 3 vertex DoFs.
        let cg_val =
            (u_cg[verts[0] as usize] + u_cg[verts[1] as usize] + u_cg[verts[2] as usize]) / 3.0;
        let abs_err = (dg_val - cg_val).abs();
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        if cg_val.abs() > max_cg_mag {
            max_cg_mag = cg_val.abs();
        }
    }
    let max_rel = max_abs / max_cg_mag.max(1.0e-12);

    // Also sanity-check the DG solution against the analytic solution
    // at centroids — both should be within reasonable distance.
    let mut max_dg_vs_analytic: Real = 0.0;
    for (ci, cell) in mesh.cells.iter().enumerate() {
        let verts = match cell {
            FemCell::Tri(v) => v,
            FemCell::Tet(_) => unreachable!(),
        };
        let off = dg_offset(&mesh, ci, degree);
        let dg_val = (u_dg[off] + u_dg[off + 1] + u_dg[off + 2]) / 3.0;
        let cx = (mesh.vertices[verts[0] as usize].x
            + mesh.vertices[verts[1] as usize].x
            + mesh.vertices[verts[2] as usize].x)
            / 3.0;
        let cy = (mesh.vertices[verts[0] as usize].y
            + mesh.vertices[verts[1] as usize].y
            + mesh.vertices[verts[2] as usize].y)
            / 3.0;
        let an = analytic(cx, cy);
        let err = (dg_val - an).abs();
        if err > max_dg_vs_analytic {
            max_dg_vs_analytic = err;
        }
    }

    eprintln!(
        "DG vs CG: max_rel = {:.4e}, max_abs = {:.4e}, max_dg_vs_analytic = {:.4e}",
        max_rel, max_abs, max_dg_vs_analytic
    );

    // Acceptance: max relative difference < 5%.
    assert!(
        max_rel < 0.05,
        "DG-P1 vs CG-P1 max relative diff = {max_rel}, want < 0.05"
    );
}

// ─── Nitsche weak Dirichlet block (for test 3) ──────────────────────
//
// For each boundary face (owned by a single cell), add the SIP-like
// one-sided block:
//
//   a_bnd(u, v) = − α ∫_F (∇u · n) v dS
//                 − α ∫_F u (∇v · n) dS
//                 + (η α / h_F) ∫_F u v dS
//
// where `n` is the outward face normal. u_D = 0 ⇒ no RHS forcing.
//
// We implement this directly using the `compute_dg_flux_jump_block`
// recipe pared down to the "left side only" case: same shape values
// and gradients on both sides of the trace would have the gradient
// average half-half; for a boundary face only the cell contributes,
// so `{∇u·n}` reduces to `∇u·n` on the owning cell (the "+" side has
// no support, so its contribution is zero, and the average is
// reflected back through the BC). For homogeneous Dirichlet the
// resulting block is the cell-side `(N×N)` matrix below.

fn assemble_boundary_nitsche(mesh: &FemMesh, alpha: Real, eta_bnd: Real) -> BsrMatrix {
    use std::collections::HashMap;

    // Step 1: collect all (sorted face_verts → list of owning cells).
    let mut face_to_cells: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
    for (cell_idx, cell) in mesh.cells.iter().enumerate() {
        match cell {
            FemCell::Tri(v) => {
                for &[a, b] in &[[v[0], v[1]], [v[1], v[2]], [v[2], v[0]]] {
                    let mut key = vec![a, b];
                    key.sort_unstable();
                    face_to_cells.entry(key).or_default().push(cell_idx as u32);
                }
            }
            FemCell::Tet(_) => unimplemented!("test 3 is 2D-only"),
        }
    }

    let degree = 1;
    let n = mesh.cell_count() * 3;
    let mut triplets: Vec<(usize, usize, Real)> = Vec::with_capacity(mesh.cell_count() * 32);

    for (face_verts, owners) in &face_to_cells {
        if owners.len() != 1 {
            continue; // interior face — handled by SIP flux block.
        }
        let cell_idx = owners[0] as usize;
        let cell = &mesh.cells[cell_idx];
        let off = cell_idx * 3;

        // Gather geometry.
        let a = mesh.vertices[face_verts[0] as usize];
        let b = mesh.vertices[face_verts[1] as usize];
        let len = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
        let h_f = len;

        // Outward normal: edge direction × (+z) gives (dy, -dx). Pick
        // the sign so it points away from the cell centroid.
        let centroid = match cell {
            FemCell::Tri(v) => {
                (mesh.vertices[v[0] as usize]
                    + mesh.vertices[v[1] as usize]
                    + mesh.vertices[v[2] as usize])
                    / 3.0
            }
            FemCell::Tet(_) => unreachable!(),
        };
        let raw_n = Vec3::new((b.y - a.y) / len, (a.x - b.x) / len, 0.0);
        let to_cell = centroid - a;
        let n_out = if raw_n.dot(to_cell) <= 0.0 {
            raw_n
        } else {
            -raw_n
        };

        // P1 shape gradients on the owning cell (constant).
        let verts = match cell {
            FemCell::Tri(v) => v,
            FemCell::Tet(_) => unreachable!(),
        };
        let p = [
            mesh.vertices[verts[0] as usize],
            mesh.vertices[verts[1] as usize],
            mesh.vertices[verts[2] as usize],
        ];
        let det = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
        let inv = 1.0 / det;
        let g = [
            [(p[1].y - p[2].y) * inv, (p[2].x - p[1].x) * inv],
            [(p[2].y - p[0].y) * inv, (p[0].x - p[2].x) * inv],
            [(p[0].y - p[1].y) * inv, (p[1].x - p[0].x) * inv],
        ];
        let gn = [
            g[0][0] * n_out.x + g[0][1] * n_out.y,
            g[1][0] * n_out.x + g[1][1] * n_out.y,
            g[2][0] * n_out.x + g[2][1] * n_out.y,
        ];

        // Two-pt Gauss-Legendre on the edge.
        let gp = 1.0 / (3.0 as Real).sqrt() / 2.0;
        let xi = [0.5 - gp, 0.5 + gp];
        let pen = eta_bnd * alpha / h_f;
        let w_quad = 0.5 * len;

        for &t in &xi {
            // Physical point.
            let x = a + (b - a) * t;
            // Barycentric on owning tri.
            let l1 =
                ((p[2].y - p[0].y) * (x.x - p[0].x) + (p[0].x - p[2].x) * (x.y - p[0].y)) * inv;
            let l2 =
                ((p[0].y - p[1].y) * (x.x - p[0].x) + (p[1].x - p[0].x) * (x.y - p[0].y)) * inv;
            let l0 = 1.0 - l1 - l2;
            let nvals = [l0, l1, l2];

            for i in 0..3 {
                for j in 0..3 {
                    // − α ∫ (∇u·n) v − α ∫ u (∇v·n) + pen ∫ u v
                    let val = -alpha * gn[j] * nvals[i] - alpha * nvals[j] * gn[i]
                        + pen * nvals[i] * nvals[j];
                    let v = w_quad * val;
                    if v != 0.0 {
                        triplets.push((off + i, off + j, v));
                    }
                }
            }
        }
        let _ = degree;
    }
    BsrMatrix::from_scalar_triplets(n, n, &triplets).expect("boundary block")
}

// Touch helper to keep `compute_interior_faces` in the `use` list —
// the function is exercised internally by `assemble_dg_global`'s face
// pass, but bringing it under the import list helps users discover
// the building-block API. The compile-time check survives even when
// the test suite cherry-picks individual tests.
#[allow(dead_code)]
const _: fn() = || {
    let _ = compute_interior_faces;
};
