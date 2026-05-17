//! Stage 43 — DG higher-degree (P2) + numerical-flux library + slope
//! limiter contract tests.
//!
//! Three deliverables:
//!
//! 1. `dg_p2_converges_at_cubic_rate` — manufactured solution
//!    `u(x, y) = sin(πx) · sin(πy)` for the Poisson problem
//!    `−∇²u = 2π² sin(πx) sin(πy)` on the unit square, refined 8×8 →
//!    16×16 → 32×32. Log-log fit of the L² error vs `h` should
//!    yield a slope in `[2.5, 3.5]` (theoretical optimum for P2 is 3).
//!
//! 2. `dg_p1_upwind_flux_advects_correctly` — 2D linear advection of a
//!    Gaussian pulse along the `+x` axis on a strip. Inflow weakly
//!    set to zero, outflow handled implicitly by the upwind flux.
//!    Single revolution (the pulse travels `T · |wind|` units) — peak
//!    displacement matches analytic to ≤ 5 %.
//!
//! 3. `dg_p1_burgers_with_limiter_stays_monotone` — 1D Burgers' shock
//!    formation on a strip. The smooth initial condition steepens
//!    into a shock; with the Barth-Jespersen limiter ON, no overshoot
//!    > 1 % of the smooth peak.

use molla_fem::{
    FemCell, FemMesh, InteriorFace, NumericalFlux, assemble_dg_global, build_cell_neighbours,
    compute_dg_flux_jump_block, compute_dg_numerical_flux_block, compute_element_dg_mass,
    compute_element_dg_stiffness, compute_euler_numerical_flux_hll_2d,
    compute_euler_numerical_flux_rusanov_2d, dg_ndof_per_cell, dg_nodal_positions,
    euler_flux_dot_n_2d, euler_pressure_2d, limit_solution,
};
use molla_math::iterative::cg_solve;
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// ─── Helpers ─────────────────────────────────────────────────────────

fn matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    let mut y = vec![0.0_f64 as Real; a.nrows()];
    a.mv(x, &mut y).unwrap();
    y
}

#[test]
fn euler_fluxes_match_physical_flux_for_equal_states() {
    let gamma = 1.4;
    let rho: Real = 1.2;
    let ux: Real = 2.0;
    let uy: Real = -0.5;
    let p: Real = 0.9;
    let e = p / (gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy);
    let state = [rho, rho * ux, rho * uy, e];
    let n = Vec3::new(0.6, 0.8, 0.0);

    let physical = euler_flux_dot_n_2d(state, n, gamma);
    let rusanov = compute_euler_numerical_flux_rusanov_2d(state, state, n, gamma);
    let hll = compute_euler_numerical_flux_hll_2d(state, state, n, gamma);

    assert!((euler_pressure_2d(state, gamma) - p).abs() < 1.0e-12);
    for k in 0..4 {
        assert!(
            (rusanov[k] - physical[k]).abs() < 1.0e-12,
            "Rusanov component {k}: {} vs {}",
            rusanov[k],
            physical[k]
        );
        assert!(
            (hll[k] - physical[k]).abs() < 1.0e-12,
            "HLL component {k}: {} vs {}",
            hll[k],
            physical[k]
        );
    }
}

/// Per-cell DG DoF offset (running sum of `dg_ndof_per_cell`).
fn dg_offsets(mesh: &FemMesh, degree: usize) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(mesh.cells.len() + 1);
    let mut running = 0usize;
    for c in &mesh.cells {
        offsets.push(running);
        running += dg_ndof_per_cell(c, degree);
    }
    offsets.push(running);
    offsets
}

/// Assemble the per-cell DG mass matrix (block-diagonal, no flux glue).
fn assemble_dg_mass(mesh: &FemMesh, degree: usize) -> BsrMatrix {
    assemble_dg_global::<_, fn(&FemCell, &FemCell, &InteriorFace, &[Vec3]) -> Vec<Vec<Real>>>(
        mesh,
        degree,
        |cell, verts| compute_element_dg_mass(cell, verts, degree),
        None,
    )
    .expect("assemble DG mass")
}

/// Project a scalar field onto the DG-P2 nodal basis by sampling at
/// each per-cell Lagrange node (vertices + edge midpoints).
fn project_scalar_dg(mesh: &FemMesh, degree: usize, f: impl Fn(Real, Real) -> Real) -> Vec<Real> {
    let offsets = dg_offsets(mesh, degree);
    let n = *offsets.last().unwrap();
    let mut u = vec![0.0_f64 as Real; n];
    for (ci, cell) in mesh.cells.iter().enumerate() {
        let nodes = dg_nodal_positions(cell, &mesh.vertices, degree);
        let off = offsets[ci];
        for (i, p) in nodes.iter().enumerate() {
            u[off + i] = f(p.x, p.y);
        }
    }
    u
}

/// L² error of a DG scalar field against analytic `truth`, evaluated
/// on a Simpson-style sample of points per cell (cheap: midpoint
/// quadrature for degree-1 fields, 6-pt Dunavant for higher accuracy).
fn l2_error_dg(
    mesh: &FemMesh,
    u_dg: &[Real],
    degree: usize,
    truth: impl Fn(Real, Real) -> Real,
) -> Real {
    use molla_fem::{TRI_EDGES, p2_tri_shape_functions};
    let offsets = dg_offsets(mesh, degree);
    let mut sq_err = 0.0_f64 as Real;
    // 6-pt Dunavant (degree-4 exact on tri).
    let a1 = 0.816_847_572_980_459;
    let b1 = 0.091_576_213_509_771;
    let w1 = 0.109_951_743_655_322;
    let a2 = 0.108_103_018_168_070;
    let b2 = 0.445_948_490_915_965;
    let w2 = 0.223_381_589_678_011;
    let baries: [[Real; 3]; 6] = [
        [a1, b1, b1],
        [b1, a1, b1],
        [b1, b1, a1],
        [a2, b2, b2],
        [b2, a2, b2],
        [b2, b2, a2],
    ];
    let ws: [Real; 6] = [w1, w1, w1, w2, w2, w2];
    for (ci, cell) in mesh.cells.iter().enumerate() {
        let v = match cell {
            FemCell::Tri(v) => v,
            FemCell::Tet(_) => unreachable!(),
        };
        let p = [
            mesh.vertices[v[0] as usize],
            mesh.vertices[v[1] as usize],
            mesh.vertices[v[2] as usize],
        ];
        let area = 0.5
            * ((p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y)).abs();
        let off = offsets[ci];
        let n = offsets[ci + 1] - off;
        for (bary, &w) in baries.iter().zip(ws.iter()) {
            let xq = p[0] * bary[0] + p[1] * bary[1] + p[2] * bary[2];
            // Evaluate u_dg at this point — reconstruct via Lagrange basis.
            let u_h: Real = match degree {
                1 => u_dg[off] * bary[0] + u_dg[off + 1] * bary[1] + u_dg[off + 2] * bary[2],
                2 => {
                    let nv = p2_tri_shape_functions(*bary);
                    let mut s = 0.0_f64 as Real;
                    for i in 0..6 {
                        s += nv[i] * u_dg[off + i];
                    }
                    s
                }
                _ => unimplemented!(),
            };
            let _ = (TRI_EDGES, n); // keep import alive
            let truth_val = truth(xq.x, xq.y);
            sq_err += w * area * (u_h - truth_val).powi(2);
        }
    }
    sq_err.sqrt()
}

// ─── Test 1: P2 cubic-rate convergence ───────────────────────────────

#[test]
fn dg_p2_converges_at_cubic_rate() {
    // Solve −∇²u = 2π² sin(πx) sin(πy) on [0, 1]² with u = 0 on ∂Ω.
    // Analytic: u = sin(πx) sin(πy). DG-P2 + SIP + Nitsche weak BC.
    let pi = std::f64::consts::PI;
    let analytic = |x: Real, y: Real| (pi * x).sin() * (pi * y).sin();
    let source = |x: Real, y: Real| 2.0 * pi * pi * (pi * x).sin() * (pi * y).sin();
    let degree = 2usize;
    let alpha: Real = 1.0;
    let eta: Real = 50.0; // P2 needs larger penalty than P1.

    let grids = [8usize, 16, 32];
    let mut hs = Vec::new();
    let mut errors = Vec::new();
    for &n_grid in &grids {
        let mesh = FemMesh::tri_grid_2d(n_grid, n_grid, 1.0, 1.0);
        let h: Real = 1.0 / n_grid as Real;
        let (k, _bnd) = assemble_dg_p2_poisson(&mesh, alpha, degree, eta);
        let m = assemble_dg_mass(&mesh, degree);
        // RHS: M_DG · f sampled at DG nodes.
        let f_at_nodes = project_scalar_dg(&mesh, degree, source);
        let rhs = matvec(&m, &f_at_nodes);
        // Solve K u = rhs (with Nitsche BC already baked in).
        let n = m.nrows();
        let k_ref = &k;
        let op = |x: &[Real]| -> Vec<Real> { matvec(k_ref, x) };
        let res = cg_solve(op, &rhs, &vec![0.0_f64 as Real; n], 1.0e-12, 20_000);
        assert!(
            res.iters < 20_000,
            "P2 CG did not converge for n={n_grid} (iters={}, residual={})",
            res.iters,
            res.residual
        );
        let err = l2_error_dg(&mesh, &res.x, degree, analytic);
        eprintln!("P2 n={n_grid}, h={h:.4}, L2 err = {err:.4e}");
        hs.push(h);
        errors.push(err);
    }

    // Fit log-log slope between consecutive levels.
    let mut rates = Vec::new();
    for i in 0..(hs.len() - 1) {
        let r = (errors[i] / errors[i + 1]).ln() / (hs[i] / hs[i + 1]).ln();
        rates.push(r);
        eprintln!("P2 rate {} -> {}: {:.3}", hs[i], hs[i + 1], r);
    }
    // Average rate should sit in [2.5, 3.5].
    let avg_rate: Real = rates.iter().sum::<Real>() / rates.len() as Real;
    eprintln!("P2 average convergence rate = {avg_rate:.3}");
    assert!(
        avg_rate >= 2.5 && avg_rate <= 3.5,
        "P2 average convergence rate = {avg_rate}, want in [2.5, 3.5]"
    );
}

/// Assemble the P2 DG Poisson system with SIP interior flux + Nitsche
/// weak Dirichlet boundary conditions (homogeneous, u_D = 0). Returns
/// the global stiffness matrix and (unused) boundary block — kept as
/// a tuple so we can extend with non-homogeneous BCs later.
fn assemble_dg_p2_poisson(
    mesh: &FemMesh,
    alpha: Real,
    degree: usize,
    eta: Real,
) -> (BsrMatrix, ()) {
    // SIP interior block.
    let k_interior = assemble_dg_global(
        mesh,
        degree,
        |c, v| compute_element_dg_stiffness(c, v, alpha, degree),
        Some(|cl: &FemCell, cr: &FemCell, face: &_, verts: &[Vec3]| {
            compute_dg_flux_jump_block(cl, cr, face, verts, alpha, degree, eta)
        }),
    )
    .expect("DG P2 interior");

    // Nitsche boundary block — add to the same global matrix via
    // triplet form. We rebuild the triplets list from a matvec round-
    // trip to avoid reaching into BsrMatrix internals.
    let nbnd = assemble_boundary_nitsche_p2(mesh, alpha, degree, eta);
    let combined = add_bsr(&k_interior, &nbnd);
    (combined, ())
}

/// Build the sum of two BsrMatrix as a new BsrMatrix. Both must have
/// the same shape.
fn add_bsr(a: &BsrMatrix, b: &BsrMatrix) -> BsrMatrix {
    let n = a.nrows();
    assert_eq!(b.nrows(), n);
    // Extract via column scans (build identity columns and stash the
    // resulting `A · e_j`).
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for j in 0..n {
        let mut e = vec![0.0_f64 as Real; n];
        e[j] = 1.0;
        let ya = matvec(a, &e);
        let yb = matvec(b, &e);
        for i in 0..n {
            let v = ya[i] + yb[i];
            if v != 0.0 {
                triplets.push((i, j, v));
            }
        }
    }
    BsrMatrix::from_scalar_triplets(n, n, &triplets).expect("add_bsr")
}

fn assemble_boundary_nitsche_p2(
    mesh: &FemMesh,
    alpha: Real,
    degree: usize,
    eta_bnd: Real,
) -> BsrMatrix {
    use molla_fem::{p2_tri_shape_functions, p2_tri_shape_gradients};
    use std::collections::HashMap;

    let offsets = dg_offsets(mesh, degree);
    let n_total = *offsets.last().unwrap();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();

    // Identify boundary edges (owned by exactly one cell).
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
            FemCell::Tet(_) => unimplemented!("P2 boundary Nitsche is 2D-only here"),
        }
    }

    // 3-pt Gauss-Legendre on the unit edge (degree-5 exact, sufficient
    // for P2 × P2 quartic face integrands).
    let g = 0.5 * (3.0_f64 / 5.0_f64).sqrt();
    let xis = [0.5 - g, 0.5, 0.5 + g];
    let ws = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0];

    for (face_verts, owners) in &face_to_cells {
        if owners.len() != 1 {
            continue;
        }
        let cell_idx = owners[0] as usize;
        let cell = &mesh.cells[cell_idx];
        let off = offsets[cell_idx];
        let nl = offsets[cell_idx + 1] - off;

        let a = mesh.vertices[face_verts[0] as usize];
        let b = mesh.vertices[face_verts[1] as usize];
        let len = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
        let h_f = len;
        let pen = eta_bnd * alpha / h_f;

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
        // Outward normal: edge direction ⟂, sign picked to point away
        // from centroid.
        let raw_n = Vec3::new((b.y - a.y) / len, (a.x - b.x) / len, 0.0);
        let centroid = (p[0] + p[1] + p[2]) / 3.0;
        let to_cell = centroid - a;
        let n_out = if raw_n.dot(to_cell) <= 0.0 {
            raw_n
        } else {
            -raw_n
        };

        for (&xi, &w) in xis.iter().zip(ws.iter()) {
            let x = a + (b - a) * xi;
            let w_quad = w * len;
            // Barycentric of x in the owning triangle.
            let l1 =
                ((p[2].y - p[0].y) * (x.x - p[0].x) + (p[0].x - p[2].x) * (x.y - p[0].y)) * inv;
            let l2 =
                ((p[0].y - p[1].y) * (x.x - p[0].x) + (p[1].x - p[0].x) * (x.y - p[0].y)) * inv;
            let l0 = 1.0 - l1 - l2;
            let bary = [l0, l1, l2];
            let nvals: Vec<Real> = if degree == 1 {
                bary.to_vec()
            } else {
                p2_tri_shape_functions(bary).to_vec()
            };
            let grads: Vec<Vec3> = if degree == 1 {
                // Constant P1 gradients in the owning tri.
                let g0 = Vec3::new((p[1].y - p[2].y) * inv, (p[2].x - p[1].x) * inv, 0.0);
                let g1 = Vec3::new((p[2].y - p[0].y) * inv, (p[0].x - p[2].x) * inv, 0.0);
                let g2 = Vec3::new((p[0].y - p[1].y) * inv, (p[1].x - p[0].x) * inv, 0.0);
                vec![g0, g1, g2]
            } else {
                p2_tri_shape_gradients(bary, &p).to_vec()
            };
            assert_eq!(nvals.len(), nl);
            assert_eq!(grads.len(), nl);
            let gn: Vec<Real> = grads.iter().map(|g| g.dot(n_out)).collect();

            for i in 0..nl {
                for j in 0..nl {
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
    }
    BsrMatrix::from_scalar_triplets(n_total, n_total, &triplets).expect("Nitsche P2 boundary")
}

// ─── Test 2: DG-P1 upwind flux advects a Gaussian correctly ─────────

#[test]
fn dg_p1_upwind_flux_advects_correctly() {
    // 2D linear advection on a strip [0, L] × [0, W] with wind = (1, 0).
    // Initial: Gaussian pulse centred at x0 = 1.0, σ = 0.15.
    // After time T = L − 2 · x0 — wait, simpler: T = 2, pulse should
    // translate by 2 → centre at x = 3.
    //
    // Inflow boundary (x=0): weak Dirichlet u = 0 (zero inflow).
    // Outflow boundary (x=L): outflow handled implicitly by the upwind
    //                         flux on boundary faces (we don't add any
    //                         boundary block — upwind with no neighbour
    //                         is a no-op, which corresponds to "use
    //                         u_L = interior").
    let l: Real = 4.0;
    let w: Real = 0.5;
    let nx = 80;
    let ny = 4;
    let wind = Vec3::new(1.0, 0.0, 0.0);
    let degree = 1usize;
    let x0: Real = 1.0;
    let sigma: Real = 0.15;
    let t_final: Real = 2.0;
    let dt: Real = 1.0e-3;
    let nsteps = (t_final / dt).round() as usize;

    let mesh = FemMesh::tri_grid_2d(nx, ny, l, w);

    // Project initial Gaussian onto DG-P1 vertex DoFs.
    let mut u = project_scalar_dg(&mesh, degree, |x, _y| {
        let r = (x - x0) / sigma;
        (-0.5 * r * r).exp()
    });

    // Assemble per-cell mass (block-diagonal). For DG-P1 we precompute
    // each 3×3 mass and invert it cheaply.
    let inv_mass_blocks: Vec<[[Real; 3]; 3]> = mesh
        .cells
        .iter()
        .map(|cell| {
            let m = compute_element_dg_mass(cell, &mesh.vertices, degree);
            assert_eq!(m.len(), 3);
            invert_3x3(&[
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ])
        })
        .collect();

    // Cache: per-cell element-advection block A_el[i][j] = −∫ (b·∇N_i) N_j
    // (the IBP volumetric contribution to the residual matrix). For
    // P1 on tris, b·∇N_i is constant, ∫ N_j = |K|/3.
    //
    // Sign rationale: weak form gives `M u_t = ∫(bu)·∇v − ∮Ĥv`, i.e.
    // `M u_t + A u = 0` with `A_el = −∫(bN_j)·∇N_i u_j` ⇒ matrix entry
    // `A_el[i][j] = −(b·∇N_i) · (|K|/3)`.
    let element_blocks: Vec<[[Real; 3]; 3]> = mesh
        .cells
        .iter()
        .map(|cell| {
            let v = match cell {
                FemCell::Tri(v) => v,
                FemCell::Tet(_) => unreachable!(),
            };
            let p = [
                mesh.vertices[v[0] as usize],
                mesh.vertices[v[1] as usize],
                mesh.vertices[v[2] as usize],
            ];
            let det = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
            let area = 0.5 * det.abs();
            let inv = 1.0 / det;
            let grads = [
                Vec3::new((p[1].y - p[2].y) * inv, (p[2].x - p[1].x) * inv, 0.0),
                Vec3::new((p[2].y - p[0].y) * inv, (p[0].x - p[2].x) * inv, 0.0),
                Vec3::new((p[0].y - p[1].y) * inv, (p[1].x - p[0].x) * inv, 0.0),
            ];
            // bdgni = wind · ∇N_i, constant per i.
            let bdgn = [wind.dot(grads[0]), wind.dot(grads[1]), wind.dot(grads[2])];
            let third = area / 3.0;
            // A_el[i][j] = − bdgn[i] · third.
            let mut a = [[0.0_f64 as Real; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    a[i][j] = -bdgn[i] * third;
                }
            }
            a
        })
        .collect();

    // Face flux: assemble once.
    let k_flux = assemble_dg_global(
        &mesh,
        degree,
        |_c, _v| vec![vec![0.0_f64 as Real; 3]; 3],
        Some(|cl: &FemCell, cr: &FemCell, face: &_, verts: &[Vec3]| {
            compute_dg_numerical_flux_block(
                cl,
                cr,
                face,
                verts,
                NumericalFlux::Upwind { wind },
                degree,
            )
        }),
    )
    .expect("DG P1 advection face flux");

    let offsets = dg_offsets(&mesh, degree);

    // Forward Euler: u^{n+1} = u^n − dt M⁻¹ (A_el u + A_face u).
    for _step in 0..nsteps {
        // Compute A u: face contribution is matvec; element is per-cell.
        let mut au = matvec(&k_flux, &u);
        for ci in 0..mesh.cells.len() {
            let off = offsets[ci];
            let a = &element_blocks[ci];
            // Local contribution.
            for i in 0..3 {
                let mut s = 0.0_f64 as Real;
                for j in 0..3 {
                    s += a[i][j] * u[off + j];
                }
                au[off + i] += s;
            }
        }
        // Inflow boundary (x=0): apply weak Dirichlet u = 0 by adding
        // a one-sided upwind flux contribution on boundary edges where
        // wind · n_outward < 0 (inflow). At u_D = 0 this just adds a
        // damping term to the LHS: we apply it implicitly by zeroing
        // out the inflow vertices each step (Dirichlet via simple
        // overwrite — first-order accurate, fine for this test).
        //
        // Actually, the cleaner approach: for cells with vertices at
        // x = 0, leave u alone; we let the upwind flux do its job on
        // the inflow boundary by treating the missing exterior cell
        // as u = 0 (which is what "no neighbour" semantics on boundary
        // faces would give if we ran a Dirichlet weak BC).
        //
        // For this test we instead just zero out the DG DoFs at the
        // inflow edge directly — pulse hasn't reached the outflow,
        // and the boundary is far from the pulse for t ≤ T.

        // u^{n+1} = u^n − dt M⁻¹ Au.
        for ci in 0..mesh.cells.len() {
            let off = offsets[ci];
            let inv_m = &inv_mass_blocks[ci];
            // Compute Δu = −dt · inv_m · (Au[off..off+3]).
            let au_local = [au[off], au[off + 1], au[off + 2]];
            let mut du = [0.0_f64 as Real; 3];
            for i in 0..3 {
                for j in 0..3 {
                    du[i] += inv_m[i][j] * au_local[j];
                }
            }
            u[off] -= dt * du[0];
            u[off + 1] -= dt * du[1];
            u[off + 2] -= dt * du[2];
        }

        // Pin inflow DoFs (x = 0) to zero to enforce homogeneous inflow.
        for (ci, cell) in mesh.cells.iter().enumerate() {
            let off = offsets[ci];
            let v = match cell {
                FemCell::Tri(v) => v,
                FemCell::Tet(_) => unreachable!(),
            };
            for (i, &vi) in v.iter().enumerate() {
                if mesh.vertices[vi as usize].x < 1.0e-9 {
                    u[off + i] = 0.0;
                }
            }
        }
    }

    // Locate peak of the DG solution along the strip — accumulate per
    // x-bin the maximum nodal value.
    let mut bins: Vec<(Real, Real)> = vec![(0.0, -1.0); nx];
    for (ci, cell) in mesh.cells.iter().enumerate() {
        let off = offsets[ci];
        let v = match cell {
            FemCell::Tri(v) => v,
            FemCell::Tet(_) => unreachable!(),
        };
        for (i, &vi) in v.iter().enumerate() {
            let x = mesh.vertices[vi as usize].x;
            let val = u[off + i];
            let mut bx = (x / l * nx as Real).floor() as usize;
            if bx >= nx {
                bx = nx - 1;
            }
            if val > bins[bx].1 {
                bins[bx] = (x, val);
            }
        }
    }
    let (peak_x, peak_val) =
        bins.iter().cloned().fold(
            (0.0, -1.0),
            |(bx, bv), (x, v)| {
                if v > bv { (x, v) } else { (bx, bv) }
            },
        );
    eprintln!(
        "advection: expected peak at x = {:.3}, observed peak at x = {:.3} (value {:.3})",
        x0 + t_final,
        peak_x,
        peak_val
    );
    let expected_x = x0 + t_final * wind.x;
    let rel_err = (peak_x - expected_x).abs() / expected_x;
    assert!(
        rel_err < 0.05,
        "peak displacement off by {rel_err:.3}: peak_x = {peak_x}, expected = {expected_x}"
    );
    // Sanity: the peak value should still be a significant fraction
    // of the initial 1.0 (numerical diffusion eats some).
    assert!(
        peak_val > 0.3,
        "advected pulse decayed below 30%: peak_val = {peak_val}"
    );
}

fn invert_3x3(m: &[[Real; 3]; 3]) -> [[Real; 3]; 3] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv = 1.0 / det;
    [
        [
            (e * i - f * h) * inv,
            (c * h - b * i) * inv,
            (b * f - c * e) * inv,
        ],
        [
            (f * g - d * i) * inv,
            (a * i - c * g) * inv,
            (c * d - a * f) * inv,
        ],
        [
            (d * h - e * g) * inv,
            (b * g - a * h) * inv,
            (a * e - b * d) * inv,
        ],
    ]
}

// ─── Test 3: 1D Burgers with BJ limiter stays monotone ──────────────

#[test]
fn dg_p1_burgers_with_limiter_stays_monotone() {
    // Solve 1D Burgers' equation `∂u/∂t + ∂(u²/2)/∂x = 0` on [0, 1] ×
    // (z = 0 strip), with smooth IC `u₀(x) = ½ + ½ sin(2πx)`, which
    // develops a shock around t ≈ 1/π. Without the limiter, DG-P1 +
    // LF flux develops oscillations near the shock (Gibbs-type
    // overshoots > 5%). With the BJ limiter, the solution stays
    // monotone — no overshoot > 1 % of the smooth peak (which is 1.0
    // — the initial max).
    let l: Real = 1.0;
    let h: Real = 0.05;
    let nx = 100;
    let ny = 2;
    let degree = 1usize;
    let mesh = FemMesh::tri_grid_2d(nx, ny, l, h);
    let pi = std::f64::consts::PI;
    let smooth_peak: Real = 1.0;

    // Initial: u(x) = 0.5 + 0.5 sin(2πx). All DG vertex DoFs are sampled.
    let mut u = project_scalar_dg(&mesh, degree, |x, _y| 0.5 + 0.5 * (2.0 * pi * x).sin());

    // Burgers: f(u) = u²/2. Lax-Friedrichs flux with global α = max|u|
    // — we cap at ~1.1 (initial peak is 1.0, will not grow much given
    // the slope limiter).
    let alpha_lf: Real = 1.2;
    let dt: Real = 5.0e-4;
    let nsteps = 800; // ~ 0.4 time units, well past shock formation at t ≈ 1/π ≈ 0.318.
    let neighbours = build_cell_neighbours(&mesh);
    let offsets = dg_offsets(&mesh, degree);

    // Per-cell inverse-mass blocks (3×3 each).
    let inv_mass_blocks: Vec<[[Real; 3]; 3]> = mesh
        .cells
        .iter()
        .map(|cell| {
            let m = compute_element_dg_mass(cell, &mesh.vertices, degree);
            invert_3x3(&[
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ])
        })
        .collect();

    // Per-cell P1 shape-function gradients (constant per cell).
    let cell_data: Vec<(Real, [Vec3; 3])> = mesh
        .cells
        .iter()
        .map(|cell| {
            let v = match cell {
                FemCell::Tri(v) => v,
                FemCell::Tet(_) => unreachable!(),
            };
            let p = [
                mesh.vertices[v[0] as usize],
                mesh.vertices[v[1] as usize],
                mesh.vertices[v[2] as usize],
            ];
            let det = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
            let area = 0.5 * det.abs();
            let inv = 1.0 / det;
            let g = [
                Vec3::new((p[1].y - p[2].y) * inv, (p[2].x - p[1].x) * inv, 0.0),
                Vec3::new((p[2].y - p[0].y) * inv, (p[0].x - p[2].x) * inv, 0.0),
                Vec3::new((p[0].y - p[1].y) * inv, (p[1].x - p[0].x) * inv, 0.0),
            ];
            (area, g)
        })
        .collect();

    // Pre-compute interior faces.
    let interior_faces = molla_fem::compute_interior_faces(&mesh);
    // Per-face geometry: outward normal (from L to R), length.
    struct FaceData {
        cell_l: usize,
        cell_r: usize,
        face_verts: [usize; 2],
        n_lr: Vec3,
        len: Real,
    }
    let face_data: Vec<FaceData> = interior_faces
        .iter()
        .map(|f| {
            let cl = f.cell_left as usize;
            let cr = f.cell_right as usize;
            let v_l = match &mesh.cells[cl] {
                FemCell::Tri(v) => v,
                FemCell::Tet(_) => unreachable!(),
            };
            let v_r = match &mesh.cells[cr] {
                FemCell::Tri(v) => v,
                FemCell::Tet(_) => unreachable!(),
            };
            let c_l = (mesh.vertices[v_l[0] as usize]
                + mesh.vertices[v_l[1] as usize]
                + mesh.vertices[v_l[2] as usize])
                / 3.0;
            let c_r = (mesh.vertices[v_r[0] as usize]
                + mesh.vertices[v_r[1] as usize]
                + mesh.vertices[v_r[2] as usize])
                / 3.0;
            let a = mesh.vertices[f.face_verts[0] as usize];
            let b = mesh.vertices[f.face_verts[1] as usize];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let len = (dx * dx + dy * dy).sqrt();
            let raw = Vec3::new(dy / len, -dx / len, 0.0);
            let n_lr = if raw.dot(c_r - c_l) >= 0.0 { raw } else { -raw };
            FaceData {
                cell_l: cl,
                cell_r: cr,
                face_verts: [f.face_verts[0] as usize, f.face_verts[1] as usize],
                n_lr,
                len,
            }
        })
        .collect();

    // Monotonicity check seed: max value never exceeds smooth_peak + 1%.
    let mut hist_max: Real = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut hist_min: Real = u.iter().cloned().fold(f64::INFINITY, f64::min);

    for step in 0..nsteps {
        // Compute residual r_i = (volumetric u²/2 · ∂N_i/∂x) − face flux.
        let n = *offsets.last().unwrap();
        let mut r = vec![0.0_f64 as Real; n];

        // ── Element residuals.
        // r_i^K = ∫_K (u²/2) ∂N_i/∂x dx. Using P1 nodal expansion
        // u_h(x) = Σ u_j N_j(x), so (u_h)² is quadratic per cell. We
        // use a 3-pt quadrature (edge midpoints, degree-2 exact for
        // the (u²)·constant integrand).
        for ci in 0..mesh.cells.len() {
            let off = offsets[ci];
            let (area, grads) = &cell_data[ci];
            // 3-pt quadrature: at edge midpoints u_h is the average of
            // two vertex values.
            let u0 = u[off];
            let u1 = u[off + 1];
            let u2 = u[off + 2];
            let uvals = [(u0 + u1) * 0.5, (u1 + u2) * 0.5, (u2 + u0) * 0.5];
            let w = *area / 3.0;
            for q in 0..3 {
                let f = uvals[q] * uvals[q] * 0.5; // f(u) = u² / 2.
                // F(u) is a vector flux F = (f, 0). Contribution:
                //   r_i += w · F · ∇N_i = w · f · (∂N_i/∂x).
                for i in 0..3 {
                    r[off + i] += w * f * grads[i].x;
                }
            }
            // Strict 0 in y component because the IC is 1D-symmetric.
        }

        // ── Face residuals: r_i -= ∫_F Ĥ ⟦N_i⟧.
        // Lax-Friedrichs: Ĥ(u_L, u_R, n) = ½(F(u_L)+F(u_R))·n − ½α(u_R−u_L)
        // For ⟦v = N_L^i⟧ = +N_L^i. For ⟦v = N_R^i⟧ = −N_R^i.
        for f in &face_data {
            let off_l = offsets[f.cell_l];
            let off_r = offsets[f.cell_r];
            // Sample u at face midpoint (P1 ⇒ midpoint value = mean of
            // the two face-vertex DG DoFs on each side). Use 2-pt
            // Gauss-Legendre on the edge (degree-3 exact for quadratic
            // u² · linear-N integrand).
            let v_l = match &mesh.cells[f.cell_l] {
                FemCell::Tri(v) => v,
                FemCell::Tet(_) => unreachable!(),
            };
            let v_r = match &mesh.cells[f.cell_r] {
                FemCell::Tri(v) => v,
                FemCell::Tet(_) => unreachable!(),
            };
            // Map global face vertex → local (0/1/2) on each side.
            let local_l: [usize; 2] = [
                v_l.iter()
                    .position(|&q| q as usize == f.face_verts[0])
                    .unwrap(),
                v_l.iter()
                    .position(|&q| q as usize == f.face_verts[1])
                    .unwrap(),
            ];
            let local_r: [usize; 2] = [
                v_r.iter()
                    .position(|&q| q as usize == f.face_verts[0])
                    .unwrap(),
                v_r.iter()
                    .position(|&q| q as usize == f.face_verts[1])
                    .unwrap(),
            ];

            let a_pt = mesh.vertices[f.face_verts[0]];
            let b_pt = mesh.vertices[f.face_verts[1]];
            let gp = 1.0 / (3.0_f64).sqrt() / 2.0;
            let xis = [0.5 - gp, 0.5 + gp];
            let w_q = 0.5 * f.len;
            for &xi in &xis {
                // Evaluate u_L, u_R at the quadrature point.
                let _xq = a_pt + (b_pt - a_pt) * xi;
                // Linear interp along the face: u = (1-xi) u(a) + xi u(b)
                // where u(a)/u(b) are the per-side DG values at the
                // face's two endpoints.
                let u_l_at_a = u[off_l + local_l[0]];
                let u_l_at_b = u[off_l + local_l[1]];
                let u_l = (1.0 - xi) * u_l_at_a + xi * u_l_at_b;
                let u_r_at_a = u[off_r + local_r[0]];
                let u_r_at_b = u[off_r + local_r[1]];
                let u_r = (1.0 - xi) * u_r_at_a + xi * u_r_at_b;

                // Flux F(u) = (u²/2, 0) → F·n = (u²/2) n.x.
                let f_l = u_l * u_l * 0.5 * f.n_lr.x;
                let f_r = u_r * u_r * 0.5 * f.n_lr.x;
                let h_hat = 0.5 * (f_l + f_r) - 0.5 * alpha_lf * (u_r - u_l);

                // Test-function values on each side at this point.
                let nl_at_face_a = (1.0 - xi) * 1.0; // N_L^{local_l[0]} at this pt
                let nl_at_face_b = xi * 1.0; //         N_L^{local_l[1]}
                // (Third L vertex has zero value on this face by Lagrange property.)
                let nr_at_face_a = (1.0 - xi) * 1.0;
                let nr_at_face_b = xi * 1.0;

                // r_i -= w_q · h_hat · ⟦N_i⟧.
                r[off_l + local_l[0]] -= w_q * h_hat * nl_at_face_a;
                r[off_l + local_l[1]] -= w_q * h_hat * nl_at_face_b;
                r[off_r + local_r[0]] -= w_q * h_hat * (-nr_at_face_a);
                r[off_r + local_r[1]] -= w_q * h_hat * (-nr_at_face_b);
            }
        }

        // ── Forward Euler: u ← u + dt · M⁻¹ · r.
        for ci in 0..mesh.cells.len() {
            let off = offsets[ci];
            let inv_m = &inv_mass_blocks[ci];
            let r_local = [r[off], r[off + 1], r[off + 2]];
            let mut du = [0.0_f64 as Real; 3];
            for i in 0..3 {
                for j in 0..3 {
                    du[i] += inv_m[i][j] * r_local[j];
                }
            }
            u[off] += dt * du[0];
            u[off + 1] += dt * du[1];
            u[off + 2] += dt * du[2];
        }

        // ── Apply slope limiter (BJ scalar).
        limit_solution(&mesh, &mut u, degree, &neighbours, 0.0, 1.0);

        // Track running min / max for monotonicity check.
        let smax: Real = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let smin: Real = u.iter().cloned().fold(f64::INFINITY, f64::min);
        if smax > hist_max {
            hist_max = smax;
        }
        if smin < hist_min {
            hist_min = smin;
        }
        if step % 100 == 0 {
            eprintln!(
                "burgers step {step}: u_min = {smin:.4}, u_max = {smax:.4} (peak budget {:.4})",
                smooth_peak * 1.01
            );
        }
    }

    eprintln!("burgers final: max(u) over history = {hist_max:.5}, min(u) = {hist_min:.5}");
    // No overshoot > 1 % of smooth_peak (= 1.0).
    assert!(
        hist_max <= smooth_peak * 1.01,
        "Burgers overshoot: max(u) = {hist_max}, smooth_peak * 1.01 = {}",
        smooth_peak * 1.01
    );
    // No undershoot below initial-data floor (0.0).
    assert!(hist_min >= -0.01, "Burgers undershoot: min(u) = {hist_min}");
}
