//! Stage 45 v4 — AMR coarsening (un-refinement) + Krylov restart after
//! re-mesh.
//!
//! The four tests cover the v4 deliverables:
//!
//! 1. `amr_coarsen_undoes_refinement` — refine uniformly then coarsen
//!    every refined parent; the result must equal the original base
//!    mesh DoF-for-DoF (same flat cell count, same flat vertex count,
//!    same per-vertex coordinates).
//!
//! 2. `amr_warm_start_reduces_iters` — set up a Poisson problem on a
//!    coarse AMR mesh, solve cold-start. Refine, transfer state,
//!    solve warm-start via `AmrMesh::solve_after_remesh`. The warm-
//!    start iteration count must be lower than the cold-start one.
//!
//! 3. `amr_coarsen_then_refine_partial_recovers` — uniform refine, then
//!    coarsen all, then refine only a subset → must equal the direct
//!    half-refinement from the base mesh.
//!
//! 4. `amr_coarsen_cell_rejects_non_leaf_children` — `coarsen_cell`
//!    must fail (returning `Err`) when one of the children has itself
//!    been refined, and must leave the mesh state untouched.

use molla_fem::{
    AmrMesh, BoundaryCondition, FemCell, FemMesh, FunctionSpace, apply_dirichlet,
    assemble_global_amr, compute_element_stiffness,
};
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// ─── Helpers ─────────────────────────────────────────────────────────

/// Find the AMR cell id of the lower triangle of quad `(i, j)` in a
/// `tri_grid_2d(nx, ny)` base mesh. Returns `None` if there is no
/// matching base cell.
fn locate_lower_tri(amr: &AmrMesh, i: usize, j: usize) -> Option<usize> {
    let n_verts = amr.base_mesh.vert_count();
    let stride = (n_verts as f64).sqrt() as usize;
    let v00 = (j * stride + i) as u32;
    let v10 = (j * stride + i + 1) as u32;
    let v11 = ((j + 1) * stride + i + 1) as u32;
    for (idx, c) in amr.cells().iter().enumerate() {
        if let FemCell::Tri(v) = c {
            if v == &[v00, v10, v11] {
                return Some(idx);
            }
        }
    }
    None
}

/// Sort the leaf cells of an AMR mesh into a deterministic order so two
/// meshes built by different histories can be compared after `flatten`.
/// We sort by the sorted vertex tuple of each cell — that is invariant
/// under remap permutations as long as the underlying geometry is the
/// same.
fn flat_cell_signatures(flat: &FemMesh) -> Vec<[Vec3; 4]> {
    let mut sigs: Vec<[Vec3; 4]> = flat
        .cells
        .iter()
        .map(|c| match c {
            FemCell::Tri(v) => {
                let mut ps = [
                    flat.vertices[v[0] as usize],
                    flat.vertices[v[1] as usize],
                    flat.vertices[v[2] as usize],
                    Vec3::new(Real::INFINITY, Real::INFINITY, Real::INFINITY),
                ];
                ps[..3].sort_by(|a, b| {
                    a.x.partial_cmp(&b.x)
                        .unwrap()
                        .then(a.y.partial_cmp(&b.y).unwrap())
                        .then(a.z.partial_cmp(&b.z).unwrap())
                });
                ps
            }
            FemCell::Tet(v) => {
                let mut ps = [
                    flat.vertices[v[0] as usize],
                    flat.vertices[v[1] as usize],
                    flat.vertices[v[2] as usize],
                    flat.vertices[v[3] as usize],
                ];
                ps.sort_by(|a, b| {
                    a.x.partial_cmp(&b.x)
                        .unwrap()
                        .then(a.y.partial_cmp(&b.y).unwrap())
                        .then(a.z.partial_cmp(&b.z).unwrap())
                });
                ps
            }
        })
        .collect();
    sigs.sort_by(|a, b| {
        for k in 0..4 {
            let o = a[k]
                .x
                .partial_cmp(&b[k].x)
                .unwrap()
                .then(a[k].y.partial_cmp(&b[k].y).unwrap())
                .then(a[k].z.partial_cmp(&b[k].z).unwrap());
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });
    sigs
}

/// Compare two cell-signature lists for equality within a small
/// geometric tolerance.
fn signatures_match(a: &[[Vec3; 4]], b: &[[Vec3; 4]]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (sa, sb) in a.iter().zip(b.iter()) {
        for k in 0..4 {
            let d = sa[k] - sb[k];
            if d.length() > 1.0e-9 {
                return false;
            }
        }
    }
    true
}

// ─── Test 1: coarsen undoes uniform refinement ───────────────────────
//
// Refine a tri grid uniformly, then coarsen every refined parent. The
// resulting flat mesh should equal the base mesh exactly.

#[test]
fn amr_coarsen_undoes_refinement() {
    let base = FemMesh::tri_grid_2d(3, 3, 1.0, 1.0);
    let base_flat_sigs = flat_cell_signatures(&base);

    let mut amr = AmrMesh::new(base.clone());
    amr.refine_uniform();
    let n_after_refine = amr.leaf_count();
    assert_eq!(
        n_after_refine,
        4 * base.cell_count(),
        "uniform refinement of tri mesh should quadruple leaf count"
    );

    // Coarsen every refined parent (predicate: always true → coarsen
    // every refined parent).
    let n_coarsened = amr.coarsen_where(|_cell: &FemCell| true);
    assert_eq!(
        n_coarsened,
        base.cell_count(),
        "expected to coarsen exactly {} parents, coarsened {}",
        base.cell_count(),
        n_coarsened
    );
    assert_eq!(
        amr.leaf_count(),
        base.cell_count(),
        "leaf count after full coarsening should match base mesh"
    );

    // Compare flattened mesh DoF-for-DoF against the base.
    let (post_flat, _) = amr.flatten_with_remap();
    assert_eq!(
        post_flat.cell_count(),
        base.cell_count(),
        "flat cell count after full coarsening mismatch"
    );
    assert_eq!(
        post_flat.vert_count(),
        base.vert_count(),
        "flat vertex count after full coarsening mismatch — orphaned midpoints leaking?"
    );

    let post_sigs = flat_cell_signatures(&post_flat);
    assert!(
        signatures_match(&base_flat_sigs, &post_sigs),
        "flattened mesh after coarsening does not match base mesh DoF-for-DoF"
    );

    // Hanging-node list must be empty on a uniform un-refinement.
    assert!(
        amr.hanging_nodes().is_empty(),
        "expected no hanging nodes after full coarsening; got {}",
        amr.hanging_nodes().len()
    );
}

// ─── Test 2: warm-start CG iters < cold-start CG iters ───────────────
//
// Setup:
// - Coarse base mesh `tri_grid_2d(8, 8)`.
// - Laplace `−Δu = 0` with non-trivial Dirichlet boundary data
//   `g(x, y) = sin(πx) · cos(πy) + 0.7 · x · y` (mixed analytic, not
//   a single Laplacian eigenmode, so the discrete solution exercises
//   many Krylov directions).
// - Boundary-driven solves keep the warm/cold magnitudes consistent
//   across coarse and fine grids: the Dirichlet pin enters the
//   reduced RHS as a column-elimination term scaled by the pinned
//   value, so refining doesn't rescale the solution magnitude.
//
// Procedure:
// 1. Solve on the coarse AMR mesh — cold start (`x0 = 0`).
// 2. Refine the AMR mesh once (uniformly).
// 3. Solve on the refined mesh in TWO ways:
//      a. Cold start (`x0 = 0`).
//      b. Warm start with `transfer_state` from the coarse solution.
// 4. Assert: warm-start CG iters < cold-start CG iters.

/// Non-trivial Dirichlet boundary data — not a single Laplacian
/// eigenmode, so the discrete Laplace solution does not collapse into
/// a low-dimensional Krylov subspace.
fn dirichlet_data(p: Vec3) -> Real {
    use std::f64::consts::PI;
    (PI * p.x).sin() * (PI * p.y).cos() + 0.7 * p.x * p.y
}

#[test]
fn amr_warm_start_reduces_iters() {
    let base = FemMesh::tri_grid_2d(8, 8, 1.0, 1.0);

    // ── Step 1: solve Laplace `K · u = 0` with Dirichlet `g` on the
    // coarse AMR mesh. Returns the **full P1** coarse solution
    // (boundary + interior, all length `coarse_flat.vert_count()`).
    let amr_coarse = AmrMesh::new(base.clone());
    let (u_coarse_full, _coarse_flat) = solve_laplace_dirichlet(&amr_coarse);

    // ── Step 2: refine the AMR mesh ───────────────────────────────────
    let mut amr_fine = AmrMesh::new(base.clone());
    amr_fine.refine_uniform();
    let (fine_flat, _) = amr_fine.flatten_with_remap();

    // ── Build the fine Laplace system once (used for both cold + warm).
    let (k_fine_full, _f_fine_zero, _t) = assemble_global_amr(
        &amr_fine,
        FunctionSpace::P1Nodal,
        |cell, verts| compute_element_stiffness(cell, verts, 1.0),
        None,
    )
    .expect("assemble fine");
    let (k_fine_pinned, f_fine_pinned, free_idx_of) =
        pin_dirichlet_boundary(&k_fine_full, &fine_flat, dirichlet_data);
    let n_fine_free = k_fine_pinned.nrows();

    // ── Step 3a: cold-start solve on the refined mesh ─────────────────
    let x_zero_cold = vec![0.0; n_fine_free];
    let (x_cold, iters_cold) = AmrMesh::solve_after_remesh(
        &k_fine_pinned,
        &f_fine_pinned,
        &x_zero_cold,
        20_000,
        1.0e-10,
    );

    // ── Step 3b: warm-start solve on the refined mesh ─────────────────
    // Transfer the **full** coarse solution onto the fine flat mesh.
    let u_warm_full = amr_fine.transfer_state(&u_coarse_full, &amr_coarse);
    // Extract free-DoF entries in the same order as `pin_dirichlet_boundary`.
    let mut x_warm_free = vec![0.0; n_fine_free];
    for (i, &fi) in free_idx_of.iter().enumerate() {
        if fi >= 0 {
            x_warm_free[fi as usize] = u_warm_full[i];
        }
    }
    let (x_warm, iters_warm) = AmrMesh::solve_after_remesh(
        &k_fine_pinned,
        &f_fine_pinned,
        &x_warm_free,
        20_000,
        1.0e-10,
    );

    println!(
        "Stage-45 v4 warm-start vs cold-start CG: \
         coarse n_total = {}, fine n_free = {}, \
         iters_cold = {}, iters_warm = {}",
        u_coarse_full.len(),
        n_fine_free,
        iters_cold,
        iters_warm
    );

    // Both must reach the same answer (within tolerance).
    let mut diff_max: Real = 0.0;
    for (a, b) in x_cold.iter().zip(x_warm.iter()) {
        let d = (a - b).abs();
        if d > diff_max {
            diff_max = d;
        }
    }
    assert!(
        diff_max < 1.0e-6,
        "cold and warm solves disagree by {} > 1e-6",
        diff_max
    );

    assert!(
        iters_warm < iters_cold,
        "warm-start CG took {} iters vs cold-start {} — expected strictly fewer",
        iters_warm,
        iters_cold
    );
}

// ─── Test 3: coarsen all then refine half ↔ direct half refinement ──
//
// Refine uniformly, coarsen everything (back to base), then refine half
// of the base cells. Separately: starting from the base, refine the
// same half directly. Assert the resulting flat meshes are identical.

#[test]
fn amr_coarsen_then_refine_partial_recovers() {
    let base = FemMesh::tri_grid_2d(3, 3, 1.0, 1.0);
    let n_base = base.cell_count();

    // ── Build "direct half-refinement" ────────────────────────────────
    // Refine only the bottom row of base cells (j = 0).
    let mut amr_direct = AmrMesh::new(base.clone());
    let mut targets_direct: Vec<usize> = Vec::new();
    for i in 0..3 {
        if let Some(id) = locate_lower_tri(&amr_direct, i, 0) {
            targets_direct.push(id);
        }
    }
    for idx in &targets_direct {
        amr_direct.refine_cell(*idx);
    }
    amr_direct.propagate_refinement();
    amr_direct.compute_hanging_nodes();
    let (direct_flat, _) = amr_direct.flatten_with_remap();
    let direct_sigs = flat_cell_signatures(&direct_flat);

    // ── Build "coarsen then partial-refine" ───────────────────────────
    let mut amr_round = AmrMesh::new(base.clone());
    amr_round.refine_uniform();
    let n_coarsened = amr_round.coarsen_where(|_cell: &FemCell| true);
    assert_eq!(
        n_coarsened, n_base,
        "expected to coarsen exactly {} cells, got {}",
        n_base, n_coarsened
    );
    // After full coarsening, base cells should once again be leaves at
    // their original flat indices.
    for i in 0..3 {
        let id =
            locate_lower_tri(&amr_round, i, 0).expect("base cell still present after round-trip");
        amr_round.refine_cell(id);
    }
    amr_round.propagate_refinement();
    amr_round.compute_hanging_nodes();
    let (round_flat, _) = amr_round.flatten_with_remap();
    let round_sigs = flat_cell_signatures(&round_flat);

    assert_eq!(
        direct_flat.cell_count(),
        round_flat.cell_count(),
        "cell counts differ between direct ({}) and round-trip ({})",
        direct_flat.cell_count(),
        round_flat.cell_count()
    );
    assert_eq!(
        direct_flat.vert_count(),
        round_flat.vert_count(),
        "vertex counts differ between direct ({}) and round-trip ({})",
        direct_flat.vert_count(),
        round_flat.vert_count()
    );

    assert!(
        signatures_match(&direct_sigs, &round_sigs),
        "round-trip mesh does not match direct mesh — coarsen+refine does not commute"
    );
}

// ─── Test 4: coarsen_cell rejects non-leaf children ───────────────────

#[test]
fn amr_coarsen_cell_rejects_non_leaf_children() {
    let base = FemMesh::tri_grid_2d(3, 3, 1.0, 1.0);
    let mut amr = AmrMesh::new(base);
    let c = locate_lower_tri(&amr, 1, 1).expect("centre lower tri");
    amr.refine_cell(c);
    let child0 = amr.refined_cells[0].children.as_slice()[0] as usize;
    amr.refine_cell(child0);

    // Try to coarsen `c` — its child0 is now refined, so this must fail.
    let n_cells_before = amr.cell_count();
    let n_refined_before = amr.refined_cells.len();
    let err = amr.coarsen_cell(c);
    assert!(
        err.is_err(),
        "expected coarsen_cell to refuse non-leaf children"
    );

    // Mesh state must be intact.
    assert!(amr.is_removed_cell(child0) == false);
    assert_eq!(amr.cell_count(), n_cells_before);
    assert_eq!(amr.refined_cells.len(), n_refined_before);

    // Coarsen child0 first (its children are leaves) — should succeed.
    amr.coarsen_cell(child0).expect("coarsen leaf-child parent");
    // Now coarsen `c` — should succeed.
    amr.coarsen_cell(c)
        .expect("coarsen original parent after child");

    // Mesh should be back to the base.
    assert_eq!(amr.refined_cells.len(), 0);
    assert!(amr.is_leaf(c));
}

// ─── Local helpers for the warm-start test ───────────────────────────

fn is_boundary(p: Vec3) -> bool {
    p.x.abs() < 1.0e-9
        || (p.x - 1.0).abs() < 1.0e-9
        || p.y.abs() < 1.0e-9
        || (p.y - 1.0).abs() < 1.0e-9
}

/// Eliminate boundary DoFs from a Laplace stiffness matrix by pinning
/// each boundary vertex to `g(p)`. Returns:
///   - `k_pinned`: the interior × interior sub-matrix of `k_full`.
///   - `f_pinned`: the column-elimination RHS — for each free row `r`
///     and each boundary column `b`, `f_pinned[r] -= k_full[r, b] * g(p_b)`.
///   - `free_idx_of`: a length-`flat.vert_count()` lookup mapping
///     full-DoF index → free-DoF index (`-1` for boundary verts).
///
/// The boundary-driven Laplace solve `K · u = 0, u|_∂Ω = g` reduces to
/// `k_pinned · u_free = f_pinned` after this column elimination.
fn pin_dirichlet_boundary<G: Fn(Vec3) -> Real>(
    k_full: &BsrMatrix,
    flat: &FemMesh,
    g: G,
) -> (BsrMatrix, Vec<Real>, Vec<i32>) {
    let mut free_idx_of: Vec<i32> = vec![-1; flat.vert_count()];
    let mut n_free = 0usize;
    for (i, p) in flat.vertices.iter().enumerate() {
        if !is_boundary(*p) {
            free_idx_of[i] = n_free as i32;
            n_free += 1;
        }
    }

    assert_eq!(k_full.b, 1, "pin_dirichlet_boundary expects scalar BSR");

    // f_pinned starts at zero and accumulates `-K[r, b] · g(p_b)` for
    // every (interior r, boundary b) pair.
    let mut f_pinned = vec![0.0; n_free];
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for r in 0..k_full.nrows() {
        let fi = free_idx_of[r];
        if fi < 0 {
            continue;
        }
        let lo = k_full.row_offsets[r];
        let hi = k_full.row_offsets[r + 1];
        for slot in lo..hi {
            let c = k_full.col_idx[slot] as usize;
            let v = k_full.values[slot];
            let fj = free_idx_of[c];
            if fj < 0 {
                // Boundary column — fold into RHS.
                let gb = g(flat.vertices[c]);
                f_pinned[fi as usize] -= v * gb;
            } else {
                triplets.push((fi as usize, fj as usize, v));
            }
        }
    }
    let k_pinned = BsrMatrix::from_scalar_triplets(n_free, n_free, &triplets).expect("k_pinned");

    // Suppress unused warnings for the v3 BC types (still imported for
    // future extensibility of the test).
    let _ = apply_dirichlet;
    let _ = BoundaryCondition::Dirichlet {
        vertex: 0,
        value: 0.0,
    };
    (k_pinned, f_pinned, free_idx_of)
}

/// Solve `K · u = 0` with Dirichlet `u = g` on the boundary on the
/// supplied AMR mesh. Returns the **full** P1 nodal solution
/// (interior interior values + boundary `g`-values) plus the flat
/// mesh — sized so the caller can pass the solution into
/// [`AmrMesh::transfer_state`].
fn solve_laplace_dirichlet(amr: &AmrMesh) -> (Vec<Real>, FemMesh) {
    let (k_full, _f, _t) = assemble_global_amr(
        amr,
        FunctionSpace::P1Nodal,
        |cell, verts| compute_element_stiffness(cell, verts, 1.0),
        None,
    )
    .expect("assemble coarse");

    let (flat, _) = amr.flatten_with_remap();
    let (k_pinned, f_pinned, free_idx_of) = pin_dirichlet_boundary(&k_full, &flat, dirichlet_data);

    let n_free = k_pinned.nrows();
    let x0 = vec![0.0; n_free];
    let (u_free, _iters) = AmrMesh::solve_after_remesh(&k_pinned, &f_pinned, &x0, 20_000, 1.0e-12);

    // Reconstruct the full solution.
    let mut u_full = vec![0.0; flat.vert_count()];
    for (i, p) in flat.vertices.iter().enumerate() {
        let fi = free_idx_of[i];
        u_full[i] = if fi >= 0 {
            u_free[fi as usize]
        } else {
            dirichlet_data(*p)
        };
    }
    (u_full, flat)
}
