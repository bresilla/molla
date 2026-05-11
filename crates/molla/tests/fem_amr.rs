//! Stage 45 v1 — `AmrMesh` foundation + uniform refinement.
//!
//! Each test maps 1:1 to one of the four `Tests` items in the Stage 45
//! v1 brief.

use std::collections::HashMap;

use molla::fem::{
    assemble_global, compute_element_stiffness, compute_edges, near_point,
    solve_steady_state, AmrMesh, BoundaryCondition, FemCell, FemMesh, FunctionSpace,
};
use molla::math::sparse::BsrMatrix;
use molla::math::{Real, Vec3};

/// Quick helper — stiffness assembly via the existing global path.
fn assemble_stiffness(mesh: &FemMesh, alpha: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .expect("assemble stiffness")
}

// ─── Test 1: triangle uniform refinement quadruples cells ───────────
//
// Refining every triangle once produces 4 sub-triangles per parent
// (red refinement). The new vertex count equals `V₀ + E₀` because every
// unique mesh edge contributes exactly one midpoint, and adjacent
// triangles share their midpoints via the cache.

#[test]
fn amr_tri_uniform_refine_doubles_resolution() {
    let base = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
    let original_cells = base.cell_count();
    let original_verts = base.vert_count();
    let (edges, _) = compute_edges(&base);
    let original_edges = edges.len();

    let mut amr = AmrMesh::new(base);
    amr.refine_uniform();

    // After one uniform refinement, every parent has split → leaf count
    // is 4× original.
    assert_eq!(amr.leaf_count(), 4 * original_cells);

    // Flatten and recheck the geometric counts on the produced FemMesh.
    let flat = amr.flatten();
    assert_eq!(flat.cells.len(), 4 * original_cells);
    assert_eq!(
        flat.vertices.len(),
        original_verts + original_edges,
        "expected V0 + E0 vertices after one red refinement"
    );
}

// ─── Test 2: tetrahedral uniform refinement gives 8× cells ──────────
//
// Bey's red refinement splits each tet into 8 sub-tets. Base mesh is
// `2×2×2` hexes → `8 hex × 6 tets/hex = 48` parent tets. After one
// uniform refinement we expect `48 × 8 = 384` leaves.

#[test]
fn amr_tet_uniform_refine_8x_cells() {
    let base = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
    let original_cells = base.cell_count();
    assert_eq!(original_cells, 8 * 6); // sanity

    let mut amr = AmrMesh::new(base);
    amr.refine_uniform();

    assert_eq!(amr.leaf_count(), 8 * original_cells);
    let flat = amr.flatten();
    assert_eq!(flat.cells.len(), 8 * original_cells);

    // Every child tet must be positively oriented — otherwise the
    // existing FEM element kernels would silently produce wrong sign
    // contributions during assembly.
    for cell in &flat.cells {
        if let FemCell::Tet(vs) = cell {
            let a = flat.vertices[vs[0] as usize];
            let b = flat.vertices[vs[1] as usize];
            let c = flat.vertices[vs[2] as usize];
            let d = flat.vertices[vs[3] as usize];
            let vol = (b - a).dot((c - a).cross(d - a)) / 6.0;
            assert!(vol > 0.0, "child tet has non-positive volume {vol}");
        }
    }
}

// ─── Test 3: `near_point` criterion refines only nearby cells ───────
//
// Build a `4×4` triangle grid on `[0,1]²`. Pick a circular region near
// the centre (radius 0.2 around `(0.5, 0.5)`). Verify that every
// refined cell's centroid sits within the radius, and that every
// cell outside the radius remains a leaf at level 0.

#[test]
fn amr_near_point_criterion_refines_only_nearby() {
    let base = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
    let mut amr = AmrMesh::new(base);
    let p = Vec3::new(0.5, 0.5, 0.0);
    let radius = 0.2;
    amr.refine_where(near_point(p, radius));

    let n_base = amr.refined_cells.iter().map(|r| r.parent).max().map(|m| m + 1).unwrap_or(0)
        as usize;
    // We expect *some* refinement, but not all — strictly between.
    assert!(!amr.refined_cells.is_empty(), "no cells were refined");

    // Every refined parent's centroid must lie inside the radius.
    for r in &amr.refined_cells {
        let cell = amr.cells()[r.parent as usize];
        let centroid = centroid_of(&cell, amr.vertices());
        let d = (centroid - p).length();
        assert!(
            d <= radius + 1e-12,
            "refined cell {} has centroid distance {} > radius {}",
            r.parent,
            d,
            radius
        );
    }

    // Every original base cell whose centroid is *outside* the radius
    // must still be a leaf at level 0.
    let base_cells = amr.cells()[..n_base.max(0)].to_vec();
    let total_base = amr.cells().len()
        - amr
            .cells()
            .iter()
            .enumerate()
            .skip(base_cells.len())
            .count();
    let _ = total_base; // unused — informational

    // Walk only the base cells (level 0) — they have indices
    // `0..base_mesh.cell_count()`. We get the count from the AmrMesh
    // via `base_mesh.cells.len()`.
    let n_base_cells_known = amr_base_cell_count(&amr);
    for idx in 0..n_base_cells_known {
        let cell = amr.cells()[idx];
        let centroid = centroid_of(&cell, amr.vertices());
        let d = (centroid - p).length();
        if d > radius {
            assert!(
                amr.is_leaf(idx),
                "base cell {idx} (centroid distance {d}) is outside the radius but was refined"
            );
        }
    }
}

// ─── Test 4: flatten → assemble → solve matches uniform-fine mesh ───
//
// Refine the base `tri_grid_2d(nx, ny)` uniformly once, flatten it,
// then run `solve_steady_state` on a harmonic Dirichlet problem
// `u(x, y) = x − y`. The result must match a *hand-constructed*
// uniform-fine mesh `tri_grid_2d(2·nx, 2·ny)` of the same resolution
// to `1e-10`.
//
// The two meshes resolve the same continuous problem with the same
// piecewise-linear basis and the same set of interior + boundary
// vertices (geometrically identical), so the solutions agree exactly
// at corresponding spatial positions (modulo CG round-off). We compare
// by *position* because the flatten path's vertex ordering is "first
// touch in leaf-iteration order" while `tri_grid_2d` is in
// `(i, j)` raster order — the matrices are equal up to a row/column
// permutation, not bit-identical.

#[test]
fn amr_flatten_then_assemble_steady_heat_matches_uniform_refined() {
    let nx = 4usize;
    let ny = 4usize;

    // ── AMR path: refine uniformly, then flatten.
    let base = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let mut amr = AmrMesh::new(base);
    amr.refine_uniform();
    let amr_flat = amr.flatten();

    // ── Reference path: hand-built uniform-fine mesh.
    let ref_mesh = FemMesh::tri_grid_2d(2 * nx, 2 * ny, 1.0, 1.0);

    // Same number of leaves and the same vertex count (every edge of
    // the base contributes 1 midpoint, which equals the count of new
    // vertices the `2·nx`-resolution mesh adds over the `nx`).
    assert_eq!(amr_flat.cells.len(), ref_mesh.cells.len());
    assert_eq!(amr_flat.vertices.len(), ref_mesh.vertices.len());

    let bcs_for = |mesh: &FemMesh| -> Vec<BoundaryCondition> {
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            let on_bd = p.x.abs() < 1e-9
                || (p.x - 1.0).abs() < 1e-9
                || p.y.abs() < 1e-9
                || (p.y - 1.0).abs() < 1e-9;
            if on_bd {
                bcs.push(BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: p.x - p.y,
                });
            }
        }
        bcs
    };

    // Solve on both meshes.
    let k_amr = assemble_stiffness(&amr_flat, 1.0);
    let f_amr = vec![0.0; amr_flat.vertices.len()];
    let u_amr = solve_steady_state(&k_amr, &f_amr, &bcs_for(&amr_flat), 5000, 1e-14);

    let k_ref = assemble_stiffness(&ref_mesh, 1.0);
    let f_ref = vec![0.0; ref_mesh.vertices.len()];
    let u_ref = solve_steady_state(&k_ref, &f_ref, &bcs_for(&ref_mesh), 5000, 1e-14);

    // Build a position → reference-index lookup so we can compare per
    // physical vertex (orderings differ between the two paths).
    let mut ref_idx_by_pos: HashMap<(i64, i64), usize> = HashMap::new();
    let scale = 1e9;
    for (i, p) in ref_mesh.vertices.iter().enumerate() {
        ref_idx_by_pos.insert(((p.x * scale).round() as i64, (p.y * scale).round() as i64), i);
    }

    let mut max_diff: Real = 0.0;
    for (i, p) in amr_flat.vertices.iter().enumerate() {
        let key = ((p.x * scale).round() as i64, (p.y * scale).round() as i64);
        let j = *ref_idx_by_pos.get(&key).unwrap_or_else(|| {
            panic!(
                "vertex {} at ({}, {}) on the AMR-flat mesh has no match on the reference mesh",
                i, p.x, p.y
            )
        });
        let d = (u_amr[i] - u_ref[j]).abs();
        max_diff = max_diff.max(d);
    }
    assert!(
        max_diff < 1e-10,
        "AMR-flat solution vs uniform-fine solution: max |Δu| = {max_diff}"
    );

    // Bonus check — both solutions agree with the analytic harmonic
    // field `u = x − y` at every vertex (the field is linear, so the
    // P1 mesh nails it exactly).
    let mut max_analytic: Real = 0.0;
    for (p, &v) in amr_flat.vertices.iter().zip(u_amr.iter()) {
        max_analytic = max_analytic.max((v - (p.x - p.y)).abs());
    }
    assert!(
        max_analytic < 1e-10,
        "AMR-flat solution deviates from analytic harmonic field by {max_analytic}"
    );
}

// ── Helpers ──────────────────────────────────────────────────────────

fn centroid_of(cell: &FemCell, vertices: &[Vec3]) -> Vec3 {
    let verts = cell.verts();
    let n = verts.len() as Real;
    let mut c = Vec3::new(0.0, 0.0, 0.0);
    for &v in verts {
        c = c + vertices[v as usize];
    }
    c * (1.0 / n)
}

/// Pull the base-mesh cell count off of an [`AmrMesh`]. There is no
/// public accessor for this in v1 — we infer it from the `base_mesh`
/// field which IS public.
fn amr_base_cell_count(amr: &AmrMesh) -> usize {
    amr.base_mesh.cell_count()
}
