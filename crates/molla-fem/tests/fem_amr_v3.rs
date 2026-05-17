//! Stage 45 v3 — P2 hanging-node constraints, DG-AMR flux balance, and
//! state interpolation between refinement events.
//!
//! The three tests cover the three v3 deliverables:
//!
//! 1. `amr_p2_hanging_constraint_preserves_quadratic_field` — P2
//!    constraint matrix reproduces a known `u = x² − y²` field on all
//!    hanging DoFs of a non-uniformly refined mesh. The field is
//!    harmonic (P2 should recover it exactly) and quadratic in `(x, y)`
//!    (sensitive to the curvature term in the P2 constraint).
//!
//! 2. `dg_amr_p1_flux_balanced_at_T_junction` — for a DG-P1 setup, the
//!    flux integral over a coarse face equals the sum of the flux
//!    integrals over its two child sub-faces, where the children come
//!    from the AMR refinement tree.
//!
//! 3. `amr_state_transfer_preserves_linear_field` — a linear P1 field
//!    `u = x + y` interpolated from a coarse AMR mesh onto a refined
//!    AMR mesh is recovered to machine precision (P1 reproduces affine
//!    functions exactly).

use molla_fem::{
    AmrMesh, BoundaryCondition, FemCell, FemMesh, P2DofMap, apply_dirichlet, compute_edges,
    compute_element_p2_stiffness, compute_ttkt, tt_times_vector,
};
use molla_math::iterative::cg_solve;
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// ─── Helpers ─────────────────────────────────────────────────────────

/// Find the AMR cell id of the lower triangle of quad `(i, j)` in a
/// `tri_grid_2d(nx, ny)` base mesh by pattern-matching its vertex
/// tuple `[v00, v10, v11]`. Assumes a square grid.
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

/// Build a P2 nodal field by evaluating `f(p)` at every P2 DoF
/// position on the flat mesh. Output layout: vertex DoFs first
/// (`flat.vertices` order), then edge mid-DoFs (`compute_edges` order
/// — same as the `P2DofMap`).
fn p2_sample_field<F: Fn(Vec3) -> Real>(flat: &FemMesh, f: F) -> Vec<Real> {
    let (edges, _) = compute_edges(flat);
    let mut out = vec![0.0; flat.vert_count() + edges.len()];
    for (i, p) in flat.vertices.iter().enumerate() {
        out[i] = f(*p);
    }
    for (i, e) in edges.iter().enumerate() {
        let p = (flat.vertices[e[0] as usize] + flat.vertices[e[1] as usize]) * 0.5;
        out[flat.vert_count() + i] = f(p);
    }
    out
}

// ─── Test 1: P2 hanging-node constraint reproduces a quadratic field ──
//
// Non-uniformly refine a 4×4 grid. Build the P2 constraint matrix.
// Sample u(p) = p.x² − p.y² at the free DoFs, expand u = T · u_free,
// and check that every hanging-DoF row recovers the analytic value to
// machine precision.
//
// The check is sharper than the v2 P1 analogue because the field is
// quadratic — only the curvature term in the P2 constraint can keep the
// error below 1e-9. A naïve "P1-style average" applied to P2 DoFs would
// miss the analytic field by O(h²) on every constrained edge.

#[test]
fn amr_p2_hanging_constraint_preserves_quadratic_field() {
    let base = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
    let mut amr = AmrMesh::new(base);

    let c1 = locate_lower_tri(&amr, 1, 1).expect("(1, 1) lower tri");
    amr.refine_cell(c1);
    amr.propagate_refinement();
    amr.compute_hanging_nodes();
    assert!(!amr.hanging_nodes().is_empty(), "expected hanging nodes");

    let (flat, remap) = amr.flatten_with_remap();
    let t = amr
        .build_constraint_matrix(2, &remap)
        .expect("build P2 constraint matrix");

    let (edges, _edge_id) = compute_edges(&flat);
    let n_vert = flat.vert_count();
    let n_edge = edges.len();
    let n_total = n_vert + n_edge;
    assert_eq!(
        t.nrows(),
        n_total,
        "P2 T has {} rows but flat P2 DoF count is {}",
        t.nrows(),
        n_total
    );

    let n_free = t.ncols();

    // Detect which DoFs are "free" — they're the rows with exactly one
    // nonzero of value 1.0.
    let mut is_free = vec![false; n_total];
    let mut free_to_total: Vec<usize> = vec![usize::MAX; n_free];
    for r in 0..n_total {
        let lo = t.row_offsets[r];
        let hi = t.row_offsets[r + 1];
        if hi - lo == 1 && (t.values[lo] - 1.0).abs() < 1e-15 {
            is_free[r] = true;
            free_to_total[t.col_idx[lo] as usize] = r;
        }
    }
    for (fi, &flat_dof) in free_to_total.iter().enumerate() {
        assert!(
            flat_dof != usize::MAX,
            "free col {fi} did not back-fill from T"
        );
    }

    // Analytic quadratic-but-harmonic field.
    let field = |p: Vec3| p.x * p.x - p.y * p.y;

    // Build u_total by evaluating the analytic field at every P2 DoF
    // position, then extract u_free by reading off the free rows.
    let u_total_analytic = p2_sample_field(&flat, field);

    let mut u_free = vec![0.0; n_free];
    for (fi, &flat_dof) in free_to_total.iter().enumerate() {
        u_free[fi] = u_total_analytic[flat_dof];
    }

    // Expand back through T.
    let mut u_expanded = vec![0.0; n_total];
    t.mv(&u_free, &mut u_expanded).expect("T · u_free");

    // Check every hanging DoF.
    let mut max_err: Real = 0.0;
    let mut checked = 0usize;
    for r in 0..n_total {
        if is_free[r] {
            continue;
        }
        let analytic = u_total_analytic[r];
        let e = (u_expanded[r] - analytic).abs();
        if e > max_err {
            max_err = e;
        }
        checked += 1;
    }
    assert!(checked > 0, "no hanging P2 DoFs checked");
    println!(
        "P2 hanging-node constraint: checked {} hanging DoFs, max error = {}",
        checked, max_err
    );
    assert!(
        max_err < 1e-9,
        "P2 constraint reproduction of u=x²-y² has max error {} > 1e-9",
        max_err
    );

    // Sanity: also reproduce the field on the free DoFs (trivially —
    // the identity rows do not move values).
    for (r, &v) in u_expanded.iter().enumerate() {
        if is_free[r] {
            let analytic = u_total_analytic[r];
            assert!(
                (v - analytic).abs() < 1e-14,
                "free DoF {r} value drifted: got {} expected {}",
                v,
                analytic
            );
        }
    }

    let _ = n_edge; // silence unused warning

    // ─── Reduced-solve check ─────────────────────────────────────────
    //
    // Assemble the P2 Laplacian on the flat mesh, reduce it through T,
    // pin the boundary (which is all free DoFs by 2:1 balance for this
    // interior refinement) to the analytic Dirichlet values
    // `u = x² − y²`, and solve.  Recover u_full = T · u_free and check
    // that every DoF — free *or* hanging — reproduces the analytic
    // field to 1e-9.
    let map = P2DofMap::build(&flat);
    assert_eq!(map.ndof(), n_total);

    let k_full = {
        let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
        for cell in &flat.cells {
            let local = compute_element_p2_stiffness(cell, &flat.vertices, 1.0);
            let nl = local.len();
            for i in 0..nl {
                let gi = map.local_to_global(cell, i);
                for j in 0..nl {
                    let gj = map.local_to_global(cell, j);
                    let v = local[i][j];
                    if v != 0.0 {
                        triplets.push((gi, gj, v));
                    }
                }
            }
        }
        BsrMatrix::from_scalar_triplets(n_total, n_total, &triplets).expect("P2 K assemble")
    };

    // K_c = T^T · K · T, f_c = T^T · 0 = 0 (Laplace).
    let mut k_c = compute_ttkt(&t, &k_full).expect("T^T K T");
    let mut f_c = tt_times_vector(&t, &vec![0.0; n_total]);

    // Walk the P2 DoF positions; pin every boundary one.
    let mut bcs: Vec<BoundaryCondition> = Vec::new();
    let pin_value_at = |p: Vec3| -> Option<Real> {
        let on_bd = p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9;
        if on_bd {
            Some(p.x * p.x - p.y * p.y)
        } else {
            None
        }
    };
    // Vertex DoF positions = flat.vertices[i]; edge DoF positions =
    // midpoint of `edges[i]`.
    for (i, p) in flat.vertices.iter().enumerate() {
        if !is_free[i] {
            continue;
        }
        if let Some(val) = pin_value_at(*p) {
            let fi = free_to_total
                .iter()
                .position(|&total| total == i)
                .expect("free DoF must back-fill");
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: fi as u32,
                value: val,
            });
        }
    }
    for (i, e) in edges.iter().enumerate() {
        let dof = n_vert + i;
        if !is_free[dof] {
            continue;
        }
        let p = (flat.vertices[e[0] as usize] + flat.vertices[e[1] as usize]) * 0.5;
        if let Some(val) = pin_value_at(p) {
            let fi = free_to_total
                .iter()
                .position(|&total| total == dof)
                .expect("free edge DoF must back-fill");
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: fi as u32,
                value: val,
            });
        }
    }

    apply_dirichlet(&mut k_c, &mut f_c, &bcs);

    let mut x0 = vec![0.0; k_c.nrows()];
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x0[*vertex as usize] = *value;
        }
    }
    let result = cg_solve(
        |x| {
            let mut y = vec![0.0; k_c.nrows()];
            k_c.mv(x, &mut y).expect("matvec");
            y
        },
        &f_c,
        &x0,
        1e-14,
        20000,
    );
    assert!(
        result.converged(1e-10),
        "P2 reduced CG did not converge: iters={} residual={}",
        result.iters,
        result.residual
    );

    let u_full = AmrMesh::expand_free_solution(&t, &result.x);
    let mut max_solve_err: Real = 0.0;
    for (r, &v) in u_full.iter().enumerate() {
        let p = if r < n_vert {
            flat.vertices[r]
        } else {
            let e = edges[r - n_vert];
            (flat.vertices[e[0] as usize] + flat.vertices[e[1] as usize]) * 0.5
        };
        let analytic = field(p);
        let e = (v - analytic).abs();
        if e > max_solve_err {
            max_solve_err = e;
        }
    }
    println!(
        "P2 reduced AMR solve: u = x² − y² recovered with max-err = {}",
        max_solve_err
    );
    assert!(
        max_solve_err < 1e-9,
        "P2 AMR reduced solve did not recover the quadratic field: max-err {} > 1e-9",
        max_solve_err
    );
}

// ─── Test 2: DG-AMR flux balance at a T-junction ─────────────────────
//
// Refine a single triangle in a 2-tri mesh. The refined parent edge
// `(a, b)` splits into `(a, m)` and `(m, b)`. For any smooth field on
// the coarse side, the per-edge flux integral over `(a, b)` must
// **equal** the sum over `(a, m)` and `(m, b)` — by the additivity of
// the line integral on the same physical curve.
//
// We don't need a full DG solve to verify this; the geometric flux
// integral is enough. We pick a smooth scalar flux `g(x) = 1 + x + 2 y`
// and compare:
//
//   ∫_{(a,b)} g ds   versus   ∫_{(a,m)} g ds + ∫_{(m,b)} g ds.
//
// The face matches returned by `compute_dg_amr_faces` give us the
// parent/child enumeration directly.

#[test]
#[allow(non_snake_case)]
fn dg_amr_p1_flux_balanced_at_T_junction() {
    // Simple 2-tri mesh of the unit square.
    let mesh = FemMesh {
        vertices: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        cells: vec![FemCell::Tri([0, 1, 2]), FemCell::Tri([0, 2, 3])],
    };

    let mut amr = AmrMesh::new(mesh);
    amr.refine_cell(0); // split the lower tri only — produces 3 hanging-edge children
    amr.compute_hanging_nodes();

    let matches = amr.compute_dg_amr_faces();
    assert!(
        !matches.is_empty(),
        "expected at least one parent→child face match"
    );

    // A 2D quadrature-free closed-form line integral for `g = 1+x+2y`
    // on a straight segment from `p` to `q`:
    //   ∫ g ds = |p-q| · g(½(p+q))                (midpoint rule is
    // exact for linear g on a straight segment).
    let g = |p: Vec3| 1.0 + p.x + 2.0 * p.y;
    let line_int = |p: Vec3, q: Vec3| -> Real {
        let mid = (p + q) * 0.5;
        let len = (q - p).length();
        g(mid) * len
    };

    let vs = amr.vertices();
    let mut checked = 0usize;
    let mut max_diff: Real = 0.0;
    for m in &matches {
        assert_eq!(m.parent_face.len(), 2, "tri mesh → edge faces");
        let pa = vs[m.parent_face[0] as usize];
        let pb = vs[m.parent_face[1] as usize];
        let parent_int = line_int(pa, pb);

        let mut child_sum: Real = 0.0;
        for cf in &m.child_faces {
            assert_eq!(cf.len(), 2, "tri mesh → child edges");
            let ca = vs[cf[0] as usize];
            let cb = vs[cf[1] as usize];
            child_sum += line_int(ca, cb);
        }
        let diff = (parent_int - child_sum).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        checked += 1;
    }
    assert!(checked > 0, "no parent→child face matches inspected");
    println!(
        "DG-AMR face balance: checked {} parent faces, max |parent − Σchild| = {}",
        checked, max_diff
    );
    assert!(
        max_diff < 1e-12,
        "DG-AMR flux-balance integral mismatch {} > 1e-12",
        max_diff
    );
}

// ─── Test 3: state interpolation preserves a linear field ────────────
//
// Build a coarse AMR mesh, sample u(p) = p.x + p.y at its DoFs, refine
// to a new AMR mesh, transfer the state, and check that the field is
// recovered to machine precision at every new DoF.

#[test]
fn amr_state_transfer_preserves_linear_field() {
    // Coarse AMR mesh.
    let base = FemMesh::tri_grid_2d(3, 3, 1.0, 1.0);
    let amr_coarse = AmrMesh::new(base.clone());
    let (coarse_flat, _) = amr_coarse.flatten_with_remap();

    // Coarse field: u(p) = p.x + p.y, sampled at every coarse flat
    // vertex.
    let field = |p: Vec3| p.x + p.y;
    let u_old: Vec<Real> = coarse_flat.vertices.iter().map(|p| field(*p)).collect();

    // Refined AMR mesh — same `base_mesh` as `amr_coarse` so the
    // geometric support of the old leaves still contains every new
    // DoF.
    let mut amr_fine = AmrMesh::new(base);
    let c1 = locate_lower_tri(&amr_fine, 1, 1).expect("(1, 1) lower tri");
    amr_fine.refine_cell(c1);
    amr_fine.propagate_refinement();
    amr_fine.compute_hanging_nodes();

    let u_new = amr_fine.transfer_state(&u_old, &amr_coarse);
    let (fine_flat, _) = amr_fine.flatten_with_remap();
    assert_eq!(u_new.len(), fine_flat.vert_count());

    let mut max_err: Real = 0.0;
    for (p, &v) in fine_flat.vertices.iter().zip(u_new.iter()) {
        let analytic = field(*p);
        let e = (v - analytic).abs();
        if e > max_err {
            max_err = e;
        }
    }
    println!(
        "AMR state transfer (linear): {} new DoFs, max error = {}",
        u_new.len(),
        max_err
    );
    assert!(
        max_err <= 1e-12,
        "linear field transfer error {} > 1e-12",
        max_err
    );

    // Sanity: the hanging DoF *values* should also linearly average
    // (since P1 reproduces affine functions).
    let (_, fine_remap) = amr_fine.flatten_with_remap();
    for hn in amr_fine.hanging_nodes() {
        let Some(&flat_h) = fine_remap.get(&hn.dof) else {
            continue;
        };
        let Some(&fp1) = fine_remap.get(&hn.parents[0]) else {
            continue;
        };
        let Some(&fp2) = fine_remap.get(&hn.parents[1]) else {
            continue;
        };
        let avg = 0.5 * (u_new[fp1 as usize] + u_new[fp2 as usize]);
        let got = u_new[flat_h as usize];
        assert!(
            (got - avg).abs() < 1e-12,
            "hanging-DoF interpolation drift: got {} avg-of-parents {}",
            got,
            avg
        );
    }
}
