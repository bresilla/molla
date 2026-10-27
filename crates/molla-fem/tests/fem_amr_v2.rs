//! Stage 45 v2 — hanging-node constraints, 2:1 balance, and the
//! reduced-system `K_c = TᵀKT, f_c = Tᵀf` solve path on a non-
//! uniformly refined mesh.

use molla_fem::{
    AmrMesh, BoundaryCondition, FemCell, FemMesh, FunctionSpace, apply_dirichlet,
    assemble_global_amr, compute_element_stiffness, tt_times_vector,
};
use molla_math::iterative::cg_solve;
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// ─── Test 1: hanging-node detection on a single T-junction ──────────
//
// Take a 2×2 quad mesh (= 8 triangles). Refine *only* triangle 0 (the
// bottom-left lower tri `[idx(0,0), idx(1,0), idx(1,1)]` = `[0, 1, 4]`).
// Its 3 edges acquire midpoints:
//   - edge (0, 1) — bottom mesh boundary. No neighbour → midpoint NOT
//     hanging.
//   - edge (1, 4) — shared with tri 3 `[1, 5, 4]`. Midpoint IS hanging.
//   - edge (0, 4) — shared with tri 1 `[0, 4, 3]`. Midpoint IS hanging.
// Expected hanging-node count: 2.

#[test]
#[allow(non_snake_case)]
fn hanging_node_detection_single_T_junction() {
    let base = FemMesh::tri_grid_2d(2, 2, 1.0, 1.0);
    assert_eq!(base.cells.len(), 8);

    let mut amr = AmrMesh::new(base);
    amr.refine_where(
        |cell: &FemCell, _verts: &[Vec3]| matches!(cell, FemCell::Tri(v) if v == &[0u32, 1, 4]),
    );

    assert_eq!(
        amr.refined_cells.len(),
        1,
        "expected exactly one refinement"
    );

    let hn = amr.hanging_nodes();
    assert_eq!(
        hn.len(),
        2,
        "expected exactly 2 hanging nodes after refining tri 0 of a 2×2 grid, got {} \
        (records: {:?})",
        hn.len(),
        hn
    );

    let mut seen_edges: Vec<(u32, u32)> = hn
        .iter()
        .map(|h| {
            let a = h.parents[0];
            let b = h.parents[1];
            if a < b { (a, b) } else { (b, a) }
        })
        .collect();
    seen_edges.sort();
    let expected: Vec<(u32, u32)> = vec![(0, 4), (1, 4)];
    assert_eq!(seen_edges, expected, "unexpected hanging-edge endpoints");
}

// ─── Test 2: 2:1 balance refines a coarser neighbour ────────────────
//
// Refine one cell twice (level 0 → level 1 via a single split, then
// one of the children to level 2). Without propagation, the level-0
// neighbours share an edge whose deepest split-depth is 2 — a 2:1
// balance violation. After `propagate_refinement`, no leaf should
// have an edge of split-depth > 1.

#[test]
fn two_one_balance_refines_neighbour() {
    let base = FemMesh::tri_grid_2d(3, 3, 1.0, 1.0);
    let mut amr = AmrMesh::new(base);

    let centre_lower = locate_lower_tri(&amr, 1, 1).expect("centre quad lower tri");
    amr.refine_cell(centre_lower);

    let child0 = amr.refined_cells[0].children.as_slice()[0];
    amr.refine_cell(child0 as usize);

    let n_before = amr.refined_cells.len();
    amr.propagate_refinement();
    amr.compute_hanging_nodes();

    let max_depth = max_leaf_edge_split_depth(&amr);
    assert!(
        max_depth <= 1,
        "2:1 balance violated — found leaf edge with split depth {}",
        max_depth
    );

    assert!(
        amr.refined_cells.len() > n_before,
        "expected propagation to trigger extra refinements, got only {} \
        (started with {})",
        amr.refined_cells.len(),
        n_before
    );
}

// ─── Test 3: T-matrix averages hanging DoFs from a linear field ─────
//
// Build T for a non-uniform mesh. Sample a known linear field at the
// free DoFs, expand u = T · u_free, and check `u_h = ½(u_a + u_b)`
// for every hanging DoF. The linear field's midpoint exactly equals
// the analytic value at the midpoint — that gives a closed-form check
// on every constrained row.

#[test]
#[allow(non_snake_case)]
fn T_matrix_averages_hanging_dof() {
    let base = FemMesh::tri_grid_2d(3, 3, 1.0, 1.0);
    let mut amr = AmrMesh::new(base);
    let centre_lower = locate_lower_tri(&amr, 1, 1).expect("centre quad lower tri");
    amr.refine_cell(centre_lower);
    amr.propagate_refinement();
    amr.compute_hanging_nodes();
    assert!(!amr.hanging_nodes().is_empty(), "expected hanging nodes");

    let (flat, remap) = amr.flatten_with_remap();
    let t = amr.build_constraint_matrix(1, &remap).expect("build T");
    assert_eq!(t.nrows(), flat.vert_count());

    let n_total = flat.vert_count();
    let n_free = t.ncols();

    let mut is_hanging = vec![false; n_total];
    for r in 0..n_total {
        let nnz = t.row_offsets[r + 1] - t.row_offsets[r];
        is_hanging[r] = nnz > 1;
    }

    let mut free_to_flat: Vec<usize> = vec![usize::MAX; n_free];
    for r in 0..n_total {
        if !is_hanging[r] {
            let slot = t.row_offsets[r];
            let col = t.col_idx[slot];
            free_to_flat[col] = r;
        }
    }
    for (fi, &flat_id) in free_to_flat.iter().enumerate() {
        assert!(
            flat_id != usize::MAX,
            "free col {fi} did not back-fill from T"
        );
    }

    let field = |p: Vec3| 2.0 * p.x + 3.0 * p.y;
    let mut u_free = vec![0.0; n_free];
    for (fi, &flat_id) in free_to_flat.iter().enumerate() {
        u_free[fi] = field(flat.vertices[flat_id]);
    }

    let mut u = vec![0.0; n_total];
    t.mv(&u_free, &mut u).expect("T · u_free");

    let mut checked = 0usize;
    for hn in amr.hanging_nodes() {
        let Some(&flat_h) = remap.get(&hn.dof) else {
            continue;
        };
        let Some(&fp1) = remap.get(&hn.parents[0]) else {
            continue;
        };
        let Some(&fp2) = remap.get(&hn.parents[1]) else {
            continue;
        };
        let expected = 0.5 * (u[fp1 as usize] + u[fp2 as usize]);
        let diff = (u[flat_h as usize] - expected).abs();
        assert!(
            diff < 1e-12,
            "hanging DoF flat={} value {} != ½(u_{} + u_{}) = {} (diff {})",
            flat_h,
            u[flat_h as usize],
            fp1,
            fp2,
            expected,
            diff
        );

        let p_h = flat.vertices[flat_h as usize];
        let expected_analytic = field(p_h);
        assert!(
            (u[flat_h as usize] - expected_analytic).abs() < 1e-12,
            "hanging DoF flat={} value {} != analytic linear field {}",
            flat_h,
            u[flat_h as usize],
            expected_analytic
        );

        checked += 1;
    }
    assert!(checked > 0, "no hanging nodes were checked");
}

// ─── Test 4: reduced solve of the harmonic Dirichlet problem ────────
//
// Non-uniformly refine a 4×4 base mesh. Pin Dirichlet `u = x − y` on
// the geometric boundary; the linear field is harmonic so P1 must
// recover it exactly. Build (K, f) on the flat mesh, then
// `K_c = TᵀKT, f_c = Tᵀf`, apply Dirichlet on the reduced system,
// solve via CG, and check `u_full = T · u_free` agrees with the
// analytic field to 1e-10.

#[test]
fn reduced_solve_harmonic_recovers_linear() {
    let base = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
    let mut amr = AmrMesh::new(base);

    let c1 = locate_lower_tri(&amr, 1, 1).expect("(1, 1) lower tri");
    let c2 = locate_lower_tri(&amr, 2, 1).expect("(2, 1) lower tri");
    amr.refine_cell(c1);
    amr.refine_cell(c2);
    amr.propagate_refinement();
    amr.compute_hanging_nodes();
    assert!(!amr.hanging_nodes().is_empty(), "expected hanging nodes");

    let (flat, remap) = amr.flatten_with_remap();
    let (mut k_c, mut f_c, t) = assemble_global_amr(
        &amr,
        FunctionSpace::P1Nodal,
        |cell, verts| compute_element_stiffness(cell, verts, 1.0),
        None,
    )
    .expect("assemble_global_amr");

    let n_total = flat.vert_count();
    let mut is_hanging = vec![false; n_total];
    let mut free_index_of: Vec<i32> = vec![-1; n_total];
    for r in 0..n_total {
        let lo = t.row_offsets[r];
        let hi = t.row_offsets[r + 1];
        let nnz = hi - lo;
        if nnz > 1 {
            is_hanging[r] = true;
        } else {
            free_index_of[r] = t.col_idx[lo] as i32;
        }
    }

    let mut bcs: Vec<BoundaryCondition> = Vec::new();
    for (i, p) in flat.vertices.iter().enumerate() {
        let on_bd = p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9;
        if !on_bd {
            continue;
        }
        assert!(
            !is_hanging[i],
            "boundary DoF {i} at ({}, {}) is hanging — \
             reduced-solve test expected boundary to be non-hanging",
            p.x, p.y
        );
        let fi = free_index_of[i];
        assert!(fi >= 0, "boundary DoF {i} has no free index");
        bcs.push(BoundaryCondition::Dirichlet {
            vertex: fi as u32,
            value: p.x - p.y,
        });
    }

    apply_dirichlet(&mut k_c, &mut f_c, &bcs);

    let mut x0 = vec![0.0; k_c.nrows()];
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x0[*vertex as usize] = *value;
        }
    }
    let tol: Real = 1e-14;
    let result = cg_solve(|x| matvec(&k_c, x), &f_c, &x0, tol, 5000);
    assert!(
        result.converged(1e-10),
        "reduced CG did not converge: iters={} residual={}",
        result.iters,
        result.residual
    );
    let u_free = result.x;

    let u_full = AmrMesh::expand_free_solution(&t, &u_free);

    let mut max_err: Real = 0.0;
    for (p, &v) in flat.vertices.iter().zip(u_full.iter()) {
        let analytic = p.x - p.y;
        let e = (v - analytic).abs();
        if e > max_err {
            max_err = e;
        }
    }
    println!("reduced AMR solve: harmonic-recovery max-err = {max_err}");
    assert!(
        max_err < 1e-10,
        "reduced AMR solve recovered harmonic field to max-err = {max_err} > 1e-10"
    );

    for hn in amr.hanging_nodes() {
        let flat_h = remap[&hn.dof] as usize;
        let fp1 = remap[&hn.parents[0]] as usize;
        let fp2 = remap[&hn.parents[1]] as usize;
        let diff = (u_full[flat_h] - 0.5 * (u_full[fp1] + u_full[fp2])).abs();
        assert!(
            diff < 1e-12,
            "hanging-node constraint slack — flat={} u_h={} avg={} diff={}",
            flat_h,
            u_full[flat_h],
            0.5 * (u_full[fp1] + u_full[fp2]),
            diff
        );
    }

    let _ = tt_times_vector; // confirm the public symbol is exported
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Find the AMR cell id of the lower triangle of quad `(i, j)` in a
/// `tri_grid_2d(nx, ny)` base mesh by pattern-matching its vertex
/// tuple `[v00, v10, v11]`. Assumes a square grid (`nx == ny`).
fn locate_lower_tri(amr: &AmrMesh, i: usize, j: usize) -> Option<usize> {
    let n_verts = amr.base_mesh.vert_count();
    let stride = (n_verts as f64).sqrt() as usize;
    let v00 = (j * stride + i) as u32;
    let v10 = (j * stride + i + 1) as u32;
    let v11 = ((j + 1) * stride + i + 1) as u32;
    for (idx, c) in amr.cells().iter().enumerate() {
        if let FemCell::Tri(v) = c
            && v == &[v00, v10, v11]
        {
            return Some(idx);
        }
    }
    None
}

/// Walk every leaf edge and return the maximum observed split depth.
/// Re-implemented against the public surface so the test is
/// independent of internals.
fn max_leaf_edge_split_depth(amr: &AmrMesh) -> usize {
    let vs = amr.vertices();
    let n = vs.len();

    let mut by_midpoint: std::collections::HashMap<u32, (u32, u32)> =
        std::collections::HashMap::new();
    for m in 0..n {
        let mut found = false;
        for a in 0..n {
            if found {
                break;
            }
            for b in (a + 1)..n {
                if m == a || m == b {
                    continue;
                }
                let mid = (vs[a] + vs[b]) * 0.5;
                if (vs[m] - mid).length() < 1e-9 {
                    by_midpoint.insert(m as u32, (a as u32, b as u32));
                    found = true;
                    break;
                }
            }
        }
    }

    fn depth(a: u32, b: u32, midpoints: &std::collections::HashMap<u32, (u32, u32)>) -> usize {
        for (&m, &(p, q)) in midpoints {
            let on_edge = (p == a && q == b) || (p == b && q == a);
            if on_edge {
                return 1 + depth(a, m, midpoints).max(depth(m, b, midpoints));
            }
        }
        0
    }

    let mut max_d = 0usize;
    for (idx, cell) in amr.cells().iter().enumerate() {
        if !amr.is_leaf(idx) {
            continue;
        }
        let verts = cell.verts();
        let nv = verts.len();
        for i in 0..nv {
            let a = verts[i];
            let b = verts[(i + 1) % nv];
            let d = depth(a, b, &by_midpoint);
            if d > max_d {
                max_d = d
            }
        }
    }
    max_d
}

fn matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    let mut y = vec![0.0; a.nrows()];
    a.mv(x, &mut y).expect("matvec");
    y
}
