//! Stage 45 v5 — DG-AMR coarsening helpers.
//!
//! Covers:
//! 1. recursive subtree coarsening (deep AMR trees can be collapsed in
//!    one call);
//! 2. DG-P1 child-to-parent L² projection for coarsening.

use molla_fem::{AmrMesh, FemCell, FemMesh, project_dg_p1_tri_children_to_parent_l2};
use molla_math::{Real, Vec3};

#[test]
fn amr_coarsen_subtree_collapses_deep_tree() {
    let base = FemMesh {
        vertices: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        cells: vec![FemCell::Tri([0, 1, 2])],
    };
    let mut amr = AmrMesh::new(base.clone());
    amr.refine_cell(0);
    let child0 = amr.refined_cells[0].children.as_slice()[0] as usize;
    amr.refine_cell(child0);
    assert!(amr.leaf_count() > 1, "setup should have a deep AMR tree");

    amr.coarsen_subtree(0).expect("recursive coarsen");
    amr.compute_hanging_nodes();

    assert_eq!(amr.leaf_count(), 1);
    assert!(amr.refined_cells.is_empty());
    let flat = amr.flatten();
    assert_eq!(flat.cells, base.cells);
    assert_eq!(flat.vertices, base.vertices);
    assert!(
        amr.hanging_nodes().is_empty(),
        "fully coarsened single-triangle tree should have no hanging nodes"
    );
}

#[test]
fn dg_p1_l2_projection_from_children_recovers_linear_parent() {
    let base = FemMesh {
        vertices: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        cells: vec![FemCell::Tri([0, 1, 2])],
    };
    let mut amr = AmrMesh::new(base);
    amr.refine_cell(0);

    let parent = amr.cells()[0];
    let child_ids = amr.refined_cells[0].children.as_slice();
    let children: Vec<FemCell> = child_ids
        .iter()
        .map(|&id| amr.cells()[id as usize])
        .collect();
    let field = |p: Vec3| -> Real { 2.0 + 3.0 * p.x - 5.0 * p.y };
    let child_values: Vec<[Real; 3]> = children
        .iter()
        .map(|cell| {
            let FemCell::Tri(v) = cell else {
                unreachable!()
            };
            [
                field(amr.vertices()[v[0] as usize]),
                field(amr.vertices()[v[1] as usize]),
                field(amr.vertices()[v[2] as usize]),
            ]
        })
        .collect();

    let projected =
        project_dg_p1_tri_children_to_parent_l2(&parent, &children, amr.vertices(), &child_values)
            .expect("L2 projection");
    let FemCell::Tri(v) = parent else {
        unreachable!()
    };
    let expected = [
        field(amr.vertices()[v[0] as usize]),
        field(amr.vertices()[v[1] as usize]),
        field(amr.vertices()[v[2] as usize]),
    ];
    for i in 0..3 {
        assert!(
            (projected[i] - expected[i]).abs() < 1.0e-12,
            "parent node {i}: projected {}, expected {}",
            projected[i],
            expected[i]
        );
    }
}
