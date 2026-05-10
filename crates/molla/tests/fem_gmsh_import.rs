//! Stage 36.6 — Gmsh MSH 4.1 import contract tests.
//!
//! These exercises pin down the [`molla::fem::FemMesh::from_gmsh_*`]
//! constructors against hand-crafted MSH 4.1 ASCII fixtures stored in
//! `tests/fixtures/gmsh/`. The tests cover:
//!
//! 1. Minimal 2D mesh (one triangle).
//! 2. Minimal 3D mesh (one tetrahedron).
//! 3. Round-trip parse + Laplacian assembly on a 6-tet unit cube;
//!    counts match the procedural [`FemMesh::tet_grid_3d(1,1,1, 1, 1, 1)`].
//! 4. Skip-with-warn behaviour for unsupported element kinds (lines).
//!
//! Tests are CPU-only and run as part of the standard `cargo test
//! -p molla` suite — no `#[ignore]`.

use molla::fem::{
    assemble_global, compute_element_stiffness, FemCell, FemMesh, FunctionSpace,
};
use molla::math::sparse::BsrMatrix;
use molla::math::Real;

// ─── Fixtures ──────────────────────────────────────────────────────
//
// Inline via `include_str!` so the tests stay self-contained and the
// fixture files live next to the parser consumers under `tests/fixtures/`.

const FIXTURE_TRIANGLE: &str =
    include_str!("fixtures/gmsh/single_triangle.msh");
const FIXTURE_TET: &str = include_str!("fixtures/gmsh/single_tet.msh");
const FIXTURE_CUBE_KUHN: &str =
    include_str!("fixtures/gmsh/unit_cube_kuhn.msh");
const FIXTURE_LINE_PLUS_TRI: &str =
    include_str!("fixtures/gmsh/line_plus_triangle.msh");

// Small helper — assemble the unit-α Laplacian for a mesh.
fn assemble_stiffness(mesh: &FemMesh) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, 1.0)
    })
    .expect("assemble stiffness")
}

// ─── Test 1: minimal 2D MSH (single triangle) ──────────────────────
//
// Verifies vertex + cell counts and that the single cell parses as a
// `FemCell::Tri` with the expected node indices (0-based after the
// MSH 1-based tag → flat-index conversion).

#[test]
fn parse_minimal_gmsh_2d_triangle_msh_string() {
    let mesh = FemMesh::from_gmsh_str(FIXTURE_TRIANGLE)
        .expect("parse minimal triangle MSH");
    assert_eq!(mesh.vert_count(), 3, "triangle MSH should have 3 vertices");
    assert_eq!(mesh.cell_count(), 1, "triangle MSH should have 1 cell");
    match mesh.cells[0] {
        FemCell::Tri(v) => {
            assert_eq!(v, [0, 1, 2], "Tri indices remap to 0..3");
        }
        FemCell::Tet(_) => panic!("expected Tri, got Tet"),
    }
    // Vertex positions match the MSH coords.
    assert!((mesh.vertices[0].x - 0.0).abs() < 1e-12);
    assert!((mesh.vertices[1].x - 1.0).abs() < 1e-12);
    assert!((mesh.vertices[2].y - 1.0).abs() < 1e-12);
}

// ─── Test 2: minimal 3D MSH (single tet) ───────────────────────────

#[test]
fn parse_minimal_gmsh_3d_tet_msh_string() {
    let mesh = FemMesh::from_gmsh_str(FIXTURE_TET).expect("parse minimal tet MSH");
    assert_eq!(mesh.vert_count(), 4, "tet MSH should have 4 vertices");
    assert_eq!(mesh.cell_count(), 1, "tet MSH should have 1 cell");
    match mesh.cells[0] {
        FemCell::Tet(v) => {
            assert_eq!(v, [0, 1, 2, 3], "Tet indices remap to 0..4");
        }
        FemCell::Tri(_) => panic!("expected Tet, got Tri"),
    }
    assert!((mesh.vertices[3].z - 1.0).abs() < 1e-12);
}

// ─── Test 3: unit cube assembly matches procedural ─────────────────
//
// `unit_cube_kuhn.msh` was hand-written to exactly mirror
// `FemMesh::tet_grid_3d(1, 1, 1, 1.0, 1.0, 1.0)`: same 8 corner
// vertices, same Kuhn six-tet decomposition along the v000 → v111
// body diagonal, all 1-indexed in MSH then remapped to 0-based here.
//
// We assemble the Laplacian on both meshes and compare row-sum
// invariants + a per-row dot product that is invariant under the
// vertex ordering. (The two meshes happen to have the same vertex
// ordering by construction, but we don't rely on that — the
// per-vertex `K · 1 == 0` check is a topology-independent partition-
// of-unity guarantee.)

#[test]
fn gmsh_unit_cube_assembly_matches_procedural() {
    let parsed = FemMesh::from_gmsh_str(FIXTURE_CUBE_KUHN)
        .expect("parse unit-cube MSH");
    let procedural = FemMesh::tet_grid_3d(1, 1, 1, 1.0, 1.0, 1.0);

    // Vertex / cell counts agree.
    assert_eq!(parsed.vert_count(), procedural.vert_count());
    assert_eq!(parsed.cell_count(), procedural.cell_count());

    // The fixture was authored to mirror the procedural numbering,
    // so vertex coordinates should also agree element-wise.
    for (a, b) in parsed.vertices.iter().zip(procedural.vertices.iter()) {
        assert!((a - b).length() < 1e-12, "vertex mismatch: {:?} vs {:?}", a, b);
    }

    // Same cell shapes + same cell connectivity.
    for (a, b) in parsed.cells.iter().zip(procedural.cells.iter()) {
        match (a, b) {
            (FemCell::Tet(va), FemCell::Tet(vb)) => assert_eq!(va, vb),
            (FemCell::Tri(va), FemCell::Tri(vb)) => assert_eq!(va, vb),
            _ => panic!("cell kind mismatch: {:?} vs {:?}", a, b),
        }
    }

    // Assemble the Laplacian on both and confirm:
    // - matrix size matches
    // - K · 1 ≡ 0 (partition of unity) on the parsed result
    // - K_parsed and K_procedural produce the same matvec on a
    //   non-trivial probe vector — sufficient evidence that the
    //   assembled operators are pointwise equal even though we
    //   don't expose the raw CSR for a direct entry-by-entry
    //   comparison.
    let k_parsed = assemble_stiffness(&parsed);
    let k_proc = assemble_stiffness(&procedural);
    assert_eq!(k_parsed.nrows(), k_proc.nrows());
    assert_eq!(k_parsed.ncols(), k_proc.ncols());

    let n = k_parsed.nrows();
    let one = vec![1.0_f64; n];
    let mut y = vec![0.0_f64; n];
    k_parsed.mv(&one, &mut y).unwrap();
    for (i, &v) in y.iter().enumerate() {
        assert!(v.abs() < 1e-12, "K_parsed · 1 row {i} = {v}");
    }

    // Probe vector: linear field x − 2y + 0.5z. Two stiffness
    // operators built from identical meshes must agree on every
    // probe; we use a non-zero one so the test is meaningful.
    let probe: Vec<Real> = parsed
        .vertices
        .iter()
        .map(|p| p.x - 2.0 * p.y + 0.5 * p.z)
        .collect();
    let mut yp = vec![0.0; n];
    let mut yq = vec![0.0; n];
    k_parsed.mv(&probe, &mut yp).unwrap();
    k_proc.mv(&probe, &mut yq).unwrap();
    for i in 0..n {
        assert!(
            (yp[i] - yq[i]).abs() < 1e-12,
            "K parsed vs procedural disagree at row {i}: {} vs {}",
            yp[i],
            yq[i]
        );
    }
}

// ─── Test 4: unsupported element kinds drop with a warn ────────────
//
// `line_plus_triangle.msh` contains one Lin2 (kind 1) followed by one
// Tri3 (kind 2). The triangle should survive, the line should be
// skipped — we don't assert on the warn output here (tests don't
// install a tracing subscriber) but we do check that the surviving
// cell is the triangle.

#[test]
fn gmsh_skip_unsupported_element_kinds() {
    let mesh = FemMesh::from_gmsh_str(FIXTURE_LINE_PLUS_TRI)
        .expect("parse mixed line + triangle MSH");
    assert_eq!(mesh.vert_count(), 3, "vertices preserved across skipping");
    assert_eq!(
        mesh.cell_count(),
        1,
        "only the Tri3 element should survive (Lin2 is dropped)"
    );
    match mesh.cells[0] {
        FemCell::Tri(v) => {
            assert_eq!(v, [0, 1, 2], "the Tri3 references nodes 1..3 (→ 0..2)");
        }
        FemCell::Tet(_) => panic!("expected Tri after skipping the line"),
    }
}

// ─── Bonus: from_gmsh_path round-trip on a fixture file ────────────
//
// Exercises the file-reading code path so we don't ship a constructor
// users can't actually invoke.

#[test]
fn gmsh_from_path_reads_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gmsh/single_tet.msh");
    let mesh = FemMesh::from_gmsh_path(&path).expect("from_gmsh_path");
    assert_eq!(mesh.vert_count(), 4);
    assert_eq!(mesh.cell_count(), 1);
}

// ─── Bonus: from_gmsh_bytes accepts the binary entry point ─────────
//
// Sanity check that the byte-slice constructor works by feeding it
// the same ASCII bytes the string constructor accepts. Binary MSH is
// produced by `gmsh -bin`; we don't ship a binary fixture in the
// repo to keep the diff text-only, but the parser path is identical
// (mshio dispatches on the header).

#[test]
fn gmsh_from_bytes_accepts_ascii() {
    let mesh = FemMesh::from_gmsh_bytes(FIXTURE_TET.as_bytes())
        .expect("from_gmsh_bytes accepts ASCII MSH");
    assert_eq!(mesh.cell_count(), 1);
}
