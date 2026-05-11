//! Informational microbench for `assemble_global` — heat-equation
//! Laplacian on `n³` tet cubes. Ignored by default; run with:
//!
//! ```text
//! cargo test -p molla --test fem_assembly_perf --release -- --ignored --nocapture
//! ```
//!
//! Prints wall time per mesh size so we can see the win from cutting
//! the per-triplet `vec![val]` allocation in `assemble_global` (now
//! routed through `BsrMatrix::from_scalar_triplets`).

use molla::fem::assembly::assemble_stiffness;
use molla::fem::function_space::FunctionSpace;
use molla::fem::mesh::FemMesh;

fn bench_one(n: usize) {
    let mesh = FemMesh::tet_grid_3d(n, n, n, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    // Warm up — first call pays the visitor JIT/branch-predict cost.
    let _ = assemble_stiffness(&mesh, space, 1.0).unwrap();
    let t0 = std::time::Instant::now();
    let k = assemble_stiffness(&mesh, space, 1.0).unwrap();
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "assemble_stiffness {n}^3 tet cube: cells={} ndof={} nnz={} time={:.2} ms",
        mesh.cells.len(),
        space.ndof(&mesh),
        k.values.len(),
        dt_ms,
    );
}

#[test]
#[ignore = "perf bench — run with --ignored --nocapture"]
fn bench_assemble_global_heat_laplacian() {
    bench_one(16);
    bench_one(32);
}
