//! Stage 36 v1 — FEM framework contract tests.
//!
//! Each test maps 1:1 to one of the five `Tests` items in PLAN.md
//! Stage 36. They are all CPU-only and run as part of the standard
//! `cargo test -p molla-fem` suite.

use molla_fem::{
    BoundaryCondition, FemCell, FemMesh, FunctionSpace, apply_dirichlet, assemble_global,
    compute_element_mass, compute_element_stiffness, solve_steady_state, solve_transient,
};
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// Helper — wraps `assemble_global` for the typical mass-matrix call.
fn assemble_mass(mesh: &FemMesh) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, compute_element_mass).expect("assemble mass")
}

fn assemble_stiffness(mesh: &FemMesh, alpha: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .expect("assemble stiffness")
}

fn matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    let mut y = vec![0.0; a.nrows()];
    a.mv(x, &mut y).unwrap();
    y
}

// ─── Test 1: P1 triangle mass-matrix on the unit triangle ───────────
//
// Analytic check: for a triangle with area `A`, the consistent mass
// matrix is `(A / 12) · [[2,1,1],[1,2,1],[1,1,2]]`. We assemble a
// single-cell mesh and read the matrix back.

#[test]
fn p1_tri_mass_matrix_unit_triangle() {
    // Unit triangle, area = 0.5 ⇒ expected scale = 0.5 / 12.
    let mesh = FemMesh {
        vertices: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        cells: vec![FemCell::Tri([0, 1, 2])],
    };
    let m = assemble_mass(&mesh);
    let expected = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]];
    let scale = 0.5 / 12.0;
    // Matvec each unit basis vector and read off the column.
    for j in 0..3 {
        let mut e = vec![0.0; 3];
        e[j] = 1.0;
        let col = matvec(&m, &e);
        for i in 0..3 {
            let want = scale * expected[i][j];
            assert!(
                (col[i] - want).abs() < 1e-12,
                "M[{i},{j}] = {} expected {}",
                col[i],
                want
            );
        }
    }
}

// ─── Test 2: P1 tet stiffness applied to a linear field returns 0 ──
//
// ∇²(linear) = 0 ⇒ K · u_lin = 0 (up to round-off). Exercises both
// the per-element kernel + global assembly + the row-sum identity
// `∑ⱼ Kᵢⱼ = 0` (constant-field case).

#[test]
fn p1_tet_stiffness_constant_field() {
    let mesh = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let k = assemble_stiffness(&mesh, 1.0);

    // Linear field u(x, y, z) = 2x − 0.5y + 0.7z + 1.3.
    let u: Vec<Real> = mesh
        .vertices
        .iter()
        .map(|p| 2.0 * p.x - 0.5 * p.y + 0.7 * p.z + 1.3)
        .collect();

    // K · u over the FULL stiffness matrix has nonzero entries on
    // boundary rows (those rows are not in the kernel of the global
    // operator without a BC because boundary-element gradients are
    // not balanced by neighbouring cells). Interior rows must be
    // exactly zero.
    let ku = matvec(&k, &u);

    // Mark which vertices are on the boundary (any face of the cube).
    let on_boundary: Vec<bool> = mesh
        .vertices
        .iter()
        .map(|p| {
            p.x.abs() < 1e-9
                || (p.x - 1.0).abs() < 1e-9
                || p.y.abs() < 1e-9
                || (p.y - 1.0).abs() < 1e-9
                || p.z.abs() < 1e-9
                || (p.z - 1.0).abs() < 1e-9
        })
        .collect();

    let n = space.ndof(&mesh);
    for i in 0..n {
        if !on_boundary[i] {
            assert!(
                ku[i].abs() < 1e-10,
                "interior vertex {i} has non-zero K·u_lin = {}",
                ku[i]
            );
        }
    }

    // Constant-field check: ∑ⱼ Kᵢⱼ ≡ 0 for ALL rows (interior +
    // boundary). This proves the assembled global matrix has the
    // partition-of-unity property, which is the strongest "linear
    // field in the kernel" statement we can make without a BC.
    let one = vec![1.0; n];
    let kone = matvec(&k, &one);
    for (i, &v) in kone.iter().enumerate() {
        assert!(v.abs() < 1e-10, "K · 1 row {i} = {v}");
    }
}

// ─── Test 3: 2D Laplacian Dirichlet solve, harmonic boundary data ──
//
// Choose a known harmonic function on `[0, 1]²`, e.g. u(x, y) = x − y.
// Apply Dirichlet on the entire boundary using the analytic values;
// the FEM solution should equal the analytic field everywhere
// (harmonic interpolation on a P1 mesh recovers a linear function
// exactly).

#[test]
fn assemble_laplacian_2d_unit_square_solves_to_analytic() {
    let nx = 8;
    let ny = 8;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let k = assemble_stiffness(&mesh, 1.0);
    let n = space.ndof(&mesh);
    let f = vec![0.0; n];

    // Harmonic boundary data: u(x, y) = x − y.
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

    let u = solve_steady_state(&k, &f, &bcs, 1000, 1e-10);
    let mut max_err: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        let want = p.x - p.y;
        max_err = max_err.max((u[i] - want).abs());
    }
    assert!(max_err < 1e-3, "max error {max_err} > 1e-3");
}

// ─── Test 4: 3D heat-equation steady state on a unit cube ──────────
//
// Dirichlet u(z=0) = 0, u(z=1) = 1, side faces *insulating* (zero
// Neumann is the natural homogeneous case). Steady state ⇒ T(z) = z.
//
// Tolerance per spec: 1e-2 at 16³ grid resolution.

#[test]
fn heat_steady_state_3d_cube_matches_analytic() {
    let n_grid = 16usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let k = assemble_stiffness(&mesh, 1.0);
    let n = space.ndof(&mesh);
    let f = vec![0.0; n];

    // Dirichlet only on top + bottom; sides naturally insulating
    // because the v1 weak form drops the boundary integral when no
    // Neumann is specified.
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.z.abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        } else if (p.z - 1.0).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 1.0,
            });
        }
    }

    let u = solve_steady_state(&k, &f, &bcs, 5000, 1e-9);
    let mut max_err: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        max_err = max_err.max((u[i] - p.z).abs());
    }
    assert!(max_err < 1e-2, "max error {max_err} > 1e-2");
}

// ─── Test 5: backward-Euler converges to the steady state ──────────
//
// Same problem as Test 4 but at lower resolution (cheaper). Step
// 200 backward-Euler steps at dt = 0.01 from u₀ ≡ 0 and check the
// last state matches T(z) = z within a moderate tolerance.

#[test]
fn time_stepping_converges_to_steady_state() {
    let n_grid = 8usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let alpha = 5.0; // crank up diffusivity so 200 × 0.01 reaches steady.
    let m = assemble_global(&mesh, space, compute_element_mass).unwrap();
    let k = assemble_stiffness(&mesh, alpha);
    let n = space.ndof(&mesh);

    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.z.abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        } else if (p.z - 1.0).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 1.0,
            });
        }
    }

    let f = vec![0.0; n];
    let mut u = vec![0.0; n];
    // Seed Dirichlet entries so the first transient step starts
    // from a consistent BC.
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            u[*vertex as usize] = *value;
        }
    }

    let dt = 0.01;
    for _ in 0..200 {
        u = solve_transient(&m, &k, &f, &u, dt, &bcs);
    }

    let mut max_err: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        max_err = max_err.max((u[i] - p.z).abs());
    }
    assert!(max_err < 5e-2, "max error after 200 steps: {max_err}");
}

// ─── Bonus: matches the FemHeat viewer scene ───────────────────────
//
// Verifies the exact recipe the viewer uses (16³ tet cube, dt=0.01,
// transient stepping) actually evolves toward T(z) = z. Mirrors the
// scene's per-frame `solve_transient` call so the viewer doesn't
// silently break in the future.

#[test]
fn fem_heat_viewer_recipe_evolves_toward_analytic() {
    let n_grid = 16usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let alpha = 1.0;
    let m = assemble_global(&mesh, space, compute_element_mass).unwrap();
    let k = assemble_stiffness(&mesh, alpha);
    let n = space.ndof(&mesh);

    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.z.abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        } else if (p.z - 1.0).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 1.0,
            });
        }
    }

    let f = vec![0.0; n];
    let mut u = vec![0.0; n];
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            u[*vertex as usize] = *value;
        }
    }

    let dt = 0.01;
    let initial_err: Real = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(i, p)| (u[i] - p.z).abs())
        .fold(0.0, Real::max);

    for _ in 0..400 {
        u = solve_transient(&m, &k, &f, &u, dt, &bcs);
    }
    let final_err: Real = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(i, p)| (u[i] - p.z).abs())
        .fold(0.0, Real::max);

    assert!(
        final_err < initial_err,
        "FemHeat recipe did not converge: initial={initial_err}, final={final_err}"
    );
    // Mid-plane cross-section should approach z = 0.5.
    let mid: Real = mesh
        .vertices
        .iter()
        .enumerate()
        .filter(|(_, p)| (p.z - 0.5).abs() < 1.0 / (n_grid as Real * 2.0))
        .map(|(i, _)| u[i])
        .sum::<Real>()
        / mesh
            .vertices
            .iter()
            .filter(|p| (p.z - 0.5).abs() < 1.0 / (n_grid as Real * 2.0))
            .count() as Real;
    assert!((mid - 0.5).abs() < 0.1, "FemHeat mid-plane drifted: {mid}");
}

// Suppress unused-import lint when only some of the re-exports are
// referenced.
#[test]
fn smoke_imports() {
    let _ = apply_dirichlet;
    let _ = compute_element_stiffness;
}
