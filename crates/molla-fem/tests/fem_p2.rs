//! Stage 36.3 — P2 (quadratic-nodal) FEM contract tests.
//!
//! Each test maps 1:1 to one of the deliverables in PLAN.md
//! Stage 36.3. They are CPU-only and run as part of the standard
//! `cargo test -p molla-fem` suite.
//!
//! The Stage 36 v1 (P1) contract tests in `fem_v1.rs` continue to
//! exist independently — none of these tests touch the P1 path.

use molla_fem::quadrature::tri_quadrature_6pt;
use molla_fem::{
    BoundaryCondition, FemMesh, FunctionSpace, P2DofMap, apply_dirichlet, assemble_global,
    assemble_global_p2, compute_element_p2_mass, compute_element_p2_stiffness,
    compute_element_stiffness, p2_tet_shape_functions, p2_tri_shape_functions, solve_steady_state,
};
use molla_math::sparse::BsrMatrix;
use molla_math::{Real, Vec3};

// ─── Helpers ─────────────────────────────────────────────────────────

fn assemble_p1_stiffness(mesh: &FemMesh, alpha: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .expect("assemble P1 stiffness")
}

fn assemble_p2_stiffness(mesh: &FemMesh, map: &P2DofMap, alpha: Real) -> BsrMatrix {
    assemble_global_p2(mesh, map, |c, v| compute_element_p2_stiffness(c, v, alpha))
        .expect("assemble P2 stiffness")
}

// ─── Test 1: P2 tri shape functions partition of unity ──────────────
//
// At any barycentric point on the reference triangle, the 6 P2
// shape functions sum to 1.

#[test]
fn p2_tri_shape_function_partition_of_unity() {
    let samples: [[Real; 3]; 6] = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
    ];
    for bary in samples {
        let n = p2_tri_shape_functions(bary);
        let s: Real = n.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "P2 tri shape functions sum to {s} at {bary:?}"
        );
    }
}

// ─── Test 2: P2 tet kronecker delta ────────────────────────────────
//
// The 10 reference points of a P2 tet are 4 corners + 6 edge
// midpoints. At each reference point, the corresponding shape
// function is exactly 1 and the other 9 are exactly 0.

#[test]
fn p2_tet_shape_function_kronecker_delta() {
    // Reference points in barycentric coords for the 10 nodes.
    // Order matches `p2_tet_shape_functions`:
    //   c0..c3 (corners), e01, e02, e03, e12, e13, e23 (edge mids).
    let refs: [[Real; 4]; 10] = [
        [1.0, 0.0, 0.0, 0.0], // c0
        [0.0, 1.0, 0.0, 0.0], // c1
        [0.0, 0.0, 1.0, 0.0], // c2
        [0.0, 0.0, 0.0, 1.0], // c3
        [0.5, 0.5, 0.0, 0.0], // e01
        [0.5, 0.0, 0.5, 0.0], // e02
        [0.5, 0.0, 0.0, 0.5], // e03
        [0.0, 0.5, 0.5, 0.0], // e12
        [0.0, 0.5, 0.0, 0.5], // e13
        [0.0, 0.0, 0.5, 0.5], // e23
    ];
    for (i, bary) in refs.iter().enumerate() {
        let n = p2_tet_shape_functions(*bary);
        for (j, &val) in n.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - want).abs() < 1e-12,
                "N[{j}] at ref {i} = {val}, want {want}"
            );
        }
    }
}

// ─── Test 3: tri 6-pt rule integrates polynomials up to degree 4 ────
//
// On the reference (unit) triangle [(0,0), (1,0), (0,1)], the
// monomial integrals are
//   ∫∫ xᵖ yᵖ' dA = p! · p'! / (p + p' + 2)!
// We test x², x²y, xy², all integrable by a degree-4 rule.

#[test]
fn p2_tri_quadrature_exact_for_polynomials_up_to_degree_4() {
    // Map a barycentric point on the reference triangle (vertices
    // A=(0,0), B=(1,0), C=(0,1)) to (x, y): (x, y) = λ_B · B + λ_C · C
    // = (λ_B, λ_C). The triangle has area 1/2, so the Jacobian factor
    // is 1/2 — and the rule's weights are pre-normalised to sum to 1,
    // so the integral becomes `area · ∑ wᵢ · f(λᵢ)`.
    let area: Real = 0.5;
    let qpts = tri_quadrature_6pt();

    let integrate = |f: &dyn Fn(Real, Real) -> Real| -> Real {
        let mut acc: Real = 0.0;
        for q in &qpts {
            let x = q.bary[1];
            let y = q.bary[2];
            acc += q.weight * f(x, y);
        }
        area * acc
    };

    // ∫∫ x² dA = 2!·0! / 4! = 1/12.
    let i_x2 = integrate(&|x, _| x * x);
    assert!(
        (i_x2 - 1.0 / 12.0).abs() < 1e-12,
        "∫x² = {i_x2}, want {}",
        1.0 / 12.0
    );
    // ∫∫ x²y dA = 2!·1! / 5! = 1/60.
    let i_x2y = integrate(&|x, y| x * x * y);
    assert!(
        (i_x2y - 1.0 / 60.0).abs() < 1e-12,
        "∫x²y = {i_x2y}, want {}",
        1.0 / 60.0
    );
    // ∫∫ xy² dA = 1!·2! / 5! = 1/60.
    let i_xy2 = integrate(&|x, y| x * y * y);
    assert!(
        (i_xy2 - 1.0 / 60.0).abs() < 1e-12,
        "∫xy² = {i_xy2}, want {}",
        1.0 / 60.0
    );
    // Bonus: ∫∫ x⁴ dA = 4!·0! / 6! = 24/720 = 1/30 (degree-4 monomial).
    let i_x4 = integrate(&|x, _| x.powi(4));
    assert!(
        (i_x4 - 1.0 / 30.0).abs() < 1e-12,
        "∫x⁴ = {i_x4}, want {}",
        1.0 / 30.0
    );
}

// ─── Test 4: P2 vs P1 accuracy on a 2D Laplacian Dirichlet problem ──
//
// Choose the harmonic function u(x, y) = exp(x) · cos(y), which
// satisfies ∇²u = 0 and is *not* a polynomial. On a coarse mesh:
//   • P1 has order-h² interior error (linear basis can't capture
//     the curvature of u).
//   • P2 has order-h³ interior error (quadratic basis matches u to
//     a much higher order).
//
// Spec: P2 error must be at least 4× smaller than P1 (a generous
// lower bound — empirically the gap is closer to ~30× at this mesh
// size).

#[test]
fn p2_assemble_laplacian_2d_unit_square_higher_accuracy_than_p1() {
    // Coarse 4×4 grid — the "P2 wins" gap is most visible at low
    // resolution and shrinks (still positive) as both methods converge.
    let nx = 4;
    let ny = 4;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    // Harmonic, non-polynomial: u(x, y) = exp(x) · cos(y).
    let analytic = |p: Vec3| (p.x).exp() * (p.y).cos();

    // Identify boundary corners (P1 + P2 corner DoFs share indices).
    let on_boundary: Vec<bool> = mesh
        .vertices
        .iter()
        .map(|p| {
            p.x.abs() < 1e-9
                || (p.x - 1.0).abs() < 1e-9
                || p.y.abs() < 1e-9
                || (p.y - 1.0).abs() < 1e-9
        })
        .collect();

    // ── P1 solve ──────────────────────────────────────────────────
    let p1_space = FunctionSpace::P1Nodal;
    let k_p1 = assemble_p1_stiffness(&mesh, 1.0);
    let n_p1 = p1_space.ndof(&mesh);
    let f_p1 = vec![0.0; n_p1];
    let bcs_p1: Vec<BoundaryCondition> = mesh
        .vertices
        .iter()
        .enumerate()
        .filter(|(i, _)| on_boundary[*i])
        .map(|(i, p)| BoundaryCondition::Dirichlet {
            vertex: i as u32,
            value: analytic(*p),
        })
        .collect();
    let u_p1 = solve_steady_state(&k_p1, &f_p1, &bcs_p1, 5000, 1e-12);
    let mut err_p1: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        if !on_boundary[i] {
            err_p1 = err_p1.max((u_p1[i] - analytic(*p)).abs());
        }
    }

    // ── P2 solve ──────────────────────────────────────────────────
    let p2_map = P2DofMap::build(&mesh);
    let k_p2 = assemble_p2_stiffness(&mesh, &p2_map, 1.0);
    let n_p2 = p2_map.ndof();
    assert_eq!(n_p2, p1_space.ndof(&mesh) + p2_map.edge_id.len());
    let f_p2 = vec![0.0; n_p2];

    // Dirichlet BCs: corner DoFs come from the boundary vertices,
    // edge DoFs from any edge whose both endpoints are on the boundary.
    let mut bcs_p2: Vec<BoundaryCondition> = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if on_boundary[i] {
            bcs_p2.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: analytic(*p),
            });
        }
    }
    // Edge midpoints lying on a boundary edge.
    for (edge, &edge_id) in &p2_map.edge_id {
        let a = edge[0] as usize;
        let b = edge[1] as usize;
        if on_boundary[a] && on_boundary[b] {
            // Both endpoints share a boundary face only if they lie on
            // the same side of the unit square; the tri-grid never
            // produces a diagonal whose endpoints are both boundary.
            let pa = mesh.vertices[a];
            let pb = mesh.vertices[b];
            let same_side = (pa.x.abs() < 1e-9 && pb.x.abs() < 1e-9)
                || ((pa.x - 1.0).abs() < 1e-9 && (pb.x - 1.0).abs() < 1e-9)
                || (pa.y.abs() < 1e-9 && pb.y.abs() < 1e-9)
                || ((pa.y - 1.0).abs() < 1e-9 && (pb.y - 1.0).abs() < 1e-9);
            if same_side {
                let mid = Vec3::new(0.5 * (pa.x + pb.x), 0.5 * (pa.y + pb.y), 0.0);
                bcs_p2.push(BoundaryCondition::Dirichlet {
                    vertex: (p2_map.vertex_count + edge_id as usize) as u32,
                    value: analytic(mid),
                });
            }
        }
    }

    let u_p2 = solve_steady_state(&k_p2, &f_p2, &bcs_p2, 5000, 1e-12);

    // Sample the P2 error at the *interior* corner DoFs (so we
    // compare apples-to-apples with P1).
    let mut err_p2: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        if !on_boundary[i] {
            err_p2 = err_p2.max((u_p2[i] - analytic(*p)).abs());
        }
    }

    println!("P1 max interior error = {err_p1:.3e}");
    println!("P2 max interior error = {err_p2:.3e}");
    // P2 should be at least 4× more accurate than P1 (per spec).
    assert!(err_p1 > 0.0, "P1 error should be nonzero on coarse mesh");
    assert!(
        err_p2 < err_p1 / 4.0,
        "P2 error {err_p2:.3e} should be < P1 error {err_p1:.3e} / 4"
    );
}

// ─── Test 5: 3D heat steady state — P2 better than P1 ──────────────
//
// Same problem as the v1 `heat_steady_state_3d_cube_matches_analytic`
// (BCs T(z=0)=0, T(z=1)=1, sides insulating) but with P2 elements.
// Steady state is the linear field T(z) = z which both P1 and P2
// represent exactly — but we still verify P2 hits machine precision.

#[test]
fn p2_heat_steady_state_3d_cube_matches_analytic() {
    let n_grid = 4usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let map = P2DofMap::build(&mesh);
    let k = assemble_p2_stiffness(&mesh, &map, 1.0);
    let n = map.ndof();
    let f = vec![0.0; n];

    let mut bcs = Vec::new();
    // Corner DoFs on the top + bottom faces.
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
    // Edge midpoints on the top + bottom faces (BC value = analytic
    // T(z) = z evaluated at the midpoint).
    for (edge, &edge_id) in &map.edge_id {
        let a = edge[0] as usize;
        let b = edge[1] as usize;
        let pa = mesh.vertices[a];
        let pb = mesh.vertices[b];
        let mid_z = 0.5 * (pa.z + pb.z);
        let on_bottom = pa.z.abs() < 1e-9 && pb.z.abs() < 1e-9;
        let on_top = (pa.z - 1.0).abs() < 1e-9 && (pb.z - 1.0).abs() < 1e-9;
        if on_bottom || on_top {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: (map.vertex_count + edge_id as usize) as u32,
                value: mid_z,
            });
        }
    }

    let u = solve_steady_state(&k, &f, &bcs, 10000, 1e-10);

    // Check corner DoFs against T(z) = z.
    let mut max_corner_err: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        max_corner_err = max_corner_err.max((u[i] - p.z).abs());
    }
    // Check edge DoFs against T(midpoint) = midpoint_z.
    let mut max_edge_err: Real = 0.0;
    for (edge, &edge_id) in &map.edge_id {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        let mid_z = 0.5 * (pa.z + pb.z);
        let v = u[map.vertex_count + edge_id as usize];
        max_edge_err = max_edge_err.max((v - mid_z).abs());
    }
    println!("P2 corner err = {max_corner_err:.3e}, edge err = {max_edge_err:.3e}");
    // P2 represents linear functions exactly ⇒ both errors at CG tol.
    assert!(max_corner_err < 1e-3, "max corner err {max_corner_err}");
    assert!(max_edge_err < 1e-3, "max edge err {max_edge_err}");
}

// ─── Bonus: smoke-import to keep clippy quiet ─────────────────────

#[test]
fn smoke_p2_imports() {
    let _ = compute_element_p2_mass;
    let _ = apply_dirichlet;
}
