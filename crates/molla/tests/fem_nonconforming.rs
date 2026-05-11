//! Stage 47 v1 — `NonconformingField` L²-projection contract tests.
//!
//! Three properties pin down the operator:
//!
//! 1. **Constant reproduction** — `u_source ≡ 1` projects to
//!    `u_target ≡ 1` exactly. Tests the partition-of-unity property of
//!    the cross-mesh mass matrix.
//! 2. **Linear (affine) reproduction** — a linear field is in P1, so the
//!    L² projector should reproduce it at every target vertex up to
//!    round-off.
//! 3. **Quadratic decay** — a quadratic source field is *not* in P1, so
//!    the projection has an O(h²) error. Doubling resolution halves the
//!    error by ≈ 4×.
//!
//! All three tests use unit-cube tet grids at different resolutions so
//! the source and target meshes are deliberately non-matching (no
//! shared DoFs, no aligned faces, no shared cells).

use molla::fem::{FemMesh, NonconformingField};
use molla::math::Real;

/// Evaluate a field defined on a mesh's vertices at vertex `i`.
fn sample(field: &[Real], i: usize) -> Real {
    field[i]
}

/// Linear test field: `f(x) = x + 2y + 3z`.
fn linear_field(mesh: &FemMesh) -> Vec<Real> {
    mesh.vertices
        .iter()
        .map(|v| v.x + 2.0 * v.y + 3.0 * v.z)
        .collect()
}

/// Quadratic test field: `f(x) = x²`.
fn quadratic_field(mesh: &FemMesh) -> Vec<Real> {
    mesh.vertices.iter().map(|v| v.x * v.x).collect()
}

#[test]
fn project_constant_field_is_constant() {
    // Source: 4³ tet grid, target: 3³ tet grid (rotated would also be
    // valid; we keep the axis-aligned case so the geometric overlap is
    // unambiguous — every target quadrature point falls inside some
    // source tet).
    let source = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
    let target = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let field = NonconformingField::new(&source, &target).expect("build NonconformingField");

    let u_source = vec![1.0; field.source_dofs()];
    let u_target = field.project_with(&u_source, 1.0e-13, 2000);

    for (i, &v) in u_target.iter().enumerate() {
        let err = (v - 1.0).abs();
        assert!(
            err < 1.0e-10,
            "vertex {i}: u_target = {v}, expected 1.0 (err = {err})"
        );
    }
}

#[test]
fn project_linear_field_is_linear() {
    // P1 reproduces affine fields exactly; the L² projector onto a
    // space that already contains the source field is the identity.
    let source = FemMesh::tet_grid_3d(5, 5, 5, 1.0, 1.0, 1.0);
    let target = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
    let field = NonconformingField::new(&source, &target).expect("build NonconformingField");

    let u_source = linear_field(&source);
    let u_target = field.project_with(&u_source, 1.0e-13, 2000);
    let expected = linear_field(&target);

    let mut max_err: Real = 0.0;
    for i in 0..u_target.len() {
        let err = (sample(&u_target, i) - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < 1.0e-9,
        "max linear-reproduction error {max_err} > 1e-9"
    );
}

/// Compute the L²-norm-style RMS error between a projected target field
/// and the analytic reference, summed over target vertices.
fn rms_error(u_target: &[Real], reference: &[Real]) -> Real {
    let n = u_target.len();
    assert_eq!(n, reference.len());
    let mut s = 0.0;
    for i in 0..n {
        let d = u_target[i] - reference[i];
        s += d * d;
    }
    (s / n as Real).sqrt()
}

#[test]
fn project_quadratic_field_loses_o_h_squared() {
    // f(x) = x². P1 best approximation has O(h²) error in L²; we
    // expect the error to drop by ~4× when we double the target
    // resolution. (Source resolution is held high enough that source
    // discretisation error is small relative to the target's.)
    let source = FemMesh::tet_grid_3d(16, 16, 16, 1.0, 1.0, 1.0);

    let target_coarse = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
    let target_fine = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);

    let field_c = NonconformingField::new(&source, &target_coarse)
        .expect("build NonconformingField (coarse)");
    let field_f = NonconformingField::new(&source, &target_fine)
        .expect("build NonconformingField (fine)");

    let u_source = quadratic_field(&source);
    let u_c = field_c.project_with(&u_source, 1.0e-12, 2000);
    let u_f = field_f.project_with(&u_source, 1.0e-12, 2000);

    let ref_c = quadratic_field(&target_coarse);
    let ref_f = quadratic_field(&target_fine);

    let err_c = rms_error(&u_c, &ref_c);
    let err_f = rms_error(&u_f, &ref_f);

    // The expected decay rate is exactly 4× for O(h²) under uniform
    // refinement; we allow a generous bracket [2.5, 6.0] to account
    // for the finite source resolution introducing its own (smaller)
    // error contribution and for the asymptotic regime not being
    // reached at h = 1/2 → 1/4.
    let ratio = err_c / err_f.max(1.0e-30);
    assert!(
        err_c > err_f,
        "coarse error {err_c} should exceed fine error {err_f}"
    );
    assert!(
        ratio > 2.5,
        "O(h²) decay ratio {ratio} too low (coarse_err={err_c}, fine_err={err_f})"
    );
    assert!(
        ratio < 6.0,
        "O(h²) decay ratio {ratio} too high — suggests off-by-one in convergence regime"
    );
    eprintln!(
        "stage 47 v1 quadratic-field O(h²) decay: err_coarse={err_c:.4e}, err_fine={err_f:.4e}, ratio={ratio:.2}"
    );
}
