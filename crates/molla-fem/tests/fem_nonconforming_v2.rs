//! Stage 47 v2 — vector projection + penalty-spring contact tests.
//!
//! Four contract tests pin down the v2 surface:
//!
//! 1. **Rigid translation reproduction** — a constant 3-vector
//!    displacement field projects to itself component-wise, to better
//!    than 1e-9 on every target vertex.
//! 2. **Small-rotation O(h²) decay** — a small rotation field about
//!    the z-axis (linear in the in-plane coordinates) is in P1, so the
//!    L² projector reproduces it exactly. We perturb it with a
//!    quadratic component (`r²`-radial twist) so the error has the
//!    same O(h²) decay rate proven for the scalar quadratic field in
//!    v1.
//! 3. **Newton's third law** — for an arbitrary `(u_source, u_target)`
//!    pair, the penalty force pair sums to zero: `Σ f_src + Σ f_tgt ≈
//!    0` to 1e-10 (component-wise).
//! 4. **Direction sanity** — initialise the source at rest and the
//!    target lifted in `+z`; the penalty pulls the source up (`+z`)
//!    and the target down (`−z`), with **equal magnitudes** per the
//!    Newton-3 cancellation.
//!
//! All four use unit-cube tet grids at different resolutions so the
//! source and target meshes are deliberately non-matching (no shared
//! DoFs, no aligned faces).

use molla_fem::{FemMesh, NonconformingField, NonconformingPenaltyContact};
use molla_math::Real;

// ─── Helpers ────────────────────────────────────────────────────────

/// Interleave a `Vec<[Real; 3]>` style field into a flat `Vec<Real>` of
/// length `n · 3` (the layout used by `project_vector` and
/// `compute_penalty_forces`).
fn interleave3(per_vertex: &[[Real; 3]]) -> Vec<Real> {
    let mut out = Vec::with_capacity(per_vertex.len() * 3);
    for v in per_vertex {
        out.push(v[0]);
        out.push(v[1]);
        out.push(v[2]);
    }
    out
}

/// Sum the three components separately for an interleaved 3D field
/// — used by the Newton-3 test.
fn sum_components(field: &[Real]) -> [Real; 3] {
    let mut s = [0.0; 3];
    let mut i = 0;
    while i < field.len() {
        s[0] += field[i];
        s[1] += field[i + 1];
        s[2] += field[i + 2];
        i += 3;
    }
    s
}

/// Component-wise RMS error between two interleaved 3D fields.
fn rms3(a: &[Real], b: &[Real]) -> Real {
    assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s += d * d;
    }
    (s / (a.len() as Real)).sqrt()
}

// ─── 1. Vector projection: rigid translation ────────────────────────

#[test]
fn vector_project_rigid_translation_is_translation() {
    // u_source = constant 3-vector (rigid body translation). The L²
    // projector must reproduce it exactly on every target vertex,
    // independent of mesh resolution.
    let source = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
    let target = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let field = NonconformingField::new(&source, &target).expect("build NonconformingField");

    let t = [0.7, -1.3, 2.1];
    let u_source: Vec<Real> = (0..field.source_dofs())
        .flat_map(|_| t.iter().copied())
        .collect();
    let u_target = field.project_vector_with(&u_source, 3, 1.0e-13, 2000);

    assert_eq!(u_target.len(), field.target_dofs() * 3);
    for i in 0..field.target_dofs() {
        for c in 0..3 {
            let err = (u_target[i * 3 + c] - t[c]).abs();
            assert!(
                err < 1.0e-9,
                "vertex {i}, component {c}: u_target = {} expected {}, err = {err}",
                u_target[i * 3 + c],
                t[c]
            );
        }
    }
}

// ─── 2. Vector projection: rotation-with-quadratic-twist → O(h²) ────

/// A small rigid rotation about the z-axis is linear in `(x, y)`
/// (`u = (−y, x, 0) · ω`); P1 reproduces it exactly. To exercise the
/// O(h²) regime we add a quadratic "twist" component `(0, 0, r²)`
/// (with `r² = x² + y²`) that is NOT in P1, so the projection has a
/// component-wise L² error that decays as `h²` under uniform
/// refinement.
fn rotation_plus_twist(mesh: &FemMesh, omega: Real) -> Vec<Real> {
    let mut u = Vec::with_capacity(mesh.vertices.len() * 3);
    for v in &mesh.vertices {
        let r2 = v.x * v.x + v.y * v.y;
        u.push(-omega * v.y);
        u.push(omega * v.x);
        u.push(r2);
    }
    u
}

#[test]
fn vector_project_pure_rotation_p1_quadratic_decay() {
    // Source held high so the source discretisation error is small
    // relative to the target's. Coarse target: 2³, fine target: 4³.
    let source = FemMesh::tet_grid_3d(16, 16, 16, 1.0, 1.0, 1.0);

    let target_coarse = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
    let target_fine = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);

    let field_c = NonconformingField::new(&source, &target_coarse).expect("coarse field");
    let field_f = NonconformingField::new(&source, &target_fine).expect("fine field");

    let omega = 0.1;
    let u_source = rotation_plus_twist(&source, omega);
    let u_c = field_c.project_vector_with(&u_source, 3, 1.0e-12, 2000);
    let u_f = field_f.project_vector_with(&u_source, 3, 1.0e-12, 2000);

    let ref_c = rotation_plus_twist(&target_coarse, omega);
    let ref_f = rotation_plus_twist(&target_fine, omega);

    let err_c = rms3(&u_c, &ref_c);
    let err_f = rms3(&u_f, &ref_f);

    // Expected O(h²) decay: doubling target resolution should cut the
    // error by ~4×. We allow the same generous bracket as the scalar
    // v1 quadratic-field test ([2.5, 6.0]).
    let ratio = err_c / err_f.max(1.0e-30);
    assert!(
        err_c > err_f,
        "coarse error {err_c} should exceed fine error {err_f}"
    );
    assert!(
        ratio > 2.5,
        "vector O(h²) decay ratio {ratio} too low (coarse={err_c}, fine={err_f})"
    );
    assert!(ratio < 6.0, "vector O(h²) decay ratio {ratio} too high");
    eprintln!(
        "stage 47 v2 vector O(h²) decay: err_coarse={err_c:.4e}, err_fine={err_f:.4e}, ratio={ratio:.2}"
    );
}

// ─── 3. Penalty force: Newton's third law ───────────────────────────

#[test]
fn penalty_force_newton_third_law() {
    // For any (u_source, u_target), the sum of all nodal forces on
    // both meshes is zero (per component) to machine precision. The
    // cancellation comes from `1ᵀ M_st = 1ᵀ M_tt` on the target side
    // (volume preservation of the cross-mesh mass operator).
    let source = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
    let target = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let contact =
        NonconformingPenaltyContact::new(&source, &target, 10.0).expect("build penalty contact");

    // Spatially non-trivial displacements so the gap is non-zero.
    let u_source: Vec<Real> = source
        .vertices
        .iter()
        .flat_map(|v| [0.05 * v.x, 0.02 * v.y - 0.1, 0.03 * v.z + 0.4])
        .collect();
    let u_target: Vec<Real> = target
        .vertices
        .iter()
        .flat_map(|v| [0.1 + 0.01 * v.y, 0.07 - 0.05 * v.z, 0.5 + 0.02 * v.x])
        .collect();

    let (f_src, f_tgt) = contact.compute_penalty_forces_with(&u_source, &u_target, 1.0e-13, 2000);

    assert_eq!(f_src.len(), source.vertices.len() * 3);
    assert_eq!(f_tgt.len(), target.vertices.len() * 3);

    let s_src = sum_components(&f_src);
    let s_tgt = sum_components(&f_tgt);
    for c in 0..3 {
        let total = s_src[c] + s_tgt[c];
        assert!(
            total.abs() < 1.0e-10,
            "Newton-3 violation in component {c}: \
             Σ f_src = {}, Σ f_tgt = {}, sum = {}",
            s_src[c],
            s_tgt[c],
            total
        );
    }
}

// ─── 4. Penalty force: direction sanity ─────────────────────────────

#[test]
fn penalty_force_pulls_meshes_together() {
    // u_source = 0, u_target = (0, 0, 0.1) — the target is "lifted"
    // 0.1 units in +z while the source sits at rest. The penalty
    // spring should:
    //   - push the source in +z (drawn toward the lifted target).
    //   - pull the target in −z (back toward the rest source).
    // And both per-component sums must cancel (Newton's 3rd law).
    let source = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
    let target = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let contact =
        NonconformingPenaltyContact::new(&source, &target, 5.0).expect("build penalty contact");

    let u_source = vec![0.0; source.vertices.len() * 3];
    let u_target: Vec<Real> = interleave3(
        &(0..target.vertices.len())
            .map(|_| [0.0, 0.0, 0.1])
            .collect::<Vec<_>>(),
    );

    let (f_src, f_tgt) = contact.compute_penalty_forces_with(&u_source, &u_target, 1.0e-13, 2000);

    let s_src = sum_components(&f_src);
    let s_tgt = sum_components(&f_tgt);

    // X and Y components should be ≈ 0 (the gap is purely in z and the
    // problem is symmetric in x, y).
    for c in 0..2 {
        assert!(
            s_src[c].abs() < 1.0e-9,
            "f_src component {c} sum should be ≈ 0 (got {})",
            s_src[c]
        );
        assert!(
            s_tgt[c].abs() < 1.0e-9,
            "f_tgt component {c} sum should be ≈ 0 (got {})",
            s_tgt[c]
        );
    }

    // z direction: target → −z (k > 0, Δu_z = +0.1 ⇒ f_tgt_z = −k Mtt Δu_z < 0).
    //              source → +z (f_src = +k M_stᵀ Δu_target_z > 0).
    assert!(
        s_tgt[2] < -1.0e-3,
        "expected target force in −z direction, got Σ f_tgt_z = {}",
        s_tgt[2]
    );
    assert!(
        s_src[2] > 1.0e-3,
        "expected source force in +z direction, got Σ f_src_z = {}",
        s_src[2]
    );

    // Magnitudes are equal by Newton's third law.
    let mag_diff = (s_src[2] + s_tgt[2]).abs();
    assert!(
        mag_diff < 1.0e-10,
        "Newton-3 magnitude mismatch in z: Σ f_src_z = {}, Σ f_tgt_z = {} (sum = {})",
        s_src[2],
        s_tgt[2],
        mag_diff
    );
}
