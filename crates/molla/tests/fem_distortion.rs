//! Stage 37.4 — symmetric Dirichlet distortion energy + Newton
//! minimisation contract tests.
//!
//! Four CPU-only checks:
//!
//! 1. **Identity → zero energy.** With `deformed == rest`, the
//!    deformation gradient is the identity and `E = 0`. The gradient
//!    is the zero vector. Both for tri (in-plane) and tet (3D).
//! 2. **Pure translation → zero energy.** Adding a constant offset to
//!    every deformed vertex leaves `F = I` ⇒ `E = 0`. Verifies that
//!    the energy is invariant under rigid translation.
//! 3. **Compressed triangle → positive monotone energy.** Scaling the
//!    deformed positions by `s` (around the centroid) makes
//!    `F = s · I` ⇒ `E = ½ (2 s² + 2 / s²) − 2`. Verify the energy
//!    is positive and grows monotonically as `s` moves away from 1.
//! 4. **Newton converges from a small perturbation.** Start from
//!    `rest + δ` (small random perturbation) and Newton-minimise the
//!    distortion energy of a 4×4 tri-grid. The total energy must drop
//!    below `1e-6` within 10 Newton iterations (the global minimum is
//!    `E = 0` at the rest pose).

use molla::fem::{
    compute_element_distortion_energy, solve_distortion_minimization, FemCell, FemMesh,
};
use molla::math::{Real, Vec3};

fn unit_tri_rest() -> Vec<Vec3> {
    vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ]
}

fn unit_tet_rest() -> Vec<Vec3> {
    vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ]
}

#[test]
fn identity_map_zero_energy() {
    // Triangle.
    let v = unit_tri_rest();
    let cell = FemCell::Tri([0, 1, 2]);
    let (e, g, _h) = compute_element_distortion_energy(&cell, &v, &v);
    assert!(e.abs() < 1e-10, "tri identity energy: expected 0, got {e}");
    for (i, gi) in g.iter().enumerate() {
        assert!(
            gi.length() < 1e-6,
            "tri identity gradient[{i}]: expected 0, got {:?}",
            gi
        );
    }

    // Tetrahedron.
    let v = unit_tet_rest();
    let cell = FemCell::Tet([0, 1, 2, 3]);
    let (e, g, _h) = compute_element_distortion_energy(&cell, &v, &v);
    assert!(e.abs() < 1e-10, "tet identity energy: expected 0, got {e}");
    for (i, gi) in g.iter().enumerate() {
        assert!(
            gi.length() < 1e-6,
            "tet identity gradient[{i}]: expected 0, got {:?}",
            gi
        );
    }
}

#[test]
fn pure_translation_zero_energy() {
    let offset = Vec3::new(0.42, -1.7, 3.14);

    // Triangle — translation in-plane (the z component is ignored by
    // the in-plane tri kernel, but the test still confirms x/y shift
    // gives zero energy).
    let rest = unit_tri_rest();
    let def: Vec<Vec3> = rest.iter().map(|p| *p + Vec3::new(offset.x, offset.y, 0.0)).collect();
    let cell = FemCell::Tri([0, 1, 2]);
    let (e, _g, _h) = compute_element_distortion_energy(&cell, &rest, &def);
    assert!(e.abs() < 1e-10, "tri translated energy: {e}");

    // Tetrahedron — full 3D translation.
    let rest = unit_tet_rest();
    let def: Vec<Vec3> = rest.iter().map(|p| *p + offset).collect();
    let cell = FemCell::Tet([0, 1, 2, 3]);
    let (e, _g, _h) = compute_element_distortion_energy(&cell, &rest, &def);
    assert!(e.abs() < 1e-10, "tet translated energy: {e}");
}

#[test]
fn compressed_triangle_positive_energy() {
    let rest = unit_tri_rest();
    let cell = FemCell::Tri([0, 1, 2]);
    // Scale the rest mesh around the origin (vertex 0). For the
    // unit triangle this gives `F = s · I`, and the symmetric
    // Dirichlet energy reduces to `E(s) = (s² + 1/s²) − 2 = 0` at
    // `s = 1`, growing monotonically away from it. We verify both
    // monotone growth and positivity for `s ∈ {0.5, 0.75, 1.5, 2.0}`.
    let scales: [Real; 4] = [0.5, 0.75, 1.5, 2.0];
    let mut energies = Vec::new();
    for &s in &scales {
        let def: Vec<Vec3> = rest.iter().map(|p| Vec3::new(p.x * s, p.y * s, p.z)).collect();
        let (e, _g, _h) = compute_element_distortion_energy(&cell, &rest, &def);
        let analytic = s * s + 1.0 / (s * s) - 2.0;
        assert!(
            e > 0.0,
            "tri compressed (s={s}) energy must be positive: got {e}"
        );
        // Closed-form match for the diagonal `F = s I` case.
        assert!(
            (e - analytic).abs() < 1e-6,
            "tri scale s={s}: numerical E={e}, analytic E={analytic}"
        );
        energies.push((s, e));
    }
    // Monotone in |s − 1|: take pairs (0.5, 0.75) — closer to 1 should
    // have smaller energy; same for (2.0, 1.5).
    assert!(energies[1].1 < energies[0].1, "0.75 closer to 1 than 0.5");
    assert!(energies[2].1 < energies[3].1, "1.5 closer to 1 than 2.0");
}

#[test]
fn newton_minimisation_converges_on_perturbation() {
    // Small 4×4 triangulated grid as the rest mesh; Newton from a
    // perturbed warm-start. Total per-mesh distortion energy must drop
    // below 1e-6 within 10 iterations.
    let mesh = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
    let rest = mesh.vertices.clone();

    // Deterministic small perturbation (in plane only). Use a simple
    // hash-based pseudo-random offset in [-2e-2, 2e-2] per coord.
    let mut init: Vec<Vec3> = rest.clone();
    for (i, p) in init.iter_mut().enumerate() {
        let h1 = ((i as u64).wrapping_mul(2654435761)) as f64 / u64::MAX as f64;
        let h2 = ((i as u64).wrapping_mul(40503)) as f64 / u64::MAX as f64;
        // Scale into [-1, 1] then * 2e-2.
        let dx = (2.0 * h1 - 1.0) * 2.0e-2;
        let dy = (2.0 * h2 - 1.0) * 2.0e-2;
        // Don't perturb the boundary vertices — Newton can't recover
        // from a moved boundary toward the rest pose without pinning,
        // because there's no constraint forcing the boundary back to
        // its original location. The minimum-energy configuration of
        // the unconstrained problem with moved boundaries is itself
        // not the rest mesh.
        let on_boundary = p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9;
        if !on_boundary {
            p.x += dx;
            p.y += dy;
        }
    }

    let final_def = solve_distortion_minimization(&mesh, &init, 10, 1e-12);

    // Total energy of the final configuration — sum the per-cell
    // contributions.
    let mut total_e: Real = 0.0;
    for cell in &mesh.cells {
        let (e, _g, _h) = compute_element_distortion_energy(cell, &rest, &final_def);
        total_e += e;
    }
    assert!(
        total_e < 1e-6,
        "total distortion energy after Newton: {total_e}; expected < 1e-6"
    );
}
