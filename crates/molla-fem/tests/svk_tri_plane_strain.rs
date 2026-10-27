//! INSPECTION.md F13 — plane-strain SVK on Tri elements (previously a
//! deliberate `unimplemented!` panic).
//!
//! Oracles:
//! 1. Rest state → zero internal force.
//! 2. Rigid translation → zero force (any strain measure).
//! 3. Rigid ROTATION → zero force. This is the SVK-vs-linear
//!    discriminator: the Green strain is rotation-invariant while the
//!    linear small-strain tensor is not.
//! 4. Small strain → matches the linear plane-strain element
//!    (`compute_element_elasticity_stiffness`) to first order.
//! 5. Force balance: element-internal forces sum to zero.

use molla_fem::{
    FemCell, compute_element_elasticity_stiffness, compute_element_st_venant_kirchhoff_stress,
};
use molla_math::{Real, Vec3};

const LAMBDA: Real = 1.2e5;
const MU: Real = 8.0e4;

fn tri() -> (FemCell, Vec<Vec3>) {
    (
        FemCell::Tri([0, 1, 2]),
        vec![
            Vec3::new(0.1, 0.2, 0.0),
            Vec3::new(0.9, 0.3, 0.0),
            Vec3::new(0.4, 1.1, 0.0),
        ],
    )
}

fn svk(u: &[Real; 6]) -> Vec<Real> {
    let (cell, verts) = tri();
    compute_element_st_venant_kirchhoff_stress(&cell, &verts, u, LAMBDA, MU)
}

#[test]
fn rest_state_has_zero_force() {
    let f = svk(&[0.0; 6]);
    assert_eq!(f.len(), 6);
    for v in &f {
        assert!(v.abs() < 1e-12, "rest force {v}");
    }
}

#[test]
fn rigid_translation_has_zero_force() {
    let f = svk(&[0.3, -0.7, 0.3, -0.7, 0.3, -0.7]);
    for v in &f {
        assert!(v.abs() < 1e-9, "translation force {v}");
    }
}

#[test]
fn rigid_rotation_has_zero_force() {
    // 25° rotation about the origin: u_i = R p_i − p_i. Green strain
    // is exactly zero, so SVK force must vanish; a linear element
    // would report large spurious stress here.
    let (_, verts) = tri();
    let th: Real = 25.0_f64.to_radians();
    let (s, c) = th.sin_cos();
    let mut u = [0.0; 6];
    for (i, p) in verts.iter().enumerate() {
        u[2 * i] = (c * p.x - s * p.y) - p.x;
        u[2 * i + 1] = (s * p.x + c * p.y) - p.y;
    }
    let f = svk(&u);
    for v in &f {
        assert!(v.abs() < 1e-7, "rotation force {v}");
    }
    // Sanity that the displacement is genuinely large (non-vacuous).
    assert!(u.iter().any(|x| x.abs() > 0.1));
}

#[test]
fn internal_forces_balance() {
    let u = [0.01, -0.004, -0.006, 0.008, 0.002, 0.005];
    let f = svk(&u);
    let fx = f[0] + f[2] + f[4];
    let fy = f[1] + f[3] + f[5];
    assert!(fx.abs() < 1e-9 && fy.abs() < 1e-9, "net force ({fx}, {fy})");
}

#[test]
fn small_strain_matches_linear_plane_strain_element() {
    let (cell, verts) = tri();
    let k = compute_element_elasticity_stiffness(&cell, &verts, LAMBDA, MU);
    // Small displacement — quadratic Green-strain terms are O(ε²).
    let eps = 1.0e-6;
    let u = [
        0.3 * eps,
        -0.1 * eps,
        -0.2 * eps,
        0.4 * eps,
        0.1 * eps,
        0.2 * eps,
    ];
    let f_svk = svk(&u);
    for r in 0..6 {
        let f_lin: Real = (0..6).map(|cidx| k[r][cidx] * u[cidx]).sum();
        let denom = f_lin.abs().max(1e-12);
        assert!(
            (f_svk[r] - f_lin).abs() / denom < 1e-4,
            "row {r}: svk {} vs linear {}",
            f_svk[r],
            f_lin
        );
    }
}
