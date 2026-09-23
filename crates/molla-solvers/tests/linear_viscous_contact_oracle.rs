//! Exact linear spring-dashpot contact references for implicit coupling.

use molla_math::{Mat3, Vec3};

fn outer(a: Vec3) -> Mat3 {
    Mat3::from_cols(a * a.x, a * a.y, a * a.z)
}

fn metric(mass: Vec3, h: f64, damping: f64, stiffness: f64) -> Mat3 {
    Mat3::from_diagonal(mass) + outer(Vec3::new(1.0, -1.0, 0.0)) * (h * damping + h * h * stiffness)
}

fn contact(predictor: Vec3, response: Mat3, normal: Vec3) -> (Vec3, f64) {
    let direction = response * normal;
    let impulse = (-normal.dot(predictor) / normal.dot(direction)).max(0.0);
    (predictor + direction * impulse, impulse)
}

fn energy(mass: Vec3, velocity: Vec3, h: f64, stiffness: f64) -> f64 {
    0.5 * mass.dot(velocity * velocity) + 0.5 * stiffness * (h * (velocity.x - velocity.y)).powi(2)
}

#[test]
fn implicit_contact_satisfies_balance_complementarity_and_dissipation() {
    let mass = Vec3::new(0.001, 0.002, 0.1);
    let initial = Vec3::new(-1.0, 0.0, 0.0);
    let rhs = mass * initial;
    for h in [1.0 / 19200.0, 1.0 / 38400.0, 1.0 / 76800.0] {
        for normal in [Vec3::X, Vec3::new(1.0, 0.0, -1.0)] {
            let a = metric(mass, h, 20.0, 10000.0);
            let inverse = a.inverse();
            let predictor = inverse * rhs;
            let (velocity, impulse) = contact(predictor, inverse, normal);
            assert!(impulse > 0.0);
            assert!(normal.dot(velocity).abs() < 1e-12);
            assert!((a * velocity - rhs - normal * impulse).length() < 1e-14);
            assert!((mass.dot(velocity - initial) - normal.element_sum() * impulse).abs() < 1e-14);
            assert!(energy(mass, velocity, h, 10000.0) <= energy(mass, initial, 0.0, 10000.0));
            if normal == Vec3::X {
                assert!(velocity.length() < 1e-12);
                assert!((impulse - mass.x).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn mass_projection_leaves_a_free_dof_implicit_residual() {
    let mass = Vec3::splat(0.001);
    let initial = Vec3::new(-1.0, 0.0, 0.0);
    let rhs = mass * initial;
    let mut previous_error = f64::INFINITY;
    for divisor in [1.0, 2.0, 4.0, 8.0, 16.0] {
        let h = 1.0 / (19200.0 * divisor);
        let a = metric(mass, h, 20.0, 10000.0);
        let predictor = a.inverse() * rhs;
        let (split, impulse) = contact(predictor, Mat3::from_diagonal(mass.recip()), Vec3::X);
        let (coupled, _) = contact(predictor, a.inverse(), Vec3::X);
        let residual = a * split - rhs - Vec3::X * impulse;
        let error = (split - coupled).length();
        assert!(split.x.abs() < 1e-12);
        assert!(residual.y.abs() > 1e-6);
        assert!(error > 0.01 && error < previous_error);
        assert!((mass.dot(split - initial) - impulse).abs() < 1e-14);
        eprintln!(
            "linear viscous contact: h={h}, split error={error}, free residual={}",
            residual.y
        );
        previous_error = error;
    }
}

#[test]
fn zero_internal_tangent_and_separating_contact_preserve_the_predictor() {
    let mass = Vec3::new(0.001, 0.002, 0.1);
    let a = metric(mass, 1.0 / 19200.0, 0.0, 0.0);
    let inverse_mass = Mat3::from_diagonal(mass.recip());
    for predictor in [Vec3::new(-1.0, 0.5, 0.0), Vec3::new(1.0, 0.5, 0.0)] {
        let (actual, impulse) = contact(predictor, a.inverse(), Vec3::X);
        let (expected, expected_impulse) = contact(predictor, inverse_mass, Vec3::X);
        assert!((actual - expected).length() < 1e-12);
        assert!((impulse - expected_impulse).abs() < 1e-14);
        if predictor.x > 0.0 {
            assert_eq!(impulse, 0.0);
            assert_eq!(actual, predictor);
        }
    }
}
