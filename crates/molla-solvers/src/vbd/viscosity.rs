use crate::math::{Mat3, Vec3};

pub(super) struct Block {
    pub(super) force: Vec3,
    pub(super) tangent: Mat3,
    #[cfg(test)]
    geometric: f64,
    #[cfg(test)]
    potential: f64,
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

fn trace(a: Mat3) -> f64 {
    a.x_axis.x + a.y_axis.y + a.z_axis.z
}

#[cfg(test)]
fn norm_squared(a: Mat3) -> f64 {
    a.x_axis.length_squared() + a.y_axis.length_squared() + a.z_axis.length_squared()
}

#[cfg(test)]
fn block(
    f: Mat3,
    previous: Mat3,
    gradient: Vec3,
    shear: f64,
    bulk: f64,
    dt: f64,
    volume: f64,
) -> Block {
    increment_block(previous, f - previous, gradient, shear, bulk, dt, volume)
}

pub(super) fn increment_block(
    previous: Mat3,
    increment: Mat3,
    gradient: Vec3,
    shear: f64,
    bulk: f64,
    dt: f64,
    volume: f64,
) -> Block {
    let f = previous + increment;
    let strain = (previous.transpose() * increment
        + increment.transpose() * previous
        + increment.transpose() * increment)
        * 0.5;
    let tr = trace(strain);
    let deviator = strain - Mat3::IDENTITY * (tr / 3.0);
    let stress = deviator * (2.0 * shear) + Mat3::IDENTITY * (bulk * tr);
    let scale = volume / dt;
    let a = f * gradient;
    Block {
        force: -(f * stress * gradient) * scale,
        tangent: (f * f.transpose() * (shear * gradient.length_squared())
            + outer(a, a) * (bulk + shear / 3.0))
            * scale,
        #[cfg(test)]
        geometric: gradient.dot(stress * gradient) * scale,
        #[cfg(test)]
        potential: (shear * norm_squared(deviator) + 0.5 * bulk * tr * tr) * scale,
    }
}

#[cfg(test)]
fn states() -> Vec<(Mat3, Mat3)> {
    let mut result = Vec::new();
    for previous in [
        Mat3::IDENTITY,
        Mat3::from_cols(
            Vec3::new(0.9, 0.1, 0.0),
            Vec3::new(0.2, 1.1, 0.05),
            Vec3::new(0.0, 0.03, 1.2),
        ),
    ] {
        for stretch in [
            Mat3::IDENTITY,
            Mat3::from_diagonal(Vec3::new(0.8, 1.2, 1.05)),
            Mat3::from_cols(Vec3::X, Vec3::new(0.3, 1.0, 0.0), Vec3::new(0.0, -0.2, 1.0)),
        ] {
            for rotation in [
                Mat3::IDENTITY,
                Mat3::from_quat(
                    crate::math::Quat::from_rotation_x(0.3)
                        * crate::math::Quat::from_rotation_y(-0.7)
                        * crate::math::Quat::from_rotation_z(1.2),
                ),
            ] {
                result.push((rotation * stretch * previous, previous));
            }
        }
    }
    result
}

#[test]
fn green_viscosity_force_and_tangent_match_incremental_potential() {
    let gradient = Vec3::new(-0.4, 0.7, 0.2);
    for (f, previous) in states() {
        let actual = block(f, previous, gradient, 3.0, 7.0, 0.02, 0.17);
        let full = actual.tangent + Mat3::IDENTITY * actual.geometric;
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let delta = outer(axis, gradient) * 1e-6;
            let plus = block(f + delta, previous, gradient, 3.0, 7.0, 0.02, 0.17);
            let minus = block(f - delta, previous, gradient, 3.0, 7.0, 0.02, 0.17);
            let numerical_force = -(plus.potential - minus.potential) / 2e-6;
            assert!((actual.force.dot(axis) - numerical_force).abs() < 1e-6);
            let numerical_tangent = -(plus.force - minus.force) / 2e-6;
            assert!((full * axis - numerical_tangent).length() < 2e-6);
            assert!(axis.dot(actual.tangent * axis) >= 0.0);
        }
        assert!(actual.potential >= 0.0);
    }
}

#[test]
fn green_viscosity_is_frame_indifferent_and_has_no_rigid_spin_drag() {
    let rotation = Mat3::from_quat(crate::math::Quat::from_rotation_z(1.7));
    let old_rotation = Mat3::from_quat(crate::math::Quat::from_rotation_y(-0.8));
    let gradient = Vec3::new(0.3, -0.2, 0.8);
    for (f, previous) in states() {
        let original = block(f, previous, gradient, 3.0, 7.0, 0.02, 0.17);
        let rotated = block(
            rotation * f,
            old_rotation * previous,
            gradient,
            3.0,
            7.0,
            0.02,
            0.17,
        );
        assert!((rotated.potential - original.potential).abs() < 1e-11);
        assert!((rotated.force - rotation * original.force).length() < 1e-11);
        assert!(
            norm_squared(rotated.tangent - rotation * original.tangent * rotation.transpose())
                < 1e-20
        );
        let rigid = block(
            rotation * previous,
            previous,
            gradient,
            3.0,
            7.0,
            0.02,
            0.17,
        );
        assert!(rigid.force.length() < 1e-12 && rigid.potential < 1e-24);
    }
}

#[test]
fn green_viscosity_conserves_internal_force_and_torque() {
    let reference = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    let gradients = [-Vec3::ONE, Vec3::X, Vec3::Y, Vec3::Z];
    for (f, previous) in states() {
        let mut force = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        for (point, gradient) in reference.into_iter().zip(gradients) {
            let b = block(f, previous, gradient, 3.0, 7.0, 0.02, 1.0 / 6.0);
            force += b.force;
            torque += (f * point + Vec3::new(12.0, -7.0, 3.0)).cross(b.force);
        }
        assert!(force.length() < 1e-11 && torque.length() < 1e-11);
        let zero = block(f, previous, Vec3::X, 0.0, 0.0, 0.02, 0.17);
        assert_eq!(zero.force, Vec3::ZERO);
        assert_eq!(zero.tangent, Mat3::ZERO);
        assert_eq!(zero.potential, 0.0);
    }
}

#[test]
fn green_viscosity_uniform_dilation_matches_scalar_oracle() {
    for stretch in [0.7, 1.0, 1.3] {
        let delta = (stretch * stretch - 1.0) * 0.5;
        let b = block(
            Mat3::IDENTITY * stretch,
            Mat3::IDENTITY,
            Vec3::X,
            3.0,
            7.0,
            0.02,
            0.17,
        );
        let expected = -0.17 / 0.02 * stretch * 7.0 * 3.0 * delta;
        assert!((b.force.x - expected).abs() < 1e-12);
        assert_eq!([b.force.y, b.force.z], [0.0; 2]);
        assert!((b.potential - 0.17 / 0.02 * 0.5 * 7.0 * (3.0 * delta).powi(2)).abs() < 1e-12);
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu;
