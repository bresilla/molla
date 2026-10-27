use molla_math::{Mat3, Quat, Real, Vec3};

pub(crate) struct EulerBounds {
    pub angles: Vec3,
    pub gradients: [Vec3; 3],
    pub scales: Vec3,
    pub singular_interval: Option<(Real, Real, Real)>,
}

fn interval_angle(angle: Real, lower: Real, upper: Real) -> Real {
    let tau = std::f64::consts::TAU;
    let turns = if lower.is_finite() && upper.is_finite() {
        (((0.5 * lower + 0.5 * upper) - angle) / tau + 0.5).floor()
    } else if lower.is_finite() {
        ((lower - angle) / tau).ceil().max(0.)
    } else if upper.is_finite() {
        ((upper - angle) / tau).floor().min(0.)
    } else {
        0.
    };
    angle + tau * turns
}

fn regular_chart(angles: Vec3, lower: Vec3, upper: Vec3) -> (Vec3, Real) {
    let lift = |angles: Vec3| {
        Vec3::from_array(std::array::from_fn(|axis| {
            interval_angle(angles[axis], lower[axis], upper[axis])
        }))
    };
    let violation = |angles: Vec3| {
        (lower - angles)
            .max(angles - upper)
            .max(Vec3::ZERO)
            .max_element()
    };
    let primary = lift(angles);
    let alternate = lift(Vec3::new(
        angles.x + std::f64::consts::PI,
        std::f64::consts::PI - angles.y,
        angles.z + std::f64::consts::PI,
    ));
    if violation(alternate) < violation(primary) {
        (alternate, -1.)
    } else {
        (primary, 1.)
    }
}

pub(crate) fn geometry(rotation: Quat, lower: Vec3, upper: Vec3) -> EulerBounds {
    let lower = Vec3::from_array(
        lower
            .to_array()
            .map(|v| if v.is_nan() { Real::NEG_INFINITY } else { v }),
    );
    let upper = Vec3::from_array(
        upper
            .to_array()
            .map(|v| if v.is_nan() { Real::INFINITY } else { v }),
    );
    let matrix = Mat3::from_quat(rotation.normalize());
    let cy = matrix.x_axis.x.hypot(matrix.y_axis.x);
    let y = matrix.z_axis.x.atan2(cy);
    if cy > 1.9073486328125e-6 {
        let x = (-matrix.z_axis.y).atan2(matrix.z_axis.z);
        let z = (-matrix.y_axis.x).atan2(matrix.x_axis.x);
        let sine = -matrix.z_axis.y / cy;
        let cosine = matrix.z_axis.z / cy;
        let (angles, direction) = regular_chart(Vec3::new(x, y, z), lower, upper);
        return EulerBounds {
            angles,
            gradients: [
                Vec3::new(cy, sine * matrix.z_axis.x, -cosine * matrix.z_axis.x),
                Vec3::new(0., cosine, sine) * direction,
                Vec3::new(0., -sine, cosine),
            ],
            scales: Vec3::new(cy, 1., cy),
            singular_interval: None,
        };
    }
    let sign = if y >= 0. { 1. } else { -1. };
    let (zlo, zhi) = if sign > 0. {
        (lower.z, upper.z)
    } else {
        (-upper.z, -lower.z)
    };
    let lo = lower.x + zlo;
    let hi = upper.x + zhi;
    let tau = std::f64::consts::TAU;
    let mut phase = matrix.y_axis.z.atan2(matrix.y_axis.y);
    if lo.is_finite() && hi.is_finite() {
        phase += tau * ((((0.5 * lo + 0.5 * hi) - phase) / tau) + 0.5).floor();
    } else if lo.is_finite() {
        phase += tau * ((lo - phase) / tau).ceil();
    } else if hi.is_finite() {
        phase += tau * ((hi - phase) / tau).floor();
    }
    let target = phase.clamp(lo, hi);
    let contribution = 0.0_f64
        .max(zlo.max(target - upper.x))
        .min(zhi.min(target - lower.x));
    let x = phase - contribution;
    let mut pitch = interval_angle(y, lower.y, upper.y);
    let mut pitch_gradient = Vec3::new(0., x.cos(), x.sin());
    if cy > 1e-12 {
        let roll = (-matrix.z_axis.y).atan2(matrix.z_axis.z);
        let yaw = (-matrix.y_axis.x).atan2(matrix.x_axis.x);
        let (regular, direction) = regular_chart(Vec3::new(roll, y, yaw), lower, upper);
        pitch = regular.y;
        pitch_gradient = Vec3::new(0., roll.cos(), roll.sin()) * direction;
    }
    EulerBounds {
        angles: Vec3::new(x, pitch, sign * contribution),
        gradients: [Vec3::ZERO, pitch_gradient, Vec3::ZERO],
        scales: Vec3::new(0., 1., 0.),
        singular_interval: (lo.is_finite() && hi.is_finite() && hi - lo < tau)
            .then_some((phase, lo, hi)),
    }
}

pub(crate) struct AngularStop {
    pub gradient: Vec3,
    pub error: Real,
    pub bilateral: bool,
}

fn point_plane(rotation: Quat, axis: usize, bound: Real) -> AngularStop {
    let (direction, normal, reference) = if axis == 0 {
        (
            rotation * Vec3::Z,
            Quat::from_rotation_x(bound) * Vec3::Y,
            Vec3::X,
        )
    } else {
        (
            rotation * Vec3::new(bound.sin(), bound.cos(), 0.),
            Vec3::X,
            Vec3::Y,
        )
    };
    let component = direction.dot(normal).clamp(-1., 1.);
    let cross = direction.cross(normal);
    let magnitude = cross.length();
    AngularStop {
        gradient: if magnitude > 1e-6 {
            cross / magnitude
        } else {
            -component.signum() * direction.cross(reference).normalize()
        },
        error: component.asin(),
        bilateral: true,
    }
}

pub(crate) fn pair_frames(lower: Vec3, free: usize) -> (Quat, Quat) {
    match free {
        0 => (
            Quat::IDENTITY,
            Quat::from_rotation_y(lower.y) * Quat::from_rotation_z(lower.z),
        ),
        1 => (
            Quat::from_rotation_x(lower.x),
            Quat::from_rotation_z(lower.z),
        ),
        _ => (
            Quat::from_rotation_x(lower.x) * Quat::from_rotation_y(lower.y),
            Quat::IDENTITY,
        ),
    }
}

fn pair_twist(relative: Quat, free: usize) -> (Real, Vec3) {
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let primary = axes[free];
    let vector = relative.xyz();
    let component = vector[free];
    let denominator = relative.w * relative.w + component * component;
    if denominator > 1e-12 {
        return (
            2. * component.atan2(relative.w),
            ((primary * relative.w + vector.cross(primary)) * relative.w + vector * component)
                / denominator,
        );
    }
    let (b0, b1) = (axes[(free + 1) % 3], axes[(free + 2) % 3]);
    let (sine, cosine) = (vector.dot(b0), vector.dot(b1));
    (
        2. * sine.atan2(cosine),
        (cosine * (b0 * relative.w + vector.cross(b0))
            - sine * (b1 * relative.w + vector.cross(b1)))
            / (sine * sine + cosine * cosine),
    )
}

pub(crate) fn stops(
    rotation: Quat,
    lower: Vec3,
    upper: Vec3,
) -> molla_core::Result<Vec<AngularStop>> {
    let locked = |axis: usize| {
        lower[axis].is_finite()
            && upper[axis].is_finite()
            && (upper[axis] - lower[axis]).abs() < 1e-12
    };
    if (0..3).all(locked) {
        return Ok(Vec::new());
    }
    if (0..3).filter(|&axis| locked(axis)).count() == 2 {
        let free = (0..3).find(|&axis| !locked(axis)).expect("one free axis");
        let (left, right) = pair_frames(lower, free);
        let alignment = molla_sim::spherical_cone::evaluate(
            molla_sim::joint_control::SphericalConeLimits {
                angle0: Some(0.),
                angle1: Some(0.),
            },
            [Vec3::X, Vec3::Y, Vec3::Z][free],
            left.conjugate() * rotation * right.conjugate(),
        )?;
        let mut rows: Vec<_> = alignment.rows[..alignment.row_count]
            .iter()
            .map(|row| AngularStop {
                gradient: left * row.angular,
                error: row.error,
                bilateral: true,
            })
            .collect();
        if !lower[free].is_finite() && !upper[free].is_finite() {
            return Ok(rows);
        }
        let relative = (left.conjugate() * rotation * right.conjugate()).normalize();
        let (angle, gradient) = pair_twist(relative, free);
        let gradient = left * gradient;
        let angle = interval_angle(angle, lower[free], upper[free]);
        let (lo, hi) = (lower[free], upper[free]);
        for (sign, bound) in [(1., lo), (-1., hi)] {
            if bound.is_finite() {
                rows.push(AngularStop {
                    gradient: gradient * sign,
                    error: sign * (angle - bound),
                    bilateral: false,
                });
            }
        }
        return Ok(rows);
    }
    let pitch_pole = locked(1) && lower.y.cos().abs() <= 1.9073486328125e-6;
    let mut rows = Vec::new();
    let chart = if pitch_pole {
        let target = Quat::from_rotation_y(lower.y);
        let alignment = molla_sim::spherical_cone::evaluate(
            molla_sim::joint_control::SphericalConeLimits {
                angle0: Some(0.),
                angle1: Some(0.),
            },
            Vec3::Z,
            target.conjugate() * rotation,
        )?;
        for row in &alignment.rows[..alignment.row_count] {
            rows.push(AngularStop {
                gradient: target * row.angular,
                error: row.error,
                bilateral: true,
            });
        }
        let matrix = Mat3::from_quat(rotation.normalize());
        geometry(
            Quat::from_euler(
                glam::EulerRot::XYZ,
                matrix.y_axis.z.atan2(matrix.y_axis.y),
                lower.y,
                0.,
            ),
            lower,
            upper,
        )
    } else {
        geometry(rotation, lower, upper)
    };
    if !pitch_pole && chart.scales.z == 0. {
        for axis in [0, 2] {
            if locked(axis) {
                rows.push(point_plane(rotation, axis, lower[axis]));
            }
        }
    }
    for axis in 0..3 {
        if pitch_pole && axis == 1 {
            continue;
        }
        let (angle, gradient, scale, lo, hi) = if chart.scales.z == 0. && axis != 1 {
            if axis == 2 {
                continue;
            }
            let Some((phase, lo, hi)) = chart.singular_interval else {
                continue;
            };
            (phase, Vec3::X, 1., lo, hi)
        } else {
            (
                chart.angles[axis],
                chart.gradients[axis],
                chart.scales[axis],
                lower[axis],
                upper[axis],
            )
        };
        let equality = lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12;
        for (sign, bound) in [(1., lo), (-1., hi)] {
            if !bound.is_finite() || (equality && sign < 0.) {
                continue;
            }
            rows.push(AngularStop {
                gradient: gradient * sign,
                error: sign * (angle - bound) * scale,
                bilateral: equality,
            });
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_pair_twist_uses_the_cone_recovery_ray_at_the_antipode() {
        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        for free in 0..3 {
            let (p, b1) = (axes[free], axes[(free + 2) % 3]);
            for angle in [-3.4, -0.3, 0.3, 3.4] {
                let cut = Quat::from_axis_angle(b1, std::f64::consts::PI)
                    * Quat::from_axis_angle(p, angle);
                for sign in [-1., 1.] {
                    let relative = cut * sign;
                    let (twist, gradient) = pair_twist(relative, free);
                    assert!(
                        (interval_angle(twist, angle - 0.2, angle + 0.2) - angle).abs() < 1e-12
                    );
                    assert!((gradient + p).length() < 1e-12);
                    for axis in axes {
                        let h = 1e-7;
                        let (after, _) =
                            pair_twist(Quat::from_scaled_axis(axis * h) * relative, free);
                        let (before, _) =
                            pair_twist(Quat::from_scaled_axis(-axis * h) * relative, free);
                        let difference = (after - before) / (2. * h);
                        assert!((difference - gradient.dot(axis)).abs() < 1e-8);
                    }
                    let recovered = Quat::from_axis_angle(b1, -0.01) * relative;
                    let (after, _) = pair_twist(recovered, free);
                    assert!(
                        (interval_angle(after, angle - 0.2, angle + 0.2) - angle).abs() < 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn authored_intervals_select_equivalent_charts_across_poles_and_angle_wraps() {
        for y in [-1.7, -1.5, 1.5, 1.7] {
            for z in [-3.4, -0.4, 3.4] {
                let angles = Vec3::new(0.3, y, z);
                let rotation = Quat::from_euler(glam::EulerRot::XYZ, angles.x, angles.y, angles.z);
                let lower = angles - Vec3::splat(0.2);
                let upper = angles + Vec3::splat(0.2);
                let actual = geometry(rotation, lower, upper);
                let antipodal = geometry(-rotation, lower, upper);
                assert!((actual.angles - angles).length() < 1e-12);
                assert_eq!(actual.angles, antipodal.angles);
                let restored = Quat::from_euler(
                    glam::EulerRot::XYZ,
                    actual.angles.x,
                    actual.angles.y,
                    actual.angles.z,
                );
                assert!((rotation.dot(restored).abs() - 1.).abs() < 1e-12);
                for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                    let h = 1e-6;
                    let after = geometry(Quat::from_scaled_axis(axis * h) * rotation, lower, upper);
                    let before =
                        geometry(Quat::from_scaled_axis(-axis * h) * rotation, lower, upper);
                    let difference = (after.angles - before.angles) / (2. * h);
                    let expected = Vec3::from_array(std::array::from_fn(|k| {
                        actual.gradients[k].dot(axis) / actual.scales[k]
                    }));
                    assert!(
                        (difference - expected).length() < 1e-7,
                        "angles={angles:?} axis={axis:?}: {difference:?} != {expected:?}"
                    );
                }
            }
        }
    }
}
