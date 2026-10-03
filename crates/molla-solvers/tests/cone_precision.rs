use molla_math::{Quat, Vec3};
use molla_sim::{joint_control::SphericalConeLimits, spherical_cone};

#[test]
fn canceled_normal_load_retains_tangential_velocity_in_f32() {
    let limits = SphericalConeLimits {
        angle0: Some(0.4),
        angle1: Some(0.9),
    };
    let direction = Vec3::new(0.0, 0.4, 0.9).normalize();
    let rotation = Quat::from_axis_angle(
        Vec3::X.cross(direction),
        0.4_f64.hypot(0.9) / 2.0_f64.sqrt(),
    );
    let normal = spherical_cone::evaluate(limits, Vec3::X, rotation)
        .unwrap()
        .rows[0]
        .angular
        .normalize()
        .as_vec3();
    let force = (normal.as_dvec3() * 3.0).as_vec3();
    let spin = (rotation * Vec3::X * 2.0).as_vec3();
    let dt = 1.0_f32 / (120.0 * 8.0);
    let mut split = spin;
    let mut combined = spin;
    for _ in 0..1920 {
        let predicted = split + force * dt;
        split = predicted - normal * (normal.dot(predicted) / normal.length_squared());
        let constrained_acceleration =
            force - normal * (normal.dot(force) / normal.length_squared());
        combined = combined - normal * (normal.dot(combined) / normal.length_squared())
            + constrained_acceleration * dt;
    }
    eprintln!(
        "f32 fixed-normal projection: split_error={:e} combined_error={:e}",
        (split - spin).length(),
        (combined - spin).length()
    );
    assert!((combined - spin).length() < 2e-6);
}

#[test]
fn combining_velocity_increments_limits_substep_roundoff() {
    let limits = SphericalConeLimits {
        angle0: Some(0.4_f32 as f64),
        angle1: Some(0.9_f32 as f64),
    };
    let direction = Vec3::new(0.0, 0.4, 0.9).normalize();
    let rotation = Quat::from_axis_angle(
        Vec3::X.cross(direction),
        0.4_f64.hypot(0.9) / 2.0_f64.sqrt(),
    );
    let normal = spherical_cone::evaluate(limits, Vec3::X, rotation)
        .unwrap()
        .rows[0]
        .angular
        .normalize();
    let force = (normal * 3.0).as_vec3();
    let spin = rotation * Vec3::X * 2.0;
    let dt = 1.0_f32 / (120.0 * 8.0);
    for (float_q, combined) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut q = rotation;
        let mut velocity = spin.as_vec3();
        let mut peak = 0.0_f64;
        for _ in 0..1920 {
            let cone = spherical_cone::evaluate(limits, Vec3::X, q).unwrap().rows[0];
            let gradient = cone.angular.as_vec3();
            let scale = gradient.length();
            let jacobian = -gradient / scale;
            let desired = cone.error as f32 / (scale * dt);
            let predicted = velocity + force * dt;
            if combined {
                let impulse = ((desired - jacobian.dot(velocity) - jacobian.dot(force) * dt)
                    / jacobian.length_squared())
                .max(0.0);
                velocity += force * dt + jacobian * impulse;
            } else {
                let impulse =
                    ((desired - jacobian.dot(predicted)) / jacobian.length_squared()).max(0.0);
                velocity = predicted + jacobian * impulse;
            }
            if float_q {
                let old = q.as_quat();
                let dq = glam::Quat::from_xyzw(
                    velocity.x * 0.5,
                    velocity.y * 0.5,
                    velocity.z * 0.5,
                    0.0,
                ) * old;
                q = (old + dq * dt).normalize().as_dquat();
            } else {
                let dq = Quat::from_xyzw(
                    velocity.x as f64 * 0.5,
                    velocity.y as f64 * 0.5,
                    velocity.z as f64 * 0.5,
                    0.0,
                ) * q;
                q = (q + dq * dt as f64).normalize();
            }
            peak = peak.max((velocity.as_dvec3() - spin).length());
        }
        eprintln!(
            "f32 velocity, f64 geometry: float_q={float_q} combined={combined} peak_velocity_error={peak:e} final_swing={:?}",
            spherical_cone::evaluate(limits, Vec3::X, q).unwrap().swing
        );
        assert!(peak.is_finite());
        if combined {
            assert!(peak < if float_q { 1e-4 } else { 5e-6 });
        }
    }
}
