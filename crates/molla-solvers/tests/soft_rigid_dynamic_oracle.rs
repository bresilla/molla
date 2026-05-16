//! Reduced dynamic soft-rigid coupling oracle.
//!
//! The current VBD soft-rigid solver has kinematic and SDF contact
//! gates. This file adds the CPU momentum oracle for the next dynamic
//! stage: a soft particle/contact patch and a finite-mass rigid body
//! receive equal-and-opposite contact impulses, including the rigid
//! angular impulse from an off-centre patch.

use molla_math::{Mat3, Real, Vec3};

#[derive(Copy, Clone, Debug)]
struct SoftPoint {
    mass: Real,
    position: Vec3,
    velocity: Vec3,
}

#[derive(Copy, Clone, Debug)]
struct RigidBody {
    mass: Real,
    inertia: Mat3,
    position: Vec3,
    velocity: Vec3,
    angular_velocity: Vec3,
}

fn rigid_point_velocity(rigid: &RigidBody, world_point: Vec3) -> Vec3 {
    let r = world_point - rigid.position;
    rigid.velocity + rigid.angular_velocity.cross(r)
}

fn total_linear_momentum(soft: &SoftPoint, rigid: &RigidBody) -> Vec3 {
    soft.velocity * soft.mass + rigid.velocity * rigid.mass
}

fn total_angular_momentum_about_rigid_com(soft: &SoftPoint, rigid: &RigidBody) -> Vec3 {
    let r_soft = soft.position - rigid.position;
    r_soft.cross(soft.velocity * soft.mass) + rigid.inertia * rigid.angular_velocity
}

fn apply_soft_rigid_normal_impulse(
    soft: &mut SoftPoint,
    rigid: &mut RigidBody,
    normal: Vec3,
    restitution: Real,
) -> Real {
    let n = normal.normalize();
    let r = soft.position - rigid.position;
    let inertia_inv = rigid.inertia.inverse();
    let point_vel = rigid_point_velocity(rigid, soft.position);
    let rel_n = (soft.velocity - point_vel).dot(n);
    if rel_n >= 0.0 {
        return 0.0;
    }

    let angular_term = (inertia_inv * r.cross(n)).cross(r).dot(n);
    let denom = 1.0 / soft.mass + 1.0 / rigid.mass + angular_term;
    assert!(denom.is_finite() && denom > 0.0);
    let j = -(1.0 + restitution) * rel_n / denom;
    assert!(j.is_finite() && j >= 0.0);

    let impulse = j * n;
    soft.velocity += impulse / soft.mass;
    rigid.velocity -= impulse / rigid.mass;
    rigid.angular_velocity -= inertia_inv * r.cross(impulse);
    j
}

#[test]
fn dynamic_soft_rigid_impulse_conserves_momentum_and_separates() {
    let mut soft = SoftPoint {
        mass: 0.2,
        position: Vec3::new(0.30, 0.55, 0.0),
        velocity: Vec3::new(0.10, -1.20, 0.0),
    };
    let mut rigid = RigidBody {
        mass: 2.0,
        inertia: Mat3::from_diagonal(Vec3::new(0.20, 0.25, 0.30)),
        position: Vec3::ZERO,
        velocity: Vec3::new(-0.05, 0.10, 0.0),
        angular_velocity: Vec3::new(0.0, 0.0, 0.15),
    };
    let normal = Vec3::Y;
    let restitution = 0.35;

    let p0 = total_linear_momentum(&soft, &rigid);
    let l0 = total_angular_momentum_about_rigid_com(&soft, &rigid);
    let rel0 = (soft.velocity - rigid_point_velocity(&rigid, soft.position)).dot(normal);
    assert!(rel0 < 0.0, "setup must be approaching, rel_n={rel0}");

    let impulse = apply_soft_rigid_normal_impulse(&mut soft, &mut rigid, normal, restitution);

    let p1 = total_linear_momentum(&soft, &rigid);
    let l1 = total_angular_momentum_about_rigid_com(&soft, &rigid);
    let rel1 = (soft.velocity - rigid_point_velocity(&rigid, soft.position)).dot(normal);
    eprintln!(
        "dynamic soft-rigid impulse: j={impulse:.6}, rel0={rel0:.6}, rel1={rel1:.6}, \
         Δp={:?}, Δl={:?}",
        p1 - p0,
        l1 - l0
    );

    assert!(soft.velocity.is_finite() && rigid.velocity.is_finite());
    assert!(rigid.angular_velocity.is_finite());
    assert!(impulse > 0.0);
    assert!(
        (p1 - p0).length() < 1.0e-12,
        "linear momentum not conserved: before={p0:?}, after={p1:?}"
    );
    assert!(
        (l1 - l0).length() < 1.0e-12,
        "angular momentum not conserved: before={l0:?}, after={l1:?}"
    );
    assert!(
        (rel1 + restitution * rel0).abs() < 1.0e-12,
        "restitution law mismatch: rel0={rel0}, rel1={rel1}, e={restitution}"
    );
    assert!(
        rigid.angular_velocity.z < 0.0,
        "off-centre downward contact should spin rigid body about -Z, got {:?}",
        rigid.angular_velocity
    );
}

#[test]
fn separating_soft_rigid_contact_applies_no_impulse() {
    let mut soft = SoftPoint {
        mass: 0.25,
        position: Vec3::new(0.20, 0.50, 0.0),
        velocity: Vec3::new(0.0, 0.40, 0.0),
    };
    let mut rigid = RigidBody {
        mass: 1.5,
        inertia: Mat3::from_diagonal(Vec3::splat(0.20)),
        position: Vec3::ZERO,
        velocity: Vec3::ZERO,
        angular_velocity: Vec3::ZERO,
    };
    let soft_before = soft;
    let rigid_before = rigid;

    let impulse = apply_soft_rigid_normal_impulse(&mut soft, &mut rigid, Vec3::Y, 0.0);

    assert_eq!(impulse, 0.0);
    assert_eq!(soft.velocity, soft_before.velocity);
    assert_eq!(rigid.velocity, rigid_before.velocity);
    assert_eq!(rigid.angular_velocity, rigid_before.angular_velocity);
}
