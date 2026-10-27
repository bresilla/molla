//! CPU reduced-oracle tests for MPM rigid two-way coupling.
//!
//! The full MPM implementation is GPU-owned, but the cell-local
//! impulse formula should be provable in default CPU tests.

use molla_math::Vec3;
use molla_solvers::{
    MpmRigidBodySample, MpmRigidCellSample, aggregate_rigid_cell_wrench, rigid_body_cell_sample,
    rigid_body_velocity_at_cell, rigid_cell_imprint_wrench,
};

#[test]
fn descending_rigid_into_stationary_sand_gets_upward_reaction() {
    let dt = 0.01;
    let sample = MpmRigidCellSample {
        cell_center: Vec3::new(0.0, 0.0, 0.0),
        sand_mass: 0.25,
        sand_momentum: Vec3::ZERO,
        rigid_center: Vec3::new(0.0, 0.0, 0.0),
        rigid_velocity: Vec3::new(0.0, -2.0, 0.0),
    };

    let wrench = rigid_cell_imprint_wrench(sample, dt);

    // J_sand = m * v = (0, -0.5, 0) N·s, so the rigid gets the
    // opposite impulse over dt = 0.01s: +50N along Y.
    assert!((wrench.force.y - 50.0).abs() < 1.0e-12);
    assert_eq!(wrench.force.x, 0.0);
    assert_eq!(wrench.force.z, 0.0);
    assert_eq!(wrench.torque, Vec3::ZERO);
}

#[test]
fn imprint_formula_conserves_pair_impulse() {
    let dt = 0.02;
    let sample = MpmRigidCellSample {
        cell_center: Vec3::new(0.1, 0.2, -0.3),
        sand_mass: 0.4,
        sand_momentum: Vec3::new(0.3, -0.1, 0.2),
        rigid_center: Vec3::new(-0.2, 0.0, 0.1),
        rigid_velocity: Vec3::new(-0.5, 1.0, 0.25),
    };

    let wrench = rigid_cell_imprint_wrench(sample, dt);
    let impulse_to_rigid = wrench.force * dt;
    let impulse_to_sand = sample.rigid_velocity * sample.sand_mass - sample.sand_momentum;

    assert!(
        (impulse_to_rigid + impulse_to_sand).length() < 1.0e-12,
        "rigid/sand impulse pair is not equal-and-opposite"
    );
}

#[test]
fn off_center_cell_produces_expected_torque() {
    let dt = 0.05;
    let sample = MpmRigidCellSample {
        cell_center: Vec3::new(1.0, 0.0, 0.0),
        sand_mass: 0.5,
        sand_momentum: Vec3::ZERO,
        rigid_center: Vec3::ZERO,
        rigid_velocity: Vec3::new(0.0, -1.0, 0.0),
    };

    let wrench = rigid_cell_imprint_wrench(sample, dt);

    // Force on rigid is +10Y. Arm is +X, so torque = X × Y = +Z.
    assert!((wrench.force.y - 10.0).abs() < 1.0e-12);
    assert!((wrench.torque.z - 10.0).abs() < 1.0e-12);
    assert_eq!(wrench.torque.x, 0.0);
    assert_eq!(wrench.torque.y, 0.0);
}

#[test]
fn batched_imprint_conserves_total_linear_and_angular_impulse() {
    let body = MpmRigidBodySample {
        rigid_center: Vec3::new(0.2, -0.1, 0.3),
        linear_velocity: Vec3::new(-0.4, 1.1, 0.25),
        angular_velocity: Vec3::new(0.7, -0.3, 1.4),
    };
    let dt = 0.015;
    let samples = [
        rigid_body_cell_sample(
            body,
            Vec3::new(0.9, -0.2, 0.4),
            0.25,
            Vec3::new(0.10, 0.20, -0.05),
        ),
        rigid_body_cell_sample(
            body,
            Vec3::new(-0.4, 0.1, 0.8),
            0.40,
            Vec3::new(-0.15, 0.05, 0.12),
        ),
        rigid_body_cell_sample(
            body,
            Vec3::new(0.3, -0.8, -0.2),
            0.35,
            Vec3::new(0.08, -0.20, 0.03),
        ),
        rigid_body_cell_sample(
            body,
            Vec3::new(0.0, 0.5, 0.0),
            0.18,
            Vec3::new(-0.02, 0.04, -0.09),
        ),
    ];

    let wrench = aggregate_rigid_cell_wrench(&samples, dt);
    let impulse_to_rigid = wrench.force * dt;
    let angular_impulse_to_rigid = wrench.torque * dt;
    let mut impulse_to_sand = Vec3::ZERO;
    let mut angular_impulse_to_sand = Vec3::ZERO;
    for sample in &samples {
        let j_sand = sample.rigid_velocity * sample.sand_mass - sample.sand_momentum;
        impulse_to_sand += j_sand;
        angular_impulse_to_sand += (sample.cell_center - sample.rigid_center).cross(j_sand);
    }

    assert!(
        (impulse_to_rigid + impulse_to_sand).length() < 1.0e-12,
        "batched rigid/sand linear impulse is not conserved: rigid={impulse_to_rigid:?}, sand={impulse_to_sand:?}"
    );
    assert!(
        (angular_impulse_to_rigid + angular_impulse_to_sand).length() < 1.0e-12,
        "batched rigid/sand angular impulse is not conserved: rigid={angular_impulse_to_rigid:?}, sand={angular_impulse_to_sand:?}"
    );
    assert!(
        impulse_to_rigid.length() > 1.0e-3 && angular_impulse_to_rigid.length() > 1.0e-3,
        "test fixture should exercise non-trivial linear and angular coupling"
    );
}

#[test]
fn empty_or_invalid_cell_contributes_no_wrench() {
    let sample = MpmRigidCellSample {
        cell_center: Vec3::new(1.0, 0.0, 0.0),
        sand_mass: 0.0,
        sand_momentum: Vec3::new(9.0, 9.0, 9.0),
        rigid_center: Vec3::ZERO,
        rigid_velocity: Vec3::new(0.0, -1.0, 0.0),
    };

    assert_eq!(rigid_cell_imprint_wrench(sample, 0.01).force, Vec3::ZERO);
    assert_eq!(rigid_cell_imprint_wrench(sample, 0.0).force, Vec3::ZERO);
}

#[test]
fn spinning_rigid_symmetric_cells_produce_pure_spin_damping_torque() {
    let body = MpmRigidBodySample {
        rigid_center: Vec3::ZERO,
        linear_velocity: Vec3::ZERO,
        angular_velocity: Vec3::Z * 3.0,
    };
    let dt = 0.02;
    let mass = 0.4;
    let cells = [
        rigid_body_cell_sample(body, Vec3::X, mass, Vec3::ZERO),
        rigid_body_cell_sample(body, -Vec3::X, mass, Vec3::ZERO),
    ];

    assert_eq!(
        rigid_body_velocity_at_cell(body, Vec3::X),
        Vec3::new(0.0, 3.0, 0.0)
    );
    assert_eq!(
        rigid_body_velocity_at_cell(body, -Vec3::X),
        Vec3::new(0.0, -3.0, 0.0)
    );

    let wrench = aggregate_rigid_cell_wrench(&cells, dt);

    // Symmetric cells have equal-and-opposite tangential forces, so net linear
    // force cancels. Both torques oppose +Z spin:
    // each cell contributes |r × F| = m * |ω × r| / dt = 0.4 * 3 / 0.02 = 60.
    assert!(
        wrench.force.length() < 1.0e-12,
        "symmetric spin imprint should have zero net force: {:?}",
        wrench.force
    );
    assert!((wrench.torque.z + 120.0).abs() < 1.0e-12);
    assert_eq!(wrench.torque.x, 0.0);
    assert_eq!(wrench.torque.y, 0.0);

    let spin_impulse_to_rigid = wrench.torque * dt;
    assert!(
        spin_impulse_to_rigid.z < 0.0,
        "torque impulse should oppose positive spin: {spin_impulse_to_rigid:?}"
    );
}

#[test]
fn co_moving_sand_and_rigid_cells_produce_zero_wrench() {
    let body = MpmRigidBodySample {
        rigid_center: Vec3::new(0.2, -0.1, 0.3),
        linear_velocity: Vec3::new(1.5, -0.25, 0.75),
        angular_velocity: Vec3::new(0.4, -0.8, 1.2),
    };
    let dt = 0.01;
    let mass = 0.3;
    let cell_centers = [
        Vec3::new(0.8, -0.1, 0.3),
        Vec3::new(-0.4, -0.1, 0.3),
        Vec3::new(0.2, 0.5, 0.3),
        Vec3::new(0.2, -0.7, 0.3),
        Vec3::new(0.2, -0.1, 0.9),
        Vec3::new(0.2, -0.1, -0.3),
    ];
    let samples: Vec<_> = cell_centers
        .iter()
        .map(|&cell_center| {
            let v = rigid_body_velocity_at_cell(body, cell_center);
            rigid_body_cell_sample(body, cell_center, mass, v * mass)
        })
        .collect();

    let wrench = aggregate_rigid_cell_wrench(&samples, dt);

    assert!(
        wrench.force.length() < 1.0e-12,
        "co-moving sand should not exchange net linear impulse with rigid body: {:?}",
        wrench.force
    );
    assert!(
        wrench.torque.length() < 1.0e-12,
        "co-moving sand should not exchange net angular impulse with rigid body: {:?}",
        wrench.torque
    );
}
