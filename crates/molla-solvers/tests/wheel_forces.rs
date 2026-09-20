use std::sync::Arc;

use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, ColliderHandle, JointDesc, JointHandle,
};
use molla_sim::{BodyParams, Heightfield, JointTargetMode, JointType, MassProperties, TriMesh};
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::wheel_forces::{FrictionGrid, WheelDesc};
use molla_vehicle::{RigidTire, TireModel};

#[path = "wheel_forces/sleep.rs"]
mod sleep;

#[path = "wheel_forces/pose.rs"]
mod pose;

fn body(w: &mut RigidWorld, mass: f64) -> BodyHandle {
    w.scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(Vec3::Y * 0.49)),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn fixture(
    constrained: bool,
) -> (
    RigidWorld,
    BodyHandle,
    BodyHandle,
    JointHandle,
    ColliderHandle,
) {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let chassis = body(&mut w, 100.0);
    let wheel = body(&mut w, 10.0);
    if constrained {
        w.scene
            .insert_joint(JointDesc {
                parent: None,
                child: chassis,
                kind: JointType::Prismatic,
                frame_parent: Transform::from_translation(Vec3::Y * 0.49),
                frame_child: Transform::IDENTITY,
                axis: Vec3::X,
            })
            .unwrap();
    }
    let spin = w
        .scene
        .insert_joint(JointDesc {
            parent: Some(chassis),
            child: wheel,
            kind: JointType::Revolute,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: -Vec3::Z,
        })
        .unwrap();
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    collider.parent = Some(wheel);
    w.scene.insert_collider(collider).unwrap();
    let ground = w
        .scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Heightfield(
            Heightfield::from_scaled(2, 3, vec![0.0; 6], Vec3::new(20.0, 1.0, 10.0)).unwrap(),
        )))
        .unwrap();
    w.wheels.register_ground(&w.scene, ground, None).unwrap();
    w.wheels
        .insert(&w.scene, descriptor(chassis, wheel, spin))
        .unwrap();
    (w, chassis, wheel, spin, ground)
}

fn descriptor(chassis: BodyHandle, wheel: BodyHandle, spin: JointHandle) -> WheelDesc {
    WheelDesc {
        body: wheel,
        spin_joint: spin,
        heading_body: chassis,
        local_heading: Vec3::X,
        local_hub: Vec3::ZERO,
        spin_sign: 1.0,
        radius: 0.5,
        normal_stiffness: 100_000.0,
        normal_damping: 0.0,
        tire: Arc::new(RigidTire {
            c_kappa: 1000.0,
            c_alpha: 1000.0,
        }),
    }
}

#[test]
fn tyre_force_accelerates_the_whole_articulation_without_double_contact() {
    let (mut w, chassis, wheel, _, _) = fixture(false);
    w.solver.substeps = 1;
    w.step(0.001).unwrap();
    let samples = w.wheels.samples(&w.scene);
    assert_eq!(samples.len(), 1);
    assert!((samples[0].output.fz - 1000.0).abs() < 1e-8);
    for body in [chassis, wheel] {
        assert!(
            (w.scene.body_velocity(body).unwrap().linear.y - 1000.0 / 110.0 * 0.001).abs() < 1e-9
        );
    }
    assert!(w.contact_pairs().is_empty());
    assert!(
        w.scene
            .state()
            .body_f
            .host()
            .unwrap()
            .iter()
            .all(|f| *f == SpatialVector::ZERO)
    );
}

#[test]
fn contact_force_reacts_against_wheel_spin_without_a_second_spin_integrator() {
    let (mut w, chassis, _, spin, _) = fixture(true);
    let joint = w.scene.joint_index(spin).unwrap().index();
    let offset = w.scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
    w.scene.state_mut().joint_qd.host_mut().unwrap()[offset] = 10.0;
    w.solver.substeps = 1;
    w.step(0.001).unwrap();
    let force = w.wheels.samples(&w.scene)[0].output.force[0];
    assert!((force - 500.0).abs() < 1e-8);
    assert!(
        (w.scene.body_velocity(chassis).unwrap().linear.x - force / 110.0 * 0.001).abs() < 1e-9
    );
    assert!(
        (w.scene.state().joint_qd.host().unwrap()[offset] - (10.0 - force * 0.5 * 0.001)).abs()
            < 1e-9
    );
}

#[test]
fn tyre_friction_stops_slip_without_reversing_it() {
    for dt in [0.005, 0.01, 0.1] {
        for direction in [-1.0, 1.0] {
            let (mut w, chassis, _, spin, _) = fixture(true);
            let joint = w.scene.joint_index(spin).unwrap().index();
            let offset = w.scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
            let initial_spin = 0.01 * direction;
            w.scene.state_mut().joint_qd.host_mut().unwrap()[offset] = initial_spin;
            w.solver.substeps = 1;
            w.step(dt).unwrap();
            let sample = &w.wheels.samples(&w.scene)[0];
            let impulse = sample.output.force[0] * dt;
            let expected = initial_spin * 0.5 / (1.0 / 110.0 + 0.25);
            assert!((impulse - expected).abs() < 1e-10);
            assert!((sample.output.fz - 1000.0).abs() < 1e-8);
            let speed = w.scene.body_velocity(chassis).unwrap().linear.x;
            let spin = w.scene.state().joint_qd.host().unwrap()[offset];
            assert!((speed - impulse / 110.0).abs() < 1e-10);
            assert!((spin - initial_spin + 0.5 * impulse).abs() < 1e-10);
            assert!((spin * 0.5 - speed).abs() < 1e-10);
            assert!(110.0 * speed * speed + spin * spin < initial_spin * initial_spin);
        }
    }
}

#[test]
fn joint_drive_torque_spins_the_wheel_and_drives_the_carrier() {
    let (mut w, chassis, wheel, spin, _) = fixture(true);
    w.scene
        .set_joint_motor(
            spin,
            JointAxis::AngX,
            JointMotor {
                mode: JointTargetMode::Effort,
                target_position: 30.0,
                max_force: 30.0,
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..120 {
        w.step(1.0 / 120.0).unwrap();
    }
    assert!(w.scene.body_pose(chassis).unwrap().position.x > 0.1);
    assert!(w.scene.body_velocity(chassis).unwrap().linear.x > 0.1);
    assert!(w.scene.body_velocity(wheel).unwrap().angular.z < -0.1);
    assert_eq!(w.wheels.samples(&w.scene).len(), 8);
    assert!(
        w.wheels
            .samples(&w.scene)
            .iter()
            .all(|s| s.output.in_contact && s.output.slip_ratio > 0.0)
    );
    assert!(w.quarantined_bodies().is_empty());
}

#[test]
fn straight_velocity_drive_with_traction_control_accelerates() {
    let (mut w, chassis, wheel, spin, _) = fixture(true);
    let mut desc = descriptor(chassis, wheel, spin);
    desc.tire = Arc::new(molla_vehicle::TmSimpleTire {
        c_kappa: 10_000.0,
        c_alpha: 8_000.0,
        pneumatic_trail: 0.015,
    });
    w.wheels.insert(&w.scene, desc).unwrap();
    for _ in 0..120 {
        let speed = w.scene.body_velocity(chassis).unwrap().linear.x;
        let surface_speed = 2.0_f64.min(speed + 0.08 * speed.abs() + 0.15).max(0.0);
        w.scene
            .set_joint_motor(
                spin,
                JointAxis::AngX,
                JointMotor {
                    mode: JointTargetMode::Velocity,
                    target_velocity: surface_speed / 0.5,
                    damping: 2250.0,
                    max_force: 225.0,
                    ..Default::default()
                },
            )
            .unwrap();
        w.step(1.0 / 120.0).unwrap();
    }
    let speed = w.scene.body_velocity(chassis).unwrap().linear.x;
    assert!(speed > 1.0, "straight speed {speed}");
}

#[test]
fn tractor_mass_mixed_wheels_accelerate_straight() {
    for substeps in [4, 8, 16] {
        for direction in [-1.0, 1.0] {
            let speed = tractor_straight_speed(substeps, direction);
            assert!(
                (1.8..2.2).contains(&speed),
                "tractor straight speed {speed}, substeps {substeps}, direction {direction}"
            );
        }
    }
}

fn tractor_straight_speed(substeps: usize, direction: f64) -> f64 {
    let mut w = RigidWorld::new().unwrap();
    w.solver.substeps = substeps;
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let chassis = body(&mut w, 4397.0);
    w.scene
        .insert_joint(JointDesc {
            parent: None,
            child: chassis,
            kind: JointType::Prismatic,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let mut ground_desc = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(100.0, 0.01, 10.0),
    });
    ground_desc.friction = 1.4;
    let ground = w.scene.insert_collider(ground_desc).unwrap();
    w.wheels.register_ground(&w.scene, ground, None).unwrap();
    let mut wheels = Vec::new();
    let load = 4917.0 * 9.81 / 4.0;
    for (i, radius) in [0.6, 0.6, 0.85, 0.85].into_iter().enumerate() {
        let wheel = body(&mut w, 130.0);
        w.scene
            .set_body_mass_properties(
                wheel,
                MassProperties {
                    mass: 130.0,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 15.6,
                },
            )
            .unwrap();
        let spin = w
            .scene
            .insert_joint(JointDesc {
                parent: Some(chassis),
                child: wheel,
                kind: JointType::Revolute,
                axis: -Vec3::Z,
                frame_parent: Transform::from_translation(Vec3::new(
                    if i < 2 { 1.4 } else { -1.4 },
                    radius * 0.96 + 0.01,
                    if i % 2 == 0 { 1.0 } else { -1.0 },
                )),
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        let mut desc = descriptor(chassis, wheel, spin);
        desc.radius = radius;
        desc.normal_stiffness = load / (0.04 * radius);
        desc.tire = Arc::new(molla_vehicle::TmSimpleTire {
            c_kappa: 10.0 * load,
            c_alpha: 8.0 * load,
            pneumatic_trail: 0.03 * radius,
        });
        w.wheels.insert(&w.scene, desc).unwrap();
        wheels.push((wheel, spin, radius));
    }
    for _ in 0..600 {
        let speed = direction * w.scene.body_velocity(chassis).unwrap().linear.x;
        for &(_, spin, radius) in &wheels {
            let torque = (0.9 * 1.4 * load * radius).min(6000.0);
            w.scene
                .set_joint_motor(
                    spin,
                    JointAxis::AngX,
                    JointMotor {
                        mode: JointTargetMode::Velocity,
                        target_velocity: direction
                            * 2.0_f64.min(speed + 0.08 * speed.abs() + 0.15).max(0.0)
                            / radius,
                        damping: torque / 0.1,
                        max_force: torque,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        w.step(1.0 / 120.0).unwrap();
    }
    direction * w.scene.body_velocity(chassis).unwrap().linear.x
}

#[test]
fn cell_friction_and_pair_filters_control_tyre_forces() {
    let (mut w, chassis, wheel, spin, ground) = fixture(true);
    w.wheels
        .register_ground(
            &w.scene,
            ground,
            Some(FrictionGrid {
                origin: [-10.0, -5.0],
                cell_size: [10.0, 10.0],
                cols: 2,
                rows: 1,
                values: vec![0.1, 0.8],
            }),
        )
        .unwrap();
    let offset = w.scene.model().joint_dof_offset.host().unwrap()
        [w.scene.joint_index(spin).unwrap().index()] as usize;
    w.scene.state_mut().joint_qd.host_mut().unwrap()[offset] = 100.0;
    w.solver.substeps = 1;
    w.step(0.0001).unwrap();
    let sample = &w.wheels.samples(&w.scene)[0];
    assert_eq!(sample.friction, 0.8);
    assert!((sample.output.force[0] - 800.0).abs() < 1e-8);
    w.scene
        .set_body_pose(
            chassis,
            Transform::from_translation(Vec3::new(-1.0, 0.49, 0.0)),
        )
        .unwrap();
    w.step(0.0001).unwrap();
    assert_eq!(w.wheels.samples(&w.scene)[0].friction, 0.1);
    w.step_filtered(0.0001, |a, b| a == Some(wheel) && b.is_none())
        .unwrap();
    assert!(w.wheels.samples(&w.scene).is_empty());
}

#[test]
fn wheel_ground_mask_preserves_obstacle_contacts_and_restores_on_removal() {
    let (mut w, _, wheel, _, ground) = fixture(true);
    let mut obstacle = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.2 });
    obstacle.pose = Transform::from_translation(Vec3::new(0.55, 0.49, 0.0));
    let obstacle = w.scene.insert_collider(obstacle).unwrap();
    w.step(1e-5).unwrap();
    assert!(
        w.contact_pairs()
            .iter()
            .any(|p| p.colliders.contains(&obstacle))
    );
    assert!(
        !w.contact_pairs()
            .iter()
            .any(|p| p.colliders.contains(&ground))
    );
    w.wheels.remove(wheel);
    w.step(1e-5).unwrap();
    assert!(
        w.contact_pairs()
            .iter()
            .any(|p| p.colliders.contains(&ground))
    );
    assert!(w.wheels.samples(&w.scene).is_empty());
}

#[test]
fn triangle_mesh_surface_and_disabled_terrain_have_live_readout() {
    let (mut w, _, _, _, ground) = fixture(true);
    w.wheels.remove_ground(ground);
    w.scene.remove_collider(ground).unwrap();
    let mesh = w
        .scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::TriMesh(TriMesh {
            vertices: vec![
                Vec3::new(-5.0, 0.0, -5.0),
                Vec3::new(5.0, 0.0, -5.0),
                Vec3::new(0.0, 0.0, 5.0),
            ],
            indices: vec![[0, 2, 1]],
        })))
        .unwrap();
    w.wheels.register_ground(&w.scene, mesh, None).unwrap();
    w.step(0.001).unwrap();
    assert_eq!(w.wheels.samples(&w.scene)[0].ground, mesh);
    let mut desc = w.scene.collider(mesh).unwrap().clone();
    desc.enabled = false;
    w.scene.replace_collider(mesh, desc).unwrap();
    assert!(w.wheels.samples(&w.scene).is_empty());
    w.step(0.001).unwrap();
    assert!(w.wheels.samples(&w.scene).is_empty());
}

struct InvalidTire;
impl TireModel for InvalidTire {
    fn evaluate(&self, _: f64, _: f64, _: f64, _: f64) -> (f64, f64, f64) {
        (f64::NAN, 0.0, 0.0)
    }
}

#[test]
fn invalid_parameters_and_force_failures_are_atomic() {
    let (mut w, chassis, wheel, spin, _) = fixture(true);
    let mut desc = descriptor(chassis, wheel, spin);
    desc.radius = f64::NAN;
    assert!(w.wheels.insert(&w.scene, desc).is_err());
    w.step(0.001).unwrap();
    assert!(!w.wheels.samples(&w.scene).is_empty());
    let mut desc = descriptor(chassis, wheel, spin);
    desc.tire = Arc::new(InvalidTire);
    w.wheels.insert(&w.scene, desc).unwrap();
    let before = w.scene.body_pose(wheel).unwrap();
    let velocity = w.scene.body_velocity(wheel).unwrap();
    assert!(w.step(0.001).is_err());
    assert_eq!(w.scene.body_pose(wheel).unwrap(), before);
    assert_eq!(w.scene.body_velocity(wheel).unwrap(), velocity);
    assert!(w.wheels.samples(&w.scene).is_empty());
}

#[test]
fn stable_wheels_survive_rebuilds_and_prune_removed_bodies() {
    let (mut w, _, wheel, _, _) = fixture(true);
    let unrelated = body(&mut w, 5.0);
    w.scene.remove_body(unrelated).unwrap();
    let collider = w.scene.body_colliders(wheel).next().unwrap();
    let descriptor = w.scene.collider(collider).unwrap().clone();
    w.scene.remove_collider(collider).unwrap();
    w.scene.insert_collider(descriptor).unwrap();
    w.step(0.001).unwrap();
    assert!(w.contact_pairs().is_empty());
    assert!(w.wheels.samples(&w.scene).iter().all(|s| s.wheel == wheel));
    w.scene.remove_body(wheel).unwrap();
    w.step(0.001).unwrap();
    assert!(w.wheels.samples(&w.scene).is_empty());
}

fn pressure_config() -> molla_solvers::pressure_tire::PressureTireConfig {
    molla_solvers::pressure_tire::PressureTireConfig {
        tire: molla_vehicle::real_tire::RealTireParams {
            radius: 0.5,
            width: 0.3,
            pressure_pa: 180_000.0,
            ..Default::default()
        },
        min_pressure_pa: 50_000.0,
        max_pressure_pa: 400_000.0,
        pressure_rate_pa_s: 10_000.0,
        supported_mass: 110.0,
        damping_ratio: 0.7,
        hysteresis_fraction: 0.1,
    }
}

#[test]
fn pressure_normal_support_replaces_the_legacy_spring() {
    let (mut w, chassis, wheel, _, _) = fixture(false);
    w.wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    w.solver.substeps = 1;
    w.step(0.0001).unwrap();
    let sample = &w.wheels.samples(&w.scene)[0];
    let pressure = sample.pressure.as_ref().unwrap();
    let normal = pressure.cells.iter().map(|c| c.normal_load).sum::<f64>();
    assert!((sample.output.fz - normal).abs() < 1e-9);
    assert!((normal - 1000.0).abs() > 1.0);
    assert!((pressure.loaded_radius - 0.49).abs() < 1e-10);
    for body in [chassis, wheel] {
        assert!(
            (w.scene.body_velocity(body).unwrap().linear.y - normal / 110.0 * 0.0001).abs() < 1e-10
        );
    }
    assert!(w.contact_pairs().is_empty());
}

#[test]
fn pressure_target_edit_preserves_actual_contact_until_stepping() {
    let (mut w, _, wheel, _, _) = fixture(true);
    w.wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    w.step(0.01).unwrap();
    let before = w.wheels.samples(&w.scene).last().unwrap().clone();
    w.wheels
        .set_pressures(&w.scene, &[(wheel, 220_000.0)])
        .unwrap();
    let sample = w.wheels.samples(&w.scene).last().unwrap();
    assert_eq!(sample.output.fz, before.output.fz);
    assert_eq!(
        sample.pressure.as_ref().unwrap().deflection,
        before.pressure.unwrap().deflection
    );
    assert_eq!(
        sample.pressure.as_ref().unwrap().target_pressure_pa,
        180_000.0
    );
    let state = w.wheels.pressure_tire(&w.scene, wheel).unwrap();
    assert_eq!(state.pressure_pa, 180_000.0);
    assert_eq!(state.target_pressure_pa, 220_000.0);
}

#[test]
fn pressure_targets_are_atomic_rate_limited_and_survive_rebuilds() {
    let (mut w, chassis, wheel, spin, _) = fixture(true);
    w.wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    for target in [f64::NAN, -1.0, 500_000.0] {
        assert!(
            w.wheels
                .set_pressures(&w.scene, &[(wheel, target)])
                .is_err()
        );
    }
    assert!(
        w.wheels
            .set_pressures(&w.scene, &[(wheel, 200_000.0), (chassis, 200_000.0)])
            .is_err()
    );
    assert!(
        w.wheels
            .set_pressures(&w.scene, &[(wheel, 200_000.0), (wheel, 250_000.0)])
            .is_err()
    );
    assert_eq!(
        w.wheels
            .pressure_tire(&w.scene, wheel)
            .unwrap()
            .target_pressure_pa,
        180_000.0
    );
    let mut invalid = pressure_config();
    invalid.tire.pressure_pa = f64::NAN;
    assert!(w.wheels.configure_pressure_tire(wheel, invalid).is_err());
    let mut invalid = pressure_config();
    invalid.min_pressure_pa = 190_000.0;
    invalid.tire.pressure_pa = 200_000.0;
    assert!(w.wheels.configure_pressure_tire(wheel, invalid).is_err());
    let mut invalid = pressure_config();
    invalid.tire.radius = 0.6;
    assert!(w.wheels.configure_pressure_tire(wheel, invalid).is_err());
    assert_eq!(
        w.wheels
            .pressure_tire(&w.scene, wheel)
            .unwrap()
            .config
            .tire
            .radius,
        0.5
    );
    w.wheels
        .set_pressures(&w.scene, &[(wheel, 200_000.0)])
        .unwrap();
    assert_eq!(
        w.wheels.pressure_tire(&w.scene, wheel).unwrap().pressure_pa,
        180_000.0
    );
    w.step(0.01).unwrap();
    assert!(
        (w.wheels.pressure_tire(&w.scene, wheel).unwrap().pressure_pa - 180_100.0).abs() < 1e-8
    );
    let samples = w.wheels.samples(&w.scene);
    assert_eq!(samples.len(), 8);
    assert!(samples.iter().all(|s| s.pressure.is_some()));
    assert!(w.contact_pairs().is_empty());
    let extra = body(&mut w, 1.0);
    w.scene.remove_body(extra).unwrap();
    w.wheels
        .insert(&w.scene, descriptor(chassis, wheel, spin))
        .unwrap();
    w.wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    assert!(
        (w.wheels.pressure_tire(&w.scene, wheel).unwrap().pressure_pa - 180_100.0).abs() < 1e-8
    );
    assert_eq!(
        w.wheels
            .pressure_tire(&w.scene, wheel)
            .unwrap()
            .target_pressure_pa,
        200_000.0
    );
    w.scene.remove_body(wheel).unwrap();
    assert!(w.wheels.pressure_tire(&w.scene, wheel).is_none());
    assert!(
        w.wheels
            .set_pressures(&w.scene, &[(wheel, 200_000.0)])
            .is_err()
    );
    w.step(0.01).unwrap();
    assert!(w.wheels.remove_pressure_tire(wheel).is_none());
}

#[test]
fn failed_step_rolls_back_pressure_and_rigid_state() {
    let (mut w, _, wheel, _, _) = fixture(true);
    w.wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    w.wheels
        .set_pressures(&w.scene, &[(wheel, 200_000.0)])
        .unwrap();
    let pose = w.scene.body_pose(wheel).unwrap();
    let velocity = w.scene.body_velocity(wheel).unwrap();
    let mut substeps = 0;
    assert!(
        w.step_with_forces(
            0.01,
            |_, _| false,
            |_, _, _, _| {
                substeps += 1;
                if substeps == 4 {
                    Err(molla_core::Error::Solver("injected failure".into()))
                } else {
                    Ok(())
                }
            }
        )
        .is_err()
    );
    assert_eq!(substeps, 4);
    assert_eq!(
        w.wheels.pressure_tire(&w.scene, wheel).unwrap().pressure_pa,
        180_000.0
    );
    assert_eq!(w.scene.body_pose(wheel).unwrap(), pose);
    assert_eq!(w.scene.body_velocity(wheel).unwrap(), velocity);
    assert!(w.wheels.samples(&w.scene).is_empty());
}

#[test]
fn pressure_tyres_drive_in_both_directions_during_inflation() {
    for direction in [-1.0, 1.0] {
        let (mut w, chassis, wheel, spin, _) = fixture(true);
        w.wheels
            .configure_pressure_tire(wheel, pressure_config())
            .unwrap();
        w.wheels
            .set_pressures(&w.scene, &[(wheel, 220_000.0)])
            .unwrap();
        w.scene
            .set_joint_motor(
                spin,
                JointAxis::AngX,
                JointMotor {
                    mode: JointTargetMode::Velocity,
                    target_velocity: direction * 2.0,
                    damping: 1000.0,
                    max_force: 100.0,
                    ..Default::default()
                },
            )
            .unwrap();
        for _ in 0..240 {
            w.step(1.0 / 120.0).unwrap();
        }
        let speed = direction * w.scene.body_velocity(chassis).unwrap().linear.x;
        assert!((0.7..1.1).contains(&speed), "pressure tyre speed {speed}");
        assert!(
            (w.wheels.pressure_tire(&w.scene, wheel).unwrap().pressure_pa - 200_000.0).abs() < 1e-6
        );
        assert!(w.quarantined_bodies().is_empty());
        assert!(w.contact_pairs().is_empty());
    }
}

fn shear_fixture() -> (
    RigidWorld,
    BodyHandle,
    BodyHandle,
    JointHandle,
    ColliderHandle,
) {
    let (mut w, chassis, wheel, spin, ground) = fixture(true);
    w.solver.substeps = 1;
    w.wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    let joint = w.scene.joint_index(spin).unwrap().index();
    let offset = w.scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
    w.scene.state_mut().joint_qd.host_mut().unwrap()[offset] = 0.2;
    w.step(0.001).unwrap();
    w.scene
        .set_body_velocity(chassis, SpatialVector::ZERO)
        .unwrap();
    w.scene
        .set_body_velocity(wheel, SpatialVector::ZERO)
        .unwrap();
    (w, chassis, wheel, spin, ground)
}

#[test]
fn pressure_shear_survives_unrelated_rebuild_and_failed_steps() {
    let (mut reference, _, _, _, _) = shear_fixture();
    reference.step(0.001).unwrap();
    let expected = reference.wheels.samples(&reference.scene)[0].output.force[0];
    assert!(
        expected > 0.1,
        "static shear must survive zero spin: {expected}"
    );
    let (mut w, chassis, wheel, spin, _) = shear_fixture();
    let extra = body(&mut w, 1.0);
    w.scene.remove_body(extra).unwrap();
    w.wheels
        .insert(&w.scene, descriptor(chassis, wheel, spin))
        .unwrap();
    w.wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    assert!(
        w.step_with_forces(
            0.001,
            |_, _| false,
            |_, _, _, _| Err(molla_core::Error::Solver("injected shear failure".into()))
        )
        .is_err()
    );
    w.step(0.001).unwrap();
    let actual = w.wheels.samples(&w.scene)[0].output.force[0];
    assert!(
        (actual - expected).abs() < 1e-10,
        "shear changed across rebuild/rollback {actual} != {expected}"
    );
}

#[test]
fn pressure_shear_clears_on_teleport_and_ground_loss() {
    for teleport in [false, true] {
        let (mut w, chassis, wheel, _, ground) = shear_fixture();
        if teleport {
            let poses: Vec<_> = [chassis, wheel]
                .into_iter()
                .map(|id| {
                    let mut pose = w.scene.body_pose(id).unwrap();
                    pose.position.x += 1.0;
                    (id, pose)
                })
                .collect();
            w.scene.set_body_poses(&poses, true).unwrap();
        } else {
            w.wheels.remove_ground(ground);
            w.step(0.001).unwrap();
            w.wheels.register_ground(&w.scene, ground, None).unwrap();
            w.scene.set_body_velocity(chassis, SpatialVector::ZERO).unwrap();
            w.scene.set_body_velocity(wheel, SpatialVector::ZERO).unwrap();
        }
        w.step(0.001).unwrap();
        let force = w.wheels.samples(&w.scene)[0].output.force[0];
        assert!(force.abs() < 1e-10, "teleport={teleport} residual force={force}");
    }
}
