use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointDesc};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::track_forces::TrackDesc;
use molla_vehicle::track::{TrackMaterial, TrackMotor};

fn fixture(ground: bool, friction: f64) -> (RigidWorld, BodyHandle, [BodyHandle; 2]) {
    let mut w = RigidWorld::new().unwrap();
    w.scene
        .set_gravity(if ground { -Vec3::Y * 9.81 } else { Vec3::ZERO })
        .unwrap();
    if ground {
        let mut g = ColliderDesc::new(ColliderGeometry::Cuboid {
            half_extents: Vec3::new(20.0, 0.1, 20.0),
        });
        g.pose = Transform::from_translation(-Vec3::Y * 0.1);
        g.friction = friction;
        w.scene.insert_collider(g).unwrap();
    }
    let chassis = w
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(Vec3::Y * 0.2)),
            additional_mass: Some(MassProperties {
                mass: 734.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(Vec3::splat(120.0)),
            }),
            ..Default::default()
        })
        .unwrap();
    let rotors = [-1.0, 1.0].map(|side| {
        let mut belt = ColliderDesc::new(ColliderGeometry::Cuboid {
            half_extents: Vec3::new(0.09, 0.06, 0.54),
        });
        belt.parent = Some(chassis);
        belt.pose = Transform::from_translation(Vec3::new(side * 0.41, -0.08, 0.0));
        belt.friction = 0.0;
        let contact = w.scene.insert_collider(belt).unwrap();
        let offset = Vec3::new(side * 0.41, 0.28, -0.40);
        let rotor = w
            .scene
            .insert_body(BodyDesc {
                params: BodyParams::new()
                    .with_pose(Transform::from_translation(Vec3::Y * 0.2 + offset)),
                additional_mass: Some(MassProperties {
                    mass: 8.0,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::from_diagonal(Vec3::splat(0.3)),
                }),
                ..Default::default()
            })
            .unwrap();
        let joint = w
            .scene
            .insert_joint(JointDesc {
                parent: Some(chassis),
                child: rotor,
                kind: JointType::Revolute,
                frame_parent: Transform::from_translation(offset),
                frame_child: Transform::IDENTITY,
                axis: Vec3::X,
            })
            .unwrap();
        w.tracks
            .insert(
                &w.scene,
                TrackDesc {
                    carrier: chassis,
                    sprocket: rotor,
                    spin_joint: joint,
                    contact_colliders: vec![contact],
                    local_axle: Vec3::X,
                    local_forward: Vec3::Z,
                    pitch_radius: 0.111,
                    material: TrackMaterial::default(),
                    motor: TrackMotor::default(),
                },
            )
            .unwrap();
        rotor
    });
    (w, chassis, rotors)
}

fn run(w: &mut RigidWorld, count: usize) {
    for _ in 0..count {
        w.step(1.0 / 120.0).unwrap();
    }
}

#[test]
fn airborne_sprockets_spin_without_linear_propulsion() {
    let (mut w, _, rotors) = fixture(false, 0.85);
    for r in rotors {
        w.tracks.set_speed(r, 0.4).unwrap();
    }
    run(&mut w, 240);
    for r in rotors {
        let sample = w.tracks.sample(&w.scene, r).unwrap();
        assert!(sample.travel > 0.5 && sample.contacts == 0, "{sample:?}");
    }
    let mut momentum = Vec3::ZERO;
    for (id, _) in w.scene.bodies() {
        momentum += w.scene.body_velocity(id).unwrap().linear
            * w.scene.body_mass_properties(id).unwrap().mass;
    }
    assert!(momentum.length() < 1e-5, "{momentum:?}");
}

#[test]
fn grounded_sprocket_torque_produces_forward_and_reverse_motion() {
    for sign in [-1.0, 1.0] {
        let (mut w, chassis, rotors) = fixture(true, 0.85);
        run(&mut w, 120);
        let start = w.scene.body_pose(chassis).unwrap().position;
        for r in rotors {
            w.tracks.set_speed(r, sign * 0.4).unwrap();
        }
        run(&mut w, 480);
        let delta = w.scene.body_pose(chassis).unwrap().position - start;
        assert!(
            delta.z * sign > 0.4,
            "{delta:?}, {:?}",
            w.tracks.sample(&w.scene, rotors[0])
        );
        assert!(delta.x.abs() < 0.05 && delta.y.abs() < 0.05, "{delta:?}");
    }
}

#[test]
fn frictionless_surface_allows_belt_slip_without_propulsion() {
    let (mut w, chassis, rotors) = fixture(true, 0.0);
    run(&mut w, 120);
    let start = w.scene.body_pose(chassis).unwrap().position;
    for r in rotors {
        w.tracks.set_speed(r, 0.4).unwrap();
    }
    run(&mut w, 240);
    assert!((w.scene.body_pose(chassis).unwrap().position.z - start.z).abs() < 0.01);
    for r in rotors {
        let s = w.tracks.sample(&w.scene, r).unwrap();
        assert!(
            s.travel > 0.5 && s.longitudinal_force == 0.0 && s.contacts > 0,
            "{s:?}"
        );
    }
}

#[test]
fn opposed_tracks_pivot_and_stop_without_setting_body_velocity() {
    let (mut w, chassis, rotors) = fixture(true, 0.85);
    run(&mut w, 120);
    w.tracks.set_speed(rotors[0], 0.4).unwrap();
    w.tracks.set_speed(rotors[1], -0.4).unwrap();
    run(&mut w, 480);
    let v = w.scene.body_velocity(chassis).unwrap();
    assert!(v.angular.y.abs() > 0.05, "{v:?}");
    assert!(v.linear.length() < 0.1, "{v:?}");
    for r in rotors {
        w.tracks.set_speed(r, 0.0).unwrap();
    }
    run(&mut w, 360);
    let v = w.scene.body_velocity(chassis).unwrap();
    assert!(
        v.angular.length() < 0.02 && v.linear.length() < 0.02,
        "{v:?}"
    );
}

#[test]
fn invalid_commands_and_removed_bodies_do_not_leave_forces() {
    let (mut w, _, rotors) = fixture(false, 0.85);
    assert!(w.tracks.set_speed(rotors[0], f64::NAN).is_err());
    let before = w.tracks.sample(&w.scene, rotors[0]).unwrap().travel;
    assert!(w.step(f64::NAN).is_err());
    assert_eq!(w.tracks.sample(&w.scene, rotors[0]).unwrap().travel, before);
    w.tracks.set_speed(rotors[0], 0.4).unwrap();
    let state_before = w.scene.state().joint_q.host().unwrap().to_vec();
    assert!(
        w.step_with_forces(
            1.0 / 120.0,
            |_, _| false,
            |_, _, _, _| { Err(molla_core::Error::Solver("fixture force failure".into())) }
        )
        .is_err()
    );
    assert_eq!(w.tracks.sample(&w.scene, rotors[0]).unwrap().travel, before);
    assert_eq!(
        w.scene.state().joint_q.host().unwrap(),
        state_before.as_slice()
    );
    w.scene.remove_body(rotors[0]).unwrap();
    run(&mut w, 1);
    assert!(w.tracks.sample(&w.scene, rotors[0]).is_none());
    w.tracks.clear();
    assert!(w.tracks.is_empty());
    assert!(
        w.scene
            .state()
            .body_f
            .host()
            .unwrap()
            .iter()
            .all(|&f| f == SpatialVector::ZERO)
    );
}

#[test]
fn dynamic_ground_receives_equal_opposite_contact_reaction() {
    let (mut w, chassis, rotors) = fixture(false, 0.85);
    let platform = w
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(-Vec3::Y * 0.1)),
            additional_mass: Some(MassProperties {
                mass: 5000.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(Vec3::splat(10000.0)),
            }),
            ..Default::default()
        })
        .unwrap();
    let mut collider = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(10.0, 0.1, 10.0),
    });
    collider.parent = Some(platform);
    collider.friction = 0.85;
    w.scene.insert_collider(collider).unwrap();
    let mut pose = w.scene.body_pose(chassis).unwrap();
    pose.position.y = 0.13;
    w.scene.set_body_poses(&[(chassis, pose)], true).unwrap();
    for r in rotors {
        w.tracks.set_speed(r, 0.4).unwrap();
    }
    for _ in 0..3 {
        w.step(1.0 / 960.0).unwrap();
    }
    let mut momentum = Vec3::ZERO;
    let mut angular = Vec3::ZERO;
    for (id, _) in w.scene.bodies() {
        let mass = w.scene.body_mass_properties(id).unwrap();
        let velocity = w.scene.body_velocity(id).unwrap();
        let rotation = Mat3::from_quat(w.scene.body_pose(id).unwrap().rotation);
        let p = velocity.linear * mass.mass;
        momentum += p;
        angular += rotation * mass.inertia * rotation.transpose() * velocity.angular
            + w.scene.body_world_com(id).unwrap().cross(p);
    }
    assert!(w.scene.body_velocity(platform).unwrap().linear.z < -1e-7);
    assert!(momentum.length() < 1e-6, "{momentum:?}");
    assert!(angular.length() < 1e-3, "{angular:?}");
}

#[test]
fn fixed_substeps_are_independent_of_outer_step_rate() {
    let mut results = Vec::new();
    for hz in [60, 120, 240] {
        let (mut w, chassis, rotors) = fixture(true, 0.85);
        w.solver.substeps = 960 / hz;
        for _ in 0..hz {
            w.step(1.0 / hz as f64).unwrap();
        }
        w.tracks.set_speed(rotors[0], 0.4).unwrap();
        w.tracks.set_speed(rotors[1], 0.3).unwrap();
        for _ in 0..hz * 2 {
            w.step(1.0 / hz as f64).unwrap();
        }
        results.push((
            w.scene.body_pose(chassis).unwrap(),
            w.tracks.sample(&w.scene, rotors[0]).unwrap().travel,
        ));
    }
    for (pose, travel) in &results[1..] {
        assert!(
            (pose.position - results[0].0.position).length() < 1e-8,
            "{results:?}"
        );
        assert!((travel - results[0].1).abs() < 1e-8, "{results:?}");
    }
}

#[test]
fn excluded_ground_has_no_traction_and_external_impulses_do_not_disable_tracks() {
    let (mut w, chassis, rotors) = fixture(true, 0.85);
    run(&mut w, 120);
    for r in rotors {
        w.tracks.set_speed(r, 0.4).unwrap();
    }
    w.apply_body_impulse_with_wake(chassis, SpatialVector::new(Vec3::X, Vec3::ZERO), true)
        .unwrap();
    w.step(1.0 / 120.0).unwrap();
    assert!(w.tracks.sample(&w.scene, rotors[0]).unwrap().contacts > 0);
    w.step_filtered(1.0 / 120.0, |_, _| true).unwrap();
    for r in rotors {
        let s = w.tracks.sample(&w.scene, r).unwrap();
        assert_eq!(s.contacts, 0);
        assert_eq!(s.longitudinal_force, 0.0);
    }
}

#[test]
fn uphill_drive_and_blocking_obstacle_remain_finite() {
    let (mut w, chassis, rotors) = fixture(true, 0.85);
    let ground = w
        .scene
        .colliders()
        .find(|(_, c)| c.parent.is_none())
        .unwrap()
        .0;
    w.scene
        .set_collider_pose(
            ground,
            Transform::new(-Vec3::Y * 0.1, Quat::from_rotation_x(0.08)),
        )
        .unwrap();
    run(&mut w, 120);
    let start = w.scene.body_pose(chassis).unwrap().position;
    for r in rotors {
        w.tracks.set_speed(r, -0.4).unwrap();
    }
    run(&mut w, 480);
    let delta = w.scene.body_pose(chassis).unwrap().position - start;
    assert!(delta.z < -0.3 && delta.y > 0.0, "{delta:?}");

    let (mut w, chassis, rotors) = fixture(true, 0.85);
    let mut wall = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(2.0, 0.5, 0.1),
    });
    wall.pose = Transform::from_translation(Vec3::new(0.0, 0.5, 1.0));
    w.scene.insert_collider(wall).unwrap();
    for r in rotors {
        w.tracks.set_speed(r, 0.4).unwrap();
    }
    run(&mut w, 480);
    let pose = w.scene.body_pose(chassis).unwrap();
    let velocity = w.scene.body_velocity(chassis).unwrap();
    assert!(pose.position.z > 0.1 && pose.position.z < 0.5, "{pose:?}");
    assert!(
        velocity.linear.length() < 0.5 && velocity.angular.length() < 0.5,
        "{velocity:?}"
    );
}

#[test]
fn inactive_drive_is_pruned_and_sleep_cannot_skip_motor_commands() {
    let (mut w, chassis, rotors) = fixture(true, 0.85);
    run(&mut w, 120);
    w.sleep_body(chassis).unwrap();
    for r in rotors {
        w.tracks.set_speed(r, 0.4).unwrap();
    }
    run(&mut w, 60);
    assert!(!w.is_sleeping(chassis));
    assert!(w.scene.body_velocity(chassis).unwrap().linear.z > 0.1);
    w.scene.set_body_enabled(rotors[0], false).unwrap();
    run(&mut w, 1);
    assert!(w.tracks.sample(&w.scene, rotors[0]).is_none());
    assert!(w.tracks.set_speed(rotors[0], 0.4).is_err());
}
