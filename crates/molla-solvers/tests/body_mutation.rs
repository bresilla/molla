use molla_core::Error;
use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyKind, ColliderDesc, ColliderGeometry, JointDesc};
use molla_sim::{ColliderMass, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn near(a: Vec3, b: Vec3) {
    assert!((a - b).length() < 1e-8, "{a:?} != {b:?}");
}

fn world() -> RigidWorld {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w
}

#[test]
fn full_tensor_impulses_match_free_body_and_are_translation_invariant() {
    for offset in [Vec3::ZERO, Vec3::new(1e6, -2e6, 3e6)] {
        let mut world = world();
        let rotation = Quat::from_rotation_y(0.7);
        let principal = Mat3::from_quat(Quat::from_rotation_z(0.3));
        let inertia =
            principal * Mat3::from_diagonal(Vec3::new(2.0, 3.0, 4.0)) * principal.transpose();
        let mut desc = BodyDesc::default();
        desc.params.pose = Transform::new(offset, rotation);
        desc.additional_mass = Some(MassProperties {
            mass: 5.0,
            local_com: Vec3::new(0.4, 0.5, -0.2),
            inertia,
        });
        let body = world.scene.insert_body(desc).unwrap();
        let impulse = SpatialVector::new(Vec3::new(2.0, 4.0, 6.0), Vec3::new(3.0, 1.0, -2.0));
        world.apply_body_impulse(body, impulse).unwrap();
        let actual = world.scene.body_velocity(body).unwrap();
        near(actual.linear, impulse.linear / 5.0);
        let r = Mat3::from_quat(rotation);
        near(
            actual.angular,
            r * inertia.inverse() * r.transpose() * impulse.angular,
        );
        assert_eq!(
            world.scene.state().body_f.host().unwrap()[0],
            SpatialVector::ZERO
        );
        assert_eq!(world.scene.body_pose(body).unwrap().position, offset);
    }
}

#[test]
fn point_impulse_and_force_use_world_com() {
    let mut world = world();
    let mut desc = BodyDesc::default();
    desc.params.pose = Transform::new(Vec3::new(3.0, 4.0, 5.0), Quat::from_rotation_z(0.8));
    desc.additional_mass = Some(MassProperties {
        mass: 2.0,
        local_com: Vec3::X,
        inertia: Mat3::IDENTITY * 3.0,
    });
    let body = world.scene.insert_body(desc).unwrap();
    let com = world.scene.body_world_com(body).unwrap();
    let point = com + Vec3::X * 2.0;
    world
        .apply_body_impulse_at_point(body, Vec3::Y * 3.0, point)
        .unwrap();
    let v = world.scene.body_velocity(body).unwrap();
    near(v.linear, Vec3::Y * 1.5);
    near(v.angular, Vec3::Z * 2.0);
    near(
        world.scene.body_velocity_at_point(body, point).unwrap(),
        Vec3::Y * 5.5,
    );
    world
        .scene
        .add_body_force_at_point(body, Vec3::Y * 3.0, point)
        .unwrap();
    let force = world.scene.state().body_f.host().unwrap()[0];
    near(force.linear, Vec3::Y * 3.0);
    near(force.angular, Vec3::Z * 6.0);
    world.scene.reset_body_forces(body).unwrap();
    assert_eq!(
        world.scene.state().body_f.host().unwrap()[0],
        SpatialVector::ZERO
    );
}

#[test]
fn impulse_on_fixed_link_uses_whole_articulation_mass() {
    let mut world = world();
    let mut desc = BodyDesc::default();
    desc.params.mass = 2.0;
    let root = world.scene.insert_body(desc.clone()).unwrap();
    desc.params.pose.position = Vec3::X;
    let child = world.scene.insert_body(desc).unwrap();
    world
        .scene
        .insert_joint(JointDesc {
            parent: Some(root),
            child,
            kind: JointType::Fixed,
            frame_parent: Transform::new(Vec3::X, Quat::IDENTITY),
            frame_child: Transform::IDENTITY,
            axis: Vec3::X,
        })
        .unwrap();
    world
        .apply_body_impulse(child, SpatialVector::new(Vec3::X * 8.0, Vec3::ZERO))
        .unwrap();
    for id in [root, child] {
        let v = world.scene.body_velocity(id).unwrap();
        near(v.linear, Vec3::X * 2.0);
        near(v.angular, Vec3::ZERO);
    }
    world.step(0.01).unwrap();
    near(
        world.scene.body_velocity(child).unwrap().linear,
        Vec3::X * 2.0,
    );
}

#[test]
fn revolute_impulse_respects_anchor_and_joint_axis() {
    let mut world = world();
    let mut desc = BodyDesc::default();
    desc.params.mass = 2.0;
    desc.params.com = Vec3::X;
    let body = world.scene.insert_body(desc).unwrap();
    world
        .scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Revolute,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    world
        .apply_body_impulse(body, SpatialVector::new(Vec3::Y * 3.0, Vec3::ZERO))
        .unwrap();
    let v = world.scene.body_velocity(body).unwrap();
    near(v.angular, Vec3::Z);
    near(v.linear, Vec3::Y);
    world
        .apply_body_impulse(body, SpatialVector::new(Vec3::X * 100.0, Vec3::Y * 100.0))
        .unwrap();
    near(world.scene.body_velocity(body).unwrap().angular, Vec3::Z);
    near(
        world
            .scene
            .body_velocity_at_point(body, Vec3::ZERO)
            .unwrap(),
        Vec3::ZERO,
    );
}

#[test]
fn singular_inertia_impulse_drops_null_modes_without_huge_velocity() {
    let mut world = world();
    let rotation = Mat3::from_quat(Quat::from_rotation_z(0.6));
    let mut desc = BodyDesc::default();
    desc.params.inertia =
        rotation * Mat3::from_diagonal(Vec3::new(0.0, 2.0, 2.0)) * rotation.transpose();
    desc.params.com = Vec3::new(0.3, 0.5, 0.7);
    let body = world.scene.insert_body(desc).unwrap();
    world
        .apply_body_impulse(
            body,
            SpatialVector::new(Vec3::ZERO, rotation.x_axis + rotation.y_axis * 4.0),
        )
        .unwrap();
    near(
        world.scene.body_velocity(body).unwrap().angular,
        rotation.y_axis * 2.0,
    );
    near(world.scene.body_velocity(body).unwrap().linear, Vec3::ZERO);
    world
        .apply_body_impulse(body, SpatialVector::new(Vec3::Y, Vec3::ZERO))
        .unwrap();
    near(world.scene.body_velocity(body).unwrap().linear, Vec3::Y);
    near(
        world.scene.body_velocity(body).unwrap().angular,
        rotation.y_axis * 2.0,
    );
}

#[test]
fn persisted_additional_mass_survives_recomputation_and_replaces_atomically() {
    let mut world = world();
    let body = world
        .scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 4.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..BodyDesc::default()
        })
        .unwrap();
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    collider.parent = Some(body);
    collider.mass = ColliderMass::Mass(2.0);
    let c = world.scene.insert_collider(collider).unwrap();
    for _ in 0..3 {
        world.recompute_body_mass(body).unwrap();
        assert_eq!(world.scene.body_mass_properties(body).unwrap().mass, 6.0);
    }
    world
        .scene
        .set_collider_mass(c, ColliderMass::Mass(3.0))
        .unwrap();
    world.recompute_body_mass(body).unwrap();
    assert_eq!(world.scene.body_mass_properties(body).unwrap().mass, 7.0);
    world
        .set_body_additional_mass(body, MassProperties::ZERO)
        .unwrap();
    world.recompute_body_mass(body).unwrap();
    assert_eq!(world.scene.body_mass_properties(body).unwrap().mass, 3.0);
    assert_eq!(
        world.scene.body_colliders(body).collect::<Vec<_>>(),
        vec![c]
    );
    let stamp = world.scene.stamp();
    assert!(
        world
            .set_body_additional_mass(
                body,
                MassProperties {
                    mass: -1.0,
                    ..MassProperties::ZERO
                }
            )
            .is_err()
    );
    assert_eq!(world.scene.stamp(), stamp);
    assert_eq!(
        world.scene.body(body).unwrap().additional_mass,
        Some(MassProperties::ZERO)
    );
}

#[test]
fn transient_force_elements_refresh_every_substep_and_never_persist() {
    let mut world = world();
    let mut desc = BodyDesc::default();
    desc.params.mass = 2.0;
    let body = world.scene.insert_body(desc).unwrap();
    world
        .scene
        .add_body_wrench(body, SpatialVector::new(Vec3::X * 2.0, Vec3::ZERO))
        .unwrap();
    let mut observed = Vec::new();
    world
        .step_with_forces(
            0.08,
            |_, _| false,
            |_, state, dt, forces| {
                assert_eq!(dt, 0.01);
                observed.push(state.body_qd.host()?[0].linear.x);
                forces[0].linear += Vec3::X * 2.0;
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(observed.len(), 8);
    for (i, v) in observed.into_iter().enumerate() {
        assert!((v - 0.02 * i as f64).abs() < 1e-9);
    }
    near(
        world.scene.body_velocity(body).unwrap().linear,
        Vec3::X * 0.16,
    );
    near(
        world.scene.state().body_f.host().unwrap()[0].linear,
        Vec3::X * 2.0,
    );
    world.step(0.08).unwrap();
    near(
        world.scene.body_velocity(body).unwrap().linear,
        Vec3::X * 0.24,
    );
}

#[test]
fn damping_reduces_free_body_velocity_without_persistent_force() {
    for damping in [0.0, 2.0, 1e100] {
        let mut world = world();
        let body = world
            .scene
            .insert_body(BodyDesc {
                velocity: SpatialVector::new(Vec3::X * 3.0, Vec3::Z * 2.0),
                linear_damping: damping,
                angular_damping: damping,
                ..BodyDesc::default()
            })
            .unwrap();
        world.step(0.08).unwrap();
        let factor = (1.0 + damping * 0.01_f64).powi(-8);
        let v = world.scene.body_velocity(body).unwrap();
        near(v.linear, Vec3::X * (3.0 * factor));
        near(v.angular, Vec3::Z * (2.0 * factor));
        assert_eq!(
            world.scene.state().body_f.host().unwrap()[0],
            SpatialVector::ZERO
        );
        world.scene.set_body_damping(body, 0.0, 0.0).unwrap();
        world.step(0.08).unwrap();
        near(world.scene.body_velocity(body).unwrap().linear, v.linear);
    }
}

#[test]
fn invalid_edits_and_failing_force_hooks_leave_state_unchanged() {
    let mut world = world();
    let body = world.scene.insert_body(BodyDesc::default()).unwrap();
    let pose = world.scene.body_pose(body).unwrap();
    let rates = world.scene.state().joint_qd.host().unwrap().to_vec();
    assert!(world.scene.set_body_damping(body, -1.0, 0.0).is_err());
    assert!(
        world
            .scene
            .add_body_wrench(body, SpatialVector::new(Vec3::splat(f64::NAN), Vec3::ZERO))
            .is_err()
    );
    assert!(
        world
            .apply_body_impulse(
                body,
                SpatialVector::new(Vec3::splat(f64::INFINITY), Vec3::ZERO)
            )
            .is_err()
    );
    let mut calls = 0;
    assert!(
        world
            .step_with_forces(
                0.1,
                |_, _| false,
                |_, _, _, f| {
                    calls += 1;
                    if calls == 3 {
                        return Err(Error::Solver("force element failed".into()));
                    }
                    f[0].linear = Vec3::X;
                    Ok(())
                }
            )
            .is_err()
    );
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.state().joint_qd.host().unwrap(), rates);
    assert!(world.contact_pairs().is_empty());
    for kind in [BodyKind::Fixed, BodyKind::Kinematic] {
        world.scene.set_body_kind(body, kind).unwrap();
        world
            .apply_body_impulse(body, SpatialVector::new(Vec3::X, Vec3::Y))
            .unwrap();
        near(world.scene.body_velocity(body).unwrap().linear, Vec3::ZERO);
    }
}

#[test]
fn teleport_updates_subtree_queries_without_a_step_and_survives_rebuild() {
    let mut world = world();
    let root = world.scene.insert_body(BodyDesc::default()).unwrap();
    let mut child_desc = BodyDesc::default();
    child_desc.params.pose.position = Vec3::X;
    let child = world.scene.insert_body(child_desc).unwrap();
    world
        .scene
        .insert_joint(JointDesc {
            parent: Some(root),
            child,
            kind: JointType::Fixed,
            frame_parent: Transform::new(Vec3::X, Quat::IDENTITY),
            frame_child: Transform::IDENTITY,
            axis: Vec3::X,
        })
        .unwrap();
    let mut shape = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    shape.parent = Some(child);
    let collider = world.scene.insert_collider(shape).unwrap();
    let pose = Transform::new(
        Vec3::X * 10.0,
        Quat::from_rotation_z(std::f64::consts::FRAC_PI_2),
    );
    let stamp = world.scene.collider_stamp(collider).unwrap();
    world.scene.set_body_pose(root, pose).unwrap();
    near(world.scene.body_pose(root).unwrap().position, pose.position);
    near(
        world.scene.body_pose(child).unwrap().position,
        Vec3::new(10.0, 1.0, 0.0),
    );
    let hit = world
        .cast_ray(Vec3::new(10.0, 3.0, 0.0), -Vec3::Y, 3.0)
        .unwrap()
        .unwrap();
    assert_eq!(hit.collider, collider);
    assert!((hit.distance - 1.5).abs() < 1e-9);
    assert_eq!(world.scene.collider_stamp(collider).unwrap(), stamp);
    world.scene.insert_body(BodyDesc::default()).unwrap();
    world.step(0.01).unwrap();
    near(
        world.scene.body_pose(child).unwrap().position,
        Vec3::new(10.0, 1.0, 0.0),
    );
}

#[test]
fn velocity_edits_reseed_joint_rates_and_propagate_to_children() {
    let mut world = world();
    let mut desc = BodyDesc::default();
    desc.params.com = Vec3::X * 0.2;
    let root = world.scene.insert_body(desc).unwrap();
    let mut child_desc = BodyDesc::default();
    child_desc.params.pose.position = Vec3::X * 2.0;
    let child = world.scene.insert_body(child_desc).unwrap();
    world
        .scene
        .insert_joint(JointDesc {
            parent: Some(root),
            child,
            kind: JointType::Fixed,
            frame_parent: Transform::new(Vec3::X * 2.0, Quat::IDENTITY),
            frame_child: Transform::IDENTITY,
            axis: Vec3::X,
        })
        .unwrap();
    let target = SpatialVector::new(Vec3::Y * 2.0, Vec3::Z * 3.0);
    world.scene.set_body_velocity(root, target).unwrap();
    near(
        world.scene.body_velocity(root).unwrap().linear,
        target.linear,
    );
    near(
        world.scene.body_velocity(child).unwrap().linear,
        Vec3::Y * 7.4,
    );
    near(
        world.scene.body_velocity(child).unwrap().angular,
        target.angular,
    );
    let rates = world.scene.state().joint_qd.host().unwrap().to_vec();
    world.scene.insert_body(BodyDesc::default()).unwrap();
    assert_eq!(
        &world.scene.state().joint_qd.host().unwrap()[..rates.len()],
        rates
    );
    assert!(
        world
            .scene
            .set_body_velocity(root, SpatialVector::new(Vec3::splat(f64::NAN), Vec3::ZERO))
            .is_err()
    );
    near(
        world.scene.body_velocity(root).unwrap().linear,
        target.linear,
    );
}

#[test]
fn fixed_and_kinematic_teleports_remain_prescribed_during_steps() {
    for kind in [BodyKind::Fixed, BodyKind::Kinematic] {
        let mut world = world();
        let body = world
            .scene
            .insert_body(BodyDesc {
                kind,
                ..BodyDesc::default()
            })
            .unwrap();
        let pose = Transform::new(Vec3::new(3.0, 4.0, 5.0), Quat::from_rotation_x(0.4));
        world.scene.set_body_pose(body, pose).unwrap();
        world
            .scene
            .set_body_velocity(body, SpatialVector::new(Vec3::X, Vec3::ZERO))
            .unwrap();
        world.step(0.1).unwrap();
        near(world.scene.body_pose(body).unwrap().position, pose.position);
        near(
            world.scene.body_velocity(body).unwrap().linear,
            if kind == BodyKind::Fixed {
                Vec3::ZERO
            } else {
                Vec3::X
            },
        );
        let stamp = world.scene.stamp();
        assert!(
            world
                .scene
                .set_body_pose(body, Transform::new(Vec3::splat(f64::NAN), Quat::IDENTITY))
                .is_err()
        );
        assert_eq!(world.scene.stamp(), stamp);
    }
}
