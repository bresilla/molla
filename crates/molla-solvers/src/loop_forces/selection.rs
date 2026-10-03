use super::*;
use crate::articulated_response::ArticulatedResponse;
use molla_math::{Mat3, Transform};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc};
use molla_sim::{BodyParams, MassProperties};

fn body(scene: &mut RigidScene, position: Vec3, kind: BodyKind) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            kind,
            params: BodyParams::new().with_position(position),
            additional_mass: Some(MassProperties {
                mass: 3.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn fixture(kind: BodyKind) -> (RigidScene, [[BodyHandle; 2]; 2]) {
    let mut scene = RigidScene::new().unwrap();
    let anchor = body(&mut scene, Vec3::ZERO, kind);
    let groups = [0.0, 10.0].map(|x| {
        let a = body(&mut scene, Vec3::new(x, 1.0, 0.0), BodyKind::Dynamic);
        let b = body(&mut scene, Vec3::new(x + 0.2, 1.1, 0.1), BodyKind::Dynamic);
        for (parent, child) in [(anchor, a), (a, b)] {
            let joint = scene
                .insert_loop_joint(
                    JointDesc {
                        parent: Some(parent),
                        child,
                        kind: JointType::Ball,
                        axis: Vec3::X,
                        frame_parent: Transform::IDENTITY,
                        frame_child: Transform::IDENTITY,
                    },
                    JointSoftness {
                        natural_frequency: 30.0,
                        damping_ratio: 1.0,
                    },
                )
                .unwrap();
            scene
                .set_joint_motor(
                    joint,
                    JointAxis::AngX,
                    JointMotor {
                        mode: JointTargetMode::Velocity,
                        target_velocity: 2.0,
                        damping: 5.0,
                        max_force: 0.3,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        [a, b]
    });
    (scene, groups)
}

#[test]
fn selected_loop_rows_preserve_active_response_and_skip_sleeping_boundaries() {
    for kind in [BodyKind::Fixed, BodyKind::Kinematic] {
        let (scene, groups) = fixture(kind);
        let seed = SpatialVector::new(Vec3::new(0.3, -0.2, 0.7), Vec3::splat(0.1));
        let mut full = vec![seed; scene.model().body_count];
        apply(
            &scene,
            &mut ArticulatedResponse::new(scene.model(), scene.state()),
            1.0 / 960.0,
            &mut full,
        )
        .unwrap();
        assert_eq!(rows(&scene, scene.state(), None).unwrap().len(), 16);
        for active in [0, 1] {
            let mut selected = vec![true; scene.model().body_count];
            for body in groups[1 - active] {
                selected[scene.body_index(body).unwrap().index()] = false;
            }
            assert_eq!(
                rows(&scene, scene.state(), Some(&selected)).unwrap().len(),
                8
            );
            let mut actual = vec![seed; full.len()];
            apply_selected(
                &scene,
                &mut ArticulatedResponse::new(scene.model(), scene.state()),
                1.0 / 960.0,
                &mut actual,
                Some(&selected),
            )
            .unwrap();
            for (group, bodies) in groups.iter().enumerate() {
                for &body in bodies {
                    let i = scene.body_index(body).unwrap().index();
                    if group == active {
                        assert!((actual[i].linear - full[i].linear).length() < 1e-8);
                        assert!((actual[i].angular - full[i].angular).length() < 1e-8);
                        assert!((actual[i].linear - seed.linear).length() > 1.0);
                    } else {
                        assert_eq!(actual[i], seed);
                    }
                }
            }
        }
        let mut selected = vec![false; scene.model().body_count];
        selected[0] = true;
        assert!(
            rows(&scene, scene.state(), Some(&selected))
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn invalid_or_split_loop_selection_leaves_force_buffer_unchanged() {
    let (scene, groups) = fixture(BodyKind::Fixed);
    let mut split = vec![true; scene.model().body_count];
    split[scene.body_index(groups[0][0]).unwrap().index()] = false;
    for selected in [vec![true; 1], split] {
        let seed = vec![SpatialVector::new(Vec3::X, Vec3::Y); scene.model().body_count];
        let mut actual = seed.clone();
        assert!(
            apply_selected(
                &scene,
                &mut ArticulatedResponse::new(scene.model(), scene.state()),
                1.0 / 960.0,
                &mut actual,
                Some(&selected),
            )
            .is_err()
        );
        assert_eq!(actual, seed);
    }
}

#[test]
fn sleeping_mass_scale_and_singular_inertia_do_not_change_active_forces() {
    let (mut scene, groups) = fixture(BodyKind::Fixed);
    let evaluate = |scene: &RigidScene| {
        let mut selected = vec![true; scene.model().body_count];
        for body in groups[1] {
            selected[scene.body_index(body).unwrap().index()] = false;
        }
        let mut forces = vec![SpatialVector::ZERO; scene.model().body_count];
        apply_selected(
            scene,
            &mut ArticulatedResponse::new(scene.model(), scene.state()),
            1.0 / 960.0,
            &mut forces,
            Some(&selected),
        )
        .unwrap();
        forces
    };
    let expected = evaluate(&scene);
    for scale in [0.0, 1e-15, 1e-9, 1.0, 1e9, 1e15] {
        for singular in [false, true] {
            for body in groups[1] {
                scene
                    .set_body_mass_properties(
                        body,
                        MassProperties {
                            mass: 3.0 * scale,
                            local_com: Vec3::ZERO,
                            inertia: Mat3::from_diagonal(Vec3::new(
                                if singular { 0.0 } else { scale },
                                scale,
                                scale,
                            )),
                        },
                    )
                    .unwrap();
                assert_eq!(scene.body_mass_properties(body).unwrap().mass, 3.0 * scale);
            }
            assert_eq!(
                evaluate(&scene),
                expected,
                "scale={scale} singular={singular}"
            );
        }
    }
}

#[test]
fn island_partition_matches_full_response_at_shared_boundaries_and_dynamic_links() {
    for kind in [BodyKind::Fixed, BodyKind::Kinematic, BodyKind::Dynamic] {
        let (mut scene, _) = fixture(kind);
        let joints: Vec<_> = scene.loop_joints().map(|(id, _)| id).collect();
        for joint in joints {
            scene.set_joint_brake(joint, JointAxis::AngX, 0.7).unwrap();
        }
        let islands = crate::rigid_islands::RigidIslands::build(&scene, []).unwrap();
        assert_eq!(
            islands.groups().len(),
            if kind == BodyKind::Dynamic { 1 } else { 2 }
        );
        for dt in [1.0 / 60.0, 1.0 / 240.0, 1.0 / 960.0] {
            compare_partition(&scene, dt);
        }
    }
}

#[test]
fn island_partition_preserves_static_only_effort_and_sorted_reactions() {
    let (mut scene, _) = fixture(BodyKind::Fixed);
    let parent = body(&mut scene, Vec3::ZERO, BodyKind::Fixed);
    let child = body(&mut scene, Vec3::ZERO, BodyKind::Kinematic);
    let joint = scene
        .insert_loop_joint(
            JointDesc {
                parent: Some(parent),
                child,
                kind: JointType::Ball,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            },
            JointSoftness {
                natural_frequency: 3.0,
                damping_ratio: 0.7,
            },
        )
        .unwrap();
    scene
        .set_joint_motor(
            joint,
            JointAxis::AngX,
            JointMotor {
                mode: JointTargetMode::Effort,
                target_position: 0.4,
                max_force: 0.3,
                ..Default::default()
            },
        )
        .unwrap();
    compare_partition(&scene, 1.0 / 240.0);
    let mut forces = vec![SpatialVector::ZERO; scene.model().body_count];
    let reactions = apply(
        &scene,
        &mut ArticulatedResponse::new(scene.model(), scene.state()),
        1.0 / 240.0,
        &mut forces,
    )
    .unwrap();
    assert!(reactions.windows(2).all(|pair| pair[0].0 < pair[1].0));
    let reaction = reactions.iter().find(|(id, _)| *id == joint).unwrap().1;
    assert!((reaction.torque - Vec3::X * 0.3).length() < 1e-12);
    for (body, sign) in [(parent, -1.0), (child, 1.0)] {
        let force = forces[scene.body_index(body).unwrap().index()];
        assert_eq!(force.linear, Vec3::ZERO);
        assert!((force.angular - Vec3::X * (sign * 0.3)).length() < 1e-12);
    }
}

fn compare_partition(scene: &RigidScene, dt: Real) {
    let seed = SpatialVector::new(Vec3::new(0.3, -0.2, 0.7), Vec3::splat(0.1));
    let mut expected = vec![seed; scene.model().body_count];
    let all_rows = rows(scene, scene.state(), None).unwrap();
    let arts: Vec<_> = (0..scene.model().articulation_count).collect();
    let mut expected_reactions = apply_rows(
        scene,
        &mut ArticulatedResponse::new(scene.model(), scene.state()),
        dt,
        &mut expected,
        &all_rows,
        &arts,
    )
    .unwrap();
    expected_reactions.sort_unstable_by_key(|(id, _)| *id);
    let mut actual = vec![seed; expected.len()];
    let reactions = apply(
        scene,
        &mut ArticulatedResponse::new(scene.model(), scene.state()),
        dt,
        &mut actual,
    )
    .unwrap();
    assert_eq!(reactions.len(), expected_reactions.len());
    for (a, b) in actual.iter().zip(&expected) {
        for (a, b) in [(a.linear, b.linear), (a.angular, b.angular)] {
            assert!(
                (a - b).length() <= 1e-9 * (1.0 + b.length()),
                "dt={dt}: {a:?} != {b:?}"
            );
        }
    }
    for ((id, actual), (expected_id, expected)) in reactions.iter().zip(expected_reactions) {
        assert_eq!(*id, expected_id);
        for (a, b) in [
            (actual.force, expected.force),
            (actual.torque, expected.torque),
        ] {
            assert!(
                (a - b).length() <= 1e-9 * (1.0 + b.length()),
                "dt={dt}, joint={id:?}: {a:?} != {b:?}"
            );
        }
    }
}
