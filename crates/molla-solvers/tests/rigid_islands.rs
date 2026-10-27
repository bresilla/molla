use std::sync::Arc;

use molla_math::{Mat3, Real, Transform, Vec3};
use molla_sim::joint_control::JointSoftness;
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, ColliderHandle, JointDesc,
    JointHandle, JointSettings, RigidScene,
};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_islands::RigidIslands;
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::wheel_forces::WheelDesc;
use molla_vehicle::RigidTire;

fn body(scene: &mut RigidScene, position: Vec3, kind: BodyKind) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(position)),
            kind,
            additional_mass: Some(MassProperties {
                mass: 1.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn desc(parent: BodyHandle, child: BodyHandle) -> JointDesc {
    JointDesc {
        parent: Some(parent),
        child,
        kind: JointType::Revolute,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
        axis: Vec3::Z,
    }
}

fn loop_joint(scene: &mut RigidScene, a: BodyHandle, b: BodyHandle) -> JointHandle {
    scene
        .insert_loop_joint(
            desc(a, b),
            JointSoftness {
                natural_frequency: 20.0,
                damping_ratio: 1.0,
            },
        )
        .unwrap()
}

fn groups(islands: &RigidIslands) -> Vec<Vec<BodyHandle>> {
    islands
        .groups()
        .iter()
        .map(|group| group.bodies.clone())
        .collect()
}

#[test]
fn tree_and_loop_edits_change_whole_articulation_groups() {
    let mut scene = RigidScene::new().unwrap();
    let a = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
    let b = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
    let c = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
    let tree = scene.insert_joint(desc(b, a)).unwrap();
    assert_eq!(
        groups(&RigidIslands::build(&scene, []).unwrap()),
        vec![vec![a, b], vec![c]]
    );
    let closure = loop_joint(&mut scene, b, c);
    let graph = RigidIslands::build(&scene, []).unwrap();
    assert_eq!(groups(&graph), vec![vec![a, b, c]]);
    assert_eq!(graph.groups()[0].articulations.len(), 2);
    scene
        .set_joint_settings(
            tree,
            JointSettings {
                enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        groups(&RigidIslands::build(&scene, []).unwrap()),
        vec![vec![a], vec![b, c]]
    );
    scene
        .set_joint_settings(
            closure,
            JointSettings {
                enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        groups(&RigidIslands::build(&scene, []).unwrap()),
        vec![vec![a], vec![b], vec![c]]
    );
    scene
        .set_joint_settings(closure, JointSettings::default())
        .unwrap();
    scene.set_body_enabled(b, false).unwrap();
    assert_eq!(
        groups(&RigidIslands::build(&scene, []).unwrap()),
        vec![vec![a], vec![c]]
    );
    scene.set_body_enabled(b, true).unwrap();
    assert_eq!(
        groups(&RigidIslands::build(&scene, []).unwrap()),
        vec![vec![a], vec![b, c]]
    );
    scene.remove_body(b).unwrap();
    let graph = RigidIslands::build(&scene, []).unwrap();
    assert_eq!(groups(&graph), vec![vec![a], vec![c]]);
    assert!(graph.affected_bodies(&[b]).is_err());
    assert!(RigidIslands::build(&scene, [[Some(a), Some(b)]]).is_err());
}

#[test]
fn immovable_boundaries_propagate_edits_without_merging_islands() {
    for kind in [BodyKind::Fixed, BodyKind::Kinematic] {
        let mut scene = RigidScene::new().unwrap();
        let anchor = body(&mut scene, Vec3::ZERO, kind);
        let a = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
        let b = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
        loop_joint(&mut scene, anchor, a);
        loop_joint(&mut scene, anchor, b);
        let graph = RigidIslands::build(&scene, []).unwrap();
        assert_eq!(groups(&graph), vec![vec![a], vec![b]]);
        assert!(graph.body_island(anchor).is_none());
        assert!(
            graph
                .groups()
                .iter()
                .all(|group| group.boundaries == [anchor])
        );
        assert_eq!(graph.affected_bodies(&[anchor]).unwrap(), vec![a, b]);
        assert_eq!(graph.affected_bodies(&[a, a]).unwrap(), vec![a]);
        scene.set_body_kind(anchor, BodyKind::Dynamic).unwrap();
        assert_eq!(
            groups(&RigidIslands::build(&scene, []).unwrap()),
            vec![vec![anchor, a, b]]
        );
    }
}

#[test]
fn fixed_root_branches_retain_one_solver_articulation() {
    let mut scene = RigidScene::new().unwrap();
    let anchor = body(&mut scene, Vec3::ZERO, BodyKind::Fixed);
    let a = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
    let b = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
    scene.insert_joint(desc(anchor, a)).unwrap();
    scene.insert_joint(desc(anchor, b)).unwrap();
    let graph = RigidIslands::build(&scene, []).unwrap();
    assert_eq!(groups(&graph), vec![vec![a, b]]);
    assert_eq!(graph.groups()[0].articulations.len(), 1);
    assert_eq!(graph.groups()[0].boundaries, vec![anchor]);
}

#[test]
fn connections_are_deterministic_and_do_not_mutate_scene() {
    let mut scene = RigidScene::new().unwrap();
    assert!(RigidIslands::build(&scene, []).unwrap().groups().is_empty());
    let a = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
    let b = body(&mut scene, Vec3::X, BodyKind::Dynamic);
    let c = body(&mut scene, Vec3::Y, BodyKind::Dynamic);
    let stamp = scene.stamp();
    let revision = scene.revision();
    let first = RigidIslands::build(&scene, [[Some(a), Some(b)], [Some(b), Some(c)]]).unwrap();
    let second = RigidIslands::build(
        &scene,
        [
            [Some(c), Some(b)],
            [Some(b), Some(a)],
            [Some(a), Some(b)],
            [Some(a), None],
        ],
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.affected_bodies(&[b]).unwrap(), vec![a, b, c]);
    assert_eq!(scene.stamp(), stamp);
    assert_eq!(scene.revision(), revision);
}

fn contacts_fixture() -> (RigidWorld, [BodyHandle; 3], BodyHandle, ColliderHandle) {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world.solver.substeps = 1;
    let ground = body(&mut world.scene, Vec3::new(0.0, -0.1, 0.0), BodyKind::Fixed);
    let mut floor = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(10.0, 0.1, 10.0),
    });
    floor.parent = Some(ground);
    world.scene.insert_collider(floor).unwrap();
    let bodies =
        [0.0, 0.9, 3.0].map(|x| body(&mut world.scene, Vec3::new(x, 0.49, 0.0), BodyKind::Dynamic));
    let mut middle = None;
    for (index, &body) in bodies.iter().enumerate() {
        let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
        sphere.parent = Some(body);
        let id = world.scene.insert_collider(sphere).unwrap();
        if index == 1 {
            middle = Some(id);
        }
    }
    (world, bodies, ground, middle.unwrap())
}

#[test]
fn world_contacts_connect_dynamic_bodies_but_not_through_ground() {
    let (mut world, [a, b, c], ground, _) = contacts_fixture();
    assert_eq!(
        groups(&world.islands().unwrap()),
        vec![vec![a], vec![b], vec![c]]
    );
    world.step(1e-5).unwrap();
    let graph = world.islands().unwrap();
    assert_eq!(groups(&graph), vec![vec![a, b], vec![c]]);
    assert!(
        graph
            .groups()
            .iter()
            .all(|group| group.boundaries == [ground])
    );
    assert_eq!(graph.affected_bodies(&[ground]).unwrap(), vec![a, b, c]);
    world
        .scene
        .set_body_pose(b, Transform::from_translation(Vec3::new(6.0, 0.49, 0.0)))
        .unwrap();
    assert_eq!(
        groups(&world.islands().unwrap()),
        vec![vec![a], vec![b], vec![c]]
    );
    assert!(
        world
            .islands()
            .unwrap()
            .groups()
            .iter()
            .all(|group| group.boundaries.is_empty())
    );
}

#[test]
fn sensors_exclusions_and_failed_steps_do_not_supply_contact_edges() {
    let (mut world, [a, b, c], _, middle) = contacts_fixture();
    world
        .step_filtered(1e-5, |x, y| {
            [x, y] == [Some(a), Some(b)] || [x, y] == [Some(b), Some(a)]
        })
        .unwrap();
    assert_eq!(
        groups(&world.islands().unwrap()),
        vec![vec![a], vec![b], vec![c]]
    );
    let mut collider = world.scene.collider(middle).unwrap().clone();
    collider.sensor = true;
    world.scene.replace_collider(middle, collider).unwrap();
    world.step(1e-5).unwrap();
    let graph = world.islands().unwrap();
    assert_eq!(groups(&graph), vec![vec![a], vec![b], vec![c]]);
    assert!(graph.body_island(b).unwrap().boundaries.is_empty());
    assert!(world.step(Real::NAN).is_err());
    assert!(
        world
            .islands()
            .unwrap()
            .groups()
            .iter()
            .all(|group| group.boundaries.is_empty())
    );
}

#[test]
fn wheel_heading_dependencies_refresh_before_prepare() {
    let mut world = RigidWorld::new().unwrap();
    let chassis = body(&mut world.scene, Vec3::ZERO, BodyKind::Dynamic);
    let wheel = body(&mut world.scene, Vec3::ZERO, BodyKind::Dynamic);
    let heading = body(&mut world.scene, Vec3::ZERO, BodyKind::Dynamic);
    let spin = world.scene.insert_joint(desc(chassis, wheel)).unwrap();
    world
        .wheels
        .insert(
            &world.scene,
            WheelDesc {
                body: wheel,
                spin_joint: spin,
                heading_body: heading,
                local_heading: Vec3::X,
                local_hub: Vec3::ZERO,
                spin_sign: 1.0,
                radius: 0.5,
                normal_stiffness: 1000.0,
                normal_damping: 0.0,
                tire: Arc::new(RigidTire {
                    c_kappa: 1000.0,
                    c_alpha: 1000.0,
                }),
            },
        )
        .unwrap();
    assert_eq!(
        groups(&world.islands().unwrap()),
        vec![vec![chassis, wheel, heading]]
    );
    world.scene.remove_body(heading).unwrap();
    assert_eq!(
        groups(&world.islands().unwrap()),
        vec![vec![chassis, wheel]]
    );
}

#[test]
fn tyre_samples_supply_ground_dependencies_without_rigid_contacts() {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world.solver.substeps = 1;
    let ground = body(&mut world.scene, -Vec3::Y * 0.1, BodyKind::Fixed);
    let mut floor = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5.0, 0.1, 5.0),
    });
    floor.parent = Some(ground);
    let collider = world.scene.insert_collider(floor).unwrap();
    world
        .wheels
        .register_ground(&world.scene, collider, None)
        .unwrap();
    let chassis = body(&mut world.scene, Vec3::Y * 0.49, BodyKind::Dynamic);
    let wheel = body(&mut world.scene, Vec3::Y * 0.49, BodyKind::Dynamic);
    let spin = world.scene.insert_joint(desc(chassis, wheel)).unwrap();
    world
        .wheels
        .insert(
            &world.scene,
            WheelDesc {
                body: wheel,
                spin_joint: spin,
                heading_body: chassis,
                local_heading: Vec3::X,
                local_hub: Vec3::ZERO,
                spin_sign: 1.0,
                radius: 0.5,
                normal_stiffness: 1000.0,
                normal_damping: 0.0,
                tire: Arc::new(RigidTire {
                    c_kappa: 1000.0,
                    c_alpha: 1000.0,
                }),
            },
        )
        .unwrap();
    assert!(world.islands().unwrap().groups()[0].boundaries.is_empty());
    world.step(1e-5).unwrap();
    assert!(world.contact_pairs().is_empty());
    assert!(world.wheels.samples(&world.scene)[0].output.fz > 0.0);
    let graph = world.islands().unwrap();
    assert_eq!(groups(&graph), vec![vec![chassis, wheel]]);
    assert_eq!(graph.groups()[0].boundaries, vec![ground]);
    world.wheels.remove_ground(collider);
    assert!(world.islands().unwrap().groups()[0].boundaries.is_empty());
}

#[test]
fn a_world_s_islands_are_shared_between_steps_and_follow_its_contacts() {
    let mut scene = RigidScene::new().unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(4.0, 0.5, 4.0),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.5);
    scene.insert_collider(ground).unwrap();
    let ball = |scene: &mut RigidScene, x: Real| {
        let ball = body(scene, Vec3::new(x, 0.25, 0.0), BodyKind::Dynamic);
        let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.25 });
        collider.parent = Some(ball);
        scene.insert_collider(collider).unwrap();
        ball
    };
    let (a, b) = (ball(&mut scene, -2.0), ball(&mut scene, 2.0));
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.step(1.0 / 60.0).unwrap();
    let first = world.islands().unwrap();
    assert!(Arc::ptr_eq(&first, &world.islands().unwrap()));
    assert_eq!(groups(&first), vec![vec![a], vec![b]]);
    world
        .scene
        .set_body_pose(b, Transform::from_translation(Vec3::new(-2.0, 0.9, 0.0)))
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &world.islands().unwrap()));
    world.step(1.0 / 60.0).unwrap();
    assert_eq!(groups(&world.islands().unwrap()), vec![vec![a], vec![b]]);
    let landed = (0..60).any(|_| {
        world.step(1.0 / 60.0).unwrap();
        groups(&world.islands().unwrap()) == vec![vec![a, b]]
    });
    assert!(landed, "the falling ball never joined the one below");
}
