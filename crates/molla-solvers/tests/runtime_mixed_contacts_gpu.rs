#![cfg(feature = "gpu")]

#[path = "support/hardware_engine.rs"]
mod hardware_engine;

use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, DistanceLimits, JointBreakLimits,
    JointDesc, JointHandle, RigidScene, SphericalConeLimits,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointType, MassProperties,
};
use molla_solvers::rigid_world::{GpuContactCapacity, RigidWorld};

fn body(scene: &mut RigidScene, position: Vec3) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(position),
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn joint(scene: &mut RigidScene, body: BodyHandle, kind: JointType, anchor: Vec3) -> JointHandle {
    scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(anchor),
            frame_child: Transform::IDENTITY,
        })
        .unwrap()
}

fn contact_scene(position: Vec3, pads: bool) -> (RigidScene, BodyHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Z * 2.).unwrap();
    let body = body(&mut scene, position);
    let material = ContactMaterial {
        stiffness: 1e5,
        damping: 0.,
        friction: Some(FrictionCoefficients {
            mu_static: 0.125,
            mu_dynamic: 0.1,
        }),
        ..Default::default()
    };
    let offsets = if pads {
        vec![
            Vec3::new(-0.3, -0.3, 0.),
            Vec3::new(-0.3, 0.3, 0.),
            Vec3::new(0.3, -0.3, 0.),
            Vec3::new(0.3, 0.3, 0.),
        ]
    } else {
        vec![Vec3::ZERO]
    };
    for offset in offsets {
        let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
        collider.parent = Some(body);
        collider.mass = ColliderMass::Mass(0.);
        collider.pose = Transform::from_translation(offset);
        collider.contact_material = material;
        scene.insert_collider(collider).unwrap();
    }
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5., 5., 0.1),
    });
    ground.pose = Transform::from_translation(-Vec3::Z * 0.1);
    ground.contact_material = material;
    scene.insert_collider(ground).unwrap();
    (scene, body)
}

fn radial() -> (RigidScene, BodyHandle, JointHandle) {
    let (mut scene, b) = contact_scene(Vec3::new(-1., 1., 0.49998), false);
    let j = joint(&mut scene, b, JointType::Distance, Vec3::Z * 0.49998);
    scene
        .set_distance_limits(
            j,
            DistanceLimits {
                min: Some(2.0_f64.sqrt()),
                max: Some(2.0_f64.sqrt()),
            },
        )
        .unwrap();
    scene
        .set_body_velocity(b, SpatialVector::new(Vec3::Y * 0.01, Vec3::ZERO))
        .unwrap();
    (scene, b, j)
}

fn contact_load(world: &RigidWorld, body: BodyHandle, center: Vec3) -> SpatialVector {
    let mut result = SpatialVector::ZERO;
    for pair in world.contact_pairs() {
        let sign = if pair.bodies[1] == Some(body) {
            1.
        } else if pair.bodies[0] == Some(body) {
            -1.
        } else {
            continue;
        };
        for point in pair.points {
            let force = point.force_on_b * sign;
            result.linear += force;
            result.angular += (point.point - center).cross(force);
        }
    }
    result
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-mixed-contacts-gpu"]
fn runtime_distance_contact_uses_common_impulse_and_publishes_reactions() {
    let engine = hardware_engine::engine();
    for capture in [false, true] {
        let start = std::time::Instant::now();
        eprintln!("live_distance_contact capture={capture} stage=scene");
        let (scene, b, j) = radial();
        let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
        cpu.solver.substeps = 1;
        cpu.set_reaction_readout(capture).unwrap();
        cpu.step(0.01).unwrap();
        eprintln!(
            "live_distance_contact capture={capture} stage=gpu elapsed={:?}",
            start.elapsed()
        );
        let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        world.set_reaction_readout(capture).unwrap();
        world.solver.substeps = 1;
        world.step(0.01).unwrap();
        eprintln!(
            "live_distance_contact capture={capture} stage=readout elapsed={:?}",
            start.elapsed()
        );
        let pairs = world.contact_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].points.len(), 1);
        let point = &pairs[0].points[0];
        assert!(
            (point.normal_force - 2.).abs() < 0.01,
            "normal={}",
            point.normal_force
        );
        let cap = 0.1 * point.normal_force * 0.01;
        let expected = 0.005 - cap / 2.0_f64.sqrt();
        let velocity = world.scene.body_velocity(b).unwrap().linear;
        assert!(
            (velocity.x - expected).abs() < 3e-6 && (velocity.y - expected).abs() < 3e-6,
            "velocity={velocity:?} expected={expected}"
        );
        let reading = world.distance_reading(j).unwrap();
        assert!((velocity - cpu.scene.body_velocity(b).unwrap().linear).length() < 3e-6);
        assert!((reading.force - cpu.distance_reading(j).unwrap().force).length() < 3e-4);
        assert!(
            (reading.force - Vec3::new(0.5, -0.5, 0.)).length() < 3e-4,
            "reading={reading:?}"
        );
        assert!(reading.violation < 3e-6);
        if capture {
            let reaction = world.joint_reaction(j).unwrap();
            assert!((reaction.child.linear - reading.force).length() < 1e-6);
            assert!(reaction.child.angular.length() < 1e-6);
        }
        assert_eq!(world.gpu_diagnostics().unwrap().completed_frames, 1);
        eprintln!(
            "live_distance_contact capture={capture} stage=complete elapsed={:?}",
            start.elapsed()
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-mixed-contacts-gpu"]
fn runtime_contact_and_scalar_stop_without_distance_joint_share_response() {
    let (mut scene, b) = contact_scene(Vec3::Z * 0.49998, false);
    let slider = joint(&mut scene, b, JointType::Prismatic, Vec3::Z * 0.49998);
    scene
        .set_joint_limits(slider, JointAxis::LinX, Some([-1., 0.]))
        .unwrap();
    scene
        .set_body_velocity(b, SpatialVector::new(Vec3::X * 0.01, Vec3::ZERO))
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, hardware_engine::engine()).unwrap();
    world.set_reaction_readout(true).unwrap();
    world.solver.substeps = 1;
    world.step(0.01).unwrap();
    assert!(world.scene.body_velocity(b).unwrap().linear.length() < 1e-7);
    assert!(world.scene.body_pose(b).unwrap().position.x.abs() < 1e-7);
    let reaction = world.joint_reaction(slider).unwrap().child;
    let contact = contact_load(&world, b, Vec3::Z * 0.49998);
    assert!(
        (reaction.linear.x + contact.linear.x + 1.).abs() < 3e-5,
        "reaction={reaction:?}, contact={contact:?}"
    );
    let compiled = world.gpu_diagnostics().unwrap().model_compilations;
    for _ in 0..10 {
        world.step(0.01).unwrap();
    }
    assert_eq!(
        world.gpu_diagnostics().unwrap().model_compilations,
        compiled
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-mixed-contacts-gpu"]
fn two_contact_articulations_share_distance_impulses_under_endpoint_exchange() {
    let engine = hardware_engine::engine();
    let mut trajectories = Vec::new();
    for reverse in [false, true] {
        let center = Vec3::Z * 0.49998;
        let (mut scene, a) = contact_scene(center - Vec3::X, false);
        let b = body(&mut scene, center + Vec3::X);
        let mut collider = scene
            .colliders()
            .find(|(_, c)| c.parent == Some(a))
            .unwrap()
            .1
            .clone();
        collider.parent = Some(b);
        scene.insert_collider(collider).unwrap();
        let link = scene
            .insert_joint(JointDesc {
                parent: Some(if reverse { b } else { a }),
                child: if reverse { a } else { b },
                kind: JointType::Distance,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_distance_limits(
                link,
                DistanceLimits {
                    min: Some(2.),
                    max: Some(2.),
                },
            )
            .unwrap();
        scene
            .set_body_velocity(a, SpatialVector::new(Vec3::X * 0.01, Vec3::ZERO))
            .unwrap();
        let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
        cpu.solver.substeps = 1;
        cpu.step(0.01).unwrap();
        let mut world = RigidWorld::from_scene_gpu_with_contact_capacity(
            scene,
            engine.clone(),
            GpuContactCapacity {
                pairs: 8,
                contacts_per_island: 2,
            },
        )
        .unwrap();
        world.set_reaction_readout(true).unwrap();
        world.solver.substeps = 1;
        world.step(0.01).unwrap();
        assert_eq!(world.contact_pairs().len(), 2);
        let loads = [a, b].map(|id| contact_load(&world, id, center));
        let expected = 0.005 + (loads[0].linear.x + loads[1].linear.x) * 0.005;
        assert!((expected - 0.003).abs() < 3e-6);
        let velocities = [a, b].map(|id| world.scene.body_velocity(id).unwrap());
        for (id, v) in [a, b].into_iter().zip(velocities) {
            let reference = cpu.scene.body_velocity(id).unwrap();
            assert!((v.linear - reference.linear).length() < 3e-6);
            assert!((v.angular - reference.angular).length() < 3e-6);
        }
        for v in velocities {
            assert!(
                (v.linear.x - expected).abs() < 3e-6,
                "reverse={reverse} v={v:?}"
            );
            assert!(
                (v.angular.y - 0.001).abs() < 3e-6,
                "reverse={reverse} v={v:?}"
            );
        }
        let force = Vec3::X * if reverse { -0.5 } else { 0.5 };
        assert!((world.distance_reading(link).unwrap().force - force).length() < 3e-4);
        assert!((world.joint_reaction(link).unwrap().child.linear - force).length() < 3e-4);
        assert!(world.distance_reading(link).unwrap().violation < 3e-6);
        trajectories.push(velocities);
    }
    for i in 0..2 {
        assert!((trajectories[0][i].linear - trajectories[1][i].linear).length() < 1e-7);
        assert!((trajectories[0][i].angular - trajectories[1][i].angular).length() < 1e-7);
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-mixed-contacts-gpu"]
fn loaded_contact_slider_hits_releases_and_mutates_both_stops() {
    let center = Vec3::Z * 0.49998;
    let (mut scene, b) = contact_scene(center, false);
    let slider = joint(&mut scene, b, JointType::Prismatic, center);
    scene
        .set_joint_limits(slider, JointAxis::LinX, Some([-0.01, 0.01]))
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, hardware_engine::engine()).unwrap();
    world.set_reaction_readout(true).unwrap();
    world.solver.substeps = 1;
    for (phase, (force, bound)) in [(1., 0.01_f64), (-1., -0.01), (1., 0.02)]
        .into_iter()
        .enumerate()
    {
        if phase == 2 {
            world
                .scene
                .set_joint_limits(slider, JointAxis::LinX, Some([-0.02, 0.02]))
                .unwrap();
        }
        world.scene.reset_body_forces(b).unwrap();
        world
            .scene
            .add_body_wrench(b, SpatialVector::new(Vec3::X * force, Vec3::ZERO))
            .unwrap();
        let mut moving = false;
        let mut warm_compilations = None;
        for frame in 0..100 {
            let before = world.scene.body_velocity(b).unwrap().linear.x;
            world.step(0.005).unwrap_or_else(|error| {
                panic!("phase={phase} frame={frame} before={before}: {error}")
            });
            let velocity = world.scene.body_velocity(b).unwrap().linear.x;
            let position = world.scene.body_pose(b).unwrap().position.x;
            moving |= velocity * force > 0.01;
            let contact = contact_load(&world, b, world.scene.body_pose(b).unwrap().position);
            let reaction = world.joint_reaction(slider).unwrap().child;
            let residual =
                (velocity - before) / 0.005 - force - contact.linear.x - reaction.linear.x;
            assert!(
                residual.abs() < 3e-4,
                "phase={phase} frame={frame} residual={residual}"
            );
            assert!(
                position.abs() <= bound.abs() + 3e-6,
                "phase={phase} frame={frame} x={position}"
            );
            let compilations = world.gpu_diagnostics().unwrap().model_compilations;
            assert_eq!(*warm_compilations.get_or_insert(compilations), compilations);
        }
        assert!(moving, "phase={phase} did not release the preceding stop");
        assert!((world.scene.body_pose(b).unwrap().position.x - bound).abs() < 3e-6);
        let velocity = world.scene.body_velocity(b).unwrap().linear.x;
        let position = world.scene.body_pose(b).unwrap().position.x;
        assert!(
            velocity.abs() < 1e-7,
            "phase={phase} velocity={velocity:e} position={position:e}"
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-mixed-contacts-gpu"]
fn runtime_locked_cone_and_contact_publish_complete_torque() {
    let engine = hardware_engine::engine();
    for speed in [0.001, 0.01] {
        let center = Vec3::Z * 0.49998;
        let (mut scene, b) = contact_scene(center, false);
        let ball = joint(&mut scene, b, JointType::Ball, center);
        scene
            .set_cone_limits(
                ball,
                SphericalConeLimits {
                    angle0: Some(0.),
                    angle1: Some(0.),
                },
            )
            .unwrap();
        scene
            .set_body_velocity(b, SpatialVector::new(Vec3::ZERO, Vec3::Y * speed))
            .unwrap();
        let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
        cpu.solver.substeps = 1;
        cpu.set_reaction_readout(true).unwrap();
        cpu.step(0.01).unwrap();
        let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        world.set_reaction_readout(true).unwrap();
        world.solver.substeps = 1;
        world.step(0.01).unwrap();
        assert!(world.scene.body_velocity(b).unwrap().angular.length() < 1e-7);
        let reaction = world.joint_reaction(ball).unwrap().child;
        let reference = cpu.joint_reaction(ball).unwrap().child;
        assert!(
            (reaction.angular - reference.angular).length() < 3e-4,
            "redundant cone/contact allocation: GPU={reaction:?} CPU={reference:?}"
        );
        let contact = contact_load(&world, b, center);
        assert!(
            (reaction.angular + contact.angular + Vec3::Y * (speed / 0.01)).length() < 3e-5,
            "reaction={reaction:?}, contact={contact:?}"
        );
        assert!((world.cone_reading(ball).unwrap().torque - reaction.angular).length() < 3e-5);
        assert_eq!(
            world.contact_pairs()[0].points[0].applied_friction,
            0.1_f32 as f64
        );
        if speed == 0.001 {
            let arm = world.contact_pairs()[0].points[0].point - center;
            let radius = arm.cross(Vec3::X).y;
            let expected = -(speed / 0.01) / (1. + radius * radius);
            assert!(
                (reaction.angular.y - expected).abs() < 3e-5,
                "interior minimum-norm reaction={reaction:?} expected={expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-mixed-contacts-gpu"]
fn redundant_cone_contact_break_decisions_match_away_from_threshold_ties() {
    let engine = hardware_engine::engine();
    for (threshold, broken) in [(0.898, true), (0.902, false)] {
        let center = Vec3::Z * 0.49998;
        let (mut scene, b) = contact_scene(center, false);
        let ball = joint(&mut scene, b, JointType::Ball, center);
        scene
            .set_cone_limits(
                ball,
                SphericalConeLimits {
                    angle0: Some(0.),
                    angle1: Some(0.),
                },
            )
            .unwrap();
        scene
            .set_body_velocity(b, SpatialVector::new(Vec3::ZERO, Vec3::Y * 0.01))
            .unwrap();
        scene
            .set_joint_break_limits(
                ball,
                JointBreakLimits {
                    force: None,
                    torque: Some(threshold),
                },
            )
            .unwrap();
        let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
        let mut gpu = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        cpu.solver.substeps = 1;
        gpu.solver.substeps = 1;
        cpu.step(0.01).unwrap();
        gpu.step(0.01).unwrap();
        assert_eq!(
            gpu.scene.joint_broken(ball).unwrap(),
            cpu.scene.joint_broken(ball).unwrap(),
            "GPU events={:?}, CPU events={:?}",
            gpu.joint_break_events(),
            cpu.joint_break_events()
        );
        assert_eq!(gpu.scene.joint_broken(ball).unwrap(), broken);
        assert_eq!(gpu.joint_break_events().len(), usize::from(broken));
        assert_eq!(cpu.joint_break_events().len(), usize::from(broken));
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-mixed-contacts-gpu"]
fn scalar_reservations_do_not_expand_contact_capacity_and_faults_roll_back() {
    let (mut scene, b) = contact_scene(Vec3::Z * 0.49998, true);
    let position = Vec3::new(3., 0., 2.);
    let ghost = body(&mut scene, position);
    let link = joint(&mut scene, ghost, JointType::Distance, position);
    scene
        .set_distance_limits(
            link,
            DistanceLimits {
                min: Some(0.),
                max: Some(0.),
            },
        )
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu_with_contact_capacity(
        scene,
        hardware_engine::engine(),
        GpuContactCapacity {
            pairs: 32,
            contacts_per_island: 1,
        },
    )
    .unwrap();
    world.solver.substeps = 2;
    let pose = world.scene.body_pose(b).unwrap();
    let error = world.step(0.002).unwrap_err();
    assert!(error.to_string().contains("2097152"), "{error}");
    assert_eq!(world.scene.body_pose(b).unwrap(), pose);
    assert_eq!(world.scene.body_pose(ghost).unwrap().position, position);
    assert_eq!(world.completed_steps(), 0);
    let compilations = world.gpu_diagnostics().unwrap().model_compilations;
    world
        .scene
        .set_body_pose(b, Transform::from_translation(Vec3::Z * 2.))
        .unwrap();
    world.step(0.002).unwrap();
    assert_eq!(
        world.gpu_diagnostics().unwrap().model_compilations,
        compilations
    );
    assert!(world.distance_reading(link).unwrap().violation < 1e-7);
    assert!((world.distance_reading(link).unwrap().force - Vec3::Z * 2.).length() < 1e-4);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-mixed-contacts-gpu"]
fn mixed_distance_contact_break_continues_remaining_substeps() {
    let (mut scene, b, j) = radial();
    scene
        .set_joint_break_limits(
            j,
            JointBreakLimits {
                force: Some(0.6),
                torque: None,
            },
        )
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, hardware_engine::engine()).unwrap();
    world.solver.substeps = 2;
    world.step(0.02).unwrap();
    assert!(world.scene.joint_broken(j).unwrap());
    assert_eq!(world.joint_break_events().len(), 1);
    let event = &world.joint_break_events()[0];
    assert_eq!(event.sample.substep, 0);
    assert!((event.sample.time_offset - 0.01).abs() < 1e-12);
    assert!((event.sample.child.linear - Vec3::new(0.5, -0.5, 0.)).length() < 3e-4);
    assert_eq!(world.contact_pairs()[0].points.len(), 2);
    assert_eq!(world.gpu_diagnostics().unwrap().model_compilations, 3);
    assert!(world.scene.body_velocity(b).unwrap().linear.is_finite());
}
