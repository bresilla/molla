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
