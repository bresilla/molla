use molla_math::{Mat3, SpatialVector, Vec3};
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry};
use molla_sim::{ColliderMass, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

#[test]
fn recomputed_mass_controls_featherstone_force_response() {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let body = world.scene.insert_body(BodyDesc::default()).unwrap();
    let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    sphere.parent = Some(body);
    sphere.mass = ColliderMass::Mass(2.0);
    let collider = world.scene.insert_collider(sphere).unwrap();
    assert_eq!(world.collider_mass_properties(collider).unwrap().mass, 2.0);
    world
        .recompute_body_mass(
            body,
            MassProperties {
                mass: 4.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            },
        )
        .unwrap();
    world.scene.state_mut().body_f.host_mut().unwrap()[0] =
        SpatialVector::new(Vec3::X * 12.0, Vec3::ZERO);
    world.step(0.01).unwrap();
    let velocity = world.scene.state().body_qd.host().unwrap()[0];
    assert!((velocity.linear.x - 0.02).abs() < 1e-9, "{velocity:?}");
    let stamp = world.scene.collider_stamp(collider).unwrap();
    world
        .scene
        .set_collider_mass(collider, ColliderMass::Mass(12.0))
        .unwrap();
    assert_eq!(world.scene.collider_stamp(collider).unwrap(), stamp);
    world
        .recompute_body_mass(body, MassProperties::ZERO)
        .unwrap();
    world.step(0.01).unwrap();
    let velocity = world.scene.state().body_qd.host().unwrap()[0];
    assert!((velocity.linear.x - 0.03).abs() < 1e-9, "{velocity:?}");
    assert!(velocity.angular.length() < 1e-10);
}
