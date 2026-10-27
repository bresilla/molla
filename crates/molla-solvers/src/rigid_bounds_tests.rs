use super::*;
use molla_sim::Heightfield;
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry};

fn ground(height: f64) -> ColliderDesc {
    ColliderDesc::new(ColliderGeometry::Heightfield(
        Heightfield::from_scaled(17, 17, vec![height; 17 * 17], Vec3::new(16.0, 1.0, 16.0))
            .unwrap(),
    ))
}

fn fixture() -> (RigidWorld, BodyHandle, ColliderHandle) {
    let mut world = RigidWorld::new().unwrap();
    let floor = world.scene.insert_collider(ground(0.0)).unwrap();
    let mut body = BodyDesc::default();
    body.params.pose.position.y = 0.4;
    let body = world.scene.insert_body(body).unwrap();
    let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.4 });
    sphere.parent = Some(body);
    world.scene.insert_collider(sphere).unwrap();
    (world, body, floor)
}

fn same(a: &RigidWorld, b: &RigidWorld) {
    let sa = a.scene.state();
    let sb = b.scene.state();
    macro_rules! state_equal {
        ($($field:ident),+) => { $(assert_eq!(sa.$field.host().unwrap(), sb.$field.host().unwrap(), stringify!($field));)+ };
    }
    state_equal!(body_q, body_qd, body_f, joint_q, joint_qd);
    assert_eq!(a.contacts.count, b.contacts.count);
    let n = a.contacts.count as usize;
    macro_rules! contact_equal {
        ($($field:ident),+) => { $(assert_eq!(a.contacts.$field.host().unwrap()[..n], b.contacts.$field.host().unwrap()[..n], stringify!($field));)+ };
    }
    contact_equal!(
        a_index, b_index, a_shape, b_shape, point, normal, depth, friction
    );
}

#[test]
fn retained_bounds_match_rebuilt_bounds_through_edits_sleep_and_failure() {
    let (mut retained, a_body, mut a_floor) = fixture();
    let (mut rebuilt, b_body, mut b_floor) = fixture();
    let mut sleeping_steps = 0;
    for tick in 0..600 {
        for (world, body, floor) in [
            (&mut retained, a_body, &mut a_floor),
            (&mut rebuilt, b_body, &mut b_floor),
        ] {
            match tick {
                100 => world.scene.replace_collider(*floor, ground(0.02)).unwrap(),
                200 => world
                    .scene
                    .set_collider_pose(
                        *floor,
                        Transform::new(Vec3::ZERO, molla_math::Quat::from_rotation_x(0.03)),
                    )
                    .unwrap(),
                250 => {
                    world.scene.remove_collider(*floor).unwrap();
                    *floor = world.scene.insert_collider(ground(0.0)).unwrap();
                }
                400 => {
                    world.sleep_body(body).unwrap();
                }
                420 => {
                    assert!(
                        world
                            .step_with_forces(
                                1.0 / 120.0,
                                |_, _| false,
                                |_, _, _, _| Err(Error::Solver("fixture failure".into()))
                            )
                            .is_err()
                    );
                }
                440 => world.wake_body(body).unwrap(),
                480 => {
                    let mut invalid = ground(0.0);
                    invalid.pose.position.x = f64::NAN;
                    assert!(world.scene.replace_collider(*floor, invalid).is_err());
                }
                _ => {}
            }
        }
        rebuilt.bounds = molla_geometry::shape_world::SceneShapeBounds::default();
        retained
            .step(1.0 / 120.0)
            .unwrap_or_else(|e| panic!("retained tick={tick}: {e}"));
        rebuilt
            .step(1.0 / 120.0)
            .unwrap_or_else(|e| panic!("rebuilt tick={tick}: {e}"));
        same(&retained, &rebuilt);
        assert_eq!(retained.is_sleeping(a_body), rebuilt.is_sleeping(b_body));
        sleeping_steps += usize::from(retained.is_sleeping(a_body));
    }
    assert!(sleeping_steps >= 10);
}
