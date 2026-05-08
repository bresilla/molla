//! Stage 0 smoke test — exercises math + storage + sim type construction
//! without any GPU dependency.

use molla::math::{Quat, SpatialInertia, Transform, Vec3};
use molla::sim::{Contacts, Control, Model, State};
use molla::storage::{Buffer, BufferUsage};

#[test]
fn build_empty_model_state_control_contacts() {
    let m = Model::empty();
    assert_eq!(m.body_count, 0);
    assert_eq!(m.world_count, 1);

    let s = State::empty();
    assert!(s.body_q.host().unwrap().is_empty());

    let c = Control::empty();
    assert!(c.joint_force.host().unwrap().is_empty());

    let mut contacts = Contacts::with_capacity(64);
    assert_eq!(contacts.count, 0);
    contacts.clear();
    assert_eq!(contacts.count, 0);
}

#[test]
fn transform_compose_inverse_round_trips() {
    let t = Transform::new(Vec3::new(1.0, -2.0, 3.0), Quat::from_rotation_y(0.5));
    let id = t.compose(&t.inverse());
    let p = Vec3::new(0.7, 0.0, -0.4);
    let p_back = id.transform_point(p);
    assert!((p_back - p).length() < 1e-12);
}

#[test]
fn spatial_inertia_helpers_match_textbook() {
    let si = SpatialInertia::sphere(1.0, 1.0);
    // I_sphere = (2/5) m r^2 = 0.4
    assert!((si.inertia.x_axis.x - 0.4).abs() < 1e-12);

    let cb = SpatialInertia::cuboid(1.0, Vec3::new(2.0, 2.0, 2.0));
    // For a cube of side 2 and mass 1, I = (1/12) * 1 * (4 + 4) = 0.6667
    assert!((cb.inertia.x_axis.x - (8.0 / 12.0)).abs() < 1e-12);
}

#[test]
fn buffer_host_mut_round_trips() {
    let mut b: Buffer<u32> = Buffer::from_vec(vec![0; 4], BufferUsage::StorageReadWrite);
    for (i, slot) in b.host_mut().unwrap().iter_mut().enumerate() {
        *slot = i as u32 * 10;
    }
    assert_eq!(b.host().unwrap(), &[0, 10, 20, 30]);
}
