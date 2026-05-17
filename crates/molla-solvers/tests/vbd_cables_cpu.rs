//! Always-on CPU VBD cable gates.
//!
//! GPU cable coverage lives in ignored VBD GPU suites. These default-lane
//! checks keep the procedural CABLE path tied to physics invariants without
//! relying on imported assets.

use molla_core::{BodyId, WorldId};
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Contacts, ModelBuilder};
use molla_solvers::{Solver, VbdSolver};

fn add_curved_cable(
    b: &mut ModelBuilder,
    offset: Vec3,
    phase: Real,
    n: u32,
    stiffness: Real,
    bending: Real,
) -> usize {
    let mut ids = Vec::with_capacity(n as usize);
    for i in 0..n {
        let t = i as Real / (n - 1) as Real;
        let x = -0.42 + 0.84 * t;
        let y = 0.78 + 0.10 * (std::f64::consts::TAU * t + phase).sin();
        let z = 0.08 * (2.0 * std::f64::consts::TAU * t + phase).cos();
        let id = b.add_particle(offset + Vec3::new(x, y, z), 0.035, 0.025, WorldId(0));
        ids.push(id);
    }
    let cable_id = b.add_cable(&ids, stiffness, 0.05);
    b.set_cable_bending_stiffness(cable_id, bending);
    cable_id
}

#[test]
fn vbd_cable_pile_settles_on_floor_without_unbounded_stretch() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(1.2);
    let cable_count = 3usize;
    let n_per_cable = 14u32;
    for c in 0..cable_count {
        let z_off = (c as Real - 1.0) * 0.065;
        let y_off = c as Real * 0.08;
        add_curved_cable(
            &mut b,
            Vec3::new(0.0, y_off, z_off),
            c as Real * 0.9,
            n_per_cable,
            5.0e5,
            2.5e3,
        );
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    assert_eq!(model.cable_count, cable_count);

    let rest_lengths = model.cable_rest_length.host().unwrap().to_vec();
    let offsets = model.cable_particle_offsets.host().unwrap().to_vec();
    let cable_particles = model.cable_particles.host().unwrap().to_vec();
    let radii = model.particle_radius.host().unwrap().to_vec();

    let mut solver = VbdSolver::new_cpu()
        .with_iterations(18)
        .with_contact_stiffness(2.0e5);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(64);
    let dt: Real = 1.0 / 240.0;
    for _ in 0..900 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.particle_q.host().unwrap();
    let qd = s_a.particle_qd.host().unwrap();
    let mut min_clearance = Real::INFINITY;
    let mut max_speed: Real = 0.0;
    for (i, (p, v)) in q.iter().zip(qd.iter()).enumerate() {
        assert!(
            p.is_finite() && v.is_finite(),
            "cable particle {i} produced non-finite state: p={p:?}, v={v:?}"
        );
        min_clearance = min_clearance.min(p.y - radii[i]);
        max_speed = max_speed.max(v.length());
    }

    let mut max_length_ratio: Real = 0.0;
    for c in 0..cable_count {
        let lo = offsets[c] as usize;
        let hi = offsets[c + 1] as usize;
        let mut length = 0.0;
        for win in cable_particles[lo..hi].windows(2) {
            length += (q[win[1] as usize] - q[win[0] as usize]).length();
        }
        let ratio = length / rest_lengths[c];
        max_length_ratio = max_length_ratio.max(ratio);
        assert!(
            ratio < 1.06,
            "cable {c} stretched too much in pile: length/rest={ratio:.5}"
        );
    }

    eprintln!(
        "VBD cable pile: min_clearance={min_clearance:.5}, max_length_ratio={max_length_ratio:.5}, max_speed={max_speed:.5}"
    );
    assert!(
        min_clearance > -0.012,
        "cable pile penetrated floor by more than 1.2 cm: clearance={min_clearance}"
    );
    assert!(
        min_clearance < 0.08,
        "cable pile never reached the floor: min_clearance={min_clearance}"
    );
    assert!(
        max_speed < 0.35,
        "cable pile did not settle enough: max_speed={max_speed}"
    );
}
