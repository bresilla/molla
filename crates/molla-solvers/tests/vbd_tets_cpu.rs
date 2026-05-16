//! Default CPU VBD tet soft-rigid contact gates.
//!
//! GPU VBD tet coverage lives in ignored `vbd_tets*_gpu.rs` suites. These
//! small CPU cases keep softbody/rigid contact behavior covered in the
//! always-on lane.

use molla_core::{BodyId, WorldId};
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Contacts, MaterialNeoHookean, ModelBuilder};
use molla_solvers::{Solver, VbdSolver};

fn cube_to_six_tets(corners: [u32; 8]) -> [[u32; 4]; 6] {
    let v000 = corners[0];
    let v100 = corners[1];
    let v010 = corners[2];
    let v110 = corners[3];
    let v001 = corners[4];
    let v101 = corners[5];
    let v011 = corners[6];
    let v111 = corners[7];
    [
        [v000, v100, v110, v111],
        [v000, v110, v010, v111],
        [v000, v010, v011, v111],
        [v000, v011, v001, v111],
        [v000, v001, v101, v111],
        [v000, v101, v100, v111],
    ]
}

#[test]
fn vbd_tet_cube_drops_on_kinematic_sphere_without_penetration() {
    let cube_center = Vec3::new(0.0, 1.45, 0.0);
    let side: Real = 0.36;
    let n = 3u32;
    let sphere_center = Vec3::new(0.0, 0.62, 0.0);
    let sphere_radius: Real = 0.30;

    let mut b = ModelBuilder::new();
    b.set_linear_damping(2.0);
    let dx = side / (n as Real - 1.0);
    let origin = cube_center - Vec3::splat(side * 0.5);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                b.add_particle(
                    origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx),
                    0.05,
                    dx * 0.5,
                    WorldId(0),
                );
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
    let mut tets = Vec::new();
    for kk in 0..n - 1 {
        for jj in 0..n - 1 {
            for ii in 0..n - 1 {
                tets.extend_from_slice(&cube_to_six_tets([
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ]));
            }
        }
    }
    let materials = vec![MaterialNeoHookean::elastic(1.0e5, 0.4)];
    let mat_idx = vec![0; tets.len()];
    b.add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .unwrap();
    b.add_shape_sphere(
        BodyId::NONE,
        sphere_radius,
        Transform::from_translation(sphere_center),
    )
    .unwrap();
    let model = b.build().unwrap();

    let mut solver = VbdSolver::new_cpu()
        .with_iterations(12)
        .with_contact_stiffness(1.0e6);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(64);
    let dt: Real = 1.0 / 240.0;
    let mut max_penetration: Real = 0.0;
    let mut min_center_dist: Real = Real::INFINITY;
    for _ in 0..900 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        for p in s_a.particle_q.host().unwrap() {
            let dist = (*p - sphere_center).length();
            min_center_dist = min_center_dist.min(dist);
            max_penetration = max_penetration.max(sphere_radius - dist);
        }
    }

    let q = s_a.particle_q.host().unwrap();
    let qd = s_a.particle_qd.host().unwrap();
    let max_speed = qd.iter().map(|v| v.length()).fold(0.0, Real::max);
    for (i, (p, v)) in q.iter().zip(qd.iter()).enumerate() {
        assert!(
            p.is_finite() && v.is_finite(),
            "tet particle {i} produced non-finite state: p={p:?}, v={v:?}"
        );
    }
    eprintln!(
        "VBD tet/sphere CPU contact: max_penetration={max_penetration:.5}, min_center_dist={min_center_dist:.5}, max_speed={max_speed:.5}"
    );
    assert!(
        max_penetration < 1.0e-2,
        "tet particles penetrated kinematic sphere by {max_penetration:.5} m"
    );
    assert!(
        min_center_dist < sphere_radius + side,
        "tet cube never reached the sphere: min_center_dist={min_center_dist}"
    );
    assert!(
        max_speed < 0.50,
        "tet cube still moving too fast after sphere contact: max_speed={max_speed}"
    );
}
