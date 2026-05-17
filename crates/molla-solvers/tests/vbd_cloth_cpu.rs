//! Default CPU VBD cloth gates for Tier-2 cloth depth.
//!
//! GPU VBD coverage lives in ignored GPU tests. These small CPU tests
//! keep the VBD cloth/bending path covered in the always-on lane.

use molla_core::{BodyId, WorldId};
use molla_geometry::SdfGrid;
use molla_math::{Real, Transform, Vec3};
use molla_sim::{BodyFlags, BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, VbdSolver};
use std::sync::Arc;

const NX: u32 = 8;
const NY: u32 = 8;

fn build_cantilever(bending_compliance: Real) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let (first, _, _) = b.add_cloth_grid_with_bending(
        Vec3::new(0.0, 1.5, -0.35),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.70),
        NX,
        NY,
        0.04,
        0.02,
        5.0e-6,
        0.05,
        bending_compliance,
        WorldId(0),
    );
    // Pin the root edge, leaving the far edge free to sag.
    for iy in 0..NY {
        b.pin_particle(first + iy * NX);
    }
    b.build().unwrap()
}

fn run_vbd_with_contact(model: &Model, steps: usize, contact_stiffness: Real) -> Vec<Vec3> {
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(12)
        .with_contact_stiffness(contact_stiffness);
    let dt: Real = 1.0 / 240.0;
    for _ in 0..steps {
        Solver::step(&mut solver, model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.host().unwrap().to_vec()
}

fn run_vbd(model: &Model, steps: usize) -> Vec<Vec3> {
    run_vbd_with_contact(model, steps, 0.0)
}

fn build_twisted_strip(bending_compliance: Real) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.25);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    let n = 7u32;
    let (first, _, _) = b.add_cloth_grid_with_bending(
        Vec3::new(-0.6, 1.0, -0.45),
        Vec3::new(1.2, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.9),
        n,
        n,
        0.04,
        0.02,
        5.0e-6,
        0.12,
        bending_compliance,
        WorldId(0),
    );
    for iy in 0..n {
        b.pin_particle(first + iy * n);
    }
    b.build().unwrap()
}

fn run_twisted_strip(model: &Model) -> (Real, Vec<Vec3>) {
    let n = 7usize;
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    {
        let q = s_a.particle_q.host_mut().unwrap();
        for iy in 0..n {
            let sign = if iy < n / 2 { 1.0 } else { -1.0 };
            for ix in 1..n {
                let weight = ix as Real / (n - 1) as Real;
                q[iy * n + ix].y += sign * 0.18 * weight;
            }
        }
    }
    let initial = s_a.particle_q.host().unwrap().to_vec();
    let initial_twist = strip_tip_twist(&initial, n);
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = VbdSolver::new_cpu().with_iterations(12);
    let dt: Real = 1.0 / 240.0;
    for _ in 0..240 {
        Solver::step(&mut solver, model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let final_q = s_a.particle_q.host().unwrap().to_vec();
    (initial_twist, final_q)
}

fn strip_tip_twist(q: &[Vec3], n: usize) -> Real {
    let ix = n - 1;
    let low = (0..n / 2).map(|iy| q[iy * n + ix].y).sum::<Real>() / (n / 2) as Real;
    let high = (n / 2 + 1..n).map(|iy| q[iy * n + ix].y).sum::<Real>() / (n / 2) as Real;
    (low - high).abs()
}

fn analytic_sphere_sdf_grid(radius: Real, half_width: Real, n: u32) -> SdfGrid {
    let cell_size = (2.0 * half_width) / (n - 1) as Real;
    let origin = Vec3::splat(-half_width);
    let mut data = Vec::with_capacity((n * n * n) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = origin + Vec3::new(x as Real, y as Real, z as Real) * cell_size;
                data.push(p.length() - radius);
            }
        }
    }
    SdfGrid {
        nx: n,
        ny: n,
        nz: n,
        origin,
        cell_size,
        data,
    }
}

fn free_edge_mean_y(q: &[Vec3]) -> Real {
    let mut sum = 0.0;
    for iy in 0..NY {
        sum += q[(iy * NX + (NX - 1)) as usize].y;
    }
    sum / NY as Real
}

#[test]
fn vbd_cloth_bending_compliance_changes_cantilever_sag() {
    let low_compliance_model = build_cantilever(1.0e-7);
    let high_compliance_model = build_cantilever(5.0e-3);

    let low_q = run_vbd(&low_compliance_model, 240);
    let high_q = run_vbd(&high_compliance_model, 240);

    let low_y = free_edge_mean_y(&low_q);
    let high_y = free_edge_mean_y(&high_q);
    let low_sag = 1.5 - low_y;
    let high_sag = 1.5 - high_y;
    let delta = (low_sag - high_sag).abs();
    eprintln!("VBD cantilever sag: low-compliance={low_sag:.5} m, high-compliance={high_sag:.5} m");

    assert!(
        low_q
            .iter()
            .chain(high_q.iter())
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite()),
        "VBD cloth produced non-finite positions"
    );
    assert!(
        delta > high_sag.max(low_sag) * 0.05,
        "bending compliance had too little effect: low={low_sag:.5}, high={high_sag:.5}"
    );
}

#[test]
fn vbd_cloth_bending_springs_change_twist_response() {
    let stiff_bending = build_twisted_strip(1.0e-7);
    let no_bending = build_twisted_strip(-1.0);

    let (initial_twist, stiff_q) = run_twisted_strip(&stiff_bending);
    let (_, no_bend_q) = run_twisted_strip(&no_bending);
    let stiff_twist = strip_tip_twist(&stiff_q, 7);
    let no_bend_twist = strip_tip_twist(&no_bend_q, 7);
    eprintln!(
        "VBD twisted strip: initial={initial_twist:.5}, stiff_bending={stiff_twist:.5}, no_bending={no_bend_twist:.5}"
    );

    assert!(
        stiff_q
            .iter()
            .chain(no_bend_q.iter())
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite()),
        "VBD twisted strip produced non-finite positions"
    );
    assert!(
        no_bend_twist < initial_twist * 0.25,
        "no-bending strip should damp the seeded twist away through membrane/shear springs: initial={initial_twist}, final={no_bend_twist}",
    );
    assert!(
        stiff_twist > no_bend_twist * 5.0,
        "skip-one bending springs should materially change the twist response: stiff={stiff_twist}, no_bending={no_bend_twist}",
    );
}

#[test]
fn vbd_cloth_pinned_root_edge_stays_fixed() {
    let model = build_cantilever(1.0e-6);
    let initial = model.state().unwrap().particle_q.host().unwrap().to_vec();
    let final_q = run_vbd(&model, 120);

    for iy in 0..NY {
        let i = (iy * NX) as usize;
        let drift = (final_q[i] - initial[i]).length();
        assert!(
            drift < 1.0e-10,
            "pinned root particle {i} drifted by {drift:.3e}"
        );
    }
}

#[test]
fn vbd_cloth_cpu_wraps_kinematic_sphere_without_penetration() {
    let n = 10u32;
    let size = 1.4;
    let top_y = 2.3;
    let sphere_center = Vec3::new(0.0, 1.0, 0.0);
    let sphere_radius = 0.45;
    let particle_radius = 0.02;

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let _ = b.add_cloth_grid(
        Vec3::new(-size * 0.5, top_y, -size * 0.5),
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, 0.0, size),
        n,
        n,
        0.05,
        particle_radius,
        5.0e-6,
        0.10,
        WorldId(0),
    );
    b.add_shape_sphere(
        BodyId::NONE,
        sphere_radius,
        Transform::from_translation(sphere_center),
    )
    .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let pos = run_vbd_with_contact(&model, 360, 5.0e4);

    for (i, p) in pos.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "particle {i} became non-finite: {p:?}"
        );
        let d = (*p - sphere_center).length();
        assert!(
            d >= sphere_radius - 3.0 * particle_radius,
            "particle {i} penetrated the sphere: d={d:.4}, radius={sphere_radius}"
        );
    }

    let mean_y = pos.iter().map(|p| p.y).sum::<Real>() / pos.len() as Real;
    assert!(
        mean_y < top_y - 0.4,
        "cloth did not fall toward the sphere: mean_y={mean_y:.4}, top_y={top_y}"
    );

    let touch_band = sphere_radius + 4.0 * particle_radius;
    let touching = pos
        .iter()
        .filter(|p| (**p - sphere_center).length() <= touch_band)
        .count();
    assert!(
        touching >= pos.len() / 20,
        "cloth never reached the kinematic sphere: touching={touching}, total={}",
        pos.len()
    );
}

#[test]
fn vbd_cloth_cpu_drapes_over_kinematic_box_without_penetration() {
    let n = 10u32;
    let size = 1.4;
    let top_y = 2.25;
    let box_center = Vec3::new(0.0, 1.0, 0.0);
    let half = Vec3::new(0.38, 0.16, 0.32);
    let particle_radius = 0.02;

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.45);
    let _ = b.add_cloth_grid(
        Vec3::new(-size * 0.5, top_y, -size * 0.5),
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, 0.0, size),
        n,
        n,
        0.05,
        particle_radius,
        5.0e-6,
        0.10,
        WorldId(0),
    );
    b.add_shape_box(BodyId::NONE, half, Transform::from_translation(box_center))
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let pos = run_vbd_with_contact(&model, 420, 6.0e4);

    let signed_distance_to_box = |p: Vec3| -> Real {
        let local = p - box_center;
        let q = Vec3::new(local.x.abs(), local.y.abs(), local.z.abs()) - half;
        let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
        outside.length() + q.x.max(q.y.max(q.z)).min(0.0)
    };

    let mut max_penetration: Real = 0.0;
    let mut touching = 0usize;
    for (i, p) in pos.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "box-drape cloth particle {i} became non-finite: {p:?}"
        );
        let clearance = signed_distance_to_box(*p);
        max_penetration = max_penetration.max(-clearance);
        if clearance <= 4.0 * particle_radius {
            touching += 1;
        }
    }

    let mean_y = pos.iter().map(|p| p.y).sum::<Real>() / pos.len() as Real;
    eprintln!(
        "VBD cloth/box drape: mean_y={mean_y:.4}, touching={touching}, max_penetration={max_penetration:.5}"
    );
    assert!(
        max_penetration <= 4.0 * particle_radius,
        "cloth penetrated kinematic box too far: max penetration={max_penetration}"
    );
    assert!(
        touching >= pos.len() / 25,
        "cloth never reached the kinematic box: touching={touching}, total={}",
        pos.len()
    );
    assert!(
        mean_y < top_y - 0.35,
        "cloth did not fall toward the kinematic box: mean_y={mean_y:.4}, top_y={top_y}"
    );
}

#[test]
fn vbd_cloth_cpu_wraps_dense_sdf_sphere_without_penetration() {
    let n = 10u32;
    let size = 1.4;
    let top_y = 2.25;
    let sphere_center = Vec3::new(0.0, 1.0, 0.0);
    let sphere_radius = 0.45;
    let particle_radius = 0.02;

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.45);
    let _ = b.add_cloth_grid(
        Vec3::new(-size * 0.5, top_y, -size * 0.5),
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, 0.0, size),
        n,
        n,
        0.05,
        particle_radius,
        5.0e-6,
        0.10,
        WorldId(0),
    );
    let model = b.build().unwrap();
    let sdf = Arc::new(analytic_sphere_sdf_grid(sphere_radius, 1.25, 51));

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(12)
        .with_contact_stiffness(5.0e4)
        .with_sdf_grid(sdf, Transform::from_translation(sphere_center));
    let dt: Real = 1.0 / 240.0;
    let touch_band = sphere_radius + 4.0 * particle_radius;
    let mut max_penetration: Real = 0.0;
    let mut max_touching = 0usize;
    for _ in 0..360 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.particle_q.host().unwrap();
        let mut touching = 0usize;
        for p in q {
            let clearance = (*p - sphere_center).length() - sphere_radius;
            max_penetration = max_penetration.max(-clearance);
            if (*p - sphere_center).length() <= touch_band {
                touching += 1;
            }
        }
        max_touching = max_touching.max(touching);
    }

    let pos = s_a.particle_q.host().unwrap();
    for (i, p) in pos.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "dense-SDF cloth particle {i} became non-finite: {p:?}"
        );
    }
    eprintln!(
        "dense SDF cloth/sphere max penetration = {max_penetration:.5} m, max_touching={max_touching}"
    );
    assert!(
        max_penetration <= 3.0 * particle_radius,
        "cloth penetrated dense SDF sphere too far: max penetration={max_penetration}"
    );

    let mean_y = pos.iter().map(|p| p.y).sum::<Real>() / pos.len() as Real;
    assert!(
        mean_y < top_y - 0.35,
        "cloth did not fall toward the dense SDF sphere: mean_y={mean_y:.4}, top_y={top_y}"
    );

    assert!(
        max_touching >= pos.len() / 20,
        "cloth never reached the dense SDF sphere: max_touching={max_touching}, total={}",
        pos.len()
    );
}

#[test]
fn vbd_cloth_cpu_tracks_moving_kinematic_sphere_without_penetration() {
    let n = 9u32;
    let size = 1.1;
    let cloth_y = 1.0;
    let radius = 0.25;
    let particle_radius = 0.018;
    let sphere_start = Vec3::new(0.0, 0.52, 0.0);
    let sphere_end = Vec3::new(0.0, 0.84, 0.0);

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.7);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    b.add_cloth_grid(
        Vec3::new(-size * 0.5, cloth_y, -size * 0.5),
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, 0.0, size),
        n,
        n,
        0.04,
        particle_radius,
        5.0e-6,
        0.10,
        WorldId(0),
    );
    let sphere = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(sphere_start))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(sphere, radius, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(14)
        .with_contact_stiffness(8.0e4);
    let dt: Real = 1.0 / 240.0;
    let mut max_penetration: Real = 0.0;
    let mut max_mean_y = cloth_y;
    let mut max_touching = 0usize;
    for step in 0..360 {
        let alpha = (step as Real / 180.0).min(1.0);
        let center = sphere_start.lerp(sphere_end, alpha);
        s_a.body_q.host_mut().unwrap()[sphere.index()].position = center;
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.particle_q.host().unwrap();
        let mean_y = q.iter().map(|p| p.y).sum::<Real>() / q.len() as Real;
        max_mean_y = max_mean_y.max(mean_y);
        let mut touching = 0usize;
        for p in q {
            let clearance = (*p - center).length() - radius;
            max_penetration = max_penetration.max(-clearance);
            if clearance <= 4.0 * particle_radius {
                touching += 1;
            }
        }
        max_touching = max_touching.max(touching);
    }

    let q = s_a.particle_q.host().unwrap();
    let qd = s_a.particle_qd.host().unwrap();
    for (i, (p, v)) in q.iter().zip(qd.iter()).enumerate() {
        assert!(
            p.is_finite() && v.is_finite(),
            "moving-sphere cloth particle {i} became non-finite: p={p:?}, v={v:?}"
        );
    }
    eprintln!(
        "VBD moving sphere cloth: max_penetration={max_penetration:.5}, max_mean_y={max_mean_y:.5}, max_touching={max_touching}"
    );
    assert!(
        max_penetration <= 4.0 * particle_radius,
        "moving kinematic sphere penetrated cloth too far: max penetration={max_penetration}"
    );
    assert!(
        max_mean_y > cloth_y + 0.025,
        "moving kinematic sphere did not lift the cloth: max_mean_y={max_mean_y}, cloth_y={cloth_y}"
    );
    assert!(
        max_touching >= q.len() / 20,
        "cloth never contacted moving kinematic sphere: max_touching={max_touching}, total={}",
        q.len()
    );
}

#[test]
fn vbd_cloth_constraints_stay_bounded_under_fast_rigid_impact() {
    let n = 8u32;
    let size = 1.0;
    let rest = size / (n - 1) as Real;
    let cloth_y = 1.0;
    let radius = 0.24;
    let particle_radius = 0.018;
    let sphere_start = Vec3::new(0.0, 0.48, 0.0);
    let sphere_end = Vec3::new(0.0, 1.08, 0.0);

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.55);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    b.add_cloth_grid(
        Vec3::new(-size * 0.5, cloth_y, -size * 0.5),
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, 0.0, size),
        n,
        n,
        0.05,
        particle_radius,
        2.0e-6,
        0.10,
        WorldId(0),
    );
    let sphere = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(sphere_start))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(sphere, radius, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(16)
        .with_contact_stiffness(1.2e5);
    let dt: Real = 1.0 / 240.0;
    let mut max_edge_ratio: Real = 1.0;
    let mut max_penetration: Real = 0.0;
    let mut max_mean_y = cloth_y;
    for step in 0..240 {
        let alpha = (step as Real / 60.0).min(1.0);
        let center = sphere_start.lerp(sphere_end, alpha);
        s_a.body_q.host_mut().unwrap()[sphere.index()].position = center;
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);

        let q = s_a.particle_q.host().unwrap();
        let mean_y = q.iter().map(|p| p.y).sum::<Real>() / q.len() as Real;
        max_mean_y = max_mean_y.max(mean_y);
        for iy in 0..n as usize {
            for ix in 0..n as usize {
                let i = iy * n as usize + ix;
                if ix + 1 < n as usize {
                    max_edge_ratio = max_edge_ratio.max((q[i + 1] - q[i]).length() / rest);
                }
                if iy + 1 < n as usize {
                    max_edge_ratio = max_edge_ratio.max((q[i + n as usize] - q[i]).length() / rest);
                }
                let clearance = (q[i] - center).length() - radius;
                max_penetration = max_penetration.max(-clearance);
            }
        }
    }

    let q = s_a.particle_q.host().unwrap();
    let qd = s_a.particle_qd.host().unwrap();
    for (i, (p, v)) in q.iter().zip(qd.iter()).enumerate() {
        assert!(
            p.is_finite() && v.is_finite(),
            "fast-impact cloth particle {i} became non-finite: p={p:?}, v={v:?}"
        );
    }
    eprintln!(
        "VBD fast rigid impact: max_edge_ratio={max_edge_ratio:.4}, max_penetration={max_penetration:.5}, max_mean_y={max_mean_y:.5}"
    );
    assert!(
        max_edge_ratio < 1.85,
        "cloth structural constraints stretched too far under fast rigid impact: ratio={max_edge_ratio}"
    );
    assert!(
        max_penetration <= 5.0 * particle_radius,
        "fast kinematic sphere penetrated cloth too far: max penetration={max_penetration}"
    );
    assert!(
        max_mean_y > cloth_y + 0.06,
        "fast kinematic impact did not transfer motion to cloth: max_mean_y={max_mean_y}"
    );
}

#[test]
fn vbd_cloth_tracks_two_link_kinematic_articulation_without_penetration() {
    let n = 9u32;
    let size = 1.15;
    let cloth_y = 1.05;
    let radius = 0.18;
    let particle_radius = 0.018;
    let base_start = Vec3::new(-0.10, 0.72, 0.0);
    let base_end = Vec3::new(-0.10, 0.98, 0.0);
    let link_len = 0.38;

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.65);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    b.add_cloth_grid(
        Vec3::new(-size * 0.5, cloth_y, -size * 0.5),
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, 0.0, size),
        n,
        n,
        0.04,
        particle_radius,
        5.0e-6,
        0.10,
        WorldId(0),
    );
    let base = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(base_start))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(base, radius, Transform::IDENTITY)
        .unwrap();
    let tip_start = base_start + Vec3::new(link_len, 0.0, 0.0);
    let tip = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::from_translation(tip_start))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(tip, radius, Transform::IDENTITY)
        .unwrap();
    b.add_joint_revolute(
        base,
        tip,
        Transform::IDENTITY,
        Transform::from_translation(Vec3::new(-link_len, 0.0, 0.0)),
        Vec3::Z,
    )
    .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(14)
        .with_contact_stiffness(9.0e4);
    let dt: Real = 1.0 / 240.0;
    let mut max_penetration: Real = 0.0;
    let mut max_mean_y = cloth_y;
    let mut base_touching_max = 0usize;
    let mut tip_touching_max = 0usize;
    for step in 0..360 {
        let alpha = (step as Real / 160.0).min(1.0);
        let base_center = base_start.lerp(base_end, alpha);
        let theta = -0.55 + 1.10 * alpha;
        let tip_center =
            base_center + Vec3::new(theta.cos() * link_len, theta.sin() * link_len, 0.0);
        {
            let body_q = s_a.body_q.host_mut().unwrap();
            body_q[base.index()].position = base_center;
            body_q[tip.index()].position = tip_center;
        }
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);

        let q = s_a.particle_q.host().unwrap();
        let mean_y = q.iter().map(|p| p.y).sum::<Real>() / q.len() as Real;
        max_mean_y = max_mean_y.max(mean_y);
        let mut base_touching = 0usize;
        let mut tip_touching = 0usize;
        for p in q {
            let base_clearance = (*p - base_center).length() - radius;
            let tip_clearance = (*p - tip_center).length() - radius;
            max_penetration = max_penetration.max(-base_clearance).max(-tip_clearance);
            if base_clearance <= 4.0 * particle_radius {
                base_touching += 1;
            }
            if tip_clearance <= 4.0 * particle_radius {
                tip_touching += 1;
            }
        }
        base_touching_max = base_touching_max.max(base_touching);
        tip_touching_max = tip_touching_max.max(tip_touching);
    }

    let q = s_a.particle_q.host().unwrap();
    let qd = s_a.particle_qd.host().unwrap();
    for (i, (p, v)) in q.iter().zip(qd.iter()).enumerate() {
        assert!(
            p.is_finite() && v.is_finite(),
            "two-link articulated cloth particle {i} became non-finite: p={p:?}, v={v:?}"
        );
    }
    eprintln!(
        "VBD two-link kinematic articulation: max_penetration={max_penetration:.5}, \
         max_mean_y={max_mean_y:.5}, base_touching={base_touching_max}, tip_touching={tip_touching_max}"
    );
    assert!(
        max_penetration <= 4.5 * particle_radius,
        "two-link kinematic articulation penetrated cloth too far: max penetration={max_penetration}"
    );
    assert!(
        max_mean_y > cloth_y + 0.035,
        "two-link kinematic articulation did not lift the cloth: max_mean_y={max_mean_y}"
    );
    assert!(
        base_touching_max > 0 && tip_touching_max > 0,
        "both articulated links should contact cloth: base={base_touching_max}, tip={tip_touching_max}"
    );
}
