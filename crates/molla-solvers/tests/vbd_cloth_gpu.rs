//! VBD cloth solver — Stage 20 GPU integration tests. All
//! `#[ignore]` because they need a real wgpu adapter (run via
//! `nixVulkan cargo test -p molla-solvers --features gpu --test vbd_cloth_gpu
//! -- --ignored`).
//!
//! Three regression scenarios:
//!
//! 1. `vbd_matches_xpbd_low_stiffness` — for soft cloth (compliance
//!    ≥ 1e-4 m/N) the steady-state drape from VBD should be close
//!    to XPBD: same energy minimum, same gravity loading. Tolerance
//!    is loose because the two solvers have different convergence
//!    behaviour, but they shouldn't diverge to a wildly different
//!    drape.
//! 2. `vbd_high_stiffness_does_not_explode` — at compliance ~1e-7
//!    (effective stiffness 1e7 N/m, the upper end of "stiff cloth")
//!    XPBD without strain limiting tends to oscillate or blow up
//!    within ~60 substeps. VBD should settle into a stable drape.
//! 3. `vbd_wraps_kinematic_sphere` — the headline AVBD demo: a
//!    cloth sheet released over a static rigid sphere should drape
//!    around it. We assert (a) no particle ends up inside the
//!    sphere, (b) the cloth's centre-of-mass is below the initial
//!    drop height (it actually fell), (c) at least some particles
//!    are within `2 * particle_radius` of the sphere surface (it
//!    actually contacted the sphere).

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, WorldId};
use molla_geometry::CollisionPipeline;
use molla_math::{Real, Transform, Vec3};
use molla_sim::{BodyFlags, BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, VbdSolver, XpbdSolver};

/// Build an N×N horizontal cloth grid centred on the origin at
/// height `top_y`. Optional kinematic sphere obstacle below the
/// cloth. Compliance / stretch arguments let tests pick the regime.
fn build_cloth(
    n: u32,
    size: f64,
    top_y: f64,
    compliance: f64,
    max_stretch: f64,
    sphere: Option<(Vec3, f64)>,
) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let origin = Vec3::new(-size * 0.5, top_y, -size * 0.5);
    let u_axis = Vec3::new(size, 0.0, 0.0);
    let v_axis = Vec3::new(0.0, 0.0, size);
    let _ = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        n,
        n,
        0.05,
        0.02,
        compliance,
        max_stretch,
        WorldId(0),
    );
    if let Some((c, r)) = sphere {
        b.add_shape_sphere(BodyId::NONE, r, Transform::from_translation(c))
            .unwrap();
    }
    // Floor catch.
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().unwrap()
}

fn run_xpbd_gpu(model: &mut Model, dt: Real, substeps: usize, iters: usize) -> Vec<Vec3> {
    let engine = Engine::new_gpu().expect("GPU adapter for XPBD baseline");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();
    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count.max(1) as u32)
        .unwrap()
        .with_iterations(iters);
    let mut pipeline = CollisionPipeline::new();
    for _ in 0..substeps {
        pipeline.collide(model, &s_a, &mut contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    s_a.particle_q.host().unwrap().to_vec()
}

fn run_vbd_gpu(
    model: &mut Model,
    dt: Real,
    substeps: usize,
    iters: usize,
    contact_k: f64,
) -> Vec<Vec3> {
    let engine = Engine::new_gpu().expect("GPU adapter for VBD test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();
    let mut solver = VbdSolver::new_gpu(&engine, model)
        .unwrap()
        .with_iterations(iters)
        .with_contact_stiffness(contact_k);
    for _ in 0..substeps {
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.particle_q.download(&device, &queue).unwrap();
    s_a.particle_q.host().unwrap().to_vec()
}

/// VBD low-stiffness drape should be in the same neighbourhood as
/// XPBD. Both solvers minimise the same elastic + gravitational
/// potential, so for soft cloth (~10 N/m equivalent stiffness) the
/// steady-state shape converges to within centimetres.
#[test]
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test -p molla-solvers --features gpu --test vbd_cloth_gpu -- --ignored"]
fn vbd_matches_xpbd_low_stiffness() {
    let n = 8;
    let size = 1.0;
    let top_y = 1.5;
    let compliance = 1.0e-2; // soft cloth, k = 100 N/m
    let max_stretch = 0.5;
    let dt: Real = 1.0 / 240.0;
    let substeps = 240; // 1 s of sim
    let iters = 8;

    let mut model_xpbd = build_cloth(n, size, top_y, compliance, max_stretch, None);
    let mut model_vbd = build_cloth(n, size, top_y, compliance, max_stretch, None);

    let xpbd_pos = run_xpbd_gpu(&mut model_xpbd, dt, substeps, iters);
    let vbd_pos = run_vbd_gpu(&mut model_vbd, dt, substeps, iters, 1.0e4);

    // Mean position residual — robust to per-particle wiggle.
    let mean_xpbd = xpbd_pos.iter().fold(Vec3::ZERO, |a, b| a + *b) / xpbd_pos.len() as f64;
    let mean_vbd = vbd_pos.iter().fold(Vec3::ZERO, |a, b| a + *b) / vbd_pos.len() as f64;
    let com_diff = (mean_xpbd - mean_vbd).length();
    eprintln!("XPBD mean: {mean_xpbd:?}  VBD mean: {mean_vbd:?}  Δ = {com_diff:.4}");
    assert!(
        com_diff < 0.30,
        "VBD low-stiffness drape COM differs from XPBD by {com_diff:.3} m (>30 cm)",
    );
}

/// VBD must remain stable at high stiffness where XPBD struggles.
/// At compliance 1e-7 (k≈1e7 N/m) cloth is "almost inextensible"
/// — XPBD typically explodes or oscillates without strain limiting.
/// VBD should converge to a tight drape with no NaNs.
#[test]
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test -p molla-solvers --features gpu --test vbd_cloth_gpu -- --ignored"]
fn vbd_high_stiffness_does_not_explode() {
    let n = 8;
    let size = 1.0;
    let top_y = 1.5;
    let compliance = 1.0e-7;
    let max_stretch = 2.0; // disable XPBD strain cap; rely on VBD's stiffness
    let dt: Real = 1.0 / 480.0;
    let substeps = 60;
    let iters = 12;

    let mut model = build_cloth(n, size, top_y, compliance, max_stretch, None);
    let pos = run_vbd_gpu(&mut model, dt, substeps, iters, 1.0e4);

    // No NaN / Inf escape.
    for (i, p) in pos.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "particle {i} non-finite at {p:?}"
        );
    }
    // Cloth shouldn't have flown to the moon — bound it to a 50m
    // box centred on origin (gravity in 60 substeps × 1/480 s ≈ 1.2
    // s of free-fall would only travel ~7m).
    let max_y = pos.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
    let min_y = pos.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    eprintln!("VBD high-stiffness y range: [{min_y:.3}, {max_y:.3}]");
    assert!(
        max_y < 50.0 && min_y > -50.0,
        "cloth escaped: y in [{min_y}, {max_y}]",
    );
    // At least some particles should still be roughly in the cloth
    // plane — a fully-imploded sheet would collapse to a point.
    let bbox = pos.iter().fold(
        (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY)),
        |(lo, hi), p| {
            (
                Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z)),
                Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z)),
            )
        },
    );
    let extent = bbox.1 - bbox.0;
    eprintln!("cloth extent after {substeps} steps: {extent:?}");
    assert!(
        extent.x > 0.5 && extent.z > 0.5,
        "cloth collapsed unexpectedly to extent {extent:?}",
    );
}

/// AVBD (soft) coupling: a cloth sheet released above a kinematic
/// rigid sphere should drape around the top half of the sphere.
/// Asserts the contact penalty actually keeps particles outside
/// the sphere and brings the drape into contact.
#[test]
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test -p molla-solvers --features gpu --test vbd_cloth_gpu -- --ignored"]
fn vbd_wraps_kinematic_sphere() {
    let n = 16;
    let size = 1.6;
    let top_y = 2.5;
    let sphere_centre = Vec3::new(0.0, 1.0, 0.0);
    let sphere_radius = 0.5;
    let particle_radius = 0.02;
    let compliance = 5.0e-6;
    let max_stretch = 0.10;
    let dt: Real = 1.0 / 480.0;
    let substeps = 480; // 1 s — enough to fall + settle
    let iters = 10;

    let mut model = build_cloth(
        n,
        size,
        top_y,
        compliance,
        max_stretch,
        Some((sphere_centre, sphere_radius)),
    );
    let pos = run_vbd_gpu(&mut model, dt, substeps, iters, 5.0e4);

    // (a) No particle inside the sphere (margin = particle radius).
    for (i, p) in pos.iter().enumerate() {
        let d = (*p - sphere_centre).length();
        assert!(
            d >= sphere_radius - particle_radius * 2.0,
            "particle {i} at {p:?} (d = {d:.3}) penetrated sphere of radius {sphere_radius}",
        );
    }

    // (b) Cloth COM dropped well below initial top_y.
    let mean_y = pos.iter().map(|p| p.y).sum::<f64>() / pos.len() as f64;
    eprintln!("VBD-on-sphere mean y after {substeps} steps: {mean_y:.3} (started at {top_y})");
    assert!(
        mean_y < top_y - 0.5,
        "cloth COM only dropped to {mean_y:.3}, expected < {:.3}",
        top_y - 0.5,
    );

    // (c) At least 5% of particles are touching (within 3 × radius)
    //     the sphere surface.
    let touch_band = sphere_radius + 3.0 * particle_radius;
    let touching = pos
        .iter()
        .filter(|p| (**p - sphere_centre).length() <= touch_band)
        .count();
    let pct = (touching as f64) / (pos.len() as f64);
    eprintln!(
        "touching particles: {touching} / {} ({:.1}%)",
        pos.len(),
        pct * 100.0
    );
    assert!(
        touching >= pos.len() / 20,
        "only {touching} particles in contact band; cloth never reached the sphere",
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-vbd-cloth-gpu"]
fn vbd_tracks_moving_kinematic_sphere_gpu() {
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
    let mut model = b.build().unwrap();

    let engine = Engine::new_gpu().expect("GPU adapter for moving VBD contact");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let contacts = Contacts::with_capacity(0);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    let mut solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
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
        s_a.body_q.upload(&queue).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        s_b.particle_q.download(&device, &queue).unwrap();
        let q = s_b.particle_q.host().unwrap();
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
        std::mem::swap(&mut s_a, &mut s_b);
    }

    s_a.particle_q.download(&device, &queue).unwrap();
    s_a.particle_qd.download(&device, &queue).unwrap();
    let q = s_a.particle_q.host().unwrap();
    let qd = s_a.particle_qd.host().unwrap();
    for (i, (p, v)) in q.iter().zip(qd.iter()).enumerate() {
        assert!(
            p.is_finite() && v.is_finite(),
            "GPU moving-sphere cloth particle {i} became non-finite: p={p:?}, v={v:?}"
        );
    }
    eprintln!(
        "GPU VBD moving sphere cloth: max_penetration={max_penetration:.5}, max_mean_y={max_mean_y:.5}, max_touching={max_touching}"
    );
    assert!(
        max_penetration <= 4.0 * particle_radius,
        "GPU moving kinematic sphere penetrated cloth too far: max penetration={max_penetration}"
    );
    assert!(
        max_mean_y > cloth_y + 0.025,
        "GPU moving kinematic sphere did not lift the cloth: max_mean_y={max_mean_y}, cloth_y={cloth_y}"
    );
    assert!(
        max_touching >= q.len() / 20,
        "GPU cloth never contacted moving kinematic sphere: max_touching={max_touching}, total={}",
        q.len()
    );
}

/// INSPECTION F7 — VBD GPU internal multi-substep. Previously the GPU
/// backend returned `Error::Solver` for `substeps > 1`; now it
/// ping-pongs `gpu_substep` through a device-resident scratch state
/// (same pattern as the XPBD B1 fix). The CPU internal-substep path
/// (which loops `dt/n` host-side) is the exact semantic twin and the
/// reference.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=vbd_cloth_gpu`"]
fn vbd_gpu_multi_substep_matches_cpu() {
    let n = 8;
    let size = 1.0;
    let top_y = 1.5;
    let compliance = 1.0e-2;
    let max_stretch = 0.5;
    let dt: Real = 1.0 / 60.0;
    let substeps = 4;
    let outer_steps = 60; // 1 s of sim at 60 Hz outer / 240 Hz substep
    let iters = 8;

    // CPU internal-substep reference.
    let model_cpu = build_cloth(n, size, top_y, compliance, max_stretch, None);
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    let control = model_cpu.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver_cpu = VbdSolver::new_cpu()
        .with_iterations(iters)
        .with_substeps(substeps);
    for _ in 0..outer_steps {
        solver_cpu
            .step(&model_cpu, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_q = s_a.particle_q.host().unwrap().to_vec();

    // GPU under test — internal substeps on-device.
    let mut model_gpu = build_cloth(n, size, top_y, compliance, max_stretch, None);
    let engine = Engine::new_gpu().expect("GPU adapter for VBD test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut g_a = model_gpu.state().unwrap();
    let mut g_b = model_gpu.state().unwrap();
    let mut control_gpu = model_gpu.control();
    let mut contacts_gpu = Contacts::with_capacity(0);
    g_a.upload_to_device(&device, &queue).unwrap();
    g_b.upload_to_device(&device, &queue).unwrap();
    control_gpu.upload_to_device(&device, &queue).unwrap();
    contacts_gpu.upload_to_device(&device, &queue).unwrap();
    let mut solver_gpu = VbdSolver::new_gpu(&engine, &model_gpu)
        .unwrap()
        .with_iterations(iters)
        .with_substeps(substeps);
    for _ in 0..outer_steps {
        solver_gpu
            .step(&model_gpu, &g_a, &control_gpu, &contacts_gpu, dt, &mut g_b)
            .unwrap();
        std::mem::swap(&mut g_a, &mut g_b);
    }
    g_a.particle_q.download(&device, &queue).unwrap();
    let gpu_q = g_a.particle_q.host().unwrap().to_vec();

    // Non-vacuous: the cloth must actually have fallen from its spawn.
    let cpu_mean_y: Real = cpu_q.iter().map(|p| p.y).sum::<Real>() / cpu_q.len() as Real;
    assert!(
        cpu_mean_y < top_y - 0.3,
        "cloth did not drape: mean y = {cpu_mean_y}"
    );

    let mut max_dp: Real = 0.0;
    for (c, g) in cpu_q.iter().zip(gpu_q.iter()) {
        max_dp = max_dp.max((*c - *g).length());
    }
    eprintln!(
        "vbd multi-substep: {} particles, mean y = {cpu_mean_y:.3}, max |Δp| = {max_dp:.4}",
        cpu_q.len()
    );
    // f64 CPU vs f32 GPU over 240 substeps of a draping cloth — the
    // existing VBD cloth gates use centimetre-scale tolerances; the
    // failure this guards against (substeps silently rejected or the
    // scratch ping-pong dropping a substep's state) diverges by
    // decimetres.
    assert!(
        max_dp < 0.05,
        "GPU internal multi-substep diverged from CPU: max |Δp| = {max_dp}"
    );
}
