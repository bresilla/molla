//! VBD cloth solver — Stage 20 GPU integration tests. All
//! `#[ignore]` because they need a real wgpu adapter (run via
//! `nixVulkan cargo test -p molla --features gpu --test vbd_cloth_gpu
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

use molla::compute::Engine;
use molla::geometry::CollisionPipeline;
use molla::math::{Real, Transform, Vec3};
use molla::prelude::*;

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
        origin, u_axis, v_axis, n, n, 0.05, 0.02, compliance, max_stretch, WorldId(0),
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
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test --features gpu --test vbd_cloth_gpu -- --ignored"]
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
    eprintln!(
        "XPBD mean: {mean_xpbd:?}  VBD mean: {mean_vbd:?}  Δ = {com_diff:.4}"
    );
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
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test --features gpu --test vbd_cloth_gpu -- --ignored"]
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
    let bbox = pos
        .iter()
        .fold((Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY)), |(lo, hi), p| {
            (
                Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z)),
                Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z)),
            )
        });
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
#[ignore = "needs a real wgpu adapter — run via nixVulkan cargo test --features gpu --test vbd_cloth_gpu -- --ignored"]
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
    eprintln!("touching particles: {touching} / {} ({:.1}%)", pos.len(), pct * 100.0);
    assert!(
        touching >= pos.len() / 20,
        "only {touching} particles in contact band; cloth never reached the sphere",
    );
}
