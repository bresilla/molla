//! GPU equality-constraint projection — golden / parity tests for
//! the WGSL `XpbdEqualityKernel` that replaced the host-side
//! `apply_equality_post_gpu` shim. `#[ignore]` because they need a
//! real wgpu adapter; run with `make test-equality-gpu`.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::geometry::CollisionPipeline;
use molla::math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use molla::prelude::*;

/// Mirror of `xpbd_equality.rs::connect_pins_two_bodies_under_gravity`
/// but driven by the **GPU** solver. Two bodies (one kinematic
/// anchor, one free) joined by a CONNECT — after settling, the free
/// body should hang directly under the anchor. The GPU equality
/// kernel is what makes this work end-to-end on device.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-equality-gpu"]
fn gpu_connect_pins_two_bodies_under_gravity() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);

    let anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(Vec3::new(0.0, 5.0, 0.0), Quat::IDENTITY))
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let free = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
            .with_position(Vec3::new(1.0, 5.0, 0.0)),
    );
    b.add_equality_connect(anchor, free, Vec3::ZERO, Vec3::ZERO);
    let mut model = b.build().unwrap();

    let engine = Engine::new_gpu().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    // capacity > 0 so the contact buffers get device mirrors even
    // when the scene generates no contacts — the XPBD kernel binds
    // them unconditionally. (Match the existing xpbd_gpu_parity
    // capacity to stay safely above any Vulkan driver minimum
    // storage-buffer-size oddities.)
    let mut contacts = Contacts::with_capacity(8192);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
        .unwrap()
        .with_iterations(8);
    let mut pipeline = CollisionPipeline::new();

    let dt: Real = 0.005;
    for _ in 0..1000 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    s_a.body_q.download(&device, &queue).unwrap();
    let q = s_a.body_q.host().unwrap();
    let dist = (q[1].position - q[0].position).length();
    eprintln!(
        "GPU connect: anchor at {:?}, free at {:?}, dist {dist:.4}",
        q[0].position, q[1].position
    );
    assert!(
        dist < 0.05,
        "GPU CONNECT failed: bodies {dist:.3} m apart"
    );
}

/// Mirror of `weld_keeps_orientation_aligned`, GPU-driven.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-equality-gpu"]
fn gpu_weld_keeps_orientation_aligned() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    let anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::IDENTITY)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let free = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            })
            .with_pose(Transform::new(
                Vec3::new(0.5, 0.0, 0.0),
                Quat::from_rotation_z(0.5),
            )),
    );
    b.add_equality_weld(anchor, free, Vec3::ZERO, Vec3::ZERO, Quat::IDENTITY);
    let mut model = b.build().unwrap();
    // No gravity so the test stays deterministic.
    model.gravity.host_mut().unwrap()[0] = Vec3::ZERO;

    let engine = Engine::new_gpu().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    // capacity > 0 so the contact buffers get device mirrors even
    // when the scene generates no contacts — the XPBD kernel binds
    // them unconditionally. (Match the existing xpbd_gpu_parity
    // capacity to stay safely above any Vulkan driver minimum
    // storage-buffer-size oddities.)
    let mut contacts = Contacts::with_capacity(8192);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
        .unwrap()
        .with_iterations(20);

    // Run a single no-op collide pass so body_contact_offsets gets
    // sized for body_count and gains a GPU device buffer. Without
    // this the XPBD kernel's contact bind group fails to bind.
    let mut pipeline = CollisionPipeline::new();
    pipeline.collide(&model, &s_a, &mut contacts).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let dt: Real = 0.005;
    for _ in 0..200 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    s_a.body_q.download(&device, &queue).unwrap();
    let q = s_a.body_q.host().unwrap();
    let dist = (q[1].position - q[0].position).length();
    let dot = q[1].rotation.x * q[0].rotation.x
        + q[1].rotation.y * q[0].rotation.y
        + q[1].rotation.z * q[0].rotation.z
        + q[1].rotation.w * q[0].rotation.w;
    eprintln!("GPU weld: dist={dist:.4} rot-dot={dot:.4}");
    assert!(dist < 0.05, "GPU WELD position drifted: dist={dist}");
    assert!(
        dot.abs() > 0.95,
        "GPU WELD orientation drifted: dot={dot}"
    );
}

/// Velocity recovery sanity check: a free body pinned to a
/// stationary kinematic anchor must end the step with near-zero
/// linear velocity (the constraint snapped the position back, the
/// recovery pass must reflect that as ≈ zero net displacement / dt).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-equality-gpu"]
fn gpu_equality_velocity_recovery_zeros_drift() {
    let mut b = ModelBuilder::new();
    let anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::IDENTITY)
            .with_flags(BodyFlags::DYNAMIC | BodyFlags::KINEMATIC),
    );
    let free = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
            .with_position(Vec3::new(0.001, 0.0, 0.0)), // tiny offset
    );
    b.add_equality_connect(anchor, free, Vec3::ZERO, Vec3::ZERO);
    let mut model = b.build().unwrap();
    model.gravity.host_mut().unwrap()[0] = Vec3::ZERO; // no gravity → equilibrium

    let engine = Engine::new_gpu().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    // capacity > 0 so the contact buffers get device mirrors even
    // when the scene generates no contacts — the XPBD kernel binds
    // them unconditionally. (Match the existing xpbd_gpu_parity
    // capacity to stay safely above any Vulkan driver minimum
    // storage-buffer-size oddities.)
    let mut contacts = Contacts::with_capacity(8192);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
        .unwrap()
        .with_iterations(20);

    // No-op collide to size body_contact_offsets — see comment in
    // the WELD test above.
    let mut pipeline = CollisionPipeline::new();
    pipeline.collide(&model, &s_a, &mut contacts).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    // Several steps so the constraint converges; velocity should
    // settle back to zero.
    for _ in 0..50 {
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    s_a.body_qd.download(&device, &queue).unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let v = qd[1].linear.length();
    eprintln!("GPU equality v_post: {v:.4}");
    assert!(v < 0.5, "expected ≈ 0 velocity at equilibrium, got {v}");
}

// ──────────────────────────────────────────────────────────────────
// Equality + contact-friction interaction regression (Stage X bug).
//
// The GPU equality kernel used to OVERWRITE `body_qd` after the main
// XPBD kernel had written the post-contact / post-friction /
// post-restitution velocity. Symptom: any body that had both an
// equality and a contact would see its contact velocity destroyed —
// rope spheres tunnelled through the floor, FourBar bars lost
// friction, etc. The fix snapshots `body_q` before the equality
// projection and ADDS the equality-only delta to `body_qd` (instead
// of overwriting it).
//
// The tests below are mirrored on the CPU `cpu_step` path which runs
// equality projection BEFORE the velocity recovery — so they pass
// trivially on CPU, then exercise the same expectation on GPU.
// ──────────────────────────────────────────────────────────────────

const REGRESSION_ROPE_LINKS: usize = 18;
const REGRESSION_ROPE_LINK_RADIUS: f64 = 0.12;
const REGRESSION_ROPE_LINK_SPACING: f64 = 0.26;
const REGRESSION_ROPE_ANCHOR_HEIGHT: f64 = 6.0;

fn build_rope_for_regression() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.10);
    let mut bodies = Vec::with_capacity(REGRESSION_ROPE_LINKS);
    for i in 0..REGRESSION_ROPE_LINKS {
        let y = REGRESSION_ROPE_ANCHOR_HEIGHT - i as f64 * REGRESSION_ROPE_LINK_SPACING;
        let x = 0.001 * i as f64;
        let mut params = BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(0.4, REGRESSION_ROPE_LINK_RADIUS))
            .with_position(Vec3::new(x, y, 0.0));
        if i == 0 {
            params = params
                .with_mass(0.0)
                .with_inertia(Mat3::ZERO)
                .with_flags(BodyFlags::KINEMATIC);
        }
        let id = b.add_body(params);
        b.add_shape_sphere(id, REGRESSION_ROPE_LINK_RADIUS, Transform::IDENTITY)
            .expect("rope sphere");
        bodies.push(id);
    }
    let half = REGRESSION_ROPE_LINK_SPACING * 0.5;
    for i in 1..REGRESSION_ROPE_LINKS {
        let _ = b.add_equality_connect(
            bodies[i - 1],
            bodies[i],
            Vec3::new(0.0, -half, 0.0),
            Vec3::new(0.0, half, 0.0),
        );
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .expect("plane");
    b.build().expect("rope model")
}

/// Rope (Scene #2) GPU regression: the GPU equality kernel previously
/// destroyed the floor-contact velocity (restitution + friction) on
/// every step, so the bottom of the rope tunnelled through the floor
/// plane. With the fix, every sphere centre stays above
/// `radius − slop` after the rope settles.
#[test]
#[ignore = "needs a real wgpu adapter — exercises the equality + contact velocity preservation fix"]
fn gpu_rope_does_not_tunnel_floor() {
    let mut model = build_rope_for_regression();
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let mut contacts = Contacts::with_capacity(8192);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
        .unwrap()
        .with_iterations(10);
    let mut pipeline = CollisionPipeline::new();
    let dt: Real = 1.0 / 240.0;
    // 4 simulated seconds — long enough for the rope to fall and
    // settle, but short enough to keep CI runtime sane.
    for _ in 0..960 {
        // CollisionPipeline runs on the host, so pull the latest GPU
        // body_q back before each broad/narrow pass; same pattern as
        // `xpbd_gpu_parity::box_stack_cpu_gpu_match`.
        s_a.download_from_device(&device, &queue).unwrap();
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        s_b.download_from_device(&device, &queue).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.body_q.host().unwrap();
    let mut min_y = f64::INFINITY;
    let mut min_idx = 0;
    for (i, p) in q.iter().enumerate() {
        if p.position.y < min_y {
            min_y = p.position.y;
            min_idx = i;
        }
    }
    // Same tolerance as the CPU rope test: ~5 mm of XPBD residual
    // penetration is accepted; tunnelling under the floor by an
    // unbounded amount (the bug) is not.
    let slop = 8e-3;
    assert!(
        min_y >= REGRESSION_ROPE_LINK_RADIUS - slop,
        "GPU rope sphere {min_idx} tunnels floor: y={min_y}, expected >= {} (this is the equality-overwrites-velocity bug)",
        REGRESSION_ROPE_LINK_RADIUS - slop,
    );
}

/// CPU vs GPU smoke test for the equality + restitution pipeline.
/// CPU is the oracle (it runs equality BEFORE velocity recovery, so
/// it never had this bug). After the fix, GPU body_qd's vertical
/// component matches CPU within a bounded slop on a bouncing-stack
/// scene. The fix keeps the project_velocity (restitution +
/// friction) contribution in body_qd; pre-fix it was overwritten by
/// `(q_post_eq − q_pre_step) / dt`, which silently dropped the
/// restitution kick.
///
/// **Source of the residual ~0.045 m/s divergence (Option C — accept
/// + document):** CPU and GPU run the substep stages in different
/// orders by design.
///
/// - CPU `cpu_step`: predict → position-projection → **equality** →
///   velocity-recovery → velocity-projection (restitution + friction).
///   Restitution sees `r_a / r_b` lever arms at the post-equality
///   pose, and `v_pre = v_pred + ω × r_post_equality`.
/// - GPU XpbdKernel + XpbdEqualityKernel: predict → position-projection
///   → velocity-recovery → velocity-projection → **equality** →
///   recover_velocity_after_equality (additive). Restitution sees
///   lever arms at the post-position-projection-but-PRE-equality pose.
///
/// For this bouncing-stack scene, the equality projection pulls the
/// top box ~mm downward each step. CPU's restitution then computes
/// `dv_n = -e·v_n_pre - v_n` against the post-equality lever arms;
/// GPU computes against the pre-equality ones. The lever-arm and
/// `v_n` differences accumulate to a deterministic ~0.045 m/s drift
/// after 30 substeps (same per-body, both are coupled by the
/// CONNECT). f32-narrowing alone would only contribute ~6e-4 m/s
/// for 60 substeps — not the dominant term.
///
/// Restoring stage-by-stage parity would require either re-ordering
/// the GPU pipeline (equality before velocity-projection) — which
/// the current order was chosen to AVOID, because the previous
/// "equality then overwrite qd" path destroyed restitution velocity
/// — or running a second equality projection before velocity-
/// projection on GPU. Both are larger surgeries than the parity gain
/// is worth; the rope / two-bar tests below catch the actually-
/// broken regression mode (qd overwrite). This test holds the
/// honest bound: dvy ≤ 0.06 m/s, which is 33% headroom over the
/// measured 0.045 — tight enough to catch a re-introduction of the
/// overwrite bug (which would push dvy past 0.5+ m/s) but loose
/// enough to absorb the documented structural difference.
#[test]
#[ignore = "needs a real wgpu adapter — CPU vs GPU parity for equality + restitution velocity preservation"]
fn gpu_friction_velocity_survives_equality_recovery() {
    fn build_model() -> Model {
        let mut b = ModelBuilder::new();
        b.set_angular_damping(0.0);
        b.set_linear_damping(0.0);
        b.set_default_friction(0.0).unwrap();
        b.set_default_restitution(1.0).unwrap();
        let half = Vec3::splat(0.15);
        let inertia = Mat3::from_diagonal(Vec3::new(0.03, 0.03, 0.03));
        let bottom = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia { mass: 1.0, com: Vec3::ZERO, inertia })
                .with_position(Vec3::new(0.0, 0.15 + 0.05, 0.0)),
        );
        b.add_shape_box(bottom, half, Transform::IDENTITY).unwrap();
        let top = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia { mass: 1.0, com: Vec3::ZERO, inertia })
                .with_position(Vec3::new(0.0, 0.15 * 3.0 + 0.05, 0.0)),
        );
        b.add_shape_box(top, half, Transform::IDENTITY).unwrap();
        b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
            .expect("plane");
        b.add_equality_connect(
            bottom,
            top,
            Vec3::new(0.0, 0.15, 0.0),
            Vec3::new(0.0, -0.15, 0.0),
        );
        b.build().expect("bouncing-stack model")
    }

    let dt: Real = 1.0 / 240.0;
    let n_steps = 30;

    // CPU oracle.
    let cpu_model = build_model();
    let mut cpu_a = cpu_model.state().unwrap();
    let mut cpu_b = cpu_model.state().unwrap();
    let cpu_control = cpu_model.control();
    let mut cpu_contacts = Contacts::with_capacity(256);
    let mut cpu_pipeline = CollisionPipeline::new();
    let mut cpu_solver = XpbdSolver::new().with_iterations(8);
    for _ in 0..n_steps {
        cpu_pipeline.collide(&cpu_model, &cpu_a, &mut cpu_contacts).unwrap();
        cpu_solver
            .step(&cpu_model, &cpu_a, &cpu_control, &cpu_contacts, dt, &mut cpu_b)
            .unwrap();
        std::mem::swap(&mut cpu_a, &mut cpu_b);
    }
    let cpu_qd = cpu_a.body_qd.host().unwrap().to_vec();

    // GPU.
    let mut gpu_model = build_model();
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    gpu_model.upload_to_device(&device, &queue).unwrap();
    let mut g_a = gpu_model.state().unwrap();
    let mut g_b = gpu_model.state().unwrap();
    let mut g_control = gpu_model.control();
    let mut g_contacts = Contacts::with_capacity(256);
    g_a.upload_to_device(&device, &queue).unwrap();
    g_b.upload_to_device(&device, &queue).unwrap();
    g_control.upload_to_device(&device, &queue).unwrap();
    g_contacts.upload_to_device(&device, &queue).unwrap();
    let mut g_solver = XpbdSolver::new_gpu(&engine, gpu_model.body_count as u32)
        .unwrap()
        .with_iterations(8);
    let mut g_pipeline = CollisionPipeline::new();
    for _ in 0..n_steps {
        g_a.download_from_device(&device, &queue).unwrap();
        g_pipeline.collide(&gpu_model, &g_a, &mut g_contacts).unwrap();
        g_contacts.upload_to_device(&device, &queue).unwrap();
        g_solver
            .step(&gpu_model, &g_a, &g_control, &g_contacts, dt, &mut g_b)
            .unwrap();
        g_b.download_from_device(&device, &queue).unwrap();
        std::mem::swap(&mut g_a, &mut g_b);
    }
    let gpu_qd = g_a.body_qd.host().unwrap().to_vec();

    // Vertical velocity parity. With the fix, GPU vy ≈ CPU vy (same
    // bounce direction + magnitude). Pre-fix the GPU vy was the
    // pre-restitution value (negative or near-zero) while CPU vy was
    // the post-restitution (positive bounce-back).
    for i in 0..2 {
        let dvy = (cpu_qd[i].linear.y - gpu_qd[i].linear.y).abs();
        eprintln!(
            "body {i}: cpu vy={:.5} gpu vy={:.5} dvy={dvy:.5}",
            cpu_qd[i].linear.y, gpu_qd[i].linear.y,
        );
        // Honest bound: ~0.045 measured + 33% headroom. See test-
        // level doc for the structural CPU/GPU pipeline-order
        // divergence that produces the residual. The hard tunnelling
        // failure mode (qd overwrite) is captured by the rope /
        // two-bar regression tests below and would push dvy past
        // 0.5 m/s.
        assert!(
            dvy < 0.06,
            "body {i} vertical velocity diverges between CPU and GPU by {dvy} m/s — equality recovery is likely destroying the project_velocity (restitution) contribution",
        );
    }
}

/// FourBar-like sanity (Scene #39): two dynamic bars connected end-
/// to-end by a CONNECT, dropped onto the floor. The closure should
/// survive the contact phase — i.e. the bars stay pinned together
/// AND don't tunnel through the floor.
#[test]
#[ignore = "needs a real wgpu adapter — verifies CONNECT closure survives contact friction"]
fn gpu_two_bar_chain_survives_floor_contact() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.10);
    b.set_default_friction(0.5).unwrap();

    let radius = 0.1;
    let inertia = Mat3::IDENTITY;
    let body_a = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass: 1.0, com: Vec3::ZERO, inertia })
            .with_position(Vec3::new(-0.5, 1.5, 0.0)),
    );
    b.add_shape_sphere(body_a, radius, Transform::IDENTITY).unwrap();
    let body_b = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass: 1.0, com: Vec3::ZERO, inertia })
            .with_position(Vec3::new(0.5, 1.5, 0.0)),
    );
    b.add_shape_sphere(body_b, radius, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY).unwrap();
    // CONNECT pinning the spheres at the midpoint between them
    // (anchor offsets along ±X half-distance).
    b.add_equality_connect(
        body_a,
        body_b,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
    );

    let mut model = b.build().unwrap();
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let mut contacts = Contacts::with_capacity(2048);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    control.upload_to_device(&device, &queue).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count as u32)
        .unwrap()
        .with_iterations(10);
    let mut pipeline = CollisionPipeline::new();
    let dt: Real = 1.0 / 240.0;
    for _ in 0..720 {
        s_a.download_from_device(&device, &queue).unwrap();
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        s_b.download_from_device(&device, &queue).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.body_q.host().unwrap();
    let dist = (q[1].position - q[0].position).length();
    let min_y = q[0].position.y.min(q[1].position.y);
    eprintln!(
        "GPU two-bar: a={:?}, b={:?}, dist={dist:.4}, min_y={min_y:.4}",
        q[0].position, q[1].position
    );
    // Closure should stay within ~5 cm of the rest distance (1.0 m).
    assert!(
        (dist - 1.0).abs() < 0.05,
        "CONNECT closure broke: dist={dist}, expected ≈ 1.0",
    );
    // Neither sphere tunnels the floor.
    assert!(
        min_y >= radius - 8e-3,
        "two-bar chain tunnels floor: min_y={min_y}, expected >= {}",
        radius - 8e-3,
    );
    assert!(q[0].position.is_finite() && q[1].position.is_finite());
}
