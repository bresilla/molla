//! GPU XPBD parity: a stack of boxes settles on the GPU within
//! f32-narrowing tolerance of the CPU result.

#![cfg(feature = "gpu")]

use molla_compute::{Backend, Engine};
use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

fn box_inertia(mass: Real, half: Vec3) -> Mat3 {
    let hx2 = half.x * half.x;
    let hy2 = half.y * half.y;
    let hz2 = half.z * half.z;
    let s = mass / 3.0;
    Mat3::from_diagonal(Vec3::new(s * (hy2 + hz2), s * (hx2 + hz2), s * (hx2 + hy2)))
}

fn build_stack(count: usize) -> Model {
    let half = Vec3::splat(0.5);
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    for i in 0..count {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 1.0,
                    com: Vec3::ZERO,
                    inertia: box_inertia(1.0, half),
                })
                .with_position(Vec3::new(0.0, 0.5 + i as Real * 1.05, 0.0)),
        );
        b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let mut m = b.build().unwrap();
    m.default_friction = 0.8;
    m.default_restitution = 0.0;
    m.contact_margin = 0.02;
    m
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=xpbd_boxes`"]
fn box_stack_cpu_gpu_match() {
    let count = 4;
    let model_cpu = build_stack(count);
    let mut model_gpu = build_stack(count);

    let engine = Engine::new(Backend::AutoGpu).expect("init wgpu");
    let device = engine.device().expect("device");
    let queue = engine.queue().expect("queue");

    model_gpu.upload_to_device(device, queue).unwrap();

    // Run CPU.
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    let control = model_cpu.control();
    let mut contacts_cpu = Contacts::with_capacity(512);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8);
    for _ in 0..400 {
        pipeline
            .collide(&model_cpu, &s_a, &mut contacts_cpu)
            .unwrap();
        solver
            .step(&model_cpu, &s_a, &control, &contacts_cpu, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_q = s_a.body_q.host().unwrap().to_vec();

    // Run GPU.
    let mut g_a = model_gpu.state().unwrap();
    let mut g_b = model_gpu.state().unwrap();
    g_a.upload_to_device(device, queue).unwrap();
    g_b.upload_to_device(device, queue).unwrap();
    let mut control_gpu = model_gpu.control();
    let _ = control_gpu.body_external_f.ensure_device(device);
    let mut contacts_gpu = Contacts::with_capacity(512);
    let mut pipeline_gpu = CollisionPipeline::new();
    let mut solver_gpu = XpbdSolver::new_gpu(&engine, count as u32).expect("new_gpu");
    solver_gpu.position_iterations = 8;
    for _ in 0..400 {
        // Run collision on host side then upload.
        g_a.download_from_device(device, queue).unwrap();
        pipeline_gpu
            .collide(&model_gpu, &g_a, &mut contacts_gpu)
            .unwrap();
        contacts_gpu.upload_to_device(device, queue).unwrap();
        solver_gpu
            .step(
                &model_gpu,
                &g_a,
                &control_gpu,
                &contacts_gpu,
                0.005,
                &mut g_b,
            )
            .unwrap();
        g_b.download_from_device(device, queue).unwrap();
        std::mem::swap(&mut g_a, &mut g_b);
    }
    let gpu_q = g_a.body_q.host().unwrap().to_vec();

    for i in 0..count {
        let dy = (cpu_q[i].position.y - gpu_q[i].position.y).abs();
        eprintln!(
            "body {i}: cpu_y={:.4} gpu_y={:.4} dy={dy:.4}",
            cpu_q[i].position.y, gpu_q[i].position.y
        );
        assert!(
            dy < 0.05,
            "GPU box {i} differs from CPU by {dy} m (cpu={}, gpu={})",
            cpu_q[i].position.y,
            gpu_q[i].position.y
        );
    }
}

/// INSPECTION B1 — GPU multi-substep with contact-time pose anchors.
/// The kernel binds a device-captured contact pose (`body_q_contact`)
/// and ping-pongs substeps through a device-resident scratch state;
/// the CPU internal-substep path (which threads `q_at_contact` on the
/// host) is the exact semantic twin and the reference.
///
/// Scenario: a box in **continuous contact** — resting on the plane
/// with an initial horizontal velocity, decelerated by Coulomb
/// friction. This keeps every substep in the frozen-anchor code path
/// (position + velocity projection, friction) while the pose actually
/// moves, without entering the impact regime: internal substepping
/// amplifies the position-correction pop velocity on impacts
/// (`Δp / (dt/n)`), a pre-existing limitation shared by BOTH backends
/// (measured: a restitution-0 box drop bounces persistently at
/// substeps = 4 on CPU and GPU alike — see INSPECTION.md B1b), so
/// impact scenarios are not parity-testable here.
///
/// This test also pinned down a second pre-existing GPU bug while
/// being written: the WGSL friction cap lacked the CPU's
/// support-impulse floor (`μ·max(|λ_n|, m·g·dt)`), so sliding friction
/// collapsed in steady contact and the box glided 2.7× too far. The
/// floor now exists in `project_velocity` (xpbd.wgsl).
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=xpbd_gpu_parity`"]
fn sliding_box_multi_substep_cpu_gpu_match() {
    let substeps = 4;
    let n_steps = 200;
    let dt: Real = 0.005;
    // Single box starting at rest height with a 2 m/s push along +X.
    let build = || {
        let half = Vec3::splat(0.5);
        let mut b = ModelBuilder::new();
        b.set_angular_damping(0.05);
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 1.0,
                    com: Vec3::ZERO,
                    inertia: box_inertia(1.0, half),
                })
                .with_position(Vec3::new(0.0, 0.5, 0.0)),
        );
        b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
        b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
            .unwrap();
        let mut m = b.build().unwrap();
        m.default_friction = 0.4;
        m.default_restitution = 0.0;
        m.contact_margin = 0.02;
        m
    };
    let push = |state: &mut molla_sim::State| {
        let qd = state.body_qd.host_mut().unwrap();
        qd[0].linear = Vec3::new(2.0, 0.0, 0.0);
    };

    let model_cpu = build();
    let mut model_gpu = build();

    let engine = Engine::new(Backend::AutoGpu).expect("init wgpu");
    let device = engine.device().expect("device");
    let queue = engine.queue().expect("queue");
    model_gpu.upload_to_device(device, queue).unwrap();

    // CPU internal-substep reference.
    let mut s_a = model_cpu.state().unwrap();
    push(&mut s_a);
    let mut s_b = model_cpu.state().unwrap();
    let control = model_cpu.control();
    let mut contacts_cpu = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8).with_substeps(substeps);
    let mut cpu_traj = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        pipeline
            .collide(&model_cpu, &s_a, &mut contacts_cpu)
            .unwrap();
        solver
            .step(&model_cpu, &s_a, &control, &contacts_cpu, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.body_q.host().unwrap()[0];
        cpu_traj.push((q.position.x, q.position.y));
    }

    // GPU under test — same internal substep semantics.
    let mut g_a = model_gpu.state().unwrap();
    push(&mut g_a);
    let mut g_b = model_gpu.state().unwrap();
    g_a.upload_to_device(device, queue).unwrap();
    g_b.upload_to_device(device, queue).unwrap();
    let mut control_gpu = model_gpu.control();
    let _ = control_gpu.body_external_f.ensure_device(device);
    let mut contacts_gpu = Contacts::with_capacity(64);
    let mut pipeline_gpu = CollisionPipeline::new();
    let mut solver_gpu = XpbdSolver::new_gpu(&engine, 1).expect("new_gpu");
    solver_gpu.position_iterations = 8;
    solver_gpu = solver_gpu.with_substeps(substeps);
    let mut gpu_traj = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        g_a.download_from_device(device, queue).unwrap();
        pipeline_gpu
            .collide(&model_gpu, &g_a, &mut contacts_gpu)
            .unwrap();
        contacts_gpu.upload_to_device(device, queue).unwrap();
        solver_gpu
            .step(&model_gpu, &g_a, &control_gpu, &contacts_gpu, dt, &mut g_b)
            .unwrap();
        g_b.download_from_device(device, queue).unwrap();
        std::mem::swap(&mut g_a, &mut g_b);
        let q = g_a.body_q.host().unwrap()[0];
        gpu_traj.push((q.position.x, q.position.y));
    }

    let mut max_dx: Real = 0.0;
    let mut max_dy: Real = 0.0;
    for ((cx, cy), (gx, gy)) in cpu_traj.iter().zip(gpu_traj.iter()) {
        max_dx = max_dx.max((cx - gx).abs());
        max_dy = max_dy.max((cy - gy).abs());
    }
    let (cpu_x, _) = *cpu_traj.last().unwrap();
    let (gpu_x, gpu_y) = *gpu_traj.last().unwrap();
    eprintln!(
        "sliding box multi-substep: final cpu_x={cpu_x:.4} gpu_x={gpu_x:.4}, \
         max |Δx|={max_dx:.4} |Δy|={max_dy:.4}"
    );
    // Non-vacuous: friction must have done real work — the box slid a
    // meaningful distance and stopped short of frictionless coasting
    // (2 m/s × 1 s = 2 m).
    assert!(
        cpu_x > 0.2 && cpu_x < 1.9,
        "friction deceleration looks wrong: slid {cpu_x} m"
    );
    // |Δx| gate: 6 cm over a ~0.5 m slide. Stick-slip transitions
    // resolve on different substeps in f32 vs f64, which shifts the
    // stopping point by a few cm — the same order as the 5 cm gate the
    // single-substep stack parity test uses. The bug classes this
    // guards against are an order of magnitude bigger: the pre-fix
    // missing support-impulse floor made the GPU glide to 1.38 m
    // (0.86 m divergence), and anchor drift diverges similarly.
    assert!(
        max_dx < 0.06 && max_dy < 0.02,
        "GPU multi-substep diverged from the CPU internal-substep \
         reference: max |Δx| = {max_dx}, |Δy| = {max_dy}"
    );
    assert!(
        (gpu_y - 0.5).abs() < 0.02,
        "box left the ground during a friction slide: y = {gpu_y}"
    );
}
