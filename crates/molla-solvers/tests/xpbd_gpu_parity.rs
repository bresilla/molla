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
#[ignore = "needs a real wgpu adapter — run via `make test-xpbd-gpu`"]
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
