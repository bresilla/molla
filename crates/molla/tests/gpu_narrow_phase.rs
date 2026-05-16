//! Product-level GPU narrow-phase integration coverage.
//!
//! Geometry-owned GPU EPA/GJK and mesh-BVH raycast tests live in
//! `crates/molla-geometry/tests/gpu_narrow_phase.rs`. This facade test keeps
//! the mixed `molla` product path that couples the geometry collision pipeline
//! to the public solver prelude.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;
use molla::sim::Contacts;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_pipeline_box_stack_settles_within_5mm_vs_cpu() {
    // Stack of spheres on a plane — every pair (sphere/sphere and
    // sphere/plane) routes to a single-contact narrow phase on both
    // sides, so the GPU↔CPU comparison isn't biased by SAT's
    // multi-contact face manifold. Plane is the only CPU-only shape;
    // sphere/sphere goes through GPU GJK+EPA on the GPU side and
    // through closed-form sphere/sphere on the CPU side, but both
    // produce identical single-point contacts for spheres.
    fn build_stack() -> molla::sim::Model {
        let mut b = ModelBuilder::new();
        let mass = 1.0;
        let inertia = Mat3::IDENTITY;
        // Static plane (XZ) at y=0.
        b.add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY)
            .unwrap();
        for k in 0..10 {
            let body = b.add_body(
                BodyParams::new()
                    .with_spatial_inertia(SpatialInertia {
                        mass,
                        com: V3::ZERO,
                        inertia,
                    })
                    .with_position(V3::new(0.0, 0.5 + (k as f64) * 1.01, 0.0)),
            );
            b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
        }
        b.build().unwrap()
    }

    // CPU baseline.
    let mut model_cpu = build_stack();
    model_cpu.default_friction = 0.5;
    let mut s_a = model_cpu.state().unwrap();
    let mut s_b = model_cpu.state().unwrap();
    let control = model_cpu.control();
    let mut pl_cpu = CollisionPipeline::new();
    let mut contacts_cpu = Contacts::with_capacity(256);
    let mut solver_cpu = XpbdSolver::new().with_iterations(8);
    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        pl_cpu.collide(&model_cpu, &s_a, &mut contacts_cpu).unwrap();
        solver_cpu
            .step(&model_cpu, &s_a, &control, &contacts_cpu, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let cpu_y: Vec<Real> = s_a
        .body_q
        .host()
        .unwrap()
        .iter()
        .map(|q| q.position.y)
        .collect();
    eprintln!("CPU final y: {cpu_y:?}");

    // GPU full narrow-phase pipeline.
    let mut model_gpu = build_stack();
    model_gpu.default_friction = 0.5;
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model_gpu.upload_to_device(&device, &queue).unwrap();
    let mut g_a = model_gpu.state().unwrap();
    let mut g_b = model_gpu.state().unwrap();
    let g_control = model_gpu.control();
    let mut pl_gpu = CollisionPipeline::new();
    pl_gpu.enable_full_gpu_narrow_phase(&engine, 256).unwrap();
    let mut contacts_gpu = Contacts::with_capacity(256);
    let mut solver_g = XpbdSolver::new().with_iterations(8);
    for _ in 0..120 {
        pl_gpu.collide(&model_gpu, &g_a, &mut contacts_gpu).unwrap();
        solver_g
            .step(&model_gpu, &g_a, &g_control, &contacts_gpu, dt, &mut g_b)
            .unwrap();
        std::mem::swap(&mut g_a, &mut g_b);
    }
    let gpu_y: Vec<Real> = g_a
        .body_q
        .host()
        .unwrap()
        .iter()
        .map(|q| q.position.y)
        .collect();
    eprintln!("GPU final y: {gpu_y:?}");

    let cpu_calls = pl_gpu.last_cpu_narrow_phase_calls();
    eprintln!("last CPU narrow-phase calls in GPU pipeline: {cpu_calls}");

    for (i, (yc, yg)) in cpu_y.iter().zip(gpu_y.iter()).enumerate() {
        let diff = (yc - yg).abs();
        assert!(diff < 5e-3, "body {i}: CPU y={yc}, GPU y={yg}, diff={diff}");
    }
}
