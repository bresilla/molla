//! Stage 15 — full GPU narrow-phase tests. Runs EPA and mesh-BVH
//! ray-cast on the device and compares against the CPU reference.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::geometry::{epa as cpu_epa, gjk as cpu_gjk, GjkResult, MeshBvh};
use molla::kernels::{
    decode_epa_out, EpaKernel, GjkKernel, MeshBvhGpu, MeshBvhRaycastKernel, EPA_OUT_STRIDE,
    EPA_STATUS_OK, GJK_SIMPLEX_STRIDE,
};
use molla::math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;
use molla::sim::ShapeGeometry;
use molla::storage::{Buffer, BufferUsage};

fn upload_body_q(
    state: &molla::sim::State,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Buffer<Transform> {
    let mut buf = Buffer::from_vec(state.body_q.host().unwrap().to_vec(), BufferUsage::Storage);
    buf.ensure_device(device).unwrap();
    buf.upload(queue).unwrap();
    buf
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_epa_box_box_matches_cpu_within_1mm() {
    // Two unit-half boxes, the second offset by 1.5 along +X (overlap 0.5).
    let mut b = ModelBuilder::new();
    let mass = 1.0;
    let inertia = Mat3::IDENTITY;
    let body_a = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(V3::ZERO),
    );
    let sa = b.add_shape_box(body_a, V3::splat(1.0), Transform::IDENTITY).unwrap();
    let body_b = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(V3::new(1.5, 0.0, 0.0)),
    );
    let sb = b.add_shape_box(body_b, V3::splat(1.0), Transform::IDENTITY).unwrap();
    let mut model = b.build().unwrap();

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let state = model.state().unwrap();
    let body_q_buf = upload_body_q(&state, &device, &queue);

    // CPU EPA baseline.
    let xf_a = molla::geometry::shape_world_transform(&model, &state, sa.index());
    let xf_b = molla::geometry::shape_world_transform(&model, &state, sb.index());
    let geom_a = ShapeGeometry::Box { half_extents: V3::splat(1.0) };
    let geom_b = ShapeGeometry::Box { half_extents: V3::splat(1.0) };
    let simplex = match cpu_gjk(geom_a, &xf_a, geom_b, &xf_b) {
        GjkResult::Overlap(s) => s,
        _ => panic!("expected overlap on CPU GJK"),
    };
    let cpu_contact = cpu_epa(geom_a, &xf_a, geom_b, &xf_b, &simplex).expect("cpu epa");

    // GPU GJK + EPA.
    let gjk_kernel = GjkKernel::new(&device);
    let epa_kernel = EpaKernel::new(&device);
    let pair_count = 1usize;
    let (mut pair_a, mut pair_b, mut overlap) =
        GjkKernel::alloc_io(&device, &queue, pair_count).unwrap();
    let mut simplex_buf = GjkKernel::alloc_simplex(&device, &queue, pair_count).unwrap();
    let mut epa_out = EpaKernel::alloc_out(&device, &queue, pair_count).unwrap();
    {
        pair_a.host_mut().unwrap()[0] = sa.index() as u32;
        pair_b.host_mut().unwrap()[0] = sb.index() as u32;
    }
    pair_a.upload(&queue).unwrap();
    pair_b.upload(&queue).unwrap();
    gjk_kernel
        .dispatch_with_simplex(
            &device,
            &queue,
            &model,
            body_q_buf.device_buffer().unwrap(),
            &pair_a,
            &pair_b,
            &mut overlap,
            &mut simplex_buf,
            pair_count as u32,
        )
        .unwrap();
    overlap.download(&device, &queue).unwrap();
    assert_eq!(overlap.host().unwrap()[0], 1, "GJK should report overlap");

    epa_kernel
        .dispatch(
            &device,
            &queue,
            &model,
            body_q_buf.device_buffer().unwrap(),
            &pair_a,
            &pair_b,
            &overlap,
            &simplex_buf,
            &mut epa_out,
            pair_count as u32,
        )
        .unwrap();
    epa_out.download(&device, &queue).unwrap();
    let host = epa_out.host().unwrap();
    let res = decode_epa_out(host, pair_count);
    let r = res[0];
    eprintln!(
        "CPU: depth={:.4} normal={:?}; GPU: status={} depth={:.4} normal={:?}",
        cpu_contact.depth, cpu_contact.normal, r.status, r.depth, r.normal
    );
    assert_eq!(r.status, EPA_STATUS_OK, "GPU EPA should succeed");
    let depth_err = (r.depth as Real - cpu_contact.depth).abs();
    let normal_err = (r.normal - cpu_contact.normal).length();
    assert!(depth_err < 1e-3, "depth diff: {} (cpu={}, gpu={})", depth_err, cpu_contact.depth, r.depth);
    assert!(normal_err < 1e-2, "normal diff: {} (cpu={:?}, gpu={:?})", normal_err, cpu_contact.normal, r.normal);
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_epa_capsule_box_falls_back_for_degenerate_simplex() {
    // Engineered case: capsule whose centre coincides with the box
    // centre (depth >= half_extent). EPA should produce a valid
    // contact OR signal fallback when the symmetric simplex degenerates.
    let mut b = ModelBuilder::new();
    let mass = 1.0;
    let inertia = Mat3::IDENTITY;
    let body_a = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(V3::ZERO),
    );
    let sa = b.add_shape_capsule(body_a, 0.4, 0.3, Transform::IDENTITY).unwrap();
    let body_b = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(V3::ZERO),
    );
    let sb = b.add_shape_box(body_b, V3::splat(0.5), Transform::IDENTITY).unwrap();
    let mut model = b.build().unwrap();

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let state = model.state().unwrap();
    let body_q_buf = upload_body_q(&state, &device, &queue);

    let gjk_kernel = GjkKernel::new(&device);
    let epa_kernel = EpaKernel::new(&device);
    let pair_count = 1usize;
    let (mut pair_a, mut pair_b, mut overlap) =
        GjkKernel::alloc_io(&device, &queue, pair_count).unwrap();
    let mut simplex_buf = GjkKernel::alloc_simplex(&device, &queue, pair_count).unwrap();
    let mut epa_out = EpaKernel::alloc_out(&device, &queue, pair_count).unwrap();
    {
        pair_a.host_mut().unwrap()[0] = sa.index() as u32;
        pair_b.host_mut().unwrap()[0] = sb.index() as u32;
    }
    pair_a.upload(&queue).unwrap();
    pair_b.upload(&queue).unwrap();
    gjk_kernel
        .dispatch_with_simplex(
            &device,
            &queue,
            &model,
            body_q_buf.device_buffer().unwrap(),
            &pair_a,
            &pair_b,
            &mut overlap,
            &mut simplex_buf,
            pair_count as u32,
        )
        .unwrap();
    epa_kernel
        .dispatch(
            &device,
            &queue,
            &model,
            body_q_buf.device_buffer().unwrap(),
            &pair_a,
            &pair_b,
            &overlap,
            &simplex_buf,
            &mut epa_out,
            pair_count as u32,
        )
        .unwrap();
    overlap.download(&device, &queue).unwrap();
    let ov = overlap.host().unwrap()[0];
    epa_out.download(&device, &queue).unwrap();
    let host = epa_out.host().unwrap();
    let res = decode_epa_out(host, pair_count);
    let r = res[0];
    eprintln!(
        "capsule-box overlap={ov} status={} depth={} normal={:?}",
        r.status, r.depth, r.normal
    );
    // Either a valid contact (kernel converged) OR fallback (host CPU
    // EPA picks it up). Both are acceptable — what matters is no
    // garbage / no infinite loop.
    assert!(r.status == EPA_STATUS_OK || r.status == molla::kernels::EPA_STATUS_FALLBACK);
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_mesh_bvh_raycast_hits_unit_cube_face() {
    // 12-tri unit cube centred at origin, ray from (-2, 0, 0) along +X.
    let v = vec![
        V3::new(-0.5, -0.5, -0.5),
        V3::new(0.5, -0.5, -0.5),
        V3::new(0.5, 0.5, -0.5),
        V3::new(-0.5, 0.5, -0.5),
        V3::new(-0.5, -0.5, 0.5),
        V3::new(0.5, -0.5, 0.5),
        V3::new(0.5, 0.5, 0.5),
        V3::new(-0.5, 0.5, 0.5),
    ];
    let i = vec![
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ];
    let bvh = MeshBvh::build(&v, &i, 4);

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mesh_gpu = MeshBvhGpu::upload(&device, &queue, &bvh, &v, &i).unwrap();
    let kernel = MeshBvhRaycastKernel::new(&device);

    let ray_count = 1usize;
    let mut origin = Buffer::from_vec(vec![V3::new(-2.0, 0.0, 0.0)], BufferUsage::Storage);
    let mut dir = Buffer::from_vec(vec![V3::X], BufferUsage::Storage);
    origin.ensure_device(&device).unwrap();
    dir.ensure_device(&device).unwrap();
    origin.upload(&queue).unwrap();
    dir.upload(&queue).unwrap();
    let (mut hit_t, mut hit_tri, mut hit_normal) =
        MeshBvhRaycastKernel::alloc_hits(&device, &queue, ray_count).unwrap();
    kernel
        .dispatch(
            &device, &queue, &mesh_gpu, &origin, &dir, &mut hit_t, &mut hit_tri, &mut hit_normal,
            100.0, ray_count as u32,
        )
        .unwrap();
    hit_t.download(&device, &queue).unwrap();
    hit_tri.download(&device, &queue).unwrap();
    hit_normal.download(&device, &queue).unwrap();
    let t = hit_t.host().unwrap()[0];
    let tri = hit_tri.host().unwrap()[0];
    eprintln!("mesh-bvh ray hit: t={t}, tri={tri}");
    assert!(tri >= 0, "expected a triangle hit");
    assert!((t as f64 - 1.5).abs() < 1e-3, "expected t ≈ 1.5, got {t}");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_pipeline_box_stack_settles_within_5mm_vs_cpu() {
    use molla::geometry::CollisionPipeline;
    use molla::prelude::Solver;
    use molla::sim::Contacts;

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
        b.add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY).unwrap();
        for k in 0..10 {
            let body = b.add_body(
                BodyParams::new()
                    .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
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
    let mut solver_cpu = molla::prelude::XpbdSolver::new().with_iterations(8);
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
    let mut solver_g = molla::prelude::XpbdSolver::new().with_iterations(8);
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

#[allow(dead_code)]
fn assert_strides() {
    // Compile-time-ish reminder: keep host & shader strides in sync.
    assert_eq!(EPA_OUT_STRIDE, 16);
    assert_eq!(GJK_SIMPLEX_STRIDE, 12);
}
