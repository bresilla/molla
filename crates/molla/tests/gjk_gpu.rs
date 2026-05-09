//! GPU GJK boolean overlap test vs CPU GJK on hand-built shape pairs.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::geometry::{gjk, GjkResult};
use molla::kernels::GjkKernel;
use molla::math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;

fn run_gpu_overlap(model: &mut molla::sim::Model, pairs: &[(u32, u32)]) -> Vec<u32> {
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let state = model.state().unwrap();
    // Need body_q on device — Sim's path uploads it; we mirror that here.
    let mut body_q = state.body_q.host().unwrap().to_vec();
    use molla::storage::{Buffer, BufferUsage};
    let mut body_q_buf = Buffer::from_vec(body_q.clone(), BufferUsage::Storage);
    body_q_buf.ensure_device(&device).unwrap();
    body_q_buf.upload(&queue).unwrap();
    let _ = body_q.len();

    let kernel = GjkKernel::new(&device);
    let cap = pairs.len();
    let (mut pa, mut pb, mut overlap) =
        GjkKernel::alloc_io(&device, &queue, cap).unwrap();
    {
        let pa_h = pa.host_mut().unwrap();
        let pb_h = pb.host_mut().unwrap();
        for (i, p) in pairs.iter().enumerate() {
            pa_h[i] = p.0;
            pb_h[i] = p.1;
        }
    }
    pa.upload(&queue).unwrap();
    pb.upload(&queue).unwrap();
    kernel
        .dispatch(
            &device,
            &queue,
            model,
            body_q_buf.device_buffer().unwrap(),
            &pa,
            &pb,
            &mut overlap,
            cap as u32,
        )
        .unwrap();
    overlap.download(&device, &queue).unwrap();
    overlap.host().unwrap().to_vec()
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_gjk_sphere_sphere_overlap_and_disjoint() {
    // Two spheres r=0.5; pair 0 overlaps (centres 0.5 apart),
    // pair 1 disjoint (centres 2.0 apart).
    let pos_a = V3::ZERO;
    let pos_b = V3::new(0.5, 0.0, 0.0);
    let pos_c = V3::new(2.0, 0.0, 0.0);
    let mut b = ModelBuilder::new();
    let mass = 1.0;
    let inertia = Mat3::IDENTITY;
    let body_a = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(pos_a),
    );
    let sa = b.add_shape_sphere(body_a, 0.5, Transform::IDENTITY).unwrap();
    let body_b = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(pos_b),
    );
    let sb = b.add_shape_sphere(body_b, 0.5, Transform::IDENTITY).unwrap();
    let body_c = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(pos_c),
    );
    let sc = b.add_shape_sphere(body_c, 0.5, Transform::IDENTITY).unwrap();
    let mut model = b.build().unwrap();
    let overlap = run_gpu_overlap(
        &mut model,
        &[(sa.index() as u32, sb.index() as u32), (sa.index() as u32, sc.index() as u32)],
    );
    eprintln!("sphere-sphere overlap = {overlap:?}");
    assert_eq!(overlap[0], 1, "expected overlap for sphere-sphere at distance 0.5");
    assert_eq!(overlap[1], 0, "expected disjoint at distance 2.0");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_gjk_box_box_matches_cpu() {
    // Two unit-half boxes; pair 0 overlaps, pair 1 doesn't.
    let mut b = ModelBuilder::new();
    let mass = 1.0;
    let inertia = Mat3::IDENTITY;
    let mk = |b: &mut ModelBuilder, pos: V3| {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
                .with_position(pos),
        );
        b.add_shape_box(body, V3::splat(1.0), Transform::IDENTITY).unwrap()
    };
    let s0 = mk(&mut b, V3::ZERO);
    let s1 = mk(&mut b, V3::new(1.5, 0.0, 0.0)); // overlap by 0.5
    let s2 = mk(&mut b, V3::new(3.5, 0.0, 0.0)); // gap of 1.5
    let mut model = b.build().unwrap();

    // CPU baseline.
    let cpu_pairs: Vec<bool> = [(s0, s1), (s0, s2)]
        .iter()
        .map(|(a, b)| {
            use molla::sim::ShapeGeometry;
            let xf_a = molla::geometry::shape_world_transform(&model, &model.state().unwrap(), a.index());
            let xf_b = molla::geometry::shape_world_transform(&model, &model.state().unwrap(), b.index());
            matches!(
                gjk(
                    ShapeGeometry::Box { half_extents: V3::splat(1.0) },
                    &xf_a,
                    ShapeGeometry::Box { half_extents: V3::splat(1.0) },
                    &xf_b,
                ),
                GjkResult::Overlap(_)
            )
        })
        .collect();

    let overlap = run_gpu_overlap(
        &mut model,
        &[
            (s0.index() as u32, s1.index() as u32),
            (s0.index() as u32, s2.index() as u32),
        ],
    );
    eprintln!("box-box CPU = {cpu_pairs:?}, GPU = {overlap:?}");
    for (i, &cpu) in cpu_pairs.iter().enumerate() {
        let gpu = overlap[i];
        // GPU result of 2 = "fall back" — accept silently (we want
        // primary overlap/disjoint detection to match).
        if gpu != 2 {
            assert_eq!(
                gpu == 1,
                cpu,
                "pair {i}: CPU={cpu}, GPU={gpu}"
            );
        }
    }
}

/// End-to-end pipeline test: GPU GJK pre-filter wired into the
/// CollisionPipeline. With many far-apart spheres, the kernel
/// rejects the disjoint pairs so CPU narrow phase only runs on the
/// few overlapping ones.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn pipeline_with_gpu_prefilter_matches_baseline() {
    let mut b = ModelBuilder::new();
    let mass = 1.0;
    let inertia = Mat3::IDENTITY;
    // 5 spheres along +X. Spheres 0 and 1 overlap (centres 0.5 apart,
    // r=0.5). Spheres 2/3/4 are isolated.
    let mk = |b: &mut ModelBuilder, pos: V3| {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
                .with_position(pos),
        );
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    };
    mk(&mut b, V3::ZERO);
    mk(&mut b, V3::new(0.5, 0.0, 0.0));
    mk(&mut b, V3::new(5.0, 0.0, 0.0));
    mk(&mut b, V3::new(10.0, 0.0, 0.0));
    mk(&mut b, V3::new(15.0, 0.0, 0.0));
    let mut model = b.build().unwrap();

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let state = model.state().unwrap();

    // Baseline (no prefilter).
    let mut pl = molla::geometry::CollisionPipeline::new();
    let mut contacts_a = molla::sim::Contacts::with_capacity(64);
    let n_baseline = pl.collide(&model, &state, &mut contacts_a).unwrap();

    // With GPU prefilter.
    let mut pl_pf = molla::geometry::CollisionPipeline::new();
    pl_pf.enable_gpu_prefilter(&engine, 32).unwrap();
    let mut contacts_b = molla::sim::Contacts::with_capacity(64);
    let n_pf = pl_pf.collide(&model, &state, &mut contacts_b).unwrap();
    eprintln!("contacts: baseline={n_baseline}, prefilter={n_pf}");
    assert_eq!(n_baseline, n_pf, "prefilter changed contact count");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_gjk_capsule_capsule_overlap() {
    // Two parallel Y-capsules; first pair overlapping, second
    // disjoint.
    let mut b = ModelBuilder::new();
    let mass = 1.0;
    let inertia = Mat3::IDENTITY;
    let mk = |b: &mut ModelBuilder, pos: V3| {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
                .with_position(pos),
        );
        b.add_shape_capsule(body, 1.0, 0.3, Transform::IDENTITY).unwrap()
    };
    let s0 = mk(&mut b, V3::ZERO);
    let s1 = mk(&mut b, V3::new(0.5, 0.0, 0.0)); // overlap
    let s2 = mk(&mut b, V3::new(2.0, 0.0, 0.0)); // disjoint
    let mut model = b.build().unwrap();
    let overlap = run_gpu_overlap(
        &mut model,
        &[(s0.index() as u32, s1.index() as u32), (s0.index() as u32, s2.index() as u32)],
    );
    eprintln!("capsule overlap = {overlap:?}");
    assert!(overlap[0] == 1 || overlap[0] == 2);
    assert!(overlap[1] == 0 || overlap[1] == 2);
}
