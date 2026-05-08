//! Stage 6.x — triangle / trimesh integration tests.

use molla::geometry::CollisionPipeline;
use molla::prelude::*;
use molla::sim::TriMesh;

#[test]
fn sphere_settles_on_triangle() {
    // Single triangle on the y=0 plane (CCW from +Y view), large
    // enough to catch a sphere dropped from height.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 5.0, 0.0)),
    );
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();

    // Static triangle at world origin in the y=0 plane, vertices in
    // CCW order looking from +Y.
    b.add_shape_triangle(
        BodyId::NONE,
        Vec3::new(-5.0, 0.0, -5.0),
        Vec3::new(5.0, 0.0, -5.0),
        Vec3::new(0.0, 0.0, 5.0),
        Transform::IDENTITY,
    )
    .unwrap();

    let mut model = b.build().unwrap();
    model.default_friction = 0.5;
    model.default_restitution = 0.0;

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8);

    for _ in 0..1000 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let y = q[0].position.y;
    let v = s_a.body_qd.host().unwrap()[0].linear.length();
    eprintln!("sphere on triangle: y={y:.4} |v|={v:.4}");
    assert!(
        (y - 0.5).abs() < 0.05,
        "sphere on triangle: y={y}, expected ~0.5"
    );
    assert!(v < 0.2, "still moving: |v|={v}");
}

#[test]
fn sphere_on_trimesh_quad_floor() {
    // Floor as a 2-triangle quad, registered via TriMesh.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 5.0, 0.0)),
    );
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();

    let mut mesh = TriMesh::new();
    mesh.vertices = vec![
        Vec3::new(-3.0, 0.0, -3.0),
        Vec3::new(3.0, 0.0, -3.0),
        Vec3::new(3.0, 0.0, 3.0),
        Vec3::new(-3.0, 0.0, 3.0),
    ];
    mesh.indices = vec![[0, 1, 2], [0, 2, 3]];
    let mesh_id = b.add_mesh(mesh);
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
        .unwrap();

    let mut model = b.build().unwrap();
    model.default_friction = 0.5;
    model.default_restitution = 0.0;

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8);

    for _ in 0..1000 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let y = q[0].position.y;
    eprintln!("sphere on trimesh quad: y={y:.4}");
    assert!((y - 0.5).abs() < 0.05, "y={y}");
}
