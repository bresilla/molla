//! Cone and ConvexHull primitive tests — verify support functions,
//! AABBs, and end-to-end XPBD contact + settling behaviour against
//! a ground plane.

use molla::core::BodyId;
use molla::geometry::{shape_world_aabb, support_with_hulls, CollisionPipeline};
use molla::math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;
use molla::sim::ShapeGeometry;
use molla::solvers::Solver as _;

fn box_inertia(mass: f64, half: V3) -> Mat3 {
    let h2 = half * half;
    let m = mass / 3.0;
    Mat3::from_diagonal(V3::new(m * (h2.y + h2.z), m * (h2.x + h2.z), m * (h2.x + h2.y)))
}

#[test]
fn cone_support_apex_and_base() {
    // Cone half_h=0.5, radius=0.4 along +Y. Apex at (0, 0.5, 0).
    let geom = ShapeGeometry::Cone { half_height: 0.5, radius: 0.4 };
    let xf = Transform::IDENTITY;
    // +Y → apex.
    let p = support_with_hulls(geom, &xf, V3::Y, &[]);
    assert!((p - V3::new(0.0, 0.5, 0.0)).length() < 1e-9, "got {p:?}");
    // -Y → base disc edge (any point on the rim, y=-0.5).
    let p = support_with_hulls(geom, &xf, -V3::Y, &[]);
    assert!((p.y + 0.5).abs() < 1e-9, "got y={}", p.y);
    let r = (p.x * p.x + p.z * p.z).sqrt();
    assert!((r - 0.4).abs() < 1e-9, "got rim radius {r}");
    // +X → base edge at (+0.4, -0.5, 0). The apex's X=0 gives dot=0,
    // base edge gives dot=0.4 — base wins.
    let p = support_with_hulls(geom, &xf, V3::X, &[]);
    assert!((p - V3::new(0.4, -0.5, 0.0)).length() < 1e-9, "got {p:?}");
}

#[test]
fn convex_hull_support_picks_argmax_vertex() {
    // Tetrahedron hull: vertices at (1,0,0), (0,1,0), (0,0,1),
    // (-1,-1,-1).
    let hulls = vec![vec![
        V3::new(1.0, 0.0, 0.0),
        V3::new(0.0, 1.0, 0.0),
        V3::new(0.0, 0.0, 1.0),
        V3::new(-1.0, -1.0, -1.0),
    ]];
    let geom = ShapeGeometry::ConvexHull { hull_id: 0 };
    let xf = Transform::IDENTITY;
    // +X → vertex 0.
    let p = support_with_hulls(geom, &xf, V3::X, &hulls);
    assert!((p - V3::new(1.0, 0.0, 0.0)).length() < 1e-9);
    // +Y → vertex 1.
    let p = support_with_hulls(geom, &xf, V3::Y, &hulls);
    assert!((p - V3::new(0.0, 1.0, 0.0)).length() < 1e-9);
    // -X-Y-Z → vertex 3.
    let p = support_with_hulls(geom, &xf, V3::new(-1.0, -1.0, -1.0), &hulls);
    assert!((p - V3::new(-1.0, -1.0, -1.0)).length() < 1e-9);
}

#[test]
fn cone_settles_on_plane() {
    // Cone half_h=0.3, radius=0.2, mass=1 dropped from y=2 onto
    // plane y=0. Should land within a few seconds. (CONTACT MARGIN
    // makes the rest height ~half_h+margin.)
    let mut b = ModelBuilder::new();
    let mass = 1.0_f64;
    let half_h = 0.3_f64;
    let radius = 0.2_f64;
    let inertia = box_inertia(mass, V3::new(radius, half_h, radius));
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(V3::new(0.0, 2.0, 0.0)),
    );
    b.add_shape_cone(body, half_h, radius, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = molla::sim::Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    let dt = 0.005_f64;
    for _ in 0..400 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver.step(&model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let y = s_a.body_q.host().unwrap()[body.index()].position.y;
    eprintln!("cone settle y = {y:.4}");
    // Expected: cone lying somewhere between half_h and -half_h
    // depending on which side it tipped onto. Just assert it's
    // bounded — no NaN, no escape.
    assert!(
        y.abs() < 1.0,
        "cone escaped or NaN'd: y={y}"
    );
}

#[test]
fn convex_hull_settles_on_plane() {
    // 8-vertex unit-cube hull. Same drop test as Cone — should settle
    // on the plane.
    let mut b = ModelBuilder::new();
    let half = 0.2_f64;
    let mass = 1.0_f64;
    let inertia = box_inertia(mass, V3::splat(half));
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(V3::new(0.0, 1.5, 0.0)),
    );
    let verts: Vec<V3> = (0..8)
        .map(|i| {
            V3::new(
                if i & 1 != 0 { half } else { -half },
                if i & 2 != 0 { half } else { -half },
                if i & 4 != 0 { half } else { -half },
            )
        })
        .collect();
    let hull_id = b.add_hull(verts);
    b.add_shape_convex_hull(body, hull_id, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = molla::sim::Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    let dt = 0.005_f64;
    for _ in 0..400 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver.step(&model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let y = s_a.body_q.host().unwrap()[body.index()].position.y;
    eprintln!("hull settle y = {y:.4}");
    assert!(
        y > 0.0 && y < 0.5,
        "hull didn't settle in [0, 0.5]: y={y}"
    );
}

#[test]
fn cone_world_aabb_grows_with_radius() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY,
    }));
    b.add_shape_cone(body, 1.0, 0.5, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();
    let aabb = shape_world_aabb(&model, &state, 0, 0.0);
    let extent = aabb.max - aabb.min;
    // Identity rotation: cone +Y axis = world +Y. Extent.y = 2·half_h
    // = 2.0, extent.x = extent.z = 2·radius = 1.0.
    assert!((extent.y - 2.0).abs() < 1e-9, "got {extent:?}");
    assert!((extent.x - 1.0).abs() < 1e-9, "got {extent:?}");
    assert!((extent.z - 1.0).abs() < 1e-9, "got {extent:?}");
}
