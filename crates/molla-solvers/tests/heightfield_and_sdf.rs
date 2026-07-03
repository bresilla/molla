//! Heightfield + SDF primitives — drop tests + AABB sanity.

use molla_core::BodyId;
use molla_geometry::{CollisionPipeline, shape_world_aabb};
use molla_math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Contacts, Heightfield, ModelBuilder, SdfShape};
use molla_solvers::{Solver, XpbdSolver};

fn strongest_contact(contacts: &Contacts) -> Option<(f64, V3, V3)> {
    if contacts.count == 0 {
        return None;
    }
    let depths = contacts.depth.host().expect("contact depth");
    let normals = contacts.normal.host().expect("contact normal");
    let points = contacts.point.host().expect("contact point");
    let best = (0..contacts.count as usize)
        .max_by(|&a, &b| depths[a].total_cmp(&depths[b]))
        .expect("non-empty contact set");
    Some((depths[best], normals[best], points[best]))
}

fn sphere_inertia(mass: f64, r: f64) -> Mat3 {
    let i = (2.0 / 5.0) * mass * r * r;
    Mat3::from_diagonal(V3::new(i, i, i))
}

fn box_inertia(mass: f64, half: V3) -> Mat3 {
    let h2 = half * half;
    let s = mass / 3.0;
    Mat3::from_diagonal(V3::new(
        s * (h2.y + h2.z),
        s * (h2.x + h2.z),
        s * (h2.x + h2.y),
    ))
}

#[test]
fn heightfield_local_aabb_matches_extents() {
    let hf = Heightfield {
        nx: 4,
        nz: 4,
        cell_size: 1.0,
        origin: V3::new(-1.5, 0.0, -1.5),
        heights: vec![0.0; 16],
    };
    let (lo, hi) = hf.local_aabb();
    // origin (-1.5, 0, -1.5), 4 samples, spacing 1 → max = (-1.5 + 3, 0, -1.5 + 3) = (1.5, 0, 1.5).
    assert!((lo - V3::new(-1.5, 0.0, -1.5)).length() < 1e-9);
    assert!((hi - V3::new(1.5, 0.0, 1.5)).length() < 1e-9);
}

#[test]
fn sphere_settles_on_flat_heightfield() {
    let mut b = ModelBuilder::new();
    let r = 0.2_f64;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: sphere_inertia(1.0, r),
            })
            .with_position(V3::new(0.0, 2.0, 0.0)),
    );
    b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
    // 4×4 flat heightfield at y = 0, covering x,z ∈ [-2, 2].
    let hf = Heightfield {
        nx: 5,
        nz: 5,
        cell_size: 1.0,
        origin: V3::new(-2.0, 0.0, -2.0),
        heights: vec![0.0; 25],
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(BodyId::NONE, field_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    let dt = 0.005_f64;
    for _ in 0..400 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let y = s_a.body_q.host().unwrap()[body.index()].position.y;
    eprintln!("sphere on heightfield: y = {y:.4} m");
    // Sphere centre should sit ~r above the surface with a bit of
    // contact-margin slack. Allow a generous band.
    assert!(
        (y - r).abs() < 0.05,
        "sphere didn't settle on heightfield: y={y}"
    );
}

#[test]
fn box_settles_on_flat_heightfield_without_tunneling() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.08);
    let half = V3::new(0.25, 0.15, 0.20);
    let mass = 1.0_f64;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia: box_inertia(mass, half),
            })
            .with_position(V3::new(0.0, 1.5, 0.0)),
    );
    b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
    let hf = Heightfield {
        nx: 5,
        nz: 5,
        cell_size: 1.0,
        origin: V3::new(-2.0, 0.0, -2.0),
        heights: vec![0.0; 25],
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(BodyId::NONE, field_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(128);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(10);
    let dt = 0.005_f64;
    let mut max_contacts = 0u32;
    for _ in 0..500 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        max_contacts = max_contacts.max(contacts.count);
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let pose = s_a.body_q.host().unwrap()[body.index()];
    let vel = s_a.body_qd.host().unwrap()[body.index()];
    let y = pose.position.y;
    eprintln!("box on heightfield: y = {y:.4}, max_contacts = {max_contacts}");
    assert!(max_contacts > 0, "box never contacted the heightfield");
    assert!(
        pose.position.is_finite() && vel.linear.is_finite() && vel.angular.is_finite(),
        "box-heightfield settle produced non-finite state: pose={pose:?}, vel={vel:?}"
    );
    assert!(
        y > half.y - 0.02,
        "box tunneled below heightfield: center_y={y}, half_y={}",
        half.y
    );
    assert!(
        y < half.y + 0.10,
        "box did not settle on heightfield: center_y={y}, half_y={}",
        half.y
    );
    assert!(
        vel.linear.length() < 0.25,
        "box still moving too fast on heightfield: |v|={}",
        vel.linear.length()
    );
}

#[test]
fn convex_hull_settles_on_flat_heightfield_without_tunneling() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.08);
    let half = 0.18_f64;
    let mass = 1.0_f64;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia: box_inertia(mass, V3::splat(half)),
            })
            .with_position(V3::new(0.0, 1.2, 0.0)),
    );
    let hull_vertices: Vec<V3> = (0..8)
        .map(|i| {
            V3::new(
                if i & 1 != 0 { half } else { -half },
                if i & 2 != 0 { half } else { -half },
                if i & 4 != 0 { half } else { -half },
            )
        })
        .collect();
    let hull_id = b.add_hull(hull_vertices);
    b.add_shape_convex_hull(body, hull_id, Transform::IDENTITY)
        .unwrap();
    let hf = Heightfield {
        nx: 5,
        nz: 5,
        cell_size: 1.0,
        origin: V3::new(-2.0, 0.0, -2.0),
        heights: vec![0.0; 25],
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(BodyId::NONE, field_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(128);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(10);
    let dt = 0.005_f64;
    let mut max_contacts = 0u32;
    for _ in 0..500 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        max_contacts = max_contacts.max(contacts.count);
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let pose = s_a.body_q.host().unwrap()[body.index()];
    let vel = s_a.body_qd.host().unwrap()[body.index()];
    let y = pose.position.y;
    eprintln!("convex hull on heightfield: y = {y:.4}, max_contacts = {max_contacts}");
    assert!(
        max_contacts > 0,
        "convex hull never contacted the heightfield"
    );
    assert!(
        pose.position.is_finite() && vel.linear.is_finite() && vel.angular.is_finite(),
        "convex-heightfield settle produced non-finite state: pose={pose:?}, vel={vel:?}"
    );
    assert!(
        y > half - 0.03,
        "convex hull tunneled below heightfield: center_y={y}, half={half}"
    );
    assert!(
        y < half + 0.12,
        "convex hull did not settle on heightfield: center_y={y}, half={half}"
    );
    assert!(
        vel.linear.length() < 0.30,
        "convex hull still moving too fast on heightfield: |v|={}",
        vel.linear.length()
    );
}

#[test]
fn sphere_heightfield_ramp_contact_has_expected_normal_and_depth() {
    let slope = 0.25_f64;
    let radius = 0.20_f64;
    let penetration = 0.05_f64;
    let surface_normal = V3::new(-slope, 1.0, 0.0).normalize();
    let sphere_center = surface_normal * (radius - penetration);

    let mut b = ModelBuilder::new();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: sphere_inertia(1.0, radius),
            })
            .with_position(sphere_center),
    );
    b.add_shape_sphere(body, radius, Transform::IDENTITY)
        .unwrap();

    // 3×3 samples covering x,z ∈ [-1, 1]. Height is y = slope * x,
    // giving a planar ramp with analytic normal `surface_normal`.
    let mut heights = Vec::new();
    for _j in 0..3 {
        for i in 0..3 {
            let x = -1.0 + i as f64;
            heights.push(slope * x);
        }
    }
    let hf = Heightfield {
        nx: 3,
        nz: 3,
        cell_size: 1.0,
        origin: V3::new(-1.0, 0.0, -1.0),
        heights,
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(BodyId::NONE, field_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();
    let mut contacts = Contacts::with_capacity(16);
    let mut pipeline = CollisionPipeline::new();

    pipeline.collide(&model, &state, &mut contacts).unwrap();
    let (depth, normal, point) = strongest_contact(&contacts).expect("expected ramp contact");

    // Contact normal points from the sphere (`a`) toward the heightfield
    // triangle (`b`), i.e. opposite the surface normal.
    assert!(
        normal.normalize().dot(-surface_normal) > 0.95,
        "heightfield normal {normal:?} did not match ramp normal {:?}",
        -surface_normal
    );
    assert!(
        (depth - penetration).abs() < 0.02,
        "heightfield depth {depth:.5} did not match expected penetration {penetration:.5}"
    );
    assert!(
        point.x.is_finite() && point.y.is_finite() && point.z.is_finite(),
        "heightfield contact point is non-finite: {point:?}"
    );
}

#[test]
fn box_sdf_sphere_contact_has_expected_normal_and_depth() {
    let half = V3::new(0.20, 0.12, 0.16);
    let sdf_radius = 1.0_f64;
    let signed_gap_to_box_center = 0.10_f64;
    let expected_depth = half.x - signed_gap_to_box_center;

    let mut b = ModelBuilder::new();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: box_inertia(1.0, half),
            })
            .with_position(V3::new(sdf_radius + signed_gap_to_box_center, 0.0, 0.0)),
    );
    b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
    let sdf_id = b.add_sdf(SdfShape::Sphere { r: sdf_radius });
    b.add_shape_sdf(BodyId::NONE, sdf_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();
    let mut contacts = Contacts::with_capacity(8);
    let mut pipeline = CollisionPipeline::new();

    pipeline.collide(&model, &state, &mut contacts).unwrap();
    let (depth, normal, point) = strongest_contact(&contacts).expect("expected box/SDF contact");

    assert!(
        (depth - expected_depth).abs() < 1.0e-9,
        "SDF depth {depth:.6} did not match expected {expected_depth:.6}"
    );
    assert!(
        normal.normalize().dot(-V3::X) > 1.0 - 1.0e-9,
        "box->SDF contact normal {normal:?} did not point inward toward the SDF sphere"
    );
    assert!(
        (point - V3::X * sdf_radius).length() < 1.0e-6,
        "SDF surface point {point:?} did not lie on the analytic sphere +X surface"
    );
}

#[test]
fn box_settles_on_sphere_sdf_without_tunneling() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.20);
    b.set_angular_damping(0.10);
    let sdf_radius = 0.50_f64;
    let half = V3::new(0.16, 0.10, 0.14);
    let mass = 1.0_f64;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia: box_inertia(mass, half),
            })
            .with_position(V3::new(0.0, 1.25, 0.0)),
    );
    b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
    let sdf_id = b.add_sdf(SdfShape::Sphere { r: sdf_radius });
    b.add_shape_sdf(BodyId::NONE, sdf_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(10);
    let dt = 0.005_f64;
    let mut max_contacts = 0_u32;
    for _ in 0..800 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        max_contacts = max_contacts.max(contacts.count);
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let pose = s_a.body_q.host().unwrap()[body.index()];
    let vel = s_a.body_qd.host().unwrap()[body.index()];
    let mut min_corner_gap = f64::INFINITY;
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let corner = pose.transform_point(V3::new(sx * half.x, sy * half.y, sz * half.z));
                min_corner_gap = min_corner_gap.min(corner.length() - sdf_radius);
            }
        }
    }
    let bottom_gap = pose.position.y - half.y - sdf_radius;
    eprintln!(
        "box on sphere-SDF: center={:?}, bottom_gap={bottom_gap:.5}, min_corner_gap={min_corner_gap:.5}, max_contacts={max_contacts}",
        pose.position
    );
    assert!(max_contacts > 0, "box never contacted the SDF sphere");
    assert!(
        pose.position.is_finite() && vel.linear.is_finite() && vel.angular.is_finite(),
        "box-SDF settle produced non-finite state: pose={pose:?}, vel={vel:?}"
    );
    assert!(
        min_corner_gap > -0.03,
        "box tunneled into analytic SDF sphere: min_corner_gap={min_corner_gap}"
    );
    assert!(
        bottom_gap.abs() < 0.08,
        "box did not settle near the top of the analytic SDF sphere: bottom_gap={bottom_gap}"
    );
    assert!(
        vel.linear.length() < 0.30 && vel.angular.length() < 0.80,
        "box still moving too fast on SDF sphere: |v|={}, |w|={}",
        vel.linear.length(),
        vel.angular.length()
    );
}

#[test]
fn sphere_settles_on_sphere_sdf_without_tunneling() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.08);
    let sdf_radius = 0.50_f64;
    let sphere_radius = 0.12_f64;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: sphere_inertia(1.0, sphere_radius),
            })
            .with_position(V3::new(0.0, 1.20, 0.0)),
    );
    b.add_shape_sphere(body, sphere_radius, Transform::IDENTITY)
        .unwrap();
    let sdf_id = b.add_sdf(SdfShape::Sphere { r: sdf_radius });
    b.add_shape_sdf(BodyId::NONE, sdf_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(32);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(10);
    let dt = 0.005_f64;
    let mut max_contacts = 0_u32;
    for _ in 0..500 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        max_contacts = max_contacts.max(contacts.count);
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let pose = s_a.body_q.host().unwrap()[body.index()];
    let vel = s_a.body_qd.host().unwrap()[body.index()];
    let center_dist = pose.position.length();
    let signed_gap = center_dist - sdf_radius - sphere_radius;
    eprintln!(
        "sphere on sphere-SDF: center={:?}, signed_gap={signed_gap:.5}, max_contacts={max_contacts}",
        pose.position
    );
    assert!(max_contacts > 0, "sphere never contacted the SDF sphere");
    assert!(
        pose.position.is_finite() && vel.linear.is_finite() && vel.angular.is_finite(),
        "sphere-SDF settle produced non-finite state: pose={pose:?}, vel={vel:?}"
    );
    assert!(
        signed_gap > -0.02,
        "sphere tunneled into analytic SDF sphere: signed_gap={signed_gap}"
    );
    assert!(
        signed_gap < 0.08,
        "sphere did not settle near analytic SDF surface: signed_gap={signed_gap}"
    );
    assert!(
        vel.linear.length() < 0.25,
        "sphere still moving too fast on SDF sphere: |v|={}",
        vel.linear.length()
    );
}

#[test]
fn sphere_settles_outside_torus_sdf() {
    let mut b = ModelBuilder::new();
    let r = 0.15_f64;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: sphere_inertia(1.0, r),
            })
            // Drop the sphere onto the top of a torus ring at y=1.
            .with_position(V3::new(0.5, 2.0, 0.0)),
    );
    b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
    // Torus: major=0.5, minor=0.1, sitting at world (0, 1, 0).
    let sdf_id = b.add_sdf(SdfShape::Torus {
        major: 0.5,
        minor: 0.1,
    });
    b.add_shape_sdf(
        BodyId::NONE,
        sdf_id,
        Transform::from_translation(V3::new(0.0, 1.0, 0.0)),
    )
    .unwrap();
    // Floor as a backstop.
    b.add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    let dt = 0.005_f64;
    for _ in 0..400 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let p = s_a.body_q.host().unwrap()[body.index()].position;
    eprintln!("sphere near torus settled at {p:?}");
    // Distance from sphere centre to torus surface = signed distance.
    // Want it ≥ -tol (i.e. NOT deeply inside the torus tube).
    let p_local = p - V3::new(0.0, 1.0, 0.0);
    let r_xz = (p_local.x * p_local.x + p_local.z * p_local.z).sqrt();
    let q = V3::new(r_xz - 0.5, p_local.y, 0.0);
    let signed = (q.x * q.x + q.y * q.y).sqrt() - 0.1;
    let pen = signed - r;
    assert!(
        pen > -0.02,
        "sphere penetrating torus by {} m at p={:?}",
        -pen,
        p
    );
}

#[test]
fn sdf_world_aabb_grows_with_torus_dims() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY,
    }));
    let sdf_id = b.add_sdf(SdfShape::Torus {
        major: 0.5,
        minor: 0.1,
    });
    b.add_shape_sdf(body, sdf_id, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();
    let aabb = shape_world_aabb(&model, &state, 0, 0.0);
    let extent = aabb.max - aabb.min;
    // Torus extent: 2·(major+minor) on x/z, 2·minor on y.
    assert!((extent.x - 2.0 * 0.6).abs() < 1e-9);
    assert!((extent.z - 2.0 * 0.6).abs() < 1e-9);
    assert!((extent.y - 2.0 * 0.1).abs() < 1e-9);
}

/// INSPECTION.md F15 / NutBoltSdf — the analytic helical-thread SDF.
///
/// Oracles:
/// 1. SCREW SYMMETRY: rotating by α about +Y while translating by
///    `pitch·α/τ` along +Y maps the thread onto itself, so the
///    distance field must be invariant (up to fp noise) for any α —
///    this is the property a stack of torus rings (the old proxy)
///    fails, since rings have no axial coupling.
/// 2. Surface points (profile-plane circle) sit at distance ≈ 0;
///    the helix core line sits at ≈ −minor; far points are positive.
/// 3. The cap plane truncates the thread.
#[test]
fn helical_thread_sdf_screw_symmetry_and_surface() {
    use molla_sim::SdfShape;
    let major = 0.30_f64;
    let minor = 0.05_f64;
    let pitch = 0.12_f64;
    let tau = std::f64::consts::TAU;
    let sdf = SdfShape::HelicalThread {
        major,
        minor,
        pitch,
        half_height: 1.0,
    };

    // 1. Screw symmetry.
    let probe = V3::new(0.27, 0.13, 0.11);
    let d0 = sdf.signed_distance(probe);
    for k in 1..24 {
        let alpha = k as f64 * 0.261799; // 15° steps
        let (s, c) = alpha.sin_cos();
        let rotated = V3::new(
            c * probe.x - s * probe.z,
            probe.y + pitch * alpha / tau,
            s * probe.x + c * probe.z,
        );
        let d = sdf.signed_distance(rotated);
        assert!(
            (d - d0).abs() < 1.0e-9,
            "screw symmetry broken at α={alpha:.3}: {d} vs {d0}"
        );
    }

    // 2. Surface + core: at angle θ the thread crest passes through
    //    y = pitch·θ/τ; the swept circle in the profile plane is
    //    (r, y) = (major + minor·cosφ, crest_y + minor·sinφ).
    for k in 0..16 {
        let theta = k as f64 * tau / 16.0;
        let crest_y = pitch * theta / tau;
        let core = V3::new(major * theta.cos(), crest_y, major * theta.sin());
        let d_core = sdf.signed_distance(core);
        assert!(
            (d_core + minor).abs() < 2.0e-3,
            "helix core at θ={theta:.2} should be ≈ −minor: {d_core}"
        );
        for j in 0..8 {
            let phi = j as f64 * tau / 8.0;
            let r = major + minor * phi.cos();
            let y = crest_y + minor * phi.sin();
            let p = V3::new(r * theta.cos(), y, r * theta.sin());
            let d = sdf.signed_distance(p);
            assert!(
                d.abs() < 2.0e-3,
                "surface point θ={theta:.2} φ={phi:.2} should be ≈ 0: {d}"
            );
        }
    }
    // Far point positive, capped region positive.
    assert!(sdf.signed_distance(V3::new(1.0, 0.0, 0.0)) > 0.5);
    assert!(sdf.signed_distance(V3::new(major, 2.0, 0.0)) > 0.5);
}
