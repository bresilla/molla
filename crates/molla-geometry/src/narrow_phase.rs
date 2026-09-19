//! Narrow phase: produces real contacts from candidate shape pairs.
//!
//! Hand-rolled tests for the cheap pairs (sphere-sphere, sphere-plane,
//! sphere-box, box-plane, box-box via SAT); capsule / cylinder /
//! convex-mesh fall through to GJK + EPA ([`crate::gjk`], [`crate::epa`])
//! and triangle meshes use the BVH in [`crate::mesh_bvh`].

use molla_math::{Mat3, Real, Transform, Vec3};
use molla_sim::{Model, ShapeGeometry, State};

use crate::broad_phase::ShapePair;
use crate::epa::epa_with_hulls;
use crate::gjk::{GjkResult, gjk_with_hulls};
use crate::mesh_bvh::MeshBvh;
use crate::mesh_cache::MeshBvhCache;
use crate::shape_world::shape_world_transform;
use crate::support::support_ellipsoid;

/// Output of the narrow phase. The contact normal points from `a` to
/// `b`; `depth > 0` means penetration. Caller is responsible for
/// pushing this into a `Contacts` buffer.
#[derive(Copy, Clone, Debug)]
pub struct ContactManifold {
    pub a_shape: u32,
    pub b_shape: u32,
    /// World-space contact point (mid-plane between the two surfaces).
    pub point: Vec3,
    /// Unit normal, points from `a` toward `b` (so we can correct
    /// `b += depth * normal` and `a -= depth * normal`).
    pub normal: Vec3,
    /// Penetration depth (m). Always non-negative on output.
    pub depth: Real,
    pub friction: Real,
    /// Rolling friction (μ_rolling) — opposes angular velocity
    /// component perpendicular to the normal.
    pub rolling_friction: Real,
    /// Torsional friction (μ_torsional) — opposes angular velocity
    /// component along the normal (spinning).
    pub torsional_friction: Real,
    pub restitution: Real,
}

/// Maximum manifold points returned per pair. Box-box / box-plane can
/// produce up to 4 (the deepest face-incident vertices).
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// Generate contacts for one candidate pair. Pushes 0..N manifolds
/// onto `out`. Stage 4 supports sphere-sphere, sphere-plane,
/// sphere-box, box-plane, box-box (SAT).
pub fn pair_contact(
    model: &Model,
    state: &State,
    pair: ShapePair,
    _default_friction: Real,
    default_restitution: Real,
    out: &mut Vec<ContactManifold>,
) {
    pair_contact_cached(model, state, pair, _default_friction, default_restitution, out, None);
}

pub(crate) fn pair_contact_cached(
    model: &Model,
    state: &State,
    pair: ShapePair,
    _default_friction: Real,
    default_restitution: Real,
    out: &mut Vec<ContactManifold>,
    mesh_cache: Option<&mut MeshBvhCache>,
) {
    let geom_a = model.shape_geometry[pair.a as usize];
    let geom_b = model.shape_geometry[pair.b as usize];
    let xf_a = shape_world_transform(model, state, pair.a as usize);
    let xf_b = shape_world_transform(model, state, pair.b as usize);

    let pair_friction = model
        .combined_shape_friction(pair.a as usize, pair.b as usize)
        .expect("shape friction material");
    let pair_restitution = pair_material(
        model.shape_restitution.host().expect("shape_restitution"),
        pair.a as usize,
        pair.b as usize,
        default_restitution,
    );
    // Rolling / torsional friction default to 0 — opt-in per shape.
    let pair_rolling = pair_material(
        model
            .shape_rolling_friction
            .host()
            .expect("shape_rolling_friction"),
        pair.a as usize,
        pair.b as usize,
        0.0,
    );
    let pair_torsional = pair_material(
        model
            .shape_torsional_friction
            .host()
            .expect("shape_torsional_friction"),
        pair.a as usize,
        pair.b as usize,
        0.0,
    );

    let start = out.len();
    match (geom_a, geom_b) {
        (ShapeGeometry::Sphere { radius: ra }, ShapeGeometry::Sphere { radius: rb }) => {
            if let Some(m) = sphere_sphere(xf_a.position, ra, xf_b.position, rb, pair.a, pair.b) {
                out.push(m);
            }
        }
        (ShapeGeometry::Sphere { radius }, ShapeGeometry::Plane { normal }) => {
            if let Some(m) = sphere_plane(
                xf_a.position,
                radius,
                xf_b.position,
                xf_b.rotation * normal,
                pair.a,
                pair.b,
            ) {
                out.push(m);
            }
        }
        (ShapeGeometry::Plane { normal }, ShapeGeometry::Sphere { radius }) => {
            if let Some(m) = sphere_plane(
                xf_b.position,
                radius,
                xf_a.position,
                xf_a.rotation * normal,
                pair.b,
                pair.a,
            ) {
                out.push(flip_manifold(m));
            }
        }
        (ShapeGeometry::Sphere { radius }, ShapeGeometry::Box { half_extents }) => {
            if let Some(m) = sphere_box(xf_a.position, radius, &xf_b, half_extents, pair.a, pair.b)
            {
                out.push(m);
            }
        }
        (ShapeGeometry::Box { half_extents }, ShapeGeometry::Sphere { radius }) => {
            if let Some(m) = sphere_box(xf_b.position, radius, &xf_a, half_extents, pair.b, pair.a)
            {
                out.push(flip_manifold(m));
            }
        }
        (ShapeGeometry::Box { half_extents }, ShapeGeometry::Plane { normal }) => {
            box_plane(
                &xf_a,
                half_extents,
                xf_b.position,
                xf_b.rotation * normal,
                pair.a,
                pair.b,
                out,
            );
        }
        (ShapeGeometry::Plane { normal }, ShapeGeometry::Box { half_extents }) => {
            let n0 = out.len();
            box_plane(
                &xf_b,
                half_extents,
                xf_a.position,
                xf_a.rotation * normal,
                pair.b,
                pair.a,
                out,
            );
            for m in &mut out[n0..] {
                *m = flip_manifold(*m);
            }
        }
        (ShapeGeometry::Box { half_extents: ha }, ShapeGeometry::Box { half_extents: hb }) => {
            box_box_sat(&xf_a, ha, &xf_b, hb, pair.a, pair.b, out);
        }

        // ── Capsule fast paths (closed-form) ─────────────────────
        (
            ShapeGeometry::Sphere { radius: rs },
            ShapeGeometry::Capsule {
                half_height: hh,
                radius: rc,
            },
        ) => {
            if let Some(m) = sphere_capsule(xf_a.position, rs, &xf_b, hh, rc, pair.a, pair.b) {
                out.push(m);
            }
        }
        (
            ShapeGeometry::Capsule {
                half_height: hh,
                radius: rc,
            },
            ShapeGeometry::Sphere { radius: rs },
        ) => {
            if let Some(m) = sphere_capsule(xf_b.position, rs, &xf_a, hh, rc, pair.b, pair.a) {
                out.push(flip_manifold(m));
            }
        }
        (
            ShapeGeometry::Capsule {
                half_height: hh,
                radius: rc,
            },
            ShapeGeometry::Plane { normal },
        ) => {
            capsule_plane(
                &xf_a,
                hh,
                rc,
                xf_b.position,
                xf_b.rotation * normal,
                pair.a,
                pair.b,
                out,
            );
        }
        (
            ShapeGeometry::Plane { normal },
            ShapeGeometry::Capsule {
                half_height: hh,
                radius: rc,
            },
        ) => {
            let n0 = out.len();
            capsule_plane(
                &xf_b,
                hh,
                rc,
                xf_a.position,
                xf_a.rotation * normal,
                pair.b,
                pair.a,
                out,
            );
            for m in &mut out[n0..] {
                *m = flip_manifold(*m);
            }
        }
        (
            ShapeGeometry::Capsule {
                half_height: ha,
                radius: ra,
            },
            ShapeGeometry::Capsule {
                half_height: hb,
                radius: rb,
            },
        ) => {
            if let Some(m) = capsule_capsule(&xf_a, ha, ra, &xf_b, hb, rb, pair.a, pair.b) {
                out.push(m);
            }
        }

        // ── Ellipsoid fast paths ─────────────────────────────────
        // Ellipsoid-vs-Sphere and Ellipsoid-vs-Plane are cheap
        // closed-form (single Newton iteration / argmax-support).
        // Ellipsoid-vs-Box / -Capsule / -Cylinder / -Cone / -Ellipsoid
        // route through the GJK + EPA arm below (ellipsoid has a
        // perfectly well-behaved support function).
        (ShapeGeometry::Sphere { radius }, ShapeGeometry::Ellipsoid { half_axes }) => {
            if let Some(m) =
                ellipsoid_sphere(&xf_b, half_axes, xf_a.position, radius, pair.b, pair.a)
            {
                out.push(flip_manifold(m));
            }
        }
        (ShapeGeometry::Ellipsoid { half_axes }, ShapeGeometry::Sphere { radius }) => {
            if let Some(m) =
                ellipsoid_sphere(&xf_a, half_axes, xf_b.position, radius, pair.a, pair.b)
            {
                out.push(m);
            }
        }
        (ShapeGeometry::Ellipsoid { half_axes }, ShapeGeometry::Plane { normal }) => {
            if let Some(m) = ellipsoid_plane(
                &xf_a,
                half_axes,
                xf_b.position,
                xf_b.rotation * normal,
                pair.a,
                pair.b,
            ) {
                out.push(m);
            }
        }
        (ShapeGeometry::Plane { normal }, ShapeGeometry::Ellipsoid { half_axes }) => {
            if let Some(m) = ellipsoid_plane(
                &xf_b,
                half_axes,
                xf_a.position,
                xf_a.rotation * normal,
                pair.b,
                pair.a,
            ) {
                out.push(flip_manifold(m));
            }
        }

        // ── GJK + EPA fallback (no closed-form path) ─────────────
        // Plane is a half-space — GJK isn't well-defined against it
        // (unbounded support). Pair-vs-plane uses a ray-test fast
        // path (only the 'lowest' support point matters).
        (ShapeGeometry::Cylinder { .. }, ShapeGeometry::Plane { .. })
        | (ShapeGeometry::Plane { .. }, ShapeGeometry::Cylinder { .. })
        | (ShapeGeometry::RoundCylinder { .. }, ShapeGeometry::Plane { .. })
        | (ShapeGeometry::Plane { .. }, ShapeGeometry::RoundCylinder { .. }) => {
            cylinder_plane_fallback(geom_a, &xf_a, geom_b, &xf_b, pair, out);
        }
        (ShapeGeometry::Cone { .. }, ShapeGeometry::Plane { .. })
        | (ShapeGeometry::Plane { .. }, ShapeGeometry::Cone { .. }) => {
            cone_plane_fallback(geom_a, &xf_a, geom_b, &xf_b, pair, out);
        }
        (ShapeGeometry::ConvexHull { .. }, ShapeGeometry::Plane { .. })
        | (ShapeGeometry::Plane { .. }, ShapeGeometry::ConvexHull { .. }) => {
            hull_plane_fallback(model, geom_a, &xf_a, geom_b, &xf_b, pair, out);
        }

        // ── TriMesh / Heightfield / SDF dispatched FIRST so they
        // catch any pair involving them, regardless of the other
        // shape's kind. (Putting the convex-vs-convex GJK arm above
        // these would route e.g. Cone-vs-Heightfield into GJK with a
        // broken Heightfield support function.)
        (ShapeGeometry::TriMesh { mesh_id }, _) => {
            trimesh_pair(model, mesh_id, &xf_a, geom_b, &xf_b, pair, false, out, mesh_cache);
        }
        (_, ShapeGeometry::TriMesh { mesh_id }) => {
            trimesh_pair(model, mesh_id, &xf_b, geom_a, &xf_a, pair, true, out, mesh_cache);
        }
        (ShapeGeometry::Heightfield { field_id }, _) => {
            heightfield_pair(model, field_id, &xf_a, geom_b, &xf_b, pair, false, out);
        }
        (_, ShapeGeometry::Heightfield { field_id }) => {
            heightfield_pair(model, field_id, &xf_b, geom_a, &xf_a, pair, true, out);
        }
        (ShapeGeometry::Sdf { sdf_id }, _) => {
            sdf_pair(model, sdf_id, &xf_a, geom_b, &xf_b, pair, false, out);
        }
        (_, ShapeGeometry::Sdf { sdf_id }) => {
            sdf_pair(model, sdf_id, &xf_b, geom_a, &xf_a, pair, true, out);
        }

        // Convex-vs-convex via GJK + EPA. Covers capsule-box,
        // box-capsule, capsule-cylinder, cylinder-*, triangle-*,
        // cone-*, hull-*, ellipsoid-{box, capsule, cylinder,
        // ellipsoid}.
        (ShapeGeometry::Capsule { .. }, ShapeGeometry::Box { .. })
        | (ShapeGeometry::Box { .. }, ShapeGeometry::Capsule { .. })
        | (ShapeGeometry::Capsule { .. }, ShapeGeometry::Cylinder { .. })
        | (ShapeGeometry::Cylinder { .. }, ShapeGeometry::Capsule { .. })
        | (ShapeGeometry::Cylinder { .. }, ShapeGeometry::Cylinder { .. })
        | (ShapeGeometry::Cylinder { .. }, ShapeGeometry::Sphere { .. })
        | (ShapeGeometry::Sphere { .. }, ShapeGeometry::Cylinder { .. })
        | (ShapeGeometry::Cylinder { .. }, ShapeGeometry::Box { .. })
        | (ShapeGeometry::Box { .. }, ShapeGeometry::Cylinder { .. })
        | (ShapeGeometry::Triangle { .. }, _)
        | (_, ShapeGeometry::Triangle { .. })
        | (ShapeGeometry::Cone { .. }, _)
        | (_, ShapeGeometry::Cone { .. })
        | (ShapeGeometry::ConvexHull { .. }, _)
        | (_, ShapeGeometry::ConvexHull { .. })
        | (ShapeGeometry::Ellipsoid { .. }, _)
        | (_, ShapeGeometry::Ellipsoid { .. })
        | (ShapeGeometry::RoundCylinder { .. }, _)
        | (_, ShapeGeometry::RoundCylinder { .. }) => {
            if matches!(geom_a, ShapeGeometry::Plane { .. })
                || matches!(geom_b, ShapeGeometry::Plane { .. })
            {
                // Plane fallback dispatched separately below.
            } else if let Some(m) =
                gjk_epa_pair_with_hulls(geom_a, &xf_a, geom_b, &xf_b, pair, &model.hulls)
            {
                out.push(m);
            }
        }

        (ShapeGeometry::Plane { .. }, ShapeGeometry::Plane { .. }) => {}
    }

    for m in &mut out[start..] {
        m.friction = pair_friction;
        m.rolling_friction = pair_rolling;
        m.torsional_friction = pair_torsional;
        m.restitution = pair_restitution;
    }
}

/// Pair-rule for per-shape materials: take `max` of the two. Per-shape
/// material arrays are guaranteed finite at this point: the builder
/// substitutes its NaN "use default" sentinels with the configured
/// defaults at finalize time, and the public setters reject NaN.
/// `default` is only used when a shape index is out of range (which
/// itself indicates an upstream bug).
fn pair_material(arr: &[Real], a: usize, b: usize, default: Real) -> Real {
    let ma = arr.get(a).copied().unwrap_or(default);
    let mb = arr.get(b).copied().unwrap_or(default);
    debug_assert!(
        ma.is_finite(),
        "pair_material: shape {a} material is not finite ({ma})"
    );
    debug_assert!(
        mb.is_finite(),
        "pair_material: shape {b} material is not finite ({mb})"
    );
    ma.max(mb)
}

// ─── Sphere-sphere ───────────────────────────────────────────────────

fn sphere_sphere(
    pa: Vec3,
    ra: Real,
    pb: Vec3,
    rb: Real,
    a_shape: u32,
    b_shape: u32,
) -> Option<ContactManifold> {
    let delta = pb - pa;
    let d_sq = delta.length_squared();
    let r_sum = ra + rb;
    if d_sq >= r_sum * r_sum {
        return None;
    }
    let dist = d_sq.sqrt();
    let normal = if dist > 1e-12 { delta / dist } else { Vec3::Y };
    let depth = r_sum - dist;
    let point = pa + normal * (ra - 0.5 * depth);
    Some(ContactManifold {
        a_shape,
        b_shape,
        point,
        normal,
        depth,
        friction: 0.0,
        rolling_friction: 0.0,
        torsional_friction: 0.0,
        restitution: 0.0,
    })
}

// ─── Sphere-plane ────────────────────────────────────────────────────

fn sphere_plane(
    sphere_p: Vec3,
    radius: Real,
    plane_p: Vec3,
    plane_normal: Vec3,
    sphere_shape: u32,
    plane_shape: u32,
) -> Option<ContactManifold> {
    let signed = (sphere_p - plane_p).dot(plane_normal);
    if signed >= radius {
        return None;
    }
    let depth = radius - signed;
    // Normal points from sphere toward plane (corrects sphere out of plane).
    let normal = -plane_normal;
    let point = sphere_p + normal * (radius - 0.5 * depth);
    Some(ContactManifold {
        a_shape: sphere_shape,
        b_shape: plane_shape,
        point,
        normal,
        depth,
        friction: 0.0,
        rolling_friction: 0.0,
        torsional_friction: 0.0,
        restitution: 0.0,
    })
}

// ─── Sphere-box ──────────────────────────────────────────────────────

fn sphere_box(
    sphere_p: Vec3,
    radius: Real,
    box_xf: &Transform,
    box_half: Vec3,
    sphere_shape: u32,
    box_shape: u32,
) -> Option<ContactManifold> {
    // Express sphere centre in box-local frame.
    let p_local = box_xf.inverse_transform_point(sphere_p);
    // Closest point on the box to the sphere centre, box-local.
    let q_local = Vec3::new(
        p_local.x.clamp(-box_half.x, box_half.x),
        p_local.y.clamp(-box_half.y, box_half.y),
        p_local.z.clamp(-box_half.z, box_half.z),
    );
    let delta_local = p_local - q_local;
    let d_sq = delta_local.length_squared();

    if d_sq > radius * radius {
        return None;
    }

    let (normal_local, depth, contact_local) = if d_sq > 1e-24 {
        // Sphere centre outside the box.
        let dist = d_sq.sqrt();
        let n_local = delta_local / dist; // points from box → sphere
        let depth = radius - dist;
        let contact = q_local + n_local * (0.5 * (depth));
        (-n_local, depth, contact) // normal a→b: sphere(a) → box(b) = −n_local
    } else {
        // Sphere centre inside the box — pick the axis with smallest
        // distance to a face and push out along it.
        let dx = box_half.x - p_local.x.abs();
        let dy = box_half.y - p_local.y.abs();
        let dz = box_half.z - p_local.z.abs();
        let (axis_local, depth) = if dx <= dy && dx <= dz {
            (Vec3::new(p_local.x.signum(), 0.0, 0.0), radius + dx)
        } else if dy <= dz {
            (Vec3::new(0.0, p_local.y.signum(), 0.0), radius + dy)
        } else {
            (Vec3::new(0.0, 0.0, p_local.z.signum()), radius + dz)
        };
        // axis_local points from box centre toward the closest face.
        // Sphere centre is *inside* the box, normal sphere→box = -axis_local.
        let contact = p_local; // sphere centre — half-buried
        (-axis_local, depth, contact)
    };

    let point_world = box_xf.transform_point(contact_local);
    let normal_world = box_xf.transform_vector(normal_local);
    Some(ContactManifold {
        a_shape: sphere_shape,
        b_shape: box_shape,
        point: point_world,
        normal: normal_world,
        depth,
        friction: 0.0,
        rolling_friction: 0.0,
        torsional_friction: 0.0,
        restitution: 0.0,
    })
}

// ─── Box-plane ───────────────────────────────────────────────────────

fn box_plane(
    box_xf: &Transform,
    box_half: Vec3,
    plane_p: Vec3,
    plane_normal: Vec3,
    box_shape: u32,
    plane_shape: u32,
    out: &mut Vec<ContactManifold>,
) {
    // Test all 8 corners of the box. Each corner that's below the
    // plane is a contact point. Convention: normal points a→b. With
    // a = box, b = plane, the normal is along -plane_normal (toward
    // the plane). XPBD then pushes the box along +plane_normal.
    let corners = box_corners_world(box_xf, box_half);
    let normal = -plane_normal;
    let start = out.len();
    for c in corners {
        let signed = (c - plane_p).dot(plane_normal);
        if signed < 0.0 {
            let depth = -signed;
            let point = c + plane_normal * (0.5 * depth);
            out.push(ContactManifold {
                a_shape: box_shape,
                b_shape: plane_shape,
                point,
                normal,
                depth,
                friction: 0.0,
                rolling_friction: 0.0,
                torsional_friction: 0.0,
                restitution: 0.0,
            });
        }
    }
    if out.len() - start > MAX_MANIFOLD_POINTS {
        keep_deepest_range(out, start, MAX_MANIFOLD_POINTS);
    }
}

fn box_corners_world(xf: &Transform, h: Vec3) -> [Vec3; 8] {
    let r = Mat3::from_quat(xf.rotation);
    let cx = r.x_axis * h.x;
    let cy = r.y_axis * h.y;
    let cz = r.z_axis * h.z;
    let p = xf.position;
    [
        p - cx - cy - cz,
        p + cx - cy - cz,
        p - cx + cy - cz,
        p + cx + cy - cz,
        p - cx - cy + cz,
        p + cx - cy + cz,
        p - cx + cy + cz,
        p + cx + cy + cz,
    ]
}

fn keep_deepest_range(out: &mut Vec<ContactManifold>, start: usize, k: usize) {
    let n = out.len();
    if n - start <= k {
        return;
    }
    out[start..].sort_by(|a, b| {
        b.depth
            .partial_cmp(&a.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(start + k);
}

// ─── Box-box (SAT) ───────────────────────────────────────────────────

fn box_box_sat(
    a_xf: &Transform,
    a_half: Vec3,
    b_xf: &Transform,
    b_half: Vec3,
    a_shape: u32,
    b_shape: u32,
    out: &mut Vec<ContactManifold>,
) {
    // Reference: Erin Catto's box-box from Bullet / Box2D, simplified.
    // Goal: find the SAT axis with the smallest penetration depth, then
    // emit a contact manifold (1–4 points).
    let ra = Mat3::from_quat(a_xf.rotation);
    let rb = Mat3::from_quat(b_xf.rotation);
    let a_axes = [ra.x_axis, ra.y_axis, ra.z_axis];
    let b_axes = [rb.x_axis, rb.y_axis, rb.z_axis];

    let t = b_xf.position - a_xf.position;

    // R[i][j] = a_axes[i] · b_axes[j], absR with epsilon.
    let mut rmat = [[0.0; 3]; 3];
    let mut absr = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rmat[i][j] = a_axes[i].dot(b_axes[j]);
            absr[i][j] = rmat[i][j].abs() + 1e-12;
        }
    }

    let a_h = [a_half.x, a_half.y, a_half.z];
    let b_h = [b_half.x, b_half.y, b_half.z];
    let t_in_a = [t.dot(a_axes[0]), t.dot(a_axes[1]), t.dot(a_axes[2])];

    // Track best axis: 0..3 = a's faces, 3..6 = b's faces, 6..15 = edge-edge.
    let mut best_depth = Real::INFINITY;
    let mut best_axis = -1i32;
    let mut best_normal = Vec3::Y; // points from a → b

    // Faces of A.
    for i in 0..3 {
        let ra_proj = a_h[i];
        let rb_proj = b_h[0] * absr[i][0] + b_h[1] * absr[i][1] + b_h[2] * absr[i][2];
        let dist = t_in_a[i].abs();
        let pen = ra_proj + rb_proj - dist;
        if pen < 0.0 {
            return; // separating axis — no contact
        }
        if pen < best_depth {
            best_depth = pen;
            best_axis = i as i32;
            let s = if t_in_a[i] >= 0.0 { 1.0 } else { -1.0 };
            best_normal = a_axes[i] * s;
        }
    }

    // Faces of B.
    let t_in_b = [t.dot(b_axes[0]), t.dot(b_axes[1]), t.dot(b_axes[2])];
    for j in 0..3 {
        let ra_proj = a_h[0] * absr[0][j] + a_h[1] * absr[1][j] + a_h[2] * absr[2][j];
        let rb_proj = b_h[j];
        let dist = t_in_b[j].abs();
        let pen = ra_proj + rb_proj - dist;
        if pen < 0.0 {
            return;
        }
        // Bias toward face axes by a hair to prefer them over edge-edge
        // (edge-edge contacts produce 1 point, faces produce 4 — much
        // more stable for stacking).
        if pen + 1e-3 < best_depth {
            best_depth = pen;
            best_axis = (3 + j) as i32;
            let s = if t_in_b[j] >= 0.0 { 1.0 } else { -1.0 };
            best_normal = b_axes[j] * s;
        }
    }

    // Edge-edge axes (a_i × b_j).
    for i in 0..3 {
        for j in 0..3 {
            let axis = a_axes[i].cross(b_axes[j]);
            let axis_len = axis.length();
            if axis_len < 1e-9 {
                continue; // parallel edges — covered by face axes
            }
            let n = axis / axis_len;
            // Project A and B half-extents onto this axis.
            let mut ra_proj: Real = 0.0;
            for k in 0..3 {
                ra_proj += (a_h[k] * a_axes[k].dot(n)).abs();
            }
            let mut rb_proj: Real = 0.0;
            for k in 0..3 {
                rb_proj += (b_h[k] * b_axes[k].dot(n)).abs();
            }
            let dist = t.dot(n).abs();
            let pen = ra_proj + rb_proj - dist;
            if pen < 0.0 {
                return;
            }
            if pen + 5e-3 < best_depth {
                best_depth = pen;
                best_axis = (6 + i * 3 + j) as i32;
                let s = if t.dot(n) >= 0.0 { 1.0 } else { -1.0 };
                best_normal = n * s;
            }
        }
    }

    if best_axis < 0 || best_depth <= 0.0 {
        return;
    }

    // Manifold generation: clip B's incident face against A's reference
    // face for axis < 6 (face contact), or use closest points for
    // edge-edge axes (axis >= 6).
    if best_axis < 6 {
        emit_face_manifold(
            a_xf,
            a_half,
            &a_axes,
            b_xf,
            b_half,
            &b_axes,
            best_normal,
            best_depth,
            a_shape,
            b_shape,
            best_axis,
            out,
        );
    } else {
        emit_edge_manifold(
            a_xf,
            a_half,
            &a_axes,
            b_xf,
            b_half,
            &b_axes,
            best_normal,
            best_depth,
            a_shape,
            b_shape,
            best_axis - 6,
            out,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_face_manifold(
    a_xf: &Transform,
    a_half: Vec3,
    a_axes: &[Vec3; 3],
    b_xf: &Transform,
    b_half: Vec3,
    b_axes: &[Vec3; 3],
    normal: Vec3, // a → b
    depth: Real,
    a_shape: u32,
    b_shape: u32,
    axis_idx: i32,
    out: &mut Vec<ContactManifold>,
) {
    // Reference body: the one whose face is the SAT separating axis.
    // - axis_idx 0..3: A's face is reference (normal == ±a_axes[i]).
    // - axis_idx 3..6: B's face is reference; we want normal pointing
    //   FROM the reference body, so flip.
    // We always emit the manifold in the convention "normal points a→b".

    let (ref_xf, ref_half, ref_axes, inc_xf, inc_half, inc_axes, ref_is_a) = if axis_idx < 3 {
        (a_xf, a_half, a_axes, b_xf, b_half, b_axes, true)
    } else {
        (b_xf, b_half, b_axes, a_xf, a_half, a_axes, false)
    };

    // Reference face: pick the face of `ref` whose outward normal best
    // aligns with `normal` (if ref_is_a) or `-normal` (if !ref_is_a).
    // We want the outward face that points TOWARD the incident body.
    let ref_face_n = if ref_is_a { normal } else { -normal };
    let mut best_dot = -Real::INFINITY;
    let mut ref_axis_idx = 0usize;
    let mut ref_axis_sign = 1.0;
    for k in 0..3 {
        for s in [-1.0, 1.0] {
            let candidate = ref_axes[k] * s;
            let d = candidate.dot(ref_face_n);
            if d > best_dot {
                best_dot = d;
                ref_axis_idx = k;
                ref_axis_sign = s;
            }
        }
    }
    let ref_n = ref_axes[ref_axis_idx] * ref_axis_sign;
    let ref_h = [ref_half.x, ref_half.y, ref_half.z];
    let ref_face_centre = ref_xf.position + ref_n * ref_h[ref_axis_idx];
    // Two in-plane axes for the reference face.
    let (u_idx, v_idx) = match ref_axis_idx {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let ref_u = ref_axes[u_idx];
    let ref_v = ref_axes[v_idx];
    let ref_hu = ref_h[u_idx];
    let ref_hv = ref_h[v_idx];

    // Incident face: pick the face of `inc` whose outward normal is
    // most opposite to `ref_n`.
    let mut best = Real::INFINITY;
    let mut inc_axis_idx = 0usize;
    let mut inc_axis_sign = 1.0;
    for k in 0..3 {
        for s in [-1.0, 1.0] {
            let candidate = inc_axes[k] * s;
            let d = candidate.dot(ref_n);
            if d < best {
                best = d;
                inc_axis_idx = k;
                inc_axis_sign = s;
            }
        }
    }
    let inc_n = inc_axes[inc_axis_idx] * inc_axis_sign;
    let inc_h = [inc_half.x, inc_half.y, inc_half.z];
    let inc_face_centre = inc_xf.position + inc_n * inc_h[inc_axis_idx];
    let (iu_idx, iv_idx) = match inc_axis_idx {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let inc_u = inc_axes[iu_idx] * inc_h[iu_idx];
    let inc_v = inc_axes[iv_idx] * inc_h[iv_idx];

    // Incident face quad in world space (4 corners).
    let mut quad = [
        inc_face_centre - inc_u - inc_v,
        inc_face_centre + inc_u - inc_v,
        inc_face_centre + inc_u + inc_v,
        inc_face_centre - inc_u + inc_v,
    ];

    // Clip the incident quad against the 4 side planes of the reference face.
    let clip_planes = [
        (ref_u, ref_face_centre - ref_u * ref_hu), // -u side, inward normal +u
        (-ref_u, ref_face_centre + ref_u * ref_hu), // +u side, inward normal -u
        (ref_v, ref_face_centre - ref_v * ref_hv), // -v side
        (-ref_v, ref_face_centre + ref_v * ref_hv), // +v side
    ];
    let mut clipped: Vec<Vec3> = quad.to_vec();
    for (n, p) in clip_planes.iter() {
        clipped = clip_polygon_to_plane(&clipped, *n, *p);
        if clipped.is_empty() {
            break;
        }
    }
    // Avoid the unused-binding warning when clip is bypassed.
    let _ = &mut quad;

    // For each survivor, project onto reference plane; keep points
    // whose signed distance below the ref face is positive (i.e.
    // penetrating). Emit (point, depth).
    let mut candidates: Vec<(Vec3, Real)> = Vec::new();
    for p in clipped {
        let signed = (p - ref_face_centre).dot(ref_n);
        if signed <= 0.0 {
            // Penetration depth (positive).
            let d = -signed;
            // Contact point: midway between the incident point and its
            // projection on the reference face.
            let mid = p - ref_n * (0.5 * d);
            candidates.push((mid, d));
        }
    }

    if candidates.is_empty() {
        // Fallback to the SAT-reported depth at the incident face centre.
        let mid = inc_face_centre - ref_n * (0.5 * depth);
        candidates.push((mid, depth));
    }

    // Reduce to <=4 points: keep the deepest, then the 3 most extreme
    // along the in-plane axes.
    if candidates.len() > MAX_MANIFOLD_POINTS {
        candidates = reduce_manifold(&candidates, ref_face_centre, ref_u, ref_v);
    }

    for (point, d) in candidates {
        out.push(ContactManifold {
            a_shape,
            b_shape,
            point,
            normal,
            depth: d,
            friction: 0.0,
            rolling_friction: 0.0,
            torsional_friction: 0.0,
            restitution: 0.0,
        });
    }
}

fn clip_polygon_to_plane(poly: &[Vec3], plane_normal: Vec3, plane_point: Vec3) -> Vec<Vec3> {
    if poly.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(poly.len() + 1);
    let mut prev = poly[poly.len() - 1];
    let mut prev_d = (prev - plane_point).dot(plane_normal);
    for &cur in poly.iter() {
        let cur_d = (cur - plane_point).dot(plane_normal);
        if cur_d >= 0.0 {
            if prev_d < 0.0 {
                let t = prev_d / (prev_d - cur_d);
                out.push(prev + (cur - prev) * t);
            }
            out.push(cur);
        } else if prev_d >= 0.0 {
            let t = prev_d / (prev_d - cur_d);
            out.push(prev + (cur - prev) * t);
        }
        prev = cur;
        prev_d = cur_d;
    }
    out
}

fn reduce_manifold(pts: &[(Vec3, Real)], centre: Vec3, u: Vec3, v: Vec3) -> Vec<(Vec3, Real)> {
    // Pick the deepest point first. Then iteratively pick the point
    // furthest from the already-picked set in the (u, v) plane.
    let mut chosen: Vec<usize> = Vec::with_capacity(MAX_MANIFOLD_POINTS);
    let deepest = pts
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    chosen.push(deepest);
    while chosen.len() < MAX_MANIFOLD_POINTS && chosen.len() < pts.len() {
        let mut best = -Real::INFINITY;
        let mut best_idx = 0usize;
        for (i, (p, _)) in pts.iter().enumerate() {
            if chosen.contains(&i) {
                continue;
            }
            let pi = *p - centre;
            let pu = pi.dot(u);
            let pv = pi.dot(v);
            let mut min_d = Real::INFINITY;
            for &c in &chosen {
                let qi = pts[c].0 - centre;
                let qu = qi.dot(u);
                let qv = qi.dot(v);
                let d = (pu - qu).powi(2) + (pv - qv).powi(2);
                if d < min_d {
                    min_d = d;
                }
            }
            if min_d > best {
                best = min_d;
                best_idx = i;
            }
        }
        chosen.push(best_idx);
    }
    chosen.into_iter().map(|i| pts[i]).collect()
}

#[allow(clippy::too_many_arguments)]
fn emit_edge_manifold(
    a_xf: &Transform,
    a_half: Vec3,
    a_axes: &[Vec3; 3],
    b_xf: &Transform,
    b_half: Vec3,
    b_axes: &[Vec3; 3],
    normal: Vec3, // a → b
    depth: Real,
    a_shape: u32,
    b_shape: u32,
    axis_pair: i32,
    out: &mut Vec<ContactManifold>,
) {
    // axis_pair = i*3 + j → A's edge along a_axes[i], B's edge along b_axes[j].
    let i = (axis_pair / 3) as usize;
    let j = (axis_pair % 3) as usize;
    // Pick edge endpoints on A: the edge most aligned with `+normal`
    // (i.e. closest to B). Same idea on B but with `-normal`.
    let a_h = [a_half.x, a_half.y, a_half.z];
    let b_h = [b_half.x, b_half.y, b_half.z];
    let (u_idx, v_idx) = match i {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let su = if a_axes[u_idx].dot(normal) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let sv = if a_axes[v_idx].dot(normal) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let a_edge_centre =
        a_xf.position + a_axes[u_idx] * (su * a_h[u_idx]) + a_axes[v_idx] * (sv * a_h[v_idx]);

    let (bu_idx, bv_idx) = match j {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let bsu = if b_axes[bu_idx].dot(-normal) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let bsv = if b_axes[bv_idx].dot(-normal) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let b_edge_centre =
        b_xf.position + b_axes[bu_idx] * (bsu * b_h[bu_idx]) + b_axes[bv_idx] * (bsv * b_h[bv_idx]);

    // Closest points between two infinite lines, then clamp.
    let pa = a_edge_centre;
    let pb = b_edge_centre;
    let da = a_axes[i];
    let db = b_axes[j];
    let r = pa - pb;
    let aa = da.dot(da);
    let bb = db.dot(db);
    let ab = da.dot(db);
    let denom = aa * bb - ab * ab;
    let (sa, sb) = if denom.abs() > 1e-12 {
        let s = (ab * r.dot(db) - bb * r.dot(da)) / denom;
        let t = (aa * r.dot(db) - ab * r.dot(da)) / denom;
        (s, t)
    } else {
        (0.0, 0.0)
    };
    let sa = sa.clamp(-a_h[i], a_h[i]);
    let sb = sb.clamp(-b_h[j], b_h[j]);

    let cpa = pa + da * sa;
    let cpb = pb + db * sb;
    let mid = (cpa + cpb) * 0.5;

    out.push(ContactManifold {
        a_shape,
        b_shape,
        point: mid,
        normal,
        depth,
        friction: 0.0,
        rolling_friction: 0.0,
        torsional_friction: 0.0,
        restitution: 0.0,
    });
}

// ─── Sphere-capsule (closed form) ────────────────────────────────────

/// Closest point on the line segment `[a, b]` to a query point `p`,
/// returned as a parameter `t ∈ [0, 1]` and the world-space point.
fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> (Real, Vec3) {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-24 {
        return (0.0, a);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    (t, a + ab * t)
}

fn sphere_capsule(
    sphere_p: Vec3,
    sphere_r: Real,
    capsule_xf: &Transform,
    capsule_half_h: Real,
    capsule_r: Real,
    sphere_shape: u32,
    capsule_shape: u32,
) -> Option<ContactManifold> {
    // Capsule axis in world: shape-local +Y, rotated.
    let axis = capsule_xf.rotation * Vec3::Y;
    let p0 = capsule_xf.position - axis * capsule_half_h;
    let p1 = capsule_xf.position + axis * capsule_half_h;
    let (_, q) = closest_point_on_segment(sphere_p, p0, p1);
    // Now reduces to sphere-sphere between (sphere_p, sphere_r) and
    // (q, capsule_r).
    sphere_sphere(
        sphere_p,
        sphere_r,
        q,
        capsule_r,
        sphere_shape,
        capsule_shape,
    )
}

// ─── Capsule-plane ──────────────────────────────────────────────────

fn capsule_plane(
    capsule_xf: &Transform,
    capsule_half_h: Real,
    capsule_r: Real,
    plane_p: Vec3,
    plane_normal: Vec3,
    capsule_shape: u32,
    plane_shape: u32,
    out: &mut Vec<ContactManifold>,
) {
    // Two end-cap centres; check each as a sphere-vs-plane.
    let axis = capsule_xf.rotation * Vec3::Y;
    let centres = [
        capsule_xf.position - axis * capsule_half_h,
        capsule_xf.position + axis * capsule_half_h,
    ];
    let normal = -plane_normal; // a → b convention: capsule(a) → plane(b)
    for &c in &centres {
        let signed = (c - plane_p).dot(plane_normal);
        if signed < capsule_r {
            let depth = capsule_r - signed;
            let point = c + plane_normal * (capsule_r - 0.5 * depth);
            out.push(ContactManifold {
                a_shape: capsule_shape,
                b_shape: plane_shape,
                point,
                normal,
                depth,
                friction: 0.0,
                rolling_friction: 0.0,
                torsional_friction: 0.0,
                restitution: 0.0,
            });
        }
    }
}

// ─── Capsule-capsule ────────────────────────────────────────────────

/// Closest points between two line segments `(a0, a1)` and `(b0, b1)`,
/// each returned with its segment parameter `t_a, t_b ∈ [0, 1]`.
/// Lifted from the standard "Real-Time Collision Detection" (Ericson).
fn closest_segment_segment(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> (Vec3, Vec3) {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = d1.length_squared();
    let e = d2.length_squared();
    let f = d2.dot(r);
    let eps = 1e-12;

    let mut s;
    let mut t;
    if a <= eps && e <= eps {
        return (a0, b0);
    }
    if a <= eps {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= eps {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            s = if denom != 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    (a0 + d1 * s, b0 + d2 * t)
}

fn capsule_capsule(
    a_xf: &Transform,
    a_half: Real,
    a_radius: Real,
    b_xf: &Transform,
    b_half: Real,
    b_radius: Real,
    a_shape: u32,
    b_shape: u32,
) -> Option<ContactManifold> {
    let a_axis = a_xf.rotation * Vec3::Y;
    let b_axis = b_xf.rotation * Vec3::Y;
    let a0 = a_xf.position - a_axis * a_half;
    let a1 = a_xf.position + a_axis * a_half;
    let b0 = b_xf.position - b_axis * b_half;
    let b1 = b_xf.position + b_axis * b_half;
    let (pa, pb) = closest_segment_segment(a0, a1, b0, b1);
    // Now sphere-sphere between (pa, a_radius) and (pb, b_radius),
    // but emitting indices from the original capsules.
    sphere_sphere(pa, a_radius, pb, b_radius, a_shape, b_shape)
}

// ─── Ellipsoid fast paths (closed-form) ─────────────────────────────

/// Closest point on an axis-aligned ellipsoid surface (semi-axes
/// `(a, b, c)` centred at origin) to query point `p`. Returns the
/// surface point in the ellipsoid's local frame.
///
/// Iterative root-find for the projection — runs the standard
/// Eberly closest-point-on-ellipsoid bisection. Converges in <30
/// iterations for typical aspect ratios. For `p` strictly inside the
/// ellipsoid we instead pick the surface point along the
/// gradient-rescaled radial — it's a "best-effort" projection
/// (the closest-point problem is ill-defined inside; molla just
/// needs a stable normal direction for the contact).
fn closest_point_on_ellipsoid(half: Vec3, p: Vec3) -> Vec3 {
    // Mirror trick: reflect into the +octant, project, reflect back.
    let signs = Vec3::new(
        if p.x >= 0.0 { 1.0 } else { -1.0 },
        if p.y >= 0.0 { 1.0 } else { -1.0 },
        if p.z >= 0.0 { 1.0 } else { -1.0 },
    );
    let p_pos = Vec3::new(p.x.abs(), p.y.abs(), p.z.abs());
    let q = closest_point_first_octant(half, p_pos);
    Vec3::new(q.x * signs.x, q.y * signs.y, q.z * signs.z)
}

/// Project `p` (assumed in the +octant) onto the surface of an
/// ellipsoid with semi-axes `half` centred at origin. Internal —
/// callers go through `closest_point_on_ellipsoid`.
fn closest_point_first_octant(half: Vec3, p: Vec3) -> Vec3 {
    // Sphere fast path: proportional projection.
    if (half.x - half.y).abs() < 1e-12 && (half.y - half.z).abs() < 1e-12 {
        let r = half.x;
        let len = p.length();
        if len < 1e-12 {
            return Vec3::new(r, 0.0, 0.0);
        }
        return p * (r / len);
    }
    // For points exactly on / past the surface in the +octant, run
    // Eberly's bisection on `t ∈ [t0, t1]` such that:
    //   f(t) = Σ (aᵢ·pᵢ / (t + aᵢ²))² − 1
    // monotone decreasing → unique root for outside-points.
    let a = half;
    let a_sq = a * a;
    // f(t) parameterised as above; the surface point is then
    //   xᵢ = aᵢ²·pᵢ / (t + aᵢ²).
    let max_a_sq = a_sq.x.max(a_sq.y).max(a_sq.z);
    let pa = Vec3::new(a.x * p.x, a.y * p.y, a.z * p.z);
    let pa_norm = pa.length();
    // f(0) = (p/a)². If p is on/past the surface this is ≥ 1.
    let f0 = (p.x / a.x).powi(2) + (p.y / a.y).powi(2) + (p.z / a.z).powi(2);
    if f0 <= 1.0 + 1e-12 {
        // Inside: pick surface intersection along the gradient.
        // Solve s·(p/a)² = 1 → s = 1/sqrt((p/a)²); xᵢ = s·pᵢ.
        if f0 < 1e-24 {
            return Vec3::new(a.x, 0.0, 0.0);
        }
        let s = 1.0 / f0.sqrt();
        return Vec3::new(p.x * s, p.y * s, p.z * s);
    }
    // Outside: bisect over t ≥ 0. Upper bound from the worst-case
    // bound `t1 = max(aᵢ)·|p·aᵢ|`.
    let mut t0: Real = 0.0;
    let mut t1: Real = max_a_sq.sqrt() * pa_norm;
    for _ in 0..60 {
        let t = 0.5 * (t0 + t1);
        let denom = Vec3::new(t + a_sq.x, t + a_sq.y, t + a_sq.z);
        let f =
            (pa.x / denom.x).powi(2) + (pa.y / denom.y).powi(2) + (pa.z / denom.z).powi(2) - 1.0;
        if f > 0.0 {
            t0 = t;
        } else {
            t1 = t;
        }
    }
    let t = 0.5 * (t0 + t1);
    let denom = Vec3::new(t + a_sq.x, t + a_sq.y, t + a_sq.z);
    Vec3::new(
        a_sq.x * p.x / denom.x,
        a_sq.y * p.y / denom.y,
        a_sq.z * p.z / denom.z,
    )
}

/// Outward-pointing surface normal of an axis-aligned ellipsoid at
/// surface point `q` (local frame). The ellipsoid's gradient at
/// `q` is `(qx/a², qy/b², qz/c²)`, normalised.
#[inline]
fn ellipsoid_normal_at(half: Vec3, q: Vec3) -> Vec3 {
    let g = Vec3::new(
        q.x / (half.x * half.x),
        q.y / (half.y * half.y),
        q.z / (half.z * half.z),
    );
    let len = g.length();
    if len < 1e-12 { Vec3::Y } else { g / len }
}

/// Ellipsoid-vs-Sphere closed-form contact. Closest-point projects
/// the sphere centre onto the ellipsoid surface (in ellipsoid-local
/// frame); sphere overlaps when `|q − p_sphere| < r_sphere`. Normal
/// is the ellipsoid's outward gradient at `q` (a → b convention,
/// ellipsoid is `a`, sphere is `b`, so the normal points from
/// ellipsoid surface OUT toward the sphere).
fn ellipsoid_sphere(
    ellipsoid_xf: &Transform,
    half_axes: Vec3,
    sphere_p: Vec3,
    sphere_r: Real,
    ellipsoid_shape: u32,
    sphere_shape: u32,
) -> Option<ContactManifold> {
    // Sphere centre in ellipsoid-local frame.
    let p_local = ellipsoid_xf.inverse_transform_point(sphere_p);
    let q_local = closest_point_on_ellipsoid(half_axes, p_local);
    let delta_local = p_local - q_local;
    let dist = delta_local.length();
    let inside = (p_local.x / half_axes.x).powi(2)
        + (p_local.y / half_axes.y).powi(2)
        + (p_local.z / half_axes.z).powi(2)
        <= 1.0;
    let depth = if inside {
        sphere_r + dist
    } else if dist < sphere_r {
        sphere_r - dist
    } else {
        return None;
    };
    let n_local = ellipsoid_normal_at(half_axes, q_local);
    // a→b: ellipsoid → sphere = +n_local (gradient points outward).
    let n_world = ellipsoid_xf.transform_vector(n_local);
    let q_world = ellipsoid_xf.transform_point(q_local);
    let point = q_world + n_world * (0.5 * depth);
    Some(ContactManifold {
        a_shape: ellipsoid_shape,
        b_shape: sphere_shape,
        point,
        normal: n_world,
        depth,
        friction: 0.0,
        rolling_friction: 0.0,
        torsional_friction: 0.0,
        restitution: 0.0,
    })
}

/// Ellipsoid-vs-Plane closed-form contact. Find the support point
/// of the ellipsoid along `-plane_normal` (the lowest point); if it
/// lies below the plane, that's our contact. Normal a→b points from
/// ellipsoid into plane = -plane_normal.
fn ellipsoid_plane(
    ellipsoid_xf: &Transform,
    half_axes: Vec3,
    plane_p: Vec3,
    plane_normal_world: Vec3,
    ellipsoid_shape: u32,
    plane_shape: u32,
) -> Option<ContactManifold> {
    // Express -plane_normal in ellipsoid-local frame, take support.
    let inv_rot = ellipsoid_xf.rotation.conjugate();
    let dir_local = inv_rot * (-plane_normal_world);
    let support_local = support_ellipsoid(half_axes, dir_local);
    let support_world = ellipsoid_xf.transform_point(support_local);
    let signed = (support_world - plane_p).dot(plane_normal_world);
    if signed >= 0.0 {
        return None;
    }
    let depth = -signed;
    let normal = -plane_normal_world; // a→b: ellipsoid → plane
    let point = support_world + plane_normal_world * (0.5 * depth);
    Some(ContactManifold {
        a_shape: ellipsoid_shape,
        b_shape: plane_shape,
        point,
        normal,
        depth,
        friction: 0.0,
        rolling_friction: 0.0,
        torsional_friction: 0.0,
        restitution: 0.0,
    })
}

// ─── Generic GJK + EPA narrow phase ─────────────────────────────────

/// Run GJK on the pair, then EPA if overlapping, returning a single
/// contact manifold. Used as the fallback for any convex-vs-convex
/// pair without a closed-form fast path (capsule-box, cylinder-*,
/// future cone / convex hull).
fn gjk_epa_pair(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    pair: ShapePair,
) -> Option<ContactManifold> {
    gjk_epa_pair_with_hulls(geom_a, xf_a, geom_b, xf_b, pair, &[])
}

#[allow(clippy::too_many_arguments)]
fn gjk_epa_pair_with_hulls(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    pair: ShapePair,
    hulls: &[Vec<Vec3>],
) -> Option<ContactManifold> {
    let res = gjk_with_hulls(geom_a, xf_a, geom_b, xf_b, hulls);
    let simplex = match res {
        GjkResult::Disjoint(_) => return None,
        GjkResult::Overlap(s) => s,
    };
    let contact = epa_with_hulls(geom_a, xf_a, geom_b, xf_b, &simplex, hulls)?;
    if contact.depth <= 0.0 {
        return None;
    }
    let point = (contact.point_a + contact.point_b) * 0.5;
    Some(ContactManifold {
        a_shape: pair.a,
        b_shape: pair.b,
        point,
        normal: contact.normal,
        depth: contact.depth,
        friction: 0.0,
        rolling_friction: 0.0,
        torsional_friction: 0.0,
        restitution: 0.0,
    })
}

/// Triangle-mesh narrow phase: dispatch each mesh triangle to the
/// generic GJK+EPA pair routine. `flip_output` is true when the
/// mesh side is the contact pair's `b` (so each emitted manifold
/// gets its normal flipped before `out.push`).
#[allow(clippy::too_many_arguments)]
fn trimesh_pair(
    model: &Model,
    mesh_id: u32,
    mesh_xf: &Transform,
    other_geom: ShapeGeometry,
    other_xf: &Transform,
    pair: ShapePair,
    flip_output: bool,
    out: &mut Vec<ContactManifold>,
    cache: Option<&mut MeshBvhCache>,
) {
    let mesh = match model.meshes.get(mesh_id as usize) {
        Some(m) => m,
        None => return,
    };
    if matches!(other_geom, ShapeGeometry::Plane { .. }) {
        // GJK doesn't support plane (unbounded support). Plane-mesh
        // is best done as "for each triangle vertex, signed-dist
        // against plane".
        let plane_normal_world = match other_geom {
            ShapeGeometry::Plane { normal } => other_xf.rotation * normal,
            // Outer `matches!(other_geom, Plane)` guarantees this arm
            // is dead — the inner match destructures the same value.
            _ => unreachable!(),
        };
        let plane_p = other_xf.position;
        // Convention: normal points pair.a → pair.b. Mesh side is
        // pair.a when !flip_output, else pair.b.
        let normal = if flip_output {
            plane_normal_world // plane(a) → mesh(b) = +plane_normal
        } else {
            -plane_normal_world // mesh(a) → plane(b) = −plane_normal
        };
        for &tri in &mesh.indices {
            for &vi in &tri {
                let local = mesh.vertices[vi as usize];
                let world = mesh_xf.transform_point(local);
                let signed = (world - plane_p).dot(plane_normal_world);
                if signed < 0.0 {
                    let depth = -signed;
                    let point = world + plane_normal_world * (0.5 * depth);
                    out.push(ContactManifold {
                        a_shape: pair.a,
                        b_shape: pair.b,
                        point,
                        normal,
                        depth,
                        friction: 0.0,
                        rolling_friction: 0.0,
                        torsional_friction: 0.0,
                        restitution: 0.0,
                    });
                }
            }
        }
        return;
    }
    // Query candidate triangles in the mesh's local frame.
    let (other_min, other_max) = shape_world_aabb_quick(other_geom, other_xf);
    let local_aabb = transform_aabb_to_mesh_local(other_min, other_max, mesh_xf);
    let temporary;
    let bvh = if let Some(cache) = cache {
        cache.get(mesh_id as usize, mesh)
    } else {
        temporary = MeshBvh::build(&mesh.vertices, &mesh.indices, 4);
        &temporary
    };
    let mut candidates = Vec::new();
    bvh.triangles_overlapping_aabb(local_aabb.0, local_aabb.1, &mut candidates);
    for &tri_idx in &candidates {
        let tri = mesh.indices[tri_idx as usize];
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];
        let tri_geom = ShapeGeometry::Triangle { v0, v1, v2 };
        let pair_for_call = if flip_output {
            ShapePair {
                a: pair.a,
                b: pair.b,
            }
        } else {
            pair
        };
        let m = if flip_output {
            gjk_epa_pair(other_geom, other_xf, tri_geom, mesh_xf, pair_for_call)
        } else {
            gjk_epa_pair(tri_geom, mesh_xf, other_geom, other_xf, pair_for_call)
        };
        if let Some(m) = m {
            out.push(m);
        }
    }
}

/// World-space AABB for a free-floating shape. Mirrors
/// `shape_world::shape_world_aabb` but takes geom + transform
/// directly so we can call it on a triangle / etc. without touching
/// `Model`.
fn shape_world_aabb_quick(geom: ShapeGeometry, xf: &Transform) -> (Vec3, Vec3) {
    use molla_math::Mat3;
    match geom {
        ShapeGeometry::Sphere { radius } => (
            xf.position - Vec3::splat(radius),
            xf.position + Vec3::splat(radius),
        ),
        ShapeGeometry::Plane { .. } => (Vec3::splat(-Real::INFINITY), Vec3::splat(Real::INFINITY)),
        ShapeGeometry::Box { half_extents } => {
            let r = Mat3::from_quat(xf.rotation);
            let abs_x = Vec3::new(r.x_axis.x.abs(), r.x_axis.y.abs(), r.x_axis.z.abs());
            let abs_y = Vec3::new(r.y_axis.x.abs(), r.y_axis.y.abs(), r.y_axis.z.abs());
            let abs_z = Vec3::new(r.z_axis.x.abs(), r.z_axis.y.abs(), r.z_axis.z.abs());
            let h = abs_x * half_extents.x + abs_y * half_extents.y + abs_z * half_extents.z;
            (xf.position - h, xf.position + h)
        }
        ShapeGeometry::Capsule {
            half_height,
            radius,
        } => {
            let axis = xf.rotation * Vec3::Y;
            let p0 = xf.position - axis * half_height;
            let p1 = xf.position + axis * half_height;
            (
                p0.min(p1) - Vec3::splat(radius),
                p0.max(p1) + Vec3::splat(radius),
            )
        }
        ShapeGeometry::Cylinder {
            half_height,
            radius,
        }
        | ShapeGeometry::RoundCylinder {
            half_height,
            radius,
            ..
        } => {
            let axis = xf.rotation * Vec3::Y;
            let absax = Vec3::new(axis.x.abs(), axis.y.abs(), axis.z.abs());
            let one = Vec3::splat(1.0);
            let perp_scale = (one - absax * absax).max(Vec3::ZERO);
            let perp = Vec3::new(
                perp_scale.x.sqrt(),
                perp_scale.y.sqrt(),
                perp_scale.z.sqrt(),
            );
            let border = match geom {
                ShapeGeometry::RoundCylinder { border_radius, .. } => border_radius,
                _ => 0.0,
            };
            let h = absax * half_height + perp * radius + Vec3::splat(border);
            (xf.position - h, xf.position + h)
        }
        ShapeGeometry::Triangle { v0, v1, v2 } => {
            let w0 = xf.transform_point(v0);
            let w1 = xf.transform_point(v1);
            let w2 = xf.transform_point(v2);
            (w0.min(w1).min(w2), w0.max(w1).max(w2))
        }
        ShapeGeometry::TriMesh { .. } => {
            (Vec3::splat(-Real::INFINITY), Vec3::splat(Real::INFINITY))
        }
        ShapeGeometry::Cone {
            half_height,
            radius,
        } => {
            // Same axis-projection trick as Cylinder.
            let axis = xf.rotation * Vec3::Y;
            let absax = Vec3::new(axis.x.abs(), axis.y.abs(), axis.z.abs());
            let one = Vec3::splat(1.0);
            let perp_scale = (one - absax * absax).max(Vec3::ZERO);
            let perp = Vec3::new(
                perp_scale.x.sqrt(),
                perp_scale.y.sqrt(),
                perp_scale.z.sqrt(),
            );
            let h = absax * half_height + perp * radius;
            (xf.position - h, xf.position + h)
        }
        ShapeGeometry::Ellipsoid { half_axes } => {
            let r = Mat3::from_quat(xf.rotation);
            let row_x = Vec3::new(r.x_axis.x, r.y_axis.x, r.z_axis.x);
            let row_y = Vec3::new(r.x_axis.y, r.y_axis.y, r.z_axis.y);
            let row_z = Vec3::new(r.x_axis.z, r.y_axis.z, r.z_axis.z);
            let h_sq = Vec3::new(
                (row_x * half_axes).length_squared(),
                (row_y * half_axes).length_squared(),
                (row_z * half_axes).length_squared(),
            );
            let h = Vec3::new(h_sq.x.sqrt(), h_sq.y.sqrt(), h_sq.z.sqrt());
            (xf.position - h, xf.position + h)
        }
        ShapeGeometry::ConvexHull { .. }
        | ShapeGeometry::Heightfield { .. }
        | ShapeGeometry::Sdf { .. } => {
            // Conservative — these don't have an obvious analytic
            // AABB without access to model storage. Falls back to
            // visiting every BVH leaf, which is fine for non-hot
            // paths (Heightfield-vs-trimesh is unusual anyway).
            (Vec3::splat(-Real::INFINITY), Vec3::splat(Real::INFINITY))
        }
    }
}

/// Transform a world-space AABB into a local-space AABB by sweeping
/// all eight world corners through the inverse transform. Conservative
/// (i.e. resulting box >= true rotated AABB).
fn transform_aabb_to_mesh_local(
    world_min: Vec3,
    world_max: Vec3,
    mesh_xf: &Transform,
) -> (Vec3, Vec3) {
    let corners = [
        Vec3::new(world_min.x, world_min.y, world_min.z),
        Vec3::new(world_max.x, world_min.y, world_min.z),
        Vec3::new(world_min.x, world_max.y, world_min.z),
        Vec3::new(world_max.x, world_max.y, world_min.z),
        Vec3::new(world_min.x, world_min.y, world_max.z),
        Vec3::new(world_max.x, world_min.y, world_max.z),
        Vec3::new(world_min.x, world_max.y, world_max.z),
        Vec3::new(world_max.x, world_max.y, world_max.z),
    ];
    let mut local_min = Vec3::splat(Real::INFINITY);
    let mut local_max = Vec3::splat(-Real::INFINITY);
    for c in corners {
        let local = mesh_xf.inverse_transform_point(c);
        local_min = local_min.min(local);
        local_max = local_max.max(local);
    }
    (local_min, local_max)
}

/// Cylinder–plane fallback. Emits up to 6 contact points: the two
/// disk centres + 4 perimeter samples per disk that lie below the
/// plane. The 4-perimeter strategy mirrors box-plane: a single
/// support point creates an off-centre torque that rocks the
/// cylinder onto its side, which is non-physical.
fn cylinder_plane_fallback(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    pair: ShapePair,
    out: &mut Vec<ContactManifold>,
) {
    let (cyl_xf, plane_xf, cyl_geom, cyl_shape, plane_shape, flip) = match (geom_a, geom_b) {
        (
            ShapeGeometry::Cylinder { .. } | ShapeGeometry::RoundCylinder { .. },
            ShapeGeometry::Plane { .. },
        ) => (xf_a, xf_b, geom_a, pair.a, pair.b, false),
        (
            ShapeGeometry::Plane { .. },
            ShapeGeometry::Cylinder { .. } | ShapeGeometry::RoundCylinder { .. },
        ) => (xf_b, xf_a, geom_b, pair.b, pair.a, true),
        _ => return,
    };
    let plane_normal_world = match (geom_a, geom_b) {
        (_, ShapeGeometry::Plane { normal }) => xf_b.rotation * normal,
        (ShapeGeometry::Plane { normal }, _) => xf_a.rotation * normal,
        _ => return,
    };
    let plane_p = plane_xf.position;

    let (half_height, radius, border) = match cyl_geom {
        ShapeGeometry::Cylinder {
            half_height,
            radius,
        } => (half_height, radius, 0.0),
        ShapeGeometry::RoundCylinder {
            half_height,
            radius,
            border_radius,
        } => (half_height, radius, border_radius),
        _ => return,
    };

    let axis = cyl_xf.rotation * Vec3::Y;
    let top = cyl_xf.position + axis * half_height;
    let bot = cyl_xf.position - axis * half_height;

    // Build an orthonormal in-plane basis (u, v) perpendicular to
    // the cylinder axis so we can sample 4 points per disk rim.
    let abs_a = Vec3::new(axis.x.abs(), axis.y.abs(), axis.z.abs());
    let helper = if abs_a.x <= abs_a.y && abs_a.x <= abs_a.z {
        Vec3::X
    } else if abs_a.y <= abs_a.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let u = axis.cross(helper).normalize_or_zero();
    let v = axis.cross(u).normalize_or_zero();

    let normal = -plane_normal_world; // a→b: cyl(a) → plane(b)

    // Try a comprehensive set: 2 disk centres + 8 perimeter samples
    // (4 per disk, at ±u, ±v). Push only the points below the plane.
    let mut candidates: Vec<Vec3> = Vec::with_capacity(10);
    candidates.push(top);
    candidates.push(bot);
    for &disk in &[top, bot] {
        candidates.push(disk + u * radius);
        candidates.push(disk - u * radius);
        candidates.push(disk + v * radius);
        candidates.push(disk - v * radius);
    }

    if matches!(cyl_geom, ShapeGeometry::RoundCylinder { .. }) {
        for point in &mut candidates {
            *point -= plane_normal_world * border;
        }
        let lowest = crate::support::support(cyl_geom, cyl_xf, -plane_normal_world);
        if !candidates
            .iter()
            .any(|p| p.distance_squared(lowest) < 1e-24)
        {
            candidates.push(lowest);
        }
    }
    let mut emitted = 0usize;
    for c in candidates {
        let signed = (c - plane_p).dot(plane_normal_world);
        if signed < 0.0 {
            let depth = -signed;
            let point = c + plane_normal_world * (0.5 * depth);
            let m = ContactManifold {
                a_shape: cyl_shape,
                b_shape: plane_shape,
                point,
                normal,
                depth,
                friction: 0.0,
                rolling_friction: 0.0,
                torsional_friction: 0.0,
                restitution: 0.0,
            };
            out.push(if flip { flip_manifold(m) } else { m });
            emitted += 1;
        }
    }
    let _ = emitted;
}

/// Cone-vs-plane: emit a contact for the apex + a few base disc
/// rim samples that fall below the plane. Same pattern as
/// `cylinder_plane_fallback`.
fn cone_plane_fallback(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    pair: ShapePair,
    out: &mut Vec<ContactManifold>,
) {
    let (cone_xf, plane_xf, cone_geom, cone_shape, plane_shape, flip) = match (geom_a, geom_b) {
        (ShapeGeometry::Cone { .. }, ShapeGeometry::Plane { .. }) => {
            (xf_a, xf_b, geom_a, pair.a, pair.b, false)
        }
        (ShapeGeometry::Plane { .. }, ShapeGeometry::Cone { .. }) => {
            (xf_b, xf_a, geom_b, pair.b, pair.a, true)
        }
        _ => return,
    };
    let plane_normal_world = match (geom_a, geom_b) {
        (_, ShapeGeometry::Plane { normal }) => xf_b.rotation * normal,
        (ShapeGeometry::Plane { normal }, _) => xf_a.rotation * normal,
        _ => return,
    };
    let plane_p = plane_xf.position;
    let (half_height, radius) = match cone_geom {
        ShapeGeometry::Cone {
            half_height,
            radius,
        } => (half_height, radius),
        _ => return,
    };
    let axis = cone_xf.rotation * Vec3::Y;
    let apex = cone_xf.position + axis * half_height;
    let base_centre = cone_xf.position - axis * half_height;
    // Two perpendicular axes for sampling the base disc rim.
    let abs_a = Vec3::new(axis.x.abs(), axis.y.abs(), axis.z.abs());
    let helper = if abs_a.x <= abs_a.y && abs_a.x <= abs_a.z {
        Vec3::X
    } else if abs_a.y <= abs_a.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let u = axis.cross(helper).normalize_or_zero();
    let v = axis.cross(u).normalize_or_zero();

    let normal = -plane_normal_world; // a→b: cone(a) → plane(b)
    let mut candidates: Vec<Vec3> = Vec::with_capacity(5);
    candidates.push(apex);
    candidates.push(base_centre + u * radius);
    candidates.push(base_centre - u * radius);
    candidates.push(base_centre + v * radius);
    candidates.push(base_centre - v * radius);
    for c in candidates {
        let signed = (c - plane_p).dot(plane_normal_world);
        if signed < 0.0 {
            let depth = -signed;
            let point = c + plane_normal_world * (0.5 * depth);
            let m = ContactManifold {
                a_shape: cone_shape,
                b_shape: plane_shape,
                point,
                normal,
                depth,
                friction: 0.0,
                rolling_friction: 0.0,
                torsional_friction: 0.0,
                restitution: 0.0,
            };
            out.push(if flip { flip_manifold(m) } else { m });
        }
    }
}

/// ConvexHull-vs-plane: emit a contact for every hull vertex below
/// the plane. Same pattern as the trimesh-plane path.
fn hull_plane_fallback(
    model: &Model,
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    pair: ShapePair,
    out: &mut Vec<ContactManifold>,
) {
    let (hull_xf, plane_xf, hull_geom, hull_shape, plane_shape, flip) = match (geom_a, geom_b) {
        (ShapeGeometry::ConvexHull { .. }, ShapeGeometry::Plane { .. }) => {
            (xf_a, xf_b, geom_a, pair.a, pair.b, false)
        }
        (ShapeGeometry::Plane { .. }, ShapeGeometry::ConvexHull { .. }) => {
            (xf_b, xf_a, geom_b, pair.b, pair.a, true)
        }
        _ => return,
    };
    let plane_normal_world = match (geom_a, geom_b) {
        (_, ShapeGeometry::Plane { normal }) => xf_b.rotation * normal,
        (ShapeGeometry::Plane { normal }, _) => xf_a.rotation * normal,
        _ => return,
    };
    let plane_p = plane_xf.position;
    let hull_id = match hull_geom {
        ShapeGeometry::ConvexHull { hull_id } => hull_id as usize,
        _ => return,
    };
    let hull = match model.hulls.get(hull_id) {
        Some(h) => h,
        None => return,
    };
    let normal = -plane_normal_world;
    let start = out.len();
    for &v_local in hull {
        let v_world = hull_xf.transform_point(v_local);
        let signed = (v_world - plane_p).dot(plane_normal_world);
        if signed < 0.0 {
            let depth = -signed;
            let point = v_world + plane_normal_world * (0.5 * depth);
            let m = ContactManifold {
                a_shape: hull_shape,
                b_shape: plane_shape,
                point,
                normal,
                depth,
                friction: 0.0,
                rolling_friction: 0.0,
                torsional_friction: 0.0,
                restitution: 0.0,
            };
            out.push(if flip { flip_manifold(m) } else { m });
        }
    }
    if out.len() - start > MAX_MANIFOLD_POINTS {
        keep_deepest_range(out, start, MAX_MANIFOLD_POINTS);
    }
}

/// Heightfield narrow phase: enumerate cells whose AABB overlaps
/// the other shape's local-frame AABB, decompose each into 2 CCW
/// triangles, and route through the generic GJK+EPA path. Local
/// frames are converted via `mesh_xf` (the heightfield's transform).
#[allow(clippy::too_many_arguments)]
fn heightfield_pair(
    model: &Model,
    field_id: u32,
    field_xf: &Transform,
    other_geom: ShapeGeometry,
    other_xf: &Transform,
    pair: ShapePair,
    flip_output: bool,
    out: &mut Vec<ContactManifold>,
) {
    let field = match model.heightfields.get(field_id as usize) {
        Some(f) => f,
        None => return,
    };
    if field.nx < 2 || field.nz < 2 || field.cell_size <= 0.0 || field.cell_size_z <= 0.0 {
        return;
    }
    if let ShapeGeometry::ConvexHull { hull_id } = other_geom {
        heightfield_hull_vertex_pair(
            model,
            field,
            field_xf,
            hull_id,
            other_xf,
            pair,
            flip_output,
            out,
        );
        return;
    }
    // Other shape's AABB → field-local frame.
    let (oa, ob) = shape_world_aabb_quick(other_geom, other_xf);
    let (lo, hi) = transform_aabb_to_mesh_local(oa, ob, field_xf);

    // Determine which cell range overlaps.
    let inv_cell = 1.0 / field.cell_size;
    let inv_cell_z = 1.0 / field.cell_size_z;
    let i0 = ((lo.x - field.origin.x) * inv_cell).floor() as i32;
    let i1 = ((hi.x - field.origin.x) * inv_cell).ceil() as i32;
    let j0 = ((lo.z - field.origin.z) * inv_cell_z).floor() as i32;
    let j1 = ((hi.z - field.origin.z) * inv_cell_z).ceil() as i32;
    let i0 = i0.clamp(0, field.nx as i32 - 1).max(0) as u32;
    let i1 = (i1.clamp(0, field.nx as i32 - 1).max(0) as u32).min(field.nx - 1);
    let j0 = j0.clamp(0, field.nz as i32 - 1).max(0) as u32;
    let j1 = (j1.clamp(0, field.nz as i32 - 1).max(0) as u32).min(field.nz - 1);

    for j in j0..j1 {
        for i in i0..i1 {
            for [v0, v1, v2] in field.cell_triangles(i, j).into_iter().flatten() {
                let tri_geom = ShapeGeometry::Triangle { v0, v1, v2 };
                let m = if flip_output {
                    gjk_epa_pair_with_hulls(
                        other_geom,
                        other_xf,
                        tri_geom,
                        field_xf,
                        pair,
                        &model.hulls,
                    )
                } else {
                    gjk_epa_pair_with_hulls(
                        tri_geom,
                        field_xf,
                        other_geom,
                        other_xf,
                        pair,
                        &model.hulls,
                    )
                };
                if let Some(m) = m {
                    out.push(m);
                }
            }
        }
    }
}

/// Emit penetrating hull-vertex contacts against the triangulated height surface.
#[allow(clippy::too_many_arguments)]
fn heightfield_hull_vertex_pair(
    model: &Model,
    field: &molla_sim::Heightfield,
    field_xf: &Transform,
    hull_id: u32,
    hull_xf: &Transform,
    pair: ShapePair,
    flip_output: bool,
    out: &mut Vec<ContactManifold>,
) {
    let Some(hull) = model.hulls.get(hull_id as usize) else {
        return;
    };
    let start = out.len();
    for &v_local in hull {
        let v_world = hull_xf.transform_point(v_local);
        let v_field = field_xf.inverse_transform_point(v_world);
        let Some((height, normal_local)) = field.height_and_normal(v_field.x, v_field.z) else {
            continue;
        };
        let surface_y = height;
        if v_field.y >= surface_y {
            continue;
        }
        let depth = (surface_y - v_field.y) * normal_local.y;
        let normal_world = field_xf.rotation * normal_local;
        let point = v_world + normal_world * (depth * 0.5);
        let normal = if flip_output {
            -normal_world
        } else {
            normal_world
        };
        out.push(ContactManifold {
            a_shape: pair.a,
            b_shape: pair.b,
            point,
            normal,
            depth,
            friction: 0.0,
            rolling_friction: 0.0,
            torsional_friction: 0.0,
            restitution: 0.0,
        });
    }
    if out.len() - start > MAX_MANIFOLD_POINTS {
        keep_deepest_range(out, start, MAX_MANIFOLD_POINTS);
    }
}

/// SDF narrow phase: ball-march the other shape's centre point
/// against the SDF surface. v1 supports Sphere ↔ SDF (closest-point
/// projection); other shapes get their support along the normal at
/// the closest point. Adequate for the demo SDFs (torus, sphere).
#[allow(clippy::too_many_arguments)]
fn sdf_pair(
    model: &Model,
    sdf_id: u32,
    sdf_xf: &Transform,
    other_geom: ShapeGeometry,
    other_xf: &Transform,
    pair: ShapePair,
    flip_output: bool,
    out: &mut Vec<ContactManifold>,
) {
    let sdf = match model.sdfs.get(sdf_id as usize) {
        Some(s) => *s,
        None => return,
    };
    // Project the other shape's centre into the SDF's local frame.
    let other_pos_local = sdf_xf.inverse_transform_point(other_xf.position);
    let signed = sdf.signed_distance(other_pos_local);
    // Other shape's "radius" along the SDF normal direction (used
    // for sphere/box/etc.). Use the tightest analytic bound we have:
    // sphere radius for spheres, max half-extent for boxes, otherwise
    // an AABB-derived fallback.
    let r_other = match other_geom {
        ShapeGeometry::Sphere { radius } => radius,
        ShapeGeometry::Box { half_extents } => {
            half_extents.x.max(half_extents.y).max(half_extents.z)
        }
        ShapeGeometry::Capsule {
            half_height,
            radius,
        } => half_height + radius,
        ShapeGeometry::Cylinder {
            half_height,
            radius,
        } => half_height.max(radius),
        ShapeGeometry::RoundCylinder {
            half_height,
            radius,
            border_radius,
        } => half_height.hypot(radius) + border_radius,
        ShapeGeometry::Cone {
            half_height,
            radius,
        } => half_height.max(radius),
        ShapeGeometry::ConvexHull { hull_id } => {
            // Bounding-sphere radius (max |v|) of the hull's local
            // vertex list. Conservative — gives an early-out radius
            // big enough to catch all contacts.
            model
                .hulls
                .get(hull_id as usize)
                .map(|h| h.iter().map(|v| v.length()).fold(0.0_f64, |a, b| a.max(b)))
                .unwrap_or(0.0)
        }
        _ => 0.0,
    };
    if signed >= r_other {
        return;
    }
    let depth = r_other - signed;
    let n_local = sdf.normal(other_pos_local);
    let n_world = sdf_xf.rotation * n_local;
    // Surface point in SDF-local frame: offset from `other_pos_local`
    // by `signed` along the normal.
    let surface_local = other_pos_local - n_local * signed;
    let surface_world = sdf_xf.transform_point(surface_local);
    // Convention: normal points pair.a → pair.b. When SDF is `a`
    // (flip_output=false), normal goes outward from SDF surface
    // toward the other shape = +n_world. When SDF is `b`
    // (flip_output=true), normal goes from other shape toward SDF
    // surface = −n_world.
    let normal = if flip_output { -n_world } else { n_world };
    out.push(ContactManifold {
        a_shape: pair.a,
        b_shape: pair.b,
        point: surface_world,
        normal,
        depth,
        friction: 0.0,
        rolling_friction: 0.0,
        torsional_friction: 0.0,
        restitution: 0.0,
    });
}

fn flip_manifold(mut m: ContactManifold) -> ContactManifold {
    std::mem::swap(&mut m.a_shape, &mut m.b_shape);
    m.normal = -m.normal;
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::Quat;

    #[test]
    fn sphere_sphere_overlap() {
        let m = sphere_sphere(Vec3::ZERO, 1.0, Vec3::new(1.5, 0.0, 0.0), 1.0, 0, 1).unwrap();
        assert!((m.depth - 0.5).abs() < 1e-9);
        assert!((m.normal - Vec3::X).length() < 1e-9);
    }

    #[test]
    fn sphere_sphere_separated() {
        assert!(sphere_sphere(Vec3::ZERO, 1.0, Vec3::new(3.0, 0.0, 0.0), 1.0, 0, 1).is_none());
    }

    #[test]
    fn sphere_above_plane_no_contact() {
        let m = sphere_plane(Vec3::new(0.0, 1.0, 0.0), 0.5, Vec3::ZERO, Vec3::Y, 0, 1);
        assert!(m.is_none());
    }

    #[test]
    fn sphere_through_plane_contacts() {
        let m = sphere_plane(Vec3::new(0.0, 0.3, 0.0), 0.5, Vec3::ZERO, Vec3::Y, 0, 1).unwrap();
        assert!((m.depth - 0.2).abs() < 1e-9);
        assert!((m.normal + Vec3::Y).length() < 1e-9);
    }

    #[test]
    fn sphere_box_face_contact() {
        // Sphere at (0, 1.4, 0), r=0.5; box at origin half=(1,1,1).
        // Penetration on +Y face = 1 + 0.5 - 1.4 = 0.1.
        let xf = Transform::IDENTITY;
        let m = sphere_box(Vec3::new(0.0, 1.4, 0.0), 0.5, &xf, Vec3::splat(1.0), 0, 1).unwrap();
        assert!((m.depth - 0.1).abs() < 1e-9);
        assert!((m.normal + Vec3::Y).length() < 1e-3);
    }

    #[test]
    fn sphere_box_inside() {
        // Sphere at the box centre — should produce a contact along the
        // closest face, depth = radius + half (here 0.5 + 1.0 = 1.5).
        let xf = Transform::IDENTITY;
        let m = sphere_box(Vec3::ZERO, 0.5, &xf, Vec3::splat(1.0), 0, 1).unwrap();
        assert!(m.depth > 1.0);
    }

    #[test]
    fn box_plane_flat_makes_4_contacts() {
        // Box half=(1,1,1) sitting at y=0.9 with y=0 ground plane → all
        // 4 bottom corners are at y = -0.1, depth 0.1.
        let xf = Transform::new(Vec3::new(0.0, 0.9, 0.0), Quat::IDENTITY);
        let mut out = Vec::new();
        box_plane(&xf, Vec3::splat(1.0), Vec3::ZERO, Vec3::Y, 0, 1, &mut out);
        assert_eq!(out.len(), 4);
        for m in &out {
            assert!((m.depth - 0.1).abs() < 1e-9);
            assert!((m.normal + Vec3::Y).length() < 1e-9);
        }
    }

    #[test]
    fn box_box_face_overlap() {
        // Two unit boxes, B sitting on top of A with 0.1 overlap.
        let a_xf = Transform::IDENTITY;
        let b_xf = Transform::new(Vec3::new(0.0, 1.9, 0.0), Quat::IDENTITY);
        let mut out = Vec::new();
        box_box_sat(
            &a_xf,
            Vec3::splat(1.0),
            &b_xf,
            Vec3::splat(1.0),
            0,
            1,
            &mut out,
        );
        assert!(!out.is_empty(), "expected contacts for overlapping boxes");
        for m in &out {
            assert!(m.depth >= 0.0);
            // a→b normal should point +Y (B is above A).
            assert!(m.normal.dot(Vec3::Y) > 0.5);
        }
    }

    #[test]
    fn box_box_separated_no_contact() {
        let a_xf = Transform::IDENTITY;
        let b_xf = Transform::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY);
        let mut out = Vec::new();
        box_box_sat(
            &a_xf,
            Vec3::splat(1.0),
            &b_xf,
            Vec3::splat(1.0),
            0,
            1,
            &mut out,
        );
        assert!(out.is_empty());
    }

    // ── Capsule fast-path unit tests ─────────────────────────────────

    #[test]
    fn sphere_capsule_side_contact() {
        // Capsule at origin along Y, half_height=1, radius=0.3.
        // Sphere at (0.5, 0, 0), radius=0.3 — surfaces overlap by
        // 0.6 − 0.5 = 0.1. Convention: normal a→b, sphere=a,
        // capsule=b → normal points from sphere(+0.5,0,0) toward
        // capsule centre(0,0,0) = −X.
        let xf_cap = Transform::IDENTITY;
        let m = sphere_capsule(Vec3::new(0.5, 0.0, 0.0), 0.3, &xf_cap, 1.0, 0.3, 0, 1).unwrap();
        assert!(
            (m.depth - 0.1).abs() < 1e-9,
            "expected depth 0.1, got {}",
            m.depth
        );
        assert!(
            (m.normal + Vec3::X).length() < 1e-9,
            "expected normal -X, got {:?}",
            m.normal
        );
    }

    #[test]
    fn sphere_capsule_endpoint_contact() {
        // Sphere beyond +Y end-cap; closest segment point is the
        // end-cap centre at (0, 1, 0). a→b normal points from sphere
        // toward capsule end-cap, i.e. −Y.
        let xf_cap = Transform::IDENTITY;
        let m = sphere_capsule(Vec3::new(0.0, 1.4, 0.0), 0.3, &xf_cap, 1.0, 0.3, 0, 1).unwrap();
        assert!((m.depth - 0.2).abs() < 1e-9);
        assert!((m.normal + Vec3::Y).length() < 1e-9);
    }

    #[test]
    fn sphere_capsule_separated() {
        let xf_cap = Transform::IDENTITY;
        assert!(sphere_capsule(Vec3::new(2.0, 0.0, 0.0), 0.3, &xf_cap, 1.0, 0.3, 0, 1).is_none());
    }

    #[test]
    fn capsule_plane_lying_horizontal() {
        // Capsule lying flat on its side along X above a Y=0 plane.
        // half_height=1, radius=0.3. Centre at (0, 0.4, 0): both
        // end-cap centres at y=0.4, both penetrating the plane by
        // depth=radius-y=-0.4+0.3=−0.1? No: signed = 0.4 < 0.3? No
        // 0.4 > 0.3. So depth = 0.3 - 0.4 = -0.1 → no contact. Move
        // capsule lower: centre at (0, 0.2, 0).
        let xf_cap = Transform::new(
            Vec3::new(0.0, 0.2, 0.0),
            Quat::from_rotation_z(std::f64::consts::FRAC_PI_2),
        );
        let mut out = Vec::new();
        capsule_plane(&xf_cap, 1.0, 0.3, Vec3::ZERO, Vec3::Y, 0, 1, &mut out);
        // Both end-cap centres now sit at y=0.2 (rotation puts them
        // on the X axis, but the plane signed-distance only cares
        // about Y). depth = radius - signed = 0.3 - 0.2 = 0.1.
        assert_eq!(out.len(), 2);
        for m in &out {
            assert!((m.depth - 0.1).abs() < 1e-9, "got depth {}", m.depth);
            assert!((m.normal + Vec3::Y).length() < 1e-9);
        }
    }

    #[test]
    fn capsule_plane_above() {
        // Capsule centre at (0, 5, 0), no contact.
        let xf_cap = Transform::from_translation(Vec3::new(0.0, 5.0, 0.0));
        let mut out = Vec::new();
        capsule_plane(&xf_cap, 1.0, 0.3, Vec3::ZERO, Vec3::Y, 0, 1, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn capsule_capsule_perpendicular() {
        // A along Y at origin (segment x=0, z=0, y∈[-1,1]).
        // B along X at (0, 0, 0.5) — segment y=0, z=0.5, x∈[-1,1].
        // Closest points: A(0,0,0), B(0,0,0.5). Distance 0.5, sum
        // radii 0.6 → depth 0.1. Normal a→b points from A's segment
        // point (0,0,0) toward B's (0,0,0.5) = +Z.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::new(
            Vec3::new(0.0, 0.0, 0.5),
            Quat::from_rotation_z(std::f64::consts::FRAC_PI_2),
        );
        let m = capsule_capsule(&xf_a, 1.0, 0.3, &xf_b, 1.0, 0.3, 0, 1).unwrap();
        assert!((m.depth - 0.1).abs() < 1e-9, "got {}", m.depth);
        assert!((m.normal - Vec3::Z).length() < 1e-9);
    }

    #[test]
    fn capsule_capsule_parallel_separated() {
        // Two parallel Y-capsules far apart on X.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(2.0, 0.0, 0.0));
        assert!(capsule_capsule(&xf_a, 1.0, 0.3, &xf_b, 1.0, 0.3, 0, 1).is_none());
    }

    #[test]
    fn capsule_capsule_parallel_overlapping() {
        // Two parallel Y-capsules close on X.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(0.5, 0.0, 0.0));
        let m = capsule_capsule(&xf_a, 1.0, 0.3, &xf_b, 1.0, 0.3, 0, 1).unwrap();
        // Distance 0.5, sum radii 0.6, depth 0.1.
        assert!((m.depth - 0.1).abs() < 1e-9, "got {}", m.depth);
        assert!((m.normal - Vec3::X).length() < 1e-9);
    }
}
