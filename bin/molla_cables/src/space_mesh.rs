//! High-detail procedural 3D meshes for the ISS IDA-2 Docking Station & Visiting
//! Dragon-Class Spacecraft featuring a real articulated 6-strut Stewart-platform
//! Soft-Capture Ring, interlocking guide petals, 12 swinging hard-capture
//! clamping hooks (Gang A + Gang B), twin O-ring seals, and pressurized-air
//! vestibule vent manifolds.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
use molla_math::{Transform, Vec3};

use crate::draw::{Camera, Surface, Triangle};
use crate::space_sim::{
    CHASER_BULKHEAD_X, RING_FACE_OFFSET_X, STATION_PORT_X, SpaceDockingSim, stewart_anchors,
};

const WHITE_HULL: Color32 = Color32::from_rgb(228, 232, 238);
const RADIATOR_WHITE: Color32 = Color32::from_rgb(240, 243, 248);
const TITANIUM: Color32 = Color32::from_rgb(164, 174, 188);
const STEEL_ROD: Color32 = Color32::from_rgb(206, 214, 224);
const DARK_CARBON: Color32 = Color32::from_rgb(38, 43, 52);
const RUBBER_SEAL: Color32 = Color32::from_rgb(28, 30, 34);
const GOLD_KAPTON: Color32 = Color32::from_rgb(218, 170, 52);
const COPPER_PETAL: Color32 = Color32::from_rgb(202, 132, 74);
const CLAMP_AMBER: Color32 = Color32::from_rgb(228, 162, 54);
const SOLAR_BLUE: Color32 = Color32::from_rgb(28, 62, 128);
const CUPOLA_GLASS: Color32 = Color32::from_rgb(48, 114, 168);
const EVA_YELLOW: Color32 = Color32::from_rgb(232, 198, 56);
const BEACON_GREEN: Color32 = Color32::from_rgb(92, 255, 164);
const BEACON_AMBER: Color32 = Color32::from_rgb(255, 198, 72);
const NOZZLE_GLOW: Color32 = Color32::from_rgb(225, 246, 255);

fn tri_mat(a: Vec3, b: Vec3, c: Vec3, color: Color32, material: f32) -> Triangle {
    Triangle {
        p: [a, b, c],
        color,
        material,
        surface: None,
    }
}

fn tri_uv(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    uv: [[f32; 2]; 3],
    color: Color32,
    material: f32,
) -> Triangle {
    let normal = (b - a).cross(c - a).normalize_or_zero();
    Triangle {
        p: [a, b, c],
        color,
        material,
        surface: Some(Surface {
            normals: [normal; 3],
            uv,
        }),
    }
}

fn push_quad(
    out: &mut Vec<Triangle>,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    color: Color32,
    material: f32,
) {
    out.push(tri_mat(a, b, c, color, material));
    out.push(tri_mat(a, c, d, color, material));
}

#[allow(clippy::too_many_arguments)]
fn push_quad_uv(
    out: &mut Vec<Triangle>,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    uvs: [[f32; 2]; 4],
    color: Color32,
    material: f32,
) {
    out.push(tri_uv(a, b, c, [uvs[0], uvs[1], uvs[2]], color, material));
    out.push(tri_uv(a, c, d, [uvs[0], uvs[2], uvs[3]], color, material));
}

fn push_box(
    out: &mut Vec<Triangle>,
    pose: Transform,
    center: Vec3,
    half: Vec3,
    color: Color32,
    material: f32,
) {
    let p = |sx: f64, sy: f64, sz: f64| {
        pose.transform_point(center + Vec3::new(sx * half.x, sy * half.y, sz * half.z))
    };
    let v = [
        p(-1.0, -1.0, -1.0),
        p(1.0, -1.0, -1.0),
        p(1.0, 1.0, -1.0),
        p(-1.0, 1.0, -1.0),
        p(-1.0, -1.0, 1.0),
        p(1.0, -1.0, 1.0),
        p(1.0, 1.0, 1.0),
        p(-1.0, 1.0, 1.0),
    ];
    let sx = (2.0 * half.x) as f32;
    let sy = (2.0 * half.y) as f32;
    let sz = (2.0 * half.z) as f32;
    for ([a, b, c, d], du, dv) in [
        ([1, 2, 6, 5], sy, sz), // +X
        ([4, 7, 3, 0], sy, sz), // -X
        ([3, 7, 6, 2], sz, sx), // +Y
        ([0, 1, 5, 4], sx, sz), // -Y
        ([4, 5, 6, 7], sx, sy), // +Z
        ([0, 3, 2, 1], sy, sx), // -Z
    ] {
        push_quad_uv(
            out,
            v[a],
            v[b],
            v[c],
            v[d],
            [[0.0, 0.0], [du, 0.0], [du, dv], [0.0, dv]],
            color,
            material,
        );
    }
}

fn basis_for_axis(axis: Vec3) -> (Vec3, Vec3) {
    let helper = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = axis.cross(helper).normalize();
    let v = axis.cross(u).normalize();
    (u, v)
}

#[allow(clippy::too_many_arguments)]
fn push_frustum(
    out: &mut Vec<Triangle>,
    pose: Transform,
    start: Vec3,
    end: Vec3,
    r0: f64,
    r1: f64,
    segments: usize,
    color: Color32,
    material: f32,
    cap_start: bool,
    cap_end: bool,
) {
    let delta = end - start;
    let len = delta.length();
    if len < 1e-6 {
        return;
    }
    let axis = delta / len;
    let (u, v) = basis_for_axis(axis);
    let w_start = pose.transform_point(start);
    let w_end = pose.transform_point(end);
    let r_avg = 0.5 * (r0 + r1);
    for i in 0..segments {
        let a0 = TAU * i as f64 / segments as f64;
        let a1 = TAU * (i + 1) as f64 / segments as f64;
        let d0 = u * a0.cos() + v * a0.sin();
        let d1 = u * a1.cos() + v * a1.sin();
        let p00 = pose.transform_point(start + d0 * r0);
        let p01 = pose.transform_point(start + d1 * r0);
        let p10 = pose.transform_point(end + d0 * r1);
        let p11 = pose.transform_point(end + d1 * r1);
        let u0 = (a0 * r_avg) as f32;
        let u1 = (a1 * r_avg) as f32;
        let v1 = len as f32;
        push_quad_uv(
            out,
            p00,
            p01,
            p11,
            p10,
            [[u0, 0.0], [u1, 0.0], [u1, v1], [u0, v1]],
            color,
            material,
        );
        if cap_start && r0 > 1e-5 {
            out.push(tri_uv(
                w_start,
                p01,
                p00,
                [
                    [0.0, 0.0],
                    [(r0 * a1.cos()) as f32, (r0 * a1.sin()) as f32],
                    [(r0 * a0.cos()) as f32, (r0 * a0.sin()) as f32],
                ],
                color,
                material,
            ));
        }
        if cap_end && r1 > 1e-5 {
            out.push(tri_uv(
                w_end,
                p10,
                p11,
                [
                    [0.0, 0.0],
                    [(r1 * a0.cos()) as f32, (r1 * a0.sin()) as f32],
                    [(r1 * a1.cos()) as f32, (r1 * a1.sin()) as f32],
                ],
                color,
                material,
            ));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_cylinder(
    out: &mut Vec<Triangle>,
    pose: Transform,
    start: Vec3,
    end: Vec3,
    radius: f64,
    segments: usize,
    color: Color32,
    material: f32,
) {
    push_frustum(
        out, pose, start, end, radius, radius, segments, color, material, true, true,
    );
}

/// Hollow machined metal ring (annulus) along X — gives real hollow docking tunnel depth!
#[allow(clippy::too_many_arguments)]
fn push_annulus_x(
    out: &mut Vec<Triangle>,
    pose: Transform,
    x0: f64,
    x1: f64,
    r_inner: f64,
    r_outer: f64,
    segments: usize,
    color: Color32,
    material: f32,
) {
    for i in 0..segments {
        let a0 = TAU * i as f64 / segments as f64;
        let a1 = TAU * (i + 1) as f64 / segments as f64;
        let r0_dir = Vec3::new(0.0, a0.cos(), a0.sin());
        let r1_dir = Vec3::new(0.0, a1.cos(), a1.sin());
        let o00 = pose.transform_point(Vec3::new(x0, 0.0, 0.0) + r0_dir * r_outer);
        let o01 = pose.transform_point(Vec3::new(x0, 0.0, 0.0) + r1_dir * r_outer);
        let o10 = pose.transform_point(Vec3::new(x1, 0.0, 0.0) + r0_dir * r_outer);
        let o11 = pose.transform_point(Vec3::new(x1, 0.0, 0.0) + r1_dir * r_outer);

        let i00 = pose.transform_point(Vec3::new(x0, 0.0, 0.0) + r0_dir * r_inner);
        let i01 = pose.transform_point(Vec3::new(x0, 0.0, 0.0) + r1_dir * r_inner);
        let i10 = pose.transform_point(Vec3::new(x1, 0.0, 0.0) + r0_dir * r_inner);
        let i11 = pose.transform_point(Vec3::new(x1, 0.0, 0.0) + r1_dir * r_inner);

        // Outer wall (+radial), inner wall (-radial), front face (+X), back face (-X) with body-locked UVs.
        let u0 = (a0 * r_outer) as f32;
        let u1 = (a1 * r_outer) as f32;
        let vx0 = x0 as f32;
        let vx1 = x1 as f32;
        push_quad_uv(
            out,
            o00,
            o01,
            o11,
            o10,
            [[u0, vx0], [u1, vx0], [u1, vx1], [u0, vx1]],
            color,
            material,
        );
        push_quad_uv(
            out,
            i00,
            i10,
            i11,
            i01,
            [[u0, vx0], [u0, vx1], [u1, vx1], [u1, vx0]],
            color,
            material,
        );
        push_quad_uv(
            out,
            i10,
            o10,
            o11,
            i11,
            [
                [(r_inner * a0.cos()) as f32, (r_inner * a0.sin()) as f32],
                [(r_outer * a0.cos()) as f32, (r_outer * a0.sin()) as f32],
                [(r_outer * a1.cos()) as f32, (r_outer * a1.sin()) as f32],
                [(r_inner * a1.cos()) as f32, (r_inner * a1.sin()) as f32],
            ],
            color,
            material,
        );
        push_quad_uv(
            out,
            o00,
            i00,
            i01,
            o01,
            [
                [(r_outer * a0.cos()) as f32, (r_outer * a0.sin()) as f32],
                [(r_inner * a0.cos()) as f32, (r_inner * a0.sin()) as f32],
                [(r_inner * a1.cos()) as f32, (r_inner * a1.sin()) as f32],
                [(r_outer * a1.cos()) as f32, (r_outer * a1.sin()) as f32],
            ],
            color,
            material,
        );
    }
}

/// Pushes an angled trapezoidal IDSS guide petal mounted around a docking ring.
#[allow(clippy::too_many_arguments)]
fn push_guide_petal(
    out: &mut Vec<Triangle>,
    pose: Transform,
    base_x: f64,
    dir_x: f64,
    angle: f64,
    radius: f64,
    inward_tilt: f64,
    length: f64,
    width_base: f64,
    width_tip: f64,
    color: Color32,
    material: f32,
) {
    let radial = Vec3::new(0.0, angle.cos(), angle.sin());
    let tang = Vec3::new(0.0, -angle.sin(), angle.cos());
    let t = 0.022; // plate thickness
    let b_ctr = Vec3::new(base_x, 0.0, 0.0) + radial * radius;
    let tip_ctr = Vec3::new(base_x + dir_x * length, 0.0, 0.0) + radial * (radius - inward_tilt);

    let corners = |o: f64| {
        [
            pose.transform_point(b_ctr + radial * o - tang * width_base),
            pose.transform_point(b_ctr + radial * o + tang * width_base),
            pose.transform_point(tip_ctr + radial * o + tang * width_tip),
            pose.transform_point(tip_ctr + radial * o - tang * width_tip),
        ]
    };
    let out_f = corners(t);
    let in_f = corners(-t);
    if dir_x >= 0.0 {
        push_quad(out, out_f[0], out_f[1], out_f[2], out_f[3], color, material);
        push_quad(out, in_f[3], in_f[2], in_f[1], in_f[0], color, material);
        for k in 0..4 {
            let kn = (k + 1) % 4;
            push_quad(out, out_f[kn], out_f[k], in_f[k], in_f[kn], color, material);
        }
    } else {
        push_quad(out, out_f[3], out_f[2], out_f[1], out_f[0], color, material);
        push_quad(out, in_f[0], in_f[1], in_f[2], in_f[3], color, material);
        for k in 0..4 {
            let kn = (k + 1) % 4;
            push_quad(out, out_f[k], out_f[kn], in_f[kn], in_f[k], color, material);
        }
    }
}

fn push_truss_z(
    out: &mut Vec<Triangle>,
    pose: Transform,
    x_center: f64,
    z0: f64,
    z1: f64,
    half: f64,
    bays: usize,
) {
    let corners = [
        Vec3::new(x_center - half, -half, 0.0),
        Vec3::new(x_center + half, -half, 0.0),
        Vec3::new(x_center + half, half, 0.0),
        Vec3::new(x_center - half, half, 0.0),
    ];
    for c in corners {
        push_cylinder(
            out,
            pose,
            c + Vec3::new(0.0, 0.0, z0),
            c + Vec3::new(0.0, 0.0, z1),
            0.028,
            6,
            TITANIUM,
            12.0,
        );
    }
    let dz = (z1 - z0) / bays as f64;
    for b in 0..bays {
        let za = z0 + b as f64 * dz;
        let zb = za + dz;
        for f in 0..4 {
            let ca = corners[f];
            let cb = corners[(f + 1) % 4];
            push_cylinder(
                out,
                pose,
                ca + Vec3::new(0.0, 0.0, zb),
                cb + Vec3::new(0.0, 0.0, zb),
                0.018,
                5,
                TITANIUM,
                12.0,
            );
            let (p0, p1) = if (b + f) % 2 == 0 {
                (ca + Vec3::new(0.0, 0.0, za), cb + Vec3::new(0.0, 0.0, zb))
            } else {
                (cb + Vec3::new(0.0, 0.0, za), ca + Vec3::new(0.0, 0.0, zb))
            };
            push_cylinder(out, pose, p0, p1, 0.016, 5, TITANIUM, 12.0);
        }
    }
}

/// Renders the inner sun-tracking root segment of a 2-segment flexible solar array wing
/// (`z_local` from `-1.125*sign_z` rotary hub to `+1.125*sign_z` elastic bending hinge).
fn build_solar_wing_inner(out: &mut Vec<Triangle>, pose: Transform, sign_z: f64) {
    let hub_z = -1.125 * sign_z;
    let hinge_z = 1.125 * sign_z;
    push_cylinder(
        out,
        pose,
        Vec3::new(0.0, 0.0, hub_z - 0.16 * sign_z),
        Vec3::new(0.0, 0.0, hub_z + 0.18 * sign_z),
        0.24,
        14,
        GOLD_KAPTON,
        15.0,
    );
    for &x_off in &[-0.86, 0.86] {
        push_cylinder(
            out,
            pose,
            Vec3::new(0.0, 0.0, hub_z + 0.12 * sign_z),
            Vec3::new(x_off, 0.0, hub_z + 0.18 * sign_z),
            0.055,
            8,
            TITANIUM,
            12.0,
        );
        let z_start = hub_z + 0.18 * sign_z;
        push_cylinder(
            out,
            pose,
            Vec3::new(x_off, 0.0, z_start),
            Vec3::new(x_off, 0.0, hinge_z),
            0.036,
            8,
            TITANIUM,
            12.0,
        );
        // Elastic bending hinge clevis knuckle at the mid-wing joint!
        push_cylinder(
            out,
            pose,
            Vec3::new(x_off, -0.075, hinge_z),
            Vec3::new(x_off, 0.075, hinge_z),
            0.048,
            10,
            GOLD_KAPTON,
            15.0,
        );
        let segs = 3;
        let span = 1.92;
        let seg_len = span / segs as f64;
        for k in 0..segs {
            let zc = z_start + sign_z * (0.12 + (k as f64 + 0.5) * seg_len);
            push_box(
                out,
                pose,
                Vec3::new(x_off, 0.0, zc),
                Vec3::new(0.72, 0.014, seg_len * 0.46),
                SOLAR_BLUE,
                13.0,
            );
            push_box(
                out,
                pose,
                Vec3::new(x_off, 0.0, z_start + sign_z * (0.12 + k as f64 * seg_len)),
                Vec3::new(0.74, 0.022, 0.018),
                GOLD_KAPTON,
                15.0,
            );
        }
    }
}

/// Renders the outer flexible bending segment of a solar array wing (hinged at `-1.30*sign_z`
/// and extending `+2.60*sign_z` to the wingtip, dynamically flexing at 0.28 Hz under CSI!).
fn build_solar_wing_outer(out: &mut Vec<Triangle>, pose: Transform, sign_z: f64) {
    let hinge_z = -1.30 * sign_z;
    let tip_z = hinge_z + 2.60 * sign_z;
    for &x_off in &[-0.86, 0.86] {
        push_cylinder(
            out,
            pose,
            Vec3::new(x_off, 0.0, hinge_z),
            Vec3::new(x_off, 0.0, tip_z),
            0.034,
            8,
            TITANIUM,
            12.0,
        );
        push_box(
            out,
            pose,
            Vec3::new(x_off, 0.0, tip_z),
            Vec3::new(0.76, 0.035, 0.045),
            GOLD_KAPTON,
            15.0,
        );
        let segs = 4;
        let span = 2.48;
        let seg_len = span / segs as f64;
        for k in 0..segs {
            let zc = hinge_z + sign_z * (0.06 + (k as f64 + 0.5) * seg_len);
            push_box(
                out,
                pose,
                Vec3::new(x_off, 0.0, zc),
                Vec3::new(0.72, 0.014, seg_len * 0.46),
                SOLAR_BLUE,
                13.0,
            );
            push_box(
                out,
                pose,
                Vec3::new(x_off, 0.0, hinge_z + sign_z * (0.06 + k as f64 * seg_len)),
                Vec3::new(0.74, 0.022, 0.018),
                GOLD_KAPTON,
                15.0,
            );
        }
    }
}

fn build_station_mesh(
    out: &mut Vec<Triangle>,
    pose: Transform,
    soft_captured: bool,
    seal_verified: bool,
    venting_air: bool,
    time_s: f64,
) {
    // 1. Main pressurized Habitat / Harmony Core along X: [-2.05, +1.48].
    push_cylinder(
        out,
        pose,
        Vec3::new(-2.05, 0.0, 0.0),
        Vec3::new(1.48, 0.0, 0.0),
        0.96,
        32,
        WHITE_HULL,
        11.0,
    );
    push_frustum(
        out,
        pose,
        Vec3::new(-2.55, 0.0, 0.0),
        Vec3::new(-2.05, 0.0, 0.0),
        0.58,
        0.96,
        24,
        GOLD_KAPTON,
        15.0,
        true,
        false,
    );
    for x in [-1.85, -0.95, 0.05, 0.95, 1.40] {
        push_cylinder(
            out,
            pose,
            Vec3::new(x - 0.055, 0.0, 0.0),
            Vec3::new(x + 0.055, 0.0, 0.0),
            1.01,
            32,
            if x < -0.5 { TITANIUM } else { GOLD_KAPTON },
            if x < -0.5 { 12.0 } else { 15.0 },
        );
    }
    // EVA yellow handrails.
    for &(y, z) in &[(0.82, 0.65), (0.82, -0.65)] {
        push_cylinder(
            out,
            pose,
            Vec3::new(-1.5, y, z),
            Vec3::new(1.1, y, z),
            0.018,
            6,
            EVA_YELLOW,
            12.0,
        );
    }

    // 2. Radial Science Lab Module (-Z).
    push_cylinder(
        out,
        pose,
        Vec3::new(-0.95, 0.0, -0.90),
        Vec3::new(-0.95, 0.0, -2.55),
        0.68,
        24,
        WHITE_HULL,
        11.0,
    );
    push_frustum(
        out,
        pose,
        Vec3::new(-0.95, 0.0, -2.55),
        Vec3::new(-0.95, 0.0, -2.85),
        0.68,
        0.44,
        20,
        GOLD_KAPTON,
        15.0,
        false,
        true,
    );

    // 3. Forward Pressurized Mating Adapter (PMA) + High-Detail Passive IDA-2 Collar!
    push_frustum(
        out,
        pose,
        Vec3::new(1.48, 0.0, 0.0),
        Vec3::new(2.26, 0.0, 0.0),
        0.92,
        0.60,
        32,
        TITANIUM,
        12.0,
        false,
        false,
    );
    // Recessed dark pressurized hatch bulkhead inside the PMA tunnel.
    push_cylinder(
        out,
        pose,
        Vec3::new(2.20, 0.0, 0.0),
        Vec3::new(2.34, 0.0, 0.0),
        0.42,
        28,
        DARK_CARBON,
        10.0,
    );
    // Machined hollow IDA-2 structural outer collar & clamping flange lip!
    push_annulus_x(
        out,
        pose,
        2.26,
        STATION_PORT_X - 0.05,
        0.42,
        0.60,
        32,
        DARK_CARBON,
        10.0,
    );
    // Clamping catch-flange lip (the machined titanium rim the 12 hooks grab onto!).
    push_annulus_x(
        out,
        pose,
        STATION_PORT_X - 0.09,
        STATION_PORT_X - 0.015,
        0.43,
        0.635,
        32,
        TITANIUM,
        12.0,
    );
    // Twin concentric dark elastomeric O-ring pressure seals on the IDA-2 mating face!
    push_annulus_x(
        out,
        pose,
        STATION_PORT_X - 0.016,
        STATION_PORT_X + 0.004,
        0.46,
        0.495,
        32,
        RUBBER_SEAL,
        10.0,
    );
    push_annulus_x(
        out,
        pose,
        STATION_PORT_X - 0.016,
        STATION_PORT_X + 0.004,
        0.535,
        0.565,
        32,
        RUBBER_SEAL,
        10.0,
    );
    // 3 Passive IDA-2 Guide-Petal Funnels (flaring outward 18° to match real physical colliders!).
    for i in 0..3 {
        let a = i as f64 * (TAU / 3.0) + PI / 3.0;
        push_guide_petal(
            out,
            pose,
            STATION_PORT_X - 0.03,
            1.0,
            a,
            0.445,
            -0.05,
            0.16,
            0.13,
            0.06,
            COPPER_PETAL,
            12.0,
        );
    }
    // 12 Passive Catch-Striker Blocks around the IDA-2 flange perimeter (matching the 12 active clamps!).
    for i in 0..12 {
        let a = i as f64 * (TAU / 12.0);
        let rad = Vec3::new(0.0, a.cos(), a.sin());
        push_box(
            out,
            pose,
            Vec3::new(STATION_PORT_X - 0.075, 0.0, 0.0) + rad * 0.635,
            Vec3::new(0.035, 0.032, 0.032),
            STEEL_ROD,
            12.0,
        );
    }
    // Twin Lateral Vestibule Pressurized-Air Vent & Equalization Valve Manifolds (+/-Z)!
    for &sign_z in &[-1.0, 1.0] {
        push_cylinder(
            out,
            pose,
            Vec3::new(STATION_PORT_X - 0.06, 0.0, sign_z * 0.54),
            Vec3::new(STATION_PORT_X - 0.06, 0.0, sign_z * 0.68),
            0.042,
            10,
            if venting_air { NOZZLE_GLOW } else { TITANIUM },
            if venting_air { 14.0 } else { 12.0 },
        );
        push_frustum(
            out,
            pose,
            Vec3::new(STATION_PORT_X - 0.06, 0.0, sign_z * 0.66),
            Vec3::new(STATION_PORT_X - 0.06, 0.0, sign_z * 0.74),
            0.036,
            0.058,
            12,
            DARK_CARBON,
            10.0,
            false,
            false,
        );
    }
    // Dorsal pressure equalization manifold & status indicator strip.
    let strobe = (time_s * 3.0).sin() > 0.0;
    let status_col = if seal_verified {
        BEACON_GREEN
    } else if soft_captured {
        BEACON_AMBER
    } else if strobe {
        BEACON_AMBER
    } else {
        Color32::from_rgb(90, 70, 28)
    };
    push_annulus_x(
        out,
        pose,
        STATION_PORT_X - 0.028,
        STATION_PORT_X - 0.012,
        0.595,
        0.618,
        32,
        status_col,
        14.0,
    );

    // 4. Nadir 7-Window Observation Cupola (-Y).
    push_frustum(
        out,
        pose,
        Vec3::new(0.25, -0.92, 0.0),
        Vec3::new(0.25, -1.36, 0.0),
        0.50,
        0.34,
        7,
        TITANIUM,
        12.0,
        false,
        true,
    );
    push_cylinder(
        out,
        pose,
        Vec3::new(0.25, -1.33, 0.0),
        Vec3::new(0.25, -1.40, 0.0),
        0.26,
        7,
        CUPOLA_GLASS,
        12.0,
    );

    // 5. Open-Lattice Integrated Truss Structure + Dorsal CMG Avionics Bay.
    push_truss_z(out, pose, -0.35, -2.0, 2.0, 0.34, 8);
    push_box(
        out,
        pose,
        Vec3::new(-0.35, 0.52, 0.0),
        Vec3::new(0.36, 0.22, 0.42),
        GOLD_KAPTON,
        15.0,
    );

    // 6. Accordion Thermal Radiators (+/-Y aft) + Ku-Band Dish.
    for &sign_y in &[-1.0, 1.0] {
        push_cylinder(
            out,
            pose,
            Vec3::new(-1.55, sign_y * 0.96, 0.0),
            Vec3::new(-1.55, sign_y * 2.85, 0.0),
            0.03,
            8,
            TITANIUM,
            12.0,
        );
        for k in 0..3 {
            push_box(
                out,
                pose,
                Vec3::new(-1.55, sign_y * (1.42 + k as f64 * 0.56), 0.0),
                Vec3::new(0.46, 0.25, 0.022),
                RADIATOR_WHITE,
                11.0,
            );
        }
    }
    push_cylinder(
        out,
        pose,
        Vec3::new(-1.95, 0.92, 0.35),
        Vec3::new(-1.95, 1.52, 0.35),
        0.035,
        8,
        TITANIUM,
        12.0,
    );
    push_frustum(
        out,
        pose,
        Vec3::new(-1.95, 1.52, 0.35),
        Vec3::new(-1.80, 1.66, 0.42),
        0.08,
        0.52,
        18,
        RADIATOR_WHITE,
        11.0,
        true,
        false,
    );
}

/// Renders the 16 physical Draco RCS thruster bells (with subtle internal chamber glow when firing).
/// The actual pressurized gas jets are rendered as expanding volumetric clouds via `studio::paint_spray`!
fn push_rcs_nozzles(
    out: &mut Vec<Triangle>,
    sim: &SpaceDockingSim,
    pose: Transform,
    thrusters: &[molla_solvers::devices::ThrusterId],
) {
    for &id in thrusters {
        let (Some(spec), Some(read)) = (
            sim.world.devices.thrusters.spec(id),
            sim.world.devices.thrusters.reading(id),
        ) else {
            continue;
        };
        let nozzle_pos = spec.frame.position;
        let exhaust_dir = -(spec.frame.rotation * Vec3::X);
        push_frustum(
            out,
            pose,
            nozzle_pos,
            nozzle_pos + exhaust_dir * 0.14,
            0.028,
            0.062,
            10,
            DARK_CARBON,
            10.0,
            true,
            false,
        );
        let frac = if spec.max_thrust > 0.0 {
            (read.thrust / spec.max_thrust).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Subtle bright inner throat flash inside the bell while firing (no solid plastic cones!).
        if frac > 0.03 {
            push_frustum(
                out,
                pose,
                nozzle_pos + exhaust_dir * 0.11,
                nozzle_pos + exhaust_dir * 0.145,
                0.022,
                0.048 * (0.6 + 0.4 * frac),
                10,
                NOZZLE_GLOW,
                14.0,
                false,
                true,
            );
        }
    }
}

/// Renders the **real physical 6-strut Stewart-Platform Soft-Capture Ring (SCR)**
/// connecting the spacecraft's forward bulkhead to the live dynamic `capture_ring` body,
/// PLUS the **12 articulated Hard-Capture Clamping Hooks (Gang A & Gang B)**!
fn build_active_docking_mechanism(
    out: &mut Vec<Triangle>,
    sim: &SpaceDockingSim,
    chaser_pose: Transform,
    ring_pose: Transform,
) {
    // 1. Machined Forward Structural Interface Flange on the spacecraft bulkhead (at CHASER_BULKHEAD_X).
    push_annulus_x(
        out,
        chaser_pose,
        CHASER_BULKHEAD_X - 0.03,
        CHASER_BULKHEAD_X + 0.09,
        0.41,
        0.58,
        32,
        TITANIUM,
        12.0,
    );
    // Dark recessed tunnel interior behind the bulkhead.
    push_cylinder(
        out,
        chaser_pose,
        Vec3::new(CHASER_BULKHEAD_X + 0.04, 0.0, 0.0),
        Vec3::new(CHASER_BULKHEAD_X + 0.12, 0.0, 0.0),
        0.41,
        24,
        DARK_CARBON,
        10.0,
    );

    // 2. 6 Telescoping Stewart-Platform Linear Actuator Struts (Hexapod Legs!)
    //    Uses the EXACT same 3D clevis anchors as the 960 Hz physics force callback!
    let (stewart_base, stewart_ring) = stewart_anchors();
    for i in 0..6 {
        let p_base = chaser_pose.transform_point(stewart_base[i]);
        let p_ring = ring_pose.transform_point(stewart_ring[i]);
        let mid = p_base.lerp(p_ring, 0.52);
        push_cylinder(
            out,
            Transform::IDENTITY,
            p_base,
            mid,
            0.026,
            8,
            DARK_CARBON,
            10.0,
        );
        push_cylinder(
            out,
            Transform::IDENTITY,
            mid,
            p_ring,
            0.017,
            8,
            STEEL_ROD,
            12.0,
        );
        push_cylinder(
            out,
            Transform::IDENTITY,
            p_base - (p_ring - p_base).normalize_or_zero() * 0.02,
            p_base + (p_ring - p_base).normalize_or_zero() * 0.03,
            0.033,
            8,
            GOLD_KAPTON,
            15.0,
        );
    }

    // 3. The Moving Active Soft-Capture Ring (SCR) body itself!
    push_annulus_x(
        out,
        ring_pose,
        RING_FACE_OFFSET_X,
        0.04,
        0.41,
        0.54,
        32,
        TITANIUM,
        12.0,
    );
    // Elastomeric seal face on the active ring.
    push_annulus_x(
        out,
        ring_pose,
        RING_FACE_OFFSET_X - 0.008,
        RING_FACE_OFFSET_X + 0.004,
        0.45,
        0.50,
        32,
        RUBBER_SEAL,
        10.0,
    );
    // 3 Active Guide Petals (at 0°, 120°, 240° — interlocking cleanly between IDA-2's 60°, 180°, 300°!).
    for i in 0..3 {
        let a = i as f64 * (TAU / 3.0);
        push_guide_petal(
            out,
            ring_pose,
            RING_FACE_OFFSET_X + 0.02,
            -1.0,
            a,
            0.46,
            0.05,
            0.16,
            0.13,
            0.05,
            COPPER_PETAL,
            12.0,
        );
    }

    // 4. 12 REAL Dynamic Hinged Hard-Capture Clamping Hooks!
    //    Rendered directly at each hook rigid body's LIVE solver `body_pose` (ZERO fake animation)!
    for i in 0..12 {
        let a = i as f64 * (TAU / 12.0);
        let rad = Vec3::new(0.0, a.cos(), a.sin());
        let pivot_local = Vec3::new(CHASER_BULKHEAD_X - 0.04, 0.0, 0.0) + rad * 0.60;
        // Hinge bracket clevis bolted to spacecraft collar.
        push_box(
            out,
            chaser_pose,
            pivot_local + Vec3::new(0.04, 0.0, 0.0),
            Vec3::new(0.045, 0.036, 0.036),
            DARK_CARBON,
            10.0,
        );
        let Ok(hook_pose) = sim.world.scene.body_pose(sim.hook_bodies[i]) else {
            continue;
        };
        let load = sim.hook_forces_n[i];
        let hook_col = if load > 3_500.0 {
            CLAMP_AMBER
        } else if sim.hook_engaged[i] {
            BEACON_AMBER
        } else {
            STEEL_ROD
        };
        // Local hook frame: +X = forward arm across seam, -Y = radially inward jaw tooth!
        push_box(
            out,
            hook_pose,
            Vec3::new(0.125, 0.0, 0.0),
            Vec3::new(0.135, 0.019, 0.020),
            hook_col,
            12.0,
        );
        push_box(
            out,
            hook_pose,
            Vec3::new(0.245, -0.052, 0.0),
            Vec3::new(0.024, 0.052, 0.022),
            hook_col,
            12.0,
        );
    }
}

fn build_dragon_capsule_mesh(
    out: &mut Vec<Triangle>,
    sim: &SpaceDockingSim,
    chaser_pose: Transform,
    ring_pose: Transform,
) {
    // 1. Aft Cylindrical Service Trunk (+X: +0.32..+1.04) with recessed cargo bay & solar wings.
    push_cylinder(
        out,
        chaser_pose,
        Vec3::new(0.32, 0.0, 0.0),
        Vec3::new(1.02, 0.0, 0.0),
        0.64,
        32,
        WHITE_HULL,
        11.0,
    );
    push_cylinder(
        out,
        chaser_pose,
        Vec3::new(1.01, 0.0, 0.0),
        Vec3::new(1.06, 0.0, 0.0),
        0.64,
        32,
        TITANIUM,
        12.0,
    );
    push_cylinder(
        out,
        chaser_pose,
        Vec3::new(1.03, 0.0, 0.0),
        Vec3::new(1.07, 0.0, 0.0),
        0.56,
        24,
        DARK_CARBON,
        10.0,
    );
    push_box(
        out,
        chaser_pose,
        Vec3::new(0.68, 0.65, 0.0),
        Vec3::new(0.30, 0.018, 0.36),
        SOLAR_BLUE,
        13.0,
    );
    for &sign_z in &[-1.0, 1.0] {
        push_box(
            out,
            chaser_pose,
            Vec3::new(0.72, 0.0, sign_z * 0.70),
            Vec3::new(0.14, 0.06, 0.08),
            GOLD_KAPTON,
            15.0,
        );
        push_box(
            out,
            chaser_pose,
            Vec3::new(0.72, 0.0, sign_z * 1.18),
            Vec3::new(0.32, 0.016, 0.44),
            SOLAR_BLUE,
            13.0,
        );
        push_cylinder(
            out,
            chaser_pose,
            Vec3::new(0.72, 0.0, sign_z * 0.64),
            Vec3::new(0.72, 0.0, sign_z * 1.65),
            0.022,
            8,
            TITANIUM,
            12.0,
        );
    }

    // 2. Dark Carbon PICA-X Heatshield Ring.
    push_cylinder(
        out,
        chaser_pose,
        Vec3::new(0.20, 0.0, 0.0),
        Vec3::new(0.34, 0.0, 0.0),
        0.69,
        32,
        DARK_CARBON,
        10.0,
    );

    // 3. Lofted Conical Gumdrop Pressure Capsule (-X: +0.20..CHASER_BULKHEAD_X).
    let rings: &[(f64, f64, Color32, f32)] = &[
        (0.20, 0.68, WHITE_HULL, 11.0),
        (-0.22, 0.64, WHITE_HULL, 11.0),
        (-0.62, 0.56, WHITE_HULL, 11.0),
        (CHASER_BULKHEAD_X + 0.06, 0.54, GOLD_KAPTON, 15.0),
    ];
    for pair in rings.windows(2) {
        push_frustum(
            out,
            chaser_pose,
            Vec3::new(pair[0].0, 0.0, 0.0),
            Vec3::new(pair[1].0, 0.0, 0.0),
            pair[0].1,
            pair[1].1,
            32,
            pair[1].2,
            pair[1].3,
            false,
            false,
        );
    }
    // 4 recessed Draco thruster pod fairings.
    for i in 0..4 {
        let a = i as f64 * FRAC_PI_2;
        let dir = Vec3::new(0.0, a.cos(), a.sin());
        push_box(
            out,
            chaser_pose,
            Vec3::new(-0.18, 0.0, 0.0) + dir * 0.63,
            Vec3::new(0.26, 0.10, 0.10),
            DARK_CARBON,
            10.0,
        );
    }
    // Crew observation windows.
    for &sign_z in &[-1.0, 1.0] {
        push_box(
            out,
            chaser_pose,
            Vec3::new(-0.30, 0.22, sign_z * 0.57),
            Vec3::new(0.11, 0.07, 0.04),
            CUPOLA_GLASS,
            12.0,
        );
    }

    // 4. Active Stewart-Platform Soft-Capture Ring + 12 Swinging Hard-Capture Clamping Hooks!
    build_active_docking_mechanism(out, sim, chaser_pose, ring_pose);

    // 5. 16 Physical Draco RCS Bells (volumetric gas plumes rendered separately by `paint_spray`).
    push_rcs_nozzles(out, sim, chaser_pose, &sim.chaser_thrusters);
}

pub fn scene_mesh(sim: &SpaceDockingSim) -> molla_core::Result<Vec<Triangle>> {
    let mut tris = Vec::with_capacity(11_000);
    let station_pose = sim.world.scene.body_pose(sim.station)?;
    let solar_port = sim.world.scene.body_pose(sim.solar_port)?;
    let solar_port_outer = sim.world.scene.body_pose(sim.solar_port_outer)?;
    let solar_starboard = sim.world.scene.body_pose(sim.solar_starboard)?;
    let solar_starboard_outer = sim.world.scene.body_pose(sim.solar_starboard_outer)?;
    let chaser_pose = sim.world.scene.body_pose(sim.chaser)?;
    let ring_pose = sim.world.scene.body_pose(sim.capture_ring)?;
    let soft_captured = sim.soft_pawls_latched.iter().all(|&l| l);

    build_station_mesh(
        &mut tris,
        station_pose,
        soft_captured,
        sim.seal_verified,
        sim.venting_air,
        sim.time_s,
    );
    build_solar_wing_inner(&mut tris, solar_port, 1.0);
    build_solar_wing_outer(&mut tris, solar_port_outer, 1.0);
    build_solar_wing_inner(&mut tris, solar_starboard, -1.0);
    build_solar_wing_outer(&mut tris, solar_starboard_outer, -1.0);
    build_dragon_capsule_mesh(&mut tris, sim, chaser_pose, ring_pose);
    Ok(tris)
}

/// Smooth 16-step Rayleigh LEO Earth atmosphere limb + 420-star color-temperature starfield.
pub fn paint_backdrop(painter: &egui::Painter, camera: &Camera, rect: Rect) {
    painter.rect_filled(rect, 0.0, Color32::from_rgb(3, 5, 11));

    let limb_y = rect.bottom() + (camera.pitch as f32 * 240.0) + rect.height() * 0.56;
    let limb_r = rect.width().max(rect.height()) * 1.18;
    let limb_center = Pos2::new(rect.center().x, limb_y + limb_r * 0.40);
    for i in (1..=16).rev() {
        let t = i as f32 / 16.0;
        let dr = (t * t) * 76.0;
        let falloff = (1.0 - t).powf(2.2);
        let r_col = (24.0 + 165.0 * falloff) as u8;
        let g_col = (90.0 + 148.0 * falloff) as u8;
        let b_col = (190.0 + 65.0 * falloff) as u8;
        let alpha = (6.0 + 34.0 * falloff) as u8;
        painter.circle_filled(
            limb_center,
            limb_r + dr,
            Color32::from_rgba_unmultiplied(r_col, g_col, b_col, alpha),
        );
    }
    painter.circle_filled(
        limb_center,
        limb_r + 1.8,
        Color32::from_rgba_unmultiplied(195, 238, 255, 110),
    );
    painter.circle_filled(limb_center, limb_r, Color32::from_rgb(10, 30, 66));
    painter.circle_filled(
        limb_center,
        (limb_r - 14.0).max(10.0),
        Color32::from_rgb(8, 24, 54),
    );
    painter.circle_filled(
        limb_center,
        (limb_r - 42.0).max(10.0),
        Color32::from_rgb(6, 18, 42),
    );
    painter.circle_filled(
        limb_center,
        (limb_r - 95.0).max(10.0),
        Color32::from_rgb(4, 13, 32),
    );

    let (eye, _) = camera.ray(rect.center(), rect);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_f = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64)
    };
    for _ in 0..420 {
        let u = next_f() * 2.0 - 1.0;
        let phi = next_f() * TAU;
        let r = (1.0 - u * u).sqrt();
        let dir = Vec3::new(r * phi.cos(), u, r * phi.sin());
        let (pos, depth) = camera.project(eye + dir * 200.0, rect);
        if depth > 1.0 && rect.contains(pos) {
            if pos.distance(limb_center) < limb_r + 2.0 {
                continue;
            }
            let mag = next_f().powi(3);
            let bright = (85.0 + 170.0 * mag) as u8;
            let tint = next_f();
            let color = if tint < 0.18 {
                Color32::from_rgb(bright, bright.saturating_sub(24), bright.saturating_sub(52))
            } else if tint > 0.78 {
                Color32::from_rgb(bright.saturating_sub(36), bright.saturating_sub(12), bright)
            } else {
                Color32::from_rgb(bright, bright, bright)
            };
            let radius = if mag > 0.78 {
                1.8
            } else if mag > 0.45 {
                1.3
            } else {
                0.9
            };
            painter.circle_filled(pos, radius, color);
        }
    }
}

/// Clean V-bar optical alignment centerline + 0.5 m range ticks + subtle approach trail.
pub fn paint_orbital_guides(
    painter: &egui::Painter,
    camera: &Camera,
    rect: Rect,
    sim: &SpaceDockingSim,
) {
    let Ok(port) = sim.station_port_pose() else {
        return;
    };
    let axis = port.rotation * Vec3::X;
    let p0 = port.position;
    let p1 = p0 + axis * 4.5;
    let (s0, d0) = camera.project(p0, rect);
    let (s1, d1) = camera.project(p1, rect);
    if d0 > 0.3 && d1 > 0.3 {
        painter.line_segment(
            [s0, s1],
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(92, 232, 184, 72)),
        );
    }
    let perp = port.rotation * Vec3::Y * 0.10;
    for m in 1..=4 {
        let c = p0 + axis * m as f64;
        let (a, da) = camera.project(c - perp, rect);
        let (b, db) = camera.project(c + perp, rect);
        if da > 0.3 && db > 0.3 {
            painter.line_segment(
                [a, b],
                Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(92, 232, 184, 85)),
            );
        }
    }
    let len = sim.chaser_trail.len();
    if len >= 2 {
        for i in 1..len {
            let frac = i as f32 / len as f32;
            let (a, da) = camera.project(sim.chaser_trail[i - 1], rect);
            let (b, db) = camera.project(sim.chaser_trail[i], rect);
            if da > 0.4 && db > 0.4 {
                painter.line_segment(
                    [a, b],
                    Stroke::new(
                        1.1_f32,
                        Color32::from_rgba_unmultiplied(120, 215, 255, (frac * 95.0) as u8),
                    ),
                );
            }
        }
    }
}

/// Draws the interactive 3D mouse grab/nudge elastic force tether on top of the scene.
pub fn paint_grab_tether(
    painter: &egui::Painter,
    camera: &Camera,
    rect: Rect,
    sim: &SpaceDockingSim,
) {
    let Some(grab) = sim.user_grab else {
        return;
    };
    let Ok(bp) = sim.world.scene.body_pose(grab.body) else {
        return;
    };
    let p_anchor = bp.transform_point(grab.local_point);
    let (sa, da) = camera.project(p_anchor, rect);
    let (st, dt) = camera.project(grab.target_world, rect);
    if da > 0.2 && dt > 0.2 {
        painter.line_segment(
            [sa, st],
            Stroke::new(2.4_f32, Color32::from_rgb(255, 168, 48)),
        );
        painter.circle_filled(sa, 5.5, Color32::from_rgb(255, 200, 82));
        painter.circle_stroke(
            st,
            7.0,
            Stroke::new(2.0_f32, Color32::from_rgb(255, 140, 48)),
        );
        let mid = Pos2::new(0.5 * (sa.x + st.x) + 10.0, 0.5 * (sa.y + st.y) - 10.0);
        painter.text(
            mid,
            egui::Align2::LEFT_CENTER,
            format!("NUDGE {:.1} kN", grab.applied_force_n * 1e-3),
            egui::FontId::monospace(12.5),
            Color32::from_rgb(255, 208, 96),
        );
    }
}
