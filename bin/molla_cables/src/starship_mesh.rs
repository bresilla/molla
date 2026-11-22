//! 3D PBR Geometry & Sky/Terrain Renderer for Full-Scale SpaceX Starship,
//! Super Heavy Booster (33 Raptors + 4 Titanium Grid Fins + Hot-Stage Ring),
//! and Starbase Mechazilla Launch/Catch Tower (140 m + Articulated Chopsticks).
//!
//! Renders centered at `render_origin` scaled by `RENDER_SCALE` (`0.04`, 1:25) so
//! 121 m tall vehicles, 140 m tower, and multi-kilometer flights sit right in the
//! precision sweet spot of `studio.wgsl`'s 0.05..100.0 depth buffer & shadow map!

use std::f64::consts::{PI, TAU};

use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
use molla_math::{Quat, Transform, Vec3};

use crate::draw::{Camera, Triangle};
use crate::starship_sim::{
    BOOSTER_CATCH_PIN_LOCAL_Y, BOOSTER_HALF_LEN, CHOPSTICK_CATCH_Y, HOTSTAGE_RING_H, OLM_DECK_Y,
    SHIP_HALF_LEN, SHIP_PAD_X, SHIP_PAD_Z, StarshipScenario, StarshipSim,
};
use molla_solvers::devices::PlanetAtmosphere;
use crate::studio::PlumeBillboard;

/// Visual world scale (`1 m SI = 0.04 studio units` -> 121 m stack = 4.84 units).
pub const RENDER_SCALE: f64 = 0.04;

const STEEL_304L: Color32 = Color32::from_rgb(206, 214, 224);
const HLS_WHITE: Color32 = Color32::from_rgb(232, 236, 242);
const STEEL_RING: Color32 = Color32::from_rgb(168, 178, 192);
const TPS_BLACK: Color32 = Color32::from_rgb(24, 27, 33);
const TITANIUM_FIN: Color32 = Color32::from_rgb(108, 114, 124);
const TOWER_STEEL: Color32 = Color32::from_rgb(84, 92, 102);
const TOWER_RED: Color32 = Color32::from_rgb(165, 58, 42);
const CONCRETE_PAD: Color32 = Color32::from_rgb(122, 124, 120);
const COAST_SAND: Color32 = Color32::from_rgb(112, 108, 92);
const MARS_REGOLITH: Color32 = Color32::from_rgb(154, 78, 48);
const MARS_PAD: Color32 = Color32::from_rgb(116, 62, 42);
const LUNAR_REGOLITH: Color32 = Color32::from_rgb(104, 106, 110);
const LUNAR_PAD: Color32 = Color32::from_rgb(78, 82, 88);
const RAPTOR_BELL: Color32 = Color32::from_rgb(56, 60, 68);
const RAPTOR_GLOW: Color32 = Color32::from_rgb(185, 205, 255);
const PIN_GOLD: Color32 = Color32::from_rgb(228, 182, 74);

fn to_render(p: Vec3, origin: Vec3) -> Vec3 {
    (p - origin) * RENDER_SCALE
}

fn push_tri(out: &mut Vec<Triangle>, a: Vec3, b: Vec3, c: Vec3, origin: Vec3, color: Color32, mat: f32) {
    let n = (b - a).cross(c - a);
    if n.length_squared() < 1e-12 {
        return;
    }
    out.push(Triangle {
        p: [to_render(a, origin), to_render(b, origin), to_render(c, origin)],
        color,
        material: mat,
        surface: None,
    });
}

fn push_quad(out: &mut Vec<Triangle>, a: Vec3, b: Vec3, c: Vec3, d: Vec3, origin: Vec3, color: Color32, mat: f32) {
    push_tri(out, a, b, c, origin, color, mat);
    push_tri(out, a, c, d, origin, color, mat);
}

fn push_box(out: &mut Vec<Triangle>, pose: Transform, center: Vec3, half: Vec3, origin: Vec3, color: Color32, mat: f32) {
    let c = |sx: f64, sy: f64, sz: f64| {
        pose.transform_point(center + Vec3::new(sx * half.x, sy * half.y, sz * half.z))
    };
    let p000 = c(-1.0, -1.0, -1.0);
    let p100 = c(1.0, -1.0, -1.0);
    let p110 = c(1.0, 1.0, -1.0);
    let p010 = c(-1.0, 1.0, -1.0);
    let p001 = c(-1.0, -1.0, 1.0);
    let p101 = c(1.0, -1.0, 1.0);
    let p111 = c(1.0, 1.0, 1.0);
    let p011 = c(-1.0, 1.0, 1.0);
    push_quad(out, p100, p110, p111, p101, origin, color, mat); // +X
    push_quad(out, p001, p011, p010, p000, origin, color, mat); // -X
    push_quad(out, p011, p111, p110, p010, origin, color, mat); // +Y
    push_quad(out, p000, p100, p101, p001, origin, color, mat); // -Y
    push_quad(out, p001, p101, p111, p011, origin, color, mat); // +Z
    push_quad(out, p100, p000, p010, p110, origin, color, mat); // -Z
}

#[allow(clippy::too_many_arguments)]
fn push_frustum_y(
    out: &mut Vec<Triangle>,
    pose: Transform,
    y0: f64,
    y1: f64,
    r0: f64,
    r1: f64,
    segs: usize,
    origin: Vec3,
    color_pos_x: Color32,
    color_neg_x: Color32,
    mat: f32,
    cap_bottom: bool,
    cap_top: bool,
) {
    let mut ring0 = Vec::with_capacity(segs);
    let mut ring1 = Vec::with_capacity(segs);
    for i in 0..segs {
        let a = (i as f64) * TAU / segs as f64;
        let u = Vec3::new(a.cos(), 0.0, a.sin());
        ring0.push(pose.transform_point(Vec3::new(0.0, y0, 0.0) + u * r0));
        ring1.push(pose.transform_point(Vec3::new(0.0, y1, 0.0) + u * r1));
    }
    for i in 0..segs {
        let next = (i + 1) % segs;
        let mid_a = ((i as f64) + 0.5) * TAU / segs as f64;
        // Heatshield TPS black tiles cover the windward `-X` half of Starship!
        let col = if mid_a.cos() < -0.05 { color_neg_x } else { color_pos_x };
        // Outward CCW around +Y: (r0[i], r1[i], r1[next], r0[next])
        push_quad(out, ring0[i], ring1[i], ring1[next], ring0[next], origin, col, mat);
    }
    if cap_bottom && r0 > 1e-4 {
        let c0 = pose.transform_point(Vec3::new(0.0, y0, 0.0));
        for i in 0..segs {
            let next = (i + 1) % segs;
            push_tri(out, c0, ring0[i], ring0[next], origin, color_pos_x, mat);
        }
    }
    if cap_top && r1 > 1e-4 {
        let c1 = pose.transform_point(Vec3::new(0.0, y1, 0.0));
        for i in 0..segs {
            let next = (i + 1) % segs;
            push_tri(out, c1, ring1[next], ring1[i], origin, color_pos_x, mat);
        }
    }
}

fn push_cylinder_seg(
    out: &mut Vec<Triangle>,
    pose: Transform,
    p0: Vec3,
    p1: Vec3,
    radius: f64,
    segs: usize,
    origin: Vec3,
    color: Color32,
    mat: f32,
) {
    let axis = p1 - p0;
    let len = axis.length();
    if len < 1e-6 {
        return;
    }
    let dir = axis / len;
    let helper = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = dir.cross(helper).normalize();
    let v = dir.cross(u).normalize();
    let mut r0 = Vec::with_capacity(segs);
    let mut r1 = Vec::with_capacity(segs);
    for i in 0..segs {
        let a = i as f64 * TAU / segs as f64;
        let off = (u * a.cos() + v * a.sin()) * radius;
        r0.push(pose.transform_point(p0 + off));
        r1.push(pose.transform_point(p1 + off));
    }
    for i in 0..segs {
        let next = (i + 1) % segs;
        push_quad(out, r0[i], r0[next], r1[next], r1[i], origin, color, mat);
    }
}

fn build_raptor_bells(
    out: &mut Vec<Triangle>,
    sim: &StarshipSim,
    stage_pose: Transform,
    engines: &[molla_solvers::devices::RocketEngineId],
    origin: Vec3,
) {
    for &eid in engines {
        let (Some(spec), Some(read)) = (
            sim.world.devices.rockets.engine_spec(eid),
            sim.world.devices.rockets.engine_reading(eid),
        ) else {
            continue;
        };
        let base_f = stage_pose.compose(&spec.frame);
        let gq = (Quat::from_rotation_z(read.gimbal_rad[0])
            * Quat::from_rotation_y(read.gimbal_rad[1]))
        .normalize();
        // Nozzle bell points along `-X` of gimbaled engine frame; orient local +Y along `-X`!
        let bell_pose = Transform::new(
            base_f.position,
            (base_f.rotation * gq * Quat::from_rotation_z(PI * 0.5)).normalize(),
        );
        let r_exit = 0.5 * spec.exit_diameter_m;
        let bell_len = spec.exit_diameter_m * 1.15;
        let col = if read.failed {
            TOWER_RED
        } else {
            RAPTOR_BELL
        };
        push_frustum_y(
            out,
            bell_pose,
            -bell_len,
            0.0,
            r_exit,
            r_exit * 0.38,
            10,
            origin,
            col,
            col,
            12.0,
            false,
            false,
        );
        if read.chamber_throttle > 0.05 {
            push_frustum_y(
                out,
                bell_pose,
                -bell_len - 0.15,
                -bell_len + 0.35,
                r_exit * 0.85,
                r_exit * 0.55,
                10,
                origin,
                RAPTOR_GLOW,
                RAPTOR_GLOW,
                14.0,
                true,
                false,
            );
        }
    }
}

fn build_booster_mesh(out: &mut Vec<Triangle>, sim: &StarshipSim, pose: Transform, origin: Vec3) {
    // Main 71 m stainless-steel cylindrical barrel (9.0 m diameter) with weld rings.
    push_frustum_y(
        out,
        pose,
        -BOOSTER_HALF_LEN,
        BOOSTER_HALF_LEN,
        4.5,
        4.5,
        28,
        origin,
        STEEL_304L,
        STEEL_304L,
        12.0,
        true,
        true,
    );
    // Dark ventilated hot-staging interstage ring on top of Booster!
    push_frustum_y(
        out,
        pose,
        BOOSTER_HALF_LEN,
        BOOSTER_HALF_LEN + HOTSTAGE_RING_H,
        4.52,
        4.52,
        24,
        origin,
        TPS_BLACK,
        TPS_BLACK,
        10.0,
        false,
        false,
    );
    // 2 Steel Mechazilla Catch Pins protruding on ±Z!
    for &sign_z in &[-1.0, 1.0] {
        push_box(
            out,
            pose,
            Vec3::new(0.0, BOOSTER_CATCH_PIN_LOCAL_Y, sign_z * 4.85),
            Vec3::new(0.55, 0.45, 0.65),
            origin,
            PIN_GOLD,
            12.0,
        );
    }
    // 4 Pivoting Titanium Grid Fins at y = +29.5 m!
    let fin_mounts = [
        (Vec3::new(4.5, 29.5, 0.0), Vec3::X, Vec3::new(1.5, 0.0, 0.0), Vec3::new(1.5, 0.25, 1.85)),
        (Vec3::new(-4.5, 29.5, 0.0), Vec3::X, Vec3::new(-1.5, 0.0, 0.0), Vec3::new(1.5, 0.25, 1.85)),
        (Vec3::new(0.0, 29.5, 4.5), Vec3::Z, Vec3::new(0.0, 0.0, 1.5), Vec3::new(1.85, 0.25, 1.5)),
        (Vec3::new(0.0, 29.5, -4.5), Vec3::Z, Vec3::new(0.0, 0.0, -1.5), Vec3::new(1.85, 0.25, 1.5)),
    ];
    for (idx, &(pivot, axis, offset, half)) in fin_mounts.iter().enumerate() {
        let defl = sim
            .world
            .devices
            .rocket_aero
            .surface_reading(sim.booster_grid_fins[idx])
            .map_or(0.0, |r| r.deflection_rad);
        let fin_pose = pose.compose(&Transform::new(pivot, Quat::from_axis_angle(axis, defl)));
        push_box(out, fin_pose, offset, half, origin, TITANIUM_FIN, 12.0);
    }
    // 33 Raptor 3 Engine Bells!
    build_raptor_bells(out, sim, pose, &sim.booster_engines, origin);
}

fn build_ship_mesh(out: &mut Vec<Triangle>, sim: &StarshipSim, pose: Transform, origin: Vec3) {
    let is_lunar = sim.scenario == StarshipScenario::LunarHlsDescent;
    let hull_col = if is_lunar { HLS_WHITE } else { STEEL_304L };
    let belly_col = if is_lunar { HLS_WHITE } else { TPS_BLACK };
    // Cylindrical 9 m barrel (-25..+7 m) + ogive nosecone (+7..+25 m) with black TPS heatshield on -X (or white HLS on Moon)!
    push_frustum_y(
        out,
        pose,
        -SHIP_HALF_LEN,
        7.0,
        4.5,
        4.5,
        28,
        origin,
        hull_col,
        belly_col,
        12.0,
        true,
        false,
    );
    let nose_rings: &[(f64, f64)] = &[
        (7.0, 4.5),
        (12.5, 4.15),
        (17.5, 3.30),
        (21.5, 2.05),
        (24.2, 0.85),
        (SHIP_HALF_LEN, 0.12),
    ];
    for pair in nose_rings.windows(2) {
        push_frustum_y(
            out,
            pose,
            pair[0].0,
            pair[1].0,
            pair[0].1,
            pair[1].1,
            24,
            origin,
            STEEL_304L,
            TPS_BLACK,
            12.0,
            false,
            pair[1].1 < 0.2,
        );
    }

    // 4 Real Actuated Body Flaps (visually pivoting live with aerodynamic deflection `δ`!)
    // [fwd_L(+Z), fwd_R(-Z), aft_L(+Z), aft_R(-Z)]
    let flap_geom = [
        (Vec3::new(0.0, 14.5, 3.65), Vec3::NEG_Y, Vec3::new(0.0, 0.0, 1.9), Vec3::new(0.18, 3.6, 1.9)),
        (Vec3::new(0.0, 14.5, -3.65), Vec3::Y, Vec3::new(0.0, 0.0, -1.9), Vec3::new(0.18, 3.6, 1.9)),
        (Vec3::new(0.0, -15.5, 4.45), Vec3::NEG_Y, Vec3::new(0.0, 0.0, 2.5), Vec3::new(0.22, 6.2, 2.5)),
        (Vec3::new(0.0, -15.5, -4.45), Vec3::Y, Vec3::new(0.0, 0.0, -2.5), Vec3::new(0.22, 6.2, 2.5)),
    ];
    for (idx, &(hinge_pt, hinge_axis, span_off, half)) in flap_geom.iter().enumerate() {
        let defl = sim
            .world
            .devices
            .rocket_aero
            .surface_reading(sim.ship_flaps[idx])
            .map_or(0.5, |r| r.deflection_rad);
        let fp = pose.compose(&Transform::new(hinge_pt, Quat::from_axis_angle(hinge_axis, defl)));
        push_box(out, fp, span_off, half, origin, TPS_BLACK, 10.0);
        push_box(
            out,
            fp,
            span_off + Vec3::new(0.20, 0.0, 0.0),
            Vec3::new(0.06, half.y * 0.92, half.z * 0.92),
            origin,
            STEEL_RING,
            12.0,
        );
    }

    // 6 Ship Raptor Bells (3 SL + 3 huge RVac bells!) + 4 Mid-Body HLS Lunar Thrusters on Lunar scenario!
    build_raptor_bells(out, sim, pose, &sim.ship_engines, origin);
    if sim.scenario == StarshipScenario::LunarHlsDescent {
        build_raptor_bells(out, sim, pose, &sim.ship_hls_thrusters, origin);
    }
}

fn build_starbase_tower_and_pad(out: &mut Vec<Triangle>, sim: &StarshipSim, origin: Vec3) {
    let ground_pose = Transform::IDENTITY;
    let ground_col = match sim.scenario {
        StarshipScenario::MarsSrpLanding => MARS_REGOLITH,
        StarshipScenario::LunarHlsDescent => LUNAR_REGOLITH,
        _ => COAST_SAND,
    };
    let pad_col = match sim.scenario {
        StarshipScenario::MarsSrpLanding => MARS_PAD,
        StarshipScenario::LunarHlsDescent => LUNAR_PAD,
        _ => CONCRETE_PAD,
    };
    let ring_col = match sim.scenario {
        StarshipScenario::LunarHlsDescent => PIN_GOLD,
        _ => TOWER_RED,
    };
    push_box(
        out,
        ground_pose,
        Vec3::new(110.0, -1.5, 0.0),
        Vec3::new(1_200.0, 1.5, 1_200.0),
        origin,
        ground_col,
        11.0,
    );
    // Starship Landing Zone Pad (`x = 260 m, z = 0 m`) with target bullseye ring!
    push_frustum_y(
        out,
        Transform::from_translation(Vec3::new(SHIP_PAD_X, 0.0, SHIP_PAD_Z)),
        0.0,
        0.55,
        36.0,
        35.0,
        24,
        origin,
        pad_col,
        pad_col,
        11.0,
        false,
        true,
    );
    push_frustum_y(
        out,
        Transform::from_translation(Vec3::new(SHIP_PAD_X, 0.56, SHIP_PAD_Z)),
        0.0,
        0.10,
        14.0,
        14.0,
        20,
        origin,
        ring_col,
        ring_col,
        11.0,
        false,
        true,
    );
    if !sim.scenario.is_earth() {
        return;
    }
    // OLM Concrete Apron & 20 m Orbital Launch Mount table (6 legs + ring deck).
    push_box(
        out,
        ground_pose,
        Vec3::new(-4.0, 0.3, 0.0),
        Vec3::new(34.0, 0.35, 34.0),
        origin,
        CONCRETE_PAD,
        11.0,
    );
    push_frustum_y(
        out,
        ground_pose,
        OLM_DECK_Y - 3.2,
        OLM_DECK_Y,
        9.5,
        9.5,
        18,
        origin,
        TOWER_STEEL,
        TOWER_STEEL,
        12.0,
        true,
        true,
    );
    for k in 0..6 {
        let a = k as f64 * (TAU / 6.0);
        push_cylinder_seg(
            out,
            ground_pose,
            Vec3::new(8.2 * a.cos(), 0.0, 8.2 * a.sin()),
            Vec3::new(7.4 * a.cos(), OLM_DECK_Y - 2.8, 7.4 * a.sin()),
            1.1,
            8,
            origin,
            CONCRETE_PAD,
            11.0,
        );
    }

    // 140 m Mechazilla Steel Truss Tower at `x = -14 m, z = 0 m`!
    let tx = -14.0;
    let half_w = 5.5;
    let corners = [
        Vec3::new(tx - half_w, 0.0, -half_w),
        Vec3::new(tx + half_w, 0.0, -half_w),
        Vec3::new(tx + half_w, 0.0, half_w),
        Vec3::new(tx - half_w, 0.0, half_w),
    ];
    for c in corners {
        push_cylinder_seg(
            out,
            ground_pose,
            c,
            c + Vec3::new(0.0, 140.0, 0.0),
            0.55,
            6,
            origin,
            TOWER_STEEL,
            12.0,
        );
    }
    let bays = 9;
    let bay_h = 140.0 / bays as f64;
    for b in 0..bays {
        let y0 = b as f64 * bay_h;
        let y1 = y0 + bay_h;
        for f in 0..4 {
            let ca = corners[f];
            let cb = corners[(f + 1) % 4];
            push_cylinder_seg(
                out,
                ground_pose,
                ca + Vec3::Y * y1,
                cb + Vec3::Y * y1,
                0.30,
                5,
                origin,
                TOWER_STEEL,
                12.0,
            );
            let (p0, p1) = if (b + f) % 2 == 0 {
                (ca + Vec3::Y * y0, cb + Vec3::Y * y1)
            } else {
                (cb + Vec3::Y * y0, ca + Vec3::Y * y1)
            };
            push_cylinder_seg(out, ground_pose, p0, p1, 0.26, 5, origin, TOWER_STEEL, 12.0);
        }
    }

    // 2 Articulated Mechazilla Chopstick Catch Arms rendered at LIVE Featherstone body poses!
    for &arm_h in &[sim.chopstick_left, sim.chopstick_right] {
        if let Ok(ap) = sim.world.scene.body_pose(arm_h) {
            push_box(
                out,
                ap,
                Vec3::new(5.5, 0.0, 0.0),
                Vec3::new(11.5, 1.35, 0.85),
                origin,
                TOWER_STEEL,
                12.0,
            );
            push_box(
                out,
                ap,
                Vec3::new(6.0, 1.45, 0.0),
                Vec3::new(10.0, 0.25, 0.55),
                origin,
                PIN_GOLD,
                12.0,
            );
        }
    }
    // Catch carriage block riding the tower face at y = CHOPSTICK_CATCH_Y:
    push_box(
        out,
        ground_pose,
        Vec3::new(tx + half_w + 0.8, CHOPSTICK_CATCH_Y, 0.0),
        Vec3::new(1.5, 4.5, 6.2),
        origin,
        TOWER_RED,
        12.0,
    );
}

pub fn build_scene_mesh(sim: &StarshipSim, render_origin: Vec3) -> molla_core::Result<Vec<Triangle>> {
    let mut tris = Vec::with_capacity(9_500);
    let bp = sim.world.scene.body_pose(sim.booster)?;
    let sp = sim.world.scene.body_pose(sim.ship)?;

    // Only emit tower/pad geometry when within 2.5 km of render_origin so depth precision stays crisp.
    if render_origin.y < 2_500.0 {
        build_starbase_tower_and_pad(&mut tris, sim, render_origin);
    }
    if sim.scenario.is_earth() {
        build_booster_mesh(&mut tris, sim, bp, render_origin);
    }
    build_ship_mesh(&mut tris, sim, sp, render_origin);
    Ok(tris)
}

pub fn build_plume_billboards(sim: &StarshipSim, render_origin: Vec3) -> Vec<PlumeBillboard> {
    sim.plumes
        .iter()
        .filter_map(|p| {
            let u = (p.age / p.max_age).clamp(0.0, 1.0) as f32;
            let fade = (1.0 - u).powf(1.4);
            let op = p.base_opacity * fade;
            (op > 0.01).then_some(PlumeBillboard {
                pos: (p.pos - render_origin) * RENDER_SCALE,
                vel: p.vel * RENDER_SCALE,
                radius: p.radius * RENDER_SCALE,
                stretch: p.stretch,
                opacity: op,
                age_frac: u,
                seed: p.seed,
                kind: p.kind,
            })
        })
        .collect()
}

/// Planetary sky backdrop: Earth coastal-to-space blue, Mars butterscotch CO₂ dust sky,
/// or pure starlit Lunar vacuum black!
pub fn paint_sky_backdrop(
    painter: &egui::Painter,
    rect: Rect,
    camera: &Camera,
    altitude_m: f64,
    atmosphere: PlanetAtmosphere,
) {
    let space_frac = match atmosphere {
        PlanetAtmosphere::Vacuum => 1.0_f32,
        PlanetAtmosphere::MarsCospar => ((altitude_m - 1_500.0) / 18_000.0).clamp(0.0, 1.0) as f32,
        PlanetAtmosphere::EarthUs76 => ((altitude_m - 2_000.0) / 28_000.0).clamp(0.0, 1.0) as f32,
    };
    let lerp_u8 = |a: f32, b: f32, t: f32| (a + (b - a) * t).clamp(0.0, 255.0) as u8;
    let bands = 14;
    for i in 0..bands {
        let t0 = i as f32 / bands as f32;
        let y0 = rect.top() + t0 * rect.height();
        let y1 = rect.top() + (i + 1) as f32 / bands as f32 * rect.height();
        let (r, g, b) = match atmosphere {
            PlanetAtmosphere::Vacuum => (3_u8, 4_u8, 8_u8),
            PlanetAtmosphere::MarsCospar => (
                lerp_u8(54.0 + 136.0 * t0.powf(1.4), 12.0 + 28.0 * t0.powf(1.8), space_frac),
                lerp_u8(28.0 + 78.0 * t0.powf(1.5), 7.0 + 14.0 * t0.powf(1.8), space_frac),
                lerp_u8(22.0 + 42.0 * t0.powf(1.6), 10.0 + 12.0 * t0.powf(1.8), space_frac),
            ),
            PlanetAtmosphere::EarthUs76 => (
                lerp_u8(16.0 + 96.0 * t0.powf(1.6), 4.0 + 18.0 * t0.powf(2.0), space_frac),
                lerp_u8(48.0 + 115.0 * t0.powf(1.5), 7.0 + 26.0 * t0.powf(2.0), space_frac),
                lerp_u8(108.0 + 108.0 * t0.powf(1.4), 16.0 + 52.0 * t0.powf(1.8), space_frac),
            ),
        };
        painter.rect_filled(
            Rect::from_min_max(Pos2::new(rect.left(), y0), Pos2::new(rect.right(), y1 + 1.0)),
            0.0,
            Color32::from_rgb(r, g, b),
        );
    }
    // Stars emerge above 6 km on Earth/Mars, and shine bright at all altitudes on the Moon!
    if space_frac > 0.15 {
        let (eye, _) = camera.ray(rect.center(), rect);
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut nf = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f64) / (u64::MAX as f64)
        };
        let a_star = ((space_frac - 0.15) / 0.85).clamp(0.0, 1.0);
        for _ in 0..220 {
            let u = nf() * 2.0 - 1.0;
            let phi = nf() * TAU;
            let r = (1.0 - u * u).sqrt();
            let dir = Vec3::new(r * phi.cos(), u, r * phi.sin());
            let (pos, depth) = camera.project(eye + dir * 80.0, rect);
            if depth > 1.0 && rect.contains(pos) {
                let bright = (180.0 * a_star * nf() as f32) as u8;
                painter.circle_filled(pos, 1.2, Color32::from_rgba_unmultiplied(230, 240, 255, bright));
            }
        }
    }
}

/// Paints flight trajectory trails & active 3D mouse nudge tether.
pub fn paint_overlays(
    painter: &egui::Painter,
    camera: &Camera,
    rect: Rect,
    sim: &StarshipSim,
    render_origin: Vec3,
) {
    let draw_trail = |pts: &std::collections::VecDeque<Vec3>, col: Color32| {
        let mut prev: Option<Pos2> = None;
        for &p in pts {
            let (s, d) = camera.project((p - render_origin) * RENDER_SCALE, rect);
            if d > 0.1 {
                if let Some(p0) = prev {
                    painter.line_segment([p0, s], Stroke::new(1.8_f32, col));
                }
                prev = Some(s);
            } else {
                prev = None;
            }
        }
    };
    draw_trail(&sim.booster_trail, Color32::from_rgba_unmultiplied(255, 185, 86, 150));
    draw_trail(&sim.ship_trail, Color32::from_rgba_unmultiplied(115, 220, 255, 165));

    if let Some(nudge) = sim.user_nudge
        && let Ok(pose) = sim.world.scene.body_pose(nudge.body)
    {
        let p_w = pose.transform_point(nudge.local_point);
        let (a, da) = camera.project((p_w - render_origin) * RENDER_SCALE, rect);
        let (b, db) = camera.project((nudge.target_world - render_origin) * RENDER_SCALE, rect);
        if da > 0.1 && db > 0.1 {
            let col = Color32::from_rgb(255, 186, 72);
            painter.line_segment([a, b], Stroke::new(3.0_f32, col));
            painter.circle_filled(a, 5.0, col);
            painter.circle_stroke(b, 8.0, Stroke::new(2.0_f32, col));
            painter.text(
                b + egui::vec2(12.0, -10.0),
                egui::Align2::LEFT_CENTER,
                format!("NUDGE {:.2} MN", nudge.force_n * 1e-6),
                egui::FontId::monospace(12.0),
                col,
            );
        }
    }
}
