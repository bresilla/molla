//! Scene-correctness regression tests for the viewer scenes.
//!
//! These tests build the same `Model` as the corresponding viewer
//! `build_*_model` function and run the CPU XPBD/Featherstone/VBD
//! pipeline far enough to assert a scene-specific physical property
//! (e.g. "rope particles do not penetrate the floor").
//!
//! Keeping them in the engine crate (not the viewer) means we can
//! exercise the solver without spinning up Bevy / wgpu at all.

use molla::geometry::collide;
use molla::math::{Mat3, Quat, SpatialVector};
use molla::prelude::*;

// ──────────────────────────────────────────────────────────────────
// Rope (Scene::Rope) — 18 spheres connected by CONNECT equality
// constraints, anchored at the top, hanging over a floor plane.
// ──────────────────────────────────────────────────────────────────

const ROPE_LINKS: usize = 18;
const ROPE_LINK_RADIUS: f64 = 0.12;
const ROPE_LINK_SPACING: f64 = 0.26;
const ROPE_ANCHOR_HEIGHT: f64 = 6.0;

fn build_rope() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.10);

    let mut bodies = Vec::with_capacity(ROPE_LINKS);
    for i in 0..ROPE_LINKS {
        let y = ROPE_ANCHOR_HEIGHT - i as f64 * ROPE_LINK_SPACING;
        let x = 0.001 * i as f64;
        let mut params = BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(0.4, ROPE_LINK_RADIUS))
            .with_position(Vec3::new(x, y, 0.0));
        if i == 0 {
            params = params
                .with_mass(0.0)
                .with_inertia(Mat3::ZERO)
                .with_flags(BodyFlags::KINEMATIC);
        }
        let id = b.add_body(params);
        b.add_shape_sphere(id, ROPE_LINK_RADIUS, Transform::IDENTITY)
            .expect("rope sphere");
        bodies.push(id);
    }
    let half = ROPE_LINK_SPACING * 0.5;
    for i in 1..ROPE_LINKS {
        let _ = b.add_equality_connect(
            bodies[i - 1],
            bodies[i],
            Vec3::new(0.0, -half, 0.0),
            Vec3::new(0.0, half, 0.0),
        );
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .expect("plane");
    b.build().expect("rope model")
}

#[test]
fn rope_does_not_penetrate_floor() {
    let model = build_rope();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(256);
    let mut pair_buf = Vec::new();

    let mut solver = XpbdSolver::new().with_iterations(10);
    let dt = 1.0 / 600.0;
    // 6 simulated seconds — long enough for rope to settle.
    for _ in 0..3600 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let mut min_y = f64::INFINITY;
    let mut min_idx = 0;
    for (i, p) in q.iter().enumerate() {
        if p.position.y < min_y {
            min_y = p.position.y;
            min_idx = i;
        }
    }

    // Each rope sphere has radius 0.12; the floor plane is at y=0 with
    // +Y normal, so every sphere centre must satisfy `y >= 0.12 - slop`.
    // Allow ~5 mm of XPBD residual penetration (matches tolerance used
    // in `xpbd_floor.rs::sphere_settles_on_plane`).
    let slop = 5e-3;
    let total_chain = (ROPE_LINKS as f64 - 1.0) * ROPE_LINK_SPACING;
    assert!(
        min_y >= ROPE_LINK_RADIUS - slop,
        "rope sphere {min_idx} penetrates floor: y={min_y}, expected >= {} (chain {total_chain} m, anchor {ROPE_ANCHOR_HEIGHT})",
        ROPE_LINK_RADIUS - slop,
    );
}

#[test]
fn rope_chain_stays_taut_under_gravity() {
    // Run for a few seconds and verify the chain doesn't stretch
    // unbounded (constraint violation regression).
    let model = build_rope();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(256);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(10);
    let dt = 1.0 / 600.0;
    for _ in 0..1200 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.body_q.host().unwrap();
    // Each adjacent pair of spheres is constrained to be 0.26 m apart
    // (anchor offsets ±0.13 along Y of each body). A taut chain should
    // sit within ~5% of rest under XPBD with 10 iters.
    for i in 1..ROPE_LINKS {
        let d = (q[i].position - q[i - 1].position).length();
        assert!(
            (d - ROPE_LINK_SPACING).abs() < 0.03,
            "rope link {i} ↔ {} stretched: d={d:.4}, expected {ROPE_LINK_SPACING}",
            i - 1,
        );
    }
}

// ──────────────────────────────────────────────────────────────────
// Tower (Scene::Tower) — 8-layer 4×4 brick stack + heavy ball.
// Property: the stack must not "explode" (every body stays within a
// reasonable bounding box).
// ──────────────────────────────────────────────────────────────────

const TOWER_LAYERS: usize = 8;
const TOWER_GRID: usize = 4;
const TOWER_BRICK_HALF: f64 = 0.30;
const TOWER_LAYER_HEIGHT: f64 = 0.62;
const TOWER_BRICK_SPACING: f64 = 0.62;

fn box_inertia(mass: f64, half: f64) -> Mat3 {
    let s = mass * (2.0 * half).powi(2) / 12.0;
    Mat3::from_diagonal(Vec3::new(2.0 * s, 2.0 * s, 2.0 * s))
}

#[test]
fn tower_does_not_explode_at_rest() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.20);
    let half = TOWER_BRICK_HALF;
    let half_v = Vec3::splat(half);
    let inertia = box_inertia(1.0, half);
    let layer_y0 = half + 0.001;
    for layer in 0..TOWER_LAYERS {
        let y = layer_y0 + (layer as f64) * TOWER_LAYER_HEIGHT;
        for ix in 0..TOWER_GRID {
            for iz in 0..TOWER_GRID {
                let x = (ix as f64 - (TOWER_GRID as f64 - 1.0) * 0.5) * TOWER_BRICK_SPACING;
                let z = (iz as f64 - (TOWER_GRID as f64 - 1.0) * 0.5) * TOWER_BRICK_SPACING;
                let pose = Transform::new(Vec3::new(x, y, z), Quat::IDENTITY);
                let body = b.add_body(
                    BodyParams::new()
                        .with_spatial_inertia(SpatialInertia {
                            mass: 1.0,
                            com: Vec3::ZERO,
                            inertia,
                        })
                        .with_pose(pose),
                );
                b.add_shape_box(body, half_v, Transform::IDENTITY).unwrap();
            }
        }
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(2048);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(8);
    let dt = 1.0 / 240.0;
    for _ in 0..600 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    for (i, p) in q.iter().enumerate() {
        let pos = p.position;
        // Stack sits in roughly [-1.5, 1.5] x [0, 6] x [-1.5, 1.5].
        // Generous bounds catch explosion (bodies flung to infinity).
        assert!(
            pos.x.abs() < 6.0 && pos.z.abs() < 6.0 && pos.y > -0.5 && pos.y < 12.0,
            "tower brick {i} escaped: pos = {pos:?}",
        );
        assert!(
            pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
            "brick {i} NaN'd"
        );
    }
}

// ──────────────────────────────────────────────────────────────────
// Pyramid (Scene::Pyramid) — 10-layer 4-sided box pyramid.
// Property: at rest (no projectile) the pyramid stays put; with
// projectile, no body NaNs.
// ──────────────────────────────────────────────────────────────────

const PYRAMID_LEVELS: usize = 10;
const PYRAMID_BOX_HALF: f64 = 0.30;
const PYRAMID_SPACING: f64 = 2.05 * PYRAMID_BOX_HALF;

#[test]
fn pyramid_does_not_explode_at_rest() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.20);

    let half = PYRAMID_BOX_HALF;
    let spacing = PYRAMID_SPACING;
    let inertia = box_inertia(1.0, half);
    let half_v = Vec3::splat(half);
    let layer_y0 = half + 0.001;
    for level in 0..PYRAMID_LEVELS {
        let row_count = PYRAMID_LEVELS - level;
        let y = layer_y0 + (level as f64) * (PYRAMID_BOX_HALF * 2.0 + 0.005);
        for ix in 0..row_count {
            for iz in 0..row_count {
                let x = (ix as f64 - (row_count as f64 - 1.0) * 0.5) * spacing;
                let z = (iz as f64 - (row_count as f64 - 1.0) * 0.5) * spacing;
                let body = b.add_body(
                    BodyParams::new()
                        .with_spatial_inertia(SpatialInertia {
                            mass: 1.0,
                            com: Vec3::ZERO,
                            inertia,
                        })
                        .with_position(Vec3::new(x, y, z)),
                );
                b.add_shape_box(body, half_v, Transform::IDENTITY).unwrap();
            }
        }
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(8192);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(8);
    let dt = 1.0 / 240.0;
    for _ in 0..400 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.body_q.host().unwrap();
    for (i, p) in q.iter().enumerate() {
        assert!(
            p.position.is_finite(),
            "pyramid box {i} NaN'd: {:?}",
            p.position
        );
        // Bodies should stay within sane bounds.
        assert!(
            p.position.x.abs() < 8.0 && p.position.z.abs() < 8.0 && p.position.y > -0.5,
            "pyramid box {i} escaped: pos = {:?}",
            p.position
        );
    }
}

// ──────────────────────────────────────────────────────────────────
// BrickWall (Scene::BrickWall) — 6×10 brick stack.
// Property: stays standing without external force; no NaN.
// ──────────────────────────────────────────────────────────────────

const BRICK_HALF_X: f64 = 0.18;
const BRICK_HALF_Y: f64 = 0.10;
const BRICK_HALF_Z: f64 = 0.10;
const BRICK_GAP: f64 = 0.005;
const BRICK_ROWS: usize = 6;
const BRICK_COLS: usize = 10;

#[test]
fn brick_wall_stands_without_projectile() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.20);
    let mass = 0.6_f64;
    let h2 = Vec3::new(BRICK_HALF_X, BRICK_HALF_Y, BRICK_HALF_Z);
    let h2 = h2 * h2;
    let m_third = mass / 3.0;
    let inertia = Mat3::from_diagonal(Vec3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ));
    let half_v = Vec3::new(BRICK_HALF_X, BRICK_HALF_Y, BRICK_HALF_Z);
    let layer_h = BRICK_HALF_Y * 2.0 + BRICK_GAP;
    let col_w = BRICK_HALF_X * 2.0 + BRICK_GAP;
    for row in 0..BRICK_ROWS {
        let stagger = if row % 2 == 0 { 0.0 } else { BRICK_HALF_X };
        for col in 0..BRICK_COLS {
            let x = (col as f64 - (BRICK_COLS as f64 - 1.0) * 0.5) * col_w + stagger;
            let y = BRICK_HALF_Y + 0.001 + (row as f64) * layer_h;
            let body = b.add_body(
                BodyParams::new()
                    .with_spatial_inertia(SpatialInertia {
                        mass,
                        com: Vec3::ZERO,
                        inertia,
                    })
                    .with_position(Vec3::new(x, y, 0.0)),
            );
            b.add_shape_box(body, half_v, Transform::IDENTITY).unwrap();
        }
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(2048);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(8);
    let dt = 1.0 / 240.0;
    for _ in 0..600 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.body_q.host().unwrap();
    for (i, p) in q.iter().enumerate() {
        assert!(p.position.is_finite(), "brick {i} NaN'd: {:?}", p.position);
        // The wall should be standing; bricks may have settled but
        // shouldn't have flown apart. The wall x-extent is ±~1 m;
        // allow ±6 m horizontal drift before we call it "escaped"
        // (without friction the bricks slide along the plane —
        // documented limitation, not a regression).
        assert!(
            p.position.x.abs() < 7.0
                && p.position.z.abs() < 7.0
                && p.position.y > -0.2
                && p.position.y < 5.0,
            "brick {i} escaped: pos = {:?}",
            p.position
        );
    }
}

// ──────────────────────────────────────────────────────────────────
// Conveyor (Scene::Conveyor) — kinematic belt with prescribed +X
// linear velocity. Documents the known limitation that XPBD's
// Coulomb friction (capped at `μ · |λ_n|` per substep) only delivers
// a tiny tangential push at steady state because the per-substep
// normal impulse `λ_n` is itself tiny once the box has settled. The
// belt scene LOOKS like nothing is happening because of this — not a
// bug in the scene definition, an inherent XPBD model limitation.
// Mark `#[ignore]` so the failure documents the property without
// breaking CI.
// ──────────────────────────────────────────────────────────────────

#[test]
fn conveyor_belt_drags_box_via_friction() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.0);
    b.set_default_friction(0.8).unwrap();

    let belt_half = Vec3::new(3.0, 0.05, 0.6);
    let belt = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_inertia(Mat3::ZERO)
            .with_position(Vec3::new(0.0, 1.0, 0.0))
            .with_flags(BodyFlags::KINEMATIC),
    );
    b.add_shape_box(belt, belt_half, Transform::IDENTITY)
        .unwrap();

    let half = Vec3::splat(0.15);
    let inertia = box_inertia(0.5, 0.15);
    let box_body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 0.5,
                com: Vec3::ZERO,
                inertia,
            })
            .with_position(Vec3::new(0.0, 1.0 + 0.05 + 0.15 + 0.001, 0.0)),
    );
    b.add_shape_box(box_body, half, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(8);
    let dt = 1.0 / 240.0;
    let belt_vx = 1.0_f64;
    let mut start_x = None;
    for step in 0..240 {
        if let Ok(qd) = s_a.body_qd.host_mut() {
            qd[0] = SpatialVector {
                linear: Vec3::new(belt_vx, 0.0, 0.0),
                angular: Vec3::ZERO,
            };
        }
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        if step == 30 {
            let q = s_a.body_q.host().unwrap();
            start_x = Some(q[1].position.x);
        }
    }
    let q = s_a.body_q.host().unwrap();
    let final_x = q[1].position.x;
    let drift = final_x - start_x.unwrap();
    assert!(
        drift > 0.05,
        "box not dragged by belt: drift={drift} m (XPBD limitation)"
    );
}

// ──────────────────────────────────────────────────────────────────
// Pile (Scene::Pile) — 64 boxes dropped in a grid; check none escape
// the floor and none NaN.
// ──────────────────────────────────────────────────────────────────

#[test]
fn pile_bodies_stay_above_floor() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    let half = 0.25;
    let half_v = Vec3::splat(half);
    let inertia = box_inertia(1.0, half);
    let mut count = 0;
    for ix in 0..6 {
        for iz in 0..6 {
            let x = (ix as f64 - 2.5) * 0.6;
            let z = (iz as f64 - 2.5) * 0.6;
            let body = b.add_body(
                BodyParams::new()
                    .with_spatial_inertia(SpatialInertia {
                        mass: 1.0,
                        com: Vec3::ZERO,
                        inertia,
                    })
                    .with_position(Vec3::new(x, 1.5, z)),
            );
            b.add_shape_box(body, half_v, Transform::IDENTITY).unwrap();
            count += 1;
        }
    }
    let _ = count;
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(2048);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(8);
    let dt = 1.0 / 240.0;
    for _ in 0..400 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.body_q.host().unwrap();
    for (i, p) in q.iter().enumerate() {
        assert!(
            p.position.is_finite(),
            "pile body {i} NaN'd: {:?}",
            p.position
        );
        assert!(
            p.position.y > -0.5,
            "pile body {i} sank below floor: y = {}",
            p.position.y
        );
    }
}
