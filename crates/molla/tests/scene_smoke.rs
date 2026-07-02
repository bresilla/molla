//! Exhaustive smoke-test coverage for ALL 46 viewer scenes.
//!
//! Each scene's `build_*_model` is mirrored inline here — we don't depend
//! on `molla_viewer` because the viewer crate pulls in Bevy / wgpu /
//! egui and tests live in `crates/molla` for fast iteration.
//!
//! Goal: catch slow-burn bugs (NaN over time, particle escape, buffer
//! corruption) by stepping the engine for ~1 simulated second on CPU.
//! Per-scene physical correctness is the job of `scene_correctness.rs`;
//! this file only asserts:
//!   1. No NaN/Inf in any body or particle position/velocity at the END.
//!   2. No body/particle has escaped a sane world bound (~±100 m).
//!   3. Body count + particle count haven't changed (no silent state
//!      corruption).
//!
//! Scenes that need GPU only (MPM, FastSoftImpact's CCD path) or that
//! aren't physics-stepped at all (DiffArm, MeshTools, …) get a "build
//! succeeds + initial state is sane" check instead of a step loop.

use molla::geometry::collide;
use molla::math::{Mat3, Quat, SpatialVector};
use molla::prelude::*;
use molla::sim::Heightfield as MollaHeightfield;

// ──────────────────────────────────────────────────────────────────────
// Inertia helpers — duplicated from `molla_viewer/src/main.rs`.
// ──────────────────────────────────────────────────────────────────────

fn box_inertia(mass: f64, half: f64) -> Mat3 {
    let h2 = half * half;
    let s = mass / 3.0;
    Mat3::from_diagonal(Vec3::new(s * (h2 + h2), s * (h2 + h2), s * (h2 + h2)))
}

fn box_inertia_xyz(mass: f64, hx: f64, hy: f64, hz: f64) -> Mat3 {
    let s = mass / 3.0;
    let ixx = s * (hy * hy + hz * hz);
    let iyy = s * (hx * hx + hz * hz);
    let izz = s * (hx * hx + hy * hy);
    Mat3::from_diagonal(Vec3::new(ixx, iyy, izz))
}

fn capsule_inertia(mass: f64, half_height: f64, radius: f64) -> Mat3 {
    let r2 = radius * radius;
    let h2 = half_height * half_height;
    let i_y = 0.5 * mass * r2;
    let i_perp = mass * (h2 / 3.0 + r2 / 4.0);
    Mat3::from_diagonal(Vec3::new(i_perp, i_y, i_perp))
}

// ──────────────────────────────────────────────────────────────────────
// Shared post-step assertion helper.
// ──────────────────────────────────────────────────────────────────────

/// World bounds — anything beyond this is treated as "escaped".
const WORLD_BOUND: f64 = 100.0;

#[track_caller]
fn assert_no_nan_or_escape(state: &State, label: &str) {
    let body_q = state.body_q.host().expect("body_q host");
    for (i, t) in body_q.iter().enumerate() {
        let p = t.position;
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "{label}: body {i} position not finite: {p:?}",
        );
        assert!(
            p.x.abs() < WORLD_BOUND
                && p.y > -WORLD_BOUND
                && p.y < WORLD_BOUND
                && p.z.abs() < WORLD_BOUND,
            "{label}: body {i} escaped world bounds (±{WORLD_BOUND}m): {p:?}",
        );
    }
    let body_qd = state.body_qd.host().expect("body_qd host");
    for (i, v) in body_qd.iter().enumerate() {
        let lin = v.linear;
        let ang = v.angular;
        assert!(
            lin.x.is_finite() && lin.y.is_finite() && lin.z.is_finite(),
            "{label}: body {i} linear velocity not finite: {lin:?}",
        );
        assert!(
            ang.x.is_finite() && ang.y.is_finite() && ang.z.is_finite(),
            "{label}: body {i} angular velocity not finite: {ang:?}",
        );
    }
    let particle_q = state.particle_q.host().expect("particle_q host");
    for (i, p) in particle_q.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "{label}: particle {i} position not finite: {p:?}",
        );
        assert!(
            p.x.abs() < WORLD_BOUND
                && p.y > -WORLD_BOUND
                && p.y < WORLD_BOUND
                && p.z.abs() < WORLD_BOUND,
            "{label}: particle {i} escaped world bounds (±{WORLD_BOUND}m): {p:?}",
        );
    }
    let particle_qd = state.particle_qd.host().expect("particle_qd host");
    for (i, v) in particle_qd.iter().enumerate() {
        assert!(
            v.x.is_finite() && v.y.is_finite() && v.z.is_finite(),
            "{label}: particle {i} velocity not finite: {v:?}",
        );
    }
}

#[track_caller]
fn assert_counts_match(model: &Model, state: &State, label: &str) {
    let body_q = state.body_q.host().expect("body_q");
    assert_eq!(
        body_q.len(),
        model.body_count,
        "{label}: body_q.len ({}) != model.body_count ({})",
        body_q.len(),
        model.body_count,
    );
    let particle_q = state.particle_q.host().expect("particle_q");
    assert_eq!(
        particle_q.len(),
        model.particle_count,
        "{label}: particle_q.len ({}) != model.particle_count ({})",
        particle_q.len(),
        model.particle_count,
    );
}

// ──────────────────────────────────────────────────────────────────────
// Step loop wrappers — XPBD / Featherstone / VBD on CPU.
// ──────────────────────────────────────────────────────────────────────

/// Standard simulated time = 1 s @ 240 Hz = 240 substeps.
const SMOKE_STEPS: usize = 240;
const SMOKE_DT: f64 = 1.0 / 240.0;

/// Run XPBD on CPU and assert the canonical smoke conditions.
fn run_xpbd_smoke(model: &Model, label: &str) {
    let mut s_a = model.state().expect("state_a");
    let mut s_b = model.state().expect("state_b");
    let control = model.control();
    let mut contacts = Contacts::with_capacity(8192);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    for _ in 0..SMOKE_STEPS {
        collide(model, &s_a, &mut contacts, &mut pair_buf).expect("collide");
        Solver::step(
            &mut solver,
            model,
            &s_a,
            &control,
            &contacts,
            SMOKE_DT,
            &mut s_b,
        )
        .expect("xpbd step");
        std::mem::swap(&mut s_a, &mut s_b);
    }
    assert_no_nan_or_escape(&s_a, label);
    assert_counts_match(model, &s_a, label);
}

/// Run Featherstone on CPU.
fn run_featherstone_smoke(model: &Model, label: &str) {
    run_featherstone_smoke_with(model, label, 0.0, false);
}

/// Run Featherstone on CPU, with custom damping + rigorous-D-solve flag.
fn run_featherstone_smoke_with(model: &Model, label: &str, damping: f64, rigorous_d: bool) {
    let mut s_a = model.state().expect("state_a");
    let mut s_b = model.state().expect("state_b");
    let control = model.control();
    let mut contacts = Contacts::with_capacity(2048);
    let mut pair_buf = Vec::new();
    let mut solver = FeatherstoneSolver::new_cpu()
        .with_damping(damping)
        .with_rigorous_d_solve(rigorous_d);
    for _ in 0..SMOKE_STEPS {
        collide(model, &s_a, &mut contacts, &mut pair_buf).expect("collide");
        Solver::step(
            &mut solver,
            model,
            &s_a,
            &control,
            &contacts,
            SMOKE_DT,
            &mut s_b,
        )
        .expect("featherstone step");
        std::mem::swap(&mut s_a, &mut s_b);
    }
    assert_no_nan_or_escape(&s_a, label);
    assert_counts_match(model, &s_a, label);
}

/// Run VBD on CPU. Soft scenes get fewer substeps (60 = 0.25 s) since
/// the particle workload is heavier.
fn run_vbd_smoke(model: &Model, label: &str, steps: usize) {
    let mut s_a = model.state().expect("state_a");
    let mut s_b = model.state().expect("state_b");
    let control = model.control();
    let mut contacts = Contacts::with_capacity(2048);
    let mut pair_buf = Vec::new();
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(4)
        .with_contact_stiffness(1.0e5);
    for _ in 0..steps {
        collide(model, &s_a, &mut contacts, &mut pair_buf).expect("collide");
        Solver::step(
            &mut solver,
            model,
            &s_a,
            &control,
            &contacts,
            SMOKE_DT,
            &mut s_b,
        )
        .expect("vbd step");
        std::mem::swap(&mut s_a, &mut s_b);
    }
    assert_no_nan_or_escape(&s_a, label);
    assert_counts_match(model, &s_a, label);
}

/// Host-driven scenes: just verify the build succeeded + the initial
/// state has no NaN/Inf.
fn run_build_only(model: &Model, label: &str) {
    let s = model.state().expect("state");
    assert_no_nan_or_escape(&s, label);
    assert_counts_match(model, &s, label);
}

// ══════════════════════════════════════════════════════════════════════
// Group A — Original viewer scenes (Pile, Rope, Tower, Cloth, …)
// ══════════════════════════════════════════════════════════════════════

// ── Scene 01: Pile ───────────────────────────────────────────────────

const PILE_GRID_N: usize = 6; // smaller than the viewer's 12 for test speed
const PILE_SPACING: f64 = 0.95;
const PILE_SPHERE_RADIUS: f64 = 0.40;
const PILE_BOX_HALF: f64 = 0.40;
const PILE_CAPSULE_HALF_H: f64 = 0.40;
const PILE_CAPSULE_RADIUS: f64 = 0.22;
const PILE_DROP_HEIGHT: f64 = 4.0;
const PILE_DROP_BAND: f64 = 6.0;

fn build_pile_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    let mut rng_state: u64 = 0x9E3779B97F4A7C15;
    fn next_unit(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (*s >> 11) as f64 / (1u64 << 53) as f64;
        v * 2.0 - 1.0
    }
    fn random_rotation(s: &mut u64) -> Quat {
        let axis = Vec3::new(next_unit(s), next_unit(s), next_unit(s)).normalize_or_zero();
        let axis = if axis.length_squared() < 1e-9 {
            Vec3::Y
        } else {
            axis
        };
        let angle = next_unit(s) * std::f64::consts::PI;
        Quat::from_axis_angle(axis, angle)
    }
    for ix in 0..PILE_GRID_N {
        for iz in 0..PILE_GRID_N {
            let x = (ix as f64 - (PILE_GRID_N as f64 - 1.0) * 0.5) * PILE_SPACING
                + 0.15 * next_unit(&mut rng_state);
            let z = (iz as f64 - (PILE_GRID_N as f64 - 1.0) * 0.5) * PILE_SPACING
                + 0.15 * next_unit(&mut rng_state);
            let y = PILE_DROP_HEIGHT
                + ((ix + iz) as f64) / ((PILE_GRID_N + PILE_GRID_N) as f64) * PILE_DROP_BAND
                + 0.1 * next_unit(&mut rng_state);
            let pos = Vec3::new(x, y, z);
            let rot = random_rotation(&mut rng_state);
            match (ix + iz) % 3 {
                0 => {
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia::sphere(1.0, PILE_SPHERE_RADIUS))
                            .with_position(pos),
                    );
                    b.add_shape_sphere(body, PILE_SPHERE_RADIUS, Transform::IDENTITY)
                        .unwrap();
                }
                1 => {
                    let half = Vec3::splat(PILE_BOX_HALF);
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia {
                                mass: 1.0,
                                com: Vec3::ZERO,
                                inertia: box_inertia(1.0, PILE_BOX_HALF),
                            })
                            .with_pose(Transform::new(pos, rot)),
                    );
                    b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
                }
                _ => {
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia {
                                mass: 1.0,
                                com: Vec3::ZERO,
                                inertia: capsule_inertia(
                                    1.0,
                                    PILE_CAPSULE_HALF_H,
                                    PILE_CAPSULE_RADIUS,
                                ),
                            })
                            .with_pose(Transform::new(pos, rot)),
                    );
                    b.add_shape_capsule(
                        body,
                        PILE_CAPSULE_HALF_H,
                        PILE_CAPSULE_RADIUS,
                        Transform::IDENTITY,
                    )
                    .unwrap();
                }
            }
        }
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("pile model")
}

#[test]
fn scene_01_pile_smoke() {
    run_xpbd_smoke(&build_pile_smoke(), "01 Pile");
}

// ── Scene 02: Rope ───────────────────────────────────────────────────

const ROPE_LINKS: usize = 18;
const ROPE_LINK_RADIUS: f64 = 0.12;
const ROPE_LINK_SPACING: f64 = 0.26;
const ROPE_ANCHOR_HEIGHT: f64 = 6.0;

fn build_rope_smoke() -> Model {
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
            .unwrap();
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
        .unwrap();
    b.build().expect("rope model")
}

#[test]
fn scene_02_rope_smoke() {
    run_xpbd_smoke(&build_rope_smoke(), "02 Rope");
}

// ── Scene 03: Tower ──────────────────────────────────────────────────

const TOWER_LAYERS: usize = 8;
const TOWER_GRID: usize = 4;
const TOWER_BRICK_HALF: f64 = 0.30;
const TOWER_BRICK_SPACING: f64 = 0.65;
const TOWER_LAYER_HEIGHT: f64 = 0.605;
const TOWER_BALL_RADIUS: f64 = 1.0;
const TOWER_BALL_MASS: f64 = 50.0;
const TOWER_BALL_HEIGHT: f64 = 14.0;

fn build_tower_smoke() -> Model {
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
    let ball = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(TOWER_BALL_MASS, TOWER_BALL_RADIUS))
            .with_position(Vec3::new(0.0, TOWER_BALL_HEIGHT, 0.0)),
    );
    b.add_shape_sphere(ball, TOWER_BALL_RADIUS, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("tower model")
}

#[test]
fn scene_03_tower_smoke() {
    run_xpbd_smoke(&build_tower_smoke(), "03 Tower");
}

// ── Scene 04: Cloth (XPBD particle cloth) ────────────────────────────

const CLOTH_GRID: u32 = 16; // smaller than the viewer's 36 for speed
const CLOTH_SIZE: f64 = 1.8;
const CLOTH_TOP_Y: f64 = 5.0;
const CLOTH_PARTICLE_MASS: f64 = 0.05;
const CLOTH_PARTICLE_RADIUS: f64 = 0.02;
const CLOTH_COMPLIANCE: f64 = 1.5e-5;
const CLOTH_MAX_STRETCH: f64 = 0.05;
const CLOTH_SPHERE_RADIUS: f64 = 0.7;
const CLOTH_SPHERE_Y: f64 = 1.2;

fn build_cloth_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);
    let nx = CLOTH_GRID;
    let ny = CLOTH_GRID;
    let size = CLOTH_SIZE;
    let origin = Vec3::new(-size * 0.5, CLOTH_TOP_Y, -size * 0.5);
    let u_axis = Vec3::new(size, 0.0, 0.0);
    let v_axis = Vec3::new(0.0, 0.0, size);
    let _ = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        nx,
        ny,
        CLOTH_PARTICLE_MASS,
        CLOTH_PARTICLE_RADIUS,
        CLOTH_COMPLIANCE,
        CLOTH_MAX_STRETCH,
        WorldId(0),
    );
    b.add_shape_sphere(
        BodyId::NONE,
        CLOTH_SPHERE_RADIUS,
        Transform::from_translation(Vec3::new(0.0, CLOTH_SPHERE_Y, 0.0)),
    )
    .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("cloth model")
}

#[test]
fn scene_04_cloth_smoke() {
    run_xpbd_smoke(&build_cloth_smoke(), "04 Cloth");
}

// ── Scene 05: Lidar (kinematic-only) ─────────────────────────────────

fn build_lidar_smoke() -> Model {
    let mut b = ModelBuilder::new();
    let kin_at = |pos: Vec3| {
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(pos, Quat::IDENTITY))
            .with_flags(BodyFlags::KINEMATIC)
    };
    let body = b.add_body(kin_at(Vec3::new(4.5, 1.2, 0.0)));
    b.add_shape_sphere(body, 1.2, Transform::IDENTITY).unwrap();
    let body = b.add_body(kin_at(Vec3::new(-3.5, 1.5, 2.0)));
    b.add_shape_box(body, Vec3::new(0.7, 1.5, 0.7), Transform::IDENTITY)
        .unwrap();
    let body = b.add_body(kin_at(Vec3::new(0.0, 0.6, 6.0)));
    b.add_shape_box(body, Vec3::new(3.0, 0.6, 0.4), Transform::IDENTITY)
        .unwrap();
    let body = b.add_body(kin_at(Vec3::new(3.0, 1.0, -3.5)));
    b.add_shape_capsule(body, 1.0, 0.4, Transform::IDENTITY)
        .unwrap();
    let body = b.add_body(kin_at(Vec3::new(-3.0, 0.8, -3.0)));
    b.add_shape_cylinder(body, 0.8, 0.5, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("lidar model")
}

#[test]
fn scene_05_lidar_smoke() {
    run_xpbd_smoke(&build_lidar_smoke(), "05 Lidar");
}

// ── Scene 06: IMU (single bouncing sphere) ───────────────────────────

fn build_imu_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.0);
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.4))
            .with_position(Vec3::new(0.0, 4.0, 0.0)),
    );
    b.add_shape_sphere(body, 0.4, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("imu model")
}

#[test]
fn scene_06_imu_smoke() {
    run_xpbd_smoke(&build_imu_smoke(), "06 Imu");
}

// ── Scene 07: Contact (3-body stack) ─────────────────────────────────

fn build_contact_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.1);
    b.set_linear_damping(0.5);
    let h = Vec3::new(1.0, 0.25, 1.0);
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 4.0,
                com: Vec3::ZERO,
                inertia: box_inertia(4.0, 0.5),
            })
            .with_position(Vec3::new(0.0, 0.4, 0.0)),
    );
    b.add_shape_box(body, h, Transform::IDENTITY).unwrap();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.5, 0.4))
            .with_position(Vec3::new(0.2, 1.6, 0.0)),
    );
    b.add_shape_sphere(body, 0.4, Transform::IDENTITY).unwrap();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: capsule_inertia(1.0, 0.4, 0.25),
            })
            .with_position(Vec3::new(-0.1, 2.6, 0.05)),
    );
    b.add_shape_capsule(body, 0.4, 0.25, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("contact model")
}

#[test]
fn scene_07_contact_smoke() {
    run_xpbd_smoke(&build_contact_smoke(), "07 Contact");
}

// ── Scene 08: Arm (6-link revolute, Featherstone) ───────────────────

const ARM_LINKS: usize = 6;
const ARM_LINK_LEN: f64 = 0.6;
const ARM_LINK_RADIUS: f64 = 0.06;
const ARM_BASE_HEIGHT: f64 = 5.0;
const ARM_LINK_MASS: f64 = 0.5;

fn build_arm_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.1);
    let half_h = ARM_LINK_LEN * 0.5;
    let radius = ARM_LINK_RADIUS;
    let inertia = capsule_inertia(ARM_LINK_MASS, half_h, radius);
    b.begin_articulation();
    let mut prev_body = BodyId::NONE;
    let mut next_pos = Vec3::new(0.0, ARM_BASE_HEIGHT - half_h, 0.0);
    for i in 0..ARM_LINKS {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: ARM_LINK_MASS,
                    com: Vec3::ZERO,
                    inertia,
                })
                .with_position(next_pos),
        );
        b.add_shape_capsule(body, half_h, radius, Transform::IDENTITY)
            .unwrap();
        let x_p = if i == 0 {
            let flip = Quat::from_axis_angle(Vec3::X, std::f64::consts::PI);
            Transform::new(Vec3::new(0.0, ARM_BASE_HEIGHT, 0.0), flip)
        } else {
            Transform::from_translation(Vec3::new(0.0, half_h, 0.0))
        };
        let x_c = Transform::from_translation(Vec3::new(0.0, -half_h, 0.0));
        let axis = if i == 0 { Vec3::Z } else { Vec3::X };
        b.add_joint_revolute(prev_body, body, x_p, x_c, axis)
            .unwrap();
        prev_body = body;
        next_pos -= Vec3::new(0.0, ARM_LINK_LEN, 0.0);
    }
    b.build().expect("arm model")
}

#[test]
fn scene_08_arm_smoke() {
    run_featherstone_smoke(&build_arm_smoke(), "08 Arm");
}

// ── Scene 09: Shapes (cone / hull / heightfield / SDF / ellipsoid) ──

const SHAPES_FIELD_N: u32 = 12; // smaller than viewer's 16 for speed
const SHAPES_FIELD_CELL: f64 = 0.6;
const SHAPES_TORUS_MAJOR: f64 = 1.0;
const SHAPES_TORUS_MINOR: f64 = 0.25;
const SHAPES_TORUS_Y: f64 = 3.5;

fn build_shapes_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);
    let n = SHAPES_FIELD_N;
    let cell = SHAPES_FIELD_CELL;
    let half = (n as f64 - 1.0) * cell * 0.5;
    let mut heights = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 - (n as f64 - 1.0) * 0.5) * cell;
            let z = (j as f64 - (n as f64 - 1.0) * 0.5) * cell;
            heights.push(0.25 * ((x * 0.5).sin() + (z * 0.5).cos()));
        }
    }
    let hf = MollaHeightfield {
        nx: n,
        nz: n,
        cell_size: cell,
        origin: Vec3::new(-half, 0.0, -half),
        heights,
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(BodyId::NONE, field_id, Transform::IDENTITY)
        .unwrap();
    let sdf_id = b.add_sdf(SdfShape::Torus {
        major: SHAPES_TORUS_MAJOR,
        minor: SHAPES_TORUS_MINOR,
    });
    b.add_shape_sdf(
        BodyId::NONE,
        sdf_id,
        Transform::from_translation(Vec3::new(0.0, SHAPES_TORUS_Y, 0.0)),
    )
    .unwrap();
    // 3 cones.
    for (i, pos) in [
        Vec3::new(-1.5, 8.0, -1.0),
        Vec3::new(0.5, 9.0, 0.5),
        Vec3::new(1.8, 8.5, -0.5),
    ]
    .iter()
    .enumerate()
    {
        let half_h = 0.35;
        let radius = 0.25;
        let mass = 0.8;
        let inertia = Mat3::from_diagonal(Vec3::new(
            mass * (radius * radius * 0.25 + (2.0 * half_h_f64()).powi(2) / 12.0),
            mass * radius * radius * 0.5,
            mass * (radius * radius * 0.25 + (2.0 * half_h_f64()).powi(2) / 12.0),
        ));
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass,
                    com: Vec3::ZERO,
                    inertia,
                })
                .with_pose(Transform::new(
                    *pos,
                    Quat::from_axis_angle(Vec3::Z, (i as f64) * 0.3 + 0.4),
                )),
        );
        b.add_shape_cone(body, half_h, radius, Transform::IDENTITY)
            .unwrap();
    }
    // 3 convex hulls.
    for (i, pos) in [
        Vec3::new(-2.5, 10.0, 1.5),
        Vec3::new(0.0, 10.5, -2.0),
        Vec3::new(2.5, 9.5, 1.5),
    ]
    .iter()
    .enumerate()
    {
        let half = Vec3::new(0.30, 0.40, 0.30);
        let verts = vec![
            Vec3::new(-half.x, -half.y, -half.z),
            Vec3::new(half.x, -half.y, -half.z),
            Vec3::new(-half.x, half.y, -half.z),
            Vec3::new(half.x, half.y, -half.z),
            Vec3::new(-half.x, -half.y, half.z),
            Vec3::new(half.x, -half.y, half.z),
            Vec3::new(-half.x, half.y, half.z),
            Vec3::new(half.x, half.y, half.z),
        ];
        let mass = 0.6;
        let h2 = half * half;
        let m_third = mass / 3.0;
        let inertia = Mat3::from_diagonal(Vec3::new(
            m_third * (h2.y + h2.z),
            m_third * (h2.x + h2.z),
            m_third * (h2.x + h2.y),
        ));
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass,
                    com: Vec3::ZERO,
                    inertia,
                })
                .with_pose(Transform::new(
                    *pos,
                    Quat::from_axis_angle(Vec3::X, (i as f64) * 0.5),
                )),
        );
        let hull_id = b.add_hull(verts);
        b.add_shape_convex_hull(body, hull_id, Transform::IDENTITY)
            .unwrap();
    }
    // 3 spheres.
    for pos in &[
        Vec3::new(-0.8, 11.0, 0.0),
        Vec3::new(0.8, 11.5, 1.5),
        Vec3::new(-1.5, 11.5, -1.5),
    ] {
        let radius = 0.3;
        let mass = 1.0;
        let i_v = (2.0 / 5.0) * mass * radius * radius;
        let inertia = Mat3::from_diagonal(Vec3::new(i_v, i_v, i_v));
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass,
                    com: Vec3::ZERO,
                    inertia,
                })
                .with_position(*pos),
        );
        b.add_shape_sphere(body, radius, Transform::IDENTITY)
            .unwrap();
    }
    // 1 ellipsoid.
    {
        let half_axes = Vec3::new(0.45, 0.20, 0.35);
        let mass = 0.9;
        let h2 = half_axes * half_axes;
        let m_fifth = mass / 5.0;
        let inertia = Mat3::from_diagonal(Vec3::new(
            m_fifth * (h2.y + h2.z),
            m_fifth * (h2.x + h2.z),
            m_fifth * (h2.x + h2.y),
        ));
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass,
                    com: Vec3::ZERO,
                    inertia,
                })
                .with_pose(Transform::new(
                    Vec3::new(1.2, 12.0, 1.2),
                    Quat::from_axis_angle(Vec3::Z, 0.6),
                )),
        );
        b.add_shape_ellipsoid(body, half_axes, Transform::IDENTITY)
            .unwrap();
    }
    b.build().expect("shapes model")
}

fn half_h_f64() -> f64 {
    0.35
}

#[test]
fn scene_09_shapes_smoke() {
    run_xpbd_smoke(&build_shapes_smoke(), "09 Shapes");
}

// ── Scene 10: GpuIkArm (kinematic IK; build-only) ───────────────────

fn build_gpu_ik_arm_smoke() -> Model {
    let mut b = ModelBuilder::new();
    let mass = 1.0;
    let half = Vec3::new(0.04, 0.04, 0.04);
    let h2 = half * half;
    let m_third = mass / 3.0;
    let inertia = Mat3::from_diagonal(Vec3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ));
    b.begin_articulation();
    let axes = [
        Vec3::Z,
        Vec3::Y,
        Vec3::Z,
        Vec3::Y,
        Vec3::Z,
        Vec3::Y,
        Vec3::Z,
    ];
    let link_lens = [0.333_f64, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107];
    let mut prev = BodyId::NONE;
    for i in 0..7 {
        let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass,
            com: Vec3::ZERO,
            inertia,
        }));
        let x_p = if i == 0 {
            Transform::from_translation(Vec3::new(0.0, 1.0, 0.0))
        } else {
            Transform::from_translation(Vec3::new(0.0, 0.0, link_lens[i - 1]))
        };
        let x_c = Transform::IDENTITY;
        b.add_joint_revolute(prev, body, x_p, x_c, axes[i]).unwrap();
        prev = body;
    }
    b.build().expect("gpu ik arm model")
}

#[test]
fn scene_10_gpu_ik_arm_smoke() {
    // Host-driven (kinematic IK); the viewer skips `step_simulation` for
    // this scene. Just verify the build + initial state are sane.
    let m = build_gpu_ik_arm_smoke();
    run_build_only(&m, "10 GpuIkArm");
}

// ── Scene 11: GpuNarrowPhase (mixed pile) ────────────────────────────

fn build_gpu_narrow_phase_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let n_side = 4i32;
    for ix in 0..n_side {
        for iz in 0..n_side {
            let x = (ix as f64 - 1.5) * 0.7;
            let z = (iz as f64 - 1.5) * 0.7;
            let y = 3.0 + ((ix + iz) as f64) * 0.3;
            let kind = (ix * n_side + iz) % 4;
            match kind {
                0 => {
                    let half = Vec3::splat(0.25);
                    let mass = 0.8;
                    let inertia = box_inertia(mass, 0.25);
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia {
                                mass,
                                com: Vec3::ZERO,
                                inertia,
                            })
                            .with_position(Vec3::new(x, y, z)),
                    );
                    b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
                }
                1 => {
                    let r = 0.25;
                    let mass = 1.0;
                    let i_v = (2.0 / 5.0) * mass * r * r;
                    let inertia = Mat3::from_diagonal(Vec3::new(i_v, i_v, i_v));
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia {
                                mass,
                                com: Vec3::ZERO,
                                inertia,
                            })
                            .with_position(Vec3::new(x, y, z)),
                    );
                    b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
                }
                2 => {
                    let half_h: f64 = 0.30;
                    let r: f64 = 0.20;
                    let mass: f64 = 0.7;
                    let inertia = Mat3::from_diagonal(Vec3::new(
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                        mass * r * r * 0.5,
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                    ));
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia {
                                mass,
                                com: Vec3::ZERO,
                                inertia,
                            })
                            .with_position(Vec3::new(x, y, z)),
                    );
                    b.add_shape_cylinder(body, half_h, r, Transform::IDENTITY)
                        .unwrap();
                }
                _ => {
                    let half_h: f64 = 0.30;
                    let r: f64 = 0.20;
                    let mass: f64 = 0.7;
                    let inertia = Mat3::from_diagonal(Vec3::new(
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                        mass * r * r * 0.5,
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                    ));
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia {
                                mass,
                                com: Vec3::ZERO,
                                inertia,
                            })
                            .with_position(Vec3::new(x, y, z)),
                    );
                    b.add_shape_cone(body, half_h, r, Transform::IDENTITY)
                        .unwrap();
                }
            }
        }
    }
    b.build().expect("gpu narrow phase model")
}

#[test]
fn scene_11_gpu_narrow_phase_smoke() {
    run_xpbd_smoke(&build_gpu_narrow_phase_smoke(), "11 GpuNarrowPhase");
}

// ── Scene 12: Camera (settles a few primitives) ─────────────────────

fn build_camera_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.20);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let sphere_specs = [
        (Vec3::new(-1.6, 1.5, 0.0), 0.5),
        (Vec3::new(0.0, 1.5, 0.0), 0.5),
        (Vec3::new(1.6, 1.5, 0.0), 0.5),
    ];
    for (pos, r) in sphere_specs {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, r))
                .with_position(pos),
        );
        b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
    }
    {
        let half = Vec3::new(0.55, 0.45, 0.55);
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 2.0,
                    com: Vec3::ZERO,
                    inertia: box_inertia(2.0, half.max_element()),
                })
                .with_position(Vec3::new(0.0, 1.0, -1.6)),
        );
        b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
    }
    {
        let half_h: f64 = 0.45;
        let radius: f64 = 0.25;
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.8,
                    com: Vec3::ZERO,
                    inertia: capsule_inertia(0.8, half_h, radius),
                })
                .with_position(Vec3::new(-2.4, 1.0, 1.6)),
        );
        b.add_shape_capsule(body, half_h, radius, Transform::IDENTITY)
            .unwrap();
    }
    {
        let half_h: f64 = 0.6;
        let radius: f64 = 0.45;
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 1.2,
                    com: Vec3::ZERO,
                    inertia: capsule_inertia(1.2, half_h, radius),
                })
                .with_position(Vec3::new(2.4, 1.2, 1.6)),
        );
        b.add_shape_cone(body, half_h, radius, Transform::IDENTITY)
            .unwrap();
    }
    b.build().expect("camera model")
}

#[test]
fn scene_12_camera_smoke() {
    run_xpbd_smoke(&build_camera_smoke(), "12 Camera");
}

// ── Scene 13: Actuator (1 revolute capsule) ──────────────────────────

const ACTUATOR_BASE_HEIGHT: f64 = 1.5;
const ACTUATOR_LINK_HALF_H: f64 = 0.4;
const ACTUATOR_LINK_RADIUS: f64 = 0.05;
const ACTUATOR_LINK_MASS: f64 = 0.5;

fn build_actuator_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    let inertia = capsule_inertia(
        ACTUATOR_LINK_MASS,
        ACTUATOR_LINK_HALF_H,
        ACTUATOR_LINK_RADIUS,
    );
    b.begin_articulation();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: ACTUATOR_LINK_MASS,
                com: Vec3::ZERO,
                inertia,
            })
            .with_position(Vec3::new(
                0.0,
                ACTUATOR_BASE_HEIGHT - ACTUATOR_LINK_HALF_H,
                0.0,
            )),
    );
    b.add_shape_capsule(
        body,
        ACTUATOR_LINK_HALF_H,
        ACTUATOR_LINK_RADIUS,
        Transform::IDENTITY,
    )
    .unwrap();
    let flip = Quat::from_axis_angle(Vec3::X, std::f64::consts::PI);
    let x_p = Transform::new(Vec3::new(0.0, ACTUATOR_BASE_HEIGHT, 0.0), flip);
    let x_c = Transform::from_translation(Vec3::new(0.0, -ACTUATOR_LINK_HALF_H, 0.0));
    b.add_joint_revolute(BodyId::NONE, body, x_p, x_c, Vec3::Z)
        .unwrap();
    b.build().expect("actuator model")
}

#[test]
fn scene_13_actuator_smoke() {
    run_featherstone_smoke(&build_actuator_smoke(), "13 Actuator");
}

// ── Scene 14: Quadrotor (FREE root + 4 rotors) ──────────────────────

const QUAD_ARM: f64 = 0.25;
const QUAD_FRAME_MASS: f64 = 1.0;
const QUAD_ROTOR_MASS: f64 = 0.05;
const QUAD_ROTOR_RADIUS: f64 = 0.06;
const QUAD_INITIAL_HEIGHT: f64 = 1.0;

fn build_quadrotor_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    b.set_angular_damping(0.6);
    b.begin_articulation();
    let half = Vec3::new(0.18, 0.04, 0.18);
    let m = QUAD_FRAME_MASS;
    let i_frame = Mat3::from_diagonal(Vec3::new(
        m * (half.y * half.y + half.z * half.z) / 3.0,
        m * (half.x * half.x + half.z * half.z) / 3.0,
        m * (half.x * half.x + half.y * half.y) / 3.0,
    ));
    let frame = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: m,
                com: Vec3::ZERO,
                inertia: i_frame,
            })
            .with_position(Vec3::new(0.0, QUAD_INITIAL_HEIGHT, 0.0)),
    );
    b.add_shape_box(frame, half, Transform::IDENTITY).unwrap();
    b.add_joint_free(
        BodyId::NONE,
        frame,
        Transform::IDENTITY,
        Transform::IDENTITY,
    )
    .unwrap();
    let r2 = QUAD_ROTOR_RADIUS * QUAD_ROTOR_RADIUS;
    let i_rotor = Mat3::from_diagonal(Vec3::new(
        QUAD_ROTOR_MASS * r2 * 0.25,
        QUAD_ROTOR_MASS * r2 * 0.5,
        QUAD_ROTOR_MASS * r2 * 0.25,
    ));
    for off in [
        Vec3::new(QUAD_ARM, 0.0, 0.0),
        Vec3::new(-QUAD_ARM, 0.0, 0.0),
        Vec3::new(0.0, 0.0, QUAD_ARM),
        Vec3::new(0.0, 0.0, -QUAD_ARM),
    ] {
        let rotor = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: QUAD_ROTOR_MASS,
                    com: Vec3::ZERO,
                    inertia: i_rotor,
                })
                .with_position(Vec3::new(off.x, QUAD_INITIAL_HEIGHT + 0.02, off.z)),
        );
        b.add_shape_sphere(rotor, QUAD_ROTOR_RADIUS, Transform::IDENTITY)
            .unwrap();
        b.add_joint_fixed(
            frame,
            rotor,
            Transform::from_translation(off + Vec3::Y * 0.02),
            Transform::IDENTITY,
        )
        .unwrap();
    }
    b.build().expect("quadrotor model")
}

#[test]
fn scene_14_quadrotor_smoke() {
    run_featherstone_smoke(&build_quadrotor_smoke(), "14 Quadrotor");
}

// ── Scene 15: BroadPhase (sphere shower — small for speed) ──────────

fn build_broad_phase_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.20);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let nx: i32 = 8; // viewer uses 32×32×4; smaller for speed.
    let nz: i32 = 8;
    let ny: i32 = 2;
    let spacing: f64 = 0.8;
    let drop_height: f64 = 6.0;
    let mut h: u32 = 0x9E37_79B9;
    let mut next = || {
        h = h.wrapping_mul(2654435761).wrapping_add(0x12345678);
        h
    };
    for ix in 0..nx {
        for iz in 0..nz {
            for iy in 0..ny {
                let x = (ix as f64 - (nx as f64 - 1.0) * 0.5) * spacing;
                let z = (iz as f64 - (nz as f64 - 1.0) * 0.5) * spacing;
                let y = drop_height + iy as f64 * (spacing * 0.9);
                let r = 0.18 + (next() as f64 / u32::MAX as f64) * 0.14;
                let mass = 1.0;
                let i_v = (2.0 / 5.0) * mass * r * r;
                let inertia = Mat3::from_diagonal(Vec3::new(i_v, i_v, i_v));
                let body = b.add_body(
                    BodyParams::new()
                        .with_spatial_inertia(SpatialInertia {
                            mass,
                            com: Vec3::ZERO,
                            inertia,
                        })
                        .with_position(Vec3::new(x, y, z)),
                );
                b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
            }
        }
    }
    b.build().expect("broad phase model")
}

#[test]
fn scene_15_broad_phase_smoke() {
    run_xpbd_smoke(&build_broad_phase_smoke(), "15 BroadPhase");
}

// ── Scene 16: VbdCloth (cloth + sphere) ──────────────────────────────

const VBD_CLOTH_GRID: u32 = 16; // viewer uses 32; smaller for CPU speed
const VBD_CLOTH_SIZE: f64 = 1.8;
const VBD_CLOTH_TOP_Y: f64 = 4.5;
const VBD_CLOTH_PARTICLE_MASS: f64 = 0.04;
const VBD_CLOTH_PARTICLE_RADIUS: f64 = 0.025;
const VBD_CLOTH_COMPLIANCE: f64 = 5.0e-6;
const VBD_CLOTH_MAX_STRETCH: f64 = 0.10;
const VBD_CLOTH_SPHERE_RADIUS: f64 = 0.6;
const VBD_CLOTH_SPHERE_Y: f64 = 1.6;

fn build_vbd_cloth_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);
    let nx = VBD_CLOTH_GRID;
    let ny = VBD_CLOTH_GRID;
    let size = VBD_CLOTH_SIZE;
    let origin = Vec3::new(-size * 0.5, VBD_CLOTH_TOP_Y, -size * 0.5);
    let u_axis = Vec3::new(size, 0.0, 0.0);
    let v_axis = Vec3::new(0.0, 0.0, size);
    let _ = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        nx,
        ny,
        VBD_CLOTH_PARTICLE_MASS,
        VBD_CLOTH_PARTICLE_RADIUS,
        VBD_CLOTH_COMPLIANCE,
        VBD_CLOTH_MAX_STRETCH,
        WorldId(0),
    );
    b.add_shape_sphere(
        BodyId::NONE,
        VBD_CLOTH_SPHERE_RADIUS,
        Transform::from_translation(Vec3::new(0.0, VBD_CLOTH_SPHERE_Y, 0.0)),
    )
    .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("vbd cloth model")
}

#[test]
fn scene_16_vbd_cloth_smoke() {
    run_vbd_smoke(&build_vbd_cloth_smoke(), "16 VbdCloth", 60);
}

// ── Scene 17: Native double pendulum ─────────────────────────────────

const REFERENCE_PENDULUM_LINK_LEN: f64 = 0.5;
const REFERENCE_PENDULUM_LINK_RADIUS: f64 = 0.05;
const REFERENCE_PENDULUM_LINK_MASS: f64 = 0.6;
const REFERENCE_PENDULUM_BASE_HEIGHT: f64 = 4.0;

fn build_reference_pendulum_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    let half_h = REFERENCE_PENDULUM_LINK_LEN * 0.5;
    let radius = REFERENCE_PENDULUM_LINK_RADIUS;
    let inertia = capsule_inertia(REFERENCE_PENDULUM_LINK_MASS, half_h, radius);
    b.begin_articulation();
    let upper_pos = Vec3::new(0.0, REFERENCE_PENDULUM_BASE_HEIGHT - half_h, 0.0);
    let upper = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: REFERENCE_PENDULUM_LINK_MASS,
                com: Vec3::ZERO,
                inertia,
            })
            .with_position(upper_pos),
    );
    b.add_shape_capsule(upper, half_h, radius, Transform::IDENTITY)
        .unwrap();
    let shoulder_x_p = Transform::new(
        Vec3::new(0.0, REFERENCE_PENDULUM_BASE_HEIGHT, 0.0),
        Quat::from_axis_angle(Vec3::X, std::f64::consts::PI),
    );
    let shoulder_x_c = Transform::from_translation(Vec3::new(0.0, -half_h, 0.0));
    b.add_joint_revolute(BodyId::NONE, upper, shoulder_x_p, shoulder_x_c, Vec3::Z)
        .unwrap();
    let lower_pos = upper_pos - Vec3::new(0.0, REFERENCE_PENDULUM_LINK_LEN, 0.0);
    let lower = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: REFERENCE_PENDULUM_LINK_MASS,
                com: Vec3::ZERO,
                inertia,
            })
            .with_position(lower_pos),
    );
    b.add_shape_capsule(lower, half_h, radius, Transform::IDENTITY)
        .unwrap();
    let elbow_x_p = Transform::from_translation(Vec3::new(0.0, half_h, 0.0));
    let elbow_x_c = Transform::from_translation(Vec3::new(0.0, -half_h, 0.0));
    b.add_joint_revolute(upper, lower, elbow_x_p, elbow_x_c, Vec3::X)
        .unwrap();
    b.build().expect("reference pendulum model")
}

#[test]
fn scene_17_reference_pendulum_smoke() {
    run_featherstone_smoke(&build_reference_pendulum_smoke(), "17 Native pendulum");
}

// ── Scene 18: DiffArm (kinematic 2-link, host-driven) ───────────────

const DIFF_ARM_BASE_Y: f64 = 1.6;
const DIFF_ARM_LINK_RADIUS: f64 = 0.04;

fn build_diff_arm_smoke() -> Model {
    let mut b = ModelBuilder::new();
    let half1 = 0.45 * 0.5; // l1=0.45 (DiffArm::default)
    let half2 = 0.45 * 0.5; // l2=0.45
    let upper = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(
                Vec3::new(0.0, DIFF_ARM_BASE_Y - half1, 0.0),
                Quat::IDENTITY,
            ))
            .with_flags(BodyFlags::KINEMATIC),
    );
    b.add_shape_capsule(upper, half1, DIFF_ARM_LINK_RADIUS, Transform::IDENTITY)
        .unwrap();
    let lower = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_pose(Transform::new(
                Vec3::new(0.0, DIFF_ARM_BASE_Y - 0.9 - half2, 0.0),
                Quat::IDENTITY,
            ))
            .with_flags(BodyFlags::KINEMATIC),
    );
    b.add_shape_capsule(lower, half2, DIFF_ARM_LINK_RADIUS, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("diff arm model")
}

#[test]
fn scene_18_diff_arm_smoke() {
    // Pure host-driven — viewer skips `step_simulation`; so do we.
    run_build_only(&build_diff_arm_smoke(), "18 DiffArm");
}

// ── Scene 19: DiffArm6 (6 kinematic capsules, host-driven) ──────────

fn build_diff_arm6_smoke() -> Model {
    let mut b = ModelBuilder::new();
    for _ in 0..6 {
        let body = b.add_body(
            BodyParams::new()
                .with_mass(0.0)
                .with_pose(Transform::new(Vec3::new(0.0, 0.5, 0.0), Quat::IDENTITY))
                .with_flags(BodyFlags::KINEMATIC),
        );
        b.add_shape_capsule(body, 0.06, 0.04, Transform::IDENTITY)
            .unwrap();
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("diff arm6 model")
}

#[test]
fn scene_19_diff_arm6_smoke() {
    run_build_only(&build_diff_arm6_smoke(), "19 DiffArm6");
}

// ── Scene 20: MeshTools (empty model — pure visualisation) ──────────

fn build_mesh_tools_smoke() -> Model {
    let b = ModelBuilder::new();
    b.build().expect("empty model")
}

#[test]
fn scene_20_mesh_tools_smoke() {
    run_build_only(&build_mesh_tools_smoke(), "20 MeshTools");
}

// ── Scene 21: NeuralPolicy (cart-pole) ──────────────────────────────

const CART_RAIL_Y: f64 = 1.0;
const CART_HALF_X: f64 = 0.15;
const CART_HALF_Y: f64 = 0.05;
const CART_HALF_Z: f64 = 0.10;
const CART_MASS: f64 = 1.0;
const POLE_HALF_LEN: f64 = 0.5;
const POLE_RADIUS: f64 = 0.03;
const POLE_MASS: f64 = 0.1;

fn build_neural_policy_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.02);
    b.begin_articulation();
    let cart_inertia = box_inertia_xyz(CART_MASS, CART_HALF_X, CART_HALF_Y, CART_HALF_Z);
    let cart = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: CART_MASS,
                com: Vec3::ZERO,
                inertia: cart_inertia,
            })
            .with_position(Vec3::new(0.0, CART_RAIL_Y, 0.0)),
    );
    b.add_shape_box(
        cart,
        Vec3::new(CART_HALF_X, CART_HALF_Y, CART_HALF_Z),
        Transform::IDENTITY,
    )
    .unwrap();
    let x_p_cart = Transform::from_translation(Vec3::new(0.0, CART_RAIL_Y, 0.0));
    b.add_joint_prismatic(BodyId::NONE, cart, x_p_cart, Transform::IDENTITY, Vec3::X)
        .unwrap();
    let pole_inertia = capsule_inertia(POLE_MASS, POLE_HALF_LEN, POLE_RADIUS);
    let pole = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: POLE_MASS,
                com: Vec3::ZERO,
                inertia: pole_inertia,
            })
            .with_position(Vec3::new(0.0, CART_RAIL_Y - POLE_HALF_LEN, 0.0)),
    );
    b.add_shape_capsule(pole, POLE_HALF_LEN, POLE_RADIUS, Transform::IDENTITY)
        .unwrap();
    let x_p_pole = Transform::from_translation(Vec3::new(0.0, CART_HALF_Y, 0.0));
    let x_c_pole = Transform::from_translation(Vec3::new(0.0, POLE_HALF_LEN, 0.0));
    b.add_joint_revolute(cart, pole, x_p_pole, x_c_pole, Vec3::Z)
        .unwrap();
    b.build().expect("neural policy model")
}

#[test]
fn scene_21_neural_policy_smoke() {
    run_featherstone_smoke(&build_neural_policy_smoke(), "21 NeuralPolicy");
}

// ── Scene 22: NeuralLstm (same cart-pole topology) ──────────────────

#[test]
fn scene_22_neural_lstm_smoke() {
    // Reuses the cart-pole topology; the controller swap is the entire
    // demo. Topology smoke = identical.
    run_featherstone_smoke(&build_neural_policy_smoke(), "22 NeuralLstm");
}

// ── Scene 23: Cable (32-particle cable + cylinder) ──────────────────

const CABLE_PARTICLES: u32 = 32;
const CABLE_PARTICLE_MASS: f64 = 0.04;
const CABLE_PARTICLE_RADIUS: f64 = 0.022;
const CABLE_STIFFNESS: f64 = 1.0e6;
const CABLE_CYLINDER_RADIUS: f64 = 0.30;
const CABLE_CYLINDER_HALF_HEIGHT: f64 = 0.50;
const CABLE_PULLER_RADIUS: f64 = 0.08;

fn build_cable_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.6);
    let pin_pos = Vec3::new(-1.2, 1.8, 0.0);
    let puller_pos = Vec3::new(1.1, 1.6, 0.0);
    let (first, n, _) = b.add_cable_polyline(
        pin_pos,
        puller_pos,
        CABLE_PARTICLES,
        CABLE_PARTICLE_MASS,
        CABLE_PARTICLE_RADIUS,
        CABLE_STIFFNESS,
        0.02,
        WorldId(0),
    );
    b.pin_particle(first);
    b.pin_particle(first + n - 1);
    b.add_shape_cylinder(
        BodyId::NONE,
        CABLE_CYLINDER_HALF_HEIGHT,
        CABLE_CYLINDER_RADIUS,
        Transform::from_translation(Vec3::new(0.0, 1.2, 0.0)),
    )
    .unwrap();
    let puller = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(0.1, CABLE_PULLER_RADIUS))
            .with_position(puller_pos)
            .with_flags(BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(puller, CABLE_PULLER_RADIUS, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("cable model")
}

#[test]
fn scene_23_cable_smoke() {
    run_vbd_smoke(&build_cable_smoke(), "23 Cable", 60);
}

// ── Scene 24: Mpm (GPU-only — build-only smoke) ──────────────────────

#[test]
fn scene_24_mpm_smoke() {
    // The MPM scene's `build_*_model` produces an empty `Model` —
    // particles + grid live in `MpmSimulation`, which is GPU-only.
    let m = build_mesh_tools_smoke(); // same shape as `build_mpm_model`
    run_build_only(&m, "24 Mpm");
}

// ── Scene 25: Jelly (8×8×8 tet cube + plane + puller) ───────────────

const JELLY_GRID: u32 = 6; // viewer uses 8; smaller for CPU speed
const JELLY_SIDE: f64 = 0.8;
const JELLY_TOP_Y: f64 = 3.0;
const JELLY_PARTICLE_MASS: f64 = 0.05;
const JELLY_YOUNGS: f64 = 5.0e5;
const JELLY_POISSON: f64 = 0.4;
const JELLY_PULLER_RADIUS: f64 = 0.18;

fn jelly_cube_to_six_tets(corners: [u32; 8]) -> [[u32; 4]; 6] {
    let v000 = corners[0];
    let v100 = corners[1];
    let v010 = corners[2];
    let v110 = corners[3];
    let v001 = corners[4];
    let v101 = corners[5];
    let v011 = corners[6];
    let v111 = corners[7];
    [
        [v000, v100, v110, v111],
        [v000, v110, v010, v111],
        [v000, v010, v011, v111],
        [v000, v011, v001, v111],
        [v000, v001, v101, v111],
        [v000, v101, v100, v111],
    ]
}

fn build_jelly_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let n = JELLY_GRID;
    let side = JELLY_SIDE;
    let dx = side / (n as f64 - 1.0);
    let centre = Vec3::new(0.0, JELLY_TOP_Y, 0.0);
    let origin = centre - Vec3::splat(side * 0.5);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + Vec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                b.add_particle(p, JELLY_PARTICLE_MASS, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
    let mut tets = Vec::new();
    let mut mat_idx: Vec<u32> = Vec::new();
    for kk in 0..n - 1 {
        for jj in 0..n - 1 {
            for ii in 0..n - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                for tet in jelly_cube_to_six_tets(corners) {
                    tets.push(tet);
                    mat_idx.push(0);
                }
            }
        }
    }
    let materials = vec![MaterialNeoHookean::elastic(JELLY_YOUNGS, JELLY_POISSON)];
    let _ = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let puller = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(0.5, JELLY_PULLER_RADIUS))
            .with_position(Vec3::new(0.6, 0.18, 0.0))
            .with_flags(BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(puller, JELLY_PULLER_RADIUS, Transform::IDENTITY)
        .unwrap();
    b.build().expect("jelly model")
}

#[test]
fn scene_25_jelly_smoke() {
    run_vbd_smoke(&build_jelly_smoke(), "25 Jelly", 60);
}

// ══════════════════════════════════════════════════════════════════════
// Group B — Newton-port basic kinematics (Pendulum, JointSampler, …)
// ══════════════════════════════════════════════════════════════════════

// ── Scene 26: Pendulum (single + 3-link) ────────────────────────────

const PENDULUM_LINK_LEN: f64 = 0.5;
const PENDULUM_LINK_RADIUS: f64 = 0.06;
const PENDULUM_LINK_MASS: f64 = 0.4;
const PENDULUM_BASE_Y: f64 = 4.0;

fn build_pendulum_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    let half_h = PENDULUM_LINK_LEN * 0.5;
    let radius = PENDULUM_LINK_RADIUS;
    let inertia = capsule_inertia(PENDULUM_LINK_MASS, half_h, radius);
    b.begin_articulation();
    let single_anchor = Vec3::new(-1.5, PENDULUM_BASE_Y, 0.0);
    let single = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: PENDULUM_LINK_MASS,
                com: Vec3::ZERO,
                inertia,
            })
            .with_position(single_anchor - Vec3::new(0.0, half_h, 0.0)),
    );
    b.add_shape_capsule(single, half_h, radius, Transform::IDENTITY)
        .unwrap();
    let flip = Quat::from_axis_angle(Vec3::X, std::f64::consts::PI);
    b.add_joint_revolute(
        BodyId::NONE,
        single,
        Transform::new(single_anchor, flip),
        Transform::from_translation(Vec3::new(0.0, -half_h, 0.0)),
        Vec3::Z,
    )
    .unwrap();
    b.begin_articulation();
    let chain_anchor = Vec3::new(1.5, PENDULUM_BASE_Y, 0.0);
    let mut prev = BodyId::NONE;
    let mut next_pos = chain_anchor - Vec3::new(0.0, half_h, 0.0);
    for i in 0..3 {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: PENDULUM_LINK_MASS,
                    com: Vec3::ZERO,
                    inertia,
                })
                .with_position(next_pos),
        );
        b.add_shape_capsule(body, half_h, radius, Transform::IDENTITY)
            .unwrap();
        let x_p = if i == 0 {
            Transform::new(chain_anchor, flip)
        } else {
            Transform::from_translation(Vec3::new(0.0, half_h, 0.0))
        };
        let x_c = Transform::from_translation(Vec3::new(0.0, -half_h, 0.0));
        b.add_joint_revolute(prev, body, x_p, x_c, Vec3::Z).unwrap();
        prev = body;
        next_pos -= Vec3::new(0.0, PENDULUM_LINK_LEN, 0.0);
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("pendulum model")
}

#[test]
fn scene_26_pendulum_smoke() {
    run_featherstone_smoke(&build_pendulum_smoke(), "26 Pendulum");
}

// ── Scene 27: JointSampler (revolute + prismatic + ball) ────────────

fn build_joint_sampler_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    let base_y = 4.0;
    let half_h = 0.4;
    let radius = 0.07;
    let mass = 0.6;
    let inertia = capsule_inertia(mass, half_h, radius);
    let flip = Quat::from_axis_angle(Vec3::X, std::f64::consts::PI);
    // Revolute.
    {
        b.begin_articulation();
        let anchor = Vec3::new(-2.0, base_y, 0.0);
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass,
                    com: Vec3::ZERO,
                    inertia,
                })
                .with_position(anchor - Vec3::new(0.0, half_h, 0.0)),
        );
        b.add_shape_capsule(body, half_h, radius, Transform::IDENTITY)
            .unwrap();
        b.add_joint_revolute(
            BodyId::NONE,
            body,
            Transform::new(anchor, flip),
            Transform::from_translation(Vec3::new(0.0, -half_h, 0.0)),
            Vec3::Z,
        )
        .unwrap();
    }
    // Prismatic.
    {
        b.begin_articulation();
        let anchor = Vec3::new(0.0, base_y, 0.0);
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass,
                    com: Vec3::ZERO,
                    inertia: box_inertia_xyz(mass, 0.15, 0.15, 0.15),
                })
                .with_position(anchor - Vec3::new(0.0, 0.5, 0.0)),
        );
        b.add_shape_box(body, Vec3::splat(0.15), Transform::IDENTITY)
            .unwrap();
        b.add_joint_prismatic(
            BodyId::NONE,
            body,
            Transform::from_translation(anchor),
            Transform::IDENTITY,
            Vec3::Y,
        )
        .unwrap();
    }
    // Ball.
    {
        b.begin_articulation();
        let anchor = Vec3::new(2.0, base_y, 0.0);
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass,
                    com: Vec3::ZERO,
                    inertia,
                })
                .with_position(anchor - Vec3::new(0.0, half_h, 0.0)),
        );
        b.add_shape_capsule(body, half_h, radius, Transform::IDENTITY)
            .unwrap();
        b.add_joint_ball(
            BodyId::NONE,
            body,
            Transform::new(anchor, flip),
            Transform::from_translation(Vec3::new(0.0, -half_h, 0.0)),
        )
        .unwrap();
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("joint sampler model")
}

#[test]
fn scene_27_joint_sampler_smoke() {
    // Featherstone CPU's default-path D-matrix solve (Stage-17 sequential
    // rank-1 deflation) goes NaN on the ball joint at 240 Hz: the ball
    // joint's D-block is non-diagonal in this scene's geometry, and the
    // deflated update doesn't preserve that. The GPU-only sibling test
    // `featherstone_multidof_gpu::ball_spherical_pendulum_falls_under_gravity_gpu`
    // is stable; the viewer runs the GPU backend so this only bites the
    // CPU smoke path. Workaround: enable `rigorous_d_solve` (dense
    // Cholesky) — see Stage 17.1 docs.
    run_featherstone_smoke_with(&build_joint_sampler_smoke(), "27 JointSampler", 0.1, true);
}

// ── Scene 28: Heightfield (waves + dropped primitives) ──────────────

fn build_heightfield_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);
    let n: u32 = 16; // viewer uses 32; smaller for speed
    let cell = 0.5;
    let half = (n as f64 - 1.0) * cell * 0.5;
    let mut heights = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 - (n as f64 - 1.0) * 0.5) * cell;
            let z = (j as f64 - (n as f64 - 1.0) * 0.5) * cell;
            let h = 0.5 * (x * 0.45).sin() * (z * 0.45).cos() + 0.2 * ((x + z) * 0.9).sin();
            heights.push(h);
        }
    }
    let hf = MollaHeightfield {
        nx: n,
        nz: n,
        cell_size: cell,
        origin: Vec3::new(-half, 0.0, -half),
        heights,
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(BodyId::NONE, field_id, Transform::IDENTITY)
        .unwrap();
    let mut rng_state: u64 = 0xCAFEBABEDEADBEEF;
    fn rng_uniform(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn random_quat(s: &mut u64) -> Quat {
        let axis = Vec3::new(rng_uniform(s), rng_uniform(s), rng_uniform(s)).normalize_or_zero();
        let axis = if axis.length_squared() < 1e-9 {
            Vec3::Y
        } else {
            axis
        };
        let angle = rng_uniform(s) * std::f64::consts::PI;
        Quat::from_axis_angle(axis, angle)
    }
    let drop_grid = 4; // viewer uses 6×5; smaller for speed
    let drop_spacing = 1.6_f64;
    for ix in 0..drop_grid {
        for iz in 0..3 {
            let x = (ix as f64 - (drop_grid as f64 - 1.0) * 0.5) * drop_spacing
                + 0.2 * rng_uniform(&mut rng_state);
            let z = (iz as f64 - 1.0) * drop_spacing + 0.2 * rng_uniform(&mut rng_state);
            let y = 6.0 + 0.5 * rng_uniform(&mut rng_state).abs();
            let pos = Vec3::new(x, y, z);
            let rot = random_quat(&mut rng_state);
            match (ix + iz) % 3 {
                0 => {
                    let r = 0.30;
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia::sphere(1.0, r))
                            .with_pose(Transform::new(pos, rot)),
                    );
                    b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
                }
                1 => {
                    let h = 0.30;
                    let inertia = box_inertia(1.0, h);
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia {
                                mass: 1.0,
                                com: Vec3::ZERO,
                                inertia,
                            })
                            .with_pose(Transform::new(pos, rot)),
                    );
                    b.add_shape_box(body, Vec3::splat(h), Transform::IDENTITY)
                        .unwrap();
                }
                _ => {
                    let half_h = 0.30;
                    let radius = 0.18;
                    let inertia = capsule_inertia(0.8, half_h, radius);
                    let body = b.add_body(
                        BodyParams::new()
                            .with_spatial_inertia(SpatialInertia {
                                mass: 0.8,
                                com: Vec3::ZERO,
                                inertia,
                            })
                            .with_pose(Transform::new(pos, rot)),
                    );
                    b.add_shape_capsule(body, half_h, radius, Transform::IDENTITY)
                        .unwrap();
                }
            }
        }
    }
    b.build().expect("heightfield model")
}

#[test]
fn scene_28_heightfield_smoke() {
    run_xpbd_smoke(&build_heightfield_smoke(), "28 Heightfield");
}

// ── Scene 29: Conveyor (kinematic belt + dropped boxes) ─────────────

const CONVEYOR_BELT_LEN: f64 = 6.0;
const CONVEYOR_BELT_HALF_WIDTH: f64 = 0.6;
const CONVEYOR_BELT_HALF_THICK: f64 = 0.05;
const CONVEYOR_BELT_Y: f64 = 1.0;

fn build_conveyor_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);
    let belt_half = Vec3::new(
        CONVEYOR_BELT_LEN * 0.5,
        CONVEYOR_BELT_HALF_THICK,
        CONVEYOR_BELT_HALF_WIDTH,
    );
    let belt = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_inertia(Mat3::ZERO)
            .with_position(Vec3::new(0.0, CONVEYOR_BELT_Y, 0.0))
            .with_flags(BodyFlags::KINEMATIC),
    );
    b.add_shape_box(belt, belt_half, Transform::IDENTITY)
        .unwrap();
    for &z_off in &[
        -CONVEYOR_BELT_HALF_WIDTH - 0.1,
        CONVEYOR_BELT_HALF_WIDTH + 0.1,
    ] {
        b.add_shape_box(
            BodyId::NONE,
            Vec3::new(CONVEYOR_BELT_LEN * 0.5, 0.15, 0.05),
            Transform::from_translation(Vec3::new(0.0, CONVEYOR_BELT_Y + 0.15, z_off)),
        )
        .unwrap();
    }
    for i in 0..8 {
        let x = -CONVEYOR_BELT_LEN * 0.5 + (i as f64 + 0.5) * (CONVEYOR_BELT_LEN / 8.0);
        let half = Vec3::splat(0.15);
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.5,
                    com: Vec3::ZERO,
                    inertia: box_inertia(0.5, 0.15),
                })
                .with_position(Vec3::new(
                    x,
                    CONVEYOR_BELT_Y + CONVEYOR_BELT_HALF_THICK + 0.20 + (i as f64) * 0.05,
                    0.0,
                )),
        );
        b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("conveyor model")
}

#[test]
fn scene_29_conveyor_smoke() {
    // Test only that the scene builds + steps without exploding. The
    // documented XPBD-Coulomb-friction limitation (`scene_correctness.rs
    // ::conveyor_belt_drags_box_via_friction`) means the boxes barely
    // move; that is not a smoke-test concern.
    run_xpbd_smoke(&build_conveyor_smoke(), "29 Conveyor");
}

// ── Scene 30: Pyramid (10-layer pyramid + heavy ball) ───────────────

const PYRAMID_LEVELS: usize = 10;
const PYRAMID_BOX_HALF: f64 = 0.30;
const PYRAMID_SPACING: f64 = 2.05 * PYRAMID_BOX_HALF;
const PYRAMID_BALL_RADIUS: f64 = 1.2;
const PYRAMID_BALL_MASS: f64 = 80.0;
const PYRAMID_BALL_HEIGHT: f64 = 14.0;

fn build_pyramid_smoke() -> Model {
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
    let ball = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(
                PYRAMID_BALL_MASS,
                PYRAMID_BALL_RADIUS,
            ))
            .with_position(Vec3::new(1.5, PYRAMID_BALL_HEIGHT, 0.0)),
    );
    b.add_shape_sphere(ball, PYRAMID_BALL_RADIUS, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("pyramid model")
}

#[test]
fn scene_30_pyramid_smoke() {
    run_xpbd_smoke(&build_pyramid_smoke(), "30 Pyramid");
}

// ══════════════════════════════════════════════════════════════════════
// Group C — Cloth + cable variations (Newton-port).
// ══════════════════════════════════════════════════════════════════════

// ── Scene 31: ClothBending (two strips, bending compliance A/B) ─────

const CLOTH_BEND_GRID_X: u32 = 8; // viewer uses 12
const CLOTH_BEND_GRID_Y: u32 = 12; // viewer uses 18
const CLOTH_BEND_W: f64 = 0.9;
const CLOTH_BEND_H: f64 = 1.4;
const CLOTH_BEND_TOP_Y: f64 = 4.0;

fn build_cloth_bending_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);
    for (side_idx, x_offset) in [-1.0_f64, 1.0].iter().enumerate() {
        let comp = if side_idx == 0 { 0.05_f64 } else { 5.0_f64 };
        let origin = Vec3::new(*x_offset - CLOTH_BEND_W * 0.5, CLOTH_BEND_TOP_Y, 0.0);
        let u_axis = Vec3::new(CLOTH_BEND_W, 0.0, 0.0);
        let v_axis = Vec3::new(0.0, -CLOTH_BEND_H, 0.0);
        let (first, _, _) = b.add_cloth_grid_with_bending(
            origin,
            u_axis,
            v_axis,
            CLOTH_BEND_GRID_X,
            CLOTH_BEND_GRID_Y,
            0.04,
            0.025,
            5.0e-6,
            0.05,
            comp,
            WorldId(0),
        );
        for ix in 0..CLOTH_BEND_GRID_X {
            b.pin_particle(first + ix);
        }
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("cloth bending model")
}

#[test]
fn scene_31_cloth_bending_smoke() {
    run_vbd_smoke(&build_cloth_bending_smoke(), "31 ClothBending", 60);
}

// ── Scene 32: ClothTwist (top + bottom pinned) ──────────────────────

const CLOTH_TWIST_GRID_X: u32 = 10; // viewer uses 18
const CLOTH_TWIST_GRID_Y: u32 = 14; // viewer uses 24
const CLOTH_TWIST_W: f64 = 1.0;
const CLOTH_TWIST_H: f64 = 1.6;
const CLOTH_TWIST_TOP_Y: f64 = 4.0;

fn build_cloth_twist_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);
    let origin = Vec3::new(-CLOTH_TWIST_W * 0.5, CLOTH_TWIST_TOP_Y, 0.0);
    let u_axis = Vec3::new(CLOTH_TWIST_W, 0.0, 0.0);
    let v_axis = Vec3::new(0.0, -CLOTH_TWIST_H, 0.0);
    let (first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        CLOTH_TWIST_GRID_X,
        CLOTH_TWIST_GRID_Y,
        0.04,
        0.025,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    for ix in 0..CLOTH_TWIST_GRID_X {
        b.pin_particle(first + ix);
    }
    let bottom_iy = CLOTH_TWIST_GRID_Y - 1;
    for ix in 0..CLOTH_TWIST_GRID_X {
        b.pin_particle(first + bottom_iy * CLOTH_TWIST_GRID_X + ix);
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("cloth twist model")
}

#[test]
fn scene_32_cloth_twist_smoke() {
    run_vbd_smoke(&build_cloth_twist_smoke(), "32 ClothTwist", 60);
}

// ── Scene 33: CablePile (8 cables in a bowl) ────────────────────────

fn build_cable_pile_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.6);
    let bowl_size = 1.6;
    let wall_h = 0.8;
    let wall_t = 0.05;
    b.add_shape_box(
        BodyId::NONE,
        Vec3::new(bowl_size, wall_h, wall_t),
        Transform::from_translation(Vec3::new(0.0, wall_h, bowl_size)),
    )
    .unwrap();
    b.add_shape_box(
        BodyId::NONE,
        Vec3::new(bowl_size, wall_h, wall_t),
        Transform::from_translation(Vec3::new(0.0, wall_h, -bowl_size)),
    )
    .unwrap();
    b.add_shape_box(
        BodyId::NONE,
        Vec3::new(wall_t, wall_h, bowl_size),
        Transform::from_translation(Vec3::new(bowl_size, wall_h, 0.0)),
    )
    .unwrap();
    b.add_shape_box(
        BodyId::NONE,
        Vec3::new(wall_t, wall_h, bowl_size),
        Transform::from_translation(Vec3::new(-bowl_size, wall_h, 0.0)),
    )
    .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let cable_len = 1.4;
    let particles_per_cable: u32 = 16; // viewer uses 24
    let mut rng_state: u64 = 0xFEEDFACECAFEBEEF;
    fn rng(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    for i in 0..4 {
        // viewer uses 8; smaller for speed
        let h = 1.5 + (i as f64) * 0.30;
        let yaw = (i as f64) * 0.7 + rng(&mut rng_state) * 0.2;
        let dx = yaw.cos() * cable_len * 0.5;
        let dz = yaw.sin() * cable_len * 0.5;
        let cx = 0.6 * rng(&mut rng_state);
        let cz = 0.6 * rng(&mut rng_state);
        let p0 = Vec3::new(cx - dx, h, cz - dz);
        let p1 = Vec3::new(cx + dx, h, cz + dz);
        let _ = b.add_cable_polyline(
            p0,
            p1,
            particles_per_cable,
            0.04,
            0.025,
            5.0e5,
            0.02,
            WorldId(0),
        );
    }
    b.build().expect("cable pile model")
}

#[test]
fn scene_33_cable_pile_smoke() {
    run_vbd_smoke(&build_cable_pile_smoke(), "33 CablePile", 60);
}

// ── Scene 34: CableYJunction (3 cables meeting at a Y) ──────────────

fn build_cable_y_junction_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let junction = Vec3::new(0.0, 2.5, 0.0);
    let branch_len = 1.2;
    let particles_per_branch: u32 = 16;
    let dirs = [
        Vec3::new(1.0, 0.7, 0.0).normalize(),
        Vec3::new(-0.5, 0.7, 0.866).normalize(),
        Vec3::new(-0.5, 0.7, -0.866).normalize(),
    ];
    let mut firsts = Vec::with_capacity(3);
    for d in &dirs {
        let p_far = junction + *d * branch_len;
        let (first, _, _) = b.add_cable_polyline(
            junction,
            p_far,
            particles_per_branch,
            0.04,
            0.025,
            5.0e5,
            0.02,
            WorldId(0),
        );
        firsts.push(first);
    }
    for f in &firsts {
        b.pin_particle(*f);
    }
    b.pin_particle(firsts[0] + particles_per_branch - 1);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("cable y junction model")
}

#[test]
fn scene_34_cable_y_junction_smoke() {
    run_vbd_smoke(&build_cable_y_junction_smoke(), "34 CableYJunction", 60);
}

// ══════════════════════════════════════════════════════════════════════
// Group D — MPM + diffsim + cart-pole + 4-bar.
// ══════════════════════════════════════════════════════════════════════

// ── Scene 35: MpmSnowBall (GPU-only — build smoke is empty model) ──

#[test]
#[ignore = "MPM requires GPU runtime; build alone is the empty model. CPU smoke is meaningless without an MpmSimulation backend."]
fn scene_35_mpm_snow_ball_smoke() {
    // The viewer's `build_mpm_model()` returns an empty `Model`; the
    // particles + grid live in `MpmSimulation` which is GPU-only.
    let m = build_mesh_tools_smoke();
    run_build_only(&m, "35 MpmSnowBall");
}

// ── Scene 36: MpmTwoWay (GPU-only) ──────────────────────────────────

#[test]
#[ignore = "MPM two-way coupling requires GPU runtime."]
fn scene_36_mpm_two_way_smoke() {
    let m = build_mesh_tools_smoke();
    run_build_only(&m, "36 MpmTwoWay");
}

// ── Scene 37: CartPole (PD-balanced cart-pole) ──────────────────────

#[test]
fn scene_37_cartpole_smoke() {
    // Identical topology to NeuralPolicy; controller is the only diff.
    run_featherstone_smoke(&build_neural_policy_smoke(), "37 CartPole");
}

// ── Scene 38: CartPoleSwarm (16 independent cart-poles) ─────────────

const CARTPOLE_SWARM_N: usize = 8; // viewer uses 16; smaller for speed
const CARTPOLE_SWARM_SPACING: f64 = 1.2;

fn build_cartpole_swarm_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.02);
    let cart_inertia = box_inertia_xyz(CART_MASS, CART_HALF_X, CART_HALF_Y, CART_HALF_Z);
    let pole_inertia = capsule_inertia(POLE_MASS, POLE_HALF_LEN, POLE_RADIUS);
    for i in 0..CARTPOLE_SWARM_N {
        let z = (i as f64 - (CARTPOLE_SWARM_N as f64 - 1.0) * 0.5) * CARTPOLE_SWARM_SPACING;
        b.begin_articulation();
        let cart = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: CART_MASS,
                    com: Vec3::ZERO,
                    inertia: cart_inertia,
                })
                .with_position(Vec3::new(0.0, CART_RAIL_Y, z)),
        );
        b.add_shape_box(
            cart,
            Vec3::new(CART_HALF_X, CART_HALF_Y, CART_HALF_Z),
            Transform::IDENTITY,
        )
        .unwrap();
        let x_p_cart = Transform::from_translation(Vec3::new(0.0, CART_RAIL_Y, z));
        b.add_joint_prismatic(BodyId::NONE, cart, x_p_cart, Transform::IDENTITY, Vec3::X)
            .unwrap();
        let pole = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: POLE_MASS,
                    com: Vec3::ZERO,
                    inertia: pole_inertia,
                })
                .with_position(Vec3::new(0.0, CART_RAIL_Y - POLE_HALF_LEN, z)),
        );
        b.add_shape_capsule(pole, POLE_HALF_LEN, POLE_RADIUS, Transform::IDENTITY)
            .unwrap();
        let x_p_pole = Transform::from_translation(Vec3::new(0.0, CART_HALF_Y, 0.0));
        let x_c_pole = Transform::from_translation(Vec3::new(0.0, POLE_HALF_LEN, 0.0));
        b.add_joint_revolute(cart, pole, x_p_pole, x_c_pole, Vec3::Z)
            .unwrap();
    }
    b.build().expect("cartpole swarm model")
}

#[test]
fn scene_38_cartpole_swarm_smoke() {
    run_featherstone_smoke(&build_cartpole_swarm_smoke(), "38 CartPoleSwarm");
}

// ── Scene 39: FourBar (4-bar JointMimic linkage) ────────────────────

const FOURBAR_BAR_RADIUS: f64 = 0.04;
const FOURBAR_BASE_Y: f64 = 2.5;

fn build_fourbar_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.10);
    let l_a = 0.6;
    let l_b = 1.2;
    let l_c = 0.6;
    let mass = 0.3;
    let inertia = box_inertia_xyz(mass, 0.05, l_a * 0.5, 0.05);
    b.begin_articulation();
    let a_pivot = Vec3::new(-0.5, FOURBAR_BASE_Y, 0.0);
    let bar_a = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: Vec3::ZERO,
                inertia,
            })
            .with_position(a_pivot - Vec3::new(0.0, l_a * 0.5, 0.0)),
    );
    b.add_shape_capsule(bar_a, l_a * 0.5, FOURBAR_BAR_RADIUS, Transform::IDENTITY)
        .unwrap();
    let flip = Quat::from_axis_angle(Vec3::X, std::f64::consts::PI);
    let driver_id = b
        .add_joint_revolute(
            BodyId::NONE,
            bar_a,
            Transform::new(a_pivot, flip),
            Transform::from_translation(Vec3::new(0.0, -l_a * 0.5, 0.0)),
            Vec3::Z,
        )
        .unwrap();
    let bar_b = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: Vec3::ZERO,
                inertia: box_inertia_xyz(mass, 0.05, l_b * 0.5, 0.05),
            })
            .with_position(Vec3::new(0.0, FOURBAR_BASE_Y - l_a, 0.0)),
    );
    b.add_shape_capsule(bar_b, l_b * 0.5, FOURBAR_BAR_RADIUS, Transform::IDENTITY)
        .unwrap();
    let coupler_id = b
        .add_joint_revolute(
            bar_a,
            bar_b,
            Transform::from_translation(Vec3::new(0.0, l_a * 0.5, 0.0)),
            Transform::from_translation(Vec3::new(0.0, -l_b * 0.5, 0.0)),
            Vec3::Z,
        )
        .unwrap();
    let bar_c = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: Vec3::ZERO,
                inertia,
            })
            .with_position(Vec3::new(0.5, FOURBAR_BASE_Y - l_a, 0.0)),
    );
    b.add_shape_capsule(bar_c, l_c * 0.5, FOURBAR_BAR_RADIUS, Transform::IDENTITY)
        .unwrap();
    let rocker_id = b
        .add_joint_revolute(
            bar_b,
            bar_c,
            Transform::from_translation(Vec3::new(0.0, l_b * 0.5, 0.0)),
            Transform::from_translation(Vec3::new(0.0, -l_c * 0.5, 0.0)),
            Vec3::Z,
        )
        .unwrap();
    let _ = b.add_equality_joint_mimic(driver_id, coupler_id, [0.0, -1.0, 0.0, 0.0, 0.0]);
    let _ = b.add_equality_joint_mimic(driver_id, rocker_id, [0.0, 1.0, 0.0, 0.0, 0.0]);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("fourbar model")
}

#[test]
fn scene_39_fourbar_smoke() {
    run_featherstone_smoke(&build_fourbar_smoke(), "39 FourBar");
}

// ── Scene 40: DiffBall (kinematic projectile + target — host-driven) ─

fn build_diff_ball_smoke() -> Model {
    let mut b = ModelBuilder::new();
    let ball = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_inertia(Mat3::ZERO)
            .with_position(Vec3::new(0.0, 0.5, 0.0))
            .with_flags(BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(ball, 0.10, Transform::IDENTITY).unwrap();
    let target = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_inertia(Mat3::ZERO)
            .with_position(Vec3::new(3.0, 1.5, 0.0))
            .with_flags(BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(target, 0.08, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("diff ball model")
}

#[test]
fn scene_40_diff_ball_smoke() {
    run_build_only(&build_diff_ball_smoke(), "40 DiffBall");
}

// ── Scene 41: DiffSpring (8 anchors + 1 particle, host-driven) ──────

fn build_diff_spring_smoke() -> Model {
    let mut b = ModelBuilder::new();
    let half = 0.6;
    let cy = 1.5;
    for sx in &[-1.0_f64, 1.0] {
        for sy in &[-1.0_f64, 1.0] {
            for sz in &[-1.0_f64, 1.0] {
                let body = b.add_body(
                    BodyParams::new()
                        .with_mass(0.0)
                        .with_inertia(Mat3::ZERO)
                        .with_position(Vec3::new(sx * half, cy + sy * half, sz * half))
                        .with_flags(BodyFlags::KINEMATIC),
                );
                b.add_shape_sphere(body, 0.04, Transform::IDENTITY).unwrap();
            }
        }
    }
    let particle = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_inertia(Mat3::ZERO)
            .with_position(Vec3::new(0.0, cy, 0.0))
            .with_flags(BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(particle, 0.08, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("diff spring model")
}

#[test]
fn scene_41_diff_spring_smoke() {
    run_build_only(&build_diff_spring_smoke(), "41 DiffSpring");
}

// ══════════════════════════════════════════════════════════════════════
// Group E — Sensors / multi-env / soft contact (Newton-port).
// ══════════════════════════════════════════════════════════════════════

// ── Scene 42: BrickWall (6×10 staggered wall + projectile) ──────────

const BRICK_HALF_VEC: Vec3 = Vec3::new(0.18, 0.10, 0.10);
const BRICK_GAP: f64 = 0.005;
const BRICK_ROWS: usize = 6;
const BRICK_COLS: usize = 10;

fn build_brick_wall_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.20);
    let mass = 0.6;
    let h2 = BRICK_HALF_VEC * BRICK_HALF_VEC;
    let m_third = mass / 3.0;
    let inertia = Mat3::from_diagonal(Vec3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ));
    let layer_h = BRICK_HALF_VEC.y * 2.0 + BRICK_GAP;
    let col_w = BRICK_HALF_VEC.x * 2.0 + BRICK_GAP;
    for row in 0..BRICK_ROWS {
        let stagger = if row % 2 == 0 { 0.0 } else { BRICK_HALF_VEC.x };
        for col in 0..BRICK_COLS {
            let x = (col as f64 - (BRICK_COLS as f64 - 1.0) * 0.5) * col_w + stagger;
            let y = BRICK_HALF_VEC.y + 0.001 + (row as f64) * layer_h;
            let body = b.add_body(
                BodyParams::new()
                    .with_spatial_inertia(SpatialInertia {
                        mass,
                        com: Vec3::ZERO,
                        inertia,
                    })
                    .with_position(Vec3::new(x, y, 0.0)),
            );
            b.add_shape_box(body, BRICK_HALF_VEC, Transform::IDENTITY)
                .unwrap();
        }
    }
    let proj_radius = 0.45;
    let proj_mass = 25.0;
    let projectile = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(proj_mass, proj_radius))
            .with_position(Vec3::new(0.0, 12.0, 3.5)),
    );
    b.add_shape_sphere(projectile, proj_radius, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("brick wall model")
}

#[test]
fn scene_42_brick_wall_smoke() {
    run_xpbd_smoke(&build_brick_wall_smoke(), "42 BrickWall");
}

// ── Scene 43: SoftbodyDrop (tet cube on hanging cloth) ──────────────

const SOFT_DROP_TET_NX: u32 = 4; // viewer uses 6
const SOFT_DROP_TET_NY: u32 = 4; // viewer uses 6
const SOFT_DROP_TET_NZ: u32 = 3;
const SOFT_DROP_TET_DX: f64 = 0.10;
const SOFT_DROP_CLOTH_NX: u32 = 12; // viewer uses 24
const SOFT_DROP_CLOTH_W: f64 = 1.6;

fn build_softbody_drop_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let origin = Vec3::new(-SOFT_DROP_CLOTH_W * 0.5, 1.0, -SOFT_DROP_CLOTH_W * 0.5);
    let u_axis = Vec3::new(SOFT_DROP_CLOTH_W, 0.0, 0.0);
    let v_axis = Vec3::new(0.0, 0.0, SOFT_DROP_CLOTH_W);
    let (cloth_first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        SOFT_DROP_CLOTH_NX,
        SOFT_DROP_CLOTH_NX,
        0.05,
        0.04,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    for ix in 0..SOFT_DROP_CLOTH_NX {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (SOFT_DROP_CLOTH_NX - 1) * SOFT_DROP_CLOTH_NX + ix);
    }
    let cube_origin = Vec3::new(
        -SOFT_DROP_TET_DX * (SOFT_DROP_TET_NX as f64 - 1.0) * 0.5,
        2.5,
        -SOFT_DROP_TET_DX * (SOFT_DROP_TET_NZ as f64 - 1.0) * 0.5,
    );
    for kk in 0..SOFT_DROP_TET_NZ {
        for jj in 0..SOFT_DROP_TET_NY {
            for ii in 0..SOFT_DROP_TET_NX {
                let p = cube_origin
                    + Vec3::new(
                        ii as f64 * SOFT_DROP_TET_DX,
                        jj as f64 * SOFT_DROP_TET_DX,
                        kk as f64 * SOFT_DROP_TET_DX,
                    );
                b.add_particle(p, 0.05, SOFT_DROP_TET_DX * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 {
        let cloth_n = SOFT_DROP_CLOTH_NX * SOFT_DROP_CLOTH_NX;
        cloth_first + cloth_n + (k * SOFT_DROP_TET_NY + j) * SOFT_DROP_TET_NX + i
    };
    let mut tets = Vec::new();
    for kk in 0..SOFT_DROP_TET_NZ - 1 {
        for jj in 0..SOFT_DROP_TET_NY - 1 {
            for ii in 0..SOFT_DROP_TET_NX - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                for tet in jelly_cube_to_six_tets(corners) {
                    tets.push(tet);
                }
            }
        }
    }
    let materials = vec![MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let cube_id = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");
    b.set_tet_mesh_friction(cube_id, 0.5);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("softbody drop model")
}

#[test]
fn scene_43_softbody_drop_smoke() {
    run_vbd_smoke(&build_softbody_drop_smoke(), "43 SoftbodyDrop", 60);
}

// ── Scene 44: ClothCollar (pinned cloth tube + self-collision) ──────

const COLLAR_NX: u32 = 16; // viewer uses 24
const COLLAR_W: f64 = 0.6;
const COLLAR_TOP_Y: f64 = 1.6;

fn build_cloth_collar_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let cy = COLLAR_TOP_Y;
    let origin = Vec3::new(-COLLAR_W * 0.5, cy, -COLLAR_W * 0.5);
    let u_axis = Vec3::new(COLLAR_W, 0.0, 0.0);
    let v_axis = Vec3::new(0.0, 0.0, COLLAR_W);
    let (first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        COLLAR_NX,
        COLLAR_NX,
        0.05,
        0.02,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    for ix in 0..COLLAR_NX {
        b.pin_particle(first + ix);
        b.pin_particle(first + (COLLAR_NX - 1) * COLLAR_NX + ix);
    }
    b.set_cloth_mesh_self_collision(0, true);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("cloth collar model")
}

#[test]
fn scene_44_cloth_collar_smoke() {
    run_vbd_smoke(&build_cloth_collar_smoke(), "44 ClothCollar", 60);
}

// ── Scene 45: FastSoftImpact (CCD; CPU only) ────────────────────────

fn build_fast_soft_impact_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.20);
    let cloth_n = 12u32; // viewer uses 16
    let cloth_w = 0.6;
    let cloth_y = 1.0;
    let (cloth_first, _, _) = b.add_cloth_grid(
        Vec3::new(-cloth_w * 0.5, cloth_y, -cloth_w * 0.5),
        Vec3::new(cloth_w, 0.0, 0.0),
        Vec3::new(0.0, 0.0, cloth_w),
        cloth_n,
        cloth_n,
        0.05,
        0.02,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    for ix in 0..cloth_n {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (cloth_n - 1) * cloth_n + ix);
        b.pin_particle(cloth_first + ix * cloth_n);
        b.pin_particle(cloth_first + ix * cloth_n + cloth_n - 1);
    }
    let nx = 3u32;
    let dx = 0.05;
    let cube_lo_y = cloth_y + 0.30;
    let cube_origin = Vec3::new(
        -dx * (nx as f64 - 1.0) * 0.5,
        cube_lo_y,
        -dx * (nx as f64 - 1.0) * 0.5,
    );
    for kk in 0..nx {
        for jj in 0..nx {
            for ii in 0..nx {
                let p = cube_origin + Vec3::new(ii as f64 * dx, jj as f64 * dx, kk as f64 * dx);
                b.add_particle(p, 0.05, dx * 0.4, WorldId(0));
            }
        }
    }
    let cloth_count = cloth_n * cloth_n;
    let v_at =
        |i: u32, j: u32, k: u32| -> u32 { cloth_first + cloth_count + (k * nx + j) * nx + i };
    let mut tets = Vec::new();
    for kk in 0..nx - 1 {
        for jj in 0..nx - 1 {
            for ii in 0..nx - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                for tet in jelly_cube_to_six_tets(corners) {
                    tets.push(tet);
                }
            }
        }
    }
    let materials = vec![MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let _ = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("fast soft impact model")
}

#[test]
fn scene_45_fast_soft_impact_smoke() {
    // The viewer uses CPU VBD with CCD on for this scene; we run the
    // same backend (also CPU) without CCD because the simple smoke loop
    // doesn't pre-launch the cube (it just falls under gravity, no
    // tunneling concern at smoke-test cadence).
    let model = build_fast_soft_impact_smoke();
    let mut s_a = model.state().expect("state_a");
    let mut s_b = model.state().expect("state_b");
    let control = model.control();
    let mut contacts = Contacts::with_capacity(2048);
    let mut pair_buf = Vec::new();
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(4)
        .with_contact_stiffness(1.0e5);
    for _ in 0..60 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).expect("collide");
        Solver::step(
            &mut solver,
            &model,
            &s_a,
            &control,
            &contacts,
            SMOKE_DT,
            &mut s_b,
        )
        .expect("vbd step");
        std::mem::swap(&mut s_a, &mut s_b);
    }
    assert_no_nan_or_escape(&s_a, "45 FastSoftImpact");
    assert_counts_match(&model, &s_a, "45 FastSoftImpact");
}

// ── Scene 46: SoftBrake (tilted gravity + soft block on soft ramp) ──

fn build_soft_brake_smoke() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);
    let ramp_tilt: f64 = 25.0_f64.to_radians();
    b.set_world_gravity(
        WorldId(0),
        Vec3::new(9.81 * ramp_tilt.sin(), -9.81 * ramp_tilt.cos(), 0.0),
    );
    // Ramp.
    let ramp_lo = Vec3::new(-0.75, 1.0, -0.30);
    let ramp_hi = Vec3::new(0.75, 1.10, 0.30);
    let mut ramp_verts: Vec<u32> = Vec::with_capacity(8);
    for &z in &[ramp_lo.z, ramp_hi.z] {
        for &y in &[ramp_lo.y, ramp_hi.y] {
            for &x in &[ramp_lo.x, ramp_hi.x] {
                let id = b.add_particle(Vec3::new(x, y, z), 0.0, 0.02, WorldId(0));
                ramp_verts.push(id);
            }
        }
    }
    let rv: [u32; 8] = ramp_verts.as_slice().try_into().unwrap();
    let ramp_tets = jelly_cube_to_six_tets(rv);
    let ramp_id = b
        .add_tet_mesh(&ramp_tets, 1.0e6, 0.3, None)
        .expect("ramp tet block");
    b.set_tet_mesh_friction(ramp_id, 0.5);
    // Block.
    let bnx: u32 = 3;
    let bdx = 0.10;
    let block_origin = Vec3::new(-0.50, 1.10 + 0.005, -bdx * (bnx as f64 - 1.0) * 0.5);
    let block_first = 8u32;
    for kk in 0..bnx {
        for jj in 0..bnx {
            for ii in 0..bnx {
                let p = block_origin + Vec3::new(ii as f64 * bdx, jj as f64 * bdx, kk as f64 * bdx);
                b.add_particle(p, 0.05, bdx * 0.5, WorldId(0));
            }
        }
    }
    let v_at_b = |i: u32, j: u32, k: u32| -> u32 { block_first + (k * bnx + j) * bnx + i };
    let mut block_tets: Vec<[u32; 4]> = Vec::new();
    for kk in 0..bnx - 1 {
        for jj in 0..bnx - 1 {
            for ii in 0..bnx - 1 {
                let corners = [
                    v_at_b(ii, jj, kk),
                    v_at_b(ii + 1, jj, kk),
                    v_at_b(ii, jj + 1, kk),
                    v_at_b(ii + 1, jj + 1, kk),
                    v_at_b(ii, jj, kk + 1),
                    v_at_b(ii + 1, jj, kk + 1),
                    v_at_b(ii, jj + 1, kk + 1),
                    v_at_b(ii + 1, jj + 1, kk + 1),
                ];
                block_tets.extend_from_slice(&jelly_cube_to_six_tets(corners));
            }
        }
    }
    let materials = vec![MaterialNeoHookean::elastic(1.0e6, 0.3)];
    let mat_idx: Vec<u32> = vec![0; block_tets.len()];
    let block_id = b
        .add_tet_mesh_per_material(&block_tets, &materials, &mat_idx, None)
        .expect("block tet mesh");
    b.set_tet_mesh_friction(block_id, 0.5);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().expect("soft brake model")
}

#[test]
fn scene_46_soft_brake_smoke() {
    run_vbd_smoke(&build_soft_brake_smoke(), "46 SoftBrake", 60);
}

// ──────────────────────────────────────────────────────────────────────
// Suppress unused-import warnings for the helpers that aren't actively
// referenced by every test path.
// ──────────────────────────────────────────────────────────────────────
#[allow(dead_code)]
fn _silence_unused_imports() {
    let _ = SpatialVector::ZERO;
}
