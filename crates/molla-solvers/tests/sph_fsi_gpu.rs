//! Stage F3 — GPU SPH fluid–solid (FSI) coupling tests.
//!
//! GPU-only; gated behind `#[ignore]`. Run via
//! `make test-gpu P=molla-solvers T=sph_fsi_gpu`.
#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_solvers::{SphConfig, SphRigidBody, SphSimulation, box_tank_walls};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

/// Lattice over `[min, max]` at `spacing`, excluding points inside the
/// box at `box_com` with the given half-extent (plus a one-cell margin)
/// so fluid never starts overlapping the rigid body's samples.
fn fluid_pool(
    min: [f32; 3],
    max: [f32; 3],
    spacing: f32,
    box_com: [f32; 3],
    box_half: [f32; 3],
) -> Vec<[f32; 3]> {
    let n = |a: f32, b: f32| (((b - a) / spacing).floor() as i32 + 1).max(1);
    let m = spacing;
    let mut v = Vec::new();
    for ix in 0..n(min[0], max[0]) {
        for iy in 0..n(min[1], max[1]) {
            for iz in 0..n(min[2], max[2]) {
                let p = [
                    min[0] + ix as f32 * spacing,
                    min[1] + iy as f32 * spacing,
                    min[2] + iz as f32 * spacing,
                ];
                let in_box = (p[0] - box_com[0]).abs() < box_half[0] + m
                    && (p[1] - box_com[1]).abs() < box_half[1] + m
                    && (p[2] - box_com[2]).abs() < box_half[2] + m;
                if !in_box {
                    v.push(p);
                }
            }
        }
    }
    v
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=sph_fsi_gpu`"]
fn sph_fsi_submerged_box_rises() {
    let engine = try_engine().expect("GPU adapter required (gated by #[ignore])");
    let cfg = SphConfig::water();
    // A narrow tank — its footprint matches the pool so the fluid
    // cannot spread shallow; it stays a deep pool around the body.
    let tank_min = [-0.15, 0.0, -0.15];
    let tank_max = [0.15, 0.55, 0.15];
    let walls = box_tank_walls(tank_min, tank_max, cfg.h, 2);
    let box_com = [0.0, 0.22, 0.0];
    let box_half = [0.05, 0.05, 0.05];
    // Body mass 0.3 vs rho0 * (0.1)^3 = 1.0 → density 300 < 1000 → floats.
    let body = SphRigidBody::box_body(box_com, box_half, cfg.h, 0.3);
    // A deep pool filling the tank footprint.
    let fluid = fluid_pool(
        [-0.12, 0.03, -0.12],
        [0.12, 0.42, 0.12],
        cfg.h,
        box_com,
        box_half,
    );
    assert!(!fluid.is_empty() && !walls.is_empty());

    let mut sim = SphSimulation::new(&engine, cfg).expect("sph new");
    let queue = engine.queue().unwrap().clone();
    sim.set_fsi_scene(&queue, &fluid, &walls, vec![body])
        .expect("set_fsi_scene");

    let y0 = sim.body(0).com[1];
    for _ in 0..150 {
        sim.step_fsi(&engine, 2.0e-4, 4).expect("step_fsi");
    }
    let com = sim.body(0).com;
    assert!(com.iter().all(|c| c.is_finite()), "box com non-finite");
    assert!(
        com[1] > y0 + 0.02,
        "buoyant box did not rise: y0={y0} y1={}",
        com[1]
    );
    assert!(
        com[1] < tank_max[1] + 0.15,
        "box flew out of the tank: y={}",
        com[1]
    );
    let fl = sim.read_positions(&engine).expect("read");
    for p in &fl {
        for &c in p {
            assert!(c.is_finite(), "fluid non-finite");
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=sph_fsi_gpu`"]
fn sph_fsi_dropped_box_transfers_momentum() {
    let engine = try_engine().expect("GPU adapter required (gated by #[ignore])");
    let cfg = SphConfig::water();
    let tank_min = [-0.15, 0.0, -0.15];
    let tank_max = [0.15, 0.65, 0.15];
    let walls = box_tank_walls(tank_min, tank_max, cfg.h, 2);
    // Box dropped from well above the water surface.
    let box_com = [0.0, 0.50, 0.0];
    let box_half = [0.05, 0.05, 0.05];
    let body = SphRigidBody::box_body(box_com, box_half, cfg.h, 0.3);
    // A deep pool resting in the lower tank (the box starts above it).
    let fluid = fluid_pool(
        [-0.12, 0.03, -0.12],
        [0.12, 0.34, 0.12],
        cfg.h,
        box_com,
        box_half,
    );

    let mut sim = SphSimulation::new(&engine, cfg).expect("sph new");
    let queue = engine.queue().unwrap().clone();
    sim.set_fsi_scene(&queue, &fluid, &walls, vec![body])
        .expect("set_fsi_scene");

    for _ in 0..300 {
        sim.step_fsi(&engine, 2.0e-4, 4).expect("step_fsi");
    }
    let com = sim.body(0).com;
    assert!(com.iter().all(|c| c.is_finite()), "box com non-finite");
    // Water arrested the fall — the box did not sink to the floor.
    assert!(
        com[1] > tank_min[1] + box_half[1],
        "box sank to the tank floor: y={}",
        com[1]
    );
    // Momentum was transferred — the impact disturbed the fluid.
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let vels = sim.read_velocities_with(&device, &queue).expect("read vel");
    let max_speed = vels
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        max_speed > 0.05,
        "fluid was not disturbed by the impact: max_speed={max_speed}"
    );
}
