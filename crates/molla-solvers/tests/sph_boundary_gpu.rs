//! Stage F2 — GPU SPH boundary-particle smoke + CPU/GPU parity tests.
//!
//! GPU-only; gated behind `#[ignore]`. Run via
//! `make test-gpu P=molla-solvers T=sph_boundary_gpu`.
#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_solvers::sph_oracle::SphOracle;
use molla_solvers::{SphConfig, SphSimulation, box_tank_walls};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

/// Cubic lattice of particles spanning `[min, max]` at `spacing`.
fn lattice(min: [f32; 3], max: [f32; 3], spacing: f32) -> Vec<[f32; 3]> {
    let n = |a: f32, b: f32| (((b - a) / spacing).floor() as i32 + 1).max(1);
    let mut v = Vec::new();
    for ix in 0..n(min[0], max[0]) {
        for iy in 0..n(min[1], max[1]) {
            for iz in 0..n(min[2], max[2]) {
                v.push([
                    min[0] + ix as f32 * spacing,
                    min[1] + iy as f32 * spacing,
                    min[2] + iz as f32 * spacing,
                ]);
            }
        }
    }
    v
}

/// Build a CPU oracle for a fluid + boundary scene mirroring `cfg`.
fn make_oracle(cfg: &SphConfig, fluid: &[[f32; 3]], boundary: &[[f32; 3]]) -> SphOracle {
    let to_f64 = |p: &[f32; 3]| [p[0] as f64, p[1] as f64, p[2] as f64];
    let mut pos: Vec<[f64; 3]> = fluid.iter().map(to_f64).collect();
    pos.extend(boundary.iter().map(to_f64));
    let n = pos.len();
    SphOracle {
        pos,
        vel: vec![[0.0; 3]; n],
        n_fluid: fluid.len(),
        density: vec![],
        pressure: vec![],
        stress: vec![],
        h: cfg.h as f64,
        mass: cfg.particle_mass as f64,
        rest_density: cfg.rest_density as f64,
        stiffness_b: cfg.stiffness_b as f64,
        gamma: cfg.gamma as f64,
        viscosity: cfg.viscosity as f64,
        gravity: [
            cfg.gravity[0] as f64,
            cfg.gravity[1] as f64,
            cfg.gravity[2] as f64,
        ],
        domain_min: [
            cfg.domain_min[0] as f64,
            cfg.domain_min[1] as f64,
            cfg.domain_min[2] as f64,
        ],
        domain_max: [
            cfg.domain_max[0] as f64,
            cfg.domain_max[1] as f64,
            cfg.domain_max[2] as f64,
        ],
        wall_restitution: cfg.wall_restitution as f64,
        friction: cfg.friction as f64,
    }
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=sph_boundary_gpu`"]
fn sph_boundary_no_leak_and_settles() {
    let engine = try_engine().expect("GPU adapter required (gated by #[ignore])");
    let cfg = SphConfig::water();
    // An open-top tank built from static boundary particles.
    let tank_min = [-0.25, 0.0, -0.25];
    let tank_max = [0.25, 0.5, 0.25];
    let walls = box_tank_walls(tank_min, tank_max, cfg.h, 2);
    assert!(!walls.is_empty());
    // A fluid block dropped in off-centre.
    let fluid = lattice([-0.20, 0.15, -0.15], [0.05, 0.40, 0.15], cfg.h);
    assert!(!fluid.is_empty());

    let mut sim = SphSimulation::new(&engine, cfg).expect("sph new");
    let queue = engine.queue().unwrap().clone();
    sim.set_scene(&queue, &fluid, &walls).expect("set_scene");
    assert_eq!(sim.fluid_count() as usize, fluid.len());

    let initial_max_y = fluid.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    for _ in 0..200 {
        sim.step_n(&engine, 2.0e-4, 4).expect("step");
    }
    let pos = sim.read_positions(&engine).expect("read");
    assert_eq!(pos.len(), fluid.len(), "fluid count must be stable");

    // No leak: fluid stays inside the tank (boundary particles hold it).
    for (i, p) in pos.iter().enumerate() {
        for &c in p {
            assert!(c.is_finite(), "fluid {i} non-finite");
        }
        assert!(
            p[1] > tank_min[1] - 0.06,
            "fluid {i} sank through floor: y={}",
            p[1]
        );
        assert!(
            p[0] > tank_min[0] - 0.06 && p[0] < tank_max[0] + 0.06,
            "fluid {i} escaped on x: {}",
            p[0]
        );
        assert!(
            p[2] > tank_min[2] - 0.06 && p[2] < tank_max[2] + 0.06,
            "fluid {i} escaped on z: {}",
            p[2]
        );
    }

    // Settled: the column collapsed toward the floor.
    let max_y = pos.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_y < initial_max_y,
        "fluid never settled (max_y={max_y}, initial={initial_max_y})"
    );

    // Free surface is roughly level: the spread of the top decile of
    // particle heights stays within a few smoothing lengths.
    let mut ys: Vec<f32> = pos.iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let top = &ys[ys.len() * 9 / 10..];
    let spread = top.last().unwrap() - top.first().unwrap();
    assert!(
        spread < 8.0 * cfg.h,
        "free surface not level: top-decile spread {spread} > {}",
        8.0 * cfg.h
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=sph_boundary_gpu`"]
fn sph_boundary_cpu_gpu_parity() {
    let engine = try_engine().expect("GPU adapter required (gated by #[ignore])");
    let cfg = SphConfig::water();
    // A small fluid block resting just above a flat boundary slab.
    let fluid = lattice([-0.06, 0.25, -0.06], [0.04, 0.35, 0.04], cfg.h);
    let floor = lattice([-0.16, 0.10, -0.16], [0.14, 0.15, 0.14], cfg.h);
    let queue = engine.queue().unwrap().clone();
    let mut sim = SphSimulation::new(&engine, cfg).expect("sph new");
    sim.set_scene(&queue, &fluid, &floor).expect("set_scene");
    let mut oracle = make_oracle(&cfg, &fluid, &floor);

    let dt = 1.0e-4_f32;
    for _ in 0..20 {
        sim.step_n(&engine, dt, 1).expect("step");
        oracle.step(dt as f64);
    }
    let gpu_pos = sim.read_positions(&engine).expect("read");
    assert_eq!(gpu_pos.len(), fluid.len());
    for (i, (g, o)) in gpu_pos
        .iter()
        .zip(oracle.pos[..fluid.len()].iter())
        .enumerate()
    {
        let drift = (((g[0] as f64) - o[0]).powi(2)
            + ((g[1] as f64) - o[1]).powi(2)
            + ((g[2] as f64) - o[2]).powi(2))
        .sqrt();
        assert!(
            drift < 1.0e-2,
            "fluid[{i}] CPU/GPU drift {drift} exceeds 1e-2"
        );
    }
}
