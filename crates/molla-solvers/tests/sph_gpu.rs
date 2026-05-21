//! Stage F1 — GPU WCSPH smoke + CPU/GPU parity tests.
//!
//! GPU-only; gated behind `#[ignore]` because CI has no wgpu adapter.
//! Run via `make test-gpu P=molla-solvers T=sph_gpu`.
#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_solvers::sph_oracle::SphOracle;
use molla_solvers::{SphConfig, SphSimulation};

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

/// Build a CPU oracle whose parameters mirror `cfg` exactly.
fn make_oracle(cfg: &SphConfig, particles: &[[f32; 3]]) -> SphOracle {
    let n = particles.len();
    SphOracle {
        pos: particles
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect(),
        vel: vec![[0.0; 3]; n],
        n_fluid: n,
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
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=sph_gpu`"]
fn sph_dam_break_stays_bounded() {
    let engine = try_engine().expect("GPU adapter required (gated by #[ignore])");
    let cfg = SphConfig::water();
    let mut sim = SphSimulation::new(&engine, cfg).expect("sph new");
    // A fluid column dropped against one wall of the domain box.
    let particles = lattice([-0.40, 0.05, -0.15], [-0.05, 0.50, 0.15], cfg.h);
    assert!(!particles.is_empty());
    let queue = engine.queue().unwrap().clone();
    sim.set_particles(&queue, &particles).expect("set");

    for _ in 0..200 {
        sim.step_n(&engine, 2.0e-4, 4).expect("step");
    }
    let pos = sim.read_positions(&engine).expect("read");
    assert_eq!(pos.len(), particles.len(), "particle count must be stable");
    for (i, p) in pos.iter().enumerate() {
        for (k, &c) in p.iter().enumerate() {
            assert!(c.is_finite(), "particle {i} axis {k} non-finite: {c}");
            assert!(
                c >= cfg.domain_min[k] - 1e-3 && c <= cfg.domain_max[k] + 1e-3,
                "particle {i} left the domain on axis {k}: {c}"
            );
        }
    }
    // The column must have collapsed: lowest particle near the floor.
    let min_y = pos.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    assert!(
        min_y < 0.2,
        "fluid never settled toward the floor (min_y={min_y})"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=sph_gpu`"]
fn sph_cpu_gpu_density_parity() {
    let engine = try_engine().expect("GPU adapter required (gated by #[ignore])");
    let cfg = SphConfig::water();
    // A small block; density is read after one step (compute_density
    // runs before integrate, so the buffer holds density at the
    // initial positions).
    let particles = lattice([-0.10, 0.30, -0.10], [0.05, 0.45, 0.05], cfg.h);
    let queue = engine.queue().unwrap().clone();
    let mut sim = SphSimulation::new(&engine, cfg).expect("sph new");
    sim.set_particles(&queue, &particles).expect("set");
    sim.step_n(&engine, 1.0e-4, 1).expect("step");
    let gpu_density = sim.read_densities(&engine).expect("read densities");

    let mut oracle = make_oracle(&cfg, &particles);
    oracle.compute_density();

    assert_eq!(gpu_density.len(), oracle.density.len());
    for (i, (g, o)) in gpu_density.iter().zip(oracle.density.iter()).enumerate() {
        let rel = ((*g as f64) - o).abs() / o.max(1e-6);
        assert!(
            rel < 5.0e-3,
            "density[{i}] gpu={g} cpu={o} rel={rel} exceeds 5e-3"
        );
    }
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=sph_gpu`"]
fn sph_cpu_gpu_trajectory_parity() {
    let engine = try_engine().expect("GPU adapter required (gated by #[ignore])");
    let cfg = SphConfig::water();
    let particles = lattice([-0.08, 0.30, -0.08], [0.04, 0.42, 0.04], cfg.h);
    let queue = engine.queue().unwrap().clone();
    let mut sim = SphSimulation::new(&engine, cfg).expect("sph new");
    sim.set_particles(&queue, &particles).expect("set");
    let mut oracle = make_oracle(&cfg, &particles);

    let dt = 1.0e-4_f32;
    for _ in 0..20 {
        sim.step_n(&engine, dt, 1).expect("step");
        oracle.step(dt as f64);
    }
    let gpu_pos = sim.read_positions(&engine).expect("read");
    assert_eq!(gpu_pos.len(), oracle.pos.len());
    for (i, (g, o)) in gpu_pos.iter().zip(oracle.pos.iter()).enumerate() {
        let drift = (((g[0] as f64) - o[0]).powi(2)
            + ((g[1] as f64) - o[1]).powi(2)
            + ((g[2] as f64) - o[2]).powi(2))
        .sqrt();
        assert!(
            drift < 1.0e-2,
            "particle[{i}] CPU/GPU drift {drift} exceeds 1e-2"
        );
    }
}
