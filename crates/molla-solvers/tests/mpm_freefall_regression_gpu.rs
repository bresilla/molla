//! F1b regression (INSPECTION.md) — the MPM node-lattice fix.
//!
//! The quadratic-weight convention (`base = floor(xp − 0.5)`,
//! `fx = xp − base`) puts grid nodes on the INTEGER lattice. A
//! half-cell "+0.5" offset in the node positions shifts every APIC
//! moment arm, so `Σ w·(xn − xp)` becomes `0.5·dx` instead of 0 and a
//! uniformly-moving particle manufactures `C ≈ 2·v/dx` of spurious
//! affine velocity — which feeds F, then stress, and explodes at
//! ×~5/substep. These gates hold the exactness properties that made
//! the bug visible:
//!
//! 1. A single particle in free fall must track `v = g·t` and never
//!    drift laterally.
//! 2. A dropped block must settle inside its column, not paint the
//!    grid walls.
#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_solvers::{MpmConfig, MpmSimulation};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn fill_box(lo: [f32; 3], hi: [f32; 3], per_axis: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut out = Vec::new();
    let d = [
        (hi[0] - lo[0]) / per_axis as f32,
        (hi[1] - lo[1]) / per_axis as f32,
        (hi[2] - lo[2]) / per_axis as f32,
    ];
    for ix in 0..per_axis {
        for iy in 0..per_axis {
            for iz in 0..per_axis {
                out.push([
                    lo[0] + (ix as f32 + 0.5) * d[0] + 0.05 * d[0] * next(),
                    lo[1] + (iy as f32 + 0.5) * d[1] + 0.05 * d[1] * next(),
                    lo[2] + (iz as f32 + 0.5) * d[2] + 0.05 * d[2] * next(),
                ]);
            }
        }
    }
    out
}

fn run_cfg(two_pass: bool, steps: usize, gravity_y: f32, per_axis: usize) -> (f32, f32) {
    let engine = try_engine().expect("adapter");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut cfg = MpmConfig::snow();
    cfg.gravity_y = gravity_y;
    let mut sim = MpmSimulation::new_with_device(&device, cfg).expect("sim");
    sim.set_p2g_mode(two_pass);
    let pts = fill_box([-0.075, 0.12, -0.075], [0.075, 0.27, 0.075], per_axis, 22);
    sim.set_particles(&queue, &pts).expect("seed");
    for _ in 0..steps {
        sim.step_with(&device, &queue, 1.0 / 480.0).expect("step");
    }
    let pos = sim.read_positions_with(&device, &queue).expect("read");
    let mut hi_y = f32::MIN;
    let mut hi_x = f32::MIN;
    for p in &pos {
        hi_y = hi_y.max(p[1]);
        hi_x = hi_x.max(p[0].abs());
    }
    (hi_y, hi_x)
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn single_particle_free_fall_is_exact() {
    let engine = try_engine().expect("adapter");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let cfg = MpmConfig::snow();
    let mut sim = MpmSimulation::new_with_device(&device, cfg).expect("sim");
    sim.set_particles(&queue, &[[0.013, 0.2, 0.017]])
        .expect("seed");
    let dt = 1.0 / 480.0;
    for step in 0..20 {
        sim.step_with(&device, &queue, dt).expect("step");
        let pos = sim.read_positions_with(&device, &queue).expect("pos");
        let vel = sim.read_velocities_with(&device, &queue).expect("vel");
        let vy_expect = -9.81 * dt * (step as f32 + 1.0);
        assert!(
            (vel[0][1] - vy_expect).abs() < 1.0e-4,
            "step {step}: vy = {} but free fall = {vy_expect}",
            vel[0][1]
        );
        // The pre-fix bug manufactured lateral velocity ∝ |v| within
        // a couple of substeps — 1e-5 is far below any legitimate
        // dynamics and far above f32 noise for these magnitudes.
        assert!(
            vel[0][0].abs() < 1.0e-5 && vel[0][2].abs() < 1.0e-5,
            "step {step}: lateral drift ({}, {})",
            vel[0][0],
            vel[0][2]
        );
        assert!(
            (pos[0][0] - 0.013).abs() < 1.0e-5 && (pos[0][2] - 0.017).abs() < 1.0e-5,
            "step {step}: x/z moved"
        );
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn dropped_block_settles_in_its_column() {
    for two_pass in [true, false] {
        let (hi_y, hi_abs_x) = run_cfg(two_pass, 300, -9.81, 6);
        eprintln!("two_pass={two_pass}: max y = {hi_y:.3}, max |x| = {hi_abs_x:.3}");
        // Block seeded in |x| ≤ 0.075, y ≤ 0.27. Settling compacts it
        // downward with at most mild lateral spreading; painting the
        // grid walls (|x| ≈ 0.65, y ≈ 1.45) is the explosion signature.
        assert!(hi_y < 0.35, "two_pass={two_pass}: block flew: y = {hi_y}");
        assert!(
            hi_abs_x < 0.25,
            "two_pass={two_pass}: block sprayed: |x| = {hi_abs_x}"
        );
    }
}
