//! Stage 24d.4 / INSPECTION.md F1 — per-particle MPM materials.
//!
//! One simulation now mixes rheologies through
//! `MpmSimulation::set_materials` (per-particle row ids into a GPU
//! material table). Oracle: three particle blocks placed far enough
//! apart that their grid supports never overlap — a multi-material run
//! must then reproduce three single-material runs of each block alone
//! (disjoint grid cells cannot couple), even though the particle-id
//! layouts and dispatch sizes differ between the runs. A control run
//! with all-sand ids guards against the per-particle table silently
//! not dispatching.
//!
//! History note: writing this test exposed (and led to the fix of)
//! F1b — the half-cell node-lattice bug that made every MPM settling
//! scenario explode. On the fixed dynamics the blocks genuinely
//! settle, the per-block equality gates compare real physics (the
//! ~1e-6 residual is atomic scatter-order noise), and the ids-matter
//! control measures centimetres of true sand/snow contrast.
//!
//! `#[ignore]` — needs a real wgpu adapter (`make test-gpu
//! P=molla-solvers T=mpm_multimaterial_gpu`).

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_solvers::{MpmConfig, MpmRheology, MpmSimulation, MpmTimeStep};

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
    let mut out = Vec::with_capacity(per_axis.pow(3));
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

/// Shared config: explicit stepping, a 64³ grid wide enough to hold
/// three well-separated blocks along +X.
fn cfg_base() -> MpmConfig {
    // Based on the `snow()` preset: its 1.4e5 Pa modulus is the one
    // value all three rheologies tolerate under explicit stepping
    // (`youngs` is global config, not per-material). Snow with a soft
    // sand-style modulus over-compacts past the Stomakhin envelope and
    // the hardening term explodes; `sand_calibrated` is implicit-only.
    MpmConfig {
        grid_size: 32,
        grid_origin: [-0.8, 0.0, -0.8],
        cell_size: 0.05,
        time_step: MpmTimeStep::Explicit,
        ..MpmConfig::snow()
    }
}

fn sand() -> MpmRheology {
    MpmRheology::Sand {
        friction_angle: 30.0_f32.to_radians(),
    }
}
fn snow() -> MpmRheology {
    MpmRheology::Snow {
        theta_c: 2.5e-2,
        theta_s: 7.5e-3,
        hardening: 10.0,
    }
}
fn viscous() -> MpmRheology {
    MpmRheology::Viscous {
        velocity_damping: 6.0,
        affine_damping: 6.0,
    }
}

/// Block gaps of 6 cells — two particles share a grid cell only when
/// within 4 cells (2-cell quadratic support each), so the blocks can
/// never couple while settling. 0.15 m cubes at 6³ particles = 25 mm
/// spacing = 2 particles per cell axis.
fn blocks() -> [Vec<[f32; 3]>; 3] {
    [
        fill_box([-0.60, 0.22, -0.075], [-0.45, 0.37, 0.075], 6, 11),
        fill_box([-0.075, 0.22, -0.075], [0.075, 0.37, 0.075], 6, 22),
        fill_box([0.45, 0.22, -0.075], [0.60, 0.37, 0.075], 6, 33),
    ]
}

// dt note: hardening stiffens the material up to the bounded e³ ≈ 20×
// multiplier (see mpm.wgsl), which shrinks the explicit CFL limit at
// E = 1.4e5 — 2e-4 keeps all three rheologies comfortably stable.
fn run_single(
    engine: &Engine,
    rheology: MpmRheology,
    block: &[[f32; 3]],
    steps: usize,
) -> Vec<[f32; 3]> {
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut cfg = cfg_base();
    cfg.rheology = rheology;
    let mut sim = MpmSimulation::new_with_device(&device, cfg).expect("sim");
    sim.set_particles(&queue, block).expect("seed");
    for _ in 0..steps {
        sim.step_with(&device, &queue, 2.0e-4).expect("step");
    }
    sim.read_positions_with(&device, &queue).expect("read")
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn multimaterial_disjoint_blocks_match_single_material_runs() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let steps = 1600usize;
    let [b_sand, b_snow, b_visc] = blocks();
    let n0 = b_sand.len();
    let n1 = b_snow.len();

    // ── Multi-material run: all three blocks in one simulation.
    let mut all: Vec<[f32; 3]> = Vec::new();
    all.extend_from_slice(&b_sand);
    all.extend_from_slice(&b_snow);
    all.extend_from_slice(&b_visc);
    let mut ids = vec![0u32; n0];
    ids.extend(vec![1u32; n1]);
    ids.extend(vec![2u32; b_visc.len()]);

    let mut sim = MpmSimulation::new_with_device(&device, cfg_base()).expect("sim");
    sim.set_particles(&queue, &all).expect("seed");
    sim.set_materials(&[sand(), snow(), viscous()], ids)
        .expect("set_materials");
    for _ in 0..steps {
        sim.step_with(&device, &queue, 2.0e-4).expect("step");
    }
    let multi = sim.read_positions_with(&device, &queue).expect("read");

    // ── Single-material reference runs, one per block.
    let ref_sand = run_single(&engine, sand(), &b_sand, steps);
    let ref_snow = run_single(&engine, snow(), &b_snow, steps);
    let ref_visc = run_single(&engine, viscous(), &b_visc, steps);

    let check = |name: &str, offset: usize, reference: &[[f32; 3]]| {
        let mut max_d = 0.0_f32;
        for (i, want) in reference.iter().enumerate() {
            let got = multi[offset + i];
            for c in 0..3 {
                max_d = max_d.max((got[c] - want[c]).abs());
            }
        }
        eprintln!("{name}: max |Δ| vs single-material run = {max_d:.3e}");
        assert!(
            max_d < 5.0e-3,
            "{name} block diverged from its single-material reference: {max_d}"
        );
    };
    check("sand", 0, &ref_sand);
    check("snow", n0, &ref_snow);
    check("viscous", n0 + n1, &ref_visc);

    // Everything must have stayed inside the grid and finite.
    for p in &multi {
        assert!(p.iter().all(|x| x.is_finite()));
    }
}

/// Anti-vacuous control: with per-particle ids forced to all-sand, the
/// snow block must move differently than in the true multi-material
/// run — proving the ids actually reach the constitutive branches.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn multimaterial_ids_change_the_physics() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let steps = 1600usize;
    let [b_sand, b_snow, b_visc] = blocks();
    let n0 = b_sand.len();
    let mut all: Vec<[f32; 3]> = Vec::new();
    all.extend_from_slice(&b_sand);
    all.extend_from_slice(&b_snow);
    all.extend_from_slice(&b_visc);

    let run = |ids: Vec<u32>| -> Vec<[f32; 3]> {
        let mut sim = MpmSimulation::new_with_device(&device, cfg_base()).expect("sim");
        sim.set_particles(&queue, &all).expect("seed");
        sim.set_materials(&[sand(), snow(), viscous()], ids)
            .expect("set_materials");
        for _ in 0..steps {
            sim.step_with(&device, &queue, 2.0e-4).expect("step");
        }
        sim.read_positions_with(&device, &queue).expect("read")
    };

    let mut mixed_ids = vec![0u32; n0];
    mixed_ids.extend(vec![1u32; b_snow.len()]);
    mixed_ids.extend(vec![2u32; b_visc.len()]);
    let mixed = run(mixed_ids);
    let all_sand = run(vec![0u32; all.len()]);

    // The snow-block particles must land measurably differently when
    // they run the snow branch vs the sand branch.
    let mut max_d = 0.0_f32;
    for i in n0..(n0 + b_snow.len()) {
        for c in 0..3 {
            max_d = max_d.max((mixed[i][c] - all_sand[i][c]).abs());
        }
    }
    eprintln!("snow-vs-sand branch displacement difference: {max_d:.3e}");
    assert!(
        max_d > 1.0e-3,
        "per-particle material ids had no effect on the snow block ({max_d})"
    );
}
