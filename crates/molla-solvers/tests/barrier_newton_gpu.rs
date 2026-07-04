//! C0 phase 2 — GPU barrier-Newton parity vs the f64 CPU oracle.
//!
//! Same scenes as `barrier_contact.rs` (the phase-1 contract), run on
//! `BarrierNewtonGpu` (f32, device gradient/Hessian/PCG/TOI):
//!
//! 1. Drop-and-drape on plane + sphere: the INVARIANT (min gap > 0 at
//!    every accepted iterate) must hold on device, the resting gap
//!    must sit inside the barrier band, and the settled positions must
//!    match the f64 CPU oracle (f32 parity gate).
//! 2. High-speed 50 m/s impact — the tunneling case: penetration-free
//!    on every step, arrested on the surface.
//!
//! Every test ends with an explicit intersection scan.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::Vec3;
use molla_solvers::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring};

use molla_kernels::{BarrierGpuObstacle, BarrierGpuParams, BarrierGpuSpring, BarrierNewtonGpu};

fn cloth_patch(n: usize, y: f64, spacing: f64) -> (Vec<[f64; 3]>, Vec<f64>, Vec<(u32, u32)>) {
    let mut pos = Vec::new();
    let mut mass = Vec::new();
    for j in 0..n {
        for i in 0..n {
            pos.push([
                (i as f64 - (n as f64 - 1.0) / 2.0) * spacing,
                y,
                (j as f64 - (n as f64 - 1.0) / 2.0) * spacing,
            ]);
            mass.push(0.02);
        }
    }
    let idx = |i: usize, j: usize| (j * n + i) as u32;
    let mut edges = Vec::new();
    for j in 0..n {
        for i in 0..n {
            if i + 1 < n {
                edges.push((idx(i, j), idx(i + 1, j)));
            }
            if j + 1 < n {
                edges.push((idx(i, j), idx(i, j + 1)));
            }
            if i + 1 < n && j + 1 < n {
                edges.push((idx(i, j), idx(i + 1, j + 1)));
                edges.push((idx(i + 1, j), idx(i, j + 1)));
            }
        }
    }
    (pos, mass, edges)
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=barrier_newton_gpu`"]
fn gpu_drop_matches_cpu_oracle_and_keeps_invariant() {
    let engine = Engine::new_gpu().expect("gpu adapter");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let (pos, mass, edges) = cloth_patch(8, 0.15, 0.03);
    let k = 400.0_f64;
    let springs_cpu: Vec<BarrierSpring> = edges
        .iter()
        .map(|&(a, b)| {
            let pa = Vec3::from_array(pos[a as usize]);
            let pb = Vec3::from_array(pos[b as usize]);
            BarrierSpring {
                a,
                b,
                rest: (pa - pb).length(),
                k,
            }
        })
        .collect();
    let obstacles_cpu = vec![
        BarrierObstacle::HalfSpace {
            n: Vec3::Y,
            offset: 0.0,
        },
        BarrierObstacle::Sphere {
            center: Vec3::ZERO,
            radius: 0.06,
        },
    ];
    let mut cpu = BarrierShell::new(
        pos.iter().map(|p| Vec3::from_array(*p)).collect(),
        mass.clone(),
        springs_cpu.clone(),
        obstacles_cpu,
        BarrierParams::default(),
    )
    .expect("cpu shell");

    let springs_gpu: Vec<BarrierGpuSpring> = springs_cpu
        .iter()
        .map(|s| BarrierGpuSpring {
            a: s.a,
            b: s.b,
            rest: s.rest as f32,
            k: s.k as f32,
        })
        .collect();
    let pos_f32: Vec<[f32; 3]> = pos
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let mass_f32: Vec<f32> = mass.iter().map(|&m| m as f32).collect();
    let gpu = BarrierNewtonGpu::new(
        &device,
        &queue,
        &pos_f32,
        &mass_f32,
        &springs_gpu,
        &[
            BarrierGpuObstacle::HalfSpace {
                n: [0.0, 1.0, 0.0],
                offset: 0.0,
            },
            BarrierGpuObstacle::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 0.06,
            },
        ],
        BarrierGpuParams::default(),
    )
    .expect("gpu shell");

    let h = 1.0 / 240.0;
    let mut worst_gap = f32::INFINITY;
    for step in 0..240 {
        cpu.step(h).expect("cpu step");
        let stats = gpu.step(h as f32).expect("gpu step");
        worst_gap = worst_gap.min(stats.min_gap);
        assert!(
            stats.min_gap > 0.0,
            "step {step}: GPU INVARIANT violated — gap {}",
            stats.min_gap
        );
    }
    let p_gpu = gpu.read_positions().expect("read");
    let v_gpu = gpu.read_velocities().expect("read v");

    // Settled on device.
    let v_max = v_gpu
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0_f32, f32::max);
    assert!(v_max < 0.08, "gpu shell did not settle: v_max = {v_max}");

    // Resting gap inside the barrier band (device-side scan).
    let gap_final = p_gpu
        .iter()
        .map(|p| {
            let plane = p[1];
            let sph = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 0.06;
            plane.min(sph)
        })
        .fold(f32::INFINITY, f32::min);
    assert!(
        gap_final > 0.0 && gap_final < 1.0e-3,
        "gpu resting gap outside barrier band: {gap_final:.3e}"
    );

    // f32-vs-f64 parity on the settled drape.
    let mut max_err: f64 = 0.0;
    let mut mean_err: f64 = 0.0;
    for (i, p) in p_gpu.iter().enumerate() {
        let c = cpu.positions[i];
        let e = ((p[0] as f64 - c.x).powi(2)
            + (p[1] as f64 - c.y).powi(2)
            + (p[2] as f64 - c.z).powi(2))
        .sqrt();
        max_err = max_err.max(e);
        mean_err += e;
    }
    mean_err /= p_gpu.len() as f64;
    let mean_y_gpu: f64 = p_gpu.iter().map(|p| p[1] as f64).sum::<f64>() / p_gpu.len() as f64;
    let mean_y_cpu: f64 =
        cpu.positions.iter().map(|p| p.y).sum::<f64>() / cpu.positions.len() as f64;
    eprintln!(
        "gpu drop: height bias {:+.3e} (gpu {mean_y_gpu:.4} cpu {mean_y_cpu:.4})",
        mean_y_gpu - mean_y_cpu
    );
    eprintln!(
        "gpu drop: worst gap {worst_gap:.2e}, final gap {gap_final:.2e}, v_max {v_max:.3e}, parity mean {mean_err:.3e} max {max_err:.3e}"
    );
    // The drape on a sphere is NOT a unique equilibrium: f32-vs-f64
    // and their different reachable Newton depths settle into nearby
    // but distinct fold configurations (tightening the f32 tolerance
    // past ~1e-2 relative makes things WORSE — the line search
    // stagnates at noise and jitters). The physically meaningful
    // parity gates are: the invariant + the resting band (asserted
    // above), no systematic height bias (a penetration or floating
    // bug would show here), and same-neighborhood folds.
    // Which state is "more settled"? Compare total potential energy
    // (gravity + springs, f64 on host): the GPU state must be an
    // equally-or-more relaxed equilibrium, not a corrupted one. A
    // penetration/floating bug would RAISE energy (or break the gap
    // band asserted above).
    let pe = |pts: &dyn Fn(usize) -> Vec3| -> f64 {
        let mut e = 0.0;
        for (i, &m) in mass.iter().enumerate() {
            e += m * 9.81 * pts(i).y;
        }
        for s in &springs_cpu {
            let c = (pts(s.a as usize) - pts(s.b as usize)).length() - s.rest;
            e += 0.5 * s.k * c * c;
        }
        e
    };
    let pe_gpu = pe(&|i| {
        Vec3::new(
            f64::from(p_gpu[i][0]),
            f64::from(p_gpu[i][1]),
            f64::from(p_gpu[i][2]),
        )
    });
    let pe_cpu = pe(&|i| cpu.positions[i]);
    eprintln!("gpu drop: PE gpu {pe_gpu:.6} vs cpu {pe_cpu:.6}");
    assert!(
        pe_gpu <= pe_cpu + 0.05 * pe_cpu.abs().max(1.0e-6),
        "gpu state has HIGHER energy than the oracle — not a valid deeper equilibrium: {pe_gpu} vs {pe_cpu}"
    );
    assert!(
        mean_err < 2.0e-2 && max_err < 5.0e-2,
        "gpu drape left the cpu oracle's neighborhood: mean {mean_err:.3e}, max {max_err:.3e}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=barrier_newton_gpu`"]
fn gpu_high_speed_impact_cannot_tunnel() {
    let engine = Engine::new_gpu().expect("gpu adapter");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let (pos, mass, edges) = cloth_patch(6, 0.05, 0.03);
    let springs: Vec<BarrierGpuSpring> = edges
        .iter()
        .map(|&(a, b)| {
            let pa = Vec3::from_array(pos[a as usize]);
            let pb = Vec3::from_array(pos[b as usize]);
            BarrierGpuSpring {
                a,
                b,
                rest: (pa - pb).length() as f32,
                k: 400.0,
            }
        })
        .collect();
    let pos_f32: Vec<[f32; 3]> = pos
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let mass_f32: Vec<f32> = mass.iter().map(|&m| m as f32).collect();
    let gpu = BarrierNewtonGpu::new(
        &device,
        &queue,
        &pos_f32,
        &mass_f32,
        &springs,
        &[BarrierGpuObstacle::HalfSpace {
            n: [0.0, 1.0, 0.0],
            offset: 0.0,
        }],
        BarrierGpuParams::default(),
    )
    .expect("gpu shell");
    gpu.set_velocities(&vec![[0.0, -50.0, 0.0]; pos.len()])
        .expect("set v");

    let h = 1.0_f32 / 60.0;
    for step in 0..30 {
        let stats = gpu.step(h).expect("gpu step");
        assert!(
            stats.min_gap > 0.0,
            "step {step}: GPU tunneled — gap {}",
            stats.min_gap
        );
    }
    let p = gpu.read_positions().expect("read");
    let y_max = p.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    let y_min = p.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    eprintln!("gpu impact arrested: y ∈ [{y_min:.4}, {y_max:.4}]");
    assert!(y_min > 0.0, "final intersection scan failed: y_min {y_min}");
    assert!(y_max < 0.1, "gpu shell should have landed: {y_max}");
}

/// C0 phase 3b — GPU self-collision: the two-layer stack and crossed
/// ribbons from the CPU phase-3 contract, with pairs discovered and
/// resolved ON DEVICE. Final intersection scan runs host-side with the
/// CPU oracle's exact O(V·T + E²) `min_self_gap` on read-back
/// positions — the GPU cannot grade its own homework.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-solvers T=barrier_newton_gpu`"]
fn gpu_self_collision_stacks_and_crosses_without_penetration() {
    let engine = Engine::new_gpu().expect("gpu adapter");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    // Triangulated patch helper (mirrors the CPU phase-3 tests).
    #[allow(clippy::type_complexity)]
    let patch = |n: usize,
                 m: usize,
                 y: f64,
                 origin: [f64; 2]|
     -> (Vec<[f64; 3]>, Vec<(u32, u32)>, Vec<[u32; 3]>) {
        let spacing = 0.03;
        let mut pos = Vec::new();
        for j in 0..m {
            for i in 0..n {
                pos.push([
                    origin[0] + i as f64 * spacing,
                    y,
                    origin[1] + j as f64 * spacing,
                ]);
            }
        }
        let idx = |i: usize, j: usize| (j * n + i) as u32;
        let mut tris = Vec::new();
        for j in 0..m - 1 {
            for i in 0..n - 1 {
                tris.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                tris.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut edges = Vec::new();
        for t in &tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                if seen.insert((a.min(b), a.max(b))) {
                    edges.push((a, b));
                }
            }
        }
        (pos, edges, tris)
    };

    // Two layers: free 5×5 above a pinned 7×7.
    let (mut pos, mut edges, mut tris) = patch(5, 5, 0.02, [-0.06, -0.06]);
    let n_top = pos.len();
    let (pos_b, edges_b, tris_b) = patch(7, 7, 0.0, [-0.09, -0.09]);
    let base = n_top as u32;
    pos.extend(pos_b.iter().copied());
    edges.extend(edges_b.iter().map(|&(a, b)| (a + base, b + base)));
    tris.extend(
        tris_b
            .iter()
            .map(|t| [t[0] + base, t[1] + base, t[2] + base]),
    );
    let n_total = pos.len();

    let springs: Vec<BarrierGpuSpring> = edges
        .iter()
        .map(|&(a, b)| {
            let pa = Vec3::from_array(pos[a as usize]);
            let pb = Vec3::from_array(pos[b as usize]);
            BarrierGpuSpring {
                a,
                b,
                rest: (pa - pb).length() as f32,
                k: 400.0,
            }
        })
        .collect();
    let pos_f32: Vec<[f32; 3]> = pos
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let pinned: Vec<u32> = (n_top as u32..n_total as u32).collect();
    let gpu = BarrierNewtonGpu::new_with_mesh(
        &device,
        &queue,
        &pos_f32,
        &vec![0.02; n_total],
        &springs,
        &[],
        &tris,
        &pinned,
        BarrierGpuParams::default(),
    )
    .expect("gpu shell");

    let h = 1.0_f32 / 240.0;
    for step in 0..200 {
        let stats = gpu.step(h).expect("gpu step");
        assert!(
            stats.min_gap > 0.0,
            "step {step}: GPU layers interpenetrated — gap {}",
            stats.min_gap
        );
    }
    let p = gpu.read_positions().expect("read");
    let v = gpu.read_velocities().expect("read v");
    let y_top: f32 = p[..n_top].iter().map(|q| q[1]).sum::<f32>() / n_top as f32;
    let v_max = v[..n_top]
        .iter()
        .map(|q| (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt())
        .fold(0.0_f32, f32::max);
    eprintln!("gpu two-layer: y_top {y_top:.4}, v_max {v_max:.3e}");
    assert!(v_max < 0.05, "gpu top layer did not settle: {v_max}");
    assert!(
        y_top > 0.0 && y_top < 5.0e-3,
        "gpu top layer should rest just above the pinned layer: {y_top:.4}"
    );

    // HOST-side final intersection scan with the CPU oracle's exact
    // O(V·T + E²) machinery — includes the pinned mesh.
    let mut scan = BarrierShell::new(
        p.iter()
            .map(|q| Vec3::new(f64::from(q[0]), f64::from(q[1]), f64::from(q[2])))
            .collect(),
        vec![0.02; n_total],
        Vec::new(),
        Vec::new(),
        BarrierParams::default(),
    )
    .expect("scan shell")
    .with_triangles(tris.clone());
    scan.velocities = vec![Vec3::ZERO; n_total];
    let final_gap = scan.min_self_gap(&scan.positions);
    eprintln!("gpu two-layer: host-side final scan gap {final_gap:.3e}");
    assert!(
        final_gap > 0.0,
        "host-side intersection scan failed: {final_gap:.3e}"
    );
}
