//! INSPECTION.md F15 / DiffsimCloth — GPU cloth adjoint gradcheck.
//!
//! The forward is gather-form and bit-deterministic, so central
//! finite differences of the GPU forward itself are a valid oracle
//! for the analytic reverse kernels. Loss: `J = Σ ‖p_T‖²` (adjoint
//! seed `2·p_T`); gradients checked for spring stiffness ∂J/∂k and
//! initial velocity ∂J/∂v₀ on sampled entries.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{ClothDiffGpu, ClothSpring};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

struct Setup {
    positions: Vec<[f32; 3]>,
    velocities: Vec<[f32; 3]>,
    inv_mass: Vec<f32>,
    springs: Vec<ClothSpring>,
}

fn cloth_grid(n: usize) -> Setup {
    let spacing = 0.1_f32;
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut inv_mass = Vec::new();
    for j in 0..n {
        for i in 0..n {
            positions.push([i as f32 * spacing, 0.0, j as f32 * spacing]);
            velocities.push([0.02 * i as f32, 0.0, -0.01 * j as f32]);
            // Pin the two far corners of the top row.
            let pinned = j == 0 && (i == 0 || i == n - 1);
            inv_mass.push(if pinned { 0.0 } else { 1.0 / 0.05 });
        }
    }
    let idx = |i: usize, j: usize| (j * n + i) as u32;
    let mut springs = Vec::new();
    let mut push = |a: u32, b: u32| {
        let pa = positions[a as usize];
        let pb = positions[b as usize];
        // Pre-tension: rest below the initial length so every spring
        // carries force from step 0 — near-rest springs give stiffness
        // gradients below the f32 FD noise floor.
        let rest = 0.87
            * ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt();
        springs.push(ClothSpring {
            a,
            b,
            rest,
            k: 60.0 + 2.0 * (a as f32 % 7.0),
        });
    };
    for j in 0..n {
        for i in 0..n {
            if i + 1 < n {
                push(idx(i, j), idx(i + 1, j));
            }
            if j + 1 < n {
                push(idx(i, j), idx(i, j + 1));
            }
            if i + 1 < n && j + 1 < n {
                push(idx(i, j), idx(i + 1, j + 1));
            }
        }
    }
    Setup {
        positions,
        velocities,
        inv_mass,
        springs,
    }
}

fn loss(sim: &ClothDiffGpu, steps: u32) -> f64 {
    sim.forward(steps).expect("forward");
    let p = sim.read_positions(steps).expect("read");
    p.iter()
        .map(|p| (p[0] as f64).powi(2) + (p[1] as f64).powi(2) + (p[2] as f64).powi(2))
        .sum()
}

#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-kernels T=cloth_diff_gpu`"]
fn cloth_adjoint_matches_finite_differences() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let setup = cloth_grid(6);
    let steps = 30u32;
    let dt = 1.0 / 240.0;
    let gravity = [0.0, -9.81, 0.0];

    let mk = |s: &Setup| {
        ClothDiffGpu::new(
            &device,
            &queue,
            &s.positions,
            &s.velocities,
            &s.inv_mass,
            &s.springs,
            steps,
            dt,
            gravity,
        )
        .expect("cloth diff")
    };

    // ── Analytic gradients.
    let sim = mk(&setup);
    let base_loss = loss(&sim, steps);
    let p_t = sim.read_positions(steps).expect("read");
    let seed_p: Vec<[f32; 3]> = p_t
        .iter()
        .map(|p| [2.0 * p[0], 2.0 * p[1], 2.0 * p[2]])
        .collect();
    let seed_v = vec![[0.0_f32; 3]; setup.positions.len()];
    sim.backward(steps, &seed_p, &seed_v).expect("backward");
    let grad_k = sim.read_grad_k().expect("grad k");
    let grad_v0 = sim.read_grad_v0().expect("grad v0");
    eprintln!("J = {base_loss:.6}");

    // ── FD oracle: stiffness — check the five LARGEST analytic
    // gradients (fixed indices can land on near-zero-signal springs
    // whose FD drowns in f32 noise).
    let mut order: Vec<usize> = (0..setup.springs.len()).collect();
    order.sort_by(|&x, &y| {
        grad_k[y]
            .abs()
            .partial_cmp(&grad_k[x].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut checked_k = 0;
    for &s_idx in order.iter().take(5) {
        let h = (setup.springs[s_idx].k * 5.0e-3).max(1.0e-2);
        let mut ks: Vec<f32> = setup.springs.iter().map(|s| s.k).collect();
        ks[s_idx] += h;
        let sim_p = mk(&setup);
        sim_p.set_stiffness(&ks).expect("set k");
        let jp = loss(&sim_p, steps);
        ks[s_idx] -= 2.0 * h;
        let sim_m = mk(&setup);
        sim_m.set_stiffness(&ks).expect("set k");
        let jm = loss(&sim_m, steps);
        let fd = (jp - jm) / (2.0 * h as f64);
        let got = grad_k[s_idx] as f64;
        let denom = fd.abs().max(got.abs());
        // f32 forward → the FD difference carries ~1e-7-scale noise in
        // J; skip entries whose true signal drowns in it.
        if denom < 3.0e-5 {
            continue;
        }
        let rel = (fd - got).abs() / denom;
        eprintln!("dJ/dk[{s_idx}]: analytic {got:.6e} fd {fd:.6e} rel {rel:.3e}");
        assert!(
            rel < 2.0e-2,
            "stiffness gradient mismatch at {s_idx}: {rel:.3e}"
        );
        checked_k += 1;
    }
    assert!(checked_k >= 3, "too few non-trivial stiffness checks");

    // ── FD oracle: initial velocity (sampled dofs).
    let mut checked_v = 0;
    for (pi, c) in [(8usize, 0usize), (14, 1), (21, 2), (27, 0)] {
        if setup.inv_mass[pi] == 0.0 {
            continue;
        }
        let h = 1.0e-3_f32;
        let mut vp = setup.velocities.clone();
        vp[pi][c] += h;
        let s_p = Setup {
            velocities: vp.clone(),
            ..Setup {
                positions: setup.positions.clone(),
                velocities: vp,
                inv_mass: setup.inv_mass.clone(),
                springs: setup.springs.clone(),
            }
        };
        let jp = loss(&mk(&s_p), steps);
        let mut vm = setup.velocities.clone();
        vm[pi][c] -= h;
        let s_m = Setup {
            positions: setup.positions.clone(),
            velocities: vm,
            inv_mass: setup.inv_mass.clone(),
            springs: setup.springs.clone(),
        };
        let jm = loss(&mk(&s_m), steps);
        let fd = (jp - jm) / (2.0 * f64::from(h));
        let got = grad_v0[pi][c] as f64;
        let denom = fd.abs().max(got.abs());
        if denom < 1.0e-6 {
            continue;
        }
        let rel = (fd - got).abs() / denom;
        eprintln!("dJ/dv0[{pi}][{c}]: analytic {got:.6e} fd {fd:.6e} rel {rel:.3e}");
        assert!(
            rel < 2.0e-2,
            "v0 gradient mismatch at [{pi}][{c}]: {rel:.3e}"
        );
        checked_v += 1;
    }
    assert!(checked_v >= 3, "too few non-trivial v0 checks");
}

/// Softbody volume-term adjoint gradcheck: a 5-tet cube (edge springs
/// plus per-tet volume preservation) squeezed by inward initial
/// velocities. FD on ∂J/∂k_v (all five tets carry signal since the
/// cube is compressing) and re-check ∂J/∂v₀ with the volume term
/// active — the adjoint must chain through BOTH force families.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn softbody_volume_adjoint_matches_finite_differences() {
    let Some(engine) = try_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    // Unit cube, 8 corners, 5-tet decomposition.
    let positions: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let center = [0.5_f32, 0.5, 0.5];
    let velocities: Vec<[f32; 3]> = positions
        .iter()
        .map(|p| {
            [
                -0.8 * (p[0] - center[0]),
                -0.8 * (p[1] - center[1]),
                -0.8 * (p[2] - center[2]),
            ]
        })
        .collect();
    let inv_mass = vec![1.0 / 0.1; 8];
    let tets_ids: [[u32; 4]; 5] = [
        [0, 1, 3, 5],
        [0, 3, 2, 6],
        [0, 5, 4, 6],
        [3, 5, 6, 7],
        [0, 3, 5, 6],
    ];
    let mk_setup = |kvs: &[f32; 5], vels: &[[f32; 3]]| -> ClothDiffGpu {
        let mut springs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for t in &tets_ids {
            for a in 0..4 {
                for b in (a + 1)..4 {
                    let (x, y) = (t[a].min(t[b]), t[a].max(t[b]));
                    if seen.insert((x, y)) {
                        let pa = positions[x as usize];
                        let pb = positions[y as usize];
                        let rest = ((pa[0] - pb[0]).powi(2)
                            + (pa[1] - pb[1]).powi(2)
                            + (pa[2] - pb[2]).powi(2))
                        .sqrt();
                        springs.push(ClothSpring {
                            a: x,
                            b: y,
                            rest,
                            k: 40.0,
                        });
                    }
                }
            }
        }
        let tets: Vec<molla_kernels::SoftTet> = tets_ids
            .iter()
            .zip(kvs.iter())
            .map(|(ids, &kv)| molla_kernels::SoftTet { ids: *ids, kv })
            .collect();
        ClothDiffGpu::new_with_tets(
            &device,
            &queue,
            &positions,
            vels,
            &inv_mass,
            &springs,
            &tets,
            40,
            1.0 / 240.0,
            [0.0, 0.0, 0.0],
        )
        .expect("softbody diff")
    };

    let steps = 40u32;
    let base_kv = [300.0_f32, 320.0, 340.0, 360.0, 500.0];
    let sim = mk_setup(&base_kv, &velocities);
    let base_loss = loss(&sim, steps);
    let p_t = sim.read_positions(steps).expect("read");
    let seed_p: Vec<[f32; 3]> = p_t
        .iter()
        .map(|p| [2.0 * p[0], 2.0 * p[1], 2.0 * p[2]])
        .collect();
    sim.backward(steps, &seed_p, &[[0.0_f32; 3]; 8])
        .expect("backward");
    let grad_kv = sim.read_grad_kv().expect("grad kv");
    let grad_v0 = sim.read_grad_v0().expect("grad v0");
    eprintln!("softbody J = {base_loss:.6}, grad_kv = {grad_kv:?}");

    let mut checked = 0;
    for t_idx in 0..5 {
        let h = base_kv[t_idx] * 5.0e-3;
        let mut kp = base_kv;
        kp[t_idx] += h;
        let jp = loss(&mk_setup(&kp, &velocities), steps);
        let mut km = base_kv;
        km[t_idx] -= h;
        let jm = loss(&mk_setup(&km, &velocities), steps);
        let fd = (jp - jm) / (2.0 * f64::from(h));
        let got = f64::from(grad_kv[t_idx]);
        let denom = fd.abs().max(got.abs());
        if denom < 3.0e-5 {
            continue;
        }
        let rel = (fd - got).abs() / denom;
        eprintln!("dJ/dkv[{t_idx}]: analytic {got:.6e} fd {fd:.6e} rel {rel:.3e}");
        assert!(rel < 2.0e-2, "kv gradient mismatch at {t_idx}: {rel:.3e}");
        checked += 1;
    }
    assert!(checked >= 3, "too few non-trivial kv checks: {checked}");

    // v0 gradients with the volume term active.
    let mut checked_v = 0;
    for (pi, c) in [(1usize, 0usize), (6, 1), (7, 2)] {
        let h = 1.0e-3_f32;
        let mut vp = velocities.clone();
        vp[pi][c] += h;
        let jp = loss(&mk_setup(&base_kv, &vp), steps);
        let mut vm = velocities.clone();
        vm[pi][c] -= h;
        let jm = loss(&mk_setup(&base_kv, &vm), steps);
        let fd = (jp - jm) / (2.0 * f64::from(h));
        let got = f64::from(grad_v0[pi][c]);
        let denom = fd.abs().max(got.abs());
        if denom < 3.0e-5 {
            continue;
        }
        let rel = (fd - got).abs() / denom;
        eprintln!("dJ/dv0[{pi}][{c}]: analytic {got:.6e} fd {fd:.6e} rel {rel:.3e}");
        assert!(rel < 2.0e-2, "v0 gradient mismatch: {rel:.3e}");
        checked_v += 1;
    }
    assert!(checked_v >= 2, "too few v0 checks");
}
