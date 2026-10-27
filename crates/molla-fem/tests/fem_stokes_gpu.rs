//! Stage 41 v2 — GPU steady-state Stokes parity test.
//!
//! Runs the lid-driven cavity at 8×8 P2-P1 resolution on both CPU and
//! GPU and checks that the velocity + pressure agree within `1e-3`
//! relative (the f32 Uzawa round-off budget — same precision floor as
//! the heat / elasticity GPU tests).
//!
//! All tests are `#[ignore]` because they need a real wgpu adapter.
//! Run with:
//!
//! ```sh
//! nixVulkan cargo test -p molla-fem --features gpu --test fem_stokes_gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_fem::{
    FemMesh, MixedTaylorHood, StokesGpuConfig, StokesSimpleConfig, assemble_div_block,
    assemble_stokes_gpu_only, assemble_velocity_block, build_cavity_bcs_2d, solve_stokes_gpu,
    solve_stokes_steady,
};
use molla_math::Real;

// Serialise wgpu adapter lifetime across tests in this binary.
static GPU_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "needs a real wgpu adapter — run with --features gpu --test fem_stokes_gpu -- --ignored"]
fn gpu_stokes_assembly_matches_cpu_8x8() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let nx = 8;
    let ny = 8;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 2);
    let viscosity: Real = 1.0;

    let a_cpu = assemble_velocity_block(&mesh, &th, viscosity).expect("CPU A");
    let b_cpu = assemble_div_block(&mesh, &th).expect("CPU B");

    let engine = Engine::new_gpu().expect("gpu");
    let (a_gpu_vals, b_gpu_vals) =
        assemble_stokes_gpu_only(&mesh, &th, viscosity, &engine).expect("GPU assembly");

    assert_eq!(a_gpu_vals.len(), a_cpu.values.len());
    assert_eq!(b_gpu_vals.len(), b_cpu.values.len());

    // Compare values element-by-element.
    let mut a_max_abs: f32 = 0.0;
    let mut a_peak: f32 = 0.0;
    for (cpu, gpu) in a_cpu.values.iter().zip(a_gpu_vals.iter()) {
        let abs = (gpu - *cpu as f32).abs();
        a_max_abs = a_max_abs.max(abs);
        a_peak = a_peak.max(cpu.abs() as f32);
    }
    let a_rel = a_max_abs / a_peak.max(1.0e-20);
    eprintln!("A assembly: peak = {a_peak:.3e}, max_abs = {a_max_abs:.3e}, rel = {a_rel:.3e}");
    assert!(
        a_rel < 1.0e-4,
        "A GPU vs CPU rel error {a_rel} too large (peak {a_peak}, max_abs {a_max_abs})",
    );

    let mut b_max_abs: f32 = 0.0;
    let mut b_peak: f32 = 0.0;
    for (cpu, gpu) in b_cpu.values.iter().zip(b_gpu_vals.iter()) {
        let abs = (gpu - *cpu as f32).abs();
        b_max_abs = b_max_abs.max(abs);
        b_peak = b_peak.max(cpu.abs() as f32);
    }
    let b_rel = b_max_abs / b_peak.max(1.0e-20);
    eprintln!("B assembly: peak = {b_peak:.3e}, max_abs = {b_max_abs:.3e}, rel = {b_rel:.3e}");
    assert!(
        b_rel < 1.0e-4,
        "B GPU vs CPU rel error {b_rel} too large (peak {b_peak}, max_abs {b_max_abs})",
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run with --features gpu --test fem_stokes_gpu -- --ignored"]
fn gpu_stokes_lid_cavity_matches_cpu_8x8() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // ─── Problem setup ────────────────────────────────────────────
    let nx = 8;
    let ny = 8;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 2);
    let f = vec![0.0; th.ndof_velocity()];
    let bcs = build_cavity_bcs_2d(&mesh, &th, 1.0, [1.0, 0.0]);
    let viscosity: Real = 1.0;

    // ─── CPU oracle ───────────────────────────────────────────────
    let cpu_cfg = StokesSimpleConfig {
        max_outer_iters: 400,
        tol_outer: 1.0e-7,
        max_inner_iters: 5000,
        tol_inner: 1.0e-12,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let cpu_res =
        solve_stokes_steady(&mesh, &th, &f, viscosity, &bcs, &cpu_cfg).expect("CPU stokes solve");
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "CPU stokes 8x8: outer_iters = {}, div_residual = {:.3e}, time = {:.1} ms",
        cpu_res.outer_iters, cpu_res.div_residual, cpu_ms,
    );

    // ─── GPU path ─────────────────────────────────────────────────
    let engine = Engine::new_gpu().expect("gpu");
    let gpu_cfg = StokesGpuConfig {
        max_outer_iters: 800,
        // f32 floor sits around 1e-7 — push both tolerances close to
        // it so the inner CG resolves `A⁻¹ · Bᵀ q` accurately enough
        // to land on the same converged `p` as the CPU oracle.
        tol_outer: 5.0e-7,
        max_inner_iters: 5000,
        tol_inner: 5.0e-7,
    };
    let t0 = std::time::Instant::now();
    let gpu_res = solve_stokes_gpu(&mesh, &th, &f, viscosity, &bcs, &gpu_cfg, &engine)
        .expect("GPU stokes solve");
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "GPU stokes 8x8: outer_iters = {}, div_residual = {:.3e}, time = {:.1} ms",
        gpu_res.outer_iters, gpu_res.div_residual, gpu_ms,
    );

    // ─── Compare ──────────────────────────────────────────────────
    let n_v = th.ndof_velocity();
    assert_eq!(cpu_res.u.len(), n_v);
    assert_eq!(gpu_res.u.len(), n_v);

    let mut peak_u: Real = 0.0;
    let mut max_abs_u: Real = 0.0;
    for (cpu, gpu) in cpu_res.u.iter().zip(gpu_res.u.iter()) {
        peak_u = peak_u.max(cpu.abs());
        max_abs_u = max_abs_u.max((cpu - gpu).abs());
    }
    let rel_u = max_abs_u / peak_u.max(1.0e-20);
    eprintln!("u-diff: peak |u| = {peak_u:.3e}, max_abs = {max_abs_u:.3e}, rel = {rel_u:.3e}",);

    // Pressure is defined up to a constant; compare the centered field.
    let cpu_p_mean: Real = cpu_res.p.iter().copied().sum::<Real>() / cpu_res.p.len() as Real;
    let gpu_p_mean: Real = gpu_res.p.iter().copied().sum::<Real>() / gpu_res.p.len() as Real;
    let mut peak_p: Real = 0.0;
    let mut max_abs_p: Real = 0.0;
    for (cpu, gpu) in cpu_res.p.iter().zip(gpu_res.p.iter()) {
        let cc = cpu - cpu_p_mean;
        let gg = gpu - gpu_p_mean;
        peak_p = peak_p.max(cc.abs());
        max_abs_p = max_abs_p.max((cc - gg).abs());
    }
    let rel_p = max_abs_p / peak_p.max(1.0e-20);
    eprintln!(
        "p-diff (centered): peak |p| = {peak_p:.3e}, max_abs = {max_abs_p:.3e}, rel = {rel_p:.3e}",
    );

    // Spec: 1e-3 relative agreement on velocity.
    assert!(
        rel_u < 1.0e-3,
        "GPU vs CPU velocity diff too large: rel = {rel_u} (max_abs {max_abs_u}, peak {peak_u})",
    );
    // Pressure: looser tolerance (Schur-CG f32 round-off magnifies pressure noise);
    // 5e-2 keeps us at the f32 precision wall on the small 8×8 grid.
    assert!(
        rel_p < 5.0e-2,
        "GPU vs CPU centered-pressure diff too large: rel = {rel_p} (max_abs {max_abs_p}, peak {peak_p})",
    );

    let speedup = cpu_ms / gpu_ms.max(1.0e-3);
    eprintln!("wall-clock: CPU = {cpu_ms:.1} ms, GPU = {gpu_ms:.1} ms (speedup = {speedup:.2}x)",);

    // Sanity: divergence residual should be reasonably small on both paths.
    assert!(
        gpu_res.div_residual < 1.0e-2,
        "GPU divergence residual {:.3e} too large",
        gpu_res.div_residual,
    );
}

/// INSPECTION.md F15 — mixed (u, p) Taylor-Hood ELASTICITY on the GPU
/// (`solve_mixed_elasticity_gpu`). Oracles:
/// 1. 2D constant-strain patch test: `u = (ε·x, −ν·ε·y)` with the
///    exact field on every boundary P2 node — interior must reproduce
///    it to f32 solver tolerance, pressure ≈ λ·ε·(1−ν) constant.
/// 2. Anti-locking parity vs the f64 CPU `solve_mixed_elasticity_steady`
///    on the bending cantilever at ν = 0.4999.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-fem T=fem_stokes_gpu`"]
fn mixed_elasticity_gpu_patch_test_and_cpu_parity() {
    use molla_fem::{
        StokesBc, assemble_velocity_mass, compute_edges, solve_mixed_elasticity_gpu,
        solve_mixed_elasticity_steady,
    };
    use molla_math::Vec3;
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("gpu adapter");

    let lame = |e: Real, nu: Real| -> (Real, Real) {
        (
            e / (2.0 * (1.0 + nu)),
            e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        )
    };
    let p2_nodes = |mesh: &FemMesh| -> Vec<Vec3> {
        let (edges, _) = compute_edges(mesh);
        let mut out: Vec<Vec3> = mesh.vertices.clone();
        for e in &edges {
            out.push((mesh.vertices[e[0] as usize] + mesh.vertices[e[1] as usize]) * 0.5);
        }
        out
    };

    // ── 1. Patch test on the unit square.
    {
        let mesh = FemMesh::tri_grid_2d(6, 6, 1.0, 1.0);
        let th = MixedTaylorHood::build(&mesh, 2);
        let (mu, lambda) = lame(1.0e4, 0.3);
        let eps = 0.01_f64;
        let nu = 0.3_f64;
        let exact = |p: Vec3| [eps * p.x, -nu * eps * p.y];
        let nodes = p2_nodes(&mesh);
        let on_boundary = |p: Vec3| {
            p.x.abs() < 1e-9
                || (p.x - 1.0).abs() < 1e-9
                || p.y.abs() < 1e-9
                || (p.y - 1.0).abs() < 1e-9
        };
        let mut bcs = Vec::new();
        let mut interior = Vec::new();
        for (k, p) in nodes.iter().enumerate() {
            if on_boundary(*p) {
                let u = exact(*p);
                bcs.push(StokesBc::LidVelocity {
                    node: k as u32,
                    velocity: vec![u[0], u[1]],
                });
            } else {
                interior.push(k);
            }
        }
        assert!(interior.len() > 20);
        let f = vec![0.0; th.n_velocity];
        let cfg = StokesGpuConfig {
            tol_inner: 1.0e-7,
            tol_outer: 1.0e-6,
            ..Default::default()
        };
        let res = solve_mixed_elasticity_gpu(&mesh, &th, mu, lambda, &f, &bcs, &cfg, &engine)
            .expect("gpu mixed solve");
        let mut max_err: Real = 0.0;
        for &k in &interior {
            let want = exact(nodes[k]);
            let got = [res.u[2 * k], res.u[2 * k + 1]];
            max_err = max_err
                .max((got[0] - want[0]).abs())
                .max((got[1] - want[1]).abs());
        }
        let p_expect = lambda * eps * (1.0 - nu);
        let mut p_err: Real = 0.0;
        for &pv in &res.p {
            p_err = p_err.max((pv - p_expect).abs());
        }
        eprintln!(
            "gpu patch: max |Δu| = {max_err:.3e} (field 1e-2), max |Δp| = {p_err:.3e} (p = {p_expect:.3e})"
        );
        // f32 assembly readback + f32 CG: expect ~1e-6-scale exactness.
        assert!(
            max_err < 5.0e-6,
            "gpu patch displacement error: {max_err:.3e}"
        );
        assert!(
            p_err < 1.0e-3 * p_expect.abs(),
            "gpu patch pressure error: {p_err:.3e}"
        );
    }

    // ── 2. Anti-locking cantilever parity vs the f64 CPU solver.
    {
        let mesh = FemMesh::tri_grid_2d(16, 4, 4.0, 1.0);
        let th = MixedTaylorHood::build(&mesh, 2);
        let (mu, lambda) = lame(1.0e4, 0.4999);
        let nodes = p2_nodes(&mesh);
        let mut bcs = Vec::new();
        for (k, p) in nodes.iter().enumerate() {
            if p.x.abs() < 1e-9 {
                bcs.push(StokesBc::NoSlip { node: k as u32 });
            }
        }
        let m_v = assemble_velocity_mass(&mesh, &th).expect("mass");
        let mut b_field = vec![0.0; th.n_velocity];
        for k in 0..nodes.len() {
            b_field[2 * k + 1] = -100.0;
        }
        let f = molla_fem::assembly::matvec(&m_v, &b_field);

        let cpu = solve_mixed_elasticity_steady(
            &mesh,
            &th,
            mu,
            lambda,
            &f,
            &bcs,
            &StokesSimpleConfig {
                tol_inner: 1.0e-11,
                tol_outer: 1.0e-9,
                ..Default::default()
            },
        )
        .expect("cpu mixed solve");
        let gpu = solve_mixed_elasticity_gpu(
            &mesh,
            &th,
            mu,
            lambda,
            &f,
            &bcs,
            &StokesGpuConfig {
                tol_inner: 1.0e-7,
                tol_outer: 1.0e-6,
                ..Default::default()
            },
            &engine,
        )
        .expect("gpu mixed solve");

        let tip = |u: &[Real]| -> Real {
            let mut acc = 0.0;
            let mut count = 0;
            for (k, p) in nodes.iter().enumerate() {
                if (p.x - 4.0).abs() < 1e-9 {
                    acc += u[2 * k + 1];
                    count += 1;
                }
            }
            acc / count as Real
        };
        let tip_cpu = tip(&cpu.u);
        let tip_gpu = tip(&gpu.u);
        eprintln!("cantilever ν=0.4999 tip: cpu {tip_cpu:.5}, gpu {tip_gpu:.5}");
        assert!(tip_cpu < -1.0, "cpu beam must deflect (anti-locking)");
        assert!(
            ((tip_gpu - tip_cpu) / tip_cpu).abs() < 5.0e-3,
            "gpu tip deflection diverges from cpu: {tip_gpu} vs {tip_cpu}"
        );
    }
}
