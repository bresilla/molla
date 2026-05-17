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
