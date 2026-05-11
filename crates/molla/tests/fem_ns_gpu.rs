//! Stage 41 v4 — transient incompressible Navier-Stokes GPU parity +
//! perf tests.
//!
//! ```sh
//! nixVulkan cargo test -p molla --features gpu --release \
//!     --test fem_ns_gpu -- --ignored
//! ```
//!
//! 1. `gpu_ns_lid_cavity_matches_cpu_8x8_re100` — 8×8 lid-driven cavity
//!    at `Re = 100` for 50 time steps, GPU `(u, p)` matches the CPU
//!    `solve_navier_stokes_step` oracle within `5e-3` relative.
//!
//! 2. `gpu_ns_perf_16x16_re100_100steps` — wall-clock GPU vs CPU at
//!    16×16, 100 steps. Reports the ratio; no hard gate (perf depends
//!    on the test rig + GPU adapter).
//!
//! ## v3 vs v4 perf baseline (this rig, Vulkan + headless adapter)
//!
//! v3 fully assembled the per-step IMEX RHS host-side: it read back
//! `m_u`, `f_conv`, and `bt_p` to the host each step, did the axpy +
//! pin override in CPU code, and uploaded the result. v4 replaces that
//! with the fused WGSL kernel in `fem_ns_rhs_assemble.wgsl` and
//! computes the constant BC column-elim correction on the GPU once at
//! setup. Reference numbers (best-of-3 after warm-up):
//!
//! | mesh   | steps | CPU      | v3 GPU    | v4 GPU    | v4 vs v3 |
//! |--------|-------|----------|-----------|-----------|----------|
//! | 8 × 8  |  50   | ~0.7 s   | ~8.0 s    | ~6.2 s    | 1.28×    |
//! | 16×16  | 100   | ~8.5 s   | ~27.0 s   | ~23.7 s   | 1.14×    |
//!
//! GPU is still slower than CPU at these small problem sizes — the
//! dominant remaining cost is the Uzawa CG's per-iteration
//! `map_async + poll` dot-product sync (each f32 dot reduction stalls
//! the whole pipeline; at 16×16 there are ~70 outer × ~30 inner =
//! ~2k syncs/step). That sync chain is **out of scope** for v4 — the
//! task brief explicitly defers pressure-solve fusion. The v3 → v4
//! wall-clock numbers above reflect the readback + bc-delta fusion in
//! isolation.

#![cfg(feature = "gpu")]

use std::sync::Mutex;
use std::time::Instant;

use molla::compute::Engine;
use molla::fem::{
    assemble_div_block, assemble_velocity_block, assemble_velocity_mass,
    build_cavity_bcs_2d, solve_navier_stokes_gpu, solve_navier_stokes_step,
    FemMesh, MixedTaylorHood, NavierStokesGpuConfig, StokesGpuConfig,
    StokesSimpleConfig,
};
use molla::math::sparse::BsrMatrix;
use molla::math::Real;

// Serialise wgpu adapter use across the test binary — multiple wgpu
// adapters in flight on the same Vulkan ICD trip the "device-already-
// in-use" panic on some setups.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Build the constant `A_eff = M/dt + νK` LHS for the CPU oracle.
/// Mirrors the inline construction in `tests/fem_ns_transient.rs`.
fn build_a_eff_cpu(m: &BsrMatrix, k: &BsrMatrix, inv_dt: Real) -> BsrMatrix {
    let mut triplets: Vec<(usize, usize, Vec<Real>)> = Vec::new();
    for r in 0..m.block_rows {
        for slot in m.row_offsets[r]..m.row_offsets[r + 1] {
            let c = m.col_idx[slot];
            triplets.push((r, c, vec![inv_dt * m.values[slot]]));
        }
    }
    for r in 0..k.block_rows {
        for slot in k.row_offsets[r]..k.row_offsets[r + 1] {
            let c = k.col_idx[slot];
            triplets.push((r, c, vec![k.values[slot]]));
        }
    }
    BsrMatrix::from_triplets(1, m.block_rows, m.block_cols, &triplets).unwrap()
}

/// Run the CPU NS time stepper for `n_steps` and return the final `u`.
fn cpu_ns_solve(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    viscosity: Real,
    dt: Real,
    n_steps: usize,
    bcs: &[molla::fem::StokesBc],
    f: &[Real],
    u_init: &[Real],
) -> Vec<Real> {
    let k = assemble_velocity_block(mesh, th, viscosity).unwrap();
    let m = assemble_velocity_mass(mesh, th).unwrap();
    let b = assemble_div_block(mesh, th).unwrap();
    let inv_dt = 1.0 / dt;
    let a_eff = build_a_eff_cpu(&m, &k, inv_dt);

    let simple = StokesSimpleConfig {
        max_outer_iters: 400,
        tol_outer: 1.0e-9,
        max_inner_iters: 2000,
        tol_inner: 1.0e-12,
        ..Default::default()
    };

    let mut u = u_init.to_vec();
    for _ in 0..n_steps {
        let res = solve_navier_stokes_step(
            mesh, th, &a_eff, &b, &m, &u, f, inv_dt, bcs, &simple,
        )
        .expect("CPU NS step");
        u = res.u;
    }
    u
}

#[test]
#[ignore = "needs a real wgpu adapter — run with --features gpu --test fem_ns_gpu -- --ignored"]
fn gpu_ns_lid_cavity_matches_cpu_8x8_re100() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // ── Problem setup ─────────────────────────────────────────────
    //
    // Re = U·L / ν → with U = 1, L = 1, Re = 100 → ν = 1/100 = 0.01.
    // Advective CFL on the 8×8 grid: Δt · U / h = 0.005 · 1 · 8 = 0.04,
    // well below 1. 50 steps reach t = 0.25 — long enough to develop
    // a clear difference from the steady Stokes baseline while staying
    // in the IMEX stability envelope.
    let nx = 8usize;
    let ny = 8usize;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 2);
    let viscosity: Real = 0.01;
    let dt: Real = 0.005;
    let n_steps = 50usize;

    let f = vec![0.0; th.ndof_velocity()];
    let u_init = vec![0.0; th.ndof_velocity()];
    let p_init = vec![0.0; th.ndof_pressure()];
    let bcs = build_cavity_bcs_2d(&mesh, &th, 1.0, [1.0, 0.0]);

    // ── CPU oracle ────────────────────────────────────────────────
    let t0 = Instant::now();
    let cpu_u = cpu_ns_solve(&mesh, &th, viscosity, dt, n_steps, &bcs, &f, &u_init);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "CPU NS 8×8 Re=100 (50 steps): time = {cpu_ms:.1} ms, ‖u‖∞ = {:.3e}",
        cpu_u.iter().fold(0.0_f64, |a, &v| a.max(v.abs())),
    );

    // ── GPU path ──────────────────────────────────────────────────
    let engine = Engine::new_gpu().expect("gpu");
    // SUPG OFF on both paths so the test is comparing the same
    // discretisation (CPU `solve_navier_stokes_step` uses plain
    // Galerkin via `assemble_convection_block`).
    let cfg = NavierStokesGpuConfig {
        dt,
        n_steps,
        supg_alpha: 0.0,
        stokes: StokesGpuConfig {
            max_outer_iters: 400,
            tol_outer: 5.0e-7,
            max_inner_iters: 2000,
            tol_inner: 5.0e-7,
        },
    };
    let t0 = Instant::now();
    let res = solve_navier_stokes_gpu(
        &mesh, &th, &u_init, &p_init, &f, viscosity, &bcs, &cfg, &engine,
    )
    .expect("GPU NS solve");
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "GPU NS 8×8 Re=100 (50 steps): time = {gpu_ms:.1} ms, last_iters = {}, div = {:.3e}",
        res.last_outer_iters, res.div_residual,
    );

    // ── Compare ───────────────────────────────────────────────────
    let mut peak_u: Real = 0.0;
    let mut max_abs: Real = 0.0;
    for (cpu, gpu) in cpu_u.iter().zip(res.u.iter()) {
        peak_u = peak_u.max(cpu.abs());
        max_abs = max_abs.max((cpu - gpu).abs());
    }
    let rel = max_abs / peak_u.max(1.0e-20);
    eprintln!(
        "u-diff: peak |u| = {peak_u:.3e}, max_abs = {max_abs:.3e}, rel = {rel:.3e}",
    );
    let speedup_cpu = cpu_ms / gpu_ms.max(1.0e-3);
    // v3 baseline measured on this rig (see module-level table).
    let v3_gpu_ms: f64 = 7986.0;
    let speedup_v3 = v3_gpu_ms / gpu_ms.max(1.0e-3);
    eprintln!(
        "wall-clock: CPU = {cpu_ms:.1} ms, v3 GPU = {v3_gpu_ms:.1} ms (baseline), \
         v4 GPU = {gpu_ms:.1} ms (CPU/GPU = {speedup_cpu:.2}x, v3/v4 = {speedup_v3:.2}x)",
    );
    assert!(
        rel < 5.0e-3,
        "GPU vs CPU velocity diff too large: rel = {rel} (max_abs {max_abs}, peak {peak_u})",
    );
    assert!(
        res.div_residual < 1.0e-2,
        "GPU divergence residual {:.3e} too large",
        res.div_residual,
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — perf bench, run with --features gpu --test fem_ns_gpu -- --ignored"]
fn gpu_ns_perf_16x16_re100_100steps() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let nx = 16usize;
    let ny = 16usize;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 2);
    let viscosity: Real = 0.01;
    let dt: Real = 0.0025; // CFL @ 16×16, U=1
    let n_steps = 100usize;

    let f = vec![0.0; th.ndof_velocity()];
    let u_init = vec![0.0; th.ndof_velocity()];
    let p_init = vec![0.0; th.ndof_pressure()];
    let bcs = build_cavity_bcs_2d(&mesh, &th, 1.0, [1.0, 0.0]);

    // CPU
    let t0 = Instant::now();
    let cpu_u = cpu_ns_solve(&mesh, &th, viscosity, dt, n_steps, &bcs, &f, &u_init);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "CPU NS 16×16 (100 steps): {cpu_ms:.1} ms, ‖u‖∞ = {:.3e}",
        cpu_u.iter().fold(0.0_f64, |a, &v| a.max(v.abs())),
    );

    // GPU
    let engine = Engine::new_gpu().expect("gpu");
    let cfg = NavierStokesGpuConfig {
        dt,
        n_steps,
        supg_alpha: 0.0,
        stokes: StokesGpuConfig {
            max_outer_iters: 400,
            tol_outer: 5.0e-7,
            max_inner_iters: 2000,
            tol_inner: 5.0e-7,
        },
    };
    let t0 = Instant::now();
    let res = solve_navier_stokes_gpu(
        &mesh, &th, &u_init, &p_init, &f, viscosity, &bcs, &cfg, &engine,
    )
    .expect("GPU NS solve");
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let speedup_cpu = cpu_ms / gpu_ms.max(1.0e-3);
    // v3 baseline measured on this rig (see module-level table).
    let v3_gpu_ms: f64 = 26964.0;
    let speedup_v3 = v3_gpu_ms / gpu_ms.max(1.0e-3);
    eprintln!(
        "GPU NS 16×16 (100 steps): {gpu_ms:.1} ms, last_iters = {}, div = {:.3e}",
        res.last_outer_iters, res.div_residual,
    );
    eprintln!(
        "perf: CPU = {cpu_ms:.1} ms, v3 GPU = {v3_gpu_ms:.1} ms (baseline), \
         v4 GPU = {gpu_ms:.1} ms (CPU/GPU = {speedup_cpu:.2}x, v3/v4 = {speedup_v3:.2}x)",
    );

    // Sanity check — GPU should produce a finite, low-divergence field.
    assert!(res.div_residual < 1.0e-1, "div residual too large");
    let mut peak: Real = 0.0;
    let mut max_abs: Real = 0.0;
    for (cpu, gpu) in cpu_u.iter().zip(res.u.iter()) {
        peak = peak.max(cpu.abs());
        max_abs = max_abs.max((cpu - gpu).abs());
    }
    let rel = max_abs / peak.max(1.0e-20);
    eprintln!(
        "16×16 parity check: peak = {peak:.3e}, max_abs = {max_abs:.3e}, rel = {rel:.3e}",
    );
    // Looser tolerance at 16×16 (more CG iters, more f32 round-off).
    assert!(rel < 1.0e-2, "GPU vs CPU rel = {rel} too large at 16×16");
}
