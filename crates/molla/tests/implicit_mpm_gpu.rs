//! Stage 40 — GPU/CPU parity for the implicit-MPM inner colored
//! Gauss-Seidel sweep over the full 27-point Newton Hessian.
//!
//! The Stage 40 GPU port is intentionally a "lift the inner solver
//! only" port: the P2G/G2P scatter still runs on the CPU, and the
//! colored-GS sweep is the part of the timestep that runs on the
//! device. The host assembles the system (mass, 27-stencil Hessian,
//! predicted velocity), uploads to the device, invokes the GPU
//! sweep, reads back `v_grid`, and lets the CPU finish the G2P +
//! advect pass.
//!
//! This test runs the **identical** scenario twice: once with the
//! pure-CPU step, once with the GPU GS sweep replacing the inner
//! loop. Final particle velocities must agree to a relative
//! tolerance of `1e-5` (mostly limited by the f32 ↔ f64 conversion
//! at the buffer boundary).
//!
//! Run with:
//! ```text
//! nixVulkan cargo test -p molla --features gpu --test implicit_mpm_gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use molla::compute::implicit_mpm::{
    implicit_mpm_step, implicit_mpm_step_with_solver, GridSystem, HessianStencil,
    ImplicitMpmConfig, MpmState,
};
use molla::compute::Engine;
use molla::kernels::{
    implicit_mpm_readback_vec3_f32, implicit_mpm_upload_storage_f32,
    implicit_mpm_upload_storage_vec3_f32, ImplicitMpmKernel,
};
use molla::math::{Real, Vec3};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

/// Build a small spherical blob of particles centred at `c` with a
/// regular cell-aligned spacing. Same helper as in
/// `tests/implicit_mpm.rs`.
fn make_blob(center: Vec3, radius: Real, dx: Real) -> Vec<Vec3> {
    let mut pos = Vec::new();
    let n = (radius / dx).ceil() as i32;
    for iz in -n..=n {
        for iy in -n..=n {
            for ix in -n..=n {
                let off = Vec3::new(ix as Real * dx, iy as Real * dx, iz as Real * dx);
                if off.length() <= radius {
                    pos.push(center + off);
                }
            }
        }
    }
    pos
}

fn parity_cfg() -> ImplicitMpmConfig {
    ImplicitMpmConfig {
        grid_size: 12,
        cell_size: 0.06,
        grid_origin: [-0.36, -0.05, -0.36],
        dt: 1.0 / 120.0,
        gravity_y: -9.81,
        yield_stress: 500.0,
        viscosity: 20.0,
        density: 1000.0,
        particle_volume: 1.0e-4,
        floor_y: 0.0,
        floor_friction: 0.6,
        // Hard-cap iterations so the GPU and CPU paths walk the same
        // GS schedule — neither path early-terminates on residual.
        max_gs_iters: 8,
        gs_tol: 0.0,
        max_newton_iters: 1,
    }
}

/// Pack a `GridSystem` for GPU upload + run the GS sweep on device.
/// Returns the GS-iteration count and a synthetic residual; the
/// override side of `implicit_mpm_step_with_solver` only needs them
/// for `StepReport` book-keeping.
fn run_gpu_sweep(
    engine: &Engine,
    kernel: &ImplicitMpmKernel,
    system: &GridSystem,
    v_grid: &mut [Vec3],
    stencil: HessianStencil,
    max_iters: u32,
) -> (usize, Real) {
    let device = engine.device().expect("gpu engine has device");
    let queue = engine.queue().expect("gpu engine has queue");

    let g_total = v_grid.len();
    let g_mass_f32: Vec<f32> = system.g_mass.iter().map(|&x| x as f32).collect();
    let k_stencil_f32: Vec<f32> = system.k_stencil.iter().map(|&x| x as f32).collect();
    let v_pred_f32: Vec<[f32; 3]> = system
        .v_pred
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let v_grid_f32: Vec<[f32; 3]> = v_grid
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();

    let g_mass_buf =
        implicit_mpm_upload_storage_f32(device, queue, "stage40 g_mass", &g_mass_f32);
    let k_stencil_buf =
        implicit_mpm_upload_storage_f32(device, queue, "stage40 k_stencil", &k_stencil_f32);
    let v_pred_buf =
        implicit_mpm_upload_storage_vec3_f32(device, queue, "stage40 v_pred", &v_pred_f32);
    let v_grid_buf =
        implicit_mpm_upload_storage_vec3_f32(device, queue, "stage40 v_grid", &v_grid_f32);

    // Convert color groups: filter inactive cells (mass <= 0) so we
    // don't dispatch wasted threads, and convert to u32.
    let color_groups_u32: Vec<Vec<u32>> = system
        .color_groups
        .iter()
        .map(|g| {
            g.iter()
                .filter(|&&ci| system.active[ci])
                .map(|&ci| ci as u32)
                .collect()
        })
        .collect();

    let use_off_diag = matches!(stencil, HessianStencil::Full27);
    kernel
        .run_gs_sweeps(
            system.grid_size as u32,
            system.dt as f32,
            use_off_diag,
            max_iters,
            &color_groups_u32,
            &g_mass_buf,
            &k_stencil_buf,
            &v_pred_buf,
            &v_grid_buf,
        )
        .expect("GPU GS sweep");

    let v_back =
        implicit_mpm_readback_vec3_f32(device, queue, &v_grid_buf, g_total).expect("readback");
    for (i, v) in v_back.iter().enumerate() {
        v_grid[i] = Vec3::new(v[0], v[1], v[2]);
    }
    (max_iters as usize, 0.0)
}

#[test]
#[ignore]
fn implicit_mpm_gs_sweep_cpu_gpu_parity() {
    let engine = match try_engine() {
        Some(e) => e,
        None => {
            eprintln!("no GPU adapter — skipping");
            return;
        }
    };
    let kernel = {
        let device = engine.device().expect("device").clone();
        let queue = engine.queue().expect("queue").clone();
        ImplicitMpmKernel::new(device, queue)
    };

    let cfg = parity_cfg();
    // ≤ 200 particles per the brief; the 0.08-radius blob with
    // dx ≈ 0.042 m yields ~150 particles in the 12³ grid.
    let particles = make_blob(Vec3::new(0.0, 0.20, 0.0), 0.10, cfg.cell_size * 0.5);
    eprintln!("parity: n_particles = {}", particles.len());
    assert!(particles.len() <= 200, "test scenario must stay ≤ 200 particles");
    assert!(particles.len() >= 32);

    // ── CPU baseline. ────────────────────────────────────────────────
    let mut state_cpu = MpmState::new(&particles, &[]);
    // A couple of warm-up steps so velocities are non-trivial.
    for _ in 0..3 {
        implicit_mpm_step(&mut state_cpu, &cfg, &mut []);
    }
    let snapshot = state_cpu.clone();

    // One more CPU step.
    let report_cpu = implicit_mpm_step(&mut state_cpu, &cfg, &mut []);

    // ── GPU replay from the same snapshot, with the GPU GS sweep
    //    plugged into the inner loop. ─────────────────────────────────
    let mut state_gpu = snapshot.clone();
    let report_gpu = {
        let mut cb = |sys: &GridSystem, v_grid: &mut [Vec3], stencil: HessianStencil| {
            run_gpu_sweep(&engine, &kernel, sys, v_grid, stencil, cfg.max_gs_iters as u32)
        };
        implicit_mpm_step_with_solver(
            &mut state_gpu,
            &cfg,
            &mut [],
            HessianStencil::Full27,
            Some(&mut cb),
        )
    };

    eprintln!(
        "parity: CPU iters={}  GPU iters={}  CPU residual²={:.3e}  GPU residual²={:.3e}",
        report_cpu.gs_iters,
        report_gpu.gs_iters,
        report_cpu.true_residual_sq,
        report_gpu.true_residual_sq
    );

    // ── Compare final particle velocities. ───────────────────────────
    let mut max_rel: Real = 0.0;
    let mut max_abs: Real = 0.0;
    let mut max_speed: Real = 1.0e-12;
    for v in &state_cpu.vel {
        let s = v.length();
        if s > max_speed {
            max_speed = s;
        }
    }
    for (i, (vc, vg)) in state_cpu.vel.iter().zip(state_gpu.vel.iter()).enumerate() {
        let d = *vc - *vg;
        let ad = d.length();
        if ad > max_abs {
            max_abs = ad;
        }
        let denom = vc.length().max(1.0e-6);
        let rel = ad / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        if i < 3 {
            eprintln!(
                "  p{i}: cpu=({:.4}, {:.4}, {:.4})  gpu=({:.4}, {:.4}, {:.4})",
                vc.x, vc.y, vc.z, vg.x, vg.y, vg.z
            );
        }
    }
    let avg_speed = max_speed;
    eprintln!(
        "parity: max_abs_dv={:.3e}, max_rel_dv={:.3e}, peak |v|={:.3e}",
        max_abs, max_rel, avg_speed
    );

    // Tolerance is set to `1e-5` per the brief. f64→f32→f64 in the
    // GPU path produces ~1e-7 floor errors per element; the GS sweep
    // accumulates O(iterations · n_neighbours) of those — well below
    // 1e-5 for the 8-iter cap.
    assert!(
        max_rel < 1.0e-5 || max_abs < 1.0e-5 * avg_speed,
        "CPU/GPU velocity parity broken: max_rel={max_rel:.3e}, max_abs={max_abs:.3e}, peak |v|={avg_speed:.3e}"
    );
}
