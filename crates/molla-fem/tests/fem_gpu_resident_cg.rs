//! Stage 36.5 Slice A — GPU-resident CG parity test.
//!
//! Runs the same heat-equation problem (16³ tet cube, Dirichlet on
//! the top + bottom faces, insulated sides) through three paths and
//! checks they agree:
//!   1. CPU CG f64 reference.
//!   2. GPU CG with per-iter readback (the v1
//!      [`solve_steady_state_gpu`] path).
//!   3. GPU-resident CG (Stage 36.5 Slice A
//!      [`solve_steady_state_gpu_resident`]) — `x`, `r`, `p`, `Ap`
//!      stay on the GPU; only the dot-product partial-sum readback
//!      crosses the host boundary per iteration.
//!
//! Tolerance vs CPU CG is `1e-3` — same f32-GPU precision budget
//! the v1 `solve_steady_state_gpu` test uses.
//!
//! Marked `#[ignore]` because it needs a real wgpu adapter — run with
//! `make test-gpu P=molla-fem T=fem_gpu_resident_cg`.
//!
//! Wall-clock telemetry is emitted via `eprintln!` for the post-stage
//! summary (resident CG should beat per-iter readback by a healthy
//! margin at 16³).

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_fem::{
    BoundaryCondition, FemMesh, FunctionSpace, apply_dirichlet, assemble_global,
    compute_element_stiffness,
    gpu_data::FemGpuData,
    solver::{solve_steady_state_gpu, solve_steady_state_gpu_resident},
};
use molla_math::Real;
use molla_math::iterative::cg_solve;
use molla_math::sparse::BsrMatrix;

/// Serialises wgpu adapter lifetime across tests in this binary.
/// Same pattern as `tests/fem_gpu.rs`.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn assemble_stiffness(mesh: &FemMesh, alpha: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .expect("CPU stiffness assembly")
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-fem T=fem_gpu_resident_cg"]
fn gpu_resident_cg_matches_cpu_and_streamed_paths() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 16³ tet cube — same recipe as the smoke test in `fem_gpu.rs`.
    // At this resolution the per-iter host bounce dominates the v1
    // GPU path, making the speedup from staying resident dramatic.
    let n_grid = 16usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let alpha: Real = 1.0;
    let k_full = assemble_stiffness(&mesh, alpha);
    let n = space.ndof(&mesh);
    let f_full = vec![0.0_f64; n];

    // BCs: bottom face T = 0, top face T = 1, sides insulated.
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.z.abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        } else if (p.z - 1.0).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 1.0,
            });
        }
    }
    let mut k_red = k_full.clone();
    let mut f_red = f_full.clone();
    apply_dirichlet(&mut k_red, &mut f_red, &bcs);

    // ── 1. CPU CG reference ────────────────────────────────────────
    let mut x_init = vec![0.0_f64; n];
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = *bc {
            x_init[vertex as usize] = value;
        }
    }
    let t_cpu0 = std::time::Instant::now();
    let cpu_cg = cg_solve(
        |x| {
            let mut y = vec![0.0_f64; n];
            k_red.mv(x, &mut y).unwrap();
            y
        },
        &f_red,
        &x_init,
        1e-9,
        5000,
    );
    let cpu_ms = t_cpu0.elapsed().as_secs_f64() * 1000.0;

    // ── 2. GPU streamed CG (v1 — per-iter readback) ────────────────
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_red).expect("upload");

    let t_streamed0 = std::time::Instant::now();
    let streamed = solve_steady_state_gpu(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
        .expect("gpu streamed CG");
    let streamed_ms = t_streamed0.elapsed().as_secs_f64() * 1000.0;

    // ── 3. GPU-resident CG (Slice A) ───────────────────────────────
    let t_resident0 = std::time::Instant::now();
    let resident = solve_steady_state_gpu_resident(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
        .expect("gpu resident CG");
    let resident_ms = t_resident0.elapsed().as_secs_f64() * 1000.0;

    // ── Parity vs CPU CG ───────────────────────────────────────────
    let mut max_err_streamed: Real = 0.0;
    let mut max_err_resident: Real = 0.0;
    let mut max_err_streamed_vs_resident: Real = 0.0;
    for i in 0..n {
        max_err_streamed = max_err_streamed.max((cpu_cg.x[i] - streamed.x[i]).abs());
        max_err_resident = max_err_resident.max((cpu_cg.x[i] - resident.x[i]).abs());
        max_err_streamed_vs_resident =
            max_err_streamed_vs_resident.max((streamed.x[i] - resident.x[i]).abs());
    }

    eprintln!(
        "fem_gpu_resident_cg: n = {n}\n  \
         CPU CG:        iters = {:>4}, wall = {cpu_ms:>7.1} ms (residual = {:.2e})\n  \
         GPU streamed:  iters = {:>4}, wall = {streamed_ms:>7.1} ms (residual = {:.2e}, max |Δ vs CPU| = {max_err_streamed:.2e})\n  \
         GPU resident:  iters = {:>4}, wall = {resident_ms:>7.1} ms (residual = {:.2e}, max |Δ vs CPU| = {max_err_resident:.2e})\n  \
         streamed vs resident max |Δ| = {max_err_streamed_vs_resident:.2e}",
        cpu_cg.iters,
        cpu_cg.residual,
        streamed.iters,
        streamed.residual,
        resident.iters,
        resident.residual,
    );

    // GPU-resident CG must agree with the CPU oracle within the f32
    // GPU precision budget — same threshold the v1 streamed test uses.
    assert!(
        max_err_resident < 1e-3,
        "GPU-resident CG diverges from CPU CG: max_err = {max_err_resident}",
    );
    // The two GPU paths should agree even more tightly with each
    // other — they share the same matvec kernel and f32 round-off.
    assert!(
        max_err_streamed_vs_resident < 1e-3,
        "GPU streamed and GPU-resident CG disagree: max_err = {max_err_streamed_vs_resident}",
    );

    // Sanity: resident CG result tracks the analytic T(z) = z within
    // the discretisation budget for the 16³ mesh.
    let mut max_analytic_err: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        max_analytic_err = max_analytic_err.max((resident.x[i] - p.z).abs());
    }
    assert!(
        max_analytic_err < 5e-2,
        "GPU-resident steady-state diverges from T(z) = z: max_err = {max_analytic_err}",
    );
}

/// Smaller 8³ smoke test — same parity check as above but cheaper to
/// run. Good for CI / first-pass validation when iterating on the
/// CG kernel.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-fem T=fem_gpu_resident_cg"]
fn gpu_resident_cg_matches_cpu_8_cube() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let n_grid = 8usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let k_full = assemble_stiffness(&mesh, 1.0);
    let n = space.ndof(&mesh);
    let f_full = vec![0.0_f64; n];

    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.z.abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 0.0,
            });
        } else if (p.z - 1.0).abs() < 1e-9 {
            bcs.push(BoundaryCondition::Dirichlet {
                vertex: i as u32,
                value: 1.0,
            });
        }
    }
    let mut k_red = k_full.clone();
    let mut f_red = f_full.clone();
    apply_dirichlet(&mut k_red, &mut f_red, &bcs);

    let mut x_init = vec![0.0_f64; n];
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = *bc {
            x_init[vertex as usize] = value;
        }
    }
    let cpu_cg = cg_solve(
        |x| {
            let mut y = vec![0.0_f64; n];
            k_red.mv(x, &mut y).unwrap();
            y
        },
        &f_red,
        &x_init,
        1e-9,
        5000,
    );

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_red).expect("upload");

    let resident = solve_steady_state_gpu_resident(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-6)
        .expect("gpu resident CG");

    let mut max_err: Real = 0.0;
    for i in 0..n {
        max_err = max_err.max((cpu_cg.x[i] - resident.x[i]).abs());
    }
    eprintln!(
        "fem_gpu_resident_cg [8³]: n = {n}, CPU iters = {}, GPU iters = {}, residual = {:.2e}, max |Δ vs CPU| = {max_err:.2e}",
        cpu_cg.iters, resident.iters, resident.residual,
    );
    assert!(
        max_err < 1e-3,
        "8³ GPU-resident CG diverges from CPU CG: max_err = {max_err}",
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — long-running benchmark"]
fn fem_gpu_resident_cg_scaling_sweep() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use std::time::Instant;

    eprintln!();
    eprintln!("─── Stage 36.5 GPU CG scaling sweep ──────────────────────");
    eprintln!("Heat-equation steady state on tet cube. CPU = f64 CG;");
    eprintln!("GPU streamed = per-iter readback; GPU resident = bind-");
    eprintln!("group-cached resident CG (Slice D).");
    eprintln!();
    eprintln!(
        "  {:>5}  {:>9}  {:>11}  {:>14}  {:>14}  {:>14}",
        "grid", "n_dofs", "n_iters_cpu", "CPU CG (ms)", "GPU stream (ms)", "GPU resid (ms)"
    );
    eprintln!("  ─────  ─────────  ───────────  ──────────────  ──────────────  ──────────────");

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    for &nx in &[8usize, 16, 24, 32, 48] {
        let mesh = FemMesh::tet_grid_3d(nx, nx, nx, 1.0, 1.0, 1.0);
        let n = mesh.vert_count();
        let k = assemble_stiffness(&mesh, 1.0);
        // Source proportional to z, Dirichlet on bottom face.
        let f: Vec<Real> = mesh.vertices.iter().map(|p| p.z).collect();
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.z <= 1e-9 {
                bcs.push(BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 0.0,
                });
            }
        }
        let mut k_red = k.clone();
        let mut f_red = f.clone();
        apply_dirichlet(&mut k_red, &mut f_red, &bcs);

        // Warm + measure CPU CG.
        let cpu_result = cg_solve(
            |x| {
                let mut y = vec![0.0_f64; n];
                k_red.mv(x, &mut y).unwrap();
                y
            },
            &f_red,
            &vec![0.0_f64; n],
            1e-8,
            10_000,
        );
        let cpu_iters = cpu_result.iters;
        let cpu_ms = {
            let t = Instant::now();
            let _ = cg_solve(
                |x| {
                    let mut y = vec![0.0_f64; n];
                    k_red.mv(x, &mut y).unwrap();
                    y
                },
                &f_red,
                &vec![0.0_f64; n],
                1e-8,
                10_000,
            );
            t.elapsed().as_secs_f64() * 1e3
        };

        // Upload to GPU once (shared across both GPU runs).
        let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_red).expect("upload");

        // Warm + measure GPU streamed.
        let _ = solve_steady_state_gpu(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-6)
            .expect("warmup streamed");
        let stream_ms = {
            let t = Instant::now();
            let _ = solve_steady_state_gpu(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-6)
                .expect("gpu streamed");
            t.elapsed().as_secs_f64() * 1e3
        };

        // Warm + measure GPU resident.
        let _ = solve_steady_state_gpu_resident(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-6)
            .expect("warmup resident");
        let resid_ms = {
            let t = Instant::now();
            let _ = solve_steady_state_gpu_resident(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-6)
                .expect("gpu resident");
            t.elapsed().as_secs_f64() * 1e3
        };

        eprintln!(
            "  {nx:>3}³  {n:>9}  {cpu_iters:>11}  {cpu_ms:>11.1}    {stream_ms:>11.1}    {resid_ms:>11.1}",
        );
    }
    eprintln!();
}
