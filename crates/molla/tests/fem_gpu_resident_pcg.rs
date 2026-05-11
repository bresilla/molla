//! Stage 36.5+ — Jacobi-preconditioned GPU-resident CG parity + perf.
//!
//! Three `#[ignore]` tests that exercise the
//! [`solve_steady_state_gpu_resident_pcg`] entry against the
//! unpreconditioned [`solve_steady_state_gpu_resident`] sibling:
//!   1. **Parity** — 16³ heat eq, both solvers must agree on the
//!      solution within `1e-3` and PCG must take FEWER iterations
//!      (the whole point of preconditioning).
//!   2. **Diagonal extraction parity** — the GPU
//!      [`FemGpuKernel::extract_diagonal`] must agree with the CPU
//!      `K[i, i]` for every row.
//!   3. **Wall-time comparison** — same 32³ heat eq through both
//!      paths; informational `eprintln!`. Asserts CG and PCG agree
//!      on the solution + PCG iters ≤ CG iters.
//!
//! Marked `#[ignore]` because they need a real wgpu adapter — run with
//! `nixVulkan cargo test -p molla --features gpu --test fem_gpu_resident_pcg -- --ignored`.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla::compute::Engine;
use molla::fem::{
    apply_dirichlet, assemble_global, compute_element_stiffness, gpu_data::FemGpuData,
    solver::{solve_steady_state_gpu_resident, solve_steady_state_gpu_resident_pcg},
    BoundaryCondition, FemMesh, FunctionSpace,
};
use molla::kernels::FemGpuKernel;
use molla::math::sparse::BsrMatrix;
use molla::math::Real;

/// Serialises wgpu adapter lifetime across tests in this binary.
/// Same pattern as `tests/fem_gpu_resident_cg.rs`.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn assemble_stiffness(mesh: &FemMesh, alpha: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .expect("CPU stiffness assembly")
}

/// Build the standard heat-equation problem on an `nx³` tet cube:
/// Dirichlet `T = 0` on z=0, `T = 1` on z=1, insulated sides. Returns
/// `(mesh, bcs, k_red, f_red)` where `k_red` and `f_red` are already
/// BC-reduced (ready for upload to the GPU CSR).
fn build_heat_eq(nx: usize) -> (FemMesh, Vec<BoundaryCondition>, BsrMatrix, Vec<Real>) {
    let mesh = FemMesh::tet_grid_3d(nx, nx, nx, 1.0, 1.0, 1.0);
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
    (mesh, bcs, k_red, f_red)
}

#[test]
#[ignore = "needs a real wgpu adapter — run via cargo test --features gpu --test fem_gpu_resident_pcg -- --ignored"]
fn pcg_matches_cg_and_takes_fewer_iters_16_cube() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let nx = 16usize;
    let (mesh, bcs, k_red, f_red) = build_heat_eq(nx);
    let n = f_red.len();

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_red).expect("upload");

    let cg = solve_steady_state_gpu_resident(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
        .expect("gpu resident CG");
    let pcg = solve_steady_state_gpu_resident_pcg(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
        .expect("gpu resident PCG");

    let mut max_err: Real = 0.0;
    for i in 0..n {
        max_err = max_err.max((cg.x[i] - pcg.x[i]).abs());
    }
    eprintln!(
        "fem_gpu_resident_pcg [16³]: n = {n}\n  \
         CG  iters = {:>4}, residual = {:.2e}\n  \
         PCG iters = {:>4}, residual = {:.2e}\n  \
         max |Δ CG vs PCG| = {max_err:.2e}",
        cg.iters, cg.residual, pcg.iters, pcg.residual,
    );

    // Solutions must agree within the f32 GPU precision budget.
    assert!(
        max_err < 1e-3,
        "PCG diverges from CG: max_err = {max_err}",
    );
    // PCG must take fewer iterations than CG — the whole point of
    // preconditioning. Jacobi on a Laplacian-like system should cut
    // iter count by 25-40% for free.
    assert!(
        pcg.iters < cg.iters,
        "PCG took {} iters vs CG {}; expected strict reduction (Jacobi should help)",
        pcg.iters,
        cg.iters,
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via cargo test --features gpu --test fem_gpu_resident_pcg -- --ignored"]
fn extract_diagonal_matches_cpu() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let nx = 8usize;
    let (mesh, _bcs, k_red, _f_red) = build_heat_eq(nx);
    let n = k_red.nrows();

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_red).expect("upload");

    // CPU reference — walk the BSR rows and pull out `K[i, i]`.
    let mut cpu_diag = vec![0.0_f32; n];
    for r in 0..n {
        for slot in k_red.row_offsets[r]..k_red.row_offsets[r + 1] {
            if k_red.col_idx[slot] == r {
                cpu_diag[r] = k_red.values[slot] as f32;
                break;
            }
        }
    }

    // GPU extraction.
    let mv_kernel = FemGpuKernel::new(device.clone(), queue.clone());
    let bytes = (n.max(4) * 4) as u64;
    let diag_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test::diag_buf"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    mv_kernel
        .extract_diagonal(&gpu_data, &diag_buf)
        .expect("extract_diagonal");

    // Read back via FemCgGpuKernel::read_buffer (the only reader we
    // need; mv_kernel doesn't expose one). Use a fresh CG kernel.
    let cg_kernel = molla::kernels::FemCgGpuKernel::new(device.clone(), queue.clone());
    let gpu_diag = cg_kernel
        .read_buffer(&diag_buf, n as u32)
        .expect("readback");

    let mut max_err: f32 = 0.0;
    for i in 0..n {
        max_err = max_err.max((cpu_diag[i] - gpu_diag[i]).abs());
    }
    eprintln!(
        "extract_diagonal [8³]: n = {n}, max |GPU diag - CPU diag| = {max_err:.2e}",
    );
    // f32 round-off vs f64 CPU oracle — well under 1e-4 in practice.
    assert!(
        max_err < 1e-4,
        "extract_diagonal mismatches CPU K[i,i]: max_err = {max_err}",
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — long-running benchmark"]
fn pcg_vs_cg_wall_time_32_cube() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let nx = 32usize;
    let (mesh, bcs, k_red, f_red) = build_heat_eq(nx);
    let n = f_red.len();

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_red).expect("upload");

    // Warm both paths first — bind-group caches + shader compilation
    // should be fully amortised before timing.
    let _ = solve_steady_state_gpu_resident(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
        .expect("CG warmup");
    let _ = solve_steady_state_gpu_resident_pcg(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
        .expect("PCG warmup");

    let t_cg = std::time::Instant::now();
    let cg = solve_steady_state_gpu_resident(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
        .expect("gpu resident CG");
    let cg_ms = t_cg.elapsed().as_secs_f64() * 1000.0;

    let t_pcg = std::time::Instant::now();
    let pcg = solve_steady_state_gpu_resident_pcg(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
        .expect("gpu resident PCG");
    let pcg_ms = t_pcg.elapsed().as_secs_f64() * 1000.0;

    let mut max_err: Real = 0.0;
    for i in 0..n {
        max_err = max_err.max((cg.x[i] - pcg.x[i]).abs());
    }
    eprintln!(
        "fem_gpu_resident_pcg [32³] wall-time:\n  \
         n = {n}\n  \
         CG : iters = {:>4}, wall = {cg_ms:>7.1} ms (residual = {:.2e})\n  \
         PCG: iters = {:>4}, wall = {pcg_ms:>7.1} ms (residual = {:.2e})\n  \
         iter ratio (PCG/CG) = {:.2}, wall ratio (PCG/CG) = {:.2}\n  \
         max |Δ CG vs PCG| = {max_err:.2e}",
        cg.iters,
        cg.residual,
        pcg.iters,
        pcg.residual,
        pcg.iters as f64 / cg.iters as f64,
        pcg_ms / cg_ms,
    );

    // Sanity checks — solutions agree (within a slightly looser
    // tolerance than 16³ because both solves stop at the same relative
    // residual but the per-DoF errors compound at higher resolution),
    // PCG iter count is no worse than CG (informational on wall time;
    // if PCG is broken it'll show up as either a divergent solution
    // or a higher iter count).
    assert!(
        max_err < 5e-3,
        "PCG diverges from CG at 32³: max_err = {max_err}",
    );
    assert!(
        pcg.iters <= cg.iters,
        "PCG took {} iters vs CG {} at 32³ — preconditioning regressed",
        pcg.iters,
        cg.iters,
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — long-running benchmark"]
fn fem_gpu_resident_cg_vs_pcg_scaling_sweep() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use std::time::Instant;

    eprintln!();
    eprintln!("─── Stage 36.5+ GPU CG vs PCG scaling sweep ──────────────");
    eprintln!("Heat-equation steady state on tet cube. CG = unpreconditioned");
    eprintln!("resident; PCG = Jacobi-preconditioned resident.");
    eprintln!();
    eprintln!(
        "  {:>5}  {:>9}  {:>10}  {:>10}  {:>14}  {:>14}  {:>10}",
        "grid", "n_dofs", "CG iters", "PCG iters", "CG (ms)", "PCG (ms)", "iter cut"
    );
    eprintln!("  ─────  ─────────  ──────────  ──────────  ──────────────  ──────────────  ──────────");

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    for &nx in &[8usize, 16, 24, 32, 48] {
        let (mesh, bcs, k_red, f_red) = build_heat_eq(nx);
        let n = f_red.len();
        let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_red).expect("upload");

        // Warm both paths so first-call kernel + bind-group cost doesn't
        // skew the measurement.
        let _ = solve_steady_state_gpu_resident(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
            .expect("warmup CG");
        let _ = solve_steady_state_gpu_resident_pcg(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
            .expect("warmup PCG");

        let cg_start = Instant::now();
        let cg = solve_steady_state_gpu_resident(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
            .expect("gpu resident CG");
        let cg_ms = cg_start.elapsed().as_secs_f64() * 1e3;

        let pcg_start = Instant::now();
        let pcg = solve_steady_state_gpu_resident_pcg(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
            .expect("gpu resident PCG");
        let pcg_ms = pcg_start.elapsed().as_secs_f64() * 1e3;

        let iter_cut = if cg.iters > 0 {
            100.0 * (1.0 - pcg.iters as f64 / cg.iters as f64)
        } else {
            0.0
        };
        eprintln!(
            "  {nx:>3}³  {n:>9}  {:>10}  {:>10}  {cg_ms:>11.1}    {pcg_ms:>11.1}    {iter_cut:>8.1}%",
            cg.iters, pcg.iters
        );
    }
    eprintln!();
}
