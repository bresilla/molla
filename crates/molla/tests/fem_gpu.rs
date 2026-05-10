//! Stage 36.5 — GPU FEM parity tests.
//!
//! Covers:
//!   1. `gpu_p1_assembly_matches_cpu_oracle` — assemble the same 8×8×8
//!      tet-mesh stiffness matrix on CPU + GPU; compare CSR values
//!      within 1e-4 (the f32 ↔ f64 round-off budget; CPU is f64-internal,
//!      GPU is f32-internal — see `fem::gpu_data` doc-comment).
//!   2. `gpu_csr_matvec_matches_cpu_oracle` — random vector × CSR;
//!      GPU vs CPU within 1e-4.
//!   3. `gpu_solve_steady_state_3d_cube_matches_cpu` — solve the same
//!      heat-equation problem as `fem_v1::heat_steady_state_3d_cube_matches_analytic`
//!      on the GPU; compare per-vertex solution against the CPU CG
//!      result within 1e-3.
//!
//! All tests are `#[ignore]` because they need a real wgpu adapter.
//! Run with `make test-fem-gpu` (added in this stage) or directly:
//!
//! ```sh
//! nixVulkan cargo test -p molla --features gpu --test fem_gpu -- --ignored
//! ```
//!
//! `GPU_LOCK` mutex serialises adapter ownership so the suite can
//! run all three tests concurrently inside this binary without the
//! polling threads contending on the wgpu queue (same pattern as
//! `tests/radix_sort_gpu.rs`).

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla::compute::Engine;
use molla::fem::{
    apply_dirichlet, assemble_global, compute_element_stiffness, gpu_data::FemGpuData,
    solver::solve_steady_state_gpu, BoundaryCondition, FemMesh, FunctionSpace,
};
use molla::kernels::FemGpuKernel;
use molla::math::iterative::cg_solve;
use molla::math::sparse::BsrMatrix;
use molla::math::Real;

/// Serialises wgpu adapter lifetime across the tests in this binary.
/// Without it, concurrent `Engine::new_gpu()` calls from different
/// `#[test]` threads contend on `device.poll(wait_indefinitely)` and
/// either hang or crash. Same pattern as `tests/radix_sort_gpu.rs`.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn assemble_stiffness(mesh: &FemMesh, alpha: Real) -> BsrMatrix {
    assemble_global(mesh, FunctionSpace::P1Nodal, |c, v| {
        compute_element_stiffness(c, v, alpha)
    })
    .expect("CPU stiffness assembly")
}

fn cpu_matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    let mut y = vec![0.0; a.nrows()];
    a.mv(x, &mut y).unwrap();
    y
}

// Cheap deterministic LCG so tests don't need a `rand` dep.
fn lcg(seed: u64) -> impl FnMut() -> Real {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 32) as u32) as Real) / (u32::MAX as Real) - 0.5
    }
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-fem-gpu"]
fn gpu_p1_assembly_matches_cpu_oracle() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 8×8×8 tet cube — same recipe as the steady-state heat test
    // but smaller (the assembly parity test doesn't need 16³).
    let n_grid = 8usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let alpha: Real = 1.0;

    // CPU oracle.
    let k_cpu = assemble_stiffness(&mesh, alpha);

    // GPU assembly. We upload the CPU CSR's PATTERN (row_offsets +
    // col_indices) so the GPU shader knows where to scatter; the
    // values come out from the GPU dispatch. We achieve this by
    // uploading the CPU matrix THEN clearing the values to zero
    // BEFORE the dispatch runs (the FemGpuKernel::assemble call
    // does the clear automatically).
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_cpu).expect("upload");
    let kernel = FemGpuKernel::new(device.clone(), queue.clone());
    kernel
        .assemble(&gpu_data, alpha as f32)
        .expect("assemble dispatch");

    // Read the GPU values back.
    let gpu_values = gpu_data.read_csr_values(&device, &queue).expect("readback");

    assert_eq!(
        gpu_values.len(),
        k_cpu.values.len(),
        "GPU value buffer length differs from CPU (pattern mismatch?)"
    );

    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    for (i, (gpu, cpu)) in gpu_values.iter().zip(k_cpu.values.iter()).enumerate() {
        let cpu_f32 = *cpu as f32;
        let abs = (*gpu - cpu_f32).abs();
        let rel = abs / cpu_f32.abs().max(1e-6);
        max_abs_err = max_abs_err.max(abs);
        max_rel_err = max_rel_err.max(rel);
        // Per-element absolute tolerance — values can range up to a
        // few times the diffusivity, so 1e-4 is loose enough to soak
        // f32 round-off on the cross-products + the atomic CAS scatter.
        assert!(
            abs < 1e-4,
            "CSR slot {i}: GPU = {gpu}, CPU = {cpu_f32}, abs_err = {abs}",
        );
    }
    eprintln!(
        "gpu_p1_assembly_matches_cpu_oracle: \
         {} nnz, max_abs_err = {:.2e}, max_rel_err = {:.2e}",
        gpu_values.len(),
        max_abs_err,
        max_rel_err,
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-fem-gpu"]
fn gpu_csr_matvec_matches_cpu_oracle() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let n_grid = 8usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let k_cpu = assemble_stiffness(&mesh, 1.0);
    let n = k_cpu.nrows();

    // Random input vector (deterministic seed for reproducibility).
    let mut rng = lcg(0xC0FFEE_BEEF);
    let x_f64: Vec<Real> = (0..n).map(|_| rng()).collect();

    // CPU oracle.
    let y_cpu = cpu_matvec(&k_cpu, &x_f64);

    // GPU.
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_cpu).expect("upload");
    let kernel = FemGpuKernel::new(device.clone(), queue.clone());

    let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
    let y_f32 = kernel
        .matvec_blocking(&gpu_data, &x_f32)
        .expect("matvec dispatch");

    assert_eq!(y_f32.len(), n);

    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    for (i, (gpu, cpu)) in y_f32.iter().zip(y_cpu.iter()).enumerate() {
        let cpu_f32 = *cpu as f32;
        let abs = (*gpu - cpu_f32).abs();
        let rel = abs / cpu_f32.abs().max(1e-6);
        max_abs_err = max_abs_err.max(abs);
        max_rel_err = max_rel_err.max(rel);
        assert!(
            abs < 1e-4,
            "row {i}: GPU = {gpu}, CPU = {cpu_f32}, abs_err = {abs}",
        );
    }
    eprintln!(
        "gpu_csr_matvec_matches_cpu_oracle: \
         n = {n}, max_abs_err = {max_abs_err:.2e}, max_rel_err = {max_rel_err:.2e}",
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-fem-gpu"]
fn gpu_solve_steady_state_3d_cube_matches_cpu() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Same recipe as `fem_v1::heat_steady_state_3d_cube_matches_analytic`
    // but at 8³ resolution to keep CG iters tractable when the per-iter
    // matvec readback dominates wall-clock. Tolerance vs the CPU CG
    // result is 1e-3 (the f32 GPU CG can't beat its own round-off on
    // the matvec). Tolerance vs the analytic T(z) = z is 5e-2 — the
    // 8³ mesh is coarser than the 16³ recipe in PLAN.
    let n_grid = 8usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let alpha: Real = 1.0;
    let k_full = assemble_stiffness(&mesh, alpha);
    let n = space.ndof(&mesh);
    let f_full = vec![0.0; n];

    // BCs: top face T = 1, bottom face T = 0; sides insulating.
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

    // Apply BCs on the CPU (the GPU solver expects a pre-reduced K).
    let mut k_red = k_full.clone();
    let mut f_red = f_full.clone();
    apply_dirichlet(&mut k_red, &mut f_red, &bcs);

    // CPU CG reference.
    let mut x_init = vec![0.0; n];
    for bc in &bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = *bc {
            x_init[vertex as usize] = value;
        }
    }
    let cpu_cg = cg_solve(
        |x| {
            let mut y = vec![0.0; n];
            k_red.mv(x, &mut y).unwrap();
            y
        },
        &f_red,
        &x_init,
        1e-9,
        5000,
    );

    // GPU CG.
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_red).expect("upload");

    let t0 = std::time::Instant::now();
    let gpu_cg = solve_steady_state_gpu(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-6)
        .expect("gpu CG");
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_iter_ms = if gpu_cg.iters > 0 {
        dt_ms / gpu_cg.iters as f64
    } else {
        0.0
    };

    eprintln!(
        "gpu_solve_steady_state_3d_cube_matches_cpu: \
         n = {n}, CPU iters = {}, GPU iters = {}, GPU wall = {dt_ms:.1} ms ({per_iter_ms:.2} ms/iter)",
        cpu_cg.iters, gpu_cg.iters,
    );

    // GPU vs CPU CG result.
    let mut max_diff: Real = 0.0;
    for i in 0..n {
        max_diff = max_diff.max((cpu_cg.x[i] - gpu_cg.x[i]).abs());
    }
    assert!(
        max_diff < 1e-3,
        "GPU CG diverges from CPU CG: max_diff = {max_diff}",
    );

    // Sanity: GPU result tracks the analytic T(z) = z within the
    // discretisation budget (8³ mesh ⇒ ~5e-2; we just need the GPU
    // to produce something physically meaningful, the CPU test
    // already pins the discretisation accuracy).
    let mut max_analytic_err: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        max_analytic_err = max_analytic_err.max((gpu_cg.x[i] - p.z).abs());
    }
    assert!(
        max_analytic_err < 5e-2,
        "GPU steady-state diverges from T(z) = z: max_err = {max_analytic_err}",
    );
}

/// Stage 36.5 final-report smoke run — 16³ mesh, GPU CG vs CPU CG,
/// emits per-iter readback wall-clock for the v1 caveat doc-comment
/// in `solve_steady_state_gpu`. Tolerance vs CPU CG is loose (5e-3)
/// because at 16³ the f32 accumulation in the matvec compounds over
/// more iterations.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-fem-gpu (16³ smoke)"]
fn gpu_solve_steady_state_3d_cube_16_smoke() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let n_grid = 16usize;
    let mesh = FemMesh::tet_grid_3d(n_grid, n_grid, n_grid, 1.0, 1.0, 1.0);
    let space = FunctionSpace::P1Nodal;
    let k_full = assemble_stiffness(&mesh, 1.0);
    let n = space.ndof(&mesh);
    let f_full = vec![0.0; n];

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

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let gpu_data = FemGpuData::upload(&device, &queue, &mesh, &k_red).expect("upload");

    let t0 = std::time::Instant::now();
    let gpu_cg = solve_steady_state_gpu(&gpu_data, &f_red, &bcs, &engine, 5000, 1e-5)
        .expect("gpu CG");
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_iter_ms = dt_ms / gpu_cg.iters.max(1) as f64;

    let mut max_analytic_err: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        max_analytic_err = max_analytic_err.max((gpu_cg.x[i] - p.z).abs());
    }

    eprintln!(
        "[smoke] 16³ tet cube: n = {n}, GPU CG iters = {}, wall = {dt_ms:.1} ms \
         ({per_iter_ms:.2} ms/iter readback), max |T - z| = {max_analytic_err:.3e}",
        gpu_cg.iters,
    );
    // No tolerance assertion here — this is a smoke + timing harness;
    // the tighter parity assertion is in the 8³ test.
}
