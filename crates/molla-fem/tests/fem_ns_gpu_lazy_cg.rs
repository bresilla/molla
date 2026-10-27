//! Stage 41 v5 — lazy-CG sanity test.
//!
//! ```sh
//! nixVulkan cargo test -p molla-fem --features gpu --release \
//!     --test fem_ns_gpu_lazy_cg -- --ignored
//! ```
//!
//! The lazy CG (`cg_solve_resident_lazy`) checks residual only every
//! `check_period` iters, so it may run up to `check_period - 1` extra
//! iters past the true convergence point. This test pins that bound:
//! lazy-mode iter count is within `check_period` of strict-mode iter
//! count on the same problem.
//!
//! Uses the same A_eff matrix the NS solver assembles internally (M/dt
//! + νK on a 16×16 cavity grid), so the iter counts here are
//!   representative of what the NS Uzawa inner CG sees per step.

#![cfg(feature = "gpu")]

use std::sync::Arc;
use std::sync::Mutex;

use molla_compute::Engine;
use molla_fem::FemCgGpuKernel;
use molla_fem::{
    FemMesh, MixedTaylorHood, StokesBc, assemble_div_block, assemble_velocity_block,
    assemble_velocity_mass, build_cavity_bcs_2d, collect_velocity_pins_pub,
};
use molla_kernels::{BsrAlgebraKernel, BsrGpu};
use molla_math::Real;
use molla_math::sparse::BsrMatrix;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn build_a_eff(m: &BsrMatrix, k: &BsrMatrix, inv_dt: Real) -> BsrMatrix {
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for r in 0..m.block_rows {
        for slot in m.row_offsets[r]..m.row_offsets[r + 1] {
            let c = m.col_idx[slot];
            triplets.push((r, c, inv_dt * m.values[slot]));
        }
    }
    for r in 0..k.block_rows {
        for slot in k.row_offsets[r]..k.row_offsets[r + 1] {
            let c = k.col_idx[slot];
            triplets.push((r, c, k.values[slot]));
        }
    }
    BsrMatrix::from_scalar_triplets(m.block_rows, m.block_cols, &triplets).unwrap()
}

fn apply_symmetric_dirichlet(k: &mut BsrMatrix, f: &mut [Real], pinned: &[(usize, Real)]) {
    let pin_map: std::collections::HashMap<usize, Real> = pinned.iter().copied().collect();
    for r in 0..k.block_rows {
        if pin_map.contains_key(&r) {
            continue;
        }
        let lo = k.row_offsets[r];
        let hi = k.row_offsets[r + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            if let Some(v) = pin_map.get(&c) {
                f[r] -= k.values[slot] * v;
                k.values[slot] = 0.0;
            }
        }
    }
    for (r, v) in pinned {
        let lo = k.row_offsets[*r];
        let hi = k.row_offsets[*r + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            k.values[slot] = if c == *r { 1.0 } else { 0.0 };
        }
        f[*r] = *v;
    }
}

#[test]
#[ignore = "needs a real wgpu adapter — run with --features gpu --test fem_ns_gpu_lazy_cg -- --ignored"]
fn lazy_cg_convergence_count_within_k_of_strict() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 16×16 cavity, A_eff = M/dt + νK with Δt=0.0025, ν=0.01 (same
    // parameters as the perf test).
    let nx = 16usize;
    let ny = 16usize;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 2);
    let viscosity: Real = 0.01;
    let inv_dt: Real = 1.0 / 0.0025;
    let bcs: Vec<StokesBc> = build_cavity_bcs_2d(&mesh, &th, 1.0, [1.0, 0.0]);

    let k = assemble_velocity_block(&mesh, &th, viscosity).unwrap();
    let m = assemble_velocity_mass(&mesh, &th).unwrap();
    let _b = assemble_div_block(&mesh, &th).unwrap();
    let mut a_eff = build_a_eff(&m, &k, inv_dt);
    let pinned = collect_velocity_pins_pub(&bcs, th.dim);
    let n_v = th.ndof_velocity();
    let mut f = vec![0.0_f64; n_v];
    // Reproduce the NS RHS structure approximately: use a synthetic
    // body force that produces a non-trivial r₀ so the inner CG has
    // real work. The actual RHS in the NS step is messier — we just
    // need *a* CG problem that takes 20+ iters.
    for i in 0..n_v {
        f[i] = ((i % 13) as f64 - 6.0) * 0.05;
    }
    apply_symmetric_dirichlet(&mut a_eff, &mut f, &pinned);

    // GPU setup.
    let engine = Engine::new_gpu().expect("gpu engine");
    let device = engine.device().expect("gpu device").clone();
    let queue = engine.queue().expect("gpu queue").clone();
    let cg_kernel = FemCgGpuKernel::new(Arc::clone(&device), Arc::clone(&queue));
    let bsr_kernel = BsrAlgebraKernel::new(Arc::clone(&device), Arc::clone(&queue));
    let a_gpu = BsrGpu::upload(&device, &queue, &a_eff).expect("upload A_eff");

    let alloc_f32_buf = |label: &str, n: u32| -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n as u64).max(1) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let b_buf = alloc_f32_buf("lazy_cg_test::b", n_v as u32);
    let f_f32: Vec<f32> = f.iter().map(|&v| v as f32).collect();
    queue.write_buffer(&b_buf, 0, bytemuck::cast_slice(&f_f32));

    let tol: f32 = 5.0e-7;
    let max_iters = 2000usize;

    // Strict CG — single sync per iter.
    let strict = cg_kernel
        .cg_solve_resident(n_v as u32, &b_buf, None, tol, max_iters, |x, y| {
            bsr_kernel.matvec(&a_gpu, x, y, 1.0, 0.0)
        })
        .expect("strict CG");

    // Lazy CG — convergence-check every check_period iters.
    let check_period = 8usize;
    let lazy = cg_kernel
        .cg_solve_resident_lazy(
            n_v as u32,
            &b_buf,
            None,
            tol,
            max_iters,
            Some(check_period),
            |x, y| bsr_kernel.matvec(&a_gpu, x, y, 1.0, 0.0),
        )
        .expect("lazy CG");

    eprintln!(
        "lazy_cg_test: strict iters = {} residual = {:.3e}; \
         lazy iters = {} residual = {:.3e}; check_period = {}",
        strict.iters, strict.residual, lazy.iters, lazy.residual, check_period,
    );

    // The lazy path checks convergence only every `check_period`
    // iters, so it can overshoot the strict path by up to
    // `check_period - 1` iters. It can also exit "early" (relative to
    // strict) if the residual happens to cross tol mid-batch and the
    // batch boundary catches it later — but it can never finish
    // *before* the strict path, since both follow the same math.
    let diff = (lazy.iters as i64) - (strict.iters as i64);
    assert!(
        diff >= 0 && diff <= (check_period as i64),
        "lazy iters ({}) outside strict iters ({}) ± [0, check_period={}]",
        lazy.iters,
        strict.iters,
        check_period,
    );
    // Both should converge to roughly the same residual tolerance.
    // f32 round-off in the GPU dot reduction means we can't compare
    // bit-exact; allow a 2x slack on the tolerance.
    assert!(
        lazy.residual <= 2.0 * tol,
        "lazy residual {:.3e} far above tol {:.3e}",
        lazy.residual,
        tol,
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run with --features gpu --test fem_ns_gpu_lazy_cg -- --ignored"]
fn cg_fused_convergence_matches_strict() {
    // Stage 41 v6 — single-pass fused CG iter (3 dispatches/iter vs
    // v5's 7) must converge to the same tolerance and finish within
    // `check_period` iters of the strict path.
    //
    // The fused path uses f32 atomic-CAS-add for the dot reductions
    // (workgroup-local tree reduce → one atomic per WG). The atomic
    // add order is nondeterministic → the residual may differ from
    // strict CG at the 1 ULP level per iter, accumulating to ~1e-6
    // over ~30 iters at tol = 5e-7. We allow 2× slack on the
    // tolerance in the residual gate.
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let nx = 16usize;
    let ny = 16usize;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 2);
    let viscosity: Real = 0.01;
    let inv_dt: Real = 1.0 / 0.0025;
    let bcs: Vec<StokesBc> = build_cavity_bcs_2d(&mesh, &th, 1.0, [1.0, 0.0]);

    let k = assemble_velocity_block(&mesh, &th, viscosity).unwrap();
    let m = assemble_velocity_mass(&mesh, &th).unwrap();
    let _b = assemble_div_block(&mesh, &th).unwrap();
    let mut a_eff = build_a_eff(&m, &k, inv_dt);
    let pinned = collect_velocity_pins_pub(&bcs, th.dim);
    let n_v = th.ndof_velocity();
    let mut f = vec![0.0_f64; n_v];
    for i in 0..n_v {
        f[i] = ((i % 13) as f64 - 6.0) * 0.05;
    }
    apply_symmetric_dirichlet(&mut a_eff, &mut f, &pinned);

    let engine = Engine::new_gpu().expect("gpu engine");
    let device = engine.device().expect("gpu device").clone();
    let queue = engine.queue().expect("gpu queue").clone();
    let cg_kernel = FemCgGpuKernel::new(Arc::clone(&device), Arc::clone(&queue));
    let bsr_kernel = BsrAlgebraKernel::new(Arc::clone(&device), Arc::clone(&queue));
    let a_gpu = BsrGpu::upload(&device, &queue, &a_eff).expect("upload A_eff");

    let alloc_f32_buf = |label: &str, n: u32| -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n as u64).max(1) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let b_buf = alloc_f32_buf("fused_cg_test::b", n_v as u32);
    let f_f32: Vec<f32> = f.iter().map(|&v| v as f32).collect();
    queue.write_buffer(&b_buf, 0, bytemuck::cast_slice(&f_f32));

    let tol: f32 = 5.0e-7;
    let max_iters = 2000usize;
    let check_period = 8usize;

    let strict = cg_kernel
        .cg_solve_resident(n_v as u32, &b_buf, None, tol, max_iters, |x, y| {
            bsr_kernel.matvec(&a_gpu, x, y, 1.0, 0.0)
        })
        .expect("strict CG");

    let fused = cg_kernel
        .cg_solve_resident_fused(
            n_v as u32,
            &a_gpu,
            &b_buf,
            None,
            tol,
            max_iters,
            Some(check_period),
        )
        .expect("fused CG");

    eprintln!(
        "fused_cg_test: strict iters = {} residual = {:.3e}; \
         fused iters = {} residual = {:.3e}; check_period = {}",
        strict.iters, strict.residual, fused.iters, fused.residual, check_period,
    );

    // Like the lazy CG path, the fused CG reads back the residual only
    // every `check_period` iters and may over-run the strict path by
    // up to `check_period - 1` iters. f32 atomic-CAS sum nondeterminism
    // can also cause it to flip across a tol boundary at the batch
    // boundary, so the brief allows "within 1 of strict (no over-run
    // beyond check_period)" semantics — same upper bound as the lazy
    // CG test above.
    let diff = (fused.iters as i64) - (strict.iters as i64);
    assert!(
        diff >= 0 && diff <= (check_period as i64),
        "fused iters ({}) outside strict iters ({}) ± [0, check_period={}]",
        fused.iters,
        strict.iters,
        check_period,
    );
    // Both should converge under 2× tol — atomic-CAS sum order
    // nondeterminism adds a few ULPs of noise vs the strict
    // tree-reduce sum.
    assert!(
        fused.residual <= 2.0 * tol,
        "fused residual {:.3e} far above tol {:.3e}",
        fused.residual,
        tol,
    );
}
