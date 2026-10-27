//! Stage 38a — GPU sparse BSR algebra parity tests.
//!
//! Four `#[ignore]` tests (need a real wgpu adapter):
//!   1. `bsr_matvec_gpu_matches_cpu`     — generalised `y = α·A·x + β·y`.
//!   2. `bsr_axpy_same_pattern_matches_cpu` — `B = α·A + β·B`.
//!   3. `bsr_transpose_matches_cpu`      — `B = Aᵀ` pattern + values.
//!   4. `bsr_spgemm_matches_cpu`         — `C = A · B`.
//!
//! Run with:
//! ```sh
//! make test-gpu P=molla-kernels T=bsr_gpu_parity
//! ```
//!
//! A `GPU_LOCK` mutex serialises adapter ownership so the test binary
//! can run all four concurrently inside the same harness (same pattern
//! as `tests/fem_gpu.rs`).

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_kernels::{BsrAlgebraKernel, BsrGpu};
use molla_math::Real;
use molla_math::sparse::BsrMatrix;

static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Cheap deterministic LCG so tests don't need a `rand` dep.
fn lcg(seed: u64) -> impl FnMut() -> Real {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 32) as u32) as Real) / (u32::MAX as Real) - 0.5
    }
}

/// Build a random SPD-ish `n × n` BSR matrix with ~5-banded sparsity:
/// diagonal + a few random off-diagonals, symmetrised by adding the
/// transposed pattern.
fn random_banded(n: usize, seed: u64) -> BsrMatrix {
    let mut rng = lcg(seed);
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    // Strong diagonal so the matrix is well-conditioned.
    for i in 0..n {
        triplets.push((i, i, 2.0 + rng().abs()));
    }
    // Small symmetric off-diagonals.
    let band = 3usize;
    for i in 0..n {
        for d in 1..=band {
            let j = (i + d) % n;
            if j == i {
                continue;
            }
            let v = (rng()) * 0.1;
            triplets.push((i, j, v));
            triplets.push((j, i, v));
        }
    }
    BsrMatrix::from_scalar_triplets(n, n, &triplets).expect("triplets")
}

/// Build an arbitrary `rows × cols` matrix with a non-trivial pattern
/// (used by the transpose test).
fn random_rectangular(rows: usize, cols: usize, seed: u64) -> BsrMatrix {
    let mut rng = lcg(seed);
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    // Sprinkle ~3 nonzeros per row.
    for i in 0..rows {
        for _ in 0..3 {
            let j = ((((rng() + 0.5) * cols as Real) as usize) % cols).min(cols - 1);
            triplets.push((i, j, rng()));
        }
        // Also force a diagonal-ish entry so the pattern's never empty.
        let j = i % cols;
        triplets.push((i, j, 1.0 + rng().abs()));
    }
    BsrMatrix::from_scalar_triplets(rows, cols, &triplets).expect("triplets")
}

fn cpu_matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    let mut y = vec![0.0; a.nrows()];
    a.mv(x, &mut y).unwrap();
    y
}

/// CPU transpose for B = 1 BSR (CSR) — reference for the GPU result.
fn cpu_transpose(a: &BsrMatrix) -> BsrMatrix {
    assert_eq!(a.b, 1);
    let rows = a.block_rows;
    let cols = a.block_cols;
    let nnz = a.values.len();
    let mut row_offsets = vec![0usize; cols + 1];
    for &c in &a.col_idx {
        row_offsets[c + 1] += 1;
    }
    for i in 1..=cols {
        row_offsets[i] += row_offsets[i - 1];
    }
    let mut col_idx = vec![0usize; nnz];
    let mut values = vec![0.0; nnz];
    let mut cursor = row_offsets.clone();
    for i in 0..rows {
        let lo = a.row_offsets[i];
        let hi = a.row_offsets[i + 1];
        for slot in lo..hi {
            let j = a.col_idx[slot];
            let dst = cursor[j];
            col_idx[dst] = i;
            values[dst] = a.values[slot];
            cursor[j] += 1;
        }
    }
    BsrMatrix {
        b: 1,
        block_rows: cols,
        block_cols: rows,
        row_offsets,
        col_idx,
        values,
    }
}

/// CPU SpGEMM `C = A · B` — reference for the GPU result. Uses a
/// straightforward row-wise accumulator.
fn cpu_spgemm(a: &BsrMatrix, b: &BsrMatrix) -> BsrMatrix {
    assert_eq!(a.b, 1);
    assert_eq!(b.b, 1);
    assert_eq!(a.block_cols, b.block_rows);
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for i in 0..a.block_rows {
        let mut accum: std::collections::BTreeMap<usize, Real> = std::collections::BTreeMap::new();
        let a_lo = a.row_offsets[i];
        let a_hi = a.row_offsets[i + 1];
        for sa in a_lo..a_hi {
            let k = a.col_idx[sa];
            let av = a.values[sa];
            let b_lo = b.row_offsets[k];
            let b_hi = b.row_offsets[k + 1];
            for sb in b_lo..b_hi {
                let j = b.col_idx[sb];
                *accum.entry(j).or_insert(0.0) += av * b.values[sb];
            }
        }
        for (j, v) in accum {
            triplets.push((i, j, v));
        }
    }
    BsrMatrix::from_scalar_triplets(a.block_rows, b.block_cols, &triplets).unwrap()
}

/// Allocate an `n`-element f32 storage buffer (STORAGE | COPY_SRC | COPY_DST)
/// and seed it from a host slice. The minimum binding size of 16 bytes
/// is observed.
fn upload_vec(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> wgpu::Buffer {
    let bytes = (data.len() as u64).max(4) * 4;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bsr_gpu_parity::upload_vec"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
    }
    buf
}

/// Read `n` f32 values from a storage buffer.
fn readback_vec(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    n: usize,
) -> Vec<f32> {
    let bytes = (n.max(4) as u64) * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bsr_gpu_parity::readback_vec staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bsr_gpu_parity::readback_vec enc"),
    });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let mut v: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    v.truncate(n);
    v
}

#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn bsr_matvec_gpu_matches_cpu() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let n = 64;
    let a_cpu = random_banded(n, 0xA1);
    let mut rng = lcg(0xCAFEF00D);
    let x_f64: Vec<Real> = (0..n).map(|_| rng()).collect();
    let y0_f64: Vec<Real> = (0..n).map(|_| rng()).collect();

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = BsrAlgebraKernel::new(device.clone(), queue.clone());
    let a_gpu = BsrGpu::upload(&device, &queue, &a_cpu).expect("upload");

    // ── Case 1: α = 1, β = 0 (the pure matvec contract). ──────────
    let y_cpu = cpu_matvec(&a_cpu, &x_f64);
    let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
    let y_buf = upload_vec(&device, &queue, &vec![0.0f32; n]);
    let x_buf = upload_vec(&device, &queue, &x_f32);
    kernel
        .matvec(&a_gpu, &x_buf, &y_buf, 1.0, 0.0)
        .expect("matvec");
    let y_gpu = readback_vec(&device, &queue, &y_buf, n);

    let mut max_err_1: f32 = 0.0;
    for i in 0..n {
        let err = (y_gpu[i] - y_cpu[i] as f32).abs();
        max_err_1 = max_err_1.max(err);
        assert!(
            err < 1e-4,
            "row {i}: gpu={} cpu={} err={err}",
            y_gpu[i],
            y_cpu[i]
        );
    }

    // ── Case 2: α = 0.5, β = 2.0 (the general BLAS-style contract). ──
    let alpha = 0.5f32;
    let beta = 2.0f32;
    // CPU oracle: y = α·A·x + β·y0
    let mut y_ref = cpu_matvec(&a_cpu, &x_f64);
    for i in 0..n {
        y_ref[i] = (alpha as Real) * y_ref[i] + (beta as Real) * y0_f64[i];
    }
    let y0_f32: Vec<f32> = y0_f64.iter().map(|&v| v as f32).collect();
    let y_buf2 = upload_vec(&device, &queue, &y0_f32);
    kernel
        .matvec(&a_gpu, &x_buf, &y_buf2, alpha, beta)
        .expect("matvec general");
    let y_gpu2 = readback_vec(&device, &queue, &y_buf2, n);

    let mut max_err_2: f32 = 0.0;
    for i in 0..n {
        let err = (y_gpu2[i] - y_ref[i] as f32).abs();
        max_err_2 = max_err_2.max(err);
        assert!(
            err < 1e-4,
            "general row {i}: gpu={} cpu={} err={err}",
            y_gpu2[i],
            y_ref[i]
        );
    }
    eprintln!(
        "bsr_matvec_gpu_matches_cpu: n={n}, max_err (α=1,β=0)={:.2e}, max_err (α=0.5,β=2.0)={:.2e}",
        max_err_1, max_err_2
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn bsr_axpy_same_pattern_matches_cpu() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Two banded matrices share the same pattern (random_banded uses a
    // deterministic banded layout — only the seed changes the VALUES).
    let n = 48;
    let a_cpu = random_banded(n, 0x33);
    let b_cpu = random_banded(n, 0x77);
    // Sanity-check that the patterns line up — if `random_banded` ever
    // got non-deterministic this assertion would catch it before the
    // GPU call.
    assert_eq!(a_cpu.row_offsets, b_cpu.row_offsets);
    assert_eq!(a_cpu.col_idx, b_cpu.col_idx);

    let alpha = 0.75f32;
    let beta = -1.5f32;
    // CPU oracle: b ← α·a + β·b (per-slot blend; same pattern).
    let mut b_ref_values = b_cpu.values.clone();
    for i in 0..b_cpu.values.len() {
        b_ref_values[i] = (alpha as Real) * a_cpu.values[i] + (beta as Real) * b_cpu.values[i];
    }

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = BsrAlgebraKernel::new(device.clone(), queue.clone());
    let a_gpu = BsrGpu::upload(&device, &queue, &a_cpu).expect("a upload");
    let mut b_gpu = BsrGpu::upload(&device, &queue, &b_cpu).expect("b upload");
    kernel
        .axpy_same_pattern_unchecked(&a_gpu, &mut b_gpu, alpha, beta)
        .expect("axpy");

    let b_read = b_gpu.read_back(&device, &queue).expect("readback");
    let mut max_err: f32 = 0.0;
    for i in 0..b_read.values.len() {
        let err = (b_read.values[i] - b_ref_values[i]).abs() as f32;
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "axpy slot {i}: gpu={} cpu={} err={err}",
            b_read.values[i],
            b_ref_values[i]
        );
    }
    eprintln!(
        "bsr_axpy_same_pattern_matches_cpu: n={n}, nnz={}, max_err={:.2e}",
        a_cpu.values.len(),
        max_err
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn bsr_transpose_matches_cpu() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 32×64 rectangular matrix — non-square exercises the count +
    // scatter for a `cols` dimension that doesn't match `rows`.
    let a_cpu = random_rectangular(32, 64, 0xBADF00D);
    let at_cpu = cpu_transpose(&a_cpu);

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = BsrAlgebraKernel::new(device.clone(), queue.clone());
    let a_gpu = BsrGpu::upload(&device, &queue, &a_cpu).expect("upload");
    let at_gpu = kernel.transpose(&a_gpu).expect("transpose");

    assert_eq!(at_gpu.nrows as usize, at_cpu.block_rows);
    assert_eq!(at_gpu.ncols as usize, at_cpu.block_cols);
    assert_eq!(at_gpu.nnz as usize, at_cpu.values.len());

    let at_read = at_gpu.read_back(&device, &queue).expect("readback");
    // row_offsets + col_idx must be byte-identical (pattern is exact).
    assert_eq!(
        at_read.row_offsets, at_cpu.row_offsets,
        "transpose row_offsets mismatch"
    );
    assert_eq!(
        at_read.col_idx, at_cpu.col_idx,
        "transpose col_idx mismatch"
    );
    // Values are the same set as `A.values` permuted by `dst_map` —
    // no FP accumulation, so the f32 round-trip is the only source of
    // error. Tolerance is the f32 ULP for values < ~10.
    let mut max_err: f32 = 0.0;
    for i in 0..at_read.values.len() {
        let err = (at_read.values[i] - at_cpu.values[i]).abs() as f32;
        max_err = max_err.max(err);
        assert!(
            err < 1e-6,
            "transpose value slot {i}: gpu={} cpu={} err={err}",
            at_read.values[i],
            at_cpu.values[i]
        );
    }
    eprintln!(
        "bsr_transpose_matches_cpu: {}x{}, nnz={}, max_err={:.2e}",
        a_cpu.block_rows,
        a_cpu.block_cols,
        a_cpu.values.len(),
        max_err
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn bsr_spgemm_matches_cpu() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Two random 32×32 SPD-ish matrices. SPD isn't strictly required
    // for SpGEMM correctness; using `random_banded` keeps the patterns
    // dense enough to exercise the contribution-list packing.
    let n = 32;
    let a_cpu = random_banded(n, 0x111);
    let b_cpu = random_banded(n, 0x222);
    let c_cpu = cpu_spgemm(&a_cpu, &b_cpu);

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = BsrAlgebraKernel::new(device.clone(), queue.clone());
    let a_gpu = BsrGpu::upload(&device, &queue, &a_cpu).expect("a upload");
    let b_gpu = BsrGpu::upload(&device, &queue, &b_cpu).expect("b upload");

    let t0 = std::time::Instant::now();
    let c_gpu = kernel.spgemm(&a_gpu, &b_gpu).expect("spgemm");
    let dt_gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    let _c_cpu2 = cpu_spgemm(&a_cpu, &b_cpu);
    let dt_cpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(c_gpu.nrows as usize, c_cpu.block_rows);
    assert_eq!(c_gpu.ncols as usize, c_cpu.block_cols);
    assert_eq!(c_gpu.nnz as usize, c_cpu.values.len());

    let c_read = c_gpu.read_back(&device, &queue).expect("readback");
    assert_eq!(c_read.row_offsets, c_cpu.row_offsets, "spgemm row_offsets");
    assert_eq!(c_read.col_idx, c_cpu.col_idx, "spgemm col_idx");
    let mut max_err: f32 = 0.0;
    for i in 0..c_read.values.len() {
        let err = (c_read.values[i] - c_cpu.values[i]).abs() as f32;
        max_err = max_err.max(err);
        assert!(
            err < 1e-4,
            "spgemm slot {i}: gpu={} cpu={} err={err}",
            c_read.values[i],
            c_cpu.values[i]
        );
    }
    eprintln!(
        "bsr_spgemm_matches_cpu: {}x{}·{}x{}, c_nnz={}, max_err={:.2e}, gpu={:.2}ms cpu={:.2}ms",
        n,
        n,
        n,
        n,
        c_cpu.values.len(),
        max_err,
        dt_gpu_ms,
        dt_cpu_ms
    );
}
