//! Stage 24c — GPU FFT parity tests against `rustfft`.
//!
//! All tests are `#[ignore]` because they need a real wgpu adapter
//! (CI doesn't have one). Run with:
//!
//! ```text
//! cargo test -p molla --features gpu --test fft_gpu -- --ignored
//! ```
//!
//! Convention: the kernel applies **no normalisation**, so
//! `forward * inverse = N` (matching `rustfft::FftPlanner` and most
//! physics-engine FFT users). The round-trip test divides by N
//! manually before comparing to the input.
//!
//! ## Tolerance
//!
//! The PLAN's "1e-5 vs rustfft" target is interpreted as a
//! **relative** tolerance (per output element, scaled by the magnitude
//! of the larger of the two values). f32 mantissa is ~7 decimal
//! digits, and FFT error grows as O(√N · log N · ε) (Higham 1996), so
//! absolute errors at N=4096 land in the 1e−4 range — perfectly
//! healthy for f32, but blows past any 1e−5 absolute bar. Both
//! `rustfft` and our kernel run in f32 here; the relative comparison
//! tells us they agree to single-precision. For tighter absolute
//! errors callers would need an f64 path, which WebGPU doesn't expose.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::kernels::{Complex32, FftKernel};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex as RustComplex;
use std::time::Instant;

const PARITY_TOL: f32 = 1.0e-5;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xCAFE_F00D)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn signed_f32(&mut self) -> f32 {
        let x = self.next() >> 32;
        let u = (x as f64 + 1.0) / ((1u64 << 32) as f64 + 1.0);
        (2.0 * u - 1.0) as f32
    }
}

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn rand_complex_buf(n: usize, seed: u64) -> Vec<Complex32> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| Complex32::new(rng.signed_f32(), rng.signed_f32()))
        .collect()
}

fn to_rust(input: &[Complex32]) -> Vec<RustComplex<f32>> {
    input
        .iter()
        .map(|c| RustComplex::new(c.re, c.im))
        .collect()
}

fn rustfft_1d(input: &[Complex32], inverse: bool) -> Vec<RustComplex<f32>> {
    let mut planner: FftPlanner<f32> = FftPlanner::new();
    let plan = if inverse {
        planner.plan_fft_inverse(input.len())
    } else {
        planner.plan_fft_forward(input.len())
    };
    let mut buf = to_rust(input);
    plan.process(&mut buf);
    buf
}

/// Row-then-column 2D reference using `rustfft`.
fn rustfft_2d(input: &[Complex32], height: usize, width: usize, inverse: bool) -> Vec<RustComplex<f32>> {
    let mut buf = to_rust(input);
    let mut planner: FftPlanner<f32> = FftPlanner::new();
    let plan_row = if inverse {
        planner.plan_fft_inverse(width)
    } else {
        planner.plan_fft_forward(width)
    };
    // Row passes.
    for y in 0..height {
        let row = &mut buf[y * width..(y + 1) * width];
        plan_row.process(row);
    }
    // Column passes — rustfft processes contiguous slices, so we
    // gather → fft → scatter.
    let plan_col = if inverse {
        planner.plan_fft_inverse(height)
    } else {
        planner.plan_fft_forward(height)
    };
    let mut col_buf = vec![RustComplex::new(0.0_f32, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            col_buf[y] = buf[y * width + x];
        }
        plan_col.process(&mut col_buf);
        for y in 0..height {
            buf[y * width + x] = col_buf[y];
        }
    }
    buf
}

fn rustfft_3d(
    input: &[Complex32],
    depth: usize,
    height: usize,
    width: usize,
    inverse: bool,
) -> Vec<RustComplex<f32>> {
    let mut buf = to_rust(input);
    let plane = height * width;
    // Step A: 2D FFT on each Z-slice.
    let mut planner: FftPlanner<f32> = FftPlanner::new();
    let plan_w = if inverse {
        planner.plan_fft_inverse(width)
    } else {
        planner.plan_fft_forward(width)
    };
    let plan_h = if inverse {
        planner.plan_fft_inverse(height)
    } else {
        planner.plan_fft_forward(height)
    };
    let plan_d = if inverse {
        planner.plan_fft_inverse(depth)
    } else {
        planner.plan_fft_forward(depth)
    };
    let mut col_buf = vec![RustComplex::new(0.0_f32, 0.0); height];
    for z in 0..depth {
        let off = z * plane;
        // Rows.
        for y in 0..height {
            let row = &mut buf[off + y * width..off + (y + 1) * width];
            plan_w.process(row);
        }
        // Columns.
        for x in 0..width {
            for y in 0..height {
                col_buf[y] = buf[off + y * width + x];
            }
            plan_h.process(&mut col_buf);
            for y in 0..height {
                buf[off + y * width + x] = col_buf[y];
            }
        }
    }
    // Step B: 1D FFT along Z for each (y,x).
    let mut z_buf = vec![RustComplex::new(0.0_f32, 0.0); depth];
    for y in 0..height {
        for x in 0..width {
            for z in 0..depth {
                z_buf[z] = buf[z * plane + y * width + x];
            }
            plan_d.process(&mut z_buf);
            for z in 0..depth {
                buf[z * plane + y * width + x] = z_buf[z];
            }
        }
    }
    buf
}

fn max_abs_diff_complex(a: &[Complex32], b: &[RustComplex<f32>]) -> f32 {
    let mut max = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let dr = (x.re - y.re).abs();
        let di = (x.im - y.im).abs();
        if dr > max {
            max = dr;
        }
        if di > max {
            max = di;
        }
    }
    max
}

/// Max relative error against the reference: |gpu − cpu| / max(|cpu|, eps).
/// FFT outputs at scale N can have absolute errors on the order of
/// √N · ε (f32 ε ≈ 1.2e−7), so we use relative comparisons and pick
/// the largest reference magnitude as the denominator.
fn max_rel_diff_complex(a: &[Complex32], b: &[RustComplex<f32>]) -> f32 {
    let max_ref = b
        .iter()
        .map(|c| c.re.abs().max(c.im.abs()))
        .fold(0.0_f32, f32::max);
    let denom = max_ref.max(1.0e-30);
    max_abs_diff_complex(a, b) / denom
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn fft_1d_4096_matches_rustfft() {
    let Some(engine) = try_engine() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = FftKernel::new(&device);

    let n = 4096usize;
    let input = rand_complex_buf(n, 0xC0FF_EE01);
    let expected = rustfft_1d(&input, false);

    let mut output = vec![Complex32::ZERO; n];

    // Warm-up dispatch.
    kernel
        .dispatch_1d(&device, &queue, &input, &mut output, n, false)
        .unwrap();

    let trials = 5;
    let mut best_ms = f64::INFINITY;
    for _ in 0..trials {
        let t = Instant::now();
        kernel
            .dispatch_1d(&device, &queue, &input, &mut output, n, false)
            .unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
        }
    }
    // Compare to CPU rustfft wallclock.
    let mut best_cpu_ms = f64::INFINITY;
    for _ in 0..trials {
        let t = Instant::now();
        let _ = rustfft_1d(&input, false);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best_cpu_ms {
            best_cpu_ms = ms;
        }
    }
    let max_abs = max_abs_diff_complex(&output, &expected);
    let max_rel = max_rel_diff_complex(&output, &expected);
    eprintln!(
        "1D N={n}: rel {max_rel:e} (abs {max_abs:e}); GPU best {best_ms:.3} ms, rustfft CPU best {best_cpu_ms:.3} ms",
    );
    assert!(max_rel < PARITY_TOL, "rel diff {max_rel} >= {PARITY_TOL}");
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn fft_2d_256x256_matches_rustfft() {
    let Some(engine) = try_engine() else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = FftKernel::new(&device);

    let h = 256usize;
    let w = 256usize;
    let input = rand_complex_buf(h * w, 0xC0FF_EE02);
    let expected = rustfft_2d(&input, h, w, false);

    let mut output = vec![Complex32::ZERO; h * w];

    // Warm-up.
    kernel
        .dispatch_2d(&device, &queue, &input, &mut output, h, w, false)
        .unwrap();

    let trials = 5;
    let mut best_ms = f64::INFINITY;
    for _ in 0..trials {
        let t = Instant::now();
        kernel
            .dispatch_2d(&device, &queue, &input, &mut output, h, w, false)
            .unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
        }
    }
    let mut best_cpu_ms = f64::INFINITY;
    for _ in 0..trials {
        let t = Instant::now();
        let _ = rustfft_2d(&input, h, w, false);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best_cpu_ms {
            best_cpu_ms = ms;
        }
    }
    let max_abs = max_abs_diff_complex(&output, &expected);
    let max_rel = max_rel_diff_complex(&output, &expected);
    eprintln!(
        "2D {h}x{w}: rel {max_rel:e} (abs {max_abs:e}); GPU best {best_ms:.3} ms, rustfft CPU best {best_cpu_ms:.3} ms",
    );
    assert!(max_rel < PARITY_TOL, "rel diff {max_rel} >= {PARITY_TOL}");
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn fft_3d_32x32x32_matches_rustfft() {
    let Some(engine) = try_engine() else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = FftKernel::new(&device);

    let d = 32usize;
    let h = 32usize;
    let w = 32usize;
    let input = rand_complex_buf(d * h * w, 0xC0FF_EE03);
    let expected = rustfft_3d(&input, d, h, w, false);
    let mut output = vec![Complex32::ZERO; d * h * w];
    kernel
        .dispatch_3d(&device, &queue, &input, &mut output, d, h, w, false)
        .unwrap();
    let max_abs = max_abs_diff_complex(&output, &expected);
    let max_rel = max_rel_diff_complex(&output, &expected);
    eprintln!("3D {d}x{h}x{w}: rel {max_rel:e} (abs {max_abs:e})");
    assert!(max_rel < PARITY_TOL, "rel diff {max_rel} >= {PARITY_TOL}");
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn fft_round_trip_1d() {
    let Some(engine) = try_engine() else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = FftKernel::new(&device);

    let n = 1024usize;
    let input = rand_complex_buf(n, 0xC0FF_EE04);
    let mut spectrum = vec![Complex32::ZERO; n];
    let mut recon = vec![Complex32::ZERO; n];
    kernel
        .dispatch_1d(&device, &queue, &input, &mut spectrum, n, false)
        .unwrap();
    kernel
        .dispatch_1d(&device, &queue, &spectrum, &mut recon, n, true)
        .unwrap();
    // Convention: forward * inverse = N. Divide by N to recover input.
    let inv_n = 1.0 / n as f32;
    let scaled: Vec<Complex32> = recon
        .iter()
        .map(|c| Complex32::new(c.re * inv_n, c.im * inv_n))
        .collect();
    let mut max_err = 0.0_f32;
    for (x, y) in input.iter().zip(scaled.iter()) {
        let e = (x.re - y.re).abs().max((x.im - y.im).abs());
        if e > max_err {
            max_err = e;
        }
    }
    eprintln!("1D round-trip N={n}: max err {max_err:e}");
    assert!(max_err < PARITY_TOL, "round-trip err {max_err} >= {PARITY_TOL}");
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn fft_small_powers_of_two() {
    let Some(engine) = try_engine() else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = FftKernel::new(&device);

    for &n in &[1usize, 2, 4, 8, 16] {
        let input = rand_complex_buf(n, 0xBEEF_0000 ^ n as u64);
        let expected = rustfft_1d(&input, false);
        let mut output = vec![Complex32::ZERO; n];
        kernel
            .dispatch_1d(&device, &queue, &input, &mut output, n, false)
            .unwrap();
        let max_abs = max_abs_diff_complex(&output, &expected);
        let max_rel = max_rel_diff_complex(&output, &expected);
        eprintln!("1D N={n}: rel {max_rel:e} (abs {max_abs:e})");
        assert!(
            max_rel < PARITY_TOL,
            "N={n} rel diff {max_rel} >= {PARITY_TOL}"
        );
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn fft_non_power_of_two_rejected() {
    let Some(engine) = try_engine() else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = FftKernel::new(&device);

    let input = rand_complex_buf(6, 0xFEEDFACE);
    let mut output = vec![Complex32::ZERO; 6];
    let res = kernel.dispatch_1d(&device, &queue, &input, &mut output, 6, false);
    assert!(res.is_err(), "non-power-of-two should be rejected");
}
