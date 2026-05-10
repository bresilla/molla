//! Stage 37.8 — CPU-only oracle tests for the tile compute primitives.
//!
//! GPU parity tests live in `tile_primitives_gpu.rs` (gated behind the
//! `gpu` feature + `#[ignore]`).

use molla::compute::tile_matmul;

/// Tiny LCG so tests are dependency-free and reproducible.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn signed(&mut self) -> f32 {
        let x = self.next() >> 11;
        let u = (x as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        (2.0 * u - 1.0) as f32
    }
}

// ── A — matmul ───────────────────────────────────────────────────────

/// Independent reference impl using f64 accumulation; used to check the
/// f32 CPU oracle picks up no bias beyond round-off.
fn matmul_oracle_f64(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

#[test]
fn tile_matmul_cpu_matches_oracle_small() {
    // 4 × 3 × 5 — small enough to trace by hand.
    let m = 4;
    let k = 3;
    let n = 5;
    let mut rng = Lcg::new(1);
    let a: Vec<f32> = (0..m * k).map(|_| rng.signed()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.signed()).collect();
    let mut c = vec![0.0_f32; m * n];
    tile_matmul::matmul_cpu(&a, &b, &mut c, m, n, k).unwrap();
    let oracle = matmul_oracle_f64(&a, &b, m, n, k);
    for (g, e) in c.iter().zip(oracle.iter()) {
        assert!((g - e).abs() < 1e-5, "got {g} expected {e}");
    }
}

#[test]
fn tile_matmul_cpu_random_256_x_128_x_256() {
    // Same shape the GPU parity test uses; here we just sanity-check
    // the CPU reference against a fresh f64 oracle.
    let m = 256;
    let k = 128;
    let n = 256;
    let mut rng = Lcg::new(2);
    let a: Vec<f32> = (0..m * k).map(|_| rng.signed()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.signed()).collect();
    let mut c = vec![0.0_f32; m * n];
    tile_matmul::matmul_cpu(&a, &b, &mut c, m, n, k).unwrap();
    let oracle = matmul_oracle_f64(&a, &b, m, n, k);
    let mut max_err: f32 = 0.0;
    for (g, e) in c.iter().zip(oracle.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 1e-3, "max_err = {max_err}");
}

#[test]
fn tile_matmul_cpu_identity_returns_input() {
    let n = 8;
    let mut a = vec![0.0_f32; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    let mut rng = Lcg::new(3);
    let b: Vec<f32> = (0..n * n).map(|_| rng.signed()).collect();
    let mut c = vec![0.0_f32; n * n];
    tile_matmul::matmul_cpu(&a, &b, &mut c, n, n, n).unwrap();
    for (g, e) in c.iter().zip(b.iter()) {
        assert!((g - e).abs() < 1e-6);
    }
}

#[test]
fn tile_matmul_cpu_rectangular_shapes_compose() {
    // (M=7, K=3, N=11) → not divisible by 16, so the CPU oracle
    // doesn't care; the GPU side does and is tested separately.
    let m = 7;
    let k = 3;
    let n = 11;
    let mut rng = Lcg::new(4);
    let a: Vec<f32> = (0..m * k).map(|_| rng.signed()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.signed()).collect();
    let mut c = vec![0.0_f32; m * n];
    tile_matmul::matmul_cpu(&a, &b, &mut c, m, n, k).unwrap();
    let oracle = matmul_oracle_f64(&a, &b, m, n, k);
    for (g, e) in c.iter().zip(oracle.iter()) {
        assert!((g - e).abs() < 1e-5);
    }
}

#[test]
fn tile_matmul_cpu_rejects_shape_mismatch() {
    let mut c = vec![0.0_f32; 4];
    let err = tile_matmul::matmul_cpu(&[1.0, 2.0], &[3.0, 4.0], &mut c, 2, 2, 2);
    assert!(err.is_err());
}
