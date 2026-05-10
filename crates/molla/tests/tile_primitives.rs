//! Stage 37.8 — CPU-only oracle tests for the tile compute primitives.
//!
//! GPU parity tests live in `tile_primitives_gpu.rs` (gated behind the
//! `gpu` feature + `#[ignore]`).

use molla::compute::tile_convolution::{self, Kernel5};
use molla::compute::tile_mcgp::{self, McgpConfig};
use molla::compute::{tile_block_cholesky, tile_filtering, tile_matmul, tile_stream_compaction};

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

// ── B — block Cholesky ───────────────────────────────────────────────

/// Build a random SPD matrix `M Mᵀ + n·I` of size `n × n`.
fn random_spd(rng: &mut Lcg, n: usize) -> Vec<f32> {
    let mut m = vec![0.0_f32; n * n];
    for v in m.iter_mut() {
        *v = rng.signed();
    }
    let mut a = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f32;
            for k in 0..n {
                s += m[i * n + k] * m[j * n + k];
            }
            a[i * n + j] = s;
            if i == j {
                a[i * n + j] += n as f32;
            }
        }
    }
    a
}

#[test]
fn tile_block_cholesky_cpu_factorises_64x64_spd() {
    let n = 64;
    let mut rng = Lcg::new(11);
    let a = random_spd(&mut rng, n);
    let mut l = a.clone();
    tile_block_cholesky::cholesky_cpu(&mut l, n).unwrap();
    let err = tile_block_cholesky::frobenius_recon_error(&a, &l, n);
    assert!(err < 1.0e-3, "‖L Lᵀ − A‖_F = {err}");

    // Also confirm the strict upper triangle is zero.
    for i in 0..n {
        for j in (i + 1)..n {
            assert_eq!(l[i * n + j], 0.0);
        }
    }
}

#[test]
fn tile_block_cholesky_cpu_diagonal_matches_sqrt() {
    // A = diag(d). L = diag(sqrt(d)).
    let n = 8;
    let mut a = vec![0.0_f32; n * n];
    for i in 0..n {
        a[i * n + i] = (i + 1) as f32 * (i + 1) as f32; // 1, 4, 9, 16, ...
    }
    tile_block_cholesky::cholesky_cpu(&mut a, n).unwrap();
    for i in 0..n {
        assert!((a[i * n + i] - (i + 1) as f32).abs() < 1e-5);
    }
}

#[test]
fn tile_block_cholesky_cpu_identity_returns_identity() {
    let n = 16;
    let mut a = vec![0.0_f32; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    let orig = a.clone();
    tile_block_cholesky::cholesky_cpu(&mut a, n).unwrap();
    for (g, e) in a.iter().zip(orig.iter()) {
        assert!((g - e).abs() < 1e-6);
    }
}

#[test]
fn tile_block_cholesky_cpu_rejects_shape_mismatch() {
    let mut a = vec![0.0_f32; 4];
    let err = tile_block_cholesky::cholesky_cpu(&mut a, 3);
    assert!(err.is_err());
}

// ── C — separable 2D convolution ─────────────────────────────────────

#[test]
fn tile_convolution_cpu_identity_returns_input() {
    let w = 16;
    let h = 16;
    let mut rng = Lcg::new(31);
    let input: Vec<f32> = (0..w * h).map(|_| rng.signed()).collect();
    let mut output = vec![0.0_f32; w * h];
    tile_convolution::convolve_separable_cpu(&input, &mut output, w, h, &Kernel5::identity())
        .unwrap();
    for (g, e) in output.iter().zip(input.iter()) {
        assert!((g - e).abs() < 1e-6);
    }
}

#[test]
fn tile_convolution_cpu_gaussian_constant_field_returns_constant() {
    // A constant field should be unchanged by any normalised filter.
    let w = 32;
    let h = 32;
    let input = vec![3.5_f32; w * h];
    let mut output = vec![0.0_f32; w * h];
    tile_convolution::convolve_separable_cpu(&input, &mut output, w, h, &Kernel5::gaussian5())
        .unwrap();
    for v in output.iter() {
        assert!((v - 3.5).abs() < 1e-5);
    }
}

#[test]
fn tile_convolution_cpu_box_smooths_to_local_mean() {
    // A spike at the centre of an otherwise-zero field, smoothed by a
    // box filter, should leave the centre tap with the expected value.
    let w = 9;
    let h = 9;
    let mut input = vec![0.0_f32; w * h];
    input[4 * w + 4] = 1.0;
    let mut output = vec![0.0_f32; w * h];
    tile_convolution::convolve_separable_cpu(&input, &mut output, w, h, &Kernel5::box5()).unwrap();
    // Box separable: each axis pass spreads 1/5 of the spike to its
    // 5-tap window. After both passes, the centre tap holds (1/5)² =
    // 0.04 and 5×5 = 25 cells around the spike are non-zero.
    assert!((output[4 * w + 4] - 0.04).abs() < 1e-6);
    let total: f32 = output.iter().sum();
    assert!((total - 1.0).abs() < 1e-5);
}

#[test]
fn tile_convolution_cpu_rejects_shape_mismatch() {
    let input = vec![0.0_f32; 8];
    let mut output = vec![0.0_f32; 6];
    let err = tile_convolution::convolve_separable_cpu(
        &input,
        &mut output,
        4,
        2,
        &Kernel5::identity(),
    );
    assert!(err.is_err());
}

// ── D — filtering helpers ────────────────────────────────────────────

#[test]
fn tile_filtering_identity_returns_input() {
    let w = 24;
    let h = 16;
    let mut rng = Lcg::new(41);
    let input: Vec<f32> = (0..w * h).map(|_| rng.signed()).collect();
    let mut output = vec![0.0_f32; w * h];
    tile_filtering::identity_filter_cpu(&input, &mut output, w, h).unwrap();
    for (g, e) in output.iter().zip(input.iter()) {
        assert!((g - e).abs() < 1e-6);
    }
}

#[test]
fn tile_filtering_gaussian_smooths_constant_to_constant() {
    let w = 32;
    let h = 32;
    let input = vec![2.5_f32; w * h];
    let mut output = vec![0.0_f32; w * h];
    tile_filtering::gaussian_filter_cpu(&input, &mut output, w, h).unwrap();
    for v in output.iter() {
        assert!((v - 2.5).abs() < 1e-5);
    }
}

#[test]
fn tile_filtering_box_smooths_constant_to_constant() {
    let w = 32;
    let h = 32;
    let input = vec![1.0_f32; w * h];
    let mut output = vec![0.0_f32; w * h];
    tile_filtering::box_filter_cpu(&input, &mut output, w, h).unwrap();
    for v in output.iter() {
        assert!((v - 1.0).abs() < 1e-5);
    }
}

#[test]
fn tile_filtering_laplacian_kills_constant() {
    // Laplacian of a constant is zero. (Boundary clamps to the
    // constant, so even edge pixels see a flat stencil.)
    let w = 16;
    let h = 16;
    let input = vec![3.14_f32; w * h];
    let mut output = vec![0.0_f32; w * h];
    tile_filtering::laplacian_filter_cpu(&input, &mut output, w, h).unwrap();
    for v in output.iter() {
        assert!(v.abs() < 1e-5, "got {v}");
    }
}

// ── E — stream compaction ────────────────────────────────────────────

#[test]
fn tile_stream_compaction_cpu_keeps_all_when_mask_all_true() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mask = vec![true; 16];
    let out = tile_stream_compaction::compact_cpu(&input, &mask).unwrap();
    assert_eq!(out, input);
}

#[test]
fn tile_stream_compaction_cpu_drops_all_when_mask_all_false() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mask = vec![false; 16];
    let out = tile_stream_compaction::compact_cpu(&input, &mask).unwrap();
    assert!(out.is_empty());
}

#[test]
fn tile_stream_compaction_cpu_random_1024() {
    let n = 1024;
    let mut rng = Lcg::new(51);
    let input: Vec<f32> = (0..n).map(|_| rng.signed()).collect();
    let mask: Vec<bool> = (0..n).map(|_| (rng.next() & 1) == 1).collect();
    let out = tile_stream_compaction::compact_cpu(&input, &mask).unwrap();
    let expected_count = mask.iter().filter(|x| **x).count();
    assert_eq!(out.len(), expected_count);

    // Content-wise: walking input + mask in lock-step should match.
    let expected: Vec<f32> = input
        .iter()
        .zip(mask.iter())
        .filter_map(|(v, &k)| if k { Some(*v) } else { None })
        .collect();
    assert_eq!(out, expected);
}

#[test]
fn tile_stream_compaction_cpu_rejects_length_mismatch() {
    let input = vec![0.0_f32; 4];
    let mask = vec![true; 3];
    assert!(tile_stream_compaction::compact_cpu(&input, &mask).is_err());
}

#[test]
fn tile_stream_compaction_mask_to_u32_round_trips() {
    let mask = vec![true, false, true, true, false];
    let u = tile_stream_compaction::mask_to_u32(&mask);
    assert_eq!(u, vec![1, 0, 1, 1, 0]);
}

// ── F — Monte Carlo gradient propagation ─────────────────────────────

/// Quadratic loss `f(p) = Σ (p_i − target_i)²`. Closed-form gradient
/// is `2 (p − target)`, perfectly smooth — ideal SPSA target.
fn quadratic_loss(p: &[f64], target: &[f64]) -> f64 {
    p.iter().zip(target.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

#[test]
fn tile_mcgp_quadratic_within_tolerance() {
    // Small-target quadratic so the per-component gradient magnitudes
    // are bounded, keeping the SPSA variance manageable. With
    // ‖∇f‖² ≈ 1.6 the per-sample stdev of any component estimate is
    // about 1.3, and N=200k brings the CI well inside 1e-2.
    let target: Vec<f64> = vec![0.5, 0.2, -0.3, 0.1];
    let params: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0];
    let f = |p: &[f64]| quadratic_loss(p, &target);

    let cfg = McgpConfig {
        n_samples: 200_000,
        epsilon: 1.0e-3,
        seed: 0xCAFE,
    };
    let mc_grad = tile_mcgp::mcgp_estimate(&f, &params, &cfg);
    let analytic: Vec<f64> = params
        .iter()
        .zip(target.iter())
        .map(|(a, b)| 2.0 * (a - b))
        .collect();

    for (g, a) in mc_grad.iter().zip(analytic.iter()) {
        assert!(
            (g - a).abs() < 1.0e-2,
            "MC grad {g} vs analytic {a} differ by {}",
            (g - a).abs()
        );
    }
}

#[test]
fn tile_mcgp_matches_finite_diff_on_smooth_loss() {
    // Same quadratic but verified against the central-FD reference
    // exposed by the module itself.
    let target: Vec<f64> = vec![1.0, 2.0, 3.0];
    let params: Vec<f64> = vec![0.0, 0.0, 0.0];
    let f = |p: &[f64]| quadratic_loss(p, &target);

    let fd = tile_mcgp::finite_diff_central(&f, &params, 1.0e-3);
    // Analytic = 2(p - target) = [-2, -4, -6].
    let expected = vec![-2.0, -4.0, -6.0];
    for (g, e) in fd.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1.0e-4);
    }
}

#[test]
fn tile_mcgp_zero_samples_returns_zero_gradient() {
    let params = vec![1.0, 2.0, 3.0];
    let cfg = McgpConfig {
        n_samples: 0,
        epsilon: 1.0e-3,
        seed: 1,
    };
    let f = |p: &[f64]| p.iter().sum();
    let grad = tile_mcgp::mcgp_estimate(&f, &params, &cfg);
    assert_eq!(grad, vec![0.0; 3]);
}

#[test]
fn tile_mcgp_seed_is_deterministic() {
    let target = vec![1.0_f64, -0.5, 0.25];
    let params = vec![0.0_f64, 0.0, 0.0];
    let cfg = McgpConfig {
        n_samples: 256,
        epsilon: 1.0e-3,
        seed: 0xABCD_1234,
    };
    let f = |p: &[f64]| quadratic_loss(p, &target);
    let g1 = tile_mcgp::mcgp_estimate(&f, &params, &cfg);
    let g2 = tile_mcgp::mcgp_estimate(&f, &params, &cfg);
    assert_eq!(g1, g2);
}
