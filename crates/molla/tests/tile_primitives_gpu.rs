//! Stage 37.8 — WGSL parity tests for the tile compute primitives.
//!
//! All `#[ignore]` because they need a real wgpu adapter (CI doesn't
//! have one); run with
//! `cargo test -p molla --features gpu --test tile_primitives_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use molla::compute::tile_matmul::{self, TileMatmulKernel};
use molla::compute::Engine;

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
    fn signed(&mut self) -> f32 {
        let x = self.next() >> 11;
        let u = (x as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        (2.0 * u - 1.0) as f32
    }
}

fn make_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

// ── A — matmul ───────────────────────────────────────────────────────

#[test]
#[ignore]
fn tile_matmul_gpu_matches_cpu_256_x_128_x_256() {
    let Some(engine) = make_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TileMatmulKernel::new(device);

    let m = 256;
    let k = 128;
    let n = 256;
    let mut rng = Lcg::new(101);
    let a: Vec<f32> = (0..m * k).map(|_| rng.signed()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.signed()).collect();
    let mut c_gpu = vec![0.0_f32; m * n];
    let mut c_cpu = vec![0.0_f32; m * n];

    kernel
        .dispatch_blocking(device, queue, &a, &b, &mut c_gpu, m, n, k)
        .unwrap();
    tile_matmul::matmul_cpu(&a, &b, &mut c_cpu, m, n, k).unwrap();

    let mut max_err: f32 = 0.0;
    for (g, e) in c_gpu.iter().zip(c_cpu.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 1.0e-4, "max_err = {max_err}");
}

#[test]
#[ignore]
fn tile_matmul_gpu_handles_non_tile_aligned_shapes() {
    // M, N, K not multiples of 16 — exercises the boundary-clamping
    // branches in the WGSL kernel.
    let Some(engine) = make_engine() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TileMatmulKernel::new(device);

    let m = 17;
    let k = 23;
    let n = 31;
    let mut rng = Lcg::new(102);
    let a: Vec<f32> = (0..m * k).map(|_| rng.signed()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.signed()).collect();
    let mut c_gpu = vec![0.0_f32; m * n];
    let mut c_cpu = vec![0.0_f32; m * n];

    kernel
        .dispatch_blocking(device, queue, &a, &b, &mut c_gpu, m, n, k)
        .unwrap();
    tile_matmul::matmul_cpu(&a, &b, &mut c_cpu, m, n, k).unwrap();
    let mut max_err: f32 = 0.0;
    for (g, e) in c_gpu.iter().zip(c_cpu.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 1.0e-4, "max_err = {max_err}");
}
