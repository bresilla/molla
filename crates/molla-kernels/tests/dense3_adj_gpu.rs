//! Stage 38b — WGSL parity for `adj_svd_kernel`.
//!
//! `#[ignore]` because it needs a real wgpu adapter (CI doesn't); run
//! with:
//!
//! ```text
//! cargo test -p molla-kernels --features gpu --test dense3_adj_gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::Dense3Kernel;
use molla_math::dense3::{adj_svd3, svd3};
use molla_math::{Mat3, Real, Vec3};

const BATCH: usize = 64;
const PARITY_TOL: Real = 1.0e-3;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x00C0_FFEE_F00D)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn uniform(&mut self) -> Real {
        let x = self.next() >> 11;
        (x as Real + 1.0) / ((1u64 << 53) as Real + 1.0)
    }
    fn signed(&mut self) -> Real {
        2.0 * self.uniform() - 1.0
    }
    fn rand_mat(&mut self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.signed(), self.signed(), self.signed()),
            Vec3::new(self.signed(), self.signed(), self.signed()),
            Vec3::new(self.signed(), self.signed(), self.signed()),
        )
    }
    fn rand_rotation(&mut self) -> Mat3 {
        let ax = Vec3::new(self.signed(), self.signed(), self.signed()).normalize();
        let theta = std::f64::consts::PI * self.uniform();
        Mat3::from_axis_angle(ax, theta)
    }
    fn rand_wellcond(&mut self) -> Mat3 {
        let q1 = self.rand_rotation();
        let q2 = self.rand_rotation();
        let s = Vec3::new(
            0.5 + 1.0 * self.uniform(),
            1.0 + 1.0 * self.uniform(),
            2.0 + 0.5 * self.uniform(),
        );
        q1 * Mat3::from_diagonal(s) * q2
    }
}

fn frob(a: Mat3) -> Real {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let v = a.col(j)[i];
            s += v * v;
        }
    }
    s.sqrt()
}

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn adj_svd_gpu_matches_cpu() {
    let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = Dense3Kernel::new(&device);

    let mut rng = Lcg::new(0xA38B_BEEF);
    let mats: Vec<Mat3> = (0..BATCH).map(|_| rng.rand_wellcond()).collect();
    // Run forward CPU SVD so the GPU adjoint sees the same (U, σ, V).
    // (Going through the GPU forward would couple two error sources.)
    let mut u_cpu = Vec::with_capacity(BATCH);
    let mut sigma_cpu = Vec::with_capacity(BATCH);
    let mut v_cpu = Vec::with_capacity(BATCH);
    for m in &mats {
        let (u, s, v) = svd3(*m);
        u_cpu.push(u);
        sigma_cpu.push(s);
        v_cpu.push(v);
    }

    // Random upstream gradients.
    let adj_u: Vec<Mat3> = (0..BATCH).map(|_| rng.rand_mat()).collect();
    let adj_sigma: Vec<Vec3> = (0..BATCH)
        .map(|_| Vec3::new(rng.signed(), rng.signed(), rng.signed()))
        .collect();
    let adj_v: Vec<Mat3> = (0..BATCH).map(|_| rng.rand_mat()).collect();

    // CPU reference.
    let adj_a_cpu: Vec<Mat3> = (0..BATCH)
        .map(|i| {
            adj_svd3(
                mats[i],
                u_cpu[i],
                sigma_cpu[i],
                v_cpu[i],
                adj_u[i],
                adj_sigma[i],
                adj_v[i],
            )
        })
        .collect();

    // GPU.
    let adj_a_gpu = kernel
        .adj_svd3_dispatch_blocking(
            &device, &queue, &mats, &u_cpu, &sigma_cpu, &v_cpu, &adj_u, &adj_sigma, &adj_v,
        )
        .expect("adj_svd dispatch");

    let mut max_abs = 0.0_f64;
    for i in 0..BATCH {
        let d = frob(adj_a_gpu[i] - adj_a_cpu[i]);
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < PARITY_TOL * (1.0 + frob(adj_a_cpu[i])),
            "matrix {i}: |adj_A_gpu - adj_A_cpu|_F = {d}, cpu_norm = {}",
            frob(adj_a_cpu[i]),
        );
    }
    eprintln!("adj_svd GPU parity max abs error over {BATCH} matrices: {max_abs:e}");
}
