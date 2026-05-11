//! Stage 38c — GPU/CPU parity for [`molla::optim`].
//!
//! Marked `#[ignore]` because they need a real wgpu adapter — run with
//! `nixVulkan cargo test -p molla --test optim_gpu_parity -- --ignored`.
//!
//! Each test runs 50 update steps with random gradients (deterministic
//! LCG seed) and asserts the GPU and CPU trajectories agree to 1e-5.
//! The tolerance budget reflects f32 vs f64 round-off — Adam's `√v̂`
//! and the running EMA accumulate single-precision error every step.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla::compute::Engine;
use molla::kernels::{AdamGpuKernel, SgdGpuKernel};
use molla::math::Real;
use molla::optim::{AdamConfig, AdamState, SgdConfig, SgdState};

static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Same LCG used by `optim_convergence` — keeps the test deterministic
/// without pulling rand into the dev-deps.
struct Lcg(u32);
impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
    }
}

fn upload_storage(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> wgpu::Buffer {
    let bytes = (data.len().max(1) * 4) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("optim_gpu_parity::storage"),
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

fn write_storage(queue: &wgpu::Queue, buf: &wgpu::Buffer, data: &[f32]) {
    queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
}

#[test]
#[ignore = "needs a real wgpu adapter — run via cargo test --test optim_gpu_parity -- --ignored"]
fn adam_gpu_matches_cpu() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let n = 1024usize;
    let mut rng = Lcg(0xADAA_DEAD);
    let theta0_f32: Vec<f32> = (0..n).map(|_| rng.next_unit()).collect();
    let theta0_f64: Vec<Real> = theta0_f32.iter().map(|&v| v as Real).collect();

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = AdamGpuKernel::new(device.clone(), queue.clone());

    // Buffers — theta starts at theta0, m / v at zero.
    let theta_buf = upload_storage(&device, &queue, &theta0_f32);
    let grad_buf = upload_storage(&device, &queue, &vec![0.0f32; n]);
    let m_buf = upload_storage(&device, &queue, &vec![0.0f32; n]);
    let v_buf = upload_storage(&device, &queue, &vec![0.0f32; n]);

    let mut theta_cpu = theta0_f64.clone();
    let mut state_cpu = AdamState::new(n);
    let cfg = AdamConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };

    let mut grad_cpu = vec![0.0_f64; n];
    for step_ix in 1..=50u32 {
        // Cook a fresh gradient vector — random in [-1, 1] for f32.
        let grad_f32: Vec<f32> = (0..n).map(|_| rng.next_unit()).collect();
        for i in 0..n {
            grad_cpu[i] = grad_f32[i] as Real;
        }
        write_storage(&queue, &grad_buf, &grad_f32);

        // GPU step.
        kernel
            .step(&theta_buf, &grad_buf, &m_buf, &v_buf, n as u32, &cfg, step_ix)
            .expect("adam gpu step");

        // CPU step — mirror the same gradient.
        state_cpu.step(&mut theta_cpu, &grad_cpu, &cfg);
    }

    let theta_gpu = kernel
        .read_buffer(&theta_buf, n as u32)
        .expect("readback theta");

    let mut max_abs: f64 = 0.0;
    for i in 0..n {
        let err = (theta_gpu[i] as f64 - theta_cpu[i]).abs();
        if err > max_abs {
            max_abs = err;
        }
    }
    eprintln!(
        "adam_gpu_matches_cpu: n = {n}, steps = 50, max |GPU − CPU| = {max_abs:.3e}",
    );
    assert!(
        max_abs < 1.0e-5,
        "Adam GPU/CPU parity violation: max_abs = {max_abs}",
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via cargo test --test optim_gpu_parity -- --ignored"]
fn sgd_gpu_matches_cpu() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let n = 1024usize;
    let mut rng = Lcg(0x5EED_BEEF);
    let theta0_f32: Vec<f32> = (0..n).map(|_| rng.next_unit()).collect();
    let theta0_f64: Vec<Real> = theta0_f32.iter().map(|&v| v as Real).collect();

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = SgdGpuKernel::new(device.clone(), queue.clone());

    let theta_buf = upload_storage(&device, &queue, &theta0_f32);
    let grad_buf = upload_storage(&device, &queue, &vec![0.0f32; n]);
    let buf_buf = upload_storage(&device, &queue, &vec![0.0f32; n]);

    let mut theta_cpu = theta0_f64.clone();
    let mut state_cpu = SgdState::new(n);
    let cfg = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 1.0e-4,
        nesterov: true,
    };

    let mut grad_cpu = vec![0.0_f64; n];
    for step_ix in 0..50u32 {
        let grad_f32: Vec<f32> = (0..n).map(|_| rng.next_unit()).collect();
        for i in 0..n {
            grad_cpu[i] = grad_f32[i] as Real;
        }
        write_storage(&queue, &grad_buf, &grad_f32);

        kernel
            .step(&theta_buf, &grad_buf, &buf_buf, n as u32, &cfg, step_ix)
            .expect("sgd gpu step");
        state_cpu.step(&mut theta_cpu, &grad_cpu, &cfg);
    }

    let theta_gpu = kernel
        .read_buffer(&theta_buf, n as u32)
        .expect("readback theta");

    let mut max_abs: f64 = 0.0;
    for i in 0..n {
        let err = (theta_gpu[i] as f64 - theta_cpu[i]).abs();
        if err > max_abs {
            max_abs = err;
        }
    }
    eprintln!(
        "sgd_gpu_matches_cpu: n = {n}, steps = 50, max |GPU − CPU| = {max_abs:.3e}",
    );
    assert!(
        max_abs < 1.0e-5,
        "SGD GPU/CPU parity violation: max_abs = {max_abs}",
    );
}
