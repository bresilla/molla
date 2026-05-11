//! Stage 38e — GPU device-wide u32 scan tests.
//!
//! Validates `ScanKernel::exclusive_scan_u32`, `inclusive_scan_u32`,
//! and `total_sum_u32` against CPU oracles. Covers correctness on
//! `n >> workgroup_size` and the edge cases (n=0, n=1, n=7,
//! n=WG_SIZE * k).
//!
//! See `radix_sort_gpu.rs` for the rationale behind the `GPU_LOCK`.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla::compute::Engine;
use molla::core::{Error, Result};
use molla::kernels::ScanKernel;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn alloc_u32_buf(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    let bytes = (len * core::mem::size_of::<u32>()) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read_u32_buf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<u32>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let bytes = (len * core::mem::size_of::<u32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scan_test staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scan_test readback enc"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let out = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    Ok(out)
}

// Cheap LCG so tests don't need a `rand` dep.
fn lcg(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 32) as u32
    }
}

fn cpu_exclusive(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc: u32 = 0;
    for &v in input {
        out.push(acc);
        acc = acc.wrapping_add(v);
    }
    out
}

fn cpu_inclusive(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc: u32 = 0;
    for &v in input {
        acc = acc.wrapping_add(v);
        out.push(acc);
    }
    out
}

fn cpu_total(input: &[u32]) -> u64 {
    let mut acc: u64 = 0;
    for &v in input {
        acc = acc.wrapping_add(v as u64);
    }
    acc
}

struct Gpu {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    kernel: ScanKernel,
}

fn make_gpu() -> Gpu {
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = ScanKernel::new(device.clone(), queue.clone());
    Gpu {
        device,
        queue,
        kernel,
    }
}

fn run_exclusive(gpu: &Gpu, input: &[u32]) -> Vec<u32> {
    let n = input.len();
    let in_buf = alloc_u32_buf(&gpu.device, "scan_test in", n.max(1));
    let out_buf = alloc_u32_buf(&gpu.device, "scan_test out", n.max(1));
    if n > 0 {
        gpu.queue
            .write_buffer(&in_buf, 0, bytemuck::cast_slice(input));
    }
    gpu.kernel
        .exclusive_scan_u32(&in_buf, &out_buf, n as u32)
        .expect("scan");
    read_u32_buf(&gpu.device, &gpu.queue, &out_buf, n).expect("read")
}

fn run_inclusive(gpu: &Gpu, input: &[u32]) -> Vec<u32> {
    let n = input.len();
    let in_buf = alloc_u32_buf(&gpu.device, "scan_test in", n.max(1));
    let out_buf = alloc_u32_buf(&gpu.device, "scan_test out", n.max(1));
    if n > 0 {
        gpu.queue
            .write_buffer(&in_buf, 0, bytemuck::cast_slice(input));
    }
    gpu.kernel
        .inclusive_scan_u32(&in_buf, &out_buf, n as u32)
        .expect("scan");
    read_u32_buf(&gpu.device, &gpu.queue, &out_buf, n).expect("read")
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn exclusive_scan_matches_cpu_oracle() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gpu = make_gpu();

    let n = 65537usize;
    let mut rng = lcg(0xC0FFEE_FEED);
    // Small values so the u32 accumulator doesn't overflow in 65k adds.
    let input: Vec<u32> = (0..n).map(|_| rng() & 0xFFFF).collect();

    let gpu_out = run_exclusive(&gpu, &input);
    let cpu_out = cpu_exclusive(&input);
    assert_eq!(gpu_out.len(), cpu_out.len());
    let mut bad = 0usize;
    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        if g != c {
            bad += 1;
            if bad < 5 {
                eprintln!("mismatch at i={i}: gpu={g} cpu={c}");
            }
        }
    }
    assert_eq!(bad, 0, "{bad} mismatches in exclusive scan over n={n}");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn inclusive_scan_matches_cpu_oracle() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gpu = make_gpu();

    let n = 65537usize;
    let mut rng = lcg(0xDEADBEEF_F00D);
    let input: Vec<u32> = (0..n).map(|_| rng() & 0xFFFF).collect();

    let gpu_out = run_inclusive(&gpu, &input);
    let cpu_out = cpu_inclusive(&input);
    assert_eq!(gpu_out, cpu_out, "inclusive scan mismatch (n={n})");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn total_sum_matches_cpu_oracle() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gpu = make_gpu();

    let n = 65537usize;
    let mut rng = lcg(0x1234_5678);
    let input: Vec<u32> = (0..n).map(|_| rng() & 0xFFFF).collect();

    let in_buf = alloc_u32_buf(&gpu.device, "scan_test sum", n);
    gpu.queue
        .write_buffer(&in_buf, 0, bytemuck::cast_slice(&input));

    let gpu_sum = gpu.kernel.total_sum_u32(&in_buf, n as u32).expect("sum");
    // u64 oracle so we can sanity-check vs the u32-wrapped sum the GPU
    // would have produced. With 16-bit inputs and 65k elements the
    // total fits in u32 easily.
    let cpu_sum = cpu_total(&input);
    assert_eq!(gpu_sum, cpu_sum, "total_sum mismatch");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn scan_handles_n_smaller_than_wg() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gpu = make_gpu();

    // n = 7.
    let input_7: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];
    let cpu_excl_7 = cpu_exclusive(&input_7);
    let gpu_excl_7 = run_exclusive(&gpu, &input_7);
    assert_eq!(gpu_excl_7, cpu_excl_7);
    let cpu_incl_7 = cpu_inclusive(&input_7);
    let gpu_incl_7 = run_inclusive(&gpu, &input_7);
    assert_eq!(gpu_incl_7, cpu_incl_7);

    // n = 1.
    let input_1: Vec<u32> = vec![42];
    assert_eq!(run_exclusive(&gpu, &input_1), vec![0]);
    assert_eq!(run_inclusive(&gpu, &input_1), vec![42]);

    // n = 0 — nothing crashes.
    let input_0: Vec<u32> = vec![];
    assert!(run_exclusive(&gpu, &input_0).is_empty());
    assert!(run_inclusive(&gpu, &input_0).is_empty());
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn scan_handles_n_multiple_of_wg() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gpu = make_gpu();

    // n = 65 536 — exact multiple of 256.
    let n = 65_536usize;
    let mut rng = lcg(0xABBA_DEAD);
    let input: Vec<u32> = (0..n).map(|_| rng() & 0xFFFF).collect();

    let gpu_excl = run_exclusive(&gpu, &input);
    let cpu_excl = cpu_exclusive(&input);
    assert_eq!(gpu_excl, cpu_excl);

    let gpu_incl = run_inclusive(&gpu, &input);
    let cpu_incl = cpu_inclusive(&input);
    assert_eq!(gpu_incl, cpu_incl);
}
