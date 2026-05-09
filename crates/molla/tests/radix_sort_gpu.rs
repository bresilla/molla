//! Stage 19.1 — GPU radix-sort tests.
//!
//! Validates the multi-pass 8-bit radix sort over (key=u32, value=u32)
//! pairs against `Vec::sort_by_key` references. Covers correctness on
//! random + edge-case inputs and stability via paired duplicates.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::core::{Error, Result};
use molla::kernels::RadixSortKernel;

/// Allocate a STORAGE | COPY_SRC | COPY_DST u32 buffer of `len` slots.
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

/// Read `len` u32s back from `buf`.
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
        label: Some("radix_sort_test staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("radix_sort_test readback enc"),
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

fn run_sort_check(keys_in: Vec<u32>, values_in: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(keys_in.len(), values_in.len());
    let n = keys_in.len();
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let keys = alloc_u32_buf(&device, "keys", n);
    let values = alloc_u32_buf(&device, "values", n);
    queue.write_buffer(&keys, 0, bytemuck::cast_slice(&keys_in));
    queue.write_buffer(&values, 0, bytemuck::cast_slice(&values_in));

    let mut kernel = RadixSortKernel::new(device.clone(), queue.clone(), n).expect("kernel");
    kernel.dispatch(&keys, &values, n).expect("dispatch");

    let sorted_keys = read_u32_buf(&device, &queue, &keys, n).expect("read keys");
    let sorted_values = read_u32_buf(&device, &queue, &values, n).expect("read values");
    (sorted_keys, sorted_values)
}

/// Reference: stable sort of (key, value) pairs by key.
fn cpu_reference(keys: &[u32], values: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut paired: Vec<(u32, u32)> = keys.iter().copied().zip(values.iter().copied()).collect();
    paired.sort_by_key(|(k, _)| *k);
    let out_k = paired.iter().map(|(k, _)| *k).collect();
    let out_v = paired.iter().map(|(_, v)| *v).collect();
    (out_k, out_v)
}

// Cheap LCG PRNG so tests don't need a `rand` dep.
fn lcg(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 32) as u32
    }
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn radix_sort_random_1k_matches_reference() {
    let n = 1000;
    let mut rng = lcg(0xC0FFEE);
    let keys_in: Vec<u32> = (0..n).map(|_| rng()).collect();
    // Use the original index as the value so we can spot-check stability.
    let values_in: Vec<u32> = (0..n as u32).collect();

    let (gpu_k, gpu_v) = run_sort_check(keys_in.clone(), values_in.clone());
    let (cpu_k, cpu_v) = cpu_reference(&keys_in, &values_in);

    assert_eq!(gpu_k, cpu_k, "1k-key sort: keys differ");
    assert_eq!(gpu_v, cpu_v, "1k-key sort: values differ (stability)");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn radix_sort_random_100k_matches_reference() {
    let n = 100_000;
    let mut rng = lcg(0xBADC0FFEE0DDF00D);
    let keys_in: Vec<u32> = (0..n).map(|_| rng()).collect();
    let values_in: Vec<u32> = (0..n as u32).collect();

    let (gpu_k, gpu_v) = run_sort_check(keys_in.clone(), values_in.clone());
    let (cpu_k, cpu_v) = cpu_reference(&keys_in, &values_in);

    assert_eq!(gpu_k.len(), cpu_k.len(), "length mismatch");
    assert_eq!(gpu_k, cpu_k, "100k-key sort: keys differ");
    assert_eq!(gpu_v, cpu_v, "100k-key sort: values differ (stability)");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn radix_sort_all_equal_keys_preserves_order() {
    let n = 4096;
    let keys_in: Vec<u32> = vec![0xDEADBEEF; n];
    let values_in: Vec<u32> = (0..n as u32).collect();

    let (gpu_k, gpu_v) = run_sort_check(keys_in.clone(), values_in.clone());
    assert_eq!(gpu_k, keys_in, "all-equal keys should be unchanged");
    assert_eq!(
        gpu_v, values_in,
        "stable sort: values must keep input order"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn radix_sort_already_sorted_input_is_idempotent() {
    let n = 8192;
    let keys_in: Vec<u32> = (0..n as u32).collect();
    let values_in: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(7)).collect();

    let (gpu_k, gpu_v) = run_sort_check(keys_in.clone(), values_in.clone());
    assert_eq!(gpu_k, keys_in);
    assert_eq!(gpu_v, values_in);
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn radix_sort_reverse_sorted_input_works() {
    let n = 8192;
    let keys_in: Vec<u32> = (0..n as u32).rev().collect();
    let values_in: Vec<u32> = (0..n as u32).collect();

    let (gpu_k, gpu_v) = run_sort_check(keys_in.clone(), values_in.clone());
    let (cpu_k, cpu_v) = cpu_reference(&keys_in, &values_in);
    assert_eq!(gpu_k, cpu_k);
    assert_eq!(gpu_v, cpu_v);
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn radix_sort_paired_duplicates_are_stable() {
    // For each key value k in [0, 256), insert two entries with
    // values (2k, 2k+1) — the sort must keep 2k before 2k+1.
    let mut keys_in: Vec<u32> = Vec::with_capacity(512);
    let mut values_in: Vec<u32> = Vec::with_capacity(512);
    // Interleave so the duplicates aren't already adjacent.
    for k in 0..256u32 {
        keys_in.push(k);
        values_in.push(2 * k);
    }
    for k in 0..256u32 {
        keys_in.push(k);
        values_in.push(2 * k + 1);
    }

    let (gpu_k, gpu_v) = run_sort_check(keys_in, values_in);
    // After stable sort the (k, 2k) entry must precede (k, 2k+1) for
    // every k.
    for k in 0..256u32 {
        let i = (2 * k) as usize;
        assert_eq!(gpu_k[i], k);
        assert_eq!(gpu_k[i + 1], k);
        assert_eq!(
            gpu_v[i], 2 * k,
            "stability: 2k should come first at key={k}"
        );
        assert_eq!(
            gpu_v[i + 1], 2 * k + 1,
            "stability: 2k+1 should come second at key={k}"
        );
    }
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn radix_sort_high_byte_only_keys() {
    // Keys that only differ in the high byte exercise pass 3 (shift=24)
    // — the previous three passes are no-ops.
    let mut rng = lcg(0xFEEDFACE);
    let n = 2048;
    let keys_in: Vec<u32> = (0..n).map(|_| (rng() & 0xFF) << 24).collect();
    let values_in: Vec<u32> = (0..n as u32).collect();

    let (gpu_k, gpu_v) = run_sort_check(keys_in.clone(), values_in.clone());
    let (cpu_k, cpu_v) = cpu_reference(&keys_in, &values_in);
    assert_eq!(gpu_k, cpu_k);
    assert_eq!(gpu_v, cpu_v);
}
