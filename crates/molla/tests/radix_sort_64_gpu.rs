//! Stage 38e — GPU 64-bit pair-keyed radix sort + segmented sort.
//!
//! Validates `RadixSortKernel64::sort_pairs_u64` and
//! `segmented_sort_pairs_u64` against CPU oracles.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla::compute::Engine;
use molla::core::{Error, Result};
use molla::kernels::RadixSortKernel64;

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
        label: Some("radix64_test staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("radix64_test readback enc"),
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

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    }
}

fn split64(k: u64) -> (u32, u32) {
    (k as u32, (k >> 32) as u32)
}

fn join64(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | (lo as u64)
}

struct Gpu {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    kernel: RadixSortKernel64,
}

fn make_gpu(capacity: usize) -> Gpu {
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = RadixSortKernel64::new(device.clone(), queue.clone(), capacity).expect("kernel");
    Gpu {
        device,
        queue,
        kernel,
    }
}

/// Build three buffers with the supplied data, sort, and read back.
fn run_sort(gpu: &mut Gpu, keys: &[u64], values: &[u32]) -> (Vec<u64>, Vec<u32>) {
    assert_eq!(keys.len(), values.len());
    let n = keys.len();
    let mut lo_in = Vec::with_capacity(n);
    let mut hi_in = Vec::with_capacity(n);
    for &k in keys {
        let (lo, hi) = split64(k);
        lo_in.push(lo);
        hi_in.push(hi);
    }

    let buf_lo = alloc_u32_buf(&gpu.device, "lo", n);
    let buf_hi = alloc_u32_buf(&gpu.device, "hi", n);
    let buf_v = alloc_u32_buf(&gpu.device, "v", n);
    gpu.queue
        .write_buffer(&buf_lo, 0, bytemuck::cast_slice(&lo_in));
    gpu.queue
        .write_buffer(&buf_hi, 0, bytemuck::cast_slice(&hi_in));
    gpu.queue
        .write_buffer(&buf_v, 0, bytemuck::cast_slice(values));

    gpu.kernel
        .sort_pairs_u64(&buf_lo, &buf_hi, &buf_v, n as u32)
        .expect("sort");

    let lo_out = read_u32_buf(&gpu.device, &gpu.queue, &buf_lo, n).expect("read lo");
    let hi_out = read_u32_buf(&gpu.device, &gpu.queue, &buf_hi, n).expect("read hi");
    let v_out = read_u32_buf(&gpu.device, &gpu.queue, &buf_v, n).expect("read v");
    let k_out: Vec<u64> = lo_out
        .iter()
        .zip(hi_out.iter())
        .map(|(&l, &h)| join64(l, h))
        .collect();
    (k_out, v_out)
}

fn cpu_sort(keys: &[u64], values: &[u32]) -> (Vec<u64>, Vec<u32>) {
    let mut paired: Vec<(u64, u32)> = keys.iter().copied().zip(values.iter().copied()).collect();
    paired.sort_by_key(|(k, _)| *k);
    let kk = paired.iter().map(|p| p.0).collect();
    let vv = paired.iter().map(|p| p.1).collect();
    (kk, vv)
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn radix_sort_64_basic_random() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let n = 8192usize;
    let mut rng = lcg(0xC0DE_FACE);
    let keys: Vec<u64> = (0..n).map(|_| rng()).collect();
    let values: Vec<u32> = (0..n as u32).collect();

    let mut gpu = make_gpu(n);
    let (gpu_k, gpu_v) = run_sort(&mut gpu, &keys, &values);
    let (cpu_k, cpu_v) = cpu_sort(&keys, &values);

    assert_eq!(gpu_k.len(), cpu_k.len());
    // Verify keys are sorted ascending.
    for w in gpu_k.windows(2) {
        assert!(w[0] <= w[1], "out of order: {} > {}", w[0], w[1]);
    }
    // Each (key, value) pair must match the CPU oracle. With unique
    // random keys ties are unlikely; for ties the GPU is stable, so
    // we expect an exact match.
    assert_eq!(gpu_k, cpu_k, "keys differ");
    assert_eq!(gpu_v, cpu_v, "values differ (pairing/stability)");
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn radix_sort_64_handles_duplicate_keys() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 4 distinct keys * 1024 duplicates each → heavy stability stress.
    let n = 4096usize;
    let mut keys: Vec<u64> = Vec::with_capacity(n);
    let mut values: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let k = (i as u64 % 4) * 0x1_0000_0001;
        keys.push(k);
        values.push(i as u32);
    }
    // Shuffle deterministically: swap adjacent pairs so values aren't
    // already monotone among duplicates.
    for i in (0..n).step_by(2) {
        if i + 1 < n {
            keys.swap(i, i + 1);
            values.swap(i, i + 1);
        }
    }

    let mut gpu = make_gpu(n);
    let (gpu_k, gpu_v) = run_sort(&mut gpu, &keys, &values);
    let (cpu_k, cpu_v) = cpu_sort(&keys, &values);

    assert_eq!(gpu_k, cpu_k, "duplicate keys: keys differ");
    // Sort is stable → values must match the stable CPU sort byte-
    // for-byte.
    assert_eq!(
        gpu_v, cpu_v,
        "duplicate keys: values differ (stability broken)"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn segmented_sort_64_three_segments() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 3 segments of sizes [50, 100, 30] → n_total = 180.
    let sizes = [50usize, 100, 30];
    let offsets: Vec<u32> = {
        let mut acc = 0u32;
        let mut out = vec![0u32];
        for &s in &sizes {
            acc += s as u32;
            out.push(acc);
        }
        out
    };
    let n_total: usize = sizes.iter().sum();

    // Per-segment, build random 48-bit keys (top 16 of keys_hi reserved
    // for the segmented-sort tag).
    let mut rng = lcg(0xFEED_FACE_2D2D_BEEF);
    let mut keys: Vec<u64> = Vec::with_capacity(n_total);
    for (seg_idx, &sz) in sizes.iter().enumerate() {
        for _ in 0..sz {
            // Anchor each segment's key range somewhere distinct so it's
            // visually obvious if segments leak into each other:
            //   seg 0 → 0x00.. to 0x0F..
            //   seg 1 → 0x10.. to 0x1F..
            //   seg 2 → 0x20.. to 0x2F..
            let base = (seg_idx as u64) * 0x10_0000;
            let k = base + (rng() & 0xFFFF);
            keys.push(k);
        }
    }
    let values: Vec<u32> = (0..n_total as u32).collect();

    // Build GPU buffers.
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut kernel =
        RadixSortKernel64::new(device.clone(), queue.clone(), n_total).expect("kernel");

    let mut lo = Vec::with_capacity(n_total);
    let mut hi = Vec::with_capacity(n_total);
    for &k in &keys {
        let (l, h) = split64(k);
        lo.push(l);
        hi.push(h);
    }
    let buf_lo = alloc_u32_buf(&device, "lo", n_total);
    let buf_hi = alloc_u32_buf(&device, "hi", n_total);
    let buf_v = alloc_u32_buf(&device, "v", n_total);
    let buf_off = alloc_u32_buf(&device, "off", offsets.len());
    queue.write_buffer(&buf_lo, 0, bytemuck::cast_slice(&lo));
    queue.write_buffer(&buf_hi, 0, bytemuck::cast_slice(&hi));
    queue.write_buffer(&buf_v, 0, bytemuck::cast_slice(&values));
    queue.write_buffer(&buf_off, 0, bytemuck::cast_slice(&offsets));

    kernel
        .segmented_sort_pairs_u64(
            &buf_lo,
            &buf_hi,
            &buf_v,
            &buf_off,
            sizes.len() as u32,
            n_total as u32,
        )
        .expect("segmented sort");

    let lo_out = read_u32_buf(&device, &queue, &buf_lo, n_total).expect("read lo");
    let hi_out = read_u32_buf(&device, &queue, &buf_hi, n_total).expect("read hi");
    let v_out = read_u32_buf(&device, &queue, &buf_v, n_total).expect("read v");
    let k_out: Vec<u64> = lo_out
        .iter()
        .zip(hi_out.iter())
        .map(|(&l, &h)| join64(l, h))
        .collect();

    // Per-segment CPU oracle.
    for (seg_idx, &sz) in sizes.iter().enumerate() {
        let start = offsets[seg_idx] as usize;
        let end = offsets[seg_idx + 1] as usize;
        assert_eq!(end - start, sz);
        let seg_keys = &keys[start..end];
        let seg_values = &values[start..end];
        let (exp_k, exp_v) = cpu_sort(seg_keys, seg_values);

        let got_k = &k_out[start..end];
        let got_v = &v_out[start..end];
        assert_eq!(got_k, &exp_k[..], "segment {seg_idx} keys differ");
        assert_eq!(got_v, &exp_v[..], "segment {seg_idx} values differ");
    }

    // Sanity: segments should NOT have leaked into each other —
    // anchor ranges enforce that.
    for (seg_idx, &sz) in sizes.iter().enumerate() {
        let start = offsets[seg_idx] as usize;
        let end = offsets[seg_idx + 1] as usize;
        let lo_anchor = (seg_idx as u64) * 0x10_0000;
        let hi_anchor = lo_anchor + 0xFFFF;
        for &k in &k_out[start..end] {
            assert!(
                k >= lo_anchor && k <= hi_anchor,
                "segment {seg_idx} ({sz} items): key {k:#x} outside [{lo_anchor:#x}, {hi_anchor:#x}]"
            );
        }
    }
}
