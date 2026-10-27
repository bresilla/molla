//! Stage 19.1 — GPU radix-sort tests.
//!
//! Validates the multi-pass 8-bit radix sort over (key=u32, value=u32)
//! pairs against `Vec::sort_by_key` references. Covers correctness on
//! random + edge-case inputs and stability via paired duplicates.
//!
//! Each `#[test]` calls `run_sort_check` which spins up its own
//! `Engine::new_gpu()` and then polls the device with
//! `wait_indefinitely`. When the cargo `--workspace -- --ignored`
//! lane runs all seven concurrently inside this binary, the
//! polling threads contend on the wgpu queue and either hang or
//! crash. `GPU_LOCK` serialises adapter ownership.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_compute_primitives::RadixSortKernel;
use molla_core::{Error, Result};

/// Serialises wgpu adapter lifetime across the tests in this binary.
/// See module docstring.
static GPU_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "needs a real GPU adapter"]
fn parallel_retained_sort_matches_stable_reference_across_scan_tiles() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    println!("adapter={:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = RadixSortKernel::new(device.clone(), queue.clone(), 256).unwrap();
    for n in [
        1, 31, 32, 33, 255, 256, 257, 65_535, 65_536, 65_537, 100_003,
    ] {
        let keys = alloc_u32_buf(device, "parallel keys", n + 7);
        let values = alloc_u32_buf(device, "parallel values", n + 7);
        let sort = kernel.prepare_parallel(&keys, &values, n).unwrap();
        assert_eq!(sort.resource_counts(), [7, 6, 4]);
        assert_eq!(sort.dispatch_count(), 16);
        assert_eq!(
            sort.scratch_bytes(),
            (n * 8 + n.div_ceil(256) * 1024 + 1088) as u64
        );
        for pattern in 0..5 {
            let mut random = lcg(n as u64 + 7);
            let input: Vec<u32> = (0..n)
                .map(|i| match pattern {
                    0 => random(),
                    1 => u32::MAX,
                    2 => (random() & 255) << 24,
                    3 => (n - i) as u32,
                    _ => (i as u32 * 73) % 257,
                })
                .collect();
            let payload: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(0x9e3779b9)).collect();
            let expected = cpu_reference(&input, &payload);
            let mut padded_keys = input.clone();
            padded_keys.extend([0x13579bdf; 7]);
            let mut padded_values = payload.clone();
            padded_values.extend([0xfdb97531; 7]);
            queue.write_buffer(&keys, 0, bytemuck::cast_slice(&padded_keys));
            queue.write_buffer(&values, 0, bytemuck::cast_slice(&padded_values));
            let mut encoder = device.create_command_encoder(&Default::default());
            sort.encode(&mut encoder);
            queue.submit([encoder.finish()]);
            let actual_keys = read_u32_buf(device, queue, &keys, n + 7).unwrap();
            let actual_values = read_u32_buf(device, queue, &values, n + 7).unwrap();
            assert_eq!(
                &actual_keys[..n],
                expected.0,
                "keys n={n} pattern={pattern}"
            );
            assert_eq!(
                &actual_values[..n],
                expected.1,
                "values n={n} pattern={pattern}"
            );
            assert_eq!(&actual_keys[n..], &[0x13579bdf; 7]);
            assert_eq!(&actual_values[n..], &[0xfdb97531; 7]);
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn parallel_retained_sort_queued_lifecycle_has_no_hot_allocations_or_transfers() {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = RadixSortKernel::new(device.clone(), queue.clone(), 256).unwrap();
    let buffers: Vec<_> = [513, 100_003]
        .into_iter()
        .map(|n| {
            (
                n,
                alloc_u32_buf(device, "independent keys", n),
                alloc_u32_buf(device, "independent values", n),
            )
        })
        .collect();
    let sorts: Vec<_> = buffers
        .iter()
        .map(|(n, k, v)| kernel.prepare_parallel(k, v, *n).unwrap())
        .collect();
    let invalid_scope = DeviceScope::start().unwrap();
    assert!(
        kernel
            .prepare_parallel(&buffers[0].1, &buffers[0].2, 0)
            .is_err()
    );
    assert!(
        kernel
            .prepare_parallel(&buffers[0].1, &buffers[0].1, 1)
            .is_err()
    );
    assert!(
        kernel
            .prepare_parallel(&buffers[0].1, &buffers[0].2, 514)
            .is_err()
    );
    assert!(
        kernel
            .prepare_parallel(&buffers[0].1, &buffers[0].2, usize::MAX)
            .is_err()
    );
    assert_eq!(invalid_scope.finish().observations, 0);
    drop(kernel);
    let output = alloc_u32_buf(device, "queued snapshots", 2 * (513 + 100_003) * 3);
    let mut expected_output = Vec::new();
    let mut offset = 0;
    for round in 0..3 {
        for (index, (n, keys, values)) in buffers.iter().enumerate() {
            let input: Vec<_> = (0..*n as u32).map(|i| (i * 97 + round) % 131).collect();
            let payload: Vec<_> = (0..*n as u32).rev().collect();
            let expected = cpu_reference(&input, &payload);
            expected_output.extend(expected.0);
            expected_output.extend(expected.1);
            queue.write_buffer(keys, 0, bytemuck::cast_slice(&input));
            queue.write_buffer(values, 0, bytemuck::cast_slice(&payload));
            let resources = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let dispatches = DispatchScope::start().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            sorts[index].encode(&mut encoder);
            encoder.copy_buffer_to_buffer(keys, 0, &output, offset, *n as u64 * 4);
            offset += *n as u64 * 4;
            encoder.copy_buffer_to_buffer(values, 0, &output, offset, *n as u64 * 4);
            offset += *n as u64 * 4;
            queue.submit([encoder.finish()]);
            let resources = resources.finish();
            let transfers = transfers.finish();
            let dispatches = dispatches.finish();
            assert_eq!(resources.observations, 0);
            assert!(!resources.invalid && !transfers.overflowed && !dispatches.overflowed);
            assert_eq!(transfers.upload_bytes + transfers.readback_bytes, 0);
            assert_eq!(dispatches.encoded_dispatches, 16);
        }
    }
    drop(sorts);
    drop(buffers);
    assert_eq!(
        read_u32_buf(device, queue, &output, expected_output.len()).unwrap(),
        expected_output
    );
}

#[test]
#[ignore = "needs a real GPU adapter; paired diagnostic benchmark"]
fn parallel_retained_sort_paired_benchmark() {
    use std::time::Instant;
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    println!("adapter={:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = RadixSortKernel::new(device.clone(), queue.clone(), 256).unwrap();
    for n in [513, 50_000, 100_003] {
        let keys = alloc_u32_buf(device, "paired keys", n);
        let values = alloc_u32_buf(device, "paired values", n);
        let sorts = [
            kernel.prepare(&keys, &values, n).unwrap(),
            kernel.prepare_parallel(&keys, &values, n).unwrap(),
        ];
        let mut random = lcg(31);
        let input: Vec<_> = (0..n).map(|_| random() % 4096).collect();
        let payload: Vec<_> = (0..n as u32).collect();
        let expected = cpu_reference(&input, &payload);
        for round in 0..8 {
            for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                queue.write_buffer(&keys, 0, bytemuck::cast_slice(&input));
                queue.write_buffer(&values, 0, bytemuck::cast_slice(&payload));
                queue.submit([]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let start = Instant::now();
                let mut encoder = device.create_command_encoder(&Default::default());
                sorts[mode].encode(&mut encoder);
                queue.submit([encoder.finish()]);
                let encode_ms = start.elapsed().as_secs_f64() * 1000.;
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let wall_ms = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(read_u32_buf(device, queue, &keys, n).unwrap(), expected.0);
                assert_eq!(read_u32_buf(device, queue, &values, n).unwrap(), expected.1);
                if round >= 2 {
                    println!(
                        "RADIX_SAMPLE n={n} round={} mode={} encode_ms={encode_ms:.6} wall_ms={wall_ms:.6} scratch_bytes={} parity=exact performance_accepted=false",
                        round - 2,
                        ["reference", "parallel"][mode],
                        sorts[mode].scratch_bytes()
                    );
                }
            }
        }
    }
}

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
    // Hold the GPU lock for the lifetime of the engine + dispatch.
    // See module docstring for why.
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn retained_sorts_keep_independent_uniforms_and_scratch_across_queued_updates() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut kernel = RadixSortKernel::new(device.clone(), queue.clone(), 1).unwrap();
    let sizes = [1, 257, 513];
    let buffers: Vec<_> = sizes
        .iter()
        .map(|&n| {
            (
                alloc_u32_buf(device, "retained keys", n),
                alloc_u32_buf(device, "retained values", n),
            )
        })
        .collect();
    let prepared: Vec<_> = buffers
        .iter()
        .zip(sizes)
        .map(|((k, v), n)| kernel.prepare(k, v, n).unwrap())
        .collect();
    assert!(prepared.iter().all(|p| p.resource_counts() == [7, 6, 3]));
    assert!(kernel.prepare(&buffers[0].0, &buffers[0].1, 0).is_err());
    assert!(kernel.prepare(&buffers[0].0, &buffers[0].0, 1).is_err());
    assert!(
        kernel
            .prepare(&buffers[0].0, &buffers[0].1, usize::MAX)
            .is_err()
    );
    let grow_keys = alloc_u32_buf(device, "grow keys", 4096);
    let grow_values = alloc_u32_buf(device, "grow values", 4096);
    kernel.dispatch(&grow_keys, &grow_values, 4096).unwrap();
    drop(kernel);
    let stride = sizes.iter().sum::<usize>() * 2;
    let output = alloc_u32_buf(device, "retained snapshots", stride * 20);
    let mut reference = Vec::new();
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut offset = 0_u64;
    for step in 0..20 {
        for (which, &n) in sizes.iter().enumerate() {
            let keys: Vec<_> = (0..n)
                .map(|i| {
                    if step % 3 == 0 {
                        u32::MAX
                    } else {
                        ((i * 31 + step * 19) % 17) as u32 * 0x01000001
                    }
                })
                .collect();
            let values: Vec<_> = (0..n as u32).collect();
            let sources = [
                alloc_u32_buf(device, "source keys", n),
                alloc_u32_buf(device, "source values", n),
            ];
            queue.write_buffer(&sources[0], 0, bytemuck::cast_slice(&keys));
            queue.write_buffer(&sources[1], 0, bytemuck::cast_slice(&values));
            encoder.copy_buffer_to_buffer(&sources[0], 0, &buffers[which].0, 0, n as u64 * 4);
            encoder.copy_buffer_to_buffer(&sources[1], 0, &buffers[which].1, 0, n as u64 * 4);
            prepared[which].encode(&mut encoder);
            for b in [&buffers[which].0, &buffers[which].1] {
                encoder.copy_buffer_to_buffer(b, 0, &output, offset, n as u64 * 4);
                offset += n as u64 * 4;
            }
            let (k, v) = cpu_reference(&keys, &values);
            reference.extend(k);
            reference.extend(v);
        }
    }
    queue.submit([encoder.finish()]);
    assert_eq!(
        read_u32_buf(device, queue, &output, reference.len()).unwrap(),
        reference
    );
}

// Cheap LCG PRNG so tests don't need a `rand` dep.
fn lcg(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
            gpu_v[i],
            2 * k,
            "stability: 2k should come first at key={k}"
        );
        assert_eq!(
            gpu_v[i + 1],
            2 * k + 1,
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
