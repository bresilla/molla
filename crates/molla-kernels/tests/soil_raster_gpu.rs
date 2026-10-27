#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireDemGpuKernel, TireDemResidentState, TireDemResidentTerrainOutput,
    TireDemResidentTerrainParams, TireDemResidentTerrainReadback, TireMpmGpuKernel,
    TireMpmTerrainRasterParams,
};
use std::time::Instant;

fn params(particles: u32, nx: u32, nz: u32) -> TireMpmTerrainRasterParams {
    TireMpmTerrainRasterParams {
        particle_count: particles,
        origin: [-2.0, -3.0],
        spacing: 0.125,
        nx,
        nz,
        rest_height: 1.25,
    }
}

fn cloud(params: &TireMpmTerrainRasterParams, crowded: bool) -> [Vec<[f32; 4]>; 3] {
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut compaction = Vec::new();
    for i in 0..params.particle_count {
        let cell = i.wrapping_mul(2654435761) % (params.nx * params.nz);
        let x = if crowded {
            0.0
        } else {
            (cell % params.nx) as f32 - 1.0
        };
        let z = if crowded {
            0.0
        } else {
            (cell / params.nx) as f32 + 0.5
        };
        positions.push([
            params.origin[0] + params.spacing * x,
            0.0,
            params.origin[1] + params.spacing * z,
            0.0,
        ]);
        velocities.push([
            (i % 17) as f32 * 0.25 - 2.0,
            (i % 13) as f32 * 0.125,
            -((i % 11) as f32) * 0.25,
            0.0,
        ]);
        compaction.push([(i % 23) as f32 * 0.003 - 0.01, 0.0, 0.0, 0.0]);
    }
    [positions, velocities, compaction]
}

fn oracle(
    params: &TireMpmTerrainRasterParams,
    data: &[Vec<[f32; 4]>; 3],
    fallback: [f32; 3],
) -> (Vec<f32>, Vec<[f64; 3]>) {
    let cells = (params.nx * params.nz) as usize;
    let mut compact = vec![0.0f32; cells];
    let mut velocity = vec![[0.0f64; 3]; cells];
    let mut counts = vec![0u32; cells];
    for (i, p) in data[0].iter().enumerate() {
        let x = (((p[0] - params.origin[0]) / params.spacing.max(1e-6)).floor() as i32)
            .clamp(0, params.nx as i32 - 1) as usize;
        let z = (((p[2] - params.origin[1]) / params.spacing.max(1e-6)).floor() as i32)
            .clamp(0, params.nz as i32 - 1) as usize;
        let cell = z * params.nx as usize + x;
        compact[cell] = compact[cell].max(data[2][i][0]);
        for axis in 0..3 {
            velocity[cell][axis] += f64::from(data[1][i][axis]);
        }
        counts[cell] += 1;
    }
    for (i, v) in velocity.iter_mut().enumerate() {
        for axis in 0..3 {
            v[axis] = if counts[i] == 0 {
                f64::from(fallback[axis])
            } else {
                v[axis] / f64::from(counts[i])
            };
        }
    }
    (compact, velocity)
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_mpm_binned_raster_matches_cpu_oracle() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireMpmGpuKernel::new(device);
    let fallback = [0.5, -0.25, 0.75];
    for (p, nx, nz, crowded) in [
        (0, 1, 1, false),
        (1, 1, 1, false),
        (63, 7, 9, false),
        (65, 13, 5, false),
        (4097, 65, 65, false),
        (65537, 3, 5, true),
        (129, 257, 257, false),
    ] {
        let params = params(p, nx, nz);
        let data = cloud(&params, crowded);
        let expected = oracle(&params, &data, fallback);
        let got = kernel
            .dispatch_raster_blocking(
                device, queue, &data[0], &data[1], &data[2], &params, fallback,
            )
            .expect("binned raster");
        for i in 0..expected.0.len() {
            assert!(
                (got.sinkage[i] - expected.0[i]).abs() <= 1e-6,
                "sinkage case {p}/{nx}/{nz} cell {i}"
            );
            assert!((got.heights[i] - (params.rest_height - expected.0[i])).abs() <= 1e-6);
            for axis in 0..3 {
                assert!(
                    (f64::from(got.surface_velocity_map[i][axis]) - expected.1[i][axis]).abs()
                        <= 2e-5,
                    "velocity case {p}/{nx}/{nz} cell {i} axis {axis}"
                );
            }
        }
    }
}

fn buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("soil raster test"),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }
    )
}

fn dem_cloud(p: TireDemResidentTerrainParams, n: u32, crowded: bool) -> [Vec<[f32; 4]>; 2] {
    let mut rows = [Vec::new(), Vec::new()];
    for i in 0..n {
        let cell = if crowded {
            0
        } else {
            i.wrapping_mul(2654435761) % (p.nx * p.nz)
        };
        let spacing = p.spacing.max(1e-9);
        let mut x = p.origin[0] + (cell % p.nx) as f32 * spacing;
        let mut z = p.origin[1] + (cell / p.nx) as f32 * spacing;
        match i % 5 {
            0 => {
                x = x.next_down();
                z = z.next_up();
            }
            1 => {
                x = x.next_up();
                z = z.next_down();
            }
            2 => {
                x += spacing * 0.5;
                z += spacing * 0.5;
            }
            3 => {
                x += spacing;
                z += spacing;
            }
            _ => {}
        }
        if crowded {
            x = p.origin[0] + spacing * 0.5;
            z = p.origin[1] + spacing * 0.5;
        }
        if x.is_subnormal() {
            x = 0.0;
        }
        if z.is_subnormal() {
            z = 0.0;
        }
        rows[0].push([x, (i % 23) as f32 * 0.03125 - 0.25, z, 0.0]);
        rows[1].push([
            (i % 17) as f32 * 0.25 - 2.0,
            (i % 11) as f32 * 0.125,
            -((i % 13) as f32) * 0.25,
            0.0,
        ]);
    }
    rows
}

fn dem_oracle(
    p: TireDemResidentTerrainParams,
    data: &[Vec<[f32; 4]>; 2],
    reference: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<[f64; 4]>) {
    let spacing = p.spacing.max(1e-9);
    let radius = p.grain_radius.max(1e-9);
    let mut heights = Vec::new();
    let mut sinkage = Vec::new();
    let mut rows = Vec::new();
    for (cell, reference) in reference.iter().enumerate() {
        let x0 = ((cell as u32 % p.nx) as f32).mul_add(spacing, p.origin[0]);
        let z0 = ((cell as u32 / p.nx) as f32).mul_add(spacing, p.origin[1]);
        let inside =
            |v: &[f32; 4]| v[0] >= x0 && v[0] < x0 + spacing && v[2] >= z0 && v[2] < z0 + spacing;
        let mut height = p.floor_y;
        let mut top = -f32::MAX;
        for v in &data[0] {
            if inside(v) {
                height = height.max(v[1] + radius);
                top = top.max(v[1]);
            }
        }
        let band = (radius * 2.25).max(spacing * 0.25);
        let mut row = [0.0; 4];
        if top > -1e30 {
            for (v, velocity) in data[0].iter().zip(&data[1]) {
                if inside(v) && v[1] >= top - band {
                    for axis in 0..3 {
                        row[axis] += f64::from(velocity[axis]);
                    }
                    row[3] += 1.0;
                }
            }
        }
        if row[3] > 0.0 {
            for axis in 0..3 {
                row[axis] /= row[3];
            }
        }
        heights.push(height);
        sinkage.push((reference - height).max(0.0));
        rows.push(row);
    }
    (heights, sinkage, rows)
}

fn assert_dem_raster(
    actual: &TireDemResidentTerrainReadback,
    expected: &(Vec<f32>, Vec<f32>, Vec<[f64; 4]>),
) {
    assert_eq!(actual.heights.len(), expected.0.len());
    for cell in 0..expected.0.len() {
        assert!(
            (actual.heights[cell] - expected.0[cell]).abs() <= 1e-6,
            "height cell {cell}: {} != {}",
            actual.heights[cell],
            expected.0[cell]
        );
        assert!(
            (actual.sinkage[cell] - expected.1[cell]).abs() <= 1e-6,
            "sinkage cell {cell}"
        );
        let row = actual.surface_velocity_map.rows[cell];
        assert_eq!(
            f64::from(row[3]),
            expected.2[cell][3],
            "surface count cell {cell}"
        );
        for axis in 0..3 {
            assert!(
                (f64::from(row[axis]) - expected.2[cell][axis]).abs() <= 2e-5,
                "velocity cell {cell} axis {axis}"
            );
        }
    }
}

fn dem_state(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[Vec<[f32; 4]>; 2],
) -> TireDemResidentState {
    let state = TireDemResidentState {
        positions: buffer(device, data[0].len() as u64 * 16),
        velocities: buffer(device, data[1].len() as u64 * 16),
        grain_count: data[0].len(),
    };
    if !data[0].is_empty() {
        queue.write_buffer(&state.positions, 0, bytemuck::cast_slice(&data[0]));
        queue.write_buffer(&state.velocities, 0, bytemuck::cast_slice(&data[1]));
    }
    state
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_dem_binned_raster_matches_half_open_oracle_and_legacy() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    for (n, nx, nz, crowded, origin, spacing) in [
        (0, 1, 1, false, [-2.0, -3.0], 0.125),
        (1, 1, 1, true, [-2.0, -3.0], 0.125),
        (63, 7, 9, false, [-2.0, -3.0], 0.125),
        (65, 13, 5, false, [-0.7, -1.1], 0.1),
        (4097, 65, 65, false, [-0.7, -1.1], 0.1),
        (65537, 3, 5, true, [-2.0, -3.0], 0.125),
        (129, 257, 257, false, [-2.0, -3.0], 0.125),
        (129, 13, 5, false, [1000000.0, -1000000.0], 0.1),
        (129, 13, 5, false, [1000000.0, -1000000.0], 0.01),
        (65, 13, 5, false, [0.0, 0.0], 1e-10),
    ] {
        let p = TireDemResidentTerrainParams {
            origin,
            spacing,
            nx,
            nz,
            floor_y: -0.125,
            grain_radius: 0.025,
        };
        let data = dem_cloud(p, n, crowded);
        let reference: Vec<_> = (0..nx * nz).map(|i| 0.5 + (i % 3) as f32 * 0.125).collect();
        let expected = dem_oracle(p, &data, &reference);
        let state = dem_state(device, queue, &data);
        eprintln!(
            "DEM oracle case particles={n} nx={nx} nz={nz} origin={origin:?} spacing={spacing}"
        );
        let binned = kernel
            .dispatch_resident_terrain_raster(device, queue, &state, &reference, p)
            .unwrap();
        let legacy = kernel
            .dispatch_resident_terrain_raster_reference(device, queue, &state, &reference, p)
            .unwrap();
        assert_dem_raster(
            &kernel
                .readback_resident_terrain(device, queue, &legacy)
                .unwrap(),
            &expected,
        );
        assert_dem_raster(
            &kernel
                .readback_resident_terrain(device, queue, &binned)
                .unwrap(),
            &expected,
        );
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_dem_surface_band_includes_threshold_and_excludes_deeper_grains() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    for spacing in [0.125f32, 0.5] {
        let p = TireDemResidentTerrainParams {
            origin: [0.0, 0.0],
            spacing,
            nx: 2,
            nz: 1,
            floor_y: -0.125,
            grain_radius: 0.025,
        };
        let threshold = 0.5 - (p.grain_radius * 2.25).max(spacing * 0.25);
        let mut data = [Vec::new(), Vec::new()];
        for (i, y) in [0.5, threshold, threshold.next_down(), threshold.next_up()]
            .into_iter()
            .enumerate()
        {
            data[0].push([spacing * 0.5, y, spacing * 0.5, 0.0]);
            data[1].push([i as f32, -(i as f32), i as f32 * 2.0, 0.0]);
        }
        let reference = [0.25, 0.75];
        let expected = dem_oracle(p, &data, &reference);
        assert_eq!(expected.2[0][3], 3.0);
        assert_eq!(expected.2[1], [0.0; 4]);
        let state = dem_state(device, queue, &data);
        for legacy in [false, true] {
            let terrain = if legacy {
                kernel.dispatch_resident_terrain_raster_reference(
                    device, queue, &state, &reference, p,
                )
            } else {
                kernel.dispatch_resident_terrain_raster(device, queue, &state, &reference, p)
            }
            .unwrap();
            assert_dem_raster(
                &kernel
                    .readback_resident_terrain(device, queue, &terrain)
                    .unwrap(),
                &expected,
            );
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_dem_prepared_raster_refreshes_bins_and_reference() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let p = TireDemResidentTerrainParams {
        origin: [-0.7, -1.1],
        spacing: 0.1,
        nx: 13,
        nz: 5,
        floor_y: -0.125,
        grain_radius: 0.025,
    };
    let mut data = dem_cloud(p, 4097, false);
    let state = dem_state(device, queue, &data);
    let cells = u64::from(p.nx * p.nz);
    let reference = buffer(device, cells * 4);
    let mut values = vec![0.5; cells as usize];
    let mut terrain = TireDemResidentTerrainOutput {
        heights: buffer(device, cells * 4),
        sinkage: buffer(device, cells * 4),
        surface_velocity_rows: buffer(device, cells * 16),
        params: p,
    };
    let prepare = |terrain: &TireDemResidentTerrainOutput| {
        kernel.prepare_resident_terrain_raster(device, &state, &reference, terrain)
    };
    terrain.params.nx = u32::MAX;
    assert!(prepare(&terrain).is_err());
    terrain.params = p;
    terrain.params.grain_radius = f32::NAN;
    assert!(prepare(&terrain).is_err());
    terrain.params = p;
    let invalid_state = TireDemResidentState {
        positions: state.positions.clone(),
        velocities: state.velocities.clone(),
        grain_count: usize::MAX,
    };
    assert!(
        kernel
            .prepare_resident_terrain_raster(device, &invalid_state, &reference, &terrain)
            .is_err()
    );
    assert!(
        kernel
            .prepare_resident_terrain_raster(device, &state, &buffer(device, 16), &terrain)
            .is_err()
    );
    let prepared = prepare(&terrain).unwrap();
    for iteration in 0..3 {
        if iteration == 1 {
            for pos in &mut data[0] {
                pos[0] = 1000.0;
            }
            values.fill(0.75);
        } else if iteration == 2 {
            data = dem_cloud(p, 4097, true);
        }
        queue.write_buffer(&state.positions, 0, bytemuck::cast_slice(&data[0]));
        queue.write_buffer(&reference, 0, bytemuck::cast_slice(&values));
        let expected = dem_oracle(p, &data, &values);
        for exhaustive in [true, false] {
            let mut encoder = device.create_command_encoder(&Default::default());
            for _ in 0..4 {
                if exhaustive {
                    prepared.encode_reference(&mut encoder);
                } else {
                    prepared.encode(&mut encoder);
                }
            }
            queue.submit([encoder.finish()]);
            assert_dem_raster(
                &kernel
                    .readback_resident_terrain(device, queue, &terrain)
                    .unwrap(),
                &expected,
            );
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_mpm_prepared_raster_reuses_scratch_and_rejects_invalid_inputs() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireMpmGpuKernel::new(device);
    let params = params(4097, 65, 65);
    let cells = u64::from(params.nx * params.nz);
    let data = cloud(&params, false);
    let inputs =
        std::array::from_fn::<_, 3, _>(|_| buffer(device, u64::from(params.particle_count) * 16));
    let outputs = [
        buffer(device, cells * 4),
        buffer(device, cells * 4),
        buffer(device, cells * 16),
    ];
    for (b, values) in inputs.iter().zip(&data) {
        queue.write_buffer(b, 0, bytemuck::cast_slice(values));
    }
    let prepare = |p: &TireMpmTerrainRasterParams, fallback| {
        kernel.prepare_raster_to_buffers(
            device,
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &outputs[0],
            &outputs[1],
            &outputs[2],
            p,
            fallback,
        )
    };
    assert!(prepare(&params, [f32::NAN, 0.0, 0.0]).is_err());
    assert!(
        prepare(
            &TireMpmTerrainRasterParams {
                nx: u32::MAX,
                nz: 2,
                ..params
            },
            [0.0; 3]
        )
        .is_err()
    );
    assert!(
        prepare(
            &TireMpmTerrainRasterParams {
                particle_count: 999999,
                ..params
            },
            [0.0; 3]
        )
        .is_err()
    );
    let prepared = prepare(&params, [0.0; 3]).unwrap();
    let readback = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("soil raster reset readback"),
            size: cells * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    prepared.encode_reference(&mut encoder);
    encoder.copy_buffer_to_buffer(&outputs[1], 0, &readback, 0, cells * 4);
    for _ in 0..4 {
        prepared.encode(&mut encoder);
    }
    encoder.copy_buffer_to_buffer(&outputs[1], 0, &readback, cells * 4, cells * 4);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&mapped);
    assert_eq!(&floats[..cells as usize], &floats[cells as usize..]);
    drop(mapped);
    readback.unmap();
}

fn raster_workloads() -> [(u32, u32, bool); 7] {
    [
        (10_000, 128, false),
        (50_000, 128, false),
        (100_000, 128, false),
        (50_000, 32, false),
        (50_000, 64, false),
        (50_000, 256, false),
        (50_000, 128, true),
    ]
}

fn raster_samples(case: &str, particles: u32, side: u32, crowded: bool, run: impl Fn(bool)) {
    run(true);
    run(false);
    let mut samples = [Vec::new(), Vec::new()];
    for iteration in 0..12 {
        for j in 0..2 {
            let mode = (iteration + j) % 2;
            let start = Instant::now();
            run(mode == 0);
            samples[mode].push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    for (name, times) in ["reference", "binned"].iter().zip(&mut samples) {
        assert!(times.iter().all(|t| t.is_finite() && *t > 0.0));
        eprintln!(
            "{{\"schema\":1,\"case\":\"{case}\",\"mode\":\"{name}\",\"particles\":{particles},\"nx\":{side},\"nz\":{side},\"crowded\":{crowded},\"raw_ms\":{times:?}}}"
        );
        times.sort_by(f64::total_cmp);
        eprintln!(
            "{{\"schema\":1,\"case\":\"{case}\",\"mode\":\"{name}\",\"particles\":{particles},\"nx\":{side},\"nz\":{side},\"crowded\":{crowded},\"samples\":12,\"p50_ms\":{},\"p95_ms\":{},\"scope\":\"export-only\"}}",
            times[6], times[11]
        );
    }
}

#[test]
#[ignore = "bounded GPU raster performance measurement"]
fn soil_mpm_raster_benchmark() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let adapter = engine.adapter_info().expect("GPU adapter identity");
    assert!(!adapter.is_software, "hardware GPU required for timings");
    eprintln!("adapter={adapter:?}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireMpmGpuKernel::new(device);
    for (particles, side, crowded) in raster_workloads() {
        let params = params(particles, side, side);
        eprintln!(
            "{{\"schema\":1,\"case\":\"mpm-raster\",\"scope\":\"export-only\",\"nx\":{side},\"nz\":{side},\"spacing_m\":0.125,\"origin_m\":[-2,-3],\"rest_height_m\":1.25,\"particles\":{particles},\"distribution\":\"multiplicative-hash-2654435761\",\"crowded\":{crowded},\"simulated_seconds\":null,\"paired_order\":\"alternating\",\"warmup_per_mode\":1}}"
        );
        let data = cloud(&params, crowded);
        let inputs = std::array::from_fn::<_, 3, _>(|_| buffer(device, u64::from(particles) * 16));
        let cells = u64::from(side * side);
        let outputs = [
            buffer(device, cells * 4),
            buffer(device, cells * 4),
            buffer(device, cells * 16),
        ];
        for (b, values) in inputs.iter().zip(&data) {
            queue.write_buffer(b, 0, bytemuck::cast_slice(values));
        }
        let prepared = kernel
            .prepare_raster_to_buffers(
                device,
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &outputs[0],
                &outputs[1],
                &outputs[2],
                &params,
                [0.0; 3],
            )
            .unwrap();
        raster_samples("mpm-raster", particles, side, crowded, |reference| {
            let mut encoder = device.create_command_encoder(&Default::default());
            if reference {
                prepared.encode_reference(&mut encoder);
            } else {
                prepared.encode(&mut encoder);
            }
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        });
    }
}

#[test]
#[ignore = "bounded GPU raster performance measurement"]
fn soil_dem_raster_benchmark() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let adapter = engine.adapter_info().expect("GPU adapter identity");
    assert!(!adapter.is_software, "hardware GPU required for timings");
    eprintln!("adapter={adapter:?}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    for (particles, side, crowded) in raster_workloads() {
        let p = TireDemResidentTerrainParams {
            origin: [-2.0, -3.0],
            spacing: 0.125,
            nx: side,
            nz: side,
            floor_y: -0.125,
            grain_radius: 0.025,
        };
        eprintln!(
            "{{\"schema\":1,\"case\":\"dem-raster\",\"scope\":\"export-only\",\"nx\":{side},\"nz\":{side},\"spacing_m\":0.125,\"origin_m\":[-2,-3],\"floor_y_m\":-0.125,\"grain_radius_m\":0.025,\"reference_height_m\":0.75,\"particles\":{particles},\"distribution\":\"multiplicative-hash-2654435761\",\"crowded\":{crowded},\"simulated_seconds\":null,\"paired_order\":\"alternating\",\"warmup_per_mode\":1}}"
        );
        let data = dem_cloud(p, particles, crowded);
        let state = dem_state(device, queue, &data);
        let cells = u64::from(side * side);
        let reference = buffer(device, cells * 4);
        queue.write_buffer(
            &reference,
            0,
            bytemuck::cast_slice(&vec![0.75f32; cells as usize]),
        );
        let terrain = TireDemResidentTerrainOutput {
            heights: buffer(device, cells * 4),
            sinkage: buffer(device, cells * 4),
            surface_velocity_rows: buffer(device, cells * 16),
            params: p,
        };
        let prepared = kernel
            .prepare_resident_terrain_raster(device, &state, &reference, &terrain)
            .unwrap();
        raster_samples("dem-raster", particles, side, crowded, |reference| {
            let mut encoder = device.create_command_encoder(&Default::default());
            if reference {
                prepared.encode_reference(&mut encoder);
            } else {
                prepared.encode(&mut encoder);
            }
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        });
    }
}
