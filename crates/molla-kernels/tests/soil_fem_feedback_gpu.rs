#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::{
    PreparedTireFemTerrainFeedback, TireFemContactFeedbackGpuTerrain, TireFemContactFeedbackInput,
    TireFemContactFeedbackKernel, TireFemContactFeedbackOutput, TireFemContactFeedbackParams,
};

const BUDGET: u64 = 64 * 1024 * 1024;

fn maps(engine: &Engine) -> [wgpu::Buffer; 5] {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    std::array::from_fn(|i| {
        let data = match i {
            0 => vec![0.0, 0.05, 0.10, 0.01, 0.06, 0.11, 0.02, 0.07, 0.12],
            1 => vec![0.6; 9],
            2 => vec![0.003; 9],
            3 => vec![0.02; 9],
            _ => [0.2, 0.0, -0.1, 0.0].repeat(9),
        };
        let b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FEM feedback fixture terrain"),
            size: data.len() as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b, 0, bytemuck::cast_slice::<f32, u8>(&data));
        b
    })
}

fn terrain(maps: &[wgpu::Buffer; 5], mask: u32) -> TireFemContactFeedbackGpuTerrain<'_> {
    TireFemContactFeedbackGpuTerrain {
        origin: [0.0, 0.0],
        spacing: 1.0,
        nx: 3,
        nz: 3,
        heights: &maps[0],
        friction: 0.9,
        friction_map: (mask & 1 != 0).then_some(&maps[1]),
        water_film_map: (mask & 2 != 0).then_some(&maps[2]),
        sinkage_map: (mask & 4 != 0).then_some(&maps[3]),
        surface_velocity_map: (mask & 8 != 0).then_some(&maps[4]),
        soil_rolling_resistance_scale: 1.0,
        surface_velocity: [0.1, 0.0, 0.0],
    }
}

fn input(n: usize, t: usize) -> TireFemContactFeedbackInput {
    TireFemContactFeedbackInput {
        positions: (0..n)
            .map(|i| {
                [
                    0.2 + (i % 3) as f32 * 0.4,
                    if i % 4 == 3 { 0.9 } else { 0.015 },
                    0.3,
                    0.0,
                ]
            })
            .collect(),
        velocities: vec![[0.3, -0.1, 0.05, 0.0]; n],
        masses: vec![2.0; n],
        tread_ids: (0..t as u32).rev().collect(),
        bristles: vec![[0.002, 0.0, -0.001, 0.0]; t],
        terrain: None,
        params: TireFemContactFeedbackParams {
            dt: 1.0 / 240.0,
            contact_band: 0.05,
            omega: 3.0,
            normal_load_per_tread: 650.0,
            area_per_tread: 0.01,
            tread_shear_stiffness: 4e5,
            mu_static: 1.0,
            mu_dynamic: 0.8,
            bristle_damping: 5e3,
            bristle_relax_rate: 8.0,
            ..Default::default()
        },
    }
}

fn legacy(
    engine: &Engine,
    kernel: &TireFemContactFeedbackKernel,
    input: &TireFemContactFeedbackInput,
    terrain: &TireFemContactFeedbackGpuTerrain<'_>,
) -> TireFemContactFeedbackOutput {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let output = kernel
        .dispatch_resident_with_gpu_terrain(device, queue, input, terrain)
        .unwrap();
    kernel
        .readback_resident_feedback_output(device, queue, &output)
        .unwrap()
}

fn exact(a: &TireFemContactFeedbackOutput, b: &TireFemContactFeedbackOutput) {
    for (a, b) in [
        (&a.velocities, &b.velocities),
        (&a.forces, &b.forces),
        (&a.bristles, &b.bristles),
    ] {
        assert!(a.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(
            bytemuck::cast_slice::<_, u32>(a),
            bytemuck::cast_slice::<_, u32>(b)
        );
    }
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn retained_feedback_all_maps_counts_and_evolving_history_are_exact() {
    let engine = Engine::new_gpu().unwrap();
    let kernel = TireFemContactFeedbackKernel::new(engine.device().unwrap());
    let maps = maps(&engine);
    for mask in 0..16 {
        let mut terrain = terrain(&maps, mask);
        let resources = DeviceScope::start().unwrap();
        let mut prepared =
            PreparedTireFemTerrainFeedback::new(&engine, &kernel, 129, 129, &terrain, BUDGET)
                .unwrap();
        assert_eq!(resources.finish().observations, 11);
        assert_eq!(prepared.buffer_bytes(), 152 * 129 + 208);
        for n in [129, 1, 63, 64, 65] {
            for t in [0, n] {
                let mut state = input(n, t);
                for step in 0..3 {
                    terrain.friction = 0.7 + step as f32 * 0.1;
                    state.params.omega += 0.2;
                    engine.queue().unwrap().write_buffer(
                        &maps[0],
                        0,
                        bytemuck::cast_slice(&[step as f32 * 0.01; 9]),
                    );
                    let expected = legacy(&engine, &kernel, &state, &terrain);
                    let resources = DeviceScope::start().unwrap();
                    let transfers = TransferScope::start().unwrap();
                    let dispatch = DispatchScope::start().unwrap();
                    let actual = prepared.run_blocking(&state, &terrain).unwrap();
                    assert_eq!(resources.finish().observations, 0);
                    let transfers = transfers.finish();
                    assert!(!transfers.overflowed);
                    assert_eq!(transfers.upload_bytes, (36 * n + 20 * t + 176) as u64);
                    assert_eq!(transfers.readback_bytes, (32 * n + 16 * t) as u64);
                    assert_eq!(dispatch.finish().encoded_dispatches, 1);
                    exact(&actual, &expected);
                    if t > 0 {
                        assert!(actual.forces.iter().flatten().any(|v| v.abs() > 0.01));
                    }
                    state.velocities = actual.velocities;
                    state.bristles = actual.bristles;
                }
            }
        }
    }
    println!(
        "480 exact comparisons; all optional maps, changing sizes/terrain/history; 11 prepared buffers, zero hot buffer allocations; exact transfer counts"
    );
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn retained_feedback_rejects_before_upload_and_survives_kernel_drop() {
    let engine = Engine::new_gpu().unwrap();
    let kernel = TireFemContactFeedbackKernel::new(engine.device().unwrap());
    let maps = maps(&engine);
    let terrain = terrain(&maps, 15);
    let bytes = 152 * 4 + 208;
    let resources = DeviceScope::start().unwrap();
    assert!(
        PreparedTireFemTerrainFeedback::new(&engine, &kernel, 4, 4, &terrain, bytes - 1).is_err()
    );
    assert_eq!(resources.finish().observations, 0);
    let mut prepared =
        PreparedTireFemTerrainFeedback::new(&engine, &kernel, 4, 4, &terrain, bytes).unwrap();
    let good = input(4, 4);
    let expected = legacy(&engine, &kernel, &good, &terrain);
    for case in 0..9 {
        let mut bad = input(4, 4);
        match case {
            0 => bad.tread_ids[1] = bad.tread_ids[0],
            1 => bad.tread_ids[0] = 4,
            2 => bad.velocities.clear(),
            3 => bad.bristles.clear(),
            4 => bad.masses[0] = -1.0,
            5 => bad.positions[0][0] = f32::NAN,
            6 => bad.params.dt = 0.0,
            7 => bad.params.omega = f32::INFINITY,
            _ => bad.positions.clear(),
        }
        let transfers = TransferScope::start().unwrap();
        assert!(prepared.run_blocking(&bad, &terrain).is_err());
        assert_eq!(transfers.finish().upload_bytes, 0);
        exact(&expected, &prepared.run_blocking(&good, &terrain).unwrap());
    }
    let replacement = self::maps(&engine);
    let other_terrain = self::terrain(&replacement, 15);
    assert!(!prepared.matches(&engine, &kernel, 4, 4, &other_terrain));
    let transfers = TransferScope::start().unwrap();
    assert!(prepared.run_blocking(&good, &other_terrain).is_err());
    assert_eq!(transfers.finish().upload_bytes, 0);
    let cloned = maps.clone();
    assert!(prepared.matches(&engine, &kernel, 4, 4, &self::terrain(&cloned, 15)));
    let new_kernel = TireFemContactFeedbackKernel::new(engine.device().unwrap());
    assert!(!prepared.matches(&engine, &new_kernel, 4, 4, &terrain));
    assert!(!prepared.matches(&engine, &kernel, 5, 4, &terrain));
    let foreign = Engine::new_gpu().unwrap();
    let resources = DeviceScope::start().unwrap();
    assert!(PreparedTireFemTerrainFeedback::new(&foreign, &kernel, 4, 4, &terrain, bytes).is_err());
    assert_eq!(resources.finish().observations, 0);
    assert!(!prepared.matches(&foreign, &kernel, 4, 4, &terrain));
    drop(kernel);
    exact(&expected, &prepared.run_blocking(&good, &terrain).unwrap());
    engine.destroy_device().unwrap();
    let transfers = TransferScope::start().unwrap();
    assert!(prepared.run_blocking(&good, &terrain).is_err());
    assert_eq!(transfers.finish().upload_bytes, 0);
}

#[test]
#[ignore = "requires a real GPU adapter; paired component timing"]
fn retained_feedback_paired_benchmark() {
    let engine = Engine::new_gpu().unwrap();
    let kernel = TireFemContactFeedbackKernel::new(engine.device().unwrap());
    let maps = maps(&engine);
    let terrain = terrain(&maps, 15);
    for n in [129, 2048] {
        let state = input(n, n / 4);
        let mut prepared =
            PreparedTireFemTerrainFeedback::new(&engine, &kernel, n, n / 4, &terrain, BUDGET)
                .unwrap();
        let mut timings = [Vec::new(), Vec::new()];
        for pair in 0..8 {
            let mut outputs = [None, None];
            for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = std::time::Instant::now();
                let output = if arm == 0 {
                    legacy(&engine, &kernel, &state, &terrain)
                } else {
                    prepared.run_blocking(&state, &terrain).unwrap()
                };
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                if pair >= 2 {
                    timings[arm].push(ms);
                    println!(
                        "fem_feedback_sample nodes={n} treads={} pair={pair} retained={} ms={ms:.6}",
                        n / 4,
                        arm == 1
                    );
                }
                outputs[arm] = Some(output);
            }
            exact(outputs[0].as_ref().unwrap(), outputs[1].as_ref().unwrap());
        }
        for times in &mut timings {
            times.sort_by(f64::total_cmp);
        }
        println!(
            "fem_feedback_median nodes={n} legacy_ms={:.6} retained_ms={:.6} samples=6 physical_trajectory=false",
            (timings[0][2] + timings[0][3]) * 0.5,
            (timings[1][2] + timings[1][3]) * 0.5
        );
    }
}
