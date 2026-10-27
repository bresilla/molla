use super::*;
use molla_kernels::BrushWheelOutput;
use molla_vehicle::{RealTireContact, terrain::HeightfieldTerrain};

fn assert_parity(gpu: &BrushWheelOutput, cpu: &RealTireContact) {
    let values = [
        ("normal load", gpu.forces[1], cpu.normal_load),
        ("longitudinal", gpu.forces[3], cpu.longitudinal_force),
        ("lateral", gpu.forces[2], cpu.lateral_force),
        ("moment", gpu.diag2[1], cpu.aligning_moment_z),
        ("budget", gpu.diag4[2], cpu.available_friction_budget),
        ("consumed", gpu.diag4[3], cpu.consumed_friction_budget),
        ("world x", gpu.world_force[0], cpu.force[0]),
        ("world y", gpu.world_force[1], cpu.force[1]),
        ("world z", gpu.world_force[2], cpu.force[2]),
    ];
    for (label, actual, expected) in values {
        assert!(
            (actual as f64 - expected).abs() <= 0.02 + expected.abs() * 0.002,
            "{label}: GPU {actual}, CPU {expected}"
        );
    }
    assert_eq!(gpu.diag1[3] as usize, cpu.cells.len());
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_width_resolves_split_grip_and_reverses_yaw() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireBrushGpuKernel::new(device);
    let vehicle_kernel = molla_kernels::TireVehicleYawGpuKernel::new(device);
    let n = 41;
    let spacing = 0.025_f32;
    let heights = vec![0.0_f32; n * n];
    let material = SurfaceMaterial {
        rolling_resistance: 0.02,
        ..SurfaceMaterial::dry_asphalt()
    };
    for heading in [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]] {
        for grip_side in [-1.0, 1.0] {
            let friction: Vec<f32> = (0..n * n)
                .map(|i| {
                    let x = -0.5 + (i % n) as f64 * spacing as f64;
                    let z = -0.5 + (i / n) as f64 * spacing as f64;
                    if grip_side * (z * heading[0] - x * heading[2]) > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let terrain = HeightfieldTerrain::with_friction_map(
                [-0.5, -0.5],
                spacing as f64,
                n,
                n,
                heights.iter().map(|&v| v as f64).collect(),
                1.0,
                friction.iter().map(|&v| v as f64).collect(),
            );
            let gpu_terrain = BrushTerrain {
                origin: [-0.5, -0.5],
                spacing,
                nx: n as u32,
                nz: n as u32,
                heights: &heights,
                friction: 1.0,
                friction_map: Some(&friction),
            };
            for pressure_pa in [80_000.0, 300_000.0] {
                for direction in [-1.0, 1.0] {
                    let params = RealTireParams {
                        width_segments: 4,
                        segments: 96,
                        pressure_pa,
                        ..Default::default()
                    };
                    let state = RealWheelState {
                        hub_pos: [0.0, params.radius * 0.91, 0.0],
                        hub_vel: heading.map(|v| v * direction * 2.0),
                        heading,
                        omega: direction * 4.0 / params.radius,
                    };
                    let cpu = evaluate_brush_tire(&params, &state, &terrain, &material);
                    assert!(cpu.aligning_moment_z * grip_side * direction > 1.0);
                    let gpu = kernel
                        .dispatch_blocking(
                            device,
                            queue,
                            &[pack_input(&params, &state, &material)],
                            Some(&gpu_terrain),
                        )
                        .unwrap();
                    assert!(
                        gpu[0].diag2[1] as f64 * grip_side * direction > 1.0,
                        "GPU ignored asymmetric shoulder grip: {:?}",
                        gpu[0]
                    );
                    assert_parity(&gpu[0], &cpu);
                    let row = pack_input(&params, &state, &material);
                    let vehicle = molla_kernels::TireVehicleBrushRowsGpuInput {
                        chassis_pose: [
                            0.0,
                            state.hub_pos[1] as f32,
                            0.0,
                            heading[2].atan2(heading[0]) as f32,
                        ],
                        chassis_velocity: row.hub_vel,
                        wheel_mounts: vec![[0.0, 0.0, 0.0, params.radius as f32]],
                        wheel_state: vec![[
                            state.omega as f32,
                            0.0,
                            params.width as f32,
                            pressure_pa as f32,
                        ]],
                        width_segments: vec![params.width_segments as u32],
                        tire_rows: vec![row.tire],
                        material_rows: vec![row.material],
                        extra_rows: vec![[material.cohesion as f32, 1.0, 1.0, 0.0]],
                        suspension_rows: vec![[
                            (params.radius - state.hub_pos[1]) as f32,
                            0.0,
                            0.0,
                            1.0,
                        ]],
                    };
                    let rows = vehicle_kernel
                        .dispatch_brush_rows_resident(device, queue, &vehicle)
                        .unwrap();
                    let output = kernel
                        .dispatch_input_buffer_resident(
                            device,
                            queue,
                            &rows.buffer,
                            1,
                            Some(&gpu_terrain),
                        )
                        .unwrap();
                    let resident = kernel
                        .readback_resident_output(device, queue, &output)
                        .unwrap();
                    assert_parity(&resident[0], &cpu);
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_width_refinement_preserves_uniform_forces_and_counts_cells() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireBrushGpuKernel::new(device);
    let terrain = FlatTerrain::new(0.0, 1.0);
    let material = SurfaceMaterial::dry_asphalt();
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for width_segments in [1, 2, 3, 4, 8, 16] {
        for segments in [8, 48, 96, 256] {
            for height_fraction in [0.91, 1.1] {
                let params = RealTireParams {
                    segments,
                    width_segments,
                    ..Default::default()
                };
                let state = RealWheelState {
                    hub_pos: [0.0, params.radius * height_fraction, 0.0],
                    hub_vel: [2.0, 0.0, 0.15],
                    heading: [1.0, 0.0, 0.0],
                    omega: 3.0 / params.radius,
                };
                inputs.push(pack_input(&params, &state, &material));
                expected.push(evaluate_brush_tire(&params, &state, &terrain, &material));
            }
        }
    }
    let gpu = kernel
        .dispatch_blocking(device, queue, &inputs, None)
        .unwrap();
    for (actual, expected) in gpu.iter().zip(&expected) {
        assert_parity(actual, expected);
    }
    let mut legacy = inputs[16];
    assert_eq!(legacy.hub_pos[3], 3.0);
    legacy.hub_pos[3] = 0.0;
    let actual = kernel
        .dispatch_blocking(device, queue, &[legacy], None)
        .unwrap();
    assert_parity(&actual[0], &expected[16]);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_width_samples_shoulder_height_and_rolling_moment() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireBrushGpuKernel::new(device);
    let n = 41;
    let spacing = 0.025_f32;
    let heights: Vec<f32> = (0..n * n)
        .map(|i| {
            let z = -0.5 + (i / n) as f32 * spacing;
            if z > 0.025 {
                ((z - 0.025) / 0.05).min(1.0) * 0.02
            } else if z < -0.025 {
                ((-z - 0.025) / 0.05).min(1.0) * -0.02
            } else {
                0.0
            }
        })
        .collect();
    let terrain = HeightfieldTerrain::new(
        [-0.5, -0.5],
        spacing as f64,
        n,
        n,
        heights.iter().map(|&v| v as f64).collect(),
        1.0,
    );
    let gpu_terrain = BrushTerrain {
        origin: [-0.5, -0.5],
        spacing,
        nx: n as u32,
        nz: n as u32,
        heights: &heights,
        friction: 1.0,
        friction_map: None,
    };
    let material = SurfaceMaterial {
        rolling_resistance: 0.1,
        ..SurfaceMaterial::new(0.0, 0.0)
    };
    for width_segments in [4, 16] {
        for direction in [-1.0, 1.0] {
            let params = RealTireParams {
                width: 0.24,
                width_segments,
                segments: 96,
                ..Default::default()
            };
            let state = RealWheelState {
                hub_pos: [0.0, params.radius * 0.91, 0.0],
                hub_vel: [2.0 * direction, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: 2.0 * direction / params.radius,
            };
            let cpu = evaluate_brush_tire(&params, &state, &terrain, &material);
            assert!(cpu.aligning_moment_z * direction < -1.0);
            let gpu = kernel
                .dispatch_blocking(
                    device,
                    queue,
                    &[pack_input(&params, &state, &material)],
                    Some(&gpu_terrain),
                )
                .unwrap();
            assert_parity(&gpu[0], &cpu);
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_width_rejects_invalid_host_rows() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireBrushGpuKernel::new(device);
    let params = RealTireParams::default();
    let state = RealWheelState {
        hub_pos: [0.0, params.radius * 0.91, 0.0],
        hub_vel: [2.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 3.0 / params.radius,
    };
    let material = SurfaceMaterial::dry_asphalt();
    let row = pack_input(&params, &state, &material);
    for invalid in [-1.0, 0.25, 17.0, f32::NAN, f32::INFINITY] {
        let mut bad = row;
        bad.hub_pos[3] = invalid;
        assert!(
            kernel
                .dispatch_blocking(device, queue, &[row, bad], None)
                .is_err()
        );
    }
    let gpu = kernel
        .dispatch_blocking(device, queue, &[row], None)
        .unwrap();
    assert_parity(
        &gpu[0],
        &evaluate_brush_tire(&params, &state, &FlatTerrain::new(0.0, 1.0), &material),
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_stopped_brush_has_no_directional_rolling_resistance() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireBrushGpuKernel::new(device);
    let params = RealTireParams::default();
    let material = SurfaceMaterial::dry_asphalt();
    for zero in [0.0, -0.0] {
        let state = RealWheelState {
            hub_pos: [0.0, params.radius * 0.91, 0.0],
            hub_vel: [zero, zero, zero],
            heading: [1.0, 0.0, 0.0],
            omega: zero,
        };
        let cpu = evaluate_brush_tire(&params, &state, &FlatTerrain::new(0.0, 1.0), &material);
        assert_eq!(cpu.longitudinal_force, 0.0);
        let gpu = kernel
            .dispatch_blocking(
                device,
                queue,
                &[pack_input(&params, &state, &material)],
                None,
            )
            .unwrap();
        assert_eq!(gpu[0].forces[3], 0.0);
        assert_parity(&gpu[0], &cpu);
    }
}
