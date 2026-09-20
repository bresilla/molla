//! Stage (i) — GPU brush-tire smoke / parity test.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{BrushTerrain, BrushWheelInput, TireBrushGpuKernel};
use molla_vehicle::{
    RealTireParams, RealWheelOrientation, RealWheelState, SurfaceMaterial, evaluate_brush_tire,
    evaluate_brush_tire_oriented, terrain::FlatTerrain,
};

#[path = "tire_brush_gpu/width.rs"]
mod width;

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn engine_or_skip(test_name: &str) -> Option<Engine> {
    let engine = try_engine();
    if engine.is_none() {
        eprintln!("skipping {test_name}: GPU adapter unavailable");
    }
    engine
}

fn pack_input(
    params: &RealTireParams,
    state: &RealWheelState,
    material: &SurfaceMaterial,
) -> BrushWheelInput {
    pack_input_with_camber(params, state, material, 0.0)
}

fn pack_input_with_camber(
    params: &RealTireParams,
    state: &RealWheelState,
    material: &SurfaceMaterial,
    camber_angle_rad: f64,
) -> BrushWheelInput {
    BrushWheelInput {
        hub_pos: [
            state.hub_pos[0] as f32,
            state.hub_pos[1] as f32,
            state.hub_pos[2] as f32,
            params.width_segments as f32,
        ],
        hub_vel: [
            state.hub_vel[0] as f32,
            state.hub_vel[1] as f32,
            state.hub_vel[2] as f32,
            0.0,
        ],
        heading: [
            state.heading[0] as f32,
            state.heading[1] as f32,
            state.heading[2] as f32,
            params.structural_lateral_bias_per_normal as f32,
        ],
        spin: [
            state.omega as f32,
            params.radius as f32,
            params.width as f32,
            params.pressure_pa as f32,
        ],
        tire: [
            params.segments as f32,
            params.carcass_normal_stiffness as f32,
            params.tread_shear_stiffness as f32,
            params.relaxation_speed as f32,
        ],
        material: [
            material.mu_static as f32,
            material.mu_dynamic as f32,
            material.bristle_damping as f32,
            material.rolling_resistance as f32,
        ],
        extra: [material.cohesion as f32, 0.0, 0.0, camber_angle_rad as f32],
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_brush_gpu_matches_cpu_on_flat_terrain() {
    let Some(engine) = engine_or_skip("tire_brush_gpu_matches_cpu_on_flat_terrain") else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireBrushGpuKernel::new(&device);

    let params = RealTireParams {
        radius: 0.32,
        width: 0.185,
        pressure_pa: 220_000.0,
        segments: 64,
        width_segments: 3,
        carcass_normal_stiffness: 4.0e5,
        tread_shear_stiffness: 2.6e6,
        structural_lateral_bias_per_normal: 0.0,
        relaxation_speed: 0.35,
    };
    let terrain = FlatTerrain::new(0.0, 1.0);
    let material = SurfaceMaterial::dry_asphalt();
    let state = RealWheelState {
        hub_pos: [0.0, 0.30, 0.0], // tire just touches the floor at radius 0.32
        hub_vel: [3.0, 0.0, 0.45],
        heading: [1.0, 0.0, 0.0],
        omega: 3.0 / 0.32 + 0.4, // small positive slip
    };

    // CPU reference.
    let cpu = evaluate_brush_tire(&params, &state, &terrain, &material);

    // GPU dispatch.
    let input = pack_input(&params, &state, &material);
    let out = kernel
        .dispatch_blocking(&device, &queue, &[input], None)
        .expect("brush GPU dispatch");
    let g = out[0];

    // Compare with generous tolerances (f32 vs f64).
    let rel_eq = |a: f64, b: f64, atol: f64, rtol: f64| -> bool {
        (a - b).abs() <= atol + rtol * a.abs().max(b.abs())
    };
    assert!(
        rel_eq(g.forces[1] as f64, cpu.normal_load, 1.0, 0.02),
        "Fn mismatch: gpu={} cpu={}",
        g.forces[1],
        cpu.normal_load
    );
    assert!(
        rel_eq(g.forces[3] as f64, cpu.longitudinal_force, 1.0, 0.05),
        "Fx_long mismatch: gpu={} cpu={}",
        g.forces[3],
        cpu.longitudinal_force
    );
    assert!(
        rel_eq(g.forces[2] as f64, cpu.lateral_force, 1.0, 0.05),
        "Fy_lat mismatch: gpu={} cpu={}",
        g.forces[2],
        cpu.lateral_force
    );
    assert!(
        rel_eq(g.diag2[1] as f64, cpu.aligning_moment_z, 1.0, 0.05),
        "Mz mismatch: gpu={} cpu={}",
        g.diag2[1],
        cpu.aligning_moment_z
    );
    assert!(
        rel_eq(g.diag1[0] as f64, cpu.patch_length, 1.0e-3, 0.05),
        "patch length mismatch: gpu={} cpu={}",
        g.diag1[0],
        cpu.patch_length
    );
    assert!(
        rel_eq(g.diag3[0] as f64, cpu.dissipated_power, 2.0, 0.06),
        "dissipated power mismatch: gpu={} cpu={}",
        g.diag3[0],
        cpu.dissipated_power
    );
    assert!(
        rel_eq(g.diag4[1] as f64, cpu.sliding_fraction, 1.0e-3, 0.05),
        "sliding fraction mismatch: gpu={} cpu={}",
        g.diag4[1],
        cpu.sliding_fraction
    );
    assert!(
        rel_eq(g.diag4[2] as f64, cpu.available_friction_budget, 2.0, 0.05),
        "available budget mismatch: gpu={} cpu={}",
        g.diag4[2],
        cpu.available_friction_budget
    );
    assert!(
        rel_eq(
            g.diag5[0] as f64,
            cpu.effective_longitudinal_stiffness,
            20.0,
            0.06
        ),
        "Kx mismatch: gpu={} cpu={}",
        g.diag5[0],
        cpu.effective_longitudinal_stiffness
    );
    for axis in 0..3 {
        assert!(
            rel_eq(g.world_force[axis] as f64, cpu.force[axis], 2.0, 0.05),
            "world force axis {axis} mismatch: gpu={} cpu={}",
            g.world_force[axis],
            cpu.force[axis]
        );
    }
}

/// Stage (i) — batched parity. Hand the GPU four different wheels in
/// one dispatch (mixed pressures, materials, cornering vs straight)
/// and check each row matches its CPU counterpart. This is the
/// realistic multi-wheel use case for a car.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_brush_gpu_matches_cpu_batched_four_wheels() {
    let Some(engine) = engine_or_skip("tire_brush_gpu_matches_cpu_batched_four_wheels") else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireBrushGpuKernel::new(&device);

    // Four corners of a car: front-left + front-right cornering, rear
    // pair straight on dirt.
    let asphalt = SurfaceMaterial::dry_asphalt();
    let dirt = SurfaceMaterial::loose_dirt();
    let params_fr = RealTireParams {
        radius: 0.32,
        width: 0.185,
        pressure_pa: 240_000.0,
        segments: 64,
        width_segments: 3,
        carcass_normal_stiffness: 4.0e5,
        tread_shear_stiffness: 2.6e6,
        structural_lateral_bias_per_normal: 0.0,
        relaxation_speed: 0.35,
    };
    let params_low = RealTireParams {
        pressure_pa: 130_000.0,
        ..params_fr
    };
    let terrain = FlatTerrain::new(0.0, 0.9);

    let wheels = [
        (
            params_fr,
            asphalt,
            RealWheelState {
                hub_pos: [0.0, 0.30, 0.0],
                hub_vel: [6.0, 0.0, 0.6],
                heading: [1.0, 0.0, 0.0],
                omega: 6.0 / params_fr.radius + 0.2,
            },
        ),
        (
            params_fr,
            asphalt,
            RealWheelState {
                hub_pos: [0.0, 0.30, 0.0],
                hub_vel: [6.0, 0.0, -0.4],
                heading: [1.0, 0.0, 0.0],
                omega: 6.0 / params_fr.radius + 0.3,
            },
        ),
        (
            params_low,
            dirt,
            RealWheelState {
                hub_pos: [0.0, 0.28, 0.0],
                hub_vel: [4.5, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: 4.5 / params_low.radius + 0.8,
            },
        ),
        (
            params_low,
            dirt,
            RealWheelState {
                hub_pos: [0.0, 0.28, 0.0],
                hub_vel: [4.5, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: 4.5 / params_low.radius + 0.85,
            },
        ),
    ];

    let cpu: Vec<_> = wheels
        .iter()
        .map(|(p, m, s)| evaluate_brush_tire(p, s, &terrain, m))
        .collect();
    let input: Vec<_> = wheels.iter().map(|(p, m, s)| pack_input(p, s, m)).collect();
    let out = kernel
        .dispatch_blocking(&device, &queue, &input, None)
        .expect("brush GPU batched dispatch");
    assert_eq!(out.len(), 4);
    let rel_eq = |a: f64, b: f64, atol: f64, rtol: f64| -> bool {
        (a - b).abs() <= atol + rtol * a.abs().max(b.abs())
    };
    for (idx, (g, c)) in out.iter().zip(cpu.iter()).enumerate() {
        assert!(
            rel_eq(g.forces[1] as f64, c.normal_load, 1.0, 0.04),
            "wheel {idx}: Fn gpu={} cpu={}",
            g.forces[1],
            c.normal_load
        );
        assert!(
            rel_eq(g.forces[3] as f64, c.longitudinal_force, 1.0, 0.06),
            "wheel {idx}: Fx_long gpu={} cpu={}",
            g.forces[3],
            c.longitudinal_force
        );
        assert!(
            rel_eq(g.forces[2] as f64, c.lateral_force, 1.0, 0.06),
            "wheel {idx}: Fy gpu={} cpu={}",
            g.forces[2],
            c.lateral_force
        );
        assert!(
            rel_eq(g.diag3[0] as f64, c.dissipated_power, 2.0, 0.08),
            "wheel {idx}: power gpu={} cpu={}",
            g.diag3[0],
            c.dissipated_power
        );
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_brush_gpu_persistent_batch_matches_blocking_dispatch() {
    let Some(engine) = engine_or_skip("tire_brush_gpu_persistent_batch_matches_blocking_dispatch")
    else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireBrushGpuKernel::new(&device);
    let batch = kernel
        .create_batch(&device, 4, 1)
        .expect("persistent brush batch");

    let material = SurfaceMaterial::loose_dirt();
    let params = RealTireParams {
        pressure_pa: 150_000.0,
        segments: 64,
        ..Default::default()
    };
    let wheels = [
        RealWheelState {
            hub_pos: [0.0, 0.285, 0.0],
            hub_vel: [3.5, 0.0, 0.25],
            heading: [1.0, 0.0, 0.0],
            omega: 3.5 / params.radius + 0.75,
        },
        RealWheelState {
            hub_pos: [0.0, 0.285, 0.0],
            hub_vel: [3.5, 0.0, -0.2],
            heading: [1.0, 0.0, 0.0],
            omega: 3.5 / params.radius + 0.55,
        },
    ];
    let input: Vec<_> = wheels
        .iter()
        .map(|state| pack_input(&params, state, &material))
        .collect();
    let blocking = kernel
        .dispatch_blocking(&device, &queue, &input, None)
        .expect("blocking brush dispatch");
    let persistent = kernel
        .dispatch_persistent_blocking(&device, &queue, &batch, &input, None)
        .expect("persistent brush dispatch");
    assert_eq!(persistent.len(), blocking.len());
    for (idx, (p, b)) in persistent.iter().zip(blocking.iter()).enumerate() {
        assert_eq!(p.forces, b.forces, "wheel {idx} forces changed");
        assert_eq!(p.diag1, b.diag1, "wheel {idx} diag1 changed");
        assert_eq!(p.diag2, b.diag2, "wheel {idx} diag2 changed");
        assert_eq!(p.diag3, b.diag3, "wheel {idx} diag3 changed");
        assert_eq!(p.diag4, b.diag4, "wheel {idx} diag4 changed");
        assert_eq!(p.diag5, b.diag5, "wheel {idx} diag5 changed");
        assert_eq!(
            p.world_force, b.world_force,
            "wheel {idx} world force changed"
        );
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_brush_gpu_matches_cpu_camber_extension() {
    let Some(engine) = engine_or_skip("tire_brush_gpu_matches_cpu_camber_extension") else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireBrushGpuKernel::new(&device);

    let params = RealTireParams {
        radius: 0.32,
        width: 0.185,
        pressure_pa: 220_000.0,
        segments: 64,
        width_segments: 3,
        carcass_normal_stiffness: 4.0e5,
        tread_shear_stiffness: 2.6e6,
        structural_lateral_bias_per_normal: 0.0,
        relaxation_speed: 0.35,
    };
    let terrain = FlatTerrain::new(0.0, 1.0);
    let material = SurfaceMaterial::dry_asphalt();
    let state = RealWheelState {
        hub_pos: [0.0, 0.30, 0.0],
        hub_vel: [5.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 5.0 / params.radius,
    };
    let camber = 8.0_f64.to_radians();
    let cpu = evaluate_brush_tire_oriented(
        &params,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation {
            wheel_up: [0.0, camber.cos(), camber.sin()],
        },
    );
    let input = pack_input_with_camber(&params, &state, &material, camber);
    let out = kernel
        .dispatch_blocking(&device, &queue, &[input], None)
        .expect("brush GPU camber dispatch");
    let g = out[0];
    let rel_eq = |a: f64, b: f64, atol: f64, rtol: f64| -> bool {
        (a - b).abs() <= atol + rtol * a.abs().max(b.abs())
    };
    assert!(
        rel_eq(g.forces[2] as f64, cpu.lateral_force, 2.0, 0.08),
        "camber Fy mismatch: gpu={} cpu={}",
        g.forces[2],
        cpu.lateral_force
    );
    assert!(
        rel_eq(g.diag3[1] as f64, cpu.camber_angle_rad, 1.0e-3, 0.02),
        "camber angle mismatch: gpu={} cpu={}",
        g.diag3[1],
        cpu.camber_angle_rad
    );
    assert!(
        rel_eq(g.diag3[3] as f64, cpu.pneumatic_trail_m, 2.0e-3, 0.15),
        "pneumatic trail mismatch: gpu={} cpu={}",
        g.diag3[3],
        cpu.pneumatic_trail_m
    );
}

/// Stage (i) ext — heightfield terrain path. Drive a wheel onto a
/// small tilted-slope heightfield and confirm both GPU and CPU
/// brush models still agree.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_brush_gpu_matches_cpu_on_heightfield() {
    use molla_vehicle::terrain::HeightfieldTerrain;

    let Some(engine) = engine_or_skip("tire_brush_gpu_matches_cpu_on_heightfield") else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireBrushGpuKernel::new(&device);

    // 5 × 5 heightfield centred at the origin with a small bump
    // around (0, 0). Spacing 0.4 m → 1.6 m square patch.
    let nx: u32 = 5;
    let nz: u32 = 5;
    let spacing = 0.4_f32;
    let origin_x = -(nx as f32 - 1.0) * 0.5 * spacing;
    let origin_z = -(nz as f32 - 1.0) * 0.5 * spacing;
    let mut heights = vec![0.0_f32; (nx * nz) as usize];
    for iz in 0..nz {
        for ix in 0..nx {
            // 5 cm bump at the centre, dropping off to 0 by the edges.
            let dx = ix as f32 - (nx as f32 - 1.0) * 0.5;
            let dz = iz as f32 - (nz as f32 - 1.0) * 0.5;
            let r2 = dx * dx + dz * dz;
            heights[(iz * nx + ix) as usize] = 0.05 * (-0.6 * r2).exp();
        }
    }

    let cpu_heights_f64: Vec<f64> = heights.iter().map(|&h| h as f64).collect();
    let cpu_terrain = HeightfieldTerrain::new(
        [origin_x as f64, origin_z as f64],
        spacing as f64,
        nx as usize,
        nz as usize,
        cpu_heights_f64,
        0.9,
    );

    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 64,
        ..Default::default()
    };
    let material = SurfaceMaterial::dry_asphalt();
    // Wheel sitting directly above the bump centre.
    let state = RealWheelState {
        hub_pos: [0.0, 0.30, 0.0],
        hub_vel: [3.0, 0.0, 0.3],
        heading: [1.0, 0.0, 0.0],
        omega: 3.0 / 0.32 + 0.4,
    };
    let cpu = evaluate_brush_tire(&params, &state, &cpu_terrain, &material);
    let mut input = pack_input(&params, &state, &material);
    input.extra[1] = 1.0; // heightfield mode
    let gpu_terrain = BrushTerrain {
        origin: [origin_x, origin_z],
        spacing,
        nx,
        nz,
        heights: &heights,
        friction: 0.9,
        friction_map: None,
    };
    let out = kernel
        .dispatch_blocking(&device, &queue, &[input], Some(&gpu_terrain))
        .expect("brush GPU heightfield dispatch");
    let g = out[0];
    let rel_eq = |a: f64, b: f64, atol: f64, rtol: f64| -> bool {
        (a - b).abs() <= atol + rtol * a.abs().max(b.abs())
    };
    assert!(
        rel_eq(g.forces[1] as f64, cpu.normal_load, 5.0, 0.06),
        "Fn mismatch on heightfield: gpu={} cpu={}",
        g.forces[1],
        cpu.normal_load
    );
    assert!(
        rel_eq(g.forces[3] as f64, cpu.longitudinal_force, 5.0, 0.08),
        "Fx_long mismatch on heightfield: gpu={} cpu={}",
        g.forces[3],
        cpu.longitudinal_force
    );
    assert!(
        rel_eq(g.forces[2] as f64, cpu.lateral_force, 5.0, 0.08),
        "Fy mismatch on heightfield: gpu={} cpu={}",
        g.forces[2],
        cpu.lateral_force
    );
    // The bump should produce more contact (deeper penetration) than
    // the flat floor at the same hub height.
    let flat_terrain = FlatTerrain::new(0.0, 0.9);
    let flat = evaluate_brush_tire(&params, &state, &flat_terrain, &material);
    assert!(
        cpu.normal_load > flat.normal_load,
        "heightfield bump didn't increase Fn: bump={} flat={}",
        cpu.normal_load,
        flat.normal_load
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_brush_gpu_world_force_tracks_heightfield_normals() {
    use molla_vehicle::terrain::HeightfieldTerrain;

    let Some(engine) = engine_or_skip("tire_brush_gpu_world_force_tracks_heightfield_normals")
    else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireBrushGpuKernel::new(&device);

    let nx: u32 = 7;
    let nz: u32 = 5;
    let spacing = 0.25_f32;
    let origin_x = -(nx as f32 - 1.0) * 0.5 * spacing;
    let origin_z = -(nz as f32 - 1.0) * 0.5 * spacing;
    let mut heights = Vec::with_capacity((nx * nz) as usize);
    for _iz in 0..nz {
        for ix in 0..nx {
            heights.push(0.08 * ix as f32);
        }
    }

    let cpu_terrain = HeightfieldTerrain::new(
        [origin_x as f64, origin_z as f64],
        spacing as f64,
        nx as usize,
        nz as usize,
        heights.iter().map(|&h| h as f64).collect(),
        1.0,
    );
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        bristle_damping: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let state = RealWheelState {
        hub_pos: [0.0, 0.31, 0.0],
        hub_vel: [0.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 0.0,
    };
    let cpu = evaluate_brush_tire(&params, &state, &cpu_terrain, &material);
    assert!(
        cpu.force[0] < -20.0,
        "CPU reference should include ramp-normal x force, got {:?}",
        cpu.force
    );
    let gpu_terrain = BrushTerrain {
        origin: [origin_x, origin_z],
        spacing,
        nx,
        nz,
        heights: &heights,
        friction: 1.0,
        friction_map: None,
    };
    let out = kernel
        .dispatch_blocking(
            &device,
            &queue,
            &[pack_input(&params, &state, &material)],
            Some(&gpu_terrain),
        )
        .expect("brush GPU ramp-normal dispatch");
    let g = out[0];
    let rel_eq = |a: f64, b: f64, atol: f64, rtol: f64| -> bool {
        (a - b).abs() <= atol + rtol * a.abs().max(b.abs())
    };
    for axis in 0..3 {
        assert!(
            rel_eq(g.world_force[axis] as f64, cpu.force[axis], 8.0, 0.08),
            "ramp world force axis {axis} mismatch: gpu={} cpu={}",
            g.world_force[axis],
            cpu.force[axis]
        );
    }
}

/// Stage (i) ext — material lookup on GPU. The same heightfield can carry an
/// ice/asphalt friction map; the GPU samples it per contact segment and should
/// match the CPU `HeightfieldTerrain::friction_at` path.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_brush_gpu_matches_cpu_heightfield_friction_map() {
    use molla_vehicle::terrain::HeightfieldTerrain;

    let Some(engine) = engine_or_skip("tire_brush_gpu_matches_cpu_heightfield_friction_map") else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireBrushGpuKernel::new(&device);

    let nx: u32 = 7;
    let nz: u32 = 5;
    let spacing = 0.25_f32;
    let origin_x = -(nx as f32 - 1.0) * 0.5 * spacing;
    let origin_z = -(nz as f32 - 1.0) * 0.5 * spacing;
    let heights = vec![0.0_f32; (nx * nz) as usize];
    let mut friction = vec![1.0_f32; (nx * nz) as usize];
    for iz in 0..nz {
        for ix in 0..nx {
            // Low-grip ice band under/just ahead of the wheel centre.
            if (2..=4).contains(&ix) {
                friction[(iz * nx + ix) as usize] = 0.18;
            }
        }
    }

    let cpu_heights: Vec<f64> = heights.iter().map(|&h| h as f64).collect();
    let cpu_friction: Vec<f64> = friction.iter().map(|&f| f as f64).collect();
    let cpu_terrain = HeightfieldTerrain::with_friction_map(
        [origin_x as f64, origin_z as f64],
        spacing as f64,
        nx as usize,
        nz as usize,
        cpu_heights,
        1.0,
        cpu_friction,
    );

    let params = RealTireParams {
        pressure_pa: 190_000.0,
        segments: 96,
        ..Default::default()
    };
    let material = SurfaceMaterial::dry_asphalt();
    let state = RealWheelState {
        hub_pos: [0.0, 0.29, 0.0],
        hub_vel: [4.0, 0.0, 0.35],
        heading: [1.0, 0.0, 0.0],
        omega: 4.0 / params.radius + 1.1,
    };

    let cpu = evaluate_brush_tire(&params, &state, &cpu_terrain, &material);
    let input = pack_input(&params, &state, &material);
    let gpu_terrain = BrushTerrain {
        origin: [origin_x, origin_z],
        spacing,
        nx,
        nz,
        heights: &heights,
        friction: 1.0,
        friction_map: Some(&friction),
    };
    let out = kernel
        .dispatch_blocking(&device, &queue, &[input], Some(&gpu_terrain))
        .expect("brush GPU heightfield+friction dispatch");
    let g = out[0];
    let rel_eq = |a: f64, b: f64, atol: f64, rtol: f64| -> bool {
        (a - b).abs() <= atol + rtol * a.abs().max(b.abs())
    };
    assert!(
        rel_eq(g.forces[3] as f64, cpu.longitudinal_force, 5.0, 0.08),
        "Fx_long mismatch with friction map: gpu={} cpu={}",
        g.forces[3],
        cpu.longitudinal_force
    );
    assert!(
        rel_eq(g.forces[2] as f64, cpu.lateral_force, 5.0, 0.08),
        "Fy mismatch with friction map: gpu={} cpu={}",
        g.forces[2],
        cpu.lateral_force
    );

    let uniform_terrain = HeightfieldTerrain::new(
        [origin_x as f64, origin_z as f64],
        spacing as f64,
        nx as usize,
        nz as usize,
        heights.iter().map(|&h| h as f64).collect(),
        1.0,
    );
    let uniform = evaluate_brush_tire(&params, &state, &uniform_terrain, &material);
    assert!(
        cpu.consumed_friction_budget < uniform.consumed_friction_budget * 0.55,
        "friction map did not reduce consumed grip enough: mapped={} uniform={}",
        cpu.consumed_friction_budget,
        uniform.consumed_friction_budget
    );
}
