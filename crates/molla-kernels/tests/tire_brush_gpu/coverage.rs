use super::*;

#[test]
#[ignore = "needs a real wgpu adapter"]
fn gpu_partial_contact_area_and_forces_match_cpu_without_plateaus() {
    let engine = Engine::new_gpu().expect("GPU adapter required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireBrushGpuKernel::new(device);
    let material = SurfaceMaterial::dry_asphalt();
    for pressure in [80_000.0, 180_000.0, 320_000.0] {
        let params = RealTireParams {
            radius: 0.88820508,
            width: 0.58,
            pressure_pa: pressure,
            segments: 48,
            width_segments: 3,
            ..Default::default()
        };
        let depths: Vec<_> = (0..81)
            .map(|i| 0.04 + i as f64 * 0.0005)
            .chain([-0.01, 0.0])
            .collect();
        let states: Vec<_> = depths
            .iter()
            .map(|&depth| RealWheelState {
                hub_pos: [0.0, params.radius - depth, 0.0],
                hub_vel: [2.0, 0.0, 0.2],
                heading: [1.0, 0.0, 0.0],
                omega: 3.0,
            })
            .collect();
        let inputs: Vec<_> = states
            .iter()
            .map(|s| pack_input(&params, s, &material))
            .collect();
        let results = kernel
            .dispatch_blocking(device, queue, &inputs, None)
            .unwrap();
        let mut previous = (0.0, 0.0);
        for ((state, gpu), depth) in states.iter().zip(results).zip(depths) {
            let cpu = evaluate_brush_tire(&params, state, &FlatTerrain::new(0.0, 1.0), &material);
            let area: f64 = cpu.cells.iter().map(|c| c.area).sum();
            for (label, actual, expected, floor) in [
                ("area", gpu.diag5[2], area, 1e-6),
                ("length", gpu.diag1[0], cpu.patch_length, 1e-6),
                ("load", gpu.forces[1], cpu.normal_load, 0.02),
                ("longitudinal", gpu.forces[3], cpu.longitudinal_force, 0.02),
                ("lateral", gpu.forces[2], cpu.lateral_force, 0.02),
                ("moment", gpu.diag2[1], cpu.aligning_moment_z, 0.02),
            ] {
                assert!(
                    (actual as f64 - expected).abs() < floor + expected.abs() * 0.002,
                    "{label}: GPU {actual}, CPU {expected}, depth {depth}, pressure {pressure}"
                );
            }
            if depth > 0.0 {
                assert!(gpu.diag5[2] > previous.0 && gpu.forces[1] > previous.1);
                previous = (gpu.diag5[2], gpu.forces[1]);
            } else {
                assert_eq!(gpu.diag5[2], 0.0);
                assert_eq!(gpu.forces[1], 0.0);
            }
        }
    }
}
