#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::Vec3;
use molla_solvers::airflow::{AirConfig, AirForces, AirState, GpuAir, Grid, RotorDisk};

fn vortex(p: Vec3) -> Vec3 {
    let r = p - Vec3::Y * 1.6;
    Vec3::new(-r.y, r.x, 0.0) * (1.4 * (-r.length_squared() / 0.25).exp())
}

fn samples() -> Vec<Vec3> {
    (0..2197)
        .map(|i| {
            Vec3::new(
                (i % 13) as f64 * 0.15 - 0.9,
                ((i / 13) % 13) as f64 * 0.15 + 0.7,
                (i / 169) as f64 * 0.15 - 0.9,
            )
        })
        .collect()
}

fn run_vortex(engine: &Engine, n: usize, steps: usize) -> Vec<Vec3> {
    let grid = Grid::new([n; 3], 3.2 / n as f64, Vec3::new(-1.6, 0.0, -1.6)).unwrap();
    let mut state = AirState::new(AirConfig {
        pressure_sweeps: 400,
        ..AirConfig::new(grid)
    })
    .unwrap();
    state.seed_velocity(vortex).unwrap();
    let mut gpu = GpuAir::new(engine, &state).unwrap();
    let forces = AirForces::new(grid);
    for _ in 0..steps {
        gpu.step(engine, 0.4 / steps as f64, &forces).unwrap();
    }
    let field = gpu.snapshot(engine).unwrap();
    assert!(
        field.diagnostics().divergence_rms < 1e-4,
        "{:?}",
        field.diagnostics()
    );
    assert_eq!(field.diagnostics().ground_flux_max, 0.0);
    gpu.sample(engine, &samples()).unwrap()
}

fn error(a: &[Vec3], b: &[Vec3]) -> f64 {
    (a.iter()
        .zip(b)
        .map(|(a, b)| a.distance_squared(*b))
        .sum::<f64>()
        / a.len() as f64)
        .sqrt()
}

#[test]
#[ignore = "requires a GPU; spatial and temporal refinement study"]
fn transported_vortex_refines_in_space_and_time() {
    let engine = Engine::new_gpu().unwrap();
    let reference = run_vortex(&engine, 64, 192);
    let mut spatial = Vec::new();
    for n in [16, 24, 32] {
        let value = error(&run_vortex(&engine, n, 192), &reference);
        println!(
            "vortex spatial n={n}, h={}, rms_error_m_s={value}",
            3.2 / n as f64
        );
        spatial.push(value);
    }
    assert!(spatial[1] < spatial[0] && spatial[2] < spatial[1]);
    let reference = run_vortex(&engine, 32, 192);
    let mut temporal = Vec::new();
    for steps in [12, 24, 48] {
        let value = error(&run_vortex(&engine, 32, steps), &reference);
        println!(
            "vortex temporal dt={}, rms_error_m_s={value}",
            0.4 / steps as f64
        );
        temporal.push(value);
    }
    assert!(temporal[1] < temporal[0] && temporal[2] < temporal[1]);
    let first = run_vortex(&engine, 32, 48);
    let second = run_vortex(&engine, 32, 48);
    let repeat_error = error(&first, &second);
    println!("vortex repeated run rms_error_m_s={repeat_error}");
    assert!(repeat_error < 1e-7);
}

#[test]
#[ignore = "requires a GPU; actuator-disk grid sensitivity study"]
fn disk_wake_refinement_diagnostics() {
    let engine = Engine::new_gpu().unwrap();
    let points: Vec<_> = (0..625)
        .map(|i| {
            Vec3::new(
                (i % 25) as f64 * 0.05 - 0.6,
                1.4,
                (i / 25) as f64 * 0.05 - 0.6,
            )
        })
        .collect();
    let mut results = Vec::new();
    for n in [16, 24, 32, 48] {
        let grid = Grid::new([n; 3], 3.2 / n as f64, Vec3::new(-1.6, 0.0, -1.6)).unwrap();
        let config = AirConfig {
            pressure_sweeps: 400,
            ..AirConfig::new(grid)
        };
        let mut air = GpuAir::new(&engine, &AirState::new(config).unwrap()).unwrap();
        let mut forces = AirForces::new(grid);
        forces
            .add_rotor(RotorDisk {
                center: Vec3::Y * 2.3,
                axis: Vec3::Y,
                radius: 0.28,
                thrust: 20.0,
                air_torque: 0.0,
            })
            .unwrap();
        for _ in 0..240 {
            air.step(&engine, 1.0 / 240.0, &forces).unwrap();
        }
        let samples = air.sample(&engine, &points).unwrap();
        let downwash = samples.iter().map(|v| -v.y).sum::<f64>() / samples.len() as f64;
        let momentum = samples
            .iter()
            .map(|v| 1.225 * v.y.min(0.0).powi(2) * 0.05f64.powi(2))
            .sum::<f64>();
        let diagnostics = air.snapshot(&engine).unwrap().diagnostics();
        println!(
            "disk grid n={n}, h={}, mean_downwash_m_s={downwash}, sampled_momentum_flux_n={momentum}, {diagnostics:?}",
            grid.spacing()
        );
        assert!(downwash > 0.2 && downwash < 20.0);
        assert!(diagnostics.divergence_rms < 0.01);
        assert_eq!(diagnostics.ground_flux_max, 0.0);
        results.push(samples);
    }
    let errors: Vec<_> = (0..3).map(|i| error(&results[i], &results[3])).collect();
    for value in &errors {
        println!("disk velocity rms error against finest: {value}");
    }
    assert!(errors[1] < errors[0] && errors[2] < errors[1]);
}
