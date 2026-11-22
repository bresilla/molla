use molla_math::Vec3;
use molla_solvers::airflow::{AirConfig, AirForces, AirState, Grid, RotorDisk};

#[path = "airflow/exterior.rs"]
mod exterior;
#[path = "airflow/turbulence.rs"]
mod turbulence;

fn grid() -> Grid {
    Grid::new([12, 10, 12], 0.15, Vec3::new(-0.9, 0.0, -0.9)).unwrap()
}

#[test]
fn coherent_background_is_divergence_free_and_extends_beyond_grid() {
    let mut c = config();
    c.ambient = Vec3::new(1.2, 0.0, 0.35);
    c.gust_strength = 0.65;
    c.time = 3.7;
    let air = AirState::new(c).unwrap();
    for p in [Vec3::new(0.3, 1.1, -0.2), Vec3::new(50.0, 2.0, -12.0)] {
        let h = 1e-5;
        let div = [Vec3::X, Vec3::Y, Vec3::Z]
            .iter()
            .enumerate()
            .map(|(i, a)| (c.background(p + *a * h)[i] - c.background(p - *a * h)[i]) / (2.0 * h))
            .sum::<f64>();
        assert!(div.abs() < 1e-8);
        assert!(c.background(p).length() > 0.3);
    }
    let far = Vec3::new(50.0, 2.0, -12.0);
    assert_eq!(air.velocity_at(far).unwrap(), c.background(far));
    assert_eq!(c.background(Vec3::new(0.7, 0.0, 0.3)).y, 0.0);
    let mut later = c;
    later.time += 1.0;
    assert!(later.background(far).distance(c.background(far)) > 0.05);
}

#[test]
fn pure_leaf_couple_preserves_zero_force_and_requested_torque() {
    let mut forces = AirForces::new(grid());
    let center = Vec3::new(0.02, 0.7, 0.03);
    let torque = Vec3::new(0.004, -0.002, 0.003);
    forces.add_torque(center, torque).unwrap();
    let (force, moment) = forces.wrench(center);
    assert!(force.length() < 1e-15 && moment.distance(torque) < 1e-15);
    let old = forces.wrench(center);
    assert!(forces.add_torque(Vec3::ZERO, torque).is_err());
    assert_eq!(old, forces.wrench(center));
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn gpu_gusts_match_cpu_at_evolving_exterior_boundaries() {
    use molla_compute::Engine;
    use molla_solvers::airflow::GpuAir;
    let engine = Engine::new_gpu().unwrap();
    let mut c = config();
    c.ambient = Vec3::new(1.2, 0.0, 0.35);
    c.gust_strength = 0.65;
    let mut cpu = AirState::new(c).unwrap();
    let mut gpu = GpuAir::new(&engine, &cpu).unwrap();
    let forces = AirForces::new(c.grid);
    let points = [
        Vec3::new(0.1, 0.7, 0.2),
        Vec3::new(1.1, 0.7, 0.2),
        Vec3::new(35.0, 3.0, -8.0),
    ];
    for _ in 0..12 {
        cpu.step(1.0 / 60.0, &forces).unwrap();
        gpu.step(&engine, 1.0 / 60.0, &forces).unwrap();
        for (p, v) in points.iter().zip(gpu.sample(&engine, &points).unwrap()) {
            assert!(cpu.velocity_at(*p).unwrap().distance(v) < 2e-5);
        }
    }
    assert_eq!(
        gpu.snapshot(&engine).unwrap().config().time,
        cpu.config().time
    );
    assert_eq!(
        gpu.snapshot(&engine).unwrap().diagnostics().ground_flux_max,
        0.0
    );
}
#[test]
fn background_wind_change_moves_the_whole_air_mass_and_stays_steady() {
    let c = config();
    let mut air = AirState::new(c).unwrap();
    let forces = AirForces::new(c.grid);
    assert!(air.set_background(Vec3::X * f64::NAN, 0.0).is_err());
    assert!(air.set_background(Vec3::X, -1.0).is_err());
    let wind = Vec3::new(3.0, 0.0, 0.5);
    air.set_background(wind, 0.0).unwrap();
    let points = [Vec3::new(0.0, 0.7, 0.0), Vec3::new(-0.8, 0.1, 0.8), Vec3::new(30.0, 2.0, -5.0)];
    for step in 0..120 {
        air.step(1.0 / 60.0, &forces).unwrap();
        for p in points {
            assert!(air.velocity_at(p).unwrap().distance(wind) < 1e-9, "step {step} at {p}");
        }
    }
    let d = air.diagnostics();
    assert!(d.divergence_max < 1e-9 && d.ground_flux_max == 0.0, "{d:?}");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn gpu_background_wind_change_matches_cpu_around_a_working_rotor() {
    use molla_compute::Engine;
    use molla_solvers::airflow::GpuAir;
    let engine = Engine::new_gpu().unwrap();
    let c = AirConfig {
        ambient: Vec3::new(1.2, 0.0, 0.35),
        gust_strength: 0.65,
        ..config()
    };
    let mut cpu = AirState::new(c).unwrap();
    let mut gpu = GpuAir::new(&engine, &cpu).unwrap();
    assert!(gpu.set_background(Vec3::X * f64::NAN, 0.0).is_err());
    assert_eq!(gpu.background(), (c.ambient, 0.65));
    let mut forces = AirForces::new(c.grid);
    forces.add_rotor(rotor()).unwrap();
    let points = [
        Vec3::new(0.1, 0.4, 0.2),
        Vec3::new(-0.8, 0.7, -0.6),
        Vec3::new(35.0, 3.0, -8.0),
    ];
    for step in 0..30 {
        if (10..16).contains(&step) {
            let wind = c.ambient.lerp(Vec3::new(6.0, 0.0, 0.35), (step - 9) as f64 / 6.0);
            cpu.set_background(wind, 1.2).unwrap();
            gpu.set_background(wind, 1.2).unwrap();
        }
        cpu.step(1.0 / 60.0, &forces).unwrap();
        gpu.step(&engine, 1.0 / 60.0, &forces).unwrap();
        for (p, v) in points.iter().zip(gpu.sample(&engine, &points).unwrap()) {
            assert!(cpu.velocity_at(*p).unwrap().distance(v) < 5e-5, "step {step} at {p}");
        }
    }
    assert!(cpu.velocity_at(points[1]).unwrap().x > 4.0);
    assert_eq!(gpu.snapshot(&engine).unwrap().diagnostics().ground_flux_max, 0.0);
}

fn config() -> AirConfig {
    AirConfig {
        pressure_sweeps: 160,
        ..AirConfig::new(grid())
    }
}
fn rotor() -> RotorDisk {
    RotorDisk {
        center: Vec3::new(0.03, 0.9, -0.02),
        axis: Vec3::Y,
        radius: 0.23,
        thrust: 2.5,
        air_torque: 0.03,
    }
}

#[test]
fn stationary_air_and_stopped_rotors_remain_stationary() {
    let mut air = AirState::new(config()).unwrap();
    let mut f = AirForces::new(grid());
    f.add_rotor(RotorDisk {
        thrust: 0.0,
        air_torque: 0.0,
        ..rotor()
    })
    .unwrap();
    for _ in 0..3 {
        air.step(1.0 / 120.0, &f).unwrap();
    }
    assert_eq!(air.diagnostics().speed_max, 0.0);
    assert_eq!(air.diagnostics().divergence_max, 0.0);
}

#[test]
fn deposited_force_preserves_force_torque_and_affine_sampling() {
    let c = config();
    let mut air = AirState::new(c).unwrap();
    air.seed_velocity(|p| Vec3::new(0.1 + 0.4 * p.y, 0.3 * p.x + 0.2, -0.2 * p.z))
        .unwrap();
    let point = Vec3::new(0.071, 0.619, -0.142);
    let force = Vec3::new(2.0, -3.0, 0.7);
    let about = Vec3::new(-0.1, 0.2, 0.4);
    let mut loads = AirForces::new(c.grid);
    loads.add(point, force).unwrap();
    let (f, q) = loads.wrench(about);
    assert!(f.distance(force) < 1e-12);
    assert!(q.distance((point - about).cross(force)) < 1e-12);
    let v = air.velocity_at(point).unwrap();
    let expected = Vec3::new(0.1 + 0.4 * point.y, 0.3 * point.x + 0.2, -0.2 * point.z);
    assert!(v.distance(expected) < 1e-12);
    assert!((v.dot(force) - expected.dot(f)).abs() < 1e-12);
}

#[test]
fn tilted_rotor_transfers_opposite_thrust_and_shaft_torque() {
    for axis in [Vec3::Y, Vec3::new(1.0, 2.0, -1.0).normalize(), Vec3::Z] {
        let rotor = RotorDisk { axis, ..rotor() };
        let mut f = AirForces::new(grid());
        f.add_rotor(rotor).unwrap();
        let (force, torque) = f.wrench(rotor.center);
        assert!(force.distance(-axis * rotor.thrust) < 1e-12, "{force:?}");
        assert!(
            torque.distance(axis * rotor.air_torque) < 1e-12,
            "{torque:?}"
        );
    }
}

#[test]
fn projection_removes_divergence_and_preserves_impermeable_ground() {
    let mut air = AirState::new(config()).unwrap();
    air.seed_velocity(|p| Vec3::new((p.x * 4.0).sin(), (p.y * 5.0).sin(), (p.z * 3.0).cos()))
        .unwrap();
    let before = air.diagnostics();
    air.project(1.0 / 60.0).unwrap();
    let after = air.diagnostics();
    assert!(before.divergence_rms > 1.0);
    assert!(
        after.divergence_rms < before.divergence_rms * 1e-8,
        "{before:?} -> {after:?}"
    );
    assert_eq!(after.ground_flux_max, 0.0);
}

#[test]
fn rotor_drives_downward_flow_without_a_prescribed_velocity_field() {
    let mut air = AirState::new(config()).unwrap();
    let mut f = AirForces::new(grid());
    f.add_rotor(rotor()).unwrap();
    for _ in 0..20 {
        air.step(1.0 / 120.0, &f).unwrap();
    }
    let below = air.velocity_at(Vec3::new(0.03, 0.66, -0.02)).unwrap();
    let d = air.diagnostics();
    println!("CPU wake below rotor {below:?}; {d:?}");
    assert!(below.y < -0.15, "{below:?}");
    assert!(d.divergence_rms < 1e-7, "{d:?}");
    assert_eq!(d.ground_flux_max, 0.0);
    f.clear();
    air.step(1.0 / 120.0, &f).unwrap();
    assert!(
        air.diagnostics().speed_max > 0.1,
        "existing wake must not vanish when a motor stops"
    );
}

#[test]
fn ambient_horizontal_flow_is_a_fixed_point() {
    let c = AirConfig {
        ambient: Vec3::new(0.3, 0.0, -0.2),
        ..config()
    };
    let mut air = AirState::new(c).unwrap();
    air.step(1.0 / 60.0, &AirForces::new(c.grid)).unwrap();
    assert!(
        air.velocity_at(Vec3::new(0.0, 0.6, 0.0))
            .unwrap()
            .distance(c.ambient)
            < 1e-12
    );
    assert!(air.diagnostics().divergence_max < 1e-12);
    assert_eq!(air.velocity_at(Vec3::splat(100.0)).unwrap(), c.ambient);
}

#[test]
fn invalid_steps_and_out_of_domain_rotors_do_not_change_state() {
    assert!(Grid::new([0, 4, 4], 0.1, Vec3::ZERO).is_err());
    assert!(Grid::new([usize::MAX, 4, 4], 0.1, Vec3::ZERO).is_err());
    let mut air = AirState::new(config()).unwrap();
    let before = air.face_velocities().to_vec();
    let mut f = AirForces::new(grid());
    assert!(
        f.add_rotor(RotorDisk {
            center: Vec3::ZERO,
            ..rotor()
        })
        .is_err()
    );
    assert_eq!(f.wrench(Vec3::ZERO), (Vec3::ZERO, Vec3::ZERO));
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(air.step(dt, &f).is_err());
    }
    assert_eq!(air.face_velocities(), &before);
    assert!(air.velocity_at(Vec3::splat(f64::NAN)).is_err());
    assert!(air.seed_velocity(|_| Vec3::splat(f64::NAN)).is_err());
    assert_eq!(air.face_velocities(), &before);
    let other = Grid::new([8, 8, 8], 0.1, Vec3::ZERO).unwrap();
    assert!(air.step(0.01, &AirForces::new(other)).is_err());
}

#[test]
fn advection_transports_a_weak_transverse_velocity_pulse() {
    let g = Grid::new([32, 8, 12], 0.1, Vec3::new(-1.6, 0.0, -0.6)).unwrap();
    let c = AirConfig {
        ambient: Vec3::X * 0.8,
        viscosity: 0.0,
        pressure_sweeps: 160,
        ..AirConfig::new(g)
    };
    let mut air = AirState::new(c).unwrap();
    air.seed_velocity(|p| c.ambient + Vec3::Z * (0.001 * (-((p.x + 0.4) / 0.2).powi(2)).exp()))
        .unwrap();
    let centroid = |state: &AirState| {
        let mut weight = 0.0;
        let mut moment = 0.0;
        for i in 0..32 {
            let x = -1.6 + (i as f64 + 0.5) * 0.1;
            let w = state.velocity_at(Vec3::new(x, 0.4, 0.0)).unwrap().z;
            weight += w;
            moment += x * w;
        }
        moment / weight
    };
    let before = centroid(&air);
    let forces = AirForces::new(g);
    for _ in 0..20 {
        air.step(0.01, &forces).unwrap();
    }
    let displacement = centroid(&air) - before;
    assert!(
        (displacement - 0.16).abs() < 0.005,
        "advected {displacement} m, expected 0.16 m"
    );
    assert!(air.diagnostics().divergence_rms < 1e-7);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU"]
fn resident_gpu_matches_cpu_with_moving_tilted_rotor() {
    use molla_compute::Engine;
    use molla_solvers::airflow::GpuAir;
    let engine = Engine::new_gpu().unwrap();
    let mut cpu = AirState::new(config()).unwrap();
    let mut gpu = GpuAir::new(&engine, &cpu).unwrap();
    let dt = 1.0 / 120.0;
    let start = std::time::Instant::now();
    for step in 0..12 {
        let mut f = AirForces::new(grid());
        f.add_rotor(RotorDisk {
            center: rotor().center + Vec3::X * (step as f64 * 0.004),
            axis: Vec3::new(0.08, 1.0, -0.03).normalize(),
            ..rotor()
        })
        .unwrap();
        gpu.step(&engine, dt, &f).unwrap();
        cpu.step(dt, &f).unwrap();
    }
    let actual = gpu.snapshot(&engine).unwrap();
    let error = actual
        .face_velocities()
        .iter()
        .zip(cpu.face_velocities())
        .map(|(a, b)| a.distance(*b))
        .fold(0.0, f64::max);
    let pressure_error = actual
        .pressures()
        .iter()
        .zip(cpu.pressures())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let d = actual.diagnostics();
    println!(
        "air CPU/GPU: velocity error {error:e}, pressure error {pressure_error:e}, {d:?}, elapsed {:?}",
        start.elapsed()
    );
    assert!(error < 2e-5, "velocity error {error:e}");
    assert!(pressure_error < 3e-5, "pressure error {pressure_error:e}");
    assert!(d.divergence_rms < 2e-5, "{d:?}");
    assert_eq!(d.ground_flux_max, 0.0);
    assert!(
        actual
            .velocity_at(rotor().center - Vec3::Y * 0.2)
            .unwrap()
            .y
            < -0.1
    );
    let points: Vec<_> = (0..4201)
        .map(|i| {
            Vec3::new(
                (i as f64 * 0.037).sin() * 0.65,
                0.2 + (i as f64 * 0.043).sin().abs(),
                (i as f64 * 0.053).cos() * 0.65,
            )
        })
        .chain([Vec3::splat(100.0), Vec3::new(0.0, 0.0, 0.0)])
        .collect();
    let samples = gpu.sample(&engine, &points).unwrap();
    let sampling_error = points
        .iter()
        .zip(&samples)
        .map(|(p, v)| actual.velocity_at(*p).unwrap().distance(*v))
        .fold(0.0, f64::max);
    assert!(sampling_error < 2e-6, "sampling error {sampling_error:e}");
    assert_eq!(samples.last().unwrap().y, 0.0);
    assert!(gpu.sample(&engine, &[Vec3::splat(f64::NAN)]).is_err());
    assert!(gpu.sample(&engine, &[]).unwrap().is_empty());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU"]
fn orchard_sized_air_timing_and_projection() {
    use molla_compute::Engine;
    use molla_solvers::airflow::GpuAir;
    let engine = Engine::new_gpu().unwrap();
    println!("air adapter: {:?}", engine.adapter_info());
    let g = Grid::new([64, 32, 48], 0.15, Vec3::new(-4.8, 0.0, -3.6)).unwrap();
    let c = AirConfig::new(g);
    let mut gpu = GpuAir::new(&engine, &AirState::new(c).unwrap()).unwrap();
    let mut forces = AirForces::new(g);
    for (x, z, spin) in [
        (-0.4, -0.4, 1.0),
        (-0.4, 0.4, -1.0),
        (0.4, -0.4, -1.0),
        (0.4, 0.4, 1.0),
    ] {
        forces
            .add_rotor(RotorDisk {
                center: Vec3::new(x, 2.6, z),
                radius: 0.28,
                thrust: 12.0,
                air_torque: 0.15 * spin,
                axis: Vec3::Y,
            })
            .unwrap();
    }
    let queries: Vec<_> = (0..3200)
        .map(|i| {
            Vec3::new(
                (i % 40) as f64 * 0.15 - 3.0,
                0.2 + (i / 400) as f64 * 0.25,
                ((i / 40) % 10) as f64 * 0.3 - 1.5,
            )
        })
        .collect();
    let mut milliseconds = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..30 {
        let start = std::time::Instant::now();
        gpu.step(&engine, 1.0 / 60.0, &forces).unwrap();
        submissions.push(start.elapsed().as_secs_f64() * 1000.0);
        gpu.sample(&engine, &queries).unwrap();
        milliseconds.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let field = gpu.snapshot(&engine).unwrap();
    let d = field.diagnostics();
    milliseconds.sort_by(f64::total_cmp);
    submissions.sort_by(f64::total_cmp);
    println!("air submission median {:.3} ms", submissions[15]);
    println!(
        "air grid {:?}: step plus 3200 local samples median {:.3} ms; max {:.3} ms; {:?}",
        g.cells(),
        milliseconds[15],
        milliseconds[29],
        d
    );
    assert!(d.divergence_rms < 0.005, "{d:?}");
    assert!(d.speed_max > 1.0 && d.speed_max < 30.0, "{d:?}");
    assert_eq!(d.ground_flux_max, 0.0);
}
#[test]
fn cylinder_drag_quadrature_is_dissipative_and_time_stable() {
    use molla_solvers::airflow::CylinderDrag;
    let cylinder = CylinderDrag {
        radius: 0.03,
        length: 0.3,
        density: 1.225,
        transverse_coefficient: 1.1,
        axial_coefficient: 0.8,
    };
    let wind = Vec3::new(12.0, -3.0, 4.0);
    let force = cylinder
        .sample_force(wind, Vec3::Y, 0.0, 0.01, 1.0)
        .unwrap();
    let half = cylinder
        .sample_force(wind, Vec3::Y, 0.0, 0.01, 0.5)
        .unwrap();
    assert!((half * 2.0 - force).length() < 1e-12);
    assert!(force.dot(wind) > 0.0);
    let reverse = cylinder
        .sample_force(-wind, Vec3::Y, 0.0, 0.01, 1.0)
        .unwrap();
    assert!((force + reverse).length() < 1e-12);
    for dt in [1e-4, 0.01, 1.0, 1000.0] {
        let f = cylinder
            .sample_force(wind, Vec3::Y, 100.0, dt, 1.0)
            .unwrap();
        let remaining = wind - f * (dt * 100.0);
        assert!(remaining.length() < wind.length());
        assert!(remaining.dot(wind) > 0.0);
    }
    assert!(
        cylinder
            .sample_force(wind, Vec3::ZERO, 0.0, 0.01, 1.0)
            .is_err()
    );
    assert!(
        cylinder
            .sample_force(wind, Vec3::Y, -1.0, 0.01, 1.0)
            .is_err()
    );
    assert!(cylinder.sample_force(wind, Vec3::Y, 0.0, 0.0, 1.0).is_err());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU"]
fn dense_sparse_and_empty_force_uploads_match_the_cpu() {
    use molla_compute::Engine;
    use molla_solvers::airflow::GpuAir;
    let engine = Engine::new_gpu().unwrap();
    let grid = Grid::new([8, 8, 8], 0.2, Vec3::ZERO).unwrap();
    let mut cpu = AirState::new(AirConfig::new(grid)).unwrap();
    let mut gpu = GpuAir::new(&engine, &cpu).unwrap();
    let mut forces = AirForces::new(grid);
    for stage in 0..4 {
        forces.clear();
        if stage == 0 {
            for x in 1..7 {
                for y in 1..7 {
                    for z in 1..7 {
                        forces
                            .add(
                                Vec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5) * 0.2,
                                Vec3::new(0.003, -0.002, 0.001),
                            )
                            .unwrap();
                    }
                }
            }
        } else if stage == 2 {
            forces
                .add(Vec3::splat(0.7), Vec3::new(-0.05, 0.03, 0.01))
                .unwrap();
        }
        cpu.step(1.0 / 120.0, &forces).unwrap();
        gpu.step(&engine, 1.0 / 120.0, &forces).unwrap();
        let actual = gpu.snapshot(&engine).unwrap();
        let error = cpu
            .face_velocities()
            .iter()
            .zip(actual.face_velocities())
            .map(|(a, b)| a.distance(*b))
            .fold(0.0, f64::max);
        assert!(error < 2e-6, "stage {stage}: {error}");
    }
}
