use molla_math::Vec3;
use molla_solvers::airflow::{AirForces, Grid, Quadrotor, QuadrotorConfig};

const DT: f64 = 1.0 / 240.0;

#[test]
fn bounded_trajectory_is_continuous_and_respects_vector_limits() {
    use molla_solvers::airflow::{FlightSegment, MotionLimits};
    let limits = MotionLimits {
        speed: 1.4,
        acceleration: 2.0,
        jerk: 5.0,
    };
    for distance in [0.0, 0.001, 0.3, 1.0, 4.0, 20.0] {
        let start = Vec3::new(0.4, 2.0, -1.0);
        let end = start + Vec3::new(1.0, 0.2, -0.7).normalize() * distance;
        for stretch in [1.0, 2.3] {
            let base = FlightSegment::new(start, end, limits).unwrap();
            let profile = base
                .with_duration((base.duration() * stretch).max(0.1))
                .unwrap();
            let dt = profile.duration() / 10_000.0;
            let mut last = profile.sample(0.0).unwrap();
            for i in 1..=10_000 {
                let p = profile.sample(i as f64 * dt).unwrap();
                assert!(p.velocity.length() <= limits.speed + 1e-8);
                assert!(p.acceleration.length() <= limits.acceleration + 1e-8);
                assert!(p.jerk.length() <= limits.jerk + 1e-8);
                assert!((p.acceleration - last.acceleration).length() <= limits.jerk * dt + 1e-8);
                assert!(
                    (p.position - last.position - (p.velocity + last.velocity) * (0.5 * dt))
                        .length()
                        < 1e-6
                );
                last = p;
            }
            assert!(last.position.distance(end) < 1e-8);
            assert!(last.velocity.length() < 1e-8 && last.acceleration.length() < 1e-8);
            assert!(profile.sample(f64::NAN).is_err());
            if distance > 0.0 {
                assert!(base.with_duration(base.duration() * 0.5).is_err());
            }
        }
    }
    assert!(
        FlightSegment::new(
            Vec3::ZERO,
            Vec3::X,
            MotionLimits {
                jerk: 0.0,
                ..limits
            }
        )
        .is_err()
    );
}

#[test]
fn flight_leans_accelerates_counterleans_brakes_and_rotates_wash_in_eight_directions() {
    use molla_solvers::airflow::{FlightReference, FlightSegment, MotionLimits};
    for dt in [1.0 / 60.0, DT] {
        for index in 0..8 {
            let a = index as f64 * std::f64::consts::FRAC_PI_4;
            let direction = Vec3::new(a.cos(), 0.0, a.sin());
            let start = Vec3::Y * 3.0;
            let mass = [2.0, 5.6, 8.0][index % 3];
            let mut q = Quadrotor::new(
                start,
                QuadrotorConfig {
                    mass,
                    ..Default::default()
                },
            )
            .unwrap();
            for _ in 0..(4.0 / dt) as usize {
                q.track(FlightReference::hold(start), Vec3::X).unwrap();
                q.step(dt, Vec3::new(0.6, 0.0, -0.3)).unwrap();
            }
            let profile =
                FlightSegment::new(start, start + direction * 4.0, MotionLimits::default())
                    .unwrap();
            let (mut lean, mut counterlean, mut tracking, mut altitude, mut air_error) =
                (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for step in 0..((profile.duration() + 4.0) / dt).ceil() as usize {
                let reference = profile.sample(step as f64 * dt).unwrap();
                q.track(reference, Vec3::X).unwrap();
                q.step(dt, Vec3::new(0.6, 0.0, -0.3)).unwrap();
                let p = q.pose().unwrap();
                let thrust_axis = p.rotation * Vec3::Z;
                let slope = thrust_axis.dot(direction).atan2(thrust_axis.y);
                if reference.acceleration.dot(direction) > 0.2 {
                    lean = lean.max(slope);
                }
                if reference.acceleration.dot(direction) < -0.2 {
                    counterlean = counterlean.min(slope);
                }
                tracking = tracking.max(p.position.distance(reference.position));
                altitude = altitude.max((p.position.y - start.y).abs());
                let air = q.air_loads().unwrap().iter().map(|l| l.force).sum::<Vec3>();
                let body = q
                    .propellers
                    .iter()
                    .map(|id| q.world.devices.propellers.reading(*id).unwrap().world_force)
                    .sum::<Vec3>();
                air_error = air_error.max((air + body).length());
                assert!(q.velocity().unwrap().is_finite());
            }
            println!(
                "direction={index} dt={dt:.4} mass={mass} lean={:.2} brake={:.2} tracking={tracking:.4} altitude={altitude:.4} stop={:.4}",
                lean.to_degrees(),
                counterlean.to_degrees(),
                q.velocity().unwrap().length()
            );
            assert!(lean > 0.05 && counterlean < -0.05);
            assert!(tracking < 0.5 && altitude < 0.15);
            assert!(q.pose().unwrap().position.distance(start + direction * 4.0) < 0.08);
            assert!(q.velocity().unwrap().length() < 0.08);
            assert!(air_error < 1e-10);
        }
    }
}

#[test]
fn propeller_phase_integrates_actual_motor_speed_without_phase_jumps() {
    let mut drone = Quadrotor::new(Vec3::Y * 3.0, Default::default()).unwrap();
    let mut phase = [0.0; 4];
    for step in 0..480 {
        if step < 240 {
            drone.command_velocity(Vec3::X, 0.1).unwrap();
        } else {
            drone.stop_motors().unwrap();
        }
        drone.step(DT, Vec3::ZERO).unwrap();
        for i in 0..4 {
            let r = drone
                .world
                .devices
                .propellers
                .reading(drone.propellers[i])
                .unwrap();
            phase[i] = (phase[i] + r.omega * DT).rem_euclid(std::f64::consts::TAU);
            assert!((r.phase - phase[i]).abs() < 1e-10);
        }
    }
}

#[test]
fn stopped_drone_lands_on_rigid_world_ground() {
    use molla_sim::runtime::{ColliderDesc, ColliderGeometry};
    let mut drone = Quadrotor::new(Vec3::Y, QuadrotorConfig::default()).unwrap();
    let mut floor = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(20.0, 0.1, 20.0),
    });
    floor.pose.position.y = -0.1;
    drone.world.scene.insert_collider(floor).unwrap();
    drone.stop_motors().unwrap();
    for _ in 0..720 {
        drone.step(DT, Vec3::ZERO).unwrap();
    }
    let position = drone.pose().unwrap().position;
    assert!(position.y > 0.17 && position.y < 0.24, "{position:?}");
    assert!(drone.velocity().unwrap().length() < 0.1);
}

#[test]
fn quadrotor_motors_ramp_hover_translate_and_spin_down() {
    let target = Vec3::Y * 3.0;
    let mut drone = Quadrotor::new(target, QuadrotorConfig::default()).unwrap();
    drone.hold(target, 1.0).unwrap();
    drone.step(DT, Vec3::ZERO).unwrap();
    let first = drone
        .world
        .devices
        .propellers
        .reading(drone.propellers[0])
        .unwrap()
        .omega
        .abs();
    assert!((first - drone.config.motor_acceleration * DT).abs() < 1e-10);
    for _ in 0..1440 {
        drone.hold(target, 1.0).unwrap();
        drone.step(DT, Vec3::ZERO).unwrap();
    }
    assert!(
        drone.pose().unwrap().position.distance(target) < 0.03,
        "{:?}",
        drone.pose().unwrap()
    );
    let loads = drone.air_loads().unwrap();
    let force: Vec3 = loads.iter().map(|l| l.force).sum();
    assert!((force.y + drone.config.mass * 9.81).abs() < 0.3);
    let next = target + Vec3::new(0.7, 0.2, -0.5);
    for _ in 0..1440 {
        drone.hold(next, 0.7).unwrap();
        drone.step(DT, Vec3::ZERO).unwrap();
    }
    assert!(drone.pose().unwrap().position.distance(next) < 0.06);
    let before = drone.pose().unwrap().position;
    drone.stop_motors().unwrap();
    for _ in 0..120 {
        drone.step(DT, Vec3::ZERO).unwrap();
    }
    assert!(drone.pose().unwrap().position.y < before.y - 0.5);
    assert!(
        drone
            .air_loads()
            .unwrap()
            .iter()
            .all(|l| l.force == Vec3::ZERO)
    );
}

#[test]
fn moving_rotor_wrench_is_transferred_to_air_without_losing_moment() {
    let g = Grid::new([32, 32, 32], 0.2, Vec3::new(-3.2, 0.0, -3.2)).unwrap();
    let mut drone = Quadrotor::new(Vec3::Y * 2.5, QuadrotorConfig::default()).unwrap();
    for _ in 0..120 {
        drone
            .command_velocity(Vec3::new(0.4, 0.1, -0.25), 0.2)
            .unwrap();
        drone.step(DT, Vec3::new(-1.0, 0.0, 0.3)).unwrap();
    }
    let mut forces = AirForces::new(g);
    for load in drone.air_loads().unwrap() {
        load.deposit(&mut forces, 1.0).unwrap();
    }
    let (air_force, air_moment) = forces.wrench(Vec3::ZERO);
    let mut body_force = Vec3::ZERO;
    let mut body_moment = Vec3::ZERO;
    for rotor in drone.propellers {
        let r = drone.world.devices.propellers.reading(rotor).unwrap();
        body_force += r.world_force;
        body_moment += r.world_moment;
    }
    assert!((air_force + body_force).length() < 1e-10);
    assert!((air_moment + body_moment).length() < 1e-10);
    assert!(body_force.x.abs() + body_force.z.abs() > 1.0);
}

#[test]
fn flight_rejects_invalid_configuration_and_commands() {
    assert!(
        Quadrotor::new(
            Vec3::ZERO,
            QuadrotorConfig {
                mass: 1000.0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        Quadrotor::new(
            Vec3::ZERO,
            QuadrotorConfig {
                arm: f64::NAN,
                ..Default::default()
            }
        )
        .is_err()
    );
    let mut q = Quadrotor::new(Vec3::Y, QuadrotorConfig::default()).unwrap();
    assert!(q.hold(Vec3::ZERO, 0.0).is_err());
    assert!(q.command_velocity(Vec3::splat(f64::NAN), 0.0).is_err());
    assert!(q.step(0.0, Vec3::ZERO).is_err());
}

#[test]
fn heading_reversal_and_variable_steps_settle_without_teleporting() {
    use molla_solvers::airflow::FlightReference;
    let start = Vec3::Y * 3.0;
    let mut drone = Quadrotor::new(start, Default::default()).unwrap();
    let mut time = 0.0;
    let mut peak_error = 0.0_f64;
    let mut saturated = false;
    for step in 0..1800 {
        let dt = if step % 2 == 0 {
            1.0 / 120.0
        } else {
            1.0 / 60.0
        };
        let heading = if time < 4.0 { Vec3::X } else { -Vec3::X };
        drone.track(FlightReference::hold(start), heading).unwrap();
        let before = drone.pose().unwrap();
        let speed = drone.velocity().unwrap().length();
        drone.step(dt, Vec3::new(1.2, 0.0, -0.5)).unwrap();
        if time >= 4.0 {
            peak_error = peak_error.max(drone.pose().unwrap().position.distance(start));
        }
        saturated |= drone
            .world
            .devices
            .copters
            .reading(drone.controller)
            .unwrap()
            .saturated;
        assert!(drone.pose().unwrap().position.distance(before.position) < (speed + 1.0) * dt);
        time += dt;
    }
    let forward = drone.pose().unwrap().rotation * Vec3::X;
    println!(
        "reversal forward={forward:?}, pose={:?}, velocity={:?}",
        drone.pose().unwrap(),
        drone.velocity().unwrap()
    );
    assert!(forward.dot(-Vec3::X) > 0.999);
    assert!(drone.pose().unwrap().position.distance(start) < 0.02);
    assert!(drone.velocity().unwrap().length() < 0.02);
    assert!(peak_error < 0.25, "peak reversal error {peak_error}");
    assert!(saturated);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU"]
fn flying_drone_drives_the_evolving_air_field() {
    use molla_compute::Engine;
    use molla_solvers::airflow::{AirConfig, AirState, GpuAir};
    let engine = Engine::new_gpu().unwrap();
    let grid = Grid::new([32, 24, 32], 0.15, Vec3::new(-2.4, 0.0, -2.4)).unwrap();
    let mut air = GpuAir::new(&engine, &AirState::new(AirConfig::new(grid)).unwrap()).unwrap();
    let target = Vec3::new(0.0, 2.4, 0.0);
    let mut drone = Quadrotor::new(target, QuadrotorConfig::default()).unwrap();
    let mut forces = AirForces::new(grid);
    for _ in 0..120 {
        let wind = air
            .sample(&engine, &[drone.pose().unwrap().position])
            .unwrap()[0];
        forces.clear();
        for _ in 0..4 {
            drone.hold(target, 0.6).unwrap();
            drone.step(DT, wind).unwrap();
            for load in drone.air_loads().unwrap() {
                load.deposit(&mut forces, 0.25).unwrap();
            }
            let (center, force) = drone.hull_air_load().unwrap();
            forces.add(center, force * 0.25).unwrap();
        }
        air.step(&engine, DT * 4.0, &forces).unwrap();
    }
    let field = air.snapshot(&engine).unwrap();
    let disk = drone.rotor_frames().unwrap()[0].position;
    let wake = field.velocity_at(disk - Vec3::Y * 0.6).unwrap();
    println!(
        "coupled drone position {:?}, below-rotor air {wake:?}, {:?}",
        drone.pose().unwrap().position,
        field.diagnostics()
    );
    assert!(wake.y < -2.0);
    assert!(drone.pose().unwrap().position.distance(target) < 0.25);
    assert!(field.diagnostics().divergence_rms < 0.005);
}
