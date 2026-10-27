use super::super::tests::{limits, make_archive};
use super::*;
use crate::mpm::checkpoint::tests::fixture;
use std::sync::atomic::{AtomicU64, Ordering};

fn config() -> MpmSoilSettlingConfig {
    MpmSoilSettlingConfig {
        dt: 0.0001,
        substeps_per_observation: 1,
        max_substeps: 5,
        quiet_seconds: 2.0 * f64::from(0.0001_f32),
        quiet_observations: 2,
        wall_limit: Duration::from_secs(30),
        limits: MpmSoilSettlingLimits {
            speed: 1e-7,
            affine_speed: 1e-7,
            displacement_rate: 1e-7,
            acceleration: 1e-7,
            elastic_rate: 1e-7,
            plastic_rate: 1e-7,
            tire_history_rates: [1e-7; 4],
        },
    }
}

fn state(gravity: f32) -> MpmSoilCheckpoint {
    let mut state = fixture(2);
    state.cfg.gravity_y = gravity;
    for s in [1, 2, 6, 8] {
        let strides = super::super::super::STRIDES;
        let start = strides[..s].iter().sum::<usize>() * state.count as usize;
        let end = start + strides[s] * state.count as usize;
        state.words[start..end].fill(0);
    }
    for matrix in state.words[40..64].chunks_exact_mut(12) {
        matrix.fill(0);
        for k in [0, 5, 10] {
            matrix[k] = 1.0_f32.to_bits();
        }
    }
    state
}

#[test]
fn settling_policy_requires_explicit_finite_bounded_observation_settings() {
    let base = config();
    base.validate().unwrap();
    for bad in [
        MpmSoilSettlingConfig { dt: 0.0, ..base },
        MpmSoilSettlingConfig {
            dt: f32::NAN,
            ..base
        },
        MpmSoilSettlingConfig {
            substeps_per_observation: 0,
            ..base
        },
        MpmSoilSettlingConfig {
            substeps_per_observation: 1025,
            ..base
        },
        MpmSoilSettlingConfig {
            max_substeps: 0,
            ..base
        },
        MpmSoilSettlingConfig {
            quiet_seconds: 0.0,
            ..base
        },
        MpmSoilSettlingConfig {
            quiet_seconds: f64::INFINITY,
            ..base
        },
        MpmSoilSettlingConfig {
            quiet_observations: 1,
            ..base
        },
        MpmSoilSettlingConfig {
            wall_limit: Duration::ZERO,
            ..base
        },
        MpmSoilSettlingConfig {
            limits: MpmSoilSettlingLimits {
                speed: -1.0,
                ..base.limits
            },
            ..base
        },
        MpmSoilSettlingConfig {
            limits: MpmSoilSettlingLimits {
                tire_history_rates: [f64::NAN; 4],
                ..base.limits
            },
            ..base
        },
    ] {
        assert!(bad.validate().is_err());
    }
    for (active, total, horizon, boundary) in [
        (0, 0, None, true),
        (1, 2, None, true),
        (2, 2, Some(1), true),
        (2, 2, None, false),
    ] {
        assert!(MpmSoilSettling::validate_world(active, total, horizon, boundary).is_err());
    }
    MpmSoilSettling::validate_world(2, 2, None, true).unwrap();
}

#[test]
fn settling_cutoffs_reject_each_activity_channel_and_empty_populations() {
    let state = state(0.0);
    let ids = [
        molla_vehicle::SoilParticleId(0),
        molla_vehicle::SoilParticleId(1),
    ];
    let quiet = state.activity_since(&state, &ids, &ids, 1.0, 1).unwrap();
    assert!(config().limits.accepts(&quiet));
    for k in 0..10 {
        for value in [1.0, f64::NAN, -1.0] {
            let mut active = quiet.clone();
            match k {
                0 => active.max_speed = value,
                1 => active.max_affine_speed = value,
                2 => active.max_displacement_rate = value,
                3 => active.max_acceleration = value,
                4 => active.max_elastic_rate = value,
                5 => active.max_plastic_rate = value,
                _ => active.max_tire_history_rates[k - 6] = value,
            }
            assert!(!config().limits.accepts(&active));
        }
    }
    let mut empty = quiet;
    empty.particles = 0;
    assert!(!config().limits.accepts(&empty));
    assert!(!MpmSoilSettlingStatus::DrainingAfterWallLimit.is_terminal());
    assert!(MpmSoilSettlingStatus::StepLimit.is_terminal());
    assert!(MpmSoilSettlingStatus::WallLimit.is_terminal());
    assert!(MpmSoilSettlingStatus::Failed.is_terminal());
}

struct Directory(std::path::PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "molla-settling-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn memory(engine: &Engine, gravity: f32) -> MpmSoilWorld {
    let state = state(gravity);
    let archive = make_archive(&state);
    let keys: Vec<_> = archive.tiles().collect();
    let mut world = MpmSoilWorld::new(
        engine,
        MpmSimulation::new_sparse(engine, state.cfg, 128).unwrap(),
        archive,
        limits(),
        20656,
    )
    .unwrap();
    world
        .begin_region_update(&[SoilRegionRequest {
            id: SoilRegionId(1),
            tiles: &keys,
        }])
        .unwrap();
    drain_memory(&mut world, engine);
    world
}

fn drain_memory(world: &mut MpmSoilWorld, engine: &Engine) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !world.poll_region_update(engine).unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn run(driver: &mut MpmSoilSettling<'_>, engine: &Engine) -> MpmSoilSettlingProgress {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !driver.poll(engine).unwrap().is_terminal() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    println!("settling progress {:?}", driver.progress());
    driver.progress().clone()
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn numerical_fault_cannot_be_reported_as_quiet_soil() {
    let engine = Engine::new_gpu().unwrap();
    let mut world = memory(&engine, 0.0);
    engine.queue().unwrap().write_buffer(
        &world.simulation.p_f,
        4,
        bytemuck::bytes_of(&f32::NAN.to_bits()),
    );
    let mut driver = MpmSoilSettling::memory(&mut world, config()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match driver.poll(&engine) {
            Err(_) => break,
            Ok(status) => assert!(
                !status.is_terminal(),
                "fault cannot converge or exhaust its budget"
            ),
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let progress = driver.progress();
    assert_eq!(progress.status, MpmSoilSettlingStatus::Failed);
    assert_eq!(progress.observations, 0);
    assert_eq!(progress.quiet_observations, 0);
    assert!(progress.latest.is_none());
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn full_bed_settling_distinguishes_quiet_and_gravity_driven_motion_in_both_owners() {
    let engine = Engine::new_gpu().unwrap();
    for gravity in [0.0, -9.81] {
        let policy = MpmSoilSettlingConfig {
            substeps_per_observation: if gravity == 0.0 { 1 } else { 2 },
            ..config()
        };
        let expected = if gravity == 0.0 {
            MpmSoilSettlingStatus::ActivityConverged
        } else {
            MpmSoilSettlingStatus::StepLimit
        };
        let mut world = memory(&engine, gravity);
        let progress = run(
            &mut MpmSoilSettling::memory(&mut world, policy).unwrap(),
            &engine,
        );
        assert_eq!(progress.status, expected);
        assert_eq!(world.active_particle_count(), 2);
        assert_eq!(world.status(), MpmSoilWorldStatus::Ready);
        assert!(world.synchronized_archive_bytes().is_ok());
        let directory = Directory::new();
        let state = state(gravity);
        let archive = make_archive(&state);
        let keys: Vec<_> = archive.tiles().collect();
        let mut store = MpmSoilStore::open(
            &directory.0,
            MpmSoilStoreLimits {
                particles: 2,
                tiles: 32,
            },
        )
        .unwrap();
        store.save(&archive).unwrap();
        let mut disk = MpmSoilDiskWorld::new(
            &engine,
            MpmSimulation::new_sparse(&engine, state.cfg, 128).unwrap(),
            store,
            limits(),
            MpmSoilStoreReadLimits {
                particles: 2,
                tiles: 32,
                encoded_bytes: 32 * 1024 + 2 * 168,
            },
            20656,
        )
        .unwrap();
        disk.begin_region_update(&[SoilRegionRequest {
            id: SoilRegionId(1),
            tiles: &keys,
        }])
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !disk.poll_region_update(&engine).unwrap() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        let disk_progress = run(
            &mut MpmSoilSettling::disk(&mut disk, policy).unwrap(),
            &engine,
        );
        assert_eq!(disk_progress.status, expected);
        assert_eq!(
            disk_progress.submitted_substeps,
            progress.submitted_substeps
        );
        assert_eq!(disk.status(), MpmSoilDiskWorldStatus::Ready);
        assert_eq!(disk.active_particle_count(), 2);
        if gravity != 0.0 {
            assert_eq!(progress.submitted_substeps, 5);
            assert_eq!(progress.observations, 3);
            assert_eq!(progress.quiet_observations, 0);
            assert!(progress.latest.unwrap().activity.max_acceleration > 1.0);
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn settling_deadlines_drain_submitted_boundaries_and_drop_does_not_cancel() {
    let engine = Engine::new_gpu().unwrap();
    let mut world = memory(&engine, 0.0);
    {
        let mut driver = MpmSoilSettling::memory(&mut world, config()).unwrap();
        driver.started = Instant::now() - Duration::from_secs(31);
        assert_eq!(
            driver.poll(&engine).unwrap(),
            MpmSoilSettlingStatus::WallLimit
        );
        assert_eq!(driver.progress().submitted_substeps, 0);
        assert_eq!(driver.progress().observations, 0);
    }
    {
        let mut driver = MpmSoilSettling::memory(&mut world, config()).unwrap();
        assert_eq!(
            driver.poll(&engine).unwrap(),
            MpmSoilSettlingStatus::Running
        );
        driver.started = Instant::now() - Duration::from_secs(31);
        let result = run(&mut driver, &engine);
        assert_eq!(result.status, MpmSoilSettlingStatus::WallLimit);
        assert_eq!(result.submitted_substeps, 1);
        assert_eq!(result.observations, 1);
    }
    {
        let mut driver = MpmSoilSettling::memory(&mut world, config()).unwrap();
        driver.poll(&engine).unwrap();
    }
    assert_eq!(world.status(), MpmSoilWorldStatus::Capturing);
    assert!(world.step_n(&engine, 0.0001, 1).is_err());
    drain_memory(&mut world, &engine);
    assert!(world.synchronized_archive_bytes().is_ok());
    world.step_n(&engine, 0.0001, 1).unwrap();
    assert!(MpmSoilSettling::memory(&mut world, config()).is_err());
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn settling_foreign_engine_failure_retains_the_borrowed_world() {
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let mut world = memory(&engine, 0.0);
    let bytes = world.synchronized_archive_bytes().unwrap();
    {
        let mut driver = MpmSoilSettling::memory(&mut world, config()).unwrap();
        assert!(driver.poll(&foreign).is_err());
        assert_eq!(driver.progress().status, MpmSoilSettlingStatus::Failed);
        assert!(driver.progress().failure.is_some());
        assert_eq!(driver.progress().submitted_substeps, 0);
    }
    assert_eq!(world.synchronized_archive_bytes().unwrap(), bytes);
    assert_eq!(world.status(), MpmSoilWorldStatus::Ready);
}
