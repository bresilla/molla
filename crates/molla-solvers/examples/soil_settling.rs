use molla_compute::Engine;
use molla_solvers::mpm::*;
use molla_vehicle::{
    SoilMaterialId, SoilParticleId, SoilRegionId, SoilRegionRequest, SoilResidencyLimits,
    SoilTileLayout,
};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[path = "soil_support/settling_resume.rs"]
mod resume;

fn output(args: &[String]) -> Result<(PathBuf, Option<PathBuf>)> {
    let (mut output, mut resume) = (None, None);
    if !args.len().is_multiple_of(2) {
        return Err("missing settling option value".into());
    }
    for pair in args.chunks_exact(2) {
        let slot = match pair[0].as_str() {
            "--output" => &mut output,
            "--resume-run" => &mut resume,
            _ => return Err("unknown settling option".into()),
        };
        if slot.is_some() || pair[1].is_empty() {
            return Err("duplicate or empty settling option".into());
        }
        *slot = Some(PathBuf::from(&pair[1]));
    }
    Ok((output.ok_or("--output NEW_DIRECTORY is required")?, resume))
}

fn policy_json(policy: MpmSoilSettlingConfig) -> Value {
    json!({"dt_s":policy.dt,"substeps_per_observation":policy.substeps_per_observation,
        "max_substeps":policy.max_substeps,"quiet_seconds":policy.quiet_seconds,
        "quiet_observations":policy.quiet_observations,"wall_limit_s":policy.wall_limit.as_secs_f64(),
        "limits":{"speed_m_s":policy.limits.speed,"affine_speed_m_s":policy.limits.affine_speed,
            "displacement_rate_m_s":policy.limits.displacement_rate,"acceleration_m_s2":policy.limits.acceleration,
            "elastic_rate_s_inv":policy.limits.elastic_rate,"plastic_rate_s_inv":policy.limits.plastic_rate,
            "tire_history_rates":policy.limits.tire_history_rates}})
}

fn workload() -> (MpmConfig, Vec<[f32; 3]>, MpmSoilSettlingConfig) {
    let spacing = 0.025_f32;
    let volume = spacing.powi(3);
    let config = MpmConfig {
        max_particles: 512,
        max_rigid: 1,
        youngs: 3000.0,
        particle_volume: volume,
        particle_mass: 1500.0 * volume,
        ..MpmConfig::sand()
    };
    let positions = (0..512)
        .map(|i| {
            [
                -0.0875 + (i % 8) as f32 * spacing,
                0.125 + (i / 8 % 8) as f32 * spacing,
                -0.0875 + (i / 64) as f32 * spacing,
            ]
        })
        .collect();
    let policy = MpmSoilSettlingConfig {
        dt: 0.0005,
        substeps_per_observation: 16,
        max_substeps: 2000,
        quiet_seconds: 0.05,
        quiet_observations: 5,
        wall_limit: Duration::from_secs(20),
        limits: MpmSoilSettlingLimits {
            speed: 0.001,
            affine_speed: 0.001,
            displacement_rate: 0.001,
            acceleration: 0.01,
            elastic_rate: 0.001,
            plastic_rate: 0.001,
            tire_history_rates: [0.0001; 4],
        },
    };
    (config, positions, policy)
}

fn activity(a: &MpmSoilActivity) -> Value {
    json!({"particles":a.particles,"substeps":a.substeps,"seconds":a.seconds,
        "particle_ids_hash":a.particle_ids_hash,"max_speed_m_s":a.max_speed,
        "mass_weighted_rms_speed_m_s":a.rms_speed,"max_affine_speed_m_s":a.max_affine_speed,
        "translational_kinetic_energy_j":a.kinetic_energy,
        "max_displacement_rate_m_s":a.max_displacement_rate,"max_acceleration_m_s2":a.max_acceleration,
        "max_elastic_rate_s_inv":a.max_elastic_rate,"max_plastic_rate_s_inv":a.max_plastic_rate,
        "max_tire_history_rates":a.max_tire_history_rates})
}

fn progress(p: &MpmSoilSettlingProgress) -> Value {
    json!({"status":format!("{:?}",p.status),"submitted_substeps":p.submitted_substeps,
        "observations":p.observations,"observed_seconds":p.observed_seconds,
        "quiet_seconds":p.quiet_seconds,"quiet_observations":p.quiet_observations,
        "wall_seconds":p.wall_seconds,"failure":p.failure,
        "latest":p.latest.as_ref().map(|r| json!({"revision":r.revision,"activity":activity(&r.activity)}))})
}

fn exit_code(status: MpmSoilSettlingStatus) -> u8 {
    if status == MpmSoilSettlingStatus::ActivityConverged {
        0
    } else {
        2
    }
}

fn run() -> Result<u8> {
    let (directory, resume_path) = output(&std::env::args().skip(1).collect::<Vec<_>>())?;
    fs::create_dir(&directory)?;
    let (config, positions, policy) = workload();
    policy.validate()?;
    let resumed = resume_path
        .as_ref()
        .map(|p| resume::load(p, config, policy))
        .transpose()?;
    let engine = Engine::new_gpu()?;
    let adapter = engine.adapter_info().ok_or("missing GPU adapter")?;
    if adapter.is_software || adapter.backend != "Vulkan" {
        return Err("real Vulkan GPU required".into());
    }
    let ids: Vec<_> = (0..512).map(SoilParticleId).collect();
    let start = if let Some(start) = resumed {
        start
    } else {
        let mut seed = MpmSimulation::new_sparse(&engine, config, 128)?;
        seed.set_particles(engine.queue().unwrap(), &positions)?;
        let checkpoint = seed.checkpoint_soil_with(engine.queue().unwrap())?;
        let archive = checkpoint.partition_tiles(
            SoilTileLayout::new([0.0; 3], 0.125)?,
            &ids,
            &[SoilMaterialId(1)],
        )?;
        resume::Start {
            archive,
            checkpoint,
            preceding_steps: 0,
            lineage: json!({"kind":"fresh"}),
            source_files: Vec::new(),
        }
    };
    let preceding_steps = start.preceding_steps;
    let lineage = start.lineage;
    for (name, bytes) in start.source_files {
        fs::write(directory.join(name), bytes)?;
    }
    let initial = start.checkpoint;
    let archive = start.archive;
    fs::write(directory.join("initial-archive.bin"), archive.to_bytes())?;
    let manifest = json!({"schema":"molla-soil-settling-v2","fixture":"gravity-bed-8x8x8-v1",
        "segment_start_substeps":preceding_steps,"segment_start_seconds":preceding_steps as f64*f64::from(policy.dt),
        "lineage":lineage,
        "physically_calibrated":false,"performance_acceptance":false,
        "adapter":format!("{adapter:?}"),"config":format!("{config:?}"),
        "particle_count":positions.len(),"spacing_m":0.025_f32,"density_kg_m3":1500.0,
        "grid_size":config.grid_size,"cell_size_m":config.cell_size,"grid_origin_m":config.grid_origin,
        "gravity_y_m_s2":config.gravity_y,"youngs_pa":config.youngs,"poisson":config.poisson,
        "boundary_friction":config.boundary_friction,"particle_mass_kg":config.particle_mass,
        "particle_volume_m3":config.particle_volume,"policy":policy_json(policy),
        "scope":"Full finite bed preparation, no machine/implement or renderer; endpoint criteria are not force equilibrium or sleep authorization."});
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let initial_bytes = initial.to_bytes();
    fs::write(directory.join("initial-checkpoint.bin"), &initial_bytes)?;
    let tiles: Vec<_> = archive.tiles().collect();
    let mut world = MpmSoilWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, config, 128)?,
        archive,
        SoilResidencyLimits {
            regions: 1,
            tiles: 4096,
            claims: 8192,
        },
        512 * 160 + 16,
    )?;
    world.begin_region_update(&[SoilRegionRequest {
        id: SoilRegionId(1),
        tiles: &tiles,
    }])?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !world.poll_region_update(&engine)? {
        if Instant::now() >= deadline {
            return Err("initial full-bed capture deadline".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let mut samples = fs::File::create_new(directory.join("activity.jsonl"))?;
    let final_progress = {
        let mut driver = MpmSoilSettling::memory(&mut world, policy)?;
        let mut observed = 0;
        loop {
            let status = driver.poll(&engine)?;
            if driver.progress().observations != observed {
                observed = driver.progress().observations;
                writeln!(samples, "{}", progress(driver.progress()))?;
                samples.flush()?;
            }
            if status.is_terminal() {
                break driver.progress().clone();
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    let bytes = world.synchronized_archive_bytes()?;
    let final_archive = MpmSoilTileArchive::from_bytes(&bytes, 512)?;
    let final_tiles: Vec<_> = final_archive.tiles().collect();
    let selected = final_archive.select(&final_tiles, 512)?;
    if selected.particle_ids() != ids || world.active_particle_count() != 512 {
        return Err("settling changed the full particle population".into());
    }
    let whole_run = if final_progress.submitted_substeps == 0 {
        None
    } else {
        Some(selected.checkpoint().activity_since(
            &initial,
            &ids,
            selected.particle_ids(),
            final_progress.observed_seconds,
            final_progress.submitted_substeps,
        )?)
    };
    let final_checkpoint = selected.checkpoint().to_bytes();
    fs::write(directory.join("final-archive.bin"), &bytes)?;
    fs::write(directory.join("final-checkpoint.bin"), &final_checkpoint)?;
    let cumulative_steps = preceding_steps
        .checked_add(final_progress.submitted_substeps)
        .ok_or("settling cumulative clock overflow")?;
    let result = json!({"schema":"molla-soil-settling-result-v2","workflow_completed":true,
        "segment_start_substeps":preceding_steps,"cumulative_substeps":cumulative_steps,
        "cumulative_seconds":cumulative_steps as f64*f64::from(policy.dt),"lineage":lineage,
        "activity_converged":final_progress.status==MpmSoilSettlingStatus::ActivityConverged,
        "physically_calibrated":false,"performance_accepted":false,
        "progress":progress(&final_progress),"whole_run_activity":whole_run.as_ref().map(activity),
        "initial_checkpoint_blake3":blake3::hash(&initial_bytes).to_hex().to_string(),
        "final_checkpoint_blake3":blake3::hash(&final_checkpoint).to_hex().to_string(),
        "final_archive_blake3":blake3::hash(&bytes).to_hex().to_string(),
        "identity_mass_volume_material_preserved":true});
    fs::write(
        directory.join("result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{result}");
    Ok(exit_code(final_progress.status))
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(code) => std::process::ExitCode::from(code),
        Err(error) => {
            eprintln!("soil settling failed: {error}");
            std::process::ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture_is_full_nonpenetrating_bed_with_frozen_mass_and_policy() {
        let (cfg, points, policy) = workload();
        policy.validate().unwrap();
        assert_eq!(points.len(), 512);
        assert_eq!(cfg.max_particles, 512);
        assert!((f64::from(cfg.particle_mass / cfg.particle_volume) - 1500.0).abs() < 0.001);
        assert!(
            points
                .iter()
                .all(|p| (0..3).all(|k| p[k] > cfg.grid_origin[k] + 2.0 * cfg.cell_size))
        );
        let keys: std::collections::BTreeSet<_> =
            points.iter().map(|p| p.map(f32::to_bits)).collect();
        assert_eq!(keys.len(), 512);
        assert_eq!(policy.max_substeps, 2000);
        assert_eq!(policy.substeps_per_observation, 16);
    }
    #[test]
    fn only_activity_convergence_has_a_success_exit() {
        assert_eq!(exit_code(MpmSoilSettlingStatus::ActivityConverged), 0);
        for status in [
            MpmSoilSettlingStatus::Running,
            MpmSoilSettlingStatus::DrainingAfterWallLimit,
            MpmSoilSettlingStatus::StepLimit,
            MpmSoilSettlingStatus::WallLimit,
            MpmSoilSettlingStatus::Failed,
        ] {
            assert_eq!(exit_code(status), 2);
        }
    }
    #[test]
    fn output_parser_rejects_missing_duplicate_and_unknown_arguments() {
        for args in [
            vec![],
            vec!["--output"],
            vec!["--other", "x"],
            vec!["--output", ""],
            vec!["--output", "x", "--output", "y"],
        ] {
            assert!(output(&args.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err());
        }
        assert_eq!(
            output(&["--output".into(), "new directory".into()]).unwrap(),
            (PathBuf::from("new directory"), None)
        );
        assert_eq!(
            output(&[
                "--resume-run".into(),
                "parent".into(),
                "--output".into(),
                "child".into()
            ])
            .unwrap(),
            (PathBuf::from("child"), Some(PathBuf::from("parent")))
        );
        assert!(output(&["--resume-run".into(), "parent".into()]).is_err());
    }
}
