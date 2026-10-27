use super::smoke;
use super::smoke_counters::{self, Scope};
use molla_compute::Engine;
use molla_kernels::{
    TireDemGpuKernel, TireDemResidentFemTerrainOwner, TireDemResidentTerrainParams,
    TireDemStepParams, tire_dem_raster_surface_height_map,
};
use serde_json::json;
use std::io::Write;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    smoke_counters::verify_build()?;
    println!(
        "{}",
        smoke::manifest_with_instrumentation(
            smoke::Backend::Dem,
            cfg!(feature = "resource-counters")
        )
    );
    std::io::stdout().flush()?;
    let engine = Engine::new_gpu()?;
    let adapter = engine.adapter_info().ok_or("missing GPU adapter")?;
    if adapter.is_software || adapter.backend != "Vulkan" {
        return Err("real Vulkan GPU required".into());
    }
    println!("adapter={:?}", engine.adapter_info());
    let device = engine.device().ok_or("missing device")?;
    let queue = engine.queue().ok_or("missing queue")?;
    let kernel = TireDemGpuKernel::new(device);
    let terrain = TireDemResidentTerrainParams {
        origin: [-1.; 2],
        spacing: 0.125,
        nx: 16,
        nz: 16,
        floor_y: -1.,
        grain_radius: 0.01,
    };
    let params = TireDemStepParams {
        grain_mass: smoke::MASS,
        grain_radius: 0.01,
        normal_stiffness: 10000.,
        damping: 80.,
        friction: 0.65,
        floor_y: -1.,
        gravity: [0., smoke::GRAVITY, 0.],
        dt: smoke::DT,
        tire_position_push_scale: 0.2,
    };
    let positions: Vec<_> = smoke::points()
        .iter()
        .map(|p| [p[0], p[1], p[2], 0.])
        .collect();
    let velocities = vec![
        [
            smoke::VELOCITY[0],
            smoke::VELOCITY[1],
            smoke::VELOCITY[2],
            0.
        ];
        smoke::PARTICLES
    ];
    let wait = || {
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(15)),
        })
    };
    for round in 0..=smoke::SAMPLES {
        let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
            device,
            queue,
            &kernel,
            terrain,
            &positions,
            &velocities,
        )?;
        owner.set_lbvh_rebuild_interval(1)?;
        owner.step_sparse_lbvh_gpu_resident(device, queue, &kernel, &[], params, 8)?;
        wait()?;
        if owner.state_revision() != 1 {
            return Err("unexpected preparation revision".into());
        }
        let counters = Scope::start()?;
        let start = std::time::Instant::now();
        for _ in 0..smoke::STEPS {
            owner.step_sparse_lbvh_gpu_resident(device, queue, &kernel, &[], params, 8)?;
        }
        let encode_ms = start.elapsed().as_secs_f64() * 1000.;
        wait()?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.;
        let counts = counters.finish()?;
        engine.check_device()?;
        if owner.state_revision() != u64::from(smoke::STEPS + 1) || owner.mirror_revision() != 0 {
            return Err("DEM revision/observation contract failed".into());
        }
        owner.refresh_mirror(device, queue, &kernel)?;
        let positions: Vec<_> = owner
            .positions()
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        let velocities: Vec<_> = owner
            .velocities()
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        let errors = smoke::kinematic_errors(&positions, &velocities, smoke::STEPS + 1)?;
        let raster = kernel.readback_resident_terrain(device, queue, owner.resident_terrain())?;
        let expected = tire_dem_raster_surface_height_map(
            owner.positions(),
            terrain.origin,
            terrain.spacing,
            terrain.nx,
            terrain.nz,
            terrain.floor_y,
            terrain.grain_radius,
        )?;
        if raster.heights.len() != 256
            || expected.len() != 256
            || raster.heights.iter().any(|v| !v.is_finite())
            || raster.sinkage.len() != 256
            || raster.sinkage.iter().any(|v| !v.is_finite())
            || raster.surface_velocity_map.rows.len() != 256
            || raster
                .surface_velocity_map
                .rows
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err("invalid DEM terrain snapshot".into());
        }
        let terrain_error = raster
            .heights
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0., f64::max);
        if round > 0 {
            println!(
                "{}",
                json!({"record":"sample","case":"smoke","sample":round-1,"wall_ms":wall_ms,"encode_submit_ms":encode_ms,
                "simulated_seconds":f64::from(smoke::DT)*f64::from(smoke::STEPS),"position_error":errors[0],"velocity_error":errors[1],"mass_relative_error":errors[2],
                "allocation_observations":counts.allocations,"readback_bytes":counts.readbacks,"upload_bytes":counts.uploads,"dispatches":counts.dispatches,
                "preparation_substeps":1,"completed_substeps":smoke::STEPS,"final_revision":owner.state_revision(),"terrain_cells":raster.heights.len(),"terrain_height_error":terrain_error})
            );
            std::io::stdout().flush()?;
        }
        if errors.iter().any(|v| *v > smoke::TOLERANCE) || terrain_error > 1e-6 || counts.failed() {
            return Err(format!(
                "DEM smoke physical/resource gate failed: {errors:?}, terrain={terrain_error}"
            )
            .into());
        }
    }
    println!(
        "{}",
        json!({"record":"complete","case":"smoke","samples":smoke::SAMPLES})
    );
    Ok(())
}
