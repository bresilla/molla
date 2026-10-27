use super::smoke;
use super::smoke_counters::{self, Scope};
use molla_compute::Engine;
use molla_solvers::{MpmConfig, MpmRheology, MpmSimulation, MpmTimeStep};
use serde_json::json;
use std::io::Write;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    smoke_counters::verify_build()?;
    println!(
        "{}",
        smoke::manifest_with_instrumentation(
            smoke::Backend::Mpm,
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
    let device = engine.device().ok_or("missing GPU device")?;
    let queue = engine.queue().ok_or("missing GPU queue")?;
    let config = MpmConfig {
        grid_size: 32,
        cell_size: 0.0625,
        grid_origin: [-1.; 3],
        gravity_y: smoke::GRAVITY,
        youngs: 3000.,
        poisson: 0.3,
        friction_angle: 0.5,
        cohesion: 0.,
        boundary_friction: 0.5,
        particle_mass: smoke::MASS,
        particle_volume: 0.000015625,
        max_particles: smoke::PARTICLES as u32,
        max_rigid: 1,
        rheology: MpmRheology::Sand {
            friction_angle: 0.5,
        },
        time_step: MpmTimeStep::Explicit,
    };
    let mut sim = MpmSimulation::new(&engine, config)?;
    sim.set_p2g_mode(true);
    let points = smoke::points();
    let velocities = vec![smoke::VELOCITY; points.len()];
    for round in 0..=smoke::SAMPLES {
        sim.set_particles_with_velocity(queue, &points, &velocities)?;
        queue.submit([]);
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(15)),
        })?;
        let counters = Scope::start()?;
        let start = std::time::Instant::now();
        sim.step_n_with(device, queue, smoke::DT, smoke::STEPS as usize)?;
        let encode_ms = start.elapsed().as_secs_f64() * 1000.;
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(15)),
        })?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.;
        let counts = counters.finish()?;
        engine.check_device()?;
        let positions = sim.read_positions_with(device, queue)?;
        let velocities = sim.read_velocities_with(device, queue)?;
        let (mass, _) = sim.read_grid_mass_momentum(&engine)?;
        let errors = smoke::physical_errors(&positions, &velocities, &mass)?;
        if round > 0 {
            println!(
                "{}",
                json!({"record":"sample","case":"smoke","sample":round-1,
                "wall_ms":wall_ms,"encode_submit_ms":encode_ms,"simulated_seconds":f64::from(smoke::DT)*f64::from(smoke::STEPS),
                "position_error":errors[0],"velocity_error":errors[1],"mass_relative_error":errors[2],
                "allocation_observations":counts.allocations,"readback_bytes":counts.readbacks,
                "upload_bytes":counts.uploads,"dispatches":counts.dispatches})
            );
            std::io::stdout().flush()?;
        }
        if errors.iter().any(|error| *error > smoke::TOLERANCE) || counts.failed() {
            return Err(format!("smoke physical/resource gate failed: {errors:?}").into());
        }
    }
    println!(
        "{}",
        json!({"record":"complete","case":"smoke","samples":smoke::SAMPLES})
    );
    Ok(())
}
