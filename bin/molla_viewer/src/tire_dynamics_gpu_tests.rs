use super::*;
use bevy::ecs::system::RunSystemOnce;

#[path = "tire_fem_stage_probe.rs"]
mod stages;

#[path = "tire_fem_dihedral_tests.rs"]
mod dihedral;

fn world() -> World {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let (model, _) = Sim::build_scene_model(Scene::TireDynamics);
    let sim = Sim {
        scene: Scene::TireDynamics,
        sensor_clock: ViewerSensorClock::default(),
        state_a: model.state().unwrap(),
        state_b: model.state().unwrap(),
        control: model.control(),
        contacts: mp::Contacts::with_capacity(0),
        solver: SolverBackend::Xpbd(mp::XpbdSolver::new_cpu()),
        pipeline: CollisionPipeline::new(),
        model,
        engine,
        adapter_label: "headless tire update fixture".into(),
        mpm: None,
        mpm_particles: Vec::new(),
        mpm_rigid_spheres: Vec::new(),
    };
    let mut world = World::new();
    let mut time = Time::<()>::default();
    time.advance_by(std::time::Duration::from_secs_f64(1. / 120.));
    world.insert_resource(time);
    world.insert_resource(sim);
    world.insert_resource(SimRunState::default());
    world
}

fn advance(world: &mut World, state: TireDynamicsState) -> TireDynamicsState {
    world.insert_resource(state);
    world.run_system_once(update_tire_dynamics_example).unwrap();
    world.remove_resource().unwrap()
}

fn observe(world: &World, state: &mut TireDynamicsState, frame: u32) -> Vec<f32> {
    assert_eq!(state.last_gpu_error, None, "frame {frame}");
    assert_eq!(
        state.last_backend,
        "GPU tetra FEM (resident vehicle+FEM owner)"
    );
    assert!(state.last_resident_vehicle_valid, "frame {frame}");
    assert_eq!(state.last_brush_wheels, 5);
    assert!(!state.last_gpu_delta_valid);
    let data = state
        .prepared_readback
        .as_mut()
        .unwrap()
        .read_blocking(
            &world.resource::<Sim>().engine,
            state.prepared_chassis.as_ref().unwrap(),
            state.resident_vehicle_fem.as_ref().unwrap(),
        )
        .unwrap();
    assert_eq!(
        data.cavity_status(),
        state.resident_cavity.then_some([0, 1, frame + 1, 0])
    );
    tire_dynamics_validate_frame(&state.tires, data).unwrap();
    let mut values = Vec::new();
    for (i, tire) in state.tires.iter().enumerate() {
        let snapshot = tire.gpu_snapshot();
        assert_eq!(snapshot.positions, data.positions(i).unwrap());
        assert_eq!(snapshot.velocities, data.velocities(i).unwrap());
        for rows in [data.positions(i).unwrap(), data.velocities(i).unwrap()] {
            values.extend(rows.iter().flatten().copied());
        }
    }
    for rows in [
        data.hub_rows(),
        data.planar_rows(),
        data.spin_rows(),
        data.suspension_rows(),
    ] {
        values.extend(rows.iter().flatten().copied());
    }
    values.extend(data.vertical_row());
    values
}

fn difference(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0., f64::max)
}

#[test]
#[ignore = "native GPU; actual five-tire viewer update and fallback, without rendering"]
fn resident_cavity_viewer_trajectory_and_fallback() {
    let mut world = world();
    let mut control = TireDynamicsState::default();
    let mut repeated = TireDynamicsState::default();
    let mut candidate = TireDynamicsState {
        resident_cavity: true,
        ..Default::default()
    };
    let mut maximum = 0f64;
    for frame in 0..6 {
        control = advance(&mut world, control);
        repeated = advance(&mut world, repeated);
        candidate = advance(&mut world, candidate);
        let a = observe(&world, &mut control, frame);
        let b = observe(&world, &mut repeated, frame);
        let c = observe(&world, &mut candidate, frame);
        assert_eq!(a, b, "reference repeat frame {frame}");
        let delta = difference(&a, &c);
        maximum = maximum.max(delta);
        println!(
            "real viewer frame {frame}: {} physical words; repeat exact; resident-vs-held max {delta:e}",
            a.len()
        );
    }
    let mut expected = candidate.tires.clone();
    let owner = candidate.resident_vehicle_fem.as_ref().unwrap();
    world
        .resource::<Sim>()
        .engine
        .queue()
        .unwrap()
        .write_buffer(
            owner.fem().hub_state_buffer(),
            4 * 32,
            &f32::NAN.to_le_bytes(),
        );
    candidate = advance(&mut world, candidate);
    assert_eq!(candidate.last_backend, "CPU fallback (GPU error)");
    assert!(
        candidate
            .last_gpu_error
            .as_ref()
            .unwrap()
            .contains("cavity frame not admitted")
    );
    assert!(!candidate.last_resident_vehicle_valid);
    assert!(candidate.resident_vehicle_fem.is_none());
    assert!(candidate.prepared_chassis.is_none());
    assert!(candidate.prepared_readback.is_none());
    expected[4].set_pressure(candidate.pressures_pa[4]);
    for (i, (expected, actual)) in expected.iter_mut().zip(&candidate.tires).enumerate() {
        expected.step(1. / 120.);
        assert_eq!(expected.nodes(), actual.nodes(), "fallback nodes tire {i}");
        assert_eq!(expected.hub(), actual.hub(), "fallback hub tire {i}");
        assert_eq!(
            expected.gpu_snapshot().velocities,
            actual.gpu_snapshot().velocities,
            "fallback velocity tire {i}"
        );
    }
    println!(
        "invalid fifth cavity: whole-frame rejection; all five CPU fallbacks exact from pre-frame mirrors"
    );
    assert!(
        maximum <= 2e-5,
        "real-tire trajectory max {maximum:e} exceeds unchanged 2e-5 gate"
    );
}

fn read_floats(engine: &Engine, source: &wgpu::Buffer) -> Vec<f32> {
    let device = engine.device().unwrap();
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    stage
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = stage.slice(..).get_mapped_range();
    let values = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    drop(data);
    stage.unmap();
    values
}

fn physical(frame: &molla::kernels::TireVehicleFemFrame) -> Vec<f32> {
    let mut values = Vec::new();
    for i in 0..5 {
        for rows in [frame.positions(i).unwrap(), frame.velocities(i).unwrap()] {
            values.extend(rows.iter().flatten().copied());
        }
    }
    for rows in [
        frame.hub_rows(),
        frame.planar_rows(),
        frame.spin_rows(),
        frame.suspension_rows(),
    ] {
        values.extend(rows.iter().flatten().copied());
    }
    values.extend(frame.vertical_row());
    values
}

#[test]
#[ignore = "native GPU; isolates held cavity load arithmetic from first-frame dispatch"]
fn resident_cavity_first_frame_held_load_isolation() {
    let mut world = world();
    world.resource_mut::<SimRunState>().paused = true;
    let mut state = advance(&mut world, TireDynamicsState::default());
    let elapsed = world.resource::<Time>().delta_secs();
    state.tires[4].set_pressure((elapsed as f64 / 12.) * 2. * 350_000.);
    let engine = &world.resource::<Sim>().engine;
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let fem = molla::kernels::TireFemGpuKernel::new(device);
    let vehicle = molla::kernels::TireVehicleYawGpuKernel::new(device);
    let brush = molla::kernels::TireBrushGpuKernel::new(device);
    let dt = 1. / 2880.;
    let inputs: Vec<_> = state
        .tires
        .iter()
        .map(|t| build_tire_fem_gpu_input(t, dt))
        .collect();
    let params: Vec<_> = state
        .tires
        .iter()
        .map(|t| tire_fem_gpu_params(t, dt))
        .collect();
    let hubs: Vec<_> = state
        .tires
        .iter()
        .map(|t| tire_fem_hub_params(t, dt))
        .collect();
    let create = || {
        let owner = fem
            .create_resident_vehicle_fem_atlas_owner(
                &vehicle,
                device,
                queue,
                &inputs,
                &tire_dynamics_spin_input(&state.tires, dt),
                &tire_dynamics_suspension_input(&state.tires),
                tire_dynamics_brush_input(&state.tires),
                &tire_dynamics_planar_input(&state.tires, dt),
                &tire_dynamics_vertical_input(dt),
            )
            .unwrap();
        let prepared = molla::kernels::PreparedTireVehicleFemStep::new(
            engine,
            &fem,
            &vehicle,
            &brush,
            &owner,
            None,
            dt as f32,
            &params,
            &hubs,
            TIRE_DYNAMICS_BRUSH_FORCE_SCALE,
            64 * 1024,
        )
        .unwrap();
        (owner, prepared)
    };
    let (mut candidate, mut guarded) = create();
    let (mut held, mut direct) = create();
    let (mut cpu, cpu_step) = create();
    let surfaces: Vec<_> = state.tires.iter().map(|t| t.cavity_surface()).collect();
    guarded
        .prepare_cavities(engine, &candidate, &surfaces, 1024 * 1024)
        .unwrap();
    let mut candidate_reader = molla::kernels::PreparedTireVehicleFemReadback::new(
        engine,
        &guarded,
        &candidate,
        1024 * 1024,
    )
    .unwrap();
    let mut held_reader =
        molla::kernels::PreparedTireVehicleFemReadback::new(engine, &direct, &held, 1024 * 1024)
            .unwrap();
    let mut cpu_reader =
        molla::kernels::PreparedTireVehicleFemReadback::new(engine, &cpu_step, &cpu, 1024 * 1024)
            .unwrap();
    guarded
        .step_cavity_frame(engine, &mut candidate, 24)
        .unwrap();
    let records = read_floats(engine, guarded.cavity_records_buffer().unwrap());
    let mut identical = params.clone();
    for (i, (p, row)) in identical
        .iter_mut()
        .zip(records.chunks_exact(4))
        .enumerate()
    {
        assert_eq!(row[3], 1.);
        let mut quantized = state.tires[i].clone();
        let h = params[i].hub;
        quantized
            .apply_gpu_node_state(
                &inputs[i].positions,
                &inputs[i].velocities,
                [h[0], h[1], h[2], 0., 0., 0., 0., 0.],
            )
            .unwrap();
        let oracle = quantized.cavity_radial_force_per_bead_n();
        let error = (row[2] as f64 - oracle).abs() / (1. + oracle.abs());
        assert!(error <= 2e-5, "same-input force component {i}: {error:e}");
        println!(
            "wheel {i}: CPU initial load {:.12e}; quantized CPU {:.12e}; GPU {:.12e}; scaled same-input error {error:e}",
            p.cavity_radial_force_per_bead, oracle, row[2]
        );
        p.cavity_radial_force_per_bead = row[2];
    }
    direct
        .update_configuration(
            engine,
            &held,
            &identical,
            &hubs,
            TIRE_DYNAMICS_BRUSH_FORCE_SCALE,
        )
        .unwrap();
    let mut first_failure = None;
    let mut final_difference = 0.;
    for substep in 1..=24 {
        let mut probe_outputs = None;
        if substep == 3 && std::env::var_os("MOLLA_TIRE_STAGE_PROBE").is_some() {
            let capture = |owner: &molla::kernels::TireResidentVehicleFemAtlasOwner,
                           loads: &[molla::kernels::TireFemGpuParams]| {
                let states = fem
                    .readback_resident_atlas_batch(device, queue, owner.fem())
                    .unwrap();
                let mut atlas = owner.fem().atlas().clone();
                atlas.positions = states
                    .iter()
                    .flat_map(|s| s.positions.iter().copied())
                    .collect();
                atlas.velocities = states
                    .iter()
                    .flat_map(|s| s.velocities.iter().copied())
                    .collect();
                for (row, p) in atlas.param_rows.iter_mut().zip(loads) {
                    row.phys3[1] = p.cavity_radial_force_per_bead;
                }
                let hubs = read_floats(engine, owner.fem().hub_state_buffer());
                (atlas, hubs)
            };
            let (gpu, gpu_hubs) = capture(&held, &identical);
            let (cpu, cpu_hubs) = capture(&cpu, &params);
            probe_outputs = Some(stages::run(engine, &gpu, &gpu_hubs, &cpu, &cpu_hubs));
        }
        direct.step(engine, &mut held).unwrap();
        cpu_step.step(engine, &mut cpu).unwrap();
        if let Some((gpu, reference)) = probe_outputs {
            for (owner, expected) in [(&held, gpu), (&cpu, reference)] {
                let rows = fem
                    .readback_resident_atlas_batch(device, queue, owner.fem())
                    .unwrap();
                let mut actual = Vec::new();
                for field in 0..3 {
                    for row in &rows {
                        let data = match field {
                            0 => &row.positions,
                            1 => &row.velocities,
                            _ => &row.forces,
                        };
                        actual.extend(data.iter().flatten().map(|v| v.to_bits()));
                    }
                }
                assert_eq!(
                    actual, expected,
                    "captured replay differs from live third node update"
                );
            }
            println!(
                "third-substep saved-input replay: position/velocity/force words match both live node updates exactly"
            );
        }
        let a = physical(held_reader.read_blocking(engine, &direct, &held).unwrap());
        let b = physical(cpu_reader.read_blocking(engine, &cpu_step, &cpu).unwrap());
        final_difference = difference(&a, &b);
        if final_difference > 2e-5 && first_failure.is_none() {
            first_failure = Some(substep);
        }
        println!("held loads substep {substep}: GPU-load vs CPU-load max {final_difference:e}");
    }
    let a = physical(
        candidate_reader
            .read_blocking(engine, &guarded, &candidate)
            .unwrap(),
    );
    let b = physical(held_reader.read_blocking(engine, &direct, &held).unwrap());
    assert_eq!(a, b, "guarded frame vs same GPU load direct frame");
    let a = fem
        .readback_resident_atlas_batch(device, queue, candidate.fem())
        .unwrap();
    let b = fem
        .readback_resident_atlas_batch(device, queue, held.fem())
        .unwrap();
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.forces, b.forces);
    }
    println!(
        "guarded/direct identical held GPU load: all 11616 physical words and force rows exact; first changed-load failure {first_failure:?}; final {final_difference:e}"
    );
}
