use super::*;

fn soil(engine: &Engine, layout: u32, cells: [i32; 3]) -> MpmSimulation {
    let cfg = MpmConfig {
        grid_size: 64,
        grid_origin: [-3.0, -0.5, -3.0],
        cell_size: 0.1,
        max_particles: 1,
        max_rigid: 1,
        youngs: 100.0,
        particle_mass: 0.5,
        particle_volume: 0.5 / 1600.0,
        ..MpmConfig::sand()
    };
    let mut sim = if layout == 2 {
        MpmSimulation::new_sparse(engine, cfg, 125)
    } else {
        MpmSimulation::new(engine, cfg)
    }
    .unwrap();
    if layout == 1 {
        sim.set_active_grid(true).unwrap();
    }
    sim.set_particles_with_velocity(
        engine.queue().unwrap(),
        &[[0.10, 0.3, 0.0]],
        &[[-0.05, 0.0, 0.0]],
    )
    .unwrap();
    sim.set_prescribed_solids(
        engine.queue().unwrap(),
        &[MpmPrescribedSolid {
            shape: MpmSolidShape::Sphere { radius: 0.12 },
            center: [0.0, 0.3, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
        }],
    )
    .unwrap();
    sim.grid_window_cells = cells;
    sim.invalidate_prepared_steps();
    sim
}

fn words(state: &MpmSoilCheckpoint) -> Vec<u32> {
    let bytes = state.to_bytes();
    bytes[bytes.len() - state.particle_bytes()..]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn error(a: &[u32], b: &[u32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            let a = f64::from(f32::from_bits(a));
            let b = f64::from(f32::from_bits(b));
            assert!(a.is_finite() && b.is_finite());
            (a - b).abs() / a.abs().max(b.abs()).max(1.0)
        })
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "requires GPU; decimal-grid mass/unilateral contact integration and stale owners"]
fn lattice_contact_gpu_coupled_decimal_window() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mass = data(&engine, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
    let jac = data(
        &engine,
        &[
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    );
    let output =
        std::path::PathBuf::from(std::env::var_os("MOLLA_LATTICE_CONTACT_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let mut records = Vec::new();
    let mut accepted = true;
    for unilateral in [false, true] {
        for layout in 0..3 {
            let mut expected = None;
            for cells in [[0; 3], [-1, 0, 1], [1, 0, -1]] {
                let mut sim = soil(&engine, layout, cells);
                let free = data(&engine, &[0.; 3]);
                let mut coupling = if unilateral {
                    let cfg = MpmUnilateralConfig {
                        node_capacity: 128,
                        search_distance: 0.05,
                        max_iterations: 64,
                    };
                    let bytes =
                        PreparedMpmUnilateralCoupling::required_bytes_for_grid(&sim, |size| {
                            PreparedMpmUnilateralCoupling::required_bytes(cfg, 3, size, 1)
                        })
                        .unwrap();
                    let owner = PreparedMpmUnilateralCoupling::new(
                        engine.clone(),
                        &mut sim,
                        MpmUnilateralInputs {
                            mass: &mass,
                            jacobians: &jac,
                            free_rates: &free,
                            dofs: 3,
                            max_owned_bytes: bytes,
                        },
                        cfg,
                    )
                    .unwrap();
                    assert_eq!(owner.owned_buffer_bytes(), bytes);
                    FaultTestCoupling::Unilateral(owner)
                } else {
                    let bytes = PreparedMpmMassCoupling::required_bytes(&sim, 3, 1, 128).unwrap();
                    let owner = PreparedMpmMassCoupling::new(
                        engine.clone(),
                        &mut sim,
                        MpmMassCouplingInputs {
                            mass: &mass,
                            jacobians: &jac,
                            free_rates: &free,
                            dofs: 3,
                            nodes_per_body: 128,
                            max_owned_bytes: bytes,
                        },
                    )
                    .unwrap();
                    assert_eq!(owner.owned_buffer_bytes(), bytes);
                    FaultTestCoupling::Sticking(owner)
                };
                let epoch = sim.step_epoch.clone();
                sim.grid_window_cells[0] += 1;
                let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                let result = match &mut coupling {
                    FaultTestCoupling::Sticking(c) => c.step(&mut sim, 0.0001),
                    FaultTestCoupling::Unilateral(c) => c.step(&mut sim, 0.0001),
                };
                assert!(result.unwrap_err().to_string().contains("stale prepared"));
                assert_eq!(traffic.finish().upload_bytes, 0);
                assert!(Arc::ptr_eq(&epoch, &sim.step_epoch));
                sim.grid_window_cells = cells;
                let mut steps = Vec::new();
                for step in 0..2 {
                    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
                    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                    coupling.step(&mut sim);
                    assert_eq!(allocations.finish().observations, 0);
                    assert_eq!(traffic.finish().readback_bytes, 0);
                    assert_eq!(read_words(device, queue, coupling.status())[0], 0);
                    let state = sim.checkpoint_soil_with(queue).unwrap();
                    assert_eq!(state.grid_window_cells(), cells);
                    let rates = match &coupling {
                        FaultTestCoupling::Sticking(c) => c.rates_buffer(),
                        FaultTestCoupling::Unilateral(c) => c.rates_buffer(),
                    };
                    let rates = read_words(device, queue, rates);
                    assert!(rates.iter().any(|&v| f32::from_bits(v).abs() > 1e-7));
                    steps.push((words(&state), rates));
                    std::fs::write(
                        output.join(format!(
                            "u-{unilateral}-l-{layout}-c-{}-s-{step}.bin",
                            cells[0]
                        )),
                        state.to_bytes(),
                    )
                    .unwrap();
                }
                let mut errors = [0.0_f64; 2];
                if let Some(expected) = &expected {
                    let expected: &Vec<(Vec<u32>, Vec<u32>)> = expected;
                    for (a, b) in expected.iter().zip(&steps) {
                        errors[0] = errors[0].max(error(&a.0, &b.0));
                        errors[1] = errors[1].max(error(&a.1, &b.1));
                        assert_eq!(a.0[35], b.0[35]);
                    }
                } else {
                    expected = Some(steps);
                }
                let pass = errors.iter().all(|&v| v <= 1e-6);
                accepted &= pass;
                records.push(serde_json::json!({"unilateral":unilateral,"layout":layout,"cells":cells,"particle_error":errors[0],"rate_error":errors[1],"pass":pass,"stale_owner_rejected_before_upload":true}));
            }
        }
    }
    let report = serde_json::json!({"accepted":accepted,"records":records,"particles":1,"steps":2,"dt":0.0001,"tolerance":1e-6,"scope":"internal detached projected contact integration; no machine drift or Newton tractor acceptance"});
    std::fs::write(
        output.join("coupling.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("lattice_coupling_report={report}");
    assert!(accepted);
}
