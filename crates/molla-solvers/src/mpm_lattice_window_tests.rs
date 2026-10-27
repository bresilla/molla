use super::*;

#[test]
fn lattice_window_shader_layout_and_variants() {
    assert_eq!(std::mem::size_of::<Uniforms>(), 112);
    assert_eq!(std::mem::offset_of!(Uniforms, grid_window_cells), 96);
    let base = include_str!("../shaders/mpm.wgsl");
    for base in [base.to_owned(), native_atomics::shader_source()] {
        for tail in ["", include_str!("../shaders/mpm_gather_base_cache.wgsl")] {
            validate_shader(&format!("{base}\n{tail}"));
        }
        validate_shader(&format!(
            "{base}\n{}\n{}",
            include_str!("../shaders/mpm_deterministic_p2g.wgsl"),
            include_str!("../shaders/mpm_gather_masks.wgsl"),
        ));
    }
    validate_shader(&format!(
        "{}\n{}",
        native_atomics::shader_source(),
        include_str!("../shaders/mpm_ordered_native_shell.wgsl"),
    ));
    validate_shader(include_str!("../shaders/mpm_active_tiles.wgsl"));
    validate_shader(include_str!("../shaders/mpm_particle_order.wgsl"));
}

fn validate_shader(source: &str) {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

fn decimal_state() -> MpmSoilCheckpoint {
    let mut state = tests::fixture(65);
    state.cfg.grid_origin = [-3.0, -0.5, -3.0];
    state.cfg.cell_size = 0.1;
    state.cfg.grid_size = 64;
    state
}

#[test]
fn lattice_window_decimal_fractions_and_node_identity() {
    let state = decimal_state();
    for cells in [[-1, 0, 1], [1, 0, -1], [-8, 0, 8]] {
        for p in state.stream(0).chunks_exact(4) {
            for axis in 0..3 {
                let xp = (f32::from_bits(p[axis]) - state.cfg.grid_origin[axis])
                    * (1.0 / state.cfg.cell_size);
                let global = (xp - 0.5).floor();
                let local = global as i32 - cells[axis];
                assert_eq!(local + cells[axis], global as i32);
                let fraction = xp - global;
                for node in 0..3 {
                    let old = state.cfg.grid_origin[axis]
                        + (global as i32 + node) as f32 * state.cfg.cell_size;
                    let new = state.cfg.grid_origin[axis]
                        + (local + node + cells[axis]) as f32 * state.cfg.cell_size;
                    assert_eq!(old.to_bits(), new.to_bits());
                    assert_eq!(
                        (node as f32 - fraction).to_bits(),
                        ((local + node - local) as f32 - fraction).to_bits()
                    );
                }
            }
        }
    }
}

fn difference(a: &MpmSoilCheckpoint, b: &MpmSoilCheckpoint) -> [f64; 9] {
    assert_eq!(a.cfg, b.cfg);
    assert_eq!(a.count, b.count);
    let mut errors = [0.0_f64; 9];
    for (stream, error) in errors.iter_mut().enumerate() {
        for (word, (&a, &b)) in a.stream(stream).iter().zip(b.stream(stream)).enumerate() {
            if stream == 7 {
                assert_eq!(a, b);
                continue;
            }
            let a = f64::from(f32::from_bits(a));
            let b = f64::from(f32::from_bits(b));
            assert!(a.is_finite() && b.is_finite());
            let scale = if stream == 0 && word % 4 < 3 {
                1.0
            } else {
                a.abs().max(b.abs()).max(1.0)
            };
            *error = error.max((a - b).abs() / scale);
        }
    }
    errors
}

#[test]
#[ignore = "requires GPU; internal integer-window coordinate migration fixture"]
fn lattice_window_gpu_decimal_transfer_and_prepared_records() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let output = std::path::PathBuf::from(std::env::var_os("MOLLA_LATTICE_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let mut records = Vec::new();
    let mut accepted = true;
    for implicit in [false, true] {
        for mode in 1..3 {
            let mut initial = decimal_state();
            initial.cfg.time_step = if implicit {
                MpmTimeStep::Implicit { jacobi_iters: 2 }
            } else {
                MpmTimeStep::Explicit
            };
            let mut runs: Vec<_> = (0..6)
                .map(|arm| {
                    let mut sim = tests::new_sim(&engine, initial.cfg, mode);
                    sim.restore_soil_with(queue, &initial).unwrap();
                    sim.prepare_parallel_particle_tile_ordering(queue, initial.cfg.max_particles)
                        .unwrap();
                    sim.set_particle_tile_ordering(true).unwrap();
                    sim.prepare_deterministic_p2g_with_compensated_stiffness(129 * 64)
                        .unwrap();
                    sim.set_deterministic_p2g(true).unwrap();
                    sim.grid_window_cells = [[0; 3], [-1, 0, 1], [1, 0, -1]][arm / 2];
                    sim.active_grid_needs_reset = true;
                    sim.invalidate_prepared_steps();
                    let captured = sim.checkpoint_soil_with(queue).unwrap();
                    assert_eq!(captured.grid_window_cells(), sim.grid_window_cells);
                    assert_eq!(captured.words, initial.words);
                    assert_eq!(
                        MpmSoilCheckpoint::from_bytes(&captured.to_bytes(), 65)
                            .unwrap()
                            .to_bytes(),
                        captured.to_bytes()
                    );
                    sim
                })
                .collect();
            let prepared: Vec<_> = runs
                .iter()
                .map(|sim| sim.prepare_step(0.0001, 1).unwrap())
                .collect();
            for step in 0..4 {
                for (sim, record) in runs.iter_mut().zip(&prepared) {
                    if step % 2 == 0 {
                        sim.step_n_with(engine.device().unwrap(), queue, 0.0001, 1)
                            .unwrap();
                    } else {
                        let encoder = engine
                            .device()
                            .unwrap()
                            .create_command_encoder(&Default::default());
                        let mut batch = sim.begin_command_batch(encoder);
                        batch.encode_step(record).unwrap();
                        batch.submit(queue);
                    }
                }
                let states: Vec<_> = runs
                    .iter_mut()
                    .map(|sim| sim.checkpoint_soil_with(queue).unwrap())
                    .collect();
                let exact = (0..3)
                    .all(|pair| states[2 * pair].to_bytes() == states[2 * pair + 1].to_bytes());
                let errors = [
                    difference(&states[0], &states[2]),
                    difference(&states[0], &states[4]),
                ];
                let pass = errors.iter().flatten().all(|&error| error <= 1e-6);
                accepted &= exact && pass;
                records.push(serde_json::json!({"implicit":implicit,"mode":mode,"step":step,"exact_self_repeats":exact,"errors":errors,"physical_parity":pass}));
                for (arm, state) in states.iter().enumerate() {
                    std::fs::write(
                        output.join(format!("i-{implicit}-m-{mode}-s-{step}-a-{arm}.bin")),
                        state.to_bytes(),
                    )
                    .unwrap();
                }
            }
        }
    }
    let report = serde_json::json!({"accepted":accepted,"particles":65,"dt":0.0001,"steps":4,"tolerance":1e-6,"records":records,"scope":"internal coordinate path only; public nonzero restore remains disabled"});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("lattice_window_report={report}");
    assert!(accepted);
}

#[test]
#[ignore = "requires GPU; direct and two-pass anchored scatter"]
fn lattice_window_gpu_decimal_scatter() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let output = std::path::PathBuf::from(std::env::var_os("MOLLA_LATTICE_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let mut records = Vec::new();
    let mut accepted = true;
    for two_pass in [false, true] {
        for mode in 0..3 {
            let mut initial = tests::fixture(1);
            initial.cfg.grid_origin = [-3.0, -0.5, -3.0];
            initial.cfg.cell_size = 0.1;
            initial.cfg.grid_size = 64;
            initial.cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 2 };
            let mut states = Vec::new();
            for cells in [[0; 3], [-1, 0, 1], [1, 0, -1]] {
                let mut sim = tests::new_sim(&engine, initial.cfg, mode);
                sim.restore_soil_with(queue, &initial).unwrap();
                sim.use_two_pass_p2g = two_pass;
                if two_pass {
                    sim.set_rigid_spheres(queue, &[([-0.1, 0.3, -0.1], 0.08, [0.01, 0.0, 0.0])])
                        .unwrap();
                }
                sim.grid_window_cells = cells;
                sim.invalidate_prepared_steps();
                sim.step_n_with(engine.device().unwrap(), queue, 0.0001, 4)
                    .unwrap();
                let state = sim.checkpoint_soil_with(queue).unwrap();
                assert_eq!(state.grid_window_cells(), cells);
                states.push(state);
            }
            let errors = [
                difference(&states[0], &states[1]),
                difference(&states[0], &states[2]),
            ];
            let pass = errors.iter().flatten().all(|&error| error <= 1e-6);
            accepted &= pass;
            records.push(serde_json::json!({"two_pass":two_pass,"sphere_contact":two_pass,"mode":mode,"errors":errors,"physical_parity":pass}));
            for (arm, state) in states.iter().enumerate() {
                std::fs::write(
                    output.join(format!("scatter-t-{two_pass}-m-{mode}-a-{arm}.bin")),
                    state.to_bytes(),
                )
                .unwrap();
            }
        }
    }
    let report = serde_json::json!({"accepted":accepted,"particles":1,"dt":0.0001,"steps":4,"tolerance":1e-6,"records":records,"scope":"single-particle direct/two-pass transfer; not atomic-order repeat certification"});
    std::fs::write(
        output.join("scatter.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("lattice_scatter_report={report}");
    assert!(accepted);
}
