use super::*;

pub(super) fn differences(a: &MpmSoilFrameCheckpoint, b: &MpmSoilFrameCheckpoint) -> [f64; 9] {
    assert_eq!(a.soil().particle_count(), b.soil().particle_count());
    let mut errors = [0.0_f64; 9];
    for (stream, error) in errors.iter_mut().enumerate() {
        for (word, (&a_word, &b_word)) in a
            .soil()
            .stream(stream)
            .iter()
            .zip(b.soil().stream(stream))
            .enumerate()
        {
            if stream == 7 {
                assert_eq!(a_word, b_word);
                continue;
            }
            let x = f64::from(f32::from_bits(a_word));
            let y = f64::from(f32::from_bits(b_word));
            assert!(x.is_finite() && y.is_finite());
            let normalized = if stream == 0 && word % 4 < 3 {
                ((x - y) + (a.world_origin()[word % 4] - b.world_origin()[word % 4])).abs()
            } else {
                (x - y).abs() / x.abs().max(y.abs()).max(1.0)
            };
            *error = error.max(normalized);
        }
    }
    errors
}

#[test]
fn recenter_trajectory_comparator_preserves_small_world_differences() {
    let mut soil = fixture(1);
    soil.cfg.grid_origin = [-4.0; 3];
    soil.cfg.grid_size = 64;
    soil.cfg.cell_size = 0.125;
    soil.words[0] = 0.125f32.to_bits();
    soil.words[1] = 2.0f32.to_bits();
    soil.words[2] = 2.0f32.to_bits();
    let a = MpmSoilFrameCheckpoint::new(soil, [1e12; 3]).unwrap();
    let mut b = a.recentered([-5.0, -4.0, -4.0]).unwrap();
    assert_eq!(differences(&a, &b), [0.0; 9]);
    b.soil.words[0] = (f32::from_bits(b.soil.words[0]) + 1.0 / 8388608.0).to_bits();
    assert_eq!(differences(&a, &b)[0], 1.0 / 8388608.0);
    assert_eq!(
        f64::from(f32::from_bits(a.soil.words[0])) + a.origin[0],
        f64::from(f32::from_bits(b.soil.words[0])) + b.origin[0]
    );
}

#[test]
#[ignore = "requires GPU; records paired continued stepping and exact unrebased self-repeat"]
fn canonical_recenter_continued_trajectory_gate() {
    let engine = Engine::new_gpu().unwrap();
    let output = std::env::var_os("MOLLA_RECENTER_TRAJECTORY_OUTPUT").map(PathBuf::from);
    if let Some(root) = &output {
        fs::create_dir(root).unwrap();
    }
    let mut records = Vec::new();
    let mut accepted = true;
    for ordered in [false, true] {
        for mode in 0..4 {
            if ordered && mode == 0 {
                continue;
            }
            let dirs = [Directory::new(), Directory::new(), Directory::new()];
            let mut worlds: Vec<_> = dirs
                .iter()
                .map(|dir| {
                    let (store, archive, cfg) = seed_canonical(dir);
                    let keys: Vec<_> = archive.tiles().collect();
                    let mut live = world(&engine, store, cfg, mode, [1e12; 3]);
                    request(&mut live, &engine, &keys).unwrap();
                    if ordered {
                        live.simulation
                            .prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), 129)
                            .unwrap();
                        live.simulation.set_particle_tile_ordering(true).unwrap();
                        live.simulation.prepare_deterministic_p2g().unwrap();
                        live.simulation.set_deterministic_p2g(true).unwrap();
                    }
                    live
                })
                .collect();
            let initial = capture(&mut worlds[0], &engine);
            for live in &mut worlds[1..] {
                assert_eq!(capture(live, &engine).to_bytes(), initial.to_bytes());
            }
            worlds[2].begin_recenter([-5.0, -4.0, -4.0]).unwrap();
            poll(&mut worlds[2], &engine).unwrap();
            assert_eq!(
                capture(&mut worlds[2], &engine).to_bytes(),
                initial.recentered([-5.0, -4.0, -4.0]).unwrap().to_bytes()
            );
            for step in 1..=8 {
                for index in if step % 2 == 0 { [2, 1, 0] } else { [0, 1, 2] } {
                    worlds[index].step_n(&engine, 0.0001, 1).unwrap();
                }
                let states: Vec<_> = worlds
                    .iter_mut()
                    .map(|live| capture(live, &engine))
                    .collect();
                let repeat_errors = differences(&states[0], &states[1]);
                let rebase_errors = differences(&states[0], &states[2]);
                let exact_repeat = states[0].to_bytes() == states[1].to_bytes();
                let physical_parity = rebase_errors.iter().all(|&error| error <= 1e-6);
                accepted &= exact_repeat && physical_parity;
                records.push(serde_json::json!({
                    "ordered_p2g":ordered,"grid_mode":mode,"step":step,
                    "seconds":f64::from(0.0001_f32)*f64::from(step),
                    "exact_self_repeat":exact_repeat,"physical_parity":physical_parity,
                    "control_repeat_stream_errors":repeat_errors,"control_rebased_stream_errors":rebase_errors,
                }));
                if let Some(root) = &output {
                    for (index, state) in states.iter().enumerate() {
                        fs::write(
                            root.join(format!(
                                "ordered-{ordered}-mode-{mode}-step-{step}-run-{index}.frame"
                            )),
                            state.to_bytes(),
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
    let report = serde_json::json!({
        "schema":1,"particles":129,"dt":0.0001_f32,"steps":8,
        "physical_tolerance":1e-6,"self_repeat":"exact",
        "position_error":"absolute metres using split frame subtraction",
        "other_stream_error":"absolute difference divided by max(1,abs(a),abs(b))",
        "source_frame":[1e12,1e12,1e12],"source_grid":[-4.0,-4.0,-4.0],"target_grid":[-5.0,-4.0,-4.0],
        "scope":"detached canonical soil; not a 50k tractor trajectory or performance gate",
        "accepted":accepted,"records":records,
    });
    println!("{report}");
    if let Some(root) = &output {
        fs::write(
            root.join("comparison.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }
    assert!(
        accepted,
        "continued rebase trajectory/self-repeat gate failed; inspect paired records"
    );
}
