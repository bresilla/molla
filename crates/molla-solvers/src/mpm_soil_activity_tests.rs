use super::super::tests::fixture;
use super::*;

fn stream(state: &mut MpmSoilCheckpoint, s: usize) -> &mut [u32] {
    let start = STRIDES[..s].iter().sum::<usize>() * state.count as usize;
    let end = start + STRIDES[s] * state.count as usize;
    &mut state.words[start..end]
}

fn quiet(count: u32) -> MpmSoilCheckpoint {
    let mut state = fixture(count);
    for s in [1, 2, 6, 8] {
        stream(&mut state, s).fill(0);
    }
    state
}

#[test]
fn activity_matches_analytic_mass_weighted_motion_and_history_rates() {
    let mut before = quiet(2);
    stream(&mut before, 0).fill(0);
    stream(&mut before, 5).copy_from_slice(&[2.0_f32.to_bits(), 6.0_f32.to_bits()]);
    let mut after = before.clone();
    stream(&mut after, 0)[..2].copy_from_slice(&[0.3_f32.to_bits(), 0.4_f32.to_bits()]);
    stream(&mut after, 1)[..2].copy_from_slice(&[3.0_f32.to_bits(), 4.0_f32.to_bits()]);
    stream(&mut after, 2)[0] = 2.0_f32.to_bits();
    let f = f32::from_bits(stream(&mut after, 3)[0]);
    stream(&mut after, 3)[0] = (f + 0.25).to_bits();
    stream(&mut after, 6)[0] = 0.125_f32.to_bits();
    stream(&mut after, 8)[..4].copy_from_slice(&[0.25, 2.0, 3.0, 0.5].map(f32::to_bits));
    let ids = [SoilParticleId(7), SoilParticleId(u64::MAX)];
    let a = after.activity_since(&before, &ids, &ids, 0.5, 2).unwrap();
    assert_eq!(a.particles, 2);
    assert_eq!(a.substeps, 2);
    assert_eq!(a.seconds, 0.5);
    assert_eq!(a.max_speed, 5.0);
    assert_eq!(a.rms_speed, 2.5);
    assert_eq!(a.kinetic_energy, 25.0);
    assert_eq!(a.max_acceleration, 10.0);
    assert!((a.max_displacement_rate - 1.0).abs() < 1e-7);
    assert!((a.max_elastic_rate - 0.5).abs() < 1e-6);
    assert_eq!(a.max_affine_speed, 2.0 * f64::from(before.cfg.cell_size));
    assert_eq!(a.max_plastic_rate, 0.25);
    assert_eq!(a.max_tire_history_rates, [0.5, 4.0, 6.0, 1.0]);
}

#[test]
fn zero_velocity_does_not_hide_plastic_flow_or_endpoint_motion() {
    let before = quiet(1);
    let mut after = before.clone();
    stream(&mut after, 6)[0] = 0.125_f32.to_bits();
    stream(&mut after, 8)[1] = 2.0_f32.to_bits();
    let ids = [SoilParticleId(7)];
    let a = after.activity_since(&before, &ids, &ids, 0.25, 1).unwrap();
    assert_eq!(a.max_speed, 0.0);
    assert_eq!(a.max_plastic_rate, 0.5);
    assert_eq!(a.max_tire_history_rates[1], 8.0);
    stream(&mut after, 1)[0] = 2.0_f32.to_bits();
    let reverse = before.activity_since(&after, &ids, &ids, 0.25, 1).unwrap();
    assert_eq!(reverse.max_speed, 2.0);
    assert_eq!(reverse.rms_speed, 0.0);
    assert_eq!(reverse.max_acceleration, 8.0);
}

#[test]
fn quiet_empty_and_padded_states_are_explicit_without_a_sleep_claim() {
    for count in [0, 1, 129] {
        let before = quiet(count);
        let mut after = before.clone();
        for s in [0, 1, 2, 3] {
            for words in stream(&mut after, s).chunks_exact_mut(4) {
                words[3] = f32::MAX.to_bits();
            }
        }
        let ids: Vec<_> = (0..count).map(|id| SoilParticleId(u64::from(id))).collect();
        let a = after.activity_since(&before, &ids, &ids, 1.0, 3).unwrap();
        assert_eq!(a.max_speed, 0.0);
        assert_eq!(a.max_affine_speed, 0.0);
        assert_eq!(a.kinetic_energy, 0.0);
        assert_eq!(a.max_elastic_rate, 0.0);
        assert_eq!(a.max_tire_history_rates, [0.0; 4]);
    }
}

#[test]
fn activity_rejects_changed_population_physics_time_and_nonfinite_rates() {
    let before = quiet(2);
    let ids = [SoilParticleId(7), SoilParticleId(9)];
    for invalid_ids in [
        vec![],
        vec![ids[0]],
        vec![ids[1], ids[0]],
        vec![ids[0], ids[0]],
    ] {
        assert!(
            before
                .activity_since(&before, &ids, &invalid_ids, 1.0, 1)
                .is_err()
        );
        assert!(
            before
                .activity_since(&before, &invalid_ids, &invalid_ids, 1.0, 1)
                .is_err()
        );
    }
    for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            before
                .activity_since(&before, &ids, &ids, seconds, 1)
                .is_err()
        );
    }
    assert!(before.activity_since(&before, &ids, &ids, 1.0, 0).is_err());
    for s in [4, 5, 7] {
        let mut after = before.clone();
        stream(&mut after, s)[0] = if s == 7 { 2 } else { 1.0_f32.to_bits() };
        assert!(after.activity_since(&before, &ids, &ids, 1.0, 1).is_err());
    }
    let mut after = before.clone();
    after.cfg.gravity_y = 0.0;
    assert!(after.activity_since(&before, &ids, &ids, 1.0, 1).is_err());
    after = before.clone();
    after.cfg.max_particles *= 2;
    after.cfg.max_rigid *= 2;
    assert!(after.activity_since(&before, &ids, &ids, 1.0, 1).is_ok());
    stream(&mut after, 1)[0] = 1.0_f32.to_bits();
    assert!(
        after
            .activity_since(&before, &ids, &ids, f64::from_bits(1), 1)
            .is_err()
    );
    stream(&mut after, 1)[0] = f32::NAN.to_bits();
    assert!(after.activity_since(&before, &ids, &ids, 1.0, 1).is_err());
}

#[test]
fn activity_clock_counts_mixed_steps_and_rejects_overflow_before_submission() {
    let clock = ActivityClock::default()
        .after_steps(0.125, 2)
        .unwrap()
        .after_steps(0.25, 3)
        .unwrap();
    assert_eq!(clock.substeps, 5);
    assert_eq!(clock.seconds, 1.0);
    assert_eq!(clock.after_steps(1.0, 0).unwrap().seconds, 1.0);
    assert!(
        ActivityClock {
            substeps: u64::MAX,
            seconds: 0.0
        }
        .after_steps(1.0, 1)
        .is_err()
    );
    assert!(
        ActivityClock {
            substeps: 1,
            seconds: f64::MAX
        }
        .after_steps(1.0, 1)
        .is_err()
    );
}

fn position(state: &mut MpmSoilCheckpoint, particle: usize, x: f32) {
    stream(state, 0)[particle * 4..particle * 4 + 3]
        .copy_from_slice(&[x, 0.0, 0.0].map(f32::to_bits));
}

fn subset(state: &MpmSoilCheckpoint, indices: &[usize]) -> MpmSoilCheckpoint {
    MpmSoilCheckpoint {
        cfg: state.cfg,
        grid_window_cells: state.grid_window_cells,
        materials: state.materials.clone(),
        count: indices.len() as u32,
        words: (0..9)
            .flat_map(|s| {
                indices.iter().flat_map(move |&i| {
                    state.stream(s)[i * STRIDES[s]..(i + 1) * STRIDES[s]]
                        .iter()
                        .copied()
                })
            })
            .collect(),
    }
}

#[test]
fn tile_activity_keeps_departing_motion_without_polluting_quiet_tiles() {
    let mut before = quiet(3);
    for (i, x) in [-0.25, 0.25, 4.0].into_iter().enumerate() {
        position(&mut before, i, x);
    }
    stream(&mut before, 5).copy_from_slice(&[2.0, 3.0, 5.0].map(f32::to_bits));
    let mut after = before.clone();
    position(&mut after, 0, 2.0);
    stream(&mut after, 1)[0] = 4.0_f32.to_bits();
    stream(&mut after, 6)[0] = 0.5_f32.to_bits();
    let ids = [
        SoilParticleId(0),
        SoilParticleId(1 << 63),
        SoilParticleId(u64::MAX),
    ];
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let (global, tiles) = after
        .activity_by_tile_since(&before, &ids, &ids, 0.5, 2, layout)
        .unwrap();
    assert_eq!(
        global,
        after.activity_since(&before, &ids, &ids, 0.5, 2).unwrap()
    );
    assert_eq!(
        tiles.iter().map(|t| t.tile).collect::<Vec<_>>(),
        [-1, 0, 2, 4].map(|x| SoilTileCoord([x, 0, 0]))
    );
    assert_eq!(tiles[0].previous_particles, 1);
    assert_eq!(tiles[0].current_particles, 0);
    assert_eq!(tiles[0].exited_particles, 1);
    assert_eq!(tiles[0].entered_particles, 0);
    assert_eq!(tiles[2].previous_particles, 0);
    assert_eq!(tiles[2].current_particles, 1);
    assert_eq!(tiles[2].exited_particles, 0);
    assert_eq!(tiles[2].entered_particles, 1);
    assert_eq!(tiles[0].activity, tiles[2].activity);
    assert_eq!(tiles[0].activity.max_speed, 4.0);
    assert_eq!(tiles[0].activity.max_displacement_rate, 4.5);
    assert_eq!(tiles[0].activity.max_plastic_rate, 1.0);
    assert_eq!(tiles[0].activity.kinetic_energy, 16.0);
    assert_eq!(global.kinetic_energy, 16.0);
    assert_eq!(
        tiles.iter().map(|t| t.activity.kinetic_energy).sum::<f64>(),
        32.0
    );
    for i in [1, 3] {
        assert_eq!(tiles[i].activity.max_speed, 0.0);
        assert_eq!(tiles[i].activity.max_plastic_rate, 0.0);
        assert_eq!(tiles[i].entered_particles, 0);
        assert_eq!(tiles[i].exited_particles, 0);
    }
}

#[test]
fn tile_activity_swap_reports_changed_membership_even_at_constant_occupancy() {
    let mut before = quiet(2);
    position(&mut before, 0, -0.25);
    position(&mut before, 1, 0.25);
    let mut after = before.clone();
    position(&mut after, 0, 0.25);
    position(&mut after, 1, -0.25);
    let ids = [SoilParticleId(7), SoilParticleId(9)];
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let (global, tiles) = after
        .activity_by_tile_since(&before, &ids, &ids, 1.0, 1, layout)
        .unwrap();
    assert_eq!(tiles.len(), 2);
    for tile in tiles {
        assert_eq!(tile.previous_particles, 1);
        assert_eq!(tile.current_particles, 1);
        assert_eq!(tile.entered_particles, 1);
        assert_eq!(tile.exited_particles, 1);
        assert_eq!(tile.activity, global);
        assert_eq!(tile.activity.max_speed, 0.0);
        assert_eq!(tile.activity.max_displacement_rate, 0.5);
    }
}

#[test]
fn tile_activity_matches_explicit_id_subsets_and_preserves_checkpoint_bits() {
    let mut before = fixture(129);
    let mut after = before.clone();
    for i in 0..129 {
        position(&mut before, i, (i % 11) as f32 - 5.125);
        position(&mut after, i, (i % 7) as f32 - 3.75);
        stream(&mut after, 1)[i * 4] = (i as f32 * 0.01).to_bits();
        stream(&mut after, 2)[i * 12 + 4] = (-0.2_f32).to_bits();
        stream(&mut after, 3)[i * 12 + 8] = 0.125_f32.to_bits();
        stream(&mut after, 6)[i] = 0.25_f32.to_bits();
        stream(&mut after, 8)[i * 4 + 2] = 2.0_f32.to_bits();
    }
    let saved_before = before.words.clone();
    let saved_after = after.words.clone();
    let ids: Vec<_> = (0..129).map(|i| SoilParticleId(3 * i)).collect();
    for layout in [
        SoilTileLayout::new([0.0; 3], 0.5).unwrap(),
        SoilTileLayout::new([-0.25, 0.5, -1.0], 0.3).unwrap(),
    ] {
        let (global, tiles) = after
            .activity_by_tile_since(&before, &ids, &ids, 0.25, 7, layout)
            .unwrap();
        assert_eq!(
            global,
            after.activity_since(&before, &ids, &ids, 0.25, 7).unwrap()
        );
        assert!(tiles.windows(2).all(|pair| pair[0].tile < pair[1].tile));
        assert_eq!(tiles.iter().map(|t| t.previous_particles).sum::<u32>(), 129);
        assert_eq!(tiles.iter().map(|t| t.current_particles).sum::<u32>(), 129);
        assert_eq!(
            tiles.iter().map(|t| t.entered_particles).sum::<u32>(),
            tiles.iter().map(|t| t.exited_particles).sum::<u32>()
        );
        for tile in tiles {
            let membership = |state: &MpmSoilCheckpoint, i: usize| {
                layout
                    .tile_at(std::array::from_fn(|axis| {
                        f32::from_bits(state.stream(0)[i * 4 + axis]) as f64
                    }))
                    .unwrap()
                    == tile.tile
            };
            let indices: Vec<_> = (0..129)
                .filter(|&i| membership(&before, i) || membership(&after, i))
                .collect();
            let local_ids: Vec<_> = indices.iter().map(|&i| ids[i]).collect();
            let expected = subset(&after, &indices)
                .activity_since(&subset(&before, &indices), &local_ids, &local_ids, 0.25, 7)
                .unwrap();
            assert_eq!(tile.activity, expected);
            assert_eq!(
                tile.previous_particles,
                (0..129).filter(|&i| membership(&before, i)).count() as u32
            );
            assert_eq!(
                tile.current_particles,
                (0..129).filter(|&i| membership(&after, i)).count() as u32
            );
            assert_eq!(
                tile.previous_particles + tile.entered_particles,
                tile.activity.particles
            );
            assert_eq!(
                tile.current_particles + tile.exited_particles,
                tile.activity.particles
            );
        }
    }
    assert_eq!(before.words, saved_before);
    assert_eq!(after.words, saved_after);
}

#[test]
fn tile_activity_empty_single_tile_and_maximal_endpoint_union_are_bounded() {
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let empty = quiet(0);
    let (global, tiles) = empty
        .activity_by_tile_since(&empty, &[], &[], 1.0, 1, layout)
        .unwrap();
    assert_eq!(global.particles, 0);
    assert!(tiles.is_empty());
    let mut before = quiet(129);
    for i in 0..129 {
        position(&mut before, i, 0.25);
    }
    let ids: Vec<_> = (0..129).map(SoilParticleId).collect();
    let (global, tiles) = before
        .activity_by_tile_since(&before, &ids, &ids, 1.0, 1, layout)
        .unwrap();
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].activity, global);
    let mut after = before.clone();
    for i in 0..129 {
        position(&mut before, i, 2.0 * i as f32);
        position(&mut after, i, 2.0 * i as f32 + 1.0);
    }
    let (_, tiles) = after
        .activity_by_tile_since(&before, &ids, &ids, 1.0, 1, layout)
        .unwrap();
    assert_eq!(tiles.len(), 258);
    assert!(tiles.iter().all(|tile| tile.activity.particles == 1));
}

#[test]
fn tile_activity_rejects_invalid_endpoints_and_unrepresentable_tiles() {
    let before = quiet(1);
    let ids = [SoilParticleId(7)];
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    assert!(
        before
            .activity_by_tile_since(&before, &ids, &[], 1.0, 1, layout)
            .is_err()
    );
    assert!(
        before
            .activity_by_tile_since(&before, &ids, &ids, 0.0, 1, layout)
            .is_err()
    );
    let mut after = before.clone();
    stream(&mut after, 5)[0] = 3.0_f32.to_bits();
    assert!(
        after
            .activity_by_tile_since(&before, &ids, &ids, 1.0, 1, layout)
            .is_err()
    );
    after = before.clone();
    position(&mut after, 0, f32::MAX);
    assert!(
        after
            .activity_by_tile_since(&before, &ids, &ids, 1.0, 1, layout)
            .is_err()
    );
    assert!(
        before
            .activity_by_tile_since(&after, &ids, &ids, 1.0, 1, layout)
            .is_err()
    );
    after = before.clone();
    stream(&mut after, 1)[0] = 1.0_f32.to_bits();
    assert!(
        after
            .activity_by_tile_since(&before, &ids, &ids, f64::from_bits(1), 1, layout)
            .is_err()
    );
}
