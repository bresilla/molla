use molla_solvers::mpm::{MpmGridStorage, MpmProfileStage as Stage, MpmStepProfile};

pub fn validate(
    profile: &MpmStepProfile,
    substeps: usize,
    storage: MpmGridStorage,
    projected: bool,
    first_frame: bool,
    fixed_order: bool,
    motion_fusion: bool,
) -> Result<(), String> {
    let active = storage != MpmGridStorage::Dense;
    let compact = matches!(storage, MpmGridStorage::Compact { .. });
    if substeps == 0
        || profile.requested_substeps != substeps
        || profile.particles != 50_000
        || !profile.native_float_atomics
        || profile.active_grid != active
        || profile.compact_grid != compact
        || profile.particle_tile_ordering != fixed_order
        || profile.parallel_particle_sort != fixed_order
        || (fixed_order && !projected)
        || (active && !projected)
    {
        return Err("MPM frame profile identity mismatch".into());
    }
    let mut expected = Vec::with_capacity(substeps * 9 + 1);
    if active && first_frame {
        expected.push((Stage::Initialization, None));
    }
    for step in 0..substeps {
        if fixed_order {
            expected.push((Stage::ParticleOrdering, Some(step)));
        }
        expected.push((Stage::GridReset, Some(step)));
        if active {
            expected.push((Stage::ActiveTiles, Some(step)));
        }
        expected.push((Stage::RigidReset, Some(step)));
        expected.push((Stage::ParticleToGrid, Some(step)));
        if projected {
            expected.push((Stage::GridUpdate, Some(step)));
            expected.push((Stage::ContactProjection, Some(step)));
        } else {
            expected.push((Stage::RigidImprint, Some(step)));
            expected.push((Stage::GridUpdate, Some(step)));
        }
        let stages = if motion_fusion {
            [Stage::GuardedGatherProjection, Stage::GuardedParticleCommit]
        } else {
            [Stage::GridToParticle, Stage::Plasticity]
        };
        expected.extend(stages.map(|stage| (stage, Some(step))));
    }
    if profile.stages.len() != expected.len()
        || profile
            .stages
            .iter()
            .zip(expected)
            .any(|(actual, expected)| {
                (actual.stage, actual.substep) != expected
                    || !actual.start_ms.is_finite()
                    || actual.start_ms < 0.0
                    || !actual.elapsed_ms.is_finite()
                    || actual.elapsed_ms < 0.0
                    || !(actual.start_ms + actual.elapsed_ms).is_finite()
            })
    {
        return Err("MPM frame profile stage sequence or duration mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_solvers::mpm::MpmStageTiming;

    #[test]
    fn layout_specific_profile_contract() {
        for storage in [
            MpmGridStorage::Dense,
            MpmGridStorage::ActiveTiles,
            MpmGridStorage::Compact {
                tile_capacity: 1024,
            },
        ] {
            for projected in [false, true] {
                let active = storage != MpmGridStorage::Dense;
                if active && !projected {
                    continue;
                }
                for fixed_order in [false, true] {
                    if fixed_order && !projected {
                        continue;
                    }
                    for first in [false, true] {
                        let mut stages = Vec::new();
                        if active && first {
                            stages.push((Stage::Initialization, None));
                        }
                        for step in 0..2 {
                            if fixed_order {
                                stages.push((Stage::ParticleOrdering, Some(step)));
                            }
                            let sequence: &[Stage] = if active {
                                &[
                                    Stage::GridReset,
                                    Stage::ActiveTiles,
                                    Stage::RigidReset,
                                    Stage::ParticleToGrid,
                                    Stage::GridUpdate,
                                    Stage::ContactProjection,
                                    Stage::GridToParticle,
                                    Stage::Plasticity,
                                ]
                            } else if projected {
                                &[
                                    Stage::GridReset,
                                    Stage::RigidReset,
                                    Stage::ParticleToGrid,
                                    Stage::GridUpdate,
                                    Stage::ContactProjection,
                                    Stage::GridToParticle,
                                    Stage::Plasticity,
                                ]
                            } else {
                                &[
                                    Stage::GridReset,
                                    Stage::RigidReset,
                                    Stage::ParticleToGrid,
                                    Stage::RigidImprint,
                                    Stage::GridUpdate,
                                    Stage::GridToParticle,
                                    Stage::Plasticity,
                                ]
                            };
                            stages.extend(sequence.iter().map(|&s| (s, Some(step))));
                        }
                        let profile = MpmStepProfile {
                            requested_substeps: 2,
                            requested_simulated_seconds: 0.001,
                            particles: 50_000,
                            active_grid: active,
                            compact_grid: matches!(storage, MpmGridStorage::Compact { .. }),
                            native_float_atomics: true,
                            particle_tile_ordering: fixed_order,
                            parallel_particle_sort: fixed_order,
                            timestamp_period_ns: 1.0,
                            readback_bytes: stages.len() as u64 * 16,
                            stages: stages
                                .into_iter()
                                .enumerate()
                                .map(|(i, (stage, substep))| MpmStageTiming {
                                    stage,
                                    substep,
                                    start_ms: i as f64,
                                    elapsed_ms: 0.5,
                                })
                                .collect(),
                            contact_stages: Vec::new(),
                            preparation_details_enabled: false,
                            preparation_stages: Vec::new(),
                        };
                        validate(&profile, 2, storage, projected, first, fixed_order, false)
                            .unwrap();
                        let mut fused = profile.clone();
                        for entry in &mut fused.stages {
                            entry.stage = match entry.stage {
                                Stage::GridToParticle => Stage::GuardedGatherProjection,
                                Stage::Plasticity => Stage::GuardedParticleCommit,
                                stage => stage,
                            };
                        }
                        validate(&fused, 2, storage, projected, first, fixed_order, true).unwrap();
                        assert!(
                            validate(&fused, 2, storage, projected, first, fixed_order, false)
                                .is_err()
                        );
                        assert!(
                            validate(&profile, 2, storage, projected, first, fixed_order, true)
                                .is_err()
                        );
                        fused
                            .stages
                            .iter_mut()
                            .find(|s| s.stage == Stage::GuardedParticleCommit)
                            .unwrap()
                            .stage = Stage::Plasticity;
                        assert!(
                            validate(&fused, 2, storage, projected, first, fixed_order, true)
                                .is_err()
                        );
                        assert!(
                            validate(&profile, 2, storage, projected, first, !fixed_order, false)
                                .is_err()
                        );
                        for mutation in 0..12 {
                            let mut bad = profile.clone();
                            match mutation {
                                0 => {
                                    bad.stages.pop();
                                }
                                1 => bad.stages.swap(1, 2),
                                2 => bad.stages[1].substep = Some(2),
                                3 => bad.active_grid = !active,
                                4 => bad.compact_grid = !bad.compact_grid,
                                5 => bad.stages[1].elapsed_ms = f64::NAN,
                                6 => bad.stages[1].start_ms = -1.0,
                                7 => bad.particle_tile_ordering = !fixed_order,
                                8 => bad.native_float_atomics = false,
                                9 => bad.particles = 64,
                                10 => bad.parallel_particle_sort = !fixed_order,
                                _ => bad.requested_substeps = 1,
                            }
                            assert!(
                                validate(&bad, 2, storage, projected, first, fixed_order, false)
                                    .is_err(),
                                "{storage:?} {projected} {first} mutation={mutation}"
                            );
                        }
                        if active {
                            assert!(
                                validate(
                                    &profile,
                                    2,
                                    storage,
                                    projected,
                                    !first,
                                    fixed_order,
                                    false
                                )
                                .is_err()
                            );
                            assert!(
                                validate(&profile, 2, storage, false, first, fixed_order, false)
                                    .is_err()
                            );
                        }
                    }
                }
            }
        }
    }
}
