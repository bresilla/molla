use super::*;
use crate::joint_mass::{JointMass, MassSeed};
use crate::rigid_impulse::MassFactor;
use molla_core::BodyId;
use molla_sim::{BodyParams, ModelBuilder};
use std::collections::BTreeMap;

#[test]
fn cached_factors_match_fresh_drive_inertias_and_preserve_fallbacks() {
    for scale in [1e-60, 1.0, 1e60] {
        for weak in [0.0, 1e-13, 1e-9, 1.0] {
            for offset in [Vec3::ZERO, Vec3::new(0.1, -0.2, 0.05)] {
                let mut builder = ModelBuilder::new();
                let body = builder.add_body(
                    BodyParams::new()
                        .with_mass(2.0 * scale)
                        .with_com(offset)
                        .with_inertia(Mat3::from_diagonal(Vec3::new(weak, 0.3, 0.5)) * scale),
                );
                builder
                    .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
                    .unwrap();
                let model = builder.build().unwrap();
                let state = model.state().unwrap();
                for stale in [false, true] {
                    let mut next = state.clone_host().unwrap();
                    if stale {
                        next.body_q.host_mut().unwrap()[0].rotation = Quat::from_rotation_z(0.31);
                    }
                    for (stable, motor_model) in [(true, 0), (false, 1), (false, 0)] {
                        for subset in [false, true] {
                            let matrix = crate::mass_matrix(&model, &state, 0).unwrap();
                            let factor = MassFactor::new(&matrix, 6).unwrap().into_full_rank();
                            assert_eq!(factor.is_some(), weak >= 1e-9);
                            let pointer = factor
                                .as_ref()
                                .and_then(|f| f.full_rank_cholesky(6))
                                .map(|(_, f)| f.as_ptr());
                            let mut seed = MassSeed::new(&model);
                            seed.record(&model, &state, BTreeMap::from([(0, (matrix, factor))]))
                                .unwrap();
                            let mut cached =
                                JointMass::with_seed(&model, &next, Some(seed)).unwrap();
                            assert_eq!(
                                cached.cached_cholesky(0, 6).is_some(),
                                weak >= 1e-9 && !stale
                            );
                            assert!(cached.cached_cholesky(0, 5).is_none());
                            if !stale {
                                assert_eq!(
                                    cached.cached_cholesky(0, 6).map(|(_, f)| f.as_ptr()),
                                    pointer
                                );
                            }
                            let mut control = model.control();
                            control
                                .joint_target_mode
                                .host_mut()
                                .unwrap()
                                .fill(JointTargetMode::Velocity as u32);
                            control
                                .joint_motor_model
                                .host_mut()
                                .unwrap()
                                .fill(motor_model);
                            if subset {
                                for i in [1, 3, 5] {
                                    control.joint_target_mode.host_mut().unwrap()[i] = 0;
                                }
                            }
                            let actual = effective_drive_inertias(
                                &model,
                                &control,
                                stable,
                                &mut cached,
                                None,
                            )
                            .unwrap();
                            let mut fresh = JointMass::new(&model, &next);
                            let expected = effective_drive_inertias(
                                &model, &control, stable, &mut fresh, None,
                            )
                            .unwrap();
                            assert_eq!(
                                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                "scale={scale}, weak={weak}, offset={offset:?}, stale={stale}, stable={stable}, model={motor_model}, subset={subset}"
                            );
                            if !stale && pointer.is_some() {
                                assert_eq!(
                                    cached.cached_cholesky(0, 6).map(|(_, f)| f.as_ptr()),
                                    pointer
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
