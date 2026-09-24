use super::*;
use molla_core::{BodyId, JointId};
use molla_sim::{BodyParams, ModelBuilder};
use std::collections::BTreeMap;

#[test]
fn seeded_steps_match_fresh_mass_after_mimic_fk_and_singular_changes() {
    for mimic in [false, true] {
        for singular in [false, true] {
            let mut builder = ModelBuilder::new();
            for art in 0..2 {
                builder.begin_articulation();
                let root = builder.add_body(
                    BodyParams::new()
                        .with_mass(4.0)
                        .with_inertia(Mat3::IDENTITY),
                );
                let tip = builder.add_body(
                    BodyParams::new()
                        .with_mass(1.0)
                        .with_com(Vec3::new(0.1, -0.2, 0.05))
                        .with_inertia(if singular {
                            Mat3::ZERO
                        } else {
                            Mat3::IDENTITY * 0.3
                        }),
                );
                builder
                    .add_joint_free(
                        BodyId::NONE,
                        root,
                        Transform::from_translation(Vec3::new(art as Real * 3.0, 2.0, -0.3)),
                        Transform::IDENTITY,
                    )
                    .unwrap();
                let hinge = builder
                    .add_joint_revolute(
                        root,
                        tip,
                        Transform::from_translation(Vec3::new(0.2, -0.4, 0.1)),
                        Transform::IDENTITY,
                        Vec3::Z,
                    )
                    .unwrap();
                if singular && art == 1 {
                    builder.set_joint_limits(hinge, &[0.0], &[0.0]).unwrap();
                }
            }
            if mimic {
                builder
                    .add_mimic_constraint(JointId(1), JointId(3), 0.1, -1.0)
                    .unwrap();
            }
            let model = builder.build().unwrap();
            let mut a = model.state().unwrap();
            if singular {
                assert_eq!(crate::mass_matrix(&model, &a, 1).unwrap()[6 * 7 + 6], 0.0);
            }
            let mut b = a.clone_host().unwrap();
            let mut output = a.clone_host().unwrap();
            let mut control = model.control();
            for dof in [6, 13] {
                control.joint_target_mode.host_mut().unwrap()[dof] =
                    JointTargetMode::Velocity as u32;
                control.joint_target_vel.host_mut().unwrap()[dof] = 0.3;
                control.joint_target_kd.host_mut().unwrap()[dof] = 8.0;
                control.joint_motor_max_force.host_mut().unwrap()[dof] = 20.0;
            }
            let mut solver = FeatherstoneSolver::new_cpu()
                .with_rigorous_d_solve(true)
                .with_stable_motor_feedback(true)
                .with_substeps(2);
            let mut pipeline = crate::geometry::CollisionPipeline::new();
            let mut contacts = Contacts::with_capacity(0);
            for step in 0..40 {
                for (cached, input) in [(true, &mut a), (false, &mut b)] {
                    solver
                        .step_with_collision_response(
                            &model,
                            input,
                            &control,
                            &mut pipeline,
                            &mut contacts,
                            1.0 / 120.0,
                            &mut output,
                            |_, _| false,
                            |model, state, _, wrenches, seed, _| {
                                wrenches[1].linear += Vec3::new(0.3, -0.1, 0.2);
                                if cached {
                                    let matrices = (0..model.articulation_count)
                                        .map(|art| {
                                            let matrix = crate::mass_matrix(model, state, art)?;
                                            let n = (matrix.len() as Real).sqrt() as usize;
                                            let factor =
                                                crate::rigid_impulse::MassFactor::new(&matrix, n)?
                                                    .into_full_rank();
                                            Ok((art, (matrix, factor)))
                                        })
                                        .collect::<Result<BTreeMap<_, _>>>()?;
                                    seed.record(model, state, matrices)?;
                                }
                                Ok(())
                            },
                            None,
                        )
                        .unwrap();
                    std::mem::swap(input, &mut output);
                }
                assert_eq!(
                    a.joint_q.host().unwrap(),
                    b.joint_q.host().unwrap(),
                    "q step={step}"
                );
                assert_eq!(
                    a.joint_qd.host().unwrap(),
                    b.joint_qd.host().unwrap(),
                    "qd step={step}"
                );
                assert_eq!(a.body_q.host().unwrap(), b.body_q.host().unwrap());
                assert_eq!(a.body_qd.host().unwrap(), b.body_qd.host().unwrap());
                assert_eq!(a.body_f.host().unwrap(), b.body_f.host().unwrap());
            }
        }
    }
}
