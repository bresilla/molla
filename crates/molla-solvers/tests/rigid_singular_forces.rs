use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::BodyDesc;
use molla_sim::{BodyParams, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

#[test]
fn continuous_force_respects_singular_angular_modes_at_offset_com() {
    for (gravity, substeps) in [(Vec3::ZERO, 1), (Vec3::ZERO, 8), (Vec3::Y * -9.81, 8)] {
        for scale in [1e-6, 1.0, 1e6] {
            for inertia in [Mat3::ZERO, Mat3::from_diagonal(Vec3::new(0.0, 0.2, 0.2))] {
                for offset in [Vec3::ZERO, Vec3::new(0.03, 0.1, -0.02)] {
                    let mut world = RigidWorld::new().unwrap();
                    world.solver = world.solver.with_damping(0.0).with_substeps(substeps);
                    world.scene.set_gravity(gravity).unwrap();
                    let rotation = Quat::from_rotation_y(0.3);
                    let body = world
                        .scene
                        .insert_body(BodyDesc {
                            params: BodyParams::new()
                                .with_pose(Transform::new(Vec3::new(0.0, 0.4, -0.2), rotation)),
                            additional_mass: Some(MassProperties {
                                mass: 2.0 * scale,
                                local_com: offset,
                                inertia: inertia * scale,
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    let force = Vec3::new(0.3, 0.1, -0.2);
                    world
                        .scene
                        .add_body_wrench(
                            body,
                            SpatialVector::new(force * scale, rotation * Vec3::X * (0.1 * scale)),
                        )
                        .unwrap();
                    for step in 1..=120 {
                        world.step(1.0 / 120.0).unwrap();
                        let velocity = world.scene.body_velocity(body).unwrap();
                        let expected = (force / 2.0 + gravity) * (step as f64 / 120.0);
                        assert!(
                            (velocity.linear - expected).length() < 1e-8,
                            "step={step}, offset={offset:?}, inertia={inertia:?}, velocity={velocity:?}"
                        );
                        assert!(
                            velocity.angular.length() < 1e-8,
                            "step={step}, offset={offset:?}, inertia={inertia:?}, velocity={velocity:?}"
                        );
                        assert!(world.quarantined_bodies().is_empty());
                    }
                }
            }
        }
    }
}

#[test]
fn singular_body_keeps_unlocked_spin_and_com_acceleration() {
    for offset in [Vec3::ZERO, Vec3::new(0.03, 0.1, -0.02)] {
        let mut world = RigidWorld::new().unwrap();
        world.solver = world.solver.with_damping(0.0);
        world.scene.set_gravity(Vec3::ZERO).unwrap();
        let rotation = Quat::from_rotation_z(0.4);
        let axis = rotation * Vec3::Y;
        let body = world
            .scene
            .insert_body(BodyDesc {
                params: BodyParams::new()
                    .with_pose(Transform::new(Vec3::new(20.0, 3.0, -7.0), rotation)),
                additional_mass: Some(MassProperties {
                    mass: 2.0,
                    local_com: offset,
                    inertia: Mat3::from_diagonal(Vec3::new(0.0, 0.2, 0.2)),
                }),
                ..Default::default()
            })
            .unwrap();
        let initial_linear = Vec3::new(0.2, -0.1, 0.3);
        world
            .scene
            .set_body_velocity(body, SpatialVector::new(initial_linear, axis * 0.03))
            .unwrap();
        let force = Vec3::new(0.3, 0.1, -0.2);
        world
            .scene
            .add_body_wrench(body, SpatialVector::new(force, axis * 0.06))
            .unwrap();
        for step in 1..=120 {
            world.step(1.0 / 120.0).unwrap();
            let velocity = world.scene.body_velocity(body).unwrap();
            let time = step as f64 / 120.0;
            assert!(
                (velocity.linear - initial_linear - force * (time / 2.0)).length() < 2e-5,
                "offset={offset:?}, step={step}, velocity={velocity:?}"
            );
            assert!(
                (velocity.angular - axis * (0.03 + time * 0.3)).length() < 1e-8,
                "offset={offset:?}, step={step}, velocity={velocity:?}"
            );
        }
    }
}

#[test]
fn raw_free_root_with_offset_frames_skips_zero_inertia_rotation() {
    use molla_core::BodyId;
    use molla_math::SpatialInertia;
    use molla_sim::{Contacts, ModelBuilder};
    use molla_solvers::{FeatherstoneSolver, Solver};

    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 2.0,
        com: Vec3::ZERO,
        inertia: Mat3::ZERO,
    }));
    builder
        .add_joint_free(
            BodyId::NONE,
            body,
            Transform::new(Vec3::new(0.0, 0.4, -0.2), Quat::from_rotation_x(0.2)),
            Transform::from_translation(Vec3::new(0.03, 0.1, -0.02)),
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    model.gravity.host_mut().unwrap().fill(Vec3::ZERO);
    let mut state = model.state().unwrap();
    let mut next = model.state().unwrap();
    let force = Vec3::new(0.3, 0.1, -0.2);
    state.body_f.host_mut().unwrap()[body.index()] =
        SpatialVector::new(force, Vec3::new(0.01, -0.03, 0.02));
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu()
        .with_rigorous_d_solve(true)
        .with_damping(0.0);
    for step in 1..=120 {
        solver
            .step(&model, &state, &control, &contacts, 1.0 / 960.0, &mut next)
            .unwrap();
        let velocity = next.body_qd.host().unwrap()[body.index()];
        assert!((velocity.linear - force * (step as f64 / 960.0 / 2.0)).length() < 1e-8);
        assert!(velocity.angular.length() < 1e-8);
        std::mem::swap(&mut state, &mut next);
    }
}
