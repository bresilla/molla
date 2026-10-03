use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder, eval_fk};
use molla_solvers::rnea::inverse_dynamics_wrenches;
use molla_solvers::{inverse_dynamics, mass_matrix, response_mass_matrix};

#[test]
fn analysis_locks_preserve_physical_mass_and_support_wrenches() {
    let mass = 2.0;
    let com = Vec3::new(0.2, -0.1, 0.3);
    let inertia = Mat3::from_cols(
        Vec3::new(1.0, 0.1, -0.2),
        Vec3::new(0.1, 2.0, 0.3),
        Vec3::new(-0.2, 0.3, 3.0),
    );
    for ball in [false, true] {
        for rotation in [
            Quat::IDENTITY,
            Quat::from_rotation_z(0.4) * Quat::from_rotation_y(0.3),
        ] {
            let mut builder = ModelBuilder::new();
            builder.set_world_gravity(WorldId(0), -Vec3::Y * 10.0);
            let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass,
                com,
                inertia,
            }));
            let frame = Transform {
                position: Vec3::new(4.0, -3.0, 5.0),
                rotation,
            };
            let joint = if ball {
                builder.add_joint_ball(BodyId::NONE, body, frame, Transform::IDENTITY)
            } else {
                builder.add_joint_d6(BodyId::NONE, body, frame, Transform::IDENTITY)
            }
            .unwrap();
            let count = if ball { 3 } else { 6 };
            let mut lower = vec![0.0; count];
            let mut upper = vec![0.0; count];
            lower[count - 1] = f64::NEG_INFINITY;
            upper[count - 1] = f64::INFINITY;
            builder.set_joint_limits(joint, &lower, &upper).unwrap();
            let model = builder.build().unwrap();
            let mut state = model.state().unwrap();
            eval_fk(&model, &mut state).unwrap();
            let analysis = mass_matrix(&model, &state, 0).unwrap();
            let response = response_mass_matrix(&model, &state, 0).unwrap();
            let angular = count - 3;
            for row in 0..3 {
                for column in 0..3 {
                    let expected = inertia.col(column)[row]
                        + mass
                            * (if row == column {
                                com.length_squared()
                            } else {
                                0.0
                            } - com[row] * com[column]);
                    let actual = response[(angular + row) * count + angular + column];
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            for row in 0..count {
                for column in 0..count {
                    let expected = if row == count - 1 && column == count - 1 {
                        response[row * count + column]
                    } else {
                        0.0
                    };
                    assert!((analysis[row * count + column] - expected).abs() < 1e-12);
                }
            }
            let acceleration = vec![0.0; count];
            let effort = inverse_dynamics(&model, &state, 0, &acceleration).unwrap();
            let physical =
                inverse_dynamics_wrenches(&model, &state, 0, &acceleration, &[SpatialVector::ZERO])
                    .unwrap();
            let force = Vec3::Y * mass * 10.0;
            let torque = (rotation * com).cross(force);
            assert!((physical.joints[0].linear - force).length() < 1e-12);
            assert!((physical.joints[0].angular - torque).length() < 1e-12);
            assert!(physical.efforts[angular].abs() > 1e-6);
            assert!((physical.efforts[angular] - (rotation * Vec3::X).dot(torque)).abs() < 1e-12);
            for slot in 0..count - 1 {
                assert_eq!(effort[slot], 0.0);
            }
            assert!((effort[count - 1] - physical.efforts[count - 1]).abs() < 1e-12);
        }
    }
}
