//! Coupled compliant loop-closure wrenches for reduced-coordinate trees.

use molla_core::{Error, Result};
use molla_math::{Quat, Real, SpatialVector, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, JointSoftness, MotorModel};
use molla_sim::runtime::RigidScene;
use molla_sim::{JointTargetMode, JointType, State};

struct Row {
    ends: Vec<(usize, SpatialVector)>,
    error: Real,
    rate: Real,
    softness: JointSoftness,
    motor: Option<JointMotor>,
    bounds: [Real; 2],
}

fn rows(scene: &RigidScene, state: &State) -> Result<Vec<Row>> {
    let poses = state.body_q.host()?;
    let velocities = state.body_qd.host()?;
    let centers = scene.model().body_com.host()?;
    let mut rows = Vec::new();
    for (id, joint) in scene.loop_joints() {
        if !scene.joint_settings(id).unwrap().enabled
            || !scene.body(joint.child).unwrap().enabled
            || joint
                .parent
                .is_some_and(|p| !scene.body(p).unwrap().enabled)
        {
            continue;
        }
        let child = scene.body_index(joint.child).unwrap().index();
        let parent = joint.parent.map(|p| scene.body_index(p).unwrap().index());
        let mut a = parent.map_or(joint.frame_parent, |p| {
            poses[p].compose(&joint.frame_parent)
        });
        let mut b = poses[child].compose(&joint.frame_child);
        if matches!(joint.kind, JointType::Revolute | JointType::Prismatic) {
            let alignment = Quat::from_rotation_arc(Vec3::X, joint.axis.normalize());
            a.rotation *= alignment;
            b.rotation *= alignment;
        }
        let translation = a.rotation.conjugate() * (b.position - a.position);
        let mut relative = a.rotation.conjugate() * b.rotation;
        if relative.w < 0.0 {
            relative = -relative;
        }
        let rotation = relative.to_scaled_axis();
        let locked = scene.joint_locked_axes(id)?;
        let softness = scene
            .joint_softness(id)?
            .ok_or_else(|| Error::Solver("loop joint lacks compliance".into()))?;
        for axis in JointAxis::ALL {
            let i = axis as usize;
            let position = if i < 3 {
                translation[i]
            } else {
                rotation[i - 3]
            };
            let setting = scene.joint_axis_settings(id, axis).unwrap_or_default();
            let interval = if locked.contains(axis) {
                Some([0.0, 0.0])
            } else {
                setting.limits
            };
            let mut terms = Vec::new();
            if let Some([lo, hi]) = interval
                && (lo == hi || position < lo || position > hi)
            {
                let bounds = if lo == hi {
                    [Real::NEG_INFINITY, Real::INFINITY]
                } else if position < lo {
                    [0.0, Real::INFINITY]
                } else {
                    [Real::NEG_INFINITY, 0.0]
                };
                if softness.natural_frequency > 0.0 {
                    terms.push((position - position.clamp(lo, hi), None, bounds));
                }
            }
            if let Some(motor) = setting.motor
                && motor.mode != JointTargetMode::None
                && (motor.stiffness > 0.0 || motor.damping > 0.0)
            {
                let target = match motor.mode {
                    JointTargetMode::Position | JointTargetMode::PositionVelocity => {
                        motor.target_position
                    }
                    _ => position,
                };
                terms.push((
                    position - target,
                    Some(motor),
                    [-motor.max_force, motor.max_force],
                ));
            }
            if terms.is_empty() {
                continue;
            }
            let mut direction = Vec3::ZERO;
            direction[i % 3] = 1.0;
            if i >= 3 {
                let angle = rotation.length();
                let coefficient = if angle < 1e-4 {
                    1.0 / 12.0 + angle * angle / 720.0
                } else {
                    (1.0 - 0.5 * angle / (0.5 * angle).tan()) / (angle * angle)
                };
                direction += 0.5 * rotation.cross(direction)
                    + coefficient * rotation.cross(rotation.cross(direction));
            }
            direction = a.rotation * direction;
            let ends: Vec<_> = [(Some(child), 1.0), (parent, -1.0)]
                .into_iter()
                .filter_map(|(body, sign)| {
                    body.map(|body| {
                        let center = poses[body].position + poses[body].rotation * centers[body];
                        let wrench = if i < 3 {
                            SpatialVector::new(direction, (b.position - center).cross(direction))
                        } else {
                            SpatialVector::new(Vec3::ZERO, direction)
                        };
                        (
                            body,
                            SpatialVector::new(wrench.linear * sign, wrench.angular * sign),
                        )
                    })
                })
                .collect();
            let rate: Real = ends
                .iter()
                .map(|(body, wrench)| {
                    wrench.linear.dot(velocities[*body].linear)
                        + wrench.angular.dot(velocities[*body].angular)
                })
                .sum();
            for (error, motor, bounds) in terms {
                rows.push(Row {
                    ends: ends.clone(),
                    error,
                    rate,
                    softness,
                    motor,
                    bounds,
                });
            }
        }
    }
    Ok(rows)
}

pub(crate) fn apply(
    scene: &RigidScene,
    cache: &mut crate::articulated_response::ArticulatedResponse<'_>,
    dt: Real,
    forces: &mut [SpatialVector],
) -> Result<()> {
    let state = cache.state();
    let _span = tracing::debug_span!("loop_forces").entered();
    let rows = tracing::debug_span!("loop_rows").in_scope(|| rows(scene, state))?;
    let count = rows.len();
    if count == 0 {
        return Ok(());
    }
    let model = scene.model();
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let children = model.joint_child.host()?;
    let mut response = vec![0.0; count * count];
    for (art, range) in starts.windows(2).enumerate() {
        let begin = range[0] as usize;
        let end = range[1] as usize;
        let n = (offsets[end] - offsets[begin]) as usize;
        if n == 0
            || !rows.iter().any(|r| {
                r.ends
                    .iter()
                    .any(|(b, _)| children[begin..end].contains(&(*b as i32)))
            })
        {
            continue;
        }
        let _response_span =
            tracing::debug_span!("loop_response", dofs = n, rows = count).entered();
        let mut projected = vec![vec![0.0; n]; count];
        for (r, row) in rows.iter().enumerate() {
            for &(body, wrench) in &row.ends {
                if !children[begin..end].contains(&(body as i32)) {
                    continue;
                }
                let j = cache.jacobian(art, body)?;
                for col in 0..n {
                    projected[r][col] += wrench.linear.x * j[col]
                        + wrench.linear.y * j[n + col]
                        + wrench.linear.z * j[2 * n + col]
                        + wrench.angular.x * j[3 * n + col]
                        + wrench.angular.y * j[4 * n + col]
                        + wrench.angular.z * j[5 * n + col];
                }
            }
        }
        cache.add_gram(art, n, &projected, &mut response)?;
    }
    let mut rhs = vec![0.0; count];
    let mut bounds = vec![[0.0; 2]; count];
    for (i, row) in rows.iter().enumerate() {
        let inverse_mass = response[i * count + i].max(0.0);
        let mass = if inverse_mass > 1e-14 {
            1.0 / inverse_mass
        } else {
            0.0
        };
        let (stiffness, damping, target_rate) = if let Some(motor) = row.motor {
            let scale = if motor.model == MotorModel::Acceleration {
                mass
            } else {
                1.0
            };
            let stiffness = match motor.mode {
                JointTargetMode::Velocity => 0.0,
                _ => motor.stiffness * scale,
            };
            let rate = match motor.mode {
                JointTargetMode::Velocity | JointTargetMode::PositionVelocity => {
                    motor.target_velocity
                }
                _ => 0.0,
            };
            (stiffness, motor.damping * scale, rate)
        } else {
            let omega = std::f64::consts::TAU * row.softness.natural_frequency;
            (
                mass * omega * omega,
                mass * 2.0 * row.softness.damping_ratio * omega,
                0.0,
            )
        };
        let gain = dt * (damping + dt * stiffness);
        if gain > 0.0 && mass > 0.0 {
            response[i * count + i] += 1.0 / gain;
            rhs[i] = -row.rate
                + (damping * target_rate - stiffness * row.error) / (damping + dt * stiffness);
            bounds[i] = [row.bounds[0] * dt, row.bounds[1] * dt];
        } else {
            response[i * count + i] += 1.0;
        }
    }
    let impulses = bounded_solve(&response, &rhs, &bounds)?;
    for (row, impulse) in rows.iter().zip(impulses) {
        for &(body, wrench) in &row.ends {
            forces[body].linear += wrench.linear * (impulse / dt);
            forces[body].angular += wrench.angular * (impulse / dt);
        }
    }
    if forces
        .iter()
        .any(|f| !f.linear.is_finite() || !f.angular.is_finite())
    {
        return Err(Error::Solver("non-finite loop joint wrench".into()));
    }
    Ok(())
}

#[cfg(test)]
fn add_mass_response(
    mass: &[Real],
    n: usize,
    projected: &[Vec<Real>],
    response: &mut [Real],
) -> Result<()> {
    crate::rigid_impulse::MassFactor::new(mass, n)?.add_gram(projected, response)
}

fn bounded_solve(matrix: &[Real], rhs: &[Real], bounds: &[[Real; 2]]) -> Result<Vec<Real>> {
    let n = rhs.len();
    let _span = tracing::debug_span!("loop_bounded_solve", rows = n).entered();
    let mut x = vec![0.0; n];
    let mut fixed: Vec<bool> = bounds.iter().map(|b| b[0] == b[1]).collect();
    for _ in 0..(4 * n + 32) {
        let free: Vec<_> = (0..n).filter(|&i| !fixed[i]).collect();
        let size = free.len();
        let a: Vec<_> = free
            .iter()
            .flat_map(|&i| free.iter().map(move |&j| matrix[i * n + j]))
            .collect();
        let b: Vec<_> = free
            .iter()
            .map(|&i| {
                rhs[i]
                    - (0..n)
                        .filter(|&j| fixed[j])
                        .map(|j| matrix[i * n + j] * x[j])
                        .sum::<Real>()
            })
            .collect();
        let candidate = crate::rigid_impulse::solve_mass(&a, &b, size)?;
        let mut alpha: Real = 1.0;
        let mut hit = None;
        for (&i, &value) in free.iter().zip(&candidate) {
            let delta = value - x[i];
            let fraction = if value < bounds[i][0] {
                (bounds[i][0] - x[i]) / delta
            } else if value > bounds[i][1] {
                (bounds[i][1] - x[i]) / delta
            } else {
                continue;
            };
            if fraction <= alpha {
                alpha = fraction.max(0.0);
                hit = Some(i);
            }
        }
        for (&i, &value) in free.iter().zip(candidate.iter()) {
            x[i] = (x[i] + alpha * (value - x[i])).clamp(bounds[i][0], bounds[i][1]);
        }
        if let Some(i) = hit {
            fixed[i] = true;
            continue;
        }
        let release = (0..n)
            .filter(|&i| fixed[i] && bounds[i][0] != bounds[i][1])
            .find(|&i| {
                let gradient = (0..n).map(|j| matrix[i * n + j] * x[j]).sum::<Real>() - rhs[i];
                let tolerance = 1e-10 * (1.0 + rhs[i].abs());
                (x[i] == bounds[i][0] && gradient < -tolerance)
                    || (x[i] == bounds[i][1] && gradient > tolerance)
            });
        if let Some(i) = release {
            fixed[i] = false;
        } else {
            return Ok(x);
        }
    }
    Err(Error::Solver(
        "loop joint active-set solve did not converge".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::dense_solve::{dense_chol, forward_sub};
    use molla_math::{Mat3, Transform};
    use molla_sim::runtime::{BodyDesc, JointDesc};
    use molla_sim::{BodyParams, MassProperties};

    fn response_fixture(n: usize, rows: usize) -> (Vec<Real>, Vec<Vec<Real>>) {
        let mut lower = vec![0.0; n * n];
        for i in 0..n {
            lower[i * n + i] = 0.2 + i as Real;
            for j in 0..i {
                lower[i * n + j] = ((i + 3 * j) % 7) as Real * 0.03;
            }
        }
        let mut mass = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                mass[i * n + j] = (0..n).map(|k| lower[i * n + k] * lower[j * n + k]).sum();
            }
        }
        let projected = (0..rows)
            .map(|row| {
                (0..n)
                    .map(|col| ((row * 13 + col * 7) % 17) as Real - 8.0)
                    .collect()
            })
            .collect();
        (mass, projected)
    }

    #[test]
    fn mass_response_matches_inverse_products_and_accumulates() {
        for singular in [false, true] {
            for scale in [1e-60, 1.0, 1e60] {
                let n = 9;
                let count = 7;
                let (mut mass, mut projected) = response_fixture(n, count);
                for i in 0..n {
                    for j in 0..n {
                        mass[i * n + j] *= scale;
                        if singular && (i == n - 1 || j == n - 1) {
                            mass[i * n + j] = 0.0;
                        }
                    }
                }
                projected[0].fill(0.0);
                let mut response = vec![0.0; count * count];
                add_mass_response(&mass, n, &projected, &mut response).unwrap();
                add_mass_response(&mass, n, &projected, &mut response).unwrap();
                for col in 0..count {
                    let solved =
                        crate::rigid_impulse::solve_mass(&mass, &projected[col], n).unwrap();
                    for row in 0..count {
                        let expected: Real = projected[row]
                            .iter()
                            .zip(&solved)
                            .map(|(a, b)| 2.0 * a * b)
                            .sum();
                        let actual = response[row * count + col];
                        assert!(
                            (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0 / scale),
                            "singular={singular} scale={scale} row={row} col={col}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "wall-clock CPU benchmark; run in release without concurrent builds"]
    fn full_rank_loop_response_timings() {
        let (mass, projected) = response_fixture(39, 22);
        let mut response = vec![0.0; 22 * 22];
        for batch in 0..5 {
            for gram in if batch % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = std::time::Instant::now();
                for _ in 0..2000 {
                    response.fill(0.0);
                    let mass = std::hint::black_box(&mass);
                    if gram {
                        add_mass_response(mass, 39, &projected, &mut response).unwrap();
                    } else {
                        let mut factor = mass.clone();
                        dense_chol(&mut factor, 39).unwrap();
                        for col in 0..22 {
                            let solved = molla_math::dense_solve::back_sub(
                                &factor,
                                &forward_sub(&factor, &projected[col], 39).unwrap(),
                                39,
                            )
                            .unwrap();
                            for row in 0..22 {
                                response[row * 22 + col] += projected[row]
                                    .iter()
                                    .zip(&solved)
                                    .map(|(a, b)| a * b)
                                    .sum::<Real>();
                            }
                        }
                    }
                    std::hint::black_box(&response);
                }
                eprintln!(
                    "39-DOF/22-row response gram={gram} batch {batch}: {:.3} us/call",
                    start.elapsed().as_secs_f64() * 1e6 / 2000.0
                );
            }
        }
    }

    #[test]
    fn closure_rows_match_pose_derivatives_and_conserve_momentum() {
        let mut scene = RigidScene::new().unwrap();
        let mut bodies = Vec::new();
        for (position, rotation) in [
            (
                Vec3::new(20.0, 3.0, -4.0),
                Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.4, 0.2),
            ),
            (
                Vec3::new(20.5, 3.2, -3.8),
                Quat::from_euler(glam::EulerRot::XYZ, -0.1, 0.2, 0.4),
            ),
        ] {
            bodies.push(
                scene
                    .insert_body(BodyDesc {
                        params: BodyParams::new().with_pose(Transform::new(position, rotation)),
                        additional_mass: Some(MassProperties {
                            mass: 2.0,
                            local_com: Vec3::new(0.2, -0.1, 0.3),
                            inertia: Mat3::IDENTITY,
                        }),
                        ..Default::default()
                    })
                    .unwrap(),
            );
        }
        scene
            .insert_loop_joint(
                JointDesc {
                    parent: Some(bodies[0]),
                    child: bodies[1],
                    kind: JointType::Fixed,
                    axis: Vec3::X,
                    frame_parent: Transform::new(
                        Vec3::new(0.3, 0.2, -0.1),
                        Quat::from_rotation_z(0.2),
                    ),
                    frame_child: Transform::new(
                        Vec3::new(-0.2, 0.1, 0.2),
                        Quat::from_rotation_x(-0.3),
                    ),
                },
                JointSoftness {
                    natural_frequency: 20.0,
                    damping_ratio: 1.0,
                },
            )
            .unwrap();
        let mut state = scene.state().clone_host().unwrap();
        state.body_qd.host_mut().unwrap()[0] =
            SpatialVector::new(Vec3::new(0.3, -0.2, 0.5), Vec3::new(0.2, 0.3, -0.1));
        state.body_qd.host_mut().unwrap()[1] =
            SpatialVector::new(Vec3::new(-0.1, 0.4, 0.2), Vec3::new(-0.3, 0.1, 0.2));
        let base = rows(&scene, &state).unwrap();
        let h = 1e-6;
        let shift = |dt: Real| {
            let mut next = state.clone_host().unwrap();
            for body in 0..2 {
                let pose = state.body_q.host().unwrap()[body];
                let velocity = state.body_qd.host().unwrap()[body];
                let com = scene.model().body_com.host().unwrap()[body];
                let rotation = Quat::from_scaled_axis(velocity.angular * dt) * pose.rotation;
                let center = pose.position + pose.rotation * com + velocity.linear * dt;
                next.body_q.host_mut().unwrap()[body] =
                    Transform::new(center - rotation * com, rotation);
            }
            rows(&scene, &next).unwrap()
        };
        let plus = shift(h);
        let minus = shift(-h);
        for (i, row) in base.iter().enumerate() {
            let derivative = (plus[i].error - minus[i].error) / (2.0 * h);
            assert!(
                (row.rate - derivative).abs() < 1e-8,
                "axis {i}: {} != {derivative}",
                row.rate
            );
            let mut force = Vec3::ZERO;
            let mut torque = Vec3::ZERO;
            for &(body, wrench) in &row.ends {
                let pose = state.body_q.host().unwrap()[body];
                let center =
                    pose.position + pose.rotation * scene.model().body_com.host().unwrap()[body];
                force += wrench.linear;
                torque += wrench.angular + center.cross(wrench.linear);
            }
            assert!(force.length() < 1e-12 && torque.length() < 1e-12);
        }
    }

    #[test]
    fn coupled_bounded_solve_satisfies_complementarity() {
        let a = [2.0, 1.2, -0.3, 1.2, 3.0, 0.8, -0.3, 0.8, 2.0];
        let bounds = [[0.0, 0.5], [-0.2, 0.2], [Real::NEG_INFINITY, 0.0]];
        for b in [[2.0, -1.0, 3.0], [-3.0, 2.0, -4.0], [0.1, -0.1, -0.1]] {
            let x = bounded_solve(&a, &b, &bounds).unwrap();
            for i in 0..3 {
                let g = (0..3).map(|j| a[3 * i + j] * x[j]).sum::<Real>() - b[i];
                assert!(x[i] >= bounds[i][0] && x[i] <= bounds[i][1]);
                if x[i] == bounds[i][0] {
                    assert!(g >= -1e-10);
                } else if x[i] == bounds[i][1] {
                    assert!(g <= 1e-10);
                } else {
                    assert!(g.abs() < 1e-10);
                }
            }
        }
    }
}
