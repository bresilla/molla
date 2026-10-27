//! Coupled compliant loop-closure wrenches for reduced-coordinate trees.

use molla_core::{Error, Result};
use molla_math::{Quat, Real, SpatialVector, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, JointSoftness, MotorModel};
use molla_sim::runtime::{JointHandle, RigidScene};
use molla_sim::{JointTargetMode, JointType, State};

pub(crate) fn brake_motor(damping: Real) -> JointMotor {
    JointMotor {
        mode: JointTargetMode::Velocity,
        damping,
        ..Default::default()
    }
}

#[cfg(test)]
#[path = "loop_forces/selection.rs"]
mod selection;

#[cfg(test)]
#[path = "loop_forces/blocks.rs"]
mod block_tests;

struct Row {
    joint: JointHandle,
    /// Joint anchor on the child, from the child's centre of mass.
    arm: Vec3,
    ends: Vec<(usize, SpatialVector)>,
    error: Real,
    rate: Real,
    softness: JointSoftness,
    motor: Option<JointMotor>,
    bounds: [Real; 2],
}

/// Force and torque a loop joint puts on its child body, world frame, the
/// torque about the joint's anchor on the child.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LoopReaction {
    pub force: Vec3,
    pub torque: Vec3,
}

fn rows(scene: &RigidScene, state: &State, selected: Option<&[bool]>) -> Result<Vec<Row>> {
    if selected.is_some_and(|mask| mask.len() != scene.model().body_count) {
        return Err(Error::Solver("loop body selection length mismatch".into()));
    }
    let poses = state.body_q.host()?;
    let velocities = state.body_qd.host()?;
    let centers = scene.model().body_com.host()?;
    let mut rows = Vec::new();
    for (id, joint) in scene.loop_joints() {
        if joint.kind == JointType::Distance || scene.joint_softness(id)?.is_none() {
            continue;
        }
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
        if let Some(selected) = selected {
            let mut awake = false;
            let mut asleep = false;
            for body in [Some(joint.child), joint.parent].into_iter().flatten() {
                if scene.body(body).unwrap().kind == molla_sim::runtime::BodyKind::Dynamic {
                    let active = selected[scene.body_index(body).unwrap().index()];
                    awake |= active;
                    asleep |= !active;
                }
            }
            if awake && asleep {
                return Err(Error::Solver(
                    "loop body selection splits a dependency".into(),
                ));
            }
            if !awake {
                continue;
            }
        }
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
        relative = relative.normalize();
        let vector = Vec3::new(relative.x, relative.y, relative.z);
        let sine = vector.length();
        let rotation = if sine > 1e-12 {
            vector * (2.0 * sine.atan2(relative.w) / sine)
        } else {
            vector * 2.0
        };
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
                && (motor.mode == JointTargetMode::Effort
                    || motor.stiffness > 0.0
                    || motor.damping > 0.0)
            {
                let target = match motor.mode {
                    JointTargetMode::Position | JointTargetMode::PositionVelocity => {
                        motor.target_position
                    }
                    _ => position,
                };
                let bounds = if motor.mode == JointTargetMode::Effort {
                    let effort = motor
                        .target_position
                        .clamp(-motor.max_force, motor.max_force);
                    [effort; 2]
                } else {
                    [-motor.max_force, motor.max_force]
                };
                terms.push((position - target, Some(motor), bounds));
            }
            if setting.brake > 0.0 {
                terms.push((
                    0.0,
                    Some(brake_motor(setting.brake)),
                    [Real::NEG_INFINITY, Real::INFINITY],
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
            let child_center = poses[child].position + poses[child].rotation * centers[child];
            for (error, motor, bounds) in terms {
                rows.push(Row {
                    joint: id,
                    arm: b.position - child_center,
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

/// Adds every loop joint's wrenches into `forces` and returns each joint's
/// reaction on its child. `islands` are the scene's joint-only islands.
pub(crate) fn apply(
    scene: &RigidScene,
    islands: &crate::rigid_islands::RigidIslands,
    cache: &mut crate::articulated_response::ArticulatedResponse<'_>,
    dt: Real,
    forces: &mut [SpatialVector],
) -> Result<Vec<(JointHandle, LoopReaction)>> {
    apply_selected(scene, islands, cache, dt, forces, None)
}

pub(crate) fn apply_selected(
    scene: &RigidScene,
    islands: &crate::rigid_islands::RigidIslands,
    cache: &mut crate::articulated_response::ArticulatedResponse<'_>,
    dt: Real,
    forces: &mut [SpatialVector],
    selected: Option<&[bool]>,
) -> Result<Vec<(JointHandle, LoopReaction)>> {
    let state = cache.state();
    let _span = tracing::debug_span!("loop_forces").entered();
    let rows = tracing::debug_span!("loop_rows").in_scope(|| rows(scene, state, selected))?;
    let count = rows.len();
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut reactions = if scene.model().articulation_count == 1 {
        apply_rows(scene, cache, dt, forces, &rows, &[0])?
    } else {
        let mut groups =
            std::collections::BTreeMap::<(Vec<usize>, Option<JointHandle>), Vec<Row>>::new();
        for row in rows {
            let joint = scene.joint(row.joint).unwrap();
            let group = islands
                .body_island(joint.child)
                .or_else(|| joint.parent.and_then(|body| islands.body_island(body)));
            let arts = group
                .map(|group| group.articulations.clone())
                .unwrap_or_default();
            let separate = arts.is_empty().then_some(row.joint);
            groups.entry((arts, separate)).or_default().push(row);
        }
        let mut reactions = Vec::new();
        for ((arts, _), rows) in groups {
            reactions.extend(apply_rows(scene, cache, dt, forces, &rows, &arts)?);
        }
        reactions
    };
    reactions.sort_unstable_by_key(|(joint, _)| *joint);
    if forces
        .iter()
        .any(|f| !f.linear.is_finite() || !f.angular.is_finite())
    {
        return Err(Error::Solver("non-finite loop joint wrench".into()));
    }
    Ok(reactions)
}

fn apply_rows(
    scene: &RigidScene,
    cache: &mut crate::articulated_response::ArticulatedResponse<'_>,
    dt: Real,
    forces: &mut [SpatialVector],
    rows: &[Row],
    arts: &[usize],
) -> Result<Vec<(JointHandle, LoopReaction)>> {
    let count = rows.len();
    let model = scene.model();
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let children = model.joint_child.host()?;
    let mut response = vec![0.0; count * count];
    let mut response_articulations = 0;
    for &art in arts {
        let begin = starts[art] as usize;
        let end = starts[art + 1] as usize;
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
        response_articulations += 1;
        let _response_span =
            tracing::debug_span!("loop_response", dofs = n, rows = count).entered();
        let project_span = tracing::debug_span!("loop_project_wrenches").entered();
        let mut projected = vec![vec![0.0; n]; count];
        for (r, row) in rows.iter().enumerate() {
            for &(body, wrench) in &row.ends {
                if !children[begin..end].contains(&(body as i32)) {
                    continue;
                }
                let j = cache.jacobian(art, body)?;
                project_wrench(j, wrench, &mut projected[r]);
            }
        }
        drop(project_span);
        if model.articulation_count > 1 {
            cache.add_gram_compact(art, n, projected, &mut response)?;
        } else {
            cache.add_gram(art, n, &projected, &mut response)?;
        }
    }
    let mut rhs = vec![0.0; count];
    let mut bounds = vec![[0.0; 2]; count];
    for (i, row) in rows.iter().enumerate() {
        if row
            .motor
            .is_some_and(|motor| motor.mode == JointTargetMode::Effort)
        {
            response[i * count + i] += 1.0;
            bounds[i] = row.bounds.map(|effort| effort * dt);
            continue;
        }
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
    let impulses = if response_articulations > 1 {
        bounded_solve_inner::<true, false>(&response, &rhs, &bounds)?
    } else {
        bounded_solve_inner::<false, false>(&response, &rhs, &bounds)?
    };
    let mut reactions: Vec<(JointHandle, LoopReaction)> = Vec::new();
    for (row, impulse) in rows.iter().zip(impulses) {
        for &(body, wrench) in &row.ends {
            forces[body].linear += wrench.linear * (impulse / dt);
            forces[body].angular += wrench.angular * (impulse / dt);
        }
        // The child is always the first end.
        let (force, torque) = (
            row.ends[0].1.linear * (impulse / dt),
            row.ends[0].1.angular * (impulse / dt),
        );
        if reactions
            .last()
            .is_none_or(|(joint, _)| *joint != row.joint)
        {
            reactions.push((row.joint, LoopReaction::default()));
        }
        let reaction = &mut reactions.last_mut().expect("pushed above").1;
        reaction.force += force;
        reaction.torque += torque - row.arm.cross(force);
    }
    Ok(reactions)
}

fn project_wrench(jacobian: &[Real], wrench: SpatialVector, projected: &mut [Real]) {
    let n = projected.len();
    debug_assert_eq!(jacobian.len(), 6 * n);
    let (lx, rest) = jacobian.split_at(n);
    let (ly, rest) = rest.split_at(n);
    let (lz, rest) = rest.split_at(n);
    let (ax, rest) = rest.split_at(n);
    let (ay, rest) = rest.split_at(n);
    let (az, _) = rest.split_at(n);
    for (col, value) in projected.iter_mut().enumerate() {
        *value += wrench.linear.x * lx[col]
            + wrench.linear.y * ly[col]
            + wrench.linear.z * lz[col]
            + wrench.angular.x * ax[col]
            + wrench.angular.y * ay[col]
            + wrench.angular.z * az[col];
    }
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

pub(crate) fn bounded_solve(
    matrix: &[Real],
    rhs: &[Real],
    bounds: &[[Real; 2]],
) -> Result<Vec<Real>> {
    bounded_solve_inner::<false, false>(matrix, rhs, bounds)
}

pub(crate) fn bounded_feedback_solve(
    matrix: &[Real],
    rhs: &[Real],
    bounds: &[[Real; 2]],
) -> Result<Vec<Real>> {
    bounded_solve_inner::<false, true>(matrix, rhs, bounds)
}

fn feedback_solve<const BLOCKS: bool>(
    matrix: &[Real],
    rhs: &[Real],
    n: usize,
) -> Result<Vec<Real>> {
    let scales: Vec<_> = (0..n).map(|i| matrix[i * n + i].sqrt()).collect();
    let mut normalized = Vec::with_capacity(n * n);
    for i in 0..n {
        normalized.extend((0..n).map(|j| matrix[i * n + j] / scales[i] / scales[j]));
    }
    let rhs: Vec<_> = rhs
        .iter()
        .zip(&scales)
        .map(|(value, scale)| value / scale)
        .collect();
    let result = if BLOCKS {
        crate::rigid_impulse::blocks::solve(&normalized, &rhs, n)?
    } else {
        crate::rigid_impulse::solve_mass(&normalized, &rhs, n)?
    };
    Ok(result
        .into_iter()
        .zip(scales)
        .map(|(value, scale)| value / scale)
        .collect())
}

fn bounded_solve_inner<const BLOCKS: bool, const FEEDBACK: bool>(
    matrix: &[Real],
    rhs: &[Real],
    bounds: &[[Real; 2]],
) -> Result<Vec<Real>> {
    let n = rhs.len();
    let _span = tracing::debug_span!("loop_bounded_solve", rows = n).entered();
    let mut x: Vec<_> = bounds.iter().map(|b| 0.0_f64.clamp(b[0], b[1])).collect();
    let mut fixed: Vec<bool> = bounds.iter().map(|b| b[0] == b[1]).collect();
    let mut free = Vec::with_capacity(n);
    let mut reduced = Vec::new();
    for _ in 0..(4 * n + 32) {
        free.clear();
        free.extend((0..n).filter(|&i| !fixed[i]));
        let size = free.len();
        let a = if size == n {
            matrix
        } else {
            reduced.clear();
            for &i in &free {
                reduced.extend(free.iter().map(|&j| matrix[i * n + j]));
            }
            &reduced[..]
        };
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
        let candidate = if FEEDBACK {
            feedback_solve::<BLOCKS>(a, &b, size)?
        } else if BLOCKS {
            crate::rigid_impulse::blocks::solve(a, &b, size)?
        } else {
            crate::rigid_impulse::solve_mass(a, &b, size)?
        };
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

    #[test]
    fn contiguous_wrench_projection_matches_indexed_arithmetic() {
        let wrenches = [
            SpatialVector::ZERO,
            SpatialVector::new(Vec3::X, Vec3::ZERO),
            SpatialVector::new(Vec3::ZERO, Vec3::Z),
            SpatialVector::new(Vec3::new(0.31, -0.72, 1.8), Vec3::new(-1.3, 0.28, -2.1)),
        ];
        for n in [0, 1, 6, 39, 78] {
            let j: Vec<_> = (0..6 * n).map(|k| (k as Real * 0.31).sin()).collect();
            let mut actual: Vec<_> = (0..n).map(|k| k as Real * 0.09).collect();
            let mut expected = actual.clone();
            for wrench in wrenches {
                for col in 0..n {
                    expected[col] += wrench.linear.x * j[col]
                        + wrench.linear.y * j[n + col]
                        + wrench.linear.z * j[2 * n + col]
                        + wrench.angular.x * j[3 * n + col]
                        + wrench.angular.y * j[4 * n + col]
                        + wrench.angular.z * j[5 * n + col];
                }
                project_wrench(&j, wrench, &mut actual);
                assert_eq!(actual, expected, "n={n}, wrench={wrench:?}");
            }
        }
    }

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
        let base = rows(&scene, &state, None).unwrap();
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
            rows(&scene, &next, None).unwrap()
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
    fn fixed_nonzero_impulses_contribute_to_other_rows() {
        let response = [2.0, 1.0, 1.0, 3.0];
        let result = bounded_solve(&response, &[0.0, 2.0], &[[0.5; 2], [-10.0, 10.0]]).unwrap();
        assert_eq!(result[0], 0.5);
        assert!((result[1] - 0.5).abs() < 1e-12);
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
