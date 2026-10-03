//! Hard external axis locks and bounds in authored endpoint frames.

use crate::distance_constraints::{DistanceBound, Feedback, Row};
use molla_core::{Error, Result};
use molla_math::{Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{JointDesc, JointHandle, RigidScene};
use molla_sim::{JointTargetMode, JointType, State};

type RotationPair = ([f64; 4], [f64; 4]);

fn rotation_product(a: RotationPair, b: RotationPair) -> RotationPair {
    let left = [a.0[3], a.0[0], a.0[1], a.0[2]];
    let low = [a.1[3], a.1[0], a.1[1], a.1[2]];
    let right = |v: [f64; 4]| {
        [
            [v[0], v[3], v[2], -v[1]],
            [v[1], -v[2], v[3], v[0]],
            [v[2], v[1], -v[0], v[3]],
            [v[3], -v[0], -v[1], -v[2]],
        ]
    };
    let high_right = right(b.0);
    let low_right = right(b.1);
    let mut result = ([0.; 4], [0.; 4]);
    for c in 0..4 {
        let mut high = 0.;
        let mut tail = 0.;
        for i in 0..4 {
            let product = left[i] * high_right[c][i];
            let sum = high + product;
            let virtual_product = sum - high;
            tail += (high - (sum - virtual_product))
                + (product - virtual_product)
                + left[i].mul_add(high_right[c][i], -product)
                + left[i] * low_right[c][i]
                + low[i] * high_right[c][i];
            high = sum;
        }
        let sum = high + tail;
        let virtual_tail = sum - high;
        result.0[c] = sum;
        result.1[c] = (high - (sum - virtual_tail)) + (tail - virtual_tail);
    }
    result
}

fn relative_rotation(parent: Option<Quat>, frames: [Transform; 2], child: Quat) -> Quat {
    let pair = |q: Quat| (q.to_array(), [0.; 4]);
    let mut a = parent.map_or(pair(frames[0].rotation), |p| {
        rotation_product(pair(p), pair(frames[0].rotation))
    });
    let b = rotation_product(pair(child), pair(frames[1].rotation));
    for i in 0..3 {
        a.0[i] = -a.0[i];
        a.1[i] = -a.1[i];
    }
    let relative = rotation_product(a, b);
    Quat::from_xyzw(
        relative.0[0] + relative.1[0],
        relative.0[1] + relative.1[1],
        relative.0[2] + relative.1[2],
        relative.0[3] + relative.1[3],
    )
    .normalize()
}

pub(crate) fn joints(scene: &RigidScene) -> impl Iterator<Item = (JointHandle, &JointDesc)> {
    scene
        .rigid_loop_joints()
        .filter(|(_, joint)| axis_kind(joint.kind) || joint.kind == JointType::Ball)
}

pub(crate) fn axis_kind(kind: JointType) -> bool {
    matches!(
        kind,
        JointType::Fixed | JointType::Revolute | JointType::Prismatic | JointType::D6
    )
}

pub(crate) fn velocity_limit(scene: &RigidScene, id: JointHandle, axis: JointAxis) -> Option<f64> {
    let settings = scene.joint_axis_settings(id, axis).ok()?;
    if scene.joint_softness(id).ok().flatten().is_none()
        && (settings.locked || settings.limits.is_some_and(|[lo, hi]| lo == hi))
    {
        return None;
    }
    settings.velocity_limit
}

pub(crate) fn velocity_joints(
    scene: &RigidScene,
) -> impl Iterator<Item = (JointHandle, &JointDesc)> {
    scene.loop_joints().filter(|(id, _)| {
        scene.joint_softness(*id).ok().flatten().is_some()
            && JointAxis::ALL
                .into_iter()
                .any(|axis| velocity_limit(scene, *id, axis).is_some())
    })
}

pub(crate) fn frames(joint: &JointDesc) -> [Transform; 2] {
    let mut frames = [joint.frame_parent, joint.frame_child];
    if matches!(joint.kind, JointType::Revolute | JointType::Prismatic) {
        let alignment = Quat::from_rotation_arc(Vec3::X, joint.axis.normalize());
        for frame in &mut frames {
            frame.rotation = (frame.rotation * alignment).normalize();
        }
    }
    frames
}

pub(crate) fn bounds(
    scene: &RigidScene,
    id: JointHandle,
    axis: JointAxis,
) -> Result<Vec<(f64, DistanceBound)>> {
    if scene.joint(id).unwrap().kind == JointType::Ball {
        return Ok(Vec::new());
    }
    if scene.joint_locked_axes(id)?.contains(axis) {
        return Ok(vec![(0.0, DistanceBound::Equality)]);
    }
    let Some([lo, hi]) = scene.joint_axis_settings(id, axis)?.limits else {
        return Ok(Vec::new());
    };
    Ok(if lo == hi {
        vec![(lo, DistanceBound::Equality)]
    } else {
        vec![(lo, DistanceBound::Minimum), (hi, DistanceBound::Maximum)]
    })
}

pub(crate) fn rows(
    scene: &RigidScene,
    state: &State,
    id: JointHandle,
    joint: &JointDesc,
    dt: f64,
) -> Result<Vec<Row>> {
    axis_rows(scene, state, id, joint, dt, false)
}

pub(crate) fn velocity_rows(
    scene: &RigidScene,
    state: &State,
    id: JointHandle,
    joint: &JointDesc,
    dt: f64,
) -> Result<Vec<Row>> {
    axis_rows(scene, state, id, joint, dt, true)
}

fn axis_rows(
    scene: &RigidScene,
    state: &State,
    id: JointHandle,
    joint: &JointDesc,
    dt: f64,
    velocity_only: bool,
) -> Result<Vec<Row>> {
    let poses = state.body_q.host()?;
    let velocities = state.body_qd.host()?;
    let centers = scene.model().body_com.host()?;
    let child = scene.body_index(joint.child).unwrap().index();
    let parent = joint
        .parent
        .map(|body| scene.body_index(body).unwrap().index());
    let frames = frames(joint);
    let a = parent.map_or(frames[0], |body| poses[body].compose(&frames[0]));
    let b = poses[child].compose(&frames[1]);
    let translation = a.rotation.conjugate() * (b.position - a.position);
    let mut relative = relative_rotation(
        parent.map(|body| poses[body].rotation),
        frames,
        poses[child].rotation,
    );
    if relative.w < 0.0 {
        relative = -relative;
    }
    let vector = Vec3::new(relative.x, relative.y, relative.z);
    let sine = vector.length();
    let rotation = if sine <= 0.125 {
        let square = sine * sine;
        vector
            * (2.0
                + square
                    * (1.0 / 3.0
                        + square * (3.0 / 20.0 + square * (5.0 / 56.0 + square * 35.0 / 576.0))))
    } else {
        vector * (2.0 * sine.atan2(relative.w) / sine)
    };
    let angle = rotation.length();
    let coefficient = if angle <= 0.25 {
        let square = angle * angle;
        1.0 / 12.0 + square * (1.0 / 720.0 + square * (1.0 / 30240.0 + square / 1209600.0))
    } else {
        (1.0 - 0.5 * angle / (0.5 * angle).tan()) / (angle * angle)
    };
    let mut rows = Vec::with_capacity(6);
    for axis in JointAxis::ALL {
        let i = axis as usize;
        let bounds = if velocity_only {
            Vec::new()
        } else {
            bounds(scene, id, axis)?
        };
        let settings = if axis.dof(joint.kind).is_ok() {
            scene.joint_axis_settings(id, axis)?
        } else {
            Default::default()
        };
        let motor = settings
            .motor
            .filter(|m| !velocity_only && m.mode != JointTargetMode::None);
        let brake = if velocity_only { 0.0 } else { settings.brake };
        let velocity_limit = velocity_limit(scene, id, axis);
        if bounds.is_empty() && motor.is_none() && brake == 0.0 && velocity_limit.is_none() {
            continue;
        }
        let angular = i >= 3;
        let mut direction = Vec3::ZERO;
        direction[i % 3] = 1.0;
        if angular {
            direction += 0.5 * rotation.cross(direction)
                + coefficient * rotation.cross(rotation.cross(direction));
        }
        direction = a.rotation * direction;
        let ends: Vec<_> = [(Some(child), 1.0), (parent, -1.0)]
            .into_iter()
            .filter_map(|(body, sign)| {
                body.map(|body| {
                    let center = poses[body].position + poses[body].rotation * centers[body];
                    let wrench = if angular {
                        SpatialVector::new(Vec3::ZERO, direction * sign)
                    } else {
                        SpatialVector::new(
                            direction * sign,
                            (b.position - center).cross(direction) * sign,
                        )
                    };
                    (body, wrench)
                })
            })
            .collect();
        let rate: f64 = ends
            .iter()
            .map(|(body, wrench)| {
                wrench.linear.dot(velocities[*body].linear)
                    + wrench.angular.dot(velocities[*body].angular)
            })
            .sum();
        let position = if angular {
            rotation[i % 3]
        } else {
            translation[i]
        };
        if let Some(cap) = velocity_limit {
            for sign in [1.0, -1.0].into_iter().take(if cap == 0.0 { 1 } else { 2 }) {
                rows.push(Row {
                    joint: Some(id),
                    cone: angular,
                    rigid: true,
                    bound: if cap == 0.0 {
                        DistanceBound::Equality
                    } else {
                        DistanceBound::Minimum
                    },
                    direction: direction * sign,
                    ends: ends
                        .iter()
                        .map(|&(body, wrench)| (body, wrench.scale(sign)))
                        .collect(),
                    direct: None,
                    target: -cap,
                    rate: rate * sign,
                    feedback: None,
                    velocity_only: true,
                });
            }
        }
        for (value, bound) in bounds {
            let sign = if bound == DistanceBound::Maximum {
                -1.0
            } else {
                1.0
            };
            rows.push(Row {
                joint: Some(id),
                cone: angular,
                rigid: true,
                bound,
                direction: direction * sign,
                ends: ends
                    .iter()
                    .map(|&(body, wrench)| (body, wrench.scale(sign)))
                    .collect(),
                direct: None,
                target: sign * (value - position) / dt,
                rate: rate * sign,
                feedback: None,
                velocity_only: false,
            });
        }
        for (stiffness, damping, target, impulses, effort, acceleration) in motor
            .into_iter()
            .map(|motor| {
                let stiffness = if motor.mode == JointTargetMode::Velocity {
                    0.0
                } else {
                    motor.stiffness
                };
                let target_rate = if matches!(
                    motor.mode,
                    JointTargetMode::Velocity | JointTargetMode::PositionVelocity
                ) {
                    motor.target_velocity
                } else {
                    0.0
                };
                let target_position = if matches!(
                    motor.mode,
                    JointTargetMode::Position | JointTargetMode::PositionVelocity
                ) {
                    motor.target_position - position
                } else {
                    0.0
                };
                (
                    stiffness,
                    motor.damping,
                    (target_position, target_rate),
                    [-dt * motor.max_force, dt * motor.max_force],
                    (motor.mode == JointTargetMode::Effort).then_some(
                        dt * motor
                            .target_position
                            .clamp(-motor.max_force, motor.max_force),
                    ),
                    motor.model == molla_sim::joint_control::MotorModel::Acceleration
                        && motor.mode != JointTargetMode::Effort,
                )
            })
            .chain((brake > 0.0).then_some((
                0.0,
                brake,
                (0.0, 0.0),
                [f64::NEG_INFINITY, f64::INFINITY],
                None,
                false,
            )))
        {
            let gain = dt * (damping + dt * stiffness);
            if effort.is_none() && gain == 0.0 {
                continue;
            }
            let (target, diagonal, impulses) = if let Some(effort) = effort {
                (0.0, 0.0, [effort; 2])
            } else {
                (
                    (damping * target.1 + stiffness * target.0) / (damping + dt * stiffness),
                    1.0 / gain,
                    impulses,
                )
            };
            if !target.is_finite() || !diagonal.is_finite() || impulses.iter().any(|p| p.is_nan()) {
                return Err(Error::Solver(
                    "nonfinite rigid-loop feedback equation".into(),
                ));
            }
            rows.push(Row {
                joint: Some(id),
                cone: angular,
                rigid: true,
                bound: DistanceBound::Inactive,
                direction,
                ends: ends.clone(),
                direct: None,
                target,
                rate,
                velocity_only: false,
                feedback: Some(Feedback {
                    impulses,
                    diagonal,
                    acceleration,
                }),
            });
        }
    }
    Ok(rows)
}
