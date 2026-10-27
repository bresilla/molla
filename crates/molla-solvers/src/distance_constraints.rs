//! Coupled radial and cone impulses on articulated endpoint Jacobians.

use crate::cone_constraints::{self, ConeReading};
use crate::{
    FeatherstoneSolver, articulated_response::ArticulatedResponse, rigid_islands::RigidIslands,
};
use molla_core::{Error, Result};
use molla_math::{Quat, Real, SpatialVector, Vec3};
use molla_sim::runtime::{DistanceLimits, JointDesc, JointHandle, RigidScene};
use molla_sim::{JointType, State};
use std::collections::BTreeMap;

#[derive(Default)]
pub(crate) struct ConstraintReadings {
    pub distance: BTreeMap<JointHandle, DistanceReading>,
    pub cones: BTreeMap<JointHandle, ConeReading>,
    pub rigid: BTreeMap<JointHandle, SpatialVector>,
}

pub(crate) fn point_constraints(
    scene: &RigidScene,
) -> impl Iterator<Item = (JointHandle, &JointDesc, DistanceLimits)> {
    scene
        .distance_joints()
        .chain(scene.rigid_ball_joints().map(|(id, joint)| {
            (
                id,
                joint,
                DistanceLimits {
                    min: None,
                    max: Some(0.0),
                },
            )
        }))
}

impl ConstraintReadings {
    pub fn accumulate(&mut self, next: Self, weight: Real, dt: Real) {
        for (joint, reading) in next.rigid {
            let entry = self.rigid.entry(joint).or_default();
            *entry = *entry + reading.scale(weight);
        }
        for (joint, reading) in next.distance {
            let entry = self.distance.entry(joint).or_default();
            let force = entry.force + reading.force * weight;
            let active = if reading.active == DistanceBound::Inactive {
                entry.active
            } else {
                reading.active
            };
            *entry = reading;
            entry.force = force;
            entry.active = active;
            entry.sample_dt = dt;
        }
        for (joint, reading) in next.cones {
            let entry = self.cones.entry(joint).or_default();
            let torque = entry.torque + reading.torque * weight;
            let active = entry.active || reading.active;
            *entry = reading;
            entry.torque = torque;
            entry.active = active;
            entry.sample_dt = dt;
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DistanceBound {
    #[default]
    Inactive,
    Minimum,
    Maximum,
    Equality,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DistanceReading {
    pub separation: Real,
    pub violation: Real,
    /// Last bound carrying an impulse during this frame, or equality.
    pub active: DistanceBound,
    /// Frame-average force on the child at its anchor, in world coordinates.
    pub force: Vec3,
    pub sample_dt: Real,
}

#[derive(Clone, Copy)]
pub(crate) struct Feedback {
    pub impulses: [Real; 2],
    pub diagonal: Real,
    pub acceleration: bool,
}

pub(crate) struct Row {
    pub joint: Option<JointHandle>,
    pub cone: bool,
    pub rigid: bool,
    pub bound: DistanceBound,
    pub direction: Vec3,
    pub ends: Vec<(usize, SpatialVector)>,
    pub direct: Option<(usize, Real)>,
    pub target: Real,
    pub rate: Real,
    pub feedback: Option<Feedback>,
    pub velocity_only: bool,
}

pub(crate) fn enabled(scene: &RigidScene, id: JointHandle, joint: &JointDesc) -> bool {
    scene.joint_settings(id).unwrap().enabled
        && scene.body(joint.child).unwrap().enabled
        && joint.parent.is_none_or(|p| scene.body(p).unwrap().enabled)
}

/// Coincident radial direction, oriented from the lower stable body handle to the higher.
pub(crate) fn coincident_axis(joint: &JointDesc) -> Vec3 {
    if joint.parent.is_some_and(|parent| parent > joint.child) {
        -Vec3::X
    } else {
        Vec3::X
    }
}

pub(crate) fn anchors(scene: &RigidScene, state: &State, joint: &JointDesc) -> Result<[Vec3; 2]> {
    let poses = state.body_q.host()?;
    let a = joint.parent.map_or(joint.frame_parent.position, |p| {
        poses[scene.body_index(p).unwrap().index()]
            .compose(&joint.frame_parent)
            .position
    });
    let b = poses[scene.body_index(joint.child).unwrap().index()]
        .compose(&joint.frame_child)
        .position;
    Ok([a, b])
}

fn equality_gap(gap: Real, scale: Real) -> Real {
    if gap.abs() <= 64. * Real::EPSILON * scale {
        0.
    } else {
        gap
    }
}

fn radial_rows(
    scene: &RigidScene,
    state: &State,
    id: JointHandle,
    joint: &JointDesc,
    limits: DistanceLimits,
    dt: Real,
) -> Result<Vec<Row>> {
    let [a, b] = anchors(scene, state, joint)?;
    let delta = b - a;
    let distance = delta.length();
    if !distance.is_finite() {
        return Err(Error::Solver(
            "non-finite distance constraint geometry".into(),
        ));
    }
    let normal = if distance > 1e-12 {
        delta / distance
    } else {
        coincident_axis(joint)
    };
    let mut terms = Vec::new();
    let scale = a.length() + b.length() + distance;
    if limits.max == Some(0.0) {
        for (i, axis) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
            terms.push((
                axis,
                equality_gap(-delta[i], scale) / dt,
                DistanceBound::Equality,
            ));
        }
    } else if let (Some(lo), Some(hi)) = (limits.min, limits.max)
        && lo == hi
    {
        terms.push((
            normal,
            equality_gap(lo - distance, scale + lo.abs()) / dt,
            DistanceBound::Equality,
        ));
    } else {
        if let Some(min) = limits.min.filter(|min| *min > 0.0) {
            terms.push((normal, (min - distance) / dt, DistanceBound::Minimum));
        }
        if let Some(max) = limits.max {
            terms.push((-normal, (distance - max) / dt, DistanceBound::Maximum));
        }
    }
    let poses = state.body_q.host()?;
    let velocities = state.body_qd.host()?;
    let centers = scene.model().body_com.host()?;
    Ok(terms
        .into_iter()
        .map(|(direction, target, bound)| {
            let ends: Vec<_> = [(Some(joint.child), b, 1.0), (joint.parent, a, -1.0)]
                .into_iter()
                .filter_map(|(body, point, sign)| {
                    body.map(|body| {
                        let index = scene.body_index(body).unwrap().index();
                        let center = poses[index].position + poses[index].rotation * centers[index];
                        let force = direction * sign;
                        (
                            index,
                            SpatialVector::new(force, (point - center).cross(force)),
                        )
                    })
                })
                .collect();
            let rate = ends
                .iter()
                .map(|(body, w)| {
                    w.linear.dot(velocities[*body].linear)
                        + w.angular.dot(velocities[*body].angular)
                })
                .sum();
            Row {
                joint: Some(id),
                cone: false,
                rigid: false,
                bound,
                direction,
                ends,
                direct: None,
                target,
                rate,
                feedback: None,
                velocity_only: false,
            }
        })
        .collect())
}

pub(crate) fn prepare(
    scene: &RigidScene,
    input: &State,
    dt: Real,
    selected: Option<&[bool]>,
) -> Result<(
    BTreeMap<(Vec<usize>, Option<JointHandle>), Vec<Row>>,
    ConstraintReadings,
)> {
    if selected.is_some_and(|mask| mask.len() != scene.model().articulation_count) {
        return Err(Error::Solver(
            "invalid distance articulation selection".into(),
        ));
    }
    if point_constraints(scene).next().is_none()
        && scene.cone_joints().next().is_none()
        && crate::rigid_loop_constraints::joints(scene)
            .next()
            .is_none()
        && crate::rigid_loop_constraints::velocity_joints(scene)
            .next()
            .is_none()
    {
        return Ok((BTreeMap::new(), ConstraintReadings::default()));
    }
    let islands = RigidIslands::build(scene, [])?;
    let mut groups: BTreeMap<(Vec<usize>, Option<JointHandle>), Vec<Row>> = BTreeMap::new();
    let mut readings = ConstraintReadings::default();
    for (id, joint, limits) in point_constraints(scene) {
        if !enabled(scene, id, joint) {
            continue;
        }
        let group = islands
            .body_island(joint.child)
            .or_else(|| joint.parent.and_then(|p| islands.body_island(p)));
        let arts = group.map(|g| g.articulations.clone()).unwrap_or_default();
        if let Some(selected) = selected {
            let active = arts.iter().filter(|&&art| selected[art]).count();
            if active != 0 && active != arts.len() {
                return Err(Error::Solver(
                    "activation splits a distance dependency".into(),
                ));
            }
            if !arts.is_empty() && active == 0 {
                continue;
            }
        }
        let independent = arts.is_empty().then_some(id);
        groups
            .entry((arts, independent))
            .or_default()
            .extend(radial_rows(scene, input, id, joint, limits, dt)?);
        readings.distance.insert(
            id,
            DistanceReading {
                sample_dt: dt,
                ..Default::default()
            },
        );
    }
    for (id, joint, velocity_only) in crate::rigid_loop_constraints::joints(scene)
        .map(|(id, joint)| (id, joint, false))
        .chain(
            crate::rigid_loop_constraints::velocity_joints(scene)
                .map(|(id, joint)| (id, joint, true)),
        )
    {
        if !enabled(scene, id, joint) {
            continue;
        }
        let group = islands
            .body_island(joint.child)
            .or_else(|| joint.parent.and_then(|parent| islands.body_island(parent)));
        let arts = group
            .map(|group| group.articulations.clone())
            .unwrap_or_default();
        if let Some(selected) = selected {
            let active = arts.iter().filter(|&&art| selected[art]).count();
            if active != 0 && active != arts.len() {
                return Err(Error::Solver(
                    "activation splits a rigid loop dependency".into(),
                ));
            }
            if !arts.is_empty() && active == 0 {
                continue;
            }
        }
        let independent = arts.is_empty().then_some(id);
        let rows = if velocity_only {
            crate::rigid_loop_constraints::velocity_rows(scene, input, id, joint, dt)?
        } else {
            crate::rigid_loop_constraints::rows(scene, input, id, joint, dt)?
        };
        groups.entry((arts, independent)).or_default().extend(rows);
        readings.rigid.insert(id, SpatialVector::ZERO);
    }
    for (id, joint, limits) in scene.cone_joints() {
        if !enabled(scene, id, joint) {
            continue;
        }
        let group = islands
            .body_island(joint.child)
            .or_else(|| joint.parent.and_then(|p| islands.body_island(p)));
        let arts = group.map(|g| g.articulations.clone()).unwrap_or_default();
        if let Some(selected) = selected {
            let active = arts.iter().filter(|&&art| selected[art]).count();
            if active != 0 && active != arts.len() {
                return Err(Error::Solver("activation splits a cone dependency".into()));
            }
            if !arts.is_empty() && active == 0 {
                continue;
            }
        }
        let (parent_rotation, geometry) = cone_constraints::geometry(scene, input, joint, limits)?;
        let independent = arts.is_empty().then_some(id);
        let rows = groups.entry((arts, independent)).or_default();
        let velocities = input.body_qd.host()?;
        for cone in &geometry.rows[..geometry.row_count] {
            let scale = if cone.angular.length() > 0.0 {
                cone.angular.length()
            } else {
                1.0
            };
            let direction = parent_rotation * (-cone.angular / scale);
            let ends: Vec<_> = [(Some(joint.child), 1.0), (joint.parent, -1.0)]
                .into_iter()
                .filter_map(|(body, sign)| {
                    body.map(|body| {
                        (
                            scene.body_index(body).unwrap().index(),
                            SpatialVector::new(Vec3::ZERO, direction * sign),
                        )
                    })
                })
                .collect();
            let rate = ends
                .iter()
                .map(|(body, w)| w.angular.dot(velocities[*body].angular))
                .sum();
            rows.push(Row {
                joint: Some(id),
                cone: true,
                rigid: false,
                bound: if cone.bilateral {
                    DistanceBound::Equality
                } else {
                    DistanceBound::Maximum
                },
                direction,
                ends,
                direct: None,
                target: cone.error / (scale * dt),
                feedback: None,
                velocity_only: false,
                rate,
            });
        }
        readings.cones.insert(
            id,
            ConeReading {
                sample_dt: dt,
                ..Default::default()
            },
        );
    }
    let boundary_delta = crate::kinematic_motion::velocity_delta(scene.model(), input, dt)?;
    for ((arts, _), rows) in &mut groups {
        crate::drive_response::scale_feedback(scene, input, arts, rows)?;
    }
    for rows in groups.values_mut() {
        for row in rows {
            row.rate += crate::kinematic_motion::project(&boundary_delta, &row.ends);
        }
    }
    Ok((groups, readings))
}

pub(crate) fn scalar_bound_axes(kind: u32) -> usize {
    match kind {
        v if v == JointType::Prismatic as u32 || v == JointType::Revolute as u32 => 1,
        v if v == JointType::D6 as u32 => 6,
        v if v == JointType::Ball as u32 => 3,
        _ => 0,
    }
}

pub(crate) fn has_scalar_bounds(model: &molla_sim::Model, joint: usize) -> Result<bool> {
    if model.joint_enabled.host()?[joint] == 0 {
        return Ok(false);
    }
    let offsets = model.joint_dof_offset.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    let speed = model.joint_velocity_limit.host()?;
    let kind = model.joint_type.host()?[joint];
    let axes = scalar_bound_axes(kind);
    let first = offsets[joint] as usize;
    if kind == JointType::Ball as u32 {
        return Ok(molla_sim::joint_control::ball_coupled_stops(
            &lower[first..first + 3],
            &upper[first..first + 3],
            &speed[first..first + 3],
        ));
    }
    if kind == JointType::D6 as u32 {
        return Ok(molla_sim::joint_control::d6_coupled_stops(
            &lower[first..first + 6],
            &upper[first..first + 6],
            &speed[first..first + 6],
        ));
    }
    Ok((first..offsets[joint + 1] as usize).any(|dof| {
        let locked = lower[dof].is_finite()
            && upper[dof].is_finite()
            && (upper[dof] - lower[dof]).abs() < 1e-12;
        !locked
            && (speed[dof].is_finite()
                || (dof - first < axes && (lower[dof].is_finite() || upper[dof].is_finite())))
    }))
}

pub(crate) fn scalar_rows(
    model: &molla_sim::Model,
    input: &State,
    arts: &[usize],
    dt: Real,
    rows: &mut Vec<Row>,
) -> Result<()> {
    scalar_rows_with_prediction(model, input, arts, dt, None, rows)
}

pub(crate) fn scalar_rows_with_prediction(
    model: &molla_sim::Model,
    input: &State,
    arts: &[usize],
    dt: Real,
    prediction: Option<&[Real]>,
    rows: &mut Vec<Row>,
) -> Result<()> {
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    for &art in arts {
        for joint in starts[art] as usize..starts[art + 1] as usize {
            if model.joint_enabled.host()?[joint] == 0 {
                continue;
            }
            let kind = model.joint_type.host()?[joint];
            let angular = if kind == JointType::Ball as u32 { 0 } else { 3 };
            for axis in 0..scalar_bound_axes(kind) {
                let dof = offsets[joint] as usize + axis;
                let lo = model.joint_limit_lower.host()?[dof];
                let hi = model.joint_limit_upper.host()?[dof];
                let coordinate = model.joint_coord_offset.host()?[joint] as usize;
                if axis >= angular {
                    if axis != angular {
                        continue;
                    }
                    let q = input.joint_q.host()?;
                    let start = offsets[joint] as usize + angular;
                    let initial = Quat::from_xyzw(
                        q[coordinate + angular],
                        q[coordinate + angular + 1],
                        q[coordinate + angular + 2],
                        q[coordinate + angular + 3],
                    );
                    let lower =
                        Vec3::from_slice(&model.joint_limit_lower.host()?[start..start + 3]);
                    let upper =
                        Vec3::from_slice(&model.joint_limit_upper.host()?[start..start + 3]);
                    let nonlinear = prediction.is_some()
                        && !(0..3).any(|d| {
                            lower[d].is_finite()
                                && upper[d].is_finite()
                                && (upper[d] - lower[d]).abs() < 1e-12
                        });
                    let predicted_rate = if nonlinear {
                        Vec3::from_slice(&prediction.unwrap()[start..start + 3])
                    } else {
                        Vec3::ZERO
                    };
                    let half_delta = predicted_rate * (0.5 * dt);
                    let rotation = if nonlinear {
                        Quat::from_xyzw(half_delta.x, half_delta.y, half_delta.z, 1.).normalize()
                            * initial
                    } else {
                        initial
                    };
                    let stops = crate::joint_angles::stops(rotation, lower, upper)?;
                    let (frame, orientation) =
                        model.joint_motion_frame(input.body_q.host()?, q, joint)?;
                    let velocities = input.body_qd.host()?;
                    for stop in stops {
                        let error = if stop.bilateral {
                            let scale = 1.
                                + lower
                                    .to_array()
                                    .into_iter()
                                    .chain(upper.to_array())
                                    .filter(|v| v.is_finite())
                                    .map(|v| v.abs())
                                    .fold(0., Real::max);
                            equality_gap(stop.error, scale)
                        } else {
                            stop.error
                        };
                        let gradient = if nonlinear {
                            (stop.gradient - half_delta.cross(stop.gradient))
                                / (1. + half_delta.length_squared())
                        } else {
                            stop.gradient
                        };
                        let shift = gradient.dot(predicted_rate);
                        let direction = frame.rotation * gradient * orientation;
                        let mut ends = Vec::new();
                        for (body, side) in [
                            (model.joint_child.host()?[joint], 1.),
                            (model.joint_parent.host()?[joint], -1.),
                        ] {
                            if body >= 0 {
                                ends.push((
                                    body as usize,
                                    SpatialVector::new(Vec3::ZERO, direction * side),
                                ));
                            }
                        }
                        let rate = ends
                            .iter()
                            .map(|(body, wrench)| wrench.angular.dot(velocities[*body].angular))
                            .sum();
                        rows.push(Row {
                            joint: None,
                            cone: false,
                            rigid: false,
                            bound: if stop.bilateral {
                                DistanceBound::Equality
                            } else {
                                DistanceBound::Minimum
                            },
                            direction: Vec3::ZERO,
                            ends,
                            direct: None,
                            target: shift - error / dt,
                            rate,
                            feedback: None,
                            velocity_only: false,
                        });
                    }
                    continue;
                }
                if (hi - lo).abs() < 1e-12 {
                    continue;
                }
                let q = input.joint_q.host()?[coordinate + axis];
                for (sign, bound) in [(1.0, lo), (-1.0, hi)] {
                    if bound.is_finite() {
                        rows.push(Row {
                            joint: None,
                            cone: false,
                            rigid: false,
                            bound: DistanceBound::Minimum,
                            direction: Vec3::ZERO,
                            ends: Vec::new(),
                            direct: Some((dof, sign)),
                            target: sign * (bound - q) / dt,
                            rate: sign * input.joint_qd.host()?[dof],
                            feedback: None,
                            velocity_only: false,
                        });
                    }
                }
            }
            let first = offsets[joint] as usize;
            let angular_fixed = (kind == JointType::Ball as u32 || kind == JointType::D6 as u32)
                && (0..3).all(|axis| {
                    let d = first + angular + axis;
                    let lo = lower[d];
                    let hi = upper[d];
                    lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12
                });
            for dof in first..offsets[joint + 1] as usize {
                let lower = model.joint_limit_lower.host()?[dof];
                let upper = model.joint_limit_upper.host()?[dof];
                let locked =
                    lower.is_finite() && upper.is_finite() && (upper - lower).abs() < 1e-12;
                let cap = model.joint_velocity_limit.host()?[dof];
                let structural = locked
                    && ((kind <= 1)
                        || (kind == JointType::Ball as u32 && angular_fixed)
                        || (kind == JointType::D6 as u32 && (dof - first < 3 || angular_fixed)));
                if structural || !cap.is_finite() {
                    continue;
                }
                for &sign in if cap == 0. { &[1.][..] } else { &[1., -1.][..] } {
                    rows.push(Row {
                        joint: None,
                        cone: false,
                        rigid: false,
                        bound: if cap == 0. {
                            DistanceBound::Equality
                        } else {
                            DistanceBound::Minimum
                        },
                        direction: Vec3::ZERO,
                        ends: Vec::new(),
                        direct: Some((dof, sign)),
                        target: -cap,
                        rate: sign * input.joint_qd.host()?[dof],
                        feedback: None,
                        velocity_only: true,
                    });
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn apply_selected(
    scene: &RigidScene,
    solver: &FeatherstoneSolver,
    input: &State,
    output: &mut State,
    dt: Real,
    selected: Option<&[bool]>,
) -> Result<ConstraintReadings> {
    let (groups, mut readings) = prepare(scene, input, dt, selected)?;
    if groups.is_empty() {
        return Ok(readings);
    }
    let model = scene.model();
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let children = model.joint_child.host()?;
    let mut response = ArticulatedResponse::new(model, input);
    let mut changed = false;
    for ((arts, _), mut rows) in groups {
        if rows.is_empty() {
            continue;
        }
        scalar_rows(model, input, &arts, dt, &mut rows)?;
        let count = rows.len();
        let mut gram = vec![0.0; count * count];
        let mut projected_arts = Vec::new();
        let mut rates: Vec<_> = rows.iter().map(|r| r.rate).collect();
        for &art in &arts {
            let begin = starts[art] as usize;
            let end = starts[art + 1] as usize;
            let first = offsets[begin] as usize;
            let n = offsets[end] as usize - first;
            if n == 0 {
                continue;
            }
            let mut projected = vec![vec![0.0; n]; count];
            for (r, row) in rows.iter().enumerate() {
                for &(body, wrench) in &row.ends {
                    if !children[begin..end].contains(&(body as i32)) {
                        continue;
                    }
                    let j = response.jacobian(art, body)?;
                    for (c, p) in projected[r].iter_mut().enumerate() {
                        *p += j[c] * wrench.linear.x
                            + j[n + c] * wrench.linear.y
                            + j[2 * n + c] * wrench.linear.z
                            + j[3 * n + c] * wrench.angular.x
                            + j[4 * n + c] * wrench.angular.y
                            + j[5 * n + c] * wrench.angular.z;
                    }
                }
                if let Some((dof, sign)) = row.direct
                    && (first..first + n).contains(&dof)
                {
                    projected[r][dof - first] = sign;
                }
                for (c, p) in projected[r].iter().enumerate() {
                    rates[r] += p
                        * (output.joint_qd.host()?[first + c] - input.joint_qd.host()?[first + c]);
                }
            }
            response.add_gram(art, n, &projected, &mut gram)?;
            projected_arts.push((art, first, projected));
        }
        let free: Vec<_> = rows
            .iter()
            .zip(rates)
            .map(|(r, rate)| rate - r.target)
            .collect();
        let domains: Vec<_> = rows
            .iter()
            .map(|r| {
                use crate::coupled_constraints::Domain;
                if let Some(feedback) = r.feedback {
                    Domain::Interval {
                        lower: feedback.impulses[0],
                        upper: feedback.impulses[1],
                    }
                } else if r.bound == DistanceBound::Equality {
                    Domain::Bilateral
                } else {
                    Domain::Nonnegative
                }
            })
            .collect();
        for (i, row) in rows.iter().enumerate() {
            if let Some(feedback) = row.feedback {
                gram[i * count + i] += feedback.diagonal;
            }
        }
        let has_feedback = rows.iter().any(|r| r.feedback.is_some());
        let impulses = if has_feedback {
            crate::coupled_constraints::solve(&gram, &free, &domains)?.impulses
        } else {
            let rhs: Vec<_> = free.iter().map(|v| -v).collect();
            let bounds: Vec<_> = rows
                .iter()
                .map(|r| {
                    [
                        if r.bound == DistanceBound::Equality {
                            Real::NEG_INFINITY
                        } else {
                            0.0
                        },
                        Real::INFINITY,
                    ]
                })
                .collect();
            crate::loop_forces::bounded_solve(&gram, &rhs, &bounds)?
        };
        for (r, &impulse) in impulses.iter().enumerate() {
            if !has_feedback {
                let residual = free[r]
                    + (0..count)
                        .map(|c| gram[r * count + c] * impulses[c])
                        .sum::<Real>();
                let tolerance = 1e-8 * (1.0 + free[r].abs());
                if !residual.is_finite()
                    || residual < -tolerance
                    || ((rows[r].bound == DistanceBound::Equality || impulse > 0.0)
                        && residual.abs() > tolerance)
                {
                    return Err(Error::Solver(
                        "infeasible distance constraint response".into(),
                    ));
                }
            }
            if let Some(joint) = rows[r].joint {
                if rows[r].rigid {
                    let reading = readings.rigid.get_mut(&joint).unwrap();
                    let vector = rows[r].direction * (impulse / dt);
                    if rows[r].cone {
                        reading.angular += vector;
                    } else {
                        reading.linear += vector;
                    }
                    continue;
                }
                if rows[r].cone {
                    let reading = readings.cones.get_mut(&joint).unwrap();
                    reading.torque += rows[r].direction * (impulse / dt);
                    reading.active |= impulse != 0.0 || rows[r].bound == DistanceBound::Equality;
                    continue;
                }
                let reading = readings.distance.get_mut(&joint).unwrap();
                reading.force += rows[r].direction * (impulse / dt);
                if impulse != 0.0 || rows[r].bound == DistanceBound::Equality {
                    reading.active = rows[r].bound;
                }
            }
        }
        for (art, first, projected) in projected_arts {
            let n = projected[0].len();
            let load: Vec<_> = (0..n)
                .map(|c| (0..count).map(|r| projected[r][c] * impulses[r]).sum())
                .collect();
            let delta = response.solve(art, &load)?;
            for (c, delta) in delta.into_iter().enumerate() {
                output.joint_qd.host_mut()?[first + c] += delta;
                changed |= delta != 0.0;
            }
        }
    }
    if changed {
        solver.drift_cpu_selected(model, input, dt, output, selected)?;
    }
    refresh_readings(scene, output, &mut readings)?;
    Ok(readings)
}

pub(crate) fn refresh_readings(
    scene: &RigidScene,
    output: &State,
    readings: &mut ConstraintReadings,
) -> Result<()> {
    for (id, joint, limits) in point_constraints(scene) {
        if let Some(reading) = readings.distance.get_mut(&id) {
            let [a, b] = anchors(scene, output, joint)?;
            reading.separation = (b - a).length();
            reading.violation = limits
                .min
                .map_or(0.0, |min| (min - reading.separation).max(0.0))
                .max(
                    limits
                        .max
                        .map_or(0.0, |max| (reading.separation - max).max(0.0)),
                );
        }
    }
    for (id, joint, limits) in scene.cone_joints() {
        if let Some(reading) = readings.cones.get_mut(&id) {
            let (_, geometry) = cone_constraints::geometry(scene, output, joint, limits)?;
            reading.swing = geometry.swing;
            reading.violation = geometry.violation;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::equality_gap;
    use molla_math::Real;

    #[test]
    fn equality_gap_resolves_arithmetic_noise_without_erasing_motion() {
        for scale in [1e-3, 1., 1e3] {
            let roundoff = 64. * Real::EPSILON * scale;
            for sign in [-1., 1.] {
                assert_eq!(equality_gap(sign * roundoff, scale), 0.);
                let motion = sign * 2. * roundoff;
                assert_eq!(equality_gap(motion, scale), motion);
            }
        }
    }
}
