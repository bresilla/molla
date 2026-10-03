//! CPU contact episodes and instantaneous impact correction.

use crate::{
    distance_constraints::{self, ConstraintReadings, DistanceBound},
    penalty_contact::{ContactForce, FrictionRegime},
    rigid_impulse::{ImpactConstraint, ImpactContact, ImpactRow, apply_impact},
};
use molla_core::{BodyId, Error, Result, ShapeId};
use molla_math::{Real, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyHandle, ColliderHandle, RigidScene, SceneStamp};
use molla_sim::{BodyFlags, Contacts, ShapeFlags, State, eval_body_velocities};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Default)]
pub(crate) struct PublishedImpactState(BTreeMap<BodyHandle, (Transform, SpatialVector)>);

impl PublishedImpactState {
    pub(crate) fn record(&mut self, scene: &RigidScene) {
        self.0 = scene
            .bodies()
            .filter_map(|(body, _)| {
                Some((
                    body,
                    (scene.body_pose(body).ok()?, scene.body_velocity(body).ok()?),
                ))
            })
            .collect();
    }

    pub(crate) fn matches(&self, scene: &RigidScene, state: &State, body: BodyHandle) -> bool {
        let Some(&(pose, velocity)) = self.0.get(&body) else {
            return false;
        };
        let Some(index) = scene.body_index(body).map(|b| b.index()) else {
            return false;
        };
        let (Ok(poses), Ok(velocities)) = (state.body_q.host(), state.body_qd.host()) else {
            return false;
        };
        let (Some(actual_pose), Some(actual_velocity)) = (poses.get(index), velocities.get(index))
        else {
            return false;
        };
        let close = |a: Vec3, b: Vec3| {
            a.is_finite()
                && b.is_finite()
                && (a - b).abs().max_element()
                    <= 64.
                        * Real::EPSILON
                        * a.abs().max_element().max(b.abs().max_element()).max(1.)
        };
        *actual_pose == pose
            && close(actual_velocity.linear, velocity.linear)
            && close(actual_velocity.angular, velocity.angular)
    }
}

#[derive(Clone, PartialEq)]
struct Identity {
    poses: [Transform; 2],
    stamps: [SceneStamp; 2],
    bodies: [Option<(SceneStamp, SceneStamp)>; 2],
}

#[derive(Clone)]
struct Episode {
    identity: Identity,
    normal: Vec3,
}

#[derive(Clone)]
struct Sample {
    row: usize,
    contact: ImpactContact,
    depth: Real,
    new: bool,
    canonical_normal: Vec3,
}

#[derive(Clone, Default)]
pub(crate) struct ImpactCache {
    episodes: BTreeMap<[ColliderHandle; 2], Episode>,
    samples: Vec<Sample>,
    active: BTreeSet<usize>,
    selection: Vec<bool>,
    published: PublishedImpactState,
}

fn normal_speed(scene: &RigidScene, state: &State, contact: &ImpactContact) -> Result<Real> {
    let mut speed = 0.;
    for side in 0..2 {
        if let Some(body) = contact.bodies[side] {
            let i = body.index();
            if scene.model().body_flags.host()?[i] & BodyFlags::FIXED != 0 {
                continue;
            }
            let pose = state.body_q.host()?[i];
            let arm =
                contact.point - (pose.position + pose.rotation * scene.model().body_com.host()?[i]);
            let velocity = state.body_qd.host()?[i];
            speed += (if side == 0 { -1. } else { 1. })
                * contact
                    .normal
                    .dot(velocity.linear + velocity.angular.cross(arm));
        }
    }
    Ok(speed)
}

impl ImpactCache {
    pub(crate) fn record_published(&mut self, scene: &RigidScene) {
        self.published.record(scene);
    }

    pub(crate) fn preserve_after_detachment(&mut self, scene: &RigidScene) {
        for (key, episode) in &mut self.episodes {
            for side in 0..2 {
                if let Some(collider) = scene.collider(key[side])
                    && scene.collider_stamp(key[side]).as_ref()
                        == Some(&episode.identity.stamps[side])
                    && collider.pose == episode.identity.poses[side]
                    && let Some(body) = collider.parent
                    && let Some((structure, state)) = &mut episode.identity.bodies[side]
                    && scene.body_state_stamp(body).as_ref() == Some(state)
                {
                    *structure = scene.body_contact_episode_stamp(body).unwrap();
                }
            }
        }
    }

    pub(crate) fn selection(&self) -> &[bool] {
        &self.selection
    }
    pub(crate) fn position_rows(
        &self,
        scene: &RigidScene,
        state: &State,
        selected: Option<&[bool]>,
    ) -> Result<Vec<ImpactConstraint>> {
        let mut rows = Vec::new();
        for &id in &self.active {
            let sample = &self.samples[id];
            if !sample.new || sample.depth <= 0. {
                continue;
            }
            let mut row = ImpactRow {
                velocity_bias: -sample.depth,
                ..Default::default()
            };
            for side in 0..2 {
                if let Some(body) = sample.contact.bodies[side] {
                    let pose = state.body_q.host()?[body.index()];
                    let arm = sample.contact.point
                        - (pose.position
                            + pose.rotation * scene.model().body_com.host()?[body.index()]);
                    let normal = sample.contact.normal * if side == 0 { -1. } else { 1. };
                    row.ends
                        .push((body, SpatialVector::new(normal, arm.cross(normal))));
                }
            }
            rows.push(ImpactConstraint::Nonnegative(row));
        }
        let (groups, _) = distance_constraints::prepare(scene, state, 1., Some(&self.selection))?;
        let mut scalars: Vec<_> = groups.into_values().flatten().collect();
        let arts: Vec<_> = (0..scene.model().articulation_count)
            .filter(|&a| self.selection[a] && selected.is_none_or(|s| s[a]))
            .collect();
        distance_constraints::scalar_rows(scene.model(), state, &arts, 1., &mut scalars)?;
        for scalar in scalars {
            if scalar.feedback.is_some() || scalar.velocity_only {
                continue;
            }
            let row = ImpactRow {
                ends: scalar
                    .ends
                    .into_iter()
                    .map(|(b, w)| (BodyId::from_index(b), w))
                    .collect(),
                direct: scalar.direct,
                velocity_bias: -scalar.target,
            };
            rows.push(if scalar.bound == DistanceBound::Equality {
                ImpactConstraint::Bilateral(row)
            } else {
                ImpactConstraint::Nonnegative(row)
            });
        }
        Ok(rows)
    }

    pub(crate) fn prepare(
        &mut self,
        scene: &RigidScene,
        state: &State,
        contacts: &mut Contacts,
    ) -> Result<()> {
        self.samples.clear();
        self.active.clear();
        self.selection.clear();
        let model = scene.model();
        let count = contacts.count as usize;
        if count > contacts.capacity
            || [
                contacts.depth.len(),
                contacts.a_shape.len(),
                contacts.b_shape.len(),
                contacts.a_index.len(),
                contacts.b_index.len(),
                contacts.point.len(),
                contacts.normal.len(),
                contacts.friction.len(),
                contacts.static_friction.len(),
                contacts.restitution.len(),
            ]
            .iter()
            .any(|&len| len < count)
        {
            return Err(Error::Solver("invalid impact contact arrays".into()));
        }
        let shape_flags = model.shape_flags.host()?;
        let body_flags = model.body_flags.host()?;
        let mut groups = BTreeMap::<[ColliderHandle; 2], Vec<Sample>>::new();
        for i in 0..contacts.count as usize {
            let depth = contacts.depth.host()?[i];
            let shapes = [contacts.a_shape.host()?[i], contacts.b_shape.host()?[i]];
            if shapes
                .iter()
                .any(|&s| s != u32::MAX && s as usize >= model.shape_count)
            {
                return Err(Error::Solver("invalid impact collider index".into()));
            }
            if depth < 0.
                || shapes.iter().any(|&s| {
                    s == u32::MAX
                        || shape_flags[s as usize] & ShapeFlags::SENSOR != 0
                        || shape_flags[s as usize] & ShapeFlags::COLLIDES == 0
                })
            {
                continue;
            }
            let handles = shapes.map(|s| {
                scene
                    .collider_handle(ShapeId::from_index(s as usize))
                    .unwrap()
            });
            let mut key = handles;
            key.sort();
            let bodies = [contacts.a_index.host()?[i], contacts.b_index.host()?[i]]
                .map(|b| (b != u32::MAX).then(|| BodyId::from_index(b as usize)));
            if bodies
                .iter()
                .flatten()
                .any(|b| b.index() >= model.body_count)
            {
                return Err(Error::Solver("invalid impact body index".into()));
            }
            if !depth.is_finite()
                || !contacts.point.host()?[i].is_finite()
                || !contacts.normal.host()?[i].is_finite()
            {
                return Err(Error::Solver("nonfinite impact geometry".into()));
            }
            let responsive = bodies
                .iter()
                .flatten()
                .any(|b| body_flags[b.index()] & BodyFlags::IMMOVABLE == 0);
            if !responsive {
                continue;
            }
            let surface_velocity = std::array::from_fn(|side| {
                let v = model.shape_surface_velocity[shapes[side] as usize];
                bodies[side].map_or(v, |b| state.body_q.host().unwrap()[b.index()].rotation * v)
            });
            let dynamic_friction = contacts.friction.host()?[i];
            let coefficient = contacts.static_friction.host()?[i];
            let static_friction = if coefficient.is_nan() {
                dynamic_friction
            } else {
                coefficient
            };
            groups.entry(key).or_default().push(Sample {
                row: i,
                depth,
                new: false,
                canonical_normal: contacts.normal.host()?[i]
                    * if handles[0] == key[0] { 1. } else { -1. },
                contact: ImpactContact {
                    bodies,
                    point: contacts.point.host()?[i],
                    normal: contacts.normal.host()?[i],
                    static_friction,
                    dynamic_friction,
                    restitution: contacts.restitution.host()?[i],
                    surface_velocity,
                },
            });
        }
        let present: BTreeSet<_> = groups.keys().copied().collect();
        self.episodes.retain(|key, _| present.contains(key));
        for (key, mut samples) in groups {
            let colliders = key.map(|id| scene.collider(id).unwrap());
            let identity = Identity {
                stamps: key.map(|id| scene.collider_stamp(id).unwrap()),
                bodies: std::array::from_fn(|side| {
                    colliders[side].parent.map(|body| {
                        (
                            scene.body_contact_episode_stamp(body).unwrap(),
                            scene.body_state_stamp(body).unwrap(),
                        )
                    })
                }),
                poses: colliders.map(|c| c.pose),
            };
            if let Some(previous) = self.episodes.get_mut(&key) {
                for side in 0..2 {
                    if let Some(body) = colliders[side].parent
                        && let Some((structure, edit)) = &mut previous.identity.bodies[side]
                        && Some(&*structure) == scene.body_contact_episode_stamp(body).as_ref()
                        && self.published.matches(scene, state, body)
                    {
                        *edit = scene.body_state_stamp(body).unwrap();
                    }
                }
            }
            let normal = samples[0].canonical_normal;
            let old = self
                .episodes
                .get(&key)
                .filter(|e| e.identity == identity && e.normal.dot(normal) > 0.95);
            let onset = old.is_none()
                && samples
                    .iter()
                    .map(|s| normal_speed(scene, state, &s.contact))
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .any(|&v| v < -1e-4);
            let offset = if onset {
                samples.iter().map(|s| s.depth).fold(0., Real::max)
            } else {
                0.
            };
            self.episodes.insert(key, Episode { identity, normal });
            for sample in &mut samples {
                sample.new = onset;
                contacts.depth.host_mut()?[sample.row] = (sample.depth - offset).max(0.);
            }
            self.samples.extend(samples);
        }
        Ok(())
    }

    pub(crate) fn resting_selection(
        &mut self,
        scene: &RigidScene,
        selected: Option<&[bool]>,
    ) -> Result<Vec<bool>> {
        self.selection = vec![false; scene.model().articulation_count];
        self.active.clear();
        if self.samples.iter().any(|s| s.new) {
            let graph = crate::rigid_islands::RigidIslands::build(
                scene,
                self.samples.iter().map(|s| {
                    s.contact
                        .bodies
                        .map(|b| b.and_then(|b| scene.body_handle(b)))
                }),
            )?;
            for sample in self.samples.iter().filter(|s| s.new) {
                for body in sample.contact.bodies.iter().flatten() {
                    if let Some(group) = scene.body_handle(*body).and_then(|b| graph.body_island(b))
                    {
                        for &art in &group.articulations {
                            self.selection[art] = selected.is_none_or(|s| s[art]);
                        }
                    }
                }
            }
            for (id, sample) in self.samples.iter().enumerate() {
                if sample.contact.bodies.iter().flatten().any(|body| {
                    scene
                        .body_handle(*body)
                        .and_then(|b| graph.body_island(b))
                        .is_some_and(|group| group.articulations.iter().any(|&a| self.selection[a]))
                }) {
                    self.active.insert(id);
                }
            }
        }
        Ok(self
            .selection
            .iter()
            .enumerate()
            .map(|(art, impact)| !impact && selected.is_none_or(|s| s[art]))
            .collect())
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        input: &State,
        output: &mut State,
        dt: Real,
        forces: &mut [ContactForce],
        readings: &mut ConstraintReadings,
        selected: Option<&[bool]>,
    ) -> Result<bool> {
        for sample in &self.samples {
            if let Some(force) = forces.get_mut(sample.row) {
                force.penetration = sample.depth;
            }
        }
        if !self.samples.iter().any(|s| s.new) {
            return Ok(false);
        }
        let model = scene.model();
        self.resting_selection(scene, selected)?;
        let mut state = input.clone_host()?;
        state
            .joint_qd
            .host_mut()?
            .copy_from_slice(output.joint_qd.host()?);
        for body in 0..model.body_count {
            if model.body_flags.host()?[body] & BodyFlags::IMMOVABLE != 0 {
                state.body_qd.host_mut()?[body] = output.body_qd.host()?[body];
            }
        }
        eval_body_velocities(model, &mut state)?;
        let mut constraints = Vec::new();
        let mut ids = Vec::new();
        for (id, sample) in self.samples.iter().enumerate() {
            if !self.active.contains(&id) {
                continue;
            }
            if normal_speed(scene, &state, &sample.contact)? > 0. {
                continue;
            }
            let mut contact = sample.contact;
            if !sample.new {
                contact.restitution = 0.;
            }
            constraints.push(ImpactConstraint::Contact(contact));
            ids.push(id);
        }
        let (groups, initial) =
            distance_constraints::prepare(scene, input, dt, Some(&self.selection))?;
        readings.distance.extend(initial.distance);
        readings.cones.extend(initial.cones);
        readings.rigid.extend(initial.rigid);
        let mut scalars: Vec<_> = groups.into_values().flatten().collect();
        let arts: Vec<_> = (0..model.articulation_count)
            .filter(|&a| self.selection[a])
            .collect();
        distance_constraints::scalar_rows(model, input, &arts, dt, &mut scalars)?;
        for scalar in &scalars {
            let row = ImpactRow {
                ends: scalar
                    .ends
                    .iter()
                    .map(|&(b, w)| (BodyId::from_index(b), w))
                    .collect(),
                direct: scalar.direct,
                velocity_bias: -scalar.target,
            };
            constraints.push(if let Some(feedback) = scalar.feedback {
                ImpactConstraint::Bounded {
                    row,
                    impulses: feedback.impulses,
                    response_diagonal: feedback.diagonal,
                }
            } else if scalar.bound == DistanceBound::Equality {
                ImpactConstraint::Bilateral(row)
            } else {
                ImpactConstraint::Nonnegative(row)
            });
        }
        let result = apply_impact(model, &mut state, &constraints)
            .map_err(|e| Error::Solver(format!("runtime impact velocity response: {e}")))?;
        for (block, &id) in ids.iter().enumerate() {
            let sample = &self.samples[id];
            let impulse = result.contact_impulses[block].unwrap();
            let force = forces
                .get_mut(sample.row)
                .ok_or_else(|| Error::Solver("missing impact force reading".into()))?;
            let normal = impulse.dot(sample.contact.normal);
            force.normal_force += normal / dt;
            force.normal_impulse += normal;
            force.tangent_impulse += impulse - sample.contact.normal * normal;
            force.force_on_b += impulse / dt;
            force.acted |= impulse != Vec3::ZERO;
            force.applied_friction = if result.solution.dynamic[block] {
                force.friction
            } else {
                force.static_friction
            };
            let mut relative = Vec3::ZERO;
            for side in 0..2 {
                let sign = if side == 0 { -1. } else { 1. };
                relative += sample.contact.surface_velocity[side] * sign;
                if let Some(body) = sample.contact.bodies[side] {
                    let i = body.index();
                    if model.body_flags.host()?[i] & BodyFlags::FIXED == 0 {
                        let pose = state.body_q.host()?[i];
                        let arm = sample.contact.point
                            - (pose.position + pose.rotation * model.body_com.host()?[i]);
                        let v = state.body_qd.host()?[i];
                        relative += (v.linear + v.angular.cross(arm)) * sign;
                    }
                }
            }
            let slip = relative - sample.contact.normal * relative.dot(sample.contact.normal);
            force.friction_regime = if force.static_friction == 0. {
                FrictionRegime::Inactive
            } else if slip.length() <= 1e-4 {
                FrictionRegime::Sticking
            } else {
                FrictionRegime::Sliding
            };
        }
        for (index, scalar) in scalars.iter().enumerate() {
            let impulse = result.solution.impulses[3 * ids.len() + index];
            if let Some(joint) = scalar.joint {
                if scalar.rigid {
                    let reading = readings
                        .rigid
                        .get_mut(&joint)
                        .ok_or_else(|| Error::Solver("missing impact rigid loop reading".into()))?;
                    let vector = scalar.direction * (impulse / dt);
                    if scalar.cone {
                        reading.angular += vector;
                    } else {
                        reading.linear += vector;
                    }
                    continue;
                }
                if scalar.cone {
                    let r = readings
                        .cones
                        .get_mut(&joint)
                        .ok_or_else(|| Error::Solver("missing impact cone reading".into()))?;
                    r.torque += scalar.direction * (impulse / dt);
                    r.active |= impulse != 0.;
                } else {
                    let r = readings
                        .distance
                        .get_mut(&joint)
                        .ok_or_else(|| Error::Solver("missing impact distance reading".into()))?;
                    r.force += scalar.direction * (impulse / dt);
                    if impulse != 0. {
                        r.active = scalar.bound;
                    }
                }
            }
        }
        output
            .joint_qd
            .host_mut()?
            .copy_from_slice(state.joint_qd.host()?);
        Ok(true)
    }
}
