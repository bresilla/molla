use std::collections::{BTreeMap, BTreeSet};

use molla_core::{Error, Result};
use molla_math::{Mat3, Real, SpatialVector};
use molla_sim::runtime::{BodyActivationStamp, BodyHandle, BodyKind, ColliderHandle, RigidScene};
use molla_sim::{Contacts, ShapeFlags, State};

use super::RigidWorld;
use crate::rigid_islands::RigidIslands;
use crate::wheel_forces::{WheelForces, WheelSample};

#[derive(Clone)]
struct SleepingGroup {
    bodies: Vec<BodyHandle>,
    watched: BTreeMap<BodyHandle, BodyActivationStamp>,
    wheels: (std::sync::Arc<()>, u64),
    wheel_samples: Vec<WheelSample>,
    contacts: BTreeSet<[ColliderHandle; 2]>,
}

impl SleepingGroup {
    fn valid(&self, scene: &RigidScene, wheels: &WheelForces) -> bool {
        wheels.matches_activation(&self.wheels)
            && self
                .watched
                .iter()
                .all(|(&body, stamp)| scene.activation_stamp(body).as_ref() == Some(stamp))
            && !self.bodies.iter().any(|&body| wheels.transitioning(body))
    }
}

#[derive(Clone, Default)]
pub(super) struct SleepingGroups {
    groups: BTreeMap<BodyHandle, SleepingGroup>,
    owners: BTreeMap<BodyHandle, BodyHandle>,
}

impl SleepingGroups {
    pub(super) fn clear(&mut self) {
        self.groups.clear();
        self.owners.clear();
    }

    pub(super) fn any_valid(&self, scene: &RigidScene, wheels: &WheelForces) -> bool {
        self.groups.values().any(|group| group.valid(scene, wheels))
    }

    fn contains(&self, body: BodyHandle) -> bool {
        self.owners.contains_key(&body)
    }

    fn refresh(&mut self, scene: &RigidScene, wheels: &WheelForces) {
        self.groups.retain(|_, group| group.valid(scene, wheels));
        self.owners
            .retain(|_, group| self.groups.contains_key(group));
    }

    fn acknowledge(&mut self, scene: &RigidScene) {
        for group in self.groups.values_mut() {
            for (body, stamp) in &mut group.watched {
                *stamp = scene
                    .activation_stamp(*body)
                    .expect("retained sleeping body");
            }
        }
    }

    fn wake(&mut self, body: BodyHandle) {
        if let Some(owner) = self.owners.get(&body).copied()
            && let Some(group) = self.groups.remove(&owner)
        {
            for body in group.bodies {
                self.owners.remove(&body);
            }
        }
    }
}

impl RigidWorld {
    /// Explicit whole-island sleep status, invalidated immediately by watched input edits.
    pub fn is_sleeping(&self, body: BodyHandle) -> bool {
        self.sleeping
            .owners
            .get(&body)
            .and_then(|owner| self.sleeping.groups.get(owner))
            .is_some_and(|group| group.valid(&self.scene, &self.wheels))
    }

    /// Zero generalized velocities and sleep the body's complete dynamic dependency group.
    /// Moving kinematic boundaries and transitioning tyre pressures keep the group awake.
    pub fn sleep_body(&mut self, body: BodyHandle) -> Result<()> {
        self.sleep_bodies(&[body])
    }

    /// Atomically sleep the complete groups containing the requested dynamic bodies.
    pub fn sleep_bodies(&mut self, bodies: &[BodyHandle]) -> Result<()> {
        if bodies.iter().any(|body| self.scene.body(*body).is_none()) {
            return Err(invalid("unknown sleep body"));
        }
        let islands = self.islands()?;
        let mut sleeping = self.sleeping.clone();
        sleeping.refresh(&self.scene, &self.wheels);
        let mut groups = BTreeMap::new();
        for &body in bodies {
            let desc = self.scene.body(body).unwrap();
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let group = islands
                .body_island(body)
                .ok_or_else(|| invalid("missing sleep island"))?;
            for &body in &group.bodies {
                sleeping.wake(body);
            }
            if group
                .bodies
                .iter()
                .any(|&body| self.wheels.transitioning(body))
                || group.boundaries.iter().any(|&body| {
                    self.scene
                        .body_velocity(body)
                        .is_ok_and(|velocity| velocity != SpatialVector::ZERO)
                })
            {
                continue;
            }
            groups.insert(group.bodies[0], group);
        }
        if groups.is_empty() {
            self.sleeping = sleeping;
            return Ok(());
        }
        let mut next = self.scene.state().clone_host()?;
        let model = self.scene.model();
        let starts = model.articulation_start.host()?;
        let dofs = model.joint_dof_offset.host()?;
        for group in groups.values() {
            for &art in &group.articulations {
                let begin = dofs[starts[art] as usize] as usize;
                let end = dofs[starts[art + 1] as usize] as usize;
                next.joint_qd.host_mut()?[begin..end].fill(0.0);
            }
            for &body in &group.bodies {
                next.body_qd.host_mut()?[self.scene.body_index(body).unwrap().index()] =
                    SpatialVector::ZERO;
            }
        }
        molla_sim::eval_body_velocities(model, &mut next)?;
        self.scene.commit_step(next)?;
        sleeping.acknowledge(&self.scene);
        let pairs = self.contact_pairs();
        for group in groups.values() {
            let owner = group.bodies[0];
            let watched = group
                .bodies
                .iter()
                .chain(&group.boundaries)
                .map(|&body| (body, self.scene.activation_stamp(body).unwrap()))
                .collect();
            let wheel_samples = self
                .wheels
                .samples(&self.scene)
                .iter()
                .filter(|sample| group.bodies.contains(&sample.wheel))
                .cloned()
                .collect();
            let contacts = pairs
                .iter()
                .filter(|pair| {
                    pair.bodies
                        .iter()
                        .flatten()
                        .any(|body| group.bodies.contains(body))
                        && pair.colliders.iter().all(|&collider| {
                            self.scene
                                .collider(collider)
                                .is_some_and(|c| c.enabled && !c.sensor)
                        })
                        && pair.points.iter().any(|point| point.penetration >= 0.0)
                })
                .map(|pair| canonical_pair(pair.colliders))
                .collect();
            sleeping.groups.insert(
                owner,
                SleepingGroup {
                    bodies: group.bodies.clone(),
                    watched,
                    wheels: self.wheels.activation_stamp(),
                    wheel_samples,
                    contacts,
                },
            );
            for &body in &group.bodies {
                sleeping.owners.insert(body, owner);
            }
        }
        self.sleeping = sleeping;
        Ok(())
    }

    /// Wake dynamic dependents of a dynamic body or an immovable boundary.
    pub fn wake_body(&mut self, body: BodyHandle) -> Result<()> {
        let graph = self.islands()?;
        let affected = graph.affected_bodies(&[body])?;
        self.settling.reset_bodies(&affected);
        self.settling.reset_bodies(&[body]);
        let watched: Vec<_> = self
            .sleeping
            .groups
            .values()
            .filter(|group| group.watched.contains_key(&body))
            .map(|group| group.bodies[0])
            .collect();
        for body in affected.into_iter().chain(watched) {
            self.sleeping.wake(body);
        }
        Ok(())
    }

    pub(super) fn step_sleeping(
        &mut self,
        dt: Real,
        mut excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
    ) -> Result<()> {
        self.contact_revision = None;
        self.wheels.invalidate();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid("step dt must be positive and finite"));
        }
        self.quarantined = self.scene.quarantine_non_finite()?;
        let scene = &self.scene;
        let mut sleeping = self.sleeping.clone();
        sleeping.refresh(scene, &self.wheels);
        let held: Vec<_> = sleeping
            .groups
            .values()
            .flat_map(|group| group.wheel_samples.iter().cloned())
            .collect();
        let mut wheel_step = self.wheels.begin_sleeping_step(&held);
        self.pipeline
            .set_extra_shape_exclusions(self.wheels.prepare(scene)?);
        let handles = dense_handles(scene)?;
        let mut connections = Vec::new();
        for (body, desc) in scene.bodies() {
            if desc.enabled
                && desc.kind == BodyKind::Dynamic
                && let Some(wheel) = self.wheels.descriptor(body)
                && scene
                    .body(wheel.heading_body)
                    .is_some_and(|body| body.enabled)
                && scene.joint_index(wheel.spin_joint).is_some()
            {
                connections.push([Some(body), Some(wheel.heading_body)]);
            }
        }
        let excluded = std::cell::RefCell::new(&mut excluded);
        let mut graph_cache: Option<(Vec<[Option<BodyHandle>; 2]>, RigidIslands)> = None;
        let wheels = &mut self.wheels;
        let mut next = scene.model().state()?;
        let bounds = self.bounds.prepare(scene);
        self.solver.step_with_active_bounds(
            scene.model(),
            scene.state(),
            scene.control(),
            &mut self.pipeline,
            &mut self.contacts,
            dt,
            &mut next,
            |a, b| {
                excluded.borrow_mut()(
                    a.map(|id| handles[id.index()]),
                    b.map(|id| handles[id.index()]),
                )
            },
            |state, contacts, sub_dt, wrenches| {
                let mut edges = connections.clone();
                let present = contact_edges(scene, contacts, &handles, &mut edges, &mut sleeping)?;
                let changed: Vec<_> = sleeping
                    .groups
                    .values()
                    .filter(|group| {
                        !group.contacts.is_subset(&present)
                            || group.wheel_samples.iter().any(|sample| {
                                excluded.borrow_mut()(
                                    Some(sample.wheel),
                                    scene
                                        .collider(sample.ground)
                                        .and_then(|ground| ground.parent),
                                )
                            })
                    })
                    .map(|group| group.bodies[0])
                    .collect();
                for body in changed {
                    sleeping.wake(body);
                }
                edges.sort_unstable();
                edges.dedup();
                if graph_cache
                    .as_ref()
                    .is_none_or(|(previous, _)| previous != &edges)
                {
                    let graph = RigidIslands::build(scene, edges.iter().copied())?;
                    graph_cache = Some((edges, graph));
                }
                let graph = &graph_cache.as_ref().unwrap().1;
                let mut selected = vec![true; scene.model().body_count];
                let mut articulations = vec![true; scene.model().articulation_count];
                let velocities = state.body_qd.host()?;
                loop {
                    let before = sleeping.owners.len();
                    for group in graph.groups() {
                        if group.bodies.iter().any(|body| !sleeping.contains(*body))
                            || group.boundaries.iter().any(|body| {
                                let i = scene.body_index(*body).unwrap().index();
                                velocities[i] != SpatialVector::ZERO
                            })
                        {
                            for &body in &group.bodies {
                                sleeping.wake(body);
                            }
                        }
                    }
                    if sleeping.owners.len() == before {
                        break;
                    }
                }
                let mut any_awake = false;
                for group in graph.groups() {
                    let awake = group.bodies.iter().any(|body| !sleeping.contains(*body));
                    any_awake |= awake;
                    for &body in &group.bodies {
                        selected[scene.body_index(body).unwrap().index()] = awake;
                    }
                    for &art in &group.articulations {
                        articulations[art] = awake;
                    }
                }
                damping(scene, state, sub_dt, wrenches, &selected)?;
                let mut response =
                    crate::articulated_response::ArticulatedResponse::new(scene.model(), state);
                wheels.apply_selected(
                    scene,
                    &mut response,
                    sub_dt,
                    wrenches,
                    &mut wheel_step,
                    |a, b| excluded.borrow_mut()(a, b),
                    Some(&selected),
                )?;
                if any_awake {
                    crate::loop_forces::apply_selected(
                        scene,
                        &mut response,
                        sub_dt,
                        wrenches,
                        Some(&selected),
                    )?;
                }
                Ok(articulations)
            },
            Some(&bounds),
        )?;
        self.scene.commit_step(next)?;
        sleeping.acknowledge(&self.scene);
        self.sleeping = sleeping;
        self.wheels.publish(&self.scene, wheel_step);
        self.contact_revision = Some(self.scene.stamp());
        Ok(())
    }
}

fn dense_handles(scene: &RigidScene) -> Result<Vec<BodyHandle>> {
    let mut indexed: Vec<_> = scene
        .bodies()
        .map(|(body, _)| {
            scene
                .body_index(body)
                .map(|index| (index.index(), body))
                .ok_or_else(|| invalid("missing activation body"))
        })
        .collect::<Result<_>>()?;
    indexed.sort_unstable_by_key(|entry| entry.0);
    Ok(indexed.into_iter().map(|(_, body)| body).collect())
}

fn contact_edges(
    scene: &RigidScene,
    contacts: &Contacts,
    handles: &[BodyHandle],
    edges: &mut Vec<[Option<BodyHandle>; 2]>,
    sleeping: &mut SleepingGroups,
) -> Result<BTreeSet<[ColliderHandle; 2]>> {
    let a = contacts.a_index.host()?;
    let b = contacts.b_index.host()?;
    let sa = contacts.a_shape.host()?;
    let sb = contacts.b_shape.host()?;
    let depth = contacts.depth.host()?;
    let flags = scene.model().shape_flags.host()?;
    let mut present = BTreeSet::new();
    for i in 0..contacts.count as usize {
        if depth[i] < 0.0
            || [sa[i], sb[i]].into_iter().any(|shape| {
                shape != u32::MAX
                    && (flags[shape as usize] & ShapeFlags::SENSOR != 0
                        || flags[shape as usize] & ShapeFlags::COLLIDES == 0)
            })
        {
            continue;
        }
        let bodies = [a[i], b[i]].map(|body| (body != u32::MAX).then(|| handles[body as usize]));
        if let [Some(a), Some(b)] =
            [sa[i], sb[i]].map(|shape| scene.collider_handle(molla_core::ShapeId(shape)))
        {
            let pair = canonical_pair([a, b]);
            present.insert(pair);
            for body in bodies.into_iter().flatten() {
                if sleeping
                    .owners
                    .get(&body)
                    .and_then(|owner| sleeping.groups.get(owner))
                    .is_some_and(|group| !group.contacts.contains(&pair))
                {
                    sleeping.wake(body);
                }
            }
        }
        edges.push(bodies);
    }
    Ok(present)
}

fn canonical_pair(mut pair: [ColliderHandle; 2]) -> [ColliderHandle; 2] {
    pair.sort_unstable();
    pair
}

fn damping(
    scene: &RigidScene,
    state: &State,
    dt: Real,
    forces: &mut [SpatialVector],
    selected: &[bool],
) -> Result<()> {
    let poses = state.body_q.host()?;
    let velocities = state.body_qd.host()?;
    let masses = scene.model().body_mass.host()?;
    let inertias = scene.model().body_inertia.host()?;
    for (body, desc) in scene.bodies() {
        let i = scene.body_index(body).unwrap().index();
        if !selected[i] || !desc.enabled || desc.kind != BodyKind::Dynamic {
            continue;
        }
        let coefficient = |d: Real| if d > 0.0 { 1.0 / (1.0 / d + dt) } else { 0.0 };
        let rotation = Mat3::from_quat(poses[i].rotation);
        forces[i].linear -= velocities[i].linear * (masses[i] * coefficient(desc.linear_damping));
        forces[i].angular -= (rotation * inertias[i] * rotation.transpose())
            * velocities[i].angular
            * coefficient(desc.angular_damping);
    }
    Ok(())
}

fn invalid(message: &str) -> Error {
    Error::Solver(message.into())
}
