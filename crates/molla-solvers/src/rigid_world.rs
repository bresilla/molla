//! Rigid-world stepping and stable-handle contact readout.

use std::collections::BTreeMap;
use std::sync::Mutex;

use molla_core::{Error, Result};
use molla_geometry::raycast_cpu::CpuRaycaster;
use molla_geometry::{
    Aabb, CollisionPipeline,
    collider_query::{self, ColliderShapeView},
};
use molla_math::{Mat3, Real, SpatialVector, Transform, Vec3};
use molla_sim::Contacts;
use molla_sim::runtime::{BodyHandle, ColliderHandle, RigidScene, SceneStamp};

use crate::FeatherstoneSolver;
pub use crate::cone_constraints::{BallAnchorReading, ConeReading};
use crate::distance_constraints::ConstraintReadings;
pub use crate::distance_constraints::{DistanceBound, DistanceReading};
use crate::joint_reactions::ReactionFrame;
pub use crate::joint_reactions::{JointReaction, JointReactionSample};
use crate::penalty_contact::ContactForce;

#[path = "rigid_sleep.rs"]
mod sleep;

#[path = "rigid_settling.rs"]
mod settling;
pub use settling::SleepSettings;

#[path = "rigid_breakage.rs"]
mod breakage;
pub use breakage::JointBreakEvent;

#[cfg(feature = "gpu")]
#[path = "rigid_world_gpu.rs"]
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::{GpuContactCapacity, GpuRuntimeDiagnostics};

#[cfg(test)]
#[path = "rigid_bounds_tests.rs"]
mod bounds_tests;

#[derive(Clone, Debug)]
pub struct ContactPair {
    pub colliders: [ColliderHandle; 2],
    pub bodies: [Option<BodyHandle>; 2],
    /// First point's normal from collider 0 to collider 1; each point retains its own normal.
    pub normal: Vec3,
    pub points: Vec<ContactForce>,
}

pub struct RigidWorld {
    pub scene: RigidScene,
    pub solver: FeatherstoneSolver,
    pub wheels: crate::wheel_forces::WheelForces,
    pub tracks: crate::track_forces::TrackForces,
    /// Motors, brakes, propellers, belts and connectors.
    pub devices: crate::devices::Devices,
    pipeline: CollisionPipeline,
    bounds: molla_geometry::shape_world::SceneShapeBounds,
    contacts: Contacts,
    impacts: crate::rigid_impact::ImpactCache,
    contact_revision: Option<SceneStamp>,
    queries: Mutex<CpuRaycaster>,
    quarantined: Vec<BodyHandle>,
    sleeping: sleep::SleepingGroups,
    settling: settling::Settling,
    distance_readings: BTreeMap<molla_sim::runtime::JointHandle, DistanceReading>,
    cone_readings: BTreeMap<molla_sim::runtime::JointHandle, ConeReading>,
    distance_stamp: Option<molla_sim::runtime::DynamicsStamp>,
    reaction_readout: bool,
    reaction_frame: ReactionFrame,
    reaction_stamp: Option<molla_sim::runtime::DynamicsStamp>,
    break_events: Vec<JointBreakEvent>,
    completed_steps: u64,
    #[cfg(feature = "gpu")]
    gpu: Option<gpu::RuntimeGpu>,
}

impl RigidWorld {
    pub fn new() -> Result<Self> {
        Self::from_scene(RigidScene::new()?)
    }

    pub fn from_scene(scene: RigidScene) -> Result<Self> {
        Self::validate_cone_geometry(&scene)?;
        Ok(Self {
            scene,
            solver: FeatherstoneSolver::new_cpu()
                .with_rigorous_d_solve(true)
                .with_stable_motor_feedback(true)
                .with_substeps(8),
            wheels: crate::wheel_forces::WheelForces::default(),
            tracks: crate::track_forces::TrackForces::default(),
            devices: crate::devices::Devices::default(),
            pipeline: CollisionPipeline::new().with_sensor_overlaps(true),
            bounds: molla_geometry::shape_world::SceneShapeBounds::default(),
            contacts: Contacts::with_capacity(64),
            impacts: crate::rigid_impact::ImpactCache::default(),
            contact_revision: None,
            queries: Mutex::new(CpuRaycaster::default()),
            quarantined: Vec::new(),
            sleeping: sleep::SleepingGroups::default(),
            settling: settling::Settling::default(),
            distance_readings: BTreeMap::new(),
            cone_readings: BTreeMap::new(),
            distance_stamp: None,
            reaction_readout: false,
            reaction_frame: ReactionFrame::default(),
            reaction_stamp: None,
            break_events: Vec::new(),
            completed_steps: 0,
            #[cfg(feature = "gpu")]
            gpu: None,
        })
    }

    pub fn step(&mut self, dt: Real) -> Result<()> {
        self.step_entry(dt, |_, _| false)
    }

    /// Enable substep reaction reconstruction and frame means/peaks.
    pub fn set_reaction_readout(&mut self, enabled: bool) -> Result<()> {
        self.reaction_readout = enabled;
        self.reaction_frame = ReactionFrame::default();
        self.reaction_stamp = None;
        Ok(())
    }

    pub fn reaction_readout_enabled(&self) -> bool {
        self.reaction_readout
    }

    /// Solved frame's mean and peak loads; sleeping joints have no sample.
    pub fn joint_reaction(&self, joint: molla_sim::runtime::JointHandle) -> Option<&JointReaction> {
        (self.reaction_readout && self.reaction_stamp == Some(self.scene.dynamics_stamp()))
            .then(|| self.reaction_frame.means.get(&joint))
            .flatten()
    }

    pub fn joint_reaction_samples(&self) -> Option<&[JointReactionSample]> {
        (self.reaction_readout && self.reaction_stamp == Some(self.scene.dynamics_stamp()))
            .then_some(self.reaction_frame.samples.as_slice())
    }

    /// Valid solved frame's radial reaction on the child, in world coordinates and N.
    /// Wholly sleeping links have no sample for an unsolved frame.
    pub fn distance_reading(
        &self,
        joint: molla_sim::runtime::JointHandle,
    ) -> Option<&DistanceReading> {
        if self.scene.joint(joint)?.kind != molla_sim::JointType::Distance {
            return None;
        }
        (self.distance_stamp == Some(self.scene.dynamics_stamp()))
            .then(|| self.distance_readings.get(&joint))
            .flatten()
    }

    /// Valid solved frame's rigid external ball anchor state; cone torque is separate.
    pub fn ball_anchor_reading(
        &self,
        joint: molla_sim::runtime::JointHandle,
    ) -> Option<BallAnchorReading> {
        if self.distance_stamp != Some(self.scene.dynamics_stamp())
            || !self.scene.rigid_ball_joints().any(|(id, _)| id == joint)
        {
            return None;
        }
        self.distance_readings
            .get(&joint)
            .map(|reading| BallAnchorReading {
                separation: reading.separation,
                force: reading.force,
                sample_dt: reading.sample_dt,
            })
    }

    /// Valid solved frame's cone state and world-space child torque in Nm.
    pub fn cone_reading(&self, joint: molla_sim::runtime::JointHandle) -> Option<&ConeReading> {
        (self.distance_stamp == Some(self.scene.dynamics_stamp()))
            .then(|| self.cone_readings.get(&joint))
            .flatten()
    }

    /// Bodies isolated before the latest step attempt with a valid time increment.
    pub fn quarantined_bodies(&self) -> &[BodyHandle] {
        &self.quarantined
    }

    pub fn cast_ray(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: Real,
    ) -> Result<Option<molla_geometry::raycast_cpu::RayHit>> {
        self.cast_ray_filtered(origin, direction, max_distance, |_| true)
    }

    pub fn cast_ray_filtered(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: Real,
        include: impl FnMut(ColliderHandle) -> bool,
    ) -> Result<Option<molla_geometry::raycast_cpu::RayHit>> {
        let mut queries = self.queries.lock().unwrap_or_else(|poison| {
            let mut queries = poison.into_inner();
            *queries = CpuRaycaster::default();
            self.queries.clear_poison();
            queries
        });
        queries.cast_ray_filtered(&self.scene, origin, direction, max_distance, include)
    }

    pub fn collider_local_aabb(&self, id: ColliderHandle) -> Result<Aabb> {
        collider_query::collider_local_aabb(&self.scene, id)
    }

    pub fn collider_world_aabb(&self, id: ColliderHandle) -> Result<Aabb> {
        collider_query::collider_world_aabb(&self.scene, id)
    }

    pub fn collider_aabb_at_pose(&self, id: ColliderHandle, pose: Transform) -> Result<Aabb> {
        collider_query::collider_aabb_at_pose(&self.scene, id, pose)
    }

    pub fn collider_shape_view(&self, id: ColliderHandle) -> Result<ColliderShapeView> {
        collider_query::collider_shape_view(&self.scene, id)
    }

    pub fn collider_mass_properties(
        &self,
        id: ColliderHandle,
    ) -> Result<molla_sim::MassProperties> {
        let collider = self
            .scene
            .collider(id)
            .ok_or_else(|| Error::Build("unknown runtime collider handle".into()))?;
        molla_geometry::mass_properties::collider_mass_properties(collider)
    }

    pub fn set_body_additional_mass(
        &mut self,
        id: BodyHandle,
        additional: molla_sim::MassProperties,
    ) -> Result<()> {
        self.set_body_additional_mass_with_wake(id, additional, true)
    }

    /// Replace additional mass with optional dependency-group activation.
    pub fn set_body_additional_mass_with_wake(
        &mut self,
        id: BodyHandle,
        additional: molla_sim::MassProperties,
        wake: bool,
    ) -> Result<()> {
        molla_geometry::mass_properties::set_body_additional_mass_with_wake(
            &mut self.scene,
            id,
            additional,
            wake,
        )
    }

    pub fn recompute_body_mass(&mut self, id: BodyHandle) -> Result<()> {
        molla_geometry::mass_properties::recompute_body_mass(&mut self.scene, id)
    }

    pub fn apply_body_impulse(&mut self, id: BodyHandle, impulse: SpatialVector) -> Result<()> {
        self.apply_body_impulse_with_wake(id, impulse, true)
    }

    /// Apply an articulated impulse, optionally activating only its dependency group.
    pub fn apply_body_impulse_with_wake(
        &mut self,
        id: BodyHandle,
        impulse: SpatialVector,
        wake: bool,
    ) -> Result<()> {
        let desc = self
            .scene
            .body(id)
            .ok_or_else(|| Error::Build("unknown runtime body handle".into()))?;
        if !impulse.linear.is_finite() || !impulse.angular.is_finite() {
            return Err(Error::Build("non-finite body impulse".into()));
        }
        if !desc.enabled
            || desc.kind != molla_sim::runtime::BodyKind::Dynamic
            || impulse == SpatialVector::ZERO
        {
            return Ok(());
        }
        let i = self
            .scene
            .body_index(id)
            .ok_or_else(|| Error::Build("unknown runtime body handle".into()))?
            .index();
        let mut next = self.scene.state().clone_host()?;
        crate::rigid_impulse::apply_impulse(self.scene.model(), &mut next, i, impulse)?;
        let velocities = next.joint_qd.host()?;
        if wake && velocities == self.scene.state().joint_qd.host()? {
            return self.wake_body(id);
        }
        self.scene
            .set_generalized_velocities_with_wake(velocities, wake)
    }

    pub fn apply_body_impulse_at_point(
        &mut self,
        id: BodyHandle,
        impulse: Vec3,
        point: Vec3,
    ) -> Result<()> {
        self.apply_body_impulse_at_point_with_wake(id, impulse, point, true)
    }

    /// Apply a world-point impulse with optional dependency-group activation.
    pub fn apply_body_impulse_at_point_with_wake(
        &mut self,
        id: BodyHandle,
        impulse: Vec3,
        point: Vec3,
        wake: bool,
    ) -> Result<()> {
        let torque = (point - self.scene.body_world_com(id)?).cross(impulse);
        self.apply_body_impulse_with_wake(id, SpatialVector::new(impulse, torque), wake)
    }

    /// Quarantine invalid input, then advance atomically; `true` excludes that body pair.
    pub fn step_filtered(
        &mut self,
        dt: Real,
        excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
    ) -> Result<()> {
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() && self.scene.model().shape_count != 0 {
            return Err(Error::Solver("GPU rigid runtime does not yet support host contact filter callbacks; use scene collision permissions".into()));
        }
        self.step_entry(dt, excluded)
    }

    fn step_entry(
        &mut self,
        dt: Real,
        excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
    ) -> Result<()> {
        let next = self
            .completed_steps
            .checked_add(1)
            .ok_or_else(|| Error::Solver("runtime step counter overflow".into()))?;
        self.step_filtered_inner(dt, excluded)?;
        self.completed_steps = next;
        Ok(())
    }

    pub fn completed_steps(&self) -> u64 {
        self.completed_steps
    }

    fn step_filtered_inner(
        &mut self,
        dt: Real,
        excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
    ) -> Result<()> {
        self.distance_stamp = None;
        self.reaction_stamp = None;
        Self::validate_cone_geometry(&self.scene)?;
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            return self.step_gpu(dt);
        }
        if self.scene.breakable_joints().next().is_some() {
            return self.step_breakable_cpu(dt, excluded, |_, _, _, _| Ok(()), false);
        }
        if dt.is_finite() && dt > 0.0 {
            self.devices.before_step(&mut self.scene, dt)?;
        }
        let changed = self.settling.changed_inputs(&self.scene, &self.wheels);
        self.tracks.prepare(&self.scene);
        if !self.tracks.is_empty() || self.devices.needs_forces() {
            return self.step_with_forces_inner(dt, excluded, |_, _, _, _| Ok(()), true);
        }
        if self.sleeping.any_valid(&self.scene, &self.wheels) {
            self.step_sleeping(dt, excluded)?;
            self.devices
                .after_step(&self.scene, self.solver.joint_efforts())?;
            if self.reaction_readout {
                self.reaction_stamp = Some(self.scene.dynamics_stamp());
            }
        } else {
            self.step_with_forces_inner(dt, excluded, |_, _, _, _| Ok(()), false)?;
        }
        self.finish_settling(dt, &changed);
        Ok(())
    }

    /// Evaluate additional transient force elements before each Featherstone substep.
    /// The scene supplies handles/descriptors; the state argument supplies substep kinematics.
    pub fn step_with_forces(
        &mut self,
        dt: Real,
        excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
        forces: impl FnMut(&RigidScene, &molla_sim::State, Real, &mut [SpatialVector]) -> Result<()>,
    ) -> Result<()> {
        let next = self
            .completed_steps
            .checked_add(1)
            .ok_or_else(|| Error::Solver("runtime step counter overflow".into()))?;
        self.step_with_forces_entry(dt, excluded, forces)?;
        self.completed_steps = next;
        Ok(())
    }

    fn step_with_forces_entry(
        &mut self,
        dt: Real,
        excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
        forces: impl FnMut(&RigidScene, &molla_sim::State, Real, &mut [SpatialVector]) -> Result<()>,
    ) -> Result<()> {
        self.distance_stamp = None;
        self.reaction_stamp = None;
        Self::validate_cone_geometry(&self.scene)?;
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            return Err(Error::Solver(
                "GPU rigid runtime does not yet support host force callbacks".into(),
            ));
        }
        if self.scene.breakable_joints().next().is_some() {
            return self.step_breakable_cpu(dt, excluded, forces, true);
        }
        if dt.is_finite() && dt > 0.0 {
            self.devices.before_step(&mut self.scene, dt)?;
        }
        self.step_with_forces_inner(dt, excluded, forces, true)
    }

    fn validate_cone_geometry(scene: &RigidScene) -> Result<()> {
        for (id, joint, limits) in scene.cone_joints() {
            if crate::distance_constraints::enabled(scene, id, joint) {
                crate::cone_constraints::geometry(scene, scene.state(), joint, limits)?;
            }
        }
        Ok(())
    }

    fn step_with_forces_inner(
        &mut self,
        dt: Real,
        mut excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
        mut forces: impl FnMut(&RigidScene, &molla_sim::State, Real, &mut [SpatialVector]) -> Result<()>,
        custom_forces: bool,
    ) -> Result<()> {
        let _span = tracing::debug_span!("rigid_step").entered();
        self.contact_revision = None;
        self.wheels.invalidate();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Error::Solver("step dt must be positive and finite".into()));
        }
        self.quarantined = self.scene.quarantine_non_finite()?;
        let scene = &self.scene;
        self.pipeline
            .set_extra_shape_exclusions(self.wheels.prepare(scene)?);
        let mut wheel_step = self.wheels.begin_step();
        self.tracks.prepare(scene);
        let mut track_step = self.tracks.clone();
        let excluded = std::cell::RefCell::new(&mut excluded);
        let wheels = &mut self.wheels;
        let devices = &mut self.devices;
        let weight = 1.0 / self.solver.substeps.max(1) as Real;
        let mut substep = 0usize;
        let mut next = scene.model().state()?;
        let mut readings = ConstraintReadings::default();
        let capture = self.reaction_readout;
        let mut reaction_frame = ReactionFrame::default();
        let mut reaction_substep = 0usize;
        let compliant = std::cell::RefCell::new(Vec::new());
        let bounds = self.bounds.prepare(scene);
        let common = crate::mixed_constraints::needed(scene)?;
        let contact_impacts = self.solver.contact_impacts;
        let impacts = std::cell::RefCell::new(self.impacts.clone());
        self.solver.step_with_collision_constraints(
            scene.model(),
            scene.state(),
            scene.control(),
            &mut self.pipeline,
            &mut self.contacts,
            dt,
            &mut next,
            |a, b| {
                excluded.borrow_mut()(
                    a.and_then(|i| scene.body_handle(i)),
                    b.and_then(|i| scene.body_handle(i)),
                )
            },
            |model, state, sub_dt, wrenches, mass_seed, contacts| {
                let poses = state.body_q.host()?;
                let velocities = state.body_qd.host()?;
                let masses = model.body_mass.host()?;
                let inertias = model.body_inertia.host()?;
                for (id, desc) in scene.bodies() {
                    if !desc.enabled || desc.kind != molla_sim::runtime::BodyKind::Dynamic {
                        continue;
                    }
                    let i = scene
                        .body_index(id)
                        .ok_or_else(|| Error::Solver("missing damping body".into()))?
                        .index();
                    let coefficient = |d: Real| {
                        if d > 0.0 {
                            1.0 / (1.0 / d + sub_dt)
                        } else {
                            0.0
                        }
                    };
                    let linear = coefficient(desc.linear_damping);
                    let angular = coefficient(desc.angular_damping);
                    let rotation = Mat3::from_quat(poses[i].rotation);
                    wrenches[i].linear -= velocities[i].linear * (masses[i] * linear);
                    wrenches[i].angular -= (rotation * inertias[i] * rotation.transpose())
                        * velocities[i].angular
                        * angular;
                }
                let mut response =
                    crate::articulated_response::ArticulatedResponse::new(scene.model(), state);
                wheels.apply(
                    scene,
                    &mut response,
                    sub_dt,
                    wrenches,
                    &mut wheel_step,
                    |a, b| excluded.borrow_mut()(a, b),
                )?;
                let reactions = crate::loop_forces::apply(scene, &mut response, sub_dt, wrenches)?;
                devices.record_reactions(&reactions, weight, substep == 0);
                if capture {
                    *compliant.borrow_mut() = reactions;
                }
                devices.apply(scene, state, wrenches, weight)?;
                substep += 1;
                track_step.apply(scene, &mut response, contacts, sub_dt, wrenches)?;
                forces(scene, state, sub_dt, wrenches)?;
                if contact_impacts {
                    impacts.borrow_mut().prepare(scene, state, contacts)?;
                }
                response.export_mass(mass_seed)
            },
            Some(&bounds),
            common,
            |solver, input, output, sub_dt, seed| {
                let resting = impacts.borrow_mut().resting_selection(scene, None)?;
                let mut next_readings = if common {
                    solver.correct_mixed_cpu(
                        scene,
                        input,
                        output,
                        sub_dt,
                        Some(&resting),
                        Some(seed),
                    )?
                } else {
                    crate::distance_constraints::apply_selected(
                        scene,
                        solver,
                        input,
                        output,
                        sub_dt,
                        Some(&resting),
                    )?
                };
                let position_reference = solver.correct_impact_cpu(
                    &mut impacts.borrow_mut(),
                    scene,
                    input,
                    output,
                    sub_dt,
                    &mut next_readings,
                    None,
                )?;
                if capture {
                    for sample in crate::joint_reactions::sample_cpu(
                        scene,
                        input,
                        output,
                        sub_dt,
                        reaction_substep,
                        &next_readings,
                        &compliant.borrow(),
                        solver.contact_forces(),
                        None,
                    )? {
                        reaction_frame.add(sample, dt)?;
                    }
                    reaction_substep += 1;
                }
                if solver.momentum_transport {
                    crate::rigid_momentum::transport(
                        scene.model(),
                        input,
                        output,
                        sub_dt,
                        position_reference.as_ref(),
                        None,
                    )?;
                }
                crate::distance_constraints::refresh_readings(scene, output, &mut next_readings)?;
                readings.accumulate(next_readings, weight, dt);
                Ok(())
            },
        )?;
        self.scene.commit_step(next)?;
        self.impacts = impacts.into_inner();
        self.impacts.record_published(&self.scene);
        self.distance_readings = readings.distance;
        self.cone_readings = readings.cones;
        self.distance_stamp = Some(self.scene.dynamics_stamp());
        self.devices
            .after_step(&self.scene, self.solver.joint_efforts())?;
        if capture {
            self.reaction_frame = reaction_frame;
            self.reaction_stamp = Some(self.scene.dynamics_stamp());
        }
        self.tracks = track_step;
        self.sleeping.clear();
        if custom_forces {
            self.settling.clear_windows();
        }
        self.wheels.publish(&self.scene, wheel_step);
        self.contact_revision = Some(self.scene.stamp());
        Ok(())
    }

    /// Built-in dependencies from active joints and the latest valid contact/tyre samples.
    pub fn islands(&self) -> Result<crate::rigid_islands::RigidIslands> {
        let scene = &self.scene;
        let mut connections = Vec::new();
        let mut handles = vec![None; scene.model().body_count];
        for (body, desc) in scene.bodies() {
            let index = scene
                .body_index(body)
                .ok_or_else(|| Error::Solver("missing island body index".into()))?;
            handles[index.index()] = Some(body);
            if desc.enabled
                && desc.kind == molla_sim::runtime::BodyKind::Dynamic
                && let Some(wheel) = self.wheels.descriptor(body)
                && scene
                    .body(wheel.heading_body)
                    .is_some_and(|body| body.enabled)
                && scene.joint_index(wheel.spin_joint).is_some()
            {
                connections.push([Some(body), Some(wheel.heading_body)]);
            }
        }
        if self.contact_revision == Some(scene.stamp()) {
            for reading in self.solver.contact_forces() {
                if reading.penetration < 0.0
                    || reading.colliders.iter().flatten().any(|&shape| {
                        scene
                            .collider_handle(shape)
                            .and_then(|id| scene.collider(id))
                            .is_none_or(|desc| !desc.enabled || desc.sensor)
                    })
                {
                    continue;
                }
                connections.push(
                    reading
                        .bodies
                        .map(|body| body.and_then(|id| handles.get(id.index()).copied().flatten())),
                );
            }
        }
        for sample in self.wheels.samples(scene) {
            if let Some(ground) = scene.collider(sample.ground) {
                connections.push([Some(sample.wheel), ground.parent]);
            }
        }
        crate::rigid_islands::RigidIslands::build(scene, connections)
    }

    /// All substep samples grouped by stable collider pair; sum impulses, not instantaneous forces.
    pub fn contact_pairs(&self) -> Vec<ContactPair> {
        if self.contact_revision != Some(self.scene.stamp()) {
            return Vec::new();
        }
        let mut pairs = BTreeMap::new();
        for reading in self.solver.contact_forces() {
            let [Some(a), Some(b)] = reading
                .colliders
                .map(|i| i.and_then(|i| self.scene.collider_handle(i)))
            else {
                continue;
            };
            let pair = pairs.entry((a, b)).or_insert_with(|| ContactPair {
                colliders: [a, b],
                bodies: reading
                    .bodies
                    .map(|b| b.and_then(|i| self.scene.body_handle(i))),
                normal: reading.normal,
                points: Vec::new(),
            });
            pair.points.push(reading.clone());
        }
        pairs.into_values().collect()
    }
}
