//! CPU rigid-world stepping and stable-handle contact readout.

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
use crate::penalty_contact::ContactForce;

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
    pipeline: CollisionPipeline,
    contacts: Contacts,
    contact_revision: Option<SceneStamp>,
    queries: Mutex<CpuRaycaster>,
    quarantined: Vec<BodyHandle>,
}

impl RigidWorld {
    pub fn new() -> Result<Self> {
        Self::from_scene(RigidScene::new()?)
    }

    pub fn from_scene(scene: RigidScene) -> Result<Self> {
        Ok(Self {
            scene,
            solver: FeatherstoneSolver::new_cpu()
                .with_rigorous_d_solve(true)
                .with_substeps(8),
            wheels: crate::wheel_forces::WheelForces::default(),
            pipeline: CollisionPipeline::new().with_sensor_overlaps(true),
            contacts: Contacts::with_capacity(64),
            contact_revision: None,
            queries: Mutex::new(CpuRaycaster::default()),
            quarantined: Vec::new(),
        })
    }

    pub fn step(&mut self, dt: Real) -> Result<()> {
        self.step_filtered(dt, |_, _| false)
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
        molla_geometry::mass_properties::set_body_additional_mass(&mut self.scene, id, additional)
    }

    pub fn recompute_body_mass(&mut self, id: BodyHandle) -> Result<()> {
        molla_geometry::mass_properties::recompute_body_mass(&mut self.scene, id)
    }

    pub fn apply_body_impulse(&mut self, id: BodyHandle, impulse: SpatialVector) -> Result<()> {
        let desc = self
            .scene
            .body(id)
            .ok_or_else(|| Error::Build("unknown runtime body handle".into()))?;
        if !impulse.linear.is_finite() || !impulse.angular.is_finite() {
            return Err(Error::Build("non-finite body impulse".into()));
        }
        if !desc.enabled || desc.kind != molla_sim::runtime::BodyKind::Dynamic {
            return Ok(());
        }
        let i = self
            .scene
            .body_index(id)
            .ok_or_else(|| Error::Build("unknown runtime body handle".into()))?
            .index();
        let mut next = self.scene.state().clone_host()?;
        crate::rigid_impulse::apply_impulse(self.scene.model(), &mut next, i, impulse)?;
        self.scene.commit_step(next)?;
        Ok(())
    }

    pub fn apply_body_impulse_at_point(
        &mut self,
        id: BodyHandle,
        impulse: Vec3,
        point: Vec3,
    ) -> Result<()> {
        let torque = (point - self.scene.body_world_com(id)?).cross(impulse);
        self.apply_body_impulse(id, SpatialVector::new(impulse, torque))
    }

    /// Quarantine invalid input, then advance atomically; `true` excludes that body pair.
    pub fn step_filtered(
        &mut self,
        dt: Real,
        excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
    ) -> Result<()> {
        self.step_with_forces(dt, excluded, |_, _, _, _| Ok(()))
    }

    /// Evaluate additional transient force elements before each Featherstone substep.
    /// The scene supplies handles/descriptors; the state argument supplies substep kinematics.
    pub fn step_with_forces(
        &mut self,
        dt: Real,
        mut excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
        mut forces: impl FnMut(&RigidScene, &molla_sim::State, Real, &mut [SpatialVector]) -> Result<()>,
    ) -> Result<()> {
        self.contact_revision = None;
        self.wheels.invalidate();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Error::Solver("step dt must be positive and finite".into()));
        }
        self.quarantined = self.scene.quarantine_non_finite()?;
        let scene = &self.scene;
        self.pipeline
            .set_extra_shape_exclusions(self.wheels.prepare(scene)?);
        let mut wheel_samples = Vec::new();
        let excluded = std::cell::RefCell::new(&mut excluded);
        let wheels = &mut self.wheels;
        let mut next = scene.model().state()?;
        self.solver.step_with_collision_and_forces(
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
            |model, state, sub_dt, wrenches| {
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
                wheels.apply(
                    scene,
                    state,
                    sub_dt,
                    wrenches,
                    &mut wheel_samples,
                    |a, b| excluded.borrow_mut()(a, b),
                )?;
                crate::loop_forces::apply(scene, state, sub_dt, wrenches)?;
                forces(scene, state, sub_dt, wrenches)
            },
        )?;
        self.scene.commit_step(next)?;
        self.wheels.publish(&self.scene, wheel_samples);
        self.contact_revision = Some(self.scene.stamp());
        Ok(())
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
