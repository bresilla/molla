//! CPU rigid-world stepping and stable-handle contact readout.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_geometry::{
    Aabb, CollisionPipeline,
    collider_query::{self, ColliderShapeView},
};
use molla_math::{Real, Transform, Vec3};
use molla_sim::Contacts;
use molla_sim::runtime::{BodyHandle, ColliderHandle, RigidScene};

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
    pipeline: CollisionPipeline,
    contacts: Contacts,
    contact_revision: Option<u64>,
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
            pipeline: CollisionPipeline::new().with_sensor_overlaps(true),
            contacts: Contacts::with_capacity(64),
            contact_revision: None,
        })
    }

    pub fn step(&mut self, dt: Real) -> Result<()> {
        self.step_filtered(dt, |_, _| false)
    }

    pub fn cast_ray(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: Real,
    ) -> Result<Option<molla_geometry::raycast_cpu::RayHit>> {
        molla_geometry::raycast_cpu::cast_ray(&self.scene, origin, direction, max_distance)
    }

    pub fn cast_ray_filtered(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: Real,
        include: impl FnMut(ColliderHandle) -> bool,
    ) -> Result<Option<molla_geometry::raycast_cpu::RayHit>> {
        molla_geometry::raycast_cpu::cast_ray_filtered(
            &self.scene,
            origin,
            direction,
            max_distance,
            include,
        )
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

    /// Advance the scene atomically; `true` excludes all contacts for that body pair.
    pub fn step_filtered(
        &mut self,
        dt: Real,
        mut excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
    ) -> Result<()> {
        self.contact_revision = None;
        let scene = &self.scene;
        let mut next = scene.model().state()?;
        self.solver.step_with_collision(
            scene.model(),
            scene.state(),
            scene.control(),
            &mut self.pipeline,
            &mut self.contacts,
            dt,
            &mut next,
            |a, b| {
                excluded(
                    a.and_then(|i| scene.body_handle(i)),
                    b.and_then(|i| scene.body_handle(i)),
                )
            },
        )?;
        if next
            .body_q
            .host()?
            .iter()
            .any(|p| !p.position.is_finite() || !p.rotation.is_finite())
            || next
                .body_qd
                .host()?
                .iter()
                .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
            || next
                .joint_q
                .host()?
                .iter()
                .chain(next.joint_qd.host()?)
                .any(|x| !x.is_finite())
        {
            return Err(Error::Solver("rigid step produced non-finite state".into()));
        }
        *self.scene.state_mut() = next;
        self.contact_revision = Some(self.scene.revision());
        Ok(())
    }

    /// All substep samples grouped by stable collider pair; sum impulses, not instantaneous forces.
    pub fn contact_pairs(&self) -> Vec<ContactPair> {
        if self.contact_revision != Some(self.scene.revision()) {
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
