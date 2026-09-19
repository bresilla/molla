//! Stable-handle tyre force elements for CPU articulated worlds.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::pressure_tire::{PressureTireConfig, PressureTireSample, PressureTireState};
use molla_core::{Error, Result, ShapeId};
use molla_geometry::{collider_query::collider_world_aabb, raycast_cpu::CpuRaycaster};
use molla_math::{Real, SpatialVector, Vec3};
use molla_sim::runtime::{
    BodyHandle, BodyKind, ColliderGeometry, ColliderHandle, JointHandle, RigidScene, SceneStamp,
};
use molla_sim::{JointType, State};
use molla_vehicle::{Terrain, TireModel, TireOutput, WheelContact, wheel_ground_wrench};

#[derive(Clone)]
pub struct WheelDesc {
    pub body: BodyHandle,
    pub spin_joint: JointHandle,
    pub heading_body: BodyHandle,
    pub local_heading: Vec3,
    pub local_hub: Vec3,
    /// Authored joint angular rate to positive rolling speed: +1 or -1.
    pub spin_sign: Real,
    pub radius: Real,
    pub normal_stiffness: Real,
    pub normal_damping: Real,
    pub tire: Arc<dyn TireModel>,
}

/// Piecewise-constant surface friction in collider-local XZ cells.
#[derive(Clone, Debug)]
pub struct FrictionGrid {
    pub origin: [Real; 2],
    pub cell_size: [Real; 2],
    pub cols: usize,
    pub rows: usize,
    pub values: Vec<Real>,
}

impl FrictionGrid {
    fn validate(&self) -> Result<()> {
        if self.cols == 0
            || self.rows == 0
            || self.cols.checked_mul(self.rows) != Some(self.values.len())
            || self.origin.iter().any(|v| !v.is_finite())
            || self.cell_size.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self.values.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(Error::Build("invalid terrain friction grid".into()));
        }
        Ok(())
    }

    fn sample(&self, point: Vec3) -> Option<Real> {
        let x = (point.x - self.origin[0]) / self.cell_size[0];
        let z = (point.z - self.origin[1]) / self.cell_size[1];
        if x < 0.0 || z < 0.0 || x >= self.cols as Real || z >= self.rows as Real {
            return None;
        }
        self.values
            .get(z.floor() as usize * self.cols + x.floor() as usize)
            .copied()
    }
}

#[derive(Clone, Debug)]
pub struct WheelSample {
    pub wheel: BodyHandle,
    pub ground: ColliderHandle,
    pub dt: Real,
    pub normal: Vec3,
    pub friction: Real,
    pub output: TireOutput,
    pub pressure: Option<PressureTireSample>,
}

pub(crate) struct WheelStep {
    samples: Vec<WheelSample>,
    pressure_states: BTreeMap<BodyHandle, PressureTireState>,
}

#[derive(Default)]
pub struct WheelForces {
    wheels: BTreeMap<BodyHandle, WheelDesc>,
    pressure_states: BTreeMap<BodyHandle, PressureTireState>,
    grounds: BTreeMap<ColliderHandle, Option<FrictionGrid>>,
    queries: CpuRaycaster,
    samples: Vec<WheelSample>,
    sample_stamp: Option<SceneStamp>,
    prepared_stamp: Option<SceneStamp>,
    height_range: Option<(Real, Real)>,
    exclusions: Vec<(ShapeId, ShapeId)>,
}

fn invalid(message: &str) -> Error {
    Error::Build(message.into())
}

fn validate_ground(scene: &RigidScene, id: ColliderHandle) -> Result<()> {
    let collider = scene
        .collider(id)
        .ok_or_else(|| invalid("unknown wheel terrain collider"))?;
    if !matches!(
        collider.geometry,
        ColliderGeometry::Heightfield(_)
            | ColliderGeometry::TriMesh(_)
            | ColliderGeometry::Cuboid { .. }
    ) || collider.sensor
        || collider
            .parent
            .is_some_and(|b| scene.body(b).is_none_or(|b| b.kind != BodyKind::Fixed))
    {
        return Err(invalid(
            "wheel terrain requires a static, nonsensor heightfield, triangle mesh or cuboid",
        ));
    }
    Ok(())
}

impl WheelForces {
    pub fn insert(&mut self, scene: &RigidScene, desc: WheelDesc) -> Result<()> {
        let joint = scene
            .joint(desc.spin_joint)
            .ok_or_else(|| invalid("unknown wheel spin joint"))?;
        if scene.body(desc.body).is_none()
            || scene.body(desc.heading_body).is_none()
            || desc.heading_body == desc.body
            || joint.kind != JointType::Revolute
            || (joint.child != desc.body && joint.parent != Some(desc.body))
            || !desc.local_heading.length().is_finite()
            || desc.local_heading.length() < 1e-9
            || !desc.local_hub.is_finite()
            || !matches!(desc.spin_sign, -1.0 | 1.0)
            || !desc.radius.is_finite()
            || desc.radius <= 0.0
            || !desc.normal_stiffness.is_finite()
            || desc.normal_stiffness < 0.0
            || !desc.normal_damping.is_finite()
            || desc.normal_damping < 0.0
        {
            return Err(invalid("invalid articulated wheel descriptor"));
        }
        if self
            .pressure_states
            .get(&desc.body)
            .is_some_and(|p| p.config.tire.radius != desc.radius)
        {
            return Err(invalid("wheel radius differs from pressure tyre geometry"));
        }
        self.wheels.insert(desc.body, desc);
        self.prepared_stamp = None;
        self.invalidate();
        Ok(())
    }

    pub fn remove(&mut self, body: BodyHandle) -> Option<WheelDesc> {
        self.prepared_stamp = None;
        self.invalidate();
        self.pressure_states.remove(&body);
        self.wheels.remove(&body)
    }

    pub fn configure_pressure_tire(
        &mut self,
        body: BodyHandle,
        config: PressureTireConfig,
    ) -> Result<()> {
        config.validate()?;
        let wheel = self
            .wheels
            .get(&body)
            .ok_or_else(|| invalid("unknown pressure tyre wheel"))?;
        if wheel.radius != config.tire.radius {
            return Err(invalid("pressure tyre radius differs from wheel geometry"));
        }
        let mut state = self
            .pressure_states
            .get(&body)
            .copied()
            .unwrap_or(PressureTireState {
                config,
                pressure_pa: config.tire.pressure_pa,
                target_pressure_pa: config.tire.pressure_pa,
            });
        if [state.pressure_pa, state.target_pressure_pa]
            .iter()
            .any(|p| !(config.min_pressure_pa..=config.max_pressure_pa).contains(p))
        {
            return Err(invalid("existing tyre pressure outside new bounds"));
        }
        state.config = config;
        self.pressure_states.insert(body, state);
        self.invalidate();
        Ok(())
    }

    pub fn pressure_tire(
        &self,
        scene: &RigidScene,
        body: BodyHandle,
    ) -> Option<&PressureTireState> {
        let wheel = self.wheels.get(&body)?;
        scene.body(body)?;
        scene.body(wheel.heading_body)?;
        scene.joint(wheel.spin_joint)?;
        self.pressure_states.get(&body)
    }

    /// Atomically set gauge-pressure targets; applied pressure advances during successful steps.
    pub fn set_pressures(
        &mut self,
        scene: &RigidScene,
        targets: &[(BodyHandle, Real)],
    ) -> Result<()> {
        let mut seen = std::collections::BTreeSet::new();
        for &(body, target) in targets {
            let state = self
                .pressure_tire(scene, body)
                .ok_or_else(|| invalid("unknown pressure tyre"))?;
            if !seen.insert(body)
                || !target.is_finite()
                || !(state.config.min_pressure_pa..=state.config.max_pressure_pa).contains(&target)
            {
                return Err(invalid("invalid or duplicate tyre pressure target"));
            }
        }
        for &(body, target) in targets {
            self.pressure_states
                .get_mut(&body)
                .unwrap()
                .target_pressure_pa = target;
        }
        Ok(())
    }

    pub fn remove_pressure_tire(&mut self, body: BodyHandle) -> Option<PressureTireState> {
        self.invalidate();
        self.pressure_states.remove(&body)
    }

    pub fn register_ground(
        &mut self,
        scene: &RigidScene,
        id: ColliderHandle,
        friction: Option<FrictionGrid>,
    ) -> Result<()> {
        validate_ground(scene, id)?;
        if let Some(grid) = &friction {
            grid.validate()?;
        }
        self.grounds.insert(id, friction);
        self.prepared_stamp = None;
        self.invalidate();
        Ok(())
    }

    pub fn remove_ground(&mut self, id: ColliderHandle) {
        self.prepared_stamp = None;
        self.grounds.remove(&id);
        self.invalidate();
    }

    /// All tyre samples from the last successful step; integrate force times `dt` for impulse.
    pub fn samples(&self, scene: &RigidScene) -> &[WheelSample] {
        if self.sample_stamp == Some(scene.stamp()) {
            &self.samples
        } else {
            &[]
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.sample_stamp = None;
        self.samples.clear();
    }

    pub(crate) fn begin_step(&self) -> WheelStep {
        WheelStep {
            samples: Vec::new(),
            pressure_states: self.pressure_states.clone(),
        }
    }

    pub(crate) fn publish(&mut self, scene: &RigidScene, step: WheelStep) {
        self.samples = step.samples;
        self.pressure_states = step.pressure_states;
        self.sample_stamp = Some(scene.stamp());
    }

    pub(crate) fn prepare(&mut self, scene: &RigidScene) -> Result<Vec<(ShapeId, ShapeId)>> {
        self.invalidate();
        if self.prepared_stamp == Some(scene.stamp()) {
            return Ok(self.exclusions.clone());
        }
        self.prepared_stamp = None;
        self.wheels.retain(|_, w| {
            scene.body(w.body).is_some()
                && scene.body(w.heading_body).is_some()
                && scene.joint(w.spin_joint).is_some()
        });
        self.pressure_states
            .retain(|body, _| self.wheels.contains_key(body));
        self.grounds.retain(|id, _| scene.collider(*id).is_some());
        for &id in self.grounds.keys() {
            validate_ground(scene, id)?;
        }
        let mut top = Real::NEG_INFINITY;
        let mut bottom = Real::INFINITY;
        for &id in self.grounds.keys() {
            if scene.collider_is_enabled(id)? {
                let bounds = collider_world_aabb(scene, id)?;
                top = top.max(bounds.max.y);
                bottom = bottom.min(bounds.min.y);
            }
        }
        self.height_range = (top.is_finite() && bottom.is_finite()).then_some((top, bottom));
        let mut pairs = Vec::new();
        for w in self.wheels.values().filter(|w| active(scene, w)) {
            for (id, collider) in scene.colliders().filter(|(_, c)| c.parent == Some(w.body)) {
                if !collider.enabled || collider.sensor {
                    continue;
                }
                for ground in self.grounds.keys() {
                    for &a in scene.collider_indices(id).unwrap_or_default() {
                        for &b in scene.collider_indices(*ground).unwrap_or_default() {
                            pairs.push((a, b));
                        }
                    }
                }
            }
        }
        self.exclusions = pairs.clone();
        self.prepared_stamp = Some(scene.stamp());
        Ok(pairs)
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        dt: Real,
        forces: &mut [SpatialVector],
        step: &mut WheelStep,
        mut excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
    ) -> Result<()> {
        let WheelStep {
            samples,
            pressure_states,
        } = step;
        for state in pressure_states.values_mut() {
            state.advance(dt);
        }
        if self.wheels.is_empty() || self.grounds.is_empty() {
            return Ok(());
        }
        let Some((top, bottom)) = self.height_range else {
            return Ok(());
        };
        let poses = state.body_q.host()?;
        let velocities = state.body_qd.host()?;
        let model = scene.model();
        let mut response = crate::articulated_response::ArticulatedResponse::default();
        for w in self.wheels.values().filter(|w| active(scene, w)) {
            let body = scene
                .body_index(w.body)
                .ok_or_else(|| invalid("missing wheel body"))?
                .index();
            let heading = scene
                .body_index(w.heading_body)
                .ok_or_else(|| invalid("missing wheel heading body"))?
                .index();
            let joint = scene
                .joint_index(w.spin_joint)
                .ok_or_else(|| invalid("missing wheel joint"))?
                .index();
            let pose = poses[body];
            let hub = pose.transform_point(w.local_hub);
            let Some(hit) = self.queries.cast_ray_filtered(
                scene,
                Vec3::new(hub.x, top + 1.0, hub.z),
                -Vec3::Y,
                top - bottom + 2.0,
                |id| {
                    self.grounds.contains_key(&id)
                        && !excluded(Some(w.body), scene.collider(id).and_then(|c| c.parent))
                },
            )?
            else {
                continue;
            };
            let normal = if hit.normal.y < 0.0 {
                -hit.normal
            } else {
                hit.normal
            };
            if normal.y < 1e-6 {
                continue;
            }
            let collider = scene
                .collider(hit.collider)
                .ok_or_else(|| invalid("missing terrain collider"))?;
            let local = scene
                .collider_world_pose(hit.collider)?
                .inverse_transform_point(hit.point);
            let friction = self.grounds[&hit.collider]
                .as_ref()
                .and_then(|g| g.sample(local))
                .unwrap_or(collider.friction);
            let terrain = Surface {
                point: hit.point,
                normal,
                friction,
            };
            let center = pose.transform_point(model.body_com.host()?[body]);
            let velocity = velocities[body].linear + velocities[body].angular.cross(hub - center);
            let offset = model.joint_dof_offset.host()?[joint] as usize;
            let axis = if model.joint_type.host()?[joint] == JointType::D6 as u32 {
                3
            } else {
                0
            };
            let contact = WheelContact {
                hub_pos: hub.to_array(),
                hub_vel: velocity.to_array(),
                heading: (poses[heading].rotation * w.local_heading.normalize()).to_array(),
                spin: state.joint_qd.host()?[offset + axis] * w.spin_sign,
                radius: w.radius,
                normal_stiffness: w.normal_stiffness,
                normal_damping: w.normal_damping,
            };
            let (mut output, mut pressure) = if let Some(state) = pressure_states.get(&w.body) {
                let distance = (hub - hit.point).dot(normal);
                let (output, sample) =
                    crate::pressure_tire::evaluate(state, &contact, normal, distance, friction)?;
                (output, Some(sample))
            } else {
                (
                    wheel_ground_wrench(&contact, &terrain, w.tire.as_ref()),
                    None,
                )
            };
            if output.in_contact {
                let tangent = Vec3::from_array(output.force) - normal * output.fz;
                let point = Vec3::from_array(output.contact_point);
                let wrench = SpatialVector::new(
                    tangent,
                    (point - center).cross(tangent)
                        + normal * output.aligning_moment
                        + pressure.as_ref().map_or(Vec3::ZERO, |p| p.rolling_moment),
                );
                let power = wrench.linear.dot(velocities[body].linear)
                    + wrench.angular.dot(velocities[body].angular);
                let mobility = response.quadratic(model, state, body, wrench)?;
                if mobility > 0.0 {
                    let scale = (-power / (dt * mobility)).clamp(0.0, 1.0);
                    output.force = (normal * output.fz + tangent * scale).to_array();
                    output.aligning_moment *= scale;
                    if let Some(p) = &mut pressure {
                        p.scale_tangent(scale);
                    }
                }
            }
            let force = Vec3::from_array(output.force);
            let point = Vec3::from_array(output.contact_point);
            let torque = (point - center).cross(force)
                + normal * output.aligning_moment
                + pressure.as_ref().map_or(Vec3::ZERO, |p| p.rolling_moment);
            if !force.is_finite()
                || !torque.is_finite()
                || !output.fz.is_finite()
                || !output.slip_ratio.is_finite()
                || !output.slip_angle.is_finite()
            {
                return Err(Error::Solver("non-finite tyre force output".into()));
            }
            forces[body].linear += force;
            forces[body].angular += torque;
            samples.push(WheelSample {
                wheel: w.body,
                ground: hit.collider,
                dt,
                normal,
                friction,
                output,
                pressure,
            });
        }
        Ok(())
    }
}

fn active(scene: &RigidScene, w: &WheelDesc) -> bool {
    scene
        .body(w.body)
        .is_some_and(|b| b.enabled && b.kind == BodyKind::Dynamic)
        && scene.body(w.heading_body).is_some_and(|b| b.enabled)
        && scene.joint_index(w.spin_joint).is_some()
}

struct Surface {
    point: Vec3,
    normal: Vec3,
    friction: Real,
}

impl Terrain for Surface {
    fn height(&self, x: Real, z: Real) -> Real {
        self.point.y
            - ((x - self.point.x) * self.normal.x + (z - self.point.z) * self.normal.z)
                / self.normal.y
    }
    fn normal(&self, _: Real, _: Real) -> [Real; 3] {
        self.normal.to_array()
    }
    fn friction(&self) -> Real {
        self.friction
    }
    fn signed_distance_and_normal(&self, point: [Real; 3]) -> (Real, [Real; 3]) {
        (
            (Vec3::from_array(point) - self.point).dot(self.normal),
            self.normal.to_array(),
        )
    }
}
