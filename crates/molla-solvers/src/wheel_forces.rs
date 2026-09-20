//! Stable-handle tyre force elements for CPU articulated worlds.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::pressure_tire::{PressureTireConfig, PressureTireSample, PressureTireState};
use molla_core::{Error, Result, ShapeId};
use molla_geometry::{collider_query::collider_world_aabb, raycast_cpu::CpuRaycaster};
use molla_math::{Real, SpatialVector, Vec3};
use molla_sim::JointType;
use molla_sim::runtime::{
    BodyHandle, BodyKind, ColliderGeometry, ColliderHandle, JointHandle, RigidScene, SceneStamp,
};
use molla_vehicle::{Terrain, TireModel, TireOutput, WheelContact, wheel_ground_wrench};

mod readout;
pub use readout::{WheelSampleIter, WheelSamples};
mod sleeping;
pub(crate) use sleeping::SleepWheelGeometry;
use sleeping::query_ground;

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

struct GroundMaterial {
    grid: Option<FrictionGrid>,
    max_friction: Real,
}

#[derive(Clone, Debug)]
pub struct WheelSample {
    pub wheel: BodyHandle,
    pub ground: ColliderHandle,
    pub ground_point: Vec3,
    /// Integration duration; zero denotes retained support from a sleeping wheel.
    pub dt: Real,
    pub normal: Vec3,
    pub friction: Real,
    pub output: TireOutput,
    pub pressure: Option<PressureTireSample>,
}

pub(crate) struct WheelStep {
    samples: Vec<WheelSample>,
    pressure_states: BTreeMap<BodyHandle, PressureTireState>,
    shear: BTreeMap<BodyHandle, WheelShear>,
    held_samples: BTreeMap<BodyHandle, WheelSample>,
}

#[derive(Clone)]
struct WheelShear {
    ground: ColliderHandle,
    state: crate::pressure_tire::PressureShear,
    poses: Option<[molla_math::Transform; 3]>,
}

#[derive(Default)]
pub struct WheelForces {
    wheels: BTreeMap<BodyHandle, WheelDesc>,
    pressure_states: BTreeMap<BodyHandle, PressureTireState>,
    shear: BTreeMap<BodyHandle, WheelShear>,
    grounds: BTreeMap<ColliderHandle, GroundMaterial>,
    queries: CpuRaycaster,
    samples: Vec<WheelSample>,
    sample_stamp: Option<SceneStamp>,
    sample_dependencies: readout::SampleDependencies,
    prepared_stamp: Option<SceneStamp>,
    height_range: Option<(Real, Real)>,
    exclusions: Vec<(ShapeId, ShapeId)>,
    activation_revision: u64,
    activation_identity: Arc<()>,
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
    pub(crate) fn activation_stamp(&self) -> (Arc<()>, u64) {
        (self.activation_identity.clone(), self.activation_revision)
    }

    pub(crate) fn matches_activation(&self, stamp: &(Arc<()>, u64)) -> bool {
        Arc::ptr_eq(&self.activation_identity, &stamp.0) && self.activation_revision == stamp.1
    }

    pub(crate) fn transitioning(&self, body: BodyHandle) -> bool {
        self.pressure_states
            .get(&body)
            .is_some_and(|state| state.pressure_pa != state.target_pressure_pa)
    }

    pub fn descriptor(&self, body: BodyHandle) -> Option<&WheelDesc> {
        self.wheels.get(&body)
    }

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
        self.invalidate_wheel(desc.body);
        self.wheels.insert(desc.body, desc);
        self.activation_revision = self.activation_revision.wrapping_add(1);
        self.prepared_stamp = None;
        Ok(())
    }

    pub fn remove(&mut self, body: BodyHandle) -> Option<WheelDesc> {
        if !self.wheels.contains_key(&body) {
            return None;
        }
        self.activation_revision = self.activation_revision.wrapping_add(1);
        self.prepared_stamp = None;
        self.invalidate_wheel(body);
        self.pressure_states.remove(&body);
        self.shear.remove(&body);
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
        self.activation_revision = self.activation_revision.wrapping_add(1);
        self.invalidate_wheel(body);
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
            if self.pressure_states[&body].target_pressure_pa != target {
                self.activation_revision = self.activation_revision.wrapping_add(1);
            }
            self.pressure_states
                .get_mut(&body)
                .unwrap()
                .target_pressure_pa = target;
        }
        Ok(())
    }

    pub fn remove_pressure_tire(&mut self, body: BodyHandle) -> Option<PressureTireState> {
        if !self.pressure_states.contains_key(&body) {
            return None;
        }
        self.activation_revision = self.activation_revision.wrapping_add(1);
        self.invalidate_wheel(body);
        self.shear.remove(&body);
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
        let max_friction = friction.as_ref().map_or(0.0, |grid| {
            grid.values.iter().copied().fold(0.0, Real::max)
        });
        self.grounds.insert(id, GroundMaterial { grid: friction, max_friction });
        self.activation_revision = self.activation_revision.wrapping_add(1);
        self.shear.retain(|_, shear| shear.ground != id);
        self.prepared_stamp = None;
        self.invalidate();
        Ok(())
    }

    pub fn remove_ground(&mut self, id: ColliderHandle) {
        self.activation_revision = self.activation_revision.wrapping_add(1);
        self.prepared_stamp = None;
        self.grounds.remove(&id);
        self.shear.retain(|_, shear| shear.ground != id);
        self.invalidate();
    }

    /// All tyre samples from the last successful step; integrate force times `dt` for impulse.
    pub fn samples<'a>(&'a self, scene: &'a RigidScene) -> WheelSamples<'a> {
        let unchanged = self.sample_stamp == Some(scene.stamp())
            && self.sample_dependencies.matches_state(scene);
        let valid = self.sample_stamp.is_some()
            && (unchanged || self.sample_dependencies.matches_grounds(scene));
        WheelSamples::new(
            if valid { &self.samples } else { &[] },
            scene,
            if unchanged { None } else { Some(&self.sample_dependencies) },
        )
    }

    fn invalidate_wheel(&mut self, body: BodyHandle) {
        self.samples.retain(|sample| sample.wheel != body);
    }

    pub(crate) fn invalidate(&mut self) {
        self.sample_stamp = None;
        self.samples.clear();
    }

    pub(crate) fn begin_step(&self) -> WheelStep {
        WheelStep {
            samples: Vec::new(),
            pressure_states: self.pressure_states.clone(),
            shear: self.shear.clone(),
            held_samples: BTreeMap::new(),
        }
    }

    pub(crate) fn begin_sleeping_step(&self, samples: &[WheelSample]) -> WheelStep {
        let mut step = self.begin_step();
        for sample in samples {
            step.held_samples.insert(sample.wheel, sample.clone());
        }
        step
    }

    pub(crate) fn publish(&mut self, scene: &RigidScene, mut step: WheelStep) {
        for (body, shear) in &mut step.shear {
            shear.poses = self
                .wheels
                .get(body)
                .and_then(|wheel| shear_poses(scene, wheel, shear.ground));
        }
        self.samples = step.samples;
        self.pressure_states = step.pressure_states;
        self.shear = step.shear;
        self.sample_dependencies.capture(
            scene,
            &self.wheels,
            &self.grounds,
            self.activation_revision,
        );
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
        self.shear.retain(|body, shear| {
            self.wheels.get(body).is_some_and(|wheel| {
                active(scene, wheel)
                    && self.pressure_states.contains_key(body)
                    && match (shear.poses, shear_poses(scene, wheel, shear.ground)) {
                        (Some(old), Some(new)) => old.iter().zip(new).all(|(a, b)| {
                            (a.position - b.position).length() < 1e-8
                                && a.rotation.dot(b.rotation).abs() > 1.0 - 1e-12
                        }),
                        _ => false,
                    }
            })
        });
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
        response: &mut crate::articulated_response::ArticulatedResponse<'_>,
        dt: Real,
        forces: &mut [SpatialVector],
        step: &mut WheelStep,
        excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
    ) -> Result<()> {
        self.apply_selected(scene, response, dt, forces, step, excluded, None)
    }

    pub(crate) fn apply_selected(
        &mut self,
        scene: &RigidScene,
        response: &mut crate::articulated_response::ArticulatedResponse<'_>,
        dt: Real,
        forces: &mut [SpatialVector],
        step: &mut WheelStep,
        mut excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
        selected: Option<&[bool]>,
    ) -> Result<()> {
        let WheelStep {
            samples,
            pressure_states,
            shear,
            held_samples,
        } = step;
        let state = response.state();
        let _span = tracing::debug_span!("wheel_forces").entered();
        for (body, state) in pressure_states.iter_mut() {
            if selected.is_none_or(|mask| scene.body_index(*body).is_some_and(|i| mask[i.index()])) {
                state.advance(dt);
            }
        }
        if self.wheels.is_empty() || self.grounds.is_empty() {
            shear.clear();
            return Ok(());
        }
        let Some(range) = self.height_range else {
            shear.clear();
            return Ok(());
        };
        let poses = state.body_q.host()?;
        let velocities = state.body_qd.host()?;
        let model = scene.model();
        for w in self.wheels.values().filter(|w| active(scene, w)) {
            let body = scene
                .body_index(w.body)
                .ok_or_else(|| invalid("missing wheel body"))?
                .index();
            let heading = scene
                .body_index(w.heading_body)
                .ok_or_else(|| invalid("missing wheel heading body"))?
                .index();
            if selected.is_some_and(|mask| !mask[body]) {
                if let Some(sample) = held_samples.get(&w.body) {
                    let mut sample = sample.clone();
                    sample.dt = 0.0;
                    sample.output.slip_ratio = 0.0;
                    sample.output.slip_angle = 0.0;
                    samples.push(sample);
                }
                continue;
            }
            let joint = scene
                .joint_index(w.spin_joint)
                .ok_or_else(|| invalid("missing wheel joint"))?
                .index();
            let pose = poses[body];
            let hub = pose.transform_point(w.local_hub);
            let Some(hit) = query_ground(
                &mut self.queries,
                scene,
                &self.grounds,
                range,
                hub,
                |id| !excluded(Some(w.body), scene.collider(id).and_then(|c| c.parent)),
            )?
            else {
                shear.remove(&w.body);
                continue;
            };
            let normal = hit.normal;
            let friction = hit.friction;
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
            let (mut output, pressure) = if let Some(state) = pressure_states.get(&w.body) {
                let distance = (hub - hit.point).dot(normal);
                let history = shear.entry(w.body).or_insert_with(|| WheelShear {
                    ground: hit.collider,
                    state: Default::default(),
                    poses: None,
                });
                if history.ground != hit.collider {
                    history.ground = hit.collider;
                    history.state = Default::default();
                }
                let friction_at = |point| hit.surface.sample(point);
                let (output, sample) = crate::pressure_tire::evaluate_with_surface(
                    state,
                    &contact,
                    normal,
                    distance,
                    hit.surface.reference,
                    Some((&mut history.state, dt)),
                    hit.surface.grid.map(|_| &friction_at as &dyn Fn(Vec3) -> Real),
                )?;
                if !output.in_contact {
                    shear.remove(&w.body);
                }
                (output, Some(sample))
            } else {
                (
                    wheel_ground_wrench(&contact, &terrain, w.tire.as_ref()),
                    None,
                )
            };
            if output.in_contact && pressure.is_none() {
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
                let mobility = response.quadratic(body, wrench)?;
                if mobility > 0.0 {
                    let scale = (-power / (dt * mobility)).clamp(0.0, 1.0);
                    output.force = (normal * output.fz + tangent * scale).to_array();
                    output.aligning_moment *= scale;
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
                ground_point: hit.point,
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

fn shear_poses(
    scene: &RigidScene,
    wheel: &WheelDesc,
    ground: ColliderHandle,
) -> Option<[molla_math::Transform; 3]> {
    Some([
        scene.body_pose(wheel.body).ok()?,
        scene.body_pose(wheel.heading_body).ok()?,
        scene.collider_world_pose(ground).ok()?,
    ])
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
