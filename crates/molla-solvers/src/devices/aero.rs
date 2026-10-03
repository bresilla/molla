//! Aerodynamics: the air bodies move through and the drag they feel.
//!
//! The wind is uniform unless the host gives a body its own (a gust sampled
//! where the body is). Drag is quadratic in the airspeed along each axis of a
//! drag frame on the body and acts at the frame's origin, the centre of
//! pressure, every substep. Propellers take their speed of advance from the
//! airspeed as well.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector, Transform, Vec3};
use molla_sim::State;
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DragId(pub u64);

/// Sea-level air density (kg/m³).
pub const AIR_DENSITY: Real = 1.225;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DragSpec {
    pub body: BodyHandle,
    /// Drag frame in the body frame; its origin is the centre of pressure.
    pub frame: Transform,
    /// Drag coefficient times area along the frame's X, Y and Z (m²).
    pub area: Vec3,
    /// Air density (kg/m³).
    pub density: Real,
}

/// A drag element's airspeed and load over the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DragReading {
    /// Airspeed of the centre of pressure in the drag frame (m/s).
    pub airspeed: Vec3,
    /// Drag force in world axes (N), averaged over the step.
    pub force: Vec3,
    /// The wind at the body (m/s, world).
    pub wind: Vec3,
}

/// Wind and drag of a rigid world.
#[derive(Default)]
pub struct Aero {
    wind: Vec3,
    body_wind: BTreeMap<BodyHandle, Vec3>,
    next: u64,
    drags: BTreeMap<DragId, (DragSpec, DragReading)>,
}

impl Aero {
    /// The wind everywhere without a body wind of its own (m/s, world).
    pub fn set_wind(&mut self, wind: Vec3) -> Result<()> {
        if !wind.is_finite() {
            return Err(Error::Build("wind must be finite".into()));
        }
        self.wind = wind;
        Ok(())
    }

    /// The wind one body flies in, or `None` for the world's.
    pub fn set_body_wind(&mut self, body: BodyHandle, wind: Option<Vec3>) -> Result<()> {
        match wind {
            Some(w) if !w.is_finite() => Err(Error::Build("wind must be finite".into())),
            Some(w) => {
                self.body_wind.insert(body, w);
                Ok(())
            }
            None => {
                self.body_wind.remove(&body);
                Ok(())
            }
        }
    }

    pub fn wind(&self) -> Vec3 {
        self.wind
    }

    pub fn wind_at(&self, body: BodyHandle) -> Vec3 {
        self.body_wind.get(&body).copied().unwrap_or(self.wind)
    }

    pub fn insert(&mut self, scene: &RigidScene, spec: DragSpec) -> Result<DragId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("drag body is missing".into()));
        }
        if !spec.area.is_finite()
            || spec.area.min_element() < 0.0
            || spec.density.is_nan()
            || spec.density < 0.0
        {
            return Err(Error::Build(
                "drag areas and air density must be nonnegative".into(),
            ));
        }
        self.next += 1;
        let id = DragId(self.next);
        self.drags.insert(id, (spec, DragReading::default()));
        Ok(id)
    }

    pub fn remove(&mut self, id: DragId) -> bool {
        self.drags.remove(&id).is_some()
    }

    pub fn reading(&self, id: DragId) -> Option<DragReading> {
        self.drags.get(&id).map(|(_, r)| *r)
    }

    /// Whether any body feels the air.
    pub fn active(&self) -> bool {
        !self.drags.is_empty()
    }

    pub(crate) fn before_step(&mut self) {
        self.body_wind.retain(|_, w| w.is_finite());
        for (_, reading) in self.drags.values_mut() {
            reading.force = Vec3::ZERO;
        }
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        wrenches: &mut [SpatialVector],
        weight: Real,
    ) -> Result<()> {
        if self.drags.is_empty() {
            return Ok(());
        }
        let poses = state.body_q.host()?;
        let velocities = state.body_qd.host()?;
        let centers = scene.model().body_com.host()?;
        let (wind, body_wind) = (self.wind, &self.body_wind);
        for (spec, reading) in self.drags.values_mut() {
            let (Some(body), Some(desc)) = (scene.body_index(spec.body), scene.body(spec.body))
            else {
                continue;
            };
            if !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let i = body.index();
            let frame = poses[i].compose(&spec.frame);
            let arm = frame.position - poses[i].transform_point(centers[i]);
            let air_wind = body_wind.get(&spec.body).copied().unwrap_or(wind);
            let air = velocities[i].linear + velocities[i].angular.cross(arm) - air_wind;
            let local = frame.rotation.conjugate() * air;
            let drag = -0.5 * spec.density * spec.area * local.abs() * local;
            let force = frame.rotation * drag;
            wrenches[i].linear += force;
            wrenches[i].angular += arm.cross(force);
            reading.airspeed = local;
            reading.force += force * weight;
            reading.wind = air_wind;
        }
        Ok(())
    }
}
