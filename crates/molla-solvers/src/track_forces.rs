//! Continuous-belt force elements evaluated at collision-refreshed CPU substeps.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector, Vec3};
use molla_sim::runtime::{BodyHandle, BodyKind, ColliderHandle, JointHandle, RigidScene};
use molla_sim::{Contacts, JointType};
use molla_vehicle::track::{TrackMaterial, TrackMotor};

use crate::articulated_response::ArticulatedResponse;

#[derive(Clone, Debug)]
pub struct TrackDesc {
    pub carrier: BodyHandle,
    pub sprocket: BodyHandle,
    pub spin_joint: JointHandle,
    pub contact_colliders: Vec<ColliderHandle>,
    pub local_axle: Vec3,
    pub local_forward: Vec3,
    pub pitch_radius: Real,
    pub material: TrackMaterial,
    pub motor: TrackMotor,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrackSample {
    pub angular_velocity: Real,
    pub travel: Real,
    pub motor_torque: Real,
    pub longitudinal_force: Real,
    pub normal_load: Real,
    pub contacts: usize,
}

#[derive(Clone)]
struct Track {
    desc: TrackDesc,
    speed: Real,
    sample: TrackSample,
}

#[derive(Clone, Default)]
pub struct TrackForces {
    tracks: BTreeMap<BodyHandle, Track>,
}

impl TrackDesc {
    fn validate(&self, scene: &RigidScene) -> Result<()> {
        let joint = scene.joint(self.spin_joint);
        if !self.pitch_radius.is_finite()
            || self.pitch_radius <= 0.0
            || !self.local_axle.is_normalized()
            || !self.local_forward.is_normalized()
            || self.local_axle.dot(self.local_forward).abs() > 1.0e-6
            || !self.material.is_valid()
            || !self.motor.is_valid()
            || self.carrier == self.sprocket
            || self.contact_colliders.is_empty()
            || [self.carrier, self.sprocket].iter().any(|&id| {
                scene
                    .body(id)
                    .is_none_or(|b| !b.enabled || b.kind != BodyKind::Dynamic)
            })
            || joint.is_none_or(|j| {
                j.kind != JointType::Revolute
                    || j.parent != Some(self.carrier)
                    || j.child != self.sprocket
                    || (j.frame_parent.rotation * j.axis)
                        .dot(self.local_axle)
                        .abs()
                        < 1.0 - 1.0e-6
            })
            || scene
                .joint_settings(self.spin_joint)
                .is_none_or(|s| !s.enabled)
            || self.contact_colliders.iter().any(|&id| {
                scene
                    .collider(id)
                    .is_none_or(|c| !c.enabled || c.sensor || c.parent != Some(self.carrier))
            })
        {
            return Err(Error::Build(
                "invalid track descriptor or inactive track handles".into(),
            ));
        }
        Ok(())
    }
}

impl TrackForces {
    pub fn insert(&mut self, scene: &RigidScene, desc: TrackDesc) -> Result<()> {
        desc.validate(scene)?;
        if self.tracks.iter().any(|(&id, t)| {
            id != desc.sprocket
                && desc
                    .contact_colliders
                    .iter()
                    .any(|c| t.desc.contact_colliders.contains(c))
        }) {
            return Err(Error::Build(
                "track contact collider already registered".into(),
            ));
        }
        self.tracks.insert(
            desc.sprocket,
            Track {
                desc,
                speed: 0.0,
                sample: TrackSample::default(),
            },
        );
        Ok(())
    }

    pub fn set_speed(&mut self, sprocket: BodyHandle, speed: Real) -> Result<()> {
        if !speed.is_finite() {
            return Err(Error::Build("non-finite track speed".into()));
        }
        self.tracks
            .get_mut(&sprocket)
            .ok_or_else(|| Error::Build("missing track sprocket".into()))?
            .speed = speed;
        Ok(())
    }

    pub fn sample(&self, scene: &RigidScene, sprocket: BodyHandle) -> Option<TrackSample> {
        let track = self.tracks.get(&sprocket)?;
        track.desc.validate(scene).ok()?;
        Some(track.sample)
    }

    pub fn remove(&mut self, sprocket: BodyHandle) -> bool {
        self.tracks.remove(&sprocket).is_some()
    }

    pub fn clear(&mut self) {
        self.tracks.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    pub(crate) fn prepare(&mut self, scene: &RigidScene) {
        self.tracks
            .retain(|_, track| track.desc.validate(scene).is_ok());
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        response: &mut ArticulatedResponse<'_>,
        contacts: &mut Contacts,
        dt: Real,
        wrenches: &mut [SpatialVector],
    ) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let shapes_a = contacts.a_shape.host()?.to_vec();
        let shapes_b = contacts.b_shape.host()?.to_vec();
        let owners: Vec<_> = (0..contacts.count as usize)
            .map(|i| {
                let pair = [shapes_a[i], shapes_b[i]]
                    .map(|s| scene.collider_handle(molla_core::ShapeId::from_index(s as usize)));
                self.tracks.iter().find_map(|(&id, t)| {
                    let owned =
                        pair.map(|c| c.is_some_and(|c| t.desc.contact_colliders.contains(&c)));
                    if owned[0] == owned[1] {
                        None
                    } else {
                        Some((id, if owned[0] { 0 } else { 1 }, pair))
                    }
                })
            })
            .collect();
        for (i, owner) in owners.iter().enumerate() {
            if owner.is_some() {
                contacts.friction.host_mut()?[i] = 0.0;
                contacts.static_friction.host_mut()?[i] = 0.0;
                contacts.rolling_friction.host_mut()?[i] = 0.0;
                contacts.torsional_friction.host_mut()?[i] = 0.0;
            }
        }
        let state = response.state();
        let model = scene.model();
        let poses = state.body_q.host()?;
        let velocities = state.body_qd.host()?;
        let com = model.body_com.host()?;
        let mut scratch = vec![SpatialVector::ZERO; model.body_count];
        let loads =
            crate::penalty_contact::eval_penalty_contacts(model, state, contacts, &mut scratch)?;
        for (&id, track) in &mut self.tracks {
            let d = &track.desc;
            let carrier = scene.body_index(d.carrier).unwrap().index();
            let rotor = scene.body_index(d.sprocket).unwrap().index();
            let axle = poses[carrier].rotation * d.local_axle;
            let forward = poses[carrier].rotation * d.local_forward;
            let omega = (velocities[rotor].angular - velocities[carrier].angular).dot(axle);
            let motor_pair = [
                (rotor, SpatialVector::new(Vec3::ZERO, axle)),
                (carrier, SpatialVector::new(Vec3::ZERO, -axle)),
            ];
            let motor_response = response_bound(response, &motor_pair)?;
            let error = track.speed / d.pitch_radius - omega;
            let torque = d.motor.torque(track.speed / d.pitch_radius, omega);
            let torque = torque.signum()
                * torque
                    .abs()
                    .min(error.abs() / (dt * motor_response).max(1e-12));
            add_scaled(wrenches, &motor_pair, torque);
            let travel = track.sample.travel + omega * d.pitch_radius * dt;
            track.sample = TrackSample {
                angular_velocity: omega,
                travel,
                motor_torque: torque,
                ..Default::default()
            };
            let count = owners
                .iter()
                .zip(&loads)
                .filter(|(o, l)| o.as_ref().is_some_and(|o| o.0 == id) && l.acted)
                .count()
                .max(1) as Real;
            for (owner, load) in owners.iter().zip(&loads) {
                let Some((owner_id, side, pair)) = owner else {
                    continue;
                };
                if *owner_id != id || !load.acted || load.normal_force <= 0.0 {
                    continue;
                }
                let other = 1 - side;
                let ground = load.bodies[other].map(|b| b.index());
                if ground == Some(carrier) || ground == Some(rotor) {
                    continue;
                }
                let normal = if *side == 0 {
                    -load.normal
                } else {
                    load.normal
                };
                let Some(long) = (forward - normal * forward.dot(normal)).try_normalize() else {
                    continue;
                };
                let lateral = normal.cross(long);
                let arm = load.point - poses[carrier].transform_point(com[carrier]);
                let ground_arm = ground.map(|g| load.point - poses[g].transform_point(com[g]));
                let mut relative =
                    velocities[carrier].linear + velocities[carrier].angular.cross(arm);
                if let Some(g) = ground {
                    relative -=
                        velocities[g].linear + velocities[g].angular.cross(ground_arm.unwrap());
                }
                let mut long_wrench = vec![
                    (
                        carrier,
                        SpatialVector::new(long, arm.cross(long) + axle * d.pitch_radius),
                    ),
                    (
                        rotor,
                        SpatialVector::new(Vec3::ZERO, -axle * d.pitch_radius),
                    ),
                ];
                let mut side_wrench =
                    vec![(carrier, SpatialVector::new(lateral, arm.cross(lateral)))];
                if let Some(g) = ground {
                    long_wrench.push((
                        g,
                        SpatialVector::new(-long, ground_arm.unwrap().cross(-long)),
                    ));
                    side_wrench.push((
                        g,
                        SpatialVector::new(-lateral, ground_arm.unwrap().cross(-lateral)),
                    ));
                }
                let slip = [
                    omega * d.pitch_radius - relative.dot(long),
                    -relative.dot(lateral),
                ];
                let inv = [
                    response_bound(response, &long_wrench)?,
                    response_bound(response, &side_wrench)?,
                ];
                let desired = [0, 1].map(|i| {
                    slip[i]
                        * d.material
                            .slip_damping
                            .min(1.0 / (2.0 * count * dt * inv[i]).max(1e-12))
                });
                let mu = pair[other]
                    .and_then(|c| scene.collider(c))
                    .map_or(0.0, |c| c.friction);
                let [fx, fy] = d.material.limit(desired, load.normal_force, mu);
                add_scaled(wrenches, &long_wrench, fx);
                add_scaled(wrenches, &side_wrench, fy);
                track.sample.longitudinal_force += fx;
                track.sample.normal_load += load.normal_force;
                track.sample.contacts += 1;
            }
        }
        Ok(())
    }
}

fn response_bound(
    response: &mut ArticulatedResponse<'_>,
    forces: &[(usize, SpatialVector)],
) -> Result<Real> {
    let mut bound = 0.0;
    for &(body, force) in forces {
        bound += response.quadratic(body, force)?.max(0.0);
    }
    Ok(bound * forces.len() as Real)
}

fn add_scaled(wrenches: &mut [SpatialVector], forces: &[(usize, SpatialVector)], scale: Real) {
    for &(body, force) in forces {
        wrenches[body].linear += force.linear * scale;
        wrenches[body].angular += force.angular * scale;
    }
}
