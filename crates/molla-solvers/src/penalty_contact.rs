//! CPU penalty forces at world-space contact points, applied as COM wrenches.

use molla_core::{BodyId, Error, Result, ShapeId};
use molla_math::{Real, SpatialVector, Vec3};
use molla_sim::{BodyFlags, ContactMaterial, Contacts, Model, ShapeFlags, State};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrictionRegime {
    #[default]
    Inactive,
    Regularized,
    Sticking,
    Sliding,
}

#[derive(Clone, Debug)]
pub struct ContactForce {
    pub bodies: [Option<BodyId>; 2],
    pub colliders: [Option<ShapeId>; 2],
    pub point: Vec3,
    /// Unit normal from the first collider towards the second.
    pub normal: Vec3,
    pub penetration: Real,
    pub friction: Real,
    pub static_friction: Real,
    pub friction_regime: FrictionRegime,
    pub applied_friction: Real,
    pub tangent_impulse: Vec3,
    pub normal_force: Real,
    /// Duration of this solver sample; zero for direct force-only evaluation.
    pub sample_dt: Real,
    /// Normal force integrated over this sample's duration.
    pub normal_impulse: Real,
    /// Total normal and tangential force on the second collider in newtons.
    pub force_on_b: Vec3,
    pub acted: bool,
}

/// Add spring, compression damping and regularized Coulomb friction to body wrenches.
pub fn eval_penalty_contacts(
    model: &Model,
    state: &State,
    contacts: &Contacts,
    forces: &mut [SpatialVector],
) -> Result<Vec<ContactForce>> {
    eval_contacts(model, state, contacts, forces, true)
}

pub(crate) fn eval_normal_contacts(
    model: &Model,
    state: &State,
    contacts: &Contacts,
    forces: &mut [SpatialVector],
) -> Result<Vec<ContactForce>> {
    eval_contacts(model, state, contacts, forces, false)
}

fn eval_contacts(
    model: &Model,
    state: &State,
    contacts: &Contacts,
    forces: &mut [SpatialVector],
    regularized: bool,
) -> Result<Vec<ContactForce>> {
    let n = contacts.count as usize;
    if n > contacts.capacity || forces.len() != model.body_count {
        return Err(Error::Solver(
            "invalid penalty contact buffer dimensions".into(),
        ));
    }
    let a = contacts.a_index.host()?;
    let b = contacts.b_index.host()?;
    let sa = contacts.a_shape.host()?;
    let sb = contacts.b_shape.host()?;
    let point = contacts.point.host()?;
    let normal = contacts.normal.host()?;
    let depth = contacts.depth.host()?;
    let friction = contacts.friction.host()?;
    let static_friction = contacts.static_friction.host()?;
    if [
        a.len(),
        b.len(),
        sa.len(),
        sb.len(),
        point.len(),
        normal.len(),
        depth.len(),
        friction.len(),
        static_friction.len(),
    ]
    .iter()
    .any(|&len| len < n)
    {
        return Err(Error::Solver("truncated penalty contact arrays".into()));
    }
    let poses = state.body_q.host()?;
    let velocities = state.body_qd.host()?;
    let com = model.body_com.host()?;
    let flags = model.body_flags.host()?;
    let shape_flags = model.shape_flags.host()?;
    if [poses.len(), velocities.len(), com.len(), flags.len()]
        .iter()
        .any(|&len| len != model.body_count)
        || shape_flags.len() != model.shape_count
        || model.shape_contact_material.len() != model.shape_count
    {
        return Err(Error::Solver(
            "invalid penalty model/state dimensions".into(),
        ));
    }
    let decode = |index: u32, count: usize| -> Result<Option<usize>> {
        if index == u32::MAX {
            Ok(None)
        } else if (index as usize) < count {
            Ok(Some(index as usize))
        } else {
            Err(Error::Solver("invalid penalty contact endpoint".into()))
        }
    };
    let mut readings = Vec::with_capacity(n);
    for c in 0..n {
        let mu_static = if static_friction[c].is_nan() {
            friction[c]
        } else {
            static_friction[c]
        };
        molla_sim::FrictionCoefficients {
            mu_static,
            mu_dynamic: friction[c],
        }
        .validate()?;
        let bodies = [
            decode(a[c], model.body_count)?,
            decode(b[c], model.body_count)?,
        ];
        let shapes = [
            decode(sa[c], model.shape_count)?,
            decode(sb[c], model.shape_count)?,
        ];
        if !point[c].is_finite()
            || !normal[c].is_finite()
            || normal[c].length_squared() < 1e-20
            || !depth[c].is_finite()
            || !friction[c].is_finite()
            || friction[c] < 0.0
        {
            return Err(Error::Solver(
                "non-finite or invalid penalty contact geometry/material".into(),
            ));
        }
        let material = match shapes {
            [Some(a), Some(b)] => {
                model.shape_contact_material[a].averaged(model.shape_contact_material[b])
            }
            [Some(s), None] | [None, Some(s)] => model.shape_contact_material[s],
            [None, None] => ContactMaterial::default(),
        };
        material.validate()?;
        let normal = normal[c]
            .try_normalize()
            .ok_or_else(|| Error::Solver("invalid contact normal magnitude".into()))?;
        let mut arms = [Vec3::ZERO; 2];
        let mut speeds = [Vec3::ZERO; 2];
        let mut belts = [Vec3::ZERO; 2];
        let mut dynamic = [false; 2];
        for side in 0..2 {
            let belt = shapes[side]
                .and_then(|s| model.shape_surface_velocity.get(s).copied())
                .unwrap_or(Vec3::ZERO);
            belts[side] = belt;
            if let Some(i) = bodies[side] {
                arms[side] = point[c] - (poses[i].position + poses[i].rotation * com[i]);
                if flags[i] & BodyFlags::FIXED == 0 {
                    speeds[side] = velocities[i].linear + velocities[i].angular.cross(arms[side]);
                }
                dynamic[side] = flags[i] & BodyFlags::IMMOVABLE == 0;
                belts[side] = poses[i].rotation * belt;
            }
        }
        let inactive = shapes.iter().flatten().any(|&s| {
            shape_flags[s] & ShapeFlags::SENSOR != 0 || shape_flags[s] & ShapeFlags::COLLIDES == 0
        });
        let active =
            !inactive && bodies[0] != bodies[1] && depth[c] > 0.0 && dynamic.iter().any(|&d| d);
        let mut normal_force = 0.0;
        let mut force = Vec3::ZERO;
        if active {
            let relative = speeds[1] - speeds[0];
            let vn = relative.dot(normal);
            let belt = belts[1] - belts[0];
            let tangent = relative - normal * vn + (belt - normal * belt.dot(normal));
            normal_force = material.stiffness * depth[c] + material.damping * (-vn).max(0.0);
            let speed = tangent.length();
            let eps = material.friction_smoothing;
            let smooth_speed = if speed < eps {
                0.5 * (speed * speed / eps + eps)
            } else {
                speed
            };
            let friction_force = if regularized && smooth_speed > 0.0 {
                tangent
                    * material
                        .friction_stiffness
                        .min(friction[c] * normal_force / smooth_speed)
            } else {
                Vec3::ZERO
            };
            force = normal * normal_force - friction_force;
            if !force.is_finite() {
                return Err(Error::Solver("non-finite penalty force".into()));
            }
        }
        readings.push(ContactForce {
            bodies: bodies.map(|i| i.map(BodyId::from_index)),
            colliders: shapes.map(|i| i.map(ShapeId::from_index)),
            point: point[c],
            normal,
            penetration: depth[c],
            friction: friction[c],
            static_friction: mu_static,
            friction_regime: if active && regularized && friction[c] > 0.0 {
                FrictionRegime::Regularized
            } else {
                FrictionRegime::Inactive
            },
            applied_friction: if active && regularized {
                friction[c]
            } else {
                0.0
            },
            tangent_impulse: Vec3::ZERO,
            normal_force,
            sample_dt: 0.0,
            normal_impulse: 0.0,
            force_on_b: force,
            acted: active && force.length_squared() > 0.0,
        });
    }
    for reading in &readings {
        for side in 0..2 {
            if let Some(body) = reading.bodies[side] {
                let i = body.index();
                if flags[i] & BodyFlags::IMMOVABLE != 0 {
                    continue;
                }
                let force = reading.force_on_b * if side == 0 { -1.0 } else { 1.0 };
                let arm = reading.point - (poses[i].position + poses[i].rotation * com[i]);
                forces[i] = forces[i] + SpatialVector::new(force, arm.cross(force));
            }
        }
    }
    Ok(readings)
}
