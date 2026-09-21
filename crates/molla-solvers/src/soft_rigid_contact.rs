//! Two-way CPU contact impulses between FEM nodes and articulated rigid bodies.

use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector, Vec3};
use molla_sim::{Model, State, eval_body_velocities};

mod primitive;
pub use primitive::gather_primitive_contacts;

#[derive(Clone, Copy, Debug)]
pub struct SoftRigidContact {
    pub particle: usize,
    pub body: Option<usize>,
    /// World-space unit normal directed from the rigid surface into the soft body.
    pub normal: Vec3,
    pub separation: Real,
    pub friction: Real,
}

#[derive(Clone, Copy, Debug)]
pub struct SoftRigidContactConfig {
    pub iterations: usize,
    pub penetration_recovery: Real,
    pub max_recovery_speed: Real,
}

impl Default for SoftRigidContactConfig {
    fn default() -> Self {
        Self {
            iterations: 12,
            penetration_recovery: 0.2,
            max_recovery_speed: 0.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SoftRigidContactReport {
    /// World impulses on the soft body, in input contact order.
    pub impulses: Vec<Vec3>,
}

struct Row {
    directions: [Vec3; 3],
    response: [Vec<Real>; 3],
    inverse_mass: [Real; 3],
    tangent_lipschitz: Real,
    lambda: [Real; 3],
    target: Real,
}

fn point_velocity(model: &Model, state: &State, body: Option<usize>, p: Vec3) -> Result<Vec3> {
    let Some(body) = body else {
        return Ok(Vec3::ZERO);
    };
    let pose = state.body_q.host()?[body];
    let com = pose.position + pose.rotation * model.body_com.host()?[body];
    let v = state.body_qd.host()?[body];
    Ok(v.linear + v.angular.cross(p - com))
}

fn unit_response(
    model: &Model,
    state: &State,
    body: Option<usize>,
    p: Vec3,
    direction: Vec3,
) -> Result<(Vec<Real>, Vec3)> {
    let mut response = vec![0.0; model.joint_dof_total];
    let Some(body) = body else {
        return Ok((response, Vec3::ZERO));
    };
    if !model.joint_child.host()?.contains(&(body as i32)) {
        return Ok((response, Vec3::ZERO));
    }
    let mut unit = state.clone_host()?;
    unit.joint_qd.host_mut()?.fill(0.0);
    eval_body_velocities(model, &mut unit)?;
    let before = point_velocity(model, &unit, Some(body), p)?;
    let pose = state.body_q.host()?[body];
    let com = pose.position + pose.rotation * model.body_com.host()?[body];
    crate::rigid_impulse::apply_impulse(
        model,
        &mut unit,
        body,
        SpatialVector::new(direction, (p - com).cross(direction)),
    )?;
    response.copy_from_slice(unit.joint_qd.host()?);
    let delta = point_velocity(model, &unit, Some(body), p)? - before;
    if !delta.is_finite() || delta.dot(direction) < -1e-10 {
        return Err(Error::Solver("invalid soft-rigid mass response".into()));
    }
    Ok((response, delta))
}

fn apply_delta(
    model: &Model,
    rigid: &mut State,
    velocity: &mut Vec3,
    inverse_mass: Real,
    row: &Row,
    delta: [Real; 3],
) -> Result<()> {
    for k in 0..3 {
        *velocity += row.directions[k] * (delta[k] * inverse_mass);
        for (v, response) in rigid.joint_qd.host_mut()?.iter_mut().zip(&row.response[k]) {
            *v -= delta[k] * response;
        }
    }
    eval_body_velocities(model, rigid)
}

/// Solve inelastic Coulomb contact at fixed positions; commit velocities only on success.
pub fn solve_cpu(
    soft_model: &Model,
    soft_state: &mut State,
    rigid_model: &Model,
    rigid_state: &mut State,
    contacts: &[SoftRigidContact],
    dt: Real,
    config: SoftRigidContactConfig,
) -> Result<SoftRigidContactReport> {
    if !dt.is_finite()
        || dt <= 0.0
        || config.iterations == 0
        || !config.penetration_recovery.is_finite()
        || !(0.0..=1.0).contains(&config.penetration_recovery)
        || !config.max_recovery_speed.is_finite()
        || config.max_recovery_speed < 0.0
    {
        return Err(Error::Solver(
            "invalid soft-rigid contact configuration".into(),
        ));
    }
    let positions = soft_state.particle_q.host()?;
    let inverse_mass = soft_model.particle_inv_mass.host()?;
    let mut velocities = soft_state.particle_qd.host()?.to_vec();
    if positions.len() != soft_model.particle_count
        || velocities.len() != positions.len()
        || inverse_mass.len() != positions.len()
        || positions.iter().chain(&velocities).any(|v| !v.is_finite())
        || inverse_mass.iter().any(|v| !v.is_finite() || *v < 0.0)
        || rigid_state.body_q.host()?.len() != rigid_model.body_count
        || rigid_state.body_qd.host()?.len() != rigid_model.body_count
        || rigid_state.joint_qd.host()?.len() != rigid_model.joint_dof_total
        || rigid_state.joint_q.host()?.len() != rigid_model.joint_coord_total
        || rigid_state
            .joint_qd
            .host()?
            .iter()
            .chain(rigid_state.joint_q.host()?)
            .any(|v| !v.is_finite())
        || rigid_state.body_q.host()?.iter().any(|p| {
            !p.position.is_finite()
                || !p.rotation.is_finite()
                || (p.rotation.length_squared() - 1.0).abs() > 1e-6
        })
        || rigid_state
            .body_qd
            .host()?
            .iter()
            .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
    {
        return Err(Error::Solver(
            "invalid soft-rigid state dimensions or values".into(),
        ));
    }
    let mut rigid = rigid_state.clone_host()?;
    let mut rows = Vec::with_capacity(contacts.len());
    for c in contacts {
        if c.particle >= positions.len()
            || c.body.is_some_and(|b| b >= rigid_model.body_count)
            || !c.normal.is_finite()
            || (c.normal.length_squared() - 1.0).abs() > 1e-6
            || !c.separation.is_finite()
            || !c.friction.is_finite()
            || c.friction < 0.0
        {
            return Err(Error::Solver("invalid soft-rigid contact".into()));
        }
        let axis = if c.normal.x.abs() < 0.8 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let tangent = c.normal.cross(axis).normalize();
        let directions = [c.normal, tangent, c.normal.cross(tangent)];
        let mut response: [Vec<Real>; 3] = std::array::from_fn(|_| Vec::new());
        let mut diagonal = [0.0; 3];
        let mut tangent_cross = 0.0;
        for k in 0..3 {
            let (r, w) = unit_response(
                rigid_model,
                &rigid,
                c.body,
                positions[c.particle],
                directions[k],
            )?;
            response[k] = r;
            diagonal[k] = inverse_mass[c.particle] + w.dot(directions[k]).max(0.0);
            if k == 1 {
                tangent_cross = w.dot(directions[2]);
            }
        }
        let target = if c.separation >= 0.0 {
            -c.separation / dt
        } else {
            (-config.penetration_recovery * c.separation / dt).min(config.max_recovery_speed)
        };
        let tangent_lipschitz = 0.5
            * (diagonal[1] + diagonal[2] + (diagonal[1] - diagonal[2]).hypot(2.0 * tangent_cross));
        rows.push(Row {
            directions,
            response,
            inverse_mass: diagonal,
            tangent_lipschitz,
            lambda: [0.0; 3],
            target,
        });
    }
    for _ in 0..config.iterations {
        for (c, row) in contacts.iter().zip(&mut rows) {
            let point = positions[c.particle];
            let relative =
                velocities[c.particle] - point_velocity(rigid_model, &rigid, c.body, point)?;
            let mut next = row.lambda;
            if row.inverse_mass[0] > 0.0 {
                next[0] = (next[0]
                    + (row.target - relative.dot(row.directions[0])) / row.inverse_mass[0])
                    .max(0.0);
            }
            apply_delta(
                rigid_model,
                &mut rigid,
                &mut velocities[c.particle],
                inverse_mass[c.particle],
                row,
                [next[0] - row.lambda[0], 0.0, 0.0],
            )?;
            let relative =
                velocities[c.particle] - point_velocity(rigid_model, &rigid, c.body, point)?;
            for k in 1..3 {
                if row.tangent_lipschitz > 0.0 {
                    next[k] -= relative.dot(row.directions[k]) / row.tangent_lipschitz;
                }
            }
            let tangential = next[1].hypot(next[2]);
            let limit = c.friction * next[0];
            if tangential > limit {
                next[1] *= limit / tangential;
                next[2] *= limit / tangential;
            }
            apply_delta(
                rigid_model,
                &mut rigid,
                &mut velocities[c.particle],
                inverse_mass[c.particle],
                row,
                [0.0, next[1] - row.lambda[1], next[2] - row.lambda[2]],
            )?;
            row.lambda = next;
        }
    }
    if velocities.iter().any(|v| !v.is_finite())
        || rigid.joint_qd.host()?.iter().any(|v| !v.is_finite())
        || rigid
            .body_qd
            .host()?
            .iter()
            .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
    {
        return Err(Error::Solver("soft-rigid contact velocity overflow".into()));
    }
    soft_state
        .particle_qd
        .host_mut()?
        .copy_from_slice(&velocities);
    *rigid_state = rigid;
    Ok(SoftRigidContactReport {
        impulses: rows
            .iter()
            .map(|r| (0..3).map(|k| r.directions[k] * r.lambda[k]).sum())
            .collect(),
    })
}
