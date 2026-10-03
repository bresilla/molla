//! World-momentum transport for unrestricted floating articulations after CPU reaction sampling.

use molla_core::{Error, Result};
use molla_math::{Mat3, Real, SpatialVector, Vec3};
use molla_sim::{BodyFlags, JointType, Model, State, eval_body_velocities};

struct Totals {
    mass: Real,
    center: Vec3,
    momentum: SpatialVector,
    inertia: Mat3,
}

fn totals(model: &Model, state: &State, children: &[i32], origin: Vec3) -> Result<Totals> {
    let poses = state.body_q.host()?;
    let velocities = state.body_qd.host()?;
    let masses = model.body_mass.host()?;
    let coms = model.body_com.host()?;
    let inertias = model.body_inertia.host()?;
    let anchor_body = children[0] as usize;
    let anchor =
        poses[anchor_body].position - origin + poses[anchor_body].rotation * coms[anchor_body];
    let mut mass = 0.0;
    let mut center = Vec3::ZERO;
    let mut momentum = SpatialVector::ZERO;
    for &child in children {
        let body = child as usize;
        let pose = poses[body];
        let com = pose.position - origin + pose.rotation * coms[body];
        let p = velocities[body].linear * masses[body];
        let h = pose.rotation
            * (inertias[body] * (pose.rotation.conjugate() * velocities[body].angular));
        mass += masses[body];
        center += (com - anchor) * masses[body];
        momentum = momentum + SpatialVector::new(p, h + com.cross(p));
    }
    if !mass.is_finite() || mass <= 0.0 {
        return Err(Error::Solver("invalid floating articulation mass".into()));
    }
    center /= mass;
    let mut inertia = Mat3::ZERO;
    for &child in children {
        let body = child as usize;
        let pose = poses[body];
        let offset = (pose.position - origin + pose.rotation * coms[body] - anchor) - center;
        let r = Mat3::from_quat(pose.rotation);
        let outer = Mat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z);
        inertia += r * inertias[body] * r.transpose()
            + (Mat3::IDENTITY * offset.length_squared() - outer) * masses[body];
    }
    Ok(Totals {
        mass,
        center: anchor + center,
        momentum,
        inertia,
    })
}

pub(crate) fn transport(
    model: &Model,
    input: &State,
    output: &mut State,
    dt: Real,
    position_reference: Option<&State>,
    selected: Option<&[bool]>,
) -> Result<()> {
    let count = model
        .articulation_count
        .max(usize::from(model.joint_count > 0));
    if !dt.is_finite() || dt <= 0.0 || selected.is_some_and(|mask| mask.len() != count) {
        return Err(Error::Solver(
            "invalid floating momentum transport input".into(),
        ));
    }
    let starts = model.articulation_start.host()?;
    let children = model.joint_child.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let limits_lo = model.joint_limit_lower.host()?;
    let limits_hi = model.joint_limit_upper.host()?;
    let caps = model.joint_velocity_limit.host()?;
    let flags = model.body_flags.host()?;
    let masses = model.body_mass.host()?;
    let coms = model.body_com.host()?;
    let worlds = model.body_world.host()?;
    let gravity = model.gravity.host()?;
    let configurations = if let Some(reference) = position_reference {
        let mut before = input.clone_host()?;
        let mut after = reference.clone_host()?;
        for state in [&mut before, &mut after] {
            state
                .joint_qd
                .host_mut()?
                .copy_from_slice(output.joint_qd.host()?);
            eval_body_velocities(model, state)?;
        }
        Some((before, after))
    } else {
        None
    };
    let mut changed = vec![false; model.body_count];
    for art in 0..count {
        if selected.is_some_and(|mask| !mask[art]) {
            continue;
        }
        let range = if model.articulation_count == 0 {
            0..model.joint_count
        } else {
            starts[art] as usize..starts[art + 1] as usize
        };
        if range.is_empty() {
            continue;
        }
        let root = range.start;
        let kind = model.joint_type.host()?[root];
        let first = offsets[root] as usize;
        if ![JointType::Free as u32, JointType::D6 as u32].contains(&kind)
            || model.joint_parent.host()?[root] >= 0
            || model.joint_enabled.host()?[root] == 0
            || children[range.clone()]
                .iter()
                .any(|&child| flags[child as usize] & BodyFlags::IMMOVABLE != 0)
            || (first..first + 6).any(|i| {
                limits_lo[i].is_finite() || limits_hi[i].is_finite() || caps[i].is_finite()
            })
        {
            continue;
        }
        let body = children[root] as usize;
        let origin = input.body_q.host()?[body].position;
        let before = totals(model, input, &children[range.clone()], origin)?;
        let mut after = totals(model, output, &children[range.clone()], origin)?;
        let input_rates = input.joint_qd.host()?;
        let output_rates = output.joint_qd.host()?;
        let acceleration: Vec<_> = (first..offsets[range.end] as usize)
            .map(|i| (output_rates[i] - input_rates[i]) / dt)
            .collect();
        let efforts = crate::rnea::inverse_dynamics_response(model, input, art, &acceleration)?;
        let (frame, direction) =
            model.joint_motion_frame(input.body_q.host()?, input.joint_q.host()?, root)?;
        let force = frame.rotation * Vec3::new(efforts[0], efforts[1], efforts[2]) * direction;
        let torque = frame.rotation * Vec3::new(efforts[3], efforts[4], efforts[5]) * direction
            + (frame.position - origin).cross(force);
        let mut derivative = SpatialVector::new(force, torque);
        for &child in &children[range.clone()] {
            let i = child as usize;
            let force = gravity[worlds[i].max(0) as usize] * masses[i];
            let pose = input.body_q.host()?[i];
            let com = pose.position - origin + pose.rotation * coms[i];
            derivative = derivative + SpatialVector::new(force, com.cross(force));
        }
        let mut target = before.momentum + derivative.scale(dt);
        let mut drift_center = before.center;
        if let Some((original, projected)) = &configurations {
            let original = totals(model, original, &children[range.clone()], origin)?;
            let projected = totals(model, projected, &children[range.clone()], origin)?;
            target = target + (projected.momentum - original.momentum);
            drift_center = projected.center;
        }
        let shift = drift_center + target.linear * (dt / after.mass) - after.center;
        let (frame, direction) =
            model.joint_motion_frame(output.body_q.host()?, output.joint_q.host()?, root)?;
        let coordinate_shift = frame.rotation.conjugate() * shift * direction;
        let coordinate = model.joint_coord_offset.host()?[root] as usize;
        for axis in 0..3 {
            output.joint_q.host_mut()?[coordinate + axis] += coordinate_shift[axis];
        }
        for &child in &children[range] {
            output.body_q.host_mut()?[child as usize].position += shift;
            changed[child as usize] = true;
        }
        after.center += shift;
        after.momentum.angular += shift.cross(after.momentum.linear);
        let delta = target - after.momentum;
        let angular = delta.angular - after.center.cross(delta.linear);
        let solved = crate::rigid_impulse::solve_mass(
            &after.inertia.to_cols_array(),
            &angular.to_array(),
            3,
        )?;
        let spin = Vec3::from_slice(&solved);
        let residual = after.inertia * spin - angular;
        let scale = target.angular.length()
            + after.momentum.angular.length()
            + after.center.length() * (target.linear.length() + after.momentum.linear.length());
        if !residual.is_finite() || residual.length() > 1e-10 * scale {
            return Err(Error::Solver(format!(
                "floating momentum lies outside inertia range: residual={:e}, scale={scale:e}, angular={angular:?}, inertia={:?}",
                residual.length(),
                after.inertia,
            )));
        }
        let (frame, direction) =
            model.joint_motion_frame(output.body_q.host()?, output.joint_q.host()?, root)?;
        let translation =
            delta.linear / after.mass + spin.cross(frame.position - origin - after.center);
        let linear = frame.rotation.conjugate() * translation * direction;
        let angular = frame.rotation.conjugate() * spin * direction;
        let qd = output.joint_qd.host_mut()?;
        for axis in 0..3 {
            qd[first + axis] += linear[axis];
            qd[first + 3 + axis] += angular[axis];
        }
        if qd[first..first + 6].iter().any(|value| !value.is_finite()) {
            return Err(Error::Solver("floating momentum transport overflow".into()));
        }
    }
    if changed.iter().any(|&active| active) {
        let retained = output.body_qd.host()?.to_vec();
        eval_body_velocities(model, output)?;
        for (i, &active) in changed.iter().enumerate() {
            if !active {
                output.body_qd.host_mut()?[i] = retained[i];
            }
        }
    }
    Ok(())
}
