use molla_core::Result;
use molla_math::{Real, SpatialVector};
use molla_sim::{BodyFlags, Model, State};

#[cfg(feature = "gpu")]
pub(crate) fn validate_gpu(model: &Model) -> Result<()> {
    for (&flag, acceleration) in model
        .body_flags
        .host()?
        .iter()
        .zip(model.body_kinematic_acceleration.host()?)
    {
        if flag & BodyFlags::KINEMATIC_DRIVEN == 0 {
            continue;
        }
        if acceleration
            .linear
            .to_array()
            .into_iter()
            .chain(acceleration.angular.to_array())
            .any(|value| value != 0.0 && !(value as f32).is_normal())
        {
            return Err(molla_core::Error::Solver(
                "prescribed acceleration is not representable on GPU".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
pub(crate) fn validate_gpu_layout(model: &Model) -> Result<()> {
    validate_gpu(model)?;
    let mut compiled = vec![false; model.body_count];
    for &child in model.joint_child.host()? {
        if let Some(compiled) = compiled.get_mut(child as usize) {
            *compiled = true;
        }
    }
    if model
        .body_flags
        .host()?
        .iter()
        .zip(compiled)
        .any(|(&flag, compiled)| flag & BodyFlags::KINEMATIC_DRIVEN != 0 && !compiled)
    {
        return Err(molla_core::Error::Solver(
            "GPU driven motion requires an articulated body".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu")]
pub(crate) fn driven_ancestors(model: &Model) -> Result<Vec<u32>> {
    let flags = model.body_flags.host()?;
    let mut ancestors: Vec<_> = flags
        .iter()
        .enumerate()
        .map(|(body, flag)| {
            if flag & BodyFlags::KINEMATIC_DRIVEN != 0 {
                body as u32
            } else {
                u32::MAX
            }
        })
        .collect();
    for (&child, &parent) in model
        .joint_child
        .host()?
        .iter()
        .zip(model.joint_parent.host()?)
    {
        let child = child as usize;
        if parent >= 0 && flags[child] & BodyFlags::IMMOVABLE == 0 {
            ancestors[child] = ancestors[parent as usize];
        }
    }
    Ok(ancestors)
}

/// Boundary-induced COM velocity increments at the input configuration.
pub(crate) fn velocity_delta(model: &Model, input: &State, dt: Real) -> Result<Vec<SpatialVector>> {
    let flags = model.body_flags.host()?;
    if !flags.iter().any(|f| f & BodyFlags::KINEMATIC_DRIVEN != 0) {
        return Ok(Vec::new());
    }
    let acceleration = model.body_kinematic_acceleration.host()?;
    let poses = input.body_q.host()?;
    let centers = model.body_com.host()?;
    let mut delta = vec![SpatialVector::ZERO; model.body_count];
    for (body, &flag) in flags.iter().enumerate() {
        if flag & BodyFlags::KINEMATIC_DRIVEN != 0 {
            delta[body] = acceleration[body].scale(dt);
        }
    }
    for (&child, &parent) in model
        .joint_child
        .host()?
        .iter()
        .zip(model.joint_parent.host()?)
    {
        let child = child as usize;
        if parent < 0 || flags[child] & BodyFlags::IMMOVABLE != 0 {
            continue;
        }
        let parent = parent as usize;
        let arm = poses[child].transform_point(centers[child])
            - poses[parent].transform_point(centers[parent]);
        let velocity = delta[parent];
        delta[child] = SpatialVector::new(
            velocity.linear + velocity.angular.cross(arm),
            velocity.angular,
        );
    }
    Ok(delta)
}

pub(crate) fn project(delta: &[SpatialVector], ends: &[(usize, SpatialVector)]) -> Real {
    ends.iter()
        .map(|&(body, wrench)| {
            delta.get(body).map_or(0.0, |v| {
                wrench.linear.dot(v.linear) + wrench.angular.dot(v.angular)
            })
        })
        .sum()
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use molla_math::{Transform, Vec3};
    use molla_sim::{BodyParams, ModelBuilder};

    fn model(value: f64, articulated: bool) -> Model {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(BodyParams::new().with_flags(BodyFlags::KINEMATIC));
        builder
            .set_kinematic_acceleration(body, Some(SpatialVector::new(Vec3::X * value, Vec3::ZERO)))
            .unwrap();
        if articulated {
            builder
                .add_joint_fixed(
                    molla_core::BodyId::NONE,
                    body,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                )
                .unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn gpu_acceleration_rejects_unrepresentable_and_unarticulated_motion() {
        for value in [1e-100, 1e-40, 1e100] {
            assert!(validate_gpu(&model(value, true)).is_err());
        }
        for value in [0., 1e-30, 1., 1e30] {
            assert!(validate_gpu(&model(value, true)).is_ok());
        }
        assert!(validate_gpu_layout(&model(0., false)).is_err());
    }
}
