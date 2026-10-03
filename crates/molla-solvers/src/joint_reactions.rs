//! Substep and frame-average world joint wrenches at authored endpoint anchors.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector, Vec3};
use molla_sim::{
    State,
    runtime::{BodyHandle, JointHandle, RigidScene},
};

use crate::{
    distance_constraints::ConstraintReadings, loop_forces::LoopReaction,
    penalty_contact::ContactForce,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JointReaction {
    /// Frame-average world load on the child, with torque about its authored anchor.
    pub child: SpatialVector,
    /// Frame-average world load on the parent, with torque about its authored anchor.
    pub parent: SpatialVector,
    pub sample_dt: Real,
    pub peak_force: Real,
    pub peak_torque: Real,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointReactionSample {
    pub joint: JointHandle,
    pub child: SpatialVector,
    pub parent: SpatialVector,
    /// Parent and child anchor positions at the substep's force-evaluation pose.
    pub anchors: [Vec3; 2],
    pub substep: usize,
    /// End time relative to the start of the frame, in seconds.
    pub time_offset: Real,
    pub sample_dt: Real,
}

#[derive(Default)]
pub(crate) struct ReactionFrame {
    pub means: BTreeMap<JointHandle, JointReaction>,
    pub samples: Vec<JointReactionSample>,
}

impl ReactionFrame {
    pub fn add(&mut self, sample: JointReactionSample, frame_dt: Real) -> Result<()> {
        let magnitude = |v: Vec3| v.x.hypot(v.y).hypot(v.z);
        let force = magnitude(sample.child.linear);
        let torque = magnitude(sample.child.angular);
        if !force.is_finite()
            || !torque.is_finite()
            || !sample.parent.linear.is_finite()
            || !sample.parent.angular.is_finite()
        {
            return Err(Error::Solver("non-finite runtime joint reaction".into()));
        }
        let reading = self.means.entry(sample.joint).or_default();
        let weight = sample.sample_dt / frame_dt;
        reading.child = reading.child + sample.child.scale(weight);
        reading.parent = reading.parent + sample.parent.scale(weight);
        reading.sample_dt += sample.sample_dt;
        reading.peak_force = reading.peak_force.max(force);
        reading.peak_torque = reading.peak_torque.max(torque);
        if !reading.child.linear.is_finite()
            || !reading.child.angular.is_finite()
            || !reading.parent.linear.is_finite()
            || !reading.parent.angular.is_finite()
        {
            return Err(Error::Solver(
                "runtime joint reaction average overflow".into(),
            ));
        }
        self.samples.push(sample);
        Ok(())
    }
}

fn opposite_at(wrench: SpatialVector, from: Vec3, to: Vec3) -> SpatialVector {
    SpatialVector::new(
        -wrench.linear,
        -wrench.angular - (from - to).cross(wrench.linear),
    )
}

fn add_load(
    scene: &RigidScene,
    input: &State,
    external: &mut [SpatialVector],
    body: Option<BodyHandle>,
    point: Vec3,
    load: SpatialVector,
) -> Result<()> {
    if let Some(body) = body {
        let index = scene.body_index(body).unwrap().index();
        let pose = input.body_q.host()?[index];
        let center = pose.position + pose.rotation * scene.model().body_com.host()?[index];
        external[index] = external[index]
            + SpatialVector::new(
                load.linear,
                load.angular + (point - center).cross(load.linear),
            );
    }
    Ok(())
}

pub(crate) fn sample_cpu(
    scene: &RigidScene,
    input: &State,
    output: &State,
    dt: Real,
    substep: usize,
    constraints: &ConstraintReadings,
    compliant: &[(JointHandle, LoopReaction)],
    contacts: &[ContactForce],
    selected: Option<&[bool]>,
) -> Result<Vec<JointReactionSample>> {
    let model = scene.model();
    let mut samples = Vec::new();
    let mut external = input.body_f.host()?.to_vec();
    crate::body_wrenches::add_control(scene.control(), &mut external)?;
    let islands = selected
        .map(|_| crate::rigid_islands::RigidIslands::build(scene, []))
        .transpose()?;
    let make_sample = |joint, child, parent, anchors| JointReactionSample {
        joint,
        child,
        parent,
        anchors,
        substep,
        time_offset: (substep + 1) as Real * dt,
        sample_dt: dt,
    };
    for contact in contacts.iter().filter(|contact| contact.acted) {
        for (body, sign) in contact.bodies.into_iter().zip([-1.0, 1.0]) {
            add_load(
                scene,
                input,
                &mut external,
                body.and_then(|body| scene.body_handle(body)),
                contact.point,
                SpatialVector::new(contact.force_on_b * sign, Vec3::ZERO),
            )?;
        }
    }
    for (id, joint) in scene.loop_joints() {
        if !crate::distance_constraints::enabled(scene, id, joint) {
            continue;
        }
        if let (Some(selected), Some(islands)) = (selected, &islands) {
            let group = islands
                .body_island(joint.child)
                .or_else(|| joint.parent.and_then(|body| islands.body_island(body)));
            if group.is_some_and(|group| !group.articulations.iter().any(|&art| selected[art])) {
                continue;
            }
        }
        let anchors = crate::distance_constraints::anchors(scene, input, joint)?;
        if let Some(&rigid) = constraints.rigid.get(&id) {
            let child = rigid
                + SpatialVector::new(
                    constraints
                        .distance
                        .get(&id)
                        .map_or(Vec3::ZERO, |r| r.force),
                    constraints.cones.get(&id).map_or(Vec3::ZERO, |r| r.torque),
                );
            let parent = if joint.kind == molla_sim::JointType::Ball {
                child.scale(-1.0)
            } else {
                opposite_at(child, anchors[1], anchors[0])
            };
            add_load(
                scene,
                input,
                &mut external,
                Some(joint.child),
                anchors[1],
                child,
            )?;
            add_load(
                scene,
                input,
                &mut external,
                joint.parent,
                anchors[0],
                parent,
            )?;
            let spring = compliant
                .iter()
                .find(|(owner, _)| *owner == id)
                .map_or(LoopReaction::default(), |(_, reading)| *reading);
            let spring = SpatialVector::new(spring.force, spring.torque);
            samples.push(make_sample(
                id,
                child + spring,
                parent + opposite_at(spring, anchors[1], anchors[0]),
                anchors,
            ));
        } else if let Some(reading) = constraints.distance.get(&id) {
            let torque = constraints
                .cones
                .get(&id)
                .map_or(Vec3::ZERO, |cone| cone.torque);
            let child = SpatialVector::new(reading.force, torque);
            let parent = child.scale(-1.0);
            add_load(
                scene,
                input,
                &mut external,
                Some(joint.child),
                anchors[1],
                child,
            )?;
            add_load(
                scene,
                input,
                &mut external,
                joint.parent,
                anchors[0],
                parent,
            )?;
            samples.push(make_sample(id, child, parent, anchors));
        } else {
            let reading = compliant
                .iter()
                .find(|(owner, _)| *owner == id)
                .map_or(LoopReaction::default(), |(_, reading)| *reading);
            let cone = constraints
                .cones
                .get(&id)
                .map_or(Vec3::ZERO, |cone| cone.torque);
            let child = SpatialVector::new(reading.force, reading.torque + cone);
            let mut parent = opposite_at(
                SpatialVector::new(reading.force, reading.torque),
                anchors[1],
                anchors[0],
            );
            parent.angular -= cone;
            add_load(
                scene,
                input,
                &mut external,
                Some(joint.child),
                anchors[1],
                SpatialVector::new(Vec3::ZERO, cone),
            )?;
            add_load(
                scene,
                input,
                &mut external,
                joint.parent,
                anchors[0],
                SpatialVector::new(Vec3::ZERO, -cone),
            )?;
            samples.push(make_sample(id, child, parent, anchors));
        }
    }
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let children = model.joint_child.host()?;
    let input_velocity = input.joint_qd.host()?;
    let output_velocity = output.joint_qd.host()?;
    let mut authored_by_articulation = vec![Vec::new(); model.articulation_count];
    for (id, joint) in scene.joints() {
        let Some(index) = scene.joint_index(id).map(|index| index.index()) else {
            continue;
        };
        let art = starts
            .partition_point(|&start| start as usize <= index)
            .saturating_sub(1);
        let authored = authored_by_articulation.get_mut(art).ok_or_else(|| {
            Error::Solver("reaction joint lies outside articulation layout".into())
        })?;
        authored.push((id, joint, index));
    }
    let mut wrench_inputs = None;
    for (art, authored) in authored_by_articulation.into_iter().enumerate() {
        if selected.is_some_and(|selected| !selected[art]) {
            continue;
        }
        let begin = starts[art] as usize;
        let end = starts[art + 1] as usize;
        if authored.is_empty() {
            continue;
        }
        let accelerations: Vec<_> = (offsets[begin] as usize..offsets[end] as usize)
            .map(|dof| (output_velocity[dof] - input_velocity[dof]) / dt)
            .collect();
        if wrench_inputs.is_none() {
            wrench_inputs = Some(crate::rnea::WrenchInputs::new(
                model,
                &external,
                Some(model.body_kinematic_acceleration.host()?),
            )?);
        }
        let result = wrench_inputs
            .as_ref()
            .unwrap()
            .evaluate(input, art, &accelerations)?;
        for (id, joint, index) in authored {
            let anchors = crate::distance_constraints::anchors(scene, input, joint)?;
            let compiled = result.joints[index - begin];
            let (child, parent) =
                if scene.body_index(joint.child).unwrap().index() == children[index] as usize {
                    (compiled, opposite_at(compiled, anchors[1], anchors[0]))
                } else {
                    (opposite_at(compiled, anchors[0], anchors[1]), compiled)
                };
            samples.push(make_sample(id, child, parent, anchors));
        }
    }
    Ok(samples)
}
