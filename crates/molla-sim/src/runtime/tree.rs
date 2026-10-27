use super::*;

type Adjacency = BTreeMap<BodyHandle, Vec<(JointHandle, BodyHandle)>>;
type Layout = (
    BTreeMap<JointHandle, JointId>,
    BTreeMap<BodyHandle, JointId>,
);

fn component(graph: &Adjacency, start: BodyHandle) -> BTreeSet<BodyHandle> {
    let mut found = BTreeSet::from([start]);
    let mut pending = vec![start];
    while let Some(body) = pending.pop() {
        for &(_, next) in graph.get(&body).into_iter().flatten() {
            if found.insert(next) {
                pending.push(next);
            }
        }
    }
    found
}

pub(super) fn compile_joints(
    description: &Description,
    bodies: &BTreeMap<BodyHandle, BodyId>,
    poses: &BTreeMap<BodyHandle, Transform>,
    builder: &mut ModelBuilder,
) -> Result<Layout> {
    let mut authored = Adjacency::new();
    let mut active = Adjacency::new();
    let mut anchors = BTreeMap::new();
    let mut authored_anchors = BTreeSet::new();
    let mut authored_children = BTreeSet::new();
    for (&id, joint) in &description.joints {
        dense_body(bodies, joint.parent)?;
        dense_body(bodies, Some(joint.child))?;
        validate_pose(joint.frame_parent)?;
        validate_pose(joint.frame_child)?;
        if joint.kind == JointType::Cable {
            return Err(Error::Build("runtime joint kind is not supported".into()));
        }
        if matches!(joint.kind, JointType::Revolute | JointType::Prismatic)
            && (!joint.axis.length().is_finite() || joint.axis.length() < 1e-9)
        {
            return Err(Error::Build(
                "runtime joint axis must be finite and nonzero".into(),
            ));
        }
        if joint.parent == Some(joint.child) {
            return Err(Error::Build("joint endpoints must differ".into()));
        }
        if description.loop_joints.contains(&id) {
            if joint.kind != JointType::Distance
                && !description.joint_softness.contains_key(&id)
                && description.joint_axes[&id].iter().any(|axis| {
                    (joint.kind == JointType::Ball && axis.limits.is_some())
                        || axis.motor.is_some_and(|motor| {
                            motor.mode != crate::JointTargetMode::None
                                && (!matches!(
                                    joint.kind,
                                    JointType::Revolute
                                        | JointType::Prismatic
                                        | JointType::D6
                                        | JointType::Ball
                                ))
                        })
                        || (axis.brake != 0.0
                            && !matches!(
                                joint.kind,
                                JointType::Revolute
                                    | JointType::Prismatic
                                    | JointType::D6
                                    | JointType::Ball
                            ))
                })
            {
                return Err(Error::Build(
                    "rigid ball loops require cone limits; this rigid-loop control kind is not supported"
                        .into(),
                ));
            }
            continue;
        }
        if description.broken_joints.contains(&id) {
            continue;
        }
        if let Some(parent) = joint.parent {
            if component(&authored, parent).contains(&joint.child) {
                return Err(Error::Build("runtime joint cycle".into()));
            }
            authored.entry(parent).or_default().push((id, joint.child));
            authored.entry(joint.child).or_default().push((id, parent));
        } else if !authored_anchors.insert(joint.child) {
            return Err(Error::Build(
                "multiple world joints constrain one body".into(),
            ));
        }
        if !description.joint_settings[&id].enabled
            || !description.bodies[&joint.child].enabled
            || joint
                .parent
                .is_some_and(|p| !description.bodies[&p].enabled)
        {
            continue;
        }
        if let Some(parent) = joint.parent {
            active.entry(parent).or_default().push((id, joint.child));
            active.entry(joint.child).or_default().push((id, parent));
            authored_children.insert(joint.child);
        } else {
            anchors.insert(joint.child, id);
        }
    }
    let mut checked = BTreeSet::new();
    for &body in bodies.keys() {
        if checked.contains(&body) {
            continue;
        }
        let group = component(&authored, body);
        if group.intersection(&authored_anchors).count() > 1 {
            return Err(Error::Build(
                "runtime joint cycle through world anchors".into(),
            ));
        }
        checked.extend(group);
    }

    let mut visited = BTreeSet::new();
    let mut joints = BTreeMap::new();
    let mut roots = BTreeMap::new();
    for &body in bodies.keys() {
        if visited.contains(&body) {
            continue;
        }
        let group = component(&active, body);
        let pinned: Vec<_> = group
            .iter()
            .copied()
            .filter(|b| {
                anchors.contains_key(b)
                    || (description.bodies[b].enabled
                        && description.bodies[b].kind != BodyKind::Dynamic)
            })
            .collect();
        if pinned.len() > 1 {
            return Err(Error::Build(
                "multiple immovable roots require a constraint loop".into(),
            ));
        }
        let root = pinned.first().copied().unwrap_or_else(|| {
            group
                .iter()
                .copied()
                .find(|b| !authored_children.contains(b))
                .unwrap_or(body)
        });
        if anchors.contains_key(&root) && description.bodies[&root].kind != BodyKind::Dynamic {
            return Err(Error::Build(
                "world joint and immovable body duplicate the root constraint".into(),
            ));
        }
        builder.begin_articulation();
        if let Some(&id) = anchors.get(&root) {
            let j = add_joint(builder, bodies, &lower_joint(description, id))?;
            set_limits(description, id, j, builder)?;
            joints.insert(id, j);
        } else {
            let desc = &description.bodies[&root];
            let j = if desc.enabled && desc.kind == BodyKind::Dynamic {
                builder.add_joint_free(
                    BodyId::NONE,
                    bodies[&root],
                    Transform::IDENTITY,
                    Transform::from_translation(desc.params.com),
                )?
            } else {
                builder.add_joint_fixed(
                    BodyId::NONE,
                    bodies[&root],
                    poses[&root],
                    Transform::IDENTITY,
                )?
            };
            roots.insert(root, j);
        }
        visited.insert(root);
        let mut pending = vec![(root, None)];
        while let Some((body, incoming)) = pending.pop() {
            if let Some((id, parent)) = incoming {
                let mut joint = lower_joint(description, id);
                let reversed = joint.child != body;
                if reversed {
                    joint.parent = Some(parent);
                    joint.child = body;
                    std::mem::swap(&mut joint.frame_parent, &mut joint.frame_child);
                }
                let j = add_joint(builder, bodies, &joint)?;
                builder.set_joint_reversed(j, reversed)?;
                set_limits(description, id, j, builder)?;
                joints.insert(id, j);
            }
            for &(id, next) in active.get(&body).into_iter().flatten().rev() {
                if visited.insert(next) {
                    pending.push((next, Some((id, body))));
                }
            }
        }
    }
    Ok((joints, roots))
}

fn set_limits(
    description: &Description,
    id: JointHandle,
    joint: JointId,
    builder: &mut ModelBuilder,
) -> Result<()> {
    let kind = description.joints[&id].kind;
    let lowered_kind = if description.joint_softness.contains_key(&id) {
        JointType::D6
    } else {
        kind
    };
    let mut speed_caps = vec![Real::INFINITY; lowered_kind.dof_count()];
    for axis in JointAxis::ALL {
        if axis.dof(kind).is_ok() {
            speed_caps[axis.dof(lowered_kind)?] = description.joint_axes[&id][axis as usize]
                .velocity_limit
                .unwrap_or(Real::INFINITY);
        }
    }
    builder.set_joint_velocity_limit(joint, &speed_caps)?;
    if let Some(&softness) = description.joint_softness.get(&id) {
        let mut lower = [Real::NEG_INFINITY; 6];
        let mut upper = [Real::INFINITY; 6];
        for axis in JointAxis::ALL {
            let setting = description.joint_axes[&id][axis as usize];
            let bounds = if axis.dof(kind).is_err() || setting.locked {
                Some([0.0, 0.0])
            } else {
                setting.limits
            };
            if let Some([lo, hi]) = bounds {
                lower[axis as usize] = lo;
                upper[axis as usize] = hi;
            }
        }
        return builder.set_joint_compliance(joint, softness, &lower, &upper);
    }
    let mut lower = vec![Real::NEG_INFINITY; kind.dof_count()];
    let mut upper = vec![Real::INFINITY; lower.len()];
    for axis in JointAxis::ALL {
        if let Ok(dof) = axis.dof(kind) {
            let setting = description.joint_axes[&id][axis as usize];
            if let Some([lo, hi]) = if setting.locked {
                Some([0.0, 0.0])
            } else {
                setting.limits
            } {
                lower[dof] = lo;
                upper[dof] = hi;
            }
        }
    }
    builder.set_joint_limits(joint, &lower, &upper)
}

fn lower_joint(description: &Description, id: JointHandle) -> JointDesc {
    let mut joint = description.joints[&id].clone();
    if description.joint_softness.contains_key(&id) {
        if matches!(joint.kind, JointType::Revolute | JointType::Prismatic) {
            let rotation = Quat::from_rotation_arc(Vec3::X, joint.axis.normalize());
            joint.frame_parent.rotation = (joint.frame_parent.rotation * rotation).normalize();
            joint.frame_child.rotation = (joint.frame_child.rotation * rotation).normalize();
        }
        joint.kind = JointType::D6;
    }
    joint
}
