use super::*;

impl RigidScene {
    /// Structural identity of a body's joint-connected component across scene rebuilds.
    pub fn body_structure_stamp(&self, body: BodyHandle) -> Option<SceneStamp> {
        self.structure_stamps.get(&body).cloned()
    }

    /// Explicit pose/velocity edits and mutable state access; solver commits retain it.
    pub fn state_edit_stamp(&self) -> SceneStamp {
        self.state_edit_stamp.clone()
    }

    /// Explicit state edits to a body's joint component; raw mutable access affects all bodies.
    pub fn body_state_stamp(&self, body: BodyHandle) -> Option<SceneStamp> {
        self.body(body)?;
        Some(
            self.body_state_stamps
                .get(&body)
                .unwrap_or(&self.state_base_stamp)
                .clone(),
        )
    }

    pub(super) fn mark_body_state_edits(&mut self, changed: &[BodyHandle]) {
        let mut neighbours = BTreeMap::<BodyHandle, Vec<BodyHandle>>::new();
        for joint in self.description.joints.values() {
            if let Some(parent) = joint.parent {
                neighbours.entry(parent).or_default().push(joint.child);
                neighbours.entry(joint.child).or_default().push(parent);
            }
        }
        let mut visited: BTreeSet<_> = changed.iter().copied().collect();
        let mut pending = changed.to_vec();
        while let Some(body) = pending.pop() {
            self.body_state_stamps
                .insert(body, self.state_edit_stamp.clone());
            for &neighbour in neighbours.get(&body).into_iter().flatten() {
                if visited.insert(neighbour) {
                    pending.push(neighbour);
                }
            }
        }
    }
}

pub(super) fn update_structure_stamps(
    stamps: &mut BTreeMap<BodyHandle, SceneStamp>,
    old: &Description,
    next: &Description,
) {
    let mut changed = BTreeSet::new();
    for &body in old.bodies.keys().chain(next.bodies.keys()) {
        if old.bodies.get(&body) != next.bodies.get(&body) {
            changed.insert(body);
        }
    }
    for &id in old.colliders.keys().chain(next.colliders.keys()) {
        let before = old.colliders.get(&id);
        let after = next.colliders.get(&id);
        if old.collider_stamps.get(&id) != next.collider_stamps.get(&id)
            || before.map(|c| c.mass) != after.map(|c| c.mass)
        {
            changed.extend(before.and_then(|c| c.parent));
            changed.extend(after.and_then(|c| c.parent));
        }
    }
    let mut neighbours = BTreeMap::<BodyHandle, Vec<BodyHandle>>::new();
    for &id in old.joints.keys().chain(next.joints.keys()) {
        let before = old.joints.get(&id);
        let after = next.joints.get(&id);
        let edited = before != after
            || old.joint_settings.get(&id) != next.joint_settings.get(&id)
            || old.joint_axes.get(&id) != next.joint_axes.get(&id)
            || old.joint_softness.get(&id) != next.joint_softness.get(&id)
            || old.loop_joints.contains(&id) != next.loop_joints.contains(&id);
        for joint in before.into_iter().chain(after) {
            if edited {
                changed.insert(joint.child);
                changed.extend(joint.parent);
            }
            if let Some(parent) = joint.parent {
                neighbours.entry(parent).or_default().push(joint.child);
                neighbours.entry(joint.child).or_default().push(parent);
            }
        }
    }
    let mut pending: Vec<_> = changed.iter().copied().collect();
    while let Some(body) = pending.pop() {
        for &neighbour in neighbours.get(&body).into_iter().flatten() {
            if changed.insert(neighbour) {
                pending.push(neighbour);
            }
        }
    }
    stamps.retain(|body, _| next.bodies.contains_key(body));
    let stamp = SceneStamp(Arc::new(()));
    for body in changed {
        if next.bodies.contains_key(&body) {
            stamps.insert(body, stamp.clone());
        }
    }
}
