use super::*;

impl RigidScene {
    /// Identity of a body's insertion; forks and subsequent edits retain it.
    pub fn body_identity_stamp(&self, body: BodyHandle) -> Option<SceneStamp> {
        self.description.body_identities.get(&body).cloned()
    }

    /// Identity of a joint's insertion; forks, repair and configuration edits retain it.
    pub fn joint_identity_stamp(&self, joint: JointHandle) -> Option<SceneStamp> {
        self.description.joint_identities.get(&joint).cloned()
    }

    /// Structural identity of a body's joint-connected component across scene rebuilds.
    pub fn body_structure_stamp(&self, body: BodyHandle) -> Option<SceneStamp> {
        self.structure_stamps.get(&body).cloned()
    }

    /// Contact response identity; force-only compliant-loop edits retain it.
    pub fn body_contact_stamp(&self, body: BodyHandle) -> Option<SceneStamp> {
        self.contact_stamps.get(&body).cloned()
    }

    /// Contact onset identity; joint-response-only edits retain it.
    pub fn body_contact_episode_stamp(&self, body: BodyHandle) -> Option<SceneStamp> {
        self.contact_episode_stamps.get(&body).cloned()
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
        for (&id, joint) in &self.description.joints {
            if !self.description.broken_joints.contains(&id)
                && let Some(parent) = joint.parent
            {
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

pub(super) fn update_structure_stamps<const CONTACTS: bool, const EPISODES: bool>(
    stamps: &mut BTreeMap<BodyHandle, SceneStamp>,
    old: &Description,
    next: &Description,
) {
    let mut changed = BTreeSet::new();
    for &(a, b) in old
        .body_pair_exclusions
        .symmetric_difference(&next.body_pair_exclusions)
    {
        changed.extend([a, b]);
    }
    for &(a, b) in old
        .collider_pair_exclusions
        .symmetric_difference(&next.collider_pair_exclusions)
    {
        for id in [a, b] {
            changed.extend(old.colliders.get(&id).and_then(|c| c.parent));
            changed.extend(next.colliders.get(&id).and_then(|c| c.parent));
        }
    }
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
            || old.distance_limits.get(&id) != next.distance_limits.get(&id)
            || old.cone_limits.get(&id) != next.cone_limits.get(&id)
            || old.loop_joints.contains(&id) != next.loop_joints.contains(&id)
            || old.broken_joints.contains(&id) != next.broken_joints.contains(&id);
        let force_only = CONTACTS
            && old.loop_joints.contains(&id)
            && next.loop_joints.contains(&id)
            && old.joint_softness.contains_key(&id)
            && next.joint_softness.contains_key(&id)
            && !old.cone_limits.contains_key(&id)
            && !next.cone_limits.contains_key(&id)
            && before.map(|joint| (joint.parent, joint.child))
                == after.map(|joint| (joint.parent, joint.child))
            && old.joint_settings.get(&id).map(|s| s.contacts_enabled)
                == next.joint_settings.get(&id).map(|s| s.contacts_enabled)
            && old.broken_joints.contains(&id) == next.broken_joints.contains(&id);
        for joint in before.into_iter().chain(after) {
            if edited && !force_only && !EPISODES {
                changed.insert(joint.child);
                changed.extend(joint.parent);
            }
            if (!old.broken_joints.contains(&id) || !next.broken_joints.contains(&id))
                && let Some(parent) = joint.parent
            {
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
