use super::*;

#[derive(Default)]
pub(super) struct SampleDependencies {
    state: Option<SceneStamp>,
    scene: Option<SceneStamp>,
    revision: u64,
    bodies: Vec<(BodyHandle, Option<SceneStamp>)>,
    grounds: Vec<(ColliderHandle, Option<SceneStamp>)>,
}

impl SampleDependencies {
    pub(super) fn capture(
        &mut self,
        scene: &RigidScene,
        wheels: &BTreeMap<BodyHandle, WheelDesc>,
        grounds: &BTreeMap<ColliderHandle, GroundMaterial>,
        revision: u64,
    ) {
        self.state = Some(scene.state_edit_stamp());
        let stamp = scene.stamp();
        if self.scene.as_ref() == Some(&stamp) && self.revision == revision {
            return;
        }
        self.scene = Some(stamp);
        self.revision = revision;
        self.bodies.clear();
        self.grounds.clear();
        for body in wheels
            .values()
            .flat_map(|w| [w.body, w.heading_body])
            .chain(grounds.keys().filter_map(|&id| scene.collider(id)?.parent))
        {
            self.bodies.push((body, scene.body_structure_stamp(body)));
        }
        for &id in grounds.keys() {
            self.grounds.push((id, scene.collider_stamp(id)));
        }
    }

    pub(super) fn matches_state(&self, scene: &RigidScene) -> bool {
        self.state == Some(scene.state_edit_stamp())
    }

    pub(super) fn matches_structure(&self, scene: &RigidScene) -> bool {
        self.bodies
            .iter()
            .all(|(body, stamp)| stamp.is_some() && *stamp == scene.body_structure_stamp(*body))
            && self
                .grounds
                .iter()
                .all(|(id, stamp)| stamp.is_some() && *stamp == scene.collider_stamp(*id))
    }
}
