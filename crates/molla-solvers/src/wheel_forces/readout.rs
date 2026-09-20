use super::*;

struct BodyWitness {
    body: BodyHandle,
    structure: Option<SceneStamp>,
    state: Option<SceneStamp>,
}

impl BodyWitness {
    fn capture(scene: &RigidScene, body: BodyHandle) -> Self {
        Self {
            body,
            structure: scene.body_structure_stamp(body),
            state: scene.body_state_stamp(body),
        }
    }

    fn matches(&self, scene: &RigidScene) -> bool {
        self.structure.is_some()
            && self.structure == scene.body_structure_stamp(self.body)
            && self.state.is_some()
            && self.state == scene.body_state_stamp(self.body)
    }
}

#[derive(Default)]
pub(super) struct SampleDependencies {
    state: Option<SceneStamp>,
    scene: Option<SceneStamp>,
    revision: u64,
    wheels: BTreeMap<BodyHandle, [BodyWitness; 2]>,
    ground_bodies: Vec<BodyWitness>,
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
        let stamp = scene.stamp();
        if self.scene.as_ref() == Some(&stamp)
            && self.revision == revision
            && self.matches_state(scene)
        {
            return;
        }
        self.state = Some(scene.state_edit_stamp());
        self.scene = Some(stamp);
        self.revision = revision;
        self.wheels.clear();
        self.ground_bodies.clear();
        self.grounds.clear();
        for wheel in wheels.values() {
            self.wheels.insert(
                wheel.body,
                [wheel.body, wheel.heading_body].map(|body| BodyWitness::capture(scene, body)),
            );
        }
        for body in grounds.keys().filter_map(|&id| scene.collider(id)?.parent) {
            self.ground_bodies.push(BodyWitness::capture(scene, body));
        }
        for &id in grounds.keys() {
            self.grounds.push((id, scene.collider_stamp(id)));
        }
    }

    pub(super) fn matches_state(&self, scene: &RigidScene) -> bool {
        self.state == Some(scene.state_edit_stamp())
    }

    pub(super) fn matches_grounds(&self, scene: &RigidScene) -> bool {
        self.ground_bodies.iter().all(|body| body.matches(scene))
            && self
                .grounds
                .iter()
                .all(|(id, stamp)| stamp.is_some() && *stamp == scene.collider_stamp(*id))
    }

    fn matches_wheel(&self, scene: &RigidScene, wheel: BodyHandle) -> bool {
        self.wheels
            .get(&wheel)
            .is_some_and(|bodies| bodies.iter().all(|body| body.matches(scene)))
    }
}

/// Borrowed last-step samples with invalidated machine components excluded.
#[derive(Clone, Copy)]
pub struct WheelSamples<'a> {
    samples: &'a [WheelSample],
    scene: &'a RigidScene,
    dependencies: Option<&'a SampleDependencies>,
}

impl<'a> WheelSamples<'a> {
    pub(super) fn new(
        samples: &'a [WheelSample],
        scene: &'a RigidScene,
        dependencies: Option<&'a SampleDependencies>,
    ) -> Self {
        Self {
            samples,
            scene,
            dependencies,
        }
    }

    pub fn iter(&self) -> WheelSampleIter<'a> {
        WheelSampleIter {
            inner: self.samples.iter(),
            scene: self.scene,
            dependencies: self.dependencies,
        }
    }

    pub fn len(&self) -> usize {
        if self.dependencies.is_none() {
            self.samples.len()
        } else {
            self.iter().count()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.first().is_none()
    }

    pub fn first(&self) -> Option<&'a WheelSample> {
        self.iter().next()
    }

    pub fn last(&self) -> Option<&'a WheelSample> {
        self.iter().next_back()
    }

    pub fn get(&self, index: usize) -> Option<&'a WheelSample> {
        if self.dependencies.is_none() {
            self.samples.get(index)
        } else {
            self.iter().nth(index)
        }
    }
}

impl std::fmt::Debug for WheelSamples<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl std::ops::Index<usize> for WheelSamples<'_> {
    type Output = WheelSample;
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("wheel sample index out of bounds")
    }
}

impl<'a> IntoIterator for WheelSamples<'a> {
    type Item = &'a WheelSample;
    type IntoIter = WheelSampleIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &WheelSamples<'a> {
    type Item = &'a WheelSample;
    type IntoIter = WheelSampleIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Clone)]
pub struct WheelSampleIter<'a> {
    inner: std::slice::Iter<'a, WheelSample>,
    scene: &'a RigidScene,
    dependencies: Option<&'a SampleDependencies>,
}

impl<'a> Iterator for WheelSampleIter<'a> {
    type Item = &'a WheelSample;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.find(|sample| {
            self.dependencies
                .is_none_or(|d| d.matches_wheel(self.scene, sample.wheel))
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.dependencies.is_none() {
            self.inner.size_hint()
        } else {
            (0, Some(self.inner.len()))
        }
    }
}

impl DoubleEndedIterator for WheelSampleIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.rfind(|sample| {
            self.dependencies
                .is_none_or(|d| d.matches_wheel(self.scene, sample.wheel))
        })
    }
}

impl std::iter::FusedIterator for WheelSampleIter<'_> {}
