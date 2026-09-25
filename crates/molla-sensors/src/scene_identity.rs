use crate::scene::SceneStamp;
use molla_core::{BodyId, Error, Result};
use std::{collections::HashMap, hash::Hash, sync::Arc};

/// Opaque body identity, scoped to one sensor scene and never recycled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SensorBodyHandle {
    scene: u64,
    serial: u64,
}

/// Opaque shape identity, independent of its current dense index or body attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SensorShapeHandle {
    scene: u64,
    serial: u64,
}

/// Entries follow the replacement model's dense order; None creates a new identity.
#[derive(Debug, Clone, Copy)]
pub struct SensorSceneRemap<'a> {
    pub bodies: &'a [Option<SensorBodyHandle>],
    pub shapes: &'a [Option<SensorShapeHandle>],
}

/// Immutable identity table retained independently of scene geometry and GPU resources.
#[derive(Debug, Clone)]
pub struct SceneIdentitySnapshot(Arc<IdentityData>);

#[derive(Debug)]
struct IdentityData {
    stamp: SceneStamp,
    next: u64,
    bodies: Vec<SensorBodyHandle>,
    shapes: Vec<SensorShapeHandle>,
    owners: Vec<i32>,
    body_indices: HashMap<SensorBodyHandle, u32>,
    shape_indices: HashMap<SensorShapeHandle, u32>,
}

impl SceneIdentitySnapshot {
    pub(crate) fn build(
        stamp: SceneStamp,
        body_count: usize,
        shape_bodies: &[i32],
        previous: Option<&Self>,
        remap: Option<SensorSceneRemap<'_>>,
    ) -> Result<Self> {
        if body_count > i32::MAX as usize || shape_bodies.len() > i32::MAX as usize {
            return Err(Error::Build(
                "sensor scene identity capacity exceeded".into(),
            ));
        }
        if previous.is_some_and(|old| {
            old.stamp().scene_id != stamp.scene_id
                || old.stamp().geometry_revision >= stamp.geometry_revision
        }) || remap.is_some() && previous.is_none()
        {
            return Err(Error::Build(
                "sensor scene identity revision mismatch".into(),
            ));
        }
        if shape_bodies
            .iter()
            .any(|&body| body < -1 || body >= 0 && body as usize >= body_count)
        {
            return Err(Error::Build(
                "sensor scene identity body index out of range".into(),
            ));
        }
        if remap.is_some_and(|map| {
            map.bodies.len() != body_count || map.shapes.len() != shape_bodies.len()
        }) {
            return Err(Error::Build(
                "sensor scene remap lengths differ from model".into(),
            ));
        }
        let mut next = previous.map_or(1, |old| old.0.next);
        let (bodies, body_indices) = handles(
            body_count,
            remap.map(|map| map.bodies),
            previous.map(|old| &old.0.body_indices),
            &mut next,
            |serial| SensorBodyHandle {
                scene: stamp.scene_id,
                serial,
            },
        )?;
        let (shapes, shape_indices) = handles(
            shape_bodies.len(),
            remap.map(|map| map.shapes),
            previous.map(|old| &old.0.shape_indices),
            &mut next,
            |serial| SensorShapeHandle {
                scene: stamp.scene_id,
                serial,
            },
        )?;
        let mut owners = Vec::new();
        owners
            .try_reserve_exact(shape_bodies.len())
            .map_err(|_| Error::Build("sensor scene identity allocation failed".into()))?;
        owners.extend_from_slice(shape_bodies);
        Ok(Self(Arc::new(IdentityData {
            stamp,
            next,
            bodies,
            shapes,
            owners,
            body_indices,
            shape_indices,
        })))
    }

    pub fn stamp(&self) -> SceneStamp {
        self.0.stamp
    }

    /// Resolve a hit's dense shape index using its exact scene revision.
    pub fn shape_handle(&self, stamp: SceneStamp, index: u32) -> Result<SensorShapeHandle> {
        self.check_stamp(stamp)?;
        self.0
            .shapes
            .get(index as usize)
            .copied()
            .ok_or_else(|| Error::Solver("sensor shape index out of range".into()))
    }

    /// Resolve a dense body index; the world/static sentinel has no body handle.
    pub fn body_handle(
        &self,
        stamp: SceneStamp,
        index: BodyId,
    ) -> Result<Option<SensorBodyHandle>> {
        self.check_stamp(stamp)?;
        if index == BodyId::NONE {
            return Ok(None);
        }
        self.0
            .bodies
            .get(index.index())
            .copied()
            .map(Some)
            .ok_or_else(|| Error::Solver("sensor body index out of range".into()))
    }

    /// Resolve the shape's body attachment in this snapshot, not in a later revision.
    pub fn shape_body(&self, stamp: SceneStamp, index: u32) -> Result<Option<SensorBodyHandle>> {
        self.check_stamp(stamp)?;
        let &body = self
            .0
            .owners
            .get(index as usize)
            .ok_or_else(|| Error::Solver("sensor shape index out of range".into()))?;
        Ok((body >= 0).then(|| self.0.bodies[body as usize]))
    }

    pub fn shape_index(&self, handle: SensorShapeHandle) -> Result<u32> {
        self.0
            .shape_indices
            .get(&handle)
            .copied()
            .ok_or_else(|| Error::Solver("sensor shape handle absent from snapshot".into()))
    }

    pub fn body_index(&self, handle: SensorBodyHandle) -> Result<BodyId> {
        self.0
            .body_indices
            .get(&handle)
            .copied()
            .map(BodyId)
            .ok_or_else(|| Error::Solver("sensor body handle absent from snapshot".into()))
    }

    fn check_stamp(&self, stamp: SceneStamp) -> Result<()> {
        if stamp != self.0.stamp {
            return Err(Error::Solver(
                "sensor result belongs to another scene revision".into(),
            ));
        }
        Ok(())
    }
}

fn handles<H: Copy + Eq + Hash>(
    count: usize,
    remap: Option<&[Option<H>]>,
    previous: Option<&HashMap<H, u32>>,
    next: &mut u64,
    create: impl Fn(u64) -> H,
) -> Result<(Vec<H>, HashMap<H, u32>)> {
    let mut values = Vec::new();
    let mut indices = HashMap::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Build("sensor scene identity allocation failed".into()))?;
    indices
        .try_reserve(count)
        .map_err(|_| Error::Build("sensor scene identity allocation failed".into()))?;
    for index in 0..count {
        let handle = if let Some(handle) = remap.and_then(|map| map[index]) {
            if previous.is_none_or(|old| !old.contains_key(&handle)) {
                return Err(Error::Build(
                    "sensor scene remap contains foreign or retired handle".into(),
                ));
            }
            handle
        } else {
            let serial = *next;
            *next = next
                .checked_add(1)
                .ok_or_else(|| Error::Build("sensor scene handle identity exhausted".into()))?;
            create(serial)
        };
        if indices.insert(handle, index as u32).is_some() {
            return Err(Error::Build(
                "sensor scene remap contains duplicate handle".into(),
            ));
        }
        values.push(handle);
    }
    Ok((values, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(scene_id: u64, geometry_revision: u64) -> SceneStamp {
        SceneStamp {
            scene_id,
            geometry_revision,
        }
    }

    #[test]
    fn remapping_preserves_handles_and_historical_body_attachments() {
        let old = SceneIdentitySnapshot::build(stamp(1, 1), 2, &[0, 1, -1], None, None).unwrap();
        let bodies = &old.0.bodies;
        let shapes = &old.0.shapes;
        let new = SceneIdentitySnapshot::build(
            stamp(1, 2),
            3,
            &[-1, 0, 2, 1],
            Some(&old),
            Some(SensorSceneRemap {
                bodies: &[Some(bodies[1]), Some(bodies[0]), None],
                shapes: &[Some(shapes[2]), Some(shapes[0]), None, Some(shapes[1])],
            }),
        )
        .unwrap();
        assert_eq!(new.body_index(bodies[0]).unwrap(), BodyId(1));
        assert_eq!(new.body_index(bodies[1]).unwrap(), BodyId(0));
        assert_eq!(new.shape_index(shapes[0]).unwrap(), 1);
        assert_eq!(new.shape_index(shapes[1]).unwrap(), 3);
        assert_eq!(new.shape_body(new.stamp(), 1).unwrap(), Some(bodies[1]));
        assert_eq!(old.shape_body(old.stamp(), 0).unwrap(), Some(bodies[0]));
        assert_eq!(new.shape_body(new.stamp(), 0).unwrap(), None);
        assert_eq!(old.body_handle(old.stamp(), BodyId::NONE).unwrap(), None);
        assert!(old.shape_index(new.0.shapes[2]).is_err());
        assert!(old.body_index(new.0.bodies[2]).is_err());
        assert!(Arc::ptr_eq(&old.0, &old.clone().0));
    }

    #[test]
    fn removed_and_unmapped_handles_never_alias_reused_dense_slots() {
        let old = SceneIdentitySnapshot::build(stamp(8, 1), 1, &[0], None, None).unwrap();
        let empty = SceneIdentitySnapshot::build(stamp(8, 2), 0, &[], Some(&old), None).unwrap();
        let new = SceneIdentitySnapshot::build(stamp(8, 3), 1, &[0], Some(&empty), None).unwrap();
        assert_ne!(old.0.bodies[0], new.0.bodies[0]);
        assert_ne!(old.0.shapes[0], new.0.shapes[0]);
        assert!(new.body_index(old.0.bodies[0]).is_err());
        assert!(new.shape_index(old.0.shapes[0]).is_err());
        let resurrect = SensorSceneRemap {
            bodies: &[Some(old.0.bodies[0])],
            shapes: &[Some(old.0.shapes[0])],
        };
        assert!(
            SceneIdentitySnapshot::build(stamp(8, 4), 1, &[0], Some(&new), Some(resurrect))
                .is_err()
        );
        let replaced =
            SceneIdentitySnapshot::build(stamp(8, 4), 1, &[0], Some(&new), None).unwrap();
        assert_ne!(new.0.bodies[0], replaced.0.bodies[0]);
        assert_ne!(new.0.shapes[0], replaced.0.shapes[0]);
        assert_eq!(old.shape_handle(old.stamp(), 0).unwrap(), old.0.shapes[0]);
    }

    #[test]
    fn invalid_remaps_revisions_and_indices_are_rejected() {
        let old = SceneIdentitySnapshot::build(stamp(10, 1), 2, &[0, 1], None, None).unwrap();
        let foreign = SceneIdentitySnapshot::build(stamp(11, 1), 2, &[0, 1], None, None).unwrap();
        let invalid = [
            SensorSceneRemap {
                bodies: &[],
                shapes: &[None, None],
            },
            SensorSceneRemap {
                bodies: &[None, None],
                shapes: &[],
            },
            SensorSceneRemap {
                bodies: &[Some(old.0.bodies[0]); 2],
                shapes: &[None, None],
            },
            SensorSceneRemap {
                bodies: &[None, None],
                shapes: &[Some(old.0.shapes[0]); 2],
            },
            SensorSceneRemap {
                bodies: &[Some(foreign.0.bodies[0]), None],
                shapes: &[None, None],
            },
            SensorSceneRemap {
                bodies: &[None, None],
                shapes: &[Some(foreign.0.shapes[0]), None],
            },
        ];
        for remap in invalid {
            assert!(
                SceneIdentitySnapshot::build(stamp(10, 2), 2, &[0, 1], Some(&old), Some(remap))
                    .is_err()
            );
        }
        for next_stamp in [old.stamp(), stamp(10, 0), foreign.stamp()] {
            assert!(SceneIdentitySnapshot::build(next_stamp, 0, &[], Some(&old), None).is_err());
        }
        for body in [-2, 2, i32::MAX] {
            assert!(
                SceneIdentitySnapshot::build(stamp(10, 2), 2, &[body], Some(&old), None).is_err()
            );
        }
        assert!(SceneIdentitySnapshot::build(stamp(12, 1), usize::MAX, &[], None, None).is_err());
        assert!(old.shape_handle(stamp(10, 2), 0).is_err());
        assert!(old.shape_body(foreign.stamp(), 0).is_err());
        assert!(old.body_handle(foreign.stamp(), BodyId::NONE).is_err());
        assert!(old.body_handle(old.stamp(), BodyId(2)).is_err());
        assert!(old.shape_handle(old.stamp(), u32::MAX).is_err());
        assert!(old.shape_body(old.stamp(), 2).is_err());
        assert_eq!(old.stamp(), stamp(10, 1));
        assert_eq!(old.0.next, 5);
    }

    #[test]
    fn identity_exhaustion_never_wraps_or_changes_the_current_snapshot() {
        let mut old = SceneIdentitySnapshot::build(stamp(20, 1), 1, &[0], None, None).unwrap();
        Arc::get_mut(&mut old.0).unwrap().next = u64::MAX;
        assert!(SceneIdentitySnapshot::build(stamp(20, 2), 1, &[0], Some(&old), None).is_err());
        let remap = SensorSceneRemap {
            bodies: &[Some(old.0.bodies[0])],
            shapes: &[Some(old.0.shapes[0])],
        };
        let next =
            SceneIdentitySnapshot::build(stamp(20, 2), 1, &[0], Some(&old), Some(remap)).unwrap();
        assert_eq!(next.0.next, u64::MAX);
        assert_eq!(old.stamp(), stamp(20, 1));
        assert_eq!(next.0.bodies, old.0.bodies);
        assert_eq!(next.0.shapes, old.0.shapes);
    }
}
