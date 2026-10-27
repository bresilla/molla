//! Connectors: docking faces that link two bodies with a breakable weld when
//! a compatible peer lines up (Webots `Connector`).

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use molla_core::{Error, Result};
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sim::JointType;
use molla_sim::joint_control::JointSoftness;
use molla_sim::runtime::{BodyHandle, JointDesc, JointHandle, RigidScene};

use crate::loop_forces::LoopReaction;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConnectorId(pub u64);

/// Which peers a connector docks with: symmetric with symmetric, active
/// with passive. A passive connector never latches itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ConnectorKind {
    #[default]
    Symmetric,
    Active,
    Passive,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConnectorSpec {
    /// Body carrying the connector; `None` fixes it to the world.
    pub body: Option<BodyHandle>,
    /// Docking frame in the body frame (world for `None`); +X points out of
    /// the docking face.
    pub frame: Transform,
    /// Only connectors of the same model dock.
    pub model: String,
    pub kind: ConnectorKind,
    /// Link as soon as a peer lines up while latched.
    pub auto_lock: bool,
    /// Latching this side alone links.
    pub unilateral_lock: bool,
    /// Unlatching this side alone unlinks.
    pub unilateral_unlock: bool,
    /// Largest origin distance (m), +X misalignment and docking-rotation
    /// error (rad) this connector accepts.
    pub distance_tolerance: Real,
    pub axis_tolerance: Real,
    pub rotation_tolerance: Real,
    /// Docking rotations about +X per turn; zero leaves the rotation free.
    pub rotations: u32,
    /// Pull the faces flush at the nearest docking rotation when linking.
    pub snap: bool,
    /// Force pulling the faces apart and sliding them (N) that breaks the
    /// link; `None` never breaks.
    pub tensile_strength: Option<Real>,
    pub shear_strength: Option<Real>,
    /// Natural frequency (Hz) of the compliant link.
    pub stiffness: Real,
}

impl Default for ConnectorSpec {
    fn default() -> Self {
        Self {
            body: None,
            frame: Transform::IDENTITY,
            model: String::new(),
            kind: ConnectorKind::Symmetric,
            auto_lock: false,
            unilateral_lock: true,
            unilateral_unlock: true,
            distance_tolerance: 0.01,
            axis_tolerance: 0.2,
            rotation_tolerance: 0.2,
            rotations: 4,
            snap: true,
            tensile_strength: None,
            shear_strength: None,
            stiffness: 30.0,
        }
    }
}

/// A connector's state after the last update.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConnectorReading {
    /// A compatible peer lined up within this connector's tolerances.
    pub presence: Option<ConnectorId>,
    /// The latch (Webots `isLocked`).
    pub locked: bool,
    /// The connector it is linked to.
    pub linked: Option<ConnectorId>,
    /// Link load over the last step (N): pulling the faces apart along the
    /// docking axis, and sliding them across it.
    pub tensile: Real,
    pub shear: Real,
}

struct Connector {
    spec: ConnectorSpec,
    locked: bool,
    /// Latched since the last update: links to a peer already present.
    just_locked: bool,
    /// Unlatched since the last update: may release its link.
    just_unlocked: bool,
    /// Peer this connector broke from; no relink until it leaves.
    broke_with: Option<ConnectorId>,
    reading: ConnectorReading,
}

struct Link {
    parent: ConnectorId,
    child: ConnectorId,
}

/// Connectors of a rigid world and the links between them.
#[derive(Default)]
pub struct Connectors {
    next: u64,
    connectors: BTreeMap<ConnectorId, Connector>,
    links: BTreeMap<JointHandle, Link>,
}

/// Rotation about +X of `relative` (radians in (−π, π]).
fn twist(relative: Quat) -> Real {
    let y = relative * Vec3::Y;
    y.z.atan2(y.y)
}

/// Turn about +X that brings a face onto its peer's.
const FLIP: Quat = Quat::from_xyzw(0.0, 0.0, 1.0, 0.0);

fn nearest_docking(angle: Real, rotations: u32) -> Real {
    if rotations == 0 {
        return angle;
    }
    let step = TAU / rotations as Real;
    (angle / step).round() * step
}

impl Connectors {
    pub(super) fn is_empty(&self) -> bool {
        self.connectors.is_empty() && self.links.is_empty()
    }

    pub fn insert(&mut self, scene: &RigidScene, spec: ConnectorSpec) -> Result<ConnectorId> {
        if spec.body.is_some_and(|b| scene.body(b).is_none()) {
            return Err(Error::Build("connector body is missing".into()));
        }
        let tolerances = [
            spec.distance_tolerance,
            spec.axis_tolerance,
            spec.rotation_tolerance,
        ];
        if tolerances.iter().any(|t| t.is_nan() || *t < 0.0)
            || spec.stiffness.is_nan()
            || spec.stiffness <= 0.0
            || [spec.tensile_strength, spec.shear_strength]
                .iter()
                .flatten()
                .any(|s| s.is_nan() || *s < 0.0)
        {
            return Err(Error::Build(
                "connector tolerances, strengths and stiffness must be nonnegative".into(),
            ));
        }
        self.next += 1;
        let id = ConnectorId(self.next);
        self.connectors.insert(
            id,
            Connector {
                spec,
                locked: false,
                just_locked: false,
                just_unlocked: false,
                broke_with: None,
                reading: ConnectorReading::default(),
            },
        );
        Ok(id)
    }

    /// Removes a connector, unlinking it first.
    pub fn remove(&mut self, scene: &mut RigidScene, id: ConnectorId) -> Result<bool> {
        let links: Vec<JointHandle> = self
            .links
            .iter()
            .filter(|(_, l)| l.parent == id || l.child == id)
            .map(|(&j, _)| j)
            .collect();
        for joint in links {
            self.unlink(scene, joint)?;
        }
        Ok(self.connectors.remove(&id).is_some())
    }

    fn connector(&mut self, id: ConnectorId) -> Result<&mut Connector> {
        self.connectors
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such connector".into()))
    }

    /// Latches the connector; it links to a peer present at the next step,
    /// or to one arriving later when it auto-locks. Passive ones ignore it.
    pub fn lock(&mut self, id: ConnectorId) -> Result<()> {
        let connector = self.connector(id)?;
        if connector.spec.kind != ConnectorKind::Passive {
            connector.locked = true;
            connector.just_locked = true;
        }
        Ok(())
    }

    /// Unlatches the connector; its link breaks at the next step as the
    /// unilateral-unlock rules allow.
    pub fn unlock(&mut self, id: ConnectorId) -> Result<()> {
        let connector = self.connector(id)?;
        if connector.spec.kind != ConnectorKind::Passive {
            connector.just_unlocked = connector.locked;
            connector.locked = false;
            connector.just_locked = false;
        }
        Ok(())
    }

    pub fn reading(&self, id: ConnectorId) -> Option<ConnectorReading> {
        self.connectors.get(&id).map(|c| c.reading)
    }

    /// The joint linking `id`, if any.
    pub fn link_joint(&self, id: ConnectorId) -> Option<JointHandle> {
        self.links
            .iter()
            .find(|(_, l)| l.parent == id || l.child == id)
            .map(|(&j, _)| j)
    }

    fn world_frame(scene: &RigidScene, spec: &ConnectorSpec) -> Result<Option<Transform>> {
        match spec.body {
            None => Ok(Some(spec.frame)),
            Some(body) if scene.body(body).is_some() => {
                Ok(Some(scene.body_pose(body)?.compose(&spec.frame)))
            }
            Some(_) => Ok(None),
        }
    }

    /// Whether `a` sees `b` lined up within `a`'s tolerances.
    fn aligned(a: &ConnectorSpec, pa: &Transform, pb: &Transform) -> bool {
        let x_a = pa.rotation * Vec3::X;
        let x_b = pb.rotation * Vec3::X;
        if pa.position.distance(pb.position) > a.distance_tolerance
            || x_a.angle_between(-x_b) > a.axis_tolerance
        {
            return false;
        }
        if a.rotations == 0 {
            return true;
        }
        let angle = twist((pb.rotation * FLIP).conjugate() * pa.rotation);
        (angle - nearest_docking(angle, a.rotations)).abs() <= a.rotation_tolerance
    }

    fn compatible(a: &ConnectorSpec, b: &ConnectorSpec) -> bool {
        use ConnectorKind::*;
        a.model == b.model
            && matches!(
                (a.kind, b.kind),
                (Symmetric, Symmetric) | (Active, Passive) | (Passive, Active)
            )
            && (a.body != b.body || a.body.is_none() && b.body.is_none())
            && (a.body.is_some() || b.body.is_some())
    }

    fn unlink(&mut self, scene: &mut RigidScene, joint: JointHandle) -> Result<()> {
        if self.links.remove(&joint).is_some() && scene.joint(joint).is_some() {
            scene.remove_joint(joint)?;
        }
        Ok(())
    }

    /// Updates presence, breaks overloaded or unlatched links and makes new
    /// ones, from the loop-joint `reactions` of the last step.
    pub(crate) fn update(
        &mut self,
        scene: &mut RigidScene,
        reactions: &BTreeMap<JointHandle, LoopReaction>,
    ) -> Result<()> {
        if self.connectors.is_empty() {
            return Ok(());
        }
        let stale: Vec<JointHandle> = self
            .links
            .keys()
            .filter(|&&j| scene.joint(j).is_none())
            .copied()
            .collect();
        for joint in stale {
            self.links.remove(&joint);
        }
        let mut frames = BTreeMap::new();
        for (&id, c) in &self.connectors {
            if let Some(frame) = Self::world_frame(scene, &c.spec)? {
                frames.insert(id, frame);
            }
        }
        let linked: BTreeMap<ConnectorId, ConnectorId> = self
            .links
            .values()
            .flat_map(|l| [(l.parent, l.child), (l.child, l.parent)])
            .collect();
        for c in self.connectors.values_mut() {
            c.reading = ConnectorReading {
                locked: c.locked,
                ..ConnectorReading::default()
            };
        }
        let ids: Vec<ConnectorId> = frames.keys().copied().collect();
        for &a in &ids {
            let spec_a = &self.connectors[&a].spec;
            let presence = ids
                .iter()
                .filter(|&&b| b != a && linked.get(&b).is_none_or(|&peer| peer == a))
                .filter(|&&b| {
                    let spec_b = &self.connectors[&b].spec;
                    Self::compatible(spec_a, spec_b)
                        && Self::aligned(spec_a, &frames[&a], &frames[&b])
                })
                .min_by(|&&b, &&c| {
                    frames[&a]
                        .position
                        .distance(frames[&b].position)
                        .total_cmp(&frames[&a].position.distance(frames[&c].position))
                })
                .copied();
            let connector = self.connectors.get_mut(&a).expect("listed above");
            connector.reading.presence = presence;
            if connector.broke_with.is_some() && connector.broke_with != presence {
                connector.broke_with = None;
            }
        }
        self.break_links(scene, reactions, &frames)?;
        self.make_links(scene, &frames)?;
        for c in self.connectors.values_mut() {
            c.just_locked = false;
            c.just_unlocked = false;
        }
        for link in self.links.values() {
            for (me, peer) in [(link.parent, link.child), (link.child, link.parent)] {
                if let Some(c) = self.connectors.get_mut(&me) {
                    c.reading.linked = Some(peer);
                }
            }
        }
        Ok(())
    }

    fn break_links(
        &mut self,
        scene: &mut RigidScene,
        reactions: &BTreeMap<JointHandle, LoopReaction>,
        frames: &BTreeMap<ConnectorId, Transform>,
    ) -> Result<()> {
        let mut broken = Vec::new();
        for (&joint, link) in &self.links {
            let (parent, child) = (
                &self.connectors[&link.parent],
                &self.connectors[&link.child],
            );
            let latched = |c: &Connector| c.spec.kind != ConnectorKind::Passive && c.locked;
            let released = |me: &Connector, other: &Connector| {
                me.just_unlocked && (me.spec.unilateral_unlock || !latched(other))
            };
            let mut breaks = released(parent, child) || released(child, parent);
            if let (Some(reaction), Some(frame)) = (reactions.get(&joint), frames.get(&link.parent))
            {
                let axis = frame.rotation * Vec3::X;
                let tensile = -reaction.force.dot(axis);
                let shear = (reaction.force - axis * reaction.force.dot(axis)).length();
                let strength = |pick: fn(&ConnectorSpec) -> Option<Real>| -> Real {
                    let sides: Vec<&Connector> =
                        [parent, child].into_iter().filter(|c| latched(c)).collect();
                    if sides.is_empty() || sides.iter().any(|c| pick(&c.spec).is_none()) {
                        Real::INFINITY
                    } else {
                        sides.iter().filter_map(|c| pick(&c.spec)).sum()
                    }
                };
                breaks |= tensile > strength(|s| s.tensile_strength)
                    || shear > strength(|s| s.shear_strength);
                broken.push((joint, breaks, tensile.max(0.0), shear));
            } else {
                broken.push((joint, breaks, 0.0, 0.0));
            }
        }
        for (joint, breaks, tensile, shear) in broken {
            let (parent, child) = {
                let link = &self.links[&joint];
                (link.parent, link.child)
            };
            for id in [parent, child] {
                let c = self.connectors.get_mut(&id).expect("linked connector");
                c.reading.tensile = tensile;
                c.reading.shear = shear;
            }
            if breaks {
                self.unlink(scene, joint)?;
                self.connectors.get_mut(&parent).expect("linked").broke_with = Some(child);
                self.connectors.get_mut(&child).expect("linked").broke_with = Some(parent);
            }
        }
        Ok(())
    }

    fn make_links(
        &mut self,
        scene: &mut RigidScene,
        frames: &BTreeMap<ConnectorId, Transform>,
    ) -> Result<()> {
        let ids: Vec<ConnectorId> = self.connectors.keys().copied().collect();
        for a in ids {
            let c = &self.connectors[&a];
            let wants = c.locked && (c.spec.auto_lock || c.just_locked);
            let Some(b) = c.reading.presence else {
                continue;
            };
            let busy =
                |id: ConnectorId| self.links.values().any(|l| l.parent == id || l.child == id);
            let (ca, cb) = (&self.connectors[&a], &self.connectors[&b]);
            if !wants
                || busy(a)
                || busy(b)
                || ca.broke_with == Some(b)
                || !(ca.spec.unilateral_lock || cb.locked || cb.spec.kind == ConnectorKind::Passive)
            {
                continue;
            }
            let (parent, child) = if ca.spec.body.is_some() && cb.spec.body.is_none() {
                (b, a)
            } else {
                (a, b)
            };
            let (pspec, cspec) = (
                &self.connectors[&parent].spec,
                &self.connectors[&child].spec,
            );
            let child_body = cspec.body.expect("compatible pairs have a body");
            let frame_child = if pspec.snap || cspec.snap {
                let relative =
                    (frames[&child].rotation * FLIP).conjugate() * frames[&parent].rotation;
                let turn = nearest_docking(twist(relative), pspec.rotations.max(cspec.rotations));
                Transform::new(
                    cspec.frame.position,
                    cspec.frame.rotation * FLIP * Quat::from_rotation_x(turn),
                )
            } else {
                scene
                    .body_pose(child_body)?
                    .inverse()
                    .compose(&frames[&parent])
            };
            let joint = scene.insert_loop_joint(
                JointDesc {
                    parent: pspec.body,
                    child: child_body,
                    kind: JointType::Fixed,
                    frame_parent: pspec.frame,
                    frame_child,
                    axis: Vec3::X,
                },
                JointSoftness {
                    natural_frequency: pspec.stiffness.min(cspec.stiffness),
                    damping_ratio: 1.0,
                },
            )?;
            self.links.insert(joint, Link { parent, child });
        }
        Ok(())
    }
}
