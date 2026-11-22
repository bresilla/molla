use super::{BondPatch, invalid};
use molla_core::Result;
use molla_geometry::convex_cut::{ConvexCell, RigidCellMotion};
use molla_math::{Mat3, Quat, Real, Transform, Vec3, color_constraints, group_by_color};
use molla_sim::MassProperties;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

fn next_revision() -> Result<u64> {
    static REVISION: AtomicU64 = AtomicU64::new(1);
    REVISION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
        .map_err(|_| invalid("revision exhausted"))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pin {
    pub local_point: Vec3,
    pub pose: Transform,
}

#[derive(Clone, Debug)]
pub struct Fiber {
    pub id: u64,
    pub shape: ConvexCell,
    pub density: Real,
    pub mass: MassProperties,
    pub rest: Transform,
    pub motion: RigidCellMotion,
    pub pin: Option<Pin>,
}

impl Fiber {
    pub fn new(id: u64, shape: ConvexCell, density: Real, rest: Transform) -> Result<Self> {
        let mass = shape.mass_properties(density)?;
        let fiber = Self {
            id,
            shape,
            density,
            mass,
            rest,
            motion: RigidCellMotion {
                pose: rest,
                linear_velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
            pin: None,
        };
        fiber.validate()?;
        Ok(fiber)
    }

    pub fn center(&self) -> Vec3 {
        self.motion.pose.transform_point(self.mass.local_com)
    }

    pub(super) fn inverse_mass(&self) -> Real {
        if self.pin.is_some() {
            0.0
        } else {
            self.mass.mass.recip()
        }
    }

    pub(super) fn world_inverse_inertia(&self) -> Result<Mat3> {
        if self.pin.is_some() {
            return Ok(Mat3::ZERO);
        }
        let r = self.motion.pose.rotation_matrix();
        Ok(r * self.mass.inverse_inertia()? * r.transpose())
    }

    pub fn contains_local(&self, point: Vec3) -> bool {
        let scale = self
            .shape
            .vertices()
            .iter()
            .map(|p| p.distance(self.mass.local_com))
            .fold(0.0, Real::max);
        self.shape.faces().iter().all(|f| {
            let a = self.shape.vertices()[f.vertices[0] as usize];
            let normal = (1..f.vertices.len() - 1)
                .map(|i| {
                    (self.shape.vertices()[f.vertices[i] as usize] - a)
                        .cross(self.shape.vertices()[f.vertices[i + 1] as usize] - a)
                })
                .sum::<Vec3>()
                .normalize();
            normal.dot(point - a) <= scale * 1e-10
        })
    }

    pub fn validate(&self) -> Result<()> {
        self.shape.validate()?;
        validate_pose(self.rest)?;
        self.validate_motion()?;
        let computed = self.shape.mass_properties(self.density)?;
        if self.mass != computed {
            return Err(invalid("invalid fiber mass or motion"));
        }
        if let Some(pin) = self.pin {
            validate_pose(pin.pose)?;
            if !pin.local_point.is_finite()
                || !self.contains_local(pin.local_point)
                || self.motion.pose != pin.pose
                || self.motion.linear_velocity != Vec3::ZERO
                || self.motion.angular_velocity != Vec3::ZERO
            {
                return Err(invalid("invalid pin or moving pinned fiber"));
            }
        }
        Ok(())
    }

    pub(super) fn validate_motion(&self) -> Result<()> {
        validate_motion(self.motion, self.pin)
    }
}

fn validate_motion(motion: RigidCellMotion, pin: Option<Pin>) -> Result<()> {
    validate_pose(motion.pose)?;
    if !motion.linear_velocity.is_finite() || !motion.angular_velocity.is_finite() {
        return Err(invalid("nonfinite fiber velocity"));
    }
    if let Some(pin) = pin
        && (motion.pose != pin.pose
            || motion.linear_velocity != Vec3::ZERO
            || motion.angular_velocity != Vec3::ZERO)
    {
        return Err(invalid("moving pinned fiber"));
    }
    Ok(())
}

fn validate_pose(pose: Transform) -> Result<()> {
    if !pose.position.is_finite()
        || !pose.rotation.is_finite()
        || (pose.rotation.length_squared() - 1.0).abs() > 1e-10
    {
        return Err(invalid("invalid material pose"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondMaterial {
    /// Translation compliance in m/N, expressed in the first fiber's frame.
    pub translation: Vec3,
    /// Rotation compliance in rad/(N m), expressed in the first fiber's frame.
    pub rotation: Vec3,
    pub force_limit: Vec3,
    pub torque_limit: Vec3,
    /// Damage per second per unit overload; zero disables fracture.
    pub damage_rate: Real,
    /// Relative-velocity damping rate in inverse seconds.
    pub damping: Real,
}

impl BondMaterial {
    pub fn validate(self) -> Result<()> {
        if [self.translation, self.rotation]
            .iter()
            .any(|v| !v.is_finite() || v.min_element() <= 0.0)
            || [self.force_limit, self.torque_limit]
                .iter()
                .any(|v| !v.is_finite() || v.min_element() <= 0.0)
            || !self.damage_rate.is_finite()
            || self.damage_rate < 0.0
            || !self.damping.is_finite()
            || self.damping < 0.0
        {
            return Err(invalid("invalid bond material"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Bond {
    pub id: u64,
    pub fibers: [u64; 2],
    pub anchors: [Vec3; 2],
    pub rest_offset: Vec3,
    pub rest_rotation: Quat,
    pub material: BondMaterial,
    pub patch: Option<BondPatch>,
    /// Lost load-bearing fraction; one removes the connection entirely.
    pub damage: Real,
}

impl Bond {
    pub fn new(
        id: u64,
        fibers: [&Fiber; 2],
        anchors: [Vec3; 2],
        material: BondMaterial,
    ) -> Result<Self> {
        material.validate()?;
        let [a, b] = fibers;
        if a.id == b.id
            || anchors.iter().any(|p| !p.is_finite())
            || !a.contains_local(anchors[0])
            || !b.contains_local(anchors[1])
        {
            return Err(invalid("invalid bond endpoints"));
        }
        Ok(Self {
            id,
            fibers: [a.id, b.id],
            anchors,
            material,
            damage: 0.0,
            patch: None,
            rest_offset: a.rest.rotation.conjugate()
                * (b.rest.transform_point(anchors[1]) - a.rest.transform_point(anchors[0])),
            rest_rotation: (a.rest.rotation.conjugate() * b.rest.rotation).normalize(),
        })
    }

    pub fn intact(&self) -> bool {
        self.damage < 1.0
    }

    pub fn with_patch(mut self, patch: BondPatch) -> Result<Self> {
        let a = patch.measure(0)?;
        let b = patch.measure(1)?;
        self.rest_offset +=
            self.rest_rotation * (b.center - self.anchors[1]) - (a.center - self.anchors[0]);
        self.anchors = [a.center, b.center];
        self.patch = Some(patch);
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct StrandModel {
    pub(super) fibers: Vec<Fiber>,
    pub(super) bonds: Vec<Bond>,
    pub(super) indices: BTreeMap<u64, usize>,
    pub(super) colors: Vec<Vec<usize>>,
    pub(super) revision: u64,
    pub(super) next_id: u64,
    pub(super) next_bond_id: u64,
}

impl StrandModel {
    pub fn new(fibers: Vec<Fiber>, bonds: Vec<Bond>) -> Result<Self> {
        if fibers.is_empty() || fibers.len() > u32::MAX as usize || bonds.len() > u32::MAX as usize
        {
            return Err(invalid("invalid graph size"));
        }
        let next_id = fibers
            .iter()
            .map(|f| f.id)
            .max()
            .unwrap()
            .checked_add(1)
            .ok_or_else(|| invalid("fiber ID exhausted"))?;
        let next_bond_id = bonds
            .iter()
            .map(|b| b.id)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| invalid("bond ID exhausted"))?;
        let mut model = Self {
            fibers,
            bonds,
            indices: BTreeMap::new(),
            colors: Vec::new(),
            revision: next_revision()?,
            next_id,
            next_bond_id,
        };
        model.rebuild()?;
        Ok(model)
    }

    pub fn fibers(&self) -> &[Fiber] {
        &self.fibers
    }
    pub fn bonds(&self) -> &[Bond] {
        &self.bonds
    }
    pub fn colors(&self) -> &[Vec<usize>] {
        &self.colors
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn fiber(&self, id: u64) -> Option<&Fiber> {
        self.indices.get(&id).map(|&i| &self.fibers[i])
    }

    pub fn set_motion(&mut self, id: u64, motion: RigidCellMotion) -> Result<()> {
        let i = *self
            .indices
            .get(&id)
            .ok_or_else(|| invalid("unknown fiber"))?;
        validate_motion(motion, self.fibers[i].pin)?;
        self.advance_revision()?;
        self.fibers[i].motion = motion;
        Ok(())
    }

    pub fn impulse(&mut self, id: u64, point: Vec3, impulse: Vec3) -> Result<()> {
        let fiber = self
            .fiber(id)
            .ok_or_else(|| invalid("unknown impulse fiber"))?;
        if !point.is_finite() || !impulse.is_finite() {
            return Err(invalid("invalid impulse"));
        }
        if fiber.pin.is_some() {
            return Ok(());
        }
        let motion = RigidCellMotion {
            linear_velocity: fiber.motion.linear_velocity + impulse * fiber.inverse_mass(),
            angular_velocity: fiber.motion.angular_velocity
                + fiber.world_inverse_inertia()? * (point - fiber.center()).cross(impulse),
            ..fiber.motion
        };
        self.set_motion(id, motion)
    }

    pub(super) fn advance_revision(&mut self) -> Result<()> {
        self.revision = next_revision()?;
        Ok(())
    }

    pub(super) fn rebuild(&mut self) -> Result<()> {
        self.indices.clear();
        for (i, fiber) in self.fibers.iter().enumerate() {
            fiber.validate()?;
            if self.indices.insert(fiber.id, i).is_some() {
                return Err(invalid("duplicate fiber ID"));
            }
        }
        let mut ids = BTreeSet::new();
        for bond in &self.bonds {
            bond.material.validate()?;
            if !ids.insert(bond.id)
                || bond.fibers[0] == bond.fibers[1]
                || !bond.rest_offset.is_finite()
                || !bond.rest_rotation.is_finite()
                || (bond.rest_rotation.length_squared() - 1.0).abs() > 1e-10
                || !bond.damage.is_finite()
                || !(0.0..=1.0).contains(&bond.damage)
            {
                return Err(invalid("invalid bond"));
            }
            for (&id, &anchor) in bond.fibers.iter().zip(&bond.anchors) {
                let fiber = self
                    .fiber(id)
                    .ok_or_else(|| invalid("unknown bond fiber"))?;
                if !anchor.is_finite() || !fiber.contains_local(anchor) {
                    return Err(invalid("bond anchor outside fiber"));
                }
            }
            if let Some(patch) = &bond.patch {
                for side in 0..2 {
                    let measure = patch.measure(side)?;
                    if measure.center.distance(bond.anchors[side]) > measure.area.sqrt() * 1e-9
                        || patch.points.iter().any(|p| {
                            !self
                                .fiber(bond.fibers[side])
                                .unwrap()
                                .contains_local(p[side])
                        })
                    {
                        return Err(invalid("invalid interface attachment"));
                    }
                }
            }
        }
        let edges: Vec<_> = self
            .bonds
            .iter()
            .map(|b| b.fibers.map(|id| self.indices[&id] as u32).to_vec())
            .collect();
        self.colors = group_by_color(&color_constraints(&edges));
        Ok(())
    }

    pub fn components(&self) -> Vec<Vec<u64>> {
        let mut neighbors = vec![Vec::new(); self.fibers.len()];
        for bond in self.bonds.iter().filter(|b| b.intact()) {
            let [a, b] = bond.fibers.map(|id| self.indices[&id]);
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
        let mut seen = vec![false; self.fibers.len()];
        let mut result = Vec::new();
        for start in 0..self.fibers.len() {
            if seen[start] {
                continue;
            }
            let mut stack = vec![start];
            let mut component = Vec::new();
            seen[start] = true;
            while let Some(i) = stack.pop() {
                component.push(self.fibers[i].id);
                for &j in &neighbors[i] {
                    if !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            component.sort_unstable();
            result.push(component);
        }
        result
    }
}
