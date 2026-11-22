use super::{Bond, BondMaterial, StrandModel, invalid};
use molla_core::Result;
use molla_geometry::convex_cut::{ConvexCell, CutPlane, FaceOrigin};
use molla_math::Vec3;

#[derive(Clone, Debug)]
pub struct SplinterReport {
    pub replaced: u64,
    pub fragments: [Vec<u64>; 2],
}

pub struct SplinterTransaction {
    revision: u64,
    candidate: StrandModel,
    report: SplinterReport,
}

impl SplinterTransaction {
    pub fn candidate(&self) -> &StrandModel {
        &self.candidate
    }
    pub fn report(&self) -> &SplinterReport {
        &self.report
    }
}

fn tagged_split(
    shape: &ConvexCell,
    plane: CutPlane,
    event: u64,
    fracture: bool,
) -> Result<Option<[ConvexCell; 2]>> {
    let Some(parts) = shape.split(plane, event)? else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for part in parts {
        let mut faces = part.faces().to_vec();
        for face in &mut faces {
            if let FaceOrigin::Cut { id, positive_side } = face.origin
                && id == event
            {
                face.origin = if fracture {
                    FaceOrigin::Fracture { id, positive_side }
                } else {
                    FaceOrigin::Interface(id)
                };
            }
        }
        out.push(ConvexCell::new(part.vertices().to_vec(), faces)?);
    }
    Ok(Some([out.remove(0), out.remove(0)]))
}

impl StrandModel {
    /// Split a point-attached cell into two nonplanar fragment assemblies at the requested plane.
    pub fn prepare_splinter(
        &self,
        id: u64,
        plane: CutPlane,
        event: u64,
    ) -> Result<SplinterTransaction> {
        let parent = self
            .fiber(id)
            .ok_or_else(|| invalid("unknown splinter cell"))?;
        if self
            .bonds
            .iter()
            .any(|b| b.fibers.contains(&id) && b.patch.is_some())
        {
            return Err(invalid("splinter refinement requires point attachments"));
        }
        let local = plane.to_local(parent.motion.pose)?;
        if parent
            .pin
            .is_some_and(|pin| local.distance(pin.local_point).abs() < 1e-10)
        {
            return Err(invalid("break intersects root attachment"));
        }
        let center = parent.mass.local_com;
        let radius = parent
            .shape
            .vertices()
            .iter()
            .map(|p| (p.truncate() - center.truncate()).length())
            .fold(0.0, f64::max);
        let lo = parent
            .shape
            .vertices()
            .iter()
            .map(|&p| local.distance(p))
            .fold(f64::INFINITY, f64::min);
        let hi = parent
            .shape
            .vertices()
            .iter()
            .map(|&p| local.distance(p))
            .fold(f64::NEG_INFINITY, f64::max);
        if lo >= 0.0 || hi <= 0.0 {
            return Err(invalid("break plane does not cross the cell"));
        }
        let depth = (radius * 1.4).min((-lo).min(hi) * 0.6);
        let mut sectors = vec![parent.shape.clone()];
        for axis in [Vec3::X, Vec3::Y] {
            let partition = CutPlane::new(center, axis)?;
            let mut next = Vec::new();
            for shape in sectors {
                match tagged_split(&shape, partition, event, false)? {
                    Some(parts) => next.extend(parts),
                    None => next.push(shape),
                }
            }
            sectors = next;
        }
        let contains_hit = |shape: &ConvexCell| {
            shape.faces().iter().enumerate().all(|(i, face)| {
                let p = shape.vertices()[face.vertices[0] as usize];
                molla_geometry::cell_contact::face_normal(shape, i).dot(local.point() - p)
                    <= radius * 1e-8
            })
        };
        let picked = sectors.iter().position(contains_hit);
        let mut shapes = Vec::new();
        for (i, sector) in sectors.into_iter().enumerate() {
            let offset = if picked == Some(i) {
                0.0
            } else {
                [-0.85, 0.7, 1.0, -0.4][i % 4] * depth
            };
            let surface = CutPlane::new(local.point() + local.normal() * offset, local.normal())?;
            match tagged_split(&sector, surface, event, true)? {
                Some(parts) => {
                    for (side, shape) in parts.into_iter().enumerate() {
                        shapes.push((side, shape));
                    }
                }
                None => {
                    let side = usize::from(
                        surface.distance(sector.mass_properties(parent.density)?.local_com) > 0.0,
                    );
                    shapes.push((side, sector));
                }
            }
        }
        let mut candidate = self.clone();
        let mut children = Vec::new();
        let mut groups = [Vec::new(), Vec::new()];
        let mut pin_assigned = false;
        for (side, shape) in shapes {
            let mut child = parent.clone();
            child.id = candidate.next_id;
            candidate.next_id = candidate
                .next_id
                .checked_add(1)
                .ok_or_else(|| invalid("fiber ID exhausted"))?;
            child.mass = shape.mass_properties(child.density)?;
            child.shape = shape;
            child.motion = parent.motion.restricted(parent.mass, child.mass)?;
            child.pin = None;
            if let Some(pin) = parent.pin
                && !pin_assigned
                && child.contains_local(pin.local_point)
            {
                child.pin = Some(pin);
                pin_assigned = true;
            }
            groups[side].push(children.len());
            children.push(child);
        }
        if groups.iter().any(Vec::is_empty) || parent.pin.is_some() && !pin_assigned {
            return Err(invalid("unresolved splinter fragments"));
        }
        for bond in &mut candidate.bonds {
            for side in 0..2 {
                if bond.fibers[side] == id {
                    bond.fibers[side] = children
                        .iter()
                        .find(|c| c.contains_local(bond.anchors[side]))
                        .ok_or_else(|| invalid("splinter attachment outside children"))?
                        .id;
                }
            }
        }
        let length = (hi - lo).max(radius);
        let area = parent.mass.mass / parent.density / length;
        let material = BondMaterial {
            translation: Vec3::splat(length / (2e8 * area)),
            rotation: Vec3::splat(length / (2e9 * area * radius * radius)),
            force_limit: Vec3::splat(1e12),
            torque_limit: Vec3::splat(1e12),
            damage_rate: 0.0,
            damping: 2.0,
        };
        for group in &groups {
            let a = &children[group[0]];
            for &index in &group[1..] {
                let b = &children[index];
                candidate.bonds.push(Bond::new(
                    candidate.next_bond_id,
                    [a, b],
                    [a.mass.local_com, b.mass.local_com],
                    material,
                )?);
                candidate.next_bond_id = candidate
                    .next_bond_id
                    .checked_add(1)
                    .ok_or_else(|| invalid("bond ID exhausted"))?;
            }
        }
        let report = SplinterReport {
            replaced: id,
            fragments: groups.map(|g| g.into_iter().map(|i| children[i].id).collect()),
        };
        candidate.fibers.retain(|f| f.id != id);
        candidate.fibers.extend(children);
        candidate.rebuild()?;
        candidate.advance_revision()?;
        Ok(SplinterTransaction {
            revision: self.revision,
            candidate,
            report,
        })
    }

    pub fn commit_splinter(&mut self, transaction: SplinterTransaction) -> Result<SplinterReport> {
        if self.revision != transaction.revision {
            return Err(invalid("stale splinter transaction"));
        }
        *self = transaction.candidate;
        Ok(transaction.report)
    }
}
