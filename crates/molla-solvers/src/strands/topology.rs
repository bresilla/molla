use super::{StrandModel, invalid};
use molla_core::Result;
use molla_geometry::convex_cut::{ConvexCell, CutPlane, FaceOrigin};
use molla_math::Real;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default)]
pub struct CutReport {
    pub replacements: Vec<(u64, [u64; 2])>,
    pub severed_bonds: Vec<u64>,
    pub split_bonds: Vec<(u64, Vec<u64>)>,
}

pub struct CutTransaction {
    revision: u64,
    candidate: StrandModel,
    report: CutReport,
}

impl CutTransaction {
    pub fn candidate(&self) -> &StrandModel {
        &self.candidate
    }
    pub fn report(&self) -> &CutReport {
        &self.report
    }
}

impl StrandModel {
    /// Prepare a zero-kerf plane cut through the selected fibers and incident bonds.
    pub fn prepare_cut(
        &self,
        plane: CutPlane,
        selected: &[u64],
        cut_id: u64,
    ) -> Result<CutTransaction> {
        let selection: BTreeSet<_> = selected.iter().copied().collect();
        if selection.len() != selected.len() || selected.iter().any(|id| self.fiber(*id).is_none())
        {
            return Err(invalid("invalid cut selection"));
        }
        self.prepare_sections(
            plane,
            selected.iter().map(|&id| (id, plane)).collect(),
            cut_id,
            false,
        )
    }

    /// Prescribed per-fiber fracture surfaces; automatic stress failure remains bond damage.
    pub fn prepare_fracture(
        &self,
        reference: CutPlane,
        surfaces: &[(u64, CutPlane)],
        event_id: u64,
    ) -> Result<CutTransaction> {
        let planes: BTreeMap<_, _> = surfaces.iter().copied().collect();
        if planes.len() != surfaces.len()
            || surfaces.iter().any(|(id, p)| {
                self.fiber(*id).is_none() || p.normal().dot(reference.normal()) <= 0.0
            })
        {
            return Err(invalid("invalid fracture surfaces"));
        }
        self.prepare_sections(reference, planes, event_id, true)
    }

    fn prepare_sections(
        &self,
        reference: CutPlane,
        planes: BTreeMap<u64, CutPlane>,
        cut_id: u64,
        fracture: bool,
    ) -> Result<CutTransaction> {
        let selection: BTreeSet<_> = planes.keys().copied().collect();
        let mut candidate = self.clone();
        let mut report = CutReport::default();
        let mut replacements = BTreeMap::new();
        let mut fibers = Vec::new();
        for fiber in &self.fibers {
            let local = planes
                .get(&fiber.id)
                .copied()
                .unwrap_or(reference)
                .to_local(fiber.motion.pose)?;
            let split = if selection.contains(&fiber.id) {
                fiber.shape.split(local, cut_id)?
            } else {
                None
            };
            if let Some(shapes) = split {
                let first = candidate.next_id;
                candidate.next_id = first
                    .checked_add(2)
                    .ok_or_else(|| invalid("fiber ID exhausted"))?;
                let ids = [first, first + 1];
                replacements.insert(fiber.id, ids);
                report.replacements.push((fiber.id, ids));
                let pin_side = if let Some(pin) = fiber.pin {
                    let distance = local.distance(pin.local_point);
                    let extent = fiber
                        .shape
                        .vertices()
                        .iter()
                        .map(|p| p.distance(pin.local_point))
                        .fold(0.0, Real::max);
                    if distance.abs() < extent * 1e-10 {
                        return Err(invalid("blade intersects root attachment"));
                    }
                    Some(usize::from(distance > 0.0))
                } else {
                    None
                };
                for (side, mut shape) in shapes.into_iter().enumerate() {
                    if fracture {
                        let mut faces = shape.faces().to_vec();
                        for face in &mut faces {
                            if let FaceOrigin::Cut { id, positive_side } = face.origin
                                && id == cut_id
                            {
                                face.origin = FaceOrigin::Fracture { id, positive_side };
                            }
                        }
                        shape = ConvexCell::new(shape.vertices().to_vec(), faces)?;
                    }
                    let mut child = fiber.clone();
                    child.id = ids[side];
                    child.shape = shape;
                    child.mass = child.shape.mass_properties(child.density)?;
                    child.motion = fiber.motion.restricted(fiber.mass, child.mass)?;
                    child.pin = if pin_side == Some(side) {
                        fiber.pin
                    } else {
                        None
                    };
                    fibers.push(child);
                }
            } else {
                fibers.push(fiber.clone());
            }
        }
        let mut bonds = Vec::new();
        for bond in &self.bonds {
            let touched = bond.fibers.iter().any(|id| selection.contains(id));
            let bond_planes = bond
                .fibers
                .map(|id| planes.get(&id).copied().unwrap_or(reference));
            if touched && let Some(patch) = &bond.patch {
                let measure = patch.measure(0)?;
                let poses = bond.fibers.map(|id| self.fiber(id).unwrap().motion.pose);
                let mut surviving = Vec::new();
                for side in 0..2 {
                    let Some(part) = patch.clip_sides(bond_planes, poses, side == 1)? else {
                        continue;
                    };
                    let part_measure = part.measure(0)?;
                    let fraction = part_measure.area / measure.area;
                    let moments = part_measure.moments / measure.moments;
                    let mut remaining = bond.clone().with_patch(part)?;
                    remaining.material.translation /= fraction;
                    remaining.material.rotation /= moments;
                    remaining.material.force_limit *= fraction;
                    remaining.material.torque_limit *= fraction;
                    for i in 0..2 {
                        if let Some(ids) = replacements.get(&bond.fibers[i]) {
                            remaining.fibers[i] = ids[side];
                        }
                    }
                    if !surviving.is_empty() {
                        remaining.id = candidate.next_bond_id;
                        candidate.next_bond_id = candidate
                            .next_bond_id
                            .checked_add(1)
                            .ok_or_else(|| invalid("bond ID exhausted"))?;
                    }
                    surviving.push(remaining.id);
                    bonds.push(remaining);
                }
                if surviving.is_empty() {
                    report.severed_bonds.push(bond.id);
                } else if surviving.len() > 1 {
                    report.split_bonds.push((bond.id, surviving));
                }
                continue;
            }
            let endpoints = [0, 1].map(|i| {
                self.fiber(bond.fibers[i])
                    .unwrap()
                    .motion
                    .pose
                    .transform_point(bond.anchors[i])
            });
            let distances = [0, 1].map(|i| bond_planes[i].distance(endpoints[i]));
            let on_plane = |i: usize| {
                let f = self.fiber(bond.fibers[i]).unwrap();
                let scale = f
                    .shape
                    .vertices()
                    .iter()
                    .map(|p| p.distance(f.mass.local_com))
                    .fold(0.0, Real::max);
                distances[i].abs() <= scale * 64.0 * Real::EPSILON
            };
            if touched
                && ((distances[0] < 0.0) != (distances[1] < 0.0) || on_plane(0) || on_plane(1))
            {
                report.severed_bonds.push(bond.id);
                continue;
            }
            let mut remapped = bond.clone();
            for i in 0..2 {
                if let Some(ids) = replacements.get(&bond.fibers[i]) {
                    remapped.fibers[i] = ids[usize::from(distances[i] > 0.0)];
                }
            }
            bonds.push(remapped);
        }
        candidate.fibers = fibers;
        candidate.bonds = bonds;
        candidate.rebuild()?;
        candidate.advance_revision()?;
        Ok(CutTransaction {
            revision: self.revision,
            candidate,
            report,
        })
    }

    /// Publish only a transaction prepared from this exact simulation revision.
    pub fn commit_cut(&mut self, transaction: CutTransaction) -> Result<CutReport> {
        if self.revision != transaction.revision {
            return Err(invalid("stale cut transaction"));
        }
        *self = transaction.candidate;
        Ok(transaction.report)
    }
}
