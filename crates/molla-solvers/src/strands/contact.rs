use super::{
    Fiber, StrandModel,
    cpu::{ConstraintRow, pair_mut, rotate, weight},
    invalid,
};
use molla_core::Result;
use molla_geometry::cell_contact::cell_manifold;
use molla_math::{Real, Vec3};
use std::collections::BTreeSet;

struct Point {
    anchors: [Vec3; 2],
    lambda: Real,
    approach: Real,
}

struct Pair {
    normal: Vec3,
    points: Vec<Point>,
}

pub(super) struct Contacts {
    pairs: Vec<Option<Pair>>,
    groups: Vec<Vec<(usize, usize)>>,
}

pub(super) fn pair_index(a: usize, b: usize) -> usize {
    let hi = a.max(b);
    hi * (hi - 1) / 2 + a.min(b)
}

fn insert_group(
    groups: &mut Vec<Vec<(usize, usize)>>,
    used: &mut [BTreeSet<usize>],
    a: usize,
    b: usize,
) {
    let mut color = 0;
    while used[a].contains(&color) || used[b].contains(&color) {
        color += 1;
    }
    used[a].insert(color);
    used[b].insert(color);
    if color == groups.len() {
        groups.push(Vec::new());
    }
    groups[color].push((a, b));
}

fn row(normal: Vec3, point: &Point, a: &Fiber, b: &Fiber) -> ConstraintRow {
    let ra = a.motion.pose.rotation * (point.anchors[0] - a.mass.local_com);
    let rb = b.motion.pose.rotation * (point.anchors[1] - b.mass.local_com);
    ConstraintRow {
        value: ((b.center() - a.center()) + (rb - ra)).dot(normal),
        linear: [-normal, normal],
        angular: [-ra.cross(normal), rb.cross(normal)],
        compliance: 0.0,
    }
}

impl Contacts {
    pub fn detect(model: &StrandModel, margin: Real, exclusion_hops: u32) -> Result<Self> {
        let count = model.fibers.len();
        let capacity = count
            .checked_mul(count - 1)
            .map(|n| n / 2)
            .ok_or_else(|| invalid("contact pair count overflow"))?;
        let mut pairs: Vec<_> = (0..capacity).map(|_| None).collect();
        let mut groups = Vec::new();
        let mut used = vec![BTreeSet::new(); count];
        let mut excluded: BTreeSet<_> = model
            .bonds
            .iter()
            .filter(|b| b.intact())
            .map(|b| pair_index(model.indices[&b.fibers[0]], model.indices[&b.fibers[1]]))
            .collect();
        if exclusion_hops == 2 {
            let mut neighbors = vec![Vec::new(); count];
            for b in model.bonds.iter().filter(|b| b.intact()) {
                let a = model.indices[&b.fibers[0]];
                let c = model.indices[&b.fibers[1]];
                neighbors[a].push(c);
                neighbors[c].push(a);
            }
            for group in neighbors {
                for (i, &a) in group.iter().enumerate() {
                    for &b in &group[i + 1..] {
                        if a != b {
                            excluded.insert(pair_index(a, b));
                        }
                    }
                }
            }
        }
        for bi in 1..count {
            for ai in 0..bi {
                let index = pair_index(ai, bi);
                let a = &model.fibers[ai];
                let b = &model.fibers[bi];
                if excluded.contains(&index) || (a.pin.is_some() && b.pin.is_some()) {
                    continue;
                }
                if let Some(manifold) =
                    cell_manifold(&a.shape, a.motion.pose, &b.shape, b.motion.pose, margin)?
                {
                    insert_group(&mut groups, &mut used, ai, bi);
                    pairs[index] = Some(Pair {
                        normal: manifold.normal,
                        points: manifold
                            .points
                            .into_iter()
                            .map(|p| Point {
                                anchors: [
                                    a.motion.pose.rotation.conjugate()
                                        * (p.a - a.motion.pose.position),
                                    b.motion.pose.rotation.conjugate()
                                        * (p.b - b.motion.pose.position),
                                ],
                                lambda: 0.0,
                                approach: (b.motion.linear_velocity
                                    + b.motion.angular_velocity.cross(p.b - b.center())
                                    - a.motion.linear_velocity
                                    - a.motion.angular_velocity.cross(p.a - a.center()))
                                .dot(manifold.normal),
                            })
                            .collect(),
                    });
                }
            }
        }
        Ok(Self { pairs, groups })
    }

    pub fn project(&mut self, model: &mut StrandModel) -> Result<()> {
        for group in &self.groups {
            for &(i, j) in group {
                let Some(pair) = &mut self.pairs[pair_index(i, j)] else {
                    continue;
                };
                let (a, b) = pair_mut(&mut model.fibers, i.min(j), i.max(j));
                for point in &mut pair.points {
                    let r = row(pair.normal, point, a, b);
                    let (w, inverse) = weight(r, a, b)?;
                    if w == 0.0 {
                        continue;
                    }
                    let lambda = (point.lambda - r.value / w).max(0.0);
                    let mut dl = lambda - point.lambda;
                    let angle = (inverse[0] * r.angular[0])
                        .length()
                        .max((inverse[1] * r.angular[1]).length())
                        * dl.abs();
                    if angle > 0.2 {
                        dl *= 0.2 / angle;
                    }
                    point.lambda += dl;
                    for (side, fiber) in [&mut *a, &mut *b].into_iter().enumerate() {
                        if fiber.pin.is_some() {
                            continue;
                        }
                        let center = fiber.center() + r.linear[side] * (fiber.inverse_mass() * dl);
                        rotate(fiber, inverse[side] * r.angular[side] * dl, center);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn velocities(
        &self,
        model: &mut StrandModel,
        coefficient: Real,
        dt: Real,
        restitution: Real,
    ) -> Result<()> {
        for group in &self.groups {
            for &(i, j) in group {
                let Some(pair) = &self.pairs[pair_index(i, j)] else {
                    continue;
                };
                let (a, b) = pair_mut(&mut model.fibers, i.min(j), i.max(j));
                let mut normal_impulses = vec![0.0; pair.points.len()];
                for _ in 0..12 {
                    for (index, point) in pair.points.iter().enumerate() {
                        if point.lambda == 0.0 {
                            continue;
                        }
                        let r = row(pair.normal, point, a, b);
                        let (w, inverse) = weight(r, a, b)?;
                        if w == 0.0 {
                            continue;
                        }
                        let rate = r.linear[0].dot(a.motion.linear_velocity)
                            + r.linear[1].dot(b.motion.linear_velocity)
                            + r.angular[0].dot(a.motion.angular_velocity)
                            + r.angular[1].dot(b.motion.angular_velocity);
                        let impulse = (-restitution * point.approach.min(0.0) - rate) / w;
                        normal_impulses[index] += impulse;
                        for (side, fiber) in [&mut *a, &mut *b].into_iter().enumerate() {
                            fiber.motion.linear_velocity +=
                                r.linear[side] * (fiber.inverse_mass() * impulse);
                            fiber.motion.angular_velocity +=
                                inverse[side] * r.angular[side] * impulse;
                        }
                    }
                }
                for (index, point) in pair.points.iter().enumerate() {
                    if point.lambda == 0.0 {
                        continue;
                    }
                    let ra = a.motion.pose.rotation * (point.anchors[0] - a.mass.local_com);
                    let rb = b.motion.pose.rotation * (point.anchors[1] - b.mass.local_com);
                    let relative = b.motion.linear_velocity + b.motion.angular_velocity.cross(rb)
                        - a.motion.linear_velocity
                        - a.motion.angular_velocity.cross(ra);
                    let tangent = relative - pair.normal * relative.dot(pair.normal);
                    let speed = tangent.length();
                    if speed == 0.0 {
                        continue;
                    }
                    let n = tangent / speed;
                    let r = row(n, point, a, b);
                    let (w, inverse) = weight(r, a, b)?;
                    if w == 0.0 {
                        continue;
                    }
                    let impulse = -(speed / w)
                        .min(coefficient * (point.lambda / dt + normal_impulses[index]).max(0.0));
                    for (side, fiber) in [&mut *a, &mut *b].into_iter().enumerate() {
                        fiber.motion.linear_velocity +=
                            r.linear[side] * (fiber.inverse_mass() * impulse);
                        fiber.motion.angular_velocity += inverse[side] * r.angular[side] * impulse;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_pair_groups_visit_each_pair_once_without_shared_bodies() {
        for size in 1usize..140 {
            let mut groups = Vec::new();
            let mut used = vec![BTreeSet::new(); size];
            for b in 1..size {
                for a in 0..b {
                    insert_group(&mut groups, &mut used, a, b);
                }
            }
            assert!(groups.len() < 2 * size);
            let mut seen = BTreeSet::new();
            for group in groups {
                let mut bodies = BTreeSet::new();
                for (a, b) in group {
                    assert!(bodies.insert(a) && bodies.insert(b));
                    assert!(seen.insert(pair_index(a, b)));
                }
            }
            assert_eq!(seen.len(), size * (size - 1) / 2);
            assert_eq!(
                seen.into_iter().collect::<Vec<_>>(),
                (0..size * (size - 1) / 2).collect::<Vec<_>>()
            );
        }
    }
}
