use crate::apple_tree_model::{self, TreeModel};
use molla_core::Result;
use molla_math::{Quat, Transform, Vec3};
use molla_solvers::{
    attachments::{Attachment, Particle},
    strands::{Fiber, StrandModel},
};

pub const ATTACHMENT_STRENGTH: f64 = 2.2;

#[derive(Clone)]
pub struct Leaf {
    pub owner: u64,
    pub rest: Transform,
    pub anchor: Vec3,
    pub offset: Vec3,
    pub direction: Vec3,
    pub normal: Vec3,
    pub length: f64,
    pub tint: f64,
    pub released_direction: Vec3,
    pub released_normal: Vec3,
}

pub struct Canopy {
    pub tree: TreeModel,
    pub twigs: Vec<u64>,
    pub leaves: Vec<Leaf>,
    pub attachments: Vec<Attachment>,
    pub particles: Vec<Particle>,
}

pub fn random(i: usize, salt: usize) -> f64 {
    let mut x = (i as u64)
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(salt as u64);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x ^ (x >> 31)) >> 11) as f64 / ((1u64 << 53) as f64)
}

fn length(f: &Fiber) -> f64 {
    let z: Vec<_> = f.shape.vertices().iter().map(|v| v.z).collect();
    z.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - z.iter().copied().fold(f64::INFINITY, f64::min)
}

fn radial_direction(axis: Vec3, angle: f64) -> Vec3 {
    let reference = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let side = axis.cross(reference).normalize();
    side * angle.cos() + axis.cross(side) * angle.sin()
}

fn clear_of_wood(shoot: &Fiber, fibers: &[Fiber], excluded: &[u64]) -> Result<bool> {
    let radius = |f: &Fiber| {
        f.shape
            .vertices()
            .iter()
            .map(|p| p.distance(f.mass.local_com))
            .fold(0.0, f64::max)
    };
    let shoot_radius = radius(shoot);
    for f in fibers.iter().filter(|f| !excluded.contains(&f.id)) {
        if shoot.center().distance(f.center()) > shoot_radius + radius(f) + 0.001 {
            continue;
        }
        if molla_geometry::cell_contact::cell_manifold(
            &shoot.shape,
            shoot.rest,
            &f.shape,
            f.rest,
            0.001,
        )?
        .is_some()
        {
            return Ok(false);
        }
    }
    Ok(shoot
        .shape
        .vertices()
        .iter()
        .all(|&p| shoot.rest.transform_point(p).y >= 0.002))
}

pub fn build() -> Result<Canopy> {
    build_with_foliage(0.022, 0.075)
}

pub fn build_with_foliage(spacing: f64, leaf_length: f64) -> Result<Canopy> {
    if !spacing.is_finite()
        || !(0.005..=0.1).contains(&spacing)
        || !leaf_length.is_finite()
        || !(0.02..=0.2).contains(&leaf_length)
    {
        return Err(molla_core::Error::Build(
            "invalid foliage dimensions".into(),
        ));
    }
    let mut tree =
        apple_tree_model::build(apple_tree_model::SOURCE, apple_tree_model::DEFAULT_LIMBS)?;
    let mut fibers = tree.model.fibers().to_vec();
    let mut bonds = tree.model.bonds().to_vec();
    let mut twigs = Vec::new();
    let paths = tree
        .limb_cells
        .iter()
        .map(|ids| (ids, false))
        .chain(std::iter::once((&tree.trunk_cells, true)));
    for (branch, (ids, trunk)) in paths.enumerate() {
        let total: f64 = ids
            .iter()
            .map(|&id| length(tree.model.fiber(id).unwrap()))
            .sum();
        let count = (total / if trunk { 0.26 } else { 0.30 }).ceil().max(1.0) as usize;
        for j in 0..count {
            let t = if count == 1 {
                0.72
            } else {
                j as f64 / (count - 1) as f64
            };
            let fraction = if trunk {
                0.06 + 0.925 * t
            } else {
                0.20 + 0.75 * t
            };
            let mut distance = total * fraction;
            let mut selected = None;
            for &id in ids {
                let f = tree.model.fiber(id).unwrap();
                let size = length(f);
                if distance <= size {
                    selected = Some((
                        id,
                        f.rest.transform_point(Vec3::Z * (distance - size * 0.5)),
                        f.rest.rotation * Vec3::Z,
                    ));
                    break;
                }
                distance -= size;
            }
            let Some((parent, base, axis)) = selected else {
                continue;
            };
            let angle = branch as f64 * 1.7 + j as f64 * 2.51;
            let lift = if base.y < 0.28 { 1.1 } else { 0.4 };
            let size = if trunk {
                0.08 + 0.16 * (1.0 - fraction)
            } else {
                (0.10 + 0.18 * (1.0 - fraction)) * (total / 0.8).sqrt().clamp(0.65, 1.15)
            };
            let radius = 0.0022 + 0.0015 * (1.0 - fraction);
            let id = fibers.len();
            let mut excluded = vec![parent];
            for bond in bonds.iter().filter(|b| b.fibers.contains(&parent)) {
                excluded.extend(bond.fibers);
            }
            let mut accepted = None;
            for attempt in 0..24 {
                let radial = radial_direction(axis, angle + attempt as f64 * 2.399963229728653);
                let direction = (radial * 0.8 + axis * 0.55 + Vec3::Y * lift).normalize();
                let end = base + direction * size;
                let pose = Transform::new(
                    (base + end) * 0.5,
                    Quat::from_rotation_arc(Vec3::Z, direction),
                );
                let shoot = Fiber::new(
                    id as u64,
                    apple_tree_model::tapered_cell(radius, radius * 0.55, size)?,
                    650.0,
                    pose,
                )?;
                if clear_of_wood(&shoot, &fibers, &excluded)? {
                    accepted = Some(shoot);
                    break;
                }
            }
            fibers.push(accepted.ok_or_else(|| {
                molla_core::Error::Build(format!("no clear leafy shoot direction on cell {parent}"))
            })?);
            apple_tree_model::connect(&mut bonds, &fibers, parent as usize, id, base, radius)?;
            twigs.push(id as u64);
        }
    }
    tree.model = StrandModel::new(fibers, bonds)?;
    let mut leaves = Vec::new();
    let mut attachments = Vec::new();
    let mut particles = Vec::new();
    for &id in &twigs {
        let f = tree.model.fiber(id).unwrap();
        let size = length(f);
        let count = (size / spacing).floor() as usize;
        let sample_weight =
            (size / 0.022).floor().max(1.0) / count.max(1) as f64 * (0.075 / leaf_length).powi(2);
        let axis = f.rest.rotation * Vec3::Z;
        for node in 0..count {
            let index = leaves.len();
            let t = (node as f64 + 0.7) / (count as f64 + 0.5);
            let angle = node as f64 * std::f64::consts::TAU * 0.4 + random(index, 31) * 0.35;
            let radial = f.rest.rotation * Vec3::new(angle.cos(), angle.sin(), 0.0);
            let dir = (radial + Vec3::Y * 0.4 + axis * 0.2).normalize();
            let normal = (Vec3::Y - dir * dir.y + radial * 0.15).normalize();
            let normal = (normal - dir * normal.dot(dir)).normalize();
            let len = leaf_length * (0.8 + 0.5 * random(index, 71));
            let anchor = Vec3::Z * (size * (t - 0.5));
            let offset = f.rest.rotation.conjugate() * dir * (len * 0.5 + 0.008);
            let mass = 0.00025 * (len / 0.075).powi(2) * sample_weight;
            let stiffness = mass * (std::f64::consts::TAU * 7.0).powi(2);
            leaves.push(Leaf {
                owner: id,
                rest: f.rest,
                anchor,
                offset,
                direction: f.rest.rotation.conjugate() * dir,
                normal: f.rest.rotation.conjugate() * normal,
                length: len,
                tint: random(index, 91),
                released_direction: dir,
                released_normal: normal,
            });
            attachments.push(Attachment {
                mass,
                stiffness,
                damping: 1.1 * (mass * stiffness).sqrt(),
                strength: mass * 9.81 * (1.8 + 5.0 * random(index, 101)) * ATTACHMENT_STRENGTH,
                drag: 0.00055 * (len / 0.075).powi(2) * sample_weight,
                radius: 0.002,
                petiole: None,
            });
            particles.push(Particle::attached(f.rest.transform_point(anchor + offset)));
        }
    }
    Ok(Canopy {
        tree,
        twigs,
        leaves,
        attachments,
        particles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new_shoots_start_clear_of_unrelated_wood_and_ground() {
        let c = build().unwrap();
        let neighbors = |id| {
            c.tree
                .model
                .bonds()
                .iter()
                .filter_map(|b| {
                    if b.fibers[0] == id {
                        Some(b.fibers[1])
                    } else if b.fibers[1] == id {
                        Some(b.fibers[0])
                    } else {
                        None
                    }
                })
                .collect::<std::collections::BTreeSet<_>>()
        };
        let mut issues = Vec::new();
        for (i, a) in c.tree.model.fibers().iter().enumerate() {
            for b in c.tree.model.fibers().iter().skip(i + 1) {
                if !c.twigs.contains(&a.id) && !c.twigs.contains(&b.id) {
                    continue;
                }
                if neighbors(a.id).contains(&b.id) || !neighbors(a.id).is_disjoint(&neighbors(b.id))
                {
                    continue;
                }
                if let Some(m) = molla_geometry::cell_contact::cell_manifold(
                    &a.shape, a.rest, &b.shape, b.rest, 0.0,
                )
                .unwrap()
                {
                    let depth = m
                        .points
                        .iter()
                        .map(|p| (p.a - p.b).dot(m.normal))
                        .fold(0.0, f64::max);
                    if depth > 1e-6 {
                        issues.push(format!("wood {}–{} penetration {depth}", a.id, b.id));
                    }
                }
            }
        }
        for (i, p) in c.particles.iter().enumerate() {
            if p.position.y < 0.002 {
                issues.push(format!("leaf {i} below floor at {}", p.position.y));
            }
        }
        assert!(issues.is_empty(), "{}", issues.join("\n"));
    }
    #[test]
    fn foliage_is_only_on_cuttable_tapered_twigs() {
        let c = build().unwrap();
        assert!(c.twigs.len() > 32 && c.tree.model.fibers().len() < 160);
        assert!(c.leaves.len() > 276 && c.leaves.len() < 900);
        assert_eq!(c.tree.model.components().len(), 1);
        for leaf in &c.leaves {
            assert!(c.twigs.contains(&leaf.owner));
            assert!(!c.tree.trunk_cells.contains(&leaf.owner));
            assert!(
                c.tree
                    .model
                    .fiber(leaf.owner)
                    .unwrap()
                    .contains_local(leaf.anchor)
            );
        }
        assert!(c.attachments.iter().all(|a| {
            let load = a.strength / (a.mass * 9.81);
            (1.8 * ATTACHMENT_STRENGTH..=6.8 * ATTACHMENT_STRENGTH).contains(&load)
        }));
    }

    #[test]
    fn every_authored_branch_has_leafy_shoots_regardless_of_depth_or_length() {
        let c = build().unwrap();
        for ids in &c.tree.limb_cells {
            let shoots: Vec<_> = c
                .tree
                .model
                .bonds()
                .iter()
                .filter_map(|b| {
                    b.fibers.iter().find(|id| ids.contains(id))?;
                    b.fibers.iter().copied().find(|id| c.twigs.contains(id))
                })
                .collect();
            assert!(!shoots.is_empty(), "bare authored branch {ids:?}");
            assert!(
                shoots
                    .iter()
                    .all(|id| c.leaves.iter().any(|leaf| leaf.owner == *id))
            );
        }
    }

    #[test]
    fn trunk_shoots_reach_the_top_and_taper_with_height() {
        let c = build().unwrap();
        let mut shoots: Vec<_> = c
            .tree
            .model
            .bonds()
            .iter()
            .filter_map(|b| {
                let side = b
                    .fibers
                    .iter()
                    .position(|id| c.tree.trunk_cells.contains(id))?;
                let id = b.fibers[1 - side];
                if !c.twigs.contains(&id) {
                    return None;
                }
                let parent = c.tree.model.fiber(b.fibers[side]).unwrap();
                Some((
                    parent.rest.transform_point(b.anchors[side]).y,
                    c.tree.model.fiber(id).unwrap(),
                ))
            })
            .collect();
        shoots.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(shoots.len() >= 8);
        let tip = c
            .tree
            .trunk_cells
            .iter()
            .flat_map(|&id| {
                let f = c.tree.model.fiber(id).unwrap();
                f.shape
                    .vertices()
                    .iter()
                    .map(move |&p| f.rest.transform_point(p).y)
            })
            .fold(0.0, f64::max);
        assert!(shoots[0].0 < 0.3);
        assert!(tip - shoots.last().unwrap().0 < 0.06);
        assert!(shoots.windows(2).all(|pair| pair[1].0 - pair[0].0 < 0.30));
        assert!(length(shoots[0].1) > length(shoots.last().unwrap().1));
        for (_, shoot) in shoots {
            assert!(c.leaves.iter().any(|leaf| leaf.owner == shoot.id));
        }
    }

    #[test]
    fn vertical_axes_have_a_finite_radial_basis() {
        for axis in [Vec3::Y, -Vec3::Y, Vec3::Z, Vec3::X] {
            let radial = radial_direction(axis, 1.7);
            assert!(radial.is_finite());
            assert!((radial.length() - 1.0).abs() < 1e-12);
            assert!(radial.dot(axis).abs() < 1e-12);
        }
    }
}
