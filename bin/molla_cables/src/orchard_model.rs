use crate::{
    apple_tree_model::TreeModel,
    leafy_model::{self, Canopy},
};
use molla_core::Result;
use molla_math::{Quat, Transform, Vec3};
use molla_solvers::strands::StrandModel;

pub const TREES_PER_ROW: usize = 8;
pub const TREE_COUNT: usize = 2 * TREES_PER_ROW;
pub const TREE_SPACING: f64 = 2.4;
pub const ROW_X: [f64; 2] = [-1.45, 1.45];
pub const ROW_HALF_LENGTH: f64 = (TREES_PER_ROW - 1) as f64 * TREE_SPACING * 0.5;
pub const WOOD_FLEXIBILITY: f64 = 4.0;

/// Deterministic yaw of `tree` about its trunk, uniform over a full turn.
pub fn tree_yaw(tree: usize) -> f64 {
    std::f64::consts::TAU * leafy_model::random(tree, 2701)
}

#[cfg(test)]
pub fn build() -> Result<Canopy> {
    build_with_flexibility(WOOD_FLEXIBILITY)
}

pub fn build_with_flexibility(flexibility: f64) -> Result<Canopy> {
    if !flexibility.is_finite() || !(0.25..=16.0).contains(&flexibility) {
        return Err(molla_core::Error::Build(
            "wood flexibility must be between 0.25 and 16".into(),
        ));
    }
    let base = leafy_model::build_with_foliage(0.010, 0.100)?;
    let mut fibers = Vec::new();
    let mut bonds = Vec::new();
    let mut trunk_cells = Vec::new();
    let mut limb_cells = Vec::new();
    let mut twigs = Vec::new();
    let mut leaves = Vec::new();
    let mut particles = Vec::new();
    let mut attachments = Vec::new();
    let fiber_stride = base.tree.model.fibers().iter().map(|f| f.id).max().unwrap() + 1;
    let bond_stride = base.tree.model.bonds().iter().map(|b| b.id).max().unwrap() + 1;
    let root = base.tree.model.fibers().iter().find(|f| f.pin.is_some()).unwrap().center();
    let axis = Vec3::new(root.x, 0.0, root.z);
    for tree in 0..TREE_COUNT {
        let site = Vec3::new(
            ROW_X[tree / TREES_PER_ROW],
            0.0,
            (tree % TREES_PER_ROW) as f64 * TREE_SPACING - ROW_HALF_LENGTH,
        );
        let turn = Quat::from_rotation_y(tree_yaw(tree));
        let transform = Transform::new(site + axis - turn * axis, turn);
        let ids = tree as u64 * fiber_stride;
        for original in base.tree.model.fibers() {
            let mut f = original.clone();
            f.id += ids;
            f.rest = transform.compose(&f.rest);
            f.motion.pose = transform.compose(&f.motion.pose);
            f.motion.linear_velocity = transform.rotation * f.motion.linear_velocity;
            f.motion.angular_velocity = transform.rotation * f.motion.angular_velocity;
            if let Some(pin) = &mut f.pin {
                pin.pose = transform.compose(&pin.pose);
            }
            fibers.push(f);
        }
        for original in base.tree.model.bonds() {
            let mut b = original.clone();
            if !b.fibers.iter().any(|id| base.twigs.contains(id)) {
                b.material.rotation.x *= flexibility;
                b.material.rotation.y *= flexibility;
            }
            b.id += tree as u64 * bond_stride;
            b.fibers = b.fibers.map(|id| id + ids);
            bonds.push(b);
        }
        trunk_cells.extend(base.tree.trunk_cells.iter().map(|id| id + ids));
        limb_cells.extend(
            base.tree
                .limb_cells
                .iter()
                .map(|path| path.iter().map(|id| id + ids).collect()),
        );
        twigs.extend(base.twigs.iter().map(|id| id + ids));
        for original in &base.leaves {
            let mut leaf = original.clone();
            leaf.owner += ids;
            leaf.rest = transform.compose(&leaf.rest);
            leaf.released_direction = transform.rotation * leaf.released_direction;
            leaf.released_normal = transform.rotation * leaf.released_normal;
            leaf.tint = (leaf.tint + tree as f64 * 0.137).fract();
            leaves.push(leaf);
        }
        for original in &base.particles {
            let mut particle = *original;
            particle.position = transform.transform_point(particle.position);
            particle.velocity = transform.rotation * particle.velocity;
            particles.push(particle);
        }
        attachments.extend_from_slice(&base.attachments);
    }
    Ok(Canopy {
        tree: TreeModel {
            model: StrandModel::new(fibers, bonds)?,
            trunk_cells,
            limb_cells,
            tree_id: format!("AppleQSM orchard · 2 rows × {TREES_PER_ROW} trees"),
            source_limbs: base.tree.source_limbs * TREE_COUNT,
            simulated_limbs: base.tree.simulated_limbs * TREE_COUNT,
        },
        twigs,
        leaves,
        attachments,
        particles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_solvers::strands::Fiber;
    #[test]
    fn flexibility_changes_structural_bending_not_leafy_twigs_or_strength() {
        let base = leafy_model::build_with_foliage(0.010, 0.100).unwrap();
        let flexible = build().unwrap();
        for (old, new) in base
            .tree
            .model
            .bonds()
            .iter()
            .zip(flexible.tree.model.bonds())
        {
            let scale = if old.fibers.iter().any(|id| base.twigs.contains(id)) {
                1.0
            } else {
                WOOD_FLEXIBILITY
            };
            assert_eq!(
                new.material.rotation,
                old.material.rotation * Vec3::new(scale, scale, 1.0)
            );
            let mut material = new.material;
            material.rotation = old.material.rotation;
            assert_eq!(material, old.material);
        }
        assert!(build_with_flexibility(f64::NAN).is_err());
        assert!(build_with_flexibility(0.0).is_err());
    }
    #[test]
    fn trees_turn_about_their_trunks_without_touching_neighbours() {
        let base = leafy_model::build_with_foliage(0.010, 0.100).unwrap();
        let c = build().unwrap();
        let per_tree = base.particles.len();
        let cells = |tree: usize| &c.tree.model.fibers()[tree * 120..(tree + 1) * 120];
        let flat = Vec3::new(1.0, 0.0, 1.0);
        let root = |fibers: &[Fiber]| fibers.iter().find(|f| f.pin.is_some()).unwrap().center();
        let origin = root(base.tree.model.fibers()) * flat;
        let mut quadrants = std::collections::BTreeSet::new();
        for tree in 0..TREE_COUNT {
            let turn = Quat::from_rotation_y(tree_yaw(tree));
            quadrants.insert((tree_yaw(tree) / std::f64::consts::FRAC_PI_2) as usize);
            let trunk = root(cells(tree)) * flat;
            for (k, p) in base.particles.iter().enumerate().step_by(97) {
                let expected = trunk + turn * (p.position - origin);
                assert!(c.particles[tree * per_tree + k].position.distance(expected) < 1e-9);
            }
        }
        assert_eq!(quadrants.len(), 4);
        let mut gap = f64::INFINITY;
        for a in 0..TREE_COUNT {
            for b in a + 1..TREE_COUNT {
                for (p, q) in cells(a).iter().flat_map(|p| cells(b).iter().map(move |q| (p, q))) {
                    let d = p.center() - q.center();
                    gap = gap.min(d.x.hypot(d.z));
                }
            }
        }
        assert!(gap > 0.4, "closest cells of neighbouring trees {gap} m apart");
    }

    #[test]
    fn orchard_has_independent_rooted_trees_and_owned_leaves() {
        let c = build().unwrap();
        assert_eq!(c.tree.model.components().len(), TREE_COUNT);
        assert_eq!(c.tree.model.fibers().len(), 120 * TREE_COUNT);
        assert!((900 * TREE_COUNT..1100 * TREE_COUNT).contains(&c.leaves.len()));
        let sparse = leafy_model::build().unwrap();
        let area = |leaves: &[crate::leafy_model::Leaf]| {
            leaves.iter().map(|l| l.length.powi(2)).sum::<f64>()
        };
        assert!(area(&c.leaves) > 3.5 * TREE_COUNT as f64 * area(&sparse.leaves));
        let total_mass = |c: &Canopy| c.attachments.iter().map(|a| a.mass).sum::<f64>();
        let total_drag = |c: &Canopy| c.attachments.iter().map(|a| a.drag).sum::<f64>();
        assert!((0.9..1.1).contains(&(total_mass(&c) / (TREE_COUNT as f64 * total_mass(&sparse)))));
        assert!((0.9..1.1).contains(&(total_drag(&c) / (TREE_COUNT as f64 * total_drag(&sparse)))));
        println!(
            "orchard leaves={} area_ratio={}",
            c.leaves.len(),
            area(&c.leaves) / (TREE_COUNT as f64 * area(&sparse.leaves))
        );
        assert_eq!(
            c.tree
                .model
                .fibers()
                .iter()
                .filter(|f| f.pin.is_some())
                .count(),
            TREE_COUNT
        );
        for (leaf, particle) in c.leaves.iter().zip(&c.particles) {
            let f = c.tree.model.fiber(leaf.owner).unwrap();
            assert!(
                f.rest
                    .transform_point(leaf.anchor + leaf.offset)
                    .distance(particle.position)
                    < 1e-12
            );
        }
        let roots: Vec<_> = c
            .tree
            .model
            .fibers()
            .iter()
            .filter(|f| f.pin.is_some())
            .collect();
        for row in roots.chunks(TREES_PER_ROW) {
            for pair in row.windows(2) {
                assert!((pair[1].center().z - pair[0].center().z - TREE_SPACING).abs() < 1e-9);
            }
            assert!(row.last().unwrap().center().z - row[0].center().z > 16.0);
        }
    }
}
