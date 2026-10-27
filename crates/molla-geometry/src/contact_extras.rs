//! Contact warm-starting / matching + voxel-AABB reduction.
//!
//! **Matching**: hash each contact's `(shape_a, shape_b, voxel)` and
//! look it up in the previous frame's table. A match means the
//! contact persists; the application (or solver) can carry forward
//! data like the XPBD `λ_n` for warm-starting.
//!
//! **Reduction**: voxel-bin contact points and keep the deepest
//! contact per voxel. Stops a 4-corner box-plane stack from emitting
//! 4 redundant contacts when 1 (or even 0, if the neighbour has
//! identical depth) would do.
//!
//! Both passes operate on `Contacts` after the narrow-phase has
//! filled it. Designed to slot in between
//! `pipeline.collide()` and the solver's `step()`.

use molla_core::Result;
use molla_math::{Real, Vec3};
use molla_sim::Contacts;

/// Stable identity hash for a contact, suitable for matching across
/// frames. Combines the (sorted) shape-id pair with a voxelised
/// contact-point key.
///
/// Two contacts on the same shape pair separated by < `voxel_size`
/// hash to the same id — i.e. they're treated as "the same persistent
/// contact" for warm-starting. Bigger `voxel_size` = more matches /
/// more numerical drift; smaller = fewer matches.
#[must_use]
pub fn contact_id(a_shape: u32, b_shape: u32, point: Vec3, voxel_size: Real) -> u64 {
    let (lo, hi) = if a_shape <= b_shape {
        (a_shape, b_shape)
    } else {
        (b_shape, a_shape)
    };
    let inv = if voxel_size > 0.0 {
        1.0 / voxel_size
    } else {
        1.0
    };
    let vx = (point.x * inv).floor() as i32;
    let vy = (point.y * inv).floor() as i32;
    let vz = (point.z * inv).floor() as i32;
    // Splat the i32 voxel coords into a u64 along with the shape pair.
    // Mix in primes to scatter buckets evenly. This is the same
    // pattern Newton's `geometry/contact_match.py` uses.
    let mut h: u64 = 14695981039346656037;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(1099511628211);
    };
    mix(lo as u64);
    mix(hi as u64);
    mix((vx as u32) as u64);
    mix((vy as u32) as u64);
    mix((vz as u32) as u64);
    h
}

/// Match `current` contacts against `previous` using `contact_id`.
/// Returns a vector indexed by `current` contact slot — `Some(i)`
/// means the matching previous contact is at index `i` in `previous`.
/// `None` means a brand-new contact this frame.
///
/// The application can then carry over per-contact state (XPBD
/// `λ_n`, friction direction, age) for warm-starting.
pub fn match_contacts(
    current: &Contacts,
    previous: &[(u64, usize)], // (id, slot) sorted by id
    voxel_size: Real,
) -> Result<Vec<Option<usize>>> {
    let n = current.count as usize;
    let mut out = vec![None; n];
    let a_idx = current.a_index.host()?;
    let b_idx = current.b_index.host()?;
    let a_shape = current.a_shape.host()?;
    let b_shape = current.b_shape.host()?;
    let point = current.point.host()?;
    for i in 0..n {
        let id = collider_contact_id(
            a_shape[i], b_shape[i], a_idx[i], b_idx[i], point[i], voxel_size,
        );
        // Binary search in the sorted previous list.
        match previous.binary_search_by_key(&id, |&(k, _)| k) {
            Ok(j) => out[i] = Some(previous[j].1),
            Err(_) => out[i] = None,
        }
    }
    Ok(out)
}

/// Build a sorted (id, slot) lookup table from a finished frame's
/// `Contacts`. Pass this to `match_contacts` next frame as the
/// `previous` argument.
pub fn build_contact_id_table(contacts: &Contacts, voxel_size: Real) -> Result<Vec<(u64, usize)>> {
    let n = contacts.count as usize;
    let a_idx = contacts.a_index.host()?;
    let b_idx = contacts.b_index.host()?;
    let a_shape = contacts.a_shape.host()?;
    let b_shape = contacts.b_shape.host()?;
    let point = contacts.point.host()?;
    let mut out: Vec<(u64, usize)> = (0..n)
        .map(|i| {
            (
                collider_contact_id(
                    a_shape[i], b_shape[i], a_idx[i], b_idx[i], point[i], voxel_size,
                ),
                i,
            )
        })
        .collect();
    out.sort_unstable_by_key(|&(k, _)| k);
    Ok(out)
}

fn collider_contact_id(
    a_shape: u32,
    b_shape: u32,
    a_body: u32,
    b_body: u32,
    point: Vec3,
    voxel: Real,
) -> u64 {
    if a_shape == u32::MAX && b_shape == u32::MAX {
        contact_id(a_body, b_body, point, voxel)
    } else {
        contact_id(a_shape, b_shape, point, voxel)
    }
}

/// Voxel-AABB pruning: bin all contacts into a uniform grid keyed
/// by `(a_shape, b_shape, voxel(point))` and keep only the
/// **deepest** contact per voxel. Contacts are reorganised in
/// place; `contacts.count` is updated to the kept count.
///
/// `voxel_size` controls how aggressive pruning is: ~`2 ×
/// max_particle_radius` for cloth-particle scenes, or
/// `~contact_margin` for rigid-body face contacts.
pub fn reduce_contacts(contacts: &mut Contacts, voxel_size: Real) -> Result<usize> {
    let n = contacts.count as usize;
    if n <= 1 {
        return Ok(n);
    }

    // Collect (id, depth, original_index) and pick the deepest per id.
    let a_idx_src = contacts.a_index.host()?.to_vec();
    let b_idx_src = contacts.b_index.host()?.to_vec();
    let a_shape_src = contacts.a_shape.host()?.to_vec();
    let b_shape_src = contacts.b_shape.host()?.to_vec();
    let point_src = contacts.point.host()?.to_vec();
    let normal_src = contacts.normal.host()?.to_vec();
    let depth_src = contacts.depth.host()?.to_vec();
    let fric_src = contacts.friction.host()?.to_vec();
    let stat_src = contacts.static_friction.host()?.to_vec();
    let roll_src = contacts.rolling_friction.host()?.to_vec();
    let tors_src = contacts.torsional_friction.host()?.to_vec();
    let rest_src = contacts.restitution.host()?.to_vec();

    // Hash → (best_depth, best_idx).
    use std::collections::HashMap;
    let mut keep: HashMap<u64, (Real, usize)> = HashMap::with_capacity(n);
    for i in 0..n {
        let id = collider_contact_id(
            a_shape_src[i],
            b_shape_src[i],
            a_idx_src[i],
            b_idx_src[i],
            point_src[i],
            voxel_size,
        );
        let d = depth_src[i];
        keep.entry(id)
            .and_modify(|(best_d, best_i)| {
                if d > *best_d {
                    *best_d = d;
                    *best_i = i;
                }
            })
            .or_insert((d, i));
    }

    let mut kept_indices: Vec<usize> = keep.into_values().map(|(_, i)| i).collect();
    // Stable order so warm-start matching remains deterministic.
    kept_indices.sort_unstable();
    let n_keep = kept_indices.len();

    let a_idx = contacts.a_index.host_mut()?;
    let b_idx = contacts.b_index.host_mut()?;
    let a_shape = contacts.a_shape.host_mut()?;
    let b_shape = contacts.b_shape.host_mut()?;
    for (slot, &src) in kept_indices.iter().enumerate() {
        a_idx[slot] = a_idx_src[src];
        b_idx[slot] = b_idx_src[src];
        a_shape[slot] = a_shape_src[src];
        b_shape[slot] = b_shape_src[src];
    }
    let point = contacts.point.host_mut()?;
    for (slot, &src) in kept_indices.iter().enumerate() {
        point[slot] = point_src[src];
    }
    let normal = contacts.normal.host_mut()?;
    for (slot, &src) in kept_indices.iter().enumerate() {
        normal[slot] = normal_src[src];
    }
    let depth = contacts.depth.host_mut()?;
    for (slot, &src) in kept_indices.iter().enumerate() {
        depth[slot] = depth_src[src];
    }
    let fric = contacts.friction.host_mut()?;
    for (slot, &src) in kept_indices.iter().enumerate() {
        fric[slot] = fric_src[src];
    }
    let stat = contacts.static_friction.host_mut()?;
    for (slot, &src) in kept_indices.iter().enumerate() {
        stat[slot] = stat_src[src];
    }
    let roll = contacts.rolling_friction.host_mut()?;
    for (slot, &src) in kept_indices.iter().enumerate() {
        roll[slot] = roll_src[src];
    }
    let tors = contacts.torsional_friction.host_mut()?;
    for (slot, &src) in kept_indices.iter().enumerate() {
        tors[slot] = tors_src[src];
    }
    let rest = contacts.restitution.host_mut()?;
    for (slot, &src) in kept_indices.iter().enumerate() {
        rest[slot] = rest_src[src];
    }
    contacts.count = n_keep as u32;
    if !contacts.body_contact_offsets.is_empty() {
        let old_offsets = contacts.body_contact_offsets.host()?.to_vec();
        let old_indices = contacts.body_contact_indices.host()?.to_vec();
        let mut mapping = vec![u32::MAX; n];
        for (slot, &src) in kept_indices.iter().enumerate() {
            mapping[src] = slot as u32;
        }
        let offsets = contacts.body_contact_offsets.host_mut()?;
        let indices = contacts.body_contact_indices.host_mut()?;
        let mut cursor = 0;
        for body in 0..old_offsets.len() - 1 {
            offsets[body] = cursor as u32;
            for &packed in &old_indices[old_offsets[body] as usize..old_offsets[body + 1] as usize]
            {
                let (index, b_side) = Contacts::decode_index(packed);
                let target = mapping[index as usize];
                if target != u32::MAX {
                    indices[cursor] = target | if b_side { Contacts::B_SIDE_BIT } else { 0 };
                    cursor += 1;
                }
            }
        }
        *offsets.last_mut().unwrap() = cursor as u32;
    }
    Ok(n_keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_sim::Contacts;

    #[test]
    fn contact_id_stable_under_pair_swap() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let id1 = contact_id(3, 7, p, 0.05);
        let id2 = contact_id(7, 3, p, 0.05);
        assert_eq!(id1, id2);
    }

    #[test]
    fn contact_id_changes_with_voxel() {
        // Voxel size 0.1 → quantises to 1 mm chunks. Two points
        // 0.05 apart fall in different voxels.
        let p1 = Vec3::new(0.0, 0.0, 0.0);
        let p2 = Vec3::new(0.15, 0.0, 0.0);
        let id1 = contact_id(0, 1, p1, 0.1);
        let id2 = contact_id(0, 1, p2, 0.1);
        assert_ne!(id1, id2);
    }

    fn fill_contacts(contacts: &mut Contacts, entries: &[(u32, u32, Vec3, Real)]) {
        let a_idx = contacts.a_index.host_mut().unwrap();
        let b_idx = contacts.b_index.host_mut().unwrap();
        for (i, e) in entries.iter().enumerate() {
            a_idx[i] = e.0;
            b_idx[i] = e.1;
        }
        let pt = contacts.point.host_mut().unwrap();
        let dp = contacts.depth.host_mut().unwrap();
        for (i, e) in entries.iter().enumerate() {
            pt[i] = e.2;
            dp[i] = e.3;
        }
        contacts.count = entries.len() as u32;
    }

    #[test]
    fn reduce_keeps_deepest_per_voxel() {
        let mut c = Contacts::with_capacity(8);
        // 4 contacts on the same pair, all in the same voxel,
        // varying depths. Reducer should keep only the deepest.
        fill_contacts(
            &mut c,
            &[
                (0, 1, Vec3::new(0.01, 0.0, 0.0), 0.005),
                (0, 1, Vec3::new(0.02, 0.0, 0.0), 0.020),
                (0, 1, Vec3::new(0.03, 0.0, 0.0), 0.010),
                (0, 1, Vec3::new(0.04, 0.0, 0.0), 0.015),
            ],
        );
        let kept = reduce_contacts(&mut c, 0.5).unwrap();
        assert_eq!(kept, 1);
        let depth_kept = c.depth.host().unwrap()[0];
        assert!((depth_kept - 0.020).abs() < 1e-12, "got {depth_kept}");
    }

    #[test]
    fn reduce_preserves_distinct_pairs() {
        let mut c = Contacts::with_capacity(8);
        // 3 contacts on different pairs — none should be deduped.
        fill_contacts(
            &mut c,
            &[
                (0, 1, Vec3::ZERO, 0.01),
                (2, 3, Vec3::ZERO, 0.02),
                (4, 5, Vec3::ZERO, 0.03),
            ],
        );
        let kept = reduce_contacts(&mut c, 0.5).unwrap();
        assert_eq!(kept, 3);
    }

    #[test]
    fn match_finds_persistent_contact() {
        let mut prev = Contacts::with_capacity(4);
        fill_contacts(&mut prev, &[(0, 1, Vec3::new(0.5, 0.0, 0.0), 0.01)]);
        let prev_table = build_contact_id_table(&prev, 0.1).unwrap();

        // Next frame: same pair, slightly drifted contact point but
        // still in the same voxel.
        let mut cur = Contacts::with_capacity(4);
        fill_contacts(&mut cur, &[(0, 1, Vec3::new(0.51, 0.0, 0.0), 0.012)]);
        let matches = match_contacts(&cur, &prev_table, 0.1).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], Some(0));
    }

    #[test]
    fn match_misses_drifted_contact() {
        let mut prev = Contacts::with_capacity(4);
        fill_contacts(&mut prev, &[(0, 1, Vec3::new(0.5, 0.0, 0.0), 0.01)]);
        let prev_table = build_contact_id_table(&prev, 0.05).unwrap();
        // Contact moved enough to land in a different voxel.
        let mut cur = Contacts::with_capacity(4);
        fill_contacts(&mut cur, &[(0, 1, Vec3::new(0.7, 0.0, 0.0), 0.012)]);
        let matches = match_contacts(&cur, &prev_table, 0.05).unwrap();
        assert_eq!(matches[0], None);
    }
}
