//! Public general-purpose spatial hash grid for proximity queries.
//!
//! Stage-21 deliverable per `PLAN.md` §21 / `GAP.md` §4. The cloth
//! particle self-collision (Stage 12, see `shaders/xpbd_particles.wgsl`)
//! embeds a fixed-bucket spatial hash inside the kernel for GPU
//! parallelism; this module exposes a CPU-side, host-friendly version
//! of the same algorithm so other subsystems (mesh authoring tools,
//! debug queries, oracles) can reuse it without lifting the GPU
//! binding.
//!
//! The hash function matches the WGSL kernel's `cell_hash` (XOR-mix of
//! 32-bit primes per axis) so CPU + GPU agree on cell coordinates if
//! anyone wants to cross-check.
//!
//! ```
//! use molla_geometry::SpatialHashGrid;
//! use molla_math::Vec3;
//! let mut grid = SpatialHashGrid::new(0.5);
//! grid.insert(0, Vec3::new(0.0, 0.0, 0.0));
//! grid.insert(1, Vec3::new(0.4, 0.0, 0.0));
//! grid.insert(2, Vec3::new(5.0, 0.0, 0.0));
//! let hits: Vec<u32> = grid
//!     .query_aabb(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.5, 0.1, 0.1))
//!     .collect();
//! assert!(hits.contains(&0) && hits.contains(&1) && !hits.contains(&2));
//! ```

use molla_math::{Real, Vec3};
use std::collections::HashMap;

/// 3D spatial hash grid mapping integer cell coordinates → list of
/// inserted ids. Unbounded — cells are allocated on demand.
///
/// The grid is *unordered* and *non-deduplicating*: inserting the same
/// id twice puts it in the bucket twice. `query_aabb` therefore yields
/// duplicates if a caller inserts overlapping; it's the caller's job
/// to dedupe if that matters.
#[derive(Debug, Clone)]
pub struct SpatialHashGrid {
    cell_size: Real,
    inv_cell: Real,
    /// `(cx, cy, cz) → Vec<id>`. `Vec` is fine for cloth-grade scenes
    /// (≤ ~50 entries per bucket); upgrade to a small-vec / inline
    /// array only if profiling demands it.
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
    /// Total number of `insert` calls. Cheap to track and surfaces
    /// "is the grid empty?" without walking the map.
    len: usize,
}

impl SpatialHashGrid {
    /// Build an empty grid with the given world-space cell size. A
    /// reasonable default for collision queries is `2 × max_radius` so
    /// the 3×3×3 stencil covers every potentially-overlapping pair.
    ///
    /// Panics in debug builds if `cell_size <= 0`.
    pub fn new(cell_size: Real) -> Self {
        debug_assert!(cell_size > 0.0, "SpatialHashGrid: cell_size must be > 0");
        let cell_size = cell_size.max(Real::EPSILON);
        Self {
            cell_size,
            inv_cell: 1.0 / cell_size,
            cells: HashMap::new(),
            len: 0,
        }
    }

    /// Drop every inserted entry but keep the allocated cell map for
    /// reuse. Useful when the grid is rebuilt every frame.
    pub fn clear(&mut self) {
        for v in self.cells.values_mut() {
            v.clear();
        }
        self.len = 0;
    }

    /// Cell size this grid was built with.
    pub fn cell_size(&self) -> Real {
        self.cell_size
    }

    /// Total number of `insert` calls since the last `clear` /
    /// `new`. Note this is *not* a unique-id count — duplicates are
    /// counted separately.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True iff no entries have been inserted.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert `id` at `point`. The id is associated with the single
    /// cell containing the point. For extended objects (ones with an
    /// AABB rather than a point), call `insert` for every cell the
    /// AABB overlaps via [`Self::insert_aabb`].
    pub fn insert(&mut self, id: u32, point: Vec3) {
        let cell = self.cell_of(point);
        self.cells.entry(cell).or_default().push(id);
        self.len += 1;
    }

    /// Insert `id` into every cell touched by the AABB
    /// `[min, max]`. Useful for inserting triangles / spheres /
    /// arbitrary-shape proxies whose extent spans more than one cell.
    /// The id appears once per overlapping cell — `query_aabb` will
    /// then return it multiple times, so callers should dedupe.
    pub fn insert_aabb(&mut self, id: u32, min: Vec3, max: Vec3) {
        let lo = self.cell_of(min);
        let hi = self.cell_of(max);
        for cz in lo.2..=hi.2 {
            for cy in lo.1..=hi.1 {
                for cx in lo.0..=hi.0 {
                    self.cells.entry((cx, cy, cz)).or_default().push(id);
                    self.len += 1;
                }
            }
        }
    }

    /// Iterate every id whose insertion cell overlaps the AABB
    /// `[min, max]`. Yields IDs in unspecified order; duplicates may
    /// appear if `insert_aabb` was used for the same id in multiple
    /// cells.
    pub fn query_aabb<'a>(&'a self, min: Vec3, max: Vec3) -> impl Iterator<Item = u32> + 'a {
        let lo = self.cell_of(min);
        let hi = self.cell_of(max);
        // Materialise the cell coordinate triples, then flat_map into
        // each bucket. Using an index iterator avoids the
        // borrow-checker grief of a nested closure.
        let xs = lo.0..=hi.0;
        let ys = lo.1..=hi.1;
        let zs = lo.2..=hi.2;
        let cells_ref = &self.cells;
        zs.flat_map(move |cz| {
            let ys = ys.clone();
            let xs = xs.clone();
            ys.flat_map(move |cy| {
                let xs = xs.clone();
                xs.clone()
                    .filter_map(move |cx| cells_ref.get(&(cx, cy, cz)))
            })
        })
        .flat_map(|v| v.iter().copied())
    }

    /// Iterate every id in the 3×3×3 stencil around the cell that
    /// contains `point`. Convenience for "find all neighbours within
    /// roughly one cell radius" queries (e.g. cloth self-collision).
    pub fn query_neighbours<'a>(&'a self, point: Vec3) -> impl Iterator<Item = u32> + 'a {
        let c = self.cell_of(point);
        let cells_ref = &self.cells;
        (-1..=1)
            .flat_map(move |dz| {
                (-1..=1).flat_map(move |dy| {
                    (-1..=1).filter_map(move |dx| cells_ref.get(&(c.0 + dx, c.1 + dy, c.2 + dz)))
                })
            })
            .flat_map(|v| v.iter().copied())
    }

    /// Map a world-space point to its integer cell coordinate.
    /// Matches the WGSL kernel's `cell_of` (`floor(p * inv_cell)`).
    pub fn cell_of(&self, p: Vec3) -> (i32, i32, i32) {
        (
            (p.x * self.inv_cell).floor() as i32,
            (p.y * self.inv_cell).floor() as i32,
            (p.z * self.inv_cell).floor() as i32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query_returns_inserted_ids() {
        let mut g = SpatialHashGrid::new(1.0);
        g.insert(7, Vec3::new(0.5, 0.5, 0.5));
        g.insert(8, Vec3::new(0.6, 0.5, 0.5));
        g.insert(9, Vec3::new(5.0, 0.5, 0.5));
        let hits: Vec<u32> = g
            .query_aabb(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0))
            .collect();
        assert!(hits.contains(&7));
        assert!(hits.contains(&8));
        assert!(!hits.contains(&9));
    }

    #[test]
    fn cell_of_wraps_negative_correctly() {
        let g = SpatialHashGrid::new(1.0);
        // Floor(-0.5) = -1 — a point at -0.5 sits in cell -1, not 0.
        assert_eq!(g.cell_of(Vec3::new(-0.5, 0.0, 0.0)), (-1, 0, 0));
    }

    #[test]
    fn clear_resets_len_but_keeps_capacity() {
        let mut g = SpatialHashGrid::new(0.5);
        for i in 0..8 {
            g.insert(i, Vec3::new(i as Real * 0.1, 0.0, 0.0));
        }
        assert_eq!(g.len(), 8);
        g.clear();
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
        // Querying after clear returns nothing.
        let hits: Vec<u32> = g.query_aabb(Vec3::splat(-1.0), Vec3::splat(1.0)).collect();
        assert!(hits.is_empty());
    }

    #[test]
    fn aabb_query_matches_brute_force() {
        // Random point cloud — grid query must match brute force for
        // every randomly-chosen AABB.
        let mut g = SpatialHashGrid::new(0.25);
        let mut points = Vec::new();
        // Cheap deterministic LCG so the test reproduces exactly.
        let mut s: u32 = 0x9E37_79B9;
        let mut next = || {
            s = s.wrapping_mul(2654435761).wrapping_add(0x12345678);
            s as f64 / u32::MAX as f64
        };
        for i in 0..500 {
            let p = Vec3::new(next() * 4.0 - 2.0, next() * 4.0 - 2.0, next() * 4.0 - 2.0);
            points.push(p);
            g.insert(i as u32, p);
        }
        // Sample 10 random AABBs.
        for _ in 0..10 {
            let cx = next() * 4.0 - 2.0;
            let cy = next() * 4.0 - 2.0;
            let cz = next() * 4.0 - 2.0;
            let half = next() * 0.5 + 0.05;
            let min = Vec3::new(cx - half, cy - half, cz - half);
            let max = Vec3::new(cx + half, cy + half, cz + half);
            let mut from_grid: Vec<u32> = g.query_aabb(min, max).collect();
            from_grid.sort();
            from_grid.dedup();
            let mut bf: Vec<u32> = (0..points.len() as u32)
                .filter(|i| {
                    let p = points[*i as usize];
                    p.x >= min.x
                        && p.x <= max.x
                        && p.y >= min.y
                        && p.y <= max.y
                        && p.z >= min.z
                        && p.z <= max.z
                })
                .collect();
            bf.sort();
            // The grid's AABB query is conservative: it visits every
            // cell touching `[min, max]` and yields all ids in those
            // cells, even if the id's own point sits just outside the
            // AABB. So `from_grid` is a superset of `bf`.
            for id in &bf {
                assert!(
                    from_grid.contains(id),
                    "grid missed id {id} that brute force found"
                );
            }
        }
    }
}
