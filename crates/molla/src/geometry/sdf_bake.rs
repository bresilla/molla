//! Mesh → voxel signed-distance field baking.
//!
//! Stage-21 deliverable per `PLAN.md` §21 / `GAP.md` §4. Voxelises a
//! [`crate::sim::TriMesh`] into a 3D grid of signed distances suitable
//! for SDF-based contact (hydroelastic, Stage 22) and for the
//! marching-cubes round trip in [`super::marching_cubes`].
//!
//! The existing `SdfShape` enum is *analytic* (torus / sphere) — this
//! file introduces a separate **`SdfGrid`** type for sampled fields.
//! The two cohabit: analytic SDFs stay analytic (no voxelisation
//! noise), and arbitrary meshes get an `SdfGrid` here.
//!
//! Algorithm:
//!
//! 1. Compute the mesh's AABB, expand by `padding` (world-space
//!    metres), and divide into a `resolution × resolution × resolution`
//!    voxel grid.
//! 2. For each grid sample, find the closest point on every triangle
//!    via the closest-point-on-triangle routine — keep the minimum
//!    unsigned distance.
//! 3. Determine sign via parity of an axis-aligned ray intersection
//!    count: odd → inside (negative), even → outside (positive).
//!    Robust enough for closed manifold meshes (the only kind we
//!    support); open meshes return ±d arbitrarily near the boundary.
//!
//! Performance: O(resolution³ × triangle_count). Fine for asset-load
//! workloads (32³ × 1k tris ≈ 33M ops, ms-range). Stage 21.1
//! parallelises the closest-distance pass with rayon — voxels are
//! independent so the fan-out is embarrassingly parallel and the
//! resulting `data` array is byte-deterministic across runs (each
//! voxel's `min` reduction sees the same triangles in the same order
//! within its task). The sign pass remains sequential because it
//! threads parity state along +X across cells in a row.

use crate::math::{Real, Vec3};
use crate::sim::TriMesh;
use super::mesh_bvh::{BvhNode, MeshBvh};
use rayon::prelude::*;

/// Sampled signed-distance field on a regular 3D grid.
///
/// Layout: `data[x + y * nx + z * nx * ny]` for `(x, y, z)` in
/// `[0, nx) × [0, ny) × [0, nz)`. Sample positions are
/// `origin + (x, y, z) * cell_size`. Grids are always cubic
/// (`nx == ny == nz`) when produced by [`bake_mesh_to_sdf`], but the
/// type stores the three axes independently to leave room for
/// non-uniform fields later.
#[derive(Debug, Clone)]
pub struct SdfGrid {
    /// Number of samples along each axis.
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    /// World-space corner of the `(0, 0, 0)` sample.
    pub origin: Vec3,
    /// Sample spacing along all axes (uniform for now).
    pub cell_size: Real,
    /// Flat distance array, length `nx * ny * nz`. Negative inside,
    /// positive outside.
    pub data: Vec<Real>,
}

impl SdfGrid {
    /// Total sample count.
    pub fn len(&self) -> usize {
        (self.nx as usize) * (self.ny as usize) * (self.nz as usize)
    }

    /// Empty (no samples) — shouldn't happen in practice but cheap to test.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sample at integer voxel coordinates. Out-of-range → `+∞`
    /// (treated as "far outside" by callers).
    pub fn at(&self, x: u32, y: u32, z: u32) -> Real {
        if x >= self.nx || y >= self.ny || z >= self.nz {
            return Real::INFINITY;
        }
        self.data[(x as usize) + (y as usize) * (self.nx as usize)
            + (z as usize) * (self.nx as usize) * (self.ny as usize)]
    }

    /// World-space position of voxel `(x, y, z)`.
    pub fn voxel_pos(&self, x: u32, y: u32, z: u32) -> Vec3 {
        self.origin
            + Vec3::new(
                (x as Real) * self.cell_size,
                (y as Real) * self.cell_size,
                (z as Real) * self.cell_size,
            )
    }
}

/// Voxelise a mesh into a signed-distance grid. `resolution` is the
/// number of samples along the longest axis of the (padded) AABB;
/// other axes get the same number of samples for a cubic grid (the
/// extents of unused border cells are pulled in by `padding`). For
/// strict cubic alignment with no degenerate axes the implementation
/// always lays down `resolution × resolution × resolution` samples.
///
/// `padding` is the world-space margin to add on every side of the
/// mesh AABB so the surface sits comfortably away from the grid edges
/// — typically `cell_size` or `2 × cell_size`.
pub fn bake_mesh_to_sdf(mesh: &TriMesh, resolution: usize, padding: Real) -> SdfGrid {
    debug_assert!(resolution >= 2, "bake_mesh_to_sdf: resolution must be ≥ 2");
    let res = resolution.max(2) as u32;

    if mesh.indices.is_empty() || mesh.vertices.is_empty() {
        return SdfGrid {
            nx: res,
            ny: res,
            nz: res,
            origin: Vec3::ZERO,
            cell_size: 1.0,
            data: vec![Real::INFINITY; (res * res * res) as usize],
        };
    }

    // ── AABB + cubic envelope ─────────────────────────────────────
    let (mut mn, mut mx) = mesh.vertices.iter().fold(
        (Vec3::splat(Real::INFINITY), Vec3::splat(-Real::INFINITY)),
        |(a, b), v| (a.min(*v), b.max(*v)),
    );
    mn -= Vec3::splat(padding);
    mx += Vec3::splat(padding);
    let extent = mx - mn;
    let max_extent = extent.x.max(extent.y).max(extent.z);
    // Cubic AABB centred on the mesh AABB (so the grid is symmetric).
    let centre = (mn + mx) * 0.5;
    let half = Vec3::splat(max_extent * 0.5);
    let mn = centre - half;
    let cell_size = max_extent / (res as Real - 1.0);

    let total = (res * res * res) as usize;

    // ── BVH-accelerated closest-distance pass ─────────────────────
    // Without a BVH the closest-point pass is O(R³ × T) which kills
    // 64³ × 10k-tri budgets (≥ 1.5 s even with rayon). With a BVH we
    // prune sibling subtrees whose AABB is farther than the running
    // best; in practice this drops the per-voxel triangle test count
    // from O(T) to O(log T) once the running `best` shrinks below
    // the bounding-box diagonal.
    //
    // `mesh.bvh` is the existing per-mesh AABB BVH built by
    // `TriMesh::build_bvh`; we always seed our own here in case the
    // caller forgot. Cheap (mesh sizes we target are ≤ 50k tris).
    let local_bvh: MeshBvh;
    let bvh: &MeshBvh = if !mesh.bvh.nodes.is_empty() {
        &mesh.bvh
    } else {
        local_bvh = MeshBvh::build(&mesh.vertices, &mesh.indices, 4);
        &local_bvh
    };

    let res_us = res as usize;
    let mut data: Vec<Real> = (0..total)
        .into_par_iter()
        .map(|idx| {
            let x = idx % res_us;
            let y = (idx / res_us) % res_us;
            let z = idx / (res_us * res_us);
            let p = mn
                + Vec3::new(
                    (x as Real) * cell_size,
                    (y as Real) * cell_size,
                    (z as Real) * cell_size,
                );
            let best2 = closest_point_distance_squared_bvh(bvh, &mesh.vertices, &mesh.indices, p);
            best2.sqrt()
        })
        .collect();

    // ── Sign pass — parity of axis-aligned ray intersections ─────
    // Cast +X for every (y, z) row and toggle sign whenever a
    // triangle crosses. Works robustly for closed manifold meshes.
    //
    // The y/z coordinates of every row are perturbed by an irrational
    // multiple of cell_size so the rays never run along a triangle's
    // edge (which would double-count crossings in a manifold mesh).
    // This is the standard "shoot a ray off-axis" trick used in
    // GVDB / OpenVDB voxelisers.
    //
    // Stage 21.1: rows are independent; we fan out across (y, z) pairs
    // with rayon and write directly into row-aligned slices of `data`
    // with `par_chunks_mut(res)`. Each slice covers one (y, z) row of
    // `res` x-cells, and the iteration is in flat-index order so the
    // mapping back to (y, z) is `chunk_idx → (y = chunk_idx % res,
    // z = chunk_idx / res)`.
    let ray_back: Real = 1.0e6;
    let nudge_y: Real = cell_size * 1.0e-3 * std::f64::consts::SQRT_2;
    let nudge_z: Real = cell_size * 1.0e-3 * std::f64::consts::PI / 4.0;
    data.par_chunks_mut(res_us)
        .enumerate()
        .for_each(|(chunk_idx, row)| {
            let y = chunk_idx % res_us;
            let z = chunk_idx / res_us;
            let row_y = mn.y + (y as Real) * cell_size + nudge_y;
            let row_z = mn.z + (z as Real) * cell_size + nudge_z;
            let p_origin = Vec3::new(mn.x - ray_back, row_y, row_z);
            let mut hits: Vec<Real> = Vec::new();
            for tri in &mesh.indices {
                let v0 = mesh.vertices[tri[0] as usize];
                let v1 = mesh.vertices[tri[1] as usize];
                let v2 = mesh.vertices[tri[2] as usize];
                if let Some(t) = ray_x_triangle(p_origin, v0, v1, v2) {
                    hits.push(t);
                }
            }
            hits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut hit_idx = 0;
            let mut inside = false;
            for (x, cell) in row.iter_mut().enumerate() {
                let p_x = ray_back + (x as Real) * cell_size;
                while hit_idx < hits.len() && hits[hit_idx] <= p_x {
                    inside = !inside;
                    hit_idx += 1;
                }
                if inside {
                    *cell = -*cell;
                }
            }
        });

    SdfGrid {
        nx: res,
        ny: res,
        nz: res,
        origin: mn,
        cell_size,
        data,
    }
}

/// Squared distance from `p` to the closest point on the mesh,
/// pruned via the per-mesh AABB BVH. Returns `+∞` for an empty mesh.
///
/// Standard "best-first" traversal: descend the BVH child whose AABB
/// is closer to `p` first, skipping any node whose AABB-to-point
/// distance exceeds the running best. The order in which sibling
/// nodes are visited is fixed (closer first → if equal, child whose
/// AABB has lower index first) so this routine is deterministic.
fn closest_point_distance_squared_bvh(
    bvh: &MeshBvh,
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    p: Vec3,
) -> Real {
    if bvh.nodes.is_empty() {
        return Real::INFINITY;
    }
    let mut best2 = Real::INFINITY;
    // Stack of (node_idx, aabb_dist²). Pruning skips entries whose
    // bound is already worse than `best2`.
    let mut stack: Vec<(u32, Real)> = Vec::with_capacity(64);
    stack.push((0, aabb_point_distance_squared(bvh.nodes[0].aabb_min, bvh.nodes[0].aabb_max, p)));
    while let Some((idx, dist2)) = stack.pop() {
        if dist2 >= best2 {
            continue;
        }
        let node: &BvhNode = &bvh.nodes[idx as usize];
        if node.is_leaf == 1 {
            let lo = node.prim_start as usize;
            let hi = lo + node.tri_count as usize;
            for &tri_idx in &bvh.triangle_order[lo..hi] {
                let tri = indices[tri_idx as usize];
                let v0 = vertices[tri[0] as usize];
                let v1 = vertices[tri[1] as usize];
                let v2 = vertices[tri[2] as usize];
                let d2 = closest_point_on_triangle(p, v0, v1, v2).0;
                if d2 < best2 {
                    best2 = d2;
                }
            }
        } else {
            let l = node.left;
            let r = node.right;
            let dl = aabb_point_distance_squared(
                bvh.nodes[l as usize].aabb_min,
                bvh.nodes[l as usize].aabb_max,
                p,
            );
            let dr = aabb_point_distance_squared(
                bvh.nodes[r as usize].aabb_min,
                bvh.nodes[r as usize].aabb_max,
                p,
            );
            // Push the farther child first so the closer one pops
            // next — this lets `best2` tighten as quickly as
            // possible, maximising prune.
            if dl < dr {
                stack.push((r, dr));
                stack.push((l, dl));
            } else {
                stack.push((l, dl));
                stack.push((r, dr));
            }
        }
    }
    best2
}

/// Squared distance from `p` to the closest point on AABB
/// `[mn, mx]`. Zero if `p` is inside.
fn aabb_point_distance_squared(mn: Vec3, mx: Vec3, p: Vec3) -> Real {
    let dx = (mn.x - p.x).max(0.0).max(p.x - mx.x);
    let dy = (mn.y - p.y).max(0.0).max(p.y - mx.y);
    let dz = (mn.z - p.z).max(0.0).max(p.z - mx.z);
    dx * dx + dy * dy + dz * dz
}

/// Squared distance from `p` to the closest point on triangle
/// `(a, b, c)`. Returns `(d², closest_point)`.
///
/// Algorithm: barycentric region classification — Ericson, "Real-Time
/// Collision Detection", §5.1.5.
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Real, Vec3) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ((p - a).length_squared(), a);
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return ((p - b).length_squared(), b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q = a + ab * v;
        return ((p - q).length_squared(), q);
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return ((p - c).length_squared(), c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q = a + ac * w;
        return ((p - q).length_squared(), q);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let q = b + (c - b) * w;
        return ((p - q).length_squared(), q);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q = a + ab * v + ac * w;
    ((p - q).length_squared(), q)
}

/// +X ray-vs-triangle. Returns hit `t` (so the world-space hit point
/// is `origin + t * Vec3::X`) for a positive crossing, else `None`.
///
/// Standard Möller-Trumbore. Edge / vertex grazing is handled by the
/// caller by nudging the ray's y/z slightly off the grid axis (see
/// [`bake_mesh_to_sdf`]) — that's cheaper and more robust than trying
/// to define a half-open inclusion convention here.
fn ray_x_triangle(origin: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<Real> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = Vec3::X.cross(edge2);
    let a = edge1.dot(h);
    if a.abs() < 1.0e-14 {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(edge1);
    let v = f * Vec3::X.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * edge2.dot(q);
    if t <= 0.0 {
        return None;
    }
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_mesh() -> TriMesh {
        let v = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let i = vec![
            [0, 2, 1], [0, 3, 2], // -Z (CCW from outside)
            [4, 5, 6], [4, 6, 7], // +Z
            [0, 1, 5], [0, 5, 4], // -Y
            [3, 6, 2], [3, 7, 6], // +Y
            [0, 7, 3], [0, 4, 7], // -X
            [1, 2, 6], [1, 6, 5], // +X
        ];
        let mut m = TriMesh {
            vertices: v,
            indices: i,
            ..Default::default()
        };
        m.build_bvh(4);
        m
    }

    #[test]
    fn bake_cube_centre_is_negative() {
        let m = unit_cube_mesh();
        let sdf = bake_mesh_to_sdf(&m, 16, 0.25);
        // The grid centre is at the cube centre; SDF should be ~ -0.5
        // (distance from centre to nearest face).
        let cx = sdf.nx / 2;
        let cy = sdf.ny / 2;
        let cz = sdf.nz / 2;
        let d = sdf.at(cx, cy, cz);
        assert!(d < 0.0, "centre SDF must be negative, got {d}");
        assert!(d.abs() > 0.3, "centre SDF magnitude should be ~0.5, got {d}");
    }

    #[test]
    fn bake_cube_corner_is_positive() {
        let m = unit_cube_mesh();
        let sdf = bake_mesh_to_sdf(&m, 16, 0.25);
        // The grid origin (corner) sits well outside the padded cube
        // — SDF must be positive.
        let d = sdf.at(0, 0, 0);
        assert!(d > 0.0, "corner SDF must be positive, got {d}");
    }
}
