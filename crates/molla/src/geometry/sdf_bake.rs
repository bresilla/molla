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
//! workloads (32³ × 1k tris ≈ 33M ops, ms-range). The implementation
//! is single-threaded today; rayon-parallelisation is a one-line
//! change if it becomes a bottleneck.

use crate::math::{Real, Vec3};
use crate::sim::TriMesh;

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
    let mut data = vec![Real::INFINITY; total];

    // ── Closest-distance pass ──────────────────────────────────────
    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let p = mn
                    + Vec3::new(
                        (x as Real) * cell_size,
                        (y as Real) * cell_size,
                        (z as Real) * cell_size,
                    );
                let mut best = Real::INFINITY;
                for tri in &mesh.indices {
                    let v0 = mesh.vertices[tri[0] as usize];
                    let v1 = mesh.vertices[tri[1] as usize];
                    let v2 = mesh.vertices[tri[2] as usize];
                    let d2 = closest_point_on_triangle(p, v0, v1, v2).0;
                    if d2 < best {
                        best = d2;
                    }
                }
                let idx = (x as usize)
                    + (y as usize) * (res as usize)
                    + (z as usize) * (res as usize) * (res as usize);
                data[idx] = best.sqrt();
            }
        }
    }

    // ── Sign pass — parity of axis-aligned ray intersections ─────
    // Cast +X for every (y, z) row and toggle sign whenever a
    // triangle crosses. Works robustly for closed manifold meshes.
    //
    // The y/z coordinates of every row are perturbed by an irrational
    // multiple of cell_size so the rays never run along a triangle's
    // edge (which would double-count crossings in a manifold mesh).
    // This is the standard "shoot a ray off-axis" trick used in
    // GVDB / OpenVDB voxelisers.
    let ray_back: Real = 1.0e6;
    let nudge_y: Real = cell_size * 1.0e-3 * std::f64::consts::SQRT_2;
    let nudge_z: Real = cell_size * 1.0e-3 * std::f64::consts::PI / 4.0;
    for z in 0..res {
        for y in 0..res {
            // Origin lives `ray_back` metres back in -X from the row's
            // first sample so every triangle hit has a positive `t`.
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
            // Walk x cells; each hit ahead of the cell flips parity.
            let mut hit_idx = 0;
            let mut inside = false;
            for x in 0..res {
                // Distance along +X from `p_origin` to this sample.
                let p_x = ray_back + (x as Real) * cell_size;
                while hit_idx < hits.len() && hits[hit_idx] <= p_x {
                    inside = !inside;
                    hit_idx += 1;
                }
                if inside {
                    let idx = (x as usize)
                        + (y as usize) * (res as usize)
                        + (z as usize) * (res as usize) * (res as usize);
                    data[idx] = -data[idx];
                }
            }
        }
    }

    SdfGrid {
        nx: res,
        ny: res,
        nz: res,
        origin: mn,
        cell_size,
        data,
    }
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
