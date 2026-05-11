# Marching cubes & SDF baking

A pair of asset-load tools that round-trip between meshes and
sampled signed-distance fields:

* `crates/molla/src/geometry/sdf_bake.rs` — `bake_mesh_to_sdf`
  voxelises a `TriMesh` into a regular 3D grid of signed
  distances.
* `crates/molla/src/geometry/marching_cubes.rs` — `marching_cubes`
  extracts a triangle mesh from a sampled SDF (the reverse
  direction).

`bake → marching_cubes` round-trips a mesh through a SDF; the
fidelity is bounded by voxel resolution.

These types support both
[hydroelastic contact](./hydroelastic.md) (the soft body's
pressure field is `E · max(0, -sdf)`) and the
[sparse grid](./sparse_grid.md) NanoVDB-equivalent (which stores
only the narrow band of the dense SDF).

## `SdfGrid`

```rust
pub struct SdfGrid {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub origin:    Vec3, // world corner of voxel (0, 0, 0)
    pub cell_size: Real, // uniform voxel side length
    pub data:      Vec<Real>, // length nx · ny · nz, row-major (x fastest)
}

impl SdfGrid {
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn at(&self, x: u32, y: u32, z: u32) -> Real;
    pub fn voxel_pos(&self, x: u32, y: u32, z: u32) -> Vec3;
}
```

Layout: `data[x + y * nx + z * nx * ny]` (X fastest, then Y, then
Z) — standard row-major to match `(i, j, k) → linear` in WGSL
loops. Sample positions are `origin + (x, y, z) * cell_size`.

`bake_mesh_to_sdf` always emits a cubic grid (`nx == ny == nz`)
but the type stores the three axes independently to leave room for
non-uniform fields later (Stage 22.2 has those slated for the
heightfield → SDF path).

`at(x, y, z)` returns `+∞` for out-of-range coordinates so callers
that walk a 3×3×3 stencil don't have to special-case the boundary.

## Sample functions

`hydroelastic.rs` ships two helpers that consumers reach for:

```rust
pub fn sample_sdf_trilinear(grid: &SdfGrid, p: Vec3) -> Real;
pub fn sample_sdf_gradient (grid: &SdfGrid, p: Vec3) -> Vec3;
```

`sample_sdf_trilinear` does the standard 8-corner trilinear blend:
`floor` the local coordinate, fetch the 8 surrounding voxels,
blend `xyz` first then `z`. Returns `+∞` (treated as "deep
outside") when the query is out-of-bounds — same convention as
`SdfGrid::at`.

`sample_sdf_gradient` uses central differences over the 6
neighbour stencil:

```text
grad.x = (sdf(x+1, y, z) - sdf(x-1, y, z)) / (2 * cell_size)
grad.y = (sdf(x, y+1, z) - sdf(x, y-1, z)) / (2 * cell_size)
grad.z = (sdf(x, y, z+1) - sdf(x, y, z-1)) / (2 * cell_size)
```

This gives the outward normal of the surface — `+∇sdf` because
the SDF increases outward.

## Baking — `bake_mesh_to_sdf`

```rust
pub fn bake_mesh_to_sdf(
    mesh:       &TriMesh,
    resolution: usize,
    padding:    Real,
) -> SdfGrid;
```

Algorithm (`sdf_bake.rs:14-34`):

1. Compute the mesh's AABB, expand by `padding` (world-space
   metres), divide into a `resolution × resolution × resolution`
   voxel grid.
2. **Unsigned distance pass.** For each grid sample, find the
   closest point on every triangle via
   [`closest_point_on_triangle`](#closest-point-on-triangle); keep
   the minimum unsigned distance.
3. **Sign pass.** Determine sign via parity of an axis-aligned ray
   intersection count: odd → inside (negative), even → outside
   (positive). Robust enough for closed manifold meshes (the only
   kind supported); open meshes return ±d arbitrarily near the
   boundary.

Performance: `O(resolution³ × triangle_count)`. Fine for
asset-load workloads — 32³ × 1k tris ≈ 33M ops, ms-range. Stage
21.1 parallelises the closest-distance pass with rayon: voxels
are independent so fan-out is embarrassingly parallel and the
resulting `data` array is byte-deterministic across runs (each
voxel's `min` reduction sees the same triangles in the same order
within its task). The sign pass remains sequential because it
threads parity state along `+X` across cells in a row.

## `closest_point_on_triangle`

```rust
pub fn closest_point_on_triangle(
    p: Vec3, a: Vec3, b: Vec3, c: Vec3,
) -> (Real, Vec3);
```

Returns `(unsigned_distance, closest_point)`. The standard 7-region
classification (Eberly 2000): for each of the 7 Voronoi regions of
the triangle (3 vertices, 3 edges, 1 face), test whether `p`
projects into that region, and if so return the closest point in
the region.

Used by:

* `bake_mesh_to_sdf` (the unsigned-distance pass).
* `bake_mesh_to_sparse_sdf` in [sparse_grid.rs](./sparse_grid.md)
  (narrow-band SDF baking — only voxels within
  `narrow_band * cell_size` are computed).
* `narrow_phase.rs::trimesh_pair` indirectly, for the per-triangle
  contact test against the other shape.

The routine works in `f64` end-to-end (no precision loss from a
mid-routine downcast); GPU ports of the same algorithm in
`shaders/raycast.wgsl` use `f32` for memory efficiency but produce
parity-equivalent results within the tolerance of single-precision
arithmetic.

## Marching cubes

```rust
pub fn marching_cubes(sdf: &SdfGrid, iso: Real) -> TriMesh;
```

Standard Lorensen-Cline 256-case lookup. The two hard-coded tables
(`EDGE_TABLE`, `TRI_TABLE`) are the canonical Newman/Lorensen ones
— see Paul Bourke's "Polygonising a Scalar Field" or Ken Museth's
openvdb mesh-extractor. `iso = 0` reproduces the underlying
surface for an SDF baked via `bake_mesh_to_sdf`.

Per cell:

1. Compute the 8 corner SDF values and form an 8-bit cube index
   `cube_index = Σᵢ (sdf_i < iso ? 1 << i : 0)`.
2. `EDGE_TABLE[cube_index]` tells which of the 12 cube edges have
   a sign change; interpolate the iso-surface crossing on each
   such edge via `interp(iso, p_a, p_b, sdf_a, sdf_b)` (linear
   interpolation of `t = (iso - sdf_a) / (sdf_b - sdf_a)`).
3. `TRI_TABLE[cube_index]` lists which of those edge vertices form
   triangles (up to 5 triangles per cell, 15 indices terminated by
   `-1`).
4. Emit the triangles with fresh vertex indices.

The output is *unwelded* — each triangle gets fresh vertices for
its 3 corners even when shared with an adjacent cell. Welding is
left to a separate `TriMesh::weld_duplicates` pass; the unwelded
output keeps marching cubes itself dead-simple (no hash for shared
edges) at the cost of ~6× the vertex count.

Cells with `EDGE_TABLE[cube_index] == 0` (entirely inside or
entirely outside the iso-surface) short-circuit immediately
(`marching_cubes.rs:57`). Grids with `nx`, `ny`, or `nz` < 2
return an empty mesh — there are no cells to iterate.
