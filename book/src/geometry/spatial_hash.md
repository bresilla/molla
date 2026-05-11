# Spatial hash

`crates/molla/src/geometry/spatial_hash.rs` exposes a CPU-side,
host-friendly version of the same fixed-bucket spatial hash that
the cloth particle self-collision kernel
(`shaders/xpbd_particles.wgsl`) embeds for GPU parallelism. The two
agree on cell coordinates byte-for-byte: the hash function and the
`cell_of` floor-floor-floor are picked so cross-checks between CPU
oracle and GPU result are trivial.

## API

```rust
pub struct SpatialHashGrid { /* ... */ }

impl SpatialHashGrid {
    pub fn new(cell_size: Real) -> Self;
    pub fn clear(&mut self);
    pub fn cell_size(&self) -> Real;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;

    pub fn insert(&mut self, id: u32, point: Vec3);
    pub fn insert_aabb(&mut self, id: u32, min: Vec3, max: Vec3);

    pub fn query_aabb<'a>(&'a self, min: Vec3, max: Vec3)
        -> impl Iterator<Item = u32> + 'a;
    pub fn query_neighbours<'a>(&'a self, point: Vec3)
        -> impl Iterator<Item = u32> + 'a;

    pub fn cell_of(&self, p: Vec3) -> (i32, i32, i32);
}
```

The grid is **unordered** and **non-deduplicating**: inserting the
same id twice puts it in the bucket twice. `query_aabb` therefore
yields duplicates if a caller inserted via `insert_aabb` for an id
spanning multiple cells. The caller dedupes if it matters.

## Storage

Cells live in a `HashMap<(i32, i32, i32), Vec<u32>>`. This is
intentional — unbounded coord ranges (negative coordinates,
arbitrarily large worlds, sparse particle distributions) all work
without a pre-sized table. The reference `Vec` is fine for cloth-
grade scenes (≤ ~50 entries per bucket). Profiling on dense
particle clouds (>500 entries per bucket) would motivate a
small-vec / inline-storage upgrade — molla doesn't ship one yet.

`SpatialHashGrid::clear()` drops every inserted entry but keeps
the cell-map capacity for reuse — which is what every per-frame
rebuild wants.

## Particle self-collision

The cell size convention follows the GPU kernel: `cell_size =
2 × max_radius` so the 3×3×3 stencil around a particle covers every
potentially-overlapping neighbour. `query_neighbours(p)` is the
direct hook for that:

```rust
for j in grid.query_neighbours(particle.position) {
    if j == i { continue; }
    if (particle.position - others[j as usize].position).length() < 2.0 * radius {
        // emit contact
    }
}
```

The bucket structure means each query touches at most 27 cell
look-ups regardless of particle density — the cost is bounded by
the local particle count, not by the global count. That's the
property the GPU kernel exploits to keep self-collision a
constant-time-per-particle workload.

## CPU/GPU agreement

The hash function matches WGSL exactly. `cell_of` is
`floor(p * inv_cell)` — and the `floor` here uses Rust `f64::floor`
which, on every IEEE-754 platform, agrees with WGSL's `floor()` for
finite inputs. `Floor(-0.5) == -1` in both Rust and WGSL, so a
particle at `x = -0.5` lands in cell `-1` on both sides; that's
explicitly tested in `spatial_hash.rs:193-197`.

The `query_aabb` parity test (`spatial_hash.rs:217-268`) generates
500 random points and ten random AABB queries, verifies the grid
results are a superset of brute-force (it has to be a superset
because the bucket-query touches every cell intersecting the AABB,
not every cell containing a point inside the AABB), and that every
brute-force hit appears in the grid result. The "grid yields a
superset, never a subset" invariant is what callers rely on when
the grid is used as a broad-phase pre-filter.

## Where it's used

* The cloth XPBD particle self-collision pass embeds a fixed-bucket
  version directly inside the kernel.
* Mesh-authoring tools that walk neighbours (e.g. detecting
  duplicate vertices during convex-hull construction) use the CPU
  grid directly.
* Custom narrow-phase variants and oracle tests that need a "fast
  nearby-particle" iterator without lifting the GPU binding.

For ordered, deterministic-iteration AABB queries, prefer the
[BVH](./bvh.md). The spatial hash is the right tool when the query
set is fundamentally point-shaped (particle systems, point clouds)
and the caller is OK with unspecified iteration order.
