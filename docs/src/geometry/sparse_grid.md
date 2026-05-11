# Sparse grids (NanoVDB-equivalent)

A 1024³ dense SDF needs `1024³ × 4 B = 4 GiB` even though the
non-empty (narrow-band) voxels typically cover well under 1% of
the volume. `crates/molla/src/geometry/sparse_grid.rs` implements
a clean-room WGSL-friendly port of NanoVDB's three-level Root →
Internal → Leaf topology (Museth 2013, "VDB: High-Resolution
Sparse Volumes with Dynamic Topology"; the "NanoVDB" name comes
from NVIDIA's CUDA-side single-allocation refactor). molla
maintains its own layout — portable wgpu, no CUDA dependency.

## Topology

```text
Root      ─ HashMap<[i32; 3], u32>     → child internal idx
Internal  ─ 32 × 32 × 32 u32 children + 1024-u32 activity bitmask
Leaf      ─ 8  × 8  × 8  f32 values   + 16-u32   activity bitmask
```

Voxel coord `(x, y, z)` decomposes as `(root_key, internal_idx,
leaf_idx)` via fixed bit shifts:

```rust
pub const LEAF_DIM:        i32   = 8;   // 8³ = 512 voxels / leaf
pub const LEAF_SHIFT:      i32   = 3;   // 2³ = 8 voxels per leaf axis
pub const LEAF_MASK_U32S:  usize = 16;  // 8³ / 32 bits

pub const INTERNAL_DIM:        i32   = 32;   // 32³ = 32768 children
pub const INTERNAL_SHIFT:      i32   = 8;    // LEAF_SHIFT + 5
pub const INTERNAL_MASK_U32S:  usize = 1024; // 32³ / 32 bits
```

```text
leaf_local(x, y, z)     = (x & 7,        y & 7,        z & 7)
internal_local(x, y, z) = ((x >> 3) & 31, (y >> 3) & 31, (z >> 3) & 31)
root_key(x, y, z)       = ( x >> 8,       y >> 8,       z >> 8     )
```

Each internal node spans `2⁸ = 256` voxels per axis. A 1024³
volume fits in `(1024 / 256) = 4` root keys per axis = 64 root
entries — well within the linear-search budget of the WGSL
traversal. NanoVDB ships with three internal levels for arbitrary
world sizes; molla collapses the upper levels into one root
because v1 targets 1024³–4096³ assets where the root never gets
large.

## Why this layout

* **Leaf 8³** matches NanoVDB's tile size. A leaf is a contiguous
  `512 × 4 = 2048 bytes` — easy to fit in shared memory if we
  ever go workgroup-per-leaf.
* **Internal 32³** keeps the child-pointer array cache-friendly
  (`32 × 32 × 32 × 4 = 128 KiB` per internal node). Wider would
  waste storage; narrower would mean more internal nodes.
* **Root as a hash map** keeps the top of the tree mutable on the
  CPU without a fixed pre-allocation. The GPU sees the root as a
  flat `(key, child_idx)` array — Stage 24e.1 adds an
  open-addressing hash table for O(1) GPU lookup; Stage 24e
  shipped a linear-scan fallback.

## Mask bit layout

Both the leaf and internal masks use the standard "one bit per
slot, packed into `u32` words" convention:

```rust
// Leaf::set (sparse_grid.rs:126-130)
let idx = leaf_local(x, y, z);                // 0..=511
self.values[idx] = value;
self.mask[idx >> 5] |= 1u32 << (idx & 31);    // word `idx/32`, bit `idx%32`

// Leaf::get (sparse_grid.rs:133-138)
if (self.mask[idx >> 5] >> (idx & 31)) & 1 == 1 {
    Some(self.values[idx])
} else {
    None
}
```

`idx >> 5` is the `u32` word index; `idx & 31` is the bit within
that word. Same pattern for the internal mask, where `idx` is the
flattened `(ix, iy, iz)` triple (`ix + 32 * iy + 32 * 32 * iz`).

`Leaf::active_count` / `InternalNode::active_count` are sums of
`u32::count_ones()` over the mask — the population count
(popcount) tells you how many voxels are active without walking
the values.

## Public CPU API

```rust
pub struct Leaf {
    pub values: Box<[f32; 512]>,
    pub mask:   [u32; LEAF_MASK_U32S],
}

pub struct InternalNode {
    pub children: Box<[u32; 32 * 32 * 32]>, // u32::MAX = no child
    pub mask:     Box<[u32; INTERNAL_MASK_U32S]>,
}

pub struct SparseGrid {
    pub root:       HashMap<[i32; 3], u32>,
    pub internals:  Vec<InternalNode>,
    pub leaves:     Vec<Leaf>,
    pub background: f32,
}

impl SparseGrid {
    pub fn new(background: f32) -> Self;
    pub fn insert(&mut self, x: i32, y: i32, z: i32, value: f32);
    pub fn lookup(&self, x: i32, y: i32, z: i32) -> f32;
    pub fn for_each_active_leaf<F: FnMut(u32, [i32; 3])>(&self, f: F);

    pub fn leaf_count(&self) -> usize;
    pub fn internal_count(&self) -> usize;
    pub fn root_count(&self) -> usize;
    pub fn active_voxel_count(&self) -> u64;
    pub fn storage_bytes(&self) -> usize;
}
```

`SparseGrid::insert` (Stage 24e.1 — mutable) allocates internal /
leaf nodes lazily and only on first touch
(`sparse_grid.rs:211-236`). The two-step allocation (alloc
internal if missing, then alloc leaf if missing) keeps the cost
proportional to the *new* topology, not the existing structure.
Stage 24e.2 adds `delete` for runtime stripping (not yet wired
up here — the topology is monotonic in v1 plus mutable insert).

`for_each_active_leaf` is the "do work per active region" hook;
on the GPU side the equivalent is one workgroup per active leaf.

## GPU upload — `SparseGridGpu`

```rust
pub struct SparseGridGpu {
    pub root_entries:        Vec<i32>, // [key.x, key.y, key.z, internal_idx]…
    pub root_table:          Vec<i32>, // open-addressing hash table (Stage 24e.1)
    pub root_table_capacity: u32,
    pub internal_children:   Vec<u32>, // 32³ u32 per internal
    pub internal_masks:      Vec<u32>, // 1024 u32 per internal
    pub leaf_values:         Vec<f32>, // 512  f32 per leaf
    pub leaf_masks:          Vec<u32>, // 16   u32 per leaf
    pub background:          f32,
}

impl SparseGridGpu {
    pub fn from_grid(grid: &SparseGrid) -> Self;
}
```

Everything is flat `u32`/`f32` so the buffers pack into `wgpu::Buffer`
without f64↔f32 conversion shenanigans. `from_grid` sorts the
root by key so the layout is byte-deterministic across runs
(`HashMap` iteration order isn't deterministic).

## Stage 24e.1 — `root_hash3` GPU lookup

The Stage 24e linear scan over `root_entries` is fine for ≤ 64
entries but scales poorly as a 4096³ asset pushes past 1024 roots.
Stage 24e.1 adds an open-addressing hash table sized to load
factor 0.7 (power-of-two capacity for cheap `slot & mask`
wrap-around). The hash function is the same in Rust and WGSL:

```rust
pub fn root_hash3(key: [i32; 3]) -> u32 {
    let mut h: u32 = 0x9E37_79B9;          // golden-ratio seed
    for &k in &key {
        let mut k = k as u32;
        k = k.wrapping_mul(0xCC9E_2D51);
        k = k.rotate_left(15);
        k = k.wrapping_mul(0x1B87_3593);
        h ^= k;
        h = h.rotate_left(13);
        h = h.wrapping_mul(5).wrapping_add(0xE654_6B64);
    }
    // Murmur3 finalizer.
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}
```

WGSL `u32` wraps modulo `2^32` exactly like Rust's `wrapping_*`,
so the two implementations agree byte-for-byte. The choice of
MurmurHash3 finalizer + golden-ratio seed is dictated by the
expected workload — sequential coordinates dominate when SDF
narrow-band voxels are baked in scan order, and the bit-mixing
above spreads them evenly across the table. The `(avg_probes,
max_probes)` stats helper (`sparse_grid.rs:435-480`) lets tests
verify the load factor / probe-length contract without
instrumenting the GPU.

`root_table_capacity == 0` selects the linear-scan fallback for
back-compat with the Stage 24e v1 traversal.

## `SparseSdf` — narrow-band SDF wrapper

```rust
pub struct SparseSdf {
    pub grid:      SparseGrid,
    pub origin:    Vec3,    // world position of voxel (0, 0, 0)
    pub cell_size: Real,
    pub nx: u32, pub ny: u32, pub nz: u32, // logical voxel extent
}

impl SparseSdf {
    pub fn sample(&self, p: Vec3) -> Real;
    pub fn normal(&self, p: Vec3) -> Vec3;
    pub fn logical_voxel_count(&self) -> u64;
}

pub fn bake_mesh_to_sparse_sdf(
    mesh: &TriMesh, resolution: u32, padding: Real, narrow_band: u32,
) -> SparseSdf;

pub fn sparse_sdf_from_dense(
    dense: &crate::geometry::sdf_bake::SdfGrid, narrow_band: u32,
) -> SparseSdf;
```

Thin wrapper mirroring the dense [`SdfGrid`](./sdf.md) API.
Trilinear `sample` returns the stored value if the surrounding 8
voxels are all active, otherwise the background. `normal` uses
central differences. `logical_voxel_count` is the dense-equivalent
voxel count — used by the storage-cost audit (a 1024³ logical SDF
can be much sparser than its dense equivalent; ratio of
`storage_bytes` to `logical_voxel_count * 4` is the compression
factor).

`bake_mesh_to_sparse_sdf` walks the mesh BVH, computes signed
distances on the narrow-band voxels only, and skips the rest. The
narrow-band width is the **inflation** in voxels around the
surface where samples are kept — typical value 3-4 for collision
queries (close enough to capture the surface, far enough not to
miss). `sparse_sdf_from_dense` is the trivial down-conversion: walk
the dense grid, keep cells whose `|sdf| < narrow_band * cell_size`.
