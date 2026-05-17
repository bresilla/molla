# BVH

Three distinct BVHs ship with molla, each tailored to a different
workload:

| Type | Source | Scope | Refit? |
|------|--------|-------|--------|
| [`Bvh`](#broad-phase-bvh) | `geometry/bvh.rs` | shape AABBs (broad phase) | rebuild |
| [`MeshBvh`](#per-mesh-bvh) | `geometry/mesh_bvh.rs` | per-mesh triangle AABBs | static |
| [`TetBvh`](#tet-bvh-stage-204) | `geometry/tet_bvh.rs` | per-tet AABBs (deformable) | CPU BVH + flat GPU AABBs |
| GPU [LBVH](#gpu-lbvh) | `molla-geometry/src/lbvh.rs` + `molla-geometry/shaders/lbvh.wgsl` | shape AABBs (broad phase) | radix-rebuild |

## Broad-phase `Bvh`

```rust
pub struct BvhNode {
    pub aabb:   Aabb,
    pub child0: i32, // > 0 internal, == shape_index when leaf
    pub child1: i32, // -1 sentinel for leaves
}

pub struct Bvh {
    pub nodes:        Vec<BvhNode>,
    pub root:         i32,
    pub infinite_idx: Vec<u32>, // half-space planes
}
```

`Bvh::build(&model, &state, margin)` rebuilds the tree from scratch
each step. Top-down construction: compute per-shape AABBs via
`shape_world_aabb`, partition planes out (their AABB is infinite),
then recursively split the finite shapes on the longest centroid
axis with a median pivot (`bvh.rs:113-141`).

`Bvh::pairs(&model, out)` traverses the tree and writes overlap
pairs to `out`. The algorithm uses the classic descend-larger-node
strategy (`bvh.rs:249-261`): when both children are internal, the
one with the larger AABB descends first so the smaller AABB is
tested against more of the tree (better pruning).

Half-space planes are excluded from the tree and paired against
every finite shape that survives the filters (`bvh.rs:262-307`).
That's because their AABB is infinite — putting them in the tree
poisons the cull. The cost is O(P × F) where P is the plane count
(typically 1).

The CPU path uses median split (`mid = indices.len() / 2`,
`bvh.rs:141`). SAH-based splitting would prune harder but the
median split is sub-millisecond on 1024 shapes and parity-tested
against brute force on both 100- and 1k-shape grid scenes
(`bvh.rs:335-371`).

## Per-mesh `MeshBvh`

```rust
pub struct BvhNode {
    pub aabb_min:   Vec3,
    pub aabb_max:   Vec3,
    pub prim_start: u32, // first tri index in triangle_order
    pub tri_count:  u32,
    pub left:       u32,
    pub right:      u32,
    pub is_leaf:    u32,
}

pub struct MeshBvh {
    pub nodes:          Vec<BvhNode>,
    pub triangle_order: Vec<u32>, // bvh-leaf-ordered tri perm
}

impl MeshBvh {
    pub fn build(vertices: &[Vec3], indices: &[[u32; 3]], max_leaf_size: usize) -> Self;
    pub fn triangles_overlapping_aabb(&self, min: Vec3, max: Vec3, out: &mut Vec<u32>);
    pub fn ray_cast(&self, origin: Vec3, dir: Vec3, max_t: Real) -> Option<RayHit>;
}
```

Built once at model finalisation (or after any `set_mesh_vertices`
call) and stored on the [`TriMesh`](../sim/model.md) entry. The
default `max_leaf_size = 4` is Embree's recommendation — small
enough that leaf scans pull in only a couple of cache lines, big
enough that the tree stays shallow.

The flat-`Vec` layout matches what a GPU upload wants: each
node fits in 8 × `f32` (32 bytes), and the triangle-order
permutation gives contiguous triangle ranges per leaf for vectorised
narrow-phase. The WGSL counterpart
(`crates/molla-geometry/shaders/mesh_bvh_raycast.wgsl`) consumes the same
node layout.

## Tet BVH (Stage 20.4)

`tet_bvh.rs` is the lighter-weight cousin used for soft-body
tet-vs-tet paths and source-side nonconforming-contact lookup. Per
Stage 20.4's risk carve-out, the GPU-facing implementation still keeps
a flat AABB list for deformable-mesh uploads, while Stage 47 v5 adds a
CPU median-split `TetBvh` over the same AABBs for point-location
prefiltering.

* Tet meshes deform every substep — a true BVH would require
  per-substep refit, and a per-vertex GPU traversal of a per-substep
  BVH is the central technical risk Stage 20.4 flags.
* The demo workload is two ~150-tet jelly cubes (≈ 22 500 per-vertex
  × per-tet checks per substep). At that scale a GPU brute-force
  AABB test against the flat array dispatches in microseconds; the
  win from a real GPU tree wouldn't justify the refit cost.
* CPU-side nonconforming projection has a different profile: many
  quadrature points query a static source mesh during matrix assembly.
  There, `TetBvh::point_candidates` avoids a brute-force source-tet
  scan while leaving the exact barycentric containment test unchanged.

```rust
#[repr(C, align(16))]
pub struct TetAabb {
    pub min_x: f32, pub min_y: f32, pub min_z: f32,
    pub tet_id: u32,
    pub max_x: f32, pub max_y: f32, pub max_z: f32,
    pub _pad:  f32,
}

pub fn recompute_tet_aabbs(
    tet_indices: &[u32], vertices: &[Vec3],
    margin: Real, out: &mut [TetAabb],
);
pub fn tet_meshes_aabb_overlap(aabbs_a: &[TetAabb], aabbs_b: &[TetAabb]) -> bool;
pub fn tet_mesh_root_aabb(aabbs: &[TetAabb]) -> Option<(Vec3, Vec3)>;
pub fn tet_closest_face(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> Option<(Real, Vec3)>;

pub struct TetBvh { /* nodes + tet_order */ }
impl TetBvh {
    pub fn build(aabbs: &[TetAabb], max_leaf_size: usize) -> Self;
    pub fn point_candidates(&self, aabbs: &[TetAabb], p: Vec3, out: &mut Vec<u32>);
}
```

The struct is intentionally `[f32; 8]` (not `Real`) for a clean
`vec4 + vec4` GPU upload (32 bytes). `tet_closest_face` is the CPU
oracle for the GPU per-vertex tet-penalty kernel: it walks the
tet's four faces, returns the maximum signed distance (positive
outside, negative inside), and the achieving face's outward normal.

The Stage 20.5 follow-up (proper GPU tree + GPU refit) is queued
behind the rigid-vs-rigid LBVH that's already in production.

## GPU LBVH

`crates/molla-geometry/src/lbvh.rs` + `crates/molla-geometry/shaders/lbvh.wgsl` implement
the standard Karras parallel LBVH:

1. **Morton-code AABB centroids** via the existing 64-bit radix
   sort.
2. **Hierarchy build** — Karras 2012 binary-radix-tree
   construction. Every internal node is assigned to one workgroup.
3. **AABB refit** — per-leaf parent walk with `atomicMin` /
   `atomicMax` on shared bounding boxes.
4. **Pair-finder** — top-down traversal in a workgroup; emits pairs
   into a global counter.

The host dispatcher (`LbvhBroadPhase`) is opt-in via
`CollisionPipeline::enable_gpu_broad_phase` and replaces the CPU
SAP/BVH pair generator. The downstream pipeline (narrow phase, GPU
prefilter, full GPU narrow phase) is unchanged — the output is the
same `Vec<ShapePair>` format. `refit_only = true` reuses last
frame's tree topology and skips the radix-sort step; appropriate
for cloth / soft-body scenes where shape topology is fixed.

`collect_shape_aabbs` is the helper that walks the model + state
and packs AABBs into the GPU-friendly `vec4 + vec4` layout the
WGSL traversal expects.
