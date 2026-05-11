# Broad phase

The broad phase generates candidate `ShapePair { a, b }` records
that the narrow phase will probe for actual contact. molla ships
three CPU broad phases plus a GPU one. All four produce the same
pair set (modulo order) — they're benchmarked head-to-head and
parity-tested against brute force in
`crates/molla/src/geometry/bvh.rs:332-371`.

| Algorithm | Source | Default | When to prefer |
|-----------|--------|---------|----------------|
| Brute force | `geometry/broad_phase.rs::brute_force_pairs` | tests only | parity checks, < 20 shapes |
| Sweep-and-Prune (SAP) | `geometry/broad_phase.rs::BroadPhase::sap_pairs` | **yes** | up to a few hundred shapes |
| Top-down BVH | `geometry/bvh.rs::Bvh::pairs` | opt-in | > few hundred shapes |
| GPU LBVH | `kernels/lbvh.rs::LbvhBroadPhase` | opt-in (`gpu` feature) | > 1 k shapes |

## Pair filtering

Every broad phase routes its candidate pairs through
`pair_passes_filters` (`broad_phase.rs:19-52`) before emitting them:

```rust
pub fn pair_passes_filters(
    model: &Model,
    shape_body: &[i32],
    body_world: &[i32],
    i: usize,
    j: usize,
) -> bool;
```

The filter applies five rules in order:

1. **Static pair** — two shapes both attached to no body (or
   anchored statics) cannot collide.
2. **Self-collision** — two shapes on the same body are skipped.
3. **World partitioning** — shapes in different worlds (multi-env
   parallel rollouts) are skipped.
4. **Collision-group bitmask** — `groups[i] & groups[j] == 0` skips.
   The default group is `0xffff_ffff` (every bit on) which means
   "collide with everything".
5. **Explicit exclusions** — `model.shape_collision_filter_pairs`
   is a sorted `Vec<(u32, u32)>` of `(lo, hi)` pairs to skip;
   binary-searched per pair.

Every algorithm calls the filter at the leaf level so the pair list
is canonical (`a < b`, deduplicated) before the narrow phase sees
it.

## Brute force

```rust
pub fn brute_force_pairs(
    model:  &Model,
    state:  &State,
    margin: Real,
    out:    &mut Vec<ShapePair>,
);
```

O(N²) AABB-overlap check over every pair. Two uses: a parity oracle
for the SAP / BVH implementations, and a fallback for tiny scenes
where the SAP setup cost wouldn't amortize.

## Sweep-and-Prune

```rust
pub struct BroadPhase { /* SAP scratch state */ }

impl BroadPhase {
    pub fn new() -> Self;
    pub fn sap_pairs(
        &mut self,
        model:  &Model,
        state:  &State,
        margin: Real,
        out:    &mut Vec<ShapePair>,
    );
}
```

Default for `CollisionPipeline` (`pipeline.rs`). Algorithm:

1. **AABB cache.** Compute world-space AABBs for every shape via
   `shape_world_aabb`, expanded by `margin`.
2. **Partition.** Planes (infinite AABB) bucketed separately from
   finite shapes — they get paired against every non-static finite
   shape regardless of axis.
3. **Axis pick.** `best_sweep_axis` picks the axis with the largest
   centroid-coordinate variance — minimises sweep collisions on
   average (`broad_phase.rs:224-241`).
4. **Sort.** Sort finite-shape indices by AABB-min on that axis.
5. **Sweep.** For each shape `i`, scan forward until shape `j`'s
   min exceeds shape `i`'s max — at which point every later shape
   also can't overlap. For shapes inside the active set, check
   full 3D AABB intersection.

The SAP scratch (`aabbs`, `finite_idx`, `infinite_idx`) is kept on
the `BroadPhase` struct so it persists across frames — no per-step
allocation.

## Top-down BVH

[`Bvh`](./bvh.md) — `Bvh::build` constructs the tree from scratch
every step (top-down median-split), then `Bvh::pairs` traverses it
with the descend-larger-node heuristic to emit overlap pairs. Wins
over SAP once shape count gets large enough that the sweep's
near-linear scan starts to hurt (a few hundred shapes is the
crossover on a modern CPU).

## GPU LBVH

`crates/molla/src/kernels/lbvh.rs` ships a Karras-2012 LBVH
("Maximizing parallelism in the construction of BVHs, octrees, and
k-d trees") with a parallel pair-finder. Enable via:

```rust
pipeline.enable_gpu_broad_phase(&engine, /* refit_only: */ false);
```

When `refit_only = true`, the kernel reuses the previous frame's
tree topology and only re-runs AABB refit + the pair-finder.
Appropriate for cloth / soft-body scenes where shape topology is
fixed across frames — saves the radix-sort step.

The output flows into `CollisionPipeline::pair_buf` exactly like
SAP would. The rest of the pipeline (GPU prefilter, full GPU narrow
phase, CPU fallback) is unchanged. `LbvhPairs` is the host-side
result struct with the pair list + a count buffer.

`collect_shape_aabbs` is the helper that walks the model + state
and packs shape AABBs into the GPU's `vec4 + vec4` layout. Half-
space planes are excluded (infinite AABB poisons the build) and
paired against the finite shapes in a separate host-side scan,
mirroring the CPU BVH's behaviour.

## Cross-cutting: `margin`

Every broad phase takes a `margin: Real` parameter that gets added
to each AABB before the overlap test. The narrow phase consumes the
same value and uses it as the contact-generation threshold — a
shape pair with a non-zero `margin` will register a contact at
penetration depth `-margin` (still positive separation but inside
the "about to touch" zone). This is what hooks the **continuous
collision detection** in [`vt_ccd`](./soft_contact.md) into the
broad phase — the CCD pre-pass inflates AABBs by the swept-volume
extent so the narrow-phase sees fast-moving pairs as candidates.
