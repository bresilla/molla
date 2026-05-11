# Contact reduction

A box-on-box face contact can emit 4–8 candidate points from SAT,
a triangle-mesh patch can emit dozens, hydroelastic integration
hundreds. The solver doesn't want them all — redundant contacts on
the same patch are wasted work and the system never quite reaches
a stable rest. **Contact reduction** coalesces many narrow-phase
contacts on the same shape pair into a small canonical manifold
(typically 4 points).

Two implementations cohabit:

* `crates/molla/src/geometry/contact_extras.rs` — the **voxel-bin
  pruner**. Per-frame, in the live `Contacts` buffer.
* `crates/molla/src/geometry/contact_reduction.rs` — the
  **slice-API reducer** + temporal warm-start. Generic
  max-N-per-pair primitive.

GPU port at `crates/molla/shaders/contact_reduction.wgsl` (Stage
28.1).

## Voxel-AABB pruning (`contact_extras`)

The per-frame helper used inside the pipeline:

```rust
pub fn contact_id(
    a_shape: u32, b_shape: u32, point: Vec3, voxel_size: Real,
) -> u64;

pub fn reduce_contacts(contacts: &mut Contacts, voxel_size: Real);

pub fn build_contact_id_table(
    a_idx: &[u32], b_idx: &[u32], point: &[Vec3],
    voxel_size: Real, out: &mut Vec<u64>,
);

pub fn match_contacts(
    prev_ids: &[u64], next: &[Contact], voxel_size: Real,
) -> Vec<i32>;
```

`contact_id` hashes `(min(a, b), max(a, b), voxel(point))` into a
`u64`. Two contacts on the same shape pair separated by less than
`voxel_size` hash to the same id — they're treated as "the same
persistent contact" for warm-starting. Picking the voxel size
trades match-rate (larger ⇒ more warm-starts survive) against
patch resolution (larger ⇒ legitimately-different contacts get
collapsed).

`reduce_contacts` bins contacts by `(a_shape, b_shape,
voxel(point))` and keeps only the deepest contact per voxel. This
stops a 4-corner box-plane stack from emitting 4 contacts that
overlap into the solver — they collapse to one per voxel cell.

`match_contacts` is the temporal-coherence hook: given the
previous frame's contact-ID list and the current frame's
contacts, return an array `prev_idx` of length `next.len()` where
`prev_idx[i]` is the index of the matching previous contact (or
`-1` if no match). Friction's Lagrange-multiplier warm-start keys
off this.

## Slice-API reducer

Standalone primitive that callers can use outside the live
`Contacts` buffer (oracle tests, MPR fixtures, custom narrow
phases):

```rust
pub struct Contact {
    pub a_shape: u32,
    pub b_shape: u32,
    pub id:      u64, // opaque; preserved on output
    pub point:   Vec3,
    pub normal:  Vec3,
    pub depth:   Real,
}

pub fn reduce_contacts(contacts: &[Contact], max_per_pair: usize) -> Vec<Contact>;
pub fn reduce_with_history(
    prev: &[Contact], current: &[Contact],
    max_per_pair: usize, match_radius: Real,
) -> Vec<Contact>;
```

### "Deepest + farthest spread" policy

The algorithm (`contact_reduction.rs:84-195`, matching Newton's
`solvers/contact/contact_reduction_global.py`):

1. Group contacts by `(a_shape, b_shape)`.
2. For each group, always keep the **deepest** contact (first
   slot).
3. Greedily add the contact whose minimum distance to any
   already-kept point is **maximal** — i.e. expand the convex
   hull of the kept set as fast as possible.
4. Stop when the group hits `max_per_pair` or runs out of
   candidates.

This produces a well-conditioned manifold: 4 corners of a face,
not 4 points clustered at one corner. For a box-on-box face
contact that gives the solver exactly the 4 corner contacts it
needs to capture the patch geometry, no more.

Tie-break rules (`contact_reduction.rs:159-184`):

* Spread first (with `eps = 1e-12`).
* On a spread tie, prefer deeper.
* On both a tie, prefer lower input index.

Output order is deterministic for a given input — deepest first,
then greedy spread order.

### Warm-start matching

`reduce_with_history` runs the reducer on `current`, then walks
each reduced contact and attaches the closest unclaimed previous
contact within `match_radius`:

```rust
// contact_reduction.rs:238-262
for cur in reduced.iter_mut() {
    let key = (cur.a_shape, cur.b_shape);
    if let Some(prev_indices) = prev_by_pair.get(&key) {
        let mut best: Option<usize> = None;
        let mut best_d2 = match_radius * match_radius;
        for &pi in prev_indices {
            if claimed[pi] { continue; }
            let d2 = (prev[pi].point - cur.point).length_squared();
            if d2 <= best_d2 { best_d2 = d2; best = Some(pi); }
        }
        if let Some(pi) = best { cur.id = prev[pi].id; claimed[pi] = true; }
    }
}
```

The `claimed` array enforces a one-to-one match so two current
contacts don't both inherit the same `id`. Rule of thumb for
`match_radius`: `2 × contact_margin` for rigid-body face
contacts, ~1 mm for cloth, ~2× expected motion for ragdolls /
hands.

## Graph-coloring partitioning

`crates/molla/src/geometry/graph_coloring.rs` provides the
constraint-graph coloring utility used by both the contact
reducer (when the solver wants conflict-free parallel batches)
and the XPBD distance-constraint pass:

```rust
pub fn greedy_color(adjacency: &[Vec<u32>]) -> Vec<u32>;
pub fn mcs_color(adjacency: &[Vec<u32>]) -> Vec<u32>; // chordal-optimal
pub fn color_constraints(constraint_verts: &[Vec<u32>]) -> Vec<u32>;
pub fn group_by_color(colors: &[u32]) -> Vec<Vec<usize>>;
```

`color_constraints` builds the line graph (constraints are edges
in the original graph; the line graph makes constraints sharing a
vertex adjacent) and returns a per-constraint color via MCS
(Pereira & Vidigal 2014). On chordal graphs MCS is optimal; on
general graphs it's near-greedy. `group_by_color` slices into
parallel-safe batches: every constraint within a color is
guaranteed to act on disjoint vertex sets, so the solver can
dispatch them concurrently without atomics.

The contact reducer doesn't currently route through this — the
reducer's per-pair groups are independent by construction (one
shape pair → one group, no cross-pair vertex sharing). The
coloring path becomes relevant when reducer output goes into the
[XPBD](../solvers/xpbd.md) substep, which needs conflict-free
batches for its Jacobi-style parallel projection.

## GPU reduction kernel

`shaders/contact_reduction.wgsl` ports the per-group greedy
algorithm verbatim. The host packs up to `MAX_PER_GROUP = 64`
contacts per shape pair into a dense slab; one workgroup per
group runs the scalar reduction (only thread 0 of each workgroup
does work — the per-group `n` is too small for workgroup-level
parallelism to pay):

```wgsl
@compute @workgroup_size(64)
fn contact_reduce(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let g = wg.x;
    if (g >= u.group_count) { return; }
    if (lid.x != 0u) { return; }      // scalar per group
    reduce_one_group(g);
}
```

The "greedy farthest-point" inner loop is byte-for-byte the same
as CPU — same tie-break with `eps = 1e-6` (slightly looser than
the CPU's `1e-12` to accommodate `f32` precision; the CPU/GPU
parity test verifies the deepest contact survives in both
paths). Per-group output cap is `MAX_OUT_PER_GROUP = 4`. Output
layout mirrors input: a same-shaped slab plus a per-group count
buffer so the host can know how many of the 4 slots are valid.

The kernel sits behind the `ContactReductionKernel` host
dispatcher (`crates/molla/src/kernels/contact_reduction.rs`); see
`pack_groups_host` and `unpack_output_host` for the
slab-packing / unpacking helpers exposed for tests.
