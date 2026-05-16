# Graph coloring

`crates/molla-math/src/coloring.rs` (Stage 38d) centralises
the constraint-graph colouring step used by VBD and XPBD. Before this
module, each parallel-constraint solver carried its own copy of the
same recipe; now there's a single source of truth.

## Why coloring matters for parallel constraint projection

VBD's per-vertex Gauss-Seidel iteration and XPBD's distance-constraint
projection are both **constraint-batch parallel**: within a single
batch, no two constraints write to the same vertex, so the GPU can
dispatch one thread per constraint without a write-hazard. Across
batches, you need a barrier — vertex updates from batch *k* must be
visible to batch *k+1*.

The colouring step partitions a constraint set into the minimum number
of batches that satisfy that invariant. It runs once per scene at
build time, never per frame, so the GPU never sees the raw graph.

## Public API

```rust
pub fn greedy_color(adjacency: &[Vec<u32>]) -> Vec<u32>;
pub fn mcs_color(adjacency: &[Vec<u32>]) -> Vec<u32>;
pub fn color_constraints(constraint_verts: &[Vec<u32>]) -> Vec<u32>;
pub fn group_by_color(colors: &[u32]) -> Vec<Vec<usize>>;
```

The algorithms mirror the reference implementation in
`xtra/warp/warp/native/coloring.cpp` (Gaia / Anka Chen, Apache-2.0).

### `greedy_color`

Degree-ordered greedy colouring (Fratarcangeli & Pellacini 2015).
Sort vertices by descending degree; for each vertex in that order,
assign the smallest colour unused by its already-coloured
neighbours. Cheap (`O(V log V + E)`) and usually within 1-2 colours
of optimal on constraint graphs.

```rust
let edges = vec![vec![0u32, 1], vec![1, 2], vec![2, 3]];
let colors = color_constraints(&edges);
// constraint 0 and 1 share vertex 1 ⇒ different colours
// constraint 1 and 2 share vertex 2 ⇒ different colours
assert_ne!(colors[0], colors[1]);
assert_ne!(colors[1], colors[2]);
```

### `mcs_color`

Maximum Cardinality Search ordering (Pereira & Palsberg 2005)
followed by the same smallest-unused-colour assignment as
`greedy_color`. The MCS step builds an ordering where, at each step,
the next vertex is the unordered one with the most already-ordered
neighbours.

On **chordal graphs** MCS produces an optimal colouring; on general
graphs it typically beats degree-ordered greedy by 1-2 colours. For
the structured grid-cloth constraint graphs molla tests against, MCS
hits the theoretical optimum (4 colours for a rectangular grid).

The implementation uses a bucket-based priority queue (`McsBuckets`)
to keep all operations amortised O(1). Time complexity is `O(V + E)`.

### `color_constraints` — the line-graph helper

The real workhorse. `constraint_verts[c]` lists the vertices touched
by constraint `c`. The function builds the conflict graph (one node
per constraint; two constraints adjacent iff they share at least one
vertex) and returns a per-constraint colour via `mcs_color`.

Naively this is `O(C²)`. The implementation bucket-groups constraints
by vertex and only adds pairwise edges within each bucket, giving
`O(Σ_v deg(v)²)` — much cheaper on the sparse constraint graphs that
real meshes produce.

### `group_by_color` — partition into batches

```rust
let groups: Vec<Vec<usize>> = group_by_color(&colors);
// groups[c] is the sorted list of constraint indices assigned colour c.
```

The outer vec has length `max(colors) + 1`. Callers dispatch one GPU
pass per colour over its index list, with a barrier between passes.

## End-to-end wiring

```rust
use molla::math::{color_constraints, group_by_color};

// 1. Build the constraint set.
let constraint_verts: Vec<Vec<u32>> = build_edges_from_mesh(&mesh);

// 2. Colour once at scene-build time.
let colors = color_constraints(&constraint_verts);
let batches = group_by_color(&colors);

// 3. Each frame, dispatch one pass per colour batch.
for batch in &batches {
    dispatch_constraint_pass(batch, /* shared state */);
    // barrier between batches handled by wgpu's pass boundary
}
```

A typical 4×4 grid cloth produces 24 constraints (12 horizontal + 12
vertical) and an MCS colouring uses 4 colours; the
`color_constraints_distance_constraint_graph` test verifies the
upper-bound contract (`n_colors ≤ 6`) and asserts no adjacent
constraints share a colour. Theory says 4 suffices; greedy
occasionally blows up to 5-6 on degenerate orderings, which is why
the test only asserts the upper bound.

## Limitations + design notes

- **Both algorithms run on the CPU.** Colouring is a one-shot
  preprocessing step; the GPU never sees the raw graph. Pushing it
  on-device is technically possible (parallel speculative colouring
  á la Çatalyürek et al.) but not motivated yet.
- **No edge weights.** The line graph is unweighted — every
  constraint costs one slot in its colour batch. Constraint-cost
  balancing (longer constraints take more flops) isn't modelled; in
  practice the variance across constraints in a single batch is
  small enough that thread-block stragglers don't matter.
- **Deterministic.** Both algorithms tie-break by lowest index for
  full reproducibility. Same input → same colour assignment across
  runs.

## Cross-references

- [XPBD](./../solvers/xpbd.md) is the main consumer for distance
  constraints.
- [VBD](./../solvers/vbd.md) consumes the same machinery for
  vertex-block descent on soft-body energy.
