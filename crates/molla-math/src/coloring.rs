//! Host-side graph coloring utility for parallel constraint solvers.
//!
//! Stage 38d deliverable. Both `solvers::vbd` and the XPBD distance-
//! constraint kernels need to partition a constraint set into batches
//! that can be solved in parallel: within a batch, no two constraints
//! share a vertex, so the GPU can dispatch one thread per constraint
//! without write-hazards. Previously each module reinvented this
//! coloring step; this module centralises it.
//!
//! The algorithms here mirror the reference implementation in
//! `xtra/warp/warp/native/coloring.cpp` (Gaia / Anka Chen, Apache-2.0):
//!
//! - [`greedy_color`]: degree-ordered greedy coloring (Fratarcangeli &
//!   Pellacini 2015). Sort vertices by descending degree, then assign
//!   each one the smallest color unused by its already-coloured
//!   neighbours. Cheap and usually within 1-2 colors of optimal on
//!   constraint graphs.
//! - [`mcs_color`]: Maximum Cardinality Search ordering (Pereira &
//!   Palsberg 2005) followed by the same greedy assignment. Tends to
//!   produce fewer colors than plain degree-ordered greedy on
//!   irregular graphs (bending constraints, soft-body tet meshes).
//!
//! Both algorithms run on the CPU — coloring is a one-shot
//! preprocessing step done at scene-build time, not per frame, so the
//! GPU never sees the raw graph.
//!
//! For constraint-set coloring (the case VBD/XPBD actually use), see
//! [`color_constraints`]: it builds the line graph (constraints are
//! nodes; two constraints are adjacent iff they share a vertex) and
//! returns a per-constraint color.
//!
//! ```
//! use molla_math::{color_constraints, group_by_color};
//! // Three distance constraints on a 4-vertex chain 0-1-2-3.
//! let edges = vec![vec![0u32, 1], vec![1, 2], vec![2, 3]];
//! let colors = color_constraints(&edges);
//! assert_eq!(colors.len(), 3);
//! // Adjacent constraints (sharing vertex 1 or 2) must differ.
//! assert_ne!(colors[0], colors[1]);
//! assert_ne!(colors[1], colors[2]);
//! let groups = group_by_color(&colors);
//! // Every constraint index appears exactly once across the groups.
//! let mut flat: Vec<usize> = groups.iter().flatten().copied().collect();
//! flat.sort();
//! assert_eq!(flat, vec![0, 1, 2]);
//! ```

use std::collections::HashSet;

/// Greedy graph coloring with descending-degree vertex ordering.
///
/// `adjacency[v]` is the list of v's neighbours (an undirected graph,
/// so if `v ∈ adjacency[u]` then `u ∈ adjacency[v]`). Self-loops are
/// ignored. Duplicates in the neighbour list are tolerated.
///
/// Returns `colors[v] = color_index ∈ [0, n_colors)` where the number
/// of colors used is `colors.iter().max().map_or(0, |c| c + 1)`.
///
/// Complexity: `O(V log V + E)` where E is the total adjacency size.
pub fn greedy_color(adjacency: &[Vec<u32>]) -> Vec<u32> {
    let n = adjacency.len();
    if n == 0 {
        return Vec::new();
    }

    // Sort vertex indices by descending degree. Ties broken by index
    // for determinism.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        let da = adjacency[a as usize].len();
        let db = adjacency[b as usize].len();
        db.cmp(&da).then_with(|| a.cmp(&b))
    });

    greedy_color_with_order(adjacency, &order)
}

/// Maximum Cardinality Search + greedy coloring.
///
/// Builds an MCS ordering — at each step, picks the unordered vertex
/// with the most already-ordered neighbours — then runs the same
/// smallest-unused-color assignment as [`greedy_color`]. On chordal
/// graphs MCS yields an optimal coloring; on general graphs it
/// typically beats plain degree-ordered greedy by 1-2 colors.
///
/// Complexity: `O(V + E)` with bucket-priority structure for the
/// weight updates (see [`McsBuckets`]).
pub fn mcs_color(adjacency: &[Vec<u32>]) -> Vec<u32> {
    let n = adjacency.len();
    if n == 0 {
        return Vec::new();
    }

    let order = mcs_ordering(adjacency);
    greedy_color_with_order(adjacency, &order)
}

/// Color a list of constraints (each touching a set of vertices) such
/// that constraints sharing a vertex get different colors.
///
/// `constraint_verts[c]` is the list of vertex indices touched by
/// constraint `c`. The function builds the conflict graph (one node
/// per constraint; two constraints adjacent iff they share at least
/// one vertex) and returns a per-constraint color via [`mcs_color`].
///
/// Complexity: `O(sum_v deg(v)^2)` for the line-graph construction
/// (group constraints by vertex; for each vertex's group, add a
/// pairwise edge). On constraint graphs this is much cheaper than the
/// naive `O(C^2)` all-pairs check.
pub fn color_constraints(constraint_verts: &[Vec<u32>]) -> Vec<u32> {
    let n_constraints = constraint_verts.len();
    if n_constraints == 0 {
        return Vec::new();
    }

    // 1. Bucket constraints by vertex.
    let mut max_v: i64 = -1;
    for verts in constraint_verts {
        for &v in verts {
            if v as i64 > max_v {
                max_v = v as i64;
            }
        }
    }
    if max_v < 0 {
        // No vertices referenced — every constraint is independent.
        return vec![0; n_constraints];
    }

    let n_verts = (max_v + 1) as usize;
    let mut vert_to_constraints: Vec<Vec<u32>> = vec![Vec::new(); n_verts];
    for (c_idx, verts) in constraint_verts.iter().enumerate() {
        for &v in verts {
            vert_to_constraints[v as usize].push(c_idx as u32);
        }
    }

    // 2. Build conflict adjacency. Use a per-row HashSet for dedup
    //    (a pair of constraints sharing multiple vertices would
    //    otherwise be inserted multiple times).
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n_constraints];
    let mut seen: Vec<HashSet<u32>> = vec![HashSet::new(); n_constraints];
    for cs in &vert_to_constraints {
        for i in 0..cs.len() {
            for j in (i + 1)..cs.len() {
                let a = cs[i];
                let b = cs[j];
                if a == b {
                    continue;
                }
                if seen[a as usize].insert(b) {
                    adjacency[a as usize].push(b);
                }
                if seen[b as usize].insert(a) {
                    adjacency[b as usize].push(a);
                }
            }
        }
    }

    mcs_color(&adjacency)
}

/// Partition constraint indices into per-color groups.
///
/// Returns `groups` where `groups[c]` is the sorted list of constraint
/// indices assigned color `c`. The outer vec has length
/// `colors.iter().max() + 1`. Callers can dispatch one GPU pass per
/// color over its index list.
pub fn group_by_color(colors: &[u32]) -> Vec<Vec<usize>> {
    if colors.is_empty() {
        return Vec::new();
    }
    let n_colors = *colors.iter().max().unwrap() as usize + 1;
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_colors];
    for (idx, &c) in colors.iter().enumerate() {
        groups[c as usize].push(idx);
    }
    groups
}

// ───────────────────────── internals ──────────────────────────────────

/// Smallest-unused-color assignment over an arbitrary vertex order.
fn greedy_color_with_order(adjacency: &[Vec<u32>], order: &[u32]) -> Vec<u32> {
    let n = adjacency.len();
    let mut colors: Vec<u32> = vec![u32::MAX; n];
    let mut used: Vec<bool> = Vec::with_capacity(64);

    for &v in order {
        let v_u = v as usize;
        used.clear();
        for &nei in &adjacency[v_u] {
            if nei == v {
                continue; // self-loop
            }
            let c = colors[nei as usize];
            if c != u32::MAX {
                let c_u = c as usize;
                if c_u >= used.len() {
                    used.resize(c_u + 1, false);
                }
                used[c_u] = true;
            }
        }
        // Smallest color not in `used`.
        let mut pick = used.len() as u32;
        for (i, &u) in used.iter().enumerate() {
            if !u {
                pick = i as u32;
                break;
            }
        }
        colors[v_u] = pick;
    }
    colors
}

/// Maximum Cardinality Search ordering of `adjacency`'s vertices.
///
/// Returns a permutation of `0..n` such that, walking the permutation
/// in order, each vertex has the maximum number of
/// already-permuted neighbours among the unvisited set. Ties broken
/// by lowest index for determinism.
fn mcs_ordering(adjacency: &[Vec<u32>]) -> Vec<u32> {
    let n = adjacency.len();
    let mut buckets = McsBuckets::new(n);
    let mut weight: Vec<i32> = vec![0; n]; // -1 once popped
    let mut order: Vec<u32> = Vec::with_capacity(n);

    // Seed with vertex 0 (matches Warp/Gaia reference).
    buckets.add(0, 0);

    for _ in 0..n {
        // Restart on a fresh connected component if necessary. The
        // C++ reference scans `weight` linearly for the first non-
        // negative vertex; the same logic here.
        if buckets.is_empty() {
            let mut seed: i64 = -1;
            for (i, &w) in weight.iter().enumerate() {
                if w >= 0 {
                    seed = i as i64;
                    break;
                }
            }
            if seed < 0 {
                // Defensive: should not happen if we loop exactly n
                // times and respect the popped set.
                break;
            }
            buckets.add(seed as u32, 0);
        }

        let max_node = buckets.pop_max();
        order.push(max_node);
        weight[max_node as usize] = -1;

        for &nei in &adjacency[max_node as usize] {
            if nei == max_node {
                continue;
            }
            let w = weight[nei as usize];
            if w < 0 {
                continue; // already in ordering
            }
            if w == 0 {
                // Not in bucket yet: insert with weight 1.
                buckets.add(nei, 1);
                weight[nei as usize] = 1;
            } else {
                buckets.increase(nei, w);
                weight[nei as usize] = w + 1;
            }
        }
    }
    order
}

/// Bucket-based priority queue: nodes keyed by integer weight. Pop
/// always returns a node with the maximum weight. Mirrors the
/// `NodeWeightBuckets` class in `coloring.cpp`.
///
/// All operations are amortized O(1); the only `O(W)` step is finding
/// the next max weight after a bucket empties (scan down from current
/// max), which is bounded by the total number of weight increases.
struct McsBuckets {
    /// `buckets[w]` is the list of node indices with weight `w`.
    buckets: Vec<Vec<u32>>,
    /// `pos[node]` is its index inside its current bucket, or
    /// `u32::MAX` if not present.
    pos: Vec<u32>,
    /// Largest `w` such that `buckets[w]` is non-empty (may lag —
    /// `pop_max` repairs).
    max_weight: usize,
}

impl McsBuckets {
    fn new(n: usize) -> Self {
        Self {
            buckets: vec![Vec::new(); 1],
            pos: vec![u32::MAX; n],
            max_weight: 0,
        }
    }

    fn add(&mut self, node: u32, weight: i32) {
        let w = weight.max(0) as usize;
        if w >= self.buckets.len() {
            self.buckets.resize(w + 1, Vec::new());
        }
        self.pos[node as usize] = self.buckets[w].len() as u32;
        self.buckets[w].push(node);
        if w > self.max_weight {
            self.max_weight = w;
        }
    }

    /// Bump `node`'s weight from `old_weight` to `old_weight + 1`.
    fn increase(&mut self, node: u32, old_weight: i32) {
        let w = old_weight as usize;
        debug_assert!(w < self.buckets.len());
        let idx_in_bucket = self.pos[node as usize] as usize;
        debug_assert!(idx_in_bucket < self.buckets[w].len());

        // O(1) swap-remove.
        let last = *self.buckets[w].last().unwrap();
        self.pos[last as usize] = idx_in_bucket as u32;
        self.buckets[w].swap_remove(idx_in_bucket);
        self.pos[node as usize] = u32::MAX;

        self.add(node, (old_weight + 1).max(0));
    }

    /// Pop a node with the maximum weight. Tie-breaking is implicit
    /// (`pop` from back of the bucket — LIFO within a weight class).
    /// Panics if empty.
    fn pop_max(&mut self) -> u32 {
        let bucket = &mut self.buckets[self.max_weight];
        let node = bucket.pop().expect("McsBuckets::pop_max on empty queue");
        self.pos[node as usize] = u32::MAX;

        if bucket.is_empty() && self.max_weight > 0 {
            // Walk down to the next non-empty bucket.
            let mut new_max = 0;
            for w in (0..self.max_weight).rev() {
                if !self.buckets[w].is_empty() {
                    new_max = w;
                    break;
                }
            }
            self.max_weight = new_max;
        }
        node
    }

    fn is_empty(&self) -> bool {
        // Empty when even bucket 0 is empty and max_weight is 0.
        self.max_weight == 0 && self.buckets[0].is_empty()
    }
}

// ──────────────────────────── tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Cheap deterministic LCG so tests reproduce exactly without
    /// pulling in `rand`.
    struct Lcg {
        s: u32,
    }
    impl Lcg {
        fn new(seed: u32) -> Self {
            Self { s: seed }
        }
        fn next_u32(&mut self) -> u32 {
            self.s = self.s.wrapping_mul(2654435761).wrapping_add(0x12345678);
            self.s
        }
        fn next_f64(&mut self) -> f64 {
            self.next_u32() as f64 / u32::MAX as f64
        }
    }

    /// Assert that no two adjacent vertices share a color.
    fn assert_proper_coloring(adjacency: &[Vec<u32>], colors: &[u32]) {
        for (v, neighbours) in adjacency.iter().enumerate() {
            for &n in neighbours {
                assert_ne!(
                    colors[v], colors[n as usize],
                    "vertices {} and {} both have color {}",
                    v, n, colors[v]
                );
            }
        }
    }

    fn n_colors(colors: &[u32]) -> u32 {
        colors.iter().copied().max().map_or(0, |c| c + 1)
    }

    #[test]
    fn greedy_color_proper_coloring() {
        // Erdős–Rényi: n=50, p=0.2.
        let n = 50;
        let p = 0.2;
        let mut rng = Lcg::new(0xC0FFEE);
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.next_f64() < p {
                    adj[i].push(j as u32);
                    adj[j].push(i as u32);
                }
            }
        }
        let colors = greedy_color(&adj);
        assert_eq!(colors.len(), n);
        assert_proper_coloring(&adj, &colors);
    }

    #[test]
    fn greedy_color_simple_path() {
        // Path 0-1-2-...-99 is bipartite → 2 colors suffice.
        let n = 100;
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for i in 0..(n - 1) {
            adj[i].push((i + 1) as u32);
            adj[i + 1].push(i as u32);
        }
        let colors = greedy_color(&adj);
        assert_proper_coloring(&adj, &colors);
        assert_eq!(
            n_colors(&colors),
            2,
            "path graph should use exactly 2 colors, got {}",
            n_colors(&colors)
        );
    }

    #[test]
    fn greedy_color_complete_graph() {
        // K_5: every vertex adjacent to every other → 5 colors.
        let n = 5;
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i].push(j as u32);
                }
            }
        }
        let colors = greedy_color(&adj);
        assert_proper_coloring(&adj, &colors);
        assert_eq!(n_colors(&colors), 5);
    }

    #[test]
    fn mcs_color_typically_better_than_greedy_on_irregular() {
        // Hand-built crown-ish graph: a 6-cycle plus a central hub
        // connected to alternate vertices, plus a few cross-edges.
        // On graphs like this, MCS's perfect-elimination-flavored
        // ordering tends to do at least as well as degree-ordered
        // greedy.
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); 7];
        let edge = |a: u32, b: u32, adj: &mut Vec<Vec<u32>>| {
            adj[a as usize].push(b);
            adj[b as usize].push(a);
        };
        // 6-cycle 0-1-2-3-4-5-0
        edge(0, 1, &mut adj);
        edge(1, 2, &mut adj);
        edge(2, 3, &mut adj);
        edge(3, 4, &mut adj);
        edge(4, 5, &mut adj);
        edge(5, 0, &mut adj);
        // Hub 6 connected to 0, 2, 4
        edge(6, 0, &mut adj);
        edge(6, 2, &mut adj);
        edge(6, 4, &mut adj);
        // Cross-edge 1-4 to make the graph irregular.
        edge(1, 4, &mut adj);

        let g = greedy_color(&adj);
        let m = mcs_color(&adj);
        assert_proper_coloring(&adj, &g);
        assert_proper_coloring(&adj, &m);
        assert!(
            n_colors(&m) <= n_colors(&g),
            "MCS used {} colors, greedy used {}",
            n_colors(&m),
            n_colors(&g)
        );
    }

    /// Build a W x H grid cloth's edge constraints (4 vertices →
    /// each adjacent pair in horizontal & vertical direction).
    fn grid_cloth_edges(w: u32, h: u32) -> Vec<Vec<u32>> {
        let idx = |x: u32, y: u32| y * w + x;
        let mut edges: Vec<Vec<u32>> = Vec::new();
        // Horizontal
        for y in 0..h {
            for x in 0..(w - 1) {
                edges.push(vec![idx(x, y), idx(x + 1, y)]);
            }
        }
        // Vertical
        for y in 0..(h - 1) {
            for x in 0..w {
                edges.push(vec![idx(x, y), idx(x, y + 1)]);
            }
        }
        edges
    }

    #[test]
    fn color_constraints_distance_constraint_graph() {
        // 4x4 grid → 12 horizontal + 12 vertical = 24 edges.
        let edges = grid_cloth_edges(4, 4);
        assert_eq!(edges.len(), 24);

        let colors = color_constraints(&edges);
        assert_eq!(colors.len(), 24);

        // Verify: any two constraints sharing a vertex get different
        // colors.
        for i in 0..edges.len() {
            for j in (i + 1)..edges.len() {
                let share = edges[i].iter().any(|v| edges[j].contains(v));
                if share {
                    assert_ne!(
                        colors[i], colors[j],
                        "constraints {} {:?} and {} {:?} share a vertex but both have color {}",
                        i, edges[i], j, edges[j], colors[i]
                    );
                }
            }
        }

        // Theory: 4 colors suffice (two horizontal + two vertical
        // stripes). A reasonable greedy can blow up to 5-6; 6 is the
        // upper sanity bound. MCS hits the optimum 4 on this graph.
        let n = n_colors(&colors);
        assert!(
            n <= 6,
            "4x4 cloth coloring used {} colors, expected <= 6",
            n
        );
        // Sanity: it had better use at least 2 (the graph is
        // non-trivial).
        assert!(n >= 2);
    }

    #[test]
    fn group_by_color_preserves_indices() {
        // Build a small coloring and check the inverse mapping is
        // exhaustive and disjoint.
        let edges = grid_cloth_edges(4, 4);
        let colors = color_constraints(&edges);
        let groups = group_by_color(&colors);

        // Outer length = max color + 1.
        let nc = n_colors(&colors) as usize;
        assert_eq!(groups.len(), nc);

        // Flat content equals `0..n_constraints`.
        let mut flat: Vec<usize> = groups.iter().flatten().copied().collect();
        flat.sort();
        let expected: Vec<usize> = (0..edges.len()).collect();
        assert_eq!(flat, expected);

        // No color group is empty (every color used at least once).
        for (c, g) in groups.iter().enumerate() {
            assert!(!g.is_empty(), "color group {} is empty", c);
        }
    }
}
