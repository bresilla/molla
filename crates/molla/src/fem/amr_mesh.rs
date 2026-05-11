//! Stage 45 — adaptive-mesh-refinement foundation for FEM.
//!
//! v1 shipped the **data-structure foundation** and **uniform refinement**.
//! v2 lands the constraint apparatus that makes *non-uniform* refinement
//! actually usable: hanging-node detection, **2:1 balance** enforcement
//! through propagated refinement, and a sparse **constraint matrix `T`**
//! that maps `u_free → u_total` so the caller can solve the reduced
//! `K_c = TᵀKT, f_c = Tᵀf` system on the free DoFs only.
//!
//! # Concepts
//!
//! - **Base mesh** — a regular [`FemMesh`] of triangles or tetrahedra.
//!   Cells of the base mesh are addressed by their flat index in
//!   `base_mesh.cells`. The base mesh is the canonical "level 0" mesh.
//!
//! - **Refined cells** — when a cell is refined we *append* its
//!   children to a growing flat cell pool together with the (new) edge-
//!   midpoint vertices. A [`RefinedCell`] records the parent → child
//!   relationship. The original parent stays in the pool but is marked
//!   as a non-leaf (its level is recorded in `refinement_levels` and the
//!   presence of a [`RefinedCell`] with `parent == idx` means it has
//!   been split).
//!
//! - **Tree** — the collection of [`RefinedCell`] entries forms a
//!   forest rooted at the base-mesh cells. Each [`RefinedCell`] holds
//!   `4` (tri) or `8` (tet) **flat cell indices** of its children.
//!
//! - **Flatten** — emit a [`FemMesh`] consisting of every *leaf* cell
//!   (a cell that has not itself been refined) plus the union of all
//!   vertices touched by those leaves. Standard FEM assembly
//!   (`assemble_global`) consumes the result unchanged.
//!
//! # Subdivision schemes
//!
//! - **Triangle, 1→4** — regular ("Loop" / red) refinement: insert the
//!   midpoint of each of the 3 edges, then emit 4 sub-triangles
//!   `(v0, m01, m20)`, `(v1, m12, m01)`, `(v2, m20, m12)`,
//!   `(m01, m12, m20)`. The centre triangle is the "inverted" one.
//!
//! - **Tetrahedron, 1→8** — Bey's red refinement: insert the midpoint
//!   of each of the 6 edges, emit 4 *corner* tetrahedra
//!   `(vi, mij, mik, mil)` for `i ∈ {0..3}`, then split the inner
//!   octahedron (formed by the 6 midpoints) into 4 tets along the
//!   `m01–m23` diagonal. Bey's classic ordering is used so that
//!   repeated refinement remains shape-regular (no "sliver"
//!   degeneracy after multiple levels).
//!
//! # Hanging nodes and the constraint matrix
//!
//! Non-uniform refinement produces *hanging nodes*: a midpoint vertex
//! `m` that lives on the shared edge of (a refined cell) and (an
//! unrefined neighbour). The unrefined neighbour does not see `m` as
//! one of its DoFs, so naive assembly leaves the linear system
//! inconsistent across the hanging edge. The standard fix is to
//! **constrain** the DoF at `m` to be the average of its two parent
//! vertices `(u_m = ½(u_a + u_b))` for P1 — encoded as a sparse
//! constraint matrix `T` that maps the *free* (non-hanging) DoF set to
//! the *full* DoF set.
//!
//! v2 enforces **2:1 balance**: after every `refine_where` we walk the
//! edge-adjacency graph and refine any leaf whose neighbour is more
//! than one level finer. This guarantees that every hanging-node midpoint
//! `m` has *both* parents non-hanging — the P1 average constraint
//! `u_m = ½(u_a + u_b)` is then well-defined.
//!
//! v2 ships P1 (`degree = 1`) constraints only; the P2 case (a
//! `½(a+b) + ⅛(a + b − 2m)` term on edges and a fitted face-mid
//! constraint for tets) is deferred.

use std::collections::{HashMap, HashSet};

use crate::core::Result;
use crate::math::sparse::BsrMatrix;
use crate::math::{Real, Vec3};

use super::assembly::assemble_global;
use super::function_space::FunctionSpace;
use super::mesh::{FemCell, FemMesh};

/// Per-parent record of a 1→N split.
///
/// `parent` is the flat cell index (into the AmrMesh cell pool) of the
/// cell that was split. `children` lists the flat cell indices of the
/// resulting child cells. `level` is the refinement level of the
/// **children** (i.e. parent.level + 1).
#[derive(Clone, Copy, Debug)]
pub struct RefinedCell {
    /// Flat index (in the AmrMesh cell pool) of the parent that was
    /// refined.
    pub parent: u32,
    /// Flat indices of the children. `Tri` → 4 indices, `Tet` → 8.
    pub children: RefinedChildren,
    /// Refinement level of the children (parent level + 1).
    pub level: u8,
}

/// Child-index payload — triangle splits to 4, tet to 8.
///
/// The variant intrinsically encodes the cell topology, matching the
/// shape of [`FemCell`]. Helpers `as_slice` and `len` make it easy to
/// iterate children without matching.
#[derive(Clone, Copy, Debug)]
pub enum RefinedChildren {
    Tri([u32; 4]),
    Tet([u32; 8]),
}

impl RefinedChildren {
    /// Borrow the children as a flat slice for iteration.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        match self {
            RefinedChildren::Tri(c) => c,
            RefinedChildren::Tet(c) => c,
        }
    }

    /// Number of children (4 for tri, 8 for tet).
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Convenience emptiness check (always `false` in v1; included so
    /// clippy doesn't complain about the bare `len`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}

/// One hanging-node DoF — a midpoint vertex `m` that appears on the
/// edge of *some* leaf without that leaf having been refined to
/// introduce it.
///
/// `dof` is the **AMR vertex id** of the hanging midpoint;
/// `parents` are the AMR vertex ids of the two endpoints of the edge
/// `m` sits on (the endpoints come from the unrefined leaf, so they
/// are themselves *not* hanging when 2:1 balance is enforced).
///
/// The P1 constraint encoded by this record is `u_m = ½(u_a + u_b)`.
#[derive(Clone, Copy, Debug)]
pub struct HangingNode {
    /// AMR vertex id of the constrained midpoint DoF.
    pub dof: u32,
    /// AMR vertex ids of the two parent endpoints (the edge `m` is on).
    pub parents: [u32; 2],
}

/// Refinement criterion — predicate over a single cell.
///
/// The trait is closure-friendly via the blanket `Fn` impl below: any
/// `Fn(&FemCell, &[Vec3]) -> bool` can be passed directly as a
/// criterion.
///
/// `vertices` is the **full AMR vertex array** (base + every midpoint
/// inserted so far); the cell's own vertex indices live in `cell.verts()`
/// and index into that array. Callers typically compute the cell's
/// centroid from those indices and test it against some spatial query.
pub trait RefinementCriterion {
    /// Return `true` if the cell should be refined.
    fn should_refine(&self, cell: &FemCell, vertices: &[Vec3]) -> bool;
}

impl<F> RefinementCriterion for F
where
    F: Fn(&FemCell, &[Vec3]) -> bool,
{
    #[inline]
    fn should_refine(&self, cell: &FemCell, vertices: &[Vec3]) -> bool {
        (self)(cell, vertices)
    }
}

/// Built-in: refine any cell whose centroid is within `radius` of `p`.
#[must_use]
pub fn near_point(p: Vec3, radius: Real) -> impl RefinementCriterion {
    move |cell: &FemCell, vertices: &[Vec3]| {
        let verts = cell.verts();
        let n = verts.len() as Real;
        let mut c = Vec3::new(0.0, 0.0, 0.0);
        for &v in verts {
            c += vertices[v as usize];
        }
        c *= 1.0 / n;
        let d2 = (c - p).dot(c - p);
        d2 <= radius * radius
    }
}

/// Adaptive-refinement mesh — base mesh + refinement tree.
///
/// In v1 every refinement produces a conforming, hanging-node-free
/// child set *as long as it is uniform across the mesh*. The struct
/// imposes no such global invariant (calling `refine_cell` on a single
/// cell will produce hanging nodes), but the assembly / flatten path
/// is only contractually correct under uniform refinement.
#[derive(Clone, Debug)]
pub struct AmrMesh {
    /// Original level-0 mesh — never modified.
    pub base_mesh: FemMesh,

    /// **Flat cell pool**. Indices `0..base_mesh.cell_count()` are the
    /// base cells (level 0). Indices beyond that are children added by
    /// refinement. Both leaves and refined parents live here; consult
    /// `is_refined[idx]` to distinguish.
    cells: Vec<FemCell>,

    /// **Flat vertex pool**. Indices `0..base_mesh.vert_count()` are
    /// the base vertices. Indices beyond that are edge midpoints
    /// inserted by refinement.
    vertices: Vec<Vec3>,

    /// Per-cell refinement level. `refinement_levels[idx]` is `0` for
    /// base cells; children inherit `parent.level + 1`.
    pub refinement_levels: Vec<u8>,

    /// Per-cell "has been split" marker. A cell is a *leaf* iff
    /// `!is_refined[idx]`.
    is_refined: Vec<bool>,

    /// One [`RefinedCell`] per non-leaf cell. Stored in refinement
    /// order (so `refined_cells[k].parent`'s `is_refined` was set by
    /// step `k`).
    pub refined_cells: Vec<RefinedCell>,

    /// Edge-midpoint cache — keyed on the *sorted* endpoint pair
    /// `(min, max)`. Lets us reuse the same midpoint vertex when two
    /// neighbouring cells refine the same edge, so the refined mesh
    /// stays watertight (no T-junction on a shared *fully refined*
    /// edge).
    midpoint_cache: HashMap<(u32, u32), u32>,

    /// Cached hanging-node records — one per midpoint DoF that lives on
    /// a leaf edge but was not introduced by that leaf's own
    /// refinement. Populated by [`Self::compute_hanging_nodes`], which
    /// is invoked automatically after every `refine_where` /
    /// `refine_uniform` call.
    hanging_nodes: Vec<HangingNode>,
}

impl AmrMesh {
    /// Wrap a base mesh into an [`AmrMesh`] with no refinement applied.
    #[must_use]
    pub fn new(base_mesh: FemMesh) -> Self {
        let cells = base_mesh.cells.clone();
        let vertices = base_mesh.vertices.clone();
        let n_cells = cells.len();
        Self {
            base_mesh,
            cells,
            vertices,
            refinement_levels: vec![0; n_cells],
            is_refined: vec![false; n_cells],
            refined_cells: Vec::new(),
            midpoint_cache: HashMap::new(),
            hanging_nodes: Vec::new(),
        }
    }

    /// Total number of cells in the flat pool (leaves + refined).
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Total number of vertices in the flat pool (base + midpoints).
    #[must_use]
    pub fn vert_count(&self) -> usize {
        self.vertices.len()
    }

    /// Borrow the full cell pool.
    #[must_use]
    pub fn cells(&self) -> &[FemCell] {
        &self.cells
    }

    /// Borrow the full vertex pool.
    #[must_use]
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// Number of *leaf* cells — those not yet refined.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.is_refined.iter().filter(|r| !**r).count()
    }

    /// Whether cell `idx` is a leaf (has not been refined).
    #[must_use]
    pub fn is_leaf(&self, idx: usize) -> bool {
        !self.is_refined[idx]
    }

    /// Get-or-insert the midpoint of edge `(a, b)`.
    fn midpoint(&mut self, a: u32, b: u32) -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&id) = self.midpoint_cache.get(&key) {
            return id;
        }
        let pa = self.vertices[a as usize];
        let pb = self.vertices[b as usize];
        let m = (pa + pb) * 0.5;
        let id = self.vertices.len() as u32;
        self.vertices.push(m);
        self.midpoint_cache.insert(key, id);
        id
    }

    /// Split a single cell. Panics if `cell_idx` is already refined.
    ///
    /// Triangle → 4 sub-triangles via regular (red) refinement.
    /// Tetrahedron → 8 sub-tetrahedra via Bey's red refinement.
    pub fn refine_cell(&mut self, cell_idx: usize) {
        assert!(
            !self.is_refined[cell_idx],
            "amr_mesh: cell {cell_idx} is already refined"
        );
        let parent_level = self.refinement_levels[cell_idx];
        let child_level = parent_level + 1;
        let parent_cell = self.cells[cell_idx];

        let children = match parent_cell {
            FemCell::Tri(v) => self.refine_tri(v, child_level),
            FemCell::Tet(v) => self.refine_tet(v, child_level),
        };

        self.is_refined[cell_idx] = true;
        self.refined_cells.push(RefinedCell {
            parent: cell_idx as u32,
            children,
            level: child_level,
        });
    }

    /// Refine every *leaf* cell once. After this call every cell that
    /// was a leaf gets split into 4 (tri) or 8 (tet) children, and
    /// the new children become the leaves. The hanging-node cache is
    /// refreshed at the end — uniform refinement produces no hanging
    /// nodes, but rebuilding ensures the cache stays consistent across
    /// repeated calls.
    pub fn refine_uniform(&mut self) {
        // Snapshot the current leaf indices — we mutate `cells` /
        // `is_refined` inside the loop, so we must not iterate them
        // directly.
        let leaves: Vec<usize> = (0..self.cells.len())
            .filter(|&i| !self.is_refined[i])
            .collect();
        for idx in leaves {
            self.refine_cell(idx);
        }
        self.compute_hanging_nodes();
    }

    /// Refine every leaf cell that `criterion` accepts, then propagate
    /// refinement until the **2:1 balance** invariant is restored:
    /// no two edge-adjacent leaves differ in refinement level by more
    /// than one. After balance is restored, `compute_hanging_nodes` is
    /// rebuilt so the constraint apparatus sees the current leaf set.
    ///
    /// 2:1 balance is what makes the P1 hanging-node constraint
    /// `u_m = ½(u_a + u_b)` *well-defined*: with the balance enforced,
    /// neither parent endpoint of `m` is itself hanging.
    pub fn refine_where<C: RefinementCriterion>(&mut self, criterion: C) {
        let to_refine: Vec<usize> = (0..self.cells.len())
            .filter(|&i| !self.is_refined[i])
            .filter(|&i| criterion.should_refine(&self.cells[i], &self.vertices))
            .collect();
        for idx in to_refine {
            self.refine_cell(idx);
        }
        self.propagate_refinement();
        self.compute_hanging_nodes();
    }

    /// Build a [`FemMesh`] containing every leaf cell + the union of
    /// vertices they touch.
    ///
    /// The vertex set is **compacted** — the output mesh only keeps
    /// vertices that are referenced by at least one leaf. Vertex
    /// ordering is "first touch in leaf-iteration order" so the
    /// flattened mesh is deterministic; downstream solvers should not
    /// rely on a specific ordering for correctness.
    #[must_use]
    pub fn flatten(&self) -> FemMesh {
        self.flatten_with_remap().0
    }

    /// Like [`Self::flatten`] but also returns the
    /// AMR-vertex-id → flat-vertex-id remap. Required by
    /// [`Self::build_constraint_matrix`] to translate hanging-node
    /// records (stored in AMR ids) into flat-DoF coordinates.
    #[must_use]
    pub fn flatten_with_remap(&self) -> (FemMesh, HashMap<u32, u32>) {
        // Compact vertex remap: AMR-vertex-id → flat-mesh-vertex-id.
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut cells: Vec<FemCell> = Vec::new();

        for (idx, cell) in self.cells.iter().enumerate() {
            if self.is_refined[idx] {
                continue;
            }
            let verts = cell.verts();
            // Remap each vertex through `remap`, inserting if new.
            let mut local: [u32; 4] = [0; 4];
            for (k, &v) in verts.iter().enumerate() {
                let flat = *remap.entry(v).or_insert_with(|| {
                    let id = vertices.len() as u32;
                    vertices.push(self.vertices[v as usize]);
                    id
                });
                local[k] = flat;
            }
            match cell {
                FemCell::Tri(_) => cells.push(FemCell::Tri([local[0], local[1], local[2]])),
                FemCell::Tet(_) => {
                    cells.push(FemCell::Tet([local[0], local[1], local[2], local[3]]))
                }
            }
        }

        (FemMesh { vertices, cells }, remap)
    }

    /// Borrow the cached hanging-node list. Empty unless
    /// [`Self::compute_hanging_nodes`] (or a `refine_*` wrapper that
    /// calls it) has been run.
    #[must_use]
    pub fn hanging_nodes(&self) -> &[HangingNode] {
        &self.hanging_nodes
    }

    /// Walk every leaf's edges and record every midpoint that lives on
    /// an edge of an unrefined leaf — that midpoint is a *hanging
    /// node* on this leaf's edge.
    ///
    /// Algorithm:
    /// - For every leaf cell, list its edges as `(min, max)` endpoint
    ///   pairs (3 per triangle, 6 per tet).
    /// - For each edge `(a, b)`, look up the midpoint cache. If
    ///   present, the edge was split somewhere — but only by a
    ///   *neighbouring* refinement (this leaf itself uses `(a, b)` as
    ///   a flat edge). The midpoint `m` is then a hanging-node DoF for
    ///   this leaf.
    /// - Deduplicate by the hanging midpoint id (the same midpoint can
    ///   show up on edges of multiple coarse leaves; one record is
    ///   enough).
    ///
    /// The result is stored on `self.hanging_nodes` and returned by
    /// the accessor [`Self::hanging_nodes`]. The order is the order in
    /// which hanging midpoints are first encountered while walking
    /// leaves; downstream code should not rely on a specific ordering
    /// for correctness.
    pub fn compute_hanging_nodes(&mut self) {
        let mut seen: HashSet<u32> = HashSet::new();
        let mut result: Vec<HangingNode> = Vec::new();

        for (idx, cell) in self.cells.iter().enumerate() {
            if self.is_refined[idx] {
                continue;
            }
            for (a, b) in leaf_edges(cell) {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = self.midpoint_cache.get(&key) {
                    if seen.insert(m) {
                        result.push(HangingNode {
                            dof: m,
                            parents: [key.0, key.1],
                        });
                    }
                }
            }
        }

        self.hanging_nodes = result;
    }

    /// Restore the **2:1 balance** invariant: walk edge-adjacency
    /// between leaves and refine any coarse leaf that shares an edge
    /// whose deeper neighbours have split it more than once.
    ///
    /// Implementation:
    /// - For every leaf, scan its edges; for each edge `(a, b)`,
    ///   recursively walk the midpoint cache to find the deepest
    ///   *split depth* a neighbour has reached on this edge.
    /// - If `depth > 1`, refining this leaf once shrinks the gap by
    ///   one and produces children of level `parent_level + 1`. The
    ///   process iterates until no leaf has an edge with depth `> 1`.
    ///
    /// The iteration is bounded by the maximum refinement level
    /// already present in the mesh: every refinement strictly
    /// increases the coarsest leaf level, so the BFS terminates in at
    /// most `max_level + 1` outer rounds.
    pub fn propagate_refinement(&mut self) {
        loop {
            let mut to_refine: Vec<usize> = Vec::new();
            for (idx, cell) in self.cells.iter().enumerate() {
                if self.is_refined[idx] {
                    continue;
                }
                for (a, b) in leaf_edges(cell) {
                    if self.edge_split_depth(a, b) > 1 {
                        to_refine.push(idx);
                        break;
                    }
                }
            }
            if to_refine.is_empty() {
                break;
            }
            for idx in to_refine {
                // A leaf may appear multiple times if multiple edges
                // violate; the `is_refined` check inside `refine_cell`
                // would panic on the second attempt, so we filter
                // explicitly.
                if !self.is_refined[idx] {
                    self.refine_cell(idx);
                }
            }
        }
    }

    /// How many recursive midpoint splits are present on edge `(a, b)`
    /// in the midpoint cache. `0` means the edge has not been
    /// subdivided at all; `1` means a single midpoint `m` exists
    /// (one neighbour-level deeper); `≥ 2` means at least one of the
    /// sub-edges `(a, m)` or `(m, b)` has itself been split.
    fn edge_split_depth(&self, a: u32, b: u32) -> usize {
        let key = if a < b { (a, b) } else { (b, a) };
        match self.midpoint_cache.get(&key) {
            None => 0,
            Some(&m) => {
                1 + self
                    .edge_split_depth(a, m)
                    .max(self.edge_split_depth(m, b))
            }
        }
    }

    /// Build the P1 hanging-node constraint matrix `T` mapping
    /// `u_free → u_total` for the **current flat-mesh DoF set**.
    ///
    /// `T` has shape `(n_total × n_free)` where `n_total` is the
    /// flat-mesh DoF count and `n_free` is the count of non-hanging
    /// DoFs. The row for a non-hanging DoF is a standard identity
    /// indicator; the row for a hanging DoF `m` with parents
    /// `(a, b)` is `½` at the free-index columns for `a` and `b`.
    ///
    /// `degree` must be `1` in v2 — the P2 constraint is the v3 work
    /// item.
    ///
    /// `remap` is the AMR-vertex-id → flat-vertex-id map returned by
    /// [`Self::flatten_with_remap`]. Building it inside this routine
    /// would duplicate work the caller has typically already done as
    /// part of assembling `K`.
    pub fn build_constraint_matrix(
        &self,
        degree: usize,
        remap: &HashMap<u32, u32>,
    ) -> Result<BsrMatrix> {
        assert_eq!(
            degree, 1,
            "build_constraint_matrix: v2 ships P1 (degree = 1) only"
        );
        let n_total = remap.len();

        // Hanging DoF set in flat-index space — only those hanging
        // midpoints whose AMR id was actually used by a leaf (i.e.
        // entered the flatten remap) get a constraint row. Hanging
        // midpoints belong to leaf edges by construction, so they all
        // map; the filter guards against pathological hand-built test
        // cases.
        let mut is_hanging_flat: Vec<bool> = vec![false; n_total];
        let mut hanging_parents_flat: Vec<Option<(u32, u32)>> =
            vec![None; n_total];
        for hn in &self.hanging_nodes {
            let Some(&flat_h) = remap.get(&hn.dof) else {
                continue;
            };
            let Some(&fp1) = remap.get(&hn.parents[0]) else {
                continue;
            };
            let Some(&fp2) = remap.get(&hn.parents[1]) else {
                continue;
            };
            is_hanging_flat[flat_h as usize] = true;
            hanging_parents_flat[flat_h as usize] = Some((fp1, fp2));
        }

        // Free DoFs are the non-hanging ones. Assign each a column
        // index in the same order as the full DoF iteration so the
        // resulting `T` row-pattern stays deterministic.
        let mut free_index: Vec<i32> = vec![-1; n_total];
        let mut n_free = 0usize;
        for i in 0..n_total {
            if !is_hanging_flat[i] {
                free_index[i] = n_free as i32;
                n_free += 1;
            }
        }

        // Emit triplets.
        let mut triplets: Vec<(usize, usize, Real)> = Vec::with_capacity(n_total + 2 * self.hanging_nodes.len());
        for i in 0..n_total {
            if is_hanging_flat[i] {
                let (p1, p2) = hanging_parents_flat[i]
                    .expect("hanging DoF marked but parents missing");
                debug_assert!(
                    !is_hanging_flat[p1 as usize] && !is_hanging_flat[p2 as usize],
                    "hanging-node parent is itself hanging — 2:1 balance violated"
                );
                let c1 = free_index[p1 as usize] as usize;
                let c2 = free_index[p2 as usize] as usize;
                triplets.push((i, c1, 0.5));
                triplets.push((i, c2, 0.5));
            } else {
                let col = free_index[i] as usize;
                triplets.push((i, col, 1.0));
            }
        }

        BsrMatrix::from_scalar_triplets(n_total, n_free, &triplets)
    }
}

/// Iterate the unordered edges of a single FEM cell as `(a, b)`
/// pairs — 3 for a triangle, 6 for a tetrahedron. Used by both the
/// hanging-node walk and the 2:1-balance propagation.
fn leaf_edges(cell: &FemCell) -> Vec<(u32, u32)> {
    match cell {
        FemCell::Tri(v) => vec![(v[0], v[1]), (v[1], v[2]), (v[2], v[0])],
        FemCell::Tet(v) => vec![
            (v[0], v[1]),
            (v[0], v[2]),
            (v[0], v[3]),
            (v[1], v[2]),
            (v[1], v[3]),
            (v[2], v[3]),
        ],
    }
}

/// Assemble the reduced AMR system
/// `K_c = TᵀKT, f_c = Tᵀf, T = constraint matrix`.
///
/// `local_fn` is the element kernel — same signature as
/// [`assemble_global`]. `f_full` is the full-mesh RHS to be reduced;
/// `None` is treated as a zero vector of the right length (Laplace
/// case).
///
/// The caller is responsible for **applying Dirichlet BCs to the
/// reduced system** (`K_c, f_c`) — that is the textbook ordering:
/// constraint-reduce first, BC-pin second. After solving
/// `K_c · u_free = f_c`, the full-DoF solution is recovered as
/// `u = T · u_free` (which the caller can compute via
/// [`BsrMatrix::mv`]).
///
/// Returns `(K_c, f_c, T)`.
pub fn assemble_global_amr<F>(
    amr_mesh: &AmrMesh,
    space: FunctionSpace,
    local_fn: F,
    f_full: Option<&[Real]>,
) -> Result<(BsrMatrix, Vec<Real>, BsrMatrix)>
where
    F: Fn(&FemCell, &[Vec3]) -> Vec<Vec<Real>>,
{
    assert!(
        matches!(space, FunctionSpace::P1Nodal),
        "assemble_global_amr: v2 ships P1Nodal only"
    );

    let (flat, remap) = amr_mesh.flatten_with_remap();
    let n_total = flat.vert_count();
    let k_full = assemble_global(&flat, space, local_fn)?;

    let f_full_owned: Vec<Real> = match f_full {
        Some(f) => {
            assert_eq!(
                f.len(),
                n_total,
                "assemble_global_amr: f_full length ({}) != flat DoF count ({})",
                f.len(),
                n_total
            );
            f.to_vec()
        }
        None => vec![0.0; n_total],
    };

    let t = amr_mesh.build_constraint_matrix(1, &remap)?;
    let n_free = t.ncols();

    // K_c = T^T · K · T. Compute it via triplets: walk the dense
    // intermediate via per-column matvec. For Stage 45 mesh sizes
    // (small) the O(n_free · nnz(K)) cost is acceptable.
    let k_c = compute_ttkt(&t, &k_full)?;

    // f_c = T^T · f. Vector form — explicit transpose-times-vector.
    let f_c = tt_times_vector(&t, &f_full_owned);

    debug_assert_eq!(k_c.nrows(), n_free);
    debug_assert_eq!(k_c.ncols(), n_free);
    debug_assert_eq!(f_c.len(), n_free);

    Ok((k_c, f_c, t))
}

/// `Tᵀ · v` for a (`n_total × n_free`) sparse `T` and a length-`n_total`
/// vector `v`. Output has length `n_free`.
#[must_use]
pub fn tt_times_vector(t: &BsrMatrix, v: &[Real]) -> Vec<Real> {
    debug_assert_eq!(t.b, 1);
    debug_assert_eq!(t.nrows(), v.len());
    let mut out = vec![0.0; t.ncols()];
    for r in 0..t.block_rows {
        let lo = t.row_offsets[r];
        let hi = t.row_offsets[r + 1];
        for slot in lo..hi {
            let c = t.col_idx[slot];
            out[c] += t.values[slot] * v[r];
        }
    }
    out
}

/// Compute `Tᵀ · K · T` for sparse `T` (n_total × n_free) and `K`
/// (n_total × n_total), returning a sparse `BsrMatrix` of shape
/// `(n_free × n_free)`.
fn compute_ttkt(t: &BsrMatrix, k: &BsrMatrix) -> Result<BsrMatrix> {
    debug_assert_eq!(t.b, 1);
    debug_assert_eq!(k.b, 1);
    debug_assert_eq!(t.nrows(), k.nrows());
    debug_assert_eq!(t.nrows(), k.ncols());

    let n_total = t.nrows();
    let n_free = t.ncols();

    // For each column α ∈ [0, n_free): t_α is a sparse column of T
    // (length n_total). Compute u = K · t_α (a dense temporary of
    // length n_total), then for each non-zero T[i, β], the entry
    // K_c[β, α] += T[i, β] · u[i].
    //
    // Build a column-major view of T by inverting the row offsets.
    let mut col_to_rows: Vec<Vec<(usize, Real)>> = vec![Vec::new(); n_free];
    for r in 0..n_total {
        let lo = t.row_offsets[r];
        let hi = t.row_offsets[r + 1];
        for slot in lo..hi {
            let c = t.col_idx[slot];
            let v = t.values[slot];
            col_to_rows[c].push((r, v));
        }
    }

    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    let mut t_col_dense = vec![0.0; n_total];
    let mut u = vec![0.0; n_total];
    for alpha in 0..n_free {
        // Densify t_α.
        for &(r, v) in &col_to_rows[alpha] {
            t_col_dense[r] = v;
        }
        k.mv(&t_col_dense, &mut u)?;
        // For each β with non-zero column in T, accumulate
        // K_c[β, α] = sum_i T[i, β] · u[i].
        for beta in 0..n_free {
            let mut acc = 0.0;
            for &(r, v) in &col_to_rows[beta] {
                acc += v * u[r];
            }
            if acc != 0.0 {
                triplets.push((beta, alpha, acc));
            }
        }
        // Re-zero the dense buffer.
        for &(r, _) in &col_to_rows[alpha] {
            t_col_dense[r] = 0.0;
        }
    }

    BsrMatrix::from_scalar_triplets(n_free, n_free, &triplets)
}

impl AmrMesh {
    /// Expand a free-DoF solution `u_free` into the full-DoF vector
    /// `u = T · u_free`. Convenience wrapper around
    /// [`crate::math::sparse::BsrMatrix::mv`].
    #[must_use]
    pub fn expand_free_solution(t: &BsrMatrix, u_free: &[Real]) -> Vec<Real> {
        let mut u = vec![0.0; t.nrows()];
        t.mv(u_free, &mut u).expect("T · u_free matvec");
        u
    }

    // ── Internal — per-shape splitters ────────────────────────────────

    /// Regular 1→4 split of a triangle. Returns the 4 newly-appended
    /// flat cell indices in order
    ///   [corner-at-v0, corner-at-v1, corner-at-v2, centre].
    fn refine_tri(&mut self, v: [u32; 3], child_level: u8) -> RefinedChildren {
        let m01 = self.midpoint(v[0], v[1]);
        let m12 = self.midpoint(v[1], v[2]);
        let m20 = self.midpoint(v[2], v[0]);

        let base = self.cells.len() as u32;
        // Four sub-triangles. Each corner sub-tri keeps the orientation
        // of the parent (counter-clockwise in 2D ↔ positive area in
        // `compute_element_*`). The centre tri is also CCW.
        self.cells.push(FemCell::Tri([v[0], m01, m20]));
        self.cells.push(FemCell::Tri([m01, v[1], m12]));
        self.cells.push(FemCell::Tri([m20, m12, v[2]]));
        self.cells.push(FemCell::Tri([m01, m12, m20]));

        for _ in 0..4 {
            self.refinement_levels.push(child_level);
            self.is_refined.push(false);
        }

        RefinedChildren::Tri([base, base + 1, base + 2, base + 3])
    }

    /// Bey's 1→8 red refinement of a tetrahedron. Returns the 8 newly-
    /// appended flat cell indices.
    ///
    /// Vertex labelling: parent corners `v0, v1, v2, v3`; edge
    /// midpoints `m01, m02, m03, m12, m13, m23`.
    ///
    /// The first 4 children are the **corner tetrahedra**:
    ///   T0 = (v0, m01, m02, m03)
    ///   T1 = (v1, m12, m01, m13)
    ///   T2 = (v2, m02, m12, m23)
    ///   T3 = (v3, m03, m23, m13)
    /// each containing one parent corner + the 3 midpoints of the
    /// edges that meet at it. Each has 1/8 the parent volume.
    /// The midpoint ordering is permuted so each corner sub-tet has
    /// **positive volume** under the canonical
    /// `det([b-a; c-a; d-a]) > 0` orientation convention.
    ///
    /// The remaining 4 children are the **octahedron split** — the
    /// interior octahedron with vertices `(m01, m02, m03, m12, m13,
    /// m23)` is split along the `m01–m23` diagonal into 4 sub-tets:
    ///   T4 = (m01, m02, m03, m23)
    ///   T5 = (m01, m02, m12, m23)
    ///   T6 = (m01, m12, m13, m23)
    ///   T7 = (m01, m13, m03, m23)
    /// This is Bey's canonical "diagonal" choice — see Bey 1995,
    /// "Tetrahedral grid refinement". All 8 sub-tets are positively
    /// oriented if the parent is.
    fn refine_tet(&mut self, v: [u32; 4], child_level: u8) -> RefinedChildren {
        let m01 = self.midpoint(v[0], v[1]);
        let m02 = self.midpoint(v[0], v[2]);
        let m03 = self.midpoint(v[0], v[3]);
        let m12 = self.midpoint(v[1], v[2]);
        let m13 = self.midpoint(v[1], v[3]);
        let m23 = self.midpoint(v[2], v[3]);

        let base = self.cells.len() as u32;
        // 4 corner tets.
        self.cells.push(FemCell::Tet([v[0], m01, m02, m03]));
        self.cells.push(FemCell::Tet([v[1], m12, m01, m13]));
        self.cells.push(FemCell::Tet([v[2], m02, m12, m23]));
        self.cells.push(FemCell::Tet([v[3], m03, m23, m13]));
        // 4 octahedron tets along the m01–m23 diagonal.
        self.cells.push(FemCell::Tet([m01, m02, m03, m23]));
        self.cells.push(FemCell::Tet([m01, m02, m23, m12]));
        self.cells.push(FemCell::Tet([m01, m12, m23, m13]));
        self.cells.push(FemCell::Tet([m01, m13, m23, m03]));

        for _ in 0..8 {
            self.refinement_levels.push(child_level);
            self.is_refined.push(false);
        }

        RefinedChildren::Tet([
            base,
            base + 1,
            base + 2,
            base + 3,
            base + 4,
            base + 5,
            base + 6,
            base + 7,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_amr_has_no_refinement() {
        let m = FemMesh::tri_grid_2d(2, 2, 1.0, 1.0);
        let amr = AmrMesh::new(m.clone());
        assert_eq!(amr.cell_count(), m.cell_count());
        assert_eq!(amr.vert_count(), m.vert_count());
        assert_eq!(amr.leaf_count(), m.cell_count());
        assert!(amr.refined_cells.is_empty());
    }

    #[test]
    fn refine_single_tri_yields_four_children() {
        let m = FemMesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            cells: vec![FemCell::Tri([0, 1, 2])],
        };
        let mut amr = AmrMesh::new(m);
        amr.refine_cell(0);
        assert_eq!(amr.cell_count(), 5); // parent + 4 children
        assert_eq!(amr.leaf_count(), 4);
        assert_eq!(amr.vert_count(), 6); // 3 corners + 3 midpoints
        assert_eq!(amr.refined_cells.len(), 1);
        assert_eq!(amr.refined_cells[0].parent, 0);
        assert_eq!(amr.refined_cells[0].children.len(), 4);
        assert_eq!(amr.refined_cells[0].level, 1);
    }

    #[test]
    fn refine_single_tet_yields_eight_children() {
        let m = FemMesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            cells: vec![FemCell::Tet([0, 1, 2, 3])],
        };
        let mut amr = AmrMesh::new(m);
        amr.refine_cell(0);
        assert_eq!(amr.cell_count(), 9); // parent + 8 children
        assert_eq!(amr.leaf_count(), 8);
        assert_eq!(amr.vert_count(), 10); // 4 corners + 6 midpoints
        assert_eq!(amr.refined_cells[0].children.len(), 8);

        // All children must have strictly positive volume.
        for &cid in amr.refined_cells[0].children.as_slice() {
            if let FemCell::Tet(vs) = amr.cells[cid as usize] {
                let a = amr.vertices[vs[0] as usize];
                let b = amr.vertices[vs[1] as usize];
                let c = amr.vertices[vs[2] as usize];
                let d = amr.vertices[vs[3] as usize];
                let vol = (b - a).dot((c - a).cross(d - a)) / 6.0;
                assert!(vol > 0.0, "tet child has non-positive volume {vol}");
            }
        }
    }

    #[test]
    fn flatten_after_uniform_refines_quadruples_tri_count() {
        let m = FemMesh::tri_grid_2d(2, 2, 1.0, 1.0);
        let original_cells = m.cell_count();
        let mut amr = AmrMesh::new(m);
        amr.refine_uniform();
        let flat = amr.flatten();
        assert_eq!(flat.cells.len(), 4 * original_cells);
    }

    #[test]
    fn double_refinement_works() {
        // After 2 uniform refinements, leaf count should be 4² per
        // original tri = 16x.
        let m = FemMesh::tri_grid_2d(1, 1, 1.0, 1.0);
        let mut amr = AmrMesh::new(m);
        amr.refine_uniform();
        amr.refine_uniform();
        assert_eq!(amr.leaf_count(), 16 * 2); // 2 base tris × 16
    }

    #[test]
    fn near_point_criterion_picks_origin_cell_only() {
        // 2×2 unit grid → cells touching the origin are the 2 tris of
        // the bottom-left quad. A small radius around the origin
        // should refine only those.
        let m = FemMesh::tri_grid_2d(2, 2, 1.0, 1.0);
        let mut amr = AmrMesh::new(m);
        amr.refine_where(near_point(Vec3::new(0.25, 0.25, 0.0), 0.2));
        // Only the bottom-left two tris (centroids near (~0.2, ~0.2))
        // should be flagged.
        let refined_count = amr.refined_cells.len();
        assert!(
            (1..=2).contains(&refined_count),
            "expected 1 or 2 refinements near origin, got {refined_count}"
        );
    }
}
