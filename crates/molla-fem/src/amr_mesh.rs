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
//! v3 extends the constraint apparatus with:
//!
//! - **P2 hanging-node constraints** — the vertex midpoint constraint
//!   (`u_m = u_M` where `M` is the parent edge's P2 mid-edge DoF) plus
//!   the two **sub-edge** P2 mid-edge DoF constraints derived from
//!   evaluating the parent's P2 polynomial at `ξ = 1/4` and `ξ = 3/4`:
//!
//!   ```text
//!   u(ξ) = u_a (1-ξ)(1-2ξ) + u_b ξ(2ξ-1) + 4 u_M ξ(1-ξ)
//!   ⇒ u(1/4) = (3/8) u_a + (-1/8) u_b + (3/4) u_M
//!   ⇒ u(3/4) = (-1/8) u_a + (3/8) u_b + (3/4) u_M
//!   ⇒ u(1/2) = u_M                        (the vertex-midpoint case)
//!   ```
//!
//!   These three constraints together exactly reproduce a quadratic
//!   field on a hanging P2 edge.
//!
//! - **DG-AMR constraint apparatus** — DG has no inter-cell DoF
//!   sharing, so the analogue of a hanging-node constraint is a face-
//!   flux balance condition. See
//!   [`AmrMesh::compute_dg_amr_faces`]: each parent face on the
//!   coarse side maps onto two child sub-faces on the refined side, and
//!   the per-face flux integral on the parent must equal the sum over
//!   the children.
//!
//! - **State interpolation** between refinement events — see
//!   [`AmrMesh::transfer_state`]: evaluate a P1 field on the *old*
//!   AMR/flat mesh at every DoF location of the *new* AMR/flat mesh.

use std::collections::{HashMap, HashSet};

use crate::core::{Error, Result};
use crate::math::iterative::{CgResult, cg_solve};
use crate::math::sparse::BsrMatrix;
use crate::math::{Real, Vec3};

use super::assembly::assemble_global;
use super::function_space::FunctionSpace;
use super::mesh::{FemCell, FemMesh};
use super::quadrature::tri_quadrature_3pt;

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
    /// `!is_refined[idx] && !is_removed[idx]`.
    is_refined: Vec<bool>,

    /// Per-cell "has been coarsened away" marker. v4 introduces
    /// coarsening; when a parent cell is coarsened, all of its children
    /// become *removed* — they remain in the flat pool for index
    /// stability but are skipped by every leaf walk
    /// (`flatten`, `compute_hanging_nodes`, `propagate_refinement`,
    /// `compute_dg_amr_faces`, …). Removed cells are never refined or
    /// flattened. The flag is monotone — a removed cell is never
    /// re-activated; if a coarsened parent is later re-refined it
    /// receives *fresh* child indices appended to the pool.
    is_removed: Vec<bool>,

    /// One [`RefinedCell`] per non-leaf, non-removed cell. Stored in
    /// refinement order (so `refined_cells[k].parent`'s `is_refined`
    /// was set by step `k`). v4 coarsening **removes** entries from
    /// this list — callers iterating `refined_cells` see only currently
    /// active refinements.
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
            is_removed: vec![false; n_cells],
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

    /// Number of *leaf* cells — those not yet refined and not removed
    /// by coarsening.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        (0..self.cells.len())
            .filter(|&i| !self.is_refined[i] && !self.is_removed[i])
            .count()
    }

    /// Whether cell `idx` is a *leaf* — not refined and not removed by
    /// coarsening.
    #[must_use]
    pub fn is_leaf(&self, idx: usize) -> bool {
        !self.is_refined[idx] && !self.is_removed[idx]
    }

    /// Whether cell `idx` has been *removed* (coarsened away) — kept in
    /// the flat pool for index stability but invisible to every leaf
    /// walk.
    #[must_use]
    pub fn is_removed_cell(&self, idx: usize) -> bool {
        self.is_removed[idx]
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

    /// Split a single cell. Panics if `cell_idx` is already refined or
    /// has been removed by a prior coarsening.
    ///
    /// Triangle → 4 sub-triangles via regular (red) refinement.
    /// Tetrahedron → 8 sub-tetrahedra via Bey's red refinement.
    pub fn refine_cell(&mut self, cell_idx: usize) {
        assert!(
            !self.is_refined[cell_idx],
            "amr_mesh: cell {cell_idx} is already refined"
        );
        assert!(
            !self.is_removed[cell_idx],
            "amr_mesh: cell {cell_idx} has been removed by coarsening"
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
        // directly. Skip removed (coarsened-away) cells.
        let leaves: Vec<usize> = (0..self.cells.len())
            .filter(|&i| !self.is_refined[i] && !self.is_removed[i])
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
            .filter(|&i| !self.is_refined[i] && !self.is_removed[i])
            .filter(|&i| criterion.should_refine(&self.cells[i], &self.vertices))
            .collect();
        for idx in to_refine {
            self.refine_cell(idx);
        }
        self.propagate_refinement();
        self.compute_hanging_nodes();
    }

    /// Stage 45 v4 — **coarsen** (un-refine) a single parent cell: mark
    /// each of its `4` (tri) or `8` (tet) children as removed and
    /// restore the parent itself to leaf status.
    ///
    /// The operation only succeeds if every precondition holds:
    /// - The parent has actually been refined (`is_refined[parent_idx]`
    ///   is true and a `RefinedCell` record exists).
    /// - All of the parent's children are themselves leaves — none has
    ///   been further refined. (Coarsening a sub-tree first requires
    ///   coarsening its deepest leaves; v4 does not recurse.)
    /// - The midpoints that would be orphaned do not currently anchor
    ///   any hanging-node constraint that a neighbouring leaf still
    ///   needs. (After coarsening, a midpoint becomes orphaned iff no
    ///   other still-refined parent uses it on one of its edges.) The
    ///   2:1-balance invariant after the coarsening must continue to
    ///   hold — i.e. no leaf edge ends up with split depth `> 1`.
    ///
    /// On success, the parent's `is_refined` flag flips back to `false`
    /// (returning it to leaf status), the children's `is_removed`
    /// flags flip to `true`, the `RefinedCell` entry is dropped, and
    /// the midpoint cache is rebuilt to reflect the new refined set.
    /// Hanging-node records are *not* recomputed — call
    /// [`Self::compute_hanging_nodes`] (or use the batched
    /// [`Self::coarsen_where`] wrapper, which does so for you).
    ///
    /// On failure, the mesh state is left **unchanged** — no partial
    /// coarsening is applied. Failures return `Error::Fem(_)`.
    pub fn coarsen_cell(&mut self, parent_idx: usize) -> Result<()> {
        if parent_idx >= self.cells.len() {
            return Err(Error::Fem(format!(
                "coarsen_cell: parent index {parent_idx} out of bounds (cell count {})",
                self.cells.len()
            )));
        }
        if self.is_removed[parent_idx] {
            return Err(Error::Fem(format!(
                "coarsen_cell: parent cell {parent_idx} has already been removed"
            )));
        }
        if !self.is_refined[parent_idx] {
            return Err(Error::Fem(format!(
                "coarsen_cell: parent cell {parent_idx} is a leaf, nothing to coarsen"
            )));
        }
        // Locate the matching RefinedCell record.
        let rec_idx = self
            .refined_cells
            .iter()
            .position(|r| r.parent as usize == parent_idx)
            .ok_or_else(|| {
                Error::Fem(format!(
                    "coarsen_cell: no RefinedCell record for parent {parent_idx} \
                     (mesh state is internally inconsistent)"
                ))
            })?;
        let record = self.refined_cells[rec_idx];

        // All children must be leaves themselves — no recursive
        // refinement.
        for &c in record.children.as_slice() {
            let ci = c as usize;
            if self.is_refined[ci] {
                return Err(Error::Fem(format!(
                    "coarsen_cell: child cell {ci} of parent {parent_idx} \
                     is itself refined — coarsen the sub-tree first"
                )));
            }
            if self.is_removed[ci] {
                return Err(Error::Fem(format!(
                    "coarsen_cell: child cell {ci} of parent {parent_idx} \
                     is already marked removed — mesh state is internally inconsistent"
                )));
            }
        }

        // Tentatively apply the coarsening, then validate post-conditions
        // (2:1 balance, hanging-node midpoint orphaning). On any
        // violation we roll back.
        for &c in record.children.as_slice() {
            self.is_removed[c as usize] = true;
        }
        self.is_refined[parent_idx] = false;
        self.refined_cells.remove(rec_idx);

        // Rebuild the midpoint cache from the *remaining* refined cells
        // so `edge_split_depth` reflects the post-coarsening state.
        self.rebuild_midpoint_cache();

        // 2:1 balance check: every remaining leaf edge must have split
        // depth ≤ 1. If the coarsening exposed a deeper neighbour
        // (e.g. parent_idx is now a level-L leaf adjacent to a
        // level-L+2 sub-cell), roll back.
        let parent_cell = self.cells[parent_idx];
        for (a, b) in leaf_edges(&parent_cell) {
            if self.edge_split_depth(a, b) > 1 {
                // Roll back.
                for &c in record.children.as_slice() {
                    self.is_removed[c as usize] = false;
                }
                self.is_refined[parent_idx] = true;
                self.refined_cells.insert(rec_idx, record);
                self.rebuild_midpoint_cache();
                return Err(Error::Fem(format!(
                    "coarsen_cell: would violate 2:1 balance on edge ({a}, {b}) of \
                     parent {parent_idx} — split depth would be > 1"
                )));
            }
        }

        Ok(())
    }

    /// Stage 45 v5 — recursively coarsen an entire refined sub-tree.
    ///
    /// Unlike [`Self::coarsen_cell`], this helper first walks to the
    /// deepest descendants and coarsens them before attempting the
    /// requested parent. This is the ergonomic path for time-dependent
    /// AMR regions that have refined more than one level and then need
    /// to un-refine as the feature leaves the region.
    pub fn coarsen_subtree(&mut self, parent_idx: usize) -> Result<()> {
        if parent_idx >= self.cells.len() {
            return Err(Error::Fem(format!(
                "coarsen_subtree: parent index {parent_idx} out of bounds (cell count {})",
                self.cells.len()
            )));
        }
        if !self.is_refined[parent_idx] {
            return self.coarsen_cell(parent_idx);
        }
        let record = self
            .refined_cells
            .iter()
            .find(|r| r.parent as usize == parent_idx)
            .copied()
            .ok_or_else(|| {
                Error::Fem(format!(
                    "coarsen_subtree: no RefinedCell record for parent {parent_idx}"
                ))
            })?;
        for &child in record.children.as_slice() {
            let ci = child as usize;
            if self.is_refined[ci] && !self.is_removed[ci] {
                self.coarsen_subtree(ci)?;
            }
        }
        self.coarsen_cell(parent_idx)
    }

    /// Stage 45 v4 — batched coarsening. Coarsen every refined cell
    /// **all of whose children** satisfy `predicate`. Cells whose
    /// children fail the predicate (or whose coarsening would violate
    /// any precondition of [`Self::coarsen_cell`]) are silently skipped.
    ///
    /// After the batch is applied, [`Self::propagate_coarsening`] runs
    /// to restore 2:1 balance (refining cells whose coarsening would
    /// have created a level-2 gap), and the hanging-node cache is
    /// refreshed.
    ///
    /// Returns the number of cells actually coarsened.
    pub fn coarsen_where<F>(&mut self, predicate: F) -> usize
    where
        F: Fn(&FemCell) -> bool,
    {
        // Snapshot the refined-cell list — we mutate it in the loop.
        let candidates: Vec<u32> = self
            .refined_cells
            .iter()
            .filter(|rec| {
                rec.children
                    .as_slice()
                    .iter()
                    .all(|&c| predicate(&self.cells[c as usize]))
            })
            .map(|rec| rec.parent)
            .collect();

        let mut coarsened = 0usize;
        for parent in candidates {
            if self.coarsen_cell(parent as usize).is_ok() {
                coarsened += 1;
            }
        }
        self.propagate_coarsening();
        self.compute_hanging_nodes();
        coarsened
    }

    /// Stage 45 v4 — counterpart of [`Self::propagate_refinement`] for
    /// coarsening. Walks the current leaf set; if any leaf has an edge
    /// of split depth `> 1` (i.e. the coarsening exposed a finer
    /// neighbour two levels deeper), refine that leaf once to restore
    /// 2:1 balance. Iterates until balance is restored.
    ///
    /// This *cannot loop forever*: every iteration strictly refines at
    /// least one offending leaf, increasing the global cell count, so
    /// the bounded number of cells in the mesh terminates the loop.
    pub fn propagate_coarsening(&mut self) {
        // The 2:1-restoration semantics is identical to
        // `propagate_refinement` (refine any leaf adjacent to a
        // depth-2 edge), so we can simply call into it.
        self.propagate_refinement();
    }

    /// Rebuild the midpoint cache from the currently-active refinement
    /// records. Useful after coarsening, which can render some cached
    /// midpoints stale (no longer referenced by any refined parent).
    ///
    /// The cache is populated by walking each [`RefinedCell`] record's
    /// parent edges (the parent's own `FemCell::verts`) and looking up
    /// the midpoint as the vertex that appears in two non-corner
    /// positions in the children's vertex lists.
    fn rebuild_midpoint_cache(&mut self) {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        // For each active refined cell, the children carry midpoint
        // vertex ids. The simplest reconstruction: walk the children
        // and, for every pair of vertices in a child cell, check
        // whether the child's vertex coincides with the geometric
        // midpoint of a parent edge. The parent edges are the
        // unordered vertex pairs from `cells[parent].verts()`.
        for rec in &self.refined_cells {
            let parent = self.cells[rec.parent as usize];
            let parent_verts = parent.verts();
            // Collect unique vertex ids from the children — these are
            // the candidate midpoints.
            let mut child_verts: HashSet<u32> = HashSet::new();
            for &c in rec.children.as_slice() {
                for &v in self.cells[c as usize].verts() {
                    child_verts.insert(v);
                }
            }
            // For each pair of parent corners, find the midpoint vertex
            // by geometric coincidence.
            let n = parent_verts.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    // Tri parent: every (i, j) pair is an edge.
                    // Tet parent: same — all 6 (i, j) pairs with i < j
                    // are edges of the tet.
                    let a = parent_verts[i];
                    let b = parent_verts[j];
                    let pa = self.vertices[a as usize];
                    let pb = self.vertices[b as usize];
                    let mid = (pa + pb) * 0.5;
                    // Find the child vertex equal to `mid` (allowing a
                    // tiny floating tolerance).
                    for &cv in &child_verts {
                        // Skip the parent corners themselves.
                        if cv == a || cv == b {
                            continue;
                        }
                        let pv = self.vertices[cv as usize];
                        if (pv - mid).length() < 1.0e-12 {
                            let key = if a < b { (a, b) } else { (b, a) };
                            cache.insert(key, cv);
                            break;
                        }
                    }
                }
            }
        }
        self.midpoint_cache = cache;
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
            if self.is_refined[idx] || self.is_removed[idx] {
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
            if self.is_refined[idx] || self.is_removed[idx] {
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
                if self.is_refined[idx] || self.is_removed[idx] {
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
                if !self.is_refined[idx] && !self.is_removed[idx] {
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
            Some(&m) => 1 + self.edge_split_depth(a, m).max(self.edge_split_depth(m, b)),
        }
    }

    /// Build the hanging-node constraint matrix `T` mapping
    /// `u_free → u_total` for the **current flat-mesh DoF set**.
    ///
    /// `T` has shape `(n_total × n_free)` where `n_total` is the
    /// flat-mesh DoF count and `n_free` is the count of non-hanging
    /// DoFs. The row for a non-hanging DoF is a standard identity
    /// indicator; the row for a hanging DoF is the per-degree
    /// interpolation:
    ///
    /// - **P1** (`degree = 1`): hanging vertex `m` constrained as
    ///   `u_m = ½(u_a + u_b)` (linear average of edge endpoints).
    ///
    /// - **P2** (`degree = 2`): three hanging DoF families per parent
    ///   edge that has been split. With `(u_a, u_b, u_M)` the parent
    ///   edge's three P2 DoFs (corners `a`, `b` and the parent edge
    ///   mid-DoF `M`), and the parent trace
    ///   `u(ξ) = u_a (1-ξ)(1-2ξ) + u_b ξ(2ξ-1) + 4 u_M ξ(1-ξ)`:
    ///   - **Vertex midpoint** at `ξ = 1/2` → `u_m = u_M`.
    ///   - **Sub-edge** `(a, m)` mid-DoF at `ξ = 1/4` →
    ///     `(3/8) u_a + (-1/8) u_b + (3/4) u_M`.
    ///   - **Sub-edge** `(m, b)` mid-DoF at `ξ = 3/4` →
    ///     `(-1/8) u_a + (3/8) u_b + (3/4) u_M`.
    ///
    ///   The full P2 DoF set on the flat mesh is `n_vert + n_edge`
    ///   (vertex DoFs first, edge mid-DoFs second — matching the
    ///   [`crate::fem::p2_element::P2DofMap`] layout). The constraint
    ///   matrix is sized accordingly.
    ///
    /// `degree ∈ {1, 2}`. P3+ is not supported.
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
        assert!(
            degree == 1 || degree == 2,
            "build_constraint_matrix: degree must be 1 or 2 (got {degree})"
        );
        match degree {
            1 => self.build_constraint_matrix_p1(remap),
            2 => self.build_constraint_matrix_p2(remap),
            _ => unreachable!(),
        }
    }

    /// P1 implementation — one DoF per vertex, hanging-vertex constraint
    /// is the linear average of its two edge endpoints.
    fn build_constraint_matrix_p1(&self, remap: &HashMap<u32, u32>) -> Result<BsrMatrix> {
        let n_total = remap.len();

        // Hanging DoF set in flat-index space — only those hanging
        // midpoints whose AMR id was actually used by a leaf (i.e.
        // entered the flatten remap) get a constraint row. Hanging
        // midpoints belong to leaf edges by construction, so they all
        // map; the filter guards against pathological hand-built test
        // cases.
        let mut is_hanging_flat: Vec<bool> = vec![false; n_total];
        let mut hanging_parents_flat: Vec<Option<(u32, u32)>> = vec![None; n_total];
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
        let mut triplets: Vec<(usize, usize, Real)> =
            Vec::with_capacity(n_total + 2 * self.hanging_nodes.len());
        for i in 0..n_total {
            if is_hanging_flat[i] {
                let (p1, p2) =
                    hanging_parents_flat[i].expect("hanging DoF marked but parents missing");
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

    /// P2 implementation — vertex DoFs + edge-midpoint DoFs of the flat
    /// mesh, in the same layout as [`crate::fem::p2_element::P2DofMap`]
    /// (vertices first, edges second). Hanging-DoF families:
    ///
    /// - Hanging vertex midpoint `m` of parent edge `(a, b)` →
    ///   `u_m = u_M` where `M` is the parent edge's P2 mid-edge DoF.
    /// - Sub-edge mid-DoFs on `(a, m)` and `(m, b)` → curvature-aware
    ///   constraint reproducing the parent's P2 polynomial at
    ///   `ξ = 1/4` and `ξ = 3/4` respectively.
    fn build_constraint_matrix_p2(&self, remap: &HashMap<u32, u32>) -> Result<BsrMatrix> {
        use super::function_space::compute_edges;

        // We need the flat mesh's vertex + edge enumeration to address
        // P2 edge-midpoint DoFs. The caller's `remap` already supplies
        // AMR vid → flat vid; rebuilding the flat mesh is cheap (no
        // per-cell heavy work) and lets us reuse `compute_edges`.
        let (flat, _remap_check) = self.flatten_with_remap();
        debug_assert_eq!(remap.len(), flat.vert_count());

        let n_vert = flat.vert_count();
        let (edges, edge_id) = compute_edges(&flat);
        let n_edge = edges.len();
        let n_total = n_vert + n_edge;

        // Each hanging-node record gives us a parent edge `(a, b)` (in
        // flat-vertex ids, after remapping) and a hanging vertex `m`.
        // We need the parent edge DoF `M`, the sub-edge DoFs `Q1` (on
        // `(a, m)`) and `Q2` (on `(m, b)`), in flat-mesh DoF coordinates.
        //
        // Layout: P2 DoFs are `[vertex_0..vertex_{n_vert-1},
        //                       edge_0..edge_{n_edge-1}]`. So edge `e`
        // (with flat-edge id `e_id`) lives at DoF `n_vert + e_id`.
        //
        // For each hanging record we set up to 3 constraint rows:
        //   row vid(m)            ← [(vid(M_dof_idx), 1.0)]
        //   row dof(Q1)           ← [(vid(a),  3/8), (vid(b), -1/8), (vid(M_dof_idx), 3/4)]
        //   row dof(Q2)           ← [(vid(a), -1/8), (vid(b),  3/8), (vid(M_dof_idx), 3/4)]
        //
        // `vid(M_dof_idx) = n_vert + edge_id[(a, b)]`.
        let mut is_hanging: Vec<bool> = vec![false; n_total];
        // `constraint[i]` = Some(Vec<(dof_index, coefficient)>) for
        // hanging DoF i. We build it as a flat list to keep allocation
        // simple.
        let mut constraint_rows: Vec<Option<Vec<(usize, Real)>>> = vec![None; n_total];

        let edge_key = |a: u32, b: u32| -> [u32; 2] { if a < b { [a, b] } else { [b, a] } };

        for hn in &self.hanging_nodes {
            let Some(&flat_m) = remap.get(&hn.dof) else {
                continue;
            };
            let Some(&flat_a) = remap.get(&hn.parents[0]) else {
                continue;
            };
            let Some(&flat_b) = remap.get(&hn.parents[1]) else {
                continue;
            };

            // Parent edge (a, b) in flat-mesh edge enumeration.
            let Some(&pe_id) = edge_id.get(&edge_key(flat_a, flat_b)) else {
                // The parent edge has to live in the flat mesh — the
                // unrefined leaf on the other side uses it. If we ever
                // hit this (e.g. corner refinement on the boundary
                // where both sides happen to refine), the hanging
                // record is spurious and there's nothing to constrain.
                continue;
            };
            let m_parent_dof = n_vert + pe_id as usize;

            // (1) Vertex midpoint constraint: u_m = u_M.
            let m_dof = flat_m as usize;
            is_hanging[m_dof] = true;
            constraint_rows[m_dof] = Some(vec![(m_parent_dof, 1.0)]);

            // (2) Sub-edge (a, m) P2 mid-edge DoF — constrained iff
            // present in the flat mesh's edge set. It will be present
            // if the refined-side leaves include this sub-edge, which
            // is the standard case under 2:1 balance.
            if let Some(&se1_id) = edge_id.get(&edge_key(flat_a, flat_m)) {
                let q1_dof = n_vert + se1_id as usize;
                is_hanging[q1_dof] = true;
                constraint_rows[q1_dof] = Some(vec![
                    (flat_a as usize, 3.0 / 8.0),
                    (flat_b as usize, -1.0 / 8.0),
                    (m_parent_dof, 3.0 / 4.0),
                ]);
            }
            if let Some(&se2_id) = edge_id.get(&edge_key(flat_m, flat_b)) {
                let q2_dof = n_vert + se2_id as usize;
                is_hanging[q2_dof] = true;
                constraint_rows[q2_dof] = Some(vec![
                    (flat_a as usize, -1.0 / 8.0),
                    (flat_b as usize, 3.0 / 8.0),
                    (m_parent_dof, 3.0 / 4.0),
                ]);
            }
        }

        // Free DoFs are the non-hanging ones. Numbered in DoF order so
        // the pattern stays deterministic.
        let mut free_index: Vec<i32> = vec![-1; n_total];
        let mut n_free = 0usize;
        for i in 0..n_total {
            if !is_hanging[i] {
                free_index[i] = n_free as i32;
                n_free += 1;
            }
        }

        // Emit triplets — identity for free DoFs, the interpolation
        // coefficients for hanging rows. 2:1 balance is enforced by
        // `refine_where`, so every DoF a hanging row depends on must be
        // free; we check that here as a debug assertion.
        let mut triplets: Vec<(usize, usize, Real)> =
            Vec::with_capacity(n_total + 3 * self.hanging_nodes.len() * 3);
        for i in 0..n_total {
            if is_hanging[i] {
                let row = constraint_rows[i]
                    .as_ref()
                    .expect("hanging DoF without constraint row");
                for &(dof_j, c) in row {
                    debug_assert!(
                        !is_hanging[dof_j],
                        "P2 hanging-DoF dependency {dof_j} is itself hanging — \
                         2:1 balance violated"
                    );
                    let col = free_index[dof_j] as usize;
                    triplets.push((i, col, c));
                }
            } else {
                let col = free_index[i] as usize;
                triplets.push((i, col, 1.0));
            }
        }

        BsrMatrix::from_scalar_triplets(n_total, n_free, &triplets)
    }

    /// Stage 45 v3 — interpolate a P1 nodal field from `u_old` (defined
    /// on the *previous* flat-mesh DoF set) onto the *current* flat
    /// mesh's DoF set.
    ///
    /// `old_amr` is the AMR mesh that produced `u_old`; `self` is the
    /// new (refined or coarsened) AMR mesh. The interpolant uses the
    /// P1 nodal shape functions of `old_amr`'s leaf containing each new
    /// DoF location.
    ///
    /// For a linear field `u(x) = α + β · x` the interpolation is
    /// **exact** (P1 reproduces affine functions). For higher-order
    /// fields the operator is `O(h²)` accurate per the standard
    /// Lagrange-interpolation estimate.
    ///
    /// Returns a `Vec<Real>` of length `self.flatten().vert_count()`
    /// (one entry per flat-mesh vertex DoF).
    ///
    /// # Panics
    /// Panics if `u_old.len()` does not match `old_amr`'s flat vertex
    /// count, or if a new DoF location falls outside the geometric
    /// support of every old leaf (indicates a topology mismatch — the
    /// two AMR meshes must share the same `base_mesh`).
    #[must_use]
    pub fn transfer_state(&self, u_old: &[Real], old_amr: &AmrMesh) -> Vec<Real> {
        let (old_flat, _) = old_amr.flatten_with_remap();
        assert_eq!(
            u_old.len(),
            old_flat.vert_count(),
            "transfer_state: u_old length ({}) != old flat-mesh DoF count ({})",
            u_old.len(),
            old_flat.vert_count()
        );

        let (new_flat, _) = self.flatten_with_remap();
        let mut u_new = vec![0.0_f64 as Real; new_flat.vert_count()];

        // Build a leaf index — list of (cell index, vertex positions)
        // we can iterate to locate each new vertex. For Stage 45 mesh
        // sizes the linear search is acceptable; a spatial-hash index
        // is the obvious follow-up if profiling demands it.
        for (i, p) in new_flat.vertices.iter().enumerate() {
            u_new[i] = evaluate_p1_field_on_mesh(&old_flat, u_old, *p);
        }

        u_new
    }

    /// Stage 45 v3 — for DG-AMR, return the list of
    /// **parent → child** face mappings induced by the refinement
    /// tree.
    ///
    /// Each `DgAmrFaceMatch` records one **parent face** (the face of
    /// a coarse leaf on the unrefined side of a T-junction) together
    /// with the **sub-faces** that the refined side contributes —
    /// 2 sub-edges per parent edge in 2D, 4 sub-triangles per parent
    /// triangular face in 3D (corresponding to a 1→4 face-split that
    /// goes along with a 1→8 tet split).
    ///
    /// The list is computed from the AMR tree (no FEM data needed)
    /// and is used by DG-AMR flux assembly to balance the parent's
    /// flux integral against the child sum so the DG semi-discrete
    /// equation stays conservative across refinement levels.
    #[must_use]
    pub fn compute_dg_amr_faces(&self) -> Vec<DgAmrFaceMatch> {
        let mut matches = Vec::new();

        // Walk leaves; for each leaf edge, look up the midpoint cache.
        // If the edge has been split exactly once (depth 1 — the 2:1
        // balance invariant), it is a coarse "parent" face whose two
        // child sub-edges live on the refined side.
        //
        // For tets, a face is a triangle (a, b, c); we look up the
        // three edge midpoints (a,b), (b,c), (c,a). If all three are
        // present, the face is split 1→4 (4 sub-triangles meeting at
        // the three midpoints + the central inverted tri).
        for (idx, cell) in self.cells.iter().enumerate() {
            if self.is_refined[idx] || self.is_removed[idx] {
                continue;
            }
            match cell {
                FemCell::Tri(v) => {
                    for k in 0..3 {
                        let a = v[k];
                        let b = v[(k + 1) % 3];
                        let key = if a < b { (a, b) } else { (b, a) };
                        if let Some(&m) = self.midpoint_cache.get(&key) {
                            matches.push(DgAmrFaceMatch {
                                parent_cell: idx as u32,
                                parent_face: vec![a, b],
                                child_faces: vec![vec![a, m], vec![m, b]],
                            });
                        }
                    }
                }
                FemCell::Tet(v) => {
                    // 4 tri faces of the tet — opposite each vertex.
                    let faces: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
                    for f in &faces {
                        let a = v[f[0]];
                        let b = v[f[1]];
                        let c = v[f[2]];
                        let key_ab = if a < b { (a, b) } else { (b, a) };
                        let key_bc = if b < c { (b, c) } else { (c, b) };
                        let key_ca = if c < a { (c, a) } else { (a, c) };
                        let m_ab = self.midpoint_cache.get(&key_ab).copied();
                        let m_bc = self.midpoint_cache.get(&key_bc).copied();
                        let m_ca = self.midpoint_cache.get(&key_ca).copied();
                        if let (Some(m_ab), Some(m_bc), Some(m_ca)) = (m_ab, m_bc, m_ca) {
                            // 1→4 sub-triangles — corner tris around
                            // each parent vertex + the central
                            // inverted tri.
                            matches.push(DgAmrFaceMatch {
                                parent_cell: idx as u32,
                                parent_face: vec![a, b, c],
                                child_faces: vec![
                                    vec![a, m_ab, m_ca],
                                    vec![m_ab, b, m_bc],
                                    vec![m_ca, m_bc, c],
                                    vec![m_ab, m_bc, m_ca],
                                ],
                            });
                        }
                    }
                }
            }
        }

        matches
    }
}

/// Evaluate a P1 nodal field `u` on `mesh` at physical point `x`.
///
/// Walks every cell, computes the barycentric coordinates of `x` inside
/// it, and returns the linear combination `Σ λᵢ uᵢ` for the first cell
/// whose barycentrics are all ≥ `-tol`. The tolerance handles the
/// "shared edge" case where `x` is exactly on a cell boundary: the
/// first containing cell wins (deterministic by iteration order).
///
/// # Panics
/// Panics if `x` is outside every cell in `mesh` (beyond `tol`).
fn evaluate_p1_field_on_mesh(mesh: &FemMesh, u: &[Real], x: Vec3) -> Real {
    use crate::math::Mat3;

    let tol: Real = 1.0e-9;
    for cell in &mesh.cells {
        match cell {
            FemCell::Tri(idx) => {
                let p0 = mesh.vertices[idx[0] as usize];
                let p1 = mesh.vertices[idx[1] as usize];
                let p2 = mesh.vertices[idx[2] as usize];
                // 2D barycentric (z assumed to be planar / matching).
                let det = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
                if det.abs() < 1.0e-30 {
                    continue;
                }
                let inv = 1.0 / det;
                let l1 = ((p2.y - p0.y) * (x.x - p0.x) + (p0.x - p2.x) * (x.y - p0.y)) * inv;
                let l2 = ((p0.y - p1.y) * (x.x - p0.x) + (p1.x - p0.x) * (x.y - p0.y)) * inv;
                let l0 = 1.0 - l1 - l2;
                if l0 >= -tol && l1 >= -tol && l2 >= -tol {
                    return l0 * u[idx[0] as usize]
                        + l1 * u[idx[1] as usize]
                        + l2 * u[idx[2] as usize];
                }
            }
            FemCell::Tet(idx) => {
                let p0 = mesh.vertices[idx[0] as usize];
                let p1 = mesh.vertices[idx[1] as usize];
                let p2 = mesh.vertices[idx[2] as usize];
                let p3 = mesh.vertices[idx[3] as usize];
                let d = Mat3::from_cols(p1 - p0, p2 - p0, p3 - p0);
                let det = d.determinant();
                if det.abs() < 1.0e-30 {
                    continue;
                }
                let inv = d.inverse();
                let r = x - p0;
                let l1 = inv.x_axis.x * r.x + inv.y_axis.x * r.y + inv.z_axis.x * r.z;
                let l2 = inv.x_axis.y * r.x + inv.y_axis.y * r.y + inv.z_axis.y * r.z;
                let l3 = inv.x_axis.z * r.x + inv.y_axis.z * r.y + inv.z_axis.z * r.z;
                let l0 = 1.0 - l1 - l2 - l3;
                if l0 >= -tol && l1 >= -tol && l2 >= -tol && l3 >= -tol {
                    return l0 * u[idx[0] as usize]
                        + l1 * u[idx[1] as usize]
                        + l2 * u[idx[2] as usize]
                        + l3 * u[idx[3] as usize];
                }
            }
        }
    }
    panic!(
        "evaluate_p1_field_on_mesh: point ({}, {}, {}) lies outside every cell",
        x.x, x.y, x.z
    );
}

/// One **parent face → child sub-faces** record for DG-AMR. The
/// constraint imposed at a 2:1-balanced face split is that the DG
/// numerical-flux integral over `parent_face` (computed from the
/// coarse side's basis) equals the sum of flux integrals over
/// `child_faces` (each computed from the refined side's basis).
///
/// For triangles (2D) the parent face is an edge `[a, b]` and the
/// children are `[a, m]` and `[m, b]`. For tetrahedra (3D) the parent
/// face is a triangle `[a, b, c]` and the children are the four sub-
/// triangles produced by inserting the three edge midpoints (the
/// "red" 1→4 split of a triangle).
#[derive(Clone, Debug)]
pub struct DgAmrFaceMatch {
    /// AMR cell index of the coarse leaf that owns the parent face.
    pub parent_cell: u32,
    /// AMR vertex ids of the parent face (length 2 for tri-mesh
    /// edges, length 3 for tet-mesh triangular faces).
    pub parent_face: Vec<u32>,
    /// AMR vertex ids of the child sub-faces (each entry has the
    /// same length as `parent_face`).
    pub child_faces: Vec<Vec<u32>>,
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
///
/// Useful when the caller has assembled `K` separately (e.g. for a P2
/// problem) and wants to reduce it against the constraint matrix `T`
/// built via [`AmrMesh::build_constraint_matrix`].
pub fn compute_ttkt(t: &BsrMatrix, k: &BsrMatrix) -> Result<BsrMatrix> {
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

    /// Stage 45 v4 — **Krylov restart after re-mesh**. Solve the linear
    /// system `K_new · x = f_new` using a **warm-start** initial guess
    /// derived from the previous mesh's solution interpolated onto the
    /// new mesh (typically the output of [`Self::transfer_state`]).
    ///
    /// The solver is plain Conjugate Gradient (`cg_solve`); on SPD
    /// systems (e.g. Poisson with Dirichlet BCs already baked into
    /// `K_new` and `f_new`) CG with a warm start converges in fewer
    /// iterations than CG started from zero, because the initial
    /// residual `‖f - K · x_warm‖` is already small.
    ///
    /// Inputs:
    /// - `k_new`: the assembled stiffness matrix for the new mesh
    ///   (post-BC, ready to solve).
    /// - `f_new`: the right-hand side of the same system.
    /// - `x_old_on_new_mesh`: the previous solution evaluated on the
    ///   new mesh's DoFs — typically `self.transfer_state(...)` after
    ///   the re-mesh. Length must equal `k_new.nrows()`.
    /// - `max_iters`, `tol`: forwarded to `cg_solve`.
    ///
    /// Returns `(x_new, iters)` — the new solution and the number of
    /// CG iterations consumed. Callers comparing warm-start vs
    /// cold-start should run `solve_after_remesh` once with the
    /// transferred state and once with `vec![0.0; n]` and compare iter
    /// counts directly.
    ///
    /// This helper deliberately does **not** call `transfer_state`
    /// internally — the caller has typically already done so to apply
    /// any post-interpolation steps (e.g. re-applying Dirichlet pins on
    /// the new mesh), and we don't want to duplicate that work.
    pub fn solve_after_remesh(
        k_new: &BsrMatrix,
        f_new: &[Real],
        x_old_on_new_mesh: &[Real],
        max_iters: usize,
        tol: Real,
    ) -> (Vec<Real>, usize) {
        assert_eq!(
            k_new.nrows(),
            f_new.len(),
            "solve_after_remesh: k_new has {} rows but f_new has {} entries",
            k_new.nrows(),
            f_new.len()
        );
        assert_eq!(
            k_new.nrows(),
            x_old_on_new_mesh.len(),
            "solve_after_remesh: k_new has {} rows but x_old has {} entries",
            k_new.nrows(),
            x_old_on_new_mesh.len()
        );
        let result: CgResult = cg_solve(
            |x: &[Real]| {
                let mut y = vec![0.0; k_new.nrows()];
                k_new.mv(x, &mut y).expect("k_new matvec");
                y
            },
            f_new,
            x_old_on_new_mesh,
            tol,
            max_iters,
        );
        (result.x, result.iters)
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
            self.is_removed.push(false);
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
            self.is_removed.push(false);
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

/// Stage 45 v5 — L²-project DG-P1 child values back onto a triangle
/// parent during coarsening.
///
/// `child_values[k][i]` is the DG value at local P1 node `i` of
/// `children[k]`. The returned `[u0, u1, u2]` are the parent DG-P1 nodal
/// coefficients that satisfy
///
/// ```text
/// ∫_parent N_i · u_parent dx = Σ_child ∫_child N_i · u_child dx
/// ```
///
/// for each parent basis function `N_i`. The three-point triangle rule
/// is degree-2 exact, so this projection is exact for every piecewise
/// linear child field.
pub fn project_dg_p1_tri_children_to_parent_l2(
    parent: &FemCell,
    children: &[FemCell],
    vertices: &[Vec3],
    child_values: &[[Real; 3]],
) -> Result<[Real; 3]> {
    let FemCell::Tri(pidx) = parent else {
        return Err(Error::Fem(
            "project_dg_p1_tri_children_to_parent_l2: parent must be a triangle".into(),
        ));
    };
    if children.len() != child_values.len() {
        return Err(Error::Fem(format!(
            "project_dg_p1_tri_children_to_parent_l2: {} child cells but {} value blocks",
            children.len(),
            child_values.len()
        )));
    }
    let p = [
        vertices[pidx[0] as usize],
        vertices[pidx[1] as usize],
        vertices[pidx[2] as usize],
    ];
    let area_parent = tri_area(p[0], p[1], p[2]).abs();
    if area_parent <= 1.0e-30 {
        return Err(Error::Fem(
            "project_dg_p1_tri_children_to_parent_l2: degenerate parent triangle".into(),
        ));
    }

    let mut rhs = [0.0_f64 as Real; 3];
    let qrule = tri_quadrature_3pt();
    for (child, values) in children.iter().zip(child_values.iter()) {
        let FemCell::Tri(cidx) = child else {
            return Err(Error::Fem(
                "project_dg_p1_tri_children_to_parent_l2: child must be a triangle".into(),
            ));
        };
        let c = [
            vertices[cidx[0] as usize],
            vertices[cidx[1] as usize],
            vertices[cidx[2] as usize],
        ];
        let area_child = tri_area(c[0], c[1], c[2]).abs();
        for q in &qrule {
            let x = c[0] * q.bary[0] + c[1] * q.bary[1] + c[2] * q.bary[2];
            let Some(parent_bary) = barycentric_tri_2d(p, x) else {
                return Err(Error::Fem(
                    "project_dg_p1_tri_children_to_parent_l2: quadrature point outside parent"
                        .into(),
                ));
            };
            let u_child = values[0] * q.bary[0] + values[1] * q.bary[1] + values[2] * q.bary[2];
            for i in 0..3 {
                rhs[i] += area_child * q.weight * parent_bary[i] * u_child;
            }
        }
    }

    // Parent P1 mass matrix on a triangle:
    // M = area/12 * [[2,1,1],[1,2,1],[1,1,2]].
    let mass = area_parent / 12.0;
    solve_3x3(
        [
            [2.0 * mass, mass, mass],
            [mass, 2.0 * mass, mass],
            [mass, mass, 2.0 * mass],
        ],
        rhs,
    )
    .ok_or_else(|| {
        Error::Fem("project_dg_p1_tri_children_to_parent_l2: singular parent mass".into())
    })
}

fn tri_area(a: Vec3, b: Vec3, c: Vec3) -> Real {
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))
}

fn barycentric_tri_2d(p: [Vec3; 3], x: Vec3) -> Option<[Real; 3]> {
    let det = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
    if det.abs() < 1.0e-30 {
        return None;
    }
    let inv = 1.0 / det;
    let l1 = ((p[2].y - p[0].y) * (x.x - p[0].x) + (p[0].x - p[2].x) * (x.y - p[0].y)) * inv;
    let l2 = ((p[0].y - p[1].y) * (x.x - p[0].x) + (p[1].x - p[0].x) * (x.y - p[0].y)) * inv;
    let l0 = 1.0 - l1 - l2;
    Some([l0, l1, l2])
}

fn solve_3x3(mut a: [[Real; 3]; 3], mut b: [Real; 3]) -> Option<[Real; 3]> {
    for k in 0..3 {
        let mut pivot = k;
        let mut best = a[k][k].abs();
        for (r, row) in a.iter().enumerate().skip(k + 1) {
            let v = row[k].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1.0e-30 {
            return None;
        }
        if pivot != k {
            a.swap(k, pivot);
            b.swap(k, pivot);
        }
        let inv = 1.0 / a[k][k];
        for j in k..3 {
            a[k][j] *= inv;
        }
        b[k] *= inv;
        for r in 0..3 {
            if r == k {
                continue;
            }
            let f = a[r][k];
            if f == 0.0 {
                continue;
            }
            for j in k..3 {
                a[r][j] -= f * a[k][j];
            }
            b[r] -= f * b[k];
        }
    }
    Some(b)
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
