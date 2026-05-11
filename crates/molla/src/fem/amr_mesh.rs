//! Stage 45 v1 — adaptive-mesh-refinement foundation for FEM.
//!
//! v1 ships the **data-structure foundation** and **uniform refinement**
//! only. The hanging-node DoF constraints required for true *non-uniform*
//! refinement (the T-junction case where a refined cell shares an edge
//! with a coarser unrefined neighbour) are intentionally deferred — see
//! the module-level "v2 outlook" comment below.
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
//! # v2 outlook — hanging-node constraints
//!
//! Non-uniform refinement produces *hanging nodes*: a midpoint vertex
//! `m` that lives on the shared edge of (a refined cell) and (an
//! unrefined neighbour). The unrefined neighbour does not see `m` as
//! one of its DoFs, so naive assembly leaves the linear system
//! inconsistent across the hanging edge. The standard fix is to
//! **constrain** the DoF at `m` to be the average of its two parent
//! vertices (for P1) or a higher-order combination (for P2+). v1
//! refuses to address this — the only AMR pattern supported is uniform
//! refinement (every leaf refined to the same depth) where every edge
//! is shared between cells of the same level and no hanging node can
//! appear. Stage 45 v2 will land the constraint matrix `T` and the
//! reduced solve `Kᶜ = TᵀKT, fᶜ = Tᵀf`.

use std::collections::HashMap;

use crate::math::{Real, Vec3};

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
    /// the new children become the leaves.
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
    }

    /// Refine every leaf cell that `criterion` accepts. Convenience
    /// closure-friendly entry point — useful for tests that drive
    /// refinement off a spatial query (`near_point(p, r)` etc.).
    ///
    /// **Caveat (v1)** — driving non-uniform refinement via a
    /// criterion *will* produce hanging nodes if neighbouring cells
    /// disagree about whether to refine. The flatten + assembly path
    /// is **only** valid for the uniform-refinement subset of the
    /// resulting tree. v1 ships this entry so the v2 hanging-node
    /// work has a working harness to plug into.
    pub fn refine_where<C: RefinementCriterion>(&mut self, criterion: C) {
        let to_refine: Vec<usize> = (0..self.cells.len())
            .filter(|&i| !self.is_refined[i])
            .filter(|&i| criterion.should_refine(&self.cells[i], &self.vertices))
            .collect();
        for idx in to_refine {
            self.refine_cell(idx);
        }
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

        FemMesh { vertices, cells }
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
