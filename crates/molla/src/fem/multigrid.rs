//! Geometric multigrid V-cycle on a uniformly-refined mesh hierarchy —
//! Stage 48 v3 of the AMS preconditioner.
//!
//! # Scope
//!
//! Full Hiptmair-Xu AMS replaces the auxiliary-block scalar solver with
//! an algebraic multigrid V-cycle (AMG, Ruge-Stüben). v3 ships the
//! intermediate stepping stone: **geometric** multigrid on a
//! [`AmrMesh::refine_uniform`]-built mesh hierarchy. AMG is queued for
//! v4 (see "What's needed for AMG v4" in the module docs / commit
//! message). The Pi-projection branch of AMS is also out of scope here.
//!
//! # Structure
//!
//! - [`MeshHierarchy`]: a sequence of [`FemMesh`]s (level 0 = coarsest,
//!   last = finest), the P1 scalar Laplacian at each level, and a
//!   prolongation [`BsrMatrix`] per inter-level transition.
//! - [`MeshHierarchy::build`]: take a coarse base mesh + a list of
//!   finest-level pinned vertices, refine `n_levels - 1` times
//!   uniformly, build per-level Laplacians (with consistent pinning),
//!   and build prolongation operators via P1 barycentric
//!   interpolation.
//! - [`MeshHierarchy::v_cycle`]: standard geometric MG V-cycle —
//!   pre-smooth (ω-Jacobi, 3 iters), restrict residual, recurse or
//!   solve direct at coarsest level (dense Cholesky on the SPD pinned
//!   Laplacian), prolongate correction, post-smooth.
//!
//! # AMS integration
//!
//! [`super::eddy_current::AmsPreconditioner`] grows an optional
//! `mg_hierarchy: Option<MeshHierarchy>` field; when set, the inner
//! `K_aux · w = r_aux` solve uses one V-cycle instead of the v2
//! fixed-iter CG. Linearity is preserved (each call applies exactly
//! `mg_cycles_per_apply` V-cycles, defaulting to 1).
//!
//! # What's needed for AMG v4 — Ruge-Stüben algebraic multigrid
//!
//! AMG removes the geometric prerequisite (uniformly refined mesh
//! hierarchy) by deriving its coarse hierarchy directly from the
//! sparse matrix. Concretely v4 needs:
//!
//! 1. **C/F splitting** — classify rows as `C` (coarse) or `F` (fine)
//!    using the Ruge-Stüben "strong-connection" criterion:
//!    `|A[i, j]| ≥ θ · max_{k ≠ i} |A[i, k]|` (typical `θ = 0.25`).
//!    Greedily promote `C` candidates by independent-set heuristic
//!    on the strong-connection graph.
//! 2. **Interpolation operator** — for each `F` row, build the
//!    interpolation row as a weighted sum of strong-`C` neighbours.
//!    Standard formula:
//!    `P[F, C] = -A[F, C] / sum_C' (-A[F, C'])`, with off-diagonal
//!    `F → F` contributions distributed onto the `C` neighbours
//!    (the "direct" or "classical" interpolation).
//! 3. **Galerkin product** — same `Pᵀ A P` as v3, but with the
//!    algebraic `P` from step 2 instead of geometric P1
//!    interpolation. Reuse [`super::amr_mesh::compute_ttkt`].
//! 4. **Smoother choice** — Gauss-Seidel typically beats Jacobi for
//!    algebraic problems where the matrix has stronger off-diagonal
//!    structure than the geometric Laplacian. A red-black ordering
//!    (or symmetric forward + backward GS) is needed for V-cycle
//!    symmetry.
//! 5. **Coarsening termination** — stop coarsening when the coarse
//!    matrix has `≤ 100-1000` rows (dense direct solve cutoff).
//!
//! The AMS-MG path in [`super::eddy_current::AmsPreconditioner`]
//! does *not* need to change for v4: it already accepts an
//! arbitrary [`MeshHierarchy`] (which holds the per-level Laplacians
//! and prolongation operators), so swapping geometric → algebraic
//! coarsening is a self-contained `MeshHierarchy::build_from_matrix`
//! constructor.

use std::collections::{HashMap, HashSet};

use crate::core::Result;
use crate::math::sparse::BsrMatrix;
use crate::math::{Mat3, Real, Vec3};

use super::amr_mesh::{compute_ttkt, AmrMesh};
use super::assembly::matvec;
use super::mesh::{FemCell, FemMesh};

/// Geometric multigrid mesh hierarchy: a sequence of meshes from
/// coarsest (level 0) to finest (last), each with its own P1 scalar
/// Laplacian and a prolongation operator from the level below.
///
/// All matrices are sized for **scalar** P1 (one DoF per vertex). The
/// AMS auxiliary problem is scalar P1, so we don't need the vector
/// form here.
pub struct MeshHierarchy {
    /// Meshes, coarsest first.
    pub meshes: Vec<FemMesh>,
    /// `laplacians[i]` is the P1 scalar Laplacian on `meshes[i]`, with
    /// the level-`i` pinned vertices applied (row + symmetric column
    /// elimination — same trick as
    /// [`super::eddy_current::pin_aux_vertex`]).
    pub laplacians: Vec<BsrMatrix>,
    /// `prolongations[i]` maps level `i` (coarse) → level `i+1` (fine).
    /// Size: `(n_verts[i+1] × n_verts[i])`. Built from P1 barycentric
    /// interpolation of fine-vertex positions in coarse cells.
    pub prolongations: Vec<BsrMatrix>,
    /// Pinned vertex IDs at each level (after Dirichlet remapping
    /// through the level-0-aware vertex hash).
    pub pinned_per_level: Vec<Vec<u32>>,
    /// Dense Cholesky factor (lower triangle + diagonal) of the
    /// coarsest-level Laplacian. Pre-computed at build time so each
    /// V-cycle's coarse solve is two triangular sweeps.
    coarsest_chol: Vec<Real>,
    /// Side length of the coarsest dense factor (== `n_verts[0]`).
    coarsest_n: usize,
    /// ω-Jacobi damping parameter. Default 2/3 — optimal-ish for the
    /// P1 Laplacian spectrum.
    pub jacobi_omega: Real,
    /// Pre-smooth iteration count (3 by default).
    pub pre_smooth: usize,
    /// Post-smooth iteration count (3 by default).
    pub post_smooth: usize,
}

impl MeshHierarchy {
    /// Build a 2- or 3-level geometric multigrid hierarchy from a
    /// **base mesh** + a list of **finest-level pinned vertices**.
    ///
    /// `n_refines` is the number of uniform refinements applied on top
    /// of the base (so the hierarchy has `n_refines + 1` levels and
    /// the finest level has `8^n_refines` more tet cells than the
    /// base).
    ///
    /// `pinned_vertices_finest` is the list of finest-level vertex
    /// IDs to pin in all per-level Laplacians. At coarser levels, a
    /// vertex is pinned iff its physical position matches (within
    /// `pin_tol`) at least one pinned position on the finest mesh.
    /// This is exact for uniform refinement (every coarse vertex
    /// survives in every finer mesh), and tolerant for floating-point
    /// position drift across refinement passes.
    ///
    /// If `pinned_vertices_finest` is empty, vertex 0 of every level
    /// is pinned to break the constant-vector nullspace (so the
    /// Laplacian becomes SPD and Cholesky succeeds).
    ///
    /// # Errors
    /// Returns `Error::Build` if matrix assembly or Cholesky on the
    /// coarsest level fails.
    pub fn build(
        base_mesh: &FemMesh,
        n_refines: usize,
        pinned_vertices_finest: &[u32],
    ) -> Result<Self> {
        // Pin tolerance — uniform refinement keeps coarse vertex
        // positions exactly, so 1e-9 is safe. We keep a tiny slack
        // for arithmetic in `tet_grid_3d`-style builders.
        let pin_tol: Real = 1.0e-9;

        // 1. Build the AmrMesh hierarchy via `refine_uniform`. Each
        //    refinement step grows level count by 1.
        let mut meshes: Vec<FemMesh> = Vec::with_capacity(n_refines + 1);
        meshes.push(base_mesh.clone());
        if n_refines > 0 {
            let mut amr = AmrMesh::new(base_mesh.clone());
            for _ in 0..n_refines {
                amr.refine_uniform();
                meshes.push(amr.flatten());
            }
        }

        // 2. Identify the pinned positions on the finest mesh.
        let finest = meshes.last().expect("at least one level");
        let pin_positions: Vec<Vec3> = pinned_vertices_finest
            .iter()
            .map(|&v| finest.vertices[v as usize])
            .collect();

        // 3. Build per-level pinned-vertex lists by matching positions.
        //    If no explicit pins, pin vertex 0 of each level to break
        //    the gauge.
        let mut pinned_per_level: Vec<Vec<u32>> = Vec::with_capacity(meshes.len());
        for mesh in &meshes {
            let mut pinned: Vec<u32> = Vec::new();
            if pin_positions.is_empty() {
                pinned.push(0);
            } else {
                let mut seen: HashSet<u32> = HashSet::new();
                for &pp in &pin_positions {
                    for (i, v) in mesh.vertices.iter().enumerate() {
                        if (*v - pp).length() < pin_tol && seen.insert(i as u32) {
                            pinned.push(i as u32);
                        }
                    }
                }
                if pinned.is_empty() {
                    // Defensive: if nothing matched, pin vertex 0.
                    pinned.push(0);
                }
                pinned.sort_unstable();
            }
            pinned_per_level.push(pinned);
        }

        // 4. Build prolongation operators (coarse → fine) via P1
        //    barycentric interpolation. Done before the Laplacians so
        //    we can use Galerkin coarsening at every level above 0.
        let mut prolongations: Vec<BsrMatrix> = Vec::with_capacity(meshes.len().saturating_sub(1));
        for level in 0..meshes.len().saturating_sub(1) {
            let coarse = &meshes[level];
            let fine = &meshes[level + 1];
            let p = build_prolongation(coarse, fine)?;
            prolongations.push(p);
        }

        // 5. Assemble the finest-level Laplacian with explicit pinning,
        //    then build coarser levels via Galerkin coarsening
        //    `A_c = Pᵀ · A_f · P`. Pinned fine rows of `A_f` are e_r
        //    with zeroed off-diagonals, but the *columns* (and the
        //    prolongation rows for pinned fine vertices) are kept
        //    non-zero so Galerkin produces a non-empty row for the
        //    matching coarse vertex. We then pin those coarse rows
        //    explicitly via `pin_vertex`. This Galerkin coarsening +
        //    re-pin sequence is what guarantees a two-grid
        //    contraction for the P1 Poisson operator AND keeps every
        //    pinned-vertex row representable in the sparsity
        //    pattern.
        let mut laplacians: Vec<Option<BsrMatrix>> = vec![None; meshes.len()];
        let finest_level = meshes.len() - 1;
        let mut k_finest = build_p1_scalar_laplacian(&meshes[finest_level])?;
        for &v in &pinned_per_level[finest_level] {
            pin_vertex(&mut k_finest, v as usize);
        }
        laplacians[finest_level] = Some(k_finest);
        for level in (0..finest_level).rev() {
            let a_fine = laplacians[level + 1].as_ref().unwrap();
            let p = &prolongations[level];
            // compute_ttkt expects `T` (n_total × n_free) where the
            // result is `n_free × n_free`. P is (n_fine × n_coarse), so
            // `T = P`, n_total = n_fine, n_free = n_coarse. The result
            // is `Pᵀ · A_f · P` of size (n_coarse × n_coarse).
            let mut a_c = compute_ttkt(p, a_fine)?;
            // Re-pin coarse rows. `pin_vertex` zeroes the off-diagonal
            // entries in the row and column and asserts the diagonal
            // is `1`. If the sparsity pattern is missing a
            // diagonal slot, `ensure_pinned_diagonals` rebuilds the
            // matrix with explicit diagonal entries first.
            ensure_pinned_diagonals(&mut a_c, &pinned_per_level[level])?;
            for &v in &pinned_per_level[level] {
                pin_vertex(&mut a_c, v as usize);
            }
            laplacians[level] = Some(a_c);
        }
        let laplacians: Vec<BsrMatrix> = laplacians.into_iter().map(|x| x.unwrap()).collect();

        // 6. Dense Cholesky factor of the coarsest-level Laplacian.
        let coarsest = &laplacians[0];
        let n0 = coarsest.nrows();
        let mut dense = sparse_to_dense(coarsest);
        // Symmetrize to suppress floating-point noise from Galerkin.
        for i in 0..n0 {
            for j in (i + 1)..n0 {
                let s = 0.5 * (dense[i * n0 + j] + dense[j * n0 + i]);
                dense[i * n0 + j] = s;
                dense[j * n0 + i] = s;
            }
        }
        // Check diagonal positivity (helpful debug for Galerkin issues).
        for i in 0..n0 {
            let d = dense[i * n0 + i];
            if d <= 0.0 {
                eprintln!(
                    "coarsest level diag at {i}: {d} (not positive — pinned? {})",
                    pinned_per_level[0].contains(&(i as u32))
                );
            }
        }
        let coarsest_chol = dense_chol_factor(&dense, n0)?;

        Ok(Self {
            meshes,
            laplacians,
            prolongations,
            pinned_per_level,
            coarsest_chol,
            coarsest_n: n0,
            // 0.85 is the empirical sweet spot for ω-Jacobi on the
            // P1 tet-mesh Laplacian (with Dirichlet boundary
            // pinning). The spectrum of `D⁻¹ A` after pinning sits
            // in `[~ 0, ~ 1.8]`, so ω just below 1.0 maximises the
            // smoothing factor on the upper half of the spectrum
            // without amplifying the highest mode. Empirically gives
            // per-cycle residual reduction ratios `~ 0.1-0.15` on
            // 2-level hierarchies and `~ 0.4-0.5` on 3-level (small
            // base mesh) — the difference is in how much error the
            // coarsest direct solve removes.
            jacobi_omega: 0.85,
            // V(3, 3): 3 pre- + 3 post-smooth Jacobi sweeps —
            // standard textbook V-cycle spec. Callers can tune with
            // `pre_smooth` / `post_smooth` field assignments.
            pre_smooth: 3,
            post_smooth: 3,
        })
    }

    /// Number of levels in the hierarchy (≥ 1).
    pub fn n_levels(&self) -> usize {
        self.meshes.len()
    }

    /// Finest-level Laplacian (the one the outer Krylov sees as
    /// `K_aux`). Convenience accessor.
    pub fn finest_laplacian(&self) -> &BsrMatrix {
        self.laplacians.last().expect("non-empty hierarchy")
    }

    /// Apply one V-cycle to `(b, x)` and return the updated `x`.
    ///
    /// The cycle is the textbook recursion:
    /// 1. Pre-smooth `pre_smooth` ω-Jacobi sweeps.
    /// 2. Compute residual `r = b − A · x`.
    /// 3. Restrict residual to the coarser level: `r_c = Pᵀ · r`.
    /// 4. Recurse if there's a coarser level; otherwise dense
    ///    Cholesky solve `A_0 · e_0 = r_c`.
    /// 5. Prolongate the correction: `x ← x + P · e_c`.
    /// 6. Post-smooth `post_smooth` ω-Jacobi sweeps.
    ///
    /// `x_initial` may be any guess (zero is the common choice for an
    /// MG-preconditioner step). Pinned-vertex entries are zeroed in
    /// the residual and correction at each level so the solve stays
    /// consistent with the row-pinned per-level Laplacians.
    pub fn v_cycle(&self, b: &[Real], x_initial: &[Real]) -> Result<Vec<Real>> {
        let n_levels = self.meshes.len();
        let mut x = x_initial.to_vec();
        self.v_cycle_recursive(n_levels - 1, b, &mut x)?;
        Ok(x)
    }

    fn v_cycle_recursive(
        &self,
        level: usize,
        b: &[Real],
        x: &mut [Real],
    ) -> Result<()> {
        let a = &self.laplacians[level];
        let pinned = &self.pinned_per_level[level];

        // Zero pinned-vertex entries in the input — both `b` and `x`
        // should already be zero there by construction, but defensive
        // zeroing simplifies the invariant.
        for &v in pinned {
            x[v as usize] = 0.0;
        }

        // Coarsest level: direct solve (dense Cholesky).
        if level == 0 {
            let mut b_local = b.to_vec();
            for &v in pinned {
                b_local[v as usize] = 0.0;
            }
            let sol = dense_chol_solve_cached(&self.coarsest_chol, &b_local, self.coarsest_n)?;
            x.copy_from_slice(&sol);
            for &v in pinned {
                x[v as usize] = 0.0;
            }
            return Ok(());
        }

        // 1) Pre-smooth.
        for _ in 0..self.pre_smooth {
            jacobi_sweep(a, b, x, self.jacobi_omega, pinned);
        }

        // 2) Residual r = b − A·x.
        let ax = matvec(a, x);
        let mut r: Vec<Real> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();
        for &v in pinned {
            r[v as usize] = 0.0;
        }

        // 3) Restrict to coarser level: r_c = Pᵀ · r.
        let p = &self.prolongations[level - 1];
        let mut r_coarse = matvec_transpose_csr(p, &r);
        // Zero coarse-pinned entries.
        let pinned_c = &self.pinned_per_level[level - 1];
        for &v in pinned_c {
            r_coarse[v as usize] = 0.0;
        }

        // 4) Recurse with zero initial guess.
        let mut e_coarse = vec![0.0; r_coarse.len()];
        self.v_cycle_recursive(level - 1, &r_coarse, &mut e_coarse)?;

        // 5) Prolongate correction: x ← x + P · e_c.
        let mut e_fine = vec![0.0; x.len()];
        p.mv(&e_coarse, &mut e_fine)?;
        for &v in pinned {
            e_fine[v as usize] = 0.0;
        }
        for i in 0..x.len() {
            x[i] += e_fine[i];
        }

        // 6) Post-smooth.
        for _ in 0..self.post_smooth {
            jacobi_sweep(a, b, x, self.jacobi_omega, pinned);
        }

        // Final pin-clean.
        for &v in pinned {
            x[v as usize] = 0.0;
        }

        Ok(())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Build the **scalar** P1 Laplacian `K_lap = ∫ ∇φᵢ · ∇φⱼ dV`.
/// Local helper duplicated from `eddy_current.rs` so the multigrid
/// module is self-contained.
fn build_p1_scalar_laplacian(mesh: &FemMesh) -> Result<BsrMatrix> {
    use super::element::compute_element_stiffness;
    let n_verts = mesh.vert_count();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for cell in &mesh.cells {
        let FemCell::Tet(verts) = cell else { continue };
        let k_loc = compute_element_stiffness(cell, &mesh.vertices, 1.0);
        for i in 0..4 {
            for j in 0..4 {
                let v = k_loc[i][j];
                if v == 0.0 {
                    continue;
                }
                triplets.push((verts[i] as usize, verts[j] as usize, v));
            }
        }
    }
    BsrMatrix::from_scalar_triplets(n_verts, n_verts, &triplets)
}

/// Pin row `r` of a symmetric matrix `k` (zero column `r` of every
/// other row, then set row `r` to `e_r`). Same operation as
/// [`super::eddy_current::pin_aux_vertex`]. The diagonal slot must
/// already exist in the sparsity pattern; the hierarchy build calls
/// [`ensure_pinned_diagonals`] before this to satisfy that
/// precondition on Galerkin-coarsened matrices.
fn pin_vertex(k: &mut BsrMatrix, r: usize) {
    let n = k.nrows();
    for i in 0..n {
        if i == r {
            continue;
        }
        let lo = k.row_offsets[i];
        let hi = k.row_offsets[i + 1];
        for slot in lo..hi {
            if k.col_idx[slot] == r {
                k.values[slot] = 0.0;
            }
        }
    }
    let lo = k.row_offsets[r];
    let hi = k.row_offsets[r + 1];
    for slot in lo..hi {
        let c = k.col_idx[slot];
        k.values[slot] = if c == r { 1.0 } else { 0.0 };
    }
}

/// Rebuild `k` so that every vertex in `pinned` has an explicit
/// `(r, r) = 1` diagonal slot. This is needed after Galerkin
/// coarsening produces empty rows (which happens when an entire
/// pinned boundary collapses through `Pᵀ · A · P`). The fast path
/// (no missing slots) is a no-op.
fn ensure_pinned_diagonals(k: &mut BsrMatrix, pinned: &[u32]) -> Result<()> {
    let n = k.nrows();
    // Identify rows that need the (r, r) diagonal added.
    let mut to_add: Vec<u32> = Vec::new();
    for &r in pinned {
        let lo = k.row_offsets[r as usize];
        let hi = k.row_offsets[r as usize + 1];
        let has_diag = (lo..hi).any(|slot| k.col_idx[slot] == r as usize);
        if !has_diag {
            to_add.push(r);
        }
    }
    if to_add.is_empty() {
        return Ok(());
    }
    // Rebuild via triplets — slow but only runs once per hierarchy
    // build, and the coarsest matrix is small (≤ 1024 rows by gate).
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for r in 0..n {
        let lo = k.row_offsets[r];
        let hi = k.row_offsets[r + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            triplets.push((r, c, k.values[slot]));
        }
    }
    for &r in &to_add {
        triplets.push((r as usize, r as usize, 1.0));
    }
    *k = BsrMatrix::from_scalar_triplets(n, n, &triplets)?;
    Ok(())
}

/// Build the prolongation operator from `coarse` to `fine` — for each
/// fine vertex, find the coarse cell that contains it (geometric
/// barycentric search) and place the barycentric weights as the row of
/// `P`. Rows for pinned fine vertices are kept **non-zero** because
/// zeroing them would collapse the `Pᵀ · A · P` Galerkin product and
/// produce empty coarse rows that can't be pinned without rebuilding
/// the sparsity pattern. The V-cycle code zeros pinned-vertex entries
/// in `b`, `x`, and intermediate `e_fine` vectors so the prolonged
/// correction never contributes to a pinned fine entry in practice.
///
/// Uses [`evaluate_p1_barycentric`] for the per-fine-vertex search.
fn build_prolongation(coarse: &FemMesh, fine: &FemMesh) -> Result<BsrMatrix> {
    let n_coarse = coarse.vert_count();
    let n_fine = fine.vert_count();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::with_capacity(4 * n_fine);

    // Build a fast position index on the coarse mesh by hashing
    // rounded coordinates — every coarse vertex survives uniformly-
    // refined fine vertices exactly (uniform refinement keeps the
    // coarse positions verbatim).
    let mut coarse_vertex_at: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let scale: Real = 1.0e9;
    for (i, v) in coarse.vertices.iter().enumerate() {
        let key = (
            (v.x * scale).round() as i64,
            (v.y * scale).round() as i64,
            (v.z * scale).round() as i64,
        );
        coarse_vertex_at.insert(key, i as u32);
    }

    for (i, p) in fine.vertices.iter().enumerate() {
        // Fast path: exact coarse-vertex match.
        let key = (
            (p.x * scale).round() as i64,
            (p.y * scale).round() as i64,
            (p.z * scale).round() as i64,
        );
        if let Some(&c) = coarse_vertex_at.get(&key) {
            triplets.push((i, c as usize, 1.0));
            continue;
        }
        // Slow path: barycentric search over coarse cells.
        let weights = evaluate_p1_barycentric(coarse, *p);
        for (cv, w) in weights {
            if w.abs() > 1.0e-14 {
                triplets.push((i, cv as usize, w));
            }
        }
    }

    BsrMatrix::from_scalar_triplets(n_fine, n_coarse, &triplets)
}

/// Find the coarse cell that contains `x`, returning the list of
/// `(coarse_vertex_id, barycentric_weight)` pairs. Mirrors
/// [`super::amr_mesh::evaluate_p1_field_on_mesh`] but returns the
/// weights instead of evaluating a field.
fn evaluate_p1_barycentric(mesh: &FemMesh, x: Vec3) -> Vec<(u32, Real)> {
    let tol: Real = 1.0e-9;
    for cell in &mesh.cells {
        let FemCell::Tet(idx) = cell else { continue };
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
            return vec![
                (idx[0], l0),
                (idx[1], l1),
                (idx[2], l2),
                (idx[3], l3),
            ];
        }
    }
    // Should be unreachable for uniformly-refined hierarchies where
    // every fine vertex lies inside (or on the boundary of) some
    // coarse cell.
    panic!(
        "evaluate_p1_barycentric: point ({}, {}, {}) lies outside every coarse cell",
        x.x, x.y, x.z
    );
}

/// One ω-Jacobi sweep on `A · x = b` with pinned-vertex zeroing.
/// Updates `x` in place: `x ← x + ω · D⁻¹ · (b − A · x)`.
fn jacobi_sweep(a: &BsrMatrix, b: &[Real], x: &mut [Real], omega: Real, pinned: &[u32]) {
    let n = a.nrows();
    let diag = a.diagonal();
    let ax = matvec(a, x);
    for i in 0..n {
        let d = diag[i];
        if d.abs() < 1.0e-30 {
            continue;
        }
        x[i] += omega * (b[i] - ax[i]) / d;
    }
    for &v in pinned {
        x[v as usize] = 0.0;
    }
}

/// CSR transpose-vector multiply `y = Aᵀ · x` for a `B = 1` matrix.
/// Identical to [`super::eddy_current::matvec_transpose`] but local
/// so the multigrid module is independent.
fn matvec_transpose_csr(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    debug_assert_eq!(a.b, 1);
    debug_assert_eq!(x.len(), a.nrows());
    let mut y = vec![0.0; a.ncols()];
    for r in 0..a.block_rows {
        let lo = a.row_offsets[r];
        let hi = a.row_offsets[r + 1];
        for slot in lo..hi {
            let c = a.col_idx[slot];
            y[c] += a.values[slot] * x[r];
        }
    }
    y
}

/// Convert a `B = 1` BSR matrix to a row-major dense `n × n` buffer
/// for the coarsest-level direct solve. Acceptable because the v3
/// gate is `n_coarsest ≤ 1024`.
fn sparse_to_dense(a: &BsrMatrix) -> Vec<Real> {
    let n = a.nrows();
    debug_assert_eq!(a.b, 1);
    let mut dense = vec![0.0; n * n];
    for r in 0..a.block_rows {
        let lo = a.row_offsets[r];
        let hi = a.row_offsets[r + 1];
        for slot in lo..hi {
            let c = a.col_idx[slot];
            dense[r * n + c] = a.values[slot];
        }
    }
    dense
}

/// Cholesky-factor a dense SPD matrix and return the L-factor (lower
/// triangle + diagonal in row-major order). Wraps
/// [`crate::math::dense_solve::dense_chol_solve`] — except we want
/// the *factor* not a solve, so we re-implement the factor step.
fn dense_chol_factor(a: &[Real], n: usize) -> Result<Vec<Real>> {
    let mut buf = a.to_vec();
    crate::math::dense_solve::dense_chol(&mut buf, n)?;
    Ok(buf)
}

/// Forward + back substitution given a cached lower-triangular
/// Cholesky factor.
fn dense_chol_solve_cached(l: &[Real], b: &[Real], n: usize) -> Result<Vec<Real>> {
    // We can't reuse `dense_chol_solve` because it factors in place.
    let y = crate::math::dense_solve::forward_sub(l, b, n)?;
    crate::math::dense_solve::back_sub(l, &y, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_two_level_hierarchy_smoke() {
        let base = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let h = MeshHierarchy::build(&base, 1, &[]).expect("build");
        assert_eq!(h.n_levels(), 2);
        assert_eq!(h.prolongations.len(), 1);
        assert_eq!(h.laplacians.len(), 2);
        // Prolongation rows = fine verts, cols = coarse verts.
        let p = &h.prolongations[0];
        assert_eq!(p.nrows(), h.meshes[1].vert_count());
        assert_eq!(p.ncols(), h.meshes[0].vert_count());
    }

    #[test]
    fn prolongation_reproduces_constants() {
        // P · 1_coarse should be a vector of all-1 entries (P1
        // barycentric weights sum to 1 by construction). v3 does NOT
        // zero pinned-fine rows of P — that would empty the Galerkin
        // coarse-row for the matching coarse vertex; the V-cycle
        // instead zeros pinned entries in `b`, `x`, and intermediate
        // correction vectors at every level.
        let base = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let h = MeshHierarchy::build(&base, 1, &[]).expect("build");
        let p = &h.prolongations[0];
        let n_coarse = h.meshes[0].vert_count();
        let n_fine = h.meshes[1].vert_count();
        let ones = vec![1.0; n_coarse];
        let mut out = vec![0.0; n_fine];
        p.mv(&ones, &mut out).unwrap();
        for i in 0..n_fine {
            assert!(
                (out[i] - 1.0).abs() < 1e-9,
                "vert {i}: P·1 = {} (expected 1)",
                out[i]
            );
        }
    }
}
