//! Algebraic multigrid (Ruge-Stüben classical AMG) — Stage 48 v4 of the
//! AMS preconditioner.
//!
//! # Scope
//!
//! v3 shipped **geometric** multigrid via [`super::multigrid::MeshHierarchy`]
//! built from uniform refinement. v4 replaces it with **algebraic**
//! multigrid: C/F splitting and prolongation are derived directly from
//! the matrix `A` — no mesh hierarchy needed.
//!
//! # Algorithm (classical Ruge-Stüben)
//!
//! Given the (symmetric, M-matrix-like) SPD operator `A`:
//!
//! 1. **Strong-connection criterion** — row `i` strongly depends on row
//!    `j` (`j ∈ S(i)`) iff `−A[i, j] ≥ θ · max_{k ≠ i, A[i,k] < 0}
//!    (−A[i, k])`. Default `θ = 0.25` (Ruge-Stüben 1987). For M-matrices
//!    the criterion only considers negative off-diagonals — those are
//!    the "strongly diffusing" connections that the smoother fails to
//!    damp.
//!
//! 2. **C/F splitting** — classical greedy:
//!    - `lambda[i]` = number of unassigned points strongly connected
//!      *to* `i` (column-wise count: `|{j : i ∈ S(j)}|`).
//!    - Repeat: pick the unassigned `i` with the largest `lambda`,
//!      promote it to **C**, mark every `j ∈ S(i)` as **F**, and
//!      increment `lambda[k]` for every `k` strongly connected to one
//!      of those `F` nodes (the "common neighbour" boost).
//!    - Edge cases: isolated rows (no strong connections) become C
//!      points by default — the prolongation row is the identity.
//!
//! 3. **Classical interpolation** — for each F-row `i`:
//!    - Strong-C neighbours `C_i = S(i) ∩ C`.
//!    - Strong-F neighbours `F_i = S(i) ∩ F` — these get **redistributed**
//!      onto `C_i`: for `j ∈ F_i`, split `A[i, j]` among `C_i` weighted
//!      by `A[j, c]` (the strong-F-to-C connections of `j`).
//!    - Weak (non-strong) neighbours are lumped onto the diagonal.
//!    - Final weight: `P[i, c] = −(A[i, c] + Σ_{j ∈ F_i} A[i, j] ·
//!      A[j, c] / Σ_{c' ∈ C_i} A[j, c']) / (A[i, i] + lumped)`.
//!
//! 4. **Galerkin coarsening** — `A_c = Pᵀ · A · P` via
//!    [`super::amr_mesh::compute_ttkt`] (same helper the geometric MG
//!    uses).
//!
//! 5. **Smoother** — same ω-Jacobi as the geometric path. Classical AMG
//!    benefits from Gauss-Seidel smoothing, but Jacobi is enough to
//!    demonstrate the v4 win on the eddy-current test problems where
//!    the matrix is well-scaled and the Jacobi smoothing factor is
//!    competitive.
//!
//! # Differences vs v3 geometric MG
//!
//! - **Build path**: `AmgHierarchy::from_matrix(a, n_levels)` consumes
//!   only the matrix; no `FemMesh` or `AmrMesh` needed.
//! - **Prolongation**: algebraic (matrix-derived) instead of P1
//!   barycentric. The Galerkin product `Pᵀ A P` and dense-Cholesky
//!   coarsest-level solve are identical.
//! - **Pinning**: AMG inherits the pinned-vertex structure through the
//!   matrix `A` — rows already pinned to `e_r` have no strong
//!   connections (off-diagonal is zero), so they become isolated C
//!   points and the prolongation is the identity on them. No
//!   separate `pinned_per_level` list is needed.
//!
//! # AMS integration
//!
//! [`super::eddy_current::AmsPreconditioner`] gains an
//! `amg_hierarchy: Option<AmgHierarchy>` field; when set, `apply`
//! switches to AMG V-cycles. The wiring is parallel to the existing
//! `mg_hierarchy` (geometric) path.

use std::collections::HashSet;

use crate::core::{Error, Result};
use crate::math::Real;
use crate::math::sparse::BsrMatrix;

use super::amr_mesh::compute_ttkt;
use super::assembly::matvec;

/// One level of the AMG hierarchy.
///
/// `matrix` is the level's operator (level 0 = finest), `prolongation`
/// maps the *next coarser* level up to this level (size
/// `n_this × n_coarser`). The coarsest level has `prolongation = None`.
pub struct AmgLevel {
    /// Level matrix (scalar B = 1 BSR).
    pub matrix: BsrMatrix,
    /// Prolongation from the next coarser level → this level. `None`
    /// for the coarsest level.
    pub prolongation: Option<BsrMatrix>,
    /// Number of C-points produced at this level (0 for the coarsest).
    /// Diagnostic only.
    pub n_coarse: usize,
}

/// AMG mesh-independent multigrid hierarchy.
///
/// `levels[0]` is the **finest** level (the one outer Krylov sees);
/// `levels.last()` is the **coarsest**, solved exactly via dense
/// Cholesky.
pub struct AmgHierarchy {
    pub levels: Vec<AmgLevel>,
    /// Dense Cholesky factor of the coarsest-level matrix.
    coarsest_chol: Vec<Real>,
    /// Side length of the coarsest dense factor.
    coarsest_n: usize,
    /// Strong-connection threshold θ used at build time (0.25 default).
    pub theta: Real,
    /// ω-Jacobi damping parameter. Default 0.85 (same as geometric MG).
    pub jacobi_omega: Real,
    /// Pre-smooth iteration count (3 by default).
    pub pre_smooth: usize,
    /// Post-smooth iteration count (3 by default).
    pub post_smooth: usize,
}

impl AmgHierarchy {
    /// Build a Ruge-Stüben AMG hierarchy with at most `n_levels` levels.
    ///
    /// Coarsening stops early when:
    /// - The coarse matrix has ≤ 32 rows (dense direct solve regime).
    /// - C/F splitting produces no C-points (everything is isolated),
    ///   which means there are no more "strong" connections to coarsen.
    /// - The coarsening ratio is too poor (`n_coarse > 0.9 · n_fine`),
    ///   indicating diminishing returns.
    ///
    /// # Errors
    /// Returns `Error::Build` if Cholesky on the coarsest level fails.
    pub fn from_matrix(a: &BsrMatrix, n_levels: usize) -> Result<Self> {
        Self::from_matrix_with_theta(a, n_levels, 0.25)
    }

    /// Variant that exposes the strong-connection threshold.
    pub fn from_matrix_with_theta(a: &BsrMatrix, n_levels: usize, theta: Real) -> Result<Self> {
        if a.b != 1 {
            return Err(Error::Build(
                "AmgHierarchy::from_matrix: only B = 1 supported".into(),
            ));
        }
        if a.nrows() != a.ncols() {
            return Err(Error::Build(
                "AmgHierarchy::from_matrix: matrix must be square".into(),
            ));
        }
        let n_levels = n_levels.max(1);

        let mut levels: Vec<AmgLevel> = Vec::with_capacity(n_levels);
        let mut current = a.clone();
        // Cap on coarse-level size — below this we stop and direct-solve.
        let coarse_cap: usize = 32;

        for _ in 0..n_levels.saturating_sub(1) {
            let n = current.nrows();
            if n <= coarse_cap {
                break;
            }
            // 1. Strong-connection graph S.
            let strong = build_strong_connections(&current, theta);
            // 2. C/F splitting.
            let split = c_f_split(&strong, n);
            let n_coarse = split.c_to_global.len();
            if n_coarse == 0 || n_coarse >= ((n as f64 * 0.95) as usize).max(1) {
                // Coarsening stalled (or worse — every node is C).
                // Bail out and direct-solve the current level.
                break;
            }
            // 3. Classical interpolation.
            let p = build_classical_interpolation(&current, &strong, &split)?;
            // 4. Galerkin coarsening.
            let a_coarse = compute_ttkt(&p, &current)?;
            // Push the fine level (with its outgoing prolongation).
            levels.push(AmgLevel {
                matrix: current,
                prolongation: Some(p),
                n_coarse,
            });
            current = a_coarse;
        }
        // Push the final (coarsest) level with no prolongation.
        levels.push(AmgLevel {
            matrix: current,
            prolongation: None,
            n_coarse: 0,
        });

        // Dense Cholesky factor of the coarsest level.
        let coarsest = &levels.last().expect("at least one level").matrix;
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
        // Guard against zero/negative diagonals — pinned rows are `e_r`
        // (diagonal 1), but Galerkin coarsening of a partially-pinned
        // matrix can produce tiny diagonals. Bump to 1e-12 to keep
        // Cholesky stable.
        for i in 0..n0 {
            if dense[i * n0 + i] <= 0.0 {
                dense[i * n0 + i] = 1.0e-12;
            }
        }
        let coarsest_chol = dense_chol_factor(&dense, n0)?;

        Ok(Self {
            levels,
            coarsest_chol,
            coarsest_n: n0,
            theta,
            jacobi_omega: 0.85,
            pre_smooth: 3,
            post_smooth: 3,
        })
    }

    /// Number of levels in the hierarchy (≥ 1).
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Finest-level matrix (the one the outer Krylov sees).
    pub fn finest_matrix(&self) -> &BsrMatrix {
        &self.levels[0].matrix
    }

    /// Apply one V-cycle to `(b, x)` and return the updated `x`.
    pub fn v_cycle(&self, b: &[Real], x_initial: &[Real]) -> Result<Vec<Real>> {
        let mut x = x_initial.to_vec();
        self.v_cycle_recursive(0, b, &mut x)?;
        Ok(x)
    }

    fn v_cycle_recursive(&self, level: usize, b: &[Real], x: &mut [Real]) -> Result<()> {
        let n_levels = self.levels.len();
        let a = &self.levels[level].matrix;

        // Coarsest level — direct dense Cholesky solve.
        if level + 1 == n_levels {
            let sol = dense_chol_solve_cached(&self.coarsest_chol, b, self.coarsest_n)?;
            x.copy_from_slice(&sol);
            return Ok(());
        }

        // 1) Pre-smooth.
        for _ in 0..self.pre_smooth {
            jacobi_sweep(a, b, x, self.jacobi_omega);
        }

        // 2) Residual r = b − A·x.
        let ax = matvec(a, x);
        let r: Vec<Real> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();

        // 3) Restrict to coarser level: r_c = Pᵀ · r.
        // The prolongation lives on the *current* level (it maps
        // coarser-to-this), so Pᵀ maps this-to-coarser.
        let p = self.levels[level]
            .prolongation
            .as_ref()
            .expect("non-coarsest level has prolongation");
        let r_coarse = matvec_transpose_csr(p, &r);

        // 4) Recurse with zero initial guess.
        let mut e_coarse = vec![0.0; r_coarse.len()];
        self.v_cycle_recursive(level + 1, &r_coarse, &mut e_coarse)?;

        // 5) Prolongate correction: x ← x + P · e_c.
        let mut e_fine = vec![0.0; x.len()];
        p.mv(&e_coarse, &mut e_fine)?;
        for i in 0..x.len() {
            x[i] += e_fine[i];
        }

        // 6) Post-smooth.
        for _ in 0..self.post_smooth {
            jacobi_sweep(a, b, x, self.jacobi_omega);
        }

        Ok(())
    }
}

// ─── Strong-connection graph ───────────────────────────────────────────

/// Strong-connection neighbour list, per row.
///
/// `strong_neighbours[i]` lists every row `j ≠ i` for which
/// `|A[i, j]| ≥ θ · max_{k ≠ i} |A[i, k]|` (the task-spec absolute-
/// value criterion), filtered to **negative** off-diagonals (the
/// "M-matrix-like" structure that the smoother fails to damp).
/// Positive off-diagonals — which can appear in Galerkin-coarsened
/// matrices — are excluded from the strong graph because the
/// classical direct-interpolation formula assumes negative weights
/// (`P[i, c] = −w_ic / a_ii > 0` requires `w_ic < 0`).
///
/// `strong_neighbours_of[j]` lists every `i` for which `j` is a strong
/// neighbour (the transpose graph). Used for the `lambda` count in
/// the C/F splitting.
#[derive(Debug, Clone)]
pub(crate) struct StrongGraph {
    pub strong_neighbours: Vec<Vec<u32>>,
    pub strong_neighbours_of: Vec<Vec<u32>>,
}

pub(crate) fn build_strong_connections(a: &BsrMatrix, theta: Real) -> StrongGraph {
    debug_assert_eq!(a.b, 1);
    let n = a.nrows();
    let mut strong_neighbours: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut strong_neighbours_of: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..n {
        let lo = a.row_offsets[i];
        let hi = a.row_offsets[i + 1];
        // Find max(|A[i, k]|) over k ≠ i — the row-wise magnitude
        // reference. The cut-off applies the *absolute-value* form
        // of the criterion (per the task spec), but only negative
        // off-diagonals are kept as strong (M-matrix assumption).
        let mut max_abs: Real = 0.0;
        for slot in lo..hi {
            let j = a.col_idx[slot];
            if j == i {
                continue;
            }
            let v = a.values[slot].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        if max_abs <= 0.0 {
            continue;
        }
        let cutoff = theta * max_abs;
        for slot in lo..hi {
            let j = a.col_idx[slot];
            if j == i {
                continue;
            }
            let v = a.values[slot];
            // Keep only negative off-diagonals with magnitude ≥ cutoff.
            if v < 0.0 && -v >= cutoff {
                strong_neighbours[i].push(j as u32);
                strong_neighbours_of[j].push(i as u32);
            }
        }
    }
    StrongGraph {
        strong_neighbours,
        strong_neighbours_of,
    }
}

// ─── C/F splitting ─────────────────────────────────────────────────────

/// Classification of a row in the strong-connection graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Marker {
    Unassigned,
    C,
    F,
}

/// Result of the C/F splitting.
pub(crate) struct CfSplit {
    /// Per-row classification.
    pub marker: Vec<Marker>,
    /// Map `c_index → global_row` (coarse index = position in this
    /// vector).
    pub c_to_global: Vec<u32>,
    /// Map `global_row → coarse_index` (or `u32::MAX` for F-points).
    pub global_to_c: Vec<u32>,
}

/// Classical Ruge-Stüben C/F splitting using the `lambda`-greedy
/// strategy.
///
/// Algorithm:
/// 1. `lambda[i] = |{j : i ∈ S(j)}| = strong_neighbours_of[i].len()`.
/// 2. Pick the unassigned `i` with the largest `lambda` (ties: smallest
///    index). Promote `i` → C.
/// 3. For every `j ∈ S(i)` (strong neighbour of `i`) that's still
///    unassigned: promote `j` → F.
/// 4. For every `k ∈ S(j)` (strong neighbour of the new F-node) that's
///    still unassigned: increment `lambda[k]` (boost — it now has more
///    F-friends to interpolate from).
/// 5. Repeat until everything is assigned.
///
/// Isolated rows (no incoming strong edges and no outgoing strong
/// edges) get classified as C — they have a single non-zero on the
/// diagonal and their prolongation row is the identity.
pub(crate) fn c_f_split(strong: &StrongGraph, n: usize) -> CfSplit {
    let mut marker: Vec<Marker> = vec![Marker::Unassigned; n];
    // λ[i] = number of unassigned points strongly connected *to* i
    // (i.e. that depend on i for interpolation).
    let mut lambda: Vec<i32> = strong
        .strong_neighbours_of
        .iter()
        .map(|v| v.len() as i32)
        .collect();

    // First pass: any row with no strong neighbours and no incoming
    // strong edges is an isolated row — classify it as C immediately.
    // This catches pinned-Dirichlet rows (which are `e_r` after
    // pinning) and the gauge-pinned vertex of the AMS auxiliary
    // Laplacian.
    for i in 0..n {
        if strong.strong_neighbours[i].is_empty() && strong.strong_neighbours_of[i].is_empty() {
            marker[i] = Marker::C;
            lambda[i] = -1; // never re-pick
        }
    }

    // Greedy loop.
    loop {
        // Find the unassigned row with max lambda. Ties: smallest index.
        let mut best: Option<(i32, usize)> = None;
        for i in 0..n {
            if marker[i] != Marker::Unassigned {
                continue;
            }
            let l = lambda[i];
            if best.map_or(true, |(bl, _)| l > bl) {
                best = Some((l, i));
            }
        }
        let Some((_, i)) = best else {
            break;
        };
        // Promote i to C.
        marker[i] = Marker::C;
        lambda[i] = -1;
        // Snapshot strong neighbours of i — these become F.
        let s_i: Vec<u32> = strong.strong_neighbours[i].clone();
        for &j in &s_i {
            if marker[j as usize] != Marker::Unassigned {
                continue;
            }
            marker[j as usize] = Marker::F;
            lambda[j as usize] = -1;
            // Boost lambda of every unassigned k strongly connected to j.
            // Higher lambda → more likely to be the next C-point.
            for &k in &strong.strong_neighbours[j as usize] {
                if marker[k as usize] == Marker::Unassigned {
                    lambda[k as usize] += 1;
                }
            }
        }
    }

    // Any leftover Unassigned (defensive — shouldn't happen with the
    // isolated-row prepass + greedy loop) becomes C.
    for i in 0..n {
        if marker[i] == Marker::Unassigned {
            marker[i] = Marker::C;
        }
    }

    // Second pass — Ruge-Stüben "interpolatory" check. For each F-point
    // `i` and each strong-F neighbour `j` of `i`, ensure there's a
    // common strong-C neighbour `c ∈ S(i) ∩ S(j) ∩ C`. If not, promote
    // `j` to C so that `j` can serve as `i`'s interpolation surrogate.
    // This is the textbook "Pass 2" of classical AMG splitting (Ruge &
    // Stüben 1987) and is essential for the direct-interpolation
    // formula to produce well-defined weights — without it, F→F
    // redistribution can fail (no common C neighbour means the
    // `Σ_{c ∈ C_i} a_jc` denominator is zero).
    //
    // Implementation: per F-point i, build the set `C_i = S(i) ∩ C`,
    // then for each strong-F-neighbour j ∈ S(i), check
    // `S(j) ∩ C_i ≠ ∅`. If empty, promote j to C and add it to C_i so
    // subsequent F-neighbours of i see the new C-point.
    let mut tries: usize = 0;
    let mut changed = true;
    while changed && tries < 3 {
        tries += 1;
        changed = false;
        for i in 0..n {
            if marker[i] != Marker::F {
                continue;
            }
            // Build C_i = strong-C neighbours of i.
            let s_i = &strong.strong_neighbours[i];
            let mut c_i: std::collections::HashSet<u32> = s_i
                .iter()
                .copied()
                .filter(|&j| marker[j as usize] == Marker::C)
                .collect();
            // For each strong-F neighbour, check overlap with C_i.
            for &j in s_i {
                if marker[j as usize] != Marker::F {
                    continue;
                }
                let s_j = &strong.strong_neighbours[j as usize];
                let has_common = s_j
                    .iter()
                    .any(|k| marker[*k as usize] == Marker::C && c_i.contains(k));
                if !has_common {
                    // Promote j to C — it now serves as i's
                    // interpolation surrogate for itself.
                    marker[j as usize] = Marker::C;
                    c_i.insert(j);
                    changed = true;
                }
            }
        }
    }

    // Build C-index ↔ global maps.
    let mut c_to_global: Vec<u32> = Vec::new();
    let mut global_to_c: Vec<u32> = vec![u32::MAX; n];
    for i in 0..n {
        if marker[i] == Marker::C {
            global_to_c[i] = c_to_global.len() as u32;
            c_to_global.push(i as u32);
        }
    }
    CfSplit {
        marker,
        c_to_global,
        global_to_c,
    }
}

// ─── Classical interpolation ──────────────────────────────────────────

/// Build the classical Ruge-Stüben direct interpolation operator.
///
/// `P` has size `n × n_coarse` where `n_coarse = split.c_to_global.len()`.
///
/// - C-row `i`: identity entry `P[i, global_to_c[i]] = 1`.
/// - F-row `i`: weighted sum of strong-C neighbours, with strong-F
///   neighbours redistributed onto C via the standard
///   `A[i, j] · A[j, c] / Σ_c' A[j, c']` formula. Weak (non-strong)
///   off-diagonals are absorbed into the diagonal.
///
/// The interpolation formula for `c ∈ C_i` is
///
/// ```text
///   P[i, c] = − (a_ic + Σ_{j ∈ F_i} α_j · a_jc) / d_i
/// ```
///
/// where
/// - `a_ic = A[i, c]` (negative for strong connection),
/// - `α_j = a_ij / Σ_{c' ∈ C_i} a_jc'` (the F-to-C redistribution
///   weight, finite as long as `j` has at least one strong-C
///   neighbour),
/// - `d_i = A[i, i] + (weak off-diagonals)` (the diagonal absorbs the
///   F-points whose redistribution is undefined plus all weak
///   neighbours).
///
/// When a strong-F neighbour `j` has *no* strong-C neighbours (a
/// degenerate case), we fall back to lumping its `A[i, j]` onto the
/// diagonal. This guarantees the interpolation is well-defined for
/// every C/F partition produced by [`c_f_split`].
pub(crate) fn build_classical_interpolation(
    a: &BsrMatrix,
    strong: &StrongGraph,
    split: &CfSplit,
) -> Result<BsrMatrix> {
    debug_assert_eq!(a.b, 1);
    let n = a.nrows();
    let n_coarse = split.c_to_global.len();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();

    // For fast random access to A[i, j] within a row, materialise a
    // per-row HashMap once: `row_map[i][j] = A[i, j]`. Saves an
    // O(nnz_row) probe inside the F-redistribution inner loop.
    let mut row_map: Vec<std::collections::HashMap<u32, Real>> = Vec::with_capacity(n);
    for i in 0..n {
        let lo = a.row_offsets[i];
        let hi = a.row_offsets[i + 1];
        let mut map = std::collections::HashMap::with_capacity(hi - lo);
        for slot in lo..hi {
            map.insert(a.col_idx[slot] as u32, a.values[slot]);
        }
        row_map.push(map);
    }

    for i in 0..n {
        if split.marker[i] == Marker::C {
            // C-row: identity row in P.
            let c_idx = split.global_to_c[i];
            triplets.push((i, c_idx as usize, 1.0));
            continue;
        }
        // F-row: build weighted interpolation.
        // Classify strong neighbours into C_i and F_i.
        let mut c_neighbours: Vec<u32> = Vec::new();
        let mut f_neighbours: Vec<u32> = Vec::new();
        let strong_set: HashSet<u32> = strong.strong_neighbours[i].iter().copied().collect();
        for &j in &strong.strong_neighbours[i] {
            if split.marker[j as usize] == Marker::C {
                c_neighbours.push(j);
            } else {
                f_neighbours.push(j);
            }
        }
        // Edge case — no strong-C neighbours. This *should* be rare
        // (the C/F splitting guarantees every F has at least one C in
        // its strong neighbours, by construction), but defensively we
        // fall back to "I am my own C-point": dump everything onto
        // the diagonal and emit no off-diagonal P entry. The V-cycle
        // then has no coarse correction for this F-row, but the
        // smoother still operates on it.
        if c_neighbours.is_empty() {
            continue;
        }

        // Diagonal absorption: weak off-diagonals plus the F-row's own
        // a_ii.
        let a_ii = row_map[i].get(&(i as u32)).copied().unwrap_or(0.0);
        let mut d_i = a_ii;
        let lo = a.row_offsets[i];
        let hi = a.row_offsets[i + 1];
        for slot in lo..hi {
            let j = a.col_idx[slot] as u32;
            if j as usize == i {
                continue;
            }
            if !strong_set.contains(&j) {
                // Weak off-diagonal: absorbed onto diagonal.
                d_i += a.values[slot];
            }
        }
        // Pre-compute per-F-neighbour α denominator: Σ_{c' ∈ C_i} a_jc'.
        // If that sum is zero, lump A[i, j] onto the diagonal instead
        // (defensive — happens when an F-neighbour is itself isolated).
        let mut alpha_denom: Vec<Real> = Vec::with_capacity(f_neighbours.len());
        for &j in &f_neighbours {
            let mut sum: Real = 0.0;
            let rm_j = &row_map[j as usize];
            for &c in &c_neighbours {
                if let Some(&v) = rm_j.get(&c) {
                    sum += v;
                }
            }
            alpha_denom.push(sum);
        }
        for (fi, &j) in f_neighbours.iter().enumerate() {
            if alpha_denom[fi] == 0.0 {
                let a_ij = row_map[i].get(&j).copied().unwrap_or(0.0);
                d_i += a_ij;
            }
        }

        if d_i.abs() < 1.0e-30 {
            // Degenerate F-row (shouldn't happen for an SPD operator
            // but defensive). Emit identity-onto-first-C-neighbour.
            let c0 = c_neighbours[0];
            let c_idx = split.global_to_c[c0 as usize];
            triplets.push((i, c_idx as usize, 1.0));
            continue;
        }

        // Build interpolation weights P[i, c] for c ∈ C_i.
        for &c in &c_neighbours {
            let a_ic = row_map[i].get(&c).copied().unwrap_or(0.0);
            let mut numerator = a_ic;
            for (fi, &j) in f_neighbours.iter().enumerate() {
                if alpha_denom[fi] == 0.0 {
                    // Already lumped onto diagonal — skip.
                    continue;
                }
                let a_ij = row_map[i].get(&j).copied().unwrap_or(0.0);
                let a_jc = row_map[j as usize].get(&c).copied().unwrap_or(0.0);
                numerator += a_ij * a_jc / alpha_denom[fi];
            }
            let w = -numerator / d_i;
            if w.abs() > 1.0e-15 {
                let c_idx = split.global_to_c[c as usize];
                triplets.push((i, c_idx as usize, w));
            }
        }
    }

    if triplets.is_empty() {
        // Empty prolongation — shouldn't happen since C-points always
        // emit identity rows.
        return Err(Error::Build(
            "build_classical_interpolation: produced empty prolongation".into(),
        ));
    }
    BsrMatrix::from_scalar_triplets(n, n_coarse, &triplets)
}

// ─── Smoother + transpose helpers (mirror of multigrid.rs) ────────────

/// One ω-Jacobi sweep on `A · x = b`. No pinned-vertex zeroing because
/// AMG inherits pinning through the matrix structure (pinned rows are
/// `e_r` so the sweep is a no-op on them).
fn jacobi_sweep(a: &BsrMatrix, b: &[Real], x: &mut [Real], omega: Real) {
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
}

/// CSR transpose-vector multiply `y = Aᵀ · x` for a `B = 1` matrix.
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

/// Convert a `B = 1` BSR matrix to a row-major dense `n × n` buffer.
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

fn dense_chol_factor(a: &[Real], n: usize) -> Result<Vec<Real>> {
    let mut buf = a.to_vec();
    crate::math::dense_solve::dense_chol(&mut buf, n)?;
    Ok(buf)
}

fn dense_chol_solve_cached(l: &[Real], b: &[Real], n: usize) -> Result<Vec<Real>> {
    let y = crate::math::dense_solve::forward_sub(l, b, n)?;
    crate::math::dense_solve::back_sub(l, &y, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::mesh::FemMesh;

    fn build_p1_laplacian_unpinned(mesh: &FemMesh) -> BsrMatrix {
        use crate::fem::element::compute_element_stiffness;
        use crate::fem::mesh::FemCell;
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
        BsrMatrix::from_scalar_triplets(n_verts, n_verts, &triplets).unwrap()
    }

    /// Pin vertex 0 to break the constant-vector nullspace (so the
    /// matrix becomes SPD).
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

    #[test]
    fn strong_connections_pick_up_negative_off_diagonals() {
        // 1D-like P1 Laplacian-ish: diag = 2, off = -1.
        // (Build by hand to keep the test small.)
        let n = 5;
        let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, -1.0));
            }
        }
        let a = BsrMatrix::from_scalar_triplets(n, n, &triplets).unwrap();
        let s = build_strong_connections(&a, 0.25);
        // Every interior row has two strong neighbours (both off-
        // diagonals are equal to the max). Endpoints have one each.
        assert_eq!(s.strong_neighbours[0].len(), 1);
        assert_eq!(s.strong_neighbours[2].len(), 2);
        assert_eq!(s.strong_neighbours[4].len(), 1);
    }

    #[test]
    fn c_f_split_partitions_a_1d_chain_roughly_half() {
        let n = 9;
        let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, -1.0));
            }
        }
        let a = BsrMatrix::from_scalar_triplets(n, n, &triplets).unwrap();
        let s = build_strong_connections(&a, 0.25);
        let split = c_f_split(&s, n);
        // For a 1D chain, classical AMG picks every other point as C.
        // n=9 ⇒ 5 C and 4 F (or vice versa depending on tie-breaking).
        let n_c = split.c_to_global.len();
        assert!(
            n_c >= 3 && n_c <= 6,
            "expected ~half coarse, got {n_c} of {n}"
        );
    }

    #[test]
    fn amg_hierarchy_builds_on_tet_grid_p1_laplacian() {
        let mesh = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
        let mut a = build_p1_laplacian_unpinned(&mesh);
        pin_vertex(&mut a, 0);
        let h = AmgHierarchy::from_matrix(&a, 3).expect("build");
        assert!(
            h.n_levels() >= 2,
            "expected ≥ 2 levels, got {}",
            h.n_levels()
        );
        let n_fine = h.levels[0].matrix.nrows();
        let n_coarse_first = h.levels[1].matrix.nrows();
        assert!(
            n_coarse_first < n_fine,
            "expected coarsening: {n_coarse_first} < {n_fine}"
        );
    }
}
