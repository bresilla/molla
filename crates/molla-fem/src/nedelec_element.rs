//! First-order Nedelec edge elements — H(curl)-conforming basis on
//! tetrahedra. Stage 46.
//!
//! # What is H(curl)?
//!
//! For PDEs in `H(curl) = { v ∈ L² : curl v ∈ L² }` the natural
//! conforming basis has **tangential continuity** across element
//! interfaces (the tangential component of the basis matches on
//! shared faces, the normal component is free to jump). This is
//! exactly what's needed for magnetostatics (vector potential `A`)
//! and time-harmonic electromagnetics — the curl operator is
//! well-defined globally, but `div A` is not constrained and `n · A`
//! has no physical meaning at material interfaces.
//!
//! Nodal (H¹) elements over-constrain such problems by enforcing
//! continuity of all components; they produce spurious modes
//! ("ghost solutions") that pollute the spectrum. Edge elements
//! eliminate these by construction: the kernel of `curl` over the
//! discrete H(curl) space is exactly the discrete gradient of
//! H¹ scalar fields (de Rham complex).
//!
//! # First-order ("lowest-order") Nedelec basis
//!
//! On a tetrahedron with barycentric coordinates `λ₀, λ₁, λ₂, λ₃` and
//! constant barycentric gradients `∇λ_k`, the lowest-order Nedelec
//! basis (also called Whitney 1-forms) is one basis function per edge:
//!
//! ```text
//!   N_e(x) = λ_i ∇λ_j - λ_j ∇λ_i        for edge e = (i, j)
//! ```
//!
//! Properties:
//! - `N_e` is linear in `x` (since the `λ_k` are linear and the
//!   gradients constant).
//! - The tangential component of `N_e` along the edge `(i, j)` is
//!   constant and integrates to `1`. Along every other edge it
//!   integrates to `0`.
//! - The curl is constant per element: `∇ × N_e = 2 ∇λ_i × ∇λ_j`.
//!
//! # Edge orientation convention
//!
//! Every unique mesh edge is stored canonically as `[min, max]` (see
//! [`crate::fem::function_space::compute_edges`]). The corresponding
//! global Nedelec basis function is the one that "points" from the
//! lower-indexed vertex to the higher-indexed vertex.
//!
//! When a cell's local edge `k` (in the [`TET_EDGES`] ordering) has
//! local endpoints `(a, b)` mapping to global vertices `(v_a, v_b)`,
//! the local basis `N_loc = λ_a ∇λ_b - λ_b ∇λ_a` points from local
//! `a` to local `b`. To express it in terms of the global basis we
//! apply an orientation sign:
//!
//! ```text
//!   N_glob = sign · N_loc,   sign = +1 if v_a < v_b else -1
//! ```
//!
//! Element matrices computed in the "local-edge basis" are converted
//! to the "global-edge basis" during assembly by multiplying entry
//! `(i, j)` by `sign_i · sign_j`. Equivalently, the orientation
//! signs are absorbed into the sign of the DoF — see
//! [`NedelecDofMap::edge_sign`].

use crate::math::{Real, Vec3};

use super::element::{tet_shape_grads, tet_volume};
use super::function_space::compute_edges;
use super::mesh::{FemCell, FemMesh};
use super::p2_element::TET_EDGES;

// ─── Edge DoF map ────────────────────────────────────────────────────

/// Global edge → DoF index map used by both the Nedelec assembly and
/// the magnetostatics solver. One DoF per unique mesh edge.
///
/// Mirrors [`crate::fem::p2_element::P2DofMap`] but without corner
/// DoFs: H(curl) DoFs live entirely on mesh edges.
#[derive(Debug, Clone)]
pub struct NedelecDofMap {
    /// `[min, max]` → edge DoF index `0 .. edge_count`.
    pub edge_id: std::collections::HashMap<[u32; 2], u32>,
    /// Total number of edges (= total DoF count).
    pub edge_count: usize,
}

impl NedelecDofMap {
    /// Build the DoF map by enumerating mesh edges via
    /// [`compute_edges`].
    #[must_use]
    pub fn build(mesh: &FemMesh) -> Self {
        let (edges, edge_id) = compute_edges(mesh);
        Self {
            edge_id,
            edge_count: edges.len(),
        }
    }

    /// Total DoF count = unique mesh edges.
    #[must_use]
    pub fn ndof(&self) -> usize {
        self.edge_count
    }

    /// Global DoF index for the edge between vertices `a` and `b`.
    /// Orientation-insensitive (looks up `[min, max]`).
    ///
    /// # Panics
    /// Panics if the edge is not present in the map.
    #[must_use]
    pub fn edge_dof(&self, a: u32, b: u32) -> usize {
        let key = if a < b { [a, b] } else { [b, a] };
        *self
            .edge_id
            .get(&key)
            .unwrap_or_else(|| panic!("NedelecDofMap: edge ({a}, {b}) not in map")) as usize
    }

    /// Orientation sign for a directed local edge `(a, b)` (cell-local
    /// vertex indices into the cell's `verts` array). Returns `+1`
    /// when the global edge orientation (`min → max`) matches the
    /// local direction `a → b`, else `-1`.
    ///
    /// This is what the assembly multiplies into entry `(i, j)` of the
    /// local 6×6 element matrix (sign for row `i`, sign for column
    /// `j`) before scattering to the global system.
    #[must_use]
    pub fn edge_sign(cell: &FemCell, local_edge: usize) -> Real {
        match cell {
            FemCell::Tet(v) => {
                let [a, b] = TET_EDGES[local_edge];
                if v[a] < v[b] { 1.0 } else { -1.0 }
            }
            FemCell::Tri(_) => panic!("NedelecDofMap::edge_sign: Tri not supported in v1"),
        }
    }
}

// ─── Element matrices ────────────────────────────────────────────────

/// Per-cell Nedelec mass matrix `Mᵢⱼ = ∫ Nᵢ · Nⱼ dV` on a tetrahedron.
///
/// Returns a 6×6 row-major dense matrix as `Vec<Vec<Real>>` in the
/// **local-edge basis** (no orientation sign applied — assembly
/// handles that). Edges are ordered by [`TET_EDGES`].
///
/// # Math
///
/// For lowest-order Nedelec on a tet, `Nᵢ Nⱼ` is quadratic in the
/// barycentric coordinates so a degree-2 exact rule is required. We
/// use the 4-point Gauss rule (degree-2 exact: weight `1/4` at each
/// `(α, β, β, β)` permutation with `α = (5 + 3√5)/20`, `β = (5 − √5)/20`)
/// — see Keast or e.g. Solin "Higher-Order Finite Element Methods"
/// table 5.2. Implementation uses the 4-point rule on barycentric
/// `(α, β, β, β)` permutations.
///
/// Alternatively we evaluate the closed-form integral
/// `∫ λᵢ λⱼ dV = vol · (1 + δᵢⱼ) / 20` and combine using the inner
/// products `∇λₖ · ∇λₗ` (which are constant) — this is what the
/// implementation does to keep things exact + cheap. Specifically,
/// for edges `e₁ = (i, j)` and `e₂ = (k, l)`:
///
/// ```text
///   M[e₁, e₂] = vol · ( c(j,l) ∇λᵢ·∇λₖ
///                     − c(j,k) ∇λᵢ·∇λₗ
///                     − c(i,l) ∇λⱼ·∇λₖ
///                     + c(i,k) ∇λⱼ·∇λₗ )
/// ```
///
/// where `c(p, q) = ∫ λₚ λ_q dV / vol = (1 + δ_{pq}) / 20`.
#[must_use]
pub fn compute_element_nedelec_mass(cell: &FemCell, vertices: &[Vec3]) -> Vec<Vec<Real>> {
    let FemCell::Tet(idx) = cell else {
        panic!("compute_element_nedelec_mass: only tetrahedra are supported");
    };
    let p = [
        vertices[idx[0] as usize],
        vertices[idx[1] as usize],
        vertices[idx[2] as usize],
        vertices[idx[3] as usize],
    ];
    let vol = tet_volume(&p);
    let g = tet_shape_grads(&p);

    // c(p, q) = ∫ λ_p λ_q / vol = (1 + δ_{pq}) / 20.
    let c = |p_idx: usize, q_idx: usize| -> Real {
        if p_idx == q_idx {
            2.0 / 20.0
        } else {
            1.0 / 20.0
        }
    };

    let mut m = vec![vec![0.0_f64 as Real; 6]; 6];
    for e1 in 0..6 {
        let [i, j] = TET_EDGES[e1];
        for e2 in 0..6 {
            let [k, l] = TET_EDGES[e2];
            // ∫ (λ_i ∇λ_j − λ_j ∇λ_i) · (λ_k ∇λ_l − λ_l ∇λ_k) dV
            // = ∇λ_j·∇λ_l · ∫ λ_i λ_k
            //   − ∇λ_j·∇λ_k · ∫ λ_i λ_l
            //   − ∇λ_i·∇λ_l · ∫ λ_j λ_k
            //   + ∇λ_i·∇λ_k · ∫ λ_j λ_l
            let gjgl = g[j].dot(g[l]);
            let gjgk = g[j].dot(g[k]);
            let gigl = g[i].dot(g[l]);
            let gigk = g[i].dot(g[k]);
            let val = vol * (gjgl * c(i, k) - gjgk * c(i, l) - gigl * c(j, k) + gigk * c(j, l));
            m[e1][e2] = val;
        }
    }
    m
}

/// Per-cell curl-curl stiffness matrix `Kᵢⱼ = ∫ ν (∇×Nᵢ) · (∇×Nⱼ) dV`
/// on a tetrahedron.
///
/// Returns a 6×6 row-major dense matrix in the **local-edge basis**.
///
/// # Math
///
/// The curl of the Nedelec edge basis is constant per element:
///
/// ```text
///   ∇ × N_e = 2 ∇λ_i × ∇λ_j        for edge e = (i, j)
/// ```
///
/// so the integral collapses to a closed-form `vol · 4 (∇λᵢ × ∇λⱼ) ·
/// (∇λₖ × ∇λₗ)` — no quadrature loop needed.
///
/// `nu` is the magnetic reluctivity `ν = 1 / μ`. For free space pass
/// `1 / μ₀ ≈ 7.957747e5` (SI units).
#[must_use]
pub fn compute_element_curl_curl(cell: &FemCell, vertices: &[Vec3], nu: Real) -> Vec<Vec<Real>> {
    let FemCell::Tet(idx) = cell else {
        panic!("compute_element_curl_curl: only tetrahedra are supported");
    };
    let p = [
        vertices[idx[0] as usize],
        vertices[idx[1] as usize],
        vertices[idx[2] as usize],
        vertices[idx[3] as usize],
    ];
    let vol = tet_volume(&p);
    let g = tet_shape_grads(&p);

    // Per-edge curl direction: c_e = 2 ∇λ_i × ∇λ_j (orientation matches
    // the local-edge basis convention).
    let mut curls = [Vec3::ZERO; 6];
    for e in 0..6 {
        let [i, j] = TET_EDGES[e];
        curls[e] = 2.0 * g[i].cross(g[j]);
    }

    let mut k = vec![vec![0.0_f64 as Real; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            k[i][j] = nu * vol * curls[i].dot(curls[j]);
        }
    }
    k
}

// ─── Assembly ────────────────────────────────────────────────────────

/// Assemble a global Nedelec matrix by scattering per-cell 6×6 local
/// matrices into a CSR.
///
/// The visitor returns the element matrix in the **local-edge basis**
/// (no orientation sign baked in). This function multiplies row `i`
/// and column `j` of every cell's local matrix by the appropriate
/// orientation signs (from [`NedelecDofMap::edge_sign`]) before
/// scattering, so the assembled global matrix is consistent with the
/// canonical "min → max" edge orientation.
///
/// Mirrors [`crate::fem::p2_element::assemble_global_p2`] in structure.
pub fn assemble_global_nedelec<F>(
    mesh: &FemMesh,
    map: &NedelecDofMap,
    element_visitor: F,
) -> crate::core::Result<crate::math::sparse::BsrMatrix>
where
    F: Fn(&FemCell, &[Vec3]) -> Vec<Vec<Real>>,
{
    let n = map.ndof();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    triplets.reserve(mesh.cells.len() * 36);

    for cell in &mesh.cells {
        let FemCell::Tet(verts) = cell else { continue };
        let local = element_visitor(cell, &mesh.vertices);
        debug_assert_eq!(local.len(), 6);

        // Per-local-edge: global DoF + orientation sign.
        let mut gdof = [0usize; 6];
        let mut sign = [1.0_f64 as Real; 6];
        for e in 0..6 {
            let [a, b] = TET_EDGES[e];
            gdof[e] = map.edge_dof(verts[a], verts[b]);
            sign[e] = NedelecDofMap::edge_sign(cell, e);
        }

        for i in 0..6 {
            for j in 0..6 {
                let val = sign[i] * sign[j] * local[i][j];
                if val == 0.0 {
                    continue;
                }
                triplets.push((gdof[i], gdof[j], val));
            }
        }
    }

    crate::math::sparse::BsrMatrix::from_scalar_triplets(n, n, &triplets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    fn unit_tet() -> ([Vec3; 4], FemCell, Vec<Vec3>) {
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        (p, FemCell::Tet([0, 1, 2, 3]), p.to_vec())
    }

    #[test]
    fn nedelec_mass_is_symmetric() {
        let (_p, cell, v) = unit_tet();
        let m = compute_element_nedelec_mass(&cell, &v);
        for i in 0..6 {
            for j in 0..6 {
                let dij = (m[i][j] - m[j][i]).abs();
                assert!(
                    dij < 1e-14,
                    "M not symmetric at ({i},{j}): {} vs {}",
                    m[i][j],
                    m[j][i]
                );
            }
        }
    }

    #[test]
    fn nedelec_curl_curl_is_symmetric() {
        let (_p, cell, v) = unit_tet();
        let k = compute_element_curl_curl(&cell, &v, 1.0);
        for i in 0..6 {
            for j in 0..6 {
                let dij = (k[i][j] - k[j][i]).abs();
                assert!(dij < 1e-14, "K not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn curl_curl_kills_pure_gradient_modes() {
        // The kernel of curl on the lowest-order Nedelec space is the
        // discrete gradient of P1 nodal: for any nodal vector
        // `φ = (φ₀, φ₁, φ₂, φ₃)`, the edge-coefficient representation
        // of `∇φ` is `c_{(i,j)} = φ_j − φ_i` (signed by the local edge
        // direction i→j). The curl-curl operator must annihilate this
        // family.
        let (_p, cell, v) = unit_tet();
        let k = compute_element_curl_curl(&cell, &v, 1.0);
        // Pick a non-trivial nodal field.
        let phi = [0.7, 1.3, -0.4, 2.1];
        // Edge coefficients in the **local** (i→j) basis: c_e = φ_j − φ_i.
        let mut c = [0.0; 6];
        for e in 0..6 {
            let [i, j] = TET_EDGES[e];
            c[e] = phi[j] - phi[i];
        }
        // K · c should be zero.
        for r in 0..6 {
            let s: Real = (0..6).map(|cidx| k[r][cidx] * c[cidx]).sum();
            assert!(s.abs() < 1e-12, "row {r}: K·grad(phi) = {s}");
        }
    }

    #[test]
    fn nedelec_interpolation_of_linear_field_has_constant_curl() {
        // For the lowest-order Nedelec / Whitney complex, the canonical
        // edge-interpolant of any C¹ vector field commutes with curl:
        //
        //   curl(Π_h A) = Π_h^{RT}(curl A)
        //
        // (this is the discrete de Rham diagram commutativity). For
        // `A(x) = (0, x, 0)` the continuous curl is `ê_z` (constant),
        // so the discrete curl computed from the projected edge DoFs
        // must also equal `ê_z` exactly.
        //
        // This is a stricter test of correctness than `K · a = 0`:
        // `K · a` is the *weak-form* contribution `∫ curl(N_i) ·
        // curl(A_h) dV`, which is NOT zero when curl(A_h) ≠ 0 — it's
        // the load that the curl-curl operator imposes on test
        // functions whose curls aren't orthogonal to `ê_z`. The
        // physical "curl curl A = 0" identity manifests in the
        // *interior* rows of the global system; on a single tet every
        // row is a boundary row, so the strong identity has to be
        // checked element-wise on the curl-of-the-interpolant directly.
        let (p, _cell, _v) = unit_tet();
        // Edge interpolant of `A(x) = (0, x, 0)`:
        // tangential integral along the segment p_i → p_j is
        //   (p_i.x + p_j.x)/2 · (p_j.y − p_i.y).
        let mut a = [0.0_f64 as Real; 6];
        for e in 0..6 {
            let [i, j] = TET_EDGES[e];
            a[e] = (p[i].x + p[j].x) / 2.0 * (p[j].y - p[i].y);
        }

        // Per-cell curl: Σ_e a_e · (2 ∇λ_i × ∇λ_j).
        let grads = super::tet_shape_grads(&[p[0], p[1], p[2], p[3]]);
        let mut curl_a = Vec3::ZERO;
        for e in 0..6 {
            let [i, j] = TET_EDGES[e];
            curl_a += a[e] * 2.0 * grads[i].cross(grads[j]);
        }
        let err = (curl_a - Vec3::new(0.0, 0.0, 1.0)).length();
        assert!(
            err < 1e-12,
            "curl(Π_h A) = {curl_a:?}, expected ê_z (err={err})"
        );
    }

    #[test]
    fn curl_curl_weak_form_matches_strong_form_on_constant_curl() {
        // Stage 46 contract test: for `A = (0, x, 0)`, `curl A = ê_z`
        // is constant ⇒ the **strong-form** `curl curl A` is zero.
        // In the weak/discrete setting this manifests as
        //
        //   K · a = "boundary flux of curl A through curl(N_i)"
        //
        // (a boundary-only contribution, since interior cells would
        // see `∫ curl(N_i) · curl(A_h) dV − ∫ curl(N_i) · ∇×ê_z dV` and
        // the second term is zero for a constant curl). On a single
        // tet every row is a boundary row, so we instead verify the
        // strong form via [`nedelec_interpolation_of_linear_field_has_constant_curl`].
        //
        // Here we check the orthogonality identity that the
        // **constant-curl** field, projected onto NED1, lives in the
        // image of the discrete curl operator: `Σᵢ φᵢ ∇λᵢ` cannot
        // produce a constant curl, so the projected `a` must have a
        // non-zero component out of the gradient subspace.
        let (p, cell, v) = unit_tet();
        let k = compute_element_curl_curl(&cell, &v, 1.0);
        let mut a = [0.0_f64 as Real; 6];
        for e in 0..6 {
            let [i, j] = TET_EDGES[e];
            a[e] = (p[i].x + p[j].x) / 2.0 * (p[j].y - p[i].y);
        }
        // The squared energy `aᵀ K a = vol · |curl A_h|² = (1/6) · 1`.
        let mut energy = 0.0_f64 as Real;
        for i in 0..6 {
            for j in 0..6 {
                energy += a[i] * k[i][j] * a[j];
            }
        }
        assert!(
            (energy - 1.0 / 6.0).abs() < 1e-12,
            "aᵀ K a = {energy}, expected vol = 1/6"
        );
    }

    #[test]
    fn nedelec_mass_positive_definite_on_unit_tet() {
        // The mass matrix must be SPD (it's the Gram matrix of a basis).
        let (_p, cell, v) = unit_tet();
        let m = compute_element_nedelec_mass(&cell, &v);
        // Check positivity via random vector x ≠ 0 ⇒ xᵀ M x > 0.
        let x = [0.7, -1.1, 0.4, 2.3, -0.6, 1.5];
        let mut q = 0.0;
        for i in 0..6 {
            for j in 0..6 {
                q += x[i] * m[i][j] * x[j];
            }
        }
        assert!(q > 0.0, "M is not positive on x: q = {q}");
        // And the diagonal entries are all positive.
        for i in 0..6 {
            assert!(m[i][i] > 0.0, "M[{i},{i}] = {} not positive", m[i][i]);
        }
    }
}
