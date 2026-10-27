//! P2 (quadratic-nodal) basis functions + local element matrices.
//!
//! Stage 36.3 adds piecewise-quadratic elements on top of the v1 P1
//! framework. A P2 triangle has 6 nodes — 3 corners + 3 edge midpoints
//! — and a P2 tet has 10 nodes — 4 corners + 6 edge midpoints. The
//! reference shape functions in barycentric coordinates are:
//!
//! - **Corners** (`λ_i (2λ_i − 1)`) — value 1 at corner `i`, 0 at the
//!   other 5 (tri) / 9 (tet) reference points.
//! - **Edges**   (`4 λ_i λ_j`) — value 1 at the midpoint of edge
//!   `(i, j)`, 0 elsewhere.
//!
//! Edge ordering is fixed for the rest of the pipeline (assembly +
//! `P2DofMap`):
//!
//! - **Triangle edges** (3): `(0,1), (1,2), (2,0)`.
//! - **Tetrahedron edges** (6): `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`.
//!
//! Element matrices integrate `φᵢ · φⱼ` (mass) and `∇φᵢ · ∇φⱼ`
//! (stiffness) numerically using the higher-order quadrature rules
//! from [`crate::fem::quadrature`]. P2 mass needs degree-4 exact
//! quadrature; P2 stiffness needs degree-2 (gradients are linear).

use crate::math::{Real, Vec3};

use super::element::{tet_shape_grads, tri_shape_grads_2d};
use super::mesh::FemCell;
use super::quadrature::{TetQuadPoint, TriQuadPoint, tet_quadrature_11pt, tri_quadrature_6pt};

// ─── Edge orderings ──────────────────────────────────────────────────

/// Triangle edge ordering — `[(corner_a, corner_b), …]`.
///
/// Edge node `e_k` lives between corners `TRI_EDGES[k][0]` and
/// `TRI_EDGES[k][1]`. The local DoF layout for a P2 triangle is
/// `[c0, c1, c2, e01, e12, e20]`.
pub const TRI_EDGES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

/// Tetrahedron edge ordering — `[(corner_a, corner_b), …]`.
///
/// Edge node `e_k` lives between corners `TET_EDGES[k][0]` and
/// `TET_EDGES[k][1]`. The local DoF layout for a P2 tet is
/// `[c0, c1, c2, c3, e01, e02, e03, e12, e13, e23]`.
pub const TET_EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

// ─── Reference shape functions ────────────────────────────────────────

/// 6 P2 shape functions evaluated at a barycentric point on the
/// reference triangle.
///
/// Returns `[c0, c1, c2, e01, e12, e20]` where:
/// - `cᵢ = λᵢ (2λᵢ − 1)` (value 1 at corner `i`, 0 at the others).
/// - `eᵢⱼ = 4 λᵢ λⱼ`     (value 1 at the midpoint of edge `(i, j)`).
#[must_use]
pub fn p2_tri_shape_functions(bary: [Real; 3]) -> [Real; 6] {
    let l = bary;
    [
        l[0] * (2.0 * l[0] - 1.0),
        l[1] * (2.0 * l[1] - 1.0),
        l[2] * (2.0 * l[2] - 1.0),
        4.0 * l[0] * l[1],
        4.0 * l[1] * l[2],
        4.0 * l[2] * l[0],
    ]
}

/// 10 P2 shape functions evaluated at a barycentric point on the
/// reference tetrahedron.
///
/// Returns `[c0, c1, c2, c3, e01, e02, e03, e12, e13, e23]` using the
/// edge ordering [`TET_EDGES`].
#[must_use]
pub fn p2_tet_shape_functions(bary: [Real; 4]) -> [Real; 10] {
    let l = bary;
    [
        l[0] * (2.0 * l[0] - 1.0),
        l[1] * (2.0 * l[1] - 1.0),
        l[2] * (2.0 * l[2] - 1.0),
        l[3] * (2.0 * l[3] - 1.0),
        4.0 * l[0] * l[1],
        4.0 * l[0] * l[2],
        4.0 * l[0] * l[3],
        4.0 * l[1] * l[2],
        4.0 * l[1] * l[3],
        4.0 * l[2] * l[3],
    ]
}

// ─── Physical-space gradients ─────────────────────────────────────────
//
// P2 shape-function gradients in physical space follow from the chain
// rule: ∇N(λ) = ∑ₖ (∂N / ∂λₖ) ∇λₖ. The barycentric gradients ∇λₖ are
// the P1 hat-function gradients (constant per element) computed by
// `tri_shape_grads_2d` / `tet_shape_grads` from the v1 element
// module. We only need to differentiate the polynomial in λ and dot
// with that constant Jacobian-row.

/// Gradients of the 6 P2 triangle shape functions in physical space,
/// evaluated at barycentric point `bary` for a triangle with the
/// given `vertices` (3D, but only `(x, y)` components are used).
///
/// Returns `[Vec3; 6]` — the `z`-component is always zero by
/// construction (planar triangle in `z = 0`).
#[must_use]
pub fn p2_tri_shape_gradients(bary: [Real; 3], vertices: &[Vec3; 3]) -> [Vec3; 6] {
    let dl = tri_shape_grads_2d(vertices);
    let g_lambda = [
        Vec3::new(dl[0][0], dl[0][1], 0.0),
        Vec3::new(dl[1][0], dl[1][1], 0.0),
        Vec3::new(dl[2][0], dl[2][1], 0.0),
    ];
    p2_tri_grad_from_bary_grads(bary, &g_lambda)
}

/// Gradients of the 10 P2 tetrahedron shape functions in physical
/// space, evaluated at barycentric point `bary` for a tet with the
/// given `vertices`.
#[must_use]
pub fn p2_tet_shape_gradients(bary: [Real; 4], vertices: &[Vec3; 4]) -> [Vec3; 10] {
    let g_lambda = tet_shape_grads(vertices);
    p2_tet_grad_from_bary_grads(bary, &g_lambda)
}

/// Internal: chain-rule the P2 triangle shape gradients given the
/// barycentric-coordinate gradients.
fn p2_tri_grad_from_bary_grads(bary: [Real; 3], g_lambda: &[Vec3; 3]) -> [Vec3; 6] {
    let l = bary;
    // Corner gradients: ∇N_i = (4λᵢ - 1) ∇λᵢ.
    let c0 = g_lambda[0] * (4.0 * l[0] - 1.0);
    let c1 = g_lambda[1] * (4.0 * l[1] - 1.0);
    let c2 = g_lambda[2] * (4.0 * l[2] - 1.0);
    // Edge gradients: ∇N_ij = 4(λⱼ ∇λᵢ + λᵢ ∇λⱼ).
    let e01 = (g_lambda[0] * l[1] + g_lambda[1] * l[0]) * 4.0;
    let e12 = (g_lambda[1] * l[2] + g_lambda[2] * l[1]) * 4.0;
    let e20 = (g_lambda[2] * l[0] + g_lambda[0] * l[2]) * 4.0;
    [c0, c1, c2, e01, e12, e20]
}

/// Internal: chain-rule the P2 tet shape gradients.
fn p2_tet_grad_from_bary_grads(bary: [Real; 4], g_lambda: &[Vec3; 4]) -> [Vec3; 10] {
    let l = bary;
    let c0 = g_lambda[0] * (4.0 * l[0] - 1.0);
    let c1 = g_lambda[1] * (4.0 * l[1] - 1.0);
    let c2 = g_lambda[2] * (4.0 * l[2] - 1.0);
    let c3 = g_lambda[3] * (4.0 * l[3] - 1.0);
    let e01 = (g_lambda[0] * l[1] + g_lambda[1] * l[0]) * 4.0;
    let e02 = (g_lambda[0] * l[2] + g_lambda[2] * l[0]) * 4.0;
    let e03 = (g_lambda[0] * l[3] + g_lambda[3] * l[0]) * 4.0;
    let e12 = (g_lambda[1] * l[2] + g_lambda[2] * l[1]) * 4.0;
    let e13 = (g_lambda[1] * l[3] + g_lambda[3] * l[1]) * 4.0;
    let e23 = (g_lambda[2] * l[3] + g_lambda[3] * l[2]) * 4.0;
    [c0, c1, c2, c3, e01, e02, e03, e12, e13, e23]
}

// ─── DoF mapping ─────────────────────────────────────────────────────

/// DoF map for the `P2Nodal` function space.
///
/// Layout of the global solution vector:
/// 1. The first `mesh.vert_count()` entries are the corner DoFs —
///    DoF index `i` ↔ vertex `i`.
/// 2. The next `edge_count` entries are the edge-midpoint DoFs in
///    the order returned by [`crate::fem::function_space::compute_edges`].
///
/// The map provides cheap lookups from a cell's local node index
/// (`0..N` where `N = 6` (tri) or `N = 10` (tet)) to its global DoF.
#[derive(Debug, Clone)]
pub struct P2DofMap {
    /// Number of corner (vertex) DoFs.
    pub vertex_count: usize,
    /// `edge_id[(min, max)]` → edge DoF offset (relative to
    /// `vertex_count`). The lookup is `O(1)`.
    pub edge_id: std::collections::HashMap<[u32; 2], u32>,
}

impl P2DofMap {
    /// Build the DoF map by enumerating the mesh edges via
    /// [`crate::fem::function_space::compute_edges`].
    #[must_use]
    pub fn build(mesh: &super::mesh::FemMesh) -> Self {
        let (_, edge_id) = super::function_space::compute_edges(mesh);
        Self {
            vertex_count: mesh.vert_count(),
            edge_id,
        }
    }

    /// Total DoF count = `vertex_count + edge_count`.
    #[must_use]
    pub fn ndof(&self) -> usize {
        self.vertex_count + self.edge_id.len()
    }

    /// Lookup the global DoF for the edge between corner indices
    /// `a` and `b` (orientation-insensitive). Returns
    /// `vertex_count + edge_id` so corner DoFs and edge DoFs share
    /// one address space.
    ///
    /// # Panics
    /// Panics if the edge `(a, b)` is not present in this map's mesh —
    /// indicates a bug in the caller (the mesh used to build the map
    /// should contain the cell whose edge is being looked up).
    #[must_use]
    pub fn edge_dof(&self, a: u32, b: u32) -> usize {
        let key = if a < b { [a, b] } else { [b, a] };
        let id = *self
            .edge_id
            .get(&key)
            .unwrap_or_else(|| panic!("P2DofMap: edge ({a}, {b}) not in map"));
        self.vertex_count + id as usize
    }

    /// Resolve the global DoF for a cell's local node index.
    /// `local_idx` is `0..6` for a tri or `0..10` for a tet,
    /// following the layouts:
    /// - Tri: `[c0, c1, c2, e01, e12, e20]`.
    /// - Tet: `[c0, c1, c2, c3, e01, e02, e03, e12, e13, e23]`.
    #[must_use]
    pub fn local_to_global(&self, cell: &FemCell, local_idx: usize) -> usize {
        match cell {
            FemCell::Tri(v) => {
                if local_idx < 3 {
                    v[local_idx] as usize
                } else {
                    let edge = TRI_EDGES[local_idx - 3];
                    self.edge_dof(v[edge[0]], v[edge[1]])
                }
            }
            FemCell::Tet(v) => {
                if local_idx < 4 {
                    v[local_idx] as usize
                } else {
                    let edge = TET_EDGES[local_idx - 4];
                    self.edge_dof(v[edge[0]], v[edge[1]])
                }
            }
        }
    }
}

// ─── Element matrices ────────────────────────────────────────────────

/// Per-cell mass matrix `Mₑᵢⱼ = ∫ Nᵢ Nⱼ dx` for P2 elements.
///
/// Returned as an `N × N` row-major `Vec<Vec<Real>>` where `N = 6`
/// (Tri) or `N = 10` (Tet). Triangles project to `(x, y)`; tets use
/// the full 3D positions. Quadrature: 6-point Dunavant (tri),
/// 11-point Keast (tet) — both degree-4 exact and so capture the
/// quartic `Nᵢ Nⱼ` integrand.
#[must_use]
pub fn compute_element_p2_mass(cell: &FemCell, vertices: &[Vec3]) -> Vec<Vec<Real>> {
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = super::element::tri_area_2d(&p);
            let qpts: Vec<TriQuadPoint> = tri_quadrature_6pt();
            let mut m = vec![vec![0.0_f64 as Real; 6]; 6];
            for q in &qpts {
                let n = p2_tri_shape_functions(q.bary);
                let w = q.weight * area;
                for i in 0..6 {
                    for j in 0..6 {
                        m[i][j] += w * n[i] * n[j];
                    }
                }
            }
            m
        }
        FemCell::Tet(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
                vertices[idx[3] as usize],
            ];
            let vol = super::element::tet_volume(&p);
            let qpts: Vec<TetQuadPoint> = tet_quadrature_11pt();
            let mut m = vec![vec![0.0_f64 as Real; 10]; 10];
            for q in &qpts {
                let n = p2_tet_shape_functions(q.bary);
                let w = q.weight * vol;
                for i in 0..10 {
                    for j in 0..10 {
                        m[i][j] += w * n[i] * n[j];
                    }
                }
            }
            m
        }
    }
}

/// Per-cell stiffness matrix `Kₑᵢⱼ = coeff · ∫ ∇Nᵢ · ∇Nⱼ dx` for P2
/// elements.
///
/// Returned as an `N × N` row-major `Vec<Vec<Real>>` (`N = 6` tri,
/// `N = 10` tet). Quadrature: same higher-order rules as the mass
/// (degree-4 exact is overkill for the degree-2 stiffness integrand
/// but keeps the kernel simple — one quadrature pass per cell shared
/// with the mass).
#[must_use]
pub fn compute_element_p2_stiffness(
    cell: &FemCell,
    vertices: &[Vec3],
    coeff: Real,
) -> Vec<Vec<Real>> {
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = super::element::tri_area_2d(&p);
            let qpts: Vec<TriQuadPoint> = tri_quadrature_6pt();
            let mut k = vec![vec![0.0_f64 as Real; 6]; 6];
            for q in &qpts {
                let g = p2_tri_shape_gradients(q.bary, &p);
                let w = q.weight * area * coeff;
                for i in 0..6 {
                    for j in 0..6 {
                        // 2D dot product (z = 0 for planar tris).
                        k[i][j] += w * (g[i].x * g[j].x + g[i].y * g[j].y);
                    }
                }
            }
            k
        }
        FemCell::Tet(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
                vertices[idx[3] as usize],
            ];
            let vol = super::element::tet_volume(&p);
            let qpts: Vec<TetQuadPoint> = tet_quadrature_11pt();
            let mut k = vec![vec![0.0_f64 as Real; 10]; 10];
            for q in &qpts {
                let g = p2_tet_shape_gradients(q.bary, &p);
                let w = q.weight * vol * coeff;
                for i in 0..10 {
                    for j in 0..10 {
                        k[i][j] += w * g[i].dot(g[j]);
                    }
                }
            }
            k
        }
    }
}

// ─── Assembly ────────────────────────────────────────────────────────

/// Assemble a global P2 matrix by scattering per-cell local matrices
/// (6×6 for Tri, 10×10 for Tet) into a CSR.
///
/// Mirrors [`crate::fem::assembly::assemble_global`] but uses
/// [`P2DofMap`] to translate local node indices → global DoFs (corner
/// DoFs come first, edge DoFs next).
pub fn assemble_global_p2<F>(
    mesh: &super::mesh::FemMesh,
    map: &P2DofMap,
    element_visitor: F,
) -> crate::core::Result<crate::math::sparse::BsrMatrix>
where
    F: Fn(&FemCell, &[Vec3]) -> Vec<Vec<Real>>,
{
    let n = map.ndof();
    // Flat scalar triplets — see `assemble_global` for the rationale.
    let mut triplets: Vec<(usize, usize, Real)> = Vec::with_capacity(mesh.cells.len() * 100);
    for cell in &mesh.cells {
        let local = element_visitor(cell, &mesh.vertices);
        let nl = local.len();
        for i in 0..nl {
            let gi = map.local_to_global(cell, i);
            for j in 0..nl {
                let gj = map.local_to_global(cell, j);
                let val = local[i][j];
                if val == 0.0 {
                    continue;
                }
                triplets.push((gi, gj, val));
            }
        }
    }
    crate::math::sparse::BsrMatrix::from_scalar_triplets(n, n, &triplets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tri() -> [Vec3; 3] {
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]
    }

    fn unit_tet() -> [Vec3; 4] {
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }

    #[test]
    fn p2_tri_partition_of_unity_at_centroid() {
        let n = p2_tri_shape_functions([1.0 / 3.0; 3]);
        let s: Real = n.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "sum = {s}");
    }

    #[test]
    fn p2_tri_corner_kronecker() {
        // At corner 0, c0 = 1, others = 0.
        let n = p2_tri_shape_functions([1.0, 0.0, 0.0]);
        assert!((n[0] - 1.0).abs() < 1e-12);
        for k in 1..6 {
            assert!(n[k].abs() < 1e-12, "n[{k}] = {}", n[k]);
        }
    }

    #[test]
    fn p2_tet_corner_kronecker() {
        let n = p2_tet_shape_functions([0.0, 1.0, 0.0, 0.0]);
        assert!((n[1] - 1.0).abs() < 1e-12);
        for k in 0..10 {
            if k == 1 {
                continue;
            }
            assert!(n[k].abs() < 1e-12, "n[{k}] = {}", n[k]);
        }
    }

    #[test]
    fn p2_tri_grad_partition_of_zero_at_centroid() {
        let g = p2_tri_shape_gradients([1.0 / 3.0; 3], &unit_tri());
        let s = g.iter().fold(Vec3::ZERO, |a, b| a + *b);
        assert!(s.length() < 1e-12);
    }

    #[test]
    fn p2_tet_grad_partition_of_zero_at_centroid() {
        let g = p2_tet_shape_gradients([0.25; 4], &unit_tet());
        let s = g.iter().fold(Vec3::ZERO, |a, b| a + *b);
        assert!(s.length() < 1e-12);
    }

    #[test]
    fn p2_tri_mass_total_equals_area() {
        // ∑ᵢⱼ Mᵢⱼ = ∫(∑ᵢ Nᵢ)(∑ⱼ Nⱼ) = area (partition of unity).
        let cell = FemCell::Tri([0, 1, 2]);
        let m = compute_element_p2_mass(&cell, &unit_tri());
        let total: Real = m.iter().flatten().sum();
        // Unit triangle area = 0.5.
        assert!((total - 0.5).abs() < 1e-10, "total = {total}");
    }

    #[test]
    fn p2_tet_mass_total_equals_volume() {
        let cell = FemCell::Tet([0, 1, 2, 3]);
        let m = compute_element_p2_mass(&cell, &unit_tet());
        let total: Real = m.iter().flatten().sum();
        // Unit tet volume = 1/6.
        assert!((total - 1.0 / 6.0).abs() < 1e-10, "total = {total}");
    }

    #[test]
    fn p2_tri_stiffness_kills_constant_field() {
        // For a constant nodal field, ∇φ ≡ 0 at every quad point ⇒
        // Σⱼ Kᵢⱼ · 1 = 0 (sum each row).
        let cell = FemCell::Tri([0, 1, 2]);
        let k = compute_element_p2_stiffness(&cell, &unit_tri(), 1.0);
        for row in &k {
            let s: Real = row.iter().sum();
            assert!(s.abs() < 1e-10, "row sum = {s}");
        }
    }

    #[test]
    fn p2_tet_stiffness_kills_constant_field() {
        let cell = FemCell::Tet([0, 1, 2, 3]);
        let k = compute_element_p2_stiffness(&cell, &unit_tet(), 1.0);
        for row in &k {
            let s: Real = row.iter().sum();
            assert!(s.abs() < 1e-10, "row sum = {s}");
        }
    }
}
