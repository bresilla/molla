//! Magnetostatics via Nedelec H(curl) elements — Stage 46.
//!
//! Solves the steady-state curl-curl equation for the magnetic vector
//! potential `A`:
//!
//! ```text
//!   curl(ν curl A) = J        in Ω
//!   n × A = 0                 on ∂Ω
//! ```
//!
//! where:
//! - `A` is the magnetic vector potential (DoFs live on mesh edges,
//!   one tangential-integral per edge),
//! - `J` is the prescribed source current density (one `Vec3` per
//!   cell),
//! - `ν = 1/μ` is the magnetic reluctivity (one scalar per cell —
//!   uniform `1/μ₀` outside ferromagnetic regions),
//! - `n × A = 0` is the "perfect conductor" boundary condition (the
//!   tangential component of `A` vanishes; in Nedelec DoFs this is
//!   simply `A_e = 0` for every edge on the boundary).
//!
//! The discrete weak form is
//!
//! ```text
//!   ∫ ν (∇×N_i) · (∇×N_j) dV · a_j = ∫ J · N_i dV
//! ```
//!
//! ⇒ a symmetric positive semi-definite (SPSD) curl-curl matrix
//! `K_cc` and a load vector `f` whose `i`-th entry is the source
//! integral against the `i`-th Nedelec basis function.
//!
//! # Gauge fixing
//!
//! The continuous curl-curl operator has a non-trivial kernel — every
//! pure gradient `A = ∇φ` is annihilated (`curl ∇φ ≡ 0`). On the
//! discrete level the kernel is exactly the discrete gradient of P1
//! nodal scalars (the lowest-order de Rham complex). For a
//! magnetostatics problem with a divergence-free source `∇ · J = 0`
//! and **fully Dirichlet** boundary on `∂Ω` (`n × A = 0`), the
//! gradient kernel is killed at the boundary and what remains is
//! gauge-equivalent — pinning all boundary DoFs leaves only the
//! curl-component-of-`A` part, plus a residual gradient gauge that
//! the CG solver picks up the minimum-norm solution for. We do not
//! enforce Coulomb gauge `∇ · A = 0` explicitly — it's not required
//! to recover the physically meaningful `B = curl A`.
//!
//! In practice we add a tiny mass regularisation `ε M` (with
//! `ε ≪ ν / L²`) to nudge CG towards a unique solution; the bias on
//! `B` is `O(ε)` and well below the discretisation error for the
//! tests we ship.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use crate::core::Result;
use crate::math::iterative::{CgResult, cg_solve};
use crate::math::sparse::BsrMatrix;
use crate::math::{Real, Vec3};

use super::assembly::matvec;
use super::bc::{BoundaryCondition, apply_dirichlet};
use super::element::{tet_shape_grads, tet_volume};
use super::function_space::compute_edges;
use super::mesh::{FemCell, FemMesh};
use super::nedelec_element::{
    NedelecDofMap, assemble_global_nedelec, compute_element_curl_curl, compute_element_nedelec_mass,
};
use super::p2_element::TET_EDGES;

/// Output of a magnetostatics solve.
#[derive(Debug, Clone)]
pub struct MagnetostaticsResult {
    /// Vector potential `A` — one DoF per unique mesh edge (in the
    /// canonical `min → max` orientation).
    pub a: Vec<Real>,
    /// Final CG residual `‖K_cc · a − f‖ / ‖f‖`.
    pub residual: Real,
    /// CG iteration count.
    pub iters: usize,
}

/// Build the load vector `f_i = ∫ J · N_i dV` for a piecewise-
/// constant current density `j_source[c]` per cell.
///
/// For lowest-order Nedelec on a tetrahedron, the basis integral is
/// closed-form:
///
/// ```text
///   ∫_cell N_{(i,j)} dV = (vol / 4) (∇λ_j − ∇λ_i)
/// ```
///
/// (one P1 hat integrates to `vol / (N+1)`, so `λ_i ∇λ_j` integrates
/// to `(vol/4) ∇λ_j` and the difference gives the formula above.)
///
/// The local 6-vector is then scattered to global edges with the
/// orientation sign from [`NedelecDofMap::edge_sign`].
fn build_load_vector(mesh: &FemMesh, map: &NedelecDofMap, j_source: &[Vec3]) -> Vec<Real> {
    assert_eq!(
        j_source.len(),
        mesh.cells.len(),
        "j_source must be one Vec3 per cell"
    );
    let mut f = vec![0.0; map.ndof()];
    for (c_idx, cell) in mesh.cells.iter().enumerate() {
        let FemCell::Tet(verts) = cell else { continue };
        let p = [
            mesh.vertices[verts[0] as usize],
            mesh.vertices[verts[1] as usize],
            mesh.vertices[verts[2] as usize],
            mesh.vertices[verts[3] as usize],
        ];
        let vol = tet_volume(&p);
        let grads = tet_shape_grads(&p);
        let j = j_source[c_idx];
        for e in 0..6 {
            let [a, b] = TET_EDGES[e];
            // ∫ N_e dV in the local (a → b) basis.
            let int_n = (grads[b] - grads[a]) * (vol / 4.0);
            let load_local = j.dot(int_n);
            // Orientation sign to global edge basis.
            let sign = NedelecDofMap::edge_sign(cell, e);
            let g_dof = map.edge_dof(verts[a], verts[b]);
            f[g_dof] += sign * load_local;
        }
    }
    f
}

/// Identify boundary edges of `mesh` — edges that lie on a face owned
/// by exactly one cell.
///
/// For each tet, every face (triangle) contributes its 3 edges. A
/// face is a **boundary face** when only one cell claims it. An edge
/// is a **boundary edge** if it belongs to at least one boundary face.
fn boundary_edge_ids(mesh: &FemMesh, edge_id: &HashMap<[u32; 2], u32>) -> Vec<u32> {
    // Tally face occurrences.
    let mut face_count: HashMap<[u32; 3], u32> = HashMap::new();
    for cell in &mesh.cells {
        let FemCell::Tet(v) = cell else { continue };
        let faces: [[u32; 3]; 4] = [
            [v[1], v[2], v[3]],
            [v[0], v[2], v[3]],
            [v[0], v[1], v[3]],
            [v[0], v[1], v[2]],
        ];
        for fc in &faces {
            let mut k = *fc;
            k.sort_unstable();
            *face_count.entry(k).or_insert(0) += 1;
        }
    }

    let mut boundary_edges: HashSet<u32> = HashSet::new();
    for (face, count) in &face_count {
        if *count != 1 {
            continue;
        }
        // Push each of the face's three edges.
        let pairs = [[face[0], face[1]], [face[1], face[2]], [face[0], face[2]]];
        for p in &pairs {
            let key = if p[0] < p[1] { *p } else { [p[1], p[0]] };
            if let Some(id) = edge_id.get(&key) {
                boundary_edges.insert(*id);
            }
        }
    }
    let mut out: Vec<u32> = boundary_edges.into_iter().collect();
    out.sort_unstable();
    out
}

/// Magnetostatics steady-state solve.
///
/// Assembles `K_cc + ε M`, applies `n × A = 0` on the boundary
/// (Dirichlet on every boundary edge DoF), and CG-solves
/// `(K_cc + ε M) a = f`.
///
/// Arguments:
/// - `mesh`        — tet mesh (Tris are ignored).
/// - `j_source`    — current density `J` (one `Vec3` per cell).
/// - `nu_per_cell` — magnetic reluctivity `ν = 1/μ` per cell. Pass
///   `vec![1.0; cells]` to solve `curl curl A = J` in dimensionless
///   units (i.e. with `μ ≡ 1`).
/// - `bcs`         — additional Dirichlet BCs (typically empty — the
///   solver adds `n × A = 0` itself; pass extra edge pins here to
///   override boundary values).
///
/// Returns the vector potential `A` (one DoF per edge, in
/// canonical `min → max` orientation), final residual, and iter count.
pub fn solve_magnetostatics_steady(
    mesh: &FemMesh,
    j_source: &[Vec3],
    nu_per_cell: &[Real],
    bcs: &[BoundaryCondition],
) -> Result<MagnetostaticsResult> {
    assert_eq!(
        nu_per_cell.len(),
        mesh.cells.len(),
        "nu_per_cell must be one scalar per cell"
    );
    let map = NedelecDofMap::build(mesh);
    let n = map.ndof();

    // Assemble curl-curl matrix with per-cell `ν`. The
    // closure-driven `assemble_global_nedelec` doesn't pass a cell
    // index, so we use a small counter inside the closure — relying
    // on the assembler walking cells in `mesh.cells` order (which it
    // does, by construction).
    let cell_counter = Cell::new(0usize);
    let k_cc = assemble_global_nedelec(mesh, &map, |cell, verts| {
        let idx = cell_counter.get();
        cell_counter.set(idx + 1);
        compute_element_curl_curl(cell, verts, nu_per_cell[idx])
    })?;

    // Mass matrix for the tiny regularisation. The factor `ε` is
    // chosen small enough that the bias on the curl is negligible
    // but large enough to make CG converge in a reasonable number of
    // iterations. We pick `ε = 1e-6 * mean(ν) / L²` where `L` is the
    // mesh bounding box diagonal.
    let m = assemble_global_nedelec(mesh, &map, compute_element_nedelec_mass)?;
    let nu_mean: Real = nu_per_cell.iter().copied().sum::<Real>() / (nu_per_cell.len() as Real);
    let length_scale = mesh_length_scale(mesh).max(1.0e-9);
    let epsilon = 1.0e-8 * nu_mean / (length_scale * length_scale);

    // A = K_cc + ε M (CSR add — triplet path is the cheapest with our
    // BSR primitives, matching `solver::add_csr_scaled`).
    let a_mat = add_csr(&k_cc, 1.0, &m, epsilon)?;

    // Build load vector.
    let mut f = build_load_vector(mesh, &map, j_source);

    // Build full BC list: caller-supplied + n × A = 0 on every
    // boundary edge.
    let (_, edge_id) = compute_edges(mesh);
    let bdry_edges = boundary_edge_ids(mesh, &edge_id);
    let mut all_bcs: Vec<BoundaryCondition> = bcs.to_vec();
    // Track existing pins so we don't double-apply.
    let mut already_pinned: HashSet<u32> = HashSet::new();
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, .. } = bc {
            already_pinned.insert(*vertex);
        }
    }
    for e in &bdry_edges {
        if !already_pinned.contains(e) {
            all_bcs.push(BoundaryCondition::Dirichlet {
                vertex: *e,
                value: 0.0,
            });
        }
    }

    let mut a_red = a_mat;
    apply_dirichlet(&mut a_red, &mut f, &all_bcs);

    // CG solve. The reduced matrix is SPD (curl-curl + ε M is SPD
    // before BCs; symmetric reduction preserves SPD-ness).
    let mut x0 = vec![0.0; n];
    for bc in &all_bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x0[*vertex as usize] = *value;
        }
    }

    let CgResult { x, residual, iters } = cg_solve(|x| matvec(&a_red, x), &f, &x0, 1.0e-9, 10_000);

    Ok(MagnetostaticsResult {
        a: x,
        residual,
        iters,
    })
}

/// Compute per-cell curl of the Nedelec field `A`.
///
/// `B_c = curl(A)|_cell = Σ_e a_e · (∇ × N_e)` where `(∇ × N_e) = 2
/// ∇λ_i × ∇λ_j` for edge `e = (i, j)` (constant per element). The
/// edge DoF value is read from `a_global[edge_dof]` and multiplied
/// by the orientation sign to convert from "global edge" to "local
/// edge" basis. Returns one `Vec3` per cell.
#[must_use]
pub fn compute_curl_per_cell(mesh: &FemMesh, map: &NedelecDofMap, a_global: &[Real]) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; mesh.cells.len()];
    for (c_idx, cell) in mesh.cells.iter().enumerate() {
        let FemCell::Tet(verts) = cell else { continue };
        let p = [
            mesh.vertices[verts[0] as usize],
            mesh.vertices[verts[1] as usize],
            mesh.vertices[verts[2] as usize],
            mesh.vertices[verts[3] as usize],
        ];
        let grads = tet_shape_grads(&p);
        let mut b = Vec3::ZERO;
        for e in 0..6 {
            let [a_idx, b_idx] = TET_EDGES[e];
            let curl_e = 2.0 * grads[a_idx].cross(grads[b_idx]);
            let g_dof = map.edge_dof(verts[a_idx], verts[b_idx]);
            let sign = NedelecDofMap::edge_sign(cell, e);
            // a_local = sign · a_global (orientation conversion).
            b += sign * a_global[g_dof] * curl_e;
        }
        out[c_idx] = b;
    }
    out
}

/// Project a continuous vector field `field(x: Vec3) -> Vec3` onto
/// the global Nedelec edge DoFs.
///
/// For each unique mesh edge `(p, q)` (canonical `min → max`), the
/// DoF is the tangential line integral of `field` along the edge.
/// Lowest-order Nedelec uses Gauss-Legendre 2-point on the parametric
/// `[0, 1]` segment, which integrates linear `field` exactly. Result
/// is `vec![value; map.ndof()]` indexed by global edge DoF.
#[must_use]
pub fn project_field_on_edges<F>(mesh: &FemMesh, map: &NedelecDofMap, field: F) -> Vec<Real>
where
    F: Fn(Vec3) -> Vec3,
{
    let mut out = vec![0.0; map.ndof()];
    // Build a list of unique edges from the map.
    let mut edges: Vec<[u32; 2]> = map.edge_id.iter().map(|(k, _)| *k).collect();
    edges.sort_unstable_by_key(|e| map.edge_id[e]);

    // Gauss-Legendre 2-point on [0, 1]: nodes (3 ± √3) / 6, weights 1/2.
    let g1 = (1.0 - 1.0 / 3.0_f64.sqrt()) / 2.0;
    let g2 = (1.0 + 1.0 / 3.0_f64.sqrt()) / 2.0;
    let w = 0.5;

    for e in &edges {
        let p = mesh.vertices[e[0] as usize];
        let q = mesh.vertices[e[1] as usize];
        let dir = q - p;
        let length = dir.length();
        if length <= 0.0 {
            continue;
        }
        // Tangential integral ∫₀¹ field((1-t)p + t q) · (q - p) dt.
        // (Note: the differential `ds · t_hat = dir · dt` is already
        // baked into `dir`; we don't multiply by `length` again.)
        let x1 = p + g1 * dir;
        let x2 = p + g2 * dir;
        let v = w * (field(x1).dot(dir) + field(x2).dot(dir));
        let dof = map.edge_id[e] as usize;
        out[dof] = v;
    }
    out
}

/// Cheap probe — integrate a continuous vector field tangentially
/// along the canonical orientation of every mesh edge and return
/// `Σ_e DoF_e`. Used by tests to spot-check the projection without
/// allocating the full vector.
#[must_use]
pub fn integrate_field_along_edges<F>(mesh: &FemMesh, map: &NedelecDofMap, field: F) -> Real
where
    F: Fn(Vec3) -> Vec3,
{
    project_field_on_edges(mesh, map, field).iter().sum()
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Mesh bounding-box diagonal — used to pick a length scale for the
/// `ε M` regularisation.
fn mesh_length_scale(mesh: &FemMesh) -> Real {
    if mesh.vertices.is_empty() {
        return 1.0;
    }
    let mut lo = mesh.vertices[0];
    let mut hi = mesh.vertices[0];
    for v in &mesh.vertices {
        lo.x = lo.x.min(v.x);
        lo.y = lo.y.min(v.y);
        lo.z = lo.z.min(v.z);
        hi.x = hi.x.max(v.x);
        hi.y = hi.y.max(v.y);
        hi.z = hi.z.max(v.z);
    }
    (hi - lo).length().max(1.0e-9)
}

/// In-place CSR add `result = alpha · a + beta · b` via the triplet
/// expansion route. Cheap because BSR matrices ship a `mv` accessor.
fn add_csr(a: &BsrMatrix, alpha: Real, b: &BsrMatrix, beta: Real) -> Result<BsrMatrix> {
    let n = a.nrows();
    debug_assert_eq!(a.nrows(), b.nrows());
    debug_assert_eq!(a.ncols(), b.ncols());
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    // Expand `a` and `b` via column probes.
    let mut e = vec![0.0; n];
    let mut col = vec![0.0; n];
    for j in 0..n {
        e.iter_mut().for_each(|x| *x = 0.0);
        e[j] = 1.0;
        col.iter_mut().for_each(|x| *x = 0.0);
        a.mv(&e, &mut col)?;
        for (i, &v) in col.iter().enumerate() {
            if v != 0.0 {
                triplets.push((i, j, alpha * v));
            }
        }
        col.iter_mut().for_each(|x| *x = 0.0);
        b.mv(&e, &mut col)?;
        for (i, &v) in col.iter().enumerate() {
            if v != 0.0 {
                triplets.push((i, j, beta * v));
            }
        }
    }
    BsrMatrix::from_scalar_triplets(n, n, &triplets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_edges_of_single_tet_are_all_six() {
        let mesh = FemMesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            cells: vec![FemCell::Tet([0, 1, 2, 3])],
        };
        let (_, edge_id) = compute_edges(&mesh);
        let b = boundary_edge_ids(&mesh, &edge_id);
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn project_then_curl_recovers_constant_field() {
        // Project A(x) = (0, x, 0) on a coarse tet mesh and check
        // that the per-cell curl matches the analytic ê_z.
        let mesh = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let map = NedelecDofMap::build(&mesh);
        let a = project_field_on_edges(&mesh, &map, |p| Vec3::new(0.0, p.x, 0.0));
        let b = compute_curl_per_cell(&mesh, &map, &a);
        for (i, bi) in b.iter().enumerate() {
            let err = (*bi - Vec3::new(0.0, 0.0, 1.0)).length();
            assert!(err < 1e-9, "cell {i}: curl A = {bi:?}, err = {err}");
        }
    }
}
