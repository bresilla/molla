//! Boundary conditions for the v1 FEM solvers.
//!
//! v1 supports:
//! - **Dirichlet** — fix `u(vertex) = value`. The "reduce" form is
//!   symmetric: for every pinned DoF `r` with value `v`, the column
//!   `K[:, r]` is also eliminated by shifting `f[i] -= K[i, r] · v`
//!   for every non-pinned row `i` and zeroing `K[i, r]`. The pinned
//!   row is then `K[r, r] = 1, K[r, j≠r] = 0, f[r] = v`. The
//!   resulting reduced matrix is symmetric (and SPD when the parent
//!   is) so CG retains its convergence guarantee — required for the
//!   adjoint solve and for non-zero Dirichlet values that previously
//!   silently relied on row-only elimination.
//! - **Neumann** — `n · ∇u = flux` over a face. v1 records the
//!   constraint but does not yet integrate the surface contribution
//!   into the RHS — the homogeneous (zero-flux) case happens
//!   automatically because P1 + bilinear weak form drops the
//!   boundary term when no Neumann is specified. The Neumann variant
//!   is wired so callers can store the data; full integration lands
//!   in Stage 36.1 alongside elasticity tractions.
//!
//! All BCs are addressed by global DoF index (= vertex index for
//! `FunctionSpace::P1Nodal`). The caller is responsible for picking
//! the right vertices given the mesh.
//!
//! ## Stage 43 v2 — DG Nitsche weak Dirichlet
//!
//! The `BoundaryCondition::WeakDirichletDG { vertex, value, sigma }`
//! variant pins the **trace** of a DG field at a boundary vertex
//! through a Nitsche-style penalty + consistency term integrated on
//! the boundary face. Unlike the strong Dirichlet variants, the BC
//! never overwrites a row directly — it modifies the global matrix
//! by adding the symmetric interior-penalty boundary block
//!
//! ```text
//!   a_∂Ω(u, v) = − α ∫_∂Ω (∇u·n) v dS
//!                − α ∫_∂Ω u (∇v·n) dS
//!                + (σ α / h_F) ∫_∂Ω u v dS
//! ```
//!
//! and contributes the matching RHS
//!
//! ```text
//!   l_∂Ω(v) = − α ∫_∂Ω u_D (∇v·n) dS + (σ α / h_F) ∫_∂Ω u_D v dS
//! ```
//!
//! where `u_D` is the prescribed Dirichlet trace (interpolated linearly
//! between the vertex `value`s on each boundary face). Faces where not
//! every vertex carries a `WeakDirichletDG` entry are skipped.
//!
//! Apply via [`apply_dg_bcs`] — the regular [`apply_bcs`] entrypoint
//! drops the variant silently (DG needs the cell + degree context that
//! `apply_bcs` doesn't have).

use crate::math::Real;
use crate::math::sparse::BsrMatrix;

/// A single boundary-condition entry.
///
/// The scalar variants (`Dirichlet`, `Neumann`) target one DoF per
/// vertex (the `P1Nodal` Stage 36 v1 case). The vector variants
/// (`DirichletVector`, `NeumannTraction`) target every component of a
/// vector field at a vertex / face (the Stage 36.1 elasticity case).
///
/// Both forms address the global DoF using the interleaved layout
/// `dof = vertex * dim + component`. The applier
/// ([`apply_dirichlet`]) infers `dim` from the matrix size + mesh
/// caller-side (the BC payload carries the per-component values).
#[derive(Clone, Debug)]
pub enum BoundaryCondition {
    /// Scalar Dirichlet: pin DoF at `vertex` to exactly `value`. For
    /// `P1Vector { dim }` this only constrains component 0; use
    /// `DirichletVector` to pin every component.
    Dirichlet { vertex: u32, value: Real },
    /// Vector Dirichlet: pin every component at `vertex` to the
    /// prescribed `values`. The slice length must match the function
    /// space's `dofs_per_vertex` (typically 2 or 3 for elasticity).
    DirichletVector { vertex: u32, values: Vec<Real> },
    /// Neumann: prescribe a scalar flux `flux` over the triangular
    /// `face` (3 vertex indices). Stage 36 v1 stores but does not
    /// integrate this (zero-flux is the homogeneous default).
    Neumann { face: [u32; 3], flux: Real },
    /// Neumann traction: prescribe a 3D traction vector over the
    /// triangular `face`. The applier integrates `∫ N · t dA` and
    /// adds the contribution to the RHS for every face vertex,
    /// component-by-component. Used by Stage 36.1 elasticity to push
    /// loads onto a free surface.
    NeumannTraction { face: [u32; 3], traction: [Real; 3] },
    /// Stage 43 v2 — DG weak (Nitsche) Dirichlet at a boundary
    /// vertex. The applier ([`apply_dg_bcs`]) integrates the
    /// symmetric-interior-penalty boundary block + matching RHS on
    /// every boundary face whose vertices all carry the variant. The
    /// default value of `sigma` recommended by the literature is
    /// `degree · (degree + 1) · 10` — see [`default_nitsche_sigma`].
    WeakDirichletDG {
        /// Boundary vertex index in the parent `FemMesh`.
        vertex: u32,
        /// Prescribed Dirichlet value `u_D(vertex)`.
        value: Real,
        /// Nitsche penalty coefficient `σ` (dimensionless — scaled by
        /// `α / h_F` internally).
        sigma: Real,
    },
}

/// Recommended Nitsche penalty for `WeakDirichletDG` of DG polynomial
/// order `degree`. Returns `degree · (degree + 1) · 10` — large enough
/// to keep the boundary block coercive across reasonable mesh sizes,
/// small enough to avoid conditioning blow-up.
#[must_use]
pub fn default_nitsche_sigma(degree: usize) -> Real {
    (degree as Real) * ((degree + 1) as Real) * 10.0
}

/// Apply Dirichlet BCs + Neumann tractions in-place.
///
/// **Dirichlet:** symmetric reduction — for every pinned DoF `r`
/// with value `v`, the column `K[:, r]` is eliminated by shifting
/// `f[i] -= K[i, r] · v` for every non-pinned row `i` and zeroing
/// `K[i, r]`, then the row `K[r, :]` is zeroed except for
/// `K[r, r] = 1` and `f[r] = v`. The reduced matrix retains the
/// symmetry of the parent — CG (which assumes self-adjoint) is safe
/// on the result, and the same RHS shift makes the solution
/// independent of whether `v` is zero or non-zero. For
/// `DirichletVector`, the same operation is repeated for each
/// component DoF `vertex * dim + c`.
///
/// **Neumann tractions:** for `NeumannTraction { face, traction }`,
/// add `(area / 3) * traction[c]` to the RHS at each face vertex
/// `v`'s component DoF `v * dim + c`. This is the consistent (P1
/// surface integral) lumping for a constant traction over a flat
/// triangular face. `vertices` is the mesh vertex array (needed for
/// the face area).
///
/// The scalar `Neumann { face, flux }` variant is still a no-op —
/// zero flux is the homogeneous default and the v1 weak form
/// already drops the boundary integral.
///
/// `dim` is the per-vertex component count (1 for scalar fields,
/// `space.dofs_per_vertex()` for vector fields). Pass 1 for the
/// classical `P1Nodal` use-case.
pub fn apply_dirichlet(k: &mut BsrMatrix, f: &mut [Real], bcs: &[BoundaryCondition]) {
    apply_bcs(k, f, bcs, &[], 1);
}

/// Vector-aware applier — same as [`apply_dirichlet`] but takes the
/// per-vertex DoF count and the mesh vertex positions (required for
/// `NeumannTraction` face-area integration).
pub fn apply_bcs(
    k: &mut BsrMatrix,
    f: &mut [Real],
    bcs: &[BoundaryCondition],
    vertices: &[crate::math::Vec3],
    dim: usize,
) {
    debug_assert_eq!(k.b, 1, "apply_bcs: only B = 1 supported in v1");
    debug_assert_eq!(k.nrows(), f.len(), "RHS length mismatch");
    debug_assert!(dim >= 1, "apply_bcs: dim must be >= 1");

    // Pass 1: integrate Neumann tractions into the RHS first. (Order
    // matters — Dirichlet overwrites RHS rows it pins, so any traction
    // spuriously assigned to a constrained DoF is silently dropped,
    // matching the textbook "Dirichlet wins" convention.)
    for bc in bcs {
        if let BoundaryCondition::NeumannTraction { face, traction } = bc {
            // Triangle face area in 3D.
            let a = vertices[face[0] as usize];
            let b = vertices[face[1] as usize];
            let c = vertices[face[2] as usize];
            let area = 0.5 * (b - a).cross(c - a).length();
            let third = area / 3.0;
            for v in face {
                let v = *v as usize;
                for ci in 0..dim.min(3) {
                    f[v * dim + ci] += third * traction[ci];
                }
            }
        }
    }

    // Pass 2: collect every pinned DoF + its value into a flat array.
    // We need the full pin set up-front so the column-elimination
    // pass can shift the RHS by the right Dirichlet values in one
    // sweep of the matrix.
    let n = k.nrows();
    let mut pinned: Vec<Option<Real>> = vec![None; n];
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, value } => {
                pinned[*vertex as usize * dim] = Some(*value);
            }
            BoundaryCondition::DirichletVector { vertex, values } => {
                debug_assert_eq!(
                    values.len(),
                    dim,
                    "DirichletVector: values.len() ({}) != dim ({})",
                    values.len(),
                    dim,
                );
                let base = *vertex as usize * dim;
                for (ci, val) in values.iter().enumerate() {
                    pinned[base + ci] = Some(*val);
                }
            }
            BoundaryCondition::Neumann { .. } => {}
            BoundaryCondition::NeumannTraction { .. } => {}
            // DG Nitsche BCs are handled by `apply_dg_bcs`, which
            // needs the cell + degree context that `apply_bcs` lacks.
            // Silently skip here so callers can mix CG + DG BCs in
            // the same vector without panicking.
            BoundaryCondition::WeakDirichletDG { .. } => {}
        }
    }

    // Pass 3: symmetric column elimination — for every non-pinned
    // row i, look for entries `K[i, j]` where j is pinned; shift
    // `f[i] -= K[i, j] · value[j]` and zero the matrix entry. This
    // restores the symmetry the textbook row-only "pin" breaks.
    for i in 0..n {
        if pinned[i].is_some() {
            continue;
        }
        let lo = k.row_offsets[i];
        let hi = k.row_offsets[i + 1];
        for slot in lo..hi {
            let j = k.col_idx[slot];
            if let Some(val_j) = pinned[j] {
                f[i] -= k.values[slot] * val_j;
                k.values[slot] = 0.0;
            }
        }
    }

    // Pass 4: pin each constrained row — zero off-diagonals, set
    // diagonal to 1, set the RHS to the prescribed value.
    for r in 0..n {
        let Some(value) = pinned[r] else { continue };
        pin_row(k, f, r, value);
    }
}

/// Internal helper — zero row `r`, set diagonal to 1, set `f[r] = value`.
/// The caller must already have done the symmetric column elimination
/// (see `apply_bcs` pass 3) before invoking this; otherwise the
/// reduced matrix loses symmetry.
fn pin_row(k: &mut BsrMatrix, f: &mut [Real], r: usize, value: Real) {
    let lo = k.row_offsets[r];
    let hi = k.row_offsets[r + 1];
    let mut diag_found = false;
    for slot in lo..hi {
        let c = k.col_idx[slot];
        if c == r {
            k.values[slot] = 1.0;
            diag_found = true;
        } else {
            k.values[slot] = 0.0;
        }
    }
    debug_assert!(
        diag_found,
        "pin_row({r}): no diagonal entry in the CSR pattern — \
         row has no assembled connectivity, system is singular",
    );
    f[r] = value;
}

// ─── Stage 43 v2 — DG Nitsche boundary applier ────────────────────────

/// Apply DG-aware boundary conditions to a DG global matrix + RHS.
///
/// Currently handles [`BoundaryCondition::WeakDirichletDG`] only; the
/// other variants are silently ignored (use [`apply_bcs`] for those —
/// it's safe to call both in sequence on different system flavours).
///
/// **Inputs:**
///
/// - `k`         — global DG stiffness / system matrix (BSR, B = 1).
///   The boundary block is **added** to the matrix in place, so the
///   caller must rebuild `k` if the BC set changes (consistent with
///   the existing strong-Dirichlet applier semantics).
/// - `f`         — global DG RHS vector (modified in place).
/// - `bcs`       — flat slice of BCs (mixing CG/DG variants is OK).
/// - `mesh`      — DG mesh (vertex + cell array — same one used by the
///   `assemble_dg_global` call that built `k`).
/// - `degree`    — DG polynomial order (matches the assembly call).
/// - `alpha`     — diffusion coefficient (same value passed to
///   `compute_element_dg_stiffness` and `compute_dg_flux_jump_block`).
///
/// **What it does:** for every boundary face (= face owned by exactly
/// one cell) whose two/three vertices **all** carry a `WeakDirichletDG`
/// entry, compute the Nitsche penalty + consistency boundary block
/// (`compute_nitsche_boundary_block`) and add it to `k`; compute the
/// matching `u_D`-side RHS contribution and add it to `f`.
///
/// **Penalty resolution:** the per-face penalty `σ` is the **maximum**
/// of the constituent vertex `sigma` values. This matches the standard
/// "vertex-wise penalty + face takes the strongest" convention used
/// when adjacent vertices come from different BC sources.
///
/// Returns `Ok(())` on success or an error from the BSR rebuild on
/// failure.
pub fn apply_dg_bcs(
    k: &mut BsrMatrix,
    f: &mut [Real],
    bcs: &[BoundaryCondition],
    mesh: &super::mesh::FemMesh,
    degree: usize,
    alpha: Real,
) -> crate::core::Result<()> {
    use std::collections::HashMap;

    // Index DG `WeakDirichletDG` BCs by vertex.
    struct DgVertexBc {
        value: Real,
        sigma: Real,
    }
    let mut by_vertex: HashMap<u32, DgVertexBc> = HashMap::new();
    for bc in bcs {
        if let BoundaryCondition::WeakDirichletDG {
            vertex,
            value,
            sigma,
        } = bc
        {
            by_vertex.insert(
                *vertex,
                DgVertexBc {
                    value: *value,
                    sigma: *sigma,
                },
            );
        }
    }
    if by_vertex.is_empty() {
        return Ok(());
    }

    // Identify boundary faces — face_to_cells with exactly one owner.
    // Mirrors `compute_interior_faces` but keeps the *boundary* faces
    // instead of dropping them.
    let mut face_to_cells: HashMap<Vec<u32>, Vec<(u32, [usize; 3])>> = HashMap::new();
    for (cell_idx, cell) in mesh.cells.iter().enumerate() {
        match cell {
            super::mesh::FemCell::Tri(v) => {
                let edges: [[u32; 2]; 3] = [[v[0], v[1]], [v[1], v[2]], [v[2], v[0]]];
                for e in &edges {
                    let mut key = vec![e[0], e[1]];
                    key.sort_unstable();
                    face_to_cells
                        .entry(key)
                        .or_default()
                        .push((cell_idx as u32, [0, 0, 0]));
                }
            }
            super::mesh::FemCell::Tet(v) => {
                let faces: [[u32; 3]; 4] = [
                    [v[1], v[2], v[3]],
                    [v[0], v[2], v[3]],
                    [v[0], v[1], v[3]],
                    [v[0], v[1], v[2]],
                ];
                for fc in &faces {
                    let mut key = vec![fc[0], fc[1], fc[2]];
                    key.sort_unstable();
                    face_to_cells
                        .entry(key)
                        .or_default()
                        .push((cell_idx as u32, [0, 0, 0]));
                }
            }
        }
    }

    // DG DoF offsets (running sum of per-cell DoF count).
    let mut offsets = Vec::with_capacity(mesh.cells.len() + 1);
    let mut running = 0usize;
    for c in &mesh.cells {
        offsets.push(running);
        running += super::function_space::dg_ndof_per_cell(c, degree);
    }
    offsets.push(running);

    // Triplets to add into `k`.
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();

    for (face_verts, owners) in &face_to_cells {
        if owners.len() != 1 {
            continue;
        }
        // Only process this face if *every* vertex of the face carries
        // a WeakDirichletDG entry — partial constraints are ignored.
        let mut bc_at: Vec<&DgVertexBc> = Vec::with_capacity(face_verts.len());
        let mut all_constrained = true;
        for v in face_verts {
            if let Some(b) = by_vertex.get(v) {
                bc_at.push(b);
            } else {
                all_constrained = false;
                break;
            }
        }
        if !all_constrained {
            continue;
        }
        // Per-face penalty = max of vertex `sigma` values.
        let sigma_face: Real = bc_at
            .iter()
            .map(|b| b.sigma)
            .fold(Real::NEG_INFINITY, Real::max);

        let cell_idx = owners[0].0 as usize;
        let cell = &mesh.cells[cell_idx];
        let off = offsets[cell_idx];
        let nl = super::function_space::dg_ndof_per_cell(cell, degree);

        // Build boundary block + RHS for this face.
        let (block, rhs) = super::dg_element::compute_nitsche_boundary_block(
            cell,
            face_verts,
            &mesh.vertices,
            alpha,
            degree,
            sigma_face,
            &bc_at.iter().map(|b| b.value).collect::<Vec<_>>(),
        );

        // Scatter block into triplets and add rhs into f.
        for i in 0..nl {
            for j in 0..nl {
                let v = block[i][j];
                if v != 0.0 {
                    triplets.push((off + i, off + j, v));
                }
            }
            f[off + i] += rhs[i];
        }
    }

    if triplets.is_empty() {
        return Ok(());
    }

    // Add the boundary triplets into `k` by rebuilding via mv expansion.
    // We extract `k` as triplets via column scans, merge in the new
    // boundary triplets, and rebuild — matches the `add_bsr` strategy
    // used by the Stage 43 v1 advanced tests.
    let n = k.nrows();
    let mut all_triplets: Vec<(usize, usize, Real)> = Vec::new();
    for j in 0..n {
        let mut e = vec![0.0_f64 as Real; n];
        e[j] = 1.0;
        let mut col = vec![0.0_f64 as Real; n];
        k.mv(&e, &mut col)?;
        for (i, &v) in col.iter().enumerate() {
            if v != 0.0 {
                all_triplets.push((i, j, v));
            }
        }
    }
    all_triplets.extend(triplets);
    *k = BsrMatrix::from_scalar_triplets(n, n, &all_triplets)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sparse::BsrMatrix;

    fn build_test_csr() -> BsrMatrix {
        // 4×4 dense-ish (full pattern), values just so we can check
        // they get zeroed correctly.
        let mut triplets = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                triplets.push((i, j, vec![(1 + i * 4 + j) as Real]));
            }
        }
        BsrMatrix::from_triplets(1, 4, 4, &triplets).unwrap()
    }

    #[test]
    fn apply_dirichlet_zeros_row_and_sets_diag() {
        let mut k = build_test_csr();
        let mut f = vec![10.0, 20.0, 30.0, 40.0];
        let bcs = [BoundaryCondition::Dirichlet {
            vertex: 2,
            value: 7.5,
        }];
        apply_dirichlet(&mut k, &mut f, &bcs);
        // Row 2 should be (0, 0, 1, 0).
        let lo = k.row_offsets[2];
        let hi = k.row_offsets[2 + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            let v = k.values[slot];
            if c == 2 {
                assert!((v - 1.0).abs() < 1e-15);
            } else {
                assert!(v.abs() < 1e-15);
            }
        }
        assert!((f[2] - 7.5).abs() < 1e-15);
        // With symmetric reduction the column is also eliminated, so
        // every other row's RHS absorbs `K[i, 2] · 7.5`. For the test
        // matrix `K[i, j] = 1 + i*4 + j`, that gives `K[0, 2] = 3`,
        // so `f[0]` shifts from 10 → 10 − 3·7.5 = −12.5.
        assert!((f[0] - (10.0 - 3.0 * 7.5)).abs() < 1e-15, "f[0] = {}", f[0]);
    }

    #[test]
    fn dirichlet_symmetric_reduction_preserves_symmetry() {
        // SPD matrix: 4×4 tridiagonal Laplacian + identity boost.
        let mut triplets = Vec::new();
        for i in 0..4usize {
            triplets.push((i, i, vec![3.0_f64]));
            if i + 1 < 4 {
                triplets.push((i, i + 1, vec![-1.0_f64]));
                triplets.push((i + 1, i, vec![-1.0_f64]));
            }
        }
        let mut k = BsrMatrix::from_triplets(1, 4, 4, &triplets).unwrap();
        let mut f = vec![1.0, 1.0, 1.0, 1.0];

        // Pin DoF 0 to a non-zero value — exercises the column shift.
        let bcs = [BoundaryCondition::Dirichlet {
            vertex: 0,
            value: 2.0,
        }];
        apply_dirichlet(&mut k, &mut f, &bcs);

        // K[0, 1] should now be zero (column elimination).
        let lo = k.row_offsets[0];
        let hi = k.row_offsets[1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            if c == 0 {
                assert!((k.values[slot] - 1.0).abs() < 1e-15);
            } else {
                assert!(k.values[slot].abs() < 1e-15, "K[0, {c}] not zeroed");
            }
        }
        // K[1, 0] should also now be zero.
        let lo = k.row_offsets[1];
        let hi = k.row_offsets[1 + 1];
        for slot in lo..hi {
            if k.col_idx[slot] == 0 {
                assert!(k.values[slot].abs() < 1e-15, "K[1, 0] not zeroed");
            }
        }
        // RHS at row 1 should have absorbed K[1, 0] · value = -1 · 2 = -2.
        // f[1] was 1, now should be 1 - (-1)·2 = 3.
        assert!((f[1] - 3.0).abs() < 1e-15, "f[1] = {} (expected 3.0)", f[1]);
        // f[0] should be the pin value.
        assert!((f[0] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn neumann_is_a_noop_in_v1() {
        let mut k = build_test_csr();
        let snapshot_values = k.values.clone();
        let mut f = vec![1.0, 2.0, 3.0, 4.0];
        let f_before = f.clone();
        let bcs = [BoundaryCondition::Neumann {
            face: [0, 1, 2],
            flux: 0.5,
        }];
        apply_dirichlet(&mut k, &mut f, &bcs);
        assert_eq!(k.values, snapshot_values);
        assert_eq!(f, f_before);
    }
}
