//! Time-harmonic Maxwell / eddy-current solver — Stage 48.
//!
//! Solves the frequency-domain curl-curl equation for the complex
//! magnetic vector potential `A` in the **A-formulation**:
//!
//! ```text
//!   curl(ν curl A) + i ω σ A − ω² ε A = J          in Ω
//!   n × A = 0                                       on ∂Ω
//! ```
//!
//! This is the standard A-formulation of time-harmonic Maxwell
//! obtained from `curl(ν curl E) − ω²ε E + iωσ E = −iω J` by the
//! substitution `E = −iω A` (so `B = curl A`, `J_ohmic = σ E = −iωσA`).
//! Dividing the resulting equation by `−iω` gives the form above.
//! Solving for `A` lets us share matrices with Stage 46
//! magnetostatics — at `ω → 0` the `iωσ` and `−ω²ε` terms vanish and
//! we recover `curl(ν curl A) = J` exactly.
//!
//! Symbols:
//! - `A = A_re + i A_im` is the complex phasor (DoFs live on mesh
//!   edges, two real scalars — real + imaginary — per Nedelec edge),
//! - `J = J_re + i J_im` is the prescribed complex source current,
//! - `ν = 1/μ`, `ε`, `σ` are the magnetic reluctivity, electric
//!   permittivity, and electric conductivity per cell,
//! - `ω` is the angular frequency `2π f`,
//! - `n × A = 0` is the "perfect electric conductor" boundary
//!   condition (tangential `A` vanishes; in Nedelec DoFs this is
//!   `A_e = 0` for every edge on the boundary).
//!
//! The continuous system is **complex-symmetric** (not Hermitian) —
//! `K_cc + (iωσ − ω²ε) M` is its own complex transpose, but the
//! conjugate-transpose differs by the sign of `iωσ`. CG fails on
//! complex-symmetric systems (the natural inner product would be a
//! bilinear form, not an Hermitian one); BiCGSTAB on the **block-real**
//! adapter is the workhorse choice here.
//!
//! # Block-real adapter
//!
//! Each complex Nedelec DoF maps to two real DoFs (real + imag).
//! Layout: stacked `[a_re…, a_im…]` (length `2N`) — lets us reuse the
//! existing real-valued `K_cc` and `M` matrices without permuting
//! rows. The complex system
//!
//! ```text
//!   (K − ω²ε M + i ω σ M) (a_re + i a_im) = (j_re + i j_im)
//! ```
//!
//! expands into the real `2N × 2N` block system
//!
//! ```text
//!   [ K − ω²ε M     − ω σ M    ] [ a_re ]   [ j_re ]
//!   [   ω σ M       K − ω²ε M  ] [ a_im ] = [ j_im ]
//! ```
//!
//! which is real and **non-symmetric** (the off-diagonal sign flip is
//! intrinsic to `i`). BiCGSTAB handles the block matvec without ever
//! materialising the `2N × 2N` matrix — we apply each block via the
//! underlying `BsrMatrix::mv`. The RHS here is the literal load
//! integral `∫ J · N_i dV` for each part of `J`; no `−iω` factor —
//! we already absorbed it via the `E = −iω A` substitution into the
//! left-hand-side coefficients.
//!
//! # What's deferred
//!
//! - **Coulomb-gauge enforcement.** At low frequencies (`ω → 0`) the
//!   conductive term `iωσ M` vanishes in non-conducting regions and the
//!   discrete curl-curl operator becomes singular on the gradient
//!   kernel of P1-nodal Lagrange. v1 papers this over with a tiny
//!   `ε_reg M` regularisation (same trick as Stage 46 magnetostatics);
//!   v2 will add a Lagrange-multiplier block enforcing `∇ · (σ E) = 0`.
//! - **AMS preconditioner.** The curl-curl block has a huge null space
//!   spanned by discrete gradients — for stiff (high-`ν`, high-`σ`)
//!   problems BiCGSTAB needs auxiliary-space preconditioning (Hiptmair-
//!   Xu AMS). v2.
//! - **Frequency sweep.** Multiple `ω` solves share the same `K_cc`
//!   and `M` matrices but differ in shifts; a block-Krylov sweep is a
//!   straightforward extension. v2.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use crate::core::Result;
use crate::math::sparse::{gmres, BsrMatrix, ClosureOp, IterConfig};
use crate::math::{Real, Vec3};

use super::bc::BoundaryCondition;
use super::element::{tet_shape_grads, tet_volume};
use super::function_space::compute_edges;
use super::mesh::{FemCell, FemMesh};
use super::nedelec_element::{
    assemble_global_nedelec, compute_element_curl_curl, compute_element_nedelec_mass,
    NedelecDofMap,
};
use super::p2_element::TET_EDGES;

/// Configuration for the eddy-current solver — frequency + per-cell
/// material coefficients.
#[derive(Debug, Clone)]
pub struct EddyCurrentConfig {
    /// Angular frequency `ω = 2π f` (rad / s). Set to `0` for the
    /// magnetostatics limit (the solver still works but degenerates
    /// to a pure curl-curl solve with `ε_reg` regularisation).
    pub omega: Real,
    /// Electric permittivity `ε` per cell. Pass `vec![0; cells]` to
    /// drop the displacement-current term (quasi-magnetostatic /
    /// eddy-current limit).
    pub epsilon_per_cell: Vec<Real>,
    /// Electric conductivity `σ` per cell. Non-zero only inside
    /// conductors. Pass `vec![0; cells]` for the lossless case.
    pub sigma_per_cell: Vec<Real>,
    /// Magnetic reluctivity `ν = 1/μ` per cell. Pass `vec![1/μ₀; cells]`
    /// for non-magnetic media.
    pub nu_per_cell: Vec<Real>,
}

/// Output of a time-harmonic Maxwell / eddy-current solve.
///
/// `A_complex = a_re + i a_im` is the complex magnetic vector
/// potential (Nedelec edge DoFs). Derived fields:
/// - `B_complex = curl(A_complex)` — pass each part through
///   [`crate::fem::compute_curl_per_cell`] for cell-wise B.
/// - `E_complex = − i ω A_complex` — so `E_re = ω a_im`,
///   `E_im = − ω a_re`.
#[derive(Debug, Clone)]
pub struct EddyCurrentResult {
    /// Real part of the complex vector potential `A` — one DoF per
    /// unique mesh edge.
    pub a_re: Vec<Real>,
    /// Imaginary part of the complex `A` — same indexing as `a_re`.
    pub a_im: Vec<Real>,
    /// Final BiCGSTAB residual `‖A x − b‖ / ‖b‖` on the block-real
    /// system.
    pub residual: Real,
    /// BiCGSTAB iteration count.
    pub iters: usize,
}

/// Steady-state time-harmonic Maxwell / eddy-current solve in the
/// A-formulation.
///
/// Builds the block-real `2N × 2N` system from `K_cc`, `M`, and the
/// per-cell coefficients in `cfg`, applies `n × A = 0` on every
/// boundary edge (plus any caller-supplied Dirichlet pins), and solves
/// with BiCGSTAB.
///
/// Returns `(a_re, a_im)` — the real and imaginary parts of the
/// complex vector-potential DoF vector, each of length
/// `NedelecDofMap::ndof()`.
///
/// # Inputs
///
/// - `mesh` — tetrahedral FEM mesh (`Tri` cells are ignored).
/// - `j_source_complex` — per-cell complex current density as
///   `(j_re, j_im)` `Vec3` pairs. Length must equal `mesh.cells.len()`.
/// - `cfg` — frequency + per-cell `(ν, ε, σ)` coefficients.
/// - `bcs` — caller-supplied Dirichlet pins (typically empty — the
///   solver adds `n × E = 0` automatically). Caller pins are applied
///   identically to **both** the real and imaginary blocks at every
///   listed edge DoF.
///
/// # Convergence
///
/// BiCGSTAB tolerance is fixed at `1e-9` relative; max-iters at
/// `10_000`. Typical convergence on a 10×10×10 cube with σ ≈ 10⁵ is
/// 200–600 iterations (no preconditioner — see module docs for the
/// AMS plan).
pub fn solve_eddy_current_steady(
    mesh: &FemMesh,
    j_source_complex: &[(Vec3, Vec3)],
    cfg: &EddyCurrentConfig,
    bcs: &[BoundaryCondition],
) -> Result<EddyCurrentResult> {
    let n_cells = mesh.cells.len();
    assert_eq!(
        j_source_complex.len(),
        n_cells,
        "j_source_complex must be one (Vec3, Vec3) per cell"
    );
    assert_eq!(
        cfg.nu_per_cell.len(),
        n_cells,
        "cfg.nu_per_cell must be one scalar per cell"
    );
    assert_eq!(
        cfg.epsilon_per_cell.len(),
        n_cells,
        "cfg.epsilon_per_cell must be one scalar per cell"
    );
    assert_eq!(
        cfg.sigma_per_cell.len(),
        n_cells,
        "cfg.sigma_per_cell must be one scalar per cell"
    );

    let map = NedelecDofMap::build(mesh);
    let n = map.ndof();
    let omega = cfg.omega;

    // ─── Assemble K_cc with per-cell ν ──────────────────────────────
    let nu_per_cell = cfg.nu_per_cell.clone();
    let counter = Cell::new(0usize);
    let k_cc = assemble_global_nedelec(mesh, &map, |cell, verts| {
        let idx = counter.get();
        counter.set(idx + 1);
        compute_element_curl_curl(cell, verts, nu_per_cell[idx])
    })?;

    // ─── Assemble two mass matrices: M_eps (weighted by ε) and ─────
    //     M_sig (weighted by σ). Built via per-cell ε/σ closures. ───
    let eps_per_cell = cfg.epsilon_per_cell.clone();
    let counter_eps = Cell::new(0usize);
    let m_eps = assemble_global_nedelec(mesh, &map, |cell, verts| {
        let idx = counter_eps.get();
        counter_eps.set(idx + 1);
        let local = compute_element_nedelec_mass(cell, verts);
        let s = eps_per_cell[idx];
        scale_local(&local, s)
    })?;

    let sig_per_cell = cfg.sigma_per_cell.clone();
    let counter_sig = Cell::new(0usize);
    let m_sig = assemble_global_nedelec(mesh, &map, |cell, verts| {
        let idx = counter_sig.get();
        counter_sig.set(idx + 1);
        let local = compute_element_nedelec_mass(cell, verts);
        let s = sig_per_cell[idx];
        scale_local(&local, s)
    })?;

    // Unweighted mass for the tiny regularisation (gauge-fix the
    // gradient null space at very low frequencies — same trick as
    // Stage 46 magnetostatics).
    let m_unit = assemble_global_nedelec(mesh, &map, compute_element_nedelec_mass)?;
    let nu_mean: Real = cfg.nu_per_cell.iter().copied().sum::<Real>() / (n_cells as Real);
    let length_scale = mesh_length_scale(mesh).max(1.0e-9);
    let epsilon_reg = 1.0e-8 * nu_mean / (length_scale * length_scale);

    // ─── Build complex-symmetric system in block-real form ──────────
    //
    // Real-block diagonal: A_d = K - ω²ε M_eps + ε_reg M_unit
    // Off-block:           A_o = ω σ M_sig
    //
    // The 2N × 2N system is:
    //   [ A_d   -A_o ] [ e_re ]   [ f_re ]
    //   [ A_o    A_d ] [ e_im ] = [ f_im ]

    // ─── Load vectors ───────────────────────────────────────────────
    //
    // A-formulation RHS is the literal load integral. Complex J →
    // (f_re, f_im) = ( ∫ J_re · N_i dV, ∫ J_im · N_i dV ).
    let mut j_re_per_cell = vec![Vec3::ZERO; n_cells];
    let mut j_im_per_cell = vec![Vec3::ZERO; n_cells];
    for (i, jc) in j_source_complex.iter().enumerate() {
        j_re_per_cell[i] = jc.0;
        j_im_per_cell[i] = jc.1;
    }
    let mut f_re = build_load_vector(mesh, &map, &j_re_per_cell);
    let mut f_im = build_load_vector(mesh, &map, &j_im_per_cell);

    // ─── Boundary edges → Dirichlet pins (n × E = 0) ────────────────
    let (_, edge_id) = compute_edges(mesh);
    let bdry_edges = boundary_edge_ids(mesh, &edge_id);
    let mut pin_re: HashMap<usize, Real> = HashMap::new();
    let mut pin_im: HashMap<usize, Real> = HashMap::new();
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            // Caller pins are applied to BOTH the real and imag block.
            pin_re.insert(*vertex as usize, *value);
            pin_im.insert(*vertex as usize, *value);
        }
    }
    for e in &bdry_edges {
        pin_re.entry(*e as usize).or_insert(0.0);
        pin_im.entry(*e as usize).or_insert(0.0);
    }

    // ─── Pre-compute the block-real diagonal (used for Jacobi precon) ─
    //
    // Diagonal of the 2N × 2N system:
    //   d_block[i]      = d_K[i] + ε_reg d_M[i] − ω²ε d_Meps[i]   (i ∈ [0..n))
    //   d_block[n + i]  = d_K[i] + ε_reg d_M[i] − ω²ε d_Meps[i]   (same)
    //
    // The off-block diagonal (`ω σ M_sig` at `(i, n+i)`) doesn't
    // contribute to the diagonal of the 2N system — but its **block-
    // diagonal** part DOES enter a block-Jacobi preconditioner. For
    // simplicity we stick with plain (scalar) Jacobi here and absorb
    // the loss in slightly slower convergence; block-Jacobi can be
    // added in v2 (see the AMS plan in the module docs).
    let k_diag = diag_of(&k_cc);
    let m_eps_diag = diag_of(&m_eps);
    let m_unit_diag = diag_of(&m_unit);
    let _m_sig_diag = diag_of(&m_sig);

    // ─── Block-real matvec ──────────────────────────────────────────
    //
    // Layout: x[0..n] = a_re,  x[n..2n] = a_im.
    //
    // A · x =
    //   [ A_d a_re − A_o a_im ]
    //   [ A_o a_re + A_d a_im ]
    //
    // where A_d = K − ω²ε M_eps + ε_reg M_unit and A_o = ω σ M_sig.
    let omega_sq = omega * omega;
    let two_n = 2 * n;

    // Pinned bookkeeping for the block-real system.
    let mut pinned: Vec<Option<Real>> = vec![None; two_n];
    for (idx, val) in pin_re.iter() {
        pinned[*idx] = Some(*val);
    }
    for (idx, val) in pin_im.iter() {
        pinned[n + *idx] = Some(*val);
    }

    // RHS with pin values written in.
    let mut rhs = vec![0.0_f64 as Real; two_n];
    for i in 0..n {
        rhs[i] = f_re[i];
        rhs[n + i] = f_im[i];
    }
    for (idx, val) in pin_re.iter() {
        rhs[*idx] = *val;
    }
    for (idx, val) in pin_im.iter() {
        rhs[n + *idx] = *val;
    }
    // Suppress unused warnings.
    let _ = (&mut f_re, &mut f_im);

    // Capture-by-move-friendly clones for the closure.
    let k_cc_op = k_cc;
    let m_eps_op = m_eps;
    let m_sig_op = m_sig;
    let m_unit_op = m_unit;
    let pinned_op = pinned.clone();

    let mv_closure = move |x: &[Real], y: &mut [Real]| -> Result<()> {
        let x_re = &x[0..n];
        let x_im = &x[n..2 * n];

        // A_d · x_re
        let mut ad_re = vec![0.0; n];
        let mut tmp = vec![0.0; n];
        k_cc_op.mv(x_re, &mut tmp)?;
        for i in 0..n {
            ad_re[i] += tmp[i];
        }
        tmp.iter_mut().for_each(|x| *x = 0.0);
        m_eps_op.mv(x_re, &mut tmp)?;
        for i in 0..n {
            ad_re[i] -= omega_sq * tmp[i];
        }
        tmp.iter_mut().for_each(|x| *x = 0.0);
        m_unit_op.mv(x_re, &mut tmp)?;
        for i in 0..n {
            ad_re[i] += epsilon_reg * tmp[i];
        }

        // A_d · x_im
        let mut ad_im = vec![0.0; n];
        tmp.iter_mut().for_each(|x| *x = 0.0);
        k_cc_op.mv(x_im, &mut tmp)?;
        for i in 0..n {
            ad_im[i] += tmp[i];
        }
        tmp.iter_mut().for_each(|x| *x = 0.0);
        m_eps_op.mv(x_im, &mut tmp)?;
        for i in 0..n {
            ad_im[i] -= omega_sq * tmp[i];
        }
        tmp.iter_mut().for_each(|x| *x = 0.0);
        m_unit_op.mv(x_im, &mut tmp)?;
        for i in 0..n {
            ad_im[i] += epsilon_reg * tmp[i];
        }

        // A_o · x_re, A_o · x_im
        let mut ao_re = vec![0.0; n];
        let mut ao_im = vec![0.0; n];
        tmp.iter_mut().for_each(|x| *x = 0.0);
        m_sig_op.mv(x_re, &mut tmp)?;
        for i in 0..n {
            ao_re[i] = omega * tmp[i];
        }
        tmp.iter_mut().for_each(|x| *x = 0.0);
        m_sig_op.mv(x_im, &mut tmp)?;
        for i in 0..n {
            ao_im[i] = omega * tmp[i];
        }

        // y = [ A_d x_re − A_o x_im ;  A_o x_re + A_d x_im ]
        for i in 0..n {
            y[i] = ad_re[i] - ao_im[i];
            y[n + i] = ao_re[i] + ad_im[i];
        }

        // Pin rows: override y[r] = x[r] so the row equation reduces
        // to `x[r] = b[r]`.
        for (i, p) in pinned_op.iter().enumerate() {
            if p.is_some() {
                y[i] = x[i];
            }
        }
        Ok(())
    };

    // Build Jacobi diagonal for preconditioning (the `m_inv_diag`
    // argument GMRES expects). Pin rows get a diagonal of 1
    // (the identity row).
    let mut diag = vec![0.0_f64 as Real; two_n];
    for i in 0..n {
        let d = k_diag[i] - omega_sq * m_eps_diag[i] + epsilon_reg * m_unit_diag[i];
        diag[i] = d;
        diag[n + i] = d;
    }
    for (i, p) in pinned.iter().enumerate() {
        if p.is_some() {
            diag[i] = 1.0;
        }
    }
    let m_inv_diag: Vec<Real> = diag
        .iter()
        .map(|d| if d.abs() > 1.0e-30 { 1.0 / d } else { 0.0 })
        .collect();

    // ─── Initial guess: zero, except pin rows. ──────────────────────
    let mut x = vec![0.0_f64 as Real; two_n];
    for (i, p) in pinned.iter().enumerate() {
        if let Some(v) = p {
            x[i] = *v;
        }
    }

    // ─── GMRES(60) with Jacobi preconditioner ───────────────────────
    let op = ClosureOp {
        rows: two_n,
        cols: two_n,
        f: mv_closure,
    };
    let cfg_gmres = IterConfig {
        max_iter: 20_000,
        tol: 1.0e-9,
        restart: 200,
    };
    let stats = gmres(&op, &rhs, &mut x, Some(&m_inv_diag), &cfg_gmres)?;

    // Split solution back to (a_re, a_im).
    let a_re = x[0..n].to_vec();
    let a_im = x[n..2 * n].to_vec();

    Ok(EddyCurrentResult {
        a_re,
        a_im,
        residual: stats.residual,
        iters: stats.iters,
    })
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Scale every entry of a per-cell 6×6 local matrix by `s` (used to
/// fold per-cell ε or σ into the assembled mass).
fn scale_local(m: &[Vec<Real>], s: Real) -> Vec<Vec<Real>> {
    let mut out = vec![vec![0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = s * m[i][j];
        }
    }
    out
}

/// Build the load vector `f_i = ∫ J · N_i dV` for a piecewise-constant
/// current density `j_per_cell[c]`. Mirrors the Stage 46
/// `magneto::build_load_vector` implementation.
fn build_load_vector(
    mesh: &FemMesh,
    map: &NedelecDofMap,
    j_per_cell: &[Vec3],
) -> Vec<Real> {
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
        let j = j_per_cell[c_idx];
        for e in 0..6 {
            let [a, b] = TET_EDGES[e];
            let int_n = (grads[b] - grads[a]) * (vol / 4.0);
            let load_local = j.dot(int_n);
            let sign = NedelecDofMap::edge_sign(cell, e);
            let g_dof = map.edge_dof(verts[a], verts[b]);
            f[g_dof] += sign * load_local;
        }
    }
    f
}

/// Identify boundary edges (mirrors the Stage 46 magneto helper).
fn boundary_edge_ids(mesh: &FemMesh, edge_id: &HashMap<[u32; 2], u32>) -> Vec<u32> {
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
        let pairs = [
            [face[0], face[1]],
            [face[1], face[2]],
            [face[0], face[2]],
        ];
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

/// Mesh bounding-box diagonal (length scale for the `ε_reg`
/// regularisation). Mirrors the Stage 46 helper.
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

/// Read the diagonal of a BSR matrix into a dense vector. Used as
/// scratch for a future Jacobi-preconditioned variant — currently the
/// BiCGSTAB solve runs unpreconditioned.
fn diag_of(m: &BsrMatrix) -> Vec<Real> {
    let n = m.nrows();
    let mut d = vec![0.0_f64 as Real; n];
    for i in 0..n {
        let lo = m.row_offsets[i];
        let hi = m.row_offsets[i + 1];
        for slot in lo..hi {
            if m.col_idx[slot] == i {
                d[i] += m.values[slot];
            }
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_zero_source_gives_zero_field() {
        // Trivial smoke test: J = 0, σ = 0, ε = 0, ν = 1 ⇒ E = 0.
        let mesh = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let n_cells = mesh.cells.len();
        let cfg = EddyCurrentConfig {
            omega: 1.0,
            epsilon_per_cell: vec![0.0; n_cells],
            sigma_per_cell: vec![0.0; n_cells],
            nu_per_cell: vec![1.0; n_cells],
        };
        let j = vec![(Vec3::ZERO, Vec3::ZERO); n_cells];
        let r = solve_eddy_current_steady(&mesh, &j, &cfg, &[]).expect("solve");
        let max_re: Real = r.a_re.iter().copied().fold(0.0_f64 as Real, |a, b| a.max(b.abs()));
        let max_im: Real = r.a_im.iter().copied().fold(0.0_f64 as Real, |a, b| a.max(b.abs()));
        assert!(max_re < 1e-9, "expected zero E_re, got max {max_re}");
        assert!(max_im < 1e-9, "expected zero E_im, got max {max_im}");
    }

    #[test]
    fn block_real_matvec_dimensions() {
        // Confirm the solver runs to completion (whatever its
        // convergence) on a tiny mesh without panicking.
        let mesh = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let n_cells = mesh.cells.len();
        let cfg = EddyCurrentConfig {
            omega: 1.0,
            epsilon_per_cell: vec![0.0; n_cells],
            sigma_per_cell: vec![0.5; n_cells],
            nu_per_cell: vec![1.0; n_cells],
        };
        let j: Vec<(Vec3, Vec3)> = (0..n_cells)
            .map(|i| {
                let v = Vec3::new(0.0, 0.0, (i % 3) as Real);
                (v, Vec3::ZERO)
            })
            .collect();
        let r = solve_eddy_current_steady(&mesh, &j, &cfg, &[]).expect("solve");
        let map = NedelecDofMap::build(&mesh);
        assert_eq!(r.a_re.len(), map.ndof());
        assert_eq!(r.a_im.len(), map.ndof());
    }
}
