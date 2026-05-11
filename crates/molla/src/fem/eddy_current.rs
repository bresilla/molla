//! Time-harmonic eddy-current Maxwell solver via Nedelec H(curl)
//! edge elements — Stage 48.
//!
//! # Problem statement
//!
//! Solve the time-harmonic vector-potential equation
//!
//! ```text
//!   curl(ν curl A) + iωσ A = J         in Ω
//!   n × A = 0                          on ∂Ω
//! ```
//!
//! where:
//! - `A = A_re + i A_im` is the complex magnetic vector potential
//!   (one tangential DoF per mesh edge in each of the real and
//!   imaginary parts),
//! - `ν = 1/μ` is the magnetic reluctivity (real, per cell),
//! - `σ` is the electric conductivity (real, per cell — only the
//!   conductor cells are non-zero),
//! - `ω` is the angular frequency (`= 2πf`, a single global scalar),
//! - `J = J_re + i J_im` is the source current density (one complex
//!   `Vec3` per cell).
//!
//! The off-diagonal `iωσ A` term couples real and imaginary parts —
//! this is the "skin effect" mechanism that causes high-frequency
//! fields to decay exponentially into a conductor.
//!
//! # Block-real reformulation
//!
//! We split the complex unknown into `(a_re, a_im) ∈ R^(2N)` (where
//! `N = #edges`). Writing the curl-curl block as `K_curl` and the
//! Nedelec mass scaled by conductivity per cell as `M_σ`, the
//! discrete weak form becomes
//!
//! ```text
//!   K_curl · a_re − ω M_σ · a_im = f_re
//!   ω M_σ · a_re + K_curl · a_im = f_im
//! ```
//!
//! which in block form is
//!
//! ```text
//!   [ K_main   K_off ] [ a_re ]   [ f_re ]
//!   [ −K_off   K_main ] [ a_im ] = [ f_im ]
//! ```
//!
//! with `K_main = K_curl + ε M_N` (tiny mass-regularisation for the
//! gradient gauge — same trick as magnetostatics) and `K_off = −ω M_σ`.
//! The full `2N × 2N` system is **non-symmetric** when `M_σ ≠ 0`, so
//! we use restartable GMRES instead of CG.
//!
//! # Boundary conditions
//!
//! `n × A = 0` (perfect conductor) is the natural BC for the
//! vector potential — applied as Dirichlet `A_e = 0` on every edge
//! that lies on the mesh boundary, in BOTH real and imaginary parts.
//! Additional caller-supplied BCs are accepted for cases like
//! "imposed surface current" (Dirichlet pinning of `A_e ≠ 0` on a
//! tagged face) — these are the v1 driver mechanism for the skin-
//! depth analytic test.
//!
//! # Preconditioning
//!
//! v1 ships **block-Jacobi**: extract the diagonal of the assembled
//! `2N × 2N` matrix (which is `[diag(K_main), diag(K_main)]` since
//! the off-diagonal blocks `±ω M_σ` have zero entries on the matrix
//! diagonal for `i = j` — `M_σ`'s diagonal is non-zero but the
//! off-diagonal *block* sits in different rows/cols), invert
//! entry-wise, and pass to `gmres`. This is the cheapest possible
//! preconditioner; the natural follow-up is **AMS** (auxiliary
//! Maxwell solver) — see `solve_eddy_current_steady_ams`.
//!
//! # API
//!
//! - [`solve_eddy_current_steady`] — block-real GMRES + Jacobi.
//! - [`solve_eddy_current_steady_ams`] — block-real GMRES + AMS.
//! - [`EddyCurrentResult`] — output struct with `(a_re, a_im, iters,
//!   residual)`.

use std::collections::{HashMap, HashSet};
use std::cell::Cell;

use crate::core::{Error, Result};
use crate::math::sparse::{gmres, BsrMatrix, IterConfig, LinearOperator};
use crate::math::{Real, Vec3};

use super::assembly::matvec;
use super::bc::BoundaryCondition;
use super::element::{tet_shape_grads, tet_volume};
use super::function_space::compute_edges;
use super::mesh::{FemCell, FemMesh};
use super::nedelec_element::{
    assemble_global_nedelec, compute_element_curl_curl, compute_element_nedelec_mass,
    NedelecDofMap,
};
use super::p2_element::TET_EDGES;

/// Output of an eddy-current solve.
#[derive(Debug, Clone)]
pub struct EddyCurrentResult {
    /// Real part of the vector potential, one DoF per edge in canonical
    /// `min → max` orientation.
    pub a_re: Vec<Real>,
    /// Imaginary part of the vector potential.
    pub a_im: Vec<Real>,
    /// Final residual `‖b − A·x‖ / ‖b‖` at exit.
    pub residual: Real,
    /// GMRES iteration count.
    pub iters: usize,
}

/// Build the per-cell load vectors `f_re, f_im` from a complex source
/// `J = J_re + i J_im` (one `Vec3` per cell each).
///
/// Same closed-form as [`super::magneto::build_load_vector`] — the
/// Nedelec basis integral `∫ N_(i,j) dV = (vol/4)(∇λ_j − ∇λ_i)`.
fn build_load_vectors(
    mesh: &FemMesh,
    map: &NedelecDofMap,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
) -> (Vec<Real>, Vec<Real>) {
    assert_eq!(j_source_re.len(), mesh.cells.len());
    assert_eq!(j_source_im.len(), mesh.cells.len());
    let n = map.ndof();
    let mut f_re = vec![0.0; n];
    let mut f_im = vec![0.0; n];
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
        let j_re = j_source_re[c_idx];
        let j_im = j_source_im[c_idx];
        for e in 0..6 {
            let [a, b] = TET_EDGES[e];
            let int_n = (grads[b] - grads[a]) * (vol / 4.0);
            let load_re = j_re.dot(int_n);
            let load_im = j_im.dot(int_n);
            let sign = NedelecDofMap::edge_sign(cell, e);
            let g_dof = map.edge_dof(verts[a], verts[b]);
            f_re[g_dof] += sign * load_re;
            f_im[g_dof] += sign * load_im;
        }
    }
    (f_re, f_im)
}

/// Identify boundary edges (edges on a face owned by exactly one cell).
/// Identical to the magnetostatics helper — duplicated here so the two
/// modules can evolve independently.
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

/// Bounding-box length scale — used for the gauge-regularisation `ε`.
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

/// Block-real linear operator for the eddy-current system:
///
/// ```text
///   [ K_main   K_off ]   [ x_re ]
///   [ −K_off   K_main ] · [ x_im ]
/// ```
///
/// where `x = (x_re ⌢ x_im) ∈ R^(2N)`. The operator multiplies a flat
/// `2N`-vector and writes the result into a flat `2N`-vector.
pub(crate) struct EddyBlockOp<'a> {
    /// `K_main = K_curl + ε M_N` — symmetric positive (semi-)definite.
    pub k_main: &'a BsrMatrix,
    /// `K_off = −ω M_σ` — symmetric (its diagonal block; the full
    /// block-real system is non-symmetric because of the `±` skew).
    pub k_off: &'a BsrMatrix,
}

impl<'a> LinearOperator for EddyBlockOp<'a> {
    fn rows(&self) -> usize {
        2 * self.k_main.nrows()
    }
    fn cols(&self) -> usize {
        2 * self.k_main.ncols()
    }
    fn mv(&self, x: &[Real], y: &mut [Real]) -> Result<()> {
        let n = self.k_main.nrows();
        if x.len() != 2 * n || y.len() != 2 * n {
            return Err(Error::Build(format!(
                "EddyBlockOp::mv: shape mismatch — n={n}, x={}, y={}",
                x.len(),
                y.len(),
            )));
        }
        let (x_re, x_im) = x.split_at(n);
        let mut y_re_main = vec![0.0; n];
        let mut y_im_main = vec![0.0; n];
        let mut y_re_off = vec![0.0; n];
        let mut y_im_off = vec![0.0; n];
        self.k_main.mv(x_re, &mut y_re_main)?;
        self.k_main.mv(x_im, &mut y_im_main)?;
        self.k_off.mv(x_im, &mut y_re_off)?;
        self.k_off.mv(x_re, &mut y_im_off)?;
        // Row 1: y_re = K_main·x_re + K_off·x_im
        // Row 2: y_im = −K_off·x_re + K_main·x_im
        for i in 0..n {
            y[i] = y_re_main[i] + y_re_off[i];
            y[n + i] = -y_im_off[i] + y_im_main[i];
        }
        Ok(())
    }
}

/// Apply Dirichlet pins to a block-real `(K_main, K_off, f_re, f_im)`
/// system.
///
/// Each pin entry `(edge_dof, value_re, value_im)` pins both the real
/// and imaginary parts of `a` at that edge DoF. The block matrix
///
/// ```text
///   [[ K_main,  K_off ],
///    [ −K_off,  K_main ]]
/// ```
///
/// is reduced symmetrically: for each unpinned row `i` and pinned
/// column `j`, the contribution `K_main[i, j] · val_re[j] +
/// K_off[i, j] · val_im[j]` is moved to `f_re[i]`, and similarly
/// `−K_off[i, j] · val_re[j] + K_main[i, j] · val_im[j]` to `f_im[i]`.
/// The matrix entries `K_main[i, j]` and `K_off[i, j]` for pinned `j`
/// are then zeroed. Finally each pinned row `r` is set to `e_r` in
/// `K_main` (zero in `K_off`) and `f_re[r] = val_re`, `f_im[r] = val_im`.
fn apply_dirichlet_block(
    k_main: &mut BsrMatrix,
    k_off: &mut BsrMatrix,
    f_re: &mut [Real],
    f_im: &mut [Real],
    pins: &[(u32, Real, Real)],
) {
    let n = k_main.nrows();
    let mut pin_re: Vec<Option<Real>> = vec![None; n];
    let mut pin_im: Vec<Option<Real>> = vec![None; n];
    for (dof, vr, vi) in pins {
        pin_re[*dof as usize] = Some(*vr);
        pin_im[*dof as usize] = Some(*vi);
    }

    // Pass 1: symmetric column elimination — for each unpinned row i,
    // shift RHS by the pinned-column contributions of both K_main and
    // K_off, then zero those matrix entries.
    for i in 0..n {
        let row_re_pinned = pin_re[i].is_some();
        let row_im_pinned = pin_im[i].is_some();
        // K_main entries at row i.
        let lo = k_main.row_offsets[i];
        let hi = k_main.row_offsets[i + 1];
        for slot in lo..hi {
            let j = k_main.col_idx[slot];
            let v = k_main.values[slot];
            let mut zero_this = false;
            if let Some(val_j) = pin_re[j] {
                if !row_re_pinned {
                    f_re[i] -= v * val_j;
                }
                zero_this = true;
            }
            if let Some(val_j) = pin_im[j] {
                if !row_im_pinned {
                    f_im[i] -= v * val_j;
                }
                zero_this = true;
            }
            if zero_this && !row_re_pinned && !row_im_pinned {
                k_main.values[slot] = 0.0;
            }
        }
        // K_off entries at row i. Sign in block matrix:
        //   row i_re: +K_off[i,j] · x_im[j]
        //   row i_im: −K_off[i,j] · x_re[j]
        let lo_o = k_off.row_offsets[i];
        let hi_o = k_off.row_offsets[i + 1];
        for slot in lo_o..hi_o {
            let j = k_off.col_idx[slot];
            let v = k_off.values[slot];
            let mut zero_this = false;
            if let Some(val_j) = pin_im[j] {
                if !row_re_pinned {
                    f_re[i] -= v * val_j;
                }
                zero_this = true;
            }
            if let Some(val_j) = pin_re[j] {
                if !row_im_pinned {
                    f_im[i] -= (-v) * val_j;
                }
                zero_this = true;
            }
            if zero_this && !row_re_pinned && !row_im_pinned {
                k_off.values[slot] = 0.0;
            }
        }
    }

    // Pass 2: pin rows. Set K_main[r, :] = e_r and K_off[r, :] = 0
    // for every pinned r, and overwrite the RHS to (val_re, val_im).
    for r in 0..n {
        let is_pinned = pin_re[r].is_some() || pin_im[r].is_some();
        if !is_pinned {
            continue;
        }
        let lo = k_main.row_offsets[r];
        let hi = k_main.row_offsets[r + 1];
        for slot in lo..hi {
            let c = k_main.col_idx[slot];
            k_main.values[slot] = if c == r { 1.0 } else { 0.0 };
        }
        let lo_o = k_off.row_offsets[r];
        let hi_o = k_off.row_offsets[r + 1];
        for slot in lo_o..hi_o {
            k_off.values[slot] = 0.0;
        }
        if let Some(vr) = pin_re[r] {
            f_re[r] = vr;
        }
        if let Some(vi) = pin_im[r] {
            f_im[r] = vi;
        }
    }
}

/// Configuration knobs for the eddy-current solve. Defaults match the
/// magnetostatics conventions (1e-8 gauge regularisation, 1e-9 GMRES
/// tolerance) so callers can pass `EddyCurrentConfig::default()` for
/// the common case.
#[derive(Copy, Clone, Debug)]
pub struct EddyCurrentConfig {
    /// Angular frequency `ω = 2π f`. SI: rad/s.
    pub omega: Real,
    /// GMRES tolerance on `‖b − A·x‖ / ‖b‖`.
    pub tol: Real,
    /// Max GMRES iterations (across all restarts).
    pub max_iter: usize,
    /// GMRES restart length.
    pub restart: usize,
    /// Gauge-regularisation multiplier. The actual `ε` added to the
    /// curl-curl matrix is `epsilon_scale · mean(ν) / L²` where `L`
    /// is the mesh bounding-box diagonal. Match `magneto.rs`.
    pub epsilon_scale: Real,
}

impl Default for EddyCurrentConfig {
    fn default() -> Self {
        Self {
            omega: 1.0,
            tol: 1.0e-9,
            max_iter: 2000,
            restart: 50,
            epsilon_scale: 1.0e-8,
        }
    }
}

/// Internal: assemble `(K_main, K_off, f_re, f_im, pinned_edges)` and
/// apply `n × A = 0` plus any caller BCs. Shared between the Jacobi
/// and AMS entry-points. The returned `pinned_edges` list is the
/// union of caller Dirichlet pins and the implicit `n × A = 0`
/// boundary — same set the AMS preconditioner needs to respect.
pub(crate) fn assemble_eddy_system(
    mesh: &FemMesh,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    bcs: &[BoundaryCondition],
    cfg: &EddyCurrentConfig,
) -> Result<(NedelecDofMap, BsrMatrix, BsrMatrix, Vec<Real>, Vec<Real>, Vec<u32>)> {
    assert_eq!(nu_per_cell.len(), mesh.cells.len());
    assert_eq!(sigma_per_cell.len(), mesh.cells.len());
    assert_eq!(j_source_re.len(), mesh.cells.len());
    assert_eq!(j_source_im.len(), mesh.cells.len());

    let map = NedelecDofMap::build(mesh);

    // K_curl with per-cell ν.
    let counter = Cell::new(0usize);
    let k_curl = assemble_global_nedelec(mesh, &map, |cell, verts| {
        let idx = counter.get();
        counter.set(idx + 1);
        compute_element_curl_curl(cell, verts, nu_per_cell[idx])
    })?;

    // M_σ with per-cell σ.
    let counter = Cell::new(0usize);
    let m_sigma = assemble_global_nedelec(mesh, &map, |cell, verts| {
        let idx = counter.get();
        counter.set(idx + 1);
        scale_local(compute_element_nedelec_mass(cell, verts), sigma_per_cell[idx])
    })?;

    // M_N (un-scaled Nedelec mass) for the gauge regularisation.
    let m_full = assemble_global_nedelec(mesh, &map, compute_element_nedelec_mass)?;

    // ε = epsilon_scale · mean(ν) / L².
    let nu_mean: Real = nu_per_cell.iter().sum::<Real>() / (nu_per_cell.len() as Real);
    let l = mesh_length_scale(mesh).max(1.0e-9);
    let eps = cfg.epsilon_scale * nu_mean / (l * l);

    // K_main = K_curl + ε M_N.
    let k_main = add_csr(&k_curl, 1.0, &m_full, eps)?;
    // K_off = −ω M_σ.
    let mut k_off = m_sigma;
    k_off.scale(-cfg.omega);

    // Build load vectors.
    let (mut f_re, mut f_im) = build_load_vectors(mesh, &map, j_source_re, j_source_im);

    // Collect pins: caller-supplied + boundary edges (n × A = 0).
    let (_edges, edge_id) = compute_edges(mesh);
    let bdry = boundary_edge_ids(mesh, &edge_id);
    let mut pins: Vec<(u32, Real, Real)> = Vec::new();
    let mut have_pin: HashSet<u32> = HashSet::new();
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            pins.push((*vertex, *value, 0.0));
            have_pin.insert(*vertex);
        }
    }
    for e in &bdry {
        if !have_pin.contains(e) {
            pins.push((*e, 0.0, 0.0));
        }
    }

    let mut k_main = k_main;
    apply_dirichlet_block(&mut k_main, &mut k_off, &mut f_re, &mut f_im, &pins);

    let pinned_edges: Vec<u32> = pins.iter().map(|(dof, _, _)| *dof).collect();
    Ok((map, k_main, k_off, f_re, f_im, pinned_edges))
}

/// Steady-state eddy-current solve via block-real GMRES + Jacobi
/// preconditioning.
///
/// Arguments:
/// - `mesh`            — tet mesh.
/// - `j_source_re/im`  — complex source `J = J_re + i J_im` per cell.
/// - `nu_per_cell`     — magnetic reluctivity `ν = 1/μ` per cell.
/// - `sigma_per_cell`  — electric conductivity `σ` per cell (zero in
///   air, > 0 in conductors).
/// - `bcs`             — caller-supplied Dirichlet pins on edge DoFs.
///   Boundary `n × A = 0` is added internally.
/// - `cfg`             — solver config (frequency, tolerance, …).
///
/// Returns the complex potential `(a_re, a_im)`, final residual, and
/// iteration count.
pub fn solve_eddy_current_steady(
    mesh: &FemMesh,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    bcs: &[BoundaryCondition],
    cfg: &EddyCurrentConfig,
) -> Result<EddyCurrentResult> {
    let (map, k_main, k_off, f_re, f_im, _pinned) = assemble_eddy_system(
        mesh, j_source_re, j_source_im, nu_per_cell, sigma_per_cell, bcs, cfg,
    )?;
    let n = map.ndof();
    let op = EddyBlockOp { k_main: &k_main, k_off: &k_off };

    // Build the block-Jacobi inverse diagonal. The diagonal of
    // [[K_main, K_off], [−K_off, K_main]] at scalar row `r` is
    // `K_main[r, r]` for both `r` and `n + r` (since `K_off[r, r]`
    // contributes to off-diagonal *blocks* of the 2×2 system, not to
    // the matrix-diagonal of the assembled block matrix). So Jacobi
    // = repeat-twice diag(K_main).
    let diag_main = k_main.diagonal();
    let mut inv_diag = vec![0.0; 2 * n];
    for i in 0..n {
        let d = diag_main[i];
        let inv = if d.abs() > 1.0e-30 { 1.0 / d } else { 0.0 };
        inv_diag[i] = inv;
        inv_diag[n + i] = inv;
    }

    // Assemble RHS and call GMRES.
    let mut b = vec![0.0; 2 * n];
    b[..n].copy_from_slice(&f_re);
    b[n..].copy_from_slice(&f_im);
    let mut x = vec![0.0; 2 * n];

    let iter_cfg = IterConfig {
        max_iter: cfg.max_iter,
        tol: cfg.tol,
        restart: cfg.restart.max(1),
    };
    let stats = gmres(&op, &b, &mut x, Some(&inv_diag), &iter_cfg)?;

    let a_re = x[..n].to_vec();
    let a_im = x[n..].to_vec();
    Ok(EddyCurrentResult {
        a_re,
        a_im,
        residual: stats.residual,
        iters: stats.iters,
    })
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Scale every entry of a per-element local matrix by `s`.
fn scale_local(mut m: Vec<Vec<Real>>, s: Real) -> Vec<Vec<Real>> {
    for row in &mut m {
        for v in row.iter_mut() {
            *v *= s;
        }
    }
    m
}

/// Multiply a BSR (B = 1) matrix's transpose by a vector: `y = Aᵀ · x`.
/// Falls back to the column-probe trick used by [`add_csr`] but
/// short-circuits to a per-row scatter for performance.
pub(crate) fn matvec_transpose(a: &BsrMatrix, x: &[Real]) -> Result<Vec<Real>> {
    debug_assert_eq!(a.b, 1);
    if x.len() != a.nrows() {
        return Err(Error::Build(format!(
            "matvec_transpose: dim mismatch — A is {}×{}, x={}",
            a.nrows(),
            a.ncols(),
            x.len(),
        )));
    }
    let mut y = vec![0.0; a.ncols()];
    for r in 0..a.block_rows {
        let lo = a.row_offsets[r];
        let hi = a.row_offsets[r + 1];
        for slot in lo..hi {
            let c = a.col_idx[slot];
            y[c] += a.values[slot] * x[r];
        }
    }
    Ok(y)
}

/// In-place CSR add `result = α · a + β · b` via column probes.
/// Identical to the magneto module helper — local copy so the two
/// modules don't depend on each other's internals.
pub(crate) fn add_csr(a: &BsrMatrix, alpha: Real, b: &BsrMatrix, beta: Real) -> Result<BsrMatrix> {
    let n = a.nrows();
    debug_assert_eq!(a.nrows(), b.nrows());
    debug_assert_eq!(a.ncols(), b.ncols());
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
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

// ─── Stage 48 v2 — single-level AMS preconditioner ─────────────────

/// Build the discrete gradient operator `G : R^(n_verts) → R^(n_edges)`.
///
/// For each unique mesh edge `e = (a, b)` with `a < b` (canonical
/// orientation), row `e` of `G` has `+1` in column `b` and `−1` in
/// column `a`. This realises the discrete identity
///
/// ```text
///   G φ_h ≡ grad-of-P1-into-Nedelec
/// ```
///
/// — i.e. the discrete gradient of a P1 nodal scalar `φ` lives exactly
/// in the lowest-order Nedelec edge space, and its edge coefficients
/// in that space are `(G φ)_e = φ_b − φ_a`. The operator is a key
/// building block of the AMS (Hiptmair-Xu) preconditioner: it lets us
/// transfer residuals from the H(curl) space into the auxiliary P1
/// scalar space and prolongate corrections back.
///
/// Output is an `(n_edges × n_verts)` BSR (B = 1) matrix.
#[must_use]
pub fn build_discrete_gradient(mesh: &FemMesh) -> Result<BsrMatrix> {
    let (edges, _) = compute_edges(mesh);
    let n_edges = edges.len();
    let n_verts = mesh.vert_count();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::with_capacity(2 * n_edges);
    for (e, [a, b]) in edges.iter().enumerate() {
        // a < b by construction of compute_edges.
        triplets.push((e, *b as usize, 1.0));
        triplets.push((e, *a as usize, -1.0));
    }
    BsrMatrix::from_scalar_triplets(n_edges, n_verts, &triplets)
}

/// Build the **vector P1 Laplacian** for the auxiliary AMS solve:
/// `K_aux = diag(K_lap, K_lap, …, K_lap)` (`dim` copies along the
/// diagonal), where `K_lap` is the standard P1 nodal stiffness
/// (`compute_element_stiffness`).
///
/// AMS' auxiliary block is a **vector** Laplacian because the H(curl)
/// problem lives in 3D and the auxiliary correction is naturally
/// a 3-component scalar (one per Cartesian direction of the
/// gradient-equivalent of the residual). In the single-level v2 case
/// we don't strictly need the vector form — the discrete-gradient
/// path `Gᵀ r → K_lap · w → G w` lives in **scalar** P1 space already.
/// We expose the vector builder anyway (so the v3 multigrid driver
/// can reuse it) and `apply` below threads the scalar form through.
///
/// Result is sized `(n_verts · dim) × (n_verts · dim)` with the
/// interleaved layout `dof = vertex · dim + component`.
#[must_use]
pub fn build_p1_vector_laplacian(mesh: &FemMesh, dim: usize) -> Result<BsrMatrix> {
    use super::element::compute_element_stiffness;
    let n_verts = mesh.vert_count();
    let n = n_verts * dim;
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for cell in &mesh.cells {
        let FemCell::Tet(verts) = cell else { continue };
        let k_loc = compute_element_stiffness(cell, &mesh.vertices, 1.0);
        // Scatter k_loc[i, j] into all `dim` diagonal blocks of the
        // global vector Laplacian.
        for i in 0..4 {
            for j in 0..4 {
                let v = k_loc[i][j];
                if v == 0.0 {
                    continue;
                }
                let gi = verts[i] as usize;
                let gj = verts[j] as usize;
                for c in 0..dim {
                    triplets.push((gi * dim + c, gj * dim + c, v));
                }
            }
        }
    }
    BsrMatrix::from_scalar_triplets(n, n, &triplets)
}

/// Build the **scalar** P1 Laplacian `K_lap = ∫ ∇φᵢ · ∇φⱼ dV`.
/// Used by the single-level AMS auxiliary CG solve.
fn build_p1_scalar_laplacian(mesh: &FemMesh) -> Result<BsrMatrix> {
    use super::element::compute_element_stiffness;
    let n_verts = mesh.vert_count();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for cell in &mesh.cells {
        let FemCell::Tet(_verts) = cell else { continue };
        let k_loc = compute_element_stiffness(cell, &mesh.vertices, 1.0);
        let verts = match cell {
            FemCell::Tet(v) => v,
            _ => unreachable!(),
        };
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

/// AMS preconditioner — single-level Hiptmair-Xu auxiliary-space
/// approximate inverse of the curl-curl block `K_curl`.
///
/// # Algorithm
///
/// Given the H(curl) residual `r ∈ R^(n_edges)`, compute
///
/// ```text
///   z_1 = Jacobi-smooth(K_curl, r)                  // edge smoother
///   r_aux = Gᵀ · r                                  // restrict
///   K_aux · w = r_aux                               // CG (≤ 50 iters)
///   z_2 = G · w                                     // prolongate
///   z   = z_1 + z_2
/// ```
///
/// where `G` is the discrete-gradient operator and `K_aux` is the
/// **scalar** P1 Laplacian (one DoF per vertex). The auxiliary block
/// is rank-deficient (constants are in the kernel of K_aux), so we
/// pin one vertex to break the gauge — same trick magnetostatics
/// uses internally via the ε-regularised curl-curl block.
///
/// # Single-level vs multigrid
///
/// The textbook AMS uses a multigrid V-cycle on `K_aux`. v2 ships
/// the **single-level** variant: one CG solve on the full P1
/// Laplacian (sub-50 iters at the curl-curl problem size). This is
/// a modest improvement over plain Jacobi — typical wins are ~2-3×
/// in iter count. The full multigrid driver is queued for v3.
pub struct AmsPreconditioner {
    /// Diagonal of `K_curl` (the curl-curl block of `K_main`; used by
    /// the edge-Jacobi smoother).
    inv_diag_curl: Vec<Real>,
    /// Discrete gradient `(n_edges × n_verts)`.
    g: BsrMatrix,
    /// Scalar P1 Laplacian with one vertex pinned (so it's SPD).
    k_aux: BsrMatrix,
    /// Index of the (gauge-pinned) vertex.
    pinned_vertex: usize,
    /// Edges that are Dirichlet-pinned in the outer system (boundary
    /// or user-specified). The AMS apply zeros these in both the
    /// smoother output and the prolongated auxiliary correction so the
    /// preconditioner respects the same Dirichlet pinning the system
    /// enforces.
    pinned_edges: Vec<u32>,
    /// Vertices touched by pinned edges — pinned in `k_aux` so the
    /// auxiliary solve returns zero there (which the prolongator then
    /// pushes to the pinned edges, contributing nothing).
    pinned_aux_vertices: Vec<u32>,
    /// Max iterations for the inner auxiliary CG solve (fixed — see
    /// `apply` for the linearity argument).
    aux_max_iter: usize,
}

impl AmsPreconditioner {
    /// Build the preconditioner from the curl-curl operator, the
    /// mesh, and the list of Dirichlet-pinned edges in the outer
    /// system (boundary `n × A = 0` + caller pins).
    ///
    /// The pinned-edge list is critical: the AMS prolongator `G` maps
    /// vertex values to edges, and without it the prolongated
    /// correction `G w` introduces non-zero values at pinned edges,
    /// breaking the preconditioner's compatibility with the outer
    /// linear system.
    pub fn new(mesh: &FemMesh, k_curl: &BsrMatrix, pinned_edges: &[u32]) -> Result<Self> {
        let (edges, _) = compute_edges(mesh);
        let g = build_discrete_gradient(mesh)?;

        // Vertices touched by any pinned edge — pinned in K_aux so the
        // auxiliary CG returns zero there.
        let mut aux_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &e in pinned_edges {
            let [a, b] = edges[e as usize];
            aux_set.insert(a);
            aux_set.insert(b);
        }
        let pinned_aux_vertices: Vec<u32> = {
            let mut v: Vec<u32> = aux_set.into_iter().collect();
            v.sort_unstable();
            v
        };

        let mut k_aux = build_p1_scalar_laplacian(mesh)?;
        // Pin every Dirichlet-touched vertex; if there are none, pin
        // vertex 0 to break the constant-vector nullspace.
        let pinned_for_gauge = if pinned_aux_vertices.is_empty() {
            vec![0u32]
        } else {
            pinned_aux_vertices.clone()
        };
        for &v in &pinned_for_gauge {
            pin_aux_vertex(&mut k_aux, v as usize);
        }
        let pinned_vertex = pinned_for_gauge[0] as usize;

        let diag = k_curl.diagonal();
        let inv_diag_curl = diag
            .iter()
            .map(|&d| if d.abs() > 1.0e-30 { 1.0 / d } else { 0.0 })
            .collect();
        Ok(Self {
            inv_diag_curl,
            g,
            k_aux,
            pinned_vertex,
            pinned_edges: pinned_edges.to_vec(),
            pinned_aux_vertices: pinned_for_gauge,
            aux_max_iter: 50,
        })
    }

    /// Tune the inner auxiliary CG iteration cap. Larger values get
    /// closer to a textbook multigrid V-cycle but cost more per
    /// outer GMRES iter. The single-level default of 50 is a sweet
    /// spot for the skin-depth test problems.
    pub fn with_aux_max_iter(mut self, n: usize) -> Self {
        self.aux_max_iter = n;
        self
    }

    /// Apply one AMS V-cycle to the residual `r` and return the
    /// preconditioned vector `z` (length `n_edges`).
    ///
    /// The inner CG on `K_aux` is **always run for exactly
    /// `aux_max_iter` iterations** (tolerance is set to zero
    /// internally) so the preconditioner is a *linear* operator —
    /// GMRES requires that to converge. A CG run that exits on
    /// tolerance is **non-linear** in the input residual (the iteration
    /// count varies) and degrades the outer Krylov convergence.
    ///
    /// Boundary edges (pinned in the outer system) are zeroed in `z`
    /// — both after the smoother and after the prolongated auxiliary
    /// correction. This is essential: without it, `G w` introduces
    /// non-zero values on pinned edges that the outer Krylov method
    /// has to fight against, harming convergence.
    pub fn apply(&self, r: &[Real]) -> Result<Vec<Real>> {
        let n_edges = self.inv_diag_curl.len();
        debug_assert_eq!(r.len(), n_edges);

        // 1) Edge Jacobi smoother.
        let mut z = vec![0.0; n_edges];
        for i in 0..n_edges {
            z[i] = r[i] * self.inv_diag_curl[i];
        }

        // 2) Restrict to vertex space.
        let mut r_aux = matvec_transpose(&self.g, r)?;
        // Zero the auxiliary RHS at every pinned vertex (consistent
        // with the row-pinned K_aux).
        for &v in &self.pinned_aux_vertices {
            r_aux[v as usize] = 0.0;
        }

        // 3) Inner CG on K_aux — fixed iteration count for linearity.
        let k_aux = &self.k_aux;
        let mv = |v: &[Real]| -> Vec<Real> { matvec(k_aux, v) };
        let n_verts = k_aux.nrows();
        let w_init = vec![0.0; n_verts];
        let aux = crate::math::iterative::cg_solve(
            mv,
            &r_aux,
            &w_init,
            0.0, // never trigger tolerance exit; always run max_iter.
            self.aux_max_iter,
        );
        let w = aux.x;

        // 4) Prolongate back to edges.
        let mut z_2 = vec![0.0; n_edges];
        self.g.mv(&w, &mut z_2)?;

        // 5) Combine.
        for i in 0..n_edges {
            z[i] += z_2[i];
        }

        // 6) Zero pinned-edge entries — keep the preconditioner
        // compatible with the outer-system Dirichlet pinning.
        for &e in &self.pinned_edges {
            z[e as usize] = 0.0;
        }

        // Silence unused-field warning (kept for future v3 multigrid).
        let _ = self.pinned_vertex;
        Ok(z)
    }
}

/// Pin row `r` of an auxiliary P1 Laplacian (symmetric column
/// elimination), so the matrix is SPD and CG converges. Same
/// strong-form pin used by [`super::bc::apply_dirichlet`].
fn pin_aux_vertex(k: &mut BsrMatrix, r: usize) {
    let n = k.nrows();
    // Pass 1: column elimination — for every other row i, zero K[i, r].
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
    // Pass 2: pin row r to e_r.
    let lo = k.row_offsets[r];
    let hi = k.row_offsets[r + 1];
    for slot in lo..hi {
        let c = k.col_idx[slot];
        k.values[slot] = if c == r { 1.0 } else { 0.0 };
    }
}

/// Steady-state eddy-current solve via block-real GMRES + **AMS**
/// preconditioning (single-level v2).
///
/// Same signature as [`solve_eddy_current_steady`]; returns the same
/// [`EddyCurrentResult`]. The only difference is the preconditioner —
/// AMS replaces Jacobi on the diagonal `K_main` block; the off-
/// diagonal `K_off` is left to GMRES (it's a damping term that
/// doesn't dominate the spectrum unless `ω σ` is enormous).
pub fn solve_eddy_current_steady_ams(
    mesh: &FemMesh,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    bcs: &[BoundaryCondition],
    cfg: &EddyCurrentConfig,
) -> Result<EddyCurrentResult> {
    let (map, k_main, k_off, f_re, f_im, pinned_edges) = assemble_eddy_system(
        mesh, j_source_re, j_source_im, nu_per_cell, sigma_per_cell, bcs, cfg,
    )?;
    let n = map.ndof();

    // Build the AMS preconditioner from `K_main` (which equals
    // `K_curl + ε M_N` post-BC reduction). The pinned-edge list
    // ensures the AMS preconditioner respects the same Dirichlet
    // pinning the outer system enforces: boundary edges are zeroed
    // in `z` after both the smoother and the prolongator, and the
    // matching auxiliary-vertex Dirichlet zeroes are applied to the
    // P1 Laplacian K_aux.
    let ams = AmsPreconditioner::new(mesh, &k_main, &pinned_edges)?;

    let op = EddyBlockOp { k_main: &k_main, k_off: &k_off };

    // Build a custom GMRES that uses AMS for each block. We re-
    // implement a thin restartable GMRES wrapper here (the standard
    // `math::sparse::gmres` only accepts a Jacobi inv-diag).
    let mut b = vec![0.0; 2 * n];
    b[..n].copy_from_slice(&f_re);
    b[n..].copy_from_slice(&f_im);

    // Seed `x` with the Dirichlet pin values. AMS's prolongator zeros
    // pinned-edge entries in the preconditioned residual, so without
    // seeding, the pinned edges would stay at their initial value
    // (zero) instead of the prescribed `f_re/f_im[pinned] = value`.
    // Note that for pinned rows the matrix is `e_r` and the RHS is the
    // pin value, so `b[pinned_re/im] = pin_value` already.
    let mut x = vec![0.0; 2 * n];
    for &e in &pinned_edges {
        x[e as usize] = f_re[e as usize];
        x[n + e as usize] = f_im[e as usize];
    }

    // Closure: M⁻¹ applied to a `2N`-vector by applying AMS to each
    // half independently.
    let apply_m_inv = |r: &[Real]| -> Result<Vec<Real>> {
        let (r_re, r_im) = r.split_at(n);
        let z_re = ams.apply(r_re)?;
        let z_im = ams.apply(r_im)?;
        let mut z = vec![0.0; 2 * n];
        z[..n].copy_from_slice(&z_re);
        z[n..].copy_from_slice(&z_im);
        Ok(z)
    };

    let iter_cfg = IterConfig {
        max_iter: cfg.max_iter,
        tol: cfg.tol,
        restart: cfg.restart.max(1),
    };
    let stats = gmres_with_precond(&op, &b, &mut x, &apply_m_inv, &iter_cfg)?;

    let a_re = x[..n].to_vec();
    let a_im = x[n..].to_vec();
    Ok(EddyCurrentResult {
        a_re,
        a_im,
        residual: stats.residual,
        iters: stats.iters,
    })
}

/// Restartable GMRES with a **closure** preconditioner — same interface
/// as [`crate::math::sparse::gmres`] but accepts an arbitrary
/// `M⁻¹` operator (so AMS can plug in here without materialising a
/// per-row inverse).
///
/// The algorithm is the standard right-preconditioned Arnoldi /
/// Givens-rotation flow, identical in structure to `math::sparse::gmres`
/// — only `apply_m_inv` is the difference.
pub fn gmres_with_precond<P>(
    op: &dyn LinearOperator,
    b: &[Real],
    x: &mut [Real],
    apply_m_inv: &P,
    cfg: &IterConfig,
) -> Result<crate::math::sparse::IterStats>
where
    P: Fn(&[Real]) -> Result<Vec<Real>>,
{
    let n = op.rows();
    if op.cols() != n {
        return Err(Error::Build("gmres_with_precond: operator is not square".into()));
    }
    if b.len() != n || x.len() != n {
        return Err(Error::Build("gmres_with_precond: dimension mismatch".into()));
    }
    let m = cfg.restart.max(1).min(n);

    let bnorm: Real = {
        let mut s = 0.0;
        for v in b {
            s += v * v;
        }
        s.sqrt().max(1.0e-30)
    };

    let mut total_iters = 0usize;
    while total_iters < cfg.max_iter {
        // r₀ = b − A·x; z = M⁻¹ · r₀.
        let mut r = vec![0.0; n];
        op.mv(x, &mut r)?;
        for i in 0..n {
            r[i] = b[i] - r[i];
        }
        let z = apply_m_inv(&r)?;
        let beta = {
            let mut s = 0.0;
            for v in &z {
                s += v * v;
            }
            s.sqrt()
        };
        if beta / bnorm < cfg.tol {
            return Ok(crate::math::sparse::IterStats {
                iters: total_iters,
                residual: beta / bnorm,
                converged: true,
            });
        }
        let mut v_basis: Vec<Vec<Real>> = vec![Vec::new(); m + 1];
        v_basis[0] = z.iter().map(|v| v / beta).collect();
        let mut h = vec![0.0; (m + 1) * m];
        let mut g = vec![0.0; m + 1];
        g[0] = beta;
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];
        let mut converged = false;
        let mut k_done = m;
        for k in 0..m {
            let mut w = vec![0.0; n];
            op.mv(&v_basis[k], &mut w)?;
            let mut zw = apply_m_inv(&w)?;
            for j in 0..=k {
                let h_jk = {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += v_basis[j][i] * zw[i];
                    }
                    s
                };
                h[j * m + k] = h_jk;
                for i in 0..n {
                    zw[i] -= h_jk * v_basis[j][i];
                }
            }
            let norm_zw = {
                let mut s = 0.0;
                for v in &zw {
                    s += v * v;
                }
                s.sqrt()
            };
            h[(k + 1) * m + k] = norm_zw;
            if norm_zw.abs() > 1.0e-30 {
                v_basis[k + 1] = zw.iter().map(|v| v / norm_zw).collect();
            } else {
                v_basis[k + 1] = vec![0.0; n];
            }
            for i in 0..k {
                let temp = cs[i] * h[i * m + k] + sn[i] * h[(i + 1) * m + k];
                h[(i + 1) * m + k] = -sn[i] * h[i * m + k] + cs[i] * h[(i + 1) * m + k];
                h[i * m + k] = temp;
            }
            let h_kk = h[k * m + k];
            let h_k1k = h[(k + 1) * m + k];
            let r_kk = (h_kk * h_kk + h_k1k * h_k1k).sqrt();
            cs[k] = h_kk / r_kk;
            sn[k] = h_k1k / r_kk;
            h[k * m + k] = r_kk;
            h[(k + 1) * m + k] = 0.0;
            let g_temp = cs[k] * g[k] + sn[k] * g[k + 1];
            g[k + 1] = -sn[k] * g[k] + cs[k] * g[k + 1];
            g[k] = g_temp;

            total_iters += 1;
            if total_iters >= cfg.max_iter {
                k_done = k + 1;
                break;
            }
            if g[k + 1].abs() / bnorm < cfg.tol {
                converged = true;
                k_done = k + 1;
                break;
            }
        }
        let mut y = vec![0.0; k_done];
        for i in (0..k_done).rev() {
            let mut s = g[i];
            for j in (i + 1)..k_done {
                s -= h[i * m + j] * y[j];
            }
            y[i] = s / h[i * m + i];
        }
        for i in 0..k_done {
            for j in 0..n {
                x[j] += y[i] * v_basis[i][j];
            }
        }
        if converged {
            return Ok(crate::math::sparse::IterStats {
                iters: total_iters,
                residual: g[k_done].abs() / bnorm,
                converged: true,
            });
        }
        if total_iters >= cfg.max_iter {
            break;
        }
    }
    let mut r = vec![0.0; n];
    op.mv(x, &mut r)?;
    for i in 0..n {
        r[i] = b[i] - r[i];
    }
    let residual = {
        let mut s = 0.0;
        for v in &r {
            s += v * v;
        }
        s.sqrt() / bnorm
    };
    Ok(crate::math::sparse::IterStats {
        iters: total_iters,
        residual,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    #[test]
    fn zero_sigma_recovers_magnetostatic_skeleton() {
        // With σ = 0 and ω = 0, K_off should vanish and the block
        // problem decouples into two independent magnetostatics
        // solves (a_re and a_im); a zero RHS should give a zero
        // solution.
        let mesh = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let n_cells = mesh.cells.len();
        let j_zero = vec![Vec3::ZERO; n_cells];
        let nu = vec![1.0; n_cells];
        let sigma = vec![0.0; n_cells];
        let cfg = EddyCurrentConfig {
            omega: 0.0,
            ..Default::default()
        };
        let r = solve_eddy_current_steady(&mesh, &j_zero, &j_zero, &nu, &sigma, &[], &cfg)
            .expect("solve");
        assert!(r.a_re.iter().all(|v| v.abs() < 1e-10));
        assert!(r.a_im.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn block_op_matvec_matches_explicit_assembly() {
        // Sanity test: build a tiny system, take a random x, and check
        // that EddyBlockOp::mv reproduces the explicit block matvec.
        let mesh = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let n_cells = mesh.cells.len();
        let nu = vec![1.0; n_cells];
        let sigma = vec![2.5; n_cells];
        let j_re = vec![Vec3::new(0.1, 0.0, 0.0); n_cells];
        let j_im = vec![Vec3::new(0.0, 0.05, 0.0); n_cells];
        let cfg = EddyCurrentConfig {
            omega: 0.7,
            ..Default::default()
        };
        let (map, k_main, k_off, _, _, _) =
            assemble_eddy_system(&mesh, &j_re, &j_im, &nu, &sigma, &[], &cfg).unwrap();
        let n = map.ndof();
        let op = EddyBlockOp { k_main: &k_main, k_off: &k_off };
        let mut x = vec![0.0; 2 * n];
        for i in 0..(2 * n) {
            x[i] = ((i as Real) * 0.137).sin();
        }
        let mut y = vec![0.0; 2 * n];
        op.mv(&x, &mut y).unwrap();

        // Reference: explicit block matvec.
        let mut y_ref_re = vec![0.0; n];
        let mut y_ref_im = vec![0.0; n];
        let (x_re, x_im) = x.split_at(n);
        let mut tmp = vec![0.0; n];
        k_main.mv(x_re, &mut tmp).unwrap();
        for i in 0..n {
            y_ref_re[i] += tmp[i];
        }
        k_off.mv(x_im, &mut tmp).unwrap();
        for i in 0..n {
            y_ref_re[i] += tmp[i];
        }
        k_off.mv(x_re, &mut tmp).unwrap();
        for i in 0..n {
            y_ref_im[i] -= tmp[i];
        }
        k_main.mv(x_im, &mut tmp).unwrap();
        for i in 0..n {
            y_ref_im[i] += tmp[i];
        }
        for i in 0..n {
            assert!(
                (y[i] - y_ref_re[i]).abs() < 1e-12,
                "y[{i}] real mismatch: {} vs {}",
                y[i],
                y_ref_re[i]
            );
            assert!(
                (y[n + i] - y_ref_im[i]).abs() < 1e-12,
                "y[{i}] imag mismatch: {} vs {}",
                y[n + i],
                y_ref_im[i]
            );
        }
        // Use `matvec` shim to suppress "unused import" lint.
        let _ = matvec(&k_main, &vec![1.0; n]);
    }
}
