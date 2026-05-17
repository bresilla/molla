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

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use crate::core::{Error, Result};
use crate::math::sparse::{BsrMatrix, IterConfig, LinearOperator, gmres};
use crate::math::{Real, Vec3};

use super::amg::AmgHierarchy;
use super::assembly::matvec;
use super::bc::BoundaryCondition;
use super::element::{tet_shape_grads, tet_volume};
use super::function_space::compute_edges;
use super::mesh::{FemCell, FemMesh};
use super::multigrid::MeshHierarchy;
use super::nedelec_element::{
    NedelecDofMap, assemble_global_nedelec, compute_element_curl_curl, compute_element_nedelec_mass,
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

/// Solver variant used by [`solve_eddy_current_frequency_sweep`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EddyCurrentSolveMode {
    /// Jacobi-preconditioned block-real GMRES.
    Jacobi,
    /// AMS-preconditioned block-real GMRES.
    Ams,
    /// Block-coupled AMS-preconditioned block-real GMRES.
    AmsBlock,
}

/// One sample from an eddy-current frequency sweep.
#[derive(Debug, Clone)]
pub struct EddyFrequencySweepSample {
    /// Angular frequency used for this sample.
    pub omega: Real,
    /// Solver result at `omega`.
    pub result: EddyCurrentResult,
    /// Discrete Coulomb-gauge residual after optional projection.
    pub gauge_residual: Real,
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
pub struct EddyBlockOp<'a> {
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
    /// Fixed block-preconditioner correction sweeps for the
    /// high-`ωσ` AMS block-coupled solve. `1` is the old block-diagonal
    /// AMS action; `2+` folds the real/imaginary off-diagonal mass
    /// coupling into the preconditioner through stationary refinement.
    pub block_precond_sweeps: usize,
}

impl Default for EddyCurrentConfig {
    fn default() -> Self {
        Self {
            omega: 1.0,
            tol: 1.0e-9,
            max_iter: 2000,
            restart: 50,
            epsilon_scale: 1.0e-8,
            block_precond_sweeps: 2,
        }
    }
}

/// Assemble `(map, K_main, K_off, f_re, f_im, pinned_edges)` and
/// apply `n × A = 0` plus any caller BCs. Shared between the Jacobi
/// and AMS entry-points. The returned `pinned_edges` list is the
/// union of caller Dirichlet pins and the implicit `n × A = 0`
/// boundary — same set the AMS preconditioner needs to respect.
///
/// Public so external callers (tests, custom preconditioner
/// experiments, …) can run their own GMRES loop on the assembled
/// block system. The output triple matches the input arguments
/// expected by [`gmres_with_precond`] and [`AmsPreconditioner::new`].
pub fn assemble_eddy_system(
    mesh: &FemMesh,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    bcs: &[BoundaryCondition],
    cfg: &EddyCurrentConfig,
) -> Result<(
    NedelecDofMap,
    BsrMatrix,
    BsrMatrix,
    Vec<Real>,
    Vec<Real>,
    Vec<u32>,
)> {
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
        scale_local(
            compute_element_nedelec_mass(cell, verts),
            sigma_per_cell[idx],
        )
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
        mesh,
        j_source_re,
        j_source_im,
        nu_per_cell,
        sigma_per_cell,
        bcs,
        cfg,
    )?;
    let n = map.ndof();
    let op = EddyBlockOp {
        k_main: &k_main,
        k_off: &k_off,
    };

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

/// Build `GᵀG`, the graph-normal matrix induced by the discrete
/// Nedelec gradient. This is the least-squares Coulomb-gauge operator
/// used to remove gradient components from edge potentials.
fn build_discrete_gradient_normal(mesh: &FemMesh) -> Result<BsrMatrix> {
    let (edges, _) = compute_edges(mesh);
    let n_verts = mesh.vert_count();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::with_capacity(edges.len() * 4);
    for [a, b] in edges {
        let a = a as usize;
        let b = b as usize;
        triplets.push((a, a, 1.0));
        triplets.push((b, b, 1.0));
        triplets.push((a, b, -1.0));
        triplets.push((b, a, -1.0));
    }
    BsrMatrix::from_scalar_triplets(n_verts, n_verts, &triplets)
}

fn pinned_aux_vertices_for_edges(mesh: &FemMesh, pinned_edges: &[u32]) -> Vec<u32> {
    let (edges, _) = compute_edges(mesh);
    let mut aux_set: HashSet<u32> = HashSet::new();
    for &e in pinned_edges {
        let [a, b] = edges[e as usize];
        aux_set.insert(a);
        aux_set.insert(b);
    }
    let mut out: Vec<u32> = aux_set.into_iter().collect();
    out.sort_unstable();
    out
}

/// Compute the max-norm of the discrete Coulomb-gauge residual
/// `Gᵀa`, ignoring vertices constrained by pinned edge DoFs.
pub fn coulomb_gauge_residual(mesh: &FemMesh, a: &[Real], pinned_edges: &[u32]) -> Result<Real> {
    let g = build_discrete_gradient(mesh)?;
    let div = matvec_transpose(&g, a)?;
    let pinned_aux = pinned_aux_vertices_for_edges(mesh, pinned_edges);
    let pinned: HashSet<u32> = pinned_aux.into_iter().collect();
    let mut max_res: Real = 0.0;
    for (i, v) in div.iter().enumerate() {
        if !pinned.contains(&(i as u32)) {
            max_res = max_res.max(v.abs());
        }
    }
    Ok(max_res)
}

/// Project an edge-element potential into a discrete Coulomb gauge by
/// removing the least-squares gradient component.
///
/// The projection solves
///
/// ```text
///   (GᵀG) φ = Gᵀ a
///   a_gauge = a - Gφ
/// ```
///
/// with the scalar gauge pinned at vertices touched by `pinned_edges`
/// (or vertex 0 when there are no edge pins). Pinned edge values are
/// restored exactly after projection, so Dirichlet boundary values are
/// preserved.
pub fn project_nedelec_coulomb_gauge(
    mesh: &FemMesh,
    a: &[Real],
    pinned_edges: &[u32],
) -> Result<Vec<Real>> {
    let g = build_discrete_gradient(mesh)?;
    if a.len() != g.nrows() {
        return Err(Error::Build(format!(
            "project_nedelec_coulomb_gauge: potential length {}, expected {}",
            a.len(),
            g.nrows()
        )));
    }

    let mut gtg = build_discrete_gradient_normal(mesh)?;
    let mut rhs = matvec_transpose(&g, a)?;
    let mut pinned_aux = pinned_aux_vertices_for_edges(mesh, pinned_edges);
    if pinned_aux.is_empty() && !rhs.is_empty() {
        pinned_aux.push(0);
    }
    for &v in &pinned_aux {
        rhs[v as usize] = 0.0;
        pin_aux_vertex(&mut gtg, v as usize);
    }

    let phi0 = vec![0.0; rhs.len()];
    let phi = crate::math::iterative::cg_solve(
        |x| matvec(&gtg, x),
        &rhs,
        &phi0,
        1.0e-12,
        rhs.len().saturating_mul(8).max(64),
    )
    .x;
    let mut grad = vec![0.0; g.nrows()];
    g.mv(&phi, &mut grad)?;
    let mut out = a.to_vec();
    for i in 0..out.len() {
        out[i] -= grad[i];
    }
    for &e in pinned_edges {
        out[e as usize] = a[e as usize];
    }
    Ok(out)
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

/// Build the AMS **Π interpolation** from vector P1 nodal fields into
/// lowest-order Nedelec edge DoFs.
///
/// The output matrix has shape
/// `(n_edges × n_verts * 3)` and uses the same interleaved vector-P1
/// layout as [`build_p1_vector_laplacian`]: component `c` of vertex
/// `v` lives at column `v * 3 + c`.
///
/// For a canonical mesh edge `e = (a, b)` (`a < b`), a vector P1 field
/// varies linearly along the segment, so the line integral that defines
/// the Nedelec edge coefficient is exactly
///
/// ```text
///   (Π u)_e = ∫_a^b u_h · dl
///           = 0.5 * (u_a + u_b) · (x_b - x_a).
/// ```
///
/// This is the missing vector-nodal branch of the Hiptmair-Xu AMS
/// auxiliary-space preconditioner: `Π^T r` restricts an edge residual
/// into vector P1 space, a vector Laplacian approximately inverts that
/// component, and `Π` prolongates the correction back to H(curl).
#[must_use]
pub fn build_p1_vector_interpolation(mesh: &FemMesh) -> Result<BsrMatrix> {
    let (edges, _) = compute_edges(mesh);
    let n_edges = edges.len();
    let n_vec = mesh.vert_count() * 3;
    let mut triplets: Vec<(usize, usize, Real)> = Vec::with_capacity(n_edges * 6);
    for (e, [a, b]) in edges.iter().enumerate() {
        let pa = mesh.vertices[*a as usize];
        let pb = mesh.vertices[*b as usize];
        let d = pb - pa;
        for (c, w) in [d.x, d.y, d.z].into_iter().enumerate() {
            let val = 0.5 * w;
            if val == 0.0 {
                continue;
            }
            triplets.push((e, *a as usize * 3 + c, val));
            triplets.push((e, *b as usize * 3 + c, val));
        }
    }
    BsrMatrix::from_scalar_triplets(n_edges, n_vec, &triplets)
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
    /// Vector-P1 interpolation `(n_edges × 3*n_verts)` for the AMS Π
    /// branch.
    pi: BsrMatrix,
    /// Scalar P1 Laplacian with one vertex pinned (so it's SPD).
    k_aux: BsrMatrix,
    /// Vector P1 Laplacian with the same auxiliary vertices pinned in
    /// all three Cartesian components.
    k_aux_vector: BsrMatrix,
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
    /// `apply` for the linearity argument). Ignored when
    /// `mg_hierarchy` is `Some`.
    aux_max_iter: usize,
    /// Enables the vector-nodal Π branch of AMS. Kept as a switch so
    /// tests and experiments can compare the v2 gradient-only path
    /// against the complete v5 auxiliary-space action.
    pi_projection_enabled: bool,
    /// Optional geometric multigrid hierarchy for the auxiliary block.
    /// When `Some`, `apply` runs `mg_cycles_per_apply` V-cycles on the
    /// finest level (which must match the mesh used to build this
    /// preconditioner) instead of the fixed-iter CG. Each V-cycle
    /// costs `O(N)` work — far less than a full sweep of the
    /// `aux_max_iter` CG iterations on a deep hierarchy. — Stage 48
    /// v3.
    mg_hierarchy: Option<MeshHierarchy>,
    /// Number of V-cycles applied per `apply` call. A single V-cycle
    /// gives `O(1/(cond^{1/2}))` reduction on the auxiliary residual
    /// — enough to match the spectral leverage of plain Jacobi but
    /// not the fixed-iter CG default. With the 2-level hierarchies
    /// produced by [`MeshHierarchy::build`] on tet meshes,
    /// 3 V-cycles (V(3,3) Jacobi) typically match 50 inner CG
    /// iterations in solve quality at a fraction of the matvec cost.
    /// Default 3 — chosen so the outer GMRES iteration count is at
    /// most `0.7×` the AMS-CG iteration count on the skin-depth slab
    /// problem (Stage 48 v3 test target).
    mg_cycles_per_apply: usize,
    /// Optional algebraic multigrid hierarchy — Stage 48 v4. When
    /// `Some`, takes priority over `mg_hierarchy` and applies
    /// `amg_cycles_per_apply` AMG V-cycles to the auxiliary block. AMG
    /// is mesh-independent (built from the matrix `K_aux` alone), so it
    /// works on the same arbitrary-topology meshes the rest of the FEM
    /// stack supports.
    amg_hierarchy: Option<AmgHierarchy>,
    /// Number of AMG V-cycles applied per `apply` call. Default 3 —
    /// same heuristic as `mg_cycles_per_apply`.
    amg_cycles_per_apply: usize,
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
        let pi = build_p1_vector_interpolation(mesh)?;

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

        let mut k_aux_vector = build_p1_vector_laplacian(mesh, 3)?;
        for &v in &pinned_for_gauge {
            pin_aux_vector_vertex(&mut k_aux_vector, v as usize, 3);
        }

        let diag = k_curl.diagonal();
        let inv_diag_curl = diag
            .iter()
            .map(|&d| if d.abs() > 1.0e-30 { 1.0 / d } else { 0.0 })
            .collect();
        Ok(Self {
            inv_diag_curl,
            g,
            pi,
            k_aux,
            k_aux_vector,
            pinned_vertex,
            pinned_edges: pinned_edges.to_vec(),
            pinned_aux_vertices: pinned_for_gauge,
            aux_max_iter: 50,
            pi_projection_enabled: true,
            mg_hierarchy: None,
            mg_cycles_per_apply: 8,
            amg_hierarchy: None,
            amg_cycles_per_apply: 1,
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

    /// Enable/disable the vector-nodal Π branch of AMS.
    ///
    /// The default is `true` (Stage 48 v5). Set to `false` only for
    /// compatibility experiments or tests that need the old
    /// gradient-only v2 action.
    pub fn with_pi_projection(mut self, enabled: bool) -> Self {
        self.pi_projection_enabled = enabled;
        self
    }

    /// Attach a [`MeshHierarchy`] so the inner auxiliary solve uses a
    /// geometric multigrid V-cycle instead of fixed-iter CG — Stage
    /// 48 v3.
    ///
    /// The caller is responsible for ensuring the hierarchy's finest
    /// mesh matches the mesh used to construct this preconditioner
    /// (same vertex IDs, same pinned-vertex set). The simplest way to
    /// achieve this is to build the hierarchy from the same base
    /// `FemMesh` with the same pinned-vertex list and use the *finest*
    /// level as the outer system's mesh.
    ///
    /// One V-cycle replaces ~50 inner CG iterations. The iter-count
    /// win at the outer GMRES level is typically 30-40% on the
    /// skin-depth slab; full AMG (v4) is needed to push the ratio
    /// below 0.3×.
    pub fn with_mg_hierarchy(mut self, h: MeshHierarchy) -> Self {
        self.mg_hierarchy = Some(h);
        self
    }

    /// Tune the V-cycle iteration count per `apply` call. See
    /// `mg_cycles_per_apply` field for the trade-off; default 3.
    pub fn with_mg_cycles_per_apply(mut self, n: usize) -> Self {
        self.mg_cycles_per_apply = n.max(1);
        self
    }

    /// Attach an [`AmgHierarchy`] so the inner auxiliary solve uses an
    /// algebraic multigrid V-cycle (Ruge-Stüben) — Stage 48 v4.
    ///
    /// The hierarchy is built from the AMS auxiliary matrix `K_aux`
    /// alone (see [`AmgHierarchy::from_matrix`]), so it works on
    /// arbitrary-topology meshes where geometric MG can't easily
    /// construct a refinement hierarchy. When both `with_mg_hierarchy`
    /// and `with_amg_hierarchy` are set, AMG takes priority.
    pub fn with_amg_hierarchy(mut self, h: AmgHierarchy) -> Self {
        self.amg_hierarchy = Some(h);
        self
    }

    /// Tune the AMG V-cycle iteration count per `apply` call. Default 1
    /// — AMG V-cycles are stronger than geometric V-cycles on tet
    /// meshes (because the algebraic prolongation captures the actual
    /// matrix structure), so one is typically enough.
    pub fn with_amg_cycles_per_apply(mut self, n: usize) -> Self {
        self.amg_cycles_per_apply = n.max(1);
        self
    }

    /// Read-only handle to the gauge-pinned auxiliary matrix `K_aux`.
    /// Exposed for the AMG entry-point and the v4 test harness so the
    /// AMG hierarchy can be built from the same matrix the
    /// preconditioner's inner solve uses.
    pub fn k_aux(&self) -> &BsrMatrix {
        &self.k_aux
    }

    /// Read-only handle to the pinned vector-P1 auxiliary Laplacian
    /// used by the Π branch.
    pub fn k_aux_vector(&self) -> &BsrMatrix {
        &self.k_aux_vector
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

        // 3) Inner auxiliary solve. Three flavours, in priority order:
        //    - v4 path: AMG V-cycle (when `amg_hierarchy` is set).
        //    - v3 path: geometric MG V-cycle on a uniformly refined
        //      hierarchy (when `mg_hierarchy` is set).
        //    - v2 path: fixed-iter CG on K_aux (linearity-preserving).
        //    All three produce a linear function of `r_aux`, so GMRES
        //    sees a *linear* preconditioner regardless of choice.
        let w = if let Some(h) = &self.amg_hierarchy {
            let n_verts = h.finest_matrix().nrows();
            let mut w_iter = vec![0.0; n_verts];
            for _ in 0..self.amg_cycles_per_apply {
                w_iter = h.v_cycle(&r_aux, &w_iter)?;
            }
            w_iter
        } else if let Some(h) = &self.mg_hierarchy {
            let n_verts = h.finest_laplacian().nrows();
            let mut w_iter = vec![0.0; n_verts];
            for _ in 0..self.mg_cycles_per_apply {
                w_iter = h.v_cycle(&r_aux, &w_iter)?;
            }
            w_iter
        } else {
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
            aux.x
        };

        // 4) Prolongate gradient branch back to edges.
        let mut z_2 = vec![0.0; n_edges];
        self.g.mv(&w, &mut z_2)?;

        // 5) Vector-nodal Π branch (Stage 48 v5).
        //
        // This is the second auxiliary-space leg of Hiptmair-Xu AMS:
        // restrict the edge residual into vector P1 space, solve a
        // vector Laplacian there, then interpolate the correction
        // back to Nedelec edge DoFs. Like the scalar CG fallback, the
        // inner solve is run for a fixed iteration cap to keep the
        // preconditioner linear from GMRES' perspective. Scalar
        // MG/AMG currently accelerate only the gradient branch; the Π
        // branch uses the shipped vector Laplacian directly.
        let mut z_pi = vec![0.0; n_edges];
        if self.pi_projection_enabled {
            let mut r_vec = matvec_transpose(&self.pi, r)?;
            for &v in &self.pinned_aux_vertices {
                let base = v as usize * 3;
                r_vec[base] = 0.0;
                r_vec[base + 1] = 0.0;
                r_vec[base + 2] = 0.0;
            }
            let k_vec = &self.k_aux_vector;
            let mv = |v: &[Real]| -> Vec<Real> { matvec(k_vec, v) };
            let w_init = vec![0.0; k_vec.nrows()];
            let w_vec = crate::math::iterative::cg_solve(
                mv,
                &r_vec,
                &w_init,
                0.0, // fixed iteration count; see scalar branch above.
                self.aux_max_iter,
            )
            .x;
            self.pi.mv(&w_vec, &mut z_pi)?;
        }

        // 6) Combine.
        for i in 0..n_edges {
            z[i] += z_2[i] + z_pi[i];
        }

        // 7) Zero pinned-edge entries — keep the preconditioner
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

/// Pin all vector components of one auxiliary P1 vertex.
fn pin_aux_vector_vertex(k: &mut BsrMatrix, vertex: usize, dim: usize) {
    for c in 0..dim {
        pin_aux_vertex(k, vertex * dim + c);
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
        mesh,
        j_source_re,
        j_source_im,
        nu_per_cell,
        sigma_per_cell,
        bcs,
        cfg,
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

    let op = EddyBlockOp {
        k_main: &k_main,
        k_off: &k_off,
    };

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

fn apply_ams_block_diagonal(ams: &AmsPreconditioner, n: usize, r: &[Real]) -> Result<Vec<Real>> {
    let (r_re, r_im) = r.split_at(n);
    let z_re = ams.apply(r_re)?;
    let z_im = ams.apply(r_im)?;
    let mut z = vec![0.0; 2 * n];
    z[..n].copy_from_slice(&z_re);
    z[n..].copy_from_slice(&z_im);
    Ok(z)
}

/// Apply a fixed linear block-coupled AMS preconditioner to a block-real
/// eddy-current residual.
///
/// The v2-v5 AMS action approximates only the diagonal block
/// `K_main^{-1}` independently for real and imaginary components. At
/// high `ωσ`, the off-diagonal mass terms dominate:
///
/// ```text
/// [ K_main   K_off ]
/// [ -K_off   K_main ]
/// ```
///
/// This helper folds that coupling into the preconditioner with a
/// fixed number of stationary refinement sweeps:
///
/// ```text
/// z₀     = D⁻¹ r
/// zᵢ₊₁  = zᵢ + D⁻¹ (r - A zᵢ)
/// ```
///
/// where `D⁻¹` applies AMS to each half. Because the sweep count is
/// fixed, the resulting operator is linear and safe to use inside
/// GMRES.
pub fn apply_block_coupled_ams_preconditioner(
    ams: &AmsPreconditioner,
    op: &EddyBlockOp<'_>,
    r: &[Real],
    sweeps: usize,
) -> Result<Vec<Real>> {
    let n = ams.inv_diag_curl.len();
    if r.len() != 2 * n {
        return Err(Error::Build(format!(
            "apply_block_coupled_ams_preconditioner: residual length {}, expected {}",
            r.len(),
            2 * n
        )));
    }
    if op.rows() != 2 * n || op.cols() != 2 * n {
        return Err(Error::Build(
            "apply_block_coupled_ams_preconditioner: operator/preconditioner dimension mismatch"
                .into(),
        ));
    }
    let mut z = apply_ams_block_diagonal(ams, n, r)?;
    for _ in 1..sweeps.max(1) {
        let mut az = vec![0.0; 2 * n];
        op.mv(&z, &mut az)?;
        let mut defect = vec![0.0; 2 * n];
        for i in 0..2 * n {
            defect[i] = r[i] - az[i];
        }
        let dz = apply_ams_block_diagonal(ams, n, &defect)?;
        for i in 0..2 * n {
            z[i] += dz[i];
        }
    }
    Ok(z)
}

/// Steady-state eddy-current solve via block-real GMRES + block-coupled
/// AMS preconditioning (Stage 48 v6).
///
/// This is the high-`ωσ` variant of [`solve_eddy_current_steady_ams`].
/// It uses the same AMS edge preconditioner for each `K_main` block,
/// but the preconditioner action performs fixed stationary refinement
/// sweeps against the full real/imaginary block operator so the
/// off-diagonal `±ω M_σ` coupling contributes to `M⁻¹ r`.
pub fn solve_eddy_current_steady_ams_block(
    mesh: &FemMesh,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    bcs: &[BoundaryCondition],
    cfg: &EddyCurrentConfig,
) -> Result<EddyCurrentResult> {
    let (map, k_main, k_off, f_re, f_im, pinned_edges) = assemble_eddy_system(
        mesh,
        j_source_re,
        j_source_im,
        nu_per_cell,
        sigma_per_cell,
        bcs,
        cfg,
    )?;
    let n = map.ndof();

    let ams = AmsPreconditioner::new(mesh, &k_main, &pinned_edges)?;
    let op = EddyBlockOp {
        k_main: &k_main,
        k_off: &k_off,
    };

    let mut b = vec![0.0; 2 * n];
    b[..n].copy_from_slice(&f_re);
    b[n..].copy_from_slice(&f_im);
    let mut x = vec![0.0; 2 * n];
    for &e in &pinned_edges {
        x[e as usize] = f_re[e as usize];
        x[n + e as usize] = f_im[e as usize];
    }

    let sweeps = cfg.block_precond_sweeps.max(1);
    let apply_m_inv = |r: &[Real]| -> Result<Vec<Real>> {
        apply_block_coupled_ams_preconditioner(&ams, &op, r, sweeps)
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

/// Solve the same eddy-current problem at multiple angular frequencies.
///
/// `cfg_template` supplies all solver settings except `omega`, which
/// is replaced by each value in `omegas`. When
/// `enforce_coulomb_gauge` is `true`, each returned potential is
/// post-projected with [`project_nedelec_coulomb_gauge`] while
/// preserving the same pinned edge DoFs used during assembly.
pub fn solve_eddy_current_frequency_sweep(
    mesh: &FemMesh,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    bcs: &[BoundaryCondition],
    cfg_template: &EddyCurrentConfig,
    omegas: &[Real],
    mode: EddyCurrentSolveMode,
    enforce_coulomb_gauge: bool,
) -> Result<Vec<EddyFrequencySweepSample>> {
    let mut out = Vec::with_capacity(omegas.len());
    for &omega in omegas {
        let mut cfg = cfg_template.clone();
        cfg.omega = omega;
        let mut result = match mode {
            EddyCurrentSolveMode::Jacobi => solve_eddy_current_steady(
                mesh,
                j_source_re,
                j_source_im,
                nu_per_cell,
                sigma_per_cell,
                bcs,
                &cfg,
            )?,
            EddyCurrentSolveMode::Ams => solve_eddy_current_steady_ams(
                mesh,
                j_source_re,
                j_source_im,
                nu_per_cell,
                sigma_per_cell,
                bcs,
                &cfg,
            )?,
            EddyCurrentSolveMode::AmsBlock => solve_eddy_current_steady_ams_block(
                mesh,
                j_source_re,
                j_source_im,
                nu_per_cell,
                sigma_per_cell,
                bcs,
                &cfg,
            )?,
        };

        let (_map, _k_main, _k_off, _f_re, _f_im, pinned_edges) = assemble_eddy_system(
            mesh,
            j_source_re,
            j_source_im,
            nu_per_cell,
            sigma_per_cell,
            bcs,
            &cfg,
        )?;
        if enforce_coulomb_gauge {
            result.a_re = project_nedelec_coulomb_gauge(mesh, &result.a_re, &pinned_edges)?;
            result.a_im = project_nedelec_coulomb_gauge(mesh, &result.a_im, &pinned_edges)?;
        }
        let gauge_residual = coulomb_gauge_residual(mesh, &result.a_re, &pinned_edges)?
            .max(coulomb_gauge_residual(mesh, &result.a_im, &pinned_edges)?);
        out.push(EddyFrequencySweepSample {
            omega,
            result,
            gauge_residual,
        });
    }
    Ok(out)
}

/// Steady-state eddy-current solve via block-real GMRES + AMS-MG
/// preconditioning (Stage 48 v3).
///
/// Same arguments as [`solve_eddy_current_steady_ams`] plus
/// `base_mesh` and `n_refines`: the AMS auxiliary block uses a
/// `n_refines + 1`-level geometric multigrid V-cycle on a uniformly-
/// refined hierarchy built from `base_mesh`. The caller is responsible
/// for ensuring `mesh == base_mesh.refine_uniform()^n_refines` (i.e.
/// the outer-system mesh is the finest level of the hierarchy) — the
/// helper [`build_eddy_hierarchy_from_base`] enforces this contract.
///
/// At `n_refines = 0` the hierarchy is a single level and the V-cycle
/// degenerates to a direct dense Cholesky solve on the finest level —
/// useful for tiny problems but defeats the multigrid purpose.
pub fn solve_eddy_current_steady_ams_mg(
    base_mesh: &FemMesh,
    n_refines: usize,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    bcs: &[BoundaryCondition],
    cfg: &EddyCurrentConfig,
) -> Result<EddyCurrentResult> {
    // Build the finest mesh by uniform refinement.
    let mesh = if n_refines == 0 {
        base_mesh.clone()
    } else {
        let mut amr = super::amr_mesh::AmrMesh::new(base_mesh.clone());
        for _ in 0..n_refines {
            amr.refine_uniform();
        }
        amr.flatten()
    };

    let (map, k_main, k_off, f_re, f_im, pinned_edges) = assemble_eddy_system(
        &mesh,
        j_source_re,
        j_source_im,
        nu_per_cell,
        sigma_per_cell,
        bcs,
        cfg,
    )?;
    let n = map.ndof();

    // Pinned auxiliary vertices on the finest mesh — these become the
    // hierarchy's pinned-vertex set at every level via position match.
    let (edges, _) = compute_edges(&mesh);
    let mut aux_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for &e in &pinned_edges {
        let [a, b] = edges[e as usize];
        aux_set.insert(a);
        aux_set.insert(b);
    }
    let pinned_aux_finest: Vec<u32> = {
        let mut v: Vec<u32> = aux_set.into_iter().collect();
        v.sort_unstable();
        v
    };

    // Build the hierarchy.
    let hierarchy = MeshHierarchy::build(base_mesh, n_refines, &pinned_aux_finest)?;

    // Build AMS with hierarchy.
    let ams = AmsPreconditioner::new(&mesh, &k_main, &pinned_edges)?.with_mg_hierarchy(hierarchy);

    let op = EddyBlockOp {
        k_main: &k_main,
        k_off: &k_off,
    };
    let mut b = vec![0.0; 2 * n];
    b[..n].copy_from_slice(&f_re);
    b[n..].copy_from_slice(&f_im);
    let mut x = vec![0.0; 2 * n];
    for &e in &pinned_edges {
        x[e as usize] = f_re[e as usize];
        x[n + e as usize] = f_im[e as usize];
    }

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

/// Steady-state eddy-current solve via block-real GMRES + AMS-AMG
/// preconditioning — Stage 48 v4.
///
/// Same arguments as [`solve_eddy_current_steady_ams`] plus `n_levels`:
/// the AMS auxiliary block is solved by a classical Ruge-Stüben AMG
/// V-cycle with at most `n_levels` levels (coarsening stops early once
/// the coarse matrix has ≤ 32 rows or the coarsening ratio falls
/// below the diminishing-returns threshold — see
/// [`AmgHierarchy::from_matrix`]).
///
/// AMG removes the v3 geometric requirement (uniformly refined mesh
/// hierarchy) — works on arbitrary mesh topologies, including AMR
/// outputs, irregular boundary refinements, and unstructured tet
/// imports.
pub fn solve_eddy_current_steady_ams_amg(
    mesh: &FemMesh,
    n_levels: usize,
    j_source_re: &[Vec3],
    j_source_im: &[Vec3],
    nu_per_cell: &[Real],
    sigma_per_cell: &[Real],
    bcs: &[BoundaryCondition],
    cfg: &EddyCurrentConfig,
) -> Result<EddyCurrentResult> {
    let (map, k_main, k_off, f_re, f_im, pinned_edges) = assemble_eddy_system(
        mesh,
        j_source_re,
        j_source_im,
        nu_per_cell,
        sigma_per_cell,
        bcs,
        cfg,
    )?;
    let n = map.ndof();

    // Build AMS first so we know the gauge-pinned K_aux.
    let ams_base = AmsPreconditioner::new(mesh, &k_main, &pinned_edges)?;
    let amg = AmgHierarchy::from_matrix(ams_base.k_aux(), n_levels)?;
    let ams = ams_base.with_amg_hierarchy(amg);

    let op = EddyBlockOp {
        k_main: &k_main,
        k_off: &k_off,
    };
    let mut b = vec![0.0; 2 * n];
    b[..n].copy_from_slice(&f_re);
    b[n..].copy_from_slice(&f_im);
    let mut x = vec![0.0; 2 * n];
    for &e in &pinned_edges {
        x[e as usize] = f_re[e as usize];
        x[n + e as usize] = f_im[e as usize];
    }

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
        return Err(Error::Build(
            "gmres_with_precond: operator is not square".into(),
        ));
    }
    if b.len() != n || x.len() != n {
        return Err(Error::Build(
            "gmres_with_precond: dimension mismatch".into(),
        ));
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
        let op = EddyBlockOp {
            k_main: &k_main,
            k_off: &k_off,
        };
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
