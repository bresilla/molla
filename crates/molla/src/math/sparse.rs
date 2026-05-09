//! Sparse linear algebra — block-sparse-row matrix + iterative solvers.
//!
//! `BsrMatrix<B>` stores an `m·B × n·B` matrix as fixed-size `B × B`
//! dense blocks, addressed by CSR-style block-row offsets and
//! per-block column indices. `B = 1` recovers a plain CSR matrix.
//!
//! The iterative solvers (`cg`, `bicgstab`, `gmres`) use the
//! [`LinearOperator`] trait so callers can plug in custom matvec
//! routines (e.g. `J^T·J + λ·I` formed implicitly) without
//! materialising the full matrix.

use crate::core::{Error, Result};
use crate::math::Real;

/// Trait for "things you can multiply a vector by". `A·x → y`.
/// Solvers in this module never form `A` explicitly — they only call
/// `mv`. Callers can wrap a `BsrMatrix`, a closure, or a custom op.
pub trait LinearOperator {
    /// Output dimension: length of `y` produced by `mv`.
    fn rows(&self) -> usize;
    /// Input dimension: length of `x` consumed by `mv`.
    fn cols(&self) -> usize;
    /// Compute `y = A · x`. `x.len() == cols()`, `y.len() == rows()`.
    fn mv(&self, x: &[Real], y: &mut [Real]) -> Result<()>;
}

/// Block-sparse-row matrix with fixed block dimension `B`.
/// Conceptual size: `(block_rows · B) × (block_cols · B)`.
#[derive(Debug, Clone)]
pub struct BsrMatrix {
    /// Block dimension. Each value is a `B × B` dense block stored
    /// row-major.
    pub b: usize,
    /// Number of block rows.
    pub block_rows: usize,
    /// Number of block columns.
    pub block_cols: usize,
    /// Block-row offsets into `col_idx` / `values`. Length `block_rows + 1`.
    pub row_offsets: Vec<usize>,
    /// Block column indices. Length = total non-zero blocks.
    pub col_idx: Vec<usize>,
    /// Block data — `b² · nnz_blocks` values, row-major within each block.
    pub values: Vec<Real>,
}

impl BsrMatrix {
    /// Allocate an all-zero BSR matrix with the given pattern. The
    /// pattern is `(block_row, block_col)` pairs — sorted lexicographically
    /// is OK but not required (we sort here).
    pub fn from_pattern(
        b: usize,
        block_rows: usize,
        block_cols: usize,
        mut pattern: Vec<(usize, usize)>,
    ) -> Result<Self> {
        if b == 0 {
            return Err(Error::Build("BsrMatrix: block size must be > 0".into()));
        }
        pattern.sort_unstable();
        pattern.dedup();
        for &(r, c) in &pattern {
            if r >= block_rows || c >= block_cols {
                return Err(Error::Build(format!(
                    "BsrMatrix: pattern entry ({r},{c}) outside ({block_rows},{block_cols})"
                )));
            }
        }
        let nnz = pattern.len();
        let mut row_offsets = vec![0usize; block_rows + 1];
        for &(r, _) in &pattern {
            row_offsets[r + 1] += 1;
        }
        for i in 1..=block_rows {
            row_offsets[i] += row_offsets[i - 1];
        }
        let col_idx: Vec<usize> = pattern.iter().map(|&(_, c)| c).collect();
        let values = vec![0.0; nnz * b * b];
        Ok(Self {
            b,
            block_rows,
            block_cols,
            row_offsets,
            col_idx,
            values,
        })
    }

    /// Build directly from `(row, col, value)` triplets — coordinates
    /// are in BLOCK indices and `value` is a `B × B` row-major slice.
    /// Repeated `(row, col)` pairs accumulate.
    pub fn from_triplets(
        b: usize,
        block_rows: usize,
        block_cols: usize,
        triplets: &[(usize, usize, Vec<Real>)],
    ) -> Result<Self> {
        let pattern: Vec<(usize, usize)> = triplets.iter().map(|t| (t.0, t.1)).collect();
        let mut m = Self::from_pattern(b, block_rows, block_cols, pattern)?;
        for (r, c, blk) in triplets {
            if blk.len() != b * b {
                return Err(Error::Build(format!(
                    "BsrMatrix: triplet block size {} != B²={}",
                    blk.len(),
                    b * b
                )));
            }
            // Find the slot for (r, c).
            let lo = m.row_offsets[*r];
            let hi = m.row_offsets[*r + 1];
            let mut slot = None;
            for i in lo..hi {
                if m.col_idx[i] == *c {
                    slot = Some(i);
                    break;
                }
            }
            let slot = slot.ok_or_else(|| {
                Error::Build(format!("BsrMatrix: pattern missing ({r},{c})"))
            })?;
            for k in 0..(b * b) {
                m.values[slot * b * b + k] += blk[k];
            }
        }
        Ok(m)
    }

    /// Total scalar rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.block_rows * self.b
    }

    /// Total scalar columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.block_cols * self.b
    }

    /// `y = A · x` with `x.len() == ncols()`, `y.len() == nrows()`.
    pub fn mv(&self, x: &[Real], y: &mut [Real]) -> Result<()> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(Error::Build(format!(
                "BsrMatrix::mv: shape mismatch — A is {}×{}, x={}, y={}",
                self.nrows(),
                self.ncols(),
                x.len(),
                y.len()
            )));
        }
        for v in y.iter_mut() {
            *v = 0.0;
        }
        let b = self.b;
        for r in 0..self.block_rows {
            let lo = self.row_offsets[r];
            let hi = self.row_offsets[r + 1];
            for slot in lo..hi {
                let c = self.col_idx[slot];
                let blk = &self.values[slot * b * b..(slot + 1) * b * b];
                for i in 0..b {
                    let mut acc = 0.0;
                    for j in 0..b {
                        acc += blk[i * b + j] * x[c * b + j];
                    }
                    y[r * b + i] += acc;
                }
            }
        }
        Ok(())
    }

    /// Per-row-block diagonal: returns the diagonal of each diagonal
    /// block as a flat `(block_rows · b)` vector. Used by the Jacobi
    /// preconditioner. Off-diagonal contribution is zero — only the
    /// `(r, r)` blocks contribute.
    #[must_use]
    pub fn get_diag(&self) -> Vec<Real> {
        let b = self.b;
        let n = self.nrows();
        let mut d = vec![0.0; n];
        for r in 0..self.block_rows {
            let lo = self.row_offsets[r];
            let hi = self.row_offsets[r + 1];
            for slot in lo..hi {
                if self.col_idx[slot] == r {
                    let blk = &self.values[slot * b * b..(slot + 1) * b * b];
                    for i in 0..b {
                        d[r * b + i] += blk[i * b + i];
                    }
                }
            }
        }
        d
    }

    /// Multiply every block by `s`.
    pub fn scale(&mut self, s: Real) {
        for v in self.values.iter_mut() {
            *v *= s;
        }
    }
}

impl LinearOperator for BsrMatrix {
    fn rows(&self) -> usize {
        self.nrows()
    }
    fn cols(&self) -> usize {
        self.ncols()
    }
    fn mv(&self, x: &[Real], y: &mut [Real]) -> Result<()> {
        BsrMatrix::mv(self, x, y)
    }
}

/// Adapter so closures can serve as a `LinearOperator`.
pub struct ClosureOp<F>
where
    F: Fn(&[Real], &mut [Real]) -> Result<()>,
{
    pub rows: usize,
    pub cols: usize,
    pub f: F,
}

impl<F> LinearOperator for ClosureOp<F>
where
    F: Fn(&[Real], &mut [Real]) -> Result<()>,
{
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn mv(&self, x: &[Real], y: &mut [Real]) -> Result<()> {
        (self.f)(x, y)
    }
}

// ─── Iterative solvers ──────────────────────────────────────────────

/// Configuration shared by `cg` / `bicgstab` / `gmres`.
#[derive(Copy, Clone, Debug)]
pub struct IterConfig {
    /// Maximum iterations before giving up.
    pub max_iter: usize,
    /// Convergence tolerance on `‖b - A·x‖ / ‖b‖`.
    pub tol: Real,
    /// Restart length for GMRES (ignored by `cg` / `bicgstab`).
    pub restart: usize,
}

impl Default for IterConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1.0e-9,
            restart: 30,
        }
    }
}

/// Outcome of an iterative solve.
#[derive(Copy, Clone, Debug)]
pub struct IterStats {
    pub iters: usize,
    pub residual: Real,
    pub converged: bool,
}

fn dot(a: &[Real], b: &[Real]) -> Real {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn norm(a: &[Real]) -> Real {
    dot(a, a).sqrt()
}

fn axpy(y: &mut [Real], alpha: Real, x: &[Real]) {
    for i in 0..y.len() {
        y[i] += alpha * x[i];
    }
}


/// Conjugate gradient for SPD `A`. Optional Jacobi preconditioner via
/// `m_inv_diag`: pass `None` for unpreconditioned, otherwise a vector
/// of length `n` containing `1 / diag(A)`. `x` is overwritten with
/// the solution; pass an initial guess if you have one (e.g. from a
/// previous solve).
pub fn cg(
    op: &dyn LinearOperator,
    b: &[Real],
    x: &mut [Real],
    m_inv_diag: Option<&[Real]>,
    cfg: &IterConfig,
) -> Result<IterStats> {
    let n = op.rows();
    if op.cols() != n {
        return Err(Error::Build("cg: operator is not square".into()));
    }
    if b.len() != n || x.len() != n {
        return Err(Error::Build(format!(
            "cg: dimension mismatch (n={n}, b={}, x={})",
            b.len(),
            x.len()
        )));
    }
    let mut r = vec![0.0; n];
    let mut ap = vec![0.0; n];
    op.mv(x, &mut r)?;
    for i in 0..n {
        r[i] = b[i] - r[i];
    }

    let bnorm = norm(b).max(1.0e-30);
    let apply_m_inv = |z: &mut [Real], r: &[Real]| {
        if let Some(d) = m_inv_diag {
            for i in 0..n {
                z[i] = r[i] * d[i];
            }
        } else {
            z.copy_from_slice(r);
        }
    };

    let mut z = vec![0.0; n];
    apply_m_inv(&mut z, &r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    for it in 0..cfg.max_iter {
        op.mv(&p, &mut ap)?;
        let pap = dot(&p, &ap);
        if pap.abs() < 1.0e-30 {
            return Ok(IterStats {
                iters: it,
                residual: norm(&r) / bnorm,
                converged: norm(&r) / bnorm < cfg.tol,
            });
        }
        let alpha = rz / pap;
        axpy(x, alpha, &p);
        axpy(&mut r, -alpha, &ap);
        let res = norm(&r) / bnorm;
        if res < cfg.tol {
            return Ok(IterStats {
                iters: it + 1,
                residual: res,
                converged: true,
            });
        }
        apply_m_inv(&mut z, &r);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    Ok(IterStats {
        iters: cfg.max_iter,
        residual: norm(&r) / bnorm,
        converged: false,
    })
}

/// BiCGStab for general (non-symmetric) `A`. Same calling convention
/// as `cg`. Jacobi-preconditioned via `m_inv_diag` if provided.
pub fn bicgstab(
    op: &dyn LinearOperator,
    b: &[Real],
    x: &mut [Real],
    m_inv_diag: Option<&[Real]>,
    cfg: &IterConfig,
) -> Result<IterStats> {
    let n = op.rows();
    if op.cols() != n {
        return Err(Error::Build("bicgstab: operator is not square".into()));
    }
    if b.len() != n || x.len() != n {
        return Err(Error::Build("bicgstab: dimension mismatch".into()));
    }
    let mut r = vec![0.0; n];
    op.mv(x, &mut r)?;
    for i in 0..n {
        r[i] = b[i] - r[i];
    }
    let r_hat = r.clone();
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut p_hat = vec![0.0; n];
    let mut s_hat = vec![0.0; n];
    let mut rho = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let bnorm = norm(b).max(1.0e-30);

    let apply_m_inv = |z: &mut [Real], r: &[Real]| {
        if let Some(d) = m_inv_diag {
            for i in 0..n {
                z[i] = r[i] * d[i];
            }
        } else {
            z.copy_from_slice(r);
        }
    };

    for it in 0..cfg.max_iter {
        let rho_new = dot(&r_hat, &r);
        if rho_new.abs() < 1.0e-30 {
            return Ok(IterStats {
                iters: it,
                residual: norm(&r) / bnorm,
                converged: false,
            });
        }
        if it == 0 {
            p.copy_from_slice(&r);
        } else {
            let beta = (rho_new / rho) * (alpha / omega);
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }
        apply_m_inv(&mut p_hat, &p);
        op.mv(&p_hat, &mut v)?;
        alpha = rho_new / dot(&r_hat, &v);
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        if norm(&s) / bnorm < cfg.tol {
            axpy(x, alpha, &p_hat);
            return Ok(IterStats {
                iters: it + 1,
                residual: norm(&s) / bnorm,
                converged: true,
            });
        }
        apply_m_inv(&mut s_hat, &s);
        op.mv(&s_hat, &mut t)?;
        let tt = dot(&t, &t).max(1.0e-30);
        omega = dot(&t, &s) / tt;
        axpy(x, alpha, &p_hat);
        axpy(x, omega, &s_hat);
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }
        let res = norm(&r) / bnorm;
        if res < cfg.tol {
            return Ok(IterStats {
                iters: it + 1,
                residual: res,
                converged: true,
            });
        }
        rho = rho_new;
    }
    Ok(IterStats {
        iters: cfg.max_iter,
        residual: norm(&r) / bnorm,
        converged: false,
    })
}

/// Restartable GMRES(m). Same calling convention; `cfg.restart`
/// controls the restart length.
pub fn gmres(
    op: &dyn LinearOperator,
    b: &[Real],
    x: &mut [Real],
    m_inv_diag: Option<&[Real]>,
    cfg: &IterConfig,
) -> Result<IterStats> {
    let n = op.rows();
    if op.cols() != n {
        return Err(Error::Build("gmres: operator is not square".into()));
    }
    if b.len() != n || x.len() != n {
        return Err(Error::Build("gmres: dimension mismatch".into()));
    }
    let m = cfg.restart.max(1).min(n);
    let bnorm = norm(b).max(1.0e-30);
    let apply_m_inv = |z: &mut [Real], r: &[Real]| {
        if let Some(d) = m_inv_diag {
            for i in 0..n {
                z[i] = r[i] * d[i];
            }
        } else {
            z.copy_from_slice(r);
        }
    };

    let mut total_iters = 0usize;
    while total_iters < cfg.max_iter {
        // r₀ = b − A·x; z = M⁻¹·r₀.
        let mut r = vec![0.0; n];
        op.mv(x, &mut r)?;
        for i in 0..n {
            r[i] = b[i] - r[i];
        }
        let mut z = vec![0.0; n];
        apply_m_inv(&mut z, &r);
        let beta = norm(&z);
        if beta / bnorm < cfg.tol {
            return Ok(IterStats {
                iters: total_iters,
                residual: beta / bnorm,
                converged: true,
            });
        }
        // V[0] = z / β
        let mut v_basis: Vec<Vec<Real>> = vec![Vec::new(); m + 1];
        v_basis[0] = z.iter().map(|x| x / beta).collect();
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
            let mut zw = vec![0.0; n];
            apply_m_inv(&mut zw, &w);
            for j in 0..=k {
                h[j * m + k] = dot(&v_basis[j], &zw);
                axpy(&mut zw, -h[j * m + k], &v_basis[j]);
            }
            h[(k + 1) * m + k] = norm(&zw);
            if h[(k + 1) * m + k].abs() > 1.0e-30 {
                v_basis[k + 1] = zw.iter().map(|x| x / h[(k + 1) * m + k]).collect();
            } else {
                v_basis[k + 1] = vec![0.0; n];
            }

            // Apply previous Givens rotations.
            for i in 0..k {
                let temp = cs[i] * h[i * m + k] + sn[i] * h[(i + 1) * m + k];
                h[(i + 1) * m + k] = -sn[i] * h[i * m + k] + cs[i] * h[(i + 1) * m + k];
                h[i * m + k] = temp;
            }
            // Compute & apply new Givens for h[k, k] / h[k+1, k].
            let h_kk = h[k * m + k];
            let h_k1k = h[(k + 1) * m + k];
            let r_kk = (h_kk * h_kk + h_k1k * h_k1k).sqrt();
            cs[k] = h_kk / r_kk;
            sn[k] = h_k1k / r_kk;
            h[k * m + k] = r_kk;
            h[(k + 1) * m + k] = 0.0;
            // Apply to g.
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
        // Solve upper-triangular Hₖ · y = ḡ.
        let mut y = vec![0.0; k_done];
        for i in (0..k_done).rev() {
            let mut s = g[i];
            for j in (i + 1)..k_done {
                s -= h[i * m + j] * y[j];
            }
            y[i] = s / h[i * m + i];
        }
        for i in 0..k_done {
            axpy(x, y[i], &v_basis[i]);
        }
        if converged {
            return Ok(IterStats {
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
    Ok(IterStats {
        iters: total_iters,
        residual: norm(&r) / bnorm,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_inv(d: &[Real]) -> Vec<Real> {
        d.iter().map(|x| if *x != 0.0 { 1.0 / x } else { 0.0 }).collect()
    }

    /// 4×4 SPD via tridiagonal `[2, -1; -1, 2, -1; -1, 2, -1; -1, 2]`
    /// stored as B=1 BSR (i.e. CSR).
    fn tridiag_4() -> BsrMatrix {
        let n = 4;
        let mut triplets: Vec<(usize, usize, Vec<Real>)> = Vec::new();
        for i in 0..n {
            triplets.push((i, i, vec![2.0]));
            if i > 0 {
                triplets.push((i, i - 1, vec![-1.0]));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, vec![-1.0]));
            }
        }
        BsrMatrix::from_triplets(1, n, n, &triplets).unwrap()
    }

    #[test]
    fn bsr_mv_matches_dense() {
        let a = tridiag_4();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];
        a.mv(&x, &mut y).unwrap();
        // Tridiag * (1, 2, 3, 4) = (0, 0, 0, 5).
        assert_eq!(y, vec![0.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn bsr_get_diag() {
        let a = tridiag_4();
        let d = a.get_diag();
        assert_eq!(d, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn cg_solves_tridiag() {
        let a = tridiag_4();
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let mut x = vec![0.0; 4];
        let cfg = IterConfig {
            max_iter: 100,
            tol: 1.0e-12,
            restart: 30,
        };
        let stats = cg(&a, &b, &mut x, None, &cfg).unwrap();
        assert!(stats.converged);
        // Verify: A·x ≈ b.
        let mut ax = vec![0.0; 4];
        a.mv(&x, &mut ax).unwrap();
        for i in 0..4 {
            assert!((ax[i] - b[i]).abs() < 1.0e-9);
        }
    }

    #[test]
    fn cg_with_jacobi_preconditioner() {
        let a = tridiag_4();
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0; 4];
        let m_inv = diag_inv(&a.get_diag());
        let cfg = IterConfig::default();
        let stats = cg(&a, &b, &mut x, Some(&m_inv), &cfg).unwrap();
        assert!(stats.converged);
    }

    #[test]
    fn bicgstab_solves_tridiag() {
        let a = tridiag_4();
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0; 4];
        let stats = bicgstab(&a, &b, &mut x, None, &IterConfig::default()).unwrap();
        assert!(stats.converged);
    }

    #[test]
    fn gmres_solves_tridiag() {
        let a = tridiag_4();
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0; 4];
        let cfg = IterConfig {
            max_iter: 100,
            tol: 1.0e-9,
            restart: 4,
        };
        let stats = gmres(&a, &b, &mut x, None, &cfg).unwrap();
        assert!(stats.converged, "GMRES did not converge: {stats:?}");
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "perf test — run with --release")]
    fn cg_meets_dod_1000x1000_in_under_100ms() {
        // 1000x1000 tridiagonal — well-conditioned, CG should converge in ≪100 iters.
        let n = 1000;
        let mut triplets: Vec<(usize, usize, Vec<Real>)> = Vec::new();
        for i in 0..n {
            triplets.push((i, i, vec![2.0]));
            if i > 0 {
                triplets.push((i, i - 1, vec![-1.0]));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, vec![-1.0]));
            }
        }
        let a = BsrMatrix::from_triplets(1, n, n, &triplets).unwrap();
        let b: Vec<Real> = (0..n).map(|i| (i as Real).sin()).collect();
        let mut x = vec![0.0; n];
        let cfg = IterConfig {
            max_iter: 2000,
            tol: 1.0e-6,
            restart: 30,
        };
        let m_inv = diag_inv(&a.get_diag());
        let t0 = std::time::Instant::now();
        let stats = cg(&a, &b, &mut x, Some(&m_inv), &cfg).unwrap();
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("cg 1000x1000: {} iters, {:.2} ms", stats.iters, dt_ms);
        assert!(stats.converged);
        assert!(dt_ms < 100.0, "CG too slow: {dt_ms:.1} ms");
    }
}
