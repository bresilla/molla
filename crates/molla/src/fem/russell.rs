//! Stage 36.7 — direct sparse solver bridge.
//!
//! Drop-in alternative to [`crate::math::iterative::cg_solve`] for
//! linear systems `K · x = b` arising from the FEM assembly pipeline.
//! Same input (`BsrMatrix` with `B = 1` + `&[Real]` RHS), same output
//! (`Vec<Real>` solution); internally factorises once and back-solves
//! many times — which is the win on stiff / indefinite / very-small
//! systems where CG either stalls or pays more iteration overhead than
//! a one-shot LU costs.
//!
//! # Why not the upstream `russell_sparse` crate?
//!
//! The Stage 36.7 spec asked for a `russell_sparse` integration. After
//! investigation the upstream crate (`russell_sparse v1.x` on
//! crates.io) is not workable in the molla devShell:
//!
//! - It requires three native libraries — `libumfpack`, `libklu`
//!   (SuiteSparse), and `liblapacke` / `libopenblas`. No pure-Rust
//!   backend.
//! - Its `build.rs` hard-codes the Linux library + include search
//!   paths to `/usr/lib/x86_64-linux-gnu` and
//!   `/usr/include/suitesparse`. There is no environment-variable
//!   override.
//! - On the molla nix-flake devShell the SuiteSparse package only
//!   ships shared libraries (`/nix/store/.../lib/libumfpack.so`),
//!   no `umfpack.h` headers anywhere on the filesystem. The crate's
//!   `cc::Build::new()...includes(...)` step would fail at build time
//!   because the headers don't exist.
//! - Patching either the upstream `build.rs` or the molla flake to
//!   wire SuiteSparse + LAPACKE in is out of scope for this stage
//!   (the spec explicitly forbids modifying unrelated code).
//!
//! Per the spec fallback ("If russell_sparse is fundamentally not
//! workable… implement a small LU-on-BsrMatrix ourselves and call it
//! `DirectSolver` anyway") this module ships a **pure-Rust dense
//! LU with partial pivoting** under the same `russell` feature flag.
//! The public API matches what an eventual `russell_sparse` swap
//! would expose (`new` / `factorize` / `back_solve` / `solve`) so
//! follow-up work that wires the upstream crate in (after sorting
//! out the build-deps story) only needs to swap the implementation
//! body.
//!
//! # Algorithm
//!
//! 1. **Convert** the input `BsrMatrix` (`B = 1`, ergo plain CSR) to
//!    a dense column-major `n × n` matrix.
//! 2. **Factorise** in-place: textbook Gaussian elimination with
//!    partial (row) pivoting — the same arithmetic LAPACK's `dgetrf`
//!    runs, just without the blocking. Stores the lower-triangular
//!    `L` (unit diagonal) below the diagonal and the upper-triangular
//!    `U` on/above the diagonal of one shared `n × n` buffer; the
//!    pivot permutation is held in a separate `Vec<usize>`.
//! 3. **Back-solve** `L · y = P · b` (forward substitution) followed
//!    by `U · x = y` (back substitution).
//!
//! Cached factorisation lets [`DirectSolver::factorize`] run once and
//! then [`DirectSolver::back_solve`] cheaply replay against many
//! different RHS vectors — the whole point of "factor-once,
//! solve-many" for things like adjoint sweeps over multiple loads.
//!
//! # Size limits
//!
//! Dense storage is `O(n²)` doubles. At `n = 1024` that's 8 MB;
//! `n = 4096` is 128 MB; `n = 8192` is 512 MB. The factorise call
//! returns [`Error::Fem`] when `n > MAX_DIRECT_DIM` to fail loudly
//! before the allocator does — callers above that limit should stay
//! on `cg_solve`. The Stage 36 v1 / 36.7 demo problems (16³ tets and
//! smaller, ≤ 4913 DoFs) sit below that ceiling.
//!
//! A future swap to either `russell_sparse` (with the build issue
//! resolved) or a hand-rolled sparse LU lifts this limit.

use crate::core::{Error, Result};
use crate::math::sparse::BsrMatrix;
use crate::math::Real;

/// Hard upper bound on the dense-LU dimension. Exceeding it returns
/// [`Error::Fem`] from [`DirectSolver::factorize`] — the dense factor
/// would be > 1 GB. Callers past this size should keep iterating with
/// `cg_solve`.
pub const MAX_DIRECT_DIM: usize = 12_000;

/// Owns a cached LU factorisation of one matrix `K`. Build with
/// [`Self::new`], populate with [`Self::factorize`], then run as many
/// [`Self::back_solve`] calls as you like against different RHS
/// vectors — they all reuse the same factorisation.
#[derive(Debug, Default)]
pub struct DirectSolver {
    /// `n × n` column-major buffer. Stores both `L` (strict lower
    /// triangle, unit diagonal implicit) and `U` (upper triangle,
    /// including the diagonal) of the BC-reduced matrix after a
    /// successful [`Self::factorize`] call. Empty until then.
    lu: Vec<Real>,
    /// Pivot row permutation produced by partial pivoting:
    /// `piv[i]` = the row that ended up at position `i` after the
    /// elimination. Used by `back_solve` to permute the RHS into
    /// `P · b` before forward substitution.
    piv: Vec<usize>,
    /// Linear-system size after the most recent `factorize`. Zero
    /// before the first factorise.
    n: usize,
    /// Set once `factorize` has run successfully; gates `back_solve`.
    factorized: bool,
}

impl DirectSolver {
    /// Construct an empty solver. No allocation happens here — the
    /// dense factor buffer is created in [`Self::factorize`] when the
    /// matrix dimension is known.
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }

    /// Run an LU-with-partial-pivoting factorisation of `k`.
    ///
    /// Replaces any previous factorisation. After this returns
    /// `Ok(())`, [`Self::back_solve`] can be called repeatedly with
    /// different RHS vectors against the *same* `k`.
    ///
    /// Errors:
    /// - [`Error::Fem`] if `k.b != 1` (only `B = 1` BSR / CSR
    ///   matrices are supported in v1).
    /// - [`Error::Fem`] if `k` is not square.
    /// - [`Error::Fem`] if `n > MAX_DIRECT_DIM`.
    /// - [`Error::Solver`] if the matrix is singular (a zero pivot
    ///   shows up after partial pivoting).
    pub fn factorize(&mut self, k: &BsrMatrix) -> Result<()> {
        if k.b != 1 {
            return Err(Error::Fem(format!(
                "DirectSolver::factorize: only B=1 BSR / CSR is supported, got B={}",
                k.b
            )));
        }
        if k.nrows() != k.ncols() {
            return Err(Error::Fem(format!(
                "DirectSolver::factorize: matrix must be square, got {}×{}",
                k.nrows(),
                k.ncols()
            )));
        }
        let n = k.nrows();
        if n > MAX_DIRECT_DIM {
            return Err(Error::Fem(format!(
                "DirectSolver::factorize: n={n} exceeds MAX_DIRECT_DIM={MAX_DIRECT_DIM}; \
                 stay on cg_solve for problems this large"
            )));
        }

        // Densify into column-major storage. CSR rows ⇒ for every
        // row r and stored column slot c, write `dense[r + c*n]`.
        let mut dense = vec![0.0_f64; n * n];
        for r in 0..n {
            let lo = k.row_offsets[r];
            let hi = k.row_offsets[r + 1];
            for slot in lo..hi {
                let c = k.col_idx[slot];
                dense[r + c * n] = k.values[slot];
            }
        }

        // Textbook LU with partial pivoting (LAPACK `dgetrf` style).
        // Storage convention: after column j is done, `dense[i + j*n]`
        // holds U entries for `i ≤ j` and L entries for `i > j` (with
        // the unit diagonal of L kept implicit).
        let mut piv = vec![0_usize; n];
        for j in 0..n {
            // Pick the row with the largest |a[i, j]| for i ≥ j.
            let mut p = j;
            let mut max_abs = dense[j + j * n].abs();
            for i in (j + 1)..n {
                let v = dense[i + j * n].abs();
                if v > max_abs {
                    max_abs = v;
                    p = i;
                }
            }
            piv[j] = p;
            if max_abs == 0.0 {
                return Err(Error::Solver(format!(
                    "DirectSolver::factorize: zero pivot at column {j} \
                     — matrix is singular or has empty rows"
                )));
            }
            // Swap rows j and p across all n columns.
            if p != j {
                for col in 0..n {
                    dense.swap(j + col * n, p + col * n);
                }
            }
            // Scale the column below the pivot by 1 / a[j, j].
            let inv_pivot = 1.0 / dense[j + j * n];
            for i in (j + 1)..n {
                dense[i + j * n] *= inv_pivot;
            }
            // Trailing sub-matrix update: `a[i, k] -= a[i, j] · a[j, k]`
            // for `i > j` and `k > j`.
            for k_col in (j + 1)..n {
                let ajk = dense[j + k_col * n];
                if ajk == 0.0 {
                    continue;
                }
                for i in (j + 1)..n {
                    dense[i + k_col * n] -= dense[i + j * n] * ajk;
                }
            }
        }

        self.lu = dense;
        self.piv = piv;
        self.n = n;
        self.factorized = true;
        Ok(())
    }

    /// Back-solve `K · x = b` using the cached factorisation.
    ///
    /// Errors:
    /// - [`Error::Fem`] if no factorisation is loaded
    ///   ([`Self::factorize`] hasn't been called yet).
    /// - [`Error::Fem`] if `b.len() != n` for the cached factor.
    pub fn back_solve(&self, b: &[Real]) -> Result<Vec<Real>> {
        if !self.factorized {
            return Err(Error::Fem(
                "DirectSolver::back_solve: no factorisation loaded — call factorize first".into(),
            ));
        }
        if b.len() != self.n {
            return Err(Error::Fem(format!(
                "DirectSolver::back_solve: RHS length {} != cached n={}",
                b.len(),
                self.n
            )));
        }
        let n = self.n;
        let mut x = b.to_vec();

        // Apply the row permutation: x ← P · b. The pivot vector
        // records, in order, which row was swapped into row j during
        // factorisation, so we replay the same swaps on the RHS.
        for j in 0..n {
            let p = self.piv[j];
            if p != j {
                x.swap(j, p);
            }
        }

        // Forward substitution: solve `L · y = P · b` in place.
        // L has unit diagonal (implicit), so `y[i] = b[i] - Σ_{j<i} L[i,j]·y[j]`.
        for i in 1..n {
            let mut acc = x[i];
            for j in 0..i {
                acc -= self.lu[i + j * n] * x[j];
            }
            x[i] = acc;
        }

        // Back substitution: solve `U · x = y` in place.
        // `x[i] = (y[i] - Σ_{j>i} U[i,j]·x[j]) / U[i,i]`.
        for i in (0..n).rev() {
            let mut acc = x[i];
            for j in (i + 1)..n {
                acc -= self.lu[i + j * n] * x[j];
            }
            x[i] = acc / self.lu[i + i * n];
        }

        Ok(x)
    }

    /// Convenience: factorise `k` and back-solve `K · x = b` in one
    /// call. Equivalent to `solver.factorize(k)?; solver.back_solve(b)`.
    /// Use the two-step form when you intend to back-solve against
    /// many RHS vectors against the same `k`.
    pub fn solve(&mut self, k: &BsrMatrix, b: &[Real]) -> Result<Vec<Real>> {
        self.factorize(k)?;
        self.back_solve(b)
    }

    /// Returns the cached system size (zero before the first
    /// factorisation).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns whether a factorisation is currently loaded.
    #[must_use]
    pub fn is_factorized(&self) -> bool {
        self.factorized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
        let mut y = vec![0.0; a.nrows()];
        a.mv(x, &mut y).unwrap();
        y
    }

    /// Build a tiny 3×3 SPD matrix as a `BsrMatrix(B=1)` for the
    /// smoke test below.
    fn build_3x3_spd() -> BsrMatrix {
        // [[ 4, 1, 0],
        //  [ 1, 3, 1],
        //  [ 0, 1, 2]]  — SPD (diagonally dominant).
        let triplets: Vec<(usize, usize, Vec<Real>)> = vec![
            (0, 0, vec![4.0]),
            (0, 1, vec![1.0]),
            (1, 0, vec![1.0]),
            (1, 1, vec![3.0]),
            (1, 2, vec![1.0]),
            (2, 1, vec![1.0]),
            (2, 2, vec![2.0]),
        ];
        BsrMatrix::from_triplets(1, 3, 3, &triplets).unwrap()
    }

    #[test]
    fn solve_small_3x3_matches_residual_zero() {
        let k = build_3x3_spd();
        let b = vec![5.0, 8.0, 3.0];
        let mut solver = DirectSolver::new().unwrap();
        let x = solver.solve(&k, &b).unwrap();
        // Verify K · x = b.
        let kx = matvec(&k, &x);
        for i in 0..3 {
            assert!(
                (kx[i] - b[i]).abs() < 1e-12,
                "row {i}: K·x = {} but b = {}",
                kx[i],
                b[i]
            );
        }
    }

    #[test]
    fn factor_once_back_solve_many() {
        let k = build_3x3_spd();
        let mut solver = DirectSolver::new().unwrap();
        solver.factorize(&k).unwrap();
        assert!(solver.is_factorized());
        assert_eq!(solver.dim(), 3);

        // Replay against several RHS vectors.
        let rhs_set = [
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![5.0, 8.0, 3.0],
            vec![-1.0, 2.5, -7.0],
        ];
        for b in &rhs_set {
            let x = solver.back_solve(b).unwrap();
            let kx = matvec(&k, &x);
            for i in 0..3 {
                assert!(
                    (kx[i] - b[i]).abs() < 1e-12,
                    "back_solve residual on row {i}: K·x = {} but b = {}",
                    kx[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn back_solve_without_factorize_errors() {
        let solver = DirectSolver::new().unwrap();
        assert!(!solver.is_factorized());
        let result = solver.back_solve(&[1.0, 2.0, 3.0]);
        assert!(result.is_err(), "expected Err, got {:?}", result);
    }

    #[test]
    fn factorize_singular_matrix_errors() {
        // Row 0 and row 1 identical ⇒ singular.
        let triplets: Vec<(usize, usize, Vec<Real>)> = vec![
            (0, 0, vec![1.0]),
            (0, 1, vec![2.0]),
            (1, 0, vec![1.0]),
            (1, 1, vec![2.0]),
        ];
        let k = BsrMatrix::from_triplets(1, 2, 2, &triplets).unwrap();
        let mut solver = DirectSolver::new().unwrap();
        assert!(solver.factorize(&k).is_err());
    }

    #[test]
    fn factorize_oversized_matrix_errors() {
        // Build a tiny pattern but lie about the dimension by setting
        // block_rows past MAX_DIRECT_DIM. We can't easily build a real
        // matrix that big in a unit test (it'd allocate gigabytes), so
        // we sanity-check the bound by constructing one just past the
        // limit with an empty pattern.
        let n = MAX_DIRECT_DIM + 1;
        let k = BsrMatrix::from_pattern(1, n, n, vec![]).unwrap();
        let mut solver = DirectSolver::new().unwrap();
        let err = solver.factorize(&k);
        assert!(err.is_err(), "expected size-limit error, got {:?}", err);
    }
}
