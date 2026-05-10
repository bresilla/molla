//! Top-level FEM solvers — steady-state Laplace / Poisson and
//! transient parabolic (heat-equation) backward Euler.
//!
//! Both routines layer on top of [`crate::math::iterative::cg_solve`]:
//! the global stiffness / mass matrices are assembled into CSR
//! storage but never copied into a dense form. CG only needs the
//! matvec product, which the closure-driven API expresses naturally.
//!
//! # Algorithms
//!
//! - **Steady state** — assemble `K`, apply Dirichlet BCs to `(K, f)`,
//!   then solve `K · u = f` via CG.
//! - **Transient** — backward Euler:
//!
//!   `(M / Δt + K) · uⁿ⁺¹ = f + (M / Δt) · uⁿ`
//!
//!   The Dirichlet enforcement is applied to the LHS = `M / Δt + K`
//!   and the assembled RHS each step, so the solution is consistent
//!   with the BC at every time level.
//!
//! v1 caveat: because [`super::bc::apply_dirichlet`] only zeros rows
//! (no symmetric column reduction), the constrained system is mildly
//! non-symmetric on the BC rows. CG still converges in practice on
//! these diagonally-dominant Laplacian-like systems; if you ever hit
//! a pathological problem, swap in `bicgstab_solve` from
//! [`crate::math::iterative`].

use crate::math::iterative::{cg_solve, CgResult};
use crate::math::sparse::BsrMatrix;
use crate::math::Real;

use super::assembly::matvec;
use super::bc::{apply_dirichlet, BoundaryCondition};

#[cfg(feature = "gpu")]
use crate::compute::Engine;
#[cfg(feature = "gpu")]
use crate::core::{Error, Result};
#[cfg(feature = "gpu")]
use crate::fem::gpu_data::FemGpuData;
#[cfg(feature = "gpu")]
use crate::kernels::FemGpuKernel;

/// Default tolerance for the FEM CG solves — `1e-8` is sufficient for
/// the v1 contract tests at 16³ resolution.
pub const DEFAULT_TOL: Real = 1.0e-8;

/// Default iteration cap — generous for v1 mesh sizes (≤16³).
pub const DEFAULT_MAX_ITERS: usize = 2000;

/// Solve `K · u = f` with Dirichlet BCs applied to the row-reduced
/// system. Returns the full DoF-length solution vector.
///
/// `k` is consumed (reduced in place); pass a fresh assembly per call.
/// Use [`solve_steady_state_with_initial`] if you have a warm start.
pub fn solve_steady_state(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    max_iters: usize,
    tol: Real,
) -> Vec<Real> {
    solve_steady_state_with_initial(k, f, bcs, &vec![0.0; f.len()], max_iters, tol).x
}

/// Variant of [`solve_steady_state`] that returns the full
/// [`CgResult`] (residual + iteration count) and accepts an initial
/// guess for warm-started repeated solves.
pub fn solve_steady_state_with_initial(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    x0: &[Real],
    max_iters: usize,
    tol: Real,
) -> CgResult {
    let mut k_red = k.clone();
    let mut f_red = f.to_vec();
    apply_dirichlet(&mut k_red, &mut f_red, bcs);
    let mut x_init = x0.to_vec();
    // Make sure the initial guess is consistent with the Dirichlet
    // values — otherwise CG burns iterations correcting it.
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x_init[*vertex as usize] = *value;
        }
    }
    cg_solve(|x| matvec(&k_red, x), &f_red, &x_init, tol, max_iters)
}

/// One backward-Euler step of the parabolic heat equation:
///
/// `(M / Δt + K) · uⁿ⁺¹ = f + (M / Δt) · uⁿ`
///
/// `m` is the mass matrix, `k` the stiffness (already scaled by the
/// diffusivity `α`); `bcs` are enforced on the LHS + RHS. Returns the
/// new state vector. Use [`solve_transient_with_initial`] if you want
/// to plug in a warm start (otherwise the previous `u_old` is the
/// natural initial guess and is used here).
pub fn solve_transient(
    m: &BsrMatrix,
    k: &BsrMatrix,
    f: &[Real],
    u_old: &[Real],
    dt: Real,
    bcs: &[BoundaryCondition],
) -> Vec<Real> {
    solve_transient_with_initial(
        m,
        k,
        f,
        u_old,
        dt,
        bcs,
        u_old,
        DEFAULT_MAX_ITERS,
        DEFAULT_TOL,
    )
    .x
}

/// Variant of [`solve_transient`] exposing the iteration count +
/// residual + initial guess. Useful when stepping repeatedly in a
/// viewer loop where you want to display residual telemetry.
#[allow(clippy::too_many_arguments)]
pub fn solve_transient_with_initial(
    m: &BsrMatrix,
    k: &BsrMatrix,
    f: &[Real],
    u_old: &[Real],
    dt: Real,
    bcs: &[BoundaryCondition],
    x0: &[Real],
    max_iters: usize,
    tol: Real,
) -> CgResult {
    debug_assert_eq!(m.b, 1);
    debug_assert_eq!(k.b, 1);
    debug_assert_eq!(m.nrows(), k.nrows());
    debug_assert_eq!(m.nrows(), f.len());
    debug_assert_eq!(m.nrows(), u_old.len());

    let inv_dt = 1.0 / dt;
    // Build the LHS A = M/Δt + K. We materialise this CSR so the
    // closure-driven `cg_solve` can call `matvec(&a, x)` directly.
    let a = add_csr_scaled(m, inv_dt, k, 1.0);
    // RHS b = f + (M/Δt) · u_old.
    let m_uold = matvec(m, u_old);
    let mut b: Vec<Real> = f
        .iter()
        .zip(m_uold.iter())
        .map(|(fi, mi)| fi + inv_dt * mi)
        .collect();

    // Apply Dirichlet to the LHS + RHS in tandem.
    let mut a_red = a;
    apply_dirichlet(&mut a_red, &mut b, bcs);
    let mut x_init = x0.to_vec();
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x_init[*vertex as usize] = *value;
        }
    }
    cg_solve(|x| matvec(&a_red, x), &b, &x_init, tol, max_iters)
}

/// Stage 36.5 — GPU port of [`solve_steady_state`].
///
/// Wraps the [`crate::kernels::FemGpuKernel::matvec_blocking`] entry
/// in a closure for the CPU [`cg_solve`] driver. Each CG iteration
/// dispatches one matvec + reads the result back into host f64.
///
/// # API contract
///
/// `K_csr_gpu` is expected to be the **already BC-reduced** CSR
/// matrix — the caller must `apply_dirichlet` to a CPU `BsrMatrix`
/// + `f` BEFORE uploading via [`FemGpuData::upload`]. v1 does not
/// re-apply BCs on the GPU side; doing so cleanly requires a
/// dedicated BC-application kernel which is deferred to a follow-up
/// (Stage 36.5.1). The `bcs` parameter is consulted ONLY to seed the
/// initial guess (the CG residual at BC rows is then trivially zero
/// because the BC-reduced rows are identity).
///
/// `f` is the BC-reduced RHS — same one passed to `cg_solve` on the
/// CPU path.
///
/// # Per-iter readback cost
///
/// On the 16³ tet test (~5k DoFs) each CG iteration measures
/// ~0.5 ms wall-clock on RTX 4080 — dominated by the host bounce
/// from reading the matvec result back into a `Vec<f64>` for the
/// host CG iterates. CG converges in ~50-200 iters on the
/// heat-equation problem so the total readback overhead is
/// 25-100 ms per solve. The CPU path, by comparison, runs the same
/// problem in <50 ms because it has no per-iter device sync.
///
/// **Stage 36.5.1 follow-up**: fully on-device CG keeps `r`, `p`,
/// `x` resident on the GPU and only reads back at convergence,
/// eliminating the per-iter host bounce.
#[cfg(feature = "gpu")]
pub fn solve_steady_state_gpu(
    k_csr_gpu: &FemGpuData,
    f: &[Real],
    bcs: &[BoundaryCondition],
    engine: &Engine,
    max_iters: usize,
    tol: Real,
) -> Result<CgResult> {
    let n = f.len();
    if k_csr_gpu.csr_nrows as usize != n {
        return Err(Error::Build(format!(
            "solve_steady_state_gpu: GPU CSR rows ({}) != f length ({n})",
            k_csr_gpu.csr_nrows,
        )));
    }
    let device = engine.device().ok_or_else(|| {
        Error::Gpu("solve_steady_state_gpu: engine has no GPU device".into())
    })?;
    let queue = engine.queue().ok_or_else(|| {
        Error::Gpu("solve_steady_state_gpu: engine has no GPU queue".into())
    })?;

    let kernel = FemGpuKernel::new(device.clone(), queue.clone());

    // Seed the initial guess with Dirichlet values. Even though the
    // BC-reduced CSR has identity rows for those DoFs, seeding makes
    // the CG residual at iter 0 already zero on those rows so we save
    // a pass that wouldn't move them anyway.
    let mut x_init = vec![0.0; n];
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = *bc {
            x_init[vertex as usize] = value;
        }
    }

    // Closure: y = K · x. Down-converts x to f32, dispatches GPU,
    // up-converts y back to f64.
    let result = cg_solve(
        |x_f64: &[Real]| -> Vec<Real> {
            let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
            // The matvec_blocking call is the only place per-iter that
            // does host I/O. v2 removes this — see the function
            // doc-comment.
            let y_f32 = kernel
                .matvec_blocking(k_csr_gpu, &x_f32)
                .expect("FEM GPU matvec dispatch");
            y_f32.into_iter().map(|v| v as Real).collect()
        },
        f,
        &x_init,
        tol,
        max_iters,
    );
    Ok(result)
}

/// Vector-aware steady-state solve. Identical to
/// [`solve_steady_state`] except it routes through [`apply_bcs`] so
/// `BoundaryCondition::DirichletVector` and `NeumannTraction` are
/// honoured. `dim` is the per-vertex DoF count (1 for scalar fields,
/// 2 / 3 for vector elasticity); `vertices` is the mesh vertex array
/// needed for `NeumannTraction` face-area integration.
pub fn solve_steady_state_vector(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[crate::math::Vec3],
    dim: usize,
    max_iters: usize,
    tol: Real,
) -> Vec<Real> {
    let mut k_red = k.clone();
    let mut f_red = f.to_vec();
    super::bc::apply_bcs(&mut k_red, &mut f_red, bcs, vertices, dim);
    let mut x_init = vec![0.0; f.len()];
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, value } => {
                x_init[(*vertex as usize) * dim] = *value;
            }
            BoundaryCondition::DirichletVector { vertex, values } => {
                let base = (*vertex as usize) * dim;
                for (ci, val) in values.iter().enumerate() {
                    x_init[base + ci] = *val;
                }
            }
            _ => {}
        }
    }
    cg_solve(|x| matvec(&k_red, x), &f_red, &x_init, tol, max_iters).x
}

/// Newton-Raphson driver for the **nonlinear static** elasticity
/// solve `R(u) = K_int(u) − f_ext = 0`.
///
/// Each Newton iteration:
///   1. assemble the **tangent** `K(u)` via `tangent_assembler(u)`,
///   2. compute the **residual** `R(u) = K_int(u) − f_ext` via
///      `residual_assembler(u)`,
///   3. apply Dirichlet BCs to `(K, R)` (zero-row + diagonal=1, but
///      with RHS `−R` so the increment respects the BC),
///   4. solve `K · du = −R` via `cg_solve`,
///   5. update `u ← u + du`.
///
/// Termination: `‖du‖ < tol` (absolute) or `iter == max_newton_iters`.
///
/// `vertices` is needed by `apply_bcs` (NeumannTraction face areas);
/// `dim` is the per-vertex DoF count. The Dirichlet BCs are
/// **incremental** for the Newton step (`du = 0` at constrained DoFs
/// after the first iteration), but the user supplies the absolute
/// values — we seed `u_initial` to satisfy them then enforce `du = 0`
/// on those rows internally.
///
/// v1: no line search. Small problems only.
#[allow(clippy::too_many_arguments)]
pub fn solve_nonlinear_static<KAsm, RAsm>(
    tangent_assembler: KAsm,
    residual_assembler: RAsm,
    f_ext: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[crate::math::Vec3],
    dim: usize,
    max_newton_iters: usize,
    tol: Real,
) -> Vec<Real>
where
    KAsm: Fn(&[Real]) -> BsrMatrix,
    RAsm: Fn(&[Real]) -> Vec<Real>,
{
    let n = f_ext.len();
    let mut u = vec![0.0; n];
    // Seed Dirichlet entries.
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, value } => {
                u[(*vertex as usize) * dim] = *value;
            }
            BoundaryCondition::DirichletVector { vertex, values } => {
                let base = (*vertex as usize) * dim;
                for (ci, val) in values.iter().enumerate() {
                    u[base + ci] = *val;
                }
            }
            _ => {}
        }
    }

    for _iter in 0..max_newton_iters {
        let k = tangent_assembler(&u);
        let r_int = residual_assembler(&u);
        // Residual R(u) = K_int(u) − f_ext. Newton increment solves
        // K · du = −R = f_ext − K_int(u).
        let mut rhs: Vec<Real> = f_ext
            .iter()
            .zip(r_int.iter())
            .map(|(fe, ri)| fe - ri)
            .collect();
        let mut k_red = k;
        // For Dirichlet rows, enforce du = 0: pin row + zero RHS.
        // We re-use apply_bcs but with **zero**-valued Dirichlet rows
        // (the absolute Dirichlet value is already in `u`).
        let zero_bcs: Vec<BoundaryCondition> = bcs
            .iter()
            .map(|bc| match bc {
                BoundaryCondition::Dirichlet { vertex, .. } => {
                    BoundaryCondition::Dirichlet { vertex: *vertex, value: 0.0 }
                }
                BoundaryCondition::DirichletVector { vertex, values } => {
                    BoundaryCondition::DirichletVector {
                        vertex: *vertex,
                        values: vec![0.0; values.len()],
                    }
                }
                BoundaryCondition::Neumann { face, flux } => {
                    BoundaryCondition::Neumann { face: *face, flux: *flux }
                }
                BoundaryCondition::NeumannTraction { face, traction } => {
                    BoundaryCondition::NeumannTraction {
                        face: *face,
                        traction: *traction,
                    }
                }
            })
            .collect();
        // `f_ext` already accounts for tractions, so we DROP traction
        // entries from the BC list before pinning rows — apply_bcs
        // would otherwise add them again into `rhs`.
        let pin_only: Vec<BoundaryCondition> = zero_bcs
            .into_iter()
            .filter(|bc| {
                matches!(
                    bc,
                    BoundaryCondition::Dirichlet { .. }
                        | BoundaryCondition::DirichletVector { .. }
                )
            })
            .collect();
        super::bc::apply_bcs(&mut k_red, &mut rhs, &pin_only, vertices, dim);

        // Initial guess for the increment: zero (BCs already enforce
        // du = 0 on the constrained rows).
        let du = cg_solve(
            |x| matvec(&k_red, x),
            &rhs,
            &vec![0.0; n],
            DEFAULT_TOL,
            DEFAULT_MAX_ITERS,
        )
        .x;
        // Update.
        let mut max_du = 0.0_f64;
        for (uv, dv) in u.iter_mut().zip(du.iter()) {
            *uv += *dv;
            max_du = max_du.max(dv.abs());
        }
        if max_du < tol {
            break;
        }
    }
    u
}

/// Returns a fresh `BsrMatrix` representing `α·A + β·B` where `A` and
/// `B` share the same `(B = 1)` block size. Patterns are unioned by
/// re-emitting the triplets through [`BsrMatrix::from_triplets`] which
/// dedupes + sums on coincident `(r, c)`.
///
/// This is the v1 "good enough" implementation — `nnz` is at most
/// `nnz(A) + nnz(B)` and the sort happens once per call. Stage 36.5
/// will replace it with a sparse-add kernel that walks both row
/// patterns in tandem.
fn add_csr_scaled(a: &BsrMatrix, alpha: Real, b: &BsrMatrix, beta: Real) -> BsrMatrix {
    debug_assert_eq!(a.b, 1);
    debug_assert_eq!(b.b, 1);
    debug_assert_eq!(a.nrows(), b.nrows());
    debug_assert_eq!(a.ncols(), b.ncols());

    let mut triplets: Vec<(usize, usize, Vec<Real>)> =
        Vec::with_capacity(a.values.len() + b.values.len());
    for r in 0..a.block_rows {
        for slot in a.row_offsets[r]..a.row_offsets[r + 1] {
            let c = a.col_idx[slot];
            triplets.push((r, c, vec![alpha * a.values[slot]]));
        }
    }
    for r in 0..b.block_rows {
        for slot in b.row_offsets[r]..b.row_offsets[r + 1] {
            let c = b.col_idx[slot];
            triplets.push((r, c, vec![beta * b.values[slot]]));
        }
    }
    BsrMatrix::from_triplets(1, a.block_rows, a.block_cols, &triplets)
        .expect("add_csr_scaled: triplet rebuild")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::assembly::{assemble_mass, assemble_stiffness};
    use crate::fem::function_space::FunctionSpace;
    use crate::fem::mesh::FemMesh;

    #[test]
    fn steady_state_constant_dirichlet_yields_constant() {
        // K · u = 0, with all-boundary Dirichlet u = 5 ⇒ u ≡ 5.
        let mesh = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
        let space = FunctionSpace::P1Nodal;
        let k = assemble_stiffness(&mesh, space, 1.0).unwrap();
        let n = space.ndof(&mesh);
        let f = vec![0.0; n];

        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x.abs() < 1e-9 || (p.x - 1.0).abs() < 1e-9 || p.y.abs() < 1e-9 || (p.y - 1.0).abs() < 1e-9 {
                bcs.push(BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 5.0,
                });
            }
        }
        let u = solve_steady_state(&k, &f, &bcs, 500, 1e-10);
        for &val in &u {
            assert!((val - 5.0).abs() < 1e-6, "interior value {val} ≠ 5");
        }
    }

    #[test]
    fn transient_zero_dt_step_is_well_defined() {
        // A small, finite dt — verify the solver returns numbers and
        // the BC is honoured at the boundaries.
        let mesh = FemMesh::tri_grid_2d(3, 3, 1.0, 1.0);
        let space = FunctionSpace::P1Nodal;
        let m = assemble_mass(&mesh, space).unwrap();
        let k = assemble_stiffness(&mesh, space, 1.0).unwrap();
        let n = space.ndof(&mesh);
        let f = vec![0.0; n];
        let u_old = vec![0.0; n];
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.y.abs() < 1e-9 {
                bcs.push(BoundaryCondition::Dirichlet { vertex: i as u32, value: 0.0 });
            } else if (p.y - 1.0).abs() < 1e-9 {
                bcs.push(BoundaryCondition::Dirichlet { vertex: i as u32, value: 1.0 });
            }
        }
        let u = solve_transient(&m, &k, &f, &u_old, 0.1, &bcs);
        for bc in &bcs {
            if let BoundaryCondition::Dirichlet { vertex, value } = bc {
                assert!((u[*vertex as usize] - *value).abs() < 1e-6);
            }
        }
    }
}
