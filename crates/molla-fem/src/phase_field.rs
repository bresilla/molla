//! Phase-field fracture helpers.
//!
//! This module implements the CPU oracle for the first AT2-style
//! scalar damage solve. It is intentionally small and functionality-owned
//! by `molla-fem`: the same mesh, mass, and stiffness element kernels are
//! reused, and future GPU dispatch can accelerate this path without
//! changing the public fracture contract.

use crate::core::{Error, Result};
use crate::math::Real;
use crate::math::sparse::BsrMatrix;

use super::element::{compute_element_mass, compute_element_stiffness};
use super::mesh::FemMesh;

/// Material and regularisation parameters for the AT2 phase-field
/// fracture model.
#[derive(Copy, Clone, Debug)]
pub struct PhaseFieldFractureParams {
    /// Critical energy-release rate `G_c`.
    pub critical_energy_release_rate: Real,
    /// Length scale `ℓ` controlling diffusive crack width.
    pub length_scale: Real,
    /// Residual stiffness `k` in `g(d) = (1-d)^2 + k`.
    pub residual_stiffness: Real,
}

impl Default for PhaseFieldFractureParams {
    fn default() -> Self {
        Self {
            critical_energy_release_rate: 1.0,
            length_scale: 0.05,
            residual_stiffness: 1.0e-8,
        }
    }
}

impl PhaseFieldFractureParams {
    fn validate(&self) -> Result<()> {
        if self.critical_energy_release_rate <= 0.0 {
            return Err(Error::Fem(format!(
                "phase-field fracture: Gc must be > 0, got {}",
                self.critical_energy_release_rate
            )));
        }
        if self.length_scale <= 0.0 {
            return Err(Error::Fem(format!(
                "phase-field fracture: length_scale must be > 0, got {}",
                self.length_scale
            )));
        }
        if self.residual_stiffness < 0.0 {
            return Err(Error::Fem(format!(
                "phase-field fracture: residual_stiffness must be >= 0, got {}",
                self.residual_stiffness
            )));
        }
        Ok(())
    }
}

/// Result of the bounded irreversible Jacobi damage solve.
#[derive(Clone, Debug)]
pub struct PhaseFieldFractureSolveResult {
    pub damage: Vec<Real>,
    pub iterations: usize,
    pub residual_inf: Real,
}

/// Degradation function used by the split elastic energy:
/// `g(d) = (1 - d)^2 + k`.
#[must_use]
pub fn phase_field_degradation(damage: Real, residual_stiffness: Real) -> Real {
    let d = damage.clamp(0.0, 1.0);
    (1.0 - d) * (1.0 - d) + residual_stiffness
}

/// Irreversible history update: `H_{n+1} = max(H_n, ψ⁺)`.
#[must_use]
pub fn update_phase_field_history(previous: Real, tensile_energy_density: Real) -> Real {
    previous.max(tensile_energy_density.max(0.0))
}

/// Local AT2 energy density contribution for a nodal/interpolated
/// damage value and gradient norm.
#[must_use]
pub fn phase_field_fracture_energy_density(
    damage: Real,
    grad_damage_norm_sq: Real,
    history: Real,
    params: PhaseFieldFractureParams,
) -> Real {
    let g = phase_field_degradation(damage, params.residual_stiffness);
    let gc = params.critical_energy_release_rate;
    let ell = params.length_scale;
    g * history + 0.5 * gc * (damage * damage / ell + ell * grad_damage_norm_sq)
}

/// Assemble the frozen-history AT2 linear damage problem:
///
/// ```text
/// ∫ Gc ℓ ∇d·∇η + (Gc/ℓ + 2H) d η dx = ∫ 2H η dx
/// ```
///
/// `history` is nodal and is averaged per element before multiplying the
/// element mass matrix. The returned scalar BSR matrix has one DoF per
/// mesh vertex; callers can solve it directly or pass it to
/// [`solve_phase_field_fracture_jacobi`] for the bounded irreversible
/// CPU oracle.
pub fn assemble_phase_field_fracture_system(
    mesh: &FemMesh,
    history: &[Real],
    params: PhaseFieldFractureParams,
) -> Result<(BsrMatrix, Vec<Real>)> {
    params.validate()?;
    let n = mesh.vert_count();
    if history.len() != n {
        return Err(Error::Fem(format!(
            "assemble_phase_field_fracture_system: history.len() = {} but mesh has {n} vertices",
            history.len()
        )));
    }

    let gc = params.critical_energy_release_rate;
    let ell = params.length_scale;
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    let mut rhs = vec![0.0; n];

    for cell in &mesh.cells {
        let verts = cell.verts();
        let h = verts
            .iter()
            .map(|&v| history[v as usize].max(0.0))
            .sum::<Real>()
            / verts.len() as Real;
        let k_e = compute_element_stiffness(cell, &mesh.vertices, gc * ell);
        let m_e = compute_element_mass(cell, &mesh.vertices);
        let reaction = gc / ell + 2.0 * h;
        for (a_local, &a) in verts.iter().enumerate() {
            let a = a as usize;
            let mut rhs_a = 0.0;
            for (b_local, &b) in verts.iter().enumerate() {
                let b = b as usize;
                triplets.push((
                    a,
                    b,
                    k_e[a_local][b_local] + reaction * m_e[a_local][b_local],
                ));
                rhs_a += 2.0 * h * m_e[a_local][b_local];
            }
            rhs[a] += rhs_a;
        }
    }

    let lhs = BsrMatrix::from_scalar_triplets(n, n, &triplets)
        .map_err(|e| Error::Fem(format!("phase-field fracture assembly failed: {e}")))?;
    Ok((lhs, rhs))
}

/// Solve the assembled AT2 phase-field fracture update with a bounded
/// Jacobi oracle and enforce crack irreversibility:
/// `damage_new >= damage_old`.
pub fn solve_phase_field_fracture_jacobi(
    mesh: &FemMesh,
    damage_old: &[Real],
    history: &[Real],
    params: PhaseFieldFractureParams,
    max_iters: usize,
    tol: Real,
) -> Result<PhaseFieldFractureSolveResult> {
    let n = mesh.vert_count();
    if damage_old.len() != n {
        return Err(Error::Fem(format!(
            "solve_phase_field_fracture_jacobi: damage_old.len() = {} but mesh has {n} vertices",
            damage_old.len()
        )));
    }
    let (lhs, rhs) = assemble_phase_field_fracture_system(mesh, history, params)?;
    let mut damage: Vec<Real> = damage_old.iter().map(|&d| d.clamp(0.0, 1.0)).collect();
    let lower = damage.clone();
    let mut next = damage.clone();
    let tol = tol.max(0.0);
    let mut residual_inf = Real::INFINITY;
    let mut iterations = 0usize;

    for it in 0..max_iters {
        residual_inf = 0.0;
        for row in 0..n {
            let lo = lhs.row_offsets[row];
            let hi = lhs.row_offsets[row + 1];
            let mut diag = 0.0;
            let mut off = 0.0;
            for slot in lo..hi {
                let col = lhs.col_idx[slot];
                let val = lhs.values[slot];
                if col == row {
                    diag += val;
                } else {
                    off += val * damage[col];
                }
            }
            if diag.abs() <= 1.0e-30 {
                return Err(Error::Fem(format!(
                    "solve_phase_field_fracture_jacobi: zero diagonal at row {row}"
                )));
            }
            let unconstrained = (rhs[row] - off) / diag;
            let bounded = unconstrained.clamp(lower[row], 1.0);
            residual_inf = residual_inf.max((bounded - damage[row]).abs());
            next[row] = bounded;
        }
        core::mem::swap(&mut damage, &mut next);
        iterations = it + 1;
        if residual_inf <= tol {
            break;
        }
    }

    Ok(PhaseFieldFractureSolveResult {
        damage,
        iterations,
        residual_inf,
    })
}
