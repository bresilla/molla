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
use super::mesh::{FemCell, FemMesh};

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

// ─── Coupled displacement ↔ damage staggered solve (roadmap C3) ─────

/// Configuration for [`solve_phase_field_coupled_step`].
#[derive(Copy, Clone, Debug)]
pub struct PhaseFieldCoupledConfig {
    /// Young's modulus / Poisson ratio of the UNDAMAGED material.
    pub youngs: Real,
    pub poisson: Real,
    /// Staggered (alternating-minimization) iteration cap + damage
    /// convergence tolerance (max |Δd| per sweep).
    pub max_staggered: usize,
    pub tol_damage: Real,
    /// Inner displacement CG controls.
    pub cg_iters: usize,
    pub cg_tol: Real,
    /// Damage Jacobi controls (existing bounded/irreversible solve).
    pub damage_iters: usize,
    pub damage_tol: Real,
    /// Drive the history field with the SPECTRAL TENSILE energy
    /// `ψ⁺ = ½λ⟨tr ε⟩₊² + μ Σ⟨εᵢ⟩₊²` (Miehe split, from the analytic
    /// 2×2 strain eigenvalues) instead of the full density — damage
    /// then cannot grow under compression. Default on; disable only
    /// to reproduce the tension-dominated v1 behaviour.
    pub tensile_split: bool,
}

impl Default for PhaseFieldCoupledConfig {
    fn default() -> Self {
        Self {
            youngs: 1.0e3,
            poisson: 0.3,
            max_staggered: 30,
            tol_damage: 1.0e-4,
            cg_iters: 4000,
            cg_tol: 1.0e-10,
            damage_iters: 400,
            damage_tol: 1.0e-10,
            tensile_split: true,
        }
    }
}

/// One load step of the coupled solve.
#[derive(Clone, Debug)]
pub struct PhaseFieldCoupledResult {
    /// Nodal displacement, 2 DoF per vertex (2D tri meshes, v1).
    pub displacement: Vec<Real>,
    pub staggered_iters: usize,
    /// Elastic strain energy of the converged (degraded) state.
    pub strain_energy: Real,
}

/// One quasi-static load step of AT2 phase-field fracture on a 2D tri
/// mesh: alternating minimization between
///
/// 1. displacement — linear elasticity with per-element stiffness
///    degraded by `g(d̄_e)` (element-mean damage), prescribed
///    displacements via symmetric Dirichlet elimination, CG solve;
/// 2. history — nodal `H ← max(H, ψ⁺(ε))`, where ψ⁺ is the Miehe
///    SPECTRAL TENSILE energy by default (`cfg.tensile_split`, so
///    compression cannot drive damage) or the full density when the
///    split is disabled;
/// 3. damage — the existing bounded, irreversible AT2 screened-Poisson
///    Jacobi solve,
///
/// until the damage sweep changes by less than `tol_damage`. `history`
/// and `damage` persist across load steps (irreversibility lives in
/// them).
pub fn solve_phase_field_coupled_step(
    mesh: &FemMesh,
    params: PhaseFieldFractureParams,
    cfg: PhaseFieldCoupledConfig,
    dirichlet: &[(usize, Real)],
    history: &mut [Real],
    damage: &mut Vec<Real>,
) -> Result<PhaseFieldCoupledResult> {
    params.validate()?;
    let n = mesh.vert_count();
    if history.len() != n || damage.len() != n {
        return Err(Error::Fem(format!(
            "solve_phase_field_coupled_step: history/damage length must be {n}"
        )));
    }
    let dim = match mesh.cells.first() {
        Some(FemCell::Tri(_)) => 2,
        Some(FemCell::Tet(_)) => 3,
        _ => {
            return Err(Error::Fem(
                "solve_phase_field_coupled_step: tri or tet meshes".into(),
            ));
        }
    };
    for c in &mesh.cells {
        let ok = match c {
            FemCell::Tri(_) => dim == 2,
            FemCell::Tet(_) => dim == 3,
        };
        if !ok {
            return Err(Error::Fem(
                "solve_phase_field_coupled_step: mixed/unsupported cell types".into(),
            ));
        }
    }
    let lambda = cfg.youngs * cfg.poisson / ((1.0 + cfg.poisson) * (1.0 - 2.0 * cfg.poisson));
    let mu = cfg.youngs / (2.0 * (1.0 + cfg.poisson));
    let ndof = dim * n;
    let constrained: std::collections::HashMap<usize, Real> = dirichlet.iter().copied().collect();

    let mut displacement = vec![0.0; ndof];
    let mut staggered_iters = 0;
    let mut strain_energy = 0.0;

    for sweep in 0..cfg.max_staggered {
        staggered_iters = sweep + 1;

        // ── 1. Degraded elasticity: assemble K(d) and solve K u = 0
        //       with prescribed displacements (load enters through the
        //       Dirichlet lift).
        let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
        for cell in &mesh.cells {
            let verts = cell.verts();
            let d_bar =
                verts.iter().map(|&v| damage[v as usize]).sum::<Real>() / verts.len() as Real;
            let g = phase_field_degradation(d_bar, params.residual_stiffness);
            let ke = crate::element::compute_element_elasticity_stiffness(
                cell,
                &mesh.vertices,
                lambda,
                mu,
            );
            for (a, row) in ke.iter().enumerate() {
                let ga = dim * verts[a / dim] as usize + (a % dim);
                for (b, &v) in row.iter().enumerate() {
                    let gb = dim * verts[b / dim] as usize + (b % dim);
                    if v != 0.0 {
                        triplets.push((ga, gb, g * v));
                    }
                }
            }
        }
        let mut rhs = vec![0.0; ndof];
        // Symmetric Dirichlet: move constrained columns to the RHS,
        // then identity rows/cols.
        // Accumulate duplicate (r, c) entries, then build scalar-block
        // BSR (from_triplets overwrites duplicates rather than
        // summing).
        let mut acc: std::collections::BTreeMap<(usize, usize), Real> =
            std::collections::BTreeMap::new();
        for &(r, c, v) in &triplets {
            match (constrained.get(&r), constrained.get(&c)) {
                (None, None) => *acc.entry((r, c)).or_insert(0.0) += v,
                (None, Some(&uc)) => rhs[r] -= v * uc,
                _ => {}
            }
        }
        for &dof in constrained.keys() {
            acc.insert((dof, dof), 1.0);
        }
        let kept: Vec<(usize, usize, Vec<Real>)> =
            acc.into_iter().map(|((r, c), v)| (r, c, vec![v])).collect();
        let k = BsrMatrix::from_triplets(1, ndof, ndof, &kept)?;
        for (&dof, &val) in &constrained {
            rhs[dof] = val;
        }

        // Plain CG (K is SPD after elimination).
        let mut u = displacement.clone();
        for (&dof, &val) in &constrained {
            u[dof] = val;
        }
        let mut r = {
            let ku = crate::assembly::matvec(&k, &u);
            (0..ndof).map(|i| rhs[i] - ku[i]).collect::<Vec<_>>()
        };
        let mut p = r.clone();
        let mut rr: Real = r.iter().map(|x| x * x).sum();
        let rr0 = rr.max(1.0e-30);
        for _ in 0..cfg.cg_iters {
            if rr <= cfg.cg_tol * cfg.cg_tol * rr0 {
                break;
            }
            let kp = crate::assembly::matvec(&k, &p);
            let pkp: Real = p.iter().zip(kp.iter()).map(|(a, b)| a * b).sum();
            if pkp <= 0.0 {
                break;
            }
            let alpha = rr / pkp;
            for i in 0..ndof {
                u[i] += alpha * p[i];
                r[i] -= alpha * kp[i];
            }
            let rr_new: Real = r.iter().map(|x| x * x).sum();
            let beta = rr_new / rr;
            rr = rr_new;
            for i in 0..ndof {
                p[i] = r[i] + beta * p[i];
            }
        }
        displacement = u;

        // ── 2. History update from nodal strain energy density
        //       (area-weighted element→node scatter).
        let mut psi_node = vec![0.0; n];
        let mut area_node = vec![0.0; n];
        strain_energy = 0.0;
        for cell in &mesh.cells {
            let verts = cell.verts();
            // Per-element measure, P1 strain and tensile energy —
            // dimension-generic (2D tri plane strain / 3D tet).
            let (measure, psi_hist, psi_full) = match cell {
                FemCell::Tri(idx) => {
                    let p0 = mesh.vertices[idx[0] as usize];
                    let p1 = mesh.vertices[idx[1] as usize];
                    let p2 = mesh.vertices[idx[2] as usize];
                    let det = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
                    let area = 0.5 * det.abs();
                    if area <= 0.0 {
                        continue;
                    }
                    let inv = 1.0 / det;
                    let g0 = [(p1.y - p2.y) * inv, (p2.x - p1.x) * inv];
                    let g1 = [(p2.y - p0.y) * inv, (p0.x - p2.x) * inv];
                    let g2 = [(p0.y - p1.y) * inv, (p1.x - p0.x) * inv];
                    let grads = [g0, g1, g2];
                    let mut exx = 0.0;
                    let mut eyy = 0.0;
                    let mut exy = 0.0;
                    for a in 0..3 {
                        let ux = displacement[2 * idx[a] as usize];
                        let uy = displacement[2 * idx[a] as usize + 1];
                        exx += grads[a][0] * ux;
                        eyy += grads[a][1] * uy;
                        exy += 0.5 * (grads[a][1] * ux + grads[a][0] * uy);
                    }
                    let tr = exx + eyy;
                    let psi_full =
                        0.5 * lambda * tr * tr + mu * (exx * exx + eyy * eyy + 2.0 * exy * exy);
                    // Tensile split via analytic 2×2 principal strains
                    // (drives HISTORY only; reported energy stays the
                    // full physical density).
                    let psi_hist = if cfg.tensile_split {
                        let mean = 0.5 * tr;
                        let r = (0.25 * (exx - eyy) * (exx - eyy) + exy * exy).sqrt();
                        let e1 = (mean + r).max(0.0);
                        let e2 = (mean - r).max(0.0);
                        let tr_pos = tr.max(0.0);
                        0.5 * lambda * tr_pos * tr_pos + mu * (e1 * e1 + e2 * e2)
                    } else {
                        psi_full
                    };
                    (area, psi_hist, psi_full)
                }
                FemCell::Tet(idx) => {
                    let p = [
                        mesh.vertices[idx[0] as usize],
                        mesh.vertices[idx[1] as usize],
                        mesh.vertices[idx[2] as usize],
                        mesh.vertices[idx[3] as usize],
                    ];
                    let e1 = p[1] - p[0];
                    let e2 = p[2] - p[0];
                    let e3 = p[3] - p[0];
                    let det = e1.cross(e2).dot(e3);
                    let vol = det.abs() / 6.0;
                    if vol <= 0.0 {
                        continue;
                    }
                    // P1 gradients: rows of the inverse edge matrix.
                    let g1v = e2.cross(e3) / det;
                    let g2v = e3.cross(e1) / det;
                    let g3v = e1.cross(e2) / det;
                    let g0v = -(g1v + g2v + g3v);
                    let grads = [g0v, g1v, g2v, g3v];
                    // ∇u (3×3), then ε = sym(∇u).
                    let mut gu = [[0.0; 3]; 3];
                    for a in 0..4 {
                        let base = 3 * idx[a] as usize;
                        let ua = [
                            displacement[base],
                            displacement[base + 1],
                            displacement[base + 2],
                        ];
                        let ga = [grads[a].x, grads[a].y, grads[a].z];
                        for (r, u_r) in ua.iter().enumerate() {
                            for (c, g_c) in ga.iter().enumerate() {
                                gu[r][c] += u_r * g_c;
                            }
                        }
                    }
                    let eps = crate::math::Mat3::from_cols(
                        crate::math::Vec3::new(
                            gu[0][0],
                            0.5 * (gu[0][1] + gu[1][0]),
                            0.5 * (gu[0][2] + gu[2][0]),
                        ),
                        crate::math::Vec3::new(
                            0.5 * (gu[0][1] + gu[1][0]),
                            gu[1][1],
                            0.5 * (gu[1][2] + gu[2][1]),
                        ),
                        crate::math::Vec3::new(
                            0.5 * (gu[0][2] + gu[2][0]),
                            0.5 * (gu[1][2] + gu[2][1]),
                            gu[2][2],
                        ),
                    );
                    let tr = gu[0][0] + gu[1][1] + gu[2][2];
                    let fro2 = (0..3)
                        .map(|r| {
                            (0..3)
                                .map(|c| {
                                    let e_rc = 0.5 * (gu[r][c] + gu[c][r]);
                                    e_rc * e_rc
                                })
                                .sum::<Real>()
                        })
                        .sum::<Real>();
                    let psi_full = 0.5 * lambda * tr * tr + mu * fro2;
                    let psi_hist = if cfg.tensile_split {
                        // Principal strains via the symmetric 3×3
                        // eigen-solver, Macaulay brackets per mode.
                        let (evals, _) = crate::math::dense3::eig3_sym(eps);
                        let pos = |x: Real| if x > 0.0 { x } else { 0.0 };
                        let tr_pos = tr.max(0.0);
                        0.5 * lambda * tr_pos * tr_pos
                            + mu * (pos(evals.x) * pos(evals.x)
                                + pos(evals.y) * pos(evals.y)
                                + pos(evals.z) * pos(evals.z))
                    } else {
                        psi_full
                    };
                    (vol, psi_hist, psi_full)
                }
            };
            let d_bar =
                verts.iter().map(|&v| damage[v as usize]).sum::<Real>() / verts.len() as Real;
            // Reported energy is the FULL physical elastic density —
            // the split feeds only the damage-driving history.
            strain_energy +=
                measure * psi_full * phase_field_degradation(d_bar, params.residual_stiffness);
            for &vi in verts {
                psi_node[vi as usize] += measure * psi_hist;
                area_node[vi as usize] += measure;
            }
        }
        for i in 0..n {
            if area_node[i] > 0.0 {
                history[i] = update_phase_field_history(history[i], psi_node[i] / area_node[i]);
            }
        }

        // ── 3. Damage sweep (bounded + irreversible).
        let result = solve_phase_field_fracture_jacobi(
            mesh,
            damage,
            history,
            params,
            cfg.damage_iters,
            cfg.damage_tol,
        )?;
        let delta = result
            .damage
            .iter()
            .zip(damage.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, Real::max);
        *damage = result.damage;
        if delta < cfg.tol_damage {
            break;
        }
    }

    Ok(PhaseFieldCoupledResult {
        displacement,
        staggered_iters,
        strain_energy,
    })
}
