use super::{PointGrab, StrandModel, bond_rows, invalid};
use molla_core::Result;
use molla_math::{Quat, Real, Vec3, dense_solve::dense_chol_solve};

#[derive(Clone, Copy, Debug)]
pub struct CoupledConfig {
    /// Frozen-Jacobian solves per timestep.
    pub nonlinear_iterations: usize,
    /// GPU PCG iteration cap; the CPU oracle uses dense Cholesky.
    pub linear_iterations: usize,
    /// GPU tolerance in the block-preconditioned residual norm, relative to the step's first
    /// Newton residual.
    pub relative_tolerance: Real,
    /// Absolute tolerance in the block-preconditioned residual norm; zero disables this criterion.
    pub absolute_tolerance: Real,
}

impl Default for CoupledConfig {
    fn default() -> Self {
        Self {
            nonlinear_iterations: 4,
            linear_iterations: 256,
            relative_tolerance: 1e-5,
            absolute_tolerance: 0.0,
        }
    }
}

impl CoupledConfig {
    pub fn validate(self) -> Result<()> {
        if !(1..=64).contains(&self.nonlinear_iterations)
            || !(1..=2048).contains(&self.linear_iterations)
            || !self.relative_tolerance.is_finite()
            || !(1e-7..=1e-2).contains(&self.relative_tolerance)
            || !self.absolute_tolerance.is_finite()
            || self.absolute_tolerance < 0.0
        {
            return Err(invalid("invalid coupled solve settings"));
        }
        Ok(())
    }
}

pub(super) fn project(
    model: &mut StrandModel,
    lambdas: &mut [[Real; 6]],
    dt: Real,
    grab: Option<PointGrab>,
    grab_lambda: &mut Vec3,
) -> Result<()> {
    let n = model.fibers.len() * 6;
    if n > 4096 {
        return Err(invalid("dense CPU strand oracle exceeds 4096 coordinates"));
    }
    let mut matrix = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    for (i, fiber) in model.fibers.iter().enumerate() {
        if fiber.pin.is_some() {
            for k in 0..6 {
                matrix[(6 * i + k) * n + 6 * i + k] = 1.0;
            }
            continue;
        }
        let rotation = fiber.motion.pose.rotation_matrix();
        let inertia = (rotation * fiber.mass.inertia * rotation.transpose()).to_cols_array_2d();
        for a in 0..3 {
            matrix[(6 * i + a) * n + 6 * i + a] = fiber.mass.mass;
            for b in 0..3 {
                matrix[(6 * i + a + 3) * n + 6 * i + b + 3] = inertia[b][a];
            }
        }
    }
    let mut frozen = Vec::new();
    for (j, bond) in model.bonds.iter().enumerate() {
        if !bond.intact() {
            continue;
        }
        let ids = bond.fibers.map(|id| model.indices[&id]);
        let rows = bond_rows(bond, &model.fibers[ids[0]], &model.fibers[ids[1]]);
        for (axis, row) in rows.into_iter().enumerate() {
            let alpha = row.compliance / (dt * dt);
            let mut gradient = Vec::with_capacity(12);
            for side in 0..2 {
                if model.fibers[ids[side]].pin.is_some() {
                    continue;
                }
                for k in 0..3 {
                    gradient.push((6 * ids[side] + k, row.linear[side][k]));
                }
                for k in 0..3 {
                    gradient.push((6 * ids[side] + k + 3, row.angular[side][k]));
                }
            }
            for &(a, ga) in &gradient {
                rhs[a] -= ga * (row.value / alpha + lambdas[j][axis]);
                for &(b, gb) in &gradient {
                    matrix[a * n + b] += ga * gb / alpha;
                }
            }
            frozen.push((j, axis, alpha, row.value, gradient));
        }
    }
    if let Some(grab) = grab {
        let i = model.indices[&grab.fiber];
        let fiber = &model.fibers[i];
        let r = fiber.motion.pose.rotation * (grab.local_point - fiber.mass.local_com);
        let error = fiber.center() + r - grab.target;
        let alpha = grab.compliance_at(fiber.center() + r) / (dt * dt);
        for axis in 0..3 {
            let direction = [Vec3::X, Vec3::Y, Vec3::Z][axis];
            let angular = r.cross(direction);
            let gradient: Vec<_> = (0..6)
                .map(|k| (6 * i + k, if k < 3 { direction[k] } else { angular[k - 3] }))
                .collect();
            for &(a, ga) in &gradient {
                rhs[a] -= ga * (error[axis] / alpha + grab_lambda[axis]);
                for &(b, gb) in &gradient {
                    matrix[a * n + b] += ga * gb / alpha;
                }
            }
            frozen.push((model.bonds.len(), axis, alpha, error[axis], gradient));
        }
    }
    if matrix.iter().chain(&rhs).any(|v| !v.is_finite()) {
        return Err(invalid("coupled system overflow"));
    }
    let diagonal: Vec<_> = (0..n).map(|i| matrix[i * n + i].sqrt()).collect();
    for a in 0..n {
        rhs[a] /= diagonal[a];
        for b in 0..n {
            matrix[a * n + b] /= diagonal[a] * diagonal[b];
        }
    }
    let system = matrix.clone();
    let mut delta = dense_chol_solve(&mut matrix, &rhs, n)?;
    let mut residual: Real = 0.0;
    let mut scale: Real = 0.0;
    for (row, &value) in system.chunks_exact(n).zip(&rhs) {
        let product = row.iter().zip(&delta).map(|(a, x)| a * x).sum::<Real>();
        residual = residual.max((product - value).abs());
        scale = scale.max(
            value.abs()
                + row
                    .iter()
                    .zip(&delta)
                    .map(|(a, x)| (a * x).abs())
                    .sum::<Real>(),
        );
    }
    if delta.iter().any(|v| !v.is_finite()) || residual > 1e-10 * scale.max(1e-30) {
        return Err(invalid("coupled CPU solve failed its residual check"));
    }
    for (value, scale) in delta.iter_mut().zip(diagonal) {
        *value /= scale;
    }
    for (j, axis, alpha, value, gradient) in frozen {
        let linearized = value + gradient.iter().map(|&(i, g)| g * delta[i]).sum::<Real>();
        if j == model.bonds.len() {
            grab_lambda[axis] = -linearized / alpha;
        } else {
            lambdas[j][axis] = -linearized / alpha;
        }
    }
    for (i, fiber) in model.fibers.iter_mut().enumerate() {
        if fiber.pin.is_some() {
            continue;
        }
        let center = fiber.center() + Vec3::new(delta[6 * i], delta[6 * i + 1], delta[6 * i + 2]);
        let angular = Vec3::new(delta[6 * i + 3], delta[6 * i + 4], delta[6 * i + 5]);
        fiber.motion.pose.rotation =
            (Quat::from_scaled_axis(angular) * fiber.motion.pose.rotation).normalize();
        fiber.motion.pose.position = center - fiber.motion.pose.rotation * fiber.mass.local_com;
    }
    Ok(())
}
