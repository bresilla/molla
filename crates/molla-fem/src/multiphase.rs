//! Multi-phase-flow primitives.
//!
//! This module provides the CPU oracle pieces needed by a level-set /
//! continuum-surface-force (CSF) incompressible two-phase solver:
//! curvature from a signed-distance field and the corresponding
//! surface-tension body force.

use crate::core::{Error, Result};
use crate::math::Real;

/// Smooth compact delta used by the CSF force.
#[must_use]
pub fn level_set_smooth_delta(phi: Real, eps: Real) -> Real {
    if eps <= 0.0 || phi.abs() >= eps {
        0.0
    } else {
        0.5 / eps * (1.0 + (std::f64::consts::PI * phi / eps).cos())
    }
}

/// Compute curvature `κ = ∇·(∇φ/|∇φ|)` on a periodic 2D grid.
pub fn level_set_curvature_2d(phi: &[Real], nx: usize, ny: usize, dx: Real) -> Result<Vec<Real>> {
    if nx == 0 || ny == 0 || phi.len() != nx * ny || dx <= 0.0 {
        return Err(Error::Build(format!(
            "level_set_curvature_2d: expected phi len {} and positive dx, got len {}, dx {}",
            nx * ny,
            phi.len(),
            dx
        )));
    }
    let idx = |i: isize, j: isize| -> usize {
        let ii = ((i % nx as isize) + nx as isize) % nx as isize;
        let jj = ((j % ny as isize) + ny as isize) % ny as isize;
        jj as usize * nx + ii as usize
    };
    let mut nx_field = vec![0.0; nx * ny];
    let mut ny_field = vec![0.0; nx * ny];
    for j in 0..ny as isize {
        for i in 0..nx as isize {
            let gx = (phi[idx(i + 1, j)] - phi[idx(i - 1, j)]) / (2.0 * dx);
            let gy = (phi[idx(i, j + 1)] - phi[idx(i, j - 1)]) / (2.0 * dx);
            let len = (gx * gx + gy * gy).sqrt().max(1.0e-12);
            let k = idx(i, j);
            nx_field[k] = gx / len;
            ny_field[k] = gy / len;
        }
    }
    let mut kappa = vec![0.0; nx * ny];
    for j in 0..ny as isize {
        for i in 0..nx as isize {
            let div_nx = (nx_field[idx(i + 1, j)] - nx_field[idx(i - 1, j)]) / (2.0 * dx);
            let div_ny = (ny_field[idx(i, j + 1)] - ny_field[idx(i, j - 1)]) / (2.0 * dx);
            kappa[idx(i, j)] = div_nx + div_ny;
        }
    }
    Ok(kappa)
}

/// Compute the continuum-surface-force body force
/// `f = σ κ δ_ε(φ) ∇φ` on a periodic 2D grid.
pub fn csf_surface_tension_force_2d(
    phi: &[Real],
    nx: usize,
    ny: usize,
    dx: Real,
    sigma: Real,
) -> Result<Vec<[Real; 2]>> {
    let kappa = level_set_curvature_2d(phi, nx, ny, dx)?;
    let idx = |i: isize, j: isize| -> usize {
        let ii = ((i % nx as isize) + nx as isize) % nx as isize;
        let jj = ((j % ny as isize) + ny as isize) % ny as isize;
        jj as usize * nx + ii as usize
    };
    let eps = 1.5 * dx;
    let mut f = vec![[0.0, 0.0]; nx * ny];
    for j in 0..ny as isize {
        for i in 0..nx as isize {
            let k = idx(i, j);
            let gx = (phi[idx(i + 1, j)] - phi[idx(i - 1, j)]) / (2.0 * dx);
            let gy = (phi[idx(i, j + 1)] - phi[idx(i, j - 1)]) / (2.0 * dx);
            let delta = level_set_smooth_delta(phi[k], eps);
            let scale = sigma * kappa[k] * delta;
            f[k] = [scale * gx, scale * gy];
        }
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_curvature_matches_inverse_radius_near_interface() {
        let n = 64usize;
        let dx = 1.0 / n as Real;
        let r = 0.2;
        let mut phi = vec![0.0; n * n];
        for j in 0..n {
            let y = (j as Real + 0.5) * dx - 0.5;
            for i in 0..n {
                let x = (i as Real + 0.5) * dx - 0.5;
                phi[j * n + i] = (x * x + y * y).sqrt() - r;
            }
        }
        let kappa = level_set_curvature_2d(&phi, n, n, dx).unwrap();
        let mut sum = 0.0;
        let mut count = 0usize;
        for (p, k) in phi.iter().zip(kappa.iter()) {
            if p.abs() < dx {
                sum += *k;
                count += 1;
            }
        }
        let avg = sum / count as Real;
        assert!(
            (avg - 1.0 / r).abs() / (1.0 / r) < 0.1,
            "curvature avg {avg} should be close to {}",
            1.0 / r
        );
    }
}
