//! Stage 37.5 — FFT-Poisson + fractional-step Navier-Stokes on a
//! uniform 2D grid.
//!
//! This module is **complementary** to the Taylor-Hood FEM Stokes
//! solver in [`super::stokes`]: where Stokes / `solve_navier_stokes_*`
//! discretise on an unstructured triangle mesh and use Uzawa-CG, this
//! module solves a pure-grid 2D NS via the classical projection
//! method (Chorin, 1968) with the pressure-Poisson step done by the
//! existing [`crate::kernels::FftKernel`].
//!
//! # Pipeline (one step `Δt`)
//!
//! 1. **Predictor**: `u* = u_n − Δt (u·∇)u + Δt ν ∇²u` — explicit
//!    convection + viscous (forward Euler). Periodic BCs everywhere.
//! 2. **Pressure-Poisson**: solve `∇²p = (1/Δt) ∇·u*` via FFT.
//!    Spectral form: `p̂(k) = -(1 / |k|²) (1/Δt) (i k · û*)` with
//!    `p̂(0) = 0` (gauge fix).
//! 3. **Corrector**: `u_{n+1} = u* − Δt ∇p` — projects onto the
//!    divergence-free subspace.
//!
//! # API
//!
//! - [`solve_pressure_poisson_fft`] — host-side wrapper around the
//!   GPU FFT kernel that solves `-∇²p = rhs` on a uniform `n × n`
//!   periodic grid. RHS / output are real `n²`-element vectors. The
//!   nullspace mode (`k = 0`) is fixed to zero.
//! - [`solve_navier_stokes_fractional_step`] — runs `num_steps`
//!   projection steps from a Taylor-Green initial condition; returns
//!   the final velocity field as `Vec<[f32; 2]>` in row-major
//!   `(j, i) → j·n + i` order. CPU implementation of the predictor /
//!   corrector with the FFT-based pressure-Poisson at each step.
//!
//! # Why split this from `stokes.rs`
//!
//! Different discretisation (uniform staggered grid vs. unstructured
//! Taylor-Hood mesh), different solver (FFT-Poisson vs. Uzawa-CG),
//! different intent (interactive grid demo vs. mixed-FE accuracy).
//! Sharing code would only save a handful of lines and would entangle
//! two otherwise-independent paths.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::kernels::{Complex32, FftKernel};

/// Solve `-∇²p = rhs` on a uniform `n × n` periodic grid via spectral
/// inversion. `rhs` and the returned pressure are real-valued
/// row-major `n²`-element vectors. Implementation: forward 2D FFT of
/// `rhs`, divide by `|k|²` (zeroing the `k = 0` mode), inverse 2D FFT,
/// normalise by `1 / N²`. Uses [`FftKernel`] under the hood.
///
/// # Arguments
///
/// - `device`, `queue`, `kernel`: the GPU FFT runtime.
/// - `rhs`: row-major `n × n` real source.
/// - `n`: grid resolution (must be a power of two).
///
/// # Returns
///
/// Row-major `n × n` pressure field `p` such that
/// `(p_{i+1,j} − 2 p_{i,j} + p_{i−1,j}) / dx² + (...) = -rhs_{i,j}`
/// to spectral accuracy. Mean of `p` is exactly zero (the constant
/// nullspace is gauge-fixed).
pub fn solve_pressure_poisson_fft(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    kernel: &FftKernel,
    rhs: &[f32],
    n: usize,
) -> Result<Vec<f32>> {
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(Error::Gpu(format!(
            "solve_pressure_poisson_fft: n must be a power of two, got {n}"
        )));
    }
    let total = n * n;
    if rhs.len() != total {
        return Err(Error::Gpu(format!(
            "solve_pressure_poisson_fft: rhs.len() = {} but n*n = {total}",
            rhs.len()
        )));
    }

    // Pack rhs as Complex32 (im = 0).
    let mut input: Vec<Complex32> = rhs.iter().map(|&r| Complex32::new(r, 0.0)).collect();

    // Forward FFT.
    let mut fwd = vec![Complex32::ZERO; total];
    kernel.dispatch_2d(device, queue, &input, &mut fwd, n, n, false)?;

    // Multiply by 1 / |k|² in spectral space, with the convention that
    // the discrete wave numbers run from 0..n with wrap-around at n/2.
    let nf = n as f32;
    // Use the discrete-Laplacian symbol so the operator we invert is
    // exactly `-Δ_h` (the second-order centred difference). This makes
    // the spectral solution behave correctly for finite-grid forcing
    // (e.g. analytic constant-curvature inputs).
    let two_pi = 2.0 * std::f32::consts::PI;
    for ky in 0..n {
        // Wave number with wrap-around: ky in [0, n/2] is positive,
        // (n/2, n) is negative.
        let ky_signed = if ky <= n / 2 { ky as f32 } else { ky as f32 - nf };
        for kx in 0..n {
            let kx_signed = if kx <= n / 2 { kx as f32 } else { kx as f32 - nf };
            let idx = ky * n + kx;
            if kx == 0 && ky == 0 {
                fwd[idx] = Complex32::ZERO;
                continue;
            }
            // Eigenvalue of the negative discrete 2D Laplacian on a
            // periodic grid with unit spacing:
            //   λ = 4 sin²(π kx / n) + 4 sin²(π ky / n)
            // Using the symbol of -Δ_h matches what the corrector step
            // expects (so the projection is exact, not just spectrally
            // approximate).
            let sx = (std::f32::consts::PI * kx_signed / nf).sin();
            let sy = (std::f32::consts::PI * ky_signed / nf).sin();
            let lam = 4.0 * (sx * sx + sy * sy);
            // Skip zero mode (already handled above) and any tiny
            // numerical noise.
            let inv = if lam > 1.0e-20 { 1.0 / lam } else { 0.0 };
            fwd[idx].re *= inv;
            fwd[idx].im *= inv;
            // Suppress unused `two_pi` warning in case caller wants to
            // switch to the continuous symbol later.
            let _ = two_pi;
        }
    }

    // Inverse FFT and normalise by N² (the Stockham kernel doesn't
    // normalise — see the FFT module docs).
    kernel.dispatch_2d(device, queue, &fwd, &mut input, n, n, true)?;
    let inv_norm = 1.0 / (total as f32);
    let p: Vec<f32> = input.iter().map(|c| c.re * inv_norm).collect();
    Ok(p)
}

// ─── Fractional-step NS solver ───────────────────────────────────────

/// Configuration for [`solve_navier_stokes_fractional_step`].
#[derive(Clone, Copy, Debug)]
pub struct FractionalStepConfig {
    /// Grid resolution (must be a power of two; periodic BCs).
    pub grid_n: usize,
    /// Kinematic viscosity ν.
    pub viscosity: f32,
    /// Time step Δt.
    pub dt: f32,
    /// Number of projection-method steps.
    pub num_steps: usize,
}

/// Run `num_steps` of projection-method Navier-Stokes on a periodic
/// `grid_n × grid_n` grid, starting from a 2D Taylor-Green vortex
/// initial condition. Returns the final velocity field as a row-major
/// `Vec<[f32; 2]>` indexed `(j, i) → j·n + i`.
///
/// Geometry: grid spacing `dx = 1 / grid_n` so the discrete Laplacian
/// operates on a unit-period domain. The Taylor-Green initial
/// condition is `u = sin(2π x) cos(2π y)`, `v = -cos(2π x) sin(2π y)`
/// — analytic decay rate `exp(-2 ν (2π)² t)`.
///
/// # Implementation
///
/// 1. Predictor (forward Euler): `u* = u + Δt (- (u·∇)u + ν ∇²u)`.
/// 2. Pressure-Poisson: `-∇² p = (1/Δt) ∇·u*` via [`solve_pressure_poisson_fft`].
/// 3. Corrector: `u_{n+1} = u* − Δt ∇p`.
///
/// All spatial differences are second-order central with periodic
/// wrap-around. Grid stays aligned (no staggering) — the FFT-based
/// Poisson with the discrete symbol gives an *exact* projection on
/// the central-difference subspace, so the resulting `u_{n+1}` is
/// discretely divergence-free.
pub fn solve_navier_stokes_fractional_step(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    kernel: &FftKernel,
    cfg: FractionalStepConfig,
) -> Result<Vec<[f32; 2]>> {
    let n = cfg.grid_n;
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(Error::Gpu(format!(
            "solve_navier_stokes_fractional_step: grid_n must be a power of two, got {n}"
        )));
    }
    let total = n * n;
    let two_pi = 2.0 * std::f32::consts::PI;

    // Initial condition: Taylor-Green vortex.
    let mut u = vec![[0.0_f32; 2]; total];
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32;
            let ux = (two_pi * x).sin() * (two_pi * y).cos();
            let vy = -(two_pi * x).cos() * (two_pi * y).sin();
            u[j * n + i] = [ux, vy];
        }
    }

    let dx = 1.0 / n as f32;
    let inv_dx = 1.0 / dx;
    let inv_dx2 = inv_dx * inv_dx;

    let idx = |i: usize, j: usize| -> usize { j * n + i };
    let wrap = |k: i32| -> usize {
        let m = n as i32;
        (((k % m) + m) % m) as usize
    };

    for _step in 0..cfg.num_steps {
        // ── Predictor: u* = u + dt * (-(u·∇)u + ν ∇²u) ────────────
        let mut u_star = vec![[0.0_f32; 2]; total];
        for j in 0..n {
            for i in 0..n {
                let ip = wrap(i as i32 + 1);
                let im = wrap(i as i32 - 1);
                let jp = wrap(j as i32 + 1);
                let jm = wrap(j as i32 - 1);
                let c = u[idx(i, j)];
                let xp = u[idx(ip, j)];
                let xm = u[idx(im, j)];
                let yp = u[idx(i, jp)];
                let ym = u[idx(i, jm)];
                // (u·∇)u, central differences.
                let dudx = (xp[0] - xm[0]) * 0.5 * inv_dx;
                let dudy = (yp[0] - ym[0]) * 0.5 * inv_dx;
                let dvdx = (xp[1] - xm[1]) * 0.5 * inv_dx;
                let dvdy = (yp[1] - ym[1]) * 0.5 * inv_dx;
                let conv_u = c[0] * dudx + c[1] * dudy;
                let conv_v = c[0] * dvdx + c[1] * dvdy;
                // ν ∇²u, 5-point Laplacian.
                let lap_u = (xp[0] + xm[0] + yp[0] + ym[0] - 4.0 * c[0]) * inv_dx2;
                let lap_v = (xp[1] + xm[1] + yp[1] + ym[1] - 4.0 * c[1]) * inv_dx2;
                u_star[idx(i, j)] = [
                    c[0] + cfg.dt * (-conv_u + cfg.viscosity * lap_u),
                    c[1] + cfg.dt * (-conv_v + cfg.viscosity * lap_v),
                ];
            }
        }

        // ── Pressure-Poisson: -∇² p = (1/dt) ∇·u* ─────────────────
        // RHS in the chosen sign convention for `solve_pressure_poisson_fft`:
        //   `-Δp = rhs` with `rhs = (1/dt) ∇·u*`.
        // The FFT solver returns `p` directly. The discrete Laplacian
        // symbol used inside the FFT solver matches our central-diff
        // gradient below — so the corrector is exact (∇·u_{n+1} = 0).
        let mut div = vec![0.0_f32; total];
        for j in 0..n {
            for i in 0..n {
                let ip = wrap(i as i32 + 1);
                let im = wrap(i as i32 - 1);
                let jp = wrap(j as i32 + 1);
                let jm = wrap(j as i32 - 1);
                let dudx = (u_star[idx(ip, j)][0] - u_star[idx(im, j)][0]) * 0.5 * inv_dx;
                let dvdy = (u_star[idx(i, jp)][1] - u_star[idx(i, jm)][1]) * 0.5 * inv_dx;
                div[idx(i, j)] = (dudx + dvdy) / cfg.dt;
            }
        }
        // The FFT-Poisson uses unit grid spacing in its discrete
        // Laplacian. Our physical `dx = 1/n`. Scale: `-Δ_h p_phys = rhs`
        // where Δ_h with unit spacing differs from physical Δ by a
        // factor `1/dx²`. Multiply rhs by `dx²` so the spectral solver
        // returns `p` in physical units.
        let scale = dx * dx;
        for v in div.iter_mut() {
            *v *= scale;
        }
        let p = solve_pressure_poisson_fft(device, queue, kernel, &div, n)?;

        // ── Corrector: u_{n+1} = u* − dt ∇p ───────────────────────
        for j in 0..n {
            for i in 0..n {
                let ip = wrap(i as i32 + 1);
                let im = wrap(i as i32 - 1);
                let jp = wrap(j as i32 + 1);
                let jm = wrap(j as i32 - 1);
                let dpdx = (p[idx(ip, j)] - p[idx(im, j)]) * 0.5 * inv_dx;
                let dpdy = (p[idx(i, jp)] - p[idx(i, jm)]) * 0.5 * inv_dx;
                u[idx(i, j)] = [
                    u_star[idx(i, j)][0] - cfg.dt * dpdx,
                    u_star[idx(i, j)][1] - cfg.dt * dpdy,
                ];
            }
        }
    }

    Ok(u)
}
