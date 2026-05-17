//! Stage 39C — 2-D Stam stable-fluid solver on a periodic grid.
//!
//! Port of Jos Stam's "Stable Fluids" (1999), wired into molla as a
//! user-friendly Eulerian fluid API for the three Warp fluid-style
//! demos (`example_fluid.py`, `example_fluid_checkpoint.py`,
//! `example_navier_stokes_perturbation.py`).
//!
//! # Algorithm
//!
//! One step of the Stam scheme on `n × n` periodic grids:
//!
//! 1. **Add forces / sources** — `u += dt * f`, `density += dt * source`.
//! 2. **Diffuse velocity** — implicit Gauss-Seidel iteration solves
//!    `(I − ν dt Δ_h) u_{n+1} = u_n`.
//! 3. **Project velocity** — reuse [`solve_pressure_poisson_fft`] so
//!    `∇·u_{n+1} = 0`. CPU path uses a direct Jacobi pressure solve so
//!    tests can run without a GPU context.
//! 4. **Advect velocity** — semi-Lagrangian backward trace + bilinear
//!    sample at the departure point. Periodic wrap.
//! 5. **Project velocity** again — semi-Lagrangian advection is not
//!    divergence-preserving, so re-project to keep the field clean.
//! 6. **Advect density** — same semi-Lagrangian trick on the scalar
//!    field.
//! 7. **Diffuse density** — Gauss-Seidel `(I − κ dt Δ_h) ρ = ρ_advected`.
//!    `κ` reuses `viscosity` (density diffusion is the same operator).
//!
//! All differences are second-order centred with periodic wrap, matching
//! the discrete Laplacian symbol the FFT-Poisson inverts so the
//! projection is exact for the operator we apply.
//!
//! # Why composition, not extension of `fft_ns`
//!
//! - `fft_ns` is a fractional-step Navier-Stokes with **forward Euler**
//!   convection and a Taylor-Green initial condition baked in. The
//!   Stam scheme uses **semi-Lagrangian** advection (unconditionally
//!   stable for any dt), Gauss-Seidel diffusion, and arbitrary
//!   user-supplied force / source fields. Two different solvers that
//!   happen to share one helper.
//! - Reusing `solve_pressure_poisson_fft` (when GPU is available) keeps
//!   the FFT plumbing in one place.

#![allow(clippy::needless_range_loop)]

#[cfg(feature = "gpu")]
use crate::core::Result;

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::fem::{PoissonStencil, solve_pressure_poisson_fft_with_stencil};
#[cfg(feature = "gpu")]
use crate::kernels::FftKernel;

/// Configuration for one Stam stable-fluid step.
#[derive(Clone, Copy, Debug)]
pub struct StamFluidConfig {
    /// Grid resolution. Power of two so the GPU FFT-Poisson projection
    /// works; the CPU path also requires power-of-two for the spectral
    /// projection oracle (see [`StamFluidState::step_cpu`]).
    pub grid_n: usize,
    /// Kinematic viscosity ν (also reused as the density-diffusion
    /// constant κ — they are the same operator).
    pub viscosity: f32,
    /// Time step Δt.
    pub dt: f32,
    /// Number of Gauss-Seidel sweeps for the diffusion solve. Stam
    /// recommends ~20 for stability; lower values are visibly diffuse
    /// but stable.
    pub n_diffusion_iters: usize,
    /// Number of Jacobi sweeps for the CPU pressure-projection solve.
    /// Only used by [`StamFluidState::step_cpu`]; the GPU path uses the
    /// direct FFT-Poisson and ignores this. ~40 iterations is the
    /// canonical Stam value; more iterations → tighter divergence.
    pub n_pressure_iters: usize,
}

impl Default for StamFluidConfig {
    fn default() -> Self {
        Self {
            grid_n: 64,
            viscosity: 1.0e-3,
            dt: 1.0 / 60.0,
            n_diffusion_iters: 20,
            n_pressure_iters: 60,
        }
    }
}

/// Mutable simulation state for the 2D Stam fluid.
///
/// `u` is the velocity field (`[ux, uy]` per cell) and `density` is the
/// advected scalar (a "dye" field in Stam's original paper). Both are
/// `grid_n × grid_n` row-major arrays indexed `(j, i) → j·n + i`.
#[derive(Clone, Debug)]
pub struct StamFluidState {
    /// Grid resolution (cached for bounds checks).
    pub grid_n: usize,
    /// Velocity field, `n²` cells.
    pub u: Vec<[f32; 2]>,
    /// Advected scalar field, `n²` cells.
    pub density: Vec<f32>,
}

impl StamFluidState {
    /// Construct a zero-initialised `n × n` fluid state.
    pub fn new(n: usize) -> Self {
        Self {
            grid_n: n,
            u: vec![[0.0_f32; 2]; n * n],
            density: vec![0.0_f32; n * n],
        }
    }

    /// One Stam step (CPU). `force` and `source` are `n²`-element
    /// row-major arrays added before the diffuse / project / advect
    /// pipeline; pass zero-filled slices if there's no forcing.
    ///
    /// Pressure projection uses an iterative Jacobi solve so the CPU
    /// path is self-contained (no GPU device required, useful for
    /// tests). `cfg.n_pressure_iters` controls how tightly the
    /// projection drives the divergence to zero.
    pub fn step_cpu(&mut self, cfg: &StamFluidConfig, force: &[[f32; 2]], source: &[f32]) {
        let n = cfg.grid_n;
        debug_assert_eq!(self.grid_n, n);
        debug_assert_eq!(self.u.len(), n * n);
        debug_assert_eq!(self.density.len(), n * n);
        debug_assert_eq!(force.len(), n * n);
        debug_assert_eq!(source.len(), n * n);

        // 1. Add forces / sources.
        for k in 0..n * n {
            self.u[k][0] += cfg.dt * force[k][0];
            self.u[k][1] += cfg.dt * force[k][1];
            self.density[k] += cfg.dt * source[k];
        }

        // 2. Diffuse velocity (Gauss-Seidel).
        let dx = 1.0 / n as f32;
        let alpha = cfg.viscosity * cfg.dt / (dx * dx);
        diffuse_vec_periodic(&mut self.u, n, alpha, cfg.n_diffusion_iters);

        // 3. Project velocity (CPU Jacobi).
        project_jacobi_periodic(&mut self.u, n, cfg.n_pressure_iters);

        // 4. Advect velocity (semi-Lagrangian backward trace).
        let u_advected = advect_vec_periodic(&self.u, &self.u, n, cfg.dt);
        self.u = u_advected;

        // 5. Project again.
        project_jacobi_periodic(&mut self.u, n, cfg.n_pressure_iters);

        // 6. Advect density along the (now divergence-free) velocity.
        let rho_advected = advect_scalar_periodic(&self.density, &self.u, n, cfg.dt);
        self.density = rho_advected;

        // 7. Diffuse density (same Gauss-Seidel scheme).
        diffuse_scalar_periodic(&mut self.density, n, alpha, cfg.n_diffusion_iters);
    }

    /// One Stam step (GPU-accelerated projection). Same pipeline as
    /// [`Self::step_cpu`] but uses [`solve_pressure_poisson_fft`] for
    /// the two projection passes — exact-to-spectral-accuracy on the
    /// wide-stencil Laplacian that our central-difference
    /// gradient/divergence pair implies.
    ///
    /// Diffusion + advection are still on the host; this matches the
    /// FFT-NS pattern (Stage 37.5) and keeps the per-frame transfer
    /// cost identical (one host→device for the divergence, one
    /// device→host for the pressure).
    #[cfg(feature = "gpu")]
    pub fn step_gpu(
        &mut self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        kernel: &FftKernel,
        cfg: &StamFluidConfig,
        force: &[[f32; 2]],
        source: &[f32],
    ) -> Result<()> {
        let n = cfg.grid_n;
        debug_assert_eq!(self.grid_n, n);
        debug_assert_eq!(self.u.len(), n * n);
        debug_assert_eq!(self.density.len(), n * n);
        debug_assert_eq!(force.len(), n * n);
        debug_assert_eq!(source.len(), n * n);

        // 1. Add forces / sources.
        for k in 0..n * n {
            self.u[k][0] += cfg.dt * force[k][0];
            self.u[k][1] += cfg.dt * force[k][1];
            self.density[k] += cfg.dt * source[k];
        }

        // 2. Diffuse velocity.
        let dx = 1.0 / n as f32;
        let alpha = cfg.viscosity * cfg.dt / (dx * dx);
        diffuse_vec_periodic(&mut self.u, n, alpha, cfg.n_diffusion_iters);

        // 3. Project (FFT).
        project_fft_periodic(device, queue, kernel, &mut self.u, n)?;

        // 4. Advect velocity.
        self.u = advect_vec_periodic(&self.u, &self.u, n, cfg.dt);

        // 5. Project again.
        project_fft_periodic(device, queue, kernel, &mut self.u, n)?;

        // 6. Advect density.
        self.density = advect_scalar_periodic(&self.density, &self.u, n, cfg.dt);

        // 7. Diffuse density.
        diffuse_scalar_periodic(&mut self.density, n, alpha, cfg.n_diffusion_iters);

        Ok(())
    }

    /// Compute the discrete divergence of `u` (central differences,
    /// periodic wrap, unit grid spacing). Useful for tests / diagnostics.
    pub fn divergence(&self) -> Vec<f32> {
        let n = self.grid_n;
        let mut div = vec![0.0_f32; n * n];
        let inv_dx = n as f32;
        for j in 0..n {
            for i in 0..n {
                let ip = (i + 1) % n;
                let im = (i + n - 1) % n;
                let jp = (j + 1) % n;
                let jm = (j + n - 1) % n;
                let dudx = (self.u[j * n + ip][0] - self.u[j * n + im][0]) * 0.5 * inv_dx;
                let dvdy = (self.u[jp * n + i][1] - self.u[jm * n + i][1]) * 0.5 * inv_dx;
                div[j * n + i] = dudx + dvdy;
            }
        }
        div
    }

    /// Sum of `density`. With no source and exact-projection
    /// advection, this should be preserved up to floating-point noise.
    pub fn total_density(&self) -> f32 {
        self.density.iter().sum()
    }
}

// ─── Helpers (CPU) ──────────────────────────────────────────────────

/// Periodic bilinear lookup on an `n × n` scalar grid sampled at
/// real-valued `(x, y)` cell coordinates.
fn sample_scalar_periodic(grid: &[f32], n: usize, x: f32, y: f32) -> f32 {
    let nx = n as f32;
    // Wrap to `[0, n)`.
    let xw = x.rem_euclid(nx);
    let yw = y.rem_euclid(nx);
    let i0 = xw.floor() as usize % n;
    let j0 = yw.floor() as usize % n;
    let i1 = (i0 + 1) % n;
    let j1 = (j0 + 1) % n;
    let tx = xw - xw.floor();
    let ty = yw - yw.floor();
    let s00 = grid[j0 * n + i0];
    let s10 = grid[j0 * n + i1];
    let s01 = grid[j1 * n + i0];
    let s11 = grid[j1 * n + i1];
    let s0 = s00 * (1.0 - tx) + s10 * tx;
    let s1 = s01 * (1.0 - tx) + s11 * tx;
    s0 * (1.0 - ty) + s1 * ty
}

/// Periodic bilinear lookup of a vector field.
fn sample_vec_periodic(grid: &[[f32; 2]], n: usize, x: f32, y: f32) -> [f32; 2] {
    let nx = n as f32;
    let xw = x.rem_euclid(nx);
    let yw = y.rem_euclid(nx);
    let i0 = xw.floor() as usize % n;
    let j0 = yw.floor() as usize % n;
    let i1 = (i0 + 1) % n;
    let j1 = (j0 + 1) % n;
    let tx = xw - xw.floor();
    let ty = yw - yw.floor();
    let s00 = grid[j0 * n + i0];
    let s10 = grid[j0 * n + i1];
    let s01 = grid[j1 * n + i0];
    let s11 = grid[j1 * n + i1];
    let sx0 = s00[0] * (1.0 - tx) + s10[0] * tx;
    let sx1 = s01[0] * (1.0 - tx) + s11[0] * tx;
    let sy0 = s00[1] * (1.0 - tx) + s10[1] * tx;
    let sy1 = s01[1] * (1.0 - tx) + s11[1] * tx;
    [sx0 * (1.0 - ty) + sx1 * ty, sy0 * (1.0 - ty) + sy1 * ty]
}

/// Semi-Lagrangian backward-trace advection of a scalar field along
/// `vel`. Trace back `dt` from each cell centre and bilinearly sample
/// the previous field. Periodic wrap.
fn advect_scalar_periodic(rho: &[f32], vel: &[[f32; 2]], n: usize, dt: f32) -> Vec<f32> {
    let nx = n as f32;
    let mut out = vec![0.0_f32; n * n];
    // Velocity is in "domain units / sec" (i.e. one period crossed per
    // unit velocity per unit time); the grid is `n` cells per period,
    // so the per-cell departure offset is `vel * dt * n`.
    for j in 0..n {
        for i in 0..n {
            let v = vel[j * n + i];
            let x = (i as f32 + 0.5) - v[0] * dt * nx;
            let y = (j as f32 + 0.5) - v[1] * dt * nx;
            // sample at cell-centre offset → subtract 0.5 to get cell coord.
            out[j * n + i] = sample_scalar_periodic(rho, n, x - 0.5, y - 0.5);
        }
    }
    out
}

/// Semi-Lagrangian advection of a vector field. Same trace as
/// [`advect_scalar_periodic`] applied to `field` along `vel`.
fn advect_vec_periodic(field: &[[f32; 2]], vel: &[[f32; 2]], n: usize, dt: f32) -> Vec<[f32; 2]> {
    let nx = n as f32;
    let mut out = vec![[0.0_f32; 2]; n * n];
    for j in 0..n {
        for i in 0..n {
            let v = vel[j * n + i];
            let x = (i as f32 + 0.5) - v[0] * dt * nx;
            let y = (j as f32 + 0.5) - v[1] * dt * nx;
            out[j * n + i] = sample_vec_periodic(field, n, x - 0.5, y - 0.5);
        }
    }
    out
}

/// Gauss-Seidel sweep solving `(I − α Δ_h) x_new = x` in place on a
/// periodic grid. `α = ν dt / dx²`. Stam's canonical form.
fn diffuse_scalar_periodic(grid: &mut [f32], n: usize, alpha: f32, iters: usize) {
    if alpha == 0.0 || iters == 0 {
        return;
    }
    let x0 = grid.to_vec();
    let inv = 1.0 / (1.0 + 4.0 * alpha);
    for _ in 0..iters {
        for j in 0..n {
            let jp = (j + 1) % n;
            let jm = (j + n - 1) % n;
            for i in 0..n {
                let ip = (i + 1) % n;
                let im = (i + n - 1) % n;
                let s = grid[j * n + ip] + grid[j * n + im] + grid[jp * n + i] + grid[jm * n + i];
                grid[j * n + i] = (x0[j * n + i] + alpha * s) * inv;
            }
        }
    }
}

/// Vector-field version of [`diffuse_scalar_periodic`] — diffuses each
/// component independently.
fn diffuse_vec_periodic(grid: &mut [[f32; 2]], n: usize, alpha: f32, iters: usize) {
    if alpha == 0.0 || iters == 0 {
        return;
    }
    let x0 = grid.to_vec();
    let inv = 1.0 / (1.0 + 4.0 * alpha);
    for _ in 0..iters {
        for j in 0..n {
            let jp = (j + 1) % n;
            let jm = (j + n - 1) % n;
            for i in 0..n {
                let ip = (i + 1) % n;
                let im = (i + n - 1) % n;
                let sx = grid[j * n + ip][0]
                    + grid[j * n + im][0]
                    + grid[jp * n + i][0]
                    + grid[jm * n + i][0];
                let sy = grid[j * n + ip][1]
                    + grid[j * n + im][1]
                    + grid[jp * n + i][1]
                    + grid[jm * n + i][1];
                grid[j * n + i][0] = (x0[j * n + i][0] + alpha * sx) * inv;
                grid[j * n + i][1] = (x0[j * n + i][1] + alpha * sy) * inv;
            }
        }
    }
}

/// CPU pressure projection: Gauss-Seidel iterate `Δ p = ∇·u` on the
/// **wide-stencil** Laplacian (periodic), then subtract `∇p` from `u`.
///
/// We use central first differences for both the divergence (∇·u_i =
/// (u_{i+1} − u_{i−1}) / 2dx) and the gradient (∇p_i = (p_{i+1} −
/// p_{i−1}) / 2dx). Their composition is the wide-stencil Laplacian
/// `(p_{i+2} − 2 p_i + p_{i−2}) / (4 dx²)`, which decouples into two
/// independent 5-point stencils on the even / odd sub-grids. Iterating
/// on this exact discrete operator keeps the projection consistent
/// with the divergence test in [`StamFluidState::divergence`] (which
/// also uses central differences) and with the GPU FFT-Poisson path
/// (which already invokes `PoissonStencil::WideStencil` to match its
/// central-difference corrector — see `fem::fft_ns`).
fn project_jacobi_periodic(u: &mut [[f32; 2]], n: usize, iters: usize) {
    if iters == 0 {
        return;
    }
    let inv_dx = n as f32;
    // Central-difference divergence (stride 2, factor 1/(2 dx)).
    let mut div = vec![0.0_f32; n * n];
    for j in 0..n {
        let jp = (j + 1) % n;
        let jm = (j + n - 1) % n;
        for i in 0..n {
            let ip = (i + 1) % n;
            let im = (i + n - 1) % n;
            let dudx = (u[j * n + ip][0] - u[j * n + im][0]) * 0.5 * inv_dx;
            let dvdy = (u[jp * n + i][1] - u[jm * n + i][1]) * 0.5 * inv_dx;
            div[j * n + i] = dudx + dvdy;
        }
    }
    // Subtract mean (nullspace gauge).
    let mean = div.iter().sum::<f32>() / (n * n) as f32;
    for v in div.iter_mut() {
        *v -= mean;
    }
    // Solve `Δ_wide p = div` with Gauss-Seidel. The wide-stencil
    // Laplacian on a periodic torus splits into two independent 5-pt
    // problems (even / odd sub-grids); a plain GS sweep that touches
    // both sub-grids in turn converges on each one. Stencil:
    //   (p[i+2] − 2p[i] + p[i−2]) / (4 dx²) per axis.
    let mut p = vec![0.0_f32; n * n];
    let dx2 = 1.0 / (inv_dx * inv_dx);
    let scale = 4.0 * dx2; // matches the (1 / 4dx²) factor in Δ_wide
    for _ in 0..iters {
        for j in 0..n {
            let jp2 = (j + 2) % n;
            let jm2 = (j + n - 2) % n;
            for i in 0..n {
                let ip2 = (i + 2) % n;
                let im2 = (i + n - 2) % n;
                let s = p[j * n + ip2] + p[j * n + im2] + p[jp2 * n + i] + p[jm2 * n + i];
                p[j * n + i] = 0.25 * (s - scale * div[j * n + i]);
            }
        }
        // Re-zero the mean to keep the constant nullspace gauged out.
        let m = p.iter().sum::<f32>() / (n * n) as f32;
        for v in p.iter_mut() {
            *v -= m;
        }
    }
    // Corrector: u = u − ∇p with the same central-difference gradient.
    let p_clone = p.clone();
    for j in 0..n {
        let jp = (j + 1) % n;
        let jm = (j + n - 1) % n;
        for i in 0..n {
            let ip = (i + 1) % n;
            let im = (i + n - 1) % n;
            let dpdx = (p_clone[j * n + ip] - p_clone[j * n + im]) * 0.5 * inv_dx;
            let dpdy = (p_clone[jp * n + i] - p_clone[jm * n + i]) * 0.5 * inv_dx;
            u[j * n + i][0] -= dpdx;
            u[j * n + i][1] -= dpdy;
        }
    }
}

/// GPU pressure projection: build `∇·u`, hand it to
/// [`solve_pressure_poisson_fft_with_stencil`] with the wide-stencil
/// symbol (matches the central-difference gradient/divergence pair we
/// use here — see the [`crate::fem::fft_ns`] docstring for why this is
/// required), then subtract `∇p` from `u` in place.
#[cfg(feature = "gpu")]
fn project_fft_periodic(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    kernel: &FftKernel,
    u: &mut [[f32; 2]],
    n: usize,
) -> Result<()> {
    let dx = 1.0 / n as f32;
    let inv_dx = 1.0 / dx;
    let mut div = vec![0.0_f32; n * n];
    for j in 0..n {
        let jp = (j + 1) % n;
        let jm = (j + n - 1) % n;
        for i in 0..n {
            let ip = (i + 1) % n;
            let im = (i + n - 1) % n;
            let dudx = (u[j * n + ip][0] - u[j * n + im][0]) * 0.5 * inv_dx;
            let dvdy = (u[jp * n + i][1] - u[jm * n + i][1]) * 0.5 * inv_dx;
            // FFT-Poisson inverts `-Δ p = rhs`; we want `Δ p = ∇·u`, so
            // feed `rhs = -∇·u`.
            div[j * n + i] = -(dudx + dvdy);
        }
    }
    // Scale rhs by `dx²` because the spectral solver assumes unit grid
    // spacing in its discrete Laplacian — mirror the `fft_ns` corrector.
    let scale = dx * dx;
    for v in div.iter_mut() {
        *v *= scale;
    }
    let p = solve_pressure_poisson_fft_with_stencil(
        device,
        queue,
        kernel,
        &div,
        n,
        PoissonStencil::WideStencil,
    )?;
    for j in 0..n {
        let jp = (j + 1) % n;
        let jm = (j + n - 1) % n;
        for i in 0..n {
            let ip = (i + 1) % n;
            let im = (i + n - 1) % n;
            let dpdx = (p[j * n + ip] - p[j * n + im]) * 0.5 * inv_dx;
            let dpdy = (p[jp * n + i] - p[jm * n + i]) * 0.5 * inv_dx;
            u[j * n + i][0] -= dpdx;
            u[j * n + i][1] -= dpdy;
        }
    }
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Seed a Taylor-Green vortex into `state`. Domain is `[0, 1]²`,
    /// velocity `(sin(2π x) cos(2π y), -cos(2π x) sin(2π y))`. The
    /// Stokes (linearised) decay rate is `exp(-2 ν (2π)² t)`; the
    /// nonlinear NS step preserves Taylor-Green up to the same rate
    /// because the convection term is exactly cancelled by the
    /// pressure gradient for this initial condition (this is the
    /// classical 2D Taylor-Green identity).
    fn seed_taylor_green(state: &mut StamFluidState) {
        let n = state.grid_n;
        let two_pi = std::f32::consts::TAU;
        for j in 0..n {
            let y = (j as f32 + 0.5) / n as f32;
            for i in 0..n {
                let x = (i as f32 + 0.5) / n as f32;
                let ux = (two_pi * x).sin() * (two_pi * y).cos();
                let vy = -(two_pi * x).cos() * (two_pi * y).sin();
                state.u[j * n + i] = [ux, vy];
            }
        }
    }

    #[test]
    fn diffusion_smooths_pulse() {
        // Plant a single hot cell, run a few diffusion sweeps, expect
        // its neighbours to inherit some mass and the centre to drop.
        let n = 16;
        let mut grid = vec![0.0_f32; n * n];
        grid[(n / 2) * n + n / 2] = 1.0;
        let before = grid[(n / 2) * n + n / 2];
        diffuse_scalar_periodic(&mut grid, n, 0.5, 20);
        let after = grid[(n / 2) * n + n / 2];
        assert!(
            after < before,
            "centre value should decrease: {after} >= {before}"
        );
        let neighbour = grid[(n / 2) * n + (n / 2 + 1)];
        assert!(
            neighbour > 0.0,
            "neighbour should pick up some mass: {neighbour}"
        );
        // Total mass conserved (Gauss-Seidel on this operator preserves
        // the constant mode → sum stays equal to the initial sum).
        let sum: f32 = grid.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-4, "sum drift {sum}");
    }

    #[test]
    fn project_jacobi_kills_divergence_constant_flow() {
        // A constant-divergence field should project to ~zero
        // divergence after enough iterations.
        let n = 16;
        let mut u = vec![[0.0_f32; 2]; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = i as f32 / n as f32;
                let y = j as f32 / n as f32;
                // Pure-gradient field — fully divergent, should be
                // projected out almost entirely.
                u[j * n + i] = [
                    (std::f32::consts::TAU * x).sin(),
                    (std::f32::consts::TAU * y).sin(),
                ];
            }
        }
        project_jacobi_periodic(&mut u, n, 200);
        // Divergence after projection.
        let inv_dx = n as f32;
        let mut max_div = 0.0_f32;
        for j in 0..n {
            let jm = (j + n - 1) % n;
            for i in 0..n {
                let im = (i + n - 1) % n;
                let d = (u[j * n + i][0] - u[j * n + im][0]) * inv_dx
                    + (u[j * n + i][1] - u[jm * n + i][1]) * inv_dx;
                max_div = max_div.max(d.abs());
            }
        }
        assert!(max_div < 1.0e-3, "max div after project: {max_div}");
    }

    #[test]
    fn taylor_green_seed_is_divergence_free() {
        let n = 32;
        let mut s = StamFluidState::new(n);
        seed_taylor_green(&mut s);
        let div = s.divergence();
        let max = div.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
        assert!(max < 1.0e-3, "max div of analytic seed: {max}");
    }
}
