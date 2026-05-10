//! Stage 37.8 — image filtering helpers.
//!
//! Wraps [`crate::compute::tile_convolution`] with named filter
//! coefficients (box, Gaussian, Laplacian, etc.). The math is identical
//! — convolution is convolution — but exposing the named filters as
//! their own API lets call-sites stay readable.
//!
//! Public functions:
//! - [`box_filter_cpu`] — equal-weight 5-tap box.
//! - [`gaussian_filter_cpu`] — `(1, 4, 6, 4, 1) / 16` 5-tap Gaussian.
//! - [`laplacian_filter_cpu`] — discrete second-derivative stencil
//!   `(0, 1, -2, 1, 0)` applied separably (so 2D output is the sum of
//!   the row and column 1-D Laplacians).
//! - [`identity_filter_cpu`] — returns the input unchanged. Used by
//!   tests as a smoke test that the kernel infrastructure round-trips.
//!
//! GPU helpers (gated on the `gpu` feature) take the same arguments
//! plus a [`wgpu::Device`] / [`wgpu::Queue`] / [`TileConvolutionKernel`]
//! and run the WGSL kernel under the hood.

use crate::compute::tile_convolution::{convolve_separable_cpu, Kernel5};
use crate::core::Result;

/// Box-filter the input in place. Equal-weight 5-tap; output is the
/// local mean. CPU path.
pub fn box_filter_cpu(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
) -> Result<()> {
    convolve_separable_cpu(input, output, width, height, &Kernel5::box5())
}

/// Gaussian-filter the input. CPU path. Standard binomial 5-tap
/// `(1, 4, 6, 4, 1) / 16`.
pub fn gaussian_filter_cpu(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
) -> Result<()> {
    convolve_separable_cpu(input, output, width, height, &Kernel5::gaussian5())
}

/// Apply the separable discrete Laplacian 1-D stencil
/// `(0, 1, -2, 1, 0)` along x and y. Useful for edge detection and
/// stage 37.x wave-eqn experiments. CPU path.
pub fn laplacian_filter_cpu(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
) -> Result<()> {
    convolve_separable_cpu(input, output, width, height, &Kernel5::laplacian1d())
}

/// Identity (δ-function centre tap). Smoke-test wrapper. CPU path.
pub fn identity_filter_cpu(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
) -> Result<()> {
    convolve_separable_cpu(input, output, width, height, &Kernel5::identity())
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use crate::compute::tile_convolution::TileConvolutionKernel;

    /// GPU box filter — runs `TileConvolutionKernel` with [`Kernel5::box5`].
    pub fn box_filter_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TileConvolutionKernel,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) -> Result<()> {
        kernel.dispatch_blocking(device, queue, input, output, width, height, &Kernel5::box5())
    }

    /// GPU Gaussian filter.
    pub fn gaussian_filter_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TileConvolutionKernel,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) -> Result<()> {
        kernel.dispatch_blocking(
            device,
            queue,
            input,
            output,
            width,
            height,
            &Kernel5::gaussian5(),
        )
    }

    /// GPU Laplacian filter.
    pub fn laplacian_filter_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TileConvolutionKernel,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) -> Result<()> {
        kernel.dispatch_blocking(
            device,
            queue,
            input,
            output,
            width,
            height,
            &Kernel5::laplacian1d(),
        )
    }

    /// GPU identity filter — useful as a smoke-test that the
    /// convolution pipeline round-trips.
    pub fn identity_filter_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TileConvolutionKernel,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) -> Result<()> {
        kernel.dispatch_blocking(
            device,
            queue,
            input,
            output,
            width,
            height,
            &Kernel5::identity(),
        )
    }
}

#[cfg(feature = "gpu")]
pub use gpu::{box_filter_gpu, gaussian_filter_gpu, identity_filter_gpu, laplacian_filter_gpu};
