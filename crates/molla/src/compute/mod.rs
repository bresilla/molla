//! GPU runtime: `wgpu` instance, adapter, device, queue.
//!
//! [`Engine`] owns the device + queue used by every GPU kernel in the
//! crate; pick a [`Backend`] to select the adapter (auto / native GPU /
//! browser WebGPU / CPU fallback). Kernel dispatch wrappers live in
//! [`crate::kernels`] — this module only stands up the runtime.
//!
//! [`MpmSimulation`] is the persistent-buffer MPM driver (Stage 24d.x);
//! see [`MpmConfig`] / [`MpmRheology`] / [`MpmTimeStep`] for configuration.

#[cfg(feature = "gpu")]
mod engine;
#[cfg(feature = "gpu")]
mod mpm;

// Stage 38.1 — CPU-only differentiable MPM. Pure-Rust, `f64` everywhere,
// records a tape so [`crate::autograd::mpm_adjoint`] can walk it
// backwards. Always compiled (no GPU dependency).
pub mod mpm_cpu_diff;

// Stage 39C — 2D Stam stable-fluid solver. CPU path is always compiled
// (uses an iterative Gauss-Seidel pressure solve); GPU `step_gpu` reuses
// `fem::fft_ns::solve_pressure_poisson_fft` for the projection.
pub mod stam_fluid;

// Stage 39A — CPU-only implicit MPM with Bingham rheology. Pure-Rust,
// `f64` everywhere. Builds on the new graph-coloring primitive for a
// colored Gauss-Seidel grid-velocity solve that's stable far above the
// explicit-CFL limit.
pub mod implicit_mpm;

// Stage 37.8 — tile compute primitives bundle. CPU oracles are
// always available; the GPU pipelines are gated behind the `gpu`
// feature inside each module.
pub mod tile_block_cholesky;
pub mod tile_convolution;
pub mod tile_filtering;
pub mod tile_matmul;
pub mod tile_mcgp;
pub mod tile_stream_compaction;

#[cfg(feature = "gpu")]
pub use engine::{AdapterInfo, Backend, Engine};
#[cfg(feature = "gpu")]
pub use mpm::{MpmConfig, MpmRheology, MpmSimulation, MpmTimeStep};
