//! First-order optimizers (`molla::optim`).
//!
//! Stage 38c — Adam + SGD. Differentiable physics, parameter ID, and
//! policy-learning code paths in molla all need gradient descent on a
//! flat parameter vector. This module ships the two canonical
//! first-order optimizers behind the same shape as Warp's
//! `wp.optim.{Adam, SGD}` (`xtra/warp/warp/_src/optim/{adam,sgd}.py`)
//! but expressed as plain Rust state structs that consume f64 slices.
//!
//! # CPU + GPU
//!
//! Host paths run on `f64` (the engine's [`crate::math::Real`]); GPU
//! kernels run on `f32` because WGSL has no `f64`. The Rust dispatch
//! wrappers live in [`crate::kernels::optim_adam`] and
//! [`crate::kernels::optim_sgd`]; both call into
//! `shaders/optim_{adam,sgd}.wgsl`.
//!
//! # Reference
//!
//! - Adam: Kingma & Ba (2014), <https://arxiv.org/abs/1412.6980>.
//! - SGD with momentum + Nesterov + decoupled weight decay: matches
//!   PyTorch's `torch.optim.SGD` semantics and the Warp implementation.
//!
//! # Math
//!
//! ## Adam (per parameter, step `t ≥ 1`):
//! ```text
//! m_t = β₁·m_{t-1} + (1−β₁)·g_t
//! v_t = β₂·v_{t-1} + (1−β₂)·g_t²
//! m̂_t = m_t / (1 − β₁^t)
//! v̂_t = v_t / (1 − β₂^t)
//! θ_t = θ_{t-1} − lr · m̂_t / (√v̂_t + ε)
//! ```
//! Optional decoupled weight decay (AdamW-style):
//! ```text
//! θ_t -= lr · wd · θ_{t-1}
//! ```
//!
//! ## SGD (per parameter):
//! ```text
//! g̃   = g + wd · θ
//! b_t = μ·b_{t-1} + (1−damp)·g̃   if t > 0
//!     = g̃                       if t == 0
//! g_eff = g̃ + μ·b_t             if Nesterov
//!       = b_t                    otherwise
//! θ_t = θ_{t-1} − lr · g_eff
//! ```

pub mod adam;
pub mod sgd;

pub use adam::{AdamConfig, AdamState};
pub use sgd::{SgdConfig, SgdState};
