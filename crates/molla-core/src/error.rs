//! Crate-wide error type.

use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// All failure modes that can be observed at the public API.
#[derive(Debug, Error)]
pub enum Error {
    /// A buffer was accessed in a way incompatible with its current state
    /// (e.g. host-only buffer asked for a device handle).
    #[error("storage error: {0}")]
    Storage(String),

    /// Builder validation rejected the model (missing inertia, dangling
    /// joint reference, …).
    #[error("model build error: {0}")]
    Build(String),

    /// Solver detected a non-recoverable condition (NaN, divergence,
    /// invalid model topology).
    #[error("solver error: {0}")]
    Solver(String),

    /// `wgpu` failed to initialise or to dispatch a kernel.
    #[cfg(feature = "gpu")]
    #[error("gpu backend error: {0}")]
    Gpu(String),

    /// FEM module failure — mesh parse error, malformed input,
    /// unsupported element kind in strict mode, etc.
    #[error("fem error: {0}")]
    Fem(String),

    /// Catch-all for invariant violations that should be impossible at
    /// runtime; reaching one indicates a crate bug.
    #[error("internal invariant violated: {0}")]
    Internal(String),
}
