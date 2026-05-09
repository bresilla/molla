//! Tape-style automatic differentiation (`molla::autograd`).
//!
//! Stage 23 (P2) — manual reverse-mode autograd built on a CPU tape
//! that records elementary operations forward and replays adjoint ops
//! in reverse to produce gradients. Mirrors the pattern from Warp's
//! `wp.Tape` but in Rust instead of Python: every "tensor" produced
//! during the forward pass gets a [`TapeId`] that the tape uses as
//! the key for an adjoint accumulator.
//!
//! # Why CPU only
//!
//! Differentiability is the **CPU-acceptable exception** to molla's
//! GPU-first rule. Tape recording fundamentally needs heap allocation
//! and a dynamic graph structure that doesn't map onto wgpu's compute
//! pipeline-state model — Newton + Warp solve the same problem with
//! Python + CUDA-graphs, neither of which we have. We use Rust + CPU
//! and accept the perf hit (orders of magnitude vs hand-written
//! adjoint kernels) in exchange for a working, testable autograd v1.
//! GPU-side adjoints are the obvious follow-up — plumb a per-op
//! kernel queue + adjoint kernels — but the structure of [`Tape`]
//! and [`TapeOp`] is engine-agnostic so the upgrade path stays open.
//!
//! # Scope of v1
//!
//! - Five elementary `TapeOp` variants — Add / Sub / Mul / MatVec /
//!   Norm — sufficient to express XPBD's distance-constraint
//!   projection on host. Real-engine code can opt into the tape via
//!   [`Tape::record`] without inheriting any GPU dispatch concerns.
//! - Manual XPBD adjoint in [`xpbd_adjoint`] that doesn't use the
//!   tape at all — it's a closed-form reverse pass over the same
//!   loop the CPU forward path runs. This is faster (no per-op heap
//!   allocations) and matches Newton's `wp.func_grad` pattern of
//!   shipping hand-written gradients for hot solver paths.
//! - **Featherstone adjoint is deferred** to a follow-up stage. The
//!   reverse-mode ABA from Featherstone (2008) §10 is multi-week
//!   work; the stub is in place but not wired through the solver.
//!
//! # Determinism
//!
//! [`Tape::backward`] is deterministic for a fixed input + recorded
//! op sequence. The verification harness exercises this directly
//! (`tests/autograd_xpbd.rs::tape_replay_is_deterministic`).

pub mod tape;
pub mod xpbd_adjoint;

pub use tape::{Tape, TapeId, TapeOp};
pub use xpbd_adjoint::{
    DistanceConstraintAdjoint, distance_chain_grad_initial_pos,
    distance_chain_grad_stiffness, xpbd_distance_step_with_grad,
};
