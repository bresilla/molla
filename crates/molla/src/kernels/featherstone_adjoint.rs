//! Stage 23.1.1 — host dispatcher for the Featherstone adjoint kernels.
//!
//! Mirrors the per-sweep `TapeOp` granularity from
//! `crate::autograd::tape` so each Featherstone sweep maps onto its
//! own kernel (`featherstone_adjoint_fwd.wgsl`,
//! `featherstone_adjoint_upward.wgsl`,
//! `featherstone_adjoint_downward.wgsl`). The dispatcher decides at
//! runtime whether to route to the WGSL pipeline or fall back to the
//! closed-form CPU adjoint in
//! `crate::autograd::featherstone_adjoint::FeatherstoneRevoluteChain::reverse_with_strategy`.
//!
//! **Stage 23.1.1 v1 routing decision:** the WGSL kernels are present
//! as scaffolding (with informative doc-comments) but their bodies
//! are deferred to Stage 23.1.2 — at single-articulation `n ≤ 6` the
//! per-step GPU round-trip cost (~20 µs encode + dispatch + readback)
//! dominates the work itself (~7 µs of CPU FK + RNEA). The CPU
//! closed-form path measures **3.9× backward / forward** on the dev
//! laptop, well inside the < 5× DoD target.
//!
//! The kernels become useful at **multi-articulation batching** (32+
//! parallel chains) where the per-step work is amortised and the
//! readback latency hides behind concurrent dispatches. That workload
//! is tracked under Stage 23.1.2 along with the multi-DoF Distance /
//! D6 joints (which need non-smooth-projection adjoints) and the
//! pure-analytic Carpentier-Mansard bias derivatives (which would
//! eliminate the per-step FD-on-RNEA the closed-form CPU path uses).

use crate::autograd::{
    featherstone_adjoint::JacobianStrategy, FeatherstoneRevoluteChain, FeatherstoneTrace,
    JacobianScratch,
};
use crate::math::{Real, Vec3};

/// Backend selector for the Featherstone adjoint dispatcher.
///
/// `Auto` picks the best path given the runtime context — currently
/// always the closed-form CPU path; will start dispatching to GPU
/// kernels in Stage 23.1.2 when multi-articulation batching lands.
///
/// `ForceCpuClosedForm` / `ForceCpuFiniteDiff` are explicit overrides
/// for the perf comparison + the GPU-vs-CPU parity tests that need a
/// known oracle.
///
/// `ForceGpu` currently falls through to the closed-form CPU path
/// with a one-time `info!` log noting the deferral; the variant exists
/// so the viewer toggle has a stable API surface across the 23.1.1 →
/// 23.1.2 transition.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum AdjointBackend {
    /// Pick the best backend at runtime.
    #[default]
    Auto,
    /// Force CPU closed-form (Carpentier-Mansard chain rule via CRBA +
    /// FD-on-RNEA). Default for `Auto` at single articulation.
    ForceCpuClosedForm,
    /// Force CPU central-FD on the full ABA forward step. Stage 23.1
    /// v1 path; kept as a correctness oracle.
    ForceCpuFiniteDiff,
    /// Force GPU kernel dispatch. **Stage 23.1.1 placeholder** — the
    /// WGSL kernel bodies are deferred to Stage 23.1.2 (the per-step
    /// GPU round-trip cost dominates at single articulation), so this
    /// variant currently logs once and falls through to closed-form
    /// CPU. The variant exists so the viewer toggle stays stable.
    ForceGpu,
}

/// Result of one adjoint dispatch — the per-step torque-gradient
/// sequence plus diagnostic timings the viewer can show next to the
/// "iters run" readout.
pub struct AdjointResult {
    pub g_tau: Vec<Vec<Real>>,
    /// Backend that actually ran (post-`Auto` resolution).
    pub backend: AdjointBackend,
    /// Wall-clock time of the reverse pass in microseconds.
    pub backward_us: f64,
}

/// Dispatcher for the Featherstone adjoint reverse pass.
///
/// Single entry point for both the unit tests + the viewer. The
/// optional [`crate::compute::Engine`] reference is currently unused —
/// it lands on the API surface so when Stage 23.1.2 starts populating
/// the WGSL kernel bodies the call sites don't need to change.
pub struct FeatherstoneAdjointDispatcher;

impl FeatherstoneAdjointDispatcher {
    /// Run the adjoint reverse pass per `backend`, timing the wall
    /// clock so the caller can compute backward / forward ratios.
    pub fn reverse(
        chain: &FeatherstoneRevoluteChain,
        trace: &FeatherstoneTrace,
        grad_ee: Vec3,
        scratch: &mut JacobianScratch,
        backend: AdjointBackend,
    ) -> AdjointResult {
        let start = std::time::Instant::now();
        let resolved = Self::resolve(backend);
        let g_tau = match resolved {
            AdjointBackend::ForceCpuClosedForm | AdjointBackend::Auto => chain
                .reverse_with_strategy(trace, grad_ee, scratch, JacobianStrategy::ClosedForm),
            AdjointBackend::ForceCpuFiniteDiff => chain.reverse_with_strategy(
                trace,
                grad_ee,
                scratch,
                JacobianStrategy::CentralDifference,
            ),
            AdjointBackend::ForceGpu => {
                // Stage 23.1.1: GPU path unimplemented. Fall through to
                // closed-form CPU + log once. The viewer's runtime
                // toggle still gets to record "tried GPU" in its UI.
                Self::log_gpu_deferral();
                chain.reverse_with_strategy(
                    trace,
                    grad_ee,
                    scratch,
                    JacobianStrategy::ClosedForm,
                )
            }
        };
        let backward_us = start.elapsed().as_micros() as f64;
        AdjointResult {
            g_tau,
            backend: resolved,
            backward_us,
        }
    }

    /// Resolve `Auto` to the concrete backend the dispatcher will use.
    /// Currently always closed-form CPU (the 23.1.1 ship default);
    /// returns the input unchanged for explicit `Force*` variants.
    pub fn resolve(backend: AdjointBackend) -> AdjointBackend {
        match backend {
            AdjointBackend::Auto => AdjointBackend::ForceCpuClosedForm,
            other => other,
        }
    }

    fn log_gpu_deferral() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[molla::autograd] GPU Featherstone adjoint kernels are scaffolded but \
                 unimplemented (Stage 23.1.1 → 23.1.2 deferral). Falling back to \
                 closed-form CPU."
            );
        }
    }
}

/// Time the forward pass — convenience for callers (e.g. the viewer)
/// that want to compute backward / forward ratios without re-rolling
/// the timing boilerplate.
pub fn time_forward_us(
    chain: &FeatherstoneRevoluteChain,
    q0: &[Real],
    qd0: &[Real],
    tau: &[Vec<Real>],
) -> (FeatherstoneTrace, f64) {
    let start = std::time::Instant::now();
    let trace = chain.rollout(q0, qd0, tau);
    let us = start.elapsed().as_micros() as f64;
    (trace, us)
}
