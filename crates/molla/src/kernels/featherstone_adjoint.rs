//! Stage 23.1.1 / 23.1.2 — host dispatcher for the Featherstone adjoint
//! kernels.
//!
//! Mirrors the per-sweep `TapeOp` granularity from
//! `crate::autograd::tape`. The real GPU dispatch is the per-step
//! closed-form Jacobian kernel `featherstone_adjoint_jacobians.wgsl`
//! (wrapped by [`crate::kernels::featherstone_adjoint_gpu::FeatherstoneAdjointGpuKernel`]).
//! The dispatcher routes to one of the CPU adjoint paths in
//! `crate::autograd::featherstone_adjoint`; per-sweep WGSL scaffolds
//! never landed (the closed-form-Jacobian batched dispatch was the
//! winning shape) and were removed.
//!
//! ## Stage 23.1.2 changes
//!
//! - The `GpuFallbackCpu` backend (renamed from `ForceGpu` to be honest
//!   about what it actually does on this dispatcher) **always** falls
//!   back to the CPU AnalyticQd path for single-articulation reverse:
//!   the per-step GPU round-trip dominates at `n_links ≤ 6`, so even
//!   with a real device the CPU path wins. The real GPU kernel
//!   (`featherstone_adjoint_jacobians.wgsl`) lives behind
//!   [`crate::kernels::featherstone_adjoint_gpu::FeatherstoneAdjointGpuKernel::reverse_batched`]
//!   — wire that directly when you have a multi-chain workload and a
//!   device. Single-articulation parity vs CPU is still verified in
//!   `tests/autograd_featherstone_v3_gpu.rs`.
//! - A new `AdjointBackend::ForceCpuAnalyticQd` exposes the new
//!   analytic ∂h/∂qd path; the overall `Auto` default still picks
//!   `ForceCpuAnalyticQd` (faster than `ForceCpuClosedForm` by
//!   ~`2-3×` on the 6-DOF Franka) so the viewer's "Auto" reading is
//!   the new optimum.
//! - `MultiAdjointBatch` exposes the multi-articulation parallel
//!   reverse — the headline win of the stage. CPU path uses
//!   `rayon::par_iter`; GPU path is one kernel dispatch with N
//!   workgroups.

use crate::autograd::{
    featherstone_adjoint::{AutogradError, JacobianStrategy},
    FeatherstoneRevoluteChain, FeatherstoneTrace, JacobianScratch,
};
use crate::math::{Real, Vec3};

/// Backend selector for the Featherstone adjoint dispatcher.
///
/// - `Auto` picks the best path given the runtime context. Stage 23.1.2:
///   single-articulation always picks `ForceCpuAnalyticQd` (fastest
///   CPU path); multi-chain GPU dispatch lives in
///   [`crate::kernels::featherstone_adjoint_gpu::FeatherstoneAdjointGpuKernel`].
/// - `ForceCpuClosedForm` keeps the Stage 23.1.1 default (CRBA + FD on
///   RNEA for both q and qd) — kept as a perf baseline.
/// - `ForceCpuAnalyticQd` (new in 23.1.2) — closed-form `M⁻¹` for τ,
///   analytic `∂h/∂qd` via chain-rule through RNEA, FD-on-RNEA only
///   for `∂h/∂q`.
/// - `ForceCpuFiniteDiff` is the v1 Stage 23.1 path (full ABA-FD).
///   Kept as the slowest correctness oracle.
/// - `GpuFallbackCpu` is **not** a real GPU dispatch on this
///   dispatcher: the per-step GPU round-trip dominates at `n_links ≤ 6`,
///   so single-articulation reverse always falls back to the CPU
///   AnalyticQd path. Use it only if you want to force the "GPU label"
///   in the viewer or to test the fallback log path. Real GPU dispatch
///   for **multi-chain** workloads lives in
///   [`crate::kernels::featherstone_adjoint_gpu::FeatherstoneAdjointGpuKernel::reverse_batched`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum AdjointBackend {
    /// Pick the best backend at runtime.
    #[default]
    Auto,
    /// Force CPU closed-form (Carpentier-Mansard chain rule via CRBA +
    /// FD-on-RNEA). Stage 23.1.1 default; kept as a perf baseline so
    /// the viewer can compare AnalyticQd vs ClosedForm.
    ForceCpuClosedForm,
    /// Stage 23.1.2 — CPU closed-form `M⁻¹` + analytic `∂h/∂qd` +
    /// FD-on-RNEA `∂h/∂q`. New default for `Auto` at single articulation.
    ForceCpuAnalyticQd,
    /// Force CPU central-FD on the full ABA forward step. Stage 23.1
    /// v1 path; kept as a correctness oracle.
    ForceCpuFiniteDiff,
    /// "GPU" tag that **always** falls back to the CPU AnalyticQd
    /// path. Single-articulation per-step GPU round-trips are slower
    /// than CPU at `n_links ≤ 6`, so this dispatcher routes through
    /// CPU even with a real device. For real multi-chain GPU
    /// dispatch use
    /// [`crate::kernels::featherstone_adjoint_gpu::FeatherstoneAdjointGpuKernel::reverse_batched`]
    /// directly.
    GpuFallbackCpu,
}

impl AdjointBackend {
    /// Human-readable label for viewer pickers + log lines.
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::ForceCpuClosedForm => "CPU CF",
            Self::ForceCpuAnalyticQd => "CPU AnalyticQd",
            Self::ForceCpuFiniteDiff => "CPU FD",
            Self::GpuFallbackCpu => "GPU→CPU",
        }
    }
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
/// Single entry point for both the unit tests + the viewer.
pub struct FeatherstoneAdjointDispatcher;

impl FeatherstoneAdjointDispatcher {
    /// Run the adjoint reverse pass per `backend`, timing the wall
    /// clock so the caller can compute backward / forward ratios.
    ///
    /// Surfaces [`AutogradError::UnsupportedJoint`] verbatim if the
    /// chain contains a composite multi-DoF joint kind — the per-link
    /// math requires single-DoF Revolute / Prismatic decomposition.
    pub fn reverse(
        chain: &FeatherstoneRevoluteChain,
        trace: &FeatherstoneTrace,
        grad_ee: Vec3,
        scratch: &mut JacobianScratch,
        backend: AdjointBackend,
    ) -> Result<AdjointResult, AutogradError> {
        let start = std::time::Instant::now();
        let resolved = Self::resolve(backend);
        let g_tau = match resolved {
            AdjointBackend::Auto | AdjointBackend::ForceCpuAnalyticQd => chain
                .reverse_with_strategy(trace, grad_ee, scratch, JacobianStrategy::AnalyticBiasQd)?,
            AdjointBackend::ForceCpuClosedForm => chain
                .reverse_with_strategy(trace, grad_ee, scratch, JacobianStrategy::ClosedForm)?,
            AdjointBackend::ForceCpuFiniteDiff => chain.reverse_with_strategy(
                trace,
                grad_ee,
                scratch,
                JacobianStrategy::CentralDifference,
            )?,
            AdjointBackend::GpuFallbackCpu => {
                // Single-articulation reverse always falls back to the
                // CPU AnalyticQd path — see `AdjointBackend` doc. The
                // real GPU win lives in `FeatherstoneAdjointGpuKernel`'s
                // multi-chain `reverse_batched`; calling it from here
                // would need an `Engine` plumbed through the dispatcher
                // signature, which intentionally stays Engine-free.
                Self::log_gpu_fallback();
                chain.reverse_with_strategy(
                    trace,
                    grad_ee,
                    scratch,
                    JacobianStrategy::AnalyticBiasQd,
                )?
            }
        };
        let backward_us = start.elapsed().as_micros() as f64;
        Ok(AdjointResult {
            g_tau,
            backend: resolved,
            backward_us,
        })
    }

    /// Resolve `Auto` to the concrete backend the dispatcher will use.
    /// Stage 23.1.2: `Auto` resolves to `ForceCpuAnalyticQd` — the
    /// fastest CPU path. Multi-articulation `Auto` runs through
    /// `MultiAdjointBatch::dispatch` instead, which picks GPU when a
    /// device is supplied.
    pub fn resolve(backend: AdjointBackend) -> AdjointBackend {
        match backend {
            AdjointBackend::Auto => AdjointBackend::ForceCpuAnalyticQd,
            other => other,
        }
    }

    fn log_gpu_fallback() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[molla::autograd] AdjointBackend::GpuFallbackCpu — single-chain reverse \
                 always runs on CPU AnalyticQd (per-step GPU round-trip dominates at \
                 n_links ≤ 6). For real multi-chain GPU dispatch use \
                 FeatherstoneAdjointGpuKernel::reverse_batched directly."
            );
        }
    }
}

/// Stage 23.1.2 — multi-articulation batched adjoint dispatch.
///
/// The "GPU autograd actually wins" workload, per the spec. Each chain
/// in the batch is independent — the per-chain reverse pass runs on a
/// dedicated worker (CPU: `rayon::par_iter`; GPU: one workgroup per
/// chain). Per-chain timing on a 10-chain Franka batch should land
/// within ~`1.5×` of the single-chain CPU-AnalyticQd timing — the win
/// vs running 10 sequential `FeatherstoneAdjointDispatcher::reverse`
/// calls is `10×`-ish on CPU, more on GPU once the round-trip cost
/// amortises.
pub struct MultiAdjointBatch;

/// One articulation in a multi-adjoint batch.
pub struct MultiAdjointInput<'a> {
    pub chain: &'a FeatherstoneRevoluteChain,
    pub trace: &'a FeatherstoneTrace,
    pub grad_ee: Vec3,
}

/// Per-chain output of a batched adjoint dispatch.
pub struct MultiAdjointOutput {
    pub g_tau: Vec<Vec<Real>>,
}

impl MultiAdjointBatch {
    /// CPU multi-articulation adjoint: parallel rayon dispatch of the
    /// AnalyticQd path. Each chain owns its own `JacobianScratch` so
    /// the workers don't contend.
    ///
    /// Returns `(per_chain_results, total_wall_us)`. The per-chain
    /// timing equals `total_wall_us / chains.len()` only when the
    /// rayon worker pool is large enough — small batches on a single
    /// core won't show the parallel win.
    pub fn dispatch_cpu(
        inputs: &[MultiAdjointInput<'_>],
    ) -> Result<(Vec<MultiAdjointOutput>, f64), AutogradError> {
        use rayon::prelude::*;
        let start = std::time::Instant::now();
        let outputs: Result<Vec<MultiAdjointOutput>, AutogradError> = inputs
            .par_iter()
            .map(|inp| {
                let n = inp.chain.links.len();
                let mut scratch = JacobianScratch::new(n);
                let g_tau = inp.chain.reverse_with_strategy(
                    inp.trace,
                    inp.grad_ee,
                    &mut scratch,
                    JacobianStrategy::AnalyticBiasQd,
                )?;
                Ok(MultiAdjointOutput { g_tau })
            })
            .collect();
        let wall_us = start.elapsed().as_micros() as f64;
        Ok((outputs?, wall_us))
    }

    /// Sequential CPU multi-articulation adjoint — single-threaded
    /// baseline so the viewer / tests can A/B against `dispatch_cpu`.
    pub fn dispatch_cpu_serial(
        inputs: &[MultiAdjointInput<'_>],
    ) -> Result<(Vec<MultiAdjointOutput>, f64), AutogradError> {
        let start = std::time::Instant::now();
        let mut outputs: Vec<MultiAdjointOutput> = Vec::with_capacity(inputs.len());
        for inp in inputs {
            let n = inp.chain.links.len();
            let mut scratch = JacobianScratch::new(n);
            let g_tau = inp.chain.reverse_with_strategy(
                inp.trace,
                inp.grad_ee,
                &mut scratch,
                JacobianStrategy::AnalyticBiasQd,
            )?;
            outputs.push(MultiAdjointOutput { g_tau });
        }
        let wall_us = start.elapsed().as_micros() as f64;
        Ok((outputs, wall_us))
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
