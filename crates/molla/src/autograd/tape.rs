//! Tape-style autograd primitive — record forward ops, replay
//! adjoints in reverse.
//!
//! Loosely models `wp.Tape` (see `xtra/warp/warp/_src/tape.py`):
//! every "tensor" produced during the forward pass earns a [`TapeId`],
//! the tape stores the recorded op + the values needed to evaluate
//! its adjoint, and [`Tape::backward`] walks the recorded list in
//! reverse, propagating the seed gradient back into a per-`TapeId`
//! accumulator.
//!
//! v1 is host-only and tracks `Vec<f64>` tensors. The five op variants
//! cover what XPBD constraint projection needs: [`TapeOp::Add`],
//! [`TapeOp::Sub`], [`TapeOp::Mul`] (elementwise scalar), [`TapeOp::MatVec`]
//! (3×3 × 3-vec, used for inertia-like Jacobian transports), and
//! [`TapeOp::Norm`] (Euclidean norm, the core of distance-constraint
//! gradient).

use std::collections::HashMap;

use crate::math::robust_loss::{
    norm_cauchy, norm_cauchy_grad, norm_huber, norm_huber_grad, norm_l1, norm_l1_grad,
    norm_l2_smooth, norm_l2_smooth_grad, norm_pseudo_huber, norm_pseudo_huber_grad, norm_tukey,
    norm_tukey_grad,
};
use crate::math::Real;

/// Stable identifier for a tensor recorded on the tape. Issued by
/// [`Tape::alloc`]; the tape uses it as the key for the per-tensor
/// adjoint accumulator computed during [`Tape::backward`].
///
/// IDs are sequential within a single tape. They become invalid once
/// the parent tape is dropped — there's no global registry, on
/// purpose: tapes are short-lived, scoped to a single forward pass.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TapeId(pub usize);

/// A single recorded forward operation. Each variant carries the
/// inputs, the output [`TapeId`], and any saved values the adjoint
/// needs at backward time (e.g. the elementwise multiplier or the
/// matrix). The forward results themselves live on the tape's
/// `values` map so the adjoint can reload them without re-running
/// the host computation.
#[derive(Clone, Debug)]
pub enum TapeOp {
    /// `out = a + b` — both `a` and `b` must have the same length.
    Add { a: TapeId, b: TapeId, out: TapeId },
    /// `out = a - b` — both `a` and `b` must have the same length.
    Sub { a: TapeId, b: TapeId, out: TapeId },
    /// `out = a * scalar` — elementwise scalar multiply. The scalar is
    /// stored as a captured constant; it's not a differentiated input
    /// in v1 (covers the XPBD constraint projection where compliance
    /// flows through Mul as a scalar).
    Mul { a: TapeId, scalar: Real, out: TapeId },
    /// `out = M · v` for a fixed `M` (3×3) and a 3-vector `v`. The
    /// matrix is captured by value and not differentiated in v1 —
    /// it's typically a Jacobian or inertia transport that's
    /// constant within the constraint projection step.
    MatVec {
        v: TapeId,
        m: [[Real; 3]; 3],
        out: TapeId,
    },
    /// `out = ||v||₂` — single-element output (norm) of a 3-vector.
    /// Adjoint: `dC/dv = (v / ||v||) · dC/dout`; degenerate `||v|| = 0`
    /// yields a zero adjoint (consistent with the XPBD code path,
    /// which short-circuits `len < 1e-9`).
    Norm { v: TapeId, out: TapeId },
    /// `out_i = ρ(r_i)` — Stage 24b robust-loss elementwise application.
    /// One adjoint variant per loss family so the backward pass can
    /// dispatch the correct derivative without storing the loss type
    /// behind a trait object. Inputs are always single-vector slices
    /// (scalars or per-residual stacks); output dimensionality matches
    /// input.
    L1Loss { r: TapeId, out: TapeId },
    /// `out = √(r² + ε²) − ε` (smooth L2/L1).
    SmoothL2Loss { r: TapeId, eps: Real, out: TapeId },
    /// `out = norm_huber(r, δ)`.
    HuberLoss { r: TapeId, delta: Real, out: TapeId },
    /// `out = norm_pseudo_huber(r, δ)`.
    PseudoHuberLoss { r: TapeId, delta: Real, out: TapeId },
    /// `out = norm_tukey(r, c)` (biweight; zero gradient outside `±c`).
    TukeyLoss { r: TapeId, c: Real, out: TapeId },
    /// `out = norm_cauchy(r, c)`.
    CauchyLoss { r: TapeId, c: Real, out: TapeId },
    /// **Featherstone forward kinematics + body-velocity sweep**
    /// (Stage 23.1) — pass 1 of the ABA. Inputs are tape ids for the
    /// joint coordinates `q` and joint velocities `qd` at the start of
    /// the step. The adjoint is implemented in
    /// [`super::featherstone_adjoint::FeatherstoneRevoluteChain::reverse`]
    /// rather than via the per-op dispatch in [`Self::backward`] —
    /// the entire 3-sweep step is differentiated together (it has to,
    /// because the inertia + acceleration sweeps depend on intermediate
    /// quantities from this sweep) and the chunked TapeOp shape exists
    /// so a future GPU port can swap each sweep for its own kernel
    /// dispatch without touching the tape format.
    FeatherstoneFwdKinematics { q: TapeId, qd: TapeId },
    /// **Featherstone upward articulated-inertia sweep** (Stage 23.1)
    /// — pass 2 of the ABA. Inputs include the joint torque vector
    /// `tau` whose gradient is the primary interest of the trajectory-
    /// optimisation use case.
    FeatherstoneUpwardInertia {
        q: TapeId,
        qd: TapeId,
        tau: TapeId,
    },
    /// **Featherstone downward acceleration sweep** (Stage 23.1) —
    /// pass 3 of the ABA. Carries the post-step tape ids
    /// `(q_new, qd_new, ee)` so downstream tape ops can chain off the
    /// result without round-tripping through the host.
    FeatherstoneDownwardAccel {
        q: TapeId,
        qd: TapeId,
        tau: TapeId,
        q_new: TapeId,
        qd_new: TapeId,
        ee: TapeId,
    },
}

impl TapeOp {
    /// Write the forward result into `values`. Used by [`Tape::record`]
    /// so callers don't have to compute the output twice.
    fn forward(&self, values: &mut HashMap<TapeId, Vec<Real>>) {
        match *self {
            TapeOp::Add { a, b, out } => {
                let av = values.get(&a).cloned().unwrap_or_default();
                let bv = values.get(&b).cloned().unwrap_or_default();
                let n = av.len().min(bv.len());
                let mut o = vec![0.0; n];
                for i in 0..n {
                    o[i] = av[i] + bv[i];
                }
                values.insert(out, o);
            }
            TapeOp::Sub { a, b, out } => {
                let av = values.get(&a).cloned().unwrap_or_default();
                let bv = values.get(&b).cloned().unwrap_or_default();
                let n = av.len().min(bv.len());
                let mut o = vec![0.0; n];
                for i in 0..n {
                    o[i] = av[i] - bv[i];
                }
                values.insert(out, o);
            }
            TapeOp::Mul { a, scalar, out } => {
                let av = values.get(&a).cloned().unwrap_or_default();
                let o: Vec<Real> = av.iter().map(|x| x * scalar).collect();
                values.insert(out, o);
            }
            TapeOp::MatVec { v, m, out } => {
                let vv = values.get(&v).cloned().unwrap_or_default();
                let x = vv.first().copied().unwrap_or(0.0);
                let y = vv.get(1).copied().unwrap_or(0.0);
                let z = vv.get(2).copied().unwrap_or(0.0);
                let o = vec![
                    m[0][0] * x + m[0][1] * y + m[0][2] * z,
                    m[1][0] * x + m[1][1] * y + m[1][2] * z,
                    m[2][0] * x + m[2][1] * y + m[2][2] * z,
                ];
                values.insert(out, o);
            }
            TapeOp::Norm { v, out } => {
                let vv = values.get(&v).cloned().unwrap_or_default();
                let n2: Real = vv.iter().map(|x| x * x).sum();
                values.insert(out, vec![n2.sqrt()]);
            }
            TapeOp::L1Loss { r, out } => {
                let rv = values.get(&r).cloned().unwrap_or_default();
                let o: Vec<Real> = rv.iter().map(|x| norm_l1(*x)).collect();
                values.insert(out, o);
            }
            TapeOp::SmoothL2Loss { r, eps, out } => {
                let rv = values.get(&r).cloned().unwrap_or_default();
                let o: Vec<Real> = rv.iter().map(|x| norm_l2_smooth(*x, eps)).collect();
                values.insert(out, o);
            }
            TapeOp::HuberLoss { r, delta, out } => {
                let rv = values.get(&r).cloned().unwrap_or_default();
                let o: Vec<Real> = rv.iter().map(|x| norm_huber(*x, delta)).collect();
                values.insert(out, o);
            }
            TapeOp::PseudoHuberLoss { r, delta, out } => {
                let rv = values.get(&r).cloned().unwrap_or_default();
                let o: Vec<Real> = rv.iter().map(|x| norm_pseudo_huber(*x, delta)).collect();
                values.insert(out, o);
            }
            TapeOp::TukeyLoss { r, c, out } => {
                let rv = values.get(&r).cloned().unwrap_or_default();
                let o: Vec<Real> = rv.iter().map(|x| norm_tukey(*x, c)).collect();
                values.insert(out, o);
            }
            TapeOp::CauchyLoss { r, c, out } => {
                let rv = values.get(&r).cloned().unwrap_or_default();
                let o: Vec<Real> = rv.iter().map(|x| norm_cauchy(*x, c)).collect();
                values.insert(out, o);
            }
            // Featherstone sweeps are evaluated in
            // `record_featherstone_step` outside this generic forward,
            // because the three-sweep dependency chain crosses tape ids
            // (the user pre-allocates the output ids and seeds them
            // with the post-step values). No-op here.
            TapeOp::FeatherstoneFwdKinematics { .. }
            | TapeOp::FeatherstoneUpwardInertia { .. }
            | TapeOp::FeatherstoneDownwardAccel { .. } => {}
        }
    }

    /// Apply the adjoint of this op, propagating `grad_out` (the
    /// adjoint already accumulated against the output id) into the
    /// adjoints of the inputs.
    ///
    /// The accumulator semantics match Warp's tape: `grads[in] += ...`
    /// (i.e. fan-in adds; downstream consumers can reuse the same
    /// input id and the gradients sum). New entries default to a
    /// zero vector sized to the input value at forward time.
    fn backward(
        &self,
        values: &HashMap<TapeId, Vec<Real>>,
        grads: &mut HashMap<TapeId, Vec<Real>>,
    ) {
        match *self {
            TapeOp::Add { a, b, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                accumulate(grads, a, &go, value_len(values, a));
                accumulate(grads, b, &go, value_len(values, b));
            }
            TapeOp::Sub { a, b, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                accumulate(grads, a, &go, value_len(values, a));
                let neg: Vec<Real> = go.iter().map(|x| -x).collect();
                accumulate(grads, b, &neg, value_len(values, b));
            }
            TapeOp::Mul { a, scalar, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                let scaled: Vec<Real> = go.iter().map(|x| x * scalar).collect();
                accumulate(grads, a, &scaled, value_len(values, a));
            }
            TapeOp::MatVec { v, m, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                // dL/dv = Mᵀ · dL/dout
                let gx = go.first().copied().unwrap_or(0.0);
                let gy = go.get(1).copied().unwrap_or(0.0);
                let gz = go.get(2).copied().unwrap_or(0.0);
                let gv = vec![
                    m[0][0] * gx + m[1][0] * gy + m[2][0] * gz,
                    m[0][1] * gx + m[1][1] * gy + m[2][1] * gz,
                    m[0][2] * gx + m[1][2] * gy + m[2][2] * gz,
                ];
                accumulate(grads, v, &gv, value_len(values, v));
            }
            TapeOp::Norm { v, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                let go0 = go.first().copied().unwrap_or(0.0);
                let vv = values.get(&v).cloned().unwrap_or_default();
                let n2: Real = vv.iter().map(|x| x * x).sum();
                let n = n2.sqrt();
                if n < 1e-12 {
                    // Degenerate — match the XPBD short-circuit. The
                    // adjoint is genuinely zero in the limit, so we
                    // don't accumulate anything.
                    return;
                }
                let gv: Vec<Real> = vv.iter().map(|x| x / n * go0).collect();
                accumulate(grads, v, &gv, value_len(values, v));
            }
            TapeOp::L1Loss { r, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                let rv = values.get(&r).cloned().unwrap_or_default();
                let n = go.len().min(rv.len());
                let mut gv = vec![0.0; rv.len()];
                for i in 0..n {
                    gv[i] = norm_l1_grad(rv[i]) * go[i];
                }
                accumulate(grads, r, &gv, value_len(values, r));
            }
            TapeOp::SmoothL2Loss { r, eps, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                let rv = values.get(&r).cloned().unwrap_or_default();
                let n = go.len().min(rv.len());
                let mut gv = vec![0.0; rv.len()];
                for i in 0..n {
                    gv[i] = norm_l2_smooth_grad(rv[i], eps) * go[i];
                }
                accumulate(grads, r, &gv, value_len(values, r));
            }
            TapeOp::HuberLoss { r, delta, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                let rv = values.get(&r).cloned().unwrap_or_default();
                let n = go.len().min(rv.len());
                let mut gv = vec![0.0; rv.len()];
                for i in 0..n {
                    gv[i] = norm_huber_grad(rv[i], delta) * go[i];
                }
                accumulate(grads, r, &gv, value_len(values, r));
            }
            TapeOp::PseudoHuberLoss { r, delta, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                let rv = values.get(&r).cloned().unwrap_or_default();
                let n = go.len().min(rv.len());
                let mut gv = vec![0.0; rv.len()];
                for i in 0..n {
                    gv[i] = norm_pseudo_huber_grad(rv[i], delta) * go[i];
                }
                accumulate(grads, r, &gv, value_len(values, r));
            }
            TapeOp::TukeyLoss { r, c, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                let rv = values.get(&r).cloned().unwrap_or_default();
                let n = go.len().min(rv.len());
                let mut gv = vec![0.0; rv.len()];
                for i in 0..n {
                    gv[i] = norm_tukey_grad(rv[i], c) * go[i];
                }
                accumulate(grads, r, &gv, value_len(values, r));
            }
            TapeOp::CauchyLoss { r, c, out } => {
                let go = grads.get(&out).cloned().unwrap_or_default();
                let rv = values.get(&r).cloned().unwrap_or_default();
                let n = go.len().min(rv.len());
                let mut gv = vec![0.0; rv.len()];
                for i in 0..n {
                    gv[i] = norm_cauchy_grad(rv[i], c) * go[i];
                }
                accumulate(grads, r, &gv, value_len(values, r));
            }
            // Featherstone adjoints are NOT evaluated through the
            // generic `Tape::backward` — the three sweeps form an
            // inseparable unit (the inertia + acceleration sweeps need
            // the kinematic intermediates), so the adjoint runs
            // externally via
            // `super::featherstone_adjoint::FeatherstoneRevoluteChain::reverse`.
            // The op variants are recorded for tape inspection /
            // determinism checks + so the GPU port can iterate the
            // tape and dispatch one kernel per sweep.
            TapeOp::FeatherstoneFwdKinematics { .. }
            | TapeOp::FeatherstoneUpwardInertia { .. }
            | TapeOp::FeatherstoneDownwardAccel { .. } => {}
        }
    }
}

fn value_len(values: &HashMap<TapeId, Vec<Real>>, id: TapeId) -> usize {
    values.get(&id).map(Vec::len).unwrap_or(0)
}

fn accumulate(
    grads: &mut HashMap<TapeId, Vec<Real>>,
    id: TapeId,
    delta: &[Real],
    target_len: usize,
) {
    let entry = grads.entry(id).or_insert_with(|| vec![0.0; target_len]);
    if entry.len() < delta.len() {
        entry.resize(delta.len(), 0.0);
    }
    for (e, d) in entry.iter_mut().zip(delta.iter()) {
        *e += *d;
    }
}

/// Records forward operations and replays their adjoints.
///
/// Lifecycle:
/// 1. [`Tape::start`] — create a new tape + reset state.
/// 2. [`Tape::alloc`] — register a new input (or intermediate) tensor.
/// 3. [`Tape::record`] — append a forward op; the tape evaluates it
///    and stores the result so the adjoint can reload it later.
/// 4. [`Tape::stop`] — closes recording; further [`Tape::record`]
///    calls panic in debug.
/// 5. [`Tape::backward`] — replay adjoints, return `grads`.
#[derive(Default)]
pub struct Tape {
    next_id: usize,
    /// Recorded ops in forward order. Public to the autograd module so
    /// solver-specific recorders (e.g.
    /// [`super::featherstone_adjoint::record_featherstone_step`]) can
    /// push their own variant chunks without round-tripping through the
    /// generic [`Self::record`] (which assumes a single output id per
    /// op — Featherstone produces three).
    pub(crate) ops: Vec<TapeOp>,
    /// Forward values keyed by [`TapeId`]. Inputs land here via
    /// [`Tape::alloc`]; intermediate outputs land here via
    /// [`Tape::record`].
    pub values: HashMap<TapeId, Vec<Real>>,
    recording: bool,
}

impl Tape {
    /// Create + immediately start a tape, ready to record.
    #[must_use]
    pub fn start() -> Self {
        Self {
            next_id: 0,
            ops: Vec::new(),
            values: HashMap::new(),
            recording: true,
        }
    }

    /// Stop recording. Subsequent [`Tape::record`] calls panic in
    /// debug. Backward is still allowed.
    pub fn stop(&mut self) {
        self.recording = false;
    }

    /// Are we still inside the recording window?
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Number of recorded ops. Useful for tests that want to assert
    /// the tape captured the expected number of kernels.
    #[must_use]
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Allocate a new tape id and seed it with `value`. Use this for
    /// **inputs** (the variables you want gradients for) and for any
    /// constants you want to reference by id.
    pub fn alloc(&mut self, value: Vec<Real>) -> TapeId {
        let id = TapeId(self.next_id);
        self.next_id += 1;
        self.values.insert(id, value);
        id
    }

    /// Allocate an empty tape id without seeding a value. Used by
    /// [`Tape::record`] for op outputs.
    fn alloc_empty(&mut self) -> TapeId {
        let id = TapeId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Record a forward op. Allocates and returns the output id;
    /// the value is computed and stashed in `self.values`.
    pub fn record(&mut self, op_factory: impl FnOnce(TapeId) -> TapeOp) -> TapeId {
        debug_assert!(self.recording, "Tape::record called after stop()");
        let out = self.alloc_empty();
        let op = op_factory(out);
        op.forward(&mut self.values);
        self.ops.push(op);
        out
    }

    /// Convenience: record `out = a + b`, return `out`.
    pub fn add(&mut self, a: TapeId, b: TapeId) -> TapeId {
        self.record(|out| TapeOp::Add { a, b, out })
    }

    /// Convenience: record `out = a - b`, return `out`.
    pub fn sub(&mut self, a: TapeId, b: TapeId) -> TapeId {
        self.record(|out| TapeOp::Sub { a, b, out })
    }

    /// Convenience: record `out = a * scalar`, return `out`.
    pub fn mul_scalar(&mut self, a: TapeId, scalar: Real) -> TapeId {
        self.record(|out| TapeOp::Mul { a, scalar, out })
    }

    /// Convenience: record `out = M · v`, return `out`.
    pub fn matvec(&mut self, v: TapeId, m: [[Real; 3]; 3]) -> TapeId {
        self.record(|out| TapeOp::MatVec { v, m, out })
    }

    /// Convenience: record `out = ||v||₂`, return `out`.
    pub fn norm(&mut self, v: TapeId) -> TapeId {
        self.record(|out| TapeOp::Norm { v, out })
    }

    /// Convenience: record elementwise `out_i = |r_i|`. Stage 24b.
    pub fn l1_loss(&mut self, r: TapeId) -> TapeId {
        self.record(|out| TapeOp::L1Loss { r, out })
    }

    /// Convenience: record elementwise smooth-L2 loss. Stage 24b.
    pub fn smooth_l2_loss(&mut self, r: TapeId, eps: Real) -> TapeId {
        self.record(|out| TapeOp::SmoothL2Loss { r, eps, out })
    }

    /// Convenience: record elementwise Huber loss. Stage 24b.
    pub fn huber_loss(&mut self, r: TapeId, delta: Real) -> TapeId {
        self.record(|out| TapeOp::HuberLoss { r, delta, out })
    }

    /// Convenience: record elementwise pseudo-Huber loss. Stage 24b.
    pub fn pseudo_huber_loss(&mut self, r: TapeId, delta: Real) -> TapeId {
        self.record(|out| TapeOp::PseudoHuberLoss { r, delta, out })
    }

    /// Convenience: record elementwise Tukey biweight loss. Stage 24b.
    pub fn tukey_loss(&mut self, r: TapeId, c: Real) -> TapeId {
        self.record(|out| TapeOp::TukeyLoss { r, c, out })
    }

    /// Convenience: record elementwise Cauchy loss. Stage 24b.
    pub fn cauchy_loss(&mut self, r: TapeId, c: Real) -> TapeId {
        self.record(|out| TapeOp::CauchyLoss { r, c, out })
    }

    /// Replay the recorded ops in reverse, propagating `grad_output`
    /// (seeded against `output_id`) back to the inputs.
    ///
    /// Returns the per-`TapeId` gradient map. Inputs registered via
    /// [`Tape::alloc`] but never reached by the reverse walk are
    /// **not** present in the map (saves an empty vec allocation per
    /// dead input). Use `.get(&id).cloned().unwrap_or_default()` if
    /// you want a guaranteed zero vector.
    pub fn backward(
        &self,
        output_id: TapeId,
        grad_output: &[Real],
    ) -> HashMap<TapeId, Vec<Real>> {
        let mut grads: HashMap<TapeId, Vec<Real>> = HashMap::new();
        let target_len = self.values.get(&output_id).map(Vec::len).unwrap_or(grad_output.len());
        accumulate(&mut grads, output_id, grad_output, target_len);
        for op in self.ops.iter().rev() {
            op.backward(&self.values, &mut grads);
        }
        grads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_grad_is_one_each_input() {
        let mut tape = Tape::start();
        let a = tape.alloc(vec![1.0, 2.0, 3.0]);
        let b = tape.alloc(vec![10.0, 20.0, 30.0]);
        let c = tape.add(a, b);
        tape.stop();
        let grads = tape.backward(c, &[1.0, 1.0, 1.0]);
        assert_eq!(grads[&a], vec![1.0, 1.0, 1.0]);
        assert_eq!(grads[&b], vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn sub_grad_is_negated_for_b() {
        let mut tape = Tape::start();
        let a = tape.alloc(vec![5.0]);
        let b = tape.alloc(vec![3.0]);
        let c = tape.sub(a, b);
        let grads = tape.backward(c, &[2.0]);
        assert_eq!(grads[&a], vec![2.0]);
        assert_eq!(grads[&b], vec![-2.0]);
    }

    #[test]
    fn mul_scalar_is_scalar_through() {
        let mut tape = Tape::start();
        let a = tape.alloc(vec![1.0, 2.0]);
        let b = tape.mul_scalar(a, 7.0);
        let grads = tape.backward(b, &[1.0, 1.0]);
        assert_eq!(grads[&a], vec![7.0, 7.0]);
    }

    #[test]
    fn matvec_grad_is_m_transpose() {
        let mut tape = Tape::start();
        let v = tape.alloc(vec![1.0, 2.0, 3.0]);
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let o = tape.matvec(v, m);
        // Adjoint of M·v is Mᵀ·g_o → gradient pattern is column-major
        let grads = tape.backward(o, &[1.0, 0.0, 0.0]);
        assert_eq!(grads[&v], vec![1.0, 2.0, 3.0]); // first row
    }

    #[test]
    fn norm_grad_is_unit_vector() {
        let mut tape = Tape::start();
        let v = tape.alloc(vec![3.0, 0.0, 0.0]);
        let n = tape.norm(v);
        let grads = tape.backward(n, &[1.0]);
        // d||v||/dv = v / ||v|| → (1, 0, 0)
        assert!((grads[&v][0] - 1.0).abs() < 1e-12);
        assert!(grads[&v][1].abs() < 1e-12);
        assert!(grads[&v][2].abs() < 1e-12);
    }

    #[test]
    fn norm_grad_is_zero_at_origin() {
        let mut tape = Tape::start();
        let v = tape.alloc(vec![0.0, 0.0, 0.0]);
        let n = tape.norm(v);
        let grads = tape.backward(n, &[1.0]);
        // Convention: zero adjoint at the singularity (matches XPBD short-circuit).
        assert!(!grads.contains_key(&v) || grads[&v].iter().all(|x| x.abs() < 1e-12));
    }

    #[test]
    fn fan_in_accumulates() {
        // a feeds two adds; gradient should be the sum
        let mut tape = Tape::start();
        let a = tape.alloc(vec![1.0]);
        let b = tape.alloc(vec![2.0]);
        let c = tape.add(a, b);
        let d = tape.add(a, c);
        let grads = tape.backward(d, &[1.0]);
        // d = a + (a + b) → ∂d/∂a = 2, ∂d/∂b = 1
        assert_eq!(grads[&a], vec![2.0]);
        assert_eq!(grads[&b], vec![1.0]);
    }

    #[test]
    fn huber_loss_grad_matches_analytic() {
        let mut tape = Tape::start();
        let r = tape.alloc(vec![0.5, -2.0, 0.0]);
        let l = tape.huber_loss(r, 1.0);
        // Sum-reduction seed: gradient = ρ'(r_i).
        let grads = tape.backward(l, &[1.0, 1.0, 1.0]);
        let g = grads[&r].clone();
        // Inside band → r; outside → δ·sign(r); kink at 0 → 0.
        assert!((g[0] - 0.5).abs() < 1e-12);
        assert!((g[1] + 1.0).abs() < 1e-12);
        assert!(g[2].abs() < 1e-12);
    }

    #[test]
    fn tukey_loss_zero_outside_band() {
        let mut tape = Tape::start();
        let r = tape.alloc(vec![0.5, 2.0]);
        let l = tape.tukey_loss(r, 1.0);
        let grads = tape.backward(l, &[1.0, 1.0]);
        let g = grads[&r].clone();
        // Inside the band → non-zero; outside → 0 (rejected).
        assert!(g[0].abs() > 1e-6);
        assert!(g[1].abs() < 1e-12);
    }

    #[test]
    fn determinism_same_input_same_grad() {
        let build_and_grad = || {
            let mut tape = Tape::start();
            let a = tape.alloc(vec![1.5, 2.5, -0.5]);
            let b = tape.alloc(vec![0.1, 0.2, 0.3]);
            let s = tape.sub(a, b);
            let n = tape.norm(s);
            let scaled = tape.mul_scalar(n, 3.0);
            let g = tape.backward(scaled, &[1.0]);
            (g.get(&a).cloned().unwrap_or_default(), g.get(&b).cloned().unwrap_or_default())
        };
        let r1 = build_and_grad();
        let r2 = build_and_grad();
        assert_eq!(r1, r2);
    }
}
