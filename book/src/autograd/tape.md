# Tape-based autograd

`molla::autograd` ships tape-style reverse-mode automatic
differentiation modelled on Warp's `wp.Tape`
(`xtra/warp/warp/_src/tape.py`). Every "tensor" produced during the
forward pass earns a `TapeId`; the tape stores the recorded op +
the values needed to evaluate its adjoint; `Tape::backward(seed)`
walks the recorded list in reverse and accumulates per-`TapeId`
gradients.

Source: `crates/molla/src/autograd/tape.rs`.

## Why CPU only (for now)

Differentiability is the **CPU-acceptable exception** to molla's
GPU-first rule. Tape recording fundamentally needs heap allocation
and a dynamic graph structure that doesn't map onto wgpu's
compute-pipeline-state model — Newton + Warp solve the same problem
with dynamic host graphs. We use Rust + CPU and accept the perf hit
(orders of magnitude vs hand-written
adjoint kernels) in exchange for a working, testable autograd v1.

Crucially, the `TapeOp` enum is engine-agnostic: a GPU port can
iterate the tape and dispatch one kernel per op without touching
the tape format. That's the migration path once a per-op
adjoint-kernel queue exists.

## Lifecycle

```rust
use molla::autograd::{Tape, TapeOp};

// 1. Start a tape.
let mut tape = Tape::start();

// 2. Allocate input tensors. `alloc` registers values and returns a TapeId.
let a = tape.alloc(vec![1.0, 2.0, 3.0]);
let b = tape.alloc(vec![4.0, 5.0, 6.0]);

// 3. Record forward ops. The tape evaluates each op and stores the
//    output value, so the adjoint can reload it later.
let c = tape.fresh_id();
tape.record(TapeOp::Add { a, b, out: c });

let n = tape.fresh_id();
tape.record(TapeOp::Norm { v: c, out: n });

// 4. Close recording.
tape.stop();

// 5. Seed the loss gradient and back-propagate.
let mut grads = std::collections::HashMap::new();
grads.insert(n, vec![1.0]);          // ∂L/∂||c|| = 1
let grads = tape.backward(grads);

// 6. Read the input gradients.
let grad_a = grads.get(&a).unwrap();  // ∂L/∂a
let grad_b = grads.get(&b).unwrap();  // ∂L/∂b
```

## Recorded operations

The v1 op set is intentionally small — just enough to express
XPBD's distance-constraint projection on host:

| Variant            | Forward                       | Adjoint                                 |
|--------------------|-------------------------------|-----------------------------------------|
| `Add{a,b,out}`     | `out = a + b`                 | `∂L/∂a += ∂L/∂out`, `∂L/∂b += ∂L/∂out`  |
| `Sub{a,b,out}`     | `out = a − b`                 | `∂L/∂a += ∂L/∂out`, `∂L/∂b -= ∂L/∂out`  |
| `Mul{a,scalar,out}`| `out = a · s`                 | `∂L/∂a += s · ∂L/∂out`                  |
| `MatVec{v,m,out}`  | `out = M · v` (fixed 3×3 `M`) | `∂L/∂v += Mᵀ · ∂L/∂out`                 |
| `Norm{v,out}`      | `out = ‖v‖₂`                  | `∂L/∂v += (v / ‖v‖) · ∂L/∂out`          |

Stage 24b added six [robust-loss](../math/iterative.md) elementwise
variants — `L1Loss`, `SmoothL2Loss`, `HuberLoss`, `PseudoHuberLoss`,
`TukeyLoss`, `CauchyLoss` — so the tape can express the M-estimator
chain used by the inverse-rendering / system-ID loops without
forcing a CPU-side recompute.

Stage 23.1 added three "Featherstone-sweep" chunks
(`FeatherstoneFwdKinematics`, `FeatherstoneUpwardInertia`,
`FeatherstoneDownwardAccel`) that *don't* dispatch through the
generic backward — see [`Featherstone reverse adjoint`](./featherstone.md).
The chunks are recorded so a future GPU port can iterate the tape and swap each
sweep for its own kernel.

## Custom adjoints — `wp.func_grad` pattern

Recording every elementary op into the tape is the **slow path**.
For hot solver loops (XPBD constraint projection, Featherstone
ABA), molla ships hand-written closed-form adjoints that bypass the tape
entirely. These live in their own modules:

- [`xpbd_adjoint`](./xpbd_vbd.md) — manual reverse pass over the
  XPBD distance-constraint Gauss-Seidel loop.
- [`featherstone_adjoint`](./featherstone.md) — tape-recorded
  three-sweep ABA adjoint with a hybrid analytic / finite-difference
  per-step Jacobian.
- [`mpm_adjoint`](./mpm.md) — particle / grid adjoint for
  diff-MPM.
- `fem::adjoint::solve_adjoint` ([FEM adjoint](./fem.md)) —
  classical adjoint method for the linear-elasticity solve.

Performance trade-off: hand-written gradients pay no per-op heap
cost and inline cleanly, but they're locked to one forward path. The
tape generalises to anything you can express as a chain of `TapeOp`s
but cleans up the heap for every recorded intermediate.

## Gradient checking

The autograd tests in `crates/molla-autograd/tests/autograd_xpbd.rs` and
`crates/molla-autograd/tests/autograd_featherstone.rs` exercise the hand-written gradients
against central-difference finite differences with tolerance `1e-3`
(the standard `gradcheck` tolerance Warp ships).

For ad-hoc checks, the pattern is:

```rust
fn loss(p: &[Real]) -> Real { /* forward pass */ }

let p = vec![/* ... */];
let analytic = my_grad_fn(&p);
let mut numerical = vec![0.0; p.len()];
for i in 0..p.len() {
    let mut pp = p.clone(); pp[i] += EPS;
    let mut pn = p.clone(); pn[i] -= EPS;
    numerical[i] = (loss(&pp) - loss(&pn)) / (2.0 * EPS);
}
for (a, n) in analytic.iter().zip(&numerical) {
    assert!((a - n).abs() < 1e-3);
}
```

## Determinism

`Tape::backward` is deterministic for a fixed input + recorded op
sequence. Every operation is a fixed-shape arithmetic loop over the
recorded ops — no `HashMap` *iteration order* matters because the
op list is `Vec<TapeOp>` walked in reverse insertion order, not via
hash iteration. Identical inputs yield bit-exact identical
gradients across runs (verified by the
`tape_replay_is_deterministic` test).

## Cross-references

- [`Featherstone reverse adjoint`](./featherstone.md) — Stage 23.1
  three-sweep ABA adjoint.
- [`XPBD / VBD adjoint`](./xpbd_vbd.md) — hand-written distance-
  constraint adjoint.
- [`MPM adjoint`](./mpm.md) — particle-grid diff-MPM tape.
- [`FEM adjoint`](./fem.md) — classical
  `Kᵀ · λ = ∂J/∂u` adjoint solve.
- [`Differentiable Gaussian splat renderer`](./splat.md) — Stage 44
  family, end-to-end differentiable 3DGS raytrace.
- [`Math / Iterative solvers`](../math/iterative.md) — robust-loss
  functions backing the elementwise `*Loss` ops.
