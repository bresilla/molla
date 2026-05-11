# Inverse kinematics

[`IkSolver`](https://docs.rs/molla/latest/molla/solvers/struct.IkSolver.html)
is molla's analytical IK — a damped least-squares **Levenberg-Marquardt**
solver with a **L-BFGS quasi-Newton fallback**, multiple position /
orientation / joint-limit objectives, robust-loss support (Stage 24b),
and a multi-seed batched GPU path (Stage 24a).

It runs entirely on top of [`crate::solvers::jacobian`](./spatial_kernels.md#jacobian--6--n-body-point-jacobian)
and [`crate::sim::eval_fk`](./featherstone.md#forward-kinematics-shortcut)
— no Featherstone dynamics, no contact solve. The same routine that
captures world poses for the dynamics also captures world poses for
the IK gradient.

```rust,ignore
use molla::solvers::{IkSolver, IkConfig, IkObjective};
use molla::math::robust_loss::RobustLoss;

let objectives = vec![
    IkObjective::Position {
        body: ee_body, point_local: Vec3::ZERO,
        target_world: Vec3::new(0.5, 0.3, 0.2),
        weight: 1.0,
    },
];

let ik = IkSolver::new(objectives)
    .with_config(IkConfig {
        max_iter: 100,
        residual_tol: 1.0e-4,
        step_tol: 1.0e-9,
        damping_init: 1.0e-3,
        damping_grow: 4.0,
        damping_shrink: 4.0,
        articulation: 0,
        loss: RobustLoss::L2,
    });

let result = ik.solve(&model, &mut state, &seed_q)?;
println!("converged in {} iters, residual = {}", result.iters, result.residual);
```

## Objectives

```rust,ignore
pub enum IkObjective {
    /// Drive `body`'s `point_local` to `target_world` (3 residual rows).
    Position {
        body: usize, point_local: Vec3,
        target_world: Vec3, weight: Real,
    },
    /// Drive a relative orientation residual `2 · sin(θ/2)` style
    /// (imaginary part of `q_target · q_current⁻¹`).
    Rotation {
        body: usize, target: Quat, weight: Real,
    },
    /// Soft penalty when `q[dof]` exceeds `[lo, hi]` (1 residual row).
    JointLimit { dof: usize, lo: Real, hi: Real, weight: Real },
}
```

Multiple objectives stack into a single residual vector `e` and a
combined Jacobian `J`. The LM step solves

```text
(JᵀJ + λI) · dq = − Jᵀ e
```

with `λ` adapted per iteration (doubled on rejected steps, halved on
accepted). The orientation residual uses the imaginary part of the
delta quaternion so the gradient is well-defined through the antipodal
identity `q ≡ −q`.

## Robust losses (Stage 24b)

The default `RobustLoss::L2` is the standard L2 normal-equations
solve. The other losses run **iteratively re-weighted least squares
(IRLS)**: at each iteration the per-residual weight `w_i = ρ'(r_i)/r_i`
is folded in as `(JᵀWJ + λI) · dq = −JᵀWe`.

| loss | character | use case |
|------|-----------|----------|
| `L2` | quadratic everywhere | clean targets |
| `Huber` | quadratic near 0, linear past `δ` | mildly noisy targets |
| `Cauchy` | log barrier, bounded influence | strong outliers (drag from sensor) |
| `Tukey` | re-descending, rejects outliers past threshold | aggressive outlier reject |

## Solver health: Cholesky vs PSD pseudo-inverse

The Stage 24a robustness pass added a **PSD pseudo-inverse fallback**
to the normal-equations solver. The primary path is dense Cholesky on
`JᵀJ + λI`; on rare numerical failure (the matrix has a non-SPD pivot
due to underflow / near-singularity at the workspace boundary) the
solver falls back to a damped pseudoinverse computed from the
symmetric eigendecomposition of `JᵀJ + λI`. Two atomic counters track
the rates:

```rust,ignore
pub static IK_CHOL_ATTEMPTS:  AtomicUsize;
pub static IK_PSD_FALLBACKS:  AtomicUsize;

pub fn ik_solver_health() -> (usize, usize);
```

In well-conditioned scenes the fallback rate stays below 0.1 % — the
counters are observability for catching pathological JᵀJ
configurations (e.g. a wrist-singularity where two axes align).

## Multi-seed sampling

When the cost landscape has local minima (long chains, tight
orientation, mixed position + rotation objectives), the user can
sweep multiple seeds and take the best:

```rust,ignore
pub fn solve_with_seeds(
    &self, model: &Model, state: &mut State, seeds: &[Vec<Real>],
) -> Result<IkResult>;
```

Returns the lowest-residual seed (early exits on the first converged
seed). 5–10 random seeds are typically enough to escape local minima
on a 7-DoF arm.

## GPU batched IK (Stage 24a)

For RL workloads that need to solve **the same objective from many
seeds in parallel**, the GPU path batches one LM solve per seed in a
single workgroup grid:

```rust,ignore
#[cfg(feature = "gpu")]
pub fn solve_gpu(
    &self, engine: &Engine, model: &Model, state: &mut State,
    q0: &[Real],
) -> Result<IkResult>;

#[cfg(feature = "gpu")]
pub fn solve_gpu_with_seeds(
    &self, engine: &Engine, model: &Model, state: &mut State,
    seeds: &[Vec<Real>],
) -> Result<IkResult>;
```

Each seed gets its own LM trajectory; the kernel runs FK, Jacobian
assembly, residual evaluation, and the Cholesky solve per workgroup
in lockstep. The Cholesky fallback to PSD-pinv lives inside the
kernel — same fallback counter is incremented per failed seed.

Typical perf: ~ 4× speedup for 32 parallel seeds on a 7-DoF arm
(dev laptop, RTX-class GPU). The fixed overhead of dispatching one
workgroup makes the GPU path slower below ~ 8 parallel seeds.

## L-BFGS fallback

When LM stalls (rejected steps fail to reduce residual past a
threshold), the solver switches to a **L-BFGS quasi-Newton** inner
loop on the same residual cost. L-BFGS doesn't need the JᵀJ build,
so it's robust to rank-deficient Jacobians at singularities, at the
cost of slower asymptotic convergence (linear vs LM's superlinear).

The switch is automatic — `IkResult.iters` counts the combined
LM + L-BFGS iterations, `IkResult.converged` reflects the residual
tolerance from `IkConfig.residual_tol`.

## When to choose IK vs RL / trajectory optimisation

|                          | `IkSolver` | RL | Trajectory opt |
|--------------------------|:----------:|:--:|:--------------:|
| Single pose target       | ✓          | ✗  | ✗              |
| Whole-arm path           | ✗          | ✓  | ✓              |
| Closed-form fast         | ✓          | ✗  | ✗              |
| Handles contact / dynamics| ✗         | ✓  | ✓              |
| Differentiable through trajectory | ✗ | ✗  | ✓             |

IK gives you a `q` for a target. To turn that `q` into a motor command
for a real robot, feed it as a `JointTargetMode::Position` target to
Featherstone with appropriate PD gains. To find a **trajectory** that
threads a sequence of IK targets, use the differentiable Featherstone
adjoint (Stage 23.1) — see [Featherstone adjoint](../autograd/featherstone.md).
