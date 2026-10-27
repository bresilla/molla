# Optimizers

`crates/molla/src/optim/` ships the two canonical first-order
optimizers — **Adam** and **SGD with momentum / Nesterov / weight
decay** — both as plain Rust state structs that consume `f64` slices.
Differentiable physics, parameter ID, and policy-learning code paths
in molla all need gradient descent on a flat parameter vector, and
these are the shared primitives that everyone depends on.

The shape mirrors Warp's `wp.optim.{Adam, SGD}`
(`xtra/warp/warp/_src/optim/{adam,sgd}.py`). On the GPU side, the
fused-update kernels live in `crates/molla-kernels/shaders/optim_{adam,sgd}.wgsl`
and are dispatched by `crate::kernels::{optim_adam, optim_sgd}`.

## CPU + GPU precision

Host paths run on `f64` (the engine's [`Real`](./types.md)); GPU
kernels run on `f32` because WGSL has no `f64`. The Rust dispatch
wrappers in `crate::kernels` convert host `f64` slices to `f32`
storage at upload time.

## Adam

```rust
#[derive(Copy, Clone, Debug)]
pub struct AdamConfig {
    pub lr: Real,
    pub beta1: Real,
    pub beta2: Real,
    pub eps: Real,
    pub weight_decay: Real,  // 0.0 disables (AdamW-style when nonzero)
}

#[derive(Clone, Debug)]
pub struct AdamState {
    pub m: Vec<Real>,    // first-moment EMA
    pub v: Vec<Real>,    // second-moment EMA
    pub step: u32,       // 1-based count of updates applied
}

impl AdamState {
    pub fn new(n_params: usize) -> Self;
    pub fn reset(&mut self);
    pub fn step(&mut self, params: &mut [Real], grads: &[Real], cfg: &AdamConfig);
}
```

`AdamConfig::default()` is the canonical Kingma & Ba choice (`lr =
1e-3`, `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, no weight decay).

The update is the standard formula with the bias-correction divisor
clamped to `Real::EPSILON` (guards against `β₁ = 1` / `β₂ = 1`
degeneracies). Optional decoupled weight decay (AdamW, Loshchilov &
Hutter 2019) is applied **after** the moment-correction step using
the pre-update `θ`:

```text
m_t = β₁·m_{t-1} + (1−β₁)·g_t
v_t = β₂·v_{t-1} + (1−β₂)·g_t²
m̂_t = m_t / (1 − β₁^t)
v̂_t = v_t / (1 − β₂^t)
θ_t = θ_{t-1} − lr · m̂_t / (√v̂_t + ε)
θ_t -= lr · wd · θ_{t-1}       # if wd != 0
```

When `weight_decay == 0.0` the path is bit-identical to plain Adam.

### Log-space Adam for positive parameters

For positive-only parameters (Young's modulus, density, mass, viscosity)
that may span orders of magnitude, Adam's scale-invariance breaks down:
a fixed `lr` either oscillates wildly when the parameter is small or
stalls when it's large. The pattern used across the parameter-ID tests
in `crates/molla-fem/tests/fem_diff_optim.rs` is to **optimise in
log-space**:

```rust
// Optimize ξ = ln(E) so the Adam step size is scale-invariant.
// Chain rule: dJ/dξ = dJ/dE · E.
let mut xi = e_init.ln();
let mut adam = AdamState::new(1);
let cfg = AdamConfig { lr: 0.05, ..AdamConfig::default() };

for _ in 0..200 {
    let e: Real = xi.exp();
    // ... forward + adjoint → dj_de ...
    let mut grad = vec![dj_de * e];          // chain rule
    let mut xi_vec = vec![xi];
    adam.step(&mut xi_vec, &grad, &cfg);
    xi = xi_vec[0];
}
let e_recovered = xi.exp();
```

The `· E` factor is the chain rule for `ξ = ln(E) ⇒ dJ/dξ = dJ/dE · E`.
Adam then runs on `ξ` where a `lr` of `0.05` is a 5% multiplicative
step on `E` regardless of the current magnitude. This is the recipe
behind every FEM-elastic parameter-ID experiment in the test suite.

## SGD

```rust
#[derive(Copy, Clone, Debug)]
pub struct SgdConfig {
    pub lr: Real,
    pub momentum: Real,      // 0.0 disables
    pub dampening: Real,
    pub weight_decay: Real,
    pub nesterov: bool,
}

#[derive(Clone, Debug)]
pub struct SgdState {
    pub buf: Vec<Real>,      // momentum buffer
    pub step: u32,
}

impl SgdState {
    pub fn new(n_params: usize) -> Self;
    pub fn reset(&mut self);
    pub fn step(&mut self, params: &mut [Real], grads: &[Real], cfg: &SgdConfig);
}
```

The semantics match molla's GPU `sgd_step_kernel`:

```text
g̃   = g + wd · θ
b_t = μ·b_{t-1} + (1−damp)·g̃    if t > 0
    = g̃                          if t == 0
g_eff = g̃ + μ·b_t                if Nesterov
      = b_t                       otherwise
θ_t = θ_{t-1} − lr · g_eff
```

The first-step special case (`b_0 = g₀` rather than `μ·0 +
(1−damp)·g₀`) is what `SgdState::step` peeks at — the `step` counter
distinguishes it from subsequent steps.

`SgdConfig::default()` is plain SGD with `lr = 1e-3`, no momentum.

## Fused-update GPU kernels

The `shaders/optim_adam.wgsl` kernel runs one thread per parameter
and does the full Adam update in registers:

```wgsl
m[i] = β₁·m[i] + (1−β₁)·g[i]
v[i] = β₂·v[i] + (1−β₂)·g[i]²
m̂   = m[i] / (1 − β₁^t)
v̂   = v[i] / (1 − β₂^t)
θ[i] -= lr · m̂ / (√v̂ + ε)
θ[i] -= lr · wd · θ_pre    // if wd != 0          (decoupled, AdamW)
```

`theta`, `m`, `v` are mutated in place; `grad` is read-only. All
buffers are `f32`. Workgroup size 256 — matches `fem_vector_ops.wgsl`
sizing for warp-aligned vector ops on NVIDIA / Intel / AMD desktop
adapters. The `optim_sgd.wgsl` kernel follows the same shape.

The `crate::kernels::optim_{adam,sgd}` Rust dispatch wrappers handle
the `f64 → f32` conversion, pipeline + bind-group setup, and the
single dispatch per call. Stage 38c parity tests pin CPU and GPU
within `f32` tolerance after many steps; finite-difference gradient
checks (`crates/molla-kernels/tests/optim_gpu_parity.rs`) catch shader
regressions early.

## Cross-references

- [Tape-based autograd](./../autograd/tape.md) feeds gradients into
  these optimizers in the differentiable-physics test suite.
- [FEM adjoint](./../autograd/fem.md) is the main producer of
  per-parameter gradients in parameter-ID experiments.
