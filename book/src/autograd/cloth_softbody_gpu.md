# GPU cloth & softbody adjoints

`molla_kernels::ClothDiffGpu` is the device-resident differentiable
mass-spring stack (Warp-style force-based diffsim — the parallel-friendly
formulation; the XPBD Gauss-Seidel adjoint stays CPU-side in
`molla-autograd` because its sequential reverse is GPU-hostile by
construction).

## Forward

Semi-implicit Euler, **gather form**: each particle loops its incident
springs (and tets) via CSR — no atomics, so the forward is
bit-deterministic. That determinism is load-bearing: it makes central
finite differences *of the GPU forward itself* a valid oracle for the
reverse kernels, with no CPU twin required.

- Springs: `f = k (L − rest) d̂` per `ClothSpring`.
- Softbody volume term (`SoftTet`, via `new_with_tets`): per-tet energy
  `½ k_v (V − V₀)²`, rest volumes measured from the initial positions.

## Reverse

Three passes per step, walking the stored trajectory backwards:

1. `backward_hat` — stage `â_v = adj_v + dt·adj_p` into a separate buffer
   (in-place updates would race on neighbours' values mid-dispatch).
2. `backward_stiffness` / `backward_volume_stiffness` — per-spring ∂J/∂k and
   per-tet ∂J/∂k_v from the staged `â_v`.
3. `backward_step` — the adjoint state update. The spring path applies the
   symmetric spring Jacobian; the volume path applies the full volume
   Hessian in closed form: the Gauss part `k_v·gᵢ(Σⱼ gⱼ·xⱼ)` plus the
   geometric part `k_v(V−V₀)·Σⱼ ∂gᵢ/∂pⱼ xⱼ`, where the Hessian blocks are
   edge-cross hat matrices and Hessian symmetry lets the adjoint reuse the
   forward-form blocks (a-row/col recovered by translation invariance).

Gradients exposed: `read_grad_k`, `read_grad_kv`, `read_grad_v0`,
`read_grad_p0`; `set_stiffness` for optimiser loops.

## Validation (`cloth_diff_gpu.rs`)

FD gradchecks against the deterministic GPU forward:

- Cloth: top-5 |∂J/∂k| springs at 1.6e-4…1.2e-3 relative; ∂J/∂v₀ at
  6e-4…6e-3. The test samples the *largest* gradients — fixed indices can
  land on near-zero-signal springs whose FD drowns in ~1e-7 f32 loss noise.
- Softbody: all five tet ∂J/∂k_v at 2e-3…9.5e-3; ∂J/∂v₀ re-checked with the
  volume term active so the adjoint is proven to chain through both force
  families.
