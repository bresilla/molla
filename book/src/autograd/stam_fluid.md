# Eulerian fluid adjoint (differentiable smoke)

`molla_autograd::stam_diff::StamFluidDiff` — reverse-mode adjoint of a
2D periodic inviscid Stam step (force add → pressure projection →
velocity self-advection → projection → density advection), completing
the diffsim family coverage (cloth, softbody, MPM, Featherstone,
XPBD/VBD, rays, splats — and now smoke).

Two structural facts carry the whole reverse pass:

1. **The projection is self-adjoint.** `P = I − G·L̃⁻¹·D` with periodic
   central differences satisfies `Dᵀ = −G`, and the K-sweep Jacobi
   approximation of `L̃⁻¹` is a polynomial in the symmetric Laplacian
   (constant diagonal on the periodic grid), hence symmetric. The
   reverse pass literally re-runs the forward projection on the
   adjoint field — no stored state.
2. **Advection splits cleanly**: the bilinear gather transposes into a
   scatter with the same weights, and the backtrace dependence feeds
   `−dt·(∇̃field)(x_back)` into the advecting velocity's adjoint,
   where `∇̃` is the interpolant's own gradient — exact a.e., with the
   same measure-zero cell-boundary caveat as the barrier stencils.

Checkpoints per step: the pre-advection velocity and density fields.
Gradients exposed: ∂J/∂force, ∂J/∂source, ∂J/∂u₀, ∂J/∂ρ₀.

**Contract** (`tests/stam_diff.rs`): FD through 12 full steps —
∂J/∂force to 1.9e-7, ∂J/∂ρ₀ to 5.0e-6; and the use case: 25 gradient
steps on a per-cell force field steer a smoke blob onto a target,
nearly doubling the captured mass. Scope notes: inviscid (Gauss-Seidel
diffusion is not self-adjoint and is deliberately out), f64 CPU; the
GPU port follows the house recipe when a workload needs it.
