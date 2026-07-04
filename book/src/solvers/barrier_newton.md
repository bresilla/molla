# Barrier-Newton contact (IPC-family)

The penetration-free contact backend (roadmap C0): interpenetration is
*mathematically impossible at every Newton iterate*, not just at
convergence. Designed after the comparison with ZOZO's
`ppf-contact-solver`; molla targets the same invariant at viewer scale
(10k–50k vertices) rather than the 180M-contact offline regime.

## Formulation

Each step minimizes the incremental potential

```text
E(x) = ½‖x − x̃‖²_M + h²·[ elastic(x) + Σ barrier(d_c(x)) + strain(x) ] + friction(x)
x̃    = x_prev + h·v + h²·g
```

with the **cubic compact-support barrier** `b(d) = κ(d̂−d)³/(3d̂²)` — C²
at d̂, convex, zero outside the activation band (no log singularity, no
global κ tuning). κ is per-contact and **elasticity-inclusive** by
default: `κ = scale·(m + h²·Σk_incident)/h²`, the TOG 43.6 scheme — a
stiff material meets a proportionally stiff barrier (the 1 kN
elastic-press oracle holds 11.4× more band margin than mass-only κ).
Damped Newton with matrix-free PCG (block mass-diagonal preconditioner,
**warm-started** from the previous direction with blocked early exit on
the GPU), and the piece that carries the guarantee: a **CCD-truncated
line search** — step length capped at 0.9×TOI *before* Armijo
backtracking, so no trial state ever crosses a surface.

Every Hessian term follows one projection discipline (SPD only):
compression-clamped springs, convex-part barriers, Gauss-part tet
volumes, positive-slope friction.

## What it covers

- **Obstacles**: analytic half-space + sphere contacts.
- **Self-collision**: non-adjacent vertex-triangle and edge-edge pairs
  with clamped closest-point weights (exact-a.e. gradients; the
  measure-zero kinks at aligned-grid degeneracies are documented).
  Candidates at 3·d̂ plus a global travel cap `α·max|p| ≤ d̂` make the
  additive-CCD argument airtight: an unseen pair cannot reach contact.
- **Volumetric tets**: `½k_v(V−V₀)²` with exact volume gradients.
- **Hard strain limiting**: the bound comes from *line-search
  truncation* (a limit crossing is a TOI); the strain barrier
  (κ_s = scale·k·rest², elasticity-derived) only keeps springs off the
  wall. An unlimited 15.8× stretch holds at 1.0486 < 1.05.
- **Lagged IPC friction with a true μ_s/μ_k split**: normal force and
  frame frozen at step start; C¹ mollifier (stick well to μ_s·λ, blend,
  kinetic plateau μ_k·λ). The discriminating incline (tan θ between
  μ_k and μ_s) sticks at 0.83 mm creep vs 0.44 m slide — the oracle
  velocity-level friction cannot pass in principle. Pair (cloth-on-
  cloth) friction uses the same recipe on the relative stencil slip.
- **True pinning**: DoF removal (x̃ = x, zero gradient row, identity
  Hessian row). A huge mass does NOT pin under an incremental
  potential — the predictor free-falls independently of mass.
- **Rigid coupling, two ways**: WEAK (`BarrierSolver`) — obstacles
  follow body poses, third-law-exact reaction wrenches, and a
  rigid-side TOI that closes the tunneling hole; and STRONG (ABD) —
  bodies as 12-DOF affine transforms INSIDE the Newton solve via
  proxy vertices (`add_affine_body`), one line search protecting
  everything, momentum machine-exact (5e-15) with no exchange
  approximation.

## API

CPU f64 oracle: `molla_solvers::barrier_contact::BarrierShell`
(`new` → `with_triangles` / `with_tets` / `pin_vertices`, `step`,
invariants `min_gap` / `max_strain`). GPU f32:
`molla_kernels::BarrierNewtonGpu` (`new` / `new_with_mesh` /
`new_full`), host-driven Newton over device gradient/Hessian/PCG/TOI
with 4-byte scalar readbacks and the f32 convergence freeze.

Pair discovery runs on device through Teschner-hash uniform grids
(cell ≥ primitive extent + 3d̂ ⇒ single-ring probes are complete; hash
collisions only add candidates; bucket overflow is a hard error), or —
for adversarial primitive-size distributions — through a
step-granularity **LBVH broad phase** (`use_lbvh`): AABBs inflated by
the travel-cap bound keep the candidate set a superset for the whole
step, and per-iterate filter kernels do the exact tests.

Pair (cloth-on-cloth) friction runs on BOTH backends: the device
freezes the pair set, frames, λ and anchors at step start
(`pair_friction_lag`) — the layer-on-layer discriminator lands the
same 0.62 mm hold on GPU as the f64 oracle.

## Parity philosophy

f32-vs-f64 drapes are *non-unique equilibria*: the GPU drop test gates
on ENERGY (the f32 drape found a 13% deeper minimum than the oracle),
plus the invariant, the resting-gap band, and — for self-collision —
a **host-side** exact intersection scan on read-back positions. Every
test in `barrier_*.rs` ends with an explicit intersection scan.
