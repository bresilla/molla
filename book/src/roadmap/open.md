# Open items

A concrete inventory of work that is **not** shipped on `main` but is on the
trajectory the recent push has set up. Items are grouped by tier:

- **Tier-A — v5 follow-ups.** Small, well-scoped pieces that close out a stage
  already in flight. Each is days, not weeks. Most are "the obvious next
  slice" on top of a v3/v4 that already landed.
- **Tier-B — algorithmic upgrades.** Larger pieces (typically week-scale) that
  upgrade an existing capability from a working v1 to something competitive
  with Newton / Warp at scale.
- **Tier-C — new physics axes.** Genuinely new capabilities. Multi-week.
  Open unless / until a concrete user need surfaces.

Every item lists:

1. A 1–2 sentence problem statement.
2. Dependencies (stages / commits that must be in place).
3. Estimated effort in days.
4. Priority recommendation.
5. A **"would change my mind if"** criterion — the signal that would
   re-rank the item up or down.

Stage numbers and commit hashes refer to entries on `main` as of
2026-05-11 evening (HEAD `2bde182`, Stage 41 v6).

---

## Tier-A — v5 follow-ups (small, well-scoped)

### A1. Kinetic-vs-static friction split (Stage 50 v2)

**Problem.** Stage 50 (`e751d17`) shipped Coulomb friction at Signorini
contact with a single friction coefficient `μ`. Real materials have
distinct static (`μ_s`) and kinetic (`μ_k`) coefficients with `μ_s > μ_k`,
which is what produces stick-slip transitions. The current solver returns
to "stick" the instant the tangential force drops below `μ |λ_n|`, which is
the right answer for `μ_s == μ_k` only.

**Dependencies.** Stage 50 (`e751d17`), Stage 51 active-set loop
(`4a4fc64`).

**Effort.** 1–2 days. The state machine is per-contact; the only invasive
change is a `friction_state: enum { Stick, Slip }` per contact carried
across the active-set sweep.

**Priority.** Medium. Cheap, but `μ_s == μ_k` is the right default for
most robotics workloads.

**Would change my mind if** a user reports inability to reproduce
stick-slip resonance / chatter (e.g. a robot finger gripping a smooth
glass surface, or a tendon-driven gripper at the friction limit) — that
maps cleanly onto distinct `μ_s` / `μ_k`.

### A2. Variable per-muscle stiffness + pennation on GPU (Stage 42 v5)

**Problem.** Stage 42 v3 (`bbd5c61`) shipped the Hill kernel + cable ABA on
GPU; v4 (`0411d0b`) added SEE Newton GPU/CPU parity. The remaining gap:
per-muscle `stiffness_ratio` and pennation angle are honored on CPU but
read from a single shared parameter on GPU (the `d38739e` critic fix did
the CPU side). Tendon-driven biomechanics with mixed fast/slow fibers
needs per-muscle parameter buffers on GPU.

**Dependencies.** Stage 42 v3 (`bbd5c61`), Stage 42 v4 (`0411d0b`),
critic-fix commit `d38739e`.

**Effort.** 1 day. Add per-muscle parameter array + bind group; the kernel
math already supports it.

**Priority.** Medium-low. Single-parameter Hill is fine for
hopper/finger demos; multi-fiber bio-mechanics is the use case that
actually needs this.

**Would change my mind if** a viewer scene needs anisotropic muscles
(typing-finger biomechanics, soft gripper with antagonist pairs).

### A3. SH GPU forward for diffuse/view split (Stage 44 v5)

**Problem.** Stage 44 v3 (`00d8083`) shipped spherical-harmonic colour on
CPU; v4 (`121c9d3`) added SH band 3 + diffuse/view split. The forward
raytrace currently still uses a per-Gaussian RGB on GPU; the SH evaluation
runs as a CPU bake. To go end-to-end GPU we need an SH→RGB shader stage
that evaluates SH at the view direction per pixel.

**Dependencies.** Stage 44 v3 (`00d8083`), Stage 44 v4 (`121c9d3`),
Stage 30 (Gaussian splat primitive).

**Effort.** 1–2 days. WGSL polynomial evaluation, plus per-pixel
view-vector compute. Bands 0–3 means 16 SH coefficients per channel — fits
comfortably in workgroup-local memory.

**Priority.** Medium. Without it, GPU diff-Gaussian for view-dependent
appearance falls back through CPU on every iteration.

**Would change my mind if** Stage 44 v2 pose-recovery (`bb42665`,
scene #99) ever needs >100 Adam iterations per second — that's the
CPU SH bake budget breaking.

### A4. DG limiter GPU port (Stage 43 v5)

**Problem.** Stage 43 v4 (`bed9c43`) shipped the HLL/HLLC/Roe SWE flux on
GPU. The Barth-Jespersen scalar limiter and Krivodonova characteristic
limiter (Stage 43 v3, `d37995e`) still run on CPU between GPU flux
sweeps. The host round-trip per timestep dominates wall time at fine
mesh resolution.

**Dependencies.** Stage 43 v3 (`d37995e`), Stage 43 v4 (`bed9c43`).

**Effort.** 2 days. The limiters are local — one workgroup per element,
read 4 neighbour cells. The Krivodonova characteristic decomposition uses
the same Roe matrix the v4 GPU flux already builds.

**Priority.** Medium. Bottleneck only kicks in once the mesh is large
enough to amortize dispatch cost.

**Would change my mind if** the DG SWE viewer scene (`#97`,
`7cd4e06`) becomes the runtime baseline for any demo we ship — at
that point the round-trip is visible.

### A5. DG-AMR coarsening + level-gap recursion (Stage 45 v5)

**Problem.** Stage 45 v4 (`66d976a`) shipped AMR coarsening for P1 + Krylov
restart after re-mesh. The DG case (Stage 45 v3 / `4442e3d`) handles
*refinement* with 2:1 face balance, but coarsening for DG with arbitrary
level gaps `> 1` is not yet implemented. This blocks long-running AMR
runs where transient regions need to *un*-refine.

**Dependencies.** Stage 45 v3 (`4442e3d`), Stage 45 v4 (`66d976a`).

**Effort.** 2–3 days. DG coarsening is an L²-projection of fine DoFs to
the coarse cell, plus a re-balance sweep. The level-gap recursion is the
loop that handles `level_max − level_min > 1` neighbours.

**Priority.** Medium. The static-AMR cases (e.g. scene #100 `AmrHeat`,
`571934a`) work today; this is for time-varying refinement zones.

**Would change my mind if** a moving-collider demo (the canonical AMR
benchmark) is on the demo roadmap — then this is a blocker.

### A6. Source-side surface quadrature + TetBvh accel (Stage 47 v5)

**Problem.** Stage 47 v3 (`0f5a27e`) shipped surface-only cross-mesh M_st
quadrature; v4 (`0f9e57e`) added the coupled penalty contact
Newton-Raphson. The source-side quadrature still loops over every source
triangle per target point; large meshes need a BVH-accelerated query. The
existing `MeshBvh` is host-side; molla has all the WGSL machinery but the
TetBvh that the nonconforming projection needs is not yet wired up on GPU.

**Dependencies.** Stage 47 v3 (`0f5a27e`), Stage 47 v4 (`0f9e57e`),
Stage 15 (GPU mesh BVH).

**Effort.** 2–3 days. The BVH traversal is already proven for raycast;
extend the kernel signature for "all triangles within distance ε".

**Priority.** Medium-high. Linear-scan source quadrature scales as
`O(N_tgt · N_src)` — fine for unit tests, painful for any real scene.

**Would change my mind if** the user wires NonconformingField into a
non-toy demo (FSI, hand-on-cloth, tool-on-deformable) — at that
point this is a perf must-have.

### A7. Active-set anti-cycling hardening (Stage 51 v2)

**Problem.** Stage 51 (`4a4fc64`) shipped the Signorini active-set update
loop. The current implementation uses a fixed iteration cap (no
cycling-detection). Pathological geometry (sharp corners, near-tangent
contact) can cause set flipping. Newton's `kamino` and most production
solvers use Bland's rule or a least-index tiebreak.

**Dependencies.** Stage 49 (`60a7792`), Stage 50 (`e751d17`),
Stage 51 (`4a4fc64`).

**Effort.** 1 day. Bland's rule is one extra index sort per active-set
update.

**Priority.** Medium. The active-set already converges on every test
case in `tests/fem_signorini_active_set.rs`; cycling is a tail-risk
hardening.

**Would change my mind if** any user-supplied geometry triggers a
fixed-iteration cap fallback. That's the canary for cycling.

---

## Tier-B — algorithmic upgrades (week-scale)

### B1. Π-projection branch of Hiptmair-Xu AMS (Stage 48 v5)

**Problem.** Stage 48 v3 (`fa3616e`) shipped a geometric multigrid V-cycle
for AMS; v4 (`c0cb1a6`) added algebraic multigrid (Ruge-Stüben). The
auxiliary-space AMS preconditioner has two branches: the gradient branch
(`G^T A G`, shipped) and the Π-projection branch (vector Laplacian on
nodal P1 space). Without the Π branch, AMS is incomplete and convergence
degrades when the curl-free component is non-negligible.

**Dependencies.** Stage 46 (H(curl) elements, `e4eab58`), Stage 48 v1+v2
(`f6fe6a1`), Stage 48 v3 (`fa3616e`), Stage 48 v4 (`c0cb1a6`).

**Effort.** 4–5 days. The Π operator is a discrete interpolation from
edge-DoFs to vector nodal DoFs; the nodal Laplacian solver already exists
in the FEM stack.

**Priority.** High. Without Π, AMS is a one-trick preconditioner — it
masks the curl-free component into the smoother, which doesn't scale.

**Would change my mind if** Stage 48 v4 AMG ends up beating AMS on
every reasonable eddy-current mesh — at that point AMS is academic
and we can drop the Π work.

### B2. Block-coupled preconditioning at high ωσ (Stage 48 v6)

**Problem.** Stage 48's eddy-current solver decouples real/imaginary
components in the preconditioner. At high frequency × conductivity
(`ω·σ → ∞`), the real-imag coupling becomes dominant and the
decoupled preconditioner stalls. Newton's reference impl uses a
block-preconditioner that lives on `(real, imag)` pairs.

**Dependencies.** Stage 48 v3 (`fa3616e`), Stage 48 v4 (`c0cb1a6`),
B1 if/when Π-projection lands.

**Effort.** 3–4 days. Reuse the AMG hierarchy; apply it to the 2×2
block at each level.

**Priority.** Medium. Only matters for explicitly high-frequency
problems (skin-effect at MHz scales, induction heating).

**Would change my mind if** a user actually tries an induction-furnace
or motor-laminate-eddy demo and the v4 AMG plateaus at >100 GMRES
iterations.

### B3. Frequency sweep + Coulomb-gauge enforcement (Stage 48 misc)

**Problem.** The eddy-current solver runs at a single frequency per call.
Real industrial workflows (motor design, induction heating) need a
parametric frequency sweep with results stored in a single buffer.
Additionally, the H(curl) discretization has a null-space (gradients of
scalar fields) that the GMRES converges through, but reporting expects a
Coulomb-gauge-projected output.

**Dependencies.** Stage 46 (`e4eab58`), Stage 48 v1+v2 (`f6fe6a1`).

**Effort.** 2–3 days. The sweep is a wrapper loop; gauge enforcement is a
post-solve `∇·A = 0` projection via the existing nodal Laplacian.

**Priority.** Low-medium. The current solver is correct *up to gauge*;
the sweep is a UX nicety.

**Would change my mind if** anyone uses molla for actual motor /
transformer design — that's the workflow this enables.

### B4. Full-GPU NS time integrator with on-device SSP-RK (Stage 41 v7)

**Problem.** Stage 41 v3 (`f2ae80f`) shipped GPU IMEX Navier-Stokes with
the time loop on the host. Stage 41 v4 (`ccbe5c6`) added a fused
RHS axpy on GPU. Stage 41 v5 (`3169fac`) folded async-dot into the Uzawa
inner CG; v6 (`2bde182`) made the inner CG a single dispatch per iter.
The remaining round-trip is the outer SSP-RK time stepper — it's host-side
Rust calling into GPU kernels per substage. Going full-on-device removes
the round-trip and unlocks long-horizon transient simulations.

**Dependencies.** Stage 41 v3–v6 (`f2ae80f`, `ccbe5c6`, `3169fac`,
`2bde182`).

**Effort.** 5–7 days. The on-device time loop needs a "command list
interpreter" pattern; molla doesn't have CUDA Graphs (see OOS), so this
is implemented as a single mega-kernel with internal `workgroupBarrier`s.

**Priority.** Medium-high. This is the last CPU-touch in the NS path.

**Would change my mind if** transient NS turns out not to be on the
demo roadmap. At that point the round-trip is a fixed cost per call
that probably isn't worth a week.

### B5. Coupled active-set + friction (Stage 49 v2 + Stage 50 integration)

**Problem.** Stage 49 (`60a7792`) shipped Signorini with a fixed active
set. Stage 50 (`e751d17`) shipped Coulomb friction *at* Signorini contact
but assuming the normal active-set is already converged. Stage 51
(`4a4fc64`) added the active-set update loop. The three pieces work in
sequence on every test we have, but a fully coupled solver — the
active-set updates *and* the friction Lagrangians live in the same
Newton-Raphson loop — would converge more uniformly on stick-slip
transitions.

**Dependencies.** Stage 49 (`60a7792`), Stage 50 (`e751d17`),
Stage 51 (`4a4fc64`), A1 if `μ_s`/`μ_k` split is wanted.

**Effort.** 4–6 days. The Schur complement of the friction multipliers
joins the active-set's KKT system; the solver structure stays the same.

**Priority.** Medium. The sequential version is correct on every existing
test; this is a robustness upgrade for harder geometry.

**Would change my mind if** any reported failure case traces to "the
friction step destabilizes the active set on the next outer iter."

### B6. Self-contact for cross-mesh contact pipelines (Stage 47/49/50)

**Problem.** Every Stage 47–51 contact pipeline (penalty in v3/v4,
Signorini Lagrange in 49, friction in 50, active-set in 51) operates on
**cross-mesh** contact only. Self-contact (a single deforming mesh
colliding with itself, e.g. a folding cloth or a curling tongue) is not
handled. Stage 33.3 (`5385`) shipped self-collision for the *XPBD soft*
pipeline; the new contact stack needs the equivalent.

**Dependencies.** Stage 47 v4 (`0f9e57e`), Stage 49 (`60a7792`),
Stage 50 (`e751d17`), Stage 51 (`4a4fc64`), Stage 33.3 (existing
self-collision BVH).

**Effort.** 5–7 days. Reuses Stage 33.3's `SoftBvhGpu` for broad-phase;
the narrow-phase needs an excluded-pair filter for same-element contact.

**Priority.** High. Self-contact is the canonical hard test for any
soft-body / FEM contact stack. Currently a real gap on the new
Signorini path.

**Would change my mind if** a user's workload is exclusively
multi-body / hand-on-object / robot-on-environment — those are
cross-mesh by construction.

---

## Tier-C — new physics axes (multi-week)

### C1. Multi-phase flow with surface tension

**Problem.** molla has single-phase NS (Stage 36.2 / 41). Two-phase flow
(water + air, oil + water) needs an interface tracking method (level-set
or VOF) plus surface tension as a `δ`-distribution force on the interface.
No primitive exists today; even the FEM convection element doesn't
support a discontinuous coefficient.

**Dependencies.** Stage 36.2 (FEM NS), Stage 41 v6 (GPU NS) for the
solver substrate. Would benefit from Stage 45 AMR (refine at the
interface).

**Effort.** 10–14 days. Level-set advection + reinitialization + CSF
(continuum surface force) + ghost-fluid for jump conditions.

**Priority.** Low. Only relevant if a real CFD workload appears. The
current FEM NS is built for "incompressible single-phase" and that's
been the user need to date.

**Would change my mind if** a robotics workload involves splashing
liquid, gripper-in-fluid, or any free-surface flow — the existing NS
can't even render the free surface.

### C2. Compressible / Euler equations on DG (Stage 43 follow-on)

**Problem.** Stage 43 v4 (`bed9c43`) shipped HLL/HLLC/Roe flux for SWE.
The same flux library applies to compressible Euler with very few
changes — a 5-equation system instead of 3, and an EOS (ideal gas)
substituted for the SWE depth-pressure relation. This unlocks
supersonic-flow demos (shock tubes, blunt-body) which the SWE solver
can't represent.

**Dependencies.** Stage 39B (DG basis, `d97b5c1`), Stage 43 v2–v4
(`00f36c1`, `d37995e`, `bed9c43`).

**Effort.** 6–8 days. Rusanov is straightforward; the limiter is reused
1:1; Roe + HLLC needs an Euler-flavoured Roe averaging.

**Priority.** Low. SWE is enough for "geometric flow" demos; Euler is for
aerodynamics. Not on any current workload.

**Would change my mind if** a user shows up wanting compressible flow.
The DG infra is already 90 % there.

### C3. Phase-field fracture

**Problem.** Brittle / quasi-brittle fracture as a coupled
displacement + damage field. The damage field `d ∈ [0, 1]` is a second
scalar PDE solved on the same FEM mesh, with a degraded elasticity
tensor `(1−d)² C`. Nothing like this exists today.

**Dependencies.** Stage 36.1 (FEM elasticity), Stage 41 (GPU FEM), Stage
45 (AMR — fracture concentrates in a thin band, AMR is essential).

**Effort.** 14–21 days. Standard reference is Miehe / Bourdin; well-known
algorithm but a lot of FEM plumbing.

**Priority.** Low. Specialty workload (structural engineering,
cracking-glass demos). Not on any roadmap.

**Would change my mind if** anyone needs material-failure simulation.
There's no proxy in molla today.

### C4. Mooney-Rivlin / Neo-Hookean tangents

**Problem.** Stage 36.1 ships linear elasticity + St. Venant-Kirchhoff
(SVK) stress. SVK is wrong in compression (the strain-energy density
isn't convex in the right way) — Neo-Hookean and Mooney-Rivlin are the
standard hyperelastic models for soft tissue / rubber. The forward stress
is easy; the consistent material tangent for an implicit Newton solve
needs the 4th-order tangent moduli, which is per-element work.

**Dependencies.** Stage 36.1 (FEM elasticity, linear + SVK).

**Effort.** 5–7 days. Closed-form 4th-order tangent for Neo-Hookean is
in every nonlinear FEM textbook; Mooney-Rivlin is one more invariant.

**Priority.** Medium-low. Real biomechanics needs Neo-Hookean as the
floor; SVK is "rubber-ish for demos."

**Would change my mind if** the tendon / muscle Stage 42 work expands to
soft-tissue substrate. Tendons are SVK-OK; muscle bellies aren't.

### C5. Rate-and-state friction (Dieterich-Ruina)

**Problem.** Stage 50 ships Coulomb friction. A1 above adds the
static / kinetic split. Geological / seismic / advanced-tribology
workloads need rate-and-state friction — a continuum law with an
internal state variable evolving by an ODE. Earthquake simulation,
wheel-rail contact, paper-on-paper.

**Dependencies.** Stage 50 (`e751d17`), Stage 51 (`4a4fc64`), A1 if
present.

**Effort.** 5–7 days. The state variable is per-contact; the ODE is
1-D and stable. Wiring into the active-set Newton step is the work.

**Priority.** Very low. No current workload.

**Would change my mind if** anyone targets paleo-seismology or
advanced bearing tribology — these are the use cases for D-R.

### C6. GPU port of muscle SEE for non-default stiffness ratios

**Problem.** Stage 42 v4 (`0411d0b`) shipped SEE (series-elastic element)
Newton GPU/CPU parity *for the default* stiffness ratio. The
`d38739e` critic fix made non-default ratios work on CPU. The GPU kernel
still hard-codes the default ratio; non-default ratios fall back to CPU.

**Dependencies.** Stage 42 v3 (`bbd5c61`), Stage 42 v4 (`0411d0b`),
critic fix `d38739e`.

**Effort.** 1–2 days — could legitimately be a v5 Tier-A item; lives in
Tier-C only because the use case (multi-fiber muscle composition) is
specialty.

**Priority.** Low. Combine with A2 (per-muscle parameter buffers) for a
single coherent muscle-on-GPU push.

**Would change my mind if** A2's use case appears — at which point these
get bundled together.

### C7. Krylov restart in transient FEM steppers

**Problem.** Stage 45 v4 (`66d976a`) wired Krylov restart into
`solve_after_remesh`. The transient steppers (`WaveStepperGpu`,
`TransientStepperGpu`, the IMEX NS stepper) don't restart Krylov between
timesteps — each step starts from `x = 0` and rebuilds the Krylov basis.
For slowly varying problems, warm-starting with the previous solution +
a few Krylov vectors cuts iteration count substantially.

**Dependencies.** Stage 36.5 (GPU FEM v1), Stage 41 v3 (`f2ae80f`),
Stage 45 v4 (`66d976a`).

**Effort.** 3–5 days. The Krylov-basis carry-over is a buffer-management
problem; the math is standard.

**Priority.** Medium. Quiet 10–30 % speedup on every transient NS / heat
demo we have.

**Would change my mind if** any benchmark shows iteration count already
< 5 per step — then there's no slack to recover.

---

## Cross-cutting (not tier-specific)

### X1. Critic round on Stages 39–51

The last focused read-only audit was `90f3708` (Stage 38 critic round,
two safety-contract fixes). Since then, **~25 stages have shipped without
a focused audit**. The `d38739e` round did spot-fixes for Stages 39–46
but was reactive, not systematic. A planned audit pass (1 day,
read-only) is overdue. Expect 0–2 critical-math issues and 5–10 contract
/ doc issues, by historical base rate.

### X2. Viewer scenes for the v3–v6 work

Scenes #94–#101 (`0076a36` through `40e9ed7`) cover the v1/v2 capabilities
of Stages 39–46. The v3–v6 work (Stages 40 v4, 41 v4–v6, 43 v3–v4,
44 v3–v4, 45 v3–v4, 47 v2–v4, 48 v3–v4, 49, 50, 51) has zero new viewer
scenes. ~1 day of mechanical work, high marginal demo impact.
