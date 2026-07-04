# Open items

> **Current open set (2026-07-04).** Most entries below have since
> **landed** and carry an inline `**Status.** Landed` block — they are
> kept as decision records, not backlog. Since 2026-07-02: X1 (critic
> audit over Stages 39–51) ran — three real findings, all fixed
> (see INSPECTION.md); the tire stream is COMPLETE (fully autonomous
> zero-readback loop, 40-wheel fused atlas at 1.53× realtime with
> heightfield terrain); the INSPECTION.md backlog is CLOSED. Remaining:
>
> | Item | What | Effort |
> |------|------|--------|
> | **C0** (Tier-C below) | **Barrier-Newton contact backend (IPC-family, GPU)** — penetration-free invariant + CCD line search; subsumes F14 and obsoletes A1 for deformables. The flagship next feature; full design in the C0 entry. | multi-week (phase 1 = 1–2 sessions) |
> | A1 (Tier-A below) | Kinetic-vs-static friction split (μ_s > μ_k) in the rigid XPBD contact pass | 1–2 days |
>
> **A1 attempt notes (2026-07-04, reverted — start here next time).** A
> velocity-pass-only split does NOT discriminate stick from slip: on an
> incline with tan θ between μ_k and μ_s, the box creeps identically in
> both cases because single-application friction under-relaxes on corner
> contacts (the friction cousin of the B1b finding) — a per-contact
> ACCUMULATED tangential impulse clamped to the cone across sweeps
> (PGS-style) fixes the velocity side and is worth keeping. But true
> stick then still leaks `g·sinθ·dt²` of POSITIONAL creep per step, and
> naive Müller §3.5 positional friction (tangential anchor correction
> gated on λ_t ≤ μ_s·λ_n per position iteration) OVER-STICKS: with
> contacts regenerated every outer step, the detected depth carries the
> contact margin + settled penetration, inflating λ_n so the ratio test
> `λ_t/λ_n vs tan θ` always passes and even tan θ > μ_s holds. The
> correct implementation needs contact-LIFETIME normal/tangential λ
> accumulation (persistent contact identity across steps — the contact
> reduction module already threads persistent IDs) so the cone ratio
> compares displacements accumulated over the same window. That is the
> real 1–2 day scope; the WGSL side additionally needs a per-(contact,
> side) accumulator buffer indexed by the body-contact CSR entry to stay
> race-free under the per-body threading model.
> | X2 (bottom of file) | Viewer scenes for the v3–v6 solver work — HELD with the maintainer's GUI split (see GUI_SPLIT.md) | held |

A concrete inventory of work from the v3–v6 push, grouped by tier:

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
2026-05-11 evening (HEAD `2bde182`, Stage 41 v6); the open-set banner
above reflects 2026-07-02. Note this snapshot predates the entire
`molla-vehicle` / tire-FEM effort, which is tracked in
`PLAN_REALTIRE.md` + `PROGRESS.md` instead.

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

**Status.** Landed (implementation shipped with the C6 push; validated
2026-07-02). The WGSL kernel reads per-muscle
`muscle_pennation_angle`/`muscle_tendon_stiffness_ratio` storage
buffers (`muscle.wgsl` bindings 7/9) uploaded from the `Model` arrays,
and the heterogeneous-parameter hardware parity lane
`per_muscle_pennation_and_stiffness_cpu_gpu_parity` in
`crates/molla-solvers/tests/joint_muscle_gpu.rs` confirms CPU/GPU
agreement (with a homogeneous-control guard against a vacuous pass).

**Problem (historical).** Stage 42 v3 (`bbd5c61`) shipped the Hill kernel + cable ABA on
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

**Status.** Landed in the modular workspace. The GPU path is
`SplatRaytraceKernel::dispatch_sh_diffuse_view` with WGSL in
`crates/molla-kernels/shaders/splat_raytrace_sh_diffuse_view.wgsl`.

**Result.** Diffuse RGB and the 15 band-1..3 SH-view coefficients are
bound as separate buffers and evaluated per pixel/view direction on GPU,
so iterative differentiable-splat workflows no longer need a CPU bake
into full 16-coefficient band-3 SH.

### A4. DG limiter GPU port (Stage 43 v5)

**Status.** Landed for DG-P1 shallow-water state buffers. The GPU path is
`DgSweLimiterKernel` with WGSL in
`crates/molla-fem/shaders/dg_swe_limiter.wgsl`.

**Result.** Both scalar component-wise Barth-Jespersen and
Krivodonova-style characteristic limiting can run as resident GPU passes
after the Stage 43 v4 flux sweep. The dispatcher reads `state_in` and
writes `state_out` to preserve the CPU snapshot semantics while allowing
ping-pong buffers between GPU RK stages.

### A5. DG-AMR coarsening + level-gap recursion (Stage 45 v5)

**Status.** Landed for DG-P1 triangle coarsening helpers.

**Result.** `AmrMesh::coarsen_subtree` collapses deep trees from the
leaves upward, and `project_dg_p1_tri_children_to_parent_l2` performs the
child-to-parent L² projection for DG-P1 triangle fields. The regression
coverage lives in `crates/molla-fem/tests/fem_amr_v5.rs`.

**Would change my mind if** a moving-collider demo (the canonical AMR
benchmark) is on the demo roadmap — then this is a blocker.

### A6. Source-side surface quadrature + TetBvh accel (Stage 47 v5)

**Status.** Landed for the current nonconforming projection paths.

**Result.** `molla_geometry::TetBvh` now provides a CPU BVH over per-tet
AABBs, and `molla-fem::NonconformingField` builds a source-side
`SourceTetIndex` once for volume and target-surface cross-mass assembly.
Each quadrature point queries BVH candidates before the exact
barycentric inside-tet test, avoiding the old linear source-tet scan.

### A7. Active-set anti-cycling hardening (Stage 51 v2)

**Status.** Landed.

**Result.** `solve_active_set_signorini` now records seen active sets and
uses a Bland-style least-slave-DoF single pivot when a proposed update
would revisit a previous set. `ActiveSetTrace::cycle_breaks` and
`ActiveSetIteration::bland_pivot` expose the diagnostic path.

---

## Tier-B — algorithmic upgrades (week-scale)

### B1. Π-projection branch of Hiptmair-Xu AMS (Stage 48 v5)

**Status.** Landed.

**Result.** `molla-fem` now builds the vector-nodal interpolation
operator `Π : (P1)^3 -> NedelecEdge`, pins the matching vector-P1
auxiliary Laplacian at Dirichlet-touched vertices, and adds the fixed
iteration-count Π auxiliary correction inside `AmsPreconditioner::apply`.
The old gradient-only action remains available through
`with_pi_projection(false)` for comparisons.

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

**Status.** Landed.

**Result.** `solve_eddy_current_steady_ams_block` adds a high-`ωσ`
preconditioner path. Its preconditioner applies AMS to the diagonal
real/imaginary blocks, then performs a fixed number of stationary
refinement sweeps against the full block-real operator so the
off-diagonal `±ωM_σ` coupling enters `M^{-1}r`. The fixed sweep count
is configured by `EddyCurrentConfig::block_precond_sweeps`, preserving
linearity for GMRES.

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

**Status.** Landed.

**Result.** `solve_eddy_current_frequency_sweep` now runs Jacobi, AMS,
or block-AMS solves across a list of angular frequencies. Optional
Coulomb-gauge enforcement projects each real/imaginary potential with
`project_nedelec_coulomb_gauge`, which subtracts the least-squares
gradient component (`G^T G φ = G^T a`) while preserving pinned edge
DoFs. `coulomb_gauge_residual` reports the remaining discrete gauge
residual for each sample.

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

**Status.** Implemented in `molla-fem`: `NavierStokesGpuTimeIntegrator`
now exposes `ImexEuler`, `SspRk2`, and `SspRk3`. SSP-RK2/3 keep all
intermediate velocity/pressure stages in GPU buffers and use the
`combine_states` WGSL entry point in `fem_ns_rhs_assemble.wgsl` for the
Shu-Osher convex combinations. The host still submits the existing GPU
stage kernels, but it no longer reads stage vectors back between SSP
substages; only the final result is downloaded.

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

**Status.** Landed.

**Result.** `solve_active_set_signorini_with_friction` now combines the
PDAS active-set contact loop with the stacked normal+tangent Coulomb
Uzawa solve. The result reports full-length per-candidate normal
multipliers, interleaved tangent multipliers, and `FrictionState`
entries so released contacts automatically release their friction
state and tangent force.

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

**Status.** Landed for the CPU candidate path.

**Result.** `molla-geometry` now exposes
`detect_vertex_triangle_self_contacts`, an `O(VT)` CPU oracle that
emits non-adjacent vertex-triangle shell contacts. `molla-fem` converts
those into active-set gap planes with
`signorini_candidates_from_self_contacts`, reducing multiple contacts
on the same slave to the deepest candidate.

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

### C0. Barrier-Newton contact backend (IPC-family, GPU) — flagship

**Status.** Designed 2026-07-04. **Phase 1 LANDED same day** —
`molla-solvers::barrier_contact` (`BarrierShell`): f64 incremental-potential
Newton with the cubic compact-support barrier, mass-consistent per-contact κ,
SPD-projected spring/barrier Hessians, matrix-free PCG (block mass-diagonal
preconditioner), and the CCD-truncated Armijo line search. Contract tests
(`tests/barrier_contact.rs`): FD gradient 8.8e-7 / Hessian-operator 1.2e-11;
a 50 m/s impact whose free-flight step crosses the floor ~16× stays
penetration-free at every accepted iterate and arrests on the surface; a
draped drop settles with the resting gap INSIDE the barrier band
(9.99e-4 < d̂ = 1e-3); every test ends with an explicit intersection scan.
**Phase 2 LANDED 2026-07-04** — `molla-kernels::BarrierNewtonGpu`
(barrier_newton.wgsl, 18 entries): device gradient/Hessian/PCG (single
submission, f32 freeze)/TOI min-reduce/fused energy+gap+grad reductions;
host Newton loop on 4-byte readbacks. GPU passes the phase-1 contract:
impact penetration-free + arrested; drop holds the invariant and lands in a
13% DEEPER energy minimum than the f64 oracle (parity gated on energy —
drapes are non-unique). f32 reality encoded in defaults: Newton stalls at
~1e-2 relative gradient (tighter ⇒ line-search noise jitter). **Phase 3 CPU oracle
LANDED 2026-07-04** — VT/EE self-contact pairs on `BarrierShell`
(clamped closest-point weights, exact-a.e. gradients FD-verified to
3.5e-7; 3·d̂ candidates + α·max|p| ≤ d̂ travel cap = airtight additive-CCD
argument; true DoF-removal pinning replacing the huge-mass fallacy). The
two-layer stack and crossed-ribbon EE oracles hold the invariant
throughout. **Phase 3b LANDED 2026-07-04** —
device pair discovery (atomic compaction + overflow flag), gather-form
pair gradient/Hessian (deterministic, no f32 atomics), pair TOI + travel
cap in the CCD reduction, device pinning; the GPU two-layer stack matches
the CPU oracle to the digit and the final intersection scan runs
host-side with the oracle's exact machinery. **F14 is closed as
superseded**: for deformables the barrier backend IS the penetration-free
GPU path; VBD's CPU-only CCD gate stays by design. **Phase 4a
LANDED 2026-07-04** — volumetric tets (exact volume gradients, Gauss-part
SPD Hessian) and HARD strain limiting where the hard bound comes from
LINE-SEARCH truncation (a strain-limit crossing is a TOI), with the
barrier κ_s derived from the spring's own stiffness: an unlimited 15.8×
stretch holds at 1.0486 < 1.05 under the same load. **Phase 4b LANDED
2026-07-04** — lagged IPC friction with the μ_s/μ_k split on the CPU
oracle: FD 4.7e-6 across all three mollifier regions; the discriminating
incline (tan θ between μ_k and μ_s) holds at 0.83 mm creep vs 0.44 m
slide — a 530× split, the oracle velocity-level friction (the A1
attempt) cannot pass in principle. **A1 is retired for deformables.**
**Phases 5a/5b/5c LANDED 2026-07-04** — full GPU parity for
tets/strain/friction (the incline discriminator lands IDENTICAL digits
on device), pair friction (layer-on-layer holds at 0.62 mm vs 1.24 m
frictionless — pure VT/EE friction), and device Teschner-hash pair
pruning replacing the brute-force sweeps (single-ring-complete cells,
collisions only add candidates, overflow is a hard error). **C0 is
FEATURE-COMPLETE** — book chapter at `solvers/barrier_newton.md`.
**Elasticity-inclusive κ LANDED 2026-07-04** (same-day follow-on):
κ = scale·(m + h²·Σk_incident)/h² on both backends, default on — the
1 kN elastic-press oracle holds 11.4× more band margin than mass-only
κ, friction λ now shares the barrier's κ, pinned vertices are
boundary conditions (excluded from barriers/invariant), and the f32
Newton gained a noise-floor exit. **Rigid two-way coupling LANDED 2026-07-04**: body-attached obstacles
follow live poses, third-law-exact reaction wrenches (momentum drift
1.4e-7 through a real impact), and a rigid-side TOI that closes the
weak-coupling tunneling hole (ball-on-trampoline rests supported).
**Warm-started PCG + blocked early exit LANDED 2026-07-05** (−30% on
the GPU barrier lane, gates unchanged). Remaining follow-ons: LBVH for
adversarial distributions, GPU behind viewer scene 125, strong
(monolithic/ABD) coupling. Subsumes the F14 GPU-CCD item
(currently a loud refusal on the GPU VBD backend).

**Why.** Comparison against `st-tech/ppf-contact-solver` (Ando/ZOZO — cubic
barrier with elasticity-inclusive dynamic stiffness, ACM TOG 43.6) showed
molla's one real deficit against the contact state of the art:
XPBD/VBD are local projection/impulse methods with **no penetration-free
invariant** — under speed, stiff stacking, or thin geometry they can tunnel
or hold residual penetration (B1b, the CCD gate, and contact margins are all
symptoms of the same class). Barrier-Newton contact makes interpenetration
*mathematically impossible* at every intermediate state, not just at
convergence. Their other advantages (offline scale, CUDA) are design points
we don't want; this one is a capability we lack.

**What it is.** A new solver backend, peer to `XpbdSolver`/`VbdSolver`:

1. **Energy.** Incremental potential per step:
   `E(x) = ½‖x − x̃‖²_M + h²·[ elastic(x) + Σ_contacts barrier(d(x)) ]`
   with the cubic barrier `b(d) = κ·(d − d̂)³/d̂²` for `d < d̂` (zero at and
   beyond d̂ — compact support, no log singularity), and
   **elasticity-inclusive dynamic stiffness**: κ per contact derived from the
   local elastic eigen-stiffness so soft and stiff materials get consistent
   barrier response without hand-tuned global κ (this is the TOG 43.6
   contribution — do not substitute plain IPC log-barrier κ scheduling).
   Friction: lagged Coulomb à la IPC (tangential quadratic mollifier around
   the previous iterate's normal force; μ_s/μ_k split composes here
   naturally — see the A1 findings, which this backend obsoletes for
   deformables).
2. **Newton loop (per step, all on device).**
   a. Broad phase: reuse the LBVH (`molla-geometry/lbvh.rs`, already GPU).
   b. Constraint set: vertex-triangle + edge-edge pairs within d̂ (GPU
      pair kernels exist in the soft-contact pipeline; extend to EE).
   c. Assemble gradient + Hessian: elastic blocks (the FEM element
      Hessians we already assemble on GPU for `elasticity_gpu`) + per-pair
      barrier Hessians (4×4 blocks of 3×3 — new WGSL, analytic, with the
      standard SPD projection of indefinite pair blocks).
   d. Solve `H Δx = −g` with the **existing** device-resident PCG
      (`fem_cg_fused.wgsl` / `BsrAlgebraKernel` — the same machinery the
      mixed-elasticity GPU saddle uses). Block-Jacobi preconditioner v1.
   e. **CCD line search**: α ← min(1, 0.9·TOI(x, Δx)) then backtrack on
      energy. TOI = GPU kernel: per-pair conservative CCD (additive-CCD is
      the pragmatic choice over exact root-finding — monotone, GPU-friendly,
      and sufficient for the invariant); min-reduce α across pairs
      (single-workgroup reduction, same pattern as the APIC CG dots).
      This kernel IS the F14 deliverable — land it here, then also wire it
      into the VBD pre-substep pass to lift the GPU CCD refusal.
   f. Converge on ‖g‖_∞ or Newton-decrement; f32 with the convergence
      freeze discipline from the APIC CG port (f32 Krylov diverges past
      convergence — proven twice in this codebase).
3. **Strain limiting** (phase 2): singular-value constraints on F per
   triangle folded into the same Newton solve as barrier terms on
   `σ_max − σ_limit` — gives ZOZO-style "strict 1–5%, not rubbery" cloth.
4. **Invariant enforcement in tests, not just code**: every test ends with
   an explicit intersection scan (their practice, worth copying verbatim).

**What we already have** (why this is tractable): GPU LBVH broad phase,
VT soft-contact pair kernels, GPU FEM element Hessians + BSR assembly,
device-resident PCG with on-device dots, the f32-freeze lesson, per-body
CSR threading patterns, and a large deformable test corpus with oracles.
The genuinely new pieces are: the barrier energy/Hessian WGSL, EE pairs,
the additive-CCD TOI kernel + α-reduction, and the Newton driver loop.

**Phasing** (each phase lands green independently):
1. CPU oracle: barrier-Newton for a triangle-mesh shell vs a plane +
   sphere, f64, with FD-validated gradient/Hessian and the intersection
   scan. (Small — this is the correctness anchor for everything after.)
2. GPU port of phase 1 (assembly + PCG + CCD line search), parity vs the
   CPU oracle; land the TOI kernel into VBD's CCD path (closes F14).
3. EE pairs + self-collision (the cursed cases: cloth-on-cloth, twist).
4. Tets (reuse `elasticity_gpu` Hessians) + strain limiting + μ_s/μ_k.
5. Scale pass: pair compaction, warm-started PCG, Hessian reuse across
   Newton iterations where the constraint set is stable.

**Oracles.** Phase 1: FD gradcheck (the `cloth_diff` recipe); analytic
sphere-drop rest depth → barrier gap ∈ (0, d̂); intersection scan after
every test. Phase 3: the twisted-cloth-cylinder stress test (their
`curriculum` scenes are Apache-2.0 — port the setups, compare
qualitative outcomes). Performance target: real-time for viewer-scale
scenes (10k–50k vertices) — we are NOT chasing their 180M-contact offline
regime; we are chasing their *invariant* at our scale.

**Effort.** Multi-week (largest single item in the roadmap). Phase 1 is a
1–2 session start with standalone value.


### C1. Multi-phase flow with surface tension

**Status.** Landed for level-set / CSF primitives.

**Result.** `molla-fem::multiphase` now provides
`level_set_curvature_2d`, `level_set_smooth_delta`, and
`csf_surface_tension_force_2d`, giving the FEM/NS stack a CPU oracle
for continuum-surface-force surface tension from a signed-distance
interface.

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

**Status.** Landed for 2D Euler flux primitives.

**Result.** `molla-fem` now exposes ideal-gas 2D Euler physical flux,
pressure, Rusanov flux, and HLL flux helpers:
`euler_flux_dot_n_2d`, `euler_pressure_2d`,
`compute_euler_numerical_flux_rusanov_2d`, and
`compute_euler_numerical_flux_hll_2d`. They plug into the existing DG
face quadrature infrastructure the same way the SWE flux helpers do.

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

**Status.** COMPLETE 2026-07-04 — the coupled staggered
displacement↔damage driver (`solve_phase_field_coupled_step`) landed on
top of the AT2 oracle: exact homogeneous identity, 8.8× notch
localization, monotone irreversibility, 13.6% softening over 4 load
steps. The spectral tensile split landed same-day
(default on — compression produces exactly zero damage, full-density
misfire 0.198 demonstrated); 3D tets remain the documented follow-on.
Original oracle notes below.

**Result.** `molla-fem` now exposes `PhaseFieldFractureParams`,
`phase_field_degradation`, `update_phase_field_history`,
`phase_field_fracture_energy_density`,
`assemble_phase_field_fracture_system`, and
`solve_phase_field_fracture_jacobi`. The implementation assembles the
frozen-history AT2 weak form on P1 triangles/tetrahedra with the
existing FEM mass/stiffness kernels, then solves a bounded irreversible
Jacobi update (`d_new >= d_old`, `0 <= d <= 1`) as the CPU correctness
oracle.

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

**Status.** Landed for P1 tetrahedra.

**Result.** `molla-fem` now exposes compressible Neo-Hookean and
Mooney-Rivlin element energies, internal forces, and consistent
finite-difference tangents:
`compute_element_neo_hookean_*` and
`compute_element_mooney_rivlin_*`. The Neo-Hookean identity tangent is
tested against the linear-elasticity stiffness, and the
Mooney-Rivlin tangent is tested for symmetry / zero identity force.

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

**Status.** Landed as reusable law/state helpers.

**Result.** `molla-fem` now exposes
`RateStateFrictionParams`, `RateStateFrictionState`,
`rate_state_friction_coeff`, and `advance_rate_state_friction` for the
Dieterich-Ruina aging law. The helper gives contact solvers a
per-contact evolving friction coefficient without baking the law into a
specific Signorini or particle pipeline.

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

**Status.** Landed with A2.

**Result.** The muscle GPU path now carries
`tendon_stiffness_ratio` through the per-muscle buffers in
`molla-actuators`, the solver binding path, and the WGSL kernel, so
non-default SEE stiffness ratios no longer require CPU fallback.

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

**Status.** Landed for the CPU backward-Euler transient PCG path.

**Result.** `molla-fem` now exposes `KrylovRestartCache` and
`solve_transient_pcg_restarted`. The restarted solve keeps a bounded set
of recent correction directions, computes a tiny minimum-residual
projection in `x0 + span(cache)` before PCG, and stores the accepted
correction for the next time step. This gives the transient steppers a
real recycled-Krylov contract beyond simply warm-starting with the
previous solution.

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
