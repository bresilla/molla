# molla — forward roadmap

_Salvaged and de-duplicated from the old PLAN.md / PROGRESS.md / INSPECTION.md logs (2026-08).
History, day-by-day progress, and self-corrected claims were cut. Only standing decisions and
real remaining work survive here._

For the current build state see [`STATUS.md`](STATUS.md). For the tire program see [`TIRE.md`](TIRE.md).

## Invariants & standing decisions (keep)

- **GPU-first.** Physics lives in WGSL kernels; CPU code is oracle/fallback only (intentional,
  not dead). Copy structure from `xtra/newton` (algorithms) + `xtra/warp` (primitives) —
  vendored read-only, never edit.
- **Never delete a viewer scene.** Broken/partial scenes get fixed in place; flag any edit
  that could look like a deletion.
- **GPU tests need hardware:** `make test-gpu P=<pkg> T=<test> [A=<fn>]` (nixVulkan). Never
  `cargo test --features gpu` (no adapter). ~245 `#[ignore]` tests are GPU-gated by design.
- **Verification ladder:** `make check` (fmt+clippy, viewer-scoped) → `make test` (CPU) →
  `make test-gpu` (per-lane) → `make harden` (full gate). No `make fmt` target exists.
- **Commits:** conventional-commit title only, ≤50 chars, no body/signature/Co-Authored-By.
- **Out of scope — recorded, do NOT implement and do NOT report as "missing":**
  Pacejka/TMeasy/Magic-Formula as a *force source*, measured-tire-data importers,
  calibration/coefficient fitting, MJCF/USD/URDF importers, FFI/Python/Torch/JAX bindings,
  external asset pipelines. Reference curves may exist only as diagnostic shape comparators.
- **Tire frame convention:** y-up, `side = heading × normal`, aligning moment `Mz = +Fy·trail`
  (NOT the ISO z-up `-Fy·t`).
- **Rolling FEM tire:** physics never rotates the mesh (Lagrangian continuum + contact
  explodes — 4 failed attempts + literature confirm); render-time `nodes_rendered()` rotation
  is the sanctioned visual. ALE steady-state transport is the only physics-correct upgrade and
  is a tagged ~3-month research project, **explicitly not backlog**.

## Open roadmap

### Features to continue implementing

- **[high] MPM multi-material per-particle IDs** — the flagship solver gap. Today the rheology
  is global (`MpmConfig.rheology`; GPU uniform packs one `rheology: u32`). Need a per-particle
  `material_id` buffer threaded through P2G/grid/G2P on CPU+GPU, per-branch constitutive
  dispatch in WGSL, one run evolving sand+snow+viscous, mass-conserving. Ref
  `example_mpm_multi_material.py`. Gate: `make test-gpu P=molla-solvers T=mpm_v3_gpu`. (L/MED)
- **[high] Zero-host-readback 40-wheel realtime** — the headline tire goal. Per-tire loop is
  zero-readback; remaining: GPU-resident hub XZ/spin integration (feedback uniforms still carry
  host-tracked hub pose/ω), atlas-path bristle stage, and a benchmark proving 40 live FEM tires
  (10 vehicles × 4 wheels) at realtime. **Investigate first:** the atlas-vs-per-tire
  node-penetration asymmetry — per-tire nodes never cross y=0 (spin block provably inert) while
  atlas nodes transiently penetrate; the ~4.5% hub-parity break traces here, not to spin wiring
  (a 5-line change once the twins agree at the contact boundary). Keep
  `make test-gpu P=molla-kernels T=tire_fem_gpu` at 38/38 — FP-sensitive. (M-L/MED). See [`TIRE.md`](TIRE.md).
- **[med] Static/kinetic friction split** — per-contact `friction_state {Stick,Slip}` with
  μ_s > μ_k on the shipped Signorini+Coulomb active-set solver. Unlocks stick-slip/chatter. (S-M)
- **[med] Promote the ~21 Partial parity scenes** (umbrella, one item each). Recurring missing
  halves: (a) GPU adjoints for the 4 diffsim/optim scenes (DiffsimCloth, DiffsimSoftBody,
  FluidCheckpointOptimization, NavierStokesPerturbationOptimization); (b) couple the reduced
  oracle into the live solver step (NutBoltHydro, CableBundleHysteresis, ClothRollers,
  MpmBeamTwist); (c) true mixed/nonconforming FEM kernels (WarpFemMixedElasticity,
  WarpFemStokesTransfer, WarpFemNonconformingContact, WarpFemConvDiffDg); (d) IK
  obstacle-avoidance as an objective (IkCubeStacking); (e) threaded-SDF narrow phase
  (NutBoltSdf). Scene `notes` in `main.rs` name each gap.
- **[med] Backfill viewer scenes for solver Stages 39-51** — ~20 shipped stages have zero
  scenes; the newest physics is invisible. ~1 scene/day, follow the ScenePhysicsAudit + cpu_proof
  convention.

### Architecture

- **[med] `molla-vehicle` is an island.** (1) Add `pub mod vehicle` to the facade
  `crates/molla/src/lib.rs` (S). (2) Consolidate duplicated `dot`/`cross`/`normalize` over
  `[f64;3]` (tire.rs, granular.rs, real_tire/internal.rs) into one vec3 module or adopt
  molla-math (S-M). (3) Chassis↔molla-sim coupling: `Vehicle::step` is a bespoke 3-DOF Euler
  with no path into a molla-sim scene — write a one-page design spike (couple vs. keep
  standalone) before building (L).
- **[med] Resolve the `molla-kernels → molla-vehicle` inverted dependency.** The tire
  dispatchers pull a high-level crate into a low-level one. Decide: move tire dispatch up, or
  formalize the exception.
- **[low] `molla-vehicle` API surface ≈7× its usage.** ~24 rig/report fns re-exported but used
  only internally; `real_tire/{rig,reports}` ≈13k LOC of diagnostic harness in `src/`;
  `real_tire/tests.rs` is one 9089-line file (184 tests). Demote internals to `pub(crate)` or
  gate behind a `diagnostics` feature; split tests.rs by subsystem.
- **[low] ~~Relabel~~ / wire the two analytic "FEM port" scenes** (WarpFemTaylorGreen,
  WarpFemStreamlines). Relabel **done 2026-08-11** (UI now says "(analytic field)"). Remaining:
  optionally drive them from the real FEM Stokes path so they run a live solve. Never delete.
- **[low, investigate only] non-hull `support()` in gjk/epa** — trace callers; consolidate only
  if just the lint shims use it.

### Tooling / CI

> Note: CI is owned by the maintainer. These are recorded observations, not tasks for an agent
> to action against Actions.

- **[med] No GPU gate in CI.** All parity protection is a human running `make test-gpu`.
  Interim: a documented pre-merge GPU checklist (which lanes for which touched paths). Full:
  self-hosted Vulkan runner (needs an infra decision).
- **[med, read-only] Critic/math audit of solver Stages 39-51** — ~25 stages shipped since the
  last read-only audit (Stage 38); these are exactly the GPU lanes CI cannot protect.
- **[low] Book deploy triggers only on `develop`** — if `main` is the release branch the
  published book can't refresh from it.

## Considered and rejected (don't resurrect without new evidence)

- ALE steady-state transport for true rolling FEM (tagged research, not backlog).
- Further body-canonical / physics-axis refactor stages.
- A general autograd tape as the primary AD strategy — hand-coded closed-form adjoints are the
  deliberate design.
- Broad dead-code deletion sweeps — the dead-code footprint is small and enumerated, not systemic.
