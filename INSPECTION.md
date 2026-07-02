# molla — Project Inspection & Handoff Document

**Audited at commit:** `265007f` (branch `develop`) — all `file:line` references are valid at this commit; re-locate by searching the quoted symbol/string if the file has drifted.
**Date:** 2026-07-02
**Method:** 6 parallel read-only audit agents (plans-vs-code, dead code, core physics crates, vehicle/support crates, viewer, docs/CI/direction), findings vetted against source by the coordinating reviewer. No source code was modified.

This document is self-contained: an executor agent needs no other context to pick up any work item. Read the two sections below (**Ground rules** and **Verified non-issues**) before touching anything.

---

## Execution log (update this section as items land)

Statuses: `TODO` (default — anything not listed here), `DONE`, `PARTIAL`, `BLOCKED`. Every entry lists verification actually run. Changes are in the working tree on `develop` unless a commit hash is noted.

| Item | Status | Date | What was done |
|------|--------|------|---------------|
| T2 | DONE | 2026-07-02 | Removed `git add --all && git commit` from the Makefile `docs:` target. `make docs` now only builds the book. |
| B1 | PARTIAL (interim fix) | 2026-07-02 | `XpbdSolver::step` now returns `Error::Solver` for GPU backend + `substeps > 1` instead of silently diverging (`crates/molla-solvers/src/xpbd.rs`, guard before the multi-substep path; GPU branch comment updated + `debug_assert!(q_at_contact.is_none())`). **Remaining:** the full fix — upload `q_at_contact` as a buffer and read anchors from it in the WGSL kernel, then remove the guard and add a substeps>1 parity test. No existing caller used GPU+substeps>1 (all GPU tests run external substep loops). Verified: `cargo test -p molla-solvers` clean, `make test-gpu P=molla-solvers T=xpbd_gpu_parity` 1/1 on hardware. |
| B3 | DONE (re-scoped) | 2026-07-02 | Investigation showed `HydroelasticPolicy` has **no consumer** — `compute_hydroelastic_contact` doesn't take a policy, so the "empty manifold" behavior was documented-future, not live code. Fixed the contract doc on `SoftOnSoft` (`crates/molla-geometry/src/hydroelastic.rs:131`) to require a typed `Error` from any future policy-accepting entry point. No runtime change needed. |
| DOC1 | DONE | 2026-07-02 | Fixed all four stale doc sites in `crates/molla-kernels/src/tire_fem.rs`: `hub_omega`/`drive_torque_n_m`/`brake_torque_n_m` field docs (kernel reads spin; hub torque integration host-side by design), matrix-free-Newton enum lead doc (was "GPU dispatch is not live"), matrix-free contract doc (was "intentionally not implemented yet"), torque-status enum doc (was "does not yet integrate them"). |
| D1 | DONE | 2026-07-02 | Deleted dead `color_particles` (~78 lines) from `crates/molla-sim/src/builder.rs`; rewrote the doc-comment on `color_particles_with_tets_and_bending` to be standalone (it referenced the deleted fn — and a never-existing `color_particles_with_tets`). |
| D2 | DONE | 2026-07-02 | Deleted empty `crates/molla/src/utils/` and the `pub mod utils;` line in `crates/molla/src/lib.rs`. |

**Verification run for the whole Wave 1 batch:** `cargo fmt --all -- --check` clean; `cargo check --workspace --all-targets` clean; `cargo clippy -p molla-solvers -p molla-sim -p molla-geometry -p molla-kernels -p molla --all-targets --all-features` — zero warnings in touched files; `cargo test` on the five touched crates 0 failures; viewer suite 82/82; `make test-gpu P=molla-solvers T=xpbd_gpu_parity` 1/1.

**Discovery for T1:** `make harden` currently fails on *pre-existing* clippy debt in `molla-vehicle` (`real_tire/reports/physical_shape.rs`, `real_tire/fem/builders.rs`, `real_tire/fem/newton_builders.rs`, `real_tire/fem/solver.rs`), `molla-math` (`dense3.rs`, `iterative.rs`, `robust_loss.rs`, `sparse.rs`, `spatial.rs`), and `molla-kernels` (`tire_fem.rs:1189` needless-range-loop, `tire_fem.rs:2481` manual-repeat-n) — warning classes: empty-line-after-doc-comment, loop-index, too-many-args, clone-on-Copy, repeat-take. Adding the clippy CI step (T1) requires clearing that debt first — new sub-task **T1a: fix pre-existing clippy warnings in molla-vehicle / molla-math / molla-kernels** (all style-level; none were introduced by the work logged here).

**T1a scale update (2026-07-02):** with `--all-features` (which is what `make harden` runs — it exposes all the GPU-gated code the default build never lints), the debt is ~500+ warnings across the whole workspace, top files `molla-kernels/src/tire_fem.rs` (32), `bin/molla_viewer/src/main.rs` (32), `molla-fem/src/stokes.rs` (18). Breakdown: 89 collapsible-if, 90 doc-list indentation, ~80 too-many-arguments, 48 no-effect operations, 23 type-complexity, plus a long tail. **Recommended approach for the executor:** (1) do it as its own isolated commit(s), not mixed with feature work; (2) `cargo clippy --fix --allow-dirty --workspace --all-targets --all-features` for the MachineApplicable classes, then full CPU suite + the tire/fem GPU lanes; (3) for the design-level lints (`too_many_arguments`, `type_complexity`), prefer workspace-level `allow` in `[workspace.lints.clippy]` (physics kernels legitimately take many args — matches the existing `module_name_repetitions = "allow"` precedent) over 100+ signature refactors. Effort revised: **L**, not S.

### Batch 2 (2026-07-02, same session)

| Item | Status | Date | What was done |
|------|--------|------|---------------|
| B2 | PARTIAL (interim fix) | 2026-07-02 | `WaveStepperGpu` and `TransientStepperGpu` construction now returns `Error::Gpu` for `DirichletVector` with >1 component instead of a release-silent `debug_assert` + truncation (`crates/molla-fem/src/solver.rs`, both BC upload loops). **Remaining:** option (a) — real vector-aware pin plumbing (thread `dim` through `apply_dirichlet` + pin bitset upload) with a CPU-vs-GPU parity test. Verified: CPU suites clean; `make test-gpu P=molla-fem T=fem_wave_gpu` 2/2 and `T=fem_transient_gpu` 2/2 on hardware. |
| T4 | DONE | 2026-07-02 | Removed the stale `MOLLA_NEWTON_PYTHON` / `Makefile::test-oracle` failure bucket from `.github/workflows/rust.yml` (the Newton oracle is pure Rust); renumbered remaining buckets, added a note pointing at `book/src/testing/strategy.md`. Reason-strings for bare `#[ignore]`s NOT done (mechanical, ~230 sites — separate pass). |
| D3 | DONE | 2026-07-02 | Deleted the `_suppress_contacts` shim and the `Contacts` import from `crates/molla-geometry/src/gjk_gpu.rs`. Verified: `make test-gpu P=molla-geometry T=gjk_gpu` 4/4 on hardware. |
| D4 | DONE | 2026-07-02 | Removed the never-constructed `TireFemMatrixFreeNewtonGpuStatus::WgslSourceGuardOnly` variant (`crates/molla-kernels/src/tire_fem.rs`); the builder only ever emitted `Live`/`CpuOnlyDebug`. Contract test suite passed. |
| A6.1 | DONE | 2026-07-02 | molla-vehicle wired into the facade: new `crates/molla/src/vehicle.rs` (`pub use molla_vehicle::*`), `pub mod vehicle;` in `crates/molla/src/lib.rs`, dep added to `crates/molla/Cargo.toml`. Viewer still uses its direct dep (fine); new consumers can now use `molla::vehicle::…`. |
| A8 | DONE | 2026-07-02 | Flattened `camera_post` re-exports in `crates/molla-sensors/src/lib.rs` (13 fns), matching the lidar/pointcloud sibling convention. |
| DOC5 | DONE | 2026-07-02 | Added a "CURRENT STATE" banner to the top of `PROGRESS.md` (A complete, B dropped, C G1–G4 closed, baseline 223/0, remaining open work → F2/F3) and a supersession note to the top of `PLAN_REALTIRE.md`. The stale tails were NOT rewritten — the banners direct readers past them (full prune remains optional). |
| DOC2 | PARTIAL | 2026-07-02 | Fixed the two `make fmt` / `make test-vbd-extended-gpu` references in `PLAN.md` (targets no longer exist). **Remaining:** the full rewrite of `book/src/architecture/build.md` against the current Makefile. |
| DOC6 | PARTIAL | 2026-07-02 | `book/src/SUMMARY.md` scene catalog now says #1-123; `book/src/roadmap/stages.md` scene count corrected to 131. **Remaining:** vehicle test story in `book/src/testing/strategy.md`, remaining stale 220/2 test-count references in book prose. |

**Verification for batch 2:** `cargo fmt --all -- --check` clean; `cargo check --workspace --all-targets` clean; clippy all-features on touched crates — zero warnings in touched files (see T1a note for pre-existing debt); full CPU test run for molla-fem/-geometry/-sensors/-kernels/-molla: all suites ok, exit 0; hardware GPU lanes: `fem_wave_gpu` 2/2, `fem_transient_gpu` 2/2, `gjk_gpu` 4/4, `xpbd_gpu_parity` 1/1.

### Batch 3 (2026-07-02, same session)

| Item | Status | Date | What was done |
|------|--------|------|---------------|
| F3 | DONE (found already implemented) | 2026-07-02 | Audit error: `TetraTireFem::nodes_rendered()` + scene-123 wireframe + rigid-rotation test already landed in `265007f`; the audit trusted `PROGRESS.md`'s "next step" framing. Verified the implementation matches the `body_to_world` rotation convention and the test passes. Corrected the F3 section above, the `PROGRESS.md` banner, and this log. |
| T1a | TODO (scoped) | 2026-07-02 | Scale measured: ~500+ warnings under `--all-features` (what `make harden` runs). Deliberately NOT auto-fixed in this session to keep the working tree reviewable — see the "T1a scale update" note above for the recommended isolated-commit approach. |
| F8 | DONE | 2026-07-02 | Transmission auto-shift: added `upshift_rpm`/`downshift_rpm` (5200/1800, hysteresis-band rationale documented) + `Transmission::auto_shift` (one gear per tick) + `Powertrain::engine_rpm` helper; `Vehicle::apply_driver` now takes `&mut Powertrain` and shifts before computing torque. All 10 call sites updated (crate tests, sysid, granular, deformable, viewer scenes 103/110 — the viewer scenes clone per frame and now **write the powertrain back** so the gear persists). Two new tests: `transmission_auto_shifts_with_hysteresis` (incl. no-hunting both directions, top/bottom bounds) and `apply_driver_upshifts_at_high_rpm`. **molla-vehicle baseline is now 225/0** (was 223/0). Verified: full molla-vehicle suite 225/0, workspace check clean, viewer suite 82/82, clippy clean in changed files (granular/103/110 hits pre-exist on HEAD — confirmed via stash). |

### Batch 5 (2026-07-02, same session — committed separately)

| Item | Status | Date | What was done |
|------|--------|------|---------------|
| B1 | DONE (full fix) | 2026-07-02 | Contact-time pose anchors now work on the GPU across substeps. `xpbd.wgsl`: new read-only `@group(1) @binding(11) body_q_contact`; `project_position` and `project_velocity` anchor computations (`q_self_in` / `q_other_in_*`) read it instead of `body_q_in` (whose integrate/velocity-recovery roles are untouched). `molla-kernels/src/xpbd.rs`: `body_q_contact` buffer + `capture_contact_pose()` (device-to-device snapshot of `state.body_q`) + `dispatch(..., use_contact_pose)` — `false` binds `state_in.body_q` at binding 11, bit-identical to pre-fix single-substep behaviour (regression-pinned by `box_stack_cpu_gpu_match` staying green). `molla-solvers/src/xpbd.rs`: `AnchorPose` enum (`LiveInput`/`Host`/`DeviceCaptured`) replaces the `Option<&[Transform]>` threading; new `step_multi_gpu` runs the substep loop fully device-resident (lazy `gpu_scratch` state, `body_f` seeding, ping-pong, device-copy finalize incl. particles); the interim `Error::Solver` guard is gone. New hardware test `sliding_box_multi_substep_cpu_gpu_match` (continuous-contact friction slide — CPU internal-substep is the exact semantic twin there and analytically correct: 0.517 m vs v²/2μg = 0.51 m). Verified: `xpbd_gpu_parity` 2/2, plus `contact_friction_modes_gpu` 2/2, `xpbd_equality_gpu` 7/7, `xpbd_particles_gpu` 6/6, `vbd_cloth_gpu` 4/4 on hardware; CPU solvers suite 0 failures; workspace check + fmt clean; both xpbd.rs clippy hits pre-exist on HEAD. |
| B1b (NEW finding — documented, not fixed) | OPEN | 2026-07-02 | Internal multi-substepping **amplifies impacts on BOTH backends** (pre-existing; not introduced by B1): position projection pops the full outer-step penetration in one substep and velocity recovery divides that pop by `dt/n`, so a landing body gains n× separation velocity. Measured repro: restitution-0 box dropped 0.1 m with `substeps=4` bounces persistently on GPU (~0.2 m period ~0.5 s) and worse on CPU (>1.3 m, energy-gaining); a 4-box stack drop reaches 13 m by step 800. `with_substeps` docs now state this precisely. Candidate fixes (maintainer decision needed): exclude projection displacement from recovered velocity for the impact substep, or refresh contact depths per substep. Until then: external substepping for impact-heavy scenes (the already-documented pattern). |
| B1c (NEW bug — FIXED) | DONE | 2026-07-02 | GPU sliding friction collapsed in steady contact: the WGSL Coulomb cap was `μ·|λ_n|` but the CPU caps at `μ·max(|λ_n|, m·g·dt)` (support-impulse floor added during hardening, never mirrored to WGSL). With λ_n→0 in resting/sliding contact the GPU box glided 2.7× too far (1.38 m vs analytic 0.51 m). Floor added to `project_velocity` in `xpbd.wgsl` (per-body world gravity, static-body guarded). Verified by the sliding test (post-fix divergence 5 cm ≈ established f32 budget) and all four friction/xpbd GPU lanes green. |

| F7 | DONE | 2026-07-02 | VBD GPU multi-substep implemented by reusing the B1 pattern: the ~180-line GPU arm of `VbdSolver::step` is extracted into `gpu_substep()` (behavior-identical for `substeps == 1`, still advancing the full `dt`), and a new `gpu_multi_substep()` ping-pongs it through a lazily-built device-resident scratch state — body poses seeded from the outer input (VBD passes bodies through), particle q/qd (+ tet plasticity buffers) device-copied out when the last substep lands in the scratch, `mark_device_dirty` preserved. The `Error::Solver("VBD GPU multi-substep is not implemented…")` guard is gone. Note: the extraction also fixed a latent inconsistency — the VT-contact pipeline previously received `dt_sub` while the main kernel got full `dt` (harmless when substeps was forced to 1; both now get the substep dt). New hardware test `vbd_gpu_multi_substep_matches_cpu` (draping cloth, 60 outer × 4 substeps, CPU internal-substep is the exact semantic twin; max |Δp| gate 5 cm, measured well under). Verified: all five VBD GPU lanes green — `vbd_cloth_gpu` 5/5 (incl. new), `vbd_extended_gpu` 3/3, `vbd_tets_gpu` 3/3, `vbd_tets_v3_gpu` 3/3, `vbd_tets_v4_gpu` 3/3 — plus CPU vbd suites 11/11, workspace check + fmt clean. |

| DOC2 | DONE | 2026-07-02 | `book/src/architecture/build.md` fully rewritten against the current Makefile: real target table (viewer-scoped build/test/check, `test-all`/`check-all`, `harden`, generic `test-gpu P= T= A= RELEASE=`), explicit "there is no `make fmt`" note, the ~40 stale per-family `make test-*-gpu` targets replaced by the generic-lane pattern + worked examples + `ls crates/*/tests \| grep gpu` discovery, stale `make run RELEASE=1 ARGS=…` / `run-profile` / `bench-cpu/-gpu/-featherstone` claims removed. Verified: `mdbook build` clean. |
| DOC3 | DONE | 2026-07-02 | molla-vehicle is now in the book: `architecture/crates.md` prose + diagram + extraction list + workspace-members block all include it; new chapter `book/src/vehicle/overview.md` (subsystem table incl. the F8 auto-shifter and F9 suspension caveat, GPU-kernel relationship + parity-ledger pointer, viewer-scene map, testing/calibration stance) wired into `SUMMARY.md` under a new "Vehicle" section. Verified: `mdbook build` clean. |

| DOC4 | DONE | 2026-07-02 | `book/src/roadmap/open.md` now opens with a "Current open set (2026-07-02)" banner table (A1, X1, X2 — the only genuinely open items; A2 landed this session) and notes the landed entries below are decision records, not backlog; snapshot line updated to point tire work at `PROGRESS.md`/`INSPECTION.md`. Full prune-to-changelog NOT done (the inline Status blocks + banner make the state unambiguous at a fraction of the churn). |
| DOC6 | DONE | 2026-07-02 | Remainder closed: `book/src/testing/strategy.md` gains a "vehicle crate's embedded harness" section (rig/reports harness, 225/0 baseline, parity-ledger pointer); the two *actionable* stale `220/2 baseline must hold` commands in `PROGRESS.md`'s resume tail corrected in place (historical log mentions left as history, covered by the top banner). |
| F9 | DONE (decision: label-only) | 2026-07-02 | `SuspensionKind` is now **documented as a descriptive label** (module doc records the scope decision + rationale): kind-driven camber/toe kinematics requires a chassis roll DOF and a camber input on the vehicle-path tire interface, both deliberately excluded by the 3-DOF `Vehicle` (`vehicle.rs` force comment). No physics change; enum/field docs + book updated. Re-open only alongside a fuller-chassis effort (A6.3/N6 design spike is the umbrella). |

| T1a | **DONE — zero warnings** (was "mostly done"; the remaining 66 were finished by hand the same session) | 2026-07-02 | Clippy sweep executed per the logged recipe, as its own commit: (1) justified workspace allows in `[workspace.lints.clippy]` — `too_many_arguments`, `type_complexity` (physics-kernel signatures), `needless_range_loop` (indexed numeric loops are the transliteration idiom), `manual_clamp` (the clamp() rewrite is not NaN-semantics-preserving); (2) two `cargo clippy --fix` rounds (122 files, net −221 lines — collapsible-ifs, no-effect ops, same-type casts, assign-ops, etc.); (3) 90 doc-list-indentation suggestions script-applied from JSON diagnostics; (4) 25 more single-span safe-class suggestions script-applied (manual-memcpy, reserve-after-init, byte groupings, map-identity). **Remaining: 66 emissions ≈ 45 unique sites**, all enumerable via `cargo clippy --workspace --all-targets --all-features`, dominated by `field_reassign_with_default` + `needless_update` (mostly in GPU test files: `tire_fem_gpu.rs` ×6, `contact_reduction_gpu.rs` ×2, `mesh_utils*.rs` ×4, `robust_loss.rs` ×2, viewer `main.rs` ×5, `eddy_current.rs` ×3, plus onesies: `double_must_use` ×5, `neg_cmp_op_on_partial_ord` ×3, `large_enum_variant` ×2, const-assertions ×2, empty-line-after-doc ×2). These need hand edits (their clippy suggestions carry no machine spans) — finish them before flipping T1's CI clippy step to `-D warnings`. Verified after the sweep: full workspace CPU suite (see below), GPU lanes `xpbd_gpu_parity` 2/2, `joint_muscle_gpu` 5/5, `vbd_cloth_gpu` 5/5, `fem_transient_gpu` 3/3, `fem_wave_gpu` 3/3, `gjk_gpu` 4/4, `tire_fem_gpu` 38/38. |

| T1a (completion) | DONE | 2026-07-02 | The remaining ~45 sites hand-fixed: 14 redundant `..Default::default()` removals; 14 `field_reassign_with_default` → struct-literal rewrites (incl. the six `BrushWheelOutput` pairs in `tire_fem_gpu.rs`); 5 `#[must_use]`-on-`Result` attribute removals (eddy_current ×3, contact_extras ×2); 2 justified `large_enum_variant` allows (`ControllerStage`, viewer `SolverBackend` — one instance each, boxing buys nothing); 1 NaN-guard `neg_cmp_op_on_partial_ord` allow (`see_newton_diag` — `!(x > 0.0)` intentionally rejects NaN); 2 const-assertions in `contact_reduction_gpu.rs`; 1 capacity-preserving vec init in `featherstone.rs`; and ~45 lines of **orphaned doc-comment shards deleted from viewer `main.rs`** (fragments left behind when scene builders moved to `examples/` — two blocks near `build_heightfield_mesh` and `draw_grab_gizmo`, the latter including a stray `#[allow(too_many_arguments)]` on a 3-arg fn; `draw_grab_gizmo` got a real one-line doc). **Result: `cargo clippy --workspace --all-targets --all-features` = 0 warnings** (from 784). Verified: molla-vehicle 225/0, viewer 82/82, robust_loss 9/9, geometry suite clean, torque-contract 4/4, GPU lanes `tire_fem_gpu` 38/38, `tire_vehicle_gpu` 21/21, `contact_reduction_gpu` 3/3, `hydroelastic_gpu` 5/5, `sparse_grid` 2/2, `joint_muscle_gpu` 5/5. |
| T1 | DONE | 2026-07-02 | CI now gates on lint + the CPU-only lane: `rust.yml` `build-test` gained `cargo clippy --workspace --all-targets --all-features -- -D warnings` plus `cargo check`/`cargo test --workspace --no-default-features` steps (both verified passing locally before gating — the no-default-features test lane runs clean, exit 0). The CI header comment documents the new matrix. T3 (GPU runner) remains the only un-gated axis. |

### Batch 6 (2026-07-02, same session)

| Item | Status | Date | What was done |
|------|--------|------|---------------|
| F2 | MOSTLY DONE (zero-readback loop landed; perf proof + hub-resident follow-ups remain) | 2026-07-02 | The per-tire zero-host-readback substep loop is real: new `TireFemResidentContact` state (ping-pong bristle buffers, full-node vel/force buffers, flat-ground terrain stand-ins) + `TireFemContactFeedbackKernel::{create_resident, dispatch_resident}` in `molla-kernels/src/tire_fem.rs`. `dispatch_resident` binds the persistent mesh's just-written pos/vel slot (same parity rule as the hub kernels), runs the existing feedback shader unchanged, device-copies the brush-updated velocities back into the mesh slot, and leaves per-node force rows on-device for the already-landed `step_hub_persistent_with_tread_forces`. Substep chain `step_persistent → dispatch_resident → step_hub_persistent_with_tread_forces` crosses the host zero times. New hardware test `tire_fem_resident_contact_loop_matches_blocking_chain` runs the G4 rolling scenario through both chains on twin meshes: **bit-identical** — hub d=(0,0) at every checkpoint, final max node |Δp| = 0, max bristle |Δ| = 0. Full tire lane 39/39; workspace clippy still 0 warnings. **Remaining for full F2 closure:** (a) GPU-resident hub XZ/spin integration — the per-substep `TireFemContactFeedbackParams` still carry host-tracked hub pose/ω/load (O(1) uniform uploads, no readback, but hub truth lives host-side; the parity-ledger already scopes this refinement); (b) wire the resident stage into the fused-atlas path for many-tire batches; (c) ~~the actual 40-wheel realtime perf measurement~~ **MEASURED 2026-07-02**: new `tire_fem_atlas_forty_wheel_walltime` probe (release build, forced-completion timing) — 40 wheels / 11,520 nodes, 7,992 substeps/s vs the 8,000/s budget at dt=1/8000 ⇒ **realtime factor 1.00x on this hardware** for the fused-atlas *node-FEM* stage (bristle/hub stages not yet in the atlas measurement; the near-exact 1.00x suggests per-`queue.submit` overhead ~125 µs dominates — batching multiple substeps per command encoder is the likely headroom). Loose 0.05x catastrophic-floor gate only; the printed factor is the record. (d) heightfield/SDF terrain variant of `dispatch_resident` (v1 is flat-ground; the blocking `dispatch_blocking_with_gpu_terrain` shows the pre-uploaded-terrain pattern to copy). |

**API change note (F8):** `Vehicle::apply_driver(&Powertrain, …)` → `apply_driver(&mut Powertrain, …)`. Any out-of-tree caller must pass `&mut` and persist the powertrain across frames. Manual gear selection via `DriverInput` was NOT added (auto-only); add `DriverInput::gear_request` if a scene ever needs it.

### Batch 4 (2026-07-02, same session)

| Item | Status | Date | What was done |
|------|--------|------|---------------|
| F5 | DONE (was implemented, now validated) | 2026-07-02 | Investigation resolved the A2/C6 roadmap contradiction: the per-muscle implementation **already shipped** with the C6 push (`muscle.wgsl` bindings 7/9 read `muscle_pennation_angle[m_id]`/`muscle_tendon_stiffness_ratio[m_id]`; `Model` uploads both arrays) — what was missing was validation, since the only parity tests used zero pennation / default ratio. Added `per_muscle_pennation_and_stiffness_cpu_gpu_parity` to `crates/molla-solvers/tests/joint_muscle_gpu.rs`: two muscles with different pennation (0 / 0.35 rad) and SEE stiffness (45 / 20) through the Newton SEE path, plus a homogeneous-defaults CPU control run that fails the test if the parameters don't measurably matter (anti-vacuous guard). Also fixed the stale "GPU's pennation-disabled assumption" comment in the SEE test. `book/src/roadmap/open.md` A2 now carries a Status: Landed block. Verified: new test passes on hardware first run (max rel < 5e-5), full `joint_muscle_gpu` lane 5/5. |
| B2 | DONE (full fix) | 2026-07-02 | Vector-aware Dirichlet BC plumbing landed in both GPU steppers (`crates/molla-fem/src/solver.rs`): `dim` is inferred as `nrows / mesh.vert_count()` (no API change — same layout contract `FemGpuData::upload` already documents; non-multiple row counts now return a typed error), the matrix BC-reduction routes through the vector-aware `apply_bcs(..., &mesh.vertices, dim)`, and the pin bitset mirrors its indexing exactly (scalar `Dirichlet` → component 0 at `vertex·dim`; `DirichletVector` → every component, with a hard error if `values.len() != dim`). Side effect: `NeumannTraction` no longer panics on the empty-vertices slice in these constructors (it now integrates via `apply_bcs` pass 1 — untested, noted for review). Two new hardware parity tests using a decoupled-interleave oracle (scalar CPU solver run once per component is *exact* for the Kronecker-interleaved system): `gpu_transient_vector_dirichlet_matches_componentwise_cpu` (dim=3, per-component pin values 0/0.5/1) and `gpu_wave_vector_dirichlet_matches_componentwise_cpu` (dim=2, Newmark), both also asserting each pinned component holds its own value (the truncation bug's smoking gun). Verified: `make test-gpu P=molla-fem T=fem_transient_gpu` 3/3 and `T=fem_wave_gpu` 3/3 on hardware; workspace check + fmt + clippy clean in touched files. |
| F10 | DONE | 2026-07-02 | New viewer scene **124 `Scene::VehicleSysid`** (`bin/molla_viewer/src/examples/124_vehicle_sysid.rs`) wires the previously-unreachable `molla_vehicle::sysid` showcase: plants a hidden μ=0.75 trajectory, unrolls `identify_friction`'s Gauss-Newton loop one iteration per frame (finite-difference sensitivity through full 3000-step rollouts), drives a live candidate car on the current estimate next to a green wireframe ghost on the truth, and plots the μ-iterate strip chart vs the truth line in gizmos. Registered at all 9 main.rs touch points (enum, ALL, label, description, audit `CpuOnly` with cpu_proof pointing at the sysid crate tests, build dispatch reusing `build_vehicle_model`, step-skip list, resource, update system). Also: sysid module doc no longer says the scene is "the remaining UI-wiring step" (it referenced a nonexistent `PLAN_ENGINE.md`); book scene counts bumped to #1-124/132. Verified: viewer suite 82/82 (includes Scene::ALL consistency tests), workspace check clean, fmt clean. NOT verified: interactive visual run of the scene — recommend a human eyeball of scene 124 via `make run` next time the viewer is opened. |

---

## Ground rules for any agent working on this repo

1. **GPU-first.** Physics is implemented in WGSL kernels, not Rust host loops. Copy structure from `xtra/newton` (algorithms) and `xtra/warp` (compute primitives) — these are vendored read-only references, never edit them. CPU implementations exist only as test oracles and fallbacks; they are intentional, not dead code.
2. **Never delete a viewer scene.** Broken or partial scenes get fixed in place. Any edit that could look like a scene deletion must be flagged explicitly.
3. **GPU tests need hardware.** Run them via `make test-gpu P=<pkg> T=<test> [A=<test_fn>]` (goes through nixVulkan). Never `cargo test --features gpu` directly — there is no adapter and it will fail spuriously. The 245 `#[ignore]`d tests in the workspace are all GPU-adapter-gated; that is by design.
4. **Verification commands:** `make check` (fmt + clippy, scoped to viewer pkg), `make test` (CPU tests), `make test-gpu` (per-test GPU lane), `make harden` (the real full gate: fmt, no-default-features build+test, `clippy --all-features -D warnings`, all-feature tests). There is **no `make fmt` target** despite docs claiming one — use `cargo fmt` or `make harden`.
5. **Do NOT run `make docs`** until work item T2 lands — it currently auto-commits your entire working tree (see T2).
6. **Explicitly out of scope for this project** (recorded decisions — do not implement, do not report as missing): Pacejka/TMeasy/Magic-Formula force sources, measured-tire-data importers, calibration/coefficient-fitting workflows, MJCF/USD/URDF importers, FFI/Python/Torch/JAX bindings, external asset pipelines. Reference curves may exist only as diagnostic shape comparators.
7. **Git commits:** conventional-commit title only (`type(scope): message ≤50 chars`), no body, no signatures, no Co-Authored-By.
8. Workspace facts: Rust edition 2024, 16 library crates under `crates/` + one facade crate `crates/molla` + single executable `bin/molla_viewer`. ~306k LOC Rust, 96 WGSL shaders (all referenced — zero dead shaders). `PROJECT` at repo root is a 12-byte name/version stub file consumed by the Makefile, **not** a directory.

---

## Verified non-issues — do NOT "fix" or re-flag these

These were investigated and confirmed intentional. Future audit passes should skip them.

- **CPU oracles/fallbacks everywhere** (`dahl_oracle`, `sph_oracle`, `mpm_oracle`, XPBD CPU backend, MPM legacy global-atomic path, `AdjointBackend::GpuFallbackCpu`) — the GPU-first design keeps CPU twins as parity oracles.
- **245 bare `#[ignore]` tests** — GPU-adapter-gated, run via `make test-gpu`. (Adding reason strings is nice-to-have, see T4.)
- **wgpu RAII keep-alive fields under `#[allow(dead_code)]`** — fields hold GPU buffers alive that the borrow checker can't see used (bind groups hold Arc-cloned refs). Deleting them releases live GPU memory. Sites include: `molla-kernels/src/optim_adam.rs:46`, `optim_sgd.rs:35`, `sparse_gpu.rs:282`, `soft_vt_contact.rs:103`, `tire_fem.rs:1326`, `molla-fem/src/apic_coupling_gpu.rs:69`, and ~40 sites in `molla-fem/src/gpu_kernels/fem_cg.rs`.
- **Lint-suppression shims** (`_suppress_unused_support_warning`, `_touch_error`, `_WG_DOC`, `_arc_keep_alive`, `_keep_particle_alive`, `_keep_warnings_quiet`, `_silence_unused_imports`) — deliberate; several protect items consumed only by GPU-gated builds.
- **`SphSimulation` / `MpmSimulation`** — alive; constructed by GPU test files and re-exported.
- **`visual_patch_factor`** (`bin/molla_viewer/src/examples/121_tire_traction.rs:402`) — visualization-only, guarded, has a regression test (`main.rs:9493`). The real physics goes through `molla-vehicle`.
- **Dropped Stream B** (tire mesh-rotation refactor, `PROGRESS.md:122`) — an explicit architecture decision with recorded rationale. Its sanctioned replacement is work item F3 (render-time rotation).
- **`apply_rigid_rotation_about_hub`** (`molla-vehicle/src/real_tire/fem/solver.rs:1385`) — currently uncalled but explicitly parked for "PLAN_ROLL Stage C proper". Keep.
- **`BodyVisual::GaussianSplat`** (`bin/molla_viewer/src/main.rs:1255`) — never-constructed variant parked for the Stage 30 splat-rendering follow-up. Keep.
- **`cargo check --workspace --all-targets`** passes with **zero warnings**. All 96 WGSL shaders referenced, no orphan crates, all 131 viewer example files wired. The dead-code problem in this repo is *small and enumerated* (section D below) — do not launch broad deletion sweeps.
- The `molla-autograd` tape (`tape.rs`, 5 op variants, host-only) is deliberately minimal; most AD is hand-coded closed-form adjoints. Not a stalled feature.

---

## Executive summary

The codebase is in better mechanical health than expected (clean build, no orphan modules, honest per-scene audit annotations) but has three systemic problems:

1. **A wide "90% done" frontier.** All 15 viewer-parity scenes from `PLAN.md` exist but every one is `status: Partial` — each visualizes a reduced/proxy model instead of running the referenced physics. Several library features are parameter-plumbed but not implemented (GPU multi-substep XPBD/VBD, vector Dirichlet BCs on GPU, multi-DoF CRBA/RNEA, gearbox shifting, suspension kinematics).
2. **Three latent correctness traps** where unfinished paths fail *silently* instead of loudly (B1–B3 below).
3. **Severe documentation drift.** The book documents a Makefile that no longer exists, omits the largest active crate (molla-vehicle), and the roadmap/plan files contradict each other about what is open vs. done. CI never runs clippy and has no GPU gate.

Recommended execution order: **B items first** (small, high-value), then **T2** (removes a footgun), then F/A/D/DOC items in any order respecting the dependency notes inside each item.

---

# B — Correctness bugs (fix first)

### B1. XPBD GPU multi-substep ignores the contact-time pose anchor — silent CPU/GPU divergence
- **Where:** `crates/molla-solvers/src/xpbd.rs:276` — GPU backend does `let _ = q_at_contact;` with the comment "contact-relative pose override is not implemented yet — fall back to using state_in's body_q (works for substeps == 1)". The CPU path (`xpbd.rs:206-238`) captures `q_at_contact` once per outer step and threads it into every substep so each contact's body-frame anchor `p_a_local` stays fixed (Müller 2020 §3.2).
- **Bug:** With `substeps > 1` on the GPU backend, contact anchors are recomputed from the drifting current pose each substep. Friction/penetration constraints diverge from the CPU reference. No error is raised.
- **Fix:** Upload `q_at_contact` as a buffer, have the GPU constraint kernel read anchors from it instead of the live pose. Mirror the CPU threading exactly.
- **Tests:** Add a `substeps > 1` CPU-vs-GPU parity test (pattern: existing `xpbd*_gpu` tests in `crates/molla-solvers/tests/`). Run via `make test-gpu P=molla-solvers T=<file>`.
- **Interim mitigation:** ✅ DONE 2026-07-02 (see execution log) — `step()` now returns `Error::Solver` for GPU + `substeps > 1`.
- **Implementation map for the full fix (derived 2026-07-02 by reading the kernel — start here):**
  1. `crates/molla-kernels/shaders/xpbd.wgsl`: the anchor freeze lives in `project_position` — `p_self_local` is computed from `q_self_in = body_q_in[tid]` and `q_other_in_pos/rot = body_q_in[other]` (comment "body-local contact points (frozen at narrow-phase)", ~line 256). `body_q_in` plays TWO roles: integration input (`integrate` stage, ~lines 152-190 — must stay the live input pose) and anchor reference (constraint stage — must become the contact-time pose). Add a new `@group(1) @binding(8) var<storage, read> body_q_contact: array<Transform>` and switch ONLY the constraint-stage `q_self_in`/`q_other_in_*` reads to it.
  2. `crates/molla-kernels/src/xpbd.rs`: extend `layout_state` + both ping-pong bind groups with binding 8; `dispatch()` gains a `q_contact: &wgpu::Buffer` param — callers with `substeps == 1` pass `state_in.body_q`'s device buffer (bit-identical behavior, existing lanes stay green).
  3. `crates/molla-solvers/src/xpbd.rs`: in the GPU multi-substep path, do NOT use `state_in.body_q.host()` (device-resident states make that a readback/staleness trap) — instead `copy_buffer_to_buffer` the device `body_q` into a persistent `q_contact` scratch buffer once at the outer step, then run the substep loop device-resident: the existing host `scratch = state_in.clone_host()` ping-pong is CPU-only in practice, so the GPU loop needs a device scratch `State` (upload once at solver build, reuse across steps — same pattern as F7's VBD scratch). Then delete the interim guard.
  4. Parity test: `xpbd_gpu_parity`-style box-stack scene, `with_substeps(4)`, CPU vs GPU trajectories; also assert the GPU `substeps=1` result is unchanged vs before (regression pin for step 2).
  - Note the overlap with **F7** (VBD GPU multi-substep needs the same device-resident scratch pattern) — implement the scratch-state helper once, use it in both.
- **Effort:** M. **Risk:** MED (hot contact dispatch path + buffer layout change).

### B2. Vector Dirichlet BCs silently truncated to 1 component on GPU transient/wave steppers in release builds
- **Where:** `crates/molla-fem/src/solver.rs:782-792` (`WaveStepperGpu`) and `solver.rs:1112-1124` (`TransientStepperGpu`) — `DirichletVector { values }` with `values.len() > 1` is guarded only by `debug_assert_eq!`; release builds pin one DOF using `values[0]` and drop the rest. The CPU path is vector-aware via `apply_bcs(dim)`.
- **Fix (two acceptable outcomes):** (a) implement per-component pin plumbing — thread `dim` through the pin bitset / pinned-value upload so each component row is pinned correctly, matching the CPU reduction; or (b) minimum viable: replace the `debug_assert` with a hard `return Err(...)` so release builds fail loudly.
- **Tests:** For (a), a CPU-vs-GPU parity test with a 3-component displacement BC on the wave stepper. For (b), a unit test asserting the error.
- **Effort:** M for (a), S for (b). **Risk:** MED for (a) — BC indexing is shared with the matvec.

### B3. `HydroelasticPolicy::SoftOnSoft` silently returns an empty contact manifold
- **Where:** `crates/molla-geometry/src/hydroelastic.rs:131-135` — variant documented "**Not implemented in v1** — will return an empty manifold with a `tracing::warn`". Only `SoftOnRigid` is real.
- **Bug class:** contact silently disappears for a selectable public enum variant. A `tracing::warn` is not a contract.
- **Fix:** minimum viable: return a typed `Error` (or reject at policy-construction time). Full feature (real 3D pressure-field intersection for soft-soft) is a separate L-effort item — do it only if a consumer needs it; the error is the correctness fix.
- **Effort:** S (error path). **Risk:** LOW.

---

# T — Tooling hazards & CI gaps

### T1. CI never runs clippy; the real gate (`make harden`) is CI-invisible
- **Where:** `.github/workflows/rust.yml` installs `components: rustfmt, clippy` but only runs `cargo fmt --check`, `cargo build`, `cargo test`. `Makefile` `harden` target (line ~70) runs the full gate including `cargo clippy --all-features -D warnings` and the `--no-default-features` (CPU-only) build+test lane. The workspace declares clippy lint levels in `Cargo.toml:32-37` that CI never enforces.
- **Fix:** Add a clippy step (`cargo clippy --all-features -- -D warnings`) and a `--no-default-features` build+test job to `rust.yml`. Expect possible first-run lint debt; fix or explicitly `allow` with justification.
- **Effort:** S. **Risk:** LOW-MED (first run may surface existing lints).

### T2. `make docs` auto-commits the entire working tree
- **Where:** `Makefile` `docs:` target — after `mdbook build`, runs `git add --all && git commit -m "docs: building website/mdbook"`. Anyone building docs with uncommitted WIP silently commits *everything* under a docs message.
- **Fix:** Delete the `git add/commit` lines (preferred), or scope to `git add docs/` only. Note the commit message also violates the repo's own conventional-commit rule.
- **Effort:** S. **Risk:** LOW. **Do this early — it protects every subsequent agent.**

### T3. No GPU gate anywhere in CI
- **Where:** `rust.yml` `ignored-tests` job runs `--ignored` on `ubuntu-latest` with `continue-on-error: true` (no GPU adapter, can never gate). All GPU parity protection is a human running `make test-gpu` on a Vulkan box.
- **Fix options:** (a) provision a self-hosted GPU runner (L, infra/security work — needs a maintainer decision); (b) interim: add a documented pre-merge GPU checklist (which `make test-gpu` lanes must pass for which touched paths) to CONTRIBUTING/book, and make the PR template reference it.
- **Effort:** S (interim) / L (runner). **Risk:** MED for the runner (infra).

### T4. Stale CI comment references removed infrastructure
- **Where:** `.github/workflows/rust.yml` failure-bucket comment cites a `MOLLA_NEWTON_PYTHON` env var and `Makefile::test-oracle` target — neither exists anywhere in the repo, and `book/src/testing/strategy.md:56-58` correctly states the Newton oracle is pure ported Rust. Fix: delete/correct the comment. While there: ~230 of 245 `#[ignore]` attributes are bare; adding `#[ignore = "needs wgpu adapter — run via make test-gpu"]` is a mechanical nicety.
- **Effort:** S. **Risk:** LOW.

### T5. Book deploy workflow only triggers on `develop`
- **Where:** `.github/workflows/book.yml` `push: branches: [develop]`; `rust.yml` gates `[main, develop]`. If `main` is the release branch, the published book can never refresh from `main`. Needs a one-line decision + edit. **Effort:** S.

---

# F — Unfinished features: continue implementing

Ordered by leverage. Each item states what exists, what's missing, and what "finished" means.

### F1. MPM multi-material: per-particle material IDs (the flagship gap)
- **Exists:** `Scene::MpmMultiMaterial` in the viewer with three procedural sand/snow/viscous *visual* regions (`bin/molla_viewer/src/examples/103_mpm_multi_material.rs:14-22` color-bands particles by x-position); a reduced CPU oracle (`crates/molla-solvers/src/mpm_oracle.rs:250`, test `reduced_multi_material_oracle_dispatches_distinct_branches`).
- **Missing:** the runtime solver is single-rheology per simulation. `MpmConfig` carries one global `rheology: MpmRheology` (`crates/molla-solvers/src/mpm.rs:275`); the GPU uniform packs a single `rheology: u32` selector (`mpm.rs:149`, `mpm.rs:1332`).
- **Finished means:** a per-particle `material_id` buffer threaded through P2G / grid-update / G2P on both CPU and GPU MPM paths, per-branch constitutive dispatch in the WGSL kernel, one run evolving sand+snow+viscous simultaneously, mass-conserving within existing tolerances; viewer scene flips from `Partial` to `Complete` and the audit note updates. Reference: `xtra/newton/newton/examples/mpm/example_mpm_multi_material.py`.
- **Effort:** L. **Risk:** MED (hot MPM kernels + parity tests). **Tests:** extend the multi-material oracle to a full parity lane; `make test-gpu P=molla-solvers T=mpm_v3_gpu` must stay green.

### F2. Tire: zero-host-readback substep loop → real 40-wheel realtime
- **Exists:** the whole GPU tire stack — persistent mesh FEM (`step_persistent`), `TireFemContactFeedbackKernel` (GPU bristles), `step_hub_persistent`, `tire_fem_atlas.wgsl`, `tire_fem_hub.wgsl`. Rolling parity CPU-vs-GPU is closed (37/37 hardware lane, `TireFemTorqueGpuStatus = LiveValidated`). A 40-wheel atlas-batch *smoke* test passes (`tire_fem_gpu_resident_atlas_batch_steps_forty_wheels`).
- **Missing (per `PROGRESS.md:56-60`, `:222-254`):** each substep still round-trips to the host — the contact-feedback kernel gets inputs from host-side state, and the hub integrates on the host with `hub_omega` forwarded each frame. Remaining CPU/GPU structural diffs: single- vs two-stage friction-cone clipping, penetration- vs load-based bristle caps. The realtime perf target (10 vehicles × 4 wheels) is unproven.
- **Finished means:** mesh FEM → contact feedback → hub integration runs fully GPU-resident with zero per-substep host readback; a benchmark demonstrates 40 live FEM tires at realtime rates.
- **Effort:** M-L. **Risk:** MED — parity lanes are FP-sensitive (accumulation reorder has flipped marginal tests before; see `PROGRESS.md`). **Tests:** the existing `tire_fem_gpu` lane (37 tests) must stay green: `make test-gpu P=molla-kernels T=tire_fem_gpu`.

### F3. Render-time tire rotation in the viewer (sanctioned Stream-B replacement)
- **STATUS UPDATE (2026-07-02): already implemented** — found during execution, missed by the audit because `PROGRESS.md` still framed it as a next step. `TetraTireFem::nodes_rendered()` exists (`crates/molla-vehicle/src/real_tire/fem/solver.rs:1326`), scene 123 renders the wireframe with it, and `tetra_tire_fem_nodes_rendered_is_rigid_rotation_about_hub` passes. Nothing to do.
- ~~**Design already written:** `PROGRESS.md` — add `nodes_rendered()` on `TetraTireFem` returning `R(hub_angle)·(pos−hub)+hub`, wire into the scene-123 renderer.~~

### F4. Static/kinetic friction split (roadmap A1)
- **Spec:** `book/src/roadmap/open.md:31-53` — per-contact `friction_state: {Stick, Slip}` with `μ_s > μ_k` on top of the shipped Signorini+Coulomb active-set solver (Stages 50/51 landed). Unlocks stick-slip/chatter.
- **Effort:** S-M (doc's own estimate: 1-2 days). **Risk:** LOW.

### F5. Per-muscle stiffness + pennation on the GPU muscle path (roadmap A2)
- **Spec:** `open.md:55-75` — CPU honors per-muscle `stiffness_ratio`/pennation; GPU reads one shared parameter. "The kernel math already supports it."
- **⚠ First reconcile a doc contradiction:** `open.md:479` (item C6) claims the SEE ratio was later plumbed "with A2", yet A2 is still listed open. Check the actual GPU muscle kernel parameter buffers before starting; the remaining gap may be only pennation. Update `open.md` with what you find.
- **Effort:** S. **Risk:** LOW.

### F6. Multi-DoF joint support in CRBA / RNEA / analytic Jacobian
- **Where:** `crates/molla-solvers/src/crba.rs:110-119`, `rnea.rs:116-121`, `jacobian.rs:124` — all three error on `Ball|Free|Distance|D6|Cable` ("revolute/prismatic only"), while these joint types are first-class elsewhere (Featherstone path, humanoid ball-joint demo, Stage 7 D6/Cable). Failure is honest (`Err`), so this is missing coverage, not a bug.
- **Finished means:** a shared per-DoF motion-subspace expansion (multi-column S, as scoped in the `crba.rs:24` comment) consumed by all three kernels; existing revolute/prismatic parity tests stay green.
- **Effort:** L. **Risk:** MED (core spatial-algebra changes).

### F7. VBD GPU multi-substep
- **Where:** `crates/molla-solvers/src/vbd.rs:840` — `substeps > 1` on GPU returns `Error::Solver("...set substeps = 1 and run the substep loop externally")`; comment at `:834` defers GPU-resident ping-pong scratch. (This is the honest-failure pattern B1 should copy in the interim.)
- **Finished means:** GPU path honors `substeps > 1` with resident scratch, matching CPU multi-substep results in a parity test.
- **Effort:** M. **Risk:** LOW (additive behind existing error).

### F8. Transmission shift logic — vehicles are stuck in first gear
- **Where:** `crates/molla-vehicle/src/powertrain.rs:47-73` — `Transmission` has 5 gear ratios and a `gear: usize` defaulting to 0; `ratio()` reads it, but **nothing in the workspace ever mutates `.gear`**. The upper 4 gears are dead data.
- **Finished means:** RPM-based auto-shift (with hysteresis to avoid hunting) and/or manual shift exposed through `DriverInput`; a test driving through an upshift; viewer vehicle scenes (103/110/123) benefit automatically.
- **Effort:** M. **Risk:** LOW.

### F9. Suspension kinematics or honest demotion
- **Where:** `crates/molla-vehicle/src/suspension.rs:56-59` — `force()` uses only stiffness/damping/max_compression; `SuspensionKind::{RigidAxle, DoubleWishbone, MacPherson}` never affects physics. Doc at `:5-6` admits the gap.
- **Fix:** either implement kind-driven camber/toe kinematics feeding the tire model (M), or demote the field to a documented label / remove it from the physics API (S). Decide based on whether F2/F3-era vehicle demos need camber — the tire model already supports camber thrust (see `molla-vehicle` brush model).
- **Effort:** S or M. **Risk:** LOW.

### F10. Wire the sysid (differentiable friction identification) showcase into a viewer scene
- **Where:** `crates/molla-vehicle/src/sysid.rs` — `drive_distance`, `distance_loss`, `identify_friction` implemented and self-tested, referenced by nothing but `lib.rs`. Module doc (`:12-14`) says the viewer scene is "the remaining UI-wiring step".
- **Finished means:** a new scene (follow the numbering convention, next free example number) that runs `identify_friction` and visualizes convergence — recovered μ vs. ground truth. Remember ground rule 2: this is a scene *addition*, allowed and encouraged.
- **Effort:** M. **Risk:** LOW.

### F11. Sparse-grid GPU delete: leaf reclamation
- **Where:** `crates/molla-kernels/src/sparse_grid_insert.rs:248`, `shaders/sparse_grid_delete.wgsl:50` — delete marks leaves inactive; reclamation/folding requires a separate manual `reclaim_empty_leaves` dispatch; freelist/`pending_reclaim` buffers exist but folding on-empty is deferred (Stage 24e.2).
- **Fix:** fold reclamation into the delete pipeline, or formally document the manual-reclaim contract at the API level. GPU freelist concurrency is race-prone — copy the approach from `xtra/warp`'s sparse-volume handling if applicable.
- **Effort:** M. **Risk:** MED.

### F12. APIC coupling: real pressure projection (Stage 37.6.1)
- **Where:** `crates/molla-fem/src/apic_coupling.rs:62-71` — "No real Stokes projection (v1). The viscous-smoothing placeholder is incompressible only in the limit ν·dt → ∞." Also `:352` — particle→tet locate is brute-force O(P·T), spatial hash deferred. Both CPU and GPU mirror the same approximation.
- **Finished means:** a pressure-Poisson solve on the P1 mesh after grid update (keep viscous smoothing optional), plus a spatial-hash locate. Changes physical behavior — existing APIC test expectations will need review, which is expected and correct.
- **Effort:** L. **Risk:** MED.

### F13. 2D plane-strain SVK elasticity
- **Where:** `crates/molla-fem/src/element.rs:635-639` — `compute_element_st_venant_kirchhoff_stress` for `FemCell::Tri` is `unimplemented!()` (deliberate loud panic); `element.rs:777` `tet_deformation_gradient` also panics on non-Tet.
- **Fix:** implement plane-strain SVK for Tri, or gate 2D out at the API boundary with a typed error. Lower priority than the silent failures above precisely because it panics loudly.
- **Effort:** M. **Risk:** LOW.

### F14. GPU CCD for the fast-impact pipeline
- **Where:** `bin/molla_viewer/src/main.rs:3640` — `FastSoftImpact` audit note: CCD is "CPU-only (TODO Stage 33.4.1)"; the GPU lane can tunnel.
- **Fix:** GPU CCD lane, or document CCD as a permanent CPU-only stage in the audit. **Effort:** L. **Risk:** LOW (opt-in scene).

### F15. The 15 `Partial` viewer-parity scenes (tracking umbrella — not one task)
All 15 scenes from `PLAN.md` Priority 1-3 exist with honest `ScenePhysicsAudit` entries but visualize reduced/proxy models. Each promotion `Partial → Complete` is its own work item; the audit `notes` field in `bin/molla_viewer/src/main.rs:2645-3239` names each gap precisely. The recurring missing halves:
  - **GPU adjoints** for the four diffsim/optim scenes (`DiffsimCloth`, `DiffsimSoftBody`, `FluidCheckpointOptimization`, `NavierStokesPerturbationOptimization`).
  - **Coupling the reduced oracle into the live solver step** (`NutBoltHydro` — hydroelastic forces not coupled into the live viewer solver; `CableBundleHysteresis`; `ClothRollers`; `MpmBeamTwist`).
  - **True mixed/nonconforming FEM kernels** (`WarpFemMixedElasticity` `main.rs:2942`, `WarpFemStokesTransfer` `main.rs:2956`, `WarpFemNonconformingContact`, `WarpFemConvDiffDg`).
  - **IK obstacle avoidance as an objective** (`IkCubeStacking`, `examples/113_ikcubestacking.rs:360` — obstacle residual reported but not an avoidance objective).
  - **F1 covers `MpmMultiMaterial`**; `NutBoltSdf` needs threaded-SDF narrow phase.
- Also in this bucket: 21 of 131 scenes total are `Partial` (audit distribution: 43 Complete, 8 CpuOnly, 8 ManualGpu, 21 Partial). **Effort:** S-M each, L in aggregate.

---

# D — Dead code cleanup (small, surgical — the full list)

The workspace is clean (zero cargo warnings; the ~60 `#[allow(dead_code)]` sites are overwhelmingly intentional — see Verified non-issues). These are the only confirmed removals:

### D1. `color_particles` in molla-sim builder
`crates/molla-sim/src/builder.rs:3316-3317` — private, `#[allow(dead_code)]`, superseded by `color_particles_with_tets_and_bending` (`:3430`, called at `:2606`). Only reference is a doc-comment link at `:3419`. **Delete the fn, rewrite the doc comment in the same edit.** ~75 lines. Verify: `make check && make test`.

### D2. Empty `molla::utils` facade module
`crates/molla/src/utils/mod.rs` contains only `//! Miscellaneous helpers.`; declared at `crates/molla/src/lib.rs:31`; zero references in the viewer (sole facade consumer). **Delete the directory and the `pub mod utils;` line** (or populate it — but nothing wants it). Verify: `make check && make test`.

### D3. `_suppress_contacts` shim in gjk_gpu
`crates/molla-geometry/src/gjk_gpu.rs:270` — no-op fn kept to protect an unused `Contacts` import "just in case a follow-up wires the kernel". Drop the shim + import until the follow-up actually lands. Verify with `make check` **and** the geometry GPU lanes (`make test-gpu P=molla-geometry T=<gjk/epa tests>`) since GPU-gated code is invisible to the default build.

### D4. `TireFemMatrixFreeNewtonGpuStatus::WgslSourceGuardOnly` — unreachable variant
`crates/molla-kernels/src/tire_fem.rs:3010` — declared; the only builder (`:3080-3084`) emits only `Live` or `CpuOnlyDebug`. Either remove the variant or make the builder emit it in the guard-passes-but-no-kernel state. Coordinate with DOC1 (same file, same seam).

### D5 (investigate only). Non-hull `support()` in gjk/epa
`crates/molla-geometry/src/gjk.rs:23`/`epa.rs:14` import `support` alongside `support_with_hulls`; both files carry `_suppress_unused_support_warning` shims (`gjk.rs:176`, `epa.rs:352`), hinting the plain variant is unused in these paths. **Trace all callers of `crate::support::support` first** — if only the shims, consolidate. LOW confidence; do not delete blind.

**Cross-cutting deletion rule:** default `cargo check` compiles WITHOUT `--features gpu`; the real consumers of much `allow(dead_code)` code are `#[ignore]`d GPU tests and viewer scenes. Any deletion in molla-kernels/molla-fem/molla-solvers must be re-verified with the relevant `make test-gpu` lanes, not just `make check`. D1/D2 are CPU-side and safe with `make check`.

---

# A — Architecture & structural debt

### A1. The viewer is one ~44k-line module (God-file via `include!`)
- **Where:** `bin/molla_viewer/src/main.rs` (11,067 lines) textually `include!`s all 131 files under `src/examples/` (33,027 lines) at `main.rs:170-300`. No `mod` boundaries except `mod tests` (`:7825`). One flat namespace: helper names must be globally unique; incremental compiles touch everything.
- **Fix:** convert `include!` → real `mod` per example under a `scenes` module; expose the shared framework (the `Sim` resource, pane/readout helpers, gizmo utils, `WarpExampleStates`) through a `pub(crate)` prelude. Examples currently see *private* items of `main.rs` — each such leak must be made `pub(crate)`. Keep the single-executable rule intact.
- **Do this incrementally** (e.g. 10-20 scenes per PR), verifying `make check && make test` each batch. **Effort:** L. **Risk:** MED (wide but mechanical).
- **Dependency:** do A1 before A2/A3 (they assume module boundaries exist).

### A2. Tire UI/logic scattered through the framework file
~1,671 lines of tire-specific egui panels — 20 `fn tire_*` functions from `tire_gpu_parity_matrix_panel` (`main.rs:5909`) to `tire_runtime_production_gate_panel` (`main.rs:7526`) — live in framework `main.rs`; the three tire scenes are the largest example files (122: 2,688 / 120: 2,248 / 121: 1,944 lines). After A1, gather panels + scenes into `scenes::tire`. **Effort:** M.

### A3. Duplicated scene bodies
- Vehicle trio: `Scene::Vehicle` (103), `VehicleOffroad` (110), `TerrainLab` (123) all build `HeightfieldTerrain` + `Vehicle::car_with(RealBrushTireModel::offroad_soft())` differing only in heightfield fn and gizmos. Extract `build_vehicle_on_heightfield(terrain_fn, tire_model)`; scenes keep their terrain generators. **All scenes stay** (ground rule 2) — only the bodies are shared.
- MPM trio: `WarpDem` (59), `WarpFluid` (60), `WarpSph` (61) run the same MPM-Sand solver at different friction angles. Share one parameterized entry point. Note `WarpSph` is labeled an SPH port but runs MPM — its label/doc should say "MPM-backed visual parity" prominently (the doc already redirects to `SphDamBreak` for real WCSPH).
- **Effort:** M. **Risk:** LOW.

### A4. Two scenes implement PDE numerics inline in the viewer (rule violation)
- `examples/96_dgburgersshock.rs:49-100` — own `minmod` + Lax-Friedrichs/MUSCL-TVD step, zero `molla` calls.
- `examples/97_shallowwaterdambreak.rs:32-171` — own LxF finite-volume SWE step, zero crate calls.
- Worse: both scenes' `physics_audit` cite crate tests (`fem_dg*`, `main.rs:2998`) as `cpu_proof` — those tests exercise the *crate* solver, not the inline code that actually runs. The audit overstates coverage.
- **Fix:** hoist the flux/limiter/SWE-step math into `molla-fem` (it already owns DG flux/limiter code — the Burgers scene's own header says its limiter is "analogous to the BJ limiter the molla DG-P1 path uses") behind a tested API; scenes call it; repoint `cpu_proof` to real tests. This is the same class of fix as the historical tire `patch_factor` cleanup. **Effort:** M. **Risk:** LOW (pure numeric extraction, identical visuals).

### A5. Mislabeled "FEM port" scenes that run analytic animations
`Scene::WarpFemTaylorGreen` (`main.rs:1597`, `examples/70_warpfemtaylorgreen.rs:39-55` — closed-form `exp(-8π²νt)` decay) and `Scene::WarpFemStreamlines` (`main.rs:1603`, `examples/72_warpfemstreamlines.rs:18-24` — analytic vortex). The audit table is honest ("no PDE solver is invoked"), but the enum doc and UI label say "Port of fem/example_taylor_green.py". **Fix in place (S):** relabel "(analytic field)" and align doc-comments; **or (M-L):** actually drive them from the real FEM Stokes path. Never delete.

### A6. molla-vehicle is an island — not in the facade, not in molla-sim, own vec-math
Three related decisions to make explicit:
1. **Facade (S, do now):** `crates/molla/src/lib.rs:18-32` re-exports every crate except molla-vehicle; the viewer pulls it as a second direct dep (`bin/molla_viewer/Cargo.toml:33`). Add `pub mod vehicle { pub use molla_vehicle::*; }` + Cargo dep, mirroring the sensors pattern.
2. **Vec-math (S-M):** `tire.rs:290-306`, `granular.rs`, `real_tire/internal.rs` each define their own `dot`/`cross`/`normalize` over `[f64;3]`; the crate has zero workspace deps (not even molla-math). Consolidate into one internal `vec3` module or adopt molla-math.
3. **Chassis integration (L, needs maintainer decision — write a spike/design doc, don't just build):** `Vehicle::step` (`vehicle.rs:129-205`) is a bespoke 3-DOF semi-implicit Euler; there is no path to place a vehicle inside a molla-sim scene (contacts with other bodies, articulated attachments). This may be an intentional "library on top" stance — the deliverable is a one-page design doc proposing either a coupling layer or a documented decision to stay standalone.

### A7. molla-vehicle public API surface is 7× its real usage
`crates/molla-vehicle/src/lib.rs:33-160` re-exports ~87 fns; ~24 rig/report fns (`tire_rig_camber_sweep`, `tire_rig_step_brake_response`, `tire_soil_rut_matrix`, …) are referenced only inside `real_tire/{rig,reports,tests}`. Also: `real_tire/reports/*` + `real_tire/rig/*` = ~13k LOC of non-test diagnostic harness in `src/`, and `real_tire/tests.rs` is a single 9,122-line file (184 tests). **Fix:** demote internals to `pub(crate)` or move rig/reports behind a `diagnostics` feature; split `tests.rs` by subsystem. Baseline to preserve: **223 tests / 0 ignored** in molla-vehicle. **Effort:** M. **Risk:** LOW.

### A8. `molla_sensors::camera_post` export asymmetry
`crates/molla-sensors/src/lib.rs:4` declares the module but (unlike lidar/pointcloud/dataset siblings) doesn't flatten its items with top-level `pub use`; the facade compensates with an inline module (`crates/molla/src/sensors/mod.rs:34-36`). Cosmetic consistency fix. **Effort:** S.

---

# DOC — Documentation drift (actively wrong docs, worst first)

### DOC1. Stale "not implemented yet" comments on shipped, validated GPU tire lanes
`crates/molla-kernels/src/tire_fem.rs` — doc-comments contradict the code's own status machinery:
- `:3107-3113` (`TireFemTorqueGpuStatus` doc): "the WGSL step kernel does not yet integrate them — that is Stage G part 2" — but the builder at `:3184` resolves `LiveValidated` (closure flag `:3216`) and the shader reads `spin_omega = u.phys5.x`.
- `:112-124` (`hub_omega` field doc): "plumbed through but ignored on dispatch" — false since the rolling-parity landing.
- `:3020` (matrix-free Newton contract doc): "The hardware path is intentionally not implemented yet" — but the builder resolves `Live` with a hardware test (`tests/tire_fem_matrix_free_newton_gpu.rs`; `parity_ledger.rs:268`).
This is dangerous drift: anyone auditing the GPU path is told validated lanes are unimplemented. **Fix:** comment-only edits matching the resolved statuses. Coordinate with D4. **Effort:** S.

### DOC2. `book/src/architecture/build.md` documents a Makefile that no longer exists
Lists `make fmt`, `make run-profile`, `make bench-cpu/-gpu/-featherstone`, and ~40 `make test-<family>-gpu` targets (`build.md:11-25`, `:60-91`). Actual Makefile: one generic `test-gpu P= T= A=` lane; `make build` builds only `-p molla_viewer` (not `--workspace` as claimed). Cause: commit `11e754b` simplified the Makefile without updating the chapter. **Rewrite the chapter against the current Makefile.** Also fix `PLAN.md:58,:90` which cite the nonexistent `make fmt`. **Effort:** M.

### DOC3. molla-vehicle absent from the entire book
Zero hits for "vehicle" in `book/src/`; `SUMMARY.md` has no chapter; `architecture/crates.md` diagram (`:26-37`), extraction list (`:67-80`), and the reproduced workspace-members block (`:84-105`) all omit it — while it's the subject of the 262KB `PLAN_REALTIRE.md` and the most active development area. **Add a crate chapter + update crates.md/SUMMARY.** **Effort:** M.

### DOC4. `book/src/roadmap/open.md` is ~90% landed items presented as open
Framed as "work that is not shipped on main" (`:3`), snapshotted at `2bde182` (2026-05-11). A3-A7, all B1-B6, all C1-C7 now carry "Status. Landed." Only **A1, A2, X1, X2** are genuinely open. **Prune landed items to a changelog section; keep the true open set.** While there, resolve the A2/C6 contradiction (see F5). **Effort:** M.

### DOC5. PROGRESS.md and PLAN_REALTIRE.md contradict themselves/each other
- `PROGRESS.md`: top says Stream A COMPLETE / Stream B DROPPED / Stream C mostly delivered; the bottom half (`:546-642`) still instructs a reader to perform the file split (done — `real_tire.rs` is now a 35-module tree) and hold a "220/2" baseline (four different counts appear: 220/2, 221/1, 222/0; ground truth is **223/0**). **Add a "CURRENT STATE" header at top and prune/mark the superseded tail.**
- `PLAN_REALTIRE.md:3679-3704` still presents Stream B (dropped) as the priority and the `:3899-3904` status table marks lanes "⏳ not yet" that the parity ledger records as Live/LiveValidated. **Add a supersession banner pointing at PROGRESS.md's top.**
- **Effort:** S-M.

### DOC6. Minor drift batch (one PR)
- `book/src/SUMMARY.md` "Scene catalog (#1-101)" — scenes reach 123 (131 total registered).
- `roadmap/stages.md` "~100 viewer scenes" — same fix.
- Stale test-count references (220/2 etc.) anywhere they appear — the baseline is 223/0.
- `book/src/testing/strategy.md` — add the vehicle-crate test story.
- **Effort:** S.

---

# N — Direction: grounded next features (maintainer picks; each has real groundwork)

These are options, not defects — evidence shows the groundwork already exists. Respect ground rule 6 (exclusions).

1. **N1 = F3 (render-time tire rotation).** S effort, design written, flagship demo payoff. Do first.
2. **N2 = F2 (zero-readback 40-wheel realtime).** The stated headline goal; parity prerequisite already met. M-L.
3. **N3 = F4 + F5 (roadmap A1 friction split, A2 muscle params).** The only two open near-term roadmap slices; 1-2 days each per the roadmap's own estimates.
4. **N4: Backfill viewer scenes for Stages 39-51** (roadmap X2, `open.md:549-554`): ~20 shipped solver stages (40 v4, 41 v4-v6, 43-51) have zero viewer scenes — the newest physics is invisible in the project's only showcase. ~1 scene/day, mechanical, follow PLAN.md scene conventions (`ScenePhysicsAudit` + cpu_proof test). S per scene.
5. **N5: Critic audit of Stages 39-51** (roadmap X1, `open.md:540-547`): ~25 stages shipped since the last focused read-only audit (`90f3708`, Stage 38); historical base rate 0-2 critical-math issues per pass — and these are exactly the GPU lanes CI can't protect (T3). S (1 day, read-only).
6. **N6: Vehicle↔molla-sim coupling design spike** (= A6.3). L to build, S to design. Write the one-pager first.

**Considered and rejected** (do not resurrect without new evidence): ALE steady-state transport for true rolling FEM (`PROGRESS.md:217-221` — ~3-month research refactor; F3 covers the visible need); further C1-C5/C7 physics axes (landed as CPU oracles, no workload demands more); a general autograd tape (hand-coded adjoints are the design).

---

## Suggested execution waves

| Wave | Items | Rationale |
|------|-------|-----------|
| 1 | T2, B3, B1-interim (error on GPU substeps>1), DOC1, D1, D2 | Hours each; removes footguns and silent-wrongness before anything else |
| 2 | B1 (full), B2, T1, T4 | Correctness + CI floor |
| 3 | F3, F4, F5, D3, D4, A6.1, A8 | Small features with outsized value |
| 4 | DOC2-DOC6, T3-interim, T5 | Make the docs stop lying to the next agent |
| 5 | F1, F2, F8, F10, A4, A5 | The big unfinished features |
| 6 | A1 → A2/A3, A7 | Structural refactors (after features, to avoid rebase pain — or before, if F-work will touch the same scene files; coordinate) |
| 7 | F6, F7, F9, F11-F15, N4, N5, N6 | Long tail, maintainer-prioritized |

**Standing verification for every wave:** `make check && make test`; for any change under molla-kernels/fem/solvers/vehicle GPU paths, the relevant `make test-gpu P=<pkg> T=<lane>` on Vulkan hardware; `make harden` before merging. molla-vehicle baseline: 223/0. tire_fem_gpu lane: 37/37.

---

## What was NOT audited

- `xtra/` (vendored Warp/Newton references) — out of scope by design.
- Security posture beyond secrets scanning (no findings; no credentials, no prompt-injection content encountered anywhere).
- Runtime performance profiling — no benches were run; perf claims herein come from code/comments.
- Metal/DX12 backends (compile but are CI-unexercised per `book/src/testing/strategy.md:152`) — only the Vulkan path is validated by the GPU lanes.
- Individual WGSL shader numerical correctness beyond what parity tests cover (see N5 — that audit is the follow-up).
