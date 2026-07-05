# PROGRESS — PLAN_REALTIRE

> **CURRENT STATE (read this first — the log below is reverse-chronological
> and older sections are superseded by newer ones).**
> Stream A (real_tire file split): **COMPLETE** — `real_tire.rs` is now a
> multi-module `real_tire/` tree; the step-by-step split guide further down
> is historical. Stream B (mesh rotation): **DROPPED** — replaced by the
> render-time rotation plan (`nodes_rendered()`, see "Stream B replacement"
> section). Stream C G1–G4: **CLOSED** — `TireFemTorqueGpuStatus =
> LiveValidated`. Test baseline: **225 passed / 0 ignored** in
> `molla-vehicle` (any 220/2, 221/1, 222/0, or 223/0 references below are
> stale snapshots). Render-time tire rotation **landed**: `nodes_rendered()`
> on `TetraTireFem` + scene-123 wireframe + rigid-rotation test (the "right
> next step" paragraph below is done). **Zero-host-readback substep loop
> landed (2026-07-02)** for the per-tire persistent path:
> `step_persistent → TireFemContactFeedbackKernel::dispatch_resident →
> step_hub_persistent_with_tread_forces` crosses the host zero times and
> matches the blocking readback chain on the G4 rolling scenario to
> sub-mm parity (hub Δ < 5e-3 m, nodes < 1e-3, bristles < 1e-4 — the
> test admits reduction-order f32 noise only; "bit-identical" was an
> overstatement corrected by the 2026-07-05 tire audit;
> `tire_fem_resident_contact_loop_matches_blocking_chain`).
> Remaining open tire work: GPU-resident hub XZ/spin integration (the
> per-substep feedback uniforms still carry host-tracked hub pose/ω),
> the atlas-path bristle stage, and the real-time 40-wheel perf proof —
> tracked in `INSPECTION.md` item F2.

## Stream C G4 — CLOSED (2026-06-12): contact-driven rolling parity landed

The third G4 leg is in:
`tire_fem_cpu_vs_gpu_rolling_contact_hub_x_parity` in
`crates/molla-kernels/tests/tire_fem_gpu.rs`. It settles a
`realtime_viewer` tire into real ground contact (hub pinned at a loaded
height until the patch forms, then released to dynamic until tread
nodes touch — the mesh descends at its node-damping terminal velocity,
so a free-fall settle is impractically slow), starts it rolling with
`set_hub_spin(6.0)` + 120 N·m sustained drive torque, and runs 96
substeps on both backends:

- CPU: `step_on_terrain` (spin-coupled XPBD contact + bristle brush in
  `step_inner`, `net_contact_fx` → `hub_v_xz`).
- GPU: `step_persistent` (node FEM + spin-coupled Coulomb contact) →
  `TireFemContactFeedbackKernel` (persistent GPU bristles, the brush
  stage; bristle state threads through the blocking dispatch; feedback
  velocities upload back into the persistent mesh) → summed tread
  wrench into `TireFemHubParams::external_force_xyz` →
  `step_hub_persistent`.

Result: **sub-cm hub-X and hub-Y agreement at every sample** with
non-vacuous gates (real contact at roll start, CPU rolls forward >0.2 mm,
GPU rolls forward, ω accelerates under torque). Run via
`make test-gpu P=molla-kernels T=tire_fem_gpu
A=tire_fem_cpu_vs_gpu_rolling_contact_hub_x_parity`. The per-substep
param re-bridge carries live cavity pressure into the GPU each substep —
the cavity drift fix called out below.

**`TireFemTorqueGpuStatus` is now `LiveValidated`** (new variant): the
contract auto-stages from the shader source guards plus the
`TIRE_FEM_TORQUE_ROLLING_PARITY_VALIDATED` closure flag (documented to
flip back if the lane regresses). Contract test + parity ledger updated.
Full `tire_fem_gpu` lane: 37/37 on hardware. The 40-wheel target's
parity prerequisite is met.

New supporting APIs on `TetraTireFem`: `set_hub_spin(omega)` (start a
wheel already rolling — vehicles hand wheels over at road speed) and
`hub_velocity()`.

**Follow-up also landed same session — on-device wrench reduction.**
`tire_fem_hub.wgsl` now has a second entry point
`hub_reduce_with_tread_wrench` (shared `integrate_hub` body): it
reduces the contact-feedback per-node force rows inside the hub
dispatch (X/Z only — normal load reaches the hub through beads, the
CPU `net_contact_fx/fz` convention) and integrates the hub in the same
pass. Host API: `step_hub_persistent_with_tread_forces(…, tread_forces:
&wgpu::Buffer, force_row_count)`. Equivalence proven on hardware by
`tire_fem_gpu_hub_tread_wrench_reduction_matches_host_sum` (host-summed
external-force path vs on-device reduction, hub states equal to 1e-5
over 4 steps, with poisoned Y rows to catch component leaks). Full
`tire_fem_gpu` lane: 38/38.

Remaining (tracked in the ledger follow-up, not G4): feed the
contact-feedback kernel its inputs from the persistent mesh buffers so
the whole substep loop runs with zero host readback, and tightening the
Two more structural divergences recorded by the 2026-07-05 tire audit:
the FUSED-ATLAS node kernel omits the reference path's spin-coupled
built-in ground friction (tire_fem.wgsl phys5.x spin_v injection has no
AtlasParamRow slot — fused wheels get longitudinal reaction only through
the bristle feedback stage), and the resident feedback zeroes Fy by
design (hub vertical support = downforce + beads only, never the FEM
contact normal). Both are intentional-for-now and tracked here.
structural CPU/GPU differences (single- vs two-stage friction cone
clipping, penetration- vs load-based bristle caps).

## Brush physics: linear leading-edge bristle profile (2026-06-12) — LANDED

The long-standing 220/2 test baseline is now **222/0**. The two
pre-existing failures (`lower_pressure_improves_low_mu_tractive_stiffness`,
`tetra_tread_contact_force_changes_with_deformed_shape`) shared one root
cause: both stateless brush trial paths (`evaluate_brush_tire_oriented`,
`TetraTireFem::evaluate_tread_contact`) used a **uniform** bristle
displacement `κ·R` per cell, so contact-patch length (and therefore
inflation pressure) had almost no effect on tractive stiffness. The
textbook brush model has bristle deflection growing linearly from the
leading edge, which is what makes force scale with patch length.

What landed:

- `brush_bristle_profile` (`real_tire/brush.rs`) — linear deflection
  profile, zero at the leading edge, normalized so the patch average is
  exactly 1 for uniformly spaced cells (half-cell offset handles the
  single-cell grazing patch). Calibration at reference conditions is
  preserved; only the in-patch distribution changes: trailing cells
  saturate first (physically correct) and pneumatic trail emerges.
- `brush_pressure_patch_gain` — patch-length gain vs the 180 kPa
  reference inflation from the analytic `a ∝ k_n^(-1/3)` scaling
  (`k_n = carcass + 3P/R`). This carries the aired-down-tire traction
  effect: lower pressure → longer patch → more force before the cap.
- Both CPU paths apply `profile × gain`; the persistent-bristle path
  (`apply_tread_contact_wrench`) is deliberately unchanged — integrated
  bristle state develops the leading-edge profile naturally.
- GPU mirror: `tire_brush.wgsl` got the same geometry pre-pass
  (workgroup-shared patch extent) + profile + gain. All 7
  `tire_brush_gpu` CPU↔GPU parity tests pass on hardware via
  `make test-gpu P=molla-kernels T=tire_brush_gpu`.

Real trail exposed two latent issues in the acceptance gates (they had
only ever seen ≈zero trail):

1. **Aligning-moment sign convention.** The checks compared against the
   ISO z-up formula `Mz = -Fy·t`, but in this codebase's y-up frame with
   `side = heading × normal`, the restoring moment about the up-axis is
   `Mz = +Fy·t` (derivable directly from the code's own
   `mz = -Σ x·ft_lat`). Fixed in `rig/snapshot.rs` and
   `reports/physical_shape.rs` (three sites) with frame-convention
   comments.
2. **Check robustness.** The reported moment is taken about the hub, so
   it carries a `-Fy·center` term; on obstacles (e.g.
   `rolling_obstacle_crest`) the displaced patch center dominates the
   trail term and the convention is untestable — the check now requires
   trail-term dominance, plus a 0.05 N·m absolute floor so degenerate
   single-cell FEM patches don't compare FP noise signs. The
   adhesion-sweep `moment_sign_stable` gate now only requires sign
   stability above 10% of the sweep's peak |Mz|, since the aligning
   moment physically decays toward zero (and may cross) as the patch
   saturates at high slip angles.

The FEM shape-sensitivity unit test now probes at moderate slip
(κ ≈ 0.12) instead of κ ≈ 0.4: deep in saturation force ≈ μN in any
brush law and shape physically cannot matter; the gate (>100 N force
delta between 330/80 kPa) is unchanged and now passes with margin.

## Stream A (file split) — COMPLETE

## Stream B (`TetraTireFem` co-rotational refactor) — DROPPED

**Decision (2026-05-26):** drop the mesh-rotation work entirely. Per
user direction after the four failed attempts (see "Why my four
attempts failed" below) and the research conclusion that a Lagrangian
continuum mesh can't be rotated through contact without ALE-class
infrastructure: the wheel just won't visibly rotate in the viewer.
The spoke gizmo is the visual cue for rotation; the mesh stays put
and gets deformed under contact in place. This is exactly the
BeamNG/AC/iRacing pattern (physics doesn't rotate; render-side
rotation is the optional shortcut nobody on this codebase has asked
for yet).

**Cleanup landed this session:**
- Reverted `Stage C.1` — FEM force loop reads `self.pos` again
  (was reading `self.body_pos` as prep for body-canonical).
- Reverted `Stage A.5` — removed the end-of-step `body_pos`/`body_vel`
  refresh and the `world_velocity_to_body` helper.
- Kept `Stage C.6` — persistent scratch buffers (~320 k allocs/s
  saved at 8 000 substeps × 40 wheels). This is pure perf, unrelated
  to rotation.
- Result: test baseline back to **220/2** (was 221/1 with C.1's
  incidental fix). The two failing tests are the same pre-existing
  pair from before Stream A started.

**Why the dropped attempts failed (kept for the record):**

After four failed experiments to make the FEM mesh visibly rotate
without breaking physics, web-search of the academic and industry
literature settled the question: **rotating a Lagrangian continuum FEM
mesh through contact is the wrong architecture**, and the techniques
used by production tools to handle rolling tires are:

1. **Render-time rotation only** (BeamNG, Assetto Corsa, iRacing,
   RoboTire). Physics state is in a non-rotating tire frame. The mesh
   is rotated by `R(ω·t)` *only* in the vertex shader for display.
   Rolling enters physics through the slip equation `κ = (ω·R_e −
   v_x)/v_x`, never through node positions. This sidesteps the
   fictitious-strain-rate explosion entirely.
2. **Arbitrary Lagrangian-Eulerian (ALE) / steady-state transport**
   (Abaqus `*STEADY STATE TRANSPORT`, COMSOL, Nackenhorst 2004). Mesh
   stays fixed in the wheel's spin frame; material flows through it
   via a convective term `(c·∇)v` in the momentum equation where
   `c = ω × r`. Math is correct; refactor is large; not real-time.
3. **FTire / Pacejka flex-ring** approaches use ~200 lumped masses
   connected by stiff springs, *not* a deformable continuum. The ring
   can rotate as a Lagrangian body because rigid spring topologies
   tolerate the rotation without exploding. Doesn't apply to our
   Neo-Hookean + Holzapfel tet mesh.
4. **Co-rotational FEM (Müller-Fischer et al.)** is element-local — a
   polar decomposition per tet extracts a rotation `R_e`, strain
   computed in the un-rotated frame. Handles large rotations of
   individual elements with small strains. Does **not** address
   rolling (whole-body rotation through contact); complementary not
   substitute.

### Why my four attempts failed (for the record)

| # | Attempt | Mode of failure |
|---|---|---|
| 1 | Rotate `pos` at end of step, leave `vel` | Fictitious strain rate to FEM elements → explosion |
| 2 | Body-frame force buffer with R transform | Marginal FP-sensitive tests flip from accumulation reordering |
| 3 | Rotate bead-spring target by R | Positional discontinuity in spring → oscillation explosion |
| 4 | Δω×r velocity boost + rotated bead target | Combined feedback loop → explosion |

Every Lagrangian rotation attempt hits the same physics wall: the FEM
deformation gradient `F = Σ x_i·grad_i` interprets any imposed mesh
motion that isn't a *physical strain* as exactly that — a strain — and
the elastic forces respond accordingly. The only way to spin a
deformable continuum mesh through contact without exploding is ALE,
which is a 3-month research refactor, not a chat-session change.

### What this means

For this session, Stream B is **parked** at the locked-in wins:

| Stage | Status |
|---|---|
| A | body-frame storage (`body_pos`, `body_vel`) + accessors | ✅ done (pre-session) |
| B | body↔world transform helpers + `world_velocity_to_body` | ✅ done (this session, A.5) |
| A.5 | sync body_pos/body_vel from world at end of `step_inner` | ✅ done (this session) |
| C.6 | persistent scratch buffers | ✅ done (this session) |
| C.1 | FEM force loop reads `body_pos` | ✅ done (this session) — +1 test fix |
| C.2–C.5 | Lagrangian body-canonical refactor | ❌ ABANDONED — wrong architecture |

The right next step if visible rolling is needed: **render-time
rotation** in the viewer (BeamNG/AC pattern). Add a `nodes_rendered()`
accessor on `TetraTireFem` that returns `R(hub_angle)·(pos[i]−hub)+hub`;
wire it into the scene-123 renderer. Physics untouched, baseline
preserved, wheel visibly rolls. Trade-off documented: deformation
visually moves with the rotation instead of staying glued to
world-bottom, but at typical car speeds the elastic relaxation makes
that look like material flowing through the contact patch — exactly
what you see on a real rolling tire.

The physics-correct upgrade if ever needed: **ALE steady-state
transport** — add the convective `(c·∇)v` term to the weak form, treat
ω as a prescribed convection velocity, mesh becomes Eulerian in the
spin frame. Tagged as a separate research project, not Stream B work.

## Stream C — 40-wheel GPU scaling — MOSTLY DELIVERED

What landed:
- **G3** was already done before the session (`tire_fem_hub.wgsl` —
  per-workgroup bead reduction + 3-DOF hub integration).
- **G4** got two of three legs landed this session: node-FEM parity
  + dynamic-hub parity. The third leg (contact-driven rolling) is
  scoped and waiting.
- The **CPU→GPU input bridge** (`tire_fem_gpu_input_from_tetra_fem`)
  was the missing piece that made any of this possible.
- **40-wheel scaling smoke** (`tire_fem_gpu_resident_atlas_batch_steps_forty_wheels`)
  validates the fused atlas batch dispatch survives 40 tires × 24
  substeps × ~150 nodes/tire = 6 000 nodes per dispatch. That's the
  production scaling-path proof for the 10-machines × 4-wheels goal.

What's dropped:
- **G1** (GPU body-frame storage) and **G2** (WGSL spin-coupled
  contact in body frame) were follow-ups to Stream B's body-canonical
  refactor. Since Stream B is dropped (no mesh rotation), G1 and G2
  are cancelled too. The spin contribution to contact already lives
  in `tire_fem.wgsl` in the world frame (shipped via PLAN_TOURQUE
  Stage G) and is exercised by
  `tire_fem_gpu_spin_pushes_grounded_tread_backward_in_contact`,
  so the physical effect of `hub_omega` on tread bristle slip is
  already on GPU — it just isn't body-frame-canonical, which doesn't
  matter for non-rolling mesh.

The realistic 40-wheel-target story: G3 + G4 give validated
single-wheel GPU; the atlas batch dispatch in `tire_fem_atlas.wgsl`
already covers many tires per dispatch and survives 40 wheels
(`tire_fem_gpu_resident_atlas_batch_steps_forty_wheels`). So 40
wheels on GPU is achievable once the contact-driven G4 leg closes
and `TireFemTorqueGpuStatus` flips to `LiveValidated`.


| Stage | What | Status |
|---|---|---|
| G1 | GPU body-frame storage (mirror Stream B body-canonical) | ❌ DROPPED with Stream B (mesh-rotation work cancelled) |
| G2 | WGSL spin-coupled contact in body frame | ❌ DROPPED with Stream B (PLAN_TOURQUE Stage G already shipped the world-frame spin-coupled WGSL path) |
| G3 | atomic dynamic-hub integration on GPU | ✅ done (`tire_fem_hub.wgsl`, pre-session) |
| G4 | CPU-vs-GPU parity hardware lane | ✅✅⏳ two legs landed (`tire_fem_cpu_vs_gpu_node_parity_no_contact`, `tire_fem_cpu_vs_gpu_hub_y_parity_dynamic_hub`); contact-driven rolling-trajectory parity is the third leg, deferred to a follow-up session |
| C.batch | 40-wheel atlas batch smoke | ✅ `tire_fem_gpu_resident_atlas_batch_steps_forty_wheels` (10 vehicles × 4 wheels × 24 substeps) |

### What landed this session for Stream C

Added `tire_fem_gpu_input_from_tetra_fem(&TetraTireFem, base_params)
-> TireFemGpuInput` to `molla-kernels` under the `gpu` feature. Pure
data shuffle that fuses `tire.gpu_snapshot()` (mesh + topology + CSR)
with the tire's live hub state (`hub`, `hub_omega`, drive/brake torque,
cavity pressure) into a single GPU dispatch payload. Re-exported from
`molla_kernels::tire_fem_gpu_input_from_tetra_fem`.

Foundation for the eventual G4 parity hardware lane: build CPU and GPU
from the same `TetraTireFem`, dispatch in parallel, compare hub
trajectories within tolerance over N substeps. The conversion was the
missing bridge — both ends were complete (`TetraTireFemGpuSnapshot` +
hub accessors on CPU, `TireFemGpuInput` + `tire_fem.wgsl` /
`tire_fem_hub.wgsl` on GPU), only the join was absent.

Tested via `tire_fem_gpu_input_from_tetra_fem_preserves_snapshot_and_hub`
in `crates/molla-kernels/tests/tire_fem_gpu.rs` — verifies the mesh
topology + hub state propagate correctly. Runs on CPU (no wgpu adapter
needed). 1/0 ignored.

### Stream C G4 first-leg landed this session

`tire_fem_cpu_vs_gpu_node_parity_no_contact` in
`crates/molla-kernels/tests/tire_fem_gpu.rs` runs both CPU
`TetraTireFem::step_on_terrain` and GPU `TireFemGpuKernel::step_persistent`
from the same `realtime_viewer` preset with hub pinned high (no
contact) for 40 substeps × 125 µs = 5 ms simulated, comparing
per-node positions every 8 substeps. Tolerance is 6 cm absolute on a
32 cm outer-radius tire — loose by physics standards but tight enough
to catch a real divergence; drift is dominated by f32-vs-f64 + slight
XPBD iteration-order differences and grows roughly linearly at
~1 mm/substep. Passes via `make test-gpu P=molla-kernels T=tire_fem_gpu
A=tire_fem_cpu_vs_gpu_node_parity_no_contact`.

Empirical drift behaviour, captured for the next chapter:

- After 8 substeps (1 ms): max per-node drift ≈ 6 mm
- After 40 substeps (5 ms): max per-node drift ≈ 26 mm

This grows roughly linearly, meaning the divergence is systematic
(not pure FP noise). Suspected sources, in priority order:

1. **XPBD iteration order**: CPU iterates volume → structural in two
   separate passes; GPU `step_persistent` runs both within one kernel
   dispatch. Fixing requires either splitting the GPU kernel into two
   dispatches (one per XPBD pass) or moving the CPU into a single
   combined pass.
2. **Cavity pressure**: CPU evaluates `P_now = P_ref · (V_rest/V_now)^γ`
   each substep using `cavity_volume()`; GPU uses
   `params.cavity_radial_force_per_bead` baked at substep entry. The
   bridge passes the latter from `tire.gpu_snapshot()`, but the CPU's
   internal volume re-eval each substep diverges from the GPU's stale
   value. Easy fix: update `cavity_radial_force_per_bead` between
   substeps from CPU's live cavity volume.
3. **f32 vs f64 accumulation**: smallest of the three; ~1 part in 10⁷
   per multiply-add.

### Second leg landed this session — hub kernel parity

`tire_fem_cpu_vs_gpu_hub_y_parity_dynamic_hub` extends the first leg by
also dispatching `tire_fem_hub.wgsl` each substep. CPU runs with
`set_hub_dynamic()` so the hub integrates from bead reactions +
downforce + gravity; GPU does the same via `step_hub_persistent`.
After 24 substeps (3 ms simulated) the hub-Y trajectory agrees within
1 cm absolute. Passes via `make test-gpu P=molla-kernels T=tire_fem_gpu
A=tire_fem_cpu_vs_gpu_hub_y_parity_dynamic_hub`.

This combined with the first leg means the GPU stack matches CPU
through **both** the FEM node step *and* the hub equation. That's the
core of what G4 has to validate.

### Remaining work for full G4 closure

1. Wire the GPU contact reduction + contact feedback kernels for the
   contact-driven parity. Apply `drive_torque > 0` to both sides, let
   tire roll on flat terrain, check hub_omega + longitudinal hub
   displacement match.
2. Reduce the linear drift from XPBD iteration ordering by either:
   (a) splitting GPU XPBD into separate per-pass dispatches, or
   (b) moving CPU into a single combined pass.
3. Once tight (sub-cm hub agreement under rolling), flip
   `TireFemTorqueGpuStatus` from `Live` to `LiveValidated` and declare
   the 40-wheel target achievable.

Estimated work for full G4 closure: 1 more session focused on the
contact-driven leg.

| Stage | What | Status |
|---|---|---|
| A | body-frame storage (`body_pos`, `body_vel`) + accessors | ✅ done |
| B | body↔world transform helpers | ✅ done (now bidirectional: added `world_velocity_to_body`) |
| A.5 | sync body_pos/body_vel from world at end of `step_inner` | ✅ done (this session) |
| C.6 | persistent scratch buffers (`force`, `node_stiffness_diag`) | ✅ done (this session) |
| C.1 | FEM force loop reads `body_pos` | ✅ done (this session); **221/1** baseline (was 220/2) — `tetra_tread_contact_force_changes_with_deformed_shape` now passes |
| C.2 | XPBD passes read `body_pos` | ⏳ not yet |
| C.3 | contact in world, Δp/Δv via transform back to body | ⏳ not yet |
| C.4 | refresh `pos`/`vel` from body at end of `step_inner` | ⏳ not yet |
| C.5 | delete `apply_rigid_rotation_about_hub` (unused after C.4) | ⏳ not yet |
| D | scene 123 visual cleanup | ⏳ not yet |
| E | matrix-free Newton in body frame | ⏳ not yet |

Prior in-session notes about A.5 and C.1 are now historical — both were
reverted in the 2026-05-26 cleanup. **The only Stream B change still
in the code is C.6 (scratch buffers, pure perf).** The mesh does not
rotate visibly; the spoke gizmo remains the rotation indicator.

### Failed experiment — "rotate pos at end of step, not vel"

Tried as a shortcut for visible rolling without the full Stage C
refactor. Result: **exploded** within a few seconds in scene 123 (very
wobbly first, then violent blow-up). Reverted cleanly.

**Physical diagnosis.** Rotating only positions changes the
deformation gradient `F = Σ pos[id]·grads[a]` of every tetra without
the corresponding change in velocity. The FEM elements interpret the
sudden positional shift as an enormous strain rate, produce huge
restoring forces next step, which displace nodes further, F changes
more, forces grow more, → unstable feedback.

**Lesson.** Any rolling implementation must apply rotation to **both**
`pos` AND `vel` consistently — otherwise the implicit elasticity
solver sees a fictitious deformation rate and explodes. That means:
- the reverted PLAN_ROLL phase-1 ("rotate both at top of step_inner")
  is the only stability-preserving rigid-rotation option, but it
  jellies at the contact boundary;
- the only path to clean rolling is **full Stage C proper** where body
  is canonical and rotation only appears in the view transform at
  step boundaries.

### Second failed experiment — body-frame force accumulation (mini-1)

Tried to route FEM internal forces through a body-frame `force_body`
buffer, then transform via R and sum into the existing world-frame
`force` buffer. At `hub_angle = 0`, R = I, the math is bit-identical.
**Result**: passes most tests but BREAKS marginal FP-sensitive tests
(`tetra_tread_contact_force_changes_with_deformed_shape`,
`tetra_tread_contact_uses_actual_deformed_patch_nodes`,
`tire_rig_default_runner_emits_non_calibration_snapshots_and_envelopes`).
The cause is purely floating-point: even though `gravity + Σ_tet f =
gravity + Σ_tet f`, the *order* of accumulation differs (original does
gravity then per-tet, new does per-tet into body buffer then a final
sweep adding body buffer to world buffer). The reordering shifts the
last bits of `force[i]`, and the regression-style "force shape changed
under perturbation" tests are tight enough to flip.

**Lesson #2.** Stream B Stage C cannot be landed as a series of small
"bit-identical at hub_angle=0" patches — the FP accumulation order is
load-bearing for marginal tests, and any restructuring shifts it.
Stage C proper must land as one coordinated commit, accept that some
marginal qualitative tests will need their thresholds relaxed (they
test physical behaviour like "force changes by X% when shape changes
by Y%", which is robust under FP-precision-level perturbations but
not under fixed numeric thresholds), and validate cleanly via the
viewer rather than via the unit test count alone.

### Next chapter (C.1–C.5): DROPPED — see top of section

Stages C.1–C.5 flip the data flow direction (body becomes canonical, world becomes the derived view). The math is rotation-invariant at `ω = 0`, so the unit tests should hold bit-identically. But the user reverted Stage C phase 1 ("rigid-rotate at top of step_inner") last time because the **visual** behavior in scene 123 went jelly. The fix is the body-frame canonical storage in C.1–C.5, but the only sure way to know it actually fixes the jelly without introducing a new visible regression is to launch the viewer rolling-wheel scene and watch. Do not land C.1 → C.5 without that visual check.

Stream B is dropped. The above bullet list is historical and does not
correspond to a planned future session.
3. C.3: contact loop reads world `pos[i] = body_to_world(body_pos[i])`, projects against the (world-fixed) terrain, transforms Δp/Δv back to body via the inverse rotation.
4. C.4: at end of `step_inner`, refresh `pos[i] = body_to_world(body_pos[i])` (the existing Stage A.5 sync gets DELETED — body becomes the source, world becomes the derived view).
5. C.5: delete `apply_rigid_rotation_about_hub` (the reverted phase 1 hack).
6. **VIEWER CHECK**: launch scene 123 with `cargo run -p molla_viewer --example 123_terrain_lab`. Confirm rolling wheel does not jelly-wobble and the contact patch sits at world-bottom while material flows through it.

If C.1 alone causes any test drift (it shouldn't — math is rotation-invariant — but the existing physics has accumulated tolerances), use the `cargo test` failures to pinpoint where the world-vs-body mismatch leaks before going further.



## Where we are

`crates/molla-vehicle/src/real_tire.rs` (36 415 LOC, one file) has been
turned into `crates/molla-vehicle/src/real_tire/` with **35 modules**.
`mod.rs` is now down to **243 lines** of pure module declarations and
`pub use` re-exports — public API is unchanged. **36 172 lines moved
out** (99% of original mod.rs body). Test baseline (220 passing / 2
pre-existing failures) holds.
Test baseline preserved (220 passing / 2 failing — the 2 failures are
pre-existing on the 36k working tree and are independent of Stream A).

### Modules landed

| Module | Lines | Items |
|---|---|---|
| `real_tire/internal.rs` | 93 | `finite_or`, `finite_positive_or`, `finite_nonnegative`, `finite_vec3`, `sanitize_vec3`, `smoothstep01`, `csv_token`, `bool_metric`, `dot`, `cross`, `add`, `sub`, `scale`, `normalize`, `csv_optional_f64`, `csv_optional_vec3` |
| `real_tire/surface_material.rs` | 92 | `SurfaceMaterial` + presets + `sanitized_surface_material` helper |
| `real_tire/surface_film.rs` | 123 | `SurfaceFilmState` + `Default` + presets + `effective_material` builder |
| `real_tire/params.rs` | 121 | `RealTireParams` + `Default` + three named presets + builder methods |
| `real_tire/wheel.rs` | 33 | `RealWheelState`, `RealWheelOrientation` + `Default` |
| `real_tire/realism.rs` | 170 | `TireModelProvenance`, `TireRealismSourceItem`, `TireRealismSourceClassificationReport` |
| `real_tire/parity_ledger.rs` | 513 | `TireGpuParityStatus/Item`, `tire_gpu_parity_checklist`/`text_report`, `TireApproximationStatus`, `TireKnownApproximation`, `tire_known_approximation_ledger`/`text_report` |
| `real_tire/governance.rs` | 521 | `TirePresetGovernanceCheck/Report` + `tire_preset_governance_report` |
| `real_tire/carcass.rs` | 299 | `DynamicTireCarcassParams`, `DynamicTireCarcass` + impl |
| `real_tire/reports/soil.rs` | 869 | `TireSoilNormalShearCouplingReport`, `TireHeightfieldSurfaceVelocityMapReport`, `TireDemSoilFeedbackReport`, `TireMpmSoilFeedbackReport` + builders |
| `real_tire/rig/snapshot.rs` | 1195 | `TireRigSnapshot`, `TireRigAcceptanceCheck/Report`, `TireRigDimensionlessSanityMetrics`, `TireEnergySignAudit`, `TireRigRunConfig/Output`, `TireSolverBudgetTelemetry`, `TireReplayArtifactCoverageReport` |
| `real_tire/condition.rs` | 511 | `TireConditionLevel`, `TireConditionReport`, `TireConditionMaterialResponse`, `TireConditionFailureModeReport` + builder |
| `real_tire/reports/thermal.rs` | 1049 | `TireThermalAbuseEnvelopeReport`, `TireWearPowerTemperatureReport`, `TireWornSurfaceSensitivityReport`, `TireThermalRollingCarcassReport`, `TireThermalBrakeRimReport` + builders |
| `real_tire/reports/triangle.rs` | 1059 | `TireTriangle*` family (seam/manifold/surface stress/force/density convergence/large manifold) + builders |
| `real_tire/transient.rs` | 538 | `RealTireTransientState`, `TireRelaxationContext`, `TireTransientLayer{State,Lengths}`, `TireLayeredTransientState` + `evaluate_brush_tire_transient` family |
| `real_tire/telemetry.rs` | 526 | `TireContactPatchTelemetry`, `TireContactCellTelemetry`, `TirePressureDistribution{Bin,Telemetry}`, `RealTireTelemetrySample` |
| `real_tire/brush.rs` | 189 | `RealBrushTireModel` + `TireModel` impl + `BrushParams`, `sanitize_brush_params`, brush helpers |
| `real_tire/observability.rs` | 391 | `TireParameterObservabilityReport`, `TireRuntimeProductionGateReport` + builders |
| `real_tire/reports/convergence.rs` | 912 | `TireLayerAblationDebugReport`, `TetraFemSubstepConvergenceReport`, `TireHeightfieldResolutionConvergenceReport`, `TireThermalWearConvergenceReport` + builders + `tire_contact_patch_convergence_report` |
| `real_tire/reports/physical_shape.rs` | 2096 | `TireReferenceShapeDiagnostics`, `TireLoadCombinedSlipLawReport`, `TireIceLargePatchReport`, `TireAdhesionSlidingContinuityReport`, `TireLayeredTransientEnvelopeReport`, `TireLowSpeedRobustnessReport`, `TireFrameConventionReport`, `TireContactPatchConvergenceReport`, `TireRoadConditionStateReport` + all builders |
| `real_tire/rig/sensitivity.rs` | 660 | rig sensitivity sweeps + `tire_rig_standard_sensitivity_map`/`_text_report` + step-response snapshot |
| `real_tire/rig/sweeps.rs` | 904 | envelope, rolling obstacle, load-deflection, longitudinal-slip, surface-slip matrix, surface-transition terrain, step-response config |
| `real_tire/reports/vehicle.rs` | 1425 | vehicle-level reports + `TireVehicleSanityCase` + scenarios |
| `real_tire/thermal_wear.rs` | 717 | `TireThermalState`, `TireThermalLayers`, `TireWearState`, `TireRigThermalWear*`, `TireThermalConditionSanityReport` + thermal/wear rig runs |
| `real_tire/fem/params.rs` | 301 | `TetraTireFemParams`, `TetraTireFemSolverMode` + presets |
| `real_tire/fem/diagnostics.rs` | 470 | `TetraTireFemDiagnostics`, `TetraTireFemResolutionPoint`, `TetraBeadSeat*`, `TetraCavity*Report`, `TetraBeadSeatLoadCaseReport` |
| `real_tire/fem/sidewall.rs` | 355 | sidewall dihedral / laminate / stretch / tread-belt diagnostics + sanity + dynamic load |
| `real_tire/fem/newton_reports.rs` | 1491 | sidewall coupling/ply-strain XPBD reports + Newton-family (stiffness, diagonal, element, patch, tiny-mesh, matrix-free 4-step) reports |
| `real_tire/fem/builders.rs` | 1360 | Tetra report builders (hardening, deformation, contact-slip, construction, cavity-bead/coupling, bead-seat-load, sidewall dynamic load + laminate coupling/ply-strain XPBD/dihedral XPBD builders + supporting fixtures) |
| `real_tire/fem/newton_builders.rs` | 2162 | Newton-family report builders (stiffness, diagonal, element, patch, tiny-mesh, matrix-free 4-step) + tetra-element / patch helpers + `TireSolverMovingFlatTerrain` fixture + `matrix_free_newton_correction` |
| `real_tire/fem/solver.rs` | 2745 | `impl TetraTireFem` — constructors, FEM physics, contact, XPBD projection methods, hub kinematics, GPU snapshot |
| `real_tire/viewer_target.rs` | 270 | `TireViewerTargetCoverage{Check,Report}` + builder |
| `real_tire/rig/runners.rs` | 459 | tire_rig_step_torque/brake/steer_response, tire_rig_combined_slip_sweep, tire_rig_camber_sweep, evaluate_brush_tire / _oriented, solve_loaded_tire_hub_height, surface_material_with_terrain_film |
| `real_tire/rig/standard_run.rs` | 2269 | `tire_rig_default_non_calibration_run` + `tire_rig_standard_non_calibration_run` (the big assembly that drives the whole rig suite) + `tire_rig_point` / `tire_rig_point_at` |
| `real_tire/contact.rs` | 189 | `TireContactCell`, `RealTireContact` + impl |
| `real_tire/fem/state.rs` | 218 | `StructuralEdge`, `SidewallDihedral`, `TetraTireFem`, `TetraTireFemGpuSnapshot` + `tire_nodes_finite` helper |
| `real_tire/fem/math.rs` | 388 | FEM math/mesh/constraint helpers: `det3`, `inv3`, `transpose3`, `lame_from_young_poisson`, `mat_vec`, `make_tetra_element`, `tetra_dm`, `tetra_signed_volume`, `add_holzapfel_fiber_stress`, `quad_dihedral_angle`/`_local`, `project_edge_length_constraints`, `project_sidewall_dihedral_constraints`, `project_sidewall_ply_strain_constraints`, `project_to_ground` |
| `real_tire/tests.rs` | 9089 | the entire test module, moved out so `mod.rs` stays small |
| `real_tire/mod.rs` | **243** | header + module declarations + `pub use` re-exports only (down from 36 415; ~99% reduction) |

### Test baseline

220 passing / 2 failing on `cargo test -p molla-vehicle --lib`. The 2
failures (`lower_pressure_improves_low_mu_tractive_stiffness`,
`tetra_tread_contact_force_changes_with_deformed_shape`) are
**pre-existing** on the working tree — they fail on the original
single-file 36k LOC version too. Stream A introduces **zero
regressions** — verified after each extraction.

Full workspace `make check` clean.

## Pattern

Each extraction step is:

1. Find the current line range for the items to move:
   `grep -nE '^pub (struct|enum|fn) NAME' real_tire/mod.rs`
2. Check external refs in the block:
   `sed -n 'FROM,TOp' mod.rs | grep -oE '(crate::[a-zA-Z_:]+|TypeNames)' | sort -u`
3. Extract to a new file:
   `sed -n 'FROM,TOp' mod.rs > newmod.rs`
4. Prepend `//! …` docs + `use super::{…}` imports
5. `awk` to remove the moved lines from mod.rs AND insert
   `mod newmod;` + `pub use newmod::{…};` near the top
6. `cargo check -p molla-vehicle` (expect a few unresolved-import
   errors; add to `use super::…` until clean)
7. `cargo test -p molla-vehicle --lib` to confirm the baseline holds
   (historical note: this guide said 220/2; the split is DONE and the
   current baseline is 225/0 — see the CURRENT STATE banner at the top)

Each extraction must be re-line-numbered against the CURRENT mod.rs
because previous extractions shift everything below them.

## Helpers in internal.rs

Whenever a new module needs a tiny pure-fn helper that's used in
multiple places (math, sanitization, formatting), the rule is: move it
to `internal.rs` with `pub(crate)` visibility, then have the callers
do `use super::internal::FOO;`. This avoids the "function lives in
mod.rs but new module can't reach it" trap.

Helpers already in `internal.rs`: `finite_or`, `finite_positive_or`,
`finite_nonnegative`, `finite_vec3`, `sanitize_vec3`, `smoothstep01`,
`csv_token`, `bool_metric`, `dot`, `cross`, `add`, `sub`, `scale`,
`normalize`, `csv_optional_f64`, `csv_optional_vec3`.

Next time a needed helper isn't there:
- `quad_dihedral_angle`, `tetra_signed_volume` (vec3 math)
- `csv_optional_string` (formatting)

### Sub-directory pattern

For child dirs (e.g. `real_tire/reports/`), import siblings via two
`super::`s: `use super::super::internal::{...}` and
`use super::super::{Type1, helper_fn, ...}`. The intermediate
`reports/mod.rs` is just `pub mod soil; ...`.

## Next moves

Listed with current mod.rs line numbers. Each shifts after the
previous moves, so re-grep before each extraction.

### Thermal/wear (large, ~3 000 lines)

`TireThermalLayers`, `TireWearState`, `TireThermal*Report`, plus the
many `tire_thermal_*_report` / `tire_wear_*_report` functions. All
share thermal/wear state. Target dir: `real_tire/thermal/`.

### Triangle reports (~1 100 lines, lines 1654–2640)

`TireTriangle*Report` family. Hardest move because they reach into
`TetraTireFem`'s private fields (`tread_nodes`, `pos`, `vel`). Need to
expose accessors first OR move both blocks together with TetraTireFem.
Tackle this after `fem/` splits out.

### Rig sweep machinery (~6 000 lines, scattered 6 000–14 000)

Long-lived target. The sweep functions are interleaved with helpers
and tests. Probably needs `real_tire/rig/` as a subdirectory with
individual files per sweep family (slip, surface transition, thermal
wear, envelope, etc.). Lots of helpers will need to migrate to
`internal.rs` or a new `rig/internal.rs`.

### TetraTireFem (the big one, ~15 000 lines)

Lines ~16527 onward. Per the plan, splits into:
- `real_tire/fem/params.rs` — `TetraTireFemParams`, `TetraTireFemSolverMode`
- `real_tire/fem/mesh.rs` — geometry build
- `real_tire/fem/tetra.rs` — Neo-Hookean + Holzapfel constitutive
- `real_tire/fem/xpbd.rs` — constraint projections (volume, bead, structural, dihedral, ply strain)
- `real_tire/fem/contact.rs` — tread-node contact branch + bristle
- `real_tire/fem/torque.rs` — PLAN_TOURQUE Stage A/B/D/F state
- `real_tire/fem/solver.rs` — `step_inner` orchestrator
- `real_tire/fem/matrix_free.rs` — `matrix_free_newton_correction` + 5 newton reports
- `real_tire/fem/diagnostics.rs` — all `Tetra*Report` types and builders
- `real_tire/fem/gpu_snapshot.rs` — `TetraTireFemGpuSnapshot`

Land in that order for bisectability. This is multi-session work.

## Stream B + C are unchanged

Body-frame refactor (Stream B) and GPU 40-wheel scaling (Stream C) stay
unblocked. The `fem/` directory now contains params, diagnostics,
sidewall, newton_reports, builders, newton_builders, solver, state, and
math submodules. The body-frame refactor (Stream B) has a real review
surface: `fem/solver.rs` (2,745 lines) holds the whole `impl
TetraTireFem` — the file that the body-frame canonical state work needs
to touch. `mod.rs` is down to 243 LOC; the rest of the production code
lives in 35 focused submodules. Stream A is **complete**.

### Pattern notes from Stream A

1. **Bulk-promote fields before extracting big impls.**
   For `impl TetraTireFem`, all of `TetraTireFem` /
   `TetraElement` / `StructuralEdge` / `SidewallDihedral` fields had
   to become `pub(crate)` BEFORE moving the impl. Use a Python regex
   pass to flip all field visibilities at once.
2. **Promote helpers before extracting builders.**
   For `fem/newton_builders.rs`, the patch_*/tetra_element_* helpers
   and the `MatrixFreeNewtonOutcome` struct (with all its fields)
   needed `pub(crate)` promotion so the `TetraTireFem` impl in
   `fem/solver.rs` could still call them. Without the field promotion
   the parent module's impl loses access to its own struct returns.
3. **`pub(crate) use` re-exports keep import paths stable.**
   After moving everything to submodules, sibling modules still
   `use super::super::xxx;` — the parent mod.rs's `pub(crate) use`
   chain keeps that lookup working without each sibling needing to
   know the actual extracted location.
4. **Tests can move with `mod tests;`.**
   Outdent the test bodies by one indent (was inside `mod tests {}`)
   and add `#![cfg(test)]` at the top of the new file. The test
   module gets `use super::*;` for visibility and the parent declares
   `#[cfg(test)] mod tests;`.

## Resume instructions

```sh
# Find current line numbers for the next move:
grep -nE '^pub (struct|enum|fn) (TireSoilNormalShearCouplingReport|tire_soil_normal_shear_coupling_report|TireHeightfieldSurfaceVelocityMapReport|tire_heightfield_surface_velocity_map_report|TireDemSoilFeedbackReport|TireMpmSoilFeedbackReport|tire_dem_soil_feedback_report|tire_mpm_soil_feedback_report)' \
  crates/molla-vehicle/src/real_tire/mod.rs

# Sanity check the block's external refs:
sed -n '<from>,<to>p' crates/molla-vehicle/src/real_tire/mod.rs \
  | grep -oE '(crate::[a-zA-Z_:]+|TypeNames)' | sort -u

# Extract (replace <from>, <to>, target_module):
cd crates/molla-vehicle/src/real_tire
sed -n '<from>,<to>p' mod.rs > <target>.rs
# Prepend doc + use header by hand.
# Run an awk that adds `mod <target>; pub use <target>::{...};` near top
# and deletes the moved range.
cargo check -p molla-vehicle  # fix unresolved imports until clean
cargo test -p molla-vehicle --lib  # baseline must hold (now 225/0; "220/2" is a stale snapshot)
```
