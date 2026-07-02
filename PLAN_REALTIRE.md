# Real Tire + Tire/Ground Contact Plan

> **SUPERSESSION NOTE (2026-07-02).** Parts of this plan are superseded by
> shipped work — always cross-check the top of `PROGRESS.md` before acting
> on stream framing or status tables below. In particular: Stream A is
> complete, Stream B (co-rotational mesh rotation) was **dropped** in favor
> of render-time rotation, Stream C G1–G4 parity is closed
> (`TireFemTorqueGpuStatus = LiveValidated`), and status tables marking GPU
> torque / matrix-free-Newton lanes "⏳ not yet" predate the
> `parity_ledger.rs` Live/LiveValidated records.

## Goal

Replace the viewer-only "patch factor" tire behavior with reusable tire physics in
`molla-vehicle`: a pressure-dependent deformable tire contact patch, a
material-aware ground contact law, and a brush/bristle traction model that can
distinguish asphalt, dirt, sand, and ice.

The viewer examples may stay simplified, but they must call the real model
instead of inventing fake traction multipliers locally.

## Calibration stance for this pass

Skip external calibration/fitting for now. Do **not** block tire work on
measured tire datasets, Pacejka/TMeasy coefficient fitting, proprietary rig
data, or import/export plumbing. The useful parts from the calibration/racing-
sim discussion still belong here, but only as native physics, diagnostics, and
repeatable falsification infrastructure:

Hard boundary for this pass:

- no measured-data importer, optimizer loop, coefficient-fitting workflow, or
  "match this real tire" target;
- no Pacejka/TMeasy/Magic-Formula force source. If reference curves appear at
  all, they are diagnostic shape/sign/saturation comparators only;
- no MJCF/USD/URDF/FFI/simulator-glue detours unless they directly improve the
  native Rust tire physics;
- no hidden fitted constants. Hand-tuned values must stay named, unit-labelled,
  range-audited, and tagged by provenance.

Accepted non-calibration deliverables from the calibration/racing-sim
discussion:

- **structural tire truth gates** — loaded radius, sidewall bulge, footprint
  width/length, pressure distribution, belt support, bead/rim drift, and cavity
  response must be inspectable before trusting traction numbers.
- **native force-shape checks** — pure longitudinal slip, lateral slip,
  combined slip, camber thrust, pneumatic trail, aligning moment, low-speed
  launch/lockup/reversal, and transient relaxation must be checked for sane
  sign, smoothness, continuity, and saturation without fitted force curves.
- **state-derived material effects** — dry/wet/ice/dirt/soil, rubber thermal
  state, tread wear, bristle shear, rolling loss, soil shear/compaction, and
  surface film must come from explicit model state and parameters, not hidden
  viewer multipliers.
- **deterministic falsification fixtures** — pressure/load/speed/camber/slip
  sweeps, torque/brake/steer steps, moving belts, curbs/cleats, ruts, thermal
  abuse, wear, and cooling recovery should be replayable with compact
  snapshots and broad pass/fail envelopes.
- **sensitivity and observability, not fitting** — expose pressure, load,
  speed, camber, water film, temperature, wear, soil stiffness, obstacle
  height, solver substeps, and mesh/contact resolution as named sweeps with
  unit-labelled telemetry, provenance, and visible discontinuities.
- **viewer falsification cockpit** — tire scenes should show truth labels,
  active physics path, expected behaviour, overlays, force/energy accounting,
  CPU/GPU deltas, and approximation rows so broken behaviour is reportable
  against a concrete expectation.
- **GPU-first production gates** — hot brush/contact, FEM stepping, terrain
  lookup, thermal/wear, deformable-soil feedback, transient relaxation, and
  many-wheel batching need GPU paths/parity tests or explicit CPU/debug ledger
  entries with a next step.

What we **do** keep from the calibration / racing-sim discussion, without
doing calibration:

- physical tire-construction checks: loaded radius, sidewall bulge, bead/rim
  stability, tread-belt support, pressure distribution, and contact width;
- native force-shape checks: pure slip, lateral slip, combined slip, camber,
  pneumatic trail, aligning moment, and transient relaxation must have sane
  sign, smooth buildup, finite low-speed behavior, and continuous saturation.
  **The shared tire truth-label panel now surfaces these broad force-shape
  envelopes under `Force-shape envelope gates`: load-sensitive combined slip,
  ice-with-large-patch grip capping, adhesion/sliding continuity, low-speed
  launch/lockup/reversal robustness, and torque-induced slip outcomes are all
  shown from reusable `molla-vehicle` reports without fitted force curves.**
- deterministic replay rigs: load, pressure, slip, camber, torque/brake/steer
  steps, wet/ice transitions, moving ground, curbs/cleats, soil/ruts, thermal
  abuse, wear, and cooling recovery;
- state-derived condition effects: wet film, loose soil, heat, cold running,
  overheating, wear, and abuse must come from explicit state variables and
  telemetry instead of hidden scene multipliers;
- force/energy/solver accounting: report friction-budget use, bristle state,
  pressure bins, dissipated power, heat/wear, rolling loss, moving-ground work,
  FEM substeps, XPBD corrections, CPU/GPU deltas, and known approximations;
- GPU-first parity: brush contact, FEM stepping, FEM contact feedback/reduction,
  terrain lookup, deformable-soil feedback, thermal/wear, transient relaxation,
  and many-wheel batching need GPU paths or explicit CPU/debug ledger entries.

- deterministic internal rig runs for load, pressure, slip, camber, wet/ice,
  soil, obstacle, moving-ground, thermal, and wear cases;
- broad qualitative acceptance envelopes instead of fitted target curves;
- shape/sign checks for smooth buildup, combined-slip saturation, load
  sensitivity, pneumatic-trail sign, and step-response continuity;
- unit-labelled telemetry and snapshot artifacts for reproducibility;
- preset provenance/range audits for hand-tuned parameters;
- scene truth labels and overlays that make wrong behavior obvious;
- CPU/GPU parity reports for high-frequency tire, contact, terrain, and soil
  paths, with explicit follow-up for anything still CPU-only or planned.

Immediate backlog from this decision:

- add rig/replay fixtures, not fitters: load/deflection, pressure footprint,
  pure slip, lateral slip, combined slip, camber/trail, torque/brake/steer
  steps, wet/ice transitions, moving-ground belts, curb/cleat hits, soil/rut
  feedback, thermal abuse, wear, and cooling recovery;
- make those fixtures emit compact internal snapshots with command inputs,
  preset names, physical state, solver knobs, contact telemetry, force/moment
  output, energy/heat/wear accounting, and optional CPU/GPU deltas;
- define broad qualitative envelopes for those snapshots: lower pressure grows
  footprint, ice/wet reduce usable force, load sensitivity is sub-linear,
  combined slip shares one friction budget, pneumatic-trail/aligning-moment
  signs are consistent, and step responses lag smoothly instead of teleporting;
- expose the same expectations in the viewer as truth labels, replay panels,
  overlays, and unit-labelled readouts so broken examples can be reported
  against a stated expectation;
- keep every hand-tuned number in a named preset/parameter group with units,
  broad range, and provenance (`physical`, `hand_tuned`,
  `numerical_stability`, `debug`, or `visualization`) instead of hiding it in
  scene code;
- treat optional Pacejka/TMeasy/racing-sim curves as future diagnostic
  comparators only. They must never become the force source in this pass and
  must not introduce fitted constants while calibration is deferred.

Latest requested non-fitting points to fold into the real-tire roadmap:

- **physical tire stack, not a single grip formula** — keep structural tire
  deformation, cavity/inflation, tread bristles, road/soil state,
  thermal/wear history, and wheel/vehicle dynamics as separable layers with
  explicit state exchange. The brush/contact law may use tunable parameters,
  but it must consume those physical states instead of replacing them with a
  hidden scene-level grip knob.
- **contact patch as the source of truth** — traction claims must be traceable
  to patch geometry: active cells, pressure/load bins, centre of pressure,
  force-resultant location, adhesion/sliding fractions, local normal support,
  and local shear capacity. If a tire visually looks inflated/deflated but the
  patch telemetry does not change, that is a bug.
- **road and soil affect both support and shear** — wet film, ice, packed
  dirt, loose soil, ruts, and moving ground must change normal support,
  tangential shear, rolling resistance, heat, and wear through terrain/contact
  state. The next soil-oriented rig should explicitly show normal-load sinkage
  plus driven-shear/rut feedback in the same replay row.
- **transient memory before peak-force polish** — torque, brake, steer, load
  transfer, carcass/tread relaxation, wheel spin, and direction reversal need
  finite history and low-speed regularization. Do not accept an instant
  steady-state lookup if it bypasses relaxation, energy, or sign accounting.
- **state-derived condition effects** — cold grip, warm-up, overheating,
  carcass softening, worn tread, wet worn-tread loss, loose-surface damage,
  and cooling recovery must come from energy, temperature, film, and wear
  state. Cooling can reduce temperature penalties; it must not silently heal
  irreversible wear.
- **vehicle coupling is part of tire truth** — four-wheel scenes should use
  the same tire APIs for load transfer, suspension travel, wheel spin,
  braking, steering/camber, obstacle hits, and off-road ruts. Steering must
  rotate the wheel frame and change patch forces, not translate the whole car
  sideways through a viewer shortcut.
- **GPU-resident hot path** — brush rows, tire FEM node/hub stepping, terrain
  lookup, contact feedback/reduction, deformable soil, thermal/wear,
  transient relaxation, and many-wheel batching should live on GPU where they
  are performance relevant. Any CPU-only path must be labelled debug/bridge
  with consequence and a concrete GPU follow-up.
- **falsification cockpit in the viewer** — scenes 120/121/122 and tire-
  adjacent soil/offroad scenes should expose expected behavior, active physics
  path, physical sliders, debug/visual sliders, approximation ledger, force
  arrows, pressure bins, patch/slip overlays, heat/wear/rut state, and CPU/GPU
  deltas so wrong behavior can be reported against a concrete row.
- **sensitivity and convergence before realism claims** — keep named sweeps
  for pressure, load, camber, speed, water film, temperature, wear, soil
  stiffness, obstacle height, mesh/contact resolution, and solver substeps.
  Reports should show response direction, smoothness, saturation, and error
  drift; they are not fitting targets.
- **parameter/provenance hygiene** — every new tire, surface, soil, thermal,
  solver, and shader constant needs units, range, provenance, and reason.
  Sliders may override named state/presets, but they must never become the
  hidden source of force.

Focused additions from the calibration discussion, with calibration removed:

- **contact-patch truth before traction claims** — keep improving the
  deformed-patch model until pressure, load, camber, terrain shape, and tread
  support visibly change where the tire touches the ground. This means more
  than one aggregate force number: the model needs pressure bins, centre of
  pressure, contact normal distribution, local sliding/adhesion state, and
  footprint width/length from the actual tire geometry.
- **wheel-frame and vehicle-coupling truth** — all traction, braking, and
  steering forces must be produced in a documented wheel frame with explicit
  slip ratio, slip angle, camber, trail, and aligning-moment signs. Steering
  must rotate the wheel/tire frame and change patch forces; it must not move a
  car sideways through a viewer shortcut.
- **dynamic load and suspension effects** — tire force should react to
  transient normal load, chassis pitch/roll/yaw, suspension travel, and wheel
  spin state. Keep adding four-wheel scenarios where load transfer changes
  grip distribution, wheel spin, contact patch, and aligning moment through the
  same reusable tire APIs.
- **combined-slip and relaxation as first-class physics** — drive/brake slip,
  lateral slip, camber thrust, pneumatic trail, and aligning moment should
  share one smooth friction budget and have finite low-speed transient lag.
  More realism work should refine these native states, not replace them with a
  fitted curve.
- **road-state richness without fitted road labels** — ice, water film, loose
  aggregate, packed dirt, deformable soil, ruts, curbs, belts, and moving
  ground should enter through terrain/contact state: local normal support,
  shear strength, bristle damping, surface velocity, sinkage, compaction,
  rolling loss, heat, and wear. A material enum or scene-local grip scalar is
  not enough.
- **thermal, wear, and abuse history** — cold tires, overheating, carcass
  softening, worn tread, brake heat, rolling carcass heat, and abuse should
  come from energy/temperature/wear history. Keep those histories visible in
  telemetry and GPU parity rows; do not add "make it grippy/slippy" viewer
  switches.
- **structural tire depth** — continue replacing guardrails/proxies with
  force-producing tire structure: laminated sidewall/carcass behaviour,
  tread-belt support, bead/rim seat contact, cavity response, ply strain and
  sidewall fold resistance, and eventually a production-size implicit or
  matrix-free FEM solve. Keep every proxy labelled until the real structural
  solve exists.
- **solver and convergence evidence** — every realism step needs resolution
  and stability evidence: tread segments, tetra resolution, terrain triangles,
  heightfield resolution, solver substeps, XPBD/Newton iterations, energy
  budget, and CPU/GPU deltas. Coarse viewer settings are allowed only if their
  error direction is visible.
- **GPU-resident production path** — hot tire paths should move toward
  resident GPU ownership: vehicle state, wheel spin, suspension, brush rows,
  FEM nodes/hubs, terrain contact, thermal/wear, and soil feedback. Any
  CPU-only path must be named as debug/bridge/proxy with a concrete GPU
  follow-up in the parity checklist.
- **falsification UI instead of calibration UI** — the viewer should expose
  expected behaviour, active physics path, overlays, telemetry, parity deltas,
  and approximation rows for each tire scene. The goal is to make "this is
  wrong" precise and reproducible without adding dataset fitting controls.

Additional non-calibration items to keep from the realistic tire / racing-sim
discussion:

- **native rig matrix instead of external calibration** — keep the rig coverage
  that a calibration workflow would normally require, but use it only for
  broad sanity/falsification: static load/deflection, pressure footprint,
  pure longitudinal slip, pure lateral slip, combined slip, camber/trail,
  torque/brake/steer steps, speed sweeps, wet/ice transitions, moving belts,
  cleats/curbs, soil/ruts, thermal abuse, wear accumulation, and cooling
  recovery. Each case must be deterministic and runnable without measured
  tire data.
- **acceptance envelopes, not target curves** — every rig row should state a
  plain-language physical expectation plus wide numeric bounds: lower pressure
  grows the patch, higher load increases force sub-linearly, dry rolling loss
  rises with speed, wet speed reduces usable shear, ice caps force even with a
  large patch, combined slip shares one friction budget, heat/wear come from
  dissipated energy, and cooling should reduce thermal penalties without
  healing wear. These are pass/fail guards, not fitted tire curves.
- **sensitivity maps instead of fitted parameters** — expose pressure, load,
  speed, camber, temperature, water film, tread wear, soil stiffness, obstacle
  height, solver substeps, and mesh/contact resolution as named sweeps. The
  output should show direction, smoothness, saturation, and discontinuities so
  bad physics is visible before anyone tries real-world calibration later.
- **parameter observability and provenance** — group tire construction,
  rubber/brush, terrain, thermal/wear, soil, and solver constants into named
  unit-labelled presets with min/max ranges, provenance tags, and a reason for
  each hand-tuned or numerical-stability value. Unknown constants must not be
  hidden in scenes, viewer UI, or shader literals. **Initial reusable
  `TireParameterObservabilityReport` is now part of the standard
  non-calibration runner as `parameter_observability`: it combines preset
  governance, realism-source classification, and shader contract source guards
  for brush, FEM, thermal/wear, and transient GPU inputs so important
  force-producing constants remain named and auditable.**
- **ablation/debug modes for model layers** — add explicit toggles/reports that
  can isolate carcass/FEM, cavity pressure, tread bristles, terrain material,
  wet film, thermal/wear, soil/rut feedback, and transient relaxation. These
  modes are for debugging and approximation ledgers only; they must be labelled
  as such and must not become alternate fake force sources. **Initial reusable
  `TireLayerAblationDebugReport` is now part of the standard non-calibration
  runner as `layer_ablation_debug`: it isolates carcass/inflation patch
  response, cavity coupling bead load, tread-bristle force contribution,
  terrain material force change, wet-film force change, thermal/wear material
  response, soil/rut feedback activity, and transient relaxation lag without
  introducing alternate fitted force sources. The shared tire truth-label panel
  now renders the same report under `Layer ablation/debug gates`, explicitly
  labelling it as debug-only layer isolation rather than an alternate fake
  force source.**
- **replay artifacts for regressions** — every important rig/viewer case should
  be recordable as compact snapshots containing command inputs, terrain state,
  tire preset, physical state, solver knobs, force/moment output, contact-cell
  telemetry, heat/wear/energy accounting, CPU/GPU deltas, and known
  approximations. The snapshots should be small enough for tests/CI and for
  a broken viewer example to be reproduced exactly. **Initial reusable
  `TireReplayArtifactCoverageReport` now audits the default non-calibration
  runner's snapshot artifacts: row/header shape, row byte size, command input
  fields, terrain state, provenance labels, force/moment telemetry, patch
  telemetry, thermal/wear state rows, FEM solver-budget rows, CPU/GPU delta
  columns, and known-approximation ledger availability. The shared tire-rig
  replay panel now exposes the same `Replay artifact coverage` rows beside the
  selected replay, so reproducibility gaps are visible in the viewer rather
  than only in the text report.**
- **physics and performance gates together** — production readiness needs both:
  physics gates for sign, continuity, energy, friction-budget use, finite
  low-speed behaviour, and construction sanity; and runtime gates for many
  wheels, resident GPU buffers, dispatch count, readback cost, LOD/convergence,
  and deterministic CPU/GPU parity tolerances. **Initial reusable
  `TireRuntimeProductionGateReport` is now part of the standard
  non-calibration runner as `runtime_production_gate`: it ties broad physics
  sanity to GPU parity rows, resident-buffer ownership, many-wheel/batch
  evidence, fused/atlas dispatch evidence, readback/no-readback seams,
  convergence/LOD report pass status, and CPU-only-debug accounting.**
- **viewer as a falsification cockpit** — the viewer should present those rig
  cases as replay buttons, truth labels, overlays, and unit-labelled readouts.
  If a user says "this tire does not deflate", "ice is not slippery", or
  "steering is fake", the UI should point to the exact expectation, telemetry,
  active physics path, and approximation ledger row that failed. **The shared
  tire truth-label panel now also surfaces the reusable
  `TireParameterObservabilityReport` and `TireRuntimeProductionGateReport`, so
  each tire scene exposes no-hidden-constants shader/preset checks beside the
  physics-plus-GPU production gate instead of leaving those reports only in
  crate text output.**
- **frame and sign convention audit** — document the exact wheel frame,
  terrain frame, slip-ratio, slip-angle, camber, pneumatic-trail, aligning
  moment, and moving-ground sign conventions used by the native tire APIs.
  Add regression checks so steering cannot silently become "move the whole car
  sideways while the wheels still point straight". **Initial reusable
  `TireFrameConventionReport` is now part of the standard non-calibration
  runner as `frame_sign_convention`: it checks `heading`, `normal`,
  `side = heading × normal`, positive drive/brake slip force signs, lateral
  side-slip force sign, positive-camber convention, pneumatic-trail moment
  sign, moving-ground work sign, and shared friction-budget accounting.**
  The shared tire truth-label panel now exposes the same
  `Frame/sign convention audit`, including the documented `heading × terrain
  normal` side-axis convention, force-sign rows, trail moment sign, moving-
  ground work sign, and friction-budget overrun metric so fake steering/sign
  regressions are visible from the viewer.
- **low-speed robustness** — keep force, slip, relaxation, and brush updates
  finite through launch, lockup, reversal, near-zero rolling speed, and
  transition from static adhesion to sliding. This is model sanity, not fitted
  data matching.
- **contact-patch convergence** — track how pressure bins, patch length/width,
  force, aligning moment, and heat/wear change as tread segments, FEM
  resolution, terrain triangle density, heightfield resolution, and solver
  substeps change. Coarse results may be acceptable for the viewer, but their
  error direction must be visible. **Initial reusable
  `TireContactPatchConvergenceReport` is now part of the standard
  non-calibration runner as `contact_patch_convergence`: it compares sampled
  brush contact across circumferential segment counts and gates finite outputs,
  active contact-cell refinement, normal-load error, patch-length stability,
  world-force stability, aligning-moment stability, peak-pressure stability,
  and pressure-bin load conservation. Initial FEM solver-substep convergence is
  now also part of the standard runner as `fem_substep_convergence` via
  `TetraFemSubstepConvergenceReport`: it advances the same settled tetra tire
  over the same interval with increasingly fine explicit substeps and gates
  hub-height drift, deformed tread patch drift, FEM-derived contact-force
  drift, tetra-volume error, and distributed contact preservation.
  Initial heightfield-resolution convergence is now emitted as
  `heightfield_resolution_convergence` via
  `TireHeightfieldResolutionConvergenceReport`: it samples the same analytic
  bump on coarse-to-fine heightfield grids and gates terrain height/normal,
  patch length, contact force, and multi-cell contact stability. Initial
  triangle-density convergence is now emitted as
  `triangle_density_convergence` via
  `TireTriangleDensityConvergenceReport`: it rebuilds the same two-material
  triangle road at coarse-to-fine subdivision counts and gates height/normal,
  material span, seam manifold support, patch length, and brush-force
  stability. Initial heat/wear convergence is now emitted as
  `thermal_wear_convergence` via `TireThermalWearConvergenceReport`: it runs
  the same combined-slip sample across coarse-to-fine brush segment counts and
  gates dissipated power, surface temperature, tread-wear depth, active
  sliding, and friction-budget stability. The shared tire truth-label panel now
  renders these five reports together under `Resolution/convergence gates`,
  exposing contact-segment, FEM-substep, heightfield-grid, triangle-density,
  and thermal/wear resolution deltas so coarse viewer settings show their error
  direction instead of silently changing tire behavior.**
- **construction sanity gates** — keep explicit checks for cavity pressure,
  bead/rim stability, belt stiffness, sidewall ply/bias behaviour, tread block
  support, and loaded-radius/footprint shape before trusting drive/brake
  behavior. **Initial reusable `TetraConstructionSanityReport` is now part of
  the standard non-calibration runner as `construction_sanity`: it checks
  high-vs-low pressure setup, bounded loaded radius, low-pressure footprint
  growth, multi-cell/widthwise tread support, tread-belt edge stretch,
  sidewall axial bulge bounds, bead mount drift, and sidewall fold drift.**
  The shared tire truth-label panel now exposes those `Construction sanity
  gates` directly, including loaded radius, low/high pressure patch length,
  contact-cell support, belt stretch, sidewall bulge, bead drift, and sidewall
  fold bounds so static tire-shape failures are visible before traction claims.
  `TetraSidewallDihedralXpbdEffectReport` now also emits
  `sidewall_dihedral_xpbd_effect`: it deliberately folds sidewall quads,
  applies the structural dihedral-XPBD projection, and gates measurable fold
  reduction so the current sidewall-bend proxy is falsifiable while the full
  laminated shell solve remains future work.
  `TetraCavityCouplingEffectReport` now emits `cavity_coupling_effect`: it
  sweeps zero/nominal/high `cavity_carcass_coupling` and gates that the named
  pressure-to-carcass proxy visibly removes, applies, and increases cavity
  bead load while bead drift remains bounded.
- **road-condition state, not surface labels only** — wetness, ice, loose
  aggregate, soil compaction, rut depth, obstacle geometry, temperature, and
  wear should be physical state that affects normal support, shear, rolling
  loss, and heat generation. A material enum by itself is not enough.
  **Initial reusable `TireRoadConditionStateReport` is now part of the
  standard non-calibration runner as `road_condition_state`: it checks
  water-film μ/force reduction, standing-water rolling loss, ice force capping,
  moving-ground telemetry/work, hot+worn state-derived μ loss, worn-tread
  wet-film penalty, and soft-soil rolling loss. The shared tire truth-label
  panel now exposes the same `Road-condition state gates`, explicitly labelling
  wet film, ice, moving ground, hot/worn rubber, worn tread, and soft soil as
  state-derived effects rather than scene-local grip labels. `TireSoilNormalShearCouplingReport`
  now also emits `soil_normal_shear_coupling` in the standard runner: it uses
  the native Bekker `DeformableTerrain::apply_tire_contact` path to prove that
  normal load creates support/sinkage while driven slip adds shear rutting and
  rolling-loss feedback through terrain state.**
- **vehicle-coupled falsification scenes** — keep heavy-load, low-friction,
  steering/camber, braking, obstacle, and off-road vehicle cases that exercise
  the same reusable tire model under changing normal load and suspension/chassis
  motion. These are sanity/falsification scenes, not calibration scenes.
  **`tire_vehicle_sanity_scenarios` now includes four-wheel heavy-load,
  low-friction driven, braking, steering/camber, obstacle/curb, and soft-soil
  off-road cases, all sourced from reusable tire APIs and carried by the
  standard non-calibration runner. `TireVehicleDynamicLoadTransferReport` now
  adds the first explicit dynamic load-transfer/suspension falsification case
  as `vehicle_dynamic_load_transfer`: braking shifts load to the front axle,
  acceleration shifts load to the rear axle, total four-wheel load is
  conserved, contact-patch length and tire force change through the native
  brush/load-sensitive law, and per-corner suspension travel follows the same
  loads instead of a viewer-only chassis shortcut.**
- **honest realism labels** — every realistic-tire UI/report must say whether a
  result is native FEM/brush/terrain physics, a diagnostic reference shape, a
  hand-tuned preset, a numerical stabilizer, or a visualization helper.
  **Initial reusable `TireRealismSourceClassificationReport` is now part of
  the standard non-calibration runner as `realism_source_classification`: it
  separates native force-producing physics from diagnostic references,
  hand-tuned presets, numerical stabilizers, debug telemetry, and
  visualization-only helpers, and checks that diagnostics/visuals are not
  labelled as force sources. The shared tire truth-label panel now surfaces
  this same report in a collapsed `Realism source classification` section so
  the viewer explicitly labels force-producing physics versus diagnostics and
  visualization helpers.**

## Non-calibration points to keep from the tire-model discussion

We are **not** doing calibration now, but the useful production/realism ideas
from real tire and racing-sim workflows still belong in this plan. Treat these
as native model-quality work, not as coefficient fitting:

- **separate structural tire from road-force law** — keep the carcass/FEM,
  inflation/cavity, tread bristles, terrain material, thermal/wear state, and
  wheel/chassis dynamics as explicit layers with clear inputs/outputs. Do not
  bury force multipliers in viewer code.
- **static shape truth first** — before trusting drive/brake forces, the tire
  must visibly and numerically show believable loaded radius, footprint length,
  footprint width, pressure distribution, bead/rim stability, and sidewall /
  belt deformation for over-inflated, nominal, and soft tires.
- **force-shape sanity instead of fitted curves** — longitudinal, lateral, and
  combined-slip force should build smoothly, saturate continuously, trade one
  axis against the other through a shared friction budget, and stay finite
  through zero speed, direction changes, and terrain transitions.
- **aligning/steering comes from patch geometry** — camber thrust, contact
  centre shift, pneumatic trail, and aligning moment should come from the
  distributed patch/bristle forces. Steering must rotate the wheel/tire frame;
  it must not fake a vehicle yaw drift while all wheel geometry stays straight.
- **transient response is required for realism** — torque, brake, and steer
  steps should produce delayed tread/belt/carcass force buildup. A model that
  instantly teleports from one steady-state force to another is not acceptable
  even if its peak force looks plausible.
- **surface state is part of physics** — asphalt, ice, wet film, moving ground,
  packed dirt, loose soil, ruts, and obstacles should affect normal support,
  shear strength, slip velocity, heat, wear, and rolling loss through the same
  terrain/contact APIs used by the tire model.
- **thermal/wear/condition must be state-derived** — cold grip, overheating,
  abuse, tread wear, and wet/loose-surface degradation should come from energy,
  temperature, film, and wear state. Do not add hidden "make grip lower" scene
  switches.
- **repeatable falsification beats eyeballing** — every tire scene should have
  an internal rig/replay case and a plain-language expected-behavior label so a
  broken tire can be reported as "this expectation failed" instead of "it looks
  weird."
- **telemetry must make cheating obvious** — expose unit-labelled forces,
  moments, contact cells, pressure bins, friction budget, bristle state,
  slip/adhesion fraction, heat/wear, energy loss, solver budget, and CPU/GPU
  deltas anywhere the viewer claims the tire is realistic.
- **GPU-first for hot paths** — per-contact brush, FEM node stepping, terrain
  lookup, deformable-soil feedback, and many-wheel batching should either have
  GPU implementations/tests or be explicitly marked as temporary CPU/debug
  paths in the approximation ledger.
- **known approximations stay visible** — explicit FEM integration, coarse tetra
  resolution, structural sidewall shortcuts, simplified bead sealing, finite
  triangle approximations, Bekker-only or partial DEM/MPM soil, and any CPU-only
  path must remain documented with consequence and next step, and the tire
  viewer should surface that ledger instead of hiding it in code comments.

Concrete plan items from those points:

- add/keep tire-bench cases for static footprint, pressure distribution,
  force-slip shape, combined-slip tradeoff, camber/trail sign, torque/brake/
  steer steps, moving ground, wet/ice bands, curb/cleat contact, thermal abuse,
  and soil/rut feedback;
- wire those same cases into viewer truth labels/replay panels instead of
  relying on ad-hoc example-specific behavior;
- keep native brush/FEM/terrain outputs as the only force source; optional
  Pacejka/TMeasy/racing-sim curves may only be used later as diagnostic shape
  comparators, never as the model being driven by fitted constants;
- require every new viewer tire control to declare whether it is a physical
  state, preset override, numerical-stability knob, debug toggle, or
  visualization-only helper; **initial `TireViewerControlMetadata` inventory
  is now shown in the shared tire truth-label panel and tested for all tire
  viewer scenes.**
- require every high-frequency tire feature to update both the GPU parity
  checklist and the known-approximation ledger when it lands.

## Calibration explicitly deferred, but keep these model-quality items

For now, do **not** add an external calibration pipeline. Do not spend this pass
on measured data ingestion, optimizer loops, Pacejka/TMeasy coefficient
fitting, or "match this real tire" workflows. The useful pieces from that
discussion still need to exist in the project, but as native physics and
debugging infrastructure:

- **structural truth before force tuning** — loaded radius, sidewall bulge,
  bead stability, tread-belt shape, pressure distribution, and contact width
  must be inspectable before any force curve is trusted.
- **native force-shape diagnostics** — pure slip, lateral slip, combined slip,
  camber, trail, and relaxation responses should be checked for smoothness,
  sign, saturation, and finite low-speed behavior without fitting coefficients.
- **repeatable internal rigs** — keep deterministic rigs/replays for pressure,
  load, speed, slip, camber, wet/ice, soil, curb/cleat, thermal abuse, wear,
  moving ground, and steering/torque/brake step inputs. **The default
  non-calibration runner now includes a `moving_ground_belt` replay snapshot
  with terrain surface velocity and positive moving-ground work telemetry.**
- **qualitative acceptance envelopes** — define broad pass/fail expectations
  such as "ice caps force below asphalt", "lower pressure grows the patch",
  "higher load increases force sub-linearly", "sliding heats and wears", and
  "step responses lag rather than teleport".
- **parameter provenance instead of hidden tuning** — every tire, surface,
  thermal, soil, and solver constant must be named, unit-labelled, range-audited,
  and tagged as physical, hand-tuned, numerical-stability, debug, or
  visualization-only.
- **state-derived condition effects** — cold grip, overheating, abuse, tread
  wear, wet film, and loose-surface degradation must come from thermal/wear/
  film/soil state and telemetry, not from scene-specific grip multipliers.
- **full force and energy accounting** — expose forces, moments, contact-cell
  pressure, bristle state, friction-budget use, dissipated power, heat/wear,
  rolling loss, moving-ground work, and solver budget so unrealistic behavior is
  visible in numbers and not only by eye.
- **viewer falsification, not calibration UI** — tire scenes should show truth
  labels, expected behavior, active physics path, approximations, contact
  overlays, pressure bins, heat/wear, rut/sinkage, and CPU/GPU deltas. They
  should not expose "fit to data" controls in this pass.
- **GPU-first parity rule** — hot paths for brush contact, FEM node stepping,
  FEM contact feedback/reduction, heightfield/material lookup, deformable-soil
  feedback, DEM/MPM coupling, and many-wheel batching need GPU implementations
  or explicit CPU-only/debug ledger entries.
- **reference curves only later as diagnostics** — if Magic Formula, TMeasy, or
  racing-sim style curves are ever added, they may only compare broad shape and
  sign against the native FEM/brush output. They must not replace the tire force
  source or introduce fitted constants in this pass. **`Scene::TireModelLab`
  now plots the native `RealBrushTireModel` adapter beside `RigidTire`,
  `TmSimpleTire`, and `Pac89Tire`, with truth-label/audit text distinguishing
  the native real-brush force law from empirical diagnostic reference curves.**

## Deferred-calibration carry-over backlog

Calibration is skipped for this pass, but the useful non-fitting work from the
calibration / racing-sim discussion should still land in native `molla`
physics. Treat this as the backlog for making the tire model easier to falsify,
debug, and eventually compare against real data without adding a fitter now:

- **bench runners, not fit loops** — keep adding deterministic load/deflection,
  pressure, slip, camber, steer, torque, brake, wet/ice, dirt/soil, cleat,
  moving-belt, thermal, wear, and cooling rigs. They should emit snapshots and
  qualitative envelopes, never optimizer targets.
- **force-shape map** — build broad native maps for longitudinal force, lateral
  force, aligning moment, pneumatic trail, combined-slip saturation, low-speed
  launch/lockup/reversal, and relaxation lag. The acceptance criteria are sign,
  smoothness, monotonic regions, bounded saturation, and continuity, not
  matching a Pacejka/TMeasy curve.
- **structural tire map** — keep instrumenting loaded radius, sidewall bulge,
  footprint width/length, pressure distribution, bead/rim drift, cavity
  response, belt support, ply strain, and sidewall fold resistance across
  pressure/load/camber. Traction claims should point back to these structural
  diagnostics.
- **road-state and terrain map** — asphalt, wet film, ice, packed dirt, loose
  soil, ruts, cleats, curbs, and moving ground must change support/shear/heat/
  wear through explicit terrain/contact state. Scene-local "grip multiplier"
  fixes are not acceptable.
- **transient state map** — torque, brake, steer, load-transfer, suspension,
  wheel-spin, and belt/carcass/tread relaxation should have finite memory and
  step responses. Viewer examples should show the lag/state variables, not only
  steady-state forces.
- **thermal/wear/abuse map** — cold running, warm-up, overheating, carcass
  softening, sliding abuse, tread wear, cooling recovery, and irreversible wear
  must come from dissipated energy and material state. Cooling may recover
  temperature penalties; it must not silently heal wear.
- **parameter and provenance audit** — every tire, rubber, terrain, soil,
  thermal, solver, and shader constant needs units, range, provenance, and a
  reason. If a value is hand-tuned or numerical-stability-only, the viewer and
  reports should say so.
- **viewer falsification workflow** — every tire scene should show the active
  physics path, expected behavior, physical controls, debug controls,
  approximation ledger, contact/pressure/heat overlays, replay snapshot, and
  CPU/GPU deltas where available.
- **GPU-first parity** — hot paths for FEM stepping, brush/contact, terrain
  lookup, thermal/wear, deformable soil, transient relaxation, and many-wheel
  batching should either have shader tests/parity rows or an explicit
  CPU-only/debug entry with consequence and next step.
- **future calibration readiness without doing calibration** — snapshot formats,
  unit labels, state histories, and broad envelopes should be stable enough
  that real data can be compared later, but this pass must not ingest datasets
  or tune coefficients to them.

## Current implementation status

- `crates/molla-vehicle/src/real_tire.rs` now contains the reusable tire work:
  - pressure/material brush contact (`evaluate_brush_tire`);
  - legacy dynamic shell-style carcass (`DynamicTireCarcass`);
  - true volumetric tetrahedral tire FEM (`TetraTireFem`).
- `Scene::TireTraction` uses the brush contact model for pressure/material
  dependent force and slip. The reusable brush API now resolves combined
  longitudinal + lateral slip, reports lateral force, slip angle and a
  self-aligning yaw moment, and caps the vector force in one friction cone.
- `Scene::TireDynamics` now uses `TetraTireFem`: an annular tire volume with
  circumferential × radial × width nodes, hexahedral parameter cells split
  into tetrahedra, finite-deformation compressible Neo-Hookean internal rubber
  forces, bead/rim coupling, zero-gap floor contact, brush/Coulomb tread
  friction, and a sealed cavity gas law.
- The tetra mesh now has first-pass production tire structure:
  - softer sidewall rubber;
  - stiffer tread/belt rubber;
  - stiff bead/rim-adjacent elements;
  - carcass fallback material for the remaining body;
  - anisotropic Holzapfel-style cord/belt fibres with explicit crossed
    ply-angle families for tread/belt and sidewall regions.
- Direct pressure traction on the coarse exterior tread remains opt-in/legacy.
  Inflation now uses an internal cavity pressure/volume path: `P · V^γ =
  const`, cavity-surface forces on the bead-side surface, and a small
  pressure-proxy tangent stiffening term for explicit-step robustness.
- `crates/molla-kernels/shaders/tire_fem.wgsl` now provides the first GPU
  tire-FEM node-step kernel. The Rust wrapper is `TireFemGpuKernel`, with raw
  snapshot export from `TetraTireFem::gpu_snapshot()`. It uses per-element
  material Lamé parameters, anisotropic fibre data, the Neo-Hookean update,
  CSR node/tet adjacency, persistent ping-pong node buffers, and a GPU
  hub-reduction kernel. The GPU test lane is:
  `make test-gpu P=molla-kernels T=tire_fem_gpu`.
- `TireFemGpuBatch` adds the first persistent multi-tire GPU owner: callers
  can upload several tire FEM meshes once, step them through one batch API,
  update hub states, and read back all outputs without ad-hoc per-wheel
  persistent state management. This is currently a host-side heterogeneous
  batch loop over persistent meshes; `tire_fem_gpu_batch_dispatch_plan` now
  records fused-atlas tire/node/tet/tread/bead/workgroup/storage counts as the
  packing contract for the single-dispatch atlas kernel. The companion
  `tire_fem_gpu_batch_atlas_layout` / `TireFemGpuBatchAtlasLayout` now records
  contiguous per-tire node/tet/tread/bead/workgroup offsets, so heterogeneous
  batch packing can be regression-tested independently from runtime ownership.
  `TireFemGpuBatchAtlasBuffers` now builds the actual packed host atlas for
  the fused dispatch, rewriting tet/bead/tread/CSR/XPBD structural indices
  from per-tire local space into atlas space while keeping per-tire parameter
  rows explicit. The packed atlas
  also carries node/tet/bead/tread tire-row routing arrays so the fused shader
  can load the correct per-tire params/hub row and reduce diagnostics
  back to the owning tire. `TireFemGpuBatchAtlasParamRow` now mirrors the
  existing single-tire uniform layout as shader-ready storage rows, preserving
  counts, XPBD counts, hub, pressure, damping, cavity, and semi-implicit
  stiffness parameters per tire without falling back to Rust-only params.
  `TireFemGpuBatchAtlasOffsetRow` now adds shader-ready node/tet/tread/bead
  and XPBD edge/dihedral offset/count rows, and the dispatch plan tracks total
  XPBD structural constraint counts so fused kernels can route both elastic and
  structural passes through the same atlas contract. The packed atlas now
  self-validates all buffer/routing/CSR/XPBD/row lengths and reports total
  storage bytes, so malformed atlases fail before a fused GPU dispatch can
  consume them. It also provides `split_outputs` to validate and slice
  atlas-wide position/velocity/force readback rows back into per-tire
  `TireFemGpuOutput` records, preserving the existing batch API boundary after
  the one-dispatch kernel writes one packed output buffer. `extract_input`
  now reconstructs per-tire `TireFemGpuInput` rows from the atlas and
  localizes all node/tet/bead/tread/CSR/XPBD indices again, giving the fused
  dispatch path a lossless CPU/reference round-trip for parity debugging.
  `join_outputs` and `split_output_rows` now convert between per-tire
  `TireFemGpuOutput` rows and atlas-wide output rows, and
  `dispatch_atlas_reference_loop` executes the packed atlas through the
  existing per-tire GPU kernel as a reference lane for the fused shader.
  `tire_fem_gpu_batch_atlas_binding_plan` now freezes the fused shader storage
  binding contract: bindings `0..=37` mirror the single-tire FEM shader and
  bindings `38..=43` add node/tet/bead/tread tire ownership plus param/offset
  rows with validated byte and row counts. `tire_fem_atlas.wgsl` now carries
  the matching WGSL binding/entry-point contract (`step_nodes_atlas_contract`)
  and ports the single-tire physical node update onto the packed atlas:
  per-node CSR Neo-Hookean force accumulation, crossed fibre families,
  bead/rim coupling, cavity pressure load, legacy tread pressure traction,
  semi-implicit stiffness damping, volume/bead/edge/dihedral XPBD projections,
  and zero-gap floor friction all run through atlas ownership/param/offset
  rows. `TireFemGpuBatchAtlasDispatchContract` now records the shader label,
  entry point, workgroup size, workgroup count, total node count, and binding
  count that the fused pipeline uses. `TireFemGpuKernel` now builds that
  one-dispatch atlas compute pipeline and `dispatch_atlas_contract_blocking`
  uploads all 44 storage bindings, launches `step_nodes_atlas_contract` once
  over the packed node atlas, and reads back packed position/velocity/force
  rows. `dispatch_batch_atlas_blocking` is now the first high-level fused batch
  entry point: it packs `TireFemGpuInput` rows into an atlas, runs the fused
  atlas shader once, and splits the packed readback back into per-tire
  `TireFemGpuOutput` rows. `TireFemGpuAtlasBatch` now adds the first resident
  fused-atlas owner: topology/material/routing buffers stay GPU-resident,
  position/velocity buffers ping-pong through two atlas bind groups, and
  `step_resident_atlas_batch` advances all tire nodes with one fused dispatch
  before `readback_resident_atlas_batch` splits outputs by tire. The companion
  `hub_reduce_atlas` entry and `step_resident_atlas_hubs` now advance all
  packed hub rows with one workgroup per tire, so the resident atlas has a
  fused GPU equivalent of the older per-tire `step_hub_persistent` loop. It
  also has
  `upload_resident_atlas_batch_state` / `upload_resident_atlas_rows`, so
  viewer/vehicle code can reseed the current ping-pong state after CPU resets,
  fallback steps, or editor manipulation without rebuilding the resident
  topology atlas. `upload_resident_atlas_tire_state` now adds the matching
  single-tire reseed path, writing only one tire's packed node slice so
  viewer/editor resets do not disturb the rest of a resident multi-tire atlas.
  `upload_resident_atlas_hub_rows`, `upload_resident_atlas_tire_hub`, and
  `readback_resident_atlas_hub_rows` add the matching packed hub-state bridge.
  Scene 120 now uses this resident owner by default; the remaining viewer work
  is live vehicle adoption, not the pressure/shape comparison path.
  The older host-loop `TireFemGpuBatch` docs now point callers at
  `TireFemGpuAtlasBatch` when they can keep one packed fused-atlas owner,
  rather than implying the atlas path is still future work.
  The reusable
  `tire_gpu_parity_checklist` now records this resident fused-atlas batch path
  under `tetra_fem_persistent_batch`, including proof strings and the remaining
  vehicle runtime follow-up; `molla-kernels` and the top-level `molla::kernels`
  facade also re-export `TireFemGpuAtlasBatch` so viewer/vehicle code can
  adopt the owner directly. `Scene::TireDynamics` now prefers this resident
  atlas owner by default: all five tire FEMs share one atlas, node updates run
  through `step_resident_atlas_batch`, hub updates run through
  `step_resident_atlas_hubs`, and the old per-tire persistent mesh path remains
  a right-rail debug/fallback toggle. The combined resident vehicle+FEM owner
  path now also reads back explicit resident chassis, vertical, wheel-spin, and
  suspension validation state into the Scene 120 right rail, so live adoption
  is not only a hidden dispatch path. The remaining scaling work is vehicle
  runtime adoption beyond the viewer examples, plus deeper live validation.
- `Scene::TireDynamics` defaults to the GPU resident-atlas tire-FEM path
  through the viewer right rail; CPU is retained as a fallback/debug toggle.
  The GPU path keeps the FEM nodes and 3-DoF hub rows on resident atlas buffers
  and reads back once per frame for drawing.
- `crates/molla-kernels/shaders/tire_brush.wgsl` and `TireBrushGpuKernel`
  provide a GPU brush/contact parity lane for batched wheels, including
  combined-slip lateral force, aligning moment, camber thrust, pneumatic-trail
  diagnostics, dissipated power, heightfield sampling, and optional
  heightfield friction-map/material lookup. The GPU brush path also samples
  local heightfield normals and returns a world-force vector accumulated from
  per-cell normals, matching CPU brush contact on ramps/bumps.
  `TireBrushGpuBatch` adds a persistent-buffer dispatch path so repeated brush
  solves do not need to allocate wheel/output/terrain buffers every frame.
- `Scene::TireTraction` has a dirt→ice slipperiness slider so wheel spin is
  easy to force and inspect. It now defaults to `TireBrushGpuKernel` for the
  per-wheel brush contact and exposes CPU only as a fallback/debug toggle. The
  user car now uses per-wheel transient relaxation state and a camber slider;
  the viewer displays raw vs effective slip, camber, contact centre and
  pneumatic trail. It also has toggleable tire debug overlays for the user car:
  world force arrows, lateral-force arrows, contact-centre / force-resultant
  markers, pneumatic-trail segment, explicit aligning/chassis yaw-moment
  glyphs, and patch-bound/normal markers. When the GPU brush path is active,
  the right rail also shows CPU-vs-GPU deltas for force, patch length, slip,
  friction-budget, and aligning-moment diagnostics.
- `TireThermalLayers` adds a cheap four-node surface/tread/carcass/rim thermal
  model. Sliding power heats the surface/tread, rolling and flex heat the
  carcass, optional brake heat enters the rim, and road/air/conduction sinks
  cool the layers. It drives thermal μ, carcass pressure/stiffness factor,
  temperature-aware wear, and a state-derived `TireConditionReport` that
  exposes cold/hot/overheated/abused tire condition without viewer-only flags.
  `TireThermalGpuKernel` / `TireThermalGpuBatch` / `tire_thermal.wgsl` now add
  a reusable persistent-buffer GPU parity lane for the same four-node thermal
  update plus tread wear-depth integration, including rubber peak, thermal μ,
  carcass pressure, wear ratio, worn-tread μ, and wear-temperature diagnostics.
  CPU remains the readable reference; scene 122 now uses that persistent GPU
  thermal/wear batch by default and converts the result back into
  `TireThermalLayers` / `TireWearState` before deriving condition-material
  response. The same scene now also dispatches a persistent GPU thermal/wear
  batch across loaded internal rig replay snapshots, using reusable
  `tire_thermal_build_rig_replay_batch` artifacts from `molla-kernels`, so
  many-wheel/replay condition telemetry exercises the GPU-resident path
  instead of staying a viewer-only CPU report. `TireThermalConditionSanityReport`
  now turns heat/cool replay runs into non-calibration pass/fail metrics for
  peak/final surface temperature, tread temperature, pressure factor, μ factor,
  and non-healing wear.
- `TireConditionMaterialResponse` turns the condition report into live tire
  parameters plus contact material coefficients: state-derived μ, pressure,
  carcass stiffness, tread shear, bristle damping, rolling resistance, and
  explicit cold/overheated/softened/worn/abused failure flags. The thermal
  wear rig now uses this response instead of manually multiplying μ only.
- `SurfaceFilmState` adds a reusable wet/thin-film surface modifier. Water
  depth, speed, and remaining tread depth reduce static/dynamic shear, lower
  bristle damping, reduce cohesion, and add water-displacement rolling loss
  before the brush/FEM contact law runs.
- `Scene::TireRealistic` uses the layered thermal model: the tire tint follows
  surface temperature, the right rail shows surface/tread/carcass/rim
  temperatures, the road panel exposes a water-film slider, and the brush
  pressure/friction/wear update uses wet-film plus thermal/condition feedback
  instead of the old single lumped temperature. The viewer now displays the
  condition label, abuse index, and state-derived wear-rate multiplier.
- The brush contact law now has first-pass load/pressure sensitivity and
  smooth combined-slip saturation. Local normal pressure adjusts μ and tread
  stiffness sub-linearly, inflation pressure shifts tread/carcass shear
  stiffness on CPU and GPU, longitudinal/lateral forces share a smooth
  load-sensitive friction budget, and CPU/GPU outputs report adhesion/sliding
  fractions, available/consumed friction budget, and effective longitudinal /
  cornering stiffness. `RealTireParams::structural_lateral_bias_per_normal`
  now adds a first conicity / ply-steer structural lateral-bias term to the
  same brush law on CPU and GPU, so tire pull comes from the tire preset rather
  than viewer-side steering drift.
- `TetraTireFem::evaluate_tread_contact` provides the first reusable
  FEM-derived brush/contact path. It builds contact cells from the actual
  deformed tread nodes, distributes hub load over active tread-node contact,
  includes node-relative slip velocity, and returns the same wheel-level
  `RealTireContact` diagnostics as the ideal sampled brush API.
- `Scene::TireRealistic` now uses the FEM-derived tread-node contact for the
  live force/heat/wear readout, while retaining ideal brush sweeps as reference
  charts.
- `Terrain` now supports per-location friction (`friction_at`), surface
  velocity (`surface_velocity`), and water-film depth
  (`water_film_depth`). `HeightfieldTerrain` can carry row-major friction,
  water-film, and surface-velocity maps, `TriangleMeshTerrain` provides triangle-mesh terrain with
  per-triangle friction, vertical height queries, and finite-triangle
  closest-surface signed distance/normal for CPU FEM projection near mesh
  edges/plates, and brush/tread contact samples local terrain friction, water
  film, and ground velocity at each contact cell so ice/asphalt bands, wet
  patches, and moving ground affect slip/traction physically. The GPU brush
  kernel mirrors the heightfield friction-map lookup for batched wheel contact,
  and the GPU FEM contact diagnostics/feedback path now samples heightfield
  water-film maps for tire-node contact; GPU FEM feedback also subtracts the
  uploaded terrain surface velocity before computing tread slip and can now
  sample an optional row-major `surface_velocity_map` per tread node, so
  moving belts / flowing terrain can vary across the patch without a
  viewer-only slip multiplier.
- CPU brush and FEM-derived tread contact now store the local terrain normal
  on each `TireContactCell` and accumulate wheel-level force from per-cell
  normals, so ramps/bumps contribute the correct normal-force direction instead
  of assuming one hub-centre up vector for the whole patch.
- `TetraTireFem::step_on_terrain` feeds terrain height, friction, and surface
  velocity into the CPU FEM contact projection. This gives heightfield/moving-
  ground contact feedback directly into tread node positions/velocities; the
  default `step()` remains the flat-terrain compatibility wrapper.
- `TriangleMeshTerrain` now builds a reusable CPU triangle BVH over
  per-triangle AABBs and uses it for height and closest finite-triangle
  signed-distance queries, plus a multi-triangle contact-manifold query for
  sphere/tread samples that straddle seams, curbs, and plates. CPU
  `TetraTireFem` tread-node projection now consumes that manifold radius so
  seam/plate contact can start before point penetration collapses to one
  triangle. GPU FEM contact diagnostics/feedback now also receive a compact
  per-triangle XZ bounding sphere and a GPU-consumable packed triangle BVH
  atlas/traversal path so far finite-triangle patches can be rejected before
  closest-point contact math. GPU FEM reduction/feedback now also accumulates
  weighted multi-triangle manifold samples inside the contact band so seam
  normals and friction are blended instead of collapsing to one nearest
  triangle. A larger packed-atlas row-layout stress test now verifies many
  finite triangles, bounded BVH leaves, packed index uniqueness, and finite
  node bounds. `TireFemContactFeedbackOutput::patch_accounting()` now also
  summarizes per-node force rows and bristle deflections so seam/triangle
  contact can expose whether force and bristle state are distributed across
  the patch. An ignored large triangle-BVH hardware dispatch stress lane now
  exercises the packed traversal through `make test-gpu P=molla-kernels
  T=tire_fem_gpu
  A='tire_fem_gpu_contact_reduction_large_triangle_bvh_dispatch_stress --nocapture'`.
- `TetraTireFem::apply_tread_contact_wrench` takes the FEM-derived wheel-level
  contact cells from actual tread nodes and applies the distributed normal,
  longitudinal and lateral contact wrench back into those FEM node velocities,
  so the brush/contact readout can feed the carcass instead of remaining
  diagnostics-only. It also advances the tire's persistent per-tread-node
  bristle state from the wheel-level slip before applying the force.
- `TireFemContactFeedbackKernel` adds the first GPU tread-contact feedback
  pass. It owns each FEM node, identifies tread nodes from `tread_ids`,
  advances a per-tread bristle buffer, computes a Coulomb-capped brush force,
  and writes the resulting impulse back into node velocities plus a force
  readback buffer. It now samples optional heightfield height, local normal,
  friction-map material data, water-film maps, and built-in sphere/box SDF
  primitives for feedback, and accounts for moving terrain surface velocity
  when computing relative tread slip. It also samples an optional spatial
  `surface_velocity_map` plus optional deformable terrain sinkage/rut maps and
  applies Bekker-style rolling loss from rut depth; `TireBekkerGpuKernel` /
  `tire_bekker.wgsl` now adds the first GPU Bekker height/sinkage writeback
  from tire contact-cell pressure and shear.
- `TireBekkerFemTerrainOwner` adds the first reusable ownership bridge between
  GPU Bekker writeback and GPU FEM contact. It owns the Bekker height/sinkage
  maps, applies contact-cell pressure/shear via CPU reference or GPU kernel,
  exports a `TireFemContactTerrain` with `sinkage_map`, and now exposes
  `apply_contacts_gpu_resident` / `TireBekkerGpuResidentTerrain` so FEM contact
  feedback can consume GPU-written height/sinkage plus material/water-film maps
  directly. It now also carries optional spatial `surface_velocity_map` rows
  through CPU and GPU-resident terrain export, so Bekker-backed FEM feedback no
  longer has to collapse moving soil/belts to one uniform velocity when a map
  is available. The CPU mirror remains an explicit viewer/reset telemetry seam.
- `Scene::TireRealistic` now has the first live viewer hookup for that owner:
  the FEM-derived contact cells write pressure/shear into a live Bekker rut map
  through the CPU or GPU owner path, the resident GPU Bekker output feeds the
  resident FEM-feedback path on the next frame when MPM terrain is not taking
  precedence, the rutted heightfield feeds the viewer FEM/brush contact
  terrain, and the right rail/gizmos expose backend, rut depth, active rut
  cells, and the rendered height/sinkage grid.
- `Scene::SoilCompaction` now owns a DEM sparse-LBVH stepping path for its
  visible `GranularBed` roller scene. The right rail can switch between the
  older `GranularBed` grid stepper and the `tire_dem` sparse-bed owner; the
  sparse path writes positions/velocities back into the visible bed and exposes
  LBVH nodes/leaves/max-leaf, one-step displacement, and a DEM top-layer
  surface-velocity map summary, so DEM offroad acceleration is visible outside
  the single-tire Scene 122 path.
- `RealTireTelemetrySample`, `TireContactPatchTelemetry`, and
  `TireContactCellTelemetry` provide reusable, unit-labelled tire telemetry
  in `molla-vehicle`. They summarize wheel force/moment accounting, patch
  pressure/area, dissipated power/energy, friction budgets, and finite sanity
  checks for repeatable tire-rig tests and viewer diagnostics. Wheel-level
  and per-contact-cell telemetry now also provide dependency-free CSV headers
  and row emitters with units in every field name, so rig/viewer logs can be
  written without inventing ad-hoc formats. `TireRigSnapshot` adds a compact
  non-calibration rig record with command inputs, preset names, terrain state,
  thermal/wear fields, provenance tags, optional CPU-vs-GPU force deltas, and
  unit-labelled CSV output. `TireRigAcceptanceReport` adds data-free
  qualitative pass/fail envelopes to those snapshots without fitting target
  curves. `TireSolverBudgetTelemetry` records reusable FEM solver-budget
  fields (substeps, XPBD iterations/compliances/corrections, max node speed,
  tet-volume error, bead mount drift, and dissipated power/energy) so rig
  snapshots can expose numerical stability costs. `TetraTireFemDiagnostics`
  also reports mean/max bead mount error and the named
  `cavity_carcass_coupling` parameter so simplified rim/bead sealing has a
  regression signal before the full bead-contact model exists.
  `TetraCavityBeadSanityReport` now adds broad non-calibration gates for
  pressure-to-bead-load scaling, loaded cavity volume bounds, bead mount drift,
  the audited pressure-to-carcass coupling, and XPBD bead correction; the
  default runner emits these as `cavity_bead_sanity`.
  `TireEnergySignAudit` adds reusable
  dissipated-energy, friction-budget, and aligning-moment sign checks.
  `TireReferenceShapeDiagnostics` adds non-fitted longitudinal and combined-
  slip shape checks for smooth buildup/saturation, friction-budget accounting,
  force tradeoff, and pneumatic-trail aligning-moment sign.
  `TireLowSpeedRobustnessReport` adds broad launch/lockup/reversal gates for
  finite brush/transient outputs, bounded regularized slip, shared friction
  budget accounting, and no force teleport at direction changes; the default
  runner emits this as `low_speed_robustness`. `TireFrameConventionReport`
  adds executable documentation for the native wheel/terrain frame and sign
  conventions, and the default runner emits this as `frame_sign_convention`.
  `TireContactPatchConvergenceReport` adds the first contact-patch convergence
  gate for segment refinement, pressure bins, patch length, force, and aligning
  moment, and the default runner emits this as `contact_patch_convergence`.
  `TetraFemSubstepConvergenceReport` adds the FEM solver-substep side of that
  convergence work, checking deformed tread patch/contact-force/hub-height and
  volume-diagnostic stability under explicit substep refinement; the default
  runner emits this as `fem_substep_convergence`.
  `TireHeightfieldResolutionConvergenceReport` adds the heightfield-resolution
  side of that convergence work, checking terrain height/normal, patch,
  force, and contact-count stability on the same analytic bump sampled across
  coarse-to-fine grids; the default runner emits this as
  `heightfield_resolution_convergence`.
  `TireTriangleDensityConvergenceReport` adds the finite-triangle terrain
  density side of that convergence work, checking height/normal, seam manifold,
  material-span, patch, and brush-force stability on the same two-plate road
  rebuilt across coarse-to-fine triangle subdivisions; the default runner emits
  this as `triangle_density_convergence`.
  `TireThermalWearConvergenceReport` adds the heat/wear side of that
  convergence work, checking dissipated slip power, surface temperature,
  wear-depth response, active sliding, and friction-budget stability across
  coarse-to-fine brush segment counts; the default runner emits this as
  `thermal_wear_convergence`.
  `TireRoadConditionStateReport` adds road-condition state gates for wet film,
  standing water, ice, moving ground, thermal/wear condition, worn wet tread,
  and soft-soil rolling loss; the default runner emits this as
  `road_condition_state`. `TetraConstructionSanityReport` adds a FEM
  construction sanity gate for pressure setup, bounded loaded radius,
  footprint response, tread support, belt stretch, sidewall axial bulge, bead
  mount drift, and sidewall fold drift; the default runner emits this as
  `construction_sanity`.
  `TireRealismSourceClassificationReport` adds a crate-side honesty check for
  native physics, diagnostic references, hand-tuned presets, numerical
  stabilizers, debug telemetry, and visualization-only helpers; the default
  runner emits this as `realism_source_classification`, and the shared viewer
  truth-label panel renders the same source-classification rows.
- `TirePressureDistributionTelemetry` exposes reusable pressure-distribution
  diagnostics along the rolling/circumferential axis and tread-width axis.
  Ideal brush contact reports the longitudinal patch pressure, while
  FEM-derived tread-node contact also reports actual widthwise pressure bins
  from deformed tread-node positions.
- `tire_fem_resolution_sweep()` adds a deterministic non-calibration FEM
  topology sweep. It records segments/radial/width layers, node/tetra/tread/
  bead counts, cavity volume, and tetra volume ranges so coarse mesh defaults
  can be reviewed before raising viewer density or adding tread-block detail.
- `tire_gpu_parity_checklist()` and `tire_gpu_parity_text_report()` provide a
  reusable CPU/GPU status matrix for the high-frequency tire stack. Each row
  is marked `cpu_and_gpu`, `cpu_only_debug`, or `planned_gpu` and must carry
  either GPU proof or an explicit reason/follow-up. It also records the
  layered transient relaxation path as CPU+GPU: the CPU model remains the
  readable reference, while `TireTransientGpuKernel` / `TireTransientGpuBatch`
  / `tire_transient.wgsl` batches tread/belt/carcass scalar relaxation per
  wheel with a count uniform so non-64 wheel batches do not overrun storage
  buffers. Scene 121 now uses that persistent transient batch before the
  resident vehicle-to-brush force path, so steering/torque lag feeds the GPU
  brush solve instead of remaining only a standalone parity lane. Scene 121 now
  asks `TireVehicleYawGpuKernel` to build resident brush rows, runs those rows
  directly through `TireBrushGpuKernel::dispatch_input_buffer_resident`, and
  uses a narrow `readback_resident_output` seam for UI/legacy CPU chassis
  telemetry. Direct GPU brush and CPU paths remain fallback/parity lanes, but
  the live force source is now the vehicle-row resident handoff rather than a
  side diagnostic. `molla-kernels` also has the first
  reusable transient-to-FEM-feedback adapter: `TireTransientGpuOutput` can now
  update FEM contact-feedback wheel omega and moving-ground surface velocity
  through `tire_fem_feedback_params_with_transient` and
  `tire_fem_contact_terrain_with_transient_surface_velocity`, so GPU FEM
  bristle feedback can consume the same effective slip state as the GPU brush
  path.
- `tire_known_approximation_ledger()` and
  `tire_known_approximation_text_report()` provide the reusable "do not
  overclaim realism" ledger. It lists explicit FEM integration, coarse tetra
  resolution, sidewall bend proxies, cavity/bead sealing, hand-tuned
  thermal/wear curves, Bekker-only soil, and triangle-surface simplifications with
  consequences, status, GPU parity, and next steps.
- `tire_preset_governance_report()` provides the first non-calibration preset
  review checklist. It records units, broad ranges, provenance tags, pass/fail
  status, and notes for named tire, surface, film, thermal, FEM-solver, and
  Bekker-soil presets.
- `RealTireParams` now has named hand-tuned presets
  (`passenger_radial`, `offroad_soft`, `high_pressure_reference`) plus
  scenario override helpers for geometry/pressure/segment count. Tire viewer
  examples use the named preset as the source of truth and only override scene
  dimensions or slider-controlled pressure/material tension.
- The supporting tire/terrain parameter structs now also have named presets:
  `SurfaceFilmState::{damp, wet_road, standing_water}`,
  `TireThermalLayers::{passenger_road, fast_response_test,
  heavy_duty_slow_response}`, `TetraTireFemParams::{realtime_viewer,
  offroad_soft, high_pressure_reference}`, and
  `DeformableTerrain::{compacted_dirt_patch, sandy_loam_patch,
  soft_clay_patch}`. These keep road-film, thermal, FEM solver, and Bekker
  soil constants grouped with units/ranges instead of scattering them through
  viewer/tests.
- `TireRigSweepPoint`, `TireRigThermalWearPhase`,
  `TireRigThermalWearPoint`, and the reusable tire-rig helpers
  (`tire_rig_load_deflection_sweep`,
  `tire_rig_longitudinal_slip_sweep`, `tire_rig_combined_slip_sweep`,
  `tire_rig_surface_slip_matrix`,
  `tire_rig_heightfield_envelope_sweep`, `tire_rig_camber_sweep`,
  `tire_rig_step_torque_response`, `tire_rig_step_brake_response`,
  `tire_rig_step_steer_response`, `tire_rig_surface_transition_sequence`,
  `tire_rig_rolling_loss_sweep`, `tire_rig_thermal_wear_run`,
  `tire_rig_thermal_wear_cycle`) provide
  deterministic non-calibration bench scenarios in the reusable crate. They
  record command inputs and the unit-labelled telemetry sample so load,
  slip/material, combined-slip, heightfield envelope contact, camber, step
  torque/brake/steer relaxation, spatial surface transitions,
  rolling-resistance / aero-drag loss, heat, wear, and cooling-recovery
  regressions can be tested without introducing external fitted data.
- `TireRigRunConfig`, `TireRigRunOutput`,
  `tire_rig_default_non_calibration_run`, and
  `tire_rig_standard_non_calibration_run` add the first built-in deterministic
  non-calibration runner. It emits reusable snapshots and acceptance-envelope
  reports for pressure/load, dry-vs-ice slip, camber/pneumatic trail, wet film,
  loose soil, moving-ground work, spatial surface transitions, rolling/aero
  loss, curb/heightfield contact, step torque/brake/steer response, and
  thermal/wear cycling without importing data or fitting coefficients.
- `DeformableTerrain::apply_tire_contact` couples reusable tire contact cells
  into Bekker soil: per-cell normal pressure plastically compacts the
  heightfield, over-shear from sliding tread adds extra excavation/sinkage, and
  the returned `TireSoilFeedback` reports rut depth plus rolling-resistance
  feedback. `tire_soil_rut_matrix` provides a reusable deformable-soil rig
  matrix that starts every slip case from the same Bekker heightfield and
  reports contact telemetry plus rut feedback.
- `TireDemGpuKernel` / `tire_dem.wgsl` adds the first GPU DEM tire-cell
  impulse-transfer path plus dense and cell-grid DEM steps. Each grain gathers
  nearby tire contact-cell force patches and receives localized normal/shear
  impulses without atomics; the DEM step kernels also solve grain-grain
  contacts, floor contact/friction, gravity, and tire-cell impulses on GPU.
  The grid paths use either a deterministic host-built cell list, an
  unbounded sorted cell-list CPU oracle, an unbounded GPU cell-key / GPU-radix /
  GPU range-scan path that keeps those buffers resident through the DEM
  neighbor step; the old GPU-built fixed-capacity API is now only a
  compatibility wrapper over that unbounded path, so large beds no longer
  require O(N²) pair scans and dense-cell overflow is explicit instead of
  silently accepted in the reference path.
  `TireMpmGpuKernel`
  / `tire_mpm.wgsl` now adds the first CPU/GPU material-point tire feedback
  pass: localized tire contact pressure/shear impulses update MPM particle
  velocity and plastic compaction diagnostics. `TireMpmFemTerrainOwner` owns
  that feedback state and rasterizes compaction into a `TireFemContactTerrain`
  sinkage/heightfield bridge for FEM/viewer contact. `MpmSimulation` in
  `molla-solvers` now has a first persistent-buffer composition bridge:
  `apply_tire_contacts_cpu_feedback_with`,
  `apply_tire_contacts_gpu_resident_with`,
  `apply_tire_contacts_gpu_feedback_with`, and
  `tire_fem_contact_terrain_with`. The GPU-resident path applies tire feedback
  in-place on live MPM particle/velocity/compaction buffers before the full
  particle/grid MPM solver continues, while diagnostic terrain export can read
  compacted state into `TireMpmFemTerrainOwner`. `TireMpmGpuKernel` also has a
  GPU compaction-to-height/sinkage/surface-velocity raster path that writes
  MPM terrain buffers without CPU rasterization, and
  `TireFemContactFeedbackKernel` can consume those GPU-resident height,
  sinkage, and spatial surface-velocity buffers directly for tread-node
  feedback; `TireFemContactReductionKernel` can consume the same height/sinkage
  buffers for diagnostics.
  `Scene::TireRealistic` now owns a live MPM offroad bridge beside the Bekker
  owner: tire contact cells write into a persistent `MpmSimulation`, rasterize
  GPU height/sinkage/surface-velocity buffers, dispatch the resident
  no-readback FEM contact-feedback path against those buffers, and use the
  centralized `readback_resident_feedback_output` seam only to mirror node /
  bristle rows back into the CPU-driven viewer tire/telemetry. It also exports
  `TireFemContactTerrain` with a CPU-side velocity map for the next FEM contact
  step. DEM now also has a Morton-sorted sparse-bed LBVH CPU oracle
  plus a GPU sparse-LBVH traversal path for non-grid grain beds. The first GPU
  topology-construction stage now exists as
  `tire_dem_build_sparse_lbvh_morton_order_gpu_radix`: it generates the same
  10-bit-per-axis Morton keys on GPU, radix-sorts grain ids, and has an
  ignored hardware parity lane against the CPU Morton order. The next GPU
  topology stage now exists as
  `tire_dem_build_sparse_lbvh_leaf_rows_gpu_radix`: it consumes that sorted
  order, builds per-leaf AABB/range rows on GPU, and has an ignored hardware
  lane against CPU sorted leaf chunks. The first internal-node GPU builder now
  exists as `tire_dem_build_sparse_lbvh_tree_rows_gpu_radix`: it pads arbitrary
  leaf counts to a complete binary sparse-LBVH row atlas, builds the tree
  bottom-up on GPU, and has ignored hardware lanes against both CPU
  complete-binary rows and non-power-of-two padded rows.
  `upload_resident_lbvh_tree_rows` now lets those GPU-built tree rows
  feed the resident sparse-LBVH DEM step; an ignored hardware lane dispatches
  DEM directly through that uploaded topology. `TireDemGpuKernel::
  build_resident_sparse_lbvh_tree_rows_gpu_radix` is now the no-readback
  resident builder for arbitrary leaf counts: Morton keys, radix sort, leaf
  rows, internal rows, and resident topology buffers stay on GPU before the
  DEM neighbor dispatch. Larger sparse-bed stress now covers
  a 14×12×5 CPU/reference row-layout case, a large ignored GPU hardware
  dispatch stress lane verified through
  `make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_gpu_large_sparse_lbvh_dispatch_stress_matches_reference --nocapture'`,
  and a matching large no-readback resident-builder stress lane verified
  through
  `make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_resident_gpu_lbvh_builder_large_sparse_bed_matches_reference --nocapture'`.
  The resident sparse-LBVH seam also keeps `TireDemResidentLbvh`
  active-count/capacity bookkeeping plus `update_resident_lbvh` for CPU/debug
  parity, but the reusable `TireDemResidentFemTerrainOwner` now uses
  `build_resident_sparse_lbvh_tree_rows_gpu_radix` in its hot step: Morton
  quantization bounds are cached at construction, while Morton keys, radix
  sort, leaf rows, padded internal rows, resident topology buffers, DEM
  stepping, and terrain rasterization stay on GPU. The CPU mirror is now a
  viewer/debug telemetry/readback seam rather than a per-step topology input.
  `TireDemResidentFemTerrainOwner::reseed_grain_state` now gives viewer/reset
  code an explicit same-count in-place GPU state rewrite path through
  `TireDemGpuKernel::update_resident_state`, refreshing cached bounds,
  reference terrain, resident raster output, and invalidating stale LBVH
  topology without reconstructing the whole owner. Scenes 119 and 122 now call
  this bridge when their CPU-visible DEM source state diverges from an existing
  resident owner, so toggling/fallback/reset transitions reseed GPU buffers
  instead of silently stepping stale resident grains.
  The ignored owner lane now verifies two consecutive GPU DEM/FEM-terrain
  steps through that no-readback topology builder instead of the older
  CPU-built in-place refresh.
  Broader DEM/mesh terrain ownership and higher-grain GPU scaling remain
  future work.
- FEM hardening tests now pin the explicit solver's safe operating envelope:
  high-load/low-pressure, high-pressure/high-speed, thousands of small
  substeps, and disturbed-energy damping all stay finite and bounded.
  `TetraFemHardeningEnvelopeReport` now packages the same hardening scenarios
  as a reusable non-calibration report with acceptance checks for support
  margin, patch existence, high-pressure radius/speed bounds, long-run
  substep count/deflection, and kinetic-energy damping. The shared tire
  truth-label panel renders this as `fem_hardening_envelope` beside the FEM
  solver stiffness gate so robustness evidence is visible in the viewer rather
  than only as isolated unit tests.
- `TetraTireFem` now has an optional XPBD volume-preservation pass over every
  tetrahedron. It runs after the explicit node update, projects signed
  tet volume back toward rest volume with compliance/iteration knobs, updates
  node velocities from the positional correction, and re-applies terrain
  projection so constraint stabilization cannot leave tread nodes underground.
  The GPU tire-FEM node kernel now mirrors this with a race-free gather-style
  single-sweep XPBD volume projection controlled by the same compliance value
  in `TireFemGpuParams`; persistent GPU stepping gets additional sweeps by
  dispatching more substeps.
- The GPU tire-FEM node kernel also mirrors the XPBD bead/rim mount projection
  using existing bead id/offset buffers and the GPU-resident hub pose. This
  gives GPU nodes a geometric bead attachment guardrail in addition to the
  existing bead spring force, controlled by `TireFemGpuParams`.
- The GPU tire-FEM node kernel now has first-pass structural XPBD edge parity
  for tread-belt and sidewall bend-proxy constraints. `TetraTireFem::gpu_snapshot`
  exports structural edge endpoint/rest/compliance buffers plus node-edge CSR
  adjacency; the node kernel gathers incident edges and applies its endpoint
  correction without float atomics. This is still an edge-distance proxy, not
  a full laminated/dihedral sidewall solver.
- `TetraTireFemParams::semi_implicit_stiffness_scale` adds the first CPU+GPU
  semi-implicit stiff-rubber stabilizer for high-pressure/high-modulus tires.
  The constitutive forces are still evaluated explicitly, but the velocity
  update can include a lumped tangent-stiffness denominator
  `m + c·dt + scale·k_diag·dt²`, and solver-budget telemetry records the
  active scale. `TireFemGpuParams` and `tire_fem.wgsl` now mirror the same
  denominator for the GPU FEM node path; a full Newton/implicit solve remains
  future solver-depth work. `TetraFemDiagonalNewtonReport` now adds the first
  executable Newton-style implicit solver gate as `fem_diagonal_newton_solver`:
  a stiff nonlinear radial tire mode is advanced with a backward-Euler
  residual/Jacobian solve, and the standard runner/viewer gate residual
  reduction, iteration count, positive tangent, explicit overshoot, and
  implicit-vs-explicit energy ratio. This is a real Newton solve for the
  reduced stiff mode, not yet the full coupled tetra-FEM Newton system.
  `TetraFemElementNewtonReport` now takes the next step as
  `fem_element_newton_solver`: it builds one actual Neo-Hookean tetra element,
  keeps all four nodes coupled through the deformation gradient, assembles a
  finite-difference 12-DoF backward-Euler residual/Jacobian, and gates Newton
  convergence, residual reduction, off-diagonal coupling, pivot health,
  bounded volume, and implicit-vs-explicit energy. This still is not the full
  tire-mesh Newton solve, but it proves the coupled element residual/Jacobian
  path instead of only a scalar mode. `TetraFemPatchNewtonReport` now extends
  that path to a connected two-tetra patch as `fem_patch_newton_solver`:
  element forces accumulate through shared nodes, a single multi-node
  residual/Jacobian Newton solve advances the whole patch, and the runner /
  viewer gate residual reduction, off-diagonal coupling, shared-node force
  accumulation, pivot health, volume bounds, and implicit-vs-explicit energy.
  This is still a compact CPU patch fixture, not the full tire mesh, but it is
  the first multi-element Newton gate. `TetraFemTinyMeshNewtonReport` now adds
  the first actual annular tire-mesh Newton gate as
  `fem_tiny_mesh_newton_solver`: the smallest allowed `TetraTireFem` mesh
  solves all node positions in one residual/Jacobian system, includes tread
  and bead rows, and gates full-mesh node/tet counts, residual reduction,
  off-diagonal coupling, volume bounds, and implicit-vs-explicit energy. This
  is CPU/debug and tiny-mesh only, but it proves mesh-level Newton plumbing
  before production-size tire and GPU parity work. `TetraFemMatrixFreeNewtonScaleReport`
  now adds the next solver-depth gate as `fem_matrix_free_newton_scale`: it
  evaluates a larger annular tire mesh residual, computes a finite-difference
  matrix-free Jacobian-vector product without assembling the dense Jacobian,
  and reports dense-matrix bytes versus residual/Jv working bytes. This keeps
  the path honest as CPU/debug readiness for a future sparse/matrix-free
  Newton/PCG solve. `TetraFemMatrixFreeNewtonSolveReport` now adds the first
  Krylov correction gate as `fem_matrix_free_newton_solve`: the same larger
  tire-mesh residual feeds finite-difference Jv-driven GMRES iterations, a
  line search applies a bounded correction, and the report gates nonlinear
  residual reduction, Jv evaluation count, correction magnitude, and the
  dense-vs-matrix-free memory ratio. `TetraFemMatrixFreeNewtonPreconditionerReport`
  now adds the next preconditioner gate as
  `fem_matrix_free_newton_preconditioner`: it samples the tangent diagonal by
  matrix-free Jv products, builds a Jacobi-style scaling, compares
  preconditioned versus unpreconditioned Krylov iterations/residuals, and
  line-searches the scaled correction against the nonlinear residual.
  `TetraFemMatrixFreeNewtonStepReport` now adds the first implicit-step state
  reconstruction gate as `fem_matrix_free_newton_step`: it applies the
  preconditioned correction to the larger mesh predictor, reconstructs
  velocities, and gates residual reduction, accepted line-search step,
  correction bounds, tetra volume bounds, velocity bounds, and implicit-vs-
  explicit energy. `TetraTireFem::step_on_terrain_matrix_free_newton_debug`
  now turns that report-only path into an actual opt-in CPU/debug timestepper,
  and `fem_matrix_free_newton_timestep` exercises it in the standard runner:
  the method mutates tire positions/velocities, reapplies structural/contact
  projections, and emits the same residual/velocity/volume/energy gates. Both
  callable timestep and the matching report path now share the
  `matrix_free_newton_correction` helper so the predictor build, Jv probe,
  diagonal preconditioner sampling, GMRES correction, and line search cannot
  drift between the report-only and the live-mutating site. The GPU parity
  row is now **CpuAndGpu** end-to-end:
  `molla_kernels::tire_fem_matrix_free_newton_gpu_contract` returns a
  host-side `TireFemMatrixFreeNewtonGpuContract` that pins the WGSL dispatch
  shape; the WGSL kernel
  `crates/molla-kernels/shaders/tire_fem_matrix_free_newton.wgsl` provides
  `residual_eval`, `jv_probe`, `diagonal_extract`, and `apply_correction`
  entry points; `TireFemMatrixFreeNewtonGpuKernel` compiles the shader and
  dispatches all four entry points end-to-end with readback; and
  `TireFemMatrixFreeNewtonGpuKernel::run_newton_outer_blocking` drives a
  full host-orchestrated GMRES + Jacobi + line-search Newton outer iteration
  on top of the GPU primitives. Two ignored hardware lanes — `…_residual_matches_cpu_oracle`
  and `…_outer_iteration_reduces_residual` — run against a real wgpu adapter
  and prove (a) per-DoF parity against a closed-form CPU oracle and
  (b) end-to-end residual reduction through the orchestrated outer iteration.
  Production-promotion seams:
  `TetraTireFemParams::solver_mode = TetraTireFemSolverMode::MatrixFreeNewtonDebug`
  opts any tire into the matrix-free implicit timestep through
  `step_on_terrain`, and the new
  `TetraTireFemParams::high_pressure_implicit()` preset ships with
  `solver_mode` already flipped. The realtime_viewer / offroad_soft presets
  still default to `ExplicitDefault` until the GPU dispatch is wired into the
  persistent FEM batch substep loop and live FEM scene envelopes are
  validated against the explicit/semi-implicit stepper.
- The tetra FEM material now exports and evaluates four explicit ply/fibre
  families on CPU and GPU: the existing crossed ± belt/sidewall families plus
  cap/hoop and radial/axial helper plies. This is a compact multi-family
  anisotropic stack, not a laminated shell or dihedral sidewall solve, but it
  removes the previous two-family-only limitation.
- `TetraSidewallDihedralDiagnostics` now measures actual sidewall quad
  fold-angle deviation from the rest tire, and `TetraTireFem` has a first-pass
  CPU XPBD sidewall dihedral-angle projection controlled by
  `xpbd_sidewall_dihedral_compliance`. This gives the remaining laminated
  sidewall work a reusable validation signal and an initial true fold-angle
  constraint instead of only the older edge-distance proxy error. GPU parity
  for the dihedral projection is now done in `tire_fem.wgsl`.
- `TetraSidewallLaminateAudit` now checks the explicit sidewall laminate
  layout itself: higher-radial-resolution tires must expose crossed ± bias
  ply families plus radial and axial helper plies, with the hand-tuned 38°
  sidewall bias angle remaining consistent around the tire. This is still not
  a full laminated shell solve, but it makes the current ply-stack auditable.
- `TetraSidewallLaminateSanityReport` now combines that static ply-stack audit
  with live sidewall dihedral fold diagnostics. A tire can therefore be
  rejected if it has the expected crossed/helper ply directions but the current
  sidewall shape is folded outside the realtime sanity envelope. This is still
  a guardrail around the compact XPBD/ply approximation, not the final
  laminated shell/carcase solve. The default non-calibration runner now emits
  `sidewall_laminate_sanity` gates for crossed ply coverage, bias-angle error,
  helper-ply alignment, and mean/max fold drift from a deterministic static
  laminate validation tire, so this construction sanity check is visible in
  standard tire-rig reports instead of only isolated unit tests. The shared
  tire truth-label panel now renders `sidewall_laminate_sanity` directly from
  the same `TetraTireFem::sidewall_laminate_sanity_report()` path, including
  crossed-ply coverage, bias-angle error, helper alignment, fold drift, and
  ply stretch/compression gates before the dynamic sidewall load cases.
- `TetraSidewallLaminateCouplingReport` now adds an executable stepping stone
  toward laminated sidewall dynamics without pretending the full shell solve
  is done: deterministic radial bulge, widthwise stretch, and fold fixtures
  must put finite tensile energy into the crossed ± bias plies plus radial and
  axial helper plies; the crossed pair must stay balanced under symmetric
  bulge; and the same folded sidewall must be improved by the structural XPBD
  projection. The standard runner emits this as
  `sidewall_laminate_coupling`, and the viewer truth-label panel surfaces the
  bias/helper energy, fold energy, and fold-reduction gates next to the other
  cavity/bead/sidewall reports. This is still a coupling/fixture gate over the
  compact tetra ply stack, not the final laminated shell carcass model. The GPU
  parity checklist tracks this report explicitly as
  `tetra_fem_sidewall_laminate_coupling_report` with `cpu_only_debug` status:
  the force-producing four-family fibre stress and dihedral XPBD projection
  have GPU paths, but this diagnostic reduction remains CPU-only until it
  becomes a real laminated shell/carcass constraint with GPU parity.
- `TetraSidewallPlyStrainXpbdReport` now adds the first force/constraint
  producing sidewall ply-strain guardrail. The CPU projection evaluates
  sidewall tetra ply stretch as `|F·a0|`, projects excessive tensile stretch
  back toward a configured envelope, and emits the `sidewall_ply_strain_xpbd`
  standard-runner gate for pre/post max tensile strain, reduction ratio, and
  correction magnitude. `TireFemGpuParams` now carries the ply stretch limit
  and compliance into both `tire_fem.wgsl` and `tire_fem_atlas.wgsl`, where
  Stage (f4) applies the same incident-tet sidewall-ply gather projection; a
  source guard plus ignored hardware lane cover the single-tire shader path.
  The viewer surfaces this beside the laminate coupling and dihedral fold
  gates. This is still a compact XPBD guardrail over the tetra ply stack, not
  a full laminated shell; replacing it with a true laminated shell/carcass
  constraint remains open.
- `TetraTireFem` now also has first-pass XPBD structural constraints for
  bead/rim attachment, tread-belt inextensibility, and sidewall bending
  resistance. Bead nodes are projected toward their mounted rim offsets with a
  compliance knob, the outer tread layer has circumferential/axial/crossed
  diagonal belt edges that resist stretch/shear, and sidewall surface quads
  get opposite-vertex hinge-distance constraints as a cheap bending proxy
  before the volume solve. CPU sidewall quads also get an initial dihedral
  fold-angle projection. This is structural constraint stabilization, not a
  full laminated multi-ply belt/sidewall solver yet, but the dihedral angle
  path now has CPU/GPU parity. `TetraSidewallDihedralXpbdEffectReport` turns
  that path into a standard-runner gate by injecting a sidewall fold and
  requiring the structural projection to reduce the measured max fold-angle
  drift.
- CPU tests cover brush contact, combined-slip friction-cone saturation, heat
  generation, single-lump and layered thermal state, temperature-accelerated
  wear, state-derived thermal/wear material-response failure curves,
  load-sensitive sub-linear peak force, smooth combined-slip saturation,
  dynamic carcass stability, volumetric mesh creation, tetra FEM floor
  contact, lower-pressure wider patches, cavity volume, GPU snapshot
  production fields, pressure-dependent cavity load, CPU FEM contact-wrench
  feedback into tread nodes, persistent CPU tread-node bristle updates,
  crossed ply-angle fibre export, XPBD volume-projection/terrain-projection
  guardrails, and XPBD bead/tread-belt/sidewall-bend constraint guardrails.
- GPU tests cover the tire FEM one-shot path, persistent-buffer FEM path, GPU
  FEM contact diagnostics reduction with heightfield/friction-map sampling and
  local-normal world-wrench output, GPU XPBD volume, bead/rim, and structural
  edge projection, GPU sidewall dihedral-angle projection, GPU FEM tread
  bristle/force feedback into node velocities, GPU brush/contact parity
  against the CPU brush model, persistent-buffer brush dispatch, GPU
  heightfield friction-map material lookup, and GPU heightfield normal/world-
  force parity, DEM tire-cell impulse parity and dense/grid grain/floor/tire
  step parity lanes, and MPM material-point tire feedback parity.

## What is wrong today

- The remaining gap is not "fake viewer traction" anymore; it is solver depth:
  - the tire FEM is explicit Neo-Hookean tetra FEM, not an implicit tire
    solver; XPBD volume stabilization now helps near-incompressibility, but
    the main rubber constitutive update is still explicit; volume, bead, and
    structural edge-distance XPBD exist on CPU and GPU;
  - cavity pressure is now geometric/gas-law based, but still uses a named,
    preset-governed `cavity_carcass_coupling` fraction until rim/bead/contact
    validation and better structural modeling exist; the cavity/bead sanity
    acceptance report now guards pressure scaling, finite loaded volume, the
    visible coupling parameter, bead mount drift, and decomposed bead-seat
    radial/axial/tangential drift while that deeper rim-seat contact solve
    remains open. `TetraCavityCouplingEffectReport` now makes the coupling
    proxy itself falsifiable in the standard runner as `cavity_coupling_effect`:
    zero coupling must remove bead load, nominal coupling must produce finite
    bead load, stronger coupling must increase the same gas-law path, and bead
    drift must stay bounded;
  - contact is zero-gap/XPBD-style against a terrain surface, with
    first-pass FEM tread-node contact diagnostics, per-cell terrain
    friction/surface velocity, and CPU node-force feedback from the
    wheel-level brush wrench; heightfield, triangle-mesh and simple SDF
    terrain queries exist, CPU FEM projection uses true SDF/triangle signed
    distance where available, and GPU FEM diagnostics/feedback now sample
    heightfield normals/friction/film plus built-in sphere/box SDF primitives
    finite triangle patches with a first GPU per-triangle broadphase sphere,
    deformable-terrain sinkage maps for rolling-loss feedback, and first GPU
    Bekker rut writeback for height/sinkage maps;
  - combined longitudinal/lateral brush behavior, aligning moment, first-pass
    transient relaxation length, camber thrust, pneumatic-trail diagnostics,
    layered heat/wear feedback, load sensitivity, and smooth combined-slip
    saturation exist; initial separate tread/belt/carcass transient lag and
    state-derived temperature/wear material-response failure modes now exist;
    `TireThermalAbuseEnvelopeReport` now adds a standard-runner
    `thermal_abuse_envelope` gate that compares healthy, cold, overheated,
    worn, and abused states through the reusable material-response bridge,
    proving grip, wear rate, rolling loss, tread shear, carcass softening, and
    condition labels are state-derived. Deeper fitted rubber-chemistry
    validation remains future work;
  - four explicit ply-angle fibre families now exist for tread/belt, bead,
    and sidewall regions on CPU and GPU; sidewall dihedral fold-angle
    diagnostics and an initial CPU/GPU dihedral-angle XPBD projection now
    exist; a sidewall laminate sanity report now gates the explicit crossed
    bias/helper ply stack against live fold drift and ply stretch/compression,
    but a true laminated sidewall carcass solve remains future structural work; first-pass XPBD
    bead mount, mean/peak tread-belt stretch diagnostics, and
    sidewall bend-proxy constraints exist;
  - Bekker-style heightfield soil coupling plus static triangle-mesh/simple
    SDF terrain exist. First-pass CPU DEM tire-cell coupling now transfers
    normal/shear tire contact impulses into `GranularBed` grains, and
    `TireDemGpuKernel` now mirrors the first localized tire-cell-to-grain
    impulse transfer, a dense grain/floor/tire contact step, a host-built
    cell-grid broadphase step, an unbounded sorted-grid CPU oracle, an
    unbounded GPU cell-key / GPU-radix / GPU range-scan path that stays
    resident through the DEM neighbor step, a compatibility wrapper for the old
    GPU-built fixed-capacity API, first CPU/GPU MPM
    material-point tire pressure/shear/compaction feedback with FEM terrain
    ownership/export, a first in-place
    `MpmSimulation` bridge that writes tire feedback back into the full
    particle/grid MPM solver, GPU MPM compaction-to-terrain raster buffers,
    and direct GPU FEM feedback/reduction consumption of those buffers, plus a
    Morton-sorted sparse-bed LBVH CPU oracle/node layout, first GPU Morton-key
    radix sorted-order stage, first GPU leaf AABB/range row stage, first GPU
    padded complete-binary internal tree row stage for arbitrary leaf counts, first
    resident upload/dispatch seam for GPU-built tree rows, first no-readback
    resident GPU builder/dispatch lane for the same padded topology, and
    first GPU sparse-LBVH traversal path for DEM sparse beds. Resident sparse-LBVH topology
    buffers now carry active-count/capacity metadata for CPU/debug refreshes,
    and repeated `TireDemResidentFemTerrainOwner` steps build their resident
    topology with the GPU Morton/radix/leaf/tree pipeline from resident DEM
    state, using cached Morton bounds rather than per-step CPU grain mirrors.
    Owner resets/reseeds can now update same-count resident grain buffers
    in place and drop stale topology before the next GPU rebuild, and the
    Scene 119/122 viewer paths use that bridge when switching back to resident
    GPU DEM after CPU-side source-state changes.
    The remaining
    GPU LBVH work is broader multi-scene owner adoption and larger scaling
    evidence after the sorted Morton-order, leaf-row, complete-binary tree-row,
    no-readback resident builder, and resident owner adoption stages. The larger sparse-bed
    stress lanes now cover a 14×12 cluster field with 840 grains, packed
    dispatch sizing, CPU/reference parity, a direct sparse-LBVH GPU dispatch
    stress test, and a no-readback resident-builder GPU dispatch stress test;
    first viewer/offroad ownership exists in `Scene::TireRealistic` through
    its live MPM bridge.

## Real model architecture

### 1. Surface material model

Add a real surface/contact material structure:

```rust
SurfaceMaterial {
    mu_static,
    mu_dynamic,
    bristle_damping,
    rolling_resistance,
    cohesion,
}
```

Initial built-ins:

- dry asphalt/concrete: high static and dynamic friction;
- dirt/gravel: medium friction, more damping;
- ice: very low friction;
- sand/soil: medium/low friction plus higher rolling resistance.

The existing `Terrain::friction()` remains as a compatibility scalar, but the
real tire APIs accept a `SurfaceMaterial` so the model can distinguish static
vs dynamic friction and rolling loss.

### 2. Pressure-dependent contact patch

Represent the tire contact patch by sampling the tire circumference into
segments. For a given hub pose, radius, width, pressure, and terrain height:

- compute each segment's ground penetration;
- compute per-segment normal load;
- solve hub height for a requested wheel load;
- report:
  - total normal load,
  - patch length,
  - contact cell count,
  - per-cell normal pressure.

Pressure affects vertical compliance:

- high pressure -> less deflection, shorter patch, higher local pressure;
- lower pressure -> more deflection, longer patch, lower local pressure;
- too low pressure -> high rolling resistance / carcass loss.

### 3. Brush/bristle longitudinal traction

For each contact segment:

- compute local slip speed from wheel spin vs hub velocity;
- convert slip speed into bristle shear displacement;
- compute trial tangential force from tread shear stiffness;
- cap by material friction:
  - static region: `|Ft| <= mu_static * Fn`;
  - sliding region: `Ft = mu_dynamic * Fn` opposite slip.

This gives the important behavior:

- asphalt: even small patch can transmit useful torque before slipping;
- ice: low `mu` caps force even if many segments touch;
- low pressure increases patch length and contact stiffness, improving
  low-frip traction up to the point where rolling resistance dominates.

### 4. Wheel spin dynamics

Driven wheels should integrate angular velocity:

```text
I * omega_dot = motor_torque - Fx * radius - bearing_loss
```

Vehicle longitudinal motion integrates:

```text
m * v_dot = sum(Fx_wheels) - rolling_resistance - drag
```

Slip is therefore an outcome: if motor torque exceeds what the tire/ground can
transmit, wheel angular velocity climbs while the car barely accelerates.

### 5. Viewer integration path

First integration target:

- `Scene::TireTraction`
  - red fixed car: high pressure;
  - blue fixed car: partial pressure;
  - green user car: pressure + payload sliders.

Replace local patch-factor fake with calls into `molla_vehicle::real_tire`:

- solve wheel hub height from payload/load;
- evaluate brush contact for the driven wheel;
- integrate wheel spin and chassis velocity;
- draw patch bars/readouts from actual contact output.

Second integration target:

- `Scene::TireDynamics`
  - now uses `TetraTireFem` instead of quasi-static shape generation;
  - continue improving toward implicit/nonlinear FEM and GPU kernels.

## Validation / tests

Add crate-level tests in `molla-vehicle`:

1. **Pressure changes patch length**
   - same load on flat terrain;
   - low pressure produces more contact cells / longer patch than high pressure.

2. **Surface material dominates ultimate grip**
   - asphalt can transmit more longitudinal force than ice for the same load
     and pressure.

3. **Lower pressure improves low-friction tractive stiffness**
   - same low-friction material and same slip speed;
   - partial pressure produces more brush force than over-inflated pressure,
     before the friction cap dominates.

4. **Torque-induced slip is an outcome**
   - with the same motor torque/load:
   - high pressure on low-friction ground accumulates higher slip;
   - partial pressure moves farther over the same simulated interval.
   **Initial reusable `tire_torque_induced_slip_outcome_report` is now part of
   the standard non-calibration runner as `torque_induced_slip_outcome`: it
   integrates wheel spin and chassis speed from the native brush force under
   equal torque/load/low-μ surface, then gates high-pressure slip, partial-
   pressure distance/speed, patch growth, and friction-budget accounting.
   `TireWheelSpinTorqueBalanceReport` now also gates the actual one-step
   wheel-spin equation as `wheel_spin_torque_balance`: motor torque, tire
   reaction torque, bearing loss, net torque, inertia, `ωdot`, integrated
   `Δω`, resulting slip, and friction-budget use are emitted by the standard
   runner and surfaced in the viewer force-shape panel so slip remains an
   outcome of torque balance rather than a scene-local patch factor.
   `TireVehicleLongitudinalMotionReport` now adds the matching vehicle-speed
   equation as `vehicle_longitudinal_motion`: summed tire drive force, rolling
   loss, aerodynamic drag, vehicle mass, acceleration, integrated `Δv`, and
   force-balance closure are checked so chassis speed also comes from native
   tire/vehicle accounting rather than a viewer velocity shortcut.**

   The lower-pressure low-friction stiffness fixture is now also promoted from
   a unit-only check into the standard `load_combined_slip_law` report:
   `low_mu_pressure_tractive_stiffness_improves` verifies partial pressure
   produces more pre-cap drive force on a low-μ surface, and
   `low_mu_pressure_patch_grows` ties that to a longer physical contact patch
   rather than a hidden grip multiplier.

5. **Volumetric tetra FEM mesh and contact**
   - the tire mesh has circumferential/radial/width nodes and many tetrahedra;
   - loaded tetra FEM tires stay finite and above the floor;
   - lower pressure gives a wider tetra FEM footprint than high pressure.

## Non-calibration realism upgrades

Calibration against real tire data is explicitly out of scope for now. Do not
block the next tire pass on fitting Pacejka/TMeasy/rig data. The next work is
structural/modeling depth that improves the physics model itself while keeping
parameters hand-tuned and inspectable.

Out of scope for this plan:

- calibration/fitting against proprietary or measured tire datasets;
- Pacejka/Magic-Formula-as-the-main-model unless it is only used later as a
  comparison curve;
- MJCF/USD/URDF import/export work;
- FFI bindings or simulator glue that does not improve the native Rust tire
  physics.

The non-calibration points we still want to keep from real tire/racing-sim
practice are:

- **load sensitivity** — peak friction and cornering/longitudinal stiffness
  should change with vertical load, not only with surface material;
- **combined-slip shape** — braking/drive slip and lateral slip should share a
  single friction budget with a smooth saturation curve, not independent clamps;
- **camber, contact-centre shift, and pneumatic trail** — steering feel should
  come from distributed patch forces, not an arbitrary yaw torque;
- **transient relaxation** — tread/carcass force buildup must lag slip and
  steering changes, with stable low-speed regularization;
- **carcass and belt dynamics** — inflated/deflated behavior should come from
  structural compliance, belt stiffness, sidewall modes, and pressure, not
  viewer-only shape scaling;
- **thermal/wear feedback** — surface temperature should affect grip quickly,
  carcass/tread temperature should affect pressure/stiffness slowly, and hot
  sliding should accelerate wear;
- **surface state** — dry asphalt, dirt, sand, ice, wet/film layers, and loose
  soil should affect both normal support and tangential shear;
- **terrain envelope/contact** — the tire should build its patch from actual
  road/heightfield/FEM tread contact, including bumps/curbs/ruts;
- **repeatable tire-rig scenarios** — even without fitting measured data, keep
  deterministic bench scenes for load/pressure/slip/camber/temperature sweeps
  so regressions are obvious;
- **telemetry and force accounting** — expose patch cells, bristle state,
  forces, moments, heat, wear, energy loss, and GPU/CPU deltas so "realistic"
  can be inspected instead of guessed;
- **parameter provenance and presets** — keep hand-tuned values grouped by
  tire/terrain preset with units and sane ranges, not scattered magic numbers
  in viewer code;
- **dimensional and numerical sanity checks** — guard against impossible
  loads, NaNs, underground nodes, energy explosions, and force discontinuities;
- **GPU parity** — every per-wheel/per-contact model that is performance
  relevant needs a GPU path, or must be explicitly marked as debug/CPU-only.

Concrete additions to keep this useful without doing calibration/fitting:

- **data-free acceptance envelopes** — define expected qualitative/monotonic
  behavior for each tire rig instead of matching measured curves: lower
  pressure should lengthen the patch, ice should cap force, added load should
  increase force sub-linearly, and sustained sliding should heat/wear the tire.
- **synthetic fixture suite instead of calibration fixtures** — keep the
  useful engineering fixtures, but use them as deterministic regressions rather
  than fitting targets:
  - load/deflection and footprint growth across pressure/load;
  - pure longitudinal slip, pure lateral slip, and combined-slip sweeps;
  - step steer/step torque relaxation response; **initial reusable
    torque/brake/steer step-response fixtures/tests done**;
  - camber / pneumatic-trail / aligning-moment sweeps;
  - rolling-resistance and drag-loss checks; **initial reusable
    `tire_rig_rolling_loss_sweep` done for material/film rolling loss,
    aero-drag force/power scaling, total coastdown loss, and finite telemetry;
    the standard non-calibration runner now emits `rolling_loss_drag`
    acceptance gates for soft-soil rolling loss, standing-water rolling power,
    aero force/power scaling, and positive coastdown loss**;
  - cleat/curb/heightfield obstacle passage; **initial reusable
    `cleat_envelope_contact` snapshot/report added beside the existing curb
    and rolling-obstacle cases, checking raised cleat height, multi-cell
    contact, extended low-pressure patch length, and horizontal obstacle
    force without external calibration data**;
  - wet-film / ice / dirt / loose-soil material transitions; **initial
    reusable spatial surface-transition sequence/test done, including
    moving-ground bands**;
  - thermal/wear/cooling cycles;
  - CPU-vs-GPU parity for every high-frequency path.
- **sensitivity sweeps** — keep pressure, load, speed, camber, water-film,
  temperature, tread-wear, soil, and obstacle sweeps in reusable tests/examples
  so parameter changes show their effect without pretending the numbers are
  fitted to a real tire. **Initial reusable
  `tire_rig_standard_sensitivity_map` /
  `tire_rig_standard_sensitivity_text_report` done for pressure, load, dry
  rolling-speed loss, wet speed, camber, surface temperature, water film,
  tread wear, soft-soil material, soil cohesive stiffness, obstacle-height,
  and contact-resolution directional checks; the default tire-rig
  runner now carries those rows into its text report and shared viewer replay
  panel.**
- **dimensionless sanity metrics** — track ratios that should stay sane across
  all fixtures: `Fx / Fz`, `Fy / Fz`, consumed/available friction budget,
  patch length/radius, contact pressure/load, slip power/thermal energy, and
  GPU/CPU relative error. These do not require measured tire data, but they
  catch impossible behavior quickly. **Initial reusable
  `TireRigDimensionlessSanityMetrics` plus broad acceptance-check generation
  done for rig snapshots. The shared viewer tire-rig replay panel now surfaces
  those same selected-snapshot ratios in a collapsed `Dimensionless sanity`
  section: `|Fx|/Fz`, `|Fy|/Fz`, consumed/available friction budget,
  patch-length/radius, pressure-area/load accounting, slip-energy/(load·R),
  and optional CPU/GPU relative force error.**
- **energy and sign accounting** — every contact path should report enough
  terms to verify that sliding/rolling dissipate non-negative energy, normal
  contact does not add tangential work by accident, aligning moment sign
  follows the force-resultant/trail convention, and moving terrain contributes
  work with the expected sign. **Initial reusable `TireEnergySignAudit` plus
  acceptance-report checks are done for non-negative dissipated power/energy,
  friction-budget residual, pneumatic-trail aligning-moment sign, and
  moving-ground work sign from snapshot terrain surface velocity. `TireRigSnapshot`
  also records `moving_ground_power_w` in its unit-labelled CSV row so this is
  visible as telemetry, not only as a pass/fail gate. The shared viewer
  tire-rig replay panel now surfaces the selected snapshot's
  `energy_sign_audit()` result, friction-budget residual, moving-ground power
  sign, and pneumatic-trail aligning-moment sign audit.**
- **model provenance tags** — mark each parameter group as physical,
  hand-tuned, numerical-stability, debug, or visualization-only. Viewer
  sliders may override presets, but they must not become hidden physics
  constants. **The viewer now has tire-scene control metadata that classifies
  controls as physical state, explicit preset overrides, numerical-stability
  knobs, debug toggles, or visualization-only helpers.**
- **preset governance without fitting** — keep a small number of named tire
  and surface presets, document units/ranges beside each field, and require a
  short acceptance checklist when a preset changes. Do not chase exact real
  tire numbers until we explicitly decide to add calibration later. **Initial
  reusable `tire_preset_governance_report()` done with unit/range/provenance
  checks for tire, surface, film, thermal, FEM-solver, and Bekker-soil
  presets.**
- **reference-model comparisons only as diagnostics** — if Magic Formula,
  TMeasy, racing-sim style curves, or published plots are used later, use them
  only to compare shape/sign/saturation behavior. Do not replace the native
  brush/FEM path with fitted curve output.
- **visual falsification passes** — every viewer tire scene should make it
  obvious when the model is lying: show patch cells, slip/adhesion state,
  force/moment vectors, pressure/cavity state, heat/wear state, and CPU/GPU
  deltas. **Initial shared internal tire-rig replay panel now lets scenes
  120/121/122 inspect the same deterministic non-calibration suite beside the
  live viewer state, and it now surfaces optional `TireRigSnapshot`
  CPU/GPU force-vector and relative-error deltas whenever the replay contains
  paired CPU/GPU evidence. The replay panel now also exposes detailed
  selected-snapshot patch/friction telemetry: contact/sliding cell counts,
  contact area, mean/peak pressure, contact centre, force-resultant position,
  pneumatic trail, adhesion/sliding fractions, slip speeds, effective
  longitudinal/cornering stiffness, and rolling-resistance force.**
- **scene acceptance checklists** — keep short per-scene "what should happen"
  notes beside the viewer examples so broken behavior is easier to report:
  over-inflated wheels should look stiff and slip sooner on low-friction
  surfaces, softer wheels should visibly flatten and build more patch, ice
  should still slip even with large patch area, and soil should rut/compact
  under repeated loaded contact. **Initial right-rail tire truth labels are
  done for the tire lab and scenes 120/121/122. The same truth-label panel now
  includes explicit `Scene checklist` rows for each tire scene, with tests
  guarding pressure/deflation, traction/slip, and realistic-bench checklist
  language.**
- **known-approximation ledger** — keep a short, blunt list in this plan for
  what is still approximate: explicit FEM integration, coarse tetra resolution,
  edge-distance sidewall bend proxies, simplified cavity/bead sealing, first-
  pass thermal curves, Bekker-only soil, and any CPU-only debug paths. The
  thermal curves remain hand-tuned/non-calibrated, but the four-node thermal
  step and wear-depth update now have a CPU+GPU batch path, and the
  non-calibration runner includes `thermal_abuse_envelope` to catch hidden
  viewer grip multipliers by checking cold/hot/worn/abused state ordering.
  This prevents us from calling a scene "realistic" just because it looks good.
  **Initial reusable `TireKnownApproximation` ledger/text report is done with
  tests requiring consequence, GPU status, and next-step accounting for each
  listed approximation. The shared tire rig replay panel used by scenes
  120/121/122 now exposes a collapsed "Known tire approximations" section
  populated from the reusable ledger, including current model, consequence,
  next step, and GPU status. The Newton solver validation path is now also
  listed separately as `newton_solver_cpu_debug_only`, so the CPU-only scalar,
  element, patch, and tiny-mesh Newton reports cannot be mistaken for the
  CPU+GPU explicit/semi-implicit FEM production kernels.**
- **GPU-first implementation rule** — any new high-frequency contact, FEM,
  terrain, or thermal/wear path should either land with a GPU design/test lane
  or be explicitly marked as temporary CPU/debug-only in this plan.

Calibration is skipped for this pass, but the useful non-calibration pieces
from real tire / racing-sim workflows still belong in the project. Add them as
physics, diagnostics, and regression infrastructure rather than fitted target
curves:

What we keep from the "calibration" conversation while explicitly **not**
doing calibration/fitting:

- **same rigs, different purpose** — keep load, slip, camber, thermal,
  obstacle, wet/ice, soil, and moving-ground rigs, but use them to catch
  regressions and impossible signs/monotonicity rather than to fit
  coefficients to measured data;
- **shape checks, not target curves** — compare curve shape only: continuous
  force buildup, smooth combined-slip saturation, sub-linear load response,
  sensible pneumatic-trail sign, and no force teleporting after step inputs;
- **input/output traceability** — every run should record the command inputs,
  preset names, physical state, solver knobs, model provenance, and telemetry
  outputs needed to reproduce the result without an external dataset;
- **acceptance envelopes** — use broad qualitative envelopes such as "ice
  caps force below asphalt", "lower pressure increases patch length", "slip
  heat is non-negative", and "moving ground work has the expected sign";
  **initial moving-ground work sign check is now part of
  `TireEnergySignAudit` / `energy_sign_report`, and the underlying
  `moving_ground_power_w` is exported by `TireRigSnapshot` CSV telemetry.**
- **falsification first** — viewer scenes should expose enough overlays and
  truth labels that a broken tire is obvious visually and numerically;
- **GPU parity matrix** — any high-frequency tire path should say whether it
  is CPU+GPU, CPU-only debug, or planned GPU, with parity deltas reported when
  both implementations exist;
- **no hidden fitted constants** — if a number is hand-tuned, numerical,
  physical, debug, or visualization-only, mark it that way and keep it in a
  named preset or documented parameter group instead of burying it in viewer
  code.

### Non-calibration implementation backlog to keep

Skip external data fitting for now, but keep the engineering infrastructure
that would make the model falsifiable and easier to tune by hand. These items
are not "calibration"; they are internal physics, diagnostics, and regression
work:

- **built-in tire rig runner** — add a reusable in-crate runner that can execute
  the existing load/pressure/slip/camber/thermal/soil/obstacle scenarios with a
  fixed seed, fixed preset names, and deterministic time stepping. It should be
  callable from tests and optionally from the viewer/debug UI. The runner emits
  snapshots only; it must not fit parameters or import datasets. **Initial
  reusable `tire_rig_standard_non_calibration_run` done for
  load/pressure, dry-vs-ice slip, camber/pneumatic trail, wet film, loose soil,
  moving-ground belt work, curb/heightfield contact, ordered rolling-obstacle
  approach/crest/exit FEM snapshots plus a sequence-level acceptance report,
  and thermal/wear cycling, with fixed seed metadata, snapshot CSV output, text
  reports, and tests. The runner text report now also appends reusable
  `gpu_parity=`, `known_approximation=`, and `preset_governance=` rows sourced
  from the same GPU-first checklist, approximation ledger, and preset
  governance audit as the viewer truth labels, so exported rig artifacts carry
  the active force/FEM/terrain GPU status, remaining realism limitations, and
  unit/range/provenance accounting for hand-tuned constants instead of
  requiring a separate UI lookup.**
- **standard snapshot artifact format** — keep one compact, dependency-free
  CSV/text format for rig snapshots: inputs, preset names, model provenance,
  solver knobs, tire state, terrain state, force/moment outputs, patch
  diagnostics, heat/wear, energy, and optional CPU/GPU deltas. This gives us
  repeatable traces without adding external measurement targets. **Initial
  `TireRigSnapshot` CSV plus `TireRigRunOutput::to_snapshot_csv_table` and
  `to_text_report` are done, and the shared viewer replay panel displays the
  optional force-vector/relative CPU-GPU delta fields for selected snapshots.
  The text artifact now includes the tire GPU parity checklist, known
  approximation ledger, and preset-governance audit as first-class rows, tying
  replay evidence to the exact implemented/debug/planned GPU paths, visible
  realism limits, and audited hand-tuned/numerical constants at the time of
  the run. The shared viewer replay panel now also surfaces a compact
  `Text artifact reports` audit that calls the same text exporters
  (`TireRigRunOutput::to_text_report`, `tire_gpu_parity_text_report`,
  `tire_known_approximation_text_report`, and
  `tire_rig_standard_sensitivity_text_report`) and shows whether the exported
  report carries non-calibration, GPU parity, approximation, sensitivity, and
  preset-governance context.
  The same panel now exposes selected-snapshot provenance/command state:
  tire/surface/solver provenance labels, hub height, deflection, speed, camber,
  terrain height/normal, and terrain surface velocity. It also exposes the
  selected snapshot's layered thermal/wear state — surface, tread, and carcass
  temperatures, tread depth, and wear ratio — so heat/wear artifacts are visible
  without opening the CSV row. Patch/friction telemetry from the same snapshot
  is also visible without opening the CSV row: contact/sliding cells, contact
  area, pressure accounting, force-resultant/trail geometry, adhesion/sliding
  fractions, slip speeds, stiffness, and rolling resistance.**
- **acceptance-envelope gates** — promote the broad qualitative checks into
  reusable pass/fail gates that can run in CI: monotonic patch growth with
  lower pressure, sub-linear load response, ice/wet force reduction, finite
  thermal/wear state, non-negative dissipated energy, sane aligning-moment sign,
  and no force discontinuities after step inputs. **Initial runner-level gates
  are done for lower-pressure patch growth, ice force cap, wet-film force
  reduction, dimensionless sanity ratios, and energy/sign accounting; the
  replay panel now shows the dimensionless ratios beside the selected snapshot
  instead of leaving them only in crate-side acceptance reports. The same panel
  now also surfaces selected-snapshot `Acceptance envelopes` directly from
  `TireRigRunOutput::reports`, including each check id, observed metric,
  min/max bound, pass/fail state, and plain-language expectation. Sequence-level
  reports such as `rolling_obstacle_sequence` are associated back to their
  replay snapshots (`rolling_obstacle_approach` / `crest` / `exit`) so selecting
  one of the ordered FEM obstacle frames shows both per-snapshot and sequence
  pass/fail gates, and the replay selector labels those frames as an explicit
  `Obstacle sequence` instead of hiding them behind raw scenario ids. Every
  emitted replay snapshot, including diagnostic snapshots added late in the
  runner, now receives dimensionless sanity and energy/sign reports so generic
  force/load/friction/energy gates do not silently skip structural
  falsification cases.**
- **scenario replay in the viewer** — add a small replay/debug panel for tire
  scenes that can run the same internal rig cases visually: load step, torque
  step, steer step, ice patch, wet patch, curb/cleat, soil rut, and moving belt.
  The panel should show "expected behavior" text beside the live telemetry, not
  a target curve fit. **Initial shared `Internal tire-rig replay` panel is done
  for scenes 120/121/122. It runs the reusable non-calibration rig suite from
  the right rail, lets the user select each snapshot, shows expected behavior,
  acceptance-envelope checks, physics path, force/patch/friction/energy/
  detailed patch/friction telemetry, layered temperature/wear readouts,
  provenance/command/terrain state, and can display the snapshot CSV row. Replay
  cases now use grouped human-readable labels (`Pressure/load`, `Surface slip`,
  `Obstacle sequence`, etc.) so ordered cases such as bump approach/crest/exit
  are discoverable while still preserving the raw scenario id in the telemetry
  and CSV. The structural/diagnostic standard-runner cases are now labeled and
  described in the same selector too: frame/sign convention, construction
  sanity, cavity/bead sanity, sidewall laminate sanity, triangle seam manifold,
  low-speed robustness, contact-patch convergence, road-condition state, and
  realism-source classification are replayable falsification cases instead of
  hidden crate-side report rows. The scenario-specific envelope and transient
  reports are replayable too: pressure patch, ice grip, wet film, banded surface
  transition, rolling/aero loss, longitudinal/combined reference-shape
  diagnostics, and torque/brake/steer step responses all emit snapshots that
  the viewer can select. `TireRigRunOutput::reports_without_replay_context`
  now guards this contract: every acceptance report must either attach to a
  selectable snapshot or to a named ordered sequence whose frames are
  selectable.**
- **truth overlays as falsification tools** — keep expanding overlays that make
  wrong behavior obvious: deformed tread nodes, pressure bins, contact normals,
  bristle vectors, adhesion/sliding colors, force/resultant/trail markers,
  cavity pressure/volume, rut depth, thermal layers, and CPU/GPU deltas.
  **Scene 122 now derives `TirePressureDistributionTelemetry` from the live
  tire contact and shows longitudinal/widthwise pressure-bin load readouts,
  pressure-span, peak pressure, and load conservation beside the FEM patch
  diagnostics.**
- **parameter registry and provenance audit** — make every tire/surface/soil/
  thermal/solver preset report its units, recommended range, and provenance
  (`physical`, `hand_tuned`, `numerical_stability`, `debug`, or
  `visualization`). Viewer sliders should be stored as explicit preset
  overrides in telemetry, never as hidden constants. **Initial tire-viewer
  control categories are now surfaced from `TireViewerControlMetadata` in the
  truth-label panel. The same panel now exposes a collapsed
  `Preset provenance/range audit` sourced directly from
  `molla_vehicle::tire_preset_governance_report()`, including units,
  provenance labels, broad ranges, pass/fail status, and notes for named tire,
  surface, thermal, solver, and soil presets.**
- **sensitivity maps instead of fitted curves** — for pressure, load, speed,
  camber, temperature, water film, tread wear, soil stiffness, and obstacle
  height, keep coarse sweeps that report direction and boundedness of the
  response. These maps are for "did the model react sensibly?" checks, not for
  matching any real tire. **Initial standard sensitivity map/report done for
  pressure patch growth, sub-linear load response, wet-speed force loss,
  camber thrust, state-derived overheat force loss, wet-film/worn-tread loss,
  soft-soil material force loss, and obstacle-height normal response, with
  the same rows surfaced in the default rig runner/viewer replay panel.**
- **CPU/GPU parity reports for every high-frequency path** — each per-wheel,
  per-contact, FEM, terrain, soil, and thermal/wear path must say one of:
  CPU+GPU with a parity test, CPU-only debug with a reason, or planned GPU with
  a concrete follow-up. The viewer should display deltas when both paths exist.
  **The shared tire truth-label panel now includes a collapsed GPU parity
  checklist sourced directly from `molla_vehicle::tire_gpu_parity_checklist()`,
  so all tire scenes expose the same CPU/GPU/proof/reason/follow-up rows used
  by crate-side tests instead of duplicating stale viewer text.**
- **solver-budget and stability envelope report** — every FEM/tread-contact
  scenario should record substeps, XPBD iterations, compliances, max correction,
  max node speed, volume error, dissipated power, and energy balance so a
  "stable" tire cannot hide numerical explosions. **Initial FEM
  `fem_solver_budget` snapshot is now part of the default non-calibration rig
  runner, participates in snapshot CSV/text artifacts, and the shared viewer
  replay panel shows the FEM substep/XPBD/error budget when that snapshot is
  selected.**
- **reference-shape diagnostics only** — optional Magic-Formula/TMeasy/racing-
  sim-like curves may be added later only as shape/sign/saturation comparators
  against the native brush/FEM output. They must not become the force source
  and must not introduce fitted coefficients in this pass. **The default
  non-calibration rig runner now includes dry longitudinal and combined-slip
  reference-shape acceptance reports, so force buildup/saturation/tradeoff
  checks are part of the standard text report without fitting a curve.**
- **plain-language scene checklists** — every tire viewer scene should keep a
  short right-rail checklist of what should happen and which approximation is
  active. This is how broken scenes get reported: "ice should still slip",
  "soft tire should flatten and grow the patch", "soil should rut under load",
  etc. **The tire truth-label panel now renders concrete checklist rows for
  the tire lab and scenes 120/121/122 beside the active physics path,
  approximations, parity checklist, and control categories. Scene 119
  (`SoilCompaction`) now also has a truth label for the roller-driven
  DEM→FEM bridge: it names the sparse-LBVH owner, resident GPU DEM owner,
  terrain-row/surface-velocity/reseed telemetry, and the roller-load proxy
  approximation. Scene 118 (`BekkerSinkage`) now has the matching soil-law
  diagnostic label/control categories: plate pressure and Bekker stiffness/
  exponent are physical soil inputs, the pressure-sinkage curve is a
  visualization/falsification readout, and the scene is clearly separated from
  Scene 122's resident GPU Bekker tire/FEM terrain owner. Scene 110
  (`VehicleOffroad`) now also gets the same tire
  truth-label/control-category treatment so its legacy CPU
  `RealBrushTireModel` bridge is explicitly labelled as debug/adoption
  coverage, with resident GPU vehicle/brush/FEM ownership called out as the
  follow-up instead of being implied.**
- **known-approximation audit stays live** — whenever a shortcut remains
  (explicit FEM, coarse tetra mesh, first-pass sidewall constraints, Bekker-only
  soil, CPU-only DEM, missing full triangle manifold), update the ledger with
  consequence, GPU status, and next-step work. **The shared viewer replay and
  truth-label panels now read directly from that ledger, so updates are visible
  in tire scenes without duplicating text in viewer-local strings.**

- **internal tire-rig snapshots** — every rig run should be able to emit a
  compact snapshot of inputs, preset names, tire state, terrain state, forces,
  moments, patch bounds, heat/wear, and CPU/GPU deltas. These snapshots are for
  repeatability and debugging, not for fitting against measured tire data.
- **reference-shape comparisons, not reference fitting** — keep optional
  comparisons against brush/Magic-Formula/TMeasy-style curve shapes only as
  diagnostics: monotonic load response, smooth saturation, combined-slip
  tradeoff, and sane aligning-moment sign. The native FEM/brush path remains
  the source of force. **Initial reusable
  `TireReferenceShapeDiagnostics`,
  `tire_rig_longitudinal_reference_shape_diagnostics`, and
  `tire_rig_combined_reference_shape_diagnostics` are done with tests for
  broad dry-asphalt force buildup, bounded adjacent force jumps, combined-slip
  tradeoff, friction-budget limits, and pneumatic-trail moment sign; the
  default runner now emits longitudinal and combined reference-shape reports
  beside its other non-calibration envelope gates.**
- **pressure-distribution diagnostics** — expose contact pressure along width
  and circumference so a stiff tire shows a shorter/higher-pressure patch,
  a softer tire spreads load over more cells, and obstacle contact does not
  collapse into one invisible point. **Initial reusable
  `TirePressureDistributionTelemetry` done for brush/FEM contacts, including
  longitudinal and lateral bins, pressure/load conservation checks, high-vs-
  low-pressure patch regressions, and FEM widthwise pressure-bin coverage.**
- **step-response regressions** — add deterministic step torque, step brake,
  and step steer fixtures to catch force teleporting, relaxation instability,
  sign flips, and low-speed NaNs. **Initial reusable
  `tire_rig_step_torque_response`, `tire_rig_step_brake_response`, and
  `tire_rig_step_steer_response` fixtures are done with tests checking raw-vs-
  effective slip/angle lag, progressive wheel-speed/force response, finite
  telemetry, reversal-through-zero behavior, speed-dependent convergence,
  friction-budget, and dissipative energy accounting. The default
  non-calibration runner now emits step torque/brake/steer acceptance reports
  so these regressions are visible in standard text output and viewer replay
  refreshes, not only isolated unit tests.**
- **surface-transition regressions** — add rigs/scenes where the same tire
  crosses asphalt → ice, dry → wet film, packed dirt → loose soil, and moving
  belt/ground patches. The expected behavior is qualitative but explicit:
  force should change continuously, slip should rise on low-μ surfaces, and
  soil should add rut/rolling resistance. **Initial reusable
  `TireRigSurfaceTransitionTerrain` / `tire_rig_surface_transition_sequence`
  done for spatial dry-asphalt, ice, wet-film, packed-dirt, loose-soil,
  smoothed transitions, and moving-belt bands, with tests for force reduction,
  water-film recording, packed-dirt/loose-soil rolling-resistance ordering,
  finite telemetry, and moving-ground force changes. The standard
  non-calibration runner now emits a `surface_transition_sequence` acceptance
  report for ice force capping, wet force reduction, packed-dirt rolling loss,
  loose-soil-over-packed-dirt rolling loss, moving-belt velocity,
  moving-belt force/work from relative ground velocity, and bounded adjacent
  force jumps. `TireHeightfieldSurfaceVelocityMapReport` now adds a
  `heightfield_surface_velocity_map` replay/report: a row-major heightfield
  velocity map must sample static versus moving regions, change tire force and
  terrain work through relative ground motion, export the sampled velocity in the
  replay snapshot, and report non-zero terrain work.**
- **vehicle-level sanity scenes** — keep at least one heavy loaded vehicle,
  one low-friction driven vehicle, one braking vehicle, one steering/camber
  vehicle, one obstacle/curb vehicle, and one soft off-road vehicle using the
  same reusable tire APIs. These are not calibration scenes; they are
  falsification scenes for "does this look and account like physics?".
  **Reusable `tire_vehicle_sanity_scenarios` now returns four-wheel heavy-load,
  low-friction driven, braking, steering/camber, curb/obstacle, and soft-soil
  off-road falsification cases with telemetry and broad pass/fail envelopes;
  the default tire-rig runner carries those cases into its pass/fail accounting
  through explicit reusable acceptance checks for vehicle load accounting,
  patch/contact support, slip bounds, friction-budget use, and each
  scenario-specific expectation. The shared viewer replay panel shows the
  vehicle sanity list beside the internal rig snapshots and surfaces the
  per-case check count/failures so these cases are falsification artifacts, not
  only descriptive UI rows. The default runner also now includes
  `vehicle_dynamic_load_transfer`, a replayable load-transfer/suspension gate
  that checks front/rear load shift, patch growth, sub-linear tire-force gain,
  and per-corner spring travel under braking and acceleration.**
- **failure/condition from state only** — cold, hot, overheated, abused,
  worn, and wet states must derive from thermal/wear/film/force history. Do
  not add viewer flags that secretly change grip without a physics state
  variable and telemetry field. **Initial reusable
  `TireConditionFailureModeReport` now gates cold/healthy/hot-or-overheated/
  abused condition labels, state-derived response flags, live μ reduction,
  carcass softening, rolling-loss increase, and tractive-force loss through
  `TireThermalLayers`, `TireWearState`, and `TireConditionMaterialResponse`.
  The shared tire truth-label panel surfaces this as `condition_failure_modes`
  beside the thermal/soil gates, so failure-mode grip changes are inspectable
  from model state instead of hidden viewer switches.**
- **solver-budget tracking** — record substep count, XPBD iterations,
  compliance values, max correction, max velocity, and energy/heat terms in
  debug telemetry so a "stable" tire is not hiding numerical explosions.
  **Initial reusable `TireSolverBudgetTelemetry` is done for tetra-FEM
  snapshots/reports, and the default rig runner now emits a FEM
  `fem_solver_budget` snapshot with those fields.**
- **GPU parity checklist** — for each high-frequency feature, record whether
  it is CPU+GPU, CPU-only-debug, or planned-GPU. Any CPU-only production path
  must have an explicit reason and a follow-up item. **Initial reusable
  `tire_gpu_parity_checklist()` / `tire_gpu_parity_text_report()` are done in
  `molla-vehicle`, covering brush, FEM node/hub/XPBD/contact, the CPU+GPU
  semi-implicit stiffness stabilizer, CPU+GPU batched four-node thermal/wear state,
  CPU+GPU batched scalar layered transient relaxation,
  finite-triangle contact, CPU+GPU Bekker deformable-soil work, and first
  CPU+GPU MPM material-point tire feedback with FEM terrain ownership.**
  The finite-triangle GPU path now also carries per-triangle broadphase sphere
  metadata so far mesh patches can be rejected before closest-point contact.
  The Bekker row now records GPU sinkage-map sampling, GPU height/sinkage
  writeback, and resident Bekker terrain buffers feeding FEM feedback; the DEM
  row now records CPU+GPU localized tire-cell impulse
  transfer plus dense, host-built cell-grid, unbounded sorted-grid CPU oracle,
  unbounded GPU key/radix/range resident grid step, and compatibility-wrapper
  coverage for the old GPU-built fixed-capacity API; the MPM row now records
  CPU+GPU
  localized material-point tire pressure/shear/compaction feedback,
  `TireMpmFemTerrainOwner` terrain export, and the first in-place
  `MpmSimulation` tire-feedback composition bridge. FEM contact feedback can
  now consume GPU-rasterized MPM terrain buffers directly, and FEM contact
  reduction diagnostics can read the same resident buffers; DEM now has a
  Morton-sorted sparse-LBVH CPU oracle/node layout plus GPU Morton-key/radix
  sorted-order, leaf AABB/range row, and padded complete-binary tree-row
  builder stages, resident upload/dispatch of GPU-built tree rows, no-readback
  resident builder/dispatch for arbitrary leaf counts, a larger 14×12×5
  sparse-bed CPU/reference stress test, dispatch-plan sizing telemetry, an
  ignored GPU sparse-LBVH traversal parity lane, an ignored large no-readback
  resident-builder parity lane, a top-layer
  `surface_velocity_map` export for FEM terrain handoff accounting,
  `TireDemResidentLbvh` capacity/revision tracking with
  `update_resident_lbvh` in-place topology refresh,
  `upload_resident_lbvh_tree_rows` for GPU-built row atlases, and
  `build_resident_sparse_lbvh_tree_rows_gpu_radix` for no-readback resident
  topology construction; first
  viewer/offroad ownership is wired through the
  `Scene::TireRealistic` MPM bridge.
- **scene truth labels** — every tire viewer example should include a short
  right-rail description of what the user should see, what physics path is
  active, which approximations remain, and which sliders are physical vs
  debug/visualization overrides. **`Tire truth label` panel coverage now spans
  every tire/offroad tire-adjacent viewer example: `TireModelLab`,
  `VehicleOffroad`, `BekkerSinkage`, `SoilCompaction`, `TireDynamics`,
  `TireTraction`, and `TireRealistic`, with tests guarding section content and
  control-role metadata for all of them. Scenes 120/121/122 also share the live
  `TireKnownApproximation` replay-panel ledger so approximation status is
  sourced from the reusable crate instead of stale viewer prose. The same
  truth-label panel now lists classified tire controls plus the reusable GPU
  parity checklist, known-approximation ledger, preset governance audit,
  realism-source classification report, and plain-language scene checklist
  rows, and tests require each tire/offroad scene to expose physical/preset
  controls plus debug/visual helpers. Scene 121's patch-size helper is now
  explicitly named `visual_patch_factor` and guarded as visualization/readout
  only, so it cannot be confused with a hidden traction multiplier in the
  force path.**

### A. Camber, pneumatic trail, and better aligning torque

Current brush contact reports a coarse self-aligning yaw moment from lateral
force distribution along the patch. Upgrade this into a real wheel-orientation
model:

**Initial implementation status:** reusable CPU and GPU brush paths now carry
this orientation model. `RealWheelOrientation` extends the legacy
`RealWheelState` heading with an explicit wheel-up vector, and
`evaluate_brush_tire_oriented` computes camber angle, camber pressure bias,
camber thrust, contact centre, force-resultant centre, pneumatic trail, and
aligning moment from distributed patch forces. Scene 121 rotates the visible
wheel geometry by steering/camber, uses those contact outputs for force/yaw
updates, and shows camber, pneumatic trail, aligning moment, and contact-centre
readouts. The GPU brush row now packs camber in `extra.w` and returns camber /
contact-centre / pneumatic-trail diagnostics in `diag3`; the GPU parity
checklist tracks this as `brush_camber_pneumatic_trail` with CPU tests and an
ignored hardware parity lane.

- Extend `RealWheelState` with wheel up/camber basis or an orientation frame
  instead of only `heading`.
- Compute camber angle relative to the terrain normal.
- Add camber thrust in the brush contact:
  - low camber produces small lateral force bias;
  - high camber changes lateral force and contact pressure distribution.
- Track contact-centre / centre-of-pressure shift along the patch.
- Compute pneumatic trail:
  - `trail = contact_center_x - force_resultant_x`;
  - `Mz = -Fy * trail + residual aligning torque`.
- Add viewer readouts for:
  - camber angle,
  - pneumatic trail,
  - aligning torque,
  - contact centre shift.
- Add tests:
  - zero camber gives near-zero camber thrust;
  - positive/negative camber flips camber-thrust sign;
  - aligning torque has sane sign and saturates as the patch slides.
- Extend later with conicity / ply-steer style lateral bias as a structural
  tire property, not as viewer-side steering drift. **Initial CPU+GPU brush
  path done via `RealTireParams::structural_lateral_bias_per_normal`; it adds
  a signed lateral load-bias before the shared friction cap, has a CPU
  sign/budget regression, and the GPU brush shader consumes the same value from
  the packed wheel row.**
- Make cornering stiffness load/pressure dependent so the same slip angle does
  not produce a linear force at every load. **Initial reusable path done:**
  `brush_pressure_stiffness_factor` now shifts CPU brush tread stiffness, the
  FEM tread-contact / contact-wrench feedback path uses the same pressure
  factor, the GPU brush shader mirrors it, and regression/source-guard tests
  verify pressure changes small-slip cornering stiffness without bypassing the
  shared friction cap.

### B. Transient tire dynamics / relaxation length

Current brush force is mostly steady-state each frame. Real tires do not build
force instantly; the contact patch, tread band, and carcass lag behind wheel
slip/steering changes.

Add per-wheel transient state:

```rust
RealTireTransientState {
    kappa_eff,
    alpha_eff,
    lateral_deflection,
    longitudinal_deflection,
    tread_yaw,
}
```

Model goals:

- Add longitudinal and lateral relaxation lengths:
  - `dkappa_eff/ds = (kappa - kappa_eff) / Lx`;
  - `dalpha_eff/ds = (alpha - alpha_eff) / Ly`.
- Use effective slip states in the brush law, not raw instantaneous slip.
- Add tread yaw / carcass lag so steering input rotates the contact patch
  progressively instead of immediately.
- Low speed must remain stable:
  - keep current low-speed regularization;
  - clamp relaxation updates by `dt` and travelled distance.
- Add tests:
  - step steering input produces force rise over several frames;
  - reversing steering decays through zero instead of teleporting;
    **explicit regression done: `transient_steering_reversal_decays_through_zero`
    proves the first reversed sample keeps the previous sign and later crosses
    zero into the new sign**;
  - high speed converges faster in time for the same relaxation length;
    **explicit regression done:
    `transient_relaxation_converges_faster_in_time_at_higher_speed` checks
    longitudinal and lateral effective slip after equal wall-clock time**;
  - no NaNs at near-zero speed. **done for alternating near-zero velocity and
    spin in `transient_relaxation_stays_finite_at_near_zero_speed`; the
    standard runner now also emits a `low_speed_robustness` acceptance report
    covering launch, lockup, reversal, finite outputs, bounded regularized
    slip, and friction-budget accounting.**
- Viewer:
  - show raw vs effective slip angle/ratio;
  - show transient force lag after steering/torque steps. **Initial default
    runner reports now include step torque/brake/steer transient acceptance
    gates, and scene 121 already shows raw/effective slip readouts.**
- Make relaxation length depend on pressure, load, speed, and temperature:
  overloaded/soft/hot tires should respond more slowly than stiff cold tires.
  **Initial reusable `TireRelaxationContext` and
  `RealTireTransientState::step_with_context` done: pressure/load/carcass-
  temperature scale the existing distance-based relaxation lengths, with tests
  showing soft overloaded hot tires build longitudinal/lateral slip more
  slowly than stiff cold/light tires.**
- Add separate carcass/belt/tread-band lag states so the model can distinguish
  quick tread shear from slower tire-body deformation. **Initial reusable
  `TireLayeredTransientState` done: it tracks fast tread, intermediate belt,
  and slow carcass effective slip/angle/yaw/deflection layers, exposes
  context-aware brush evaluation helpers, and tests confirm tread > belt >
  carcass lag ordering plus pressure/load/temperature slowdown across every
  layer. `TireLayeredTransientEnvelopeReport` now brings that ordering into
  the standard non-calibration runner as `layered_transient_envelope`, gating
  tread/belt/carcass ordering, aggregate effective slip placement, hot/soft/
  overloaded slowdown, bounded effective slip, and force buildup over repeated
  samples; the shared viewer replay selector labels it as a first-class
  transient case. GPU batching is now started too: `TireTransientGpuKernel` /
  `TireTransientGpuBatch` / `tire_transient.wgsl` updates the same
  tread/belt/carcass scalar state for many wheels and has CPU reference/shader
  tests plus ignored hardware one-shot and persistent parity lanes. Scene 121
  now dispatches this persistent transient batch before its GPU brush solve,
  maps the effective slip back into the wheel state, and keeps CPU fallback
  for parity/debug. Scene 122 now also dispatches the persistent transient
  batch before FEM tread feedback: raw belt/slip controls are relaxed into
  effective κ/α, the reusable transient-to-FEM adapter drives feedback omega
  and moving-ground surface velocity, and the right rail reports raw vs
  effective slip plus feedback force/bristle diagnostics. Remaining deeper
  work is fully GPU-resident vehicle/chassis ownership beyond the single-tire
  bench. The shared tire truth-label panel now also surfaces
  `layered_transient_envelope` inside `Thermal/transient/soil gates`, showing
  tread/belt/carcass lag ordering, hot/soft slowdown, and force buildup from
  the reusable report rather than relying only on per-scene readouts.**

### C. Multi-layer thermal and wear model

Current thermal state is one lumped tread temperature and one wear depth. Keep
that as fallback, but add a richer model that is still cheap enough for the
viewer.

**Initial implementation status:** done for the reusable CPU model, scene 122
viewer wiring, and a batched GPU thermal/wear-step parity lane. `TireThermalState`
remains as the fallback compatibility type; `TireThermalLayers` is the richer
CPU reference path. `TireThermalGpuKernel` / `TireThermalGpuBatch` /
`tire_thermal.wgsl` mirrors the four-node surface/tread/carcass/rim thermal
update and tread wear-depth integration for many wheels with persistent
buffers, then emits the same thermal μ, carcass pressure, rubber-peak, wear
ratio, worn-tread μ, and wear-temperature diagnostics used by condition
feedback. Scene 122's right rail now exposes a GPU thermal/wear batch toggle
and backend/error readout; the default path dispatches the persistent
`TireThermalGpuBatch`, writes the result back into the shared thermal/wear
state, and then derives `TireConditionReport` from that state. The same scene
also runs the loaded internal rig replay snapshots through a persistent GPU
thermal/wear batch via reusable GPU-ready rig replay artifacts and exposes the
replay batch backend/count/error beside the replay panel. The shared internal
tire-rig replay panel now also shows the selected replay snapshot's recorded
surface/tread/carcass temperatures, tread depth, and wear ratio in a collapsed
`Thermal / wear snapshot` section, explicitly labelling these as state-derived
telemetry rather than hidden grip multipliers. `TireConditionReport` now
derives cold/hot/overheated/abused condition, abuse index, grip factor, carcass
stiffness factor, and wear-rate multiplier from thermal layers plus tread
wear. `TireThermalConditionSanityReport` adds the first reusable
non-calibration condition sanity gate for heat/cool replays, and the standard
non-calibration runner now attaches those gates to `thermal_wear_cycle` so
finite condition outputs, cooling drop, non-healing wear, positive μ factor,
and positive pressure factor are checked beside the replay snapshot. The
runner now also emits separate `thermal_wear_heat_peak` and
`thermal_wear_cooling_recovery` replay snapshots, with explicit gates for hot
sliding above ambient, non-negative dissipated power, zero-slip cooling,
surface-temperature recovery, and non-healing tread wear, so the heat and cool
phases can be inspected independently in the viewer. `TireWearPowerTemperatureReport`
now adds the missing power/temperature wear sensitivity gate as
`thermal_wear_power_temperature`: higher sliding power must remove more tread
at the same surface temperature, and a hot surface must remove more tread at
the same sliding power. `TireWornSurfaceSensitivityReport` now adds the worn-
tread surface-vulnerability gate as `worn_surface_sensitivity`: wet film, ice,
and loose soil must lose more drive from worn tread than dry asphalt through
the native thermal/wear/material response path. The shared tire truth-label panel now also renders
`thermal_condition_sanity` directly from the deterministic heat/cool rig so
those state-derived heat/wear gates are visible next to the
thermal/transient/soil reports, not only in runner text.
`TireThermalAbuseEnvelopeReport` adds a second standard-runner thermal gate
for healthy/cold/overheated/worn/abused ordering, and the shared viewer replay
selector labels `thermal_abuse_envelope` as a first-class thermal/wear abuse
case so the expected state-derived grip loss is visible beside the pass/fail
envelope. `TireThermalRollingCarcassReport` adds the sustained low-/zero-slip
rolling case that was still missing from the plan: it runs rolling resistance
and casing-flex heat through the same four-node thermal model, gates nonzero
carcass/tread heating, pressure-factor rise, clamped finite μ, and near-zero
sliding power, and exposes the result as the replayable
`thermal_rolling_carcass` viewer case. Remaining thermal work is deeper rubber
chemistry validation if/when calibration becomes explicit, not external
calibration in this pass. `TireThermalBrakeRimReport` now gates the optional
brake/rim heat-source path from this section too: `thermal_brake_rim` injects
heat at the rim with zero brush sliding, then checks rim-first heating,
carcass conduction, pressure-factor feedback, finite/clamped μ, and no tread
wear growth from fake sliding. The shared tire truth-label panel now surfaces
`thermal_abuse_envelope`, `thermal_rolling_carcass`, and `thermal_brake_rim`
under `Thermal/transient/soil gates`, so cold/hot/worn/abused ordering,
rolling/carcass heat, and brake-to-rim heat flow are visible as state-derived
gates without hidden scene switches.

Add:

```rust
TireThermalLayers {
    surface_c,
    tread_c,
    carcass_c,
    rim_c,
}
```

Model goals:

- Heat sources:
  - sliding/brush dissipation -> surface/tread;
  - cyclic carcass flex -> carcass;
  - rolling resistance -> carcass/tread;
  - optional brake/rim heat source -> rim/carcass.
- Heat sinks:
  - road conduction from surface;
  - air convection from tire surface;
  - conduction between surface/tread/carcass/rim.
- Friction feedback:
  - surface temperature affects peak grip quickly;
  - carcass/tread temperature affects pressure/vertical stiffness slowly.
- Wear:
  - wear rate depends on sliding power and surface temperature;
  - excessive temperature accelerates wear;
  - worn tread reduces wet/loose/ice performance more than dry asphalt.
    **The standard runner now emits `thermal_wear_power_temperature`, with
    acceptance gates proving tread loss increases with both sliding power and
    hot surface temperature from explicit wear state. It also emits
    `worn_surface_sensitivity`, with gates proving worn tread hurts wet film,
    ice, and loose soil more than dry asphalt through explicit wear/surface
    state.**
- Add tests:
  - sliding heats surface faster than carcass;
  - sustained rolling heats carcass even with low sliding;
  - cooling brings surface down faster than carcass;
  - wear increases with sliding power and hot surface temperature;
  - friction factor stays finite and clamped;
  - state-derived condition report labels cold/healthy/hot/abused tires and
    keeps grip/stiffness/wear multipliers finite. **done**
- Viewer:
  - show surface/tread/carcass/rim temperatures;
  - tint tire by surface temperature;
  - show carcass temperature separately in the right rail;
  - show condition label, abuse index, and condition-driven wear-rate
    multiplier in scene 122. **done**
- Add racing-sim style failure modes only when they come from state variables:
  overheating reduces grip, cold tires under-grip, severe hot sliding increases
  wear, and long high-energy running can produce a visible "abused" tire state.
  **initial state-derived report + scene-122 wiring done; reusable
  `TireConditionMaterialResponse` now applies those states to live μ,
  pressure, carcass stiffness, tread shear, bristle damping, rolling loss, and
  explicit cold/overheated/softened/worn/abused flags, with regression tests
  checking reduced tractive force for abused worn tires.**
- Keep the model deterministic and cheap enough for many wheels; expensive
  chemistry/rubber aging belongs outside the current scope.

### C2. Load sensitivity and combined-slip force law

The current brush model already shares a friction cone between longitudinal and
lateral force, but the next pass should make the force buildup and saturation
more tire-like without turning the model into pure curve fitting.

**Initial implementation status:** done for the reusable CPU brush model,
`TireBrushGpuKernel`, and scene 121 readouts. The current model uses
hand-tuned load sensitivity from local normal pressure, a broad
pressure-dependent stiffness factor, and a smooth vector saturation curve;
future passes can refine the exponents/curves without changing the API. The
standard non-calibration runner now emits `load_combined_slip_law`, which gates
sub-linear load sensitivity, a shared combined-slip budget, smooth
adhesion-to-sliding saturation, ice force capping, and hot/worn state-derived
tread-shear reduction. The same report now also gates effective longitudinal
and cornering stiffness outputs under hot/worn condition state, so temperature
and wear must change the actual brush telemetry rather than only a parameter
row. It also records the start/end sliding fractions through the saturation
sweep and gates their growth, proving the sliding-fraction diagnostic follows
the adhesion-to-sliding transition.

Model goals:

- Add vertical-load sensitivity:
  - peak friction factor decreases mildly at very high load;
  - longitudinal and cornering stiffness grow sub-linearly with load;
  - pressure can shift those stiffnesses through the CPU/FEM/GPU brush path;
  - temperature can shift those stiffnesses through the condition-material
    response path. **`load_combined_slip_law` now explicitly gates hot/worn
    reductions in both effective longitudinal stiffness and effective
    cornering stiffness.**
- Replace hard force clipping with a smooth brush saturation curve:
  - force should transition progressively from adhesion to sliding;
  - combined slip should form a smooth friction ellipse/friction cone;
  - sliding fraction should be reported as a diagnostic.
    **`load_combined_slip_law` now gates start/end sliding-fraction growth
    across the saturation sweep.**
- Preserve physical inputs:
  - pressure,
  - vertical load,
  - patch length/width,
  - material friction,
  - bristle/tread stiffness,
  - temperature/wear factors.
- Add outputs:
  - adhesion fraction,
  - sliding fraction,
  - available friction budget,
  - consumed friction budget,
  - effective cornering stiffness,
  - effective longitudinal stiffness.
- Add tests:
  - doubling load gives less than double peak force;
  - combined high braking + cornering reduces each axis correctly;
  - force remains continuous around the adhesion/sliding boundary;
  - ice still caps force far below asphalt even with a large patch.

### D. FEM/contact coupling

Current `TetraTireFem` deforms the carcass and has tread-node contact, while
the reusable brush API still works mostly from a sampled ideal tire patch.
The important next realism step is to make the FEM deformation drive the
traction contact directly.

**Initial implementation status:** partially done on CPU. The new
`TetraTireFem::evaluate_tread_contact` builds the wheel-level contact from
actual deformed tread nodes and scene 122 uses it for live force/heat/wear
readouts. `TetraTireFem::step_on_terrain` also feeds terrain height/friction/
surface velocity into the CPU tread-node projection.
`TetraTireFem::apply_tread_contact_wrench` now distributes the FEM-derived
wheel-level brush wrench back into the source tread nodes and uses persistent
per-node bristles for the CPU wheel-level solve. `TireFemContactReductionKernel`
now provides the first GPU FEM contact diagnostics reduction from actual
tread-node positions, including heightfield height, friction-map, water-film
sampling, built-in sphere/box SDF primitives, and mean-film diagnostics. It
also reduces local terrain/SDF-normal contact load into a world-force vector
and world moment.
`TireFemContactFeedbackKernel` now provides first-pass GPU bristle force
feedback into FEM node velocities, including heightfield normal, friction-map,
water-film sampling, built-in sphere/box SDF primitive contact, finite triangle
patch sampling with a first per-triangle broadphase sphere, uploaded terrain
surface velocity for moving-ground relative slip, and deformable-terrain
sinkage/rut-map sampling for Bekker-style rolling loss. `TireBekkerGpuKernel`
now provides first GPU Bekker height/sinkage writeback from tire contact-cell
pressure and shear. `TireBekkerFemTerrainOwner` now provides the first
reusable owner that applies those maps and exports `TireFemContactTerrain` for
GPU FEM contact sampling, and `Scene::TireRealistic` now hooks that owner into
the live viewer terrain/rut display. `Scene::SoilCompaction` can now step its
visible grain bed through the sparse-LBVH DEM owner and write the resulting
state back into the scene, and the sparse-LBVH path has a larger CPU/reference
stress test for packed row/layout sanity.
`TireDemSoilFeedbackReport` now brings the DEM tire-contact-cell coupling into
the standard non-calibration runner as `dem_soil_feedback`: it checks that
soft-soil tire contact cells affect grains, transfer normal/shear load,
produce positive sinkage/rolling-loss, and that driven slip injects more DEM
shear and horizontal grain motion than free rolling.
`TireMpmSoilFeedbackReport` similarly brings a crate-side material-point
tire-soil feedback contract into the standard runner as `mpm_soil_feedback`:
it applies real tire contact cells to a small MPM-style material plug and gates
positive compaction, normal/shear impulse, comparable normal compaction under
slip, and increased horizontal material velocity for FEM terrain export.
The shared tire truth-label panel now surfaces both `dem_soil_feedback` and
`mpm_soil_feedback` inside `Thermal/transient/soil gates`, exposing active
contact cells/material points, shear transfer, grain/material motion, sinkage,
and compaction evidence directly in the viewer.
Open work remains: fully GPU-resident FEM
terrain feedback in the wider scene/solver pipeline, broader viewer/runtime
adoption of the resident coupling APIs, and accelerated full triangle-surface
contact/manifold support beyond the first finite-triangle sampling path. The
first no-readback feedback primitive now exists:
`TireFemContactFeedbackKernel::dispatch_resident_with_gpu_terrain` returns
GPU-resident velocity/bristle/force buffers as
`TireFemContactFeedbackResidentOutput` instead of staging rows back to CPU.
Those feedback force rows can now drive resident atlas hubs directly:
`tire_fem_hub_params_with_feedback_output` is the CPU/reference adapter, while
`update_resident_atlas_hub_params_from_feedback_output` dispatches
`tire_fem_atlas.wgsl::hub_params_from_feedback_force_rows` to sum each tire's
packed node-force range into resident atlas hub external-force rows before
`step_resident_atlas_hubs_from_feedback_output` integrates the hubs without
staging the force rows through CPU.
The reduced-world-force-to-hub path is now resident too:
`update_resident_atlas_hub_params_from_contact_reduction_output` dispatches
`tire_fem_atlas.wgsl::hub_params_from_contact_reduction_outputs` to add packed
FEM contact-reduction row-2 `world_force.xyz` directly into resident atlas hub
parameter rows, and `step_resident_atlas_hubs_from_contact_reduction_output`
integrates those rows without overwriting them from host parameters.
The same packed contact-reduction rows can now drive resident chassis state:
`dispatch_planar_from_contact_reduction_output_blocking` /
`integrate_planar_from_contact_reduction_outputs` and
`step_planar_resident_from_contact_reduction_output` /
`integrate_planar_state_from_contact_reduction_outputs` consume row-2
`world_force.xyz` plus row-3 `world_moment.y` directly for planar X/Z/yaw
integration without staging reduced FEM contact wrenches through CPU.
The direct FEM contact-feedback force rows now have the same vehicle coupling
bridge: `tire_vehicle_planar_from_feedback_output_reference` sums each tire's
atlas node-force range for CPU/reference parity, and
`step_planar_resident_from_feedback_output` dispatches
`tire_vehicle_yaw.wgsl::integrate_planar_state_from_feedback_force_rows` so the
resident `TireFemContactFeedbackResidentOutput::forces` buffer can drive
planar X/Z/yaw chassis state without first collapsing through CPU staging; the
ignored `tire_vehicle_planar_resident_state_ping_pongs_from_feedback_output`
hardware lane covers that resident feedback-force buffer handoff.
The first reduced-wrench-to-hub handoff also exists:
`tire_fem_hub_params_with_contact_reduction` maps GPU contact-reduction
`world_force` into `TireFemHubParams::external_force_xyz`, and
`tire_fem_hub.wgsl` now applies all XYZ external-force components.
`tire_fem_hub_params_with_brush_outputs` now adds the first reusable
vehicle/brush-to-FEM hub assembly bridge: per-wheel `BrushWheelOutput`
`world_force` rows are mapped into FEM hub external-force parameters for
`step_resident_atlas_hubs` without hiding sign or scaling in viewer scene code.
That host-side adapter now has the matching GPU-resident writer:
`update_resident_atlas_hub_params_from_brush_output` dispatches
`tire_fem_atlas.wgsl::hub_params_from_brush_outputs` to update resident atlas
hub parameter rows directly from GPU-resident `BrushWheelOutput` buffers before
hub integration. `step_resident_atlas_hubs_with_resident_params` then advances
the hubs without overwriting those GPU-assembled rows, and
`step_resident_atlas_hubs_from_brush_output` plus
`step_resident_vehicle_fem_hubs_from_brush_owner` provide the first reusable
resident vehicle→brush→FEM-hub sequence. `step_resident_vehicle_fem_atlas_from_brush_owner`
now wraps the first coherent resident substep scheduler: fused FEM atlas node
stepping, resident vehicle/brush stepping, GPU brush-output-to-hub-row
assembly, and resident hub integration happen in one reusable no-readback API.
`TireResidentVehicleFemAtlasOwner` now bundles that resident chassis/brush
owner with the resident fused FEM atlas, and
`create_resident_vehicle_fem_atlas_owner` /
`step_resident_vehicle_fem_atlas_owner` expose the first long-lived combined
runtime owner API. Scene 120 now has the first viewer adoption of this combined
owner: its default GPU path builds a resident vehicle+brush+FEM owner, steps
vehicle brush rows and FEM atlas hubs through that owner, and keeps raw
resident-atlas plus per-tire persistent GPU fallbacks as explicit debug
toggles. The remaining work is moving the more complete live vehicle scenes
and multi-scene runtime code onto this owner, plus deeper vehicle-level
ownership; not building the resident handoff or the first combined owner
itself. Scene 120 now also surfaces the combined owner's resident planar
chassis state, vertical state, spin finiteness, and suspension finiteness in
the right rail to make live owner validation visible instead of only trusting
the GPU dispatch sequence.
`TireVehicleYawGpuKernel` / `tire_vehicle_yaw.wgsl` now adds the first
vehicle-level GPU chassis-yaw lane: it sums per-wheel
`off_x * Fz - off_z * Fx + tire_Mz` and integrates yaw/yaw-rate on GPU with
CPU/GPU parity. The same kernel now also exposes a first planar chassis-state
lane (`integrate_planar` / `dispatch_planar_blocking`) that sums per-wheel
world X/Z tire forces plus `Mz` and integrates chassis X/Z position, X/Z
velocity, yaw, and yaw-rate on GPU.
`integrate_planar_from_brush_outputs` /
`dispatch_planar_from_brush_output_blocking` now consumes GPU-resident
`BrushWheelOutput` rows directly from
`TireBrushGpuKernel::dispatch_input_buffer_resident`, so world force and
aligning moment can drive the planar chassis lane without staging brush output
rows back through CPU. `TireVehiclePlanarResidentState` /
`integrate_planar_state_from_brush_outputs` now keeps chassis
position/velocity/yaw rows in ping-pong GPU buffers and advances them from
resident brush output. `TireVehicleResidentChassisBrushOwner` now combines the
resident brush owner with that resident planar chassis state: one API steps
resident wheel spin, resident suspension, GPU brush-row generation,
GPU-resident brush output, and planar chassis integration without staging brush
output rows through CPU. `build_brush_rows_from_spin_suspension_planar_state`
and `step_resident_chassis_brush_owner_from_planar_state` now close the first
resident X/Z/yaw feedback loop: brush-row generation reads planar
position/velocity/yaw directly from the resident chassis state instead of from
stale caller-supplied pose rows.
`build_brush_rows_from_spin_suspension_chassis_state` and
`step_resident_chassis_brush_owner_from_resident_state` now feed both resident
planar and resident vertical chassis states into brush-row generation; the
combined owner steps resident spin, resident suspension-from-vertical, resident
brush output, resident planar integration, and resident vertical integration
without CPU-supplied chassis pose/velocity scalars. FEM assembly handoff plus
viewer/runtime adoption remain open; both the host-side brush-output-to-FEM
hub-parameter bridge and the GPU-resident brush-output-to-atlas-hub-row writer
now exist, and the reusable `step_resident_vehicle_fem_hubs_from_brush_owner`
sequence connects the resident chassis/brush owner to resident FEM hub
integration without CPU brush staging. `step_resident_vehicle_fem_atlas_from_brush_owner`
also schedules the resident FEM node dispatch around that handoff, so the next
step is moving the viewer/live examples onto this reusable substep sequence.
It also exposes a first GPU wheel-spin
lane (`integrate_wheel_spin` / `dispatch_wheel_spin_blocking`) that mirrors the
CPU wheel equation `drive_torque - brake_torque*tanh(spin) - tire_fx*radius`
for each wheel and reports next spin, spin acceleration, net torque, and
finite status. A companion GPU wheel-contact kinematics lane
(`compute_wheel_kinematics` / `dispatch_wheel_kinematics_blocking`) now mirrors
the CPU `Vehicle::step` contact-frame setup: rotated corner offsets, yaw-rate
corner velocity, `yaw + steer` wheel basis, longitudinal/lateral contact
velocity, slip ratio, and slip angle are produced on GPU for each wheel.
`compute_wheel_kinematics_from_spin_state` /
`dispatch_wheel_kinematics_from_spin_resident_blocking` now adds the matching
resident path, consuming `TireVehicleWheelSpinResidentBatch` directly so slip
ratio is computed from stepped GPU spin without staging spin rows through CPU.
The same kernel now adds a GPU suspension/normal-load lane (`compute_suspension` /
`dispatch_suspension_blocking`) that mirrors the CPU spring-damper travel,
travel-rate, and normal-load calculation from chassis height, wheel radius,
terrain height, rest length, stiffness, damping, and bump-stop compression.
`TireVehicleSuspensionResidentBatch` now keeps suspension geometry, parameter,
and output rows resident on GPU, and
`dispatch_brush_rows_from_spin_suspension_resident` consumes those resident
travel rows together with resident wheel-spin rows when building GPU brush
inputs. `TireVehicleVerticalResidentState` /
`integrate_vertical_state_from_suspension` now keeps chassis `y/vy` resident as
well, summing resident suspension normal loads against gravity and damping, and
`compute_suspension_from_vertical_state` lets resident suspension rows read that
vertical state without CPU staging. That closes the first resident vertical
chassis ping-pong path, and the combined chassis+brush owner now consumes it in
the full resident-state stepping path. The no-readback
spin+suspension-to-brush handoff is already in place;
`TireVehicleResidentBrushOwner` now wraps the spin batch, suspension batch, and
brush-row template into the first small resident vehicle owner, sequencing spin
step, suspension step, and brush-row generation into one API. The follow-on
`TireVehicleResidentChassisBrushOwner` also sequences resident brush output into
the resident planar chassis state; its planar-state variant feeds resident
chassis X/Z/yaw state back into brush-row generation, and its full resident
variant feeds both planar and vertical chassis state back into brush-row
generation. The first host-side FEM hub assembly bridge from brush outputs is
available, and `tire_fem_atlas.wgsl::hub_params_from_brush_outputs` now covers
the fully resident atlas hub-parameter update from resident brush output
buffers. `tire_fem_atlas.wgsl::hub_params_from_feedback_force_rows` now does
the same for GPU-resident FEM contact-feedback node-force rows.
`step_resident_atlas_hubs_with_resident_params` prevents the hub step from
erasing GPU-assembled rows, and
`step_resident_vehicle_fem_hubs_from_brush_owner` provides the first reusable
resident runtime sequence. `RealBrushTireModel` now bridges the reusable
pressure/contact brush tire into the legacy `TireModel` trait, and
`Scene::TireModelLab` plots `RealBrushTireModel::passenger_radial` beside the
diagnostic empirical reference curves while `Scene::VehicleOffroad` uses
`Vehicle::car_with(RealBrushTireModel::offroad_soft)` instead of the older
simple tire, so more viewer/runtime paths now get native brush tire forces
while full resident GPU vehicle/FEM adoption remains open. The GPU parity
checklist explicitly marks this legacy `TireModel` bridge as
`legacy_vehicle_real_brush_adapter` / `cpu_only_debug`: it has proof and a
follow-up to replace the CPU bridge with resident vehicle-to-brush/FEM owners
when those scenes own full GPU chassis/wheel/suspension state. The same bridge
is also listed in the known-approximation ledger, because it is not only a GPU
parity gap: it adapts the native brush law through a legacy API that lacks full
wheel pose, terrain maps, resident GPU buffers, FEM carcass state, suspension
rows, and terrain feedback.
`TireVehicleWheelSpinResidentBatch` now adds the first resident wheel-spin
owner: spin/radius/inertia rows ping-pong on GPU through
`integrate_wheel_spin_state`, commands can be updated in-place, and an ignored
hardware lane proves two consecutive resident steps against the CPU reference.
`build_brush_rows_from_spin_state` /
`dispatch_brush_rows_from_spin_resident` now consumes that resident wheel-spin
state directly when constructing `BrushWheelInput` rows, so the GPU brush
kernel can see stepped spin without staging spin rows back through CPU.
The first direct vehicle-to-tire handoff also exists now:
`build_brush_rows` / `dispatch_brush_rows_blocking` emits
`BrushWheelInput`-compatible seven-vec4 rows from chassis pose/velocity,
wheel mounts, steering/spin, suspension travel, tire parameters, material
rows, and terrain height. `dispatch_brush_rows_resident` plus
`dispatch_brush_rows_from_spin_resident` plus
`dispatch_brush_rows_from_spin_suspension_resident` plus
`TireBrushGpuKernel::dispatch_input_buffer_blocking` /
`dispatch_input_buffer_resident` now run generated buffers directly through the
GPU brush kernel without staging/readback in the handoff tests, and the resident
brush output can feed the planar chassis lane directly; the combined resident
chassis+brush owner now exercises that path as a single reusable API. Scene 121
now uses the direct resident vehicle-to-brush handoff as its primary GPU force
path, with direct GPU brush and CPU paths retained only as fallback/parity
lanes and `readback_resident_output` used as the viewer/CPU-chassis telemetry
seam. Broader resident chassis-state-driven runtime use and equivalent direct
FEM tire assembly coupling remain later vehicle-level work.
The CPU
`Vehicle` chassis integrator also consumes tire
aligning moments directly in its yaw equation, with a regression test proving
pure tire `Mz` changes chassis yaw.
`TireTriangleSeamManifoldReport` now adds a reusable seam/manifold sanity gate
for the finite-triangle CPU path: a tread sample straddling two plates must
produce multi-triangle contacts, preserve per-triangle material span, and let
FEM tread projection lift the node before point-only penetration collapses the
seam to a single triangle. The standard non-calibration runner emits this as
`triangle_seam_manifold`, and the viewer truth-label panel surfaces the same
report beside the broader finite-triangle stress/force/large-manifold gates so
the seam projection and material-span sanity checks are not hidden in tests.
`TireTriangleSurfaceStressReport` extends that gate beyond the first seam-only
sample: the `triangle_surface_stress` runner scenario samples low-friction and
high-friction plate interiors plus the seam, then checks low/high material
lookup, seam multi-patch manifold construction, material span across the seam,
finite outputs, and positive tread-radius penetration for every sampled contact.
`TireTriangleSurfaceForceReport` now extends the same two-plate fixture into
the brush-force path as `triangle_surface_force`: low/high/seam samples must
all build distributed contact, exercise sliding/bristle diagnostics, preserve
seam material span, and produce stronger tractive force on the high-friction
plate while keeping the seam response bounded between the two interiors.
`TireTriangleLargeManifoldReport` now adds the larger static-triangle side as
`triangle_large_manifold`: a highly subdivided two-material road must build a
nontrivial BVH, narrow the seam query to local candidates, preserve
multi-triangle material span, and keep FEM tread-node brush/feedback force
active on that large mesh.
The standard non-calibration runner now also attaches explicit
`fem_solver_budget` acceptance gates for the FEM contact-coupling seam:
contact cells must be sourced from actual tread nodes, retain a multi-cell
distributed patch, push those source tread-node velocities forward under drive
slip, and stay inside the shared material friction budget.
`TetraFemDeformationForceReport` now adds the standard-runner
`fem_deformation_force_shape` gate from the same contact-coupling section:
after high- and low-pressure tetra tires settle under the same load, the same
drive+cornering command must produce a longer low-pressure patch and a
measurably different force while keeping normal loads comparable, proving the
force path depends on deformed FEM shape instead of only an ideal wheel state.
`TetraFemContactSlipPatchReport` now covers the remaining contact-cell
diagnostic tests in the default runner as `fem_contact_slip_patch`: low
pressure must grow the actual tread-node patch and keep widthwise support,
while high commanded slip must increase sliding contact-cell count, sliding
fraction, and dissipated brush power without exceeding the shared friction
budget. The shared tire truth-label panel now surfaces these FEM/terrain
reports under `FEM/terrain contact gates`, including semi-implicit stiffness,
FEM deformation-force coupling, tread-node slip/patch diagnostics, finite-
triangle seam stress/force, and large-triangle manifold evidence so realistic
tire scenes cannot silently fall back to flat-floor or single-point contact
assumptions.

Model goals:

- Build contact cells from actual deformed tread nodes or tread surface
  triangles, not only from ideal circular samples.
- For each active tread node/triangle:
  - get actual contact normal/penetration;
  - compute local contact area;
  - compute local slip velocity from hub spin + node velocity + terrain
    velocity;
  - evolve a local bristle state; **done for CPU tread nodes in
    `apply_tread_contact_wrench`; GPU bristle buffer path with heightfield /
    friction-map / water-film / moving-surface velocity / built-in SDF /
    deformable sinkage-map sampling done with `TireFemContactFeedbackKernel`;
    GPU Bekker height/sinkage writeback done as `TireBekkerGpuKernel`, and the
    reusable `TireBekkerFemTerrainOwner` bridge now exports those maps as
    `TireFemContactTerrain`; first live scene-122 owner hookup is done, scene
    122 now feeds GPU transient-relaxed slip into GPU FEM feedback and applies
    returned node velocities/bristles back to the bench tire, and the first
    GPU-resident no-readback FEM feedback dispatch now returns velocity/
    bristle/force buffers for downstream GPU stages; full multi-body scene
    ownership remains open.**
  - cap force by material friction/cohesion.
- Feed the resulting contact forces back into the FEM nodes, not just into
  chassis readouts. **Initial CPU path done with
  `TetraTireFem::apply_tread_contact_wrench`; GPU node velocity feedback with
  heightfield/friction-map/water-film/moving-surface/SDF sampling done with
  `TireFemContactFeedbackKernel`.**
- Accumulate wheel-level outputs from node/triangle forces:
  - `Fx`, `Fy`, `Fz`, `Mx`, `My`, `Mz`;
  - contact patch bounds;
  - contact centre;
  - sliding fraction;
  - dissipated power.
- Add GPU path:
  - persistent per-node/per-contact bristle buffers;
  - contact reduction kernel for patch diagnostics and wheel wrench; **first
    heightfield/material/SDF diagnostics plus local-normal world-wrench
    reduction done with `TireFemContactReductionKernel`; GPU force/bristle
    feedback with heightfield normal/friction-map/water-film/moving-surface/SDF
    sampling plus deformable sinkage-map rolling-loss sampling done with
    `TireFemContactFeedbackKernel`; GPU Bekker height/sinkage writeback done
    with `TireBekkerGpuKernel`**;
  - heightfield/material sampling on GPU. **done for FEM contact diagnostics;
    force/bristle feedback has the same heightfield normal/friction-map path,
    and both FEM diagnostics and feedback now sample water-film maps;
    built-in sphere/box SDF sampling is done for FEM diagnostics and feedback;
    deformable-terrain sinkage-map sampling is done for FEM diagnostics and
    feedback rolling loss, and GPU Bekker height/sinkage writeback is done.**
- Add tests:
  - deformed low-pressure tire produces wider actual tread-node patch;
  - FEM-derived `Fz` agrees with hub load within tolerance;
  - sliding node count grows with slip;
  - tire force changes when the same ideal wheel state has a different FEM
    deformation shape.
  - reusable telemetry stays finite, unit-labelled, and consistent with the
    contact force/friction/energy accounting. **Initial
    `RealTireTelemetrySample` / patch / cell telemetry tests done.**
- Add tire-envelope contact for small obstacles:
  - tread should wrap/bridge over bumps based on carcass/belt stiffness;
  - contact should not collapse to a single sample point on curbs;
  - low pressure should visibly increase obstacle conformity.
    **Initial reusable envelope-contact gates are now part of the standard
    non-calibration runner: `curb_envelope_contact` checks non-vertical curb
    normals, multi-cell contact, extended patch length, horizontal force from
    the curb normal, and low-pressure-vs-high-pressure patch conformity; the
    ordered `rolling_obstacle_sequence` still covers approach/crest/exit FEM
    wrapping over a bump.**

### E. Surface-to-surface terrain contact

Current tire contact is mostly flat-floor / heightfield-oriented. Realistic
offroad and obstacle behavior needs richer terrain contact.

**Initial implementation status:** partially done for heightfield/material
lookup and CPU FEM feedback. The `Terrain` trait now exposes `friction_at`,
`surface_velocity`, and `signed_distance_and_normal`. `HeightfieldTerrain`
supports friction/water-film maps, tire brush / FEM-derived tread contact
samples local terrain friction and moving-ground velocity per contact cell,
`TriangleMeshTerrain` provides static triangle terrain with per-triangle
friction plus closest finite-triangle signed-distance normals for CPU FEM node
projection near mesh edges/plates, `SdfTerrain` provides native sphere/box SDF
obstacle queries, `TriangleMeshTerrain` now uses a CPU triangle BVH over
per-triangle AABBs for height, closest-surface signed-distance, and
multi-triangle contact-manifold queries; CPU `TetraTireFem` tread projection
uses the manifold contact radius for triangle seams/plates,
and `TetraTireFem::step_on_terrain` now projects nodes with
true SDF/triangle signed distance and closest normal when a terrain backend
provides it instead of always using vertical height projection. CPU brush/FEM-
derived contacts report per-cell local normals and accumulate the wheel force
from those normals, `GranularBed::apply_tire_contact` now injects real tire
contact-cell normal/shear impulses into DEM grains on CPU, and the GPU brush
kernel samples heightfield normals/friction maps per contact cell and returns
world-force output. GPU FEM diagnostics/feedback now sample deformable
sinkage/rut maps for mean rut depth and Bekker-style rolling loss, and
`TireBekkerFemTerrainOwner` now owns Bekker height/sinkage writeback maps and
exports them as FEM contact terrain; its resident GPU terrain output can feed
FEM feedback directly while preserving an explicit CPU mirror seam for viewer
telemetry. `Scene::TireRealistic` now uses this owner for live rut writeback,
resident FEM feedback, and viewer terrain/rut diagnostics. `MpmSimulation`
now has a first in-place tire-feedback composition bridge that applies GPU
tire feedback directly on live MPM particle/velocity/compaction buffers before
the full particle/grid solver continues, with optional compacted FEM terrain
export for diagnostics/viewer coupling. `TireMpmGpuKernel` can now rasterize
the live MPM compaction/material-velocity field into GPU height/sinkage/surface-
velocity terrain buffers without CPU rasterization, and
`TireFemContactFeedbackKernel` can consume those buffers directly for
tread-node feedback; `TireFemContactReductionKernel` consumes the same
height/sinkage buffers for diagnostics. `Scene::TireRealistic` now owns a live
MPM offroad bridge that applies tire contact to persistent MPM particles,
rasterizes GPU terrain buffers, uses those resident buffers in the resident
no-readback FEM feedback path when available, and exports FEM contact terrain
for the next frame; the only host readback left in that viewer path is the
explicit mirror seam needed to update CPU-side gizmo/thermal telemetry. The
sparse-LBVH DEM path now has a larger 14×12×5 CPU/reference stress test for
its packed GPU row layout plus larger ignored GPU hardware dispatch and
no-readback resident-builder stress lanes. `TireTriangleLargeManifoldReport` now also includes a reusable
force/bristle reduction over the large triangle-mesh FEM feedback cells:
active force-cell count, bristle force sum/mean/max, and widthwise support span
are gated beside BVH narrowing, seam manifold/material span, distributed FEM
contact, feedback force, and sliding fraction, and the shared tire truth-label
panel renders those values under `triangle_large_manifold`. Open work remains:
GPU-side large-triangle force/bristle reduction parity beyond the CPU report,
higher-grain sparse-LBVH scaling as granular scenes grow, and full multi-scene
offroad ownership. GPU finite-triangle sampling now has a cheap horizontal
broadphase sphere plus a packed BVH atlas/traversal path, but not a full
surface-to-surface contact manifold.

- Add terrain/material lookup per contact cell:
  - height, **done for CPU brush/FEM tread contact, GPU brush, and GPU FEM
    contact diagnostics**;
  - normal, **done for CPU brush/FEM tread contact and GPU brush world-force
    output; done for GPU FEM contact diagnostics world-wrench output and GPU
    FEM force/bristle feedback**;
  - material, **done for CPU brush/FEM tread contact, GPU brush friction maps,
    GPU FEM contact diagnostics mean-friction sampling, and GPU FEM
    force/bristle feedback friction-map sampling**;
  - local surface velocity, **done for CPU brush/FEM tread contact, done as a
    uniform moving-surface velocity in GPU FEM force/bristle feedback, and
    now done for GPU FEM force/bristle feedback as an optional row-major
    `surface_velocity_map` sampled per tread node; MPM terrain export now
    writes CPU and GPU-resident spatial material-velocity maps, and the Bekker
    FEM terrain owner now also accepts/validates/exports an optional
    `surface_velocity_map` through both CPU `TireFemContactTerrain` and
    GPU-resident `TireBekkerGpuResidentTerrain`; DEM sparse-bed output now has
    `tire_dem_raster_surface_velocity_map`, which exports top-layer grain
    velocity rows, and `tire_fem_contact_terrain_with_dem_surface_velocity`
    now validates grid origin/spacing/dimensions and attaches those rows
    directly to `TireFemContactTerrain` for FEM contact feedback.
    `TireDemFemTerrainOwner` now wraps DEM grain state as a reusable
    height/sinkage/surface-velocity terrain owner, and Scene 119 uses that
    owner in the sparse-LBVH stepper to show DEM→FEM terrain rows and max
    sinkage in the SoilCompaction right rail; Scene 119 can now also use the
    reusable resident GPU DEM terrain owner so the visible compaction bed steps
    through resident DEM state and resident height/sinkage/surface-velocity
    terrain rows when a GPU device is available. Scene 122 now also has an
    explicit live DEM→FEM terrain bridge toggle: tire contact cells step a
    `TireDemFemTerrainOwner` / resident-owner sparse-LBVH DEM grain bed, the
    owner keeps positions/velocities as source-of-truth, exports
    height/sinkage and surface-velocity rows, and GPU FEM contact feedback can
    consume those rows through the normal `TireFemContactTerrain` or GPU
    terrain path. The CPU owner also exposes a GPU blocking sparse-LBVH step
    wrapper for parity/hardware lanes. An ignored owner-level hardware parity
    lane now checks that `TireDemFemTerrainOwner::step_sparse_lbvh_gpu_blocking`
    updates owned grain state, height/sinkage rows, and surface-velocity rows
    consistently with the CPU owner step. `TireDemGpuKernel::dispatch_lbvh_step_resident`
    plus `readback_resident_output` now provides the first no-readback DEM
    sparse-LBVH output seam, so downstream GPU terrain/FEM stages can consume
    DEM grain buffers without forcing an immediate CPU collapse.
    `TireDemResidentState` plus `dispatch_lbvh_step_resident_state` now keeps
    positions/velocities on GPU across consecutive sparse-LBVH steps, with an
    ignored ping-pong hardware lane proving two resident steps against the CPU
    reference. `TireDemResidentLbvh` plus
    `TireDemGpuKernel::upload_resident_lbvh` and
    `dispatch_lbvh_step_resident_state_with_resident_lbvh` now split the
    CPU-built sparse-LBVH topology upload from the DEM step dispatch, so
    packed node/index rows live in explicit GPU buffers instead of being hidden
    inside the step call. `TireDemGpuKernel::dispatch_resident_terrain_raster` and
    `TireDemResidentTerrainOutput` now add the matching resident DEM terrain
    export seam: height, sinkage, and top-layer surface-velocity rows are
    rasterized from resident grain buffers on GPU, with
    `readback_resident_terrain` kept as the explicit mirror/debug seam and an
    ignored hardware parity lane checking the output against the CPU DEM
    raster helpers without reading back grains first.
    `TireDemResidentTerrainOutput::fem_contact_feedback_gpu_terrain` also
    exposes those resident height/sinkage/surface-velocity buffers through the
    normal FEM GPU-terrain contract so FEM contact feedback can consume DEM
    terrain without CPU terrain re-upload. Scene 122 now adopts that seam when
    GPU DEM is enabled: it steps `TireDemResidentState` through the sparse-LBVH
    kernel, rasterizes `TireDemResidentTerrainOutput`, keeps that terrain for
    the resident FEM contact-feedback path, and uses readback only as the
    explicit viewer/CPU mirror. `TireDemResidentFemTerrainOwner` now packages
    that pattern as a reusable owner with persistent DEM state, GPU-built
    resident sparse-LBVH topology buffers, height/sinkage/surface-velocity
    terrain output, FEM GPU-terrain handoff, cached Morton bounds, and an
    explicit CPU mirror only for viewer/readback telemetry. The same owner now
    has a reset/reseed bridge that rewrites same-count resident grain state
    buffers in place, refreshes cached bounds and resident terrain output, and
    invalidates stale topology before the next GPU builder pass; Scenes 119
    and 122 use this owner and reseed it instead of reconstructing or stepping
    stale GPU DEM state when CPU-visible grain rows have changed. Both scenes
    now surface the resident DEM reseed counter in the right rail so
    reset/fallback transitions and CPU-visible source-row changes are visible
    instead of silently hidden behind the GPU owner;
    instead of hand-assembling the resident buffers inline. Deeper work remains
    to adopt this direct DEM-terrain handoff across broader FEM/offroad
    owners and keep scaling sparse-bed stress cases**;
  - deformable-terrain state if available, **done for Bekker-style
    `DeformableTerrain::apply_tire_contact` on CPU and first-pass DEM
    grain impulse coupling via `GranularBed::apply_tire_contact` /
    `TireDemFeedback`; GPU FEM contact now samples deformable sinkage maps for
    rolling loss, and `TireBekkerGpuKernel` writes Bekker height/sinkage maps
    from tire contact cells; `TireDemGpuKernel` now adds first-pass GPU
    localized tire-cell impulse transfer into DEM grains plus dense,
    host-built cell-grid, unbounded GPU-radix resident grid, and compatibility
    old-API grid grain/floor/tire contact steps; `TireMpmGpuKernel` now adds localized
    CPU/GPU MPM material-point pressure/shear/compaction feedback,
    `TireMpmFemTerrainOwner` exports the compacted material-point state as FEM
    sinkage/height terrain, and `MpmSimulation` has a first in-place GPU
    tire-feedback composition bridge back into the full particle/grid MPM
    solver plus a GPU compaction-to-terrain raster path; GPU FEM contact
    feedback and reduction diagnostics can consume those terrain buffers
    directly; DEM now has a Morton-sorted sparse-LBVH CPU oracle plus GPU
    sparse-LBVH traversal path and `Scene::SoilCompaction` can use the sparse-
    LBVH owner to step/write back its visible grain bed and export a top-layer
    DEM `surface_velocity_map` through `TireDemFemTerrainOwner`, including
    height/sinkage rows and the strict
    `tire_fem_contact_terrain_with_dem_surface_velocity` bridge; Scene 122 can
    now route a live owner-stepped sparse-LBVH DEM bed into FEM contact feedback
    using the same owner; a larger
    14×12×5 CPU/reference sparse-bed stress test now covers packed row/layout
    sanity plus dispatch-plan sizing for grain/node/row/workgroup/storage
    counts, an ignored large sparse-LBVH GPU dispatch stress lane now checks the same
    packed traversal on real hardware, and an ignored large no-readback
    resident-builder lane checks the padded topology builder at the same
    scale; full multi-scene offroad ownership and
    higher-grain scaling remain open**.
- Support heightfields first, then triangle/SDF terrain. **Heightfields,
  static triangle meshes, and simple native sphere/box SDF terrains are done
  for CPU terrain queries; CPU FEM node projection now uses
  `Terrain::signed_distance_and_normal` so SDF obstacles and finite triangle
  mesh plates project along their closest surface normal instead of only
  vertical top-surface height. GPU FEM diagnostics and force/bristle feedback
  now sample built-in sphere/box SDF primitives and finite triangle patches
  with closest-distance normals and primitive/triangle friction; CPU
  triangle-mesh queries now have a reusable triangle BVH over precomputed
  AABBs plus a multi-triangle contact-manifold query consumed by CPU
  `TetraTireFem` tread projection, and GPU finite-triangle rows now carry
  per-triangle XZ broadphase spheres plus a packed BVH atlas consumed by the
  contact shaders; GPU FEM reduction/feedback now performs weighted
  multi-triangle manifold accumulation for seam contacts, and a large packed-
  atlas row-layout stress test covers CPU-side export sanity;
  `TireFemContactFeedbackPatchAccounting` now exposes total force,
  per-row force magnitudes, and bristle deflection summaries for the feedback
  patch; an ignored large triangle-BVH hardware dispatch stress lane now
  exercises the GPU traversal on real hardware.**
- For deformable soil:
  - couple tire contact pressure to Bekker/DEM/MPM soil state; **done for
    Bekker heightfield soil, GPU sinkage-map readback into FEM rolling loss,
    GPU Bekker height/sinkage writeback, initial CPU DEM `GranularBed` grain
    impulses, first GPU localized tire-cell-to-grain impulse transfer, dense
    GPU DEM stepping, a host-built cell-grid GPU DEM step, an unbounded
    sorted-grid CPU oracle plus GPU cell-key / GPU-radix / GPU range-scan
    resident DEM stepping that keeps every grain in dense cells, a
    compatibility wrapper for the old GPU-built fixed-capacity DEM API, and first CPU/GPU MPM material-point tire pressure/shear/compaction feedback
    with FEM terrain export plus a first in-place `MpmSimulation` bridge back
    into the full particle/grid MPM solver plus GPU MPM compaction-to-terrain
    raster buffers and direct FEM contact-feedback consumption of those
    buffers plus FEM reduction diagnostics on the same resident buffers, and a
    Morton-sorted sparse-LBVH CPU oracle plus GPU sparse-LBVH traversal path
    for sparse DEM beds; `Scene::SoilCompaction` can use the sparse-LBVH owner
    to step/write back its visible roller/grain scene and expose DEM top-layer
    surface-velocity map telemetry, and a larger 14×12×5 CPU/reference sparse-
    bed stress test covers packed row/layout sanity plus dispatch-plan sizing
    for grain/node/row/workgroup/storage counts; an
    ignored large sparse-LBVH GPU dispatch stress lane now runs the packed
    traversal on real hardware, and an ignored large no-readback resident-
    builder lane runs the same sparse bed through GPU-built resident topology,
    while full multi-scene offroad ownership and
    higher-grain scaling remain**;
  - soil returns both normal sinkage resistance and shear traction; **done as
    `TireSoilFeedback` normal/shear sinkage plus rolling resistance for Bekker
    and `TireDemFeedback` normal/shear load, impulse, grain sinkage, and
    rolling-loss proxy for DEM**;
  - tire tread can excavate / compact / build a rut. **done for Bekker
    heightfield soil from tire contact cells; initial DEM grain push/shear
    tests show driven/slipping tires transfer more grain motion than rolling
    tires**.
- Viewer targets:
  - tire rolling over a bump/curb;
  - low-pressure tire wrapping around an obstacle;
  - tire on ice patch then asphalt;
  - tire compacting soft soil/rut.
  **Initial reusable `TireViewerTargetCoverageReport` is now emitted by the
  standard non-calibration runner and text report. It maps those viewer targets
  to deterministic rig evidence: ordered rolling-obstacle approach/crest/exit
  plus curb support, low-pressure bump crest patch/contact evidence,
  surface-transition ice/asphalt evidence, and soft-soil off-road plus
  road-condition/rolling-loss evidence. The coverage report now also maps the
  plain-language scene checklist items to deterministic evidence for soft tire
  patch growth, over-inflated tire slip under the same torque/load, ice still
  sliding with a large low-pressure patch, and thermal/wear state visibility.
  The shared tire-rig replay panel now renders the same coverage rows with
  pass/fail, expectation text, and evidence strings so the viewer target
  checklist is visible in the UI, not only in the exported text artifact.**
- Add wet/film surface state after the dry/ice/soil path:
  - thin water film lowers shear strength; **done in `SurfaceFilmState`**;
  - speed-dependent film effects can reduce contact shear further; **done in
    `SurfaceFilmState::apply_to`**;
  - keep this separate from calibration-heavy aquaplaning until the basic
    heightfield/material contact is stable. **done: it is a deterministic
    material modifier, not a fitted aquaplaning model**;
  - spatially varying wet patches on heightfields; **done for CPU brush
    contact, FEM-derived tread contact, and FEM tread-wrench feedback by
    sampling `Terrain::water_film_depth` per contact cell and applying
    `SurfaceFilmState` before the friction cap / bristle damping / rolling
    loss solve**;
  - GPU-side water-film-map sampling for FEM contact diagnostics and FEM
    bristle/force feedback; **done: `TireFemContactReductionKernel` now reports
    mean water-film depth, and `TireFemContactFeedbackKernel` samples
    heightfield film maps to reduce μ, bristle damping, and cohesion before
    applying the tread-node impulse**.

### F. Numerical robustness / solver depth

Explicit FEM is the main stability limitation.

- Add XPBD or semi-implicit constraints for:
  - volume preservation / near-incompressibility; **initial XPBD volume
    projection done on CPU and GPU with compliance knobs and tests**;
  - bead/rim coupling; **initial XPBD bead mount projection done on CPU and
    GPU with compliance/iteration knobs and tests; the default non-calibration
    runner now emits `cavity_bead_sanity` gates for pressure scaling, loaded
    cavity volume bounds, bead mount drift, and bead projection correction**;
  - sidewall bending; **initial XPBD opposite-vertex hinge-distance bending
    proxy done on CPU and GPU with compliance/iteration knobs and tests;
    initial true sidewall quad dihedral-angle projection is done on CPU and
    GPU with diagnostics/parity tests**;
  - tread belt inextensibility; **initial XPBD circumferential/axial/crossed
    tread-belt edge constraints done on CPU and GPU with tests; construction
    sanity now records `TetraTreadBeltStretchDiagnostics` with mean and peak
    relative belt-edge stretch so one local overstretched belt edge cannot hide
    inside an average**.
- Add Newton/semi-implicit FEM option for stiff rubber and high pressure.
  **Initial CPU+GPU diagonal semi-implicit stiffness option done via
  `TetraTireFemParams::semi_implicit_stiffness_scale`: it damps stiff-rubber
  velocity spikes by adding a per-node tangent-stiffness denominator to the
  velocity solve and records the setting in solver-budget telemetry. The GPU
  FEM node kernel mirrors the same denominator through `TireFemGpuParams`.
  `TetraFemSolverStiffnessReport` now adds a standard-runner
  `fem_solver_stiffness_stabilizer` gate proving that the semi-implicit guard
  reduces a stiff-rubber velocity spike versus explicit integration and that a
  high-pressure rolling case stays finite, radius-bounded, and supported. This
  is not a full Newton solve. `TetraFemDiagonalNewtonReport` now adds the first
  stronger implicit/Newton building block as `fem_diagonal_newton_solver`: it
  solves a nonlinear stiff rubber mode with backward Euler plus an explicit
  residual/Jacobian Newton iteration, then gates convergence, residual
  reduction, tangent positivity, explicit overshoot, and bounded implicit
  energy in the standard non-calibration runner and viewer truth panel. The
  next element-level step is also started: `TetraFemElementNewtonReport`
  emits `fem_element_newton_solver`, solving one real Neo-Hookean tetra element
  as a coupled 12-DoF backward-Euler Newton system with residual/Jacobian,
  off-diagonal coupling, pivot, volume, and energy gates surfaced in the
  standard runner and viewer. `TetraFemPatchNewtonReport` now adds the first
  connected multi-element CPU patch solve as `fem_patch_newton_solver`,
  accumulating forces through shared nodes and solving the whole patch with
  one residual/Jacobian Newton iteration path. `TetraFemTinyMeshNewtonReport`
  now takes that into the actual tire topology as `fem_tiny_mesh_newton_solver`,
  solving the smallest annular `TetraTireFem` mesh as one coupled
  residual/Jacobian system and surfacing the same mesh-level gates in the
  standard runner/viewer. `TetraFemMatrixFreeNewtonScaleReport` now adds the
  next scaling gate as `fem_matrix_free_newton_scale`, evaluating a larger
  tire mesh residual/Jv without building the dense Jacobian and reporting the
  memory ratio versus dense assembly. `TetraFemMatrixFreeNewtonSolveReport`
  now follows with `fem_matrix_free_newton_solve`, a CPU/debug GMRES-style
  matrix-free correction gate that uses Jv products plus line search to reduce
  the larger tire-mesh nonlinear residual without dense assembly.
  `TetraFemMatrixFreeNewtonPreconditionerReport` adds
  `fem_matrix_free_newton_preconditioner`, sampling the tangent diagonal with
  matrix-free Jv products, proving the Jacobi-style scaled Krylov solve is not
  worse than the unpreconditioned probe, and keeping the dense-vs-matrix-free
  memory accounting visible. `TetraFemMatrixFreeNewtonStepReport` then turns
  that correction into `fem_matrix_free_newton_step`, a CPU/debug implicit-step
  readiness gate that reconstructs bounded positions/velocities and checks
  volume/energy envelopes after line search. The callable
  `TetraTireFem::step_on_terrain_matrix_free_newton_debug` path plus
  `fem_matrix_free_newton_timestep` runner gate now prove the same correction
  can mutate a tire state behind an explicit debug API, sharing the
  `matrix_free_newton_correction` helper with the report-only path so the
  Jv probe, diagonal preconditioner, GMRES correction and line search cannot
  drift between the two sites. The GPU parity row is **CpuAndGpu** end-to-end:
  `molla_kernels::tire_fem_matrix_free_newton_gpu_contract` pins the dispatch
  shape via `TireFemMatrixFreeNewtonGpuContract`; the WGSL kernel
  `tire_fem_matrix_free_newton.wgsl` provides `residual_eval`, `jv_probe`,
  `diagonal_extract`, and `apply_correction` entry points;
  `TireFemMatrixFreeNewtonGpuKernel::dispatch_blocking` compiles the shader and
  runs all four entry points end-to-end with readback;
  `TireFemMatrixFreeNewtonGpuKernel::run_newton_outer_blocking` drives a full
  host-orchestrated GMRES + Jacobi + line-search Newton outer iteration on top
  of those GPU primitives; and two ignored hardware lanes
  (`…_residual_matches_cpu_oracle` and `…_outer_iteration_reduces_residual`)
  prove parity against a closed-form CPU oracle and end-to-end residual
  reduction. Production-promotion seams are also in place:
  `TetraTireFemParams::solver_mode = TetraTireFemSolverMode::MatrixFreeNewtonDebug`
  opts a tire into the matrix-free implicit timestep through
  `step_on_terrain`, the new `TetraTireFemParams::high_pressure_implicit()`
  preset ships with that mode pre-selected, and a scene-coverage test
  validates the timestep on realtime_viewer / offroad_soft /
  high_pressure_reference presets. The remaining open work is rewiring the
  GPU persistent FEM batch substep so high_pressure_implicit tires call
  `run_newton_outer_blocking` directly each substep, validating that path
  through viewer scenes 109/120/121/122 and the vehicle scene against the
  explicit envelopes, and only then flipping the realtime_viewer /
  offroad_soft presets to MatrixFreeNewtonDebug by default. The GPU-first
  checklist marks `tetra_fem_newton_solver_debug` as `cpu_and_gpu`, and the
  approximation ledger lists `newton_solver_cpu_debug_only` with that
  remaining persistent-batch wiring as the explicit follow-up.**
- Keep GPU-first:
  - if a CPU solver is added, it must have a GPU plan or be clearly marked
    debug-only.
- Add hardening tests:
  - high load + low pressure does not explode; **done**;
  - high pressure + high speed does not expand unbounded; **done**;
  - tire can run thousands of substeps without NaNs; **done**;
  - energy/damping stays bounded. **done**.

## Implementation stages

### Stage A — reusable real-tire module

- Add `crates/molla-vehicle/src/real_tire.rs`.
- Export it from `molla-vehicle`.
- Implement:
  - `SurfaceMaterial`;
  - `RealTireParams`;
  - `RealWheelState`;
  - `RealTireContact`;
  - `solve_loaded_tire_hub_height`;
  - `evaluate_brush_tire`.
- Add the first crate tests.

### Stage B — use it in `Scene::TireTraction`

- Replace local `patch_factor()` force calculation.
- Store wheel angular velocity per car.
- Let tire force and slip come from `evaluate_brush_tire`.
- Keep the right-rail pressure/weight sliders.

### Stage C — improve tire shape demo

- Replace the old shape demo with `TetraTireFem`.
- Draw the actual tetra FEM surface wireframe and floor-contact tread nodes.
- Keep pressure bars and fixed/animated pressure comparison.

### Stage C2 — nonlinear tire FEM pass

- Replace the previous small-strain linear tetra model with compressible
  Neo-Hookean/hyperelastic tetra elements. **Done.**
- Add first-pass scalar material regions for sidewall, tread/belt, bead, and
  carcass. **Done.**
- Export per-element Lamé parameters to GPU and use them in WGSL. **Done.**
- Move from explicit time integration to implicit, semi-implicit Newton, or
  XPBD FEM constraints for larger stable time steps. **Partially done for
  damping/contact plus CPU XPBD volume, bead mount, tread-belt structural
  constraints, sidewall bend-proxy constraints, and an initial true
  sidewall dihedral-angle projection; GPU XPBD parity is done for volume,
  bead mount, structural edge-distance constraints, and the sidewall
  dihedral-angle projection. A first CPU+GPU diagonal
  semi-implicit stiffness denominator is done for stiff/high-pressure presets,
  but FEM elastic forces are still not a true Newton solve and full laminated
  sidewall modeling remains open.**
- Replace pressure-as-stiffening with a true internal cavity gas law:
  pressure = f(volume, gas amount), forces from the gradient of enclosed
  volume, and optional mounting/inflation initialization. **Initial gas-law
  cavity path done; the previous hidden carcass-leak scalar is now a named
  `cavity_carcass_coupling` preset parameter covered by governance and
  cavity/bead acceptance gates. `TetraBeadSeatDiagnostics` now decomposes bead
  mount drift into radial, axial, and tangential rim-seat errors and adds those
  bounds to the cavity/bead sanity report. `TetraBeadSeatLoadCaseReport` and
  the standard runner's `bead_seat_load_cases` snapshot now exercise the bead
  guardrail under static inflation, vertical load, lateral-slip load,
  drive/brake longitudinal force, and high-pressure load, gating mount drift,
  radial/axial/tangential seat drift, and proof that lateral and longitudinal
  tire forces were actually applied. A full validated rim/bead contact solve
  and stronger structural modeling remain open. The shared tire truth-label
  panel now surfaces `cavity_bead_sanity`, `cavity_coupling_effect`, and
  `bead_seat_load_cases` under `Cavity/bead/sidewall gates`, including pressure
  scaling, cavity volume, bead load, radial/axial/tangential rim-seat drift,
  coupling-sweep load response, and load-case mount errors.**
- Add anisotropic cord/belt material directions rather than scalar stiffness
  regions. **Done for a compact four-family explicit ply stack on CPU and GPU:
  crossed ± belt/sidewall plies plus cap/hoop and radial/axial helper plies
  exported through `TetraTireFemGpuSnapshot`, `TireFemGpuInput`, and
  `tire_fem.wgsl`; reusable `TetraSidewallDihedralDiagnostics` now measures
  sidewall fold-angle drift for validation, `TetraSidewallLaminateAudit`
  verifies the crossed ± sidewall bias plus radial/axial helper ply layout,
  `TetraSidewallLaminateStretchDiagnostics` measures tensile and compressive
  sidewall ply strain from the same `F·a0` stretch used by the fibre stress,
  `TetraSidewallLaminateSanityReport` rejects tires whose ply layout is right
  but whose sidewall is currently folded or stretched beyond the realtime
  envelope, `TetraSidewallDynamicLoadReport` now exercises that sanity gate
  under deterministic vertical, lateral, and steered slip-angle tire loads in
  the standard runner's `sidewall_dynamic_load_cases` snapshot; the steered
  load gate requires lateral force and aligning moment from the wheel-frame
  angle while keeping fold drift and ply strain bounded.
  `TetraSidewallLaminateCouplingReport` now adds the
  `sidewall_laminate_coupling` runner snapshot: radial bulge balances the
  crossed ± bias-ply energy, widthwise stretch activates the axial helper,
  radial bulge activates the radial helper, and a folded fixture must produce
  positive ply energy while the structural projection reduces the measured
  fold. The GPU parity matrix now accounts for this as
  `tetra_fem_sidewall_laminate_coupling_report` with explicit CPU-only/debug
  reason and follow-up rather than silently treating the diagnostic as a GPU
  production path.
  `TetraSidewallPlyStrainXpbdReport` now adds `sidewall_ply_strain_xpbd`:
  a deliberately over-stretched sidewall must have its maximum tensile ply
  strain reduced and bounded by the XPBD ply-strain projection. The same
  guardrail is now present in the single-tire and fused-atlas GPU node
  shaders through `TireFemGpuParams::xpbd_sidewall_ply_stretch_limit` and
  `xpbd_sidewall_ply_compliance`. This is the first force/constraint-producing
  ply-stretch guardrail, but the full laminated shell/carcass constraint is
  still future work.
  `TetraSidewallDihedralXpbdEffectReport` now makes the initial CPU/GPU XPBD
  dihedral-angle projection a standard-runner gate as
  `sidewall_dihedral_xpbd_effect`: a deliberately folded sidewall must report
  lower max fold-angle drift after the structural projection. A true laminated
  shell sidewall solve remains future structural work. The shared tire
  truth-label panel now surfaces `sidewall_dynamic_load_cases` and
  `sidewall_laminate_coupling` and `sidewall_dihedral_xpbd_effect` under
  `Cavity/bead/sidewall gates`, including fold drift, ply strain,
  steered-force/aligning-moment evidence, ply-energy coupling evidence, and
  measured XPBD fold-drift reduction so the current sidewall approximation is
  visible in the viewer.**
- Couple the FEM tread nodes directly to the brush/terrain material contact
  model instead of the current flat-floor viewer contact. **Partially done:
  tread-node brush/Coulomb contact, terrain height/friction/surface-velocity
  sampling, local normals, SDF signed-distance projection, and reusable
  combined-slip brush diagnostics exist; CPU triangle-mesh finite-triangle
  signed-distance projection exists for FEM nodes near mesh edges/plates; CPU
  wheel-level brush wrench feedback into FEM nodes exists, and GPU contact
  diagnostics plus heightfield/material plus SDF/finite-triangle bristle
  feedback exist; GPU finite-triangle diagnostics/feedback now blends
  weighted multi-triangle manifold normals/friction; reusable feedback-output
  patch accounting now reports force-row and bristle-deflection summaries for
  seam diagnostics; an ignored large triangle-BVH GPU dispatch stress lane now
  covers the packed traversal, and remaining open work is deeper seam-specific
  visualization if fixtures still need more detail. `TireTriangleSeamManifoldReport`
  now provides the first reusable seam-specific sanity report for multi-patch
  seam contact, friction/material span, and FEM projection before point-only
  penetration.**

### Stage D — GPU path

The first GPU node-step path now exists in `molla-kernels`:

- `shaders/tire_fem.wgsl`
  - one invocation per tire FEM node;
  - each node uses node/tet CSR adjacency and accumulates only its own force
    contribution, avoiding float atomics/races;
  - applies compressible Neo-Hookean tetra stress, crossed-ply anisotropic
    fibre stress,
    bead/rim coupling, cavity gas-law bead load, optional legacy pressure
    traction, and zero-gap flat-floor contact;
  - writes next positions, velocities, and force diagnostics.
- `shaders/tire_fem_hub.wgsl`
  - reduces bead reactions and integrates the 1-DoF hub on GPU;
  - writes a GPU-resident hub state consumed by the next node-step dispatch.
- `shaders/tire_brush.wgsl`
  - evaluates batched combined-slip brush contact on flat terrain or an
    uploaded heightfield;
  - outputs longitudinal force, lateral force, normal load, patch length,
    sliding count, slip ratio, lateral slip speed, slip angle and aligning
    moment;
  - samples heightfield normals and emits `world_force` accumulated from
    per-contact normal/tangent bases;
  - can sample optional row-major heightfield friction maps per contact cell
    and scale material friction consistently with the CPU terrain API;
  - matches the CPU brush model in the GPU parity test.
- `TireBrushGpuBatch`
  - keeps reusable wheel/input, output, terrain, uniform and readback buffers
    for repeated brush-contact dispatches;
  - matches the blocking one-shot brush dispatch in the GPU parity test.
- `TireFemGpuKernel`
  - blocking wrapper for tests and small batches;
  - persistent-buffer path for viewer stepping without per-substep mesh
    re-upload;
  - consumes f32 raw buffers exported by `TetraTireFem::gpu_snapshot()`;
  - validated by the ignored hardware lane
    `make test-gpu P=molla-kernels T=tire_fem_gpu`.
- `TireFemGpuBatch`
  - owns several persistent FEM meshes as one reusable batch object;
  - advances per-tire FEM params and optional hub params through batch APIs;
  - reads back all tire outputs once per frame;
  - has an ignored GPU test for multiple resident tires;
- `TireFemGpuAtlasBatch`
  - owns a packed fused atlas for many tire FEM meshes;
  - advances all nodes through one atlas dispatch per substep;
  - advances all hub rows through one fused hub-reduction dispatch per substep;
  - supports packed/per-tire node-state reseeding plus packed/per-tire hub
    reseeding/readback;
  - is exercised by the ignored resident-atlas GPU hardware lane.
- Viewer wiring:
  - `Scene::TireDynamics` defaults to resident-atlas GPU node/hub stepping;
  - right rail exposes `Use GPU tire FEM kernel` and resident-atlas fallback
    toggles;
  - state is read back once per frame for the Bevy gizmo visualisation.

Follow-up GPU scaling work:

- add GPU reductions for contact patch diagnostics; **first heightfield/material
  FEM tread-node diagnostics plus local-normal world-wrench reduction done;
  reduced world force can now feed the GPU hub integrator through
  `tire_fem_hub_params_with_contact_reduction`; CPU `Vehicle` yaw now consumes
  tire aligning moment directly, and `TireVehicleYawGpuKernel` now provides
  GPU yaw/yaw-rate plus a first planar chassis translation/velocity lane from
  per-wheel tire forces plus Mz, along with a first per-wheel spin integration
  lane from drive/brake torque and tire longitudinal force, a first resident
  wheel-spin ping-pong owner that keeps spin/radius/inertia state on GPU
  across steps and feeds that resident spin state into GPU brush rows without
  CPU staging, plus GPU wheel-contact kinematics lanes for steering frame,
  contact velocity, slip ratio, and slip angle including a resident spin-state
  variant that avoids CPU staging, plus a first GPU suspension normal-load lane
  for spring-damper travel/rate/load with resident suspension rows feeding GPU
  brush row generation, plus `TireVehicleVerticalResidentState` with resident
  suspension-from-vertical and vertical-from-suspension ping-pong, plus a small
  `TireVehicleResidentBrushOwner` that sequences resident spin, suspension, and
  brush-row generation, plus a
  resident `BrushWheelOutput` path that feeds planar chassis integration
  without CPU staging and a resident planar chassis state owner that keeps
  position/velocity/yaw in ping-pong GPU buffers, plus
  `TireVehicleResidentChassisBrushOwner` that sequences resident spin,
  suspension, brush rows, resident brush output, and resident planar chassis
  stepping behind one API, plus
  `build_brush_rows_from_spin_suspension_planar_state` /
  `step_resident_chassis_brush_owner_from_planar_state` that feed resident
  planar X/Z/yaw state back into GPU brush-row generation, plus
  `build_brush_rows_from_spin_suspension_chassis_state` /
  `step_resident_chassis_brush_owner_from_resident_state` that feed both
  resident planar and resident vertical chassis state back into GPU brush-row
  generation, and a first GPU
  vehicle-to-brush row builder that writes `BrushWheelInput`-compatible rows
  plus a resident no-readback handoff into
  `TireBrushGpuKernel`; the first direct resident GPU FEM hub handoff is also
  available through `step_resident_vehicle_fem_hubs_from_brush_owner`; packed
  FEM contact-reduction rows can now feed both resident FEM hub integration and
  resident planar chassis X/Z/yaw integration without CPU staging; direct FEM
  contact-feedback force rows can now feed resident planar chassis X/Z/yaw via
  `step_planar_resident_from_feedback_output` /
  `integrate_planar_state_from_feedback_force_rows` without CPU staging; and
  `step_resident_vehicle_fem_atlas_from_brush_owner` sequences resident FEM
  node stepping, vehicle/brush output, hub-row assembly, and hub integration in
  one reusable substep API, and `TireResidentVehicleFemAtlasOwner` /
  `create_resident_vehicle_fem_atlas_owner` /
  `step_resident_vehicle_fem_atlas_owner` now provide the first long-lived
  combined resident vehicle+FEM atlas owner for runtime adoption, and Scene 120
  now defaults to that owner with raw resident-atlas/per-tire persistent GPU
  fallbacks, while broader live vehicle-scene migration onto that owner and
  deeper full-vehicle ownership remain future vehicle-level work**;
- feed transient slip into GPU FEM contact feedback; **initial reusable
  adapter done: `TireTransientGpuOutput` now drives FEM feedback omega and
  moving-ground surface velocity, with a non-hardware test guarding the
  kinematic handoff. Scene 122 now uses that adapter in the live bench path:
  persistent GPU transient output drives GPU FEM contact feedback, the
  returned velocity/bristle rows are applied to the tire state, and the right
  rail exposes raw/effective slip plus feedback force/bristle readouts. The
  same right rail now also shows `TireFemContactFeedbackOutput::patch_accounting`
  force-row/bristle-row counts, summed patch force, and mean/max force/bristle
  support so seam/curb regressions are visible even when a single max-force
  number looks plausible; it also shows live longitudinal/widthwise
  pressure-bin loads from `RealTireContact::pressure_distribution(8, 5)` so
  soft/stiff footprint shape changes are visible in unit-labelled bins. Full
  vehicle/chassis GPU ownership remains future integration work**;
- add GPU XPBD structural parity; **volume, bead, tread-belt edge, and
  sidewall bend-proxy XPBD are done in the node kernel; the initial sidewall
  quad dihedral-angle projection is also done in the node kernel; the CPU
  `sidewall_laminate_coupling` gate now records the crossed/helper ply-energy
  coupling expected from the eventual laminated sidewall constraint; the
  CPU/GPU `sidewall_ply_strain_xpbd` guardrail now reduces excessive tensile
  `|F·a0|` ply stretch in the standard runner and GPU node shaders, but true
  laminated sidewall constraints remain open**;
- move heightfield/deformable-terrain lookup onto GPU for FEM reductions
  (brush heightfield + friction-map lookup is done; FEM diagnostics
  heightfield + friction-map + water-film + built-in SDF lookup is done; FEM
  force/bristle feedback heightfield + friction-map + water-film +
  moving-surface velocity + built-in SDF lookup is done; FEM diagnostics and
  force/bristle feedback now also sample finite triangle patches with
  weighted multi-triangle manifold accumulation, a first per-triangle
  broadphase sphere, and a packed triangle BVH atlas/traversal path; FEM
  diagnostics/feedback now sample
  deformable-terrain sinkage maps for mean rut depth and rolling loss;
  `TireBekkerGpuKernel` now writes Bekker height/sinkage maps from tire
  contact cells; `TireBekkerFemTerrainOwner` now bridges those maps into
  `TireFemContactTerrain`; scene-122 live owner/rut hookup is done; fully
  GPU-resident FEM terrain feedback remains);
- support many tires with batched persistent FEM state instead of one mesh
  object per tire. **Initial `TireFemGpuBatch` host-side persistent batch API
  done with multi-tire GPU test; `TireFemGpuBatchDispatchPlan` now records
  fused-atlas sizing for heterogeneous batches; `TireFemGpuBatchAtlasLayout`
  now records contiguous per-tire node/tet/tread/bead/workgroup offsets as the
  packing contract; `TireFemGpuBatchAtlasBuffers` now packs and index-rewrites
  the node/tet/tread/bead/CSR/XPBD topology buffers for that atlas and records
  node/tet/bead/tread tire-row routing arrays for fused-shader param lookup and
  diagnostics reduction; `TireFemGpuBatchAtlasParamRow` now provides
  shader-layout-compatible per-tire parameter rows for that fused path;
  `TireFemGpuBatchAtlasOffsetRow` now provides shader-ready per-tire
  node/tet/contact/XPBD offset rows, including structural edge/dihedral
  ranges; `TireFemGpuBatchAtlasBuffers::validate` and `storage_bytes` now
  guard the full packed contract before dispatch; `split_outputs` now validates
  fused-atlas readback lengths and reconstructs per-tire `TireFemGpuOutput`
  rows; `extract_input` now round-trips packed atlas inputs back to per-tire
  `TireFemGpuInput` for CPU/reference parity debugging; `join_outputs`,
  `split_output_rows`, and `dispatch_atlas_reference_loop` now provide the
  reference-loop output path that the one-dispatch fused kernel can be
  compared against; `tire_fem_gpu_batch_atlas_binding_plan` now records the
  fused shader storage binding indices, row counts, read/write flags, and byte
  sizes; `tire_fem_atlas.wgsl` now declares the matching atlas-only WGSL
  bindings and contract entry that route nodes through tire ownership, param,
  and offset rows while running the same FEM physics as the single-tire GPU
  kernel; `TireFemGpuBatchAtlasDispatchContract` now locks the host-side launch
  metadata for that entry; `TireFemGpuKernel` now owns a real atlas physical
  compute pipeline and `dispatch_atlas_contract_blocking` binds all 44 packed
  atlas buffers, runs one GPU dispatch, and reads back atlas-wide rows;
  `dispatch_batch_atlas_blocking` now exposes that fused physical atlas shader
  through a high-level batch API for non-persistent callers;
  `TireFemGpuAtlasBatch` now provides resident packed atlas buffers, ping-pong
  bind groups, in-place per-tire param updates, one-dispatch fused node
  stepping, one-dispatch fused hub reduction/integration via `hub_reduce_atlas`
  / `step_resident_atlas_hubs`, split per-tire readback, state reseeding from
  either per-tire output rows, individual tire output rows, or already-packed
  atlas rows, plus packed hub-row upload/readback and one-tire hub reseeding
  for incremental viewer/vehicle adoption; the reusable GPU parity checklist now tracks
  `TireFemGpuAtlasBatch` and `tire_fem_atlas.wgsl` as the multi-tire batch GPU
  path, and `molla-kernels` exports both the resident atlas owner and the
  combined `TireResidentVehicleFemAtlasOwner` public API; stale host-loop batch
  docs now direct callers to resident owners when they want fused atlas
  ownership; Scene 120 now prefers the combined resident vehicle+FEM owner by
  default while retaining raw resident atlas and old per-tire persistent paths
  as debug/fallback toggles; the next higher-level work is moving the richer
  viewer/vehicle runtime scenes onto the combined vehicle+FEM owner and doing
  deeper live validation.**

### Stage E — non-calibration validation, telemetry, and tuning workflow

This stage deliberately does **not** mean fitting against real tire datasets.
It is the practical workflow needed to keep the hand-tuned physical model from
turning into hidden viewer hacks.

- Add deterministic tire-rig examples/tests:
  - vertical load/deflection sweep at several pressures; **initial reusable
    load/deflection sweep helper plus monotonic test done**;
  - longitudinal slip sweep on asphalt, dirt, wet film, soil, and ice;
    **initial reusable longitudinal slip sweep helper plus asphalt-vs-ice
    separation test done; reusable dirt/wet/soft-soil/ice surface matrix
    helper/test done for the brush/material path; reusable deformable-rut
    matrix helper/test done for CPU Bekker soil; default runner now includes a
    moving-belt snapshot and explicit `moving_ground_work_positive` acceptance
    envelope**;
  - combined braking/cornering sweep; **initial reusable combined-slip grid
    helper plus one-friction-budget test done**;
  - camber sweep with aligning-moment and pneumatic-trail readouts;
    **initial reusable camber sweep helper plus lateral-force sign test done;
    richer moment/trail boundedness plus low-friction sliding-saturation
    assertions done; structural conicity/ply-steer lateral-bias regression
    done without changing the wheel steering frame**;
  - step torque, step brake, and step steer response; **initial reusable
    fixtures/tests done, with wheel-spin integration for torque/brake and
    transient relaxation checks for raw-vs-effective slip/angle lag; the
    standard non-calibration runner now includes acceptance reports for those
    step responses**;
  - bump/curb/heightfield envelope-contact sweep; **initial reusable
    heightfield envelope helper/test done for bump normals, horizontal force
    from terrain normals, and pressure-dependent patch length; explicit
    heightfield curb test done to guard against single-point contact collapse;
    FEM tread curb wrapping/bridging regression done for CPU
    `TetraTireFem::step_on_terrain`; initial ordered rolling obstacle sequence
    helper/test done for FEM bump approach → crest → exit with deformed
    tread-node contact and brush-wrench feedback; the default non-calibration
    runner now emits `rolling_obstacle_approach`, `rolling_obstacle_crest`, and
    `rolling_obstacle_exit` snapshots so the viewer replay panel can inspect
    the ordered sequence rather than only a static curb sample, and it emits a
    `rolling_obstacle_sequence` acceptance report for crest height, exit
    recovery, multi-cell patch support, patch length, and horizontal force from
    non-flat bump normals. The replay panel now links that sequence-level
    report back to each ordered obstacle frame so the acceptance gates are
    visible while inspecting approach, crest, or exit telemetry**;
  - thermal/wear run with sustained slip and cooling recovery. **initial
    sustained-slip thermal/wear rig helper plus heat/wear test done; explicit
    multi-phase cooling-recovery bench helper/test done; the standard
    non-calibration runner now emits separate `thermal_wear_heat_peak` and
    `thermal_wear_cooling_recovery` snapshots with heat/cool acceptance gates
    so the viewer can inspect the phase split rather than only the final cycle
    state**.
- Add viewer/debug overlays:
  - contact-cell locations and normals; **initial scene-121 patch
    bound/normal markers done for the user car; scene-121 now also draws a
    sampled per-cell front patch with local normals; scene-121 now mirrors the
    axle contact solve onto all four visible tires for all three cars and draws
    all sampled contact cells/normals when debug overlays are enabled**;
  - bristle shear vectors and sliding/adhesion coloring; **initial scene-121
    per-cell bristle-force arrows plus green adhesion / red sliding coloring
    done for the user-car front patch, using reusable contact-cell data even
    when GPU aggregate force is active; scene-121 now draws the same
    adhesion/sliding + shear arrows for every visible tire on every car**;
  - per-wheel force/moment arrows; **initial world-force, lateral-force,
    aligning-moment, and chassis-yaw moment glyphs done in scene 121; scene-121
    now also draws per-wheel force/lateral-force arrows, per-wheel aligning
    moment glyphs, and small per-contact yaw-moment contribution bars**;
  - contact centre, force resultant, and pneumatic trail; **initial scene-121
    markers/segment done for the user car; scene-121 now draws these markers
    for every visible tire on all three cars**;
  - surface/tread/carcass/rim temperatures; **initial scene-122 layered
    thermal overlay done: the right rail already reports each thermal layer,
    and gizmos now draw separate surface/tread/carcass/rim temperature bars
    beside the heat/wear/μ stack**;
  - pressure, cavity volume, and XPBD correction magnitudes; **initial
    scene-120 FEM diagnostics done: the right rail reports animated-tire
    pressure, cavity volume/rest ratio, effective cavity pressure, bead load,
    tet-volume residual, and CPU XPBD volume/bead/structural correction
    magnitudes; gizmos draw pressure/cavity/XPBD bars above each tire**;
  - CPU-vs-GPU output deltas when both paths are enabled. **done for scene-121
    brush force/patch/slip/friction-budget/aligning-moment readouts; initial
    scene-120 animated-tire FEM CPU-vs-GPU node/hub/cavity deltas done**.
- Add telemetry structs/logging hooks in the reusable crates, not only in the
  viewer. **Initial unit-labelled wheel/patch/cell telemetry structs plus
  finite/force-accounting tests are done in `molla-vehicle`; scene-121 viewer
  force/contact/trail overlays are started; dependency-free CSV logging hooks
  are done for wheel-level and per-contact-cell telemetry; `TireRigSnapshot`
  now records command inputs, preset names, terrain state, heat/wear state,
  provenance tags, optional CPU-vs-GPU force deltas, and unit-labelled CSV
  rows for reusable rig reports; `TireSolverBudgetTelemetry` now attaches FEM
  substeps, XPBD iterations/compliances/corrections, max node speed,
  tet-volume error, bead mount drift, and dissipated power/energy to those snapshots;
  `TirePressureDistributionTelemetry` now exposes longitudinal and widthwise
  patch pressure bins for brush/FEM contact diagnostics; `TireRigSnapshot`
  now exports `moving_ground_power_w` so moving-belt/work-sign checks are
  traceable in logs.**
  - one wheel-level sample per simulation step;
  - optional per-contact sample summaries for patch debugging;
  - explicit units in field names or docs;
  - no external data fitting requirement.
- Add sanity/regression gates:
  - no NaNs/infinities for all tire scenes;
  - tires do not tunnel below supported terrains after projection;
  - lower pressure increases patch length/width under the same load;
    **initial pressure-distribution diagnostics test done for high-vs-low
    pressure longitudinal span/peak pressure and FEM widthwise bins**;
  - ice caps force far below dry asphalt even with a large patch;
    **done in the standard runner as `ice_large_patch_grip` via
    `TireIceLargePatchReport`: it compares a low-pressure large-patch tire on
    dry asphalt vs ice, proves the low-pressure patch is larger than a
    high-pressure reference, and still requires ice to cap force, expose
    sliding, and stay inside the shared friction budget**;
  - force and moment curves are continuous across adhesion/sliding transition;
    **initial standard-runner gate done as `adhesion_sliding_continuity` via
    `TireAdhesionSlidingContinuityReport`: it sweeps combined brush slip from
    mostly adhesive cells into sliding saturation and checks finite outputs,
    bounded adjacent force/moment jumps, stable force/moment signs, and one
    shared friction budget. The viewer replay selector exposes the same case
    as `Slip/contact · adhesion to sliding` with an expectation string so
    transition jumps are reportable instead of eyeballed.**
  - GPU brush/FEM contact outputs stay within tolerance of CPU reference paths;
  - total dissipated energy from slip/rolling/thermal paths is non-negative.
    **Initial `TireEnergySignAudit` report done for dissipated power/energy,
    friction-budget residual, pneumatic-trail sign convention, and
    moving-ground work sign, with `moving_ground_power_w` carried in reusable
    snapshot CSV telemetry and shown in the shared tire-rig replay panel's
    energy/sign audit readouts.**
  - dimensionless tire-rig sanity ratios stay finite and inside broad
    non-calibration envelopes. **Initial
    `TireRigDimensionlessSanityMetrics` + acceptance-report test done for
    force/load, friction-budget, patch/radius, pressure/load, energy/load, and
    CPU/GPU relative-error ratios.**
- Add non-calibration acceptance reports:
  - every reusable rig emits the command inputs, preset names, model
    provenance tags, and unit-labelled telemetry used for the check;
    **initial `TireRigSnapshot` record + CSV row/header test done for reusable
    rig points**;
  - every report lists qualitative expectations and pass/fail envelopes, not
    fitted target curves; **initial `TireRigAcceptanceCheck` /
    `TireRigAcceptanceReport` text report done with pass/fail qualitative
    envelope tests; the default runner now asserts every emitted snapshot has
    the generic dimensionless sanity and energy/sign reports in addition to any
    scenario-specific acceptance report**;
  - every high-frequency path reports CPU-vs-GPU absolute and relative deltas
    when a GPU implementation exists;
    **initial reusable tire GPU parity checklist/report done with CPU+GPU,
    CPU-only-debug, and planned-GPU status rows plus tests that require proof
    or explicit reason/follow-up per row**;
  - every scene with tire physics has a short "what should happen" description
    that can be compared against the live behavior. **Initial right-rail tire
  truth labels done for the tire lab, legacy offroad vehicle, Bekker sinkage,
  soil-compaction, and 120/121/122 scenes, including physics path, expected
  behavior, physical controls/state, debug controls, and known approximations;
  the shared 120/121/122 replay panel now also
    shows the live known-approximation ledger from `molla-vehicle`. The shared
    replay selector now exposes the standard diagnostic reports with explicit
    descriptions, so construction sanity, frame/sign conventions, patch
    convergence, road-condition state, FEM solver budget/stiffness, DEM/MPM
    soil feedback, cavity/bead support and bead-seat load cases, sidewall
    laminate/dynamic-load checks, triangle seam/surface stress checks,
    low-speed robustness, thermal/wear abuse ordering, and realism-source
    classification can be inspected from the viewer next to their acceptance
    envelopes.
    Scenario-specific report-only cases have also
    been promoted to replay snapshots, so pressure/grip/film/transition/loss/
    shape/step-response envelopes are no longer hidden in text output only.
    The default runner now fails if a report has no replay context, preventing
    future report-only tire checks from silently bypassing viewer
    falsification.**
- Add parameter organization:
  - move tire, road, soil, thermal, and solver constants into named presets;
    **done for `RealTireParams`, `SurfaceFilmState`, `TireThermalLayers`,
    `TetraTireFemParams`, and `DeformableTerrain`; viewer scenes 120/121/122
    use the tire preset as source of truth; future passes can add more
    specialized presets without changing the pattern**;
  - document units and recommended ranges beside the struct fields;
    **initial reusable preset-governance report done with unit labels, broad
    ranges, provenance, pass/fail status, text output for preset reviews, and
    shared viewer right-rail exposure through the tire truth-label panel**;
  - keep the viewer sliders as preset overrides, not as the source of truth;
    **viewer tire controls now carry stable telemetry keys in the truth-label
    panel, and tests require every preset override control to be an explicit
    named `_override` telemetry input with non-hidden provenance text instead
    of an anonymous force multiplier**;
  - mark parameters as physical, numerical-stability, or debug/visualization.
    **initial governance tags distinguish physical, hand-tuned, and numerical
    stability parameters across named tire/terrain presets; viewer-side tire
    control metadata now classifies physical state, preset override,
    numerical-stability, debug, and visualization-only controls in the
    right-rail truth label**.

## External modeling notes shaping the next pass

Realistic tire work usually combines a structural tire model with a local
tread/road contact model:

- Brush models remain a useful physics-grounded basis for tire/road contact,
  and even advanced tire formulations commonly keep brush-like local contact
  behavior.
- Real-time physical tire models often discretize tire geometry and couple
  bristle kinematics to carcass deformation instead of using a single point
  friction coefficient.
- Validated FEM tire studies use nonlinear tire/rim and tire/road contact,
  load-deflection curves, pressure-distribution checks, and experimental
  comparison. Calibration against those datasets is out of scope here, but the
  same structural lessons still matter: rim/bead contact, inflation setup,
  surface-to-surface tread/road contact, material regions/reinforcements, and
  clear diagnostics for static/dynamic behavior.

---

# Big Picture — May 2026 consolidation

This section absorbs `PLAN_TOURQUE.md` and `PLAN_ROLL.md` (now deleted) plus
the new file-split + scaling plan. Three streams of work, in priority order:

1. **Stream A — File split.** `crates/molla-vehicle/src/real_tire.rs` is
   36 000+ LOC in a single file. Cold compile time, IDE responsiveness,
   review surface, and merge-conflict blast-radius are all unmanageable.
   Split into a `real_tire/` directory of focused modules before any
   further physics work. **Block until done.**
2. **Stream B — `TetraTireFem` co-rotational refactor** (the un-shipped
   half of `PLAN_ROLL.md`). The rigid-rotation prototype was reverted
   because of the contact-clamp / velocity-rotation oscillation. The
   only stable answer is body-frame canonical storage with contact
   applied via the body↔world transform.
3. **Stream C — 40-wheel scaling target.** The user goal is ~10 machines
   × 4 wheels = 40 live FEM tires real-time, plus chassis + terrain +
   render. CPU per-wheel is already near saturation, so the GPU
   persistent atlas path has to be the production path. Stream B has
   to keep the body-frame layout the same on host and GPU so Stream C
   drops in cleanly.

## Stream A — file split

### Why this is urgent

- `real_tire.rs` = 36 415 LOC, 218 public items, one `#[cfg(test)]
  mod tests` near line 27 329 that bundles every test in the crate.
- Every edit recompiles the whole file → ~30 s rebuild on the dev
  laptop, kills tight iteration.
- A second engineer touching the file means certain merge conflicts.
- The file mixes seven loosely-related concerns: brush model,
  transient relaxation, contact telemetry, governance/realism/audit
  reports, rig sweeps, thermal/wear, FEM tire, GPU parity ledger.
  These have separate review audiences.

### Target module layout (`crates/molla-vehicle/src/real_tire/`)

Each module re-exports its public items via `pub use` from a thin
`real_tire/mod.rs` so external consumers (viewer scenes, kernels) see
no breakage. Numbers in parentheses are current source-line ranges in
the single file; the split moves the matching items into the named
module.

```
real_tire/
├── mod.rs                  # pub use re-exports + crate-internal
│                           #   shared helpers (finite_or, vec3 math,
│                           #   smoothstep01)
├── surface_material.rs     # SurfaceMaterial, SurfaceFilmState
│                           #   (~ lines 19–98, 17216–17630)
├── brush.rs                # RealTireParams, RealBrushTireModel,
│                           #   BrushParams, smooth_combined_brush_force,
│                           #   pressure / load sensitivity helpers
│                           #   (~ lines 99–421, 17338–17415)
├── transient.rs            # RealTireTransientState,
│                           #   TireRelaxationContext,
│                           #   TireLayeredTransientState, layer
│                           #   advancement (~ lines 458–908)
├── evaluate.rs             # evaluate_brush_tire,
│                           #   evaluate_brush_tire_oriented,
│                           #   evaluate_brush_tire_transient,
│                           #   evaluate_brush_tire_layered_transient*
│                           #   (~ lines 909–974, 17374–17415)
├── contact.rs              # TireContactCell, RealTireContact,
│                           #   TireContactPatchTelemetry,
│                           #   TireContactCellTelemetry,
│                           #   TirePressureDistributionBin/Telemetry
│                           #   (~ lines 975–1528)
├── telemetry.rs            # RealTireTelemetrySample,
│                           #   TireRigSnapshot, TireSolverBudgetTelemetry,
│                           #   TireRigAcceptanceCheck/Report,
│                           #   energy/sign audit, dimensionless metrics
│                           #   (~ lines 1529–1670, 4342–5113)
├── realism.rs              # TireModelProvenance,
│                           #   TireRealismSourceItem/Report,
│                           #   TirePresetGovernance*,
│                           #   TireParameterObservabilityReport
│                           #   (~ lines 1671–1825, 2875–3634)
├── triangle.rs             # TireTriangleSeam/Stress/Force/Density/
│                           #   LargeManifold reports + helpers
│                           #   (~ lines 1826–2874)
├── parity_ledger.rs        # TireGpuParityStatus, TireGpuParityItem,
│                           #   tire_gpu_parity_checklist,
│                           #   TireRuntimeProductionGateReport,
│                           #   TireApproximationStatus,
│                           #   TireKnownApproximation,
│                           #   tire_known_approximation_ledger
│                           #   (~ lines 3635–4341)
├── rig/                    # TireRigRunConfig/Output sweep machinery
│   ├── mod.rs              # pub use; tire_rig_standard_non_calibration_run
│   ├── config.rs           # TireRigRunConfig, TireRigSurfaceKind, etc.
│   ├── envelope.rs         # heightfield envelope sweep, rolling obstacle
│   │                       #   sequence, load-deflection sweep
│   ├── slip_sweep.rs       # longitudinal/combined/camber slip sweeps
│   ├── thermal_wear.rs     # tire_rig_thermal_wear_run/cycle,
│   │                       #   thermal/wear phase + point types
│   └── replay.rs           # TireReplayArtifactCoverageReport
├── reports/                # the ~80 standalone diagnostic report
│   ├── mod.rs              #   functions live here, one file per
│   │                       #   logical family
│   ├── reference_shape.rs
│   ├── slip_law.rs
│   ├── adhesion.rs
│   ├── transient_envelope.rs
│   ├── low_speed.rs
│   ├── frame.rs
│   ├── road_condition.rs
│   ├── layer_ablation.rs
│   ├── convergence.rs
│   ├── thermal_abuse.rs
│   ├── thermal_carcass.rs
│   ├── brake_rim.rs
│   ├── soil.rs
│   └── condition.rs
├── carcass.rs              # DynamicTireCarcass / Params
│                           #   (~ lines 17631–17925)
└── fem/                    # The big one. `TetraTireFem`.
    ├── mod.rs              # public surface; orchestrates the
    │                       #   per-substep step_inner that lives in
    │                       #   solver.rs
    ├── params.rs           # TetraTireFemParams + presets +
    │                       #   TetraTireFemSolverMode
    ├── mesh.rs             # mesh build (cavity, beads, tread, tets,
    │                       #   structural edges, dihedrals)
    ├── tetra.rs            # TetraElement, Neo-Hookean + Holzapfel
    │                       #   fibre force, polar decomposition
    ├── xpbd.rs             # all XPBD constraint projections
    │                       #   (volume, bead, structural edges,
    │                       #   dihedrals, sidewall ply strain)
    ├── contact.rs          # tread-node contact branch (XPBD floor
    │                       #   projection, bristle integration,
    │                       #   spin slip from PLAN_TOURQUE Stage C)
    ├── solver.rs           # step_inner + step_on_terrain; calls
    │                       #   tetra/xpbd/contact modules
    ├── matrix_free.rs      # matrix_free_newton_correction +
    │                       #   step_on_terrain_matrix_free_newton_debug
    │                       #   + the 5 newton reports
    ├── diagnostics.rs      # TetraTireFemDiagnostics,
    │                       #   TetraBeadSeat*, TetraCavity*,
    │                       #   TetraSidewall* reports, TetraFem*
    │                       #   reports (hardening, deformation,
    │                       #   contact slip, construction sanity)
    ├── gpu_snapshot.rs     # TetraTireFemGpuSnapshot
    │                       #   (host-side mirror for GPU readback)
    └── torque.rs           # PLAN_TOURQUE Stage A/B/D/F additions:
                            #   spin DOF (hub_angle, hub_omega),
                            #   drive/brake torque, dynamic-hub
                            #   integration, set_hub_dynamic +
                            #   set_hub_position/velocity hand-off,
                            #   apply_rigid_rotation_about_hub
                            #   (dead-code today; revived by Stream B)
```

### Acceptance criteria for the split

- **Zero functional change.** Every test in the workspace passes
  byte-for-byte before/after the split, including the long-running
  default runner + viewer panel tests + ignored GPU lanes.
- **Public API preserved.** External re-exports
  (`molla_vehicle::TetraTireFem`, `evaluate_brush_tire`, etc.) keep
  their names through `pub use` from the new `mod.rs`.
- **`real_tire.rs` ceases to exist** as a single file. The path
  `crates/molla-vehicle/src/real_tire.rs` is replaced by
  `crates/molla-vehicle/src/real_tire/mod.rs`.
- **No module > ~3 500 LOC.** If one ends up that big, split further.
- **One PR per top-level family.** Land in this order so each step is
  reviewable and bisectable:
  1. `surface_material.rs` + `brush.rs` + `transient.rs` + `evaluate.rs`
     (the brush stack, ~1 800 LOC moved).
  2. `contact.rs` + `telemetry.rs` (RealTireContact + sampling, ~3 000
     LOC).
  3. `realism.rs` + `triangle.rs` + `parity_ledger.rs` (governance +
     triangle reports + ledger, ~3 500 LOC).
  4. `rig/` (the rig sweeps, ~6 000 LOC).
  5. `reports/` (standalone diagnostic reports, ~7 000 LOC).
  6. `carcass.rs` (the small `DynamicTireCarcass`, ~300 LOC).
  7. `fem/` (the big one). Land its sub-modules in this order so each
     is bisectable:
     - `fem/params.rs` (presets + solver mode)
     - `fem/mesh.rs` (init geometry)
     - `fem/tetra.rs` (constitutive model)
     - `fem/xpbd.rs` (constraint projections)
     - `fem/contact.rs` (contact branch + bristle)
     - `fem/torque.rs` (PLAN_TOURQUE state)
     - `fem/solver.rs` (the orchestrating step_inner)
     - `fem/matrix_free.rs`
     - `fem/diagnostics.rs`
     - `fem/gpu_snapshot.rs`
- **`tests` module split too.** The current `#[cfg(test)] mod tests`
  near line 27 329 covers every public function. After the split,
  each module owns its own `#[cfg(test)] mod tests` so the failing-test
  output names the right concern. The full runner test stays in
  `rig/replay.rs` because that's where it logically lives.
- **No new `pub` items.** The split must NOT widen any private API.
  Use `pub(crate)` for cross-module crate-internal helpers; preserve
  the existing public surface exactly.

### Risks for the split

- **Test test_module gets enormous.** If `mod tests` ends up at
  10 000+ LOC after the move, split it per public family too.
- **Cross-module helper churn.** Helpers like `finite_or`,
  `sanitize_vec3`, `normalize3`, `smoothstep01` are used everywhere.
  Put them in `real_tire/mod.rs` (or a small `internal_math.rs`) as
  `pub(crate)` so every module imports from one place.
- **`impl` blocks split across modules.** Today every `impl` for a
  type lives next to the struct. After split, an `impl X` in
  `realism.rs` plus an `impl X` in `parity_ledger.rs` is fine — Rust
  allows multiple impl blocks per type as long as they're in the same
  crate. Keep methods with their concern.
- **Compile time temporary spike.** The split itself recompiles a lot.
  Plan it for a quiet session.

## Stream B — `TetraTireFem` co-rotational refactor

### Status (carried from `PLAN_ROLL.md`, now deleted)

| Stage | What | Status |
|---|---|---|
| **A** | body-frame storage (`body_pos`, `body_vel`) + accessors | ✅ done |
| **B** | body↔world transform helpers (`body_to_world`, `world_to_body`, `body_velocity_to_world`) | ✅ done |
| **C phase 1** | rigid-rotate FEM state by ω·dt at top of `step_inner` | ⚠️ REVERTED — jelly wobble |
| **C proper** | body-frame canonical storage; physics in body, contact in world via transform | ⏳ not yet |
| **D** | scene 123 visual cleanup (drop rotating-spoke crutch once mesh truly rolls) | ⏳ not yet |
| **E** | matrix-free Newton timestep + body-frame integration | ⏳ not yet |

### Stage C proper — the real refactor

**Architectural change.** `body_pos` / `body_vel` become the **canonical**
state. `pos` / `vel` are a **derived view** rebuilt at the end of each
substep from `R(hub_angle) · body + hub`. The FEM physics (forces,
constraints, projections) operates on body. Contact runs in world (terrain
is world-fixed) and the resulting Δp / Δv are transformed back into body
via `R⁻¹`. The spin contribution `v_spin = ω × r` falls out of the
body↔world transform automatically.

**Stage C proper PRs, in order.** Each is independently reviewable; each
maintains green tests with `hub_omega = 0`.

1. **C.1 — promote `body_pos` to canonical for the FEM internal force
   loop.** The Neo-Hookean + Holzapfel pass reads `body_pos`, writes to a
   body-frame force buffer. Strain energy is invariant under rotation
   (proof: `F_world = R · F_body`, `F^T F` rotation-invariant), so the
   physics is identical at ω = 0. Tetra grads + fibre directions stay
   constant (body-frame already). Test: with ω = 0 and identical mesh
   state, body-frame and world-frame produce bit-identical residuals.
2. **C.2 — promote `body_pos` to canonical for the XPBD passes.** Volume,
   bead, structural edge, dihedral, ply strain projections all run on
   `body_pos`. Bead target is `body_offset` directly (no rotation needed
   because both are body-frame). Same ω = 0 invariance test.
3. **C.3 — run contact in world, apply Δp / Δv to body.** Each tread node:
   `world_p = body_to_world(body_p)`. Probe terrain at `world_p`. XPBD
   contact projection produces a world-frame `Δp_world`. Transform back:
   `Δp_body = world_to_body_vec(Δp_world)` (vector version of
   `world_to_body` that skips the hub translation). Apply to
   `body_pos[i]`. Same for `Δv`. Spin contribution drops out of the
   transform — the body-frame node that's currently at world-bottom is
   the one that gets the contact correction this step, exactly what
   the user asked for ("deformation stays at the world bottom while
   material flows through it").
4. **C.4 — refresh `pos` / `vel` from body at the end of `step_inner`.**
   `pos[i] = body_to_world(body_pos[i])`. Velocity refresh uses
   `body_velocity_to_world(body_pos[i], body_vel[i])` (already in
   Stage B). Public accessors (`nodes()`, `velocities()`) still return
   the world-frame view; nothing external breaks.
5. **C.5 — `apply_rigid_rotation_about_hub` becomes unused and gets
   deleted.** The kinematic rotation pass was a workaround for not
   having body-frame canonical storage; with C.1–C.4 in place,
   rotation falls out of the body↔world transform and no per-substep
   rotation is needed.
6. **C.6 — performance: persistent scratch buffers.** Today each
   `step_inner` invocation does `let mut force = vec![[0.0; 3]; n]` —
   one heap allocation per substep. At 8 000 substeps/s × 40 wheels =
   320 k allocs/s. Move `force`, `node_stiffness_diag`, and any
   per-substep `Vec` into a `StepScratch { force, node_stiffness_diag,
   .. }` field on `TetraTireFem`; reuse across substeps; `clear()`
   instead of re-allocating.

### Stage D — scene 123 cleanup

Once C.4 is live, the FEM mesh visibly rotates because `nodes()` now
returns rotated positions. Drop the rotating-spoke gizmo (or keep it
as a small inset since users like reading angular velocity that way).
Confirm "deformation stays at world-bottom" visually with a recorded
slow-motion capture.

### Stage E — matrix-free Newton timestep

`step_on_terrain_matrix_free_newton_debug` and the matrix-free
correction helper currently bypass `step_inner`. Once Stage C is in,
mirror the body-frame canonical pattern: residual / Jv probe / diagonal
extraction all run on `body_pos`; contact integration in world; result
back to body. PLAN_TOURQUE Stage G GPU `phys5` row already carries
`hub_omega`; mirroring it in the matrix-free path is straightforward.

### Stage F — chassis hand-off & dynamic hub on GPU (also from PLAN_TOURQUE Stage G part 3)

- `set_hub_position` / `set_hub_velocity` / `last_hub_reaction_xyz`
  already exist (PLAN_TOURQUE Stage F).
- The GPU dynamic-hub atomic accumulation is the open Stream C piece:
  per-workgroup atomic sum of the contact reaction across all tread
  nodes of a wheel, then a tiny per-wheel hub-integrator kernel.
  Stream C section below.

## Stream C — 40-wheel scaling

### Performance budget

Per-wheel cost at viewer-typical resolution (14×3×3, ~150 nodes,
~500 tets):

- 8 000 substeps/s (per-step `dt ≈ 1.25e-4` to keep elastic CFL).
- ~500 tets × O(matrix3 + log) per substep ≈ 50 ns × 500 = 25 µs
  on CPU per substep just for the Neo-Hookean inner loop.
- That's 200 ms of CPU per wheel per simulated second. 5× real-time
  on a single core. **One wheel saturates a core.** 40 wheels does not
  fit on CPU, period.

### What ships on GPU instead

GPU atlas batch in `crates/molla-kernels/src/tire_fem.rs` already
stamps multiple tires per dispatch. Stream C work:

- **G1 — GPU body-frame storage.** Mirror Stream B's body-frame
  layout in `TireFemPersistentMesh`. Add `body_pos_buf` / `body_vel_buf`
  per wheel; the WGSL step reads body, writes body; `pos_out` is the
  derived world view (used only for readback / rendering).
  Source-guard already pinned by `TireFemTorqueGpuContract` from
  PLAN_TOURQUE Stage G.
- **G2 — WGSL spin-coupled contact in body frame.** Replace the
  current `v_spin = ω · r_y` term (which lives in world) with the
  same body↔world transform Stream B uses, so the GPU and CPU paths
  arrive at byte-identical (within FP tolerance) per-node positions
  after one substep.
- **G3 — atomic dynamic-hub integration.** Per-workgroup reduce of
  contact reaction; per-wheel hub integrator; `hub_buf` updated
  on-GPU; `host_hub_readback` only when chassis / camera needs it.
- **G4 — CPU-vs-GPU parity hardware lane.** Drives both with
  identical torque + terrain + initial state. Asserts hub trajectory
  matches within tolerance over N substeps. Becomes the gate that
  flips `TireFemTorqueGpuStatus` from `Live` (Stream B done) to
  `LiveValidated`.

### What stays on CPU

- Single-wheel debug / unit-test path. The CPU `TetraTireFem` is the
  authoritative reference for the parity lane.
- `step_on_terrain_matrix_free_newton_debug` (the implicit-step
  validation lane). Will be migrated to body frame in Stage E above,
  but stays CPU-only — it's a correctness gate, not a production
  stepper.

## Execution order across all three streams

A single linear plan, not parallel work, so each merge is small and
bisectable:

1. **Stream A — file split** (block until done).
2. **Stream B.C.1 + C.2** — body-frame canonical for FEM forces +
   XPBD passes. ω = 0 invariance tests.
3. **Stream B.C.3 + C.4** — body-frame canonical for contact +
   end-of-step world refresh. **First user-visible win:** rolling
   tire with deformation at the world bottom.
4. **Stream B.C.5 + C.6** — delete `apply_rigid_rotation_about_hub`
   dead code; add `StepScratch` to kill the per-substep allocation.
5. **Stream B.D** — scene 123 visual cleanup, slow-motion sanity check.
6. **Stream B.E** — matrix-free Newton timestep in body frame.
7. **Stream C.G1 + G2** — GPU body-frame storage + spin-coupled contact
   in body frame.
8. **Stream C.G3** — GPU dynamic-hub atomic accumulation.
9. **Stream C.G4** — CPU-vs-GPU parity hardware lane; flip status to
   `LiveValidated`; declare 40-wheel target achievable.

## Historical PLAN_TOURQUE summary (file deleted)

Shipped end-to-end:

- **Stage A** — `hub_angle`, `hub_omega`, `drive_torque_n_m`,
  `brake_torque_n_m`, `hub_reaction_xyz` fields + getters/setters on
  `TetraTireFem`.
- **Stage B** — `integrate_spin_dof(dt)` in `step_inner`: drive/brake
  torque → ω via thin-walled-cylinder I_wheel.
- **Stage C** — spin contribution `v_spin = ω·r_y / −ω·r_x` injected
  into tread bristle slip (still world-frame; will move to body
  frame in Stream B.C.3).
- **Stage D** — dynamic-hub XZ integration: contact reaction
  accumulated per substep, integrated into `hub_v_xz`, hub
  translates. `set_hub_dynamic()` / `set_hub_kinematic()` toggle.
- **Stage E** — viewer scene 123 rewritten to use the real torque
  API; `set_hub_kinematic` workaround removed.
- **Stage F** — chassis hand-off: `set_hub_position(p)`,
  `set_hub_velocity(v)`, `last_hub_reaction_xyz()`.
- **Stage G parts 1–3** — GPU surface for torque: host params row
  (`hub_omega`, `drive_torque_n_m`, `brake_torque_n_m`),
  `TireFemTorqueGpuContract`, WGSL `phys5` uniform row, spin-coupled
  XPBD contact branch in `tire_fem.wgsl`, hardware lane
  `tire_fem_gpu_spin_pushes_grounded_tread_backward_in_contact`.
  Status auto-advances to `Live`.

Remaining open follow-ups from PLAN_TOURQUE folded into Streams B/C
above (chiefly: matrix-free Newton spin coupling, GPU dynamic-hub
atomic integration, CPU-vs-GPU parity lane). No standalone
`PLAN_TOURQUE.md` needed any more.

## Historical PLAN_ROLL summary (file deleted)

Shipped: Stages A + B (body-frame storage + transform helpers).
Reverted: Stage C phase 1 (rigid rotation per substep — jelly wobble).
Open: Stage C proper + D + E + the matching GPU mirror, all folded
into Stream B and Stream C above. No standalone `PLAN_ROLL.md` needed
any more.
