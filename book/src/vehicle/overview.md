# molla-vehicle overview

`crates/molla-vehicle` is the wheeled-vehicle simulation crate: terrain,
tires, suspension, driveline, a lightweight chassis, and the
differentiable system-identification demo. It is deliberately
**dependency-free** (no other workspace crates), and is re-exported
through the facade as `molla::vehicle::*`. It is also the home of the
`real_tire/` tree — the deformable tetra-FEM tire that the tire viewer
scenes (119–123) and the 40-wheel GPU scaling effort build on
(`PLAN_REALTIRE.md` / `PROGRESS.md` at the repo root track that work).

## Subsystems

| Module | What it provides |
|--------|------------------|
| `terrain` | `FlatTerrain`, `HeightfieldTerrain`, SDF and triangle-mesh terrain, plus the Bekker-pressure-sinkage `DeformableTerrain` and the DEM `GranularBed` (`deformable.rs` / `granular.rs`) |
| `tire` | Force-element tire models: `RigidTire`, `TmSimpleTire`, `Pac89Tire` (diagnostic shape only — molla ships no fitted Magic-Formula source), and the reusable `RealBrushTireModel` |
| `real_tire/` | The deformable tire stack: tetra tire FEM (`TetraTireFem`), brush/bristle traction, cavity pressure, thermal/wear state, parity ledger against the GPU kernels in `molla-kernels`, and the embedded validation rig/report harness |
| `suspension` | Per-corner spring/damper (`force()` from stiffness, damping, max compression). The `SuspensionKind` variants are descriptive labels by decision (INSPECTION.md F9): kind-driven camber/toe kinematics needs a chassis roll DOF the 3-DOF `Vehicle` deliberately excludes |
| `powertrain` | `Engine` (torque curve + redline), `Transmission` (gear ratios + RPM-triggered auto-shift with hysteresis), `Driveline` (FWD/RWD/AWD split), Ackermann steering, and `Vehicle::apply_driver` |
| `vehicle` | The 3-DOF chassis (`pos`/`vel`/`yaw`/`yaw_rate`, semi-implicit Euler) with four `Corner`s carrying wheel spin, steer, and tire state |
| `sysid` | Differentiable-simulation demo: recover tyre-ground friction from an observed trajectory by Gauss-Newton over finite-difference rollouts (`identify_friction`). Viewer scene 124 drives it live |

## Framed soil persistence

`SoilTileLayout::tile_at_local` classifies split local/world coordinates without
rounding them into an absolute position. The solver/facade compute API supplies
`MpmSoilFrameTileArchive` and `MpmSoilFrameSelection` through
`MpmSoilFrameCheckpoint::partition_tiles`. These retain world offset, world-tile
keys, global IDs and all nine local particle/history streams through selection,
same-frame writeback and checked serialization. Lossless rebasing returns a new
archive/selection epoch; stale or mismatched-frame writes reject.

The detached `MpmSoilFrame` can explicitly `restore_selection` with physical
state and GPU global IDs together. Its `checkpoint_tiles_with` captures identified
framed archives, and `recenter_with` preserves the ID mapping/buffers. Plain
particle-only restore invalidates IDs. Capture/recenter are blocking boundaries;
mapping epochs do not certify coordinate freshness or world-residency ownership.

`MpmSoilFrameStore` adds blocking durable full saves/loads, bounded selected reads
and same-frame writeback through the existing atomic publication protocol. The
shared frame survives reopen, and wrong-family opens reject. Its `begin_read` and
`begin_writeback` move bounded selected I/O to exclusive-owner workers; polling
returns framed inputs without copying payloads. Dropping tasks does not cancel
work or release locks early. `MpmSoilDiskWorld::new_framed` integrates framed
storage with live residency at a fixed shared offset. Swept bounds stay local;
residency/activity/pins use world tiles. Optional GPU pins require tile-aligned
frames; fractional frames retain exact host pinning. Moving-window/machine
rebasing and physical streaming
boundaries remain unimplemented. No implicit unframed GPU/store conversion occurs.
[API and verification scope](../../../crates/molla-solvers/examples/README.md#framed-soil-tile-archives).

`MpmSoilCanonicalTileArchive` separately stores per-tile f64 positions and unchanged
f32 history. Bounded selections can assemble in an explicit solver frame, rejecting
lossy conversions; transactional writeback preserves unloaded records. Its distinct
codec and signed-zero flags retain exact coordinate/history roundtrips.
`MpmSoilCanonicalStore` adds blocking bounded reads and incremental durable
writeback; unrelated tiles do not need rebasing. Its explicit background tasks
also move frame assembly off the caller thread and return owned inputs through
nonblocking polling. `MpmSoilDiskWorld::new_canonical` integrates these workers
with live capture, publication, residency and recovery at an explicit frame.
Its `begin_recenter`/`poll_region_update` transaction translates captured local
soil/grid coordinates exactly, compensates the world offset and reuses identity
and pin buffers without rewriting canonical tiles. It preserves swept limits,
but still assembles host selections and uploads physical state at the boundary.
Fixed-order P2G is available as a runtime opt-in through budgeted scratch preparation
and `set_deterministic_p2g(true)` before installing the solver in the owner. Ordering
and gather resources have separate budgets/lifetimes. This supplies repeatable
accumulation for the recorded streaming fixtures, not whole-machine or performance
acceptance. Ordinary accumulation remains the default.
All active particles remain pinned; physical sleep, machine/renderer rebasing and
physically valid moving windows remain open.
[Canonical archive contracts](../../../crates/molla-solvers/examples/README.md#canonical-per-tile-soil-archives).

## GPU relationship

### Experimental native MPM grid storage

`MpmSimulation::new_native_explicit` retains dense storage by default and accepts
`set_active_grid(true)` for active-tile dispatch. `new_native_sparse(engine,
config, tile_capacity)` uses a fixed compact tile pool with mandatory active
dispatch. Both require `SHADER_FLOAT32_ATOMIC` and explicit stepping. Compact
callers must inspect active-grid status before accepting advancement: overflow
freezes particle updates until explicit recovery/growth; prepared records become
stale after growth or dispatch changes. Standalone `PreparedMpmMassCoupling`
supports active/compact sticking-volume projection with physical-slot owners
and world-space contact geometry. Grid overflow rejects machine publication;
growth requires re-preparing the coupling. Standalone unilateral normal and
paired-wall friction also use indexed active/compact candidates. Their lifecycle
and paired feedback components pass. Default-feature Newton also passes a
bit-exact dense/active/compact comparison in 54.72 s including cold setup on a
shared engine; the previous separate-engine timeout remains retained. This is
not method-23 or timestep-performance acceptance. `RigidWorld::attach_mpm_soil`
now accepts active/compact explicit soil with kinematics-only bindings. Call an
explicit `enable_mpm_soil_*_coupling` method before stepping; an unconfigured
active attachment rejects before frame mutation and does not select a fallback.
Runtime budgets use physical/indexed capacity, including candidate sort scratch.
The standalone boundary/mass gate is
`make test-soil-active-mass-coupling`; it is not tractor or timing acceptance.

Device-local frame backup now supports active/compact grids and restores their
indirect dispatch with tile metadata and particle histories. It rejects storage
or configuration changes during an armed frame. This transaction component is
covered by `make test-soil-frame-backup`. Active projected publication propagates
soil faults to the machine; a retained direct guard covers the no-projection
branch. Detach before growing compact storage, then reattach and configure again.

`make test-soil-active-runtime-joint-types RUN_WITH=nixVulkan` checks actual
one-body sticking transactions for portable/native active/compact soil using the
explicit `ExperimentalJointTypes` machine profile. The separate Generic check
remains timed out during construction; Generic is still the default. This is
not full-tractor, unilateral/Newton runtime, arbitrary disabled-joint, physical
calibration or 4 ms/60 FPS acceptance.
[Scope and retained failure](../../../benchmarks/soil/2026-10-12-active-runtime-joint-types/SUMMARY.md).

`make test-soil-compact-tractor-runtime RUN_WITH=nixVulkan` extends friction
verification to the actual six-body tractor and tine using native compact soil
and the same opt-in scalar backend. The 65-particle fixture passes impulse,
momentum/energy, rollback, late-particle faults, capacity rejection and lifecycle
checks in 69.02 s including setup. It exercises tine contact only, not loaded
wheels or the 50k real-time target; render-buffer checks are not a rendered-rut
capture. Existing `SoilTractor` constructors still select dense storage by default.
[Compact articulated evidence](../../../benchmarks/soil/2026-10-12-compact-tractor-runtime/SUMMARY.md).

For explicit layout selection, use `MpmGridStorage` from `molla::compute::mpm`
with `SoilTractor::new_with_grid_storage` or the native
`new_native_explicit_with_grid_options` constructor. Paired wall-friction and
Newton constructors also expose grid-aware variants. Scalar active/compact
callers must enable projected coupling before stepping. Compact capacity is
validated before allocation; `RigidWorld::mpm_soil_grid_storage()` reports live
layout and physical-grid/metadata bytes without readback, excluding other owners.

The tractor benchmark accepts `SOIL_TRACTOR_BENCH_GRID=compact:1024` through Make
or `--soil-grid compact:1024` directly; dense remains default. Active/compact
selections require explicit coupling and manifest both requested and live layout.
`make test-soil-tractor-grid-selection RUN_WITH=nixVulkan` covers factory seed/
history identity, bounds, growth and CLI selection. These checks do not establish
full 50k compact stepping or paired/Newton runtime performance.

The default 50k seed initially occupies 320 tiles without rigid bounds and 544
with wheel/tine support. A 512-tile pool is valid configuration but insufficient
for this fixture: it faults without completing a gather, preserves positions/
velocities and refuses checkpoint capture. The 1024-tile initial-support control
passes. `make test-soil-tractor-grid-support RUN_WITH=nixVulkan` verifies these
six cases; initial support does not prove sufficient capacity over a trajectory.

The actual 1024-tile, 50k-particle public tractor path completes 33 frames with
Generic paired machine execution and opt-in `retained-local-newton`. The measured
30 frames allocate no application GPU buffers after warmup, but median wall time
is 566.731 ms and all miss 60 Hz. This is short-run integration, not 4 ms, rendered
rut, calibrated traction or dense/compact trajectory-parity acceptance.
[Actual workload and capacity regression](../../../benchmarks/soil/2026-10-12-compact-50k/SUMMARY.md).

The `molla-solvers` feature `conditioned-projection-cache-newton` extends
`retained-local-newton` with cached contact/row projections during conditioned
core assembly. It reuses inactive Krylov scratch and retains the original loop
when the DOF width exceeds the iteration capacity. Defaults, storage budgets and
numerical tolerances are unchanged. Failure captures identify this backend with
method tag 24. `RigidWorld::mpm_soil_newton_backend()` reports the prepared backend
shared by both coupling parities; non-Newton coupling returns `None`.
The benchmark checks this live identity against its requested configuration.
Paired component evidence shows 3.37–5.43% lower solver time on five captured
packets, with three near-neutral results. This is not a whole-frame speedup or
real-time acceptance claim.
[Component evidence](../../../benchmarks/soil/2026-10-12-newton-conditioned-cache/SUMMARY.md),
[runtime verification](../../../benchmarks/soil/2026-10-12-conditioned-cache-runtime/SUMMARY.md).

`shared-conditioned-core-newton` extends that opt-in selection with one common
matrix per release and paired subtraction of each chosen contact's contribution.
It replaces, rather than stacks with, the projection-cache transform because
both reuse Krylov scratch. Method tag 25 identifies it. Insufficient scratch
capacity or a nonfinite common matrix retains the original assembly path.
The accumulation order differs, so numerical tolerances apply rather than a
claim of cross-backend bitwise equality. The prior paired component improves
five captured solve times by 4.28–8.43% relative to the projection cache, while
three are near-neutral. Full-frame and physical acceptance remain separate.
[Shared-core runtime evidence](../../../benchmarks/soil/2026-10-12-shared-core-runtime/SUMMARY.md).

`stream-block-newton` adds direct Jacobian stores to the shared-core backend,
identified by method tag 26. It preserves the lazy-Jacobian gate, zero entries,
derivative arithmetic and fail-closed clearing. Current and legacy captured
solves preserve all public bytes, with paired reductions of 2.18–5.10% across
16 inputs. Runtime defaults remain unchanged; these component measurements do
not establish the full frame budget.
[Streaming component evidence](../../../benchmarks/soil/2026-10-12-newton-stream-block/SUMMARY.md),
[runtime verification](../../../benchmarks/soil/2026-10-12-stream-block-runtime/SUMMARY.md).

`SOIL_TRACTOR_BENCH_PROFILE=1` validates dense versus active stage sequences and
records active/compact layout flags. The compact workload averages 314.369 ms
MPM GPU work and 273.411 ms machine GPU work per 1/60 simulated second. Contact
projection accounts for 306.937 ms of MPM, with 230.991 ms nested Newton solving.
These are diagnostic GPU timings, not a soil-only wall budget; nested contact
stages and host completion waits must not be added again.
[Full-workload attribution](../../../benchmarks/soil/2026-10-12-compact-profile/SUMMARY.md).

The `molla-solvers` feature `zero-term-newton-preparation` opts into finite-guarded
zero-coefficient elision during response preparation. Defaults remain unchanged;
the benchmark records `soil_newton_preparation_backend` explicitly. Exact/oracle
component and fault/recovery checks pass, with about 14.8% lower paired component
time on the sparse 251-contact fixture. Its actual compact tractor run completes
33 frames but still measures 580.194 ms median wall time. This is not a 14.8%
whole-frame improvement or a real-time acceptance result.
[Opt-in scope and evidence](../../../benchmarks/soil/2026-10-12-zero-terms-runtime/SUMMARY.md).

`zero-residual-newton-preparation` additionally elides zero terms in residual
calculations without changing the certificate or its scale. Its distinct backend
label is `compensated-zero-term-and-residual-elision`. The sparse component is
about 26.5% faster than the earlier zero-term control; exact/oracle, fault/recovery
and dense/active/compact retained-local feedback checks pass. Its separately
identified actual compact 50k tractor run completes 33 frames with zero hot GPU
allocations, but measures 581.432 ms median wall time and misses every measured
60 Hz deadline. This is integration evidence, not a causal whole-frame speedup
or real-time acceptance. The feature remains opt-in and physically uncalibrated.
[Residual-elision evidence](../../../benchmarks/soil/2026-10-12-sparse-residual-runtime/SUMMARY.md).

The optional `molla-solvers/lazy-paired-drift` feature selects lazy anchor
calculation only for storage-rounded unrestricted paired machine drift. Defaults and other
machine phases remain unchanged. Split paired models still reject newly imposed
finite limits before stepping; unsplit models retain their generic fallback.
The separate pipeline adds cold compilation work. Public constructor analytic,
corrected-drift and fault/recovery checks pass. Earlier paired component timing
shows about 9.8% lower drift-pass median, not a whole-frame improvement. Its actual
50k run completes 33 frames with zero hot GPU allocations but still measures
590.422 ms median wall time, missing every measured 60 Hz deadline. Benchmark
`machine_paired_drift_build_variant` reports build selection, not the phase last
dispatched. Kernel `unrestricted_drift_backend()` reports its retained pipeline.
[Lazy drift runtime evidence](../../../benchmarks/soil/2026-10-12-lazy-drift-runtime/SUMMARY.md).

The optional `molla-solvers/native-geometry-drift` feature replaces only geometric
paired sums in the unrestricted storage-rounded drift pipeline. Request
`FeatherstoneKernel::paired_drift_required_features()` when creating the device;
native geometry requires enabled `SHADER_F64`, otherwise construction returns an
error. Native geometry takes precedence if only those two drift features are
enabled; they are not implicitly composed. Other machine phases and default builds remain unchanged.
Dedicated exact/f64 and public force/lifecycle checks pass, but a two-slot cold
construction exceeded the 90 s cap. Production capacity is not padded. The
measured 24% six-body drift-pass improvement is not coupled whole-frame acceptance.
Its actual compact-1024 50k diagnostic completes 33 frames with zero hot GPU
allocations, 554.870 ms median frame time and 159.477 s construction. Every
measured frame misses 60 Hz; full-frame/profile and build-identity audits pass,
but causal speedup and controlled trajectory parity remain unverified.
[Native geometry runtime evidence](../../../benchmarks/soil/2026-10-12-native-geometry-runtime/SUMMARY.md).

The explicit `molla-solvers/native-lazy-geometry-drift` feature composes native
geometry with lazy anchor construction in the same eligible unrestricted,
storage-rounded paired drift path. It retains f64 device admission and leaves
other machine phases unchanged. `gpu_unrestricted_drift_backend()` on the solver
reports the prepared backend, and benchmark
`live_machine_unrestricted_drift_backend` records it separately from build
selection. Ten exact moving-state checkpoints and 48 f64 reconstruction/lifecycle
cases pass; the paired drift-component median is 8.98% lower than native geometry
alone. Public runtime and coupled tractor physical, profile and rollback/retry
checks pass. This component improvement is not whole-frame real-time acceptance.
The actual 50k diagnostic completes 33 frames but takes 388.416 ms median per
frame and misses every measured 60 Hz deadline. Separate-run trajectory
differences exceed the existing 2e-5 bound; matched-input full trajectory parity
is not established. Keep this backend experimental and physically uncalibrated.
[Composed runtime evidence](../../../benchmarks/soil/2026-10-12-native-lazy-runtime/SUMMARY.md).

These native active modes are **experimental, not performance-accepted**.
Overflow/recovery, bounded storage and continuous transport pass, but the
mixed-material rigid-force parity gate exceeds its unchanged 2e-5 bound. Use
`make test-soil-native-active` for the known-failing gate; passing individual
storage/lifecycle checks does not clear it.
[Evidence and current limitation](../../../benchmarks/soil/2026-10-11-native-active/SUMMARY.md).

A test-only single-pass prototype corrects transfer residuals before contact and
passes the focused cross-layout force bound. It additionally requires SHADER_F64;
it is not a public runtime option or proof that the ordinary failing gate is fixed.
Whole-step performance and broader physical integration remain unverified.
[Pre-contact compensation evidence](../../../benchmarks/soil/2026-10-12-native-precontact-compensation/SUMMARY.md).

### Vehicle kernel relationship

The crate itself is host-side; its GPU twins live in `molla-kernels`
(`tire_brush.wgsl`, `tire_fem*.wgsl`, bekker/DEM/thermal/transient tire
kernels) and are validated by the `tire_*` hardware lanes:

```bash
make test-gpu P=molla-kernels T=tire_fem_gpu
make test-gpu P=molla-kernels T=tire_brush_gpu
```

The CPU↔GPU status of each tire lane is recorded in
`real_tire/parity_ledger.rs` and the `TireFemTorqueGpuStatus` /
`TireFemMatrixFreeNewtonGpuStatus` contracts in `molla-kernels`.

## Retained DEM refitting

`TireDemResidentFemTerrainOwner` defaults to rebuilding its Morton ordering every
step. `set_lbvh_rebuild_interval(32)` explicitly reuses that ordering between
rebuilds while refitting every tree bound before contact queries. Refit steps
accumulate forces in current Morton/particle-ID order using complete 32-contact
batches; overflow triggers another batch, not dropped contacts. The ordered
pipeline is prepared lazily and retained with the owner.

Explicit refits update bounds using workgroup-local subtrees of up to 64 leaves
and atomic ancestor expansion. At 50k particles with eight grains per leaf, this
uses two tree dispatches instead of fourteen, without additional tree buffers.
An incremental production comparison reports about 8% lower paired median time
than per-level refitting, but one retained tail spike regresses. Neither this
change nor the earlier force-order repair achieves the real-time target.
[Grouped bounds evidence](../../../benchmarks/soil/2026-10-13-dem-grouped-runtime/SUMMARY.md).

`make test-soil-dem-refit RUN_WITH=nixVulkan` runs exact grouped tree/trajectory
and ordered overflow checks alongside the bounded public topology,
lifecycle, resource and original 50k benchmark checks. Two earlier, per-level RTX 4080 packets show
about 41–44% lower paired solver/export time than retained full rebuilding, with
unchanged particle/terrain gates. The resulting median still normalizes to about
31 ms per simulated 60 Hz frame, not the 4 ms target. These are soil-component
results, not tractor/render or physically calibrated acceptance.
[Evidence and limits](../../../benchmarks/soil/2026-10-13-dem-ordered-public/SUMMARY.md).

## Delayed frame-integrated load observations

After `prepare_mpm_soil_frame_loads`, a committed world can call
`try_request_mpm_soil_frame_loads` and `poll_mpm_soil_frame_loads`. The corresponding
tractor methods omit `mpm_`. Request returns false while the capacity-one staging
slot is occupied; poll returns None until completion. Neither explicitly waits
for the GPU. The immutable observation exposes captured frame, simulated interval
and body impulses through `loads()`. Divide impulse by that captured interval,
not the current frame's timestep, to obtain the corresponding mean force.

`is_current_mpm_soil_frame_load_observation` checks owner/generation, body model
association, attachment fault state and known device health; it does not assert
that a delayed sample belongs to the latest frame. Label delayed force samples
with their captured frame. `reset_mpm_soil_frame_load_observations` invalidates
old samples without reusing a pending stage before mapping completes. Continue
polling to discard that obsolete completion before requesting again. Revisions
must increase within a generation; reset permits a same-frame recapture.

The existing blocking read API remains available but rejects an occupied async
stage. Model replacement requires re-preparation. Known map/poll failures latch
on the prepared observer. No additional GPU allocation is needed beyond the
existing frame-load budget. `make test-soil-frame-loads-gpu` exercises the bounded
component path; it is also part of the S03 GPU lane. The live viewer uses this
API for normal force observations and idle completions. Missing samples are
labelled pending rather than zero force. Force frame/age/interval are separate
from the geometry revision and physical clock. Final capture endpoints explicitly
wait for matching loads; normal display updates do not.

## Delayed geometric surface observations

`SoilTractor::try_request_soil_surface` queues an optional point-height export
tagged with the current committed frame. Call it once after a new committed frame;
`false` means the prepared surface's single retained staging slot is still busy.
`poll_soil_surface` makes nonblocking device progress and returns an optional
`SoilSurfaceObservation`. Its `revision()` identifies the captured frame, not
necessarily the latest physics frame. Display the revision when showing delayed
geometry; do not use it as current collision/contact data.

`PreparedSoilSurface::reset_snapshots` invalidates old observations without waiting
or prematurely reclaiming pending storage. Poll to retire obsolete completions.
Reset explicitly after same-count reseeding; reprepare after source/count/device
changes. `is_current_snapshot` checks owner/generation and observed map errors,
not equality with the newest physical frame or external device health. The MPM
and tractor polling wrappers also check their live engine/source.

This API retains the existing 24-byte-per-cell staging allocation. Blocking
observation methods remain available but reject simultaneous pending async reads.
The Bevy viewer uses this API for normal geometry observations, retaining explicit
waits for bootstrap/endpoints; force observations use their separate delayed API. Direct
buffer sharing still spans incompatible wgpu versions.
[Checks and limits](../../../benchmarks/soil/2026-10-14-surface-observation/SUMMARY.md).

## Persistent GPU soil regions

`molla::compute::MpmSoilWorld` owns a GPU MPM solver and a host tile archive.
`MpmSoilDiskWorld` instead uses selected reads and background writes through an
exclusive `MpmSoilStore`; it does not retain the entire field's particle payload
in memory. Both preserve particle/material IDs and deformation history. Their
residency metadata lives in `molla::vehicle::SoilTileResidency`, not in the
dependency-free crate's CPU soil models.

Use `begin_swept_region_update` with `MpmSoilRegionEnvelope` to turn closed swept
machine/tool bounds into archive-tile requests. Include rotation and contact
margins in those bounds. Supply an upper bound on every soil particle's
componentwise displacement per substep, including acceleration, and the number
of substeps until the next refresh. Request expansion adds three MPM cell widths
per substep, plus one cell per configured Jacobi sweep and the supplied particle
travel. Archive tile size is independent of solver cell size.

The bounds include an outward f32 arithmetic allowance; an ideal three-cell
radius rounded only in f64 can miss a neighbor used by the GPU stencil. Swept
requests reject non-normal cell spacing/reciprocals, non-finite grid origins,
horizons of 131,072 substeps or more, and unsupported coordinate precision ranges.
Roundoff padding can increase the tile count; capacity failures remain explicit
rather than silently reducing coverage. This does not implement large-world rebasing.

Poll `poll_region_update` from the application update loop; stepping remains
frozen until the transition finishes. After publication,
`remaining_region_substeps()` reports the shortest committed envelope horizon.
`step_n` rejects batches that exceed it before submitting GPU work. Refreshing
resets the horizon only after a successful transition. Motion bounds remain the
caller's responsibility. Raw `begin_region_update` tile requests retain their
manual-coverage contract and remove the horizon guard only when committed.

Every already-active particle still stays pinned, even outside machine requests.
Envelope expansion is **not physical sleep or a bounded-traversal guarantee**:
equilibrium, support transmission, wake propagation, nonlocal machine coupling
and long-traverse acceptance remain unfinished. Disk publication and GPU selection
publication are separate; a load failure after a successful write can leave the
disk generation ahead of the faulted device owner.

The RTX 4080 / 50,000-active-particle target is not yet performance-accepted.
Current presets are physically uncalibrated. See `SOIL_AND_PARTICLES.md` and
`SOIL_AND_PARTICLES_REPORT.md` at the repository root for the contract and evidence.

## Experimental fixed-boundary contact acceleration

`molla::solvers::soil_contact_projection_gpu::SoilContactProjectionGpu` provides
`new_with_fixed_boundaries_cached` alongside the unchanged reference constructor.
It caches up to 128 boundary-bearing nodes in deterministic order for the
active-set solve, using 2048 bytes of workgroup storage. Overflow takes the full
original GPU scan; it never drops constraints. Device-buffer ownership, dispatch
count and numerical tolerances are unchanged.

The cache has focused exact-output, independent QP and overflow/reuse evidence,
plus a paired saved-input timing improvement. Select it explicitly through
`SoilTractor::enable_mass_coupling_with_boundary_scan` or
`RigidWorld::enable_mpm_soil_mass_coupling_with_boundary_scan`, passing
`SoilBoundaryScan::ExperimentalCached`. The original methods select `Full`.
The policy remains fixed until soil is detached, survives projection preparation,
and is reported by `RigidWorld::mpm_soil_boundary_scan`.

`make bench-soil-tractor SOIL_TRACTOR_BENCH_COUPLING_NODES=4096
SOIL_TRACTOR_BENCH_BOUNDARY_SCAN=cached` selects the option in the actual tractor
diagnostic. Its artifacts distinguish requested and runtime-selected policy.
The cached path is not accepted against the complete simulator frame budget.

## Viewer scenes

Scenes 103 (flat-ground car), 110 (offroad heightfield), 118–122 (Bekker
sinkage, soil compaction, tire FEM dynamics/traction/test-bench), 123
(terrain lab), and 124 (sysid friction recovery) exercise the crate
interactively.

## Testing

The crate is CPU-testable with plain `cargo test -p molla-vehicle`.
The suite also hosts the tire validation harness (`real_tire/rig`,
`real_tire/reports`) — structural truth gates (loaded radius, footprint,
pressure distribution, bead drift) and force-shape checks that guard the
tire physics without external measured-data calibration (that scope is
deliberately excluded — see the "Calibration stance" section of
`PLAN_REALTIRE.md`).


## Fully autonomous GPU substep loop (F2)

The tire FEM substep loop runs with zero per-substep host values:
`step_persistent → dispatch_resident(resident-spin) →
step_hub_persistent_with_tread_forces_spinning`. The wheel spin DOF lives
in `hub_state[1].zw` (integrated on-device with the exact
`integrate_spin_dof` formula, soft-sign brake saturator included), and the
feedback shader reads ω from the hub rows. Proven bit-identical to a
host-bridged reference over 96 rolling substeps.

The fused-atlas path scales this to fleets: one dispatch runs the bristle
physics for every tire (`step_resident_atlas_feedback`), the hub-param
bridge folds per-node force rows into each tire's `extf`, and
`hub_reduce_atlas` integrates all hubs (spin DOF included). Measured:
**1.53× realtime for 40 wheels** running the complete loop. A shared fleet
heightfield (`TireFemAtlasTerrain`) adds bilinear terrain sampling — an
all-zero heightfield is bitwise identical to the flat path, and an incline
makes the wheel drift downhill (the tilted contact normal redirects the
bristle reaction, which a flat-ground model cannot produce).

## Recorded soil/tractor inspection

### Cooperative contact-response preparation

Enable the opt-in Cargo feature `row-parallel-newton-preparation` on `molla` or
`molla-solvers` to use cooperative Cholesky columns and response rows in Newton
contact preparation.
The public preparation constructor reports
`compensated-row-parallel-cooperative-factor`. Existing defaults are unchanged;
if `strict-rounding-newton-preparation` is also enabled on the solver crate, its
strict axis-parallel implementation takes precedence.

This path requires 128 workgroup invocations and 2048 bytes of supported workgroup
storage, checked before allocation. Persistent buffer budgets, mandatory
refinements, component error bounds and fault clearing are unchanged. Eight
retained tractor inputs show exact factors/response/status outputs. Row cooperation
reduced paired preparation time by 51--57%; a separate comparison measured another
9--11% reduction from cooperative factors. These are component comparisons, not a
combined full-frame speedup. Coupled publication checks pass. Full tractor trajectory
and real-time acceptance have not been established for this option.
[Evidence and limits](../../../benchmarks/soil/2026-10-15-preparation-factor/runtime/SUMMARY.md).

For the viewer, use `make view-soil-tractor SOIL_LIVE_PROFILE=soa
SOIL_LIVE_PREDICTION=native SOIL_LIVE_TRANSFER=fixed-order
SOIL_LIVE_PREPARATION=cooperative` (one shell command). The independent preparation
selection defaults to `default`; binary expectations reject mismatched builds.
The overlay and ready record identify the preparation profile and compiled backend.
A 33-frame rendered 50k run exactly matches the retained native/default-preparation
control's poses and particle endpoints. This is not full-state or long-horizon
parity: whole-physics median remains 1901.649 ms and all measured frames miss 60 Hz.
[Rendered capture and scope](../../../benchmarks/soil/2026-10-15-viewer-cooperative-preparation/SUMMARY.md).

Add `SOIL_LIVE_SUBMISSIONS=composed` to opt into combined projected guard/solve/
publication submissions. `separate` remains default; this setting is independent
of the profiles above. The worker verifies `RigidWorld::mpm_soil_composes_submissions`
against its build before Ready, and the overlay/capture records the selection.
The count applies to that projected sequence, not every queue submission per frame.
A 33-frame rendered 50k native/SoA/cooperative/fixed-order capture preserves all
recorded six-body poses and both particle endpoints exactly versus the retained
separate control. Whole-physics median/p95 is 471.583/671.800 ms, still far from
60 FPS. The historical comparison is not causal speed evidence or full-state
trajectory acceptance; defaults and physical calibration status are unchanged.
[Composed viewer evidence](../../../benchmarks/soil/2026-10-15-viewer-composed-submissions/SUMMARY.md).

For bounded diagnostic captures, add `SOIL_LIVE_STAGE_PROFILE=1`:

```sh
make capture-soil-tractor-live SOIL_LIVE_OUTPUT=/tmp/soil-stage-capture \
  SOIL_LIVE_FRAMES=33 SOIL_LIVE_HEADLESS=1 SOIL_LIVE_STAGE_PROFILE=1 \
  SOIL_LIVE_PROFILE=soa SOIL_LIVE_PREDICTION=native \
  SOIL_LIVE_TRANSFER=fixed-order SOIL_LIVE_PREPARATION=cooperative \
  SOIL_LIVE_SUBMISSIONS=composed
```

The output directory must not exist. Each committed physical frame writes a
`stage_profile` record to `capture/frames.jsonl` before its ordinary frame record.
MPM top-level, nested contact, nested preparation, machine GPU and host durations
remain separate. Contact/preparation times are already included in their parent
intervals; summing all arrays double-counts work. Machine timestamps have a
different origin from MPM timestamps.

Profiling is off by default and rejected for unbounded interactive runs. It
adds retained timestamp resources and **blocking diagnostic reads** after each
committed frame. `profile_read_ms` reports those reads outside `physics_ms` and
`observation_ms`; the instrumented timings are not production-overhead or
real-time acceptance evidence. No solver work, time step or numerical threshold
is reduced by this option.

`SOIL_LIVE_DRIFT=deferred` forwards the existing rates-only native predictor to
the viewer; it requires `SOIL_LIVE_PREDICTION=native`. Native prediction then
produces rates, soil projection corrects them, and one final drift reconstructs
the published poses. The intermediate predicted poses are not computed. `full`
remains the default; non-native prediction is not changed. This option does not
enable the separate native/lazy geometry arithmetic feature.

The worker verifies `mpm_soil_defers_native_prediction_drift()` before readiness.
Capture metadata records both `native_drift_profile` and the effective
`native_drift_deferred` branch. Explicit command-line expectations reject
mismatched binaries, duplicate settings and deferred selection without native
prediction. Stage profiling remains independent and optional.

Add `SOIL_LIVE_GEOMETRY=native-lazy` to select the existing native geometry sums
plus lazy body-anchor reconstruction for eligible paired unrestricted drift.
This is separate from the full/deferred prediction schedule; `eager` geometry
remains default. The device must support native f64, which the viewer requests
explicitly. The worker validates the prepared pipeline identity, recorded as
`prepared_unrestricted_drift_backend`. Prediction, momentum and refresh
arithmetic are not changed by this option.

The fixed-order 33-frame native/deferred viewer capture matches recorded poses
and particle endpoints exactly against eager geometry. Its final drift median
is 53.874 ms and whole physics 358.531 ms, still not real-time; construction is
99.920 s. These are instrumented samples, not a paired whole-frame speedup or
complete physical-state acceptance. The earlier atomic-run discrepancy remains
unresolved.
[Geometry capture and limits](../../../benchmarks/soil/2026-10-15-viewer-native-lazy-geometry/SUMMARY.md).

Optional stage profiles also contain `newton_work`: ordered substep fault flags,
Newton steps, mixed Krylov/condensed linear iterations and trial counts. These
32-byte reports share the existing machine timestamp readback; use the reported
`machine_readback_bytes`, not timestamp count alone, for traffic accounting.
The viewer rejects missing or faulted committed reports. The current 50k capture
shows large trial counts in difficult substeps and remains far from real-time.
[Work-report evidence](../../../benchmarks/soil/2026-10-15-viewer-newton-work/SUMMARY.md).

### Experimental machine execution

`make bench-soil-tractor SOIL_TRACTOR_BENCH_COUPLING_NODES=4096 SOIL_TRACTOR_BENCH_EXECUTION=scratch SOIL_RUN_DIR=<new-run>` selects workgroup scratch for machine dynamics. `generic` remains the default. This is independent of `SOIL_TRACTOR_BENCH_ROUNDING`; existing constructors retain generic execution and storage rounding. Rust callers can select `FeatherstoneExecution::ExperimentalWorkgroupScratch` using the explicit machine-option constructors. Cache identity, device-limit preflight and runtime rebuilds preserve the selection.

The recorded 50k diagnostic lowers median frame time from 248.87 to 159.65 ms, but still misses 60 FPS and fails the strict velocity-parity bound. It also increases cold construction time by compiling another program. Treat this profile as experimental, not accepted physics or real-time performance.

`SOIL_TRACTOR_BENCH_EXECUTION=constraint-scratch` selects `FeatherstoneExecution::ExperimentalConstraintScratch`, which also places articulated-mass arrays in workgroup storage. This is a separate opt-in profile, not a replacement for `scratch`; both use one invocation per articulation, but have distinct cache identities and resource bounds. The six-joint extension requires 5552 bytes of workgroup storage. Its runtime acceptance evidence is tracked in [the extension checkpoint](../../../benchmarks/soil/2026-10-09-runtime-constraint-scratch/SUMMARY.md); component improvements alone do not establish full-workload performance or physical calibration.

The partial/extended 50k diagnostic observes medians of 160.73/155.54 ms, but fails independent-trajectory velocity parity and still misses 60 FPS. Identical-input machine predictions match exactly through 33 diagnostic frames; coupled rollback/rebuild and three-profile cache checks pass. These narrower successes do not waive the failed trajectory gate or justify changing defaults.

### Experimental unilateral normal component

`molla::solvers::soil_normal_projection_gpu::ExperimentalSoilNormalProjectionGpu` prepares a resident finite-mass, gap-aware normal solve for one machine island. `SoilNormalInputs` supplies mass, free rates, normal Jacobian rows and `SoilNormalContact` records; `SoilNormalCapacity` bounds DOFs, contact slots, iterations and owned bytes. Encode into a caller-owned command encoder, then require zero fault status before consuming rates, outward impulses or corrected soil normal velocities. The caller owns publication and rollback.

This is not a tractor benchmark/backend option. It has no geometry generation, friction, penetration repair or fixed-boundary integration. Contact rows must represent distinct scalar soil normal DOFs; shared/nonorthogonal soil constraints cannot be treated as independent rows. Existing sticking coupling is unchanged. [Verified scope and remaining integration](../../../benchmarks/soil/2026-10-09-unilateral-soil-normal/SUMMARY.md).

`molla::solvers::soil_floor_projection_gpu::ExperimentalSoilFloorProjectionGpu` adds an experimental one-normal-per-node block with fixed axis walls. Preparation and recovery preserve the shared physical 3D velocity and distinguish contact from external floor impulses. Fully blocked normals impose exact affine machine inequalities; the bounded working-set solve reports faults for nonconvergence or invalid systems instead of substituting a large mass. Five GPU suites pass independent full-space QP and physical/fault/resource checks, including redundant rows and a 32-DOF corner. The fourth status word packs working-set additions in its low u16 and removals in its high u16. This is not an integrated tractor backend or calibrated model; friction, geometry and runtime publication/rollback remain open. [Supported scope and evidence](../../../benchmarks/soil/2026-10-09-soil-hard-rows/SUMMARY.md).

### Resident contact geometry

`molla::solvers::soil_contact_geometry_gpu::ExperimentalSoilContactGeometryGpu` fills floor-solver nodes/rows from explicit unique node/body queries using current analytic shape and COM Jacobian buffers. Construct the floor solver on caller-owned node/row buffers, then construct geometry with those buffers and `floor.status_buffer()`. Clear status, encode geometry, encode floor, submit, and require zero status before consuming outputs. Signed distances are available separately; negative gaps do not inject penetration-repair energy. Candidate enumeration, multiple body contacts per node and runtime publication remain external. [Geometry contract and evidence](../../../benchmarks/soil/2026-10-09-soil-contact-geometry/SUMMARY.md).

### Resident candidates and coverage

`molla::solvers::soil_candidates_gpu::ExperimentalSoilCandidatesGpu` fills geometry query buffers from a dense free grid in deterministic index order. Encode `encode_candidates`, geometry, floor solve, then `encode_certify_and_publish_grid` into the same submission. The final status must be zero before machine publication: the certificate checks every occupied node/body inequality using corrected rates, so an earlier successful floor solve alone is insufficient. Coverage failure requires a rebuilt/wider candidate solve; multiple-body nodes are unsupported rather than assigned to the first body. Configuration and all input geometry remain fixed across these stages. [Limits and evidence](../../../benchmarks/soil/2026-10-09-soil-candidates/SUMMARY.md).

### Experimental unilateral runtime

The normal solve retains low-order rate increments and uses compensated contact arithmetic. Success also requires the momentum balance of the rounded output machine rates and solved impulses to satisfy the numerical bound. Bounded nonconvergence remains an explicit fault; it does not increase the iteration cap or relax tolerances. A captured actual-workload failure and independent f64 replay exercise this path. [Numerical scope and evidence](../../../benchmarks/soil/2026-10-09-normal-replay/SUMMARY.md).

After attaching dense explicit MPM soil, select `RigidWorld::enable_mpm_soil_unilateral_coupling(MpmUnilateralConfig { node_capacity, search_distance, max_iterations })`, or call `SoilTractor::enable_unilateral_coupling`. The configuration is under `molla::solvers::mpm`; selection cannot change while attached. `mpm_soil_unilateral_config()` returns the active configuration. Existing mass-coupling APIs retain sticking behavior.

The dense prepared path supports grids from 5 to 128 cells per side; indexed active/compact candidates instead budget checked tile capacity within the MPM world's valid dimensions. Both support 1–32 machine DOFs, 1–64 bound bodies, up to 16,384 candidate nodes and 1–64 solver iterations. The search distance is a finite nonnegative world-space margin, not a guarantee of swept collision coverage. Uncovered corrected normal motion, multiple candidate bodies per node, capacity overflow and solver failure fault the frame rather than silently dropping constraints. Corrected machine rates and MPM particle dispatch require final zero status; normal runtime rollback and recovery apply. Reported body wrenches contain contact reactions only, not external fixed-floor reactions.

The actual six-body/65-particle fixture passes conservation, reaction parity, changed timesteps, rollback/resume, budget/overflow and empty-soil checks; steady frames allocate zero GPU buffers. This mode is frictionless and does not repair existing penetration. It remains physically uncalibrated and has not met the full 50k/4 ms workload target. [Integrated evidence](../../../benchmarks/soil/2026-10-09-unilateral-runtime/SUMMARY.md).

Select the actual 50k diagnostic with `make bench-soil-tractor SOIL_TRACTOR_BENCH_UNILATERAL_NODES=4096 SOIL_TRACTOR_BENCH_SECONDS=600`. Optional `SOIL_TRACTOR_BENCH_UNILATERAL_MARGIN` (default 0.05 m) and `SOIL_TRACTOR_BENCH_UNILATERAL_ITERATIONS` (default 64) control the search/solve bounds. The corresponding executable flags are `--unilateral-nodes`, `--unilateral-margin` and `--unilateral-iterations`. Capacity counts total candidate nodes, not nodes per body. This is mutually exclusive with nonzero sticking `SOIL_TRACTOR_BENCH_COUPLING_NODES` and cached boundary scanning. Requested and live configurations are recorded and checked; existing invocations remain unchanged. [Diagnostic evidence](../../../benchmarks/soil/2026-10-09-unilateral-tractor/SUMMARY.md).

### Live coupled tractor fixture

`make view-soil-tractor` launches the opt-in `soil_tractor_viewer` binary. It runs
the actual 50k-particle dense soil owner, protected Newton contact,
four driven wheels and one fixed tine. The existing default viewer is unchanged.
Native float atomics and the protected Vulkan route are required; unsupported
devices reject rather than switch to a kinematic display or a different solver.

`SOIL_LIVE_PROFILE=volatile` remains the default. Select `SOIL_LIVE_PROFILE=soa`
to build the opt-in `soil-live-soa` viewer using the existing optimized Newton
profile stack through coefficient-major scratch. This selects the complete
profile, not an isolated scratch-layout experiment. The worker rejects a
constructed backend that differs from the compiled profile. Make also passes
`--expect-profile` so a stale executable of the other profile rejects before GPU
setup. Ready records contain both the profile and actual backend name; the overlay
identifies the profile. This selection does not change physical parameters or
relax numerical bounds, and is not a default promotion or real-time acceptance.

`SOIL_LIVE_TRANSFER=atomic` remains the default. Select `fixed-order` (binary flag
`--soil-transfer fixed-order`) to use stable particle ordering and fixed-order P2G
with 3,200,000 bytes of gather scratch, plus separate ordering storage. The worker
checks live mode flags and storage before Ready; capture records include those
values and the overlay identifies the transfer. This controls upstream input
order for backend comparisons; it is not a speedup or calibrated accuracy claim.
Neither option changes the paired machine algorithm, timestep or contact limits.

`SOIL_LIVE_PREDICTION=featherstone` is the machine-prediction default. Select
`native` (binary flag `--machine-prediction native`) for the experimental native
control/prediction and shared-mass path. It requires f64 device support and retains
the same timestep, contact limits and explicit P2G selection. The worker verifies
actual native-prediction ownership before Ready; metadata and overlay identify it.
Both machine modes complete the recorded 33-frame rendered fixture, but their
cross-prediction trajectory comparison fails the unchanged 2e-5 bound (final
particle delta 1.113534e-3 m). Treat native prediction as a diagnostic opt-in, not
an interchangeable replacement. Concurrent GPU applications prevent a controlled
speed conclusion from those captures; neither establishes real-time acceptance.
[Machine-prediction comparison](../../../benchmarks/soil/2026-10-15-viewer-native-prediction/SUMMARY.md).

```sh
make view-soil-tractor SOIL_LIVE_PROFILE=soa SOIL_LIVE_TRANSFER=fixed-order SOIL_LIVE_PREDICTION=native
```

Earlier atomic and fixed-order 33-frame volatile/SoA comparisons failed the
cross-profile pose/particle bound. After gap-independent certification and a
dense-inversion fallback for coupled Schur blocks, a fresh fixed-order comparison
passes the unchanged 2e-5 bound: maximum final particle difference is 4.768372e-7 m.
Endpoint surface/count and delayed-load policy checks pass. Atomic transfer was
not revalidated after these repairs. SoA remains a diagnostic opt-in, not a
universally interchangeable replacement. Both profiles still miss 60 FPS:
sequential physics medians are 856.47 ms volatile and 562.79 ms SoA, not isolated
soil timings or controlled performance acceptance.
[Post-fix bounded comparison](../../../benchmarks/soil/2026-10-15-viewer-certified-schur/SUMMARY.md).
[Fixed-order control evidence](../../../benchmarks/soil/2026-10-15-viewer-fixed-order/SUMMARY.md).
[Rendered profile evidence and retained parity failure](../../../benchmarks/soil/2026-10-14-viewer-soa-profile/SUMMARY.md).

Physics construction/stepping runs on a dedicated worker. Space pauses after the
current physical frame, Enter requests one frame while paused, arrows orbit/zoom,
and Escape closes the viewer. Only committed frames update the display and
simulation clock. Shutdown joins the physics worker before renderer teardown.
The overlay identifies numerical results as physically uncalibrated and reports
nominal stationary-ground slip and frame-averaged world-space tine force. Force
observations carry their own captured frame/age/interval, and may be pending or
older than the displayed physical frame.
The default view renders a 64×40 geometric particle-centre height envelope from
a bounded 61,440-byte terrain bridge per delivered sample. Normal frames reuse
the latest completed surface and label its captured frame/generation and age.
Unchanged revisions do not rebuild the mesh. Empty cells do not
create supported triangles; outside particles are counted explicitly. This
display surface is separate from the material contact/compaction law.
Pending surface observations continue polling while awaiting commands, including
while paused. Separate surface events update only geometry and its age label,
not the simulation clock, body poses or step scheduling. Pending work is polled
every 5 ms when idle; a ready command takes priority. With no pending observation
the worker blocks on the command channel without periodic polling.
Use `make view-soil-tractor SOIL_LIVE_PARTICLES=1` for full-particle glyph mode.

```sh
make test-soil-live-viewer
make test-soil-live-viewer SOIL_LIVE_PROFILE=soa
make test-soil-live-idle-surface-gpu
make view-soil-tractor SOIL_LIVE_PROFILE=soa
make view-soil-tractor SOIL_LIVE_PROFILE=soa SOIL_LIVE_TRANSFER=fixed-order
make capture-soil-tractor-live SOIL_LIVE_OUTPUT=benchmarks/soil/new-live-capture SOIL_LIVE_FRAMES=33
make capture-soil-tractor-live SOIL_LIVE_HEADLESS=1 SOIL_LIVE_OUTPUT=benchmarks/soil/new-offscreen-capture SOIL_LIVE_FRAMES=4
```

Capture precompiles separately, then imposes a fixed 180-second execution limit
including GPU setup. A fresh output directory contains source/hash provenance,
hardware telemetry, exit status, JSONL committed states, initial/final particle
positions and surface maps, `capture/committed-frame.png`, and a delayed-frame image
when stale geometry is displayed. Headless mode renders the same scene into a Bevy
image without a window; it requires a bounded capture. Default captures
download full particles only at the two endpoints; optional glyph mode downloads
them each frame. Raster and load observation occur outside reported physics
timing. Normal surface request/poll does not wait; bootstrap and final captures
explicitly wait for matching geometry. Idle completions are logged separately as
`surface` and `loads` events rather than additional committed frames. Normal load
request/poll also does not wait; the final capture requires matching force data.
Coupled physics still has synchronization, and explicit particle diagnostics still
block: this is not fully resident display or a performance acceptance run.
Optional glyph size is visual, not the physical
material-point volume. Field-length traversal, repeat-pass ruts and 4 ms / 60 FPS
acceptance remain open.

The current offscreen capture passes independent endpoint checks. Wayland/Xvfb
presentation failures and incomplete overlay glyphs remain unresolved.
[Asynchronous consumer and headless evidence](../../../benchmarks/soil/2026-10-14-viewer-headless-surface/SUMMARY.md).

The force-consumer migration passes actual-worker pause/resume and headless
capture checks. Its five soil bindings transfer 240 bytes per completed force
sample, independent of the six rendered machine bodies.
[Delayed force and endpoint evidence](../../../benchmarks/soil/2026-10-14-viewer-async-loads/SUMMARY.md).

[Geometric surface and endpoint evidence](../../../benchmarks/soil/2026-10-11-geometric-surface/SUMMARY.md).

[Live implementation and rendered evidence](../../../benchmarks/soil/2026-10-11-live-tractor-viewer/SUMMARY.md).

### Same-device resident particle rendering

`MpmSimulation::particle_render_view(&engine)` borrows live positions without
allocation, upload, submission or readback. `SoilTractor::soil_particle_render_view()`
and `RigidWorld::mpm_soil_particle_render_view()` additionally return the committed
frame ordinal and reject faulted runtime state. Positions use read-only WGSL
`array<vec4<f32>>`, xyz in world metres, with a 16-byte stride; draw only the
returned particle count. Empty views require a zero-instance draw.

Create the renderer on the simulation's wgpu 29 device and submit on its queue
after physics. Retain `view.source()` alongside a bind group and check
`view.matches_source(&source)` before reuse; rebuild bindings on mismatch and
refresh the live count for every draw. The token identifies the buffer owner,
not physical time, count or a frozen frame. Rendering must not mutate the source.
The raw simulation view does not wait for or certify queued numerical success.

`make test-soil-particle-render RUN_WITH=nixVulkan` precompiles and runs the
90-second-bounded rendered fixture. It uses one retained position binding across
MPM stepping, empty reseed and a 50,000-particle reseed. Only final RGBA images
are mapped; the renderer does not download particle positions. Set
`MOLLA_PARTICLE_RENDER_ARTIFACTS` to an existing empty directory to retain PPM images.

The existing Bevy 0.18 viewer uses wgpu 27, not the solver's wgpu 29 device.
This interface does not make those handles interchangeable or remove that
viewer's existing CPU bridge. The fixture's screen-aligned markers are not a
terrain mesh, physical grain radius, live tractor render or 60 FPS acceptance.

### Same-device machine transforms

`RigidWorld::gpu_body_render_view()` borrows the two internal transform buffers,
the active output side, body count and committed world-frame ordinal. WGSL uses
32-byte records: position `vec4<f32>`, then xyzw quaternion `vec4<f32>`. Bodies are
indexed with `world.scene.body_index(handle)`. The public `scene.state()` is a
host mirror, not the source of GPU render bindings.

Retain a bind group for each buffer, verify both buffer identities before each
draw, and select `view.active_side()`. Rebuild bindings and body-indexed geometry
after model replacement. Queue rendering before the next world mutation; these
buffers are live, not frozen snapshots. Scene edits (including controls), device
faults and failed steps invalidate access until a successful step republishes
state. A recoverable soil rollback does not certify GPU machine pose rollback.
This accessor performs no upload, submission or readback.

`make render-soil-tractor-resident RUN_WITH=nixVulkan SOIL_RESIDENT_OUTPUT=NEW_DIR`
precompiles and runs the actual 50k coupled fixture under a 180-second cap. It
draws 33 committed frames using two retained machine bindings and the live
particle binding, then writes endpoint PPM images and render timestamps. The
scene uses analytic chassis/wheel/tine geometry and display tetrahedra, not
imported assets or reconstructed soil. Only images and timestamps are read back
for rendering; the physics runtime still performs its existing transfers.
This is an offscreen wgpu 29 example, not a replacement for the Bevy frontend.

### Offline captures

For a temporal sequence, run the diagnostic with `SOIL_TRACTOR_BENCH_CAPTURE_EVERY=11` (default zero disables extra captures). The default 33-frame run records committed frames 0, 11, 22 and 33. Render one with `SOIL_REPLAY_FRAME=11` added to the command below, using a fresh output directory for each selection. `final` is the default. Missing or inconsistent captured frames reject; no interpolation or physics stepping occurs during replay. Capture readbacks and file-writing time are reported outside frame timing, so capture-enabled runs are diagnostic, not production performance acceptance.

Completed native tractor benchmarks now include `data/render-geometry.json`, body poses and final particle positions. From the repository root, render a new offline capture directory with:

```sh
make render-soil-tractor SOIL_REPLAY_RUN=benchmarks/soil/run-id SOIL_REPLAY_OUTPUT=benchmarks/soil/new-capture-id
```

The headless GPU renderer writes `snapshot.png`, `implement.png` and `replay.json`. It displays all 50,000 material points and the recorded analytic body shapes; it does not run physics or invent intermediate frames. Capture metadata records input hashes, simulated time, camera settings and display-marker size. Older runs without recorded geometry, unsuccessful runs and inconsistent snapshots reject explicitly. Existing capture directories are never overwritten. `make test-soil-replay` runs the bounded loader and marker-geometry checks.

This is inspection evidence only: marker size is not physical grain size, the backdrop is not a collision plane, and these captures do not prove terrain reconstruction, calibrated accuracy, persistent ruts or live rendering performance.
