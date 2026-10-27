# Live primitive API examples

## Coupled direct reaction reduction

`RigidWorld::set_mpm_soil_compensated_reactions(enabled, max_bytes)` and
`SoilTractor::set_soil_compensated_reactions` select the existing compensated
reaction reduction after soil attachment. Preparation is cold: first enablement
may allocate and compile. It does not advance a frame or discard committed state.
The separate logical storage budget is `32 * max(max_rigid, 1)` bytes, not part
of `MpmSoilRuntimeStats::retained_bytes` or the motion-guard budget.

`mpm_soil_reaction_info()` reports selection and retained bytes without readback.
Repeated selection is allocation/transfer-free. Disabling retains storage;
budgets below the retained allocation reject even when disabling. Detach and use
the standalone simulation's release API to free that storage.

This option applies only to portable direct coupling. Native atomics and projected
coupling reject enablement; disable it before enabling projected coupling. The
default is unchanged. Compensation reduces accumulation error but does not promise
general determinism or a solver speedup. The portable one-particle direct fixture
has exact matched-control fault/retry evidence, not full tractor acceptance.

## Coupled soil motion bounds

`RigidWorld::set_mpm_soil_motion_bound` and `SoilTractor::set_soil_motion_bound`
take optional finite nonnegative per-axis metres per soil substep and a separate
proposal-storage budget: `16 + 128 * max(max_particles, 1)` bytes. Bounds round
conservatively to f32; subnormal effective bounds become zero. `None` disables
but retains storage. `mpm_soil_motion_info` reports effective bounds and logical
retained bytes without a GPU readback.

Changing bounds invalidates prepared coupling. `prepare_mpm_soil_coupling` (or
`SoilTractor::prepare_soil_coupling`) explicitly rebuilds it outside stepping,
without advancing time. Otherwise the next step prepares it. Preparation is cold
work and may allocate/compile; it is not a zero-latency operation.

The tractor benchmark accepts `--soil-motion-bound off` (default), or isotropic
`METRES:BYTES`, for example `1:6400016` at 50k particles. The Make equivalent is
`SOIL_TRACTOR_BENCH_MOTION_BOUND=1:6400016`. Requested and effective live bounds
and storage are recorded and checked before stepping. A permissive bound tests
integration/overhead; it does not certify a residency envelope or collision coverage.

Motion failures reject the coupled frame. Error-only metadata reports overflow,
numerical and motion-bound flags under the frame's completion deadline. The raw
G2P counter includes entered failed gathers, not only committed substeps.

The current 50k tractor control/guarded/repeat comparison has exact recorded machine
states and final particle checkpoints, but all measured frames miss 60 Hz. Full
performance, long-horizon, rendered-asset and physical-calibration acceptance remain
open; the earlier small cross-world retry-equivalence failure remains unresolved.

## Framed soil tile archives

`MpmSoilFrameCheckpoint::partition_tiles` returns `MpmSoilFrameTileArchive`,
retaining the f64 world offset, exact world-tile keys, global particle/material
IDs and all local particle/history streams. `select` deduplicates requested tiles
and returns a global-ID-sorted `MpmSoilFrameSelection`. Its `soil()` is explicitly
local-coordinate state; `world_origin()` must accompany it. The copying
`to_frame_checkpoint()` helper reconstructs an owned frame, not device IDs.

`writeback` accepts updated framed state in the selected slot order and requires
the same frame, physical configuration, mass, reference volume and material
identity. Moved particles merge into destination tiles. Stale/foreign selections
and invalid changes reject before mutation. `recentered` returns a new archive
and epoch after a lossless translation; previous selections cannot write to it.

The checked versioned `to_bytes`/`from_bytes` envelope includes frame, layout,
IDs and history. Durability is caller-owned. These distinct types do not convert
implicitly to the existing unframed GPU/world/store APIs. Whole-archive host
copying, one shared coordinate frame and no automatic residency remain explicit
limits; this is not canonical per-tile local storage or live-machine rebasing.
`make test-soil-fast PHASE=S07 RELEASE=1` and `make test-soil-tiles RELEASE=1`
cover CPU lifecycle and evolved GPU capture/reload/resume respectively.

The detached `MpmSoilFrame::restore_selection` explicitly restores a framed
selection's physical/history streams and global IDs in one GPU submission, then
publishes its world offset. Different grid origins are accepted after existing
physical/device/capacity validation. First use allocates retained ID buffers;
subsequent restores reuse them but still create temporary upload buffers.
`checkpoint_tiles_with` requires installed IDs and performs a blocking physical
capture followed by host partitioning. It is not a hot-step or durable-store API.

`recenter_with` preserves those IDs, their buffers and mapping epoch while
invalidating prepared physical work. Consumers must refresh the world origin;
an ID epoch describes the mapping, not coordinate freshness. Plain `restore_with`
still invalidates identity, and subsequent identified capture rejects rather than
inventing IDs. These operations do not grant world-residency ownership or rebase
machines, renderers or persistent stores.

`MpmSoilFrameStore` provides explicit blocking durable storage in an existing
private local-filesystem directory. Full save/load retain the shared frame;
bounded `read_tiles` and same-frame `writeback` preserve unloaded tiles and merge
moved particles. Unchanged saves write no new bytes. Full saves can replace the
frame and invalidate earlier read generations. Framed and unframed committed
stores reject cross-opening rather than silently dropping offsets.

`MpmSoilFrameStoreRead::selection_snapshot` copies selected payload/IDs for
`MpmSoilFrame::restore_selection`; retain the original read as the writeback token.
Publication errors poison the handle: reopen to inspect HEAD, since the last
reported generation may predate a completed rename. The store uses the existing
lock, blob/manifest synchronization and atomic-publication protocol.

`begin_read` consumes the exclusive framed store into a bounded worker read;
`begin_writeback` consumes the store, current read and updated framed checkpoint
into same-frame partial publication. Their typed task `poll` methods do not wait
for filesystem I/O and return completions once, moving payload buffers rather
than copying them. Read completions return the owner and selected result; write
completions return the owner, original read, original update and outcome, including
startup/validation failures. Poll disconnection is terminal. A caught worker panic
poisons the returned owner. Frame/provenance and request bounds are checked before
spawning; payload validation and I/O execute on the worker. Thread creation and
metadata checks are still caller work, not a hard latency guarantee.

Dropping a task neither joins nor cancels its worker. The exclusive lock remains
held until work finishes; dropped writes can publish without observed completion.
After a changed write, the returned read is stale. Obtain a new read before reuse.
Full save/load and snapshot/capture APIs remain blocking. Filesystem guarantees
remain conditional on the documented local filesystem contract, not universal
power-loss certification. Persistent verification must set `MOLLA_STORE_TEST_ROOT`
to an owned disk-backed directory; the host's default `/tmp` is tmpfs.

`MpmSoilDiskWorld::new_framed` consumes a committed framed store and an empty
compatible solver. `world_origin()` exposes the fixed shared offset; the ordinary
`new` constructor remains unframed and returns None. Region requests and GPU pin
domains use world tile keys. Solver/grid coordinates, `captured_state` positions
and swept envelope AABBs stay local to this offset. Do not add the offset into
f32 particle positions or pass absolute world AABBs as local swept bounds.

The owner preserves frame metadata through asynchronous capture/write/load and
device publication, retaining stable IDs, step freezing, failure recovery and
capacity checks. Activity groups and host/swept pins use exact world tile keys.
GPU pin preparation supports tile-aligned frames with the existing zero-origin
power-of-two layout: its [-2^30, 2^30] domain constraint applies after translation
to local tile coordinates. Fractional frames can use host pins; GPU preparation
rejects instead of rounding them. Every active particle is still pinned. There
is no runtime frame change, physical sleep or demonstrated bounded traversal;
machine/renderer rebasing and canonical per-tile coordinates remain separate work.

## Canonical per-tile soil archives

`MpmSoilFrameCheckpoint::partition_canonical_tiles` creates a
`MpmSoilCanonicalTileArchive` with f64 position offsets relative to each tile's
exact world anchor. Other particle/history words, material slots, position padding
and global IDs remain unchanged. This is storage precision, not a different
physical solver. Tiny offsets that disappear in absolute world arithmetic stay
separate. Non-exact anchors/residuals reject, including some decimal layouts and
extremely small boundary residuals; no precision fallback or coarsening occurs.

`select(tiles, max_particles, world_origin, grid_origin)` assembles only bounded
requested tiles in global-ID order. Both origins are explicit; converting any
coordinate to f32 must be exact. `MpmSoilCanonicalSelection::framed_selection()`
can be passed to the existing detached `MpmSoilFrame::restore_selection`. Keep
the canonical selection token for `writeback`; it enforces same-frame/config
updates and unchanged mass/reference volume/material identity. Movement merges
destination tiles without rewriting unrelated records. A successful write stales
every older token. Archive selections are snapshots, not live residency leases.

The checked `MOLLCTIL` v1 codec is distinct from both existing raw/framed codecs.
It stores 181 bytes per particle and 28 bytes per occupied tile, plus metadata;
negative-zero flags preserve exact position roundtrips. `from_bytes` takes an
explicit particle bound. Existing raw/framed stores do not consume this format;
the canonical store below does. All operations here run on host archives, not GPU hot
steps. Choosing a new grid origin does not prove physical boundary/halo correctness
or authorize moving machine/renderer state independently.

`MpmSoilCanonicalStore` adds blocking incremental save/load, bounded `read_tiles`,
same-selection `writeback` and explicit collection using the existing exclusive
publication protocol. Read results retain canonical payloads without assembling
absolute or solver-frame positions. `read.select(world_origin, grid_origin)`
assembles the entire bounded read into a `MpmSoilCanonicalSelection`. Keep both
tokens: `store.writeback(&read, &selection, &updated, limits)` requires current
store/generation provenance and the read's own selection. A different frame may
be chosen on a later read without rebasing unrelated stored tiles.

I/O byte bounds reserve `1052 * tiles + 181 * particles`; write limits apply to
the source/destination union, not just the initially selected set. Invalid or
oversized requests leave HEAD unchanged. Publication errors poison the handle,
requiring drop/reopen to determine what committed. Changed generations stale old
reads; unchanged saves write zero bytes. Unchanged cached blobs are not reread for
integrity, while full load/collection validates every referenced payload. The
manifest remains field-sized. These direct blocking APIs are not a production
hot-step route.

`begin_read(tiles, limits, world_origin, grid_origin)` moves both I/O and frame
assembly to one worker. Poll its `MpmSoilCanonicalStoreReadTask` until a completion
arrives; it returns the exclusive owner and `Result<Option<(read, selection)>>`.
None means the store was uncommitted. Assembly errors such as lossy f32 conversion
return a healthy owner for a new request. Frame finiteness and manifest bounds are
checked before spawning without payload reads.

`begin_writeback(read, selection, updated, limits)` consumes all inputs. Its task
completion returns the owner and original inputs even on error. Caller preflight
checks provenance/frame/count/bounds; particle scans, validation, destination I/O
and publication run on the worker. Polls do not assemble or copy payloads and
consume completion once. Disconnection is terminal; publication errors and caught
worker panics require reopening. Dropping tasks does not join/cancel work or
release locks early, and a dropped write may still publish. Thread creation and
metadata work have no hard latency guarantee. Full save/load and detached capture
remain blocking.

`MpmSoilDiskWorld::new_canonical` integrates those workers with the live owner's
capture/write/load/device-publication lifecycle. Supply an explicit world origin;
the simulation supplies the grid origin. Construction reads metadata without loading particle blobs. Canonical
I/O budgets include the affected source/destination union. Existing identity,
pinning, activity and fault/recovery contracts apply; all active particles remain
pinned. Fractional frames support host pins but may reject GPU pin preparation.
Physical sleep and bounded traversal remain unimplemented.

For canonical owners, `begin_recenter(grid_origin)` translates the local grid and
particles exactly and compensates the world offset without rewriting durable
tiles. Poll `poll_region_update` until it returns true. Other updates and stepping
remain frozen meanwhile. Preflight assembles the selected archive token on the
caller; polling validates the captured current state and performs a full physical
upload. These are explicit boundary costs, not an accepted hot per-step path.
Rounding and unsupported GPU pin offsets reject before publication; invalid
capture/device failures fault the owner. Device loss is not rollback. Identity
buffers/epoch, world residency, swept limits and activity duration are retained.
External coordinate consumers must refresh after success. This does not move
physical grid boundaries, rebase machines/renderers, or establish equivalence of
later independent trajectories. Raw/shared-frame disk owners reject this method.

Fixed-order P2G is an explicit runtime option, not just a unit-test path. For
non-native active/compact solvers, configure before transferring the solver into
the live owner:

```rust,ignore
soil.set_active_grid(true)?;
soil.prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), capacity)?;
soil.set_particle_tile_ordering(true)?;
soil.prepare_deterministic_p2g_with_budget(scratch_budget)?;
soil.set_deterministic_p2g(true)?;
```

The scratch budget covers 64 bytes per maximum particle, not ordering or driver
memory. Native explicit solvers support dense, active and compact dispatch.
Disabling gather retains scratch; `release_deterministic_p2g()` releases it only
when disabled. Ordering replacement requires disabling gather and invalidates its
prepared bindings. `set_p2g_mode` selects ordinary scatter and disables gather.
The fixed-order gather prunes tile ranges outside each node's stencil while
preserving accumulation order. The recorded saved-50k transfer comparison improves
median GPU time by 4.22% with exact paired/repeated grid output; it does not include
machine solving or establish full-frame performance.
Defaults remain unchanged; fixed-order soil accumulation alone does not certify
whole-machine reaction repeatability. Run `make test-soil-fixed-order-runtime
RELEASE=1 RUN_WITH=nixVulkan` for the public non-test-library integration checks.

The recorded 129-particle, eight-step paired rebase comparison stays below the
1e-6 physical threshold in all tested modes, but ordinary atomic P2G fails exact
self-repeat. Fixed-order P2G passes both checks in the three supported active/
sparse modes. Index ordering alone is insufficient. This short component result
does not establish long-traverse or 50k machine trajectory acceptance.

## Opt-in shared particle gather

After preparing and enabling particle ordering, select shared immutable chunks:

```rust,ignore
soil.prepare_deterministic_p2g_shared_chunks(scratch_budget, false)?;
soil.set_deterministic_p2g(true)?;
assert!(soil.deterministic_p2g_shared_chunks_enabled());
```

The second argument selects compensated stiffness, which is supported only for
non-native active/compact solvers. Native gather requires `false`. The shared path
requires 16 KiB workgroup storage and the same 64-byte/maximum-particle retained
scratch budget. It does not add a device buffer or dispatch. Preparation alone does
not enable gather; ordinary gather preparation switches back to direct loads.
Existing enable, mode-change, prepared-record and disable-before-release contracts
apply. Unsupported policies or insufficient budgets return errors without fallback.

`make test-soil-shared-gather RUN_WITH=nixVulkan` runs exact small-layout checks and
147 public paired-history cases with bounded GPU setup. CPU contracts use
`make test-soil-shared-gather-fast RELEASE=1`. The fixed-state native 50k component
retains exact outputs and measured median transfer time 0.597504 / 0.510464 ms
(14.57% lower). This advances zero simulated time, is not tractor/frame acceptance,
and does not establish tractor/frame acceptance. The tractor's `shared-chunk:BYTES`
selector uses this API. Defaults remain unchanged.
Evidence: `benchmarks/soil/2026-10-13-gather-shared-runtime-admission/`.

## Opt-in compensated rigid reactions

Non-native MPM can retain a compensated reduction for legacy sphere and prescribed
solid reactions. Preparation does not enable it:

```rust
let bytes = 32 * u64::from(soil.config().max_rigid.max(1));
soil.prepare_compensated_rigid_wrench(bytes)?;
soil.set_compensated_rigid_wrench(true)?;
```

The budget covers the reaction residual buffer, not particle/grid storage. The
reducer survives dense/active dispatch changes and compact-pool growth without
repreparation; transfer policies have their own disable/reprepare requirements.
Prepared step records become stale when configuration changes. Disable before
`release_compensated_rigid_wrench()`. Restore checkpoints before attaching bodies.
Native-atomic MPM is explicitly unsupported; there is no silent fallback.

Run `make test-soil-compensated-wrench RUN_WITH=nixVulkan` for the bounded unit and
public GPU gates, or `make test-soil-compensated-wrench-fast RELEASE=1` for the CPU
source contract. GPU binaries compile before each 90-second setup-inclusive cap.

Public lifecycle checks preserve exact paired particle checkpoints across grid
growth, dispatch changes, explicit/implicit stepping and cancelled submissions.
This does not close the separate 50k cross-transfer force comparison: its recorded
maximum remains 3.051758e-5 versus the 2e-5 bound. Defaults remain unchanged, and no
4 ms/frame or physical calibration acceptance is implied.

## Stagnant-contact branch option

`grouped-eight-newton` enables experimental method 29. It retains method 28's
addition tree with three reduction stages instead of four and the same device,
precision and work-limit requirements. Component gains do not establish full-frame
performance or physical calibration; defaults remain unchanged.

`grouped-sum-newton` enables experimental method 28, grouping compensated
reduction stages while retaining the same addition tree. It requires the protected
256-lane Vulkan route and rejects unsupported devices before solver allocation.
Defaults remain unchanged; captured component gains are not full-frame acceptance.

`stagnant-branch-newton` enables experimental method 27 on the protected
256-lane Newton route and includes `stream-block-newton`. After a previous floor
release, the first zero-smoothing stagnant step can trigger another floor-release
search before a Newton direction. Follow-up settling considers only the
highest-residual unvisited contact and requires decreasing global residual.
Every applied change consumes the existing outer-step and release budgets;
physical/publication checks and default features remain unchanged.

The captured frame-22 failure passes at 64 iterations with this policy; eight
saved successful inputs retain exact baseline state and public outputs. These
are component results, not full-scene performance, trajectory or calibration
acceptance. Reproduction and runtime acceptance status are recorded in
`benchmarks/soil/2026-10-12-branch-transitions/SUMMARY.md`.

## Actual-tractor performance and numerical-agreement report

Run `make report-soil-frontier BASELINE=<reference-data-dir>
INPUT=<candidate-data-dir> SOIL_FRONTIER_OUTPUT=<new-dir>` to compare completed
33-frame, 50k-particle RTX 4080 diagnostic artifacts. Both inputs must include
their raw manifest, result, frame CSV, MPM profiles, machine trajectory and final
particle positions. For additional observations, use the `soil_benchmark frontier`
CLI with repeated `--input` arguments. `make test-soil-frontier` runs its CPU
contracts and is included in `make test-soil-fast PHASE=S00`.

`frontier.json` recomputes timing statistics and numerical discrepancies, checks
declared fixture compatibility, and fingerprints the inputs. Source-provenance
audits are separate. Exit zero means the report is valid, **not** that performance
or physical accuracy passed. The supported diagnostics exclude rendering/export
and complete soil wall time; their accepted frontier therefore stays empty.
Observed nondominance is not a paired speedup, calibrated accuracy or a complete
quality/resolution sweep. Existing output directories are rejected.

## Fixed-order tractor transfer

`bench-soil-tractor` accepts `SOIL_TRACTOR_BENCH_TRANSFER=atomic` (default),
`fixed-order:BYTES`, `shared-chunk:BYTES` or `grid-reuse:BYTES`. Fixed-order transfer requires the native-prediction Newton
configuration and works with dense, active and compact soil storage. For the
50k fixture, `fixed-order:3200000` provides the exact gather stress-cache budget;
particle ordering and driver storage are excluded and reported separately.
The executable option is `--soil-transfer fixed-order:3200000`.

`shared-chunk:3200000` selects the same contributor order with workgroup-local
immutable particle chunks. Its stress-cache budget is unchanged; the device must
provide 16 KiB workgroup storage. Use `SoilTractorTransfer::SharedChunk` through the
public constructor. The live transfer query includes `shared_chunks`; benchmark
manifests distinguish the selections and reject requested/live mismatches.
Ordinary `fixed-order` remains the comparison control, not an alias for shared chunks.

`grid-reuse:14075712` retains exact-input transfer outputs for the default 50k,
grid64 fixture. The budget includes particle keys, dense logical node outputs,
tile flags and a header, even when physical grid storage is sparse. Ordering and
driver storage remain separate. `SoilTractorTransfer::GridReuse` exposes the same
option; `validate_with_grid` checks its full budget before soil allocation.
`MpmSoilTransferInfo::grid_reuse` reports the actual prepared cache. The viewer
accepts `SOIL_LIVE_TRANSFER=grid-reuse`; atomic remains the default. This is not
physical sleeping, eviction or a claim of real-time performance.

The same-source 50k control/shared/repeat runs each complete 33 frames with exact
recorded machine states and full final particle checkpoints. Mean P2G time is
15.05 / 10.08 / 9.99 ms per frame, but total wall medians remain 357.54 / 346.71 /
343.14 ms and every measured frame misses 60 Hz. Contact Newton still costs about
212–214 ms per frame. This is bounded integration evidence, not real-time or
uncontended overall-speed acceptance. Source and artifacts:
`benchmarks/soil/2026-10-13-shared-gather-tractor/`.

The opt-in gather uses workgroup integer stencil-support masks and visits bits in
the same tile/particle accumulation order. Mask storage is workgroup-local; the
retained scratch budget and dispatch count are unchanged. Paired saved50k transfer
measurements improve GPU median by 43.71%; actual tractor runs preserve all recorded
machine/final particle-state bits. Full-frame performance and a separate non-native
scatter stiffness gate remain failed; this does not promote the atomic default or
establish physical accuracy. Evidence: `benchmarks/soil/2026-10-12-gather-masks-runtime/`.

Ordering and gather are prepared before attachment. The manifest records requested
and live transfer identities, sort selection and retained bytes. Unsupported
combinations, insufficient budgets and live-selection mismatches are errors;
there is no fallback to atomic transfer. This option is a repeatability diagnostic,
not whole-machine accuracy, performance or calibration acceptance.
`make test-soil-tractor-fixed-order RUN_WITH=nixVulkan` checks the public constructor
and short independent-owner replay under a 90-second setup-inclusive GPU cap.
The initial integrated test reached that cap without phase-localizing the stall;
runtime integration is not yet GPU-verified. The timeout is retained in
`benchmarks/soil/2026-10-12-tractor-fixed-order-runtime/`.

### Final particle history

`SOIL_TRACTOR_BENCH_FINAL_STATE=1` (`--final-soil-state 1`) opts into one final,
blocking `final-soil-state.bin` capture after all measured frames. Default is off.
The existing lossless `MOLLSOIL` format stores particle position, velocity, APIC C,
elastic F, reference volume, mass, plastic volume history, material IDs/table and
compaction/impulse diagnostics. The result records frame identity, BLAKE3 digest,
encoded size and separate observation counters/timing. At 50k particles the GPU
staging/readback is 8,000,016 bytes; it is not included in hot-frame counters.

`SoilTractor::read_soil_checkpoint(max_particles)` and
`RigidWorld::read_mpm_soil_checkpoint(max_particles)` reject uncommitted, faulted,
stale-model and over-limit state before allocating or reading back. Observations
retain the published frame after successful rollback. These are particle archives,
not whole-world restart saves: machine state, grid scratch, Newton warm histories,
global soil identities and scheduling state are excluded. Capturing matching final
particle histories does not prove calibrated accuracy or long-horizon equivalence.

## Active-contact publication option

`active-publication-newton` enables method 21 on the protected 256-lane Newton
route and includes `norm-tree-newton`. Two serial publication scans visit the
ascending active-contact list; full-capacity validation and output clearing remain.
Ordinary WGSL and default features are unchanged.

Run `make test-soil-newton-active-publication-runtime RUN_WITH=nixVulkan` for
source/metadata checks and bounded capture/native-feedback GPU checks.
For the full diagnostic fixture use
`benchmarks/soil/2026-10-11-newton-active-publication-runtime/actual.sh` with
`bash`; it uses Make, a fresh output directory and a 180-second cap.
The captured solver speedup is not a full-frame speed or calibration claim.

The test-only equal-input shadow diagnostic is run by
`bash benchmarks/soil/2026-10-11-newton-publication-shadow/run.sh`.
It compares methods 20/21 on every contact solve through 14 live frames, retains
first-difference metadata, and never feeds shadow outputs into the world.
Its extra solver/storage/dispatches/readbacks exclude performance acceptance.

## Experimental response preparation

For experimental Newton response preparation, enable the solver crate feature
`axis-parallel-newton-preparation`. It solves the three contact response columns
on separate GPU lanes, retaining contact-wide convergence, compensated arithmetic
and the original error/refinement limits. The benchmark records the selection as
`soil_newton_preparation_backend`; defaults remain unchanged. Exact component
checks and faster preparation do not establish whole-frame real-time acceptance.

`strict-rounding-newton-preparation` adds experimental non-contracting SPIR-V
arithmetic and volatile lane-private rounding to axis-parallel preparation.
It requires a native Vulkan engine with `PASSTHROUGH_SHADERS`; unsupported
devices fail construction rather than silently selecting another backend.
The benchmark requests that feature for Newton runs and labels this route
`compensated-axis-parallel-strict-volatile-rounding`. Arithmetic operations,
refinement limits and tolerances are unchanged. Bare volatile rounding is
test-only and is not a runtime option. Defaults remain unchanged.

## Resident tractor rendering

`soil_tractor_resident.rs` renders the actual 50k-particle tractor/tine workload
on the physics device using live particle positions and committed GPU body
transforms. Run `make render-soil-tractor-resident RUN_WITH=nixVulkan
SOIL_RESIDENT_OUTPUT=<new-dir>`; it precompiles before the 180-second execution cap.
The output contains endpoint PPM images and a JSON timing report for 33 frames.
It uses analytic fixture geometry, not imported assets or the Bevy frontend.
Render-pass timing excludes physics, copies and submission overhead; existing
physics transfers remain. Reports explicitly retain uncalibrated status and
`acceptance: false` rather than treating cheap rendering as real-time physics.

## Soil-bed preparation

`soil_settling.rs` runs a fixed 512-particle gravitational bed through the bounded
full-population settling driver. Run `make settle-soil SOIL_SETTLING_DIR=<new-dir>`;
the target compiles first, freezes source/executable provenance, then applies a
30-second execution cap. `SOIL_BENCH_SECONDS` may lower that cap to 1–30 seconds.

The example saves its policy, every activity observation, initial/final checkpoints
and final archive. Exit 0 means the configured endpoint activity criteria held;
exit 2 means nonconvergence or error. A missing final result means incomplete work,
not a settled bed. The checked-in one-second case reaches StepLimit and remains
unsettled. Do not treat this small preparation case as force equilibrium, physical
calibration, particle sleeping or the tractor/implement real-time acceptance gate.

Continue a completed segment with
`make settle-soil SOIL_SETTLING_DIR=<new-dir> SOIL_SETTLING_RESUME=<parent-results-dir>`.
The loader checks bounded inputs, parent hashes, policy/configuration, particle
identity and archive/checkpoint agreement before GPU initialization. Parent
evidence is copied into the new output; v2 metadata records cumulative physical
steps/time. The quiet window restarts conservatively for each segment. The saved
one- and two-second gravitational-bed segments both remain nonconverged; resuming
does not authorize changing activity thresholds or claiming calibration.

## Analysis and physical response

`mass_matrix` and legacy `inverse_dynamics` omit finite equality Ball/D6 motion
columns: locked slots have zero matrix rows/columns and generalized efforts.
`response_mass_matrix` retains partially constrained angular columns for the
coupled response, matching the GPU mass kernel. Physical support loads come from
`RigidWorld::joint_reaction` or `rnea::inverse_dynamics_wrenches`, not zero legacy
effort slots. The wrench API retains partial angular support components and
reports world torque about the authored anchor. Independent translated/rotated
Ball/D6 full-inertia/COM oracles and the original D6/revolute analysis oracle are
in `analysis_response_contract.rs` and `multidof_oracle.rs`.

## Runtime examples

`runtime_primitives.rs` uses public `RigidScene` and `RigidWorld` APIs without a
USD importer, model assets, or benchmark helpers. Each named function constructs
an independent world and prints physical readout. These examples demonstrate API
usage; they do not replace the plan's numerical, resource, or performance gates.
The primitive implementation and its support matrix are still in progress; see
the repository's `MOLLA_PRIMITIVES_REPORT.md` for current limitations.

## Simultaneous drives and constrained inertia

`make test-primitives-mixed-drives-cpu` checks 24 world-anchored D6 bodies with
simultaneous linear effort/angular velocity drives and real sliding contacts,
plus 48 two-body Ball/D6 partial-angular-lock chains on CPU. Masses, full
off-diagonal inertia, COM offset, rotated root frames, Force/Acceleration scaling,
zero/finite/infinite caps and live angular targets are exercised. The independent
one-second rate/pose/reaction oracles retain CPU1e-8/GPU1e-3 gates; quaternion
increments follow normalized Euler, not exact exponential integration.

`make test-primitives-mixed-drives-gpu` checks the 24 root cases against both the
independent oracle and the actual CPU trajectory on hardware. Both lanes require
exact advancement, finite state/loads, zero quarantine/break events and sliding
impulses. GPU model compilation must remain unchanged after initial readout
setup. Bounds are CPU120s/GPU180s; the latter includes observed146.01s pipeline
preparation plus the full240-frame/two-substep trajectory, not reduced coverage.
Current maximum GPU pose/parity error is7.565e-6; reaction error is3.548e-6 when
normalized by mass/inertia. See `primitives-articulated-drives-all-axis-frames-mixed-regression-gpu.log`
in `benchmarks/primitives/impact-history-gpu-2026-09-29/`.

Partial angular locks change effective drive mobility. CPU feedback uses the
bilateral angular Jacobian projection across the articulation. GPU feedback now
uses the same constrained mobility across the tree, retaining the isolated-root
fast path. `make test-primitives-locked-drives-gpu` checks 48 independent two-body
articulations: Ball/D6, X/Y/Z free axes, identity/rotated signed endpoint frames,
all four Force/Acceleration pairs, full inertia/COM and one-second trajectories.
An independent two-by-two pivot-inertia oracle and actual CPU trace both pass
the original1e-3 gate. Maximum normalized state/drive-reaction error is5.501e-6;
CPU/GPU pose/velocity parity is5.159e-7. No extra GPU buffer/submission or CPU
solve is introduced. Private Gram scratch is bounded by two angular rows per
joint; the articulated projection checks every constraint row after rank
selection/refinement. The full
unprojected GPU ABA/stop/contact response is retained for constraint recovery,
rather than making late pose corrections impossible through a zero mobility.
This lane does not certify every loop/contact/lifecycle combination or P5 as a
whole. Earlier release benchmark source manifests remain historical snapshots
after this shader change; the final-source performance matrix is still open.

Current chain evidence is in `primitives-articulated-drives-all-axis-frames-mixed-regression-gpu.log`
under `benchmarks/primitives/impact-history-gpu-2026-09-29/`. The native180-second
GPU limit includes observed117.83-second pipeline preparation plus all100 frames,
two substeps and both reaction endpoints. CPU checks keep the1e-8 gate.

`make test-primitives-locked-drive-planes-cpu` and
`make test-primitives-locked-drive-planes-gpu` extend that coverage to24 noncoaxial
Ball/D6 chains with one locked and two free angular axes per joint. Different
root/descendant lock axes, signed rotated endpoint frames, full tensor/COM,
mass2/17 and all Force/Acceleration pairs exercise passive-axis motion over1.005 s.
An independent world-space pivot-inertia projection checks initial angular rates
(CPU1e-6/GPU1e-3) and support torque (1%). Subsequent reversal/disable/re-enable
checks retain anchors/locks and actual CPU/GPU trajectory parity within1e-3.
Current GPU maximum locked-angle error is1.722e-6 rad; parity is1.475e-7.
The native180-second limit includes114.40-second preparation and the full101
steps. State/reactions remain finite, no body is quarantined, and model
compilation is retained through hot controls. Evidence is
`primitives-noncoaxial-drive-planes-hot-controls-cpu-gpu.log` in the same directory.
These checks are not final all-combination P5 or performance certification.

From the repository root:

```sh
export NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
nix develop --impure -c make example-primitives
nix develop --impure -c make example-primitives A=distance
nix develop --impure -c make example-primitives-gpu A=cone
nix develop --impure -c make example-primitives-gpu
```

Cases are `distance`, `cone`, `reaction`, `break`, `friction`, `d6`, or `all`
(the default). CPU and GPU are explicit choices; requesting `--gpu` in a binary
compiled without the GPU feature returns an error instead of executing on CPU.
GPU examples require hardware adapter information and reject software adapters.
One hardware Engine is created per invocation and shared by the selected GPU
examples. Each example still constructs its own scene, model, buffers and state;
only immutable programs are reusable across these independent worlds. Selecting
`all` no longer creates a separate device and cold program cache for every case.
The GPU command uses the repository's `nixVulkan` wrapper. Serialize it with other
GPU validation, and do not run either command during performance measurements.

| Function | Authoring calls | Readout |
|---|---|---|
| `distance` | `insert_joint(Distance)`, `set_distance_limits` | `distance_reading`: separation, violation, active bound and child force |
| `cone` | `insert_joint(Ball)`, `set_cone_limits` | `cone_reading`: swing coordinates, violation and child torque |
| `reaction` | `insert_joint(Fixed)`, `set_reaction_readout(true)` | `joint_reaction`: frame mean and peak support loads |
| `breaking` | `set_joint_break_limits` | `drain_joint_break_events`, then a second step without a repeated event |
| `friction` | per-collider `ContactMaterial.friction` | `contact_pairs` and body velocity after sliding/rolling contact |
| `d6` | `set_joint_locked_axes`, `set_joint_motor(PositionVelocity)` | body pose after simultaneous position/velocity feedback |

## Frames, units and lifecycle

- Scene gravity and positions use metres and seconds. Mass is in kilograms,
  inertia in kg·m², forces in N, torques in N·m, and cone angles in radians.
- `parent: None` is a world endpoint, not a request to freeze the child. The
  authored endpoint frames locate each anchor. Distance leaves transverse motion
  free; the ball/cone example preserves ball anchoring and cone twist freedom.
- Distance bounds are independent `Option<f64>` values. This example uses equal
  one-metre bounds; use `None` for an absent endpoint, not a fabricated zero.
  Cone extents are separate geodesic swing half-extents, not rectangular Euler
  clamps. The loaded example uses 0.25 and 0.4 radians.
- General reactions require opt-in readout and a successful solved frame. Loads
  are in world coordinates, with each endpoint torque taken about that endpoint's
  authored anchor. They are forces/torques, not impulses; `sample_dt` is seconds.
  Missing or invalidated readout is `None`, not a physical zero wrench. Read it
  before changing the scene. Sleeping joints can have no solved sample.
- Distance, cone and ball-anchor samples also invalidate on hot force, gravity,
  control or mutable-state access, even when the compiled scene is retained.
  A successful subsequent frame publishes fresh readings; rejected and no-op
  edits preserve the existing sample.
- Featherstone adds `Control::body_external_f` to `State::body_f` once per
  substep. Both use world axes and torques about the body's COM; control loads
  are not accumulated back into state forces. `add_body_wrench` accumulates the
  state channel until `reset_body_forces`; replacing a control entry updates the
  separate control channel. Resolve `scene.body_index(handle)` when writing a
  control entry, rather than retaining a dense index across topology edits.
  Reaction readout and break thresholds include both channels.
- The fixed two-kilogram reaction example supports 20 N against gravity. The
  break example instead sets a 5 N threshold. Threshold evaluation and detachment
  occur during stepping; committed events remain queued until drained. The event
  contains its sample, trigger, original handle and frame/substep timing. The
  detached body's handle remains usable. Draining is not joint repair.
- Friction sets both static and dynamic coefficients explicitly on both contact
  colliders. The ball is free to rotate: nonzero rolling speed does not mean the
  tangential contact is slipping. This example is not a static block-holding test.
  Its collider mass is zero because the body's mass/inertia are supplied explicitly.
- The D6 example frees linear X, locks the other five axes and requests a 0.25 m
  position with zero target speed, capped at 5 N. Its printed one-second pose is
  a response sample, not a promise that the transient has fully settled.

The GPU live-runtime route retains device substeps but currently synchronizes
host state and diagnostics at frame boundaries. These examples do not certify
arbitrary loop, compliant, sleeping-contact, actuator, wheel or track combinations.
Unsupported combinations must continue to return explicit errors. Gearbox must
handle USD units, defaults, collections and schema mapping separately.

## Verification status

All six CPU examples and all six actual-GPU examples pass on the engine-local
program-cache source on 2026-10-01. Current logs are
`benchmarks/primitives/engine-program-cache-2026-10-01/examples-shared-engine-cpu-clippy.log`
and `examples-shared-engine-gpu.log`. GPU execution uses RTX 4080 / Vulkan /
NVIDIA 595.91.07 and one Engine across the six independent worlds. Feature-disabled
GPU selection also returns the explicit no-CPU-fallback error; see
`examples-no-gpu-fail-closed.log`. Full workspace formatting, all-target check and
strict all-target/all-feature Clippy pass in `post-example-workspace-static.log`.
Earlier example runs remain historical evidence, not current certification.
These examples demonstrate usage, not complete physical or
scale certification. The current benchmark partition, source identity and
remaining original 56-case matrix are recorded in `MOLLA_PRIMITIVES_REPORT.md`;
older distance measurements are not silently counted as current-source results.
Existing integration tests remain the numerical evidence:
`runtime_distance*`, `runtime_cone*`, `runtime_joint_reactions`,
`runtime_reactions_gpu`, `runtime_joint_breakage*`, `runtime_friction`,
`runtime_mixed_contacts*`, and `runtime_multidof*` / `runtime_d6_axis_certification`.

## Bounded implementation wiring

```sh
nix develop --impure -c make check-primitives-shader-wiring
nix develop --impure -c make check-primitives-gpu-wiring
nix develop --impure -c make check-primitives-gpu-step-wiring
```

The shader check parses and validates five production WGSL modules without
requesting an adapter. GPU preparation constructs two reversed-D6 worlds but
advances no physics. The optional step check compares one frame of three worlds
against matching CPU references, including an asserted active compliant angular
limit. Both GPU targets have a 120-second process deadline and ten-second kill
grace. The fixture retains nonidentity frames, full tensor/COM, a soft external
loop, motors, brakes, speed caps, damping and break thresholds. Hardware execution
is required; there is no software-adapter or CPU fallback.

The three one-frame comparisons passed on 2026-09-30 after correcting reversed
joint momentum transport, with maximum angular-rate error 2.956379141e-7 against
the declared 1e-3 tolerance. See `primitives-one-frame-loaded-compliance.log` in
the log directory above. This is not sustained numerical, contact, pole-chart,
mutation, resource or performance certification.

Active rigid XYZ phase rows have separate short gates:

```sh
nix develop --impure -c make test-primitives-singular-wiring-cpu
nix develop --impure -c make test-primitives-singular-wiring-gpu
```

These check both exact Y poles for one frame, using independently computed
combined X/Z phase intervals. They retain hard bounds and do not substitute
compliance. The CPU task covers both poles in one process; the GPU task composes
two original per-pole processes, each with a 120-second deadline. Current full
16-contact-reservation GPU checks pass in
`primitives-xyz-original-positive-current-gpu.log` and
`primitives-xyz-original-negative-current-gpu.log`. Preparation takes about
83–85 seconds per world, versus about 0.033 seconds for its one physics step.
These are preparation diagnostics, not performance-matrix timings.

```sh
nix develop --impure -c make test-primitives-chart-wiring-cpu
nix develop --impure -c make test-primitives-chart-wiring-gpu
```

These execute 1,000 one-millisecond frames across each pitch pole with bounded
but unlocked angular axes and zero gravity. The original sixteen-frame crossing
is retained, then motion continues to the authored +/-1.9 rad pitch bound and
stops. Independent orientation/rate checks include the fractional boundary step
and remaining stationary frames. Both CPU cases pass in
`primitives-xyz-sustained-crossing-current-cpu-clippy.log`; current hardware cases
pass in `primitives-xyz-sustained-positive-current-gpu.log` and
`primitives-xyz-sustained-negative-current-gpu.log`. The first CPU log also records
a progress-condition Clippy failure, corrected in the positive GPU log.
The GPU task composes both per-pole processes with 120-second deadlines.
Individual jobs can be selected without changing either physical fixture:

```sh
nix develop --impure -c make test-primitives-singular-pole-gpu PRIMITIVE_XYZ_POLE=positive
nix develop --impure -c make test-primitives-chart-pole-gpu PRIMITIVE_XYZ_POLE=negative
```

D6 angular bounds select an equivalent
intrinsic-XYZ chart using the authored intervals, including periodic angle lifts
and the alternate pitch branch; they do not expose an accumulated revolution
counter. Mixed angular locks, general crossing trajectories and active contact,
energy and resource acceptance remain open.

```sh
nix develop --impure -c make test-primitives-velocity-wiring-cpu
nix develop --impure -c make test-primitives-velocity-wiring-gpu
```

These check one-frame coupled force-model velocity drives against an independent
implicit-rate oracle. The xy inertia has principal moments 0.001/1.999, with a
physical orthogonal moment 2.0. Six cases cover rotated frames and force caps
zero/20/infinity. Both CPU and hardware GPU pass in
`primitives-velocity-active-set-physical-inertia.log`. The feedback solve uses a
bounded active set rather than a fixed count of relaxation sweeps; failed pivots
or exhaustion remain errors.

```sh
nix develop --impure -c make test-primitives-velocity-sustained-cpu
nix develop --impure -c make test-primitives-velocity-sustained-gpu
nix develop --impure -c make test-primitives-velocity-raw-gpu
```

The sustained fixture retains the original Ball cases and adds D6/Free: eighteen
closures, 100 frames at dt=0.01 with one substep, simultaneous x/y force drives
at damping 1e6 and the same physical inertia/caps/rotated frames. An independent
common-eigenvector oracle checks bounded implicit rates and normalized-Euler
orientation every frame. Targets reverse after 0.5 s, disable after 0.75 s and
re-enable after 0.9 s without layout changes. CPU gates remain 1e-8 and GPU gates
1e-3; finite state, no quarantine/breaks, exact step count and retained GPU model
compilation are checked. The raw target separately checks all eighteen roots
without coordinate compensation. Its actual hardware run passes in 47.14 s,
with peak rate error 4.269676746e-5 rad/s under the original 1e-3 rate gate;
see `primitives-velocity-raw-fallback-current-gpu.log`. This is one-step raw
coverage, not the one-second live-runtime trajectory.

`primitives-velocity-sustained-current-cpu-clippy.log` proves the CPU trajectory
in 0.07 s. `primitives-velocity-paired-response-corrected-gpu.log` proves actual
RTX 4080/Vulkan execution in 48.27 s: all-frame peak rate error
7.191560496e-4 rad/s and orientation error 3.732427834e-5 rad. Earlier failed
hardware attempts are preserved, not passing evidence. Isolated quaternion
roots retain paired motion-frame mobility through generalized response and use
matching gyroscopic/COM dynamics; stable no-brake force velocity rows defer to
the coupled feedback solve. No per-step CPU solve or GPU pipeline creation is
introduced. This does not certify arbitrary mixed drive/brake/contact scenes,
all dt/mass sweeps, or the complete performance matrix.

Affected hardware regressions also pass: original 540 off-centre Ball pivot
closures with physical motor caps/model scaling/brakes/reactions (82.73 s), and
the original 144 D6 axis cases over one second with full COM/tensor/mass sweeps
(84.49 s). See `primitives-velocity-paired-pivot-regression-gpu.log` and
`primitives-velocity-paired-all-axis-regression-gpu.log`. These are scoped
regressions, not aggregate acceptance. The new sustained trajectory's observed
GPU preparation was 47.14 s versus 23.99 s before repair; those uncontrolled
setup timings are not a throughput benchmark. Performance measurement remains
required before any performance or whole-plan completion claim.

## Coupled hinge and slider motor models

```sh
nix develop --impure -c make test-primitives-motor-coupling-cpu
nix develop --impure -c make test-primitives-motor-coupling-gpu
```

The existing two-body slider fixture now also covers coaxial hinge chains.
Seventy-two cases per feedback mode preserve the original mass, Force/Acceleration
model combinations, capped/uncapped controls and loaded/unloaded configurations.
An independent two-by-two inverse mass/inertia oracle checks joint coordinates,
physical body rates/poses and child reactions. Four substeps and forty frames
remain unchanged: explicit dt=.0004 spans .016 s, stable dt=.02 spans .8 s.
This is not a one-second trajectory claim. Original CPU 1e-8 and GPU 2e-4 scaled
gates remain unchanged, with finite state/reactions, no quarantine/breaks and
exact completed steps. GPU compilation is retained after the initial explicit
reaction-readout configuration boundary, not compared against its constructor.

CPU default both-mode test passes in 1.67 s. Actual RTX 4080/Vulkan explicit and
stable runs pass in 49.09/47.49 s, with coordinate maxima
2.374711344e-8/6.623242264e-7, rate 2.939390508e-6/3.710364803e-6 and reaction
7.035145036e-5/4.604191384e-5. Logs are
`primitives-coupled-hinge-slider-current-cpu-clippy.log`,
`primitives-coupled-hinge-slider-explicit-current-gpu.log` and
`primitives-coupled-hinge-slider-stable-readout-gpu.log`. The first GPU attempt
failed a newly added compilation-baseline assertion, not a physical oracle;
preserve `primitives-coupled-hinge-slider-stable-current-gpu.log` as failed.

The canonical GPU task composes both original mode jobs with 120-second deadlines.
`test-primitives-motor-feedback-gpu PRIMITIVE_MOTOR_FEEDBACK=stable` selects one;
the underlying test still defaults to both when the environment variable is absent.

## Mixed D6 angular point locks

Partially locked Euler angular motion uses physical equality rows while keeping
Cartesian anchor-frame angular rates. Linear locks and fully locked angular
motion retain structural zero columns. A pitch point at an XYZ pole uses two
swing-alignment equalities. Two angular points define an exact hinge with two
alignment rows and a stable twist coordinate for the remaining axis's bounds.
No dummy colliders are needed. Standalone CPU dynamics/drift and GPU dynamics
use coupled mass response for partial angular locks, bounds and speed caps;
retained GPU capacity changes require recreating the solver.

```sh
nix develop --impure -c make test-primitives-angular-lock-wiring-cpu
nix develop --impure -c make test-primitives-angular-lock-wiring-gpu
```

Both backends check 84 regular/pole, rotated-frame and one/two-lock fixtures
against independent physical pose/rate oracles, through runtime and standalone
routes, including bounds on all three remaining hinge axes. Runtime checks two
frames; standalone additionally corrects deliberately injected forbidden rates
on a third public drift frame. CPU checks pass in
`primitives-mixed-domain-drift-wiring.log`; actual RTX 4080/Vulkan checks pass in
`primitives-mixed-domain-integration-gpu.log` (24.36 seconds for the angular file).
Standalone GPU status must remain zero. Maximum GPU rate error is 3.86e-5 and
rotation error 6.48e-7 against 1e-3 gates. No software fallback or tolerance
relaxation is used.
Earlier memory and physical failure logs are preserved.
Floating/loaded-contact, mutation/break and sustained acceptance remain open;
these short fixtures are not full mixed-lock certification.

Independent mass/frame lanes cover 64 single X, single Z, pole-pitch and two-point
locks with off-diagonal inertia, COM/anchor offsets, nonidentity endpoint frames,
both pole signs, both tree traversals and allowed/forbidden loads. Runtime checks
analytic authored reaction wrenches as well as body motion. CPU/GPU runtime also
check dt=1e-4 without relaxing the 1% reaction gate. Authored fixed reference
quaternions retain their high/low components through the GPU residual evaluation;
small position-rounding errors must not create spurious reaction loads.

```sh
nix develop --impure -c make test-primitives-angular-mass-wiring-cpu
nix develop --impure -c make test-primitives-angular-mass-wiring-gpu
```

Fresh CPU/shader checks pass in `primitives-angular-point-reference-checks.log`;
the three actual RTX 4080/Vulkan tests pass in
`primitives-angular-point-reference-gpu.log` (79.27 seconds). These are focused
one-step oracles, not long-duration or loaded-contact certification.

Finite-bound runtime lanes add 32 full-inertia/COM fixtures at dt=1e-4:
lower/upper stops, inward/outward torque, both pitch poles, rotated frames and
reversed traversal. Upper-stop fixtures retain negative quaternion coordinates;
equivalent quaternion signs must not change the physical stop reaction.
Near a stop, the GPU forms a compensated local twist gap before atan2 instead of
subtracting rounded absolute angles. The authored interval lift still selects
the global branch. An undefined twist at an opposed-swing antipode remains an
explicit error; these checks do not certify that recovery case.

```sh
nix develop --impure -c make test-primitives-angular-bound-wiring-cpu
nix develop --impure -c make test-primitives-angular-bound-wiring-gpu
```

Actual GPU evidence passes in `primitives-angular-bound-reference-gpu.log`:
32 reactions meet the unchanged 1% gate, maximum torque error 2.80e-6 N m.
The GPU mass lane excludes the separately named bound/regular checks to stay
within its 120-second cap; the full baseline includes every test in the file.

Regular XYZ point lanes cover 128 full-tensor/COM/frame/reversed fixtures at
dt=1e-4, including principal and alternate pitch charts. Selected-chart matrix
components retain their tails through the near-stop angle difference; pitch
cosine uses a compensated square root with staged rounding barriers.

```sh
nix develop --impure -c make test-primitives-angular-regular-wiring-cpu
nix develop --impure -c make test-primitives-angular-regular-wiring-gpu
```

`primitives-angular-regular-reference-charts.log` passes CPU and actual GPU
with unchanged 1% reaction gates; GPU maximum torque error is 4.94e-6 N m.
These one-step checks do not certify singular phase precision or sustained
contact-loaded motion.

Narrow finite intervals retain their authored point/interval classification.
Euler rows carry exact point bits; `Model::joint_motion_flags` also carries
authored axis-point bits for ABA, mass, Jacobian and reaction response masks.
Full model upload refreshes these derived flags after limit/enable edits.
This adds one persistent 4-byte/joint host/device buffer, using existing enable
bindings. It adds no dispatch or per-frame readback; the public enable mirror
remains boolean.

```sh
nix develop --impure -c make test-primitives-angular-narrow-wiring-cpu
nix develop --impure -c make test-primitives-angular-narrow-wiring-gpu
```

`primitives-motion-metadata-gpu.log` passes 32 runtime full-tensor/COM/frame/
reversed fixtures at dt=5e-5 with width=1e-8 and equal f32 endpoint casts.
Inward motion remains free inside the authored interval; outward load reacts.
The unchanged 1% gate passes, maximum torque error 8.14e-6 N m. Existing angular,
mixed Distance and disabled-joint GPU checks also pass. Standalone sub-f32
stop-value accuracy remains separate acceptance work.

Singular phase rows now retain authored combined endpoints and sine/cosine low
tails. Physical pole sign is selected before authored pitch winding; no extra
row, buffer or dispatch is added. Focused lanes are:

```sh
nix develop --impure -c make test-primitives-angular-phase-wiring-cpu
nix develop --impure -c make test-primitives-angular-phase-wiring-gpu
```

The 64-case full-tensor/COM/frame/reversed physical oracle covers +/-pi/2,
+/-3pi/2 and negative quaternion representatives at dt=1e-5, width=1e-9.
CPU and actual RTX 4080 GPU pass; the latter takes 38.01 seconds with maximum
torque error 1.53e-4 N m against the unchanged 1% gate. Its combined command
later times out in a separate legacy regression. That regression's negative
pole passes but positive pole remains unverified after three bounded attempts;
the original 16-contact reservation is restored. This is not complete sustained,
contact-loaded or performance acceptance.

Finite two-point angular locks use a deterministic one-sided twist chart at
opposed swing. With cyclic basis `(p,b0,b1)`, the existing zero-extent cone
recovers along positive swing about b1, giving twist `2 atan2(q.b0,q.b1)` at the
cut. The analytic left-rotation derivative is -p there. The ordinary primary
twist chart is unchanged away from its singularity; both charts preserve
quaternion-sign equivalence and authored winding. Retained runtime references
select matching transverse components and low tails. No bound is silently
retired and no buffer/binding/row count changes.

```sh
nix develop --impure -c make test-primitives-hinge-antipode-wiring-cpu
nix develop --impure -c make test-primitives-hinge-antipode-wiring-gpu
```

The authored physical fixture covers all three free axes, two world frames,
forward/reversed trees, negative quaternion representatives and interior/both
violated finite stops. It uses full off-diagonal inertia, rotated local frames,
independent first-step velocity/reaction predictions and 32-step recovery gates.
The 72-case fixture passes CPU in 0.88 seconds and actual hardware GPU in 39.53
seconds. First-step reactions pass the unchanged 1% gate; all cases recover
within 1e-3 swing/bound error after 32 steps without quarantine or break events.
Strict Clippy, eight exact shader compositions and analytic derivative/recovery
unit checks also pass. This centered-anchor/zero-COM fixture does not certify
contact-loaded, floating or offset-COM cut-locus behavior. The separate 72-case
standalone CPU first-step oracle passes in 0.73 seconds. The original standalone
GPU fault 32 came from final projection feasibility/complementarity checks:
radial residual 6.10e-5 exceeded the unchanged 2e-5 internal gate. Refinement
now retains high/low mass-response corrections and evaluates residuals with
staged paired products/sums, preserving coordinate-residue scratch. Signed
bilateral corrections remain free; unilateral corrections remain nonnegative.
No row/buffer/binding/dispatch is added, but private per-articulation scratch
grows by one scalar tail per local response component. Working-precision gates
are unchanged; published velocities remain f32 and pass the physical 1e-3 gate.
All 72 standalone hardware fixtures now pass, maximum angular-rate error
1.17e-4 rad/s. Final-source Clippy and shader checks pass; runtime/standalone GPU
fixtures pass together in 57.74 seconds. Original failures/diagnostics remain
preserved. Standalone sustained/contact/floating/COM acceptance remains open.
The separate compensated-state 72-case first-step lane passes in 14.76 seconds.
Related mixed Distance and all 84 normal-chart runtime/raw/public-drift cases
also pass on CPU/GPU; GPU Distance takes 11.77 seconds and the two angular tests
57.36 seconds. These checks do not substitute for full sustained/performance
certification or standalone sub-f32 authored stop accuracy.

The standalone Make lanes include explicit compensated-state coverage:

```sh
nix develop --impure -c make test-primitives-hinge-antipode-standalone-cpu
nix develop --impure -c make test-primitives-hinge-antipode-standalone-gpu
```

The same Make lanes include a mixed partial-D6/Distance chain. A slack radial
maximum leaves the child free, while an active maximum reacts its outward load;
the independent legacy Distance projection does not override this coupled solve.
Two bodies under a disabled tree joint retain their frozen relative transform
and transmit child inertia/load to a floating parent. Focused native lanes are:

```sh
nix develop --impure -c make test-primitives-disabled-wiring-cpu
nix develop --impure -c make test-primitives-disabled-wiring-gpu
```

### Standalone authored narrow hinges

Full model upload derives `joint_limit_references` from authored f64 bounds:
high/low endpoints and full/half-angle sine/cosine references, 80 bytes per DOF.
It retains buffer identity at unchanged DOF count and refreshes after bound edits;
always use full model upload after changing raw limits. This adds one persistent
Featherstone binding (51 across five storage groups), no dispatch or per-step
metadata upload. Finite two-point hinges consume the half-angle references and
paired hard-frame removal. Raw regular XYZ rows use full-angle references and
the shared runtime paired-matrix geometry. Singular pitch-point rows compose
paired X/Z phase references and remove pitch/point frames with half-angle tails.
Prismatic/revolute and mixed translation row gaps use paired authored bounds;
scalar point classification uses authored motion flags, and endpoint publication
preserves low parts when compensation is enabled. Generic bounded-D6 activation
now includes nonstructural finite limits and velocity caps without requiring an
angular point axis; CPU activation and retained GPU row sizing share one predicate.
Structural translations publish paired authored endpoints; fully fixed D6 angular
motion publishes paired authored rotations and zero rates. Ball, Distance, D6 and
Free compensated quaternion advancement updates both high values and tails.
External runtime scalar-row ownership remains unchanged. Nonpoint near-pole pitch
rows now use paired bound references and retained sine/cosine values in raw and
runtime paths, including resolvable principal/alternate pitch tangents.
Rigorous Distance radial rows use a scaled paired vector norm and authored bound
tails; narrow radial intervals no longer collapse to a rounded equality. Ball
full-angular points now publish paired authored rotations and zero incoming
forbidden rates, matching CPU full-point semantics. Partial Ball angular limits
and Cartesian caps use coupled angular rows at quaternion/DOF offset zero; all
three Cartesian columns remain active unless all three angular axes are points.
CPU/GPU body-rate publication, shared GPU mass/Jacobian response and momentum
metadata follow that same structural rule. Public runtime external row export
includes Ball bounds/caps without changing rigid external Ball's cone-only limit
contract. Default standalone Distance also uses coupled
anchor effective mass, without requiring a D6 activator. Unbounded/slack radial
motion stays free, independent minimum/maximum bounds are unilateral, and zero
length uses three coincident-anchor rows while retaining rotational freedom.
Explicit `with_rigorous_distance(false)` retains standalone position clamping;
that approximation is not the production live-runtime distance path. Public
runtime external constraint ownership is unchanged. Retained row capacity follows
authored bounds/caps; no new binding, buffer, dispatch or per-step metadata upload.

Prepare and explicitly seed uploaded host state before initial integration:

```rust
solver.prepare_gpu_coordinate_compensation(&[&initial, &next])?;
solver.seed_gpu_coordinate_compensation(&initial)?;
```

Preparation still zeros residues. Seeding initializes quaternion/coordinate and
body-translation low parts from current host data; it rejects stale device-owned
host mirrors or host edits not uploaded. Do not reseed evolving device state from
old host data. Use an explicit import/edit boundary for a new seed/reset.

```sh
nix develop --impure -c make test-primitives-limit-reference-cpu
nix develop --impure -c make test-primitives-angular-narrow-standalone-cpu
nix develop --impure -c make test-primitives-angular-narrow-standalone-gpu
nix develop --impure -c make test-primitives-scalar-precision-cpu
nix develop --impure -c make test-primitives-scalar-precision-gpu
nix develop --impure -c make test-primitives-angular-regular-standalone-cpu
nix develop --impure -c make test-primitives-angular-regular-standalone-gpu
nix develop --impure -c make test-primitives-angular-phase-standalone-cpu
nix develop --impure -c make test-primitives-angular-phase-standalone-gpu
nix develop --impure -c make test-primitives-d6-activation-cpu
nix develop --impure -c make test-primitives-d6-interval-standalone-cpu
nix develop --impure -c make test-primitives-d6-interval-standalone-gpu
nix develop --impure -c make test-primitives-d6-interval-runtime-cpu
nix develop --impure -c make test-primitives-d6-interval-runtime-gpu
nix develop --impure -c make test-primitives-d6-point-publication-cpu
nix develop --impure -c make test-primitives-d6-point-publication-gpu
nix develop --impure -c make test-primitives-pitch-pole-cpu
nix develop --impure -c make test-primitives-pitch-pole-gpu
nix develop --impure -c make test-primitives-radial-precision-cpu
nix develop --impure -c make test-primitives-radial-precision-gpu
nix develop --impure -c make test-primitives-distance-activation-cpu
nix develop --impure -c make test-primitives-ball-point-publication-cpu
nix develop --impure -c make test-primitives-ball-point-publication-gpu
nix develop --impure -c make test-primitives-quaternion-motion-cpu
nix develop --impure -c make test-primitives-quaternion-motion-gpu
```

The 32-case dt=5e-5, width=1e-8 full-tensor/COM/frame/reversed/lower-upper/
inward-outward GPU fixture passes its independent 1% angular-acceleration oracle
with maximum error 8.95e-7 rad/s^2. Distinct authored endpoints cast identically
to f32. Strengthened retained-coordinate checks also pass: no bound violation at
1e-12 and hard-axis quaternion components below 1e-10. Reference refresh/resize,
retained buffer identity and seeding error-boundary checks pass. This fixture's
free axis is Y; wider all-axis/sustained/contact/floating acceptance remains open.
The unseeded antipodal fault is fixed: paired arithmetic now has rounding scratch
for every joint coordinate, not a four-byte fallback. That existing fallback
grows to 4*max(coordinate_count,1) bytes, without a new buffer/dispatch. Signed
bilateral rows remain active through the box solve while unilateral impulses
remain nonnegative. Both 72-case antipodal modes pass in 36.91 seconds. The
strengthened narrow test passes in 17.94 seconds, maximum acceleration error
9.43e-7 rad/s^2 and zero retained bound violations. Historical all-index internal
rounding acceptance before the scratch fix must not be assumed from body-only
physical passes; the corrected indexing is covered by these fresh checks.

Fresh standalone additions pass on RTX 4080/Vulkan/NVIDIA 595.91.07:
24 scalar cases (51.44 seconds, maximum acceleration error 1.93e-7), 128 regular
XYZ point cases (29.20 seconds, maximum acceleration error 1.07e-6), and 64 narrow
singular-phase cases (12.84 seconds). All use independent 1% acceleration gates;
scalar and phase retained bounds are checked at 1e-12. Scalar coverage includes
both joint kinds, full tensor/COM/frames, actual points and width=1e-8 intervals,
lower/upper endpoints/interiors and inward/outward effort. Regular points cover
principal/alternate charts, full tensor/COM/frames and reverse orientation at
dt=1e-4. Phase cases use width=1e-9, dt=1e-5, both poles, +/-3pi/2 winding and
negative quaternions; pitch transverse components are checked at 1e-10. Matching
CPU fixtures, shader wiring and strict Clippy pass. These are first-step proofs,
not full sustained/contact/floating/mutation or performance certification.

Bounded D6 without angular point locks also passes the independent anchored
full-inertia oracle: 192 regular width=1e-9 intervals plus 192 cap-only cases at
dt=1e-5 on each raw/public CPU/GPU route. The actual RTX 4080 raw lane finishes
in 34.31 seconds, maximum angular acceleration error 3.92e-5 rad/s^2 under the
unchanged 1% gate. Carried regular interval violations stay below 8.89e-16 with
a 1e-12 gate; cap-zero generalized rates stay below 1e-9. The public GPU runtime
finishes in 35.48 seconds; maximum reaction force/torque errors are 8.24e-5 N /
2.43e-4 N*m under independent 1% gates. Both principal/alternate charts, all
angular axes, full COM/tensor/frames, forward/reverse, lower/upper, negative
quaternions and inward/outward loads are covered. These regular interval/cap
cases do not substitute for contact/floating/sustained
acceptance. Corresponding raw/public CPU checks finish in 9.45/9.49 seconds;
pure activation, shader validation and strict Clippy pass.

Fresh publication and near-pole evidence (2026-09-30): the 96-case, four-step
CPU/GPU point fixture publishes exact translations/rotations and zero locked rates,
including perturbed starts, negative quaternions, authored frames and full bound
edits after two steps. GPU finishes in 15.64 seconds. Strengthened raw narrow-hinge
GPU checks also prove actual retained rotation advancement at 1e-12, rather than
only bound containment, in 14.11 seconds. Earlier quaternion-bound checks did not
detect stale tails and must not be interpreted as sustained-motion acceptance.

The 64-case nonpoint near-pole pitch fixture uses width=1e-9, dt=1e-5 and offsets
of +/-1e-7 about both poles, principal/alternate charts, full COM/tensor/frames,
forward/reverse, negative quaternions and both load directions. Raw/public CPU
checks pass in 0.58 seconds combined; actual GPU raw/public checks pass in 35.85
seconds combined. Raw acceleration and public force/torque use independent 1%
gates; retained pitch bounds and actual quaternion advancement use 1e-12 gates.
This is focused near-pole acceptance, not exact-pole nonpoint recovery, all-axis
sustained/contact/floating certification, or the required performance matrix.

The narrow standalone targets now also exercise eight steps without reseeding:
inward loads reach the opposite endpoint and hold; outward loads stay at their
initial stop. The two tests pass CPU/GPU in 0.15/27.98 seconds; GPU maximum final
endpoint error is 3.56e-15 under a 1e-12 gate and angular rate 1.47e-10 rad/s
under a 1e-7 gate. This proves sustained free-Y narrow motion, not all-axis,
moving-parent, floating/contact-loaded or full numerical certification.

Mixed Distance radial precision has 48 independent full-COM/off-diagonal-inertia
cases: points, width=1e-8 intervals with identical f32 endpoint casts,
lower/upper/interior starts, both load directions, axial/nonaxial displacement,
authored frames and negative quaternions. The cap-constrained D6 parent activates
the coupled solver; the oracle derives anchor/COM acceleration and angular
response from the world inertia and off-centre radial reaction, not another
backend. CPU checks plus the slack/maximum/drift regression pass in 0.01 seconds;
actual GPU counterparts pass in 28.47 seconds. Independent acceleration gates
remain 1%, and retained position/rotation/bounds gates remain 1e-12. Maximum GPU
errors are 4.30e-5 m/s^2 / 1.91e-5 rad/s^2, carried position 6.23e-15,
rotation 1.41e-15 rad and radial violation 6.00e-15. This proves first-step mixed
radial response/Distance quaternion motion, not long-run radial drift, legacy
standalone projection, mutation/contact-loaded or full performance acceptance.

The radial precision targets additionally exercise the same 48-case independent
COM/inertia oracle with standalone Distance and no D6 parent, plus 32 unbounded,
slack minimum-only/maximum-only/interval, boundary inward/outward, nonzero equality
and zero-equality cases under authored frames and free nonzero angular velocity.
CPU selected counts are four tests (including the existing mixed/slack/drift
regression); activation has a separate pure unit check. Four matching actual GPU
tests pass in 66.85 seconds. Semantic maximum retained position error is 2.67e-13,
rotation error 4.55e-16 and velocity error 2.66e-8 under unchanged gates. Both
200-step CPU legacy distance companions also pass with zero hanging-length error.
Long-horizon public-runtime acceptance, all GPU regressions and performance remain
separate requirements; these focused additions do not certify the full plan.

Ball full-point publication now has the same 96-case, four-step/edit fixture as
D6, with incoming rates and valid/perturbed quaternion starts, full COM/inertia
and authored frames. CPU passes; actual GPU Ball/D6 checks pass in 16.24/16.22
seconds, retained rotation error <=3.44e-15 (gate 1e-12), forbidden generalized
rates exactly zero and body rates within 1e-9. A separate 64-case, eight-step
constant-rate fixture proves Ball/Distance/Free/D6 retained quaternion advancement
for X/Y/Z/nonaxial rates, authored frames and negative quaternions, without
reseeding evolving state. CPU/GPU pass in 0.01/16.25 seconds; GPU maximum retained
rotation error 5.30e-15 (gate 1e-12). Its spherical inertia/zero COM is an exact
constant-rate oracle, not a substitute for loaded/contact/floating acceptance.

`test-primitives-ball-angular-cpu/gpu` now checks 384 independent raw Ball cases
and 384 public-runtime cases per backend: one/two angular points, all-axis width
1e-9 intervals and Cartesian zero caps, principal/alternate pitch, negative
quaternions, authored frames, off-centre COM and full off-diagonal inertia.
The independent anchored-inertia oracle gates COM/angular accelerations at 1%;
public reactions also have a 1% analytic gate. Raw retained quaternion motion and
Euler violations retain the 1e-12 gate; zero-cap residuals retain 1e-9. Public GPU
host snapshots are rounded f32 and are not used as 1e-12 retained-coordinate proof.
CPU passes two selected tests in 5.99 seconds plus the activation unit. Matching
actual RTX 4080/Vulkan/NVIDIA 595.91.07 GPU tests pass in 67.88 seconds. Raw GPU
maximum rotation error is 6.46e-15 rad, Euler violation 6.22e-15 rad, angular
acceleration error 5.77e-5 rad/s^2 and COM error 5.06e-5 m/s^2. Public GPU maximum
reaction force/torque errors are 9.87e-5 N / 3.31e-4 Nm. Logs are
`primitives-ball-angular-runtime-cpu-corrected.log` and
`primitives-ball-angular-gpu-final.log`. Earlier failed/diagnostic logs remain.
This is first-step tree response/reaction proof, not sustained/contact/floating,
redundant-row, mutation/breakage, full-plan numerical or performance acceptance.

These low-level disabled checks cover six joint kinds with fixed/floating roots;
they do not replace runtime enable/disable, detachment or reaction acceptance.

### Paired chain frames and sustained all-axis motion

`test-primitives-frame-reference-gpu` validates four seven-body mixed chains and
two all-fixed/zero-coordinate chains on hardware GPU. Three authored frame
uploads, reversed/disabled joints, signed quaternion tails and all four revolute
half-angle quadrants pass retained position/paired-component gates of 1e-12.
Only initial state is seeded; frame uploads do not reseed device poses. Worst
position/component errors are 1.23e-13/1.47e-14 (90.64 seconds). This is retained
kinematic/mutation evidence, not dynamic/contact acceptance.

Shared axis-angle FK now evaluates argument reduction and trigonometric Horner
polynomials using paired scalar arithmetic. Shared quaternion FK explicitly
rounds two_sum's error tail, matching Featherstone's retained arithmetic boundary.
No new buffers are introduced by those arithmetic repairs. Their extra private
work still requires the full performance matrix.

`test-primitives-all-axis-sustained-cpu/gpu` retains the original 36 simple D6
cases and adds 108 full-COM/off-diagonal-inertia cases at 1/17/83 kg. Every axis
is free/bounded/locked under both effort signs, with rotated and offset frames.
Independent projected anchored inertia sizes efforts so all bounded axes reach
their stops. The one-second, 240-frame/two-substep trajectory checks pose and COM
velocity against its analytic oracle. CPU/GPU peaks are 1.83e-7/1.65e-5 pose and
1.24e-7/4.02e-5 rate, under unchanged 1e-6/1e-3 gates (2.54/60.62 seconds).
This does not replace floating/contact trajectories or the full 10-second,
three-step-size numerical matrix.

Runtime scalar projection stops at an exactly bit-identical full sweep, then
applies its original finite/KKT/cap publication checks; physical tolerances and
row domains are unchanged. The worst-case changing-iterate bound remains 32768.

### Redundant point and zero-cap rows

`test-primitives-redundant-limits-cpu/gpu` includes independent Ball/D6/Distance
oracles: 144 loaded cases held for eight steps and 18 cases with nonzero permitted
motion. Raw CPU and hardware GPU pass; the GPU pair takes 36.44 seconds on RTX
4080/Vulkan/NVIDIA 595.91.07. Zero joint/body rates are exact in the hold fixture;
retained coordinate error is <=1.12e-16. Permitted-motion acceleration gates are
1%, and retained coordinate/rate gates are 1e-12/1e-9. No fixture gates relaxed.

`test-primitives-redundant-runtime-cpu/gpu` exercises actual RigidScene/RigidWorld.
Runtime Distance is an external radial link, not a Cartesian-axis control joint;
its cap cases use an explicit free-axis D6 tree plus the external Distance link
at matching frames. Holding reactions sum the physical contributions at their
shared anchor. Scene wrenches persist, so tests reset before assigning each load.
CPU's three selected fixtures pass in 2.98 seconds: eight-step loaded holds,
permitted motion, and an incompatible radial point/cap problem rejected without
advancing state or the completed-step counter. A separate equality-arithmetic
unit passes. Equality geometry below a scale-derived f64 roundoff bound is
resolved before division by dt; finite unilateral gaps and resolvable errors
are retained, and the common solver's convergence/allocation contract is unchanged.

The original 18-case public-runtime GPU motion fixture passes in 25.22 seconds,
and the 144-case eight-step loaded/reaction fixture passes in 40.61 seconds on
RTX 4080/Vulkan/NVIDIA 595.91.07. Exact zero-cap rates remain zero. Independent
motion/reaction gates are unchanged. The demonstrated radial/cap fault arose
when runtime FK refresh reconstructed body positions using rounded authored
frames while radial anchors used authored pairs. Compensated FK now retains
frame/axis tails and propagates paired rotations through the articulation.
The clean staged refresh regression measures 4.44e-15 m retained position error.

Frame/axis references use 80 bytes per joint, refreshed on model upload rather
than per step. Featherstone binds 52 buffers; paired FK publication adds private
work but no extra dispatch or body storage. Existing paired radial quaternion-FK
resources/dispatches and bounded rank-deficient solver work still need full
performance-matrix evidence. External radial/cone/loop rows now solve scalar
caps/bounds together even without a partial Ball/D6 angular lock; the selected
significantly incompatible radial/cap GPU fixture rejects with status 512 without
advancing pose, rates, completed steps or break events (25.65 seconds).

These selected gates do not certify the full P5 numerical/integrated scope or
complete P6/performance acceptance. No CPU substitution or hidden compliance is
claimed. Fresh evidence is in primitives-runtime-*-fk-*.log and earlier
primitives-redundant-* logs under
`benchmarks/primitives/impact-history-gpu-2026-09-29/`.

## Antipodal cone recovery

The cone evaluator now accepts constrained 180-degree swing using a deterministic
one-sided chart; see `MOLLA_CONE_GEOMETRY.md`. A disabled/pi-wide extent also
provides a short return-to-pole branch instead of a remote radial stop.
Retained GPU rows retire and change equality mode with the live chart.

`primitives-antipodal-retained-modes-gpu.log` records GPU recovery evidence;
the CPU counterpart is `primitives-antipodal-live-cpu.log`. Tree, rigid external
and compliant external routes recover in 32 short frames. Admissible-pole
checks preserve twist/free swing without angular-rate
spikes. The GPU counterpart used RTX 4080/Vulkan/NVIDIA 595.91.07 and finished
within its 120-second process cap. General rotated/floating/moving, zero-extent,
contact and sustained energy/scale acceptance remain open.
