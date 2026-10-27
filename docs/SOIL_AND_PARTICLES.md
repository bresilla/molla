# Soil and particles: performance-first implementation plan

**Status:** implementation in progress; MPM and DEM binned terrain exports verified on RTX 4080, full-plan acceptance incomplete.

**Confirmed acceptance target:** RTX 4080, one tractor plus implement, 50k active
soil particles, 60 FPS, and soil work at most 4 ms per 1/60 second of simulated
time. These are requirements, not achieved performance.

**Calibration scope:** the user has no measured soil or traction data. The initial
release must be labelled **physically uncalibrated**, even when numerical checks
pass. Analytic fixtures and numerical agreement do not establish real-soil
traction, rutting or draft-force accuracy; measured-data calibration is a later
phase, not a prerequisite for implementing the initial release.

**Fitted ADMM response storage (2026-10-15; current experimental runtime):** A
test-only actual-workload trace found 251–283 contacts, overflowing the previous
256-contact cache in **59/102 substeps**. Core/pivot/rate shared arrays now fit
the admitted DOF width, retaining the 32-entry core row stride and all arithmetic.
Ten DOFs admit **298 contacts in 38320 aligned shared bytes**, within the unchanged
38400-byte requirement. All 32 width/layout checks and 15 nonzero boundary cases
pass, including exact control and poison recovery across cache thresholds.

Both new actual native runs verify **102/102 cache hits** with the same active
count sequence. All **176 state artifacts remain exact**; rollback passes and
hot allocations, transfers and dispatches match the preceding runtime. Observed
paired solve ratios are **0.920740 / 0.928098** under GPU contention. Warm soil
means remain **949.882 / 978.531 ms**, far above 4 ms. These short, instrumented
runs are not uncontended, percentile, long-horizon or full-plan acceptance.
The trace is test-only, preallocated, uses existing submissions and reads once
after the measured frames. Evidence:
`benchmarks/soil/2026-10-15-admm-fitted-native/SUMMARY.md`.

**Initial ADMM resident responses (2026-10-15; earlier runtime increment):** ADMM retained
active response coordinates in workgroup memory when the full set fits, while
cooperatively staging mutable points. Ten DOFs admit 256 contacts within 36256
aligned shared bytes; overflow uses packed tiled staging. Arithmetic, contact
order, iteration count, device admission and independent publication are unchanged.
Standalone/default Newton is unchanged. Explicit original-source controls remain.

Twenty-eight CPU checks and both captured packets pass exact five-buffer and
boundary/recovery gates. Integrated articulated rollback passes in 4.61 s; four
native 50k tractor runs complete in 70.28 s with **176 state artifacts exact** and
zero hot allocations in all 12 frames. This is short-run integration evidence,
not full acceptance. Contended warm soil means are **1152.911 / 1199.696 ms** for
the candidate versus **1187.606 / 1147.902 ms** for controls: no reliable integrated
speedup, and nowhere near the 4 ms requirement. The mutable-state cache and the
direct-global-point response variant were not promoted. Evidence:
`benchmarks/soil/2026-10-15-admm-response-native/SUMMARY.md`.

**ADMM on-chip/layout probes (2026-10-15; test-only):** Cached invariant contact
terms preserve captured outputs and boundary recovery, with approximately 2.74%
lower paired component totals in the initial run. A separate padded response-row
layout passes 26 CPU checks and the captured/boundary GPU batch (33.48 s), but
paired timing changes of -1.1183% / +1.3892% under contention are inconclusive.
Neither candidate is promoted; neither meets or establishes full-frame acceptance.
Evidence: `benchmarks/soil/2026-10-15-admm-padded-staging/SUMMARY.md` and
`benchmarks/soil/2026-10-15-admm-onchip-cache/SUMMARY.md`.

**ADMM publication summation (2026-10-15; experimental):** Guarded active ADMM
publication now uses magnitude-sorted FastTwoSum with a canonical positive-zero
residual, reducing atomic rounding round trips per compensated sum from **six to
three**. All independent numerical/physical, energy/momentum, wall, active-index
and failure checks remain; standalone/default Newton publication is unchanged.
Forty-two CPU checks, **99344 GPU arithmetic cases per captured packet**, existing
failure/analytic fixtures and exact five-buffer controls pass in a **27.63 s** batch.

The actual native 50k follow-up completes four AB/BA runs in **60.32 s**, with all
**176 state binaries byte-exact** against the prior cooperative baseline and
articulated rollback passing separately in **3.50 s**. **No consistent speedup is
established.** Component paired changes are +314.209% / -22.290%; actual warm soil
ratios are 1.017 / 0.446, but unchanged solve and machine work also roughly halve
in the second block. Pre-existing 100% GPU utilization remains a confounder.
Candidate warm soil means are **660.704 / 662.755 ms**, still far from 4 ms.
[Component evidence](benchmarks/soil/2026-10-15-admm-sorted-publication/SUMMARY.md);
[actual workload and complete timing scope](benchmarks/soil/2026-10-15-admm-sorted-native/SUMMARY.md).

**Cooperative ADMM preparation (2026-10-15; explicit profile):** The actual ADMM
tractor can use the existing cooperative Cholesky/response preparation through
`row-parallel-newton-preparation`. Backend identity is now exposed by the prepared
world and recorded/asserted by the actual-workload harness. Selected preparation
execution limits are checked before ADMM parent allocations; defaults and strict
feature precedence remain unchanged.

Fresh dense/cooperative/cooperative/dense native runs complete in **54.39 s**.
All **176 state binaries** match the pre-change dense baseline exactly through
three frames per run; the cooperative world rollback/profile and budget checks
also pass. Observed warm soil means fall from **2049.948 / 1790.497 ms** to
**1475.688 / 1240.770 ms** (28.0 / 30.7% lower in the two blocks). Response
preparation falls from **627.237 / 495.844 ms** to **3.120 / 3.763 ms** per frame.
These are short, contended instrumented runs with pre-existing 100% GPU utilization,
not uncontended scaling, percentiles or **4 ms / 60 FPS acceptance**. Full-plan
acceptance remains incomplete; next target the solve/publication and machine costs.

The bounded diagnostic is available in the project environment as:
`make test-soil-admm-tractor RUN_WITH=nixVulkan SOIL_ADMM_PREPARATION=cooperative SOIL_ADMM_OUTPUT=/tmp/molla-admm-new`.
The output directory must not exist; `dense` selects the control and
`SOIL_ADMM_ARM=featherstone` selects the alternate prediction arm. This diagnostic
is not the full-plan acceptance suite. A stale generic CPU label expectation was
corrected and its initial failed build preserved; no numerical gate changed.
[Evidence and complete scope](benchmarks/soil/2026-10-15-admm-cooperative-preparation/SUMMARY.md).

**ADMM width-sized staging (2026-10-15; experimental):** Resident ADMM now sizes
its staged-response array to the machine width, retaining at least four rows for
the alternate helper. For 10 DOFs, the staging arrays shrink from **33792 to
11264 bytes**, with all arithmetic, global budgets and guards unchanged. All 32
widths validate; 23 CPU tests and a frozen **36.13 s** GPU replay pass. Both
captured packets match all five full-capacity control buffers exactly across nine
runs per mode, including the existing failure and analytic fixtures. **No speedup
is established:** paired total-time upper-median changes are +3.099% / +1.202%
with pre-existing 100% GPU utilization. This is a workgroup-footprint improvement,
not 4 ms acceptance or a new full-frame result.
[Evidence](benchmarks/soil/2026-10-15-admm-width-staging/SUMMARY.md).

**Actual-workload follow-up:** the changed kernel also completes the native 50k
tractor's three frames in a **15.19 s** batch. All **44 binaries**—initial and
three committed full soil checkpoints, public states and nine internal machine
buffers per sample—match the pre-change native run **byte-for-byte**. This does
not establish cross-arm full-state or longer-trajectory parity. Contended soil
times remain **2034.921 / 2016.901 / 2137.791 ms**, not 4 ms. A harness directory
creation error failed before GPU setup and remains separately sealed; only that
setup was corrected for the follow-up. Featherstone was not rerun here.
[Integration evidence](benchmarks/soil/2026-10-15-admm-width-native/SUMMARY.md).

**Actual 50k ADMM tractor execution (2026-10-15; experimental):**
Both native-prediction and paired-Featherstone ADMM constructors now complete
three actual 50k-particle frames, with the unchanged default material/seed,
1/60-second frame, 34 substeps, 4096 contact slots, 64 iterations, fixed-order P2G
and 300 N m wheel torques. Recorded particle positions/velocities and six-body
poses/velocities agree within **7.66001e-7**, passing the unchanged **2e-5** gate
through frame three. Full checkpoints and nine machine-buffer snapshots are
retained, but their complete internal-state equivalence and longer trajectories
are not claimed. Each successful frame observes zero allocations and 884 bytes
of completion readback; post-frame diagnostic readbacks are separate.

**Performance fails the target.** The two post-cold-frame soil samples are
**968.552 / 1031.901 ms** for paired Featherstone and **1850.066 / 1887.160 ms**
for native prediction, versus 4 ms. The GPU was already at **100% utilization**
before the batch, with Astrocraft/display workloads present; detailed profiling
was enabled. These are contended instrumented measurements, not clean comparative
solver timings or percentile acceptance. Contact preparation, solve and publication
dominate the measured soil intervals. The complete batch passes in **92.65 s**
under the unchanged 180-second cap. An earlier 70.38 s harness failure requested
a committed checkpoint at frame zero; it is sealed, and initial capture now uses
explicit raw test state while preserving public commitment guards.
[Evidence](benchmarks/soil/2026-10-15-admm-tractor-initial-state/SUMMARY.md).

**Articulated-world ADMM and exact rollback (2026-10-15; experimental):**
Added explicit ADMM world selection and tractor constructors for paired
Featherstone or native prediction, retaining Newton defaults. The native-prediction
world fixture now advances the six-body tractor geometry with **65 particles and
five loaded wheel/tine bindings**, three substeps per frame. Four measured frames
pass existing impulse, angular momentum and full soil-energy gates; control-edit
and injected GPU-fault rollback/retry pass, including exact soil and nine machine
buffers. This is a small zero-gravity integration fixture, **not the 50k field
workload or rendered/physical calibration acceptance**.

Integration exposed and fixed two issues: missing detailed ADMM preparation
timestamps, and paired rollback failing to restore scalar joint-rate mirrors.
Detailed profiling now retains the strict eight-stage contract; its optional
history scope is empty for ADMM. Failure-only GPU copies restore scalar high words
without new storage or normal-step dispatches. Twenty-eight CPU checks and direct
rate-lifecycle/world GPU checks pass. A later budget test exposed two stale
16-byte literals; only those expectations changed, and its **1.91 s** follow-up
passes for all seven coupling modes across dense/active/compact grids. Production
source hashes match the sealed successful world run, so unchanged world GPU work
was not repeated. All three earlier failed batches remain sealed and labelled
failed. Full 50k constructor execution, trajectory/performance comparisons and
**4 ms/60 FPS acceptance remain open**.
[Evidence](benchmarks/soil/2026-10-15-admm-world-budget-fix/SUMMARY.md).

**Guarded ADMM in prepared MPM coupling (2026-10-15; experimental):**
Added explicit `new_experimental_admm` construction, dense/active/compact grid
budget queries, paired machine-rate publication and separate preparation failure
access. Candidate enumeration/certification, particle-fault propagation, reaction
wrenches and coarse preparation/solve/publication profiling remain in the actual
MPM stepping path. Twenty-two CPU checks and a frozen **23.91 s** RTX 4080 batch
pass. A 65-particle/3-DOF/128-slot fixture advances four steps in dense, active,
compact and profiled-dense configurations; all 13 final buffers match exactly.
Maximum normalized momentum error is **9.27e-9** against the unchanged 1e-6 gate.
Candidate overflow and invalid mass prevent particle advancement; machine
publication rejects without changing rates/deltas; sticky faults, reset recovery
and an injected particle-fault bridge check pass. Successful steps observe zero
allocations/readbacks, not zero uploads. Existing captured and geometry checks
remain intact after splitting the private encode phases for profiling.
This is real MPM feedback, but **not articulated tractor-world integration**,
dense-contact/long-horizon evidence or 50k/4 ms acceptance. Newton defaults,
history, certificate and backend-specific failure schemas are unchanged.
[Evidence](benchmarks/soil/2026-10-15-admm-mpm-integration/SUMMARY.md).

**Analytic geometry-to-ADMM adapter (2026-10-15; experimental):**
Added `ExperimentalSoilAdmmGeometryGpu`, joining analytic body contacts, checked
GPU preparation, guarded ADMM and world-space velocity/impulse recovery. Explicit
capacity and byte-budget admission precede allocation; immutable geometry
programs are shared while instance buffers remain private. Twenty CPU checks
and one frozen **17.80 s** RTX 4080 batch pass. Six small analytic fixtures run
in both packet passes: empty contacts, spheres, a box and a cylinder, friction,
signed-wall constraints and sparse 65/4096-slot layouts with at most two active
contacts. Maximum independent physical error is **8.95e-8**, projected residual
**2.24e-8** and CPU-reference delta **4.19e-8**. Invalid coefficients/mass fail
closed, faults persist until reset and recovery/repeats are exact. Encode uses
eight dispatches (six when empty), with zero observed allocation/upload/readback.
The 4096-slot/3-DOF adapter owns **2621924 bytes**. This is component evidence,
not dense-contact coverage, MPM/tractor integration or a speed comparison.
Newton defaults and all full-frame acceptance gates remain unchanged.
[Evidence](benchmarks/soil/2026-10-15-admm-geometry-adapter/SUMMARY.md).

**Shared ADMM programs and GPU-prepared wall cases (2026-10-15):**
Resident ADMM instances now share immutable compute pipelines/layouts through the
engine-local cache, keyed by layout version and complete shader hash. Buffers,
uniforms, bind groups and publication remain private. Seventeen CPU contracts
and one frozen **20.97 s** RTX 4080 batch pass. Six analytic cases connect GPU
mass/response preparation directly to guarded ADMM publication: frictionless and
frictional no-wall/positive-X-wall cases, mirrored negative-X walls in 65 slots,
and rotated negative-Z walls in 4096 slots. Each has two active contacts and runs
in both captured-packet passes; these are **not dense 4096-active cases**.
Independent CPU residual/physical/public-output checks pass, repeat and reset
recovery are exact, invalid mass fails closed, and shared programs preserve
instance isolation. Combined encode uses five dispatches with zero observed
allocations/uploads/readbacks. Existing captured exactness/failure gates remain.
This establishes reusable program sharing and small GPU-prepared wall coverage,
not measured construction speed, tractor integration, long trajectories or
4 ms/60 FPS acceptance. [Evidence](benchmarks/soil/2026-10-15-admm-prepared-walls/SUMMARY.md).

**Resident guarded ADMM API (2026-10-15; experimental, not integrated):**
Added `ExperimentalSoilAdmmGpu` with prepared GPU input bindings, explicit capacity
and owned-byte admission, persistent scratch, reset/encode methods and mandatory
active-index/physical/numerical publication checks. It selects the uncached
ordered solver mode, not the unpromoted fixed-contact cache. Supported admission
is 1–32 DOFs, 0–16384 contact slots and 1–64 iterations; larger/full-plan workloads
remain unresolved rather than silently truncated. Fifteen CPU contracts and a
frozen **13.88 s** RTX 4080 follow-up pass. Both captured cases match the replay's
five buffers exactly, with poisoned scratch/report recovery, upstream and NaN
contact rejection, fault persistence/reset, independent instances, mandatory
numerical rejection and empty-contact reuse. Encode observes zero allocations,
uploads or readbacks and two dispatches. Owned buffers total **1261748 bytes**
for 4096 slots/10 DOFs and **416 bytes** for zero slots/one DOF. The first GPU run
failed after **7.34 s** because an empty runtime-array binding was four bytes
instead of the required eight; the failure is sealed, sizing/admission corrected
and a Naga layout regression added. All report words now reset on cold start.
No simulator default, production certificate or performance acceptance changes.
[Evidence](benchmarks/soil/2026-10-15-admm-resident-empty-fix/SUMMARY.md).

**Independent ADMM numerical publication (2026-10-15; experimental):**
Added GPU-side absolute **1e-6** projected contact and signed-wall residual
certification to standalone publication, retaining all physical checks and
zero-on-failure behavior. The shared WGSL guard is byte-identical to the earlier
resident guard before standalone symbol substitution. Thirteen CPU contracts
and a frozen **2.21 s** RTX 4080 follow-up pass: valid ADMM outputs remain exact
against prior control, between guard modes and over nine repeats per mode.
Physical-only publication accepts the old early-stopped states; the numerical
guard rejects them with fault8, clears all rates/outputs, retains the fault until
reset, and restores valid ADMM publication exactly. One-iteration solves also
reject (independent residuals **1.31445 / 1.31200**). Initial verification stopped
after **1.51 s** on an auxiliary archived-status exactness assumption: the old
resident diagnostic scale differs by **three ULPs**, despite exact rates/output
buffers and fault fields. That failure remains sealed and is still recorded as
false; ADMM exactness and numerical/physical/public gates are unchanged.
No production promotion, wall-regime coverage or real-time acceptance follows.
[Evidence](benchmarks/soil/2026-10-15-admm-numerical-rejection/SUMMARY.md).

**Fixed-contact ADMM cache (2026-10-15; exact, not promoted):**
Added a GPU-owned 64-byte-per-slot cache for normalized normals, inverse-mass
penalties, restricted local weights and free relative velocities. Candidate mode
rebuilds it once after input validation, rather than repeating those calculations
inside all 64 iterations; both modes retain full-width ordered staging and
validated active-index publication. Twelve CPU contracts and one frozen
**2.59 s** RTX 4080 batch pass. All five buffers remain exact against prior
control, between modes and across nine repetitions; the cache also rebuilds
exactly after NaN poisoning, with inactive slots zeroed and independent f64 cache
checks below **1.79e-7** normalized error. The cache costs **262144 bytes** for
4096 slots and adds no dispatch. However paired total upper-median changes are
**+22.01% / +88.80%**, with candidate total upper medians
**10.169344 / 11.269120 ms per captured solve** under external GPU load. This is
not a speedup or production acceptance; keep the uncached path as control and
do not infer 4 ms/60 FPS acceptance. All old-public failures remain unchanged.
[Evidence](benchmarks/soil/2026-10-15-admm-fixed-contact-cache/SUMMARY.md).

**Validated active-index publication (2026-10-15; exact, experimental):**
Replaced the standalone publication probe's two serial full-slot machine
reductions with ascending active-index traversal. Independent full-input checks,
lane-owned contact/energy reductions and fail-closed clearing are unchanged.
The index list must match the independently counted positive-mass contacts,
contain only in-range active slots, and be strictly increasing. Eleven CPU
contracts and one frozen **2.89 s** RTX 4080 batch pass: both captured packets
retain exact five-buffer control/mode parity and nine exact repeats per mode;
seven corrupted-list cases per packet reject with zero rates/outputs and recover
exactly without rerunning the solver. Numerical and guarded-public gates pass;
the prior old-public failures remain visible. Publication paired upper-median
changes are **-55.04% / -36.70%**, with seven of eight pairs faster per packet,
but total-time outliers span **-82.10%..+363.51% / -74.61%..+287.99%** under
external GPU load. Candidate solve-plus-publication upper medians are
**5.725184 / 8.090624 ms per captured solve**, not per frame. No reliable overall
speedup, production promotion, 50k integration or 4 ms/60 FPS acceptance is
claimed. [Evidence](benchmarks/soil/2026-10-15-admm-active-publication/SUMMARY.md).

**Full-machine ordered staging (2026-10-15; exact, no reliable speed claim):**
Added an all-DOF staging path that removes repeated four-DOF tile passes while
preserving dot/add order. The probe explicitly requires **38,400 shared bytes**;
the tested RTX 4080 exposes **49,152**. Nine CPU contracts and a frozen **4.66 s**
GPU batch pass, preserving all five buffers and nine repeats per mode. Paired
total-time upper-median changes are **-8.71% / -7.15%**, but total upper medians
are **29.232 / 43.332 ms**, with outliers and one regression versus control.
Both layouts remain experimental; no reliable overall speedup or real-time claim.
[Full-width staging and retained timings](benchmarks/soil/2026-10-15-admm-gpu-wide-stage/SUMMARY.md).

**Ordered workgroup staging (2026-10-15; exact, timing inconclusive):**
Added cooperative operand staging for four machine DOFs and 64-contact tiles,
preserving serial dot-product accumulation order. Eight CPU contracts and both
captured GPU comparisons pass; all five buffers remain exact, with nine repeats
per mode. The **4.84 s** batch retains all timing samples. Paired total-time
upper-median changes are **-5.74% / -14.46%**, but wide outliers and conflicting
absolute medians prevent a reliable speedup claim. No production promotion or
real-time acceptance. [Evidence](benchmarks/soil/2026-10-15-admm-gpu-staged/SUMMARY.md).

**Deterministic GPU active traversal (2026-10-15; exact, still too slow):**
The ADMM probe now builds an ascending active-index list on GPU and uses it for
core assembly, machine reductions and contact iterations. Seven CPU contracts
pass. Both 251-of-4096 captured cases preserve all five buffers exactly against
the original FP32 control and between traversal modes; nine repeats per mode
are exact. A frozen **5.24 s** batch measures eight alternating pairs per packet.
Paired total-time upper-median changes are **-67.21% / -65.51%** against this
prototype's full-slot scan, not against Newton. Active solve/publication total
upper medians are still **31.785 / 34.316 ms**; solve-only medians are about
**15.13 ms**. Timing excludes preparation/uploads/readbacks/frame work and has
large variation under current GPU load. No real-time or production acceptance.
[Exact traversal and retained raw timings](benchmarks/soil/2026-10-15-admm-gpu-active/SUMMARY.md).

**GPU mass-metric ADMM (2026-10-15; captured numerical admission passes):**
Implemented the restricted Woodbury/contact split on GPU: one 10-DOF pivoted
factorization, 64 mass-scaled iterations and independent GPU publication. No CPU
solve/seed enters the GPU execution path. Six CPU contracts pass. One frozen GPU
batch completes in **2.44 s** including setup and diagnostic readbacks; this is
**not solve timing**. Both 251-active/4096-slot cases pass physical/projected
residual **1e-6**, CPU impulse agreement **2e-5**, and full guarded-Newton public
agreement **2e-5** (maximum **9.11e-6**). All five buffers repeat exactly three
times. Old-output differences near **0.0032** remain failed and preserved.
This is a test-only FP32 probe, not production or integrated 50k acceptance.
[GPU implementation and numerical evidence](benchmarks/soil/2026-10-15-admm-gpu-fp32/SUMMARY.md).

**Explicit guard lifecycle (2026-10-15; test-only lifecycle passes):**
The numerical guard is now explicitly enabled and read back during lifecycle
checks; compiling its branch alone is not counted as activation. Eight CPU
contracts pass. One frozen RTX 4080 run completes in **72.33 s**, covering sparse
65-of-4096 contacts, poisoned scratch, upstream/contact faults and recovery,
bounded iteration failure, private buffer/shared pipeline ownership, empty and
invalid-mass scenes, frictionless/frictional and wall fixtures. Independent
projected residual checks pass at **1e-6** for both internal paired impulses and
represented public impulses. A preflight-only missing COPY_SRC usage was preserved
and fixed in test allocations before the GPU run. Production behavior/certificate
remain unchanged; prior old-output parity failures and the 4 ms target remain open.
[Activation, lifecycle and preflight evidence](benchmarks/soil/2026-10-15-newton-guard-lifecycle-readback/SUMMARY.md).

**GPU numerical stopping guard (2026-10-15; test-only, old-output parity fails):**
Added an unsmoothed projected-impulse residual guard alongside the existing
publication checks. Seven CPU contracts pass. One frozen RTX 4080 replay completes
in **67.98 s** under the unchanged 90-second cap. Both retained-seed controls remain
archive-exact; control and guarded modes each repeat all five buffers exactly
three times. The guard takes **12 rather than 8 Newton steps**, reduces projected
residual from about **8.94e-5 to 2.55e-7**, and agrees with the independent ADMM root
within **2.55e-7**. Old-output public differences are **0.003183 / 0.003203**, above
**2e-5**; those failures remain explicit. No production certificate/default change,
archive replacement or performance claim. Lifecycle coverage and a versioned
stopping contract remain prerequisites to promotion.
[GPU guard, controls and retained failures](benchmarks/soil/2026-10-15-newton-numerical-guard/SUMMARY.md).

**Independent ADMM reference audit (2026-10-15; discrepancy located):**
An independent projected-contact Newton step from each archived state agrees with
the mass-scaled ADMM result within **5.84e-10**. The largest archive mismatch is
slot 73: it retains about **8.94e-5** normal impulse despite positive separation;
ADMM removes that impulse. The archive passes its normalized product-based
physical check but has projected-contact residual about **8.94e-5**. This explains
the comparison failure; it does **not** waive the frozen **2e-5** requirement or
authorize replacing the archive. Five CPU contracts and both bounded diagnostics
pass; no GPU/production promotion or real-time claim.
[Independent residual and Newton evidence](benchmarks/soil/2026-10-15-admm-reference-audit/SUMMARY.md).

**Mass-scaled ADMM (2026-10-15; physical pass, comparison failure):**
The test-only restricted split now uses each contact's inverse soil mass as its
metric instead of one global penalty. Four CPU contracts pass, including analytic
frictionless/sliding/sticking contacts spanning twelve orders of mass. Both frozen
64-step replays pass independent and prepared physical bounds (errors below
**1.7e-11**), but archived impulse differences remain about **8.94e-5** and rate
differences **2.56e-5**, above **2e-5**. This is not full mechanism acceptance and
does not authorize GPU or production promotion. Diagnose the reference discrepancy
before more optimization; do not tune iterations to reproduce an approximate state.
[Mass-scaled algebra and failed comparisons](benchmarks/soil/2026-10-15-admm-mass-metric/SUMMARY.md).

**Inactive-wall elimination (2026-10-15; CPU-only mechanism gate failed):**
The captured packets have no active wall coordinates. Eliminating their fixed-zero
impulses reduces the split from 1,506 to 753 coordinates while preserving the
10-DOF Woodbury core. Three CPU contracts pass, including every signed/absent
wall-axis combination and asymmetric response algebra. At the unchanged 64-step
limit, physical errors improve to **2.958e-6 / 3.009e-6** but still exceed **1e-6**;
impulse and rate comparisons also fail. No GPU promotion or speed claim.
[Restricted algebra and retained failures](benchmarks/soil/2026-10-15-admm-wall-elimination/SUMMARY.md).

**Low-rank ADMM contact reference (2026-10-15; CPU-only mechanism gate failed):**
Added a structurally different hard-contact NCP reference: one shifted response
factorization, Coulomb/wall projections and De Saxce correction, rather than a
new per-iteration Fischer Jacobian. A block-local Woodbury solve reduces the
current packets' 1,506 contact/wall unknowns to a 10-DOF factored core. Both packets
have **251 active contacts**; the report value 24 is a linear-iteration count.
Three CPU algebra/projection tests pass, including 1e-12 comparisons with the full
shifted operator and an asymmetric prepared response. The original dense
128-contact capacity failure is preserved separately.
Both authenticated cold-start replays complete and repeat exactly, but after the
fixed 64 steps physical errors are **6.876e-6 / 6.956e-6**, above **1e-6**; impulse
differences are about **6.8e-4**, above **2e-5**. No GPU implementation, production
fallback, increased iteration/time budget or real-time gain is claimed. This
changes the candidate algorithm, not the contact law or time step.
[Research, algebra and retained failed gates](benchmarks/soil/2026-10-15-admm-low-rank/SUMMARY.md).

**Batched selected index paths (2026-10-15; production S07 increment):**
Selected publications now traverse shared radix paths once per batch instead of
updating each key independently. Unchanged subtrees remain referenced; successful
updater writes are all reachable from the final root. Patch scratch depends on the
selected edit count and bounded key depth, not the full field. File formats and
publication/recovery contracts are unchanged; mixed insert/delete batches may
produce a different valid index shape while preserving exact archive contents.
For 64 updates each dirtying four tiles in a 4,096-tile field, same-build serial
versus batched logical writes are **988,672 / 358,912 bytes**. The older one-dirty-tile
fixture remains at 269,056 bytes; these are different workloads. Ninety-seven CPU
checks and eighteen frozen RTX 4080 live suites pass (GPU **27.02 s**). This is
storage I/O evidence, not solver-frame speed, physical sleeping or 4 ms acceptance.
[Batch traversal, exact-state comparisons and recovery](benchmarks/soil/2026-10-15-store-batched-paths/SUMMARY.md).

**Streaming index construction (2026-10-15; production S07 increment):**
Migration and full replacement now construct the immutable radix index in one
sorted pass, holding only a bounded leaf and right spine. Every successful page
write belongs to the final root; the builder issues no page-read requests (the
file backend still checks existing content-addressed files before reusing them).
The resulting root is byte-identical to repeated sorted insertion; file formats
and publication/recovery semantics are unchanged. Selected path edits
were unchanged at this checkpoint; the later batch increment above improves them.
On the same 4,096-tile integrated workload, one-time migration writes fall from
**6,873,544 to 272,648 bytes** (about 25.2x less); the following 64 selected updates
still write **269,056 bytes**. These are logical I/O counts, not latency or physical
solver performance. Ninety-two CPU checks pass; eighteen frozen RTX 4080 lifecycle
suites pass in **26.72 s**, including fault/reopen and all store families.
Full real-time, support/sleep/wake and physical traversal gates remain open.
[Bounded builder, root parity and recovery evidence](benchmarks/soil/2026-10-15-store-sorted-builder/SUMMARY.md).

**True-residual pass fusion (2026-10-15; test-only, performance unaccepted):**
The direction operator can now form the rounded residual and its per-lane norm
maximum in one traversal. The original operator rounding, residual addition,
contact/component ordering, norm reduction and scaled-square pass remain intact.
This removes two contact traversals and one storage barrier without new resources
or changed physics/solver budgets. CPU/source/compiler and feature checks pass.
One frozen RTX 4080 replay completes in **75.87 s** under the 90-second cap;
both current-certificate packets match all five archived baseline buffers exactly,
with 16 exact timed repeats each and unchanged physical/public gates.
Paired upper-median timing changes are **-0.335% / +5.062%**, with approximately
-75% to +300% pair outliers and unrelated GPU workloads present. This does **not**
establish speedup; no production promotion or unchanged timing retry follows.
The separate frozen lifecycle/fault check passes in **76.85 s**, also below 90 s. Full
50k/4 ms/60 FPS, trajectory, support and agricultural acceptance remain open.
[Frozen implementation and evidence](benchmarks/soil/2026-10-15-newton-residual-fusion/SUMMARY.md).

**Integrated incremental root publication (2026-10-15):**
Raw/framed/canonical stores now support explicit `upgrade_index()` from v1 to an
immutable-page v2 index. Open validates both formats; changed v2 selections publish
only affected paths and a bounded root HEAD. Migration invalidates old reads, while
already-upgraded calls are no-ops. New stores remain v1 unless explicitly upgraded;
old binaries cannot read v2. Page installation, directory/root synchronization,
atomic rename, poisoning/reopen and validated collection preserve the existing
publication protocol. Reopen validates every reachable index page.
For the same 64 four-particle updates over 4,096 tiles, logical writes fall from
**15,781,632 to 269,056 bytes**; the original migration implementation wrote
**6,873,544 bytes**, reduced by the streaming-builder increment above.
These include changed payloads, newly installed index pages and root publication,
not filesystem overhead. Exact final archive parity passes. Eighty-nine CPU checks
and eighteen frozen RTX 4080 lifecycle suites pass in **26.84 s**, including upgraded
raw/framed/canonical live owners, barrier faults and process-exit recovery. This is
storage correctness/I/O evidence, not hardware power-loss or 4 ms/60 FPS acceptance.
[Integrated root evidence](benchmarks/soil/2026-10-15-store-incremental-root/SUMMARY.md).

**Copy-on-write index pages (2026-10-15; initial test-only candidate):**
Implemented immutable, hash-addressed radix pages with at most 32 entries per leaf,
a 2,048-byte page-read cap, checked subtree summaries and strictly increasing split
positions over signed 192-bit tile keys. Insert/update/delete rewrite only the
changed path; unchanged edits write nothing and old roots remain readable. A single
edit at 64/1,024/4,096 tiles writes **2,128/2,896/3,280 page bytes**, respectively,
instead of the v1 whole-index rewrite. These exclude root publication, filesystem
overhead and payload writes; no integrated speedup is claimed.
Eighty-one CPU checks pass, including 4,096 deterministic reference-matched edits,
full deletion, old roots, maximum 193-page paths, malformed pages, injected page-write
failures and temporary-file reopen/corruption. This checkpoint was test-only; the
subsequent increment above integrates root publication, migration and live workers.
[Candidate design and evidence](benchmarks/soil/2026-10-15-store-cow-pages/SUMMARY.md).

**Storage accounting and fixed-working-set traverse (2026-10-15):**
Raw/framed/canonical stores expose `storage_stats()`; ready disk owners expose
`store_storage_stats()`. These report archive dimensions, encoded index bytes,
retained index-buffer bytes and bounded cache counts without file access or locks.
They exclude allocator/OS overhead, selected captures, workers and GPU storage.
The asynchronous storage-only traverse covers 64/1,024/4,096 tiles with four
selected particles and 16/256/1,024 updates. Index buffers remain **8,492 bytes**,
cache entries at most four, and selected payload reads at most **1,872 bytes**.
Exact history/IDs survive updates and repeated reopen. At 4,096 tiles, logical
writes total **252,506,112 bytes**: v1 still rewrites the index per publication.
Seventy-five CPU checks and fifteen frozen RTX 4080 lifecycle suites pass in
**22.14 s**. The storage test uses tmpfs; its elapsed time is not disk-latency or
solver-performance acceptance. Physical eviction/sleep/support and full S07
`field-stream/compare` remain incomplete.
[Storage accounting evidence](benchmarks/soil/2026-10-15-store-stream-accounting/SUMMARY.md).

**File-backed manifest and bounded reuse cache (2026-10-15):**
Production stores now retain a validated file index, not a whole-field entry map.
Lookups use signed-key binary search; selected writeback streams unchanged entries
and merges only its bounded affected map. Version-one bytes, digest checks and
atomic HEAD publication are preserved. Reuse epochs are capped at 1,024 tiles and
replaced on each save/load; cache misses verify existing blobs. Full enumeration,
full load/save and collection remain explicit field-sized operations. Open and
changed publication still perform O(field) I/O; this is not a bounded-latency claim.
Seventy-three CPU checks and fourteen frozen RTX 4080 suites pass in **14.56 s**.
Tests include constant index buffers through 16,384 tiles, exact streamed merge,
old-generation inode retention, errors, and a 1,025-tile save/load/migration/reopen.
Bounded index/cache retention is implemented; full physical traversal/sleep/support
and the 4 ms/60 FPS target remain incomplete.
[File-index evidence](benchmarks/soil/2026-10-15-store-file-index/SUMMARY.md).

**Worker-side disk transition admission (2026-10-15):**
Disk-world polling now prepares bounded active-particle counts without manifest
lookups. The write worker checks the prospective selection against solver,
particle, tile and family-specific encoded-byte limits before publication; failure
returns the healthy store, original selection and captured state for retry. The
load worker filters absent requested tiles, with separate bounds for requested
regions and actual payload tiles. Raw public reads remain strict about missing
keys. Seventy CPU checks and nine frozen RTX 4080 transition suites pass in
**9.10 s**, including capacity retry, host/GPU swept pins, raw/framed/canonical
owners, publication failure and post-write corruption. No physics or timing-gate
change. The subsequent file-index increment above bounds manifest/cache retention;
the 4 ms/60 FPS target remains open.
[Transition worker evidence](benchmarks/soil/2026-10-15-store-transition-worker/SUMMARY.md).

**Worker-side manifest preflight and cached totals (2026-10-15):**
Raw/framed and canonical background reads now bound/deduplicate keys on the caller,
then check manifest membership, particle counts and encoded-byte limits on the
worker before payload access. Invalid requests return the same healthy exclusive
owner; malformed frame/key-count requests still reject before spawn. Stores expose
a cached `total_particle_count`, and disk-world construction no longer enumerates
the visited field to obtain that total. No file-format or physical change.
Sixty-four CPU checks and six frozen RTX 4080 GPU suites pass in **8.43 s**, including
paused-worker rejection-before-payload, cache publication/reopen/failure boundaries,
raw/framed/canonical persistence and both live world owners. The full manifest map
and reuse-epoch cache still remain. The later transition-admission increment above
moves disk-world count preflight off the polling caller ahead of file-backed lookups.
[Worker ownership and aggregate evidence](benchmarks/soil/2026-10-15-store-worker-preflight/SUMMARY.md).

**Streaming manifest migration started (2026-10-15; earlier S07 increment):**
Store open and publication now stream version-one manifests through fixed 8 KiB
buffers instead of allocating whole encoded HEAD vectors. Existing byte format,
digest, locking, publication barriers and poisoning semantics are preserved.
A validated fixed-record file index supports bounded-memory lookup in tests up
to 16,384 entries; production open still materializes the existing BTreeMap.
Sixty CPU store checks and five frozen GPU persistence suites pass in **6.65 s**,
including raw/framed/canonical reload, background I/O and resumed physical state.
This is not yet bounded visited-field metadata: the authoritative entry map,
partial-publication clones and reuse-epoch cache remain to migrate. Full map removal
must also preserve infallible explicit enumeration and keep disk preflight out of
nonblocking callers. Streaming publication still performs field-sized I/O; no
latency, real-time or full S07 acceptance follows.
[Format compatibility and migration evidence](benchmarks/soil/2026-10-15-store-manifest-stream/SUMMARY.md).

**Nonblocking quiescence acquisition verified (2026-10-15):**
Both owners expose `try_request_swept_quiescence`, `poll_swept_quiescence` and
`cancel_swept_quiescence`. The existing staging buffer holds one bounded read;
blocking reads share that slot. Identity, epoch, attempt and byte count are captured
at request time. Cancellation drains before reuse; map/poll faults require
disable/reprepare. Six CPU checks and eight frozen RTX 4080 GPU suites pass in
**16.72 s** under the unchanged 90-second cap, including dense/active/compact parity,
busy/foreign rejection, dt changes, cancelled steps, actual restore/ID replacement,
reseed/count changes, faults, queued drop and explicit logical-device destruction.
Both owners compose async quiescence with all support kinds and reject operations
during region transitions. No-op region transfers preserve current provenance;
actual subsequent steps invalidate it. The initial failed fixture below is retained.
For 129 particles, request/poll uses zero observed hot GPU-buffer allocations,
zero uploads/dispatches and 1,048 readback bytes; quiescence storage remains 2,080
bytes. These are observation APIs, not automatic active-region scheduling, safe
physical deactivation or whole-frame speed acceptance. Continuing to step can make
the returned observation historical; the current-state validator is unchanged.
[Provenance fixtures, source and evidence](benchmarks/soil/2026-10-15-quiescence-async-provenance/SUMMARY.md).

**Quiescence async initial failed fixture (2026-10-15; retained history):**
Request/poll/cancel is implemented for quiescence acquisition on both owners.
Six CPU checks pass; the first frozen GPU run completes in **25.00 s** with
seven suites passing and one owner fixture failing. The fixture incorrectly
expected an unchanged region selection to invalidate provenance, although that
path deliberately skips restore. The verified fixtures above distinguish no-op
preservation from actual restore/identity changes without altering the validator.
[Initial failed fixture and evidence](benchmarks/soil/2026-10-15-quiescence-async/SUMMARY.md).

**All support query kinds have nonblocking APIs (2026-10-15):**
Both memory and disk owners now expose request/poll methods for transitive
closure, stable-ID retention and region-seeded retention, alongside direct
support. All share one retained staging slot; polling the wrong kind returns an
error without consuming the pending result. Any support poll can drain a cancelled
request. Stable-ID requests check backpressure before building the particle mask.
Four CPU contracts and six frozen RTX 4080 GPU suites pass in **12.02 s** under
the unchanged 90-second cap, including dense/active/compact blocking parity,
wrong-kind polling, busy-slot rejection, copied IDs, short-round failure/recovery,
cross-kind cancellation/drain/reuse, stale provenance and both public owners.
The four-particle async queries observe zero hot GPU-buffer allocations; physics
checkpoint bytes are unchanged. This removes waits from support request/poll,
not the entire scheduling path. Quiescence acquisition was still blocking at this
checkpoint; the increment above adds its async API. No physical sleep, eviction,
integrated speedup or 4 ms acceptance follows.
[Typed lifecycle and frozen evidence](benchmarks/soil/2026-10-15-support-async-kinds/SUMMARY.md).

**Nonblocking direct support observations (2026-10-15; earlier API increment):**
Memory-backed and disk-backed soil owners now expose
`try_request_swept_support`, `poll_swept_support`, and `cancel_swept_support`.
One retained staging slot provides backpressure; request and poll do not wait for
GPU completion. Cancellation drains completion before reuse, and map/poll faults
require disable/reprepare. Historical observations retain request-time provenance
and must pass existing current-state validation before current use. Two CPU
contracts and three GPU suites pass in **8.06 s**, including blocking parity,
both owners, cancellation/reuse, stale provenance, fault cleanup and queued drop.
The focused request/poll path observes zero hot GPU-buffer allocations and uses
the existing storage budget. Earlier compile/cancellation failures are retained.
This is not physical sleep/eviction authorization. Closure/retention/region queries
were still blocking at this checkpoint; the extension above adds their async APIs.
Full S03/S07 and 4 ms/60 FPS acceptance remain open.
[API, lifecycle repair and frozen evidence](benchmarks/soil/2026-10-15-support-async-drain/SUMMARY.md).

**Operator row-reuse checkpoint (2026-10-15; test-only, performance unaccepted):**
An isolated operator transform shares each input component across six named row
accumulators while retaining every row's original direct-then-machine summation
order. CPU/compiler and feature-lane checks pass. Both current-certificate
packets match all five unmodified baseline buffers exactly, including 16 exact
timed repeats each and unchanged physical/public gates. The combined run then
times out at **90 seconds** before completing lifecycle checks; this remains a
failed run. A separate lifecycle-only run of the **same frozen executable** passes
in **87.56 seconds**, without rerunning the captured timing comparison. Paired
timings have extreme outliers and do not establish speedup. No promotion or
full-frame acceptance; preserve the timeout and do not repeat unchanged timing.
[Implementation, independent audits and deadline evidence](benchmarks/soil/2026-10-15-newton-operator-rows/SUMMARY.md).

**Guarded scalar-product outcome (2026-10-15; rejected for production):**
A test-only operator specialization reuses protected paired-by-scalar arithmetic
with raw-bit range guards and the original product as fallback. CPU/compiler
contracts pass. One frozen RTX 4080 run completes in **79.58 s** under the unchanged
90-second cap: 129,536 component inputs, both current-certificate packets, and
candidate-enabled lifecycle checks pass. All five replay buffers match the
unmodified baseline exactly; each packet has 16 exact timed repeats and unchanged
physical/public bounds. Nevertheless, the candidate is slower in all 16 paired
comparisons: paired upper-median regressions are **0.5695% / 0.5412%**. Keep it
test-only; no promotion, unchanged retry or full-frame speedup claim. Defaults and
the 4 ms/60 FPS requirements remain unchanged.
[Frozen source, cases and rejection evidence](benchmarks/soil/2026-10-15-newton-scalar-product/SUMMARY.md).

**Current linear-phase attribution (2026-10-15; diagnostic only):**
An alternate eight-slot trace now separates Krylov, norms, recovery, true-error,
preconditioning, operator and projection intervals. Original broad tracing remains
available; both CPU/compiler contracts and the scheduling contract pass. One
frozen GPU run completes in **70.80 s** including setup; both current-certificate
packets match all five unmodified baseline buffers exactly, with 16 timed exact
repeats per packet and unchanged physical/publication gates.

Operator inclusive upper medians are **1,458,634 / 1,460,718 subgroup ticks**,
versus precondition **384,437 / 385,516**; hybrid totals are
**4,140,343 / 4,152,830**. These nested, lane-zero observations are not nanoseconds
or an additive partition. Norm/projection totals include calls outside Krylov.
They support focusing on paired operator/true-residual arithmetic rather than
more preconditioner unrolling; no speedup is claimed. Sampling overhead medians
are +0.45%/+0.30% with timing outliers and unrelated GPU work present. No unchanged
rerun or production setting change follows.
[Trace contracts, repeated samples and next target](benchmarks/soil/2026-10-15-newton-linear-clock/SUMMARY.md).

**Publication input-validation cache (2026-10-15; test-only, not promoted):**
A per-Newton-dispatch cache skips repeated validation of immutable contact,
row/response, mass and free-rate inputs only after the collective input check
succeeds. Every entry resets the flag; state, represented-output, conservation
and physical checks remain unchanged. Two CPU/source/compiler contracts and the
pair-order check pass. One frozen GPU batch passes both current-certificate
packets plus scratch/fault/input-edit/empty lifecycle checks in **65.77 s**.
All five buffers match both modes and the unmodified baseline exactly, with
16 timed self-repeats per packet and unchanged physical/public gates.

Paired upper-median changes are only **-0.71% / -0.57%** in a single compiled
branch-controlled probe, with pair outliers up to **+30.24% / +476.28%** and
unrelated GPU work present. These results do not justify promotion or further
tuning of this path. No independent-binary or whole-frame speedup is established;
do not repeat the unchanged timing run. Production selection remains unchanged.
[Source, lifecycle and paired evidence](benchmarks/soil/2026-10-15-newton-publication-input-cache/SUMMARY.md).
The subsequent Rust-only feature-guard correction passes protected-only test
type-checking and regenerates byte-identical GPU-tested WGSL/SPIR-V; no GPU rerun
was needed. [Feature compatibility audit](benchmarks/soil/2026-10-15-newton-input-cache-feature-guard/SUMMARY.md).

**Bounded Krylov scratch reset (2026-10-15; test-only):**
The conditioned SoA Newton prototype resets only the Hessenberg rows and RHS
prefix needed by the current cycle, reducing explicit reset writes from 4,225
to 585 for an eight-iteration cycle. Three CPU contracts cover the source
transform, cycle limits 1–64, and publication scratch write-before-read bounds.
One frozen RTX 4080 replay completes in **66.70 s** including GPU construction,
below the 90-second cap. Both fresh current-certificate packets match all five
unmodified baseline buffers bit-for-bit, repeat exactly three times, and pass
the unchanged independent physical 1e-6 check. This is a write-count reduction,
**not a measured solve-time speedup** or full-frame acceptance; unrelated GPU
work remains active and untouched. Production selection is unchanged.
[Source audit and exact replay](benchmarks/soil/2026-10-15-newton-bounded-reset/SUMMARY.md).

The same frozen candidate also passes the existing conditioned-SoA lifecycle
gate in **67.61 s**: 65 active contacts in 4096 slots, storage-scratch poisoning,
reuse, fault/reset recovery, iteration exhaustion, empty solve and invalid-mass
publication. Independent physical error is **8.591260670999163e-9** for the
nonempty fixture; hot allocation/transfer and dispatch-count checks pass.
This closes that lifecycle check, not trajectory or matched-work timing gates.
[Lifecycle evidence](benchmarks/soil/2026-10-15-newton-reset-lifecycle/SUMMARY.md).

**Current-contract tractor captures and control snapshots (2026-10-15):**
Fresh actual 50k tractor/implement first-frame capture passes in **144.82 s**
including setup, below 180 s. Both parity packets carry capture-time certificate
version 1 and same-solve initial seeds. Unmodified replay passes in **66.43 s**:
archived states/reports/publication are exact, all five buffers repeat across
three executions per packet, and independent physical errors stay below 1e-6.
These are new actual solves, not retagged legacy reports.

The pending control-load snapshot prototype now passes both current packets in
**76.68 s** under the 90-second cap: scalar/snapshot outputs are bit-exact,
16 timed executions per packet repeat, physical checks pass and public delta is
zero. **No speedup is established.** Scalar/snapshot upper-median GPU times
reverse between packets: **35.505 / 4.972 ms** and **4.944 / 35.421 ms**. GPU load
rises during the run; an unrelated `astrocraft` process is observed afterward
using about 5.3 GiB and is left untouched. Keep this prototype test-only; timing
does not support promotion. Full solver regression, trajectory, support/wake and
4 ms/60 FPS acceptance remain open.
[Fresh actual controls](benchmarks/soil/2026-10-15-newton-current-certificate/SUMMARY.md),
[paired comparison and contention evidence](benchmarks/soil/2026-10-15-newton-current-snapshot/SUMMARY.md).

**Capture-time certificate versions (2026-10-15; implemented):**
Newton failure/success packets now store certificate version 1
(`gap-independent-velocity-v1`) in reserved header word 10. Export derives
`certificate_version` and `certificate` metadata from the hashed packet bytes;
it never stamps current provenance onto an old report. Packet layout size,
physical payloads, solver arithmetic and numerical thresholds are unchanged.
Version zero means unspecified, including untagged post-fix captures; unknown
versions reject. Future certificate-definition changes require a new version.

Exact-success, seed-cost and first-release comparisons now require matching
packet/metadata/current versions. Bounded reads validate digest, shape and
provenance; batch replay checks every packet before creating its shared GPU
engine. Single replay checks before GPU setup. Untagged packets remain available
for input/state analysis without an exact archived-report compatibility claim.
Legacy fixtures are not rewritten or retrospectively certified.

Seven CPU checks and all three GPU capture lifecycle cases pass (**0.37 + 0.57 s**,
under the shared 90-second cap), including current/legacy export and unknown
version rejection before file publication. The actual old batch-replay entry
point now rejects the October 13 packet in **0.00 s** with the provenance error,
before GPU construction. This expected rejection does not turn its archived
report check green. Fresh current-contract solve captures are still needed for
solver optimization; full 4 ms/60 FPS and physical acceptance remain open.
[Implementation and evidence](benchmarks/soil/2026-10-15-newton-certificate-provenance/SUMMARY.md).

**Newton barrier folding and certificate provenance (2026-10-15; test-only):**
A protected-compiler prototype removes two redundant static SPIR-V barriers
(one each in `krylov_cycle` and `hybrid_direction`) across invocation-local or
immutable instructions only. All non-barrier words remain identical and Vulkan
validation passes. Calls, control-flow changes, shared/storage accesses and
unknown operations stop folding. Production selection is unchanged.

The candidate and a separately frozen unmodified current control each complete
three exact repeats, then fail the same October 13 archived-report comparison
in **65.69 / 65.83 s**, below the unchanged 90-second cap. Their five captured
buffers are byte-identical to each other; archived state/publication also match.
Independent physical error is **2.555279284772259e-7**, below 1e-6. No timing
improvement or complete optimization acceptance is established.

Source attribution identifies incompatible certificate definitions: those old
captures include gap in velocity normalization; the October 15 correctness fix
removed it from GPU and CPU certificates. Thus this report mismatch also exists
without either the barrier or control-snapshot optimization. Keep the failed
archive checks intact. Next, record certificate provenance at capture time and
reject incompatible exact-report comparisons before GPU construction; establish
fresh, separately frozen current-contract controls without rewriting old evidence.
[Candidate](benchmarks/soil/2026-10-15-newton-barrier-fold/SUMMARY.md),
[unmodified control and source attribution](benchmarks/soil/2026-10-15-newton-barrier-control/SUMMARY.md).

**Frontier-only support propagation (2026-10-15):**
The bounded support closure now skips particles already reached, stops node
search at the first reached node, and scatters only newly reached particles.
A temporary frontier flag is cleared before publication. Round ordering,
fixed-point rejection, scratch sizes, dispatch counts and public result semantics
are unchanged; no physical sleep or stepping shortcut is introduced.

The frozen original shader and production shader produce identical particle
results and complete node fields in **44 comparisons** across 29, 129 and 50k
particles, seed/nonquiet patterns and 1–64 rounds. Independent CPU graph checks
cover the 32 small cases. All nine GPU suites pass in **15.00 s**; six CPU shader/
contract checks, 136 checkpoint regressions and facade/format checks also pass.
Six alternating-order 50k/64-round timing pairs observe nearest-rank wall p50
**1.853383 → 1.429533 ms**, a **22.87%** reduction for this explicit support query.
Timing includes command encoding and diagnostic result/node readback. It is not
full physics throughput, 4 ms soil acceptance, or renderer-inclusive 60 FPS.
[Frozen reference, timing rows and evidence](benchmarks/soil/2026-10-15-soil-support-frontier/SUMMARY.md).

**GPU region support seeds (2026-10-15; no deactivation):**
Both live owners now provide `prepare_swept_support_regions`,
`swept_support_region_bytes` and `read_swept_region_support`. Bounded swept
requests expand to whole archive-tile boxes, rounded outward into the current
local frame. A GPU pass selects current particle positions inside those closed
boxes, then runs the existing retained-support closure. Selection requires no
particle-position download or per-particle seed upload. It does not load missing
tiles, publish region claims, remove pins or change ordinary physics stepping.

Optional storage costs `32 * region_capacity` bytes (capacity 1–64), separately
reported from base support storage and excluding pipeline/driver allocations.
A nonempty request adds one dispatch and uploads 32 bytes per region, on top of
the existing 64-byte parameters; readback remains `16 + 4 * particle_count`.
The explicit query still blocks for its result. Empty region lists use the
existing nonquiet closure without a seed dispatch.

Five quiescence CPU contracts, two geometry checks (496 boundary cases), 136
checkpoint regressions and facade/format checks pass. Eight GPU tests pass on
RTX 4080 in **14.64 s**, including dense/active/compact seed parity, exact
physical checkpoint bytes, region unions, lifecycle/budget rejection, both-owner
provenance and 129-particle cross-workgroup coverage. This is component correctness,
not 50k throughput or safe sleep/eviction acceptance. Boundary forces, wake
coverage, bounded traversal and the 4 ms/60 FPS requirements remain open.
[Evidence](benchmarks/soil/2026-10-15-soil-region-support/SUMMARY.md).

**Explicit retained-support seeds (2026-10-15; stable IDs, no deactivation):**
Both live owners now expose `read_swept_retained_support` and
`validate_swept_retained_support`. The new immutable result distinguishes
`quiet`, explicitly `requested`, and transitively `retain` particles. It unions
current nonquiet particles with caller-requested stable global IDs before the
bounded GPU closure. Thus a quiet particle retained under a machine cannot lose
the rest of its connected support merely because the whole component is quiet.
This separate result does not change `connected_to_nonquiet` or direct-query
semantics. Unknown/duplicate IDs, missing/currently invalid identity and incomplete
propagation reject.

The query reuses existing support buffers and pipelines. Nonempty seed requests
upload a four-byte mask per live particle, in addition to the existing 64-byte
parameters; empty seed lists omit that upload. Dispatch and readback costs remain
`2 * rounds + 2` and `16 + 4 * particle_count` respectively. Explicit host ID
mapping/result allocations are separate from the unchanged logical GPU budget.
Four CPU contracts, 134 checkpoint regressions and facade/format checks pass.
The initial seven-case GPU run passed six cases in 11.11 s, including both-owner
seeded lifecycle checks; its new fixture omitted the mandatory identity for an
implicit material row. That failure remains sealed. Only the fixture changed,
and its repaired dense/active/compact and 129-particle checks pass in **1.98 s**.
They cover quiet chains, isolated seeds, unions, empty requests, high-bit IDs,
budget exhaustion, old-query independence and exact physical checkpoint bytes.
Six successful unchanged GPU cases were not rerun.

These are conservative retention dependencies, not machine-contact coverage,
force equilibrium, sleep/eviction authorization or reduced active physics work.
All residency pins and full-plan acceptance requirements remain unchanged.
[Initial evidence](benchmarks/soil/2026-10-15-soil-retained-support/SUMMARY.md),
[fixture repair](benchmarks/soil/2026-10-15-soil-retained-support-repair/SUMMARY.md).

**Transitive support query (2026-10-15; conservative connectivity, not sleep):**
Both `MpmSoilWorld` and `MpmSoilDiskWorld` now expose
`read_swept_support_closure` and `validate_swept_support_closure`. The explicit
GPU query propagates nonquiet membership through shared quadratic-transfer
stencils, including all 27 nodes and zero-weight endpoints. It catches quiet
particles connected to nonquiet soil through other quiet particles, which the
existing direct occupancy query cannot identify.

The caller supplies 1–64 propagation rounds. Separate gather/scatter dispatches
give deterministic rounds, followed by a fixed-point check; an unfinished chain
rejects the whole query rather than returning a partially quiet region. The query
reuses the existing bounded support buffers: `80 + 4 * grid_size³ +
8 * max(capacity,1)` logical bytes, with three additional prepared pipelines
outside that byte count. Nonempty queries use `2 * rounds + 2` dispatches,
64 upload bytes and `16 + 4 * particle_count` readback bytes. There is no added
ordinary-step work or hot query buffer allocation. Snapshots are immutable and
bound to the current quiescence/identity boundary; stale, foreign, pending,
clipped and unsupported requests reject.

Three CPU contracts, 134 checkpoint CPU regressions and facade/format checks pass.
The initial six-case GPU run passed five cases in 9.65 s, including both-owner
closure lifecycle checks, but its support fixture incorrectly expected an
immediate readable snapshot after empty reseeding. That failure remains sealed.
Only the fixture changed; its corrected dense/active/compact run passes in
**1.62 s**, checking CPU graph connectivity, incomplete-budget rejection,
1/2/3/64-round behavior, 129-particle boundaries, invalid geometry, cancellation
and unchanged physical checkpoint bytes. Existing direct support queries still
use two dispatches. No successful unchanged GPU case was rerun.

This proves conservative current connectivity, not force equilibrium, future
load coverage, safe boundary removal or physical sleep. All active residency pins
remain unchanged. Bounded physical traversal, actual sleep/wake, static support
contributions and the 4 ms/60 FPS target remain open.
[Initial evidence](benchmarks/soil/2026-10-15-soil-support-closure/SUMMARY.md),
[repaired fixture](benchmarks/soil/2026-10-15-soil-support-closure-repair/SUMMARY.md).

**Contact control snapshots (2026-10-15; experimental, exact-report gate fails):**
A test-only transform combines two hybrid-handoff control loads and three
refinement-stop control loads into one uniform struct snapshot per site. Original
arithmetic, bounded iteration limits, stopping predicates and GPU storage remain
unchanged. Two CPU/compiler contracts and the paired scheduling contract pass.
The first CPU source check caught the bounded solver's rewritten `limit` anchor;
the corrected candidate and original failure are both preserved.

One frozen GPU run terminates at the exact archive-report assertion in **65.43 s**,
under the unchanged 90-second cap. On the first packet, all five buffers match
between scalar and snapshot modes; state bytes and publication report also match
the archive. Only report word 5 differs: physical residual **2.55527937e-7** versus
archived **2.49547355e-7**. Both modes share that discrepancy, so it is not isolated
to the enabled snapshot path. Independent physical checks, paired timing, repeated
solves and the second packet were not reached. No numerical gate is relaxed,
no speedup is claimed, and production remains unchanged.
[Preserved experiment](benchmarks/soil/2026-10-15-newton-control-snapshot/SUMMARY.md).

**Tractor grid-reuse integration (2026-10-15; exact recorded outputs, no speed gain):**
`SoilTractorTransfer::GridReuse`, benchmark `grid-reuse:14075712` and viewer
`SOIL_LIVE_TRANSFER=grid-reuse` now expose the retained transfer cache explicitly.
The default remains atomic. Constructors reject insufficient full logical-grid
budgets before soil allocation, and live metadata verifies the actual cache mode
and its 14,075,712-byte gather storage for the 50k/grid64 workload. Ordering and
driver storage remain separate.

Five tractor CPU checks, 23 benchmark checks and 13 viewer checks in each of two
geometry configurations pass. One frozen RTX 4080 capture completes 33 frames,
34 substeps per frame, within the unchanged 180-second cap. All published machine
poses, initial/final particle-position bytes and 1,122 Newton work reports match
the sealed historical fixed-order control exactly. Surface-envelope and load
observation-ledger audits also pass. These are not full rates/history/forces,
rollback, repeatability or calibrated physical acceptance.

The moving tractor does **not** demonstrate a speed gain: whole-physics median/p95
is **366.576 / 562.329 ms**, versus the historical control's **356.112 / 552.111 ms**.
Summed P2G median per simulated frame is **23.781 ms**, versus **15.280 ms**.
These are sequential historical comparisons, not a fresh paired timing experiment.
The saved image shows the diagnostic tractor/tine and geometric point envelope,
not validated rutting or soil displacement. The option stays off by default;
4 ms/60 FPS and full agricultural acceptance remain unmet.
[Frozen integration evidence](benchmarks/soil/2026-10-15-grid-reuse-integration/SUMMARY.md).

**Exact transfer-tile reuse (2026-10-15; opt-in, held-input gain only):**
`prepare_deterministic_p2g_grid_reuse` now retains complete fixed-order P2G node
outputs. Twenty-six exact input components per particle cover position, velocity,
mass, APIC C and the prepared stress/stiffness. Changed or removed particles
invalidate both old and new transfer footprints; invalid tiles stay invalid while
absent from the active grid. Grid-origin/spacing/window changes invalidate the
logical cache. A known transfer fault cannot advance the retained particle keys.
Untouched tiles restore their original mass, momentum and stiffness into the current
dense/compact storage mapping; dirty tiles execute the existing arithmetic/order.
Every particle still participates in subsequent grid/particle evolution. No soil is
evicted, no physical sleep certificate is inferred, and ordinary preparation restores
the uncached route. Defaults remain unchanged; explicit viewer selection is described above.

Retention costs `176 * max(capacity,1) + 20 * grid_size³ +
8 * ceil(grid_size/4)³ + 64` bytes, excluding ordering/driver storage. The cache is
dense in the bounded logical window even for compact solvers; unsupported storage,
index and dispatch sizes reject. It adds **two GPU dispatches per transfer**, four
storage bindings and no cache-path upload/readback. The 129-capacity fixtures retain
886,100 bytes at grid size 35 and 5,298,416 at grid size 64.

Seventeen CPU contracts, four GPU checks (**8.90 s**), two prior stress-cache GPU
regressions (**6.00 s**) and facade/format checks pass. Eight configurations preserve
eight evolved physical checkpoints and three held grid transfers each exactly.
Further checks cover removal/empty-tile reactivation, 25 changed particle input
components, an integer-window move, faulted key publication and an actual hit bypass
with a deliberately faulting miss path. The initial one-pass/three-fail run remains
sealed: its raw-transfer fixtures omitted normal setup uploads/initialization and
retained stale material IDs after reseeding. Only that test file changed in the repair.

A separate **1.25 s** frozen 50k detached run preserves 12 evolving checkpoints and
12 held-input transfer pairs exactly. Ten warmed paired comparisons show held
transfer **1.617408 → 0.170496 ms**, median paired change **-89.36%**. Evolving P2G
instead regresses **+2.97%**, summed GPU stages **+2.08%**, and completed wall time
**+1.15%**. This is a substantial unchanged-input transfer gain, not a moving-soil or
tractor speedup. The evolving lane advances only 0.0012 simulated seconds; the held
lane does not advance physics. No 4 ms / 60 FPS, sleep/eviction, long-traverse or
full-plan acceptance follows. The focused root lane uses `make test-soil-gather
RELEASE=1 SOIL_GATHER_FILTER=gather_grid_reuse_gpu_ RUN_WITH=nixVulkan`.
[Retained failure, exactness and paired tradeoff](benchmarks/soil/2026-10-15-grid-reuse-lifecycle/SUMMARY.md).

**Exact-input gather stress cache (2026-10-15; opt-in, no accepted speedup):**
`prepare_deterministic_p2g_stress_cache` now memoizes the existing fixed-order
gather's stress/stiffness calculation when all nine deformation components and
eight scalar constitutive inputs match bit-for-bit. The original arithmetic runs
on misses. Every particle still contributes mass, momentum and stiffness through
the unchanged gather; no support is removed and no sleep permission is inferred.
Direct/shared and native/non-native policies retain their existing compatibility
constraints. Ordinary preparation removes the cache again. Defaults are unchanged.

Retained storage rises from 64 to **160 bytes per capacity slot** (8 MB at 50k,
including the existing 3.2 MB outputs); ordering and driver storage are separate.
The extra 96-byte keys use one additional storage binding. Budget/capability errors
preserve existing prepared state. There is no additional dispatch, upload or readback
in the cache path. The pure-input key covers material changes, dt, volume and elastic/
plastic history without relying on endpoint rest or dropping physical state.

The CPU shader/layout contract passes. Two bounded GPU tests finish in **6.01 s**:
eight dense/active/sparse, direct/shared and stiffness configurations each match
eight evolved checkpoints and mass/momentum/stiffness grids exactly, including a
dt change. A fault-on-miss probe proves actual hit bypass and separate misses for
all **17 scalar input components**. Allocation/budget/mode-restoration checks pass.
A separate **1.26 s** frozen 50k detached-soil run matches 12 evolving checkpoints
and 12 held-input transfer pairs exactly; hot application buffer allocations and
readbacks remain zero. The full run advances only **0.0012 simulated seconds**.

The ten warmed paired medians are small and mixed: evolving P2G **-0.86%**, summed
GPU stages **-1.30%**, wall time **-0.06%**; held-input transfer **-2.18%**, with some
pairs regressing. This does **not** establish a reliable end-to-end speedup or the
4 ms budget. The held-input lane is not a physical step, and this fixture is not a
loaded tractor. No default/viewer promotion or unchanged timing retry follows.
The focused root lane is `make test-soil-gather RELEASE=1
SOIL_GATHER_FILTER=gather_stress_cache_gpu_ RUN_WITH=nixVulkan`; frozen verification
and the retained initial benchmark-build error are recorded separately.
[Exactness, paired measurements and scope](benchmarks/soil/2026-10-15-gather-stress-cache/SUMMARY.md).

**Direct transfer-support dependencies (2026-10-15; implemented, not sleep):**
Both live owners expose an opt-in retained `prepare_swept_support` /
`read_swept_support` query bound to a current immutable exact-rest snapshot.
Two GPU passes mark and gather quiet/nonquiet occupancy over all 27 quadratic
transfer nodes, including zero-weight endpoints conservatively. The query detects
quiet particles sharing nodes with nonquiet particles; it does not infer transitive
connectivity, force equilibrium, future load safety or permission to evict.
Detached explicit soil is required. Clipped/unrepresentable stencils, zero streak
thresholds, over-bound reads and stale/foreign/pending owner state reject.

Additional retained GPU storage is `80 + 4 * grid_size³ + 8 * max(capacity,1)`
bytes, independently budgeted even for sparse solvers: **1,049,688 bytes** for
64³ nodes / 129 particle capacity. This is dense diagnostic scratch in the current
logical window, not an out-of-core support cache. Explicit nonempty queries use
two dispatches, a 64-byte parameter upload and `16 + 4 * particle_count` readback
bytes; returned host vectors are additional. Queries perform no particle/history
writes and add no work to ordinary substeps. No timing speedup is claimed.

Three CPU contracts and **134 checkpoint regressions pass**, as does the facade
check. GPU checks cover dense/active/sparse modes, translated integer windows,
4/129 particles, controlled mixed dependencies versus an independent pairwise
oracle, malformed/clipped stencils, exact query repeats, stale/cancelled snapshots,
allocation reuse and byte-exact physical checkpoints. The initial six-test run
finishes in **9.78 s** with five passes and a retained owner-test setup failure
(missing test directory). After fixing only that setup in a separate harness,
the same frozen owner test passes in **0.98 s**; no successful tests were rerun.
The existing `make test-soil-quiescence RELEASE=1 RUN_WITH=nixVulkan` lane includes
the new GPU check and both owner integrations. All particles remain pinned;
dormant support contributions, load-driven wake, eviction, long traversal and
4 ms / 60 FPS acceptance remain open.
[Frozen source, failed setup and repaired owner evidence](benchmarks/soil/2026-10-15-soil-stencil-support/SUMMARY.md).

**Stage-recovery continuation completion implemented; exact gate still failed
(2026-10-15):** Final-certificate tracing preserves all seven failed-candidate
outputs and ordinary report words exactly. Both differing cases return while
smoothing remains positive, before zero-smoothing settling is eligible. A new
test-only completion condition lets successful stage recovery return only after
the existing continuation schedule reaches zero; ordinary solves retain early
certification. No constants, work caps or tolerances change. Substep 12 uses
**42/287/245** versus original 42/372/307, and its maximum output delta falls to
**8.73115e-11**. Substep 19 uses 32/188/192 and differs in outputs by 6.76556e-16.
Physical checks and three exact self-repeats pass, but both cases still fail the
frozen byte-exact gate. No robustness/timing/full-scene run or promotion follows.
The pending acceptance-scope question does not authorize changing any gate.
[Certificate trace](benchmarks/soil/2026-10-15-newton-finish-trace/SUMMARY.md).
[Completion implementation and retained failure](benchmarks/soil/2026-10-15-newton-stage-completion/SUMMARY.md).

**Distinct-stage recovery implemented; exact gate failed (2026-10-15):** A repaired
CPU prefix oracle uses captured smoothing and prepared response, preserves all
17 legacy rows byte-exactly, and distinguishes strict top-contact settling from
the broader eight-contact search. It finds a floor-release candidate at step 32
that is absent at sampled earlier stages. A bounded retry policy therefore allows
at most eight attempts at distinct existing positive smoothing stages, without
changing continuation constants or solver/release limits. The dominant replay
drops from **42/372/307 to 39/273/242** steps/mixed iterations/trials. However,
two of seven inputs fail the frozen exact state/rate/output gate (maximum output
delta **8.7123e-7**). Physical checks and exact self-repeats pass, but do not replace
that failure. No wall/timing/full-scene run or promotion follows. Investigate final
contact-118 branch/certification differences; no speedup is claimed from counts.
[Oracle repair and witnesses](benchmarks/soil/2026-10-15-newton-prefix-oracle/SUMMARY.md).
[Candidate and retained failure](benchmarks/soil/2026-10-15-newton-smoothed-stages/SUMMARY.md).

**Recovery invocation traced; guarded branch settling implemented (2026-10-15):**
The dominant solve does attempt recovery at step 10, with zero invalid local
caches, but that attempt fails. Seven instrumented solves preserve all physical
buffers and ordinary report words exactly. A subsequent bounded, merit-decreasing
branch-settle candidate permits retaining the floor rather than requiring its
release; it passes seven-packet byte-exact physical-output checks and all six
unequal-gap wall directions. It still leaves the dominant **42/372/307** work
counts unchanged. Both variants remain test-only; no current-workload speedup or
production promotion. Locate candidate availability at the saved later prefixes
before another scheduling change; do not repeat these same guards or timing runs.
[Trace evidence](benchmarks/soil/2026-10-15-newton-smoothed-trace/SUMMARY.md).
[Branch-settle implementation](benchmarks/soil/2026-10-15-newton-smoothed-settle/SUMMARY.md).

**Smoothed-stall recovery implemented, test-only (2026-10-15):** The repaired
control trace passes 12 exact prefix state/report comparisons, locating the
dominant plateau during positive smoothing with no completed recovery. A new
once-per-solve attempt after two stalled accepted steps preserves the existing
continuation schedule, budgets and certificates. Seven current packets pass
three exact repeats; archived states/reports/publication remain byte-identical.
Their work counts are also unchanged, including **42 steps / 372 mixed linear
iterations / 307 trials** for the dominant solve. Six unequal-gap wall directions
pass with exact repeats, at 10/89/49 versus the preceding guarded-first fixture's
14/144/69. This is a fixture work reduction, not a measured production speedup.
No promotion or additional timing batch follows the unchanged current workload.
[Control trace](benchmarks/soil/2026-10-15-newton-plateau-control/SUMMARY.md).
[Implementation and gates](benchmarks/soil/2026-10-15-newton-smoothed-release/SUMMARY.md).

**Guarded first recovery implemented, test-only (2026-10-15):** A once-per-solve
recovery attempt after two unsmoothed residual-stagnating steps passes the six
unequal-gap wall directions and seven captured current-path solves. All seven
archived states remain byte-exact. Frame-22 substep 17 drops from 368 to 323
mixed linear iterations and 201 to 185 trials; the other six cases, including
the main 307-trial case, retain their work counts. Independent physical checks
and three exact repeats pass. A subsequent single-program, shared-call paired
comparison measures substep 17 at **79.498→71.574 ms** (paired median -9.97%),
with all physical buffers byte-exact and all 112 timed samples retained. The
main substep 12 stays at about **82.45 ms**. Other paired median changes are
between -0.089% and +0.051%; one anchor has a pronounced timing-regime transition,
so its ratio of separate medians is not a reliable effect estimate. Keep
test-only: this does not prove original-binary or whole-frame speedup, and no
defaults, convergence limits or full-plan acceptance change.
[Candidate and evidence](benchmarks/soil/2026-10-15-newton-first-release/SUMMARY.md).
[Paired timing and limits](benchmarks/soil/2026-10-15-newton-first-release-cost/SUMMARY.md).

**Current difficult solves isolated (2026-10-15):** Bounded selected-substep
capture now retains frame-22 anchors and five expensive solves instead of all
34 packets. The 22-frame diagnostic matches every recorded live pose and all
34 frame-22 work reports exactly. Three isolated solves reproduce archived
state/report/publication bytes and three exact five-buffer repeats each.
Prefix inspection of the 307-trial solve localizes the stalled residual to
contact 115, a near-opposed machine/floor contact: it contributes 99.35% of
squared residual at step 32. Independent physical error there is still 1.01e-4,
so truncating iteration work would fail the unchanged 1e-6 gate. Investigate
guarded contact-branch recovery; do not reintroduce the previously rejected
unconditional early pivot. No production arithmetic or speedup is claimed.
[Capture, replay and witnesses](benchmarks/soil/2026-10-15-newton-hard-frame/SUMMARY.md).

**Live Newton work profiling implemented (2026-10-15):** Optional stage profiling
now retains per-substep fault, Newton-step, mixed linear-work and trial counts in
the existing machine-profile map. No additional solver dispatch/submission or
shader change is introduced. CPU, GPU reuse/rollback and one frozen 50k capture
pass their scoped checks; recorded poses and particle endpoints remain exact.
The 1,122 reports expose substantial late-frame trial growth: frame 22 totals
494 Newton steps, 2,790 mixed linear iterations and 1,629 trials; its slowest
substep takes 83.658 ms with 307 trials. Investigate this difficult-case work
before another uniform microkernel change; do not relax convergence criteria.
Whole-physics median/p95 remains **356.112/552.111 ms**, contact solve median
**210.358 ms**. This is instrumentation, not an achieved solver speedup or
physical acceptance. [Evidence](benchmarks/soil/2026-10-15-viewer-newton-work/SUMMARY.md).

**Native/lazy drift geometry integrated into the live viewer, opt-in (2026-10-15):**
`SOIL_LIVE_GEOMETRY=native-lazy` exposes the existing composed unrestricted
paired drift backend independently of deferred prediction. Eager geometry stays
default. Required f64 device support and the prepared backend identity are checked;
predictor, momentum and refresh arithmetic are not redirected. Both viewer
configurations pass 13 CPU checks. One bounded fixed-order 50k capture completes
33 frames with all recorded six-body pose components exactly equal to the eager/
deferred control and byte-identical particle endpoints. Surface/load-ledger and
timestamp audits pass. Latest instrumented median/p95 whole physics is
**358.531/556.123 ms**, final drift **53.874 ms**, total machine **96.602 ms**,
and nested contact solve **211.262 ms**. Construction is **99.920 s**, versus
76.293 s in the retained eager capture; this is not a cold-start improvement.
Separate captures are not a fresh paired whole-frame speedup estimate.
The older atomic-run trajectory failure remains failed: this fixed-order scoped
comparison neither isolates its cause nor proves full history/rates/forces,
long-horizon physics, default promotion or 4 ms/60 FPS acceptance.
[Implementation and frozen evidence](benchmarks/soil/2026-10-15-viewer-native-lazy-geometry/SUMMARY.md).

**Deferred native drift integrated into the live viewer, opt-in (2026-10-15):**
`SOIL_LIVE_DRIFT=deferred` now exposes the existing rates-only native predictor;
explicit selection requires native prediction. Soil projection is followed by
one final corrected pose reconstruction instead of also computing intermediate
predicted poses that are discarded. Full drift remains default. Runtime branch
identity, CLI expectations, metadata and overlay agree; no separate native/lazy
geometry arithmetic change is bundled. Both viewer configurations pass 13 CPU
checks. One bounded 50k capture completes 33 frames / 1,122 substeps with exact
recorded pose agreement and byte-identical particle endpoints against the
retained profiled full-drift control; surface/load-ledger and timestamp audits
pass. Latest instrumented whole-physics median/p95 is **391.786/582.844 ms**,
native prediction **9.697 ms**, and total machine work **127.445 ms**. The prior
full-drift capture measured 475.609/667.647 ms whole physics and 94.851 ms
prediction; separate captures are not a fresh paired speedup estimate.
Contact solve (210.817 ms) and final corrected drift (84.744 ms) remain dominant.
This establishes the viewer opt-in and scoped recorded-state comparison, not
full machine/history/force trajectory, default promotion or 4 ms/60 FPS acceptance.
[Implementation and frozen evidence](benchmarks/soil/2026-10-15-viewer-deferred-drift/SUMMARY.md).

**Live stage profiling implemented, opt-in (2026-10-15):**
`SOIL_LIVE_STAGE_PROFILE=1` enables bounded capture of the existing retained
MPM, nested contact/preparation, machine GPU and host timing streams. The worker
checks committed-frame identity and workload before publishing each profile;
the viewer rejects missing, duplicate or out-of-order profile events. Blocking
timestamp reads are measured separately from physics and normal observation.
Nested stages are explicitly non-additive and machine/MPM clock origins remain
separate. Ordinary runs request no timestamp features or profile readbacks.
Both submission configurations pass 13 CPU checks. The first GPU attempt failed
before readiness because inside-pass timestamp features were omitted; its
failure is preserved. The corrected feature request passes one bounded actual
50k capture: 33 profiles cover all 34 substeps/frame, recorded poses are exactly
equal to the retained unprofiled control, and both particle endpoints are
byte-identical. Independent nesting/readback, surface and load-ledger audits pass.
Measured-frame medians are **475.609 ms whole physics**, **247.337 ms top-level
MPM GPU**, **210.372 ms nested contact solve**, and **211.890 ms machine GPU**.
Machine dynamics (94.851 ms), corrected drift (84.275 ms) and momentum (30.104 ms)
are substantial costs even with native prediction; further work must address
machine execution as well as nonlinear contact solving, not only transfer or
preparation microkernels. Diagnostic reads add a separately measured median
1.552 ms and 11,984 logical bytes/frame. These are instrumented stage samples,
not fresh paired profiling-overhead evidence, a speedup or real-time acceptance.
[Initial failure](benchmarks/soil/2026-10-15-viewer-stage-profile/SUMMARY.md);
[corrected verification](benchmarks/soil/2026-10-15-viewer-stage-profile-enabled/SUMMARY.md).

**Composed submissions integrated into the live 50k viewer, opt-in (2026-10-15):**
`SOIL_LIVE_SUBMISSIONS=composed` forwards the verified projected-submission
feature independently of prediction, transfer and preparation choices; `separate`
remains default. A runtime query shared with the actual branch checks the
worker identity before Ready, and the CLI, capture metadata and overlay identify
the selected projected sequence rather than implying one whole-frame submission.
Both viewer configurations pass 11 CPU checks. One bounded RTX 4080 capture
completes 33 native/SoA/cooperative/fixed-order frames with 50k particles; all
recorded six-body poses match the retained separate control numerically exactly,
and both particle endpoints are byte-identical. Surface/load-ledger audits pass
and the committed image was inspected. Whole-physics median/p95 is
**471.583/671.800 ms**, with every measured frame missing 60 Hz. The historical
control is not a fresh paired performance comparison, so no causal batching
speedup is claimed. Full rates/history/force trajectory, support/calibration,
residency and 4 ms/60 FPS acceptance remain open.
[Implementation, rendered evidence and limits](benchmarks/soil/2026-10-15-viewer-composed-submissions/SUMMARY.md).

**Distributed linear preparation evaluated, not promoted (2026-10-15):**
A test-only kernel splits global reset from independent contact inverse work,
using 64-lane groups while retaining the current SoA arithmetic and fallback.
Sixteen captured systems pass 112 lifecycle pairs and 128 alternating timed
pairs exactly; the independent audit verifies 80 raw buffer pairs and unchanged
input coefficients. Paired median time ratios range **0.894--1.098**: six cases
regress 4.40--9.78%, eight improve only 1.04--2.08%, and two improve 8.89--10.64%.
This mixed component result does not justify new nonlinear dispatch boundaries;
keep the prototype test-only and do not rerun it unchanged.
The initial archived-inverse equality failure is preserved: those old references
predate the corrected Schur lower-block fallback. New explicitly labelled
references recompute only inverse/status on unchanged captured coefficients;
the correctness guard and strict comparison gates are not weakened. No full-solve
speedup, physical publication or 4 ms/60 FPS acceptance is established.
[Failed historical comparison](benchmarks/soil/2026-10-15-parallel-linear-preparation/SUMMARY.md);
[current-reference evidence and decision](benchmarks/soil/2026-10-15-parallel-linear-current/SUMMARY.md).

**Native tractor submission comparison passes; profiling overflow fixed (2026-10-15):**
The actual six-body native-prediction fixture now completes both separate and
composed submission arms in **67.99 s including setup**, within the unchanged
90 s component cap. It uses 65 particles contacting all four wheels and the
implement, three substeps per frame, fixed-order transfer and protected Newton.
Four frames, unsupported-control rejection and recovery match exactly across
66 paired state files and ten paired load/metadata files. The projected
guard/solve/publication sequence uses **one submission instead of three**;
this is not a whole-frame submission count or a measured speedup.
The earlier failed attempt exposed ordinary profiling's insufficient timestamp
budget when preparation details were disabled. Retained query-set pages now
cover all nested stages without lowering the 204 ordinary/102 detailed-substep
capacities. CPU budget checks, 22 GPU boundary/reuse captures and the existing
profile-window physics-preservation check pass; both actual-machine arms pass
detail toggles and their unchanged physical checks. Defaults remain unchanged.
This closes the small native integration comparison, **not** generic
Featherstone construction, native-versus-Featherstone trajectory parity,
full50k throughput, rendered acceptance or the 4 ms/60 FPS target.
[Frozen captures, independent audit and limits](benchmarks/soil/2026-10-15-profile-query-budget/SUMMARY.md).

**Canonical sticking-contact order implemented, opt-in (2026-10-15):**
A targeted diagnostic proves that the separate-submission reference itself
varies on identical first-step state and contact-node multiset: atomic append
changes the input order of its floating-point reduction. The new
`ordered-sticking-nodes` feature adds a retained GPU canonical ordering pass
before sticking projection, independently of `batched-soil-projection`.
It preserves all node payloads, ownership and boundary masks; extra storage is
32 bytes per body/node-capacity slot, checked before allocation.
With canonical ordering enabled in both arms, all twelve dense/active/compact,
native/non-native, sticking/unilateral comparisons pass **exactly**, across
three changing steps and five output buffers per case. Independent sort,
ten fault/rollback cases, 36 fixed-boundary physical cases and compact lifecycle
checks pass without changing their gates. The focused command is
`make test-soil-canonical-sticking`.
This closes the tested canonical-profile submission comparison, not the old
unordered failure, actual-tractor parity or the 4 ms/60 FPS requirement. Sorting
adds one dispatch and its net performance is unmeasured; defaults stay unchanged.
[Implementation, provenance and remaining limits](benchmarks/soil/2026-10-15-canonical-sticking-runtime/SUMMARY.md).

**Projected submission composition implemented, opt-in (2026-10-15):**
`batched-soil-projection` composes guard/Jacobian updates, the unchanged soil
substep, and guarded machine publication/load accumulation into one queue
submission instead of three. Each substep still uploads its current parameters
and submits before the next; no frozen-force batching, timestep change or
host-wait removal is claimed. Defaults retain separate submissions.
Ten late-particle-fault/rollback/retry combinations pass in 10.19 s. The exact
parity matrix passes two dense cases, then fails on the third step of the
non-native active-grid sticking case: maximum APIC difference 1.6997e-8 and
machine-rate difference 3.7253e-9; positions and fault outputs match exactly.
The stricter exact gate remains failed, not replaced by a looser tolerance.
The original generic six-body runtime check times out during construction at
its 90 s cap; that failure remains. Later reference-repeat isolation and the
small native runtime comparison are recorded above; full50k integrated
acceptance and speedup remain unproven. The focused Make target is
`test-soil-projection-commands` and currently has known failing gates.
[Implementation and retained failures](benchmarks/soil/2026-10-15-projection-submit-fixture/SUMMARY.md).

**Width-sized preparation evaluated, kept private (2026-10-15):**
Selecting 32/64/128 response lanes for 1--10/11--21/22--32 DOFs preserves exact
factors, responses and statuses across boundary-width fixtures, fault/recovery
checks and eight actual inputs. Actual paired median ratios are 0.975610--0.993056,
but typical savings are only **1--3 microseconds** and individual pairs have large
timing interruptions. Small cases include regressions of similar size. Do not
promote or repeat unchanged; this does not establish a meaningful full-solver
gain. Existing runtime selection and every acceptance gate remain unchanged.
[Private implementation, controls and raw audit](benchmarks/soil/2026-10-15-preparation-width/SUMMARY.md).

**Cooperative certificate-energy prototype verified (2026-10-15):**
Independent machine-energy products now have a private parallel implementation;
lane zero retains the original row-major energy and scale sums. No physical
check, dispatch or buffer budget changes. All 875 publication lifecycle cases
and one actual native/SoA input pass with exact outputs and archived state/report
agreement. The actual-input physical error is **3.22212e-7**, below 1e-6.
Timing is not reliable adoption evidence: paired upper-median change is -2.74%,
but individual pairs span -92.27% to +1066.75%. Keep this test-only; no runtime
speedup or 4 ms acceptance is established. The test harness now batches five
snapshot reads into one map, retaining all cases and assertions; the full suite
finishes in 74.45 s under the original 90 s cap, after a preserved initial timeout.
[Implementation, raw results and independent audit](benchmarks/soil/2026-10-15-publication-energy/batched/SUMMARY.md).

**Early Schur rejection evaluated and removed (2026-10-15):**
A private lower-block preflight avoids a reduced inverse attempt when its dense
fallback is already indicated. Six CPU/compiler checks, inverse edge controls
and one actual-packet replay pass, including exact five-buffer agreement and
eight repeats per arm. Paired median ratio is **0.996080**, only 0.39% lower amid
large shared timing variation; this does not establish a useful gain. The
candidate was not promoted and its live hooks were removed. Subnormal routing
was not certified. Do not repeat this unchanged experiment; target dominant
linear/certification work instead. All full-plan acceptance gates remain open.
[Frozen implementation and raw paired evidence](benchmarks/soil/2026-10-15-schur-preflight/SUMMARY.md).

**Cooperative preparation verified in the rendered tractor (2026-10-15):**
The viewer accepts `SOIL_LIVE_PREPARATION=cooperative` independently of its
Newton profile, validates the expected compiled backend and records its identity
in capture metadata and the overlay. Default preparation is unchanged. Both
viewer configurations pass 11 CPU checks (one GPU check intentionally ignored).
One bounded RTX 4080 run completes 33 frames with 50k particles, native prediction
and fixed-order transfer. All recorded machine poses are numerically exact and
both particle endpoints byte-identical against the retained native/default-preparation control;
surface and delayed-load ledger checks pass. The committed render was inspected.
Whole-physics median/p95 is **1901.649/2129.654 ms**, all 30 measured frames miss
60 Hz. Historical control timing is not a fresh paired speed comparison. This
does not establish full rates/history/force parity, long trajectories, calibrated
soil behavior or the 4 ms target.
[Frozen sources, capture and audits](benchmarks/soil/2026-10-15-viewer-cooperative-preparation/SUMMARY.md).

**Cooperative Cholesky added to the row-parallel opt-in (2026-10-15):**
The existing `row-parallel-newton-preparation` now also factors the mass matrix
cooperatively, retaining each entry's ascending sum order. Its backend identity
is `compensated-row-parallel-cooperative-factor`; defaults, resource budgets,
strict-rounding precedence and numerical gates are unchanged.

Synthetic checks, fault/reset controls and eight retained actual inputs pass with
**exact factors, responses and statuses**. Public-constructor replay and coupled
Newton/publication also pass. Runtime paired preparation ratios are 0.890--0.910:
another **9--11% lower component time** versus row-parallel with serial factors
(medians 0.138--0.167 -> 0.126--0.150 ms). This is not a full-frame speed claim;
tiny synthetic matrices showed about 0.001 ms overhead. The subsequent rendered
tractor check is recorded above; full trajectory and real-time acceptance remain open.
[Prototype](benchmarks/soil/2026-10-15-preparation-factor/SUMMARY.md);
[runtime checks and raw audit](benchmarks/soil/2026-10-15-preparation-factor/runtime/SUMMARY.md).

**Cooperative-row contact preparation integrated as opt-in (2026-10-15):**
`row-parallel-newton-preparation` is available through `molla-solvers` and the
`molla` facade. A contact-owned workgroup parallelizes forward substitution and
residual certificates while preserving per-row accumulation order, two mandatory
refinements, the eight-refinement cap, and 1e-6 component certification. Defaults
are unchanged; explicit strict-rounding preparation retains precedence.

Six synthetic fixtures, fault/reset controls, eight retained actual input packets,
public-constructor replay and coupled Newton/publication checks pass. Response
and status buffers remain exact against the control and prototype. Runtime paired
preparation medians fall from **0.282--0.391 ms to 0.137--0.167 ms**, with median
paired ratios 0.427--0.487 (**51--57% lower component time**). Timed submissions
allocate no GPU resources and perform no host transfers. This is not full-frame
speedup, trajectory acceptance or the 4 ms target. Full-workload validation remains
open; the subsequent cooperative-factor extension is recorded above.
[Prototype evidence](benchmarks/soil/2026-10-15-preparation-rows/SUMMARY.md);
[runtime integration and paired audit](benchmarks/soil/2026-10-15-preparation-rows/runtime/SUMMARY.md).

**Refreshed machine operators verified against actual first inputs (2026-10-15):**
A machine-only replay reproduces both retained 100-word mass matrices and all
20 native paired-rate words exactly after articulated pose refresh. Legacy/native
COM-Jacobian errors against CPU are 3.08e-8 / 1.95e-8, and whole-matrix errors
4.23e-8 / 6.83e-9, all below their existing 1e-6 bounds. One CPU contract and the
GPU replay pass; the replay takes 1.62 s including setup, without soil/Newton
construction. This is diagnostic duration, not simulation timing acceptance.
No production change is warranted by this first-input evidence. Return work to
dominant soil solve/preparation costs; contact-point geometry, later states,
cross-prediction trajectory and the 4 ms target remain unverified or unmet.
[Frozen reconstruction and independent checks](benchmarks/soil/2026-10-15-refreshed-machine-inputs/SUMMARY.md).

**CPU first-input reference added; stricter diagnostic failure retained (2026-10-15):**
Independent CPU CRBA and COM-Jacobian mass assembly agree to 5.86e-16. Both
retained first-substep packets pass established whole-matrix, free-rate and
response-residual bounds. An additional entrywise mass diagnostic fails for
Featherstone: a nominal zero coupling is -1.90735e-6, versus -4.17233e-7 for
native. Its 1e-6 assertion remains unchanged and the failure is retained.
This compares against unrounded authored inputs, not identical represented GPU
state; it neither diagnoses a production defect nor validates contact geometry.
No GPU rerun, production change or speed acceptance follows.
[CPU reference, exact metrics and retained failure](benchmarks/soil/2026-10-15-machine-input-reference/SUMMARY.md).

**Machine-input discrepancy localized, not yet a diagnosed defect (2026-10-15):**
The bounded first-frame trace now selects and verifies the machine predictor;
two CPU checks pass. A fresh Featherstone control times out during construction
at the unchanged 180-second cap, before simulation. The native trace completes
in 87.47 s, retaining eight packets and matching the rendered native initial and
frame-1 poses exactly. The failed control was not retried.

Against the historical fixed-order Featherstone first-substep packet, contact
bytes are identical, but mass, free rates, Jacobian rows and responses differ
before Newton. Source-manifest auditing limits this comparison to unchanged
upstream preparation; it is not a fresh paired control or proof that either
predictor is wrong. Inspect the machine inputs against an independent reference
before changing production arithmetic. Defaults, tolerances and performance
requirements remain unchanged.
[Input capture and scoped historical evidence](benchmarks/soil/2026-10-15-prediction-inputs/SUMMARY.md).

**Native machine prediction integrated into the viewer; parity fails (2026-10-15):**
`SOIL_LIVE_PREDICTION=native` / `--machine-prediction native` selects the existing
native control/prediction and shared-mass tractor path independently of Newton
profile and P2G transfer. Featherstone remains default. Native selection requests
f64 support; the worker verifies live ownership before Ready, and capture/overlay
identify the actual route. Eleven CPU tests pass per Newton profile.

Fresh same-source SoA/fixed-order captures finish all 33 frames in both machine
modes under the unchanged 180-second caps; endpoint terrain/count and delayed-load
audits pass, and both final images were inspected. **Cross-prediction trajectory
parity fails:** position/rotation/final particle component differences reach
2.819305e-4 m / 1.440376e-3 / 1.113534e-3 m against 2e-5. Initial particles are
exact; body poses first differ at frame 1 and exceed the bound at frame 17.
Native prediction is a diagnostic opt-in, not a default replacement.

Whole-physics median/p95 is 1159.56/2791.56 ms Featherstone and 2730.22/2993.71 ms
native. Other GPU applications were active and left untouched; these are
contended observations, not controlled speed evidence or isolated soil timings.
No 4 ms/60 FPS acceptance follows. The earlier passing Newton-profile comparison
used Featherstone in both arms and remains distinct from this new failed gate.
[Implementation, rendered captures and preserved failure](benchmarks/soil/2026-10-15-viewer-native-prediction/SUMMARY.md).

**Shared Schur matrix assembly evaluated, not promoted (2026-10-15):**
A test-only candidate avoids assembling mobility and coefficients twice when
reduced inversion falls back to dense elimination. Five CPU checks and inverse
edge controls pass. The retained actual input matches all five public buffers
exactly, with independent physical/publication checks and eight exact repeats per
variant. The first run omitted timestamps and is correctness-only evidence.
A separately recorded timed run finishes in 52.14 s under the unchanged 90-second
cap. Control/candidate medians are 16.615936/16.204800 ms, but the median paired
ratio is 1.005416 (candidate 0.54% slower); no reliable speed improvement is
established. **Keep test-only; runtime is unchanged.** No full-tractor rerun or
4 ms/60 FPS claim follows. Target larger solver costs next rather than repeating
this local optimization.
[Implementation, exact checks and raw timing evidence](benchmarks/soil/2026-10-15-schur-assembly-reuse/SUMMARY.md).

**Latest production repair: coupled Schur fallback restores bounded profile parity (2026-10-15):**
Isolating Schur inversion reproduces the full SoA discrepancy on the retained
common-input packet. The production path now uses existing dense inversion when
lower coupling is nonzero; decoupled blocks retain reduced inversion. Five public
buffers match byte-for-byte in fresh replays, with independent physical checks
and three exact repeats per profile. Tolerances and runtime defaults are unchanged.

Fresh fixed-order, actual-50k rendered captures pass the unchanged 33-frame
trajectory gate: maximum body position/rotation component differences are
1.192093e-7 m / 4.768372e-7, and final particle difference is 4.768372e-7 m.
Endpoint terrain/count and delayed-load audits pass; both final images were
inspected. This supersedes the earlier fixed-order failure for this bounded
fixture only; atomic transfer and longer trajectories are not revalidated here.
Sequential physics median/p95 is 856.47/985.24 ms volatile and 562.79/644.43 ms
SoA. Both miss all 30 sampled 60 Hz frames. **Performance remains far from the
target**; these are whole-physics observations, not isolated soil-budget timings
or a controlled speedup comparison. Prioritize solver speed next, without
loosening numerical gates or claiming calibrated agricultural behavior.
[Isolation](benchmarks/soil/2026-10-15-schur-input-isolation/SUMMARY.md),
[production guard](benchmarks/soil/2026-10-15-schur-coupling-guard/SUMMARY.md),
[rendered comparison](benchmarks/soil/2026-10-15-viewer-certified-schur/SUMMARY.md).

The entries below retain earlier outcomes; the latest evidence above supersedes
their common-input and bounded fixed-order parity failures, not their broader
performance or physical-acceptance limitations.

**Earlier Newton/publication velocity-scale repair; pre-guard parity failure (2026-10-15):**
The retained failure is contact 88: adding speculative gap to velocity scale
inflated 3.57814 to 60.28801, admitting normalized complementarity error
1.017331e-6 above the existing 1e-6 independent output gate. Removed that gap term
from Newton/publication velocity normalization and the CPU reference; gap remains
in separation and strict normal feasibility. Thresholds and iteration limits are
unchanged. New large-gap rejection and valid controls pass. The old actual packet
is rejected (normal/wall fault mask 24), with all outputs/rates cleared.

Fresh solves of the same actual packet now pass independent publication checks
and three exact repeats in both profiles (20.55/66.33 s). Physical errors are
about 1.017334e-8; both take 10 Newton / 51 linear iterations versus the old invalid
8/43 candidate. However, **common-input profile parity still fails**: maximum state/
output differences 6.768407e-5 / 6.866455e-5 exceed 2e-5; rates differ by 7.249128e-9.
This establishes a profile discrepancy without differing P2G inputs, not full
trajectory causation. Seven GPU controls, focused CPU suites and facade checks
pass; harness-selection/bitmask mistakes remain explicitly recorded separately.
The 4 ms target and full post-fix trajectory/integration acceptance remain open.
[Fix and retained parity failure](benchmarks/soil/2026-10-15-gap-independent-certificate/SUMMARY.md),
[original represented-output diagnosis](benchmarks/soil/2026-10-15-publication-complementarity/SUMMARY.md).

**Frame-14 baseline publication failure reproduced (2026-10-15):**
Added test-only selected-frame tracing with checked 64 MiB retained-trace storage,
up to 34 packets, and incremental pose records. Both CPU validation lanes pass.
Volatile captures all 34 frame-14 solves in 127.68 s and reproduces all recorded
viewer poses through frame 14 exactly. SoA times out at the unchanged 180-second
cap while creating its first Newton pipeline, before simulation; its machine
setup already consumed 108.35 s. No unchanged capture was retried.

A separately frozen common-input replay then fails on volatile substep 33 after
21.34 s. Three repetitions and the archived report/publication/state match exactly;
GPU status is zero and reference physical error is 4.465522e-7. However, the
independent published-contact oracle rejects normal complementarity at the
unchanged 1e-6 gate. **This is a baseline correctness failure, not demonstrated
SoA causation.** The remaining selected cases and SoA replay were not run.
Prioritize retaining the failing public outputs and repairing this contract,
without weakening gates. Full trajectory and 4 ms/60 FPS acceptance remain open.
[Frozen packets, timeout and replay failure](benchmarks/soil/2026-10-15-frame14-inputs/SUMMARY.md).

**Viewer transfer control implemented; later trajectory still fails (2026-10-15):**
`SOIL_LIVE_TRANSFER=fixed-order` / `--soil-transfer fixed-order` now selects the
paired tractor's controlled P2G path in both Newton viewer profiles; atomic remains
default. The worker verifies live transfer flags and scratch bytes before Ready,
and capture/overlay report the selected path. Ten CPU tests pass per profile.
Both frozen 33-frame actual-50k headless rendered captures complete under their
180-second caps; exact endpoint terrain/count oracles and delayed load/surface
policy checks pass. Both final images were inspected.

**The unchanged 2e-5 trajectory gate fails.** Logged body poses match exactly
through frame 13, then differ at frame 14. Rotation first exceeds the bound at
frame 17, position at frame 18. Maximum position/rotation/final particle component
differences are 2.733471e-4 m / 1.258940e-3 / 7.082224e-4 m. This rules out the
measured early P2G divergence as a sufficient explanation for the entire failure,
but does not identify the first differing substep/kernel at frame 14.
Sequential measured physics medians/p95 are 1098.02/1326.43 ms volatile and
762.18/956.14 ms SoA; both miss all 30 sampled 60 Hz frames. These are not a
controlled speedup comparison or 4 ms acceptance. Defaults, tolerances and
physical parameters remain unchanged; no failed capture was repeated.
[Controls, retained failure and renders](benchmarks/soil/2026-10-15-viewer-fixed-order/SUMMARY.md).

**Controlled paired transfer removes early divergence (2026-10-15):**
The regular paired-tractor API now exposes explicit atomic/fixed-order/shared-chunk
transfer selection; existing constructors remain atomic. With fixed-order P2G,
both actual 50k cold captures complete under 180 seconds (SoA 170.07 s, volatile
110.40 s). All six exported input/state buffers match byte-for-byte across the
first eight substeps. Each frame has 34 substeps, so this does not establish full
frame or long-trajectory equality. Gather scratch is 3,200,000 bytes plus 1,201,840
ordering bytes. Instrumented first-step wall spans are 2899.99/911.80 ms for
SoA/volatile: this is an input control, not a speedup or 4 ms acceptance result.
Both CPU suites, facade check and frozen provenance checks pass.
[Evidence](benchmarks/soil/2026-10-15-paired-fixed-order-transfer/SUMMARY.md).

**First-substep input divergence localized (2026-10-15):**
Added bounded, flushed test-only setup tracing and captured both actual-50k cold
profiles under the same frozen source, now including geometry program reuse.
SoA/volatile complete in 167.99/106.56 seconds under unchanged 180-second caps.
At substep zero, exactly **163 contact soil-mass words differ**, by at most
4.768372e-7; all other contact fields, machine mass, free rates, Jacobians and
prepared responses match byte-for-byte. States already differ after that solve,
and subsequent inputs diverge. The first input difference therefore precedes
Newton arithmetic; its upstream accumulation implementation and the earlier
33-frame failure's full cause are not established. Control/isolate soil-mass
accumulation next rather than repeat unchanged fixed-input replays.

Tracing attributes 84.518/86.827 seconds to volatile/SoA machine setup and
20.093/79.068 seconds to first Newton pipeline creation. Both two-test CPU trace
lanes and the offline structural audit pass. This is not a cache speedup proof,
steady-state benchmark, trajectory acceptance or 4 ms/60 FPS result.
[Paired inputs, setup spans and field-level evidence](benchmarks/soil/2026-10-15-cold-input-attribution/SUMMARY.md).

**Geometry program reuse (2026-10-15; production setup increment):**
Analytic contact geometry and Newton prepare/recover now share immutable pipelines
through the bounded per-Engine program cache, keyed by full source and entry-point
identity. Uniform dimensions, bind groups, state, outputs and faults remain local
to each owner. Existing driver pipeline-cache data is supplied when available.
Two GPU lifecycle tests pass in 54.33 seconds, including one/65-contact owners,
fault isolation, cache-clear execution and separate Engines; four existing geometry
regressions pass in 1.70 seconds. CPU shader/budget and facade checks pass. The new
`test-soil-geometry-program-cache-gpu` target is part of S03. This removes duplicate
construction, not solver work; no measured startup gain, resolution of the SoA
capture timeout, trajectory acceptance or 4 ms/60 FPS claim follows.
[Cache ownership and executed evidence](benchmarks/soil/2026-10-15-geometry-program-cache/SUMMARY.md).

**Cold-start profile isolation (2026-10-14/15; replay passes, paired capture incomplete):**
The live profiles use cold starts, unlike the historical retained-seed replay
below. Strict replay now validates either original initial-guess policy, rejecting
ignored or mismatched seeds. A 132.32-second actual-50k volatile capture saves the
first eight nontrivial substeps. Replaying its first two inputs through both
profiles gives byte-identical state/rates/output/status/publication, three exact
repeats per profile and maximum physical error 3.146373e-7 against 1e-6. Component
processes finish in 29.73/79.93 seconds under 90-second caps. The separate SoA
full-tractor capture hits its unchanged 180-second cap; that failed run is preserved
and not retried. No independent input-trace comparison, trajectory acceptance,
native-atomic causal attribution or speedup follows. Defaults and numerical limits
are unchanged; 4 ms/60 FPS remains unmet.
[Cold provenance, bounded captures and replay evidence](benchmarks/soil/2026-10-14-newton-cold-profile/SUMMARY.md).

**Fixed-input profile isolation (2026-10-14; two captured problems pass):**
Strict test-only volatile/SoA replay now records actual backend, hashed input/seed,
limits, exact-repeat evidence and independent physical error. Both historical
actual-tractor problems produce bit-identical state/rates/output/report/publication
across profiles, with three exact repeats per profile and maximum physical error
2.495474e-7 against 1e-6. The two setup-inclusive processes take 22.45/77.48 seconds
under unchanged 90-second caps. This does not clear the viewer trajectory failure
below: these are not its first-divergence inputs. Capture and compare evolving
machine/soil/contact inputs before attributing that failure to native atomics or
Newton arithmetic; do not repeat the unchanged broad timing comparison.
[Strict replay and offline audit](benchmarks/soil/2026-10-14-newton-profile-replay/SUMMARY.md).

**Opt-in optimized live solver profile (2026-10-14; cross-profile trajectory fails):**
The live viewer now supports `SOIL_LIVE_PROFILE=soa`, selecting the existing
optimized Newton profile stack through coefficient-major scratch; `volatile`
remains default. Constructed backend identity is checked before Ready, Make rejects
unknown selections, and the executable rejects an expected/compiled profile
mismatch before GPU setup. Both profile builds pass nine CPU contracts. Fresh
same-source 33-frame rendered runs complete with exact endpoint surface oracles
and valid delayed-force ledgers. However, strict cross-profile pose/particle parity
fails: maximum position/rotation component differences are 2.670288e-5 / 1.825690e-4,
and final particle difference is 7.838011e-5 m against the unchanged 2e-5 bounds.
Sequential physics medians are 1407.73 ms volatile / 751.77 ms SoA, with all 30
measured frames missing 60 Hz in both runs. These are not controlled speedup
evidence or grounds for promotion. Keep SoA diagnostic/opt-in and volatile default.
Both final images were inspected. Results and offline drift localization are recorded
in the [profile packet](benchmarks/soil/2026-10-14-viewer-soa-profile/SUMMARY.md).
This is not an isolated layout measurement, interchangeable-trajectory admission,
default promotion or full acceptance.

**Viewer asynchronous force migration (2026-10-14; worker and render verified):**
Normal live-worker frames no longer call the blocking frame-load reader. Force
observations carry their own captured frame and interval, tolerate pending/delayed
values and refresh independently while paused. The overlay and JSONL expose force
age and byte/wait accounting; final captures still explicitly require matching
loads. Nine CPU checks and the actual-50k paused-worker test pass (74.99 seconds
including cold setup). A fresh four-frame headless capture records one pending
force frame, two delayed force frames and two idle completions; the final force
frame matches physics. Both screenshots were inspected. Actual-worker and rendered verification are recorded
in the [migration packet](benchmarks/soil/2026-10-14-viewer-async-loads/SUMMARY.md).
This removes an optional observer synchronization, not the coupled solver's
remaining synchronization or the unmet 4 ms/full-plan gates.

**Asynchronous frame-integrated loads (2026-10-14; production API):**
World and tractor APIs now request/poll frame-integrated impulses without an
explicit GPU wait, reusing the existing single staging buffer. Captured frame,
duration, body association and owner/generation travel with the observation;
reset invalidates without prematurely reclaiming in-flight storage. Blocking
reads reject a pending asynchronous stage, and both readers reject foreign devices.
Three GPU checks pass in 1.01 seconds, including 65-body captured-buffer overwrite
parity, bounded reuse, stale-model/device rejection and invalid-payload recovery.
The facade compiles. Viewer consumer migration is recorded above; injected
map/device failures and broader actual-world lifecycle evidence remain
open, as do full residency and 4 ms acceptance.
[Implementation and checks](benchmarks/soil/2026-10-14-frame-load-observation/SUMMARY.md).

**Idle viewer surface completion (2026-10-14; worker and render verified):**
The live worker now polls pending surface observations while awaiting commands.
A separate bounded surface event refreshes geometry and its revision label without
advancing the committed frame, simulation time, bodies or step scheduler. Normal
command handling takes priority; no GPU wait was added. Eight CPU tests pass,
including atomic rejection of invalid geometry/revisions and unchanged physical
state. The actual-50k paused-worker check passes in 95.00 seconds including cold
setup; both idle surfaces match independent particle geometry exactly. A fresh
four-frame headless capture consumes two separate surface events without advancing
the committed frame and retains exact endpoint geometry. Both images were inspected.
Dedicated actual-50k paused-worker and rendered checks are recorded in the
[verification packet](benchmarks/soil/2026-10-14-viewer-idle-surface/SUMMARY.md).
This addresses stale paused geometry, not the remaining blocking compact loads
or the unmet 4 ms/full-plan acceptance gates.

**Viewer asynchronous surface migration (2026-10-14; headless render verified):**
Normal live-worker frames now request/poll a capacity-one surface observation and
reuse labelled older geometry without a surface wait; matching bootstrap/final
captures remain explicit. Unchanged surface revisions avoid mesh reconstruction.
Seven CPU checks and a four-frame actual 50k headless Bevy capture pass; three normal
frames show age-one geometry, and both endpoints have exact counts/heights against
the independent particle oracle. Both images were inspected. Headless capture is
explicit (`SOIL_LIVE_HEADLESS=1`), not a replacement for interactive acceptance:
Wayland timed out acquiring a swapchain and Xvfb crashed during window creation;
both failures are preserved. Some overlay glyphs remain incomplete. Compact load
reads still block and 4 ms/60 FPS remains unmet. The paused-geometry limitation of
this historical capture is addressed by the idle-completion change above.
[Implementation, images and remaining gates](benchmarks/soil/2026-10-14-viewer-headless-surface/SUMMARY.md).

**Asynchronous geometric surface observations (2026-10-14; production API):**
Prepared surfaces now expose nonblocking request/poll, owner/revision/generation
tags and bounded backpressure using the existing single staging buffer. Reset
invalidates without prematurely reusing in-flight memory; obsolete generations
are discarded without downloading their payload. Source/device-checked MPM and
committed-frame-tagged world/tractor wrappers are available. Two CPU and six GPU
surface checks pass, including delayed source updates, reset, recovery and resource
reuse. The viewer migration is recorded above; isolated device-loss/error gates
and full S03 acceptance remain open. Bevy uses wgpu 27 while
physics uses wgpu 29, so raw buffer sharing is not a drop-in replacement.
[Implementation, checks and remaining scope](benchmarks/soil/2026-10-14-surface-observation/SUMMARY.md).

**Straight-line Newton preconditioner (2026-10-14; test-only, not promoted):**
An unrolled six-by-six application preserves all 36 protected arithmetic terms,
their order and padding while removing dynamic private vector/array indexing.
Two captured actual-tractor seeds pass exact cross-mode, archived-output and
32-repeat checks in 75.38 s including setup; independent physical error remains
at most 2.495474e-7. Paired median changes are only -0.47%/-0.82%, with individual
regressions up to +71.76%; continuous utilization spans 0–98%. This is not reliable
speedup evidence, so runtime/defaults remain unchanged and no unchanged rerun is
planned. [Source, raw evidence and decision](benchmarks/soil/2026-10-14-newton-precondition-unroll/SUMMARY.md).

**Production grouped DEM bounds (2026-10-13; audited 2026-10-14):**
The explicit retained-refit path now constructs 64-leaf subtrees cooperatively and
atomically expands upper bounds. At 50k/leaf-eight, tree updates use two dispatches
instead of fourteen, without new buffers or bindings; interval-one full rebuilds
remain default. Nine tree/trajectory cases, separate 52-step/overflow checks and
nine public topology/residency tests pass. Eight hot refits now use 64 dispatches
(previously 104), with zero observed explicit allocations/downloads.
Against the previous ordered per-level path, prototype/production paired median
reductions are 8.65%/7.97%. Production improves 11/12 pairs, but retains a
**11.702782 ms tail spike** versus reference 6.637560 ms; tail improvement is not
proven. Production marginal medians are 4.239814/3.818169 ms per **0.002 simulated
seconds**, still far above 4 ms per 1/60 second. These are DEM component results,
not actual-machine or MPM/Newton acceptance.
[Implementation, audited samples and limits](benchmarks/soil/2026-10-13-dem-grouped-runtime/SUMMARY.md).
The standard Make aggregate now includes exact grouped tree/trajectory and ordered
overflow checks: two CPU and eleven GPU tests pass on 2026-10-14. This rerun had
82–93% endpoint GPU utilization and is not a clean timing comparison.
[Reusable gate and preserved evidence](benchmarks/soil/2026-10-14-dem-grouped-public/SUMMARY.md).

**Production ordered DEM refit (2026-10-13; per-level baseline before grouped bounds):**
The explicitly selected retained refit path now uses the order-preserving kernel;
full rebuilding remains default. The ordered pipeline is prepared lazily and
reused. All original public topology/residency checks and the formerly failing
50k three-arm benchmark pass; the production 52-step/overflow test is bit-exact.
Eight warmed refits observe zero explicit buffer allocations/downloads and 104
dispatches versus 208 for full rebuilding. Two 12-pair public packets favor refit
in every pair, with median paired time reductions of 40.94% and 43.51%. Marginal
refit medians are 3.74426 / 3.71337 ms per **0.002 simulated seconds**—about
31 ms per simulated 60 Hz frame, still above the 4 ms target. The bounded
`make test-soil-dem-refit` aggregate now backs S04. Worst occupancy/quality,
absolute performance and integrated tractor gates remain open.
[Public implementation and evidence](benchmarks/soil/2026-10-13-dem-ordered-public/SUMMARY.md).

**Order-preserving DEM refit prototype (2026-10-13; superseded by production integration above):**
The first 50k refit divergence is now isolated: at step 5 the affected particle has
identical contact IDs and pair forces, but a different force-addition order. A new
candidate queries the refitted tree and processes contacts in current Morton/ID
order, using complete 32-hit batches rather than dropping overflow. Dynamic scene
bounds preserve the original quantization path. All 52 states match rebuilt
positions/velocities bit-for-bit for both 50k particles and a dense 65-particle case.
One 12-pair solver-plus-export packet reports a 0.64324 median paired time ratio;
final particle states/heights/sinkage are exact and surface velocities pass 2e-5.
Candidate marginal median is 4.25005 ms per **0.002 simulated seconds**, not per
frame; the 4 ms/60 FPS goal still fails. Production integration and broader
lifecycle/repeated timing gates remain. Constant-bound and overstrict surface-test
failures are retained. [Evidence and limits](benchmarks/soil/2026-10-13-dem-ordered-timing-surface/SUMMARY.md).

**FEM topology admission (2026-10-13; invalid-input correctness):**
The shared tire input validator now rejects out-of-range/repeated constraint
vertices, malformed CSR offsets, missing/duplicate/misowned incidence, and invalid
bead/tread sets before packing or GPU upload. Valid arbitrary adjacency order,
isolated nodes and absent optional constraints remain supported. Three new CPU
tests cover 23 corruptions at two public boundaries; 16 existing CPU tests and
two RTX 4080 GPU tests pass, including valid resident-atlas parity. This adds
linear host preparation validation, not resident-step work or a speedup. The
verification harness missed pre-run executable hashes; that provenance limitation
and its hardened extraction are recorded rather than silently claiming a full seal.
[Scope, checks and limitations](benchmarks/soil/2026-10-13-tire-topology/SUMMARY.md).

**Newton clock controls (2026-10-13; diagnostic overhead measured):**
Extended the monolithic test probe to eight nested functions and added same-shader
sampling-off/on paired checks. Two saved inputs pass in 69.35 s including setup,
with exact cross-mode public buffers, archived success and 32 exact timed repeats.
Sampling adds paired upper-median costs of 0.42235% / 0.31051%; an 8.81873% pair
and timing transition are retained. This excludes static code/register effects
versus an uninstrumented binary. Inner products run 96 times but record much smaller
inclusive intervals than the full hybrid solve; counts alone do not justify another
dot-product micro-optimization. No production or performance acceptance change.
The initial shader-string/test-selection failure is retained, and the corrected
harness requires a nonzero test count. The stale S08 ledger now includes previously
verified synthetic substep-18 rollback, after rechecking its archived integrity.
[Paired overhead, nested records and limits](benchmarks/soil/2026-10-13-newton-clock-controls/SUMMARY.md).

**Cooperative publication projection (2026-10-13; rejected, test-only):**
Implemented axis-parallel machine-velocity projection during certification, keeping
the same per-axis arithmetic order and all physical/fault checks. Two current saved
inputs preserve exact cross-mode results, archived successful state/report/publication
and 32 timed self-repeats; independent physics passes. However, all 16 timing pairs
regress: paired upper medians +0.35285% / +0.45105%. This shared-compiled-code probe
does not compare against the original binary. The candidate is rejected; defaults
and production remain unchanged. Initial pre-GPU width-contract failure is retained.
[Implementation, rejected timing and limits](benchmarks/soil/2026-10-13-publication-axis-projection-width/SUMMARY.md).

**Monolithic Newton clock probe (2026-10-13; diagnostic, not a speedup):**
Added test-only subgroup-clock instrumentation inside the current solver, with a
retained 128-byte trace and unchanged physical algorithm/budgets. The RTX 4080
current-success replay passes in 67.15 s including setup: archived state/report/
publication match, five public buffers repeat exactly three times, and independent
physical checks pass. The final repeat records 17 evaluations, 8 preparations,
8 hybrid solves, 9 certifications and zero conditioned releases. Certification
records 2,408,150 ticks alongside hybrid solving's 3,695,817; these are inclusive
lane-zero observations, not additive phase shares or elapsed-time units. Overhead,
sampling variability and compiler movement still need controls. No default or
throughput acceptance changes.
[Clock contract, raw results and next steps](benchmarks/soil/2026-10-13-newton-phase-clock/SUMMARY.md).

**Newton compiler-cache integration (2026-10-13; setup only):**
Newton pipelines now honor the Engine's existing optional compiler cache rather
than ignoring it. A frozen RTX 4080 successful-packet lifecycle passes in 66.91 s:
initial setup 66,136 ms, same-device rebuild 111 ms, fresh-device snapshot rebuild
100 ms. All phases preserve archived state/report/publication and three exact
repeats; independent output checks pass. These sequential observations are not
paired cache-disabled controls, and Newton work remains 8/24/8 steps/linear/trials.
Callers still explicitly request/enable the cache; no disk import or default change.
This addresses repeated setup, **not per-frame Newton cost or the 4 ms target**.
[Lifecycle evidence and limits](benchmarks/soil/2026-10-13-newton-compiler-cache/SUMMARY.md).

**Strain-dependent fiber tangent (2026-10-13; optional integrator corrected):**
CPU and single/atlas GPU semi-implicit stiffness now use the existing fiber force
law's strain-dependent local tangent instead of constant fiber parameters. The GPU
shares force/exponential evaluation; material parameters and force law are unchanged.
Independent 180-case CPU derivatives and 160-case GPU force/integration checks pass
(GPU maximum scaled error 1.549e-6 versus 2e-5). Scene 120 sets the stiffness scale
to zero, so this cannot repair its explicit trajectory: its maximum remains 16.87255.
A GPU skip branch measured about 1.75% slower on the cleaner packet and was rejected;
its noisy second packet and setup failure are retained. CPU disabled-path skip stays.
No new speed or full-plan acceptance claim.
[Implementation, rejected optimization and limits](benchmarks/soil/2026-10-13-tire-fiber-tangent-adopt/SUMMARY.md).

**Analytic bending paired timing (2026-10-13; component only):**
Eight alternating legacy/analytic pairs on one saved 1,440-node input reduce median
GPU node-pass time from 0.061657000 to 0.057897250 ms (6.10%). Each sample has 32
warmups and 64 timestamped immutable-input dispatches; both routes repeat exactly
and preserve force words. A second saved input has a timing regime change/outlier,
so its 0.958847 aggregate ratio is not stable confirmation. All samples are retained.
No full-frame, advancing-trajectory or 50k speed acceptance follows from this result.
[Timing method and noise](benchmarks/soil/2026-10-13-tire-analytic-bending-timing/SUMMARY.md).

**Production analytic tire bending (2026-10-13; constraint verified, trajectory still fails):**
Single/atlas production shaders now share the analytic unsigned-dihedral derivative,
removing 24 perturbed angle evaluations per hinge gradient. The complete XPBD correction
passes 36 independent five-point-reference cases (scaled position/velocity maximum
2.502e-6 versus 2e-5), and derivatives pass on 1,460 synthetic/saved hinges. The old
finite-difference implementation remains a test-only control. Current-law force
words are preserved; direct replays repeat exactly. The six-frame viewer trajectory
still fails at 16.8726, so cavity loading remains default-off. No timing or full-plan
acceptance claim. [Implementation and retained failure](benchmarks/soil/2026-10-13-tire-analytic-bending/SUMMARY.md).

**Edge-relative tire deformation (2026-10-13; arithmetic fix, trajectory still fails):**
Single-tire, atlas and matrix-free GPU deformation gradients now use edges relative
to tetrahedron vertex zero rather than cancellation-prone absolute positions.
The dependent shape gradient is reconstructed consistently; single/atlas sidewall
ply evaluation shares the edge-relative calculation. A 63-case translated skew-tet
energy oracle reduces maximum scaled errors from 8.440e-5 to 1.922e-7 for actual
single/atlas forces and from 8.616e-4 to 1.971e-6 for matrix-free deformation/stress.
The unchanged gate is 2e-5. This does not recover precision already lost when world
positions were quantized. Existing components pass; the six-frame viewer trajectory
still fails at 16.8326, and experimental cavity loads remain default-off. No speed
or calibration claim. [Frozen controls and results](benchmarks/soil/2026-10-13-tire-local-gradient/SUMMARY.md).

**Tire stress transpose correction (2026-10-13; physical bug fixed, trajectory still fails):**
The single-tire, atlas and matrix-free WGSL laws used F^-1 where their Neo-Hookean
stress requires F^-T: the cross-product helper already returned cofactor columns,
and an extra transpose reversed it. Corrected all three GPU expressions; CPU
row-major math is unchanged. An independent energy-derivative check on 21 rotation,
shear and general cases now passes with maximum scaled errors 6.972e-8 for actual
single/atlas forces and 3.576e-7 for the matrix-free stress expression; old errors
were 2.333/2.333/14. Retained-owner, atlas routing, smoke and analytic bending
checks pass. The six-frame viewer trajectory still fails, now at maximum 20.1949
versus 2.1018 before correction; preserve this unfavorable result. Direct repeats
and exact CPU fallback still pass. This fixes the law, not stability or speed.
[Negative control, correction and failed trajectory](benchmarks/soil/2026-10-13-tire-stress-transpose-fix/SUMMARY.md).

**Analytic tire bending candidate (2026-10-13; private, not adopted):**
Implemented an analytic unsigned-dihedral gradient with atan2 angle evaluation.
The candidate replaces the 24 perturbed angle evaluations used for each gradient
with a direct derivative. CPU five-point derivative/rigid-mode checks pass on
144 hinges; GPU derivatives pass on 1,424 synthetic/saved hinges with maximum
scaled error 9.069e-7 against the unchanged 2e-5 gate. Saved third-step input-pair
velocity difference falls from 0.005697697 to 0.000416756, with unchanged force
output words and exact candidate repeats. The earlier integration error therefore
remains above tolerance; no full trajectory repair or measured speedup is claimed.
Production shaders/defaults are unchanged. Initial fixture-path failure is retained.
[Implementation, numerical evidence and scope](benchmarks/soil/2026-10-13-analytic-tire-dihedral-path/SUMMARY.md).

**Third-substep FEM stage capture (2026-10-13; two amplification stages):**
Saved-input node replays match both live third-substep outputs bit-for-bit, and
instrumented replays preserve those output words twice per input. Velocity error
already exceeds 2e-5 after force integration (0.000416756); dihedral projection
then increases it to 0.005697697. Volume, bead, edge, ply and floor stages do not
increase the sampled maximum. Thus dihedral projection is not the sole cause.
Both complete input packets and stage traces are retained for direct replay,
avoiding repeated scene construction. The probe passes in 2.39 s; it is diagnosis,
not a repair or cleared trajectory gate. No production physics changed.
[Stage evidence and packet format](benchmarks/soil/2026-10-13-cavity-stage-probe/SUMMARY.md).

**Real-tire held-load isolation (2026-10-13; trajectory failure localized):**
A first-frame control using the exact same held GPU cavity loads matches guarded
stepping across all 11,616 physical words and FEM force rows. The guarded schedule
is not the source of this sampled discrepancy. Same-input load errors remain below
the 2e-5 component gate (maximum scaled error 1.781e-6), but ordinary direct FEM
stepping amplifies the changed loads past the unchanged 2e-5 trajectory gate at
substep 3 (0.005698), reaching the same 0.365108 difference at substep 24.
The isolation passes in 1.14 s; it does not clear the failed trajectory. UI metadata
now explicitly labels that failure. Next locate amplification inside the FEM
update on identical captured inputs; do not relax limits or promote the mode.
[Held-load controls and evidence](benchmarks/soil/2026-10-13-cavity-held-load-isolation/SUMMARY.md).

**Real-tire cavity trajectory (2026-10-13; failed, mode remains default-off):**
The actual Scene 120 five-tire update now has a headless GPU trajectory/fallback
fixture. Six 24-substep frames preserve exact direct-reference repeatability and
CPU mirrors, but resident-vs-CPU-held loads exceed the unchanged absolute 2e-5
gate in the first frame (0.3651 maximum difference), reaching 2.1018 overall.
Invalid fifth-cavity input correctly rejects and all five CPU fallbacks match
pre-frame inputs exactly. The run fails in 2.41 s, not by timeout. Next isolate
load calculation from guarded dispatch using identical held GPU loads; do not
promote or rerun unchanged. Rendered and physical acceptance remain open.
[Frozen failed trajectory](benchmarks/soil/2026-10-13-cavity-viewer-trajectory/SUMMARY.md).

**Cavity-aware frame observation (2026-10-13; experimental viewer integration):**
The retained frame bridge now copies the cavity admission header in the same
submit/map as physical state and rejects unadmitted or sticky-faulted frames.
It binds the prepared cavity identity: preparing or replacing cavities requires
a new reader. Cavity mode adds 16 staging/readback bytes; legacy layout is unchanged.
Scene 120 exposes an opt-in resident-cavity mode, evaluates pressure once before
its 24 substeps, and prevalidates all tire/chassis outputs before updating CPU
mirrors or falling back. The steady configuration path skips CPU cavity integration.
One layout test, three viewer checks, GPU-only compilation and five GPU component
checks pass, including fault/readback/budget/identity cases. Real-tire coupled
trajectory and rendered validation remain pending; the viewer checks are not
rendered evidence. Blocking observation, the 4 ms soil target and whole-frame
performance remain unresolved. Defaults are unchanged.
[Implementation and evidence](benchmarks/soil/2026-10-13-cavity-frame-observation/SUMMARY.md).

**Guarded cavity frame consumer (2026-10-13; prior API checkpoint):**
The retained FEM owner now supports explicit cavity preparation and frame stepping.
It evaluates all cavity loads once at the frame boundary, admits them together,
and holds them through substeps. A bad record latches a fault and gates all nine
coupled stages; current-to-next copies preserve physical state across parity
changes. Configuration refresh owns changed pressure uploads; frame stepping
observes zero hot GPU buffers/uploads/readbacks. Four analytic two-tire frames
with ten total substeps match the same-load direct path bit-for-bit; second-wheel
fault, odd/even no-advance, sticky repair and existing direct regressions pass.
Two CPU and five GPU tests pass under the existing cap. Admission does not certify
later solver stability or device-fault recovery. Viewer status observation,
real-tire coupled trajectories and readback removal remain open; no speedup or
4 ms/60 FPS claim follows.
[Evidence](benchmarks/soil/2026-10-13-cavity-frame-optional-metrics/SUMMARY.md).

**Resident tire cavity producer (2026-10-13; S03 integration pending):** Added a
borrowed CPU cavity-surface descriptor and `PreparedTireFemCavity` to compute
volume, polytropic pressure and per-bead force directly from resident node/hub
buffers. It retains both position parities, explicit pressure refresh and a
validity-tagged output. Invalid records must not be consumed as zero loads.
CPU validation and 48 deformed real-tire cases pass, maximum scaled CPU/GPU error
4.991e-7 against the component 2e-5 bound; each hot dispatch observes no GPU buffer
allocation, host upload or readback. The existing viewer still uses blocking
readback: guarded consumer integration, once-per-frame pressure scheduling,
coupled trajectories and asynchronous display observation remain required.
No full-frame timing, physical calibration or 4 ms/60 FPS claim follows.
[Evidence](benchmarks/soil/2026-10-13-resident-cavity/SUMMARY.md).

**Matched warm/cold work comparison (2026-10-13; no material gain):** Implemented
an identical-input paired benchmark with one compiled method-31 solver, retained
zero/history seed buffers, eight alternating pairs per packet, exact repeats and
independent accuracy checks. Both first-frame packets require **8 Newton steps,
24 linear iterations and 8 trials** in either mode. Paired warm-minus-cold upper
medians are +0.04352% and +0.02656%; no meaningful speedup is established. Five
CPU tests and the GPU batch pass (65.40 s including setup, unchanged 90 s cap),
with 32 exact five-buffer repetitions, maximum physical error 2.496e-7 and public
delta 1.4902e-8. Warm starts stay optional; no new tractor run or default promotion.
Do not pursue history seeding alone or repeat rejected continuation-constant
sweeps as the performance solution.
[Evidence](benchmarks/soil/2026-10-13-newton-warm-cold-cost/SUMMARY.md).

**Warm success replay provenance (2026-10-13; verified prerequisite):** Added
test-only same-solve initial-guess snapshots and conditional seed latching, rather
than pairing a last-nontrivial success with the latest attempted solve's seed.
Missing-seed exports/replays are rejected. Four CPU and three GPU lifecycle tests
pass. The actual first 50k frame captures both parity packets in 151.70 s including
cold setup; each contains 251 nonzero seeded contacts. Three replays per packet
pass in 64.91 s including cold setup, with exact archived report/publication/state
and exact repeated rates/outputs; independent errors are at most 2.496e-7.
The original fixture's missing protected-shader device feature was corrected;
the failed attempt remains archived. No production overhead, solver arithmetic,
iteration limits or defaults changed. This establishes replay provenance, **not
a warm-start speedup, trajectory acceptance or 4 ms/60 FPS**.
[Evidence](benchmarks/soil/2026-10-13-newton-success-seed-protected/SUMMARY.md).

**Newton vector workspace experiment (2026-10-13; S06, rejected):** Added a
test-only field-major layout for direction/vector/scratch on top of the admitted
block/inverse SoA profile. It preserves all eight Six-record words, buffer sizes,
bindings, arithmetic, solver limits and whole-record Restrict bounds behavior.
This is linear component-major addressing, not the previously rejected tiled
basis layout. Two CPU/parser/address/compiler checks and the bounded GPU replay
pass: eight captured inputs, 40 exact candidate/control public buffers plus 40
archived-control matches, 128 repeat checks and 144 timestamp records. Maximum
physical residual is 8.042703606542908e-7; public delta is zero.

However, **all eight paired medians regress 7.16–12.88%**. GPU capture takes
84.16 s within the unchanged 90 s cap, including parallel cold preparation.
The candidate remains test-only; production is unchanged and there is no actual
tractor rerun or speedup claim. Do not iterate on vector layouts without new
instruction/cost attribution. [Rejected vector-layout evidence](benchmarks/soil/2026-10-13-newton-vector-soa/SUMMARY.md).

**Combined frame readback (2026-10-13; S03 increment):** Scene 120's default
combined FEM/vehicle path now uses `PreparedTireVehicleFemReadback`. One retained
staging buffer and CPU row vector replace the separate node/hub/chassis readers:
six copy submissions and eight map/waits become one submission and one map/wait
per frame. Unused FEM force rows are omitted; fresh position/velocity/hub data
still feeds the CPU mirror and next-frame cavity calculation. Staging/download
size is `32*nodes + 64*wheels + 64` bytes (46,464 for five 288-node tires).
The cache clears with prepared stepping on rebuild/reset/mode change/error.

One layout/budget CPU test and two executed viewer tests pass. Four GPU fixtures
pass under one unchanged 90 s cap (chassis 0.75 s, combined FEM 0.69 s). Four
changing-configuration frames match existing node, hub, planar/vertical, spin and
suspension observations; the two-tire fixture downloads exactly 448 bytes/read,
with zero observed hot buffer creation or upload. Wrong owner/CPU engine,
undersized budget before allocation and logical device loss reject. Poll/map
failure latches the reader unusable, but injected map/poll failure is not covered
by this capture. This remains a **blocking frame bridge**, not asynchronous
physical feedback, whole-frame residency, rendered acceptance or 4 ms/60 FPS.
[Retained frame bridge evidence](benchmarks/soil/2026-10-13-fem-frame-readback/SUMMARY.md).

**Tire frame overhead (2026-10-13; earlier S01/S03 increment):** Scene 120 no longer
clones and CPU-steps its animated reference tire in normal GPU mode. CPU
comparison is an explicit, default-off control shared by all three GPU paths;
disabling it clears comparison validity. GPU stepping, substeps and readbacks
are unchanged. Per-frame FEM parameter construction now directly evaluates the
existing cavity bead-force scalar instead of exporting full mesh/topology for it.

Four focused CPU tests pass: scalar/snapshot bit equality over 36 pressure/mesh/
deformation cases, paired scalar cost, viewer opt-in selection and the resident
path contract. Eight alternating paired rounds on a 288-node/640-tet tire measure
64 scalar extractions at **1.709 ms via snapshot vs 0.0367 ms directly** (upper
medians; 512 calls/arm). This is isolated host cost, not GPU or whole-frame timing.
The initial capture's whitespace-sensitive source assertion failed and was
corrected; both runtime and source viewer tests now execute and pass. Full
4 ms/60 FPS, rendered agricultural behavior and physical calibration remain open.
[Frame-overhead evidence](benchmarks/soil/2026-10-13-tire-frame-overhead-contract/SUMMARY.md).

**Actual delayed fused rollback (2026-10-13; S08/S10 increment):** The unchanged
50k-particle native Newton/SoA tractor now has an actual delayed-fault regression.
After three frames, a test-only fused gather hook lets 17 internal substeps
complete and injects the existing motion-fault bit in substep 18. The diagnostic
gather counter is exactly 120 (102 prior + 17 successful + one rejected gather).
This is **synthetic fault injection**, not a naturally exceeded physical bound.

The frame restores the complete soil checkpoint, four persistent machine
precision streams, all six CPU bodies and committed time; render publication is
invalidated. Restoring the normal pipeline and retrying yields exact soil, all
nine machine streams and CPU history against two uninterrupted controls for all
33 frames / 34 substeps per frame. The GPU fixture passes in **143.19 s** under
the unchanged 180 s cap. Offline audits check 66 frame comparisons, 99 history
records, 50 raw comparisons and archived final-soil equality; corrupt/missing/
duplicate history is rejected. A first attempt failed before the intended fault
because the test hook invalidated only the inner preparation epoch; that capture
is retained, and the corrected layout-preserving hook keeps the epoch unchanged.
Production physics/defaults are unchanged. Other fault classes, real-time
performance, physical support/traversal and full-plan acceptance remain open.
[Delayed rollback evidence](benchmarks/soil/2026-10-13-fused-tractor-late-retry-epoch/SUMMARY.md).

**Fully prepared vehicle/FEM loop (2026-10-13; earlier S03 increment):** Scene 120 now
uses `PreparedTireVehicleFemStep`: nine ordered dispatches (FEM nodes, six chassis
stages, brush-to-hub assembly, hub integration) in one submission per substep.
The existing 24 substeps and physics are unchanged. Explicit frame configuration
refresh uploads only changed FEM/hub/wheel rows or force scale. Each substep
restores immutable FEM/base-hub parameters using device copies, preventing
brush-force accumulation in the mutable assembly destination. Four independent
parity selectors advance only after submission. Total owned buffers are
`608*N + 468` bytes (2,900 for four wheels), excluding existing atlas/maps,
CPU storage, driver overhead and command allocations.

One CPU layout/budget test and four GPU fixtures pass: three chassis regressions
in **0.67 s**, combined FEM in **0.66 s**, under one unchanged 90 s cap. Four
steps with changing pressure, FEM/hub parameters and force scale match both
legacy schedules exactly for brush, FEM positions/velocities/forces and hubs.
The 36 combined dispatches observe **zero hot buffer allocation, upload or
readback**; unchanged configuration refresh also uploads nothing. Budget/count,
foreign-owner/CPU-engine and device-health rejection, plus queued owner/prepared
drop, pass. Viewer and facade compilation are checked. This is not rendered
acceptance, physical calibration/stability, whole-frame residency (viewer still
reads back), native Newton performance or full S03 completion.
[Full prepared FEM evidence](benchmarks/soil/2026-10-13-prepared-vehicle-fem/SUMMARY.md).

**Prepared chassis application adoption (2026-10-13; earlier S03 increment):** Scene 120
now retains the prepared chassis across its existing 24 FEM substeps and frames.
Explicit fixed-count wheel configuration refresh uploads `96*N` bytes only when
rows change, with no buffer allocation; unchanged pressure uploads nothing.
The combined FEM API preserves node → chassis/brush → hub ordering and validates
the prepared owner before FEM advancement. Rebuild, reset, mode changes and GPU
failure invalidate the viewer cache. Existing CPU fallback is unchanged.

Viewer compilation and four GPU fixtures pass (two suites of **0.66 s** each,
one unchanged 90 s cap). Four pressure-changing combined substeps match legacy
brush, node positions/velocities/forces and hubs exactly; two-wheel refresh is
192 bytes with zero observed allocation. The first capture failed because an
initial-step velocity-sign assertion was incorrectly extended to later steps;
the legacy fixture itself reverses into its existing velocity clamp. The fresh
capture preserves initial-step sign and all-step exact parity. This establishes
scheduling equivalence, **not stable physical dynamics or rendered acceptance**.
FEM allocation/transfers, native Newton throughput and full S03 remain open.
[Adoption evidence](benchmarks/soil/2026-10-13-prepared-chassis-adoption-parity/SUMMARY.md).

**Prepared complete brush chassis loop (2026-10-13; S03 increment):** Added
`PreparedTireVehicleChassisStep` for the existing six-stage brush vehicle model.
Spin, suspension, wheel-row generation, brush contact, planar integration and
vertical integration now reuse prepared resources in **one submission**. Three
independent state-buffer parities are supported via eight prepared combinations.
Immutable dt and owner identities are checked before encoding; changed wheel
configuration requires explicit refresh (added above) or reprepare. Commands and terrain buffer contents
may update in queue order. Owned buffers use `432*N + 436` bytes for N wheels,
excluding existing owner/maps, CPU storage and driver allocations.

Two CPU contracts and three GPU fixtures pass in **0.64 s** under the unchanged
90 s cap. The new full-loop fixture exercises all eight parity combinations and
32 changing-terrain/command frames: **160 exact raw comparisons**, 128 positive
wheel-load outputs, 192 dispatches and zero observed hot buffer allocations or
logical transfers. Stale/foreign owner, budget, timestep and device-health
rejection pass; prior brush fixtures also pass. This is not native Newton actual
tractor throughput, viewer migration, shared soil fault rollback or full S03
completion. Legacy APIs/defaults remain unchanged.
[Prepared chassis evidence](benchmarks/soil/2026-10-13-prepared-chassis/SUMMARY.md).

**Resident brush terrain consumer (2026-10-13; earlier S03 increment):** Implemented
`BrushTerrainGpu` / `PreparedTireBrushTerrain` and an opt-in resident chassis
consumer. It binds existing GPU height/friction maps, copies generated GPU wheel
rows into retained scratch and runs the unchanged brush shader. The prepared
brush stage no longer allocates outputs/maps, uploads terrain or rebuilds bindings
per call. Its explicit owned buffer budget is `224*wheel_count + 36` bytes;
borrowed maps and driver overhead are excluded. Returned output handles are
retained views overwritten by subsequent encodes, not immutable snapshots.

One CPU layout/budget check and two GPU fixtures pass in **0.52 s**: flat,
heightfield and friction-map outputs match legacy bytes; 24 prepared encodes
observe zero hot application buffer allocations or logical host transfers.
Four changing-terrain chassis frames match brush, planar, vertical, spin and
suspension buffers exactly. Count mismatch rejects before spin advancement;
source/budget/lifetime/engine-health checks pass. Other chassis stages retain
their existing allocation/submission behavior. This is not whole-frame residency,
atomic machine/soil stepping, loaded-tractor traction or 4 ms/60 FPS acceptance.
[Retained brush evidence](benchmarks/soil/2026-10-13-brush-resident-terrain/SUMMARY.md).

**Resident terrain query interface (2026-10-13; S03 increment):** Implemented
`PreparedSoilTerrainQueries` over GPU query-position and regular terrain buffers.
It emits 64-byte samples containing height/normal, friction/surface velocity,
water/sinkage/rolling scale and per-query status. Sampling reuses the existing
FEM bilinear/normal functions. Immutable records encode after terrain producers
without hot uploads, allocation, submission or waits; the owned byte budget is
`112 + 64*max(query_count,1)`, excluding borrowed buffers and driver overhead.

One CPU layout/shader check and three GPU fixtures pass in **0.64 s** under the
90 s cap: 2064 independent oracle samples across all 16 optional-map combinations,
queued terrain replacement, retained source lifetime, invalid coordinates/maps,
budget/capacity rejection, empty work, foreign-engine/device-loss rejection, and
actual binned MPM export followed by a query in one encoder. The resident chain
observes zero hot buffer allocations and logical transfers. This is a heightfield
query interface, not volumetric collision, physical support/sleep, consumer
migration or a whole-frame speed result. S03 and full acceptance remain open.
[Resident query evidence](benchmarks/soil/2026-10-13-terrain-queries/SUMMARY.md).

**Newton block initialization (2026-10-13; not promoted):** Added a test-only
candidate that removes 72 overwritten scalar zero stores per linearized contact.
Only structurally unwritten Jacobian coefficients are initialized on the valid
linearization path; residual-only passes and rejected contacts retain complete
clearing. Arithmetic, capacities, solver limits and dispatches are unchanged.
Two CPU write-coverage/source reconstruction tests pass. All eight paired GPU
cases complete in **75.86 s including setup** under the unchanged 90 s cap:
40 candidate/control and 40 archived-control output comparisons are exact,
128 repeats pass, physical residual is at most **8.042703607e-7**. However, six
paired medians regress **0.07–0.53%** and only two improve **0.07–0.43%**; some
captures show large timing variation. No material speed benefit is established.
The change remains test-only; do not iterate on this store-count hypothesis
without new cost attribution. The 4 ms/60 FPS requirement remains unmet.
[Block initialization evidence](benchmarks/soil/2026-10-13-newton-block-initialization/SUMMARY.md).

**Guarded fusion actual tractor (2026-10-13; opt-in, target unmet):** The same
binary now exposes explicit benchmark fusion selection and verifies its live
mode and profile stages. Three matched 50k arms (separate, fused, fused repeat)
complete 33 frames each at unchanged 34 substeps. Machine frame history, final
particle positions and complete final soil checkpoints are byte-exact across
arms. Dispatches fall from 70,410 to 69,390 over 30 measured frames; uploads,
readbacks and hot allocation observations are unchanged. Mean particle
publication GPU time is **3.414 / 3.280 / 3.237 ms**, but whole-frame medians are
**351.742 / 357.065 / 342.474 ms** and all measured frames miss 60 Hz. Sequential
captures do not prove an uncontended whole-frame speedup; defaults stay off.
Fault recovery is covered separately by the integrated fixture below.
[Actual fusion comparison](benchmarks/soil/2026-10-13-guarded-fusion-tractor/SUMMARY.md).

**Fused/SoA actual tractor rollback (2026-10-13):** Three independent 50k tractor
worlds use the current SoA Newton profile, two unfused controls and one fused
retry arm. After three committed frames, a real zero-motion-bound rejection
restores full soil, four retained machine precision streams, all six committed
CPU body poses/velocities and time; failed GPU render publication is invalidated.
Restoring the same original bound and retrying rejoins both controls exactly.
All 33 committed frame snapshots match (66 comparisons covering full soil, nine
machine streams and CPU state), with 50 saved raw comparisons and an archived
final-soil comparison also passing. Live fusion selection is checked throughout.
The GPU test completes in **141.85 s including setup**, below the unchanged 180 s
cap. This closes first-substep motion rejection for this actual fused/SoA profile,
not later-substep or other-fault recovery, throughput, traversal or calibration.
[Integrated rollback evidence](benchmarks/soil/2026-10-13-fused-tractor-retry/SUMMARY.md).

**Guarded particle proposal fusion (2026-10-13; component evidence):**
Implemented fused G2P and plasticity over the existing private proposal buffer,
followed by a separate fault-checked commit dispatch. Public selection is exposed
on `MpmSimulation`, `RigidWorld` and `SoilTractor`. Active motion bounds are
required; projection-cache coexistence rejects in both directions. Guard byte
budgets, quiescence observation, and prepared-record invalidation are preserved.
Profiling names the fused proposal and commit separately rather than attributing
combined work to G2P alone. This is **not full G2P2G / next-grid transfer fusion**.

Twelve 129-particle grid/material cases pass byte-exact checkpoints, two fused
repeats, motion/numerical rejection and profile/reselection checks. Each substep
encodes **one fewer dispatch**, with unchanged 16,528 retained guard bytes and no
observed hot buffer allocation/readback. A separate compensated direct-coupling
fixture compares a fused retry arm against two unfused controls: soil and all nine
machine streams match at prefix/final; real motion-fault rollback restores full
soil, four retained precision streams and committed CPU body/time state, with
failed GPU body publication invalidated. Twelve checkpoint pairs and 65 coupled
raw comparisons pass. Both GPU tests finish in **3.90 / 13.65 s** within one
unchanged 90 s cap. The later actual-workload comparison above extends trajectory
coverage; this component fixture alone does not prove actual-workload acceptance.
[Guarded fusion implementation and evidence](benchmarks/soil/2026-10-13-guarded-particle-fusion/SUMMARY.md).

**Tiled Krylov basis experiment (2026-10-13; rejected for runtime):** Implemented
a test-only 32-record coefficient-major tile layout for the 16.25 MiB basis at
4096 contacts / 64 Krylov iterations, including the shortened final tile and
conditioned-core scratch alias. It retains the same bytes, bindings, arithmetic
and solver limits. CPU address/source contracts pass. The complete eight-packet
paired replay finishes in **76.98 s including setup**, within the unchanged 90 s
cap: 40 candidate/control output buffers and 40 archived-control buffers match
exactly; 128 repeat checks pass. Physical residual is at most
**8.042703607e-7**, public delta zero. Nevertheless, all eight paired medians are
**0.11–1.28% higher** than the existing field-major contact/inverse profile. There
is no demonstrated performance benefit: the basis layout stays test-only, and
runtime/default profiles are unchanged. Do not infer that every SoA conversion
helps or spend further full-tractor runs on this rejected candidate.
[Basis layout evidence](benchmarks/soil/2026-10-13-newton-basis-tiles/SUMMARY.md).

**Field-major scratch runtime integration (2026-10-13; opt-in):** Added the
`molla-solvers/soa-scratch-newton` Cargo feature on top of parallel-indices-newton.
The production conditioned factory emits the same previously verified candidate
shader; default profiles remain unchanged. Distinct failure/backend tag 31 and
tractor metadata identify this layout. Runtime diagnostic reads reconstruct the
logical contact/inverse records; raw AoS-only access rejects SoA rather than
silently misreading it. CPU source/bit-layout/tag checks and 22 benchmark tests
pass. Sparse, poisoned, faulted, bounded-iteration, reused and empty solver cases
pass together in **66.77 s including setup**, within the unchanged 90 s cap.
The initial two-engine lifecycle attempt timed out during the second cold setup;
it is retained separately, and the admitted test shares one engine without
dropping cases or increasing the cap.

Two production **50k-particle / 33-frame** tractor runs complete under their
unchanged 180 s caps. Both machine histories and final particle/full-soil
checkpoints are byte-exact against the archived unguarded baseline; all six file
comparisons pass. Each run retains 34 substeps/frame, 69,390 measured dispatches,
584,760 uploaded bytes, 26,520 readback bytes and zero observed hot allocations.
Measured wall medians are **340.070 / 346.697 ms per simulated 1/60 s**; mean
Newton GPU work is **202.895 / 201.945 ms**. All 30 measured frames in each run
miss 60 Hz. These are candidate/repeat trajectory checks, **not a fresh paired
whole-frame speed comparison**. The 4 ms target, rendered acceptance, long
traversal and physical calibration remain open.
[Runtime and actual-tractor evidence](benchmarks/soil/2026-10-13-newton-soa-runtime-admission/SUMMARY.md).

**Field-major Newton scratch layout (2026-10-13; initial candidate verification,
before runtime integration above):** Implemented a test-only coefficient-major layout for
the 456-byte contact blocks and 144-byte inverse blocks. It preserves the same
logical storage sizes, bindings, arithmetic, solver limits and publication
contracts; coefficients are adjacent across contacts rather than embedded in
per-contact records. CPU address-coverage and generated-shader checks pass.
The complete eight-packet paired replay passes in **73.32 s including setup**
under the unchanged 90 s cap: **40 candidate/control buffers are byte-exact**,
controls match the archived current baseline, and 128 exact repeat checks pass.
Maximum physical residual is **8.042703607e-7**, public delta zero. Paired median
solve times decrease by **3.78–6.60%** across the eight cases. No additional
scratch allocation, binding or dispatch is introduced. This supports further
integration/lifecycle and actual-tractor evaluation, not a whole-frame speedup
or the 4 ms/60 FPS target. Runtime defaults remain unchanged.
[Layout and matched-input evidence](benchmarks/soil/2026-10-13-newton-soa-scratch/SUMMARY.md).

**Bounded parallel kernel setup and paired admission (2026-10-13):** The production
program cache now permits at most two different cold builds concurrently, retains
one initializer per type/key and the existing 64-entry ready LRU, and invalidates
retention of in-flight builds across clear. Factory failures/panics release
reservations. Evicted/cleared programs drop outside the mutex, after admission
cleanup and waiter notification; destructor-panic regressions pass.
Sixteen compute CPU tests and three GPU cache/error-scope checks pass on the final
cache source. The earlier pre-teardown-hardening paired solver run completes in
**74.01 s** under the unchanged 90 s combined cache/replay cap, with both kernels
prepared concurrently on one device and subsequent solves timed sequentially.
All eight cases pass; **40 candidate/control buffers match exactly**, and 128
repeat checks pass. The branch-free candidate lowers paired packet medians by
**1.30–2.24%** across these cases. This is a modest component result, not a
whole-tractor speedup or sufficient evidence to enable the numerical variant.
The final cleanup-only cache change has separate CPU/GPU lifecycle verification;
paired solver timing was not rerun or attributed to changed teardown code.
[Paired evidence](benchmarks/soil/2026-10-13-parallel-program-build/SUMMARY.md),
[final cache cleanup](benchmarks/soil/2026-10-13-parallel-cache-drop/SUMMARY.md).

**Branch-free selective numerical admission (2026-10-13; no speed claim):**
Added an explicitly selected candidate-only numerical lane, separate from the
unchanged paired-performance gate. The exact same candidate SPIR-V as the prior
paired setup timeout now completes all eight frozen actual-tractor input packets
within **64.95 s including setup** (90 s cap unchanged). All **40 saved output
buffers** match the frozen current control exactly, the maximum independent
physical residual is **8.042703607e-7** (limit 1e-6), public delta is zero, and
**128 exact candidate repeat checks** pass. Six CPU/source/selection checks and
offline raw comparisons pass. Timestamp collection is rejected in this lane;
these are numerical results, not paired timing. Runtime promotion, full-tractor
trajectory/performance and the 4 ms/60 FPS target remain open.
[Numerical admission and limitations](benchmarks/soil/2026-10-13-specialized-numerical-admission/SUMMARY.md).

**Branch-free selective float controls (2026-10-13; test-only, paired GPU admission
incomplete):** Implemented a separately compiled Newton candidate whose retained
rounding helper has no memory accesses or conditional branch, with per-instruction
float controls limited to protected arithmetic. This differs from the earlier
shared-kernel selective experiment: the memory-rounding arm is absent, not merely
unselected. Five CPU/SPIR-V checks pass. The paired replay reaches the unchanged
90 s cap during candidate pipeline construction after a **64.081 s** control
setup. No completed solve, numerical parity or speedup is established for this
candidate; no runtime default changes, increased deadline or promotion follows.
[Implementation and preserved timeout](benchmarks/soil/2026-10-13-float-controls-specialized/SUMMARY.md).

**Actual 50k tractor motion-fault recovery (2026-10-13):** A new integrated
regression uses the actual native-prediction/Newton tractor and tine, shared-chunk
transfer, 1024 compact tiles, 4096 contact slots and 34 substeps per 1/60 s frame.
All three independent arms commit 33 frames (0.55 simulated seconds each). After
three identical frames, the retry arm encounters a real zero-motion-bound fault
on the first gather of its fourth frame. Complete soil state, four retained
machine precision streams, all six CPU body poses/velocities and committed time
restore exactly; failed GPU body publication is invalidated. The same-bound retry
and every later frame match both controls exactly: **66 full-state frame
comparisons**, independently checked histories plus **50 raw saved comparisons**.
The final soil checkpoint also matches the archived unguarded 33-frame workload.
GPU verification passes in **136.70 s including setup**, within the unchanged
180 s actual-workload cap. This closes the first-substep motion-rejection/retry
case for this actual configuration, not later-substep actual-tractor faults,
other fault classes, native atomic-transfer repeatability, bounded traversal,
physical calibration or 4 ms/60 FPS acceptance. Runtime defaults are unchanged.
[Integrated recovery evidence](benchmarks/soil/2026-10-13-tractor-motion-retry/SUMMARY.md).

**Public coupled reaction selection (2026-10-13):** Added
`RigidWorld::set_mpm_soil_compensated_reactions`, a tractor wrapper and readback-free
`mpm_soil_reaction_info`. The existing compensated reduction is now selectable
after attachment through a separately budgeted cold API, without private fixture
access. Portable direct coupling only; native/projected enablement rejects, and
projected coupling requires disabling the direct reduction first. Disable retains
storage; smaller retained budgets reject; idempotent selection does not invalidate
work unnecessarily. The public path passes the exact matched fault/retry fixture,
**65 raw comparisons**, resource/transfer-free repeated selection and configuration
preservation, plus **144 CPU admission combinations**. GPU test **13.88 s**, cap90s
unchanged. No default, general determinism or speedup claim; native/50k acceptance
and the 4 ms target remain open.
[Implementation and evidence](benchmarks/soil/2026-10-13-coupled-reaction-api/SUMMARY.md).

**Matched-input direct motion retry (2026-10-13; bounded compensated-reaction
configuration verified):** Added raw snapshots of nine machine streams and a
control self-repeat before fault injection. Two ordinarily reduced, identically
guarded controls already differ by **12 machine words before any fault**, while
their soil checkpoints match. This demonstrates a repeatability problem in the
comparison baseline, not a rollback diagnosis. With the existing opt-in
compensated reaction reduction (32 bytes for this one-body fixture), the complete
soil checkpoint and all nine machine streams match both controls at prefix and
after a real motion-bound fault/retry. Four retained precision streams and the
committed CPU pose/velocity restore exactly; failed GPU body publication is
invalidated. **45 independent raw comparisons pass**, GPU test **15.69 s**, within
the unchanged 90 s cap. Scope: portable dense direct coupling, one particle,
two substeps per frame. Ordinary atomic-path repeatability remains open. The
later actual-tractor recovery entry above covers native/projected shared-chunk
first-substep rejection, not all native/projected retry configurations. No runtime
default or performance target changed.
[Matched retry evidence](benchmarks/soil/2026-10-13-motion-compensated-retry/SUMMARY.md).

**Selective float-controls follow-up (2026-10-13; numerical gates pass, no useful
speedup demonstrated):** Shared compilation fits the original cap. Its initial
kernel-wide strict control fails exact archived parity by **1197 words**, so that
variant is rejected. Restricting float controls to compensated helpers and the
computed-root operation preserves the archived control and all candidate public
buffers across **eight actual-work packets**: **40/40 buffers byte-exact**, maximum
physical residual **8.042703606542908e-7**, zero public delta and **128 exact repeat
checks**. The complete five-mode arithmetic fixture also passes **40,965 cases**.
GPU arithmetic plus full replay finish within the unchanged **90 s** cap (0.29 s
and 71.75 s test wall times; first pipeline construction 69.803 s). Eight paired
timing changes range from **0.096% faster to 0.173% slower**, with mixed signs;
there is no meaningful demonstrated performance gain. Keep this candidate
**test-only**, without runtime promotion or another full tractor benchmark.
Actual 4 ms/60 FPS, support/wake, bounded traversal and integrated visual/physical
acceptance remain incomplete.
[Frozen selective comparison](benchmarks/soil/2026-10-13-float-controls-selective-admission/SUMMARY.md).

**Original float-controls Newton experiment (2026-10-13; arithmetic passed, full solver
unverified):** The live RTX 4080 supports Vulkan float-controls2. Added a test-only
logical device that explicitly enables the extension feature, plus a validated
SPIR-V lowering replacing rounding-memory operations with strict fast-math flags.
Five arithmetic modes, **40,965 cases total**, match the protected atomic reference
and repeat bit-for-bit; the GPU fixture completes in **0.39 s** including setup.
The subsequent frozen eight-packet full-Newton comparison **times out at its
unchanged 90-second cap during pipeline construction**: control setup completes
in **64.277 s**, candidate setup does not complete, and no solve is submitted by
the comparison. No physical/public full-solver gate or timing comparison passes;
no production promotion or speedup is claimed. The separate-pipeline design needs
a changed preparation strategy before another bounded trial, not a longer cap or
unchanged rerun. Actual 4 ms/60 FPS acceptance remains far unmet.
[Frozen arithmetic and full-solver attempt](benchmarks/soil/2026-10-13-float-controls-newton/SUMMARY.md).

**Motion guard on the actual tractor (2026-10-13; exact bounded integration):**
Added live effective-bound/storage queries, explicit cold coupling preparation,
and benchmark/Make selection with checked requested/live manifest fields. Default
remains off. With the unchanged compact/shared-chunk 50k workload, control, guarded
and guarded repeat each complete **33 frames / 0.55 simulated seconds**, with all
34 substeps per frame. Every recorded machine-state line, final particle position
and **8,000,136-byte lossless particle checkpoint** matches byte-for-byte.
Final loads, machine energy, axle coordinates and commanded work also match.

The selected bound is deliberately permissive: **1 m per axis per soil substep**,
with **6,400,016 bytes** of separate retained guard storage. It is not residency
envelope acceptance. Measured hot buffer allocations remain zero; guard work adds
exactly one dispatch and 16 upload bytes per substep, with no added readback.
Wall medians are **348.900 / 350.125 / 356.039 ms**; all 30 measured frames miss
60 Hz in all three arms. Mean MPM time remains **245.521 / 246.196 / 247.919 ms**,
with Newton about 212 ms. No speedup or 4 ms acceptance is claimed.
The small-fixture cross-world retry mismatch remains a separate unresolved gate.
[Source-frozen actual workload evidence](benchmarks/soil/2026-10-13-motion-tractor/SUMMARY.md).

**Later-substep motion rollback (2026-10-13; native compact Newton case passes):**
Added bounded, error-only grid diagnostics that distinguish motion-bound,
numerical and tile-overflow faults, preserving the existing completion deadline.
Healthy grid flags do not suppress unrelated machine/contact diagnostics. The
16-byte metadata readback is outside the successful stepping path; its G2P counter
includes failed gather attempts and is not a committed-substep count.

A two-particle fixture includes a freely accelerating particle and a contact
particle. Measured reference vertical displacements **8.940697e-8 / 2.086163e-7 m**
bracket the fixed **1.5e-7 m/substep** bound. In the actual three-substep coupled
frame, the second gather fails; the initial full particle checkpoint and body
pose/velocity are restored exactly, frame counts remain zero and no failed loads
are published. A permissive three-substep retry commits. Final-source GPU/setup
passes in **82.42 s** under the unchanged 90-second cap; flag-label and four
completion-deadline CPU tests also pass. This verifies a native compact Newton
component, not exact cross-world retry or the 4 ms target; the subsequent actual
tractor admission above has its own bounded scope.
[Evidence](benchmarks/soil/2026-10-13-motion-fault-admission/SUMMARY.md).

**Projected motion rollback (2026-10-13; bounded verification passes):**
The independent compact-grid Newton fixture passes for portable and native MPM
with native machine prediction. A real zero-bound fault rejects the coupled frame;
particle checkpoint, body pose/velocity, committed impulse words, frame ordinal
and simulated-duration metadata remain exact. Re-enabling a permissive guard then
stepping, and disabling it then stepping, both commit normally. This checks recovery,
not equivalence to a separate retry trajectory. The two-case aggregate completes in
**85.88 s**, including GPU setup under the unchanged 90-second cap.

The initial 32 MiB fixture allocation was correctly rejected; the corrected case
uses the existing coupled-window Newton fixture's 128 MiB buffer budget without
reducing particle/contact capacity, substeps or iterations. This is test configuration,
not a production-budget/default change. The earlier cross-world retry gate remains
failed: offline decoding locates six differing particle words, maximum absolute
delta **1.629815e-13**, with unequal preceding machine states. No tolerance replaces
that exact check and its cause remains unresolved. The later check above covers a
native compact later-substep failure; actual tractor/50k guard integration remains
open, and no performance target was achieved.
[Current evidence](benchmarks/soil/2026-10-13-motion-projected-capacity/SUMMARY.md).

**Coupled motion bounds (2026-10-13; implemented, aggregate acceptance open):**
Added world/tractor per-axis displacement-bound configuration with a separate
`16 + 128 * max(max_particles, 1)` byte budget. Disabling retains this storage;
invalid bounds and insufficient budgets reject before allocation. Projected stepping
now uploads bounds and uses guarded indirect gather/plasticity/commit dispatches.
Guarded dense grids now propagate faults to machine status, not only active grids.
Changing bounds preserves rigid bindings and the last committed publication.

The native-prediction machine / portable dense-soil fixture rejects a real zero-bound
violation and restores particle checkpoint, body pose and velocity **exactly**.
However, the latest aggregate fails its subsequent byte-exact particle comparison
against a separate control; their prior machine states already differed within the
unchanged 2e-5 trajectory gate. The cause of the retry difference is not established.
Projected Newton and native-MPM cases were not reached by that failed aggregate;
the independent two-case rollback check above now covers them. Earlier unsupported fixture
configurations and a 90-second setup timeout are retained, not hidden or counted as
passes. CPU bound/source checks and compilation pass. This opt-in feature is not
full coupled-motion acceptance, a speedup, or 50k tractor verification.
[Current evidence and remaining checks](benchmarks/soil/2026-10-13-coupled-motion-newton/SUMMARY.md).

**Exact inverse-cache experiment (2026-10-13; rejected):**
A test-only lane-local cache checks contact identity and all 108 coefficient words
before reusing a successful inverse within the same solve. Immutable row/response
inputs remain fixed for that dispatch; misses preserve the existing factorization
and fallback. This is not lagged preconditioning. Eight frozen current-workload
packets pass all five public-buffer comparisons against both arms and archived
method 30, with 128 exact self-repeats and maximum physical residual 8.042704e-7.
The instrumented candidate reports **zero cache hits and 24,487 builds**. All eight
paired medians regress by **0.35–0.91%**; **do not promote or rerun this unchanged
cache**. GPU/setup terminates in 66.23 s under the existing 90-second cap.
[Experiment](benchmarks/soil/2026-10-13-newton-exact-inverse-cache/SUMMARY.md).

A separate opt-in CUDA prototype has been proposed to the user, not authorized or
implemented. NVIDIA documents explicit scalar rounding intrinsics; whether they
can replace current protected rounding machinery with a material speedup is an
unverified hypothesis. Any approved prototype must retain the wgpu fallback,
numerical/physical gates, workload and integration/transfer accounting; no CUDA
dependency or backend is added without the separate decision required by S06.
[Primary arithmetic API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).

**Shared-chunk actual tractor (2026-10-13; exact histories, budget still fails):**
Added explicit `SoilTractorTransfer::SharedChunk { max_gather_bytes }` and benchmark
`shared-chunk:3200000`, preserving both atomic default and direct fixed-order control.
Live transfer queries report `shared_chunks`; requested/live modes, scratch bytes,
ordering and backend identity must agree before stepping. The same-source control,
candidate and candidate repeat each complete **33 frames / 0.55 simulated seconds**
with 50k particles and unchanged 34 substeps per frame. All recorded machine states,
final positions and **8,000,136-byte full particle checkpoints match byte-for-byte**.
Final forces, torques, machine energy/work and resource counters also match exactly.

Mean P2G GPU time is **15.054541 / 10.079539 / 9.985638 ms per frame** (control /
shared / repeat), about 33% lower for this stage. Overall wall medians remain
**357.539 / 346.712 / 343.136 ms**; **all 30 measured frames miss 60 Hz in every arm**.
Mean Newton contact work is still **212.273 / 214.359 / 211.614 ms**. These sequential
captures and before/after GPU snapshots do not establish an uncontended whole-frame
speedup. No tolerance, workload, substep count or default changed. The next performance
work must address dominant contact/machine costs, not extrapolate transfer savings
to 4 ms acceptance. Support/wake, traversal, real-mesh rendering and full integration
remain open. [Actual workload evidence](benchmarks/soil/2026-10-13-shared-gather-tractor/SUMMARY.md).

**Shared-chunk production API (2026-10-13; opt-in, frame gate open):**
`prepare_deterministic_p2g_shared_chunks(max_bytes, compensated_stiffness)` now
prepares the retained shared-data variant without changing default stepping.
Existing gather enable/release semantics, scratch budgets and prepared-record
invalidation apply. It requires 16 KiB workgroup storage, retains the same 64 bytes
per maximum particle and rejects native compensated stiffness before allocation.
The public API passes **147 exact paired-history cases** across native dense/active/
compact and non-native active/compact layouts, both non-native stiffness policies,
partial/empty chunks, layout switches and compact growth. Hot GPU buffer/binding/
pipeline allocations and uploads/readbacks are zero in the prepared submission path.
The Make aggregate passes in 1.82 s (small exactness) plus 5.24 s (public lifecycle).
The production-source 50k pair retains exact outputs with GPU medians
**0.597504 / 0.510464 ms** (14.57% lower). The preceding empty-grid test assertion
failure is retained; zero-particle stepping correctly leaves status uninitialized.
At this earlier API snapshot, tractor selection/trajectories were pending; the
subsequent actual-workload result above verifies that bounded integration but not
4 ms/60 FPS or full-plan acceptance.
[Public/runtime evidence](benchmarks/soil/2026-10-13-gather-shared-runtime-admission/SUMMARY.md).

**Shared-chunk gather experiment (2026-10-13; component improvement only):**
The original gather arithmetic and contributor order are preserved while each
64-particle workgroup chunk shares immutable particle data. Four layout modes
pass eight exact evolution/grid comparisons each; all sixteen native 50k fixed-state
transfer captures match exactly and leave particle history unchanged. Six measured
alternating pairs give GPU medians **0.598016 ms control / 0.500736 ms candidate**
(16.27% lower); GPU/setup completes in 2.51 s. The archived 96-byte uniform fixture
is explicitly decoded into the current 112-byte layout with zero window offset,
preserving all historical fields; the preceding direct-POD failure is retained.
This advances zero simulated time, not the tractor/frame budget. The candidate
remains test-only in this evidence snapshot. [Evidence](benchmarks/soil/2026-10-13-gather-shared-chunk-admission/SUMMARY.md).

The separate identical-input contact diagnostic reproduces the archived force
mismatch exactly: small grid differences produce f64 contact-sum differences
within 2e-5, but final adjacent f32 values differ by 3.0517578125e-5. Both individual
reductions meet their own bound; the cross-method gate still fails. Do not bias
outputs or relax the bound. [Diagnostic](benchmarks/soil/2026-10-13-transfer-contact-terms/SUMMARY.md).

**Public reaction lifecycle (2026-10-13; bounded public API verified):**
The non-test-library integration fixture passes **36 cases** across dense/active
switching, compact growth from 128 to 256 to 512 tiles, explicit/implicit stepping,
legacy/prescribed bodies, zero/count changes, checkpoint restore and cancelled
batches. Paired particle checkpoints match exactly; force differences stay at most
2.384186e-7. Hot buffers, bindings, pipelines, uploads and readbacks remain zero.
Preparation budgets, stale records, release and native rejection are checked.

Added and executed `make test-soil-compensated-wrench RUN_WITH=nixVulkan`, with
compilation outside the separate 90-second GPU caps. CPU-only source verification
is `make test-soil-compensated-wrench-fast RELEASE=1`. The reducer remains opt-in;
the 50k cross-transfer force failure, native support, actual frame budget and
physical streaming remain open. No default or tolerance changes were made.
[Public API evidence and original fixture rejection](benchmarks/soil/2026-10-13-wrench-public-admission/SUMMARY.md).

**Retained reaction compensation (2026-10-13; repeatability repaired, cross-path gate open):**
Added an opt-in non-native reaction reducer with **32 × max(max_rigid, 1) bytes**
of retained residual storage and one final resolve dispatch. Reset, imprint and
legacy shell accumulation share the residual; prescribed solids resolve after
imprint without adding shell forces. No diagnostic force-term buffers or readbacks
are retained in production. Preparation/selection/release invalidate stale records.

The unchanged 12-step, 50k evolution now has **zero differences across independent
candidate particle and wrench outputs**. All final particle checkpoints also match
the pre-repair run exactly. However, comparison against corrected gather still
fails: **3.051758e-5 > 2e-5** in reaction force. The failure remains explicit; no
reference or tolerance change, runtime promotion or actual-tractor rerun follows.
Fifteen dense/active/compact body-mode transitions pass, as do a captured-force
f64 oracle and 134 checkpoint CPU regressions. Frame performance and all broader
streaming/physical acceptance remain open.
[Implementation, preserved failures and next isolation](benchmarks/soil/2026-10-13-retained-wrench-admission/SUMMARY.md).

**All-field transfer at 50k (2026-10-13; contact-force gate still fails):**
The frozen four-arm transfer comparison passes in **1.12 s**: eight all-field
candidate world grids repeat exactly, stiffness matches the archived rounded
reference, and all particle checkpoints remain unchanged. Median GPU cost is
**0.815104 ms all-field versus 2.127360 ms corrected gather**, but ordinary scatter
is faster at 0.524288 ms. This is a **zero-simulated-time component**, not frame
acceptance. [Paired accuracy/cost evidence](benchmarks/soil/2026-10-13-compensated-transfer-50k/SUMMARY.md).

A subsequent 12-step, 50k-particle prescribed-sphere fixture **fails the unchanged
2e-5 wrench bound** at **6.103516e-5**, both against corrected gather and between
independent candidate runs. Candidate particle history repeats exactly; maximum
particle-field difference from gather is 6.529726e-6. This separates remaining
reaction-force nondeterminism from particle-trajectory divergence in this fixture,
but does not yet isolate imprint versus shell summation. No runtime promotion or
unchanged retry follows. [Preserved contact failure](benchmarks/soil/2026-10-13-compensated-transfer-evolution/SUMMARY.md).

**All-field compensated transfer (2026-10-13; opt-in, component-verified):**
Added non-native single/two-pass P2G compensation for mass, signed momentum and
stiffness. Preparation retains **20 bytes per physical grid cell**, with explicit
budget checks; stepping uses clear/scatter/resolve without hot allocation or host
transfers. Existing stiffness-only mode and production defaults are unchanged.
The candidate passes the **12-mode command-batch comparison at the unchanged
2e-5 bound**, with zero reaction-force differences across six reference pairs.
The corrected stage diagnostic uses the selected production transfer, rather than
bypassing it: six fixed-particle grid replays now match exactly in this fixture.

Seven supported grid/scatter configurations cover particle-count transitions,
stale preparation, storage growth and rejection of single-pass deferred lookup;
28 all-field state comparisons stay below 2e-5. Legacy lifecycle, 134 checkpoint
CPU regressions, formatting and facade checks pass. These results do **not** repair
the default uncorrected lane, establish universal deterministic summation, or prove
50k throughput/trajectory acceptance. The full RTX 4080 4 ms/60 FPS target, support,
sleep/eviction and traversal remain open.
[API, evidence and preserved fixture failure](benchmarks/soil/2026-10-13-compensated-transfer-admission/SUMMARY.md).

**Newton subgroup reduction (2026-10-13; correct but slower, not promoted):**
Implemented a test-only exact-order subgroup reduction over the current method-30
Newton program. It folds the same 256-term binary tree using two workgroup barriers
instead of four, without assuming a mapping between local and subgroup lanes.
Non-32 subgroup widths explicitly use the reference path. The protected compiler's
subgroup capability/SPIR-V1.3 opt-in is test-marker gated; production selection and
solver caps are unchanged.

The symbolic tree, protected full-program compile and 32 adversarial/permuted-lane
GPU fixtures pass; the final primitive run takes **0.25 s** including setup. The
complete eight-packet Newton replay passes in **83.03 s** inside the unchanged
90-second cap, including **80.843 s first pipeline setup**. All five state/output
buffers match both arms and archived method30, all 128 alternating repeats match,
and the independent physical residual is at most **8.042704e-7 < 1e-6** with zero
public delta. However paired median costs increase **1.0645–2.2101%** on all eight
packets. **Do not promote or rerun this unchanged candidate.** Fewer barriers did
not improve this solver. The first post-run archive audit used the endpoint-only
directory; a separate read-only audit verifies the sealed full-solver archive and
all comparisons without rerunning GPU work. Full machine 4 ms/60 FPS, support,
sleep/eviction and traversal gates remain open.
[Sources, results and rejected optimization](benchmarks/soil/2026-10-13-newton-subgroup-replay/SUMMARY.md).

**Cache moving-soil cost and bypass (2026-10-13; remains opt-in):**
The frozen 50k detached moving-soil probe found **zero cache hits** over 36
instrumented substeps. Unconditional caching increased projection-plus-commit mean
from **0.037888 to 0.067243 ms**. The implementation now bypasses cache lookup and
writes when the existing exact-rest streak is zero, without removing any particle
from P2G/G2P. A later paired run measures **0.037581 ms uncached versus 0.039185 ms
with bypass**: overhead remains, so this is not evidence to enable caching by default.
Wall means vary with run order; no robust whole-step speedup is claimed.

The final probe passes in **2.32 s** including setup, preserving **108 exact 50k
checkpoint boundaries** across three arms (control, production cache, instrumented
cache) and three alternating-order repeats. Every instrumented step still has
zero hits. The hit reducer has a last-slot positive control and clears markers
after sampling, preventing stale-hit attribution following a bypass. Two CPU
shader/layout tests and 134 checkpoint regressions pass. The production bypass
also passes five lifecycle/parity GPU tests in **5.61 s**, including 117 exact
small-fixture boundaries. All source freezes and earlier slower results remain.

This seed comes from the deterministic-P2G component fixture, **not a loaded
tractor checkpoint**. Each replay advances twelve 0.0001-second steps; none is a
1/60-second machine frame. No extrapolated FPS or 4 ms claim is permitted. Further
active-soil work should target the dominant coupled solver and physical streaming,
not assume constitutive-cache hits. [Measurements and evidence](benchmarks/soil/2026-10-13-projection-cache-hit-sampling/SUMMARY.md).

**Exact-input constitutive cache (2026-10-13; initial implementation):**
Both owners now expose `prepare_swept_projection_cache(max_bytes)`, byte accounting
and disable. This requires the swept guard and quiescence observer. The cached
projection compares every physical F/Jp input bit plus dt in the current material,
identity and configuration epoch; a hit reuses the previously computed F/Jp output.
Misses execute the unchanged constitutive function. All particles still scatter
mass and stress, gather grid motion and publish through the global fault guard:
no dormant support is omitted, and no particle is evicted. Cache reset shares the
observer's restore/configuration/encoder-access and cancelled-reset lifecycle.
Additional retained GPU storage is **112 bytes per capacity slot** (14,448 at 129;
5,600,000 at 50k), with no additional dispatch or hot readback/allocation.

Nine paired dense/active/compact and sand/snow/viscous fixtures match byte-for-byte
at **117 physical boundaries** with strain, material and dt changes. An injected
fault on the miss path proves a warmed identical input bypasses projection and
changed F/Jp/dt, material edit, restore, external encoder access and cancelled reset
do not. Five GPU tests pass in **5.11 s** including setup; two shader/layout checks
and **134 checkpoint regressions** pass. Two earlier fixture configuration failures
are preserved, not waived. This remains opt-in: actual cache-hit distribution,
timings, loaded coupled-machine behavior, sleep/wake and **4 ms/60 FPS** acceptance
remain open. [Frozen evidence and limitations](benchmarks/soil/2026-10-13-projection-cache-lifecycle/SUMMARY.md).

**State-bound exact-rest snapshots (2026-10-13; implemented, not sleep):**
Observation payloads now have read-only `identity()` and `particles()` accessors
and private solver-configuration/encoding provenance. Both live owners expose
`validate_swept_quiescence`; it rejects foreign owners, pending/faulted owners,
stale configuration/identity, and any intervening nonempty encoded step, including
cancelled work. Merely reading/checkpointing or requesting zero substeps does not
invalidate a snapshot. Mutable command-encoder access invalidates the observations
until a newly observed step. Counter exhaustion fails closed without wrapping;
the check adds no GPU work or per-step allocation. The 129-particle dense, active
and compact checks, plus memory/disk owner checks, pass in **2.16 s** including GPU
setup; **134 checkpoint CPU regressions** and shader validation pass. These are
snapshot provenance checks, not physical support, sleep/wake, safe eviction or
performance acceptance. All active particles remain pinned. The earlier two
command-batch force-parity failures remain unresolved; they were not rerun here.
[Frozen source, executable, fixtures and results](benchmarks/soil/2026-10-13-quiescence-provenance/SUMMARY.md).

**Per-substep exact-rest observations (2026-10-13; implemented, not sleep):**
Both live owners now expose opt-in `prepare_swept_quiescence`, bounded blocking
`read_swept_quiescence`, byte accounting and disable APIs on the swept guard.
The publication shader records consecutive exact-rest substeps in physical-slot
order: old/new velocity and C must be zero, and position/F/plastic history must
match bit-for-bit. Changes in dt restart the count; configuration/restore changes
reset it, and cancelled prepared batches retain their pending reset. Reads include
global-ID provenance and reject pending, disabled, stale, faulted or over-bound state.
This uses **16 + 16 * max(max_particles,1) additional retained GPU bytes** (2080 at
129 capacity), not hidden inside guard/capture budgets. It adds no physics dispatch
or hot readback; explicit observation copies 16 + 8 * particle_count bytes.
`make test-soil-quiescence RELEASE=1 RUN_WITH=nixVulkan` passes **three GPU tests in
2.13 s including setup**, under the original 90-second cap, and is included in S07.
One CPU shader/budget check and 134 checkpoint regressions pass. Five controlled
publication probes expose intermediate position/velocity/C/F/plastic changes even
when checkpoint endpoint bytes become identical again. These probes are not five
physical loading experiments. Both live owners retain IDs; moving soil reports no
exact-rest streak. Six existing motion-guard tests also pass.
**Broader regression remains failed:** command-batch force parity differs by
3.8146973e-5 and its unchanged-reference repeatability diagnostic by 3.0517578e-5,
both above the existing 2e-5 bound. This is the previously documented failing lane,
not repaired or excused by observer success. No tolerance or default was changed.
Exact-rest observations are **not a force/support certificate or permission to
evict**. All active particles remain pinned. Next: bind candidate provenance to
support-preserving residency, validate dormant boundary contributions and wake
propagation against the full active reference before enabling release.
Sleep/wake, bounded traversal, full S07 and **4 ms / 60 FPS** remain incomplete.
[Implementation and bounded root-target evidence](benchmarks/soil/2026-10-13-soil-quiescence-gate/SUMMARY.md).

**Guarded plasticity publication and regular Make gates (2026-10-13; verified):**
The guarded transaction now includes plasticity: gather writes proposals, the
existing return-mapping algorithm updates only those proposals, then one dispatch
publishes position/velocity/C/F/plastic history only if global status is healthy.
Plastic history shares the first 16-byte proposal slot with position; retained
memory remains **128 bytes per particle + 16 uniform bytes**. There is still only
one additional dispatch compared with unguarded gather/plasticity. Unguarded and
coupled solver paths are unchanged; this is not machine/grid/clock rollback.
The new `make test-soil-motion-guard RELEASE=1 RUN_WITH=nixVulkan` passes **six
GPU tests in 4.96 s including setup**. Nine sand/snow/viscous × dense/active/compact
cases match successful unguarded outputs exactly; 18 staged F/plastic-history fault
injections preserve all physical streams and pass prepared restore/retry. Earlier
motion rejection, late-substep, owner and arithmetic checks also pass.
`make test-soil-window-integer RELEASE=1 RUN_WITH=nixVulkan` passes in **23.90 s**,
retaining all 21 exact own-repeat pairs and 14 physical comparisons below 1e-6.
Both targets precompile before their unchanged 90-second GPU cap and now appear in
S07 alongside all earlier gates. One CPU shader/layout test and 134 checkpoint
regressions pass. **This does not declare full S07, sleep/wake, bounded traversal,
actual 50k physical acceptance or 4 ms / 60 FPS complete.**
[Frozen implementation and Make-target evidence](benchmarks/soil/2026-10-13-motion-projection/SUMMARY.md).

**Transactional swept-motion publication (2026-10-13; earlier component):**
Guarded G2P now writes retained proposals, then publishes through a separate GPU
dispatch only if the global grid status is healthy. A displacement violation in
one workgroup cannot leave other particles partially advanced. Existing conservative
displacement bounds and sticky fault/capture rejection remain unchanged. Both world
owners expose budgeted `prepare_swept_motion_guard` and `swept_motion_guard_bytes`;
storage is **16 + 128 * max(max_particles, 1) bytes**, separate from readback staging.
The guarded path adds one dispatch per substep; unguarded stepping is unchanged.
One CPU shader/layout/budget check and **134 checkpoint regressions** pass.
**Five GPU tests pass in 2.76 s including setup**: six dense/active/compact,
explicit/implicit 129-particle cases preserve every particle word on first-step
rejection and match unguarded successful results exactly. Three gravity-driven
prepared-batch cases reject the second substep while retaining the exact first
substep. Owner tests retain both physical streams and durable archive generation.
There are zero hot GPU allocations/readbacks; 129-particle guard storage is 16,528
bytes. The arithmetic probe admits 1,867 of 4,103 cases without false acceptance
against f64. The first suite's malformed test shader is preserved separately.
This increment provided **atomic guarded G2P publication**; plasticity was added
subsequently above. Earlier successful substeps, grid scratch and machine state
are still not rolled back.
It is a safety prerequisite, **not implemented sleep/wake, bounded traversal or
4 ms/60 FPS acceptance**. Failed-owner recovery still requires known-good state.
[Implementation and evidence](benchmarks/soil/2026-10-13-motion-publication/SUMMARY.md),
[initial fixture failure](benchmarks/soil/2026-10-13-motion-transaction/SUMMARY.md).

**Canonical integer windows and disk imports (2026-10-13; component verified):**
Canonical archives retain global particle data, not an arbitrary client's grid.
New `select_window` / `begin_read_window` APIs carry the explicit client anchor and
signed cell offsets through selection, background reads and writeback provenance.
`MpmSoilDiskWorld::begin_grid_window_cells` publishes a supported move through the
existing frozen capture boundary with no particle upload. Canonical construction
accepts an empty simulation's nonzero window; raw storage stays origin-only.
Mismatched transfer windows now reject, including the unchanged-ID shortcut.
**134 CPU tests pass.** The frozen RTX 4080 lifecycle check passes in **1.82 s
including setup**, covering active/compact layouts, explicit/two-sweep implicit
stepping, 129 particles at h=0.1, grid 64 and anchor [-3,-0.5,-3]. Four cases preserve
65 resident particles during movement, import 64 through device gather, advance,
write back and reopen with exact frame bytes and IDs. Publication allocates zero
GPU buffers, uploads zero bytes and reads 10,416 validation bytes per case.
Range/support rejection preserves state and HEAD. Canonical global metadata remains
window-independent; reopening requires the caller's saved window, not a global
archive inference. The initial one-tile fixture failed its import precondition;
that failure is retained separately, not retried unchanged.
This does **not** establish physical sleep/wake, bounded traversal, machine/disk
coordination, rendered persistence or **4 ms / 60 FPS** acceptance.
[Canonical/disk evidence](benchmarks/soil/2026-10-13-canonical-integer-import/SUMMARY.md),
[initial fixture failure](benchmarks/soil/2026-10-13-canonical-integer-window/SUMMARY.md).

**Public integer-window movement and restore (2026-10-13; component verified):**
Added `MpmSoilCheckpoint::with_grid_window_cells`, bounded detached
`MpmSimulation::move_soil_grid_window_cells_with`, and committed-frame
`RigidWorld::move_mpm_soil_grid_window_cells` / `SoilTractor::move_soil_grid_window_cells`.
Offsets are **absolute cells from the unchanged lattice anchor**, not rounded origins
or particle translations. Nonzero checkpoint restore now validates finite inverse
spacing, integer ranges and complete stencil support; moves retain old/new moved-axis
halos. Raw world construction rejects nonzero offsets; canonical support followed above.
The external public-library fixture passes movement, save/reload, continued stepping,
return to version 1 and stale-record checks across seven native/non-native layouts:
**21 exact own-repeat pairs**, **14 shifted/control comparisons**, maximum error
**7.480899e-8**, below the original 1e-6 gate. Duration **24.02 s including setup**.
The declared projected/Newton lane passes committed publication, unchanged machine
state, continued coupling and exact failed-frame rollback/retry on the **0.1-metre,
64-cubed grid with 125 compact tiles**, in **79.90 s including setup**. Its original
128 MiB budget, 4096-node capacity, solver limits and 90-second cap are unchanged.
Publication uploads zero bytes and reads 336 validation bytes for two particles.
The separate 2 MB direct lane remains rejected (dense backup budget, then compact
without projection); neither failure was hidden or accepted by increasing its cap.
Two admission checks and **133 checkpoint CPU regressions** pass. **Still open:**
coordinated machine/disk windows, actual 50k paired trajectories, rendered
persistence, sleep/wake, bounded traversal and the **4 ms / 60 FPS** requirement.
[Public detached API evidence](benchmarks/soil/2026-10-13-integer-window-api/SUMMARY.md),
[direct compact rejection](benchmarks/soil/2026-10-13-integer-window-compact/SUMMARY.md),
[projected lifecycle evidence](benchmarks/soil/2026-10-13-integer-window-projected/SUMMARY.md).

**Anchored contact coordinates (2026-10-13; internal integration verified):**
Finite-mass contact arms and unilateral candidate/SDF query positions now reconstruct
global integer nodes against the same lattice anchor as core MPM. Prepared contact
owners retain the offset and reject mismatches before queue uploads, independently
of their epoch checks. Contact parameter blocks grow by 16 bytes; required-byte
accounting follows actual storage without increasing configured memory caps.
The 0.1-metre, 64-cubed fixture passes dense/active/compact sticking and normal
unilateral coupling with offsets [0,0,0], [-1,0,1], [1,0,-1]: 18 owners, two steps
each, all shifted comparisons below 1e-6. Maximum particle/rate errors are
**4.656613e-10 / 2.910383e-11**. Nine candidate/query/publication cases agree exactly.
Two CPU checks and both GPU tests pass; GPU duration is **14.28 s including setup**
under the unchanged 90-second cap. The start snapshot had 85% GPU utilization;
this is correctness evidence, not uncontended performance evidence.
This contact-only increment did not open public admission; the later API/lifecycle
increment above does. Canonical client windows were integrated subsequently, above.
These contact-only tests do not execute machine drift, paired-rate GPU/Newton coupling, an
actual 50k tractor trajectory, or rendered persistence.
[Contact migration evidence](benchmarks/soil/2026-10-13-lattice-contact/SUMMARY.md).

**Anchored GPU coordinates (2026-10-13; internal component verified):**
Core MPM scatter/gather, G2P, node positions, clamps, active/compact marking and
particle ordering now use a stable lattice anchor plus signed integer window
offsets. Normal uploads and immutable prepared records carry the same offsets;
capture metadata retains them. The main uniform grows from 96 to 112 bytes,
active parameters from 48 to 64, and ordering parameters from 32 to 48.
At the actual **0.1-metre spacing and [-3, -0.5, -3] anchor**, positive/negative
cell shifts pass **32 physical comparisons and 48 exact self-repeat pairs** across
active/compact explicit/implicit stepping, alternating ordinary/prepared encoding.
Maximum physical-stream error is **1.192093e-7**, below the unchanged 1e-6 bound;
positions are exact. Twelve single-particle dense/active/compact scatter comparisons
also have zero stream error, including sphere-contact cases. The GPU suite takes
**2.88 s including setup**, below the unchanged 90-second cap. Two focused CPU
checks, 131 checkpoint regressions and two source-contract checks pass.
Public admission, committed-frame checks and canonical windows were added later,
above; full traversal integration remains incomplete. This kernel increment is neither actual
tractor traversal nor a 4 ms/60 FPS result; native shaders are CPU-validated here,
not GPU-certified. The first gather fixture requested unsupported non-native dense
gather and failed admission; its evidence is retained, not hidden by a retry.
[Initial fixture failure](benchmarks/soil/2026-10-12-lattice-kernels/SUMMARY.md),
[supported-layout GPU evidence](benchmarks/soil/2026-10-13-lattice-kernels-active/SUMMARY.md).

**Integer-window checkpoint foundation (2026-10-12; CPU verified):**
Added three signed cell offsets with a version 2 checkpoint encoding. Zero-offset
saves remain version 1 and byte-compatible: the historical 50k fixture reencodes
to the same **8,000,136 bytes**; nonzero offsets add **12 bytes**, preserving all
particle words. Version 2 binds its version into the digest; framed saves retain
the offset metadata. Four focused tests and 129 checkpoint CPU regressions pass.
At this codec-only stage live restore and origin-only tile/canonical consumers
rejected nonzero offsets explicitly. Live restore is now admitted by the validated
API above; raw tile consumers still reject, canonical client windows are now explicit.
No public path silently drops
offsets. This codec evidence is not traversal acceptance.
[Codec and compatibility evidence](benchmarks/soil/2026-10-12-lattice-checkpoint/SUMMARY.md).

**Projected compact window lifecycle (2026-10-12; verified component):**
Compact fixed-order soil with projected Newton coupling passes the coupled window
publication/cache-invalidation/continued-step/rollback fixture in **78.86 s including
setup**, below the unchanged 90-second cap. Publication uploads zero bytes; the
two-particle validation capture reads 336 bytes. This is not a paired trajectory
or actual tractor traversal gate. **Actual-grid limitation:** the 0.1-metre tractor
grid cannot represent a one-cell origin change as the exact lattice shift required
by the legacy origin-only API; these older fixtures use 0.125. The new integer-cell
API above addresses the decimal-grid limitation without changing resolution or
weakening exact-origin admission. Full traversal remains separate.
The actual-configuration CPU regression confirms positive/negative one-cell ratios
of **+/-0.99999903142453683**, rejected without changing checkpoint bytes. Three
window CPU checks pass; no GPU run was needed for that arithmetic result.
[Projected lifecycle evidence](benchmarks/soil/2026-10-12-coupled-window-projected/SUMMARY.md),
[decimal-grid contract](benchmarks/soil/2026-10-12-window-decimal-contract/SUMMARY.md).

**Next required coordinate implementation (not complete):**
1. Keep an explicit lattice anchor separate from the integer active-window offset;
   a cell move must not repeatedly round the physical origin into f32.
2. Finish the same anchored coordinate expression for transfer weights, node positions,
   collision queries, clamps, active/compact indexing and particle ordering.
   Compute the global stencil base/fraction from anchor-relative coordinates first;
   subtract the integer offset only from the base. Reconstruct node positions from
   local node index plus the offset, not a repeatedly rounded physical origin.
   Core MPM/marking/ordering and separate mass-contact/candidate paths are now
   implemented and component-verified above; integrated actual-tractor proof remains.
3. Bound integer ranges and local precision; handle world-frame rebase separately
   from physical window movement, retaining canonical global identities/anchors.
   Node/stencil bounds and moved-axis support admission are implemented; arbitrary
   world-frame precision/rebase and long-horizon support remain separate gates.
4. Version checkpoint/frame persistence and restore compatibility explicitly. Old
   archives must retain their original zero-offset semantics and validation.
   Versioned checkpoint save/restore, canonical selection/writeback and disk-world
   integer-offset state now pass component checks; full traversal remains open.
5. Update coupled bindings, rollback epochs and render/surface observation contracts
   together; no cached consumer may silently use the previous grid interpretation.
   Committed projected lifecycle/rollback is component-verified; actual renderer
   and persistent-terrain traversal acceptance remains open.
6. Verify nominal 0.1-metre cell moves, boundaries and save/reload, then paired actual
   tractor trajectories under the unchanged physical/trajectory/resource gates.
   Do not claim this coordinate work alone supplies sleep or bounded traversal.

**Coupled machine/window API (2026-10-12; native-prediction lifecycle verified):**
Added `RigidWorld::move_mpm_soil_grid_window` and
`SoilTractor::move_soil_grid_window` at committed frame boundaries. Bounded capture
validates lattice/support; publication retains particle buffers and machine poses,
invalidates prepared coupling and last-substep observations, and advances no frame.
The guard now checks only moved axes, allowing unchanged floor contact and no-op
windows. Two CPU checks pass. A two-particle/one-body native-prediction fixture
passes exact publication, unchanged machine state, continued coupling and exact
failed-frame rollback/retry in **13.30 s including setup**, with **zero upload bytes**
and **336 capture-readback bytes** during publication. The generic constructor
hit its unchanged 90-second cap before window assertions and remains unverified.
No paired machine trajectory, active/compact projected coupling, actual tractor
traversal, renderer coordination or 4 ms/60 FPS acceptance is claimed.
[Generic constructor cap](benchmarks/soil/2026-10-12-coupled-grid-window/SUMMARY.md),
[native coupled lifecycle](benchmarks/soil/2026-10-12-coupled-window-native/SUMMARY.md).

**Public grid-window repeat/physical contract (2026-10-12; verified):**
Independent shifted and unshifted owners now pass the original separate gates:
**eight bit-exact self-repeats** and **four cross-window physical comparisons**
below 1e-6, across active/compact and explicit/implicit configurations. All sixteen
owners preserve particle IDs/counts through stepping and canonical writeback/reopen.
Maximum difference is **5.104810e-8** in affine velocity; positions and material
history remain exact. The external non-test-library run takes **10.42 s including
setup**. The earlier stronger cross-window bitwise assertion remains a recorded
failure, not a requirement silently relaxed or a cause proven. This does not close
sleep/wake, machine coordination, bounded traversal or 4 ms/60 FPS.
`make test-soil-window RELEASE=1 RUN_WITH=nixVulkan` exposes this check and joins
the S07 chain; it precompiles before the 90-second GPU cap. Older failing gates
remain in place, so this does not declare the complete S07 lane passing.
[Public matched-control evidence](benchmarks/soil/2026-10-12-grid-window-public-repeat/SUMMARY.md),
[Make routing](benchmarks/soil/2026-10-12-grid-window-make-route/SUMMARY.md).

**Resident grid-window publication (2026-10-12; explicit/implicit verified):**
Physical-window changes now retain GPU particle payloads instead of uploading
unchanged streams. Six measured publications allocate **zero GPU buffers** and
upload **zero bytes**; each still reads back **20,656 bytes** for required validation.
Three layouts under explicit and two-sweep implicit stepping pass **48 continued
comparisons with zero physical-stream difference and exact control repeats**,
plus durable reopen. Nine CPU checks and four GPU tests pass, GPU time **3.43 s
including setup**. Coordinate recentering still uploads translated state. This
does not establish bounded traversal, support release, or the frame budget.
The external public-library fixture is **not yet accepted**: after correcting
empty-tile enumeration, its active/explicit shifted and control runs differ by
up to **2.753222e-8** in affine velocity (positions and material state exact).
That is below the original 1e-6 physical bound, but fails this fixture's stronger
cross-window exact assertion. Independent self-repeat attribution remains needed;
implicit/compact public cases were not reached. No default or full-gate promotion.
[Resident-path evidence](benchmarks/soil/2026-10-12-resident-grid-window/SUMMARY.md),
[public-boundary failure and offline comparison](benchmarks/soil/2026-10-12-grid-window-public-payload/SUMMARY.md).

**Physical grid-window movement (2026-10-12; initial detached explicit fixture):**
`MpmSoilCheckpoint::with_grid_window` and canonical disk-owner
`begin_grid_window_move` move solver bounds without translating soil or the world
frame. They require an exact lattice shift and preserve particle stencil fractions;
transfer/Jacobi support must fit both windows. No particles are silently evicted.
Asynchronous publication preserves identity, pins and canonical ownership; a
support-loss rejection leaves the previous live state intact. Three supported
fixed-order layouts pass **24 continued-step comparisons with zero physical stream
difference and exact controls**, plus durable drop/reopen, in **1.33 s including
setup**. Nine CPU checks and four shared-publication GPU regressions also pass
(**2.49 s including setup**), including normal-suite use without capture settings.
This is a 129-particle explicit fixture, not physical sleep/wake, machine
coordination, implicit trajectory acceptance, bounded long traversal or 4 ms/60 FPS.
[Implementation and evidence](benchmarks/soil/2026-10-12-grid-window/SUMMARY.md),
[transaction regressions](benchmarks/soil/2026-10-12-grid-window-regressions/SUMMARY.md).

**Corrected transfer pair (2026-10-12; opt-in gather integration verified):**
Rounded non-native gather now has a public preparation policy with budget checks,
owner reuse and prepared-record invalidation when precision changes. Standard
preparation restores standard arithmetic; defaults and native shaders are unchanged.
Eight frozen50k corrected scatter/gather comparisons give exact stiffness agreement,
**9.532756e-7** captured-term error and unchanged particle checkpoints. Gather
repeats exactly; active/compact implicit evolution differs by at most **8.940697e-7**.
Both GPU tests pass in **1.07 s including setup**. A separately compiled non-test
library passes all **eight public lifecycle cases in 1.44 s including setup**;
the first fixture's missing ordering preparation is retained as a failed attempt.
Six measured alternating pairs (after two warmups) include
selected ordering, active-grid work and all transfer dispatches: median GPU time
is **0.513024 ms original scatter, 0.541696 ms corrected scatter, 2.027008 ms corrected
gather**. Gather costs **3.26–4.12x** corrected scatter in these paired measurements.
This zero-simulated-time fixture does **not** establish frame throughput or the 4 ms
target. The old default numerical comparison still fails; no default is promoted.
[Corrected transfer evidence](benchmarks/soil/2026-10-12-corrected-transfer/SUMMARY.md),
[public integration](benchmarks/soil/2026-10-12-corrected-gather-public-prepared/SUMMARY.md).

**Retained stiffness owner (2026-10-12; opt-in runtime integration verified):**
Added public `prepare_compensated_stiffness(max_bytes)`,
`set_compensated_stiffness(enabled)`, state/storage queries and disabled release.
The non-native two-pass owner retains one tail buffer and dispatches clear/scatter/
resolve without diagnostic contribution buffers or readbacks. Dense, active,
compact and deferred-lookup layouts are supported; budget, stale owner/record,
exclusive selection, release and grid-change guards preserve lifecycle contracts.
Six uninstrumented frozen50k runs retain **exact reference stiffness**, full particle
checkpoint identity and **9.532756e-7** maximum captured-term error. Sixteen unit
evolution cases stay within **7.85235e-7** of original scatter across particle
history streams, with exact material IDs. A separately compiled **non-test library
integration test passes all 16 public lifecycle cases**. Unit GPU work completes
in **1.09 s**, public GPU work in **2.27 s**, both including setup. Prepared encoding
observes zero hot buffers/uploads/readbacks; unit pipeline/binding counters stay
unchanged. **Defaults/native selection are unchanged.** Rounded non-native gather
integration and matched-work timing remain pending; the old default comparison,
4 ms/60 FPS and full-plan acceptance are not declared fixed.
[Uninstrumented identity and evolution](benchmarks/soil/2026-10-12-retained-stiffness/SUMMARY.md),
[public API integration](benchmarks/soil/2026-10-12-retained-stiffness/public/SUMMARY.md).

**Two-level non-native stiffness correction (2026-10-12; component passes):**
Implemented a private compensated scatter retaining rounding residuals at both
workgroup and global accumulation levels, including collision fallback. Local-only
compensation fails and remains recorded. The two-level candidate passes six
alternating frozen50k comparisons: **zero stiffness difference from rounded gather**,
**9.532756e-7** maximum error against the captured-term f64 sum, all **1.35 million
contributions exact**, and the full particle checkpoint unchanged. Mass/momentum
differences are below **1.68e-8 / 5.53e-10**. Three CPU contracts and the **0.69 s
setup-inclusive GPU run** pass without changing the 2e-5 field bound. The original
scatter control still fails; this does not close its production regression gate.
The candidate adds **131,072 tail bytes, one clear, one resolve dispatch and 512
workgroup bytes**. **Production integration and uninstrumented lifecycle/performance
validation remain pending**; native/production selection and the 4 ms/60 FPS result
are unchanged.
[Failed local-only candidate](benchmarks/soil/2026-10-12-stiffness-scatter/SUMMARY.md),
[two-level correction and exact evidence](benchmarks/soil/2026-10-12-stiffness-scatter/global/SUMMARY.md).

**Non-native stiffness error localized (2026-10-12; accumulation diagnostic):**
Private in-kernel capture shows **all 1,350,000 stencil records identical** between
instrumented scatter and gather: cell, weight, particle stiffness and contribution.
Instrumented gather remains byte-identical to current and archived controls; the
full restored particle checkpoint is unchanged. The successful capture takes
**0.46 s including setup**. Comparing the earlier six-round captures against f64
sums of these GPU terms finds maximum errors of **3.078547e-5 scatter**,
**2.634451e-5 plain gather**, and **9.532756e-7 rounded gather**. This resolves the
earlier CPU-formula ambiguity on this fixture: rounded gather improves accumulation
accuracy, while scatter also exceeds 2e-5 against the captured-term sum.
**The original scatter/gather comparison still fails**; neither its tolerance nor
the production backend changes. The next correction targets scatter accumulation,
not the material law. Two corrected harness lifecycle failures remain recorded.
This is not a continuum, calibrated, performance or whole-solver acceptance result.
[Captured terms, exact controls and decision](benchmarks/soil/2026-10-12-stiffness-contributions/initialized/SUMMARY.md).

**Non-native stiffness compensation (2026-10-12; gate still fails):**
Added a private compensated fixed-order stiffness sum, independent CPU rounded-term
reference, six alternating scatter/control/candidate realizations and a GPU rounding
probe. Plain source compensation produces exactly the control grids. Explicit
per-lane rounding makes compensation effective: the analytic probe's maximum error
falls from **0.9888 to 0.00024128**, within its 0.00025 rounding bound. Nevertheless,
the changed 50k candidate still differs from scatter by **3.051758e-5 > 2e-5**;
its CPU formula-reference error is **5.130374e-5**, with GPU/CPU contribution
arithmetic not yet isolated. The two GPU comparisons take **0.48 / 0.38 s** including
setup; two source contracts pass. Fixed-order repeats, mass/momentum and particle
checkpoint preservation pass. Thirty nodes miss the interval allowed by six observed
scatter realizations; those observations do not prove an unavoidable tolerance failure.
Both candidates remain **test-only**, with no production or native-path change,
no tolerance relaxation and no actual-workload rerun. **Stiffness, 4 ms/60 FPS and
full-plan acceptance remain open**.
[Plain failed candidate](benchmarks/soil/2026-10-12-gather-stiffness/SUMMARY.md),
[rounded candidate and per-node evidence](benchmarks/soil/2026-10-12-gather-stiffness/rounded/SUMMARY.md).

**Grouped Newton norm maxima (2026-10-12; exact but kept test-only):**
A private candidate groups the existing 256-leaf maximum-expression tree into
three barrier stages instead of eight, without changing operand order, invalid
flags, scaled sums, buffers or numerical budgets. Both CPU contracts pass; the
control WGSL/SPIR-V exactly match method 30. All eight saved inputs pass independent
physics (**max 8.042704e-7**), all five public buffers match control and archived
method-30 buffers exactly, and **128 self-repeat checks** pass. Setup-inclusive
replay completes in **74.45 s** under the unchanged 90-second cap. Paired median
gains are only **0.94–1.48%**; the candidate remains test-only, with no runtime
promotion, constant sweep or actual-workload rerun. The prior mask-gather production
state remains, and **4 ms/60 FPS and the non-native stiffness gate remain unmet**.
[Candidate, exact controls and decision](benchmarks/soil/2026-10-12-newton-grouped-norm/SUMMARY.md).

**Workgroup support-mask gather (2026-10-12; integrated opt-in improvement):**
Fixed-order P2G now emits exact integer stencil-support masks in workgroup memory
and visits contributing particles in the original tile/chunk/slot order. It retains
the same floating expressions, two dispatches and 64-byte/particle scratch budget;
atomic defaults are unchanged. Six paired saved50k component measurements improve
GPU median **1.037824 → 0.584192 ms (43.71%)**, all with exact grids/history.
The actual50k runs each complete 33 frames / 0.55 seconds; **all recorded machine
states, final positions and the full 8,000,136-byte particle archive match the
preceding direct-gather control and each other exactly**. Observed mean P2G is
**15.111 / 15.121 ms/frame**, versus earlier independent observations around 32 ms;
that whole-run comparison is not paired causal attribution. Wall medians remain
**350.187 / 347.941 ms**; mean Newton is **211.52 / 211.63 ms/frame**.
Seven gather CPU and 21 benchmark checks, nine GPU lifecycle/reference checks and
strict actual audits pass. A separate **non-native scatter stiffness comparison
fails 2e-5 (error 2.861023e-5)**; it is retained, not relaxed or rerun unchanged.
One missing-fixture harness failure is also retained; corrected setup runs only
previously unexecuted work. **Physical/reference and 4 ms/60 FPS gates remain open**.
[Paired component evidence](benchmarks/soil/2026-10-12-gather-masks/SUMMARY.md),
[runtime integration, actual repeats and failed gate](benchmarks/soil/2026-10-12-gather-masks-runtime/SUMMARY.md).

**Bounded final particle-history observation (2026-10-12; actual repeat exact):**
Added public committed-frame soil checkpoint observation with an explicit particle
limit and pre-allocation uncommitted/faulted/stale-model guards. The benchmark's
opt-in final capture runs after all measured frames and records its own costs.
Two independent actual 50k executions match **all 8,000,136 encoded bytes**, including
velocity, APIC C, elastic F, mass/reference volume, log-Jp, material IDs/table and
compaction/impulse diagnostics. All 33 machine records and final positions also
match the previous fixed-order control. This verifies the **final particle state**,
not every intermediate material state, Newton/grid scratch, world IDs or restart.
**21 CPU checks pass**. The changed lifecycle test hits its unchanged **90 s cap
during machine construction**, before observer assertions; rejection/rollback
coverage remains unverified and is not erased by actual-workload success.
Wall medians **395.60 / 372.73 ms** still miss every measured 60 Hz deadline.
No default, physical budget or numerical limit changed; **4 ms remains unmet**.
[Full-state implementation, exact comparison and retained timeout](benchmarks/soil/2026-10-12-full-soil-state/SUMMARY.md).

**Earlier fixed-order 50k self-repeat (2026-10-12; machine/position only):**
Two independent public benchmark executions each complete **33 frames / 0.55
simulated seconds**, retaining all 34 substeps/frame. All recorded machine frames
and all **50,000 final particle positions match byte-for-byte**. No full final
material/deformation history was captured in these earlier runs; the later
final-state observation above extends this evidence without claiming world restart.
Both runs observe zero hot buffers. Wall medians are **377.08 / 384.92 ms**, and
every measured frame misses 60 Hz; **4 ms acceptance still fails**. Both timings
are retained, not selected for a favorable result. The atomic default is unchanged.
This provides a real repeatable machine/position control for subsequent work; it
does not explain the earlier small-fixture timeout or establish calibrated accuracy.
[Two-run evidence and scope](benchmarks/soil/2026-10-12-fixed-order-actual/SUMMARY.md).

**Constructor isolation and fixed-order profiling (2026-10-12):**
A distinct single-owner, construction-only diagnostic passes in **68.45 s** under
the unchanged 90-second cap. Soil/ordering/gather finish by **0.022 s**, rigid
attachment by **0.079 s**, and coupling by **68.109 s**. This localizes construction
cost but does not explain the earlier full-test timeout or verify any stepping.
Also fixed the benchmark's profile validator: it had rejected ordering even when
fixed-order transfer was selected. It now requires matching sort/ordering flags
and the exact ordering-before-grid-reset stage per substep; atomic checks remain.
All **20 benchmark CPU tests pass**, including both transfer selections. Subsequent
actual self-repeats are recorded above; full integration and **4 ms remain open**.
[Constructor isolation](benchmarks/soil/2026-10-12-tractor-construction-diagnostic/SUMMARY.md),
[profile contract fix](benchmarks/soil/2026-10-12-fixed-order-profile/SUMMARY.md).

**Public tractor transfer selection (2026-10-12; GPU integration incomplete):**
Added `SoilTractorTransfer`, a native-prediction constructor that prepares ordering
and gather before attachment, and readback-free live transfer/resource information.
The existing constructor still selects atomic transfer. The benchmark now accepts
`--soil-transfer fixed-order:3200000` (Make: `SOIL_TRACTOR_BENCH_TRANSFER`) with
canonical parsing, explicit native-prediction requirements and requested/live
identity checks. The 3.2 MB budget covers gather scratch, not ordering/driver memory.
**21 CPU checks pass**. The short public-API GPU test reaches its unchanged **90 s
cap** before reporting a result; logs cannot locate the stalled phase. No GPU
acceptance or speed claim, cap increase or unchanged retry follows. Later constructor
isolation and actual self-repeats above do not close this small-fixture timeout.
[Implementation and retained timeout](benchmarks/soil/2026-10-12-tractor-fixed-order-runtime/SUMMARY.md).

**Native active/compact fixed-order transfer (2026-10-12; runtime prerequisite):**
The retained gather now supports native active-tile and compact grids, not only
native dense grids. Dispatch layout and native explicit zero stiffness are separate
specializations. Enabled gather rejects grid-mode changes; after disabling it,
selection rejects a stale pipeline until explicit reprepare matches the new mode.
**160 exact full-history comparisons** pass across paired dense/active/compact
owners and four reseed sizes, with zero hot buffers and zero explicit stiffness.
Existing gather, dense-dispatch, public lifecycle and durable-recenter regressions
pass. Defaults are unchanged. Later public tractor selection and actual final-state
self-repeats are recorded above; **full-plan and 4 ms acceptance remain open**.
[Runtime prerequisite evidence](benchmarks/soil/2026-10-12-native-active-gather/SUMMARY.md).

**Method-30 full50k follow-up (2026-10-12; acceptance failed):**
The changed runtime completes all **33 frames / 0.55 simulated seconds**, with
three warmups and thirty measured frames, under the unchanged 180-second cap.
Machine-plus-soil wall median is **328.990841 ms**; all **30/30** measured frames
miss 60 Hz. Mean Newton work is **212.145869 ms/frame** and remaining MPM work is
**22.647501 ms/frame**: even non-Newton MPM exceeds the whole 4 ms soil budget.
Zero hot GPU-buffer allocations are observed. The separate-run method-28 trajectory
comparison first exceeds **2e-5 at frame 13** and reaches **0.02419255** maximum
component difference. This is not matched-input causal attribution. Defaults and
numerical limits remain unchanged; no unchanged retry follows. Deterministic
whole-workload inputs and larger cost reductions remain necessary.
[Full-workload evidence](benchmarks/soil/2026-10-12-parallel-indices-actual/SUMMARY.md).

**Stable parallel active indices runtime (2026-10-12; opt-in verified):**
The `parallel-indices-newton` feature now applies the stable scan in the real
conditioned runtime, with capture method **30** and explicit benchmark metadata.
All eight saved-input cases match all five method-29 buffers exactly, with three
identical cold executions per case and physical residual at most **8.043e-7**.
Replay passes in **65.72 s including setup**. The four-frame coupled world passes
in **39.08 s**, including zero hot allocations, rollback/retry, profiling lifecycle
and exact soil/recovery parity with method 29. Unsupported-device rejection and
capture roundtrips pass. Defaults, capacities and numerical budgets are unchanged.
This integrates the previously measured solver improvement; it is **not a new
full-50k timing or trajectory result**. The **4 ms/60 FPS target remains unmet**.
[Runtime evidence](benchmarks/soil/2026-10-12-parallel-indices-runtime/SUMMARY.md).

**Stable parallel active indices prototype (2026-10-12; runtime integrated above):**
Replaced the test candidate's lane-zero contact scan with parallel contiguous
counts, an ordered prefix and disjoint index writes in existing scratch. **192
index/fault/suffix checks and 384 endpoint checks pass exactly**. The isolated
setup-plus-certificate component improves **76–83%**, but the relevant full-Newton
comparison shows **2.44–8.60%** gains across seven stable cases. All eight cases pass
independent physics, zero public-byte difference, 128 exact repeats and archived
method-29 control equality in **83.29 s including setup**. The first case's changing
timing regime is retained and not claimed stable. Proceed to opt-in runtime
integration; production/defaults and the unmet 4 ms target remain unchanged.
[Stable-scan evidence](benchmarks/soil/2026-10-12-newton-active-indices/SUMMARY.md).

**Standalone certification diagnostic (2026-10-12; endpoint evidence):**
The small entry point completes eight archived-input cases in **1.11 s including
setup**, with **384 exact checks** across accepted/cold/poisoned/recovered states.
Accepted publication matches archived method 29 and independent physical checks;
input states remain unchanged. Accepted one-copy medians are **0.676–0.760 ms**,
including serial active-index setup; adding a second certificate adds **0.120–0.158
ms**. These are endpoint diagnostics, not additive whole-solve shares or speedup.
The next hypothesis is stable parallel construction of the currently serial
4096-slot active list; it is not yet implemented or measured. Production and the
unmet real-time target remain unchanged.
[Standalone evidence](benchmarks/soil/2026-10-12-certification-isolated/SUMMARY.md).

**Certification-cost diagnostic (2026-10-12; no measurement):**
A single-call-site, shared-program 1x/2x certification probe passes its source/compiler
contract but times out during complete Newton pipeline setup at the unchanged 90 s
cap. No solve or timing pair runs; **certification cost remains unattributed**.
The failed attempt is preserved, with no unchanged retry or production change.
Further attribution requires a smaller isolated entry point, not another expansion
of this monolithic pipeline. The full acceptance target is unchanged.
[Diagnostic evidence](benchmarks/soil/2026-10-12-newton-certification-cost/SUMMARY.md).

**Invocation-private complementarity samples (2026-10-12; test-only):**
A different formulation retains original function signatures and shares samples
through invocation-private arrays. It completes the setup-inclusive replay in
**69.01 s**; eight inputs pass with **zero public-byte difference, 128 exact
self-repeats**, maximum physical residual **8.04271e-7**, and 32 exact edge checks.
However, paired median reductions are only **0.02–0.43%**. It remains test-only;
this ends the duplicate-sample optimization line without production promotion or
another full50k run. Also corrected feature guards for three method-29 source tests
and verified that protected-only builds exclude them. **The 4 ms target remains unmet.**
[Private-sample evidence](benchmarks/soil/2026-10-12-newton-private-samples/SUMMARY.md).

**Complementarity sample sharing (2026-10-12; setup gate failed):**
Added a test-only variant that shares normal/wall complementarity results between
contact residual and Jacobian evaluation using explicit local values. Two compiler/
source contracts and **32 exact GPU edge comparisons** pass. However, the complete
Newton pipeline does not finish setup within the unchanged **90 s** cap. No solved
input or timing pair is produced; there is **no demonstrated speedup or nonlinear
acceptance**. The variant remains test-only, with no unchanged retry or budget
increase. Production selection and the unmet 4 ms target are unchanged.
[Sample-sharing evidence](benchmarks/soil/2026-10-12-newton-complementarity/SUMMARY.md).

**Accepted-trial rate reuse (2026-10-12; test-only):**
Newton accepted trials repeat the same machine-rate projection on the next base
evaluation. A one-use reuse flag skips only that projection; smoothing, derivative
evaluation, arithmetic and solver budgets are unchanged. Eight frozen-input pairs
pass with **zero public-byte difference, 128 exact self-repeats** and maximum physical
residual **8.04271e-7**. All paired median changes improve, but only **0.32–0.83%**.
The candidate remains test-only: no production promotion or full-tractor rerun is
justified by this small gain. Source/compiler contracts and the setup-inclusive
**68.09 s** replay pass. **The 4 ms target remains unmet.**
[Rate-reuse evidence](benchmarks/soil/2026-10-12-newton-accepted-rates/SUMMARY.md).

**Cached stencil-base trial (2026-10-12; rejected):**
Three cached integer coordinates fit the stress record's existing padding and
preserve its 64-byte stride. Exact state/grid comparisons pass, but all six measured
50k transfer pairs regress: median GPU **1.006080 → 1.038336 ms (3.21% slower)**,
wall **1.121644 → 1.155834 ms (3.05% slower)**. The candidate is retained only as a
test shader; the production shader is restored byte-for-byte to the accepted
range-pruned version. **134 CPU and 7 GPU restoration checks pass**. No unchanged
performance retry or full-frame claim. **The 4 ms target remains unmet.**
[Rejected trial and restoration](benchmarks/soil/2026-10-12-gather-base-cache/SUMMARY.md).

**Fixed-order gather range pruning (2026-10-12; measured component gain):**
Each grid node now skips source tile ranges outside its quadratic transfer stencil,
reducing eligible ranges from 27 to at most eight without reordering contributing
particles. No extra buffers or dispatches are added. **134 CPU and 9 GPU checks
pass**, including 32 bit-exact evolved state/grid comparisons across active-dense,
compact and two native-dense layouts. On the frozen 50k native fixture, all control/
candidate pairs and repeats are bit-exact; median ordering/reset/P2G/grid-update GPU
time falls **1.054720 → 1.010176 ms (4.22%)**, completed wall **1.174096 → 1.126243 ms
(4.08%)**. All six measured pairs improve; two warmup pairs are excluded. This
advances no simulated time and includes no machine solve, so **the full 4 ms soil
target remains unmet**. Ordinary atomic defaults and physical thresholds are unchanged.
[Pruning evidence](benchmarks/soil/2026-10-12-gather-range-pruning/SUMMARY.md).

**Runtime fixed-order P2G (2026-10-12; explicit opt-in):**
The fixed-order gather used by the passing streaming controls is now available in
the non-test library, together with its mode/ordering-replacement guards and retained
resource lifecycle. `prepare_deterministic_p2g_with_budget` bounds the stress scratch
(64 bytes per maximum particle, excluding ordering/driver storage); enable explicitly
with `set_deterministic_p2g(true)`. Disabled scratch can be released explicitly.
Default accumulation remains unchanged. **133 CPU and 7 GPU checks pass**, including
two external integration tests against the non-test library: repeatable physical
history across reseeds, zero hot owned allocations/transfers, canonical live rebase,
eight exact repeated evolved storage boundaries and durable reopen. Native dense
regressions also pass. This enables the chosen streaming route; it does not make
ordinary atomics repeatable or certify whole-machine reactions, full 50k trajectory,
bounded field traversal or **4 ms performance**.
[Runtime fixed-order evidence](benchmarks/soil/2026-10-12-runtime-fixed-order/SUMMARY.md).

**Continued rebase trajectory gate (2026-10-12; acceptance still fails):**
Added paired eight-step, 129-particle comparisons across four ordinary grid modes
and three supported fixed-order modes. Every case has two unre-based controls and
one rebased owner; comparisons subtract split world offsets before evaluating small
position differences. All **56 physical comparisons pass 1e-6**, maximum **2.38419e-7**.
Fixed-order stepping passes **24/24 exact self-repeats**; ordinary atomic stepping
passes only **1/32**. The combined GPU gate therefore **fails**, despite unchanged
physical thresholds passing here. This directly supports fixed-order stepping for
repeatable streaming work; it does not certify ordinary stepping, longer trajectories,
physical moving windows, the 50k tractor case or timing. **132 CPU checks pass**.
The failed GPU result, all 168 framed snapshots and frozen sources are retained.
[Paired trajectory evidence](benchmarks/soil/2026-10-12-recenter-trajectory/SUMMARY.md).

**Canonical live coordinate rebasing (2026-10-12; S07 increment):**
`MpmSoilDiskWorld::begin_recenter` now coordinates an exact local grid/particle
translation with the compensating world offset and canonical selection provenance.
Poll through `poll_region_update`; capture is asynchronous, and other updates and
steps are frozen until publication. The operation preserves device ID buffers and
identity epoch, world-tile residency, physical history, consumed swept horizon and
activity duration. GPU pins reuse their buffers with translated local parameters.
Durable canonical tiles need no rewrite. **131 CPU and 27 GPU checks pass**, including
all four grid modes, evolved-state rebase, further write/reopen/resume and precision
failure retry. This is an explicit boundary operation with host selection assembly,
capture validation and full upload, not an accepted hot-step route. Physical moving
boundaries/sleep, machine/renderer rebasing, bounded traversal and the **4 ms target
remain open**; later independent trajectory equivalence is not established.
[Live recenter evidence](benchmarks/soil/2026-10-12-canonical-live-recenter/SUMMARY.md).

**Canonical live disk ownership (2026-10-12; S07 increment):**
`MpmSoilDiskWorld::new_canonical` now uses canonical background reads, selected-frame
assembly and writes through the existing capture/publication lifecycle. Construction
does not load particle blobs. Initially the frame was fixed for the owner's lifetime;
the later exact coordinate transaction above lifts that restriction for canonical storage.
all active particles remain pinned. **131 CPU and 24 GPU checks pass**, including
four grid modes, actual device identities, overlap/resume, moved-tile write budgets,
durable recovery, host/GPU pins and conversion faults. Initial fixture failures
and corrected frozen runs are retained. This does not establish bounded traversal,
physical moving boundaries/sleep, in-session rebasing or the **still-unmet 4 ms
target**. No new full-50k timing run was performed.
[Canonical live evidence](benchmarks/soil/2026-10-12-canonical-disk-world/SUMMARY.md).

**Canonical background I/O and frame assembly (2026-10-12; S07 increment):**
`MpmSoilCanonicalStore::begin_read` now performs bounded reads and exact requested-
frame assembly on one worker. `begin_writeback` owns the store/read/selection/update
through validation and publication. Nonblocking polls move results and original
inputs without payload copies; startup/retryable errors preserve the owner, while
publication failures and caught worker panics require reopen. Dropping tasks
neither joins nor cancels publication or releases locks early. **130 CPU and
20 GPU checks pass** on the recorded verification setup, including off-thread
assembly, pointer-identical payload return, failure/retry/drop semantics and a
129-particle background durable reload/restore/resume/writeback. Canonical workers
were subsequently integrated into fixed-frame live residency (see above). **Physical moving boundaries/sleep,
machine rebasing, bounded traversal and the 4 ms target remain open.**
[Canonical task evidence](benchmarks/soil/2026-10-12-canonical-background/SUMMARY.md).

**Incremental canonical tile store (2026-10-12; S07 increment):**
`MpmSoilCanonicalStore` now persists per-tile coordinates through bounded reads,
selected-frame partial writeback, destination merging and reopen. Existing
exclusive locks, synchronized blob installation and atomic HEAD publication are
shared with raw/framed storage. Reads retain canonical coordinates; explicit
frame assembly happens only for selected particles. Unchanged saves write zero
bytes, unrelated blobs remain untouched, and wrong-family/stale/foreign requests
reject. **123 CPU and 23 GPU checks pass**, with store/disk fixtures on ext4.
Eight canonical publication-failure cases recover complete old/new generations;
the actual GPU fixture reloads/restores evolved soil in an alternate frame,
resumes and commits it exactly. Direct operations remain blocking; the worker
increment above provides explicit background I/O. **Canonical live ownership, physical moving boundaries,
machine rebasing, bounded traversal and the absolute performance target remain
open.**
[Durable canonical storage evidence](benchmarks/soil/2026-10-12-canonical-store/SUMMARY.md).

**Canonical per-tile host archives (2026-10-12; S07 increment):**
`partition_canonical_tiles` now stores f64 tile-local positions, exact global IDs
and unchanged f32 history in `MpmSoilCanonicalTileArchive`. Bounded selections
assemble into an explicitly chosen world/grid frame without rebasing unloaded
tiles. Lossy anchor/residual/f32 conversions reject; signed-zero provenance is
retained. Transactional selected writeback merges moved-particle destinations,
preserves unloaded records and invalidates stale tokens. A distinct bounded,
checksummed codec rejects corrupt, cross-family or invalid payloads. **117 CPU
and four GPU checks pass**, including evolved 129-particle capture, codec reload,
exact selected-frame restore/device IDs, resume and writeback in four grid modes.
The durable store increment above consumes this representation; **canonical
background/live streaming is not integrated**.
Supplied grid origins do not certify physical moving-boundary equivalence.
Dynamic/machine rebasing, physical support/sleep, bounded field traversal and
the 4 ms target remain open.
[Canonical data and verification scope](benchmarks/soil/2026-10-12-canonical-tiles/SUMMARY.md).

**Live fixed-frame disk ownership (2026-10-12; S07 increment):**
`MpmSoilDiskWorld::new_framed` now owns a framed durable store through the existing
capture/write/load/device-publication lifecycle. Local solver positions remain
separate from the immutable `world_origin`; residency, activity and support pins
use exact split-coordinate world keys. Swept envelope bounds are local, with
local numerical padding and world-tile enumeration. Optional GPU pins translate
tile-aligned frames into the existing local integer domain; unsupported fractional
frames reject GPU pin preparation and can use host pins. **111 CPU and 23 GPU
checks pass**, including overlapping requests, retained transfers, activity,
exact reload/resume in four grid modes, swept horizons and retryable capacity
errors. Store/disk fixtures ran on recorded ext4. No long-traverse physical
equivalence or speedup is established. **Dynamic/machine rebasing, integration of
canonical per-tile storage, physical sleep/support acceptance and absolute performance remain
open.**
[Live framed ownership evidence](benchmarks/soil/2026-10-12-framed-disk-world/SUMMARY.md).

**Framed background store tasks (2026-10-12; S07 increment):**
`MpmSoilFrameStore::begin_read` and `begin_writeback` now move bounded selected
I/O into the existing exclusive-owner workers. Typed, nonblocking task polls
return the framed owner and move original payload buffers without cloning or
revalidating them. Writes reject changed frames before spawning; failed requests
return retry inputs. Publication errors and caught worker panics require reopen.
Dropping tasks neither joins nor cancels publication or releases locks early.
**109 CPU tests and three GPU store checks pass on ext4**, including an actual
129-particle background reload, exact physical/history/device-ID restore, resume,
background writeback and exact reopen. No full-frame speedup or later trajectory
equivalence is established. Full save/load and detached capture/snapshot remain
blocking; thread creation/metadata checks have no hard latency guarantee.
**Dynamic rebasing, canonical per-tile coordinates, physical
traversal and absolute performance remain open.**
[Lifecycle and GPU evidence](benchmarks/soil/2026-10-12-framed-background/SUMMARY.md).

**Durable framed tile store (2026-10-12; S07 increment):**
`MpmSoilFrameStore` now preserves the shared world offset through incremental
full save/load, bounded selected reads, same-frame moved-particle writeback and
reopen. Wrong-family, stale, foreign and changed-frame partial updates reject;
unchanged saves write zero bytes. Full frame replacements advance generations.
The existing lock/synchronization/atomic-publication protocol is shared, not
replaced. **102 CPU tests and two GPU store checks pass on recorded ext4**, after
retaining the initial tmpfs run and twelve disk-world GPU regressions. Eight
framed publication-failure cases recover complete old/new generations; these
are not power-loss tests. Its direct methods block and copy selection snapshots;
the background task increment above supplies worker-backed selected I/O.
**Dynamic rebasing, canonical per-tile
coordinates, physical traversal and absolute performance remain open.**
[Persistent-filesystem evidence and API scope](benchmarks/soil/2026-10-12-framed-tile-store/persistent/SUMMARY.md).

**Framed GPU identity handoff (2026-10-12; S07 increment):**
`MpmSoilFrame::restore_selection` now installs local physical/history state and
global IDs together, validates grid-origin changes and publishes the frame offset
after submission. `checkpoint_tiles_with` captures identified framed archives;
recenter preserves identity mapping/buffers while invalidating physical prepared
work. Plain restore still invalidates identity. **96 CPU and 13 GPU checks pass**,
including four grid modes, exact device ID words, archive reload/resume, stale
work, resource reuse and nonmutating capacity/device/configuration failures.
First identified restore creates two retained ID buffers plus two upload buffers;
later restores reuse IDs but still create the two upload buffers. This is a
detached, blocking archive/frame boundary API, **not live-world/machine rebasing,
zero-allocation streaming or full-plan performance acceptance**.
[Identity handoff and regression evidence](benchmarks/soil/2026-10-12-framed-device-identity/SUMMARY.md).

**Identified framed tile archives (2026-10-12; S07 increment):**
`MpmSoilFrameCheckpoint::partition_tiles` now returns a distinct framed archive
with exact world-tile keys, the shared world offset and stable global IDs. It
supports deduplicated selection, transactional same-frame writeback with moved
particle merging, checked versioned save/load and lossless whole-archive
rebasing into a new selection epoch. Unframed archive/codec semantics remain
unchanged; framed types cannot be consumed implicitly by old GPU/store owners.
The S07 CPU lane passes **96 tests**; three GPU archive checks pass, including
129 evolved particles reloaded/resumed in four grid modes with exact pre-resume
history and frame preservation. No later trajectory-equivalence or speed claim
is made. This is one shared host frame, **not canonical per-tile local storage,
live-world/machine rebasing, physical boundary closure or bounded field traversal**.
[Contracts, GPU scope and remaining work](benchmarks/soil/2026-10-12-framed-tile-archive/SUMMARY.md).

**Split-coordinate world-tile queries (2026-10-12; S07 foundation):**
`SoilTileLayout::tile_at_local` computes exact represented-operand floor membership
without adding local positions into a large world origin. It preserves tiny
offsets across tile boundaries and leaves the older rounded `tile_at` semantics
unchanged. `MpmSoilFrameCheckpoint::particle_tiles` exposes membership in slot
order, including after lossless recentering and codec round trips. The corrected
`make test-soil-fast PHASE=S07 RELEASE=1` passes **90 CPU tests**, including
3,960 independent integer-oracle cases and 2,575 rebase comparisons; **54 GPU tests
are explicitly ignored**. No GPU work, speedup or live-world rebasing is claimed.
Canonical tile-local persistence, live-owner/machine/renderer frames and physical
support/window boundary correctness remain open.
[Implementation, retained compile failure and root-lane evidence](benchmarks/soil/2026-10-12-split-tile-coordinates/phase/SUMMARY.md).

**Three-stage compensated reduction (2026-10-12; experimental method 29):**
`grouped-eight-newton` preserves method 28's exact addition tree while reducing
four synchronization stages to three. Eight paired component medians improve
**0.3345–0.9950%**, with zero public delta and 128 exact repeat checks. Runtime
replays match forty prototype buffers over three cold executions per input in
**68.81 s including setup**. The four-frame coupled Tine check passes unchanged
physical/angular/energy, profile, rollback and retry gates in **40.70 s**; soil
states and recovery exactly match method 28. Unsupported devices reject before
owned allocation. Defaults remain unchanged; no full50k rerun follows this small
component gain. **4 ms/60 FPS, full trajectory and full-plan acceptance remain unmet.**
[Component evidence](benchmarks/soil/2026-10-12-newton-grouped-eight/SUMMARY.md),
[runtime verification](benchmarks/soil/2026-10-12-newton-grouped-eight/runtime/SUMMARY.md).

**Grouped compensated sums (2026-10-12; experimental method 28):**
`grouped-sum-newton` groups two binary reduction levels per synchronization stage
while preserving the same parenthesized arithmetic. All eight saved-input paired
medians improve **1.956–4.461%**, with zero public delta and 128 exact self-repeats.
These are shared-program component timings, not full-frame or separate-binary
speedup evidence. The runtime feature rejects unsupported devices before owned
allocation and carries distinct method/capture/benchmark metadata; defaults stay
unchanged. Runtime replay passes in **63.32 s including setup**, matching forty
prototype buffers exactly over three cold executions per input. The coupled
four-frame Tine check passes physical, energy, profile and rollback gates in
**41.86 s**; four soil states and recovery match the prior runtime exactly.
The subsequent full50k diagnostic is recorded below. **4 ms/60 FPS, trajectory and full-plan acceptance remain
unmet**; this is an opt-in implementation, not production-default promotion.
[Component evidence](benchmarks/soil/2026-10-12-newton-grouped-sum/verified/SUMMARY.md),
[runtime verification](benchmarks/soil/2026-10-12-newton-grouped-sum/runtime/SUMMARY.md).

**Method-28 full50k diagnostic (2026-10-12; acceptance failed):**
All 33 frames completed within the setup-inclusive 180-second cap; construction
took **78.506525 s**. Wall median/p95 was **354.899966 / 461.586718 ms** and all
30 measured frames missed 60 Hz. Mean Newton work was **230.885513 ms/frame**;
remaining MPM work alone averaged **22.688188 ms/frame**, above the entire soil
budget. The separate-run machine trajectory comparison against method 27 first
exceeded **2e-5** at frame 13 and reached **0.02479109691902437**. This does not
isolate the source of divergence or establish causal whole-workload speedup.
Artifact/fixture/backend audits pass, but the frontier has no accepted points.
No unchanged rerun or default promotion is justified by this evidence.
[Full diagnostic and remaining work](benchmarks/soil/2026-10-12-grouped-sum-actual/SUMMARY.md).

**Disk-backed GPU occupancy pins (2026-10-12; S07 increment):**
The disk owner now supports bounded retained GPU occupancy queries and integer
tile dilation. Checkpoint and pin results join by revision before durable writes;
swept particle-support coverage remains included even with zero GPU halo. Domain
and capacity failures preserve the old device/store state and allow retry.
Eighty CPU and 21 focused GPU checks pass, including nine-stream/ID preservation,
actual tile crossing, support beyond occupancy, and resource accounting. The
measured transition creates no GPU buffers and uses 1,104 retained pin bytes;
this still reads the full active checkpoint plus 272 pin bytes at capacity 64.
Shared pin dispatch telemetry now counts both encoded stages. **No speedup,
physical sleeping, long-traverse or full S07 acceptance is claimed.**
[Implementation and bounded GPU evidence](benchmarks/soil/2026-10-12-disk-gpu-pins/coverage/SUMMARY.md).

**Guarded adjugate preconditioner (2026-10-12; rejected experiment):**
A test-only normalized cofactor inverse, guarded by determinant, backward-error
and finite-output checks, falls back to the original pivoted Schur path. Eight
saved-input comparisons pass physical/public gates, 128 exact self-repeat checks
and forty archived-runtime control-buffer comparisons. The shared-program check
completes in **66.15 seconds including setup**, but every paired median regresses
by **1.804–4.734%**. Both policies use one compiled graph; this is not a timing
comparison against an independently compiled production binary. Two earlier
dual-pipeline setup timeouts remain recorded. **No runtime promotion or full-scene
rerun**; numerical thresholds, work caps and defaults remain unchanged.
[Rejected candidate and bounded comparison evidence](benchmarks/soil/2026-10-12-newton-adjugate/shared/SUMMARY.md).

**Pivot-suffix preconditioner (2026-10-12; no promotion):**
A test-only variant skips already reduced identity columns during pivoted Schur
inversion, retaining the original pivot selection, fallback and final guards.
The inverse oracle/fallback check passes; eight paired saved-input cases retain
zero public delta, 128 exact self-repeats and forty archived control buffers.
The shared-program comparison takes **65.90 seconds including setup**, but paired
median changes are mixed (**-0.142% to +0.173%**, five faster and three slower).
This establishes **no consistent speedup**. Defaults and thresholds remain
unchanged; no full-workload rerun follows.
[Implementation and inconclusive performance evidence](benchmarks/soil/2026-10-12-newton-pivot-suffix/SUMMARY.md).

**Bounded stagnant-branch scheduler (2026-10-12; experimental method 27):**
`stagnant-branch-newton` resolves the saved frame-22 failure in **64 outer steps**
without changing iteration, release or publication limits. After a previous
floor release, the first zero-smoothing stagnant step may trigger the original
floor-release search before another Newton direction; follow-up settling targets
only the highest-residual unvisited contact and requires decreasing global merit.
The captured-case physical error is **6.19e-8**, with exact complete buffers over
three cold executions. Eight successful saved workload inputs preserve all forty
historical control buffers exactly. Runtime-feature replay matches the prototype;
tag/name/capture and benchmark metadata distinguish this policy from method 26.
Defaults remain unchanged. The coupled-world lifecycle test passes, and the full
50k diagnostic now completes **33 frames / 0.55 simulated seconds** with no hot
allocations. Measured wall median is **375.991 ms**, p95 **440.078 ms**; all thirty
measured frames miss 60 Hz. Comparison against the previous completed method-26
run exceeds the unchanged 2e-5 trajectory bound (maximum **0.0667944**, first
over-bound frame 13). These separate atomic runs do not isolate causality or
establish a paired speedup. **4 ms/60 FPS and trajectory acceptance remain unmet**;
method 27 stays opt-in, not a production-default promotion.
[Implementation, rejected alternatives and verification](benchmarks/soil/2026-10-12-branch-transitions/SUMMARY.md).

**Preparation program reuse and integrated strict runtime (2026-10-12):**
Preparation now caches immutable layout/pipelines per engine and shader source;
mutable owner buffers remain separate. Ownership/fault isolation, source variants,
cache clear/rebuild survival and cross-engine checks pass. Thirteen CPU, nineteen
benchmark, eight saved-input, fault/reset and coupled Tine lifecycle checks pass.
The changed coupled run takes **65.18 s**; tracing attributes about **64.2 s** to
the first Newton pipeline build versus **7 ms** for preparation. Other setup was
warm, so the overall improvement over the retained timeout is not attributed to
this cache. The non-test 50k diagnostic completes **33 frames / 0.55 simulated
seconds**, with zero hot allocations, but wall median **365.579 ms**, p95
**514.301 ms**, and all thirty measured frames missing 60 Hz. MPM p95 is
**409.758 ms**; response preparation median is **3.566592 ms per frame**.
Trajectory comparison against the prior method-27 run exceeds **2e-5** (maximum
**0.0671828**, first over-bound frame 13); separate atomic runs do not isolate
causality. **4 ms/60 FPS, trajectory and full-plan acceptance remain unmet**;
the frontier still has no accepted points. Defaults remain unchanged.
[Program reuse, coupled proof, full trace and failed acceptance gates](benchmarks/soil/2026-10-12-preparation-program-cache/SUMMARY.md).

**Strict preparation runtime integration (2026-10-12; experimental opt-in):**
`strict-rounding-newton-preparation` selects strict volatile preparation on native
Vulkan with explicitly requested passthrough shaders; unsupported devices fail
before owned allocation, with no silent backend fallback. Runtime versus ordinary
axis preparation passes six synthetic and eight saved-input exact/oracle checks;
actual-input medians improve from **0.407552–0.409600 ms to 0.165888 ms**
(**59.30–59.50%**). Missing-feature and fault/reset checks pass, and 112 saved
complete-buffer comparisons preserve reference and prototype parity. Twelve CPU
and nineteen benchmark tests pass. The coupled Tine lifecycle **times out during
construction at the unchanged 90-second cap**, before simulation; no unchanged
retry or full50k run follows. The feature is implemented, but integrated lifecycle
and full-frame acceptance remain incomplete. Defaults and numerical caps remain
unchanged.
[Runtime implementation, component evidence and retained timeout](benchmarks/soil/2026-10-12-strict-preparation-runtime/SUMMARY.md).

**Strict volatile preparation rounding (2026-10-12; component candidate):**
A test-only compiler replaces lane-private rounding atomic round trips with
volatile load/store pairs and decorates floating arithmetic with NoContraction,
retaining every arithmetic operation, refinement and tolerance. Bare volatile
rounding failed exact cancellation-fixture parity and is rejected; the strict
variant passes twelve CPU, six synthetic, eight actual-input and fault/reset
checks. Actual paired preparation medians improve from **0.419840–0.420864 ms**
to **0.165888 ms**, a **60.49–60.58% reduction versus the SPIR-V atomic control**.
The control separately preserves ordinary-axis results exactly but is slightly
slower; this is not a direct paired timing claim against ordinary WGSL. Raw
audits cover 504 samples and 196 complete response/status comparisons, including
ordinary-axis parity. The component remains test-only pending runtime opt-in
integration and lifecycle checks. **No full-frame 4 ms/60 FPS or trajectory
acceptance is established; defaults remain unchanged.**
[Compiler, rejected bare variant and strict component evidence](benchmarks/soil/2026-10-12-preparation-rounding/strict/SUMMARY.md).

**Preparation phase attribution (2026-10-12; matched control, no speedup):**
Eight saved workload inputs and 144 samples put roughly **81%** of preparation
time in response solving: median **0.329728–0.331776 ms** versus total
**0.407552–0.409600 ms**. Factorization takes **0.065536–0.066560 ms** and cleanup
**0.005120–0.006144 ms**. Both arms use unchanged axis-parallel preparation;
phase timestamps, raw-layout metadata, exact/oracle gates and 48 saved complete
response/status comparisons pass. The GPU check takes **0.32 s including setup**.
This prioritizes response arithmetic for the next experiment, but neither
identifies a particular arithmetic instruction as the cause nor establishes
full-frame savings. No runtime change or full-scene retry follows from this
diagnostic alone.
[Raw timing and audit evidence](benchmarks/soil/2026-10-12-preparation-phases/SUMMARY.md).

**Workgroup-local preparation intermediates (2026-10-12; not promoted):**
A test-only variant moves axis-parallel response/correction intermediates to
DOF-specialized workgroup arrays, retaining arithmetic, two mandatory/eight
maximum refinements, certification, device buffers and dispatch count. Eleven
CPU, six synthetic, eight actual-packet and candidate fault/reset checks pass;
raw artifacts preserve exact responses/status and 56 complete self-repeat
comparisons. All eight actual-input preparation medians regress slightly:
**0.402432–0.403456 ms → 0.404480 ms** (about 0.25–0.51%). Rejected for runtime
promotion; no full-workload retry. Workgroup storage requires 5504 bytes at ten
DOFs or 16768 bytes at 32, checked against the device before allocation.
[Source, cost comparison and exact/fault evidence](benchmarks/soil/2026-10-12-preparation-axis-shared/SUMMARY.md).

**Axis-parallel response preparation (2026-10-12; explicit runtime opt-in):**
`axis-parallel-newton-preparation` assigns separate GPU lanes to the three
response columns while retaining whole-contact convergence, original paired
arithmetic, two mandatory/eight maximum corrections and final certification.
Six synthetic and eight actual-packet comparisons preserve exact response/status
and physical/oracle gates; actual-input paired preparation medians improve by
about **55%**. Candidate fault/reset checks pass. Runtime source/metadata checks
and the four-frame coupled Tine fixture pass, including full soil-energy and
exact control-fault rollback/retry; coupled execution takes **64.19 s including
setup**. The subsequent 50k run **fails at frame 22**: first-parity preparation
succeeds, but Newton exhausts 64 outer iterations (fault 8; 397 inner iterations,
430 trials). Both failure packets are retained. Defaults stay unchanged; the
feature remains experimental, and partial timings do not establish full-frame
4 ms/60 FPS or trajectory acceptance. Saved-input replay passes in **0.32 s**:
serial and axis-parallel kernels both reproduce all **655360 archived response
bytes exactly**. This localizes the captured immediate fault away from changed
preparation values, not the complete trajectory's root cause. No unchanged
full-scene retry; investigate the saved Newton convergence failure next.
[Implementation and component evidence](benchmarks/soil/2026-10-12-preparation-axes/SUMMARY.md);
[runtime evidence](benchmarks/soil/2026-10-12-preparation-axes/runtime/SUMMARY.md).

**Captured Newton failure localization (2026-10-12):** bounded replay reproduces
the complete frame-22 failed state exactly. Final residual is concentrated at
contact 107; smoothing is already zero and only two of eight releases were used.
Both independent f64 and read-only GPU inspection find a valid conditioned
branch candidate at that archived state (GPU predicted merit **8.23e-8**, not an
applied solve). Two eligibility-only policies still fail cold convergence under
the original limits and are not promoted. Investigate when the candidate becomes
available and how branch checks are scheduled before the cap; neither extra
precision nor larger iteration limits is established as the remedy.
[Traces, rejected policies and exact-state inspection](benchmarks/soil/2026-10-12-axis-newton-convergence/SUMMARY.md).

**Post-release branch scheduling (2026-10-12; rejected prototype):** a bounded
test-only scheduler checks merit-decreasing branches immediately after a floor
release, retaining all iteration/release caps and publication checks. It still
fails the captured cold solve (physical error **0.139659**), so it is not promoted.
Per-step traces now locate the baseline releases precisely: contact 116 at step
55 and contact 106 at step **64**, leaving no outer step for the previously found
contact-107 candidate. The generic post-release search instead selects low-residual
contacts 118/121/61/113/51/155, consuming release slots through corrections to other
contacts rather than resolving contact 107. The next change must distinguish
an actual branch transition from an ordinary coupled Newton correction, and
schedule necessary transitions before the cap; merely broadening eligibility or
chaining generic merit decreases is insufficient. This is convergence evidence,
not a performance or completed solver fix.
[Prototype, exact prefix comparisons and rejection evidence](benchmarks/soil/2026-10-12-post-release-settle/SUMMARY.md).

**Cached preparation certificates (2026-10-12; not promoted):** a private
32-word mask avoids repeating component certificates before residual correction.
The original six synthetic comparisons pass; follow-up on eight actual tractor
packets passes exact response/status and unchanged 1e-6 physical/oracle gates,
with 32 exact saved self-repeat comparisons. Nine CPU checks pass. The actual
packet comparison takes **0.50 s**, and original invalid-input, clearing,
downstream fault, sticky-state and reset/recovery checks pass in **11.00 s**,
both including setup. Actual-input paired median gains are only **0.91–1.35%**,
smaller than the synthetic sparse-fixture 1.97%. Keep the candidate test-only;
this does not justify a full-runtime rerun or establish trajectory/performance
acceptance. Production preparation remains unchanged.
[Initial fixtures](benchmarks/soil/2026-10-12-preparation-certificate-mask/SUMMARY.md);
[actual packets, fault checks and decision](benchmarks/soil/2026-10-12-preparation-certificate-corpus/SUMMARY.md).

**Explicit diagnostic frontier report (2026-10-12; S00):**
`make report-soil-frontier BASELINE=<reference-data-dir> INPUT=<candidate-data-dir>
SOIL_FRONTIER_OUTPUT=<new-dir>` now recomputes frame/GPU statistics and numerical
discrepancies from raw completed 50k RTX 4080 artifacts, with strict sequence,
scope, declared-fixture and aggregate checks. Source provenance remains a
separate audit; observed nondominance is not calibrated accuracy or a paired
speedup. The method-25/26 report has **no accepted point**: p95 frame times are
529.646/520.819 ms and measured MPM p95 work exceeds 411 ms. Method 26's machine
discrepancy is 5.17118e-4 against the unchanged 2e-5 bound; final slotwise particle
position discrepancy is 5.00679e-6 m. Mean MPM work outside Newton is already
47.415/52.796 ms/frame, not a predicted speedup from removing Newton. The native
S00 lane passes 18 CPU checks; ten malformed-artifact cases, duplicate inputs
and output-overwrite checks pass separately without GPU work. This is a
fixed-resolution numerical-agreement/performance matrix, **not a completed
physical-accuracy/resolution frontier or full S00 acceptance**.
[Report, scope and reproduction](benchmarks/soil/2026-10-12-soil-frontier/SUMMARY.md).

**Owned persistent coordinate frame (2026-10-12; S07):** added public
`MpmSoilFrame` and `MpmSoilFrameCheckpoint`. The detached owner coordinates
local recentering with an exact f64 world-offset update and serializes both in
a bounded, versioned, digest-protected envelope. Frame-aware energy/momentum
queries use exact reference-point subtraction rather than absolute f32 world
positions. Offset precision loss rejects before GPU work; framed restore permits
only the local grid-origin difference, retaining ordinary physical-config and
capacity checks. Plain checkpoint restore stays strict. Two new CPU checks,
two original codec regressions and eight GPU checks pass; the frame GPU lane
takes **0.43 s including setup**. This does not provide tile-local archive storage,
machine rebasing, durable filesystem commits, sleep/support closure or bounded
field streaming. Full S07 and 4 ms/60 FPS gates remain open.
[Ownership, format and verification evidence](benchmarks/soil/2026-10-12-soil-frame/SUMMARY.md).

**Rejected lagged preconditioner (2026-10-12; S06):** a test-only method-26
candidate reuses the inverse within one Newton solve and invalidates it after
a failed Krylov prefix. Jacobians, publication checks and work caps remain
unchanged. All eight current inputs pass physical/public-output gates and 128
exact self-repeats in **27.89 s including setup**. However, every paired median
regresses (**8.35–44.13%**); total Krylov work rises **475 → 654 iterations**
with unchanged Newton steps/trials. Rejected, not promoted, and no full-workload
rerun: fresh preconditioning remains production behavior. This is distinct
from the already implemented warm-start history experiment.
[Evidence and bounded reproduction](benchmarks/soil/2026-10-12-newton-lagged-preconditioner/SUMMARY.md).

**Detached lossless recenter (2026-10-12; S07):** checkpoint `recentered` and
live `recenter_soil_with` translate particle positions and grid origin together.
They reject f64 intermediate loss, f32 coordinate rounding/flush, changed
relative grid coordinates, and changed clamp endpoints under either fused or
separate arithmetic. Other particle/history words are untouched. Live recenter
captures submitted state, reuses solver buffers and invalidates prepared work
and observation caches; attached rigids, installed global soil IDs and swept
guards reject rather than detach from their owners. Four CPU checks cover
145,125 integer-oracle coordinates and 125 boundary configurations. Two GPU
checks pass all four grid modes in **0.45 s including setup**, and four existing
checkpoint regressions pass in **0.96 s**. This is a blocking detached-solver
primitive, **not general large-world streaming**: canonical tile-local storage,
durable frame metadata, archive/machine rebasing, physical sleep and bounded
traversal remain open. It rejects rather than approximates unsupported shifts;
no trajectory/performance acceptance or 4 ms/60 FPS gain follows.
[Implementation and reproduction](benchmarks/soil/2026-10-12-soil-recenter/SUMMARY.md).

**APIC and elastic energy accounting (2026-10-12; S08):** added public
checkpoint `energy_at_height` / `MpmSoilEnergy`, reporting translational and
quadratic-APIC affine kinetic energy, gravitational potential and fixed-corotated
elastic potential with actual particle masses, reference volumes, material IDs
and frozen hardening. Evaluation is CPU-only over an existing checkpoint, with
no new GPU work or per-particle scratch. Elastic/total mechanical energy is
explicitly unavailable for inverted/degenerate elastic F or stretches outside
the shader's stress-clamp domain; these are not silently treated as zero.

Five CPU checks pass, including 129-particle independent quadrature and 54
finite-difference PK1 derivative checks across all three rheologies and positive/
negative Poisson ratios. Offline checks pass on **all eight saved frames** from
the two all-contact native runs. Previously omitted affine energy is
**0.07068–0.07927 J**, with elastic energy **1.58e-6–2.92e-5 J**. Corrected
machine-plus-soil energy is **0.38124–0.38539 J**, below the initial
**0.56250 J** plus the unchanged **1e-4 J** allowance.

The future native runtime fixture uses this stronger gate through its existing
checkpoint capture; it compiles, and its recorded inputs pass offline without
an unchanged GPU rerun. All 1,766 source hashes, executable, 30 input artifacts,
formatting and facade checks pass. **This is an unforced initial-energy bound,
not a complete plastic/contact/drive-work ledger, physical calibration,
gravity-loaded full50k or 4 ms/60 FPS acceptance.**
[Energy implementation and evidence](benchmarks/soil/2026-10-12-soil-energy/SUMMARY.md).

**External-prediction mode pruning (2026-10-12; S06/S08):** the native world's
primary paired Featherstone shader now removes mode-0 acceleration branches
that its CPU dispatch guard already rejects. Drift, refresh and displacement
remain supported; ordinary paired force prediction and the specialized drift
shader are unchanged. Both primary and specialized pipelines are still required.
No new allocation, binding, dispatch, buffer or numerical relaxation is added.

Source specialization is identical for reachable modes 1/2/3. A direct GPU
reference/candidate check passes **30,420 exact state words** across 65 independent
free bodies, six alternating modes, inactive articulations, compensation and
forbidden-ABA rejection, in **85.68 s including setup**. The all-contact native
soil fixture also passes; combined machine/APIC angular residual is at most
**4.88e-8 < 2e-5**. Source/executable/artifact and facade checks pass.

**No accepted performance or full-trajectory gain:** the native fixture's
11.65 s wall time benefits from warm unchanged pipelines; it must not be called
a 66.55-to-11.65 s optimization gain. Changed-primary construction is separately
observed at 9.96 s versus prior 11.51 s, not a paired timing result. Independent
atomic-P2G runs are not byte-identical; the failed exact comparison is retained.
No full50k rerun or 4 ms/60 FPS acceptance is claimed.
[Pruning, exact mode coverage and limits](benchmarks/soil/2026-10-12-external-mode-pruning/SUMMARY.md).

**Loaded native-machine angular balance (2026-10-12; S08):** the supported
native prediction/method-26 path now verifies angular reaction balance with
13 soil particles at each of four wheels and the tine. All five bound shapes
receive nonzero impulses. Four frames plus fault/rollback/recovery pass in
**66.55 s including setup**, under the unchanged 90 s cap. Maximum machine
angular residual is **4.22e-9 < 2e-5**. The 256-contact fixture capacity is
declared before execution and CPU-checked against its disjoint initial stencils;
Newton64 and numerical tolerances are unchanged. This is a distinct native
runtime fixture, not a retry of the timed-out generic constructor.

Added public checkpoint `momenta_about(origin)` / `MpmSoilMomenta`, summing
actual particle mass, linear momentum and quadratic-APIC angular momentum in
f64 without new GPU work. Three CPU tests include independent 129-particle
grid quadrature, spin versus symmetric strain, origin covariance and invalid
input/overflow checks. A separate CPU-only check of all four saved native
checkpoints passes **combined machine + soil angular balance**, maximum
**4.41e-8 < 2e-5**. Omitting APIC affine spin would report errors of
**0.00632–0.00689**. Fifteen captured artifacts are retained for auditing.

**Scope:** short-run zero-gravity, 65-particle analytic-shape integration,
physically uncalibrated. This does not close full50k angular/trajectory,
work/energy, gravity-loaded traction, rendered actual-asset or 4 ms/60 FPS
acceptance. The generic construction timeout remains recorded below.
[Native fixture, APIC accounting and evidence](benchmarks/soil/2026-10-12-native-angular/SUMMARY.md).

**Fixed-origin reaction impulses (2026-10-12; S08 accounting):** optional frame
loads now accumulate both existing instantaneous-COM torque and torque about
world zero, using each substep's contact-time COM before drift. The existing
dispatch binds resident centers; no new dispatch, buffer or hot host readback
is added. Double-buffered output plus staging grows from `96*N+16` to
`144*N+16` bytes, with the coupling budget updated. Benchmark frame-load JSONL
includes `world_origin_angular_impulse_nm_s`; it remains a soil-reaction
observation, not total external impulse, work or a conservation verdict.

Four GPU observer/bridge checks pass in **0.67 s including setup**, covering
changing centers, 65 bodies, independent f64 sums, translation covariance,
nonzero local COM, readback rejection and lifecycle/rollback. Nineteen benchmark
CPU contracts pass. A new `2e-5` six-body angular-impulse gate is implemented,
but its generic-machine fixture **times out during construction at the unchanged
90 s cap**, before any frame or angular residual. No unchanged retry or raised
cap was used. **Full-machine angular acceptance is still open**, as are loaded
wheels, combined soil APIC angular balance, energy/work and 4 ms/60 FPS gates.
[Implementation, bounded failure and provenance](benchmarks/soil/2026-10-12-origin-impulses/SUMMARY.md).

**Retained slot-constant cache (2026-10-12; not adopted):** the current Newton
hot loops already skip inactive slots. A new test-only prototype instead caches
immutable nonlinear contact constants in the unused retained index-buffer tail,
with a checked `13 * active_count <= capacity` admission and original-computation
fallback. It adds no buffers/bindings/private or workgroup storage. All eight
current packets are eligible (219–283 active contacts in 4,096 slots), and all
five public outputs plus 128 self-repeats remain bit-exact; independent physical
checks pass in **49.10 s including setup**. However, five paired gains are only
**0.10–0.23%** and three cases regress **0.52–0.57%**. **No runtime promotion**;
do not repeat unchanged. Raw timing transitions, endpoint contention, source,
executable and 112 capture audits are retained. Legacy/lifecycle/fallback GPU
acceptance and actual-workload testing were not pursued for this marginal
candidate. The 4 ms/60 FPS and full-plan gates remain open.
[Prototype, results and rejection](benchmarks/soil/2026-10-12-newton-slot-constants/SUMMARY.md).

**Swept motion guard (2026-10-12; S07 support contract):** swept-region owners
now check actual particle displacement in every G2P substep, rather than trusting
the travel bound only as a caller promise. A lazily prepared gather variant uses
one retained 16-byte uniform and no additional dispatch or per-particle buffer;
raw/default stepping and the original 96-byte replay uniform stay unchanged.
Bounds change only with committed residency transitions. Exceeding a bound
latches a fault before that particle's state write; later capture rejects and
faults the owner without publishing invalid archive state. Other particles may
already have advanced: **this is not whole-substep rollback**. Admission is
conservative, requiring rounding slack and rejecting changing subnormal positions
or potentially overflowing coordinate subtraction.

Twelve CPU checks pass. Three GPU checks pass in **0.64 s**, including 4,103
independent admission cases, dense/active/compact stepping and both owner fault
paths; nineteen owner/settling regressions pass in **3.48 s**. Original failed
fixture-capacity evidence is retained separately from the corrected/final source
captures. Source/executable, formatting and facade checks pass. **This closes the
unchecked travel-bound premise, not physical sleep, support-equilibrium,
long-traverse, trajectory or performance acceptance.**
[Guard contract and evidence](benchmarks/soil/2026-10-12-swept-motion-guard/SUMMARY.md).

**Tile-local activity (2026-10-12; S07 prerequisite):** both live world owners
now publish coordinate-sorted tile activity at existing opt-in capture boundaries,
alongside unchanged population metrics. Reports track previous/current occupancy,
entries/exits and the union of global particle IDs at either endpoint. A departing
clod remains visible in its source tile; swapping particles cannot appear as
unchanged membership merely because occupancy is constant. Each migrating ID
contributes once to each endpoint tile, so tile energies/counts are explicitly
non-additive. Intermediate crossed tiles and within-interval transients are not
observed. A shared reducer computes particle measurements once, with O(T) scratch,
O(N log T) work and at most 2N endpoint tiles; no additional GPU copies occur.
Ten CPU checks and eighteen actual RTX 4080 owner/settling regressions pass, the
latter in **3.88 s including setup** under the unchanged 90 s cap. Frozen source,
executable, formatting and facade checks pass. **No physical sleep, pin removal,
bounded-traverse acceptance or frame-time improvement is claimed.** Monitoring
remains off by default; force-equilibrium and active/dormant support coupling are
still required before using these observations to deactivate soil.
[Implementation and verification](benchmarks/soil/2026-10-12-tile-activity/SUMMARY.md).

**Streaming contact Jacobians (2026-10-12; runtime integration):**
Removed the returned whole-Block temporary in favor of direct Jacobian stores,
preserving arithmetic, the lazy gate, implicit zeros and fault clearing.
Eight current tractor packets and eight legacy difficult packets retain all five
public buffers bit-for-bit and 256 exact self-repeats. Paired median solve
reductions span **2.18–5.10%**; both bounded lanes pass (47.93 s / 52.22 s
including setup). Thirty-two independent poisoned/full/lazy/failure lifecycle
cases pass in 1.72 s. These are solver component results, not full-frame gains.

Added experimental `stream-block-newton` atop shared core, with capture tag 26,
live backend identification and benchmark metadata; defaults remain unchanged.
Coupled runtime verification passes in **73.72 s**; soil/recovery captures match
method 25 exactly. The actual 50k diagnostic completes 33 frames with zero hot
buffer allocations, but median/p95/max is **389.627 / 520.819 / 585.913 ms**:
all 30 measured frames miss 60 Hz. Mean Newton work is **228.579 ms/frame**.
This is **not an end-to-end speedup** over the prior separate 382.324 ms run.
Separate-run trajectory difference reaches **0.0005171 > 2e-5**; causality is
not isolated and trajectory acceptance remains open. Source/executable,
component and actual artifact audits pass. Full-plan acceptance is incomplete.
[Component and lifecycle evidence](benchmarks/soil/2026-10-12-newton-stream-block/SUMMARY.md),
[runtime integration](benchmarks/soil/2026-10-12-stream-block-runtime/SUMMARY.md).

**Mobility cache follow-up (2026-10-12; not adopted):** a keyed per-invocation
cache avoids repeating fixed contact-mobility assembly in both inverse builders.
Eight current packets remain bit-exact and pass in 51.35 s, but six paired gains
are only 0.69–0.91% and two cases regress slightly. The candidate stays test-only;
no legacy or actual rerun is warranted. Runtime and latest actual timing remain
unchanged. Input audit finds 219–283 active contacts, not 4096 active contacts.
[Cache implementation and negative admission evidence](benchmarks/soil/2026-10-12-newton-mobility-cache/SUMMARY.md).

Refreshed offline compilation statistics for current method 26: **209 registers,
1,381,248 executable bytes, 24,512 shared bytes**, with no simulation dispatch.
The suspicious raw local-memory value is preserved, not interpreted as spills.
This guides investigation of a smaller common-path shader with state-preserving
handoff, not a claimed bottleneck or accepted speedup; cold restart must not
silently exceed existing work limits.
[Current resource evidence](benchmarks/soil/2026-10-12-newton-resource-snapshot/SUMMARY.md).

**Common-path handoff experiment (2026-10-12; not adopted):** implemented two
pipelines with an exact-state fallback rather than a cold restart. Six completed
inputs preserve all public bytes and 96 self-repeats, including damping-base
handoff after 12 accepted steps. Their paired medians regress **8.46–18.29%**;
the eight-input batch also hits its unchanged 90 s cap during seventh-input setup.
The full batch therefore fails, and linear-tail handoff remains unverified.
No retry, runtime promotion or actual rerun followed. Offline common-shader
statistics shrink to 579,456 executable bytes / 128 registers, but reported
shared use rises to 47,968 bytes. This does not demonstrate a useful performance
gain or a cause for the regression. Latest runtime/actual evidence is unchanged.
[Partial exactness, failed batch and resource tradeoff](benchmarks/soil/2026-10-12-newton-common-handoff/SUMMARY.md).

**Ordinary-solve candidates (2026-10-12; not adopted):**
Tested a faster smoothing reduction without changing physical certificates or
iteration budgets. The first current packet passes both physical certificates,
but public difference is **0.008772 > 2e-5**; the candidate is rejected, with no
legacy or actual rerun. Physical certification alone does not prove parity.

Separately implemented a residual-value helper that omits unused derivative
outputs while preserving the original arithmetic prefix and smoothing policy.
Eight current packets pass exact public buffers and 128 exact self-repeats in
**53.89 s including setup**, with maximum physical residual **8.043e-7 < 1e-6**.
Paired reductions are only **0.21–0.63%**, insufficient to justify integration;
the legacy lane remains unrun. Runtime and latest actual median **382.324 ms**
are unchanged. [Rejected continuation](benchmarks/soil/2026-10-12-newton-fast-continuation/SUMMARY.md),
[exact residual helper and small timing result](benchmarks/soil/2026-10-12-newton-residual-value/SUMMARY.md).

**Actual Newton work attribution (2026-10-12; instrumented diagnostic):**
Added bounded, test-only per-substep report tracing and eight selected current
input captures. The corrected 50k/33-frame diagnostic passes in **78.88 s
including setup**, collecting 1122 successful reports. Its initial missing-
timestamp-feature failure is retained. All 19 corrected output artifacts and
source/executable hashes pass audit; instrumentation overhead is explicit.

Of 1020 measured solves, **990 take 7–14 Newton steps and consume 92.87% of
observed solver time**; 9–12-step solves consume 71.60%. This shifts the next
optimization toward ordinary repeated evaluation/linear work, not only rare
stalled-release paths. Eight current cold-input packets are available alongside
the older regression corpus. These counters do not isolate internal phase
latencies, and instrumented timings are not acceptance evidence. Latest
uninstrumented actual median remains **382.324 ms**; full-plan gates stay open.
[Bounded trace, current packets and distribution evidence](benchmarks/soil/2026-10-12-newton-actual-work/SUMMARY.md).

**Shared conditioned core runtime (2026-10-12; experimental opt-in):**
Integrated `shared-conditioned-core-newton` with exclusive scratch-transform
selection, capture tag 25 and live prepared-backend checks. Independent capacity
and nonfinite-cache fallback probes pass 96 invocations in **2.77 s**; an exit-
recorder error is preserved with an explicit transcript/hash audit, not a fabricated
exit code or GPU rerun. Coupled physical/profile/rollback checks pass in **63.53 s**,
and captured soil states match method 24 exactly.

The 50k diagnostic completes **33 frames / 0.55 simulated seconds**, with zero
hot GPU buffer allocations and unchanged dispatch/transfer counts. Median/p95/max
is **382.324 / 529.646 / 542.611 ms**; all 30 measured frames miss 60 Hz. Mean
Newton work is **231.202 ms/frame**. This does **not establish an end-to-end
speedup** over the previous separate 378.659 ms run. The separate-run trajectory
comparison first exceeds **2e-5** at frame 13 and reaches **0.034351** maximum
difference; causality is not isolated and parity acceptance remains open.
Further work must target representative actual solve costs. **Full-plan
performance, physical, visual, residency and calibration gates remain open.**
[Runtime, fallback checks and complete diagnostic evidence](benchmarks/soil/2026-10-12-shared-core-runtime/SUMMARY.md).

**Shared conditioned core (2026-10-12; verified component, test-only):**
Implemented one common paired matrix per conditioned release, then removes each
chosen contact's contribution rather than rebuilding the matrix for every
candidate. Reuses dead Krylov `.v` storage, invalidates on local-cache refresh,
and retains original assembly when scratch capacity is insufficient. Independent
GPU assembly/solve/recovery and scratch checks pass in **1.65 s**. Eight captured
solves pass original physical/public limits and 128 exact self-repeats in
**51.81 s including setup**; maximum public difference is **2.16e-12 < 2e-5**.
Five paired solves improve **4.28–8.43% over the current projection cache**;
three regress **0.15–0.28%**. Runtime integration and a dedicated nonfinite-cache
fallback probe remain next. No actual tractor rerun was made; latest full-frame
median remains **378.659 ms**, and full-plan acceptance is incomplete.
[Shared-core implementation and paired evidence](benchmarks/soil/2026-10-12-newton-shared-core/SUMMARY.md).

**Conditioned projection cache runtime (2026-10-12; experimental opt-in):**
Integrated the verified cache through `conditioned-projection-cache-newton`,
with capture method tag 24 and live prepared-backend inspection across both
coupling parities. Defaults, resource capacities and numerical budgets remain
unchanged. Coupled physical/profile/rollback checks pass in **65.25 s including
setup**; four soil states and the recovered soil checkpoint match the prior
native/lazy backend exactly. These are not full machine-trajectory comparisons.

The actual 50k diagnostic completes **33 frames / 0.55 simulated seconds** under
180 s with zero hot GPU buffer allocations. Median/p95/max is **378.659 /
442.748 / 596.735 ms**, and all 30 measured 60 Hz deadlines are missed. Mean
Newton work remains **228.238 ms/frame**. Separate-run timings do not establish
a causal whole-frame gain. The separate atomic-run trajectory comparison first
exceeds **2e-5** at frame 13 and reaches **0.068924** maximum difference; its
causality is not isolated and parity acceptance remains open. **4 ms/60 FPS,
trajectory, rendered terrain and calibration acceptance are still unmet.**
[Runtime integration and complete diagnostic evidence](benchmarks/soil/2026-10-12-conditioned-cache-runtime/SUMMARY.md).

**Conditioned projection cache (2026-10-12; verified component, test-only):**
Implemented one paired projection per active contact/machine row instead of
recomputing it for every conditioned matrix column. Reuses dead Krylov `.v`
slots without new bindings, buffers, shared storage or larger budgets; widths
above available iteration capacity retain the original loop. Independent GPU
assembly/solve/recovery and scratch-isolation checks pass in **1.13 s**.
Eight retained full solves pass exact public buffers and 128 exact repeats in
**50.75 s including setup**, with maximum physical residual **5.638e-7 < 1e-6**.
Five paired solve timings improve **3.37–5.43%**; three show small regressions
of **0.09–0.21%**. This supports runtime integration next, not full-frame
acceptance. Runtime selection is unchanged and latest actual median remains
**388.416 ms**. [Frozen numerical and timing evidence](benchmarks/soil/2026-10-12-newton-conditioned-cache/SUMMARY.md).

**Newton fixed machine width (2026-10-12; test-only, not adopted):**
Implemented compile-time DOF-width specialization after the existing protected
arithmetic and execution-lane transforms. Source contracts compile widths
1/3/10/32 without altering contact capacities, synchronization or numerical
budgets. Eight retained actual solves pass exact cross-variant public buffers,
128 exact self-repeat checks and maximum physical residual **5.638e-7 < 1e-6**
in **49.55 s including setup**. Paired timing changes are only **-0.14% to
+0.25%**, which do not establish a useful gain. This candidate is not promoted
and no new full-tractor run follows; latest actual median remains **388.416 ms**.
[Frozen source, numerical and paired timing evidence](benchmarks/soil/2026-10-12-newton-fixed-width/SUMMARY.md).

**Composed native/lazy drift runtime (2026-10-12; experimental opt-in):**
Added `native-lazy-geometry-drift` with a distinct backend/cache identity,
unchanged paired/storage/unrestricted eligibility and f64 device admission.
Existing native/lazy feature precedence is unchanged without this explicit
feature. Solver inspection, coupled tests and the benchmark record the prepared
`native-geometry-lazy-anchors` backend rather than inferring it from build flags.

Public runtime/device checks and native tractor/65-particle physical, profile and
rollback/retry checks pass in **71.97 / 64.56 s including setup**. Original
impulse/momentum/energy checks remain intact. Four captured soil states plus the
recovered soil checkpoint match the previous native backend exactly; these are
soil captures, not full machine-trajectory proof.

The actual 50k diagnostic completes **33 frames / 0.55 simulated seconds** under
180 s. Median/p95/max is **388.416 / 564.451 / 567.407 ms**, with all 30 measured
60 Hz deadlines missed and zero hot GPU buffer allocations. Mean machine drift
is **55.726 ms/frame**, total machine work **100.315 ms**, MPM **287.624 ms** and
Newton solving **238.553 ms**. Dispatch/transfer counts remain **51030 / 503160
upload bytes / 26520 readback bytes**; profile readback is **359520 bytes** outside
timing. The earlier paired component supports its 8.98% drift reduction; separate
full runs do not establish a whole-frame speedup.

The separate atomic-run machine trajectory comparison first exceeds **2e-5** at
frame 13 and reaches **0.051573** maximum component difference. It does not isolate
backend causality and does not pass parity acceptance. **4 ms/60 FPS, full
trajectory, rendered terrain and physical calibration remain open.**
[Runtime, complete diagnostic and preserved trajectory discrepancy](benchmarks/soil/2026-10-12-native-lazy-runtime/SUMMARY.md).

**Native geometry plus lazy anchors (2026-10-12; verified component, test-only):**
Implemented a composed drift pass that retains native geometry arithmetic while
avoiding unused body-velocity anchor construction. The existing runtime still
selects native geometry or lazy anchors separately; no default or feature
selection behavior changes in this checkpoint.

All ten moving tractor checkpoints match the native control bit-for-bit across
250 words. Nine alternating measured pairs after warmup give **5.122464 ms
control / 4.662432 ms candidate median**, **8.98% lower drift cost**. A separate
48-case joint/reversal/body-state reconstruction and lifecycle check also matches
exactly; maximum independent f64 body-velocity error is **2.354e-14** in both
arms. Runs take **47.81 s / 46.86 s including setup**, each below the unchanged
90 s cap. Source/executable and all 116 state captures are audited.

This supports explicit opt-in runtime integration next, not a whole-frame speedup
or acceptance claim. Latest actual full-frame median remains **382.179 ms**;
4 ms/60 FPS, coupled trajectory, rendered terrain and calibration gates remain
open. [Composition, exact state and timing evidence](benchmarks/soil/2026-10-12-native-lazy-geometry/SUMMARY.md).

**Guarded unit-division candidate (2026-10-12; rejected, test-only):**
Implemented an isolated shortcut for exactly +/-1 paired denominators, retaining
the original arithmetic for other denominators and exceptional inputs. The first
two helper probes were invalid: a misplaced experiment marker bypassed protected
compilation and lane specialization. Fixed program construction and added a
fail-closed runtime prefix check with source regression coverage; the original
failures remain archived rather than being attributed to the arithmetic.

With corrected routing, the unchanged 1776-case helper passes in **0.24 s**.
Eight captured full solves pass physical residual **5.638e-7 < 1e-6**, exact
cross-variant public buffers and 128 exact self-repeats in **73.79 s including
setup**. However, all eight paired timings regress by **9.87–25.50%**. The
shortcut remains test-only; the compiler-routing safeguard is retained. No new
actual tractor run follows. Latest full-frame timing remains **382.179 ms
median**, and 4 ms/60 FPS acceptance remains open.
[Corrected routing, numerical and timing evidence](benchmarks/soil/2026-10-12-newton-unit-division/compiler-route/SUMMARY.md).

**Native Newton quotient candidate (2026-10-12; rejected, test-only):**
Implemented isolated native-f64 division with paired-f32 output in the otherwise
unchanged method-23 solver. Eight retained actual contact packets pass original
physical/public bounds and 128 exact self-repeat checks in **59.51 s including
setup**. Maximum physical/public errors are **5.638e-7 / 9.043e-11**, below
**1e-6 / 2e-5**. However, all eight paired timings regress by **16.94–42.59%**.
This candidate is not promoted and does not trigger another full tractor run.
Production scheduling/arithmetic and the latest **382.179 ms** full-frame median
remain unchanged. [Frozen source, complete timing and numerical evidence](benchmarks/soil/2026-10-12-newton-native-division/SUMMARY.md).

**Deferred native prediction drift (2026-10-12; opt-in runtime):**
The `deferred-native-drift` feature runs native rate prediction, soil projection,
then one final pose drift. Projection already consumes input geometry/Jacobians
and predicted rates, so the intermediate predicted pose reconstruction was
discarded work. Nonprojected worlds, ordinary public prediction and diagnostic
prediction-state probes retain the full predictor. Timestamp end points now
cover the rates-only graph correctly; final drift preserves copied-force host
mirror invalidation on both state parities. No timestep, tolerance, iteration
limit or physical law changes.

Source-frozen component and coupled world checks pass, including exact
state/rate/delta comparisons, synthetic corrected rates, inactive/fault handling,
momentum/energy, profile lifecycle and exact rollback/retry. The actual 50k run
completes **33 frames / 0.55 simulated seconds** under 180 s. Measured dispatches
fall to **51030** and uploads to **503160 bytes**, exactly one dispatch and
48 bytes removed per substep; hot GPU buffer allocations remain zero.

Full-frame median/p95/max is **382.179 / 492.442 / 536.435 ms**, still missing
all 30 measured 60 Hz deadlines. Mean machine GPU work is **101.242 ms/frame**
(prediction **9.755**, final drift **59.216**, momentum **29.857**, refresh
**2.414**); mean MPM is **274.823 ms**, including **227.748 ms Newton solving**.
These separate full runs do not establish paired trajectory or whole-frame
speedup acceptance. **4 ms/60 FPS, full trajectory, rendered terrain and physical
calibration remain open.** Newton solving remains the largest measured cost.
[Scheduling, exact state checks and actual workload evidence](benchmarks/soil/2026-10-12-deferred-native-drift/SUMMARY.md).

**Parallel candidate indexing (2026-10-12; opt-in runtime):**
Connected the existing parallel stable radix sorter to candidate world/physical
indexing through `parallel-candidate-index`. Retained scratch grows by 1024 bytes
per index (2048 for two coupling parity owners); dense/default behavior stays
unchanged. Capacity/device preflight and runtime budget accounting include the
extra storage, and the benchmark records the actual live index backend.

Exact CPU stable-order comparisons cover dense/compact indices, large world IDs,
tile permutations, duplicate-key stability, partial/empty capacity and sticky
fault/reset handling. In alternating matched-input GPU measurements, index-build
medians fall from **1.194–1.196 ms to 0.108–0.110 ms**, about **11× faster**, at
65536-cell capacity with 1024, 128 or 3 active tiles. This is index-stage evidence,
not a whole-solver speedup. Existing candidate publication integration passes in
**1.32 s**; native tractor/65-particle momentum/energy, profile lifecycle and exact
rollback/retry checks pass in **62.35 s**, both including GPU setup.

The instrumented actual 50k workload completes all **33 frames / 0.55 simulated
seconds** under the unchanged 180 s cap. Mean candidate construction is now
**5.787 ms/frame**, versus 42.093 ms in the earlier separate instrumented run;
mean total preparation is **37.895 ms**, Newton solving **232.924 ms**, MPM
**281.535 ms**, and machine work **160.608 ms**. Separate full runs are not paired
trajectory/speedup evidence. Full-frame median/p95/max remains
**442.773 / 598.151 / 599.339 ms**: all 30 measured frames miss 60 Hz. No hot GPU
buffers are allocated; the expected extra four index dispatches per substep
increase measured dispatches to **52050**, without changing upload/readback
bytes. **4 ms/60 FPS, full trajectory, rendered terrain and calibration gates
remain open.** The largest remaining measured costs are Newton solving and
machine prediction/drift, not candidate sorting.
[Source-frozen checks, raw paired timings and actual workload audits](benchmarks/soil/2026-10-12-parallel-candidate-index/SUMMARY.md).

**Actual preparation attribution (2026-10-12; explicit instrumentation):**
Added retained, optional eight-phase Newton preparation profiling, with
parent-scope preservation, strict nesting/order validation, bounded query
capacity and exact readback accounting. Existing APIs retain their ordinary
profile layout; `--profile-preparation 1` / Make
`SOIL_TRACTOR_BENCH_PREPARATION_PROFILE=1` opts in. Integrated enable/disable,
unread/capacity rejection, zero-hot-allocation and rollback/retry checks pass
in **65.71 s including setup**, with original physical gates unchanged.

The instrumented actual 50k run completes 33 frames under the unchanged cap.
Mean preparation is **72.274 ms/frame**, split chiefly into **42.093 ms
candidate construction**, **26.745 ms point response/certification** and
**2.096 ms mass factorization**. Nested phases are not additive to parent
totals. Inspection finds compact candidate indexing still calls the serial
radix-sort preparation API for all 65536 cell identities each substep, despite
an existing parallel stable-sort API. Integrating that primitive with explicit
scratch budgets, ordering and physical checks is the next concrete candidate;
the measured candidate scope does not yet isolate sorting from classification.
Full-frame median/p95/max is **478.147 / 653.600 / 734.679 ms**; all 30 measured
frames miss 60 Hz, with zero hot buffer allocations. This detailed capture adds
timestamp work and has **359520 bytes** of measured profile readback outside
timing; it is not a paired speedup or zero-overhead claim. **4 ms/60 FPS, full
trajectory, rendered-terrain and calibration acceptance remain open.**
[Implementation, lifecycle checks and exact 33-frame attribution audit](benchmarks/soil/2026-10-12-preparation-attribution/SUMMARY.md).

**Further preparation candidates (2026-10-12; rejected, defaults unchanged):**
Implemented three test-only preparation variants without relaxing existing
refinement, physical or exact-output gates. A single native-f64 accumulator
fails exact response at the tractor cancellation fixture despite passing the
independent physical/oracle checks. Eight-lane workgroups and native paired
arithmetic pass all six exact/physical fixtures, but their representative
4096-slot/251-active-contact paired medians regress from **0.727040 ms** to
**1.445888 / 1.447936 ms**. All runs finish in 0.73–1.33 s including setup;
failed and passing raw artifacts are preserved. None is promoted and no
unchanged full tractor rerun is justified. Next separate actual preparation
geometry/factor/response costs instead of treating its aggregate as one kernel.
[Source, exact checks and paired evidence](benchmarks/soil/2026-10-12-native-preparation/SUMMARY.md).

**Native COM Jacobian sharing (2026-10-12; integrated and actual 50k):**
Native prediction now publishes COM Jacobians from the same ancestor columns
and geometry used for its Gram mass matrix. Projected soil constructs neither
legacy CRBA nor legacy Jacobian dispatches on this opt-in path; the five
wheel/tine rows use one 1200-byte device copy after prediction per substep.
Parity ownership, binding order, retained budgets and fault-preserved output
are explicit. All 132 reconstructed-state evaluations pass the independent CPU
Jacobian gate (worst **2.3655e-8 < 1e-6**), with the previous predictor gates
unchanged. The integrated 65-particle tractor test passes in **65.41 s including
setup**, including momentum/energy checks, zero hot allocations and exact
invalid-control rollback/retry. This is not full agricultural acceptance.

The updated actual **50k / 34-substep** run completes 33 frames under the
unchanged 180-second cap. Construction is **61.216 s**; measured wall
median/p95/max are **472.363 / 618.642 / 846.505 ms**, with **30/30 missed
60 Hz deadlines** and zero hot buffer allocations. Mean machine/MPM GPU work
is **160.702 / 321.560 ms**; Newton solve/preparation still cost
**238.476 / 72.469 ms**. The old 462.683 ms run is not a paired control, so
neither speedup nor regression attribution is established. Separate-run
trajectory comparison still exceeds **2e-5 at frame 13** (maximum **0.015720**
against the stored ABA route); it is not matched-input causal evidence.
**The 4 ms target, full trajectory acceptance and physical calibration remain
open.** Defaults are unchanged; renderer and terrain export are excluded.
[Implementation and frozen evidence](benchmarks/soil/2026-10-12-native-jacobians/SUMMARY.md).

**Native predictor physical correction (2026-10-12; matched state replay):**
Replaying all 33 published states from each 50k run with zero drive and actual
wheel effort exposed an incremental-force failure: **2.2765e-6 > 1e-6**.
Publishing/solving paired mass entries alone still failed (**2.1825e-6**).
The native operator also used a unit-quaternion shortcut for anchor/COM vector
transforms, whereas the CPU transform contract uses quaternion-vector
multiplication; rounded stored body quaternions are not exactly unit length.
Corrected those vector transforms while preserving inertia-matrix rotation,
and normalized admitted hinge/slider axes consistently with CPU subspaces.
The final **132 evaluations / 66 reconstructed states** pass unchanged gates:
incremental force **4.8664e-7**, absolute force **4.2388e-8**, bias **4.2381e-8**,
rate **9.8039e-10**, matrix **1.9140e-8**. The replay takes **0.75 s including
setup**. The mixed-tree regression, now including near-unit authored axes,
also passes physical, late-fault preservation and explicit shared-reset checks.
Native prediction now consumes paired mass, with the f32 high mirror still
shared with soil; retained graph/adapter budgets for six bodies and ten DOFs
are **2520 / 2792 bytes**. Scalar matrix APIs remain available.
This fixes an instantaneous numerical failure, not the complete frame-13
trajectory discrepancy. Replay reconstructs published frame states rather than
raw divergent contact substeps. The **462.683 ms** full-run measurement below
predates this correction; corrected-path full workload, contact parity and the
**4 ms target remain unverified/unmet**. Both failed precision experiments are
preserved; no gate was relaxed.
[Frozen inputs, failed controls and corrected physical replay](benchmarks/soil/2026-10-12-native-trajectory-replay/SUMMARY.md).

**Actual 50k native-prediction run (2026-10-12; diagnostic, not acceptance):**
The explicit `SoilTractor::new_native_prediction_with_newton_grid` factory and
`--native-machine-prediction 1` benchmark option now exercise the coupled native
predictor/shared mass. Defaults remain unchanged; native selection requires
generic paired Newton/shared-wall coupling and an f64-capable device. The
manifest verifies the live predictor and distinguishes its five compute stages
from the legacy two-stage ABA path. All 18 benchmark CPU contracts pass.
The unchanged 50k/compact:1024/method-23 workload completes 33 frames under the
180 s cap: construction **75.681 s**, measured wall median/p95/max
**462.683 / 595.416 / 625.398 ms**, with **zero hot GPU buffer allocations**.
All 30 measured frames miss 60 Hz. Mean machine/MPM GPU work is
**154.502 / 309.907 ms**; nested Newton solve/preparation remains
**228.711 / 70.533 ms**. Source/executable/artifact, fixture and profile audits
pass. These are separate runs from the previous 554.870 ms median and do not
establish a controlled speedup.
The separate-run trajectory comparison also exceeds the original **2e-5**
bound, first at frame 13 (**4.688e-4**) and reaching **0.06422**. Atomic P2G and
changed prediction/mass cannot be separated causally by this comparison; it
does **not** establish parity and the bound is not waived. Keep this path
opt-in. Next isolate the discrepancy with matched-input replay, then address
the dominant contact solve. The **4 ms target, full trajectory acceptance and
physical calibration remain open**; this run excludes rendering/terrain export.
[Full workload, identities, profiles and preserved trajectory discrepancy](benchmarks/soil/2026-10-12-native-tractor/SUMMARY.md).

**Native prediction in the coupled world (2026-10-12; opt-in runtime):**
`RigidWorld::from_scene_gpu_with_native_prediction` now owns both prepared
prediction parities at a fixed substep. Projected soil directly borrows their
input-configuration mass matrices; it allocates no duplicate matrix and creates
or dispatches no CRBA kernel on this path. Existing constructors are unchanged.
Prediction timestamps span control evaluation through paired drift across queue
submissions, rather than measuring drift alone. The actual six-body tractor
with **65 particles** and protected method-23 Newton coupling passes four
three-substep frames, zero application GPU buffer allocations and the existing
impulse/momentum/energy gates. Worst impulse and total momentum errors are
**2.45e-9 / 6.03e-7**, below **2e-5 / 2e-4**. An unsupported motor mode rolls
back soil/chassis exactly; restoring effort control commits the next frame.
Five checkpoints and finite timestamp intervals are retained. The test takes
**67.40 s including construction** under the unchanged 90 s cap. An earlier
fallback-backend construction timeout is preserved; the passing run explicitly
requests the protected shader feature used by the actual workload.
This small coupled fixture is not 50k performance, long-horizon trajectory,
model-rebuild acceptance or measured-soil calibration. Its three prediction/
drift substeps alone measure **17.35–17.37 ms**, not the full 1/60-second frame
or a controlled speedup. Wire this explicit path into the 50k benchmark next;
the latest full 50k result remains **554.870 ms median**, and the **4 ms** goal
is still unmet.
[Runtime, preserved timeout and coupled-state evidence](benchmarks/soil/2026-10-12-native-world/SUMMARY.md).

**Native controlled prediction and drift (2026-10-12; opt-in component):**
`PreparedNativePrediction` now evaluates admitted feed-forward/direct-effort
controls, combines COM loads, predicts paired rates and advances the existing
paired coordinate/body drift. A drift-only solver profile avoids compiling
unused ABA and rejects ordinary force stepping. Exact model snapshots and
retained buffer identities reject stale owners; unsupported control modes fail
with sticky status. Twenty six-body/ten-DOF steps plus post-fault recovery pass
the unchanged **2e-5** trajectory gate with worst error **1.3274e-7**, including
inactive-state preservation, effort saturation and copied-force ownership.
The bounded GPU test takes **37.69 s including construction**, not per step.
Each adapter step encodes five dispatches, uploads 48 uniform bytes, performs
no host readback and makes no application GPU buffer allocations; completed
backend buffer counts remain stable. The initial missing COPY_DST failure and
an unsettled backend-counter failure are preserved. The latter passes with
test-only completion barriers and unchanged count assertions.
Sharing the native mass operator with live soil coupling, world-level lifecycle
integration and full-scene trajectory/performance acceptance remain open.
The latest full 50k measurement remains **554.870 ms median**, not the 4 ms
target. The user reconfirmed no measured calibration data: initial soil remains
**physically uncalibrated**, independently of numerical verification.
[Implementation, failures and raw trajectory evidence](benchmarks/soil/2026-10-12-native-prediction-drift/SUMMARY.md).

**Generalized-rate prediction (2026-10-12; experimental graph):**
`PreparedGeneralizedPrediction` now connects GPU operators and acceleration to
checked paired/scalar rate and delta publication. A four-body/eight-DOF
free/hinge/slider/fixed tree with rotated anchors, anisotropic inertia, nonzero
COM offsets and loads passes eight changing configurations against full CPU
RNEA/CRBA: bias/force/mass residuals **2.59e-8 / 3.54e-8 / 2.53e-8 < 1e-6**,
rate error **3.18e-9 < 1e-5**. Late faults preserve all rate/delta outputs;
internal reset cannot clear shared solver status. The three retained dispatches
have zero hot buffer allocation or host transfer. Raw GPU timestamps measure
**0.140288 ms median** over seven post-warmup samples, scoped to this graph and
fixture—not a tractor step, controlled speedup or full-frame acceptance.
Control evaluation/admission, model mutation guards, actual drift, shared soil
operator ownership and lifecycle integration remain open. The 4 ms target is
unmet and initial soil remains physically uncalibrated.
[Implementation and timestamp evidence](benchmarks/soil/2026-10-12-generalized-rate-prediction/SUMMARY.md).

**Retained GPU mass/bias chain (2026-10-12; experimental):** the missing bias
producer is now implemented for an admitted dynamic fixed/hinge/slider/free tree,
with paired rates/position tails and gravity/external COM loads. Reusing legacy
f32 CRBA fails the original incremental force bound (**1.0236e-6 > 1e-6**).
A new f64 COM-Gram mass assembly in the same producer fixes this tested chain
without relaxing the bound: **7.8603e-7** incremental, **8.0194e-8** absolute and
**6.9998e-10** bias residual; rate error **3.0820e-9 < 1e-5**. Both GPU-produced
operators feed the retained solve in two allocation/transfer-free dispatches.
The **0.31 s setup-inclusive test** also passes sticky upstream-fault and partial/
full-reset checks. This is not a per-frame timing or production integration.
The frozen six-body case does not establish moving trajectory, general-tree,
control/constraint or lifecycle parity. Integration must share the new mass
operator with soil coupling rather than duplicate production CRBA. The original
CRBA and ABA failures remain preserved; the 4 ms target is still unmet.
[Implementation and preserved failures](benchmarks/soil/2026-10-12-retained-bias/SUMMARY.md).

**Retained mass predictor (2026-10-12; experimental algebra component):**
`ExperimentalMassPredictor` retains a width-1..32 dense f32 mass binding,
paired force/bias inputs and paired acceleration output. It factors in f64,
solves force and bias separately, certifies both solves and the published result,
and preserves the previous output on sticky faults. Dispatch has no host transfer
or GPU buffer allocation. RTX 4080 verification takes **0.43 s including setup**:
widths 1/32 exercise numerical and fault/reset behavior; the ten-DOF biased
fixture passes absolute/incremental force residuals **1.9333e-9 / 1.8127e-8**
against **1e-6**, and dt-scaled rate error **3.0734e-9** against **1e-5**.
This uses independently CPU-assembled M/bias, not live runtime operators.
The existing RNEA rejects free joints and allocates dispatch resources. The next
increment above supplies an experimental bias producer; control/constraint
semantics, operator epochs/rollback, trajectory parity and performance remain
open. Production prediction and its
older biased-increment failure are unchanged. **0.43 s is test duration, not
per-frame cost or a speedup.** No calibration data is available.
[Implementation and evidence](benchmarks/soil/2026-10-12-retained-mass-predictor/SUMMARY.md).

**Native paired sums (2026-10-12; geometry-only candidate measured):** unlike the
failed scalar-native experiment, the stronger paired-rate reference uses atomic
RMW reads and separately rounded TwoSum residual subtractions. A paired native
candidate preserving final scratch writeback now matches all **48 articulated
cases exactly**, with independent f64 error **2.354e-14** against **4e-12** and
unchanged fault/checkpoint/recovery checks. The comparison takes **77.68 s**
including setup under the existing 90 s cap. However, the full native moving
tractor comparison subsequently fails at step 2 in two residue words. A final
coordinate barrier and a position-only substitution both retain that failure:
`position_sum` also serves rate integration. Those variants remain rejected.
A geometry-only candidate preserves the original rate/coordinate arithmetic and
now passes ten exact moving six-body checkpoints. Nine alternating-order pairs
measure **6.930624 -> 5.268448 ms median (~24.0% lower)** for that drift pass,
within a 41.36 s setup-inclusive test. The opt-in `native-geometry-drift` now
integrates this variant only in unrestricted storage-rounded paired drift.
Its own 48-case exact/f64/lifecycle oracle passes in 45.87 s with six-slot
programs; two-slot cold construction hit the unchanged 90 s cap and remains
unaccepted. Public force/rate/corrected-position and lifecycle checks pass in
72.01 s including setup. The supplied device must enable `SHADER_F64`; missing
features are rejected before cached construction. Native geometry takes explicit
precedence if lazy drift is also enabled; the transformations are not combined.
Defaults and other machine phases stay unchanged. Its actual 50k run now
completes 33 frames within the unchanged 180 s cap, with **554.870 ms median /
693.398 ms p95**, 159.477 s construction and zero hot GPU allocations. Every
measured frame misses 60 Hz. Mean machine/MPM GPU work is **233.150 / 312.704 ms**;
nested contact solve remains **233.858 ms**. Source, build identity, full-frame
and profile audits pass. Separate earlier atomic runs do not establish causal
whole-frame gain or trajectory parity; the older scalar failure remains.
[Paired-only implementation and evidence](benchmarks/soil/2026-10-12-paired-native-sum/SUMMARY.md).
[Moving-state failures and geometry-only comparison](benchmarks/soil/2026-10-12-paired-native-drift/SUMMARY.md).
[Geometry-only runtime integration](benchmarks/soil/2026-10-12-native-geometry-runtime/SUMMARY.md).

**Acceleration extension (2026-10-12; test-only, unaccepted):** a separate
geometry-only native acceleration probe preserves the existing integration/drift
paths and compares full six-body state under ten force configurations. Source
composition/WGSL checks pass, but setup never reaches the first reference-program
completion marker within the unchanged 90 s component cap. The log does not
separately locate the delay in engine/fixture versus pipeline setup. No candidate dispatch or
cost measurement occurs. Runtime acceleration remains unchanged; neither drift
correctness nor its speed improvement is transferred to this phase.
[Preserved construction failure](benchmarks/soil/2026-10-12-native-geometry-acceleration/SUMMARY.md).

**Fixed-rigorous acceleration probe (2026-10-12; component-only):** specializing
the fixture's rigorous flag allows the reference to construct in 69.76 s, but
separate candidate compilation still exceeds the unchanged 90 s total cap. A
distinct shared-program probe fixes the same rigorous/zero-damping settings and
switches only geometry arithmetic, keeping isolated state buffers. It passes ten
exact full 250-word comparisons in **77.31 s** including setup. Nine alternating
pairs measure **4.032352 -> 3.137856 ms median (22.18% lower)** for isolated
acceleration. All raw-state/timing and integrity audits pass. This is not a moving
trajectory, independent ordinary-program parity, physical oracle or frame-speed
claim; production acceleration remains unchanged pending those gates.
[Specialization, shared-program evidence and limits](benchmarks/soil/2026-10-12-rigorous-acceleration/SUMMARY.md).

**Independent acceleration mass check (2026-10-12; failing gate):** an offline
f64 COM-Jacobian mass assembly and pivoted solve now checks the saved six-body
force-response differences without reusing ABA. Exact GPU-path agreement does
not suffice: both paths fail the unchanged **1e-6** normalized incremental-force
gate at **1.28508e-5**. The worst vertical-force residual is **0.00219749** versus
a conservative final-output rounding bound of **0.00158342**; quantization alone
is not a complete explanation. The dt-scaled rate error **5.60624e-8** passes
1e-5 but does not waive that failure. This rounded-input incremental check covers
one load direction, not absolute bias or full operator accuracy. Resolve oracle/
response-precision consistency before acceleration promotion; defaults remain unchanged.
[Independent oracle and preserved failure](benchmarks/soil/2026-10-12-acceleration-mass-oracle/SUMMARY.md).

**Absolute acceleration balance (2026-10-12; scoped pass):** a separate
independent Newton-Euler check now includes gravity, COM centripetal acceleration,
transported wheel spin and gyroscopic torque. Its initial failure exposed an
oracle input error: the frozen `ModelBuilder` fixture uses **9.80665**, not
9.81 m/s². With the correct rounded input, full-force residual is
**2.26212e-7** and dt-scaled rate error **6.24949e-8**, within the unchanged
1e-6/1e-5 bounds. The smaller-load incremental gate above still fails and is
not replaced by this absolute check. Ordinary-reference parity, broader force
directions and integrated trajectory/lifecycle acceptance remain open.
[Corrected oracle, regression guards and evidence](benchmarks/soil/2026-10-12-acceleration-balance/SUMMARY.md).

**Full-basis acceleration operator (2026-10-12; scoped pass):** the same six-body
machine now passes zero-bias response checks for a zero-force checkpoint and all
ten independent generalized-force directions. Eleven full 250-word records are
exact between storage/native paths in **77.54 s** including setup. Independent
f64 force residual is **7.52128e-7**, dt-scaled rate error **4.11523e-9**, and
reciprocity error **1.48818e-10**; response energy is positive. Ten alternating
pairs measure **3.978656 -> 3.122368 ms median (21.52% lower)**. This isolates
the response operator without replacing the still-failing biased sensitivity
case. It is not ordinary-program parity, moving/integrated or frame-speed
acceptance; production acceleration remains unchanged.
[Operator, physical oracle and artifact audits](benchmarks/soil/2026-10-12-acceleration-operator/SUMMARY.md).

**Lazy machine anchors (2026-10-12; opt-in unrestricted drift integrated):** paired
body-velocity reconstruction has a verified variant that computes only the
selected anchor rotation and origin rather than both discarded alternatives.
All 48 two-body cases preserve exact state and the unchanged independent 4e-12
f64 oracle, including reversed joints, eight joint kinds, boundary bodies and
fault recovery. A six-body translated moving tractor drift comparison retains
exact full compensated state for ten steps. Nine alternating-order timestamp
pairs after warmup measure **6.921152 -> 6.243872 ms** median (**9.8% lower**)
for this drift pass. This is not force-driven dynamics, soil coupling, whole-frame
speedup or 4 ms acceptance. The new `lazy-paired-drift` feature selects it only
in a separate storage-rounded unrestricted paired drift pipeline; defaults, force prediction,
momentum, refresh and position-displacement sources stay unchanged. The public
cached constructor passes existing analytic force/rate/position, corrected-drift,
fault/recovery and model-mutation checks in **69.83 s** including setup. Split
paired models still reject loss of unrestricted eligibility before mutation;
unsplit models retain generic/constrained fallback. The extra pipeline adds cold
construction work. Its own actual 50k run now completes 33 frames under the
unchanged 180 s cap, with **590.422 ms median / 725.574 ms p95** wall time and
zero hot GPU allocations. Construction is **138.259 s**; all 30 measured frames
miss 60 Hz. Mean machine/MPM GPU work is **276.655 / 314.936 ms**. This verifies
integration, not a whole-frame gain or controlled full-trajectory parity against
earlier separate atomic runs. Build identity, source integrity and full frame/
profile audits pass; the 4 ms/60 FPS target remains unmet.
[Implementation, failed setup and corrected evidence](benchmarks/soil/2026-10-12-machine-lazy-anchors/SUMMARY.md).
[Runtime integration](benchmarks/soil/2026-10-12-lazy-drift-runtime/SUMMARY.md).

**Sparse residual preparation (2026-10-12; coupled and actual opt-in verified):** added
`zero-residual-newton-preparation` without replacing defaults or the earlier
zero-term feature. It extends finite-guarded zero elision to two residual sites,
preserving scale accumulation, finite checks, correction counts and the 1e-6
certificate. Against the already-optimized zero-term control, paired sparse
4096-slot/251-active median improves **0.989184 to 0.727040 ms (~26.5%)**, with
exact responses/status and independent physical/oracle checks. Fault/clear/
sticky/reset rejection passes in 10.54 s. Dense/active/compact retained-local
Newton four-step feedback passes in 25.59 s with exact cross-layout records and
maximum normalized momentum error **5.914e-9**. Its separately identified actual
compact 50k tractor run completes all 33 frames under the unchanged 180 s cap,
with **581.432 ms median / 675.626 ms p95** wall time and zero hot GPU allocations.
All 30 measured frames miss 60 Hz. Mean machine GPU work is **269.211 ms**;
MPM is **310.222 ms**, including **232.216 ms** Newton and **67.347 ms** preparation.
Noncontact MPM alone is **7.426 ms**, already over the 4 ms soil budget. This
validates integration, not real-time acceptance or a causal whole-frame speedup:
separate public atomic runs are not paired trajectory/performance controls.
[Component comparison](benchmarks/soil/2026-10-12-preparation-sparse-residual/SUMMARY.md),
[opt-in feedback and failure evidence](benchmarks/soil/2026-10-12-sparse-residual-runtime/SUMMARY.md).

**Zero-term preparation (2026-10-12; explicit opt-in integrated):** added
`zero-term-newton-preparation`, which skips five factor/triangular compensated
product sites only when the coefficient is zero and the operand finite. Input
validation, two mandatory and up to six selective refinements, and the 1e-6
certificate remain unchanged. Six component fixtures preserve exact response/
status and independent physical/oracle checks. Sparse 4096-slot/251-active paired
medians improve **1.077248 to 0.917504 ms (~14.8%)**; the 32-DOF fixture improves
9.253888 to 5.711872 ms. Fault/clearing/sticky/reset and downstream rejection
contracts pass in 13.63 s. Defaults stay unchanged.

The feature-enabled compact 50k tractor completes all 33 frames under the
unchanged 180-second cap, with explicit preparation identity and zero hot GPU
buffer allocations. Wall median is **580.194 ms**; all 30 measured frames miss
60 Hz. Mean full contact-preparation GPU work is 70.171 ms, which includes other
stages beyond this optimized kernel. No matched whole-frame speedup, full
trajectory parity, calibration or release acceptance is claimed.
[Component gain](benchmarks/soil/2026-10-12-preparation-zero-terms/SUMMARY.md),
[opt-in lifecycle and actual workload](benchmarks/soil/2026-10-12-zero-terms-runtime/SUMMARY.md).

**Preparation optimization candidates (2026-10-12; not promoted):** implemented
test-only fixed-DOF-width and private-response-scratch variants. Both preserve
exact response/status and independent 1e-6 physical/oracle checks across six
fixtures, including 4096 slots / 251 active contacts, cancellation and empty
inputs. GPU comparisons finish in 2.36 / 2.37 s under the unchanged 90-second cap.
Neither improves relevant paired cost: sparse fixed-width medians are
1.076224 / 1.077248 ms, private-scratch medians 1.077248 / 1.084416 ms. Other
representative cases are essentially unchanged or slightly worse. Defaults and
physics remain unchanged; no full tractor rerun or speedup claim for either.
Next optimize computational work, not another width/scratch-only variation.
[Fixed-width evidence](benchmarks/soil/2026-10-12-preparation-fixed-width/SUMMARY.md),
[private-scratch evidence](benchmarks/soil/2026-10-12-preparation-private-response/SUMMARY.md).

**Compact cost attribution (2026-10-12):** fixed the benchmark's dense-only
seven-stage profile assumption. Exact layout/substep checks now accept eight
active stages plus first-frame initialization and reject mismatched metadata,
stage order and invalid durations; 17 CPU tests pass. One actual compact 50k
profile completes all 33 frames under the unchanged cap. Mean GPU work per
1/60 simulated second is **314.369 ms MPM + 273.411 ms machine**. Within MPM,
contact projection is 306.937 ms: nested Newton 230.991 ms, preparation 72.676 ms,
publication 3.271 ms. Non-contact MPM still averages **7.431 ms**. Do not add
nested stages or host completion waits to these GPU totals.

The measured wall median is 594.789 ms; all 30 measured frames miss 60 Hz, with
zero hot application GPU buffer allocations. This is attribution, not a matched
speedup, profiling-overhead or trajectory-parity comparison. Machine work is now
about 46.5% of measured machine-plus-MPM GPU work, so Newton-only optimization
cannot solve the full-frame deficit. Target both machine execution and coupling,
preserving all physical gates and existing failed-candidate evidence.
[Profile fix, full run and independent stage audit](benchmarks/soil/2026-10-12-compact-profile/SUMMARY.md).

**Actual compact 50k tractor (2026-10-12; short-run integration verified):** the
public native atomic P2G path with 1024 compact tiles, Generic paired machine and
opt-in retained-local Newton completes all 33 frames / 0.55 simulated seconds.
Construction takes 105.705 s; the run finishes under the unchanged 180-second cap.
All 30 measured frames have zero application GPU buffer allocations, but median
wall time is **566.731 ms**, p95 **675.602 ms** and all miss 60 Hz. The physical
grid/metadata occupies 3,694,608 bytes, excluding other owners. Final reaction
observations are nonzero at all four wheels and the tine; this does not establish
independent traction/energy/angular accuracy or rendered-rut acceptance.

The first 512-tile attempt correctly faults before any committed frame. A focused
probe identifies 320 soil-only tiles versus **544 with initial rigid support**;
the six-case regression passes in 2.01 s, checking overflow, frozen positions/
velocities and rejection of faulted checkpoint capture. Capacity is selected
explicitly, never auto-grown. Source/executable and offline full-frame audits pass.
No dense/compact trajectory parity or speedup is claimed: the historical ordered
gather is a different transfer path. Performance, long-horizon and calibration
gates remain open; no measured calibration data is available.
[Actual run, capacity failure and regression](benchmarks/soil/2026-10-12-compact-50k/SUMMARY.md).

**Tractor grid selection (2026-10-12; factory verified):** additive public
constructors now select dense, active or capacity-bounded compact soil. Existing
constructors remain dense. The benchmark accepts `--soil-grid compact:512`
(`SOIL_TRACTOR_BENCH_GRID=compact:512` through Make), requires projected coupling
for active/compact selection, and records requested and live storage separately.
Six portable/native layout cases pass exact seed/history and layout checks in
1.92 s, including compact growth and invalid-capacity rejection before allocation.
CPU configuration, all 16 CLI tests and the public facade check pass. This does
not verify full 50k compact stepping or achieve the 4 ms/60 FPS target. No measured
soil/traction data is available; initial results remain physically uncalibrated.
[Factory and CLI evidence](benchmarks/soil/2026-10-12-tractor-grid-selection/SUMMARY.md).

**Compact articulated friction (2026-10-12; opt-in scalar backend):** the actual
six-body tractor/tine fixture now uses 1024-tile native compact soil in both its
main and fixed-wall rejection cases. Its unchanged physical/lifecycle assertions
pass in 69.02 s including setup: consumed transverse impulse 0.151543704 N s,
zero application GPU buffer allocations after warmup, exact rollback/retry,
late-particle fault rejection, empty soil and model rebuild. Contact is on the
tine only; wheels report zero supporting nodes. This 65-particle check does not
establish loaded-wheel, rendered-rut, paired/Newton, 50k or 4 ms acceptance.
[Full articulated evidence and limits](benchmarks/soil/2026-10-12-compact-tractor-runtime/SUMMARY.md).

**Active attachment transaction evidence (2026-10-12; opt-in scalar backend):**
the existing `ExperimentalJointTypes` profile passes the actual one-body,
two-particle sticking matrix in 52.35 s including setup. Portable/native active/
compact cases transfer nonzero forward impulse; maximum total momentum error is
3.20e-7 against the unchanged 2e-5 bound. Unconfigured-step and budget rejection,
exact soil/scene rollback, retry, real compact overflow and grow/reattach recovery
pass. The separate Generic timeout below remains failed and was not rerun;
Generic remains default. This is not articulated tractor, unilateral/Newton,
cross-layout trajectory, angular/rendered or 4 ms/60 FPS acceptance.
[Actual interaction and transaction evidence](benchmarks/soil/2026-10-12-active-runtime-joint-types/SUMMARY.md).

**Active runtime wiring (2026-10-12; actual integration unverified):** active and
compact attachment now prepares kinematics-only bindings and rejects stepping
until projected coupling is explicitly enabled. A retained direct machine-fault
guard covers the no-projection branch; its portable/native overflow/rollback/
growth test passes in 1.52 s. Existing contact-fault regression passes in 8.39 s.
The actual RigidWorld transaction test times out during GPU machine construction
at the unchanged 90-second cap, before reaching any attachment assertions.
Earlier attachment guards below are historical; the new wiring is not accepted
integration evidence. Cold construction must be addressed before another run;
do not repeat the unchanged failure or extend its cap. The 4 ms gate remains open.
[Implementation, passing components and retained timeout](benchmarks/soil/2026-10-12-active-runtime-attachment/SUMMARY.md).

**Projected runtime capacity budgets (2026-10-12; attachment prerequisite):**
runtime preparation now shares physical-cell sticking and indexed-candidate
unilateral budgets with standalone owners. All six mode budgets remain identical
across 17³/96³/1536³ compact worlds at 125 tiles; dense identity, capacity growth
and arithmetic/dimension guards pass in 2.23 s. Actual sticking/unilateral owner
byte agreement and paired-wall feedback regressions pass separately. Newton is
budget-checked here, not execution-verified. Active RigidWorld attachment remains
guarded pending shared machine-fault integration, including disabled bindings.
[Implementation and bounded evidence](benchmarks/soil/2026-10-12-runtime-grid-budget/SUMMARY.md).

**Active rigid kinematics (2026-10-12; attachment prerequisite):** added a
kinematics-only resident bridge without force backups. Six native/portable by
dense/active/compact cases plus the dense regression pass in 2.43 s, checking
both input states, rotated COM/motion and unchanged histories, forces and wrench.
Stale-handle/source guards pass separately in 0.60 s. Explicit reactions remain
prohibited on this bridge. Actual active RigidWorld attachment and the 4 ms gate
remain open; next integration work is projected runtime budgeting and shared
machine-fault handling, not removing the guard alone.
[Implementation and retained failed/corrected evidence](benchmarks/soil/2026-10-12-active-kinematics/SUMMARY.md).

**Active frame rollback (2026-10-12; attachment prerequisite):** transaction
storage now includes active indirect dispatch alongside the full grid metadata
and histories. Begin/rollback validate full buffer-list lengths; rollback also
rejects in-frame step/rigid epoch changes. All five backup tests pass in 8.67 s,
including eight native/portable active/compact ordered/unordered combinations,
nonzero dispatch restoration, overflow, growth, edit rejection and dense controls.
Actual RigidWorld attachment is still guarded: restoring soil state alone does
not establish machine overflow handling or tractor acceptance.
[Implementation and bounded evidence](benchmarks/soil/2026-10-12-active-frame-backup/SUMMARY.md).

**Compact Newton component verified (2026-10-12; default features):** fixed the
comparison harness to keep one engine while creating independent solver owners,
preserving the existing immutable Newton program cache. The full dense/active/
compact comparison now passes in 54.72 s including cold setup under the unchanged
90 s cap, with bit-exact histories, rates, status and wrench. First program setup
is 52.311 s; subsequent cached setups report 0 ms. No physical state is shared.
Original timeout evidence remains. This does not verify the opt-in method-23
runtime feature, actual tractor attachment, full trajectories or the 4 ms budget.
[Shared-engine verification and next transaction work](benchmarks/soil/2026-10-12-shared-engine-unilateral/SUMMARY.md).

**Standalone active unilateral integration (2026-10-12; partial):** indexed
candidates now drive normal and paired-wall friction coupling with checked
physical-capacity budgets. Normal compact lifecycle passes in 3.48 s; paired
feedback passes in 4.15 s with bit-exact dense/active/compact histories, machine
rates, status and wrench after four substeps. Normal coupling-owned bytes remain
362,252 at 125 tiles across 17³/96³/1536³. The internally wired Newton comparison
initially hits its 90 s cap; the changed shared-engine run above subsequently
passes, with original failure evidence retained. Actual rigid-world attachment
still rejects active/compact soil, and the
4 ms and full-plan acceptance gates remain open.
[Implementation, passing components and Newton timeout](benchmarks/soil/2026-10-12-active-unilateral/SUMMARY.md).

**World-ordered active candidates (2026-10-12; internal dependency):** implemented
retained world/physical cell indexing and stable radix ordering for the
unilateral candidate path. Twelve dense/indexed comparisons pass bit-exact query,
world-ID and corrected-grid publication in 1.52 s, including changing tile order,
partial edge tiles, closing contact, empty state and grid-fault suppression.
The 27-tile fixture owns 69,584 candidate/index bytes; this is not a timing
improvement claim. The subsequent standalone integration above connects its
budgeting and scheduling; full-plan gates remain open.
[Index implementation and precise verification scope](benchmarks/soil/2026-10-12-indexed-candidates/SUMMARY.md).

**Active-grid finite-mass coupling (2026-10-12; experimental):**
`PreparedMpmMassCoupling` now accepts dense-backed active and compact explicit
grids. Gather/apply use indirect active dispatch; contact ownership uses physical
slots while arms and boundary normals use decoded world cells. Overflow is
propagated before the projected machine solve. The 29.38 s component gate passes
24 native/portable six-face cases, overflow publication rejection, all-history
freeze, stale-growth rejection/recovery, empty soil and independent mass/rate
checks. Coupling-owned storage stays 36,404 bytes with 125 tiles across
17³/96³/1536³ domains. An initial 64-tile recovery fixture correctly overflowed;
its conservative rigid influence needs 125 tiles, and the failure is retained.
Actual RigidWorld active/compact attachment remains unsupported. This does not
clear legacy-wrench repeatability, trajectory,
rendered-tractor or 4 ms acceptance.
[Implementation and bounded evidence](benchmarks/soil/2026-10-12-active-mass-coupling/SUMMARY.md).

**Pre-contact native transfer compensation (2026-10-12; test-only):** implemented
residual correction immediately after native single-pass P2G, before rigid
imprint reads mass/momentum. The earlier isolated prototype corrected during
grid update, too late for imprint reactions. The new dense/active/compact matrix
passes all 18 comparisons at unchanged absolute 2e-5 in 0.92 s. A separate
0.84 s lifecycle gate passes retained batches, independent mass/gravity,
residual clearing, poisoned-reset recovery, compact overflow/growth and stale
record/mode rejection, with zero hot GPU-buffer allocations or host transfers.
Requires SHADER_F64 plus native float atomics; combined P2G remains outside this
prototype. An initial fixture error assumed reseeding resets the grid epoch
counter; it is fixed using exact wrapping deltas, with failure evidence retained.
No public policy/default change, speedup or tractor acceptance is claimed.
Whole-step 50k-particle fixed-sphere proxy timing now rejects promotion as a
speed repair: corrected dense/active/compact medians are 12.060/12.981/16.751 ms
per 1/60 simulated second, versus ordinary 7.952/8.910/13.085 ms. All 18 measured
corrected frames miss 4 ms. This is not articulated-tractor acceptance; keep the
prototype test-only and require both accuracy and budget from any new policy.
[Paired whole-step timing and scope](benchmarks/soil/2026-10-12-native-precontact-cost/SUMMARY.md).
[Implementation, corrected fixture and bounded evidence](benchmarks/soil/2026-10-12-native-precontact-compensation/SUMMARY.md).

**Native force substitution and ordered repair (2026-10-12; unresolved):**
implemented five saved-input substitutions and a bounded, fixed-order per-body
shell-force reducer. Substitutions complete in 1.54 s with exact per-cell
dense/active terms: candidate mass and velocity both affect force, while
momentum-only changes do not. The captured force-x term-sum difference is
1.624972e-5 versus a 3.051758e-5 final-wrench difference. Nevertheless, the new
ordered reducer still fails the physical matrix in 0.97 s: force-y differs by
about 2.288818e-5 against the unchanged 2e-5 bound. Its pipeline/API are retained
only under `cfg(test)`; no runtime default or tolerance changes. Stop unchanged
summation-only variants and isolate upstream transfer/history differences from
the saved inputs. Native active and 4 ms acceptance remain open.
[Substitution provenance](benchmarks/soil/2026-10-11-native-force-substitution/SUMMARY.md),
[implemented reducer and failed gate](benchmarks/soil/2026-10-11-ordered-native-shell/SUMMARY.md).

**Native force mismatch localized further (2026-10-11; unresolved):** added
identical-grid shell-force replay and first-divergence capture. The former
produces exact per-cell force/torque terms across dense/active dispatch, with
eight paired final sums differing by at most 7.629395e-6. The instrumented full
comparison captures a 3.051758e-5 force discrepancy at substep 2, alongside
existing differences in post-step grid mass, momentum and velocity. It retains
both complete inputs instead of only a panic message. This rules out treating
the identical-grid shell result as proof of full-path parity; it does not yet
identify P2G, history or imprint as the cause. Keep the failed gate and reuse
the capture for pre-imprint/input substitution rather than another full run.
[Exact-term replay and captured failed boundary](benchmarks/soil/2026-10-11-native-force-replay/SUMMARY.md).

**Native active/compact MPM (2026-10-11; experimental, parity gate open):**
added native-float-atomic active dispatch over dense backing and
`MpmSimulation::new_native_sparse` over a fixed tile pool. The shared integer
tile map, overflow latch, reset/growth and prepared-step lifecycle are retained;
native implicit stepping and active/compact machine coupling remain rejected.
No default changes. GPU checks verify overflow freezes all particle/history
streams, recovery after growth, conserved mass, empty/reseed, and fixed 230,928
grid/lookup bytes for a 64-tile pool across 32³–1536³ domains. Dense-backed active
and compact paths each advance 128 steps / 0.5 m, retire original tiles, preserve
constant-velocity motion and perform zero hot application allocations/readbacks.
However, mixed-material rigid parity fails the unchanged 2e-5 force bound:
88.324974 versus 88.324950. A separate 16-case dense/dense control stays within
the bound (maximum 1.525879e-5), so the failure cannot be dismissed as a failed
reference gate. Three native dense regressions pass. No performance or tractor
acceptance follows; the new `make test-soil-native-active` lane joins S05 and is
known failing. Next isolate grid-input differences from rigid-force reduction
order, without rerunning the unchanged failed matrix or relaxing tolerance.
[Implementation, frozen failure and control](benchmarks/soil/2026-10-11-native-active/SUMMARY.md).

**Targeted direct-quality correction (2026-10-11; not promoted):** implemented
a test-only correction of globally accepted direct directions with high
contact-local relative residual, reusing cached factors and existing storage.
It retains the unchanged global guard, requires improved local quality, and
restores the original accepted high/low direction on correction rejection.
All 176 GPU lifecycle cases and nine captured nonlinear cases pass the frozen
physical/public limits and 144 exact own repeats. No capture reduces iteration
or trial counts; paired changes range from -0.082% to +4.223%, including a 3.217%
regression on the prior eager-benefit case. Do not promote or rerun the full
tractor for this candidate. Local relative quality alone did not predict less
nonlinear work; stop unchanged threshold/fallback variants. Real-time acceptance
remains open.
[Frozen implementation and bounded comparison](benchmarks/soil/2026-10-11-newton-direct-quality/targeted/SUMMARY.md).

**Retained direct-direction variants (2026-10-11; test-only):** extended the
retained-local formulation to direct Newton directions. Both eager validation
and a lazy, cached-factor fallback pass 112 GPU algebra/lifecycle cases each,
including singular local blocks in nonsingular coupled systems, maximum width,
multiple-invalid rejection, nonfinite input and empty active sets. Both variants
pass all nine captured cases, unchanged physical/public tolerances and exact
own repeats. Eager validation reduces one case's linear iterations 281→227 and
paired time 3.37%, but regresses others up to 2.97%. Lazy fallback preserves all
45 public-buffer files exactly but removes that improvement; paired changes
range from -0.006% to +0.496%. Neither is promoted or triggers a full-tractor
rerun. Investigate the quality of already-accepted direct directions rather
than assuming rejected direct solves explain the remaining cost.
[Frozen variants, independent oracles and timing limits](benchmarks/soil/2026-10-11-newton-retained-direct/SUMMARY.md).

**Dispatch-local publication validation (2026-10-11; not promoted):** a new
test-only method-23 variant validates immutable contact/row/mass inputs once per
dispatch while retaining every state-dependent and physical check. All 875
two-certificate lifecycle cases, state-poison/dispatch-reset checks and nine
captured-packet comparisons pass. The eight prior-packet median gains are only
0.001–0.450%; the retained-block packet's -2.518% ratio of medians includes timing
transitions, with a median paired change of about -0.210%. No runtime/default
change or actual-tractor rerun follows. Stop publication micro-optimizations;
the dominant solver work and full-plan acceptance remain open.
[Implementation and complete paired evidence](benchmarks/soil/2026-10-11-newton-publication-validation/SUMMARY.md).

**Adaptive publication tiles (2026-10-11; not promoted):** implemented a
test-only method-23 variant that fills existing scratch instead of using fixed
16-contact tiles, preserving accumulation order and all numerical limits.
All 965 lifecycle cases and nine captured packets pass exact five-buffer and
physical checks. Alternating timings show no material gain: eight prior-packet
median changes range from -0.514% to +0.126%. The retained-block packet's apparent
-1.73% ratio of medians includes large timing transitions; its median paired
change is only about -0.064%. No runtime/default change or actual-tractor rerun
is justified. Next optimization should target the dominant solver work rather
than publication tile barriers. The 4 ms requirement remains open.
[Source-frozen comparisons and retained failed invocation](benchmarks/soil/2026-10-11-newton-publication-adaptive/SUMMARY.md).

**Retained local-block runtime (2026-10-11; opt-in method 23):**
`retained-local-newton` now has distinct runtime/capture/replay, benchmark and
renderer identity. Forty retained-core lifecycle cases pass, including maximum
width and multiple-invalid fallback; runtime packet outputs match the prototype
exactly. Two independent actual **50k / 33-frame** ordered-transfer runs complete
under 180 seconds and match **all 3,267 machine words, full final soil checkpoint
and final particle positions exactly**. Both runs have zero hot GPU allocations.
This resolves the captured convergence failure on the bounded fixture, not the
full plan: observed median frame walls remain **601.56 / 588.19 ms**, every frame
misses 60 Hz, physical calibration is unavailable and defaults remain unchanged.
[Runtime integration, actual repeats and audited limits](benchmarks/soil/2026-10-11-retained-local-runtime/SUMMARY.md).

Use `make test-soil-newton-retained-local-runtime` for component gates, and
`make test-soil-ordered-tractor SOIL_ORDERED_NEWTON_FEATURE=retained-local-newton`
with a fresh `SOIL_ORDERED_TRACTOR_DIR` for the bounded diagnostic fixture.

**Retained local-block correction (2026-10-11; test-only):** the captured
ordered-tractor failure now solves from cold in **16 steps**, with independent
physical error **3.39e-7** and successful publication, versus the prior 64-step
failure. The existing eight-packet corpus also passes unchanged physical checks.
The correction keeps one inaccurate local block inside the small coupled system
instead of forming its unstable inverse; existing GPU storage and numerical
bounds remain. This is nine captured-case successes, not production integration,
whole-tractor admission or the 4 ms performance target. Defaults are unchanged.
[Cause, correction, frozen runs and remaining gates](benchmarks/soil/2026-10-11-local-factorization/SUMMARY.md).

**Ordered-tractor convergence diagnosis (2026-10-11):** bounded prefix probes
locate the remaining error at two opposed floor/machine contacts. Neither
test-only release revisitation nor non-monotone step acceptance resolves the
64-step failure; neither is promoted. At the archived final state, the isolated
GPU linear direction satisfies the unchanged 1e-5 relative bound and closely
matches an independent f64 direction. Both achieve only about 0.0122% residual
reduction over the same 16 trial step sizes. This directs further work toward
nonlinear branch transitions, not larger iteration budgets or relaxed physical
bounds. No production speed or full-trajectory acceptance is claimed.
[Captured controls and rejected changes](benchmarks/soil/2026-10-11-ordered-convergence/SUMMARY.md).

**Conditioned local-candidate isolation (2026-10-11):** the independent f64
implementation of the existing floor-release policy reduces the archived state's
physical error to **7.28e-9**. GPU inspection finds contact **122** alone has an
invalid local cache, preventing the conditioned release of contact **113**.
Broader branch candidates do not fix the cold GPU solve. One bounded local
residual-correction experiment also leaves the original failure unchanged; do
not promote either. Investigate the exact local factorization/validation rejection
next, rather than increasing global iterations. This is not GPU solution or
whole-trajectory acceptance.
[Independent pivot controls, GPU inspection and retained failures](benchmarks/soil/2026-10-11-contact-branch-pivot/SUMMARY.md).

**Actual ordered tractor integration (2026-10-11; diagnostic-only):** fixed-order
native transfer is now prepared before tractor attachment, including backup and
both projection owners. The actual 50k/method-22 run commits **16 frames**, then
fails frame index 16 at the unchanged 64-step Newton limit; rollback preserves
the host machine state exactly. A changed failure-capture run reproduces the same
reached machine trajectory (**1,683 scalar words including rollback**) and exports
the failing packet. Isolated replay matches the archived report and solver state;
its physical error is **1.32082e-5**, above the unchanged 1e-6 bound. This is a
reproducible bounded-convergence failure, **not** 33-frame completion or
repeatability/performance acceptance. No default or numerical gate changed.
[Actual integration, retained failures and isolated replay](benchmarks/soil/2026-10-11-ordered-tractor/SUMMARY.md).

**Ordered projected coupling (2026-10-11):** sticking and unilateral dense MPM
coupling now upload ordering parameters, regenerate sorted indices before P2G,
and use the shared transfer encoder. Prepared-state and capacity guards remain.
Frame rollback supports the full material/index-tail buffer; derived sorting and
gather scratch are rebuilt before reuse. Portable/native ordered scatter and
test-only native gather pass tile-crossing feedback/history parity (maximum
9.974e-6, unchanged 2e-5 bound), empty/count lifecycle and analytical impulse/profile
checks. All ten late-particle-fault cases block machine publication, restore
state exactly and retry successfully. No hot GPU allocation/readback is observed;
ordering adds a declared 32-byte parameter upload per substep. Subsequent actual
ordered tractor evidence is recorded above; defaults and unmet speed gates remain.
[Implementation, retained failures and bounded verification](benchmarks/soil/2026-10-11-ordered-coupling/SUMMARY.md).

**Native dense fixed-order transfer (2026-10-11; diagnostic-only):** the
test-only gather now supports native dense grids without enabling active-grid
mode. Native lifecycle, prepared/direct steps, loaded edge cells and 1D/2D
dispatch checks pass, as do the existing active/compact gather regressions.
On the saved evolved 50k state, mass, momentum and velocity XYZ repeat exactly
and remain within the unchanged 2e-5 scatter comparison bound. Including sorting,
median GPU reset/P2G/grid time is **1.05728 ms versus 0.09728 ms** for native
scatter (10.87 times slower). Keep diagnostic-only; prescribed machine coupling
integration was still pending at this checkpoint and is recorded above.
This does not establish whole-simulator determinism or real-time acceptance.
[Source, lifecycle, parity and paired timing evidence](benchmarks/soil/2026-10-11-native-dense-gather/SUMMARY.md).

**Full live method-21/22 isolation (2026-10-11):** the equal-input shadow now
supports an explicit 33-frame run. All **1,122 solves / 165,486,024 words** match
bit-for-bit across output, rate, publication, state and report buffers. The actual
50k diagnostic completes in **167.36 s** under the unchanged 180 s cap, with no
physical runtime fault. This finds no tiled-solver divergence on that instrumented
trajectory; it neither identifies the cause of the earlier independent-trajectory
failure nor waives its 2e-5 gate. Defaults and unmet real-time targets remain.
[Full live comparison, negative controls and scope](benchmarks/soil/2026-10-11-newton-products-shadow/SUMMARY.md).

**Tiled publication runtime (2026-10-11; opt-in method 22):**
`publication-products-newton` now integrates the tiled arithmetic with distinct
capture/replay, benchmark and renderer identity; defaults remain unchanged.
Runtime lifecycle and native MPM feedback pass. Same-source method-21/22 actual
50k runs both finish all 33 frames under their 180 s caps. Observed median wall
time is **588.33 → 567.45 ms** (3.55% lower), contact solve **238.37 → 225.11 ms**
(5.56% lower), but tails worsen and **trajectory admission fails from frame 13**
at the unchanged 2e-5 bound. All measured frames still miss 60 Hz. Keep opt-in;
do not treat captured exactness or sequential timing as full runtime acceptance.
[Integration, paired measurements and retained admission failure](benchmarks/soil/2026-10-11-newton-publication-products-runtime/SUMMARY.md).

**Tiled publication arithmetic (2026-10-11; test-only improvement):** 256 lanes
now experimentally compute response products and contact/DOF reactions in
16-contact tiles, reusing dead Krylov scratch. Final paired sums and energy
order remain unchanged. All 875 GPU lifecycle/width/tile cases pass; eight
captured complete solves retain exact five-buffer outputs and independent
physical checks, with medians **2.80–11.96% lower than method 21**. No new buffer
or dispatch is needed. Subsequent opt-in runtime integration is recorded above;
these captures alone do not establish whole-frame or 4 ms success.
[Implementation, scratch lifetime, exact comparison and timings](benchmarks/soil/2026-10-11-newton-publication-products/SUMMARY.md).

**Zero-impulse publication experiment (2026-10-11; rejected):** a test-only
branch retains the first paired normalization and skips two axes for exactly
zero represented impulses. All 175 lifecycle/edge cases and eight captured
solves preserve exact outputs and physical checks, but every captured median
regresses **1.96–7.82%** against method 21. None of those final active contact
sets contains a zero impulse vector. Offline inspection also finds no zero
response coefficients, despite about 61% zero Jacobian row coefficients.
Do not promote or repeat the unchanged branch; default behavior, tolerances and
unmet real-time gates are unchanged.
[Implementation, retained fixture failure and measured rejection](benchmarks/soil/2026-10-11-newton-zero-publication/SUMMARY.md).

**Bounded residual fast path (2026-10-11; test-only):** compensated native P2G
now has an experimental common-range f32 path with unchanged f64 fallback.
All **12,373** arithmetic cases match independent f64 results and original GPU
f64 bits. On the saved evolved 50k state, corrected grids match the f64 control
with no observed repeat variation, but median GPU reset/transfer/grid time only
decreases **0.194816 → 0.189200 ms** (2.88%). A separate matched native pair is
**0.090208 → 0.189440 ms**: compensation is still about 2.1 times slower.
Keep test-only; no default change, determinism guarantee or real-time acceptance.
[Arithmetic bounds, retained state and timing evidence](benchmarks/soil/2026-10-11-fast-compensated-p2g/SUMMARY.md).

**Live equal-input solver isolation (2026-10-11):** a test-only method-20
shadow now shares each method-21 solve's immutable prepared inputs without feeding
its outputs back into physics. All **476 solves across 14 actual 50k frames**
(through frame 13) have bit-identical outputs, rates, publication, state and
reports: **70,206,192 words** checked in a 143.51 s bounded run. The checker passes
injected mismatch, first-word, reuse and reset controls. No equal-input solver
output difference was found on this instrumented trajectory; the preceding
independent-trajectory failure remains unresolved, not waived. This is not
performance acceptance or a new deterministic native-MPM backend.
[Implementation, exact comparison and scope](benchmarks/soil/2026-10-11-newton-publication-shadow/SUMMARY.md).

**Ordered publication scans (2026-10-11; opt-in runtime integrated):**
`active-publication-newton` adds protected method 21 on top of method 20,
replacing two serial capacity-wide scans with the ascending active-contact prefix.
Validation, arithmetic order, physical gates and defaults are unchanged.
The prototype preserves exact five-buffer outputs on eight captures, with median
durations **21.14–53.50% lower**, and passes 65 GPU lifecycle cases.
Runtime integration passes 97 source/compiler checks, 15 CLI/metadata checks,
failure-capture lifecycle and native MPM feedback. The actual 50k/33-frame fixture
completes under 180 s: **574.16 ms median**, 778.09 ms p95, 815.36 ms p99;
all 30 measured frames miss 60 Hz. A subsequent same-source method-20 control
completes at **794.08 ms median**: observed median time is 27.70% lower for
method 21, but **full trajectory equivalence fails** the unchanged 2e-5 bound
from frame 13. Both runs use snapshot-only telemetry; do not present this as an
isolated causal speedup, default promotion or 4 ms acceptance.
[Prototype and direct phase measurements](benchmarks/soil/2026-10-11-newton-active-publication/SUMMARY.md),
[runtime integration and actual workload](benchmarks/soil/2026-10-11-newton-active-publication-runtime/SUMMARY.md).

**Whole-hybrid solver boundary (2026-10-11; not promoted):** a test-only dispatch
split preserves exact outputs and physical checks on six reached packets, but
all five fully timed cases regress by roughly **7.57–17.19%** versus method 20.
The second command reaches the unchanged 90 s cap with two packets unreached and
one timing matrix partial; full coverage is explicitly failed. Keep production
unchanged and do not repeat this losing split as a throughput optimization.
[Implementation, partial evidence and retained timeout](benchmarks/soil/2026-10-11-newton-hybrid-boundary/SUMMARY.md).

**Actual resident tractor rendering (2026-10-11):** the same-device renderer now
draws the actual 50k-particle, six-body coupled workload directly from GPU position
and published body-transform buffers. All 33 frames complete under the unchanged
180 s cap; initial/final images are inspected. Render-pass median is **0.0448 ms**,
but blocking machine-plus-soil stepping remains **800.40 ms median**, with all 30
measured frames missing 60 Hz. This is offscreen analytic geometry, not the Bevy
frontend, calibrated soil, reconstructed terrain or real-time acceptance.
[Implementation, images, failure recovery and timing scope](benchmarks/soil/2026-10-11-resident-tractor-render/SUMMARY.md).

**Resident particle rendering input (2026-10-11):** same-device renderers can now
borrow counted MPM positions without CPU transfer, with binding-owner validation
and fault-checked committed-frame access from the world/tractor. Actual GPU
rendering covers evolved particles, empty and 50k reseeds through one retained
binding; the tractor rollback/provenance gate passes in 71.04 s. This does not
replace the Bevy viewer's wgpu-27 CPU bridge or establish real-time acceptance.
[API, rendered evidence and remaining integration](benchmarks/soil/2026-10-11-resident-particle-render/SUMMARY.md).

**Lane-local Krylov scratch (2026-10-11; test-only):** two MGS sweeps now have an
experimental lane-local path for at most 256 active contacts, with the original
path above that count. All eight captured solves preserve exact outputs and
physical publication; 24 capacity/basis boundary cases and 95 CPU/compiler checks
pass. Captured medians improve 0.37–1.58%, but no integrated frame or trajectory
claim follows. Defaults remain unchanged and real-time acceptance remains open.
[Implementation and verification](benchmarks/soil/2026-10-11-newton-private-mgs/SUMMARY.md).

**Norm reduction experiment (2026-10-11; not promoted):** a test-only four-way
maximum tree halves barrier stages and preserves all eight captured solves
bit-for-bit, including independent physical publication. Expanded component
checks and 94 CPU/compiler checks pass. Seven later packet median improvements
are below 0.2%; the first aggregate worsens during a timing transition. This is
not a useful contact-solve speedup; runtime defaults and method 20 remain unchanged.
[Evidence and retained subnormal limitation](benchmarks/soil/2026-10-11-newton-norm-quaternary/SUMMARY.md).

**Paired joint-type specialization (2026-10-11; setup benefit only):** the opt-in
profile now preserves the existing split paired-rate prediction and reaches the
actual 50k Newton workload. Both bounded same-source arms complete 33 frames;
576 component state/history words match exactly. Observed construction is
122.91 s Generic / 77.68 s specialized, but frame medians remain **808.76 / 801.62
ms**, with all deadlines missed and worse candidate p95/p99. Full machine rate
comparison fails the existing 2e-5 bound from frame 13, despite particle positions
passing. Keep Generic default and return optimization to contact/machine stepping;
do not present this as a real-time gain or accepted trajectory equivalence.
[Implementation, matched workload and retained failure](benchmarks/soil/2026-10-11-paired-joint-types/SUMMARY.md).

**Model-specific machine compilation (2026-10-11; opt-in):**
`ExperimentalJointTypes` specializes the actual model's joint-type sequence,
retains dynamic controls/limits/state, separates program-cache identity and
rejects type/count mutations before dispatch. The one-free-body analytic/guard
fixture passes, and the actual six-body/65-particle friction-runtime gate now
passes in **67.06 seconds including setup**, under the unchanged 90-second cap.
This includes late particle-fault rollback/recovery and zero steady buffer
allocations. Twenty-three kernel CPU checks, strict CLI selection and the facade
build pass. Generic remains default; paired-rate 50k Newton, same-input trajectory
parity and per-frame performance acceptance remain open.
[Implementation, bounded runtime evidence and limits](benchmarks/soil/2026-10-11-joint-vocabulary/SUMMARY.md).

**Opt-in compiler cache (2026-10-11; machine gate still open):** engine-owned
driver caches and opaque snapshots now feed engine-backed Featherstone pipeline
creation, separately from the existing 64-program LRU. No shader arithmetic or
default feature changes. Ten compute and 21 kernel CPU checks pass; a GPU
fixture preserves exact outputs across device recreation and source/constant
changes. The one-joint physical comparison still exceeds 90 seconds in its first
cold pipeline, before cache export or stepping. No machine speedup or physical
acceptance is claimed. Snapshots are explicit maintenance operations; external
byte imports require trusted get_data provenance, with no automatic disk I/O.
[Implementation, usage and retained failed gate](benchmarks/soil/2026-10-11-pipeline-cache/SUMMARY.md).

**Compiler probe closed (2026-10-11; no accepted remedy):** a bounded parent-GDB
sample places the stalled thread inside NVIDIA's shader compiler. A test-only
SPIR-V loop-preservation hint changes exactly 341 loop-control words; both
binaries validate and 21 CPU checks pass. The hinted pipeline still exceeds
the unchanged 90-second construction limit, before any physics dispatch.
No matched control, production adoption or speedup claim follows this failure.
Keep implementation and initial numerical/performance acceptance independent
of unavailable physical calibration data; the confirmed RTX 4080 target remains
one tractor plus implement, 50k active particles, 60 FPS and at most 4 ms soil
work per 1/60 second of simulated time.
[Compiler stack, candidate and retained timeout](benchmarks/soil/2026-10-11-compiler-stack/SUMMARY.md).

**Construction stall localized (2026-10-11):** opt-in
`MOLLA_GPU_CONSTRUCTION_TRACE=1` identifies the first six-joint scalar
Featherstone pipeline creation as the timeout, before soil coupling setup.
Shader modules finish in roughly 17 ms. A test-only phase-0 specialization still
times out at 90 seconds; other phases are not reached. A separate unrestricted
scalar source probe compiles in 86.6 seconds but has no simulation/physical proof
and is not adopted. Twenty CPU checks pass. This is setup diagnosis, not a soil
step-speed improvement; full tractor fault verification and real-time acceptance
remain open.
[Trace, prototypes and preserved limits](benchmarks/soil/2026-10-11-construction-trace/SUMMARY.md).

**Coupled late-fault guard (2026-10-11):** projected MPM now propagates particle
faults after gather/plasticity into contact status before machine publication.
The portable/native sticking/unilateral fixture proves rejected machine rates
and velocity delta, exact soil/latch rollback and successful caller recovery,
with no hot buffer allocation/readback. Four existing coupling regressions and
82 CPU checks pass; facade and warm-Newton solver checks pass. The actual tractor
runtime hits its 90-second construction deadline before the added assertions,
so full-world evidence remains open. This adds one GPU dispatch per coupled step;
its performance cost is unmeasured. Compact-grid coupling remains unsupported.
[Implementation and preserved runtime timeout](benchmarks/soil/2026-10-11-coupled-particle-fault/SUMMARY.md).

**Numerical-fault containment (2026-10-11):** gather/plasticity no longer replace
NaN history with zero velocity or identity deformation. Dense, active and compact
grids latch nonfinite state; captures reject it and settling reports Failed,
not quiet soil. Capacity and numerical faults have separate diagnostics. Tests
cover 162 injected matrix faults, an infinite gathered velocity, exact state
preservation in the single-particle fixture and checkpoint recovery. Ten GPU
tests, 82 CPU checks and the facade build pass. A fault may leave a partial
multi-particle substep: restore known-good state, do not treat the latch as rollback.
This does not establish physical sleep, universal solver-fault containment or
real-time performance.
[Implementation, limits and frozen evidence](benchmarks/soil/2026-10-11-numerical-fault/SUMMARY.md).

**Verified settling continuation (2026-10-11; still not converged):**
`make settle-soil SOIL_SETTLING_RESUME=<parent-results>` now validates bounded
parent inputs, hashes, policy, clocks and full state/identity before restoring.
The second segment starts from byte-identical saved history and reaches 4000
cumulative steps / 2.000000095 simulated seconds. All 125 new records pass
independent interval/lineage checks, but peak endpoint speed is still 18.28 mm/s
and plastic activity 0.213/s. It correctly returns StepLimit/exit 2 with no quiet
window. Six CPU contracts pass. Do not infer sleep or real-time acceptance from
decaying activity or the small-bed timing.
[Continuation implementation and retained nonconvergence](benchmarks/soil/2026-10-11-settling-continuation/SUMMARY.md).

**Gravitational-bed preparation (2026-10-11; not converged):** added
`make settle-soil`, a reproducible 512-particle full-bed workflow with frozen
source/executable, policy, activity history and initial/final archives. The first
bounded run completes 2000 steps / 1.0000000475 simulated seconds and 125 samples,
then correctly exits 2 with StepLimit. Peak endpoint speed remains 45.29 mm/s
against the 1 mm/s cutoff; elastic/plastic rates remain about 0.60/s. All particles
and mass/volume/material bits are retained. Independent record/state checks pass
and reject a falsified success result. No unchanged retry or threshold relaxation;
the bed is not accepted as settled. Small-bed driver timing excludes setup and
does not establish the 50k tractor/implement real-time target.
[Frozen workload, nonconvergence and retained state](benchmarks/soil/2026-10-11-gravity-settling/SUMMARY.md).

**Bounded settling preparation (2026-10-11):** added an explicit, nonblocking
driver borrowing either live world. It requires the whole nonempty soil population
at a committed raw boundary and uses caller-supplied activity cutoffs, quiet
duration/count, batch size and step/wall budgets. ActivityConverged, StepLimit,
WallLimit, Failed and deadline-draining states are distinct. It never sleeps or
evicts particles. Sixty-one CPU checks, three real-GPU lifecycle tests and the
facade check pass. Positive evidence is a two-particle zero-gravity fixture;
gravity-driven motion correctly exhausts the step budget. Gravitational-bed
preparation, physical sleep/support and real-time acceptance remain open.
[API, commands, usage and bounded evidence](benchmarks/soil/2026-10-11-bounded-settling/SUMMARY.md).

**Checkpoint activity measurements (2026-10-11):** both world owners now offer
opt-in motion/deformation/plastic/contact-history measurements using existing
mandatory captures. Reports retain physical interval, global-ID hash and committed
request provenance, without counting newly imported particles as observed. Failed
transitions retain elapsed time; no solver-iteration readbacks or automatic sleep
are added. Fifty-nine CPU and 15 GPU checks plus the facade build-check pass.
These endpoint measurements can miss intervening transients and are not force
residuals, equilibrium certificates or field-stream performance acceptance.
[Activity API, transaction evidence and research context](benchmarks/soil/2026-10-11-soil-activity/SUMMARY.md).

**Active soil outside machine envelopes (2026-10-11):** both live world owners
now expand swept-update pins around captured active soil, using occupied-tile
particle bounds, the shortest refresh horizon, conservative relative travel and
the existing roundoff-aware transfer stencil. Optional GPU dilation is retained
in the union; raw requests keep their manual-coverage contract. Eighteen CPU and
19 GPU tests pass, including detached-neighbor activation in both owners and
capacity failure without changing either authority. All active particles remain
pinned, including stationary material. This does not establish physical sleep,
active/dormant support correctness, bounded field traversal or real-time timing.
[Implementation, GPU witnesses and limits](benchmarks/soil/2026-10-11-active-swept-pins/SUMMARY.md).

**Preconditioner-only f32 arithmetic (2026-10-11; test-only):** retaining original
matrix products and true residuals avoids the prior mixed-inner regressions on
eight captured packets. All 64 frozen linear records, 16 physical checks, eight
public comparisons and 128 exact own-repeat checks pass unchanged gates. Paired
timings change by less than 0.5% in seven cases; one case improves 10.41% with a
different nonlinear path (46 to 34 outer iterations). Two cases show substantial
within-run timing transitions. No general speedup, trajectory acceptance or
runtime promotion is claimed. The 4 ms/60 FPS target remains unmet.
[Preconditioner scope, complete evidence and limitations](benchmarks/soil/2026-10-11-mixed-precondition/SUMMARY.md).

**Resident mixed inner product (2026-10-11; rejected experiment):** added a
separate f32 Arnoldi product while retaining paired true residuals, recovery,
fallback and physical certification. A real-GPU cancellation fixture confirms
that an approximate zero residual cannot hide paired relative error one.
Sixteen frozen systems and eight full tractor packets pass unchanged numerical
gates; maximum public-output difference is 7.630e-6 against the 2e-5 bound,
with 128 exact own-repeat checks. Performance regresses in six of eight cases,
up to 74.99%, as additional linear/nonlinear work outweighs cheaper arithmetic.
Keep the original operator and runtime defaults. No real-time acceptance.
[Accuracy guard, complete-solve evidence and retained regression](benchmarks/soil/2026-10-11-mixed-inner/SUMMARY.md).

**Fused distributed-prefix stages (2026-10-11; test-only):** fused
precondition/projection into initialization and advancement, retaining explicit
barriers and the original math. Dispatch count falls from 2275 to 1235; all
80 five-mode numerical comparisons on eight actual packets preserve the five
public buffers and pass physical checks. Timing is incomplete: the 90-second
cap stops the matrix at 358/360 samples, and the strict summary rejects it.
Across seven fully measured cases, fusion improves fixed distribution by
4.75–20.89% but remains 9.18–89.93% slower than the original resident solver.
No unchanged retry or runtime adoption. Investigate resident arithmetic cost
rather than assuming further distribution will meet the target.
[Fusion, retained timeout and partial timing evidence](benchmarks/soil/2026-10-11-distributed-fused/SUMMARY.md).

**GPU-driven prefix dispatch counts (2026-10-11; test-only):** added retained
indirect arguments, written by initialization/advance kernels in a separate
storage bind group. Completed iterations dispatch zero consumer groups;
products use actual active contacts rather than capacity. All 64 four-mode
comparisons and 288 timed solves preserve complete outputs and pass physical
checks. This does **not** fix performance: indirect execution is another
2.68–6.06% slower than fixed distributed dispatch in paired samples. The 2275
encoded commands remain; reducing command count is the next target. Preserve
the regression and original defaults, with no real-time acceptance.
[Indirect ownership, scope and unfavorable timing](benchmarks/soil/2026-10-11-distributed-indirect/SUMMARY.md).

**Distributed prefix in complete Newton solves (2026-10-11; test-only):**
integrated dispatch-separated prefix solving into suspended/resumed nonlinear
scheduling, preserving the original hybrid direct/fallback, damping, release,
backtracking and publication path. Eight actual packets, reaching 46 outer
steps, match all five public buffers exactly across three modes and repeats;
independent physical checks pass. All 216 timed solves also preserve outputs.
However, the distributed prefix is **slower by 14.98–136.28%** in these paired
samples: 2275 fixed dispatches and capacity-sized product groups are the next
scheduling target. Post-direct fallback/damping remain resident. Preserve this
regression; no runtime promotion or 4 ms/60 FPS acceptance.
[Complete-solve integration, timing and remaining work](benchmarks/soil/2026-10-11-distributed-nonlinear/SUMMARY.md).

**Captured distributed linear solves (2026-10-11; test-only):** implemented
hashed capture/replay of actual method-20 tractor linearizations, preserving
paired blocks, inverse and active map. Sixteen systems from eight packets at
outer steps zero and one pass all 64 resident/staged repeat records exactly.
Every accepted direction passes the unchanged independent f64 1e-5 residual
gate; the maximum is 6.552e-6. Actual coverage reaches four iterations, not the
hardest later solves. Source generation now retains the complete method-20
profile. Nonlinear scheduling, hybrid fallback, complete-solve timing and
real-time acceptance remain open; runtime defaults are unchanged.
[Frozen input, residual and scheduling evidence](benchmarks/soil/2026-10-11-frozen-linearization/SUMMARY.md).

**Distributed refinement/restart (2026-10-11; test-only):** extended the staged
Krylov cycle with shared total-budget accounting, paired base-direction handoff
and restart from the true residual. All 108 synthetic configurations and repeats
match resident refined-solver outputs exactly; 24 cases exercise an explicitly
forced restart. Accepted directions pass the unchanged f64 1e-5 residual gate,
and capped failures preserve the original budget. Current cases exercise at
most two iterations/one restart, not hard captured trajectories. Actual tractor
linearizations, nonlinear scheduling and complete-solve timing are next; runtime
defaults and full-plan acceptance remain unchanged.
[Refinement state, actual coverage and remaining integration](benchmarks/soil/2026-10-11-distributed-refinement/SUMMARY.md).

**Persisted distributed Krylov cycle (2026-10-11; test-only):** implemented
dispatch-separated precondition/project/multi-workgroup-product/orthogonalize
steps with explicit Hessenberg, rotation, norm, stagnation and index state.
All 54 width/limit/active-count cases and repeats match the resident cycle
exactly. Coverage includes 24 two-iteration cases and 12 expected capped
failures; accepted directions pass the independent 1e-5 f64 residual gate.
Measured-scope resource checks pass, and completed-cycle dispatches are no-ops.
This is one bounded cycle, not yet refinement/restart or nonlinear Newton
integration. Captured harder cycles and full-solve timing remain required;
defaults, limits and physical gates are unchanged.
[Persisted state, exact cycle evidence and remaining integration](benchmarks/soil/2026-10-11-distributed-cycle/SUMMARY.md).

**Distributed Newton foundation (2026-10-11; test-only):** implemented an
explicit multi-workgroup forward/paired-direction product sequence. Keep the
ordered 256-lane machine projection, persist its pairs, then distribute ten
contacts per 64-lane group with independent matrix rows. No spin barriers,
host intermediate readback or changed dot-product order. All 36 edge
configurations and repeats match the resident products exactly and pass f64
checks. This is not yet a distributed Krylov/Newton solve: persisted iteration
state, scratch ownership and captured-solve performance remain to implement.
The prototype does not change runtime defaults or establish a speedup.
[Kernel structure, compiler boundary and exact edge evidence](benchmarks/soil/2026-10-11-newton-distributed/SUMMARY.md).

**Transfer cost/native lifecycle (2026-10-11; test-only):** combined transfer
passes native/portable parity through 193/65/0/1-particle epochs and unsupported
mode checks. Paired 50k measurements now include the entire prepass and four
explicit substeps. It is slower: median paired GPU changes are +15.07% portable
legacy, +14.97% portable two-pass, +0.74% native legacy and +6.24% native two-pass.
All 72 timing records and the unfavorable results are retained. Zero observed
in-scope allocations/transfers are not full-workload acceptance. Keep test-only:
this is an accuracy/performance tradeoff, not a speedup. Sparse/consumer lifecycle
and prior trajectory mismatches remain open, as does integrated real-time work.
[Matched timing scope, lifecycle and evidence](benchmarks/soil/2026-10-11-transfer-timing/SUMMARY.md).

**Evolved-input accuracy checkpoint (2026-10-11; test-only):** the combined
transfer path evolves three substeps and passes all 96 fixed-input fourth-step
replay checks. The original solver, restored from those exact particles, fails
four fresh-scatter torque repeats. Both variants pass the independent f64
canonical wrench oracle in all four cases and a new f64 gather oracle over
200,000 particles each, including all nine affine components. These are
one-step checks on captured evolved inputs, not an independent full trajectory;
the prior 9/24 cross-variant failure remains open. Hardened oracle input/finite
checks and fixed optional uniform capture on owners lacking COPY_SRC.
Matched-work timing and native/resource lifecycle are next; no default change
or real-time acceptance.
[Independent references, retained failures and reproduction](benchmarks/soil/2026-10-11-evolved-transfer/SUMMARY.md).

**Combined transfer checkpoint (2026-10-11; test-only):** composed bounded
relative-momentum scatter with centered affine gathering in one installer.
The unchanged 50k matrix passes all 72 candidate dense-repeat/active/compact
comparisons; the original control fails 25/72. Affine/clipped-stencil and
large-velocity analytic fixtures pass. Candidate-versus-original torque still
fails 9/24 comparisons, so the complete gate fails and defaults remain unchanged.
The original also fails its own repeats: obtain independent evolved-input
accuracy evidence rather than attributing every mismatch to either variant.
No unchanged retry, tolerance relaxation or speedup claim.
[Implementation, frozen source and full results](benchmarks/soil/2026-10-11-bounded-centered/SUMMARY.md).

**Bounded velocity reference (2026-10-11; test-only):** replaced arbitrary
first-particle selection with a device-wide per-component minimum-magnitude
reference for same-sign velocities, otherwise zero. Exact selection/reset and
the retained high-dynamic-range fixture pass. All 96 fixed-input replay checks
and four independent f64 canonical cases pass, with all 36 particle input files
identical to the retained originals. However, the evolving 50k matrix still
fails torque: control 28/72, candidate 5/72 and cross-variant 8/24 comparisons.
Do not promote or rerun unchanged for a lucky pass. Two extra dispatches per
live substep are unmeasured overhead; existing stage timings do not separately
include the prepass. Defaults and frozen tolerances remain unchanged.
[Implementation and retained results](benchmarks/soil/2026-10-11-bounded-reference/SUMMARY.md).

**Independent oracle and range rejection (2026-10-11):** implemented an
offline f64 scatter/sphere-reaction reference using existing f64 SVD, with
analytic force/affine fixtures and mass/linear/angular conservation checks.
Both original and relative-momentum canonical reactions pass all four captured
50k reference cases. This does not close their stricter pairwise gate.
A new disjoint-particle range fixture exposes a decisive defect in the relative
prototype: choosing an arbitrary first-particle velocity loses small finite
velocities when that reference is large. Reject that selection for production;
a bounded device-wide reference reduction is required before reconsideration.
No defaults, tolerances or performance claims changed.
[Oracle, preserved failures and next implementation](benchmarks/soil/2026-10-11-wrench-oracle/SUMMARY.md).

**Relative-momentum candidate (2026-10-11; test-only):** implemented consistent
velocity-offset scatter, rigid reaction and absolute-momentum restoration
without changing G2P/plasticity or adding storage/dispatches. All 96 exact-input
replay checks pass. The 50k matrix passes all 72 candidate dense-repeat/active/
compact comparisons, while its unchanged original control fails 21/72.
However, candidate-versus-control torque fails 7/24, and one of four retained
canonical reference comparisons fails. Keep test-only: independent numerical
accuracy, conservation, consumer compatibility and performance remain required;
repeatability is not production or full-plan acceptance.
[Implementation, complete evidence and admission decision](benchmarks/soil/2026-10-11-relative-momentum/SUMMARY.md).

**Scatter/wrench isolation (2026-10-11):** a new fixed-input replay retains
the exact evolved 50k particle state and alternates fresh scatter with an exact
saved pre-imprint grid. Across explicit/implicit and legacy/two-pass paths,
all 48 identical-grid wrench replays pass; one of 48 fresh-scatter cases fails
torque at 1.01328 times the frozen tolerance despite passing grid-field gates.
All 96 samples and the failing grid are retained. This identifies a scatter
contribution to the fixed-input failure, not a complete explanation of every
trajectory mismatch or an accepted solver optimization. Production unchanged.
[Replay evidence and next numerical target](benchmarks/soil/2026-10-11-scatter-wrench/SUMMARY.md).

**Large-domain/APIC checkpoint (2026-10-11):** after the production
local-coordinate correction, the existing eight-case dense/active/compact
benchmark still fails rigid-torque parity in five patch cases (8.28 seconds).
A distinct test-only centered affine moment eliminates the remaining
constant-grid gradient exactly and passes analytic affine/clipped-support
checks, but its six-configuration 50k matrix still fails torque: 27/72 control,
23/72 candidate and 10/24 cross-variant comparisons. Both variants fail
independent dense repeats; positions, velocities and forces pass. Keep the
candidate test-only and active/compact modes opt-in; no real-time acceptance.
Also fixed the preceding sliced experiment's missing protected-feature guard,
which prevented GPU-only unit-test compilation.
[Large-domain evidence](benchmarks/soil/2026-10-11-local-transfer-domain/SUMMARY.md)
and [analytic candidate, rejected admission and full matrix](benchmarks/soil/2026-10-11-centered-affine/SUMMARY.md).

**Dispatch-sliced Newton experiment (2026-10-11; test-only):** preserve the
method-20 arithmetic and caps, yielding after each accepted outer update with
continuation state in existing scratch storage. The corrected replay harness
accounts for 65 bounded dispatches versus the control's one, while retaining
zero-allocation/transfer and exact-output checks. Eight captures with eight
alternating pairs pass five-buffer/report equality and independent physical
publication in 47.06 seconds. Encoded GPU medians regress 0.947–1.061% on two
cases and improve 1.742–2.777% on six; no runtime/default promotion or additional
actual-tractor run is justified by this mixed component result. The initial
single-dispatch harness failure remains archived. Full acceptance stays open.
[Source-frozen measurements and decision](benchmarks/soil/2026-10-11-newton-sliced/SUMMARY.md).

**Actual workload after APIC correction (2026-10-11):** the source-frozen
method-20 run completes 33 frames / 0.55 simulated seconds with 50k particles
within the original 180-second setup-inclusive bound. Median/p95/p99 are
**789.883 / 982.866 / 1038.036 ms** machine-plus-soil wall time; all 30 measured
frames miss 60 Hz. Mean Newton work is **481.241 ms/frame**. Rendering/export
remain excluded; physically uncalibrated. This renews short-run execution
evidence, not performance acceptance. Old/new particle trajectories differ by
up to **1.463 mm** and do not meet trajectory-equivalence bounds. The numerical
repair is not a speedup; contact solving and machine dynamics remain dominant.
[Current actual evidence](benchmarks/soil/2026-10-11-local-transfer-actual/SUMMARY.md).

**Persistent-store lock repair (2026-10-11):** explicit RAII unlocking now ties
the exclusive file lock to the store owner rather than the lifetime of every
duplicated/inherited descriptor. A deterministic failed-before regression passes,
including failed construction and unwinding. The formerly failing parallel CPU
lane passes **65 checks**; **29 store checks** retain asynchronous worker ownership
and crash-recovery behavior. This resolves the demonstrated lock-lifetime defect,
not full streaming or performance acceptance.
[Implementation and retained failure](benchmarks/soil/2026-10-11-store-lock-release/SUMMARY.md).

**APIC coordinate correction (2026-10-11):** production scatter and G2P now
evaluate node-particle offsets in stencil-local coordinates. A constant-velocity
GPU witness exposes a false affine gradient of **762.95074 /s** with the old
world-coordinate reconstruction; the corrected worst case is **0.000429789 /s**.
The previously failing batched 128-step particle-order transport now passes its
unchanged 2e-5 gate (maximum velocity difference **5.24521e-6**). All 1536 corrected
matched/evolving replay comparisons pass. Four ordering, five compact-grid,
six active/deferred, three focused gather and native 16-step parity fixtures pass;
63 serialized CPU checks and the facade build pass. Parallel CPU store-lock
reopen failures remain unresolved and recorded. Backend selection is unchanged.
These are numerical corrections, not a speedup claim. Earlier actual tractor
timings below predate this arithmetic change and are not renewed acceptance.
[Defect, repair, retained failures and scope](benchmarks/soil/2026-10-11-local-transfer/SUMMARY.md).

**Particle-order drift isolation (2026-10-11):** implemented a six-owner replay
with unordered/reference-sort/parallel-sort own-repeat controls. All 768
identical-input one-step comparisons stay within the frozen 2e-5 position/velocity
gate. Evolving trajectories produce 44 failing comparisons, including the
reference sort against itself from step 100; the failure is not unique to the
parallel sorter. Both phases complete 128 steps with checked counters and no
overflow. First divergent states and endpoints are retained; numerical drift
must be addressed before ordering adoption. This observed single-step diagnostic
does not establish equivalence to the original batched run or a performance gain.
[Implementation and complete failure evidence](benchmarks/soil/2026-10-11-order-replay/SUMMARY.md).

**Parallel norm reduction (2026-10-11):** implemented `norm-tree-newton`, an
opt-in method-20 route replacing only the robust norm's serial maximum scan
with a workgroup tree. All 48 norm edge combinations and eight captured cases
pass exact old/new output checks and independent numerical/physical gates.
Seven later captured cases improve **1.52–6.57%** with unchanged reports; the
first crosses a timing transition and its aggregate is not a stable estimate.
Defaults and ordinary WGSL are unchanged. Native feedback passes; the actual
50k diagnostic completes 33 frames at **787.776 ms median / 972.480 ms p95 /
1036.616 ms p99** machine-plus-soil time, with mean Newton work **484.197 ms**.
All 30 measured frames miss 60 Hz. Rendering/export are excluded and results
remain physically uncalibrated. The separate method-19 run is not a paired
whole-frame speedup control; evolving trajectories differ. This is not
4 ms/60 FPS acceptance.
[Implementation and frozen evidence](benchmarks/soil/2026-10-11-newton-norm-tree/SUMMARY.md).

**Operation attribution (2026-10-11):** exact-output-checked invocation counters
and bounded phase perturbations now separate norms, inner products, preparation
and generalized-rate projection. Norm 2x/4x cost increments scale approximately;
projection repetition has near-zero/mixed increments, so it does not establish
timing dominance or justify a projection rewrite. Counts are not additive timing
shares. Retained captures now include clock/P-state telemetry.
[Counts, controls and caveats](benchmarks/soil/2026-10-11-newton-operation-counts/SUMMARY.md).

**Linear execution experiments (2026-10-11):** completed the loop-local audit:
34 named declarations inspected, with no demonstrated read-before-assignment
on the current method-19 path. No production initialization fix is justified.
Implemented a private row-parallel forward/direction product; 27 edge cases
pass exact controls and an independent f64 oracle, and eight captured solves
retain all five buffers exactly. Throughput is essentially unchanged, so do not
promote it. A separate two-pass classical Gram-Schmidt candidate now batches
Krylov projections; 78 CPU/compiler checks and all eight independent captured
comparisons pass, but seven later cases regress 1.83–7.88%, so it also remains
test-only. First-case timing transitions do not establish gains for either.
All numerical tolerances, iteration caps and runtime defaults remain unchanged.
[Audit](benchmarks/soil/2026-10-11-newton-loop-initialization/SUMMARY.md),
[row-product evidence](benchmarks/soil/2026-10-11-newton-row-products/SUMMARY.md),
[CGS2 implementation/research](benchmarks/soil/2026-10-11-newton-cgs2/SUMMARY.md).

**Compiler boundary and initialization investigation (2026-10-11):** the isolated
division-helper DontInline candidate also exceeds the unchanged 90-second
pipeline-creation limit; control compiles in 20.919 s. No runtime admission or
speedup claim. A minimal ordinary-WGSL test separately reproduces stale values
in an implicitly initialized loop-local struct; explicit zero construction
produces the expected result. The later audit above finds no demonstrated solver
impact. This is not a production correctness fix or performance
improvement. [Frozen diagnostic evidence and next action](benchmarks/soil/2026-10-11-newton-division-boundary/SUMMARY.md).

**Structured Krylov preconditioner (2026-10-11):** implemented opt-in
`schur-newton` / method 19, preserving dense fallback, physical checks and all
limits. Eight captured cases, independent width/fallback gates, 72 CPU/compiler
checks, metadata/lifecycle and native MPM feedback pass. A source-frozen 50k
tractor run completes **33 frames / 0.55 s** at **815.743 ms median / 966.914 ms
p95 / 1043.750 ms p99** machine-plus-soil time, with mean Newton work **501.889
ms/frame**. All 30 measured frames miss 60 Hz. Rendering/export excluded;
physically uncalibrated. Separate prior method-18 trajectories are not a paired
whole-frame speedup or equivalence control. Defaults and viewer selection remain
unchanged; the feature is not full-plan acceptance.
[Implementation, retained failures and actual evidence](benchmarks/soil/2026-10-11-newton-schur-inverse/SUMMARY.md).

**Structured contact factorization (2026-10-11):** added a private paired 3×3
Schur solve for Fischer contact blocks, with unchanged dense fallback and full
physical/publication checks. All 70 CPU/compiler checks and the bounded
eight-packet comparison pass. Six ordinary cases change by less than 0.4%; the
hardest improves from 115.047 to 97.806 ms but follows a different nonlinear path.
The first packet has a timing transition, not dependable speedup evidence.
**No runtime promotion or actual-tractor speedup is claimed.** Next investigate
the still-dense Krylov preconditioner rather than repeat this unchanged probe.
[Frozen sources, all samples and decision](benchmarks/soil/2026-10-11-newton-schur-contact/SUMMARY.md).

**Live geometric surface (2026-10-11):** `make view-soil-tractor` runs the actual
50k-particle protected-Newton tractor/tine owner on a bounded worker and renders
only committed states. Default terrain observation is now a **61,440-byte raster**,
not a full particle download. Optional glyph mode and capture endpoints retain
full particle diagnostics. A source-frozen RTX 4080 capture advances **33 frames /
0.55 simulated seconds**, exits cleanly and has an inspected screenshot. Both
50k-particle endpoints independently match exported cell counts and heights
exactly. Initial decimal-boundary failures were fixed with explicit stored cell
edges and integer-ordered coordinate comparisons; failed artifacts are retained.
Physics wall median is **863.210 ms** with concurrent rendering; surface export
and bounded readback median is **0.574 ms**, outside physics timing. These are
diagnostics, not a paired speedup or whole-frame acceptance. The point envelope
is not a calibrated contact surface, and observation remains blocking. Full
traversal/repeat-pass ruts, fully resident display, device-loss/reset breadth and
the original 4 ms / 60 FPS gates remain open.
[Surface implementation and endpoint evidence](benchmarks/soil/2026-10-11-geometric-surface/SUMMARY.md).
[Earlier viewer and retained teardown failures](benchmarks/soil/2026-10-11-live-tractor-viewer/SUMMARY.md).

**Linear schedule experiments (2026-10-11):** implemented private direct-first
and four-iteration-prefix alternatives to method 18's eight-iteration prefix.
All 68 CPU/compiler checks and both bounded eight-packet GPU comparisons pass
independent physical/state/publication gates, with exact repeats within each
variant. Different algorithms are not required to produce identical trajectories.
Direct-first regresses six cases despite two gains; prefix 4 also regresses most
cases. **Neither is promoted**, and no actual tractor speedup is claimed.
Current runtime, tolerances, work caps and full-plan acceptance remain unchanged.
[Frozen sources, all samples and decisions](benchmarks/soil/2026-10-11-newton-schedule/SUMMARY.md).

**Counters-disabled controls (2026-10-11):** smoke now supports
`SOIL_SMOKE_COUNTERS=0`, compiling out core metric hooks and wgpu counters rather
than just leaving scopes unopened. Both compiled feature states are verified;
disabled metrics are explicit **null/unavailable**, not fabricated zeros.
Fourteen S00 contracts, both feature-build checks and eight alternating-order
MPM/DEM runs pass with unchanged physical gates. Explicit comparison requires
matching source, hardware and physical scope, and archived executables. Observed
on-over-off median changes are -1.235%/-0.097% for MPM and +3.700%/-0.344% for
DEM; mixed signs do **not establish a stable causal overhead estimate**.
Full-workload performance and the 4 ms/60 FPS target remain unmet.
[Implementation, raw controls and comparison limits](benchmarks/soil/2026-10-11-smoke-counter-controls/SUMMARY.md).

**Unified DEM smoke (2026-10-11):** the same baseline command now accepts
`SOIL_SMOKE_BACKEND=dem` (default `mpm`). The retained DEM fixture includes
topology/contact stepping **and terrain export**; a declared preparation step
outside each measured interval preserves owner/resource lifecycle, and the
independent freefall oracle accounts for it. Five samples pass physical,
grain-count, terrain-height, revision and zero-hot-allocation/readback checks.
Median/p95 is **2.325356/2.353324 ms** per eight measured steps; the MPM control
also passes with its existing manifest. Thirteen CPU/process contracts pass;
cross-backend baseline comparisons and unknown backend selections are rejected.
Both are 512-particle diagnostic fixtures, not 50k tractor or real-time acceptance.
[Implementation, scopes and raw evidence](benchmarks/soil/2026-10-11-unified-dem-smoke/SUMMARY.md).

**Unified smoke baseline (2026-10-11):** `make bench-soil CASE=smoke MODE=baseline`
now runs actual dense MPM through `soil_benchmark`, not a test-function wrapper.
The frozen 512-particle freefall fixture resets one warmup and five measured
eight-substep samples; independent position/velocity/mass checks pass, with zero
observed hot buffer allocations/readbacks. Latest median/p95 completed wall time
is **1.156021/1.321441 ms per 0.00400000019 simulated seconds**. This is a small
solver-only diagnostic, not the 50k tractor target. Twelve S00 contracts pass;
owned-process timeout, deliberately unavailable Vulkan adapter and output-directory
reuse are verified to fail without claiming completion or overwriting evidence.
Checked JSON/CSV, full source provenance and hardware timelines are retained.
DEM smoke, broader benchmark schemas and uninstrumented/integrated acceptance
remain open; full-plan scope is unchanged.
[Implementation and bounded success/failure evidence](benchmarks/soil/2026-10-11-unified-smoke/SUMMARY.md).

**Compiled-kernel inspection (2026-10-11):** added a development-only Vulkan
pipeline-statistics diagnostic and bounded Make target. Method 18 reports
**208 registers, 1,244,416 executable bytes and 24,512 shared bytes**; the rejected
shared-matrix version has the same register count. The local-memory statistic
has a suspicious 64-GB high part and is preserved with a warning, not used as
spill evidence. These are separately compiled diagnostic pipelines, not runtime
occupancy or timing measurements. A subsequent test-only ten-function DontInline
hint passes all 66 CPU/compiler checks but **cannot compile within 90 seconds**;
the isolated control completes in 19.55 seconds. No workload replay, runtime
promotion or speedup claim for that candidate. Latest actual method-18 results
and all full-plan acceptance gaps remain unchanged.
[Resource diagnostic](benchmarks/soil/2026-10-11-newton-pipeline-statistics/SUMMARY.md),
[function-control implementation and deadline evidence](benchmarks/soil/2026-10-11-newton-noinline/SUMMARY.md).

**Shared preconditioner storage experiment (2026-10-11):** test-only lane-owned
6x12 matrices reuse existing Krylov scratch with guarded lifetime barriers and
unchanged arithmetic. All 65 CPU/compiler checks and eight exact captured-solve
comparisons pass, including independent physical publication. Every candidate
median is slower: **6.29–14.09%**, with the first case's timing transition retained
in the raw samples. Not promoted; no actual tractor rerun or expanded edge gates
for this losing candidate. Inspect emitted linear-kernel resource/dispatch cost
before further arithmetic micro-optimizations. Latest actual method-18 results
and full-plan acceptance gaps remain unchanged.
[Source, raw samples and negative evidence](benchmarks/soil/2026-10-11-newton-shared-inverse/SUMMARY.md).

**Structured linear prototypes (2026-10-11):** test-only Fischer forward,
transpose and paired-direction products skip known matrix zeros, preserving
the dense path for non-finite operands. All eight captured solves remain exact
and physically verified, but median gains are only **0.66–1.55%**. A separate
guarded zero-elimination shortcut in preconditioning also preserves captured
outputs but is **0.76–2.95% slower on seven stable cases**; its first case crosses
timing regimes and is retained without a speedup claim. 64 CPU/compiler checks
pass. Neither is promoted; isolated edge acceptance is incomplete. Next target
is preconditioner matrix storage, not more unchanged zero-skipping variants.
Latest actual method-18 evidence and all full-plan acceptance gaps are unchanged.
[Source, exact controls and negative/marginal timing](benchmarks/soil/2026-10-11-newton-sparse-products/SUMMARY.md).

**Phase-amplification diagnostics (2026-10-11):** test-only 2x/4x evaluation
and reset linear-solve repetitions preserve exact five-buffer/report outputs
and independent physical checks on all eight captured inputs. All four bounded
comparisons pass, with 62 CPU/compiler checks. On the hardest input, control
is about 115 ms, evaluation 2x/4x about 140/186 ms, and linear 2x/4x about
223/367 ms. Linear work is the stronger next target, but imperfect scaling and
compiler/layout effects mean these are **not precise phase percentages or
production speedups**. Next implementation should exploit the assembled Fischer
block's sparse lower rows, subject to dense-reference, fault and physical gates.
No runtime change or actual tractor rerun; full-plan acceptance remains open.
[Probe design, raw samples and scaling limits](benchmarks/soil/2026-10-11-newton-phase-cost/SUMMARY.md).

**Generated-source correction and invariant cache (2026-10-11):** the assembled
Fischer route already has chunked rate projection and lazy Jacobians; the prior
raw-WGSL inference of DOF-only serial evaluation was incorrect. Its proposed
rewrite failed the source guard before GPU execution and was removed. A separate
test-only lane-private cache now reuses nonlinear-contact normal/mobility
constants across evaluations, keyed by contact and reset each invocation.
61 CPU/compiler checks and all eight captured exact-output/physical comparisons
pass. Seven stable-case medians improve only **0.34–0.43%**; the first case
straddles a timing transition and has a **73.10% higher median**, with all samples
retained. No runtime admission or full-tractor rerun. Next work should attribute
phase cost before further local optimization. Latest actual method-18 result
and all full-plan acceptance gaps remain unchanged.
[Corrected source diagnosis and cache evidence](benchmarks/soil/2026-10-11-newton-nonlinear-cache/SUMMARY.md).

**Candidate-generation reuse prototype (2026-10-11):** test-only recovery
generates each mask's candidate list once on eight lanes, then scores on the
existing 160 lanes with unchanged order/ties. Reuses existing Krylov scratch,
clears invalid-mask counts and guards scratch reset. All 60 CPU/compiler checks
and eight captured exact-output/physical comparisons pass, but every paired
median is **0.19–0.45% slower**. Not promoted; no full-tractor rerun. Latest
actual method-18 evidence remains 825.92 ms/frame. The subsequently proposed
serial-accumulation target was rejected by generated-source inspection above;
phase cost is not yet isolated.
[Prototype and negative performance evidence](benchmarks/soil/2026-10-11-newton-candidate-cache/SUMMARY.md).

**Tiled core prototype (2026-10-11):** test-only assembly computes each
contact/response-row projection once per 16-contact tile rather than once per
matrix column. Both global-scratch and reused Krylov-workgroup-scratch variants
pass dimension edges, independent assembly oracles and all eight captured
solves with exact five-buffer/report equality. Shared scratch changes medians
from **2.44% slower to 4.26% faster**, with nearly no change on the hardest
captured input. The mixed result does not justify runtime admission or another
full-tractor timing run. No new runtime feature, buffers, tolerances or solver
caps; method 18 remains the latest opt-in runtime. Next target is duplicated
per-candidate generation work, not more unchanged tile comparisons.
[Prototype and matched evidence](benchmarks/soil/2026-10-11-newton-tiled-core/SUMMARY.md).

**Volatile retained rounding (2026-10-11):** opt-in `volatile-newton` replaces
only the protected helper's per-lane atomic store/add-zero with guarded volatile
store/load, keeping all rounding sites, arithmetic and solver limits. Method
18 identifies the route; defaults and portable fallback remain unchanged.
40,965 arithmetic cases and eight captured full solves preserve exact control
outputs; alternating captured-solve medians improve 2.19–16.20%. Compiler,
dimension-edge, lifecycle and native feedback gates pass. The actual cold
50k/33-frame run completes with zero hot allocations and source integrity
intact: machine-plus-soil median/p95/p99 **825.92/1023.11/1075.75 ms**, mean
Newton work **519.97 ms**. Observed median is only 0.71% below method 17;
separate evolving trajectories are neither an equivalence check nor a
controlled whole-frame speedup. All measured frames miss 60 Hz. Keep opt-in;
4 ms, rendering, terrain export and long-horizon acceptance remain open.
[Implementation, paired controls and actual evidence](benchmarks/soil/2026-10-11-newton-volatile-rounding/SUMMARY.md).

**Parallel runtime admission (2026-10-11):** opt-in `parallel-newton` now
combines stable parallel candidate selection with scalar-form cell-parallel
core assembly on the protected 256-lane route. Method tag 17 and benchmark
metadata identify it; defaults, portable fallback, buffers and mathematical
limits remain unchanged. Protected GPU checks pass n=1/3/32 with 301 contacts,
inactive poisoned records, selected-contact exclusion, preparation rejection,
byte-exact row/cell/repeat comparisons and an independent f64 assembly oracle.
57 CPU checks, 15 benchmark tests, capture lifecycle and native MPM feedback
pass. The cold 50k/33-frame run completes with zero hot allocations and source
integrity intact: machine-plus-soil median/p95/p99
831.82/1022.52/1121.27 ms; mean Newton GPU work 530.33 ms. Median is 2.01%
below the preceding cached run, but these are separate evolving trajectories,
not equivalence or a controlled whole-frame speedup proof. All measured frames
still miss 60 Hz; 4 ms, rendering, terrain export and long-horizon gates remain
open. Keep the route opt-in.
[Runtime admission and actual evidence](benchmarks/soil/2026-10-11-newton-parallel-runtime/SUMMARY.md).

**Parallel recovery prototype (2026-10-11):** test-only 160-task candidate
scoring preserves serial tie order using existing shared scratch. Candidate
scoring alone improves captured-input medians only 0.53–1.74%, so it was not
promoted. Adding cell-parallel direct/conditioned core assembly produces
2.18–12.10% lower medians versus the cached 256-lane control across eight
captured inputs. Every timed sample retains all five buffers byte-identically
and passes independent physical publication. The initial core implementation
failed exact-output checks; restoring the original scalar projection
accumulator form resolves the eight-case regression without changing limits.
The underlying compiler/driver cause is not established. First-packet timings
straddle a transition and remain archived. Production is unchanged; dimension
edges, runtime integration and whole-frame evidence remain pending.
[Prototype source and evidence](benchmarks/soil/2026-10-11-newton-parallel-release/SUMMARY.md).

**Conditioned recovery caches (2026-10-11):** opt-in `cached-newton` adds
lane-private correction reuse and recovery-local contact constants to the
256-lane protected route. Eight captured problems retain all five output
buffers byte-identically, with independent physical verification on every
timed sample. Two paired comparisons show about 9–10% savings on three
recovery-heavy cases, smaller changes elsewhere; an initial first-case timing
regression is retained in the evidence. Runtime checks pass. The actual cold
50k/33-frame diagnostic completes, but median/p95/p99 are
848.89/1023.98/1173.35 ms: median is **1.57% higher**, not lower, than the prior
wide-only run. Mean Newton GPU time is 544.94 ms versus 546.78 ms. Evolving
trajectories differ, so captured-input equality is not whole-run equivalence.
Keep this route experimental and opt-in; it does not establish a frame-time
gain or satisfy 4 ms/60 FPS. Next optimization should address serial candidate
scoring, rather than repeating cache variants.
[Source, paired samples and actual-run evidence](benchmarks/soil/2026-10-11-newton-delta-cache/SUMMARY.md).

**Wider resident Newton (2026-10-11):** opt-in `wide-newton` uses 256 lanes on
the protected Vulkan route, retaining 64 lanes by default and on the WGSL
fallback. Lane-owned loops, reduction trees, rate projection and scratch scale
with execution width; mathematical budgets and physical tolerances do not.
Both 128/256 variants pass all eight captured inputs with independent physical
publication. Eight alternating same-input timing pairs per problem show
256-lane median solve times 23.52–45.43% below 64 lanes. Runtime method tag 15,
manifest width and pre-allocation device-limit checks identify the new route.
53 CPU checks, 15 benchmark tests, capture lifecycle and native MPM feedback
pass. The cold-start 50k/34-substep/33-frame run completes with zero hot
allocations and source integrity intact: machine-plus-soil median/p95/p99
835.79/1047.35/1186.45 ms; mean Newton GPU work 546.78 ms versus 876.43 ms in
the prior 64-lane run. Observed whole-run median is 30.17% lower, but these
are separate evolving trajectories, not equivalence evidence. Final particle
positions differ by up to 1.742 mm; machine angular-velocity components differ
by up to 0.08484 rad/s over the short run. Keep the feature experimental:
long-horizon accuracy, production visuals/integration and 4 ms/60 FPS acceptance
remain open. Warm starts are not enabled in this measurement.
[Implementation, paired timings and full short-run evidence](benchmarks/soil/2026-10-11-newton-wide-lanes/SUMMARY.md).

**Adjacent-substep history (2026-10-11):** opt-in `adjacent-newton` connects each
owner's restore binding to the immediately preceding owner's certified cache,
without history copies or hot allocations. Two GPU history tests and 52 CPU
checks pass. The unchanged 50k/34-substep workload completes 33 frames with
source integrity intact: machine-plus-soil median/p95/p99
1068.78/1277.12/1552.25 ms. Against the preceding same-parity warm run, observed
median is 2.04% lower but p99 is 11.92% higher; mean Newton GPU work is still
773.05 ms per simulated frame. Separate short trajectories differ, so this is
neither equivalence nor an accepted speedup. Both warm modes remain opt-in and
cold remains default. The legacy last-nontrivial success-capture diagnostic
now rejects warm features rather than archiving a packet without its matching
initial guess. Failure-seed capture remains supported. Next investigate
parallelism inside the resident Newton solve instead of more seed variants.
[Adjacent-history implementation and evidence](benchmarks/soil/2026-10-11-newton-adjacent-history/SUMMARY.md).

**Identity-safe warm Newton experiment (2026-10-11):** opt-in `warm-newton`
retains paired impulses by sorted grid-node/body/wall identity, with timestep,
friction, normal and mass guards. Both alternating owners invalidate at frame
begin; each reuses its own preceding substep, not an arbitrary contact slot.
Existing solve/publication tolerances and work caps remain unchanged. Failure
captures now preserve and hash the exact initial guess for seeded replay.
52 CPU checks and three bounded GPU gates pass; native feedback restores 81
nonzero contacts with maximum normalized momentum error 3.352e-9. The profiled
50k run completes 33 frames, with zero hot allocations and source integrity
intact. Completed machine-plus-soil median/p95/p99 are
1090.99/1290.23/1386.87 ms; mean Newton GPU time is 788.94 ms versus 876.43 ms
in the prior cold run. Median wall time is 8.85% lower in these separate short
runs, not an alternating matched-input speedup claim. Final particle positions
differ by up to 1.522 mm; maximum machine angular-velocity component difference
over the 33 frames is 0.05517 rad/s. Do not claim trajectory equivalence or
promote the feature from this evidence. Cold starts remain the default. Full
trajectory, persistence, production visual and 4 ms/60 FPS gates remain open.
[Warm-history implementation and comparison](benchmarks/soil/2026-10-11-newton-warm-history/SUMMARY.md).
**Planned against:** `aecdeab94ec4a26970edfe830f79a17a182e7aeb`.
**Date:** 2026-10-08.
**Priority:** make the existing solvers and their live integration fast before expanding soil features.

**Contact sub-stage attribution (2026-10-11):** added retained nested timings
without changing the seven aggregate MPM stages or double-counting totals.
Profiled and unprofiled four-step native runs preserve all particle checkpoint,
rate, status and wrench words exactly; stepping still has no buffer allocations
or inner readbacks. The fresh 50k/33-frame diagnostic completes with source
integrity intact. Across 30 measured frames, mean contact preparation / Newton
solve / external publication-recovery are **37.13 / 876.43 / 3.69 ms**. Newton
accounts for **95.54%** of contact GPU time; it includes integrated physical
certificates. Completed machine-plus-soil median/p95/p99 are
1196.93/1486.26/1720.39 ms. This establishes the dominant optimization target,
not a speedup or real-time acceptance. Preparation also exceeds the soil budget
by itself. The next implementation must substantially reduce Newton work while
retaining the full coupling contract and acceptance requirements.
[Nested profiling evidence](benchmarks/soil/2026-10-11-contact-stage-profile/SUMMARY.md).

**Stagnation recovery (2026-10-11):** sparse iteration traces show the new
219-contact failure accepting 16 steps for only 1.2% residual reduction.
Once smoothing is zero, two residual-stagnating steps now tighten line-search
decrease to 1%, invoking the existing release/damped recovery rather than
accepting repeated tiny improvements. Physical tolerances and all work caps
are unchanged. The captured failure now passes independent publication in
59 of 64 allowed steps, with physical error 1.91e-7 and three identical resets.
All seven earlier cases retain all 35 buffers byte-identically. Runtime method
13 (WGSL) / 14 (protected) records the policy. Protected lifecycle/feedback and
all eight captured inputs pass. The fresh profiled actual run completes all
33 frames with 50k particles, zero hot allocations and source integrity intact;
median/p95/p99 completed machine-plus-soil work is 1172.32/1692.17/1723.88 ms.
Contact projection accounts for 99.49% of MPM GPU time across all 30 measured
frames. Two offline replay views were rendered and inspected; they show recorded
analytic tractor/tine shapes and particle markers, not a production asset,
continuous trajectory, terrain surface or live rendering acceptance. The next
implementation priority is contact-coupling cost, not more unchanged short-run
correctness repeats. This is not full-plan acceptance.
[Stagnation recovery evidence](benchmarks/soil/2026-10-11-newton-stalled-release/SUMMARY.md).

**Product-boundary optimization (2026-10-11):** protected compiler V2 retains
only the low limb at paired-product materialization boundaries, preserving the
computed-root and normal-rate boundaries. All six arithmetic suites pass; all
35 buffers across seven captured solves and every common-solve Newton prefix
match the previous runtime exactly. Eight alternating GPU-timed pairs measure
17.855488 ms for the prior protected runtime versus 17.451008 ms for the
candidate (2.27% lower single-solve time). Method 12 identifies the new runtime;
historical method 11 remains readable. Runtime lifecycle/feedback and all seven
captured cases pass. The profiled actual run commits frames 0–20, then reaches
the unchanged Newton cap at frame 21; source integrity passes. Its 18 measured
frames attribute 99.47% of MPM GPU time to contact projection (mean 881.45 ms),
versus 4.72 ms for other MPM work and 261.52 ms for machine stages. Contact
projection includes preparation, Newton and grid publication. The new captured
failure also reproduces with the previous both-limb compiler; this is an open
convergence case, not evidence that dropping the high-limb boundary caused it.
No whole-frame speedup, complete trajectory or real-time acceptance is claimed.
[Boundary optimization evidence](benchmarks/soil/2026-10-11-newton-quotient-boundary/SUMMARY.md).

**Publication-projection repair (2026-10-11):** the captured frame-13 failure
was a false sliding-friction rejection caused by treating a rounded normal as
exactly unit length. Paired double-cross projection using the raw normal's squared
length fixes certification without changing tolerances or represented outputs.
All 486 projection cases and seven publication suites pass. The new captured
solve now certifies in 7 steps; all seven retained problems pass and the prior
six cases keep all 30 output buffers byte-identical. Runtime lifecycle/feedback
passes, and the fresh actual 50k run completes all 33 requested frames within
the unchanged deadline, with zero hot buffer allocations. Median/p95/p99
completed machine-plus-soil times are 1186.87/1474.68/1769.63 ms; every measured
frame misses 60 Hz. This is short-run completion, not 4 ms/60 FPS, long-horizon
or rendered acceptance. Performance work is next.
[Publication repair evidence](benchmarks/soil/2026-10-11-newton-publication-contact/SUMMARY.md).

**Protected-arithmetic runtime checkpoint (2026-10-11):** the faster Newton
compiler is now available through the native `protected-newton` feature.
Conditioned Newton selects method 11 only on a Vulkan engine with explicitly
enabled passthrough; other engines retain WGSL. Pinned Naga generation and
in-process SPIRV-Tools validation replace the private subprocess-based compiler.
The production route matches all 30 output buffers across six captured inputs,
passes lifecycle/empty-contact and native MPM feedback checks, and preserves
the original tolerances and work limits. The 2.42x prior speedup is still
single-solve evidence, not frame acceptance. The fresh 50k run completes frames
0–12 at roughly 0.97–1.34 seconds per measured frame, then faults in Newton
globalization at frame 13. Both WGSL and protected arithmetic reproduce the
failure and archived state exactly on that input; the publication guard rejects
contact 84 despite a small iterate residual. Publication diagnosis, the 33-frame
diagnostic and full-plan acceptance remain open.
[Runtime integration evidence](benchmarks/soil/2026-10-11-newton-protected-runtime/SUMMARY.md).

**Confirmed acceptance inputs (2026-10-10):** the user has no measured soil or
traction data. The initial release must therefore be explicitly **physically
uncalibrated**; numerical verification does not establish real-soil accuracy.
Retain the confirmed RTX 4080 target: one tractor plus implement, 50,000 active
soil particles, 60 FPS, and soil work ≤4 ms per 1/60 second of simulated time.
These are acceptance requirements, not achieved results. Measured-data
calibration remains a separate future gate; it must not prevent implementation
or numerical/performance verification of the initial uncalibrated release.

**Iterative-first candidate checkpoint (2026-10-11):** added an opt-in hybrid
that tries eight GMRES iterations before direct elimination and allows only
the remaining Krylov budget after direct rejection. The prefix is provisional;
the runtime owner is unchanged. A test-only capture now retains successful
published inputs separately from failures. The actual first 50k tractor frame
completed and produced two successful packets. On one packet, the hybrid and
current runtime method produce identical reports (8 Newton steps, 24 Krylov
iterations, 8 trials). Three cold-state repetitions of the difficult frame-16
input also pass unchanged publication checks, with 59 Newton steps and 774
trials. Timings are diagnostic only: common-case samples vary substantially,
and the hard-case hybrid median is 1916.60 ms. Neither runtime promotion nor
the 4 ms target is accepted. A singular-local GPU probe confirms prefix
exhaustion at budgets 1/8 without restarting the budget; budget 64 converges
after 18 total iterations with independent relative residual 2.03e-7. Hybrid
reuse, scratch poisoning, sticky upstream failure and reset checks also pass.
[Hybrid implementation and successful-input evidence](benchmarks/soil/2026-10-11-newton-hybrid/SUMMARY.md).

**Whole-frame follow-up:** experimental hybrid integration completed frames
0–15 but failed frame 16 at the unchanged 64-Newton-step limit. Source integrity
passed; the runtime switch was reverted. Both CPU and GPU direct-only replay
also fail the newly captured input, so improving linear-solve speed alone does
not resolve this nonlinear failure. A default-off adaptive-smoothing reference
experiment passes that input and all four retained regression packets. The
explicit GPU adaptive-direct candidate (method 8) passes the new input in
59 Newton steps with three identical cold-state reports, and the preceding
frame-16 input in 51 steps. Independent physical errors are 6.60e-10 and
6.12e-9 respectively; unchanged publication gates pass. Remaining GPU
regressions and full integration are still open. These are convergence results,
not runtime promotion or 4 ms performance acceptance.
[Actual failure, rollback and adaptive experiment](benchmarks/soil/2026-10-11-newton-hybrid-integration/SUMMARY.md).

**Adaptive-hybrid follow-up:** combined adaptive smoothing with the bounded
iterative-first solve (method 9). All five retained packets pass three
zero-reset GPU repetitions, including the common report unchanged at 8 Newton
steps / 24 Krylov iterations. Native Newton owners now reuse immutable
source/entrypoint/layout-keyed programs without sharing state buffers; a
five-packet batch completes in 62.41 s under the existing 90 s command bound.
Lifecycle, empty/invalid-mass and diagnostic-limit checks pass. However, the
actual tractor still fails frame 16, now on parity 0 at 64 Newton steps. The
runtime selection was reverted to method 4; the cache and opt-in candidate
remain. Whole-scene construction was 130.49 s versus the preceding 130.15 s
sample, so no startup speedup is claimed. Full integration and 4 ms acceptance
remain open.
[Program reuse, replay corpus and actual negative evidence](benchmarks/soil/2026-10-11-newton-adaptive-hybrid/SUMMARY.md).

**Conditioned floor-release reference:** a new default-off CPU candidate
accounts for the linearized response of the other contacts while solving a
selected contact nonlinearly, then applies the coordinated correction to all
contact states. This resolves the newest parity-0 failure in **13 CPU Newton
steps**, versus exhausting 64 before; all six retained packets pass unchanged
physical gates. Independent dense algebra checks cover 3/8/32 DOFs and exact/
prepared responses. The original floor-release rule and eight-pivot ceiling
remain; an unhelpful active-wall-selector experiment was removed. This is
reference evidence only; see the subsequent GPU checkpoint below. Full
trajectory and 4 ms acceptance remain unverified, and runtime selection is
unchanged.
[Conditional response derivation, checks and GPU port requirements](benchmarks/soil/2026-10-11-newton-active-wall/SUMMARY.md).

**Conditioned floor-release GPU checkpoint (2026-10-11):** opt-in method 10
coordinates all contact states using four verified machine-sized solves in the
existing condensed workspace. An independent dense oracle passes at 3/8/32
DOFs, including sparse slots, singular selected-block exclusion and singular
unselected-block rejection. All six captured inputs pass three cold-state
replays at unchanged physical/publication gates. The newest parity-0 failure
converges in **12 Newton steps**, versus exhausting 64; the prior parity-1
failure takes **13**, and the common packet retains its exact 8-step report.
Frozen-solve timings are still far above 4 ms and exclude whole-frame work.
This is an explicit candidate, not runtime promotion or trajectory acceptance.
[GPU implementation, independent algebra and six-packet evidence](benchmarks/soil/2026-10-11-newton-conditioned-gpu/SUMMARY.md).

**Conditioned runtime follow-up (2026-10-11):** method 10 is now selected in
the existing experimental, opt-in Newton machine/soil path, with matching
geometry, direct-MPM and rigid-world memory/device preflight. Native paired
MPM feedback passes. The actual 50k tractor completes **frames 0–20**, passing
the preceding frame-16 solver failure, then hits the unchanged **180-second**
setup-inclusive deadline. Construction takes **128.67 s**; completed measured
frames take **1.85–3.95 s**. The source check passes. This implementation is
retained rather than restoring a known earlier failure, but the requested
33-frame run, full trajectory and real-time gates remain incomplete. Default
non-Newton soil routes are unchanged; no production/performance acceptance is
claimed. Next work targets measured cost, not larger limits.
[Integrated prefix, deadline evidence and current routing decision](benchmarks/soil/2026-10-11-newton-conditioned-integration/SUMMARY.md).

**Paired-arithmetic correction and private speedup (2026-10-11):** prefix and
contact-level captures traced the earlier state differences to incorrect low
parts when squaring computed roots in the ordinary reference path. Explicit
root materialization fixes all 225 failures in an 8193-case CPU-referenced
regression; `make test-soil-newton-root` exercises the ordinary runtime helper.
The corrected WGSL and private protected-arithmetic candidate pass all six
captured solves and match all 30 exposed buffers exactly. Every common-input
prefix through convergence also matches. Eight alternating measured pairs give
**43.125248 -> 17.833984 ms** median solve spans (**58.65% less / 2.42x faster**),
with unchanged physical and hot-resource gates. Root materialization entered
the existing runtime at this checkpoint; the faster compiler was then test-only.
The protected-runtime checkpoint above records subsequent integration and
lifecycle checks. Neither checkpoint establishes whole-frame or 4 ms/60 FPS acceptance.
[Original investigation](benchmarks/soil/2026-10-11-newton-precise-pairs/SUMMARY.md),
[root fix, exact parity and paired timing](benchmarks/soil/2026-10-11-newton-precision-prefix/SUMMARY.md).

**Direct Newton integration checkpoint (preceding, 2026-10-11):** added verified paired
direct Newton solves through a machine-sized core, with bounded GMRES fallback
for singular/inaccurate local elimination. Trial Jacobians are deferred until
an accepted base. Independent GPU directions and a singular-local fallback
fixture pass; all three cold captured problems pass unchanged publication gates.
The hardest replay has median **1677.63 ms**, versus **1911.36 ms** for the
same direct method with full trial Jacobians and the preceding **3796.72 ms**
damping-only candidate. The latter is a successive-session comparison; the
smaller cases did not improve in their single samples. No broad percentile or
real-time acceptance is implied. An experimental Newton geometry run used this
candidate with matching storage/device preflight; ordinary
non-Newton defaults remain unchanged. **The runtime switch was subsequently
reverted:** the candidate completed frames 0–17, including the previous frame-16
failure, but common frames were slower (**2.42–4.23 s** measured) and the command
hit its unchanged **180-second setup-inclusive deadline**. Construction took
119.36 s; source integrity passed. The direct method remains available only by
its explicit low-level constructor. The geometry owner again uses the preceding
continuation-plus-release method and retains its known frame-16 limitation.
Next work must preserve cheap iterative solves on easy states rather than
unconditionally paying for direct elimination. Full runtime and real-time
acceptance remain open.
[Direct solver, controls and integration evidence](benchmarks/soil/2026-10-11-newton-direct-core/SUMMARY.md).

**GPU condensed damping checkpoint (preceding, 2026-10-11):** implemented an explicit
opt-in paired small-core damped solve with original-operator verification and
bounded LSQR fallback. Independent dense-direction GPU checks pass through 32
machine DOFs, and all three cold failure packets pass without changing accuracy,
iteration or timeout gates. On the hardest packet, three reset-to-zero GPU
timestamp samples have median **4040.49 ms** for the preceding method and
**3796.72 ms** for condensation, about **6% lower** in this small comparison.
These are isolated solves, not frame times or percentile acceptance. The new
method adds a 33,280-byte core and 432 bytes per contact; cold pipeline setup is
about 34–36 seconds. It remains opt-in: the geometry runtime route is unchanged,
the integrated frame-16 timeout remains unresolved, and 4 ms/60 FPS acceptance
is not achieved. Its linear counter mixes Krylov iterations with direct solves
and cannot substitute for timestamps.
[Implementation, controls and GPU evidence](benchmarks/soil/2026-10-11-newton-structured-damping/SUMMARY.md).

**Local-release runtime checkpoint (2026-10-10):** fixed a GPU polynomial-root
accumulator defect found by isolated production-function probes, and consolidated
candidate evaluation. Cold Newton pipeline setup drops from roughly **68–69 s
to 27 s** in these runs; this is not a soil-frame speedup. Both previous cold
packets still pass. The experimental Newton geometry route now uses this method
with matching storage preflight; ordinary non-Newton defaults are unchanged.
The actual tractor advances through frame 15, then **frame 16 times out** at the
unchanged completion deadline. Completed measured frames cost **1.82–3.42 s**.
A diagnostic-only capture of the new hard state passes separately on CPU (62
steps) and GPU (**59 steps / 2473 inner iterations**, physical **1.66e-7**).
That does not prove the full frame converges or meets its deadline. Runtime and
4 ms/60 FPS acceptance remain open; optimize work, not tolerances or deadlines.
[Root fix, setup reduction, integration failure and new replay](benchmarks/soil/2026-10-10-newton-release-integration/SUMMARY.md).

**GPU local sliding release (preceding checkpoint, 2026-10-10):** an explicit continuation-plus-release
constructor now solves the frame-13 packet from zero on the RTX 4080 in **35
outer steps / 586 inner iterations**, with independent physical error **8.23e-8**
and zero publication faults. No CPU seed is used. The fallback enumerates bounded
local sliding branches before damping; it shares the existing outer budget and
retains the final physical gate. The older cold packet and GPU lifecycle,
empty/fault, diagnostic-ceiling and capture-provenance checks also pass. Cold
pipeline compilation takes roughly 68–69 seconds and needs attention separately
from resident solve performance. This supersedes the reference-only limitation
below, not the outstanding runtime or throughput gates. Runtime promotion and
4 ms/60 FPS acceptance remain open.
[GPU implementation and verification checkpoint](benchmarks/soil/2026-10-10-newton-local-release/SUMMARY.md).

**Local sliding-release reference (preceding checkpoint, 2026-10-10):** the frame-13 packet now passes
from zero in the CPU reference: **39 outer steps**, unchanged physical error gate
(measured **7.69e-9**). Bounded exact local sliding branches release two floor
constraints, temporarily increasing the global residual before Newton converges.
The original GPU publication accepts the CPU-generated paired state with **zero
GPU Newton steps**. This is a reference/representation result, **not a GPU solver
fix**: the local release still needs a GPU implementation, and live promotion,
4 ms throughput and full-plan acceptance remain open.
[Reference implementation and positive/negative evidence](benchmarks/soil/2026-10-10-newton-conical/SUMMARY.md).

**Cold-branch diagnosis and provenance (2026-10-10):** the new frame-13 failure
persists with independently inverted f64 responses. Exact-wall projection and
a bounded friction path also fail; their experiments were archived and removed
from active code. Focus shifts to friction/contact branch handling, not another
GPU-precision or smoothing-parameter retry. Failure captures now carry the
actual solver method in header word 7 and export it accurately; tagged GPU
round-trip and constructor checks pass with unchanged buffer budgets. Runtime
and real-time acceptance remain open.
[Evidence and next algorithm target](benchmarks/soil/2026-10-10-newton-cold-branches/SUMMARY.md).

**Cold-start continuation checkpoint (2026-10-10):** the previously failing cold
captured GPU solve passes using explicit smoothing continuation: **20 Newton
steps / 145 Krylov iterations**, independently checked physical error **2.95e-7**,
with unchanged publication and iteration bounds. CPU cold controls and GPU
lifecycle/empty/fault/work-limit checks also pass. This does **not** establish
runtime acceptance: an integration trial completes only 13 frames before GPU
completion timeout, with completed frames roughly **1.76–2.67 seconds** and
100.3-second construction. No real-time speedup or full-plan completion is
claimed. Runtime promotion was reverted; continuation remains an explicit
standalone constructor. A bounded diagnostic captured the frame-13 failure;
that new packet also fails tested f64 cold controls.
[Implementation and negative runtime evidence](benchmarks/soil/2026-10-10-newton-cold-continuation/SUMMARY.md).

**Latest bounded residual correction (2026-10-10):** the difficult warm captured
GPU solve now passes under normal bounds: **16 Newton steps / 171 Krylov
iterations**, independent physical error **1.45e-7**, publication accepted.
Correction cycles share the original 64-iteration limit per linear solve;
the linear 1e-5 and physical 1e-6 gates are unchanged. Cold-start diagnostic
replay still fails at physical error **1.99e-5**, so there is no runtime promotion
or claimed tractor speedup. Next: establish a verified cold-start path while
retaining all performance, particle, rendered and long-horizon requirements.
[Correction, normal-bound warm pass and cold failure](benchmarks/soil/2026-10-10-newton-refinement/SUMMARY.md).

**Previous Newton scheduling repair (2026-10-10):** the Fischer outer loop now
uses one evaluation call site rather than five. Full pipeline creation completes
in **16.6 s**, and the paired 65-contact lifecycle passes. The difficult warm
replay still fails the 512-inner diagnostic at physical error **7.02e-5**.
The eighth linear solve is isolated: GPU stagnates at **9.55e-5**, while f64
solves the same paired coefficients in 12 iterations near **3.02e-9**. Exact
prepared-response CPU continuation also converges, ruling out response
reconstruction as the cause. No runtime promotion or real-time acceptance.
[Schedule, setup timings and isolated remaining failure](benchmarks/soil/2026-10-10-newton-fischer-schedule/SUMMARY.md).

**Previous Jacobian precision repair (2026-10-10):** an f64 control proves that
rounding even correctly evaluated coefficients to f32 can exceed the linear
gate. The experimental Fischer variant now stores/evaluates paired Jacobian
coefficients. Initial and reference-one-step seeded GPU linear probes pass,
the latter at **9.12e-6** versus **5.93e-5** for correctly rounded coefficients.
Full Newton setup exceeds its unchanged 90-second command limit; this repair
is not nonlinear, runtime or performance acceptance. Next: isolate/reduce full
pipeline setup cost, then verify convergence under the original bounds.
[Paired coefficients, controls and setup failures](benchmarks/soil/2026-10-10-newton-fischer-jacobian/SUMMARY.md).

**Previous paired precision repairs (2026-10-10):** the standalone GPU Fischer
variant now retains paired contact residuals and Newton directions. The first
captured linear step passes independent assembled and reference checks; the
second passes its assembled gate but still differs from the f64 reference.
Bounded warm replay improves to physical error **1.50e-5**, still above **1e-6**.
No runtime promotion, full-tractor speedup or acceptance is claimed. Next: isolate
the remaining coefficient/linearization discrepancy without increasing limits.
[Comparisons, precision fixes and negative results](benchmarks/soil/2026-10-10-newton-fischer-linear/SUMMARY.md).

**Previous complementarity prototype (2026-10-10):** CPU Fischer-Burmeister
continuation solves the stalled packet in 16 steps, with independent physical
error **1.79e-10**; the existing GPU publication certificate accepts that seed
unchanged. The new standalone GPU variant passes smaller coupled/lifecycle
checks but **does not solve the difficult packet**, and cold CPU startup also
fails. It is not wired into the tractor runtime. Next: compare its GPU linear
system/direction with the successful f64 continuation before promotion.
[Reference solution, GPU prototype and retained failures](benchmarks/soil/2026-10-10-newton-fischer/SUMMARY.md).

**Previous stalled-input capture (2026-10-10):** an explicit diagnostic-only inner
work ceiling now stops the frame-15 solve before device quarantine and preserves
its inputs. Exact replay matches all report/candidate bytes. The candidate has
physical error **0.0847**; CPU f64 also fails, and no separating-wall pivot is
eligible. This isolates a different convergence failure, not repaired runtime
stability. Normal limits and numerical gates are unchanged; diagnostic runs
cannot establish performance acceptance.
[Bounded capture, exact replay and failed controls](benchmarks/soil/2026-10-10-newton-work-capture/SUMMARY.md).

**Previous recovery ordering (2026-10-10):** wall active-set recovery now runs
after ordinary Newton/backtracking fails but before damped least-squares trials.
On the saved failure this reduces inner iterations **785 → 96**, retaining
independent physical error **2.16e-7**. Six wall-direction regressions and runtime
checks pass. An unconditional earlier pivot was rejected after a regression.
The actual run still times out at **frame 15**; completed measured frames cost
**632--735 ms per 1/60 s**, not 4 ms. No integrated improvement is claimed.
[Guarded ordering, rejected variant and actual timeout](benchmarks/soil/2026-10-10-newton-early-release/SUMMARY.md).

**Previous nonlinear recovery (2026-10-10):** first-failure device packets now
retain exact prepared Newton inputs and candidate state. Cold replay reproduced
the frame-15 line-search failure bit-for-bit; a CPU f64 control also stalled.
A bounded, opt-in wall active-set release fixes this captured case in 10 outer
steps with independent physical error **2.06e-7**, below the unchanged 1e-6 gate.
It runs only after ordinary backtracking/damping fails and does not certify a
candidate without the existing momentum, contact, wall and energy checks.
The changed-source actual run still stops at **frame 15**, now on the unchanged
GPU completion timeout (device quarantined), with completed measured frames at
**629--739 ms per 1/60 s**. Source integrity passes. Full integrated trajectory
and performance acceptance remain open; the captured fix is not a runtime fix.
[Captured failure, controls and recovery](benchmarks/soil/2026-10-10-newton-solve-capture/SUMMARY.md).

**Previous strict-wall publication (2026-10-10):** bounded projection now produces
exactly feasible represented wall velocities inside Newton publication, with
wall reactions recomputed before independent momentum, contact and energy
checks. Six-face/adversarial checks and resident/standalone parity pass. The
actual 50k tractor passes the prior wall fault and completes **15 frames**, then
rejects Newton line search at frame 15 after 16 Newton steps / 2506 inner
iterations / 61 trials. Preparation succeeded; later upstream faults must not
be mistaken for its cause. Completed measured frames cost **628--729 ms per
1/60 s**, not 4 ms. Next: retain and replay that first failed nonlinear solve;
full trajectory, performance, rendering and long-horizon acceptance stay open.
[Strict-wall implementation and actual nonlinear failure](benchmarks/soil/2026-10-10-newton-wall-publication/SUMMARY.md).

**Previous preparation correction (2026-10-10):** retained first-failure GPU
packets exposed two tractor response-accuracy failures. More full corrections
fixed one but stalled on the other; selective correction of uncertified
components makes both pass the unchanged 1e-6 backward-error gate, with independent
errors below 3.55e-8. Focused reference, sparse, coupled and fault checks pass.
The actual 50k tractor now completes **13 frames**, then fails strict fixed-wall
velocity coverage at frame 13: a published Y velocity of -4.24e-10 passes the
Newton wall tolerance but violates the consumer's exact sign requirement.
Preparation/solve/publication status is otherwise successful. Completed measured
frames cost **631--734 ms per 1/60 s**, not 4 ms. Next: reconcile represented-wall
publication and certification without relaxing coverage. No full-run, trajectory
or real-time acceptance is claimed.
[Failure packets, selective correction and actual evidence](benchmarks/soil/2026-10-10-newton-preparation-capture/SUMMARY.md).

**Previous runtime wiring (2026-10-10):** added an explicit prepared-Newton
geometry/MPM/rigid-world path and tractor benchmark selector, leaving defaults
unchanged. Small native MPM feedback, fault latching and clearing pass. The real
50k tractor **fails frame 0**: latched diagnostics identify response preparation
fault 4 before any Newton iteration, not the older wide-case convergence issue.
Both bounded actual runs terminate with source checks passing; no runtime speed
or trajectory acceptance is claimed. Next: capture and fix the first failing
preparation input without relaxing numerical gates.
[Runtime integration and retained actual failure](benchmarks/soil/2026-10-10-newton-runtime/SUMMARY.md).

**Previous device preparation (2026-10-10):** implemented retained GPU mass
factorization, contact-response solves, two residual corrections and a 1e-6
backward-error check. Preparation and resident Newton now run in one submission
without CPU response uploads. All three captured tractor inputs pass physical
checks in four Newton steps, with 15/73/70 inner iterations. Small coupled,
sparse, lifecycle and fault tests pass. Initial unrefined preparation changed
branches and failed a small fixture; that failure is retained, not hidden.
Actual geometry/runtime wiring, preparation timing, wide convergence and full
4 ms/50k/60 FPS acceptance remain open.
[Device preparation, correction and coupled evidence](benchmarks/soil/2026-10-10-newton-device-preparation/SUMMARY.md).

**Previous parallel publication (2026-10-10):** distributed independent machine
momentum checks across DOF lanes, preserving energy accumulation order and the
1e-6 certificate. Matched captured solve medians improve from
**25.62 / 71.45 / 70.37 ms** to **18.64 / 65.89 / 64.79 ms**, with bit-identical
outputs, rates and statuses. Reference/adversarial publication and all 32
momentum-lane rejection checks pass. Device preparation, full runtime acceptance,
wide convergence and the 4 ms target remain open.
[Parallel certificate and paired evidence](benchmarks/soil/2026-10-10-newton-parallel-publication/SUMMARY.md).

**Previous guarded method switch (2026-10-10):** added opt-in
`new_stagnation_pairs`, combining chunked products with true-residual-checked
GMRES stagnation fallback. Matched captured friction medians fall from
**160.25 / 159.15 ms** to **72.40 / 71.31 ms**, with unchanged physical checks
and four accepted Newton steps. Normal median changes from **24.46 to 25.31 ms**;
existing routes/defaults are retained. Sparse lifecycle/fault checks pass. An
inconsistent linear-system probe rejects a false zero recurrence estimate using
the true residual. Wide convergence still fails, and neither full runtime
integration nor the 4 ms target is achieved.
[Method-switch implementation, checks and retained failures](benchmarks/soil/2026-10-10-newton-stagnation/SUMMARY.md).

**Previous chunked products (2026-10-10):** split contact-to-machine projections
across the existing 64 lanes, without new buffers or dispatches. Matched captured
solve medians improve from **26.72 / 179.86 / 177.88 ms** to
**25.28 / 164.69 / 163.61 ms**, with unchanged iteration counts and passing
physical output checks. Independent product and damped-system checks pass.
The wide fixture still fails the unchanged convergence limit; the new variant
remains **test-only**, not production-ready. These are solve-only measurements,
not 50k-particle frame acceptance; the 4 ms target remains unmet.
[Chunked projection evidence and retained failure](benchmarks/soil/2026-10-10-newton-chunked-products/SUMMARY.md).

**Previous cooperative reduction (2026-10-10):** added an explicit
`new_cooperative_pairs` route for six-level paired Krylov reductions. It reduces
matched friction medians from **216.31 / 214.25 ms** to **184.25 / 182.24 ms**,
but increases the normal capture from **24.50 to 25.85 ms**. Therefore
`new_sorted_pairs` retains the preceding serial-reduction behavior; no blanket
replacement or production selection change is made. All captured physical checks
pass. The wide case still fails convergence, and the 4 ms target remains unmet.
[Numerical checks, measured tradeoff and explicit routing](benchmarks/soil/2026-10-10-newton-tree-reduction/SUMMARY.md).

**Previous resident work reduction (2026-10-10):** cached each contact's local
transpose projection once and screened GMRES direction recovery while retaining
explicit true-residual acceptance. Matched-input GPU medians improve from
**27.83 / 256.01 / 250.72 ms** to **24.43 / 220.77 / 218.73 ms**, with complete
physical checks and unchanged iteration counts. Fixed a separate sparse-index
scratch-lane alias; deliberately colliding contact indices now pass in both
resident modes. Wide-case execution no longer times out, but still fails its
64-step convergence limit. The 4 ms target and full runtime acceptance remain
unmet. Next: cooperative reductions and wide-case convergence.
[Cached transpose, sparse-lane fix and verification](benchmarks/soil/2026-10-10-newton-transpose-cache/SUMMARY.md).

**Previous arithmetic comparison (2026-10-10):** added opt-in magnitude-sorted
FP32 pair summation. On eight alternating measured pairs, median frozen solve
times are **27.82 / 260.44 / 255.16 ms**, versus **36.45 / 373.18 / 365.59 ms**
for baseline on the same three captures, with identical iteration counts and
full physical output checks. Arithmetic parity covers 4,100 cancellation and
exponent-range cases. Native FP64 was slower and is retained only in tests.
The wide fixture still times out; neither general convergence nor the 4 ms
target is achieved. Production selection is unchanged. Next: parallelize the
serialized products/reductions and resolve the wide branch failure.
[Arithmetic alternatives and matched-input evidence](benchmarks/soil/2026-10-10-newton-native-f64/SUMMARY.md).

**Previous damped-solver stopping fix (2026-10-10):** resident LSQR now checks the
regularized normal-equation residual, screened by its recurrence estimate but
always explicitly verified before convergence. All three captured outputs still
pass the unchanged physical checks in four Newton steps. Friction inner counts
drop from 140/140 to **105/102**. A frozen old-friction timing diagnostic measures
**373.18 ms median**, still far above the entire 4 ms soil budget; this is not a
paired speedup comparison or integrated acceptance. The 32-DOF fixture still
fails convergence. Production selection and limits remain unchanged.
[Stopping correction, independent checks and retained failures](benchmarks/soil/2026-10-10-newton-stationarity/SUMMARY.md).

**Previous certified resident solve (2026-10-10):** added independent publication
checks for represented impulses/velocities, paired machine rates, momentum,
strict normal separation, friction, walls and energy. All three saved tractor
inputs now pass in **4 Newton steps**; resident and standalone publication are
bit-identical and pass independent f64 output checks. Preserving paired impulse
addition before rounding soil velocity fixes the initial publication failures
without weakening the 1e-6 gates. The certificate now stops the resident loop.
**Performance is still unacceptable:** frozen solve-only GPU medians are
**38.48 ms** (normal) and **487.72 ms** (old friction), excluding preparation,
particles and rendering. These are diagnostics, not full-workload acceptance or
a paired speedup comparison. Production defaults remain unchanged; the wide
fixture, arithmetic/reduction speed and runtime integration remain open.
[Publication, capture evidence and measured performance gap](benchmarks/soil/2026-10-10-newton-publication/SUMMARY.md).

**Previous nonlinear prototype (2026-10-10):** implemented paired impulse updates,
bounded Newton/backtracking and damped LSQR fallback inside a **single GPU
dispatch**, without inner host readbacks. The 65-contact fixture solves in
**5 Newton steps**, including sparse 65-of-4096 reuse, with independent physical
error **7.99e-9**. Three smaller scenes and the seven-contact scene also pass.
**Acceptance remains incomplete:** the 32-DOF fixture reaches its 64-step limit;
all three actual snapshots reject at the nonlinear line-search gate. Their
physical-error diagnostics alone do not establish full publication acceptance.
An adaptive-damping variant regresses and is reverted. Next: diagnose the saved
later wide-state linearization and implement independent physical publication.
No runtime selection or 4 ms/60 FPS improvement is claimed.
[Prototype, preserved failures and next evidence](benchmarks/soil/2026-10-10-newton-resident-nonlinear/SUMMARY.md).

**Previous resident linear solve (2026-10-10):** bounded flexible GMRES and an
independent undamped LSQR route now run entirely inside a GPU workgroup, with
retained buffers and two dispatches per solve. GMRES solves the 65-contact
fixture in **6 iterations** (true relative residual **4.29e-7**); the initial
linear systems from all three saved tractor inputs pass in **3 iterations**
each (worst **3.42e-6**). A per-contact accumulator reset fixes the incorrect
resident matrix product on a lane's second contact. These are frozen linear
directions, **not full nonlinear solves or measured runtime speedups**.
Next: bounded nonlinear updates, backtracking/regularization and independent
physical publication checks, then runtime integration. Production defaults,
the 1e-6 physical certificate and 4 ms/60 FPS target remain unchanged.
[Resident solver, regression and scoped evidence](benchmarks/soil/2026-10-10-newton-resident-krylov/SUMMARY.md).

**Previous resident Newton operator (2026-10-10):** the GPU now evaluates the
global residual, contact derivative blocks and matrix-free forward/transpose
products in retained buffers. RTX 4080 checks pass against the f64 reference,
including the three actual input snapshots, sparse slots, sticky faults, reset,
high/low cancellation and exact budget/hot-resource checks. Worst scaled
residual discrepancy at saved terminal impulses is **3.69e-7**. This is an
operator, **not yet a complete GPU Newton solve or runtime acceleration**.
Next: resident bounded Krylov/Newton iteration and unchanged physical publication
gates. Production defaults and the 4 ms/60 FPS requirements remain unchanged.
[GPU implementation, verification and remaining solver work](benchmarks/soil/2026-10-10-newton-gpu-operator/SUMMARY.md).

**Previous global-Newton reference (2026-10-10):** implemented a test-only f64
matrix-free coupled solver with analytic Jacobian/transpose products, block
preconditioning and bounded step regularization. All three saved actual inputs
now solve **from zero impulses in 4/11/8 Newton steps**, passing independent
physical checks at the existing 1e-6 bound. Seven physical fixtures, derivative
and linear-solver checks pass. This is CPU reference evidence, not GPU/runtime
selection or a throughput result. Next: GPU operator and resident solver port,
explicit resource accounting, then unchanged integrated acceptance gates.
[Reference, preserved failures and GPU implementation handoff](benchmarks/soil/2026-10-10-global-newton/SUMMARY.md).

**Previous full-state prototype result (2026-10-10):** two-history mixing passes
the prior saved failures and component checks, but the actual 50k workload
still fails after 14 committed frames. The new friction input fails with
two-history mixing yet passes the original rate-only accelerator. A bounded
per-sweep choice between those directions also fails the new input and regresses
a 32-DOF fixture. **Neither variant is promoted; production defaults are restored.**
Captured inputs and resource-accounting fixes are retained. Actual frames take
751.723–1375.023 ms; the 4 ms/60 FPS and full-trajectory gates remain unmet.
[Prototype, integrated regression and rejected direction selection](benchmarks/soil/2026-10-10-full-state-mixing/SUMMARY.md).

**Previous coupled-sweep diagnosis (2026-10-10):** exact per-sweep replay shows
the contact-96 normal defect decaying slowly, not remaining exactly stationary;
the last nine acceleration proposals reject. Two paired-merit arithmetic changes
solve that capture but regress the eight-DOF/seven-contact fixture, so neither
is promoted. A new CPU regression demonstrates that the accelerator's rate-only
metric can miss opposing soil residuals that cancel at the machine. The next
bounded prototype will use full machine/contact/wall residual mixing with
explicit history/budget accounting, not further local or merit threshold tuning.
This is a tested limitation and a proposed repair direction, not a verified
cause of every failure or an accepted solver fix. Production defaults and all
acceptance requirements remain unchanged.
[Trace, rejected candidates and full-state prototype specification](benchmarks/soil/2026-10-10-coupled-sweep-progress/SUMMARY.md).

**Previous normal diagnosis (2026-10-10):** contact 96's local solve is accurate;
subsequent coupled updates cause the failed normal certificate. Always-on local
Newton refinement solves the captured case in 58 sweeps but regresses the
eight-DOF/seven-contact fixture. Stagnation-triggered refinement preserves that
fixture but still fails the captured normal gate. **Neither is promoted**;
defaults retain the previous friction fix. Both candidates are test-only, and
no full workload was retried. Next work must address the coupled convergence
path rather than continuing local-publication tuning. All acceptance gates
remain unchanged and incomplete.
[Evidence and rejected candidates](benchmarks/soil/2026-10-10-shared-normal-accuracy/SUMMARY.md).

**Previous friction-accuracy fix (2026-10-10):** the captured contact-100 friction
failure was caused by local publication accepting a 2.25178e-6 defect while the
coupled certificate requires 1e-6. The paired shader now uses that same bound
locally; one more Newton iteration resolves the contact and the coupled fixture
converges in **24 sweeps**, with independent physical checks. Standalone/scalar
contracts are preserved; eight local and four paired GPU tests pass. The actual
50k run still commits **13 frames** before a separate **normal-feasibility**
failure at contact 96. That new failure reproduces bit-for-bit in **1.25 s**,
with normalized normal residual **-1.30311e-6**. Actual frames remain
**699.17–1128.89 ms**; full trajectory and 4 ms/60 FPS acceptance remain unmet.
[Fix, regressions and next exact normal fixture](benchmarks/soil/2026-10-10-shared-friction-accuracy/SUMMARY.md).

**Previous coupled-failure replay (2026-10-10):** opt-in runtime fault capture
now preserves hashed shared inputs, paired rates, contacts and retained state.
The new actual run commits 14 frames, then fails the unchanged 64-sweep limit
at the friction-slip certificate, contact 100. Its isolated replay reproduces
the certificate and final paired rates bit-for-bit in **1.33 seconds**; the
independent normalized friction defect is **2.25178e-6**, above the unchanged
1e-6 bound. Completed actual frames take **686.56–1247.21 ms**. The differing
first failed check/frame across runs is not a convergence fix or speedup.
Use this captured coupled fixture for the next solver change; no further full
workload retry is needed to reproduce it. Calibration remains separate.
[Capture, replay and remaining work](benchmarks/soil/2026-10-10-shared-failure-replay/SUMMARY.md).

**Previous shared-convergence checkpoint (2026-10-10):** the local line search
now requires strict merit decrease; rounded Armijo factors previously accepted
41 equal-merit steps in a captured failure. All three captured inputs now pass
in 4/11/26 iterations, with unchanged limits and physical checks. Eight local
GPU tests and four paired-solver tests pass. The actual 50k run still commits
**13 frames**, but now completes its local solves and fails the **shared
64-sweep limit**. The retained certificate identifies **normal feasibility at
contact 96** as its first failed check. Frame times are **692.72–1066.82 ms**;
full trajectory, rendering and performance acceptance remain open.
[Evidence and next shared-state replay](benchmarks/soil/2026-10-10-local-friction-progress/SUMMARY.md).

**Previous local-friction checkpoint (2026-10-10):** runtime errors now retain
the exact failed local input. Two actual floor-contact failures reproduce in
subsecond local fixtures. Active-wall publication now enforces zero normal
velocity with an explicit finite 1e-6 momentum bound and corresponding energy
accounting. A one-time admissible sticking seed recovers a failed line search
within the existing iteration budget; it does not replace ordinary Newton or
bypass physical acceptance. The two inputs pass in 4/15 iterations. All eight
local tests (including 4,374 manufactured fixtures) and the six-case paired
physical fixture pass. Always-on sticking candidates failed coupled convergence
and were rejected. The changed actual run still commits **13 frames**, then
fails at a later contact in shared sweep 2 (local iteration limit, contact 102).
Its exact input is retained for the next local replay. Completed frames take
**693.26–1100.10 ms**; neither speed nor full trajectory acceptance is achieved.
[Implementation and evidence](benchmarks/soil/2026-10-10-local-friction-failure/SUMMARY.md).

**Previous normal-contact checkpoint (2026-10-10):** paired shared-wall and
candidate certificates now share retained-rate normal contraction and residual
arithmetic. A rounded false-success GPU regression rejects correctly while
zero/separating low tails pass; physical, feedback, publication and candidate
checks pass individually. The broad paired test command also selected a replay
probe without required input and therefore is not a green suite. The changed
actual 50k run commits **13 frames**, then fails with **shared-wall-local-solve**
status 64 on attempted frame 13. Completed frames take **721.95–1154.82 ms**.
Local fault/contact and exact input capture are the next diagnostic needed;
neither full trajectory nor performance acceptance is claimed.
[Implementation and evidence](benchmarks/soil/2026-10-10-normal-rate-contract/SUMMARY.md).

**Previous coupled-workload checkpoint (2026-10-10):** the actual six-body paired
tractor passes four CPU-compared frames, including a late injected failure,
bit-exact retained-state rollback and successful retry. The native-atomic 50k
soil workload initially committed three warmup frames, then failed coverage.
The verified first-failure witness run commits six frames (0.1 simulated
seconds, including three measured frames), then fails on a selected wheel
contact: node 123087, body/owner 4, slack -1.4947309e-6 against an approximately
-1.4662565e-6 limit. This is not an omitted candidate. Inner and outer normal
certificates use different arithmetic (the paired source already aligns scaling); their numerical contract
needs reconciliation without weakening the outer gate. Completed frame times
are 654.67–970.18 ms, not performance acceptance. Candidate margins and
certificate tolerances remain unchanged. [Evidence](benchmarks/soil/2026-10-10-paired-tractor-runtime/SUMMARY.md).

**Previous paired predictor checkpoint (2026-10-10):** unrestricted paired
prediction now splits acceleration evaluation from integration/reconstruction.
The retained scalar delta buffer carries acceleration between the two dispatches;
integration replaces it with the public prediction delta. No additional state
buffers or weakened arithmetic are required. Model bounds, caps, joint kinds and
structural-lock bits gate specialization and are rechecked before dispatch;
the cached program identity includes this selection. Scalar defaults are unchanged.
The eight-cycle prediction/corrected-drift fixture passes in 46.00 seconds,
including construction, against the unchanged 90-second limit. The first actual
paired `RigidWorld` fixture also passes in 52.61 seconds: sub-ULP force integration,
alternating substeps, a host velocity edit and a mass-triggered rebuild. These
are correctness/construction results, not real-time timings. A two-body runtime
also passes four child-force transmission frames through a fixed joint in 50.67
seconds, including construction, with alternating substeps and unchanged 2e-13
velocity checks. The six-body articulated machine,
soil-coupled trajectory and 50k/4 ms/60 FPS acceptance remain open. The generic
constrained monolithic paired predictor remains unaccepted.
The final staged predictor check also verifies rejection of a newly uploaded
finite joint bound before changing output; source validation, model-guard checks
and the GPU-enabled facade build pass.
[Staged predictor evidence](benchmarks/soil/2026-10-10-paired-loop-control/SUMMARY.md).

**Current acceptance checkpoint (2026-10-10):** no accepted native 50k tractor
frame or 4 ms/60 FPS result. The bounded dual candidate passes six saved failure
cases and focused GPU gates, but the changed actual trajectory still rolls back
at substep 26. A positive numerical capacity bound identifies insufficient
nearest-rate correction capacity on that new input. The next implementation
must preserve compensated generalized rates throughout the runtime. The paired
solve/prediction/publication boundary is now implemented and GPU-verified,
including persistent feedback, but is not yet connected to the actual tractor.
Adding sweeps or only certifying hidden low bits is not a fix.
[Paired contact boundary](benchmarks/soil/2026-10-10-wall-friction-paired-rates/SUMMARY.md).

**Featherstone rate integration checkpoint:** an opt-in, separately cached program
now retains generalized rate pairs, prediction deltas and low-part translation/
quaternion drift. The default program specializes paired-rate branches out.
Focused GPU prediction arithmetic and actual corrected drift pass; full generic
prediction verification remains incomplete after a bounded 90-second timeout.
ABA force evaluation still uses scalar views. Subsequent transport and lifecycle
work is recorded below; live runtime selection and full predictor acceptance remain open.
This is not an accepted tractor frame or an enabled production runtime path.
[Integration evidence](benchmarks/soil/2026-10-10-featherstone-paired-rates/SUMMARY.md).

**Body-velocity checkpoint:** paired generalized rates now reach reconstructed
world COM velocities, including parent velocity inheritance, reversed traversal
and all eight joint kinds. A 48-case f64 kinematics comparison passes for dynamic,
kinematic and fixed roots. It exposed a lost TwoSum residual in a cross product;
the paired program now rounds both residual subtractions explicitly. Neither
physical tolerances nor the default scalar arithmetic were changed. Full
runtime/frame acceptance is still open.
[Body-velocity evidence](benchmarks/soil/2026-10-10-featherstone-body-rates/SUMMARY.md).

**Momentum and frame lifecycle checkpoint:** world-momentum transport now retains
rate/body-velocity pairs and publishes scalar high views alongside them. Two GPU
fixtures pass, including an articulated f64/projected-response comparison at
8.76845e-8 normalized error against the unchanged 1e-6 gate. A one-invocation
paired workgroup resolves the observed construction timeout without reducing the
fixture or raising its timeout. Runtime precision now supports paired host edits,
identity remapping, ping-pong retention, predictor-delta rollback and extended
frame readback. Default runtime construction still selects scalar prediction: accepted live paired selection and
full generic paired ABA verification are required before the next actual tractor
diagnostic. None of these component checks establishes 50k/4 ms/60 FPS acceptance.
[Transport](benchmarks/soil/2026-10-10-paired-momentum/SUMMARY.md),
[lifecycle](benchmarks/soil/2026-10-10-runtime-paired-rates/SUMMARY.md).

**Paired soil-consumer checkpoint:** the runtime projection constructor now passes
an explicit paired encoding through shared-wall geometry/solve, candidate
certification, contact-motion observation and machine publication. Four native
GPU checks pass, including four MPM feedback steps with maximum normalized
momentum error 3.53762e-8 (unchanged 1e-6 gate). Candidate coverage and motion
observation retain low-rate contributions through cancellation rather than
reading pair storage as scalars. Scalar geometry regression checks pass. Paired
captures carry an explicit schema/encoding marker. Full paired ABA and live
`RigidWorld` selection/trajectory acceptance remain open; other coupling modes
reject paired construction rather than silently consuming the wrong layout.
[Soil-consumer evidence](benchmarks/soil/2026-10-10-paired-soil-consumers/SUMMARY.md).

**Predictor-construction checkpoint:** an explicit experimental paired `RigidWorld`
constructor now connects the retained precision path, with guards against
unsupported scene/constraint modes. It is not enabled by `SoilTractor` or accepted
as a working frame path. Four changed-source attempts still time out within the
unchanged 90-second full-predictor construction gate. Unsuccessful extra pipeline
families and a reconstruction split were removed; their evidence is retained.
Paired one-lane dispatch remains, with matching host counts and device-limit
checks; the existing 48-case body-kinematics regression passes. These results do
not establish predictor execution or a whole-machine speed improvement.
[Predictor evidence and next gate](benchmarks/soil/2026-10-10-paired-predictor-dispatch/SUMMARY.md).

### Read this first

This is the detailed execution plan, not a claim that the simulator is finished. Start with the phase ledger in section 6; the source audit in section 2 describes the original baseline, while the ledger and linked report describe subsequent work.

The intended order is **remove avoidable work → keep simulation on the GPU → bound work to disturbed soil → integrate and calibrate actual machines**. Do not replace the solver family merely because an unoptimized integration is slow. Research links, implementation paths, dependencies, correctness gates, and performance gates are included below.

At the current checkpoint, binned terrain export is verified, retained DEM stepping is partially integrated, and opt-in compact MPM grid storage passes focused dense-reference checks. Physical fields and world lookup now scale with configured active-tile capacity, not domain volume. There is no accepted active/compact-grid timing benchmark, and real-time agricultural acceptance is still open. Optional DEM topology refit is experimental: small-scene contact checks pass, but the 50k-grain trajectory gate fails. Dense MPM/full-rebuild DEM remain defaults. The user confirmed the RTX 4080 target on 2026-10-09; detailed fixture/percentile gates still require freezing, and initial results remain physically uncalibrated without measured soil data.

**Latest preparation checkpoint (2026-10-10):** captured preparation passes on
baseline/native devices; actual-context and isolated post-factor storage match
byte-for-byte. Retained handles alone do not resolve the actual first-preparation
timeout (five seconds, attributable Xid 109, no completed frame). A private split
preparation candidate and replay probes are implemented but not GPU-verified or
promoted. RTX 4080 acceptance remains unmet and physical calibration unavailable.
[Evidence](benchmarks/soil/2026-10-10-wall-friction-prepare/SUMMARY.md).

**Split follow-up:** split preparation passes three focused GPU suites and exact
captured-solve comparison, but actual 50k execution still times out in the first
point-response stage, before cached metric setup. It is not a runtime fix. A
test-only storage-backed response candidate now removes the local 32-vector
array from that stage; GPU verification is pending, with no promotion or claimed
root cause. [Changed-source failure](benchmarks/soil/2026-10-10-wall-friction-split-prepare/SUMMARY.md).

**Storage-response follow-up:** three GPU suites and captured full-solve parity
pass. Actual native 50k point-response preparation now completes, but the next
cache-setup stage times out before any sweep/frame. A direct-storage cache-setup
candidate is implemented, not yet GPU-tested or promoted. No 60 FPS/4 ms claim.
[Stage progression and pending candidate](benchmarks/soil/2026-10-10-wall-friction-storage-response/SUMMARY.md).

**Storage-setup follow-up:** component parity passes, and the actual synchronized
50k run no longer times out. It still commits **zero frames**: substep 7 exhausts
64 sweeps and the frame rolls back. Captured replay identifies the first failing
gate as f32 represented-rate momentum (DOF 1); the internal compensated residual
is much smaller. No gate is relaxed,
no candidate is promoted, and no performance acceptance follows.
[Captured convergence failure](benchmarks/soil/2026-10-10-wall-friction-storage-setup/SUMMARY.md).

**Represented-rate reconciliation:** a guarded mass-weighted impulse correction
solves the original saved failure in four GPU sweeps with unchanged certification;
four focused GPU suites pass. The next actual-context input still fails: correcting
vertical momentum violates coupled rotational momentum, and the guard restores
state before the frame rolls back. No valid frame or promotion. Joint momentum
reconciliation must pass both saved cases before another full run.
[Candidate and counterexample](benchmarks/soil/2026-10-10-wall-friction-represented-rates/SUMMARY.md).

**Coupled reconciliation follow-up:** bounded, rank-aware coupled correction
passes both saved momentum failures in two GPU sweeps and passes focused guards.
The actual run passes shared solves through substep 9, then rolls back on coverage.
Selected contact 211 exposes a normal-separation scaling mismatch between the
shared solver and bridge. The private certificate is now strengthened to the
existing bridge scale, and that saved input passes in three sweeps. Full-runtime
confirmation and all performance acceptance remain open; no promotion. The
stronger certificate also exposes 64-sweep exhaustion in a 32-DOF component case,
in both matched reference and candidate; that regression remains unresolved.
[Coupled correction and coverage evidence](benchmarks/soil/2026-10-10-wall-friction-coupled-rounding/SUMMARY.md).

**Wide convergence isolation:** the 32-DOF failure is reproduced with per-sweep
certificate traces. Tighter local solves do not fix it. A bounded acceleration
half-step passes the wide case in 58 sweeps with independent 1e-6 physics,
bridge-normal and repeatability checks, but regresses the eight-DOF case. Further
full/half selection and alternating traversal also regress other fixtures. None
is promoted; the current private near-root retry still fails wide convergence.
No caps or physical gates changed, and no further full tractor run was launched.
[Preserved experiments and next action](benchmarks/soil/2026-10-10-wall-friction-wide/SUMMARY.md).

**Cancellation-safe merit:** replacing cancellation-prone projected residual
subtractions with equivalent branch expressions resolves the strict wide
component regression: 32 DOFs certify in 52 sweeps, eight-DOF/seven-contact in 55,
and 65-contact in 51. All five focused GPU suites and six CPU checks pass. The
three saved actual-input failures still pass in two/two/three sweeps. Physical
gates, sweep limits and production defaults are unchanged. The changed-source
native 50k diagnostic reaches substep 10, then rolls back: contact 44's local
early exit accepts zero impulse at -1.080112e-6 normal slack, outside the outer
1e-6 bound. Local publication now enforces that same bound. All six GPU suites
pass, including a one-sweep near-threshold regression; all four saved runtime
failures solve in two sweeps. The second changed-source full-frame diagnostic
passes shared solves through substep 13, then rolls back on represented momentum
at substep 14. Its saved replay exposes insufficient bounded correction; both
coupled correction and rate-only CPU search still fail the unchanged oracle.
No committed frame, performance acceptance or promotion yet.
[Evidence](benchmarks/soil/2026-10-10-wall-friction-stable-merit/SUMMARY.md).

**Bounded redistribution:** saturated contacts now receive bounded corrections
while unsaturated contacts share the remainder. An all-row momentum interval
guards the final correction fraction. The saved substep-14 input certifies in
three GPU sweeps; six solver GPU suites, two geometry GPU suites, seven CPU
checks and geometry budget preflight pass. Earlier captures remain two-sweep
successes. Explicit scratch costs 16 bytes per contact and is included in budget
and binding-size checks. No per-contact limit, physical tolerance or sweep cap
was raised. The changed-source native 50k diagnostic passes shared solves
through substep 25, then rolls back at substep 26: zero committed frames. A
short replay identifies another represented-momentum failure. No production
promotion or performance acceptance yet.
[Evidence](benchmarks/soil/2026-10-10-wall-friction-redistribution/SUMMARY.md).

**Bounded dual follow-up:** replacing frozen saturated corrections with a
ball/halfspace-projected dual Newton method solves the saved substep-26 input
in three GPU sweeps. The first GPU result fails the strengthened independent
1e-6 check because componentwise absolute momentum terms give a looser scale
than absolute net contact momentum. The private candidate now uses the stricter
net-contact scale for its correction and full certificate. All six saved runtime
counterexamples pass the independent 1e-6 oracle; eight CPU checks pass. The
original per-contact correction limits and shared/local caps are unchanged.
Additional workgroup scratch is 640 bytes, preflight checked; retained buffers
are unchanged from redistribution. Six shared and two geometry GPU suites pass.
The actual frame still rolls back at substep 26 on a changed input; its nearest
rounded rates exceed bounded correction capacity. Adjacent single-DOF trials
also fail. A self-contained high-mass contact case verifies the precision issue
and passes the full physical oracle with high+low rates; nine CPU checks pass.
This is not integrated compensated runtime state, promotion or performance
acceptance.
[Evidence](benchmarks/soil/2026-10-10-wall-friction-bounded-dual/SUMMARY.md).

**Paired-rate boundary:** a separate experimental component now consumes and
publishes explicit high/low rate pairs. Momentum and energy certification use
those pairs; a status-gated GPU publisher retains them in caller-owned state
and transport-delta buffers. Paired increment prediction starts a new step's
delta without discarding sub-ULP motion. Eight prediction/solve/publication
cycles, six physical/repeatability fixtures, 32-DOF publication failure/restore,
and all seven captured failures pass independent 1e-6 checks. The newly changed
substep-26 case certifies in three sweeps without impulse-rounding repair.
Ten CPU checks and the legacy shared-wall/publication controls pass. Scalar
layouts remain unchanged; paired storage adds four bytes per DOF to the solver
rate output and uses an explicitly distinct owner. **Not runtime integration:**
Featherstone, corrected drift, world-momentum transport, edit/readback and frame
rollback still need the paired state wired through them. No new actual-frame,
4 ms/60 FPS or calibrated-soil acceptance is claimed.
[Evidence](benchmarks/soil/2026-10-10-wall-friction-paired-rates/SUMMARY.md).

## 1. Objective and execution rules

Build a GPU-first soil/particle system suitable for interactive tractors and working implements. Preserve useful physical behavior: traction, sinkage, persistent deformation, material displacement, and reaction forces. Do not achieve a higher displayed FPS by advancing less simulated time, disabling coupling, silently reducing resolution, or replacing physical soil with visual stamping.

The immediate work is performance engineering of existing MPM, DEM, terrain export, and tire coupling. A complete new agricultural material model is a later, separately measured milestone. Do not spend the first implementation cycle replacing the solver family or building another standalone demonstration.

Execution rules:

- Implement one bounded change, run its focused correctness checks and short paired benchmark, then continue. Do not repeatedly run the entire workspace while the selected change is unfinished.
- Keep CPU references for correctness and small fixtures. Production device paths must not silently fall back to CPU.
- Use `make` for build, test, benchmark, and formatting tasks. Add narrowly scoped Make targets where needed.
- Edit with patch/edit tools, never Python. Comments describe behavior, not the history or justification of a change; keep inline comments within the repository's 20% guideline.
- Preserve existing APIs and reference paths until callers are migrated. An optimization is not permission to change the physical model.
- No dependency/backend replacement, commits, pushes, user application shutdowns, driver changes, or unrelated cleanup without separate authorization. If a commit is requested, use an unsigned, title-only Conventional Commit, with a message of at most 50 characters and no signature/trailer.
- Update the phase ledger in this file with evidence, not estimated completion percentages. A kernel passing its tests is not integrated machine acceptance.

### First action for an executor

Read this file in full, inspect `git status --short`, and check drift:

```sh
git diff --stat aecdeab94ec4a26970edfe830f79a17a182e7aeb..HEAD -- \
  Makefile crates/molla-solvers crates/molla-kernels crates/molla-vehicle \
  crates/molla-sim crates/molla/src/compute bin/molla_viewer/src
```

Also inspect uncommitted changes in these paths. Line numbers below are navigation aids; symbols and current code are authoritative. If a cited path has changed, re-establish the relevant behavior before implementing. Do not overwrite other work.

## 2. Baseline implementation and identified costs

These findings describe the plan-creation baseline at the revision above, before the implementation increments recorded in the phase ledger. They came from source inspection, not a timing run, and establish work being performed rather than its measured percentage of frame time. See [the implementation report](SOIL_AND_PARTICLES_REPORT.md) for subsequent changes, focused verification and measured export timings.

| Component | Current source | Consequence |
|---|---|---|
| DEM live stepping | `crates/molla-kernels/src/tire_dem.rs`, `step_sparse_lbvh_gpu_mirrored`, around 310–354 | Rebuilds LBVH, steps particles, exports terrain, and downloads terrain plus grain state every call. |
| Compaction viewer | `bin/molla_viewer/src/examples/119_soil_compaction.rs:107–108,512–531` | 16 substeps at 0.0005 s per update. Its mirrored path invokes five buffer readers per substep: 80 map/wait calls per update, not 80 measured stall durations. |
| MPM terrain export | `crates/molla-kernels/shaders/tire_mpm.wgsl:173–212`, `raster_terrain` | Each output terrain cell scans all particles: O(H × P). |
| DEM terrain export | `crates/molla-kernels/shaders/tire_dem.wgsl:405–452` | Two all-particle scans per terrain cell, including surface-layer velocity selection. |
| MPM viewer integration | `bin/molla_viewer/src/examples/122_tire_realistic.rs:1483–1502` | Performs device terrain rasterization, then creates a CPU terrain owner by reading particle positions, velocities, and compaction. |
| MPM snapshot construction | `crates/molla-solvers/src/mpm.rs:1155–1173,1276–1290` | A convenience export synchronizes full state. This is not a free handle/view operation. |
| Dense MPM stepping | `crates/molla-solvers/src/mpm.rs:1512–1593` | Grid reset/update dispatches cover N³ cells per substep. Each Jacobi iteration adds two full-grid passes; rigid coupling adds more. |
| Duplicate backend work | `bin/molla_viewer/src/examples/122_tire_realistic.rs:678–700,1396,1525` | MPM and Bekker are enabled by default and updated independently, while FEM terrain selection prioritizes one. Comparison mode is useful, but should not be compulsory production work. |
| Existing optimizations | `crates/molla-solvers/src/mpm.rs:1522–1596` | Substeps already share an encoder/submit, and workgroup-local P2G accumulation already exists. Do not propose these as entirely new features. |

Here P means particles and H means output heightfield cells. At 128² cells and 100,000 particles, the MPM export performs roughly 1.64 billion particle/cell checks. This is an operation-count illustration, not a measured runtime or the current default scene size.

### Drift-check excerpts

Current MPM grid dispatch sizing:

```rust
let grid_total = self.cfg.grid_size.pow(3);
let groups_grid = grid_total.div_ceil(WORKGROUP_SIZE);
let groups_part = self.particle_count.div_ceil(WORKGROUP_SIZE);
```

Current DEM ownership bridge includes:

```rust
let terrain_readback = kernel.readback_resident_terrain(device, queue, &terrain)?;
let mirror = kernel.readback_resident_state(device, queue, &next)?;
```

Current MPM viewer export follows device rasterization with:

```rust
match mpm.tire_fem_terrain_owner_with(device, queue, raster) {
```

Treat these as evidence of data flow, not instructions to delete calls blindly. CPU roller queries, force application, diagnostics, and rendering currently consume some of the returned state.

### Existing physics that must not be overstated

- `crates/molla-vehicle/src/deformable.rs` implements permanent heightfield sinkage using `p = stiffness * z^n`, plus heuristic excess-shear sinking and rolling loss. This is not a complete layered, moisture-dependent soil law.
- The CPU Bekker contact mapping around lines 242–244 uses `local_x` but not the available `local_z` footprint coordinate. Correct this in an isolated correctness change, with width/rotation tests and a fresh benchmark baseline afterward; do not mix changed physics into an optimization comparison.
- `crates/molla-vehicle/src/granular.rs` contains actual particle interaction and tire feedback, but its simple equal-grain model is not a calibrated agricultural soil representation.
- Scene 122 actually feeds tire contacts into `MpmSimulation` and advances MPM. It is not merely an unused GPU kernel. It is also not proof of integrated four-wheel tractor or implement behavior.
- A terrain water-film map is not bulk moisture, saturation, or pore pressure.
- Existing sand/tire test gates mostly establish numerical or qualitative behavior. For example, `mpm_v3_gpu.rs:386–392` accepts a pile slope anywhere between 0 and 90 degrees. That does not establish a calibrated angle of repose.

## 3. Research basis and limits

### R1 — Latency-oriented GPU MPM

[Fei, Huang and Gao: Principles towards Real-Time Simulation of Material Point Method on Modern GPUs](https://raymondyfei.github.io/gpu_mpm/), with [GTC slides](https://raymondyfei.github.io/gpu_mpm/GTC_slides.pdf), report the following single-V100 simulation-frame measurements on slide 57:

| Particles | Reported time/frame |
|---:|---:|
| 13.8k | 3.1 ms |
| 55.3k | 6.3 ms |
| 389.3k | 21.2 ms |

Useful techniques include particle ordering, less frequent expensive mapping rebuilds, reduced transfer conflicts, and careful kernel/atomic design. These are research-scene timings, not calibrated agricultural-machine or renderer-inclusive timings. Reproduce relevant mechanisms, not headline FPS. CUDA-specific techniques require capability and performance evaluation before translation to WGSL.

### R2 — Fused particle/grid transfer

[Wang et al.: Claymore / A Massively Parallel and Scalable Multi-GPU Material Point Method](https://github.com/penn-graphics-research/claymore) introduces G2P2G fusion and particle data layouts that reduce launches and memory traffic. Evaluate this after fixing Molla's integration and terrain-export costs. Fusion must preserve the placement of stress updates, plasticity, boundary conditions, and coupling; it is not simply concatenating two shaders.

### R3 — Active domains and reusable neighborhoods

[Chrono CRM terramechanics paper, 2025](https://arxiv.org/html/2507.05643v2) studies persistent neighbor lists and active domains. Table 4 reports an optimized tracked-vehicle real-time factor of 22.5, despite a 15.9× improvement over its baseline. Here real-time factor is wall time divided by simulated time: 22.5 is slower than real time. Use its locality/reuse methods, not an assumption that adopting CRM solves the runtime budget. A fixed rebuild-every-ten-steps rule from another setup is not a universal correctness bound.

### R4 — Persistent terrain versus active material

[AGX Terrain documentation](https://www.algoryx.se/documentation/complete/agx/html/doc/UserManual/source/agxTerrain.html) describes persistent terrain mass/compaction and conversion to localized dynamic soil. This is an architectural reference for bounded active work, not a drop-in solver or evidence that a particular Molla implementation meets a frame budget. Resting-to-active transitions must preserve mass and material history.

### R5 — Physical calibration is separate from solver speed

[Soil–tool DEM study](https://www.sciencedirect.com/science/article/abs/pii/S0168169914001306) links soil mechanics to moisture and bulk density and compares implement forces against soil-bin measurements. [DEM-to-SCM calibration research](https://arxiv.org/abs/2407.18903) illustrates calibrating a cheaper representation from a more expensive reference. Neither justifies claiming measured field accuracy from CPU/GPU parity alone.

### R6 — Contact-space NCP alternatives

[Menager and Carpentier: Frictional Contact Solving for Material Point Method](https://arxiv.org/html/2602.02038v1)
uses a contact-space NCP with De Saxce correction and ADMM linear/cone-projection
steps. Its iteration and residual results are not evidence for Molla's 4 ms target.
The test-only reference above adapts the splitting idea to Molla's existing
soil/machine response; it does not adopt the paper's full implicit MPM model.
Do not replace hard-contact complementarity with a compliant convex approximation
or change simulated-time integration solely to obtain a faster benchmark.

## 4. Success metrics and provisional workload contract

On 2026-10-09 the user confirmed the RTX 4080 in response to the proposed one-tractor-plus-implement, 50k-active-particle, 60 FPS, ≤4 ms soil-work target per 1/60 s simulated time. Use this as the initial hardware/workload target; the detailed fixture and percentile gates below still require freezing in S00. The user has no measured soil/traction dataset. Initial results must be explicitly physically uncalibrated: numerical verification does not establish agricultural accuracy, and S09 remains open until independent physical data is available.

### Initial interactive target

- Display target: 60 FPS, 16.67 ms total frame budget.
- Initial soil budget: p95 ≤ 4 ms and p99 ≤ 6 ms of completed wall-clock work per 1/60 s of simulated time, including soil stepping, required terrain export, and soil-side coupling. These are targets, not current measurements or guaranteed feasibility.
- Report machine/tire solving, rendering, and total application latency separately; do not hide them inside an unexplained aggregate or claim the full application meets the soil-only budget.
- Initial end-to-end fixture: one four-wheel tractor and one simple tine. Also measure one, four, and eight independently moving active regions; an active region is not automatically one whole machine.
- Initial solver sweep: 10k, 50k, 100k, and 250k particles. The first proposed interactive particle target is 50k active particles; 100k/250k are scaling cases until explicitly promoted to release gates.
- Raster sweep: 32², 64², 128²; grid sweep: dense 32³/64³ references and matching sparse domains. Record physical spacing and extent, not only element counts.
- Keep the existing timestep for before/after tests. For new physical workloads, derive and verify a stable timestep; never assume the viewer's current value is suitable for arbitrary stiffness or resolution.
- For the final machine fixture, freeze mesh, masses, soil depth/material, contact spacing, solver iterations/tolerances, and camera/render settings before performance acceptance.

If these targets cannot be met at the agreed resolution, present an explicit accuracy/performance frontier. Do not silently substitute a different material, coarser representation, or smaller workload.

### Required metrics

1. Completed wall time / simulated time, with both numerator and denominator recorded.
2. Median, p95, and p99 latency for fixed simulated-time windows, plus maximum observed latency.
3. GPU pass timings where timestamp queries are supported; wall-clock completion timings remain required.
4. CPU encode/prepare time, queue submissions, synchronous waits, transferred bytes, allocations, and bind-group/pipeline creation after warmup.
5. Total/active particles, total/active grid cells, raster cells, contact count, neighbor candidates, actual accepted neighbors, and rebuild frequency.
6. GPU memory capacity/usage, particle/scratch capacities, overflow events, high-water allocation, adapter/driver/backend, compiler/profile and revision.
7. Physical outputs: material mass, momentum/reaction balance, energy or dissipative-work diagnostics, rut geometry, traction/draft force, and accumulated simulated time.

Scene 119 currently advances 16 × 0.0005 = 0.008 simulated seconds per update. Therefore update FPS by itself does not establish real-time progression. Use a fixed-time accumulator with an explicit catch-up policy in integrated scenes; report missed deadlines rather than dropping physical time silently.

## 5. Architecture contracts

### 5.1 Separate state, execution, observation, and rendering

Use existing crates by domain, not new CPU/GPU crate families:

- `molla-solvers`: authoritative MPM/SPH stepping and active-grid scheduling.
- `molla-kernels`: reusable device terrain export, DEM stepping, coupling kernels, and persistent GPU resources.
- `molla-vehicle`: host material/query contracts and CPU correctness references.
- `molla-sim`: public scene lifecycle and machine force/impulse integration.
- `molla` facade: re-export supported reusable APIs; viewer-specific state must not become the public API.
- `bin/molla_viewer`: demonstration, selection UI, and observation only; no hidden physical implementation unique to a scene.

Follow `book/src/compute/persistent.md`: allocate once, dispatch many, snapshot explicitly. Follow the existing `Result`/`Error` conventions and GPU resource macros; do not bypass resource counters by allocating through an untracked helper.

Proposed responsibilities, with final names chosen to match nearby APIs:

| Responsibility | Required behavior |
|---|---|
| Resident state owner | Owns particle/material/history buffers, scratch capacities, terrain outputs, revision and device identity. |
| Encode step/batch | Records work into a caller-controlled encoder; does not map buffers or wait for the device. |
| Terrain device view | Exposes revisioned buffers and dimensions without creating a CPU terrain copy. |
| Snapshot request/poll | Explicit asynchronous observation, tagged with simulation step/revision; bounded staging ring and backpressure. |
| Blocking reference wrapper | Available for tests/offline tools, visibly named and excluded from production stepping. |
| Compact reaction output | GPU-consumable force/torque or impulse buffers with body identity and coordinate-frame metadata. |

No full particle-state download in steady-state production stepping. For a CPU machine integrator, allow only an explicitly classified compact reaction synchronization at the coupling boundary; report it separately and do not label that path fully GPU-resident. The device-machine path must consume reactions directly on-device.

Uniform parameters that differ between recorded substeps need distinct dynamic offsets or equivalent immutable records. Repeated `queue.write_buffer` to the same range before one submit must not cause all substeps to see only the last parameters.

### 5.2 Lifecycle and bounded resources

- Preallocate known workloads; provide explicit checked growth when capacities change. Steady-state fixed-capacity frames create no new GPU buffers, pipelines, or bind groups.
- Empty scenes, reset, particle insertion/removal, tile changes, material edits, and device loss have explicit behavior. No stale terrain or reaction buffers after reset/removal.
- In-flight snapshots keep their source lifetime valid and identify their generation. Never apply an old readback to a new scene.
- A full telemetry ring skips or coalesces optional observations, not physics steps. Bound memory rather than accumulating pending requests.
- Overflow is detected and reported. Never drop particles, contacts, active tiles, or neighbors silently to preserve frame rate.
- No synchronization in each inner iteration merely to read convergence state. If convergence is GPU-controlled, bound iterations and expose diagnostics without an implicit fallback.

### 5.3 Active work without lost physics

Inactive soil retains material and deformation history. Activation includes the full transfer/contact stencil, boundary halo, particle travel envelope, and connected moving/disturbed material. A stationary implement can continue loading or compacting soil; camera visibility and object velocity alone are not valid sleep criteria.

Tile and particle identities must survive reordering and migration. Multiple nearby machines share soil ownership and merge overlapping active regions; they must not simulate duplicate copies or apply forces twice. Detached/airborne material stays active until physical sleep criteria permit settlement, even after the implement leaves.

## 6. Phase ledger and dependencies

Phase evidence is recorded in `SOIL_AND_PARTICLES_REPORT.md`. Partial work never implies completion of the phase or its downstream acceptance gates.

| Phase | Deliverable | Priority | Depends on | Risk | Status |
|---|---|---|---|---|---|
| S00 | Bounded baseline harness and frozen workload manifest | P0 | — | Low | IN PROGRESS: unified MPM/DEM smoke, active-domain checker and true counters-disabled smoke controls implemented. Fourteen CPU/process contracts, both feature-build checks, bounded GPU/scope/failure gates pass. Other schemas, full-workload overhead attribution and full acceptance remain |
| S01 | One selected production backend; separate diagnostics | P0 | S00 | Medium | IN PROGRESS: exclusive viewer selection, dirty material-map cache and optional MPM/Bekker summaries implemented; integrated counters/timing and remaining diagnostic consumers need acceptance |
| S02 | Particle-binned linear-work terrain export | P0 | S00 | Medium | IMPLEMENTED: MPM/DEM focused GPU checks and independent P/H sweeps pass on RTX 4080; see report for precision/portability limits |
| S03 | Resident stepping and migrated consumers without inner-loop readbacks | P0 | S01, S02 | High | IN PROGRESS: retained feedback/bridges/telemetry, GPU terrain queries, and the full six-stage prepared brush chassis loop with one submit and zero observed hot buffers/transfers. All eight parity combinations and 32 frames match legacy bytes. CPU-tire/viewer migration, soil-coupled full-frame residency and experimental caller-encoded MPM parity gates remain |
| S04 | Persistent DEM resources and safe neighborhood reuse | P1 | S03 | High | IN PROGRESS: production ordered refit passes public topology/residency and original 50k gates; 52 exact states and dense overflow pass. Two public packets reduce paired time by 40.94% / 43.51%; normalized median still about 31 ms/frame. Worst occupancy/quality and absolute budgets remain; full rebuild stays default |
| S05 | Active-tile MPM with dense reference parity | P1 | S03 | High | IN PROGRESS: bounded physical pool/hash lookup, growth and overflow handling; APIC local-coordinate correction closes the bounded 128-step parallel-index velocity gate and 1536-comparison matrix. Stable indexing remains opt-in; actual50k torque, sleeping and timing remain open |
| S06 | Profile-driven transfer/layout/fusion optimization | P1 | S05 | High | IN PROGRESS: shared-chunk transfer, SoA Newton scratch and guarded particle fusion are opt-in. Latest actual50k separate/fused/repeat states match exactly; mean P2G about 10.1 ms/frame, particle publication 3.414/3.280/3.237 ms, Newton 205.6/202.0/200.4 ms. Whole-frame medians 351.7/357.1/342.5 ms do not establish speedup or 60 FPS. Separate non-native stiffness gate remains unresolved. Defaults unchanged; dominant solver reductions and 4 ms acceptance remain open |
| S07 | Persistent field state and active-region lifecycle | P1 | S04/S05 as applicable | High | IN PROGRESS — memory-backed and disk-backed live owners, bounded background selected I/O and durable tile updates; physical sleep/support boundaries, long traversal and streaming performance pending |
| S08 | Actual machine and implement two-way coupling | P1 | S03, S07 | High | IN PROGRESS: conditioned Newton captured cases and SoA lifecycle admission pass. Actual fused/SoA 50k first-substep motion rejection and synthetic substep-18 rejection after 17 successful internal substeps roll back and exactly rejoin two unfused controls across 33 frames. Complete soil, nine machine streams and CPU states are compared on committed frames. Natural late displacement exceedance/other-fault recovery, cross-solver trajectory, long-horizon accuracy, loaded-wheel trajectory, angular and production rendered acceptance remain |
| S09 | Agricultural material/accuracy calibration | P2 | S08 | High | DEFERRED measured-data calibration: user has no data; initial release explicitly physically uncalibrated, with numerical verification still required |
| S10 | Integrated performance, visual and lifecycle acceptance | P0 release gate | Required preceding phases | Medium | TODO |

Recommended first delivery is S00–S03, not the entire roadmap. S04 and S05 are backend-specific; choose which goes first from the measured production workload. S06 is conditional: do not undertake a large fusion rewrite if earlier work already meets the agreed budget. Mark deferred phases explicitly, never DONE. Full agricultural completion requires S07–S10; a performance-only delivery must name its narrower scope.

## 7. Implementation phases

### S00 — Establish a short, reproducible baseline

**Scope:** `Makefile`; new focused harness/tests under `crates/molla-solvers/tests/`, `crates/molla-kernels/tests/`, and a `crates/molla-solvers/examples/soil_benchmark.rs` executable; minimal instrumentation in existing soil paths. Add a bounded result directory under `benchmarks/soil/` during execution, not during planning.

1. Freeze an initial manifest: adapter, driver, release profile, fixed seed, physical units, particle spacing/mass, grid/raster sizes, timestep, substeps, material constants, boundaries, contact trajectory, and snapshot mode.
2. Separate measurements for solver-only, solver plus terrain export, solver plus required coupling, and full viewer/machine application. Measure DEM and MPM independently; retain comparison mode as a separate workload.
3. Warm pipelines and buffers, drain warmup before timing, then measure completed GPU work. Encoder submission duration alone is not a solver time. Reset equivalent states for A/B runs; do not compare different stages of a collapsing pile.
4. Instrument allocation/readback/submit counters and passes; batch timing queries and collect outside inner loops. Measure instrumentation overhead and retain an uninstrumented control.
5. Produce JSON/CSV with manifest hash, source revision, workload identity, p50/p95/p99, RTF, physical diagnostics, and resource counters. Record exit status and whether the run was complete.
6. Make quick mode bounded: compile separately, warm up, run a short fixed-duration sample, terminate only the owned process on its deadline, and classify timeout as incomplete. Initial quick-run cap: 30 seconds of benchmark execution per selected case; acceptance runs may be longer and must declare duration first.
7. Record ambient GPU utilization/memory and thermal/power state. Do not terminate unrelated simulators or applications. A contended run is diagnostic evidence, not uncontended performance acceptance.

**Gate:** new `make test-soil-fast PHASE=S00` and `make bench-soil CASE=smoke MODE=baseline` return nonempty, schema-valid results; work advances the declared simulated duration; deliberate timeout/device-unavailable tests report failure/incomplete rather than success. Freeze numerical tolerances and final workload selection before optimization. Do not spend days collecting a perfect baseline before S01/S02 implementation.

**Current increment:** `make test-soil-fast PHASE=S00` runs fourteen CPU/process contracts. `make check-soil-results INPUT=<run> [BASELINE=<compatible-run>]` checks active-domain and frozen MPM/DEM smoke schemas, writing checked JSON/CSV with recomputed quantiles and physical-time ratios. Missing, duplicate, changed, nonfinite and incompatible records fail; roundtrip-preserving JSON retains exact f32-derived manifest values. `make bench-soil CASE=smoke MODE=baseline [SOIL_SMOKE_BACKEND=mpm|dem] [SOIL_SMOKE_COUNTERS=0|1]` precompiles and executes under a 30-second ceiling with physical checks, source archive/integrity, executable copies and hardware identity/timeline. Disabled core/wgpu counter features are verified and metrics remain unavailable, never zero evidence. Use `INSTRUMENTATION_BASELINE=<opposite-mode-run>` and `SOIL_CHECK_OUTPUT=<fresh-dir>` for explicit same-source/same-physics comparisons; ordinary baseline compatibility remains strict. MPM is solver-only; DEM includes terrain export and a declared preparation step outside measured duration. Both 512-particle fixtures are physically uncalibrated, not full-workload acceptance. Timeout/unavailable-adapter, no-overwrite and cross-backend rejection gates pass. Other schemas, legacy active-domain provenance limitations, full-workload overhead attribution and integrated acceptance remain open. [MPM/failure evidence](benchmarks/soil/2026-10-11-unified-smoke/SUMMARY.md), [DEM scope controls](benchmarks/soil/2026-10-11-unified-dem-smoke/SUMMARY.md), [compiled-out counter controls](benchmarks/soil/2026-10-11-smoke-counter-controls/SUMMARY.md).

### S01 — Separate production stepping from comparison/telemetry

**Scope:** viewer scenes 119/122, their local state/UI, `tire_bekker.rs`, and new backend-selection contract tests.

1. Introduce explicit mutually exclusive production selection for Bekker, DEM, or MPM. Put multi-backend comparison behind a separately named option.
2. Allocate/update material maps only when their inputs or dimensions change. Remove unused exports only after proving no physical consumer depends on them.
3. Split physical queries from diagnostic statistics. Preserve the existing comparison ability without paying its cost in ordinary stepping.
4. Surface the selected solver, resident/mirrored mode, simulated-time rate, and optional diagnostics in the viewer.

**Gate:** `make test-soil-fast PHASE=S01`; selected-backend counters show exactly one production soil solver, comparison mode shows its explicitly requested solvers, and toggling diagnostics does not change physical outputs at the same simulated time. `make bench-soil CASE=integration MODE=compare` compares identical single-backend physics before/after, with no hidden workload reduction used as a solver speedup claim.

**Partial implementation:** Bekker material maps retain GPU buffers until their values or producing kernel change. Uniform friction/film setters avoid repeated host-vector allocation/fill when unchanged; scene 122 uses these setters. Changed maps receive new buffers, preserving previously returned terrain snapshots and cloned-owner independence. Kernel lifetime tokens invalidate cached buffers on replacement, including a different device; this does not add validation for incorrectly paired device/queue/kernel arguments. GPU resource/transfer checks cover unchanged maps, individual edits, removal/re-addition, owner cloning and kernel/device replacement.

Scene 122's MPM compaction and Bekker rut summaries are opt-in and disabled by default; their UI identifies retained values as last samples. Bekker skips its CPU scan when disabled; MPM skips the GPU summary dispatch and its 16-byte download. An unused full Bekker terrain clone is removed. Required physical terrain snapshots and DEM's current mirrored-report calculations remain: this is not yet asynchronous GPU telemetry or fully resident application acceptance. The S01 Make lanes execute focused material-cache and viewer contract tests; the integrated benchmark/counter gate above remains open.

### S02 — Replace O(H × P) terrain rasterization

**Scope:** `crates/molla-kernels/src/tire_mpm.rs`, `src/tire_dem.rs`, their WGSL shaders, a reusable terrain-raster helper if justified, and tests `tire_mpm_gpu.rs`, `tire_dem_gpu.rs`, plus new `soil_raster_gpu.rs`.

1. Preserve the current raster as a small CPU/reference path. Write explicit contracts for bounds/clamping, empty cells, floor/reference heights, compaction maximum, surface height, and velocity averaging.
2. Compute particle-to-cell keys once. Use a histogram/prefix/scatter or fixed-pass radix/segmented-reduction pipeline; select from measured contention and existing reusable scan/sort facilities.
3. Reduce each occupied cell's particles rather than scanning the full bed per output cell. Reuse per-step particle bins when compatible; do not assume DEM collision bins and a 2D surface raster have identical semantics.
4. Preserve DEM's top-surface selection: first determine the surface, then reduce velocity only over particles satisfying the existing surface-band rule. An all-particle average is a different physical boundary.
5. Keep persistent scratch buffers and checked capacities. Handle one extremely crowded cell without dropped entries or an unbounded fixed bucket. Use multi-stage reductions where a single-cell serial reduction would dominate.
6. Initially preserve legacy out-of-domain behavior for equivalence. If clamping every external particle to a border is inappropriate for field streaming, change that contract in a separate correctness phase with new references.

**Gate:** `make test-soil-fast PHASE=S02` and `make test-soil-gpu PHASE=S02`; test empty/single particle, exact cell boundaries, negative coordinates, irregular dimensions, mixed velocities, dense single-cell occupancy, randomized clouds, reset, resize, and overflow. Compare against the CPU/reference formula with predeclared tolerances. `make bench-soil CASE=raster MODE=compare` must show bounded-pass particle/cell work rather than H×P candidate scans, with timing curves for independent P and H growth. Never promise a billion-fold or other speedup from an operation-count ratio.

**Evidence:** five focused GPU tests, one fast contract test, six existing resident DEM regressions and two existing tire-MPM regressions passed. [Latest paired sweep](benchmarks/soil/2026-10-08-mpm-dem-binned-final/SUMMARY.md) covers 10k/50k/100k particles, 32²/64²/128²/256² rasters and crowded cells. DEM uses exact half-open cell predicates after bin lookup, not MPM-style physical border clamping. Decimal-boundary parity is verified against the unchanged legacy GPU raster on the recorded adapter; cross-adapter floating-point behavior remains an integration verification requirement.

### S03 — Make the live loop genuinely resident

**Scope:** DEM/MPM owners and dispatchers, `MpmSimulation`, existing facade exports, viewer consumers, and new `soil_residency_gpu.rs`/viewer contract tests.

1. Add encode-into-caller-encoder step/export APIs and keep blocking wrappers for tests. Do not break existing consumers mid-migration.
2. Introduce persistent ping-pong particle/output buffers, prepared bind groups, and immutable per-substep parameter records. Record feedback → solver → required export → consuming contact work in a documented order.
3. Remove the automatic MPM terrain-owner snapshot from the resident viewer branch. Expose resident terrain views directly to the FEM contact consumer.
4. Replace DEM's per-substep full mirror with device-side queries needed by the roller/contact path. Render particles/terrain from device buffers or a bounded renderer bridge, not a compulsory full-state CPU snapshot.
5. Move optional max-compaction/active-count statistics to GPU reductions and asynchronous compact readbacks. Identify any CPU force consumer and migrate it or classify its compact synchronization honestly.
6. Add snapshot request/poll support with revision tagging, bounded staging storage, and reset/device-loss handling. UI must tolerate stale-but-labeled telemetry.
7. Batch only where coupling permits. If forces require reevaluation every substep, include that device computation; do not batch by freezing changing boundary conditions unintentionally.

**Gate:** `make test-soil-gpu PHASE=S03`; runtime counters show zero full-state downloads and zero blocking waits inside steady-state device substeps, and no GPU resource creation after warmup at fixed capacity. Preserve trajectory/forces against the same reference step ordering. Test delayed snapshots, saturated rings, repeated reset, owner drop with queued work, device mismatch, and changing per-substep parameters. `make bench-soil CASE=integration MODE=compare` reports both solver and end-to-end gains. CPU-machine compact synchronization, if retained, is a separate labeled mode and cannot pass the fully resident contract.

**Partial evidence:** four focused resident GPU tests pass, including changing parameters, reset/count/leaf/contact-capacity changes, cross-instance device/queue/kernel rejection, queued owner drop, and 16-substep frame-mirror parity. Eight fixed-capacity substeps record zero explicit buffer-allocation observations, zero logical downloads, stable live bind-group/pipeline counts, and 208 dispatches. This excludes wgpu's internal command/staging allocation behavior. Scene 119 retains all 16 × 0.0005 s steps but downloads once per frame, with a 16× reduction in measured logical readback bytes. It remains a blocking CPU query/render bridge, not a fully resident application. [Solver+export benchmark](benchmarks/soil/2026-10-08-dem-retained-final/SUMMARY.md) still fails real-time performance at the tested 50k workload.

**MPM consumer migration:** scene 122 now retains the already rasterized `TireFemContactTerrain`, not a second particle-owning CPU MPM state. Its CPU tire/contact consumers use a prepared terrain-only blocking readback; GPU FEM feedback still consumes resident maps directly. The bridge reuses staging, uses one submit/map/wait, and downloads 24 bytes per terrain cell, not 48 bytes per particle. The original blocking helper can still append 16 bytes of reduced compaction statistics for reference/tools, but scene 122 now requests those optional statistics through the asynchronous ring described below. At high raster resolution relative to particle count this need not reduce bytes; benchmark both representative small and large workloads rather than assuming a speedup.

Two focused bridge GPU tests cover buffer validation, repeated reads, changing/decreasing summaries, disabled dispatch, empty particle counts and retained source handles. A six-window paired MPM integration test covers actual feedback/solver stepping, legacy terrain parity, diagnostic toggles, final particle trajectories at the frozen 2e-5 tolerance, and empty reset. It now deliberately fills the two-slot observation ring, skips one observation while advancing physics, and checks a delayed summary against its stamped historical oracle rather than the current terrain. Position/velocity errors remain within the unchanged gate. This does not remove the terrain wait or prove actual machine/render performance. `make bench-soil CASE=mpm-bridge MODE=compare` remains a blocking-bridge diagnostic, not a benchmark of the new ring; it excludes common solver/raster work and reports no real-time factor.

**Asynchronous compaction observations:** `MpmSimulation::prepare_tire_compaction_readback(engine, slots)` prepares a fixed-source `PreparedTireMpmCompactionReadback` with 1–64 retained 16-byte staging slots, one summary buffer and one immutable parameter buffer. `try_request(revision)` immediately submits reduction/copy after preceding queue work and requests asynchronous mapping; it never polls or waits. A full ring returns `false` before submitting any work. `poll_latest()` uses only nonblocking device polling, coalesces completed captures and reads at most 16 bytes from the newest ready current-generation sample. Revisions must increase among accepted requests. Owner identity plus generation prevents applying old-instance/reset samples; valid current-generation observations may still be old revisions and must be labelled accordingly. `reset()` increments generation without reclaiming pending slots; stale completions are unmapped/discarded without host decoding. Empty inputs produce zero statistics without a reduction dispatch.

Scene 122 retains three slots, labels the last observation's generation/revision and terrain-revision age, displays skipped/full-ring count and telemetry errors separately, and leaves fresh CPU terrain queries on `read_blocking(false)`. Optional telemetry failure or backpressure cannot turn a successful terrain update into a physics failure. Disabling diagnostics, resetting/replacing the MPM source or changing count clears the corresponding observation owner/labels. The legacy `active_particles` summary field counts compaction above the prepared threshold; it is not the number of moving or solver-active particles.

Five GPU ring fixtures pass: bounded/coalesced immutable captures, retained source handles, reset with pending work, owner identity, decreasing/empty summaries, logical device destruction, foreign native device rejection and invalid-summary failure latching. A CPU contract proves the reader is Send+Sync for ECS storage; four viewer-local contracts and the viewer build pass. Device loss is surfaced as an error and requires reconstruction on a live engine/source, not silent recovery or CPU fallback. This is a compact-observation implementation, not asynchronous physical terrain queries, a general reaction/snapshot service or whole-viewer device-loss recovery. See [async observation evidence](benchmarks/soil/2026-10-08-mpm-async-observation/SUMMARY.md).

**Retained MPM feedback increment (2026-10-09):** `TireMpmResidentFeedback` retains contact/uniform buffers, bind groups and the unchanged in-place pipeline. `MpmSimulation::reserve_tire_feedback_with` prepares capacity outside stepping; the existing resident wrapper lazily prepares/grows otherwise and reuses resources across particle reseeds. CPU input validation precedes uploads/growth. Empty contacts still run advection; zero particles enqueue no feedback work. Three kernel fixtures and one actual-solver integration fixture pass: queued changing counts/coefficients, exact legacy GPU and 2e-5 CPU parity, invalid context/capacity/alias rejection, immutable caller-encoded records, downstream raster ordering and queued owner drop. Ten feedback steps use 992 logical upload bytes, zero explicit hot GPU buffer allocations/downloads, and stable bind-group/pipeline counts; this excludes driver/wgpu staging allocations.

`PreparedTireMpmFeedback` binds caller-owned GPU contacts and an immutable numeric parameter record. Multiple records can be encoded with GPU contact copies and terrain export in one encoder without shared-uniform overwrite. Raw buffers/queues must belong to the supplied device; GPU-written contacts must satisfy the finite/radius contract. Native device/queue identity checks reject mismatched retained contexts, but do not establish ownership of opaque raw buffers or implement device-loss recovery. The full MPM solver still owns its submission, and scene 122's CPU tire/contact model still requires fresh blocking terrain observations. S03 is not complete; no end-to-end speedup is claimed. Focused lane: `make test-soil-feedback RELEASE=1`; [evidence](benchmarks/soil/2026-10-09-mpm-retained-feedback/SUMMARY.md).

**Caller-encoded MPM increment (2026-10-09, experimental):** `prepare_step(dt, substeps)` creates immutable parameter records; `begin_command_batch(encoder)` takes a caller encoder and exclusively borrows the simulation. `encode_step` copies fixed records on the GPU and uses the same dense/active/compact solver schedule as the submitted wrapper, including implicit steps, material IDs and optional particle ordering. No per-record queue writes can overwrite earlier step parameters. `encoder_mut` composes contact producers/feedback/export between solver records. `submit(queue)` commits initialization state; dropping the batch discards commands and restores the pending active-grid reset, without consuming pending material IDs. Reseeding, material edits, rigid-count changes, dispatch-mode changes and buffer growth invalidate old records. Raw encoder/queue device ownership remains a caller contract; this is not device-loss recovery.

Cancellation/stale/empty-state checks and a GPU feedback → six solver substeps → terrain export fixture pass. The twelve-case solver-mode matrix passed eight cases, then failed active-grid single-pass-P2G implicit rigid-force parity (`-89.925735` versus `-89.92571`, frozen absolute tolerance 2e-5); the remaining three compact implicit cases were not reached. A separate six-pair submission-reference-versus-itself diagnostic also fails, with maximum force difference 2.670288e-5 and two failing pairs. This establishes reference repeatability trouble, not equivalence of the new path or a localized cause. The failed matrix was not rerun; no tolerance was changed. Default viewer submission behavior is unchanged. `make test-soil-command-batch RELEASE=1` is a known-failing experimental gate included in S03, separate from the focused passing feedback lane. [Evidence](benchmarks/soil/2026-10-09-mpm-command-batch/SUMMARY.md).

**Retained CPU-tire feedback bridge (2026-10-09, opt-in):** `PreparedTireFemTerrainFeedback` now retains the existing FEM contact shader's input/output buffers, uniforms, terrain bindings and one staging slot. Scene 122 exposes it for the resident MPM, Bekker and DEM terrain branches through an experimental checkbox, disabled by default. Each retained update uploads fresh CPU node/bristle state, computes against current resident terrain maps, and copies velocities, forces and bristles into one mapped readback with one submit/wait rather than three separate readers. Required fresh CPU terrain reads and the CPU tire solve remain: this is not a fully GPU-resident machine or removal of all synchronization.

Preparation validates capacities, device/kernel context, source storage sizes and an explicit logical byte budget before allocation. The viewer budget is 64 MiB. Eleven retained GPU buffers occupy `100 * node_capacity + 52 * max(tread_capacity, 1) + 208` bytes, excluding borrowed terrain and driver/CPU overhead. Hot calls use no new application GPU buffers, upload `36*N + 20*T + 176` bytes and download `32*N + 16*T` bytes. Cache invalidation includes engine, kernel, all terrain buffer identities/map presence and capacity; reset clears the cache. Terrain buffers must belong to the engine. Invalid shapes, duplicate/out-of-range tread indices and nonfinite CPU inputs reject before uploads; mapping/device failure surfaces without CPU fallback.

Focused GPU checks pass 480 bit-exact legacy comparisons across all optional map combinations, changing counts, current terrain contents and evolving node/bristle state. Lifecycle checks cover allocation budget, pre-upload rejection, cloned/replaced sources, foreign engine/kernel context, source-kernel drop and logical device loss. The viewer helper fixture uses the actual 216-node/72-tread testbench tire, applies returned velocities, and exercises terrain/kernel replacement and reset. No rendered or full-machine acceptance is claimed.

A short alternating-order diagnostic observed medians of **0.061425 ms legacy vs 0.051332 ms retained** at 129 nodes/32 treads, but **0.425655 ms legacy vs 3.530727 ms retained** at 2,048 nodes/512 treads. The RTX 4080 was 90–93% utilized; these synthetic feedback replays are not physical trajectories or uncontended timing. Preserve the unfavorable larger case: no general speedup or default adoption is accepted, and no timing retry was performed. The focused lanes are `make test-soil-fem-feedback RELEASE=1` and `make test-soil-fem-viewer`; they are included through the existing S03 feedback lane. [Evidence and remaining integration limits](benchmarks/soil/2026-10-09-fem-feedback-retained/SUMMARY.md).

### S04 — Reuse DEM resources and neighborhoods safely

**Scope:** `tire_dem.rs`, DEM shaders, existing radix/LBVH helpers only where necessary, DEM tests and the soil benchmark.

1. Retain key/index/tree/neighbor/output capacities and prepared resources between steps. Remove per-substep full-state cloning and conversion from production paths.
2. Compare uniform-cell lists against LBVH on uniform beds, clumps, sparse particles, and varied grain radii. Do not assume a tree is always faster.
3. If using Verlet-style cached neighbors, define an interaction cutoff plus skin and rebuild when a conservative relative-displacement bound consumes the skin. Include particle size changes, insertions, material changes, and boundary motion in invalidation.
4. If retaining BVH topology, refit all affected bounds before queries. Rebuild for topology quality or membership changes; refit is not permission to query stale bounding boxes.
5. Record candidate/accepted neighbor counts, rebuild cost, memory, and worst occupancy. Check CPU/GPU pair sets on small adversarial examples.

**Gate:** `make test-soil-gpu PHASE=S04`; no missed physical contacts versus an exhaustive small-scene oracle for high-speed motion, crossing cells, new particles, boundary motion, or clustered beds. Rebuilds occur on every documented invalidation. `make bench-soil CASE=dem MODE=compare` measures unchanged physical timestep and contact law. Never lengthen a DEM timestep solely to hit a speed target.

**Experimental checkpoint:** `set_lbvh_rebuild_interval` defaults to 1. Larger intervals skip Morton sorting between rebuilds but recompute leaf and internal bounds before every contact query. Three focused GPU tests cover exact directed contact pairs, moving contacts, radius changes, rebuild scheduling and membership invalidation. The diagnostic pair audit specializes the production traversal and is restricted to small scenes; it is not a production readback path.

The historical legacy 50k-grain comparisons failed the frozen 2e-5 final velocity
tolerance: original refit-32, the reverted compensated-summation experiment and
refit-4. Their lower timings remain rejected experiments. Production ordered
refitting now supersedes that third arm and passes the unchanged original gate;
do not retroactively count the historical runs as passing. Full rebuild remains
the default and independent reference.

**First-divergence follow-up (2026-10-13):** the identical-input diagnostic now isolates
the first rounding difference at step 5/particle 42420. Selected contact sets and
pair forces match exactly; force-addition order differs and recorded prefixes
reproduce each production output. The independent refit trajectory still fails.
The new ordered-batch candidate preserves all 52 full states for the 50k fixture
and a dense overflow fixture, with initial paired timing evidence above. This
supersedes the outstanding diagnosis task, not the legacy refit failures or phase
completion. Production opt-in wiring, public lifecycle/resource regressions and
two matched-work timing packets now pass (see the production increment above).
Worst-occupancy/quality policy and absolute budgets remain. Do not guess more rebuild intervals or loosen
tolerances. [Diagnostic](benchmarks/soil/2026-10-13-dem-first-divergence/SUMMARY.md).

### S05 — Add active-tile MPM

**Scope:** `crates/molla-solvers/src/mpm.rs`, `shaders/mpm.wgsl`, reusable sparse-grid helpers where suitable, and new `mpm_active_tiles_gpu.rs`.

1. Preserve the dense implementation as a reference and a possible faster path for fully occupied small grids.
2. Build active tile identities from particle support, rigid/implement influence, and required boundary stencils. Allocate/grow a checked tile pool outside steady-state hot loops.
3. Restrict reset, grid update, rigid processing, and applicable solver sweeps to active tiles plus halo. Correctly clear retired/reused tiles so old mass or momentum cannot leak.
4. Reorder or index particles by tile while preserving stable external particle/material identity. Handle particles moving across several cells/tiles through a conservative activation or explicit step rejection.
5. Implement sleeping only after active transfer correctness. Retain stress/plastic history and wake on new load, neighboring disturbance, material edit, or penetrating implement.
6. For implicit stepping, construct complete stencil support and explicit boundary conditions. Omitting a neighboring active unknown changes the linear system and is not an acceptable optimization.

**Gate:** `make test-soil-gpu PHASE=S05`; compare dense and sparse results on identical material/timestep for free fall, pile collapse, rigid contact, tile crossings, inactive/reactivated patches, and multi-material boundaries. Check particle count/mass and momentum balances. `make bench-soil CASE=active-domain MODE=compare`: enlarging empty domain at fixed active work must not add full-volume reset/update work; log active cell counts and timing. Measure sparse overhead on fully dense controls and retain explicit policy for choosing the faster equivalent path.

**Partial implementation checkpoint:** checked `set_active_grid(true)` selects GPU-generated indirect dispatch over the original dense arrays. `MpmSimulation::new_sparse` instead allocates physical grid fields in a fixed pool of 4³-cell tiles plus a power-of-two hash table with at least twice the tile capacity; active dispatch is mandatory. Dense construction remains the default. Each substep clears previous slots and their recorded hash buckets, rebuilds a deduplicated map from particle support and rigid influence, then executes the existing transfer/material/coupling order. Particle support includes a one-cell halo for the existing implicit stencil. Metadata occupies the tail of the existing stiffness buffer, avoiding an additional main-solver storage binding. P2G/G2P and implicit neighbors use hashed compact addressing; grid sweeps use their known physical dispatch slot directly.

Ten focused fixtures have passed: four dense/active scheduling fixtures, five compact-pool fixtures and a private adversarial hash test. They cover explicit/implicit stepping, both transfer modes, materials, rigid spheres, partial edge tiles, reseeding, continuous transport, growth, overflow, resource accounting, collision chains and retirement. With 129 particles and a 64-tile pool, 32³/96³/512³/1536³ domains all use exactly 230,928 grid/lookup bytes and schedule eight tiles for the fixed patch. A remote 1536³ patch checks mass and gravity response at world indices beyond signed-i32 range. A domain-covering rigid exits marking after overflow instead of enumerating the entire domain. This is verified storage/work-count evidence, not timing acceptance. Empty-to-reseed is covered; immediate retirement on an empty no-op step is not claimed.

**Compact-mode consumer contract:** `step_n_with` enqueues work without a blocking fault query. A fault-aware caller must inspect `read_active_grid_status_with` before accepting physical advancement or coupling outputs. Overflow is latched, stops further particle updates, and reports completed substeps; `read_active_grid_stats_with` returns an error when the latch is set. The substep field is a wrapping u32 count within a grid epoch; compute deltas with wrapping subtraction and track explicit resets/growth as new epochs. A partially completed batch is not an all-or-nothing transaction. `reserve_sparse_tiles` grows outside steady-state stepping while retaining particle/material/history buffers, and starts a new grid epoch on the next nonempty step. `reset_active_grid` clears the latch/grid without reseeding particles. Dense world-index diagnostics reject compact storage; `read_grid_storage_mass_momentum` explicitly returns physical slot order. Asynchronous fault telemetry and device-loss handling remain part of S03/S10.

**Next bounded increments, in order:**

1. Resolve the rigid-wrench repeatability finding below and collect uncontended paired evidence before changing the default. `CASE=active-domain` now compares dense, dense-backed active and compact storage on matched 50k-particle patches at 32³/64³/96³ and fully occupied 32³ controls, with explicit/implicit stepping and overlapping/remote spheres. It rotates mode order over twelve measured rounds after two warmups. Initialization, resets and diagnostic readbacks are outside completed solver timings. All eight cases are collected before failure if any parity check fails. Compact diagnostics reconstruct world-indexed mass/momentum from physical slots and tile IDs, so grid comparisons retain the original per-cell tolerance rather than using only totals.
2. Extend physical cases with pile collapse and zero-particle lifecycle. Genuine continuous travel is checked in both dense-backed and compact modes: 129 particles move one metre over 128 unchanged steps with dense parity and independent mass/gravity-momentum bounds. Overflow tests cover immediate freeze, explicit recovery, and a batch completing two of three steps before growing the pool and advancing only the missing step. These are transfer/lifecycle fixtures, not agricultural calibration.
3. Use the opt-in S06 stage profiler to improve the retained stable particle tile index described below. Bounded world lookup, stable indexing, the three-way comparison and a default-off deferred-lookup experiment are implemented; they do not remove all scatter cost or its repeatability issue. Report physical fields, lookup metadata, particle memory and transfer costs separately. Existing contended stage diagnostics do not justify changing the default.
4. Introduce sleeping/history retention after transfer correctness. Test wake propagation, distant rigid activation, slot reuse and repeated domain traversal. Report active memory separately from persistent archived field state.

Dense construction still has its original dispatch/domain limit. Compact construction checks physical-pool/hash capacities and passes a 1536³-domain fixture; world coordinates still use a checked u32 cell index, so this is not arbitrarily large or streamed world support. S07 owns persistent field state and world streaming.

**Stable particle-index increment — 2026-10-09:** `prepare_particle_tile_ordering(queue, capacity)` creates retained tile-key/identity buffers and a fixed-length stable radix dispatch outside stepping. `set_particle_tile_ordering(true)` explicitly enables the prepared path; the default is off. Every substep regenerates keys from current positions and sorts `(world tile, original particle ID)` pairs. Equal tile keys preserve ascending original identity; padded capacity entries use the maximum key and remain after all live particles. Out-of-domain positions receive clamped ordering keys only: physical positions, support activation and boundary handling are unchanged. The exact-permutation fixture covers partial workgroups, duplicate tiles, changing/empty populations and remote 1536³ coordinates.

Only P2G dispatch order changes. Positions, velocities, APIC affine matrices, deformation/plastic history, mass, volume, tire compaction and material identities remain in their original arrays/slots. A copy of sorted IDs occupies a checked tail after the material-ID prefix, avoiding another main-solver storage binding. Preparation copies the existing material prefix on-device and rebuilds its binding without downloading history. Dense and compact, legacy and local-accumulator P2G, and deferred compact lookup can use the same index. The ordinary disabled dense path retains its original single compute pass per batch.

Capacity is an explicit cost contract: the sort processes the **prepared capacity**, not only the live count. Choose capacity deliberately; shrink/reprepare outside the hot loop when warranted. Counts above the prepared capacity reject an enabled nonempty step before uploads or physical advancement; there is no silent fallback. Reprepare is explicit, supports growth/shrink, and preserves pending material uploads. Disabling retains the allocation. Particle reseeding below capacity, changed materials/positions, and compact-pool growth do not require a new sort allocation; the index is rebuilt every substep. Reference-sort ordering storage is `20 × capacity + 1024 × ceil(capacity / 256) + 96` bytes, excluding existing particle arrays, allocator padding and pipeline/backend overhead. At 50k capacity this is 1,200,800 bytes. Reference requests add 32 uniform-upload bytes per submitted batch and 13 dispatches plus one on-device identity copy per substep; no observation/download or device buffer creation occurs in the fixed-capacity step. The parallel variant below adds 1024 retained bytes and uses 17 dispatches per indexed substep instead.

The original reference-sort checkpoint passed four focused GPU fixtures via `make test-soil-particle-order RELEASE=1`, also included in S05. They cover exact ordering/history retention, six dense/active/compact and explicit/implicit/transfer combinations with mixed materials and sphere coupling, overflow freeze/recovery, 128-step transport, resource counters and timestamp query limits. Final position/velocity comparisons use absolute 2e-5 gates; history/wrench comparisons use 2e-5 absolute/relative scaling. Transport independently checks aggregate mass and gravity/momentum, retaining the existing conservation bounds. An initial stronger per-particle analytic displacement assertion failed; diagnostics show the unchanged reference also exceeds that proposed bound. It is preserved as diagnostic output, not claimed as satisfied. These fixtures do not establish large-workload repeatability or agricultural acceptance. See [ordering contracts and audit](benchmarks/soil/2026-10-08-mpm-particle-order/SUMMARY.md).

**Benchmark checkpoint:** the eight-case `CASE=active-domain` run is a failing diagnostic, not accepted speed evidence. Four patch cases failed rigid torque at the frozen 2e-5 tolerance. A separate 50k implicit repeated-state diagnostic found a four-step dense-versus-dense torque failure as well as dense-versus-active failures, while particle, grid and force comparisons stayed within their gates. A subsequent identical-pre-imprint-grid replay produced bit-identical grid fields across twelve dense/active coupling executions and stayed within the wrench gate (maximum torque error 0.19669533 times tolerance). The larger failures involve upstream state variation as well; this does not prove a particular scatter operation is the entire cause. Keep the reproducer and isolate upstream contributions before changing numerics; do not weaken tolerance or rerun until a random pass.

The compact marker claims a hashed key, publishes a physical slot and records its bucket for retirement. A focused test forces fifteen keys into the same initial bucket, includes tile zero and duplicate producers, fills all sixteen physical slots, retires them using the actual reset kernel, reinserts a new colliding set, and checks overflow/recovery. Lookup metadata no longer grows with empty domain size.

The latest [three-way benchmark](benchmarks/soil/2026-10-08-mpm-three-way-domain/SUMMARY.md) still exits 2: active-mode torque fails four patch cases; compact-mode torque fails the 96³ explicit and implicit cases. Per-cell grid, particle and force checks preceding those torque checks pass. GPU utilization was 88% before / 93% after; timings are contended diagnostics, not accepted speed evidence. Keep dense default and active/compact modes opt-in. See [bounded-hash verification](benchmarks/soil/2026-10-08-mpm-hashed-grid/SUMMARY.md); older page-table/timing checkpoints remain historical evidence.

### S06 — Optimize transfer, layout, and fusion only where profiling supports it

**Latest S05 index gate:** the subsequent parallel-sort integration run passes exact indices/lifecycle, mixed-material sphere coupling and query/profiling checks, but the 128-requested-step transport fixture fails its unchanged absolute velocity tolerance. Its first reported mismatch is `4.0000687` versus `4.0000467` at flattened velocity index 272 (about 2.2e-5 error, limit 2e-5). Position error and aggregate mass/momentum checks preceding the failure pass. The final completion-counter assertion is after the failed comparison and was not reached; do not count it as verified by this run. This failure is retained in [parallel radix evidence](benchmarks/soil/2026-10-09-parallel-radix/SUMMARY.md). No tolerance, timestep, material, workload or substep count was changed, and the failed fixture was not rerun until a random pass. Stable sort correctness does not establish trajectory repeatability. Add matched repeated-state/order controls before attributing the error or adopting either ordering path.

**Resolution of that bounded gate (2026-10-11):** matched controls subsequently
show reference-sort own-repeat drift too. Correcting APIC moment-arm coordinates
then passes both the full 1536-comparison control matrix and the original batched
128-step integration fixture, including its final completion-counter assertion.
Maximum integration velocity difference is 5.24521e-6 at the unchanged 2e-5 bound.
Historical failures above remain evidence; the large 50k torque/performance gates
are not closed by this repair and ordering remains opt-in.
[Correction and integration evidence](benchmarks/soil/2026-10-11-local-transfer/SUMMARY.md).

**Scope:** MPM storage/layout and shader transfers; keep public physical parameters and reference path unchanged.

1. Profile P2G, grid, G2P, plasticity/SVD, rigid coupling, and raster separately after S03/S05. Rank by completed GPU time, not shader length.
2. Evaluate spatial ordering and workgroup/subgroup reductions against the existing local accumulation path. WGSL capabilities and the actual adapter determine which subgroup/atomic techniques are available.
3. Prototype G2P2G behind an explicit experimental option. Document exactly when each stress/plasticity update and boundary interaction occurs. Correctly initialize/drain first and last transfers in a batch.
4. Measure structure-of-arrays/AoSoA layouts, coalescing, scratch traffic, and register/workgroup pressure. Cache pipelines; do not add compilation to a live update.
5. Retain f32 as the baseline. Quantization, mixed precision, or approximate math is a separate experiment with range, drift, and physical-error evidence; do not enable it merely because a graphics paper used it.
6. Do not introduce CUDA or another backend as an incidental patch. If WGSL prevents the required performance, present measured limitations and the maintenance/portability tradeoff for a separate decision.

**Gate:** `make test-soil-gpu PHASE=S06`; unfused/fused parity, conservation and long-run drift within frozen tolerances. `make bench-soil CASE=mpm-transfer MODE=compare` measures completed simulated-time windows at matched work. Keep the optimized path only when it improves the declared workload without unacceptable regressions. A rejected experiment is a valid outcome, not a reason to loosen tests.

**Profiling increment:** `prepare_step_profiling(max_substeps)` allocates one timestamp query set and two retained buffers for 1–204 substeps. `step_n_profiled_with` records timestamps inside the original passes, resolves them once and submits without a host wait; `read_step_profile` is a separate explicit blocking diagnostic. Device creation must explicitly request timestamp-query, in-encoder and in-pass features. Ordinary stepping neither requires those features nor allocates query resources. An unread capture rejects another profiled batch before uploads/advancement, but ordinary stepping may continue without overwriting it. Stages cover initialization, grid retirement/reset, active marking, rigid reset/imprint/force, P2G, grid update, grouped Jacobi sweeps, G2P and plasticity. Profile durations are labelled **requested**, not completed physical time: a compact overflow can stop advancement while timestamped no-op dispatches continue. Fault-aware consumers still inspect the existing completion/status API.

**Transfer experiment:** compact two-pass P2G can opt into `set_deferred_p2g_lookup(true)`. Workgroup-local tags then contain world-cell identities; hash lookup happens only for global collision fallback or local-accumulator commit, rather than before every local contribution. Particle/material identity, arithmetic and solver stage order remain unchanged. Dense and legacy modes reject enabling this option; selecting legacy P2G clears it. The original compact implementation remains the default and reference. Five S06 GPU fixtures pass, including profiling lifecycle/resource accounting and mixed-material, rigid, edge, scattered, remote-world and overflow cases for the deferred path. This is not large-case or long-run acceptance.

The optional particle ordering stage has its own `particle-ordering` timestamp interval covering key construction, all radix passes and the identity copy; P2G time is reported separately. Profiles record whether ordering was enabled. Preparation still accepts 1–204 substeps; ordered profiled batches are capped at 186 by the 4096-query limit and reject larger requests before uploads. Unprofiled batches retain their existing size contract. The additional timestamps do not change solver stage order or hide sorting in a setup cost.

`make bench-soil CASE=mpm-profile MODE=diagnostic` supplies paired profiled/unprofiled samples, preserving raw stage and completed wall times separately. A six-case baseline run pointed to P2G as the largest median individual stage; an eight-case follow-up includes the deferred variant and checks active completion counters per batch. Both were externally contended (initial 93%/79%, follow-up 93%/99% utilization). The deferred variant's explicit and implicit medians do not show a consistent improvement under this load, and **no speedup or instrumentation-overhead ratio is accepted**. The large profiling diagnostic does not check physical parity and is not handled by the active-domain-only result checker. See [profile/transfer contracts](benchmarks/soil/2026-10-08-mpm-profile-contracts/SUMMARY.md) and [deferred diagnostic](benchmarks/soil/2026-10-08-mpm-deferred-lookup/SUMMARY.md).

The [twelve-case ordering diagnostic](benchmarks/soil/2026-10-09-mpm-particle-order-profile/SUMMARY.md) adds dense/compact ordered arms: 72 recorded samples, four physical substeps per sample, with sorting inside measured wall/stage scope. It completed in 7.84 seconds but GPU utilization was 96% before / 100% after. Sorting cost was substantial and mode-to-mode stage times varied strongly; **no ordering speedup is accepted**, and ordering remains disabled by default. The retained radix helper uses a single-workgroup tiled global scan and a serial scatter within each block; this motivated the parallel variant implemented below, rather than further whole-simulator reruns. Preserve stable ties/identity and regress the shared primitive's other consumers before replacing it. Large-case torque repeatability, pile collapse/sleeping, fusion/layout experiments and uncontended integrated acceptance remain open.

**Parallel stable radix increment:** `RadixSortKernel::prepare_parallel` is a separate explicit preparation path; `prepare` and ordinary dispatch keep their reference implementation. Four eight-bit digits each use the existing histogram, 256 independent per-digit block scans, a scan of the 256 digit totals, and parallel scatter. Each input lane sets a unique bit in a per-digit workgroup membership mask; population counts of earlier lane bits give stable local rank without a serial lane-zero scatter or subgroup features. Scans carry prefixes across blocks larger than one 256-entry scan tile. Integer key/payload semantics and stable ties are unchanged.

The retained parallel sort uses 16 dispatches rather than 12, the same seven scratch/uniform buffers and six bindings, four pipeline handles rather than three, and 1024 extra scratch bytes for digit totals. Its preparation validates length/alias/capacity and required workgroup storage; fixed-capacity encoding allocates/uploads/downloads nothing. `scratch_bytes()` and `dispatch_count()` describe the selected prepared path. `make test-soil-sort RELEASE=1`, also included in S06, passes ten tests: seven existing ordinary-sort cases, the original retained-sort queue test, and two new parallel exact/lifecycle/resource fixtures. The new exact fixture covers 55 length/pattern combinations through 100,003 pairs, including 65,535/65,536/65,537 scan-boundary sizes, all-equal/max-u32/high-byte keys, duplicates, reversed input and untouched suffix sentinels. A second fixture queues independent retained sorts and snapshots without waits, drops owners before readback and verifies exact payloads plus zero hot allocations/transfers and 16 recorded dispatches. Existing DEM reference-sort consumers pass four focused regression fixtures after the shared dispatcher refactor.

MPM can explicitly select `prepare_parallel_particle_tile_ordering(queue, capacity)` without enabling ordering. Switching variants at the same capacity correctly reprepares resources and preserves particle/material state. Profiles and benchmark manifests record the parallel choice; parallel index storage at capacity 50k is 1,201,824 logical bytes. The standard preparation continues selecting the reference sorter. The integration failure above means neither option is promoted to the default. The profiling harness now has optional dense/compact parallel-ordered arms, but that expanded full stage matrix was not run at this checkpoint.

`make bench-soil CASE=radix MODE=compare` now provides a short isolated primitive comparison with stable CPU-oracle checks after every sample, two warmups and six measured alternating-order pairs for 513/50,000/100,003 elements. [Recorded evidence](benchmarks/soil/2026-10-09-parallel-radix-paired/SUMMARY.md): 36 measured samples, 0.51-second test, unchanged key/payload workload. At 50k elements the observed median completed sort time was 0.970745 ms reference versus 0.174477 ms parallel. The GPU was externally contended (92% before / 98% after), so this is a promising measured component result, not clean-machine latency acceptance, a solver speedup or simulated-time evidence. The active-domain-only result checker does not support this primitive log schema. Preserve the long-transport failure and collect controlled physical-repeatability evidence before an adoption decision.

**Fixed-input repeatability and rejected gather trial (2026-10-09):** a six-replay control restores identical pre-imprint grid buffers before rigid coupling; all output grid and wrench words are identical. Repeating P2G against unchanged particle inputs instead varies mass, momentum and stiffness. For this captured fixture, P2G is the earliest observed varying stage; this does not prove all wrench reductions are deterministic under other inputs/schedules.

A fixed-order GPU gather was implemented to test that finding. It reuses stable particle-tile sorting, caches per-particle stress/stiffness and visits neighboring tile ranges in a fixed order, with one owner per grid cell and no floating atomic accumulation. Dense-backed/compact fixed-input fixtures repeat bit-exactly; five-substep particle/history results are bit-identical across submitted and caller-encoded schedules, with force error below 2e-5. However, the 50k diagnostic has a 3.2424927e-5 stiffness discrepancy against two-pass scatter and worse observed completed-transfer medians (9.8819765 ms gather versus 1.636508 ms scatter; GPU utilization 90%). These are transfer-only samples, not simulated-time or frame-budget evidence. The trial is **rejected for production** and isolated behind `cfg(test)` as a numerical research reference; no new gather option or storage is present in the production solver. Default scatter, ordering and viewer scheduling remain unchanged.

`make test-soil-gather RELEASE=1` exercises the isolated small fixtures and is included in S06. The known-failing 50k diagnostic is explicit: `make test-soil-gather RELEASE=1 SOIL_GATHER_FILTER=gather_transfer_benchmark SOIL_GATHER_ARGS=`. Preserve its failure and do not tune/rerun it for a lucky result. A faster bounded transfer design, large-case numerical agreement and full physical-time acceptance remain open. [Stage isolation and trial evidence](benchmarks/soil/2026-10-09-mpm-stage-repeatability/SUMMARY.md).

**G2P/plasticity fusion trial (2026-10-09; test-only, not adopted):** a single dispatch invokes the original gather and constitutive-projection bodies in order for each particle, with the same layouts, materials and compact-grid specialization. Generated source changes only the two entry declarations and appends the ordered wrapper; the main WGSL source is unchanged. It needs no extra particle scratch. Five GPU fixtures pass: fixed-input bit-exact replay of all particle/grid words across dense, active and compact modes and 1/63/64/65/129 particles; six mixed-material 32-step trajectory cases with separate reference-repeat controls; four prescribed-sphere force/torque cases; prepared-record/resource/profile lifecycle; and overflow freeze/growth recovery. Full-history and reaction comparisons retain absolute 2e-5 gates. These fixtures do not close older ordering/refit/force failures or actual-machine acceptance.

A paired 50k diagnostic advances four unchanged f32 0.0001 substeps per sample (about 0.0004 simulated seconds), with two warmups and six measured alternating-order pairs per mode. All completed pairs pass their full-history checks. Dense medians are **7.142925 ms separate vs 9.358464 ms fused**; compact medians are **12.876606 ms separate vs 16.564002 ms fused**. GPU utilization was 94–95%, so this does not prove an uncontended regression or gain. It provides no basis to promote fusion. The first run completed the dense case but stopped in compact warmup on a benchmark assertion that incorrectly treated a cumulative completion counter as reset by reseeding. The corrected check requires exactly four additional completed steps; only the unfinished compact case was then run. Dense timing was not retried or replaced.

The candidate, mode fields, profile annotation and preparation API are now entirely behind `cfg(test)`; production retains its original separate dispatches and no fusion resources/option. `make test-soil-fusion RELEASE=1` covers the five fixtures; the explicit timing command adds `SOIL_FUSION_FILTER=paired_fifty_thousand_benchmark SOIL_FUSION_ARGS=` and optional `MOLLA_FUSION_BENCH_MODE=dense|compact|both`. S06 fast/GPU lanes include source integrity and focused checks, respectively. Final test-only checks and three existing production-profile regressions pass; facade/viewer check passes. [Fusion evidence, preserved benchmark failure and binaries](benchmarks/soil/2026-10-09-mpm-fused-gather/SUMMARY.md). No timing acceptance or default solver speedup is claimed.

**Single-lane private machine dynamics (2026-10-10, S06):** added `FeatherstoneExecution::ExperimentalSingleLane`, selectable with `SOIL_TRACTOR_BENCH_EXECUTION=single-lane`. It changes only dynamics workgroup width and matching dispatch geometry, retaining private scratch, storage rounding, constraints, generic refresh and existing unrestricted drift. Generic remains default; the new profile has a distinct program-cache/manifest identity and checked workgroup count/storage limits. The single-tractor paired fixture is bit-identical, with seven measured dynamics passes totaling **68.159392 → 55.421632 ms**. A separate alternating-order, three-translated-tractor fixture also preserves all state/residue words and stop/cap/compliance/disabled-joint behavior, with **75.023776 → 62.954240 ms**. These are component observations, not 50k or 4 ms acceptance. Eleven CPU checks, strict CLI selection, formatting and facade compilation pass. The actual six-body friction-runtime check passes in **382.07 seconds** including cold setup, preserving impulse/momentum/energy, rollback, zero steady buffer allocations and profile identity through model rebuild. The broader cross-profile cache GPU check timed out before a result and remains incomplete. [Evidence, bounds and incomplete gates](benchmarks/soil/2026-10-10-single-lane-dynamics/SUMMARY.md).

**Single-lane full-workload diagnostic (2026-10-10, S06):** the new profile and same-source Generic control each complete 33 frames / 0.55 simulated seconds at 50k particles. Source/executable hashes match and manifests differ only in execution identity. Reference/candidate median wall times are **223.133371 / 210.844144 ms** (5.51% lower in this sequential sample), with dynamics **109.563520 / 95.636256 ms**. Both miss all 30 measured 60 Hz deadlines; zero steady buffer allocations and identical dispatch/transfer counts do not close the budget. **Trajectory comparison fails:** machine velocity differences first cross the existing comparison bound at frame 15, maximum body-position difference is 0.000306115 m and maximum final particle-position difference is 0.002286226 m. Native atomic variation/contact transitions remain present, so this pair does not isolate the cause. Generic stays default; no promotion, 4 ms, calibration or rendering acceptance. Independent frame audits and deliberately corrupted percentile/workload checks pass; ambient telemetry and failed physical comparisons are retained. [Actual workload evidence and next replay target](benchmarks/soil/2026-10-10-single-lane-live/SUMMARY.md).

**Single-lane identical-input isolation (2026-10-10, S06):** the actual native 50k unilateral trajectory now compares Generic/storage dynamics and corrected drift against a discarded single-lane candidate using identical resident machine inputs, forces, compensation and projected rates. All **1122 dynamics plus 1122 drift checks** across 33 frames pass bit-for-bit; an independent audit covers 2244 records and 460,020 words. Runtime/source exits are zero; cold-inclusive duration is 361.71 seconds, not stepping latency. This narrows attribution but does not reproduce the exact independent soil histories or clear their failed trajectory bounds. Keep Generic default and reuse the saved evolved P2G checkpoint for further accumulation/contact work, rather than repeating machine-width or full-timing comparisons. [Exact invariant and remaining gaps](benchmarks/soil/2026-10-10-single-lane-shadow/SUMMARY.md).

**Compensated native transfer experiment (2026-10-10, S06):** a test-only second atomic accumulator corrects native P2G rounding error before grid update. The unchanged evolved 50k checkpoint passes component bounds and positive-mass support checks over eight alternating pairs, without advancing simulated time or changing particle history. The candidate has zero observed repeat differences in this sample, not a general determinism guarantee. It requires native f64 residual evaluation and 4,194,304 additional bytes on the 64³ grid. Six measured pairs show median GPU reset/transfer/grid time **0.088784 → 0.191808 ms** and completed wall **0.1830915 → 0.296984 ms**. This is slower, so it remains test-only; no default change or integrated acceptance. [Arithmetic failure, correction and timing evidence](benchmarks/soil/2026-10-10-compensated-p2g/SUMMARY.md).

**Selective contraction-control experiment (2026-10-10, S06):** a test-only Vulkan path decorates twelve add/subtract instructions in the two compensated-sum helpers while removing two intermediate roundtrips. Inspection found that the initial ReadZeroSkipWrite generation lowered indexed atomic load/store to ordinary operations; explicit Restrict bounds checks now preserve the intended initial/final atomic barriers. A pre-submission guard and negative compiler-output fixture enforce that assumption. Source reconstruction, 4105 independent arithmetic cases and a separate 4105-case composed caller-product comparison pass. The corrected three-translated-tractor comparison completes in **346.33 seconds**, but **619 cumulative state/residual words differ** across observations. Seven measured dynamics passes total **98.541056 / 72.784576 ms**, not an accepted speedup because exact compatibility fails. Production defaults remain unchanged. Isolate generation-policy versus arithmetic effects before another full workload; do not repeat the unchanged failed comparison. [Corrected generation, regression guard and current failure](benchmarks/soil/2026-10-10-selective-restrict/SUMMARY.md); [earlier timeout and specification](benchmarks/soil/2026-10-10-selective-no-contraction/SUMMARY.md).

**Unchanged-arithmetic passthrough control (2026-10-10, S06):** the same guarded SPIR-V setup now has a control preserving both sum bodies and applying zero NoContraction decorations. It matches ordinary WGSL bit-for-bit on all eight observations of the same translated/constraint-mutating machine fixture; the independent capture audit confirms zero differences in 3744 compared words. This narrows the earlier failure to the arithmetic transformation in this fixture rather than the tested generation setup alone; it does not identify the exact divergent operation. Three CPU checks, 4105 arithmetic cases, formatting and the facade build pass. Saved named output fields and a strict Make-driven offline auditor avoid repeating the cold run merely to inspect discrepancies. Keep the twelve-decoration optimization test-only; next constrain the change to the eliminated intermediate barriers. No integrated 50k or real-time acceptance follows. [Control, audit and reproduction](benchmarks/soil/2026-10-10-passthrough-control/SUMMARY.md).

**Contraction-control repair outcome (2026-10-10, S06):** two bounded repairs are rejected. Protecting only four intermediate subtracts loses the 1e-20 residual in the component oracle, so no machine run used it. Protecting the intermediate/residual arithmetic with ten decorations passes both 4105-case component checks but still produces **619 cumulative machine-state differences**; the independent capture audit confirms the failure. Maximum state/rate differences are approximately **5.82e-11 / 3.73e-9**, not grounds to replace the exact gate with a looser one. Generic reference snapshots repeat exactly across all eight observations. Four structural tests, formatting and the facade build pass; all processes are terminal and defaults are unchanged. Stop guessed decoration variants or unchanged cold reruns: further arithmetic work requires a concrete divergent-operation explanation; independent unfinished soil-plan implementation can proceed. [Preserved failures, per-field audit and stop decision](benchmarks/soil/2026-10-10-intermediate-rounding/SUMMARY.md).

### S07 — Persist field state while bounding dynamic work

**Scope:** reusable soil state/query APIs in `molla-vehicle`, device ownership in solvers/kernels, and public exports. New modules are allowed within those domains; no unrelated asset or simulator integration edits.

1. Define authoritative material identity, mass/bulk volume, compaction/history, and world-tile coordinates. Distinguish fast heightfield state from volumetric particle state; do not imply identical physical fidelity.
2. Introduce activation/deactivation and tile streaming with explicit save/load/checkpoint behavior. Persistent ruts must survive leaving and returning to an area, not just remain in one viewer allocation.
3. Preserve mass, momentum, soil type, and relevant plastic history during representation changes. Compression can reduce volume at fixed mass; excavation transports mass. A lower height alone must not erase material.
4. Share overlapping machine regions; define deterministic ownership and transfer across tile boundaries. Keep free-moving material active outside the original machine patch.
5. Separate authoritative physical resolution from rendering LOD. Any physical coarsening requires documented error bounds and its own calibration/benchmark class.

**Gate:** `make test-soil-gpu PHASE=S07`; repeated activate/deactivate and save/load preserve declared state, overlapping regions do not duplicate mass or reactions, no particle disappears on tile boundaries, and resource usage remains bounded during a long traverse. `make bench-soil CASE=field-stream MODE=compare` holds active work fixed while increasing visited field area; record archive memory/I/O separately from active solver cost.

**Implemented checkpoint increment (2026-10-09; not the full gate):** `MpmSoilCheckpoint` and `MpmSimulation::{checkpoint_soil_with,restore_soil_with}` preserve the nine physical particle/history streams, physical configuration and local material table. The versioned, little-endian archive preserves float bits and uses a BLAKE3 digest. Decode bounds particle payload allocation and rejects invalid dimensions, material IDs, non-finite history, nonpositive mass/reference volume, unknown versions and malformed data. Configuration capacities returned by `config()` still need device validation when constructing a solver; the decode limit is not a GPU allocation authorization.

The original `checkpoint_soil_with` capture is an explicit **blocking archive operation**, with one particle readback plus active-grid fault metadata, not a per-step path. Pending material edits are included without consuming them; overflow is rejected even after requesting a deferred reset. Restore validates before submitting one copy transaction into retained buffers, invalidates prepared steps, drops count-dependent internal caches and rebuilds grid scratch on the next step. It requires matching physical settings, adequate capacity and no attached rigid bodies. Queue/device pairing is a caller contract, and device loss is not transactional rollback. External terrain snapshots, count-dependent consumers and observation generations must be refreshed/reset by their owner.

This primitive is volumetric MPM state, **not** a heightfield conversion, automatic sleeping/activation system or whole-machine checkpoint. Its particle slots and material-table indices alone are not global persistent identities; the tile-archive layer below supplies explicit stable IDs. Rigid state/reactions, clocks, grid scratch and scheduling caches are intentionally not serialized. Archive I/O/durable replacement are caller-owned. [Checkpoint evidence](benchmarks/soil/2026-10-09-mpm-soil-checkpoint/SUMMARY.md) does not close the full S07 gate.

**Implemented tile/archive increment (2026-10-09; runtime streaming still pending):** `molla-vehicle::soil_state` supplies signed world-tile coordinates/layout, caller-assigned stable particle/material/region IDs, and bounded `SoilTileResidency` metadata. Region overlaps share one tile entry; the lowest requesting region ID is a deterministic coordinator, **not** an exclusive force/contact owner. Separate pins keep awake soil resident outside machine regions when supplied by the physics owner. Plans validate region/tile/input-claim limits before commit; canceled, stale and foreign plans cannot change residency. The 2,000-update metadata traverse retains four live entries rather than accumulating visited tiles. This is CPU lifecycle bookkeeping, not a CPU physical solver or evidence of bounded GPU streaming.

`MpmSoilTileArchive` partitions an explicit checkpoint by current position, saves the layout and global IDs in a checked versioned container, and assembles deduplicated tile selections in global-particle-ID order. Unknown tiles and particle-capacity overflow fail rather than silently dropping material. `writeback` accepts updated GPU-observed state in the selected slot order, repartitions moved particles, merges destination tiles and preserves untouched records. Stale/foreign selections and changes to count, mass, reference volume, material identity or physical configuration are rejected before archive mutation. All nine per-particle streams are retained bit-for-bit; no coarsening or visual height stamping is involved. Sorting by global ID can change later unordered reduction order, so this is not a solution to the older force-repeatability failure.

The S07 CPU/GPU lanes now include these focused contracts. Actual GPU fixtures demonstrate 129 evolved particles in four archive tiles, 91 unique particles for overlapping requests, updated-state writeback followed by exact region revisit, and a globally identified particle crossing a negative-to-positive tile boundary with conserved mass/history. [Tile ownership/archive evidence](benchmarks/soil/2026-10-09-soil-tile-ownership/SUMMARY.md). Still required: a live residency owner that atomically coordinates archive and device state; GPU-generated awake/halo pins and physical sleep/wake criteria; identity-preserving device compaction; boundary/halo physical correctness; asynchronous retained tile transfers; rebase/large-world stepping; durable incremental tile I/O; actual overlapping-machine reaction ownership; bounded physical long-traverse and `field-stream/compare` evidence. Selection/writeback operates on explicit host archive snapshots, using either blocking capture or the asynchronous increment below, and is not wired into the default stepping/viewer path. S07 remains incomplete.

**Implemented asynchronous archive capture (2026-10-09):** `MpmSimulation::prepare_soil_checkpoint_readback` allocates 1–64 retained slots under an explicit logical staging budget. `MpmSoilCheckpointReadback::try_capture` submits copies without a GPU wait; `poll_next` uses nonblocking device polling and returns accepted snapshots in FIFO order. This is required-state capture, not optional telemetry: no automatic coalescing or replacement of old snapshots. Full capacity returns explicit backpressure without allocation, upload or consumed revision. Retrying captures the state at retry time, not the refused boundary; the caller must preserve a mandatory boundary or handle refusal explicitly.

Each accepted request freezes count/configuration/material metadata, pending material edits and source epoch. Later reseeding/config changes do not relabel old GPU data. Tickets distinguish ring/generation/config compatibility from physical freshness. Explicit cancellation invalidates old tickets but retains in-flight slots until map completion. Numerical/overflow errors are reported per snapshot; map/device loss poisons the ring and aborts outstanding requests. Engine/native-device and stable simulation-owner checks reject foreign sources. Ready-snapshot consumption still copies/validates host data, and staging/driver/host costs are not zero. No per-frame CPU archive processing budget or 50k performance acceptance is claimed.

Six new GPU fixtures pass, including three FIFO captures across six actual solver substeps followed by global-ID archive writeback and exact fresh-solver reload. Existing blocking checkpoint and tile regressions pass after sharing copy/metadata/validation helpers. `make test-soil-checkpoint-async RELEASE=1` is included in S07. [Asynchronous capture evidence](benchmarks/soil/2026-10-09-soil-async-checkpoint/SUMMARY.md). This removes an explicit wait only for callers choosing the new archive API; scene 122's required physical CPU terrain wait and the remaining live streaming/acceptance work are unchanged.

**Implemented GPU identity increment (2026-10-09):** `MpmSimulation::restore_soil_selection` validates the source context and restores particle/history streams plus global particle/material IDs in one submission. The first use allocates two retained ID buffers; subsequent restores reuse them. IDs use exact low/high u32 pairs (`array<vec2<u32>>`), with no f32 conversion or shader-u64 feature requirement. `MpmSoilIdentityBuffers` exposes read-only counted prefixes; `MpmSoilIdentitySnapshot` provides an immutable shared CPU mapping and epoch validation for retained consumers. Plain checkpoint restoration, reseeding and material-table replacement invalidate installed IDs; ordinary stepping, index-only particle ordering and grid growth preserve them. No identity buffers are allocated for callers that do not opt in.

Asynchronous capture now retains the request-time identity snapshot when present. `MpmSoilCheckpointCompletion::into_tile_archive` uses that mapping, rather than a later simulation's IDs, and rejects unidentified captures instead of inventing identities. The particle-only checkpoint encoding and `into_result` remain identity-free. Current device ID buffers are read-only: the immutable CPU mirror is authoritative for capture; arbitrary device-side ID writes or physical array permutations are unsupported. Future compaction must update the device IDs and mirror consistently or capture identities from the GPU. This is not an exclusive world-residency lease or proof of reaction ownership.

Three new GPU checks cover shader consumption of 129 IDs (including zero/high-bit/u64::MAX values), exact local-material→global-material mapping, stepping/parallel ordering/grid growth, retained resource reuse, invalidation/rejection and asynchronous identified save/load across source changes/drop. Twelve existing checkpoint/tile/async GPU regressions pass. `make test-soil-identity RELEASE=1` is included in S07. [GPU identity evidence](benchmarks/soil/2026-10-09-soil-device-identity/SUMMARY.md). Production solver shaders, default route and full S07/S08–S10 acceptance status remain unchanged.

**Implemented live-owner increment (2026-10-09):** `MpmSoilWorld` owns one archive, one detached solver, the current global-ID selection, residency metadata and one retained asynchronous capture slot. `begin_region_update` freezes stepping; `poll_region_update` captures the actual submitted boundary, stages conservative archive patches, assembles the deduplicated destination, restores physical/history/ID streams, then commits archive and residency epochs. Foreign engines, overlapping requests, stale export and stepping during capture cannot bypass the owner. Validation/capacity failures before restore leave both authorities unchanged and allow a different request; capture/numerical faults stop the owner. Device loss is not GPU rollback. No mutable solver or archive is exposed; independent reloads still create independent worlds, not a global uniqueness registry.

Every previously active particle is retained through a pin at its **captured current position**, even after leaving all machine requests. This is deliberately conservative retention, not a sleep classification: stationary and moving particles are both kept, and a growing working set eventually returns a capacity error rather than evicting material. Pins are host lifecycle bookkeeping over the already-required capture, not a CPU physical solver or a GPU awake/halo algorithm. Requested empty tiles are valid. Required interaction halos remain the caller's responsibility; the crossing test proves state transfer, not coupled boundary physics. Restore still transfers the full selected checkpoint and creates upload sources; this is not zero-copy device compaction or an accepted hot per-frame route.

`synchronized_archive_bytes` refuses stale saves after stepping and exports the complete identified archive only at a synchronized boundary. Five new GPU fixtures pass: 129 unique mixed-material particles across overlapping requests, eight physical substeps with four subsequent captures and exact fresh-owner reload; particle and metadata capacity rollback/retry; negative-to-positive movement pinned without a requesting region; empty-space/drop/foreign-device/device-loss handling; and actual compact overflow refusing archive replacement. Two CPU owner/transaction tests and seven prior CPU checks pass, plus both existing tile GPU regressions. `make test-soil-world RELEASE=1` joins S07. [Live-owner evidence](benchmarks/soil/2026-10-09-soil-live-owner/SUMMARY.md).

The owner also exposes `last_committed_archive_bytes` and `committed_revision` for explicitly historical recovery after a fault. That export may predate current active physics; its request ordinal is not physical time or a successful-substep count. It does not bypass synchronized-save freshness checks. The live owner is opt-in and not integrated into the viewer or machines. Extend it with GPU-generated awake/halo sets, validated physical sleep/deactivation, identity-preserving device compaction/transfers, incremental durable I/O, large-world rebasing and actual bounded physical field traversal. Preserve all older numerical/performance failures; S07 and full-plan acceptance remain incomplete.

**Retained transfer increment and adoption decision (2026-10-09):** the live owner now skips restoration when its frozen captured selection has the same particle/material IDs. It still validates the snapshot and writes the archive, but issues no transfer upload, copy, dispatch or submission and preserves the existing identity/prepared-step epochs and grid state. This shortcut is valid only under the owner's frozen-current-state contract; the public `restore_soil_selection` operation still restores explicitly supplied history even when IDs match. Archive capture/download and host partitioning costs are not removed.

An optional device transfer gathers all nine physical/history streams and exact u64 IDs using u32 words. Retained records are copied from the current device buffers, new records are uploaded, and output is copied back into the original buffer handles. Mapping generation is a linear merge of sorted global IDs. Three retained buffers use `508*capacity + 16` logical bytes (25,400,016 at 50k capacity). The stream layout and ID CPU mirror update together after submission; count/layout changes invalidate old mapping/prepared-step epochs. Pure compaction, empty/refill and queued reuse preserve every history bit. Material-table changes and pending local-material edits reject rather than remap silently.

**Do not promote mixed device gather as a general speedup.** A short alternating-order 50k diagnostic observed unchanged-selection medians of 4.478606 ms full upload versus 0.574375 ms for the shortcut, but half-import medians were **3.431442 ms full upload versus 5.903832 ms device gather**. The GPU was 92%/83% busy and a facade check build overlapped the diagnostic. These are completed transfer-only component samples, with zero simulated-time advancement, not uncontended latency or the 4 ms soil gate. Each measured output is checked bit-exactly. Preserve the unfavorable mixed result; do not rerun it to obtain a preferred median.

Consequently, changed selections retain full uploads by default and allocate **no experimental transfer scratch**. `prepare_device_transfers(engine, max_scratch_bytes)` explicitly enables the experimental mechanism under a separate logical memory budget; `disable_device_transfers` returns to full uploads. Preparation/disabling reject during a pending boundary. `last_transfer_stats` reports the chosen path, global-ID membership counts and logical upload bytes, not physical elapsed time. The optional route uploads `16 + 4*N + 168*imports` bytes for a changed selection; an unchanged selection uploads zero in either mode.

Four focused GPU transfer tests, five live-owner regressions, three identity regressions and eleven CPU checks pass; the 50k diagnostic also passes its bit-exact checks. Live opt-in expansion retains 65 evolved particles, imports 64 and uploads 11,284 bytes with no hot instrumented GPU buffer allocations. Default unchanged transitions similarly allocate/upload nothing in their transfer phase while retaining the required archive readback. `make test-soil-transfer RELEASE=1` joins S07. The diagnostic is explicit: `make test-soil-transfer RELEASE=1 SOIL_TRANSFER_FILTER=fifty_thousand_transfer_benchmark SOIL_TRANSFER_ARGS=`. [Transfer evidence and performance disposition](benchmarks/soil/2026-10-09-soil-device-transfer/SUMMARY.md).

Physical sleep/deactivation, awake classification, active/dormant boundary correctness, production transfer performance, incremental durable storage, actual bounded field traversal, machine integration and full acceptance remain open. The owner still conservatively pins every active particle; supporting low-level compaction is not permission to discard moving soil.

**GPU occupancy/halo increment (2026-10-09):** `prepare_gpu_pins(engine, MpmSoilPinConfig, max_bytes)` optionally replaces host position-to-pin classification with a retained GPU hash query and cubic tile dilation. All active particles contribute, including stationary material. The owner joins this asynchronous result with the physical checkpoint at the same frozen request revision. A ready physical checkpoint is retained while pins are pending; stepping remains forbidden until both complete. No partial pin set commits on domain, invalid-position or capacity errors. Such data errors preserve the old archive/device state and permit reconfiguration/retry; map/device faults stop the owner. Disabling restores the existing host bookkeeping path, not a CPU physical solver.

This exact mode requires zero layout origin, a binary power-of-two edge in `2^-126..2^127`, tile coordinates within `[-2^30,2^30]`, a declared half-open coordinate box and halo radius 0–4. Other layouts reject before replacing the current query. Integer decoding of f32 position bits matches the archive's f64 floor membership without floating-point rounding or subnormal flushing. A single packed atomic key identifies each hash entry; output order is sorted on the host. Unique-output, dispatch, index and logical memory limits are checked. Five retained buffers consume `4*next_power_of_two(2*C) + 8*C + 80` bytes, with a 48-byte upload and `16+4*C` readback per capture. No hot application GPU buffers are allocated. The mandatory full physical checkpoint still downloads separately; no boundary speedup is claimed.

Four GPU fixtures pass: exact signed-zero/subnormal/boundary and randomized membership; 50k crowded particles generating 27 halo tiles with 1,104 retained bytes and 272-byte readback at capacity 64; source/drop/map/device-loss lifecycle; and a live moved-particle boundary with delayed pin completion, overflow rollback and retry. An initial unconditional staging-buffer unmap caused a destructor abort; explicit idle/pending/completed/faulted drop checks pass after relying on owned wgpu buffer cleanup. Thirteen CPU checks, five existing owner and four transfer GPU regressions, and the facade/viewer build pass. `make test-soil-pins RELEASE=1` joins S07. [GPU pin evidence and limitations](benchmarks/soil/2026-10-09-soil-gpu-pins/SUMMARY.md).

**Not physical halo acceptance:** dilation covers already-active particles, not future motion or newly requested machine tiles. It does not derive a support radius or classify sleep. Repeated nonzero dilation can activate neighbors that expand the next set again, eventually reaching capacity. Caller-supplied machine/support halos, physically validated deactivation, active/dormant coupling and bounded long traversal still require implementation. The optional query is not enabled in the default viewer; full-plan acceptance remains incomplete.

**Incremental durable store (2026-10-09):** `MpmSoilStore` stores content-addressed tile blobs and a checked versioned `HEAD` manifest under an exclusive nonblocking file lock. Each changed tile retains all physical/history streams and global IDs using the existing tile container; untouched tile epochs reuse prior content hashes without serializing their particle payloads. Metadata/manifest work still scales with total tile count. Loading checks bounded lengths, hashes, metadata, exact coordinate/count consistency and cross-tile global-ID uniqueness before returning any archive. It currently materializes the entire archive, not an out-of-core active selection.

Publication writes and syncs new blobs, syncs their directory entries, writes/syncs a temporary manifest, atomically replaces `HEAD`, and syncs the directory again. This requires a private existing directory on a filesystem supporting the declared locking/rename/sync operations; caller-owned directory creation must itself be durable. Pre-publication failure leaves the prior `HEAD`; an error after replacement may expose the new generation and poisons the writer until reopen. No partial generation is returned. Leftover temporary files are discarded on reopen; unreachable blobs are ignored. Explicit `collect_unreferenced` first validates the complete committed archive, then removes only unreferenced regular hash-named tile blobs and syncs the directory. It discards old generations; there is no automatic cleanup or historical snapshot API.

`MpmSoilWorld::save_synchronized_archive` performs this blocking I/O only on an explicit caller request at a synchronized boundary. Pending captures, faulted worlds and unsaved GPU advancement reject before storage writes. Storage failures do not roll back/fault independent GPU physics. Unchanged saves publish no new generation and write zero bytes; generations count changed storage snapshots, not physical time. No disk work is added to solver substeps or default viewer behavior.

Five new CPU fixtures and thirteen persistence regressions pass on the recorded ext4 workspace filesystem, including four subprocess exits without destructors around publication. These prove process-interruption behavior, not simulated machine power loss or every filesystem's durability. A 129-particle/eight-tile fixture writes 24,912 logical bytes initially, 4,668 for one changed tile, and zero when unchanged. One new actual GPU save/drop/reopen/restore/resume check, five owner and two tile GPU regressions pass; facade/viewer check passes. `make test-soil-store RELEASE=1` and `make test-soil-store-gpu RELEASE=1` provide focused lanes; the checkpoint CPU and S07 GPU lanes include them. [Durable storage evidence](benchmarks/soil/2026-10-09-soil-durable-store/SUMMARY.md).

**Bounded selected reads (2026-10-09):** `MpmSoilStore::read_tiles` now loads only requested payloads after checking unique-tile, particle and conservative encoded-byte limits. Manifest-only enumeration and same-handle/generation validation support caller scheduling. The returned selection cannot be used as a full replacement archive; integrity and ID uniqueness cover selected blobs only. Nine CPU store checks pass, including four new selected-read cases and existing durability checks. These reads are explicitly blocking, exclude the already-open manifest and allocator overhead from their byte bound, and do not change live-world ownership. [Selected-read evidence](benchmarks/soil/2026-10-09-store-selected-read/SUMMARY.md).

**Background selected reads (2026-10-09):** `begin_read` consumes the exclusive store into one worker after manifest-only bounded request preparation. `MpmSoilStoreReadTask::poll` never waits for disk I/O and returns the store plus the read result once. Startup/read failures return the owner; an unwinding read panic poisons it. Dropping a task does not join or interrupt disk I/O: the worker keeps the lock until it finishes, then drops an undeliverable result. Fourteen store tests pass, including five new lifecycle cases. This is a bounded request/result lifecycle, not a hard latency guarantee, asynchronous writeback or live-world activation. [Background-read evidence](benchmarks/soil/2026-10-09-store-background-read/SUMMARY.md).

**Selected durable writeback (2026-10-09):** `MpmSoilStore::writeback` now commits an updated current read without replacing unloaded manifest entries. It preserves global IDs, count, mass, reference volume and material identity; moved particles merge with checked existing destination blobs. Limits cover the affected source/destination tile union and its particles/logical encoded bytes, not the whole manifest or allocator overhead. Unchanged/empty updates keep the generation; changed updates invalidate prior reads. The shared full/partial publication protocol retains poisoned-writer/reopen behavior on publication errors. Seven new cases and existing checkpoint checks pass: **34 CPU tests**, including twelve bounded selected moves across reopen and four injected publication-stage failures. This remains blocking storage work, not live traversal or asynchronous writeback. [Selected-writeback evidence](benchmarks/soil/2026-10-09-store-selected-writeback/SUMMARY.md).

**Background selected writeback (2026-10-09):** `begin_writeback` now moves the store, read snapshot and updated checkpoint into one bounded worker request. Polling returns all three inputs plus the result once, including on startup/validation/I/O failure. Publication errors and unwinding panics return a poisoned store requiring reopen; the generation may already have changed. Dropping the task does not cancel publication or release its worker lock early. Twenty-seven store tests pass, including six new write-lifecycle cases and existing durability checks. This removes caller-thread payload I/O from this API, not thread-creation/validation overhead or the live world's full archive. [Background-write evidence](benchmarks/soil/2026-10-09-store-background-write/SUMMARY.md).

**Disk-backed live owner (2026-10-09):** `MpmSoilDiskWorld` now owns an actual GPU solver and exclusive store without retaining a full host particle archive. Construction reads only committed metadata and an empty selection. Region updates freeze stepping through asynchronous GPU capture, background selected writeback and background next-selection loading; only then do they publish device membership and region revision. Manifest counts plus captured positions reject oversized future selections before storage publication. Every active particle stays pinned; overlaps share IDs and empty requested tiles remain empty.

The existing memory-backed owner's complete-archive API is unchanged. Both owner types remain Send + Sync. Disk and GPU publication are separate: pre-write capacity errors leave a retryable old device state; publication errors or post-write load failures fault the disk owner instead of claiming rollback. Recovered captures and global IDs remain inspectable. Dropping does not implicitly save newer physics or cancel a submitted write.

Five real-GPU tests pass in **3.93 s** on RTX 4080, including exact nine-stream/ID capture-save-load-reopen, shared region claims, pre-write capacity retry, retained failure capture, post-publication corrupt-load handling and an actually moving particle crossing a tile boundary. The CPU checkpoint suite passes **41 tests**; the facade release build passes. `make test-soil-disk-world RELEASE=1` joins S07. [Disk-owner evidence](benchmarks/soil/2026-10-09-disk-soil-world/SUMMARY.md).

**Disk-backed GPU transfer increment (2026-10-09):** the disk owner now optionally prepares the existing bounded device-gather/import helper. Retained histories remain on the GPU; unchanged membership skips uploads/dispatches; disabling returns to the unchanged full-upload default. Preparation/reconfiguration rejects during transitions or faults. The verified fixture retains 21 particles and imports 12, uploading **2164 bytes versus 5568** for full upload, with **3376 bytes** of mandatory active capture readback and **65548 bytes** of retained transfer scratch. One gather dispatch and zero observed hot buffer allocations match instrumentation; a missing shared-helper dispatch counter was fixed. Six disk-owner and four existing transfer GPU tests pass, alongside 41 CPU checks and the facade build. This is logical transfer evidence, not an accepted timing speedup. New statistics distinguish successful durable-write revision from device-committed read generation. [Evidence](benchmarks/soil/2026-10-09-disk-device-transfer/SUMMARY.md).

Physical sleep/wake, support-boundary correctness, actual bounded long traversal, reduced active-state readback, world rebasing, machine/renderer integration and absolute timing remain open. The disk owner retains bounded selected payloads, but the manifest still scales with stored tiles, active particles remain pinned, and no field-stream memory/timing benchmark has established the complete S07 gate. At this increment, optional GPU pin dilation was available only on the memory-backed owner; the 2026-10-12 disk-pin increment above extends it to both owners without closing these gates.

**Swept activation envelopes (2026-10-09):** both live owners now accept `MpmSoilRegionEnvelope` through `begin_swept_region_update`. Closed machine/tool swept bounds expand by the configured quadratic-transfer reach, optional Jacobi-neighbor reach and a caller-supplied per-substep soil-travel bound over a specified refresh horizon. Enumeration rejects oversized or unrepresentable requests before capture; overlaps still share physical tiles. Successful publication installs the shortest envelope horizon, and stepping rejects an oversized batch before submitting GPU work. A failed refresh does not extend the previous horizon. Raw tile requests retain their manual-coverage contract. Six new CPU cases cover geometry, local stencil reach, bounds, sharing and horizon publication; an actual GPU case covers both owners, neighbor activation across a tile seam, full-selection step agreement, exhaustion and refresh. **This is request coverage, not physical sleep/support acceptance**: displacement bounds remain caller obligations, nonlocal coupling is not covered by the local-stencil bound, and every active particle remains pinned. [Evidence](benchmarks/soil/2026-10-09-region-envelope/SUMMARY.md).

**Activation roundoff correction (2026-10-09):** ideal f64 `3*h` padding alone can miss a neighboring archive tile used by the actual f32 shader stencil. A fixed nonzero-weight counterexample is reproduced on RTX 4080. Swept bounds now add a conservative f32 arithmetic allowance and reject unsupported spacing, reciprocal, horizon and coordinate precision ranges before enumeration. The production-source shader probe checks 241 pairs, with 104 connected pairs covered in both directions; 49 CPU checkpoint tests and seven disk-owner GPU regressions pass. `make test-soil-regions` is included in the S07 GPU lane. This corrects numerical request coverage, not sleep, physical support or large-world rebasing. [Counterexample, supported range and evidence](benchmarks/soil/2026-10-09-region-roundoff/SUMMARY.md).

### S08 — Couple a real tractor and one working implement

**Scope:** explicit coupling additions in `molla-sim/src/runtime.rs`, `molla-solvers/src/rigid_world.rs`, soil/vehicle APIs and one focused viewer fixture. Do not replace existing unrelated rigid-contact behavior.

1. Audit the actual public CPU/GPU machine route before wiring it. Existing MPM sphere-imprint APIs and CPU `RigidScene::add_body_wrench` are useful references, not proof of arbitrary rotating mesh/device coupling. Reject unsupported combinations explicitly.
2. Define contact ownership: soil contact supplies its designated normal/shear resistance, with no duplicate rigid-ground, brush, or ad-hoc rolling-loss contribution.
3. Use the true tire footprint, including its width coordinate, and an actual simple tine/blade geometry for the first implement. Preserve angular motion and moment arms; do not certify arbitrary tool meshes using only a sphere or kinematic roller.
4. Exchange equal-and-opposite impulses/wrenches with explicit world/local frames, application points, units and substep time. Include externally prescribed motion as an external momentum/work source in conservation checks.
5. Use stable body handles across machine edits and deletion. Removing a wheel/tool invalidates its contacts and wakes or releases affected soil correctly.
6. Implement a visible driven tractor traversing virgin and previously disturbed soil, then towing a single tine. Log wheel slip, drawbar force, rut profile, tool draft, soil mass movement, and per-subsystem timing.

**Gate:** `make test-soil-gpu PHASE=S08`; reaction-balance, force/torque frame, width/rotation, machine removal, pressure/load sweep, repeated passes, and rotating-tool checks pass. `make bench-soil CASE=tractor-tine MODE=acceptance` includes contact discovery, actual reaction consumption, soil stepping and terrain output. A renderer capture demonstrates ruts and material displacement at the logged simulated times. Passing an isolated transfer kernel is insufficient.

**Prescribed solid geometry increment (2026-10-09):** `MpmPrescribedSolid` and `MpmSolidShape` add explicit, separate `set_prescribed_solids` coupling for centered spheres, oriented boxes and finite cylinders (local-Y axle). World-space linear/angular velocities evaluate `v + omega × (node - center)`. The cylinder's axial half-width and orientation participate in node membership rather than approximating a tire by an infinite cylinder or bounding sphere. Rotationally invariant bounding spheres feed the existing active-tile marking; actual contact uses the supplied geometry.

This route is a **sticking occupied-volume grid boundary**, not a calibrated tire/soil friction law or an arbitrary mesh collider. First input order owns overlaps. It preserves physical grid mass, applies momentum at occupied interior nodes, and accumulates world force and torque about the supplied center of mass. Reaction includes the gravity impulse suppressed by the constrained velocity. It does not use the legacy artificial heavy-cell mass or the legacy outside-shell wrench contribution. `set_rigid_spheres` remains the distinct legacy route with its existing behavior; it is not silently redefined.

Only explicit stepping is supported. Unsupported implicit configuration, nonfinite/invalid poses, nonunit quaternions, invalid dimensions/capacity and invalid prescribed timesteps reject rather than fall back. The retained motion buffer grows from 16 to 64 bytes per rigid capacity; each replacement uploads 80 bytes per solid and creates no application GPU buffer. Parameter uploads occur between submitted batches; reactions are valid for the last completed substep before replacing poses/order. Empty replacement removes these contacts. There are not yet stable body handles, generation-tagged reactions, device machine integration, moving-pose records within one encoder, arbitrary local shape offsets or a rendered tractor/tine fixture. Tool dimensions and soil spacing must resolve occupied grid nodes; subcell swept contact is not implemented.

`make test-soil-solids RELEASE=1` is the focused geometry/conservation lane, not the complete S08 gate. [Evidence](benchmarks/soil/2026-10-09-mpm-prescribed-solids/SUMMARY.md) separates grid impulse/mass/torque checks and short dense/active/compact transport from full machine and physical accuracy acceptance. No performance gain is claimed for adding this missing geometry capability.

**Resident rigid-state/reaction adapter (2026-10-09):** `PreparedMpmRigidDeviceBridge` maps stable `RigidScene` body handles to the two existing resident `State` buffers. A device kernel computes each bound solid's world COM from body-origin pose plus rotated local COM, copies current world linear/angular velocity, and writes the existing MPM prescribed-solid rows. After one soil substep, a second kernel adds its world COM wrench directly to the selected rigid input's `body_f`. There is no pose or wrench host download in these operations. The caller then executes one GPU machine step at the same dt and restores both external-force buffers; pending bridge drop also queues restoration on a healthy device.

The adapter retains four buffers (48 bytes per binding, a 16-byte uniform, and backups of both complete external-force buffers), two pipelines and two bind groups. One live bridge leases each soil's rigid bindings. Duplicate/missing handles, foreign buffer ownership, source replacement, model edits/removal, host rigid overrides and invalid step sequencing reject rather than use stale mapping. Restoration can still clean up the original state after scene/soil mapping invalidation. It does not manage general rigid collisions, machine sleep/wake, full-frame failure rollback or `RigidWorld` scheduling.

**Supported scope is dense explicit soil only.** Active/compact mode is rejected at preparation and dispatch until soil overflow can halt machine advancement through a shared device fault contract; applying a stale previous wrench after tile overflow would be incorrect. GPU completion/error checks remain caller-owned. The focused lane is `make test-soil-rigid-device RELEASE=1`; [evidence](benchmarks/soil/2026-10-09-mpm-rigid-device/SUMMARY.md) distinguishes fast mapping/lifecycle checks from actual Featherstone-consumption verification and its expensive existing GPU-program initialization. This adapter does not close the complete S08 gate or demonstrate tractor/tine performance.

The actual GPU Featherstone fixture passes: reversed handle/slot order, rotated nonzero local COM, a nonzero soil reaction consumed as the expected free-body linear velocity change, zero solver fault status, external-force restoration and alternating-source/drop cleanup. During the adapter operation, counters observe zero application buffer creations, zero readbacks and 352 bytes of ordinary soil parameter uploads; those counters do not cover the complete machine step or renderer. Total fixture time was 105.15 seconds, dominated by existing GPU-program initialization. Earlier 30/90-second setup timeouts are retained; this is correctness evidence, not an accepted step latency. The fast lifecycle fixture covers stale mapping and foreign device rejection separately.

**GPU rigid-world integration (2026-10-09):** `RigidWorld::attach_mpm_soil` takes ownership of dense explicit soil and primitive body bindings. Each actual rigid substep updates solid geometry from the current GPU state, advances soil with the same timestep, applies its reaction, steps the existing GPU Featherstone solver and restores the original external-force buffers. No CPU solver fallback or per-soil-substep pose/wrench readback is introduced. The existing rigid runtime still uploads state and reads completion/state at frame boundaries; this is not a claim that the entire application is GPU-resident.

Bound bodies wake their sleeping groups before runtime selection. Model/source changes reprepare the bridge; disabled bindings temporarily leave the active table, reenabled bodies rejoin, and removed handles are pruned. Soil-bound bodies cannot also own rigid colliders, preventing duplicate rigid contact for this initial route. Loop and breakable joints explicitly reject while soil is attached; ordinary tree dynamics retain the existing GPU route. This restriction does not replace the complete tractor/implement requirement.

One retained, explicitly budgeted device buffer captures all nine physical/history streams, six grid streams and three solid/reaction streams before a frame. A propagated runtime frame failure on a healthy device queues force cleanup and soil restoration alongside the rigid runtime's existing rollback. Pending material IDs and reset flags are restored too. Successful frames commit; device-loss/failed rollback latches the attachment faulted rather than returning it as valid soil. A quarantined device does not receive rollback submissions. Full-capacity device copies add bandwidth and memory cost that has not yet been benchmarked. Active/compact grids, particle ordering and implicit soil remain unsupported by this transaction path.

**Bounded soil-frame completion:** attached soil frames use a five-second GPU completion/readback deadline, configurable with `set_mpm_soil_completion_timeout(Duration)`. This is a fault-containment deadline, **not** the 4 ms performance target; CPU preparation/program compilation and standalone observation APIs are outside this budget. Deadline/poll failure marks the shared logical device unavailable without resetting the physical GPU, restores the CPU scene checkpoint, clears soil publication and rejects further stepping/readout/detach as valid soil. No physical GPU rollback or completed frame is claimed after a context fault. `make test-soil-completion RELEASE=1` exercises deadline/future/channel behavior; the actual runtime fixture separately checks healthy rollback and expiry quarantine. [Input/fault evidence](benchmarks/soil/2026-10-09-p2g-inputs/SUMMARY.md) records the observed NVIDIA Xid 109 events and exact verification status.

The focused lanes are `make test-soil-frame-backup RELEASE=1` and `make test-soil-runtime RELEASE=1`. The initial actual-world fixture passed coupled advancement, a failure injected after the second soil substep, bit-exact particle/history rollback plus unchanged rigid pose/velocity, successful retry and detach. This is a small free-body correctness fixture, not the articulated tractor, consumed angular-response, workload timing or rendering gate. [Runtime evidence](benchmarks/soil/2026-10-09-rigid-soil-runtime/SUMMARY.md) records exact verification scope. S08 and full-plan acceptance remain open.

**Articulated workload implementation (2026-10-09):** `molla::solvers::soil_tractor::SoilTractor` assembles six actual dynamic bodies: a 700 kg chassis, four 35 kg finite-width cylinder wheels with revolute axles, and a 20 kg box tine on a fixed hitch. Local cylinder Y maps to world Z; positive drive torque commands negative axle rotation for chassis +X travel. It uses the existing GPU `RigidWorld` attachment and joint motor API, not kinematically moved wheels, the brush model or CPU vehicle dynamics. The soil is still the explicitly uncalibrated sticking-volume model described above.

The default lattice is exactly 50,000 particles over a 5.6 × 0.25 × 3.2 m bed, with per-particle reference volume/mass derived from spacing and configured density. Dense 64³/0.1 m grid, 3,000 Pa diagnostic stiffness, 1,600 kg/m³ density, ≤0.0005 s requested substeps and a 128 MiB coupling-storage budget are initial workload inputs, **not frozen/physically calibrated or performance-accepted presets**. The owner reports committed simulated time, chassis pose, wheel circumferential/forward speeds and nominal slip relative to stationary ground. Actual local soil-relative slip, steering/suspension, terrain output and rendering remain integration work; optional frame-integrated loads are described below.

`read_mpm_soil_wrenches`/`read_soil_loads` explicitly read the last substep's world force and COM torque, tagged with stable body handles, committed frame and substep dt. They reject absent/faulted/stale model associations. This is not a frame average or a new hot-path readback. Soil forces remain applied until all substep consumers, including joint-reaction reconstruction, have run; failure cleanup still uses frame rollback.

`make test-soil-tractor RELEASE=1` checks geometry, mass and motor direction with a CPU oracle. `make test-soil-tractor-gpu RELEASE=1` is a bounded 7,168-particle physical fixture checking rigid linear/angular impulse balance, driven motion and tine loads; [evidence](benchmarks/soil/2026-10-09-soil-tractor/SUMMARY.md) records its actual status, including failures. Initial GPU execution timed out after construction; a reproduced completion-wake hazard in the runtime wait helper was hardened separately. These commands are not the full S08 or 50k/4 ms acceptance gate. The later opt-in live viewer now drives this owner; its short rendered run is not full S08 acceptance.

**Stall localization (2026-10-09):** [per-submission diagnostics and exact captures](benchmarks/soil/2026-10-09-tractor-stall/SUMMARY.md) locate the first-frame stall in P2G after successful GPU kinematic refresh, grid reset and wrench reset, before force application or machine dynamics. The failing integrated inputs and passing soil-only control are byte-identical, including all particle/history streams. Standalone replay and a control with frame backup complete; neither proves articulated acceptance. A bounded local-CAS candidate did not fix the asynchronous stall and was reverted. Test-only capture, replay, split-kernel and global-scatter controls retain the original production defaults and physical tolerances. The original articulated gate remains open.

**Native backend result (2026-10-09):** `MpmSimulation::new_native_explicit` and `SoilTractor::new_native_explicit` provide explicit opt-in dense stepping on an Engine created with `wgpu::Features::SHADER_FLOAT32_ATOMIC`. Native mass, momentum and rigid-wrench additions replace global CAS retries; global P2G is selected initially. Unused explicit Jacobi stiffness is not accumulated, and its shared integer metadata is not reinterpreted. Implicit stepping and active-grid dispatch are rejected. Existing constructors and portable defaults are unchanged; the portable integrated CAS failure remains unresolved.

The native variant passes the **same** six-body, 7,168-particle articulated fixture, including all original 2e-5 impulse gates (linear error 2.076371615e-7; angular 5.706007700e-9), seven frames, driven motion, tine loads and stale-readout checks. Focused 16-step particle/history/wrench parity also passes. Run `make test-soil-native-mpm RELEASE=1` and `make test-soil-tractor-gpu RELEASE=1 SOIL_TRACTOR_FILTER=native_articulated_tractor_consumes_force_torque_and_drives`; use the repository GPU wrapper. [Native evidence](benchmarks/soil/2026-10-09-native-mpm/SUMMARY.md) separates those results from the still-open 50k/4 ms, renderer and agricultural-accuracy gates. The 153.07 s test duration includes GPU articulation construction, not simulated-frame timing.

**Full-size diagnostic (2026-10-09):** `make bench-soil-tractor` now runs the actual native six-body machine and tine with the default 50,000-particle bed, 34 unchanged substeps per 1/60 s frame, three warmup frames and 30 measured frames. It records manifests, per-frame CSV, final JSON, source/executable hashes, hardware telemetry and a bounded process result in a fresh `SOIL_RUN_DIR`. This is a diagnostic lane, not release acceptance. [The initial run](benchmarks/soil/2026-10-09-tractor-50k-development/SUMMARY.md) completes 0.55 simulated seconds but measures combined machine/soil median **148.815 ms**, p95 **149.426 ms**, and p99 **149.458 ms** per frame; all measured frames miss 60 Hz. Soil-only work is not isolated, and renderer/terrain export are excluded. Strong late upward chassis motion also remains physically unresolved. The initial dispatch count omitted MPM kernels; dispatch instrumentation has since been fixed and checked separately, without rewriting the original measurements. Neither performance nor long-horizon physical acceptance is achieved.

**Coupled-frame attribution (2026-10-09):** `SOIL_TRACTOR_BENCH_PROFILE=1` adds actual per-kernel MPM timestamps across all 34 machine substeps, with one explicit observation per completed frame and no inner-loop readback. The retained `prepare_mpm_soil_frame_profiling`/`read_mpm_soil_frame_profile` APIs tag captures with committed frame IDs; unread or over-capacity observations are skipped without skipping physics, and failed-frame captures are discarded. The profiled 50k run measures MPM kernel median **5.456896 ms** (p95 **5.929984 ms**) against combined wall median **151.235365 ms**. P2G accounts for median **3.031040 ms**. Most total latency is outside measured MPM kernels; machine/coupling/queue attribution remains necessary. Kernel sums are not the complete soil-side budget.

**GPU-span/host follow-up:** relative GPU offsets and optional host section timing now distinguish submission cost from GPU-side gaps, without extra GPU queries or inner readbacks. The post-no-op-removal 50k diagnostic measures median **156.442592 ms wall**, **147.634176 ms GPU capture span**, and **5.310464 ms MPM kernels**. The 990 inter-substep gaps have median **4.307968 ms**, whereas total host dynamics submission is only **1.905249 ms/frame** median. Target exact Featherstone/momentum/bridge GPU attribution next, not host-loop tuning based on the wall total. This is not a paired speedup result or physical acceptance: final machine kinetic energy reaches **1.093 MJ** and chassis Y **3.877302 m**, reinforcing unresolved long-horizon variation and missing energy/work balance. [Evidence](benchmarks/soil/2026-10-09-coupled-attribution-development/SUMMARY.md).

**Exact machine attribution:** retained compute-pass queries now identify the dominant cost as Featherstone dynamics: **141.988192 ms/frame** median, versus momentum transport **1.608224 ms**, refresh **1.326656 ms**, and MPM **5.301248 ms**. Total completed wall median remains **155.829037 ms**. Captures contain 69 distinct machine pass records per default frame, preserve unread/rollback identity, and resolve in the existing final readback submission. Optimize machine arithmetic without disabling compensated precision; the first workgroup store/load rounding candidate failed low-residue parity and was not adopted. [Machine evidence](benchmarks/soil/2026-10-09-machine-profile-development/SUMMARY.md).

**Rejected scratch optimization:** an atomic read-modify-write alternative preserves bit-identical state and compensation residues on a translated multi-articulation fixture and reduces that fixture's dynamics time, but **slows the actual tractor**: dynamics median **152.176064 ms**, wall **164.254523 ms**. The legacy default is restored. Do not extrapolate small-articulation timings or discard precision residues to meet the target. The failed initial variant, corrected arithmetic oracle, exact comparison and adverse full-size measurements are retained in [the experiment evidence](benchmarks/soil/2026-10-09-machine-local-rounding/SUMMARY.md).

The profile exposes an inert legacy sphere shell-force pass for prescribed shapes. It has now been omitted for prescribed contacts, while legacy sphere coupling retains it. GPU tests verify the removed pass leaves nonzero prescribed wrenches bit-identical and that seven versus eight passes execute; numerical tolerances are unchanged. No post-removal whole-frame timing improvement is claimed. Machine-only energy observations are also recorded, but do not close work/energy balance. The longer profiled and unprofiled trajectories differ substantially, so repeatability and physical stability remain open. [Profiling, optimization and lifecycle evidence](benchmarks/soil/2026-10-09-coupled-profile-development/SUMMARY.md).

**Frame-integrated load implementation (2026-10-09):** optional `prepare_mpm_soil_frame_loads` / `read_mpm_soil_frame_loads` (also exposed by `SoilTractor`) accumulate all substep reactions on the GPU. They return frame identity, integrated linear impulse, instantaneous-COM torque integral and actual soil interval sum for frame-average loads. The original last-substep API remains distinct. Double buffering preserves committed observations through healthy rollback; model changes reject stale body associations. Storage is bounded by the coupling budget; no inner readback is introduced. `make bench-soil-tractor SOIL_TRACTOR_BENCH_LOADS=1` records the observations outside wall timing. [Implementation and evidence](benchmarks/soil/2026-10-09-frame-loads/SUMMARY.md).

**Work/stability status:** the completed portable-machine 50k run reports -9.295 kJ commanded wheel work and +124.783 kJ machine mechanical-energy change. Drive work does not explain the growth; full soil/contact work accounting and the stability fix remain open. Native-f64 machine addition is test-only after full-state comparison failures despite a passing component oracle; no production speedup is claimed. [Details](benchmarks/soil/2026-10-09-machine-native-sum/SUMMARY.md).

**Finite-mass coupling foundation (2026-10-09):** `SoilContactProjectionGpu` implements a retained joint machine/soil sticking projection from GPU mass matrices and world COM Jacobians, reducing contact nodes to 16 moments/body before the small generalized solve. Synthetic mass-ratio checks and a ten-DOF tractor fixture pass independent f64 rate, generalized impulse and kinetic-energy gates. Empty contact mass preserves rates exactly. This component is **not yet integrated into MPM stepping**: staged free-grid/free-machine prediction, corrected drift, single reaction application and shared rollback/status remain required before claiming a stability fix. [Design and evidence](benchmarks/soil/2026-10-09-contact-projection/SUMMARY.md). The production coupling and all full-workload acceptance gates are unchanged.

**Projected MPM transfer implementation (2026-10-09):** `PreparedMpmMassCoupling` now connects free grid momentum to the finite-mass projection and corrected G2P/plasticity, without the old imprint reaction. Portable/native 65-particle checks, impulse/gravity consistency, exact empty-soil behavior, exclusive overlap ownership and full backed-stream rollback pass. Capacity and fixed-grid-boundary faults suppress particle publication on the GPU. `RigidWorld` free prediction, projected-rate/momentum-delta publication and compensated drift are still needed; this does not yet replace the production tractor bridge. [Evidence and integration steps](benchmarks/soil/2026-10-09-mpm-mass-coupling/SUMMARY.md).

**Projected machine publication (2026-10-09):** retained GPU publication now updates machine rates and the momentum-transport delta together, with full-island validation before any write and sticky shared-status propagation. Kinematics-only bridge encoding updates solids from either resident input state without advancing soil or applying an explicit reaction. Focused publication, portable/native MPM and bridge lifecycle checks pass. Actual tractor free prediction, projection over 365 synthetic nodes, rate/delta publication and compensated drift also pass; full `RigidWorld` scheduling, its momentum-transport stage and long-horizon coupled stability remain open. [Evidence and remaining integration](benchmarks/soil/2026-10-09-soil-rate-publication/SUMMARY.md).

**Opt-in projected runtime (2026-10-09):** `RigidWorld::enable_mpm_soil_mass_coupling` now schedules input-pose mass/Jacobians, free machine prediction, coupled MPM, rate/delta publication, corrected drift and existing momentum transport. The actual six-body tractor/65-particle fixture passes three-substep conservation, frame-load, rollback/resume, GPU overflow, budget and empty-soil checks. This mode supports one unrestricted articulation and rejects unsupported constraints/readouts explicitly; the legacy path remains default. Actual 50k diagnostics and all broad stability/performance/rendering gates remain separate. [Evidence and remaining requirements](benchmarks/soil/2026-10-09-projected-runtime/SUMMARY.md).

**Full-workload boundary/performance result (2026-10-09):** the projected 50k route commits 24 frames / 0.4 simulated seconds, then rejects occupied fixed-grid-boundary contact. Error-only diagnostics confirm local status 33 (`input-or-upstream|fixed-grid-boundary`), with the other state halted by upstream status. The completed 21-sample measured prefix has roughly 600–628 ms median wall time across two diagnostic runs, not a real-time result; dynamics and corrected drift dominate. Boundary response, full-run stability and the original 4 ms/60 FPS gates remain open. Source/hardware evidence and failed runs are preserved in the runtime checkpoint above.

**Latest S08 boundary checkpoint (2026-10-09):** the earlier reaction-precision failure is repaired by preserving small multiplier corrections separately and refining against actual world-wrench impulses. The production shader passes all 76 independent exhaustive-QP cases at unchanged tolerances. MPM now carries occupied-boundary masks into the coupled solve; all six faces pass with portable/native atomics, shared machine/particle motion, independent external impulse accounting, energy checks and no-penetration. Working-row rank checks prevent duplicate constraints from producing singular bases. The actual 50k tractor now completes all 33 frames / 0.55 simulated seconds, passing the former frame-24 boundary failure with source checks passing. Performance still fails: all 30 measured frames miss 60 Hz; median wall is 616.792124 ms and MPM kernels 9.482240 ms. This short diagnostic is not long-horizon, rendering or calibration acceptance. [Current evidence](benchmarks/soil/2026-10-09-boundary-wrench/SUMMARY.md).

**Fixed-input projection audit (2026-10-09):** an actual frame-24 tractor snapshot has one **0.47644216 kg** tine boundary node, while the adjacent resident parity has none. Reordering the captured contact nodes changes rates by at most **1.19e-7**; with the reduced system held fixed, all tested permutations are bit-exact. Removing the one boundary node changes the independent f64 solution by **0.01094245**, matching the scale of the earlier full-frame discrepancy. The GPU agrees with that oracle within **1.99e-7**. This is evidence of contact-membership sensitivity, not a completed activation fix or proof of every discrepancy's cause. The first capture driver stopped on an incorrect per-sample boundary assumption; only one fixture was archived, and the corrected wider collector remains unverified. Saved replay now passes three checks in **3.53 s** without rebuilding the tractor. Production contact selection and default rounding are unchanged. [Replay artifacts, failed run and remaining work](benchmarks/soil/2026-10-09-projection-order-replay/SUMMARY.md).

**Boundary-node cache increment (2026-10-09):** the explicit experimental `SoilContactProjectionGpu::new_with_fixed_boundaries_cached` constructor builds an ordered cache of up to 128 boundary-bearing nodes once per boundary invocation, avoiding repeated full-node scans inside the active-set solve. Overflow uses the complete original GPU scan, not truncation. It uses 2048 bytes of workgroup storage with unchanged owned device buffers, dispatch count, 32-wide matrix scratch, constraint order and tolerances. All 1–32 DOF occupied/empty comparisons are bit-exact; the independent 76-case QP gate and overflow/reuse cases pass. A paired saved-input diagnostic observes median **28.788113 ms original / 19.503018 ms cached** for 34 repeated projections, with exact outputs and zero hot buffer allocations/transfers. The GPU became externally contended; this is not integrated simulated-time throughput or a basis for an automatic default change. The actual tractor runtime still uses the original constructor. A preceding shared-matrix trial remains private after only a small combined component improvement. [Evidence, limitations and next integration](benchmarks/soil/2026-10-09-boundary-workgroup/SUMMARY.md).

**Moving-soil slip observation (2026-10-10, S08):** unilateral coupling now exposes an explicit committed last-substep sample of impulse-weighted corrected soil velocity and tangential machine-point-minus-soil velocity, including angular motion and moment arms. Empty support is `None`, not zero slip. This reports world m/s rather than a normalized wheel ratio or a calibrated friction law. The existing nominal stationary-ground ratio is unchanged. `RigidWorld::read_mpm_soil_contact_motion` and the tractor wrapper use stable body identities, generation checks and the existing completion deadline. Both prepared parities add 992 retained bytes for five bodies; observing downloads 240 bytes and adds no stepping dispatch/readback. Optional `SOIL_TRACTOR_BENCH_CONTACT_MOTION=1` writes benchmark `contact-motion.jsonl` with separate observation cost outside physics timing. The synthetic rotation/translation/velocity-shift/fault checks and actual six-body/65-particle tine-contact runtime pass, including parity, compact traffic, rollback/resume and empty soil. Loaded-wheel behavior, 50k logging and direct stale-edit runtime evidence remain open. [Implementation and exact scope](benchmarks/soil/2026-10-10-contact-motion/SUMMARY.md).

**Loaded-wheel logging checkpoint (2026-10-10, S08):** added impulse-weighted mean local slip speed beside the signed vector mean; equal opposite slips retain nonzero speed. Two GPU tests pass with the same 240-byte tractor readback. A native 50k diagnostic completes 33 frames / 0.55 simulated seconds; independent validation covers all 165 contact-motion records and observes all four wheels loaded, with correct nulls when support is absent. Five corrupted-log cases reject. Observation median/p95 is 0.196002/0.250468 ms outside physics timing, with 7,200 measured readback bytes. Physics remains far from target: 223.068961 ms median, 240.600713 ms p95, all 30 measured 60 Hz deadlines missed. No speedup or physical calibration is claimed; stale-edit runtime evidence, friction, long-horizon/live-rendering and full acceptance remain open. [Live evidence and cancellation check](benchmarks/soil/2026-10-10-contact-motion-live/SUMMARY.md).

**Local tangential friction primitive (2026-10-10, S08):** `ExperimentalSoilFrictionDiskGpu` now solves independent anisotropic maximum-dissipation disks for an explicitly supplied fixed normal-impulse-derived cap. Three focused GPU tests pass: 394 f64/KKT/energy/scaling/rotation cases, finite-mass linear/angular reaction balance and failure/reuse/resource checks. Encoding performs one batched dispatch with no hot buffer allocation or logical host transfers. This is not a coupled contact solve: normal/tangential interaction, floor constraints, shared machine response and guarded runtime publication still need integration. The tractor runtime remains frictionless; no performance, physical calibration or S08 acceptance is claimed. [Implementation and numerical limits](benchmarks/soil/2026-10-10-friction-disk/SUMMARY.md).

**Shared-machine tangential stage (2026-10-10, S08):** `ExperimentalSoilTangentialProjectionGpu` now eliminates distinct soil-node tangent velocities into finite-mass Huber potentials and solves the shared generalized machine response with bounded damped Newton. This avoids a contact-count-squared matrix and does not independently apply local disk impulses. Two GPU suites pass, covering ten numerical fixtures, a distinct f64 dual-coordinate oracle, shared-contact analytic response, moment-arm reaction/energy balance, soil masses 0.001–1000, 32 machine DOFs, 65 contact slots, empty work, sticky/reset faults and exact retained-budget checks. Small rate corrections are retained with compensated arithmetic; the high-mass failure and repair are preserved. This stage still takes fixed caps: joint normal/cap iteration, floor constraints and guarded runtime publication remain open. No 4 ms, calibrated traction or complete S08 result is claimed. [Coupled-stage evidence](benchmarks/soil/2026-10-10-coupled-friction/SUMMARY.md).

**Joint normal/friction prototype (2026-10-10, S08):** added `ExperimentalSoilFrictionProjectionGpu`, combining normal hinge and tangential finite-mass potentials, then updating Coulomb caps from the resulting normal impulses. Thirteen fixtures solve against a distinct f64 dual oracle, including traction-dependent normal loads, planar moment arms, analytic sticking/sliding, gaps, separation, mass 1000, 65 contact slots and 32 DOFs. Three GPU suites pass, but one suite deliberately verifies rejection of a known cap-cycle case with a valid analytic solution. This is a real convergence limitation, not solver acceptance: repair the cap iteration before general runtime use. Fixed-floor treatment, transaction integration, calibrated traction and 4 ms acceptance remain open; production behavior is unchanged. [Exact counterexample and evidence](benchmarks/soil/2026-10-10-normal-friction/SUMMARY.md).

**Cap-cycle repair (2026-10-10, S08):** the prototype now solves the full generalized-momentum residual with the derivative of the sliding friction cap, using a nonsymmetric pivoted Jacobian and residual-norm merit line search. The known analytic cycle solves in two iterations, rather than exhausting 64; all prior fixtures remain within unchanged numerical/physical tolerances. Four GPU suites cover 15 solved fixtures, including a separate pivot/backtracking case. The heavy-soil fixture drops from 19 to three iterations, but no simulator speedup is claimed. Conservative workgroup storage is now 9,552 bytes; owned buffers, hot allocation/transfer counts and single-dispatch encoding are unchanged. Exact floor treatment and transactional runtime integration remain open, and arbitrary Coulomb configurations still have bounded-failure rather than global-convergence guarantees. [Repair and verification](benchmarks/soil/2026-10-10-friction-residual/SUMMARY.md).

**Geometry-to-friction bridge (2026-10-10, S08):** `ExperimentalSoilFrictionGeometryGpu` now composes existing analytic shape queries, deterministic orthonormal contact frames, normal/tangent spatial Jacobians, the joint residual solve and world-space recovery. Explicit per-body coefficients remain uncalibrated inputs. Twelve solved scenes cover sphere/box/cylinder contacts, a shared seven-DOF assembly with an independent spinning axis, two loaded bodies, 65 slots, mixed/zero friction and rotation/translation covariance; world linear/angular and generalized reaction checks pass. Two GPU suites also verify all six wall-mask rejections, invalid coefficients, sticky/reset faults, inactive reuse and budgets. Encoding has four dispatches with zero hot buffer allocations or logical transfers; a 65-slot/seven-DOF owner retains 16,992 bytes. This is not yet RigidWorld integration: joint friction/floor handling, guarded publication, rollback, full tractor trajectories and real-time acceptance remain open. [Bridge implementation and evidence](benchmarks/soil/2026-10-10-friction-geometry/SUMMARY.md).

**Opt-in runtime friction (2026-10-10, S08):** `RigidWorld::enable_mpm_soil_friction_coupling` and the tractor forwarder now connect the joint solver to certified candidates, machine-rate publication, reaction reduction, corrected drift, contact observation and frame transactions. Explicit coefficients are matched by body handle and immutable while attached; missing, duplicate, foreign or invalid entries reject before preparation. The actual six-body/65-particle tine-contact test passes in **223.21 seconds**, including cold setup, unchanged linear-momentum/energy gates, consumed transverse impulse, both state parities, changed timestep, rollback/resume, capacity/budget rejection and empty soil. Fixed-wall friction remains unsupported and explicitly rolls back both systems; the test verifies this failure path. Steady frames allocate zero GPU buffers, but cold setup allocates. Existing defaults and the frictionless fixed-wall route are unchanged. Loaded-wheel friction, full-machine angular acceptance, joint wall response, 50k performance, rendering and calibration remain open. [Runtime evidence and supported limits](benchmarks/soil/2026-10-10-friction-runtime/SUMMARY.md).

**Residual-only friction line search (2026-10-10, S06/S08):** line-search trials now accumulate only the compensated momentum residual, leaving Jacobian construction to Newton iterations. Residual arithmetic and convergence gates are unchanged. Five focused GPU suites pass in **9.15 seconds**; the independent f64/physical fixtures and exact matched full-assembly comparisons pass, including 13 backtracks in the pivoted fixture and a new 32-DOF/1,024-contact analytic case. Twelve alternating measured pairs after two warmups give **18.703360 → 14.565888 ms** median GPU time (22.1% lower) for that fixed-input component. Encoding remains one dispatch with zero hot GPU-buffer allocations or logical transfers. Two geometry/reaction suites, formatting and the ordinary facade build also pass. This is not simulated-time throughput, the actual tractor workload, rendering, physical calibration or 4 ms acceptance. [Implementation, control and bounded checks](benchmarks/soil/2026-10-10-friction-residual/SUMMARY.md).

**Sparse friction capacity work (2026-10-10, S06/S08):** the joint solver now records the last positive-mass slot during full input validation and bounds repeated contact evaluation to that prefix; residual/Jacobian assembly also skips zero-mass holes. Every slot retains validation and initialized output velocities. Six GPU suites pass in **11.76 seconds**, including exact full-capacity-control matches through 16,384 slots, growing/shrinking/gapped/empty reuse, invalid-tail faults, signed-zero outputs and independent f64 checks. For two contacts in 1,024 slots, paired GPU medians are **3.318784 → 0.070656 ms** at slots 0/63 and **3.319808 → 0.323584 ms** at slots 0/1023. The separately matched fully occupied 32-DOF case regresses **14.839296 → 15.016960 ms (~1.2%)**; this is a sparse-workload improvement, not a universal speedup. Two geometry suites, facade build and formatting pass. Validation still scans all capacity, no simulated time advances in these comparisons, and full-workload/4 ms acceptance remains open. [Signed-zero repair, controls and evidence](benchmarks/soil/2026-10-10-friction-prefix/SUMMARY.md).

**Actual 50k friction trajectory (2026-10-10, S08):** the tractor benchmark now accepts explicit `--friction-mu` / `SOIL_TRACTOR_BENCH_FRICTION_MU`, requires a unilateral candidate configuration, assigns the coefficient to all four wheels plus the tine, and verifies actual runtime configuration/coefficients before stepping. Defaults remain unchanged; metadata distinguishes friction from the frictionless floor solver. Eleven option/fixture CPU tests pass. One native-atomic Generic run with uncalibrated coefficient 0.3 commits **11 frames / 0.183333 s**, then attempted frame 11 fails with soil status 65 including **fixed-wall-friction-unsupported**. The 210.112-second construction and failed run are retained; source verification passes. An explicitly partial audit validates 55 contact-motion body records and observes all four wheels loaded before the fault. The eight completed measured frames have median **208.125285 ms**, all missing 60 Hz; these are not full-workload timings. Complete-run audits correctly reject the truncated result. **Joint fixed-wall/friction response is now a demonstrated trajectory gap and the next coupling priority**; do not remove the rejection or rerun unchanged. [Integration, failure and scoped audits](benchmarks/soil/2026-10-10-friction-tractor/SUMMARY.md).

**Local joint wall/friction block (2026-10-10, S08):** implemented a batched six-impulse GPU solve with normal-only coordinate walls and finite SPD machine point response. Four GPU suites pass in **3.87 seconds** across **4374 cases**, preserving independent 2e-5 physical/numerical gates, exact repeatability and zero hot GPU-buffer allocations/transfers. Maximum Newton iterations: 21. Eight cases choose a different admissible Coulomb branch; independent f64 refinement verifies both branches on the same rounded input rather than assuming a unique manufactured solution. **Shared-machine integration is still required**; independent application would be incorrect. Runtime wall rejection, full-trajectory failure, uncalibrated status and unmet performance target remain unchanged. [Local implementation and nonuniqueness evidence](benchmarks/soil/2026-10-10-wall-friction-block/SUMMARY.md).

**Shared-machine wall/friction baseline (2026-10-10, S08):** implemented generalized-mass factorization, parallel point responses and conditional shared-rate block sweeps using the common local solver. Four analytic two-node configurations pass, but the seven-node/eight-DOF case reaches 256 sweeps without meeting the unchanged global certificate; the larger case remains unrun. A tighter local-tolerance attempt also fails and is reverted. This is a diagnostic coupled baseline, **not accepted general coupling or performance**. Resolve global convergence before geometry/runtime adoption; wall rejection and full-tractor acceptance remain unchanged. [Code, passing cases and explicit failures](benchmarks/soil/2026-10-10-shared-wall-friction/SUMMARY.md).

**Guarded shared-solve acceleration (2026-10-10, S08):** bounded rate-projected mixing with natural-map merit rejection resolves the preceding seven-node failure in **50 sweeps**; a matched disabled control still fails at 256. The 65-node/eight-DOF fixture converges in **51 sweeps**, and four analytic pair cases take three. Five GPU suites pass in **10.03 seconds**, including unchanged independent f64 physical gates, exact repeated output, 65/0/7/65 reuse, sticky/reset and budget checks. Stable energy differences repair a separate empty-contact cancellation failure without loosening gates. **This does not establish latency or runtime acceptance**: conditional sweeps remain serial; analytic geometry, guarded runtime integration and the actual wall-contact tractor trajectory are next. [Acceleration, rejected steps and preserved evidence](benchmarks/soil/2026-10-10-shared-wall-acceleration/SUMMARY.md).

**Shared wall/friction geometry bridge (2026-10-10, S08):** analytic sphere/box/cylinder geometry now packs full world point Jacobians into the shared solver and recovers machine-contact and external-wall impulses separately. Two GPU suites pass in **6.73 seconds**, covering all 27 wall masks, two contacts on a shared seven-DOF machine, frame covariance, independent f64 physical laws, invalid-input publication guards, reset/reuse and budgets. A supported lower-Z fixture has contact/wall Z impulses **-6.9999986 / +6.9999986**; external wall support is excluded from the machine reaction. CPU budget checks and facade build pass. **RigidWorld integration is still next**; runtime defaults/wall rejection, uncalibrated status and unmet 4 ms target are unchanged. [Bridge and evidence](benchmarks/soil/2026-10-10-shared-wall-geometry/SUMMARY.md).

**Opt-in shared wall/friction runtime (2026-10-10, S08):** the new explicit runtime/tractor API now prepares both parities with shared wall geometry while preserving certified candidates, contact-only machine reactions, publication guards and transactional rollback. The six-body/65-particle actual runtime check passes in **215.14 seconds** including cold setup: ordinary momentum/energy gates, transverse machine impulse, zero steady GPU-buffer allocations, 12 retained lower-Z conditional contacts, boundary contact loads, rollback/resume, capacity/budget rejection and empty soil. A failed legacy boundary-fixture reuse was corrected only in the new test; production grid and tolerances are unchanged. Benchmark `--friction-walls 1` / `SOIL_TRACTOR_BENCH_FRICTION_WALLS=1` explicitly selects this route and requires a friction coefficient; all 12 option/fixture CPU tests and the facade build pass. At that checkpoint the full 50k wall-friction trajectory was unrun; the subsequent result is below. The small fixture does not load wheels or establish full-machine angular acceptance. Default/no-wall routes, uncalibrated status and unmet 4 ms/60 FPS target remain unchanged. [Integrated scope and evidence](benchmarks/soil/2026-10-10-wall-friction-runtime/SUMMARY.md).

**50k shared-wall execution failure (2026-10-10, S06/S08):** one changed native-atomic workload with 4096 candidates and coefficient 0.3 reaches attempted frame zero after **206.806 seconds** construction, then GPU completion times out and the logical device is quarantined. Kernel evidence records **Xid 109 / CTX SWITCH TIMEOUT** for that process. **Zero committed frames, no throughput sample**. Frozen-source and explicit requested/runtime configuration checks pass. The shared solver's serial contact/Newton sweeps in one dispatch are a scalability suspect, not a proven root cause. Isolate and bound that work before another full workload; do not raise deadlines, reset the device, or retry unchanged. The small runtime pass does not generalize to 50k execution. A separate hard-coded boundary metadata description is corrected in live source without rewriting failed artifacts. [Failure evidence](benchmarks/soil/2026-10-10-wall-friction-tractor/SUMMARY.md).

**Retained shared-wall sweeps (2026-10-10, S06/S08):** split the all-sweeps dispatch into one sweep per dispatch, retaining compensated rates and acceleration history in 400 additional bytes. Seven GPU suites pass in **28.29 seconds**; rates, contact/wall outputs and status match monolithic small-fixture controls exactly, including fault cases. Independent physical, reuse/reset and resource gates remain unchanged. At 65 contacts/eight DOFs/51 sweeps, six alternating measured pairs give median longest GPU pass **402.706144 -> 8.873664 ms**, but total span **402.768224 -> 402.612512 ms**. **Scheduling improves, throughput does not.** One sweep still has serial contact work; this is not a 50k timeout fix or performance acceptance. Reduce and bound contact-solve cost before another full workload, without dropping walls or coupling. [Scheduling evidence](benchmarks/soil/2026-10-10-wall-friction-sweeps/SUMMARY.md).

**Shared-wall register rounding (2026-10-10, S06/S08):** replacing shared-memory atomic rounding pairs with runtime-key f32 materialization reduces the 65-contact/eight-DOF component median GPU span **402.503168 -> 147.900512 ms (63.3%)** across six alternating measured pairs. Atomic-reference output words match in 4374 local physical cases, 4103 arithmetic cases and shared 3/8/32-DOF cases, including iteration faults. Independent f64 gates are unchanged. The existing experimental shared-wall owner now uses the candidate; default no-wall and standalone local routes are unchanged. Nine final shared suites and two geometry suites pass. No new feature, buffer or hot allocation is required. This is current RTX 4080/Vulkan component evidence, not universal compiler certification, full-trajectory success or the 4 ms soil-frame gate; no 50k retry was made. [Optimization and controls](benchmarks/soil/2026-10-10-wall-friction-rounding/SUMMARY.md).

**Publication-aware local convergence and rejected optimizations (2026-10-10, S06/S08):** a captured conditional input exposes premature Newton termination in both cold and warm starts: small residual/correction passes while the Coulomb-direction publication gate still fails. Continue the existing Newton step rather than return that early failure; the same 64-iteration bound and all physical tolerances remain. The one-contact replay now passes independent f64 gates in eight cold/six warm iterations; its old-rule control still fails. Direct coordinate-wall arithmetic is rejected after a paired **140.310016 -> 143.368320 ms** regression. Warm-starting remains private and rejected because its repaired shared solve still exhausts 64 sweeps. Production remains cold-started/general-axis; no full tractor retry or performance acceptance is claimed. [Repair and limits](benchmarks/soil/2026-10-10-wall-friction-warm/SUMMARY.md), [axis experiment](benchmarks/soil/2026-10-10-wall-friction-axes/SUMMARY.md).

**Parallel shared-wall setup cache (2026-10-10, S06/S08):** contact basis, local response metric and residual scales are now prepared once per solve in parallel rather than rebuilt at every conditional Newton solve. Final alternating paired GPU medians improve **143.040768 -> 130.797760 ms (~8.6%)** at 65 contacts/eight DOFs/51 sweeps. Cached/uncached outputs and status match bit-for-bit at 3/8/32 DOFs; changed inputs, reset/reuse and 65 noncontiguous contacts in 4096 slots pass. Both geometry suites retain all 27 wall masks and independent physical gates. Extra retained storage is **272 bytes per capacity slot**, included in shared/geometry/runtime budgets before allocation; no extra hot transfer, allocation, binding or dispatch. Only the existing experimental shared-wall route changes. Serial sweep cost, the 50k timeout, 4 ms/60 FPS and calibration gates remain unresolved. [Source, paired controls and resource checks](benchmarks/soil/2026-10-10-wall-friction-cache/SUMMARY.md).

**Fixed-size shared-wall linear solve (2026-10-10, S06/S08):** unrolled partial-pivot elimination/back-substitution reduces the latest paired 65-contact/eight-DOF GPU median **132.128352 -> 108.083136 ms (18.2%)**, longest pass **2.916608 -> 2.375296 ms**. Initial implicit-contraction unrolling failed exact comparison; explicit fused multiply-subtracts restore all output words in 4374 local cases, the captured publication-failure replay and shared 3/8/32-DOF cases. The existing experimental shared-wall route now selects it; standalone local/default no-wall routes and all physical gates remain unchanged. Geometry, default-owner reset/fault/budget and 4096-slot sparse checks pass. No new storage or dispatch. This still leaves serial sweeps far above the 4 ms budget; no full tractor rerun or acceptance claim. [Initial failure, arithmetic repair and measurements](benchmarks/soil/2026-10-10-wall-friction-linear/SUMMARY.md).

**Cooperative shared-wall reductions (2026-10-10, S06/S08):** independent rate rebuilds, per-contact acceleration-merit terms and impulse/history updates now use all 64 workgroup lanes. Original contact order, compensated reduction order and physical certification remain unchanged. Final alternating GPU medians improve **107.725440 -> 67.974912 ms (36.9%)**, longest pass **2.363584 -> 1.640288 ms**, for 65 contacts/eight DOFs/51 sweeps. Exact comparisons pass for 3/8/32 DOFs, one-sweep faults, changed inputs, empty scenes and 65 noncontiguous contacts in 4096 slots. Default-owner reuse/budget and both geometry suites pass. Workgroup preflight increases from 256 to a conservative **832 bytes**, propagated through geometry/runtime; owned buffers, bytes and dispatch counts do not change. Only the experimental shared-wall route changes. Contact blocks/final certification remain serial; full 50k timeout, 4 ms/60 FPS, loaded-wheel and calibration acceptance remain open. [Algorithm, controls and telemetry-backed evidence](benchmarks/soil/2026-10-10-wall-friction-cooperative/SUMMARY.md).

**Sparse-capacity bottleneck and private local parallelism (2026-10-10, S06/S08):** DOF-parallel per-contact impulse updates and three-lane point velocities preserve exact 3/8/32-DOF results, unchanged physical gates and lifecycle behavior. The initial unmasked compact variant measures **67.597248 -> 57.584544 ms**; a 64-slot active-mask revision measures **67.656544 -> 63.162400 ms**. Neither is promoted: the representative-capacity comparison with 65 noncontiguous active contacts in 4096 slots measures **410.705472 -> 406.696000 ms (~1%)**, with transient spikes. Both sparse variants match the compact control's physical words. The dominant issue is repeated capacity scanning, not just local updates. Next: stable ordered active-contact indexing, retaining all-slot validation and node/output identities. A reserved WGSL identifier and a hardcoded diagnostic readback length were corrected; failed artifacts are retained. Production/defaults and all gates remain unchanged. [Evidence and next implementation target](benchmarks/soil/2026-10-10-wall-friction-local-parallel/SUMMARY.md).

**Ordered active-contact indexing (2026-10-10, S06/S08):** factorization now builds an ascending positive-mass contact index once per solve. Preparation still validates every allocated slot; repeated sweeps, rebuilds, merit and certification visit indexed contacts without changing output identities or compensated order. Final paired 4096-slot/65-active-contact medians improve **410.962528 -> 72.221248 ms (82.4%)**, longest pass **9.322016 -> 1.721408 ms**. The fully occupied 65-slot case regresses **67.598336 -> 69.337568 ms (2.6%)**, retained as an explicit tradeoff. Exact 3/8/32-DOF, zero/16384-slot boundary, inactive-input rejection, refresh/reset/budget and independent physical checks pass; both geometry suites pass. The index uses existing history padding and count uses continuation state; no new retained buffer, bytes or dispatch. Workgroup preflight is 848 bytes. Enabled only in the experimental shared-wall owner; local parallelism stays private. Full 50k trajectory, 4 ms/60 FPS, loaded wheels and calibration remain open. [Source, sparse gain and compact tradeoff](benchmarks/soil/2026-10-10-wall-friction-ordered/SUMMARY.md).

**Actual ordered-solver runtime recheck (2026-10-10, S06/S08):** one unchanged native 50k workload with the substantially changed solver still fails at frame zero: GPU completion timeout and benchmark-attributed Xid 109; **zero committed frames or throughput samples**. Configuration/frozen-source audits pass, while trajectory audits reject. Added a private candidate limiting local contact work to 1–64 contacts per dispatch, with compensated continuation and 256 extra retained bytes. CPU shader validation passes; GPU equivalence, cost, resource checks and runtime integration are pending. Public selection remains unchanged. This neither localizes the failing kernel conclusively nor establishes a timeout fix; no unchanged full retry or increased deadline. [Terminal failure and candidate scope](benchmarks/soil/2026-10-10-wall-friction-ordered-tractor/SUMMARY.md).

**Chunk evaluation and first-input isolation (2026-10-10, S06/S08):** chunked contacts preserve GPU output/status words and physical/resource gates, but paired spans regress **69.243456 -> 80.679584 ms** (compact) and **71.751552 -> 112.756704 ms** (sparse). Not promoted. Test-only fail-closed capture saved the real 50k tractor's first **10-DOF / 251-active-contact** input. Both existing ordered and chunked isolated solves converge in two sweeps, with byte-identical outputs/status and unchanged independent f64 physical checks. The first solve does not reproduce the full-frame timeout; later substeps/surrounding execution remain to be isolated. Capture's post-save rollback-flag assertion was wrong and corrected; that failed run is preserved and subsequent frame-statistics assertions were not reached. No whole-frame, speed or calibration acceptance follows. [Capture, failed harness assertion and exact replay evidence](benchmarks/soil/2026-10-10-wall-friction-chunks/SUMMARY.md).

**Timeout localized to preparation (2026-10-10, S06/S08):** test-only synchronized tracing identifies **substep 0 / first shared_prepare** as the five-second timeout, before any contact sweep; upstream/geometry and mass factorization complete, followed by attributable Xid 109. This corrects the earlier inference toward later substeps. Captured matrix/rates/rows match the earlier successful isolated input; only 151 contact mass words differ by at most 3.57627868652344e-7. Corrected STORAGE-only trace equivalence passes. Preparation-only replay probes now separate native-feature versus baseline devices, but are not yet GPU-run. No production deadline/profile changes or timeout-resolution claim. [Exact failing stage and next isolation](benchmarks/soil/2026-10-10-wall-friction-stage-trace/SUMMARY.md).

### S09 — Calibrate only the physical behaviors actually required
**Scope:** soil material laws, presets, reference datasets and quantitative reports; keep performance-only improvements separately attributable.

**Data status (2026-10-09):** the user has no measured calibration dataset. Continue implementation and numerical verification without fabricating measured presets or claiming field accuracy. Published datasets can support later calibration only after checking provenance, units, operating range and reuse conditions; synthetic or solver-reference results alone do not close this phase.

1. Pick the first intended soil and operating range; document source/provenance and units. Do not label arbitrary constants as measured sandy loam or clay.
2. Identify pressure–sinkage, shear-displacement/strength, unloading/reloading, and repeat-pass compaction behavior. Add density/layer/moisture dependence only with an explicit model and required inputs.
3. For implements, compare draft/vertical force and displaced volume against physical or justified independent reference data. Fit on one set and validate on a different load/speed/depth set.
4. Record error metrics and predeclared acceptance tolerances for traction versus slip, sinkage, compaction and draft. A lower-cost representation is acceptable only with its stated validity range and measured error.
5. Re-run the performance workload after material stiffness/timestep changes. Graphics sand performance is not evidence for arbitrarily stiff or wet agricultural soil.

**Gate:** `make test-soil-fast PHASE=S09` plus relevant GPU material tests; quantitative held-out comparisons pass. `make bench-soil CASE=calibrated-soil MODE=acceptance` meets the frozen physical-time budget or reports an explicit remaining performance gap. If measurements are unavailable, label results numerically verified but physically uncalibrated; do not invent acceptance data.

### S10 — Integrated acceptance and handoff

1. Run the complete selected workload matrix on the declared hardware with released build settings and no unexplained contention. Include both steady state and activation/reset/streaming spikes.
2. Verify one/four/eight active-region scaling, actual machine reactions, persistent soil, rendering enabled/disabled controls, and the calibrated workload if S09 is in release scope.
3. Test invalid dimensions, non-finite input, capacity overflow, unsupported features, owner reset/drop, and device loss. Invalid calls must not partially mutate authoritative state.
4. Run focused gates first, then the relevant workspace integration gates once at the completed milestone. Attribute unrelated pre-existing failures; never change unrelated code just to make this report green.
5. Record commands, manifest/hash, sample count, result files, hardware, timings, physical residuals, rendered evidence and deviations. Update this ledger and `book/src/vehicle/overview.md`/new soil performance documentation to match actual support.

**Gate:** `make test-soil-fast PHASE=all`, `make test-soil-gpu PHASE=all`, `make bench-soil CASE=release MODE=acceptance`, and the agreed final integration commands in section 8 succeed with nonzero executed cases. All required accuracy, lifecycle and absolute-budget gates must pass. Report performance-only completion separately from calibrated agricultural completion.

## 8. Verification commands and harness contract

### Existing commands, verified from this Makefile

```sh
make check APP_TARGET='-p molla-solvers -p molla-kernels --features gpu'
make check APP_TARGET='-p molla_viewer'
make test APP_TARGET='-p molla-vehicle --lib'
make test APP_TARGET='-p molla-solvers --test mpm_material_oracles'
make test-gpu P=molla-kernels T=tire_mpm_gpu RELEASE=1 A='--nocapture --test-threads=1'
make test-gpu P=molla-kernels T=tire_dem_gpu RELEASE=1 A='--nocapture --test-threads=1'
make test-gpu P=molla-kernels T=tire_bekker_gpu RELEASE=1 A='--nocapture --test-threads=1'
make test-gpu P=molla-solvers T=mpm_gpu RELEASE=1 A='--nocapture --test-threads=1'
make test-gpu P=molla-solvers T=mpm_v3_gpu RELEASE=1 A='--nocapture --test-threads=1'
make test-gpu P=molla-solvers T=mpm_perf RELEASE=1 A='--nocapture --test-threads=1'
make fmt-check
make check-all
```

The GPU target uses `RUN_WITH` (default `nixVulkan`) and executes ignored hardware tests. Match the working repository environment; do not bypass a missing adapter with CPU success. `mpm_perf` is an existing exploratory benchmark/test, not by itself the new acceptance harness. A command that runs zero matching tests is not a pass.

For final integration, use `make test-all` and, when needed by the touched scope/release contract, `make harden`. These are broad and potentially long; do not run them after every small edit. Capture their terminal results, not just startup output. Formatting checks inspect source; no Rust formatting/build is necessary merely to edit this Markdown plan.

### Targets to create in S00 — not present at plan creation

| Target | Required contract |
|---|---|
| `make test-soil-fast PHASE=Sxx` | Dispatch selected CPU/contract fixtures; reject unknown phases and empty selection. |
| `make test-soil-gpu PHASE=Sxx` | Real-adapter focused tests for that phase, serial execution and explicit device identity; no silent skips. |
| `make bench-soil CASE=... MODE=...` | Run the release-profile soil benchmark executable, persist manifest and results, enforce declared deadline and nonempty workload. |
| `make check-soil-results INPUT=...` | Validate result schema, manifest compatibility, finite outputs, physical-time accounting, counters and acceptance thresholds; nonzero exit on failure. |

New targets should delegate through the existing Make conventions and `RUN_WITH`. Implement the result checker in Rust or a small suitable shell tool, not a Python editing script. The benchmark executable should reject unknown case/mode names and distinguish baseline, comparison, quick diagnostic, and full acceptance runs.

The current Rust `soil_benchmark` executable implements the `check` subcommand only; GPU workloads still execute through their focused test binaries. The checker deliberately rejects older two-mode or unrelated result schemas instead of interpreting them as compatible. `SOIL_BENCH_SECONDS` may shorten the quick execution deadline to 1–30 seconds (default 30); compilation and offline result checking remain outside it. Process tests use synthetic commands solely to verify Make error/timeout propagation, not to claim GPU correctness or speed. Source hashes identify recorded inputs and are not signatures or a revalidation against today's worktree.

Per-phase `make bench-soil ... MODE=compare` must launch matched baseline/candidate paths or read a manifest-compatible saved baseline. If the baseline is from another revision, build/run it in a separate explicitly managed location; never reset the user's checkout to obtain it.

Suggested test organization:

- `crates/molla-kernels/tests/soil_raster_gpu.rs`: small CPU oracle and device raster comparisons, following `tire_bekker_gpu.rs`'s explicit adapter and numeric assertions.
- `crates/molla-kernels/tests/soil_residency_gpu.rs`: actual runtime counters and lifecycle, not source-string searches as sole proof.
- `crates/molla-solvers/tests/mpm_active_tiles_gpu.rs`: dense/sparse/fused matched-state comparisons.
- `crates/molla-solvers/tests/soil_benchmark_contract.rs`: manifest validation, duration accounting, finite metrics, timeout and zero-case rejection.
- `crates/molla-sim/tests/runtime_soil.rs`: stable handles, body reactions, overlapping regions and removal.
- Viewer-local tests: exclusive backend mode and diagnostic-independent physical state.

Do not require bitwise equality from unordered floating-point reductions. Use frozen absolute/relative tolerances, exact integer membership/count checks where applicable, and aggregate conservation bounds. Preserve deterministic seeds and canonical ordering where needed for repeatable comparison.

## 9. Scope boundaries and failure policy

**Allowed implementation scope:** files/modules named in the relevant phase, their direct imports/exports, focused tests and Make targets, benchmark artifacts, and soil documentation. Shared scan/sort/sparse-grid changes require regression coverage for their existing consumers.

**Not part of this plan:** unrelated rigid-joint features, sensors, general renderer rewrites, USD assets/importers, Gearbox source changes, tire constitutive-law rewrites, new packaging/CI infrastructure, global dependency upgrades, multi-GPU infrastructure, or changing every particle solver at once. SPH optimizations are deferred unless a measured selected workload uses SPH; do not extrapolate MPM/DEM findings to it.

Stop the affected phase and report a specific next action when:

- A performance improvement requires changing physical timestep/resolution/material or dropping a force term; split it into a separately approved model/quality experiment.
- Moving a readback exposes a CPU physical dependency; migrate or explicitly classify that dependency, rather than returning stale forces or removing it.
- Active tiles/neighborhood reuse miss contacts, break a stencil, lose mass/history, or exceed capacity; keep the reference path and repair correctness before speed claims.
- A desired subgroup/atomic/device feature is unavailable; report the capability limit and alternatives, without silent backend substitution.
- The agreed budget is infeasible at the measured accuracy: present absolute timings and a quality/performance frontier, not a success label based only on relative speedup.
- Two focused repair attempts fail on the same issue; record the smallest reproducer and diagnosis, stop rerunning broad suites, and request a scoped decision if needed.
- The GPU is externally busy or an owned benchmark times out: retain the result as contended/incomplete and continue independent implementation where possible. Never kill unrelated applications or relax the gate.

## 10. Evidence and completion checklist

Store bounded evidence under `benchmarks/soil/<run-id>/`: manifest, raw samples, summary, command/exit log, hardware/load metadata and links to selected rendered captures. Avoid committing enormous generated logs or binary caches. No secrets or environment dumps containing credentials.

- [ ] S00 workload, hardware, numerical tolerances and absolute budgets are explicit and frozen.
- [ ] Production runs exactly the selected backend; comparison work is opt-in.
- [x] Production MPM/DEM terrain export uses particle bins, not H×P global scans; exhaustive GPU references remain explicitly selectable.
- [ ] No hidden full-state downloads or per-substep blocking waits in the resident path.
- [ ] Fixed-capacity steady state does not allocate GPU buffers/pipelines/bind groups.
- [ ] DEM neighbor reuse or BVH refit has conservative correctness bounds.
- [ ] MPM active work scales with occupied/disturbed tiles, with correct halos and preserved history.
- [ ] Any fusion/layout change has matched-work timing and physical-error evidence.
- [ ] Physical time advances at the reported rate, including all required substeps and coupling.
- [ ] Soil persists across region changes; material transport conserves mass within declared accounting.
- [ ] Real tractor/implement reactions and visible material displacement are demonstrated if agricultural integration is claimed.
- [ ] Calibrated claims have independent held-out data; numerical-only evidence is labeled accordingly.
- [ ] The agreed p95/p99 and real-time-factor gates pass for every required workload, including activation spikes.
- [ ] All required processes are terminal; no unfinished test/benchmark is described as passed.
- [ ] This ledger identifies implemented, verified, deferred and unresolved work separately.

### Compensated drift specialization checkpoint — 2026-10-09

S06 now includes a retained unrestricted-drift shader specialization with per-dispatch bound eligibility and generic fallback. Three translated six-body tractors match the generic path bit-for-bit, including physical compensation and corrected-rate preservation. Seven paired drift dispatches total **11.362368 ms generic versus 10.629184 ms specialized** (approximately 6.45% lower). This is a component result only.

The actual 50k diagnostic still fails at attempted frame 24 on the same occupied fixed-grid boundary. Its incomplete 21-frame measured prefix has **252.796176 ms** median wall time, not real-time acceptance. Different contention and faster unchanged dynamics prevent attributing the historical wall-time difference to this specialization. Build, formatting and source checks pass; all owned processes are terminal. Boundary response, integrated performance, rendering and calibration remain open. [Evidence and next steps](benchmarks/soil/2026-10-09-specialized-drift/SUMMARY.md).

### Runtime boundary-cache integration checkpoint — 2026-10-09

`SoilBoundaryScan::ExperimentalCached` is now explicitly selectable through prepared MPM coupling, the rigid world and the tractor fixture. Existing APIs retain `Full`. The selection survives both ping-pong states and preparation rebuilds, remains immutable until detach, and is exposed in benchmark manifests and runtime-result metadata. Use `make bench-soil-tractor SOIL_TRACTOR_BENCH_BOUNDARY_SCAN=cached SOIL_TRACTOR_BENCH_COUPLING_NODES=4096` for the experimental path.

The actual native tractor runtime fixture passes unchanged conservation, energy, zero steady-state buffer-allocation, rollback/resume, budget, overflow and detach/default-restoration checks in **232.82 seconds**, including cold construction. All five benchmark CLI/helper tests and the GPU-enabled facade build pass. The same-source/executable 50k runs complete 33 frames each: wall medians are **258.101843 ms full / 257.996333 ms cached**, only a 0.041% difference, with all measured frames missing 60 Hz. There is no meaningful demonstrated whole-frame gain.

The unchanged 2e-5 trajectory comparison fails: maximum generalized-rate difference is 0.01112545, generalized-coordinate difference 2.275e-5 and final particle-coordinate difference 5.740e-5 m. Generalized rates first exceed the bound at frame 4, earlier than the previously captured frame-24 fixed-boundary transition. Existing reference-repeatability failures and exact component replay do not explain away these results. Keep the cache opt-in; isolate identical inputs at the first divergence rather than repeatedly running the full fixture. All processes are terminal; no acceptance gate is closed by this comparison. [Evidence and next steps](benchmarks/soil/2026-10-09-runtime-boundary-cache/SUMMARY.md).

### Dynamics dispatch specialization trial — 2026-10-09; test-only

An experimental normal-step shader specializes only `uniforms.drift_only` to zero. This does not assume unrestricted joints: stops, compliance, force terms, precision settings and rigorous-solve controls remain in the same source. It was temporarily integrated, then moved behind `cfg(test)` after the follow-up below. Production retains the original generic dynamics and unrestricted-drift programs, without the extra candidate compilation. Benchmark manifests identify the current `machine_dynamics_backend=runtime-mode-generic`.

A paired generic/specialized GPU fixture with three translated six-body tractors is bit-exact across refresh and seven normal steps, including compensated residues and velocity deltas. Summed measured dynamics time is **42.242240 ms generic / 40.297088 ms specialized** under changing GPU utilization; some individual candidate samples are slower. The actual native coupled-tractor fixture also passes conservation, rollback/resume, budget/overflow and zero steady-state buffer-allocation gates in **436.01 seconds**, including construction of the extra retained program. Five benchmark helper tests and the facade release build pass. This short component observation is not a robust or whole-frame gain; cold startup has additional work. Broader constrained-model regression and full-size performance/trajectory acceptance remain open. [Evidence and current verification](benchmarks/soil/2026-10-09-dynamics-specialization/SUMMARY.md).

**Constraint follow-up:** the candidate remains bit-exact through active scalar stops, velocity caps, D6 spring compliance, joint disable and restoration on three translated six-body articulations. The corrected fixture passes in **391.49 seconds**. Its seven heterogeneous dynamics observations total **45.345120 ms generic / 46.138560 ms candidate**, not a speedup. Combined with unproven whole-frame benefit and extra startup cost, this does not justify production promotion. The candidate remains available only in focused tests; no timing retry or relaxed numerical gate is introduced.

The first fixture incorrectly placed compliance on a revolute axle and correctly faulted in the generic shader. That exposed missing host preflight validation: enabled compliance records now require six-DOF D6 ownership, with checked DOF-offset coverage before any GPU submission. The corrected fixture uses D6 root translation for its spring and retains axle stop/cap/disable checks. The original failure and corrected pass are both preserved. [Preflight fix and constraint evidence](benchmarks/soil/2026-10-09-dynamics-constraints/SUMMARY.md).

### Recorded tractor visualization checkpoint — 2026-10-09

`make render-soil-tractor SOIL_REPLAY_RUN=<completed-run> SOIL_REPLAY_OUTPUT=<new-directory>` now renders actual recorded machine poses, six body-local analytic shapes and all 50,000 final material-point positions. The benchmark emits `render-geometry.json` outside measured windows; older runs without this file are rejected rather than rendered with guessed geometry. The standalone Bevy replay validates schema, successful run/source exits, final frame/time alignment, identities, finite poses and exact particle payload size. Captures include input hashes and on-image uncalibrated/offline labels.

A fresh native coupled run completes 33 frames / 0.55 simulated seconds. The overview and rear/tine images were rendered on the RTX 4080 and visually inspected. The first windowed attempt failed on swapchain acquisition; the final renderer uses a headless GPU image and no compositor. [Overview](benchmarks/soil/2026-10-09-tractor-replay/two-views/snapshot.png), [rear/tine view](benchmarks/soil/2026-10-09-tractor-replay/two-views/implement.png), [provenance and commands](benchmarks/soil/2026-10-09-tractor-replay/SUMMARY.md).

This closes only the missing recorded-state inspection path. Markers are not physical grain geometry or a reconstructed terrain surface; analytic fixture bodies are not asset meshes. A single final state does not establish persistent ruts, long-horizon behavior, calibration, live viewer integration or renderer-inclusive performance. The physics run still misses all 30 measured 60 Hz deadlines (wall median **259.071557 ms**). Full S08/S10 acceptance remains open.

### Time-aligned capture checkpoint — 2026-10-09

`SOIL_TRACTOR_BENCH_CAPTURE_EVERY=11` now opts into aligned machine/particle snapshots at committed frames 0, 11, 22 and 33 for the default 33-frame diagnostic. The default interval is zero (disabled); enabled schedules include both endpoints exactly once and reject more than 64 captures before GPU construction. Capture transfers and file-writing time are reported separately, outside timed frame windows. Final positions are reused rather than downloaded twice.

`make render-soil-tractor SOIL_REPLAY_RUN=<completed-run> SOIL_REPLAY_OUTPUT=<new-directory> SOIL_REPLAY_FRAME=11` selects an explicitly captured frame; `final` remains the default. The loader checks the index against manifest/result schedules, canonical file names, time/count/endpoints, and exact machine-state identity against the committed JSONL row. Missing captures are rejected, never interpolated.

The fresh 50k run completed with matching source hashes. All four recorded times (0, 0.183333, 0.366667 and 0.55 s) were rendered from two fixed cameras and visually inspected. The images show evolving settlement and local particle displacement, not established persistent ruts or calibrated response. Four captures read back **3,200,000 bytes** in total (800,000 each; serialized xyz files are 600,000 bytes), with zero observed GPU allocations/uploads and **4.545037 ms** total capture wall time outside timing. Frame wall median remains **262.076455 ms**, p95 **266.477118 ms**, with all 30 measured deadlines missed. Seven benchmark helper and six replay tests pass. [Temporal evidence and images](benchmarks/soil/2026-10-09-tractor-temporal/SUMMARY.md). This is diagnostic visual progress, not 60 FPS, long-horizon or physical acceptance.

### Machine workgroup-scratch experiment — 2026-10-09

A separate **test-only** candidate moves 13 non-pointer per-joint scratch arrays from function-local to workgroup storage and runs one articulation per workgroup. Arithmetic, compensated rounding, runtime-stage expressions and pointer-passed constraint arrays are unchanged. Dispatch geometry changes with candidate selection; the original generic program remains the control and production default.

Three translated six-body machines pass eight stages (refresh, unrestricted stepping, scalar stops, velocity caps, D6 compliance, disable, restore and continuation) with zero mismatched full-state/compensated-residue words and independent constraint-behavior assertions. Seven dynamics steps total **115.988672 ms reference / 57.467296 ms candidate**. A separate single free-root tractor also passes exactly: **113.150944 / 55.454048 ms**, a 50.99% reduction in this short paired sample. Both fixtures retain all state comparisons and unchanged tolerances; six CPU checks pass. [Source, logs, telemetry and next steps](benchmarks/soil/2026-10-09-dynamics-scratch/SUMMARY.md).

These tests contain no soil. They establish a promising single-machine latency candidate, not a whole-frame speedup, general joint-family parity or 60 FPS. Production defaults remain unchanged; integrated exposure must use a distinct execution profile with explicit device limits and cache/rebuild identity, followed by actual coupling/rollback and 50k trajectory/resource comparisons. The experiment does not alter the failed or unresolved earlier rounding results.

### Runtime machine-scratch profile — 2026-10-09

`FeatherstoneExecution::{Generic, ExperimentalWorkgroupScratch}` now separates execution layout from compensated rounding. Existing constructors retain `Generic`. Explicit kernel/solver/runtime/tractor machine-option constructors preserve both selections across rebuilds; cache keys include execution, and preflight checks workgroup storage (with per-variable alignment) and one-workgroup-per-articulation dispatch limits. Candidate dynamics gets its own program; generic refresh and specialized drift are unchanged. Select the diagnostic with `SOIL_TRACTOR_BENCH_EXECUTION=scratch` (`generic` by default). The manifest names the profile and the benchmark verifies the actual kernel selection.

The actual 65-particle coupled tractor passes impulse/momentum/energy, rollback/resume, capacity/budget rejection, empty soil, zero subsequent-frame allocations and execution-profile preservation after a mass-edit rebuild. Same-source/same-executable 50k controls complete 33 frames each: median **248.866375 ms generic / 159.650926 ms scratch**, a **35.8487% observed reduction**; p95 **253.493475 / 164.913322 ms**. Required dispatches and transfers are unchanged, with zero hot buffer allocations. Both still miss every measured 60 Hz deadline. Extra-program cold construction increases **185.777 → 307.114 seconds**.

**Strict velocity parity fails, so default promotion is not justified.** Maximum generalized-rate difference is **0.000193595886** against the unchanged 2e-5 bound; linear/angular velocities also exceed it. Positions, rotations, generalized coordinates and final particle xyz remain below the bound. Rates first exceed it at committed frame **18**, with difference **0.000106632710**. Prior reference-repeat discrepancies and exact component tests do not establish the cause of this new discrepancy. Preserve these artifacts and investigate identical-input steps rather than rerunning unchanged broad timings. [Runtime, full-workload and resource evidence](benchmarks/soil/2026-10-09-runtime-dynamics-scratch/SUMMARY.md). Full S06/S08/S10 acceptance remains open.

### Identical live-input machine comparison — 2026-10-09

A `cfg(test)`-only runtime probe now compares the generic and scratch machine programs on the same resident model/control/body/joint inputs and exact copied input compensation residues, before authoritative soil projection continues. The candidate has a separate retained output and compensation state; its result is observed and discarded, never published into the live trajectory. The probe checks poses, velocities, forces, coordinates/rates, physical low parts, velocity deltas and zero fault status. The established nonphysical fourth position-residue lane remains excluded.

The actual native **50k** tractor completes **20 frames / 680 paired predictions** through the previously failing frame-18 window. All **139,400 compared words** match exactly. Run/source checks pass; total test time including cold construction is **313.18 seconds**. [Per-step artifacts, scope and reproduction](benchmarks/soil/2026-10-09-live-dynamics-parity/SUMMARY.md).

This rules out a machine-arithmetic mismatch for these observed identical inputs; it does not prove why independent trajectories differ, cover later frames or excuse the failed full-workload velocity gate. Diagnostic synchronization changes scheduling, so it is not production timing. Continue soil/contact input isolation; defaults and all acceptance bounds remain unchanged.

### Articulated-mass scratch extension trial — 2026-10-09

A further **test-only** variant moves six counts/motion-subspace/inverse-mass arrays into workgroup storage, in addition to the 13 arrays in the public experimental profile. Function-local scalar temporaries remain unchanged. The first pointer-parameter version failed validation: installed Naga 29.0.4 does not implement unrestricted workgroup pointer parameters. The corrected generated source accesses the workgroup globals directly, preserves call arity with unused scalar parameters, and retains all arithmetic and constraints. Candidate modules now validate before expensive generic pipeline compilation.

The corrected three-translated-machine fixture passes all eight mutating-constraint stages with zero mismatched state/residue words: seven dynamics steps total **116.398976 ms generic / 51.808928 ms extended**. A separate direct comparison against the **existing partial-scratch** backend on one free-root tractor also passes exactly: **54.467168 / 51.124064 ms**, an additional **6.14% observed component reduction**. Eight CPU checks pass; original failure and both successful source archives/logs are preserved. [Trial evidence](benchmarks/soil/2026-10-09-constraint-scratch/SUMMARY.md).

This is not a new public option or full-workload speed result. The extended variant remains private until coupled/live-input, resource and 50k evidence justify exposure. Generic remains default, and independent-trajectory parity, 4 ms soil, 60 FPS and physical acceptance remain unresolved.

### Runtime constraint-scratch integration — 2026-10-09

Following the private trial above, the extension is now explicitly selectable as `FeatherstoneExecution::ExperimentalConstraintScratch` / `SOIL_TRACTOR_BENCH_EXECUTION=constraint-scratch`. It remains separate from partial scratch and from rounding. Kernel/program execution identity is stored explicitly because both scratch profiles use one invocation per workgroup. Device preflight reserves 1200/5552/28320 bytes at 1/6/32 joints; existing constructors remain generic.

Nine kernel CPU checks and eight benchmark/parser checks pass. The actual coupled six-body/65-particle fixture passes in **305.84 seconds**, including cold construction, conservation checks, zero subsequent-frame allocations, rollback/resume, rejection paths and preservation after runtime rebuild. Three-profile public cache isolation/reuse passes in **301.06 seconds**. The actual 50k identical-input comparison passes **33 frames / 1122 predictions / 230,010 words**, with zero mismatches, in **426.94 seconds** including both cold profiles. Candidate outputs are discarded, so this diagnostic does not establish performance or independent-trajectory parity. The facade release build, source identity, formatting and whitespace checks pass.

Same-source/same-executable 50k runs observe median completed-frame time **160.729094 ms partial / 155.539683 ms extended**, a **3.23% reduction**; p95 **168.315010 / 160.454194 ms**. Both retain 783 dispatches, 18,404 upload bytes, 612 mandatory readback bytes per measured frame and zero hot buffer allocations. All 30 measured frames still miss 60 Hz. The **unchanged 2e-5 velocity gate fails**: generalized-rate maximum **0.005904138088**, linear velocity **0.006211431231**, angular velocity **0.003526702523**. First generalized-rate failure is committed frame **25**; pose/coordinate and final particle differences stay below the bound. This is not default-promotion evidence. Preserve the failure and investigate differing soil/contact inputs instead of repeating unchanged timing runs or relaxing tolerance. [Complete evidence and remaining gates](benchmarks/soil/2026-10-09-runtime-constraint-scratch/SUMMARY.md).

### Contact-transition substep isolation — 2026-10-09

A bounded `cfg(test)` trace now retains every projected substep across committed frames **23–26** for independent partial/extended 50k trajectories: **136 records each**, exact GPU ownership-validated grid-node identity, current centers/shapes, live packed node order, mass/Jacobians/free rates, reduced systems and projected outputs. Retained staging is capped at 4 MiB; no observer or extra geometry fields exist in production builds. Both trajectories complete 26 frames and all trace coverage checks pass in **487.82 seconds**, including cold construction. This is not a timing benchmark.

The captured window contains **34 contact/boundary membership disagreements**. At the first, frame **25 / substep 15**, extended includes one **0.47464246 kg** tine node (binding 4, grid index **122961**, +Y normal), while partial excludes it. Maximum rate difference grows from **0.00002998114** before projection to **0.00645053387** afterward. A saved-geometry GPU probe using the production membership functions reproduces that inclusion/exclusion in **0.45 seconds**: signed box margins are **-1.013279 µm / +4.559755 µm**.

Both compact exported snapshots replay original-order outputs exactly. Three saved-input/capture/f64 tests pass in **3.04 seconds**, including permutation checks without relaxed bounds. For the occupied snapshot, removing only the node changes the independent f64 rate solution by **0.00647188931**; disabling only its normal changes it by **0.00615081398**. GPU/f64 errors remain below **1.83e-7**. This identifies a concrete geometric hard-switch amplification in this transition, not the earlier cause of all trajectory variation or a contact-model repair. Contact activation/withdrawal needs physical gap and conservation oracles; arbitrary geometry epsilons or mass cutoffs are not justified. Defaults and failed gates remain unchanged. [Saved traces, replay commands and limits](benchmarks/soil/2026-10-09-contact-transition-trace/SUMMARY.md).

### Unilateral normal-contact component — 2026-10-09

Implemented a separate `ExperimentalSoilNormalProjectionGpu`, not a replacement runtime selection. It solves finite-mass, frictionless normal contact for one machine island (1–32 DOFs) with distinct scalar soil normal DOFs. Inputs are resident machine mass/free rates, contact Jacobian rows, soil masses/free normal velocities and nonnegative `gap / dt` allowances. Separation is permitted; the component does not pull withdrawing soil back onto the body. Zero mass denotes an inactive slot, not an arbitrary small-mass cutoff.

Eliminating the independent soil normal velocities gives a strictly convex hinge-quadratic objective. A bounded damped semismooth Newton solve and parallel impulse recovery use two encoded passes, no inner readback or CPU fallback. Sticky faults reject invalid input/mass, nonfinite arithmetic, failed line search or exhausted iterations. Convergence checks machine-rate correction, induced contact-impulse correction and generalized momentum residual; these are all needed at disparate mass ratios. Line search evaluates slope plus branchwise curvature rather than subtracting nearly cancelling energies.

Four GPU suites pass in **3.55 seconds**, including analytic mass/gap/withdrawal cases, continuity at zero impulse, a deliberate active-set crossing, **36 exhaustive f64 comparisons at widths 1–8 and 32**, and an 860 kg six-DOF spatial impulse fixture. Two CPU checks pass. Initial impulse and momentum/line-search failures are preserved and corrected without relaxing their oracle gates. Hot encoding observes zero buffer allocations/uploads/readbacks. This is not an integrated speed or contact-repair result. [Derivation, constraints, failures and reproduction](benchmarks/soil/2026-10-09-unilateral-soil-normal/SUMMARY.md).

**Integration remains open:** rows sharing soil DOFs cannot be duplicated as independent contacts; fixed-grid normals and general friction require the corresponding coupled block treatment. Geometry generation, penetration/withdrawal policy, runtime publication/rollback and full-workload verification are not implemented for this component. Existing sticking coupling and all default/failed acceptance gates remain unchanged. The split free-motion/contact design is informed by [convex MPM/articulated contact research](https://arxiv.org/html/2403.13783v2), but this frictionless lumped-node component is not that paper's complete implicit/frictional method.

### Shared soil-node/floor component — 2026-10-09

Added `ExperimentalSoilFloorProjectionGpu` around the normal solver: one contact normal and fixed axis walls act on the same physical 3D soil velocity. Resident preparation emits at most three hinge-potential terms per node; physical recovery publishes node velocity and separate contact/external-floor impulses. The terms are not duplicate physical soil masses. Existing runtime/default coupling is unchanged.

Three GPU suites pass in **5.48 seconds**, including 36 single-node full-space f64 QP comparisons, two nodes sharing a machine, near-wall-normal conditioning, actual wall activation, momentum/energy/complementarity and fault/resource checks. One CPU layout test passes. A fully blocked normal requires a hard affine machine constraint: this component deliberately faults rather than inventing a large mass. Hard-row support, friction, geometry generation and runtime publication/rollback remain open before tractor integration. No full-workload speed or physical-calibration claim follows. [Model, evidence and reproduction](benchmarks/soil/2026-10-09-soil-floor-condensation/SUMMARY.md).

**Hard-row extension:** fully blocked normals now generate the exact inequality `a*q <= gap/dt` rather than faulting solely for zero remaining normal mobility. A bounded resident working-set Newton solve combines those inequalities with the finite hinge potential; recovery adds hard multipliers to physical contact impulses and preserves separate external-floor reactions. Five floor GPU suites pass in **10.97 seconds**, including 24 hard/mixed independent full-space QPs, explicit negative-multiplier removal, duplicate/opposing constraints, a full-rank 32-DOF corner, nonfinite rejection and iteration-exhaustion publication protection. This supersedes the unsupported-hard-row limitation above, not the runtime/friction/calibration/performance gaps. [Algorithm, bounds and evidence](benchmarks/soil/2026-10-09-soil-hard-rows/SUMMARY.md).

### Maintenance notes

**Exact-zero position rotation (2026-10-09):** a bitwise-zero/finite-quaternion guard skips redundant compensated rotations without discarding tiny offsets. 4,824 helper cases under two rounding modes and three translated six-body dynamics fixtures pass; all physical state/residue words remain exact. Paired seven-step GPU totals improve **105.427040 → 64.761504 ms**. The changed 50k run completes 33 frames at **220.259558 ms median wall**, machine dynamics **108.018496 ms**, corrected drift **27.910528 ms**, with zero hot buffer allocations. Every measured frame still misses 60 Hz; independent velocity differences reach **5.3405762e-5**, above the unchanged 2e-5 bound. Actual coupled-input shortcut equivalence and full acceptance remain open. [Evidence and remaining gates](benchmarks/soil/2026-10-09-zero-position-rotation/SUMMARY.md).

**Additional shortcut coverage (2026-10-10):** focused corrected-drift and changing-constraint fixtures now pass with zero mismatched state/compensation words at origins 0/1024/65536. Corrected drift retains injected projected rates/nonzero velocity deltas; the D6-root fixture exercises scalar stops, velocity caps, compliance, joint disable and restoration. Seven paired GPU steps total **26.007584 → 17.349472 ms** for drift and **41.743968 → 26.630560 ms** for constraints. Seven CPU tests, facade build, formatting and source checks pass. This extends sampled equivalence only; no unchanged full benchmark was rerun and no performance/calibration gate is closed. [Focused evidence](benchmarks/soil/2026-10-10-zero-rotation-coverage/SUMMARY.md).

**Rejected XYZ-only product promotion (2026-10-10):** omitting the discarded scalar row of the final position quaternion product passes 13,017 helper cases and translated dynamics/drift exact-state comparisons. Ordinary dynamics improves only 0.34% in the short paired sample, with some steps slower; this is not robust performance evidence. Production is restored byte-for-byte and the candidate remains test-only. Do not repeat unchanged or spend a full workload on this weak signal. [Implementation and measurements](benchmarks/soil/2026-10-10-position-product/SUMMARY.md).

**Rejected normal-contact cache promotion (2026-10-10):** serial and parallel bounded ordered caches preserve both captured inputs bit-for-bit, including the independent f64 oracle. Neither yields robust speed improvement; serial construction regresses the later input and parallel construction regresses the earlier one. Production is restored byte-for-byte; the retained candidate has complete original-scan overflow fallback and focused sparse/empty/capacity/hard-row/reuse checks. Six normal and five floor GPU suites pass. No full workload was run. [Implementation, measurements and rejected hypotheses](benchmarks/soil/2026-10-10-normal-contact-cache/SUMMARY.md).

**Deferred final momentum check (2026-10-09):** nonconverged normal iterations now skip the unused compensated publication calculation, retaining all final bounds and hard-row checks. Ten GPU/two CPU checks pass; both saved actual cases and all **1,122 live normal solves** remain bit-identical. Paired component GPU lower medians improve **1.161856 → 1.050336 ms** and **1.901664 → 1.668960 ms**. The changed full 50k run completes 33 frames at **310.841679 ms median wall**, **65.607680 ms contact projection**, zero hot buffer allocations, but all measured frames miss 60 Hz. Independent joint-rate/angular-velocity differences still reach **2.6702881e-5**, above the unchanged 2e-5 gate. Full accuracy/performance acceptance remains open. [Source, timings and retained failure](benchmarks/soil/2026-10-09-lazy-publication/SUMMARY.md).

**Shared rounding dynamics/drift differential check (2026-10-09):** the test-only live probe now compares shared rounding on identical actual unilateral-coupled inputs for both machine prediction and corrected drift. The corrected 50k/33-frame run passes **1,122 + 1,122 comparisons**, all **460,020 physical/status words bit-identical**, in 420.97 seconds including cold construction. A first invalid-input diagnostic and its storage-only-copy repair are preserved. This closes the sampled corrected-drift comparison gap, not independent-trajectory repeatability, timing, long-horizon or physical acceptance; shared rounding remains experimental and prior gates stay unchanged. [Exact coverage and reproduction](benchmarks/soil/2026-10-09-live-rounding-drift/SUMMARY.md).

**Native stencil split rejected (2026-10-09):** test-only stress preparation plus multi-thread stencil scatter passes boundary and evolved 50k component checks, including unchanged 2e-5 bounds and positive-mass support, but costs more than the ordinary native scatter. Paired GPU medians are **0.074256 → 0.128048 ms** for 32 threads/particle and **0.074192 → 0.108672 ms** for four, plus a 2.4 MB cache. Both variants remain outside production; no timestep, tolerance or default changed. Each evolved comparison takes 0.63 seconds without machine construction. Do not retry these unchanged or equate component parity with trajectory acceptance. [Source and evidence](benchmarks/soil/2026-10-09-native-stencil/SUMMARY.md).

**Evolved native P2G isolation (2026-10-09):** a complete soil checkpoint and exact uniform snapshot now preserve the actual 50k tractor state after 15 committed frames. Six native reset/P2G/grid-update evaluations leave all particle/history bytes unchanged but vary mass, momentum and free-grid velocity. Standalone replay completes in **0.55 seconds**, with maximum mass/momentum/XYZ-velocity spreads **1.430511e-6 / 9.536743e-7 / 1.192093e-6**. This identifies an upstream varying stage without proving the cause or amplification of the larger independent-trajectory differences. The diagnostic avoids further cold tractor construction; it changes no production solver, timestep, buffer flags or tolerance. Repeatability, physical accuracy and real-time acceptance remain open. [Capture, corrected field scopes and replay evidence](benchmarks/soil/2026-10-09-native-repeatability/SUMMARY.md).

**Parallel publication check (2026-10-09):** per-DOF compensated momentum validation now runs cooperatively, preserving the original bound and hard-row contributions. Paired saved solves remain bit-identical and measure **1.672832 → 1.157504 ms** and **2.399040 → 1.906816 ms**; ten normal/floor GPU suites and two CPU tests pass. The full 50k run completes 33 frames at **324.198409 ms median wall**, with **72.313856 ms** contact projection and zero hot GPU buffer allocations, but independent-run trajectory differences exceed 2e-5. A test-only shadow diagnostic then compares both shaders on **all 1,122 identical resident normal solves** in a full run: every status/rate/impulse/velocity word matches. This verifies sampled component equivalence, not whole-trajectory repeatability or its divergence cause. All 30 measured frames still miss 60 Hz; the 4 ms target and integrated accuracy remain open. [Reverted variant, component/full timings and differential evidence](benchmarks/soil/2026-10-09-normal-publication/SUMMARY.md).

**Cooperative normal solve and cancellation repair (2026-10-09):** contact validation/evaluation and independent Hessian/gradient rows now run across a 64-lane workgroup. A validated contributing-contact prefix avoids scanning unused capacity during Newton iterations, and the existing parallel recovery pass publishes contact velocities. Compensated gradient accumulation repairs a second captured iteration failure without changing numerical bounds. Both saved cases solve with independent f64 errors below 1e-6; five normal and five floor GPU suites pass, plus two CPU accounting/curvature tests. Alternating same-input timing for the earlier successful case is **42.682048 ms serial → 1.553568 ms candidate**; this is component evidence, not full-frame acceptance. The later case changes from a 64-iteration failure to a three-iteration solve. The changed-source actual 50k workload completes **33 frames / 0.55 simulated seconds**, with zero hot GPU buffer allocations and workload/source exits 0. Its 30 measured frames have median wall **342.779768 ms**, contact projection **89.518080 ms**, machine dynamics **184.321568 ms** and corrected drift **46.217856 ms**. All measured frames miss 60 Hz; renderer/export remain excluded. The 4 ms target, long-horizon stability, friction and calibration remain open. [Sources, failed variants, paired timings and full-run evidence](benchmarks/soil/2026-10-09-normal-parallel/SUMMARY.md).

**Captured normal-solver stagnation (2026-10-09):** a test-only first-local-failure capture now preserves exact solver inputs before later substeps/rollback overwrite them. The actual 50k failure replays in 0.62 seconds rather than requiring another three-minute machine construction. The repaired resident solver retains low-order rate increments, evaluates canceling contact terms accurately, and validates the momentum balance of rounded published rates with solved impulses before accepting a result. The saved case now converges in two iterations instead of exhausting 64, with independent f64 rate/impulse errors below 1e-6 and all original gates unchanged. Two replay/capture checks and all nine normal/floor GPU suites pass. The new full run commits 15 frames / 0.25 seconds but fails on a later iteration limit at attempted frame 15. Its incomplete wall median is **1786.071758 ms**, with contact projection **1528.342528 ms**: slower, not accepted throughput. Next work is capturing that later fault and reducing/parallelizing serial contact work. This does not close stability, the 4 ms target, friction or calibration. [Failure variants, final arithmetic and reproduction](benchmarks/soil/2026-10-09-normal-replay/SUMMARY.md).

**50k unilateral diagnostic and mass symmetry (2026-10-09):** benchmark selection is now explicit through `SOIL_TRACTOR_BENCH_UNILATERAL_NODES`, with separately recorded margin/iteration bounds and a live configuration check. Nine helper tests pass. The initial 50k run fails closed in frame zero on nonsymmetric runtime mass input, before any measured result. A focused GPU regression reproduces unequal diagonal-block transpose coefficients. CPU/GPU CRBA now compute each lower-triangle coefficient once and mirror it; strict normal-solver validation is unchanged. Three articulated-response GPU suites and four CPU CRBA tests pass. The corrected workload commits 14 frames / 0.233333 seconds before normal-solver iteration exhaustion at attempted frame 14. Its incomplete measured prefix has median wall **984.961542 ms**, with **733.389824 ms** in contact projection; all measured frames miss 60 Hz. This is a failed diagnostic, not acceptance or a paired speedup. Next work is exact failing-input capture, solver convergence and parallel contact assembly; defaults remain unchanged. [Failure, fix and run evidence](benchmarks/soil/2026-10-09-unilateral-tractor/SUMMARY.md).

**Experimental unilateral runtime (2026-10-09):** `RigidWorld::enable_mpm_soil_unilateral_coupling` and `SoilTractor::enable_unilateral_coupling` now connect candidates, analytic geometry, fixed-floor normal solving and exhaustive frozen-contact coverage to actual MPM/machine stepping. Selection is immutable while attached and remains opt-in; the sticking default is unchanged. Machine rates and particle dispatch require the final shared status, and frame rollback restores both systems. Deterministic contact-wrench reduction excludes external floor reactions and is not reapplied as a second machine impulse. The actual six-body/65-particle fixture passes in **189.93 seconds**, including cold compilation, changed timesteps, both state parities, unchanged momentum/energy and reaction gates, rollback/resume, capacity-fault rollback, budget rejection and empty soil. Steady frames allocate zero GPU buffers. This is short integrated correctness evidence, not the 50k workload, friction, penetration repair, multi-body shared-node contact, long-horizon stability or the 4 ms target. [Evidence and reproduction](benchmarks/soil/2026-10-09-unilateral-runtime/SUMMARY.md).

**Certified resident candidates (2026-10-09):** `ExperimentalSoilCandidatesGpu` now scans the dense free grid, emits unique node/body queries in deterministic grid-index order and composes with analytic geometry and the floor solver. An exhaustive GPU check of every occupied node/body normal inequality uses **corrected** machine/node motion before guarded grid publication. Narrow envelopes that miss a contact fault rather than silently omit it; multiple-body candidates and capacity overflow also fault. Five GPU suites pass in **2.77 seconds**, including 64³ occupied-grid coverage, exact candidate replays, incoming outside nodes and contacts activated by the coupled solve. This is frozen-contact component evidence, not nonlinear swept collision or live tractor acceptance. Actual runtime state publication/rollback, general shared-node multi-body contact, friction and the 4 ms target remain open. [Contract and evidence](benchmarks/soil/2026-10-09-soil-candidates/SUMMARY.md).

**Resident contact geometry (2026-10-09):** `ExperimentalSoilContactGeometryGpu` now writes floor-solver nodes and generalized normal rows directly from the existing sphere/box/local-Y-cylinder shape and world-COM Jacobian buffers. It retains signed distance for diagnostics, uses nonnegative `gap/dt` for unilateral contact, and composes with the floor solver under shared sticky status in one encoder. Four GPU suites pass in **1.48 seconds**: independent nearest-boundary geometry, 108 signed-gap derivative checks, a tine-surface/positive-gap release sweep, captured partial/extended tine transforms and invalid-input/publication/resource checks. Captured signed gaps are +1.013279/-4.5597553 µm. This supplies geometry for explicitly paired unique node/body queries; it does **not** yet enumerate conservative candidates, handle multiple body normals on one node, repair penetration, provide friction or integrate runtime rollback. Default coupling and full-workload gates remain unchanged. [Supported scope and evidence](benchmarks/soil/2026-10-09-soil-contact-geometry/SUMMARY.md).

**Shared rounding experiment (2026-10-09):** shared scratch with persistent coordinate writeback is now explicitly selectable through `FeatherstoneRounding::ExperimentalShared`; existing constructors still default to storage rounding. Corrected drift and momentum-delta preservation pass exact translated-tractor checks. GPU cache isolation/reuse and the actual coupled-runtime conservation, overflow, rollback/resume and resource fixture pass. Select the diagnostic with `make bench-soil-tractor SOIL_TRACTOR_BENCH_ROUNDING=shared`; the manifest identifies it.

Same-source, same-executable 50k runs complete 33 frames each: wall medians are **259.114929 ms storage versus 204.279447 ms shared**, an observed 21.16% reduction, not real-time acceptance. A subsequent unchanged storage control measures **249.443700 ms**. Every measured frame still misses 60 Hz. Full-state artifacts show small position differences but velocity differences above 2e-5; the unchanged storage repeat also exceeds that bound (maximum generalized-rate difference 0.01097816). Reference repeatability is therefore unresolved, and this does not prove all candidate differences are harmless. Keep the profile experimental, preserve the failed comparisons, and isolate identical-input substeps around the boundary-contact transition before default promotion. [Current evidence and commands](benchmarks/soil/2026-10-09-shared-rounding-drift/SUMMARY.md); [earlier component evidence](benchmarks/soil/2026-10-09-machine-shared-rounding/SUMMARY.md).

**Historical boundary-response experiment (2026-10-09):** the initial GPU feasible-active-set QP failed an offset-arm/opposing-normal impulse fixture and was not then connected to MPM. [That failed evidence remains preserved](benchmarks/soil/2026-10-09-boundary-projection/SUMMARY.md). The later precision repair, integration and current limitations are recorded under the latest S08 boundary checkpoint above and in [the boundary-wrench checkpoint](benchmarks/soil/2026-10-09-boundary-wrench/SUMMARY.md).

**Projection scratch trial (2026-10-09):** narrower private arrays pass exact boundary/empty-contact comparisons across all 32 supported DOF widths and saved actual-input replay. Under 100% GPU utilization, alternating-order component wall p50 is 54.859825 ms reference versus 55.251008 ms candidate; timing is inconclusive. Production retains 32-wide scratch; the candidate remains private/test-only. Repeated fixed-input projections are not simulated-time throughput. [Original evidence](benchmarks/soil/2026-10-09-projection-scratch/SUMMARY.md).

A later run starting at 0% GPU utilization observes only a small sized-scratch component improvement: nearest-rank wall p50 **11.908406 versus 11.602331 ms**. Contention subsequently returns. New test-only per-pass timestamps reveal opposing plain-solve/boundary costs in the contended fixture, motivating a split candidate that retains original boundary scratch. Its saved-input outputs remain exact, but variable timings do not justify default promotion. Constant-width substitution is separately rejected by the unchanged bit-exact width gate: width-1 empty-contact output signed-zero bits differ. No normalization or tolerance relaxation is introduced. The initial oversized query-set validation failure is retained; profiling now uses two bounded 3400-query sets. [New candidates, attribution and retained failures](benchmarks/soil/2026-10-09-projection-scratch-idle/SUMMARY.md).

Watch changes to contact timestep, material stiffness, grain size, tile stencil width, renderer consumers, and snapshot ownership: each can invalidate old performance or conservation evidence. Neighbor rebuild bounds must evolve with collision radii and interaction laws. A new diagnostic must not reintroduce synchronous full-state downloads. A new terrain backend must implement the same explicit ownership/lifecycle contract without pretending all backends have the same physical fidelity.

## Key Learnings

1. Existing soil integration contains avoidable CPU/GPU synchronization and multiplicative terrain-export work; these should be addressed before selecting a new solver family.
2. GPU-resident storage alone does not guarantee a GPU-resident execution path.
3. Real-time acceptance requires completed simulated-time accounting, absolute latency, and preserved physical behavior, not headline FPS or relative speedup alone.
