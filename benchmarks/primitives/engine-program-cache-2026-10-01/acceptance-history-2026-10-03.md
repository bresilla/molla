# Molla primitives implementation report

**Status: LBVH timing passes under low load; final unchanged full GPU aggregate is running.**
Updated 2026-10-03. The original [plan](MOLLA_PRIMITIVES.md) and its unchecked
acceptance checklist remain unchanged. No Gearbox/USD conformance is claimed.

## Session checkpoint

### Goal
Implement all original MOLLA_PRIMITIVES.md requirements,including P0–P6 and
the original acceptance gates,without reducing scope to passing components.

### Instructions
Implementation before validation; concise honest progress. Preserve physical
fixtures/tolerances and performance workloads. Use Make/Nix and patch edits.
No commits,branches,worktrees,sibling/model edits,dependency-pin changes or
termination of external applications.

### Discoveries
- On resume at2026-10-03 10:24+02 external GPU load had dropped:11% before
  and4% after the unchanged full broadphase_lbvh_gpu file; Astrocraft reports
  no active SM usage. All8 tests pass,including100k best22.022ms
  (22.34/22.02/23.27ms) versus30ms and10k best3.434ms versus8ms,31.87x CPU
  SAP speedup for that fixture. All1482 source hashes verify before/after.
  The only changed condition is observed load,no code,test or threshold edits.
  This resolves the immediate timing blocker without treating the desktop as
  a completely isolated GPU. Earlier high-load failures remain preserved.
- At2026-10-03 07:00+02 the same contention blocker remains on the third
  consecutive goal turn since the timing failure. All Molla test processes
  are terminal; the GPU is93% busy and Astrocraft PID2289973 remains active
  with67% SM usage. The user has not responded to the pause/close request.
  Remaining physical tests and the requirement/artifact audit are finished;
  no meaningful acceptance work remains possible without uncontended GPU
  availability. Preserve the failure rather than relaxing30ms or repeatedly
  retrying a saturated device. Goal status is blocked,not complete or paused.
- The unchanged original full GPU aggregate stopped at2026-10-03 04:22:06+02
  on the100k LBVH wall-time gate:103823 shapes,best34.247ms versus the unchanged
 30ms limit(samples40.52/34.55/34.25ms). Pair/topology/refit checks passed;
  the whole file was7 passed/1 failed. At04:27,with our runner terminal,the
  GPU was93% busy and external Astrocraft PID2289973 accounted for80% SM use.
  This supports a contention hypothesis,not an isolated regression diagnosis.
  No external application was stopped. User was asked asynchronously to pause
  Astrocraft for an uncontended timing check. One unchanged-fixture diagnostic
  rerun is recorded separately; the original aggregate failure stays preserved.
- The single diagnostic rerun also fails the unchanged30ms gate:best32.132ms
  from41.08/32.13/35.92ms,terminal2. Before/after telemetry records94–99% GPU
  utilization with external Astrocraft consuming59–68% SM. Fixture SHA verifies
  unchanged. Do not repeatedly retry until a lucky pass or weaken the threshold;
  an uncontended observation is still needed to distinguish contention from
  an independent performance regression.
- Quaternion FK boundary failures are stale normalization expectations:
  initialize now normalizes prescribed fixed/kinematic f32 quaternions into
  high/low pairs. The old CPU oracle left them unnormalized,and the fixed-only
  test required zero tails. Updated oracle normalizes input poses in f64 and
  checks normalized boundary pairs; the existing2e-9/3e-7 FK tolerances remain
  unchanged. Unit-norm1e-12 now covers every body,not only dynamic ones; the
  zero-coordinate fixture additionally checks components within2e-14. Production
  unchanged. Original fixture SHA f85046367633ad370656a50420898e309d7a8af2aefe6d7fd4ab6e96cd5bb8b7
  is preserved in quaternion-boundary-original-fixture.rs.
- Impact-island capacity counted separating contacts before predicted/onset
  selection,although those contacts correctly remain graph edges and are
  excluded from the impulse solve. A connected chain with only two solving
  contacts therefore incorrectly faulted2097152 at capacity2. contact_islands.wgsl
  now counts impact contact capacity only after selection; graph union and
  non-impact loaded/frictionless contact accounting remain unchanged. The
  original3/130-body endpoint-exchange regression is unchanged.
- Stage-level device readback localizes the new analytic failure to floating
  momentum transport,not compensated Featherstone integration. Refresh is
  exactly1; predictor and redrift both produce high1+low4e-8; only momentum
  transport adds the spurious half-ulp translation. It formed COM shift by
  subtracting separately rounded0.5m-scale centers after adding tiny travel.
  rigid_momentum.wgsl now subtracts high/low body positions before reducing
  relative COM offsets and computes shift from anchor/offset differences.
  Composite-inertia arms use the same direct relative centers. The unsuccessful
  rounded-operands candidate was fully reverted; all temporary tracing removed.
- The new independent motion oracle exposes a genuine pre-detachment failure
  at one substep,frame0:both GPU worlds publish1.000000099604648 rather than
  analytic1.0000000400000009,an extra5.960464722e-8m. Matching trajectories
  alone had hidden this. Original refined-oracle failure and diagnostic are
  preserved; its tolerance was not changed after failure. The candidate that
  rounded both two-sum operands did not fix it and was fully reverted. The
  accepted relative-COM momentum correction and evidence are described above.
- Detached-root diagnosis finds an over-constrained bitwise trajectory oracle:
  even before breaking,parent position+1 differs from published child position
  by one low-part ulp because their paired-f32 high exponents differ. Independent
  binary32 two-sum stepping also differs after topology changes. The revised
  test preserves the original80 frames/four substeps,adds1/8-substep refinement,
  exact boundary continuity,exact velocity/rotation/unaffected-root checks and
  an independent analytic-motion oracle. Post-detachment positional comparison
  uses an explicit step/scale-derived paired-roundoff budget,not exact equality.
  contact-break-precision-oracle.md documents the derivation and original
  failures. No production change or physical-plan tolerance change is made.
- The unchanged public contact-breakage file reaches4/5 passes,then exact
  free-root detachment parity differs by3.552713679e-15m at frame10,body2:
  2.0000004400000293 versus2.0000004400000257. Preserved original fixture in
  contact-break-precision-original-fixture.rs,SHA037c01608c00c7fe86a1c10e2ae388ea25f1a2e3b707fe8ed92a6c01d366ac62.
  No tolerance change; targeted diagnostic records poses,velocities and joint
  coordinates before diagnosing the retained-compensation arithmetic.
- Immovable contact reaction loads and predictor-force assertions pass,but the
  old final pose assertion expected a single-f32 -0.1 coordinate. Published
  poses now include the retained low part (-0.09999999999999998). The corrected
  oracle computes the paired-f32 representation from authored position and keeps
  exact pose equality; added1/3/8-substep frames also require exactly zero fixed
  velocity. Production code,force/torque assertions and physical fixture remain
  unchanged. A search found no other identical f32-pose expectation.
- Zero-body static-collider scenes allocated only4 bytes for contact activation
  dependencies,while WGSL array<vec2<u32>> requires an8-byte binding even with
  zero live edges. The constructor now allocates max(body_count,1)*8; dynamic
  layouts are unchanged. Added a zero-body actual-dispatch regression beside
  the existing one-boundary-body test; original public static-only fixture is
  unchanged. No shader,solver law or physical tolerance changed.
- ScalarConstraintRow kind3 is supported bounded feedback; an older contact-island
  validation fixture incorrectly used3 as invalid. It now rejects4 and active0,
  explicitly accepts bilateral1/nonnegative2/bounded3,and retains capacity,
  sticky-fault,invalid-active-mask and out-of-range-body checks over two cycles.
  Production graph validation and solver code are unchanged.
- The next drift failure is an obsolete oracle,not a loss of precision. The
  fixed-frame fixture expected narrowed f32 frame offsets,but JointFrameReference
  and compensated_body_position preserve authored f64 frame tails. At scale1e-6,
  authored-minus-rounded offset=1.577973279e-16m,matching the reported discrepancy.
  Corrected only the expected frame term; all fixtures,step counts and the
  scale*2e-13 tolerance remain unchanged. No production change for this correction.
- Removing double-counting fixes the exact COM mutation fixture,but exposed a
  second detachment difference of3.637978807e-12m at frame18:kinematic position
  stepping used a different two-sum order than free-coordinate stepping. The
  kinematic displacement now uses the same three-stage compensated addition as
  coordinate_advance,retaining the low-part sum error instead of dropping it.
  Neither original physical fixture nor any assertion/tolerance was changed.
- GPU pose mutation exposed double-counted translation residues after COM edits
  and automatic break detachment. Published host coordinates already include the
  device low part; rebuilding from them must seed their own f32 rounding residue,
  not add the old body's residue again. rigid_contact_precision_gpu.rs now keeps
  rounded seeds for changed coordinates and newly free detached coordinates.
  Unchanged coordinate/body mappings still retain their device low parts.
- The prior tool host exited with SIGTERM at approximately12:41+02; session3451
  subsequently disappeared and its runner/Cargo PIDs were absent. Native output
  ends with Make Terminated,not an assertion failure. At13:10+02 the host reports
  only26 minutes uptime,so previous process identities must not be reused.
  Current detached runner records PID and Linux start ticks and writes its final
  exit atomically; no resilience across an actual host reboot is claimed.
- The11:34+02 failure is a stale rejection assertion,not failed loop physics:
  runtime_break_detachment.rs expected explicit repair of a compliant Fixed
  loop to be unsupported,but rigid_world_gpu::validate already supports it and
  runtime_loop_compliance exercises physical repair. Do not regress that support
  to satisfy the old guard. The replacement uses a loaded independent recurrence
  across broken/repaired/broken/repaired phases on CPU and GPU.
- Engram tools remain unavailable; this workspace checkpoint preserves context.
- Both previous solver regressions now pass without relaxed assertions:retained
  corrected rates prevent distance rate/acceleration cancellation,and paired
  high/low rates prevent hinge equality-row convergence from losing low bits.
- Device friction transition fixtures and public-runtime drive/friction/reversal
  fixtures provide different evidence; a CPU-only filename is not by itself
  proof that the corresponding GPU behavior is absent.
- A report containing18,367 chronological lines mixed stale/current status.
  Its complete history is now archived separately; this file is the current
  section14 return package. No production source or fixture changed for this.

### Accomplished
- At2026-10-03 14:28+02 final original GPU rerun2709083/start_ticks7808502
  is verified live:155 passes,zero failures/ignored across45 completed summaries;
  runtime_cone_limits is executing its three-rate10-second pendulum. The37-file
  baseline,public runtime contract and distance suites have passed again.
  Original workload/selection/source remain
  unchanged. This is verified waiting on the existing runner,not a restart;
  terminal aggregate acceptance is still pending.
- At2026-10-03 10:25:30+02 low-load-full-gpu-acceptance.sh started the
  unchanged original make test-primitives-gpu with no lane overrides.
  Runner2709083/start_ticks7808502 is verified live. It uses the same1482-path
  quaternion-boundary manifest,12h timeout,Make/Nix and serialized original
  selection. This current run,not prior segmented evidence,must reach terminal
  success. No external process was stopped or paused by the agent.
- Remaining-scope runner2431847 completed with exit0 at2026-10-03 06:56:14+02.
  Barrier2,normal4,drift5,contact-tail84 and later-lanes61 total156 passing
  invocations across31 summaries,zero failures/ignored. All1482 source hashes
  verify afterward. Together with the source-identical original aggregate,
  the original GPU selection has412 passes and one failed timing assertion
  across97 summaries. The diagnostic timing rerun is separate,not hidden in
  those counts. There are no active Molla test runners now.
- At06:58+02,after our tests finished,the GPU still shows90% utilization and
  external Astrocraft PID2289973 accounts for89% SM use. No user response to
  the isolated-timing request has arrived. Do not claim isolation or retest
  repeatedly under the same load.
- Final-scope artifact check confirms56 performance rows,43 fields each,
  unique family/count/backend keys,55 collected snapshots and one capacity
  rejection; every referenced source manifest/raw log exists. The expanded
  dirty snapshot records3315 paths:248 modified,3066 untracked and the one
  pre-existing lbvh.wgsl deletion. develop/base/one-worktree remain unchanged;
  Cargo.lock and flake.lock have no diff. Acceptance is not complete.
- At2026-10-03 04:36+02 remaining-scope runner2431847 is verified live:
  barrier2/2 and normal-contact4/4 pass with exit0 and source-after guards;
  drift is executing. The LBVH timing failure remains open pending an
  uncontended check; no response to the Astrocraft request has arrived yet.
- At2026-10-03 04:29:44+02 a source-guarded remaining-scope runner started:
  lbvh-contention-remaining.sh,PID2431847,start_ticks5673914. It runs the exact
  original contact-pipeline suffix after broadphase,then the unchanged later
  production lanes. This is explicitly segmented evidence,not a passing full
  aggregate. The manifest is still quaternion-boundary-source-sha256.txt;
  no production or test changes were made. The barrier lane is executing.
- The original full GPU aggregate is terminal2 at2026-10-03 04:22:06+02:
  256 passed,one LBVH timing failure,zero ignored across66 summaries. All1482
  source hashes verify afterward. Runner664386/start_ticks2001673 is no longer
  live; do not poll it as active or promote its partial results to aggregate
  success. Cone limits7/7,Ball loops5/5,reaction/breakage/kinematic lanes,
  pose precision5/5 and resident geometry17/17 completed before the timing gate.
- At2026-10-03 02:55+02 the unchanged full GPU aggregate has155 passes,
  zero failures/ignored across45 completed summaries. The entire37-file GPU
  baseline phase and public runtime contract12/12 pass. Distance runtime10/10,
  sleep3/3,contract4/4,COM roots4/4 and scale-library4/4 pass. The current file
  is runtime_cone_limits; cone geometry2/2 and capability2/2 already pass.
  Runner664386/start_ticks2001673 is verified live. Current10-second distance
  errors are4.367491819e-5/1.091913809e-5/2.729811238e-6m at60/120/240Hz;
  one-second CPU/GPU position error peaks8.907788264e-7m. The40-case/rate
  scale sweep's worst relative length error is2.331603355e-9,analytic position
  error1.052800337e-5m,relative reaction1.951173826e-7 and relative momentum
  1.554844129e-4 with the fixture's1e-3 absolute floor. Original thresholds
  and workloads remain unchanged; full GPU acceptance is still pending.
- At2026-10-03 00:54+02 the unchanged full GPU aggregate has80 passes,
  zero failures and zero ignored across29 completed files. The current file
  is runtime_d6_angular_mass; runner664386/start_ticks2001673 is verified live.
  Newly completed evidence includes rigid-loop drives9/9,acceleration drives2/2,
  Ball drives3/3,loop brakes1/1,contact sleep7/7,D6 singular bounds2/2,
  velocity coupling3/3,angular locks2/2,disabled joints1/1 and mixed distance4/4.
  The source guard quaternion-boundary-progress-source-2026-10-03-0054.log
  verifies all1482 paths; git diff --check passes. Only report updates occurred
  during this continuation; no test selection or implementation changes.
- At22:52+02 the unchanged original full GPU aggregate has22 passes,zero
  failures and zero ignored across nine completed files; runtime_loop_contacts
  is executing. Runner664386/start_ticks2001673 is verified live. The loaded
  static-friction10-second fixture passes at60/120/240Hz,eight substeps,four
  cases per rate:peak tangential drift2.683825373e-5/3.511552531e-5/
  3.607930302e-5m,well below the original1e-3m gate. Post-settle creep maxima
  are6.433362158e-7/3.308690779e-7/5.594152496e-7m. Coupled motor models
  and compliant-loop controls also pass their independent oracles on this
  source. No source changes or overlapping test jobs were introduced.
- Original harden completed with exit0 at21:48:41+02:4774 passing tests,
  zero failures,2439 ignored across937 summaries. This comprises2396 passes
  in no-default tests/doctests and2378 passes in all-target/all-feature tests.
  All464 Criterion benchmark smoke Success lines are present; unchanged chain128,
  million-body XPBD and public primitive CPU/GPU smoke workloads completed.
  All1482 source hashes verify afterward. The unchanged original full GPU
  aggregate began automatically at21:48:41+02 and is now live on the same runner.
- At20:44+02 harden remains live in its all-target/all-feature aggregate:
  cumulative3124 passed,zero failed,1535 ignored across635 summaries.
  Runner664386/start_ticks2001673 is verified live; full GPU remains queued.
  quaternion-boundary-progress-source-2044.log verifies all1482 source paths;
  git diff --check passes. This continuation changed only the return report,
  correcting superseded instructions and recording the public API audit.
- At20:31+02 harden has completed its no-default-features test phase:
  2396 passed,zero failed,1218 ignored across474 summaries,including doctests.
  Strict Clippy passed afterward; all-target/all-feature compilation completed
  in11m02s and its test/benchmark aggregate is now running. The live runner is
  still664386/start_ticks2001673. Cumulative harden counts are2461 passing
  tests,zero failures; harden is not yet terminal and full GPU remains queued.
- Public API/return-package audit inspected current declarations,validation,
  settings wake/stamp behavior,borrowed readouts,explicit repair and GPU
  construction. All twelve CPU/GPU example invocations are present in the
  current terminal-pass log. Section7 records this bounded audit evidence.
- At20:10+02 harden remains live in its no-default-features test phase:
  376 completed summaries,1711 passed,zero failed,1021 ignored; current
  distance-contract scale sweep is executing. Runner664386 remains live with
  start_ticks2001673. Harden has not yet reached its all-features aggregate;
  original full GPU gate stays queued. No source edits occurred during these gates.
- Original workspace test-all passes0 at19:40:21+02 on current source:
  2364 passed,zero failed,1212 ignored across462 summaries,plus464 native
  Criterion Success lines. Full original workloads were retained,including
  GPU chain128 and million-body XPBD smoke. All1482 source hashes verify
  afterward. quaternion-boundary-acceptance-test-all.log/.exit are authoritative.
  Original harden began19:40:21+02 and remains live; its no-default compilation
  is progressing. Full GPU aggregate is still queued,not accepted.
- At19:38+02 original test-all remains live with no observed failure/panic.
  The original Featherstone chain benchmark completed GPU2/8/32/128 without
  reducing its workload; its old process857230 is terminal. Native benchmark
  smoke continued through XPBD million-body cases and synchronized Featherstone
  chain/batch/mixed sweeps. The molla-vehicle library's remaining real-tire/FEM CPU
  tests are executing. Runner664386 still matches start_ticks2001673; current
  source guard passes in quaternion-boundary-progress-source-1938.log. Completed
  summary count is460,with2012 passes,zero failures and1200 ignored.
- At19:12+02 original test-all remains live:456 completed summaries,2009
  passes,zero failures,1200 ignored. Native Criterion benches are now executing;
  featherstone_chain CPU2/8/32/128 and GPU2/8 report Success. The owned
  bench_featherstone_chain process857230 is active at~100% CPU and~7.6GiB RSS
  while preparing the next GPU variant. Raw kernel specialization follows the
  largest articulation size (including32/128 here); this is not the32-joint
  live-runtime admission limit. Do not treat slow shader compilation as terminal
  or shrink the original benchmark workload to force a pass.
- At18:58+02 the original workspace test-all gate is live,not a completed pass:
  318 completed summaries,1451 passes,zero failures,881 ignored; currently in
  implicit_mpm CPU cases. Runner664386 still matches start_ticks2001673.
  All1482 source hashes verify in quaternion-boundary-progress-source-1858.log.
  Original harden and unchanged full GPU aggregate remain queued behind it.
- Current-source public examples pass0 at18:22:12+02:all six cases on CPU
  and actual RTX4080 GPU. Original CPU primitive aggregate passes0 at18:25:57+02;
  586 passed,zero failed,10 ignored across90 summaries.
  example and CPU source-after guards both pass. Runner664386 is now running
  the original test-all gate with TEST_ALL_ARGS=--no-fail-fast,then harden and
  unchanged full GPU aggregate. These full gates remain unaccepted until terminal.
- At18:36+02 original test-all is still compiling/linking,not hung:runner664386
  owns Cargo678860 and active Nix1.94.1 rustc/rust-lld children. No completed
  test summaries yet; memory reports53GiB available and disk1.2TiB free.
  No external processes were stopped or competing test runners launched.
- Final previously unreached production stage passes0 at18:19:36+02 in
  quaternion-boundary-acceptance-remaining-production.log. Loop rows,quaternion
  FK,mass inverse,live compliant-loop resource retention,activation and program
  cache all pass (13 tests,six summaries); source-after guard passes. Static/check-all/strict Clippy/
  ten shaders passed0 at18:17:48+02. Runner664386 is live in examples at18:20+02.
  This segmented coverage does not replace the unchanged original full-GPU gate.
- Current detached runner:quaternion-boundary-acceptance.sh,PID664386,
  start_ticks2001673,started18:17:41+02. All1482 source hashes verify after
  the quaternion oracle correction. Last four production lanes precede
  examples,CPU primitives,original native/hardening and full GPU aggregate.
- Corrected quaternion FK file passes2/2 in0.94s,exit0. Across156 joints and18
  articulations,max component errors are1.354472090e-14 (small angles) and
  4.397774853e-10 (wide/large angles),below unchanged2e-9/3e-7 gates. Prescribed
  and zero-coordinate poses satisfy the strengthened normalization checks.
  quaternion-boundary-gpu.log/.exit preserve evidence; session19847 is terminal0.
- impact-capacity runner586322 ended naturally2 at18:10:36+02. Current-source
  stage24 passes/two failures,nine summaries. Multidof limits3/3 in349.36s,
  D6 axis certification1/1 in123.76s,and compliant loop rows1/1 in0.42s pass.
  D6 certification covers144 cases,240 frames,two substeps,masses1/17/83 and
  full COM/tensor; peak pose2.703058440e-6,rate4.017618297e-5. Two quaternion FK
  fixtures failed stale boundary normalization oracles. Source-after guard
  passes. Corrected whole quaternion GPU file is running in session19847;
  quaternion-boundary-source-sha256.txt changes only that test file.
- At18:05+02 runner586322 remains live; restitution6/6 passes in900.78s,
  batched dispatch metrics1/1 in105.70s,and multidof feedback1/1 in146.01s.
  Restitution includes0/0.5/1 rebound/one-impulse oracles,250-frame penalty
  support,oblique energy/slip,rollback,impact-induced break and reaction-capture
  invariance. Dispatch counts remain independent of1 versus32 bodies at1/4
  substeps for Fixed/Free/D6. Multidof limits are now executing; no new failure.
- impact-capacity runner586322 remains live with matching start_ticks1732786
  at17:44+02. Static/check-all/strict Clippy/shaders pass0 at17:33:10+02.
  Whole impact-position and impact/friction-partition suites pass8/8 and3/3;
  independent articulated position-displacement peak error4.869755828e-9.
  Public runtime_restitution_gpu (six tests) is now executing. All1482 current
  source hashes verify in impact-capacity-progress-source-1744.log.
- Previous detached runner:impact-capacity-acceptance.sh,PID586322,
  start_ticks1732786,started17:32:52+02. It resumes unvisited impact-position/
  partition,restitution,dispatch,multidof,loop resources,activation/cache gates;
  then examples,CPU primitives,original native/hardening and full GPU aggregate.
- Separating-impact capacity fix passes all selected actual GPU suites,exit0:
  impact response7/7,radix8/8,contact islands7/7,and impact constraints11/11
  (33 tests total). Original3/130-body forward/reversed chain now passes;
  all1482 source hashes and original impact fixture hash verify afterward.
  impact-capacity-gpu.log/.exit retain the evidence. No fixture,tolerance,
  graph-connectivity or non-impact capacity policy change was needed.
- relative-momentum-acceptance ended naturally2 at17:27:15+02. Whole contact
  breakage passes5/5 in930.39s; impact episodes7/7 in1.62s; impact constraints
  11/11 in2.68s. Impact response6/7 in3.80s fails only separating-chain capacity.
  Stage29 passes/one failure,four summaries; source-after guard passes.
  impact-capacity-source-sha256.txt adds only contact_islands.wgsl to the
  preceding manifest. Unchanged impact-response/island/impact-constraint suites
  are running through Make in the focused impact-capacity-gpu command.
- Previous detached runner:relative-momentum-acceptance.sh,PID556997,
  start_ticks1602164,started17:11:06+02. All1482 source hashes verify after
  focused GPU acceptance. It runs static/strict Clippy/shaders,whole contact
  breakage and remaining impact/resource lanes,examples,CPU primitives,
  original test-all/harden,and original full GPU aggregate,serially.
  Static/check-all/strict Clippy/ten composed shaders pass0 at17:11:31+02;
  remaining production is live from the whole contact-breakage GPU file.
- Relative-COM momentum correction passes the strengthened actual GPU
  detachment regression,1/1 in289.38s,exit0. Original four-substep case plus
  one/eight-substep refinements all pass80 frames. Maximum analytic errors are
  1.758593271e-13/9.769962617e-14/8.792966355e-13m; parity maxima are
  1.421085472e-13/8.526512829e-14/8.739675650e-13m. Exact boundary continuity,
  unaffected-root poses,velocities,rotations and single event checks pass.
  relative-momentum-contact-break-gpu.log preserves all values. The prior
  5.960464722e-8m one-substep error is removed without changing its oracle.
- Rounded-operands46447 failed the unchanged analytic oracle; raw5738 and
  stage42040 diagnostics confirmed the post-drift momentum transport source.
  relative-momentum-source-sha256.txt now records only rigid_momentum.wgsl and
  runtime_contact_breakage_gpu.rs changed from fixed-reaction source. Ten
  composed shaders validate; actual refined GPU regression runs in session34829.
- Refinement85656 and detailed diagnostic38722 ended2 on the one-substep
  independent analytic oracle (123.38s and110.70s). Candidate rounded-operand
  shader validation is running in session46447,rounded-operands-contact-break-gpu.log.
  rounded-operands-source-sha256.txt records the current source; delta contains
  the GPU fixture and featherstone.wgsl only. Do not use the preceding test-only
  manifest as current source identity.
- Targeted diagnostic79451 reproduced the failure,exit2 in221.52s; temporary
  diagnostic prints were removed. Revised refinement/oracle fixture is now
  running in session85656. contact-break-precision-source-sha256.txt and its
  delta record the sole change:runtime_contact_breakage_gpu.rs. Production
  sources are unchanged from the preceding passing component gates.
- fixed-reaction-acceptance ended naturally2 at16:34:53+02,not a timeout.
  Public contact breakage4 passed/one failed in932.58s; source-after guard
  passes. Capacity rollback,collision exclusions,sequential event timing and
  loaded break continuation pass. Exact successful detachment parity remains
  open; targeted diagnostic is running in session79451.
- Corrected private contact reaction GPU suite passes2/2 in246.48s,exit0;
  fixed-reaction-gpu.log includes impact-history recovery and exact immovable
  pose/zero-velocity checks. All1482 source hashes verify afterward. The latest
  user progress percentage was a rough estimate,not a checklist measurement.
- empty-dependency-acceptance ended naturally2 at16:09:34+02. Public contact
  reaction file passes5/5,740.33s; private impact-history recovery passes,but
  fixed-ground final pose assertion fails due to its obsolete single-f32 oracle.
  Stage total6 passes,one failure,two summaries; source-after guard passes.
  Preserved fixture SHA23a78ad86c84ff0d66328588399ac990c39f510e8c175f11a8220ba0549d0b76
  in fixed-reaction-original-fixture.rs. Corrected private two-test GPU suite
  passed via Nix/nixVulkan Make; fixed-reaction-source-sha256.txt
  and delta prove only rigid_contact_reaction_tests.rs changed.
- Previous detached runner:empty-dependency-acceptance.sh,PID445385,
  start_ticks1132487,started15:52:49+02. Static/strict Clippy/ten composed shaders
  pass0 at15:53:10+02; ended2 on the stale private reaction pose oracle.
- Previous detached runner:fixed-reaction-acceptance.sh; source-guarded gates
  resume with contact breakage,then impact,restitution/resources,examples,
  native/harden and original full GPU aggregate. PID/start ticks are recorded
  separately and must be checked against live processes before polling.
- Runner473061,start_ticks1290330,started16:19:08+02; static/check-all/strict
  Clippy/ten composed shaders pass0 at16:19:17+02. Remaining production began
  with the original five-test public contact-breakage GPU file.
- bounded-island-acceptance ended naturally2 at15:47:42+02:device response9/9,
  publication2/2,public runtime contacts5 passes/one empty-dependency binding
  failure. Original public holding and kinetic checks passed before the failure:
  CPU position errors3.367104648e-7m and1.407441567e-5m respectively; kinetic
  acceleration3.999957815m/s2 versus analytic4.0m/s2. Source-after guard passed.
- Applied the8-byte dependency allocation fix in rigid_contact_activation_gpu.rs;
  actual GPU activation suite passes6/6,0.32s; unchanged exact public static-only
  case passes1/1,122.58s,exit0 in session66610. Zero-body dispatch,substep changes,
  dynamic-body insertion and later disabling now succeed without binding errors.
  Source manifest:empty-dependency-source-sha256.txt;
  delta lists only that production file and its component test file.
- Current contact-response device suite passes9/9,1495.44s; publication passes2/2,
  0.49s. Loaded10-second holding drift2.277088606e-5m,post-settle creep
  5.585206989e-7m. Kinetic acceleration2.981366962m/s2 versus analytic
  2.981347547m/s2. Stop/re-stick reversal-window displacement7.817521691e-6m.
  Breakaway refinements at1/2/4/8 substeps agree on11.815N trigger load.
  These are device trajectory components,not substitutes for public-runtime
  tests. The public runtime_contacts_gpu file is live in PID422394 under runner
  379026 at15:37+02; no fresh failure. Observed source guard passes0.
- Historical runner:bounded-island-acceptance.sh,PID379026,start_ticks867631,
  started15:08:41+02. Fresh static/check-all/strict Clippy/ten composed shaders
  pass0 at15:08:47+02. Remaining GPU production is live from contact response/
  publication onward; examples,native/harden and original full GPU remain queued.
- authored-frame-acceptance ended naturally2 at15:05:21+02:54 passes and one
  stale bounded-row rejection failure across six summaries. Completed scopes:
  Coulomb disk7/7,coupled constraints15/15,coupled response9/9,mixed live-runtime
  contacts9/9,radix8/8. Source guards passed. Corrected whole contact-island file
  now passes7/7,4.62s,exit0 in bounded-island-gpu.log/.exit. Preserved original
  fixture SHA f93cea034e74c7f9c7d73d89a66e3265c33985e3ab6484abb63cf1dbc7c2cecf
  in bounded-island-original-fixture.rs. bounded-island-source-delta.diff proves
  exactly that test file changed; completed unrelated scopes remain equivalent.
- Historical detached runner:authored-frame-acceptance.sh,PID336014,start_ticks734321,
  started14:46:28+02. Fresh static/check-all/strict Clippy/ten composed shaders
  pass0 at14:46:37+02. Remaining production lanes are running from friction/
  coupled contacts onward; accepted geometry and CPU fixtures are not repeated
  before unvisited work. Examples,native/harden and original full GPU aggregate
  remain queued. The only latest source change is the drift oracle correction.
- Corrected authored-frame drift GPU file passes5/5,exit0,609.35s at14:45:23+02.
  Maximum relative FK error8.470329472543003e-16 is below the unchanged2e-13
  gate across all original scales; engine source,fixture geometry and workload
  are unchanged. Evidence:authored-frame-drift-gpu.log/.exit. The prior48
  production passes remain source-equivalent for their unchanged files.
- paired-position-acceptance ended naturally2 at14:32:43+02. Nearby breakage/
  kinematics passed20/20 across five files,exit0 at14:19:03+02. Production then
  passed48 tests and failed one stale fixed-frame drift oracle across nine files.
  All source guards passed. Preserved original fixture SHA
  d5bef58bfaf410de4903febb558f563bf579f91bd839b8fb60c119c547c76009 in
  authored-frame-original-fixture.rs. Corrected whole drift GPU file passed
  in session88970; authored-frame-source-delta.diff proves exactly one test-file
  change since paired-position source. Engine and other fixtures are unchanged.
- New detached runner paired-position-acceptance.sh started13:47:50+02,
  PID200228,start_ticks382505. Static/strict Clippy/ten composed shaders pass0
  at13:48:45+02; original CPU aggregate passes0 at13:52:42+02:90 summaries,
  586 passes,0 failures,10 ignored. Source-after guards pass. Nearby GPU
  breakage/kinematics is running; unvisited contact/impact lanes,examples,
  native/harden and unchanged full GPU aggregate follow serially.
- At14:14+02 the current nearby GPU stage has18 passes,0 failures,0 ignored
  across four whole files:detachment4/4,joint breakage8/8,kinematic motion2/2,
  kinematic reactions4/4. The final runtime_kinematic_constraints file is running
  in PID260035 under the same live runner. Observed source guard passes0.
  Kinematic prediction/redrift peak6.091392592e-6; largest prescribed support
  force error6.451337885e-7N and torque error2.799515382e-6Nm.
- Paired-position correction passes all five original GPU precision tests,exit0,
  870.25s at13:46:42+02. Exact pose equality after COM mutation and break detachment
  is restored,including failed-step rollback. Final detached travel is
  6.000001540087396e-4m versus6e-4m; kinematic retained travel is
  6.000001030770363e-4m. Ten composed shader modules also validate; CPU mapping
  oracles pass8/8. paired-position-pose-source-after.log confirms frozen source.
  This completes the focused bug fix,not full P0–P6 acceptance.
- compliant-repair-detached ended naturally with exit2 at13:26:07+02:24 passes,
  two genuine pose-precision failures across seven summaries. The unchanged
  fixture failed after COM edit at frame53 (1025.0005817190176 versus
  1025.0005400001355),and after break at frame17 (1025.0002312631032 versus
  1025.0001800000427). Source-after guards passed; no timeout or skipped test.
- Fixed the shared residue-transfer calculation; focused CPU precision tests
  pass8/8,including nonrepresentable-coordinate COM and detachment regression
  oracles. Original GPU fixture and exact-equality assertions are unchanged,
  SHA45126eda682b42774b27cdbd00ad11a47ba890b80038b380328ede48964a5673.
  published-residue-source-delta.diff proves only rigid_contact_precision_gpu.rs
  changed since the prior manifest. First GPU rerun is terminal2 at13:29:51+02:
  four passes,one3.64e-12m post-detachment equality failure; both contact/no-contact
  COM mutation routes now pass. Raw evidence:published-residue-pose-precision-gpu.log.
  The follow-up compensated-position shader change passes in session15481:
  Make composed-shader gate plus the unchanged five-test GPU precision file.
  Current source:paired-position-source-sha256.txt; only rigid_contact_precision_gpu.rs
  and featherstone.wgsl differ from the pre-failure manifest.
- Started source-guarded compliant-repair-acceptance.sh in session3451. Fresh
  static/strict Clippy/composed-shader gate passes0 at11:46:33+02; original CPU
  aggregate passes0 at11:50:29+02:90 summaries,586 passes,0 failures,10 ignored.
  All six examples pass on CPU and actual RTX4080 GPU,exit0 at11:53:25+02.
  Original test-all was interrupted during Featherstone benchmark smoke after
  456 completed test summaries,2008 passes,0 failures,1199 ignored. Its final
  exit file is empty; this is NOT a pass or a natural assertion failure.
  compliant-repair-interrupted-source-after.log verifies unchanged source.
- Launched compliant-repair-detached-acceptance.sh at13:10:30+02 under nohup/setsid.
  Former runner PID17541,start_ticks158534 is now terminal2. GPU detachment
  passes4/4 with resource-counters,joint breakage8/8,kinematic motion2/2 and
  kinematic reactions4/4; the later pose-precision failures stop this runner.
  No external process was signalled; no completed static/CPU/example gate repeated.
- paired-rates-next ended naturally with exit2 at11:34:31+02. Its production
  stage has17 summaries,82 passes,one stale-assertion failure,0 ignored; combined
  with its baseline stage this is105 passes/one failure,not full acceptance.
- Preserved the original failing fixture as broken-loop-repair-original-fixture.rs,
  hash2980934eccb112348456a604ceb7c3bbd12e4ad05a58a8575c78b673e4eb4eae.
  That correction changed only the test file; subsequent precision fixes are
  separately recorded above and in their source manifests.
- Corrected CPU detachment file passes4/4,including the new64-frame,four-phase
  independent compliance oracle. Maximum errors:position8.673617380e-19m,
  velocity1.110223025e-16m/s,reaction5.684341886e-14N. Full GPU detachment file
  now passes4/4,0 failures,exit0 in241.84s. GPU oracle maximum errors:
  position7.819078877e-9m,velocity2.828209286e-7m/s,reaction8.241129979e-5N.
  Evidence:compliant-repair-detachment-gpu.log and matching.exit. This is a
  focused pass,not full acceptance or the resource-counters configuration.
- Progress answer:approximately85% overall,a subjective estimate rather than
  a measured checklist ratio. Main implementation is present; full validation
  and any fixes it exposes remain required.
- Prior static/strict Clippy/shader gate passes0; prior original CPU aggregate
  passes585 tests across90 summaries,0 failures,10 ignored,exit0. The test-only
  correction adds one CPU case; the later586-pass aggregate is recorded above.
- Current nearby/unreached GPU baseline stage passes23 tests in eight files,
  0 failures,0 ignored,exit0 at10:18:41+02. This includes public hinge recovery,
  mixed-distance regression,scalar/Ball/redundant bounds,hot-edit readout,
  articulated response and density-derived live mass response. This is a
  completed component stage,not the unchanged original full GPU aggregate.
- Public runtime,distance,cone,Ball-loop and reaction files passed before the
  stale detachment assertion. Full GPU/native/hardening acceptance is unfinished.
- Archived prior report without discarding raw failure/diagnostic history.
- Verified all56 performance rows have43 fields and unique family/count/backend
  keys; all referenced source manifests/raw logs exist.55 rows are measured
  snapshots; one is an explicit checked GPU capacity error.

### Next Steps
- Observe low-load-full-gpu-acceptance runner2709083/start_ticks7808502.
  Verified live at14:28+02,155 passes across45 completed summaries,zero failures;
  runtime_cone_limits is executing. The timing file passes unchanged; final
  original full GPU aggregate is live. Preserve its source guards and terminal status,then refresh the dirty
  inventory and final report. Avoid overlapping test jobs or stopping external
  processes. The goal is active again,not blocked or complete.
- Runner664386 is terminal2; runner2431847 is terminal0; diagnostic50750 is
  terminal2. No current test process remains. Do not poll them as live.
- Preserve both earlier failed timing attempts alongside the low-load pass;
  source-identical segmented passes still do not replace the current aggregate.
- Current-source remaining-production lanes pass13/13; all six CPU and all
  six GPU examples pass; CPU primitives pass586 tests; original workspace
  test-all passes2364 tests; harden passes4774 tests across both configurations.
  None substitutes for the pending full GPU aggregate.
- Audit each original requirement against final source and physical evidence;
  queued commands,ignored tests and compile success are not acceptance.
- Refresh the final expanded dirty-path inventory and consolidate terminal
  evidence only after the original gates finish and the audit is satisfied.

### Relevant Files
- MOLLA_PRIMITIVES.md — unchanged original scope and acceptance checklist.
- MOLLA_PRIMITIVES_HANDOFF.md — actual Molla APIs and proposed Gearbox signatures.
- MOLLA_PRIMITIVES_PERFORMANCE.tsv —56 source-labelled performance attempts.
- benchmarks/primitives/engine-program-cache-2026-10-01/implementation-history.md — complete archived chronology.
- benchmarks/primitives/engine-program-cache-2026-10-01/quaternion-boundary-acceptance.sh — serialized current gates.
- crates/molla-solvers/src/rigid_contact_activation_gpu.rs — complete empty vec2 binding allocation.
- crates/molla-solvers/src/rigid_contact_precision_gpu.rs — published-coordinate residue transfer fix.
- crates/molla-solvers/shaders/rigid_momentum.wgsl — relative-COM momentum shift without cancellation.
- crates/molla-solvers/shaders/contact_islands.wgsl — selected impact rows alone consume solve capacity.
- crates/molla-solvers/tests/runtime_contact_breakage_gpu.rs — independent analytic detachment regression/refinement.
- crates/molla-kernels/shaders/featherstone_mixed_stops.wgsl — latest physical fix.

## 1. Revision,environment and provenance

| Item | Recorded state |
|---|---|
| Checkout | /home/bresilla/data/code/OUSD/molla |
| Branch/base | develop,da495a0971c666c7230dd49e0b2a48d0307fc7a4 |
| Worktrees | One existing develop worktree; no new branch/worktree/commit |
| Native toolchain | Nix Cargo/rustc1.94.1 |
| Actual adapter | NVIDIA GeForce RTX4080,Vulkan,driver595.91.07; hardware assertions enabled |
| Source identity | quaternion-boundary-source-sha256.txt,1482 source paths |
| Acceptance-audit dirty snapshot2026-10-03 06:58+02 | 3315 expanded paths:248 modified,3066 untracked,one pre-existing deletion |
| Scope | Molla only; unrelated existing changes preserved |

Artifact paths below are relative to
`benchmarks/primitives/engine-program-cache-2026-10-01/` unless another path is
given. `paired-rates-next-final-observed-source.log` records source verification0
after the accepted hinge/distance/precision files. The manifest covers source,
fixtures,Makefile and dependency inputs; changing this report does not change
the recorded physical source. Before/after guards surround every runner stage.
`compliant-repair-source-delta.diff` proves the sole change since the paired-rate
manifest is runtime_break_detachment.rs; production source,other fixtures,
Makefile and dependency inputs are unchanged.

The subsequent paired-position-source-delta.diff records exactly two changed
production paths:rigid_contact_precision_gpu.rs and featherstone.wgsl. Older
aggregate results below are labelled historical and are not promoted to current
shader certification. The five-test precision file is accepted on this source.

The historical05:44+02 dirty snapshot contains2953 expanded paths:248 modified,
2704 untracked,one pre-existing deletion of
`crates/molla-geometry/shaders/lbvh.wgsl`. It is preserved in
`cloud-native-mode-final-dirty-files.txt`; it is not a fresh final path count.
`paired-rates-next-dirty-files.txt` contains the newer10:21:53+02 progress snapshot.
`acceptance-audit-2026-10-03-dirty-files.txt` is the current expanded inventory.
Refresh it after any further changes; do not use a grouped-directory status
count or silently attribute all pre-existing changes to this task.

Baseline environment/commands/failures are preserved under
`benchmarks/primitives/baseline-2026-09-28/`. Historical evidence is not silently
promoted to current-source certification.

## 2. Public API and runnable examples

Authoring uses `molla_sim::runtime::RigidScene`,opaque handles and fallible
mutations. Physical values use metres,kilograms,seconds,radians,N and Nm.

| Capability | Public surface |
|---|---|
| Radial limits | DistanceLimits { min: Option<Real>,max: Option<Real> }; set_distance_limits / distance_limits |
| Ball cones | SphericalConeLimits { angle0: Option<Real>,angle1: Option<Real> }; set_cone_limits / cone_limits |
| Break settings | JointBreakLimits { force: Option<Real>,torque: Option<Real> }; set_joint_break_limits / joint_break_limits |
| Break state | joint_broken / repair_joint |
| Explicit friction | FrictionCoefficients { mu_static,mu_dynamic }; ContactMaterial.friction: Option<FrictionCoefficients> |
| CPU world | RigidWorld::from_scene(scene) |
| GPU world | RigidWorld::from_scene_gpu(scene,Arc<Engine>) |
| Solved readout | set_reaction_readout,joint_reaction,joint_reaction_samples,distance_reading,cone_reading |
| Events | joint_break_events / drain_joint_break_events |

Scene settings getters return `Result`; solved world readouts return `Option`
because disabled,unsolved or stale data is not a fabricated zero sample.

The complete compiling minimal public-runtime examples are in
[runtime_primitives.rs](crates/molla-solvers/examples/runtime_primitives.rs).
Its `distance`,`cone`,`reaction`,`breaking`,`friction` and `d6` functions create
scenes,bodies,joints/materials and execute through the selected world backend.

```sh
export NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
nix develop --impure -c make -j1 example-primitives A=all
nix develop --impure -c make -j1 example-primitives-gpu A=all
```

These authoring fragments use a valid scene/joint of the stated kind; the linked
example supplies complete imports,body insertion,world construction and stepping:

```rust,ignore
// Distance joint.
scene.set_distance_limits(joint, DistanceLimits {
    min: Some(1.0), max: Some(1.0),
})?;

// Ball joint.
scene.set_cone_limits(joint, SphericalConeLimits {
    angle0: Some(0.25), angle1: Some(0.4),
})?;

// Any supported breakable joint.
scene.set_joint_break_limits(joint, JointBreakLimits {
    force: Some(5.0), torque: None,
})?;

// Collider contact material.
collider.contact_material.friction = Some(FrictionCoefficients {
    mu_static: 0.8, mu_dynamic: 0.4,
});

let mut world = RigidWorld::from_scene(scene)?;
world.set_reaction_readout(true)?;
world.step(1.0 / 240.0)?;
let reaction = world.joint_reaction(joint);
let events = world.drain_joint_break_events();
```

The snippets are additive authoring examples,not a single scene with mutually
incompatible joint kinds. GPU selection in the executable requires a hardware
adapter and errors when unavailable; it never substitutes CPU execution.

## 3. Execution routes and support boundaries

CPU: scene authoring/compiler -> compiled Model/State -> Featherstone and
runtime coupled constraint/contact response -> reaction recovery and substep
break decisions -> boundary-safe topology changes -> scene publication.

GPU: authoring/compiler -> model/control upload on rebuild -> retained
Featherstone ping-pong state -> device distance/cone/loop/contact response ->
device reaction accumulation and break reduction -> compact notification ->
boundary-safe host topology rebuild -> checked frame readout -> scene commit.

Host validation,sleep selection,topology orchestration and frame publication
are explicit. They are not a whole-scene per-step CPU physical solve.

| Capability | CPU live runtime | GPU live runtime | Certification scope |
|---|---|---|---|
| Distance,optional bounds,world/body/external endpoints | Implemented | Device response/readout | Distance runtime,contract,sleep,COM and10-second gates pass; mixed-contact file9/9 on authored-frame source; full fresh aggregate pending |
| Ball anchors,elliptical swing/free twist | Implemented | Device anchor/cone rows | Cone geometry/capability/limits and rigid Ball-loop10-second gates pass; mixed-contact file9/9 on authored-frame source; full fresh aggregate pending |
| Fixed/Revolute/Prismatic/Ball/D6 external rigid/compliant links | Implemented | Retained device response | Current full GPU baseline loop/drive files pass |
| Tree/loop reactions and automatic break | Implemented | Device wrench/peak/threshold solve; host topology boundary | Current reaction/break files pass; contact breakage5/5 in source-identical remainder |
| Static/dynamic contacts and restitution | Implemented | Resident contacts/material/impact response | Current static10s/three-rate,kinetic/reversal/contact lifecycle and restitution6/6 pass |
| D6 axes,motors,frames,COM/full tensor | Implemented | Device coupled bounds/drives | Current full baseline and144-case axis certification pass |
| Density aggregation/live mass edits | Implemented | Host validation/recompile,device force response | Current CPU1/1 and GPU1/1 pass |
| Edits,stable handles,sleep,pair permissions and faults | Implemented | Atomic frame publication/retained buffers | Current CPU aggregate,public GPU contract12/12,contact sleep7/7 and resident filtering/lifecycle pass |
| Primitive/hull/mesh authoring | Explicit CPU representations | Packed primitive contact kinds0–6 | GPU unsupported admitted pairs error,not CPU proxies |
| Registered wheel/track/actuator stepping | Legacy CPU path; automatic break with devices rejected | Explicitly rejected | Not certified by this package |
| Runtime Cable | Explicitly unsupported | Explicitly unsupported | Lower-level Cable/FEM is separate |

GPU articulations above32 joints are rejected. Unknown motor models,unsupported
external kinds and unsupported coupled bounds return errors. Supported scalar
bound kinds include Revolute,Prismatic,D6 and Ball. Capacity errors abort rather
than truncate; unsupported or unrepresentable positive f32 parameters are not
silently dropped.

Resident GPU narrowphase supports sphere,plane,box,capsule,cylinder,cone and
ellipsoid. Rounded-cylinder,hull/convex-parts,triangle-mesh and heightfield
authoring is not a GPU collision substitute. Unsupported packed shapes use
sentinel0xff and an admitted pair faults; disabled/excluded/nonoverlapping
pairs need not generate a fault. Separate legacy geometry fallback behavior
does not certify this production lane.

Mass computation uses analytic primitive/mesh integration and additive convex
pieces. Heightfields have zero volume; rounded-cylinder mass uses the documented
inner-cylinder approximation,not a claim of exact rounded volume.

Immutable program initialization scopes Validation,OutOfMemory and Internal
errors before cache publication. Models/state remain world-owned. Checked
capacity recovery is tested separately from actual driver OOM; a genuine late
native-OOM recovery is not claimed by the existing capacity fixture.

## 4. Conventions and lifecycle

- Distance constrains authored anchors,not origins/COM. Optional bounds are
  independent; equal bounds are bilateral; zero equality does not weld rotation.
  Bounds are finite,nonnegative and ordered. Radial internal slots are not public
  LinX semantics. Coincident-anchor limiting response is deterministic.
- Cones use geodesic swing in the cyclic transverse basis of the authored primary
  axis,not independent Euler clamps. Extents are independent radians in[0,pi].
  Symmetric extents reduce to primary-direction angular separation; authored
  frames,quaternion sign and free twist are preserved. The exact representation
  and Jacobian are documented in [MOLLA_CONE_GEOMETRY.md](MOLLA_CONE_GEOMETRY.md).
- Reactions are transmitted child/parent world-space force and torque about their
  respective authored anchors. Reference-point shifts recover COM torques.
  Samples use N/Nm,not impulses; sample duration,substep peaks and frame means
  are distinct. Fixed support is not inferred from commanded motor effort.
- None or positive infinity disables a break threshold. Negative/NaN values error.
  Comparison is inclusive; a valid unloaded zero sample can reach a zero threshold.
  Device decisions use physical substep samples. One event is emitted per episode;
  detachment preserves pose/velocity; draining events does not repair the joint.
  Repair is explicit. Explicit user pair exclusions survive removal of joint policy.
- Friction is a coupled stick/slip impulse solve using articulated tangential
  effective mass. It does not use a persistent tangent-displacement spring.
  Impact episode state is separate. Static response stays inside the static disk;
  sliding uses the dynamic disk and relative surface velocity,not rolling speed.
- Authoring requires finite0<=mu_dynamic<=mu_static; invalid values error rather
  than clamp. Explicit pairs override legacy scalar friction; None maps the scalar
  to equal coefficients. Combine priority is Average<Min<Multiply<Max,applied
  independently to both coefficients. Specialized tyre/track laws remain separate.
- Hot dynamics edits invalidate solved readings even when compilation stamps do
  not change. Disabled,stale or unsolved readouts remain None. Rejected mutations
  are atomic; stable handles and unrelated state survive valid compilation.
- Legacy mass_matrix/inverse_dynamics locked-column analysis differs from physical
  response_mass_matrix and transmitted-wrench APIs. Zero analysis effort is not
  proof of zero support load.

## 5. Source-labelled acceptance evidence

All commands use Make under Nix. The current runner is
`quaternion-boundary-acceptance.sh`; CARGO_BUILD_JOBS=6. Each stage preserves raw
output,terminal exit and source guards. Timeout124 is not a pass; manual stops
in historical logs are labelled separately from deadline expiration.

| Check | Result | Raw evidence |
|---|---|---|
| Low-load unchanged full LBVH file | PASS8/8,exit0;100k22.022ms,10k3.434ms;GPU load11% before/4% after;source guards pass | lbvh-low-load-gpu.log/.exit;lbvh-low-load-environment-{before,after}.log |
| Final unchanged full GPU aggregate after low-load timing pass | LIVE at2026-10-03 14:28+02;155 passes,zero failed/ignored,45 summaries;no lane overrides | low-load-full-gpu-acceptance.log;low-load-full-gpu-acceptance-runner.log |
| Current original harden | PASS0 at21:48:41+02;937 summaries,4774 passes,zero failures,2439 ignored;464 benchmark Success lines;source-after passes | quaternion-boundary-acceptance-harden.log/.exit |
| Current unchanged full GPU aggregate | TERMINAL2 at2026-10-03 04:22:06+02;256 passed,one LBVH timing failure,zero ignored,66 summaries;source-after passes | quaternion-boundary-acceptance-original-full-gpu.log/.exit |
| Unchanged remaining GPU selection | PASS0 at2026-10-03 06:56:14+02;156 passes,31 summaries,zero failed/ignored;all source-after guards pass | lbvh-contention-remaining-{barrier,normal,drift,contact-tail,later-lanes}.log/.exit |
| Timing diagnostic,unchanged fixture | FAIL32.132ms versus30ms;external GPU load94–99%;not isolated | lbvh-contention-diagnostic.log/.exit;lbvh-contention-environment-{before,after}.log |
| Current full GPU baseline/public runtime contract | PASS37 baseline files and12/12 public contract cases;no aggregate failure | Same original-full-gpu log |
| Current distance runtime/sleep/contract/COM/scale | PASS10/3/4/4/4;10s at60/120/240Hz and40-case/rate scale sweep;peak anchor4.367491819e-5m,short parity8.907788264e-7m | Same original-full-gpu log |
| Current loaded static friction10s,three step sizes | PASS1/1,897.76s;four cases each60/120/240Hz,eight substeps;peak drift3.607930302e-5m versus1e-3m gate | Same original-full-gpu log |
| Current coupled motor feedback | PASS1/1,109.22s;72 cases each stable=false/true;peak coordinate6.643140809e-7,rate3.710364803e-6,reaction7.035145036e-5 | Same original-full-gpu log |
| Current compliant-loop lifecycle/controls | PASS3/3 and4/4;break/repair,independent coupled/brake recurrence,60/120/240Hz,one/four substeps | Same original-full-gpu log |
| Current original workspace test-all | PASS0 at19:40:21+02;462 summaries,2364 passes,zero failures,1212 ignored;464 benchmark Success lines;source-after passes | quaternion-boundary-acceptance-test-all.log/.exit |
| Current six CPU and six GPU examples | PASS0 at18:22:12+02;source-after passes | quaternion-boundary-acceptance-examples.log/.exit |
| Current original CPU primitives | PASS586 tests,90 summaries,zero failures,10 ignored;exit0 at18:25:57+02;source-after passes | quaternion-boundary-acceptance-cpu-primitives.log/.exit |
| Final remaining production GPU lanes | PASS0 at18:19:36+02;loop rows/FK/mass inverse,retained loop resources,activation/cache | quaternion-boundary-acceptance-remaining-production.log/.exit |
| Quaternion-boundary static/check/strict Clippy/shaders | PASS0 at18:17:48+02 | quaternion-boundary-acceptance-static-shaders.log/.exit |
| Quaternion FK normalization oracle | PASS2/2,0.94s;156 joints/18 articulations,unchanged angle tolerances,stronger unit norms | quaternion-boundary-gpu.log/.exit |
| Multidof limits and D6 axis certification | PASS3/3 and1/1;144 axis cases,240 frames,full tensors | impact-capacity-acceptance-remaining-production.log |
| Public runtime restitution | PASS6/6,900.78s;analytic rebound,penalty support,oblique energy,rollback/break/capture | impact-capacity-acceptance-remaining-production.log |
| Batched dispatch metrics | PASS1/1,105.70s;1/32 bodies,1/4 substeps,Fixed/Free/D6 | Same remaining-production log |
| Public multidof feedback | PASS1/1,146.01s;D6/Free/Ball,position/velocity/combined,stable modes | Same remaining-production log |
| Impact-capacity static/check/strict Clippy/shaders | PASS0 at17:33:10+02 | impact-capacity-acceptance-static-shaders.log/.exit |
| Impact-position and impact/friction partition | PASS8/8 and3/3;peak displacement error4.870e-9 | impact-capacity-acceptance-remaining-production.log |
| Separating-impact capacity correction | PASS33/33;impact response7,radix8,islands7,impact constraints11;unchanged fixtures | impact-capacity-gpu.log/.exit |
| Relative-momentum whole contact breakage | PASS5/5,930.39s | relative-momentum-acceptance-remaining-production.log |
| Relative-momentum impact episodes/constraints | PASS7/7 and11/11;subsequent response capacity failure preserved | relative-momentum-acceptance-remaining-production.log |
| Relative-COM momentum regression | PASS1/1,289.38s;80 frames at1/4/8 substeps; worst analytic error8.793e-13m | relative-momentum-contact-break-gpu.log/.exit |
| Relative-momentum static/check/strict Clippy/shaders | PASS0 at17:11:31+02 | relative-momentum-acceptance-static-shaders.log/.exit |
| Empty-dependency acceptance | TERMINAL2;public contact reactions5/5,private suite1/2 due stale pose oracle | empty-dependency-acceptance-runner.log |
| Corrected private contact reactions | PASS2/2,246.48s; source-after guard passes | fixed-reaction-gpu.log/.exit |
| Current bounded-island static/strict Clippy/composed shaders | PASS0; ten modules,device_execution=false | bounded-island-acceptance-static-shaders.log/.exit |
| Bounded-island production GPU lanes | TERMINAL2;16 passes,one static-only binding failure across three summaries | bounded-island-acceptance-remaining-production.log/.exit |
| Empty-dependency binding regression | PASS0:activation6/6 plus unchanged public static-only1/1 | empty-dependency-gpu.log/.exit |
| Device contact-response trajectories | PASS9/9,1495.44s; loaded hold,slide,breakaway refinement,stop/re-stick/reversal and faults | bounded-island-acceptance-remaining-production.log |
| Device contact publication | PASS2/2,0.49s | Same remaining-production log |
| Current authored-frame static/strict Clippy/composed shaders | PASS0; ten modules,device_execution=false | authored-frame-acceptance-static-shaders.log/.exit |
| Authored-frame remaining GPU production lanes | TERMINAL2;54 passes,one stale bounded-row rejection failure | authored-frame-acceptance-remaining-production.log/.exit |
| Corrected contact-island GPU file | PASS7/7,4.62s,exit0; supported and invalid row kinds explicitly exercised | bounded-island-gpu.log/.exit |
| Current paired-position static/strict Clippy/composed shaders | PASS0; ten modules,device_execution=false | paired-position-acceptance-static-shaders.log/.exit |
| Current paired-position original CPU aggregate | PASS0;90 summaries,586 passes,0 failures,10 ignored | paired-position-acceptance-cpu-primitives.log/.exit |
| Paired-position nearby GPU breakage/kinematics | PASS0; five files,20 passes,0 failures,0 ignored | paired-position-acceptance-breakage-kinematics.log/.exit |
| Paired-position production GPU lanes | TERMINAL2; nine files,48 passes,one stale fixed-frame oracle failure | paired-position-acceptance-remaining-production.log/.exit |
| Corrected authored-frame drift GPU file | PASS5/5,609.35s,exit0; max relative FK error8.470329473e-16 versus unchanged2e-13 | authored-frame-drift-gpu.log/.exit |
| fmt-check,check-all,strict workspace/all-target/all-feature clippy-check,check-primitives-shader-wiring | PASS0 on corrected test source; ten composed modules; device_execution=false | compliant-repair-acceptance-static-shaders.log/.exit |
| Original make test-primitives | PASS0 on corrected test source;90 summaries,586 passed,0 failed,10 ignored | compliant-repair-acceptance-cpu-primitives.log/.exit |
| Whole runtime_d6_hinge_antipodes GPU file | PASS3/3,313.82s | paired-rates-next-nearby-and-unreached-baselines.log |
| Whole featherstone_mixed_distance GPU file | PASS4/4,403.16s | Same baseline log |
| Whole standalone_limit_precision GPU file | PASS4/4,405.12s | Same baseline log |
| Whole standalone_ball_limits GPU file | PASS2/2,221.42s | Same baseline log |
| Whole standalone_redundant_limits GPU file | PASS5/5,558.18s; expected incompatible-input fault is asserted,not a test failure | Same baseline log |
| Whole runtime_constraint_readout GPU file | PASS1/1,113.21s | Same baseline log |
| Whole articulated_response_gpu file | PASS3/3,4.21s | Same baseline log |
| Whole runtime_density_mass GPU file | PASS1/1,103.83s | Same baseline log |
| Nearby/unreached baseline component stage | PASS0; eight summaries,23 passed,0 failed,0 ignored | paired-rates-next-nearby-and-unreached-baselines.log/.exit |
| Unreached production GPU lanes | TERMINAL2:17 summaries,82 passed,one stale unsupported-repair assertion failed | paired-rates-next-unreached-production-lanes.log/.exit |
| First production file:runtime_primitives_contract_gpu | PASS12/12,329.06s; mutations,frames/mass/motors,fault recovery,world isolation and retention | Same production log |
| runtime_distance_gpu | PASS10/10,515.42s; includes one hardware-engine identity test | Same production log |
| runtime_distance_sleep | PASS3/3,134.07s | Same production log |
| runtime_distance_contract | PASS4/4,508.47s | Same production log |
| runtime_com_roots | PASS4/4,340.66s | Same production log |
| rigid_world::gpu::scale_tests library filter | PASS4/4,458.63s | Same production log |
| cone_geometry_gpu | PASS2/2,0.49s | Same production log |
| runtime_cone_capability | PASS2/2,117.29s | Same production log |
| runtime_cone_limits | PASS7/7,611.88s | Same production log |
| runtime_ball_loop | PASS5/5,588.17s | Same production log |
| Corrected runtime_break_detachment CPU file | PASS4/4; new loaded repair lifecycle oracle | compliant-repair-oracle-cpu.log/.exit |
| Corrected runtime_break_detachment GPU file | Prior source PASS4/4,241.84s; resource-counters also PASS4/4,8.75s in detached production stage | compliant-repair-detachment-gpu.log/.exit; compliant-repair-detached-remaining-production.log |
| Fresh six CPU and GPU examples | PASS0; six cases on each backend,hardware RTX4080 Vulkan | compliant-repair-acceptance-examples.log/.exit |
| Historical original make test-all | INTERRUPTED;456 summaries,2008 passes,0 failures,1199 ignored; superseded by current terminal pass above | compliant-repair-acceptance-test-all.log |
| Prior remaining production GPU lanes | TERMINAL2;24 passes,two pose-precision failures,seven summaries | compliant-repair-detached-remaining-production.log/.exit |
| First residue fix,pose-precision GPU file | TERMINAL2;4 passed,one3.64e-12m post-detachment equality failure | published-residue-pose-precision-gpu.log/.exit |
| Paired kinematic/free position advance,GPU fixture | PASS5/5,870.25s,exit0; original exact assertions unchanged | paired-position-pose-gpu.log/.exit |

Historical component baseline and production-lane overrides resumed near
demonstrated failures and then covered previously unreached files. They do not
replace the unchanged original aggregate now running after terminal native
and hardening passes.

Current hinge standalone modes retain144 case evaluations and original1e-3
angular-rate assertions. Maximum observed error is5.35299787e-5. Public recovery,
reactions,finite state and unchanged linear state also pass. The original fixture
hash is35d9139e78a5bd7fb66735230c00a1e4212edc8a88695e2f8245f674f52c50e4.

Current mixed-distance tests retain the original slack/unilateral/zero-bound
fixture hash a11a14272db02b0f7f8d9aeb474a7ddda84ba86f0c884952ed2a76c5e0f288bf.
No threshold,iteration budget or original physical fixture was relaxed.

Current public distance pendulum runs cover10 seconds at60/120/240Hz,eight
substeps. Peak anchor-distance errors are4.367491819e-5/1.091913809e-5/
2.729811238e-6m; maximum one-second CPU/GPU position difference is8.907788264e-7m.
Offset/rotated/full-tensor loop parity peaks at7.515518932e-7m over one second.

The distance contract additionally covers40 cases per rate,lengths0.01/0.1/1/10m,
equal masses0.1/1/100kg and both endpoint orders of1:1000kg,fixed and dynamic
parents. Over all three10-second sweeps:maximum relative length error2.237754870e-9,
one-second analytic position error1.052869982e-5m,relative quasi-static reaction
error1.951173827e-7 and relative momentum error1.554843533e-4. Momentum uses the
fixture's explicit1e-3 absolute denominator floor. These are separate metrics,
not all metres. Original finite-state,free rotation,step-count and no-quarantine
assertions remain enabled.

Current cone pendulums cover10 seconds at60/120/240Hz,eight substeps,with
4/5/2 active impact frames. Reported maximum cone violation and anchor error
are0 for this reduced-coordinate fixture. Energy peaks are5.600631225/
5.598089981/5.596823820J against initial5.595546871J and the unchanged bounded
energy assertion; final energies are approximately0.775J.

The separate loaded diagonal ellipse retains free twist:maximum angular-rate
error1.506736173e-4rad/s,dimensionless ellipse residual4.057678935e-7 and mean
torque error7.439448045e-6Nm. Loaded X/Y/Z/arbitrary-axis cases remain active,
with maximum reported angular violation1.881781252e-7rad and torque error
1.153383792e-5Nm under3Nm load.

Rigid external Ball-loop pendulums also pass all three10-second rates:maximum
anchor errors1.837989699e-5/4.694507960e-6/1.248545928e-6m,reported cone violation0,
and bounded energy. Closed-chain CPU/GPU trajectory differences are
7.519814721e-8m and4.848347164e-4rad. These external-loop measurements are not
replaced by the tree fixture's exactly represented anchor.

Density/live mass response retains the original2e-5 GPU oracle:maximum linear
velocity error1.6862083260912762e-9 and angular error7.711644600150974e-10.
Analytic COM/full-tensor aggregation and force/torque response are separate
assertions,not merely successful material setters.

Current device friction trajectories retain their original physical assertions.
Loaded holding covers10 seconds atdt0.001:maximum tangential drift2.277088606e-5m
and post-one-second-settling creep5.585206989e-7m. Kinetic sliding acceleration
is2.981366962m/s2 against analytic2.981347547m/s2 (relative error about6.51e-6,
well below the2% gate). Stop/re-stick reversal-window displacement is
7.817521691e-6m. Breakaway at1/2/4/8 substeps occurs at11.815N in each case;
largest CPU/GPU position difference2.031508559e-5m,orientation4.214684851e-8rad,
velocity1.002225231e-4m/s. Sliding-case total travel is not static holding drift.
All captured impulse energy,friction-regime,cap and independent host KKT
assertions pass,with peak KKT below1e-7. This device-component proof remains
separate from the public-runtime contact lifecycle suite:five cases passed on
bounded-island source; its failed static-only case passes after the binding fix.
An unchanged full-suite rerun is still required.

### Preserved failures and historical evidence

- `cloud-native-mode-final-gpu-primitives.log/.exit` naturally failed2 at09:09:27+02:
  31 summaries,96 passes,two standalone hinge failures,zero ignored. Native/harden
  never ran in that attempt. It is not OOM,timeout or an observer loss.
- `hinge-antipode-stage-diagnostic.log` preserves temporary shader reason2097184,
  proving16-iteration outer exhaustion rather than an inner response failure.
  Temporary diagnostic bits were removed before production validation.
- Earlier standalone distance and singular-root failures and fixes remain in
  the archived history and raw source-labelled logs. They are not hidden.
- Prior six CPU/GPU public examples passed on the preceding source snapshot.
  Fresh execution on fixed-reaction source is queued,not yet accepted.
- Camera smoke uses bounded dimensions while preserving original performance
  resolutions/aperture/contact workloads. Criterion0.5.1 native cloud args=[]
  means test mode; corrected smoke sizes1024/65536 pass18 actual operations.
  Explicit --bench retains the original1000003 size. Genuine old OOM logs remain.

## 6. Performance evidence

[MOLLA_PRIMITIVES_PERFORMANCE.tsv](MOLLA_PRIMITIVES_PERFORMANCE.tsv) has56 rows:
distance,cone,D6,reaction,break,friction and mixed,CPU/GPU,1/100/1000/10000 instances.
55 rows are collected snapshots; GPU mixed10000 is an explicit scalar-Jacobian
binding-capacity rejection,not a successful measurement or a native-OOM event.
Its measured workload fields are blank,not fabricated zero timing.

Successful rows use20 warmups,1000 measured steps,dt1/240,four substeps and
reaction readout. Fields preserve setup and wall times,device timeline,dispatch
counts,transfer/buffer/host allocation accounting,break events,source manifest
and raw log. Unavailable native/transient device peaks remain unavailable,not0.

Examples of preserved outcomes:

| Case | Measured wall | Other evidence |
|---|---|---|
| GPU mixed1000 | 153.672224360s | Setup158.956916940s; one break; four compilations;489429 dispatches |
| CPU mixed10000,60000 bodies/articulations | 1716.064103869s | One break; host peak1028237284B |
| GPU mixed10000,60000 bodies/articulations | Not measured | Checked dense scalar-Jacobian binding capacity error |

Rows refer to several source snapshots and a heavily shared RTX4080. They are
not final-current-source isolated speedup evidence. Device timeline includes
queue/host idle intervals and is not silently described as pure kernel time.
No external desktop/simulator application was stopped for measurements.

## 7. Original requirement audit

This is the final-scope audit map,not a completion checklist. Test paths below
are under `crates/molla-solvers/tests/` unless another crate/source is stated.
Current CPU aggregate proves its selected assertions; fresh GPU/native evidence
must still be inspected for every mapped requirement before acceptance.

| Requirement | Concrete coverage and acceptance evidence |
|---|---|
| P0 baseline/environment/routes | baseline-2026-09-28; runtime.rs/compiler,rigid_world.rs,rigid_world_gpu.rs; final provenance/source guards |
| P0 atomic settings/stamps/wake/handles | runtime scene/configuration/kind-wake/nonwaking suites; full native gate |
| P1 independent bounds,equality,zero/slack | runtime_distance*,runtime_distance_contract; public runtime,not setters alone |
| P1 world/two-body,1:1000 masses,offsets/frames/tensor | runtime_distance_gpu,contract,COM roots and coupled response fixtures |
| P1 boundary velocity/tangent/free rotation,exchange/momentum | endpoint exchange/coincident fixtures,free-body momentum,mixed-distance semantics |
| P1 loops,disconnected articulations,edits/deletion/compaction/sleep | distance contract/sleep and mixed-contact runtime fixtures |
| P1 residual/reactions/scales,10s at60/120/240Hz | distance contract40 cases/rate,eight substeps; independent reaction/momentum oracles |
| P2 ellipse/Jacobian,symmetric reduction/diagonal swing | MOLLA_CONE_GEOMETRY.md,molla-sim spherical_cone,cone_geometry_gpu |
| P2 axes/frames/free twist,q/-q,zero/disabled/wide/antipodal | runtime_cone_capability,configuration,limits and coupled angular geometry suites |
| P2 tree/external/contact anchors,reactions,energy and10s | runtime_cone_limits pendulum/loaded/two-body cases,runtime_ball_loop,mixed contacts |
| P2 mutation/sleep/compaction | cone mutation_sleep; native and actual GPU execution |
| P3a world child/parent anchors,means/peaks,fixed/motor/limit loads | reaction_wrenches*,runtime_joint_reactions,runtime_reactions_gpu; no double-counting/COM shift oracles |
| P3a stale/disabled/invalidated readout and retention | joint reaction/runtime reaction input/fault/resource fixtures |
| P3b inclusive force/torque/both/zero/disabled thresholds | runtime_joint_breakage*,substep peak and clear below/above oracles |
| P3b all six kinds,tree/loop,detachment/motion/repair | tree_and_loop_joint_kinds_detach_at_the_same_support_load,runtime_break_detachment |
| P3b simultaneous breaks,sleep/deletion,collision policy,rollback | breakage_gpu,contact_breakage_gpu,sequential boundary and explicit exclusion fixtures |
| P4 pair/legacy/combine rules and diagnostics | molla-geometry contact_materials,contact_friction,coulomb_disk_gpu; full native consumers |
| P4 inclined loaded10s,kinetic2%,re-stick/reversal/energy | runtime_friction_contract,public runtime_contacts_gpu,contact_response_gpu_tests and runtime_rigid_loop_drives |
| P4 moving surfaces,edits/separation/manifolds,sleep,compaction | runtime_friction/contact_sleep,public contact material/topology tests and device contact trajectory/material response |
| P5 D6 every axis,mixed/simultaneous drives and rotated frames | runtime_d6_axis_certification,mixed_drives,locked_drive_chains/planes,D6 angular and Ball interval suites |
| P5 hinge/slider motor models,gains,caps,mass/inertia scaling | runtime_motor_models/coupling,acceleration_drives,Ball drives and overflow validation |
| P5 all world-anchor kinds,fixed frames,free Ball and topology | joint_control,tree/loop/scene,public runtime contract and primitive-specific tests |
| P5 COM/full tensor,density/live edits | runtime_mass,COM roots,runtime_density_mass; independently derived force/torque response |
| P5 collision enable/permissions/materials/restitution/geometry | runtime_contacts_gpu,runtime_restitution*,runtime_shapes,collider_mass,resident geometry; declared unsupported boundaries |
| P5 independent worlds | runtime shared-engine gravity/state/mutation isolation and CPU counterpart |
| GPU execution/retention/dispatch/faults/no fallback | resource/dispatch/program-cache/rollback fixtures and inspected batched device route |
| P6 examples,measurements,conventions,adapter signatures | Six runnable public examples,56-row ledger,this report,MOLLA_PRIMITIVES_HANDOFF.md |
| Section11 all selected tests/nonzero original commands | CPU,test-all,harden pass;complete original GPU selection covered across two source-identical runs:412 passes,one100k LBVH timing failure;full aggregate not green |
| Section14 exact return package/scope preservation | Report/history/provenance/support matrix/handoff,56-row ledger and3315-path expanded inventory inspected;timing acceptance still open |

### Public surface audit,2026-10-02 20:18+02

Inspected current declarations and implementations,not only test names:

- `molla-sim/src/joint_control.rs` validates independent finite nonnegative
  distance bounds,independent cone extents in[0,pi],and nonnegative non-NaN
  break thresholds. Its inclusive trigger accepts valid zero-load samples;
  positive infinity cannot trigger a finite physical sample.
- `molla-sim/src/runtime.rs:1338–1516` provides the handoff's getters,setters,
  broken-state query and explicit repair. Distance/cone edits rebuild
  transactionally; break-limit edits touch dynamics without a topology rebuild.
  Successful changed settings activate endpoint dependencies. Batched breaks
  disable the authored joints and preserve stable handles via scene rebuild.
- `molla-solvers/src/rigid_world.rs:125–186` checks dynamics stamps before
  exposing reaction,distance or cone readings. Joint reactions additionally
  require enabled readout. Returned data is borrowed; the proposed Gearbox
  owned DTOs therefore correctly require copying before its mutex is released.
- `joint_reactions.rs`,`rigid_breakage.rs`,`distance_constraints.rs` and
  `cone_constraints.rs` match the documented public readout/event fields.
  `molla-sim/src/contact_material.rs` confirms explicit friction pairs,legacy
  uniform coefficients,and Average<Min<Multiply<Max precedence.
- `rigid_world_gpu.rs:209–242` validates and compiles the GPU backend before
  installing it on the existing RigidWorld; errors propagate without CPU
  fallback. `runtime_primitives.rs` explicitly rejects a software adapter.
- The current example log contains all six cases on both backends and hardware
  RTX4080/Vulkan identification. Its analytic fixed-load sample is20N on both
  backends,its distance equality reads1m with zero violation,and each break
  example checks exactly one event for the authored handle. The whole example
  gate has terminal exit0; these examples do not replace the long physical gates.

Branch/base/worktree were rechecked:develop at
da495a0971c666c7230dd49e0b2a48d0307fc7a4,one existing worktree. This audit
does not by itself certify the failed full GPU aggregate. Remaining physical
lanes subsequently passed on the same source; the timing gate remains open.

### Original acceptance checklist disposition,2026-10-03

The original unchecked plan is preserved. These dispositions distinguish
physical evidence from the still-failed command-level acceptance gate.

| Original checklist item | Inspected disposition |
|---|---|
| P0 baseline/routes | Recorded baseline environment/logs exist; native1.94.1,base revision and driver agree; CPU/GPU public entry points and production route inspected |
| Runtime distance | Current public runtime,sleep,contract,COM and scale tests pass; three-rate10s errors and independent reaction/momentum oracles satisfy the original gates |
| Spherical cones | Geometry contract/Jacobian,axes and sign tests,loaded ellipse/free twist,10s impacts and tree/loop anchors pass; not Euler clamps |
| General reactions | Current reaction/boundary/kinematic and contact-reaction files pass; independent loads,anchor shifts,means/peaks and stale/faulted readout assertions inspected |
| Break/detachment | Current tree/loop kind matrix,threshold channels,one-event lifecycle,rollback/repair and contact detachment pass; independent translation refinement peaks below9e-13m |
| Static/dynamic rigid friction | Loaded10s three-rate hold and live runtime hold/slide pass; slide2.981366962 versus analytic2.981347547,breakaway refinement,re-stick/reversal and material/filter edits pass |
| Existing D6/motor/frame/mass/filter contracts | Complete37-file GPU baseline,144-case D6 axis certification,public contract12/12,resident geometry/filter17/17 and native consumers pass |
| CPU and actual GPU lanes | Physical selections pass on hardware RTX4080; command-level acceptance remains incomplete because original full GPU aggregate failed LBVH timing |
| No silent fallback | Existing RigidWorld GPU construction validates/compiles before publication; hardware assertions,unsupported mutation/atomic rollback,retained resources and batched dispatch tests pass |
| Regression/benchmark evidence | Original failure and diagnostic preserved;412 GPU passes plus one failed timing assertion,156 remainder passes included; all97 GPU summaries select nonzero tests;56-row performance ledger has valid references |
| Scope preserved | develop/base/one worktree unchanged,no commits or dependency-lock changes; expanded3315-path inventory distinguishes the pre-existing deletion; no sibling/model edits performed |
| Examples/adapter return package | All six CPU and six hardware GPU examples pass; handoff matches current public types/methods,owned readout policy and explicitly proposed sibling signatures |

CPU distance-scale momentum peaks7.894477612e-13 relative with its stated floor,
below the original1e-6 CPU gate; the GPU counterpart peaks1.554844129e-4,
below1e-3. The unchanged test assertions establish the physical limits; benchmark
smoke Success lines are not substituted for measured performance. No ignored or
zero-selected GPU result is used as acceptance. The full goal remains unachieved
until the LBVH performance issue and the original aggregate gate are resolved.

## 8. Gearbox handoff and remaining work

[MOLLA_PRIMITIVES_HANDOFF.md](MOLLA_PRIMITIVES_HANDOFF.md) records actual Molla
methods and proposed neutral Gearbox signatures. In particular:optional world
parent,fallible insertion/mutation/step,independent bound fields,owned reaction
and event DTOs,explicit friction pairs and a documented legacy setter policy.
These are proposed sibling changes,not implemented Gearbox APIs.

Gearbox still owns schema reading/defaults,sentinels,degrees/unit conversion,
collection/filter resolution,scene routing,principal-axis mapping and USD/model
acceptance fixtures. No Gearbox,Blender or published machine assets were edited.

Remaining Molla work:

1. Finish the running original full GPU aggregate without lane overrides;
   the full LBVH file now passes under low observed load with unchanged limits.
   Preserve terminal exit and source guards. Native test-all and harden remain
   accepted on unchanged source; earlier high-load failures stay documented.
2. Revalidate any changed source against the original requirement map and
   numerical thresholds; do not turn segmented evidence into aggregate success.
3. Refresh the expanded dirty inventory after further changes,and only then
   complete the original acceptance checklist and goal.

## Key Learnings:

1. Focused fixes can pass while full acceptance remains open; ignored/queued tests
   and shader-only validation are not actual GPU certification.
2. Source-labelled history must stay distinct from the current return package.
3. Matching GPU trajectories can share a real bias; independent analytic
   substep refinement exposed cancellation in floating momentum COM transport.
4. Quaternion FK normalizes prescribed boundary rotations into high/low pairs;
   a zero-tail or unnormalized oracle contradicts that retained-precision contract.
