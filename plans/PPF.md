# PPF completeness on wgpu

## Objective and reference

Add the missing PPF physics and integration capabilities identified in the
2026-11-15 comparison, retaining native Rust/WGSL execution on wgpu and
measuring performance rather than assuming that GPU execution is fast.
Upstream reference: `st-tech/ppf-contact-solver` at
`71a2e26c8b0d6dc5b12094cabdc16acc7c1017a1`.
Starting Molla revision: `97a604ce`.

This is not a request to embed the upstream native runtime. Credit remains in
`ACKNOWLEDGEMENT.md`. The scope below remains open until its implementation,
integration, correctness, resource, and performance evidence is recorded.

## Required work

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| P0 | Runnable native CPU/GPU verification and measured baseline | Make targets; real adapter identity; repeated elapsed and allocation/readback measurements on fixed workloads | In progress |
| P1 | Robust input and accepted-state contracts | CPU/GPU refuse invalid configuration, nonfinite state, intersecting starts and invalid accepted geometry; rejected steps do not publish state; adversarial tests | In progress: CPU/GPU geometry gates and transactional host/readback publication; broader geometry, lifecycle and device-loss qualification open |
| P2 | Material-complete barrier solve | Shell membrane and bending, tet constitutive energies, damping and relevant plasticity integrated with contact; analytical derivatives checked against finite differences; WGSL parity | In progress: CPU/GPU membrane/tet contact, initial Model authoring and coupled-operator parity; broader authoring qualification, bending, damping and plasticity open |
| P3 | Triangle principal-stretch limiting | Deformation-gradient singular-value limits, matching barrier derivatives and CCD/line-search constraints; shear/directional/rotated tests, not just edge length tests | Open |
| P4 | PPF dynamic stiffness | Local material Hessian projection and gap-dependent inertia; CPU oracle, WGSL implementation and stiff-contact tests | Open |
| P5 | Complete mixed GPU integration | Explicit rod edges/bending and relevant rod models, GPU rigid/deformable coupling, supported collider geometries and moving boundary controls; Model/State integration, no silently skipped shapes | Open |
| P6 | Scalable resident execution | Resident broad phase and sparse contact adjacency; checked capacity growth; persistent scratch/readback buffers; no added per-element CPU hot loops; benchmark dense/adversarial contacts | In progress: scalar staging and resident validation BVH; performance qualification open |
| P7 | Faithful end-to-end validation | Original moving-boundary problems rather than static stand-ins; cloth stacks, knot/rods, coupled rigid contact, solids/plasticity and applicable mixed/granular scenes; independent intersection checks and inspected rendered results | Open |

Existing Molla FEM, VBD, MPM, vehicle and articulation functionality must not
be reported as absent or replaced merely because its integration with the
barrier backend is incomplete. Inventory upstream terms when implementing P2
and P5; do not silently drop capabilities or substitute different physics.

## Execution and verification

1. Restore the library workspace's runnable gates, then record baseline
   correctness and performance before numerical changes.
2. Establish P1 rejection/publication safety and GPU validation infrastructure.
3. Implement P2-P4 together where shared constitutive derivatives require it.
4. Complete P5 integration and P6 resource/scaling work without weakening P1.
5. Qualify P7 using the real public construction/step path and actual hardware.

Use `make test-ppf` and `make test-ppf-gpu` for focused regression lanes.
Use `make test APP_TARGET='...'`, `make test-gpu P=... T=...`, and
`make fmt-files FILES='...'` for individual checks. Run release benchmarks
through Makefile targets as they are added. Do not treat focused tests as
whole-project, platform, visual, or performance acceptance.

Record adapter/driver, source identity, parameters, scene size, contacts,
substeps, warmup and repetitions. Measure median and tail latency, memory,
allocation/dispatch/readback counts, and both setup and steady-state costs.
Compare performance only under matching physical/correctness contracts;
removing safety checks or changing the scene is not an optimization.

## Current evidence

- Initial `nix develop --impure -c make test ...` failed before compilation:
  the workspace listed absent `bin/molla_viewer/Cargo.toml`. This vendored
  checkout contains reusable physics crates, not that viewer. Removed the
  nonexistent member so focused library verification can run.
- No PPF completeness or performance acceptance has been established yet.

### Input validation and resource checkpoint

Changes are based on `97a604ce`. The workspace manifest and
lockfile now describe the libraries actually present in this vendor checkout;
the default viewer-oriented Make targets are not a library acceptance lane.
The focused commands below explicitly select the physics packages.

- CPU and GPU parameters reject NaN/infinity, invalid coefficient ranges and
  empty iteration budgets. Timestep checks precede any integration work.
- GPU mesh construction checks indices, finite state, masses, springs,
  triangle/tet degeneracy, pins and analytic obstacles before indexing or
  allocating device buffers. Velocity uploads reject nonfinite values before
  writing. These are setup/input checks, not proof against runtime overflow.
- The CPU constructor checks positions and springs before building adjacency;
  CPU stepping checks particle-state lengths and finiteness. The common solver
  facade validates parameters and timesteps before building or updating a backend.
- Zero-rest Hookean springs are intentional: the existing elastic-press
  regression uses one. They remain valid with strain limiting disabled. A
  length/rest strain limit on such a spring is refused, including when the
  caller enables the limit after construction. An initial overstrict rejection
  of all zero-rest springs failed that regression and was corrected.
- GPU scalar readback retains one 64-byte staging buffer and copies directly
  into a stack array. It unmaps the retained buffer on success and mapping/poll
  errors. Vector readbacks, LBVH setup and other scratch paths remain unchanged.
- `plans/PPF.md` has an explicit repository ignore exception; the developer's
  global `plans/` ignore must not hide the active completeness checklist.

Baseline verification (RTX 4080, Vulkan, NVIDIA 595.112.07):
`make test-ppf` passed 25 CPU tests and `make test-ppf-gpu` passed 8 GPU tests
before solver edits. Commands use
`NVIDIA_VERSION=595.112.07 nix develop --impure -c make ...` on this host.

Additional lanes:

```
make test-ppf-inputs
make test-ppf
make test-ppf-gpu GPU_FEATURES=resource-counters
make bench-ppf-gpu
make bench-ppf-replay PPF_BENCH_BINARY=/absolute/path/to/release/test-binary
```

The input unit lane enables `resource-counters`: existing unrelated kernel
unit-test modules reference metrics scopes unconditionally and do not compile
with only `--features gpu`. This does not affect the standalone barrier GPU
integration lane; it is not evidence that the entire workspace is verified.

The release resource workload is a spring cloth over a plane and sphere,
hash-grid self-contact enabled, 64 or 576 vertices, 98 or 1,058 triangles,
161 or 1,633 springs, default GPU parameters, dt=1/240, 48 warmup steps and
32 timed steps, three fresh solvers per size. Timing includes queue completion
after every step. Setup is measured separately; adapter construction is excluded.
Final positions are checked independently against analytic obstacles, not
against all self intersections. Resource scopes count instrumented application
buffer creations and logical transfers, not every driver or host allocation.

| Vertices | Buffers created during 32 steps, before → after | Readback bytes, both | Dispatches, both | Resident bytes, before → after |
|---|---|---|---|---|
| 64 | 1,339 → 0 | 85,696 | 51,765 | 10,182,144 → 10,182,208 |
| 576 | 2,435 → 0 | 155,840 | 79,631 | 79,961,856 → 79,961,920 |

Newton totals also matched (173 / 234 iterations; 1 / 4 capped steps). The
resource regression asserts zero instrumented step-time buffer creations.
This is an allocation reduction, **not a latency acceptance claim**. Initial
before/after timing cohorts were mixed, and a later after-only cohort was
slower. The WGSL program has not changed in this checkpoint.

Post-change verification: **29 CPU regressions, 5 input-validation unit tests,
11 hardware GPU regressions**, the release resource test and the touched-file
format checks passed. GPU rejection tests check unchanged published positions
and velocities, recovery with valid input, zero-rest spring limit changes and
zero instrumented allocations for invalid constructors. These tests do not
cover malformed mutable CPU topology, device loss, runtime floating-point
overflow, general intersections or all-or-nothing mixed-body publication.

An A/B/B/A replay of separate baseline/current release binaries still showed
mixed latency and substantially different setup times. A second diagnostic
used **one binary** and changed only whether each scalar readback allocated a
temporary staging buffer or reused the retained buffer. Both branches used
the same validated solver and stack-array copy; both constructed the retained
buffer (unused in the allocation branch). This isolates steady-state staging,
but does not isolate the original constructor/setup difference.

Median of six per-run reported percentiles, in milliseconds (not percentiles
of pooled samples):

| Vertices | Allocating p50 | Retained p50 | Allocating p95 | Retained p95 |
|---|---|---|---|---|
| 64 | 14.087 | 14.959 | 36.215 | 34.512 |
| 576 | 28.268 | 29.900 | 57.371 | 48.322 |

Retention reduced tail latency in this diagnostic but worsened central latency
by about 6%. It remains a measured resource improvement with an **unresolved
latency/setup tradeoff**, not a qualified speed improvement. P6 stays open.
No GPU clock, power or other-user process settings were changed for these runs.

Local evidence (temporary files, not portable project dependencies):

- `/tmp/molla-ppf-validation.log`, `/tmp/molla-ppf-after-cpu.log`,
  `/tmp/molla-ppf-after-gpu.log`, `/tmp/molla-ppf-fmt-check.log`.
- `/tmp/molla-ppf-resources-before.log`,
  `/tmp/molla-ppf-resources-after.log`,
  `/tmp/molla-ppf-resources-after-repeat.log`.
- `/tmp/molla-ppf-abba-{1-legacy,2-retained,3-retained,4-legacy}.log` and
  `/tmp/molla-ppf-toggle-{1-legacy,2-retained,3-retained,4-legacy}.log`.
- Baseline source archive: `/tmp/molla-ppf-baseline.3F4Xxy`, originally
  `97a604ce` plus the corrected workspace/lockfile and benchmark instrumentation.
  It was subsequently modified for the single-binary diagnostic; the original
  driver source is preserved as `/tmp/molla-barrier-newton-before.rs`.
- Saved release binaries: `/tmp/molla-ppf-legacy-bench`,
  `/tmp/molla-ppf-retained-bench`, `/tmp/molla-ppf-toggle-bench`.
  The runtime toggle exists only in the temporary diagnostic source, **not**
  in Molla's production API or repository tests.
- Shared WGSL SHA-256:
  `48c1c45428ac0768f6298b95841f249e1cb7d112bf8582d4fbeb5886386d5723`.
- Baseline binary SHA-256:
  `5bb29cdd9d7ea3e63da65e81697324b7360ace99dd2e9e65972e408b1da2b4d3`.
- Current binary SHA-256:
  `3f633e948d4adcc4bb654253d2deba7f6b33f9168d61117193b86b3da6246c7b`.
- Diagnostic binary SHA-256:
  `b4610288501d715200af22b4a4516c1386ae7070834b17d12bc510294d36aebd`.

### GPU surface-geometry checkpoint

The barrier GPU backend now validates initial and final geometry using
`crates/molla-kernels/shaders/barrier_geometry.wgsl`, orchestrated by
`src/barrier_newton/geometry.rs`. The solved iterate is checked before the
existing commit kernel can publish positions and velocities. Invalid steps
reset the CG warm-start validity; successful retries are covered by tests.

- Initial checks cover finite committed/predicted state and velocity, enabled
  spring strain limits, triangle/tet degeneracy, tet orientation relative to
  rest, analytic obstacle penetration and mesh-surface intersections.
- Segment/triangle and point/triangle queries include coplanar clipping and
  isolated vertices. Shared triangle edges and shared endpoints are excluded
  topologically. Analytic obstacle checks exclude pinned vertices, matching
  the existing anchor convention.
- The final check also verifies finite output velocities and gradients.
  Newton evaluations, CG residuals and line-search scalars refuse nonfinite
  results before proceeding. This is not a proof against every possible
  floating-point overflow or convergence pathology.
- The validator owns a static-topology BVH. Triangle AABBs are produced from
  the live iterate on the GPU and refitted with no pair enumeration or mesh
  readback. `LbvhBroadPhase::build_bounds` and `encode_refit_bounds` expose
  this path; ordinary pair-generating APIs retain their behavior.
- Bounds, tree nodes, binding groups and status storage are retained. Traversal
  has explicit node/count/stack failure guards, rather than silently dropping
  candidates. Only status is copied through the retained scalar readback.
- Once a committed state has passed the final contact check, its initial
  contact result is cached. This is valid because topology, obstacles and
  committed positions have no external mutation API. Initial numerical and
  strain checks still run, and **every final candidate receives the full
  geometry check**. Any future moving-boundary/position update API must
  invalidate this cache or supply an equivalent validation contract.
- `BarrierGpuParams::reject_intersecting_start` defaults to true and is mapped
  from the common solver parameters. Disabling it permits a recovery attempt,
  but cannot publish a still-invalid result. The CPU oracle's historical
  recovery behavior is not yet aligned with that final-publication rule.

Geometry verification uses `make test-ppf-geometry`: one direct GPU-buffer
test (moved BVH bounds, intersection detection, degenerate/nonfinite iterates,
cleared error status and retained allocations); three resident BVH tests;
five integration tests covering piercing, coplanar containment and edge
crossings, axis/winding variants, isolated vertices, valid adjacency at several
scales, moving separated surfaces, pinned obstacles, strain rejection,
final tet inversion, incomplete recovery and numerical overflow. Published
positions and velocities are compared exactly before/after rejected steps.

An initially failing small-scale adjacency fixture used an activation distance
as large as its edges, so the existing self-barrier correctly moved the mesh.
Its no-force test now scales activation distance with the mesh; the geometry
predicate was not weakened to make the fixture pass.

This is an f32 surface-geometry validator with scale-dependent roundoff
allowances, not an exact-predicate or cross-device geometric certificate.
It does not yet close the CPU coplanar/final-state gaps, arbitrary malformed
mutable CPU topology, fully transactional Model/State publication, device-loss
recovery, containment of closed solids, explicit GPU rod edges or generalized
rigid coupling. Those remain required alongside P2-P7.

Verified on the same RTX 4080 / Vulkan / NVIDIA 595.112.07 environment:

- `make test-ppf-geometry`: **1 direct validator + 3 BVH + 5 geometry tests
  passed** (`/tmp/molla-ppf-geometry-tests-final.log`).
- `make test-ppf-gpu GPU_FEATURES=resource-counters`: **11 passed**
  (`/tmp/molla-ppf-geometry-regression-final.log`).
- `make bench-ppf-gpu`: **passed**, including zero instrumented step-time
  buffer allocations (`/tmp/molla-ppf-geometry-resources-final.log`).
- Touched Rust formatting and `git diff --check`: passed. No CPU-only full
  regression rerun or rendered/cross-platform qualification is claimed for
  this checkpoint. CPU solver source was not changed in this checkpoint.

Current resource totals per 32 measured steps, with the initial contact cache
already established during warmup:

| Vertices | Buffer creations | Resident buffer bytes | Readback bytes | Upload bytes | Dispatches | Per-run p50 range (ms) |
|---|---|---|---|---|---|---|
| 64 | 0 | 10,201,552 | 89,792 | 79,544 | 52,117 | 15.070–21.504 |
| 576 | 0 | 80,122,352 | 159,936 | 154,992 | 80,111 | 30.088–35.376 |

Versus scalar retention without geometry checks, validation adds 19,344 /
160,432 resident bytes, 4,096 readback bytes, 3,072 upload bytes and 352 / 480
dispatches. The readback increment is **two fixed 64-byte status reads per
step**, not mesh-sized traffic. Newton iterations and capped-step totals
remain 173 / 234 and 1 / 4 in these workloads.

The first implementation redundantly ran initial contact traversal every
step; it encoded 52,405 / 80,527 dispatches for the same measured windows
(`/tmp/molla-ppf-geometry-resources.log`). Caching the already-validated
committed contact state eliminates 288 / 416 dispatches without omitting the
final gate. Elapsed timing remains variable and the larger workload has
visible validation cost; this is not a latency acceptance or speedup claim.

### CPU geometry and output publication checkpoint

The CPU oracle now validates mutable particle/topology arrays, derived edge
provenance, obstacles, affine ownership and inertia before indexed access in
`step`. Spring stiffness sums are refreshed from the current public springs.
Initial and accepted geometry checks cover coplanar/piercing edge-triangle
intersections, isolated vertices, degeneracy, tet orientation, obstacle gaps
and enabled strain limits. The CPU predicate uses projected segment/edge
tests, rather than copying the GPU's coplanar interval-clipping implementation.
Neither predicate is an exact-arithmetic geometric certificate or a closed
solid containment test.

Predictions, Newton gradients, PCG scalars, energy and output velocities are
checked for numerical overflow. Cloth positions/velocities and all affine
poses/velocities publish only after every final check succeeds. Disabling
intersecting-start rejection permits an attempted recovery, not publication
of still-intersecting geometry; both a successful recovery and an incomplete
recovery refusal are tested. Existing legacy infallible construction methods
such as `with_tets`, `pin_vertices` and `add_affine_body` still require valid
arguments; step-time checks are not proof that every builder is panic-free.

The Model/State facade preflights host authority and exact output/input lengths
before solving. Construction checks consumed Model arrays, indices, material
coefficients, shape references/transforms and soft-surface vertex ranges.
All four motion arrays publish together, with diagnostics updated last.
CPU rollback retains the previous shell, affine state and obstacle bindings
if a later body reaction/rotation fails. It clones CPU-oracle state; this is
not an extra per-element host loop on the resident GPU path. Particle/body
count changes rebuild the CPU backend. Same-count topology/material edits,
moving GPU boundaries and externally edited GPU trajectories still require
the P5 lifecycle work; construction validation is not a per-step Model hash.

`BarrierNewtonGpu::step_with_readback` solves into existing scratch buffers,
stages candidate velocities on the GPU in the existing `cg_hd` buffer, and
reads both candidate arrays before queuing the commit. The facade uses this
method. Standalone `step` retains its no-vector-readback path. No new retained
GPU buffer is required. Vector reads still allocate two temporary staging
buffers, as the old facade did; P6 must retain/coalesce those separately.
The readback path adds one velocity-staging dispatch and one uniform upload.
Injected failures in either candidate readback preserve committed GPU arrays,
invalidate CG warm starts, and permit a valid retry. This is not device-loss
recovery, nor an atomicity guarantee if a device fails after commit is queued.

Unsupported Model tet materials, unimplemented collider geometries,
body-attached GPU shapes, and explicit GPU rod-edge requests now fail loudly
at construction rather than being silently omitted. These are temporary
refusals, **not** implementations of P2/P5 or a reduction of their scope.

Focused verification uses `make test-ppf`, `make test-ppf-publication`,
`make test-ppf-gpu`, `make test-ppf-geometry` and `make bench-ppf-gpu`.
Final qualification on the same RTX 4080/Vulkan/595.112.07 adapter:

- **41 CPU regressions**: `/tmp/molla-ppf-publication-cpu-final.log`.
- **5 input-validation unit tests**, touched Rust formatting checks:
  `/tmp/molla-ppf-publication-format-inputs-final.log`.
- **11 GPU regressions**:
  `/tmp/molla-ppf-publication-regression-final.log`.
- **9 geometry/BVH hardware tests** (1 direct validator, 3 LBVH, 5 barrier):
  `/tmp/molla-ppf-publication-geometry-final.log`.
- **4 publication hardware tests** (1 injected-readback unit test, 3 public
  integration tests): `/tmp/molla-ppf-publication-gpu-final.log`.
- **Release resource workload passed**:
  `/tmp/molla-ppf-publication-resources-final.log`. Standalone stepping still
  creates zero instrumented buffers, with unchanged resident bytes, uploads,
  readbacks, dispatches, Newton totals and capped-step counts relative to the
  geometry checkpoint. This benchmark does not exercise facade vector reads.
  Three-run p50 ranges were 9.553–10.229 ms (64 vertices) and 16.973–18.354 ms
  (576 vertices). These are unmatched later cohorts, **not** evidence of a
  qualified speedup over the earlier variable timings.
- `git diff --check` passed. No rendered, cross-platform, whole-workspace or
  device-loss acceptance is implied.

Current main WGSL SHA-256:
`d268e7ae732d285d9ec4b0eaa664e238bad97e72d717cb6f685b77de17b62653`.
Geometry WGSL SHA-256:
`a242e882f30e4c70e7c359f4d7b5a6ce0dbfc1fc4539f13e374344dd21a3eaa4`.
The source-acknowledgement commit is separate from these implementation changes.

### P2-P4 source inventory and implementation constraints

Checked directly against the pinned upstream checkout. Paths below are
relative to `crates/ppf-cts-solver/src/` in that upstream revision:

| Source | Required behavior / integration constraint |
|---|---|
| `data.rs`, `kernels/energy/model/arap.hpp` | The upstream `Arap` selector currently evaluates **corotated linear** energy: `mu * sum((sigma - 1)^2) + lambda/2 * sum(sigma - 1)^2`. Do not substitute a differently scaled textbook ARAP term. |
| `kernels/energy/model/stvk.hpp`, `detsqr.hpp` | This revision's `StVK` implementation uses `mu/4 * sum((sigma - 1)^4) + lambda/2 * (product(sigma) - 1)^2`, not the usual Green-strain formula. Preserve the referenced behavior explicitly rather than silently replacing it based on its name. |
| `kernels/energy/model/snhk.hpp` | Stable Neo-Hookean spectral gradient/Hessian in 2D and 3D; 3D energy is `mu/2 * (sum(sigma^2) - 3) - mu * (J - 1) + lambda/2 * (J - 1)^2`. Verify the parameter conversion at assembly, not merely the function name. |
| `kernels/energy/model/baraffwitkin.hpp` | Directional membrane stretch and shear from the two columns of the 3x2 deformation gradient, with positive-semidefinite Hessian projection. This is not represented by an undirected spring graph. |
| `kernels/energy/model/shell_bend.kernel.cpp`, `shell_bend_stiffness.kernel.cpp`, `kernels/utility/hinge_damping.kernel.cpp` | Four-vertex dihedral bending, material/geometric stiffness and lagged damping. Rest angles and directional coefficients belong to the material state. |
| `kernels/plasticity/plasticity.kernel.cpp` | Thresholded singular-value/rest-angle creep with `alpha = 1 - exp(-rate * dt)` and updated inverse rest matrices. This differs from Molla VBD's von-Mises return mapping; do not label the latter PPF plasticity. |
| `kernels/strainlimiting/shell_strain.kernel.cpp`, `strain_toi.kernel.cpp` | Limits on shifted principal stretches; authored barrier limit and shrink-adjusted stiffness limit are distinct. Line search must protect the singular-value bound, not just mesh-edge lengths. |
| `kernels/barrier/contact_stiffness.kernel.cpp` | Contact direction is the normalized vector of contact weights times the separation vector. Contract the assembled local Hessian along this direction, then add directional inertia proportional to mass/gap squared. Incident spring sums are not equivalent. |
| `kernels/energy/model/pdrd_rigid.kernel.cpp`, `cosserat.kernel.cpp`, `rod_bend.kernel.cpp` | Rigid-fit and rod material families remain part of P5, alongside explicit edge collision and moving-boundary integration. |

Implement shared element constitutive records and independent CPU f64 energy,
gradient and raw/projected Hessian checks, then resident WGSL assembly and
matrix-free products through element-to-vertex adjacency. Keep material and
plastic state transactional with the position update. The acceptance gates
must include finite differences before PSD projection, projected-operator
symmetry/PSD checks, repeated singular values, rotated/sheared elements,
material-coordinate anisotropy, compression, and CPU/GPU trajectories.
Caching per-element tangents once per Newton iterate avoids recomputing SVDs
inside every PCG product; benchmark storage/dispatch cost before qualification.

Molla's existing `vbd.rs::neo_hookean_vertex_block` is a **diagonal vertex
block**, not the full coupled material Hessian needed here. Its current
plastic branch also treats `det(Fp)` near one as identity; a nonidentity
isochoric plastic deformation can have determinant one. Do not carry that
shortcut into the new barrier material implementation. Existing VBD code has
not been modified or independently qualified by this checkpoint.

### Constitutive and CPU element checkpoint

`molla-math::constitutive` provides f64 energy, gradient and full coupled
deformation-gradient Hessians for corotated, quartic-stretch and stable
Neo-Hookean membranes/solids, plus directional Baraff-Witkin membranes.
Raw derivatives and positive-semidefinite projected tangents are separate.
Tests cover finite differences, repeated singular values, compression,
rotations, symmetry, projected positive semidefiniteness and material axes.
The CPU energy-only path does not form tangents during line search.

`molla_sim::ElasticElement` constructs rest coordinates for triangles and
tetrahedra. `BarrierShell::with_elastic_elements` adds their energy, gradient
and matrix-free coupled Hessian products to the CPU contact solve, caching
projected tangents once per Newton iteration. A conservative singular-value
step bound prevents crossing rank-deficient configurations even when the
trial endpoint has positive determinant. Collision topology remains a
separate input. Tests exercise spring-free membrane and solid relaxation,
the element-to-vertex chain rule, pinned vertices and inversion rejection.

Element weighting is explicit: rest area times thickness for membranes and
rest volume for solids. The pinned upstream assembly instead multiplies by
element mass (`density * area` or `density * volume`). Matching the energy
functions alone does **not** establish scene-parameter parity; density,
thickness and material-coefficient conversion still need integration tests.
Authored membrane material-coordinate frames are not exposed yet.

`molla-kernels` has reusable WGSL material functions and a resident batch
kernel with checked input/output bindings. Hardware tests compare 140 raw
and projected responses against the f64 oracle, including full tangents,
and reject malformed device inputs. The observed maximum scaled error was
`1.011e-6`. This is a **standalone constitutive kernel**, not GPU mesh-element
assembly or material integration into `BarrierNewtonGpu`.

Verification targets:

```
make test-ppf-materials
make test-ppf-materials-gpu
make bench-ppf-materials
```

The first two passed 4 constitutive tests, 4 CPU element/contact tests,
1 offline WGSL validation and 2 hardware tests. Local evidence is in
`/tmp/molla-ppf-material-integration-first.log` and
`/tmp/molla-ppf-constitutive-gpu-first.log`. The main `test-ppf` lane now
includes the CPU element/contact tests.

The release batch resource test passed on RTX 4080/Vulkan/595.112.07
(`/tmp/molla-ppf-constitutive-resources.log`). For 256/4,096 elements, retained
input/output buffers occupy 118,784/1,900,544 bytes. Each measured 16-dispatch
window created zero instrumented buffers and performed no uploads or
readbacks; outputs were checked after timing. Most 4,096-element per-run
medians were 0.123–0.177 ms, but corotated and Baraff-Witkin cohorts reached
1.277/1.341 ms. These variable isolated-kernel timings establish neither a
solver speedup nor full-solver performance acceptance.

P2 remains incomplete: resident GPU element assembly, cached matrix-free
products within the contact solver, GPU energy-only line search, Model/State
authoring, bending, damping and plastic state are still required. P3/P4
principal-stretch barriers and directional material-based contact stiffness
are also open; the CPU rank-preserving step bound does not replace them.

Pre-commit verification on the same adapter passed:

- 45 CPU barrier regressions, including the 4 material/contact tests.
- 5 input-validation unit tests, 4 constitutive tests and 1 offline WGSL check.
- 11 main GPU regressions, 9 geometry/BVH tests, 4 publication tests and
  2 constitutive hardware tests.
- Touched-file formatting with `FMT_ARGS='--check --config skip_children=true'`
  and `git diff --check`. Recursive formatting also visited unrelated files
  with existing formatting differences; those files were left unchanged.

The geometry Make target explicitly selects its supported `gpu` feature;
inheriting `resource-counters` from an aggregate command had failed because
`molla-geometry` does not expose that feature. The corrected lane passed.
Evidence: `/tmp/molla-ppf-precommit-focused-checks.log` (through the main GPU
regressions) and `/tmp/molla-ppf-precommit-gpu-final.log` (remaining hardware
lanes). These remain focused checks, not whole-workspace or PPF acceptance.

### Resident GPU material integration checkpoint

Implementation continues from `70b3ad6c` and is not yet committed.
`BarrierNewtonGpu::new_with_elastic_elements` accepts authored
`molla_sim::ElasticElement` rest coordinates alongside collision geometry.
Spring and legacy volume terms remain available independently; an empty
element list allocates no material buffers and dispatches no material work.

- Setup validates element indices, rest coordinates, f32 coefficient range,
  dispatch limits and storage capacities before allocating solver buffers.
  Element records and vertex-to-element CSR adjacency are uploaded once.
- WGSL assembles the material gradient, coupled matrix-free Hessian products,
  energy and rank-preserving step limits inside the existing Newton/PCG and
  Armijo sequence. Vertex gathers avoid floating-point atomic force sums.
- Projected tangents and minimum principal stretch are cached per element
  once per Newton iterate. PCG products do not recompute SVDs or tangents.
  Each element retains a 96-byte record, 400-byte tangent response and
  48-byte shared scratch, plus CSR storage.
- GPU energy-only evaluation avoids tangent construction during backtracking.
  Corotated/quartic energy uses the shared SVD helper; stable Neo-Hookean and
  Baraff-Witkin energy use direct expressions.
- Initial and final element geometry checks precede publication. Material
  error bits share the existing 64-byte scalar readback; there are no added
  material-sized host transfers or per-element CPU loops during stepping.
  This does not remove the pre-existing optional LBVH host AABB path.
- The cached minimum singular value bounds the Newton segment away from
  rank loss. Positive endpoint determinant alone is insufficient: a segment
  can cross two inversions before reaching another positive determinant.
  This bound is not P3's upper principal-stretch barrier.

The hardware assembly test compares a mixed seven-element membrane/solid
mesh, shared vertices and pinned rows against the independent f64 oracle.
It checks energy, gradient, coupled Hessian-vector products, step bounds,
invalid trials and geometry-status clearing. The assembly dispatch window
creates zero instrumented buffers and performs no uploads or readbacks.

The integrated solver tests cover all seven supported model/dimension
combinations without springs, 60-step CPU/GPU trajectories, fixed anchors,
material relaxation, an actual tetrahedron drop into plane contact, and
transactional failure/retry after a material tangent overflows.
With matching 40-iteration Newton budgets, observed worst trajectory error
was `3.333e-4` metres on the metre-scale fixtures. The first comparison used
the different CPU/GPU defaults (40 versus 12 iterations) and diverged under
the mesh travel cap; it was corrected to compare equal budgets, not by
weakening CCD. Production defaults remain unchanged.

Required commands:

```
make test-ppf-materials
make test-ppf-materials-gpu
make test-ppf-material-solver-gpu
make bench-ppf-material-solver
```

The resource workload uses spring-free stable Neo-Hookean membrane triangles
with thickness 0.01, mu/lambda 10,000, a plane and sphere, hash-grid contact,
48 warmup steps, 16 measured steps and three fresh runs per size at dt=1/240.
Setup and queue-completed step latency are measured separately. Final
positions are independently checked against both obstacles and each
element's deformation domain. This is not a rendered or cross-device test.

#### Material product staging decision

Retained a two-dispatch Hessian product: one stress product per element,
then a CSR vertex gather. A tested single-dispatch variant recomputed that
product at each incident vertex. Its assembly outputs matched the staged
fixture, and all seven trajectory metrics, Newton totals and capped-step
counts matched, but it was slower despite fewer dispatches.

An A/B/B/A replay used saved release binaries on RTX 4080/Vulkan/595.112.07,
with the same workload and no GPU clock/power changes. Values below are
medians of six per-run percentiles, not percentiles of pooled samples:

| Vertices / elements | Staged p50 (ms) | Fused p50 (ms) | Staged p95 (ms) | Fused p95 (ms) |
|---|---|---|---|---|
| 64 / 98 | 45.975 | 63.713 | 85.128 | 92.520 |
| 256 / 450 | 72.303 | 107.529 | 121.685 | 147.332 |

Fusing reduced measured-window dispatches from 30,863 to 26,948 and 43,555
to 38,088, but worsened median latency. The fused production path was
removed. This comparison supports that local implementation choice only;
these step times are still costly and **do not establish P6 performance
acceptance or a speedup over the earlier spring-only solver**.

Both variants allocated zero instrumented buffers during stepping and had
identical logical traffic: 43,456/66,368 readback bytes and 43,272/67,496
upload bytes per 16 measured steps. Resident buffer totals were
10,252,992/39,784,160 bytes. Newton totals were 90/97, with no capped steps.
Cold setup was approximately 0.98 seconds; later setup was generally
19–48 ms. Shader/pipeline setup is not included in step latency.

Local replay artifacts (temporary, not project dependencies):

- `/tmp/molla-ppf-material-ab.OSkevV/{1-staged,2-fused,3-fused,4-staged}.log`.
- Staged binary `staged`, SHA-256
  `2925de8785d540c3a7613a1ddac9f9e44dcdf385ddb7c386e381583cf1187e28`.
- Fused binary `fused`, SHA-256
  `418afdc3b70789e9b1ae0e0621e567c6f2bf49ca3122364bc2c029c939b877f2`.
- Candidate sources `fused.rs`, `fused.wgsl` and both source-hash manifests
  are in the same directory. Replay with
  `make bench-ppf-material-replay PPF_BENCH_BINARY=/absolute/path/to/binary`.
- Initial integration and candidate checks:
  `/tmp/molla-ppf-material-solver-qualification.log` and
  `/tmp/molla-ppf-material-fused-first.log`.

This checkpoint does not close P2: the public Model/State facade still needs
material authoring and parameter mapping, and bending, damping and plastic
state are absent from the new element path. P3/P4 and the mixed/rod/rigid,
scaling and scene-fidelity requirements P5-P7 remain open.

Final focused gates on the retained staged implementation passed:

- 15 non-hardware checks: 4 constitutive, 4 CPU element/contact, 2 WGSL
  validation and 5 input-validation tests.
- 30 hardware checks: 2 constitutive, 1 coupled element assembly, 3 material
  solver, 11 existing barrier regressions, 9 geometry/BVH and 4 publication.
  The assembly test also checks a positive-endpoint double-inversion path;
  the GPU bound stays below its first singularity. The seven trajectory
  cases enforce a 0.001-metre maximum error gate with matched budgets.
- Both release resource workloads passed. The no-material path retains its
  prior buffer, transfer, dispatch, Newton and capped-step totals; material
  stepping also creates zero instrumented GPU buffers.
- Evidence: `/tmp/molla-ppf-material-final-gates.log`. The final material
  timing cohort reported p50 ranges of 15.625–18.745 ms (64 vertices) and
  30.347–35.083 ms (256 vertices). These are later, unmatched cohorts with
  substantial timing variation, **not** proof of a speedup over the A/B/B/A
  measurements. No whole-workspace, rendered or cross-platform acceptance
  is claimed.

Current constitutive WGSL SHA-256:
`726c087ddf9d6cf8c7c69a47f7ec63fa9897d4c80d23360018891171d0e97e5e`.
Current material-assembly WGSL SHA-256:
`22620bd97631919686223c5fb3b5de3a232283df36408973a573c7bc3456302e`.

### Model authoring and lifecycle checkpoint

The public Model/State path now accepts immutable `ElasticElementSet`
snapshots. Builder methods add membranes, tetrahedra or authored inverse
rest coordinates without replacing existing spring/tet energies. Boundary
triangles omit shared tet faces and deduplicate geometry, not material
energy contributions. `Model::set_elastic_elements` validates a replacement
before publishing it.

Purely elastic legacy Model tets retain their inverse rest matrices and
rest-volume overrides. Their Stable Neo-Hookean mapping uses the stored
`mu` and `lambda + mu`, matching the existing VBD elastic formula. Legacy
viscosity, plastic parameters and nonidentity plastic state are explicitly
refused; the new material registry is also refused by the other solver
implementations rather than silently ignored. This does not establish PPF
scene-parameter equivalence or material calibration.

Buffer identity/generation stamps and immutable registry snapshots detect
same-count material changes without per-element hashing every step. The
GPU facade rebuilds a candidate from caller State after external motion
edits and publishes that candidate only after successful validation and
readback. Material-only CPU rebuilds preserve cached affine-body motion;
inverse inertia is consumed live for body reactions. Same-length edits to
raw shape-geometry and soft-surface vectors still require
`BarrierSolver::reset()`. Full lifecycle/device-loss qualification remains
open, and the facade still allocates its two vector readback buffers.

Focused regression coverage includes buffer revision semantics, boundary
deduplication, all seven authored CPU material/dimension combinations
against the direct shell from a deformed first State, affine trajectory
preservation across registry replacement, and authored/legacy GPU material
steps with external State and same-count material edits. Broader authoring
failure cases, plastic-state cache invalidation, CPU body-pose resets and
steady-state facade resource qualification remain to be expanded.

Pre-commit checks passed on RTX 4080/Vulkan/595.112.07: the CPU PPF,
constitutive and input lanes; the new storage, registry and CPU publication
tests; GPU constitutive, coupled material, publication and all 11 existing
barrier regressions. Formatting and `git diff --check` passed. Evidence is
in `/tmp/molla-precommit-cpu.log`, `/tmp/molla-precommit-updated.log` and
`/tmp/molla-precommit-format-check.log`. These are focused checks, not
whole-workspace, rendered, cross-platform or renewed performance acceptance.

### Next required work

1. Resolve the measured staging latency/setup tradeoff during P6 qualification;
   retain the baseline and do not infer speed from allocation counts.
2. Extend Model/State authoring and lifecycle qualification, then continue
   P2-P4 with bending, damping, plasticity,
   principal-stretch limits and directional dynamic stiffness.
   Keep remaining P1 geometry/lifecycle/device-loss qualification explicit.
3. Qualify validator overhead on larger/deformed/dense workloads and other
   hardware. Preserve the final-state gate while reducing synchronization and
   traversal cost; the cached initial contact result requires immutable input.
4. Complete P5-P7; input validation and transactional publication do not fulfill
   material completeness, moving-boundary coupling or PPF scene parity.
