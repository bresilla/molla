# Cuttable elastic branches and fiber fracture

## Required end state

Implement the user's clean cuts **at the actual blade position**, together
with progressive stress-driven fiber breakage, in the existing Cable Lab.
Branches bend, twist, oscillate and recover toward an authored material rest
shape; unrooted pieces inherit motion and fall with contact. Repeated cuts,
oblique cuts, partial attachment and branching are part of the contract.
Never substitute nearest-node/joint breakage, zeroing velocities, fixed-world
rest-position springs, or unrendered proxy tests for these requirements.

The earlier PPF goal was cleared. Preserve its dirty work without claiming
that its unverified adaptive strain implementation is accepted. There is no
authorization to commit or push these new changes.

## Sources and boundaries

Sources are acknowledged in `ACKNOWLEDGEMENT.md` [1, 3, 4, 11–22]. EvoEngine's
current source is restricted: independently implement the published physical
ideas; do not copy/translate its implementation or redistribute its assets.
Reuse Molla's math, state, compute, collision, mass-property and solver
infrastructure. Reusable physics belongs in `crates/`, interaction/rendering
in `bin/molla_cables`. No importer/FFI/CUDA runtime expansion.

## Design

- An oriented segment is a finite convex fiber volume with position,
  orientation, linear/angular velocity, density, mass and inertia. Elastic
  connections use local attachment/rest frames and independent translational
  and bend/twist compliance. A bundle adds inter-fiber connections; connected
  components, not predetermined branch joints, determine detached pieces.
- Shape data retains polygon faces, including tagged cut caps. Plane clipping
  acts on the actual volume, not only its centerline: an oblique cut must
  produce coplanar exposed faces. Compute the two child volumes and inertias
  from clipped geometry. The implementation is a finite discretization, not
  a claim of continuum wood fracture.
- A clean cut intersects the chosen blade plane/region with every affected
  fiber and splits crossing pieces. Preserve independent authored rest and
  current configurations. Transfer rigid velocity fields to new centers of
  mass so mass, linear/angular momentum and pre-cut rigid kinetic energy are
  conserved for an ideal zero-kerf cut. Remove all connections across the cut;
  remap surviving local anchors. Tiny/invalid fragments are explicitly handled,
  never silently snapped or deleted. A future finite kerf must account for
  removed mass separately.
- Progressive breakage evaluates elastic connection loading and material
  strength/damage. Fibers can fail at different times; surviving bonds keep
  a partial break attached. Fracture is not implemented by random deletion
  or by routing a stress trigger through a whole-bundle clean cut.
- Mutations are transactions at step boundaries. Validate geometry, transferred
  state, graph/coloring and GPU capacity before publishing a replacement.
  Clear stale multipliers, rebuild affected contact exclusions and expose cut
  surfaces. Do not add overlapping spherical endcaps to touching cut planes.
- CPU is the oracle; wgpu kernels perform ordinary dynamic steps and failure
  detection. CPU geometry/topology work is permitted on cut events, not as
  hidden per-fiber host work every step. Reuse constraint coloring. Profile
  stiff-chain convergence and introduce an appropriate global/block solve if
  local iteration cannot meet the physical and performance gates.

## Acceptance ledger

| ID | Requirement | Evidence required | Status |
|---|---|---|---|
| F0 | Sources and executable verification | Citations, native Make targets, baseline logs | In progress |
| F1 | Exact geometric cut foundation | Off-node/oblique/repeated cuts, closed caps, positive child volume, mass/inertia/momentum conservation, atomic rejection | Passed CPU geometry gates |
| F2 | Elastic oriented fibers | CPU finite differences, objective rest-shape return, bending/twist, damped oscillation, timestep/resolution convergence | CPU oracle passes; spatial/stiff-chain convergence pending |
| F3 | GPU dynamic path | CPU/GPU agreement, pinned-root and free-flight invariants, invalid-state refusal, no silent CPU fallback | GPU fixture gates pass; broader convergence/platform acceptance pending |
| F4 | Cut transaction | Exact blade placement, all cross-cut constraints removed, remaining bonds retained, state transfer, repeated cut/reset, allocation failure rollback | Finite interface splitting, CPU/GPU cuts, and capacity rollback pass; interactive reset and stress qualification pending |
| F5 | Fiber bundle fracture | Multiple interacting fibers, load-dependent progressive damage, partial attachments, complete separation and branching | Tiled/staggered bundles, cuts and stress-fracture demo pass; branching-tree construction pending |
| F6 | Contact and performance | Ground/obstacles and piece-piece contact, no cut-induced explosive overlaps, resident resources, measured interactive latency and cut cost | Open |
| F7 | Interactive acceptance | Cable Lab cut/break/push controls, visible capped surfaces, real GPU rendered captures and interaction tests, launched viewer | Click/drag cuts, manual uneven breaks, push, Escape, reset and captures verified; full-scene/contact/performance acceptance remains open |

## Verification workflow

Use `NVIDIA_VERSION=595.112.07 nix develop --impure -c make ...` on this host.
Use targeted `make test APP_TARGET='...'`, `make test-gpu P=... T=...`,
`make fmt-files FILES='...' FMT_ARGS='--check --config skip_children=true'`.
The current unmodified Cable Lab can be run with `make run-cables` and its
old VBD smoke suite with `make test-cables`; these are not fracture gates.
The live viewer from the previous request may still be running; do not use
concurrent GPU timings as isolated benchmark evidence.

## Progress

- Source inspection confirmed that existing VBD cable bending is a
  zero-rest second-difference penalty, not a framed/rest-curvature rod model.
  Existing runtime joint breakage operates on authored joints, not interior
  cutting. Neither alone fulfills the user's requested feature.
- Initial implementation begins with convex fiber clipping and volume/mass
  conservation; remaining acceptance items above are not reduced to this
  foundation.

### 2026-11-15: CPU cut geometry and oriented-fiber checkpoint

- `molla-geometry::convex_cut` clips closed convex polygonal cells against
  the actual plane, retains tagged caps, and rejects unresolved slivers
  rather than snapping the blade or dropping their mass. Roundoff-scale
  coplanar vertices are projected onto the unchanged blade plane.
- Cut mass properties use normalized tetrahedral moment integration with
  full tensors. An initial reuse of the existing general Parry-backed
  mass-property path failed the tight additivity test by approximately
  3.7e-9 relative inertia on an axisymmetric case. The new cell-specific
  integration passed without weakening the conservation gate. The general
  collider path is unchanged; its 8 regression tests still pass.
- `molla-solvers::strands` owns finite-volume fibers, independent authored
  rest/current frames, local anchors, anisotropic translation and principal
  rotation-vector compliance, graph coloring, pins, and progressive bonds.
  Exact residual Jacobians pass central finite differences. Rotation
  residuals use SO(3) logarithms, including restoring response near a half
  turn; they do not represent accumulated multi-turn twist in a single bond.
- CPU dynamics are an XPBD reference with COM-centered rigid prediction,
  preserved free-body angular momentum, compliance-scaled multipliers,
  relative damping, and convex-vertex floor contact. No claim of a direct
  stiff-rod solve, pairwise collision, CCD, or GPU execution is made yet.
- Damage is an uncalibrated reduced-section model: remaining stiffness and
  strength scale with `1-d`; overload of the remaining section advances
  damage monotonically. Failure removes only that bond, leaving other
  connections intact. This is not a validated wood fracture law.
- Cut preparation leaves the source unchanged, conserves parent rigid
  velocity fields, preserves surviving bond rest data, and keeps only the
  root-containing child pinned. Commit rejects stale transactions, including
  those from another model or a divergent clone. GPU allocation rollback is
  not implemented yet.

Verified commands on this checkout:

```sh
NVIDIA_VERSION=595.112.07 nix develop --impure -c make test-strand-cuts test-strands
NVIDIA_VERSION=595.112.07 nix develop --impure -c make test APP_TARGET='-p molla-geometry --test collider_mass --test collider_mass_wake'
```

Results: 6 geometry tests (including more than 600 scaled/vertex-plane cuts),
13 strand tests, and 8 existing collider-mass tests passed. Logs:
`/tmp/molla-strands-foundation-final.log` and
`/tmp/molla-strands-mass-regression.log`.
The strand suite covers exact cut placement, surviving bonds, state and
momentum transfer, objective curved rest shapes, oscillation/recovery,
near-half-turn twist, analytic-oscillator timestep refinement, partial
attachment after load-driven failure, atomic input errors, and a cut fragment
falling onto its actual polygonal faces.

Strict targeted Clippy currently stops on three existing
`neg_cmp_op_on_partial_ord` diagnostics in
`molla-geometry/src/narrow_phase.rs:1541,1554`; no unrelated collision code
was changed to hide that baseline failure. A warning-mode diagnostic run
also reports an existing `manual_range_contains` in
`molla-solvers/src/barrier_contact/validation.rs:340`, and no diagnostics in
the new strand code. This is not a passing strict Clippy gate.

The public facade re-exports `molla::geometry::convex_cut` and
`molla::solvers::strands`; the reusable implementation remains in shared
crates rather than the example.

Next: GPU-resident ordinary dynamics/failure path with CPU parity and fault
handling, bundle construction with spatial/stiff-chain convergence tests,
piece contact, finite blade selection, and the actual Cable Lab integration.
The running old Cable Lab has not acquired cutting controls yet. No new
commit or push has been made.

### 2026-11-15: resident GPU path and finite bundle interfaces

- `molla-kernels::{strands, shaders/strands.wgsl}` and
  `molla-solvers::strands::GpuStrands` implement resident prediction,
  color-batched XPBD connections, convex-vertex floor contact, velocity
  recovery, section-damage evaluation, damping, friction and publication.
  CPU damping now follows the same graph-color order. Shader faults are
  sticky and prevent publishing any portion of an invalid step;
  `last_valid_snapshot` exposes the retained state plus explicit fault bits.
- Resources and a single staging buffer are retained. Ordinary steps upload
  48 bytes of global parameters, not per-fiber state. A 24-step resource
  counter gate verifies zero readback and zero Molla buffer-allocation
  observations. Geometry/state readback happens at requested snapshots and
  topology edits. This is not yet an isolated latency/throughput benchmark.
- `Engine::checked_gpu_setup` catches setup validation/allocation/internal
  errors and device loss. GPU cut replacement constructs new resources
  before replacing the live solver. Capacity refusal, setup validation
  errors, wrong-device use, unrepresentable parameters and numerical GPU
  faults have dedicated tests. Mutable solver access serializes submissions
  and use of its shared readback buffer. Actual hardware OOM was not induced.
- `BundleDesc` builds hexagonally tiled strands, staggered interior segment
  boundaries, longitudinal material-scaled bonds and side-face connections.
  Its beam/section and interface parameters are an uncalibrated discrete
  model, not certified wood material constants.
- The first oblique bundle cut exposed an actual topology bug: a plane
  crossing a segment joint produced 3 components instead of the intended 2.
  A point anchor cannot represent both surviving portions of a finite
  interface. `BondPatch` now retains corresponding interface polygons in
  both material frames. Cutting clips both sides, removes cross-blade
  portions, splits surviving bonds, transfers their rest frames and damage,
  and scales translation/strength by remaining area and rotational
  compliance by section moments. Staggered/oblique/repeated cuts now pass
  without discarding the unintended fragment. Bare `Bond::new` remains a
  point connection; finite-volume interfaces must attach a `BondPatch`.
- Interface rotational compliance remains diagonal in the material frame;
  arbitrary clipped sections can have off-diagonal section moments. This
  approximation and spatial/stiff-chain convergence still need qualification,
  rather than claiming continuum fracture or exact constitutive equivalence.

Checkpoint evidence:

```sh
NVIDIA_VERSION=595.112.07 nix develop --impure -c make test-strands
NVIDIA_VERSION=595.112.07 nix develop --impure -c make test-strands-gpu
```

`/tmp/molla-strands-patch-final.log`: 5 bundle and 13 CPU strand tests pass.
`/tmp/molla-strands-gpu-patches.log`: WGSL validation and 11 real GPU tests
pass on RTX 4080/Vulkan 595.112.07. Maximum observed CPU/GPU center error in
those fixtures was 0.000124 m; the cut bundle fixture differed by 0.00000293 m.
The bundle patch test verifies 4 successive oblique cuts produce exactly
5 components, preserve mass and authored zero-strain rest configurations,
and conserve split-interface translation stiffness and nominal strength.

Remaining full-goal work: stiff-chain/spatial convergence, long-running
bundle stress fracture and branching, piece-piece and obstacle contact,
finite blade selection, Cable Lab integration, actual interaction/rendered
evidence, and isolated performance qualification. The old live Cable Lab
process is unchanged and still has no cutting UI.

### Loaded bundle diagnostic: historical blanket-gap gate

The 56-cell / seven-strand loaded bundle ran 480 GPU steps with 32 local
iterations, but its peak material-interface gap was **0.0056576 m** and its
tip dropped to **1.12937 m** from 1.5 m. End-to-end time, including 60 snapshots
and while the old viewer remained open, was **10.355 s** for 2 s of simulated
time. This is diagnostic evidence, not isolated performance acceptance.
The first loose 1 cm smoke gate passed. A blanket **0.0001 m** gap gate then
failed. That was initially interpreted as proof of underconvergence, but
the coupled oracle subsequently showed that the gate conflated intended
axial elastic strain with solver error. The qualification below supersedes
that interpretation; the original failing measurements remain recorded.

Next solver action: a coupled/global constraint solve with CPU oracle and
resident GPU convergence checks, using the existing mass/Jacobian and
linear-solver infrastructure where applicable. Candidate infrastructure was
inspected: `coupled_constraints.rs` has a CPU dense Cholesky bilateral solve;
`coupled_constraint_gpu.rs` wraps retained Gram minimization, but its dense
constraint-space layout may be excessive for a fiber bundle;
`molla-kernels/src/bsr_solve.rs` performs host-side Krylov recurrence and
per-iteration transfers, so it cannot be silently used for the resident
ordinary-step path. A body-space matrix-free coupled solve avoids that
transfer and the larger redundant constraint-space matrix. Qualify it
against the loaded interface gate, timestep/spatial refinement and elapsed
performance before claiming a rigid branch demo.

Loaded diagnostic log: `/tmp/molla-strands-gpu-loaded.log` (the first smoke
threshold). The strengthened named test is part of `make test-strands-gpu`;
the earlier 11 GPU tests remain separate correctness evidence, not proof
that this new stiffness/performance gate is satisfied.
The blanket-gap test was executed and failed at the same
0.0056576 m gap: `/tmp/molla-strands-gpu-stiffness-gate.log`. This is the
historical failed diagnostic, not an external blocker or a current physical
acceptance criterion.

### 2026-11-15: coupled body-space solver checkpoint

- `CoupledConfig` selects a global frozen-Jacobian XPBD solve in body space:
  `(M + J^T alpha^-1 J) delta = -J^T (C / alpha + lambda)`, followed by
  `lambda_new = -(C + J delta) / alpha`. Here `alpha = compliance / dt^2`.
  Pinned coordinates have zero Jacobians and identity mass blocks. Default
  local projections remain available; coupled execution is explicit opt-in.
- The CPU oracle uses Molla's existing dense Cholesky routines, diagonal
  equilibration, finite checks and a backward-residual check, with an
  explicit 4096-coordinate capacity bound. Its axial-chain result matches
  an independently assembled tridiagonal implicit solve to 1e-12 m.
- WGSL uses a resident matrix-free PCG solve and full 6-by-6 block-Jacobi
  Cholesky preconditioning. One cooperative workgroup spans the body array;
  rows, adjacency, Krylov vectors and factors stay resident. Actual residuals
  are recomputed before accepting convergence, with a restart when the
  recursively updated residual was overoptimistic. Default linear tolerance
  is 1e-5 in the preconditioned norm, capped at 256 iterations per solve.
  Nonconvergence sets sticky fault bit 4 and retains the previous complete
  state, rather than silently accepting the result or falling back to CPU.
- A first f32 matvec form stalled at a true relative residual of 1.40e-5.
  Evaluating row products through relative body translations/rotations
  reduced cancellation enough to pass the original 1e-5 tolerance. The
  tolerance was not relaxed. `last_coupled_diagnostics` exposes the last
  read-back linear iteration count and residual, not unsynchronized work.
- The CPU constitutive residual gate exposed tiny-rotation quantization in
  the previous `to_scaled_axis` route. An atan2/series rotation logarithm
  now resolves 1e-12-radian twists and restores constitutive convergence;
  positions and velocities both use that continuous principal logarithm.
- The loaded bundle's exact CPU solve produces roughly 0.98 mm of axial
  connection extension within 24 steps at the specified material compliance.
  A blanket 0.1 mm gap limit would reject the intended elastic model. The
  replacement gates check the actual constitutive equations, independent
  axial-chain solution, explicit GPU linear residual, and CPU/GPU position
  error below **0.0001 m**, without changing material stiffness. A negative
  control requires the expected compliant extension to remain present.
- The 56-cell loaded GPU fixture matches CPU centers within 0.00000035 m
  over its first 24 steps and remains finite/connected for 480 steps. The
  final tip is at 1.44043 m, versus 1.12937 m for 32 local iterations. The
  different response demonstrates why local and global convergence cannot
  be inferred from the visual gap alone. Long-horizon CPU parity and spatial
  refinement remain separate open gates.
- Parameters are now 64 bytes per step. Coupled cut-state parity and scoped
  resource counters verify zero ordinary-step readback and zero observed
  Molla buffer allocations. The new shader requires 13 storage bindings;
  adapter limits and retained-buffer sizes are checked at setup.
- Citations [19] (Shewchuk's PCG reference) and [20] (glam dependency) were
  added to `ACKNOWLEDGEMENT.md`. No outside implementation was copied.

This is still **not interactive-performance acceptance**: the loaded test
took about 16.8 s including 24 CPU reference steps, 480 GPU steps and 60
snapshots, while the old viewer was running. The cooperative solver needs
profiling/optimization for real-time workloads and larger bundles. The old
Cable Lab remains unchanged; finite-blade interaction, branching and stress
demonstrations, piece contact and actual rendered acceptance are still open.

Verified checkpoint commands:

```sh
NVIDIA_VERSION=595.112.07 nix develop --impure -c make test-strand-cuts test-strands test-strands-gpu
NVIDIA_VERSION=595.112.07 nix develop --impure -c make check APP_TARGET='-p molla --no-default-features'
```

`/tmp/molla-strands-coupled-final.log`: 6 geometry, 6 bundle, 15 CPU strand,
1 shader-validation and 14 real GPU tests passed. The obliquely cut bundle
passes both local and coupled CPU/GPU checks; coupled center error is
0.00000113 m. That repeat loaded run took 19.4 s, reinforcing that the
non-isolated timings are diagnostic rather than a performance gate.
`/tmp/molla-strands-coupled-facade.log`: public CPU-only facade check passed.
Warning-mode Clippy (`/tmp/molla-strands-coupled-clippy.log`) reports no new
strand diagnostics, but the pre-existing workspace warnings remain; this
does not replace a strict lint gate. No commit or push was made.

### 2026-11-15: interactive elastic branch and click tools

The user specifically requested a more elastic branch and selectable **Cut**
or **Break** tools that operate with a single click at the selected position.
The running Cable Lab now has five scenes, keeping the original three VBD
examples and adding an elastic cuttable branch and a stress-fracture scene.

- The branch default now uses E = 30 MPa and G = 12 MPa, 30% of the earlier
  demonstration moduli, and damping 0.6 instead of 1.0. A same-impulse
  comparison measured peak tip travel of 0.67653 m versus 0.22553 m for the
  previous stiffness over 96 steps. The on-screen rigidity multiplier changes
  compliance without rebuilding the authored rest shape or resetting motion.
- Cut [C] and Break [B] both pick the actual convex surface under the cursor.
  Click Cut uses a plane through that hit perpendicular to the picked fiber's
  current local axis. Click Break uses deterministic, staggered per-fiber
  fracture planes around the same hit; the picked fiber's plane passes
  exactly through it. Neither operation snaps to a simulation node. Only the
  picked connected component's local section is targeted by the click tool.
- `StrandModel::prepare_fracture` and `GpuStrands::fracture` expose the
  prescribed noncoplanar operation in reusable crates. Paired interface
  polygons clip against separate planes on each side, retaining same-region
  bonds and removing bridges across the rough surface. Caps are tagged
  `FaceOrigin::Fracture`, distinct from clean-cut caps. CPU tests verify
  nonplanarity, two components, rest consistency and conservation of mass,
  linear/angular momentum and inherited rigid kinetic energy.
- Manual Break is explicitly a prescribed topology edit, **not** a
  stress-predicted fracture mechanism. The separate stress scene overloads
  bonds through the ordinary GPU dynamics; its verification observed partial
  damage, a still-attached partly broken bundle, then 7 failed bonds and two
  components, without invoking either topology tool. Its strengths are
  deliberately lower demonstration values, not calibrated wood constants.
- Cut still supports finite angled drag strokes. Escape cancels a gesture.
  Push [P] applies an impulse at the picked surface; pluck applies a batch of
  mass-scaled impulses. `GpuStrands::impulses` batches these atomically, with a
  regression proving that an invalid later item cannot publish an earlier
  item's change. R resets topology, caps, motion and tool-operation counters.
- Actual convex face triangles and exposed caps are rendered. Front-face
  culling and surface-preserving triangle subdivision reduce artifacts of
  the existing small painter-sorted inspector; this remains a lightweight
  renderer, not a general-purpose depth-buffered graphics engine.
- Two real regressions were found using the softer, moving, obliquely cut
  bundle. Slivers made the f32 block preconditioner numerically indefinite;
  diagonal equilibration and a 1e-5 shift **only to the preconditioner** fixed
  that without changing the physical operator or the 1e-5 residual tolerance.
  Single-lowest-point floor contact then exceeded the penetration bound.
  CPU/GPU floor contact now keeps an independent multiplier for each actual
  convex vertex and applies contact-specific friction. The same cut-and-fall
  test passes, with final minimum clearance about -1.2e-8 m. No fragments were
  snapped, removed or given spherical replacement endcaps.
- egui/eframe and image are credited in `ACKNOWLEDGEMENT.md` [21–22]. The
  viewer links to that local document rather than an unpublished remote
  revision. All scene geometry is procedural; no EvoEngine material copied.

Verification on RTX 4080/Vulkan:

```sh
NVIDIA_VERSION=595.112.07 nix develop --impure -c make test-strands test-strand-cuts test-strands-gpu test-cable-branches test-cables
WAYLAND_DISPLAY=wayland-1 DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 NVIDIA_VERSION=595.112.07 nix develop --impure -c make test-cable-click-ui CABLE_ARGS='--screenshot /tmp/molla-branch-click-ui.png'
```

`/tmp/molla-branch-final-regression.log`: 7 bundle, 15 CPU strand, 6 geometry,
1 shader, 15 real GPU and 3 viewer geometry/picking tests passed. Both branch
verification scenarios and all three original VBD scenes also passed.
`/tmp/molla-branch-click-ui-final.log`: native input-hook events exercised
one click Cut, one click Break, a cancelled blade gesture, a surface push,
falling pieces, framebuffer capture and keyboard reset. This uses the real
eframe input/render path and GPU physics; it is not a manual usability test.
The captured `/tmp/molla-branch-click-ui.png` was visually inspected.

The updated viewer was launched with `make run-cables CABLE_ARGS='--scene 3'`;
live process PID 212209 was verified as a visible native `molla-cables`
window on workspace 4. Runtime log: `/tmp/molla-branch-live.log`.
Process IDs/workspaces are historical evidence and must be rechecked later.

The **full goal is still active**: branching-tree construction, piece-piece
and obstacle contact, broader timestep/spatial/stiffness qualification and
interactive performance remain incomplete. The preview openly labels its
floor-only contact and slower-than-wall-time simulation. No commit/push.

### 2026-11-15: convex contact implementation in progress

The existing GPU GJK/EPA implementation accepts analytic primitives, not
arbitrary convex-hull vertex tables (`molla-geometry/shaders/gjk.wgsl`). It
cannot directly collide the clipped strand volumes. The new discrete path
uses face/edge SAT and reference/incident-face clipping, reusing the existing
CPU contact reducer. Sources [23] and [24] are added to ACKNOWLEDGEMENT.md.

Contact is being developed as explicit opt-in, with bounded resident pair
storage and errors on exhausted capacity. It will exclude only directly
intact bonds, not entire connected components. This is not CCD, and neither
the broad-phase scaling nor contact-scene acceptance is established yet.

### 2026-11-15: user correction — same branch, denser fibers

The user explicitly wants Cut and Break to have identical branch dynamics;
only the separation surface should differ, with more fibers in both. The
previous scene 5 used lower strengths and automatic overload damage, unlike
scene 4. Both visible branch presets now construct the same model and only
preselect different tools. The automatic stress fixture is retained in
headless verification, not disguised as a different Break-mode material.

The preview increases from 7 to 37 longitudinal fibers (56 to 296 cells),
with radius scaled by sqrt(7/37) to preserve area and mass. Uneven manual
breaks use deterministic per-fiber offsets in the authored cross section;
the picked fiber still splits exactly through the clicked surface point.
Dense-bundle GPU/UI qualification is in progress.

Contact work is parked while addressing this correction: the CPU geometry
oracle passes four tests including 600 rotated cases against existing hull
GJK. A CPU dynamics integration is added but not yet qualified; GPU steps
explicitly reject opt-in pair contact instead of silently ignoring it.
No GPU pair-contact implementation or viewer contact claim is made.

Subsequent user requests explicitly require piece-piece collision, the same
click **and drag** interaction for Break as Cut, additional elasticity,
video-quality presentation and responsive playback. These supersede the
earlier preview-only interaction scope, not the full physics acceptance gates.

The visible presets now use E=15 MPa, G=6 MPa, damping=0.35 and no spontaneous
failure. A 37-fiber E=100 MPa comparison stalled at a true PCG residual near
1.02e-5 against the unchanged 1e-5 tolerance. Increasing the linear cap and
compensated arithmetic experiments did not establish a fix and were not
adopted. Broader stiff/dense qualification is still open. The requested
softening comparison is now against the immediately previous E=30 MPa.

Performance discovery: each GPU snapshot was revalidating immutable hulls
and recomputing all cell mass tensors on the CPU. Snapshot validation now
checks only changing motion/pins; full geometry/mass validation remains at
construction and topology transactions. This change awaits regression and
timing qualification. The viewer also advances only 2/240 simulated seconds
per frame, which is half speed even at 60 FPS; scheduling and rendering work
remain pending.

Convex pair contact now has matching CPU and GPU code. Three CPU fixtures
cover impacts, a fixed convex obstacle without a floor, and re-enabled
contact after a direct bond breaks. Two real GPU tests pass for surface
impacts/obstacles and atomic contact-capacity rejection. An initial single
velocity sweep produced order-dependent torques and failed the 3 mm
CPU/GPU trajectory gate. Twelve normal-velocity sweeps followed by one
Coulomb-bounded friction pass corrected that fixture; the gate was not
relaxed. Scale-aware contact-reduction tie handling preserves the existing
generic reducer's default behavior. Log: `/tmp/molla-contact-gpu.log`.

The viewer enables bounded all-pairs contact with 262144 slots and e=0.15.
Ordinary contact work remains GPU-resident; pinned convex cells can act as
obstacles. Dense-scene acceptance, active-contact performance and CCD are
still unproven. The denser preview explicitly uses a PCG tolerance of 2e-5;
the reusable solver default and its existing gates remain 1e-5. A dense
CPU-oracle accuracy check is required before accepting this preview setting.

### 2026-11-15: focused speed request, broader goal paused

The user paused the full fracture goal and explicitly prioritised substantial
speed gains over fiber count. The work now targets the existing preview,
not additional fracture/visual features. No commit or push is authorised.

Measured debug baseline, 12 samples of two steps plus a snapshot, with all
contacts enabled: 37 fibers took 249/253 ms median intact/cut; 19 took
109/111 ms; 7 took 37/34 ms. `/tmp/molla-speed-before.log` records the run.
GPU-resident active-pair coloring replaces all-pairs round-robin projection
rounds; CPU oracle ordering matches the same first-fit contact graph. The
37-fiber result fell to approximately 195/199 ms without disabling contact.
The default is now seven fibers / 56 cells, preserving cross-sectional area,
mass, elasticity, exact cuts and uneven breaks. A small-system shared-vector
cache is under evaluation; it has not demonstrated a distinct speed gain.

Timing breakdown identified substantial **debug CPU submission** overhead:
roughly 12–18 ms per two-step batch even for seven fibers, in addition to GPU
wait/decode. An optimised viewer build is being measured rather than claiming
real-time performance from the debug measurements. Source: renderer and
collision references remain [10], [21], [23], [24].

Previously completed dense accuracy check: the 296-cell, 30 MPa one-step
GPU solve at preview tolerance 2e-5 matched a full double-precision Cholesky
oracle within 0.301 micrometres in position and 0.000071 m/s in velocity;
reported GPU residual 1.800388e-5. `/tmp/molla-dense-oracle.log`.
This does not resolve the separate dense angled-cut floor-penetration failure.

### 2026-11-15: separate Appleskel apple-tree example

User request: create another example in a different file using the skeleton
from `../../../../OUSD/appleskel/`. The broader full-fracture goal remains
paused; this is a separate, explicitly requested example. No commit/push.

Implemented `bin/molla_cables/src/apple_tree.rs` as the `molla_apple_tree`
binary, plus `apple_tree_model.rs`. The original cable binary and its five
presets remain separate. Both reuse the existing GPU strand solver and
studio renderer; no new physics solver or Python/Blender runtime was added.
Make targets import, run, verify, UI-verify and benchmark the tree.

Source: Appleskel's `tree1_characterization.json`, derived from Tian Qiu's
AppleQSM-Row13 dataset (CC BY 4.0). The embedded geometry subset and source
SHA-256 hashes are recorded in `bin/molla_cables/data/ATTRIBUTION.md`.
References [25] and [26] acknowledge the data, paper and Appleskel hierarchy,
smoothing and taper approach. No external images or foliage assets were used.

Default geometry retains a parent-connected subset of 20 of the 162 source
limbs, plus the trunk: 59 tapered octagonal solid cells. Each span is one
solid, not a dense multi-fiber bundle. Minimum radius 3.5 mm; roughly 28 cm
span spacing. Small twigs are omitted, not rendered as fake cuttable wood.
The root span is pinned. Click cuts split the actual hit cell; angled Break
is deliberately documented as a planar reduced-model operation, not fibrous
splintering. Detached descendants retain their joints and collide discretely.

The 1/120-second trial failed PCG after the native repeated-cut scenario;
the final tree uses 1/240 s, two nonlinear solves, eight contact iterations,
512 maximum PCG iterations and 2e-5 relative tolerance. Translation/angular
moduli are illustrative, separately chosen 20 MPa / 2 GPa. Core defaults
were not changed. Capped eight-step wall-clock catch-up avoids the earlier
four-step cap dropping fractional time at the 60 Hz display boundary.

Acceptance evidence (release, RTX 4080, Vulkan):
- Seven tree-binary unit tests pass, including actual off-node cap geometry,
  connected topology, bounded cell count and mass conservation.
- Headless CPU/GPU one-step maximum center error 0.844 micrometres.
- Headless sway, pinned root, no spontaneous failure, cut detachment and
  two seconds of falling motion/floor checks pass.
- Two steps + snapshot: median 7.150 ms, worst 7.558 ms over 20 samples;
  those two steps represent 8.333 ms of simulated time.
- Native tree UI: actual pointer Cut + Break, three pieces/four cap faces,
  two simulated seconds, floor clearance, screenshot and reset pass.
- Native cable Cut/Break/Push/Escape/reset regression passes. Its fixture
  now waits 20 frames for initial layout before injecting pointer events.
- Live tree capture: 303 UI passes, 5.242 wall seconds including startup,
  5.125 simulated seconds, recent playback 1.00x. Screenshot inspected at
  `/tmp/molla-apple-tree-live.png`; cut UI image at
  `/tmp/molla-apple-tree-ui.png`. This is not long-duration/worst-case proof.

The shared studio floor now receives shadows without casting into its own
shadow map; this removes the visible floor banding. A larger fogged floor
also removes the hard horizon edge. Both native examples were rerun.

Relevant files: `src/apple_tree.rs` (standalone app/verification),
`src/apple_tree_model.rs` (source-derived reduced hierarchy),
`data/apple_tree1.json`, `data/ATTRIBUTION.md`, `APPLE_TREE.md`, shared
`branch_draw.rs`/`studio.rs`/`studio.wgsl`, package manifest and root Makefile.
Next work, if requested: qualify larger budgets, long-duration collisions,
full fibrous tree fracture and the pre-existing dense angled-cut failures.
These are not established by this example's focused checks.

## Apple-tree error and junction repair — 2026-11-15

### Goal
Fix the reported sticky solver error first, then junction flicker. The old
viewer log reported PCG fault 0x4 after three drag cuts at t=954.66 s;
repeated Play attempts retained the same device fault. That exact long-run
state was not saved, so qualification uses cut/recovery regression fixtures.

### Instructions
Keep the separate example, exact cuts, GPU physics and CPU oracle. Use Make;
preserve unrelated dirty material/strain work and acknowledge references.
No commit or push was requested for this repair.

### Discoveries
- Two authored, non-direct-neighbor junction pairs overlap by 8.85 and
  13.01 mm while sharing an intact neighbor. Their contacts fought the joints.
- A purely relative convergence test is ill-conditioned near zero load.
  Preview absolute tolerance 1e-10 avoids repeated refinement of residuals
  already around 1e-11; relative tolerance remains 2e-5, library absolute
  default remains zero. CPU comparisons still gate the actual motion.
- Compensating only the solution and residual was insufficient for the
  strict loaded-bundle test. The final GPU PCG also compensates its RHS and
  matrix-vector products and uses a positive quadratic-form denominator.
- Free rotation now solves the body midpoint with Newton iterations and
  uses matching Cayley reconstruction. Failed rotation convergence has
  separate sticky bit 16; contact faults retain bit 8.

### Accomplished
- Whole-display-batch rollback and bounded finer-step replay preserve cuts,
  pins, topology and elapsed time. Recoverable faults are PCG/rotation;
  other faults restore the checkpoint and remain visible errors.
- Opt-in two-hop contact exclusions remove local intact-junction conflicts,
  not all same-component contacts. Detachment re-enables piece collisions.
- Buried junction display faces are clipped; coplanar exterior ownership is
  deterministic; actual cut/fracture caps remain. Procedural grain is filtered
  by pixel footprint. No extra decorative collision geometry was added.
- Ten tree unit tests, 26 CPU strand/bundle/contact tests, 17 GPU strand
  tests, three GPU contact tests and shader validation passed. Strict-default
  loaded-bundle and dense Cholesky-oracle checks were not loosened.
- One-step tree CPU/GPU center error 0.135 µm. Two-step plus readback median
  6.322 ms, worst 7.097 ms over 20 samples (8.333 ms simulated per sample).
- Acknowledgement references [27–29] cover WGSL derivatives, midpoint/Cayley
  reconstruction and compensated arithmetic; [17] also covers junction clipping.
- Final repeated-cut stress passed: 60 simulated seconds in 42.50 wall
  seconds, three natural refinements, four pieces, fixed root and floor.
  Injected sticky-fault replay and nonrecoverable rollback both passed.
- Final native tree Cut/Break/reset and original Cable Lab click/push/reset
  tests passed. Close-up live tree capture: 303 frames, 5.216 wall seconds,
  5.075 simulated seconds, recent playback 1.00x. Actual screenshots inspected:
  `/tmp/molla-apple-final-ui.png`, `/tmp/molla-apple-final-live.png`.
- One corrected release viewer was relaunched on workspace 4, title
  `Molla — Apple Tree`; startup and continued running checked without errors.

### Next Steps
Larger budgets, arbitrary severe cuts, CCD, calibrated wood and full fibrous
fracture remain separate scope; the older broader roadmap remains paused.

### Relevant Files
- `bin/molla_cables/src/adaptive_step.rs` — atomic batch replay and rollback.
- `bin/molla_cables/src/apple_tree_verify.rs` — recovery and stress fixtures.
- `bin/molla_cables/src/junction_surface.rs` — exposed junction display mesh.
- `bin/molla_cables/src/apple_tree.rs`, `apple_tree_model.rs` — preview setup/tests.
- `bin/molla_cables/src/branch_draw.rs`, `studio.wgsl` — rendering corrections.
- `crates/molla-solvers/src/strands/{cpu,contact,coupled,gpu}.rs` — matching solver/configuration changes.
- `crates/molla-kernels/shaders/strands*.wgsl`, `src/strands.rs` — GPU implementation/uniform layout.
- `crates/molla-solvers/tests/{strands_gpu,strand_contacts}.rs` — GPU/CPU regressions.
- `Makefile`, `bin/molla_cables/APPLE_TREE.md`, `ACKNOWLEDGEMENT.md` — workflow, evidence and citations.

## Distinct rough Break and load-driven Grab — 2026-11-15

### Goal
Make breaking visibly and physically different from clean cutting in the
separate apple-tree example. The user explicitly requested both immediate
rough fracture at the click and holding/pushing a branch until load breaks it.

### Instructions
Keep the original example separate, use Make, acknowledge research, preserve
GPU-first reusable physics and the CPU oracle, and do not edit with Python.
Preserve unrelated dirty PPF/material/strain changes. No commit was requested.

### Discoveries
- The old apple Break merely changed a cut plane. A different cap color or
  angle does not create an uneven fracture surface.
- Local convex decomposition avoids turning the whole 59-cell tree into a
  dense fiber model. Four longitudinal sectors with staggered fracture planes
  create up to eight cells in two assemblies, adding at most seven cells per
  selected span. Internal bonds keep each assembly together.
- Point grabs can use the existing step uniform and a GPU constraint dispatch;
  target updates need no per-drag resource rebuild. CPU and WGSL use matching
  accumulated compliant multipliers and linear/angular inertia.
- Refined fragments can keep requesting smaller timesteps after release.
  The preview now retains successful subdivisions (up to four), probing a
  coarser step after twelve clean batches instead of retrying every frame.
- The severe native drag still runs below real time. Background GPU workload
  also varied during this session; timings cannot establish isolated speedup.

### Accomplished
- Added transactional `prepare_splinter`/`commit_splinter` and GPU `splinter`.
  Children preserve mass, rigid motion, rest/material coordinates and point
  attachments. The picked sector passes through the requested hit. Capacity
  and stale-edit failures do not publish partial topology.
- Kept Cut smooth. Break now creates staggered, noncoplanar physical fracture
  geometry, not decorative triangles. Finite-interface attachments are
  explicitly rejected by this API; the reduced tree uses point attachments.
- Added Grab / bend [G]: hold the picked material point, drag in its initial
  screen-parallel plane, release to drop. Escape, focus loss, reset or tool
  change release it. Force/torque-driven bond damage triggers natural failure;
  failed ends receive local physical splinter geometry and chips. The grab
  follows its material point through refinement.
- New geometry is procedural and load-driven failure occurs at sampled
  joints. This is not continuous microscopic wood fracture or an EvoEngine
  port. No EvoEngine code/assets were copied. Updated acknowledgement [11]
  and [13] for the research relationship and compliant grab respectively.
- Increased the single coupled-solver workgroup to 128 lanes; retained strict
  tolerances, dense CPU oracle and resource checks. GPU program key is v16.
- Passed ten tree unit tests, 27 CPU strand/bundle/contact/splinter tests,
  shader validation, 17 GPU strand tests, three GPU contact tests and one
  GPU splinter/grab test. New tests cover nonplanarity, exact hit, conserved
  mass/momenta/energy, repeated edits, capacity rollback, invalid-grab atomicity
  and CPU/GPU dynamics. Existing strict loaded/dense tests were not loosened.
- New click-Break stress passed 20 simulated seconds in 26.78 wall seconds;
  gentle grabbing survived, strong grabbing caused GPU failure without a
  click operation, and released pieces fell onto the floor (73 cells/four
  components after the tested load-driven fracture).
- Legacy planar recovery fixtures remain explicitly planar, preserving the
  prior inputs and tolerances. They passed sticky-fault replay against CPU
  and fresh GPU, nonrecoverable rollback and 30 seconds of repeated-cut
  stress in 21.28 wall seconds. New rough geometry is qualified separately.
- Final native tests passed Cut/rough-Break/reset, actual mouse
  Grab/bend/fail/release/reset and the original Cable Lab click/push/reset.
  Screenshots `/tmp/molla-apple-rough-ui.png` and
  `/tmp/molla-apple-grab-ui.png` were inspected. Severe native drag advanced
  5.450 simulated seconds in 13.951 wall seconds, eleven recovery batches,
  final playback 0.62x; this is not real-time acceptance.
- Final intact-tree benchmark: 6.401 ms median / 6.884 ms worst for two steps
  plus readback (8.333 ms simulated), with one-step CPU/GPU error 0.135 µm.
- Relaunched one release viewer on workspace 4 (`Molla — Apple Tree`,
  PID 1436191 at verification); startup log and native window confirmed.

### Next Steps
Improve severe-grab/fragment contact performance without loosening correctness
gates. Qualify arbitrary repeated breaks, larger budgets, CCD, calibrated wood,
finite-interface splinter transfer and full progressive fibrous fracture
separately. Those are not established by this reduced-tree demonstration.

### Relevant Files
- `crates/molla-solvers/src/strands/splinter.rs` — local fragment transactions.
- `crates/molla-solvers/src/strands/{cpu,gpu}.rs` — CPU/GPU point grab API.
- `crates/molla-solvers/tests/strand_splinter.rs` — new geometry/parity/rollback tests.
- `crates/molla-kernels/shaders/strands.wgsl`, `src/strands.rs` — grab dispatch/uniforms.
- `crates/molla-kernels/shaders/strands_coupled.wgsl` — 128-lane PCG workgroup.
- `bin/molla_cables/src/apple_tree.rs` — tool actions, stress refinement, adaptive reuse.
- `bin/molla_cables/src/apple_tree_model.rs` — finite force/torque-driven damage.
- `bin/molla_cables/src/apple_tree_verify.rs` — gentle/strong grab and separate fixtures.
- `bin/molla_cables/src/branch_draw.rs` — shared interaction/rendering and close zoom.
- `Makefile`, `bin/molla_cables/APPLE_TREE.md`, `ACKNOWLEDGEMENT.md` — workflow, limits, sources.

## Grab stability and multi-branch qualification — 2026-11-15

### Goal
Fix jerky pulling/pushing without teleporting or suppressing physical motion.
The user explicitly rejected qualification on one branch in one direction.

### Instructions
Use Make and patch tools, preserve unrelated PPF work, retain GPU physics and
CPU oracles, acknowledge references, and test multiple branches/directions.
No commit or push was requested. Engram tools were unavailable in this session.

### Discoveries
- The first grab implementation projected a single body after the coupled
  branch solve. The grab and the connected constraints could fight each other.
- A raw cursor jump is an unbounded spring load. A distant pointer could
  stretch/bend small wood substantially even when its speed was filtered.
- Broad strong-drag tests exposed release recoil missed by the first tip-only
  fixture. Energy diagnostics distinguished real stored elastic energy from
  numerical energy creation: one loaded span held about 0.96 J, then lost total
  mechanical energy while converting it into fast recoil. Experiments changing
  axial stiffness, joint damping and coupled bond damping were not retained.
- The selected interaction uses a gentler default hand load and a visible grip
  strength control, rather than clamping body velocities or freezing releases.
- Very small PCG right-hand sides need the existing configurable absolute
  floor; new point-grab oracle fixtures use 1e-10. Existing strict-default
  loaded-bundle tolerances remain unchanged.

### Accomplished
- Point-grab rows now join the coupled CPU dense solve and GPU PCG, including
  preconditioner, compensated products/residuals and multiplier updates. The
  local solver retains its matching projection path. Shader cache key is v20.
- Added target velocity, implicit viscous point damping and optional combined
  force saturation to `PointGrab`. Extra frozen storage is three rows; ordinary
  target updates remain uniforms, without extra per-drag resource rebuilds.
- Added an analytic backward-Euler spring/damper check, bounded-force checks,
  CPU/GPU coupled-chain direction reversals and singleton/no-bond coverage.
- Mouse handle motion is simulation-time filtered at 1.5 m/s and 12 m/s²;
  adaptive replays follow the same interpolated target path. The hand uses
  compliance 0.002 m/N and damping 10 N s/m, with default grip limit 4 N and
  a 1–20 N slider. A tether marks the physical grab and filtered handle.
- Push is a 0.32-second controlled load, not a velocity impulse. Release,
  tool changes, focus loss and reset clear the handle. Grabbing limits display
  catch-up to four nominal steps. Successful refinement up to eight subdivisions
  is retained while holding; coarsening resumes after release.
- Bending strength is now illustrative 10 MPa so progressive failure can occur
  under the gentler hand load. Original axial/bending compliance and joint
  damping are preserved; no whole-tree stiffness/damping workaround remains.
- Added headless matrix and native test controls. Matrix gentle coverage is
  every movable span in six signed world axes; strong coverage is every sixth
  span with ramps and abrupt reversals. Tests measure motion, root, mass, floor,
  release and unintended failure; CSV files retain each result. Native fixtures
  can pick different ray-visible spans and screen directions.
- Intermediate 12 N gentle matrix passed 348 cases with no failure or recovery;
  six native gentle cases on cells 57, 49, 41, 32, 24 and 16 passed. Strong
  12 N recoil bounds failed and were not represented as acceptance. Final
  4 N / revised-strength broad qualification is recorded below.
- Final headless matrices passed: 348 gentle cases on all 58 movable spans
  in six directions, no fracture or recovery; 60 harder cases on ten spans,
  no failures, eight fractured cases and 23 recovered batches. Gentle maximum
  center movement was 7.725 mm per two-step batch and joint gap 1.519 mm.
- Final shader, 23 GPU strand/contact/splinter/dense-oracle tests and 29 CPU
  strand tests passed. Eleven tree unit tests, baseline sway/cut/fall, separate
  gentle/strong/Push checks, sticky rollback and 30-second recovery stress passed.
- Native hard-pull exploration showed that the original tip/rightward fixture
  does not fracture at 4 N; the same input does fracture at 8 N. This is a load
  threshold, not a solver failure. The default required-fracture fixture now
  picks span 49 downward at 4 N. General native hard-load sweeps may explicitly
  allow intact wood; they retain release/root/floor/error checks, and the
  fracture-required fixture remains separate. No physical parameter was changed
  to force every direction to fracture.
- Final native matrix passed all 13 cases: six ray-visible spans in different
  screen directions, gentle and hard inputs, plus the required-fracture case.
  Two of six hard cases fractured at 4 N; the others survived. Gentle runs
  advanced 5.450 simulated seconds in 5.918–6.184 wall seconds with no recovery.
  Fracturing native runs took 11.526–18.930 wall seconds, so heavy-fracture
  real-time performance is explicitly not accepted. Actual tether, span-49
  fracture and span-16 fracture framebuffer captures were inspected.
- Re-ran all eleven tree unit tests, explicit formatting and diff whitespace
  checks after the native-fixture changes. Original Cable Lab native checks
  also passed. Updated `APPLE_TREE.md` with exact commands, final results and
  limitations; XPBD acknowledgement covers the damping and own force limiter.
- Relaunched one release `Molla — Apple Tree` viewer on workspace 4 (PID
  1883569), with confirmed RTX 4080 startup and no startup error. Existing
  Gearbox windows/processes were left untouched. No commit or push was made.
- Evidence: `/tmp/molla-grab-final-{strong,gentle,core,recovery}.log`,
  `/tmp/molla-grab-final-native-matrix.log`, per-case CSVs and native captures
  under `/tmp/molla-apple-grab-ui-*`. These temporary paths are not durable assets.

### Next Steps
User interaction review on the reopened tree. Further optimize fragmented
high-angular-speed cases without sacrificing recovery/oracle guarantees;
qualify larger branch budgets, stronger grip settings and arbitrary prior
cuts separately. Do not claim all edits/loads are now real-time or error-free.

### Relevant Files
- `crates/molla-solvers/src/strands/{cpu,coupled,gpu}.rs` — point-grab solve/damping/limits.
- `crates/molla-kernels/shaders/strands{,_coupled}.wgsl`, `src/strands.rs` — GPU rows and dispatch.
- `crates/molla-solvers/tests/strand_splinter.rs` — analytic, force-limit and CPU/GPU tests.
- `bin/molla_cables/src/grab_handle.rs` — filtered hand and jump/reversal unit test.
- `bin/molla_cables/src/{apple_tree,apple_tree_model,apple_tree_verify,adaptive_step}.rs` — interaction, fixtures and replay.
- `Makefile`, `bin/molla_cables/APPLE_TREE.md`, `ACKNOWLEDGEMENT.md` — commands, behavior and attribution.
