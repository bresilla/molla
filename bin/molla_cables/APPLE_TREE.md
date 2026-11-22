# Separate apple-tree example

Entry point: **`src/apple_tree.rs`**, executable **`molla_apple_tree`**.
For the offscreen 1080p video exporter, see [`SHOWCASE.md`](SHOWCASE.md).
This is not another Cable Lab preset. `make run-cables` still launches the
original demo. Both executables reuse Molla's GPU strand solver, exact
convex cuts, picking helpers, and depth-buffered studio renderer.

```sh
make run-apple-tree
make run-apple-tree TREE_ARGS='--branches 12'
make test-apple-tree
make test-apple-tree-ui
make test-apple-tree-tools
make test-apple-tree-grab-ui
make test-apple-tree-grab-ui-matrix
make test-apple-tree-grab
make test-apple-tree-grab-matrix
make test-apple-tree-grab-matrix TREE_ARGS='--strong'
make test-apple-tree-recovery TREE_ARGS='--stress-seconds 60'
make bench-apple-tree
```

On this Nix/NVIDIA workstation, prefix with
`NVIDIA_VERSION=595.112.07 nix develop --impure -c` and use the desktop's
`WAYLAND_DISPLAY=wayland-1 DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000`.
Interactive and verification targets build optimized release executables.

## Geometry and physics

The bundled geometry comes from Appleskel's reconstruction of AppleQSM
Row13 **tree1**, with 162 source branch polylines and a reconstructed trunk path.
The default retains 20 long branches plus the trunk, including each selected
branch's ancestors within that budget. Paths are smoothed and sampled at
approximately 28 cm, producing **59 cells** by default. Each span is one tapered octagonal solid, not a bundle
of dozens of fibers. The first trunk span is clamped. Rest-pose elastic
joints connect the branching hierarchy; the same GPU solver runs both
intact and detached pieces, with discrete convex contacts and floor contact.

This is a reduced, bare pruning model: omitted twigs are not simulated or
silently shown as cuttable decoration. Radii use illustrative hierarchical
taper with a 3.5 mm minimum. Wood compliance and damping are uncalibrated.
The larger budget slider rebuilds the model and resets edits and motion.

**Cut [C]** clicks the actual wood surface and cuts just that cell at the hit,
not at its nearest endpoint. Downstream branches remain attached to the
appropriate half and fall with it. **Break [B]** immediately makes a rough,
nonplanar fracture around the selected position. Only the selected span is
partitioned into four longitudinal sectors with staggered fracture faces:
up to eight physical cells, joined into two fragment assemblies. This adds
at most seven cells per selected span, not a dense bundle across the tree.
The picked sector's fracture face passes through the actual hit. Mass and
rigid motion are transferred to all pieces; the teeth are collision geometry,
not decorative triangles. Dragging either section tool defines a finite blade.

**Grab / bend [G]** holds the actual material point under the cursor. Hold the
left mouse button and drag to bend, then release to let go. A compliant GPU
point constraint applies the load; existing GPU force/torque-driven bond
damage determines failure. Gentle loading can survive; sufficient loading
breaks the wood without invoking the click tools. Broken ends are locally
refined into rough geometry and physical chips. Escape or losing focus releases
the grab, and it follows the selected material point through refinement.
The grab participates in the same coupled solve as the connected branches,
instead of correcting one cell after that solve. Its compliance is 0.002 m/N
and viscous damping is 10 N s/m. A 4 N default combined force limit prevents a distant
cursor from applying an arbitrarily large spring load. A simulation-time mouse handle limits speed
to 1.5 m/s and acceleration to 12 m/s²; its path is interpolated over physics
substeps, including adaptive replays. The amber marker/tether identifies the
held point and filtered handle. Push applies a 0.32-second controlled load
through this same mechanism rather than an instantaneous velocity kick.
Interactive grabs cap catch-up at four nominal steps per display update,
keeping the mouse responsive rather than batching eight steps before redraw.
The Grip force slider adjusts that limit from 1 to 20 N. Stronger grip can
bend thicker wood, but stores more elastic energy and produces faster recoil.
The example uses an illustrative 10 MPa bending-strength scale; its elastic
compliance and axial-strength scale are unchanged.

Load-driven failure is resolved at the sampled elastic joints. The visible
splinter pattern is procedural, not a prediction of microscopic wood grain
or the complete EvoEngine fracture model. The strengths are illustrative.
**Push [P]**, **Breeze / sway**, **Space**, **R**, right-drag, and scroll
provide controlled pushes, a sway impulse, pause/reset, orbit, and zoom.

The timestep is 1/240 s with two nonlinear solves and eight contact iterations.
Translation and angular compliances use separate illustrative moduli (20 MPa
and 2 GPa); they are not a calibrated isotropic wood material.
The UI shows solver-plus-readback time and actual approximate playback rate;
catch-up is capped rather than accumulating an unbounded work queue.
Contact is discrete: fast motion may tunnel. This example does not resolve
the broader dense-fiber fracture roadmap's outstanding validation cases.

## Data and captures

`data/apple_tree1.json` is embedded, so running does not require the adjacent
Appleskel checkout, Python, Blender, or USD. Refresh it from that checkout
with `make import-apple-tree`, or load a compatible characterization JSON
using `TREE_ARGS='--skeleton /path/to/tree_characterization.json'`.

`TREE_ARGS='--screenshot /tmp/molla-apple-tree.png'` captures the intact,
paused tree's native framebuffer and exits. The UI verification target
clicks Cut and Break, advances GPU physics, captures the result, and resets.
Add `--live` to capture after 300 frames of the wall-clock-paced simulation.
Native grab checks accept `--grab-pick N` (ordinal among ray-visible cells),
`--grab-angle DEGREES` (screen direction), and `--gentle` (reversing elastic
motion without intended fracture). Each run logs the actual selected cell.
The default native fracture fixture uses pick 8, screen direction 90 degrees,
and requires load-driven failure at the default 4 N grip. `--allow-intact`
tests a hard pull without requiring that every branch/direction break at that
force. The native matrix runs six picks in different directions with both
gentle and hard inputs, then runs the required-fracture fixture separately.
The headless grab matrix covers all movable cells in six signed world axes;
`--strong` applies larger loads to every sixth movable cell. Both write
per-case CSV results under `/tmp/molla-grab-matrix-{gentle,strong}.csv`.
Use `--grip-force N` to select the UI/matrix grip strength; matrix reproducers
also accept `--grab-cell ID --grab-axis INDEX`, with axes ordered
`+X, -X, +Y, -Y, +Z, -Z`. Invalid selections fail rather than passing no cases.

## Solver and junction corrections

The tree excludes collisions within two intact joint connections. This removes
two overlapping authored junction pairs that previously pushed against their
own attachments; contact returns when a cut disconnects that neighborhood.
This is a local exclusion, not a blanket exclusion of all attached wood.
Buried junction faces are clipped from the display mesh; actual cut/fracture
caps remain visible. Pixel-footprint filtering reduces wood-grain shimmer.

The coupled GPU solve uses compensated solution/residual arithmetic and a
positive quadratic-form denominator. Its relative tolerance remains **2e-5**;
this example also sets an absolute preconditioned-residual tolerance of
**1e-10** for nearly zero loads. The library's absolute default remains zero.
Free rotation uses a Newton-solved body-frame midpoint with a matching Cayley
orientation update, rather than unchecked fixed-point iterations.

On convergence or excessive angular-sweep failure, the viewer restores the
last displayed state and replays the entire batch with smaller timesteps,
including steps queued before the fault. Cuts, topology, pins and elapsed
simulation time are preserved. Retries are bounded to six halvings; other
faults still pause with an error and restore a healthy checkpoint. This is
not a promise that arbitrary cuts, loads or branch budgets cannot fail.
The preview retains a successful refined timestep for subsequent batches
(up to eight subdivisions). While holding wood it retains that resolution;
after release it probes a coarser step after twelve clean batches. Spinning
pieces do not force the same failed coarse attempt every display frame.

## Multi-branch grab verification (2026-11-15, RTX 4080, Vulkan, release)

The final default is 4 N grip with the 20-branch / 59-cell tree. These checks
do not qualify every slider setting, larger branch budget or arbitrary edit.

- **348 gentle cases passed:** all 58 movable spans in six signed world axes,
  including return motion and release. No unintended fracture or adaptive
  recovery; maximum center movement was **7.725 mm per two-step batch** and
  maximum translational joint residual was **1.519 mm**.
- **60 hard-load cases passed:** ten distributed spans in six directions,
  including ramps, abrupt targets, reversals and release. Eight cases fractured;
  23 batches required recovery. Maximum center movement was **71.994 mm per
  two-step batch**. Hard fracture/recoil is not as slow-moving as gentle bending.
  Both matrices checked root, mass, floor and released-handle invariants.
- **13 native checks passed:** six actual ray-visible spans (57, 49, 41, 32,
  24, 16) in six different screen directions, each with gentle and hard input,
  followed by a separate required-fracture case. Gentle cases had no breaks
  or recoveries. Two of the six hard cases fractured; the other four survived
  the 4 N load. Requiring every branch/direction to break would be incorrect.
  Screenshots of the active tether and different released fragments were inspected.
- Gentle native runs advanced **5.450 simulated seconds in 5.918–6.184 wall
  seconds**, including startup. Fracturing runs took **11.526–18.930 wall
  seconds** for the same simulated interval. **Heavy-fracture real-time
  performance remains unresolved.** These are interaction checks with other
  desktop workloads present, not isolated before/after benchmarks.
- Eleven tree unit tests, 29 CPU strand tests, shader validation and 23 GPU
  strand/contact/splinter/dense-oracle tests passed. New checks cover analytic
  damped motion, force saturation, coupled direction reversals and an isolated
  grabbed cell. Existing strict loaded/dense tolerances were not relaxed.
- Baseline sway/cut/fall, four-second gentle reversal/hold, load-driven rough
  failure, timed Push, sticky-fault replay and nonrecoverable rollback passed.
  Repeated-cut recovery stress completed 30 simulated seconds in 28.13 wall
  seconds. The original Cable Lab native click/push/reset regression passed.

Per-case matrix CSVs are `/tmp/molla-grab-matrix-{gentle,strong}.csv`.
The reproducible native sweep is `make test-apple-tree-grab-ui-matrix`.
Native captures are `/tmp/molla-apple-grab-ui-{gentle,load}-N.png` and
`/tmp/molla-apple-grab-ui-fracture.png`. Timing and errors are printed by Make.

## Earlier separate Break and Grab verification (RTX 4080, Vulkan, release)

- Ten tree unit tests, 27 CPU strand/bundle/contact/splinter tests and shader
  validation passed. Seventeen GPU strand tests, three GPU contact tests and
  the new splinter/grab GPU test passed, including the strict loaded bundle
  and dense CPU oracle. Existing tolerances were not relaxed.
- Splinter tests check noncoplanar faces at the actual hit, conserved mass,
  linear/angular momentum and kinetic energy, repeated refinement, stale edit
  rejection and GPU capacity rollback. The point-grab CPU/GPU test also checks
  that invalid input leaves motion unchanged.
- The new rough click completed 20 simulated seconds with two fragments,
  conserved mass and floor checks in **26.78 wall seconds**. Gentle grabbing
  survived; stronger loading failed a joint without invoking either click
  tool, produced physical splinter geometry, and fell after release.
- The legacy planar-cut recovery fixtures remain planar to preserve their
  exact CPU/GPU comparison. They passed sticky-fault replay, nonrecoverable
  rollback and 30 simulated seconds of repeated-cut stress in **21.28 wall
  seconds**. These are distinct from the new rough-geometry tests.
- Intact-tree two-step/readback median was **6.401 ms**, worst **6.884 ms**
  over 20 samples, advancing 8.333 ms per sample. Fracture adds local cells
  and can require finer steps; these checks do not establish real-time speed
  for arbitrary edits or higher branch budgets. Other GPU workloads varied
  during this session, so timings are not an isolated before/after benchmark.
- Native Cut/rough-Break/reset and Grab/bend/fail/release/reset checks passed;
  actual framebuffer captures were inspected. The scripted severe drag
  advanced **5.450 simulated seconds in 13.951 wall seconds**, with eleven
  recovery batches and final playback **0.62x**. Severe-grab real-time
  performance is therefore still a limitation, not an accepted speed claim.
  The original Cable Lab native click/push/reset regression also passed.

The reusable splinter API currently accepts point attachments, as used by
this tree; it rejects finite-interface attachments rather than approximating
their transfer. This does not replace the original Cable Lab's dense-fiber
fracture implementation.

## Error-fix verification (RTX 4080, Vulkan, release)

- Ten tree unit tests passed; 17 GPU strand tests, including the dense CPU
  oracle and strict default-tolerance loaded bundle, passed. Three GPU
  collision tests passed, including re-enabled collisions after detachment.
- The tree's one-step CPU/GPU maximum center difference was **0.135 µm**.
- Two steps plus readback: **6.322 ms median**, **7.097 ms worst**, over
  20 samples; each sample advances 8.333 ms. These are not display FPS.
- With gravity and input disabled, the old junction filter induced a peak
  **0.201689 m/s**; the corrected filter produced **0.000038485 m/s**.
- Recovery tests inject a sticky PCG fault after a partially completed batch
  and verify replay against fresh GPU and CPU results, without losing edits
  or simulation time. Nonrecoverable faults restore a usable checkpoint.
- Repeated-cut stress completed **60 simulated seconds in 42.50 wall seconds**,
  with three automatic refinements, four pieces, fixed root and floor checks.
- Native Cut/Break/reset tests passed in both the tree and original Cable Lab.
  A close-up live capture advanced **5.075 simulated seconds in 5.216 wall
  seconds**, including startup, with final playback **1.00x**. Screenshots
  were inspected; this is not a guarantee against all temporal aliasing.

## Initial checkpoint before these corrections (RTX 4080, Vulkan, release)

- Seven unit tests passed, including off-node cut mass/cap geometry checks.
- The GPU headless test passed sway, pinned root, no spontaneous fracture,
  mass preservation, subtree detachment and two seconds of falling motion.
- One-step CPU/GPU maximum center error: **0.844 micrometres**.
- Two physics steps plus readback: **7.15 ms median**, 7.56 ms worst over
  20 samples, representing 8.33 ms of simulated time. This is not display FPS.
- Native UI verification passed actual Cut and Break clicks, three pieces,
  four cap faces, two seconds of motion, floor clearance and keyboard reset.
- A 300-frame live viewport capture advanced 5.125 simulated seconds in
  5.242 wall seconds including startup; the final playback indicator was
  1.00x. This is a short default-scene check, not a worst-case frame-rate claim.
- Cable Lab's native click/push/reset regression also passed after shared
  renderer changes. Dense-fiber roadmap acceptance is still separate.

See [data attribution](data/ATTRIBUTION.md) for the source path, hashes,
CC BY 4.0 dataset citation, modifications, and Appleskel credit; see
[`ACKNOWLEDGEMENT.md`](../../ACKNOWLEDGEMENT.md) for solver references.
