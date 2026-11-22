# Cable Lab

Native, procedural demonstrations of Molla's **GPU VBD cables** and
**GPU coupled oriented-fiber branches**. No assets or external simulation
runtime are required. Sources are listed in `../../ACKNOWLEDGEMENT.md`.

The measured apple skeleton is a **separate executable and source file**:
`make run-apple-tree`, `src/apple_tree.rs`. See [APPLE_TREE.md](APPLE_TREE.md)
for its reduced physics model and dataset attribution.

```sh
make run-cables
make test-cables
make test-cable-branches
make test-cable-click-ui
```

On this Nix/NVIDIA machine:

```sh
NVIDIA_VERSION=595.112.07 nix develop --impure -c make run-cables
```

The desktop session must provide `WAYLAND_DISPLAY` or `DISPLAY`. For the
current session those are `wayland-1` and `:0`, respectively.

## Scenes

1. **Suspended cable:** two fixed ends, gravity and a lateral initial velocity.
2. **Bending comparison:** three cantilevers with identical masses and axial
   stiffness but different discrete bending coefficients. Two pinned nodes
   at each root fix the initial tangent.
3. **Drape over a sphere:** three free cables fall onto a fixed sphere and floor.
4. **Elastic branch — Cut:** rooted **7-fiber** bundle (28 cells), with the
   clean-cut tool selected initially.
5. **Elastic branch — Break:** exactly the same geometry, mass, material,
   elasticity and damping, with the uneven-break tool selected initially.
   Either scene supports both tools, pushing, and the same rigidity control.
   Neither enables automatic overload failure. The lower-strength stress
   fracture fixture remains in the headless verification suite only.

Use the scene buttons or **1–5**, **Space** to pause, **R** to reset, and
**Pluck cables** to add a lateral velocity pulse. Drag the scene to orbit;
scroll to zoom. **Step** advances one 1/240-second cable step or one
1/120-second branch step. The viewer launches in release mode by default.

In branch scenes, select **Cut [C]** or **Break [B]**, then **click on the
branch**. Picking intersects the actual convex volumes, not their nearest
nodes. A click cuts perpendicular to the picked fiber's local axis at the
surface hit. Break prescribes individually staggered fiber fracture planes around that
hit, with the picked fiber's plane passing exactly through it; it is an
interactive topology operation, not a claim of stress-predicted fracture.
Only the picked connected piece and its local section are targeted.

**Push [P]** applies an impulse at the picked surface point. **Bend / pluck
branch** applies a distributed velocity impulse. **Rigidity ×** changes
elastic compliance while preserving motion and topology. The branch default
uses E = 15 MPa and G = 6 MPa,
with lower damping. These are uncalibrated demonstration parameters.
The lighter preview replaces the 37-fiber configuration while preserving
cross-sectional area and mass. Four cells per fiber reduce longitudinal
resolution; this is a speed/detail tradeoff, not identical discretization.

With either tool selected, dragging defines an angled finite blade. The stroke
must cover each affected fiber cross section; a shorter stroke can leave
other fibers attached. **Escape** cancels a gesture, **right-drag** or
**Orbit [O]** rotates the camera, and scrolling zooms. Cut faces and uneven
fracture caps are triangulations of the actual clipped geometry, not spheres
or decorative caps. Tiny unresolved cuts and root-attachment intersections
are rejected atomically rather than snapped to a node.

`CABLE_ARGS='--scene 2'` starts the third scene. A reproducible rendered
capture can be requested with
`CABLE_ARGS='--scene 2 --screenshot /tmp/molla-cables.png'`; the viewer saves
its framebuffer after 120 frames and closes. `make test-cables` runs all
three scenes for 720 GPU steps each, including a pluck, and checks finite
motion, fixed anchors, stretch below 1.05, particle/obstacle clearance above
-2 mm, and rendered tube/obstacle clearance above -5 mm. It also checks
that the stiff cantilever tip remains at least 30 cm above the flexible one.

`make test-cable-branches` checks finite blade/picking geometry, the softer
default against the previous stiffness, plucking, repeated oblique cuts,
mass preservation, falling pieces, floor clearance and progressive stress
fracture. `make test-cable-click-ui` opens the native GPU-rendered window,
injects pointer/key events through eframe's normal input hook, clicks Cut
then Break, checks Escape cancellation, pushes a surface, lets pieces fall,
captures the framebuffer and tests keyboard reset. Use
`CABLE_ARGS='--screenshot /tmp/molla-branch-click-ui.png'` to choose its image
path. `make test-cable-ui` exercises the retained drag-cut interaction.
These are scripted UI interactions, not a claim of a manual usability review.

The suspended/drop scenes use 32 nodes per cable and 24 VBD iterations.
The cantilevers use 16 nodes and 64 iterations. Resolution and convergence
affect the discrete bending response; the comparison is within one shared
discretization, not a resolution-independent material calibration.

## Scope

- Physics uses Molla's public `ModelBuilder`, `VbdSolver`, and `State` APIs,
  with WGSL kernels on wgpu 30. State stays on the simulation device between
  steps; positions are read once per display frame for visualization.
- The viewer is separate from reusable physics crates. Eframe currently
  renders through its own wgpu 29 device. Tube/sphere triangles are built
  from the simulated coordinates and rendered with GPU depth and shadows; this small
  inspector is not a general-purpose renderer.
- VBD's implemented cable bending term is a discrete second-difference
  penalty. The coefficients shown are numerical model coefficients, not
  calibrated material bending moduli. There is no torsional frame model.
- Branches use `molla-solvers::strands`, with material-relative orientations,
  bend/twist compliance, finite interface polygons, and a resident coupled
  solve. Geometry/topology changes are atomic step-boundary operations;
  ordinary dynamics do not read per-fiber state back to the CPU. The viewer
  requests one state snapshot per display frame.
- Branch floor contacts use the actual clipped vertices with independent
  unilateral multipliers and Coulomb-limited friction. Discrete convex
  piece-piece contacts are enabled. The separate apple-tree example builds
  a reduced branch hierarchy. Broad convergence, dense moving oblique-cut
  validation, CCD and full fracture-roadmap acceptance remain open.
- Live branches use a capped wall-clock accumulator and display their
  approximate actual playback rate. The headless repeated oblique-cut gate
  still has a component-count failure; native click/break/reset passes do
  not establish that broader acceptance.
- Obstacle contact is particle-based, discrete and compliant. Cable/cable
  continuous collision, knot preservation and PPF rod parity are **not**
  claimed. These examples do not exercise the unfinished PPF strain work.
- Simulation time advances by a fixed number of steps per displayed frame;
  it can run slower than wall time. The displayed latency includes the
  blocking position readback and is not a benchmark acceptance result.
