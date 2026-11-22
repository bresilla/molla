# Drone orchard

For the agricultural drone that stops and sprays each tree, use
`make run-ag-orchard`; see [AG_SPRAY.md](AG_SPRAY.md) for capture, verification,
droplet physics and modeling limits. The original continuous-flight mode remains
available with `make run-drone-orchard`.

A separate example: sixteen AppleQSM apple trees in two rows of eight, a dynamically flown
quadrotor, a shared three-dimensional air field, elastic wood and 15,440
individually breakable leaf attachments. Previous tree examples are unchanged
in their authored material and leaf-strength settings.

The orchard defaults to **downwash level 60**, where 0 is the previous gentle
2 kg setup and 100 is the original strong 8 kg setup. Linear interpolation of
hover force gives a **5.6 kg** configuration, approximately 54.9 N total thrust,
or 2.15 times the previous level-10 hover load.
The level is not a percentage of the original force or a linear airspeed scale.
The same physical force reacts on the air, so flight and downwash remain coupled.
Leaf attachment thresholds are **67% of the original strength**, retuned from
57% to limit extra shedding under the new mixed gusts. This changes
release loads, not spring stiffness, damping, mass or falling-leaf dynamics.
Structural trunk/main-branch bending compliance is now **4x** its previous
value. Axial compliance, torsion, damping, fracture limits, leafy-twig material
and leaf dynamics are unchanged. The roots remain fixed; structural sway comes
from the simulated wind loads, not camera-space or animated offsets.
An eight-second GPU comparison measures 10.4 cm peak trunk movement and 21.8 cm
main-limb movement, versus 2.4 cm and 4.5 cm with the old level-10/stiffness setup.
Each is measured from its own settled pose, with no root drift or wood failures.
Rotor geometry and ambient wind remain unchanged. Each row spans 16.8 m between
its first and last trunks; the complete outward/return flight takes 78 seconds.
The capture opens with an overview, follows the drone, then returns to the overview.

The current [full natural-leaf movie](../../artifacts/drone-orchard/natural-leaves-full-1080p.mp4)
is **78 seconds, 1920x1080, 30 fps, 2,340 frames**. It shows both rows, mixed
gusts and curved leaves, with no vapor or text frame. Full decoding and the
separate motion checks pass. The captured run sheds 4,109 leaves versus the
previous full strong-flow baseline's 5,479 (**25.00% fewer**); 3,873 distinct
leaves re-enter flight after touching ground. Roots stay fixed, no wood fails,
and all 16 overflight gaps stay below 5.67 cm. Trunk/main-limb peak motion is
20.8/55.4 cm. Ground settling and wind-driven accumulation remain possible.

The original full-row gate **still fails** its foliage-retention requirement:
73.39% retained versus 75% required. It was not relaxed. Media completion and
motion verification do not imply that older gentle-shedding target passes.
Capture took 933.82 wall seconds, not real-time acceptance. SHA256:
`54d30653b40c41a1b282e9a1c65f0874667e73e0ae6eea6dad1687927dd713d4`.

The previous [free-leaf sway preview](../../artifacts/drone-orchard/free-leaves-sway-preview.mp4)
is 12 seconds at 1080p/30 fps, with no vapor and the same camera/tree settings.
It sheds **739 versus 915 leaves: 19.23% fewer** in the matched sequence.
Attachment strength is 0.57; the airflow remains level 60 and wood flexibility
4. The exterior sampling change described below reduces the fractions of leaves
in the former grid-edge accumulation bands by over 90%. Leaves still settle
through ground friction; they are not forced to remain airborne. Trunk/main-limb
motion remains 10.4/21.8 cm with fixed roots and no wood failures. Complete video
decoding and `make verify-orchard-leaf-transport` pass. This is a preview
comparison, not requalification of the full 78-second flight at the new strength.

The earlier [whole-plant sway preview](../../artifacts/drone-orchard/whole-plant-sway-preview.mp4)
shows level 60 and strength 0.5 **without vapor**, for 12 seconds at 1080p/30 fps.
Complete media decoding passes. The exact capture retains 14,525 leaves and
releases 915, with no wood failures or root drift. The 78-second movies below
are historical lower-power settings, not re-recordings of the stronger setup.
That setting's 78-second physics run reports peak trunk/main-limb motion of 16.8/40.0 cm,
fixed roots, no wood failures and overflight of all 16 trees. It sheds 5,479
leaves and retains 9,961 (64.5%), so **the existing full-row acceptance check
fails its 75% foliage-retention requirement**. That requirement has not been
lowered; the stronger setting is not claimed to pass the old gentle-shedding gate.

The earlier [over-row vapor video](../../artifacts/drone-orchard/over-row-vapor-1080p.mp4)
is 78 seconds at 1920x1080/30 fps. Full-flight and complete media decoding checks
pass. Measured horizontal overflight gaps are below 5.5 cm for all sixteen trees
(below 5.2 mm on the first row), rather than the former 65 cm inboard route.
There are 304 detached and 15,136 retained leaves, with no wood failures or root
motion. The first-row preview and the second-row decoded frames were inspected.
The complete export took 972.9 wall seconds; this is not real-time acceptance.

The earlier [long-row video](../../artifacts/drone-orchard/long-rows-downwash10-1080p.mp4)
is 76 seconds at 1920x1080/30 fps. Exact-capture full-flight verification and
full media decoding pass: 912 leaves shed, 14,528 retained, both rows visited,
no wood failures or root motion. The final overview and the 48–52 s follow
sequence were inspected. Rendering took 999.3 wall seconds on the shared
workstation; this is not real-time performance acceptance.

The earlier [level-35 video](../../artifacts/drone-orchard/downwash-35-1080p.mp4)
is 32 seconds at 1920x1080/30 fps. Its exact-capture report passes full-flight
verification: 237 leaves shed, 5,553 retained, stable return flight and no wood
failures. Media decoding and frame-count checks also pass.
The [gentle-downwash video](../../artifacts/drone-orchard/gentle-downwash-1080p.mp4)
shows this 2 kg setup: 32 seconds, 1920x1080, 30 fps, complete-flight and
full-decode verification passed. The earlier videos below show the heavier setup.
The earlier gentle 32-second check (`artifacts/drone-orchard/gentle-flight.json`)
passes: two leaves shed, both canopy sides visited and stable return flight;
the four-second causal check also passes without stripping the opening canopy.

## Run

```sh
make run-drone-orchard
make test-drone-orchard
make test-studio-upload
make verify-drone-orchard
make verify-drone-orchard-row ORCHARD_REPORT=artifacts/drone-orchard/full-row.json
make probe-drone-orchard ORCHARD_SECONDS=8
make capture-drone-orchard ORCHARD_SECONDS=78 ORCHARD_VIDEO=artifacts/drone-orchard/flight.mp4
make verify-orchard-leaf-transport ORCHARD_REPORT=artifacts/drone-orchard/free-leaves-sway-preview.json
make verify-orchard-leaf-motion ORCHARD_REPORT=artifacts/drone-orchard/natural-leaves-full-1080p.json ORCHARD_STATE=artifacts/drone-orchard/natural-leaves-full-state.json
```

On this workstation, prefix commands with
`CARGO_HOME=/pkg/cargo NVIDIA_VERSION=595.112.07 nix develop --impure -c`.
Native launch also needs the current desktop's display environment.
Capture produces 1920x1080/30 fps MP4, a final PNG, JSON measurements and an
encoder log. Existing destination videos are not overwritten.
Add `ORCHARD_ARGS=--verify-flight` to a 78-second capture to run the full-row
acceptance gates against that exact recording and store the result in its JSON.

Controls: **Pause**, **Reset**, **Fly row** (off commands a hover at the current
position), **Air coupling**, **Air vapor**, and **Leaf strength**. Right-drag
orbits; the mouse wheel zooms. The native application slows simulation time
when physics cannot keep up; offline video advances fixed simulation steps.
The example is not yet a verified real-time 60 fps implementation.

`ORCHARD_ARGS` forwards arguments. Useful controls are `--no-coupling`,
`--motors-off`, `--physics-hz 240`, `--cold-start`, `--hide-airflow`, and
`--leaf-strength 1.0`. `--still-air` disables the ambient wind/gust background;
`--camera-drone` selects a close-up camera for captures.
`--downwash 10 --wood-flex 1` reproduces the previous air load and bending
settings for a same-build comparison. Downwash accepts 0–100 and wood flexibility
accepts 0.25–16; settings are preserved by native Reset and causal controls.
**Air visualization is off by default**, including captures. `--show-airflow`
explicitly enables the optional display; `--hide-airflow` takes precedence.
When enabled, twelve passive advected paths carry broad, softly tapered ribbons.
A dedicated WGSL shader uses animated multiscale value noise, Gaussian edge
falloff and at most 3% alpha per layer. The pass is depth-tested against the
trees, aircraft and ground, with premultiplied blending and no depth writes.
There are no bright center lines or particle sprites. This is a vapor-like
display of sampled air motion, not simulated condensation or volumetric smoke.
The hash-method credit and license are recorded in `ACKNOWLEDGEMENT.md` [34].
Before flight, vegetation settles under its normal gravity, damping and leaf
loads; the more flexible configuration needs about 14.9 simulation seconds
to settle on the tested GPU. `--cold-start` skips this and
exposes the initial gravity sag. Settling is not counted as flight time.
All mechanical-rate choices use the same 60 Hz initialization before switching
to the requested flight timestep, so their initial state is identical.
The closer canopy-flight route defaults to strength scale 0.67;
the previous leafy-tree example is unchanged.
This threshold is a demonstration material parameter, not measured apple
petiole strength. `verify-drone-orchard` compares four seconds with and without
air/tree coupling at the same strength, and fails unless branches respond,
the quiet control retains its leaves, roots stay fixed, the drone moves,
states stay finite and the pressure/ground diagnostics meet its bounds.
The gentle opening hover need not detach leaves; shedding remains a full-flight check.
The earlier level-10 four-second coupled/uncoupled comparison passes: 9 leaves detach with
air coupling versus none without. A previous motors-off control failed when
the unpowered 2.6 kg drone hit the ground: its friction KKT residual slightly
exceeded the rigid solver tolerance. That optional ground-contact mode has
not been requalified at the new 5.6 kg setting.
`verify-drone-orchard-row` checks a complete 78-second outward/return flight,
including both row centers, arrival at the far end and return close to the
starting hover. It also requires positive rotor/leaf clearance below 0.65 m,
measured airflow during shedding, measured visits to both row sides, and at least 75% of
the leaves remaining. Every tree must have a measured horizontal drone-center
overflight gap below 0.3 m; reports include all 16 gaps and a one-second flight trace.
`--state PATH` additionally saves wood and leaf states in probe/row-check mode.

## What is simulated

- Existing copter controller, motor acceleration, propeller forces/torques,
  rigid-body integration and hull drag. Waypoints command velocities; the
  drone's pose is not animated along a prescribed trajectory.
- New reusable `molla-solvers::airflow`: a staggered 3D velocity grid,
  semi-Lagrangian transport, viscosity and pressure projection. Rotor disks
  receive the exact opposite averaged force/moment applied to the aircraft.
- Spatial air-relative drag on wood and individual leaf particles. Leaf
  spring forces react on wood at their petioles. Aerodynamic reaction impulses
  return to the air; loads outside the supported deposition region are
  reported separately as boundary impulse.
- Leaf shedding follows spring load exceeding attachment strength. Optional
  airflow paths are passive samples; they do not push trees or remove leaves.
- Detached leaves use the Andersen–Pesavento–Wang falling-card equations:
  translational/rotational circulation, angle-dependent drag, added air mass
  and inertia, and dissipative torque. A world-space tensor/vector extension
  permits pitch, roll and yaw. The renderer uses the simulated blade axes;
  there are no random kicks or prescribed sine rotations. Still-air tests
  require sustained flutter reversals, full tumbles and upward turning arcs.
- Elliptical blade inertia and RK4 resolve free flight at 480 Hz or faster,
  with airspeed/spin-dependent substeps for rotor downwash. The planar limit
  matches independently integrated dimensionless paper equations. The 3D
  extension is approximate and is not a calibrated leaf/CFD reproduction.
  Coefficients: CT=1.2, CR=pi, A=1.4, B=1.0, mu1=mu2=0.2; density 1.225 kg/m³.
  Orchard detached-blade area varies from 8 to 13 m²/kg times sample mass;
  canopy weights and attached spring/drag coefficients are unchanged.
  Curvature varies from 3% to 8% of blade length. Six rim contacts plus the
  recessed center match the detached render mesh. Offset drag samples velocity
  at its pressure center, supplying a dissipative physical turning moment.
  Zero-camber callers retain the original planar APW/contact model.
  Energy diagnostics include the blade's unresolved added air, which exchanges
  energy with the blade. Air reactions report solid momentum changes before
  contact; the ground's reaction is not deposited into the airflow.
- Ground contact uses blade extent, normal impulses and Coulomb friction
  (coefficient 0.3), not exponential horizontal braking. Leaves may stop when
  ground friction balances wind, but are never stopped at a spatial boundary.
- Elastic tree islands run in parallel GPU workgroups. Convex wood contact
  still includes all trees; an occupancy bitmap accelerates deterministic
  contact coloring without dropping candidate pairs.

The default grid extends along both long rows at 0.15 m spacing, with a free-slip impermeable floor and
open ambient side/top boundaries. Air and flight/attachments/wood default to
60 Hz; guarded wood stepping can subdivide further. Mechanics can be selected
at 60, 120, 240 or 480 Hz without changing the air grid or its 60 Hz timestep.
Other tree examples retain their original 240 Hz mechanics.

The default atmosphere has mean wind (1.2, 0, 0.35) m/s plus deterministic,
spatially coherent divergence-free gusts at scale 0.65 m/s. The field seeds
the grid and supplies its open-boundary inflow; rotor forcing, advection and
pressure projection determine the interior flow. The background remains
defined at arbitrary distances outside the near-field grid. Leaf positions
are never clamped, wrapped, recycled or expired at the grid/camera edge.
Exterior rotor wake is not pressure-resolved. Particle queries continue the
boundary velocity's deviation from background with a Gaussian distance decay
of length 3 m, adding it to the local ambient gust field. This is an approximate
far-field extension, not exterior CFD; it replaces the earlier 0.15 m transition
that collected leaves along the grid edge. In-grid advection and projection
are unchanged. Momentum, orientation, aerodynamic forces and floor friction
keep evolving outside the grid. `AirConfig::exterior_decay_length` selects this
query policy; its default zero preserves the original one-cell blend for other
callers. The orchard explicitly selects 3 m.
The gust modes are a synthetic background, not calibrated atmospheric turbulence.
Six additional time-evolving curl-potential modes provide mixed directions at
several scales, with amplitude 2.2 and a 1.4 m near-ground envelope. Their floor
normal velocity is zero. They are evaluated consistently in CPU/WGSL air queries
inside and outside the grid, including the air used by wood and aircraft.
This is an explicitly modeled subgrid velocity, not pressure-resolved LES.
Curved leaves can rock, roll and re-enter flight under changing aerodynamic
loads; no random position/velocity kicks or perpetual-flight rules are used.
The construction and approximation are credited in `ACKNOWLEDGEMENT.md` [35].
Force and torque reactions outside the deposition footprint are reported as
boundary impulses rather than silently projected back onto the grid edge.

## Canopy and drone appearance

Each tree has 965 independent leaf attachments rather than 420. Denser
phyllotaxis and larger blades produce approximately 4.08 times the previous
rendered leaf-area proxy, without adding wood degrees of freedom. Per-twig
represented mass and effective drag budgets are distributed over the extra
leaf samples; spring stiffness, damping and strength scale with sample mass.
This preserves the existing effective canopy loading instead of artificially
quadrupling it. Every sample still has its own force, threshold and motion.
The visual leaf area is not a calibrated aerodynamic area measurement.
At the default camera distance, each leaf uses a 14-triangle display mesh;
zooming within 4 m restores finer blade detail. This changes rendering only,
not attachment count, physical forces or failure thresholds.

The flight starts above the first tree at x=-1.45 m and travels directly over
that row, crosses to x=+1.45 m at the far end, returns directly over the second
row, then crosses back to its initial hover above the first tree. Longitudinal
legs each allow 30 seconds; the wider crossings allow seven and nine seconds.
Target rotor height starts 0.55 m above the authored highest leaf center.
Aircraft motion follows the existing controller and motor dynamics; no leaf
release or branch impulse is triggered by proximity or waypoint time.

The aircraft now uses original procedural agricultural-drone geometry:
a shaped pearl/orange fairing, dark structural arms, detailed motor housings
and cooling rings, twisted propeller blades, landing skids, GPS, status lights
and gimbal camera. Cached meshes need no runtime assets or downloads. Motor
spacing and blade radii match the simulated aircraft. It is not manufacturer
CAD or the source of flight coefficients. The rejected Innerscene asset,
legacy loader and [asset provenance](data/drone/ATTRIBUTION.md) remain available
but are no longer the displayed model.

## Evidence and current limits

The circulation revision supersedes the earlier resistive-panel leaf model.
Its [32-second orchard video](../../artifacts/drone-orchard/leaf-circulation-orchard.mp4)
passes full-flight verification: 1,278 leaves shed, 4,512 retained, 22.94 m
maximum horizontal travel, no wood failures/root motion. The
[still-air drop](../../artifacts/drone-orchard/leaf-flutter-still-air.mp4) and
[breeze drop](../../artifacts/drone-orchard/leaf-flutter-breeze.mp4) show the
same GPU solver close up. All three are complete, decoded and inspected at
1920x1080/30 fps. The native preview is not a real-time performance acceptance.
Run `make capture-leaf-flight ORCHARD_SECONDS=8 ORCHARD_ARGS=--still-air
ORCHARD_VIDEO=artifacts/drone-orchard/leaf-flutter-still-air.mp4` for a close
GPU-rendered nine-leaf drop; omit `--still-air` to add the orchard's analytic
background breeze. Use a fresh output path. Research attribution is in
[`ACKNOWLEDGEMENT.md` [33]](../../ACKNOWLEDGEMENT.md).

The following numerical flight/video results describe the earlier panel model;
new circulation-model acceptance is recorded in `plans/DRONE_ORCHARD.md`.

On RTX 4080/Vulkan, all 5,790 attachments survive still-air initialization.
The physical-leaf/wind revision's four-second causal comparison sheds 15
leaves with coupling versus zero without, with 1.75 cm maximum final induced
wood-position difference. The quiet control relaxes less than 0.5 mm.
The final 32-second flight sheds 1,277 leaves and retains 4,513; 784 detached leaves
end outside the air grid and maximum measured horizontal travel is 7.84 m.
Minimum sampled rotor/leaf gap is 0.495 m, peak local shedding wind 8.32 m/s,
roots remain fixed, no wood fails, and ground air flux is zero. Divergence RMS
ends at 2.38e-4 /s. These are numerical checks, not measured orchard validation.

Eleven CPU attachment tests cover momentum/torque balance, dissipation,
sideways drift, angular response, long transport, friction and quiet landing.
Five GPU tests cover CPU parity, oriented ground contact, moving-owner release,
restore and the unchanged legacy point-particle path. Dedicated
wind tests check divergence-free gusts, zero floor-normal flow, evolving
CPU/GPU boundary samples and conservative torque deposition. A single-leaf
uniform-wind test travels from x=4.7 to x=68.73 m without any spatial limiter.

The final recording took 199.7 wall seconds for 32 simulation seconds on the
shared desktop. That includes rendering and encoding, not just physics.
This is not a real-time 60 fps application. Historical physics-only timings
remain in [`plans/DRONE_ORCHARD.md`](../../plans/DRONE_ORCHARD.md).
The renderer uses bounded mapped uploads and lower-detail distant leaves,
but the live 3348x1348 desktop still advances slower than real time. The
smooth offline video does not demonstrate interactive frame-rate acceptance.

The final inspected [1080p video](../../artifacts/drone-orchard/leaf-dynamics-showcase.mp4)
has 960 frames at 30 fps. Its `leaf-dynamics-showcase.json` reports
`verified_full_row=true`; `leaf-dynamics-causal.json` reports `verified=true`.
The ambient-wind-only control (`physical-leaf-wind-only.json`) sheds zero leaves
with motors off during the same four-second interval.
The earlier `leafy-canopy-flight.mp4` is historical: it uses point-mass leaves,
no ambient wind, and the rejected drone asset, not the current physics/model.
Reports also include sample trajectories and blade normals for up to
eight detached leaves; those diagnostics do not control their motion.

`make test-airflow-refinement` checks a transported smooth vortex against
finer spatial/temporal references and repeats the same run. Spatial RMS error
decreased 0.00947 → 0.00542 → 0.00282 m/s over 16³, 24³ and 32³ grids against
64³. Temporal error decreased 1.04e-4 → 4.79e-5 → 2.04e-5 m/s at 30, 60 and
120 Hz against 480 Hz. In a separate 20 N rotor study, sampled momentum flux
was 19.55–20.05 N across four grids, but mean downwash changed from 2.04 to
1.48 m/s. Thus the local wake profile, and consequently leaf shedding, remain
grid-sensitive; the default orchard grid is not a grid-independent CFD result.

This is an underresolved incompressible actuator-disk model, not blade-resolved
CFD or a calibrated turbulence/abscission model. Rotor thrust uses static
motor-speed coefficients, not pressure integration over resolved blades.
Wood couples through drag,
not a resolved impermeable moving boundary. Leaf/leaf and leaf/wood collisions
are not modeled. The drone hull has floor contact but is not coupled to tree
collision geometry; the default route stays above the canopy edges. Refinement and
long-run tests establish numerical behavior for the tested cases, not physical
calibration, general flight-envelope validation, or a guaranteed frame rate.
The resolved rotor-flow domain is finite, and its transition to a synthetic
ambient atmosphere is approximate. There is no claimed grid-independent wake
or quantitatively calibrated flutter, abscission or litter-transport model.

## Sources

Full citations and adaptation details are in [`ACKNOWLEDGEMENT.md`](../../ACKNOWLEDGEMENT.md),
especially [31]–[33]. The tree geometry/elastic-fracture lineage is documented in
[`APPLE_TREE.md`](APPLE_TREE.md). PPF is a separate backend, not the tree solver
used by this example.

- Bridson and Müller-Fischer (2007), [*Fluid Simulation*](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf): numerical fluid-method reference.
- Bangura et al. (2016), [*Aerodynamics of Rotor Blades for Quadrotors*](https://doi.org/10.48550/arXiv.1601.00733): rotor-model reference, not a full blade-element port.
- Bauersfeld et al. (2024), [*Robotics meets Fluid Dynamics*](https://arxiv.org/abs/2403.13321): measured quadrotor-wake research context.
- Fu et al. (2025), [canopy/downwash study](https://doi.org/10.3390/agronomy15081867): orchard-flow research context, not reproduced results.
- Andersen, Pesavento and Wang (2005), [*Unsteady aerodynamics of fluttering and tumbling plates*](https://doi.org/10.1017/S002211200500594X): falling-plate aerodynamics reference.
- Andersen, Pesavento and Wang (2005), [*Analysis of transitions between fluttering, tumbling and steady descent of falling cards*](https://doi.org/10.1017/S0022112005005847): implemented planar equations, with original reduced 3D and curved-leaf extensions; not an experimental reproduction.
- Bridson, Hourihan and Nordenstam (2007), [*Curl-Noise for Procedural Fluid Flow*](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf): curl potentials and floor-boundary treatment; Molla uses analytic evolving modes rather than the paper's Perlin-noise implementation.
- Innerscene Corp. (2026), [*Quadcopter with flight electronics*](https://www.innerscene.com/tools/library/3d-parts/quadcopter-with-flight-electronics-a25968c1): superseded CC0 visualization mesh, retained with attribution.
