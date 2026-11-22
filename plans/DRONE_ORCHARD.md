# Drone, airflow, and orchard coupling

## Crosswind drift and varied tree yaw (2026-11-20; full video verified)

### Goal
A video like the AG mission with a light breeze for the first row and a strong
crosswind from one side for the second. On-screen text announces the wind before
it rises. A close-up then shows a second-row leaf that caught far less spray
because of drift. Every tree turns to its own yaw so the copies don't look
identical.

### Discoveries
Changing `AirConfig.ambient` alone never reaches the interior. Semi-Lagrangian
backtraces only pick up the background where they leave the grid, and from
still air nothing leaves, so the interior stayed at exactly 0 after 180 steps.
A large-scale wind change is a uniform body acceleration of the air mass (the
synoptic pressure gradient). A uniform horizontal velocity increment is
divergence-free, keeps the ground impermeable, and preserves the rotor wake and
gusts on top of it. Leaf strength is calibrated to the drone's shedding, so wind
speed decides how many leaves fall. 140 s probes, wind starting at the second
row:

| crosswind | leaves shed | second-row spray vs first row | branch retries |
|---|---|---|---|
| 0 (light breeze) | ~1,050 | — | 0 |
| 3.5 m/s | 2,204 | −88 % | 12 |
| 4.5 m/s | 4,068 | −93 % | 29 |
| 6 m/s | 6,882 | −97 % | 34 |

All retries were single rotation faults (0x10) at 1/60 s that passed at 1/120 s.
The drone completed all 16 stops in every case (max hover error 0.043 m).
Comparing the best-covered leaf of each tree understates drift (97 against 113
parcels at 3.5 m/s), while the trees themselves differ tenfold.

### Accomplished
`Settings.drive` (WGSL) and `AirState.drive` (CPU) hold a uniform velocity
increment that the advect step adds to every face. `set_background(ambient,
gusts)` on `GpuAir` and `AirState` queues the change in ambient wind as that
increment. The settings uniform grew to 96 bytes; cache keys `mac-air-v7-drive`
and `breakable-attachments-v8-drive`. Tests: on the CPU a wind change moves the
whole air mass and stays steady; on the GPU it matches the CPU through a six-step
ramp around a working rotor.

`crosswind.rs` schedules the wind. When the mission reaches the second row the
crosswind is announced, rises 2.5 s later over 3.5 s to `--crosswind` m/s along
+x (gusts 0.2·speed), and the gauge (speed, state, camera-projected arrow) stays
up the whole capture. `orchard_model::tree_yaw` turns each tree about its own
trunk axis by a deterministic yaw in [0, 2π); trunks stay on the grid, and the
closest cells of neighbouring trees stay over 0.4 m apart. The second close-up
inspects tree 15 (station 9) after the wind is fully up. It shows the twin of the
first close-up's leaf (same twig of the same model; falls back to the best
leaf if the twin was shed), with the light-breeze leaf as a reference line.
Report: `crosswind`, `drift_leaf`, `tree_yaw_rad`, per-tree parcels and wet
leaves. `make capture-ag-crosswind` (`AG_CROSSWIND` 3.5) and
`make verify-ag-crosswind`.

Capture `artifacts/ag-spray/crosswind/` (3.5 m/s, 1,306 s wall): `verify-ag-spray`
and `verify-ag-crosswind` pass. The wind is announced at 59.6 s and fully up at
65.6 s. Leaves shed: 2,204; no wood failures. Row 1 received 138.4 mL and row 2
16.4 mL. Tree 1's leaf 962 got 113 parcels (≈81,632 droplets). Its twin on tree
15 got 52 parcels (≈30,216 droplets), and the whole tree 660 parcels on 71 wet
leaves against 6,293 on 471. The best-leaf render is kept in
`crosswind/best-leaf/`.

### Next Steps
The background wind is uniform with height; an atmospheric profile would lower it
within the canopy and reduce wind shedding.

### Relevant Files
`bin/molla_cables/src/crosswind.rs`, `drone_orchard.rs`, `leaf_focus.rs`,
`orchard_model.rs`, `orchard_sim.rs`, `verify_ag_crosswind.jq`,
`crates/molla-kernels/shaders/air_field.wgsl`, `airflow.wgsl`,
`crates/molla-solvers/src/airflow/{cpu,gpu}.rs`, `attachments/gpu.rs`,
`crates/molla-solvers/tests/airflow.rs`.

## Branch solver: tree solve, warm start, contact broad phase (2026-11-20; full video verified)

### Goal
Same wood physics, faster: exact tree solve and warm start in the coupled solve.

### Instructions
Equations, fracture, contacts and adaptive replay unchanged. The PCG tolerance
was changed with the user's approval to be relative to the step's first Newton
residual.

### Discoveries
The coupled matrix `M + Σ k̃ g gᵀ` has the bond graph's sparsity; orchard bond
graphs are forests (pinned roots decouple), so block LDLᵀ from the leaves has no
fill-in. With it PCG fell from a median 133 to 1–2 iterations, but the step did
not get faster: contacts dominated. Detection tested all n(n−1)/2 = 1.84 M pairs
every step (full vertex AABB per pair) and `color_contacts` ran on one GPU thread
over the 57.6 k-word active bitmask. From a re-created GPU state (dropped bond
multipliers) the second Newton solve stalls at the f32 floor (relative residual
2–5·10⁻⁵ against 2·10⁻⁵) and faults; in the running orchard no faults occur
(1 substep). Timings swing 2× while another session's GPU app runs; only
interleaved A/B runs are comparable. The first tree-solve full capture took 1,887 s.
At t = 76.25 s a rotation fault (0x10) at 1/60 s started the replay. Retries at
1/120, 1/240 and 1/480 s then failed on PCG (0x4: 512 iterations, relative
residual 4–5·10⁻⁵ against each later Newton solve's own, already small, start).
The solver held 16 substeps for the rest of the run because every 2 s coarsening
failed the same way.

### Accomplished
Host plans a BFS spanning forest per island (pinned cells excluded), levels
deepest-first, appended to `colors` (`StrandParams.tree`). `solve_coupled`
factors D_c = A_cc − Σ A_ck W_k and W_c = D_c⁻¹A_cp per level and preconditions by
forward/backward sweeps; a failed pivot falls back to block Jacobi. Warm start:
per-Newton-iteration solutions (`status[4]` counter, two slots), kept only when
their true residual beats the zero start; tolerance stays relative to the zero
start; a warm start already within tolerance counts as converged. Contacts:
bounding-sphere early-out (radius in `inertia0.w`, 1 % + margin slack) and a
256-thread compaction of the active words before the serial coloring, same order
(bit-identical results). Interleaved A/B, 10 s cold-start probe: wood step
37–57 → 11.5–12.5 ms, full physics step 73–98 → 34–35 ms; shed 233 both;
displacement 0.9159 → 0.9167 (PCG iterate within tolerance). New GPU test: a
40-cell branch converges in ≤3 PCG iterations and matches the CPU oracle.

Step-relative tolerance: Newton solve 0 stores its zero-start residual per island
(`Krylov.reference`, `KRYLOV_BYTES` 768). Every solve then converges at
`max(initial, reference)·rtol²` and reports its residual against the same
reference. The failure test now closes a loop with an extra bond, so the tree
factor is inexact and a 1-iteration cap still faults. Strands GPU (21), contacts
(4) and splinter (3) pass. The 82 s trace probe had 0 retries (previously 6):
shed 803 → 800, peak wood 0.60635 m in both. Full capture
`artifacts/ag-spray/step-tolerance/`: verify passes; 1,472 s wall (1,887 s);
wood step mean 36.1 ms (88.8 ms tree solve, 52.1 ms original solver); shed 1,044;
canopy 293.3 mL; 6,871 wet leaves; focus leaf 962 with ≈82,165 droplets;
peak wood 0.619 m; 0 wood failures. The other session's GPU app was running, so
the air, leaf and sampling stages were 20–70 % slower than in the `realtime`
capture.

### Next Steps
The rotation fault did not recur in the new trajectory. A replay at reduced dt has
not been exercised in the orchard since the tolerance change. Remaining step cost:
leaf kernel + snapshot (~9 ms), air solve (~7 ms), wood readbacks (~3–6 ms),
CPU spray (~4 ms). Detection still launches one thread per cell pair; a spatial
broad phase would remove that.

### Relevant Files
`crates/molla-kernels/shaders/strands_coupled.wgsl`, `strands_contact.wgsl`,
`strands.wgsl`, `crates/molla-kernels/src/strands.rs`,
`crates/molla-solvers/src/strands/gpu.rs`, `crates/molla-solvers/tests/strands_gpu.rs`.

## Faster orchard step and capture (2026-11-20; full video verified)

### Goal
Move the orchard toward real time: keep the leaf↔air coupling on the GPU (1),
draw leaves on the GPU (3) and overlap video readback (4). The branch solver (2)
is out of scope until discussed.

### Instructions
Physics results must not change beyond float rounding; branch solver untouched.

### Discoveries
`perf` cannot open events on this machine; temporary step timers were used.
Cold-start 10 s probe, per 1/60 s step: branch solver 37.5 ms + 3.3 ms impulse
readback; non-branch ~28 ms (leaf host packing/validation 4.5, CPU leaf→air splat
3.7, air sampling with a GPU sync per 4,096-point chunk 2.6, spray 4.6, air step
6.3 incl. repacking the whole force grid). The CPU built ~1 M leaf triangles
(31–147 MB) per video frame. Derived (`layout: None`) bind group layouts are
exclusive to their pipeline in wgpu, so the leaf pipelines own their bind groups.

### Accomplished
`GpuAir` gains a fixed-point (2^-24 N) reaction accumulator folded in during
advection and cleared each step, a boundary-impulse accumulator
(`take_boundary_impulses`) and `coupling()`. Sampling functions moved to
`air_field.wgsl`, shared with the leaf kernel. `GpuAttachments::couple_air` and
`step_coupled_blades`: leaves sample the MAC field on the device, record the
sampled speed (`Particle::wind_speed`) and scatter their reaction force/torque
with the `AirForces` stencils; leaves leave the host wind queries and CPU
reaction loops; leaf host validation is O(1) per step; the air step packs only
non-zero force spans. Step: non-branch ~28 → ~18 ms; identical shed count at
10 s (233). Leaves render as instances of one 8-segment canonical mesh
(`studio_leaves.rs`, `leaf_vertex`/`leaf_shadow_vertex`) from 80-byte per-leaf
records built by `leafy_draw::instances`; frames compared against the CPU mesh
match. Capture renders into two readback buffers and delivers the previous frame
to an encoder writer thread. New GPU tests: device sampling equals host queries;
device reactions equal host deposits (incl. boundary impulses). 17 s preview:
9.7 s wall per simulated second (was 13.6). Full capture
`artifacts/ag-spray/realtime/ag-orchard-full-1080p.mp4`: 1,475 s wall (was 1,910 s),
`make verify-ag-spray` passes; shed 1,053 (1,071 before, 1,066 baseline), canopy
293.3 mL, 6,872 wet leaves, same close-up leaf (962, ≈82,667 droplets).

### Next Steps
Remaining non-branch cost: CPU spray parcels (~3–5 ms), leaf snapshot for spray
and diagnostics (~2 ms), air solve; then the branch solver discussion.

### Relevant Files
`crates/molla-solvers/src/airflow/gpu.rs`, `crates/molla-kernels/shaders/air_field.wgsl`,
`airflow.wgsl`, `attachments.wgsl`, `crates/molla-solvers/src/attachments/gpu.rs`,
`crates/molla-solvers/tests/attachments/coupled.rs`, `bin/molla_cables/src/studio_leaves.rs`,
`studio.rs`, `studio.wgsl`, `leafy_draw.rs`, `branch_draw.rs`, `video_capture.rs`,
`orchard_sim.rs`.

## Attached leaves on elastic petioles (2026-11-19; full video verified)

### Goal
Leaves under the drone should move far more, increasingly close to the drone,
from physically based airflow response, while the shed (drop) rate stays the same.

### Instructions
Physics on the GPU with a CPU oracle; keep the 140 s AG mission, square boom,
spray and leaf close-up; render the full video when calibrated.

### Discoveries
Attached leaves were a point mass on a 7 Hz spring with orientation-blind drag
(≈ edge-on plate drag) and a fixed world orientation, so they hardly moved. A
rigid pin joint passed wood-solver velocity jumps straight into the abscission
load (39–627 leaves shed in still air); the 6-DOF bushing with the old
translational spring fixes that. Full isolated-plate drag on every attached leaf
bent limbs 0.9 m and overflowed GPU contact pairs: canopy sheltering is needed
(canopy models use ~0.2 per leaf area). The unloaded-to-loaded settling sag
swings leaves hard enough to shed them, so abscission is off while settling and
the budget is 24 s (settles in ~17–19 s).

### Accomplished
`attachments::Petiole` and `step_cpu_blade` (CPU oracle, `petiole.rs`) and the
matching `hinge` path in `attachments.wgsl`: attached leaf bodies with a petiole
are 6-DOF blades; the translational spring-damper joins the petiole joint to the
twig and is the abscission load; bending (2.5–4 Hz about the joint), torsion
(EI/GJ = 4), ζ = 0.12, gravity preload, four APW plate elements with incidence-
dependent centre of pressure, scaled by sheltering 0.3. Rest frames now follow
the twig; drawing, spray hits and beads use the moving blade frame. Leaves start
at their static joint sag. Bend 2° at 0.5 m/s, 24° at 2 m/s, ~70° folded at
6 m/s (unsheltered). Strength scale 0.85 → 1.25 from full 140 s probes
(1.2 → ~1,185 shed, 1.3 → 926; previous 1,066). 23 attachment CPU tests, 8 GPU
tests (new CPU/GPU parity), 53 orchard tests pass.

Full capture `artifacts/ag-spray/leaf-motion/ag-orchard-full-1080p.mp4`: 4,200
frames, 140 s, 1080p30; `make verify-ag-spray` passes all 15 checks. Shed 1,071
versus 1,066 before (10 s: 223/204, 50 s: 531/444, 100 s: 908/881); 14,369
attached; canopy 292.8 mL (was 247.6), ground 372.3 mL (was 417.5), 6,861 wet
leaves; peak limb 0.567 m, trunk 0.166 m. Close-up leaf 962: 115 parcels ≈
82,667 droplets, 332 µL. SHA256
`b2a134151ec79ae357e7ce18c4b5577b27149cc5b5f36427a99885f73bc083bc`.

### Next Steps
None required. Attached leaves exert their joint force on the twig but not the
petiole bending moment.

### Relevant Files
`crates/molla-solvers/src/attachments.rs`, `attachments/petiole.rs`,
`attachments/gpu.rs`, `crates/molla-kernels/shaders/attachments.wgsl`,
`crates/molla-solvers/tests/attachments/petiole.rs`,
`bin/molla_cables/src/orchard_sim.rs`, `leafy_draw.rs`, `ACKNOWLEDGEMENT.md` [43].

## Scanned bark and leaf textures (2026-11-19; full video verified)

### Goal
Make the orchard bark more realistic and give the leaves at least 20 different
textures that are used and combined across the canopy.

### Instructions
Rendering only: physics, deposition and the leaf close-up must stay identical.
Use Make; credit every downloaded asset.

### Discoveries
ambientCG (CC0) has photoscanned leaf atlases with opacity and normal maps; six
ovate, serrated, apple-like sets hold 33 separate upright leaves. Wood cells
are tapered 8-sided prisms on local +Z built with `from_rotation_arc`, so
cylindrical bark coordinates line up between bonded cells. Scans are brighter
and more olive than apple leaves; Bark005 is dark olive and read as blue-grey
sticks in canopy shade until warmed toward reddish-brown young shoots.

### Accomplished
`molla_foliage_bake` (`make import-foliage-textures`) downloads the nine
hash-pinned ambientCG 2K zips, cuts out each leaf by its opacity component,
aligns it centroid→tip, drops the petiole and resamples it so its smoothed
outline fills the simulated blade outline (serrations, holes, spots, veins and
normals kept), bleeds edges for mips and packs `data/foliage/`. The renderer
loads 33 leaf layers and 3 bark layers as texture arrays with
coverage-preserving alpha mips. Each leaf uses one scan, a mirror flag, part
of a second scan's palette and the existing tint; undersides are paler.
Bark blends scaly trunk (Bark004), fissured limb (Bark003) and warmed young
shoot (Bark005) by cell radius with patchy noise; u wraps a whole number of
tiles, v accumulates along the bond graph from each root, normals follow the
cone. Both use derivative-based normal mapping. Leaf polygons extend 6% past the
outline so the alpha defines the margin; vein overlay triangles are gone.
Blade outline lost its 9% lobing so physics outline, texture and close-up
outline coincide. 52 orchard, 32 leafy-tree, 10 cable and 18 apple-tree tests
pass. 17 s preview: same leaf 559, same ≈76,222 droplets; inspected.

Full capture `artifacts/ag-spray/textured/ag-orchard-full-1080p.mp4`: 4,200
frames, 140 s, 1080p30; `make verify-ag-spray` passes all 15 checks. Physics
matches the untextured export exactly (14,374 attached; canopy 247.57 mL;
ground 417.53 mL; 6,575 wet leaves; leaf 559, 97 parcels, window
7.67–23.07 s). Wall 1,910 s versus 1,231 s untextured.
SHA256 `0a4cca2af894c747f5423d9f9b2dfb5166e1827efe42ccc7ba880f6b210e7714`.

### Next Steps
None required. Capture wall time rose ~55%; not yet profiled whether the CPU
leaf normals/UVs or the GPU texture sampling dominate.

### Relevant Files
`bin/molla_cables/src/foliage_layout.rs`, `foliage_bake.rs`,
`studio_foliage.rs`, `studio.rs`, `studio.wgsl`, `leafy_draw.rs`,
`branch_draw.rs`, `data/foliage/ATTRIBUTION.md`, `ACKNOWLEDGEMENT.md` [42].

## First-tree leaf close-up with droplet count (2026-11-19; full video verified)

### Goal
In the full AG video, after the first tree is sprayed and the drone moves on,
hold the camera on one leaf of that tree and show how many droplets landed on it.

### Instructions
Keep the approved drone, boom, spray, wind and leaf settings. Use Make. The
count must come from the simulated deposition, not an animation.

### Discoveries
Codex never received this request; its last turn only re-recorded the 140 s
square-boom video (completed and verified, see next section). Each parcel
carries ~2.9 µL, so one parcel stands for ~480–1,650 physical droplets of
160–240 µm; the hit test used the collision blade frame, while the drawn leaf
uses the wood-driven frame, so landing points need a frame mapping to render.

### Accomplished
`Spray` now records every leaf hit as a `Deposit` (leaf, base→tip fraction,
lateral offset, struck face, time, litres, equivalent droplets). Hit acceptance
is unchanged. `leafy_draw::surface` is the single drawn-blade surface used by
both the leaf mesh and the water beads. New `leaf_focus` picks the attached
first-tree leaf (|normal_y| > 0.5) with the most landed parcels 0.6 s after the
drone departs, zooms over 3.2 s, holds 9 s and returns to the drone. Beads use
the parcel's merged volume and a new water material (13) in `studio.wgsl`.
Overlay shows equivalent droplets, parcels, µL and the tree context. Report
gains `focus_leaf`; `verify_ag_spray.jq` gains `leaf_close_up`.
`--no-leaf-focus` restores the previous camera. 49 orchard tests and 6
droplet tests pass. GPU apps segfault inside NVIDIA's driver init when started
from Claude's nested devshell env; captures run from an `env -i` launcher.
26 s preview: leaf 559 of tree 1 selected, 97 parcels ≈ 76,222 droplets,
280 µL (tree mean 11.9 parcels per wet leaf). It was framed too wide, so
the hold now sits at 2.3 leaf lengths with a projected outline of the
counted leaf. The 17 s re-preview `preview2.leaf.png` was inspected and approved.

Full capture `artifacts/ag-spray/leaf-focus/ag-orchard-full-1080p.mp4`: 4,200
frames, 140 s, 1080p30, wall 1,231 s. `make verify-ag-spray` passes all 15
checks including `leaf_close_up`. Physics is identical to the square-boom
export (shed 1,066; canopy 247.57 mL; ground 417.53 mL; 6,575 wet leaves).
Close-up window 7.67–23.07 s; leaf 559 stays attached and no further
parcels land after the shown count. Inspected `.leaf.png` and the 30 s
keyframe (camera back on the drone).
SHA256 `6dcb0ff61f55ad7c931f593bd09657c34296b629759a3d7c4a5b69d91ad4d983`.

### Next Steps
None required. The native viewer draws beads when zoomed within 1.6 m but has
no leaf-focus camera.

### Relevant Files
`bin/molla_cables/src/leaf_focus.rs`, `orchard_spray.rs`, `leafy_draw.rs`,
`drone_orchard.rs`, `studio.wgsl`, `verify_ag_spray.jq`.

## Full square-boom orchard video (2026-11-19; verified)

Completed: 4,200 frames, 140 s, 1920x1080 at 30 fps; capture and
`make verify-ag-spray` both exit 0. Mission Complete, 665 mL emitted, 248 mL on
leaves, 418 mL on the ground, none airborne; 6,575 wet leaves, peak 7,420
parcels, no capacity stops; 14,374/15,440 leaves attached. Wall time 1,169 s.
SHA256 `7e0352eb8bdbd1b47430782207cbffd13e3666ebefc67fe3ffe15a9e279ded8b`.

### Goal
Record and deliver the complete 140-second orchard mission with the approved
industrial aircraft and square spray boom, not another short preview.

### Instructions
Preserve approved settings, use Make, acknowledge sources and keep prior exports.

### Discoveries
The full AG mission has sixteen two-second spray stops. Existing Make gates
check mission completion, flight hover, liquid conservation, canopy deposition,
finite state, hidden air vapor, and full 1080p30 video decode. Engram is unavailable.

### Accomplished
Inspected the current square-boom model, capture path, verification gate and
running native instance. Selected `artifacts/ag-spray/square-boom/` for the new
full export; all prior footage remains untouched. No physics changes requested.
Started the Make-driven 140 s / 4,200-frame 1080p30 capture at
`square-boom/ag-orchard-full-1080p.mp4`, with `full-frames/` and `full-state.json`.
Stopped the previous verified native process to avoid GPU contention. Capture
wrapper PID 2917215; log `/tmp/molla-square-full.log`, capture exit marker
`/tmp/molla-square-full-capture.exit`. On success the wrapper runs
`make verify-ag-spray`; log `/tmp/molla-square-full-verify.log`, exit marker
`/tmp/molla-square-full-verify.exit`. Source parameters are unchanged.
Actual capture PID 2917449. Reran `make test-ag-spray`: six droplet and 45 orchard
tests pass, one explicit GPU upload ignore. Saved test output, exact command,
source hashes and a local copy of `ACKNOWLEDGEMENT.md` beside the capture.
Inspected the actual six-second output frame: square boom, white spray, no UI
overlay. Capture is progressing without recovery errors; completion is pending.

### Next Steps
Capture the full run, verify all stops/video/state, inspect actual frames and
deliver the finished MP4 with its source credits.

### Relevant Files
`artifacts/ag-spray/square-boom/`, `Makefile`,
`bin/molla_cables/verify_ag_spray.jq`, `plans/DRONE_ORCHARD.md`.

## Square spray boom (2026-11-19; verified)

### Goal
Replace the line-shaped spray boom with a square beneath the aircraft.

### Instructions
Keep the approved drone, wind, pressure, droplets and leaf settings; use Make
and patch tools. Keep source acknowledgements current.

### Discoveries
The visual straight rail and four collinear physical injection points were
separate constructions. Both must move; changing only the rail leaves the actual
spray in a line. Engram tools are unavailable; this section records the session.

### Accomplished
Selected a centered 0.56 m square with one nozzle per corner, preserving the
existing outlet height. The boom perimeter and physical emitters now share
`nozzle_local`; added four supports and routed the feeds to the new frame.
Kept the total flow, pressure, nozzle velocities, rotor/flight and leaf settings.
Added a square-layout/tilted-emission/equal-flow test and report coordinates.
Acknowledged this as an original layout extension of the existing references.
Six droplet and 45 orchard tests pass; one explicit GPU upload ignore. This
includes square geometry, equal outlet flow, tilted body-space emission and
visible nozzle registration. Parenthesized the array-map expression inside
`serde_json::json!` to avoid its array-literal parsing ambiguity.
Captured and inspected the actual square-frame GPU model. The 12-second 1080p30
turntable and four PNG angles are in `artifacts/ag-spray/square-boom/`; video
frame-count/resolution and full decode verification pass. Native Make relaunch
is verified on workspace 5, PID 2863905, after normal 14.917-second settling.
Inspected `native-preview.png` and `native-spray.png`; the latter shows the square
boom spraying at the first tree (6.4 s, pump ON, 29 mL applied). Focused formatting
and diff hygiene pass. All earlier videos are preserved; no commit or push.

### Next Steps
Requested square boom, physical nozzle relocation, tests, rendered inspection
and native relaunch are complete. No full-route video was requested or remade.

### Relevant Files
`bin/molla_cables/src/ag_drone.rs`, `bin/molla_cables/src/orchard_spray.rs`,
`bin/molla_cables/src/drone_orchard.rs`, `ACKNOWLEDGEMENT.md`,
`artifacts/ag-spray/square-boom/`, `/tmp/molla-square-native.log`.

## Reference-based agricultural drone proportions (2026-11-19; verified)

### Goal
Replace the toy-like agricultural drone silhouette with a proportionate industrial
aircraft, using real agricultural drones as references.

### Instructions
Preserve the approved spray, leaf retention and flight physics. Acknowledge
references. Four or six rotors are acceptable; use Make and patch tools.

### Discoveries
DJI AGRAS T25 and XAG P100 Pro are both quadrotors. DJI lists a 1,925 mm diagonal
wheelbase, 1,270 mm propellers and 2,585 x 2,675 x 780 mm unfolded envelope.
The current procedural aircraft has an oversized domed canopy and tall silhouette.
Physical rotor hubs, radii, phases and spray anchors must remain registered with
the visible model. Engram tools are unavailable in this session.

### Accomplished
Read the mesh, renderer and physical rotor configuration; researched official DJI
and XAG product/specification sources. Downloaded reference images to /tmp for
inspection. Rebuilt the procedural model as a compact industrial quad: flattened
chassis, battery and carry handle, tapered molded tank, hinges, fittings, motor
details, slender landing gear and shared-location nozzles. Original geometry;
DJI/XAG references acknowledged in [41] and `data/drone/AG_ATTRIBUTION.md`.
Kept all existing dynamics and spray parameters. Added distinct matte and metal
materials, keeping textured Mavic IDs 8–9 separate from new AG IDs 10–12.
Added a Make-driven 1080p turntable with four saved views. Baseline preserved at
`artifacts/ag-spray/industrial/before/`; first new render at `industrial/v1/`.
Two geometry tests pass: finite nondegenerate triangles/unit normals, low-profile
envelope, shared nozzle endpoints, and blade/hub registration with physical air
disks at multiple phases. Reduced small-component tessellation to stay under
25,000 triangles. Inspected the front and opposite three-quarter GPU renders.
The close-ups revealed shadow acne on flat hardware; doubled the receiver bias
only for AG materials, then rendered again and verified the striping disappeared.
Included aircraft material IDs in shadow-frustum sizing. Cached static arm/motor
geometry as well as the body, with a fallback for different physical arm lengths.
Final turntable: `artifacts/ag-spray/industrial/model.mp4` (12 s, 1080p30).
Six droplet tests and 43 orchard tests pass; one GPU-only upload test is ignored.
Started an 18-second orchard preview at `industrial/orchard-preview.mp4`,
log `/tmp/molla-industrial-preview.log`, exit `/tmp/molla-industrial-preview.exit`.
The final mesh has 16,896 triangles, a 0.456 m visual height and 1.400 m swept
rotor envelope. Added a third regression for changing arm lengths across cached
meshes; all three geometry tests pass. Final turntable decode/frame/resolution
verification and focused formatting pass. Orchard settling remains 14.917 s,
all 15,440 leaves initially attached. The 18-second actual orchard preview
completes with exit 0, 540 frames, 1080p30; full decode verification passes.
Comparison with the saved approved 0.85-strength probe is exact for flight
dynamics, spray diagnostics, leaf trajectories, settings and shedding (245).
There are zero wood failures and zero root error. Capture wall time is 151.356 s
after settling; this is offline footage, not a real-time performance claim.
Final CPU checks: six droplet tests and 44 orchard tests pass, one explicit GPU
upload ignore. Compilation, focused formatting and diff hygiene pass.
Inspected actual six-second spraying frame as well as multiple model angles.
Credits, checksum manifest and `physics-comparison.json` accompany the previews.
Native Make relaunch is verified after the normal settling period: PID 2683776,
workspace 5, title `Molla · drone orchard`. Inspected `industrial/native-preview.png`.
Log `/tmp/molla-industrial-native.log`. No commits or pushes were made.

### Next Steps
Requested model redesign, rendered verification and native relaunch are complete.
The full 140-second movie has not been re-rendered for this model-only request;
all previous full exports remain untouched.

### Relevant Files
`bin/molla_cables/src/ag_drone.rs`, `bin/molla_cables/src/studio.wgsl`,
`bin/molla_cables/src/studio.rs`, `bin/molla_cables/src/ag_model_lab.rs`,
`bin/molla_cables/src/drone_orchard.rs`, `Makefile`, `ACKNOWLEDGEMENT.md`,
`bin/molla_cables/data/drone/AG_ATTRIBUTION.md`, `artifacts/ag-spray/industrial/`.

## Whiter spray and another 30% fewer falling leaves (2026-11-19; video verified)

### Goal
Make the fine mist whiter and reduce shedding by another 30% relative to the
current 1,482-leaf full run, targeting about 1,037 falling leaves. Update the video.

### Instructions
Preserve the approved settling physics, drone, wind, spray dose and fine droplet
appearance. Change physical attachment strength, not visible leaf count.

### Discoveries
The spray shader uses blue-tinted RGB (0.80, 0.89, 0.96), and density-kernel
opacity is 0.007. The current physical leaf strength is 0.74. A further 30%
reduction is relative to this version, not the original 1,869-leaf baseline.

### Accomplished
Changed spray color to neutral white and kernel opacity to 0.012, keeping
billboard radii and core grains unchanged. No spray physics changed. Started
an 18-second GPU probe at strength 0.82. Generalized the retention gate's target
and tolerance with backward-compatible 20% defaults; new capture will use the
current settling video as baseline and a 30% target. Engram is unavailable.
The 0.82 trial triggers adaptive wood-step recovery up to 16 subdivisions,
unlike the previous 0.74 run. This makes it substantially slower; check both
measured detachment and adaptive recovery before selecting the final preset.
Existing settling, drag and wind physics have not been edited. Positive
synthetic 30% and real historical 20% retention checks both pass after the gate
generalization.
The completed 0.82 probe sheds 272 leaves (20.23% fewer than 341), takes 553.697
wall seconds and requires repeated 16-way wood stepping. The 0.85 probe sheds
245 (28.15% fewer), takes 116.828 seconds and has zero recovery events. Selected
0.85 as the new default. This illustrates why count and recovery must both be
measured rather than assuming a monotonic stability cost with strength.
Six droplet + 42 orchard tests pass, one explicit GPU upload ignore. Rerun these
after setting the final default and verify the actual full-route reduction.
Final 0.85 tests pass: six droplet + 42 orchard tests, one explicit GPU ignore;
compilation, focused format and diff checks pass. Updated the strength-test
fixture displacement from 30 to 37 mm so its load lies between the new 0.85
and reference 1.0 release thresholds; motion-equivalence assertions remain.
The full capture is now running at
`artifacts/ag-spray/white/ag-orchard-full-1080p.mp4`, 140 seconds/4,200 frames.
Log `/tmp/molla-ag-white-full.log`, exit marker `/tmp/molla-ag-white-full.exit`;
state `white/full-state.json`, frames `white/frames/`. Preserve all old exports.
Capture PID 1656173, wrapper PID 1655943. Inspected the six-second real GPU
render: neutral, brighter white mist with unchanged soft footprint size. No
adaptive retries in the opening 13 seconds. Final measured reduction is pending.
The full capture subsequently completes with exit 0 and zero adaptive recovery
events. Final shedding is 1,066 versus 1,482: **28.070% fewer**, within the
preselected 30% +/- 3-percentage-point acceptance band, not exactly 30%.
14,374 leaves remain attached. `make verify-ag-spray` and the explicitly
configured `make verify-ag-retention` pass, including full MP4 decode.
All 16 two-second stops complete. The 140-second/4,200-frame H.264/yuv420p movie
is 1920x1080 at 30 fps and 125,946,157 bytes. SHA256:
`a32a866428d96571d9984b68c20695fec620bfa472439e7eb166d27c0134584e`.
All spray settings except appearance match the preceding video. No drop rises
above injection; upward liquid residence is 1.864%, final airborne water is zero.
0.665108 L emitted = 0.309919 L canopy + 0.355189 L ground; no capacity stops.
All wood/leaf states are finite, counts match and roots/wood remain intact.
Inspected actual frames at 6/40/60/90/120 seconds and the 50-second contact sheet.
Reports, state, credits and checksum are in `artifacts/ag-spray/white/`.
The full offline capture takes 1,862.997 wall seconds; no real-time performance
claim. Prior videos are preserved. The new default is 0.85, not the rejected
0.82 trial. Six droplet + 42 orchard tests pass; one explicit GPU upload ignore.
Synthetic unchanged-shedding and changed-wind rejection checks pass too.
Native Make relaunch is verified on workspace 5, PID 1832213, after normal
14.917-second settling. Inspected `white/native-preview.png`; the UI displays
strength 0.85. Log `/tmp/molla-ag-white-native.log`. Final diff hygiene passes.

### Next Steps
Requested appearance and approximately 30% further shedding reduction are
verified in the updated full video; native relaunch is verified too. No remaining
work for this request.
Report the actual 28.070% reduction rather than calling it exactly 30%.

### Relevant Files
`bin/molla_cables/src/studio_spray.wgsl`, `bin/molla_cables/src/drone_orchard.rs`,
`bin/molla_cables/src/orchard_sim.rs`, `bin/molla_cables/verify_ag_retention.jq`,
`Makefile`, `artifacts/ag-spray/white/`.

## Stronger spray settling (2026-11-19; full video verified)

### Goal
Keep fine-looking spray from behaving like smoke in moderate swirling updrafts.

### Instructions
Preserve the approved aircraft, wind, spray pressure and 20% reduced leaf shedding.
Use physical motion, not scripted downward paths. Acknowledge sources.

### Discoveries
Gravity is already 9.81 m/s2. The 30–60 micrometre droplets have very small
settling slip and rapidly entrain even after the previous 2x nozzle-speed boost.
The completed old video reports a maximum liquid horizontal drift of 174.454 m.

### Accomplished
Selected 160–240 micrometre water droplets for stronger gravitational settling
under the existing Schiller–Naumann drag, without changing gravity, wind or nozzle
launch conditions. Keep the mist renderer and parcel rate unchanged. Added
upward liquid residence-time and rise diagnostics; strong physical updrafts remain
possible. OpenFOAM's primary drag reference was rechecked; existing citations
[38]–[40] apply. Engram tools remain unavailable.
Still-air solver measurements are 0.520/0.709/0.898 m/s settling at
160/200/240 micrometres. A 0.35 m/s updraft no longer reverses these drops;
a 2 m/s updraft still does. In a matched 30-second sinusoidal-updraft unit
fixture, old aerosol deposits 0%, rises up to 11.04 m, and spends 74.17% of its
airborne liquid residence time rising. The new preset deposits 100%, never rises
above injection, and spends 22.30% of airborne residence rising. Maximum drift
is 23.04 versus 37.34 m; some wind-carried motion remains. These are controlled
solver tests, not actual orchard footage. Original aerosol dispersion/convergence
gates are preserved, with separate heavier-droplet spread checks.
Six droplet and 42 orchard tests pass (one explicit GPU upload ignore), plus
compilation and focused formatting. Adding report fields hit serde_json macro
recursion depth; grouping settling diagnostics in a separate object fixes that
without raising the crate recursion limit. No existing source contract removed.
Full 140-second capture is running at
`artifacts/ag-spray/settling/ag-orchard-full-1080p.mp4`, log
`/tmp/molla-spray-settling-full.log`, exit marker
`/tmp/molla-spray-settling-full.exit`. The earlier full video is preserved.
Capture PID 192575, wrapper PID 192363. Inspected the actual six-second image:
the spray remains fine-looking and travels downward in an expanding plume.
Added `make verify-ag-settling` for unchanged aircraft/leaf/wind/dose parameters,
reduced full-route drift/residual water and bounded upward residence/rise.
Synthetic positive and changed-wind/excessive-rise rejection checks pass;
these fixtures are not actual GPU measurements. The real full report is pending.
The subsequent complete GPU run passes `make verify-ag-spray`,
`make verify-ag-retention` and `make verify-ag-settling`. It records 140 seconds,
4,200 frames at 1920x1080/30 fps, H.264/yuv420p; full decode passes. Size is
126,249,925 bytes; SHA256
`484c7932ec77613b4c059561b9ebb1f4226605110a1652fe16dcc60a32a185a9`.
Maximum drift falls from 174.454 to 17.066 m; zero liquid remains airborne at
the finish versus 15.851 mL before. No parcel rises above its injection height.
Only 1.866% of liquid-weighted airborne residence involves upward movement;
upward motion below injection remains physically possible, not clamped out.
0.665108 L emitted is conserved as 0.299200 L canopy + 0.365907 L ground.
All sixteen two-second stops complete, 5,389 wet leaf samples, 9,591 peak parcels,
no capacity stops or nonfinite values. Aircraft telemetry is exactly equal to
the previous recording; leaf shedding stays at 1,482 (20.706% below the original
1,869 baseline). Root drift/wood failure stay zero. Final wood/leaf states are
finite with conserved counts. Actual images at 6/40/60/90/120 seconds and the
50-second contact sheet were inspected. All verification JSONs and credits are
in `artifacts/ag-spray/settling/`. Capture wall time: 1,170.798 seconds, not a
real-time claim. Previous videos are untouched.
Extended the settling regression to the actual 36.374 m/s nozzle launch with
reversing wind; 60 Hz and 1,200 Hz results agree within the existing 6 mm bound.
Final six droplet + 42 orchard tests pass, one explicit GPU upload ignore.
Native `make run-ag-orchard` relaunch is verified on workspace 5, PID 305694,
after normal 14.917-second settling. Inspected `settling/native-preview.png`;
log `/tmp/molla-spray-settling-native.log`. Final diff hygiene passes.

### Next Steps
Requested spray behavior, updated full video and native relaunch are complete;
no remaining work for this request. No general CFD accuracy/calibration claim:
stronger external updrafts can still lift the larger drops.

### Relevant Files
`bin/molla_cables/src/orchard_spray.rs`, `bin/molla_cables/src/drone_orchard.rs`,
`crates/molla-solvers/src/airflow/droplet.rs`, `bin/molla_cables/AG_SPRAY.md`.

## Fewer falling leaves and full AG video (2026-11-19; verified)

### Goal
Reduce leaf detachment by about 20% and deliver the complete updated 1080p movie.

### Instructions
Preserve current fine, downward-focused spray, aircraft, wind and natural leaf
motion. Tune physical attachment strength rather than hiding falling leaves.
The user explicitly requests the full video, not another preview.

### Discoveries
The prior complete 140-second AG video sheds 1,869 leaves at strength 0.67; its
leaf/wind/flight configuration matches the current one. Current spray samples
air one-way, so intervening spray-only changes do not change leaf shedding.
The recent matched 18-second captures all shed 433 leaves. A 20% count reduction
is not equivalent to a 20% strength increase; measure the result.

### Accomplished
Selected an initial strength trial of 0.75, keeping all leaf mechanics except
detachment threshold unchanged. Target is about 346 shed at 18 seconds and
about 1,495 over the complete route. Engram tools are unavailable.
The matched 18-second GPU probe sheds 329 instead of 433 leaves (24.02% fewer)
at 0.75, so selected 0.74 for the full recording to ease that reduction slightly.
Probe report: `artifacts/ag-spray/final/strength075-probe.json`. Updated the
default and added `make verify-ag-retention`, which compares full-duration
reports, identical scene/wind settings and a measured 17–23% shedding reduction.
Its synthetic acceptance, unchanged-shedding rejection and changed-wind rejection
fixtures pass. The 0.75 trial passes five droplet and 41 orchard tests (one
explicit GPU upload ignore), compilation and focused format/diff checks; the
final 0.74 setting is rechecked before capture. Existing sources are acknowledged
in `artifacts/ag-spray/final/CREDITS.md`.
Final 0.74 unit checks pass (five droplet + 41 orchard, one explicit GPU ignore).
The full 140-second/4,200-frame recording is running under detached wrapper
PID 4181902, log `/tmp/molla-ag-retention-full.log`, exit marker
`/tmp/molla-ag-retention-full.exit`. Output:
`artifacts/ag-spray/final/ag-orchard-full-1080p.mp4`; report has the same basename,
state is `final/full-state.json`, keyframes `final/frames/`. Do not deliver a
partial movie or claim full shedding reduction until complete verification.
The running full capture reaches 18 seconds with 341 shed versus 433 previously
(21.25% fewer for that matched opening). Inspected its six-second frame; the
approved fine spray, drone and camera are retained. Capture process PID 4182155.
Compaction checkpoint: the same capture remains live at 1,620/4,200 frames
(54 seconds), with 675 leaves shed. At 40 seconds it shed 512 versus the prior
668; the full-route reduction remains unverified. Engram remains unavailable.
The full capture subsequently finished with exit 0: 140 seconds, 4,200 frames,
1920x1080 at 30 fps, H.264/yuv420p, 127,405,607 bytes. Final shedding is 1,482
versus 1,869, a measured 20.706% reduction; 13,958 leaves remain attached.
`make verify-ag-retention` and `make verify-ag-spray` both pass, including full
MP4 decode and exact frame count. All 16 trees receive their two-second dose;
the mission reaches Complete, no capacity stops, root drift or wood failures.
Liquid totals are 0.665108 L emitted, 0.228307 L canopy, 0.420949 L ground and
0.015851 L airborne; conservation passes. 5,555 leaf samples receive water.
All final leaf and wood position/velocity values are finite, and state counts
match the report. Inspected rendered frames at 6, 40, 60, 90, 120 and 140 seconds
plus the 50-second contact sheet. Verification reports and source credits are
in `artifacts/ag-spray/final/`. SHA256:
`418fde3121d16692181afa3207403fccf8c27db0aea85aec1b80e296b58303bb`.
Capture wall time is 1,245.123 seconds; this is an offline recording, not a
real-time performance claim. The previous complete movie is preserved.
Updated native demo relaunched through `make run-ag-orchard`, PID 119755 on
workspace 5. Settling completes with all 15,440 leaves attached before the
mission starts. Inspected `final/native-preview.png`; the demo is rendering.
Native log: `/tmp/molla-ag-retention-native.log`. Final diff hygiene passes.

### Next Steps
The requested full video, measured shedding reduction and native relaunch are
complete. No remaining work for this request. Physical models remain approximate
and uncalibrated; the verified offline video is not real-time performance proof.

### Relevant Files
`bin/molla_cables/src/orchard_sim.rs`, `bin/molla_cables/AG_SPRAY.md`,
`artifacts/ag-spray/final/`, `plans/DRONE_ORCHARD.md`.

## Double downward injection (2026-11-19; preview verified)

### Goal
Give the spray twice the downward drive without increasing sideways scattering.

### Instructions
Preserve fine mist, physical wind/drag, aircraft motion, dose and previous exports.

### Discoveries
Doubling the entire launch vector would also double lateral launch speed. The
requested directional change is represented as a 2x nozzle-axis velocity gain,
not a claim of calibrated pressure doubling. Fine droplets still relax toward
the real local air; this setting cannot guarantee downward travel against every
updraft and must not be presented as doing so.

### Accomplished
Chose to double the nozzle-frame downward component only, retaining lateral
velocity, aircraft point velocity, droplet size and dispersion. Report both the
base speed and resulting axial speed/effective angle. Existing acknowledgements
[38]–[40] apply; no new physical model is introduced. Engram is unavailable.
The 36.4 m/s regression exposed 7.04 mm timestep error for 45 um droplets with
the old 2 ms drag substeps. Reduced the maximum substep to 1 ms rather than
loosening the 6 mm convergence gate. The regression now passes. Five droplet
tests and 41 orchard tests pass, with one explicit Vulkan upload ignore; log
`/tmp/molla-ag-downward-tests.log`. A new regression verifies exact 2x axial
velocity for 7,200 injection directions, unchanged lateral components, and
unchanged aircraft linear/angular point velocity. Effective cone is about 48.5
degrees; the downward launch range is 27.0–36.4 m/s.
Compilation, focused formatting and `git diff --check` pass. Inspected the actual
six-second render, and verified the complete 18-second 1080p/30 fps preview:
`artifacts/ag-spray/downward/downward-preview-1080p.mp4`, 540 frames, full decode
passed. SHA256 `9a08a2b91fecb6bf19412dccad33c8d18b0b83ac5a781c7c21e095b87b203dee`.
Actual GPU-run RMS spread is 0.188/0.236/0.314/0.453 m at the four measured
depths, lower than the balanced preset but still expanding downward. This is a
modest downstream improvement, not twice the speed throughout the plume: fine
mist still entrains into the unchanged air. Two stops emit 83.138 mL: 32.516 mL
canopy, 35.097 mL ground, 15.525 mL airborne; conservation passes. 744 wet leaf
samples, 13,787 peak parcels, no capacity stops. Flight telemetry, dose, diameter
and dispersion match the prior preview. `verification.json` records the checks.
Capture takes 155.057 wall seconds; no full-route or real-time claim is made.
Native Make relaunch is verified on workspace 5, PID 4083051, after normal
settling. Inspected `artifacts/ag-spray/downward/native-preview.png`; log
`/tmp/molla-ag-downward-native.log`.

### Next Steps
Requested directional gain, convergence fix, tests, preview and native relaunch
are complete. Previous full video is unchanged. If sustained downwards transport is still
insufficient, inspect the resolved air near the nozzles rather than assuming
further launch-speed increases will overcome fine-droplet entrainment.

### Relevant Files
`bin/molla_cables/src/{orchard_spray,drone_orchard}.rs`,
`crates/molla-solvers/src/airflow/droplet.rs`,
`bin/molla_cables/AG_SPRAY.md`, `ACKNOWLEDGEMENT.md`.

## Balanced downward mist (2026-11-19; preview verified)

### Goal
Set the spread between the earlier narrow mist and the latest wide plume, with
less mist hanging around the aircraft and a clearer downward flow.

### Instructions
Keep physically based wind/drag motion, fine droplets and the approved aircraft.
Do not force downward particle positions or remove real wind response.

### Discoveries
The latest preset combines a 110-degree cone with 25% RMS air fluctuations in
all directions. The added fluctuations perturb vertical as well as lateral
motion. Nozzle positions coincide with the ends of the rendered hardware; this
inspection does not establish an aircraft collision or resolved-air defect.

### Accomplished
Chose an 84-degree full cone (midway between approximately 58 and 110 degrees)
and 12.5% dispersion intensity (half the latest preset). Keep jet speed, droplet
size, dose, resolved wind and rendering unchanged. No new model or source is
introduced; existing acknowledgement [40] remains applicable. Engram tools are
unavailable, so this note records the decision.
Implemented the two-parameter retune without modifying the drag solver, flight,
air field or sprite sizes. Nozzle regression now requires at least 13.5 m/s
downward speed at emission (previous edge rays were about 10.4 m/s).
`make test-ag-spray` passes five droplet/dispersion and 40 orchard tests, with
one explicit Vulkan upload ignore. The intermediate-spread regression measures
RMS widths 0.033/0.084/0.147/0.214 m versus 0.050/0.167/0.294/0.429 m with the
stronger dispersion. Timestep refinement and conservation pass. Compilation,
focused formatting and `git diff --check` pass; logs `/tmp/molla-ag-balanced-*.log`.
Inspected the actual six-second preview frame: a tighter, clearer downward mist
with gentle broadening, between the previous narrow and wide versions.
The 18-second 1080p/30 fps capture is complete and its 540-frame count and full
decode pass. Path `artifacts/ag-spray/balanced/balanced-preview-1080p.mp4`;
SHA256 `a9f37c5723e0ec6dcc2081662c67a5936b1d19ec17d70fd06b863efbf1200ad4`.
Real GPU-run RMS widths are 0.216/0.261/0.339/0.473 m at the four depths,
all below the wider preset while still increasing downward. Two stops emit
83.138 mL: 31.503 mL canopy, 35.392 mL ground, 16.244 mL airborne; conservation
passes. 712 leaf samples receive water, peak live parcels 14,053, no capacity
stops. Flight telemetry, size, nozzle speed and dose exactly match the previous
preview. Evidence: `artifacts/ag-spray/balanced/verification.json`. Capture takes
153.006 wall seconds; no full-route or real-time performance claim is made.
Verified native launch after settling on workspace 5, PID 4010084; inspected
`artifacts/ag-spray/balanced/native-preview.png`. Log `/tmp/molla-ag-balanced-native.log`.

### Next Steps
Requested intermediate tuning, checks, preview and native relaunch are complete.
Previous full-length video and previews are preserved; new full-route export
and calibration remain future work.

### Relevant Files
`bin/molla_cables/src/orchard_spray.rs`, `bin/molla_cables/AG_SPRAY.md`,
`ACKNOWLEDGEMENT.md`, `artifacts/ag-spray/balanced/`.

## Wider source and dispersing plume (2026-11-19; preview verified)

### Goal
Spread the mist more from the nozzle and progressively farther down the plume.

### Instructions
Keep the fine droplets, faster jet and physically based motion. Do not fake the
spread with larger sprites or prescribed cone-shaped positions. Credit sources.

### Discoveries
Fine droplets rapidly lose their initial lateral velocity, so increasing only
the nozzle angle cannot prevent narrow streams in a smooth 15 cm air grid.

### Accomplished
Chose a 110-degree full nozzle cone and a seeded continuous-random-walk air
dispersion closure: temporally correlated, zero-mean unresolved air fluctuations
enter droplet drag alongside the resolved GPU wind. This is an approximate
Langevin model, not a calibrated k-epsilon/LES closure. Flight, resolved air,
tree physics, dosage and rendering kernels remain unchanged. Sources consulted:
Ansys Fluent CRW theory and OpenFOAM stochastic particle dispersion. Engram
tools are unavailable; decisions are recorded here.
Implemented the seeded OU process, wider solid-angle injection and physical
spread-crossing telemetry. `make test-ag-spray` passes five droplet/dispersion
tests and 40 orchard tests (one explicit Vulkan upload ignore). Statistical
tests cover target variance, time correlation, timestep independence, reproducible
seeds, invalid-input rollback and progressive physical plume widening. The first
compile required an explicit f64 annotation in a new statistical test; corrected.
Rendering kernels and flight/air/leaf parameters are unchanged. Reference [40]
and `artifacts/ag-spray/spread/CREDITS.md` acknowledge the source concepts.
Uniform-downwash regression measures physical RMS widths of 0.054, 0.168, 0.295
and 0.429 m at the four depths, versus about 0.032 m without dispersion. Halving
the timestep gives 0.051, 0.163, 0.301 and 0.428 m. Compilation, focused formatting
and diff checks pass; logs `/tmp/molla-ag-spread-{tests,widths,check,fmt}.log`.
Rendered and inspected the actual six-second frame: the separate narrow streams
now merge into a broader diffuse cloud. `artifacts/ag-spray/spread/spread-preview-1080p.mp4`
contains 540 frames, 18 seconds, 1920x1080/30 fps; full decode/media verification
passes. SHA256 `285ad35521cdb3c6fdd21fbf339472b579ac0b14bd81fda0d836ab16d7fb9c41`.
Actual GPU-run RMS widths increase 0.218 -> 0.285 -> 0.440 -> 0.902 m at 0.25,
0.75, 1.5 and 2.5 m below injection. All bins have over 23,000 crossings. Two
two-second stops emit 83.138 mL: 27.055 mL canopy, 43.177 mL ground, 12.907 mL
airborne. Liquid conservation passes, 1,525 leaf samples receive water, peak
parcels 13,956, no capacity stops, no tree failures/root drift. Flight telemetry,
jet speed, diameter distribution and total dose match the previous preview.
Evidence is saved in `artifacts/ag-spray/spread/verification.json`. Capture takes
152.911 wall seconds; no full-route, calibration or real-time claim is made.
Native relaunch is verified on workspace 5, PID 3947536, after normal settling;
actual window screenshot inspected at `artifacts/ag-spray/spread/native-preview.png`.
Log `/tmp/molla-ag-spread-native.log`.

### Next Steps
Requested widening, physics checks, preview and native relaunch are complete.
The earlier full-length movie is preserved; an updated full movie and physical
calibration remain separate future tasks.

### Relevant Files
`crates/molla-solvers/src/airflow{,/droplet}.rs`,
`bin/molla_cables/src/{orchard_spray,drone_orchard}.rs`, `ACKNOWLEDGEMENT.md`.

## Wind-carried atomized mist (2026-11-19; preview verified)

### Goal
Replace the still-too-large beads with near-fog particles and triple nozzle speed.

### Instructions
The user accepts approximate physics for now, but requires physically based
motion, not animation. Mist must interact with wind and rotor downwash. Preserve
flight, tree response, previous videos and source acknowledgements.

### Discoveries
Spray already samples the same GPU air field as the aircraft and foliage, and
integrates gravity and Schiller–Naumann drag. Its previous 150–300 um diameters
and enlarged opaque particle footprints still looked like beads. Tiny droplets
rapidly lose nozzle-relative velocity; sustained motion must follow local air,
not a forced threefold multiplier on every trajectory.

### Accomplished
Selected 30–60 um droplets, 18.187 m/s exit speed (3x previous) and 7,200 parcels/s.
Relative pressure becomes 27; relative nozzle flow coefficient becomes 1/3,
preserving the previous dose and tank budget while increasing exit speed.
Rendering uses low-opacity density kernels and subpixel grains at simulated
parcel positions, with no separate animated vapor. Engram tools are unavailable.
Implemented the tuning and wind-sample/drift reporting. `make test-ag-spray`
passes three droplet tests and 39 orchard tests (one explicit Vulkan upload
ignore), including reversed crosswind, downwash, 30–60 um timestep convergence,
full-route flight and tank conservation. Log `/tmp/molla-ag-mist-tests.log`.
Compilation, focused formatting and `git diff --check` pass. Rendered and inspected
the 6/12-second frames of `artifacts/ag-spray/mist/mist-preview-1080p.mp4`:
18 seconds, 540 frames, 1920x1080 at 30 fps; full decode/media checks pass.
SHA256 `5865b122f82fcf3b4e8e244d84f92f625972e86c70d9180b69ba44201fa6bfa8`.
The real GPU capture sampled air 4,169,063 times (maximum 8.664 m/s), with maximum
horizontal droplet drift 29.628 m. Some water remains airborne instead of being
clipped or animated onto leaves: 83.138 mL emitted, 39.534 mL canopy, 20.984 mL
ground, 22.621 mL airborne. Conservation and no-capacity-stop checks pass; peak
live parcels 14,025. Flight telemetry matches the earlier preview exactly.
Evidence is in `artifacts/ag-spray/mist/verification.json`; credits accompany it.
This demonstrates physical coupling and stability, not experimental validation.
Capture took 152.030 wall seconds; no real-time performance claim is made.
Native Make launch is verified on workspace 5, executable PID 3871501; log
`/tmp/molla-ag-mist-native.log`. Actual window screenshot inspected at
`artifacts/ag-spray/mist/native-preview.png` after the normal settling phase.

### Next Steps
Requested mist tuning, wind-response tests, preview and native relaunch are
complete. The previous full-length movie remains unchanged; a new full GPU
route/video and accuracy calibration are future work.

### Relevant Files
`bin/molla_cables/src/{orchard_spray,drone_orchard}.rs`,
`crates/molla-solvers/src/airflow/droplet.rs`, `bin/molla_cables/AG_SPRAY.md`.

## Finer, denser, faster spray (2026-11-19; preview verified)

### Goal
Increase spray pressure and make droplets smaller, more numerous and faster.

### Instructions
Keep the approved AG model, flight mechanics and two-second tree stops. Preserve
the previous full movie; acknowledge pressure-scaling sources.

### Discoveries
The initial spray uses 250–500 um droplets, 1,800 parcels/s, 3.5 m/s nozzle speed
and 0.012 L/s total flow. TeeJet's manufacturer reference describes square-root
flow-versus-pressure scaling. The 0.75 L tank can cover all sixteen stops at 3x
relative pressure: 0.665108 L required, about 0.084892 L remains.

### Accomplished
Set relative pressure to 3, deriving both flow and exit speed from its square
root (6.062 m/s; 0.020785 L/s). Set physical diameter to 150–300 um, parcel count
to 3,600/s and billboard radii to about 60% of their old values. Added nozzle
speed/size/density and reservoir-budget regressions and report metadata. Full
verification derives expected dose from report flow, retaining compatibility with
the old movie's report. Cited TeeJet as [39]. Engram tools are unavailable.
Two droplet and 38 orchard unit tests pass, with one explicit Vulkan upload ignore;
the orchard executable check passes. Logs `/tmp/molla-ag-pressure-{tests,check}.log`.
The old full movie still passes the updated verifier using its original flow.
The agent-launched old native PID 3641112 was closed after executable verification.
Recorded an 18-second updated preview under detached wrapper PID 3748433;
log `/tmp/molla-ag-pressure-preview.log`, marker `/tmp/molla-ag-pressure-preview.exit`.
The 18-second 1080p/30 fps preview is complete and its frame count/full decode
pass. SHA256 `b5062f02de04a26b5c60371b02727966eccbc3893a3759b2b3e21a4fbfd7245f`.
Actual 6-second render was inspected: visibly smaller and more numerous droplets.
Two measured stops still receive two seconds each. 83.138 mL emitted, 46.159 mL
intercepted by leaves, 36.979 mL on the ground; liquid conservation passes and
there are no capacity stops (peak 3,873 parcels). All flight telemetry samples
exactly match the same interval of the previous full movie: no flight retune.
Full-route liquid demand is covered by the unit budget test; a full new 140-second
GPU spray run is not claimed. The original full movie remains at lower pressure.
Relaunched the native AG example on workspace 5, executable PID 3789840, log
`/tmp/molla-ag-pressure-native.log`. Inspected the actual native window screenshot
`artifacts/ag-spray/pressure/native-preview.png`: AG model, follow camera, orchard
and simulation are rendering after settling. Preview credits are in
`artifacts/ag-spray/pressure/CREDITS.md`.

### Next Steps
Requested spray tuning and preview verification are complete. The previous full
movie is preserved; a full updated export is a separate follow-up.

### Relevant Files
`orchard_spray.rs`, `drone_orchard.rs`, `verify_ag_spray.jq`, `AG_SPRAY.md`,
`ACKNOWLEDGEMENT.md`, `artifacts/ag-spray/pressure/`.

## Agricultural spraying example (2026-11-19; delivered)

### Goal
Replace the rejected Mavic look with an agricultural drone that stops above every
tree and sprays down briefly. This supersedes waiting for a reference image.

### Instructions
Use Make, preserve flight and leaf mechanics, and acknowledge sources. Retain the
previous examples/videos; add AG mode rather than silently replacing their route.

### Discoveries
The current GPU air sampler supports batched extra parcel queries without an
additional readback. Existing bounded-jerk trajectory and rigid quadrotor control
can stop at each tree; a state machine must gate spray on measured pose/speed,
not just scheduled arrival. DJI's official T25 page provides layout inspiration;
OpenFOAM documents the Schiller–Naumann droplet drag correlation.

### Accomplished
Chose an original compact four-rotor AG mesh with a white tank, green detailing,
carbon frame, landing skids, boom and four downward nozzles. Added optional AG
mission/parcel modules, a reusable water-droplet integrator and a translucent,
depth-tested spray renderer. One-way water parcels sample the actual GPU wind;
volume is accounted across tank, air, leaf interception and ground deposition.
The aircraft mass remains fixed; this is not a calibrated DJI or pesticide model.
References added as [38]. Engram tools are unavailable; decisions are saved here.
The first GPU-rendered 18-second preview shows two measured stops and downward
spray; 48 mL emitted, about 28 mL intercepted by leaves and 20 mL on the ground,
280 wet leaf samples, no capacity stops. Reviewed actual 2/6/12-second frames.
Reduced displayed parcel size and doubled sampling density for a finer spray.
The emitter now rechecks measured pose and speed after the flight step before
committing each dose tick. Native AG mode has a toggleable follow camera.

CPU checks: two droplet tests (settling, crosswind, timestep refinement, invalid
input/overflow rollback), 37 orchard tests including the full integrated 16-tree
mission, plus swept moving-leaf deposition and liquid conservation. The first
ad-hoc long-distance test exposed a fixed-leg duration assumption; a leg now
stretches to at least its bounded-profile duration. All-target compilation passed
before the final emitter/follow-camera changes; final recheck remains pending.

Full 140-second recording started through `make capture-ag-orchard`; output
`artifacts/ag-spray/ag-orchard-full-1080p.mp4`, state `full-state.json`, keyframes
`full-frames/`, log `/tmp/molla-ag-full.log`, marker `/tmp/molla-ag-full.exit`.
The final tests preceding capture are in `/tmp/molla-ag-final-tests.log`.
Those final tests pass (2 droplet + 37 orchard, 1 explicit GPU-upload ignore), as
do the final all-target compilation, formatting and diff checks. The finer spray
was inspected at 6 seconds in the full-run keyframe.
The first long export's exec process group exited with signal status 143 after
600/4200 frames, without a simulation error or completion marker; it is not a
completed movie. Its PID-suffixed partial file is preserved. Restarted under a
detached session, wrapper PID in `/tmp/molla-ag-full.pid`, log
`/tmp/molla-ag-full-detached.log`, same final target and exit marker. Do not claim
full GPU route acceptance until the final report and media gates pass.

Final capture completed successfully: **140 seconds, 4,200 frames, 1920x1080 at
30 fps**, `artifacts/ag-spray/ag-orchard-full-1080p.mp4`. SHA256
`333b4830896a35eefcd5713e8b624db7ec5e16b7a3c06e5f9e8a8bf5c089b4d3`.
`make verify-ag-spray ORCHARD_SECONDS=140` passes all mission/liquid gates and
the full ffmpeg decode/frame checks. Log `/tmp/molla-ag-full-verify.log`, marker
zero. Each of 16 trees has exactly two seconds of spray; maximum measured nozzle-on
speed 0.09994 m/s and position error 0.04290 m. Mission ends Complete, back at the
first tree, final speed about 0.000403 m/s. All 16 trees receive water.
384 mL emitted: 204.827 mL intercepted by leaves and 179.173 mL on the ground;
366 mL remains in the tank, none remains airborne. 2,867 leaf samples wetted;
peak 2,347 parcels, zero capacity stops. The final leaf/wood state is finite,
roots stay fixed and no wood fails. 13,571/15,440 leaves remain attached.

Inspected actual GPU frames at 2, 6, 12, 20, 60, 100 and 140 seconds, including
spraying, pump-off transit and final hover. References accompany the video as
`artifacts/ag-spray/CREDITS.md`. Capture cost 1,145.04 wall seconds, so this is
not realtime performance evidence. Existing unrelated work and old videos are
preserved. No commit or push was requested or made.
Native AG mode is running on workspace 5, verified PID 3641112, with follow-camera,
mission/pump status and no displayed error. Actual 3348x1348 screenshot at 0.9 s:
`artifacts/ag-spray/native-preview.png`; log `/tmp/molla-ag-native.log`. The single
displayed air-frame cost was 113.9 ms, not a realtime acceptance measurement.
The launch uses a detached session so closing an exec cell does not close the UI.

### Next Steps
No required work remains for this AG example and full movie. Hardware calibration,
liquid feedback/mass changes and pesticide efficacy remain explicitly out of scope.

### Relevant Files
`ag_drone.rs`, `ag_mission.rs`, `orchard_spray.rs`, `studio_spray.{rs,wgsl}`,
`orchard_sim.rs`, `drone_orchard.rs`, `airflow/droplet.rs`, `ACKNOWLEDGEMENT.md`.

## Full Mavic orchard movie (2026-11-18; stopped by user)

### Goal
Deliver the complete orchard movie with the new Mavic model, not just its close-up.

### Instructions
Keep the accepted flight, wind and leaves. Preserve previous movies and credits.
User stopped recording because they dislike this Mavic model and will provide a
reference image. Wait for that image before further drone modeling or recording.

### Discoveries
The existing full 78-second movie still uses the old model; the new Mavic preview
is only the 30-second isolated flight laboratory.

### Accomplished
Started `make capture-drone-orchard ORCHARD_SECONDS=78` to a new path:
`artifacts/drone-mavic/full/mavic-3-orchard-1080p.mp4`. Copied model credits/license
alongside it. Closed only the verified agent-launched orchard PID 3361553 to avoid
GPU contention. Render log `/tmp/molla-mavic-full-render.log`, exit marker
`/tmp/molla-mavic-full-render.exit`. Engram tools remain unavailable.
Restarted during initialization, before any movie existed, with `--verify-flight`,
`--state .../final-state.json` and `--keyframes .../keyframes` to retain physical
acceptance and visual evidence. The prior 75% leaf-retention gate is unchanged;
its already-known failure may produce exit 2 after successfully finalizing media.
On the user's stop request, terminated the verified capture PID 3409365 during
initial vegetation settling. No completed full Mavic video was produced. The
native orchard had already been closed and has not been reopened.

### Next Steps
Wait for the user's reference image. Replace the visual model to resemble that
reference while preserving flight physics; only then resume video work as directed.

### Relevant Files
`artifacts/drone-mavic/full/`, `Makefile`, `bin/molla_cables/src/drone_orchard.rs`.

## Mavic visual replacement (2026-11-18; delivered)

### Goal
Replace the rejected procedural drone with a recognizable DJI Mavic 3 asset.

### Instructions
The user dislikes the previous model, not the improved flight physics. Preserve
flight/controller/air/leaf parameters and retain resource acknowledgements.

### Discoveries
A 12,701-triangle Mavic 3 derivative is publicly distributed by Skyfront, with
explicit CC BY 4.0 attribution to llirikslon and reduction/conversion credits.
It retains four propeller groups and two PNG textures. The 127,011-triangle source
is mirrored by Drone-Customisation-Workshop. No simulation/game code is imported.

### Accomplished
Imported the checksum-pinned lower-poly asset and CC BY license/credits. Source
textures/normals are retained, visual hubs map onto the existing simulated rotor
frames, and propellers use integrated motor phases. The orchard's 5.6 kg aircraft
remains an illustrative simulation, not physical specifications of a real Mavic.
The first GPU render exposed material-ID interpolation noise at integer thresholds:
interpolated 8.0 selected the wrong material on some pixels. Fixed with a flat
material varying; the rerender is clean with shadows retained. The source has no
coincident duplicate triangles. Engram tools are unavailable; decisions are here.
The workstation's global Git ignore hides all `data` directories, including the
embedded model and license. A checkout-level exception keeps the example's
bundled source assets visible to Git; nothing was staged or committed.
Close-up views at 1.5, 3.5 and 19 seconds were inspected on Vulkan. The delivered
`artifacts/drone-mavic/mavic-3-preview-1080p.mp4` is 30 seconds, 900 frames,
1920x1080 at 30 fps; full decoding and all motion gates pass. SHA256:
`3e96d0e9807d83a1c17cbc71f4503e870773a92acb7468d8c9bebe678a98e530`.
All 900 trajectory telemetry samples exactly equal the previously accepted
`flight-control-final-1080p.json`. This is a visual replacement, not a retune.
Asset credits are copied alongside the movie. Earlier diagnostic movies are not
the delivery; the old 78-second orchard video still has the previous model.

Regression evidence: orchard 32 unit passes, apple 16, leafy 29, plus both tree
Vulkan verification scenarios and the separately run studio Vulkan upload test.
The normal unit invocations explicitly skip five GPU-only tests across these
binaries; that is not a claim that all five ran. All-target example compilation,
focused formatting and `git diff --check` pass. Existing unrelated warnings remain.
Logs: `/tmp/molla-mavic-{regression,verify,check,format-check}.log`.
Native preview relaunched through `make run-drone-orchard`, PID 3361553, Wayland
workspace 5. Actual window capture `artifacts/drone-mavic/native-preview.png`
shows the Mavic above the 16-tree orchard, running at 1.2 simulation seconds,
15,205/15,440 attached leaves and no error. The displayed 112.4 ms air-frame cost
is a single frame, not a realtime performance acceptance claim. Log:
`/tmp/molla-mavic-native.log`. No unrelated window/session settings were changed.

### Next Steps
No required visual-replacement work remains. No full 78-second re-recording was
requested for this change. Actual Mavic mass/inertia identification remains out
of scope. Preserve the existing flight/air/leaf configuration on future model edits.

### Relevant Files
`drone_asset.rs`, `draw.rs`, `studio.rs`, `studio.wgsl`, `data/drone/`,
`ACKNOWLEDGEMENT.md`, `Makefile`.

## Reusable drone motion upgrade (2026-11-18; flight and videos verified)

### Goal
Improve aircraft geometry and physical acceleration, lateral translation,
opposite-tilt braking, stabilization and tilted rotor-wash coupling.

### Instructions
Flight physics is the user's highest priority and will be reused. Research and
acknowledge primary sources; preserve the approved leaf dynamics and hidden vapor.

### Discoveries
The existing rigid-body/rotor simulation is real, but orchard targets jump
between row endpoints and lack trajectory feed-forward. The disturbance observer
subtracts requested mixer thrust rather than the previous measured rotor impulse;
motor slew can therefore be mistaken for wind. Renderer rotor phase is time times
current speed, which is discontinuous during speed changes.
The new 180-degree-heading regression exposed a pre-existing actuator allocation
hazard: independent motor clipping under excessive yaw demand destroys collective
and attitude authority. Fixed with bounded collective / roll-pitch / yaw priority.
The original test crashed hundreds of meters; the corrected case finishes within
0.3 mm of station, with approximately 0.0005 m/s velocity. Quaternion shortest-arc
error also removes the old exact-180-degree sine-error dead zone.
The orchard's 90-degree turns exposed a second issue: a yaw-rate cap alone does
not start braking soon enough for the available shaft torque. Added a stopping-
distance yaw-rate bound derived from rotor capacity and yaw inertia, plus
thrust-axis-first attitude tracking. The full 78 s CPU route now passes its
unchanged 0.1 m final position / 0.1 m/s speed gates; no gate was weakened.

### Accomplished
Reviewed PX4 controller and jerk-limited trajectory documentation and Lee, Leok,
McClamroch's SE(3) control reference. Chose an original reusable bounded-jerk
rest-to-rest trajectory with position/velocity/acceleration tracking, actual-load
disturbance estimation, and physically integrated propeller phase. Not PX4 SITL,
not manufacturer-calibrated dynamics. Engram tools are unavailable.
CPU acceptance: 14 device tests, eight rotorcraft tests, 32 orchard tests with
one explicit Vulkan renderer-upload ignore. Vulkan acceptance: one coupled
flying-drone/air test and six air tests, including CPU/GPU moving tilted-rotor
parity. Log `/tmp/molla-flight-acceptance.log`, exit marker zero. The new
0.25 m heading-change gate excludes initial motor spool-up, which is separately
covered by the existing ramp/hover test; it includes the entire reversal.
The original wrench-conservation fixture now samples during acceleration
(120 instead of 360 ticks), since the faster controller is near level at the
old sample time. Its force and moment tolerances are unchanged.
The revised mesh has a removable battery, stereo front optics, gimbal bracket,
motor fins, wiring, clamps and opposite-handed propellers. Small ellipsoids use
10x24 rather than 12x32 tessellation to retain the existing <25k triangle budget.
Full updated orchard movie is running under wrapper PID 3165692, log
`/tmp/molla-physical-drone.log`, marker `/tmp/molla-physical-drone.exit`, output
`artifacts/drone-orchard/physical-drone-full-1080p.mp4`. Preserve the previous
approved natural-leaves movie. Final isolated 30 s movie is
`artifacts/drone-flight/flight-control-final-1080p.mp4` with source telemetry.
Its full decode/frame checks and acceleration/braking/air-reaction gates pass:
peak tracking error 0.15246 m, peak tilt 16.3578 degrees, final position error
0.005614 m and speed 0.005683 m/s. No fluid grid is evolved in this isolated lab.
Final flight-lab MP4 SHA256:
`da83b253f4cdce49eb7ff795ef297b873f7a1e094c061d41d438622a9e019b18`.
Its decoded 3–7 s contact sheet and 6.5 s braking frame were inspected.
The full orchard export is complete: 78 s, 2340 frames, 1920x1080 at 30 fps.
MP4 SHA256 `9a9ba98d4194a5cc313d28aca4bb68fe5f84f6f41c1fabc77a770720f8da387e`.
All 16 overflight gaps are below 4.64 mm; final speed is 0.01565 m/s, final
position (-1.44543, 3.06237, -8.39996) m. Peak sampled tilt is 9.76 degrees.
Minimum rotor/leaf gap is 0.22365 m, roots do not drift, no wood fails, resolved
divergence RMS 1.889e-4 /s and floor flux zero. 4,482 unique leaves relaunch;
45,620 relaunch events. The state dump is finite and full-flight motion checks
pass. `verify-orchard-video` fully decodes/checks all frames; the decoded 30–34 s
sheet and actual 10 s / 40 s frames were inspected. Capture wall time 562.62 s,
not real-time acceptance. Log `/tmp/molla-physical-drone-verify.log`, exit zero.
IMPORTANT: the original full-row gate still FAILS, now because 10,674 of 15,440
leaves remain (69.13%, below 75%). There are 4,766 shed versus 4,109 in the
previous approved movie. Leaf strength/area/camber/wind and wood parameters were
not changed; flight trajectory/orientation changed the coupled forcing and visits.
The capture exits 2 after finalizing video/report. Do not claim that gate passes
or that the previous exact shedding reduction was retained. Both videos remain.
Public `molla --all-targets` check, explicit formatting and diff checks pass.
Only pre-existing `soil_newton_gpu.rs` unused-variable warnings remain.
Binary SHA256 `e00f7d3428027a4900d3d2007c61c5f9df8cadaef21417d1079263301033f6db`.
Native application relaunched and visually verified: PID 3233099, workspace 5,
log `/tmp/molla-physical-drone-native.log`, screenshot
`artifacts/drone-orchard/physical-drone-native.png`. At 0.9 sim seconds it shows
15,219 attached leaves, strength 0.67, Fly row and Air coupling enabled, vapor
disabled, and no error. Reported 107.2 ms/air-frame is not real-time acceptance.
No commit or push was made. The only process replaced was the previous verified
agent-owned orchard window; user workspace/session settings were not changed.

### Next Steps
Flight work, both recordings and the native-window check are complete.
Any exact leaf-retention retuning, hardware parameter identification, rotor inflow
or gyroscopic models, collision-hull detail and real-time qualification are
separate follow-ups; do not present this as a calibrated aircraft digital twin.

### Relevant Files
`devices/copter.rs`, `devices/propellers.rs`, `airflow/rotorcraft.rs`,
`airflow/trajectory.rs`, `drone_asset.rs`, `orchard_sim.rs`, `ACKNOWLEDGEMENT.md`.
`drone_flight_lab.rs`, `drone_orchard.rs`, `verify_drone_flight.jq`, `Makefile`,
`bin/molla_cables/DRONE_FLIGHT.md` — reusable API, executable evidence and sources.

## Mixed gusts, curved leaves, full movie (2026-11-18; full video delivered)

### Goal
The user rejects the remaining monotonous leaf motion and explicitly requests
the complete 78-second video, not another short preview.

### Instructions
Keep the stronger whole-tree sway, long rows, hidden vapor and approved camera.
Improve physical leaf flight and near-ground motion, not just the location of
the old pileup. Preserve attribution; use Make and patch tools.

### Discoveries
The old detached model becomes a perfectly planar plate with four coplanar
ground contacts. Its previous gusts have little variation near the floor;
flat blades then slide or settle without a realistic rocking mechanism.
Moving a Gaussian exterior transition alone does not solve that limitation.

### Accomplished
Added six evolving curl-potential modes spanning several spatial scales,
zero floor-normal velocity and a smooth near-ground envelope. This is an
explicit synthetic subgrid wind model added to air queries, not resolved LES.
Added variable curved leaf bodies, an offset aerodynamic center, and contact
points matching the rendered detached surface. Planar zero-camber bodies
retain the original APW equations/contact path. Orchard leaf aerodynamic area
and camber vary per leaf; spring loads/thresholds remain separate. Engram tools
are unavailable; decisions and user constraints are recorded here.
The offset drag uses the air-relative velocity at its pressure center, so its
combined force/torque dissipates energy rather than adding spurious spin.
The new still-air energy regression passes. Zero-camber APW reference and
CPU/GPU parity tests remain passing. Current results: 15 CPU air tests, six
Vulkan air tests, 17 CPU attachment tests, seven Vulkan attachment tests,
32 orchard unit tests (one explicit renderer-upload ignore). The curved render
test checks actual triangles against the solver's contact surface.
The 16-second GPU leaf laboratory captures nine heterogeneous curved leaves.
All nine have upward-moving samples after t=5 s; their later peak heights
range from 0.070 to 1.249 m. Full MP4 decode and dimensions/frame checks pass;
the 8–12 s contact sheet was inspected. The independent CPU ground-start test
shows 26.48 radians accumulated normal rotation, 38.03 rad/s peak spin,
32 rising samples and 8.17 cm minimum-surface clearance above ground.
The actual full export is now running, with no native demo competing for GPU:
PID 2786128, wrapper 2785718, log `/tmp/molla-natural-full.log`, exit marker
`/tmp/molla-natural-full.exit`. Output is
`artifacts/drone-orchard/natural-leaves-full-1080p.mp4` (78 s / 2340 frames).
Do not deliver until it finishes and full decode/frame checks and scene-frame
inspection complete. The original full-row retention gate remains enabled;
if it fails, record that separately from media completion, without weakening it.
That first recording was stopped during calibration: 719 shed by 9 s versus
557 with the prior approved field. To preserve reduced shedding under the new
gusts, the default threshold is increased to 0.67 before the full delivery run.
The corresponding unit fixture uses a 0.030 m anchor displacement so its test
load still lies between the new threshold and the original full-strength one.
The interrupted partial MP4 is not a deliverable; final capture uses the same
destination with a new PID-specific temporary file and `molla-natural-final`
logs/exit marker. No motion model or validation threshold was relaxed.
Final run PID is 2810669. Its release binary SHA256 is
`c7eddea05df50cdbbfee23de6b5bbbdf3fde8a1ac5fa4faa56a319f25ac28807`.
The first 12 seconds shed 632 leaves, versus 739 with the prior field at 0.57
and 915 in the earlier level-60/0.5 baseline. This opening comparison is not
a full-flight shedding percentage. Final keyframes at 2, 6, 10, 12 and 20 s
exist; the 12 s frame was inspected and shows varied blade orientations and
airborne/ground scatter. Do not confuse these with the stopped-run keyframes.
Both shader-parser tests pass; the older apple-tree and leafy-tree unit suites
pass 16 and 29 tests respectively (one and three explicit ignores).
Formatting check and `git diff --check` pass. The new motion verification target
checks the full 78 s route and actual relaunch/rolling counters separately from
the unchanged original retention gate, which it reports explicitly.
The final capture completed all 2,340 frames / 78 seconds at 1920x1080/30 fps.
MP4 SHA256: `54d30653b40c41a1b282e9a1c65f0874667e73e0ae6eea6dad1687927dd713d4`.
Full media decode, frame/dimension checks and `verify-orchard-leaf-motion`
pass. The 12 s and 40 s actual scene frames, final overview and decoded
38–42 s contact sheet were inspected. No vapor/overlay was introduced.
Actual full capture: 4,109 shed, 11,331 retained, 25.00% fewer shed than the
previous strong full-flight baseline (5,479). 3,873 distinct leaves lift again
after ground contact; 38,715 relaunch events, 434,879 rising steps and
3,884,637 near-ground rolling steps. Maximum travel is 154.07 m, peak spin
132.63 rad/s. Natural settling and wind-driven clustering are still possible;
there is no promise of perpetual flight or absence of every accumulation.
No root drift or wood failures. Trunk/main-limb maxima: 20.78/55.39 cm.
All 16 overflight gaps are below 5.67 cm; final position is approximately
(-1.44549,3.05969,-8.40012) m, final speed 0.06741 m/s. Resolved-grid divergence
RMS 3.882e-4 /s and zero floor flux. All dumped leaf/wood positions and
velocities are finite. Capture wall time: 933.82 s, not real-time acceptance.
IMPORTANT: original full-row verification remains false because only 73.39%
of leaves remain attached versus its unchanged 75% requirement. The capture
process exits 2 after successfully finalizing the video/report. Separate full
media and motion checks exit 0; do not claim the original full-row gate passes.
Report: `natural-leaves-full-1080p.json`; state: `natural-leaves-full-state.json`;
verification log: `/tmp/molla-natural-final-verify.log`. Native PID 2890574 is
running in workspace 5. Inspected `natural-leaves-native.png` at 9.1 s: strength
0.67, vapor unchecked, 14,929 attached, no error. One frame is 91.9 ms while
capture was still active, not performance acceptance. No commit/push performed.

### Next Steps
Requested motion changes and full video are complete. The old 75% retention
target is still unmet; the current video deliberately preserves stronger
flow and 73.39% foliage. Keep synthetic subgrid/exterior and reduced-leaf model
limitations explicit. Earlier motors-off contact and real-time performance
qualification remain separate follow-ups, not blockers to this video delivery.

### Relevant Files
`airflow/turbulence.rs`, `airflow.rs`, `airflow/cpu.rs`, `airflow/gpu.rs`,
`airflow.wgsl`, `attachments/leaf.rs`, `attachments/gpu.rs`,
`attachments.wgsl`, `orchard_sim.rs`, `leafy_draw.rs`, `leaf_flight_lab.rs`.
`verify_leaf_motion.jq`, `Makefile`, `DRONE_ORCHARD.md`, `ACKNOWLEDGEMENT.md`;
`artifacts/drone-orchard/natural-leaves-full-1080p.mp4` — final complete movie.

## Exterior leaf transport and reduced shedding (2026-11-18; verified preview)

### Goal
Keep the approved level-60 flow, flexible trees and no vapor. Remove the
apparent leaf border and target about 20% fewer shed leaves in a matched clip.

### Instructions
Preserve the approved camera, tree sway, stronger airflow and hidden vapor.
Leaves may settle through physics, but must not freeze at a simulation boundary.
Use Make and patch tools; preserve existing recordings and source attribution.

### Discoveries
Detached leaves have no lifetime, freeze or horizontal position clamp. Their
air query formerly replaced the resolved wake by ambient wind within one
0.15 m cell beyond the grid, producing an abrupt velocity transition. Older
tracks show leaves on the upwind side collecting around x=-4.9 and z=-12.1 m,
while downwind leaves continue beyond 50 m during the full flight.

### Accomplished
Added optional CPU/WGSL exterior query continuation: the boundary's excess
velocity decays as exp(-(distance/length)^2), superposed on the local ambient
gust field. Orchard length is 3 m. This is an approximate far-field sampling
extension, not a pressure-projected exterior CFD solution. The in-grid air
solver, ground friction and free-leaf integration are unchanged. Default
attachment strength 0.55 gives 782 shed versus the baseline 915 (14.5% fewer).
The final default is 0.57: 739 shed versus 915, a measured 19.23% reduction
in the same 12-second sequence. This is not a claim about a proportional
strength/count relationship or the untested full 78-second flight.
Baseline 12-second rerun exactly reproduces 915 shed leaves; 376 are in the
x=-5.1..-4.7 m border band and 99 in z=-12.3..-11.9 m. 333 have speeds below
0.08 m/s. Regression testing caught the post-step GPU gust-uniform upload
resetting the new decay length; all three upload sites now preserve it.
The report's large JSON macro reached its recursion limit, so the extra
diagnostic is inserted separately without raising the crate recursion limit.
The 0.55 transport probe has 17 leaves near the old x boundary and 12 near
the old z boundary versus 376 and 99, with 117 below 0.08 m/s versus 333.
Trunk/main-limb sway remains 10.4/21.8 cm and there are no wood failures or
root drift. Minimum leaf x/z reaches -7.58/-15.57 m. CPU regressions include
edge continuity, far-field decay, unchanged in-grid evolution, a leaf crossing
the grid edge, and eventual frictional settling when wind stops. All 14 CPU
air tests, the shader test, five Vulkan air tests and 31 orchard unit tests
pass (one renderer GPU-upload test explicitly ignored in the orchard lane).
Engram tools are unavailable; implementation decisions are recorded here.
Final capture `artifacts/drone-orchard/free-leaves-sway-preview.mp4` completed:
12 seconds, 360 frames, 1920x1080 at 30 fps. Full decode, frame count and
the new `make verify-orchard-leaf-transport` comparison all pass. Final still
and decoded 6–10 s contact sheet were inspected: no vapor, unchanged camera,
and leaf scatter beyond the formerly sharp grid-edge line. The captured state
has 21 leaves in the old x band and seven in the old z band; those normalized
fractions are over 90% lower than baseline. 120 leaves move below 0.08 m/s;
natural ground settling remains. Minimum x/z: -7.697/-15.591 m, maximum travel
19.502 m. No root drift or wood failure; trunk/main-limb peaks 10.426/21.773 cm.
MP4 SHA256: `520cc33f03a0b520e689ee62d334e0e397e90b587143103e3bb6e2dfaa57d4c0`.
Capture wall time 156.37 s on the shared workstation, not real-time acceptance.
Verification log: `/tmp/molla-free-leaves-verify.log`; capture/tests:
`/tmp/molla-free-leaves-video.log`, `/tmp/molla-border-tests2.log`.
Native PID 2654587 is running on workspace 5. Screenshot
`free-leaves-sway-native.png` shows strength 0.57, vapor unchecked, 14,951 leaves
attached at 6.4 s, and no error. The inspected frame was 111.5 ms during a
concurrent capture; no performance improvement is claimed. Only the previous
agent-owned orchard window was replaced. No commit or push was performed.

### Next Steps
Requested changes are implemented, compared and shown. The complete 78-second
flight/retention gate has not been requalified at strength 0.57; the old strong
setting failed its retention criterion and that gate remains unchanged.
Exterior continuation remains approximate, not a pressure-resolved CFD domain.
Earlier motors-off contact and real-time performance work remain separate.

### Relevant Files
`airflow.rs`, `airflow/cpu.rs`, `airflow/gpu.rs`, `airflow.wgsl`,
`tests/airflow/exterior.rs`, `orchard_sim.rs`, `drone_orchard.rs`,
`verify_leaf_transport.jq`, `Makefile`, `ACKNOWLEDGEMENT.md`, `DRONE_ORCHARD.md`.

## Stronger airflow, whole-plant sway, no vapor (2026-11-18; implemented and previewed)

### Goal
Increase rotor airflow, hide its visualization, and make the whole plant sway
more. The user explicitly confirmed plant/tree, not aircraft motion.

### Instructions
Keep the corrected over-row path and rooted trees. Use physical loads and
elasticity rather than animated offsets. Preserve leaf strength and dynamics.
Use Make and patch tools. Engram memory tools remain unavailable.

### Discoveries
The previous 5 cm peak displacement includes flexible leafy shoots and does
not prove structural tree sway. New diagnostics separately measure trunk and
main-limb displacement from the settled state. Structural bending compliance
can be changed independently of twig, axial, torsional and strength settings.

### Accomplished
Current settings: downwash 60 (5.6 kg, roughly 2.15 times the previous hover
force), fourfold structural bending compliance, unchanged leaf strength 0.5.
Vapor is off by default in both the native demo and recordings. CLI overrides
`--downwash` and `--wood-flex` permit same-build comparisons.
31 unit tests pass, including actual integrated flight at the new mass and
checks that flexibility changes only structural X/Y rotation compliance.
The additional CLI test requires explicit opt-in for airflow visualization.
The 8 s GPU comparison against old 10/1 settings is complete: trunk movement
0.10426 m versus 0.02378 m; main limbs 0.21773 m versus 0.04495 m. These are
physical cell-center displacements from each configuration's settled pose,
not rendered offsets. New settings shed 669 leaves versus 26, with zero root
drift or wood failures in both runs. All leaves survive static initialization;
the softer tree needs 14.917 s to settle versus 5.450 s previously.
Evidence: `sway-60-flex4-probe.json` and `sway-10-flex1-baseline.json`.
The no-vapor 12 s preview is complete and inspected. The full 78 s flight run
completed, but its old gentle-shedding acceptance gate fails. Log:
`/tmp/molla-whole-plant-flight.log`; exit status 2 is recorded in
`/tmp/molla-whole-plant-flight.exit`.
`whole-plant-sway-preview.mp4` has 360 frames at 1920x1080/30 fps, 12 seconds;
media dimensions, frame count and complete decode pass. SHA256:
`2cffde92c4d0e6daabc4916acf7ffe420d5414d459865d66ab34bb005ac37c73`.
Exact capture reports `airflow_visible=false`, downwash 60, flexibility 4,
915 shed and 14,525 attached leaves, no wood failure or root drift. The 6–10 s
decoded contact sheet and final frame were inspected; wind graphics are absent.
Capture wall time was 174.22 s on the shared workstation, not real-time proof.
Full-flight report: 5,479 shed and 9,961 attached leaves (64.5% retained, below
the original 75% requirement), hence `verified_full_row=false`. That gate has
not been weakened and this run is not claimed as a full-row verification pass.
The reported motion/flight metrics show no root drift or wood failures, peak
trunk motion 0.16759 m, main-limb motion 0.40049 m, all 16 horizontal overflight
gaps below 0.05666 m, final return near (-1.44629,3.06254,-8.39993) m with
0.0672 m/s speed. Final divergence RMS is 2.560e-4 /s, floor air flux zero,
and minimum rotor/leaf clearance 0.34738 m. Run wall time: 472.81 s.
Native PID 2486744 is running from the current release binary on Wayland
workspace 5. Inspected `whole-plant-sway-native.png`: Air vapor unchecked,
strength 0.50, 14,776 / 15,440 attached leaves at 7.6 s, no displayed error.
The log is `/tmp/molla-whole-plant-native.log`; one frame reports 49.7 ms,
not a real-time acceptance. Only the earlier agent-owned preview was replaced.
No full stronger-setting video, commit or push was requested. Existing videos
and research attributions are preserved. `git diff --check` passes.

### Next Steps
The requested stronger flow, hidden vapor and whole-plant motion are implemented
and shown. Review the heavier leaf loss before recording a longer strong-flow
movie; the original gentle-shedding verification target remains unmet. Do not
represent that target as passing or silently relax it. Prior motors-off floor
contact and real-time performance work remain separate follow-ups.

### Relevant Files
`orchard_model.rs`, `orchard_sim.rs`, `drone_orchard.rs`, and orchard documentation.

## Over-row flight and subtle vapor (2026-11-18; recorded and verified)

### Goal
Fly directly over both row centers, not the aisle, and replace the distracting
bright airflow strokes with much fainter shader-rendered vapor.

### Instructions
Preserve level 10, half-strength attachments, long rows and leaf dynamics.
Use Make and patch tools. Engram tools remain unavailable.

### Discoveries
- Old waypoints x=+-0.8 were 0.65 m inboard of tree row centers x=+-1.45.
  Flight now starts over the first tree; wider turns need seven/nine seconds.
  A 78 s route keeps the previous speed limit and both 30 s longitudinal legs.
- The old airflow was an egui line overlay without scene-depth occlusion.
  Replace it with 12 broad, softly faded, noise-modulated alpha ribbons using
  actual advected air paths, a depth-tested WGSL pass and no depth writes.
- Hovering the viewport incorrectly clamped the 22 m overview to 12 m even
  without wheel movement. Zoom now changes only on actual scrolling.

### Accomplished
29 orchard unit tests pass, with one explicit Vulkan-upload ignore. This
includes a complete dynamically integrated 78 s controller flight visiting
every tree within 0.15 m horizontally and returning to a stable starting hover.
The renderer now has a separate depth-tested, non-depth-writing vapor pipeline
with 3% maximum per-layer alpha. The 8 s / 240-frame GPU preview
`over-row-vapor-preview.mp4` completes without validation errors; the final
frame was inspected and shows faint vapor, not bright lines. Aircraft x at
8 seconds is -1.45342 m versus row center -1.45 m. The first two trees have
measured horizontal overflight gaps below 4 mm; 26 leaves shed, no wood fails.
The full 78 s capture with all-tree overflight verification is complete.
The preview passes full decode and 1080p/30 frame-count checks. Its SHA256 is
`31547ecd25424e2c988af0f06fb29b13d77ac31071873b87ef70b1b67b23fb81`.
The four-second causal check also passes: 9 leaves shed versus zero without
coupling, 2.57 cm induced wood response, no wood failures/root motion.
`make check APP_TARGET='--release -p molla_cables --bins'` passes for all
example binaries; existing/unused-code warnings remain non-fatal.
Hoskins' sinless-hash lineage was checked against LYGIA's published source;
`ACKNOWLEDGEMENT.md` [34] and `data/HASH_NOISE_LICENSE.md` retain credit and
the MIT notice. The surrounding vapor composition is display-only, not CFD.
The first full export was terminated with status 143 around frame 1,500 after
roughly ten minutes of encoding, without an application error or final report.
Its `.2189375.partial.mp4` is incomplete and must not be delivered. A fresh
export is running under a persistent session, recording an explicit exit-status
file at `/tmp/molla-over-row-persistent.exit`; the short preview remains valid.
Persistent capture PID 2257119 completed with exit 0; log
`/tmp/molla-over-row-persistent.log`, destination `over-row-vapor-1080p.mp4`.
The accepted video has 2,340 frames at 1920x1080/30 fps, 78 seconds, about 76 MiB.
SHA256: `2f20041673a52dc92cf1ac2557a24540914ea6a63700789a8dd2bc0a2fdb12f8`.
`make verify-orchard-video ORCHARD_SECONDS=78 ORCHARD_PREVIEW_START=43
ORCHARD_VIDEO=artifacts/drone-orchard/over-row-vapor-1080p.mp4` passes full decode
and media checks. The decoded 43–47 s return-row contact sheet was inspected:
soft, faint vapor with no bright center strokes, and visible detached leaves.
Exact-capture JSON has `verified_full_row=true`, all 16 measured horizontal
overflight gaps below 0.05466 m (first row below 0.00514 m), 304 leaves shed,
15,136 retained, no wood failures or root error. Final divergence RMS is
8.876e-5 /s, floor air flux zero, minimum rotor/leaf gap 0.4187 m, maximum
detached horizontal travel 54.58 m. Level 10 and strength 0.5 remain unchanged.
Full capture wall time was 972.86 s; this is not real-time acceptance.
The earlier long-row movie retains its hash. Native PID 2360735 is running on
Wayland workspace 5 from the exact current release executable. The inspected
`over-row-vapor-native.png` shows the drone over the first row, Air vapor enabled,
strength 0.50, 15,435 / 15,440 leaves attached at 1.6 simulation seconds and no
displayed error. One native frame reports 165.7 ms; no real-time claim is made.
Only the earlier agent-launched orchard window was replaced. No commit or push
was requested, and unrelated work was preserved. Final `git diff --check` passes.

### Next steps
No requested work remains for the row-alignment and vapor changes. The prior
motors-off ground-contact limitation and real-time optimization remain separate
follow-ups; no physics tolerance was relaxed. The abandoned partial export is
not a deliverable; use only the verified final MP4 linked above.

### Relevant Files
`orchard_model.rs`, `orchard_sim.rs`, `drone_orchard.rs`, `studio_vapor.rs`,
`studio_vapor.wgsl`, `studio.rs`, `branch_draw.rs`, and the orchard Make targets.

## Long rows, level 10 and easier detachment (2026-11-18; recorded and verified)

### Goal
Replace the 2-by-3 layout with two rows of eight trees, reduce downwash from
35 to 10 on the established scale, halve attachment thresholds and make the
translucent airflow more visible. Record the updated flight and reopen the demo.

### Instructions
Use Make and patch tools, preserve older movies and examples, keep the actual
falling-leaf solver and ambient wind. No particle-style airflow or scripted shedding.

### Discoveries and decisions
- Level 10 maps to 2.6 kg; threshold scale 0.5 changes release strength only.
- The 16.8 m rows require a longer air domain, 76 s flight, larger contact-pair
  capacity, overview/follow camera, and shadow coverage beyond the old six-tree box.
- Flow visualization uses 48 passive paths, 40 samples each, with a soft halo
  and roughly 16% core opacity. It is enabled by default and can be hidden.
- Engram tools remain unavailable; session state is recorded here instead.
- The expanded shadow projection needs a uniform-only bind group in the
  depth pass; the lighting group cannot sample the shadow attachment during
  that pass. The first preview caught the missing binding before delivery.

### Accomplished
27 orchard unit tests pass, one GPU upload test is explicitly ignored by that
target. All 15,440 leaves survive normal static settling with threshold 0.5.
The uniform-only shadow binding fix is verified by the completed 6 s / 180-frame
`long-rows-preview.mp4`, full decode/frame-size check and inspected images.
The preview sheds 33 leaves, has zero wood failures/root error and confirms
level 10, mass 2.6 kg, strength 0.5 and visible curved airflow.
Its 83.25 s recording wall time is not real-time performance acceptance.
The complete 76 s recording passes full-flight verification. The native demo
has been relaunched and inspected below.
The four-second coupled/uncoupled check passes (`long-rows-causal.json`):
33 leaves shed with coupling, none without; maximum induced wood displacement
2.57 cm; no wood failures or root drift. The final unit-test command exits 0.
An additional `--motors-off` four-second probe is not accepted: the unpowered
aircraft falls onto the floor and the rigid-contact solver rejects a friction
KKT residual of 1.0707283e-7 against 1e-7. No report was produced. The regular
flight and coupled/uncoupled check do not encounter this contact; no solver
tolerance has been loosened to conceal it. Ambient-only four-second acceptance
for the new mass is therefore not claimed.

- `long-rows-downwash10-1080p.mp4`: 1920x1080, 30 fps, 2,280 frames, 76 seconds,
  approximately 84 MiB. SHA256:
  `c6bd3d50e8e01074ac29009ea3d6a998b3124447be76aeb1fe59d2b8eedec14b`.
- Exact capture JSON: `verified_full_row=true`, 16 trees, 1,920 wood cells,
  15,440 leaves, 912 shed and 14,528 attached; both canopy sides visited.
  Furthest aircraft z=8.5706 m, final position (0.00229,3.06306,-8.80204) m.
  Rotor/leaf clearance minimum 0.46877 m; peak shedding wind 5.3733 m/s.
  No wood failures or root drift, zero floor air flux, final divergence RMS
  1.704e-4 /s. Detached horizontal travel reaches 50.25 m, 820 outside the grid.
- Capture wall time 999.3 s on the shared workstation, including overlapping
  causal/control checks; no real-time performance claim.
- `make verify-orchard-video` passes dimensions, frame count and full decoding.
  Inspected opening preview, 48–52 s follow contact sheet and final overview:
  long rows, curved translucent airflow, retained foliage and detached leaves.
- 27 unit tests pass, the explicit Vulkan upload test passes separately,
  focused Rust formatting and `git diff --check` pass. The older level-35
  and gentle movies retain their prior hashes. No commit or push was requested.
- Native PID 1967068, exact current release binary, confirmed on Wayland
  workspace 5. `long-rows-native.png` shows both complete rows, visible airflow,
  strength 0.50, 15,438 / 15,440 leaves attached at 1.1 simulation seconds and
  no displayed error. One frame reports 92.1 ms; this is not real-time acceptance.
  Only the previous agent-launched orchard window was replaced.

### Next steps
Requested settings, long rows, airflow visualization, recording and native
preview are complete. The optional motors-off floor-contact issue remains a
separate follow-up; no acceptance tolerance was relaxed. Further performance
work is also separate; the larger orchard is not claimed to run in real time.

### Relevant files
`bin/molla_cables/src/{orchard_model,orchard_sim,drone_orchard,studio}.rs`,
`bin/molla_cables/src/studio.wgsl`, `Makefile`, and the orchard documentation.

## Downwash level 35 (2026-11-18; verified and recorded)

User defines the current gentle setup as 0 and the original strong setup as
100, requesting 35. Interpolate the corresponding physical hover loads:
mass = 2 + 0.35 * (8 - 2) = 4.1 kg, about 40.2 N total hover thrust. This is
35% of the interval, not 35% of the original force or a linear airspeed claim.
Retain the original controller, rotors, ambient wind, leaf strength and solver.
Record a new 32 s, 1080p/30 video with full-flight and media verification;
preserve both existing versions. Engram tools remain unavailable.

### Level-35 session summary

#### Goal
Set airflow to 35 on the user's scale and deliver the updated recording.

#### Instructions
0 means the recent gentle setup; 100 means the original strong setup. Keep
leaf physics/strength and ambient wind unchanged. Use Make and patch tools.

#### Discoveries
Interpolating physical hover load yields a 4.1 kg drone and preserves equal
and opposite air reactions. Shedding is nonlinear: the 32 s capture sheds 237
leaves versus 2 at the gentle endpoint and 1,278 at the original strong endpoint.

#### Accomplished
- `DOWNWASH_LEVEL=35`, derived mass 4.1 kg, metadata and three-configuration
  hover regression. All 26 orchard unit tests pass (1 explicit GPU-upload ignore).
- `artifacts/drone-orchard/downwash-35-1080p.mp4`: 1920x1080, 30 fps, 960 frames,
  32 s, approximately 7 MiB. SHA256
  `845a6d51f9f758e9448edb92b486d6032e59c75b3c059cc1ad1d1461afc930ac`.
- Exact capture JSON has `verified_full_row=true`, level 35, mass 4.1 kg,
  237 shed, 5,553 retained, both row sides visited, 22.531 m detached travel,
  zero wood failures/root motion. Capture wall time 192.419 s, not real time.
- Full media decode and dimension/rate/frame-count checks pass; rendered
  20–24 s contact sheet inspected. No UI frame, scripted shedding or airflow
  visualization. Both older movies retain their previously recorded hashes.
- Native preview confirmed running on workspace 5, PID 1735070, after successful
  settling with all 5,790 leaves attached. Only the previous agent-owned orchard
  preview was replaced. Formatting and whitespace checks pass. No commit/push.

#### Next Steps
None required for this adjustment and recording.

#### Relevant Files
- `bin/molla_cables/src/orchard_sim.rs`: level/mass mapping and hover test.
- `bin/molla_cables/src/drone_orchard.rs`: reported level metadata.
- `artifacts/drone-orchard/downwash-35-1080p.*`: movie and validation artifacts.
- `bin/molla_cables/DRONE_ORCHARD.md`: documented scale and current movie.

## Gentler-airflow video (2026-11-18; complete)

User explicitly requested recording the updated gentle setup. Capture the
unchanged 2 kg orchard at 1920x1080/30 fps for a complete 32 s route, with
normal wind enabled, airflow visualization off, and no UI/text frame.
Output: `artifacts/drone-orchard/gentle-downwash-1080p.mp4` with matching
JSON/PNG/encoder log. Run full-flight checks on the exact capture, then verify
frame count, dimensions, rate, complete decode and rendered frames. Previous
videos are preserved. No solver or scene tuning is part of this recording.

### Recording session summary

#### Goal
Record and deliver the gentler-airflow orchard demonstration.

#### Instructions
Use the existing 2 kg configuration and Make capture workflow; preserve older
videos and unrelated work. No solver changes, new graphics or commit/push.

#### Discoveries
The exact capture matches the verified gentle run: 2 leaves shed, 5,788 retained,
both row sides visited and 8.085 m maximum detached travel. Reduced shedding is
the requested gentler behavior; no extra releases were scripted for the video.

#### Accomplished
- Complete `gentle-downwash-1080p.mp4`: 1920x1080, 30 fps, 960 frames, 32 s,
  approximately 5.7 MiB; capture took 163.469 s wall time.
- Matching JSON reports `verified_full_row=true`, mass 2 kg, ambient wind on,
  zero wood failures/root motion and 6.312e-5/s divergence RMS.
- `make verify-orchard-video` passed exact dimensions/rate/frame count and
  full decode. Inspected the rendered 16–20 s contact sheet; no UI frame or
  airflow visualization is included. Existing native preview left running.
- SHA256: `587999a57acef4513ff72b21d533848c72581d0af761838969dde084fb280da3`.
  The prior heavy-drone movie's SHA256 remains
  `161470f518e4847d00fbed25844e3ead1652911527c50e889a78912ec5ab8ff8`.
- Engram tools are unavailable; this session summary preserves the handoff.

#### Next Steps
None required for the requested recording. No commit or push was made.

#### Relevant Files
- `artifacts/drone-orchard/gentle-downwash-1080p.mp4`: delivered movie.
- Matching `.json`, `.png`, `.contact.png`, `.ffmpeg.log`: evidence and preview.
- `bin/molla_cables/DRONE_ORCHARD.md`: link to the current gentle video.

## Gentler orchard downwash (2026-11-18)

User requested much weaker drone airflow, particularly the downward leaf blast.
The orchard now uses a 2 kg configuration instead of the shared 8 kg default,
with unchanged rotor size, route, leaf physics, strengths and ambient breeze.
This reduces required hover thrust/reaction to one quarter without decoupling
flight forces from air forces or artificially clamping leaf velocity. The shared
quadrotor default and other examples remain unchanged. Reports record drone mass.
Hover-reaction and full-flight checks passed. Engram is unavailable.
The hover regression passes with one-quarter air reaction and stable altitude.
The gentle opening retains its leaves; the short causal gate therefore checks
induced wood response and a quiet control rather than requiring early shedding.
Full-flight shedding and all finite/root/pressure/clearance gates remain separate.
The large report JSON macro was already at its recursion limit; append the
mass metadata after construction rather than raising the crate recursion limit.
Initial 2 kg flight shed only two leaves with peak shedding airspeed 3.81 m/s.
The old gate required shedding in both rows; replace that proxy with measured
drone visits to both sides, while retaining full-flight shedding somewhere.
A trial increasing position feedback introduced oscillation and was rejected;
the original controller remains unchanged. Comparing the original 2 kg endpoint
with the prior passing 8 kg run showed the alleged altitude bias was a mistaken
inference, not a flight defect. No finite/root/clearance/return tolerance was
loosened. The final 32 s rerun passes (`gentle-flight.json`): both row sides
visited, 2 leaves shed and 5,788 retained, 3.808 m/s peak shedding wind,
8.085 m detached travel, zero wood failures/root motion, 0.501 m minimum
rotor/leaf clearance and 6.312e-5/s divergence RMS. The original 8 kg run shed
1,278 leaves; these are different force regimes, not a paired flow calibration.
All 26 orchard unit tests pass, including actual hover/reaction comparison;
one explicit GPU-upload test remains ignored in the ordinary unit run.

### Gentle-downwash session summary

#### Goal
Make drone downwash much less forceful without changing leaf aerodynamics.

#### Instructions
Use Make and patch tools, preserve prior work and research acknowledgements.

#### Discoveries
Reducing example mass keeps equal/opposite rotor reactions consistent. Row
coverage must be measured from drone motion, not inferred from destructive
leaf loss. Higher controller gains were unnecessary and were not retained.

#### Accomplished
2 kg orchard configuration, unchanged shared 8 kg default, unchanged controller,
rotors, wind and leaf material; quarter-force hover regression, full-flight
verification, explicit report metadata and updated demonstration documentation.

#### Next Steps
No requested work remains. `gentle-causal.json` passes with 0.019790 m induced
wood response and no opening-hover shedding in either run. The updated native
preview is running on workspace 5, PID 1602321; `gentle-native.png` shows all
5,790 leaves attached at 1.3 s and no displayed error. Only the previous
agent-owned orchard window was replaced. Formatting and whitespace checks pass.
No new movie, commit or push was requested or produced. Earlier movies retain
the heavier drone. Interactive performance is still not a real-time acceptance.

#### Relevant Files
- `bin/molla_cables/src/orchard_sim.rs`: lighter configuration and hover test.
- `bin/molla_cables/src/drone_orchard.rs`: mass/row metadata and gentle-flight gates.
- `bin/molla_cables/DRONE_ORCHARD.md`: settings and verification semantics.

## Falling-leaf circulation revision (2026-11-18)

The previous surface-panel implementation passed drift/spin tests but did not
establish sustained flutter. Replace it with APW (2005b) equations 3.1–3.9:
normal added mass, in-plane added inertia, translational/rotational lift,
angle-dependent drag and dissipative torque. The 3D extension is independent
and approximate; its planar limit matches a separately implemented nondimensional
reference. No prescribed sine/noise trajectory or rotational animation is used.
The CPU prototype produces 15 lateral reversals in 8 s at I*=1.1, and about
10 full tumbles at I*=1.4, with actual upward intervals in still air. Tumbling
can drift one way, so reversal counts apply to flutter, not to every regime.
Air reactions use solid momentum changes before contact; energy diagnostics
include unresolved added air. Ground contacts retain physical blade inertia.
Legacy point attachments and tree examples remain opt-out and unchanged.
CPU tests now cover the independent planar reference, sustained regimes,
3D rotation invariance/roll, energy dissipation, exact solid momentum reaction,
calm settling and unlimited exterior travel. All 15 pass. Six actual Vulkan
tests pass, including 8 s GPU flutter/tumble classification, owner-release
frame inheritance, restore, ground contacts and 12 m/s downwash parity.
The GPU records 15 lateral flutter reversals and a 63.263 rad tumbling sweep.
The fixed 480 Hz trial had a 13.2 cm error in the strong-downwash convergence
test; airspeed/spin-adaptive RK4 reduces that to 4.76 micrometres against the
1,920 Hz reference without clamping velocity. This is a numerical test, not
experimental validation. Both shaders validate. Close-up still-air capture
is complete (8 s, 240 frames, 1080p). Full-orchard rendered acceptance passes.
Sources are acknowledged in ACKNOWLEDGEMENT.md [33].

### Circulation revision delivery evidence

- `artifacts/drone-orchard/leaf-flutter-still-air.mp4`: 8 s, 240 frames,
  1920x1080/30; actual GPU trajectories include repeated upward intervals for
  all nine blades, and visibly changing blade orientations without wind.
  SHA256 `b0d0c427f8db70f14dfc899938f5a29d5dcf806bdcbc87f0e98f5ba55de441ca`.
- `leaf-flutter-breeze.mp4`: same format, analytic ambient wind enabled;
  physical ground friction and continuing exterior transport.
  SHA256 `f043a5b11b442238760e8b57d444d3b3eac1f47cc083554c3d2f4c12d21d6324`.
- `leaf-circulation-orchard.mp4`: 32 s, 960 frames, 1920x1080/30.
  Its JSON has `verified_full_row=true`: 1,278 shed, 4,512 retained,
  1,274 detached leaves outside the air grid, 22.943 m maximum horizontal
  travel, 141.538 rad/s peak spin, zero wood failures/root motion and
  2.374e-4/s divergence RMS. Ordinary tree-step subdivision recovered several
  stiff frames without lost edits or terminal errors. Capture took 212.877 s
  wall time under overlapping verification workloads; not a real-time benchmark.
  SHA256 `161470f518e4847d00fbed25844e3ead1652911527c50e889a78912ec5ab8ff8`.
- All three movies passed `make verify-orchard-video`: exact frame count,
  dimensions/rate and full decode. Inspected still-air, windy and orchard
  contact sheets (orchard 22–26 s), plus the detailed leaf still.
- `circulation-causal-final.json`: 15 leaves shed with coupling versus zero
  in the control, 0.017545 m maximum induced wood displacement, verified.
- Final gates: 15 CPU leaf/attachment tests, 6 actual GPU tests, 2 shader
  validators, 25 orchard units (1 explicit GPU test ignored in ordinary run),
  27 legacy leafy units (3 explicit ignores) and actual legacy GPU rest/shake/
  cut/break/remap/floor check. All passed. Formatting and diff whitespace pass.
- Native preview relaunched and visually inspected on workspace 5, PID 1362006.
  `leaf-circulation-native.png` shows the actual 3348x1348 window at 2.8 s,
  5,775 retained leaves and no displayed error. The sampled last air-frame
  time was 128 ms; interactive real-time performance is not established.
  Only the previous agent-owned orchard process was replaced.

### Circulation session summary

#### Goal
Replace plain descending leaf motion with researched flutter, rolling,
tumbling and wind transport in the existing orchard.

#### Instructions
Use Make, patch tools rather than Python, preserve unrelated work and existing
examples, and acknowledge research. Do not commit or push without a new request.

#### Discoveries
- Pure resistive panel forces and nonzero-spin tests did not prove flutter.
- APW's coupled lift and added-fluid dynamics provide sustained motion without
  noise. Body-only kinetic energy can rise while total body/added-air energy
  dissipates; the energy regression must include both.
- Strong downwash needs flow/spin-dependent integration, not velocity clamps.
- The 3D circulation/tensor extension and aerodynamic sampling remain reduced
  approximations, not measured leaf properties or wake-resolved CFD.

#### Accomplished
CPU/WGSL solver replacement, planar independent oracle, stronger physical and
GPU regressions, reusable leaf-only rendering/capture, citations and three
complete verified videos. Existing tree mechanics and legacy attachments kept.
Engram tools are unavailable; session findings are recorded here instead.

#### Next steps
Requested dynamics, research, tests and visual verification are complete.
Interactive performance tuning, calibration, leaf-leaf/leaf-wood contacts and
resolved wake interactions remain future work. No commit or push was made.

#### Relevant files
- `crates/molla-solvers/src/attachments/leaf.rs`: 3D APW dynamics and contacts.
- `crates/molla-kernels/shaders/attachments.wgsl`: GPU counterpart.
- `crates/molla-solvers/tests/attachments/leaf_flight.rs`: physics acceptance.
- `bin/molla_cables/src/leaf_flight_lab.rs`, `leafy_draw.rs`: GPU close-up capture.
- `bin/molla_cables/src/drone_orchard.rs`, `Makefile`: scene and media entrypoints.
- `ACKNOWLEDGEMENT.md`, `bin/molla_cables/DRONE_ORCHARD.md`: sources and limits.

Status: implemented and verified on 2026-11-18. The final verification and
delivery sections below supersede the historical checkpoints.

## Leafier canopy and close-flight revision (2026-11-18)

### Subsequent physical-leaf/wind correction — implemented and verified

User rejected straight falling leaves, the old drone mesh, and apparent
stopping at the airflow boundary; requested background wind and continuing
physical travel. Source inspection confirmed isotropic point-mass leaf drag,
render-only sine flutter, zero exterior wind, and exponential ground damping.

Implemented an opt-in reusable leaf body in `attachments/leaf.rs` and WGSL:
orientation, angular velocity, thin-plate inertia, four cambered surface
quadrature points, orientation-dependent pressure/skin drag, conservative
air impulse/torque reporting, and unilateral/Coulomb ground impulses. Leaf
geometry renders the integrated orientation. Attached spring proxies and
legacy non-leaf particles retain their previous behavior. No random kicks,
expiry, horizontal position clamp, wrap or leaf recycling were introduced.

Air now supports divergence-free coherent background gusts and a nonzero mean
wind, used both to initialize the near-field grid and at its open boundaries.
Exterior queries keep evaluating the same background at arbitrary positions.
The orchard defaults to (1.2, 0, 0.35) m/s mean wind and 0.65 m/s gust scale;
`--still-air` disables the background for comparisons. Rotor loads, transport
and projection remain dynamically solved; synthetic background gusts are not
a calibrated atmospheric turbulence model. Leaf torque reaction is deposited
as conservative force couples; unsupported exterior reactions are reported.

The rejected imported model is replaced by an original procedural agricultural
airframe, with sculpted fairing, motor housings, cooling rings, twisted blades,
skids, status lights and a gimbal camera. The old model's source/attribution
remain; research and limits are acknowledged in [33].

Initial tests: ten CPU attachment tests and three actual GPU tests passed,
including energy dissipation, angular/linear air impulse, orientation-dependent
descent, >30 m unconstrained transport, ground wind transport and restore/parity.
Full scene, updated video, native launch and additional wind gates remain
pending at this checkpoint. No commit or push requested. Engram tools remain
unavailable, so discoveries and progress are recorded here instead.

Further acceptance checkpoint:

- First complete physical-leaf flight passed: 1,280 shed, 4,510 retained,
  868 outside the near-field grid, 7.72 m maximum sideways travel, no wood
  failures or root motion. Ambient-wind-only/motors-off comparison sheds zero
  in four seconds; the active drone sheds 15 in the same interval.
- A new still-air landing test caught residual single-corner rocking
  (~2 rad/s). Four-corner unilateral contact with accumulated Coulomb impulses
  replaces that approximation. The same leaf settles to 4.7e-6 m/s and
  3.5e-4 rad/s without a velocity reset, artificial brake or sleeping rule.
  Eleven CPU tests and four GPU attachment tests pass, including ground
  impulse parity; a moving-owner/release parity test is also being run.
- The physical detached-leaf display mesh now follows the planar collision
  body, rather than adding the old petiole droop below the ground. Its actual
  vertices pass orientation/ground-clearance checks. Orchard tests: 25 pass,
  one explicit GPU upload test ignored in this unit lane.
- Original leafy-tree GPU rest/shake/cut/break/remap/floor regression passes;
  its unit lane reports 26 passed, three explicit ignored tests at this
  checkpoint. Non-leaf attachment paths remain unchanged.
- The first complete video `physical-leaf-wind-flight.mp4` was inspected at
  22 and 26 seconds, showing broad sideways dispersal rather than columns.
  It predates the final contact/mesh correction. The intermediate
  `physical-leaf-wind-final.mp4` capture was deliberately stopped and is NOT
  a completed delivery. Final verified replacement rendering is
  `leaf-dynamics-showcase.mp4` with `--verify-flight`: it applies the same
  full-flight acceptance function used by `--verify-row`, including the new
  external travel/rotation gates. Do not mark delivery until its completion,
  JSON gate, media decode and native launch have been checked.

#### Final physical-leaf acceptance (2026-11-18)

- `leaf-dynamics-causal.json`: current CPU/GPU integration passes the
  four-second causal gate, 15 shed versus zero, 1.75 cm final induced wood
  response. Motors-off ambient-wind control also sheds zero.
- `leaf-dynamics-showcase.json`: `verified_full_row=true` with the final
  four-corner contact and matching display mesh. 1,277 shed, 4,513 retained,
  784 detached leaves outside the near-field grid, 7.841 m maximum horizontal
  travel. Both rows shed; min rotor/leaf gap 0.495 m; peak shedding wind
  8.321 m/s. Fixed roots, zero wood failures/ground flux, finite state and
  return-flight gates all pass. Divergence RMS 2.381e-4 /s.
- `leaf-dynamics-showcase.mp4`: complete 1920x1080/30 fps, 960 frames,
  32.000 seconds. Full FFmpeg decode passed; final frames at 22 and 26 seconds
  were inspected, along with the close-up drone preview. SHA256:
  `a143126ed9f3f56c2d40b2c870d0f27e9d56a45945de124c418b46e4184e0c11`.
  Recording wall time 199.697 s includes rendering/encoding, not a pure
  physics benchmark; the application remains slower than real time.
- Current focused gates: 11 attachment CPU tests, five attachment GPU tests,
  11 air CPU tests, four air GPU tests, both WGSL validation tests, and
  25 orchard unit tests pass (one explicit studio-upload test ignored in
  this unit lane). Legacy tree GPU interaction verification also passed.
  Focused formatting and `git diff --check` pass. These are not a claim that
  every unrelated dirty workspace package was validated.
- Native window reopened and inspected on workspace 5, PID 1037140 at this
  checkpoint, 3348x1348; `leaf-dynamics-native.png` shows t=14.1 s, new drone,
  wind-driven detached leaves, and no application error. Historical PID only.

#### Session summary — physical leaf correction

**Goal:** replace visual-only flutter/straight point-mass descent, add normal
wind and gusts, remove exterior wind cutoff behavior, and improve the drone.

**Instructions:** source acknowledgements, Make-based gates, patch editing,
preserve previous examples, no unrequested commit/push or unrelated cleanup.

**Discoveries:** rotating surface forces must share the rendered orientation;
zero exterior air and artificial ground damping caused apparent stopping.
A single ground contact rocked indefinitely; a small contact manifold settles
under forces without a reset. Rendering droop also had to match the rigid
detached blade's collision extent. Wind changes wake/canopy asymmetry naturally.

**Accomplished:** reusable CPU/WGSL leaf dynamics, spatial background gusts,
unbounded exterior leaf advection, conservative torque transfer, original
agricultural drone mesh, credits, causal/regression/flight/render checks,
final 1080p recording, and actual native launch.

**Next Steps:** no requested change remains unimplemented. Quantitative
aerodynamic calibration, resolved far-field wake, leaf/wood collision and
real-time performance remain limitations/follow-up work, not established
acceptance. Grounded leaves can stop when friction balances local wind; they
are not artificially kept moving. No publication/commit was requested.

**Relevant Files:** `crates/molla-solvers/src/attachments{.rs,/leaf.rs,/gpu.rs}`,
`crates/molla-solvers/src/airflow{.rs,/cpu.rs,/gpu.rs}`, both WGSL shaders,
attachment/airflow tests, `bin/molla_cables/src/{orchard_sim,drone_orchard,
drone_asset,drone_asset_legacy,leafy_draw}.rs`, example README, acknowledgements,
this plan and the final artifacts listed above. Engram tools remained
unavailable; no persistent-memory folder was edited.

User request: fuller trees, no prominent particle visualization, a sourced
drone model, and a flight close enough to foliage to visibly blow leaves off.
Revision implemented and verified; the historical delivery below describes
the previous 2,520-leaf, center-aisle example.

- Orchard foliage now has 5,790 independently simulated attachments and
  approximately 4.08 times the rendered leaf-area proxy. Wood remains 720
  cells. Existing tree examples still use their original foliage builder.
- Increasing both leaf count and size without redistributing represented
  canopy mass overloaded the light shoots: still-air initialization shed
  72 leaves and retained motion after 16 seconds. The denser sampling now
  distributes each twig's previous mass/drag budget across its extra leaves;
  stiffness, damping and failure strength scale with particle mass. The
  ordinary sparse build is unchanged. This is an effective canopy model,
  not a claim of measured physical leaf area or areal mass density.
- With stable budgets, all 5,790 leaves survive natural settling (5.45 seconds)
  and the uncoupled control. Strength 0.65 shed 2,005 leaves during the closer
  full pass and failed the unchanged 75%-retention requirement. The new
  orchard default is 1.0: 1,157 shed and 4,633 remain after 32 seconds. Other
  leafy examples retain their existing material settings.
- Imported Innerscene's CC0 quadcopter GLB (5,036 source triangles), rather
  than adding more primitive placeholder geometry. Source/hash/CC0 notices
  are in `data/drone/ATTRIBUTION.md` and acknowledgement [32]. Motor spacing,
  rotor radius and animation match the simulated aircraft; this visual model
  does not replace or calibrate the existing flight coefficients.
- Air particles are removed from rendering. Air guides default off. The
  optional display is 32 thin, alpha-16/255 advected paths, not opaque tubes
  or emitted particle sprites. It does not control physics or shedding.
- Flight now visits both canopy edges (x=±0.8 m) and returns to the aisle.
  Rotor/leaf distance and local airflow at shedding are reported. No
  proximity-based impulses, scripted shaking or manual leaf release is used.

Relevant files: `orchard_model.rs`, `orchard_sim.rs`, `drone_orchard.rs`,
`drone_asset.rs`, `leafy_model.rs`, `leafy_draw.rs`, `studio.rs`, `data/drone/`, `Makefile`, and
`ACKNOWLEDGEMENT.md`. No Git commit or push requested for this revision.

### Revision acceptance and delivery

Verified on RTX 4080 / Vulkan / NVIDIA 595.112.07:

- `make test-drone-orchard`: 24 passed; one explicit GPU upload test ignored
  in the normal unit run and executed separately below.
- `make test-leafy-tree`: 25 passed, two explicit ignored tests; the actual
  GPU rest/shake/cut/break/remap/finite-state/floor verification also passed.
- `make verify-drone-orchard`: `leafy-canopy-causal.json` reports
  `verified=true`, two leaves shed with coupling versus zero without,
  2.244 cm maximum induced wood response, and less than 0.5 mm quiet drift.
- `make verify-drone-orchard-row`: `leafy-canopy-full-row.json` reports
  `verified_full_row=true`. Six-tree shedding is [24, 212, 204, 207, 205, 305];
  minimum sampled disk/leaf gap 0.5077 m; peak sampled local shedding wind
  8.5385 m/s. Roots remain fixed, wood failures and ground air flux are zero.
  The default flight returns within the existing position/velocity bounds.
  Mean headless physics cost was 32.97 ms per air frame.
- `make capture-drone-orchard ORCHARD_SECONDS=32` produced
  `artifacts/drone-orchard/leafy-canopy-flight.mp4`: 1920x1080, 30 fps,
  960 frames, 32 seconds. Frames at 8, 16 and 24 seconds were inspected.
  SHA256: `06089faae7861a6b48b3cd998d43ad9c2c606dcd17530e38fcfd37a14550efdf`.
  The optional faint-flow capture was also visually inspected.
- Native example reopened on workspace 5, PID 551036 at this checkpoint;
  both `leafy-canopy-fast-native.png` and `leafy-canopy-fast-native-later.png`
  show the updated mesh, denser trees and airflow-driven detached leaves,
  with no application error. Process IDs are historical, not durable handles.

### Renderer optimization and remaining performance limit

The enlarged canopy exposed native rendering cost. Distant orchard leaves
now use 14 triangles instead of 36; zooming within 4 m selects the detailed
mesh. All 5,790 physics attachments remain. Other examples keep their existing
mesh detail. The shared studio upload now uses bounded 256-byte mapped writes
instead of one bulk copy; staging submission and recall are unchanged.

`make test-studio-upload` passed actual Vulkan readback for seven lengths from
4 to 32,000,012 bytes, including incomplete block tails. A single 32 MB copy
measurement fell from 4.020 to 1.820 ms; this is a copy microbenchmark, not an
end-to-end frame-rate claim. The pre/post upload-only faint-flow PNGs have
identical SHA256 `78a7504f1b5a9211a1518a7e97a2b9aa83f03994ea8a7f8fedf02376805918ad`.
The lower-detail four-second capture was separately inspected.

The native 3348x1348 window remains slower than real time: two screenshots
17.6 wall seconds apart progressed from 11.9 to 13.8 simulation seconds,
with last-step physics readings 25.5 and 77.2 ms. These changes reduce mesh
and copy work but do not establish a satisfactory native frame-rate fix.
The smooth fixed-timestep movie is not evidence of real-time performance.

### Revision session close

**Goal:** fuller foliage, subtle/hidden airflow, a credited drone asset, and
close rotor-driven leaf shedding.

**Instructions:** use Make targets, patch files without Python, preserve
existing examples, acknowledge sources, and do not commit unrequested work.

**Discoveries:** extra visual leaf samples require redistribution of effective
canopy mass/drag to avoid still-air overload; close flight requires stronger
attachments than the old center-aisle pass. Bounded upload writes preserve
pixels, but native performance still needs end-to-end profiling.

**Accomplished:** all requested scene changes, sourced CC0 asset, causal and
full-flight GPU checks, regression checks, inspected 1080p movie, and live
native launch. No core airflow/wood solver was replaced in this revision.

**Next Steps:** native frame-time profiling remains a performance follow-up;
finite-domain ground-leaf accumulation and uncalibrated near-canopy flow
remain documented numerical/modeling limitations, not validated litter CFD.
No additional implementation or Git publication is authorized by this note.

**Relevant Files:** the revision file list above, this plan, and
`bin/molla_cables/DRONE_ORCHARD.md`; artifacts are under
`artifacts/drone-orchard/`. Engram tools were unavailable during this session.

## Requested end state

A separate executable with several apple trees in rows and a dynamically
flown drone between them. Rotor-generated air must load wood and leaves;
elastic branches respond, attachment loads determine shedding, and the air
carries detached leaves. Preserve the previous tree examples. Credit sources.

## Repository audit (2026-11-18)

- Reuse `molla-solvers/src/devices/{copter,propellers,aero}.rs` and
  `RigidWorld`: cascaded flight control, rotor mixing, motor acceleration,
  thrust/reaction torque, and body drag already exist. No rotor wake field
  exists there: wind is supplied globally or per body.
- `molla-fem/src/stam_fluid.rs` and `fft_ns.rs` are two-dimensional periodic
  grids, with host advection and GPU pressure FFT. They cannot directly
  represent a three-dimensional orchard with a ground boundary.
- Taylor-Hood Stokes supports CPU tetrahedra, but `stokes_gpu.rs` currently
  rejects dimension 3. It is not the resident 3D air backend needed here.
- Reuse GPU strands, convex wood contacts, and breakable leaf attachments.
  Extend attachment inputs to spatially varying air instead of prescribing
  shake motions. Batch orchard mechanics where possible; avoid one full
  fluid readback or repeated independent solver setup per tree.

## Implementation decision

Add a reusable `molla-solvers::airflow` module: 3D staggered MAC velocities,
semi-Lagrangian advection, viscous diffusion, and pressure projection.
Ground is impermeable/free-slip; other domain faces have ambient pressure
and ambient inflow. A CPU reference and resident WGSL path use the same
discrete operators. Expose convergence diagnostics rather than claiming
fixed pressure sweeps are exact incompressibility.

Spread each rotor's measured thrust and reaction torque over a finite disk
with conservative interpolation. Apply equal/opposite rotor force to air
and aircraft. Couple local air-relative drag to wood and attached/detached
leaves, with reaction forces returned to the grid. Flight follows waypoints
through motor commands, never by assigning body poses along a spline.

This is a low-speed, incompressible, actuator-disk simulation. It does not
resolve rotating blades, acoustic/compressible flow, blade-vortex interaction,
or a calibrated turbulence/leaf-abscission model. Do not call it validated
agricultural CFD. Grid convergence, projection residuals and elapsed costs
must be reported. No distance-triggered shake or random leaf removal.

## Acceptance gates (all remain required)

- [x] Air CPU: interpolation, conservative force/torque transfer, resting
  air, ground flux, divergence reduction, advection and invalid-input checks.
- [x] Air GPU: shader validation, multi-step CPU parity, resident timestepping,
  local sampling and measured timing on the actual GPU.
- [x] Air GPU repeatability, grid/time convergence and longer coupled-run
  checks with the completed orchard.
- [x] Flight: hover thrust approximately mg; motor lag; climb, lateral and yaw
  response; no prescribed poses; rotor-off free fall; wake sources agree
  with actual rotor wrench, including tilted flight.
- [x] Coupling: local relative wind acts on wood and individual leaves;
  reaction impulse is accounted for; attached leaves survive still air;
  sufficient loads shed some, not all, leaves; detached leaves advect/fall.
- [x] Orchard: separate source/executable/Make targets; several trees in rows;
  visible drone passage, downstream wake, branch response and shedding.
- [x] Causality: compare against the same flight without air/tree coupling,
  separating induced motion from ordinary gravitational settling. Rotor-off
  flight tests produce no rotor forcing; falling hull drag is not expected
  to produce perfectly still air. All checked states remain finite.
- [x] Actual rendered/native example and an inspected capture, not only unit
  tests; record performance and any numerical/physical limitations.
- [x] Documentation, citations, controls, commands and verification evidence.

## Research and attribution

1. Robert Bridson and Matthias Müller-Fischer (2007), *Fluid Simulation*,
   SIGGRAPH course notes. MAC discretization, semi-Lagrangian transport,
   pressure projection and boundary conditions. Independent Rust/WGSL
   implementation, no source copied.
   <https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf>
2. Moses Bangura, Marco Melega, Roberto Naldi and Robert Mahony (2016),
   *Aerodynamics of Rotor Blades for Quadrotors*.
   <https://doi.org/10.48550/arXiv.1601.00733>. Reference for momentum/blade
   modeling and force/torque conventions; not a full blade-element port.
3. Leonard Bauersfeld, Koen Muller, Dominic Ziegler, Filippo Coletti and
   Davide Scaramuzza (2024), *Robotics meets Fluid Dynamics: A Characterization
   of the Induced Airflow below a Quadrotor as a Turbulent Jet*, IEEE RA-L.
   <https://arxiv.org/abs/2403.13321>. Their measured mean far-field model
   applies beyond roughly 2.5 drone diameters; do not substitute it for
   near-canopy transient flow. Reference and potential far-field comparison,
   not a claim of reproducing their experiments.
4. Shenghui Fu et al. (2025), *An Improved Hierarchical Leaf Density Model
   for Spatio-Temporal Distribution Characteristic Analysis of UAV Downwash
   Air-Flow in a Fruit Tree Canopy*, Agronomy 15(8), 1867.
   <https://doi.org/10.3390/agronomy15081867>. Canopy-flow research context,
   not a reproduction of their RANS or measured orchard results.

AppleQSM/Appleskel, tree-fracture inspiration, and existing flight-device
references remain credited in `ACKNOWLEDGEMENT.md`. No new external source
code or assets are imported by the airflow implementation.

## Historical foundation checkpoint

Implemented `molla-solvers::airflow` with a double-precision CPU reference,
resident f32 WGSL air fields, red/black SOR projection, conservative rotor
force/torque deposition, and batched local velocity sampling. The upper and
side boundaries interpolate toward ambient air over one exterior grid cell;
the floor extrapolates tangential velocity with zero normal velocity.

Attachment particles now expose the actual integrated aerodynamic force.
`GpuAttachments::step_spatial` accepts per-particle wind; the previous
uniform-wind API and existing leaf behavior are preserved. Returning this
force to the air and coupling wood drag are still integration work, not
completed orchard behavior.

Verified on RTX 4080 / Vulkan / NVIDIA 595.112.07, release:

- Eight CPU air tests: quiescent/stopped rotors, tilted force and torque
  transfer, affine sampling, projection/ground, wake persistence after
  shutdown, ambient flow, invalid inputs, and transport of a weak velocity
  pulse (0.16 m expected displacement).
- WGSL parser/validator test and two explicit GPU tests passed. Twelve
  moving/tilted-rotor steps agree with CPU to 3.46e-6 m/s in face velocity
  and 6.77e-6 Pa in pressure. Batched sampling agrees within 2e-6 m/s,
  including a query set larger than the internal sampling chunk capacity.
- 64x32x48 cells at 0.15 m, four 12 N rotors, 120 pressure sweeps,
  dt=1/60 s, 30 steps plus 3,200 local samples each: median 1.203 ms,
  maximum 1.695 ms in the staging-buffer benchmark. The final full-gate rerun
  measured median 2.399 ms / maximum 3.829 ms on the same grid; these runs
  were on a shared desktop, not an isolated benchmark host. This is air only,
  not the frame rate of a completed orchard. Divergence RMS 3.60e-4 1/s,
  maximum 1.27e-3 1/s, floor normal velocity exactly zero; maximum face
  velocity component 8.36 m/s. This short transient is not a steady-hover
  validation or a grid-convergence study.
- The original upload path took roughly 132 ms. Profiling isolated about
  128 ms in host submission, predominantly `Queue::write_buffer` uploads.
  Reusing `wgpu::util::StagingBelt` for force and query data removed that
  cost without reducing resolution, pressure sweeps or physical scope.
- Four CPU attachment tests and two GPU tests passed: per-leaf winds,
  momentum/air-force accounting, load-driven shedding, quiet-leaf survival,
  invalid input atomicity and switching back to the uniform-wind API.
- Existing leafy example regression: 25 unit tests passed, two unrelated
  explicit Vulkan/capture tests remained ignored, and the actual GPU
  `--verify` run passed rest, shake, heterogeneous shedding, topology/remap
  continuity, finite-state and floor checks. Its 120-cell/420-leaf timing
  was 11.704 ms per nominal step. Orchard assembly must batch mechanics and
  profile these transfers rather than multiply that cost by the tree count.
- All 14 existing device tests passed, including actual hover, climb,
  lateral flight, yaw, attitude control, rotor-off falling, steady-wind
  station holding, and propeller thrust/reaction torque. This verifies the
  reusable flight baseline, not its yet-to-be-implemented wake coupling.

An initial CPU/GPU discrepancy came from discontinuous ambient sampling at
open faces: rounding could select different sides of the boundary. A
continuous one-cell ambient extension fixed it; parity thresholds were not
weakened. The air algorithm is independently implemented from the numerical
references above; these checks are numerical evidence, not calibration to
physical drone/canopy measurements.

Commands: `make test-airflow test-airflow-gpu test-attachments
test-attachments-gpu`. `make profile-airflow` runs a kernel diagnostic;
its lib-test build requires `resource-counters` because unrelated existing
GPU unit tests import the metric scope types. On this workstation use
`CARGO_HOME=/pkg/cargo NVIDIA_VERSION=595.112.07 nix develop --impure -c`
before the make command.

Final gate log: `/tmp/molla-drone-foundation.log`. Upload profiling evidence:
`/tmp/molla-air-submission.log` and `/tmp/molla-air-staging.log`.
Existing flight gate: `/tmp/molla-drone-existing-flight.log`.

Next: connect existing propeller readings/flight dynamics to disk forcing,
implement bidirectional wood/leaf drag coupling, build the separate multi-tree
orchard executable and controls, run causal/long-run/convergence tests, and
inspect an actual rendered capture. The full goal is not complete.

## Historical integration checkpoint

The reusable `airflow::Quadrotor` now uses the existing `RigidWorld`, copter
controller, propellers and hull drag. Commands drive motor acceleration;
positions are never assigned during flight. Propeller readings accumulate
actual world-space force and moment over integration substeps. The air disk
receives their exact opposite wrench, including tilted/moving rotors.

The existing copter mixer incorrectly rejected small but full-rank SI-unit
rotor coefficients using an absolute determinant cutoff. Row/column scaling
of the Gram matrix now makes the rank test relative. Three new CPU flight
tests and all 14 existing device tests passed. The explicit two-second GPU
flight/air test passed: air 0.6 m below a rotor reached -9.37 m/s vertically,
divergence RMS was 1.27e-4 /s, and ground flux was zero. This is an actuator
disk with static thrust coefficients, not a calibrated blade-element model.
Log: `/tmp/molla-rotorcraft.log`.

Added `molla_drone_orchard` in a separate source file, with six AppleQSM trees
in two rows, 720 wood cells and 2,520 leaf attachments. Local air samples drive
leaf/wood drag; equal and opposite aerodynamic impulses return to the grid.
Leaf spring loads act at the actual petiole on the wood. Loads outside the
interior force-splat support are reported as boundary impulse. A reusable
`CylinderDrag` now contains the cylinder quadrature force law. Flight, leaves
and wood step at 240 Hz; the evolving air field steps at 60 Hz.

The first two-second scene probe stayed finite with fixed roots, no broken
wood and no lost leaves. This is not yet the required visible shedding or
long-run acceptance. It was too slow: approximately 567 ms per air frame,
including 500 ms in the wood step on the shared workstation. Air alone took
7.16 ms in that run. Very short probes also include first-use GPU warm-up and
have highly variable timings; they are not steady-state frame-rate claims.
Log: `/tmp/molla-orchard-two-seconds.log`.

Performance work in progress: pooled attachment/motion/renderer uploads,
motion-only validation for motion edits on immutable validated geometry,
independent elastic islands solved in parallel workgroups, and compact
deterministic contact-occupancy masks before serial graph coloring. Island
parity passed against separate GPU solves, including interleaved indices and
a component larger than the shared-memory fast path. Contact/render/legacy
regressions and end-to-end performance still need verification.

Commands added: `make test-rotorcraft test-rotorcraft-gpu test-drone-orchard`,
`make probe-drone-orchard ORCHARD_SECONDS=12`, `make run-drone-orchard`, and
`make capture-drone-orchard ORCHARD_SECONDS=12`. Do not interpret the presence
of these commands as a completed visual/performance acceptance gate.

## Historical first-flight checkpoint

- `make verify-drone-orchard` passed twice. The final report is
  `artifacts/drone-orchard/causal-verification.json`: four seconds, 3.473 cm
  maximum / 2.395 mm RMS wood-position difference, 11 leaves shed with air
  coupling versus zero without, unchanged root poses/velocities, no wood
  failures, finite particles and zero ground-normal air velocity. The second
  run records each world separately rather than assigning both worlds' wall
  time to each scene. Corresponding air divergence RMS values are 3.04e-4
  and 3.34e-4 /s.
- An eight-second flight shed 26/2,520 leaves. Scale 0.65 is now the orchard
  default; the old leafy example's authored strengths are untouched. The
  four-second control at this same scale shed none. This is an uncalibrated
  demo material parameter, not an empirical apple-petiole strength claim.
- `artifacts/drone-orchard/first-flight.mp4` is an inspected 1080p, 30 fps,
  240-frame, eight-second preview, with accompanying PNG, JSON and ffmpeg log.
  The render shows all six trees, the airborne drone, downward/outward air
  tracers and detached leaves. This is not the previous tree-fracture video.
- Native `Molla · drone orchard` launched on Wayland workspace 5, PID 3954087
  at this checkpoint. `/tmp/molla-drone-native.png` shows it at 9.4 simulation
  seconds, 2,494 attached leaves, no displayed error, and 66.4 ms per air frame.
  The native session is left running; do not kill unrelated user applications.
- End-to-end eight-second probe: 62.9 ms per air frame, versus the initial
  567 ms; other runs with concurrent work measured 79–95 ms. This is still
  slower than wall-clock real time. Wood mechanics and synchronization remain
  the bottleneck. The native loop performs at most one air frame per redraw
  rather than doubling UI latency by processing two catch-up frames.
- Profiling separated a roughly 160 ms host mapped-copy cost from the air
  kernel itself. `molla-compute::copy_to_mapped` uses safe fixed 256-byte
  write-only blocks. Sparse force spans clear the dense GPU force buffer and
  upload only supported spans, with a dense fallback. Final byte-roundtrip
  tests include partial blocks and 2,000,012 bytes (0.064 ms in the isolated
  mapped-copy test). Three GPU air tests passed, including dense → empty →
  sparse → empty parity. The isolated 64x32x48 air-plus-3,200-sample benchmark
  measured median 1.323 ms / maximum 2.257 ms. No temporary tracing remains.
- Nineteen strand GPU regressions passed, including atomic failure, cut
  continuity, impulse/resource contracts and interleaved independent-island
  parity. Four contact GPU tests passed, including high-index sparse contact
  masks and clearing after rebound. A stricter per-island tolerance initially
  regressed an existing tiny cut model. Models of at most 128 cells now retain
  their original single-workgroup/global solve; larger independent models use
  conservative partitioned tolerances. The original failing test now passes
  without weakening its assertions.
- Nine CPU air/drag tests, the air shader validator, four new CPU rotorcraft
  tests and all 14 existing device tests passed. Two attachment GPU tests,
  the actual flight/air GPU test, and the previous leafy example's 25 unit
  tests plus actual GPU `--verify` passed (two unrelated explicit tests remain
  ignored in its normal unit target). New orchard unit tests: 21 passed.

Key files: `bin/molla_cables/{DRONE_ORCHARD.md,src/drone_orchard.rs,
src/orchard_model.rs,src/orchard_sim.rs}`, `crates/molla-solvers/src/airflow/`,
`crates/molla-compute/src/upload.rs`, the strand/attachment GPU implementations
and associated regression tests. Citation [31] in `ACKNOWLEDGEMENT.md` covers
the air/rotor research; existing AppleQSM and fracture credits are retained.

Logs: `/tmp/molla-orchard-final-gates.log`,
`/tmp/molla-drone-strands-regression-final.log`,
`/tmp/molla-drone-coupling-regression-final.log`,
`/tmp/molla-airflow-final-transfer.log`, `/tmp/molla-upload-final.log`,
`/tmp/molla-orchard-flight-capture.log`, `/tmp/molla-drone-orchard-native.log`.

Remaining: spatial/time convergence and longer full-row/return-flight checks;
further whole-scene performance work; optional initial vegetation settling
before presenting the flight (the current initial gravitational sag is about
27 cm, whereas the controlled air-induced difference is about 3.5 cm).
Do not present the initial gravity transient as wholly caused by rotor wash.
Drone/tree collision geometry, resolved moving impermeable wood boundaries,
and leaf/wood or leaf/leaf contact are not implemented. Those limitations are
documented, not silently treated as solved. The active goal is not complete.

## Final numerical and performance verification

This section supersedes the historical checkpoint status above. The requested
drone/orchard implementation, numerical gates, refreshed capture and native
launch checks are complete within the explicitly documented model scope.

### Initialization and mechanical timestep

The orchard now settles vegetation under its normal gravity, damping, leaf
loads and still-air drag before starting the flight. At the 60 Hz default it
takes 5.383 simulation seconds to reach the quiet-speed criterion (0.004665 m/s,
below 0.005 for at least 0.2 seconds), with every leaf attached and no wood
damage. Initial drone position, rotor state and air are untouched. Flight time
and displacement references restart after settling. `--cold-start` exposes
the original gravity transient. Increasing damping during settling was
rejected: in the existing integrator it also attenuated integrated gravity
and left an artificial relaxation when normal damping resumed.

`Tree` now stores its step configuration; old examples retain 240 Hz. The
orchard defaults to 60 Hz mechanics, with `--physics-hz 120`, `240` or `480`
available. The air grid and its 60 Hz update are unchanged. An earlier paired
32-second rate comparison shed the same 41 leaf IDs at 60 and 240 Hz, with
0.857 mm maximum / 0.111 mm RMS final wood-position difference. That comparison
used the earlier extra-damping initializer, not the final natural settling;
the final initialization receives a separate four-second rate comparison.
Initializing directly at 240 Hz did not reach the quiet-speed threshold within
the eight-second limit. All rate choices now use the same natural 60 Hz
initialization, restoring the requested timestep before flight. This preserves
the initializer's physical loading and gives rate comparisons identical
starting states. Post-initialization relaxation is checked separately at
240 Hz; changing the settling threshold or artificially zeroing motion was
not used to bypass the failure.

The final common-initial-state four-second comparison passes: maximum wood
position difference 1.783 mm, RMS 0.225 mm, attached-leaf position RMS 0.383 mm,
zero attachment-state mismatches, and 11 shed at either rate. Peak wood
displacement is 5.897 cm at 60 Hz versus 6.604 cm at 240 Hz; aircraft positions
differ by about 1.5 cm. This qualifies a visual/performance tradeoff, not
identical integration. The 240 Hz causal check also passes (11 versus zero
shed, 1.935 cm maximum air-induced wood difference); its control relaxes by
at most 1.332 mm. Evidence: `artifacts/drone-orchard/settled-causal-240.json`,
`mechanical-rate-comparison.json`, and `/tmp/molla-orchard-final-four-*.log`.

`GpuStrands::impulses_snapshot_cached` avoids a second readback only when the
hint has exactly the globally unique revision of the latest successful GPU
publication and no step/upload has invalidated it. Stale hints synchronize;
host states returned by uploads are not accepted as GPU readbacks. Its
regression checks zero readback for the matching hint, stale/modified hints,
step invalidation, numerical parity and atomic rejection of invalid batches.
Motion edits also validate motion directly, without cloning immutable cell
geometry per leaf impulse.

### Air refinement and repeatability

`make test-airflow-refinement` passes both explicit GPU studies. The smooth
transported vortex's spatial RMS velocity errors against a 64³ reference are
0.009465, 0.005421 and 0.002823 m/s at 16³, 24³ and 32³. On the 32³ grid,
temporal errors against 480 Hz are 1.0397e-4, 4.7938e-5 and 2.0438e-5 m/s at
30, 60 and 120 Hz. Errors decrease monotonically; repeating the same 32³ run
produces zero sampled difference on this adapter.

The 20 N rotor study samples a plane 0.9 m below a radius-0.28 m disk:

| Spacing (m) | Mean downwash (m/s) | Sampled vertical momentum flux (N) |
| --- | --- | --- |
| 0.2000 | 2.0448 | 19.5597 |
| 0.1333 | 1.7453 | 19.5543 |
| 0.1000 | 1.6366 | 20.0150 |
| 0.0667 | 1.4822 | 20.0486 |

Velocity-profile RMS errors against the finest case decrease 1.2083 → 0.7612
→ 0.5743 m/s. These are finite-grid comparisons, not an exact analytical
solution or physical calibration. Integrated flux is relatively insensitive
here, but the local wake is still grid-sensitive. Do not present the default
0.15 m orchard grid, leaf counts or petiole thresholds as converged physical
predictions. All cases retain zero ground-normal air velocity.

### Full flight, causality and regressions

- `artifacts/drone-orchard/full-row.json`: verified 32-second outward/return
  flight, 41/2,520 shed, furthest z=2.8701 m, return within 1.2 mm of the
  starting hover command, fixed roots, no wood failure, divergence RMS
  1.2886e-4 /s and zero ground-normal velocity. Mean physics cost 21.751 ms
  per air frame; wood 12.736 ms. This is not guaranteed real-time 60 fps.
- `artifacts/drone-orchard/settled-causal.json`: verified four-second paired
  experiment, 1.7936 cm maximum / 1.2171 mm RMS air-induced wood difference,
  11 leaves shed versus zero in the control. Control relaxation stays below
  0.5 mm. Repeated coupled runs reproduce the same numerical report fields;
  timings vary on the shared desktop.
- Final focused tests: 21 orchard units; 9 CPU air/drag; 1 air shader;
  3 air GPU; 2 refinement; 4 rotorcraft CPU plus all 14 existing devices;
  1 flight/air GPU; 4 attachment CPU and 2 attachment GPU; 1 strand shader;
  20 strand GPU, 4 contact GPU and 3 splinter GPU. All passed. The explicit
  dense-bundle stress oracle was excluded; no full-workspace claim is made.
- The previous leafy example still passes 25 units (two explicit tests
  ignored by the ordinary unit run) and its actual GPU rest/shake/topology/
  finite/floor verification. Its authored parameters remain unchanged.
- Final isolated air benchmark: median 1.173 ms, but a 248.128 ms maximum
  outlier in this run. Earlier runs had lower maxima. Do not turn median
  kernel cost into a guaranteed whole-scene or native-window frame rate.

Commands: `make verify-drone-orchard`, `make verify-drone-orchard-row`,
`make test-airflow-refinement`, `make test-strands-gpu RELEASE=1` and the
existing foundation/legacy targets. Workstation environment remains the Nix
prefix documented above. Logs: `/tmp/molla-orchard-final-{row,causal,
regressions,strands,legacy}.log`.

### Final delivery

- `artifacts/drone-orchard/orchard-flight.mp4`: 1920x1080, 30 fps, 960 frames,
  32.000 seconds, title `Molla | Drone orchard`. Frames at 4, 16 and 26 seconds
  were inspected: all six trees, changing aircraft location, rotor wake,
  ground shadows and detached leaves are visible. Accompanying PNG, JSON and
  encoder log are present. The final JSON matches the full-row physics run:
  41 leaves shed, fixed roots, no wood failure, identical final divergence.
  SHA256: `47713f47f9cbbdde86f1601f5d21070ce055d8461c3ada5ccf1a9e0a2e4fdd2e`.
- The updated native `Molla · drone orchard` is running on Wayland workspace
  5, PID 4119541 at delivery. `artifacts/drone-orchard/native-preview.png`
  shows the actual 3348x1348 window at 6.3 simulation seconds, 2,506 leaves
  attached, no displayed error, and a 31.9 ms last-step timing. This is a
  single native frame measurement, not the 21.75 ms mean headless probe.
  Only the previous agent-launched orchard preview was closed for replacement;
  no unrelated applications or desktop settings were changed.
- Focused formatting checks and `git diff --check` pass. Existing unrelated
  dirty PPF/tree work was preserved. No commit or push was made for this goal.

## Session close

### Goal
Research/reuse flight mechanics and implement shared 3D rotor airflow with a
separate dynamically flown orchard example, elastic branches and load-driven
leaf shedding.

### Instructions
Use Make targets, no Python edits, preserve old examples, and acknowledge
research sources. Keep this goal separate from the old tree-fracture task.

### Discoveries
- Existing flight devices were reusable; existing 2D/CPU-Stokes fluids were
  not the needed resident 3D shared-air backend.
- Contact coloring, repeated readback and host upload behavior dominated
  the initial scene cost; the fluid kernel alone did not predict performance.
- Settled initialization and uncoupled controls separate ordinary gravity
  relaxation from rotor-induced response. A shared initializer makes mechanical
  rate comparisons start from identical states.
- Conserved rotor force/torque deposition and low divergence do not establish
  local wake convergence or physical canopy/abscission calibration.

### Accomplished
Reusable air/rotor coupling, bidirectional wood/leaf drag, GPU batching and
transfer optimizations, settled 60 Hz example with higher-rate options,
causal/refinement/long-flight/regression gates, inspected 1080p video and a
running native preview. Citations and limitations are documented.

### Next steps
No required work remains for this scoped goal. Physical calibration, finer
near-rotor flow, drone/tree collision coupling and further real-time
optimization are future extensions, not claims of this implementation.

### Relevant files
- `crates/molla-solvers/src/airflow.rs`, `airflow/` and
  `crates/molla-kernels/shaders/airflow.wgsl`: shared air and flight coupling.
- `crates/molla-solvers/src/attachments/` and `strands/`: local wind,
  reactions, independent islands and cached-state impulse publication.
- `crates/molla-compute/src/upload.rs`: bounded write-only mapped uploads.
- `bin/molla_cables/src/{drone_orchard,orchard_sim,orchard_model}.rs`: the
  separate orchard executable, simulation orchestration and scene assembly.
- `bin/molla_cables/DRONE_ORCHARD.md`, `Makefile`, `ACKNOWLEDGEMENT.md`:
  operation, reproducible gates and scientific attribution.
