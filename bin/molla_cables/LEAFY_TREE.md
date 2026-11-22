# Leafy apple-tree example

For the second video with four extra bending sequences, close-up splintered
breaks and large branch/crown cuts, see [`LEAFY_DETAIL.md`](LEAFY_DETAIL.md).
Its recording is separate from the original pruning video documented below.

## Goal
Separate cuttable tree example with tapered leafy shoots and individual
leaves whose petiole attachments fail under force. Preserve the bare-tree
example and its completed video.

Every retained branch receives shoots, regardless of its hierarchy depth
or length. The trunk receives small shoots from its lower stem to within
6 cm of its tip, with less than 30 cm between attachment heights. Shoots
and leaves taper toward the top. Leaves remain attached to these physical
shoots, not painted directly onto the original trunk or branches.

## Design
- Reuse the shared apple-tree simulation and GPU wood solver.
- Add a bounded number of real tapered twig cells, not decorative branches.
- Adapt Appleskel's ovate, cupped, serrated leaf geometry and alternate layout.
- Batch leaf point masses and implicit spring/damper attachments on the GPU.
- Return attachment reactions to the wood. Different strengths belong to
  individual leaves; failure is irreversible until reset.
- Remap attached leaves by their original material points after a cut/break.
- Detached leaves retain velocity, experience drag/gravity and land on the
  ground. Leaf blades are a reduced point-mass model, not deformable shells.

## Status
The full-canopy revision has 61 real tapered shoots and 420 individually
attached leaves, 120 total wood cells. All 20 retained branches have shoots,
including nested and short limbs. At-rest control keeps all leaves. The
current default shake sheds four leaves with no wood failure (41 before
strengthening and two in the first, overly strong trial).
GPU cut, rough-break, material remapping and falling checks passed.
Leaf attachments are 2.2 times the original default, eased back from the
first 3-times-strength trial at the user's request. The new video opening
shakes two branches in different directions, shedding six leaves with no
wood failure. All motion and damage carry forward into the pruning sequence.
The palette spans forest green, pine, apple green, sage, lime and yellow-green;
vein colors follow each blade. Colors are stable and independent of strength.
The complete leafy pruning recording is delivered: 4,448 frames,
148.267 seconds at 1920×1080/30 fps, H.264/yuv420p with BT.709 color.
It has no panels or framing UI, only tool markers and the alpha watermark.

The video opening shakes cells 21 and 11 at different rotations. The loaded
branch is then released after 86 four-step batches and remains released for
one video second before cell 44 is rough-broken. Bond 48 survives an unedited
three-simulated-second control at damage 0.83482, but fails 0.85 simulated
seconds after that separate break. This is verified with the current leaf
state and carried damage, not the bare-tree timing or forced damage values.
Shaking changes the load history: copying the bare-tree bend duration caused
premature failure, so the leafy release scenario has its own checked input.

The earlier generator only populated primary branches longer than 30 cm
and skipped attachments below 28 cm. Those exclusions have been removed.
Shoot azimuths are checked against unrelated wood before creation, while
preserving the solver's two-hop junction neighborhood. Three initial
intersections in the first full-canopy trial caused idle shedding; selecting
clear azimuths removed that failure without weakening contacts or leaves.
The radial frame also handles vertical trunk segments without degeneracy.

The leaf attachment reaction is applied at the actual petiole attachment
point in the wood, not at the offset leaf particle. Applying it at the
particle introduced spurious thin-twig torsion and idle shedding; the
correct point removed that instability. This is a force-only petiole
coupling, not a full rigid-leaf rotational constraint. Detached blade
orientation/flutter is visual; particle translation and failure are solved.

Leaf strength is proportional to its mass times gravity, with deterministic
variation from 3.96 to 14.96 times its own weight. The default horizontal shake
of a stiff proximal cell did not exceed those thresholds; shaking a distal
branch vertically as well produces selective failure without weakening
leaves below their static weight.

Sources are credited in `ACKNOWLEDGEMENT.md` [26] and [4].

## Acceptance
- 21 example tests passed, including palette diversity, capture safety,
  shoot collars, coverage of every branch, trunk-tip
  coverage and taper, vertical frames, and no unrelated initial overlaps.
- Three CPU attachment tests and a 720-step CPU/GPU oracle passed: static
  weight, different strengths, missing anchors, irreversible failure,
  velocity continuity, drag, ground contact and invalid-input atomicity.
- Retained wood impulse batches matched the CPU and preserved atomicity;
  scoped GPU counters recorded no geometry-buffer reallocations.
- Before strengthening, the full-canopy four-branch/direction matrix passed
  cells 11, 21, 37, 54 at rotations 0, 1.7, 3.1, 4.7 radians, shedding
  35, 50, 5, 10 leaves respectively with no wood failure.
- The 3-times-strength trial shed 3, 3, 0, 0 in that matrix. Its native
  shake/cut check passed with four fallen leaves and exactly one cut.
  Native verification isolates scripted input and keeps the click location
  fixed through press/release rather than following the moving twig.
- The bare-tree example's 15 tests and GPU sway/cut/fall/oracle gate passed.
- Intact and post-shake/cut native framebuffers were inspected.
- The final leafy example's 21 tests and GPU rest/shake/cut/break/remapping
  check passed at the final 2.2-times attachment setting. At rest all 420
  leaves survived; the default shake dropped four without wood failure.
- The complete recording passed action ordering, opening shedding, released
  control, delayed failure, four rough breaks, 23 whole-branch/shoot cuts,
  two trunk cuts and exactly three trunk-component checks. Wood mass remained
  1.447210507686616 kg (difference below 1e-14); final minimum wood clearance
  was -1.04e-8 m. All 420 leaf particles remained; 36 were still attached.
- The encoded file passed full decode, format and exact frame/duration checks.
  The opening, delayed failure, four-breaks, pruning and trunk sequences were
  inspected using rendered stills and decoded contact sheets.

Measured full-canopy initial nominal step cost was approximately 10.4 ms on RTX 4080
Vulkan, including CPU/GPU synchronization; some runs shared the GPU with
other checks. This is not a real-time guarantee. Live updates stop adding
steps after an 8 ms budget (one step can exceed it). Fine splinters can
require 32 substeps and substantially slower simulation. The leafy example
retains successful refinements up to 64 subdivisions, checking coarsening
after 120 batches rather than repeatedly retrying a failed coarse step.
The original bare example retains its previous 8/12 settings.

## Run
```sh
make run-leafy-tree
make test-leafy-tree
make test-leafy-tree-matrix
make test-leafy-tree-ui
make test-attachments test-attachments-gpu
make test-leafy-tree-capture
make test-leafy-tree-intro
make capture-leafy-tree
make verify-leafy-tree-video
make bundle-leafy-tree-video
```
On the current workstation use the Nix/NVIDIA prefix in `APPLE_TREE.md`.
Grab/bend [G], Shake selected branch [S], Cut [C], Break [B], Reset [R].
Right-drag orbits; the wheel zooms. Each leaf's load view is normalized by
its own strength. Changing the strength multiplier affects future failure;
only Reset reattaches already detached leaves.

## Relevant files
- `src/leafy_tree.rs` — separate native example, leaf/wood coupling and tools.
- `src/leafy_model.rs` — tapered twig bodies and individually seeded leaves.
- `src/leafy_draw.rs` — curved, serrated, double-sided leaf mesh and veins.
- `src/leafy_showcase.rs` — shake, released bend/delayed failure, four rough
  breaks, branch pruning and three-part trunk capture with event manifest.
- `src/leafy_probe.rs` — released-state/control comparison for the leafy
  delayed-failure sequence; no forced damage or detachments.
- `src/video_capture.rs` — shared fixed-step 1080p framebuffer/MP4 capture.
- `src/apple_tree_sim.rs` — shared original wood simulation, extracted intact.
- `crates/molla-solvers/src/attachments.rs` — reusable CPU attachment oracle.
- `crates/molla-solvers/src/attachments/gpu.rs` — retained GPU leaf buffers.
- `crates/molla-kernels/shaders/attachments.wgsl` — implicit attachment step.

## Limitations / follow-up
Leaf particles contact the ground, not each other or branch collision
meshes. The wood retains its existing wood/wood and ground collision solver.
Leaf flutter orientation is visual, not a solved rigid-leaf torque model;
there is no shell deformation or calibrated abscission biology.
No image-generation service or Blender runtime is required. Credits remain
in `ACKNOWLEDGEMENT.md`; the original pruning video is unchanged.

## Delivery
- Video: `artifacts/leafy-tree-video/molla-leafy-alpha-0.0.1-1080p.mp4`.
- Bundle: `artifacts/molla-leafy-tree-1080p.zip`, with the MP4, source/data
  credits, event manifest, release-control evidence, preview images and hash.
- Opening preview: `artifacts/leafy-tree-opening-preview.mp4`, 405 frames.
- Native launch: `make run-leafy-tree` (full-canopy example).

The completed bare-tree video was not overwritten. Its SHA-256 remains
`55a1031ec8642e3e4965c962ddf51001af198d2d07ce99396f359ad5b4b7de67`.
No commit or push was made. Full leaf/wood collision and rotational petiole
coupling remain potential extensions, not implemented features.
