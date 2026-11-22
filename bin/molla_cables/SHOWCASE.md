# Apple-tree showcase capture

## Requested presentation
Full-screen simulation, only small cut/break tool indicators and a
`molla alpha 0.0.1` watermark. No panels, frame, explanatory cards or outro.
Release the bent branch and show free motion before breaking another branch.
Show four rough breaks, remove the remaining full branches at their collars,
then make two cuts through the main trunk, leaving three trunk sections.
Source citations accompany the video rather than covering it.

## Capture
Offscreen Vulkan rendering reuses the example's actual egui/wgpu studio
renderer. Fixed simulation intervals produce each frame independently of
wall-clock speed. No desktop resizing, screen recording, dropped frames,
body teleportation, fragment deletion or artificial damage assignment.
FFmpeg receives raw RGBA frames and writes silent H.264/yuv420p MP4 with
BT.709 and fast-start metadata. Motion is 0.5×, except the first one-second
intact-tree hold. Camera moves and tool-selection cues advance physics.
This is an offline showcase, not a real-time performance measurement.

```sh
make test-apple-tree-capture
make test-apple-tree
make test-apple-tree-showcase
make capture-apple-tree
make verify-apple-tree-video
make bundle-apple-tree-video
```

Default MP4: `artifacts/apple-tree-clean/molla-alpha-0.0.1-1080p.mp4`.
Default bundle: `artifacts/molla-apple-tree-clean.zip`.
`VIDEO_OUTPUT`, `VIDEO_BUNDLE` and `SHOWCASE_REPORT` override their paths.
Existing final movies and bundles are refused. An unsuccessful capture
removes its partial MP4. The original designed video is preserved separately
under `artifacts/apple-tree/`.

Use the Nix/NVIDIA prefix from `APPLE_TREE.md` when needed.
FFmpeg with libx264, ffprobe and jq must be available. No display server is
needed; physics and rendering both use the GPU.

## Released-branch delayed failure
The default tree has 20 limbs and 59 cells, with the default 4 N grip.
Grab the center of span 49 and for `step = 0..97` move the target to
`initial_point - Z * step / 240`, advancing four nominal physics steps
per sample. Release. Joint 48 has accumulated damage 0.700553894.

Advance thirty four-step batches after release: 0.5 simulated seconds,
or one full second of visible free motion. Then rough-break span 44.
The previously weakened joint 48 fails after 56 further batches
(0.933333 simulated seconds). It is never selected by either section tool.
The unedited released checkpoint survived five simulated seconds in the
scenario probe; the full sequence independently checks a three-second
control from the post-gap checkpoint.

Probe evidence: `/tmp/molla-showcase-release-scenario.json` and
`/tmp/molla-showcase-release-probe2.log`. These are temporary development
files, not portable delivery assets. Higher-load trials caused spontaneous
control failures and were rejected. The earlier immediate-release/span-50
scenario did not survive an added release gap as the desired causal example.

## Pruning and trunk sections
The other three rough breaks target spans 32, 41 and 56. Clean cuts then
remove every remaining primary branch at its actual trunk intersection.
Small solid collars are retained; the model checks that no other branch
material remains attached to the root. All detached wood remains simulated.

Two oblique cuts at two-thirds and one-third of the original trunk
centerline length produce three connected components containing trunk
material. The oblique faces allow gravity to slide the detached sections,
without an artificial kick. The bottom section remains clamped.
Mass conservation, finite state, ground clearance, event ordering and
exactly three final trunk components are required.

## Validation status
The revised one-second 1080p capture was decoded and visually inspected:
full-frame tree, no framing UI, small version watermark.
Fifteen example unit tests and the baseline GPU sway/cut/fall/CPU-oracle
checks passed. The complete revised headless sequence passed: 3,305 frames
(110.167 seconds), four rough breaks, one delayed load-driven failure,
thirteen whole-branch cuts and two trunk cuts. Exactly three trunk sections
remain, with mass conserved to floating-point roundoff and floor clearance
above -8e-9 m. The final movie has the exact same event manifest as the
headless test. Complete MP4 decode and codec/resolution/frame-count/duration
checks passed. The whole-film contact sheet and release, branch-cut and
trunk-cut temporal strips were visually inspected, along with full-size
snapshots of the four breaks, bare trunk and final three sections.

MP4 SHA-256:
`55a1031ec8642e3e4965c962ddf51001af198d2d07ce99396f359ad5b4b7de67`.
The delivery folder includes citations in `CREDITS.md`, the complete source
acknowledgements, dataset attribution and the physical event manifest.

The video verifier checks event order, separated release/break timing,
four rough breaks, two trunk cuts, final component count, mass and floor
invariants; exact codec/resolution/frame count/duration; and complete decode.
It creates a whole-film contact sheet and decoded temporal strips for
release/delayed failure, clean cutting and the final trunk cut.

## Relevant files
- `src/video_capture.rs` — fixed-size offscreen renderer and MP4 encoder.
- `src/apple_showcase.rs` — minimal presentation and physical sequence.
- `src/apple_showcase_probe.rs` — counterfactual scenario search.
- `src/apple_tree_model.rs` — existing geometry plus trunk-cell metadata.
- `src/draw.rs` — close-up camera target; interactive defaults unchanged.
- `src/branch_draw.rs` — shared geometry with optional labels.
- `verify_video.sh` — movie acceptance and decoded frame sheets.
- Root `Makefile` — reproducible capture/check/bundle commands.

## Follow-up
After this revised recording, create a separate example with tapered
third-level branches and leaves restricted to those small branches.
Leaf attachments must release under physical load when shaken, not by
random visibility toggles. Preserve the original bare-tree example.
The leafy example has not been implemented.
