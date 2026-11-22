# Leafy tree: bending, fracture close-ups and large cuts

This second recording preserves the first leafy movie and the bare-tree movie.
It uses the same physics and leaf strength; only tool inputs and camera work
change. No force, damage, collision or geometry parameters are weakened for
the shots. Three extra bends use a 2 N grip; the thicker low branch uses
the default 4 N grip at its distal span.

## Sequence

1. Shake two branches, shedding a few leaves.
2. Bend and release the original weakened branch. After a visible release
   interval, break a different branch and observe the delayed failure.
   An unedited copy of the released state must survive three simulated seconds.
3. Bend four further branches in different directions, reversing the input
   and leaving free recovery time after each release.
4. Show three more rough breaks close-up, then pull back to see the falling
   branches. The exposed teeth are actual fracture/contact geometry, not
   overlaid decorative fibers. The reduced model does not resolve microscopic
   wood grain.
5. Cut a large branch halfway along its original path, watch it fall, then
   remove its lower remnant. Every falling piece remains simulated.
6. Cut the trunk at 53% of its original length while its crown still carries
   branches. Allow time for the large assembly to fall and interact with debris.
7. Cut another remaining branch in two stages, then cut the lower trunk at 22%.

The capture remains full-frame 1920×1080 at 30 fps with only tool markers and
the small alpha watermark. It is fixed-step offline capture at half-speed,
not a real-time benchmark. The first two seconds hold the initial state.

## Commands

Use the workstation Nix/NVIDIA prefix in `APPLE_TREE.md` when needed.

```sh
make test APP_TARGET='--release -p molla_cables --bin molla_leafy_tree'
make test-leafy-detail-bend test-leafy-overlay
make test-leafy-detail
make capture-leafy-detail
make verify-leafy-detail
make bundle-leafy-detail
```

Default output: `artifacts/leafy-tree-detail/molla-leafy-bending-1080p.mp4`.
Override `LEAFY_DETAIL_OUTPUT` for another destination. Existing videos are
refused rather than overwritten. `MOLLA_SHOWCASE_PREVIEW=/path/preview.mp4`
also renders checkpoint stills during a headless sequence check.

The manifest records input directions, actual bend excursions, released
intervals, camera distances, cut-detached masses and cell counts, and the
final mass/floor invariants. The video verifier checks those requirements,
the encoded dimensions/frame count/duration, and decodes the entire movie.
Rendered stills and temporal strips still need visual inspection.

Overlays are painted on the active scene UI layer after its GPU callback.
The GPU capture test checks that the small version mark is present in the
actual encoded-frame source image.

## Verified delivery

- 4,363 frames, 145.433 seconds, H.264/yuv420p, BT.709, 1920×1080 at 30 fps.
- Four additional bend/reversal/release sequences on spans 21, 11, 37 and 55.
  Actual peak excursions were 11.05, 11.38, 4.15 and 11.18 cm; none caused
  new stress fractures. The initial weakened-branch sequence also remains.
- Four rough breaks, six clean cuts, and two load-driven failures in total.
  The second load failure occurred after the falling crown interacted with
  the remaining tree. No extra fracture was explicitly selected for it.
- The crown cut detached 41 cells, approximately 0.258 kg of wood. The
  later lower-trunk cut detached 38 cells, approximately 0.458 kg.
- Three trunk-containing components remain. Wood mass differs by less than
  1e-14 kg; final minimum floor clearance was -6.48e-9 m. All 420 leaf
  particles persist, with 79 still attached to wood.
- Twenty-five example/unit tests passed. Both separately invoked GPU tests
  passed: distal bending without fracture, and captured watermark visibility.
- The recording passed choreography checks, exact metadata/frame-count
  checks, and full decode. Bend stills, close-up fracture strips, the crown
  fall, whole-film contact sheet and final debris frame were inspected.
  Leaves briefly cross the lens during one fracture, then clear its exposed
  splintered face; no leaves were hidden to stage the close-up.
- The original leafy and bare movies retain their original SHA-256 hashes.

Movie SHA-256:
`5b15415fae81c392d274dccb613dddb8883fc75e0c0dd75b15b07183ce5d87a4`.

The bundle is `artifacts/molla-leafy-bending-1080p.zip`, with citations,
data attribution, the event manifest, metadata, cover and contact sheet.

## Sources

Molla's independent GPU strand solver and reduced fracture model underpin
this example. Published tree-strand/fracture inspiration is credited to
Li et al., *Stressful Tree Modeling: Breaking Branches with Strands* (2025),
https://doi.org/10.1145/3721238.3730745, and EvoEngine. This is not a port of
EvoEngine or its complete volumetric fiber model.

Ando's *A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness* (2024),
https://doi.org/10.1145/3687908, and ZOZO's PPF Contact Solver are the primary
references for Molla's separate barrier-contact work, not the solver running
this tree video. See `ACKNOWLEDGEMENT.md` for the complete research lineage.
AppleQSM/Appleskel tree-data and procedural leaf attribution accompanies the
delivery. Material strengths remain illustrative rather than calibrated.
