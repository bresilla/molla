# Brush contact geometry and acceptance

## Geometry and force integration

The CPU brush and `tire_brush.wgsl` approximate the tyre circumference with a
polygon. Each persistent sample owns half of each adjacent polygon interval.
Penetration varies linearly along those intervals; only its positive part is in
contact. A boundary cell can therefore contribute a fraction of its length.

The projected covered length times strip width gives cell area. Its integrated
penetration gives normal load; the pressure centroid locates the sample. Normal
stiffness, tread forces, friction/cohesion limits and readout consume those same
covered cells. Persistent bristle indices are retained, with history reset when
coverage vanishes. This is a reduced brush approximation, not a FEM tyre.

For an undeformed circular section of radius `R`, hub height `h` above a flat
plane and tread width `w`, the circle intersection has length
`2 sqrt(R² - h²)` and area `w` times that length. The polygon lies between circles
of radii `R cos(pi/N)` and `R`. Those circles bound its footprint. Refinement
tests also compare the integrated linear normal spring law with
`w [R² asin(a/R) - h a]`, where `a = sqrt(R² - h²)`.

The previous centre-to-centre patch length omitted fractional boundary cells.
A hand-tuned passenger preset's 5 kN / 1.2 bar result consequently no longer fits
an old, unsourced 0.35 m upper bound. Acceptance now checks the actual polygon/
circle bounds and normal-load balance, not a widened empirical size interval.
These geometry tests are not evidence that the preset matches a real tyre.

## Aligning-moment checks

The stateless brush uses different static and kinetic friction coefficients and
a smooth utilization-dependent transition between their caps. Its force law can
decrease over part of that transition. A fixed lateral-slip direction alone does
not imply that the moment computed from this law keeps one sign at every slip
magnitude.

Controlled tests isolate this behavior at unchanged pressure, footprint and normal
load: the dry-surface coefficient gap gives moment reversals at 64–512 samples;
setting only the kinetic coefficient equal to the static coefficient removes them.
Mirroring lateral slip reverses lateral force and moment while preserving
longitudinal force. This is evidence about this implemented model, not a claim
that its reversing moment curve is calibrated or correct for a particular tyre.

The continuity report retains its original dry-surface force/moment jump and
friction-budget limits. It additionally evaluates mirrored lateral-slip samples
and an equal-static/kinetic-friction control. The sign gate requires a restoring
moment in that control; a separate tight numerical gate checks odd symmetry of
the original variable-friction moment. No production friction coefficient was
tuned to satisfy these checks.

Acceptance constructors reject nonfinite measurements and bounds before
normalizing their printable fields. Mutation tests cover NaN and both infinities,
alongside finite inclusive bounds and mirrored-moment failures.

The distinction between static and kinetic friction is consistent with the
framework discussed by [Kikuuwe, 2019](https://onlinelibrary.wiley.com/doi/10.1155/2019/9747605).
That paper does not validate this implementation or its hand-tuned parameters.

## Evidence and remaining scope

- Coverage tests check clipped integrals, continuous area/load growth, resolution
  convergence, width invariance, geometric bounds and matched normal loads.
- GPU brush tests compare area, length, force and moment, including varying
  compression, pressure, camber, heightfields and width-dependent friction.
  `BrushWheelOutput.diag5.z` contains contact area in square metres; buffer layout
  is unchanged.
- The imported Kubota regression adds/removes a 1500 kg payload at unchanged
  1.8 bar and checks per-wheel compression, growing area, support balance and
  restoration. Separate mesh/uniform tests exercise weight-driven visible shape.

Real tyre calibration, broad terrain/soil validation, transient and integrated
GPU pressure/history parity, and all remaining `FULL_COMPACT.md` acceptance gates
remain separate requirements. A passing brush parity test does not satisfy them.

## Independent-machine lifecycle

Wheel readouts retain per-machine structural and explicit-state witnesses.
Registering, configuring or removing one wheel invalidates only that wheel's
samples. A changed articulation invalidates its own samples, while ground edits
still invalidate samples depending on the registered terrain. Raw mutable state
access invalidates all body witnesses. No-wake pose and velocity edits update
witnesses without changing their requested wake behavior.

`WheelForces::samples` returns a borrowed `WheelSamples` view rather than a slice.
It supports forward/reverse iteration, indexing, length and endpoint access without
allocating a filtered copy. Unchanged scene/state stamps retain the direct path;
component checks occur when the scene or explicit state has changed.

The isolated Gearbox material importer records the last applied authored material
and collider identity. Unchanged USD material values no longer overwrite live
controller friction/restitution. Authored edits, rebinding and collider replacement
still apply the current authored values. This is necessary because live machine
projection previously reset existing Kubota wheel friction from 1.1 to 0.9.

The live imported regression spawns another real Kubota 25 m away while the first
drives and inflates, sets the second machine to 0.8 bar independently, then despawns
it through the normal dead-physics cleanup. Projection, alignment and despawn must
preserve the first machine's complete pose/velocity/tyre snapshot exactly. Two
identical scripted runs must match bitwise across every simulated step. Comparing
the different one-machine and two-machine layouts uses `1e-9 * max(1, abs(value))`
per numeric field: the initial bitwise cross-layout trial differed by approximately
`2e-17` in velocity, while the full corrected trial's peak normalized difference
was `1.39e-11`. These are numerical lifecycle checks, not tyre calibration bounds.
