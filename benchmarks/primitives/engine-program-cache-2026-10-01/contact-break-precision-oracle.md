# Detached translation oracle audit

The unchanged fixture failed at frame 10 with a 3.552713679e-15 m difference
between a newly free root and a body still translated from its parent's joint
coordinate. The preserved diagnostic shows identical velocities, unchanged
root motion, and matching poses before detachment. Even before detachment,
the parent's position plus the exact one-metre offset differs from the child's
published position: 1.0000004000000224 + 1 versus 2.000000400000019 at frame 9.

Each published position is a sum of two binary32 numbers. Changing the high
part's exponent changes the representable low-part spacing. Rounding a child's
world translation at each free-root step is not bitwise equivalent to evolving
its parent's coordinate then adding the fixed frame. A strict binary32 two-sum
host simulation (without GPU contraction) also differs: start at 1, advance
by f32(1e-8), copy the translated pair at step 41, then evolve the detached pair.
At step 54 the reference is 2.0000005400000305 and the detached value is
2.0000005400000234. Maximum difference through 320 steps is
7.105427357601002e-15 m. Thus bitwise trajectory equality does not follow from
preserving the state exactly at the detachment boundary.

The revised fixture retains the original bodies, velocity, 80 frames, dt=0.004,
four-substep run, and event/velocity assertions. It adds one/eight-substep runs.
The unaffected root and all pre-break poses remain exactly equal; the
one-substep run additionally checks exact equality at the breaking boundary,
before any detached-root integration. Rotation, transverse position and velocity
remain exact for every frame.

For the subsequently integrated detached translation, the fixture checks both
GPU trajectories against independent uniform-motion kinematics using
the GPU's binary32 velocity and timestep. A conservative paired-roundoff budget
is 4 * epsilon32^2 * origin * (step_count + 1) per trajectory, twice that for
the difference of two trajectories. The scale and step count are explicit,
not fitted to the observed discrepancy. Two-sum residuals are order epsilon32
times the high part; rounding their additions contributes order epsilon32^2
per step. The original four-substep end-of-run per-trajectory bound is
3.6493474908638746e-11 m for the child, many orders below lost binary32 world
translation residues. Maximum analytic and parity errors are printed separately
for all three refinements.

This changes an over-constrained numerical oracle, not production code or the
plan's physical acceptance thresholds. Original failure logs and fixture remain
available; acceptance requires the revised physical assertions to pass.

## Independent oracle result

The added one-substep analytic assertion failed before detachment on both GPU
worlds: 1.000000099604648 instead of 1.0000000400000009. The repeated error was
5.960464722e-8 m, not paired-float roundoff within the declared bound. Stage
readbacks showed correct refresh, prediction and drift, followed by a spurious
momentum-transport shift. The COM correction added the tiny translation to a
0.5 m center before subtracting another rounded center.

The production fix computes center differences directly from high/low body
positions and relative rotated COM offsets before summing the transport shift.
The failed two-sum-operand experiment was reverted and tracing removed.
The unchanged strengthened oracle then passed for 1/4/8 substeps: maximum
analytic errors 1.758593271e-13, 9.769962617e-14 and 8.792966355e-13 m; maximum
trajectory differences 1.421085472e-13, 8.526512829e-14 and 8.739675650e-13 m.
Exact boundary continuity and velocity/rotation/event checks also passed.
See relative-momentum-contact-break-gpu.log (1 test, 289.38 seconds, exit 0).
