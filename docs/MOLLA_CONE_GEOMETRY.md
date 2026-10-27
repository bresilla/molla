# Spherical cone geometry contract

This defines the P2 geometry and initial runtime impulse layer, not complete
certification of cone dynamics. Contact coupling, general reactions and all
tree/loop acceptance gates remain required by `MOLLA_PRIMITIVES.md`.

## Axes and frames

Let `p` be the normalized `JointDesc::axis` in each authored joint frame. For
positive X/Y/Z, transverse axes `b0,b1` are respectively Y/Z, Z/X and X/Y.
For an arbitrary axis, take its largest absolute component (X wins ties), project
the next cyclic coordinate axis onto its perpendicular plane to obtain `b0`, and
set `b1 = p cross b0`. Both authored frame rotations participate in the relative
rotation `R = inverse(R_parent_frame) R_child_frame`.

OpenUSD defines half-angles toward the cyclic transverse axes, independently
disabled negative values, and degree units. Molla uses optional nonnegative
radians instead. OpenUSD does not specify the numerical ellipse used below;
the geodesic construction is an engine design choice.
[Schema contract](https://openusd.org/release/api/class_usd_physics_spherical_joint.html).

## Geodesic ellipse

Set `n = R p`, `c = dot(p,n)`, `t = n-c p`, `s = length(t)`, and
`theta = atan2(s,c)`. The two swing coordinates are
`u_i = (theta/s) dot(b_i,t)`, with their continuous zero-angle limit.
Their squared norm is `theta^2`. Thus equal positive extents `a` give exactly
`theta <= a`, while unequal extents define a disk-log ellipse, not Euler clamps.
Postmultiplying R by any twist about p does not change n or the constraint.
Quaternion sign does not enter this construction.

For positive enabled extents, let `h` be the smallest such extent and
`w_i = (h/a_i)^2` (zero for disabled or zero extents). The unilateral row is
`F = sum(w_i u_i^2) - h^2 <= 0`. Scaling keeps the coefficients bounded.
A zero extent adds a bilateral row `u_i = 0` instead of dividing by zero.
Two disabled extents produce no rows. Extents are in [0,pi]; when both extents
are disabled or pi, all primary directions are admissible and no rows are needed.

The radial angular violation of the positive-extent ellipse is
`max(theta - theta/sqrt(sum((u_i/a_i)^2)), 0)`, with zero violation for a zero
denominator. Zero-extent violations are the absolute locked swing coordinates.
Outside the antipodal chart, reported violation is the maximum of these values,
in radians; it is not a general closest-point distance to an anisotropic ellipse.
For a disabled or pi-wide extent, the antipode is admissible. If its geodesic
distance `pi-theta` is smaller than the radial violation, geometry uses that
return branch and reports the smaller distance instead.

## Angular Jacobian

For relative angular velocity omega expressed in the parent joint frame,
`n_dot = omega cross n`. Away from the poles define
`k = theta/s`, `q = (c*s-theta)/s^3`. Then

```
d_i = n cross (k*b_i + dot(b_i,t)*(q*t-p))
u_i_dot = dot(d_i,omega)
F_dot = dot(2*sum(w_i*u_i*d_i),omega)
```

At zero swing, `d_i = p cross b_i`; near zero use the series
`k = 1 + theta^2/6 + 7*theta^4/360` and
`q = -2/3 - theta^2/5`. Each Jacobian is orthogonal to n, so it cannot oppose
child twist. Rotate the row into world coordinates for equal/opposite endpoint
torques. The kernel only evaluates geometry; it does not apply these torques.

The GPU request expresses the relative quaternion in the orthonormal basis
`(p,b0,b1)`. It evaluates the rotated primary direction in this canonical frame,
then maps angular rows back to the parent joint frame. Packing uses the original
f64 quaternion, before narrowing its basis components to f32. Computing `n-c p`
directly with non-cardinal f32 axes amplified cancellation near the antipode.
Future device-generated requests must preserve this convention and pass their
own integrated near-antipodal checks; host-prepared geometry parity is not proof
of accuracy for a device-only frame transformation.

Away from zero swing, write `e = t/s`, `r = p cross e`. A cancellation-resistant
form of the derivatives is

```
d_i = dot(b_i,e)*r + (theta/s)*dot(b_i,r)*(n cross r)
F_omega = 2*theta*sum(w_i*e_i^2)*r
        + 2*theta^2/s*(w_1-w_0)*e_0*e_1*(n cross r)
```

The GPU uses this form for `theta >= 1e-3` and the series form below that angle.
In particular, the azimuthal term cancels exactly for a circular cone.

## Configuration and execution status

`RigidScene::cone_limits` and `set_cone_limits` accept only ball-joint handles.
Default settings mean both extents disabled. Configured tree/loop joints appear
in `cone_joints`, including disabled joints. Mutations rebuild transactionally,
preserve stable handles, and activate endpoint dependencies; identical settings
leave stamps unchanged. Clearing settings removes their sparse entry, and joint
or incident-body deletion removes it as well.

`RigidWorld` now assembles cone angular rows alongside distance rows and scalar
stops in each articulated island. A row's Jacobian and error are scaled by the
Jacobian magnitude, with a unit divisor for a zero gradient. Rows use equal and
opposite endpoint torques, the shared mass response, and re-integration from the
substep input after the coupled velocity solve. Bilateral zero extents accept
signed impulses; positive extents accept inward impulses only. This first-order
constraint linearization is not continuous collision detection for fast swings.

CPU tree and compliant external-ball routes consume these rows. Rigid external
ball constraints are available on both CPU and GPU through
`RigidScene::insert_rigid_loop_joint`; their three translational anchor rows share
the cone solve without merging endpoint trees. CPU and GPU runtime paths also
consume compliant external ball/cone rows. Current coupling, control and
capacity restrictions are recorded in `MOLLA_PRIMITIVES_REPORT.md`; focused
component or runtime checks are not full contact/energy/performance certification.
`cone_reading` reports final swing and angular violation, whether a row was active
in the frame, and frame-average cone torque on the child. It is invalidated by
scene edits and failed steps; skipped sleeping joints have no fresh reading.
This is the cone contribution, not P3's complete joint wrench reconstruction.

`ball_anchor_reading` reports a rigid external ball's final separation and
frame-average force on its child anchor in world coordinates. Its reference
point is the anchor, not the COM. The corresponding pure anchor force acts
oppositely at the parent anchor; endpoint COM torques are the anchor arms crossed
with those forces. Cone torque is reported independently, so the two contributions
are not counted twice. Current rigid-ball motor, brake, limit and speed-cap
routes are inventoried in `MOLLA_PRIMITIVES_REPORT.md`. Compliance-mode edits
preserve physical pose and velocity; unsupported authored controls reject a
rigid-mode transition atomically.

Active input geometry is checked before force callbacks and actuator updates.
Invalid geometry fails without committing state. GPU substeps build their
geometry on device; eligible tree ball joints read the relative quaternion
directly from generalized coordinates, rather than subtracting world rotations.
No CPU constraint solve or per-substep readback is used by the GPU route. The raw
dense `Model`/`FeatherstoneSolver` API does not consume runtime cone settings.

The GPU predictor retains each generalized-velocity increment separately from
the rounded predicted velocity. The coupled constraint solve projects these
increments, combines them with the impulse correction, then adds the net change
to the input velocity. Caps and locked-coordinate overrides retain their actual
velocity difference. This avoids recovering a small force increment by
subtracting two already-rounded velocities. A single device scratch buffer is
reused across substeps; re-integration does not overwrite its predictor values.

## Singularities and unfinished work

The disk logarithm has no unique azimuth at the antipode. CPU/GPU geometry use a
deterministic one-sided limiting chart when `c < 0` and `s <= 1e-6`: choose the
widest transverse extent, treating disabled as pi and selecting the first axis
on ties. A finite maximum below pi supplies residual `theta-extent` and an
inward radial row. Its angular vector is projected orthogonal to the child
primary direction, preserving free child twist. The other zero extent uses
`asin(dot(n,b_i))` and its exact transverse-plane Jacobian; two zero extents
retain two bilateral rows. No unique smooth logarithm derivative is claimed.

An admissible antipode gets no artificial radial stop. Outside that immediate
chart, a nearer admissible-pole return branch uses error `pi-theta` and angular
derivative `-p cross (t/s)`, replacing the longer radial recovery. This changes
the recovery path and reported violation branch, not the admitted orientation
set. General moving/floating/rotated-frame transitions, outgoing velocities,
impact/energy behavior and sustained wide-cone acceptance remain required by
the original plan. The short checks in the report do not certify all of them.

GPU compilation must reject coefficients that cannot retain their enabled state
in f32; it must not turn a tiny positive extent into a zero lock or disabled row.
The CPU and GPU evaluators use the same basis, thresholds, row order and units.
The standalone geometry kernel is not evidence of runtime physical acceptance.
The initial runtime path still requires every remaining physical, mutation,
singularity, reaction, contact and resource gate in the plan.
