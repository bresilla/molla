# CABLE joint

`JointType::Cable` is molla's 2-DoF Newton-style rope joint, introduced
in Stage 39E. It models a short tendon / rope segment between two
rigid bodies with a **linear stretch DoF** along one axis and an
**isotropic bend DoF** about a perpendicular axis. The two combine
into a spring + damper torque pair that the Featherstone solver
integrates as a regular reduced-coordinate joint.

## DoF layout

```text
JointType::Cable = 7        // (2 coords, 2 dofs)

joint_q[coord_offset[j] + 0]  = q_l   (linear stretch, m)
joint_q[coord_offset[j] + 1]  = q_b   (bend angle, rad)

joint_qd[dof_offset[j] + 0]   = q̇_l  (stretch velocity, m/s)
joint_qd[dof_offset[j] + 1]   = q̇_b  (bend angular velocity, rad/s)
```

The cable's local transform is **translate-then-rotate** along the
joint-local frame:

```text
J(q) = T(q_l · ẑ) · R(q_b, x̂)
```

Convention:

- **Cable axis** = joint-local `+z` (the rope direction at rest).
- **Bend axis** = joint-local `+x` (the perpendicular bend axis).

The bend is isotropic (the choice of `+x` is deterministic but
arbitrary — any rotation about a fixed perpendicular axis captures
the curvature DoF in 1D).

## Motion subspace

Two columns:

```text
S_l = (z_world_x, z_world_y, z_world_z; 0, 0, 0)         (linear)
S_b = (0, 0, 0; x_world_x, x_world_y, x_world_z)         (angular)
```

`z_world` and `x_world` are the joint-local `+z` / `+x` expressed in
world axes, computed via the parent's world pose + the joint's
parent-frame transform.

## Spring + damper dynamics

The cable joint differs from the other joint types in that it carries
**built-in restoring forces** via per-joint static parameters stored on
the `Model`:

```rust,ignore
pub joint_cable_stiffness_linear: Buffer<Real>,   // k_l (N/m)
pub joint_cable_stiffness_bend:   Buffer<Real>,   // k_b (N·m/rad)
pub joint_cable_damping_linear:   Buffer<Real>,   // c_l (N·s/m)
pub joint_cable_damping_bend:     Buffer<Real>,   // c_b (N·m·s/rad)
```

The Featherstone backward pass folds an extra term into the per-DoF
generalised force:

```text
τ_l = − k_l · q_l   − c_l · q̇_l            (stretch)
τ_b = − k_b · q_b   − c_b · q̇_b            (bend)
```

Combined with the standard Featherstone PD drive (if a
`JointTargetMode` is active), feed-forward `joint_force`, and Coulomb
joint friction, the full effort vector at the DoF is

```text
u_k = τ_drive_k + τ_ff_k + τ_friction_k + τ_cable_k − c_damping · qd_k
```

then clamped by `joint_effort_limit`.

## API

```rust,ignore
pub fn add_joint_cable(
    &mut self,
    parent: BodyId, child: BodyId,
    x_p: Transform,
    x_c: Transform,
    stiffness_linear: Real,    // k_l (N/m)
    stiffness_bend:   Real,    // k_b (N·m/rad)
    damping_linear:   Real,    // c_l (N·s/m)
    damping_bend:     Real,    // c_b (N·m·s/rad)
) -> Result<JointId>;
```

All four parameters must be finite and `≥ 0` — `Error::Build` fires
otherwise.

```rust,ignore
let tendon = b.add_joint_cable(
    bone_a, bone_b,
    x_p, x_c,
    /* k_l */ 1.0e4,     // stiff axial spring
    /* k_b */ 0.5,       // soft bend
    /* c_l */ 50.0,
    /* c_b */ 0.05,
)?;
```

## Use cases

- **Tendons / muscles**. The Hill-type muscle actuator (Stage 42)
  attaches to a CABLE joint's **linear stretch DoF** via
  `b.add_muscle(joint, MuscleConfig{ … })`. The muscle's active
  contractile force `f = a · f_max · f_L · f_V` is added as a torque
  on `q_l`, pulling the cable shorter. See [muscle actuator](../actuators/muscle.md).
- **Rope segments connecting rigid pulleys**. Each CABLE joint is one
  pulley-to-pulley segment; the spring + damper handle rope elasticity.
- **Soft skeletal links** (e.g. cervical spine articulation in a
  biomechanical model). Multiple CABLE joints chained between
  vertebrae produce a smooth elastic curve.

## Comparison: CABLE joint vs cable polyline

molla ships **two** cable primitives that look superficially similar
but live at different layers:

|                 | CABLE joint (`JointType::Cable`) | Cable polyline (`ModelBuilder::add_cable`) |
|-----------------|----------------------------------|--------------------------------------------|
| Backing storage | Featherstone joint (2 DoFs)       | Chain of XPBD distance constraints between particles |
| Solver          | [Featherstone ABA](./featherstone.md) | [VBD](../solvers/vbd.md) or [XPBD](../solvers/xpbd.md) |
| Geometry        | One rigid segment (parent ↔ child) | Polyline of `N` particles |
| Elastic model   | Per-segment k + damper             | Per-segment XPBD compliance + bending hinge energy |
| Use case        | Robotics tendons, biomechanical muscles | Soft ropes, hair, fishing lines |
| Bending         | Isotropic, 1-DoF                   | Discrete hinge energy `E_b = ½·k_b·(2 sin θ/2)²` |
| Muscle attach   | Yes (`add_muscle`)                 | No (muscles drive a joint DoF) |

The **CABLE joint** is the rigid-body, reduced-coordinate primitive;
the **cable polyline** is the particle-based, position-projection primitive.
Pick the former for muscle-driven biomechanics; pick the latter for
visually-rich rope simulation. Stage 39E originated as part of the
muscle deliverable so the muscle force has a "joint DoF to drive";
the polyline primitive predates it (Stage 20.1) and remains the
canonical path for soft ropes.
