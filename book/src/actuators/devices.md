# Rigid-world devices (Webots set)

`RigidWorld::devices` (`crates/molla-solvers/src/devices/`) runs the
actuators of the [Webots actuator guide](https://cyberbotics.com/doc/guide/actuators)
inside the rigid step. Hosts command the devices and read back what they
did; the physics stays in the core.

| Webots device | Molla | Reads back |
|---|---|---|
| RotationalMotor, LinearMotor | `devices.motors`: position, velocity or force control on a revolute or prismatic tree joint | position, velocity, commanded speed, applied force/torque |
| Brake | `devices.motors.set_brake`: joint damping | damping, brake force/torque |
| Propeller | `devices.propellers` | rotor speed, thrust, torque, speed of advance |
| Track (conveyor or tank) | `devices.belts`: collider surface velocity | travel, speed |
| Connector | `devices.connectors` | presence, latch, link, tensile and shear load |

## Step order

1. `Devices::before_step` (from `RigidWorld::step_filtered` and
   `step_with_forces`):
   1. connectors update presence, break overloaded or released links and
      make new ones;
   2. motors write their joint drives;
   3. belts set their colliders' surface velocity;
   4. propellers set their rotor speed.
2. Every substep adds propeller wrenches and records loop-joint reactions.
   Active propellers take the substep force path, like tracks.
3. After the step, motors and brakes read the solver's per-DOF efforts.

## Motors

`MotorController` follows Webots' motor P-controller:

```text
error = target − position
v     = P·error + I·∫error + D·d(error)/dt, clamped to ±speed
v     = v_prev + clamp(v − v_prev, ±acceleration·dt)
```

- The drive is a velocity motor with a very large gain. The stable-feedback
  backward-Euler form makes it reach `v` within a substep when the available
  force allows, like an ODE joint motor.
- Force control writes an `Effort` drive clamped to the available force.
- Soft position limits clip the target; the velocity command is clipped to
  the maximum velocity.
- A new motor holds its joint's current position.

`FeatherstoneSolver::joint_efforts()` returns every DOF's drive effort after
its cap and its brake effort, averaged over the step's substeps. This is the
motor's force feedback.

## Brakes

`Control::joint_damping` adds `−c·q̇` to a DOF. It is solved backward-Euler
together with an implicit drive on the same DOF. The total effort uses both
denominators; the brake takes `−c·q̇⁺` of the predicted rate and the drive
the rest. A stiff velocity motor and a hard brake therefore stay stable
together. `JointAxisSettings::brake` keeps the damping across rebuilds.

## Propellers

With rotor speed `ω` and speed of advance `V` (the centre of thrust's
velocity along the shaft):

```text
T = t1·|ω|·ω − t2·|ω|·V     thrust along the shaft, at the centre of thrust
Q = q1·|ω|·ω − q2·|ω|·V     clamped to the rotor motor's torque
```

The body takes the thrust and `−Q` about the shaft, as in Webots.

## Belts

`ColliderDesc::surface_velocity` gives a collider's surface a velocity in its
own frame. Penalty friction adds each side's tangential surface velocity to
the slip, so a box on a belt moving at `u` settles at `u`. This is exactly a
static floor seen from a frame moving at `−u`. A belt's motor has position or
velocity control of its travel. `RigidScene::set_collider_surface_velocity`
changes the speed without a rebuild.

## Connectors

- **Presence:** a compatible peer (same model; symmetric–symmetric or
  active–passive) within this connector's own tolerances:
  - origin distance;
  - anti-parallel +X axes;
  - the docking rotation about +X (`rotations` per turn).
- **Linking:** a latched connector links to a present peer at the next
  update, or later when it auto-locks. It needs `unilateral_lock` or a
  latched peer. Passive connectors never latch.
- **The link:** a compliant fixed loop joint. With snap its child frame is
  the peer's face turned onto the parent's at the nearest docking rotation,
  so the faces pull flush. Without snap it keeps the current pose.
- **Breaking:** `loop_forces` now returns each loop joint's reaction on its
  child. Tensile load is the pull along the parent face's −X, shear the rest.
  A link breaks when either exceeds the summed strength of the latched sides.
  Unlocking breaks it as the unilateral-unlock rules allow. A broken pair does
  not relink until the peer has left.

## Tests

```sh
make test APP_TARGET='-p molla-solvers --test devices'
```

Each test checks a device against an analytic answer:
- **Motors:**
  - position control moves at the speed limit, then settles at the target;
  - holding torque equals `m·g·L`;
  - force control gives `ω = τ·t/I`.
- **Brake:** decays the spin as `e^(−c·t/I)` and reports `−c·ω`.
- **Propeller:** thrust and reaction torque drive a free body at `T/m` and
  `−Q/I`.
- **Belt:** a crate rides at belt speed.
- **Connectors:**
  - snap flush and hold 30 N, reporting it as tensile load;
  - break at 70 N against a 50 N strength;
  - auto-lock a pod arriving at the dock, and release it on unlock.
