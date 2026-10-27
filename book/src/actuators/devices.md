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

## Copters

`devices.copters` flies a multirotor on its propellers. Each step, before the
propellers update, it reads its body's pose, velocity, mass and inertia and
sets every rotor's speed:

- **Velocity mode** (forward, left, up, yaw rate in the heading frame):
  - The velocity error times `velocity_gain`, plus gravity, minus the
    observed disturbance, is the specific force wanted.
  - Its horizontal part is capped by `max_tilt`.
  - Its direction is the desired body up. The heading keeps the current one.
- **Disturbance observer:**
  - Each step compares the acceleration the body had with the thrust the
    rotors delivered plus gravity.
  - It low-passes the difference over `disturbance_time`: wind, drag, payload
    or a rotor's lost thrust.
  - The velocity loop cancels the estimate, capped at 0.3 g along up and the
    tilt limit across. Setpoint changes never pass through it, so there is no
    overshoot.
- **Attitude mode:** roll (right side down), pitch (nose down), yaw rate and
  climb rate.
- **Attitude loop:** the geometric error `½·vee(R_dᵀR − RᵀR_d)` sets body
  rates, with a critically damped cascade at `attitude_frequency`. The yaw
  rate is fed straight in.
- **Rates to torque:** the rate error, through the body inertia plus the
  gyroscopic term.
- **Mixer:**
  - Each rotor's column is its thrust along up, and `t1·(r × a) − q1·a` torque,
    per unit `|ω|·ω`.
  - The least-norm solution `Mᵀ(MMᵀ)⁻¹·[T, τ]` gives every rotor's `|ω|·ω`.
  - It is clamped to non-reversing speeds within each rotor's limit; the
    reading flags the saturation.

Rotors spin opposite ways by the sign of their thrust constant. A rotor with
a negative `t1` runs at negative `ω` for positive thrust, and its reaction
torque reverses.

## Air

`devices.aero` holds the wind and the drag elements:

- **Wind:**
  - `set_wind` sets the wind everywhere.
  - `set_body_wind` gives one body its own, for example a gust the host
    sampled where the body is.
- **Drag:**
  - Each element is a frame on a body with `CdA` along its axes and an air
    density.
  - Its origin is the centre of pressure.
  - Every substep it adds `−½·ρ·CdA·|a|·a` per axis, where `a` is the
    airspeed of that point in the frame: body velocity minus wind.
  - Readings: airspeed, the step's mean force, and the wind.
- **Propellers:** they take their speed of advance from the airspeed. A
  headwind loads them as flying forward does.

## Tests

```sh
make test APP_TARGET='-p molla-solvers --test devices'
```

- **Quad:** a 25 kg X-layout quad hovers within 5 cm at the analytic rotor
  speed, with the diagonals counter-rotating. It climbs at 1 m/s, cruises at
  3 m/s nose down, and yaws at 0.5 rad/s.
- **Attitude mode:** it banks to 0.2 rad and slides right holding height.
  Switched off, it free-falls.
- **Wind:** in a steady 5 m/s wind the quad holds station within 1 m. It
  leans nose-up by `atan(½·ρ·CdA·w² / (m·g))`, and its drag reads that load.
- **Drag:** a free 1 kg body in a 3 m/s wind follows `v(t) = w − 1/(1/w + k·t)`,
  with `k = ½·ρ·CdA/m`.

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
