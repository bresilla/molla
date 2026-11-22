# Reusable multirotor motion

The aircraft is one six-degree-of-freedom rigid body. Four motor-driven rotors
apply thrust, lever-arm moments and opposite shaft torques. Its mesh follows
the integrated body pose; neither position nor bank angle is animated.

The current visual mesh is a textured **DJI Mavic 3** model by llirikslon,
reduced/converted by Lexik and Krutnet under CC BY 4.0. Source credits, license,
checksum and modifications are in [MAVIC_ATTRIBUTION.md](data/drone/MAVIC_ATTRIBUTION.md).
It replaces the rejected procedural model. It is fitted to the existing simulated
rotor positions/radii; it does not change the 5.6 kg orchard flight model into a
calibrated or true-scale Mavic. Smooth normals and filtered texture mipmaps are
rendered by wgpu; the source remains 12,701 triangles with separate propellers.

## Flight references

```rust,ignore
use molla_solvers::airflow::{FlightSegment, MotionLimits, Quadrotor, QuadrotorConfig};
use molla_math::Vec3;

let start = Vec3::new(0.0, 3.0, 0.0);
let mut drone = Quadrotor::new(start, QuadrotorConfig::default())?;
let leg = FlightSegment::new(start, start + Vec3::X * 4.0, MotionLimits {
    speed: 1.5, acceleration: 2.0, jerk: 5.0,
})?;
// Call at each physics tick, with elapsed time since the leg began.
let reference = leg.sample(elapsed)?;
drone.track(reference, Vec3::Z)?;
drone.step(1.0 / 120.0, wind)?;
```

All references are world-space SI units. The wrapper uses world Y up; local X
is nose direction and local Z the rotor shaft. `heading` sets the nose direction
independently of travel, so X motion while facing Z is a genuine lateral slide.
`sample` holds the endpoints outside the segment interval. Every segment starts
and ends at rest; it is not an online splice for an already-moving vehicle.
`with_duration` may slow a segment but rejects speeding it past its bounds.
Arbitrary moving references can be supplied through `FlightReference` directly.

The controller combines position feedback, velocity feedback and reference
acceleration, then requests a thrust direction and collective lift. It tilts
toward acceleration, approaches level at constant speed in still air, and leans
oppositely to brake before the endpoint. Steady wind/drag can require a nonzero
cruise/hover tilt. Translation control prioritizes thrust-axis alignment over
yaw. The allocator preserves collective, then roll/pitch, then yaw within rotor
speed limits. Yaw braking uses the available rotor torque and yaw inertia to
reduce the requested turn rate before reaching the target heading.
The disturbance observer uses actual previous averaged rotor loads
and the previous step duration, not requested thrust. Motors retain finite slew.

`QuadrotorConfig::flight_limits` exposes controller gains and flight bounds;
mass, arm length, disk radius, motor acceleration and thrust/torque coefficients
are also configurable. Generic `devices::Copters` supports other rotor layouts.
Keep reference speeds/accelerations within the aircraft's controller and thrust
limits; an infeasible reference saturates rather than teleporting the vehicle.

## Air coupling

`drone.air_loads()` returns the opposite measured rotor forces and shaft torques
at the real rotor frames. Deposit those into `AirForces`, and sample the evolving
air at the drone for hull drag. Acceleration toward +X pushes air toward -X;
braking reverses that horizontal component. The orchard already performs this
coupling. The isolated flight laboratory records the wrench but does not run a
fluid grid. Propeller graphics use integrated shaft phase and opposite handedness.

## Run and verify

```sh
make test-rotorcraft
make test-rotorcraft-gpu
make test-drone-orchard
make capture-drone-flight ORCHARD_VIDEO=artifacts/drone-flight/flight.mp4
make verify-drone-flight ORCHARD_VIDEO=artifacts/drone-flight/flight.mp4
make run-drone-orchard
```

The 30-second laboratory shows lateral motion, forward motion, a diagonal return
with yaw, ascent and gust rejection. Its sidecar JSON records position, velocity,
orientation, rotor speeds, air reaction and tracking/settling acceptance. The
orchard keeps its 78-second, 16-tree route and the existing leaf/wood parameters.

## Scope

This is not a calibrated aircraft digital twin, full blade-element aerodynamics,
PX4 SITL or a hardware flight controller. Current inertia/drag and thrust constants
are illustrative, the collision hull is simplified, and sensor noise, battery/ESC
electrical dynamics, rotor gyroscopic inertia and rotor inflow identification are
not modeled. Rate-limited motors are not an identified ESC transfer function.
Trajectory bounds constrain the reference, not guaranteed tracking error or
obstacle clearance. Mesh details do not add separate mass/colliders. For hardware
matching, identify the aircraft parameters and compare measured maneuvers first.

Sources: [PX4 control architecture](https://docs.px4.io/main/en/flight_stack/controller_diagrams),
[PX4 jerk-limited trajectories](https://docs.px4.io/main/en/config_mc/mc_jerk_limited_type_trajectory),
[Lee, Leok and McClamroch](https://arxiv.org/abs/1003.2005).
Attribution and implementation boundaries are in `../../ACKNOWLEDGEMENT.md`, [36].
