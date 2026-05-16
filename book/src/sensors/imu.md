# IMU

`ImuSensor` produces six-DoF inertial readings for a batch of
sensor mounts: linear acceleration (specific force) plus angular
velocity, both expressed in the sensor's local frame. It mirrors
Newton's `SensorImu` accelerometer + gyroscope combo and is the
primitive that locomotion / balance policies build their
proprioceptive observation vector on.

Source: `crates/molla/src/sensors/imu.rs`. Shader:
`crates/molla-kernels/shaders/imu.wgsl`.

## Reading + mount types

```rust
pub struct ImuMount {
    pub body: BodyId,
    pub local_pose: Transform,   // sensor pose in body frame
}

pub struct ImuReading {
    pub accelerometer: Vec3,     // specific force, sensor frame [m/s²]
    pub gyroscope: Vec3,         // angular velocity, sensor frame [rad/s]
}
```

The `local_pose` is the sensor's pose in the body's frame:
`position` is the mount point, `rotation` rotates the sensor's local
axes (`+X` forward, `+Y` up, etc. — whatever convention your IMU
chip uses) into the body's frame. The kernel pushes both
accelerations and angular velocities into the sensor frame before
returning them, so the policy sees the same numbers the embedded
controller would see on the real robot.

## Specific force semantics

The accelerometer outputs **specific force**, not coordinate
acceleration. At rest on the ground, the IMU reads `+g` along
whichever local axis points upward through the sensor (gravity is
the reaction force the table applies to keep the chip stationary,
not the literal acceleration of the chip in the world frame). A
body in free fall reads ≈ 0. This matches every real-world IMU
datasheet and Newton's convention.

## Acceleration via finite difference

Newton reads `body_qdd` (linear + angular accelerations) directly
from the solver. molla doesn't carry that field as part of `State`,
so `ImuSensor::update` finite-differences successive `body_qd`
snapshots:

```text
a_lin = (qd.linear  - qd_prev.linear ) / dt
a_ang = (qd.angular - qd_prev.angular) / dt
```

The sensor owns a per-body shadow `prev_body_qd` buffer that gets
rolled forward after each successful dispatch. On the very first
`update()` call the sensor is unprimed: it just snapshots the
current `body_qd` into `prev_body_qd` and returns an empty `Vec`
(the alternative would be a nonsense impulse from comparing against
zeros). Discard the first frame's reading on real captures, or call
`reset()` after a teleport / scene reset to force re-priming on the
next step.

The kernel then transforms the linear acceleration into the sensor
frame, **adds the centripetal + tangential terms** that come from
the IMU not sitting at the body's COM (`a = a_com + α × r + ω × (ω
× r)` in body frame, where `r` is the sensor offset), then
subtracts gravity to produce the specific-force output.

## API

```rust
use molla::compute::Engine;
use molla::sensors::{ImuMount, ImuSensor};
use molla::math::{Transform, Vec3};

let engine = Engine::new_gpu()?;
let mounts = vec![
    ImuMount {
        body: chassis_id,
        local_pose: Transform::new(
            Vec3::new(0.0, 0.05, 0.0),  // 5 cm above the chassis origin
            Quat::IDENTITY,
        ),
    },
];
let mut imu = ImuSensor::new(&engine, &model, &mounts)?;

// First call returns Vec::new() (priming snapshot).
let _ = imu.update(&engine, &model, &state, dt)?;

// Subsequent calls return one reading per mount.
for step in 0..n_steps {
    integrator.step(&engine, &model, &mut state, &control, dt)?;
    let readings = imu.update(&engine, &model, &state, dt)?;
    for (i, r) in readings.iter().enumerate() {
        println!("imu[{i}] a={:?} ω={:?}", r.accelerometer, r.gyroscope);
    }
}
```

`ImuSensor::new` validates every mount's body index against
`model.body_count` and rejects empty mount lists. The number of
sensors is fixed at construction time; if a scene needs more, build
a separate `ImuSensor` (each owns its own GPU buffers and dispatch
state).

## Reset semantics

`ImuSensor::reset()` clears the primed flag. The next `update()`
call will re-snapshot `body_qd` into `prev_body_qd` and return an
empty vec, identical to the very first call. Use this after:

- Scene re-initialisation / episode boundary.
- Teleporting a body by writing directly into `state.body_q`.
- Any operation that breaks the finite-difference assumption that
  the velocity changed smoothly between the previous and current
  step.

## Cross-references

- [`Sensors / Frame transform`](./frame_transform.md) — pose
  tracking instead of inertial; complementary observation channel.
- [`Sensors / Contact sensor`](./contact_sensor.md) — for the
  "robot is on the ground" signal that complements an IMU's
  accelerometer reading.
- [`Solvers / Semi-implicit integrator`](../solvers/semi_implicit.md)
  — supplies `state.body_qd` each step.
