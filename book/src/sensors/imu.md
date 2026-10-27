# IMU

`encode` supports [direct leased outputs](device_results.md) and mixed sensor
command batches. Priming is explicit, and history advances only after submission;
discarded/poisoned batches leave the previous sample unchanged. Model/state
inputs require canonical device ownership. Existing `dispatch`/`update` remain
available and share the same history with the direct path.

`ImuSensor` produces six-DoF inertial readings for a batch of
sensor mounts: linear acceleration (specific force) plus angular
velocity, both expressed in the sensor's local frame. It mirrors
Newton's `SensorImu` accelerometer + gyroscope combo and is the
primitive that locomotion / balance policies build their
proprioceptive observation vector on.

Source: `crates/molla-sensors/src/imu.rs`. Shader:
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

Construction rejects missing body indices, non-finite positions, positions that
overflow f32, and non-finite or non-unit quaternions before allocating mount
buffers. Quaternion squared length must differ from one by at most `1e-6`;
invalid rotations are not silently normalized.

## Fixed-array storage budgets

`ImuSensor`, `ContactSensor` and `FrameTransformSensor` each default to a
1 GiB combined retained host/device array-payload cap. `new_with_limits` accepts
`SensorArrayLimits { max_storage_bytes }` as its last argument; existing `new`
calls retain their signatures. All constructors validate the combined budget,
individual buffer/binding limits and dispatch dimensions before array allocation.

Pure `SensorArrayStorage` estimates are available before creating a GPU engine:

- `for_imu(mounts, bodies)` includes mount body indices/local poses, both output
  channels and previous-velocity history for **every model body**, not only mounts.
- `for_contact(bodies)` includes four per-body output arrays. Zero bodies have
  zero payload and can use a zero-byte cap.
- `for_frames(targets, has_reference)` includes target body indices/local poses
  and all four outputs, with one additional slot when a reference is present.

Each sensor's `storage_usage()` reports host, device, total and largest-device-buffer
payload bytes. These estimates exclude caller-owned model/state/contact inputs,
uniforms/bindings, pipeline metadata, transient packing, driver/upload staging,
readback/host collection and leased output pools. They are not process peak-memory
caps. Inputs still must satisfy the device limits when bound; frame transforms
check the full body-pose buffer even though they do not own or budget its storage.
Empty IMU mount and frame target lists remain invalid.

Source: `crates/molla-sensors/src/sensor_storage.rs`. Host tests exercise extreme
counts without allocating those arrays; GPU tests compare estimates with retained
host lengths and actual device-buffer sizes, including empty contacts and optional
frame references. Run `make test-sensors` for both lanes.

## Specific force semantics

The accelerometer outputs **specific force**, not coordinate
acceleration. At rest on the ground, the IMU reads `+g` along
whichever local axis points upward through the sensor: the kernel subtracts
gravitational acceleration from coordinate acceleration. At rest, coordinate
acceleration is zero; in free fall it equals gravity, so the IMU reads ≈ 0.

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

History is copied device-to-device after evaluation in the same command
stream. A GPU solver's velocity buffer does not need to be downloaded first.
CPU edits must be uploaded before sampling; stale device input is rejected.
The public `update` call collects its two output channels together through one
reusable staging slot. It remains synchronous, with an explicit 30-second wait
limit. `dispatch` instead returns borrowed device readings without a host copy
or wait; the first priming call returns `None`.
`dt` is elapsed simulation time between samples, in seconds, and must convert
to a positive normal f32; zero, negative, non-finite and underflowing values
are rejected without advancing history.

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

Replacing a model's GPU storage requires a reset even when body counts match.
A changed body count requires recreating the sensor and validating its mounts.
In-place topology/index changes also require explicit reset or recreation;
the sensor does not infer application-level scene revisions from buffer contents.

## Device-only and asynchronous use

Use `dispatch` for GPU consumers or externally scheduled collection. A returned
`ImuDeviceReadings` exposes typed accelerometer/gyroscope `DeviceBuffer`s.
Submit consumers on the same engine queue before dispatching the next sample;
the borrowed view prevents concurrent mutable access to the sensor in Rust.
`readback_source(range)` selects active elements for a [readback ring](./readback.md).

`readback_stats()` measures only the sensor's internal synchronous collector.
Pure `dispatch` does not allocate staging or submit readback; an external ring
tracks its own collection statistics. Reset external rings separately when
invalidating queued scene results, and attach scene/sample identity to requests.

`dispatch_stats()` counts submissions, evaluations, history copies and binding
preparations. Priming submits a history copy without evaluation. Subsequent
samples encode evaluation and the following history copy into one submission.
Two least-recently-used bound-buffer variants are retained, supporting physics
ping-pong without new bindings after warm-up. In-place state or `dt` changes
reuse resources; replacing a bound buffer prepares a new set. Reset invalidates
history, not reusable bindings or lifetime counters. Each preparation creates
one uniform buffer and four bind groups; construction, command encoders, driver
staging and readback allocations are outside this counter.

`ImuKernel::prepare` / `encode_prepared` support lower-level command composition.
The caller owns history sequencing, range/index validity and output authority.
Submit before updating a reused parameter buffer; different `dt` values in one
pending batch need separate prepared bindings. The public `encode` path manages
leased output bindings and commits history only on successful batch submission.

Viewer example 06 uses this path with bounded asynchronous collection, simulation
timestamps and reset invalidation. It does not sample repeated paused states;
after backpressure its finite-difference interval spans the time since the last
submitted snapshot. See [viewer integration and acceptance status](viewer.md).

## Cross-references

- [`Sensors / Frame transform`](./frame_transform.md) — pose
  tracking instead of inertial; complementary observation channel.
- [`Sensors / Contact sensor`](./contact_sensor.md) — for the
  "robot is on the ground" signal that complements an IMU's
  accelerometer reading.
- [`Solvers / Semi-implicit integrator`](../solvers/semi_implicit.md)
  — supplies `state.body_qd` each step.
