# Frame transform sensor

`FrameTransformSensor` tracks the world pose of an arbitrary set of
*frames* (a frame = a body + a local pose) and — optionally — every
frame's pose **expressed in a reference frame**. It mirrors Newton's
`SensorFrameTransform` and is the primitive for end-effector
tracking, chase-cam targets, and trajectory recording.

Source: `crates/molla-sensors/src/frame_transform.rs`. Shader:
`crates/molla-kernels/shaders/frame_transform.wgsl` (one thread per target).

## Targets + reading

```rust
pub struct FrameTarget {
    pub body: BodyId,
    pub local_pose: Transform,
}

pub struct FrameReading {
    pub world: Transform,    // target's world pose
    pub relative: Transform, // target's pose expressed in reference frame
}
```

`local_pose` defines where on the body the frame sits and how it is
oriented (the "tool tip", "camera bracket", or "left-foot heel" of
the body). The world pose is the body's `state.body_q` pose
composed with `local_pose`; the relative pose is the world pose
pre-multiplied by the inverse of the reference frame's world pose.

If the sensor was built without a reference (`reference = None` at
construction), `relative == world` for every target — i.e. the
sensor degenerates to a pure forward-kinematics readout.

## API

```rust
use molla::compute::Engine;
use molla::sensors::{FrameTarget, FrameTransformSensor};
use molla::math::{Transform, Vec3, Quat};

let engine = Engine::new_gpu()?;

let base_ref = FrameTarget {
    body: base_id,
    local_pose: Transform::IDENTITY,
};
let tool_tip = FrameTarget {
    body: last_link_id,
    local_pose: Transform::new(Vec3::new(0.0, 0.0, 0.12), Quat::IDENTITY),
};

let mut sensor = FrameTransformSensor::new(
    &engine,
    &model,
    Some(base_ref),
    &[tool_tip],
)?;

let r = sensor.update(&engine, &state)?;
println!("tool world  = {:?}", r[0].world.position);
println!("tool in base= {:?}", r[0].relative.position);
```

When a reference is supplied, it occupies index 0 of the sensor's
internal target array. Each invocation independently computes the reference
world pose; there is no dependency on another invocation finishing first.
The returned `Vec` only
contains the caller-supplied targets; the reference is an
implementation slot.

All retained target/configuration and output arrays, including the optional
reference slot, share a default 1 GiB host/device payload cap.
`new_with_limits(engine, model, reference, targets, SensorArrayLimits)` overrides
it; `storage_usage()` reports the checked estimate. The caller-owned body-pose
buffer is checked against device limits but is not included in owned storage.
Collection and leased output pools have separate budgets. See
[fixed-array storage budgets](imu.md#fixed-array-storage-budgets).

### Device-resident outputs

`encode` can also write directly into protected output slots and compose multiple
producers in one submission. See [direct leased outputs](device_results.md) for
pool configuration, input ownership, active ranges and pending-result lifetimes.

`dispatch(engine, state)` returns `FrameDeviceReadings` without collection
or CPU waits. Its `world_position`, `world_rotation`, `relative_position`
and `relative_rotation` channels are borrowed `DeviceBuffer`s. Always use
`target_range`: `0..N` without a reference, `1..N+1` with one. The underlying
buffers include the internal reference slot when configured.

Submit GPU consumers before redispatch, or enqueue selected channels/ranges
into an external [`ReadbackRing`](readback.md). Unsubmitted command buffers
do not prevent output reuse. `update()` uses one reusable staging slot and
one bounded grouped collection of exactly 64 bytes per requested target,
excluding the internal reference. `readback_stats()` records this internal
collection only.

State poses must be current on the same engine, including GPU-authoritative
states whose host mirror is stale. Local mount poses must be finite rigid
transforms with unit quaternions; the caller must also supply valid rigid
body poses. Recreate after replacing/remapping bodies; count changes are
rejected but same-sized index remapping is not detectable through raw State.
There is no implicit scheduling or stable-handle remapping yet.

`dispatch_stats()` exposes submissions, kernel dispatches and binding
preparations. Two least-recently-used bound-buffer variants are retained;
alternating two physics pose buffers needs no new uniforms/bind groups after
warm-up. In-place pose changes reuse bindings; buffer replacement prepares a
new set. Each preparation creates one uniform and three bind groups. These
counters exclude construction, command encoders, readback and driver staging.

`FrameTransformKernel::prepare` / `encode_prepared` allow lower-level shared
command streams. The caller validates indices/ranges, tracks output authority
and submits before rewriting a prepared parameter buffer. Use distinct bindings
for differing parameters in one pending batch. The public sensor `encode` path
composes these operations into directly leased output slots.

If you want both an absolute readout and a chain-relative readout
of the same frame, register it twice: once with `None` reference (or
with a different reference), once with the chain root.

## Use cases

- **End-effector tracking** for IK / manipulation: register the tool
  tip with the robot base as reference. The `relative.position` is
  what an IK solver targets; the `world.position` is what the viewer
  draws.
- **Chase-camera targets**: register the head bone with the world
  root as reference. The camera follows `world.position`.
- **Per-trajectory logging**: register every relevant body with a
  fixed reference (e.g. the scene root) and snapshot the readings
  each step — directly produces the per-frame transform tensor a
  motion-capture replay or RL reward function consumes.

## Cross-references

- [`Sim / Joints`](../sim/joints.md) — the body poses returned by
  this sensor are the composition of `state.body_q` (FK output) with
  the user-supplied `local_pose`.
- [`Sim / Inverse kinematics`](../sim/ik.md) — typical consumer of
  the `relative` channel.
- [`Sensors / IMU`](./imu.md) — complementary observation channel.
