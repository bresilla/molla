# Frame transform sensor

`FrameTransformSensor` tracks the world pose of an arbitrary set of
*frames* (a frame = a body + a local pose) and — optionally — every
frame's pose **expressed in a reference frame**. It mirrors Newton's
`SensorFrameTransform` and is the primitive for end-effector
tracking, chase-cam targets, and trajectory recording.

Source: `crates/molla/src/sensors/frame_transform.rs`. Shader:
`crates/molla/shaders/frame_transform.wgsl` (one thread per target).

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
internal target array — the kernel computes its world pose first so
later threads can chain through its inverse. The returned `Vec` only
contains the caller-supplied targets; the reference is an
implementation slot.

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
