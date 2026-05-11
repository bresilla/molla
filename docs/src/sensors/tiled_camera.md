# TiledCamera

`TiledCamera` is molla's batched, multi-channel pinhole camera
sensor. It mirrors Newton's `SensorTiledCamera`: a single
`dispatch()` runs a two-pass kernel for every camera in the batch
and writes per-pixel `(depth, color, normal, albedo, shape_index)`
into device buffers of shape `(num_cameras, height, width)`.

It's the workhorse for pixel-observation RL environments, where you
typically have one camera per parallel world and want all of them
rendered in lock-step so the policy sees a tensor of shape
`(num_envs, H, W, channels)` each step.

Source: `crates/molla/src/sensors/tiled_camera.rs`. Shaders:
`crates/molla/shaders/camera_rays.wgsl` (ray generation) and
`crates/molla/shaders/camera_trace.wgsl` (closest-hit + shading).

## Pipeline

Each `dispatch()` runs two kernels back-to-back:

1. **Ray generation** (`CameraRaysKernel`). Per-pixel `(camera, x, y)`
   triples are turned into world-space `(ray_origin, ray_direction)`
   pairs using the per-camera world pose + shared intrinsics
   (vertical FOV, aspect ratio). One thread per pixel.

2. **Closest-hit trace** (`CameraTraceKernel`). One thread per pixel
   walks every primitive shape, keeps the nearest hit within
   `max_distance`, and writes whichever channels the caller asked for
   via [`CameraChannels`](#channels). Hits beyond `max_distance` are
   reported as miss.

Both kernels share Model + body-pose buffers with the rest of the
engine — no data round-trips to the host between passes, no
per-frame uploads beyond the per-camera pose array.

## Configuration

```rust
pub struct TiledCameraConfig {
    pub width: u32,
    pub height: u32,
    pub fov_y_radians: f32,
    pub channels: CameraChannels,
    pub ambient: Vec3,           // ambient RGB added to every shaded pixel
    pub light_dir: Vec3,         // world-space directional light vector
    pub light_color: Vec3,       // directional-light colour (× n·-L)
    pub max_distance: f32,
    pub near: f32,               // informational, not enforced in v1
}
```

`Default` produces a 128×96 pinhole at 60° vertical FOV with a
warm-white directional light and a soft blue ambient — useful for a
viewer thumbnail without wiring custom lighting.

Width / height / camera count are fixed at construction time because
the output buffers are sized then. Everything else can be flipped at
any frame via `set_config()` — including the channels mask, so you
can render only depth on most frames and turn on the full RGBA +
normal stack only for the frames you log.

## Channels

`CameraChannels` is a bitmask of which output buffers to write:

```rust
impl CameraChannels {
    pub const COLOR: Self        = Self(1 << 0);
    pub const DEPTH: Self        = Self(1 << 1);
    pub const NORMAL: Self       = Self(1 << 2);
    pub const ALBEDO: Self       = Self(1 << 3);
    pub const SHAPE_INDEX: Self  = Self(1 << 4);
    pub const ALL: Self          = ...; // all five OR'd together
    pub const NONE: Self         = Self(0);
}
```

The semantics:

- **COLOR** — Lambert-shaded RGB packed into a `u32` (8 bits per
  channel). Cheap, suitable for visualisation thumbnails.
- **DEPTH** — `Real` (`f64`) ray-parametric `t` per pixel. `-1.0`
  marks a miss. This is the channel the depth-loss / pose-recovery
  loops consume.
- **NORMAL** — world-space outward normal per pixel as `Vec3`. Zero
  vector on miss. Useful for inverse-rendering or as a feature
  channel for the policy.
- **ALBEDO** — unshaded per-shape colour (no lighting). Useful for
  rendering pipelines that want to apply post-process lighting
  themselves.
- **SHAPE_INDEX** — `Model::shape_geometry` index of the hit shape
  (`u32::MAX` for miss). This is the instance-segmentation channel:
  every pixel tells you exactly which body-owned shape lit it up.

Bits are combined with `|`:

```rust
let cfg = TiledCameraConfig {
    channels: CameraChannels::DEPTH | CameraChannels::SHAPE_INDEX,
    ..Default::default()
};
```

Channels not in the mask are skipped at write time inside the kernel
(the backing buffers are always allocated, so the mask can be
flipped per-frame without reallocation).

## API

```rust
use molla::compute::Engine;
use molla::sensors::{TiledCamera, TiledCameraConfig, CameraChannels};
use molla::math::Vec3;

let engine = Engine::new_gpu()?;
let cfg = TiledCameraConfig {
    width: 256,
    height: 192,
    channels: CameraChannels::COLOR | CameraChannels::DEPTH,
    ..Default::default()
};
let mut cam = TiledCamera::new(&engine, cfg, /* num_cameras = */ 16)?;

// Aim the cameras. The convenience helper builds a lookAt pose.
for i in 0..16 {
    let eye = Vec3::new(0.0, 1.5, 4.0);
    let tgt = Vec3::ZERO;
    cam.set_camera_look_at(i, eye, tgt, Vec3::Y)?;
}

cam.dispatch(&engine, &model, &state)?;
let depth = cam.download_depth(&engine)?;
let color = cam.download_color(&engine)?;
```

Per-camera poses use molla's `Transform` (`position + rotation`),
where rotation maps the canonical camera axes (`+X` right, `+Y` up,
`-Z` forward) into world. `set_camera_look_at(idx, eye, target,
up_hint)` builds the same pose without the caller having to compose
a quaternion by hand — it's the same matrix as the classical OpenGL
`lookAt`, just expressed via a rotation quaternion.

Output buffers are laid out image-major, row-major within an image.
The linear pixel index for `(cam, x, y)` is exposed via:

```rust
let idx = cam.pixel_index(cam_id, x, y);
let d   = depth[idx];
```

The `download_*` helpers wrap `Buffer::download` + `Buffer::host`
into a single call that returns a borrow on the per-pixel slice —
useful when the policy code wants to view the depth tensor without
copying.

## Tiled multi-camera × multi-world layout

In Newton's RL recipe, each parallel world owns its own camera and
all of them render together. molla handles that the same way: the
caller picks `num_cameras = num_worlds` at construction time, then
either calls `set_camera_pose(world_i, world_i_camera_pose)` each
frame, or — more often — pre-bakes the per-world camera offset into
the pose buffer once and updates only the world-root body each step.
The resulting depth / colour buffers are tensors of shape
`(num_worlds, H, W)` ready to feed straight into the policy.

## Scope (Stage 14)

Sphere · Plane · Box · Capsule · Cylinder · Cone are intersected on
the GPU. TriMesh / ConvexHull / Heightfield / SDF carry the `0xff`
"skip" sentinel in `Model::shape_geom_kind` — they're invisible to
the camera until the GPU mesh BVH integration lands (same blocker as
[`Raycaster`](./raycaster.md)).

## Cross-references

- [`Sensors / Raycaster`](./raycaster.md) — same intersection logic,
  but caller-supplied rays instead of on-device generation.
- [`Geometry / Gaussian splats`](../geometry/gaussian_splat.md) — the
  3DGS family of differentiable renderers covered in
  [`Differentiable Physics / Gaussian splat renderer`](../autograd/splat.md);
  unrelated raster path but the rays it consumes match
  `camera_rays.wgsl`'s output format.
- [`Architecture / Design rules`](../architecture/design_rules.md)
  for the GPU-first contract these kernels follow.
