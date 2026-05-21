//! Multi-channel raytraced **camera sensor** running entirely on the
//! GPU. Mirrors Newton's `SensorTiledCamera`: per-camera pinhole
//! pose + intrinsics in, per-pixel `(depth, color, normal, albedo,
//! shape_index)` channels out, all sized
//! `(camera_count × width × height)`.
//!
//! Two-pass dispatch:
//! 1. [`CameraRaysKernel`] turns each `(camera, x, y)` triple into a
//!    world-space ray.
//! 2. [`CameraTraceKernel`] walks every primitive shape per pixel,
//!    keeps the closest hit, and writes whichever channels the
//!    caller asked for via [`CameraChannels`].
//!
//! **Scope (Stage 14)**: Sphere · Plane · Box · Capsule · Cylinder ·
//! Cone. TriMesh / ConvexHull / Heightfield / SDF are skipped on the
//! GPU (carry the `0xff` sentinel in `Model::shape_geom_kind`); they
//! land when the GPU mesh BVH lands.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{CameraRaysKernel, CameraTraceKernel};
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sim::{Model, State};
use molla_storage::{Buffer, BufferUsage};

/// Bitmask of which output channels [`TiledCamera::dispatch`] should
/// write each frame. Channels not in the mask are left untouched in
/// their backing buffers — useful when only depth or only the colour
/// thumbnail matters and you want to skip the per-pixel writes.
///
/// Values mirror the in-shader `CH_*` constants in
/// `shaders/camera_trace.wgsl`. Combine with the bitwise OR of the
/// associated `*_BIT` constants.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CameraChannels(pub u32);

impl CameraChannels {
    pub const COLOR: Self = Self(1 << 0);
    pub const DEPTH: Self = Self(1 << 1);
    pub const NORMAL: Self = Self(1 << 2);
    pub const ALBEDO: Self = Self(1 << 3);
    pub const SHAPE_INDEX: Self = Self(1 << 4);

    /// All five channels enabled. Convenient default for the viewer
    /// demo that wants to show every thumbnail.
    pub const ALL: Self =
        Self(Self::COLOR.0 | Self::DEPTH.0 | Self::NORMAL.0 | Self::ALBEDO.0 | Self::SHAPE_INDEX.0);

    /// Empty mask — useful as the initial value before bitwise-OR'ing
    /// the channels you want.
    pub const NONE: Self = Self(0);

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl core::ops::BitOr for CameraChannels {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for CameraChannels {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Static configuration for a [`TiledCamera`] batch. All cameras in a
/// batch share the same intrinsics + lighting; only the per-camera
/// pose differs.
#[derive(Clone, Copy, Debug)]
pub struct TiledCameraConfig {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Vertical field of view, in radians.
    pub fov_y_radians: f32,
    /// Channels the dispatcher should fill each frame.
    pub channels: CameraChannels,
    /// Ambient term added to every shaded pixel (RGB × albedo).
    pub ambient: Vec3,
    /// World-space directional light direction (light shines along
    /// this vector). Will be normalized internally if non-unit.
    pub light_dir: Vec3,
    /// Directional-light colour (RGB). Multiplied by `max(0, n·-L)`.
    pub light_color: Vec3,
    /// Maximum ray distance (m). Hits past this are reported as miss.
    pub max_distance: f32,
    /// Near plane — currently informational, not enforced. Reserved
    /// for the future projective-depth output mode.
    pub near: f32,
    /// H10 — physical optics. Defaults preserve the pinhole renderer
    /// (`lens_diameter == 0`); downstream sim-to-real pipelines can
    /// set these to match a real lens.
    pub optics: PhysicalOptics,
}

/// H10 — physical camera intrinsics carried alongside the projective
/// ones. The renderer is currently pinhole; these fields are stored
/// so the H11 post-process and any future depth-of-field renderer
/// can read them without breaking the existing API.
#[derive(Clone, Copy, Debug)]
pub struct PhysicalOptics {
    /// Physical focal length (mm). For a 50 mm "normal" lens on a 35
    /// mm full-frame sensor (width 36 mm), HFOV ≈ 39.6°.
    pub focal_length_mm: f32,
    /// Aperture entrance diameter (m). `0.0` (default) → pinhole.
    /// Positive values feed a future DOF renderer; the present
    /// rasterizer treats every primary ray as a pinhole.
    pub lens_diameter_m: f32,
    /// Plane the lens is focused on (m). Objects exactly at this
    /// distance render sharp; nearer/farther objects blur once the
    /// DOF renderer reads `lens_diameter_m > 0`.
    pub focal_distance_m: f32,
    /// Physical sensor width (mm). Combined with `focal_length_mm`
    /// gives HFOV as `2 · atan(sensor_width / (2 · focal_length))`.
    pub sensor_width_mm: f32,
    /// Number of aperture samples averaged per pixel when the DOF
    /// renderer is active. `1` (default) = single pinhole sample.
    pub aperture_samples: u32,
}

impl PhysicalOptics {
    /// Compute horizontal field of view (rad) from `focal_length_mm`
    /// and `sensor_width_mm`. Returns `None` when either is `≤ 0`.
    pub fn hfov_radians(&self) -> Option<f32> {
        if self.focal_length_mm > 0.0 && self.sensor_width_mm > 0.0 {
            Some(2.0 * (self.sensor_width_mm / (2.0 * self.focal_length_mm)).atan())
        } else {
            None
        }
    }
}

impl Default for PhysicalOptics {
    fn default() -> Self {
        // Pinhole defaults — the existing TiledCamera ray-gen kernel
        // ignores these because `lens_diameter_m == 0`.
        Self {
            focal_length_mm: 0.0,
            lens_diameter_m: 0.0,
            focal_distance_m: 0.0,
            sensor_width_mm: 0.0,
            aperture_samples: 1,
        }
    }
}

impl Default for TiledCameraConfig {
    fn default() -> Self {
        Self {
            width: 128,
            height: 96,
            fov_y_radians: 60f32.to_radians(),
            channels: CameraChannels::ALL,
            ambient: Vec3::new(0.15, 0.15, 0.20),
            light_dir: Vec3::new(-0.4, -0.8, -0.45).normalize(),
            light_color: Vec3::new(0.9, 0.9, 0.85),
            max_distance: 100.0,
            near: 0.05,
            optics: PhysicalOptics::default(),
        }
    }
}

/// A batch of pinhole cameras with shared intrinsics. Each camera has
/// its own world-space pose (set via [`Self::set_camera_pose`]); a single
/// [`Self::dispatch`] runs the two-pass kernel for all cameras at once.
///
/// Output buffers are sized `width * height * num_cameras`, indexed
/// row-major within an image, image-major across the batch
/// (`idx = cam * (W*H) + y * W + x`).
pub struct TiledCamera {
    config: TiledCameraConfig,
    num_cameras: u32,
    rays_kernel: CameraRaysKernel,
    trace_kernel: CameraTraceKernel,

    /// Per-camera world pose. Cheap to push to the device each
    /// dispatch; the pose array is small (`num_cameras` entries).
    poses: Buffer<Transform>,

    /// Per-pixel scratch — one entry per `(cam, y, x)`.
    ray_origin: Buffer<Vec3>,
    ray_direction: Buffer<Vec3>,

    /// Output channel buffers. Always allocated regardless of the
    /// `channels` mask so callers can flip the mask between frames
    /// without reallocating.
    pub depth: Buffer<Real>,
    pub color: Buffer<u32>,
    pub normal: Buffer<Vec3>,
    pub albedo: Buffer<u32>,
    pub shape_index: Buffer<u32>,
}

impl TiledCamera {
    /// Build a new tiled-camera batch with `num_cameras` cameras,
    /// allocating all per-pixel + per-camera buffers up-front.
    /// Initial poses default to identity (camera at world origin
    /// looking at -Z) — call [`Self::set_camera_pose`] before the first
    /// [`Self::dispatch`] to point each camera at the scene.
    pub fn new(engine: &Engine, config: TiledCameraConfig, num_cameras: usize) -> Result<Self> {
        if num_cameras == 0 {
            return Err(Error::Solver(
                "TiledCamera::new: num_cameras must be >= 1".into(),
            ));
        }
        if config.width == 0 || config.height == 0 {
            return Err(Error::Solver(
                "TiledCamera::new: width/height must be >= 1".into(),
            ));
        }

        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("TiledCamera: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("TiledCamera: engine has no GPU queue".into()))?
            .clone();

        let n_pixels = (config.width as usize) * (config.height as usize) * num_cameras;
        let mut poses =
            Buffer::from_vec(vec![Transform::IDENTITY; num_cameras], BufferUsage::Storage);
        let mut ray_origin =
            Buffer::from_vec(vec![Vec3::ZERO; n_pixels], BufferUsage::StorageReadWrite);
        let mut ray_direction =
            Buffer::from_vec(vec![Vec3::ZERO; n_pixels], BufferUsage::StorageReadWrite);

        let mut depth = Buffer::from_vec(vec![-1.0_f64; n_pixels], BufferUsage::StorageReadWrite);
        let mut color = Buffer::from_vec(vec![0_u32; n_pixels], BufferUsage::StorageReadWrite);
        let mut normal =
            Buffer::from_vec(vec![Vec3::ZERO; n_pixels], BufferUsage::StorageReadWrite);
        let mut albedo = Buffer::from_vec(vec![0_u32; n_pixels], BufferUsage::StorageReadWrite);
        let mut shape_index = Buffer::from_vec(
            vec![0xffff_ffff_u32; n_pixels],
            BufferUsage::StorageReadWrite,
        );

        poses.ensure_device(&device)?;
        ray_origin.ensure_device(&device)?;
        ray_direction.ensure_device(&device)?;
        depth.ensure_device(&device)?;
        color.ensure_device(&device)?;
        normal.ensure_device(&device)?;
        albedo.ensure_device(&device)?;
        shape_index.ensure_device(&device)?;

        poses.upload(&queue)?;
        ray_origin.upload(&queue)?;
        ray_direction.upload(&queue)?;
        depth.upload(&queue)?;
        color.upload(&queue)?;
        normal.upload(&queue)?;
        albedo.upload(&queue)?;
        shape_index.upload(&queue)?;

        Ok(Self {
            config,
            num_cameras: num_cameras as u32,
            rays_kernel: CameraRaysKernel::new(&device),
            trace_kernel: CameraTraceKernel::new(&device),
            poses,
            ray_origin,
            ray_direction,
            depth,
            color,
            normal,
            albedo,
            shape_index,
        })
    }

    /// Number of cameras in the batch.
    pub fn num_cameras(&self) -> u32 {
        self.num_cameras
    }

    /// Camera config (intrinsics + lighting + channels).
    pub fn config(&self) -> &TiledCameraConfig {
        &self.config
    }

    /// Update the camera config in-place. Width / height / camera
    /// count cannot change without rebuilding the camera (output
    /// buffers are sized at construction time); this only modifies
    /// the per-dispatch parameters (FOV, channels, lighting,
    /// max_distance).
    pub fn set_config(&mut self, config: TiledCameraConfig) -> Result<()> {
        if config.width != self.config.width || config.height != self.config.height {
            return Err(Error::Solver(
                "TiledCamera::set_config: width/height cannot change post-construction".into(),
            ));
        }
        self.config = config;
        Ok(())
    }

    /// Set the world-space pose of camera `idx`. Call once per
    /// dispatch (or whenever the camera moves). `pose.position` is
    /// the pinhole; `pose.rotation` rotates camera-local axes
    /// (+X right, +Y up, -Z forward) into world space.
    pub fn set_camera_pose(&mut self, idx: usize, pose: Transform) -> Result<()> {
        if idx >= self.num_cameras as usize {
            return Err(Error::Solver(format!(
                "TiledCamera::set_camera_pose: idx {idx} out of range (n={})",
                self.num_cameras
            )));
        }
        let host = self.poses.host_mut()?;
        host[idx] = pose;
        Ok(())
    }

    /// Convenience helper — point camera `idx` from world position
    /// `eye` toward world point `target`, with `up_world` as the
    /// approximate up direction (defaults to world +Y if you pass
    /// `Vec3::Y`).
    pub fn set_camera_look_at(
        &mut self,
        idx: usize,
        eye: Vec3,
        target: Vec3,
        up_world: Vec3,
    ) -> Result<()> {
        let pose = look_at_pose(eye, target, up_world);
        self.set_camera_pose(idx, pose)
    }

    /// Run both kernels: ray generation then closest-hit trace.
    ///
    /// The model + state must already have their device mirrors
    /// uploaded (call `Model::upload_to_device` and
    /// `State::upload_to_device` once before the first call).
    ///
    /// **H10 — Physical optics.** When the `PhysicalOptics` fields on
    /// the config are non-default, this dispatch honours them:
    ///
    /// * `focal_length_mm` + `sensor_width_mm` (both > 0): the
    ///   effective vertical FOV is computed from
    ///   `2·atan(sensor_height / (2·focal_length))` rather than the
    ///   `fov_y_radians` field. `sensor_height` is derived from
    ///   `sensor_width_mm` and the image aspect ratio.
    /// * `lens_diameter_m > 0` **and** `aperture_samples > 1`:
    ///   K aperture-jittered passes run with concentric-disc samples
    ///   and the `color` buffer is averaged on the host. The other
    ///   channels (`depth` / `normal` / `albedo` / `shape_index`) are
    ///   kept from the centre sample so they stay geometrically clean.
    ///   `lens_diameter_m > 0` with `aperture_samples == 1` still
    ///   takes the DOF path but is a single jittered sample — useful
    ///   for cheap focus-distance previews.
    /// * Pinhole (the default) is bit-identical to the pre-H10 output.
    pub fn dispatch(&mut self, engine: &Engine, model: &Model, state: &State) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("TiledCamera::dispatch: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("TiledCamera::dispatch: no GPU queue".into()))?
            .clone();

        // Push the pose buffer if the user mutated it via set_camera_pose.
        self.poses.upload(&queue)?;

        let aspect = self.config.width as f32 / self.config.height as f32;
        let fov_y = effective_fov_y(&self.config, aspect);
        let lens_radius = self.config.optics.lens_diameter_m.max(0.0) * 0.5;
        let focal_distance = self.config.optics.focal_distance_m.max(0.0);
        let k_samples = self.config.optics.aperture_samples.max(1);

        let body_q = state.body_q.device_buffer().ok_or_else(|| {
            Error::Gpu("TiledCamera::dispatch: state.body_q not on device".into())
        })?;

        let light_dir = if self.config.light_dir.length_squared() > 1e-12 {
            self.config.light_dir.normalize()
        } else {
            Vec3::new(0.0, -1.0, 0.0)
        };

        // Single-dispatch fast path (pinhole, or single-sample
        // jittered DOF). Identical to the pre-H10 output when the
        // optics fields are defaults.
        if lens_radius <= 0.0 || k_samples == 1 {
            self.rays_kernel.dispatch(
                &device,
                &queue,
                &self.poses,
                &mut self.ray_origin,
                &mut self.ray_direction,
                self.config.width,
                self.config.height,
                self.num_cameras,
                fov_y,
                aspect,
                self.config.near,
                lens_radius,
                focal_distance,
                [0.0, 0.0],
            )?;
            self.trace_kernel.dispatch(
                &device,
                &queue,
                model,
                body_q,
                &self.ray_origin,
                &self.ray_direction,
                &mut self.depth,
                &mut self.color,
                &mut self.normal,
                &mut self.albedo,
                &mut self.shape_index,
                self.config.width,
                self.config.height,
                self.num_cameras,
                self.config.max_distance,
                self.config.channels.bits(),
                self.config.ambient,
                light_dir,
                self.config.light_color,
            )?;
            return Ok(());
        }

        // K-sample DOF: centre pass + (K - 1) concentric-disc samples;
        // colour is averaged on host. Other channels come from the
        // centre pass (averaging depth across samples that hit
        // different surfaces is geometrically wrong).
        let pixels = (self.config.width as usize)
            * (self.config.height as usize)
            * (self.num_cameras as usize);
        let mut accum: Vec<[f32; 4]> = vec![[0.0; 4]; pixels];

        for s in 0..k_samples {
            let offset = if s == 0 {
                [0.0, 0.0]
            } else {
                concentric_disc_sample(s, k_samples)
            };
            self.rays_kernel.dispatch(
                &device,
                &queue,
                &self.poses,
                &mut self.ray_origin,
                &mut self.ray_direction,
                self.config.width,
                self.config.height,
                self.num_cameras,
                fov_y,
                aspect,
                self.config.near,
                lens_radius,
                focal_distance,
                offset,
            )?;
            self.trace_kernel.dispatch(
                &device,
                &queue,
                model,
                body_q,
                &self.ray_origin,
                &self.ray_direction,
                &mut self.depth,
                &mut self.color,
                &mut self.normal,
                &mut self.albedo,
                &mut self.shape_index,
                self.config.width,
                self.config.height,
                self.num_cameras,
                self.config.max_distance,
                self.config.channels.bits(),
                self.config.ambient,
                light_dir,
                self.config.light_color,
            )?;
            // Accumulate colour. The other channels are left as-is;
            // after the loop the centre sample (s == 0) is the one
            // visible in `depth` / `normal` / `albedo` / `shape_index`
            // only because we restore those at the end of the loop.
            self.color.download(&device, &queue)?;
            let host = self.color.host()?;
            for (i, &packed) in host.iter().enumerate() {
                let r = (packed & 0xFF) as f32;
                let g = ((packed >> 8) & 0xFF) as f32;
                let b = ((packed >> 16) & 0xFF) as f32;
                let a = ((packed >> 24) & 0xFF) as f32;
                accum[i][0] += r;
                accum[i][1] += g;
                accum[i][2] += b;
                accum[i][3] += a;
            }
        }

        // Average colour, pack back to RGBA8 u32, push to device.
        let inv_k = 1.0 / k_samples as f32;
        {
            let host = self.color.host_mut()?;
            for (dst, src) in host.iter_mut().zip(accum.iter()) {
                let r = (src[0] * inv_k).clamp(0.0, 255.0).round() as u32;
                let g = (src[1] * inv_k).clamp(0.0, 255.0).round() as u32;
                let b = (src[2] * inv_k).clamp(0.0, 255.0).round() as u32;
                let a = (src[3] * inv_k).clamp(0.0, 255.0).round() as u32;
                *dst = r | (g << 8) | (b << 16) | (a << 24);
            }
        }
        self.color.upload(&queue)?;

        // Re-run the centre sample for the non-colour channels so
        // they end up geometrically clean (the loop above left them
        // at the *last* sample's values).
        self.rays_kernel.dispatch(
            &device,
            &queue,
            &self.poses,
            &mut self.ray_origin,
            &mut self.ray_direction,
            self.config.width,
            self.config.height,
            self.num_cameras,
            fov_y,
            aspect,
            self.config.near,
            lens_radius,
            focal_distance,
            [0.0, 0.0],
        )?;
        // Keep the averaged colour by routing the trace's colour
        // write into a throwaway buffer would require a buffer swap
        // — instead, snapshot the host colour, run trace, restore.
        let snapshot = self.color.host()?.to_vec();
        self.trace_kernel.dispatch(
            &device,
            &queue,
            model,
            body_q,
            &self.ray_origin,
            &self.ray_direction,
            &mut self.depth,
            &mut self.color,
            &mut self.normal,
            &mut self.albedo,
            &mut self.shape_index,
            self.config.width,
            self.config.height,
            self.num_cameras,
            self.config.max_distance,
            self.config.channels.bits(),
            self.config.ambient,
            light_dir,
            self.config.light_color,
        )?;
        // Restore the averaged colour.
        self.color.download(&device, &queue)?;
        {
            let host = self.color.host_mut()?;
            host.copy_from_slice(&snapshot);
        }
        self.color.upload(&queue)?;
        Ok(())
    }

    /// Download the depth channel into the host mirror and return a
    /// borrow of the per-pixel slice. Same semantics for the other
    /// `download_*` helpers below — they're convenience wrappers
    /// around `Buffer::download` + `Buffer::host`.
    pub fn download_depth(&mut self, engine: &Engine) -> Result<&[Real]> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("no queue".into()))?
            .clone();
        self.depth.download(&device, &queue)?;
        self.depth.host()
    }

    pub fn download_color(&mut self, engine: &Engine) -> Result<&[u32]> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("no queue".into()))?
            .clone();
        self.color.download(&device, &queue)?;
        self.color.host()
    }

    pub fn download_normal(&mut self, engine: &Engine) -> Result<&[Vec3]> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("no queue".into()))?
            .clone();
        self.normal.download(&device, &queue)?;
        self.normal.host()
    }

    pub fn download_albedo(&mut self, engine: &Engine) -> Result<&[u32]> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("no queue".into()))?
            .clone();
        self.albedo.download(&device, &queue)?;
        self.albedo.host()
    }

    pub fn download_shape_index(&mut self, engine: &Engine) -> Result<&[u32]> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("no queue".into()))?
            .clone();
        self.shape_index.download(&device, &queue)?;
        self.shape_index.host()
    }

    /// Linear pixel index for `(cam, x, y)`. Useful for indexing the
    /// downloaded channel slices.
    pub fn pixel_index(&self, cam: u32, x: u32, y: u32) -> usize {
        let w = self.config.width;
        let h = self.config.height;
        ((cam * w * h) + y * w + x) as usize
    }

    /// Borrow the host-side per-camera pose array. Useful for read-back
    /// in tooling / viewer code; no GPU sync involved (we always
    /// authoritative-write poses on the host via `set_camera_pose`).
    pub fn poses_host(&self) -> Result<&[Transform]> {
        self.poses.host()
    }
}

/// Build a Transform whose +X is "right", +Y is "up", -Z is "forward
/// = (target - eye).normalize()". Equivalent to the classical
/// `lookAt` matrix, but expressed as `(position, rotation)` for
/// molla's [`Transform`].
fn look_at_pose(eye: Vec3, target: Vec3, up_hint: Vec3) -> Transform {
    let f = (target - eye).normalize_or_zero();
    if f.length_squared() < 1e-12 {
        return Transform::new(eye, Quat::IDENTITY);
    }
    // Camera looks at -Z, so "forward" in local space = -Z. We want
    // -Z_local → +f_world, so we build a basis where the new -Z
    // points along f, +X along right, +Y along up. Quaternion =
    // rotation that maps the canonical basis to this one.
    let up_hint = if up_hint.length_squared() < 1e-12 {
        Vec3::Y
    } else {
        up_hint.normalize()
    };
    // Compute right = f × up (world-space). Re-orthogonalize up.
    let right = f.cross(up_hint).normalize_or_zero();
    let right = if right.length_squared() < 1e-12 {
        // f is parallel to up_hint — pick an alternate up.
        let alt = if up_hint.dot(Vec3::Z).abs() > 0.9 {
            Vec3::X
        } else {
            Vec3::Z
        };
        f.cross(alt).normalize()
    } else {
        right
    };
    let up = right.cross(f).normalize();
    // Camera-local axes in world frame:
    //   x_local → right
    //   y_local → up
    //   z_local → -f (since camera forward is -Z)
    let z_axis = -f;
    let m = molla_math::Mat3::from_cols(right, up, z_axis);
    let rot = Quat::from_mat3(&m);
    Transform::new(eye, rot)
}

/// Effective vertical FOV (radians). When the physical optics fields
/// are populated (`focal_length_mm > 0` and `sensor_width_mm > 0`),
/// the FOV is computed photographically:
///
/// ```text
///   sensor_height = sensor_width / aspect
///   fov_y         = 2 · atan( sensor_height / (2 · focal_length) )
/// ```
///
/// Otherwise the legacy `fov_y_radians` config field is honoured.
fn effective_fov_y(cfg: &TiledCameraConfig, aspect: f32) -> f32 {
    let f_mm = cfg.optics.focal_length_mm;
    let s_w_mm = cfg.optics.sensor_width_mm;
    if f_mm > 0.0 && s_w_mm > 0.0 && aspect > 0.0 {
        let sensor_height = s_w_mm / aspect;
        2.0 * (sensor_height / (2.0 * f_mm)).atan()
    } else {
        cfg.fov_y_radians
    }
}

/// `i`-th aperture sample (`1..k`) of the standard concentric-disc
/// mapping: well-stratified, uniform-on-the-disc, deterministic.
/// `(0, 0)` is reserved for the centre pass (handled by the caller).
fn concentric_disc_sample(i: u32, k: u32) -> [f32; 2] {
    let n = k.max(2);
    // Slot inside the disc — golden-angle Fibonacci spiral so the K
    // samples (centre + K-1) stay separated regardless of K.
    let frac = (i as f32) / ((n - 1) as f32);
    let r = frac.sqrt(); // uniform-on-the-disc radial CDF.
    let theta = (i as f32) * std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    [r * theta.cos(), r * theta.sin()]
}
