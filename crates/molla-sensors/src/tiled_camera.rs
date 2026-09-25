//! Multi-channel raytraced **camera sensor** running entirely on the
//! GPU. Mirrors Newton's `SensorTiledCamera`: per-camera pinhole
//! pose + intrinsics in, per-pixel `(depth, color, normal, albedo,
//! shape_index)` channels out, all sized
//! `(camera_count × width × height)`.
//!
//! Per-sample passes in one frame submission:
//! 1. [`CameraRaysKernel`] turns each `(camera, x, y)` triple into a
//!    world-space ray.
//! 2. [`Raycaster`] queries primitives and mesh instances on device.
//! 3. [`CameraShadeKernel`] shades hits into selected [`CameraChannels`].
//!
//! Aperture color samples accumulate and resolve on device; geometric channels
//! use the center sample. Owned [`SensorScene`] queries share compiled mesh
//! BVHs and include triangle/heightfield conversion.

use crate::{
    CameraStorageUsage, TiledCameraLimits,
    device_results::{
        EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool,
        SensorFramePoolConfig,
    },
    dispatch::BindingsCache,
    raycast::{RayQueryStats, Raycaster, RaycasterLimits},
    readback::{ReadbackSource, ReadbackStamp},
    scene::{SceneStamp, SensorScene},
};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{
    CameraAccumulateBindings, CameraAccumulateKernel, CameraRaysBindings, CameraRaysKernel,
    CameraRaysParams, CameraShadeBindings, CameraShadeKernel, CameraTraceParams,
};
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sim::{Model, State};
use molla_storage::{Buffer, BufferElement, BufferUsage};

/// Camera work counters, excluding constructor buffers, command encoders and driver staging.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CameraDispatchStats {
    pub submissions: u64,
    pub encoded_frames: u64,
    pub ray_passes: u64,
    pub trace_passes: u64,
    pub accumulation_passes: u64,
    pub resolve_passes: u64,
    pub rays_preparations: u64,
    /// Preparations of query-result shading bindings, excluding scene-query bindings.
    pub trace_preparations: u64,
    pub accumulator_preparations: u64,
    pub accumulator_buffer_allocations: u64,
}

/// Channels: depth, color, normal, albedo, shape ID; disabled channels have empty ranges.
pub struct EncodedCamera {
    pub frame: EncodedSensorFrame,
    pub width: u32,
    pub height: u32,
    pub camera_count: u32,
    pub channels: CameraChannels,
    pub scene_stamp: SceneStamp,
}

const MAX_APERTURE_SAMPLES: u32 = 1024;

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
#[derive(Clone, Copy, Debug, PartialEq)]
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

/// Physical intrinsics and deterministic thin-lens aperture sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalOptics {
    /// Physical focal length (mm). For a 50 mm "normal" lens on a 35
    /// mm full-frame sensor (width 36 mm), HFOV ≈ 39.6°.
    pub focal_length_mm: f32,
    /// Aperture entrance diameter (m). `0.0` (default) → pinhole.
    /// Positive values offset rays on the aperture disc.
    pub lens_diameter_m: f32,
    /// Plane the lens is focused on (m). Objects exactly at this
    /// distance render sharp; nearer/farther objects blur with a nonzero aperture.
    pub focal_distance_m: f32,
    /// Physical sensor width (mm). Combined with `focal_length_mm`
    /// gives HFOV as `2 · atan(sensor_width / (2 · focal_length))`.
    pub sensor_width_mm: f32,
    /// Number of aperture samples averaged per pixel when the DOF
    /// renderer is active. `0`/`1` = center sample; maximum is 1024.
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

/// A batch of pinhole or thin-lens cameras with shared intrinsics. Each camera has
/// its own world-space pose (set via [`Self::set_camera_pose`]); a single
/// [`Self::dispatch`] submits all samples for all cameras in one command stream.
///
/// Output buffers are sized `width * height * num_cameras`, indexed
/// row-major within an image, image-major across the batch
/// (`idx = cam * (W*H) + y * W + x`).
pub struct TiledCamera {
    device: std::sync::Arc<wgpu::Device>,
    health: std::sync::Arc<molla_compute::DeviceHealth>,
    config: TiledCameraConfig,
    num_cameras: u32,
    storage_usage: CameraStorageUsage,
    rays_kernel: CameraRaysKernel,
    shade_kernel: CameraShadeKernel,
    raycaster: Raycaster,
    rays_bindings: Option<CameraRaysBindings>,
    shade_bindings: BindingsCache<CameraShadeBindings>,
    direct_shade: BindingsCache<CameraShadeBindings>,
    direct_accumulation: BindingsCache<CameraAccumulateBindings>,
    channel_stamps: [Option<SceneStamp>; 5],
    accumulator: Option<CameraAccumulateKernel>,
    dispatch_stats: CameraDispatchStats,

    /// Per-camera world pose. Cheap to push to the device each
    /// dispatch; the pose array is small (`num_cameras` entries).
    poses: Buffer<Transform>,
    pose_upload: Vec<<Transform as BufferElement>::Gpu>,
    empty_color: Buffer<Vec3>,

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
    /// allocating base pixel/pose buffers up-front; aperture scratch is lazy.
    /// Initial poses default to identity (camera at world origin
    /// looking at -Z) — call [`Self::set_camera_pose`] before the first
    /// [`Self::dispatch`] to point each camera at the scene.
    pub fn new(engine: &Engine, config: TiledCameraConfig, num_cameras: usize) -> Result<Self> {
        Self::new_with_limits(engine, config, num_cameras, TiledCameraLimits::default())
    }

    /// Override the ray sub-budget while retaining the default combined camera-array budget.
    pub fn new_with_ray_limits(
        engine: &Engine,
        config: TiledCameraConfig,
        num_cameras: usize,
        limits: RaycasterLimits,
    ) -> Result<Self> {
        Self::new_with_limits(
            engine,
            config,
            num_cameras,
            TiledCameraLimits {
                ray_limits: limits,
                ..Default::default()
            },
        )
    }

    /// Validate ray and combined fixed-array budgets before constructing any camera buffers.
    pub fn new_with_limits(
        engine: &Engine,
        config: TiledCameraConfig,
        num_cameras: usize,
        limits: TiledCameraLimits,
    ) -> Result<Self> {
        engine.check_device()?;
        validate_config(&config)?;
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

        let storage_usage =
            limits.check(config.width, config.height, num_cameras, &device.limits())?;
        let n_pixels = storage_usage.pixels as usize;
        let raycaster = Raycaster::new_with_limits(engine, n_pixels as u32, limits.ray_limits)?;
        let mut poses = Buffer::from_vec(
            crate::host_arrays::filled(num_cameras, Transform::IDENTITY)?,
            BufferUsage::Storage,
        );
        let mut empty_color = Buffer::from_vec(
            crate::host_arrays::filled(1, Vec3::ZERO)?,
            BufferUsage::Storage,
        );

        let mut depth = Buffer::from_vec(
            crate::host_arrays::filled(n_pixels, -1.0_f64)?,
            BufferUsage::StorageReadWrite,
        );
        let mut color = Buffer::from_vec(
            crate::host_arrays::filled(n_pixels, 0_u32)?,
            BufferUsage::StorageReadWrite,
        );
        let mut normal = Buffer::from_vec(
            crate::host_arrays::filled(n_pixels, Vec3::ZERO)?,
            BufferUsage::StorageReadWrite,
        );
        let mut albedo = Buffer::from_vec(
            crate::host_arrays::filled(n_pixels, 0_u32)?,
            BufferUsage::StorageReadWrite,
        );
        let mut shape_index = Buffer::from_vec(
            crate::host_arrays::filled(n_pixels, 0xffff_ffff_u32)?,
            BufferUsage::StorageReadWrite,
        );

        poses.ensure_device(&device)?;
        empty_color.ensure_device(&device)?;
        depth.ensure_device(&device)?;
        color.ensure_device(&device)?;
        normal.ensure_device(&device)?;
        albedo.ensure_device(&device)?;
        shape_index.ensure_device(&device)?;

        let mut pose_upload = crate::host_arrays::reserved(num_cameras)?;
        poses.upload_with_scratch(&queue, &mut pose_upload)?;
        empty_color.upload(&queue)?;
        depth.upload(&queue)?;
        color.upload(&queue)?;
        normal.upload(&queue)?;
        albedo.upload(&queue)?;
        shape_index.upload(&queue)?;

        Ok(Self {
            device: device.clone(),
            health: engine.device_health().expect("checked GPU engine").clone(),
            rays_bindings: None,
            shade_bindings: BindingsCache::new(),
            direct_shade: BindingsCache::with_capacity(128),
            direct_accumulation: BindingsCache::with_capacity(128),
            channel_stamps: [None; 5],
            accumulator: None,
            dispatch_stats: CameraDispatchStats::default(),
            config,
            num_cameras: num_cameras as u32,
            storage_usage,
            rays_kernel: CameraRaysKernel::new(&device),
            shade_kernel: CameraShadeKernel::new(&device),
            poses,
            pose_upload,
            empty_color,
            raycaster,
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

    pub fn storage_usage(&self) -> CameraStorageUsage {
        self.storage_usage
    }

    pub(crate) fn output_specs(&self) -> [SensorChannelSpec; 5] {
        let pixels = (self.config.width as usize)
            .checked_mul(self.config.height as usize)
            .and_then(|pixels| pixels.checked_mul(self.num_cameras as usize))
            .expect("camera dimensions are bounded by the construction budget");
        [
            SensorChannelSpec::of::<Real>(pixels),
            SensorChannelSpec::of::<u32>(pixels),
            SensorChannelSpec::of::<Vec3>(pixels),
            SensorChannelSpec::of::<u32>(pixels),
            SensorChannelSpec::of::<u32>(pixels),
        ]
    }

    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        self.check_engine(engine)?;
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    pub(crate) fn preflight_scene(
        &self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
        stamp: ReadbackStamp,
    ) -> Result<(usize, u32)> {
        self.check_engine(engine)?;
        scene.check_engine(engine)?;
        if stamp
            .simulation_time
            .is_some_and(|time| !time.is_finite() || time < 0.0)
        {
            return Err(Error::Solver("camera: invalid timestamp".into()));
        }
        let pixels = checked_pixel_count(
            self.config.width,
            self.config.height,
            self.num_cameras as usize,
            &self.device.limits(),
        )?;
        if self.config.channels != CameraChannels::NONE {
            self.raycaster.preflight_rebuild(scene)?;
            if state.body_q.len() != scene.model.body_count {
                return Err(Error::Solver(
                    "camera: state/model body counts differ".into(),
                ));
            }
            if scene.model.body_count != 0 {
                state.body_q.device_buffer_for(&self.device)?;
            }
        }
        let shape_count = u32::try_from(scene.model.shape_count)
            .map_err(|_| Error::Solver("camera: too many shapes".into()))?;
        Ok((pixels, shape_count))
    }

    /// Encode selected channels into a leased frame; scene/state/producer borrows last through submission.
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sim::State;
    /// use molla_sensors::{TiledCamera, SensorCommandBatch, SensorFramePool, SensorScene};
    /// fn reuse(engine: &Engine, camera: &mut TiledCamera, scene: &SensorScene, state: &State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     camera.encode_scene(&mut batch, engine, scene, state, pool, Default::default()).unwrap();
    ///     camera.set_config(Default::default()).unwrap();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sim::State;
    /// use molla_sensors::{TiledCamera, SensorCommandBatch, SensorFramePool, SensorScene};
    /// fn mutate(engine: &Engine, camera: &mut TiledCamera, scene: &SensorScene, state: &mut State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     camera.encode_scene(&mut batch, engine, scene, state, pool, Default::default()).unwrap();
    ///     state.body_q.host_mut().unwrap()[0] = molla_math::Transform::IDENTITY;
    ///     batch.submit().unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sim::{Model, State};
    /// use molla_sensors::{TiledCamera, SensorCommandBatch, SensorFramePool, SensorScene};
    /// fn replace(engine: &Engine, camera: &mut TiledCamera, scene: &mut SensorScene, model: &Model, state: &State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     camera.encode_scene(&mut batch, engine, scene, state, pool, Default::default()).unwrap();
    ///     scene.replace_model(engine, model).unwrap();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn encode_scene<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        engine: &Engine,
        scene: &'a SensorScene,
        state: &'a State,
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedCamera>> {
        batch.check_engine(engine)?;
        let (pixels, shape_count) = self.preflight_scene(engine, scene, state, stamp)?;
        let specs = self.output_specs();
        let ranges = [
            CameraChannels::DEPTH,
            CameraChannels::COLOR,
            CameraChannels::NORMAL,
            CameraChannels::ALBEDO,
            CameraChannels::SHAPE_INDEX,
        ]
        .map(|channel| {
            0..if self.config.channels.contains(channel) {
                pixels
            } else {
                0
            }
        });
        let frame = batch.encode_output(
            pool,
            &specs,
            &ranges,
            stamp,
            |device, queue, encoder, outputs| {
                if self.config.channels == CameraChannels::NONE {
                    return Ok(0);
                }
                let hits = self.raycaster.device_hits(pixels);
                let buffers = [
                    hits.distance.device_buffer()?,
                    hits.shape.device_buffer()?,
                    hits.normal.device_buffer()?,
                    if shape_count == 0 {
                        self.empty_color.device_buffer_current()?
                    } else {
                        scene.model.shape_color.device_buffer_current()?
                    },
                    &outputs[0],
                    &outputs[1],
                    &outputs[2],
                    &outputs[3],
                    &outputs[4],
                ];
                let (shade_bindings, prepared) = self.direct_shade.try_get_or_prepare(
                    |bindings| bindings.matches(buffers),
                    || self.shade_kernel.prepare(device, buffers),
                )?;
                self.dispatch_stats.trace_preparations += u64::from(prepared);
                self.poses
                    .upload_with_scratch(queue, &mut self.pose_upload)?;
                if self.rays_bindings.is_none() {
                    self.rays_bindings = Some(self.rays_kernel.prepare(
                        device,
                        [
                            self.poses.device_buffer_current()?,
                            hits.origins.device_buffer()?,
                            hits.directions.device_buffer()?,
                        ],
                        MAX_APERTURE_SAMPLES,
                    )?);
                    self.dispatch_stats.rays_preparations += 1;
                }
                let rays_bindings = self.rays_bindings.as_ref().expect("prepared camera rays");
                let aspect = self.config.width as f32 / self.config.height as f32;
                let lens_radius = self.config.optics.lens_diameter_m * 0.5;
                let use_accumulation = lens_radius > 0.0
                    && self.config.optics.aperture_samples > 1
                    && self.config.channels.contains(CameraChannels::COLOR);
                let samples = if use_accumulation {
                    self.config.optics.aperture_samples
                } else {
                    1
                };
                let accumulation = if use_accumulation {
                    if self.accumulator.is_none() {
                        self.accumulator = Some(CameraAccumulateKernel::new(
                            device,
                            &outputs[1],
                            [self.config.width, self.config.height, self.num_cameras],
                        )?);
                        self.dispatch_stats.accumulator_buffer_allocations += 2;
                    }
                    let kernel = self.accumulator.as_ref().expect("camera accumulation");
                    let (bindings, prepared) = self.direct_accumulation.try_get_or_prepare(
                        |bindings| bindings.matches(&outputs[1]),
                        || kernel.prepare(device, &outputs[1]),
                    )?;
                    self.dispatch_stats.accumulator_preparations += u64::from(prepared);
                    kernel.encode_reset(queue, encoder, samples)?;
                    Some((kernel, bindings))
                } else {
                    None
                };
                let light_dir = if self.config.light_dir.length_squared() > 1e-12 {
                    self.config.light_dir.normalize()
                } else {
                    Vec3::new(0.0, -1.0, 0.0)
                };
                let mut dispatches = 0;
                for sample in 0..samples {
                    self.rays_kernel.encode_prepared(
                        queue,
                        encoder,
                        rays_bindings,
                        sample,
                        CameraRaysParams {
                            width: self.config.width,
                            height: self.config.height,
                            camera_count: self.num_cameras,
                            fov_y: effective_fov_y(&self.config, aspect),
                            aspect,
                            near: self.config.near,
                            lens_radius,
                            focal_distance: self.config.optics.focal_distance_m,
                            aperture_offset: if sample == 0 {
                                [0.0, 0.0]
                            } else {
                                concentric_disc_sample(sample, samples)
                            },
                        },
                    )?;
                    dispatches += self.raycaster.encode_camera_sample(
                        engine,
                        encoder,
                        &scene.model,
                        state,
                        pixels,
                        self.config.max_distance as Real,
                        Some(scene),
                        sample,
                    )?;
                    self.shade_kernel.encode_prepared(
                        queue,
                        encoder,
                        shade_bindings,
                        u32::from(sample != 0),
                        CameraTraceParams {
                            width: self.config.width,
                            height: self.config.height,
                            camera_count: self.num_cameras,
                            shape_count,
                            max_distance: self.config.max_distance,
                            channels: if sample == 0 {
                                self.config.channels.bits()
                            } else {
                                CameraChannels::COLOR.bits()
                            },
                            ambient: self.config.ambient,
                            light_dir,
                            light_color: self.config.light_color,
                        },
                    )?;
                    dispatches += 2;
                    if let Some((kernel, bindings)) = accumulation {
                        kernel.encode_accumulate_prepared(encoder, bindings);
                        dispatches += 1;
                    }
                }
                if let Some((kernel, bindings)) = accumulation {
                    kernel.encode_resolve_prepared(encoder, bindings);
                    dispatches += 1;
                    self.dispatch_stats.accumulation_passes += samples as u64;
                    self.dispatch_stats.resolve_passes += 1;
                }
                self.dispatch_stats.ray_passes += samples as u64;
                self.dispatch_stats.trace_passes += samples as u64;
                Ok(dispatches)
            },
        )?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        self.dispatch_stats.encoded_frames += 1;
        if self.config.channels != CameraChannels::NONE {
            batch.record_input_upload(
                self.num_cameras as u64
                    * std::mem::size_of::<<Transform as BufferElement>::Gpu>() as u64,
            );
        }
        Ok(Some(EncodedCamera {
            frame,
            width: self.config.width,
            height: self.config.height,
            camera_count: self.num_cameras,
            channels: self.config.channels,
            scene_stamp: scene.stamp(),
        }))
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
        validate_config(&config)?;
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
        validate_camera_pose(pose)?;
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

    /// Generate rays, query primitive/mesh visibility and shade selected channels.
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
    ///   K aperture-disc samples accumulate packed-byte color on device,
    ///   then resolve once. Other channels use only the center sample.
    ///   Without COLOR, or with `aperture_samples <= 1`, only the center runs.
    /// * Pinhole (the default) is bit-identical to the pre-H10 output.
    pub fn dispatch(&mut self, engine: &Engine, model: &Model, state: &State) -> Result<()> {
        self.dispatch_for_model(engine, model, state, None)
    }

    /// Query owned geometry/materials with shared mesh BVHs and live body poses.
    pub fn dispatch_scene(
        &mut self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
    ) -> Result<()> {
        scene.check_engine(engine)?;
        self.dispatch_for_model(engine, &scene.model, state, Some(scene))
    }

    /// Revision of the last sensor write to one channel; disabled channels retain
    /// their earlier stamp. Raw-model writes have no owned scene stamp.
    pub fn channel_scene_stamp(&self, channel: CameraChannels) -> Result<Option<SceneStamp>> {
        let bits = channel.bits();
        if bits.count_ones() != 1 || bits & !CameraChannels::ALL.bits() != 0 {
            return Err(Error::Solver(
                "camera stamp: select exactly one channel".into(),
            ));
        }
        Ok(self.channel_stamps[bits.trailing_zeros() as usize])
    }

    pub fn query_stats(&self) -> RayQueryStats {
        self.raycaster.query_stats()
    }

    pub fn scene_rebuild_policy(&self) -> crate::SceneRebuildPolicy {
        self.raycaster.scene_rebuild_policy()
    }

    pub fn set_scene_rebuild_policy(&mut self, policy: crate::SceneRebuildPolicy) -> Result<()> {
        self.raycaster.set_scene_rebuild_policy(policy)
    }

    pub fn mesh_build_count(&self) -> u64 {
        self.raycaster.mesh_build_count()
    }

    fn dispatch_for_model(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        scene: Option<&SensorScene>,
    ) -> Result<()> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("camera: no GPU device".into()))?;
        if !std::sync::Arc::ptr_eq(device, &self.device) {
            return Err(Error::Gpu("camera belongs to a different engine".into()));
        }
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("camera: no GPU queue".into()))?;
        let pixels = checked_pixel_count(
            self.config.width,
            self.config.height,
            self.num_cameras as usize,
            &device.limits(),
        )?;
        if state.body_q.len() != model.body_count
            || [
                model.shape_geom_kind.len(),
                model.shape_geometry.len(),
                model.shape_geom_data.len(),
                model.shape_body.len(),
                model.shape_transform.len(),
                model.shape_color.len(),
            ]
            .into_iter()
            .any(|count| count != model.shape_count)
            || self.depth.len() != pixels
            || self.color.len() != pixels
            || self.normal.len() != pixels
            || self.albedo.len() != pixels
            || self.shape_index.len() != pixels
        {
            return Err(Error::Solver(
                "camera: inconsistent state or output sizes".into(),
            ));
        }
        let shape_count = u32::try_from(model.shape_count)
            .map_err(|_| Error::Solver("camera: too many shapes".into()))?;
        if model.body_count != 0 {
            state.body_q.device_buffer_current()?;
        }
        if shape_count != 0 {
            model.shape_geom_kind.device_buffer_current()?;
            model.shape_geom_data.device_buffer_current()?;
            model.shape_body.device_buffer_current()?;
            model.shape_transform.device_buffer_current()?;
        }
        let hits = self.raycaster.device_hits(pixels);
        let buffers = [
            hits.distance.device_buffer()?,
            hits.shape.device_buffer()?,
            hits.normal.device_buffer()?,
            if shape_count == 0 {
                self.empty_color.device_buffer_current()?
            } else {
                model.shape_color.device_buffer_current()?
            },
            self.depth.device_buffer_current()?,
            self.color.device_buffer_current()?,
            self.normal.device_buffer_current()?,
            self.albedo.device_buffer_current()?,
            self.shape_index.device_buffer_current()?,
        ];
        if self.config.channels == CameraChannels::NONE {
            return Ok(());
        }
        if let Some(scene) = scene {
            self.raycaster.preflight_rebuild(scene)?;
        }
        let (shade_bindings, prepared) = self.shade_bindings.try_get_or_prepare(
            |binding| binding.matches(buffers),
            || self.shade_kernel.prepare(device, buffers),
        )?;
        self.dispatch_stats.trace_preparations += u64::from(prepared);
        self.poses
            .upload_with_scratch(queue, &mut self.pose_upload)?;
        if self.rays_bindings.is_none() {
            self.rays_bindings = Some(self.rays_kernel.prepare(
                device,
                [
                    self.poses.device_buffer_current()?,
                    hits.origins.device_buffer()?,
                    hits.directions.device_buffer()?,
                ],
                MAX_APERTURE_SAMPLES,
            )?);
            self.dispatch_stats.rays_preparations += 1;
        }
        let rays_bindings = self.rays_bindings.as_ref().expect("prepared ray bindings");
        let aspect = self.config.width as f32 / self.config.height as f32;
        let lens_radius = self.config.optics.lens_diameter_m * 0.5;
        let use_accumulation = lens_radius > 0.0
            && self.config.optics.aperture_samples > 1
            && self.config.channels.contains(CameraChannels::COLOR);
        let samples = if use_accumulation {
            self.config.optics.aperture_samples
        } else {
            1
        };
        if use_accumulation
            && self
                .accumulator
                .as_ref()
                .is_none_or(|a| !a.matches(buffers[5]))
        {
            if let Some(accumulator) = &mut self.accumulator {
                accumulator.rebind_color(device, buffers[5])?;
            } else {
                self.accumulator = Some(CameraAccumulateKernel::new(
                    device,
                    buffers[5],
                    [self.config.width, self.config.height, self.num_cameras],
                )?);
                self.dispatch_stats.accumulator_buffer_allocations += 2;
            }
            self.dispatch_stats.accumulator_preparations += 1;
        }
        let light_dir = if self.config.light_dir.length_squared() > 1e-12 {
            self.config.light_dir.normalize()
        } else {
            Vec3::new(0.0, -1.0, 0.0)
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::camera frame"),
        });
        if use_accumulation {
            self.accumulator
                .as_ref()
                .expect("aperture accumulator")
                .encode_reset(queue, &mut encoder, samples)?;
        }
        for sample in 0..samples {
            self.rays_kernel.encode_prepared(
                queue,
                &mut encoder,
                rays_bindings,
                sample,
                CameraRaysParams {
                    width: self.config.width,
                    height: self.config.height,
                    camera_count: self.num_cameras,
                    fov_y: effective_fov_y(&self.config, aspect),
                    aspect,
                    near: self.config.near,
                    lens_radius,
                    focal_distance: self.config.optics.focal_distance_m,
                    aperture_offset: if sample == 0 {
                        [0.0, 0.0]
                    } else {
                        concentric_disc_sample(sample, samples)
                    },
                },
            )?;
            self.raycaster.encode_camera_sample(
                engine,
                &mut encoder,
                model,
                state,
                pixels,
                self.config.max_distance as Real,
                scene,
                sample,
            )?;
            self.shade_kernel.encode_prepared(
                queue,
                &mut encoder,
                shade_bindings,
                u32::from(sample != 0),
                CameraTraceParams {
                    width: self.config.width,
                    height: self.config.height,
                    camera_count: self.num_cameras,
                    shape_count,
                    max_distance: self.config.max_distance,
                    channels: if sample == 0 {
                        self.config.channels.bits()
                    } else {
                        CameraChannels::COLOR.bits()
                    },
                    ambient: self.config.ambient,
                    light_dir,
                    light_color: self.config.light_color,
                },
            )?;
            if use_accumulation {
                self.accumulator
                    .as_ref()
                    .expect("aperture accumulator")
                    .encode_accumulate(&mut encoder);
            }
        }
        if use_accumulation {
            self.accumulator
                .as_ref()
                .expect("aperture accumulator")
                .encode_resolve(&mut encoder);
        }
        queue.submit([encoder.finish()]);
        self.dispatch_stats.submissions += 1;
        self.dispatch_stats.ray_passes += samples as u64;
        self.dispatch_stats.trace_passes += samples as u64;
        if use_accumulation {
            self.dispatch_stats.accumulation_passes += samples as u64;
            self.dispatch_stats.resolve_passes += 1;
        }
        for (index, stamp) in self.channel_stamps.iter_mut().enumerate() {
            if self.config.channels.bits() & (1 << index) != 0 {
                *stamp = scene.map(SensorScene::stamp);
            }
        }
        if self.config.channels.contains(CameraChannels::DEPTH) {
            self.depth.mark_device_dirty();
        }
        if self.config.channels.contains(CameraChannels::COLOR) {
            self.color.mark_device_dirty();
        }
        if self.config.channels.contains(CameraChannels::NORMAL) {
            self.normal.mark_device_dirty();
        }
        if self.config.channels.contains(CameraChannels::ALBEDO) {
            self.albedo.mark_device_dirty();
        }
        if self.config.channels.contains(CameraChannels::SHAPE_INDEX) {
            self.shape_index.mark_device_dirty();
        }
        Ok(())
    }

    pub fn dispatch_stats(&self) -> CameraDispatchStats {
        self.dispatch_stats
    }

    /// Select one channel and active pixel range for grouped readback or a copied
    /// snapshot. Enqueue consumers before redispatch; this view does not lease output.
    pub fn readback_source(
        &self,
        channel: CameraChannels,
        range: std::ops::Range<usize>,
    ) -> Result<ReadbackSource<'_>> {
        self.health.check()?;
        match channel {
            CameraChannels::DEPTH => ReadbackSource::from_owned(&self.device, &self.depth, range),
            CameraChannels::COLOR => ReadbackSource::from_owned(&self.device, &self.color, range),
            CameraChannels::NORMAL => ReadbackSource::from_owned(&self.device, &self.normal, range),
            CameraChannels::ALBEDO => ReadbackSource::from_owned(&self.device, &self.albedo, range),
            CameraChannels::SHAPE_INDEX => {
                ReadbackSource::from_owned(&self.device, &self.shape_index, range)
            }
            _ => Err(Error::Solver(
                "camera readback: select exactly one channel".into(),
            )),
        }
    }

    /// Download the depth channel into the host mirror and return a
    /// borrow of the per-pixel slice. Same semantics for the other
    /// `download_*` helpers below — they're convenience wrappers
    /// around `Buffer::download` + `Buffer::host`.
    pub fn download_depth(&mut self, engine: &Engine) -> Result<&[Real]> {
        self.check_engine(engine)?;
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
        self.check_engine(engine)?;
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
        self.check_engine(engine)?;
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
        self.check_engine(engine)?;
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
        self.check_engine(engine)?;
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

    fn check_engine(&self, engine: &Engine) -> Result<()> {
        engine.check_device()?;
        if engine
            .device()
            .is_none_or(|device| !std::sync::Arc::ptr_eq(device, &self.device))
        {
            return Err(Error::Gpu("camera belongs to a different engine".into()));
        }
        Ok(())
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

pub(crate) fn checked_pixel_count(
    width: u32,
    height: u32,
    cameras: usize,
    limits: &wgpu::Limits,
) -> Result<usize> {
    let cameras = u32::try_from(cameras)
        .map_err(|_| Error::Solver("camera: camera count exceeds u32".into()))?;
    let pixels = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(cameras))
        .ok_or_else(|| Error::Solver("camera: pixel count exceeds u32".into()))?;
    let binding_bytes = (pixels as u64 * 16).max(cameras as u64 * 32);
    if pixels == 0
        || binding_bytes > limits.max_storage_buffer_binding_size as u64
        || binding_bytes > limits.max_buffer_size
        || width.div_ceil(8) > limits.max_compute_workgroups_per_dimension
        || height.div_ceil(8) > limits.max_compute_workgroups_per_dimension
        || cameras > limits.max_compute_workgroups_per_dimension
        || pixels.div_ceil(64) > limits.max_compute_workgroups_per_dimension
    {
        return Err(Error::Solver(
            "camera: dimensions exceed GPU limits or are zero".into(),
        ));
    }
    Ok(pixels as usize)
}

pub(crate) fn validate_config(config: &TiledCameraConfig) -> Result<()> {
    let optics = config.optics;
    let scalars = [
        config.fov_y_radians,
        config.max_distance,
        config.near,
        optics.focal_length_mm,
        optics.sensor_width_mm,
        optics.lens_diameter_m,
        optics.focal_distance_m,
    ];
    if scalars.iter().any(|value| !value.is_finite())
        || config.fov_y_radians <= 0.0
        || config.fov_y_radians >= std::f32::consts::PI
        || config.max_distance <= 0.0
        || config.near < 0.0
        || optics.focal_length_mm < 0.0
        || optics.sensor_width_mm < 0.0
        || optics.lens_diameter_m < 0.0
        || optics.focal_distance_m < 0.0
        || optics.aperture_samples > MAX_APERTURE_SAMPLES
        || config.channels.bits() & !CameraChannels::ALL.bits() != 0
        || [config.ambient, config.light_dir, config.light_color]
            .iter()
            .any(|v| !v.is_finite() || !v.as_vec3().is_finite())
    {
        return Err(Error::Solver(
            "camera: invalid configuration or aperture sample budget".into(),
        ));
    }
    let aspect = config.width as f32 / config.height as f32;
    let effective = effective_fov_y(config, aspect);
    if !effective.is_finite() || effective <= 0.0 || effective >= std::f32::consts::PI {
        return Err(Error::Solver(
            "camera: invalid effective field of view".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_camera_pose(pose: Transform) -> Result<()> {
    if !pose.position.is_finite()
        || !pose.position.as_vec3().is_finite()
        || !pose.rotation.is_finite()
        || (pose.rotation.length_squared() - 1.0).abs() > 1e-6
    {
        return Err(Error::Solver(
            "camera: pose must be finite and rigid".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    #[ignore = "requires GPU"]
    fn camera_storage_estimate_matches_allocated_arrays_and_aperture() {
        let engine = Engine::new_gpu().unwrap();
        let config = TiledCameraConfig {
            width: 7,
            height: 5,
            ..Default::default()
        };
        let usage = CameraStorageUsage::for_dimensions(7, 5, 3).unwrap();
        let limits = TiledCameraLimits {
            max_storage_bytes: usage.total_bytes,
            ..Default::default()
        };
        let camera = TiledCamera::new_with_limits(&engine, config, 3, limits).unwrap();
        assert_eq!(camera.storage_usage(), usage);
        let rays = crate::RayStorageUsage::for_capacity(usage.pixels).unwrap();
        let buffers = [
            camera.poses.device_buffer().unwrap(),
            camera.empty_color.device_buffer().unwrap(),
            camera.depth.device_buffer().unwrap(),
            camera.color.device_buffer().unwrap(),
            camera.normal.device_buffer().unwrap(),
            camera.albedo.device_buffer().unwrap(),
            camera.shape_index.device_buffer().unwrap(),
        ];
        let allocated = buffers.iter().map(|b| b.size()).sum::<u64>();
        assert_eq!(
            allocated + rays.device_bytes + usage.aperture_device_bytes,
            usage.device_bytes
        );
        let host = size_of_val(camera.poses.host().unwrap())
            + size_of_val(camera.empty_color.host().unwrap())
            + size_of_val(camera.depth.host().unwrap())
            + size_of_val(camera.color.host().unwrap())
            + size_of_val(camera.normal.host().unwrap())
            + size_of_val(camera.albedo.host().unwrap())
            + size_of_val(camera.shape_index.host().unwrap())
            + camera.pose_upload.capacity() * size_of::<<Transform as BufferElement>::Gpu>();
        assert_eq!(host as u64 + rays.host_bytes, usage.host_bytes);
        let aperture = CameraAccumulateKernel::new(
            engine.device().unwrap(),
            camera.color.device_buffer().unwrap(),
            [7, 5, 3],
        )
        .unwrap();
        assert_eq!(aperture.storage_bytes(), usage.aperture_device_bytes);
    }

    #[test]
    fn invalid_camera_parameters_are_rejected_before_allocation() {
        let config = TiledCameraConfig::default();
        assert!(validate_config(&config).is_ok());
        for value in [f32::NAN, f32::INFINITY, -1.0, 0.0, std::f32::consts::PI] {
            assert!(
                validate_config(&TiledCameraConfig {
                    fov_y_radians: value,
                    ..config
                })
                .is_err()
            );
        }
        for value in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
            assert!(
                validate_config(&TiledCameraConfig {
                    max_distance: value,
                    ..config
                })
                .is_err()
            );
        }
        for samples in [0, 1, 1024, 1025, u32::MAX] {
            let cfg = TiledCameraConfig {
                optics: PhysicalOptics {
                    aperture_samples: samples,
                    ..config.optics
                },
                ..config
            };
            assert_eq!(validate_config(&cfg).is_ok(), samples <= 1024);
        }
        assert!(
            validate_config(&TiledCameraConfig {
                channels: CameraChannels(32),
                ..config
            })
            .is_err()
        );
        assert!(
            validate_config(&TiledCameraConfig {
                ambient: Vec3::splat(f64::MAX),
                ..config
            })
            .is_err()
        );
    }

    #[test]
    fn camera_dimensions_are_checked_before_allocation() {
        let limits = wgpu::Limits::default();
        assert_eq!(
            checked_pixel_count(640, 480, 4, &limits).unwrap(),
            1_228_800
        );
        for (width, height, cameras) in [
            (u32::MAX, 2, 1),
            (1, 1, usize::MAX),
            (65_536, 65_536, 1),
            (8192, 8192, 4),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
        ] {
            assert!(checked_pixel_count(width, height, cameras, &limits).is_err());
        }
        let dispatch_limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..limits
        };
        assert!(checked_pixel_count(9, 1, 1, &dispatch_limits).is_err());
        assert!(checked_pixel_count(1, 9, 1, &dispatch_limits).is_err());
        assert!(checked_pixel_count(1, 1, 2, &dispatch_limits).is_err());
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
