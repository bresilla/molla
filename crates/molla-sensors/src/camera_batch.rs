//! Heterogeneous camera groups with stable image IDs and checked output slices.

use crate::readback::{ReadbackSource, ReadbackStamp};
use crate::tiled_camera::{checked_pixel_count, validate_camera_pose, validate_config};
use crate::{
    CameraChannels, CameraDispatchStats, EncodedCamera, ImageOp, PendingAnnotationJob,
    PendingImageJob, RayQueryStats, SceneStamp, SensorCommandBatch, SensorFramePool,
    SensorFramePoolConfig, SensorFramePoolStats, SensorScene, TiledCamera, TiledCameraConfig,
    TiledCameraLimits,
};
use arrayvec::ArrayVec;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{Transform, Vec3};
use molla_sim::{Model, State};
use std::{collections::HashSet, ops::Range, sync::Arc};

#[derive(Debug, Clone, Copy)]
pub struct CameraSpec {
    pub sensor_id: u64,
    pub config: TiledCameraConfig,
    pub pose: Transform,
}

/// Per-batch limits, independent of readback/snapshot and process-wide GPU memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraBatchLimits {
    pub max_cameras: usize,
    pub max_groups: usize,
    pub max_total_pixels: u64,
}

impl Default for CameraBatchLimits {
    fn default() -> Self {
        Self {
            max_cameras: 64,
            max_groups: 16,
            max_total_pixels: 4_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraImageLayout {
    pub sensor_id: u64,
    pub group_index: usize,
    pub camera_index: u32,
    pub width: u32,
    pub height: u32,
    pub channels: CameraChannels,
    pub pixels: Range<usize>,
}

/// Sampling identity; owner/generation checks remain valid for queued historical copies.
#[derive(Debug, Clone)]
pub struct CameraBatchStamp {
    owner: Arc<()>,
    generation: u64,
    sample_id: u64,
}

impl CameraBatchStamp {
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn sample_id(&self) -> u64 {
        self.sample_id
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CameraBatchStats {
    pub frames: u64,
    pub encoded_frames: u64,
    pub submissions: u64,
    pub failed_frames: u64,
    pub reconfigurations: u64,
}

/// One independently leased pool per homogeneous group; the byte budget covers all groups.
pub struct CameraBatchOutputPool {
    owner: Arc<()>,
    health: Arc<molla_compute::DeviceHealth>,
    images: Arc<[CameraImageLayout]>,
    groups: Vec<SensorFramePool>,
    slots: usize,
}

impl CameraBatchOutputPool {
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
    pub fn group_stats(&self, index: usize) -> Option<SensorFramePoolStats> {
        self.groups.get(index).map(SensorFramePool::stats)
    }
    pub fn available_slots(&mut self) -> Result<usize> {
        self.health.check()?;
        let mut available = self.slots;
        for group in &mut self.groups {
            available = available.min(group.available_slots()?);
        }
        Ok(available)
    }
    pub fn reset(&mut self) -> Result<()> {
        self.health.check()?;
        for group in &mut self.groups {
            group.reset()?;
        }
        Ok(())
    }
}

pub struct EncodedCameraBatch {
    pub groups: ArrayVec<EncodedCamera, 64>,
    pub images: Arc<[CameraImageLayout]>,
    pub stamp: CameraBatchStamp,
    pub simulation_time: Option<f64>,
}

impl EncodedCameraBatch {
    pub fn image(&self, sensor_id: u64) -> Option<(&CameraImageLayout, &EncodedCamera)> {
        let layout = self
            .images
            .iter()
            .find(|image| image.sensor_id == sensor_id)?;
        Some((layout, self.groups.get(layout.group_index)?))
    }

    pub fn readback_stamp(&self, sensor_id: u64) -> Result<ReadbackStamp> {
        let (_, group) = self
            .image(sensor_id)
            .ok_or_else(|| Error::Solver("camera batch: unknown sensor ID".into()))?;
        Ok(ReadbackStamp {
            sensor_id,
            sample_id: self.stamp.sample_id,
            scene_revision: group.scene_stamp.geometry_revision,
            simulation_time: self.simulation_time,
        })
    }

    pub fn image_job<'a>(
        &'a self,
        sensor_id: u64,
        operations: &'a [ImageOp],
    ) -> Result<PendingImageJob<'a>> {
        let (layout, group) = self
            .image(sensor_id)
            .ok_or_else(|| Error::Solver("camera batch: unknown sensor ID".into()))?;
        if !layout.channels.contains(CameraChannels::COLOR) {
            return Err(Error::Solver("camera batch: color was not rendered".into()));
        }
        Ok(PendingImageJob {
            frame: &group.frame,
            channel: 1,
            pixels: layout.pixels.clone(),
            operations,
            stamp: self.readback_stamp(sensor_id)?,
        })
    }

    pub fn annotation_job(&self, sensor_id: u64) -> Result<PendingAnnotationJob<'_>> {
        let (layout, group) = self
            .image(sensor_id)
            .ok_or_else(|| Error::Solver("camera batch: unknown sensor ID".into()))?;
        if !layout.channels.contains(CameraChannels::SHAPE_INDEX) {
            return Err(Error::Solver(
                "camera batch: shape IDs were not rendered".into(),
            ));
        }
        Ok(PendingAnnotationJob {
            frame: &group.frame,
            channel: 4,
            pixels: layout.pixels.clone(),
            width: layout.width,
            height: layout.height,
            stamp: self.readback_stamp(sensor_id)?,
        })
    }
}

/// Fixed-budget grouping by the complete camera configuration. Poses stay independent.
pub struct CameraBatch {
    device: Arc<wgpu::Device>,
    owner: Arc<()>,
    limits: CameraBatchLimits,
    storage_limits: TiledCameraLimits,
    storage_bytes: u64,
    groups: Vec<TiledCamera>,
    images: Arc<[CameraImageLayout]>,
    generation: u64,
    next_sample: u64,
    stats: CameraBatchStats,
}

/// Borrowed frame; enqueue consumers or copied snapshots before the next mutable batch call.
pub struct CameraBatchFrame<'a> {
    batch: &'a CameraBatch,
    stamp: CameraBatchStamp,
}

impl<'a> CameraBatchFrame<'a> {
    pub fn stamp(&self) -> &CameraBatchStamp {
        &self.stamp
    }
    pub fn images(&self) -> &[CameraImageLayout] {
        &self.batch.images
    }

    pub fn image(&self, sensor_id: u64) -> Option<CameraImage<'a>> {
        let layout = self
            .batch
            .images
            .iter()
            .find(|image| image.sensor_id == sensor_id)?;
        Some(CameraImage {
            layout,
            camera: &self.batch.groups[layout.group_index],
            sample_id: self.stamp.sample_id,
        })
    }
}

pub struct CameraImage<'a> {
    layout: &'a CameraImageLayout,
    camera: &'a TiledCamera,
    sample_id: u64,
}

impl<'a> CameraImage<'a> {
    pub fn layout(&self) -> &CameraImageLayout {
        self.layout
    }
    pub fn config(&self) -> &TiledCameraConfig {
        self.camera.config()
    }

    fn check_channel(&self, channel: CameraChannels) -> Result<()> {
        if channel.bits().count_ones() != 1 || !self.layout.channels.contains(channel) {
            return Err(Error::Solver(
                "camera batch: channel was not rendered".into(),
            ));
        }
        Ok(())
    }

    pub fn readback_source(&self, channel: CameraChannels) -> Result<ReadbackSource<'a>> {
        self.check_channel(channel)?;
        self.camera
            .readback_source(channel, self.layout.pixels.clone())
    }

    pub fn scene_stamp(&self, channel: CameraChannels) -> Result<Option<SceneStamp>> {
        self.check_channel(channel)?;
        self.camera.channel_scene_stamp(channel)
    }

    /// Retain the frame's batch stamp and full SceneStamp beside an asynchronous ticket.
    pub fn readback_stamp(
        &self,
        channel: CameraChannels,
        simulation_time: Option<f64>,
    ) -> Result<ReadbackStamp> {
        if simulation_time.is_some_and(|time| !time.is_finite() || time < 0.0) {
            return Err(Error::Solver(
                "camera batch: invalid simulation time".into(),
            ));
        }
        Ok(ReadbackStamp {
            sensor_id: self.layout.sensor_id,
            sample_id: self.sample_id,
            scene_revision: self
                .scene_stamp(channel)?
                .map_or(0, |stamp| stamp.geometry_revision),
            simulation_time,
        })
    }
}

impl CameraBatch {
    pub fn new(engine: &Engine, specs: &[CameraSpec], limits: CameraBatchLimits) -> Result<Self> {
        Self::new_with_storage_limits(engine, specs, limits, TiledCameraLimits::default())
    }

    /// Preflight the combined fixed-array payload of every group before GPU allocation.
    pub fn new_with_storage_limits(
        engine: &Engine,
        specs: &[CameraSpec],
        limits: CameraBatchLimits,
        storage_limits: TiledCameraLimits,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("camera batch requires GPU".into()))?
            .clone();
        let plan = plan(specs, limits, &device.limits())?;
        let storage_bytes = plan.storage_bytes(storage_limits, &device.limits())?;
        let mut groups = Vec::with_capacity(plan.groups.len());
        for group in plan.groups {
            let mut camera = TiledCamera::new_with_limits(
                engine,
                group.config,
                group.members.len(),
                storage_limits,
            )?;
            for (index, member) in group.members.into_iter().enumerate() {
                camera.set_camera_pose(index, specs[member].pose)?;
            }
            groups.push(camera);
        }
        Ok(Self {
            device,
            owner: Arc::new(()),
            limits,
            storage_limits,
            storage_bytes,
            groups,
            images: plan.images.into(),
            generation: 0,
            next_sample: 0,
            stats: Default::default(),
        })
    }

    pub fn layouts(&self) -> &[CameraImageLayout] {
        &self.images
    }

    /// Sum of configured group array payloads, including reserved lazy aperture scratch.
    pub fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }

    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<CameraBatchOutputPool> {
        self.check_engine(engine)?;
        let bytes = self.groups.iter().try_fold(0u64, |bytes, group| {
            let n = group.output_specs()[0].capacity() as u64;
            n.checked_mul(32)
                .and_then(|n| n.checked_mul(config.slots as u64))
                .and_then(|n| bytes.checked_add(n))
                .ok_or_else(|| Error::Storage("camera batch: output budget overflow".into()))
        })?;
        if config.slots == 0 || config.slots > 64 || bytes > config.max_total_bytes {
            return Err(Error::Storage(
                "camera batch: invalid or exceeded output budget".into(),
            ));
        }
        let groups = self
            .groups
            .iter()
            .map(|group| group.output_pool(engine, config))
            .collect::<Result<Vec<_>>>()?;
        Ok(CameraBatchOutputPool {
            owner: self.owner.clone(),
            health: engine.device_health().expect("checked engine").clone(),
            images: self.images.clone(),
            groups,
            slots: config.slots,
        })
    }

    /// Encode every group atomically with respect to validation and backpressure.
    /// Successful encodes reserve a sample ID even when their command batch is later cancelled.
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sim::State;
    /// use molla_sensors::{CameraBatch, CameraBatchOutputPool, SensorCommandBatch, SensorScene};
    /// fn reset(engine: &Engine, cameras: &mut CameraBatch, scene: &SensorScene, state: &State, pool: &mut CameraBatchOutputPool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     cameras.encode_scene(&mut batch, engine, scene, state, pool, None).unwrap();
    ///     cameras.reset().unwrap();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sim::State;
    /// use molla_sensors::{CameraBatch, CameraBatchOutputPool, SensorCommandBatch, SensorScene};
    /// fn mutate(engine: &Engine, cameras: &mut CameraBatch, scene: &SensorScene, state: &mut State, pool: &mut CameraBatchOutputPool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     cameras.encode_scene(&mut batch, engine, scene, state, pool, None).unwrap();
    ///     state.body_q.host_mut().unwrap()[0] = molla_math::Transform::IDENTITY;
    ///     batch.submit().unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sim::{Model, State};
    /// use molla_sensors::{CameraBatch, CameraBatchOutputPool, SensorCommandBatch, SensorScene};
    /// fn replace(engine: &Engine, cameras: &mut CameraBatch, scene: &mut SensorScene, model: &Model, state: &State, pool: &mut CameraBatchOutputPool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     cameras.encode_scene(&mut batch, engine, scene, state, pool, None).unwrap();
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
        pool: &mut CameraBatchOutputPool,
        simulation_time: Option<f64>,
    ) -> Result<Option<EncodedCameraBatch>> {
        self.check_engine(engine)?;
        batch.check_engine(engine)?;
        scene.check_engine(engine)?;
        if simulation_time.is_some_and(|time| !time.is_finite() || time < 0.0) {
            return Err(Error::Solver(
                "camera batch: invalid simulation time".into(),
            ));
        }
        if !Arc::ptr_eq(&self.owner, &pool.owner)
            || self.images != pool.images
            || self.groups.len() != pool.groups.len()
        {
            return Err(Error::Storage(
                "camera batch: foreign or incompatible output pool".into(),
            ));
        }
        batch.check_output_capacity(self.groups.len())?;
        let sample_id = self.next_sample;
        let next_sample = sample_id
            .checked_add(1)
            .ok_or_else(|| Error::Solver("camera batch: sample identity exhausted".into()))?;
        let group_stamp = ReadbackStamp {
            sensor_id: 0,
            sample_id,
            scene_revision: scene.stamp().geometry_revision,
            simulation_time,
        };
        for (camera, output) in self.groups.iter().zip(&pool.groups) {
            camera.preflight_scene(engine, scene, state, group_stamp)?;
            output.validate_layout(&self.device, &camera.output_specs())?;
        }
        let mut available = true;
        for output in &mut pool.groups {
            available &= output.preflight_available()?;
        }
        if !available {
            return Ok(None);
        }
        let mut groups = ArrayVec::new();
        for (camera, output) in self.groups.iter_mut().zip(&mut pool.groups) {
            match camera.encode_scene(batch, engine, scene, state, output, group_stamp) {
                Ok(Some(frame)) => groups.push(frame),
                result => {
                    batch.poison();
                    self.stats.failed_frames += 1;
                    return Err(match result {
                        Err(error) => error,
                        _ => Error::Storage(
                            "camera batch: output availability changed during encoding".into(),
                        ),
                    });
                }
            }
        }
        self.next_sample = next_sample;
        self.stats.encoded_frames += 1;
        Ok(Some(EncodedCameraBatch {
            groups,
            images: self.images.clone(),
            stamp: CameraBatchStamp {
                owner: self.owner.clone(),
                generation: self.generation,
                sample_id,
            },
            simulation_time,
        }))
    }
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
    pub fn stats(&self) -> CameraBatchStats {
        self.stats
    }
    pub fn group_stats(&self, index: usize) -> Option<CameraDispatchStats> {
        self.groups.get(index).map(TiledCamera::dispatch_stats)
    }
    pub fn query_stats(&self, index: usize) -> Option<RayQueryStats> {
        self.groups.get(index).map(TiledCamera::query_stats)
    }

    pub fn scene_rebuild_policy(&self, group: usize) -> Option<crate::SceneRebuildPolicy> {
        self.groups
            .get(group)
            .map(TiledCamera::scene_rebuild_policy)
    }

    pub fn set_scene_rebuild_policy(
        &mut self,
        group: usize,
        policy: crate::SceneRebuildPolicy,
    ) -> Result<()> {
        self.groups
            .get_mut(group)
            .ok_or_else(|| Error::Build("camera batch: unknown group".into()))?
            .set_scene_rebuild_policy(policy)
    }

    pub fn is_current(&self, stamp: &CameraBatchStamp) -> bool {
        Arc::ptr_eq(&self.owner, &stamp.owner) && self.generation == stamp.generation
    }

    pub fn set_camera_pose(&mut self, sensor_id: u64, pose: Transform) -> Result<()> {
        let layout = self.find(sensor_id)?;
        let (group, camera) = (layout.group_index, layout.camera_index as usize);
        self.groups[group].set_camera_pose(camera, pose)
    }

    pub fn set_camera_look_at(
        &mut self,
        sensor_id: u64,
        eye: Vec3,
        target: Vec3,
        up: Vec3,
    ) -> Result<()> {
        let layout = self.find(sensor_id)?;
        let (group, camera) = (layout.group_index, layout.camera_index as usize);
        self.groups[group].set_camera_look_at(camera, eye, target, up)
    }

    /// Build a replacement before swapping. IDs/order/configurations may change;
    /// existing copied results survive, but belong to the previous generation.
    pub fn reconfigure(&mut self, engine: &Engine, specs: &[CameraSpec]) -> Result<()> {
        engine.check_device()?;
        self.check_engine(engine)?;
        let generation = self.next_generation()?;
        let replacement =
            Self::new_with_storage_limits(engine, specs, self.limits, self.storage_limits)?;
        self.groups = replacement.groups;
        self.images = replacement.images;
        self.storage_bytes = replacement.storage_bytes;
        self.generation = generation;
        self.stats.reconfigurations += 1;
        Ok(())
    }

    /// Invalidate batch receipts without changing layouts, buffers or sample numbering.
    pub fn reset(&mut self) -> Result<()> {
        self.generation = self.next_generation()?;
        Ok(())
    }

    pub fn dispatch(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
    ) -> Result<CameraBatchFrame<'_>> {
        engine.check_device()?;
        self.dispatch_groups(engine, |camera| camera.dispatch(engine, model, state))
    }

    pub fn dispatch_scene(
        &mut self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
    ) -> Result<CameraBatchFrame<'_>> {
        scene.check_engine(engine)?;
        self.dispatch_groups(engine, |camera| camera.dispatch_scene(engine, scene, state))
    }

    fn dispatch_groups(
        &mut self,
        engine: &Engine,
        mut dispatch: impl FnMut(&mut TiledCamera) -> Result<()>,
    ) -> Result<CameraBatchFrame<'_>> {
        self.check_engine(engine)?;
        let failure_generation = self.next_generation()?;
        let sample_id = self.next_sample;
        self.next_sample = self
            .next_sample
            .checked_add(1)
            .ok_or_else(|| Error::Solver("camera batch: sample identity exhausted".into()))?;
        for group in &mut self.groups {
            let before = group.dispatch_stats().submissions;
            let result = dispatch(group);
            self.stats.submissions += group.dispatch_stats().submissions - before;
            if let Err(error) = result {
                self.generation = failure_generation;
                self.stats.failed_frames += 1;
                return Err(error);
            }
        }
        self.stats.frames += 1;
        Ok(CameraBatchFrame {
            batch: self,
            stamp: CameraBatchStamp {
                owner: self.owner.clone(),
                generation: self.generation,
                sample_id,
            },
        })
    }

    fn find(&self, sensor_id: u64) -> Result<&CameraImageLayout> {
        self.images
            .iter()
            .find(|image| image.sensor_id == sensor_id)
            .ok_or_else(|| Error::Solver("camera batch: unknown sensor ID".into()))
    }
    fn check_engine(&self, engine: &Engine) -> Result<()> {
        engine.check_device()?;
        if engine
            .device()
            .is_none_or(|device| !Arc::ptr_eq(device, &self.device))
        {
            return Err(Error::Gpu(
                "camera batch belongs to a different engine".into(),
            ));
        }
        Ok(())
    }
    fn next_generation(&self) -> Result<u64> {
        self.generation
            .checked_add(1)
            .ok_or_else(|| Error::Solver("camera batch: generation exhausted".into()))
    }
}

struct GroupPlan {
    config: TiledCameraConfig,
    members: Vec<usize>,
}
struct Plan {
    groups: Vec<GroupPlan>,
    images: Vec<CameraImageLayout>,
}

impl Plan {
    fn storage_bytes(&self, limits: TiledCameraLimits, device: &wgpu::Limits) -> Result<u64> {
        let mut total = 0u64;
        for group in &self.groups {
            let usage = limits.check(
                group.config.width,
                group.config.height,
                group.members.len(),
                device,
            )?;
            total = total
                .checked_add(usage.total_bytes)
                .ok_or_else(|| Error::Solver("camera batch: array storage overflow".into()))?;
            if total > limits.max_storage_bytes {
                return Err(Error::Solver(
                    "camera batch: combined group array budget exceeded".into(),
                ));
            }
        }
        Ok(total)
    }
}

fn plan(specs: &[CameraSpec], limits: CameraBatchLimits, device: &wgpu::Limits) -> Result<Plan> {
    if limits.max_cameras == 0
        || limits.max_cameras > 1024
        || limits.max_groups == 0
        || limits.max_groups > 64
        || limits.max_total_pixels == 0
        || specs.len() > limits.max_cameras
    {
        return Err(Error::Solver(
            "camera batch: invalid or exceeded batch limits".into(),
        ));
    }
    let mut ids = HashSet::with_capacity(specs.len());
    let mut groups: Vec<GroupPlan> = Vec::new();
    let mut images = Vec::with_capacity(specs.len());
    let mut total = 0u64;
    for (index, spec) in specs.iter().enumerate() {
        if !ids.insert(spec.sensor_id) {
            return Err(Error::Solver("camera batch: duplicate sensor ID".into()));
        }
        validate_config(&spec.config)?;
        validate_camera_pose(spec.pose)?;
        let pixels = checked_pixel_count(spec.config.width, spec.config.height, 1, device)?;
        total = total
            .checked_add(pixels as u64)
            .ok_or_else(|| Error::Solver("camera batch: total pixel overflow".into()))?;
        if total > limits.max_total_pixels {
            return Err(Error::Solver(
                "camera batch: total pixel budget exceeded".into(),
            ));
        }
        let group_index =
            if let Some(group) = groups.iter().position(|group| group.config == spec.config) {
                group
            } else {
                if groups.len() == limits.max_groups {
                    return Err(Error::Solver("camera batch: group budget exceeded".into()));
                }
                groups.push(GroupPlan {
                    config: spec.config,
                    members: Vec::new(),
                });
                groups.len() - 1
            };
        let group = &mut groups[group_index];
        let camera_index = group.members.len();
        let end = checked_pixel_count(
            spec.config.width,
            spec.config.height,
            camera_index + 1,
            device,
        )?;
        group.members.push(index);
        images.push(CameraImageLayout {
            sensor_id: spec.sensor_id,
            group_index,
            camera_index: camera_index as u32,
            width: spec.config.width,
            height: spec.config.height,
            channels: spec.config.channels,
            pixels: end - pixels..end,
        });
    }
    Ok(Plan { groups, images })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::Quat;

    #[test]
    fn fixed_array_budget_sums_groups_before_construction() {
        let config = TiledCameraConfig {
            width: 8,
            height: 8,
            ..Default::default()
        };
        let specs = [
            CameraSpec {
                sensor_id: 1,
                config,
                pose: Transform::IDENTITY,
            },
            CameraSpec {
                sensor_id: 2,
                config: TiledCameraConfig {
                    fov_y_radians: 0.8,
                    ..config
                },
                pose: Transform::IDENTITY,
            },
        ];
        let device = wgpu::Limits::default();
        let planned = plan(&specs, Default::default(), &device).unwrap();
        let one = crate::CameraStorageUsage::for_dimensions(8, 8, 1)
            .unwrap()
            .total_bytes;
        let exact = TiledCameraLimits {
            max_storage_bytes: one * 2,
            ..Default::default()
        };
        assert_eq!(planned.storage_bytes(exact, &device).unwrap(), one * 2);
        assert!(
            planned
                .storage_bytes(
                    TiledCameraLimits {
                        max_storage_bytes: one * 2 - 1,
                        ..exact
                    },
                    &device
                )
                .is_err()
        );
        let shared = plan(
            &[specs[0], CameraSpec { config, ..specs[1] }],
            Default::default(),
            &device,
        )
        .unwrap();
        assert_eq!(
            shared.storage_bytes(exact, &device).unwrap(),
            crate::CameraStorageUsage::for_dimensions(8, 8, 2)
                .unwrap()
                .total_bytes
        );
        assert!(shared.storage_bytes(exact, &device).unwrap() < one * 2);
        assert_eq!(
            plan(&[], Default::default(), &device)
                .unwrap()
                .storage_bytes(
                    TiledCameraLimits {
                        max_storage_bytes: 0,
                        ..exact
                    },
                    &device
                )
                .unwrap(),
            0
        );
    }

    fn spec(id: u64) -> CameraSpec {
        CameraSpec {
            sensor_id: id,
            config: TiledCameraConfig {
                width: 3,
                height: 5,
                ..Default::default()
            },
            pose: Transform::IDENTITY,
        }
    }

    #[test]
    fn nonadjacent_members_keep_input_order_ids_and_group_local_ranges() {
        let mut specs = [spec(9), spec(3), spec(0), spec(u64::MAX)];
        specs[1].config.width = 7;
        specs[2].pose.position = Vec3::X;
        specs[3].config = specs[1].config;
        let result = plan(&specs, Default::default(), &wgpu::Limits::default()).unwrap();
        assert_eq!(result.groups.len(), 2);
        assert_eq!(result.groups[0].members, [0, 2]);
        assert_eq!(result.groups[1].members, [1, 3]);
        for (i, (group, camera, range)) in
            [(0, 0, 0..15), (1, 0, 0..35), (0, 1, 15..30), (1, 1, 35..70)]
                .into_iter()
                .enumerate()
        {
            let image = &result.images[i];
            assert_eq!(image.sensor_id, specs[i].sensor_id);
            assert_eq!(
                (image.group_index, image.camera_index, image.pixels.clone()),
                (group, camera, range)
            );
        }
    }

    #[test]
    fn every_configuration_field_participates_in_grouping() {
        let base = spec(0);
        let changes: &[fn(&mut TiledCameraConfig)] = &[
            |c| c.width += 1,
            |c| c.height += 1,
            |c| c.fov_y_radians *= 0.9,
            |c| c.channels = CameraChannels::DEPTH,
            |c| c.ambient += Vec3::X,
            |c| c.light_dir += Vec3::Z,
            |c| c.light_color += Vec3::Y,
            |c| c.max_distance += 1.0,
            |c| c.near += 1.0,
            |c| c.optics.focal_length_mm += 1.0,
            |c| c.optics.sensor_width_mm += 1.0,
            |c| c.optics.lens_diameter_m += 0.1,
            |c| c.optics.focal_distance_m += 1.0,
            |c| c.optics.aperture_samples += 1,
        ];
        for change in changes {
            let mut other = spec(1);
            change(&mut other.config);
            let result =
                plan(&[base, other], Default::default(), &wgpu::Limits::default()).unwrap();
            assert_eq!(result.groups.len(), 2, "{:?}", other.config);
        }
    }

    #[test]
    fn budgets_validation_and_empty_plan_precede_gpu_allocations() {
        let limits = CameraBatchLimits::default();
        let device = wgpu::Limits::default();
        assert!(plan(&[], limits, &device).unwrap().groups.is_empty());
        for invalid in [
            CameraBatchLimits {
                max_cameras: 0,
                ..limits
            },
            CameraBatchLimits {
                max_cameras: 1025,
                ..limits
            },
            CameraBatchLimits {
                max_groups: 0,
                ..limits
            },
            CameraBatchLimits {
                max_groups: 65,
                ..limits
            },
            CameraBatchLimits {
                max_total_pixels: 0,
                ..limits
            },
        ] {
            assert!(plan(&[], invalid, &device).is_err());
        }
        let pair = [spec(0), spec(1)];
        assert!(
            plan(
                &pair,
                CameraBatchLimits {
                    max_cameras: 1,
                    ..limits
                },
                &device
            )
            .is_err()
        );
        assert!(
            plan(
                &pair,
                CameraBatchLimits {
                    max_total_pixels: 29,
                    ..limits
                },
                &device
            )
            .is_err()
        );
        assert!(
            plan(
                &pair,
                CameraBatchLimits {
                    max_total_pixels: 30,
                    ..limits
                },
                &device
            )
            .is_ok()
        );
        assert!(plan(&[spec(0), spec(0)], limits, &device).is_err());
        let mut other = spec(1);
        other.config.width += 1;
        assert!(
            plan(
                &[spec(0), other],
                CameraBatchLimits {
                    max_groups: 1,
                    ..limits
                },
                &device
            )
            .is_err()
        );
        for invalid in [
            Transform::new(Vec3::splat(f64::NAN), Quat::IDENTITY),
            Transform::new(Vec3::splat(f64::MAX), Quat::IDENTITY),
            Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
        ] {
            other = spec(1);
            other.pose = invalid;
            assert!(plan(&[other], limits, &device).is_err());
        }
        for invalid in [0, u32::MAX] {
            other = spec(1);
            other.config.width = invalid;
            assert!(plan(&[other], limits, &device).is_err());
        }
        other = spec(1);
        other.config.channels = CameraChannels(32);
        assert!(plan(&[other], limits, &device).is_err());
        other.config = spec(1).config;
        other.config.ambient.x = f64::NAN;
        assert!(plan(&[other], limits, &device).is_err());
    }

    #[test]
    fn combined_group_capacity_is_checked_not_only_individual_images() {
        let specs = [spec(0), spec(1)];
        let device = wgpu::Limits {
            max_storage_buffer_binding_size: 15 * 16,
            ..Default::default()
        };
        assert!(plan(&specs[..1], Default::default(), &device).is_ok());
        assert!(plan(&specs, Default::default(), &device).is_err());
        let specs = [
            CameraSpec {
                config: TiledCameraConfig {
                    width: 8,
                    height: 8,
                    ..spec(0).config
                },
                ..spec(0)
            },
            CameraSpec {
                config: TiledCameraConfig {
                    width: 8,
                    height: 8,
                    ..spec(1).config
                },
                ..spec(1)
            },
        ];
        let device = wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..Default::default()
        };
        assert!(plan(&specs[..1], Default::default(), &device).is_ok());
        assert!(plan(&specs, Default::default(), &device).is_err());
    }

    #[test]
    #[ignore = "needs a real GPU"]
    fn failed_partial_frame_invalidates_receipts_and_identity_exhaustion_is_checked() {
        let engine = Engine::new_gpu().expect("GPU required");
        let mut specs = [spec(0), spec(1)];
        specs[1].config.width += 1;
        let model = molla_sim::ModelBuilder::new().build().unwrap();
        let state = model.state().unwrap();
        let mut batch = CameraBatch::new(&engine, &specs, Default::default()).unwrap();
        let old = batch
            .dispatch(&engine, &model, &state)
            .unwrap()
            .stamp()
            .clone();
        let mut count = 0;
        assert!(
            batch
                .dispatch_groups(&engine, |camera| {
                    count += 1;
                    if count == 2 {
                        return Err(Error::Solver("injected second-group failure".into()));
                    }
                    camera.dispatch(&engine, &model, &state)
                })
                .is_err()
        );
        assert_eq!(
            batch.stats(),
            CameraBatchStats {
                frames: 1,
                encoded_frames: 0,
                submissions: 3,
                failed_frames: 1,
                reconfigurations: 0
            }
        );
        assert!(!batch.is_current(&old));
        let next = batch
            .dispatch(&engine, &model, &state)
            .unwrap()
            .stamp()
            .clone();
        assert_eq!(next.sample_id(), 2);
        assert_eq!(next.generation(), 1);
        let stats = batch.stats();
        batch.next_sample = u64::MAX;
        assert!(batch.dispatch(&engine, &model, &state).is_err());
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let mut pool = batch
            .output_pool(
                &engine,
                SensorFramePoolConfig {
                    slots: 1,
                    max_total_bytes: 1 << 20,
                },
            )
            .unwrap();
        let mut commands = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            batch
                .encode_scene(&mut commands, &engine, &scene, &state, &mut pool, None)
                .is_err()
        );
        assert!(commands.submit().unwrap().index.is_none());
        for i in 0..batch.group_count() {
            assert_eq!(pool.group_stats(i).unwrap().reserved_frames, 0);
        }
        assert_eq!(batch.stats(), stats);
        assert!(batch.is_current(&next));
        batch.next_sample = 3;
        batch.generation = u64::MAX;
        assert!(batch.dispatch(&engine, &model, &state).is_err());
        assert!(batch.reset().is_err());
        assert!(batch.reconfigure(&engine, &[]).is_err());
        assert_eq!(batch.next_sample, 3);
        assert_eq!(batch.stats(), stats);
    }
}
