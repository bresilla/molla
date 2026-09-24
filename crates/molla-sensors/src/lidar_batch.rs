//! Device-resident terrestrial/aerial ray layouts and sub-ray return selection.

use crate::{
    AerialScanMetadata, RayDeviceHits, Raycaster, ScanMetadata,
    device_results::{
        EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool,
        SensorFramePoolConfig,
    },
    dispatch::BindingsCache,
    readback::{DeviceBuffer, ReadbackStamp, ReadbackStats},
};
use crate::{SceneStamp, SensorScene};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{LidarScanBindings, LidarScanDescriptor, LidarScanKernel};
use molla_math::{Real, Vec3};
use molla_sim::{Model, State};
use molla_storage::BufferUsage;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum LidarPattern {
    Terrestrial(ScanMetadata),
    Aerial(AerialScanMetadata),
}

#[derive(Debug, Clone, Copy)]
pub struct LidarScan {
    pub sensor_id: u64,
    pub pattern: LidarPattern,
}

/// Contiguous flattened scan ranges, in caller order; sub-rays are pulse-major.
#[derive(Debug, Clone, Copy)]
pub struct LidarScanRange {
    pub scan: LidarScan,
    pub ray_offset: u32,
    pub ray_count: u32,
    pub pulse_offset: u32,
    pub pulse_count: u32,
    pub subray_count: u32,
}

/// First/last contain global ray indices, or -1 for an all-miss pulse.
/// Tied nearest selects the first sub-ray; tied farthest selects the last.
/// Submit consumers or enqueue snapshots before dispatching another batch.
pub struct LidarDeviceBatch<'a> {
    pub hits: RayDeviceHits<'a>,
    pub pulse_count: usize,
    pub first_ray: &'a DeviceBuffer<i32>,
    pub last_ray: &'a DeviceBuffer<i32>,
    pub hit_count: &'a DeviceBuffer<u32>,
    pub scans: &'a [LidarScanRange],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LidarBatchStats {
    pub batches: u64,
    pub encoded_batches: u64,
    pub submissions: u64,
    pub descriptor_upload_bytes: u64,
    pub binding_preparations: u64,
}

/// Channels: origin, direction, distance, shape, point, normal, first, last, hit count.
pub struct EncodedLidarBatch {
    pub frame: EncodedSensorFrame,
    pub ray_count: usize,
    pub pulse_count: usize,
    pub scans: Vec<LidarScanRange>,
    pub scene_stamp: SceneStamp,
}

/// Fixed-capacity multi-scan pipeline; at most 1024 scans and 1024 sub-rays per pulse.
/// This selects returns among divergent sub-rays, not multiple intersections on one ray.
pub struct LidarBatch {
    device: Arc<wgpu::Device>,
    raycaster: Raycaster,
    kernel: LidarScanKernel,
    direct: BindingsCache<LidarScanBindings>,
    descriptors: wgpu::Buffer,
    host_descriptors: Vec<LidarScanDescriptor>,
    ranges: Vec<LidarScanRange>,
    first: DeviceBuffer<i32>,
    last: DeviceBuffer<i32>,
    count: DeviceBuffer<u32>,
    max_scans: usize,
    max_rays: u32,
    stats: LidarBatchStats,
}

impl LidarBatch {
    pub fn new(engine: &Engine, max_scans: usize, max_rays: u32) -> Result<Self> {
        engine.check_device()?;
        if max_scans == 0 || max_scans > 1024 || max_rays == 0 || max_rays > i32::MAX as u32 {
            return Err(Error::Solver(
                "lidar batch: invalid scan/ray capacity".into(),
            ));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("lidar batch requires GPU".into()))?
            .clone();
        let descriptor_bytes = max_scans as u64 * 128;
        if descriptor_bytes > device.limits().max_storage_buffer_binding_size as u64
            || descriptor_bytes > device.limits().max_buffer_size
        {
            return Err(Error::Solver(
                "lidar batch: descriptors exceed GPU capacity".into(),
            ));
        }
        let raycaster = Raycaster::new(engine, max_rays)?;
        let first = DeviceBuffer::from_host(
            engine,
            vec![-1; max_rays as usize],
            BufferUsage::StorageReadWrite,
        )?;
        let last = DeviceBuffer::from_host(
            engine,
            vec![-1; max_rays as usize],
            BufferUsage::StorageReadWrite,
        )?;
        let count = DeviceBuffer::from_host(
            engine,
            vec![0_u32; max_rays as usize],
            BufferUsage::StorageReadWrite,
        )?;
        let descriptors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::lidar descriptors"),
            size: descriptor_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hits = raycaster.device_hits(0);
        let kernel = LidarScanKernel::new(
            &device,
            &descriptors,
            [
                hits.origins.device_buffer()?,
                hits.directions.device_buffer()?,
            ],
            [
                hits.distance.device_buffer()?,
                hits.shape.device_buffer()?,
                hits.point.device_buffer()?,
                hits.normal.device_buffer()?,
            ],
            [
                first.device_buffer()?,
                last.device_buffer()?,
                count.device_buffer()?,
            ],
        );
        Ok(Self {
            device,
            raycaster,
            kernel,
            direct: BindingsCache::with_capacity(128),
            descriptors,
            host_descriptors: Vec::with_capacity(max_scans),
            ranges: Vec::with_capacity(max_scans),
            first,
            last,
            count,
            max_scans,
            max_rays,
            stats: LidarBatchStats {
                binding_preparations: 1,
                ..Default::default()
            },
        })
    }

    fn output_specs(&self) -> [SensorChannelSpec; 9] {
        let n = self.max_rays as usize;
        [
            SensorChannelSpec::of::<Vec3>(n),
            SensorChannelSpec::of::<Vec3>(n),
            SensorChannelSpec::of::<Real>(n),
            SensorChannelSpec::of::<i32>(n),
            SensorChannelSpec::of::<Vec3>(n),
            SensorChannelSpec::of::<Vec3>(n),
            SensorChannelSpec::of::<i32>(n),
            SensorChannelSpec::of::<i32>(n),
            SensorChannelSpec::of::<u32>(n),
        ]
    }

    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        engine.check_device()?;
        self.first.check_device()?;
        if !Arc::ptr_eq(engine.device().expect("checked engine"), &self.device) {
            return Err(Error::Gpu("lidar batch: foreign engine".into()));
        }
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    /// Generate, query and select directly into leased outputs in one command batch.
    /// The sensor, scene and state remain borrowed until submission or cancellation.
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sim::State;
    /// use molla_sensors::{LidarBatch, SensorCommandBatch, SensorFramePool, SensorScene};
    /// fn reuse(engine: &Engine, sensor: &mut LidarBatch, scene: &SensorScene, state: &State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     sensor.encode_scene(&mut batch, engine, scene, state, &[], pool, Default::default()).unwrap();
    ///     sensor.clear_mesh_cache();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sim::State;
    /// use molla_sensors::{LidarBatch, SensorCommandBatch, SensorFramePool, SensorScene};
    /// fn mutate(engine: &Engine, sensor: &mut LidarBatch, scene: &SensorScene, state: &mut State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     sensor.encode_scene(&mut batch, engine, scene, state, &[], pool, Default::default()).unwrap();
    ///     state.body_q.host_mut().unwrap()[0] = molla_math::Transform::IDENTITY;
    ///     batch.submit().unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sim::{Model, State};
    /// use molla_sensors::{LidarBatch, SensorCommandBatch, SensorFramePool, SensorScene};
    /// fn replace(engine: &Engine, sensor: &mut LidarBatch, scene: &mut SensorScene, model: &Model, state: &State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     sensor.encode_scene(&mut batch, engine, scene, state, &[], pool, Default::default()).unwrap();
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
        scans: &[LidarScan],
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedLidarBatch>> {
        batch.check_engine(engine)?;
        scene.check_engine(engine)?;
        let device = engine.device().expect("checked engine");
        if !Arc::ptr_eq(device, &self.device) {
            return Err(Error::Gpu("lidar batch: foreign engine".into()));
        }
        if stamp
            .simulation_time
            .is_some_and(|time| !time.is_finite() || time < 0.0)
        {
            return Err(Error::Solver("lidar batch: invalid timestamp".into()));
        }
        let (rays, pulses, max_distance) = self.plan(scans)?;
        if rays != 0 && scene.model.body_count != 0 {
            if state.body_q.len() != scene.model.body_count {
                return Err(Error::Solver(
                    "lidar batch: state/model body counts differ".into(),
                ));
            }
            state.body_q.device_buffer_for(device)?;
        }
        let specs = self.output_specs();
        let ranges = std::array::from_fn::<_, 9, _>(|i| {
            0..if i < 6 {
                rays as usize
            } else {
                pulses as usize
            }
        });
        let frame = batch.encode_output(
            pool,
            &specs,
            &ranges,
            stamp,
            |device, queue, encoder, outputs| {
                if rays == 0 {
                    return Ok(0);
                }
                queue.write_buffer(
                    &self.descriptors,
                    0,
                    bytemuck::cast_slice(&self.host_descriptors),
                );
                let buffers = std::array::from_fn(|i| &outputs[i]);
                let (bindings, prepared) = self.direct.get_or_prepare(
                    |bindings| bindings.matches(buffers),
                    || self.kernel.prepare(device, buffers),
                );
                self.stats.binding_preparations += u64::from(prepared);
                self.kernel.encode_generate_prepared(
                    queue,
                    encoder,
                    bindings,
                    scans.len() as u32,
                    rays,
                    pulses,
                );
                let queries = self.raycaster.encode_owned_scene(
                    device,
                    queue,
                    encoder,
                    scene,
                    state,
                    outputs,
                    rays,
                    max_distance,
                )?;
                self.kernel
                    .encode_reduce_prepared(encoder, bindings, pulses);
                Ok(queries + 2)
            },
        )?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        let bytes = if rays == 0 {
            0
        } else {
            scans.len() as u64 * 128
        };
        self.stats.encoded_batches += 1;
        self.stats.descriptor_upload_bytes += bytes;
        batch.record_input_upload(bytes);
        Ok(Some(EncodedLidarBatch {
            frame,
            ray_count: rays as usize,
            pulse_count: pulses as usize,
            scans: self.ranges.clone(),
            scene_stamp: scene.stamp(),
        }))
    }

    /// Generate, query, clip to each scan's range, and select first/last sub-rays on GPU.
    /// Only compact scan descriptors are uploaded; no CPU ray vectors are constructed.
    pub fn dispatch(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        scans: &[LidarScan],
    ) -> Result<LidarDeviceBatch<'_>> {
        engine.check_device()?;
        self.dispatch_for_model(engine, model, state, scans, None)
    }

    pub fn dispatch_scene(
        &mut self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
        scans: &[LidarScan],
    ) -> Result<LidarDeviceBatch<'_>> {
        scene.check_engine(engine)?;
        self.dispatch_for_model(engine, &scene.model, state, scans, Some(scene))
    }

    fn dispatch_for_model(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        scans: &[LidarScan],
        scene: Option<&SensorScene>,
    ) -> Result<LidarDeviceBatch<'_>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("lidar batch requires GPU".into()))?;
        if !Arc::ptr_eq(device, &self.device) {
            return Err(Error::Gpu("lidar batch: foreign engine".into()));
        }
        let (rays, pulses, max_distance) = self.plan(scans)?;
        if rays > 0 {
            let queue = engine
                .queue()
                .ok_or_else(|| Error::Gpu("lidar batch requires queue".into()))?;
            let bytes = bytemuck::cast_slice(&self.host_descriptors);
            queue.write_buffer(&self.descriptors, 0, bytes);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::lidar complete batch"),
            });
            self.kernel
                .encode_generate(queue, &mut encoder, scans.len() as u32, rays, pulses);
            self.raycaster.encode_generated(
                engine,
                &mut encoder,
                model,
                state,
                rays as usize,
                max_distance as Real,
                scene,
            )?;
            self.kernel.encode_reduce(&mut encoder, pulses);
            queue.submit([encoder.finish()]);
            self.stats.submissions += 1;
            self.first.mark_device_dirty();
            self.last.mark_device_dirty();
            self.count.mark_device_dirty();
            self.stats.batches += 1;
            self.stats.descriptor_upload_bytes += bytes.len() as u64;
        }
        self.raycaster
            .set_scene_stamp(scene.map(SensorScene::stamp));
        Ok(LidarDeviceBatch {
            hits: self.raycaster.device_hits(rays as usize),
            pulse_count: pulses as usize,
            first_ray: &self.first,
            last_ray: &self.last,
            hit_count: &self.count,
            scans: &self.ranges,
        })
    }

    fn plan(&mut self, scans: &[LidarScan]) -> Result<(u32, u32, f32)> {
        if scans.len() > self.max_scans {
            return Err(Error::Solver("lidar batch: too many scans".into()));
        }
        self.host_descriptors.clear();
        self.ranges.clear();
        let mut rays = 0u32;
        let mut pulses = 0u32;
        let mut max_distance = 0.0f32;
        for &scan in scans {
            let (descriptor, range) = prepare(scan, rays, pulses)?;
            rays = rays
                .checked_add(range.ray_count)
                .filter(|&n| n <= self.max_rays)
                .ok_or_else(|| Error::Solver("lidar batch: rays exceed capacity".into()))?;
            pulses = pulses
                .checked_add(range.pulse_count)
                .ok_or_else(|| Error::Solver("lidar batch: pulse count overflow".into()))?;
            max_distance = max_distance.max(descriptor.origin[3]);
            self.host_descriptors.push(descriptor);
            self.ranges.push(range);
        }
        Ok((rays, pulses, max_distance))
    }

    pub fn stats(&self) -> LidarBatchStats {
        self.stats
    }

    pub fn readback_stats(&self) -> ReadbackStats {
        self.raycaster.readback_stats()
    }

    pub fn query_stats(&self) -> crate::raycast::RayQueryStats {
        self.raycaster.query_stats()
    }

    /// Invalidate static mesh snapshots after geometry/model changes.
    pub fn clear_mesh_cache(&mut self) {
        self.raycaster.clear_mesh_cache();
    }
}

fn vector(v: Vec3, w: Real) -> Result<[f32; 4]> {
    let value = [v.x as f32, v.y as f32, v.z as f32, w as f32];
    if value.iter().any(|v| !v.is_finite()) {
        return Err(Error::Solver(
            "lidar batch: metadata exceeds finite f32".into(),
        ));
    }
    Ok(value)
}

fn prepare(
    scan: LidarScan,
    ray_offset: u32,
    pulse_offset: u32,
) -> Result<(LidarScanDescriptor, LidarScanRange)> {
    let mut descriptor = LidarScanDescriptor::default();
    let (rays, pulses, subrays) = match scan.pattern {
        LidarPattern::Terrestrial(s) => {
            s.validate()?;
            let (f, u, r) = s.local_frame();
            if f.dot(u).abs() > 1e-6 || f.cross(u).length_squared() < 0.99 {
                return Err(Error::Solver(
                    "lidar batch: forward/up do not define an orthonormal frame".into(),
                ));
            }
            descriptor.grid = [s.n_phi, s.n_theta, 0, 0];
            descriptor.origin = vector(s.origin, s.max_distance)?;
            descriptor.forward = vector(f, s.beam_divergence)?;
            descriptor.up = vector(u, 0.0)?;
            descriptor.right = vector(r, 0.0)?;
            descriptor.angles = [
                s.theta_range.0 as f32,
                s.theta_range.1 as f32,
                s.phi_range.0 as f32,
                s.phi_range.1 as f32,
            ];
            if descriptor.angles.iter().any(|v| !v.is_finite())
                || !(descriptor.angles[1] - descriptor.angles[0]).is_finite()
                || !(descriptor.angles[3] - descriptor.angles[2]).is_finite()
            {
                return Err(Error::Solver(
                    "lidar batch: angular range exceeds finite f32".into(),
                ));
            }
            (
                s.checked_ray_count()?,
                s.cell_count(),
                s.effective_subray_count(),
            )
        }
        LidarPattern::Aerial(s) => {
            s.validate()?;
            vector(s.footprint_max, 0.0)?;
            descriptor.grid[2] = 1;
            descriptor.origin = vector(
                Vec3::new(s.footprint_min.x, s.footprint_max.y, s.footprint_min.z),
                s.max_distance,
            )?;
            descriptor.extent = vector(
                Vec3::new(
                    s.footprint_max.x - s.footprint_min.x,
                    s.footprint_max.z - s.footprint_min.z,
                    0.0,
                ),
                0.0,
            )?;
            descriptor.forward[3] = s.beam_divergence as f32;
            descriptor.up[3] = s.cone_angle as f32;
            (
                s.checked_ray_count()?,
                s.checked_pulse_count()?,
                s.effective_subray_count(),
            )
        }
    };
    if descriptor.origin[3] <= 0.0 || subrays > 1024 {
        return Err(Error::Solver(
            "lidar batch: range underflows f32 or sub-rays exceed 1024".into(),
        ));
    }
    descriptor.counts = [ray_offset, pulse_offset, pulses, subrays];
    Ok((
        descriptor,
        LidarScanRange {
            scan,
            ray_offset,
            ray_count: rays,
            pulse_offset,
            pulse_count: pulses,
            subray_count: subrays,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_reject_invalid_gpu_metadata_without_allocation() {
        let mut s = ScanMetadata::new(Vec3::ZERO, 1, 1);
        let check = |s| {
            prepare(
                LidarScan {
                    sensor_id: 0,
                    pattern: LidarPattern::Terrestrial(s),
                },
                0,
                0,
            )
        };
        s.origin.x = f64::MAX;
        assert!(check(s).is_err());
        s.origin = Vec3::ZERO;
        s.max_distance = 1e-100;
        assert!(check(s).is_err());
        s.max_distance = 10.0;
        s.subray_count = 1025;
        assert!(check(s).is_err());
        s.subray_count = 1;
        s.forward = s.up;
        assert!(check(s).is_err());
        s.forward = Vec3::Z;
        s.theta_range = (-f64::MAX, f64::MAX);
        assert!(check(s).is_err());
    }
}
