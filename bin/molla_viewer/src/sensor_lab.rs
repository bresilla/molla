use super::{
    CameraChannels, ImuMount, ImuReading, ImuSensor, MVec3, SensorCommandBatch, SensorPreview, Sim,
    TiledCamera, TiledCameraConfig, mp, preview_pool_config,
};
use molla::core::{Error, Result};
use molla::sensors::{
    LidarBatch, LidarPattern, LidarScan, ScanMetadata, ScheduledSample, SensorFrame, SensorScene,
    SensorSchedule, SensorScheduler,
};
use std::time::{Duration, Instant};

pub(super) const CAMERA_WIDTH: u32 = 160;
pub(super) const CAMERA_HEIGHT: u32 = 120;
const MAX_RAYS: u32 = 768;

#[derive(Clone, Copy, PartialEq)]
pub(super) struct LabSettings {
    pub rates: [u32; 3],
    pub columns: u32,
    pub lidar_fov: f32,
    pub range: f32,
    pub camera_fov: f32,
}

impl Default for LabSettings {
    fn default() -> Self {
        Self {
            rates: [10, 5, 30],
            columns: 65,
            lidar_fov: 1.4,
            range: 30.0,
            camera_fov: 55f32.to_radians(),
        }
    }
}

impl LabSettings {
    fn validate(self) -> Result<()> {
        if self.rates.iter().any(|&hz| !(1..=60).contains(&hz))
            || !(9..=129).contains(&self.columns)
            || !self.lidar_fov.is_finite()
            || !(0.1..=2.5).contains(&self.lidar_fov)
            || !self.camera_fov.is_finite()
            || !(0.2..=2.0).contains(&self.camera_fov)
            || !self.range.is_finite()
            || !(2.0..=100.0).contains(&self.range)
        {
            return Err(Error::Solver(
                "sensor lab settings outside supported bounds".into(),
            ));
        }
        Ok(())
    }

    fn schedules(self, defer_imu: bool) -> [SensorSchedule; 3] {
        std::array::from_fn(|index| {
            let period = Duration::from_secs_f64(1.0 / f64::from(self.rates[index]));
            SensorSchedule {
                sensor_id: index as u64,
                period,
                phase: if index == 2 && defer_imu {
                    period
                } else {
                    Duration::ZERO
                },
            }
        })
    }

    pub fn scan(self) -> ScanMetadata {
        let mut meta = ScanMetadata::new(MVec3::new(0.0, 1.0, 5.0), 5, self.columns);
        meta.forward = -MVec3::Z;
        meta.theta_range = (
            std::f64::consts::FRAC_PI_2 - 0.35,
            std::f64::consts::FRAC_PI_2 + 0.35,
        );
        meta.phi_range = (
            -f64::from(self.lidar_fov) * 0.5,
            f64::from(self.lidar_fov) * 0.5,
        );
        meta.max_distance = f64::from(self.range);
        meta
    }

    pub fn camera(self) -> TiledCameraConfig {
        TiledCameraConfig {
            width: CAMERA_WIDTH,
            height: CAMERA_HEIGHT,
            channels: CameraChannels::COLOR,
            fov_y_radians: self.camera_fov,
            max_distance: 40.0,
            ..Default::default()
        }
    }
}

pub(super) struct LabRuntime {
    pub scene: SensorScene,
    pub lidar: LidarBatch,
    pub camera: TiledCamera,
    imu: ImuSensor,
    scheduler: SensorScheduler,
    settings: LabSettings,
    pub displays: [SensorPreview; 3],
    pub skipped: [u64; 3],
    pub cloud: Vec<(MVec3, i32)>,
    pub reading: Option<ImuReading>,
    retained: [Option<SensorFrame>; 2],
    enqueued: [Option<u64>; 2],
}

impl LabRuntime {
    pub fn new(sim: &Sim, settings: LabSettings) -> Result<Self> {
        settings.validate()?;
        let scene = SensorScene::from_model(&sim.engine, &sim.model, Default::default())?;
        let lidar = LidarBatch::new(&sim.engine, 1, MAX_RAYS)?;
        let mut camera = TiledCamera::new(&sim.engine, settings.camera(), 1)?;
        camera.set_camera_look_at(
            0,
            MVec3::new(0.0, 2.0, 6.0),
            MVec3::new(0.0, 1.0, 0.0),
            MVec3::Y,
        )?;
        let imu = ImuSensor::new(
            &sim.engine,
            &sim.model,
            &[ImuMount {
                body: mp::BodyId::from_index(0),
                local_pose: mp::Transform::IDENTITY,
            }],
        )?;
        let lp = lidar.output_pool(&sim.engine, preview_pool_config(u64::from(MAX_RAYS) * 84)?)?;
        let pixels = u64::from(CAMERA_WIDTH) * u64::from(CAMERA_HEIGHT);
        let cp = camera.output_pool(&sim.engine, preview_pool_config(pixels * 32)?)?;
        let ip = imu.output_pool(&sim.engine, preview_pool_config(32)?)?;
        let mut displays = [
            SensorPreview::new(&sim.engine, lp, u64::from(MAX_RAYS) * 20)?,
            SensorPreview::new(&sim.engine, cp, pixels * 4)?,
            SensorPreview::new(&sim.engine, ip, 32)?,
        ];
        for (preview, rate) in displays.iter_mut().zip(settings.rates) {
            preview.cadence_hz = Some(f64::from(rate));
        }
        let scheduler = SensorScheduler::new(
            Duration::from_secs_f64(sim.sensor_clock.time()),
            sim.sensor_clock.revision,
            &settings.schedules(false),
        )?;
        Ok(Self {
            scene,
            lidar,
            camera,
            imu,
            scheduler,
            settings,
            displays,
            skipped: [0; 3],
            cloud: Vec::new(),
            reading: None,
            retained: [None, None],
            enqueued: [None, None],
        })
    }

    pub fn update(
        &mut self,
        sim: &Sim,
        settings: LabSettings,
        visible: bool,
    ) -> Result<Option<Vec<u32>>> {
        settings.validate()?;
        let mut color = None;
        if let Some(frame) = self.displays[0].poll_latest(sim.sensor_clock)? {
            self.cloud.clear();
            self.cloud.extend(
                frame
                    .channel_iter::<MVec3>(0)?
                    .zip(frame.channel_iter::<i32>(1)?),
            );
        }
        if let Some(frame) = self.displays[1].poll_latest(sim.sensor_clock)? {
            color = Some(frame.channel::<u32>(0)?);
        }
        if let Some(frame) = self.displays[2].poll_latest(sim.sensor_clock)? {
            self.reading = Some(ImuReading {
                accelerometer: frame
                    .channel_iter::<MVec3>(0)?
                    .next()
                    .ok_or_else(|| Error::Storage("sensor lab empty accelerometer".into()))?,
                gyroscope: frame
                    .channel_iter::<MVec3>(1)?
                    .next()
                    .ok_or_else(|| Error::Storage("sensor lab empty gyroscope".into()))?,
            });
        }
        let now = Duration::from_secs_f64(sim.sensor_clock.time());
        if settings != self.settings {
            self.camera.set_config(settings.camera())?;
            self.scheduler.replace(
                now,
                sim.sensor_clock.revision,
                &settings.schedules(self.displays[2].submitted_tick == Some(sim.sensor_clock.tick)),
            )?;
            self.settings = settings;
            for (preview, rate) in self.displays.iter_mut().zip(settings.rates) {
                preview.cadence_hz = Some(f64::from(rate));
            }
        }
        let mut due: [Option<ScheduledSample>; 3] = std::array::from_fn(|_| None);
        for (slot, sample) in due.iter_mut().zip(self.scheduler.advance(now)?) {
            *slot = Some(sample.clone());
        }
        for request in due.into_iter().flatten() {
            let index = request.sensor_id as usize;
            self.skipped[index] += request.skipped_samples;
            if !self.displays[index].begin_sample(request.sample_id)? {
                continue;
            }
            match index {
                0 => self.sample_lidar(sim, &request)?,
                1 => self.sample_camera(sim, &request, visible)?,
                2 => self.sample_imu(sim, &request, visible)?,
                _ => unreachable!(),
            }
        }
        if visible {
            self.enqueue_retained()?;
        }
        Ok(color)
    }

    fn sample_lidar(&mut self, sim: &Sim, request: &ScheduledSample) -> Result<()> {
        let preview = &mut self.displays[0];
        let start = Instant::now();
        let mut batch = SensorCommandBatch::new(&sim.engine)?;
        let timed = preview.begin_gpu(&mut batch, &sim.engine)?;
        let sample = self
            .lidar
            .encode_scene(
                &mut batch,
                &sim.engine,
                &self.scene,
                &sim.state_a,
                &[LidarScan {
                    sensor_id: 0,
                    pattern: LidarPattern::Terrestrial(self.settings.scan()),
                }],
                &mut preview.pool,
                request.readback_stamp(),
            )?
            .ok_or_else(|| {
                Error::Storage("sensor lab LiDAR pool unavailable after preflight".into())
            })?;
        preview.end_gpu(&mut batch, &sim.engine, timed)?;
        let submission = batch.submit()?;
        preview.submitted(sim.sensor_clock.tick, start);
        preview.collect_gpu(request.readback_stamp(), timed)?;
        let frame = submission.frame(&sample.frame)?;
        preview.enqueue(
            &[
                frame
                    .channel::<MVec3>(4)?
                    .readback_source(0..sample.ray_count)?,
                frame
                    .channel::<i32>(3)?
                    .readback_source(0..sample.ray_count)?,
            ],
            frame.stamp(),
        )?;
        Ok(())
    }

    fn sample_camera(&mut self, sim: &Sim, request: &ScheduledSample, visible: bool) -> Result<()> {
        let preview = &mut self.displays[1];
        let start = Instant::now();
        let mut batch = SensorCommandBatch::new(&sim.engine)?;
        let timed = visible && preview.begin_gpu(&mut batch, &sim.engine)?;
        let sample = self
            .camera
            .encode_scene(
                &mut batch,
                &sim.engine,
                &self.scene,
                &sim.state_a,
                &mut preview.pool,
                request.readback_stamp(),
            )?
            .ok_or_else(|| {
                Error::Storage("sensor lab camera pool unavailable after preflight".into())
            })?;
        preview.end_gpu(&mut batch, &sim.engine, timed)?;
        let submission = batch.submit()?;
        preview.submitted(sim.sensor_clock.tick, start);
        preview.collect_gpu(request.readback_stamp(), timed)?;
        let frame = submission.frame(&sample.frame)?;
        self.retained[0] = Some(frame.clone());
        Ok(())
    }

    fn sample_imu(&mut self, sim: &Sim, request: &ScheduledSample, visible: bool) -> Result<()> {
        let preview = &mut self.displays[2];
        let start = Instant::now();
        let mut batch = SensorCommandBatch::new(&sim.engine)?;
        let timed = visible && preview.begin_gpu(&mut batch, &sim.engine)?;
        let sample = self
            .imu
            .encode(
                &mut batch,
                &sim.engine,
                &sim.model,
                &sim.state_a,
                preview.sample_dt(sim.sensor_clock.tick),
                &mut preview.pool,
                request.readback_stamp(),
            )?
            .ok_or_else(|| {
                Error::Storage("sensor lab IMU pool unavailable after preflight".into())
            })?;
        preview.end_gpu(&mut batch, &sim.engine, timed)?;
        let submission = batch.submit()?;
        preview.submitted(sim.sensor_clock.tick, start);
        preview.collect_gpu(request.readback_stamp(), timed)?;
        if !sample.priming {
            let frame = submission.frame(&sample.frame)?;
            self.retained[1] = Some(frame.clone());
        }
        Ok(())
    }

    fn enqueue_retained(&mut self) -> Result<()> {
        for index in 0..2 {
            let Some(frame) = &self.retained[index] else {
                continue;
            };
            let stamp = frame.stamp();
            let preview = &mut self.displays[index + 1];
            if self.enqueued[index] == Some(stamp.sample_id)
                || preview.readback.available_slots() == 0
            {
                continue;
            }
            let queued = if index == 0 {
                preview.enqueue(
                    &[frame
                        .channel::<u32>(1)?
                        .readback_source(0..(CAMERA_WIDTH * CAMERA_HEIGHT) as usize)?],
                    stamp,
                )?
            } else {
                preview.enqueue(
                    &[
                        frame.channel::<MVec3>(0)?.readback_source(0..1)?,
                        frame.channel::<MVec3>(1)?.readback_source(0..1)?,
                    ],
                    stamp,
                )?
            };
            if queued {
                self.enqueued[index] = Some(stamp.sample_id);
            }
        }
        Ok(())
    }
}
