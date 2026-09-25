use super::{
    MVec3, SensorCommandBatch, SensorPreview, Sim, TiledCamera, TiledCameraConfig, mp,
    preview_pool_config,
};
use molla::core::{Error, Result};
use molla::sensors::{
    ImageOp, ImageProcessor, ImageProcessorConfig, PendingImageJob, SensorFrame, SensorFramePool,
    SensorScene, readback::ReadbackFrame,
};
use std::collections::VecDeque;

#[derive(Clone, Copy, PartialEq)]
pub(super) struct CameraRequest {
    pub tick: u64,
    pub config: TiledCameraConfig,
    pub eye: MVec3,
    pub target: MVec3,
    pub exposure: f32,
    pub tone_map: bool,
    pub gamma: bool,
}

impl CameraRequest {
    fn needs_processing(self) -> bool {
        (self.exposure - 1.0).abs() > 1e-3 || self.tone_map || self.gamma
    }
}

struct DeviceImage {
    camera: SensorFrame,
    color: SensorFrame,
    color_channel: usize,
    pose: mp::Transform,
}

pub(super) struct CameraPreview {
    pub camera: TiledCamera,
    scene: SensorScene,
    processing: ImageProcessor,
    processed_pool: SensorFramePool,
    pub display: SensorPreview,
    desired: Option<CameraRequest>,
    request_id: u64,
    last_device: Option<DeviceImage>,
    last_enqueued: Option<u64>,
    pending_poses: VecDeque<(u64, mp::Transform)>,
    pixels: usize,
}

impl CameraPreview {
    pub fn new(sim: &Sim, config: TiledCameraConfig) -> Result<Self> {
        let pixels = config
            .width
            .checked_mul(config.height)
            .ok_or_else(|| Error::Storage("camera preview pixel count overflow".into()))?;
        let camera = TiledCamera::new(&sim.engine, config, 1)?;
        let scene = SensorScene::from_model(&sim.engine, &sim.model, Default::default())?;
        let pool = camera.output_pool(&sim.engine, preview_pool_config(u64::from(pixels) * 32)?)?;
        let processing = ImageProcessor::new(
            &sim.engine,
            ImageProcessorConfig {
                max_pixels: pixels,
                max_images: 1,
                max_operations_per_image: 3,
                max_buffer_bytes: u64::from(pixels) * 8 + 4096,
            },
        )?;
        let processed_pool = processing.output_pool(
            &sim.engine,
            preview_pool_config(u64::from(pixels) * 4 + 48)?,
        )?;
        let display = SensorPreview::new(&sim.engine, pool, u64::from(pixels) * 32)?;
        Ok(Self {
            camera,
            scene,
            processing,
            processed_pool,
            display,
            desired: None,
            request_id: 0,
            last_device: None,
            last_enqueued: None,
            pending_poses: VecDeque::with_capacity(3),
            pixels: pixels as usize,
        })
    }

    pub fn update(
        &mut self,
        sim: &Sim,
        request: CameraRequest,
        visible: bool,
    ) -> Result<Option<(ReadbackFrame, mp::Transform)>> {
        let latest = if let Some(frame) = self.display.poll_latest(sim.sensor_clock)? {
            let pose = self
                .pending_poses
                .iter()
                .find(|(id, _)| *id == frame.stamp.sample_id)
                .map(|(_, pose)| *pose)
                .ok_or_else(|| Error::Storage("camera preview pose missing".into()))?;
            self.pending_poses
                .retain(|(id, _)| *id > frame.stamp.sample_id);
            Some((frame, pose))
        } else {
            None
        };
        if self.desired != Some(request) {
            self.request_id = self
                .request_id
                .checked_add(1)
                .ok_or_else(|| Error::Storage("camera request identity exhausted".into()))?;
            self.desired = Some(request);
        }
        if self
            .last_device
            .as_ref()
            .is_none_or(|image| image.camera.stamp().sample_id != self.request_id)
        {
            if request.needs_processing() && self.processed_pool.available_slots()? == 0 {
                self.display.deferred += 1;
            } else if self.display.begin_request(self.request_id)? {
                self.submit(sim, request, visible)?;
            }
        }
        if visible {
            self.enqueue_latest()?;
        }
        Ok(latest)
    }

    fn submit(&mut self, sim: &Sim, request: CameraRequest, visible: bool) -> Result<()> {
        let start = std::time::Instant::now();
        if !request.exposure.is_finite() {
            return Err(Error::Storage(
                "camera preview exposure must be finite".into(),
            ));
        }
        self.camera.set_config(request.config)?;
        self.camera
            .set_camera_look_at(0, request.eye, request.target, MVec3::Y)?;
        let pose = self.camera.poses_host()?[0];
        let mut operations = [ImageOp::Exposure(1.0); 3];
        let mut count = 0;
        if (request.exposure - 1.0).abs() > 1e-3 {
            operations[count] = ImageOp::Exposure(request.exposure);
            count += 1;
        }
        if request.tone_map {
            operations[count] = ImageOp::Reinhard;
            count += 1;
        }
        if request.gamma {
            operations[count] = ImageOp::LinearToSrgb;
            count += 1;
        }
        let mut stamp = sim.sensor_clock.stamp(12);
        stamp.sample_id = self.request_id;
        let mut batch = SensorCommandBatch::new(&sim.engine)?;
        let timed = visible && self.display.begin_gpu(&mut batch, &sim.engine)?;
        let Some(camera) = self.camera.encode_scene(
            &mut batch,
            &sim.engine,
            &self.scene,
            &sim.state_a,
            &mut self.display.pool,
            stamp,
        )?
        else {
            return Err(Error::Storage(
                "camera preview pool unavailable after preflight".into(),
            ));
        };
        let color = if count != 0 {
            Some(
                self.processing
                    .encode_pending(
                        &mut batch,
                        &[PendingImageJob {
                            frame: &camera.frame,
                            channel: 1,
                            pixels: 0..self.pixels,
                            operations: &operations[..count],
                            stamp,
                        }],
                        &mut self.processed_pool,
                        stamp,
                    )?
                    .ok_or_else(|| {
                        Error::Storage("camera processing pool unavailable after preflight".into())
                    })?,
            )
        } else {
            None
        };
        self.display.end_gpu(&mut batch, &sim.engine, timed)?;
        let submission = batch.submit()?;
        let raw = submission.frame(&camera.frame)?;
        let (color, color_channel) = match &color {
            Some(color) => (submission.frame(&color.frame)?.clone(), 0),
            None => (raw.clone(), 1),
        };
        self.last_device = Some(DeviceImage {
            camera: raw.clone(),
            color,
            color_channel,
            pose,
        });
        self.display.submitted(sim.sensor_clock.tick, start);
        self.display.collect_gpu(stamp, timed)?;
        Ok(())
    }

    fn enqueue_latest(&mut self) -> Result<()> {
        let Some(image) = &self.last_device else {
            return Ok(());
        };
        let stamp = image.camera.stamp();
        if self.last_enqueued == Some(stamp.sample_id)
            || self.display.readback.available_slots() == 0
        {
            return Ok(());
        }
        let n = self.pixels;
        if self.display.enqueue(
            &[
                image.camera.channel::<f64>(0)?.readback_source(0..n)?,
                image
                    .color
                    .channel::<u32>(image.color_channel)?
                    .readback_source(0..n)?,
                image.camera.channel::<MVec3>(2)?.readback_source(0..n)?,
                image.camera.channel::<u32>(3)?.readback_source(0..n)?,
                image.camera.channel::<u32>(4)?.readback_source(0..n)?,
            ],
            stamp,
        )? {
            self.pending_poses.push_back((stamp.sample_id, image.pose));
            self.last_enqueued = Some(stamp.sample_id);
        }
        Ok(())
    }
}
