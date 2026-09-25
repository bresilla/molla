use super::{
    MVec3, SensorCommandBatch, SensorPreview, Sim, TiledCamera, preview_pool_config,
    synth_annotation_config,
};
use molla::core::{Error, Result};
use molla::sensors::{
    AnnotationConfig, AnnotationProcessor, DeviceBoundingBox, PendingAnnotationJob,
    SensorFramePool, SensorScene, ShapeLabel, readback::ReadbackFrame,
};

pub(super) struct AnnotationPreview {
    pub camera: TiledCamera,
    scene: SensorScene,
    pub annotations: AnnotationProcessor,
    camera_pool: SensorFramePool,
    pub display: SensorPreview,
}

impl AnnotationPreview {
    pub fn new(sim: &Sim) -> Result<Self> {
        let config = synth_annotation_config();
        let pixels = u64::from(config.width) * u64::from(config.height);
        let mut camera = TiledCamera::new(&sim.engine, config, 1)?;
        camera.set_camera_look_at(
            0,
            MVec3::new(0.0, 2.6, 5.0),
            MVec3::new(0.3, 0.5, 0.0),
            MVec3::Y,
        )?;
        let scene = SensorScene::from_model(&sim.engine, &sim.model, Default::default())?;
        let camera_pool = camera.output_pool(&sim.engine, preview_pool_config(pixels * 32)?)?;
        let mut annotations = AnnotationProcessor::new(
            &sim.engine,
            AnnotationConfig {
                max_pixels: pixels as u32,
                max_images: 1,
                max_labels: 3,
                max_buffer_bytes: pixels * 8 + 264,
            },
        )?;
        annotations.set_labels(&[
            ShapeLabel {
                shape_id: 1,
                class_id: 1,
            },
            ShapeLabel {
                shape_id: 2,
                class_id: 2,
            },
            ShapeLabel {
                shape_id: 3,
                class_id: 3,
            },
        ])?;
        let pool = annotations.output_pool(&sim.engine, preview_pool_config(pixels * 8 + 112)?)?;
        let display = SensorPreview::new(&sim.engine, pool, 96)?;
        Ok(Self {
            camera,
            scene,
            annotations,
            camera_pool,
            display,
        })
    }

    pub fn update(&mut self, sim: &Sim) -> Result<Option<ReadbackFrame>> {
        let latest = self.display.poll_latest(sim.sensor_clock)?;
        if !self.display.begin_sample(sim.sensor_clock.tick)? {
            return Ok(latest);
        }
        if self.camera_pool.available_slots()? == 0 {
            self.display.deferred += 1;
            return Ok(latest);
        }
        let start = std::time::Instant::now();
        let stamp = sim.sensor_clock.stamp(108);
        let config = *self.camera.config();
        let mut batch = SensorCommandBatch::new(&sim.engine)?;
        let timed = self.display.begin_gpu(&mut batch, &sim.engine)?;
        let camera = self
            .camera
            .encode_scene(
                &mut batch,
                &sim.engine,
                &self.scene,
                &sim.state_a,
                &mut self.camera_pool,
                stamp,
            )?
            .ok_or_else(|| {
                Error::Storage("annotation camera pool unavailable after preflight".into())
            })?;
        let annotations = self
            .annotations
            .encode_pending(
                &mut batch,
                &[PendingAnnotationJob {
                    frame: &camera.frame,
                    channel: 4,
                    pixels: 0..(config.width * config.height) as usize,
                    width: config.width,
                    height: config.height,
                    stamp,
                }],
                &mut self.display.pool,
                stamp,
            )?
            .ok_or_else(|| Error::Storage("annotation pool unavailable after preflight".into()))?;
        self.display.end_gpu(&mut batch, &sim.engine, timed)?;
        let submission = batch.submit()?;
        self.display.submitted(sim.sensor_clock.tick, start);
        self.display.collect_gpu(stamp, timed)?;
        let frame = submission.frame(&annotations.frame)?;
        self.display.enqueue(
            &[frame
                .channel::<DeviceBoundingBox>(2)?
                .readback_source(0..3)?],
            frame.stamp(),
        )?;
        Ok(latest)
    }
}
