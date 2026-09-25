use super::{MVec3, SensorCommandBatch, SensorPreview, Sim, preview_pool_config};
use molla::core::{Error, Result};
use molla::sensors::{
    LidarBatch, LidarScan, SensorScene,
    readback::{ReadbackFrame, ReadbackStamp},
};

#[derive(Clone, Copy)]
enum Projection {
    Hits,
    Origins,
    Rays,
}

impl Projection {
    fn bytes(self) -> u64 {
        match self {
            Self::Hits => 24,
            Self::Origins => 36,
            Self::Rays => 52,
        }
    }
}

pub(super) struct LidarPreview {
    pub sensor: LidarBatch,
    scene: SensorScene,
    pub display: SensorPreview,
    projection: Projection,
    capacity: u32,
    once: bool,
}

impl LidarPreview {
    pub fn new(sim: &Sim, rays: u32, origins: bool, once: bool) -> Result<Self> {
        Self::with_projection(
            sim,
            rays,
            if origins {
                Projection::Origins
            } else {
                Projection::Hits
            },
            once,
        )
    }

    pub fn ray_fan(sim: &Sim, rays: u32) -> Result<Self> {
        Self::with_projection(sim, rays, Projection::Rays, false)
    }

    fn buffers(
        sim: &Sim,
        rays: u32,
        projection: Projection,
        once: bool,
    ) -> Result<(LidarBatch, SensorPreview, u32)> {
        let capacity = rays
            .checked_add(63)
            .map(|count| count / 64 * 64)
            .filter(|&count| count != 0)
            .ok_or_else(|| Error::Storage("LiDAR preview capacity overflow/empty".into()))?;
        let sensor = LidarBatch::new(&sim.engine, 1, capacity)?;
        let pool =
            sensor.output_pool(&sim.engine, preview_pool_config(u64::from(capacity) * 84)?)?;
        let mut display =
            SensorPreview::new(&sim.engine, pool, u64::from(capacity) * projection.bytes())?;
        if once {
            display.cadence_hz = None;
        }
        Ok((sensor, display, capacity))
    }

    fn with_projection(sim: &Sim, rays: u32, projection: Projection, once: bool) -> Result<Self> {
        let scene = SensorScene::from_model(&sim.engine, &sim.model, Default::default())?;
        let (sensor, display, capacity) = Self::buffers(sim, rays, projection, once)?;
        Ok(Self {
            sensor,
            scene,
            display,
            projection,
            capacity,
            once,
        })
    }

    pub fn grow(&mut self, sim: &Sim, rays: u32) -> Result<bool> {
        if rays <= self.capacity {
            return Ok(false);
        }
        let (sensor, display, capacity) = Self::buffers(sim, rays, self.projection, self.once)?;
        self.sensor = sensor;
        self.display = display;
        self.capacity = capacity;
        Ok(true)
    }

    pub fn update(&mut self, sim: &Sim, scan: LidarScan) -> Result<Option<ReadbackFrame>> {
        let latest = self.display.poll_latest(sim.sensor_clock)?;
        if self.once && self.display.submitted != 0
            || !self.once && sim.sensor_clock.tick == 0
            || !self.display.begin_sample(sim.sensor_clock.tick)?
        {
            return Ok(latest);
        }
        self.submit(sim, scan, sim.sensor_clock.stamp(scan.sensor_id))?;
        Ok(latest)
    }

    pub fn request(&mut self, sim: &Sim, scan: LidarScan, id: u64) -> Result<bool> {
        if !self.display.begin_request(id)? {
            return Ok(false);
        }
        let mut stamp = sim.sensor_clock.stamp(scan.sensor_id);
        stamp.sample_id = id;
        self.submit(sim, scan, stamp)
    }

    fn submit(&mut self, sim: &Sim, scan: LidarScan, stamp: ReadbackStamp) -> Result<bool> {
        let start = std::time::Instant::now();
        let mut batch = SensorCommandBatch::new(&sim.engine)?;
        let timed = self.display.begin_gpu(&mut batch, &sim.engine)?;
        let Some(sample) = self.sensor.encode_scene(
            &mut batch,
            &sim.engine,
            &self.scene,
            &sim.state_a,
            &[scan],
            &mut self.display.pool,
            stamp,
        )?
        else {
            self.display.deferred += 1;
            return Ok(false);
        };
        self.display.end_gpu(&mut batch, &sim.engine, timed)?;
        let submission = batch.submit()?;
        self.display.submitted(sim.sensor_clock.tick, start);
        self.display.collect_gpu(stamp, timed)?;
        let frame = submission.frame(&sample.frame)?;
        let n = sample.ray_count;
        let first = match self.projection {
            Projection::Origins | Projection::Rays => {
                frame.channel::<MVec3>(0)?.readback_source(0..n)?
            }
            Projection::Hits => frame.channel::<f64>(2)?.readback_source(0..n)?,
        };
        if matches!(self.projection, Projection::Rays) {
            self.display.enqueue(
                &[
                    first,
                    frame.channel::<i32>(3)?.readback_source(0..n)?,
                    frame.channel::<MVec3>(4)?.readback_source(0..n)?,
                    frame.channel::<MVec3>(1)?.readback_source(0..n)?,
                ],
                frame.stamp(),
            )
        } else {
            self.display.enqueue(
                &[
                    first,
                    frame.channel::<i32>(3)?.readback_source(0..n)?,
                    frame.channel::<MVec3>(4)?.readback_source(0..n)?,
                ],
                frame.stamp(),
            )
        }
    }
}
