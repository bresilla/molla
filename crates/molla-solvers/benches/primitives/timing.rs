use molla_compute::Engine;

pub const FEATURES: wgpu::Features =
    wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

pub struct Timer {
    query: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl Timer {
    pub fn new(engine: &Engine) -> Self {
        let device = engine.device().unwrap();
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            query: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("primitive runtime timeline"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: buffer(
                "primitive timestamps",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: buffer(
                "primitive timestamp readback",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }

    pub fn begin(&self, engine: &Engine) {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.write_timestamp(&self.query, 0);
        engine.queue().unwrap().submit([encoder.finish()]);
    }

    pub fn end(&self, engine: &Engine) -> Result<f64, Box<dyn std::error::Error>> {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.write_timestamp(&self.query, 1);
        encoder.resolve_query_set(&self.query, 0..2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, 16);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(120)),
        })?;
        rx.recv()??;
        let view = self.readback.slice(..).get_mapped_range();
        let words: &[u64] = bytemuck::cast_slice(&view);
        let elapsed = words[1]
            .checked_sub(words[0])
            .ok_or("non-monotonic GPU timestamps")?;
        drop(view);
        self.readback.unmap();
        Ok(elapsed as f64 * f64::from(queue.get_timestamp_period()) * 1e-9)
    }
}
