use molla_compute::Engine;
use molla_sensors::{
    SensorCommandBatch,
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing},
};
use molla_storage::BufferUsage;
use std::{sync::Arc, time::Duration};

const FEATURES: wgpu::Features =
    wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

pub fn engine(timestamps: bool) -> Engine {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .unwrap();
    let features = if timestamps && adapter.features().contains(FEATURES) {
        FEATURES
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("molla::sensor latency benchmark"),
        required_features: features,
        required_limits: adapter.limits(),
        experimental_features: Default::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .unwrap();
    let engine = Engine::from_device(
        Arc::new(device),
        Arc::new(queue),
        Some(molla_compute::AdapterInfo::from_wgpu(adapter.get_info())),
    );
    let health = engine.device_health().unwrap().clone();
    engine
        .device()
        .unwrap()
        .set_device_lost_callback(move |reason, message| health.record(reason, message));
    engine
}

pub struct Timer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    words: DeviceBuffer<u32>,
    ring: ReadbackRing,
    period_ns: f64,
}

impl Timer {
    pub fn new(engine: &Engine) -> Option<Self> {
        let device = engine.device().unwrap();
        if !device.features().contains(FEATURES) {
            return None;
        }
        Some(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            words: DeviceBuffer::from_host(engine, vec![0u32; 4], BufferUsage::StorageReadWrite)
                .unwrap(),
            ring: ReadbackRing::new(
                engine,
                ReadbackConfig {
                    slots: 1,
                    bytes_per_slot: 16,
                    max_total_bytes: 16,
                },
            )
            .unwrap(),
            period_ns: engine.queue().unwrap().get_timestamp_period() as f64,
        })
    }

    pub fn begin(&self, batch: &mut SensorCommandBatch<'_>) {
        batch
            .encode_auxiliary(|encoder| {
                encoder.write_timestamp(&self.queries, 0);
                Ok(())
            })
            .unwrap();
    }

    pub fn end(&self, batch: &mut SensorCommandBatch<'_>) {
        batch
            .encode_auxiliary(|encoder| {
                encoder.write_timestamp(&self.queries, 1);
                encoder.resolve_query_set(&self.queries, 0..2, &self.resolve, 0);
                encoder.copy_buffer_to_buffer(&self.resolve, 0, self.words.device_buffer()?, 0, 16);
                Ok(())
            })
            .unwrap();
    }

    pub fn collect_us(&mut self) -> f64 {
        self.words.mark_device_dirty();
        let ticket = self
            .ring
            .try_enqueue(
                &[self.words.readback_source(0..4).unwrap()],
                Default::default(),
            )
            .unwrap()
            .unwrap();
        let frame = self.ring.wait(&ticket, Duration::from_secs(30)).unwrap();
        let words = frame.channel::<u32>(0).unwrap();
        let start = u64::from(words[0]) | (u64::from(words[1]) << 32);
        let end = u64::from(words[2]) | (u64::from(words[3]) << 32);
        end.checked_sub(start)
            .expect("non-monotonic GPU timestamps") as f64
            * self.period_ns
            / 1000.0
    }
}

pub fn wait(engine: &Engine) {
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
}
