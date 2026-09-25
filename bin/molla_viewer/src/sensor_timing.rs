use super::{Engine, SensorCommandBatch};
use molla::core::{Error, Result};
use molla::sensors::readback::{
    DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp, ReadbackTicket,
};
use std::{collections::VecDeque, sync::Arc};

const FEATURES: wgpu::Features =
    wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

pub(super) fn viewer_engine() -> Result<Engine> {
    if std::env::var_os("MOLLA_SENSOR_TIMING").as_deref() == Some(std::ffi::OsStr::new("1")) {
        timestamp_engine()
    } else {
        Engine::new_gpu()
    }
}

pub(super) fn timestamp_engine() -> Result<Engine> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| Error::Gpu(format!("timestamp viewer adapter: {e}")))?;
    let features = if adapter.features().contains(FEATURES) {
        FEATURES
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("molla::viewer timestamp device"),
        required_features: features,
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| Error::Gpu(format!("timestamp viewer device: {e}")))?;
    let engine = Engine::from_device(
        Arc::new(device),
        Arc::new(queue),
        Some(molla::compute::AdapterInfo::from_wgpu(adapter.get_info())),
    );
    let health = engine.device_health().expect("adopted GPU").clone();
    engine
        .device()
        .expect("adopted GPU")
        .set_device_lost_callback(move |reason, message| health.record(reason, message));
    Ok(engine)
}

pub(super) struct SensorGpuTimer {
    device: Arc<wgpu::Device>,
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    words: DeviceBuffer<u32>,
    ring: ReadbackRing,
    pending: VecDeque<ReadbackTicket>,
    period_ns: f64,
    pub latest: Option<(ReadbackStamp, f64)>,
    pub deferred: u64,
}

impl SensorGpuTimer {
    pub fn new(engine: &Engine) -> Result<Option<Self>> {
        engine.check_device()?;
        let device = engine.device().expect("checked GPU").clone();
        if !device.features().contains(FEATURES) {
            return Ok(None);
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("molla::viewer sensor timing"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::viewer sensor timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let words = DeviceBuffer::from_host(
            engine,
            vec![0u32; 4],
            super::mp::BufferUsage::StorageReadWrite,
        )?;
        let ring = ReadbackRing::new(
            engine,
            ReadbackConfig {
                slots: 3,
                bytes_per_slot: 16,
                max_total_bytes: 48,
            },
        )?;
        Ok(Some(Self {
            device,
            queries,
            resolve,
            words,
            ring,
            pending: VecDeque::with_capacity(3),
            period_ns: f64::from(engine.queue().expect("checked GPU").get_timestamp_period()),
            latest: None,
            deferred: 0,
        }))
    }

    pub fn available(&mut self) -> bool {
        if self.ring.available_slots() == 0 {
            self.deferred += 1;
            false
        } else {
            true
        }
    }

    pub fn begin(&self, batch: &mut SensorCommandBatch, engine: &Engine) -> Result<()> {
        self.check_engine(engine)?;
        batch.encode_auxiliary(|encoder| {
            encoder.write_timestamp(&self.queries, 0);
            Ok(())
        })
    }

    pub fn end(&self, batch: &mut SensorCommandBatch, engine: &Engine) -> Result<()> {
        self.check_engine(engine)?;
        batch.encode_auxiliary(|encoder| {
            encoder.write_timestamp(&self.queries, 1);
            encoder.resolve_query_set(&self.queries, 0..2, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, self.words.device_buffer()?, 0, 16);
            Ok(())
        })
    }

    pub fn enqueue(&mut self, stamp: ReadbackStamp) -> Result<bool> {
        self.words.mark_device_dirty();
        if let Some(ticket) = self
            .ring
            .try_enqueue(&[self.words.readback_source(0..4)?], stamp)?
        {
            self.pending.push_back(ticket);
            Ok(true)
        } else {
            self.deferred += 1;
            Ok(false)
        }
    }

    pub fn poll(&mut self, revision: u64) -> Result<()> {
        if self
            .latest
            .is_some_and(|(stamp, _)| stamp.scene_revision != revision)
        {
            self.latest = None;
        }
        while let Some(ticket) = self.pending.front() {
            let Some(frame) = self.ring.poll(ticket)? else {
                break;
            };
            self.pending.pop_front();
            if frame.stamp.scene_revision == revision {
                let words = frame.channel::<u32>(0)?;
                let start = u64::from(words[0]) | (u64::from(words[1]) << 32);
                let end = u64::from(words[2]) | (u64::from(words[3]) << 32);
                if let Some(ticks) = end.checked_sub(start) {
                    self.latest = Some((frame.stamp, ticks as f64 * self.period_ns / 1000.0));
                } else {
                    return Err(Error::Gpu(
                        "sensor timestamps are reversed or wrapped".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_engine(&self, engine: &Engine) -> Result<()> {
        engine.check_device()?;
        if !Arc::ptr_eq(&self.device, engine.device().expect("checked GPU")) {
            return Err(Error::Gpu("sensor timer: foreign engine".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    #[ignore = "requires GPU with timestamp queries"]
    fn sensor_gpu_timestamps_bound_readback_and_report_device_duration() {
        let engine = timestamp_engine().unwrap();
        let mut timer = SensorGpuTimer::new(&engine)
            .unwrap()
            .expect("timestamp-capable GPU");
        let device = engine.device().unwrap();
        let work = DeviceBuffer::from_host(
            &engine,
            vec![0u32; 65536],
            super::super::mp::BufferUsage::StorageReadWrite,
        )
        .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("timestamp fixture compute"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> words: array<u32>;
                 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                     var x = id.x + 7u;
                     for (var i = 0u; i < 128u; i++) { x = x * 1664525u + 1013904223u; }
                     words[id.x] = x;
                 }".into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("timestamp fixture"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("timestamp fixture"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: work.device_buffer().unwrap().as_entire_binding(),
            }],
        });
        for sample_id in 1..=3 {
            assert!(timer.available());
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            timer.begin(&mut batch, &engine).unwrap();
            batch
                .encode_auxiliary(|encoder| {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("timestamp fixture work"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bindings, &[]);
                    pass.dispatch_workgroups(1024, 1, 1);
                    Ok(())
                })
                .unwrap();
            timer.end(&mut batch, &engine).unwrap();
            batch.submit().unwrap();
            assert!(
                timer
                    .enqueue(ReadbackStamp {
                        sensor_id: 1,
                        sample_id,
                        scene_revision: 9,
                        simulation_time: Some(sample_id as f64),
                    })
                    .unwrap()
            );
        }
        assert!(!timer.available());
        assert_eq!(timer.deferred, 1);
        engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        timer.poll(9).unwrap();
        let (stamp, duration_us) = timer.latest.unwrap();
        assert_eq!(stamp.sample_id, 3);
        assert!(
            duration_us.is_finite() && duration_us > 0.0,
            "{duration_us}"
        );
        assert!(timer.available());
        assert_eq!(timer.ring.stats().copied_bytes, 48);
        assert_eq!(timer.ring.stats().blocking_waits, 0);
        assert_eq!(timer.ring.stats().staging_allocations, 3);
        let mut result = ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: 8,
                max_total_bytes: 8,
            },
        )
        .unwrap();
        let ticket = result
            .try_enqueue(&[work.readback_source(0..1).unwrap()], stamp)
            .unwrap()
            .unwrap();
        let expected = (0..128).fold(7u32, |x, _| {
            x.wrapping_mul(1664525).wrapping_add(1013904223)
        });
        assert_eq!(
            result
                .wait(&ticket, Duration::from_secs(30))
                .unwrap()
                .channel::<u32>(0)
                .unwrap(),
            [expected]
        );
        let default = Engine::new_gpu().unwrap();
        assert!(SensorGpuTimer::new(&default).unwrap().is_none());
        let mut foreign = SensorCommandBatch::new(&default).unwrap();
        assert!(timer.begin(&mut foreign, &default).is_err());
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        timer.begin(&mut batch, &engine).unwrap();
        timer.end(&mut batch, &engine).unwrap();
        batch.submit().unwrap();
        timer
            .enqueue(ReadbackStamp {
                scene_revision: 8,
                ..stamp
            })
            .unwrap();
        engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        timer.poll(9).unwrap();
        assert_eq!(timer.latest.unwrap().0.scene_revision, 9);
        timer.poll(10).unwrap();
        assert!(timer.latest.is_none());
        let viewer = viewer_engine().unwrap();
        let opted_in =
            std::env::var_os("MOLLA_SENSOR_TIMING").as_deref() == Some(std::ffi::OsStr::new("1"));
        assert_eq!(
            viewer.device().unwrap().features().contains(FEATURES),
            opted_in
        );
    }
}
