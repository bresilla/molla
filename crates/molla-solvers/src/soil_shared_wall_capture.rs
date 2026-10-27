use super::*;

pub(crate) const CAPTURE_STOP: u32 = 1 << 30;

pub(crate) struct SharedWallTrace {
    engine: std::sync::Arc<Engine>,
    root: std::path::PathBuf,
    next: std::sync::atomic::AtomicU32,
    copy: SharedWallCapture,
}

impl SharedWallTrace {
    pub fn new(
        engine: std::sync::Arc<Engine>,
        root: std::path::PathBuf,
        inputs: &[wgpu::Buffer; 4],
        status: &wgpu::Buffer,
        capacity: SoilSharedWallCapacity,
        paired_rates: bool,
    ) -> Self {
        std::fs::create_dir(&root).unwrap();
        let copy = SharedWallCapture::new(&engine, inputs, status, capacity, paired_rates);
        Self {
            engine,
            root,
            next: std::sync::atomic::AtomicU32::new(0),
            copy,
        }
    }

    fn flush(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        path: &std::path::Path,
        stage: &str,
    ) -> f64 {
        let device = self.engine.device().unwrap();
        let next = device.create_command_encoder(&Default::default());
        let pending = std::mem::replace(encoder, next);
        let start = std::time::Instant::now();
        eprintln!("shared-wall trace {}: {stage} begin", path.display());
        self.engine.queue().unwrap().submit([pending.finish()]);
        let result = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if let Err(error) = result {
            std::fs::write(path.join("timeout.json"), serde_json::to_vec_pretty(&serde_json::json!({
                "stage": stage, "error": error.to_string(), "elapsed_ms": elapsed, "acceptance": false
            })).unwrap()).unwrap();
            eprintln!(
                "shared-wall trace {}: {stage} failed: {error}",
                path.display()
            );
            std::process::exit(76);
        }
        eprintln!(
            "shared-wall trace {}: {stage} complete ({elapsed:.3} ms)",
            path.display()
        );
        elapsed
    }

    pub fn before(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &[wgpu::Buffer; 4],
        capacity: SoilSharedWallCapacity,
    ) -> std::path::PathBuf {
        let sequence = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = self.root.join(format!("solve-{sequence:03}"));
        std::fs::create_dir(&path).unwrap();
        let bindings: Vec<_> = inputs.iter().enumerate().map(|(binding, buffer)| {
            serde_json::json!({"binding": binding, "bytes": buffer.size(), "usage": buffer.usage().bits()})
        }).collect();
        std::fs::write(
            path.join("input-resources.json"),
            serde_json::to_vec_pretty(&bindings).unwrap(),
        )
        .unwrap();
        self.flush(encoder, &path, "upstream-and-geometry");
        self.copy.copy_current(encoder);
        self.flush(encoder, &path, "input-snapshot");
        for (name, buffer) in ["mass", "free", "rows", "contacts"]
            .iter()
            .zip(&self.copy.snapshot.inputs)
        {
            std::fs::write(path.join(format!("{name}.bin")), read(&self.engine, buffer)).unwrap();
        }
        std::fs::write(path.join("capture.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "schema": if self.copy.snapshot.paired_rates { 2 } else { 1 }, "dofs": capacity.dofs, "contacts": capacity.contacts,
            "rate_encoding": if self.copy.snapshot.paired_rates { "f32x2" } else { "f32" },
            "max_sweeps": capacity.max_sweeps, "local_iterations": capacity.local_iterations,
            "sequence_in_parity": sequence, "acceptance": false,
            "scope": "Actual input before shared solve under test-only synchronized stage tracing. No solver bypass or numerical change; not throughput or full-frame acceptance."
        })).unwrap()).unwrap();
        path
    }

    pub fn checkpoint(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        path: &std::path::Path,
        pass: usize,
        status: &wgpu::Buffer,
        split: bool,
    ) {
        let stage = match pass {
            0 => "factor".into(),
            1 => "prepare".into(),
            2 if split => "prepare-cache".into(),
            _ => format!("sweep-{:03}", pass - 1 - usize::from(split)),
        };
        let elapsed = self.flush(encoder, path, &stage);
        let bytes = read(&self.engine, status);
        std::fs::write(path.join(format!("{stage}-status.bin")), &bytes).unwrap();
        let words: Vec<_> = bytes[..32]
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        std::fs::write(
            path.join(format!("{stage}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({
                "stage": stage, "status": words, "completion_ms": elapsed, "acceptance": false
            }))
            .unwrap(),
        )
        .unwrap();
    }

    pub fn finish(&self, path: &std::path::Path, solver: &ExperimentalSoilSharedWallFrictionGpu) {
        for (name, buffer) in [
            ("status", &solver.status),
            ("rates", &solver.rates),
            ("outputs", &solver.outputs),
            ("conditional", &solver.conditional_inputs),
        ] {
            std::fs::write(path.join(format!("{name}.bin")), read(&self.engine, buffer)).unwrap();
        }
    }

    pub fn factor_state(&self, path: &std::path::Path, buffers: &[wgpu::Buffer; 4]) {
        if std::env::var_os("SOIL_WALL_TRACE_SKIP_FACTOR_DUMP").is_some() {
            return;
        }
        let mut sizes = serde_json::Map::new();
        for (name, buffer) in ["lower", "response", "params", "history"]
            .iter()
            .zip(buffers)
        {
            let data = read(&self.engine, buffer);
            sizes.insert(
                (*name).into(),
                serde_json::json!({"bytes": data.len(), "usage": buffer.usage().bits()}),
            );
            std::fs::write(path.join(format!("factor-{name}.bin")), data).unwrap();
        }
        std::fs::write(
            path.join("factor-resources.json"),
            serde_json::to_vec_pretty(&sizes).unwrap(),
        )
        .unwrap();
        if std::env::var_os("SOIL_WALL_TRACE_STOP_BEFORE_PREPARE").is_some() {
            std::fs::write(path.join("intentional-stop.json"), serde_json::to_vec_pretty(&serde_json::json!({
                "stage": "after-factor-before-prepare", "exit": 77, "acceptance": false,
                "scope": "Intentional diagnostic process stop after retained state capture; no preparation dispatch or completed frame"
            })).unwrap()).unwrap();
            eprintln!(
                "shared-wall trace {}: intentional stop before preparation",
                path.display()
            );
            std::process::exit(77);
        }
    }

    pub fn stop_prepared(
        &self,
        path: &std::path::Path,
        solver: &ExperimentalSoilSharedWallFrictionGpu,
    ) {
        if std::env::var_os("SOIL_WALL_TRACE_STOP_AFTER_PREPARE").is_none() {
            return;
        }
        self.finish(path, solver);
        std::fs::write(path.join("intentional-stop.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "stage": "after-prepare-before-sweeps", "exit": 77, "acceptance": false,
            "factor_dump_skipped": std::env::var_os("SOIL_WALL_TRACE_SKIP_FACTOR_DUMP").is_some(),
            "scope": "Intentional diagnostic process stop after preparation; no shared sweep or completed frame"
        })).unwrap()).unwrap();
        eprintln!(
            "shared-wall trace {}: intentional stop after preparation",
            path.display()
        );
        std::process::exit(77);
    }
}

#[derive(Clone)]
pub(crate) struct SharedWallSnapshot {
    pub paired_rates: bool,
    pub inputs: [wgpu::Buffer; 4],
    pub control: wgpu::Buffer,
    pub capacity: SoilSharedWallCapacity,
}

pub(crate) struct SharedWallCapture {
    pub snapshot: SharedWallSnapshot,
    pipelines: [wgpu::ComputePipeline; 3],
    groups: [wgpu::BindGroup; 4],
}

impl SharedWallCapture {
    pub fn new(
        engine: &Engine,
        inputs: &[wgpu::Buffer; 4],
        status: &wgpu::Buffer,
        capacity: SoilSharedWallCapacity,
        paired_rates: bool,
    ) -> Self {
        let device = engine.device().unwrap();
        let buffer = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test-only shared-wall input snapshot"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let snapshot = SharedWallSnapshot {
            paired_rates,
            inputs: inputs.each_ref().map(|b| buffer(b.size())),
            control: buffer(16),
            capacity,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &(0..4)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding == 0,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test-only shared-wall capture and stop"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_shared_wall_capture.wgsl").into(),
            ),
        });
        let pipelines = ["copy_input", "stop", "copy_current"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[&inputs[i], &snapshot.inputs[i], status, &snapshot.control]
                    .iter()
                    .enumerate()
                    .map(|(binding, b)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        Self {
            snapshot,
            pipelines,
            groups,
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for i in 0..5 {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[usize::from(i == 4)]);
            pass.set_bind_group(0, &self.groups[i.min(3)], &[]);
            pass.dispatch_workgroups(
                if i == 4 {
                    1
                } else {
                    (self.snapshot.inputs[i].size() / 4).div_ceil(64) as u32
                },
                1,
                1,
            );
            molla_core::gpu_metrics::record_dispatch();
        }
    }

    pub fn copy_current(&self, encoder: &mut wgpu::CommandEncoder) {
        for i in 0..4 {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[2]);
            pass.set_bind_group(0, &self.groups[i], &[]);
            pass.dispatch_workgroups(
                (self.snapshot.inputs[i].size() / 4).div_ceil(64) as u32,
                1,
                1,
            );
            molla_core::gpu_metrics::record_dispatch();
        }
    }
}

pub(crate) fn read(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<u8> {
    read_many(engine, &[buffer]).pop().unwrap()
}

pub(crate) fn read_many(engine: &Engine, buffers: &[&wgpu::Buffer]) -> Vec<Vec<u8>> {
    if buffers.is_empty() {
        return Vec::new();
    }
    let device = engine.device().unwrap();
    let mut end = 0u64;
    let ranges: Vec<_> = buffers
        .iter()
        .map(|buffer| {
            assert!(buffer.size() > 0 && buffer.size() % wgpu::COPY_BUFFER_ALIGNMENT == 0);
            let start = end;
            end = end.checked_add(buffer.size()).unwrap();
            start..end
        })
        .collect();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-only shared-wall snapshot read"),
        size: end,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for (buffer, range) in buffers.iter().zip(&ranges) {
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, range.start, buffer.size());
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(15)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(15))
        .unwrap()
        .unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    let bytes = ranges
        .iter()
        .map(|range| mapped[range.start as usize..range.end as usize].to_vec())
        .collect();
    drop(mapped);
    staging.unmap();
    bytes
}

#[test]
#[ignore = "requires GPU; exact batched snapshot boundaries and duplicate buffers"]
fn shared_wall_capture_batched_read() {
    use wgpu::util::DeviceExt;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let inputs = [
        vec![0x80000000u32],
        vec![0x7fc00001, 1, 0xffffffff],
        (0..16).collect(),
    ];
    let buffers: Vec<_> = inputs
        .iter()
        .map(|words| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batched read oracle"),
                contents: bytemuck::cast_slice(words),
                usage: wgpu::BufferUsages::COPY_SRC,
            })
        })
        .collect();
    assert!(read_many(&engine, &[]).is_empty());
    let order = [2usize, 0, 1, 0, 2];
    let refs: Vec<_> = order.iter().map(|&i| &buffers[i]).collect();
    for _ in 0..2 {
        let actual = read_many(&engine, &refs);
        for (&i, bytes) in order.iter().zip(actual) {
            assert_eq!(bytes, bytemuck::cast_slice::<u32, u8>(&inputs[i]));
        }
    }
    assert_eq!(
        read(&engine, &buffers[1]),
        bytemuck::cast_slice::<u32, u8>(&inputs[1])
    );
}

impl SharedWallSnapshot {
    pub fn save(&self, engine: &Engine, path: &std::path::Path) -> bool {
        let control = read(engine, &self.control);
        assert_eq!(control.len(), 16);
        if u32::from_ne_bytes(control[..4].try_into().unwrap()) == 0 {
            return false;
        }
        std::fs::create_dir(path).unwrap();
        for (name, buffer) in ["mass", "free", "rows", "contacts"]
            .iter()
            .zip(&self.inputs)
        {
            std::fs::write(path.join(format!("{name}.bin")), read(engine, buffer)).unwrap();
        }
        std::fs::write(path.join("capture.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "schema": if self.paired_rates { 2 } else { 1 }, "dofs": self.capacity.dofs, "contacts": self.capacity.contacts,
            "rate_encoding": if self.paired_rates { "f32x2" } else { "f32" },
            "max_sweeps": self.capacity.max_sweeps, "local_iterations": self.capacity.local_iterations,
            "capture_stop_bit": CAPTURE_STOP, "acceptance": false,
            "scope": "First shared-wall input before factor, preparation or solve. Test-only deliberate frame abort, no physical frame committed."
        })).unwrap()).unwrap();
        true
    }
}
