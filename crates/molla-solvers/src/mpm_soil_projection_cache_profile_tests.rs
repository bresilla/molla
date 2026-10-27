use super::*;
use crate::mpm::replay_tests::read_words;

fn setup(engine: &Engine, initial: &MpmSoilCheckpoint, arm: usize) -> MpmSimulation {
    let queue = engine.queue().unwrap();
    let mut sim = MpmSimulation::new_sparse(engine, initial.config(), 512).unwrap();
    sim.restore_soil_with(queue, initial).unwrap();
    sim.prepare_parallel_particle_tile_ordering(queue, 50_000)
        .unwrap();
    sim.set_particle_tile_ordering(true).unwrap();
    sim.prepare_deterministic_p2g_with_budget(3_200_000)
        .unwrap();
    sim.set_deterministic_p2g(true).unwrap();
    sim.prepare_soil_motion_guard_with_budget(6_400_016)
        .unwrap();
    sim.set_soil_motion_bound(Some([0.01; 3]));
    sim.prepare_soil_quiescence(800_016).unwrap();
    if arm > 0 {
        sim.prepare_soil_projection_cache(5_600_000).unwrap();
    }
    if arm == 2 {
        let original = source();
        let marker = "if soil_projection_hit(pid, f, jp) {";
        assert_eq!(original.matches(marker).count(), 1);
        let instrumented = original.replace(
            marker,
            &format!("{marker}\n soil_projection_cache[pid].valid = 2u;"),
        );
        let layout = sim
            .motion_guard
            .as_ref()
            .unwrap()
            .quiescence
            .as_ref()
            .unwrap()
            .projection_cache
            .as_ref()
            .unwrap()
            .pipeline
            .get_bind_group_layout(0);
        let pipeline = ProjectionCache::pipeline(&sim, &layout, instrumented);
        sim.motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .projection_cache
            .as_mut()
            .unwrap()
            .pipeline = pipeline;
    }
    sim.prepare_step_profiling(1).unwrap();
    sim
}

struct HitProbe {
    counts: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

impl HitProbe {
    fn validate(&self, engine: &Engine, sim: &MpmSimulation) {
        assert_eq!(self.read(engine), 0);
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let records = &sim
            .motion_guard
            .as_ref()
            .unwrap()
            .quiescence
            .as_ref()
            .unwrap()
            .projection_cache
            .as_ref()
            .unwrap()
            .records;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("diagnostic hit positive control"),
            source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> records: array<u32>;\n@compute @workgroup_size(1) fn main() { records[49999u * 28u + 27u] = 2u; }".into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[buffer_entry(0, records)],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        assert_eq!(self.read(engine), 1);
        assert_eq!(self.read(engine), 0);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(records, 0, None);
        queue.submit([encoder.finish()]);
        assert_eq!(self.read(engine), 0);
        println!("projection hit probe positive control: 0,1,0 at particle 49999");
    }

    fn new(sim: &MpmSimulation) -> Self {
        let device = &sim.device;
        let counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diagnostic projection hits"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("diagnostic projection hit count"),
            source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> records: array<u32>;\n@group(0) @binding(1) var<storage, read_write> hits: atomic<u32>;\n@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) { if gid.x < 50000u && records[gid.x * 28u + 27u] == 2u { atomicAdd(&hits, 1u); records[gid.x * 28u + 27u] = 1u; } }".into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let records = &sim
            .motion_guard
            .as_ref()
            .unwrap()
            .quiescence
            .as_ref()
            .unwrap()
            .projection_cache
            .as_ref()
            .unwrap()
            .records;
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[buffer_entry(0, records), buffer_entry(1, &counts)],
        });
        Self {
            counts,
            bindings,
            pipeline,
        }
    }

    fn read(&self, engine: &Engine) -> u32 {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.counts, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups(50_000_u32.div_ceil(256), 1, 1);
        }
        queue.submit([encoder.finish()]);
        read_words(device, queue, &self.counts)[0]
    }
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires timestamp GPU; frozen 50k detached-soil cache profile, not machine acceptance"]
fn projection_cache_50k_profile() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_CACHE_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_CACHE_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let initial = MpmSoilCheckpoint::from_bytes(&std::fs::read(input).unwrap(), 50_000).unwrap();
    assert_eq!(initial.particle_count(), 50_000);
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )
    .unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sims = [0, 1, 2].map(|arm| setup(&engine, &initial, arm));
    let hits = HitProbe::new(&sims[2]);
    hits.validate(&engine, &sims[2]);
    let mut all = Vec::new();
    let mut reference: Vec<String> = Vec::new();
    for round in 0..3 {
        for arm in if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
            let sim = &mut sims[arm];
            sim.restore_soil_with(queue, &initial).unwrap();
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            for step in 0..12 {
                let resources = DeviceScope::start().unwrap();
                let transfers = TransferScope::start().unwrap();
                let begin = std::time::Instant::now();
                sim.step_n_profiled_with(device, queue, 0.0001, 1).unwrap();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let wall_ms = begin.elapsed().as_secs_f64() * 1000.0;
                let allocations = resources.finish().observations;
                let transfers = transfers.finish();
                assert_eq!(allocations, 0);
                assert_eq!(transfers.readback_bytes, 0);
                let profile = sim.read_step_profile().unwrap().unwrap();
                assert_eq!(profile.requested_substeps, 1);
                assert_eq!(profile.particles, 50_000);
                let checkpoint = sim.checkpoint_soil_with(queue).unwrap();
                let bytes = checkpoint.to_bytes();
                let digest = blake3::hash(&bytes).to_hex().to_string();
                if round == 0 && arm == 0 {
                    reference.push(digest.clone());
                }
                assert_eq!(
                    digest, reference[step],
                    "round={round} arm={arm} step={step}"
                );
                let hit_count = if arm == 2 {
                    Some(hits.read(&engine))
                } else {
                    None
                };
                let stages: Vec<_> = profile
                    .stages
                    .iter()
                    .map(|s| serde_json::json!({"name":s.stage.name(),"ms":s.elapsed_ms}))
                    .collect();
                let record = serde_json::json!({"round":round,"arm":arm,"step":step,"warmup":step<2,
                    "wall_ms":wall_ms,"stages":stages,"hits":hit_count,"particles":50_000,
                    "dt":0.0001_f32,"substeps":1,"checkpoint_blake3":digest,
                    "hot_allocations":allocations,"hot_upload_bytes":transfers.upload_bytes,
                    "hot_readback_bytes":transfers.readback_bytes,"profile_readback_bytes":profile.readback_bytes});
                std::fs::write(
                    output.join(format!("round-{round}-arm-{arm}-step-{step}.json")),
                    serde_json::to_vec_pretty(&record).unwrap(),
                )
                .unwrap();
                all.push(record);
                if step == 11 {
                    std::fs::write(output.join(format!("round-{round}-arm-{arm}.soil")), bytes)
                        .unwrap();
                }
            }
            println!(
                "projection cache profile round={round} arm={arm} steps=12 particles=50000 exact_trajectory=true"
            );
        }
    }
    std::fs::write(
        output.join("records.json"),
        serde_json::to_vec_pretty(&all).unwrap(),
    )
    .unwrap();
}
