use super::*;

pub(super) const OFFSET: u64 = std::mem::size_of::<Uniforms>() as u64;

pub(super) fn source(mut source: String) -> String {
    let field = "    grid_window_cells: vec4<i32>,\n}";
    assert_eq!(source.matches(field).count(), 1);
    source = source.replace(field,"    grid_window_cells: vec4<i32>,\n    reference_min: vec4<u32>,\n    reference_flags: vec4<u32>,\n}");
    let a = source.find("fn transfer_reference()->vec3<f32> {").unwrap();
    let b = a + source[a..].find("\n}").unwrap() + 2;
    source.replace_range(
        a..b,
        r#"fn transfer_reference()->vec3<f32> {
    let minimum = bitcast<vec3<f32>>(u.reference_min.xyz);
    let flags = u.reference_flags.xyz;
    return select(vec3<f32>(0.0), minimum, flags == vec3<u32>(1u))
        - select(vec3<f32>(0.0), minimum, flags == vec3<u32>(2u));
}"#,
    );
    source
}

pub(super) struct Reference {
    bindings: wgpu::BindGroup,
    reset: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
}

impl Reference {
    pub(super) fn new(sim: &mut MpmSimulation) -> Self {
        assert_eq!(
            sim.particle_count, 0,
            "install before particle initialization"
        );
        assert!(sim.bounded_reference.is_none());
        assert_eq!(OFFSET % 16, 0);
        let device = &sim.device;
        let buffer = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("test bounded-reference uniform tail"),
                size: OFFSET + 32,
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        sim.bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &sim.layout_uniforms,
            entries: &[buffer_entry(0, &buffer)],
        });
        sim.uniform_buf = buffer;
        let mut entries = storage_entries(1, true);
        entries.extend(storage_entries(2, false).into_iter().skip(1));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                buffer_entry(0, &sim.p_vel),
                buffer_entry(1, &sim.uniform_buf),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let source = format!(
            "const TAIL:u32={}u;\n{}",
            OFFSET / 4,
            include_str!("../shaders/mpm_bounded_reference.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test bounded velocity reduction"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            bindings,
            reset: pipeline("reference_reset"),
            reduce: pipeline("reference_reduce"),
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.set_pipeline(&self.reset);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
        if count > 0 {
            pass.set_pipeline(&self.reduce);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
    }
}

impl MpmSimulation {
    pub(super) fn encode_bounded_reference(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(reference) = &self.bounded_reference {
            reference.encode(encoder, self.particle_count);
        }
    }
}

#[test]
fn bounded_reference_shader_contract() {
    assert_eq!(OFFSET % 16, 0);
    let source = format!(
        "const TAIL:u32={}u;\n{}",
        OFFSET / 4,
        include_str!("../shaders/mpm_bounded_reference.wgsl")
    );
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(module.entry_points.len(), 2);
}

#[test]
#[ignore = "requires a GPU adapter; selection, reset and steady resource contract"]
fn bounded_reference_selection_and_reset() {
    use super::replay_tests::read_words;
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 8;
    cfg.max_particles = 257;
    let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
    sim.install_bounded_relative_transfer();
    for count in [0, 1, 63, 64, 65, 257, 1, 0] {
        for mode in 0..6 {
            let mut velocities: Vec<[f32; 3]> = (0..count)
                .map(|i| {
                    let positive = [
                        (i % 7 + 1) as f32,
                        (i + 1) as f32 * 0.125,
                        if i == 0 { 1e8 } else { 0.0001 },
                    ];
                    match mode {
                        0 => positive,
                        1 => positive.map(|v| -v),
                        2 => [
                            if i % 2 == 0 { -1. } else { 1. },
                            if i == 0 { -0. } else { 2. },
                            if i % 3 == 0 { 0. } else { 3. },
                        ],
                        4 => [f32::MAX, -f32::MAX, f32::MIN_POSITIVE],
                        5 => [
                            f32::from_bits(i as u32 + 1),
                            -f32::from_bits(i as u32 + 1),
                            0.,
                        ],
                        _ => positive,
                    }
                })
                .collect();
            if mode == 3 && count > 0 {
                velocities[0][0] = f32::NAN;
                velocities[count - 1][1] = f32::INFINITY;
                velocities[count / 2][2] = f32::NEG_INFINITY;
            }
            sim.set_particles_with_velocity(queue, &vec![[0.; 3]; count], &velocities)
                .unwrap();
            let uniforms = sim.step_uniforms(0.001);
            queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
            let expected: [f32; 3] = std::array::from_fn(|axis| {
                if count == 0 {
                    return 0.;
                }
                if velocities
                    .iter()
                    .all(|v| v[axis].is_finite() && v[axis] > 0.)
                {
                    velocities.iter().map(|v| v[axis]).fold(f32::MAX, f32::min)
                } else if velocities
                    .iter()
                    .all(|v| v[axis].is_finite() && v[axis] < 0.)
                {
                    velocities.iter().map(|v| v[axis]).fold(-f32::MAX, f32::max)
                } else {
                    0.
                }
            });
            for repeat in 0..2 {
                queue.write_buffer(
                    &sim.uniform_buf,
                    OFFSET,
                    bytemuck::cast_slice(&[0xdeadbeefu32; 8]),
                );
                let allocation = DeviceScope::start().unwrap();
                let traffic = TransferScope::start().unwrap();
                let dispatch = DispatchScope::start().unwrap();
                let mut encoder = device.create_command_encoder(&Default::default());
                sim.encode_bounded_reference(&mut encoder);
                queue.submit([encoder.finish()]);
                let allocation = allocation.finish();
                let traffic = traffic.finish();
                let dispatch = dispatch.finish();
                assert!(!allocation.invalid && !traffic.overflowed && !dispatch.overflowed);
                assert_eq!(allocation.observations, 0);
                assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
                assert_eq!(dispatch.encoded_dispatches, 1 + u64::from(count > 0));
                let words = read_words(device, queue, &sim.uniform_buf);
                let prefix: &[u32] = bytemuck::cast_slice(bytemuck::bytes_of(&uniforms));
                assert_eq!(&words[..OFFSET as usize / 4], prefix);
                let tail = &words[OFFSET as usize / 4..];
                assert_eq!(tail.len(), 8);
                assert_eq!((tail[3], tail[7]), (0x7f7fffff, 0));
                let actual: [f32; 3] = std::array::from_fn(|axis| match tail[4 + axis] {
                    1 => f32::from_bits(tail[axis]),
                    2 => -f32::from_bits(tail[axis]),
                    0 | 3 => 0.,
                    _ => panic!("invalid sign flags"),
                });
                assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
                println!(
                    "bounded-selection count={count} mode={mode} repeat={repeat} reference={actual:?} verified"
                );
            }
        }
    }
}
