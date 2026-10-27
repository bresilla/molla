use super::*;
use crate::soil_contact_projection_gpu::{SoilPairedRatePublicationInputs, SoilRatePublicationGpu};

#[path = "soil_shared_wall_failure_tests.rs"]
mod failure;

#[path = "soil_shared_wall_full_state_tests.rs"]
mod full_state_tests;

#[path = "soil_shared_wall_newton_reference.rs"]
mod newton_reference;

fn storage<T: bytemuck::Pod>(engine: &Engine, data: &[T]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    engine
        .device()
        .unwrap()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paired rate fixture"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

fn create(
    engine: &Engine,
    f: &Fixture,
    free: &wgpu::Buffer,
) -> ExperimentalSoilPairedWallFrictionGpu {
    let mass = buffer(engine, &f.mass, 4);
    let rows = buffer(engine, &f.rows, 16 * f.free.len() as u64);
    let contacts = buffer(engine, &f.contacts, 64);
    let capacity = SoilSharedWallCapacity {
        dofs: f.free.len() as u32,
        contacts: f.contacts.len() as u32,
        max_sweeps: 64,
        local_iterations: 64,
        max_owned_bytes: u64::MAX,
    };
    let required = ExperimentalSoilPairedWallFrictionGpu::required_bytes(capacity).unwrap();
    let inputs = || SoilSharedWallInputs {
        mass: &mass,
        free_rates: free,
        point_rows: &rows,
        contacts: &contacts,
    };
    let scalar_free = storage(engine, &f.free);
    let counts = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ExperimentalSoilPairedWallFrictionGpu::new(
            engine,
            SoilSharedWallInputs {
                free_rates: &scalar_free,
                ..inputs()
            },
            capacity
        )
        .is_err()
    );
    assert!(
        ExperimentalSoilPairedWallFrictionGpu::new(
            engine,
            inputs(),
            SoilSharedWallCapacity {
                max_owned_bytes: required - 1,
                ..capacity
            }
        )
        .is_err()
    );
    assert_eq!(counts.finish().observations, 0);
    let solver = ExperimentalSoilPairedWallFrictionGpu::new(
        engine,
        inputs(),
        SoilSharedWallCapacity {
            max_owned_bytes: required,
            ..capacity
        },
    )
    .unwrap();
    assert_eq!(solver.owned_buffer_bytes(), required);
    solver
}

fn read_pairs(engine: &Engine, buffer: &wgpu::Buffer, n: usize) -> Vec<[f32; 2]> {
    capture::read(engine, buffer)
        .chunks_exact(8)
        .take(n)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}
fn combined(pairs: &[[f32; 2]]) -> Vec<f64> {
    pairs.iter().map(|q| q[0] as f64 + q[1] as f64).collect()
}
fn check_result(
    engine: &Engine,
    solver: &ExperimentalSoilPairedWallFrictionGpu,
    f: &Fixture,
    free: &[f64],
) -> Vec<[f32; 2]> {
    let bytes = capture::read(engine, solver.status_buffer());
    let status = bytemuck::pod_read_unaligned::<[u32; 8]>(&bytes[..32]);
    assert_eq!(status[0], 0, "paired status={status:?}");
    let pairs = read_pairs(engine, solver.rates_buffer(), free.len());
    let bytes = capture::read(engine, solver.output_buffer());
    let outputs: Vec<SoilWallFrictionOutput> = bytes
        .chunks_exact(80)
        .take(f.contacts.len())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    verify_precise_free_tolerance(f, free, &combined(&pairs), &outputs, 1e-6);
    let rates = combined(&pairs);
    for (c, output) in outputs.iter().enumerate() {
        let contact = &f.contacts[c];
        if contact.soil_mass[3] == 0.0 {
            continue;
        }
        let normal = vec(contact.normal_gap);
        let machine: f64 = rates
            .iter()
            .enumerate()
            .map(|(i, rate)| normal.dot(vec(f.rows[c * rates.len() + i])) * rate)
            .sum();
        let soil = normal.dot(vec(output.soil_velocity_energy));
        let gap = contact.normal_gap[3] as f64;
        let scale = 1f64.max(soil.abs()).max(machine.abs()).max(gap);
        assert!(
            soil - machine + gap >= -1e-6 * scale,
            "contact {c} retained normal residual {} scale {scale}",
            soil - machine + gap
        );
    }
    pairs
}

#[test]
fn shared_wall_paired_publication_shader_cpu() {
    let source = include_str!("../shaders/soil_paired_rate_publication.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let c = SoilSharedWallCapacity {
        dofs: 32,
        contacts: 1,
        max_sweeps: 64,
        local_iterations: 64,
        max_owned_bytes: u64::MAX,
    };
    assert_eq!(
        ExperimentalSoilPairedWallFrictionGpu::required_bytes(c).unwrap(),
        ExperimentalSoilSharedWallFrictionGpu::required_bytes(c).unwrap()
            + 128
            + full_state_history_bytes(c.contacts)
            + full_state_status_bytes()
    );
}

#[test]
#[ignore = "requires GPU; retained-tail normal acceptance"]
fn shared_wall_paired_normal_certificate_rejects_rounded_false_success() {
    let engine = Engine::new_gpu().unwrap();
    let source = format!(
        "{}\n{}\n{}\n{}",
        crate::soil_rate_encoding::arithmetic(1),
        include_str!("../shaders/soil_normal_rate_contract.wgsl"),
        include_str!("../shaders/soil_shared_wall_normal_contract.wgsl"),
        r#"
struct Params { width:u32 }
struct Contact { normal_gap:vec4<f32> }
var<private> shared_params:Params;
var<private> shared_contacts:array<Contact,1>;
var<private> shared_rows:array<vec4<f32>,1>;
var<private> shared_q:array<vec2<f32>,1>;
@group(0) @binding(0) var<storage,read_write> result:array<vec4<f32>,3>;
@compute @workgroup_size(1)
fn probe() {
    shared_params.width=1u;
    shared_contacts[0].normal_gap=vec4<f32>(1.0,0.0,0.0,0.0);
    shared_rows[0]=vec4<f32>(1.0,0.0,0.0,0.0);
    let soil=vec3<f32>(1.466255,0.0,0.0);
    for (var i=0u;i<3u;i++) {
        shared_q[0]=vec2<f32>(1.4662564,select(0.0,4e-8,i==1u));
        if i==2u { shared_q[0].y=-4e-8; }
        let machine=normal_rate_add(vec2<f32>(0.0),shared_contacts[0].normal_gap.xyz,shared_rows[0].xyz,shared_q[0]);
        let value=normal_rate_residual(soil,shared_contacts[0].normal_gap.xyz,0.0,machine);
        let high=rate_round(machine.x+machine.y);
        let old=soil.x-high;
        let old_accept=old>=-1e-6*max(1.0,max(abs(soil.x),abs(high)));
        result[i]=vec4<f32>(value,select(0.0,1.0,old_accept),select(0.0,1.0,shared_normal_covered(0u,soil)));
    }
}
"#
    );
    let device = engine.device().unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paired normal certificate regression"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = storage(&engine, &[[0f32; 4]; 3]);
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let bytes = capture::read(&engine, &output);
    let rows: Vec<[f32; 4]> = bytes
        .chunks_exact(16)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    assert_eq!(rows.iter().map(|r| r[2]).collect::<Vec<_>>(), [1., 1., 1.]);
    assert_eq!(rows.iter().map(|r| r[3]).collect::<Vec<_>>(), [1., 0., 1.]);
    let reference = 1.466255f32 as f64 - (1.4662564f32 as f64 + 4e-8f32 as f64);
    assert!((rows[1][0] as f64 - reference).abs() < 1e-12);
    println!(
        "rounded control passes; retained-tail residual {reference:e} rejects; zero/separating tails pass"
    );
}

#[test]
#[ignore = "requires native GPU; paired contact, articulated, wide and repeatability gates"]
fn shared_wall_paired_physical_cases() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let mut empty = pair(false, 0.0);
    empty.contacts.clear();
    empty.rows.clear();
    for (case, f) in [
        pair(false, 0.0),
        pair(true, 0.3),
        articulated(7),
        articulated(65),
        wide_fixture(),
        empty,
    ]
    .into_iter()
    .enumerate()
    {
        println!(
            "paired physical case={case} dofs={} contacts={}",
            f.free.len(),
            f.contacts.len()
        );
        let pairs: Vec<_> = f.free.iter().map(|q| [*q, 0.0]).collect();
        let free = storage(&engine, &pairs);
        let solver = create(&engine, &f, &free);
        let mut first = None;
        for _ in 0..2 {
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            encoder.clear_buffer(solver.status_buffer(), 0, None);
            solver.encode(&mut encoder);
            engine.queue().unwrap().submit([encoder.finish()]);
            assert_eq!(allocations.finish().observations, 0);
            let traffic = traffic.finish();
            assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
            let result = check_result(&engine, &solver, &f, &combined(&pairs));
            let raw = (
                result,
                capture::read(&engine, solver.output_buffer()),
                capture::read(&engine, solver.status_buffer()),
            );
            if let Some(first) = &first {
                assert_eq!(first, &raw);
            } else {
                first = Some(raw);
            }
        }
    }
}

#[test]
#[ignore = "requires native GPU; paired solve, persistent publication and feedback"]
fn shared_wall_paired_persistent_feedback() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let f = Fixture {
        mass: vec![
            1_000_000.0,
            0.0,
            0.0,
            0.0,
            1_000_000.0,
            0.0,
            0.0,
            0.0,
            1_000_000.0,
        ],
        free: vec![1.0, 0.0, 0.0],
        rows: vec![
            pack(DVec3::X, 0.0),
            pack(DVec3::Y, 0.0),
            pack(DVec3::Z, 0.0),
        ],
        contacts: vec![SoilSharedWallContact {
            soil_mass: [0.0, 0.0, 0.0, 1.0],
            normal_gap: pack(DVec3::X, 0.0),
            friction: [0.0; 4],
            boundary: [0; 4],
        }],
    };
    let initial = [1.0, 2.0_f32.powi(-27)];
    let free = storage(&engine, &[initial, [0.0; 2], [0.0; 2]]);
    let delta = storage(&engine, &[[0.0_f32; 2]; 3]);
    let machine = storage(&engine, &[0u32]);
    let solver = create(&engine, &f, &free);
    let increments = storage(&engine, &[[1e-8_f32, 1e-16], [0.0; 2], [0.0; 2]]);
    let advance = SoilRatePublicationGpu::from_paired_increments(
        &engine,
        &increments,
        solver.status_buffer(),
        3,
        SoilPairedRatePublicationInputs {
            rates: &free,
            velocity_delta: &delta,
            machine_status: &machine,
            max_owned_bytes: 16,
        },
    )
    .unwrap();
    let publication = SoilRatePublicationGpu::from_paired_buffers(
        &engine,
        solver.rates_buffer(),
        solver.status_buffer(),
        3,
        SoilPairedRatePublicationInputs {
            rates: &free,
            velocity_delta: &delta,
            machine_status: &machine,
            max_owned_bytes: 16,
        },
    )
    .unwrap();
    for _ in 0..8 {
        let start = combined(&read_pairs(&engine, &free, 3));
        let mut before = start.clone();
        before[0] += 1e-8_f32 as f64 + 1e-16_f32 as f64;
        let counts = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.clear_buffer(solver.status_buffer(), 0, None);
        advance.encode(&mut encoder);
        solver.encode(&mut encoder);
        publication.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(counts.finish().observations, 0);
        let traffic = traffic.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
        let projected = check_result(&engine, &solver, &f, &before);
        assert_eq!(read_pairs(&engine, &free, 3), projected);
        assert_ne!(projected[0][1], 0.0);
        let accumulated = combined(&read_pairs(&engine, &delta, 1))[0];
        assert!((accumulated - (combined(&projected)[0] - start[0])).abs() < 1e-12);
        assert_eq!(capture::read(&engine, &machine), [0; 4]);
    }
}

#[test]
#[ignore = "requires native GPU; paired publication atomicity, failure and restore"]
fn shared_wall_paired_publication_transaction() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let initial = vec![[1.0_f32, -2.0_f32.powi(-28)]; 32];
    let initial_delta = vec![[2e-9_f32, 2e-17]; 32];
    let valid: Vec<_> = (0..32)
        .map(|i| [1.0_f32, (i + 1) as f32 * 2.0_f32.powi(-30)])
        .collect();
    let projected = storage(&engine, &valid);
    let soil_status = storage(&engine, &[0u32]);
    let rates = storage(&engine, &initial);
    let delta = storage(&engine, &initial_delta);
    let machine = storage(&engine, &[0u32]);
    let short = storage(&engine, &[0.0_f32; 32]);
    let inputs = |rates, bytes| SoilPairedRatePublicationInputs {
        rates,
        velocity_delta: &delta,
        machine_status: &machine,
        max_owned_bytes: bytes,
    };
    let counts = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    for input in [
        inputs(&rates, 15),
        inputs(&projected, 16),
        inputs(&short, 16),
    ] {
        assert!(
            SoilRatePublicationGpu::from_paired_buffers(
                &engine,
                &projected,
                &soil_status,
                32,
                input
            )
            .is_err()
        );
    }
    assert_eq!(counts.finish().observations, 0);
    let publication = SoilRatePublicationGpu::from_paired_buffers(
        &engine,
        &projected,
        &soil_status,
        32,
        inputs(&rates, 16),
    )
    .unwrap();
    let snapshot = storage(&engine, &[0u32; 128]);
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&rates, 0, &snapshot, 0, 256);
    encoder.copy_buffer_to_buffer(&delta, 0, &snapshot, 256, 256);
    engine.queue().unwrap().submit([encoder.finish()]);
    let publish = || {
        let counts = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        publication.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(counts.finish().observations, 0);
        let traffic = traffic.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    };
    for (soil, bad) in [
        (0u32, [1.0, f32::NAN]),
        (0, [f32::INFINITY, 0.0]),
        (16, valid[31]),
    ] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&projected, 31 * 8, bytemuck::bytes_of(&bad));
        engine
            .queue()
            .unwrap()
            .write_buffer(&soil_status, 0, bytemuck::bytes_of(&soil));
        engine
            .queue()
            .unwrap()
            .write_buffer(&machine, 0, bytemuck::bytes_of(&0u32));
        publish();
        assert_eq!(read_pairs(&engine, &rates, 32), initial);
        assert_eq!(read_pairs(&engine, &delta, 32), initial_delta);
        assert_eq!(capture::read(&engine, &machine), 32u32.to_ne_bytes());
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&projected, 0, bytemuck::cast_slice(&valid));
    engine
        .queue()
        .unwrap()
        .write_buffer(&soil_status, 0, bytemuck::bytes_of(&0u32));
    publish();
    assert_eq!(read_pairs(&engine, &rates, 32), initial);
    engine
        .queue()
        .unwrap()
        .write_buffer(&machine, 0, bytemuck::bytes_of(&0u32));
    publish();
    let result = combined(&read_pairs(&engine, &rates, 32));
    let transport = combined(&read_pairs(&engine, &delta, 32));
    let expected = combined(&valid);
    let before = combined(&initial);
    let old_delta = combined(&initial_delta);
    for i in 0..32 {
        assert_eq!(result[i], expected[i]);
        assert!((transport[i] - (old_delta[i] + expected[i] - before[i])).abs() < 1e-14);
    }
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&snapshot, 0, &rates, 0, 256);
    encoder.copy_buffer_to_buffer(&snapshot, 256, &delta, 0, 256);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(read_pairs(&engine, &rates, 32), initial);
    assert_eq!(read_pairs(&engine, &delta, 32), initial_delta);
}

#[test]
#[ignore = "requires native GPU and captured input; physical paired-rate replay only"]
fn shared_wall_paired_capture_probe() {
    let root = std::path::PathBuf::from(std::env::var_os("SOIL_WALL_INPUT_REPLAY").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_WALL_PROBE_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let f = captured_fixture(&root);
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let pairs: Vec<_> = f.free.iter().map(|q| [*q, 0.0]).collect();
    let free = storage(&engine, &pairs);
    let solver = create(&engine, &f, &free);
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    let pairs = check_result(&engine, &solver, &f, &combined(&pairs));
    let bytes = capture::read(&engine, solver.status_buffer());
    let status = bytemuck::pod_read_unaligned::<[u32; 8]>(&bytes[..32]);
    std::fs::write(output.join("rates.bin"), bytemuck::cast_slice(&pairs)).unwrap();
    std::fs::write(
        output.join("outputs.bin"),
        capture::read(&engine, solver.output_buffer()),
    )
    .unwrap();
    std::fs::write(output.join("probe.json"),serde_json::to_vec_pretty(&serde_json::json!({"schema":2,"input":root,
        "rate_layout":"interleaved-f32-high-low","status":status,"owned_buffer_bytes":solver.owned_buffer_bytes(),
        "independent_physical_tolerance":1e-6,"acceptance":false,"scope":"paired component replay, not integrated runtime or timing acceptance"})).unwrap()).unwrap();
    eprintln!("paired captured status={status:?}");
}
