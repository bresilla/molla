use super::*;

#[test]
#[ignore = "requires native GPU; paired sparse layout and poisoned history reset"]
fn shared_wall_paired_sparse_history_reset() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let f = articulated(65);
    let pairs: Vec<_> = f.free.iter().map(|q| [*q, 0.0]).collect();
    let free = storage(&engine, &pairs);
    let dense = create(&engine, &f, &free);
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    dense.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    let expected = check_result(&engine, &dense, &f, &combined(&pairs));
    let expected_output = capture::read(&engine, dense.output_buffer());
    let sparse = spread_contacts(&f, 4096, 63);
    let solver = create(&engine, &sparse, &free);
    for poisoned in [false, true] {
        if poisoned {
            let history = &solver.inner.capture_storage[3];
            engine.queue().unwrap().write_buffer(
                history,
                0,
                bytemuck::cast_slice(&vec![f32::NAN; history.size() as usize / 4]),
            );
        }
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
        let counts = traffic.finish();
        assert_eq!((counts.upload_bytes, counts.readback_bytes), (0, 0));
        let actual = check_result(&engine, &solver, &sparse, &combined(&pairs));
        assert_eq!(
            bytemuck::cast_slice::<_, u32>(&actual),
            bytemuck::cast_slice::<_, u32>(&expected)
        );
        let bytes = capture::read(&engine, solver.output_buffer());
        for c in 0..4096 {
            if c % 63 == 0 && c / 63 < 65 {
                assert_eq!(
                    &bytes[c * 80..(c + 1) * 80],
                    &expected_output[c / 63 * 80..(c / 63 + 1) * 80]
                );
            } else {
                assert!(bytes[c * 80..(c + 1) * 80].iter().all(|b| *b == 0));
            }
        }
    }
}

#[test]
#[ignore = "requires GPU; independent two-history coefficient oracle"]
fn full_state_two_coefficient_oracle() {
    let engine = Engine::new_gpu().unwrap();
    let source = format!(
        "{}\n{}\n{}",
        include_str!("../shaders/soil_wall_friction_block.wgsl"),
        include_str!("../shaders/soil_shared_wall_full_state_two.wgsl"),
        r#"
struct Params { width:u32 }
struct Status { sweeps:u32, control:vec4<u32>, full_older_q:array<f32,32> }
struct Contact { soil_mass:vec4<f32> }
struct History { active_index:u32, full_start_p:vec4<f32>, full_start_w:vec4<f32>, full_previous_p:vec4<f32>, full_previous_w:vec4<f32>, full_older_fp:vec4<f32>, full_older_fw:vec4<f32> }
var<private> shared_params:Params; var<private> shared_status:Status;
var<private> shared_mass:array<f32,1024>; var<private> shared_q:array<vec2<f32>,32>;
var<private> shared_previous_f:array<f32,32>; var<private> shared_contacts:array<Contact,2>;
var<private> shared_outputs:array<Output,2>; var<private> shared_history:array<History,2>;
@group(0) @binding(0) var<storage,read_write> result:array<vec4<f32>,5>;
@compute @workgroup_size(1)
fn probe() {
    rounding_slot=0u; shared_params.width=1u; shared_mass[0]=1.0;
    var start:array<vec2<f32>,32>;
    for (var test=0u;test<5u;test++) {
        shared_status.sweeps=3u; shared_status.control=vec4<u32>(0u,2u,0u,0u);
        let scale=select(1.0,1e-10,test==2u);
        for (var c=0u;c<2u;c++) {
            shared_history[c]=History(); shared_history[c].active_index=c;
            shared_contacts[c].soil_mass=vec4<f32>(0.0,0.0,0.0,1.0);
            shared_outputs[c]=Output();
            var current=select(-0.25,-0.0625,c==1u); var previous=select(-0.5,-0.25,c==1u); var older=-1.0;
            if test==1u { current=select(-0.5,0.5,c==1u); previous=2.0*current; older=4.0*current; }
            if test==3u { current=0.0; previous=0.0; older=0.0; }
            shared_outputs[c].machine_impulse_normal.x=scale*current;
            shared_history[c].full_previous_p.x=scale*previous;
            shared_history[c].full_older_fp.x=scale*older;
        }
        if test==4u { shared_status.sweeps=1u; }
        let first=coop_coefficient(start,test!=4u); let second=bitcast<f32>(shared_status.control.w);
        result[test]=vec4<f32>(first,second,abs(first)+abs(second),0.0);
    }
}
"#
    );
    let device = engine.device().unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("full state coefficient oracle"),
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
    let output = storage(&engine, &[[0f32; 4]; 5]);
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
    let rows: Vec<[f32; 4]> = capture::read(&engine, &output)
        .chunks_exact(16)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    for (i, expected) in [
        [5.0 / 3.0, -1.0 / 3.0],
        [1.0, 0.0],
        [5.0 / 3.0, -1.0 / 3.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    .iter()
    .enumerate()
    {
        for j in 0..2 {
            assert!(
                (rows[i][j] as f64 - expected[j]).abs() < 1e-5,
                "case {i}: {:?}",
                rows[i]
            );
        }
        assert!(rows[i][2] <= 2.000001);
    }
    println!("full-state coefficients={rows:?}");
}
