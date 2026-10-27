use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
};

#[test]
fn newton_delta_cache_contact_constants_source_contract() {
    let original = Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
        NewtonVariant::Fischer,
    )));
    let source = Solver::cached_contact_source(original.clone());
    let merit = source
        .split_once("fn release_merit(")
        .unwrap()
        .1
        .split_once("fn release_consider(")
        .unwrap()
        .0;
    assert!(!merit.contains("trace="));
    assert!(!merit.contains("fischer_length(node.normal_gap.xyz)"));
    for offset in ["18u", "19u", "20u+axis"] {
        assert!(merit.contains(&format!("condensed[cached+{offset}]")));
    }
    assert_eq!(
        source
            .matches("conditioned_contact_constants(c);var valid=direct_contact(c);")
            .count(),
        1
    );
    let direct = |s: &str| {
        s.split_once("fn direct_contact(")
            .unwrap()
            .1
            .split_once("fn direct_core(")
            .unwrap()
            .0
            .to_owned()
    };
    assert_eq!(direct(&source), direct(&original));
    for lanes in [64, 256] {
        let text = if lanes == 64 {
            source.clone()
        } else {
            format!("{}{lanes}\n{source}", crate::soil_newton_width::MARKER)
        };
        crate::soil_newton_protected::compile(&text, "newton").unwrap();
    }
}

#[test]
fn newton_delta_cache_source_contract() {
    let source = Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
        NewtonVariant::Fischer,
    )));
    assert_eq!(
        source
            .matches(
                "fn conditioned_prepare(chosen:u32)->bool {\n    conditioned_delta_valid=false;"
            )
            .count(),
        1
    );
    assert_eq!(source.matches("fn conditioned_state(c:u32,chosen:u32,next:Six)->State {\n    conditioned_delta_refresh(next);").count(),1);
    assert_eq!(
        source
            .matches("let dq=conditioned_delta_values[i];")
            .count(),
        1
    );
    assert_eq!(source.matches("conditioned_delta(next,i)").count(), 1);
    assert!(source.contains("bitcast<vec4<u32>>(next.k)"));
    for lanes in [64, 256] {
        let source = if lanes == 64 {
            source.clone()
        } else {
            format!("{}{lanes}\n{source}", crate::soil_newton_width::MARKER)
        };
        crate::soil_newton_protected::compile(&source, "newton").unwrap();
    }
}

#[test]
#[ignore = "requires GPU; private correction-cache keys and invalidation"]
fn newton_delta_cache_gpu_keys_and_reset() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source=format!("struct Six {{k:vec4<f32>,w:vec4<f32>}}\nstruct Params {{width:u32}}\nconst params=Params(3u);\nvar<private> calls:u32=0u;var<private> generation:f32=1.0;\n@group(0) @binding(0) var<storage,read_write> result_words:array<u32>;\nfn conditioned_delta(next:Six,i:u32)->vec2<f32> {{calls++;return vec2<f32>(generation*next.k.x+f32(i),0.0);}}\n{}\n@compute @workgroup_size(1) fn probe() {{
        var next=Six(vec4<f32>(1.0,0.0,0.0,0.0),vec4<f32>(0.0));
        conditioned_delta_refresh(next);result_words[0]=calls;
        conditioned_delta_refresh(next);result_words[1]=calls;
        next.w.x=7.0;conditioned_delta_refresh(next);result_words[2]=calls;
        next.k.x=2.0;conditioned_delta_refresh(next);result_words[3]=calls;result_words[4]=bitcast<u32>(conditioned_delta_values[0].x);
        generation=2.0;conditioned_delta_refresh(next);result_words[5]=calls;result_words[6]=bitcast<u32>(conditioned_delta_values[0].x);
        conditioned_delta_valid=false;conditioned_delta_refresh(next);result_words[7]=calls;result_words[8]=bitcast<u32>(conditioned_delta_values[0].x);
        next.k.x=0.0;conditioned_delta_refresh(next);result_words[9]=calls;
        next.k.x=bitcast<f32>(0x80000000u);conditioned_delta_refresh(next);result_words[10]=calls;
    }}",include_str!("../shaders/soil_newton_delta_cache.wgsl"));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Newton delta cache bookkeeping"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Newton delta cache bookkeeping"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("delta cache counts"),
        size: 44,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    for _ in 0..2 {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let bytes = crate::soil_shared_wall_friction_gpu::capture::read(&engine, &output);
        let actual: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            actual,
            [
                3,
                3,
                3,
                6,
                2f32.to_bits(),
                6,
                2f32.to_bits(),
                9,
                4f32.to_bits(),
                12,
                15
            ]
        );
    }
}
