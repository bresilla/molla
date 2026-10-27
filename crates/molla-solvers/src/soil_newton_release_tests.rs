use super::*;

#[test]
#[ignore = "requires GPU; production release selection, visited set and eight-release ceiling"]
fn resident_newton_local_release_guards() {
    let source = include_str!("../shaders/soil_newton_local_release.wgsl");
    let declarations = source.split_once("fn release_poly(").unwrap().0;
    let selector = &source[source.find("fn release_sliding_wall(").unwrap()..];
    let source = format!("struct Six {{k:vec4<f32>,w:vec4<f32>}}\n
        struct State {{high:Six,low:Six}}\n
        struct Outer {{releases:u32}}\n
        struct Status {{active_count:u32}}\n
        struct Block {{residual:array<f32,6>}}\n
        var<private> lane:u32;
        var<workgroup> outer:Outer;var<workgroup> status:Status;
        var<workgroup> active_indices:array<u32,17>;
        var<workgroup> states:array<State,17>;var<workgroup> blocks:array<Block,17>;
        var<workgroup> visits:array<atomic<u32>,17>;
        @group(0) @binding(0) var<storage,read> inputs:array<f32>;
        @group(0) @binding(1) var<storage,read_write> outputs:array<f32>;
        {declarations}\n{selector}\n
        fn release_local(c:u32,mask:u32)->ReleaseChoice {{
            atomicAdd(&visits[c],1u);
            return ReleaseChoice(Six(vec4<f32>(f32(c+1u),0.0,0.0,0.0),vec4<f32>(0.0)),f32(c),1u);
        }}
        @compute @workgroup_size(64) fn probe(@builtin(local_invocation_index) id:u32) {{
            lane=id;if lane==0u {{status.active_count=u32(inputs[0]);}}
            if lane<17u {{active_indices[lane]=lane;blocks[lane].residual[0]=f32(lane+1u);}}
            workgroupBarrier();
            for(var attempt=0u;attempt<9u;attempt++) {{
                let changed=release_sliding_wall();
                if lane==0u {{outputs[2u+attempt]=select(0.0,1.0,changed);outputs[1]+=select(0.0,1.0,changed);}}
                workgroupBarrier();
            }}
            if lane==0u {{outputs[0]=f32(outer.releases);}}
            if lane<8u {{outputs[11u+lane]=f32(release_visited[lane]);}}
            if lane<17u {{outputs[19u+lane]=states[lane].high.k.x;outputs[36u+lane]=f32(atomicLoad(&visits[lane]));}}
        }}");
    let engine = Engine::new_gpu().unwrap();
    for count in [0, 1, 17] {
        let result = probe(&engine, source.clone(), &[count as f32], 53, 1);
        let releases = count.min(8);
        assert_eq!(result[0] as usize, releases);
        assert_eq!(result[1] as usize, releases);
        for i in 0..9 {
            assert_eq!(result[2 + i], if i < releases { 1. } else { 0. });
        }
        let chosen: Vec<usize> = if count == 17 {
            (2..=9).rev().collect()
        } else if count == 1 {
            vec![0]
        } else {
            vec![]
        };
        for (i, &c) in chosen.iter().enumerate() {
            assert_eq!(result[11 + i] as usize, c);
            assert_eq!(result[36 + c], 8.);
        }
        for c in 0..17 {
            assert_eq!(
                result[19 + c],
                if chosen.contains(&c) {
                    (c + 1) as f32
                } else {
                    0.
                }
            );
        }
        if count == 17 {
            assert_eq!(result[36 + 16], 64.);
        }
    }
}

fn pure_source() -> String {
    let source = include_str!("../shaders/soil_newton_local_release.wgsl");
    let structures = source.split_once("struct ReleaseChoice").unwrap().0;
    let roots = &source
        [source.find("fn release_poly(").unwrap()..source.find("fn release_merit(").unwrap()];
    let candidates = &source[source.find("fn release_candidates(").unwrap()
        ..source.find("fn release_sliding_wall(").unwrap()];
    format!(
        "{structures}\nfn finite(x:f32)->bool {{return abs(x)<=3.402823e38;}}\n{roots}\n{candidates}"
    )
}

fn probe(engine: &Engine, source: String, input: &[f32], size: usize, count: u32) -> Vec<f32> {
    let device = engine.device().unwrap();
    let input = storage(engine, input);
    let output = storage(engine, &vec![0f32; size]);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("local release oracle"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("local release oracle"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[&input, &output]
            .into_iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count, 1, 1);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    values(engine, &output)
}

#[test]
#[ignore = "requires GPU; production quartic root isolation and sliding branch oracle"]
fn resident_newton_local_release_oracles() {
    let engine = Engine::new_gpu().unwrap();
    let s = 0.5f32.sqrt();
    let t = 0.125f32.sqrt();
    let cases = [
        ([0.0625, 0., -0.625, 0., 1.], vec![-s, -t, t, s]),
        ([0.25, -1., 1., 0., 0.], vec![0.5]),
        ([-1., 0., 1., 0., 0.], vec![-1., 1.]),
        ([-0.25, 1., 0., 0., 0.], vec![0.25]),
        ([-2., 1., 0., 0., 0.], vec![]),
        ([0.; 5], vec![]),
        ([1., 0., 0., 0., 0.], vec![]),
    ];
    let mut input = Vec::new();
    for scale in [1f32, 1e-20, 1e20] {
        for (p, _) in &cases {
            input.extend(p.map(|v| v * scale));
        }
    }
    let source = format!(
        "{}\n
        @group(0) @binding(0) var<storage,read> inputs:array<f32>;
        @group(0) @binding(1) var<storage,read_write> outputs:array<f32>;
        @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            var p:array<f32,5>;for(var i=0u;i<5u;i++) {{p[i]=inputs[id.x*5u+i];}}
            let roots=release_unit_roots(p);outputs[id.x*9u+8u]=f32(roots.count);
            for(var i=0u;i<roots.count;i++) {{outputs[id.x*9u+i]=roots.values[i];}}
        }}",
        pure_source()
    );
    let result = probe(
        &engine,
        source,
        &input,
        9 * 3 * cases.len(),
        (3 * cases.len()) as u32,
    );
    for (i, output) in result.chunks_exact(9).enumerate() {
        let expected = &cases[i % cases.len()].1;
        assert_eq!(
            output[8] as usize,
            expected.len(),
            "root case {i}: {output:?}"
        );
        for (a, b) in output.iter().zip(expected) {
            assert!((a - b).abs() < 3e-6, "root case {i}: {a} != {b}");
        }
    }
    let a = [2., 0.2, -0.1, 0.2, 0.7, 0.05, -0.1, 0.05, 1.3];
    let multiply = |x: DVec3| {
        DVec3::from_array(std::array::from_fn(|i| {
            (0..3).map(|j| a[3 * i + j] * x[j]).sum()
        }))
    };
    let mut input = Vec::<f32>::new();
    let mut expected = Vec::new();
    for n in [
        DVec3::Y,
        DVec3::X,
        DVec3::new(0.0238544, -0.9997154, 1.4e-7).normalize(),
    ] {
        let t1 = n
            .cross(if n.x.abs() < 0.8 { DVec3::X } else { DVec3::Y })
            .normalize();
        let t2 = n.cross(t1);
        for mu in [0., 0.1, 0.3, 1.1] {
            for angle in [
                0.,
                1.2,
                std::f64::consts::PI,
                std::f64::consts::PI + 1e-6,
                -1.4,
            ] {
                let tangent = t1 * angle.cos() + t2 * angle.sin();
                let k = 2. * (n - mu * tangent);
                let b = 0.7 * tangent - 0.2 * n - multiply(k);
                input.extend(pack(n, mu));
                for column in 0..3 {
                    input.extend([
                        a[column] as f32,
                        a[3 + column] as f32,
                        a[6 + column] as f32,
                        0.,
                    ]);
                }
                input.extend(pack(b, 0.2));
                expected.push(k);
            }
        }
    }
    let source = format!("{}\n
        @group(0) @binding(0) var<storage,read> inputs:array<vec4<f32>>;
        @group(0) @binding(1) var<storage,read_write> outputs:array<vec4<f32>>;
        @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            let i=id.x*5u;let n=inputs[i];let b=inputs[i+4u];
            let a=mat3x3<f32>(inputs[i+1u].xyz,inputs[i+2u].xyz,inputs[i+3u].xyz);
            let candidates=release_candidates(normalize(n.xyz),n.w,b.w,a,b.xyz);
            outputs[id.x*21u+20u]=vec4<f32>(f32(candidates.count),0.0,0.0,0.0);
            for(var j=0u;j<candidates.count;j++) {{outputs[id.x*21u+j]=vec4<f32>(candidates.values[j],0.0);}}
        }}", pure_source());
    let result = probe(
        &engine,
        source,
        &input,
        84 * expected.len(),
        expected.len() as u32,
    );
    for (i, (output, expected)) in result.chunks_exact(84).zip(expected).enumerate() {
        let count = output[80] as usize;
        assert!((1..=20).contains(&count));
        let error = output[..4 * count]
            .chunks_exact(4)
            .map(|v| (DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64) - expected).length())
            .fold(f64::INFINITY, f64::min);
        assert!(error < 3e-5, "sliding case {i}: error={error:e}");
    }
}
