use crate::soil_shared_wall_friction_gpu::capture;

const SOURCE: &str = r#"
struct Pair {values:array<u32,2>}
@group(0) @binding(0) var<storage,read_write> output:array<u32>;
@compute @workgroup_size(1)
fn probe() {
    for(var i=0u;i<4u;i++) {
        var local:Pair;
        if i%2u==0u {local.values[0]=7u+i;}
        output[2u*i]=local.values[0];output[2u*i+1u]=local.values[1];
    }
}
"#;

#[test]
#[ignore = "requires GPU; loop-local implicit/explicit initialization diagnostic"]
fn newton_loop_local_initialization_gpu_diagnostic() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let expected = [7u32, 0, 0, 0, 9, 0, 0, 0];
    for explicit in [false, true] {
        let source = if explicit {
            SOURCE.replace("var local:Pair;", "var local=Pair();")
        } else {
            SOURCE.to_owned()
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("loop-local initialization diagnostic"),
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
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
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let bytes = capture::read(&engine, &output);
        let actual: [u32; 8] = bytemuck::pod_read_unaligned(&bytes);
        eprintln!(
            "loop-local explicit={explicit} actual={actual:?} expected={expected:?} conforms={}",
            actual == expected
        );
        if explicit {
            assert_eq!(actual, expected);
        }
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_INITIALIZATION_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("explicit-{explicit}.wgsl")), source).unwrap();
            std::fs::write(root.join(format!("explicit-{explicit}.bin")), bytes).unwrap();
        }
    }
}
