use super::*;
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[test]
fn newton_shadow_shader_contract() {
    let source = include_str!("../shaders/soil_newton_shadow.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(source.matches("@binding(").count(), 11);
    assert!(source.contains("if summary[2]==0u"));
    assert!(source.contains("word(buffer,i,false)!=word(buffer,i,true)"));
}

#[test]
#[ignore = "requires GPU; exact five-buffer comparison and sticky first difference"]
fn newton_shadow_comparator_gpu_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let values: [Vec<u32>; 5] =
        [1usize, 65, 32, 257, 8].map(|n| (0..n).map(|i| 0x7fc00000u32 + i as u32).collect());
    let create = || {
        values.each_ref().map(|v| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Newton comparison fixture"),
                contents: bytemuck::cast_slice(v),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        })
    };
    let control = create();
    let candidate = create();
    let comparison = Comparison::new(&engine, control.each_ref(), candidate.each_ref());
    let run = |reset: bool| {
        let mut encoder = device.create_command_encoder(&Default::default());
        if reset {
            encoder.clear_buffer(&comparison.summary, 0, None);
        }
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        comparison.encode(&mut encoder);
        queue.submit([encoder.finish()]);
        let a = allocations.finish();
        let t = traffic.finish();
        let d = dispatch.finish();
        assert!(!a.invalid && !t.overflowed && !d.overflowed);
        assert_eq!(
            (
                a.observations,
                t.upload_bytes,
                t.readback_bytes,
                d.encoded_dispatches
            ),
            (0, 0, 0, 1)
        );
        let bytes = capture::read(&engine, &comparison.summary);
        bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
    };
    let first = run(false);
    assert_eq!(&first[..8], &[1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(first[8], 363);
    for buffer in 0..5 {
        for i in 0..5 {
            queue.write_buffer(&candidate[i], 0, bytemuck::cast_slice(&values[i]));
        }
        let last = values[buffer].len() - 1;
        queue.write_buffer(
            &candidate[buffer],
            last as u64 * 4,
            bytemuck::bytes_of(&(values[buffer][last] ^ 1)),
        );
        let state = run(false);
        assert_eq!(state[0], buffer as u32 + 2);
        assert_eq!(state[1], buffer as u32 + 1);
        assert_eq!(&state[2..7], &[2, 0, 0, 0x7fc00000, 0x7fc00001]);
        assert_eq!(state[7], buffer as u32 + 1);
        assert_eq!(state[8], 363 * state[0]);
    }
    for i in 0..5 {
        queue.write_buffer(&candidate[i], 0, bytemuck::cast_slice(&values[i]));
        assert_eq!(
            capture::read(&engine, &control[i]),
            bytemuck::cast_slice::<u32, u8>(&values[i])
        );
    }
    let sticky = run(false);
    assert_eq!(sticky[0], 7);
    assert_eq!(sticky[1], 5);
    assert_eq!(sticky[2], 2);
    let reset = run(true);
    assert_eq!(&reset[..8], &[1, 0, 0, 0, 0, 0, 0, 0]);
    for buffer in 0..5 {
        let last = values[buffer].len() - 1;
        let old = values[buffer][last];
        queue.write_buffer(
            &candidate[buffer],
            last as u64 * 4,
            bytemuck::bytes_of(&(old ^ 1)),
        );
        let state = run(true);
        assert_eq!(
            &state[..8],
            &[1, 1, 1, buffer as u32, last as u32, old, old ^ 1, 1]
        );
        queue.write_buffer(&candidate[buffer], 0, bytemuck::cast_slice(&values[buffer]));
    }
}
