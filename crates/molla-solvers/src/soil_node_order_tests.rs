use super::*;

#[test]
fn canonical_node_order_source_and_limits() {
    for capacity in [1, 3, 63, 64, 65, 127, 129, 512, 1024, 4096] {
        let padded = width(capacity, 32768).unwrap();
        assert!(padded >= capacity && padded.is_power_of_two());
        let module = naga::front::wgsl::parse_str(&source(capacity, padded)).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    }
    for capacity in [0, 8192, u32::MAX] {
        assert!(width(capacity, 32768).is_err());
    }
    assert!(width(1, 11).is_err());
    assert_eq!(width(1, 12).unwrap(), 1);
}

#[test]
#[ignore = "requires real GPU; canonical node multiset, bounds, fault and reuse"]
fn canonical_node_order_gpu() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let buffer = |size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node order test"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    for capacity in [1u32, 3, 63, 64, 65, 127, 129, 512, 1024, 4096] {
        let output = buffer(u64::from(capacity) * 2 * 32);
        let spans = buffer(16);
        let status = buffer(4);
        let owner = SoilNodeOrderGpu::new(device, &output, &spans, &status, capacity, 2).unwrap();
        let mut input = Vec::new();
        for body in 0..2u32 {
            for i in 0..capacity {
                let k = (i.wrapping_mul(1664525).wrapping_add(body * 1013904223)) % 97;
                let node = [
                    (k as f32 - 48.0).to_bits(),
                    if i % 2 == 0 {
                        0.0f32.to_bits()
                    } else {
                        (-0.0f32).to_bits()
                    },
                    ((k % 3) as f32).to_bits(),
                    0.5f32.to_bits(),
                    ((i % 5) as f32).to_bits(),
                    (-0.125f32).to_bits(),
                    0.0f32.to_bits(),
                    ((i % 64) as f32).to_bits(),
                ];
                input.push(node);
            }
        }
        queue.write_buffer(owner.input(), 0, bytemuck::cast_slice(&input));
        for counts in [
            [capacity, 0],
            [capacity - 1, capacity / 2],
            [1, capacity.min(2)],
        ] {
            queue.write_buffer(
                &spans,
                0,
                bytemuck::cast_slice(&[0, counts[0], capacity, counts[1]]),
            );
            let mut expected = vec![[0u32; 8]; 2 * capacity as usize];
            for body in 0..2 {
                let first = body * capacity as usize;
                let end = first + counts[body] as usize;
                expected[first..end].copy_from_slice(&input[first..end]);
                expected[first..end].sort_unstable();
            }
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let mut e = device.create_command_encoder(&Default::default());
            owner.encode(&mut e);
            queue.submit([e.finish()]);
            assert_eq!(allocations.finish().observations, 0);
            let traffic = traffic.finish();
            assert_eq!(traffic.upload_bytes + traffic.readback_bytes, 0);
            let actual = crate::soil_shared_wall_friction_gpu::capture::read(&engine, &output);
            assert_eq!(
                actual,
                bytemuck::cast_slice::<_, u8>(&expected),
                "capacity={capacity} counts={counts:?}"
            );
        }
        let sentinel = vec![123u32; 2 * capacity as usize * 8];
        queue.write_buffer(&output, 0, bytemuck::cast_slice(&sentinel));
        queue.write_buffer(&status, 0, bytemuck::bytes_of(&16u32));
        let mut e = device.create_command_encoder(&Default::default());
        owner.encode(&mut e);
        queue.submit([e.finish()]);
        assert_eq!(
            crate::soil_shared_wall_friction_gpu::capture::read(&engine, &output),
            bytemuck::cast_slice::<_, u8>(&sentinel)
        );
        assert_eq!(
            crate::soil_shared_wall_friction_gpu::capture::read(&engine, &status),
            16u32.to_le_bytes()
        );
        queue.write_buffer(&status, 0, bytemuck::bytes_of(&0u32));
        let mut e = device.create_command_encoder(&Default::default());
        owner.encode(&mut e);
        queue.submit([e.finish()]);
        let actual = crate::soil_shared_wall_friction_gpu::capture::read(&engine, &output);
        assert_ne!(actual, bytemuck::cast_slice::<_, u8>(&sentinel));
        println!(
            "canonical nodes capacity={capacity}: two-body exact multiset, changing counts, empty/tail clearing, fault/reset and zero hot resources pass"
        );
    }
}
