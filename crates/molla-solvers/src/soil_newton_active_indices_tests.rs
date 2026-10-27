use super::*;
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires protected GPU; stable prefix, invalid masses and untouched suffix"]
fn newton_active_indices_gpu_edges() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_ACTIVE_INDICES_EDGES").unwrap());
    std::fs::create_dir(&root).unwrap();
    let buffer = |bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("active-index oracle input"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let mass = buffer(bytemuck::bytes_of(&1f32));
    let free = buffer(&[0; 8]);
    let upstream = buffer(bytemuck::bytes_of(&8u32));
    let mut checks = 0;
    for count in [
        0usize, 1, 2, 255, 256, 257, 301, 1023, 1024, 1025, 4096, 4097,
    ] {
        let size = count.max(1);
        let rows = buffer(&vec![0; size * 16]);
        let contacts = buffer(&vec![0; size * 64]);
        let solvers = [false, true].map(|candidate| {
            probe_new(
                &engine,
                SoilNewtonInputs {
                    mass: &mass,
                    free_rates: &free,
                    point_rows: &rows,
                    point_response: &rows,
                    contacts: &contacts,
                    upstream_status: &upstream,
                },
                SoilNewtonCapacity {
                    dofs: 1,
                    contacts: count as u32,
                    newton_iterations: 64,
                    krylov_iterations: 64,
                    max_owned_bytes: u64::MAX,
                },
                candidate,
                false,
            )
            .unwrap()
        });
        for mode in 0..4 {
            let mut words = vec![0f32; size * 16];
            let mut expected = Vec::new();
            let mut fault = 8u32;
            for c in 0..count {
                let m = match mode {
                    0 => 0.0,
                    1 => 1.0,
                    2 => {
                        if c % 7 == 0 {
                            1.0
                        } else {
                            -0.0
                        }
                    }
                    _ => [
                        1.0,
                        0.0,
                        -1.0,
                        f32::NAN,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                        -0.0,
                    ][c % 7],
                };
                words[c * 16 + 3] = m;
                if !m.is_finite() || m < 0.0 {
                    fault |= 1;
                }
                if m > 0.0 {
                    expected.push(c as f32);
                }
            }
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&words));
            let active = expected.len();
            expected.resize(size, -1.0);
            let mut previous = None;
            for (variant, solver) in solvers.iter().enumerate() {
                for repeat in 0..2 {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    solver.encode(&mut encoder);
                    queue.submit([encoder.finish()]);
                    let all = capture::read(&engine, solver.precision_local_buffer());
                    let bytes = &all[..8 * (size + 1)];
                    let values: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    assert_eq!(&values[..2], &[fault as f32, active as f32]);
                    for (c, &index) in expected.iter().enumerate() {
                        assert_eq!(&values[2 + 2 * c..4 + 2 * c], &[index, 0.0]);
                    }
                    if let Some(previous) = &previous {
                        assert_eq!(bytes, previous);
                    }
                    previous = Some(bytes.to_vec());
                    std::fs::write(
                        root.join(format!("n{count}-m{mode}-v{variant}-r{repeat}.bin")),
                        bytes,
                    )
                    .unwrap();
                    checks += 1;
                }
            }
        }
    }
    assert_eq!(checks, 192);
    eprintln!("stable active indices: {checks} exact oracle/candidate/repeat checks");
}
