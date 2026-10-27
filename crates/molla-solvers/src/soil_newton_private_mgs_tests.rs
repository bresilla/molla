use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu as Solver, SoilNewtonCapacity, SoilNewtonInputs,
};
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires protected GPU and MOLLA_NEWTON_NORM_COMPARE=private-mgs"]
fn newton_private_mgs_gpu_boundaries() {
    assert_eq!(
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref(),
        Ok("private-mgs")
    );
    let engine =
        molla_compute::Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("private MGS fixture"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    const C: usize = 601;
    let mut data = vec![0.0f32; C * 16];
    let mut initial = vec![[0.0f64; 6]; C];
    for c in 0..C {
        for axis in 0..6 {
            let value = ((c * 7 + axis * 3) % 19) as f32 / 4096.0 - 9.0 / 4096.0;
            data[c * 16 + axis + usize::from(axis >= 3)] = value;
            initial[c][axis] = value as f64;
        }
    }
    let contacts = storage(&data);
    let zero = storage(&[0.0]);
    let rows = storage(&vec![0.0; C * 4]);
    let rates = storage(&[0.0; 2]);
    let solvers = [false, true].map(|candidate| {
        Solver::new_norm_tree_edge_probe(
            &engine,
            SoilNewtonInputs {
                mass: &zero,
                free_rates: &rates,
                point_rows: &rows,
                point_response: &rows,
                contacts: &contacts,
                upstream_status: &zero,
            },
            SoilNewtonCapacity {
                dofs: 1,
                contacts: C as u32,
                newton_iterations: 64,
                krylov_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
            candidate,
        )
        .unwrap()
    });
    for count in [0usize, 1, 255, 256, 257, C] {
        for j in [0usize, 7, 31, 63] {
            data[12] = f32::from_bits(count as u32);
            data[13] = f32::from_bits(j as u32);
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&data));
            let mut reference = initial.clone();
            let mut h = [0.0f64; 64];
            for _ in 0..2 {
                for i in 0..=j {
                    let mut projection = 0.0;
                    for slot in 0..count {
                        let c = C - 1 - slot;
                        let factor = (1 + (i * 5 + c * 3) % 7) as f32 / 8.0;
                        for axis in 0..6 {
                            projection +=
                                (initial[c][axis] as f32 * factor) as f64 * reference[c][axis];
                        }
                    }
                    h[i] += projection;
                    for slot in 0..count {
                        let c = C - 1 - slot;
                        let factor = (1 + (i * 5 + c * 3) % 7) as f32 / 8.0;
                        for axis in 0..6 {
                            reference[c][axis] -=
                                projection * (initial[c][axis] as f32 * factor) as f64;
                        }
                    }
                }
            }
            let mut expected = None;
            for solver in &solvers {
                for _ in 0..2 {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    solver.encode_reset(&mut encoder);
                    solver.encode(&mut encoder);
                    queue.submit([encoder.finish()]);
                    let scratch = capture::read(&engine, solver.transpose_probe_output());
                    let coefficients =
                        capture::read(&engine, solver.precision_local_buffer())[..512].to_vec();
                    let snapshot = [scratch, coefficients];
                    if let Some(old) = &expected {
                        assert_eq!(&snapshot, old, "count={count} j={j}");
                    } else {
                        expected = Some(snapshot.clone());
                    }
                    let word = |bytes: &[u8], i: usize| {
                        f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
                    };
                    for c in 0..C {
                        for axis in 0..6 {
                            let actual = word(&snapshot[0], c * 8 + axis + usize::from(axis >= 3));
                            if c < C - count {
                                assert_eq!(actual.to_bits(), (initial[c][axis] as f32).to_bits());
                            }
                            assert!(
                                actual.is_finite()
                                    && (actual as f64 - reference[c][axis]).abs() <= 2e-5,
                                "scratch count={count} j={j} c={c} axis={axis}"
                            );
                        }
                    }
                    for i in 0..64 {
                        let actual = word(&snapshot[1], i * 2);
                        assert!(
                            actual.is_finite() && (actual as f64 - h[i]).abs() <= 2e-5,
                            "h count={count} j={j} i={i}"
                        );
                    }
                }
            }
            eprintln!(
                "private MGS count={count} j={j}: exact variants/repeats, inactive preservation and f64 reference pass"
            );
        }
    }
}
