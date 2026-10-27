use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu as Solver, SoilNewtonCapacity, SoilNewtonInputs,
};
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires protected GPU; exact norm reductions and f64 edge oracle"]
fn newton_norm_tree_gpu_edges() {
    let quaternary = std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("quaternary");
    let engine =
        molla_compute::Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("norm edge input"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    const C: usize = 601;
    let contacts = storage(&vec![0.0; C * 16]);
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
    for count in [0usize, 1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, C] {
        for mode in 0..10 {
            let mut data = vec![0.0; C * 16];
            let mut reference = 0.0f64;
            let mut invalid = false;
            for c in 0..C {
                data[c * 16 + 3] = f32::NAN;
                data[c * 16 + 7] = f32::INFINITY;
                for i in 0..6 {
                    let base = (((c * 7 + i * 3) % 19) as i32 - 9) as f32 / 16.0;
                    let value = match mode {
                        0 => 0.0,
                        1 => base,
                        2 => base * 1e38,
                        3 => base * 1e-30,
                        4 if c == 0 && i == 0 => f32::NAN,
                        5 if c == 0 && i == 0 => f32::NEG_INFINITY,
                        6 => -0.0,
                        7 if c == C - 1 && i == 0 => f32::NAN,
                        8 => base * f32::MIN_POSITIVE,
                        9 => base * 1e-20,
                        _ => base,
                    };
                    data[c * 16 + i + usize::from(i >= 3)] = value;
                    if c < count {
                        invalid |= !value.is_finite();
                        reference += (value as f64) * (value as f64);
                    }
                }
            }
            data[12] = f32::from_bits(count as u32);
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&data));
            let expected = if invalid {
                -1.0
            } else {
                reference.sqrt() as f32
            };
            let mut bits = None;
            let mut dot_bits = None;
            for solver in &solvers {
                for _ in 0..2 {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    solver.encode_reset(&mut encoder);
                    solver.encode(&mut encoder);
                    queue.submit([encoder.finish()]);
                    let bytes = capture::read(&engine, solver.precision_local_buffer());
                    let actual = f32::from_le_bytes(bytes[..4].try_into().unwrap());
                    if quaternary {
                        let dot = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
                        if let Some(old) = dot_bits {
                            assert_eq!(dot.to_bits(), old, "dot count={count} mode={mode}");
                        } else {
                            dot_bits = Some(dot.to_bits());
                        }
                        if invalid || reference > f32::MAX as f64 {
                            assert!(!dot.is_finite());
                        } else if matches!(mode, 0 | 1 | 6) {
                            assert!(
                                dot.is_finite()
                                    && (dot as f64 - reference).abs() <= 2e-6 * reference,
                                "dot count={count} mode={mode} actual={dot} reference={reference}"
                            );
                        }
                    }
                    if let Some(old) = bits {
                        assert_eq!(actual.to_bits(), old, "norm count={count} mode={mode}");
                    } else {
                        bits = Some(actual.to_bits());
                    }
                    if !expected.is_finite() || expected <= 0.0 {
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    } else if mode == 8 && actual == 0.0 {
                        eprintln!(
                            "norm subnormal inputs flushed: count={count} expected={expected}"
                        );
                    } else {
                        assert!(
                            actual.is_finite()
                                && ((actual as f64) - (expected as f64)).abs()
                                    <= 2e-6 * (expected as f64),
                            "count={count} mode={mode} actual={actual} reference={expected}"
                        );
                    }
                }
            }
            eprintln!(
                "norm edges count={count} mode={mode}: exact variants/repeats; normal f64 oracle or all-subnormal flush contract pass"
            );
        }
    }
}
