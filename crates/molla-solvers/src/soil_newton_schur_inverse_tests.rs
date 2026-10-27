use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu as Solver, SoilNewtonCapacity, SoilNewtonInputs,
};
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires protected GPU; independent inverse oracle and dense fallback edges"]
fn newton_schur_inverse_gpu_edges() {
    schur_inverse_edges(false, false);
}

#[test]
#[ignore = "requires protected GPU; guarded adjugate oracle and fallback edges"]
fn newton_adjugate_gpu_edges() {
    schur_inverse_edges(true, false);
}

#[test]
#[ignore = "requires protected GPU; pivot suffix inverse oracle and fallback edges"]
fn newton_pivot_suffix_gpu_edges() {
    schur_inverse_edges(false, true);
}

fn schur_inverse_edges(adjugate: bool, pivot_suffix: bool) {
    let engine =
        molla_compute::Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Schur inverse edge input"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let c = 301usize;
    let contacts = storage(&vec![0.0; c * 16]);
    let upstream = storage(&[0.0]);
    for n in [1usize, 3, 32] {
        let mut rows = vec![0.0; c * n * 4];
        let mut response = rows.clone();
        for k in 0..c {
            for i in 0..n {
                for a in 0..3 {
                    rows[4 * (k * n + i) + a] =
                        (((k * 3 + i * 5 + a) % 17) as i32 - 8) as f32 / 128.0;
                    response[4 * (k * n + i) + a] =
                        (((k * 7 + i * 3 + a) % 13) as i32 - 6) as f32 / 256.0;
                }
            }
        }
        let row_buffer = storage(&rows);
        let response_buffer = storage(&response);
        let mass = storage(&vec![0.0; n * n]);
        let rates = storage(&vec![0.0; 2 * n]);
        let mut dense = None;
        for candidate in [false, true] {
            let solver = Solver::new_schur_inverse_edge_probe(
                &engine,
                SoilNewtonInputs {
                    mass: &mass,
                    free_rates: &rates,
                    point_rows: &row_buffer,
                    point_response: &response_buffer,
                    contacts: &contacts,
                    upstream_status: &upstream,
                },
                SoilNewtonCapacity {
                    dofs: n as u32,
                    contacts: c as u32,
                    newton_iterations: 64,
                    krylov_iterations: 64,
                    max_owned_bytes: u64::MAX,
                },
                candidate,
                adjugate,
                pivot_suffix,
            )
            .unwrap();
            let mut previous = None;
            for _ in 0..2 {
                let mut encoder = device.create_command_encoder(&Default::default());
                solver.encode_reset(&mut encoder);
                solver.encode(&mut encoder);
                queue.submit([encoder.finish()]);
                let status = capture::read(&engine, solver.status_buffer());
                assert_eq!(u32::from_le_bytes(status[..4].try_into().unwrap()), 2);
                let bytes = capture::read(&engine, solver.inverse_buffer());
                if let Some(old) = &previous {
                    assert_eq!(&bytes, old);
                } else {
                    previous = Some(bytes.clone());
                }
                let inv: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let diagnostics = capture::read(&engine, solver.precision_local_buffer());
                let diagnostics: Vec<f32> = diagnostics
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                for k in 0..c {
                    let mode = k % 7;
                    assert_eq!(
                        diagnostics[2 * k],
                        if mode < 5 { 1.0 } else { 0.0 },
                        "n={n} contact={k} candidate={candidate}"
                    );
                    assert_eq!(
                        diagnostics[2 * k + 1],
                        if candidate && mode == 1 { 1.0 } else { 0.0 }
                    );
                    if mode >= 5 {
                        continue;
                    }
                    let a = coefficient(k, n, &rows, &response);
                    for i in 0..6 {
                        for j in 0..6 {
                            let product: f64 = (0..6)
                                .map(|q| a[i][q] * f64::from(inv[36 * k + 6 * q + j]))
                                .sum();
                            let expected = if i == j { 1.0 } else { 0.0 };
                            assert!(
                                (product - expected).abs() <= 1e-5,
                                "inverse n={n} contact={k} ({i},{j}) {product}"
                            );
                        }
                    }
                    if candidate && mode != 1 {
                        let original: &Vec<u8> = dense.as_ref().unwrap();
                        assert_eq!(
                            &bytes[144 * k..144 * (k + 1)],
                            &original[144 * k..144 * (k + 1)],
                            "dense fallback must be exact"
                        );
                    }
                }
                if !candidate {
                    dense = Some(bytes);
                }
            }
        }
        eprintln!(
            "Schur inverse n={n}: 301 contacts, structured/free/mixed walls, unsupported lower block, zero-diagonal dense fallback, singular and NaN fault cases pass"
        );
    }
}

fn coefficient(k: usize, n: usize, rows: &[f32], response: &[f32]) -> [[f64; 6]; 6] {
    let mode = k % 7;
    let mut matrix = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            let direct = if i == j {
                4.0 + (k % 5) as f64 * 0.125
            } else {
                0.125
            };
            let correction: f64 = (0..3)
                .map(|axis| {
                    let mobility: f64 = (0..n)
                        .map(|d| {
                            f64::from(rows[4 * (k * n + d) + axis])
                                * f64::from(response[4 * (k * n + d) + j])
                        })
                        .sum();
                    (if i == axis { 0.125 } else { 0.03125 }) * f64::from(mobility as f32)
                })
                .sum();
            matrix[i][j] = f64::from((direct - correction) as f32);
            matrix[i][3 + j] = if i == j { 0.25 } else { 0.0625 };
        }
    }
    for i in 0..3 {
        matrix[3 + i][i] = if mode == 1 || (mode == 2 && i == 0) {
            0.0
        } else {
            0.25
        };
        matrix[3 + i][3 + i] = 1.0 + i as f64 * 0.125;
    }
    if mode == 3 {
        matrix[3][1] = 0.125;
    }
    if mode == 4 {
        matrix[3][3] = 0.0;
    }
    matrix
}
