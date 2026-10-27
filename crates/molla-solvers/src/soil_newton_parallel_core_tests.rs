use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu as Solver, SoilNewtonCapacity, SoilNewtonInputs,
};
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires protected GPU; dimension edges and independent core assembly oracle"]
fn newton_parallel_core_gpu_edges() {
    let engine =
        molla_compute::Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |words: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("core oracle input"),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let c = 301usize;
    let mut contacts = vec![0.0; c * 16];
    for k in 0..c {
        contacts[16 * k + 3] = if k % 7 == 0 { 0.0 } else { 1.0 };
    }
    let contacts = storage(&contacts);
    let upstream = storage(&[0.0]);
    for n in [1usize, 3, 32] {
        let mut rows = vec![0.0; c * n * 4];
        let mut response = rows.clone();
        let mut mass = vec![0.0; n * n];
        for i in 0..n {
            mass[i * n + i] = 1.0;
        }
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
        let mass = storage(&mass);
        let rates = storage(&vec![0.0; 2 * n]);
        for mode in 0..5u32 {
            let mut expected_bytes: Option<Vec<u8>> = None;
            for parallel in [false, true] {
                let solver = Solver::new_parallel_core_probe(
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
                    parallel,
                    mode,
                )
                .unwrap();
                for repeat in 0..2 {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    solver.encode_reset(&mut encoder);
                    solver.encode(&mut encoder);
                    queue.submit([encoder.finish()]);
                    let report = capture::read(&engine, solver.status_buffer());
                    assert_eq!(
                        u32::from_le_bytes(report[..4].try_into().unwrap()),
                        if mode == 3 { 8 } else { 0 },
                        "n={n} mode={mode} parallel={parallel}"
                    );
                    let all = capture::read(&engine, solver.precision_local_buffer());
                    let mut bytes = all[..4160 * 8].to_vec();
                    if std::env::var_os("MOLLA_NEWTON_TILED_EDGES").is_some()
                        && std::env::var_os("MOLLA_NEWTON_TILED_SHARED").is_none()
                    {
                        bytes[2304 * 8..3840 * 8].fill(0);
                    }
                    if let Some(expected) = &expected_bytes {
                        assert!(
                            bytes == *expected,
                            "core differs n={n} mode={mode} parallel={parallel} repeat={repeat}, first byte {:?}",
                            bytes.iter().zip(expected).position(|(a, b)| a != b)
                        );
                    } else {
                        expected_bytes = Some(bytes.clone());
                    }
                    if mode < 3 {
                        check_oracle(&bytes, n, c, mode, &rows, &response);
                    }
                    if mode == 3 {
                        assert!(bytes.iter().all(|&b| b == 0));
                    }
                }
            }
            eprintln!(
                "parallel core edge n={n} contacts={c} mode={mode}: exact row/cell/repeat equality and applicable f64 oracle pass"
            );
        }
    }
}

fn check_oracle(bytes: &[u8], n: usize, c: usize, mode: u32, rows: &[f32], response: &[f32]) {
    let stride = n + if mode == 0 { 1 } else { 4 };
    let chosen = if mode == 2 { c - 1 } else { 1 };
    let coefficient = |k: usize, j: usize| {
        ((k * 7 + j) % 31) as f64 / 64.0 - 15.0 / 64.0
            + (((k + j) % 5) as f64 - 2.0) * 2f64.powi(-34)
    };
    for i in 0..n {
        for j in 0..stride {
            let mut expected = if i == j { 1.0 } else { 0.0 };
            if j > n {
                expected = response[4 * (chosen * n + i) + j - n - 1] as f64;
            } else {
                for k in 0..c {
                    if k % 7 == 0 || (mode != 0 && k == chosen) {
                        continue;
                    }
                    let r = &response[4 * (k * n + i)..];
                    for a in 0..3 {
                        if j == n {
                            expected += if mode == 0 { 1.0 } else { -1.0 }
                                * r[a] as f64
                                * coefficient(k, 36 + a);
                        } else {
                            let mut projected = 0.0;
                            for b in 0..3 {
                                projected -= r[b] as f64 * coefficient(k, b * 3 + a);
                            }
                            expected += rows[4 * (k * n + j) + a] as f64 * projected;
                        }
                    }
                }
            }
            for offset in [0, if mode == 0 { 0 } else { 1152 }] {
                let at = 8 * (offset + i * stride + j);
                let actual = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as f64
                    + f32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as f64;
                assert!(
                    (actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
                    "core oracle n={n} mode={mode} ({i},{j}) actual={actual:e} expected={expected:e}"
                );
            }
        }
    }
}
