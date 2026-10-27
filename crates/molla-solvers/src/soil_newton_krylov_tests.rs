use super::*;
use crate::soil_newton_krylov_gpu::{
    ExperimentalSoilNewtonKrylovGpu, SoilNewtonKrylovCapacity, SoilNewtonKrylovInputs,
};

fn capacity(n: usize, c: usize) -> SoilNewtonKrylovCapacity {
    SoilNewtonKrylovCapacity {
        dofs: n as u32,
        contacts: c as u32,
        iterations: 64,
        relative_tolerance: 1e-5,
        max_owned_bytes: u64::MAX,
    }
}

pub(super) fn create_krylov(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    cap: SoilNewtonKrylovCapacity,
) -> ExperimentalSoilNewtonKrylovGpu {
    let [rows, response, contacts] = op.point_inputs();
    let inputs = || SoilNewtonKrylovInputs {
        blocks: op.blocks_buffer(),
        point_rows: rows,
        point_response: response,
        contacts,
        operator_status: op.status_buffer(),
    };
    let required = ExperimentalSoilNewtonKrylovGpu::required_bytes(cap).unwrap();
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ExperimentalSoilNewtonKrylovGpu::new(
            engine,
            inputs(),
            SoilNewtonKrylovCapacity {
                max_owned_bytes: required - 1,
                ..cap
            }
        )
        .is_err()
    );
    assert!(
        ExperimentalSoilNewtonKrylovGpu::new(
            engine,
            inputs(),
            SoilNewtonKrylovCapacity {
                iterations: 65,
                ..cap
            }
        )
        .is_err()
    );
    assert_eq!(scope.finish().observations, 0);
    let solver = ExperimentalSoilNewtonKrylovGpu::new(
        engine,
        inputs(),
        SoilNewtonKrylovCapacity {
            max_owned_bytes: required,
            ..cap
        },
    )
    .unwrap();
    assert_eq!(solver.owned_buffer_bytes(), required);
    solver
}

pub(super) fn run(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonKrylovGpu,
    reset: bool,
) -> [u32; 8] {
    run_method(engine, solver, reset, false)
}

fn run_method(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonKrylovGpu,
    reset: bool,
    least_squares: bool,
) -> [u32; 8] {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        solver.encode_reset(&mut encoder);
    }
    if least_squares {
        solver.encode_least_squares(&mut encoder);
    } else {
        solver.encode(&mut encoder);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatches.finish().encoded_dispatches, 2);
    bytemuck::pod_read_unaligned(&capture::read(engine, solver.status_buffer())[..32])
}

fn check_direction(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    solver: &ExperimentalSoilNewtonKrylovGpu,
    p: &Reference,
    c: usize,
) -> Vec<f64> {
    check_method(engine, op, solver, p, c, false)
}

fn check_method(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    solver: &ExperimentalSoilNewtonKrylovGpu,
    p: &Reference,
    c: usize,
    least_squares: bool,
) -> Vec<f64> {
    let status = run_method(engine, solver, true, least_squares);
    println!(
        "resident Krylov contacts={} status={:?} relative={:e}",
        p.contacts.len(),
        &status[..4],
        f32::from_bits(status[4])
    );
    assert!(status[1] <= 64);
    assert_eq!(status[2] as usize, p.contacts.len());
    let out = values(engine, solver.direction_buffer());
    let blocks = values(engine, op.blocks_buffer());
    let mut d = Vec::new();
    let mut f = Vec::new();
    let mut direct = Vec::new();
    let mut machine = Vec::new();
    for contact in &p.contacts {
        let c = contact.index;
        d.extend((0..6).map(|i| out[8 * c + if i < 3 { i } else { i + 1 }] as f64));
        f.extend((0..6).map(|i| blocks[60 * c + i] as f64));
        direct.push(std::array::from_fn(|i| {
            std::array::from_fn(|j| blocks[60 * c + 6 + 6 * i + j] as f64)
        }));
        machine.push(std::array::from_fn(|i| {
            std::array::from_fn(|j| blocks[60 * c + 42 + 3 * i + j] as f64)
        }));
    }
    let linear = Linearization {
        problem: p,
        direct,
        machine,
    };
    if status[0] != 0 && !least_squares {
        let pre = values(engine, solver.scratch_buffers()[0]);
        let pre: Vec<Block> = p
            .contacts
            .iter()
            .map(|c| {
                std::array::from_fn(|i| {
                    std::array::from_fn(|j| pre[36 * c.index + 6 * i + j] as f64)
                })
            })
            .collect();
        let rhs: Vec<_> = f.iter().map(|x| -x).collect();
        let (cpu, iterations, error) = gmres(|v| linear.product(&apply_blocks(&pre, v)), &rhs, 64);
        eprintln!(
            "CPU direction norm={:e}, GPU direction norm={:e}",
            norm(&apply_blocks(&pre, &cpu)),
            norm(&d)
        );
        let exact_pre = p.preconditioner(&vec![0.0; d.len()]);
        let pre_error = pre
            .iter()
            .flatten()
            .flatten()
            .zip(exact_pre.iter().flatten().flatten())
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0_f64, f64::max);
        let basis = values(engine, solver.scratch_buffers()[1]);
        let mut columns = Vec::new();
        for j in 0..(status[1] as usize).min(12) {
            columns.push(
                p.contacts
                    .iter()
                    .flat_map(|contact| {
                        (0..6).map(|i| {
                            basis[16 * (j * c.max(1) + contact.index)
                                + if i < 3 { i } else { i + 1 }] as f64
                        })
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let mut orthogonal = 0.0_f64;
        for i in 0..columns.len() {
            for j in 0..=i {
                orthogonal = orthogonal
                    .max((dot(&columns[i], &columns[j]) - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        eprintln!(
            "Krylov diagnostic CPU iterations={iterations} relative={error:e} pre_error={pre_error:e} basis_orthogonal={orthogonal:e} estimated={:e}",
            f32::from_bits(status[7])
        );
    }
    assert_eq!(status[0], 0, "Krylov status={status:?}");
    let residual: Vec<_> = linear
        .product(&d)
        .iter()
        .zip(&f)
        .map(|(a, b)| a + b)
        .collect();
    let relative = if norm(&f) > 0.0 {
        norm(&residual) / norm(&f)
    } else {
        norm(&residual)
    };
    assert!(
        relative <= 1e-5,
        "independent linear relative error {relative:e}"
    );
    for i in 0..c.max(1) {
        assert_eq!(out[8 * i + 3].to_bits(), 0);
        assert_eq!(out[8 * i + 7].to_bits(), 0);
        if !p.contacts.iter().any(|contact| contact.index == i) {
            assert!(out[8 * i..8 * (i + 1)].iter().all(|x| x.to_bits() == 0));
        }
    }
    d
}

#[test]
#[ignore = "requires GPU; resident LSQR against independent true linear residual"]
fn resident_lsqr_gpu_fixtures() {
    let engine = Engine::new_gpu().unwrap();
    let mut empty = pair(false, 0.0);
    empty.contacts.clear();
    empty.rows.clear();
    for f in [
        pair(false, 0.0),
        pair(true, 0.3),
        articulated(7),
        articulated(65),
        wide_fixture(),
        empty,
        spread_contacts(&articulated(65), 4096, 63),
    ] {
        let (p, mut x, v) = sample(&f);
        x.fill(0.0);
        let (op, _, _) = create_operator(&engine, &f, &p, &x, &v);
        assert_eq!(encode(&engine, &op, true), 0);
        let solver = create_krylov(&engine, &op, capacity(f.free.len(), f.contacts.len()));
        let first = check_method(&engine, &op, &solver, &p, f.contacts.len(), true);
        let second = check_method(&engine, &op, &solver, &p, f.contacts.len(), true);
        assert_eq!(first, second);
    }
}

#[test]
fn resident_krylov_shader_and_budget() {
    let cap = capacity(32, 4096);
    assert_eq!(
        ExperimentalSoilNewtonKrylovGpu::required_bytes(cap).unwrap(),
        18038832
    );
    for iterations in [0, 65] {
        assert!(
            ExperimentalSoilNewtonKrylovGpu::required_bytes(SoilNewtonKrylovCapacity {
                iterations,
                ..cap
            })
            .is_err()
        );
    }
    for relative_tolerance in [0.0, 1.0, f32::NAN] {
        assert!(
            ExperimentalSoilNewtonKrylovGpu::required_bytes(SoilNewtonKrylovCapacity {
                relative_tolerance,
                ..cap
            })
            .is_err()
        );
    }
    let source = format!(
        "{}\n{}",
        crate::soil_rate_encoding::arithmetic(64),
        include_str!("../shaders/soil_newton_krylov.wgsl")
    );
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[ignore = "requires GPU; bounded resident GMRES against f64 linear residual"]
fn resident_krylov_gpu_fixtures() {
    let engine = Engine::new_gpu().unwrap();
    let mut empty = pair(false, 0.0);
    empty.contacts.clear();
    empty.rows.clear();
    for f in [
        pair(false, 0.0),
        pair(true, 0.3),
        articulated(7),
        articulated(65),
        wide_fixture(),
        empty,
    ] {
        let (p, mut x, v) = sample(&f);
        x.fill(0.0);
        let (op, _, _) = create_operator(&engine, &f, &p, &x, &v);
        assert_eq!(encode(&engine, &op, true), 0);
        let solver = create_krylov(&engine, &op, capacity(f.free.len(), f.contacts.len()));
        let first = check_direction(&engine, &op, &solver, &p, f.contacts.len());
        let second = check_direction(&engine, &op, &solver, &p, f.contacts.len());
        assert_eq!(first, second);
    }
}

#[test]
#[ignore = "requires GPU; sparse retained basis, bounded rejection and reset"]
fn resident_krylov_gpu_sparse_and_faults() {
    for least_squares in [false, true] {
        sparse_and_faults(least_squares);
    }
}

fn sparse_and_faults(least_squares: bool) {
    let run = |engine: &Engine, solver: &ExperimentalSoilNewtonKrylovGpu, reset| {
        run_method(engine, solver, reset, least_squares)
    };
    let check_direction =
        |engine: &Engine,
         op: &ExperimentalSoilNewtonOperatorGpu,
         solver: &ExperimentalSoilNewtonKrylovGpu,
         p: &Reference,
         c| check_method(engine, op, solver, p, c, least_squares);
    let engine = Engine::new_gpu().unwrap();
    let f = spread_contacts(&articulated(65), 4096, 63);
    let (p, mut x, v) = sample(&f);
    x.fill(0.0);
    let (op, _, _) = create_operator(&engine, &f, &p, &x, &v);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = create_krylov(&engine, &op, capacity(f.free.len(), f.contacts.len()));
    let first = check_direction(&engine, &op, &solver, &p, f.contacts.len());
    for buffer in solver
        .scratch_buffers()
        .into_iter()
        .chain([solver.direction_buffer()])
    {
        engine.queue().unwrap().write_buffer(
            buffer,
            0,
            bytemuck::cast_slice(&vec![f32::NAN; buffer.size() as usize / 4]),
        );
    }
    assert_eq!(
        first,
        check_direction(&engine, &op, &solver, &p, f.contacts.len())
    );
    let bounded = create_krylov(
        &engine,
        &op,
        SoilNewtonKrylovCapacity {
            iterations: 1,
            ..capacity(f.free.len(), f.contacts.len())
        },
    );
    let status = run(&engine, &bounded, true);
    assert_eq!(status[0], 8);
    assert_eq!(status[1], 1);
    engine
        .queue()
        .unwrap()
        .write_buffer(op.status_buffer(), 0, bytemuck::bytes_of(&1u32));
    assert_eq!(run(&engine, &solver, true)[0], 1);
    assert_eq!(encode(&engine, &op, true), 0);
    assert_eq!(run(&engine, &solver, false)[0], 1);
    assert_eq!(
        first,
        check_direction(&engine, &op, &solver, &p, f.contacts.len())
    );
    let original = capture::read(&engine, op.blocks_buffer());
    engine
        .queue()
        .unwrap()
        .write_buffer(op.blocks_buffer(), 0, bytemuck::bytes_of(&f32::NAN));
    assert_eq!(run(&engine, &solver, true)[0], 1);
    engine
        .queue()
        .unwrap()
        .write_buffer(op.blocks_buffer(), 0, &original);
    assert_eq!(
        first,
        check_direction(&engine, &op, &solver, &p, f.contacts.len())
    );
}

#[test]
#[ignore = "requires GPU; singular frozen linear system rejection"]
fn resident_krylov_gpu_singular() {
    let engine = Engine::new_gpu().unwrap();
    let f = pair(false, 0.0);
    let (p, x, v) = sample(&f);
    let (op, _, _) = create_operator(&engine, &f, &p, &x, &v);
    assert_eq!(encode(&engine, &op, true), 0);
    let mut blocks = vec![0.0_f32; 120];
    blocks[0] = 1.0;
    blocks[60] = 1.0;
    engine
        .queue()
        .unwrap()
        .write_buffer(op.blocks_buffer(), 0, bytemuck::cast_slice(&blocks));
    let solver = create_krylov(&engine, &op, capacity(3, 2));
    let status = run(&engine, &solver, true);
    assert_eq!(status[0], 8);
    assert_eq!(status[1], 0);
    assert_eq!(status[3], 2);
    assert!((f32::from_bits(status[4]) - 1.0).abs() < 1e-6);
    assert!(
        values(&engine, solver.direction_buffer())
            .iter()
            .all(|x| *x == 0.0)
    );
}

#[test]
#[ignore = "requires GPU; nonsymmetric systems with small and large right-hand sides"]
fn resident_lsqr_gpu_scaled_rhs() {
    let engine = Engine::new_gpu().unwrap();
    let f = pair(false, 0.0);
    let (p, x, v) = sample(&f);
    let (op, _, _) = create_operator(&engine, &f, &p, &x, &v);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = create_krylov(&engine, &op, capacity(3, 2));
    for scale in [1e-20_f32, 1.0, 1e20] {
        let mut blocks = vec![0.0_f32; 120];
        for c in 0..2 {
            for i in 0..6 {
                blocks[60 * c + i] = scale * (i + c + 1) as f32 / 7.0;
                blocks[60 * c + 6 + 6 * i + i] = (i + 1) as f32;
                blocks[60 * c + 6 + 6 * i + (i + 1) % 6] = 0.25;
            }
        }
        engine
            .queue()
            .unwrap()
            .write_buffer(op.blocks_buffer(), 0, bytemuck::cast_slice(&blocks));
        for least_squares in [false, true] {
            check_method(&engine, &op, &solver, &p, 2, least_squares);
        }
    }
}

#[test]
#[ignore = "requires GPU; resident forward and transpose products across lane strides"]
fn resident_lsqr_gpu_products() {
    let engine = Engine::new_gpu().unwrap();
    for count in [7, 64, 65, 129] {
        let f = articulated(count);
        let (p, mut x, v) = sample(&f);
        x.fill(0.0);
        let (op, _, _) = create_operator(&engine, &f, &p, &x, &v);
        assert_eq!(encode(&engine, &op, true), 0);
        compare_operator(&engine, &op, &p, &x, &v, count);
        let solver = create_krylov(&engine, &op, capacity(f.free.len(), f.contacts.len()));
        run(&engine, &solver, true);
        let data: Vec<_> = v.chunks_exact(6).map(vector).collect();
        engine.queue().unwrap().write_buffer(
            solver.scratch_buffers()[2],
            0,
            bytemuck::cast_slice(&data),
        );
        let expected = values(&engine, op.products_buffer());
        for transpose in [false, true] {
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            solver.encode_product(&mut encoder, transpose);
            engine.queue().unwrap().submit([encoder.finish()]);
            let actual = values(&engine, solver.scratch_buffers()[3]);
            for c in 0..count {
                for i in 0..8 {
                    let wanted = expected[16 * c + usize::from(transpose) * 8 + i];
                    let got = actual[8 * c + i];
                    assert!(
                        (got - wanted).abs() <= 1e-6 * (1.0 + wanted.abs()),
                        "count={count} transpose={transpose} c={c} i={i}: {got} != {wanted}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires GPU and saved shared input snapshot"]
fn resident_krylov_gpu_captured_input() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_SHARED_FAILURE_REPLAY").unwrap());
    let (f, pairs) = failure::load_paired(&root).unwrap();
    let p = Reference::new(&f, combined(&pairs));
    let x = vec![0.0; 6 * p.contacts.len()];
    let engine = Engine::new_gpu().unwrap();
    let (op, _, _) = create_operator(&engine, &f, &p, &x, &x);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = create_krylov(&engine, &op, capacity(f.free.len(), f.contacts.len()));
    check_direction(&engine, &op, &solver, &p, f.contacts.len());
}
