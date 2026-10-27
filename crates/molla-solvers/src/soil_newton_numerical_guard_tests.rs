use super::*;

fn maximum(values: impl Iterator<Item = f64>) -> f64 {
    values
        .map(|v| {
            if v.is_finite() {
                v.abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0., f64::max)
}

pub(super) fn check_published(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonGpu,
    p: &Reference,
    x: &[f64],
) {
    let parameters = capture::read(engine, solver.numerical_guard_probe_parameters());
    assert_eq!(
        u32::from_le_bytes(parameters[28..32].try_into().unwrap()),
        1
    );
    let output = values(engine, solver.outputs_buffer());
    let represented: Vec<_> = p
        .contacts
        .iter()
        .flat_map(|c| {
            (0..6).map(|axis| {
                if axis < 3 {
                    -f64::from(output[20 * c.index + axis])
                } else {
                    f64::from(output[20 * c.index + 12 + axis - 3])
                }
            })
        })
        .collect();
    let state_residual = maximum(p.residual(x).into_iter());
    let output_residual = maximum(p.residual(&represented).into_iter());
    assert!(
        state_residual <= 1e-6,
        "paired-state numerical residual={state_residual:e}"
    );
    assert!(
        output_residual <= 1e-6,
        "represented-output numerical residual={output_residual:e}"
    );
    eprintln!(
        "NUMERICAL_GUARD_LIFECYCLE active={} slots={} enabled=1 state_residual={state_residual:e} output_residual={output_residual:e}",
        p.contacts.len(),
        output.len() / 20
    );
}

#[test]
#[ignore = "requires protected GPU; explicitly enabled guard through faults and reuse"]
fn resident_newton_numerical_guard_lifecycle() {
    for flag in [
        "MOLLA_NEWTON_NUMERICAL_GUARD",
        "MOLLA_NEWTON_NUMERICAL_GUARD_LIFECYCLE",
        "MOLLA_NEWTON_TEST_CONDITIONED",
    ] {
        assert!(std::env::var_os(flag).is_some(), "missing {flag}");
    }
    let engine = replay_engine();
    reuse_and_faults_on(&engine, NewtonVariant::Fischer);
    empty_and_mass_on(&engine, NewtonVariant::Fischer);
    for f in [pair(false, 0.), pair(false, 0.3), pair(true, 0.3)] {
        let p = Reference::new(&f, f.free.iter().map(|&q| f64::from(q)).collect());
        let zero = vec![0.; 6 * p.contacts.len()];
        let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
        assert_eq!(encode(&engine, &op, true), 0);
        let solver = create_capacity_arithmetic(
            &engine,
            &op,
            &p,
            capacity(p.free.len(), f.contacts.len()),
            NewtonVariant::Fischer,
        );
        let first = check(&engine, &solver, &f, &p);
        assert_eq!(first, check(&engine, &solver, &f, &p));
    }
}

#[test]
fn newton_numerical_guard_signed_wall_and_nonfinite_contract() {
    assert_eq!(maximum([0., f64::NAN].into_iter()), f64::INFINITY);
    assert_eq!(maximum([f64::NEG_INFINITY].into_iter()), f64::INFINITY);
    for axis in 0..3 {
        for sign in [-1., 1.] {
            let f = pair(false, 0.3);
            let mut p = Reference::new(&f, vec![0.; f.free.len()]);
            p.contacts.truncate(1);
            let c = &mut p.contacts[0];
            c.rows.fill(DVec3::ZERO);
            c.normal = if axis == 1 { DVec3::X } else { DVec3::Y };
            c.mass = 0.028;
            c.rho = c.mass;
            c.soil = DVec3::ZERO;
            c.soil[axis] = sign * 0.001;
            c.gap = 0.;
            c.wall = [0.; 3];
            c.wall[axis] = sign;
            let mut x = vec![0.; 6];
            x[3 + axis] = sign * 9e-5;
            assert!(p.physical_error(&x) <= 1e-6);
            assert!(maximum(p.residual(&x).into_iter()) > 2e-5);
            assert_eq!(maximum(p.residual(&[0.; 6]).into_iter()), 0.);
        }
    }
}

#[test]
fn newton_numerical_guard_rejects_separating_impulse() {
    let f = pair(false, 0.3);
    let mut p = Reference::new(&f, vec![0.; f.free.len()]);
    p.contacts.truncate(1);
    let c = &mut p.contacts[0];
    c.rows.fill(DVec3::ZERO);
    c.normal = DVec3::Z;
    c.mass = 0.028;
    c.rho = c.mass;
    c.soil = DVec3::new(0.4, 0., 0.001);
    c.gap = 0.;
    c.wall = [0.; 3];
    let x = vec![-0.3 * 9e-5, 0., 9e-5, 0., 0., 0.];
    assert!(p.physical_error(&x) <= 1e-6);
    assert!(maximum(p.residual(&x).into_iter()) > 2e-5);
    assert_eq!(maximum(p.residual(&[0.; 6]).into_iter()), 0.);
}

pub(super) fn compare(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonGpu,
    reference: &Reference,
    seed: &[u8],
    archived: &[u8],
    packet: &std::path::Path,
    output: &std::path::Path,
) {
    let bytes = std::fs::read(packet.join("newton.bin")).unwrap();
    let hash = blake3::hash(&bytes).to_hex().to_string();
    let root = output.join(&hash);
    assert!(!root.exists());
    std::fs::create_dir_all(&root).unwrap();
    assert_eq!(seed.len() as u64, solver.state_buffer().size());
    let seed_buffer = storage(engine, seed);
    let execute = |mode: usize| {
        solver.set_numerical_guard_probe(engine, mode != 0);
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        solver.encode_reset(&mut encoder);
        encoder.copy_buffer_to_buffer(&seed_buffer, 0, solver.state_buffer(), 0, seed.len() as u64);
        solver.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        mixed_tests::snapshot(engine, solver)
    };
    let saved = [execute(0), execute(1)];
    for (mode, buffers) in saved.iter().enumerate() {
        for (name, data) in ["states", "rates", "outputs", "status", "publication"]
            .iter()
            .zip(buffers)
        {
            std::fs::write(root.join(format!("{mode}-{name}.bin")), data).unwrap();
        }
    }
    for mode in [1, 0, 0, 1] {
        assert_eq!(
            execute(mode),
            saved[mode],
            "exact five-buffer repeat mode={mode}"
        );
    }
    let exact_control =
        saved[0][0] == archived && saved[0][3] == bytes[64..96] && saved[0][4] == bytes[96..128];
    let reports: [[u32; 8]; 2] =
        std::array::from_fn(|i| bytemuck::pod_read_unaligned(&saved[i][3]));
    let publications: [[u32; 8]; 2] =
        std::array::from_fn(|i| bytemuck::pod_read_unaligned(&saved[i][4]));
    let states: [Vec<f64>; 2] =
        std::array::from_fn(|i| mixed_tests::states(&saved[i][0], reference));
    let physical: [f64; 2] = std::array::from_fn(|i| reference.physical_error(&states[i]));
    let residual: [f64; 2] =
        std::array::from_fn(|i| maximum(reference.residual(&states[i]).into_iter()));
    let admm = admm::trial(reference).unwrap().0;
    let admm_delta = maximum(admm.iter().zip(&states[1]).map(|(a, b)| a - b));
    let public_delta = mixed_tests::public_delta(&saved[0], &saved[1], reference);
    let valid =
        reports[1][0] == 0 && publications[1][0] == 0 && physical[1] <= 1e-6 && residual[1] <= 1e-6;
    let result = serde_json::json!({"acceptance":false,"packet":hash,"scope":"test-only projected-impulse guard; same pipeline and retained seed; not a changed reference acceptance","exact_control":exact_control,"exact_repeats_per_mode":3,"reports":reports,"publication":publications,"physical":physical,"projected_residual":residual,"physical_bound":1e-6,"numerical_bound":1e-6,"public_bound":2e-5,"public_delta":public_delta,"public_passed":public_delta<=2e-5,"admm_delta":admm_delta,"candidate_valid":valid});
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    println!("NUMERICAL_GUARD {result}");
    assert!(
        exact_control,
        "control must retain archived state/report/publication"
    );
}
