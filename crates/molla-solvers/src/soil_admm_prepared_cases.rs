use super::*;
use crate::soil_admm_gpu::{ExperimentalSoilAdmmGpu, SoilAdmmCapacity, SoilAdmmInputs};
use crate::soil_newton_prepare_gpu::{ExperimentalSoilNewtonPrepareGpu, SoilNewtonPrepareInputs};

fn fixture(case: usize) -> Fixture {
    let mut f = pair(case >= 2, if case == 0 || case == 2 { 0. } else { 0.3 });
    if case >= 4 {
        let permutation = if case == 4 { [0, 1, 2] } else { [2, 0, 1] };
        let transform = |value: [f32; 4]| {
            let mut result = [0f32; 4];
            result[3] = value[3];
            for axis in 0..3 {
                result[permutation[axis]] = value[axis] * if axis == 0 { -1. } else { 1. };
            }
            result
        };
        for c in &mut f.contacts {
            c.soil_mass = transform(c.soil_mass);
            c.normal_gap = transform(c.normal_gap);
            c.boundary[0] = 2 << (2 * permutation[0]);
        }
        for row in &mut f.rows {
            *row = transform(*row);
        }
        let count = if case == 4 { 65 } else { 4096 };
        let mut contacts = vec![SoilSharedWallContact::default(); count];
        let mut rows = vec![[0f32; 4]; count * 3];
        for (old, index) in [0, count - 1].into_iter().enumerate() {
            contacts[index] = f.contacts[old];
            rows[index * 3..index * 3 + 3].copy_from_slice(&f.rows[old * 3..old * 3 + 3]);
        }
        f.contacts = contacts;
        f.rows = rows;
    }
    f
}

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

#[test]
fn admm_prepared_fixture_contract() {
    for case in 0..6 {
        let f = fixture(case);
        let p = Reference::new(&f, f.free.iter().map(|&v| f64::from(v)).collect());
        assert_eq!(p.contacts.len(), 2);
        let x = admm::trial(&p).unwrap().0;
        assert!(p.physical_error(&x) <= 1e-6, "fixture physical {case}");
        assert!(
            maximum(p.residual(&x).into_iter()) <= 1e-6,
            "fixture numerical {case}"
        );
        for c in &f.contacts {
            if c.soil_mass[3] > 0. {
                assert!((c.normal_gap[..3].iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-4);
            }
        }
    }
}

fn run(
    engine: &Engine,
    prep: &ExperimentalSoilNewtonPrepareGpu,
    solver: &ExperimentalSoilAdmmGpu,
    reset: bool,
) -> ([u32; 8], [Vec<u8>; 5]) {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        prep.encode_reset(&mut encoder);
        solver.encode_reset(&mut encoder);
    }
    prep.encode(&mut encoder);
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatch.finish().encoded_dispatches, 5);
    let status = capture::read(engine, prep.status_buffer());
    let [state, report] = solver.diagnostic_buffers();
    (
        bytemuck::pod_read_unaligned(&status[..32]),
        [
            state,
            solver.rates_buffer(),
            solver.outputs_buffer(),
            report,
            solver.status_buffer(),
        ]
        .map(|b| capture::read(engine, b)),
    )
}

pub(super) fn check(engine: &Engine, root: &std::path::Path) -> serde_json::Value {
    let mut results = Vec::new();
    let mut anchor: Option<ExperimentalSoilAdmmGpu> = None;
    for case in 0..6 {
        let f = fixture(case);
        let path = root.join(format!("prepared-{case}"));
        std::fs::create_dir(&path).unwrap();
        let mass = storage(engine, &f.mass);
        let rows = storage(engine, &f.rows);
        let contacts = storage(engine, &f.contacts);
        let upstream = storage(engine, &[0u32]);
        let free = storage(
            engine,
            &f.free.iter().map(|&v| [v, 0f32]).collect::<Vec<_>>(),
        );
        let prep = ExperimentalSoilNewtonPrepareGpu::new(
            engine,
            SoilNewtonPrepareInputs {
                mass: &mass,
                point_rows: &rows,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            3,
            f.contacts.len() as u32,
            u64::MAX,
        )
        .unwrap();
        let solver = ExperimentalSoilAdmmGpu::new(
            engine,
            SoilAdmmInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: prep.response_buffer(),
                contacts: &contacts,
                upstream_status: prep.status_buffer(),
            },
            SoilAdmmCapacity {
                dofs: 3,
                contacts: f.contacts.len() as u32,
                iterations: 64,
                max_owned_bytes: u64::MAX,
            },
        )
        .unwrap();
        if let Some(anchor) = &anchor {
            assert!(solver.shares_program(anchor));
        }
        let (prepared, saved) = run(engine, &prep, &solver, true);
        for (name, data) in ["states", "rates", "outputs", "report", "publication"]
            .iter()
            .zip(&saved)
        {
            std::fs::write(path.join(format!("{name}.bin")), data).unwrap();
        }
        std::fs::write(
            path.join("preparation.bin"),
            bytemuck::cast_slice(&prepared),
        )
        .unwrap();
        assert_eq!(prepared[0], 0, "GPU preparation {case}");
        let published: [u32; 8] = bytemuck::pod_read_unaligned(&saved[4]);
        assert_eq!(published[0], 0, "prepared ADMM publication {case}");
        let response = capture::read(engine, prep.response_buffer());
        let prepared_rows: Vec<[f32; 4]> = response
            .chunks_exact(16)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let p = Reference::new(&f, f.free.iter().map(|&v| f64::from(v)).collect());
        let x = mixed_tests::states(&saved[0], &p);
        let mut independent = p.clone();
        independent.use_prepared_response(&prepared_rows);
        let cpu = admm::trial(&independent).unwrap().0;
        let physical = p.physical_error(&x);
        let residual = maximum(independent.residual(&x).into_iter());
        let cpu_delta = delta(x.iter().copied(), cpu.into_iter()).unwrap();
        assert!(
            physical <= 1e-6 && residual <= 1e-6 && cpu_delta <= 2e-5,
            "prepared fixture {case}: physical={physical} residual={residual} cpu={cpu_delta}"
        );
        super::super::super::publication::verify_outputs(
            engine,
            solver.outputs_buffer(),
            solver.rates_buffer(),
            &f,
            &p,
        );
        assert_eq!(
            run(engine, &prep, &solver, false),
            (prepared, saved.clone()),
            "prepared repeat {case}"
        );
        engine
            .queue()
            .unwrap()
            .write_buffer(&mass, 0, &(-1f32).to_le_bytes());
        let (bad_prep, bad) = run(engine, &prep, &solver, true);
        assert_ne!(bad_prep[0], 0);
        assert!(bad[1..3].iter().flatten().all(|&v| v == 0));
        let bad_status: [u32; 8] = bytemuck::pod_read_unaligned(&bad[4]);
        assert_ne!(bad_status[0] & 1, 0);
        engine
            .queue()
            .unwrap()
            .write_buffer(&mass, 0, &1f32.to_le_bytes());
        let (latched, failed) = run(engine, &prep, &solver, false);
        assert_ne!(latched[0], 0);
        assert!(failed[1..3].iter().flatten().all(|&v| v == 0));
        assert_eq!(
            run(engine, &prep, &solver, true),
            (prepared, saved),
            "prepared reset recovery {case}"
        );
        let result = serde_json::json!({"case":case,"slots":f.contacts.len(),"active":2,"physical":physical,"projected_residual":residual,"cpu_delta":cpu_delta,"published":published,"preparation":prepared,"bad_mass_preparation":bad_prep,"bad_mass_publication":bad_status,"repeat_exact":true,"reset_recovery_exact":true,"shared_program":true,"dispatches":5,"encode_allocations":0,"encode_upload_bytes":0,"encode_readback_bytes":0});
        std::fs::write(
            path.join("result.json"),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
        results.push(result);
        if anchor.is_none() {
            anchor = Some(solver);
        }
    }
    serde_json::json!(results)
}
