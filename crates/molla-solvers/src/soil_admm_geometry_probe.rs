use super::*;
use crate::soil_admm_geometry_gpu::{
    ExperimentalSoilAdmmGeometryGpu, SoilAdmmGeometryCapacity, SoilAdmmGeometryInputs,
};
use crate::soil_contact_geometry_gpu::SoilContactQuery;

fn fixture(case: usize) -> (Fixture, Vec<SoilContactQuery>, f32) {
    let count = [0, 1, 65, 1, 65, 4096][case];
    let active = if count == 0 {
        0
    } else if count == 1 {
        1
    } else {
        2
    };
    let wall = case == 3 || case == 4;
    let mu = if case == 2 || case == 5 { 0.3 } else { 0. };
    let mut f = Fixture {
        mass: vec![1., 0., 0., 0., 1., 0., 0., 0., 1.],
        free: vec![1., 0., 1.],
        rows: vec![[0.; 4]; count * 3],
        contacts: vec![SoilSharedWallContact::default(); count],
    };
    let mut queries = vec![SoilContactQuery::default(); count.max(1)];
    for slot in 0..active {
        let index = if slot == 0 { 0 } else { count - 1 };
        let axis = if slot == 0 { 0 } else { 2 };
        let mut normal = [0f32; 4];
        normal[axis] = 1.;
        let flags = if wall { 2 << (2 * axis) } else { 0 };
        f.contacts[index] = SoilSharedWallContact {
            soil_mass: [0., 0., 0., 1.],
            normal_gap: normal,
            friction: [mu, 0., 0., 0.],
            boundary: [flags, 0, 0, 0],
        };
        for j in 0..3 {
            f.rows[index * 3 + j][j] = 1.;
        }
        let mut position = [0f32; 4];
        position[axis] = 0.5;
        position[3] = 1.;
        queries[index] = SoilContactQuery {
            position_mass: position,
            velocity_dt: [0., 0., 0., 0.01],
            body_boundary: [0, flags, 0, 0],
        };
    }
    (f, queries, mu)
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
fn admm_geometry_fixture_contract() {
    for case in 0..6 {
        let (f, _, _) = fixture(case);
        let p = Reference::new(&f, f.free.iter().map(|&v| f64::from(v)).collect());
        let x = admm::trial(&p).unwrap().0;
        assert!(
            p.physical_error(&x) <= 1e-6,
            "geometry fixture physical {case}"
        );
        assert!(
            maximum(p.residual(&x).into_iter()) <= 1e-6,
            "geometry fixture residual {case}"
        );
    }
}

fn run(
    engine: &Engine,
    owner: &ExperimentalSoilAdmmGeometryGpu,
    reset: bool,
    empty: bool,
) -> [Vec<u8>; 6] {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        owner.encode_reset(&mut encoder);
    }
    owner.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(
        dispatch.finish().encoded_dispatches,
        if empty { 6 } else { 8 }
    );
    [
        owner.status_buffer(),
        owner.rates_buffer(),
        owner.velocities_buffer(),
        owner.contact_impulses_buffer(),
        owner.wall_impulses_buffer(),
        owner.geometry_buffer(),
    ]
    .map(|b| capture::read(engine, b))
}

fn fail(root: &std::path::Path, label: &str, output: &[Vec<u8>; 6], mask: u32) -> [u32; 8] {
    let status: [u32; 8] = bytemuck::pod_read_unaligned(&output[0]);
    for (name, data) in [
        "status",
        "rates",
        "velocities",
        "impulses",
        "wall",
        "geometry",
    ]
    .iter()
    .zip(output)
    {
        std::fs::write(root.join(format!("{label}-{name}.bin")), data).unwrap();
    }
    assert_ne!(status[0] & mask, 0, "geometry rejection {label}");
    assert!(
        output[1..5].iter().flatten().all(|&v| v == 0),
        "geometry fail closed {label}"
    );
    status
}

pub(super) fn check(engine: &Engine, root: &std::path::Path) -> serde_json::Value {
    let mut results = Vec::new();
    let mut anchor: Option<ExperimentalSoilAdmmGeometryGpu> = None;
    for case in 0..6 {
        let (f, queries, mu) = fixture(case);
        let count = f.contacts.len();
        let path = root.join(format!("geometry-{case}"));
        std::fs::create_dir(&path).unwrap();
        let mass = storage(engine, &f.mass);
        let free = storage(
            engine,
            &f.free.iter().map(|&v| [v, 0f32]).collect::<Vec<_>>(),
        );
        let centers = storage(engine, &[0f32; 4]);
        let shape: f32 = if case == 2 {
            2.
        } else if case == 4 {
            3.
        } else {
            1.
        };
        let shapes = storage(
            engine,
            &[
                [1., 0., 1., 0.],
                [0.; 4],
                [0., 0., 0., 1.],
                [0.5, 0.5, 0.5, shape],
            ],
        );
        let jacobians = storage(
            engine,
            &[
                1f32, 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
        );
        let queries = storage(engine, &queries);
        let coefficients = storage(engine, &[mu]);
        let inputs = || SoilAdmmGeometryInputs {
            mass: &mass,
            free_rates: &free,
            centers: &centers,
            shapes: &shapes,
            jacobians: &jacobians,
            queries: &queries,
            coefficients: &coefficients,
        };
        let mut cap = SoilAdmmGeometryCapacity {
            dofs: 3,
            bodies: 1,
            contacts: count as u32,
            iterations: 64,
            max_owned_bytes: u64::MAX,
        };
        cap.max_owned_bytes = ExperimentalSoilAdmmGeometryGpu::required_bytes(cap).unwrap();
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(
            ExperimentalSoilAdmmGeometryGpu::new(
                engine,
                inputs(),
                SoilAdmmGeometryCapacity {
                    max_owned_bytes: cap.max_owned_bytes - 1,
                    ..cap
                }
            )
            .is_err()
        );
        assert_eq!(allocations.finish().observations, 0);
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let owner = ExperimentalSoilAdmmGeometryGpu::new(engine, inputs(), cap).unwrap();
        assert_eq!(allocations.finish().observations, 27);
        assert_eq!(owner.owned_buffer_bytes(), cap.max_owned_bytes);
        if let Some(anchor) = &anchor {
            assert!(owner.shares_program(anchor));
        }
        let first = run(engine, &owner, true, count == 0);
        for (name, data) in [
            "status",
            "rates",
            "velocities",
            "impulses",
            "wall",
            "geometry",
        ]
        .iter()
        .zip(&first)
        {
            std::fs::write(path.join(format!("{name}.bin")), data).unwrap();
        }
        let status: [u32; 8] = bytemuck::pod_read_unaligned(&first[0]);
        assert_eq!(&status[..4], &[0; 4], "geometry status {case}");
        let p = Reference::new(&f, f.free.iter().map(|&v| f64::from(v)).collect());
        let floats = |bytes: &[u8]| {
            bytes
                .chunks_exact(4)
                .map(|v| f64::from(f32::from_le_bytes(v.try_into().unwrap())))
                .collect::<Vec<_>>()
        };
        let rates = floats(&first[1])
            .chunks_exact(2)
            .map(|v| v[0] + v[1])
            .collect::<Vec<_>>();
        let velocity = floats(&first[2]);
        let impulses = floats(&first[3]);
        let walls = floats(&first[4]);
        let geometry = floats(&first[5]);
        let x = p
            .contacts
            .iter()
            .flat_map(|c| {
                impulses[4 * c.index..4 * c.index + 3]
                    .iter()
                    .chain(&walls[4 * c.index..4 * c.index + 3])
                    .copied()
            })
            .collect::<Vec<_>>();
        let cpu = admm::trial(&p).unwrap().0;
        let physical = p.physical_error(&x);
        let residual = maximum(p.residual(&x).into_iter());
        let cpu_delta = delta(x.iter().copied(), cpu.into_iter()).unwrap();
        assert!(
            physical <= 1e-6 && residual <= 1e-6 && cpu_delta <= 2e-5,
            "geometry solution {case}: {physical} {residual} {cpu_delta}"
        );
        assert!(delta(rates.iter().copied(), p.rates(&x, true).into_iter()).unwrap() <= 2e-5);
        for (c, impulse) in p.contacts.iter().zip(x.chunks_exact(6)) {
            let (soil, _) = Reference::velocities(c, impulse, &rates, true);
            for axis in 0..3 {
                assert!((velocity[4 * c.index + axis] - soil[axis]).abs() <= 1e-6);
                assert_eq!(geometry[4 * c.index + axis], c.normal[axis]);
                if c.wall[axis] != 0. {
                    assert!(c.wall[axis] * velocity[4 * c.index + axis] >= 0.);
                }
            }
            assert_eq!(geometry[4 * c.index + 3], 0.);
            assert!(
                (impulses[4 * c.index + 3]
                    - c.normal.dot(DVec3::new(impulse[0], impulse[1], impulse[2])))
                .abs()
                    <= 1e-6
            );
        }
        for c in 0..count.max(1) {
            if count == 0 || f.contacts[c].soil_mass[3] == 0. {
                for data in &first[2..6] {
                    assert!(data[16 * c..16 * c + 16].iter().all(|&v| v == 0));
                }
            }
        }
        assert_eq!(
            run(engine, &owner, false, count == 0),
            first,
            "geometry repeat {case}"
        );
        let mut faults = Vec::new();
        if count > 0 {
            engine
                .queue()
                .unwrap()
                .write_buffer(&coefficients, 0, &(-1f32).to_le_bytes());
            faults.push(fail(
                &path,
                "coefficient-fault",
                &run(engine, &owner, true, false),
                128,
            ));
            engine
                .queue()
                .unwrap()
                .write_buffer(&coefficients, 0, &mu.to_le_bytes());
            fail(
                &path,
                "latched-fault",
                &run(engine, &owner, false, false),
                128,
            );
            assert_eq!(
                run(engine, &owner, true, false),
                first,
                "geometry reset {case}"
            );
            engine
                .queue()
                .unwrap()
                .write_buffer(&mass, 0, &(-1f32).to_le_bytes());
            faults.push(fail(
                &path,
                "mass-fault",
                &run(engine, &owner, true, false),
                4096,
            ));
            engine
                .queue()
                .unwrap()
                .write_buffer(&mass, 0, &1f32.to_le_bytes());
            assert_eq!(
                run(engine, &owner, true, false),
                first,
                "geometry mass recovery {case}"
            );
        }
        let result = serde_json::json!({"case":case,"slots":count,"active":p.contacts.len(),"shape":shape,"physical":physical,"projected_residual":residual,"cpu_delta":cpu_delta,"status":status,"faults":faults,"owned_bytes":owner.owned_buffer_bytes(),"budget_rejection":true,"shared_program":true,"repeat_exact":true,"recovery_exact":true,"dispatches":if count==0 {6}else {8},"encode_allocations":0,"encode_upload_bytes":0,"encode_readback_bytes":0});
        std::fs::write(
            path.join("result.json"),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
        results.push(result);
        if anchor.is_none() {
            anchor = Some(owner);
        }
    }
    serde_json::json!(results)
}
