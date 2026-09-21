use molla_core::WorldId;
use molla_math::{Real, Vec3};
use molla_sim::soft_belt::{RubberBeltMesh, RubberBeltParams, RubberCordParams};
use molla_sim::{Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, VbdSolver, XpbdSolver};

fn belt(reinforced: bool) -> (RubberBeltMesh, Model) {
    let mesh = RubberBeltMesh::circular(RubberBeltParams {
        segments: 32,
        width_cells: 2,
        thickness_cells: 2,
        ..Default::default()
    })
    .unwrap();
    let mut b = ModelBuilder::new();
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    b.set_linear_damping(0.0);
    if reinforced {
        mesh.append_reinforced_to(&mut b, WorldId(0), Vec3::ZERO, RubberCordParams::default())
            .unwrap();
    } else {
        mesh.append_to(&mut b, WorldId(0), Vec3::ZERO).unwrap();
    }
    (mesh, b.build().unwrap())
}

fn belt_energy(model: &Model, q: &[Vec3], v: &[Vec3]) -> Real {
    let mut energy: Real = v
        .iter()
        .zip(model.particle_mass.host().unwrap())
        .map(|(v, m)| 0.5 * m * v.length_squared())
        .sum();
    for (i, tet) in model
        .tet_indices
        .host()
        .unwrap()
        .chunks_exact(4)
        .enumerate()
    {
        let [a, b, c, d] = [
            q[tet[0] as usize],
            q[tet[1] as usize],
            q[tet[2] as usize],
            q[tet[3] as usize],
        ];
        let f =
            molla_math::Mat3::from_cols(b - a, c - a, d - a) * model.tet_dm_inv.host().unwrap()[i];
        let j = f.determinant();
        let mu = model.tet_mu.host().unwrap()[i];
        let bulk = model.tet_lambda.host().unwrap()[i] + mu;
        let alpha = 1.0 + mu / bulk;
        energy += model.tet_rest_volume.host().unwrap()[i]
            * (0.5 * mu * (f.to_cols_array().iter().map(|v| v * v).sum::<Real>() - 3.0)
                + 0.5 * bulk * ((j - alpha).powi(2) - (1.0 - alpha).powi(2)));
    }
    for s in 0..model.spring_count {
        let a = model.spring_a.host().unwrap()[s] as usize;
        let b = model.spring_b.host().unwrap()[s] as usize;
        let extension =
            ((q[a] - q[b]).length() - model.spring_rest_length.host().unwrap()[s]).max(0.0);
        energy += 0.5 * extension * extension / model.spring_compliance.host().unwrap()[s];
    }
    energy
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn deformed_reinforced_loop_preserves_gpu_momentum() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for offset in [Vec3::ZERO, Vec3::new(0.7, -0.4, 0.9)] {
        let (_, mut model) = belt(true);
        let masses = model.particle_mass.host().unwrap().to_vec();
        let mut state = model.state().unwrap();
        let rotation =
            molla_math::Quat::from_rotation_x(1.3) * molla_math::Quat::from_rotation_y(0.7);
        for q in state.particle_q.host_mut().unwrap() {
            q.y *= 1.02;
            q.z *= 0.99;
            *q = rotation * *q + offset;
        }
        let mut next = model.state().unwrap();
        model.upload_to_device(device, queue).unwrap();
        state.upload_to_device(device, queue).unwrap();
        next.upload_to_device(device, queue).unwrap();
        let mut solver = VbdSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_iterations(128)
            .with_contact_stiffness(0.0);
        for _ in 0..128 {
            solver
                .step(
                    &model,
                    &state,
                    &model.control(),
                    &Contacts::with_capacity(0),
                    1.0 / 19200.0,
                    &mut next,
                )
                .unwrap();
            std::mem::swap(&mut state, &mut next);
        }
        state.particle_qd.download(device, queue).unwrap();
        let momentum: Vec3 = state
            .particle_qd
            .host()
            .unwrap()
            .iter()
            .zip(&masses)
            .map(|(v, m)| *v * *m)
            .sum();
        eprintln!("deformed reinforced loop offset={offset:?}, momentum={momentum:?}");
        assert!(
            momentum.length() < 1e-4,
            "unforced loop changed momentum {momentum:?}"
        );
    }
}

#[test]
fn reinforced_belt_adds_elastic_restoring_force() {
    let mut forces = Vec::new();
    for reinforced in [false, true] {
        let (mesh, model) = belt(reinforced);
        let mut state = model.state().unwrap();
        for q in state.particle_q.host_mut().unwrap() {
            q.y *= 1.02;
            q.z *= 1.02;
        }
        let initial_energy = belt_energy(
            &model,
            state.particle_q.host().unwrap(),
            state.particle_qd.host().unwrap(),
        );
        let mut next = model.state().unwrap();
        VbdSolver::new_cpu()
            .with_iterations(128)
            .with_contact_stiffness(0.0)
            .step(
                &model,
                &state,
                &model.control(),
                &Contacts::with_capacity(0),
                1.0 / 4800.0,
                &mut next,
            )
            .unwrap();
        let radial: Real = mesh
            .positions
            .iter()
            .zip(next.particle_qd.host().unwrap())
            .zip(model.particle_mass.host().unwrap())
            .map(|((q, v), m)| -m * Vec3::new(0.0, q.y, q.z).normalize().dot(*v))
            .sum();
        forces.push(radial);
        assert!(
            belt_energy(
                &model,
                next.particle_q.host().unwrap(),
                next.particle_qd.host().unwrap()
            ) <= initial_energy * 1.001
        );
    }
    assert!(
        forces[1] > 2.0 * forces[0],
        "reinforced radial impulse: {forces:?}"
    );
}

#[test]
fn invalid_tension_elements_do_not_append_constraints() {
    let mut b = ModelBuilder::new();
    b.add_particle(Vec3::ZERO, 1.0, 0.0, WorldId(0));
    b.add_particle(Vec3::X, 1.0, 0.0, WorldId(0));
    for (a, c, rest, k) in [
        (0, 0, 1.0, 1.0),
        (0, 5, 1.0, 1.0),
        (0, 1, 0.0, 1.0),
        (0, 1, Real::NAN, 1.0),
        (0, 1, 1.0, -1.0),
        (0, 1, 1.0, 1e-310),
        (0, 1, 1.0, Real::INFINITY),
    ] {
        assert!(b.add_tension_element(a, c, rest, k).is_err());
    }
    assert_eq!(b.build().unwrap().spring_count, 0);
}

#[test]
fn barrier_solver_rejects_unsupported_tension_elements() {
    let model = element(1000.0);
    let state = model.state().unwrap();
    let mut next = model.state().unwrap();
    let error = molla_solvers::BarrierSolver::new()
        .step(
            &model,
            &state,
            &model.control(),
            &Contacts::with_capacity(0),
            0.01,
            &mut next,
        )
        .unwrap_err();
    assert!(error.to_string().contains("tension-only"));
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn reinforced_rubber_loop_gpu_cpu_energy_and_force_parity() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut restoring = Vec::new();
    for reinforced in [false, true] {
        let (mesh, mut model) = belt(reinforced);
        model.upload_to_device(device, queue).unwrap();
        let mut cpu = model.state().unwrap();
        let mut gpu = model.state().unwrap();
        for state in [&mut cpu, &mut gpu] {
            for q in state.particle_q.host_mut().unwrap() {
                q.y *= 1.02;
                q.z *= 1.02;
            }
        }
        let initial = belt_energy(
            &model,
            cpu.particle_q.host().unwrap(),
            cpu.particle_qd.host().unwrap(),
        );
        let mut cn = model.state().unwrap();
        let mut gn = model.state().unwrap();
        gpu.upload_to_device(device, queue).unwrap();
        gn.upload_to_device(device, queue).unwrap();
        let mut cs = VbdSolver::new_cpu()
            .with_iterations(128)
            .with_contact_stiffness(0.0);
        let mut gs = VbdSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_iterations(128)
            .with_contact_stiffness(0.0);
        for i in 0..48 {
            cs.step(
                &model,
                &cpu,
                &model.control(),
                &Contacts::with_capacity(0),
                1.0 / 4800.0,
                &mut cn,
            )
            .unwrap();
            gs.step(
                &model,
                &gpu,
                &model.control(),
                &Contacts::with_capacity(0),
                1.0 / 4800.0,
                &mut gn,
            )
            .unwrap();
            std::mem::swap(&mut cpu, &mut cn);
            std::mem::swap(&mut gpu, &mut gn);
            if i == 0 {
                let radial: Real = mesh
                    .positions
                    .iter()
                    .zip(cpu.particle_qd.host().unwrap())
                    .zip(model.particle_mass.host().unwrap())
                    .map(|((q, v), m)| m * Vec3::new(0.0, q.y, q.z).normalize().dot(*v))
                    .sum();
                restoring.push(-radial);
            }
        }
        assert!(gpu.particle_q.host().is_err());
        assert!(gpu.particle_qd.host().is_err());
        gpu.particle_q.download(device, queue).unwrap();
        gpu.particle_qd.download(device, queue).unwrap();
        let q = gpu.particle_q.host().unwrap();
        let v = gpu.particle_qd.host().unwrap();
        let error = q
            .iter()
            .zip(cpu.particle_q.host().unwrap())
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0, Real::max);
        let momentum: Vec3 = v
            .iter()
            .zip(model.particle_mass.host().unwrap())
            .map(|(v, m)| *v * *m)
            .sum();
        let final_energy = belt_energy(&model, q, v);
        let angular: Vec3 = q
            .iter()
            .zip(v)
            .zip(model.particle_mass.host().unwrap())
            .map(|((q, v), m)| q.cross(*v * *m))
            .sum();
        eprintln!(
            "reinforced={reinforced}: energy {initial} -> {final_energy}, CPU/GPU position error={error}, P={momentum:?}, restoring={:?}",
            restoring.last()
        );
        assert!(error < 1e-4);
        assert!(momentum.length() < 1e-3);
        assert!(
            angular.length() < 1e-3,
            "free reinforced angular momentum {angular:?}"
        );
        assert!(final_energy < initial * 1.001);
        assert!(mesh.minimum_volume_ratio(q).unwrap() > 0.75);
    }
    assert!(
        restoring[1] > restoring[0] * 2.0,
        "cords must increase longitudinal restoring force: {restoring:?}"
    );
}

fn element(stiffness: Real) -> Model {
    let mut b = ModelBuilder::new();
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    b.set_linear_damping(0.0);
    let a = b.add_particle(Vec3::ZERO, 1.0, 0.0, WorldId(0));
    let c = b.add_particle(Vec3::X, 1.0, 0.0, WorldId(0));
    b.pin_particle(a);
    b.add_tension_element(a, c, 1.0, stiffness).unwrap();
    b.build().unwrap()
}

fn check_element(model: &Model, solver: &mut dyn Solver, length: Real, dt: Real, stiffness: Real) {
    let mut state = model.state().unwrap();
    state.particle_q.host_mut().unwrap()[1].x = length;
    let mut next = model.state().unwrap();
    solver
        .step(
            model,
            &state,
            &model.control(),
            &Contacts::with_capacity(0),
            dt,
            &mut next,
        )
        .unwrap();
    let expected = if length <= 1.0 {
        length
    } else {
        (length + dt * dt * stiffness) / (1.0 + dt * dt * stiffness)
    };
    assert!((next.particle_q.host().unwrap()[1].x - expected).abs() < 1e-11);
    assert!((next.particle_qd.host().unwrap()[1].x - (expected - length) / dt).abs() < 1e-9);
}

#[test]
fn tension_element_matches_axial_implicit_euler_and_has_no_compression() {
    for (k, dt) in [(1000.0, 0.01), (2e8, 1e-5)] {
        let model = element(k);
        for length in [0.8, 1.0, 1.1] {
            check_element(
                &model,
                &mut VbdSolver::new_cpu().with_iterations(16),
                length,
                dt,
                k,
            );
            check_element(
                &model,
                &mut XpbdSolver::new_cpu().with_iterations(16),
                length,
                dt,
                k,
            );
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_tension_element_matches_axial_implicit_euler() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (k, dt) in [(1000.0, 0.01), (2e8, 1e-5)] {
        let mut model = element(k);
        model.upload_to_device(device, queue).unwrap();
        for xpbd in [false, true] {
            for length in [0.8, 1.0, 1.1] {
                let mut state = model.state().unwrap();
                state.particle_q.host_mut().unwrap()[1].x = length;
                let mut next = model.state().unwrap();
                state.upload_to_device(device, queue).unwrap();
                next.upload_to_device(device, queue).unwrap();
                let mut solver: Box<dyn Solver> = if xpbd {
                    Box::new(XpbdSolver::new_gpu(&engine, 0).unwrap().with_iterations(16))
                } else {
                    Box::new(
                        VbdSolver::new_gpu(&engine, &model)
                            .unwrap()
                            .with_iterations(16),
                    )
                };
                solver
                    .step(
                        &model,
                        &state,
                        &model.control(),
                        &Contacts::with_capacity(0),
                        dt,
                        &mut next,
                    )
                    .unwrap();
                assert!(next.particle_q.host().is_err());
                next.particle_q.download(device, queue).unwrap();
                let expected = if length <= 1.0 {
                    length
                } else {
                    (length + dt * dt * k) / (1.0 + dt * dt * k)
                };
                let actual = next.particle_q.host().unwrap()[1].x;
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "xpbd={xpbd} k={k}, length={length}: {actual} vs {expected}"
                );
            }
        }
    }
}
