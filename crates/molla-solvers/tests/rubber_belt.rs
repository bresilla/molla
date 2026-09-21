use molla_core::WorldId;
use molla_math::{Mat3, Quat, Real, Vec3};
use molla_sim::{
    Contacts, Model, ModelBuilder, State,
    soft_belt::{RubberBeltMesh, RubberBeltParams},
};
use molla_solvers::{Solver, VbdSolver};

fn fixture() -> (RubberBeltMesh, Model) {
    let mesh = RubberBeltMesh::circular(RubberBeltParams {
        segments: 32,
        width_cells: 2,
        thickness_cells: 2,
        ..Default::default()
    })
    .unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.set_linear_damping(8.0);
    mesh.append_to(&mut builder, WorldId(0), Vec3::ZERO)
        .unwrap();
    (mesh, builder.build().unwrap())
}

fn strained_state(model: &Model) -> State {
    let mut state = model.state().unwrap();
    let rotation = Quat::from_rotation_y(0.31);
    for p in state.particle_q.host_mut().unwrap() {
        p.y *= 1.06;
        p.z *= 0.96;
        *p = rotation * *p;
    }
    state
}

fn elastic_energy(mesh: &RubberBeltMesh, model: &Model, q: &[Vec3]) -> Real {
    let mu = model.tet_mu.host().unwrap();
    let lambda = model.tet_lambda.host().unwrap();
    let inv = model.tet_dm_inv.host().unwrap();
    mesh.tets
        .iter()
        .enumerate()
        .map(|(i, tet)| {
            let [a, b, c, d] = tet.map(|v| q[v as usize]);
            let f = Mat3::from_cols(b - a, c - a, d - a) * inv[i];
            let j = f.determinant();
            let bulk = lambda[i] + mu[i];
            let alpha = 1.0 + mu[i] / bulk;
            mesh.volumes[i]
                * (0.5 * mu[i] * (f.to_cols_array().iter().map(|v| v * v).sum::<Real>() - 3.0)
                    + 0.5 * bulk * ((j - alpha).powi(2) - (1.0 - alpha).powi(2)))
        })
        .sum()
}

#[test]
fn free_rubber_loop_recovers_without_prescribed_motion() {
    let (mesh, model) = fixture();
    let mut current = strained_state(&model);
    let mut next = model.state().unwrap();
    let initial_energy = elastic_energy(&mesh, &model, current.particle_q.host().unwrap());
    assert!(initial_energy > 1.0);
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(16)
        .with_contact_stiffness(0.0);
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut history = String::from("time_s,elastic_energy_j,min_volume_ratio\n");
    for i in 0..480 {
        solver
            .step(
                &model,
                &current,
                &control,
                &contacts,
                1.0 / 480.0,
                &mut next,
            )
            .unwrap();
        std::mem::swap(&mut current, &mut next);
        let q = current.particle_q.host().unwrap();
        let ratio = mesh.minimum_volume_ratio(q).unwrap();
        assert!(ratio > 0.65, "element collapse at {i}: J={ratio}");
        if i % 12 == 0 {
            history.push_str(&format!(
                "{},{},{}\n",
                (i + 1) as Real / 480.,
                elastic_energy(&mesh, &model, q),
                ratio
            ));
        }
    }
    let final_energy = elastic_energy(&mesh, &model, current.particle_q.host().unwrap());
    eprintln!(
        "rubber loop: {} nodes, {} tets, energy {initial_energy:.6} -> {final_energy:.6} J",
        mesh.positions.len(),
        mesh.tets.len()
    );
    assert!(
        final_energy < initial_energy * 0.15,
        "rubber did not release stored strain energy"
    );
    assert!(
        current
            .particle_qd
            .host()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
    if let Some(path) = std::env::var_os("MOLLA_BELT_EVIDENCE_DIR") {
        let path = std::path::Path::new(&path);
        std::fs::create_dir_all(path).unwrap();
        std::fs::write(path.join("recovery.csv"), history).unwrap();
        for (name, positions) in [
            ("rest", mesh.positions.as_slice()),
            ("recovered", current.particle_q.host().unwrap()),
        ] {
            let mut obj = positions
                .iter()
                .map(|q| format!("v {} {} {}\n", q.x, q.y, q.z))
                .collect::<String>();
            for tri in molla_sim::compute_tet_surface_triangles(&mesh.tets) {
                obj.push_str(&format!("f {} {} {}\n", tri[0] + 1, tri[1] + 1, tri[2] + 1));
            }
            std::fs::write(path.join(format!("{name}.obj")), obj).unwrap();
        }
    }
}

#[test]
fn rigid_rotation_does_not_create_rubber_stress() {
    let (mesh, model) = fixture();
    let mut state = model.state().unwrap();
    let rotation = Quat::from_rotation_y(1.1) * Quat::from_rotation_x(0.7);
    for q in state.particle_q.host_mut().unwrap() {
        *q = rotation * *q + Vec3::new(0.3, 0.8, -0.2);
    }
    let initial = state.particle_q.host().unwrap().to_vec();
    let mut next = model.state().unwrap();
    VbdSolver::new_cpu()
        .with_iterations(16)
        .step(
            &model,
            &state,
            &model.control(),
            &Contacts::with_capacity(0),
            1.0 / 480.0,
            &mut next,
        )
        .unwrap();
    let error = initial
        .iter()
        .zip(next.particle_q.host().unwrap())
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0, Real::max);
    assert!(error < 1e-10, "rigid-motion error {error}");
    assert!(elastic_energy(&mesh, &model, &initial).abs() < 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn rubber_loop_cpu_gpu_deformation_parity() {
    let (mesh, mut model) = fixture();
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut cpu = strained_state(&model);
    let mut cpu_next = model.state().unwrap();
    let mut gpu = strained_state(&model);
    let mut gpu_next = model.state().unwrap();
    gpu.upload_to_device(&device, &queue).unwrap();
    gpu_next.upload_to_device(&device, &queue).unwrap();
    let mut cpu_solver = VbdSolver::new_cpu()
        .with_iterations(12)
        .with_contact_stiffness(0.0);
    let mut gpu_solver = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(12)
        .with_contact_stiffness(0.0);
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    for _ in 0..48 {
        cpu_solver
            .step(
                &model,
                &cpu,
                &control,
                &contacts,
                1.0 / 480.0,
                &mut cpu_next,
            )
            .unwrap();
        gpu_solver
            .step(
                &model,
                &gpu,
                &control,
                &contacts,
                1.0 / 480.0,
                &mut gpu_next,
            )
            .unwrap();
        std::mem::swap(&mut cpu, &mut cpu_next);
        std::mem::swap(&mut gpu, &mut gpu_next);
    }
    gpu.particle_q.download(&device, &queue).unwrap();
    let error = cpu
        .particle_q
        .host()
        .unwrap()
        .iter()
        .zip(gpu.particle_q.host().unwrap())
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0, Real::max);
    eprintln!("rubber CPU/GPU maximum position difference: {error} m");
    assert!(error < 0.002);
    assert!(
        mesh.minimum_volume_ratio(gpu.particle_q.host().unwrap())
            .unwrap()
            > 0.65
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn rubber_loop_gpu_rest_and_momentum_convergence() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for moving in [false, true] {
        let mut coarse_error = 0.0;
        for iterations in [16, 64] {
            let (mesh, mut model) = fixture();
            model.linear_damping = 0.0;
            model.upload_to_device(device, queue).unwrap();
            let mut gpu = model.state().unwrap();
            let mut cpu = model.state().unwrap();
            if moving {
                gpu.particle_qd.host_mut().unwrap()[0] = Vec3::Z * 2.0;
                cpu.particle_qd.host_mut().unwrap()[0] = Vec3::Z * 2.0;
            }
            let initial: Vec3 = gpu
                .particle_qd
                .host()
                .unwrap()
                .iter()
                .zip(&mesh.masses)
                .map(|(v, m)| *v * *m)
                .sum();
            let mut gpu_next = model.state().unwrap();
            let mut cpu_next = model.state().unwrap();
            gpu.upload_to_device(device, queue).unwrap();
            gpu_next.upload_to_device(device, queue).unwrap();
            let mut gpu_solver = VbdSolver::new_gpu(&engine, &model)
                .unwrap()
                .with_iterations(iterations)
                .with_contact_stiffness(0.0);
            let mut cpu_solver = VbdSolver::new_cpu()
                .with_iterations(iterations)
                .with_contact_stiffness(0.0);
            let control = model.control();
            let contacts = Contacts::with_capacity(0);
            for _ in 0..48 {
                gpu_solver
                    .step(
                        &model,
                        &gpu,
                        &control,
                        &contacts,
                        1.0 / 2400.0,
                        &mut gpu_next,
                    )
                    .unwrap();
                cpu_solver
                    .step(
                        &model,
                        &cpu,
                        &control,
                        &contacts,
                        1.0 / 2400.0,
                        &mut cpu_next,
                    )
                    .unwrap();
                std::mem::swap(&mut gpu, &mut gpu_next);
                std::mem::swap(&mut cpu, &mut cpu_next);
            }
            assert!(gpu.particle_qd.host().is_err());
            gpu.particle_qd.download(device, queue).unwrap();
            let momentum: Vec3 = gpu
                .particle_qd
                .host()
                .unwrap()
                .iter()
                .zip(&mesh.masses)
                .map(|(v, m)| *v * *m)
                .sum();
            let cpu_momentum: Vec3 = cpu
                .particle_qd
                .host()
                .unwrap()
                .iter()
                .zip(&mesh.masses)
                .map(|(v, m)| *v * *m)
                .sum();
            eprintln!(
                "free rubber: moving={moving}, iterations={iterations}, initial={initial:?}, GPU P={momentum:?}, CPU P={cpu_momentum:?}"
            );
            if !moving {
                assert!(
                    gpu.particle_qd
                        .host()
                        .unwrap()
                        .iter()
                        .all(|v| *v == Vec3::ZERO)
                );
            }
            assert!(
                (momentum - cpu_momentum).length() < 5e-4,
                "GPU/CPU momentum disagreement"
            );
            let error = (momentum - initial).length();
            if iterations == 16 {
                coarse_error = error;
            } else {
                assert!(error < 1e-3, "free rubber momentum drift {error}");
                if moving {
                    assert!(
                        error < coarse_error * 0.05,
                        "elastic iteration refinement did not converge"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn rubber_loop_gpu_rigid_motion_and_sub_ulp_velocity() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for rotated in [false, true] {
        let (_, mut model) = fixture();
        model.linear_damping = 0.0;
        model.upload_to_device(device, queue).unwrap();
        let mut state = model.state().unwrap();
        let velocity = if rotated {
            Vec3::ZERO
        } else {
            Vec3::new(1e-6, 2e-6, 3e-6)
        };
        state.particle_qd.host_mut().unwrap().fill(velocity);
        if rotated {
            let rotation = Quat::from_rotation_y(1.1) * Quat::from_rotation_x(0.7);
            for q in state.particle_q.host_mut().unwrap() {
                *q = rotation * *q + Vec3::new(0.3, 0.8, -0.2);
            }
        }
        let initial = state.particle_q.host().unwrap().to_vec();
        let mut next = model.state().unwrap();
        state.upload_to_device(device, queue).unwrap();
        next.upload_to_device(device, queue).unwrap();
        VbdSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_iterations(64)
            .with_contact_stiffness(0.0)
            .step(
                &model,
                &state,
                &model.control(),
                &Contacts::with_capacity(0),
                1.0 / 2400.0,
                &mut next,
            )
            .unwrap();
        next.particle_q.download(device, queue).unwrap();
        next.particle_qd.download(device, queue).unwrap();
        let displacement = initial
            .iter()
            .zip(next.particle_q.host().unwrap())
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0, Real::max);
        let velocity_error = next
            .particle_qd
            .host()
            .unwrap()
            .iter()
            .map(|v| (*v - velocity).length())
            .fold(0.0, Real::max);
        eprintln!(
            "GPU rigid motion: rotated={rotated}, displacement={displacement}, velocity error={velocity_error}"
        );
        assert!(
            displacement < 2e-6,
            "rigid motion creates elastic displacement"
        );
        assert!(
            velocity_error < if rotated { 2e-3 } else { 1e-10 },
            "rigid motion velocity error {velocity_error}"
        );
    }
}
