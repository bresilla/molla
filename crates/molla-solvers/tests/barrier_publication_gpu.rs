#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_kernels::{BarrierGpuParams, BarrierGpuSpring, BarrierNewtonGpu};
use molla_math::{Transform, Vec3};
use molla_sim::{BodyParams, Contacts, ModelBuilder};
use molla_solvers::{BarrierSolver, Solver};
use molla_storage::{Buffer, BufferUsage};

#[test]
#[ignore = "requires GPU; run via make test-ppf-publication"]
fn authored_and_legacy_materials_follow_external_state_edits() {
    use molla_math::constitutive::{ElasticMaterial, ElasticModel};
    let engine = Engine::new_gpu().unwrap();
    for legacy in [false, true] {
        let mut b = ModelBuilder::new();
        for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
            b.add_particle(p, 1.0, 0.01, WorldId(0));
        }
        if legacy {
            b.add_tet_mesh(&[[0, 1, 2, 3]], 1000.0, 0.3, Some(&[0.2]))
                .unwrap();
        } else {
            b.add_elastic_tetrahedra(
                &[[0, 1, 2, 3]],
                ElasticMaterial {
                    model: ElasticModel::StableNeoHookean,
                    mu: 1000.0,
                    lambda: 1000.0,
                },
            )
            .unwrap();
        }
        let mut m = b.build().unwrap();
        let mut input = m.state().unwrap();
        input.particle_q.host_mut().unwrap()[1].x = 1.2;
        let mut solver = BarrierSolver::new_gpu(&engine).unwrap();
        solver.params.gravity = Vec3::ZERO;
        let mut output = m.state().unwrap();
        solver
            .step(
                &m,
                &input,
                &m.control(),
                &Contacts::with_capacity(0),
                1.0 / 240.0,
                &mut output,
            )
            .unwrap();
        assert!(output.particle_q.host().unwrap()[1].x < 1.2);
        for edit_material in [false, true] {
            if edit_material {
                if legacy {
                    m.tet_mu.host_mut().unwrap()[0] *= 2.0;
                } else {
                    let e = m.elastic_elements().as_slice()[0].clone();
                    let mut material = e.material();
                    material.mu *= 2.0;
                    m.set_elastic_elements(vec![e.with_material(material).unwrap()])
                        .unwrap();
                }
            } else {
                output.particle_q.host_mut().unwrap()[1].x = 1.3;
                output.particle_qd.host_mut().unwrap()[2].y = 0.1;
            }
            let mut fresh = BarrierSolver::new_gpu(&engine)
                .unwrap()
                .with_params(solver.params);
            let mut actual = m.state().unwrap();
            let mut expected = m.state().unwrap();
            solver
                .step(
                    &m,
                    &output,
                    &m.control(),
                    &Contacts::with_capacity(0),
                    1.0 / 240.0,
                    &mut actual,
                )
                .unwrap();
            fresh
                .step(
                    &m,
                    &output,
                    &m.control(),
                    &Contacts::with_capacity(0),
                    1.0 / 240.0,
                    &mut expected,
                )
                .unwrap();
            assert_eq!(
                actual.particle_q.host().unwrap(),
                expected.particle_q.host().unwrap()
            );
            assert_eq!(
                actual.particle_qd.host().unwrap(),
                expected.particle_qd.host().unwrap()
            );
            output = actual;
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-publication"]
fn staged_state_matches_direct_commit_including_pins_and_velocity() {
    let engine = Engine::new_gpu().unwrap();
    let new = || {
        BarrierNewtonGpu::new_with_mesh(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            &[[0.0, 1.0, 0.0], [0.1, 1.0, 0.0], [0.0, 1.0, 0.1]],
            &[1.0; 3],
            &[BarrierGpuSpring {
                a: 0,
                b: 1,
                rest: 0.1,
                k: 100.0,
            }],
            &[],
            &[[0, 1, 2]],
            &[0],
            BarrierGpuParams::default(),
        )
        .unwrap()
    };
    let mut staged = new();
    let mut direct = new();
    for _ in 0..12 {
        let (stats, positions, velocities) = staged.step_with_readback(1.0 / 240.0).unwrap();
        let reference = direct.step(1.0 / 240.0).unwrap();
        assert_eq!(stats.newton_iters, reference.newton_iters);
        assert_eq!(positions, direct.read_positions().unwrap());
        assert_eq!(velocities, direct.read_velocities().unwrap());
        assert_eq!(positions, staged.read_positions().unwrap());
        assert_eq!(velocities, staged.read_velocities().unwrap());
        assert_eq!(positions[0], [0.0, 1.0, 0.0]);
        assert_eq!(velocities[0], [0.0; 3]);
    }
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-publication"]
fn output_preflight_does_not_advance_gpu_or_publish_partial_state() {
    let engine = Engine::new_gpu().unwrap();
    let mut b = ModelBuilder::new();
    b.add_particle(Vec3::Y, 1.0, 0.01, WorldId(0));
    b.add_body(BodyParams::new().with_mass(1.0));
    let m = b.build().unwrap();
    let input = m.state().unwrap();
    let control = m.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = BarrierSolver::new_gpu(&engine).unwrap();
    let mut reference = BarrierSolver::new_gpu(&engine).unwrap();
    let mut output = m.state().unwrap();
    let mut expected = m.state().unwrap();
    solver
        .step(&m, &input, &control, &contacts, 1.0 / 240.0, &mut output)
        .unwrap();
    reference
        .step(&m, &input, &control, &contacts, 1.0 / 240.0, &mut expected)
        .unwrap();
    for malformed_length in [true, false] {
        let mut rejected = m.state().unwrap();
        rejected
            .particle_q
            .host_mut()
            .unwrap()
            .fill(Vec3::splat(41.0));
        rejected
            .particle_qd
            .host_mut()
            .unwrap()
            .fill(Vec3::splat(42.0));
        if malformed_length {
            rejected.body_qd = Buffer::from_vec(vec![], BufferUsage::StorageReadWrite);
        } else {
            rejected.body_qd.mark_device_dirty();
        }
        let stats = (solver.last_newton_iters, solver.last_min_gap);
        assert!(
            solver
                .step(&m, &input, &control, &contacts, 1.0 / 240.0, &mut rejected)
                .is_err()
        );
        assert_eq!(rejected.particle_q.host().unwrap(), &[Vec3::splat(41.0)]);
        assert_eq!(rejected.particle_qd.host().unwrap(), &[Vec3::splat(42.0)]);
        assert_eq!((solver.last_newton_iters, solver.last_min_gap), stats);
        solver
            .step(&m, &input, &control, &contacts, 1.0 / 240.0, &mut output)
            .unwrap();
        reference
            .step(&m, &input, &control, &contacts, 1.0 / 240.0, &mut expected)
            .unwrap();
        assert_eq!(
            output.particle_q.host().unwrap(),
            expected.particle_q.host().unwrap()
        );
        assert_eq!(
            output.particle_qd.host().unwrap(),
            expected.particle_qd.host().unwrap()
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-publication"]
fn unsupported_body_attached_shapes_are_not_silently_dropped() {
    let engine = Engine::new_gpu().unwrap();
    for dynamic in [false, true] {
        let mut b = ModelBuilder::new();
        b.add_particle(Vec3::Y, 1.0, 0.01, WorldId(0));
        let body = b.add_body(BodyParams::new().with_mass(if dynamic { 1.0 } else { 0.0 }));
        b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
        let m = b.build().unwrap();
        let input = m.state().unwrap();
        let mut output = m.state().unwrap();
        output
            .particle_q
            .host_mut()
            .unwrap()
            .fill(Vec3::splat(41.0));
        let error = BarrierSolver::new_gpu(&engine)
            .unwrap()
            .step(
                &m,
                &input,
                &m.control(),
                &Contacts::with_capacity(0),
                1.0 / 240.0,
                &mut output,
            )
            .unwrap_err();
        assert!(error.to_string().contains("body-attached"), "{error}");
        assert_eq!(output.particle_q.host().unwrap(), &[Vec3::splat(41.0)]);
    }
}
