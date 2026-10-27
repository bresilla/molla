use molla_core::WorldId;
use molla_math::{Mat3, Quat, Vec3};
use molla_sim::{Contacts, MaterialNeoHookean, Model, ModelBuilder, State, TetViscosity};
use molla_solvers::{Solver, VbdSolver};

#[cfg(feature = "gpu")]
#[allow(dead_code)]
#[path = "common/gpu_readback.rs"]
mod readback;

const REST: [Vec3; 4] = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
const VISC: TetViscosity = TetViscosity {
    shear: 30.0,
    bulk: 80.0,
};

fn model(viscosity: TetViscosity) -> Model {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.set_linear_damping(0.0);
    for point in REST {
        builder.add_particle(point, 1.0, 0.01, WorldId(0));
    }
    let material = MaterialNeoHookean {
        viscosity,
        ..MaterialNeoHookean::elastic(2000.0, 0.3)
    };
    builder
        .add_tet_mesh_per_material(&[[0, 1, 2, 3]], &[material], &[0], None)
        .unwrap();
    builder.build().unwrap()
}

#[derive(Clone, Debug)]
struct Frame {
    q: Vec<Vec3>,
    v: Vec<Vec3>,
}

fn host(state: &State) -> Frame {
    Frame {
        q: state.particle_q.host().unwrap().to_vec(),
        v: state.particle_qd.host().unwrap().to_vec(),
    }
}

fn energy(frame: &Frame) -> f64 {
    let f = Mat3::from_cols(
        frame.q[1] - frame.q[0],
        frame.q[2] - frame.q[0],
        frame.q[3] - frame.q[0],
    );
    let mu = 2000.0 / 2.6;
    let lambda = 2000.0 * 0.3 / (1.3 * 0.4) + mu;
    let alpha = 1.0 + mu / lambda;
    let squared = f.x_axis.length_squared() + f.y_axis.length_squared() + f.z_axis.length_squared();
    let elastic = (mu * (squared - 3.0)
        + lambda * ((f.determinant() - alpha).powi(2) - (1.0 - alpha).powi(2)))
        / 12.0;
    elastic
        + frame
            .v
            .iter()
            .map(|v| 0.5 * v.length_squared())
            .sum::<f64>()
}

fn initial(model: &Model, translation: Vec3) -> State {
    let mut state = model.state().unwrap();
    let stretch = Mat3::from_cols(
        Vec3::new(1.08, 0.0, 0.0),
        Vec3::new(0.04, 0.95, 0.0),
        Vec3::new(0.0, 0.02, 1.03),
    );
    for (q, p) in state.particle_q.host_mut().unwrap().iter_mut().zip(REST) {
        *q = stretch * p;
    }
    for v in state.particle_qd.host_mut().unwrap() {
        *v = translation;
    }
    state
}

fn cpu_run(
    viscosity: TetViscosity,
    dt: f64,
    steps: usize,
    iterations: usize,
    translation: Vec3,
) -> Vec<Frame> {
    let model = model(viscosity);
    let mut input = initial(&model, translation);
    let mut output = model.state().unwrap();
    let mut solver = VbdSolver::new_cpu().with_iterations(iterations);
    let mut frames = vec![host(&input)];
    for _ in 0..steps {
        solver
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                dt,
                &mut output,
            )
            .unwrap();
        std::mem::swap(&mut input, &mut output);
        frames.push(host(&input));
    }
    frames
}

fn check_energy(frames: &[Frame], tolerance: f64) {
    for pair in frames.windows(2) {
        let before = energy(&pair[0]);
        let after = energy(&pair[1]);
        assert!(
            before.is_finite() && after.is_finite() && after <= before + tolerance,
            "mechanical energy increased: {before} -> {after}"
        );
        let q = &pair[1].q;
        let j = Mat3::from_cols(q[1] - q[0], q[2] - q[0], q[3] - q[0]).determinant();
        assert!(j > 0.5, "tet inverted or collapsed: {j}");
    }
}

fn distance(a: &Frame, b: &Frame) -> f64 {
    a.q.iter()
        .zip(&b.q)
        .map(|(x, y)| (*x - *y).length_squared())
        .sum::<f64>()
        .sqrt()
}

#[test]
fn runtime_viscosity_dissipates_energy_and_refines() {
    let damped = cpu_run(VISC, 0.002, 100, 32, Vec3::ZERO);
    let elastic = cpu_run(TetViscosity::default(), 0.002, 100, 32, Vec3::ZERO);
    check_energy(&damped, 1e-10);
    assert!(energy(damped.last().unwrap()) < 0.5 * energy(elastic.last().unwrap()));
    let fine_iterations = cpu_run(VISC, 0.002, 100, 64, Vec3::ZERO);
    assert!(distance(damped.last().unwrap(), fine_iterations.last().unwrap()) < 1e-10);
    let half = cpu_run(VISC, 0.001, 200, 32, Vec3::ZERO);
    let quarter = cpu_run(VISC, 0.0005, 400, 32, Vec3::ZERO);
    let eighth = cpu_run(VISC, 0.00025, 800, 32, Vec3::ZERO);
    let coarse_error = distance(damped.last().unwrap(), eighth.last().unwrap());
    let half_error = distance(half.last().unwrap(), eighth.last().unwrap());
    let quarter_error = distance(quarter.last().unwrap(), eighth.last().unwrap());
    eprintln!(
        "viscosity energies damped={} elastic={}; refinement={coarse_error},{half_error},{quarter_error}",
        energy(damped.last().unwrap()),
        energy(elastic.last().unwrap())
    );
    assert!(half_error < 0.7 * coarse_error && quarter_error < 0.7 * half_error);
    check_energy(&half, 1e-10);
    check_energy(&quarter, 1e-10);
}

#[test]
fn runtime_viscosity_preserves_free_translation() {
    let base = cpu_run(VISC, 0.002, 50, 32, Vec3::ZERO);
    let boost = Vec3::new(0.2, -0.3, 0.1);
    let boosted = cpu_run(VISC, 0.002, 50, 32, boost);
    for (step, (base, moved)) in base.iter().zip(&boosted).enumerate() {
        for i in 0..4 {
            assert!((moved.q[i] - base.q[i] - boost * (step as f64 * 0.002)).length() < 1e-10);
            assert!((moved.v[i] - base.v[i] - boost).length() < 1e-10);
        }
        assert!(base.v.iter().copied().sum::<Vec3>().length() < 1e-10);
    }
}

#[test]
fn runtime_viscosity_leaves_finite_rigid_rotation_unchanged() {
    let dt = 0.002;
    let rotation = Quat::from_rotation_z(0.1);
    for viscosity in [TetViscosity::default(), VISC] {
        let model = model(viscosity);
        let mut input = model.state().unwrap();
        for (v, p) in input.particle_qd.host_mut().unwrap().iter_mut().zip(REST) {
            *v = (rotation * p - p) / dt;
        }
        let mut output = model.state().unwrap();
        VbdSolver::new_cpu()
            .with_iterations(32)
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                dt,
                &mut output,
            )
            .unwrap();
        for (q, p) in output.particle_q.host().unwrap().iter().zip(REST) {
            assert!((*q - rotation * p).length() < 1e-12);
        }
    }
}

#[test]
fn runtime_viscosity_rejects_invalid_input() {
    let mut model = model(VISC);
    let input = initial(&model, Vec3::ZERO);
    let mut output = model.state().unwrap();
    let original = host(&output);
    for dt in [0.0, -0.001, f64::NAN, f64::INFINITY] {
        assert!(
            VbdSolver::new_cpu()
                .step(
                    &model,
                    &input,
                    &model.control(),
                    &Contacts::with_capacity(0),
                    dt,
                    &mut output
                )
                .is_err()
        );
        assert_eq!(output.particle_q.host().unwrap(), original.q.as_slice());
    }
    model.tet_viscosity.host_mut().unwrap()[0].bulk = -1.0;
    assert!(
        VbdSolver::new_cpu()
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                0.002,
                &mut output
            )
            .is_err()
    );
    model.tet_viscosity.host_mut().unwrap()[0] = VISC;
    let mut invalid = input;
    invalid.particle_q.host_mut().unwrap()[0].x = f64::NAN;
    assert!(
        VbdSolver::new_cpu()
            .step(
                &model,
                &invalid,
                &model.control(),
                &Contacts::with_capacity(0),
                0.002,
                &mut output
            )
            .is_err()
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU"]
fn runtime_viscosity_gpu_preserves_rigid_motion_and_exposes_overflow() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let dt = 0.002;
    let rotation = Quat::from_rotation_z(0.1);
    let boost = Vec3::new(0.2, -0.3, 0.1);
    for viscosity in [
        TetViscosity::default(),
        VISC,
        TetViscosity {
            shear: f32::MAX as f64,
            bulk: f32::MAX as f64,
        },
    ] {
        let overflow = viscosity.shear > 1e30;
        let mut model = model(viscosity);
        let mut input = if overflow {
            initial(&model, Vec3::ZERO)
        } else {
            model.state().unwrap()
        };
        for (v, p) in input.particle_qd.host_mut().unwrap().iter_mut().zip(REST) {
            *v = (rotation * p - p) / dt + boost;
        }
        let mut output = model.state().unwrap();
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        output.upload_to_device(device, queue).unwrap();
        VbdSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_iterations(32)
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                dt,
                &mut output,
            )
            .unwrap();
        let actual =
            readback::read::<[f32; 4]>(&engine, output.particle_q.device_buffer().unwrap(), 4);
        if overflow {
            assert!(
                actual.iter().flatten().any(|v| !v.is_finite()),
                "overflow must reach FEM nonfinite health diagnostics"
            );
            use molla_kernels::tet_volume::{TetVolumeBindings, TetVolumeMonitor};
            use wgpu::util::DeviceExt;
            let status = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viscosity health"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let monitor = TetVolumeMonitor::new(device);
            let mut encoder = device.create_command_encoder(&Default::default());
            monitor
                .encode(
                    device,
                    &mut encoder,
                    TetVolumeBindings {
                        positions: output.particle_q.device_buffer().unwrap(),
                        reference: model.initial_particle_q.device_buffer().unwrap(),
                        indices: model.tet_indices.device_buffer().unwrap(),
                        inverse_rest: model.tet_dm_inv.device_buffer().unwrap(),
                        status: &status,
                        particle_count: 4,
                        tet_count: 1,
                    },
                )
                .unwrap();
            queue.submit([encoder.finish()]);
            assert_eq!(readback::read::<u32>(&engine, &status, 1), [4]);
            assert_eq!(
                readback::read::<u32>(&engine, monitor.diagnostics(), 2),
                [0, 0]
            );
        } else {
            for (q, p) in actual.iter().zip(REST) {
                let q = Vec3::new(q[0] as f64, q[1] as f64, q[2] as f64);
                assert!((q - rotation * p - boost * dt).length() < 2e-7);
            }
        }
    }
    let mut invalid = model(VISC);
    invalid.tet_viscosity.host_mut().unwrap()[0].bulk = f64::NAN;
    assert!(invalid.upload_to_device(device, queue).is_err());
    assert!(VbdSolver::new_gpu(&engine, &invalid).is_err());
}

#[cfg(feature = "gpu")]
fn gpu_run(
    engine: &molla_compute::Engine,
    viscosity: TetViscosity,
    dt: f64,
    steps: usize,
    iterations: usize,
    boost: Vec3,
) -> Vec<Frame> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = model(viscosity);
    let mut input = initial(&model, boost);
    let mut output = model.state().unwrap();
    let mut frames = vec![host(&input)];
    model.upload_to_device(device, queue).unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let mut solver = VbdSolver::new_gpu(engine, &model)
        .unwrap()
        .with_iterations(iterations);
    for _ in 0..steps {
        solver
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                dt,
                &mut output,
            )
            .unwrap();
        std::mem::swap(&mut input, &mut output);
        let unpack = |data: Vec<[f32; 4]>| {
            data.into_iter()
                .map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64))
                .collect::<Vec<_>>()
        };
        frames.push(Frame {
            q: unpack(readback::read(
                engine,
                input.particle_q.device_buffer().unwrap(),
                4,
            )),
            v: unpack(readback::read(
                engine,
                input.particle_qd.device_buffer().unwrap(),
                4,
            )),
        });
    }
    frames
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU"]
fn runtime_viscosity_gpu_refines_and_preserves_free_translation() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let coarse = gpu_run(&engine, VISC, 0.002, 100, 32, Vec3::ZERO);
    let half = gpu_run(&engine, VISC, 0.001, 200, 32, Vec3::ZERO);
    let quarter = gpu_run(&engine, VISC, 0.0005, 400, 32, Vec3::ZERO);
    let eighth = gpu_run(&engine, VISC, 0.00025, 800, 32, Vec3::ZERO);
    let iterations = gpu_run(&engine, VISC, 0.002, 100, 64, Vec3::ZERO);
    for frames in [&coarse, &half, &quarter, &eighth, &iterations] {
        check_energy(frames, 1e-5);
    }
    assert!(distance(coarse.last().unwrap(), iterations.last().unwrap()) < 2e-6);
    let errors =
        [&coarse, &half, &quarter].map(|f| distance(f.last().unwrap(), eighth.last().unwrap()));
    eprintln!("GPU viscosity refinement: {errors:?}");
    assert!(errors[1] < 0.7 * errors[0] && errors[2] < 0.7 * errors[1]);
    let reference = cpu_run(VISC, 0.00025, 800, 32, Vec3::ZERO);
    assert!(distance(eighth.last().unwrap(), reference.last().unwrap()) < 2e-5);
    let boost = Vec3::new(0.2, -0.3, 0.1);
    let moved = gpu_run(&engine, VISC, 0.002, 100, 32, boost);
    for (step, (base, moved)) in coarse.iter().zip(&moved).enumerate() {
        for i in 0..4 {
            assert!((moved.q[i] - base.q[i] - boost * (step as f64 * 0.002)).length() < 2e-5);
            assert!((moved.v[i] - base.v[i] - boost).length() < 5e-4);
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU"]
fn runtime_viscosity_gpu_matches_cpu_timesteps_and_dissipates() {
    use molla_compute::Engine;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for viscosity in [
        TetViscosity::default(),
        VISC,
        TetViscosity {
            shear: 30.0,
            bulk: 0.0,
        },
        TetViscosity {
            shear: 0.0,
            bulk: 80.0,
        },
    ] {
        let reference = cpu_run(viscosity, 0.002, 30, 32, Vec3::ZERO);
        let mut model = model(viscosity);
        let mut input = initial(&model, Vec3::ZERO);
        let mut output = model.state().unwrap();
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        output.upload_to_device(device, queue).unwrap();
        let mut solver = VbdSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_iterations(32);
        let mut frames = vec![reference[0].clone()];
        for expected in &reference[1..] {
            solver
                .step(
                    &model,
                    &input,
                    &model.control(),
                    &Contacts::with_capacity(0),
                    0.002,
                    &mut output,
                )
                .unwrap();
            std::mem::swap(&mut input, &mut output);
            let unpack = |data: Vec<[f32; 4]>| {
                data.into_iter()
                    .map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64))
                    .collect::<Vec<_>>()
            };
            let frame = Frame {
                q: unpack(readback::read(
                    &engine,
                    input.particle_q.device_buffer().unwrap(),
                    4,
                )),
                v: unpack(readback::read(
                    &engine,
                    input.particle_qd.device_buffer().unwrap(),
                    4,
                )),
            };
            assert!(
                distance(&frame, expected) < 2e-5,
                "GPU position mismatch: {viscosity:?} step{} error{} actual{frame:?} expected{expected:?}",
                frames.len(),
                distance(&frame, expected)
            );
            for (a, b) in frame.v.iter().zip(&expected.v) {
                assert!(
                    (*a - *b).length() < 5e-4,
                    "GPU velocity mismatch: {a:?} != {b:?}"
                );
            }
            assert!(frame.v.iter().copied().sum::<Vec3>().length() < 5e-4);
            frames.push(frame);
        }
        check_energy(&frames, 1e-5);
    }
}
