#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_kernels::VbdClothKernel;
use molla_math::{Mat3, Vec3};
use molla_sim::{MaterialNeoHookean, Model, ModelBuilder, State, TetViscosity};

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn model(tets: usize) -> Model {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.set_linear_damping(0.0);
    for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z, Vec3::NEG_Z] {
        builder.add_particle(p * 0.02, 0.002, 0.0, WorldId(0));
    }
    if tets > 0 {
        let mut material = MaterialNeoHookean::elastic(2e6, 0.45);
        material.viscosity = TetViscosity {
            shear: 1379.3103448275863,
            bulk: 13333.333333333336,
        };
        builder
            .add_tet_mesh_per_material(
                &[[0, 1, 2, 3], [0, 2, 1, 4]][..tets],
                &[material],
                &vec![0; tets],
                None,
            )
            .unwrap();
    }
    builder.build().unwrap()
}

fn snapshot(engine: &Engine, state: &State) -> Vec<Vec<u32>> {
    [
        state.particle_q.device_buffer(),
        state.particle_qd.device_buffer(),
        state.tet_f_plastic.device_buffer(),
        state.tet_plastic_strain.device_buffer(),
        state.tet_back_stress.device_buffer(),
    ]
    .into_iter()
    .flatten()
    .map(|buffer| readback::read(engine, buffer, buffer.size() as usize / 4))
    .collect()
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gradient_cache_refreshes_history_geometry_and_topology() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut reused = VbdClothKernel::new(device, queue, 5, 0, &[]);
    for tets in [1, 2, 0, 2, 1] {
        let mut model = model(tets);
        let mut input = model.state().unwrap();
        let mut output = model.state().unwrap();
        let mut reference = model.state().unwrap();
        output.upload_to_device(device, queue).unwrap();
        reference.upload_to_device(device, queue).unwrap();
        for round in 0..3 {
            let h = (1.0 + round as f64) / 19200.0;
            input.particle_q.host_mut().unwrap()[2].y = 0.0201 + round as f64 * 0.0002;
            for v in input.particle_qd.host_mut().unwrap() {
                *v = Vec3::new(0.1, -0.3, 0.05);
            }
            for f in input.tet_f_plastic.host_mut().unwrap() {
                *f = Mat3::from_diagonal(Vec3::new(1.0, 1.0 + round as f64 * 0.001, 1.0));
            }
            model.initial_particle_q.host_mut().unwrap()[2].x += 0.00001;
            for inv in model.tet_dm_inv.host_mut().unwrap() {
                *inv *= 1.0001;
            }
            model.upload_to_device(device, queue).unwrap();
            input.upload_to_device(device, queue).unwrap();
            let mut fresh = VbdClothKernel::new(device, queue, 5, 0, &[]);
            for mode in 0..3 {
                if mode == 0 {
                    reused
                        .dispatch(device, queue, &model, &input, &mut output, h, 8, 0.0, false)
                        .unwrap();
                    fresh
                        .dispatch(
                            device,
                            queue,
                            &model,
                            &input,
                            &mut reference,
                            h,
                            8,
                            0.0,
                            false,
                        )
                        .unwrap();
                } else {
                    for begin in [true, false] {
                        reused
                            .material_iteration(
                                device,
                                queue,
                                &model,
                                &input,
                                &mut output,
                                h,
                                4,
                                begin,
                            )
                            .unwrap();
                        fresh
                            .material_iteration(
                                device,
                                queue,
                                &model,
                                &input,
                                &mut reference,
                                h,
                                4,
                                begin,
                            )
                            .unwrap();
                    }
                }
                assert_eq!(
                    snapshot(&engine, &output),
                    snapshot(&engine, &reference),
                    "tets={tets} round={round} mode={mode}"
                );
                input.particle_q.host_mut().unwrap()[2].y += 0.00001;
                input.upload_to_device(device, queue).unwrap();
                let mut force_reference = VbdClothKernel::new(device, queue, 5, 0, &[]);
                let before = snapshot(&engine, &output);
                let input_before = snapshot(&engine, &input);
                let actual = readback::read::<u32>(
                    &engine,
                    reused
                        .force_residual(device, queue, &model, &input, &mut output, h)
                        .unwrap(),
                    20,
                );
                let expected = readback::read::<u32>(
                    &engine,
                    force_reference
                        .force_residual(device, queue, &model, &input, &mut reference, h)
                        .unwrap(),
                    20,
                );
                assert_eq!(
                    actual, expected,
                    "force tets={tets} round={round} mode={mode}"
                );
                assert!(actual.iter().all(|v| f32::from_bits(*v).is_finite()));
                assert_eq!(before, snapshot(&engine, &output));
                assert_eq!(input_before, snapshot(&engine, &input));
            }
        }
    }
}
