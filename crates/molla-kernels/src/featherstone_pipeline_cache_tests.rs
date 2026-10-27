use super::*;
use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder};
use std::time::Instant;

fn run(engine: &Engine, label: &str) -> Vec<Vec<u32>> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(
        BodyParams::new()
            .with_mass(2.0)
            .with_inertia(Mat3::IDENTITY),
    );
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    let mut input = State::for_model(&model).unwrap();
    input.joint_qd.host_mut().unwrap()[0] = 0.25;
    input.body_qd.host_mut().unwrap()[0] = SpatialVector::linear(Vec3::X * 0.25);
    let mut output = input.clone_host().unwrap();
    let mut control = Control::for_model(&model);
    model.upload_to_device(device, queue).unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let started = Instant::now();
    let mut kernel = FeatherstoneKernel::new_cached(engine, &model).unwrap();
    eprintln!(
        "compiler cache {label}: one-joint full-program construction {:.6}s",
        started.elapsed().as_secs_f64()
    );
    let reused = FeatherstoneKernel::new_cached(engine, &model).unwrap();
    assert_eq!(kernel.pipeline, reused.pipeline);
    assert_ne!(kernel.velocity_delta, reused.velocity_delta);
    let initial =
        rounding_tests::words(device, queue, input.body_q.device_buffer_current().unwrap());
    let mut result = Vec::new();
    for _ in 0..4 {
        kernel
            .dispatch(
                device,
                queue,
                &model,
                &input,
                &control,
                &Contacts::with_capacity(0),
                0.0,
                0.001,
                true,
                true,
                &mut output,
                None,
            )
            .unwrap();
        assert_eq!(
            rounding_tests::words(device, queue, kernel.status_buffer()),
            [0]
        );
        for buffer in [
            output.body_q.device_buffer_current().unwrap(),
            output.body_qd.device_buffer_current().unwrap(),
            output.joint_q.device_buffer_current().unwrap(),
            output.joint_qd.device_buffer_current().unwrap(),
        ] {
            let words = rounding_tests::words(device, queue, buffer);
            assert!(words.iter().all(|&word| f32::from_bits(word).is_finite()));
            result.push(words);
        }
        std::mem::swap(&mut input, &mut output);
    }
    assert_ne!(result[0], initial);
    result
}

#[test]
#[ignore = "requires real GPU with PIPELINE_CACHE; bounded one-joint physical comparison"]
fn compiler_cache_preserves_featherstone_across_device_recreation() {
    let mut engine = Engine::new_gpu_with_features(wgpu::Features::PIPELINE_CACHE).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    engine.enable_pipeline_cache(None).unwrap();
    let cold = run(&engine, "empty");
    let snapshot = engine.pipeline_cache_snapshot().unwrap().unwrap();
    eprintln!(
        "Featherstone compiler snapshot {} bytes",
        snapshot.as_bytes().len()
    );
    engine.destroy_device().unwrap();
    drop(engine);
    let mut engine = Engine::new_gpu_with_features(wgpu::Features::PIPELINE_CACHE).unwrap();
    engine.enable_pipeline_cache(Some(&snapshot)).unwrap();
    let warm = run(&engine, "restored");
    assert_eq!(warm, cold);
    engine.clear_gpu_program_cache();
    engine.disable_pipeline_cache();
    let uncached = run(&engine, "disabled");
    assert_eq!(uncached, cold);
    eprintln!(
        "all {} physical state words exact across empty/restored/disabled caches",
        cold.iter().map(Vec::len).sum::<usize>()
    );
}
