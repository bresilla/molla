//! GPU parity tests for batched layered tire transient relaxation.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireTransientGpuInput, TireTransientGpuKernel, TireTransientGpuState,
    tire_transient_reference_step,
};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn sample_inputs() -> (Vec<TireTransientGpuInput>, Vec<TireTransientGpuState>) {
    let inputs = vec![
        TireTransientGpuInput {
            command: [0.40, 0.12, 0.08, 0.32],
            tread_lengths: [0.16, 0.22, 0.28, 1.0],
            belt_lengths: [0.45, 0.65, 0.80, 0.0],
            carcass_lengths: [1.05, 1.35, 1.60, 0.0],
        },
        TireTransientGpuInput {
            command: [-0.25, -0.08, 0.21, 0.38],
            tread_lengths: [0.22, 0.28, 0.34, 1.45],
            belt_lengths: [0.55, 0.75, 0.95, 0.0],
            carcass_lengths: [1.25, 1.55, 1.80, 0.0],
        },
    ];
    let states = vec![
        TireTransientGpuState::default(),
        TireTransientGpuState {
            tread: [0.05, 0.02, 0.01, 0.0],
            belt: [0.02, 0.01, 0.0, 0.0],
            carcass: [0.01, 0.0, 0.0, 0.0],
        },
    ];
    (inputs, states)
}

#[test]
fn tire_transient_reference_batches_layered_relaxation() {
    let (inputs, states) = sample_inputs();
    let (next0, out0) = tire_transient_reference_step(&inputs[0], &states[0]);
    assert!(out0.effective.iter().all(|v| v.is_finite()));
    assert!(out0.effective[0] > 0.0 && out0.effective[0] < inputs[0].command[0]);
    assert!(next0.tread[0] > next0.belt[0] && next0.belt[0] > next0.carcass[0]);
    let (next1, out1) = tire_transient_reference_step(&inputs[1], &states[1]);
    assert!(out1.effective[0] < states[1].tread[0]);
    assert!(next1.tread[1] < states[1].tread[1]);
    assert!(
        out1.deflection[1].abs() <= inputs[1].command[3] * 4.0 + 1.0e-5,
        "lateral deflection should obey the same low-speed clamp as CPU transient state: {out1:?}"
    );
}

#[test]
fn tire_transient_shader_contains_layered_weights_and_gpu_batch_path() {
    let shader = include_str!("../shaders/tire_transient.wgsl");
    assert!(shader.contains("0.58 * tread.x + 0.30 * belt.x + 0.12 * carcass.x"));
    assert!(shader.contains("0.55 * tread.y + 0.32 * belt.y + 0.13 * carcass.y"));
    assert!(shader.contains("@compute @workgroup_size(64)"));
    assert!(shader.contains("i >= uniforms.count"));
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_transient_persistent_batch_matches_one_shot_dispatch() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireTransientGpuKernel::new(&device);
    let batch = kernel
        .create_batch(&device, 8)
        .expect("persistent transient batch allocation");
    let (inputs, states) = sample_inputs();
    let one_shot = kernel
        .dispatch_blocking(&device, &queue, &inputs, &states)
        .expect("one-shot transient dispatch");
    let persistent = kernel
        .dispatch_persistent_blocking(&device, &queue, &batch, &inputs, &states)
        .expect("persistent transient dispatch");
    assert_eq!(persistent.0.len(), one_shot.0.len());
    assert_eq!(persistent.1.len(), one_shot.1.len());
    for i in 0..one_shot.0.len() {
        for axis in 0..3 {
            assert!(
                (persistent.0[i].tread[axis] - one_shot.0[i].tread[axis]).abs() < 1.0e-6,
                "persistent tread mismatch wheel={i} axis={axis}"
            );
            assert!(
                (persistent.1[i].effective[axis] - one_shot.1[i].effective[axis]).abs() < 1.0e-6,
                "persistent output mismatch wheel={i} axis={axis}"
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_transient_gpu_matches_reference_batch() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireTransientGpuKernel::new(&device);
    let (inputs, states) = sample_inputs();
    let expected = inputs
        .iter()
        .zip(states.iter())
        .map(|(input, state)| tire_transient_reference_step(input, state))
        .collect::<Vec<_>>();
    let (got_states, got_outputs) = kernel
        .dispatch_blocking(&device, &queue, &inputs, &states)
        .expect("GPU transient dispatch");
    assert_eq!(got_states.len(), expected.len());
    assert_eq!(got_outputs.len(), expected.len());
    for i in 0..expected.len() {
        for axis in 0..3 {
            assert!(
                (got_states[i].tread[axis] - expected[i].0.tread[axis]).abs() < 1.0e-5,
                "tread state mismatch wheel={i} axis={axis}: got={:?} expected={:?}",
                got_states[i],
                expected[i].0
            );
            assert!(
                (got_outputs[i].effective[axis] - expected[i].1.effective[axis]).abs() < 1.0e-5,
                "effective output mismatch wheel={i} axis={axis}: got={:?} expected={:?}",
                got_outputs[i],
                expected[i].1
            );
        }
    }
}
