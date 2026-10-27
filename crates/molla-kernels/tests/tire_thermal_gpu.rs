//! GPU parity tests for batched layered tire thermal stepping.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireThermalGpuInput, TireThermalGpuKernel, TireThermalGpuState,
    tire_thermal_build_rig_replay_batch, tire_thermal_reference_rig_replay_batch,
    tire_thermal_reference_step,
};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn sample_inputs() -> (Vec<TireThermalGpuInput>, Vec<TireThermalGpuState>) {
    let inputs = vec![
        TireThermalGpuInput {
            powers: [4_800.0, 520.0, 210.0, 0.0],
            environment: [22.0, 0.40, 0.0, 0.0],
        },
        TireThermalGpuInput {
            powers: [900.0, 2_200.0, 480.0, 7_500.0],
            environment: [38.0, 1.25, 0.0, 0.0],
        },
        TireThermalGpuInput {
            powers: [0.0, 0.0, 0.0, 0.0],
            environment: [5.0, 2.0, 0.0, 0.0],
        },
    ];
    let mut hot = TireThermalGpuState::heavy_duty_slow_response();
    hot.temperatures = [92.0, 84.0, 71.0, 58.0];
    let mut cooling = TireThermalGpuState::fast_response_test();
    cooling.temperatures = [78.0, 65.0, 48.0, 40.0];
    (
        inputs,
        vec![TireThermalGpuState::passenger_road(), hot, cooling],
    )
}

#[test]
fn tire_thermal_reference_batches_layered_heat_flow() {
    let (inputs, states) = sample_inputs();
    let (next0, out0) = tire_thermal_reference_step(&inputs[0], &states[0]);
    assert!(
        next0.temperatures[0] > states[0].temperatures[0],
        "sliding power should heat tire surface: {next0:?}"
    );
    assert!(out0.diagnostics.iter().all(|v| v.is_finite()));
    assert!(out0.diagnostics[1] >= 0.45 && out0.diagnostics[1] <= 1.08);
    assert!(
        next0.wear[0] > states[0].wear[0],
        "sliding power should also advance state-derived tread wear: {next0:?}"
    );
    assert!(
        out0.wear[2] > 0.0 && out0.wear[2] < 1.0 && out0.wear[3] <= 1.0,
        "wear output should report depth ratio and worn-tread μ factor: {out0:?}"
    );

    let (next1, out1) = tire_thermal_reference_step(&inputs[1], &states[1]);
    assert!(
        next1.temperatures[3] > states[1].temperatures[3],
        "brake power should heat the rim node: {next1:?}"
    );
    assert!(
        out1.diagnostics[2] >= 0.85 && out1.diagnostics[2] <= 1.20,
        "pressure/stiffness factor must stay bounded: {out1:?}"
    );

    let (next2, _) = tire_thermal_reference_step(&inputs[2], &states[2]);
    assert!(
        next2.temperatures[0] < states[2].temperatures[0],
        "cold road/no power should cool the surface: {next2:?}"
    );
}

#[test]
fn tire_thermal_builds_gpu_ready_rig_replay_artifact() {
    let run = molla_vehicle::tire_rig_default_non_calibration_run();
    let replay = tire_thermal_build_rig_replay_batch(&run.snapshots, 1.0 / 120.0, 20.0);
    assert_eq!(replay.len(), run.snapshots.len());
    assert_eq!(replay.inputs.len(), replay.states.len());
    assert_eq!(replay.scenario_ids.len(), replay.inputs.len());
    assert!(replay.capacity_hint() >= replay.len());
    assert!(
        replay
            .scenario_ids
            .iter()
            .any(|id| id.contains("thermal") || id.contains("wear")),
        "the non-calibration rig should include thermal/wear replay samples: {:?}",
        replay.scenario_ids
    );
    assert!(
        replay
            .inputs
            .iter()
            .all(|input| input.powers.iter().all(|v| v.is_finite())
                && input.environment.iter().all(|v| v.is_finite())
                && input.environment[1] > 0.0),
        "replay inputs must be finite GPU artifacts: {:?}",
        replay.inputs
    );
    assert!(
        replay
            .states
            .iter()
            .all(|state| state.temperatures.iter().all(|v| v.is_finite())
                && state.wear[0].is_finite()
                && state.wear[1] > 0.0),
        "replay states must be finite GPU artifacts: {:?}",
        replay.states
    );
    let outputs = tire_thermal_reference_rig_replay_batch(&replay);
    assert_eq!(outputs.len(), replay.len());
    assert!(
        outputs
            .iter()
            .all(|(_, output)| output.diagnostics.iter().all(|v| v.is_finite())),
        "reference replay outputs must be finite diagnostics: {outputs:?}"
    );
}

#[test]
fn tire_thermal_shader_contains_four_node_batch_path() {
    let shader = include_str!("../shaders/tire_thermal.wgsl");
    assert!(shader.contains("@compute @workgroup_size(64)"));
    assert!(shader.contains("i >= uniforms.count"));
    assert!(shader.contains("q_surface_in = 0.70 * sliding_power_w"));
    assert!(shader.contains("q_carcass_in"));
    assert!(shader.contains("wear_temperature_factor"));
    assert!(shader.contains("step_wear"));
    assert!(shader.contains("wear_mu_factor"));
    assert!(!shader.contains("isFinite"));
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_thermal_persistent_batch_matches_one_shot_dispatch() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireThermalGpuKernel::new(&device);
    let batch = kernel
        .create_batch(&device, 8)
        .expect("persistent thermal batch allocation");
    assert_eq!(batch.capacity(), 8);
    let (inputs, states) = sample_inputs();
    let one_shot = kernel
        .dispatch_blocking(&device, &queue, &inputs, &states)
        .expect("one-shot thermal dispatch");
    let persistent = kernel
        .dispatch_persistent_blocking(&device, &queue, &batch, &inputs, &states)
        .expect("persistent thermal dispatch");
    assert_eq!(persistent.0.len(), one_shot.0.len());
    assert_eq!(persistent.1.len(), one_shot.1.len());
    for i in 0..one_shot.0.len() {
        for axis in 0..4 {
            assert!(
                (persistent.0[i].temperatures[axis] - one_shot.0[i].temperatures[axis]).abs()
                    < 1.0e-6,
                "persistent state mismatch tire={i} axis={axis}"
            );
            assert!(
                (persistent.1[i].diagnostics[axis] - one_shot.1[i].diagnostics[axis]).abs()
                    < 1.0e-6,
                "persistent output mismatch tire={i} axis={axis}"
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_thermal_gpu_matches_reference_batch() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireThermalGpuKernel::new(&device);
    let (inputs, states) = sample_inputs();
    let expected = inputs
        .iter()
        .zip(states.iter())
        .map(|(input, state)| tire_thermal_reference_step(input, state))
        .collect::<Vec<_>>();
    let (got_states, got_outputs) = kernel
        .dispatch_blocking(&device, &queue, &inputs, &states)
        .expect("GPU thermal dispatch");
    assert_eq!(got_states.len(), expected.len());
    assert_eq!(got_outputs.len(), expected.len());
    for i in 0..expected.len() {
        for axis in 0..4 {
            assert!(
                (got_states[i].temperatures[axis] - expected[i].0.temperatures[axis]).abs()
                    < 1.0e-4,
                "temperature mismatch tire={i} axis={axis}: got={:?} expected={:?}",
                got_states[i],
                expected[i].0
            );
            assert!(
                (got_outputs[i].diagnostics[axis] - expected[i].1.diagnostics[axis]).abs() < 1.0e-4,
                "diagnostic mismatch tire={i} axis={axis}: got={:?} expected={:?}",
                got_outputs[i],
                expected[i].1
            );
            assert!(
                (got_states[i].wear[axis] - expected[i].0.wear[axis]).abs() < 1.0e-7,
                "wear state mismatch tire={i} axis={axis}: got={:?} expected={:?}",
                got_states[i],
                expected[i].0
            );
        }
    }
}
