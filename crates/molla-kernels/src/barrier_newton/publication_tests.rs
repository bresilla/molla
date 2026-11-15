use super::*;
use molla_compute::Engine;

#[test]
#[ignore = "requires GPU; run via make test-ppf-publication"]
fn failed_candidate_readback_preserves_committed_state_and_retry() {
    let engine = Engine::new_gpu().unwrap();
    let new = || {
        BarrierNewtonGpu::new(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            &[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            &[1.0; 2],
            &[BarrierGpuSpring {
                a: 0,
                b: 1,
                rest: 1.0,
                k: 10.0,
            }],
            &[],
            BarrierGpuParams::default(),
        )
        .unwrap()
    };
    for failure in [1, 2] {
        let mut sim = new();
        let mut reference = new();
        sim.step(1.0 / 240.0).unwrap();
        reference.step(1.0 / 240.0).unwrap();
        let before = (
            sim.read_positions().unwrap(),
            sim.read_velocities().unwrap(),
        );
        let mut calls = 0;
        let error = sim
            .step_readback(1.0 / 240.0, |solver, buffer| {
                calls += 1;
                if calls == failure {
                    Err(Error::Gpu("injected candidate readback failure".into()))
                } else {
                    solver.read_vec3(buffer)
                }
            })
            .unwrap_err();
        assert!(error.to_string().contains("injected"));
        assert_eq!(calls, failure);
        assert!(!sim.warmed);
        assert_eq!(sim.read_positions().unwrap(), before.0);
        assert_eq!(sim.read_velocities().unwrap(), before.1);
        let (_, positions, velocities) = sim.step_with_readback(1.0 / 240.0).unwrap();
        reference.warmed = false;
        reference.step(1.0 / 240.0).unwrap();
        assert_eq!(positions, sim.read_positions().unwrap());
        assert_eq!(velocities, sim.read_velocities().unwrap());
        assert_eq!(positions, reference.read_positions().unwrap());
        assert_eq!(velocities, reference.read_velocities().unwrap());
    }
}
