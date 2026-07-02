//! TEMPORARY F1b probe.
use molla_autograd::mpm_cpu_diff::{MpmCpuConfig, ParticleState, forward};
use molla_math::Vec3;

#[test]
fn probe_fall() {
    let cfg = MpmCpuConfig::default();
    let mut state = ParticleState::new(&[Vec3::new(0.0, 0.5, 0.0)], &[Vec3::ZERO]);
    let dt = 1.0 / 240.0;
    for step in 0..30 {
        state = forward(&cfg, &state, dt, 1);
        let p = state.pos[0];
        let v = state.vel[0];
        eprintln!(
            "step {step}: p=({:.4},{:.4},{:.4}) v=({:.4},{:.4},{:.4}) C00={:.3e}",
            p.x, p.y, p.z, v.x, v.y, v.z, state.c[0].x_axis.x
        );
        if !p.y.is_finite() || v.length() > 100.0 {
            break;
        }
    }
}
