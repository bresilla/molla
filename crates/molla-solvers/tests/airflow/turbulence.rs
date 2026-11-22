use super::*;

#[test]
fn mixed_curl_wind_is_divergence_free_bounded_and_floor_tangent() {
    let mut c = config();
    c.turbulence_strength = 2.2;
    let h = 1e-5;
    let mut low = Vec3::splat(f64::INFINITY);
    let mut high = Vec3::splat(f64::NEG_INFINITY);
    for i in 0..240 {
        c.time = i as f64 * 0.11;
        let p = Vec3::new(0.17 * i as f64 - 10.0, 0.15 + (i % 7) as f64 * 0.03, -4.3);
        let v = c.turbulence(p);
        low = low.min(v);
        high = high.max(v);
        let div: f64 = [Vec3::X, Vec3::Y, Vec3::Z]
            .iter()
            .enumerate()
            .map(|(j, axis)| {
                (c.turbulence(p + *axis * h)[j] - c.turbulence(p - *axis * h)[j]) / (2.0 * h)
            })
            .sum();
        assert!(div.abs() < 1e-6, "div={div}");
        assert_eq!(c.turbulence(Vec3::new(p.x, 0.0, p.z)).y, 0.0);
        assert!(v.is_finite() && v.length() < 10.0);
    }
    eprintln!("curl near-floor component range: {low:?} .. {high:?}");
    assert!(low.y < -0.2 && high.y > 0.2 && low.x < -0.5 && high.x > 0.5);
    assert_eq!(config().turbulence(Vec3::ONE), Vec3::ZERO);
    for strength in [-1.0, f64::NAN, f64::INFINITY] {
        c.turbulence_strength = strength;
        assert!(AirState::new(c).is_err());
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn gpu_mixed_wind_matches_cpu_inside_outside_and_after_time_advance() {
    use molla_compute::Engine;
    use molla_solvers::airflow::GpuAir;
    let mut c = config();
    c.turbulence_strength = 2.2;
    c.gust_strength = 0.65;
    c.exterior_decay_length = 3.0;
    c.ambient = Vec3::new(1.2, 0.0, 0.35);
    let mut cpu = AirState::new(c).unwrap();
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = GpuAir::new(&engine, &cpu).unwrap();
    let points = [
        Vec3::new(0.1, 0.0, 0.2),
        Vec3::new(-0.9, 0.03, 0.2),
        Vec3::new(-0.91, 0.13, 0.2),
        Vec3::new(-2.0, 0.3, -2.0),
        Vec3::new(30.0, 0.2, 11.0),
        Vec3::new(0.2, 1.2, 0.1),
    ];
    for _ in 0..12 {
        for (p, v) in points.iter().zip(gpu.sample(&engine, &points).unwrap()) {
            assert!(
                cpu.velocity_at(*p).unwrap().distance(v) < 2e-4,
                "{p:?}: {v:?}"
            );
        }
        cpu.step(1.0 / 60.0, &AirForces::new(c.grid)).unwrap();
        gpu.step(&engine, 1.0 / 60.0, &AirForces::new(c.grid))
            .unwrap();
    }
}
