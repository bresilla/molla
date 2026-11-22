use molla_math::Vec3;
use molla_solvers::attachments::{Anchor, Attachment, Config, Particle, step_cpu};

#[path = "attachments/coupled.rs"]
mod coupled;

#[path = "attachments/petiole.rs"]
mod petiole;

#[path = "attachments/leaf_flight.rs"]
mod leaf_flight;

fn coefficient(strength: f64) -> Attachment {
    Attachment {
        mass: 0.0003,
        stiffness: 0.6,
        damping: 0.015,
        strength,
        drag: 0.00055,
        radius: 0.002,
        petiole: None,
    }
}
fn anchor() -> Anchor {
    Anchor {
        position: Vec3::Y,
        velocity: Vec3::ZERO,
        valid: true,
    }
}

#[test]
fn static_weight_survives_and_reacts_back() {
    let a = coefficient(0.01);
    let mut p = Particle::attached(Vec3::Y);
    for _ in 0..1200 {
        step_cpu(a, &mut p, anchor(), Config::default()).unwrap();
    }
    assert!(p.attached);
    assert!((p.force.y - a.mass * 9.81).abs() < 1e-7);
    assert!((p.position.y - (1.0 - a.mass * 9.81 / a.stiffness)).abs() < 1e-6);
}

#[test]
fn strength_not_visibility_selects_failure() {
    let mut weak = Particle::attached(Vec3::Y);
    let mut strong = weak;
    let a = Anchor {
        position: Vec3::Y + Vec3::X * 0.05,
        ..anchor()
    };
    step_cpu(coefficient(0.008), &mut weak, a, Config::default()).unwrap();
    step_cpu(coefficient(0.2), &mut strong, a, Config::default()).unwrap();
    assert!(!weak.attached && strong.attached);
    assert_eq!(weak.force, Vec3::ZERO);
    assert!(weak.peak_force > 0.008);
    for _ in 0..600 {
        step_cpu(coefficient(0.008), &mut weak, anchor(), Config::default()).unwrap();
    }
    assert!(!weak.attached && weak.position.y >= 0.002);
}

#[test]
fn invalid_input_is_atomic_and_missing_anchor_preserves_velocity() {
    let mut p = Particle::attached(Vec3::Y);
    p.velocity = Vec3::X;
    let before = p;
    assert!(
        step_cpu(
            coefficient(0.01),
            &mut p,
            anchor(),
            Config {
                dt: f64::NAN,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert_eq!(p, before);
    let mut a = coefficient(0.01);
    a.drag = 0.0;
    step_cpu(
        a,
        &mut p,
        Anchor {
            valid: false,
            ..anchor()
        },
        Config {
            gravity: Vec3::ZERO,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(p.velocity, Vec3::X);
    assert!(!p.attached);
}

#[test]
fn air_force_matches_particle_momentum_including_spring_failure() {
    for strength in [0.0001, 1.0] {
        let a = coefficient(strength);
        let mut p = Particle::attached(Vec3::Y);
        let c = Config {
            wind: Vec3::new(5.0, -2.0, 1.0),
            ..Default::default()
        };
        let before = p;
        step_cpu(a, &mut p, anchor(), c).unwrap();
        let residual = (p.velocity - before.velocity) * a.mass
            - (c.gravity * a.mass + p.force + p.air_force) * c.dt;
        assert!(residual.length() < 1e-15, "{residual:?}");
        assert_eq!(p.attached, strength == 1.0);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU"]
fn spatial_wind_matches_cpu_and_preserves_uniform_api() {
    use molla_compute::Engine;
    use molla_solvers::attachments::GpuAttachments;
    let engine = Engine::new_gpu().unwrap();
    let a = coefficient(0.02);
    let coefficients = vec![a; 3];
    let mut cpu = vec![Particle::attached(Vec3::Y); 3];
    let mut gpu = GpuAttachments::new(&engine, coefficients, &cpu).unwrap();
    let winds = [
        Vec3::new(9.0, -1.0, 0.0),
        Vec3::ZERO,
        Vec3::new(-1.0, 0.0, 2.0),
    ];
    let anchors = [anchor(); 3];
    for step in 0..180 {
        let local = if step < 150 { winds } else { [Vec3::ZERO; 3] };
        for (p, wind) in cpu.iter_mut().zip(local) {
            step_cpu(
                a,
                p,
                anchor(),
                Config {
                    wind,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        if step < 150 {
            gpu.step_spatial(&engine, &anchors, Config::default(), &local)
                .unwrap();
        } else {
            gpu.step(&engine, &anchors, Config::default()).unwrap();
        }
    }
    let actual = gpu.snapshot(&engine).unwrap();
    for (g, c) in actual.iter().zip(&cpu) {
        assert_eq!(g.attached, c.attached);
        assert!(g.position.distance(c.position) < 2e-5);
        assert!(g.velocity.distance(c.velocity) < 1e-4);
        assert!(g.air_force.distance(c.air_force) < 1e-6);
    }
    assert!(!actual[0].attached && actual[1].attached && actual[2].attached);
    assert!(
        gpu.step_spatial(&engine, &anchors, Config::default(), &winds[..2])
            .is_err()
    );
    assert!(
        gpu.step_spatial(
            &engine,
            &anchors,
            Config::default(),
            &[Vec3::splat(f64::NAN); 3]
        )
        .is_err()
    );
    assert_eq!(actual, gpu.snapshot(&engine).unwrap());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU"]
fn gpu_matches_cpu_static_shaken_detached_and_floor() {
    use molla_compute::Engine;
    use molla_solvers::attachments::GpuAttachments;
    let engine = Engine::new_gpu().unwrap();
    let coefficients = vec![coefficient(0.008), coefficient(0.2), coefficient(0.05)];
    let mut cpu = vec![Particle::attached(Vec3::Y); 3];
    let mut gpu = GpuAttachments::new(&engine, coefficients.clone(), &cpu).unwrap();
    for step in 0..720 {
        let t = step as f64 / 240.0;
        let a = if step < 120 {
            anchor()
        } else {
            Anchor {
                position: Vec3::Y + Vec3::X * (0.05 * (t * 20.0).sin()),
                velocity: Vec3::X * (t * 20.0).cos(),
                valid: true,
            }
        };
        let anchors = vec![
            a,
            a,
            Anchor {
                valid: step < 180,
                ..anchor()
            },
        ];
        let config = Config::default();
        for ((&a, p), &anchor) in coefficients.iter().zip(&mut cpu).zip(&anchors) {
            step_cpu(a, p, anchor, config).unwrap();
        }
        gpu.step(&engine, &anchors, config).unwrap();
        if step % 30 == 29 {
            let actual = gpu.snapshot(&engine).unwrap();
            for (a, b) in actual.iter().zip(&cpu) {
                assert_eq!(a.attached, b.attached);
                assert!(
                    a.position.distance(b.position) < 2e-5,
                    "position {a:?} {b:?}"
                );
                assert!(a.velocity.distance(b.velocity) < 1e-4);
                assert!(a.force.distance(b.force) < 1e-5);
            }
        }
    }
    assert!(!cpu[0].attached && cpu[1].attached && !cpu[2].attached);
    let before = gpu.snapshot(&engine).unwrap();
    assert!(gpu.step(&engine, &[anchor()], Config::default()).is_err());
    assert_eq!(before, gpu.snapshot(&engine).unwrap());
}
