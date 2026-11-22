use super::*;

fn wake() -> AirState {
    let mut c = config();
    c.ambient = Vec3::new(1.2, 0.0, 0.35);
    c.exterior_decay_length = 3.0;
    let mut air = AirState::new(c).unwrap();
    air.seed_velocity(|_| Vec3::new(-3.0, 0.0, -2.0)).unwrap();
    air
}

#[test]
fn wake_crosses_both_grid_sides_and_decays_smoothly_to_background() {
    let air = wake();
    for edge in [Vec3::new(-0.9, 0.0, 0.0), Vec3::new(0.0, 0.0, -0.9)] {
        let outward = edge.normalize();
        let boundary = air.velocity_at(edge).unwrap();
        let near = air.velocity_at(edge + outward * 1e-5).unwrap();
        let beyond = air.velocity_at(edge + outward * 0.3).unwrap();
        assert!(near.distance(boundary) < 1e-8);
        assert!(beyond.distance(boundary) < 0.06);
        assert_eq!(beyond.y, 0.0);
        let far = edge + outward * 30.0;
        assert!(
            air.velocity_at(far)
                .unwrap()
                .distance(air.config().background(far))
                < 1e-12
        );
        let mut largest_step = 0.0_f64;
        let mut previous = boundary;
        for i in 1..=600 {
            let v = air.velocity_at(edge + outward * (i as f64 * 0.01)).unwrap();
            largest_step = largest_step.max(v.distance(previous));
            previous = v;
        }
        assert!(largest_step < 0.015, "{largest_step}");
    }
}

#[test]
fn exterior_policy_does_not_change_grid_evolution_or_interior_queries() {
    let mut extended = wake();
    let mut c = extended.config();
    c.exterior_decay_length = 0.0;
    let mut original = AirState::new(c).unwrap();
    original
        .seed_velocity(|_| Vec3::new(-3.0, 0.0, -2.0))
        .unwrap();
    for _ in 0..3 {
        let loads = AirForces::new(c.grid);
        extended.step(1.0 / 60.0, &loads).unwrap();
        original.step(1.0 / 60.0, &loads).unwrap();
        assert_eq!(extended.face_velocities(), original.face_velocities());
        assert_eq!(extended.pressures(), original.pressures());
        for point in [Vec3::new(0.2, 0.8, 0.1), Vec3::ZERO] {
            assert_eq!(
                extended.velocity_at(point).unwrap(),
                original.velocity_at(point).unwrap()
            );
        }
    }
    for length in [-1.0, f64::NAN, f64::INFINITY] {
        c.exterior_decay_length = length;
        assert!(AirState::new(c).is_err());
    }
}

#[test]
fn grounded_leaf_crosses_grid_edge_then_settles_when_air_stops() {
    use molla_solvers::attachments::{
        Anchor, Attachment, BladeFrame, Config, LeafBody, Particle, step_cpu,
    };
    let air = wake();
    let a = Attachment {
        mass: 0.0003,
        stiffness: 0.6,
        damping: 0.015,
        strength: 0.02,
        drag: 0.00066,
        radius: 0.002,
        petiole: None,
    };
    let mut p = Particle::attached(Vec3::new(-0.7, 0.002, 0.0));
    p.attached = false;
    p.leaf = Some(LeafBody {
        frame: BladeFrame {
            direction: Vec3::Z,
            normal: Vec3::Y,
            angular_velocity: Vec3::ZERO,
        },
        length: 0.1,
        width: 0.06,
        area: a.mass * 9.0,
        camber: 0.0,
    });
    let anchor = Anchor {
        position: Vec3::ZERO,
        velocity: Vec3::ZERO,
        valid: false,
    };
    for _ in 0..360 {
        let c = Config {
            dt: 1.0 / 120.0,
            wind: air.velocity_at(p.position).unwrap(),
            ..Default::default()
        };
        step_cpu(a, &mut p, anchor, c).unwrap();
        assert!(p.position.is_finite() && p.velocity.is_finite() && p.position.y >= 0.0);
    }
    eprintln!(
        "exterior ground leaf: {:?}, velocity {:?}",
        p.position, p.velocity
    );
    assert!(p.position.x < -1.8 && p.position.z < -0.9);
    for _ in 0..360 {
        step_cpu(
            a,
            &mut p,
            anchor,
            Config {
                dt: 1.0 / 120.0,
                ..Default::default()
            },
        )
        .unwrap();
    }
    assert!(p.velocity.length() < 0.03);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn gpu_exterior_wake_matches_cpu_at_sides_corners_ground_and_far_field() {
    use molla_compute::Engine;
    use molla_solvers::airflow::GpuAir;
    let engine = Engine::new_gpu().unwrap();
    let mut air = wake();
    let mut gpu = GpuAir::new(&engine, &air).unwrap();
    let points = [
        Vec3::new(-0.9, 0.0, 0.0),
        Vec3::new(-0.91, 0.0, 0.0),
        Vec3::new(-1.1, 0.002, 0.0),
        Vec3::new(-2.5, 0.1, -2.5),
        Vec3::new(1.1, 1.0, 0.0),
        Vec3::new(0.0, 1.7, 0.0),
        Vec3::new(0.0, 0.3, -2.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(35.0, 3.0, -8.0),
    ];
    for _ in 0..4 {
        for (point, actual) in points.iter().zip(gpu.sample(&engine, &points).unwrap()) {
            let expected = air.velocity_at(*point).unwrap();
            assert!(
                expected.distance(actual) < 2e-5,
                "{point:?}: {expected:?} vs {actual:?}"
            );
        }
        let forces = AirForces::new(air.config().grid);
        air.step(1.0 / 60.0, &forces).unwrap();
        gpu.step(&engine, 1.0 / 60.0, &forces).unwrap();
    }
}
