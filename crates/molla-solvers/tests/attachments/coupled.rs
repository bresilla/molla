#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::Vec3;
use molla_solvers::airflow::{AirConfig, AirForces, AirState, GpuAir, Grid};
use molla_solvers::attachments::{
    Anchor, Attachment, BladeFrame, Config, GpuAttachments, LeafBody, Particle, Petiole,
};

const DT: f64 = 1.0 / 60.0;

fn grid() -> Grid {
    Grid::new([24, 16, 20], 0.15, Vec3::new(-1.8, 0.0, -1.5)).unwrap()
}

fn air(engine: &Engine) -> GpuAir {
    let mut config = AirConfig::new(grid());
    config.ambient = Vec3::new(0.8, -1.5, 0.4);
    config.gust_strength = 0.6;
    config.turbulence_strength = 0.8;
    config.exterior_decay_length = 1.0;
    GpuAir::new(engine, &AirState::new(config).unwrap()).unwrap()
}

/// Attached hinged leaves, a free leaf and a leaf outside the depositable interior.
fn leaves() -> (Vec<Attachment>, Vec<Particle>, Vec<Anchor>, Vec<BladeFrame>) {
    let mass = 0.0001;
    let frame = BladeFrame {
        direction: Vec3::X,
        normal: Vec3::Y,
        angular_velocity: Vec3::ZERO,
    };
    let body = LeafBody {
        frame,
        length: 0.1,
        width: 0.06,
        area: mass * 10.0,
        camber: 0.004,
    };
    let a = Attachment {
        mass,
        stiffness: mass * (std::f64::consts::TAU * 7.0).powi(2),
        damping: 0.0005,
        strength: 1.0,
        drag: 0.0002,
        radius: 0.002,
        petiole: Some(Petiole {
            sheltering: 0.3,
            ..Petiole::from_frequency(mass, body, 0.008, 3.0, 4.0, 0.12)
        }),
    };
    let points = [
        Vec3::new(0.1, 1.0, 0.0),
        Vec3::new(-0.4, 1.3, 0.3),
        Vec3::new(0.5, 0.8, -0.2),
        Vec3::new(1.5, 2.3, 1.4),
    ];
    let mut particles = Vec::new();
    let mut anchors = Vec::new();
    for (i, &anchor) in points.iter().enumerate() {
        let mut p = Particle::attached(anchor + Vec3::X * 0.058);
        p.leaf = Some(body);
        if i == 2 {
            p.attached = false;
        }
        particles.push(p);
        anchors.push(Anchor {
            position: anchor,
            velocity: Vec3::ZERO,
            valid: i != 2,
        });
    }
    (vec![a; points.len()], particles, anchors, vec![frame; points.len()])
}

fn config() -> Config {
    Config {
        dt: DT,
        ..Default::default()
    }
}

#[test]
#[ignore = "requires Vulkan"]
fn coupled_leaves_sample_the_device_air_like_host_queries() {
    let engine = Engine::new_gpu().unwrap();
    let mut air = air(&engine);
    let (coefficients, particles, anchors, frames) = leaves();
    let mut host = GpuAttachments::new(&engine, coefficients.clone(), &particles).unwrap();
    let mut device = GpuAttachments::new(&engine, coefficients, &particles).unwrap();
    device.couple_air(&engine, &air.coupling()).unwrap();
    for _ in 0..30 {
        let positions: Vec<_> = host.snapshot(&engine).unwrap().iter().map(|p| p.position).collect();
        let wind = air.sample(&engine, &positions).unwrap();
        host.step_spatial_blades(&engine, &anchors, config(), &wind, &frames)
            .unwrap();
        device
            .step_coupled_blades(&engine, &anchors, config(), &frames, None)
            .unwrap();
        air.step(&engine, DT, &AirForces::new(grid())).unwrap();
    }
    let (h, d) = (host.snapshot(&engine).unwrap(), device.snapshot(&engine).unwrap());
    for (i, (h, d)) in h.iter().zip(&d).enumerate() {
        assert_eq!(h.attached, d.attached, "leaf {i}");
        assert!(h.position.distance(d.position) < 1e-4, "leaf {i}: {h:?} {d:?}");
        assert!(h.velocity.distance(d.velocity) < 1e-3, "leaf {i}");
        assert!((h.wind_speed - d.wind_speed).abs() < 1e-4 && d.wind_speed > 0.5, "leaf {i}");
    }
}

#[test]
#[ignore = "requires Vulkan"]
fn device_reactions_match_host_force_deposits() {
    let engine = Engine::new_gpu().unwrap();
    let mut host_air = air(&engine);
    let mut device_air = air(&engine);
    let grid = grid();
    let (coefficients, particles, anchors, frames) = leaves();
    let mut host = GpuAttachments::new(&engine, coefficients.clone(), &particles).unwrap();
    let mut device = GpuAttachments::new(&engine, coefficients, &particles).unwrap();
    host.couple_air(&engine, &host_air.coupling()).unwrap();
    device.couple_air(&engine, &device_air.coupling()).unwrap();
    let mut boundary = (Vec3::ZERO, Vec3::ZERO);
    for _ in 0..20 {
        let before = host.snapshot(&engine).unwrap();
        host.step_coupled_blades(&engine, &anchors, config(), &frames, None)
            .unwrap();
        device
            .step_coupled_blades(&engine, &anchors, config(), &frames, Some(1.0))
            .unwrap();
        let after = host.snapshot(&engine).unwrap();
        let mut forces = AirForces::new(grid);
        let inner = grid.origin() + Vec3::splat(grid.spacing());
        let outer = grid.upper() - Vec3::splat(grid.spacing());
        let margin = Vec3::splat(2.0 * grid.spacing());
        for (old, new) in before.iter().zip(&after) {
            if old.position.cmpge(inner).all() && old.position.cmple(outer).all() {
                forces.add(old.position, -new.air_force).unwrap();
            } else {
                boundary.0 -= new.air_force * DT;
            }
            let torque = -new.air_torque;
            if torque.length_squared() < 1e-24 {
                continue;
            }
            if new.position.cmpge(grid.origin() + margin).all()
                && new.position.cmple(grid.upper() - margin).all()
            {
                forces.add_torque(new.position, torque).unwrap();
            } else {
                boundary.1 += torque * DT;
            }
        }
        host_air.step(&engine, DT, &forces).unwrap();
        device_air.step(&engine, DT, &AirForces::new(grid)).unwrap();
    }
    let probes: Vec<_> = (0..64)
        .map(|i| {
            grid.origin()
                + Vec3::new(
                    0.3 + 0.05 * (i % 4) as f64,
                    0.6 + 0.12 * (i / 4 % 4) as f64,
                    0.4 + 0.15 * (i / 16) as f64,
                ) * 3.0
        })
        .collect();
    let (h, d) = (
        host_air.sample(&engine, &probes).unwrap(),
        device_air.sample(&engine, &probes).unwrap(),
    );
    let largest = h.iter().map(|v| v.length()).fold(0.0, f64::max);
    for (h, d) in h.iter().zip(&d) {
        assert!(h.distance(*d) < 1e-4 * largest.max(1.0), "{h:?} {d:?}");
    }
    let (linear, angular) = device_air.take_boundary_impulses(&engine).unwrap();
    assert!(linear.distance(boundary.0) < 1e-6 && boundary.0.length() > 1e-6, "{linear:?} {boundary:?}");
    assert!(angular.distance(boundary.1) < 1e-6, "{angular:?} {boundary:?}");
    assert_eq!(device_air.take_boundary_impulses(&engine).unwrap().0, Vec3::ZERO);
}
