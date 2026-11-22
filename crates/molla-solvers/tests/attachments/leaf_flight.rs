use molla_math::{Quat, Vec3};
use molla_solvers::attachments::{
    Anchor, Attachment, BladeFrame, Config, LeafBody, Particle, step_cpu,
};

fn case(angle: f64) -> (Attachment, Particle, Anchor) {
    let a = Attachment {
        mass: 0.0003,
        stiffness: 0.6,
        damping: 0.015,
        strength: 0.02,
        drag: 0.00066,
        radius: 0.002,
        petiole: None,
    };
    let q = Quat::from_rotation_z(angle);
    let mut p = Particle::attached(Vec3::Y * 8.0);
    p.attached = false;
    p.leaf = Some(LeafBody {
        frame: BladeFrame {
            direction: q * Vec3::X,
            normal: q * Vec3::Y,
            angular_velocity: Vec3::ZERO,
        },
        length: 0.1,
        width: 0.06,
        area: a.mass * 9.0,
        camber: 0.0,
    });
    let anchor = Anchor {
        position: p.position,
        velocity: Vec3::ZERO,
        valid: false,
    };
    (a, p, anchor)
}

#[test]
fn still_air_produces_repeated_flutter_and_tumbling_without_noise() {
    for inertia in [1.1, 1.4] {
        let (a, mut p, anchor) = case(0.6);
        let leaf = p.leaf.as_mut().unwrap();
        leaf.area = 4.0 * a.mass / (std::f64::consts::PI * 1.225 * leaf.length * inertia);
        let c = Config {
            dt: 1.0 / 240.0,
            ground: -100.0,
            ..Default::default()
        };
        let mut reversals = 0;
        let mut sign = 0;
        let mut upward = 0;
        let mut angle = 0.6;
        let mut total_angle = 0.6;
        let mut lo = 0.6f64;
        let mut hi = 0.6f64;
        let mut path = 0.0;
        for _ in 0..1920 {
            step_cpu(a, &mut p, anchor, c).unwrap();
            let s = if p.velocity.x > 0.1 {
                1
            } else if p.velocity.x < -0.1 {
                -1
            } else {
                0
            };
            if s != 0 {
                if sign != 0 && s != sign {
                    reversals += 1;
                }
                sign = s;
            }
            if p.velocity.y > 0.05 {
                upward += 1;
            }
            let d = p.leaf.unwrap().frame.direction;
            let next = d.y.atan2(d.x);
            let delta = (next - angle).sin().atan2((next - angle).cos());
            total_angle += delta;
            angle = next;
            lo = lo.min(total_angle);
            hi = hi.max(total_angle);
            path += p.velocity.x.abs() * c.dt;
        }
        eprintln!(
            "APW I*={inertia}: reversals={reversals}, upward={upward}, angle_span={}, lateral_path={path}, position={:?}",
            hi - lo,
            p.position
        );
        assert!(path > 1.0 && upward > 10);
        if inertia < 1.2 {
            assert!(reversals >= 4 && hi - lo < std::f64::consts::TAU);
        } else {
            assert!(hi - lo > std::f64::consts::TAU * 2.0);
        }
    }
}

#[test]
fn planar_limit_matches_independent_apw_dimensionless_equations() {
    use std::f64::consts::PI;
    let (a, mut p, anchor) = case(0.6);
    let leaf = p.leaf.unwrap();
    let scale = leaf.length * 0.5;
    let u = (2.0 * a.mass * 9.81 / (PI * 1.225 * leaf.area)).sqrt();
    let inertia = 4.0 * a.mass / (PI * 1.225 * leaf.area * leaf.length);
    let rate = |q: [f64; 4]| {
        let [x, y, w, theta] = q;
        let speed = x.hypot(y).max(1e-12);
        let circulation = 2.0 / PI * (-1.2 * x * y / speed + PI * w);
        let drag = (0.4 + 2.0 * y * y / (speed * speed)) * speed / PI;
        [
            ((inertia + 1.0) * w * y - circulation * y - theta.sin() - drag * x) / inertia,
            (-inertia * w * x + circulation * x - theta.cos() - drag * y) / (inertia + 1.0),
            (-x * y - 0.2 * (1.0 + w.abs()) * w) / (0.25 * inertia + 0.125),
            w,
        ]
    };
    let c = Config {
        dt: 1.0 / 960.0,
        ground: -100.0,
        ..Default::default()
    };
    let h = c.dt * u / scale;
    let mut q = [0.0, 0.0, 0.0, 0.6];
    let add = |q: [f64; 4], k: [f64; 4], h: f64| std::array::from_fn(|i| q[i] + h * k[i]);
    for _ in 0..1920 {
        let k1 = rate(q);
        let k2 = rate(add(q, k1, h * 0.5));
        let k3 = rate(add(q, k2, h * 0.5));
        let k4 = rate(add(q, k3, h));
        q = std::array::from_fn(|i| q[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
        step_cpu(a, &mut p, anchor, c).unwrap();
        let expected = Vec3::new(
            q[0] * q[3].cos() - q[1] * q[3].sin(),
            q[0] * q[3].sin() + q[1] * q[3].cos(),
            0.0,
        ) * u;
        assert!(
            p.velocity.distance(expected) < 0.0001,
            "{:?} vs {expected:?}",
            p.velocity
        );
        assert!((p.leaf.unwrap().frame.angular_velocity.z - q[2] * u / scale).abs() < 0.002);
    }
}

#[test]
fn leaf_orientation_changes_trajectory_and_wind_drives_rotation() {
    let (a, mut tilted, anchor) = case(0.75);
    let (_, mut flat, _) = case(0.0);
    let mut turn = 0.0f64;
    for _ in 0..120 {
        for p in [&mut flat, &mut tilted] {
            step_cpu(
                a,
                p,
                anchor,
                Config {
                    dt: 1.0 / 120.0,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        turn = turn.max(tilted.leaf.unwrap().frame.angular_velocity.length());
    }
    eprintln!(
        "leaf fall flat={:?}, tilted={:?}, spin={turn}",
        flat.position, tilted.position
    );
    assert!(flat.position.distance(tilted.position) > 0.04);
    assert!(tilted.position.x.abs() > 0.03);
    assert!(turn > 0.5);
    assert!(tilted.velocity.y > -6.0);
}

#[test]
fn free_leaf_air_load_matches_linear_and_angular_momentum() {
    let (a, mut p, anchor) = case(0.6);
    p.velocity = Vec3::new(0.2, -1.0, 0.3);
    p.leaf.as_mut().unwrap().frame.angular_velocity = Vec3::new(0.3, -0.2, 0.1);
    let old = p;
    let c = Config {
        dt: 1.0 / 480.0,
        wind: Vec3::new(2.0, 0.1, -0.5),
        ..Default::default()
    };
    step_cpu(a, &mut p, anchor, c).unwrap();
    assert!(
        ((p.velocity - old.velocity) * a.mass - (c.gravity * a.mass + p.air_force) * c.dt).length()
            < 1e-15
    );
    assert!(
        (p.leaf.unwrap().angular_momentum(a.mass)
            - old.leaf.unwrap().angular_momentum(a.mass)
            - p.air_torque * c.dt)
            .length()
            < 1e-15
    );
}

#[test]
fn leaf_and_added_air_dissipate_energy_without_gravity() {
    let (a, mut p, anchor) = case(0.5);
    p.velocity = Vec3::new(2.0, -1.0, 0.4);
    p.leaf.as_mut().unwrap().frame.angular_velocity = Vec3::new(2.0, 0.5, -1.0);
    let wind = Vec3::new(0.5, 0.0, -0.3);
    let energy = |p: Particle| p.leaf.unwrap().kinetic_energy(a.mass, p.velocity - wind);
    let initial = energy(p);
    let mut last = initial;
    for _ in 0..600 {
        step_cpu(
            a,
            &mut p,
            anchor,
            Config {
                dt: 1.0 / 480.0,
                gravity: Vec3::ZERO,
                wind,
                ..Default::default()
            },
        )
        .unwrap();
        let e = energy(p);
        assert!(e <= last + 1e-8, "{last} -> {e}");
        last = e;
    }
    assert!(last < initial * 0.4);
}

#[test]
fn leaf_crosses_old_grid_boundary_and_keeps_traveling() {
    let (a, mut p, anchor) = case(0.7);
    p.position.x = 4.7;
    let c = Config {
        dt: 1.0 / 120.0,
        wind: Vec3::new(3.0, 0.0, 0.5),
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    for _ in 0..2400 {
        step_cpu(a, &mut p, anchor, c).unwrap();
    }
    eprintln!(
        "unbounded leaf final {:?}, velocity {:?}",
        p.position, p.velocity
    );
    assert!(p.position.x > 30.0 && p.velocity.x > 2.0);
    assert!(!p.attached);
}

#[test]
fn ground_contact_has_friction_but_does_not_freeze_windblown_leaves() {
    let (a, mut p, anchor) = case(0.2);
    p.position = Vec3::Y * 0.05;
    let c = Config {
        dt: 1.0 / 120.0,
        wind: Vec3::new(5.0, 0.0, 1.0),
        ..Default::default()
    };
    for _ in 0..1200 {
        step_cpu(a, &mut p, anchor, c).unwrap();
        assert!(p.position.y >= 0.0 && p.velocity.is_finite());
    }
    eprintln!("ground transport {:?}", p.position);
    assert!(p.position.x > 5.0);
}

#[test]
fn curved_leaf_rocks_and_tumbles_in_mixed_ground_wind() {
    use molla_solvers::airflow::{AirConfig, Grid};
    let (a, mut p, anchor) = case(0.0);
    p.leaf.as_mut().unwrap().camber = 0.006;
    p.position = Vec3::new(0.0, 0.005, 0.0);
    let mut air = AirConfig::new(Grid::new([4, 4, 4], 1.0, Vec3::ZERO).unwrap());
    air.ambient = Vec3::new(1.2, 0.0, 0.35);
    air.turbulence_strength = 2.2;
    let mut spin = 0.0_f64;
    let mut arc = 0.0;
    let mut rising = 0;
    let mut height = 0.0_f64;
    for i in 0..1920 {
        air.time = i as f64 / 120.0;
        let old = p.leaf.unwrap().frame.normal;
        let wind = air.background(p.position) + air.turbulence(p.position);
        step_cpu(
            a,
            &mut p,
            anchor,
            Config {
                dt: 1.0 / 120.0,
                wind,
                ..Default::default()
            },
        )
        .unwrap();
        let leaf = p.leaf.unwrap();
        arc += old.dot(leaf.frame.normal).clamp(-1.0, 1.0).acos();
        spin = spin.max(leaf.frame.angular_velocity.length());
        let low = leaf
            .contact_offsets()
            .iter()
            .map(|r| p.position.y + r.y)
            .fold(f64::INFINITY, f64::min);
        height = height.max(low);
        if low > 0.02 && p.velocity.y > 0.1 {
            rising += 1;
        }
        assert!(low >= a.radius - 0.001, "ground penetration {low}");
    }
    eprintln!(
        "curved leaf: arc={arc}, spin={spin}, rising={rising}, clearance={height}, position={:?}",
        p.position
    );
    assert!(arc > 2.0 && spin > 1.0 && p.position.length() > 1.0);
    assert!(rising > 20 && height > 0.05);
}

#[test]
fn curved_leaf_offset_drag_is_dissipative_in_still_air() {
    let (a, mut p, anchor) = case(0.4);
    p.leaf.as_mut().unwrap().camber = 0.006;
    p.leaf.as_mut().unwrap().frame.angular_velocity = Vec3::new(2.0, 1.0, -3.0);
    p.velocity = Vec3::new(1.5, -1.0, 0.7);
    let mut previous = p.leaf.unwrap().kinetic_energy(a.mass, p.velocity);
    for _ in 0..960 {
        step_cpu(
            a,
            &mut p,
            anchor,
            Config {
                dt: 1.0 / 480.0,
                gravity: Vec3::ZERO,
                ground: -100.0,
                ..Default::default()
            },
        )
        .unwrap();
        let energy = p.leaf.unwrap().kinetic_energy(a.mass, p.velocity);
        assert!(energy <= previous + 1e-10, "{previous} -> {energy}");
        previous = energy;
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn gpu_curved_leaf_contact_and_camber_match_cpu() {
    use molla_compute::Engine;
    use molla_solvers::attachments::GpuAttachments;
    let engine = Engine::new_gpu().unwrap();
    let (a, mut p, anchor) = case(0.2);
    p.leaf.as_mut().unwrap().camber = 0.006;
    p.position.y = 0.025;
    let mut gpu = GpuAttachments::new(&engine, vec![a], &[p]).unwrap();
    let c = Config {
        dt: 1.0 / 240.0,
        wind: Vec3::new(2.0, 0.2, 0.5),
        ..Default::default()
    };
    for _ in 0..60 {
        step_cpu(a, &mut p, anchor, c).unwrap();
        gpu.step(&engine, &[anchor], c).unwrap();
        let q = gpu.snapshot(&engine).unwrap()[0];
        assert!((q.leaf.unwrap().camber - 0.006).abs() < 1e-8);
        assert!(
            p.position.distance(q.position) < 0.002,
            "{:?} vs {:?}",
            p.position,
            q.position
        );
        assert!(p.velocity.distance(q.velocity) < 0.02);
    }
}

#[test]
fn invalid_leaf_geometry_does_not_mutate_state() {
    let (a, mut p, anchor) = case(0.2);
    p.leaf.as_mut().unwrap().area = -1.0;
    let old = p;
    assert!(step_cpu(a, &mut p, anchor, Config::default()).is_err());
    assert_eq!(p, old);
}

#[test]
fn three_dimensional_flight_is_rotation_invariant_and_resolves_roll() {
    let (a, mut p, anchor) = case(0.6);
    p.leaf.as_mut().unwrap().frame.angular_velocity = Vec3::new(2.0, 0.3, -1.0);
    let q = Quat::from_rotation_y(1.3);
    let mut rotated = p;
    let frame = &mut rotated.leaf.as_mut().unwrap().frame;
    frame.direction = q * frame.direction;
    frame.normal = q * frame.normal;
    frame.angular_velocity = q * frame.angular_velocity;
    let wind = Vec3::new(1.5, 0.2, 0.6);
    let mut roll = 0.0;
    for _ in 0..480 {
        step_cpu(
            a,
            &mut p,
            anchor,
            Config {
                dt: 1.0 / 240.0,
                wind,
                ..Default::default()
            },
        )
        .unwrap();
        step_cpu(
            a,
            &mut rotated,
            anchor,
            Config {
                dt: 1.0 / 240.0,
                wind: q * wind,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rotated.position.distance(q * p.position) < 1e-10);
        assert!(
            rotated
                .leaf
                .unwrap()
                .frame
                .normal
                .distance(q * p.leaf.unwrap().frame.normal)
                < 1e-9
        );
        let f = p.leaf.unwrap().frame;
        roll += f.angular_velocity.dot(f.direction).abs() / 240.0;
    }
    assert!(roll > 1.0 && p.position.z.abs() > 0.1);
}

#[test]
fn gust_and_downwash_steps_converge_without_velocity_clamps() {
    for wind in [
        Vec3::ZERO,
        Vec3::new(3.0, 0.8, 1.0),
        Vec3::new(1.0, -12.0, 2.0),
    ] {
        let (a, p, anchor) = case(0.6);
        let integrate = |hz: usize| {
            let mut p = p;
            for _ in 0..hz {
                step_cpu(
                    a,
                    &mut p,
                    anchor,
                    Config {
                        dt: 1.0 / hz as f64,
                        wind,
                        ground: -100.0,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(
                    p.velocity.is_finite() && p.leaf.unwrap().frame.angular_velocity.is_finite()
                );
            }
            p
        };
        let coarse = integrate(60);
        let fine = integrate(1920);
        eprintln!(
            "wind {wind:?} timestep position error {}",
            coarse.position.distance(fine.position)
        );
        assert!(coarse.position.distance(fine.position) < 0.005);
        assert!(
            coarse
                .leaf
                .unwrap()
                .frame
                .normal
                .distance(fine.leaf.unwrap().frame.normal)
                < 0.05
        );
    }
}

#[test]
fn calm_air_leaf_lands_and_settles_without_artificial_horizontal_braking() {
    let (a, mut p, anchor) = case(0.6);
    p.position.y = 2.0;
    for _ in 0..2400 {
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
    eprintln!(
        "settled leaf position={:?} velocity={:?} spin={:?}",
        p.position,
        p.velocity,
        p.leaf.unwrap().frame.angular_velocity
    );
    assert!(
        p.position.y < 0.02
            && p.velocity.length() < 0.001
            && p.leaf.unwrap().frame.angular_velocity.length() < 0.01
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn gpu_still_air_flutter_and_tumbling_are_dynamic_not_visual() {
    use molla_compute::Engine;
    use molla_solvers::attachments::GpuAttachments;
    let engine = Engine::new_gpu().unwrap();
    let (a, p, anchor) = case(0.6);
    let mut initial = vec![p; 2];
    for (p, inertia) in initial.iter_mut().zip([1.1, 1.4]) {
        let leaf = p.leaf.as_mut().unwrap();
        leaf.area = 4.0 * a.mass / (std::f64::consts::PI * 1.225 * leaf.length * inertia);
    }
    let mut gpu = GpuAttachments::new(&engine, vec![a; 2], &initial).unwrap();
    let mut reversals = [0; 2];
    let mut signs = [0; 2];
    let mut upward = [0; 2];
    let mut previous = [0.6; 2];
    let mut angles = [0.6; 2];
    let mut lo = [0.6f64; 2];
    let mut hi = [0.6f64; 2];
    let c = Config {
        dt: 1.0 / 60.0,
        ground: -100.0,
        ..Default::default()
    };
    for _ in 0..480 {
        gpu.step(&engine, &[anchor; 2], c).unwrap();
        for (i, p) in gpu.snapshot(&engine).unwrap().iter().enumerate() {
            let sign = if p.velocity.x > 0.1 {
                1
            } else if p.velocity.x < -0.1 {
                -1
            } else {
                0
            };
            if sign != 0 {
                if signs[i] != 0 && sign != signs[i] {
                    reversals[i] += 1;
                }
                signs[i] = sign;
            }
            if p.velocity.y > 0.05 {
                upward[i] += 1;
            }
            let d = p.leaf.unwrap().frame.direction;
            let angle = d.y.atan2(d.x);
            angles[i] += (angle - previous[i])
                .sin()
                .atan2((angle - previous[i]).cos());
            previous[i] = angle;
            lo[i] = lo[i].min(angles[i]);
            hi[i] = hi[i].max(angles[i]);
        }
    }
    eprintln!(
        "GPU reversals={reversals:?}, upward={upward:?}, spans={:?}",
        [hi[0] - lo[0], hi[1] - lo[1]]
    );
    assert!(reversals[0] >= 8 && hi[0] - lo[0] < std::f64::consts::TAU);
    assert!(hi[1] - lo[1] > 4.0 * std::f64::consts::TAU);
    assert!(upward.iter().all(|n| *n > 10));
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn gpu_oriented_leaf_ground_impulse_matches_cpu() {
    use molla_compute::Engine;
    use molla_solvers::attachments::GpuAttachments;
    let engine = Engine::new_gpu().unwrap();
    let (a, mut cpu, anchor) = case(0.4);
    cpu.position.y = 0.015;
    cpu.velocity = Vec3::new(0.5, -2.0, 0.7);
    cpu.leaf.as_mut().unwrap().frame.angular_velocity = Vec3::new(1.0, 0.2, 0.8);
    let mut gpu = GpuAttachments::new(&engine, vec![a], &[cpu]).unwrap();
    let c = Config {
        dt: 1.0 / 480.0,
        wind: Vec3::X * 2.0,
        ..Default::default()
    };
    for _ in 0..8 {
        step_cpu(a, &mut cpu, anchor, c).unwrap();
        gpu.step(&engine, &[anchor], c).unwrap();
        let p = gpu.snapshot(&engine).unwrap()[0];
        assert!(p.position.distance(cpu.position) < 1e-5);
        assert!(p.velocity.distance(cpu.velocity) < 1e-4);
        assert!(
            p.leaf
                .unwrap()
                .frame
                .angular_velocity
                .distance(cpu.leaf.unwrap().frame.angular_velocity)
                < 0.01
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn gpu_leaf_force_rotation_boundary_and_restore_match_cpu() {
    use molla_compute::Engine;
    use molla_solvers::attachments::GpuAttachments;
    let engine = Engine::new_gpu().unwrap();
    let (a, p, anchor) = case(0.7);
    let mut cpu = vec![p; 3];
    cpu[1].position.x = 4.7;
    cpu[2].attached = true;
    cpu[2].position.y = 20.0;
    let frames = vec![p.leaf.unwrap().frame; 3];
    let anchors = vec![anchor; 3];
    let mut gpu = GpuAttachments::new(&engine, vec![a; 3], &cpu).unwrap();
    let winds = vec![
        Vec3::new(2.0, 0.3, 0.5),
        Vec3::new(2.0, 0.3, 0.5),
        Vec3::new(1.0, -12.0, 2.0),
    ];
    let c = Config {
        dt: 1.0 / 120.0,
        ..Default::default()
    };
    for _ in 0..120 {
        for (p, wind) in cpu.iter_mut().zip(&winds) {
            step_cpu(a, p, anchor, Config { wind: *wind, ..c }).unwrap();
        }
        gpu.step_spatial_blades(&engine, &anchors, c, &winds, &frames)
            .unwrap();
    }
    let actual = gpu.snapshot(&engine).unwrap();
    for (g, c) in actual.iter().zip(&cpu) {
        assert!(
            g.position.distance(c.position) < 0.002,
            "GPU {:?}, CPU {:?}",
            g.position,
            c.position
        );
        assert!(
            g.leaf
                .unwrap()
                .frame
                .normal
                .distance(c.leaf.unwrap().frame.normal)
                < 0.02
        );
        assert!(g.velocity.distance(c.velocity) < 0.01);
    }
    assert!(actual[1].position.x > 4.8);
    gpu.restore(&engine, &cpu).unwrap();
    let restored = gpu.snapshot(&engine).unwrap();
    for (a, b) in restored.iter().zip(&cpu) {
        assert!(
            a.leaf
                .unwrap()
                .frame
                .angular_velocity
                .distance(b.leaf.unwrap().frame.angular_velocity)
                < 1e-4
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn moving_owner_rotation_is_inherited_once_at_release() {
    use molla_compute::Engine;
    use molla_solvers::attachments::GpuAttachments;
    let engine = Engine::new_gpu().unwrap();
    let (mut a, mut cpu, mut anchor) = case(0.0);
    a.strength = 1.0;
    cpu.attached = true;
    let mut gpu = GpuAttachments::new(&engine, vec![a], &[cpu]).unwrap();
    let c = Config {
        dt: 1.0 / 120.0,
        wind: Vec3::new(1.0, 0.0, 0.3),
        ..Default::default()
    };
    for i in 0..120 {
        let q = Quat::from_rotation_z(i as f64 * c.dt * 3.0);
        let frame = BladeFrame {
            direction: q * Vec3::X,
            normal: q * Vec3::Y,
            angular_velocity: Vec3::Z * 3.0,
        };
        anchor.valid = i < 60;
        if cpu.attached {
            cpu.leaf.as_mut().unwrap().frame = frame;
        }
        step_cpu(a, &mut cpu, anchor, c).unwrap();
        gpu.step_spatial_blades(&engine, &[anchor], c, &[c.wind], &[frame])
            .unwrap();
    }
    let p = gpu.snapshot(&engine).unwrap()[0];
    assert!(!p.attached);
    assert!(p.position.distance(cpu.position) < 0.001);
    assert!(
        p.leaf
            .unwrap()
            .frame
            .normal
            .distance(cpu.leaf.unwrap().frame.normal)
            < 0.002
    );
    assert!(
        p.leaf
            .unwrap()
            .frame
            .angular_velocity
            .distance(cpu.leaf.unwrap().frame.angular_velocity)
            < 0.01
    );
}
