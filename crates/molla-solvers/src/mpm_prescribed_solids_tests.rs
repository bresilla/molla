use super::*;
use crate::mpm::replay_tests::read_words;

fn solid(shape: MpmSolidShape) -> MpmPrescribedSolid {
    MpmPrescribedSolid {
        shape,
        center: [0.0; 3],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.125, -0.25, 0.0625],
        angular_velocity: [0.5, -0.25, 1.0],
    }
}

fn config() -> MpmConfig {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 129;
    cfg.max_rigid = 2;
    cfg.youngs = 1000.0;
    cfg.gravity_y = -2.0;
    cfg
}

#[test]
fn prescribed_solid_validation_and_rotated_bounds() {
    for shape in [
        MpmSolidShape::Sphere { radius: 0.4 },
        MpmSolidShape::Box {
            half_extents: [0.1, 0.6, 0.2],
        },
        MpmSolidShape::Cylinder {
            radius: 0.4,
            half_width: 0.1,
        },
    ] {
        let body = solid(shape);
        let (bounds, motion) = body.rows().unwrap();
        assert!(bounds[3] >= 0.4);
        assert_ne!(motion[3][3], 0.0);
        let mut invalid = body;
        invalid.orientation = [0.0; 4];
        assert!(invalid.rows().is_err());
        invalid = body;
        invalid.angular_velocity[1] = f32::NAN;
        assert!(invalid.rows().is_err());
        invalid = body;
        invalid.center[0] = f32::INFINITY;
        assert!(invalid.rows().is_err());
    }
    assert!(
        solid(MpmSolidShape::Cylinder {
            radius: 1.0,
            half_width: 0.0
        })
        .rows()
        .is_err()
    );
    assert!(
        solid(MpmSolidShape::Box {
            half_extents: [1.0, -1.0, 1.0]
        })
        .rows()
        .is_err()
    );
}

fn point(cfg: &MpmConfig, world: usize) -> [f32; 3] {
    let n = cfg.grid_size as usize;
    [world % n, world / n % n, world / n / n].map(|x| cfg.grid_origin[0] + x as f32 * cfg.cell_size)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn prescribed_solid_grid_impulses_preserve_mass_and_balance_wrench() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = config();
    let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
    let world = 9 + 8 * 16 + 9 * 16 * 16;
    let p = point(&cfg, world);
    let dt = 1.0 / 128.0;
    let mass = 0.125_f32;
    let momentum = [-0.03125_f32, 0.0625, 0.015625];
    let cases = [
        (solid(MpmSolidShape::Sphere { radius: 0.3 }), true),
        (
            solid(MpmSolidShape::Box {
                half_extents: [0.2, 0.4, 0.2],
            }),
            true,
        ),
        (
            solid(MpmSolidShape::Box {
                half_extents: [0.1, 0.4, 0.2],
            }),
            false,
        ),
        (
            solid(MpmSolidShape::Cylinder {
                radius: 0.3,
                half_width: 0.1,
            }),
            true,
        ),
        (
            MpmPrescribedSolid {
                center: [0.0, -0.125, 0.0],
                ..solid(MpmSolidShape::Cylinder {
                    radius: 0.3,
                    half_width: 0.1,
                })
            },
            false,
        ),
        (
            MpmPrescribedSolid {
                orientation: [
                    0.0,
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                    std::f32::consts::FRAC_1_SQRT_2,
                ],
                ..solid(MpmSolidShape::Cylinder {
                    radius: 0.3,
                    half_width: 0.1,
                })
            },
            false,
        ),
    ];
    for (case, (body, inside)) in cases.into_iter().enumerate() {
        let owner = case % 2;
        let first = if owner == 0 {
            body
        } else {
            MpmPrescribedSolid {
                center: [10.0; 3],
                ..body
            }
        };
        sim.set_prescribed_solids(queue, &[first, body]).unwrap();
        queue.write_buffer(
            &sim.uniform_buf,
            0,
            bytemuck::bytes_of(&sim.step_uniforms(dt)),
        );
        let mut masses = vec![0.0_f32; 16usize.pow(3)];
        masses[world] = mass;
        let mut momenta = vec![0.0_f32; masses.len() * 3];
        momenta[3 * world..3 * world + 3].copy_from_slice(&momentum);
        queue.write_buffer(&sim.g_mass, 0, bytemuck::cast_slice(&masses));
        queue.write_buffer(&sim.g_momentum, 0, bytemuck::cast_slice(&momenta));
        let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&sim.g_rigid_mask, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_pipeline(&sim.pipe_rigid_wrench_reset);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&sim.pipe_rigid);
            pass.dispatch_workgroups(64, 1, 1);
            pass.set_pipeline(&sim.pipe_grid);
            pass.dispatch_workgroups(64, 1, 1);
            pass.set_pipeline(&sim.pipe_rigid_force);
            pass.dispatch_workgroups(64, 1, 1);
        }
        queue.submit([encoder.finish()]);
        assert_eq!(resources.finish().observations, 0);
        assert_eq!(transfers.finish().readback_bytes, 0);
        let actual_mass = read_words(device, queue, &sim.g_mass);
        assert_eq!(
            actual_mass,
            masses.iter().map(|m| m.to_bits()).collect::<Vec<_>>()
        );
        let actual_momentum = read_words(device, queue, &sim.g_momentum);
        let wrench = sim.read_rigid_wrenches_with(device, queue).unwrap();
        assert_eq!(
            wrench[1 - owner],
            ([0.0; 3], [0.0; 3]),
            "overlap must have one owner"
        );
        let arm = std::array::from_fn(|i| p[i] - body.center[i]);
        let spin = cross(body.angular_velocity, arm);
        let expected_momentum: [f32; 3] = std::array::from_fn(|i| {
            if inside {
                mass * (body.linear_velocity[i] + spin[i])
            } else {
                momentum[i]
            }
        });
        let force = std::array::from_fn(|i| {
            -(expected_momentum[i] - momentum[i]) / dt
                + if inside && i == 1 {
                    mass * cfg.gravity_y
                } else {
                    0.0
                }
        });
        let torque = cross(arm, force);
        for i in 0..3 {
            assert!(
                (f32::from_bits(actual_momentum[3 * world + i]) - expected_momentum[i]).abs()
                    <= 2e-5
            );
            assert!(
                (wrench[owner].0[i] - force[i]).abs() <= 2e-5,
                "force case {body:?}"
            );
            assert!((wrench[owner].1[i] - torque[i]).abs() <= 2e-5);
        }
    }
    let before = read_words(device, queue, &sim.rigid_vel);
    let mut invalid = solid(MpmSolidShape::Sphere { radius: 0.3 });
    invalid.orientation = [0.0; 4];
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    assert!(sim.set_prescribed_solids(queue, &[invalid]).is_err());
    assert_eq!(transfers.finish().upload_bytes, 0);
    assert_eq!(before, read_words(device, queue, &sim.rigid_vel));
    sim.set_time_step(MpmTimeStep::Implicit { jacobi_iters: 2 });
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    assert!(sim.step(&engine, dt).is_err());
    assert!(sim.prepare_step(dt, 1).is_err());
    assert!(
        sim.set_prescribed_solids(queue, &[solid(MpmSolidShape::Sphere { radius: 0.3 })])
            .is_err()
    );
    assert_eq!(transfers.finish().upload_bytes, 0);
    sim.set_time_step(MpmTimeStep::Explicit);
    assert!(sim.step(&engine, 0.0).is_err());
    sim.set_prescribed_solids(queue, &[]).unwrap();
    assert_eq!(sim.rigid_count, 0);
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn prescribed_solids_dense_active_compact_motion_and_removal() {
    let engine = Engine::new_gpu().unwrap();
    let cfg = config();
    let points: Vec<_> = (0..65)
        .map(|i| {
            [
                -0.125 + (i % 5) as f32 * 0.0625,
                -0.125 + (i / 5 % 5) as f32 * 0.0625,
                (i / 25) as f32 * 0.0625,
            ]
        })
        .collect();
    let body = MpmPrescribedSolid {
        linear_velocity: [0.01, 0.0, 0.02],
        angular_velocity: [0.0, 0.25, 0.0],
        ..solid(MpmSolidShape::Box {
            half_extents: [0.3, 0.3, 0.1],
        })
    };
    let mut outputs = vec![];
    for mode in 0..3 {
        let mut sim = if mode == 2 {
            MpmSimulation::new_sparse(&engine, cfg, 64).unwrap()
        } else {
            MpmSimulation::new(&engine, cfg).unwrap()
        };
        sim.set_active_grid(mode != 0).unwrap();
        sim.set_particles(engine.queue().unwrap(), &points).unwrap();
        sim.set_prescribed_solids(engine.queue().unwrap(), &[body])
            .unwrap();
        sim.step_n(&engine, 0.0001, 8).unwrap();
        let before = sim
            .read_velocities_with(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        assert!(before.iter().flatten().any(|v| v.abs() > 1e-4));
        sim.set_prescribed_solids(engine.queue().unwrap(), &[])
            .unwrap();
        sim.step_n(&engine, 0.0001, 2).unwrap();
        assert!(sim.read_rigid_wrenches(&engine).unwrap().is_empty());
        outputs.push(
            [
                &sim.p_pos,
                &sim.p_vel,
                &sim.p_c,
                &sim.p_f,
                &sim.p_volume,
                &sim.p_mass,
                &sim.p_jp,
                &sim.p_material,
                &sim.tire_compaction_buf,
            ]
            .map(|buffer| read_words(engine.device().unwrap(), engine.queue().unwrap(), buffer)),
        );
    }
    for output in &outputs[1..] {
        for (field, (a, b)) in outputs[0].iter().zip(output).enumerate() {
            if field == 7 {
                assert_eq!(a, b);
                continue;
            }
            for (a, b) in a.iter().zip(b) {
                let (a, b) = (f32::from_bits(*a), f32::from_bits(*b));
                assert!(
                    a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-5,
                    "{a} != {b}"
                );
            }
        }
    }
}
