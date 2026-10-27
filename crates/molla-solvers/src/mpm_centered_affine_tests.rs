use super::*;

struct Snapshot {
    positions: Vec<[f32; 3]>,
    velocities: Vec<[f32; 3]>,
    wrenches: Vec<([f32; 3], [f32; 3])>,
}

fn errors(a: &Snapshot, b: &Snapshot) -> [f64; 4] {
    let maximum = |a: &[f32], b: &[f32], relative: bool| {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&a, &b)| {
                assert!(a.is_finite() && b.is_finite());
                f64::from((a - b).abs() / (2e-5 * if relative { b.abs().max(1.) } else { 1. }))
            })
            .fold(0., f64::max)
    };
    assert_eq!(a.wrenches.len(), b.wrenches.len());
    let mut result = [
        maximum(
            a.positions.as_flattened(),
            b.positions.as_flattened(),
            false,
        ),
        maximum(
            a.velocities.as_flattened(),
            b.velocities.as_flattened(),
            false,
        ),
        0.,
        0.,
    ];
    for (a, b) in a.wrenches.iter().zip(&b.wrenches) {
        result[2] = result[2].max(maximum(&a.0, &b.0, true));
        result[3] = result[3].max(maximum(&a.1, &b.1, true));
    }
    result
}

#[test]
#[ignore = "requires a GPU adapter; bounded 50k centered-affine diagnostic"]
fn local_transfer_centered_wrench_matrix() {
    wrench_matrix("centered", MpmSimulation::install_centered_affine_transfer);
}

#[test]
#[ignore = "requires a GPU adapter; bounded 50k relative-momentum diagnostic"]
fn relative_transfer_wrench_matrix() {
    wrench_matrix(
        "relative-momentum",
        MpmSimulation::install_relative_momentum_transfer,
    );
}

fn wrench_matrix(name: &str, install: fn(&mut MpmSimulation)) {
    println!("wrench-matrix candidate={name}");
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let points: Vec<_> = (0..50_000)
        .map(|i| {
            [
                -0.65 + (i % 40) as f32 * 0.025,
                -0.65 + (i / 40 % 40) as f32 * 0.025,
                -0.65 + (i / 1600) as f32 * 0.025,
            ]
        })
        .collect();
    let velocities = vec![[0.03, -0.02, 0.01]; points.len()];
    let spheres = [
        ([0.2, -0.2, -0.2], 0.1, [0.02, -0.01, 0.]),
        ([-0.87, 0.8, -0.87], 0.08, [0.; 3]),
    ];
    let mut failed = [0_usize; 3];
    for implicit in [false, true] {
        for size in [32, 64, 96] {
            let mut cfg = MpmConfig::sand();
            cfg.grid_size = size;
            cfg.grid_origin = [-1.; 3];
            cfg.cell_size = 0.0625;
            cfg.max_particles = 50_000;
            cfg.max_rigid = 4;
            cfg.youngs = 3000.;
            cfg.particle_volume = 0.025_f32.powi(3);
            cfg.particle_mass = 1600. * cfg.particle_volume;
            cfg.time_step = if implicit {
                MpmTimeStep::Implicit { jacobi_iters: 3 }
            } else {
                MpmTimeStep::Explicit
            };
            let mut owners: Vec<_> = (0..8)
                .map(|owner| {
                    let mode = owner % 4;
                    let mut sim = if mode == 3 {
                        MpmSimulation::new_sparse(&engine, cfg, 256).unwrap()
                    } else {
                        MpmSimulation::new(&engine, cfg).unwrap()
                    };
                    if mode == 2 {
                        sim.set_active_grid(true).unwrap();
                    }
                    if owner >= 4 {
                        install(&mut sim);
                    }
                    sim.set_rigid_spheres(queue, &spheres).unwrap();
                    sim
                })
                .collect();
            for round in 0..4 {
                let mut outputs: Vec<Option<Snapshot>> = (0..8).map(|_| None).collect();
                for offset in 0..8 {
                    let owner = (round + offset) % 8;
                    let sim = &mut owners[owner];
                    sim.set_particles_with_velocity(queue, &points, &velocities)
                        .unwrap();
                    sim.step_n_with(device, queue, 0.0001, 4).unwrap();
                    if owner % 4 >= 2 {
                        let status = sim.read_active_grid_stats_with(device, queue).unwrap();
                        assert!(!status.overflow);
                        assert_eq!(status.completed_substeps, 4 * (round as u32 + 1));
                    }
                    outputs[owner] = Some(Snapshot {
                        positions: sim.read_positions_with(device, queue).unwrap(),
                        velocities: sim.read_velocities_with(device, queue).unwrap(),
                        wrenches: sim.read_rigid_wrenches_with(device, queue).unwrap(),
                    });
                }
                for (variant, count) in failed[..2].iter_mut().enumerate() {
                    for mode in 1..4 {
                        let error = errors(
                            outputs[variant * 4 + mode].as_ref().unwrap(),
                            outputs[variant * 4].as_ref().unwrap(),
                        );
                        let passed = error.into_iter().all(|e| e <= 1.);
                        *count += usize::from(!passed);
                        println!(
                            "centered-wrench grid={size} implicit={implicit} round={round} candidate={} mode={mode} error_over_tolerance={error:?} passed={passed}",
                            variant == 1
                        );
                    }
                }
                let error = errors(outputs[4].as_ref().unwrap(), outputs[0].as_ref().unwrap());
                let passed = error.into_iter().all(|e| e <= 1.);
                failed[2] += usize::from(!passed);
                println!(
                    "centered-wrench grid={size} implicit={implicit} round={round} cross_variant=true error_over_tolerance={error:?} passed={passed}"
                );
            }
        }
    }
    println!(
        "centered-wrench failed_comparisons control={} candidate={} cross_variant={}",
        failed[0], failed[1], failed[2]
    );
    assert_eq!(
        failed[1], 0,
        "centered own-repeat/domain parity remains open"
    );
    assert_eq!(
        failed[2], 0,
        "centered versus original dense parity remains open"
    );
}

#[test]
#[ignore = "requires a GPU adapter; bounded 50k safe-reference diagnostic"]
fn bounded_reference_wrench_matrix() {
    wrench_matrix(
        "bounded-relative",
        MpmSimulation::install_bounded_relative_transfer,
    );
}

#[test]
#[ignore = "requires a GPU adapter; bounded 50k combined-transfer diagnostic"]
fn bounded_centered_wrench_matrix() {
    wrench_matrix(
        "bounded-relative-centered-affine",
        MpmSimulation::install_bounded_centered_transfer,
    );
}
