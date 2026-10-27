use super::*;
use glam::{DMat3, DVec3};

#[path = "mpm_gather_oracle.rs"]
mod gather;

#[derive(Clone, Default)]
struct Cell {
    mass: f64,
    momentum: DVec3,
    velocity: DVec3,
    rigid: bool,
}

fn vector(data: &[f64], offset: usize) -> DVec3 {
    DVec3::from_slice(&data[offset..offset + 3])
}

fn matrix(data: &[f64], offset: usize) -> DMat3 {
    DMat3::from_cols(
        vector(data, offset),
        vector(data, offset + 4),
        vector(data, offset + 8),
    )
}

fn node(cfg: &MpmConfig, i: usize) -> DVec3 {
    let n = cfg.grid_size as usize;
    DVec3::from_array(cfg.grid_origin.map(f64::from))
        + f64::from(cfg.cell_size)
            * DVec3::new((i % n) as f64, (i / n % n) as f64, (i / (n * n)) as f64)
}

fn scatter(cfg: &MpmConfig, dt: f64, input: &[Vec<f64>; 7]) -> Vec<Cell> {
    let mut grid = vec![Cell::default(); cfg.grid_size.pow(3) as usize];
    let dx = f64::from(cfg.cell_size);
    let origin = DVec3::from_array(cfg.grid_origin.map(f64::from));
    let mu = f64::from(cfg.youngs) / (2. * (1. + f64::from(cfg.poisson)));
    let lam = f64::from(cfg.youngs) * f64::from(cfg.poisson)
        / ((1. + f64::from(cfg.poisson)) * (1. - 2. * f64::from(cfg.poisson)));
    let mut expected_mass = 0.;
    let mut expected_momentum = DVec3::ZERO;
    let mut expected_angular = DVec3::ZERO;
    for p in 0..input[4].len() {
        let pos = vector(&input[0], p * 4);
        let vel = vector(&input[1], p * 4);
        let c = matrix(&input[2], p * 12);
        let f = matrix(&input[3], p * 12);
        let volume = input[4][p];
        let mass = input[5][p];
        let hardening = input[6][p].clamp(-3., 3.).exp();
        let (u, s, v) = molla_math::dense3::svd3(f);
        let reconstructed = u * DMat3::from_diagonal(s) * v.transpose();
        assert!(
            (reconstructed - f)
                .to_cols_array()
                .into_iter()
                .all(|x| x.abs() < 1e-10)
        );
        let s = s.clamp(DVec3::splat(0.1), DVec3::splat(10.));
        let fc = u * DMat3::from_diagonal(s) * v.transpose();
        let r = u * v.transpose();
        let cof =
            u * DMat3::from_diagonal(DVec3::new(s.y * s.z, s.x * s.z, s.x * s.y)) * v.transpose();
        let pk1 =
            (fc - r) * (2. * mu * hardening) + cof * (lam * hardening * (s.x * s.y * s.z - 1.));
        let stress = (pk1 * fc.transpose()) * (-dt * volume);
        let xp = (pos - origin) / dx;
        let base = (xp - DVec3::splat(0.5)).floor();
        let fx = xp - base;
        let weights = fx.to_array().map(|f| {
            [
                0.5 * (1.5 - f).powi(2),
                0.75 - (f - 1.).powi(2),
                0.5 * (f - 0.5).powi(2),
            ]
        });
        assert!(
            base.min_element() >= 0.
                && (base + DVec3::splat(2.)).max_element() < f64::from(cfg.grid_size)
        );
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let x = base.x as usize + i;
                    let y = base.y as usize + j;
                    let z = base.z as usize + k;
                    let n = cfg.grid_size as usize;
                    let cell = &mut grid[x + n * (y + n * z)];
                    let w = weights[0][i] * weights[1][j] * weights[2][k];
                    let arm = (DVec3::new(i as f64, j as f64, k as f64) - fx) * dx;
                    cell.mass += w * mass;
                    cell.momentum +=
                        (vel + c * arm) * (w * mass) + stress * arm * (w * 4. / (dx * dx));
                }
            }
        }
        expected_mass += mass;
        expected_momentum += mass * vel;
        expected_angular += pos.cross(mass * vel)
            + mass
                * dx
                * dx
                * 0.25
                * DVec3::new(
                    c.y_axis.z - c.z_axis.y,
                    c.z_axis.x - c.x_axis.z,
                    c.x_axis.y - c.y_axis.x,
                );
    }
    let mass: f64 = grid.iter().map(|c| c.mass).sum();
    let momentum: DVec3 = grid.iter().map(|c| c.momentum).sum();
    let angular: DVec3 = grid
        .iter()
        .enumerate()
        .map(|(i, c)| node(cfg, i).cross(c.momentum))
        .sum();
    assert!((mass - expected_mass).abs() <= 1e-10 * expected_mass.max(1.));
    assert!(
        (momentum - expected_momentum).abs().max_element()
            <= 1e-10 * expected_momentum.abs().max_element().max(1.)
    );
    assert!(
        (angular - expected_angular).abs().max_element()
            <= 1e-10 * expected_angular.abs().max_element().max(1.)
    );
    grid
}

fn couple(
    cfg: &MpmConfig,
    dt: f64,
    grid: &mut [Cell],
    spheres: &[(DVec3, f64, DVec3)],
    implicit: bool,
) -> Vec<[[f64; 3]; 2]> {
    let n = cfg.grid_size as usize;
    let mut wrench = vec![[DVec3::ZERO; 2]; spheres.len()];
    for (index, cell) in grid.iter_mut().enumerate() {
        let pos = node(cfg, index);
        for (body, &(center, radius, velocity)) in spheres.iter().enumerate() {
            if pos.distance(center) < radius {
                if cell.mass > 1e-9 {
                    let force = (cell.momentum - cell.mass * velocity) / dt.max(1e-6);
                    wrench[body][0] += force;
                    wrench[body][1] += (pos - center).cross(force);
                }
                cell.mass = 1e6;
                cell.momentum = cell.mass * velocity;
                cell.rigid = true;
                break;
            }
        }
        if cell.mass <= 1e-9 {
            continue;
        }
        cell.velocity = cell.momentum / cell.mass;
        if cell.rigid {
            continue;
        }
        cell.velocity.y += dt * f64::from(cfg.gravity_y);
        let coords = [index % n, index / n % n, index / (n * n)];
        for axis in 0..3 {
            if coords[axis] < 2 || coords[axis] >= n - 2 {
                cell.velocity[axis] = if coords[axis] < 2 {
                    cell.velocity[axis].max(0.)
                } else {
                    cell.velocity[axis].min(0.)
                };
                for other in 0..3 {
                    if other != axis {
                        cell.velocity[other] *= f64::from(cfg.boundary_friction);
                    }
                }
            }
        }
    }
    if implicit {
        for _ in 0..3 {
            let mut next: Vec<_> = grid.iter().map(|c| c.velocity).collect();
            for (index, cell) in grid.iter().enumerate() {
                if cell.rigid || cell.mass <= 1e-9 {
                    continue;
                }
                let coords = [index % n, index / n % n, index / (n * n)];
                let mut sum = DVec3::ZERO;
                let mut count = 0.;
                for axis in 0..3 {
                    for delta in [1_i32, -1] {
                        let value = coords[axis] as i32 + delta;
                        if value < 0 || value >= n as i32 {
                            continue;
                        }
                        let mut neighbor = coords;
                        neighbor[axis] = value as usize;
                        let neighbor = &grid[neighbor[0] + n * (neighbor[1] + n * neighbor[2])];
                        if neighbor.mass > 1e-9 {
                            sum += neighbor.velocity;
                            count += 1.;
                        }
                    }
                }
                let avg = if count > 0. {
                    sum / count
                } else {
                    cell.velocity
                };
                let alpha = dt * dt * cell.mass * f64::from(cfg.youngs)
                    / 1500.
                    / f64::from(cfg.cell_size).powi(2);
                next[index] = (cell.mass * cell.velocity + alpha * avg) / (cell.mass + alpha);
            }
            for (cell, value) in grid.iter_mut().zip(next) {
                cell.velocity = value;
            }
        }
    }
    for (index, cell) in grid.iter().enumerate() {
        if cell.rigid || cell.mass <= 1e-9 {
            continue;
        }
        let pos = node(cfg, index);
        for (body, &(center, radius, velocity)) in spheres.iter().enumerate() {
            let distance = pos.distance(center);
            if distance > radius + f64::from(cfg.cell_size) {
                continue;
            }
            let weight = if distance > radius {
                ((radius + f64::from(cfg.cell_size) - distance) / f64::from(cfg.cell_size)).max(0.)
            } else {
                1.
            };
            let force = -(cell.mass * weight / dt.max(1e-6)) * (velocity - cell.velocity);
            wrench[body][0] += force;
            wrench[body][1] += (pos - center).cross(force);
        }
    }
    wrench
        .into_iter()
        .map(|w| w.map(|v| v.to_array()))
        .collect()
}

#[test]
fn wrench_oracle_affine_particle_transfer() {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 8;
    cfg.grid_origin = [-0.5; 3];
    cfg.cell_size = 0.125;
    let fields = [
        vec![0.; 4],
        vec![0.1, 0.2, 0.3, 0.],
        vec![0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0.],
        vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
        vec![0.001],
        vec![2.],
        vec![0.],
    ];
    let grid = scatter(&cfg, 0.0001, &fields);
    let center = &grid[4 + 8 * (4 + 8 * 4)];
    assert!((center.mass - 0.84375).abs() < 1e-14);
    assert!(
        (center.momentum - center.mass * DVec3::new(0.1, 0.2, 0.3))
            .abs()
            .max_element()
            < 1e-14
    );
    let corner = &grid[3 + 8 * (3 + 8 * 3)];
    assert!((corner.mass - 0.00390625).abs() < 1e-14);
    assert!(
        (corner.momentum - corner.mass * DVec3::new(0.225, 0.075, 0.3))
            .abs()
            .max_element()
            < 1e-14
    );
}

#[test]
fn wrench_oracle_single_cell_reaction() {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 4;
    cfg.grid_origin = [-0.25; 3];
    cfg.cell_size = 0.125;
    let mut grid = vec![Cell::default(); 64];
    grid[2 + 4 * (2 + 4 * 2)].mass = 2.;
    grid[2 + 4 * (2 + 4 * 2)].momentum = DVec3::new(0.6, -0.4, 0.2);
    let w = couple(
        &cfg,
        0.5,
        &mut grid,
        &[(DVec3::new(0.01, 0., 0.), 0.02, DVec3::new(0.1, 0.2, 0.3))],
        false,
    );
    for (actual, expected) in w[0]
        .into_iter()
        .flatten()
        .zip([0.8, -1.6, -0.8, 0., -0.008, 0.016])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
#[ignore = "CPU f64 oracle requires captured 50k particle inputs"]
fn wrench_oracle_captured_transfer() {
    let original = std::path::PathBuf::from(
        std::env::var_os("MOLLA_WRENCH_ORACLE_INPUT").expect("input captures"),
    );
    let candidate = std::path::PathBuf::from(
        std::env::var_os("MOLLA_WRENCH_ORACLE_CANDIDATE").expect("candidate captures"),
    );
    let output = std::path::PathBuf::from(
        std::env::var_os("MOLLA_WRENCH_ORACLE_OUTPUT").expect("fresh output directory"),
    );
    std::fs::create_dir(&output).unwrap();
    let mut failures = [0; 2];
    for implicit in [false, true] {
        for two_pass in [false, true] {
            let name = format!("implicit-{implicit}-two-pass-{two_pass}");
            let input = original.join(&name);
            for i in 0..9 {
                let name = format!("particle-{i}.bin");
                assert_eq!(
                    std::fs::read(input.join(&name)).unwrap(),
                    std::fs::read(
                        candidate
                            .join(format!("implicit-{implicit}-two-pass-{two_pass}"))
                            .join(&name)
                    )
                    .unwrap(),
                    "oracle comparisons require identical particle inputs: {name}"
                );
            }
            let fields: [Vec<f64>; 7] = std::array::from_fn(|i| {
                let bytes = std::fs::read(input.join(format!("particle-{i}.bin"))).unwrap();
                bytes
                    .chunks_exact(4)
                    .map(|b| {
                        let x = f32::from_le_bytes(b.try_into().unwrap());
                        assert!(x.is_finite());
                        f64::from(x)
                    })
                    .collect()
            });
            assert_eq!(fields[4].len(), 50_000);
            let materials = std::fs::read(input.join("particle-7.bin")).unwrap();
            assert!(materials.iter().all(|b| *b == 0));
            let mut cfg = MpmConfig::sand();
            cfg.grid_size = 32;
            cfg.grid_origin = [-1.; 3];
            cfg.cell_size = 0.0625;
            cfg.youngs = 3000.;
            let dt = f64::from(0.0001_f32);
            let mut grid = scatter(&cfg, dt, &fields);
            let spheres = [
                (
                    DVec3::from_array([0.2_f32, -0.2, -0.2].map(f64::from)),
                    f64::from(0.1_f32),
                    DVec3::from_array([0.02_f32, -0.01, 0.].map(f64::from)),
                ),
                (
                    DVec3::from_array([-0.87_f32, 0.8, -0.87].map(f64::from)),
                    f64::from(0.08_f32),
                    DVec3::ZERO,
                ),
            ];
            let oracle = couple(&cfg, dt, &mut grid, &spheres, implicit);
            let mut results = Vec::new();
            for (variant, root) in [&original, &candidate].into_iter().enumerate() {
                let actual: Vec<[[f32; 3]; 2]> = serde_json::from_slice(
                    &std::fs::read(root.join(&name).join("reference.json")).unwrap(),
                )
                .unwrap();
                assert_eq!(actual.len(), oracle.len());
                let mut maximum = [0_f64; 2];
                for (a, b) in actual.iter().zip(&oracle) {
                    for kind in 0..2 {
                        for axis in 0..3 {
                            assert!(a[kind][axis].is_finite() && b[kind][axis].is_finite());
                            maximum[kind] = maximum[kind].max(
                                (f64::from(a[kind][axis]) - b[kind][axis]).abs()
                                    / (2e-5 * b[kind][axis].abs().max(1.)),
                            );
                        }
                    }
                }
                let passed = maximum.into_iter().all(|e| e <= 1.);
                failures[variant] += usize::from(!passed);
                println!(
                    "wrench-oracle case={name} variant={variant} force_torque_error_over_tolerance={maximum:?} passed={passed}"
                );
                results.push(serde_json::json!({"variant":variant,"actual":actual,"error_over_tolerance":maximum,"passed":passed}));
            }
            std::fs::write(
                output.join(format!("{name}.json")),
                serde_json::to_vec_pretty(&serde_json::json!({"oracle":oracle,"results":results}))
                    .unwrap(),
            )
            .unwrap();
        }
    }
    println!(
        "wrench-oracle failures control={} candidate={}",
        failures[0], failures[1]
    );
    assert_eq!(failures[1], 0, "candidate physical-reference gate");
}
