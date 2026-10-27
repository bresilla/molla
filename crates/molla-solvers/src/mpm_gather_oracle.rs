use super::*;

fn gather(cfg: &MpmConfig, pos: DVec3, grid: &[f64]) -> (DVec3, DMat3) {
    let dx = f64::from(cfg.cell_size);
    let xp = (pos - DVec3::from_array(cfg.grid_origin.map(f64::from))) / dx;
    let base = (xp - DVec3::splat(0.5)).floor();
    let fx = xp - base;
    let weights = fx.to_array().map(|x| {
        [
            0.5 * (1.5 - x).powi(2),
            0.75 - (x - 1.).powi(2),
            0.5 * (x - 0.5).powi(2),
        ]
    });
    let n = cfg.grid_size as i32;
    let mut velocity = DVec3::ZERO;
    let mut affine = DMat3::ZERO;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let xyz = [
                    base.x as i32 + i as i32,
                    base.y as i32 + j as i32,
                    base.z as i32 + k as i32,
                ];
                if xyz.into_iter().any(|x| x < 0 || x >= n) {
                    continue;
                }
                let index = (xyz[0] + n * (xyz[1] + n * xyz[2])) as usize;
                let v = vector(grid, 4 * index);
                let w = weights[0][i] * weights[1][j] * weights[2][k];
                let arm = (DVec3::new(i as f64, j as f64, k as f64) - fx) * dx;
                velocity += w * v;
                affine += DMat3::from_cols(v * arm.x, v * arm.y, v * arm.z) * (4. * w / (dx * dx));
            }
        }
    }
    (velocity, affine)
}

fn read(path: &std::path::Path, count: usize) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len(), 4 * count, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|b| {
            let value = f32::from_le_bytes(b.try_into().unwrap());
            assert!(value.is_finite(), "{}", path.display());
            f64::from(value)
        })
        .collect()
}

#[test]
fn gather_oracle_constant_and_affine() {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 8;
    cfg.grid_origin = [-0.5; 3];
    cfg.cell_size = 0.125;
    let position = DVec3::new(-0.03125, 0.09375, 0.);
    let translation = DVec3::new(0.2, -0.3, 0.1);
    for gradient in [
        DMat3::ZERO,
        DMat3::from_cols(
            DVec3::new(1., 0., 0.25),
            DVec3::new(0.5, 1., 0.),
            DVec3::new(0., -0.25, 1.),
        ),
    ] {
        let grid: Vec<_> = (0..512)
            .flat_map(|i| {
                let v = translation + gradient * node(&cfg, i);
                [v.x, v.y, v.z, 0.]
            })
            .collect();
        let (velocity, affine) = gather(&cfg, position, &grid);
        assert!(
            (velocity - translation - gradient * position)
                .abs()
                .max_element()
                < 1e-14
        );
        assert!(
            (affine - gradient)
                .to_cols_array()
                .into_iter()
                .all(|e| e.abs() < 1e-14)
        );
    }
}

#[test]
#[ignore = "CPU f64 gather oracle requires captured evolved fields"]
fn gather_oracle_captured_transfer() {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_GATHER_ORACLE_INPUT").expect("capture directory"),
    );
    let output = std::path::PathBuf::from(
        std::env::var_os("MOLLA_GATHER_ORACLE_OUTPUT").expect("fresh output directory"),
    );
    std::fs::create_dir(&output).unwrap();
    let mut failures = 0;
    for implicit in [false, true] {
        for two_pass in [false, true] {
            let name = format!("implicit-{implicit}-two-pass-{two_pass}");
            let case = root.join(&name);
            let mut cfg = MpmConfig::sand();
            cfg.grid_size = 32;
            cfg.grid_origin = [-1.; 3];
            cfg.cell_size = 0.0625;
            let count = 50_000;
            let materials = std::fs::read(case.join("particle-7.bin")).unwrap();
            assert_eq!(materials.len(), 4 * count);
            assert!(materials.iter().all(|x| *x == 0));
            let input = read(&case.join("particle-0.bin"), 4 * count);
            let grid = read(&case.join("gather-grid-velocity.bin"), 4 * 32_usize.pow(3));
            let actual = [
                read(&case.join("gather-position.bin"), 4 * count),
                read(&case.join("gather-velocity.bin"), 4 * count),
                read(&case.join("gather-affine.bin"), 12 * count),
            ];
            let mut maxima = [0_f64; 3];
            let mut worst = [0; 3];
            for p in 0..count {
                let pos = vector(&input, 4 * p);
                let (mut velocity, affine) = gather(&cfg, pos, &grid);
                let next = pos + f64::from(0.0001_f32) * velocity;
                let lo = DVec3::splat(-1. + 2. * f64::from(cfg.cell_size));
                let hi = DVec3::splat(-1. + 29. * f64::from(cfg.cell_size));
                for axis in 0..3 {
                    if next[axis] < lo[axis] || next[axis] > hi[axis] {
                        velocity[axis] = 0.;
                    }
                }
                let expected = [
                    next.clamp(lo, hi).to_array().to_vec(),
                    velocity.to_array().to_vec(),
                    affine.to_cols_array().to_vec(),
                ];
                let observed = [
                    vector(&actual[0], 4 * p).to_array().to_vec(),
                    vector(&actual[1], 4 * p).to_array().to_vec(),
                    matrix(&actual[2], 12 * p).to_cols_array().to_vec(),
                ];
                for field in 0..3 {
                    for (a, b) in observed[field].iter().zip(&expected[field]) {
                        assert!(a.is_finite() && b.is_finite());
                        let e = (a - b).abs() / (2e-5 * b.abs().max(1.));
                        if e > maxima[field] {
                            maxima[field] = e;
                            worst[field] = p;
                        }
                    }
                }
            }
            let passed = maxima.into_iter().all(|e| e <= 1.);
            failures += usize::from(!passed);
            println!(
                "gather-oracle case={name} position_velocity_affine_error_over_tolerance={maxima:?} worst_particles={worst:?} passed={passed}"
            );
            std::fs::write(output.join(format!("{name}.json")),serde_json::to_vec_pretty(&serde_json::json!({"error_over_tolerance":maxima,"worst_particles":worst,"passed":passed,"particles":count})).unwrap()).unwrap();
        }
    }
    assert_eq!(failures, 0, "gather physical-reference gate");
}
