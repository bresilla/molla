use super::*;

fn set(checkpoint: &mut MpmSoilCheckpoint, stream: usize, values: &[f32]) {
    let offset = STRIDES[..stream].iter().sum::<usize>() * checkpoint.count as usize;
    assert_eq!(values.len(), checkpoint.count as usize * STRIDES[stream]);
    for (word, value) in checkpoint.words[offset..].iter_mut().zip(values) {
        *word = value.to_bits();
    }
}

fn set_matrix(checkpoint: &mut MpmSoilCheckpoint, stream: usize, matrix: DMat3) {
    let values: Vec<_> = (0..3)
        .flat_map(|c| {
            let v = matrix.col(c).as_vec3();
            [v.x, v.y, v.z, 99.0]
        })
        .collect();
    set(checkpoint, stream, &values);
}

fn single() -> MpmSoilCheckpoint {
    let mut checkpoint = super::super::tests::fixture(1);
    checkpoint.materials.clear();
    checkpoint.cfg.rheology = MpmRheology::Sand {
        friction_angle: 0.5,
    };
    checkpoint.cfg.cell_size = 0.25;
    checkpoint.cfg.gravity_y = -10.0;
    set(&mut checkpoint, 7, &[0.0]);
    set(&mut checkpoint, 5, &[2.0]);
    set(&mut checkpoint, 4, &[3.0]);
    set(&mut checkpoint, 6, &[0.0]);
    set(&mut checkpoint, 0, &[0.0, 3.0, 0.0, 0.0]);
    set(&mut checkpoint, 1, &[3.0, 4.0, 0.0, 0.0]);
    set_matrix(
        &mut checkpoint,
        2,
        DMat3::from_diagonal(DVec3::new(1.0, 2.0, 3.0)),
    );
    set_matrix(&mut checkpoint, 3, DMat3::IDENTITY);
    checkpoint
}

#[test]
fn energy_components_and_height_reference_are_explicit() {
    let checkpoint = single();
    let before = checkpoint.to_bytes();
    let result = checkpoint.energy_at_height(1.0).unwrap();
    assert_eq!(result.translational_kinetic, 25.0);
    assert_eq!(result.affine_kinetic, 0.21875);
    assert_eq!(result.gravitational_potential, 40.0);
    assert_eq!(result.elastic_potential, Some(0.0));
    assert_eq!(result.elastic_unavailable_particles, 0);
    assert_eq!(result.mechanical_energy(), Some(65.21875));
    let shifted = checkpoint.energy_at_height(2.0).unwrap();
    assert_eq!(shifted.gravitational_potential, 20.0);
    assert_eq!(shifted.affine_kinetic, result.affine_kinetic);
    assert_eq!(checkpoint.to_bytes(), before);
}

#[test]
fn kinetic_energy_matches_independent_particle_quadrature() {
    let checkpoint = super::super::tests::fixture(129);
    let h = f64::from(checkpoint.cfg.cell_size);
    let origin = DVec3::from_array(checkpoint.cfg.grid_origin.map(f64::from));
    let mut expected = 0.0;
    for p in 0..129 {
        let value = |s, k| f64::from(f32::from_bits(checkpoint.stream(s)[p * STRIDES[s] + k]));
        let vector = |s, k| DVec3::new(value(s, k), value(s, k + 1), value(s, k + 2));
        let position = vector(0, 0);
        let velocity = vector(1, 0);
        let c = DMat3::from_cols(vector(2, 0), vector(2, 4), vector(2, 8));
        let xp = (position - origin) / h;
        let base = (xp - DVec3::splat(0.5)).floor();
        let f = xp - base;
        let w = f.to_array().map(|x| {
            [
                0.5 * (1.5 - x).powi(2),
                0.75 - (x - 1.0).powi(2),
                0.5 * (x - 0.5).powi(2),
            ]
        });
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let node = origin + h * (base + DVec3::new(i as f64, j as f64, k as f64));
                    expected += 0.5
                        * value(5, 0)
                        * w[0][i]
                        * w[1][j]
                        * w[2][k]
                        * (velocity + c * (node - position)).length_squared();
                }
            }
        }
    }
    let energy = checkpoint.energy_at_height(0.0).unwrap();
    assert!((energy.translational_kinetic + energy.affine_kinetic - expected).abs() < 1e-12);
}

#[test]
fn elastic_gradient_matches_frozen_material_pk1() {
    let f = DMat3::from_cols(
        DVec3::new(1.125, 0.0625, 0.0),
        DVec3::new(-0.03125, 0.9375, 0.0625),
        DVec3::new(0.0, 0.03125, 1.0625),
    );
    for material in [
        MpmRheology::Sand {
            friction_angle: 0.5,
        },
        MpmRheology::Snow {
            theta_c: 0.025,
            theta_s: 0.0075,
            hardening: 7.0,
        },
        MpmRheology::Viscous {
            velocity_damping: 3.0,
            affine_damping: 5.0,
        },
    ] {
        for poisson in [0.25, -0.125] {
            let mut checkpoint = single();
            checkpoint.cfg.youngs = 128.0;
            checkpoint.cfg.poisson = poisson;
            checkpoint.cfg.rheology = material;
            set(&mut checkpoint, 6, &[-0.25]);
            let h = match material {
                MpmRheology::Sand { .. } => (-0.25f64).exp(),
                MpmRheology::Snow { .. } => 1.75f64.exp(),
                _ => 1.0,
            };
            let mu = 128.0 / (2.0 * (1.0 + f64::from(poisson)));
            let lambda = 128.0 * f64::from(poisson)
                / ((1.0 + f64::from(poisson)) * (1.0 - 2.0 * f64::from(poisson)));
            let (u, _, v) = molla_math::dense3::svd3(f);
            let cof = DMat3::from_cols(
                f.y_axis.cross(f.z_axis),
                f.z_axis.cross(f.x_axis),
                f.x_axis.cross(f.y_axis),
            );
            let pk1 = 3.0
                * h
                * (2.0 * mu * (f - u * v.transpose()) + lambda * (f.determinant() - 1.0) * cof);
            let epsilon = 2f64.powi(-15);
            for index in 0..9 {
                let mut high = f.to_cols_array();
                high[index] += epsilon;
                set_matrix(&mut checkpoint, 3, DMat3::from_cols_array(&high));
                let plus = checkpoint
                    .energy_at_height(0.0)
                    .unwrap()
                    .elastic_potential
                    .unwrap();
                let mut low = f.to_cols_array();
                low[index] -= epsilon;
                set_matrix(&mut checkpoint, 3, DMat3::from_cols_array(&low));
                let minus = checkpoint
                    .energy_at_height(0.0)
                    .unwrap()
                    .elastic_potential
                    .unwrap();
                let derivative = (plus - minus) / (2.0 * epsilon);
                let expected = pk1.to_cols_array()[index];
                assert!(
                    (derivative - expected).abs() < 1e-6 * expected.abs().max(1.0),
                    "{material:?} nu={poisson} entry={index} {derivative} != {expected}"
                );
            }
        }
    }
}

#[test]
fn elastic_energy_uses_material_ids_volumes_and_bounded_hardening() {
    let mut checkpoint = super::super::tests::fixture(3);
    let f = [
        1.125, 0.0, 0.0, 0.0, 0.0, 0.9375, 0.0, 0.0, 0.0, 0.0, 1.0625, 0.0,
    ];
    set(&mut checkpoint, 3, &f.repeat(3));
    set(&mut checkpoint, 4, &[1.0, 2.0, 3.0]);
    set(&mut checkpoint, 6, &[0.5, 0.5, 0.5]);
    let mu = f64::from(checkpoint.cfg.youngs) / (2.0 * (1.0 + f64::from(checkpoint.cfg.poisson)));
    let lambda = f64::from(checkpoint.cfg.youngs) * f64::from(checkpoint.cfg.poisson)
        / ((1.0 + f64::from(checkpoint.cfg.poisson))
            * (1.0 - 2.0 * f64::from(checkpoint.cfg.poisson)));
    let density = mu * (0.125f64.powi(2) + 0.0625f64.powi(2) * 2.0)
        + 0.5 * lambda * (1.125f64 * 0.9375 * 1.0625 - 1.0).powi(2);
    let energy = checkpoint
        .energy_at_height(0.0)
        .unwrap()
        .elastic_potential
        .unwrap();
    assert!((energy - density * (0.5f64.exp() + 2.0 * (-3.0f64).exp() + 3.0)).abs() < 1e-11);
    checkpoint.materials.clear();
    checkpoint.cfg.rheology = MpmRheology::Sand {
        friction_angle: 0.5,
    };
    set(&mut checkpoint, 7, &[0.0; 3]);
    set(&mut checkpoint, 6, &[100.0, -100.0, 0.0]);
    let energy = checkpoint
        .energy_at_height(0.0)
        .unwrap()
        .elastic_potential
        .unwrap();
    assert!((energy - density * (3.0f64.exp() + 2.0 * (-3.0f64).exp() + 3.0)).abs() < 1e-10);
}

#[test]
fn elastic_scope_and_invalid_energy_inputs_are_not_silently_zero() {
    let mut checkpoint = single();
    for x in [-1.0, 0.0, 0.05, 10.5] {
        set_matrix(
            &mut checkpoint,
            3,
            DMat3::from_diagonal(DVec3::new(x, 1.0, 1.0)),
        );
        let energy = checkpoint.energy_at_height(0.0).unwrap();
        assert_eq!(energy.elastic_unavailable_particles, 1);
        assert_eq!(energy.elastic_potential, None);
        assert_eq!(energy.mechanical_energy(), None);
        assert_eq!(energy.translational_kinetic, 25.0);
    }
    set_matrix(
        &mut checkpoint,
        3,
        DMat3::from_cols(DVec3::Y, -DVec3::X, DVec3::Z),
    );
    assert_eq!(
        checkpoint.energy_at_height(0.0).unwrap().elastic_potential,
        Some(0.0)
    );
    assert!(checkpoint.energy_at_height(f64::NAN).is_err());
    assert!(checkpoint.energy_at_height(f64::INFINITY).is_err());
    checkpoint.cfg.gravity_y = f32::MAX;
    assert!(checkpoint.energy_at_height(f64::MAX).is_err());
    set(&mut checkpoint, 5, &[-1.0]);
    assert!(checkpoint.energy_at_height(0.0).is_err());
    let empty = super::super::tests::fixture(0)
        .energy_at_height(0.0)
        .unwrap();
    assert_eq!(empty.mechanical_energy(), Some(0.0));
}

#[test]
#[ignore = "requires completed native captures; CPU only"]
fn captured_native_energy_includes_affine_and_elastic_terms() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_SOIL_ENERGY_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("MOLLA_SOIL_ENERGY_OUTPUT").unwrap());
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap()
    };
    let fixture = read("fixture.json");
    let run = read("result.json");
    assert_eq!(fixture["fixture"], "AllContacts");
    assert_eq!(run["fixture"], "AllContacts");
    assert_eq!(run["control_fault_exact_rollback_and_retry"], true);
    let frames = run["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 4);
    let velocity = fixture["initial_velocity"].as_array().unwrap();
    assert_eq!(velocity.len(), 3);
    let speed_squared = velocity
        .iter()
        .map(|v| v.as_f64().unwrap().powi(2))
        .sum::<f64>();
    let mut records = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(frame["frame"].as_u64(), Some(i as u64 + 1));
        let state = MpmSoilCheckpoint::from_bytes(
            &std::fs::read(root.join(format!("soil-{}.bin", i + 1))).unwrap(),
            65,
        )
        .unwrap();
        assert_eq!(state.count, 65);
        assert_eq!(state.cfg.gravity_y, 0.0);
        let energy = state.energy_at_height(0.0).unwrap();
        let mass = state.momenta_about([0.0; 3]).unwrap().mass;
        assert_eq!(mass, 65.0 * f64::from(state.cfg.particle_mass));
        let initial = 0.5 * mass * speed_squared;
        let previous = frame["energy"].as_f64().unwrap();
        assert_eq!(energy.elastic_unavailable_particles, 0);
        let elastic = energy.elastic_potential.unwrap();
        let corrected = previous + energy.affine_kinetic + elastic;
        let machine = previous - energy.translational_kinetic;
        assert!(previous.is_finite() && corrected.is_finite() && initial.is_finite());
        assert!(machine >= -1e-12);
        assert!(energy.affine_kinetic > 0.0 && elastic >= 0.0);
        assert!(
            corrected <= initial + 1e-4,
            "frame={} energy={corrected} initial={initial}",
            i + 1
        );
        eprintln!(
            "energy frame={} old={previous:.12e} affine={:.12e} elastic={elastic:.12e} corrected={corrected:.12e} initial={initial:.12e}",
            i + 1,
            energy.affine_kinetic
        );
        records.push(serde_json::json!({"frame":i+1,"initial":initial,"previous_translational_gate_energy":previous,
            "machine_kinetic":machine,"soil_translational_kinetic":energy.translational_kinetic,
            "soil_affine_kinetic":energy.affine_kinetic,"soil_elastic":elastic,"corrected":corrected}));
    }
    let mut file = std::fs::File::create_new(output).unwrap();
    std::io::Write::write_all(&mut file,&serde_json::to_vec_pretty(&serde_json::json!({
        "frames":records,"absolute_tolerance":1e-4,"acceptance":false,"physically_calibrated":false,
        "scope":"unforced zero-gravity all-contact fixture; initial-energy upper bound, not dissipative-work accounting",
    })).unwrap()).unwrap();
}
