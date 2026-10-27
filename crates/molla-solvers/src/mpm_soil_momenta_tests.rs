use super::*;
use glam::DMat3;

fn set(checkpoint: &mut MpmSoilCheckpoint, stream: usize, values: &[f32]) {
    let offset = STRIDES[..stream].iter().sum::<usize>() * checkpoint.count as usize;
    assert_eq!(values.len(), checkpoint.count as usize * STRIDES[stream]);
    for (word, value) in checkpoint.words[offset..].iter_mut().zip(values) {
        *word = value.to_bits();
    }
}

#[test]
fn quadratic_spin_and_symmetric_strain_are_distinct() {
    let mut checkpoint = super::super::tests::fixture(1);
    checkpoint.cfg.cell_size = 0.25;
    set(&mut checkpoint, 0, &[0.0; 4]);
    set(&mut checkpoint, 1, &[0.0; 4]);
    set(&mut checkpoint, 5, &[2.0]);
    set(
        &mut checkpoint,
        2,
        &[
            0.0, 3.0, -2.0, 99.0, -3.0, 0.0, 1.0, 99.0, 2.0, -1.0, 0.0, 99.0,
        ],
    );
    let spin = checkpoint.momenta_about([0.0; 3]).unwrap();
    assert_eq!(spin.mass, 2.0);
    assert_eq!(spin.linear_momentum, [0.0; 3]);
    assert_eq!(spin.angular_momentum, [0.0625, 0.125, 0.1875]);
    assert_eq!(spin.angular_momentum, spin.affine_angular_momentum);
    set(
        &mut checkpoint,
        2,
        &[
            1.0, 2.0, 3.0, 99.0, 2.0, 4.0, 5.0, 99.0, 3.0, 5.0, 6.0, 99.0,
        ],
    );
    assert_eq!(
        checkpoint.momenta_about([0.0; 3]).unwrap().angular_momentum,
        [0.0; 3]
    );
}

#[test]
fn momenta_match_independent_quadratic_grid_quadrature() {
    let checkpoint = super::super::tests::fixture(129);
    let bytes = checkpoint.to_bytes();
    let origin = DVec3::new(0.3, -0.4, 0.2);
    let result = checkpoint.momenta_about(origin.to_array()).unwrap();
    let h = f64::from(checkpoint.cfg.cell_size);
    let grid_origin = DVec3::from_array(checkpoint.cfg.grid_origin.map(f64::from));
    let mut grid_linear = DVec3::ZERO;
    let mut grid_angular = DVec3::ZERO;
    let mut grid_mass = 0.0;
    for p in 0..129 {
        let scalar = |s, k| f64::from(f32::from_bits(checkpoint.stream(s)[p * STRIDES[s] + k]));
        let vector = |s, k| DVec3::new(scalar(s, k), scalar(s, k + 1), scalar(s, k + 2));
        let position = vector(0, 0);
        let velocity = vector(1, 0);
        let c = DMat3::from_cols(vector(2, 0), vector(2, 4), vector(2, 8));
        let mass = scalar(5, 0);
        let xp = (position - grid_origin) / h;
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
                    let node = grid_origin + h * (base + DVec3::new(i as f64, j as f64, k as f64));
                    let m = mass * w[0][i] * w[1][j] * w[2][k];
                    let momentum = m * (velocity + c * (node - position));
                    grid_mass += m;
                    grid_linear += momentum;
                    grid_angular += (node - origin).cross(momentum);
                }
            }
        }
    }
    assert!((result.mass - grid_mass).abs() < 1e-12);
    assert!((DVec3::from_array(result.linear_momentum) - grid_linear).length() < 1e-12);
    assert!((DVec3::from_array(result.angular_momentum) - grid_angular).length() < 1e-12);
    assert!(DVec3::from_array(result.affine_angular_momentum).length() > 1e-8);
    assert_eq!(checkpoint.to_bytes(), bytes);
}

#[test]
fn origin_shift_empty_and_invalid_momenta() {
    let mut checkpoint = super::super::tests::fixture(1);
    let a = checkpoint.momenta_about([0.0; 3]).unwrap();
    let shift = DVec3::new(1.0, -2.0, 3.0);
    let b = checkpoint.momenta_about(shift.to_array()).unwrap();
    assert_eq!(a.linear_momentum, b.linear_momentum);
    assert_eq!(a.affine_angular_momentum, b.affine_angular_momentum);
    assert!(
        (DVec3::from_array(b.angular_momentum) - DVec3::from_array(a.angular_momentum)
            + shift.cross(DVec3::from_array(a.linear_momentum)))
        .length()
            < 1e-14
    );
    let empty = super::super::tests::fixture(0)
        .momenta_about([0.0; 3])
        .unwrap();
    assert_eq!(empty.mass, 0.0);
    assert_eq!(empty.angular_momentum, [0.0; 3]);
    for x in [f64::NAN, f64::INFINITY] {
        assert!(checkpoint.momenta_about([x, 0.0, 0.0]).is_err());
    }
    set(&mut checkpoint, 1, &[0.0, f32::MAX, 0.0, 0.0]);
    assert!(checkpoint.momenta_about([f64::MAX, 0.0, 0.0]).is_err());
    set(&mut checkpoint, 5, &[-1.0]);
    assert!(checkpoint.momenta_about([0.0; 3]).is_err());
    set(&mut checkpoint, 5, &[1.0]);
    set(&mut checkpoint, 2, &[f32::NAN; 12]);
    assert!(checkpoint.momenta_about([0.0; 3]).is_err());
}

#[test]
#[ignore = "requires completed native all-contact captures; CPU only"]
fn captured_native_world_conserves_combined_apic_angular_momentum() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_NATIVE_ANGULAR_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("MOLLA_NATIVE_ANGULAR_OUTPUT").unwrap());
    let read_json = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap()
    };
    let fixture = read_json("fixture.json");
    let result = read_json("result.json");
    assert_eq!(fixture["fixture"], "AllContacts");
    assert_eq!(result["fixture"], "AllContacts");
    assert_eq!(result["world_origin_angular_gate"], true);
    assert_eq!(result["control_fault_exact_rollback_and_retry"], true);
    assert_eq!(result["acceptance"], false);
    assert_eq!(
        result["body_impulse_magnitudes"].as_array().unwrap().len(),
        5
    );
    assert!(
        result["body_impulse_magnitudes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v.as_f64().unwrap() > 1e-6)
    );
    let vector = |v: &serde_json::Value| -> DVec3 {
        let values = v.as_array().unwrap();
        assert_eq!(values.len(), 3);
        let result = DVec3::new(
            values[0].as_f64().unwrap(),
            values[1].as_f64().unwrap(),
            values[2].as_f64().unwrap(),
        );
        assert!(result.is_finite());
        result
    };
    let velocity = vector(&fixture["initial_velocity"]);
    let positions = fixture["positions"].as_array().unwrap();
    assert_eq!(positions.len(), 65);
    assert_eq!(result["frames"].as_array().unwrap().len(), 4);
    let mut records = Vec::new();
    let mut previous_machine = DVec3::ZERO;
    for frame in 1..=4 {
        let checkpoint = MpmSoilCheckpoint::from_bytes(
            &std::fs::read(root.join(format!("soil-{frame}.bin"))).unwrap(),
            65,
        )
        .unwrap();
        assert_eq!(checkpoint.count, 65);
        assert_eq!(checkpoint.cfg.gravity_y, 0.0);
        let moments = checkpoint.momenta_about([0.0; 3]).unwrap();
        let mass = f64::from(checkpoint.cfg.particle_mass);
        assert_eq!(moments.mass, 65.0 * mass);
        let initial: DVec3 = positions
            .iter()
            .map(|p| vector(p).cross(mass * velocity))
            .sum();
        let angular = read_json(&format!("angular-{frame}.json"));
        assert_eq!(angular["bodies"].as_array().unwrap().len(), 5);
        assert_eq!(
            result["frames"][frame - 1]["frame"].as_u64(),
            Some(frame as u64)
        );
        let before = vector(&angular["before"]);
        let after = vector(&angular["after"]);
        assert_eq!(before, previous_machine);
        previous_machine = after;
        let impulse: DVec3 = angular["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| vector(&b["world_origin_angular"]))
            .sum();
        let machine_error = (after - before - impulse).length();
        assert!(machine_error < 2e-5);
        let combined = after + DVec3::from_array(moments.angular_momentum);
        let combined_error = (combined - initial).length();
        let omitted_affine_error =
            (combined - DVec3::from_array(moments.affine_angular_momentum) - initial).length();
        eprintln!(
            "combined APIC angular frame={frame} error={combined_error:e} without_affine={omitted_affine_error:e}"
        );
        assert!(
            combined_error < 2e-5,
            "combined angular frame={frame} residual={combined_error:e}"
        );
        records.push(serde_json::json!({"frame":frame,"mass":moments.mass,
            "machine_angular":after.to_array(),"soil_angular":moments.angular_momentum,
            "soil_affine_angular":moments.affine_angular_momentum,"initial_angular":initial.to_array(),
            "combined_angular_error":combined_error,"machine_angular_error":machine_error,
            "omitted_affine_error":omitted_affine_error}));
    }
    let mut file = std::fs::File::create_new(output).unwrap();
    std::io::Write::write_all(&mut file, &serde_json::to_vec_pretty(&serde_json::json!({
        "frames":records,"tolerance":2e-5,"scope":"zero-gravity 65-particle all-contact native fixture",
        "acceptance":false,"physically_calibrated":false,
    })).unwrap()).unwrap();
}
