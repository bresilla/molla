use super::*;

fn component_residual(matrix: &[f64], vector: &[f64], rhs: &[f64]) -> f64 {
    let n = vector.len();
    assert_eq!(matrix.len(), n * n);
    assert_eq!(rhs.len(), n);
    assert!(
        matrix
            .iter()
            .chain(vector)
            .chain(rhs)
            .all(|x| x.is_finite())
    );
    (0..n)
        .map(|i| {
            let products = (0..n).map(|j| matrix[i * n + j] * vector[j]);
            let scale = rhs[i].abs() + products.clone().map(f64::abs).sum::<f64>();
            (products.sum::<f64>() - rhs[i]).abs() / if scale == 0.0 { 1.0 } else { scale }
        })
        .fold(0.0, f64::max)
}

#[test]
fn machine_input_reference_residual_contract() {
    assert_eq!(
        component_residual(&[2.0, 0.0, 0.0, 4.0], &[3.0, 2.0], &[6.0, 8.0]),
        0.0
    );
    assert_eq!(component_residual(&[0.0], &[0.0], &[0.0]), 0.0);
    assert!(component_residual(&[2.0], &[3.01], &[6.0]) > 1e-6);
    assert!(std::panic::catch_unwind(|| component_residual(&[f64::NAN], &[0.0], &[0.0])).is_err());
}

#[test]
#[ignore = "requires frozen first-substep tractor packets; CPU only"]
fn machine_input_reference_first_substep() {
    use crate::math::{SpatialVector, Vec3};
    let (scene, _) = crate::soil_tractor::build_soil_tractor_scene().unwrap();
    let model = scene.model();
    let state = scene.state();
    assert_eq!((model.body_count, model.joint_dof_total), (6, 10));
    assert!(state.joint_qd.host().unwrap().iter().all(|q| *q == 0.0));
    let mass = crate::crba::mass_matrix(model, state, 0).unwrap();
    let mut assembled = vec![0.0; 100];
    for body in 0..6 {
        let jac =
            crate::jacobian::jacobian(model, state, 0, body, model.body_com.host().unwrap()[body])
                .unwrap();
        let rotation = crate::math::Mat3::from_quat(state.body_q.host().unwrap()[body].rotation);
        let inertia = rotation * model.body_inertia.host().unwrap()[body] * rotation.transpose();
        let linear = |i| Vec3::new(jac[i], jac[10 + i], jac[20 + i]);
        let angular = |i| Vec3::new(jac[30 + i], jac[40 + i], jac[50 + i]);
        for i in 0..10 {
            for j in 0..10 {
                assembled[10 * i + j] += model.body_mass.host().unwrap()[body]
                    * linear(i).dot(linear(j))
                    + angular(i).dot(inertia * angular(j));
            }
        }
    }
    let oracle_error = mass
        .iter()
        .zip(&assembled)
        .map(|(a, b)| (a - b).abs() / (1.0 + a.abs()))
        .fold(0.0, f64::max);
    assert!(
        oracle_error <= 1e-12,
        "independent mass assembly: {oracle_error}"
    );
    let bias = crate::rnea::inverse_dynamics_wrenches(
        model,
        state,
        0,
        &[0.0; 10],
        &[SpatialVector::ZERO; 6],
    )
    .unwrap()
    .efforts;
    let rhs: Vec<_> = bias
        .iter()
        .enumerate()
        .map(|(i, b)| if i >= 6 { -300.0 - b } else { -b })
        .collect();
    let acceleration = solve(&mass, &rhs);
    let dt = f64::from((1.0f64 / 60.0 / 34.0) as f32);
    let expected_rate: Vec<_> = acceleration.iter().map(|a| a * dt).collect();
    let mut reports = Vec::new();
    for key in [
        "MOLLA_INPUT_REFERENCE_FEATHERSTONE",
        "MOLLA_INPUT_REFERENCE_NATIVE",
    ] {
        let dir = std::path::PathBuf::from(std::env::var_os(key).unwrap());
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("capture.json")).unwrap()).unwrap();
        let bytes = std::fs::read(dir.join("newton.bin")).unwrap();
        assert_eq!(metadata["blake3"], blake3::hash(&bytes).to_hex().as_str());
        assert_eq!(metadata["schema"], 1);
        assert_eq!(metadata["byte_order"], "little");
        assert_eq!(metadata["stage"], "newton-success");
        assert_eq!(metadata["bytes"], bytes.len());
        assert_eq!(bytes.len(), 128 + 400 + 80 + 448 * 4096);
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(
            (words[0], words[1], words[2], words[7], words[8], words[16]),
            (1, 10, 4096, 31, 1, 0)
        );
        let floats: Vec<_> = words
            .iter()
            .map(|x| f64::from(f32::from_bits(*x)))
            .collect();
        let captured_mass = &floats[32..132];
        let rates: Vec<_> = floats[132..152]
            .chunks_exact(2)
            .map(|x| x[0] + x[1])
            .collect();
        assert!(captured_mass.iter().chain(&rates).all(|x| x.is_finite()));
        let mass_scale = 1.0 + mass.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let mass_global_error = captured_mass
            .iter()
            .zip(&mass)
            .map(|(a, b)| (a - b).abs() / mass_scale)
            .fold(0.0, f64::max);
        let worst_mass_entry = captured_mass
            .iter()
            .zip(&mass)
            .enumerate()
            .max_by(|(_, (a, b)), (_, (c, d))| {
                ((**a - **b).abs() / (1.0 + b.abs()))
                    .total_cmp(&((**c - **d).abs() / (1.0 + d.abs())))
            })
            .unwrap()
            .0;
        let matrix_error = captured_mass
            .iter()
            .zip(&mass)
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0, f64::max);
        let rate_error = rates
            .iter()
            .zip(&expected_rate)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let mut response_error = 0.0f64;
        let mut active = 0;
        for block in floats[152..].chunks_exact(112) {
            if block[83] == 0.0 {
                continue;
            }
            assert!(block[83].is_finite() && block[83] > 0.0);
            active += 1;
            for axis in 0..3 {
                let row: Vec<_> = (0..10).map(|i| block[4 * i + axis]).collect();
                let response: Vec<_> = (0..10).map(|i| block[40 + 4 * i + axis]).collect();
                response_error =
                    response_error.max(component_residual(captured_mass, &response, &row));
            }
        }
        assert_eq!(active, words[26]);
        assert!(active > 0);
        reports.push(serde_json::json!({"route":key, "active_contacts":active,
            "mass_component_error":matrix_error, "free_rate_absolute_error":rate_error,
            "mass_global_error":mass_global_error,
            "worst_mass_entry":{"row":worst_mass_entry/10,"column":worst_mass_entry%10,
                "captured":captured_mass[worst_mass_entry],"reference":mass[worst_mass_entry]},
            "strict_component_diagnostic_pass":matrix_error <= 1e-6,
            "response_component_residual":response_error}));
    }
    let result = serde_json::json!({"cpu_mass_oracle_agreement":oracle_error, "reports":reports,
        "scope":"initial unrounded CPU scene reference; historical Featherstone and current native first-substep inputs; response checks use captured rows and mass, not independent contact geometry; no trajectory or performance acceptance"});
    std::fs::write(
        std::env::var_os("MOLLA_INPUT_REFERENCE_REPORT").unwrap(),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!("{result}");
    for report in &reports {
        assert!(report["mass_global_error"].as_f64().unwrap() <= 1e-6);
        assert!(report["mass_component_error"].as_f64().unwrap() <= 1e-6);
        assert!(report["free_rate_absolute_error"].as_f64().unwrap() <= 1e-5);
        assert!(report["response_component_residual"].as_f64().unwrap() <= 1e-6);
    }
}
