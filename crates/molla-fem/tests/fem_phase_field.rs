use molla_fem::{
    FemMesh, PhaseFieldFractureParams, assemble_phase_field_fracture_system,
    phase_field_degradation, phase_field_fracture_energy_density,
    solve_phase_field_fracture_jacobi, update_phase_field_history,
};

#[test]
fn degradation_and_history_are_irreversible() {
    assert!((phase_field_degradation(0.0, 1.0e-6) - 1.000001).abs() < 1.0e-12);
    assert!((phase_field_degradation(1.0, 1.0e-6) - 1.0e-6).abs() < 1.0e-12);
    assert_eq!(update_phase_field_history(2.0, 1.0), 2.0);
    assert_eq!(update_phase_field_history(2.0, 3.0), 3.0);
    assert_eq!(update_phase_field_history(2.0, -5.0), 2.0);
}

#[test]
fn at2_energy_increases_with_damage_gradient() {
    let params = PhaseFieldFractureParams {
        critical_energy_release_rate: 2.0,
        length_scale: 0.25,
        residual_stiffness: 1.0e-8,
    };
    let flat = phase_field_fracture_energy_density(0.4, 0.0, 1.5, params);
    let steep = phase_field_fracture_energy_density(0.4, 3.0, 1.5, params);
    assert!(steep > flat);
}

#[test]
fn phase_field_system_is_symmetric_with_positive_diagonal() {
    let mesh = FemMesh::tri_grid_2d(1, 1, 1.0, 1.0);
    let history = vec![0.2; mesh.vert_count()];
    let params = PhaseFieldFractureParams::default();
    let (lhs, rhs) = assemble_phase_field_fracture_system(&mesh, &history, params).unwrap();

    assert_eq!(lhs.nrows(), mesh.vert_count());
    assert!(rhs.iter().all(|v| *v > 0.0));
    for row in 0..lhs.nrows() {
        let mut diag = 0.0;
        for slot in lhs.row_offsets[row]..lhs.row_offsets[row + 1] {
            if lhs.col_idx[slot] == row {
                diag += lhs.values[slot];
            }
        }
        assert!(diag > 0.0, "row {row} has non-positive diagonal {diag}");
    }

    for row in 0..lhs.nrows() {
        for slot in lhs.row_offsets[row]..lhs.row_offsets[row + 1] {
            let col = lhs.col_idx[slot];
            let a = lhs.values[slot];
            let mut b = 0.0;
            for slot_t in lhs.row_offsets[col]..lhs.row_offsets[col + 1] {
                if lhs.col_idx[slot_t] == row {
                    b = lhs.values[slot_t];
                    break;
                }
            }
            assert!(
                (a - b).abs() < 1.0e-12,
                "lhs is not symmetric at ({row},{col})"
            );
        }
    }
}

#[test]
fn jacobi_damage_update_is_bounded_and_monotone() {
    let mesh = FemMesh::tri_grid_2d(2, 2, 1.0, 1.0);
    let mut damage_old = vec![0.0; mesh.vert_count()];
    damage_old[0] = 0.25;
    let history_low = vec![0.1; mesh.vert_count()];
    let history_high = vec![5.0; mesh.vert_count()];
    let params = PhaseFieldFractureParams {
        critical_energy_release_rate: 1.0,
        length_scale: 0.2,
        residual_stiffness: 1.0e-8,
    };

    let low =
        solve_phase_field_fracture_jacobi(&mesh, &damage_old, &history_low, params, 200, 1.0e-10)
            .unwrap();
    let high =
        solve_phase_field_fracture_jacobi(&mesh, &damage_old, &history_high, params, 200, 1.0e-10)
            .unwrap();

    assert!(
        low.damage
            .iter()
            .zip(&damage_old)
            .all(|(d, old)| *d >= *old)
    );
    assert!(
        high.damage
            .iter()
            .zip(&low.damage)
            .all(|(hi, lo)| *hi >= *lo)
    );
    assert!(high.damage.iter().all(|d| (0.0..=1.0).contains(d)));
    assert!(high.residual_inf < 1.0e-6);
}

#[test]
fn localized_history_drives_localized_irreversible_crack_growth() {
    let mesh = FemMesh::tri_grid_2d(6, 2, 1.0, 0.3);
    let damage_old = vec![0.05; mesh.vert_count()];
    let mut history = vec![0.05; mesh.vert_count()];

    // Seed a high tensile-history band near the centerline. The AT2 update
    // should grow damage most strongly there while diffusion keeps the field
    // finite and smooth.
    for (i, p) in mesh.vertices.iter().enumerate() {
        if (p.x - 0.5).abs() <= 0.12 {
            history[i] = 8.0;
        }
    }

    let params = PhaseFieldFractureParams {
        critical_energy_release_rate: 1.0,
        length_scale: 0.10,
        residual_stiffness: 1.0e-8,
    };
    let result =
        solve_phase_field_fracture_jacobi(&mesh, &damage_old, &history, params, 500, 1.0e-10)
            .unwrap();

    assert!(result.residual_inf < 1.0e-6);
    assert!(result.damage.iter().all(|d| (0.0..=1.0).contains(d)));
    assert!(
        result
            .damage
            .iter()
            .zip(&damage_old)
            .all(|(new, old)| *new >= *old),
        "phase-field damage must be irreversible"
    );

    let mut crack_sum = 0.0;
    let mut crack_n = 0usize;
    let mut far_sum = 0.0;
    let mut far_n = 0usize;
    for (i, p) in mesh.vertices.iter().enumerate() {
        if (p.x - 0.5).abs() <= 0.12 {
            crack_sum += result.damage[i];
            crack_n += 1;
        } else if p.x < 0.20 || p.x > 0.80 {
            far_sum += result.damage[i];
            far_n += 1;
        }
    }
    let crack_avg = crack_sum / crack_n as f64;
    let far_avg = far_sum / far_n as f64;
    eprintln!("phase-field localized crack: center avg d={crack_avg:.4}, far avg d={far_avg:.4}");
    assert!(
        crack_avg > far_avg + 0.15,
        "localized high history did not produce localized crack growth"
    );
}
