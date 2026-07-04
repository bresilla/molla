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

/// C3 — coupled displacement ↔ damage staggered solve.
///
/// 1. Homogeneous AT2 identity: an unnotched patch under uniform
///    biaxial-free stretch (linear Dirichlet field on every boundary
///    node) develops HOMOGENEOUS damage matching the closed form
///    d* = 2H/(G_c/ℓ + 2H) of the AT2 screened Poisson with Δd = 0.
/// 2. Notch localization + softening: a strip with a pre-seeded
///    history notch pulled in tension over load steps — damage
///    localizes at the notch (max d at the notch row ≫ far field) and
///    grows monotonically; the degraded strain energy per unit
///    stretch² SOFTENS as the crack develops.
#[test]
fn coupled_staggered_solve_matches_at2_homogeneous_identity() {
    use molla_fem::{PhaseFieldCoupledConfig, solve_phase_field_coupled_step};
    let mesh = FemMesh::tri_grid_2d(6, 6, 1.0, 1.0);
    let params = PhaseFieldFractureParams {
        critical_energy_release_rate: 5.0,
        length_scale: 0.2,
        residual_stiffness: 1.0e-6,
    };
    let cfg = PhaseFieldCoupledConfig::default();
    let eps = 0.04;
    let nu = 0.3;
    // Linear field u = (ε·x, −ν·ε·y) on EVERY node ⇒ truly homogeneous
    // strain (constrains the whole field so damage feedback cannot
    // redistribute it — the identity needs uniform ε).
    let mut dirichlet = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        dirichlet.push((2 * i, eps * p.x));
        dirichlet.push((2 * i + 1, -nu * eps * p.y));
    }
    let n = mesh.vert_count();
    let mut history = vec![0.0; n];
    let mut damage = vec![0.0; n];
    solve_phase_field_coupled_step(&mesh, params, cfg, &dirichlet, &mut history, &mut damage)
        .expect("coupled step");

    // Closed form from the converged history.
    let h_ref = history[n / 2];
    let d_expect =
        2.0 * h_ref / (params.critical_energy_release_rate / params.length_scale + 2.0 * h_ref);
    let d_min = damage.iter().cloned().fold(f64::INFINITY, f64::min);
    let d_max = damage.iter().cloned().fold(0.0, f64::max);
    eprintln!("homogeneous AT2: d ∈ [{d_min:.5}, {d_max:.5}], closed form {d_expect:.5}");
    assert!(
        d_expect > 0.05,
        "load too weak to exercise damage: {d_expect}"
    );
    assert!(
        (d_max - d_min).abs() < 1.0e-6,
        "damage must be homogeneous: [{d_min}, {d_max}]"
    );
    assert!(
        ((d_max - d_expect) / d_expect).abs() < 1.0e-6,
        "AT2 closed form violated: {d_max} vs {d_expect}"
    );
}

#[test]
fn notch_localizes_damage_and_softens() {
    use molla_fem::{PhaseFieldCoupledConfig, solve_phase_field_coupled_step};
    let (nx, ny) = (16, 8);
    let mesh = FemMesh::tri_grid_2d(nx, ny, 2.0, 1.0);
    let params = PhaseFieldFractureParams {
        critical_energy_release_rate: 1.0,
        length_scale: 0.12,
        residual_stiffness: 1.0e-6,
    };
    let cfg = PhaseFieldCoupledConfig::default();
    let n = mesh.vert_count();
    let mut history = vec![0.0; n];
    let mut damage = vec![0.0; n];
    // Pre-seed a history notch: left half of the horizontal midline.
    for (i, p) in mesh.vertices.iter().enumerate() {
        if (p.y - 0.5).abs() < 1.0e-6 && p.x < 0.6 {
            history[i] = 40.0;
        }
    }

    let mut energies = Vec::new();
    let mut peaks = Vec::new();
    for step in 1..=4 {
        let stretch = 0.006 * step as f64;
        let mut dirichlet = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.y.abs() < 1.0e-9 {
                dirichlet.push((2 * i, 0.0));
                dirichlet.push((2 * i + 1, 0.0));
            } else if (p.y - 1.0).abs() < 1.0e-9 {
                dirichlet.push((2 * i, 0.0));
                dirichlet.push((2 * i + 1, stretch));
            }
        }
        let res = solve_phase_field_coupled_step(
            &mesh,
            params,
            cfg,
            &dirichlet,
            &mut history,
            &mut damage,
        )
        .expect("coupled step");
        // Normalized stiffness: energy / stretch² (softening metric).
        energies.push(res.strain_energy / (stretch * stretch));
        peaks.push(damage.iter().cloned().fold(0.0, f64::max));
    }

    // Localization: damage at the notch row far exceeds the far field.
    let mut d_notch = 0.0f64;
    let mut d_far = 0.0f64;
    for (i, p) in mesh.vertices.iter().enumerate() {
        if (p.y - 0.5).abs() < 1.0e-6 && p.x < 0.6 {
            d_notch = d_notch.max(damage[i]);
        }
        if p.x > 1.6 {
            d_far = d_far.max(damage[i]);
        }
    }
    eprintln!(
        "notch: d_notch {d_notch:.4}, d_far {d_far:.4}, peaks {peaks:?}, k_norm {energies:?}"
    );
    // Closed form for the H = 40 seed gives d = 0.906 BEFORE the
    // element-mean degradation on this coarse mesh — 0.8 still means
    // "strongly damaged"; the physics gates are localization,
    // monotonicity and softening below.
    assert!(d_notch > 0.8, "notch must be near-cracked: {d_notch:.4}");
    assert!(
        d_far < 0.5 * d_notch,
        "damage must localize: far {d_far:.4} vs notch {d_notch:.4}"
    );
    // Irreversibility: peak damage is monotone across load steps.
    for w in peaks.windows(2) {
        assert!(w[1] >= w[0] - 1.0e-12, "damage went backwards: {peaks:?}");
    }
    // Softening: the normalized stiffness drops as the crack grows.
    assert!(
        energies.last().unwrap() < &(0.9 * energies[0]),
        "no softening: {energies:?}"
    );
}
