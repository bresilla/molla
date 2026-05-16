//! CPU reduced-oracle tests for MPM material updates.
//!
//! GPU MPM owns the full particle/grid simulation. These tests cover
//! small material formulas in the default CPU lane so shader branches
//! have an inspectable correctness reference.

use molla_math::{Mat3, Vec3};
use molla_optim::{AdamConfig, AdamState};
use molla_solvers::implicit_mpm::bingham_project;
use molla_solvers::{
    ReducedMpmMaterial, ReducedMpmMaterialState, drucker_prager_alpha, reduced_material_update,
    sand_project_singular_values, snow_clamp_singular_values, viscous_damping_step,
};

fn frob_norm(m: Mat3) -> f64 {
    (m.x_axis.length_squared() + m.y_axis.length_squared() + m.z_axis.length_squared()).sqrt()
}

#[test]
fn sand_oracle_computes_shader_drucker_prager_alpha() {
    let phi = 30.0_f64.to_radians();
    let expected = (2.0_f64 / 3.0).sqrt() * (2.0 * phi.sin()) / (3.0 - phi.sin());
    assert!((drucker_prager_alpha(phi) - expected).abs() < 1.0e-15);
    assert_eq!(
        drucker_prager_alpha(-10.0_f64.to_radians()),
        drucker_prager_alpha(1.0_f64.to_radians())
    );
}

#[test]
fn sand_oracle_resets_tensile_volume_expansion() {
    let step = sand_project_singular_values(Vec3::new(1.2, 1.1, 1.0), 0.0, 1.0e5, 0.3, 0.5);

    assert_eq!(step.singular_values, Vec3::ONE);
    assert!(
        step.jp > 0.0,
        "tensile reset should accumulate plastic flow"
    );
}

#[test]
fn sand_oracle_keeps_elastic_state_inside_cone() {
    let singular_values = Vec3::splat(0.95);
    let step = sand_project_singular_values(singular_values, 0.2, 1.0e5, 0.3, 30f64.to_radians());

    assert_eq!(step.singular_values, singular_values);
    assert_eq!(step.jp, 0.2);
}

#[test]
fn sand_oracle_projects_shear_outside_yield_cone() {
    let singular_values = Vec3::new(0.50, 0.90, 1.10);
    let step = sand_project_singular_values(singular_values, 0.0, 1.0e5, 0.3, 5.0_f64.to_radians());

    assert!(
        (step.singular_values - singular_values).length() > 1.0e-3,
        "outside-yield state was not projected"
    );
    assert!(step.jp > 0.0, "projection should accumulate plastic flow");
    assert!(
        step.singular_values.x > singular_values.x,
        "compression singular value should move toward the yield cone"
    );
}

#[test]
fn sand_friction_angle_monotonically_preserves_shear_strength() {
    // Reduced angle-of-repose oracle: for the same compressed/sheared
    // state, a larger internal-friction angle expands the Drucker-Prager
    // cone. The high-friction sand should therefore retain more
    // deviatoric/shear strain and accumulate less plastic flow than the
    // low-friction sand.
    let singular_values = Vec3::new(0.42, 0.82, 1.05);
    let low = sand_project_singular_values(singular_values, 0.0, 1.0e5, 0.3, 5.0_f64.to_radians());
    let high =
        sand_project_singular_values(singular_values, 0.0, 1.0e5, 0.3, 35.0_f64.to_radians());

    let mean = |s: Vec3| (s.x + s.y + s.z) / 3.0;
    let deviatoric_norm = |s: Vec3| (s - Vec3::splat(mean(s))).length();
    let input_dev = deviatoric_norm(singular_values);
    let low_dev = deviatoric_norm(low.singular_values);
    let high_dev = deviatoric_norm(high.singular_values);
    let low_correction = (low.singular_values - singular_values).length();
    let high_correction = (high.singular_values - singular_values).length();

    eprintln!(
        "sand friction-angle oracle: input_dev={input_dev:.4}, low_dev={low_dev:.4}, \
         high_dev={high_dev:.4}, low_correction={low_correction:.4}, \
         high_correction={high_correction:.4}, low_jp={:.4}, high_jp={:.4}",
        low.jp, high.jp,
    );

    assert!(
        high_dev > low_dev,
        "higher-friction sand should retain more deviatoric strain: low={low_dev}, high={high_dev}"
    );
    assert!(
        high_dev <= input_dev + 1.0e-12,
        "projection should not create more deviatoric strain than the input"
    );
    assert!(
        high_correction < low_correction,
        "higher-friction sand should need a smaller yield projection: low={low_correction}, high={high_correction}"
    );
    assert!(
        high.jp < low.jp,
        "higher-friction sand should accumulate less plastic flow: low_jp={}, high_jp={}",
        low.jp,
        high.jp
    );
}

#[test]
fn sand_oracle_bounds_plastic_flow_accumulator() {
    let step = sand_project_singular_values(Vec3::new(20.0, 20.0, 20.0), 4.9, 1.0e5, 0.3, 0.5);
    assert_eq!(step.jp, 5.0);
}

#[test]
fn viscous_oracle_damps_velocity_and_affine_by_expected_factors() {
    let velocity = Vec3::new(3.0, -4.0, 0.5);
    let affine = [
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(-2.0, 0.5, 1.5),
        Vec3::new(0.0, -1.0, 4.0),
    ];
    let dt = 0.25;
    let step = viscous_damping_step(velocity, affine, dt, 4.0, 8.0);

    let expected_v = (-1.0_f64).exp();
    let expected_c = (-2.0_f64).exp();
    assert!((step.velocity_factor - expected_v).abs() < 1.0e-15);
    assert!((step.affine_factor - expected_c).abs() < 1.0e-15);
    assert!((step.velocity - velocity * expected_v).length() < 1.0e-15);
    for (actual, input) in step.affine.iter().zip(affine.iter()) {
        assert!((*actual - *input * expected_c).length() < 1.0e-15);
    }
}

#[test]
fn viscous_oracle_clamps_negative_damping_and_dt_to_noop() {
    let velocity = Vec3::new(1.0, 2.0, 3.0);
    let affine = [Vec3::X, Vec3::Y, Vec3::Z];
    let step = viscous_damping_step(velocity, affine, -1.0, -4.0, -8.0);

    assert_eq!(step.velocity_factor, 1.0);
    assert_eq!(step.affine_factor, 1.0);
    assert_eq!(step.velocity, velocity);
    assert_eq!(step.affine, affine);
}

#[test]
fn viscous_oracle_repeated_steps_decay_energy_monotonically() {
    let velocity0 = Vec3::new(2.0, -1.0, 0.5);
    let affine0 = [
        Vec3::new(0.4, 0.1, -0.2),
        Vec3::new(-0.3, 0.5, 0.2),
        Vec3::new(0.1, -0.4, 0.6),
    ];
    let dt = 0.02;
    let velocity_damping = 3.0;
    let affine_damping = 5.0;
    let mut velocity = velocity0;
    let mut affine = affine0;
    let mut last_energy =
        velocity.length_squared() + affine.iter().map(|col| col.length_squared()).sum::<f64>();

    for step_idx in 1..=40 {
        let step = viscous_damping_step(velocity, affine, dt, velocity_damping, affine_damping);
        velocity = step.velocity;
        affine = step.affine;
        let energy =
            velocity.length_squared() + affine.iter().map(|col| col.length_squared()).sum::<f64>();
        assert!(
            energy < last_energy,
            "viscous reduced energy should strictly decay at step {step_idx}: {last_energy} -> {energy}"
        );
        last_energy = energy;
    }

    let one_shot = viscous_damping_step(
        velocity0,
        affine0,
        40.0 * dt,
        velocity_damping,
        affine_damping,
    );
    assert!(
        (velocity - one_shot.velocity).length() < 1.0e-14,
        "repeated velocity damping should compose exactly with one-shot damping"
    );
    for (actual, expected) in affine.iter().zip(one_shot.affine.iter()) {
        assert!(
            (*actual - *expected).length() < 1.0e-14,
            "repeated affine damping should compose exactly with one-shot damping"
        );
    }
}

#[test]
fn bingham_oracle_switches_between_plug_and_flow_regimes() {
    let d = Mat3::from_cols(
        Vec3::new(0.02, 0.004, 0.0),
        Vec3::new(0.004, -0.01, 0.002),
        Vec3::new(0.0, 0.002, -0.01),
    );
    let mu_b = 10.0;

    let below = d * 0.5;
    let below_norm = frob_norm(below);
    let (plugged, below_yield) = bingham_project(below, below_norm, mu_b, 1.0);
    assert!(below_yield, "below-yield state should become a rigid plug");
    assert!(
        frob_norm(plugged) < 1.0e-15,
        "plugged Bingham deviator should be zero, got {plugged:?}"
    );

    let above = d * 10.0;
    let above_norm = frob_norm(above);
    let tau_y = 1.0;
    let (flowing, is_plug) = bingham_project(above, above_norm, mu_b, tau_y);
    assert!(!is_plug, "above-yield state should remain flowing");
    let flowing_norm = frob_norm(flowing);
    let expected_norm = above_norm - tau_y / (2.0 * mu_b);
    assert!(
        (flowing_norm - expected_norm).abs() < 1.0e-12,
        "Bingham return map norm mismatch: got {flowing_norm}, expected {expected_norm}"
    );
    assert!(
        flowing_norm > 0.0 && flowing_norm < above_norm,
        "flowing Bingham projection should reduce but not zero strain rate"
    );

    let (low_yield, _) = bingham_project(above, above_norm, mu_b, 0.5);
    let (high_yield, _) = bingham_project(above, above_norm, mu_b, 2.0);
    assert!(
        frob_norm(high_yield) < frob_norm(low_yield),
        "larger yield stress should produce a stronger Bingham projection"
    );
}

#[test]
fn snow_oracle_clamps_singular_values_to_elastic_envelope() {
    let theta_c = 0.025;
    let theta_s = 0.0075;
    let step = snow_clamp_singular_values(Vec3::new(0.80, 1.20, 1.0), 0.0, theta_c, theta_s);

    assert!((step.singular_values.x - (1.0 - theta_c)).abs() < 1.0e-15);
    assert!((step.singular_values.y - (1.0 + theta_s)).abs() < 1.0e-15);
    assert_eq!(step.singular_values.z, 1.0);
    assert!(step.jp.is_finite());
}

#[test]
fn snow_oracle_tracks_log_plastic_volume_change() {
    let singular_values = Vec3::new(0.5, 1.5, 1.1);
    let old_jp = 0.25;
    let step = snow_clamp_singular_values(singular_values, old_jp, 0.025, 0.0075);

    let clamped = step.singular_values;
    let expected = old_jp
        + (singular_values.x * singular_values.y * singular_values.z)
            .max(1.0e-6)
            .ln()
        - (clamped.x * clamped.y * clamped.z).max(1.0e-6).ln();
    assert!((step.jp - expected.clamp(-5.0, 5.0)).abs() < 1.0e-15);
}

#[test]
fn snow_repeated_compression_accumulates_plastic_compaction_boundedly() {
    let theta_c = 0.025;
    let theta_s = 0.0075;
    let input = Vec3::new(0.82, 0.94, 1.06);
    let lo = 1.0 - theta_c;
    let hi = 1.0 + theta_s;
    let mut jp = 0.0;
    let mut previous_jp = jp;

    for step_idx in 1..=12 {
        let step = snow_clamp_singular_values(input, jp, theta_c, theta_s);
        assert!(
            (lo..=hi).contains(&step.singular_values.x)
                && (lo..=hi).contains(&step.singular_values.y)
                && (lo..=hi).contains(&step.singular_values.z),
            "snow step {step_idx} left the elastic envelope: {:?}",
            step.singular_values
        );
        assert!(
            step.jp <= previous_jp,
            "repeated compression/shear should accumulate compaction monotonically: prev={previous_jp}, next={}",
            step.jp
        );
        assert!(
            (-5.0..=5.0).contains(&step.jp),
            "snow plastic accumulator escaped bounds: {}",
            step.jp
        );
        previous_jp = step.jp;
        jp = step.jp;
    }

    eprintln!("snow repeated compaction: final jp={jp:.4}");
    assert!(
        jp < -0.5,
        "repeated compression should leave a measurable permanent compaction set, got jp={jp}"
    );
}

#[test]
fn snow_oracle_bounds_hardening_accumulator() {
    let hi = snow_clamp_singular_values(Vec3::new(20.0, 20.0, 20.0), 4.9, 0.025, 0.0075);
    let lo = snow_clamp_singular_values(Vec3::new(1.0e-3, 1.0e-3, 1.0e-3), -4.9, 0.025, 0.0075);

    assert_eq!(hi.jp, 5.0);
    assert_eq!(lo.jp, -5.0);
}

#[test]
fn reduced_multi_material_oracle_dispatches_distinct_branches() {
    let affine = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, -2.0, 0.0),
        Vec3::new(0.0, 0.0, 3.0),
    ];
    let states = [
        ReducedMpmMaterialState {
            singular_values: Vec3::new(0.50, 0.90, 1.10),
            velocity: Vec3::new(1.0, 0.0, 0.0),
            affine,
            jp: 0.0,
        },
        ReducedMpmMaterialState {
            singular_values: Vec3::new(0.80, 1.20, 1.0),
            velocity: Vec3::new(0.0, 2.0, 0.0),
            affine,
            jp: 0.25,
        },
        ReducedMpmMaterialState {
            singular_values: Vec3::new(1.0, 1.0, 1.0),
            velocity: Vec3::new(3.0, -4.0, 0.5),
            affine,
            jp: 0.5,
        },
    ];
    let materials = [
        ReducedMpmMaterial::Sand {
            youngs: 1.0e5,
            poisson: 0.3,
            friction_angle_rad: 5.0_f64.to_radians(),
        },
        ReducedMpmMaterial::Snow {
            theta_c: 0.025,
            theta_s: 0.0075,
        },
        ReducedMpmMaterial::Viscous {
            velocity_damping: 4.0,
            affine_damping: 8.0,
        },
    ];

    let out: Vec<_> = materials
        .iter()
        .zip(states.iter())
        .map(|(m, s)| reduced_material_update(*m, *s, 0.25))
        .collect();

    // Sand projects the shear state and accumulates plastic flow, but leaves
    // velocity/affine untouched in this reduced branch oracle.
    assert!((out[0].singular_values - states[0].singular_values).length() > 1.0e-3);
    assert!(out[0].jp > states[0].jp);
    assert_eq!(out[0].velocity, states[0].velocity);
    assert_eq!(out[0].affine, states[0].affine);

    // Snow clamps to its Stomakhin elastic envelope and updates `jp`, but does
    // not damp momentum.
    assert!(out[1].singular_values.x >= 1.0 - 0.025 - 1.0e-15);
    assert!(out[1].singular_values.y <= 1.0 + 0.0075 + 1.0e-15);
    assert_ne!(out[1].jp, states[1].jp);
    assert_eq!(out[1].velocity, states[1].velocity);

    // Viscous branch keeps deformation scalars intact and damps momentum.
    assert_eq!(out[2].singular_values, states[2].singular_values);
    assert_eq!(out[2].jp, states[2].jp);
    assert!(out[2].velocity.length() < states[2].velocity.length());
    assert!(out[2].affine[2].length() < states[2].affine[2].length());

    for (i, step) in out.iter().enumerate() {
        assert!(
            step.singular_values.x.is_finite()
                && step.singular_values.y.is_finite()
                && step.singular_values.z.is_finite()
                && step.velocity.x.is_finite()
                && step.velocity.y.is_finite()
                && step.velocity.z.is_finite()
                && step.jp.is_finite(),
            "material step {i} produced non-finite output: {step:?}"
        );
    }
}

#[test]
fn reduced_viscous_mpm_material_parameter_fit_recovers_damping() {
    fn loss_and_grad(
        theta: [f64; 2],
        velocity0: Vec3,
        affine0: [Vec3; 3],
        target_velocity: Vec3,
        target_affine: [Vec3; 3],
        dt: f64,
    ) -> (f64, [f64; 2]) {
        let velocity_damping = theta[0].exp();
        let affine_damping = theta[1].exp();
        let step = viscous_damping_step(velocity0, affine0, dt, velocity_damping, affine_damping);

        let dv = step.velocity - target_velocity;
        let mut loss = 0.5 * dv.length_squared();
        let grad_velocity = dv.dot(-dt * velocity_damping * step.velocity);

        let mut grad_affine = 0.0;
        for (actual, target) in step.affine.iter().zip(target_affine.iter()) {
            let da = *actual - *target;
            loss += 0.5 * da.length_squared();
            grad_affine += da.dot(-dt * affine_damping * *actual);
        }

        (loss, [grad_velocity, grad_affine])
    }

    let velocity0 = Vec3::new(2.0, -1.2, 0.4);
    let affine0 = [
        Vec3::new(0.7, -0.2, 0.1),
        Vec3::new(-0.3, 0.5, 0.2),
        Vec3::new(0.1, -0.4, 0.8),
    ];
    let true_velocity_damping = 1.7;
    let true_affine_damping = 3.4;
    let dt = 0.35;
    let target = viscous_damping_step(
        velocity0,
        affine0,
        dt,
        true_velocity_damping,
        true_affine_damping,
    );
    let target_velocity = target.velocity;
    let target_affine = target.affine;

    let theta_probe = [0.6_f64.ln(), 1.2_f64.ln()];
    let (_loss_probe, grad_probe) = loss_and_grad(
        theta_probe,
        velocity0,
        affine0,
        target_velocity,
        target_affine,
        dt,
    );
    for k in 0..2 {
        let eps = 1.0e-5;
        let mut plus = theta_probe;
        let mut minus = theta_probe;
        plus[k] += eps;
        minus[k] -= eps;
        let (loss_plus, _) =
            loss_and_grad(plus, velocity0, affine0, target_velocity, target_affine, dt);
        let (loss_minus, _) = loss_and_grad(
            minus,
            velocity0,
            affine0,
            target_velocity,
            target_affine,
            dt,
        );
        let fd = (loss_plus - loss_minus) / (2.0 * eps);
        assert!(
            (grad_probe[k] - fd).abs() < 1.0e-8,
            "viscous damping log-param grad {k} mismatch: analytic={} fd={} diff={}",
            grad_probe[k],
            fd,
            (grad_probe[k] - fd).abs()
        );
    }

    let mut theta = [0.2_f64.ln(), 0.5_f64.ln()];
    let (initial_loss, _) = loss_and_grad(
        theta,
        velocity0,
        affine0,
        target_velocity,
        target_affine,
        dt,
    );
    let mut adam = AdamState::new(2);
    let cfg = AdamConfig {
        lr: 0.08,
        ..AdamConfig::default()
    };
    for _ in 0..350 {
        let (_loss, grad) = loss_and_grad(
            theta,
            velocity0,
            affine0,
            target_velocity,
            target_affine,
            dt,
        );
        adam.step(&mut theta, &grad, &cfg);
    }

    let (final_loss, _) = loss_and_grad(
        theta,
        velocity0,
        affine0,
        target_velocity,
        target_affine,
        dt,
    );
    let fitted_velocity_damping = theta[0].exp();
    let fitted_affine_damping = theta[1].exp();
    eprintln!(
        "reduced viscous MPM fit: loss {initial_loss:.6e} -> {final_loss:.6e}, \
         velocity_damping={fitted_velocity_damping:.5} target={true_velocity_damping:.5}, \
         affine_damping={fitted_affine_damping:.5} target={true_affine_damping:.5}"
    );

    assert!(
        final_loss < initial_loss * 1.0e-8,
        "Adam did not reduce reduced viscous material loss enough: initial={initial_loss}, final={final_loss}"
    );
    assert!(
        (fitted_velocity_damping - true_velocity_damping).abs() < 1.0e-3,
        "fitted velocity damping mismatch: got {fitted_velocity_damping}, expected {true_velocity_damping}"
    );
    assert!(
        (fitted_affine_damping - true_affine_damping).abs() < 1.0e-3,
        "fitted affine damping mismatch: got {fitted_affine_damping}, expected {true_affine_damping}"
    );
}
