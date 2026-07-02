//! Stage 42 — Hill-type muscle actuator integration tests.
//!
//! Three end-to-end checks on the CPU Featherstone solver — each
//! attaches a single muscle to a CABLE joint's linear stretch DoF
//! and exercises a different corner of the activation × length ×
//! velocity space.
//!
//!   1. `muscle_active_at_rest_generates_isometric_force` —
//!      pin the cable so the body cannot translate (high stiffness,
//!      light dt), full activation, and confirm the steady-state
//!      deflection matches the analytic `q_l = -f_max / k_lin`
//!      (force balance: spring restoring = muscle pulling).
//!
//!   2. `muscle_contracts_when_activated` —
//!      free cable (no gravity coupling on the stretch axis),
//!      ramp activation 0 → 1, confirm `q_l` ends up negative
//!      (contraction) and that increasing activation produces
//!      increasing |q_l| (monotone Hill response).
//!
//!   3. `muscle_zero_at_zero_activation` —
//!      cable + body at non-rest length and velocity, activation = 0,
//!      confirm muscle torque is **bit-exactly zero**: removing the
//!      muscle from the model gives the same trajectory.

use molla_actuators::{Muscle, MuscleConfig};
use molla_core::{BodyId, JointId, WorldId};
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

/// A spherical-inertia body with mass `m` and diagonal inertia tensor
/// `I·diag` — same scaffolding as `tests/joint_cable.rs`.
fn unit_body(b: &mut ModelBuilder, mass: Real, inertia_scalar: Real) -> BodyId {
    b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * inertia_scalar,
    }))
}

/// Build a single-cable + single-body model with the muscle attached.
/// `k_lin` selects the cable's linear stiffness (the spring that
/// counteracts the muscle pull); `0.0` ⇒ free cable.
fn single_muscle_model(
    mass: Real,
    inertia: Real,
    k_lin: Real,
    c_lin: Real,
    config: MuscleConfig,
) -> Model {
    let mut b = ModelBuilder::new();
    let body = unit_body(&mut b, mass, inertia);
    let joint = b
        .add_joint_cable(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            k_lin,
            0.0,   // no bend stiffness
            c_lin, // optional linear damping
            0.0,   // no bend damping
        )
        .unwrap();
    b.add_muscle(joint, config).unwrap();
    // Disable gravity — keep the stretch DoF a clean force balance.
    b.set_world_gravity(WorldId(0), V3::ZERO);
    b.build().unwrap()
}

#[test]
fn muscle_force_velocity_curve_scales_torque_directionally() {
    let muscle = Muscle::new(
        JointId::from_index(0),
        MuscleConfig {
            f_max: 100.0,
            l_opt: 0.10,
            v_max: 1.0,
            activation: 1.0,
            ..MuscleConfig::default()
        },
    );

    let shortening = muscle.torque(0.0, -0.50).abs();
    let isometric = muscle.torque(0.0, 0.0).abs();
    let lengthening = muscle.torque(0.0, 0.50).abs();
    eprintln!(
        "muscle force-velocity torque: shortening={shortening:.4}, isometric={isometric:.4}, lengthening={lengthening:.4}"
    );

    assert!(
        shortening < isometric,
        "shortening fibre should produce less force than isometric: shortening={shortening}, isometric={isometric}"
    );
    assert!(
        lengthening > isometric,
        "lengthening fibre should produce more force than isometric: lengthening={lengthening}, isometric={isometric}"
    );

    let antagonist = Muscle::new(
        JointId::from_index(0),
        MuscleConfig {
            polarity: -1.0,
            f_max: 100.0,
            l_opt: 0.10,
            v_max: 1.0,
            activation: 1.0,
            ..MuscleConfig::default()
        },
    );
    assert!(
        muscle.torque(0.0, 0.0) < 0.0 && antagonist.torque(0.0, 0.0) > 0.0,
        "agonist/antagonist polarity should flip torque sign"
    );
}

/// Test 1 — isometric force balance.
///
/// With a stiff spring (high `k_lin`) and a damper that kills
/// transients (`c_lin = 2·√(k·m)` ≈ critical), the cable settles to
/// the equilibrium where the spring force equals the active muscle
/// force:
///
/// ```text
///   τ_l = −k_lin · q_l + f_muscle = 0
///         ↓
///   q_l = f_muscle / k_lin       (negative — contraction)
/// ```
///
/// At full activation, rest length (`l_opt`), and zero velocity, the
/// Hill model produces `f_active = f_max`. So we expect
/// `q_l → −f_max / k_lin` within 5 %.
#[test]
fn muscle_active_at_rest_generates_isometric_force() {
    let mass: Real = 1.0;
    let inertia: Real = 0.05;
    // Pick numbers so the equilibrium deflection sits well inside the
    // active range of the force-length curve (|q_l/l_opt| < 0.15).
    let f_max: Real = 200.0;
    let l_opt: Real = 0.1;
    let k_lin: Real = 5000.0;
    // Critical damping for fast settling without overshoot.
    let c_lin: Real = 2.0 * (k_lin * mass).sqrt();

    let model = single_muscle_model(
        mass,
        inertia,
        k_lin,
        c_lin,
        MuscleConfig {
            f_max,
            l_opt,
            v_max: 0.5,
            activation: 1.0,
            ..MuscleConfig::default()
        },
    );

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();

    let dt: Real = 1.0e-4;
    // 2 s of simulated time at dt=1e-4 → 20 000 steps. Critical
    // damping with ω = √(k/m) ≈ 70 rad/s settles in ≪ 1 s.
    for _ in 0..20_000 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q_l = s_a.joint_q.host().unwrap()[0];
    let qd_l = s_a.joint_qd.host().unwrap()[0];

    // Analytical equilibrium: q_l = -f_active / k_lin. At |q_l/l_opt|
    // around 0.04 the Hill force–length factor is ~0.992, force–vel
    // is exactly 1 at qd=0. We tolerate 5 % to absorb the small
    // Gaussian drop without iterating the implicit equilibrium.
    let f_active_estimate = f_max * molla_actuators::force_length(1.0 + q_l / l_opt);
    let q_expected = -f_active_estimate / k_lin;
    let rel_err = (q_l - q_expected).abs() / q_expected.abs().max(1e-6);
    eprintln!(
        "isometric: q_l = {:.5} m, qd_l = {:.5} m/s, expected = {:.5} m, rel_err = {:.2}%",
        q_l,
        qd_l,
        q_expected,
        rel_err * 100.0
    );

    assert!(
        q_l < 0.0,
        "muscle should contract (q_l < 0), got q_l = {q_l}"
    );
    assert!(
        rel_err < 0.05,
        "isometric force balance off by {:.2}% (expected {:.5} m, got {:.5} m)",
        rel_err * 100.0,
        q_expected,
        q_l
    );
    assert!(
        qd_l.abs() < 1e-3,
        "velocity should be near zero at steady state, got qd_l = {qd_l}"
    );
}

/// Test 2 — free cable contracts and the deflection grows with
/// activation.
///
/// With `k_lin = 0` the cable has no spring; the muscle is the only
/// force on the stretch DoF. We integrate at three activation levels
/// (`0.25`, `0.5`, `1.0`) and check
///   - final `q_l < 0` (contraction);
///   - `|q_l(1.0)| > |q_l(0.5)| > |q_l(0.25)|` (monotone Hill).
///
/// The fibre length stays inside the active region of the Hill
/// curves because the integration window is short relative to the
/// "natural" muscle time-scale `l_opt / v_max`.
#[test]
fn muscle_contracts_when_activated() {
    fn run(activation: Real) -> Real {
        let mass: Real = 1.0;
        let inertia: Real = 0.05;
        let model = single_muscle_model(
            mass,
            inertia,
            0.0, // no spring
            0.0, // no damper
            MuscleConfig {
                f_max: 50.0,
                l_opt: 0.1,
                v_max: 0.5,
                activation,
                ..MuscleConfig::default()
            },
        );

        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new_cpu();

        let dt: Real = 1.0e-4;
        // 0.05 s window: short enough that the fibre stays well
        // inside [0.6 · l_opt, l_opt], i.e. force-length factor
        // remains > 0.7 and the Hill response is monotone in
        // activation.
        for _ in 0..500 {
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        s_a.joint_q.host().unwrap()[0]
    }

    let q_low = run(0.25);
    let q_mid = run(0.5);
    let q_high = run(1.0);
    eprintln!(
        "contraction: q(a=0.25) = {:.5}, q(a=0.5) = {:.5}, q(a=1.0) = {:.5}",
        q_low, q_mid, q_high
    );

    assert!(q_low < 0.0, "a=0.25 should contract, got q = {q_low}");
    assert!(q_mid < 0.0, "a=0.5 should contract, got q = {q_mid}");
    assert!(q_high < 0.0, "a=1.0 should contract, got q = {q_high}");

    // Hill model is monotone in activation: higher a ⇒ larger force ⇒
    // larger |q_l| at the same simulated time (we're in the linear
    // regime of the force-length curve).
    assert!(
        q_mid.abs() > q_low.abs(),
        "expected |q(0.5)| > |q(0.25)|, got {} vs {}",
        q_mid.abs(),
        q_low.abs()
    );
    assert!(
        q_high.abs() > q_mid.abs(),
        "expected |q(1.0)| > |q(0.5)|, got {} vs {}",
        q_high.abs(),
        q_mid.abs()
    );
}

/// Test 3 — zero activation yields zero muscle force regardless of
/// length and velocity.
///
/// Approach: drive the cable with a non-zero `q_l(0)`, `qd_l(0)` and
/// integrate once with the muscle attached (`activation = 0`) and
/// once on an identical model **without** the muscle. The two
/// trajectories must be bit-identical.
#[test]
fn muscle_zero_at_zero_activation() {
    // Pure-host unit test on `Muscle::torque` — covers the full
    // length × velocity grid the solver could ask about.
    let m = Muscle::new(
        molla_core::JointId::from_index(0),
        MuscleConfig {
            f_max: 500.0,
            l_opt: 0.05,
            v_max: 0.5,
            activation: 0.0,
            ..MuscleConfig::default()
        },
    );
    for &q in &[-0.03, -0.01, 0.0, 0.01, 0.03] {
        for &qd in &[-0.4, -0.1, 0.0, 0.1, 0.4] {
            let t = m.torque(q, qd);
            assert!(
                t.abs() < 1e-12,
                "expected exact zero torque at a=0, got {t} (q={q}, qd={qd})"
            );
        }
    }

    // Solver-level check: stepping a model with `activation = 0`
    // matches stepping a model with no muscle at all.
    let mass: Real = 1.0;
    let inertia: Real = 0.05;
    let k_lin: Real = 100.0; // ω = 10 rad/s
    let q0: Real = 0.02;
    let qd0: Real = 0.05;

    fn build(with_muscle: bool, mass: Real, inertia: Real, k_lin: Real) -> Model {
        let mut b = ModelBuilder::new();
        let body = unit_body(&mut b, mass, inertia);
        let joint = b
            .add_joint_cable(
                BodyId::NONE,
                body,
                Transform::IDENTITY,
                Transform::IDENTITY,
                k_lin,
                0.0,
                0.0,
                0.0,
            )
            .unwrap();
        if with_muscle {
            b.add_muscle(
                joint,
                MuscleConfig {
                    f_max: 500.0,
                    l_opt: 0.05,
                    v_max: 0.5,
                    activation: 0.0,
                    ..MuscleConfig::default()
                },
            )
            .unwrap();
        }
        b.set_world_gravity(WorldId(0), V3::ZERO);
        b.build().unwrap()
    }

    let model_a = build(true, mass, inertia, k_lin);
    let model_b = build(false, mass, inertia, k_lin);

    let mut s_a0 = model_a.state().unwrap();
    let mut s_a1 = model_a.state().unwrap();
    let mut s_b0 = model_b.state().unwrap();
    let mut s_b1 = model_b.state().unwrap();
    for s in [&mut s_a0, &mut s_b0] {
        s.joint_q.host_mut().unwrap()[0] = q0;
        s.joint_qd.host_mut().unwrap()[0] = qd0;
    }
    let control_a = model_a.control();
    let control_b = model_b.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver_a = FeatherstoneSolver::new_cpu();
    let mut solver_b = FeatherstoneSolver::new_cpu();
    let dt: Real = 1.0e-3;

    for step in 0..400 {
        solver_a
            .step(&model_a, &s_a0, &control_a, &contacts, dt, &mut s_a1)
            .unwrap();
        solver_b
            .step(&model_b, &s_b0, &control_b, &contacts, dt, &mut s_b1)
            .unwrap();
        std::mem::swap(&mut s_a0, &mut s_a1);
        std::mem::swap(&mut s_b0, &mut s_b1);

        let qa = s_a0.joint_q.host().unwrap()[0];
        let qb = s_b0.joint_q.host().unwrap()[0];
        let qda = s_a0.joint_qd.host().unwrap()[0];
        let qdb = s_b0.joint_qd.host().unwrap()[0];
        assert!(
            (qa - qb).abs() < 1e-12,
            "step {step}: q diverged with-muscle({qa}) vs without({qb})"
        );
        assert!(
            (qda - qdb).abs() < 1e-12,
            "step {step}: qd diverged with-muscle({qda}) vs without({qdb})"
        );
    }
}

// ── Stage 42 v2 — co-contraction / lag / SEE ───────────────────────

/// Test 4 — antagonist pair on the same joint at equal activation
/// cancels out so the cable doesn't translate.
///
/// Two muscles share one CABLE joint:
///   * agonist `polarity = +1` pulls `q_l → −`
///   * antagonist `polarity = −1` pulls `q_l → +`
///     Both at activation = 1.0, identical f_max / l_opt / v_max. The
///     Hill curves are symmetric at q_l = 0 / qd_l = 0, so the net
///     torque is zero ⇒ no motion. We integrate a free cable (no spring)
///     for half a second and check `|q_l| < 1 mm`.
#[test]
fn cocontraction_antagonist_pair_cancels_torque() {
    let mass: Real = 1.0;
    let inertia: Real = 0.05;
    let mut b = ModelBuilder::new();
    let body = unit_body(&mut b, mass, inertia);
    let joint = b
        .add_joint_cable(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        .unwrap();
    let cfg = MuscleConfig {
        f_max: 200.0,
        l_opt: 0.1,
        v_max: 0.5,
        activation: 1.0,
        ..MuscleConfig::default()
    };
    let agonist_id = b.add_muscle(joint, cfg).unwrap();
    let antagonist_id = b
        .add_muscle(
            joint,
            MuscleConfig {
                polarity: -1.0,
                ..cfg
            },
        )
        .unwrap();
    assert_ne!(agonist_id, antagonist_id);
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();
    // Sanity-check the CSR: joint 0 should map to both muscle ids.
    let offsets = model.joint_muscle_offsets.host().unwrap();
    let indices = model.joint_muscle_indices.host().unwrap();
    assert_eq!(offsets.len(), model.joint_count + 1);
    let lo = offsets[0] as usize;
    let hi = offsets[1] as usize;
    assert_eq!(
        hi - lo,
        2,
        "expected 2 muscles on the joint, got {}",
        hi - lo
    );
    let mut got: Vec<u32> = indices[lo..hi].to_vec();
    got.sort();
    assert_eq!(got, vec![agonist_id as u32, antagonist_id as u32]);

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();
    let dt: Real = 1.0e-4;
    for _ in 0..5_000 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q_l = s_a.joint_q.host().unwrap()[0];
    let qd_l = s_a.joint_qd.host().unwrap()[0];
    eprintln!("cocontraction: q_l = {:.6} m, qd_l = {:.6} m/s", q_l, qd_l);
    assert!(
        q_l.abs() < 1.0e-3,
        "antagonist pair should cancel; |q_l| = {} > 1e-3",
        q_l.abs()
    );
    assert!(qd_l.abs() < 1.0e-2);
}

/// Test 5 — first-order activation filter low-passes a step
/// excitation. With `u = 1`, `a(0) = 0`, `τ = 20 ms`, implicit
/// Euler converges to `a(t) ≈ 1 − exp(−t/τ)` (continuous solution).
/// We check `a(τ) ≈ 0.63` to ~5 % at `dt = τ/40`.
#[test]
fn activation_lag_low_pass_filters_step_excitation() {
    let mut b = ModelBuilder::new();
    let body = unit_body(&mut b, 1.0, 0.05);
    let joint = b
        .add_joint_cable(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        .unwrap();
    let tau: Real = 0.020;
    b.add_muscle(
        joint,
        MuscleConfig {
            f_max: 100.0,
            l_opt: 0.05,
            v_max: 0.5,
            activation: 0.0,
            excitation: 1.0,
            tau_activation: tau,
            ..MuscleConfig::default()
        },
    )
    .unwrap();
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let mut model = b.build().unwrap();

    let dt: Real = tau / 40.0;
    // Drive the excitation buffer to 1.0 (it was already set by
    // builder, but we re-stamp to mirror real-world usage where the
    // caller writes excitation each step).
    {
        let u = model.muscle_excitation.host_mut().unwrap();
        u[0] = 1.0;
    }

    // Run for one time constant.
    let n_steps = 40;
    for _ in 0..n_steps {
        molla_actuators::update_muscle_activation(&mut model, dt).unwrap();
    }
    let a = model.muscle_activation.host().unwrap()[0];
    let expected: Real = 1.0 - (-1.0_f64).exp(); // 1 - 1/e ≈ 0.6321
    eprintln!(
        "activation lag: a(τ) = {:.5}, expected = {:.5}",
        a, expected
    );
    assert!(
        (a - expected).abs() < 0.05 * expected,
        "a(τ) = {a} not within 5 % of {expected}"
    );

    // Continue out to ~10 τ total and verify it saturates near 1.
    // Implicit Euler with `r = dt/τ = 1/40` after N steps gives
    // `1 - (1 + r)^{-N}` ≈ `1 - exp(-N·r)` for small r — at N = 360
    // (9 τ extra after the first τ) the gap is ≈ `exp(-9) ≈ 1.2 ×
    // 10⁻⁴`, well below the 5 × 10⁻³ tolerance.
    for _ in 0..360 {
        molla_actuators::update_muscle_activation(&mut model, dt).unwrap();
    }
    let a_sat = model.muscle_activation.host().unwrap()[0];
    assert!(
        (a_sat - 1.0).abs() < 5.0e-3,
        "after ≈10τ activation should saturate, got {a_sat}"
    );

    // Step excitation back to 0 → activation decays with same τ.
    {
        let u = model.muscle_excitation.host_mut().unwrap();
        u[0] = 0.0;
    }
    for _ in 0..n_steps {
        molla_actuators::update_muscle_activation(&mut model, dt).unwrap();
    }
    let a_decay = model.muscle_activation.host().unwrap()[0];
    let expected_decay: Real = (-1.0_f64).exp(); // 1/e ≈ 0.368
    eprintln!(
        "deactivation: a(τ) = {:.5}, expected = {:.5}",
        a_decay, expected_decay
    );
    assert!(
        (a_decay - expected_decay).abs() < 0.05,
        "deactivation off: a = {a_decay}, expected ≈ {expected_decay}"
    );
}

/// Test 6 — series-elastic tendon softens the isometric response.
///
/// Two single-muscle models, identical except for `l_tendon_slack`:
///   * v1 (rigid tendon): `l_tendon_slack = 0`.
///   * v2 (SEE): `l_tendon_slack = 0.1 m` (same order as `l_opt`).
///     Both at full activation, cable + critically-damped spring.
///
/// Expectations:
///   * **Same steady-state deflection** within 10 % — both balance
///     `−k · q_l` against `f_max` once velocities die out.
///   * **Smoother time response**: the SEE model deflects more
///     gradually because the tendon absorbs a fraction of the
///     muscle force before it reaches the spring. We check that at
///     a short horizon (t = 0.02 s) the SEE deflection is *smaller*
///     than the rigid case (slower onset).
#[test]
fn tendon_see_softens_isometric_response() {
    fn build(l_tendon_slack: Real) -> Model {
        let mass: Real = 1.0;
        let inertia: Real = 0.05;
        let k_lin: Real = 1000.0;
        let c_lin: Real = 2.0 * (k_lin * mass).sqrt();
        let mut b = ModelBuilder::new();
        let body = unit_body(&mut b, mass, inertia);
        let joint = b
            .add_joint_cable(
                BodyId::NONE,
                body,
                Transform::IDENTITY,
                Transform::IDENTITY,
                k_lin,
                0.0,
                c_lin,
                0.0,
            )
            .unwrap();
        b.add_muscle(
            joint,
            MuscleConfig {
                f_max: 100.0,
                l_opt: 0.1,
                v_max: 0.5,
                activation: 1.0,
                excitation: 1.0,
                l_tendon_slack,
                ..MuscleConfig::default()
            },
        )
        .unwrap();
        b.set_world_gravity(WorldId(0), V3::ZERO);
        b.build().unwrap()
    }

    fn run(model: &Model, n_steps: usize, dt: Real) -> Vec<Real> {
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new_cpu();
        let mut trace = Vec::with_capacity(n_steps + 1);
        trace.push(s_a.joint_q.host().unwrap()[0]);
        for _ in 0..n_steps {
            solver
                .step(model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
            trace.push(s_a.joint_q.host().unwrap()[0]);
        }
        trace
    }

    let dt: Real = 1.0e-4;
    let n_steady = 5_000; // 0.5 s ⇒ critically damped settles
    let m_rigid = build(0.0);
    let m_see = build(0.10);
    let trace_rigid = run(&m_rigid, n_steady, dt);
    let trace_see = run(&m_see, n_steady, dt);

    let q_rigid_ss = *trace_rigid.last().unwrap();
    let q_see_ss = *trace_see.last().unwrap();
    eprintln!(
        "SEE steady-state: rigid q = {:.5}, SEE q = {:.5}",
        q_rigid_ss, q_see_ss
    );
    // Both should be negative (contractile).
    assert!(q_rigid_ss < 0.0);
    assert!(q_see_ss < 0.0);
    // Steady-state forces match to within 10 %: f_spring =
    // −k · q_l = f_muscle. Larger tolerance because the tendon's
    // small remaining strain shifts the fibre operating point on
    // the f_L curve.
    let rel = (q_see_ss - q_rigid_ss).abs() / q_rigid_ss.abs();
    assert!(
        rel < 0.10,
        "steady-state divergence too large: rigid = {q_rigid_ss}, SEE = {q_see_ss} (rel = {})",
        rel
    );

    // Short horizon: SEE should show slower onset.
    let short_idx = 200; // t = 0.02 s
    let q_short_rigid = trace_rigid[short_idx].abs();
    let q_short_see = trace_see[short_idx].abs();
    eprintln!(
        "SEE short-horizon: rigid |q| = {:.5}, SEE |q| = {:.5}",
        q_short_rigid, q_short_see
    );
    assert!(
        q_short_see < q_short_rigid,
        "SEE should soften time response; |q_see| = {} >= |q_rigid| = {}",
        q_short_see,
        q_short_rigid
    );
}
