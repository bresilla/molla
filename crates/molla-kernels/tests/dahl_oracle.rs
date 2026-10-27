use molla_kernels::{DahlConfig, dahl_oracle_step};
use molla_math::{Real, Vec3};

fn integrate_dahl(
    mut z: Vec3,
    v_t: Vec3,
    steps: usize,
    normal: Vec3,
    f_n: Real,
    mu: Real,
    dt: Real,
    cfg: DahlConfig,
) -> (Vec3, Vec3) {
    let mut f_t = Vec3::ZERO;
    for _ in 0..steps {
        let (z_new, f_new) = dahl_oracle_step(z, v_t, normal, f_n, mu, dt, cfg);
        z = z_new;
        f_t = f_new;
    }
    (z, f_t)
}

#[test]
fn dahl_oracle_hysteresis_delays_force_flip_on_velocity_reversal() {
    let cfg = DahlConfig {
        sigma0: 1.0e3,
        ..DahlConfig::default()
    };
    let normal = Vec3::Y;
    let f_n: Real = 12.0;
    let mu: Real = 0.45;
    let dt: Real = 1.0 / 1000.0;
    let cap = mu * f_n;

    let (z_forward, f_forward) =
        integrate_dahl(Vec3::ZERO, Vec3::X * 0.35, 2_500, normal, f_n, mu, dt, cfg);
    assert!(
        (f_forward.length() - cap).abs() / cap < 0.02,
        "forward Dahl state did not reach the Coulomb cap: |f|={} cap={cap}",
        f_forward.length()
    );
    assert!(
        f_forward.x < -0.98 * cap,
        "positive slip should saturate negative tangent force: {f_forward:?}"
    );

    let (z_first_reverse, f_first_reverse) =
        dahl_oracle_step(z_forward, -Vec3::X * 0.35, normal, f_n, mu, dt, cfg);
    assert!(
        f_first_reverse.x < 0.0,
        "Dahl memory should delay force sign flip on the first reversed step: {f_first_reverse:?}"
    );
    assert!(
        z_first_reverse.x < z_forward.x,
        "reversed slip should unwind the bristle state: before={z_forward:?} after={z_first_reverse:?}"
    );

    let (_z_reverse, f_reverse) = integrate_dahl(
        z_first_reverse,
        -Vec3::X * 0.35,
        2_500,
        normal,
        f_n,
        mu,
        dt,
        cfg,
    );
    assert!(
        f_reverse.x > 0.98 * cap,
        "sustained negative slip should eventually saturate positive tangent force: {f_reverse:?}"
    );
    assert!(
        f_reverse.length() <= cap * 1.000_001,
        "Dahl force exceeded Coulomb cone after reversal: |f|={} cap={cap}",
        f_reverse.length()
    );
}

#[test]
fn dahl_oracle_projects_state_and_velocity_to_contact_tangent_plane() {
    let cfg = DahlConfig {
        sigma1: 25.0,
        ..DahlConfig::default()
    };
    let normal = Vec3::Y;
    let f_n: Real = 8.0;
    let mu: Real = 0.6;
    let z0 = Vec3::new(0.01, 3.0, -0.02);
    let v_t = Vec3::new(0.4, 7.5, -0.2);

    let (z, f_t) = dahl_oracle_step(z0, v_t, normal, f_n, mu, 1.0 / 240.0, cfg);

    assert!(
        z.dot(normal).abs() < 1.0e-12,
        "Dahl bristle state kept a normal component: z={z:?}"
    );
    assert!(
        f_t.dot(normal).abs() < 1.0e-12,
        "Dahl friction force kept a normal component: f_t={f_t:?}"
    );
    assert!(
        f_t.length() <= mu * f_n * 1.000_001,
        "Dahl tangent force exceeded Coulomb cap: |f|={} cap={}",
        f_t.length(),
        mu * f_n
    );
}
