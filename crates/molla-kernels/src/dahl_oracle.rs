//! CPU-side Dahl friction oracle shared by default tests and the GPU kernel.

use molla_math::{Real, Vec3};

/// Configuration for the Dahl friction oracle/kernel. All units SI.
#[derive(Copy, Clone, Debug)]
pub struct DahlConfig {
    /// Tangent stiffness sigma0 (N/m). Larger = sharper stick→slip
    /// transition.
    pub sigma0: Real,
    /// Visco-elastic damping sigma1 (N·s/m). Set 0.0 for pure Dahl.
    pub sigma1: Real,
    /// Per-contact normal-force threshold below which the bristle
    /// state is reset (N).
    pub f_normal_eps: Real,
}

impl Default for DahlConfig {
    fn default() -> Self {
        Self {
            sigma0: 1.0e5,
            sigma1: 0.0,
            f_normal_eps: 1.0e-9,
        }
    }
}

/// CPU oracle for the Dahl ODE — exact mirror of the WGSL `dahl_step`
/// kernel. Returns `(z_new, f_t)`. Used by default CPU tests and by
/// ignored GPU parity tests to check the analytical stick→slip transition.
pub fn dahl_oracle_step(
    z: Vec3,
    v_t: Vec3,
    normal: Vec3,
    f_n: Real,
    mu: Real,
    dt: Real,
    cfg: DahlConfig,
) -> (Vec3, Vec3) {
    let f_max = mu * f_n;
    if f_max < cfg.f_normal_eps {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let v_t_proj = v_t - normal * v_t.dot(normal);
    let z_proj = z - normal * z.dot(normal);
    let z_max = f_max / cfg.sigma0.max(1.0e-12);
    let mut dz = Vec3::ZERO;
    for axis in 0..3 {
        let vti = v_t_proj[axis];
        let zi = z_proj[axis];
        let s = if vti >= 0.0 { 1.0 } else { -1.0 };
        let sat = (1.0 - s * zi / z_max.max(1.0e-12)).max(0.0);
        dz[axis] = vti * sat;
    }
    let mut z_new = z_proj + dz * dt;
    let z_len = z_new.length();
    if z_len > z_max {
        z_new *= z_max / z_len.max(1.0e-12);
    }
    let mut f_t = -(z_new * cfg.sigma0 + dz * cfg.sigma1);
    let f_len = f_t.length();
    if f_len > f_max {
        f_t *= f_max / f_len.max(1.0e-12);
    }
    (z_new, f_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: pure Dahl ODE integrates from rest to a steady-state
    /// friction force equal to `μ · F_n` under constant tangent
    /// velocity (the textbook "slipping" branch).
    #[test]
    fn dahl_oracle_reaches_coulomb_cone() {
        let cfg = DahlConfig::default();
        let v_t = Vec3::new(0.5, 0.0, 0.0); // 0.5 m/s slip
        let normal = Vec3::Y;
        let f_n = 10.0_f64;
        let mu = 0.4_f64;
        let dt = 1.0 / 480.0;
        let mut z = Vec3::ZERO;
        let mut f_t = Vec3::ZERO;
        // Integrate long enough for the bristle to saturate.
        for _ in 0..2_000 {
            let (z_new, f_new) = dahl_oracle_step(z, v_t, normal, f_n, mu, dt, cfg);
            z = z_new;
            f_t = f_new;
        }
        let target = mu * f_n;
        let f_mag = f_t.length();
        assert!(
            (f_mag - target).abs() / target < 0.02,
            "Dahl steady-state {f_mag} vs analytical {target}",
        );
        // Force points opposite to velocity.
        assert!(f_t.x < 0.0, "Dahl friction must oppose motion: {f_t:?}");
    }
}
