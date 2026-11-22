//! Orbital and space-environment force fields for zero-G rigid worlds.
//!
//! Supports four standard orbital dynamics models alongside pure inertial
//! zero-G (`OrbitalField::None`):
//!
//! - [`OrbitalField::ClohessyWiltshire`] — linearized Hill / LVLH relative
//!   orbital dynamics around a circular reference orbit (ISS / space station
//!   rendezvous, proximity operations, and gravity-gradient torque).
//! - [`OrbitalField::EccentricRelative`] — exact nonlinear (or linearized
//!   Tschauner–Hempel) relative orbital dynamics around an arbitrary elliptical
//!   reference orbit `0 <= e < 1`, paired with the closed-form
//!   [`YamanakaAnkersen`] state transition matrix and two-impulse rendezvous
//!   targeting solver.
//! - [`OrbitalField::CentralGravity`] — inverse-square Newtonian gravity
//!   towards a central body, with optional `J2` oblateness perturbation and
//!   gravity-gradient torque.
//! - [`OrbitalField::CentralGravityZonal`] — inverse-square Newtonian gravity
//!   with full `J2`, `J3`, and `J4` zonal harmonic geopotential perturbations
//!   and gravity-gradient torque.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;

use molla_core::{Error, Result};
use molla_math::{Mat3, Real, SpatialVector, Vec3};
use molla_sim::State;
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};

/// Earth standard gravitational parameter `GM` (m³/s²).
pub const EARTH_MU: Real = 3.986_004_418e14;
/// Earth equatorial radius (m, WGS84).
pub const EARTH_EQUATORIAL_RADIUS: Real = 6_378_137.0;
/// Earth zonal harmonic `J2` oblateness coefficient (EGM96).
pub const EARTH_J2: Real = 1.082_626_68e-3;
/// Earth zonal harmonic `J3` pear-shape asymmetry coefficient (EGM96).
pub const EARTH_J3: Real = -2.532_656_48e-6;
/// Earth zonal harmonic `J4` higher-order oblateness coefficient (EGM96).
pub const EARTH_J4: Real = -1.619_621_59e-6;
/// Moon standard gravitational parameter `GM` (m³/s²).
pub const MOON_MU: Real = 4.904_869_5e12;
/// Mars standard gravitational parameter `GM` (m³/s²).
pub const MARS_MU: Real = 4.282_837e13;

/// Oblate-planet `J2` zonal harmonic perturbation for [`OrbitalField::CentralGravity`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct J2Perturbation {
    /// Dimensionless `J2` coefficient (e.g. [`EARTH_J2`]).
    pub j2: Real,
    /// Planet equatorial radius (m).
    pub equatorial_radius: Real,
    /// Unit spin/polar axis of the central body in world coordinates.
    pub pole_axis: Vec3,
}

impl J2Perturbation {
    /// Earth `J2` with polar axis along `pole_axis`.
    pub fn earth(pole_axis: Vec3) -> Self {
        Self {
            j2: EARTH_J2,
            equatorial_radius: EARTH_EQUATORIAL_RADIUS,
            pole_axis,
        }
    }
}

/// Zonal geopotential harmonics `J2`, `J3`, and `J4` for [`OrbitalField::CentralGravityZonal`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZonalHarmonics {
    /// Dimensionless `J2` oblateness coefficient.
    pub j2: Real,
    /// Dimensionless `J3` north–south pear-shape harmonic.
    pub j3: Real,
    /// Dimensionless `J4` fourth-degree zonal harmonic.
    pub j4: Real,
    /// Planet equatorial reference radius `R_e` (m).
    pub equatorial_radius: Real,
    /// Unit spin/polar axis of the central body in world coordinates.
    pub pole_axis: Vec3,
}

impl ZonalHarmonics {
    /// Earth EGM96 `J2`–`J4` harmonics with polar axis along `pole_axis`.
    pub fn earth_egm96(pole_axis: Vec3) -> Self {
        Self {
            j2: EARTH_J2,
            j3: EARTH_J3,
            j4: EARTH_J4,
            equatorial_radius: EARTH_EQUATORIAL_RADIUS,
            pole_axis,
        }
    }

    /// Promote a `J2`-only specification to [`ZonalHarmonics`].
    pub fn from_j2(p: J2Perturbation) -> Self {
        Self {
            j2: p.j2,
            j3: 0.0,
            j4: 0.0,
            equatorial_radius: p.equatorial_radius,
            pole_axis: p.pole_axis,
        }
    }

    fn validated(self) -> Result<Self> {
        if !self.j2.is_finite()
            || !self.j3.is_finite()
            || !self.j4.is_finite()
            || !self.equatorial_radius.is_finite()
            || self.equatorial_radius <= 0.0
            || !self.pole_axis.is_finite()
            || self.pole_axis.length_squared() < 1e-18
        {
            return Err(Error::Build(
                "orbital zonal harmonics parameters must be valid and finite".into(),
            ));
        }
        Ok(Self {
            pole_axis: self.pole_axis.normalize(),
            ..self
        })
    }

    /// Perturbation potential `U_{J2..J4}(r)` (m²/s², excluding point-mass `μ/r`).
    pub fn perturbation_potential(&self, mu: Real, r_vec: Vec3) -> Real {
        let r = r_vec.length();
        if r <= 0.0 {
            return 0.0;
        }
        let p_hat = self.pole_axis.normalize();
        let s = (r_vec.dot(p_hat) / r).clamp(-1.0, 1.0);
        let s2 = s * s;
        let re_r = self.equatorial_radius / r;
        let re_r2 = re_r * re_r;
        let re_r3 = re_r2 * re_r;
        let re_r4 = re_r2 * re_r2;
        let p2 = 0.5 * (3.0 * s2 - 1.0);
        let p3 = 0.5 * s * (5.0 * s2 - 3.0);
        let p4 = 0.125 * (35.0 * s2 * s2 - 30.0 * s2 + 3.0);
        -(mu / r) * (self.j2 * re_r2 * p2 + self.j3 * re_r3 * p3 + self.j4 * re_r4 * p4)
    }

    /// Gravitational potential `U(r)` (m²/s², where `a = +∇U`) including the
    /// point-mass term `μ/r` and `J2`–`J4` zonal harmonics.
    pub fn gravitational_potential(&self, mu: Real, r_vec: Vec3) -> Real {
        let r = r_vec.length();
        if r <= 0.0 {
            return 0.0;
        }
        mu / r + self.perturbation_potential(mu, r_vec)
    }

    /// Analytic perturbation acceleration `∇U_{J2..J4}` (m/s²), excluding the
    /// Newtonian `-μ r / r³` central term.
    pub fn perturbation_acceleration(&self, mu: Real, r_vec: Vec3, min_radius: Real) -> Vec3 {
        let r_raw = r_vec.length();
        let r = r_raw.max(min_radius);
        let r_hat = if r_raw > 1e-24 {
            r_vec / r_raw
        } else {
            Vec3::Y
        };
        let p_hat = self.pole_axis;
        let s = (r_hat.dot(p_hat)).clamp(-1.0, 1.0);
        let s2 = s * s;
        let re = self.equatorial_radius;
        let inv_r2 = 1.0 / (r * r);
        let inv_r4 = inv_r2 * inv_r2;

        let mut accel = Vec3::ZERO;
        if self.j2 != 0.0 {
            let c2 = -1.5 * self.j2 * mu * (re * re) * inv_r4;
            accel += c2 * ((1.0 - 5.0 * s2) * r_hat + (2.0 * s) * p_hat);
        }
        if self.j3 != 0.0 {
            let c3 = 0.5 * self.j3 * mu * (re * re * re) * (inv_r4 / r);
            accel += c3 * ((5.0 * s * (7.0 * s2 - 3.0)) * r_hat - (3.0 * (5.0 * s2 - 1.0)) * p_hat);
        }
        if self.j4 != 0.0 {
            let c4 = self.j4 * mu * (re * re * re * re) * (inv_r4 * inv_r2);
            accel += c4
                * ((1.875 * (21.0 * s2 * s2 - 14.0 * s2 + 1.0)) * r_hat
                    - (2.5 * s * (7.0 * s2 - 3.0)) * p_hat);
        }
        accel
    }
}

/// Keplerian elliptical chief orbit specification (`0 <= e < 1`) for eccentric
/// relative dynamics and [`YamanakaAnkersen`] propagation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EccentricOrbit {
    /// Central body gravitational parameter `μ = G M` (m³/s²).
    pub mu: Real,
    /// Semi-major axis `a` (m), strictly positive.
    pub semi_major_axis: Real,
    /// Orbital eccentricity `e ∈ [0, 1)`.
    pub eccentricity: Real,
    /// Chief true anomaly `ν₀` at simulation time `t = 0` (rad).
    pub initial_true_anomaly: Real,
}

impl EccentricOrbit {
    pub fn validated(self) -> Result<Self> {
        if !self.mu.is_finite()
            || self.mu <= 0.0
            || !self.semi_major_axis.is_finite()
            || self.semi_major_axis <= 0.0
            || !self.eccentricity.is_finite()
            || !(0.0..1.0).contains(&self.eccentricity)
            || !self.initial_true_anomaly.is_finite()
        {
            return Err(Error::Build(
                "eccentric orbit requires finite mu > 0, a > 0, 0 <= e < 1".into(),
            ));
        }
        Ok(self)
    }

    /// Semi-latus rectum `p = a (1 - e²)` (m).
    pub fn semi_latus_rectum(&self) -> Real {
        self.semi_major_axis * (1.0 - self.eccentricity * self.eccentricity)
    }

    /// Specific orbital angular momentum `h = sqrt(μ p)` (m²/s).
    pub fn angular_momentum(&self) -> Real {
        (self.mu * self.semi_latus_rectum()).sqrt()
    }

    /// Mean motion `n = sqrt(μ / a³)` (rad/s).
    pub fn mean_motion(&self) -> Real {
        let a = self.semi_major_axis;
        (self.mu / (a * a * a)).sqrt()
    }

    /// Orbital period `T = 2π / n` (s).
    pub fn period_s(&self) -> Real {
        TAU / self.mean_motion()
    }

    /// Yamanaka–Ankersen scaling constant `k² = h / p² = sqrt(μ / p³)` (rad/s).
    pub fn k_squared(&self) -> Real {
        let p = self.semi_latus_rectum();
        (self.mu / (p * p * p)).sqrt()
    }

    /// Dimensionless radius factor `ρ(ν) = 1 + e cos(ν)`.
    pub fn rho(&self, nu: Real) -> Real {
        1.0 + self.eccentricity * nu.cos()
    }

    /// Instantaneous chief orbital radius `r(ν) = p / (1 + e cos(ν))` (m).
    pub fn radius_at_nu(&self, nu: Real) -> Real {
        self.semi_latus_rectum() / self.rho(nu)
    }

    /// Instantaneous chief angular rate `ν̇ = h / r²` (rad/s).
    pub fn angular_rate_at_nu(&self, nu: Real) -> Real {
        let r = self.radius_at_nu(nu);
        self.angular_momentum() / (r * r)
    }

    /// Instantaneous chief radial speed `ṙ = (μ / h) e sin(ν)` (m/s).
    pub fn radial_rate_at_nu(&self, nu: Real) -> Real {
        (self.mu / self.angular_momentum()) * self.eccentricity * nu.sin()
    }

    /// Instantaneous chief angular acceleration `ν̈ = -2 ṙ ν̇ / r` (rad/s²).
    pub fn angular_accel_at_nu(&self, nu: Real) -> Real {
        let r = self.radius_at_nu(nu);
        let r_dot = self.radial_rate_at_nu(nu);
        let nu_dot = self.angular_rate_at_nu(nu);
        -2.0 * r_dot * nu_dot / r
    }

    /// Instantaneous inertial chief speed `v = sqrt(μ (2/r - 1/a))` (m/s).
    pub fn speed_at_nu(&self, nu: Real) -> Real {
        let r = self.radius_at_nu(nu);
        (self.mu * (2.0 / r - 1.0 / self.semi_major_axis)).max(0.0).sqrt()
    }

    /// Convert true anomaly `ν` (rad) to eccentric anomaly `E` (rad), preserving
    /// integer revolution count.
    pub fn eccentric_anomaly_from_nu(&self, nu: Real) -> Real {
        let revs = (nu / TAU).floor();
        let nu_wrapped = nu - revs * TAU;
        let factor = ((1.0 - self.eccentricity) / (1.0 + self.eccentricity)).sqrt();
        let half_e = (factor * (0.5 * nu_wrapped).tan()).atan();
        let mut e_wrapped = 2.0 * half_e;
        if e_wrapped < 0.0 {
            e_wrapped += TAU;
        }
        revs * TAU + e_wrapped
    }

    /// Convert true anomaly `ν` (rad) to mean anomaly `M = E - e sin(E)` (rad).
    pub fn mean_anomaly_from_nu(&self, nu: Real) -> Real {
        let e_anom = self.eccentric_anomaly_from_nu(nu);
        e_anom - self.eccentricity * e_anom.sin()
    }

    /// Elapsed flight time `t(ν) - t(ν₀)` (s) between true anomalies `nu0` and `nu`
    /// (preserving multi-orbit revolution counts).
    pub fn flight_time_between(&self, nu0: Real, nu: Real) -> Real {
        let dm = self.mean_anomaly_from_nu(nu) - self.mean_anomaly_from_nu(nu0);
        dm / self.mean_motion()
    }

    /// Solve Kepler's equation to machine precision for true anomaly `ν(t)` at
    /// `elapsed_s` seconds after `initial_true_anomaly`.
    pub fn true_anomaly_at(&self, elapsed_s: Real) -> Real {
        self.true_anomaly_after(self.initial_true_anomaly, elapsed_s)
    }

    /// Solve Kepler's equation for true anomaly `ν` at `dt_s` seconds after `nu0`.
    pub fn true_anomaly_after(&self, nu0: Real, dt_s: Real) -> Real {
        let m0 = self.mean_anomaly_from_nu(nu0);
        let m = m0 + self.mean_motion() * dt_s;
        let revs = (m / TAU).floor();
        let m_wrapped = m - revs * TAU;
        let e = self.eccentricity;
        let mut e_anom = if e < 0.8 {
            m_wrapped + e * m_wrapped.sin()
        } else {
            std::f64::consts::PI
        };
        for _ in 0..16 {
            let f = e_anom - e * e_anom.sin() - m_wrapped;
            let fp = 1.0 - e * e_anom.cos();
            let delta = f / fp;
            e_anom -= delta;
            if delta.abs() < 1e-14 {
                break;
            }
        }
        let factor = ((1.0 + e) / (1.0 - e)).sqrt();
        let half_nu = (factor * (0.5 * e_anom).tan()).atan();
        let mut nu_wrapped = 2.0 * half_nu;
        if nu_wrapped < 0.0 {
            nu_wrapped += TAU;
        }
        revs * TAU + nu_wrapped
    }
}

/// Closed-form Yamanaka–Ankersen (2002) state transition matrix and eccentric
/// two-impulse rendezvous solver in the chief LVLH frame `(r̂, v̂, ĥ)`.
pub struct YamanakaAnkersen;

impl YamanakaAnkersen {
    /// Fundamental 4×4 in-plane solution matrix `Ψ(ν)` for scaled Tschauner–Hempel
    /// state `[x̃, ỹ, x̃', ỹ']^T` where `x̃ = ρ x, ỹ = ρ y` and `J = k² (t - t₀)`.
    fn psi_inplane(e: Real, nu: Real, j_val: Real) -> [[Real; 4]; 4] {
        let sn = nu.sin();
        let cn = nu.cos();
        let rho = 1.0 + e * cn;
        let s = rho * sn;
        let c = rho * cn;
        let s_p = cn + e * (2.0 * nu).cos();
        let c_p = -(sn + e * (2.0 * nu).sin());
        let inv_rho = 1.0 / rho;
        [
            [0.0, s, c, 2.0 - 3.0 * e * s * j_val],
            [1.0, c * (1.0 + inv_rho), -s * (1.0 + inv_rho), -3.0 * rho * rho * j_val],
            [0.0, s_p, c_p, -3.0 * e * (s_p * j_val + sn * inv_rho)],
            [0.0, -2.0 * s, -2.0 * c + e, 6.0 * e * s * j_val - 3.0],
        ]
    }

    /// Invert a small 4×4 matrix via Gaussian elimination with partial pivoting.
    fn invert_4x4(mut a: [[Real; 4]; 4]) -> Result<[[Real; 4]; 4]> {
        let mut inv = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for col in 0..4 {
            let mut pivot = col;
            let mut max_val = a[col][col].abs();
            for row in (col + 1)..4 {
                if a[row][col].abs() > max_val {
                    max_val = a[row][col].abs();
                    pivot = row;
                }
            }
            if max_val < 1e-14 {
                return Err(Error::Solver("singular Yamanaka-Ankersen matrix".into()));
            }
            a.swap(col, pivot);
            inv.swap(col, pivot);
            let diag = a[col][col];
            for j in 0..4 {
                a[col][j] /= diag;
                inv[col][j] /= diag;
            }
            for row in 0..4 {
                if row == col {
                    continue;
                }
                let f = a[row][col];
                for j in 0..4 {
                    a[row][j] -= f * a[col][j];
                    inv[row][j] -= f * inv[col][j];
                }
            }
        }
        Ok(inv)
    }

    fn invert_3x3(m: [[Real; 3]; 3]) -> Result<[[Real; 3]; 3]> {
        let mat = Mat3::from_cols(
            Vec3::new(m[0][0], m[1][0], m[2][0]),
            Vec3::new(m[0][1], m[1][1], m[2][1]),
            Vec3::new(m[0][2], m[1][2], m[2][2]),
        );
        if mat.determinant().abs() < 1e-14 {
            return Err(Error::Solver(
                "singular Yamanaka-Ankersen transfer block (half-orbit resonance)".into(),
            ));
        }
        let inv = mat.inverse();
        Ok([
            [inv.x_axis.x, inv.y_axis.x, inv.z_axis.x],
            [inv.x_axis.y, inv.y_axis.y, inv.z_axis.y],
            [inv.x_axis.z, inv.y_axis.z, inv.z_axis.z],
        ])
    }

    /// Physical 6×6 State Transition Matrix `Φ(ν, ν₀)` mapping LVLH relative state
    /// `[x, y, z, vx, vy, vz]` (with `x = radial`, `y = along-track`, `z = cross-track`)
    /// from true anomaly `nu0` to `nu`.
    pub fn stm(orbit: &EccentricOrbit, nu0: Real, nu: Real) -> Result<[[Real; 6]; 6]> {
        let orbit = orbit.validated()?;
        let e = orbit.eccentricity;
        let k2 = orbit.k_squared();
        let rho0 = orbit.rho(nu0);
        let rho = orbit.rho(nu);
        let dt = orbit.flight_time_between(nu0, nu);
        let j_val = k2 * dt;

        // 1. In-plane scaled STM: Φ̃_in = Ψ(ν, J) * Ψ(ν₀, 0)⁻¹
        let psi0 = Self::psi_inplane(e, nu0, 0.0);
        let psi0_inv = Self::invert_4x4(psi0)?;
        let psi1 = Self::psi_inplane(e, nu, j_val);
        let mut phi_tilde = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for m in 0..4 {
                    sum += psi1[i][m] * psi0_inv[m][j];
                }
                phi_tilde[i][j] = sum;
            }
        }

        // Convert between physical [x, y, vx, vy] and scaled [x̃, ỹ, x̃', ỹ']:
        //   x̃ = ρ x,  x̃' = -e sin(ν) x + vx / (k² ρ)
        //   x = x̃ / ρ, vx = k² (e sin(ν) x̃ + ρ x̃')
        let es0 = e * nu0.sin();
        let es1 = e * nu.sin();
        let inv_k2_rho0 = 1.0 / (k2 * rho0);
        let t0 = [
            [rho0, 0.0, 0.0, 0.0],
            [0.0, rho0, 0.0, 0.0],
            [-es0, 0.0, inv_k2_rho0, 0.0],
            [0.0, -es0, 0.0, inv_k2_rho0],
        ];
        let inv_rho = 1.0 / rho;
        let t1_inv = [
            [inv_rho, 0.0, 0.0, 0.0],
            [0.0, inv_rho, 0.0, 0.0],
            [k2 * es1, 0.0, k2 * rho, 0.0],
            [0.0, k2 * es1, 0.0, k2 * rho],
        ];

        // tmp = phi_tilde * t0
        let mut tmp = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for m in 0..4 {
                    sum += phi_tilde[i][m] * t0[m][j];
                }
                tmp[i][j] = sum;
            }
        }
        // phi_in = t1_inv * tmp
        let mut phi_in = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for m in 0..4 {
                    sum += t1_inv[i][m] * tmp[m][j];
                }
                phi_in[i][j] = sum;
            }
        }

        // 2. Out-of-plane 2×2 STM: z̃'' + z̃ = 0
        let dnu = nu - nu0;
        let cd = dnu.cos();
        let sd = dnu.sin();
        // [z̃, z̃']_1 = [[cd, sd], [-sd, cd]] * [[rho0, 0], [-es0, 1/(k² rho0)]] * [z, vz]_0
        let z00 = (cd * rho0 - sd * es0) * inv_rho;
        let z01 = (sd * inv_k2_rho0) * inv_rho;
        let zp0 = -sd * rho0 - cd * es0;
        let zp1 = cd * inv_k2_rho0;
        let z10 = k2 * (es1 * (cd * rho0 - sd * es0) + rho * zp0);
        let z11 = k2 * (es1 * (sd * inv_k2_rho0) + rho * zp1);

        // Assemble 6×6 physical STM for [x, y, z, vx, vy, vz]:
        let mut phi = [[0.0; 6]; 6];
        let map = [0, 1, 3, 4];
        for i in 0..4 {
            for j in 0..4 {
                phi[map[i]][map[j]] = phi_in[i][j];
            }
        }
        phi[2][2] = z00;
        phi[2][5] = z01;
        phi[5][2] = z10;
        phi[5][5] = z11;
        Ok(phi)
    }

    /// Propagate LVLH relative `(pos, vel)` (radial `x`, along-track `y`,
    /// cross-track `z`) forward by `dt_s` seconds from true anomaly `nu0`.
    pub fn propagate(
        orbit: &EccentricOrbit,
        nu0: Real,
        dt_s: Real,
        rel_pos0: Vec3,
        rel_vel0: Vec3,
    ) -> Result<(Vec3, Vec3)> {
        let nu = orbit.true_anomaly_after(nu0, dt_s);
        let phi = Self::stm(orbit, nu0, nu)?;
        let s0 = [
            rel_pos0.x,
            rel_pos0.y,
            rel_pos0.z,
            rel_vel0.x,
            rel_vel0.y,
            rel_vel0.z,
        ];
        let mut s1 = [0.0; 6];
        for i in 0..6 {
            let mut sum = 0.0;
            for j in 0..6 {
                sum += phi[i][j] * s0[j];
            }
            s1[i] = sum;
        }
        Ok((Vec3::new(s1[0], s1[1], s1[2]), Vec3::new(s1[3], s1[4], s1[5])))
    }

    /// Closed-form two-impulse eccentric rendezvous targeting solver: returns
    /// `(dv1_departure, dv2_arrival)` in the chief LVLH frame `(r̂, v̂, ĥ)` to
    /// transfer from `(rel_pos0, rel_vel0)` to `(target_pos, target_vel)` in `dt_s`.
    pub fn rendezvous_delta_v(
        orbit: &EccentricOrbit,
        nu0: Real,
        dt_s: Real,
        rel_pos0: Vec3,
        rel_vel0: Vec3,
        target_pos: Vec3,
        target_vel: Vec3,
    ) -> Result<(Vec3, Vec3)> {
        let nu = orbit.true_anomaly_after(nu0, dt_s);
        let phi = Self::stm(orbit, nu0, nu)?;
        let mul3 = |b: [[Real; 3]; 3], v: Vec3| {
            Vec3::new(
                b[0][0] * v.x + b[0][1] * v.y + b[0][2] * v.z,
                b[1][0] * v.x + b[1][1] * v.y + b[1][2] * v.z,
                b[2][0] * v.x + b[2][1] * v.y + b[2][2] * v.z,
            )
        };
        let phi_rr = [
            [phi[0][0], phi[0][1], phi[0][2]],
            [phi[1][0], phi[1][1], phi[1][2]],
            [phi[2][0], phi[2][1], phi[2][2]],
        ];
        let phi_rv = [
            [phi[0][3], phi[0][4], phi[0][5]],
            [phi[1][3], phi[1][4], phi[1][5]],
            [phi[2][3], phi[2][4], phi[2][5]],
        ];
        let phi_vr = [
            [phi[3][0], phi[3][1], phi[3][2]],
            [phi[4][0], phi[4][1], phi[4][2]],
            [phi[5][0], phi[5][1], phi[5][2]],
        ];
        let phi_vv = [
            [phi[3][3], phi[3][4], phi[3][5]],
            [phi[4][3], phi[4][4], phi[4][5]],
            [phi[5][3], phi[5][4], phi[5][5]],
        ];
        let phi_rv_inv = Self::invert_3x3(phi_rv)?;
        let v0_plus = mul3(phi_rv_inv, target_pos - mul3(phi_rr, rel_pos0));
        let dv1 = v0_plus - rel_vel0;
        let vf_minus = mul3(phi_vr, rel_pos0) + mul3(phi_vv, v0_plus);
        let dv2 = target_vel - vf_minus;
        Ok((dv1, dv2))
    }
}

/// Orbital dynamics force model active on a rigid world.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum OrbitalField {
    /// Pure inertial space (no orbital forces).
    #[default]
    None,
    /// Clohessy–Wiltshire (Hill / LVLH) relative dynamics in a rotating
    /// circular-orbit reference frame.
    ///
    /// With orthonormal basis `(r̂, v̂, ĥ = r̂ × v̂)` where `r̂` points radially
    /// away from the planet and `v̂` points along the orbital velocity:
    /// - `a_cw = -2 n (ĥ × v) + 3 n² (p·r̂) r̂ - n² (p·ĥ) ĥ`
    /// - optional gravity-gradient torque `τ_gg = 3 n² r̂ × (I_world r̂)`
    ClohessyWiltshire {
        /// Reference orbit mean motion `n = sqrt(μ / r₀³)` (rad/s).
        mean_motion: Real,
        /// Unit radial vector `r̂` (away from planet centre).
        radial_axis: Vec3,
        /// Unit along-track vector `v̂` (orthogonal to `radial_axis`).
        along_track_axis: Vec3,
        /// World position of the reference orbit origin.
        origin: Vec3,
        /// Enable gravity-gradient torque on rigid bodies.
        gravity_gradient: bool,
    },
    /// Eccentric chief-orbit LVLH relative orbital dynamics (`0 <= e < 1`).
    ///
    /// Propagates the chief's exact Keplerian true anomaly `ν(t)` internally.
    /// When `linearized == false`, applies the exact nonlinear two-body relative
    /// equations in the non-uniformly rotating LVLH frame; when `linearized == true`,
    /// applies the exact Tschauner–Hempel / Yamanaka–Ankersen differential equations.
    EccentricRelative {
        orbit: EccentricOrbit,
        /// Unit radial vector `r̂` (away from planet centre).
        radial_axis: Vec3,
        /// Unit along-track vector `v̂` (orthogonal to `radial_axis`).
        along_track_axis: Vec3,
        /// World position of the chief orbit origin.
        origin: Vec3,
        /// Enable gravity-gradient torque `τ_gg = (3 μ / r(ν)³) r̂ × (I_world r̂)`.
        gravity_gradient: bool,
        /// `false` = exact nonlinear eccentric relative EOM; `true` = linearized Tschauner–Hempel.
        linearized: bool,
    },
    /// Inverse-square Newtonian gravity towards `center`, with optional `J2`
    /// oblateness and gravity-gradient torque.
    CentralGravity {
        /// Planet centre in world coordinates (m).
        center: Vec3,
        /// Standard gravitational parameter `μ = G M` (m³/s²).
        mu: Real,
        /// Minimum distance clamp (m) to guard against singularity at `center`.
        min_radius: Real,
        /// Optional `J2` oblateness perturbation.
        j2: Option<J2Perturbation>,
        /// Enable gravity-gradient torque `τ_gg = (3 μ / r³) r̂ × (I_world r̂)`.
        gravity_gradient: bool,
    },
    /// Inverse-square Newtonian gravity with full `J2`, `J3`, and `J4` zonal
    /// geopotential harmonics and optional gravity-gradient torque.
    CentralGravityZonal {
        /// Planet centre in world coordinates (m).
        center: Vec3,
        /// Standard gravitational parameter `μ = G M` (m³/s²).
        mu: Real,
        /// Minimum distance clamp (m) to guard against singularity at `center`.
        min_radius: Real,
        /// Zonal harmonics `J2`–`J4` specification.
        zonal: ZonalHarmonics,
        /// Enable gravity-gradient torque `τ_gg = (3 μ / r³) r̂ × (I_world r̂)`.
        gravity_gradient: bool,
    },
}

impl OrbitalField {
    /// Circular-orbit LVLH Clohessy–Wiltshire frame from `mu` and `orbit_radius`.
    pub fn circular_lvlh(
        mu: Real,
        orbit_radius: Real,
        radial_axis: Vec3,
        along_track_axis: Vec3,
        gravity_gradient: bool,
    ) -> Result<Self> {
        if !mu.is_finite() || mu <= 0.0 || !orbit_radius.is_finite() || orbit_radius <= 0.0 {
            return Err(Error::Build(
                "orbital mu and orbit_radius must be finite and positive".into(),
            ));
        }
        let mean_motion = (mu / (orbit_radius * orbit_radius * orbit_radius)).sqrt();
        let field = Self::ClohessyWiltshire {
            mean_motion,
            radial_axis,
            along_track_axis,
            origin: Vec3::ZERO,
            gravity_gradient,
        };
        field.validated()
    }

    /// ISS-like Low Earth Orbit (408 km altitude, `r̂ = +Y`, `v̂ = +X`).
    pub fn iss_leo() -> Self {
        let r0 = EARTH_EQUATORIAL_RADIUS + 408_000.0;
        let n = (EARTH_MU / (r0 * r0 * r0)).sqrt();
        Self::ClohessyWiltshire {
            mean_motion: n,
            radial_axis: Vec3::Y,
            along_track_axis: Vec3::X,
            origin: Vec3::ZERO,
            gravity_gradient: true,
        }
    }

    /// Earth EGM96 central gravity with full `J2`, `J3`, and `J4` zonal harmonics.
    pub fn earth_egm96(center: Vec3, pole_axis: Vec3, gravity_gradient: bool) -> Result<Self> {
        Self::CentralGravityZonal {
            center,
            mu: EARTH_MU,
            min_radius: 100_000.0,
            zonal: ZonalHarmonics::earth_egm96(pole_axis),
            gravity_gradient,
        }
        .validated()
    }

    /// Eccentric LVLH relative orbital field around an elliptical chief orbit.
    pub fn eccentric_lvlh(
        orbit: EccentricOrbit,
        radial_axis: Vec3,
        along_track_axis: Vec3,
        gravity_gradient: bool,
        linearized: bool,
    ) -> Result<Self> {
        Self::EccentricRelative {
            orbit,
            radial_axis,
            along_track_axis,
            origin: Vec3::ZERO,
            gravity_gradient,
            linearized,
        }
        .validated()
    }

    fn validated(self) -> Result<Self> {
        match self {
            Self::None => Ok(Self::None),
            Self::ClohessyWiltshire {
                mean_motion,
                radial_axis,
                along_track_axis,
                origin,
                gravity_gradient,
            } => {
                if !mean_motion.is_finite() || mean_motion < 0.0 {
                    return Err(Error::Build(
                        "orbital mean_motion must be finite and nonnegative".into(),
                    ));
                }
                if !origin.is_finite() {
                    return Err(Error::Build("orbital origin must be finite".into()));
                }
                if !radial_axis.is_finite()
                    || radial_axis.length_squared() < 1e-18
                    || !along_track_axis.is_finite()
                    || along_track_axis.length_squared() < 1e-18
                {
                    return Err(Error::Build(
                        "orbital axes must be finite and non-zero".into(),
                    ));
                }
                let r_hat = radial_axis.normalize();
                let v_hat = along_track_axis.normalize();
                if r_hat.dot(v_hat).abs() > 1e-6 {
                    return Err(Error::Build(
                        "orbital radial_axis and along_track_axis must be orthogonal".into(),
                    ));
                }
                Ok(Self::ClohessyWiltshire {
                    mean_motion,
                    radial_axis: r_hat,
                    along_track_axis: v_hat,
                    origin,
                    gravity_gradient,
                })
            }
            Self::EccentricRelative {
                orbit,
                radial_axis,
                along_track_axis,
                origin,
                gravity_gradient,
                linearized,
            } => {
                let orbit = orbit.validated()?;
                if !origin.is_finite() {
                    return Err(Error::Build("orbital origin must be finite".into()));
                }
                if !radial_axis.is_finite()
                    || radial_axis.length_squared() < 1e-18
                    || !along_track_axis.is_finite()
                    || along_track_axis.length_squared() < 1e-18
                {
                    return Err(Error::Build(
                        "orbital axes must be finite and non-zero".into(),
                    ));
                }
                let r_hat = radial_axis.normalize();
                let v_hat = along_track_axis.normalize();
                if r_hat.dot(v_hat).abs() > 1e-6 {
                    return Err(Error::Build(
                        "orbital radial_axis and along_track_axis must be orthogonal".into(),
                    ));
                }
                Ok(Self::EccentricRelative {
                    orbit,
                    radial_axis: r_hat,
                    along_track_axis: v_hat,
                    origin,
                    gravity_gradient,
                    linearized,
                })
            }
            Self::CentralGravity {
                center,
                mu,
                min_radius,
                j2,
                gravity_gradient,
            } => {
                if !center.is_finite() {
                    return Err(Error::Build("orbital center must be finite".into()));
                }
                if !mu.is_finite() || mu < 0.0 {
                    return Err(Error::Build(
                        "orbital mu must be finite and nonnegative".into(),
                    ));
                }
                if !min_radius.is_finite() || min_radius <= 0.0 {
                    return Err(Error::Build(
                        "orbital min_radius must be finite and positive".into(),
                    ));
                }
                let j2 = match j2 {
                    Some(p) => {
                        if !p.j2.is_finite()
                            || !p.equatorial_radius.is_finite()
                            || p.equatorial_radius <= 0.0
                            || !p.pole_axis.is_finite()
                            || p.pole_axis.length_squared() < 1e-18
                        {
                            return Err(Error::Build(
                                "orbital J2 parameters must be valid and finite".into(),
                            ));
                        }
                        Some(J2Perturbation {
                            pole_axis: p.pole_axis.normalize(),
                            ..p
                        })
                    }
                    None => None,
                };
                Ok(Self::CentralGravity {
                    center,
                    mu,
                    min_radius,
                    j2,
                    gravity_gradient,
                })
            }
            Self::CentralGravityZonal {
                center,
                mu,
                min_radius,
                zonal,
                gravity_gradient,
            } => {
                if !center.is_finite() {
                    return Err(Error::Build("orbital center must be finite".into()));
                }
                if !mu.is_finite() || mu < 0.0 {
                    return Err(Error::Build(
                        "orbital mu must be finite and nonnegative".into(),
                    ));
                }
                if !min_radius.is_finite() || min_radius <= 0.0 {
                    return Err(Error::Build(
                        "orbital min_radius must be finite and positive".into(),
                    ));
                }
                let zonal = zonal.validated()?;
                Ok(Self::CentralGravityZonal {
                    center,
                    mu,
                    min_radius,
                    zonal,
                    gravity_gradient,
                })
            }
        }
    }
}

/// Orbital force and torque experienced by one body over the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrbitalReading {
    /// Specific orbital acceleration (m/s², world), averaged over the step.
    pub acceleration: Vec3,
    /// Orbital force applied to the body (N, world), averaged over the step.
    pub force: Vec3,
    /// Gravity-gradient torque applied to the body (N·m, world), averaged over
    /// the step.
    pub torque: Vec3,
}

/// Orbital environment of a rigid world.
#[derive(Default)]
pub struct Orbital {
    field: OrbitalField,
    elapsed_s: Real,
    excluded: BTreeSet<BodyHandle>,
    readings: BTreeMap<BodyHandle, OrbitalReading>,
}

impl Orbital {
    pub fn set_field(&mut self, field: OrbitalField) -> Result<()> {
        self.field = field.validated()?;
        self.elapsed_s = 0.0;
        self.readings.clear();
        Ok(())
    }

    pub fn field(&self) -> &OrbitalField {
        &self.field
    }

    /// Elapsed orbital propagation time (s) since the current field was set.
    pub fn elapsed_s(&self) -> Real {
        self.elapsed_s
    }

    /// Current chief true anomaly `ν(t)` (rad) when [`OrbitalField::EccentricRelative`] is active.
    pub fn chief_true_anomaly(&self) -> Option<Real> {
        match self.field {
            OrbitalField::EccentricRelative { orbit, .. } => Some(orbit.true_anomaly_at(self.elapsed_s)),
            _ => None,
        }
    }

    /// Current chief orbital radius `r(t)` (m) when [`OrbitalField::EccentricRelative`] is active.
    pub fn chief_radius(&self) -> Option<Real> {
        match self.field {
            OrbitalField::EccentricRelative { orbit, .. } => {
                let nu = orbit.true_anomaly_at(self.elapsed_s);
                Some(orbit.radius_at_nu(nu))
            }
            _ => None,
        }
    }

    /// Current chief orbital speed `v(t)` (m/s) when [`OrbitalField::EccentricRelative`] is active.
    pub fn chief_speed(&self) -> Option<Real> {
        match self.field {
            OrbitalField::EccentricRelative { orbit, .. } => {
                let nu = orbit.true_anomaly_at(self.elapsed_s);
                Some(orbit.speed_at_nu(nu))
            }
            _ => None,
        }
    }

    /// Exempt a body from the orbital field (e.g. a station hull held fixed).
    pub fn exclude_body(&mut self, body: BodyHandle) {
        self.excluded.insert(body);
    }

    /// Re-include a previously excluded body.
    pub fn include_body(&mut self, body: BodyHandle) {
        self.excluded.remove(&body);
    }

    pub fn reading(&self, body: BodyHandle) -> Option<OrbitalReading> {
        self.readings.get(&body).copied()
    }

    pub fn active(&self) -> bool {
        !matches!(self.field, OrbitalField::None)
    }

    pub(crate) fn before_step(&mut self) {
        for reading in self.readings.values_mut() {
            *reading = OrbitalReading::default();
        }
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        wrenches: &mut [SpatialVector],
        sub_dt: Real,
        weight: Real,
    ) -> Result<()> {
        if matches!(self.field, OrbitalField::None) {
            return Ok(());
        }
        // Midpoint chief time for second-order symplectic accuracy on eccentric orbits.
        let t_mid = self.elapsed_s + 0.5 * sub_dt;
        self.elapsed_s += sub_dt;

        let poses = state.body_q.host()?;
        let velocities = state.body_qd.host()?;
        let model = scene.model();
        let masses = model.body_mass.host()?;
        let inertias = model.body_inertia.host()?;
        let centers = model.body_com.host()?;

        for (id, desc) in scene.bodies() {
            if !desc.enabled
                || desc.kind != BodyKind::Dynamic
                || self.excluded.contains(&id)
            {
                continue;
            }
            let Some(body) = scene.body_index(id) else {
                continue;
            };
            let i = body.index();
            let mass = masses[i];
            if mass <= 0.0 {
                continue;
            }
            let com = poses[i].transform_point(centers[i]);
            let (accel, torque) = match self.field {
                OrbitalField::None => unreachable!(),
                OrbitalField::ClohessyWiltshire {
                    mean_motion: n,
                    radial_axis: r_hat,
                    along_track_axis: v_hat,
                    origin,
                    gravity_gradient,
                } => {
                    let h_hat = r_hat.cross(v_hat);
                    let p = com - origin;
                    let v = velocities[i].linear;
                    let n2 = n * n;
                    let coriolis = -2.0 * n * h_hat.cross(v);
                    let tidal = (3.0 * n2 * p.dot(r_hat)) * r_hat - (n2 * p.dot(h_hat)) * h_hat;
                    let accel = coriolis + tidal;
                    let tau = if gravity_gradient {
                        let rot = Mat3::from_quat(poses[i].rotation);
                        let i_world = rot * inertias[i] * rot.transpose();
                        (3.0 * n2) * r_hat.cross(i_world * r_hat)
                    } else {
                        Vec3::ZERO
                    };
                    (accel, tau)
                }
                OrbitalField::EccentricRelative {
                    orbit,
                    radial_axis: r_hat,
                    along_track_axis: v_hat,
                    origin,
                    gravity_gradient,
                    linearized,
                } => {
                    let h_hat = r_hat.cross(v_hat);
                    let nu = orbit.true_anomaly_at(t_mid);
                    let r_chief = orbit.radius_at_nu(nu);
                    let nu_dot = orbit.angular_rate_at_nu(nu);
                    let nu_ddot = orbit.angular_accel_at_nu(nu);
                    let omega_c = h_hat * nu_dot;
                    let alpha_c = h_hat * nu_ddot;

                    let p = com - origin;
                    let v = velocities[i].linear;
                    let coriolis = -2.0 * omega_c.cross(v);
                    let euler = -alpha_c.cross(p);
                    let centrifugal = -omega_c.cross(omega_c.cross(p));

                    let inv_r3 = 1.0 / (r_chief * r_chief * r_chief);
                    let grav_diff = if linearized {
                        let k = orbit.mu * inv_r3;
                        let x = p.dot(r_hat);
                        let y = p.dot(v_hat);
                        let z = p.dot(h_hat);
                        k * ((2.0 * x) * r_hat - y * v_hat - z * h_hat)
                    } else {
                        let r_dep = r_hat * r_chief + p;
                        let rd_mag = r_dep.length().max(1.0);
                        let rd_inv3 = 1.0 / (rd_mag * rd_mag * rd_mag);
                        -orbit.mu * rd_inv3 * r_dep + (orbit.mu / (r_chief * r_chief)) * r_hat
                    };
                    let accel = coriolis + euler + centrifugal + grav_diff;
                    let tau = if gravity_gradient {
                        let rot = Mat3::from_quat(poses[i].rotation);
                        let i_world = rot * inertias[i] * rot.transpose();
                        (3.0 * orbit.mu * inv_r3) * r_hat.cross(i_world * r_hat)
                    } else {
                        Vec3::ZERO
                    };
                    (accel, tau)
                }
                OrbitalField::CentralGravity {
                    center,
                    mu,
                    min_radius,
                    j2,
                    gravity_gradient,
                } => {
                    let r_vec = com - center;
                    let r_raw = r_vec.length();
                    let r = r_raw.max(min_radius);
                    let r_hat = if r_raw > 1e-24 {
                        r_vec / r_raw
                    } else {
                        Vec3::Y
                    };
                    let inv_r2 = 1.0 / (r * r);
                    let inv_r3 = inv_r2 / r;
                    let mut accel = -mu * inv_r3 * r_vec;
                    if let Some(p) = j2 {
                        let z = r_vec.dot(p.pole_axis);
                        let s = z / r;
                        let inv_r5 = inv_r3 * inv_r2;
                        let coeff =
                            -1.5 * p.j2 * mu * p.equatorial_radius * p.equatorial_radius * inv_r5;
                        accel += coeff * ((1.0 - 5.0 * s * s) * r_vec + (2.0 * z) * p.pole_axis);
                    }
                    let tau = if gravity_gradient {
                        let rot = Mat3::from_quat(poses[i].rotation);
                        let i_world = rot * inertias[i] * rot.transpose();
                        (3.0 * mu * inv_r3) * r_hat.cross(i_world * r_hat)
                    } else {
                        Vec3::ZERO
                    };
                    (accel, tau)
                }
                OrbitalField::CentralGravityZonal {
                    center,
                    mu,
                    min_radius,
                    zonal,
                    gravity_gradient,
                } => {
                    let r_vec = com - center;
                    let r_raw = r_vec.length();
                    let r = r_raw.max(min_radius);
                    let r_hat = if r_raw > 1e-24 {
                        r_vec / r_raw
                    } else {
                        Vec3::Y
                    };
                    let inv_r3 = 1.0 / (r * r * r);
                    let accel =
                        -mu * inv_r3 * r_vec + zonal.perturbation_acceleration(mu, r_vec, min_radius);
                    let tau = if gravity_gradient {
                        let rot = Mat3::from_quat(poses[i].rotation);
                        let i_world = rot * inertias[i] * rot.transpose();
                        (3.0 * mu * inv_r3) * r_hat.cross(i_world * r_hat)
                    } else {
                        Vec3::ZERO
                    };
                    (accel, tau)
                }
            };

            let force = accel * mass;
            wrenches[i].linear += force;
            wrenches[i].angular += torque;
            let entry = self.readings.entry(id).or_default();
            entry.acceleration += accel * weight;
            entry.force += force * weight;
            entry.torque += torque * weight;
        }
        Ok(())
    }
}
