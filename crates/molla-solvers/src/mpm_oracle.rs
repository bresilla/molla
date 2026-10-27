//! CPU oracle helpers for small MPM coupling formulas.
//!
//! These functions deliberately avoid `wgpu` so default CPU tests can
//! verify the math behind GPU-only MPM kernels. They are not a CPU MPM
//! simulator; they are reduced, inspectable formulas for Tier-1 parity
//! checks.

use molla_math::Vec3;

/// One grid-cell sample used by the GPU MPM rigid-imprint coupling
/// kernel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MpmRigidCellSample {
    /// World-space center of the grid cell.
    pub cell_center: Vec3,
    /// Existing sand mass in the cell before rigid imprint.
    pub sand_mass: f64,
    /// Existing sand momentum in the cell before rigid imprint.
    pub sand_momentum: Vec3,
    /// Rigid sphere center.
    pub rigid_center: Vec3,
    /// Rigid sphere linear velocity at this cell.
    pub rigid_velocity: Vec3,
}

/// Rigid-body state used to derive the velocity seen by an MPM grid
/// cell. The full GPU path computes this per cell before applying the
/// same imprint impulse formula as [`rigid_cell_imprint_wrench`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MpmRigidBodySample {
    /// Rigid body's world-space center of mass.
    pub rigid_center: Vec3,
    /// Rigid body's center-of-mass linear velocity.
    pub linear_velocity: Vec3,
    /// Rigid body's world-space angular velocity.
    pub angular_velocity: Vec3,
}

/// Force/torque contribution on one rigid body from one MPM grid cell.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RigidCellWrench {
    pub force: Vec3,
    pub torque: Vec3,
}

impl RigidCellWrench {
    pub const ZERO: Self = Self {
        force: Vec3::ZERO,
        torque: Vec3::ZERO,
    };
}

/// CPU oracle for the `rigid_imprint` wrench formula in
/// `shaders/mpm.wgsl`.
///
/// Before imprint, the cell contains sand momentum `p_old`. Imprint
/// gives the sand mass the rigid velocity, so the impulse delivered to
/// sand is
///
/// ```text
/// J_sand = m_sand * v_rigid - p_old
/// ```
///
/// By Newton's third law, the impulse on the rigid is `-J_sand`, so
/// the rigid force contribution is `-J_sand / dt`.
pub fn rigid_cell_imprint_wrench(sample: MpmRigidCellSample, dt: f64) -> RigidCellWrench {
    if sample.sand_mass <= 1.0e-12 || dt <= 0.0 {
        return RigidCellWrench::ZERO;
    }
    let impulse_to_sand = sample.rigid_velocity * sample.sand_mass - sample.sand_momentum;
    let force_on_rigid = -impulse_to_sand / dt;
    let arm = sample.cell_center - sample.rigid_center;
    RigidCellWrench {
        force: force_on_rigid,
        torque: arm.cross(force_on_rigid),
    }
}

/// Velocity of a rigid body at a world-space cell center.
///
/// ```text
/// v_cell = v_com + ω × (x_cell − x_com)
/// ```
#[must_use]
pub fn rigid_body_velocity_at_cell(body: MpmRigidBodySample, cell_center: Vec3) -> Vec3 {
    body.linear_velocity + body.angular_velocity.cross(cell_center - body.rigid_center)
}

/// Build the reduced cell sample for one grid cell by evaluating the
/// rigid body's linear + angular velocity at that cell.
#[must_use]
pub fn rigid_body_cell_sample(
    body: MpmRigidBodySample,
    cell_center: Vec3,
    sand_mass: f64,
    sand_momentum: Vec3,
) -> MpmRigidCellSample {
    MpmRigidCellSample {
        cell_center,
        sand_mass,
        sand_momentum,
        rigid_center: body.rigid_center,
        rigid_velocity: rigid_body_velocity_at_cell(body, cell_center),
    }
}

/// Sum a batch of reduced rigid-imprint cell samples into one net
/// force/torque wrench.
#[must_use]
pub fn aggregate_rigid_cell_wrench(samples: &[MpmRigidCellSample], dt: f64) -> RigidCellWrench {
    let mut out = RigidCellWrench::ZERO;
    for sample in samples {
        let wrench = rigid_cell_imprint_wrench(*sample, dt);
        out.force += wrench.force;
        out.torque += wrench.torque;
    }
    out
}

/// Output of the reduced viscous-rheology oracle.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ViscousDampingStep {
    pub velocity: Vec3,
    pub affine: [Vec3; 3],
    pub velocity_factor: f64,
    pub affine_factor: f64,
}

/// CPU oracle for the viscous MPM shader branch in `g2p`.
///
/// The GPU path computes:
///
/// ```text
/// velocity *= exp(-velocity_damping * dt)
/// C        *= exp(-affine_damping * dt)
/// ```
pub fn viscous_damping_step(
    velocity: Vec3,
    affine: [Vec3; 3],
    dt: f64,
    velocity_damping: f64,
    affine_damping: f64,
) -> ViscousDampingStep {
    let velocity_factor = (-velocity_damping.max(0.0) * dt.max(0.0)).exp();
    let affine_factor = (-affine_damping.max(0.0) * dt.max(0.0)).exp();
    ViscousDampingStep {
        velocity: velocity * velocity_factor,
        affine: [
            affine[0] * affine_factor,
            affine[1] * affine_factor,
            affine[2] * affine_factor,
        ],
        velocity_factor,
        affine_factor,
    }
}

/// Output of the reduced snow plasticity oracle.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SnowClampStep {
    /// Elastic singular values after the Stomakhin envelope clamp.
    pub singular_values: Vec3,
    /// Updated `log(det Fp)` hardening accumulator.
    pub jp: f64,
}

/// CPU oracle for the snow singular-value clamp used by the MPM
/// `plasticity` kernel.
///
/// This is the scalar part of Stomakhin snow plasticity: clamp the
/// elastic singular values to `[1 - theta_c, 1 + theta_s]`, then add
/// `log(det F_old) - log(det F_clamped)` to `jp`.
pub fn snow_clamp_singular_values(
    singular_values: Vec3,
    old_jp: f64,
    theta_c: f64,
    theta_s: f64,
) -> SnowClampStep {
    let lo = 1.0 - theta_c;
    let hi = 1.0 + theta_s;
    let clamped = Vec3::new(
        singular_values.x.clamp(lo, hi),
        singular_values.y.clamp(lo, hi),
        singular_values.z.clamp(lo, hi),
    );
    let det_old = (singular_values.x * singular_values.y * singular_values.z).max(1.0e-6);
    let det_new = (clamped.x * clamped.y * clamped.z).max(1.0e-6);
    let dlog = det_old.ln() - det_new.ln();
    SnowClampStep {
        singular_values: clamped,
        jp: (old_jp + dlog).clamp(-5.0, 5.0),
    }
}

/// Output of the reduced Drucker-Prager sand plasticity oracle.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SandProjectStep {
    /// Elastic singular values after projection to the DP cone.
    pub singular_values: Vec3,
    /// Updated scalar plastic-flow accumulator.
    pub jp: f64,
}

/// Reduced MPM material branch used by default-lane CPU oracle tests.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ReducedMpmMaterial {
    /// Drucker-Prager sand projection on elastic singular values.
    Sand {
        youngs: f64,
        poisson: f64,
        friction_angle_rad: f64,
    },
    /// Stomakhin snow singular-value clamp.
    Snow { theta_c: f64, theta_s: f64 },
    /// Viscous velocity/affine damping.
    Viscous {
        velocity_damping: f64,
        affine_damping: f64,
    },
}

/// Minimal per-particle state consumed by [`reduced_material_update`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReducedMpmMaterialState {
    pub singular_values: Vec3,
    pub velocity: Vec3,
    pub affine: [Vec3; 3],
    pub jp: f64,
}

/// Minimal per-particle state produced by [`reduced_material_update`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReducedMpmMaterialStep {
    pub singular_values: Vec3,
    pub velocity: Vec3,
    pub affine: [Vec3; 3],
    pub jp: f64,
}

/// Host-side Drucker-Prager alpha used by the MPM shader.
pub fn drucker_prager_alpha(friction_angle_rad: f64) -> f64 {
    let phi = friction_angle_rad.max(1.0_f64.to_radians());
    let sin_phi = phi.sin();
    (2.0_f64 / 3.0).sqrt() * (2.0 * sin_phi) / (3.0 - sin_phi)
}

/// Dispatch a single reduced material update for CPU oracle tests.
///
/// This is intentionally not a CPU MPM simulator. It is the smallest
/// branch-level oracle needed to exercise multi-material material logic in the
/// default test lane: sand and snow update the elastic singular values / `jp`,
/// while viscous material damps velocity and affine momentum.
pub fn reduced_material_update(
    material: ReducedMpmMaterial,
    state: ReducedMpmMaterialState,
    dt: f64,
) -> ReducedMpmMaterialStep {
    match material {
        ReducedMpmMaterial::Sand {
            youngs,
            poisson,
            friction_angle_rad,
        } => {
            let step = sand_project_singular_values(
                state.singular_values,
                state.jp,
                youngs,
                poisson,
                friction_angle_rad,
            );
            ReducedMpmMaterialStep {
                singular_values: step.singular_values,
                velocity: state.velocity,
                affine: state.affine,
                jp: step.jp,
            }
        }
        ReducedMpmMaterial::Snow { theta_c, theta_s } => {
            let step =
                snow_clamp_singular_values(state.singular_values, state.jp, theta_c, theta_s);
            ReducedMpmMaterialStep {
                singular_values: step.singular_values,
                velocity: state.velocity,
                affine: state.affine,
                jp: step.jp,
            }
        }
        ReducedMpmMaterial::Viscous {
            velocity_damping,
            affine_damping,
        } => {
            let step = viscous_damping_step(
                state.velocity,
                state.affine,
                dt,
                velocity_damping,
                affine_damping,
            );
            ReducedMpmMaterialStep {
                singular_values: state.singular_values,
                velocity: step.velocity,
                affine: step.affine,
                jp: state.jp,
            }
        }
    }
}

/// CPU oracle for the sand Drucker-Prager singular-value projection
/// used by the MPM `plasticity` kernel.
///
/// This mirrors the shader's log-strain cone projection and `jp`
/// accumulation, but only for a single particle's singular values.
pub fn sand_project_singular_values(
    singular_values: Vec3,
    old_jp: f64,
    youngs: f64,
    poisson: f64,
    friction_angle_rad: f64,
) -> SandProjectStep {
    let mu = youngs / (2.0 * (1.0 + poisson));
    let lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let d = 3.0;
    let alpha = drucker_prager_alpha(friction_angle_rad);
    let eps_log = Vec3::new(
        singular_values.x.max(1.0e-6).ln(),
        singular_values.y.max(1.0e-6).ln(),
        singular_values.z.max(1.0e-6).ln(),
    );
    let trace_eps = eps_log.x + eps_log.y + eps_log.z;
    let mean_eps = trace_eps / d;
    let dev_eps = eps_log - Vec3::splat(mean_eps);
    let dev_norm = dev_eps.length();
    let kappa = (d * lam + 2.0 * mu) / (2.0 * mu);

    let (projected, dq) = if trace_eps > 0.0 {
        (Vec3::ONE, eps_log.length())
    } else if dev_norm == 0.0 {
        (singular_values, 0.0)
    } else {
        let f_yield = dev_norm + kappa * trace_eps * alpha;
        if f_yield <= 0.0 {
            (singular_values, 0.0)
        } else {
            let r_cone = (-alpha * kappa * trace_eps).max(0.0);
            let new_dev = dev_eps * (r_cone / dev_norm);
            let new_eps = new_dev + Vec3::splat(mean_eps);
            (
                Vec3::new(new_eps.x.exp(), new_eps.y.exp(), new_eps.z.exp()),
                f_yield,
            )
        }
    };

    SandProjectStep {
        singular_values: projected,
        jp: (old_jp + dq).clamp(0.0, 5.0),
    }
}
