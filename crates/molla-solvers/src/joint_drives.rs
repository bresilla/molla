//! Velocity drives solved against every force of the step.

use molla_core::{Error, Result};
use molla_math::Real;
use molla_sim::{Control, JointTargetMode, Model, State};

use crate::featherstone::JointEffort;

/// Acceleration-model drives scale by the joint's own inertia; only force
/// drives are coupled here.
const ACCELERATION_MODEL: u32 = 1;
const SWEEPS: usize = 64;

/// A force-model velocity drive is the implicit damper `τ = kd·(v* − v⁺)`.
/// Its joint's own inertia leaves out gravity, loads carried through
/// compliant and loop joints, and the other drives, so under load a drive
/// stalls far below its force limit. Here `v⁺` is the rate the step
/// predicts with every force in it, and the drives' torques reach every
/// joint through the articulation's inverse mass:
/// `(K⁻¹ + dt·G)·τ = v* − v_pred + dt·G·τ₀`, each `τ` within its limit.
/// Joints under a brake keep the per-joint estimate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn react_velocity_drives(
    model: &Model,
    control: &Control,
    state: &State,
    articulation: usize,
    acceleration: &mut [Real],
    efforts: &mut [JointEffort],
    dt: Real,
    joint_mass: &mut crate::joint_mass::JointMass<'_>,
) -> Result<()> {
    let starts = model.articulation_start.host()?;
    let (begin, end) = if model.articulation_count == 0 {
        (0, model.joint_count)
    } else {
        (
            starts[articulation] as usize,
            starts[articulation + 1] as usize,
        )
    };
    let offsets = model.joint_dof_offset.host()?;
    let first = offsets[begin] as usize;
    let count = (offsets[end] - offsets[begin]) as usize;
    let modes = control.joint_target_mode.host()?;
    let gains = control.joint_target_kd.host()?;
    let targets = control.joint_target_vel.host()?;
    let limits = control.joint_motor_max_force.host()?;
    let models = control.joint_motor_model.host()?;
    let brakes = control.joint_damping.host()?;
    let drives: Vec<usize> = (first..first + count)
        .filter(|&dof| {
            modes.get(dof) == Some(&(JointTargetMode::Velocity as u32))
                && models.get(dof) != Some(&ACCELERATION_MODEL)
                && gains.get(dof).is_some_and(|&kd| kd > 0.0)
                && brakes.get(dof).is_none_or(|&c| c <= 0.0)
        })
        .collect();
    if drives.is_empty() || dt <= 0.0 {
        return Ok(());
    }
    let _span = tracing::debug_span!("velocity_drives", drives = drives.len()).entered();
    let units: Vec<_> = drives
        .iter()
        .map(|&dof| {
            let mut unit = vec![0.0; count];
            unit[dof - first] = 1.0;
            unit
        })
        .collect();
    let responses = joint_mass.factor(articulation, count)?.solve_batch(&units)?;
    let qd = state.joint_qd.host()?;
    let n = drives.len();
    let coupling = |i: usize, j: usize| {
        0.5 * (responses[j][drives[i] - first] + responses[i][drives[j] - first])
    };
    let applied: Vec<Real> = drives.iter().map(|&dof| efforts[dof].drive).collect();
    let mut matrix = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        let dof = drives[i];
        for j in 0..n {
            matrix[i * n + j] = dt * coupling(i, j);
        }
        matrix[i * n + i] += 1.0 / gains[dof];
        let predicted = qd[dof] + dt * acceleration[dof];
        rhs[i] = targets[dof] - predicted
            + dt * (0..n).map(|j| coupling(i, j) * applied[j]).sum::<Real>();
    }
    // Projected Gauss–Seidel over the box of force limits, from the torques
    // the joints' own estimates applied.
    let mut torques = applied.clone();
    for _ in 0..SWEEPS {
        let mut change: Real = 0.0;
        for i in 0..n {
            let diagonal = matrix[i * n + i];
            if diagonal <= 0.0 {
                continue;
            }
            let residual = rhs[i] - (0..n).map(|j| matrix[i * n + j] * torques[j]).sum::<Real>();
            let limit = limits.get(drives[i]).copied().unwrap_or(Real::INFINITY).abs();
            let next = (torques[i] + residual / diagonal).clamp(-limit, limit);
            change = change.max((next - torques[i]).abs());
            torques[i] = next;
        }
        if change <= 1e-9 * (1.0 + torques.iter().fold(0.0, |m: Real, t| m.max(t.abs()))) {
            break;
        }
    }
    for (i, &dof) in drives.iter().enumerate() {
        let delta = torques[i] - applied[i];
        if delta == 0.0 {
            continue;
        }
        for (k, response) in responses[i].iter().enumerate() {
            acceleration[first + k] += delta * response;
        }
        efforts[dof].drive = torques[i];
    }
    if acceleration[first..first + count].iter().any(|a| !a.is_finite()) {
        return Err(Error::Solver("non-finite velocity drive reaction".into()));
    }
    Ok(())
}
