//! Velocity drives solved against every force of the step.

use molla_core::{Error, Result};
use molla_math::Real;
use molla_sim::{Control, JointTargetMode, Model, State};

use crate::featherstone::JointEffort;

/// Acceleration-model drives scale by the joint's own inertia; only force
/// drives are coupled here.
const ACCELERATION_MODEL: u32 = 1;

/// Solve force-model velocity drives against articulation inverse mass and force caps.
/// `(K⁻¹ + dt·G)·τ = v* − v_pred + dt·G·τ₀`.
/// Braked and acceleration-model axes retain their per-joint estimates.
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
    let responses = joint_mass.drive_responses(articulation, count, &units)?;
    let qd = state.joint_qd.host()?;
    joint_mass.project_drive_acceleration(
        articulation,
        &qd[first..first + count],
        &mut acceleration[first..first + count],
        dt,
    )?;
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
    let bounds: Vec<_> = drives
        .iter()
        .map(|&dof| {
            let limit = limits.get(dof).copied().unwrap_or(Real::INFINITY).abs();
            [-limit, limit]
        })
        .collect();
    let torques = crate::loop_forces::bounded_feedback_solve(&matrix, &rhs, &bounds)?;
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
    if acceleration[first..first + count]
        .iter()
        .any(|a| !a.is_finite())
    {
        return Err(Error::Solver("non-finite velocity drive reaction".into()));
    }
    Ok(())
}
