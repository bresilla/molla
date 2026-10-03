use molla_core::{Error, Result};
use molla_solvers::rigid_world::RigidWorld;

pub fn check_frame(world: &RigidWorld, expected_steps: usize) -> Result<()> {
    if !world.quarantined_bodies().is_empty() {
        return Err(Error::Solver("benchmark frame quarantined a body".into()));
    }
    let expected = u64::try_from(expected_steps)
        .map_err(|_| Error::Solver("benchmark step count exceeds runtime capacity".into()))?;
    if world.completed_steps() != expected {
        return Err(Error::Solver(format!(
            "benchmark completed {} steps, expected {expected}",
            world.completed_steps()
        )));
    }
    Ok(())
}

pub fn check_final_state(world: &RigidWorld) -> Result<()> {
    let state = world.scene.state();
    let poses = state
        .body_q
        .host()?
        .iter()
        .all(|p| p.position.is_finite() && p.rotation.is_finite());
    let velocities = state
        .body_qd
        .host()?
        .iter()
        .all(|v| v.linear.is_finite() && v.angular.is_finite());
    let forces = state
        .body_f
        .host()?
        .iter()
        .all(|f| f.linear.is_finite() && f.angular.is_finite());
    let coordinates = state.joint_q.host()?.iter().all(|q| q.is_finite());
    let rates = state.joint_qd.host()?.iter().all(|v| v.is_finite());
    if !(poses && velocities && forces && coordinates && rates) {
        return Err(Error::Solver("benchmark final state is nonfinite".into()));
    }
    Ok(())
}
