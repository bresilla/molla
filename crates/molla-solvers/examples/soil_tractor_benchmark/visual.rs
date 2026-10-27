use super::*;

pub fn frames(total: usize, every: usize) -> Result<Vec<usize>, String> {
    if total == 0 || total > 330 || every > 330 {
        return Err("visual capture range exceeds 330 frames".into());
    }
    if every == 0 {
        return Ok(Vec::new());
    }
    if total.div_ceil(every) + 1 > 64 {
        return Err("visual capture exceeds 64-snapshot budget".into());
    }
    Ok((0..=total)
        .filter(|frame| *frame == 0 || *frame == total || frame % every == 0)
        .collect())
}

pub fn machine(
    scene: &molla_sim::runtime::RigidScene,
    committed: u64,
    seconds: f64,
) -> Result<Value, String> {
    let mut bodies = Vec::new();
    for (body, _) in scene.bodies() {
        let pose = scene.body_pose(body).map_err(|e| e.to_string())?;
        let velocity = scene.body_velocity(body).map_err(|e| e.to_string())?;
        bodies.push(json!({"body": format!("{body:?}"), "position": pose.position.to_array(),
            "rotation": pose.rotation.to_array(), "linear_velocity": velocity.linear.to_array(), "angular_velocity": velocity.angular.to_array()}));
    }
    Ok(
        json!({"frame": committed.checked_sub(1), "committed_frame": committed, "simulated_seconds": seconds,
        "bodies": bodies, "joint_q": scene.state().joint_q.host().map_err(|e| e.to_string())?,
        "joint_qd": scene.state().joint_qd.host().map_err(|e| e.to_string())?}),
    )
}

pub fn capture(
    tractor: &mut SoilTractor,
    output: &Path,
    state: &Value,
) -> Result<(Value, Vec<[f32; 3]>), String> {
    let frame = state["committed_frame"]
        .as_u64()
        .ok_or("visual frame missing")?;
    let started = Instant::now();
    let allocation_scope = DeviceScope::start().map_err(|e| e.to_string())?;
    let scope = TransferScope::start().map_err(|e| e.to_string())?;
    let positions = tractor.read_soil_positions().map_err(|e| e.to_string())?;
    let transfers = scope.finish();
    let allocations = allocation_scope.finish();
    let observation_ms = started.elapsed().as_secs_f64() * 1000.0;
    if allocations.invalid
        || transfers.overflowed
        || positions.len() != 50000
        || positions.iter().flatten().any(|v| !v.is_finite())
    {
        return Err("invalid visual particle observation".into());
    }
    let particles = format!("visual-particles-{frame:06}.bin");
    let machine = format!("visual-machine-{frame:06}.json");
    let bytes: Vec<_> = positions
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    File::create_new(output.join(&particles))
        .and_then(|mut file| file.write_all(&bytes))
        .map_err(|e| e.to_string())?;
    File::create_new(output.join(&machine))
        .and_then(|mut file| file.write_all(&serde_json::to_vec(state).unwrap()))
        .map_err(|e| e.to_string())?;
    Ok((
        json!({"committed_frame": frame, "simulated_seconds": state["simulated_seconds"],
        "particles_file": particles, "machine_file": machine, "particles": positions.len(),
        "particle_file_bytes": bytes.len(), "readback_bytes_outside_timing": transfers.readback_bytes,
            "upload_bytes_outside_timing": transfers.upload_bytes,
            "buffer_allocation_observations_outside_timing": allocations.observations,
            "readback_wall_ms_outside_timing": observation_ms,
            "capture_wall_ms_outside_timing": started.elapsed().as_secs_f64() * 1000.0}),
        positions,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capture_schedule_is_bounded_and_keeps_endpoints_once() {
        assert_eq!(frames(33, 11).unwrap(), [0, 11, 22, 33]);
        assert_eq!(frames(33, 12).unwrap(), [0, 12, 24, 33]);
        assert!(frames(33, 0).unwrap().is_empty());
        assert_eq!(frames(3, 1).unwrap(), [0, 1, 2, 3]);
        assert_eq!(frames(3, 330).unwrap(), [0, 3]);
        assert_eq!(frames(63, 1).unwrap().len(), 64);
        for (total, every) in [(64, 1), (330, 1), (0, 1), (331, 1), (33, 331)] {
            assert!(frames(total, every).is_err());
        }
    }
}
