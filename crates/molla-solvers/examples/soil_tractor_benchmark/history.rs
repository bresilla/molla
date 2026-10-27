use super::*;
use molla_solvers::mpm::MpmSoilCheckpoint;

pub(super) const SCOPE: &str = "final committed particle state: position, velocity, APIC C, elastic F, reference volume, mass, log-Jp, material IDs/table and compaction/impulse diagnostics in slot order; excludes machine state, grid scratch, Newton history and world-tile IDs; blocking readback and file I/O after all measured frames, not a whole-world restart or performance acceptance";

pub(super) fn capture(
    tractor: &mut SoilTractor,
    output: &Path,
    expected_frame: u64,
    particles: u32,
) -> Result<Value, String> {
    let before = tractor.world().mpm_soil_stats().ok_or("soil missing")?;
    let started = Instant::now();
    let allocations = DeviceScope::start().map_err(|e| e.to_string())?;
    let transfers = TransferScope::start().map_err(|e| e.to_string())?;
    let dispatches = DispatchScope::start().map_err(|e| e.to_string())?;
    let capture = tractor.read_soil_checkpoint(particles);
    let dispatches = dispatches.finish();
    let transfers = transfers.finish();
    let allocations = allocations.finish();
    let readback_ms = started.elapsed().as_secs_f64() * 1000.0;
    let (frame, checkpoint) = capture.map_err(|e| e.to_string())?;
    if frame != expected_frame
        || checkpoint.particle_count() != particles
        || tractor.world().mpm_soil_stats() != Some(before)
        || transfers.upload_bytes != 0
        || transfers.readback_bytes != u64::from(particles) * 160 + 16
        || allocations.observations != 1
        || dispatches.encoded_dispatches != 0
        || transfers.overflowed
        || dispatches.overflowed
        || allocations.invalid
    {
        return Err(
            "final soil state identity, observation counters or publication changed".into(),
        );
    }
    let bytes = checkpoint.to_bytes();
    let decoded = MpmSoilCheckpoint::from_bytes(&bytes, particles).map_err(|e| e.to_string())?;
    if decoded.to_bytes() != bytes || decoded.particle_count() != particles {
        return Err("final soil checkpoint failed lossless roundtrip".into());
    }
    let name = "final-soil-state.bin";
    fs::write(output.join(name), &bytes).map_err(|e| e.to_string())?;
    Ok(json!({
        "schema": 1, "scope": SCOPE, "path": name, "committed_frame": frame,
        "particles": particles, "particle_payload_bytes": checkpoint.particle_bytes(),
        "encoded_bytes": bytes.len(), "blake3": blake3::hash(&bytes).to_hex().as_str(),
        "readback_ms_outside_timing": readback_ms,
        "readback_bytes_outside_timing": transfers.readback_bytes,
        "upload_bytes_outside_timing": transfers.upload_bytes,
        "allocation_observations_outside_timing": allocations.observations,
        "dispatches_outside_timing": dispatches.encoded_dispatches,
        "capture_and_write_ms_outside_timing": started.elapsed().as_secs_f64() * 1000.0,
        "lossless_decode_verified": true
    }))
}

#[test]
fn final_soil_state_option_is_explicit_and_strict() {
    let parse = |args: &[&str]| {
        parse_options(
            ["--output", "unused"]
                .into_iter()
                .chain(args.iter().copied())
                .map(str::to_owned),
        )
    };
    assert!(!parse(&[]).unwrap().final_soil_state);
    assert!(
        !parse(&["--final-soil-state", "0"])
            .unwrap()
            .final_soil_state
    );
    assert!(
        parse(&["--final-soil-state", "1"])
            .unwrap()
            .final_soil_state
    );
    for value in ["", "2", "true", "01", "+1", "1 "] {
        assert!(parse(&["--final-soil-state", value]).is_err());
    }
    assert!(parse(&["--final-soil-state"]).is_err());
    assert!(parse(&["--final-soil-state", "0", "--final-soil-state", "0"]).is_err());
}
