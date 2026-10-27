use super::*;
use std::{io::Read, path::Path};

pub(super) struct Start {
    pub archive: MpmSoilTileArchive,
    pub checkpoint: MpmSoilCheckpoint,
    pub preceding_steps: u64,
    pub lineage: Value,
    pub source_files: Vec<(&'static str, Vec<u8>)>,
}

fn bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("settling resume input exceeds byte limit".into());
    }
    Ok(bytes)
}

pub(super) fn metadata(
    manifest: &Value,
    result: &Value,
    policy: MpmSoilSettlingConfig,
) -> Result<u64> {
    let version = match (manifest["schema"].as_str(), result["schema"].as_str()) {
        (Some("molla-soil-settling-v1"), Some("molla-soil-settling-result-v1")) => 1,
        (Some("molla-soil-settling-v2"), Some("molla-soil-settling-result-v2")) => 2,
        _ => return Err("unsupported settling resume schema".into()),
    };
    let steps = result["progress"]["submitted_substeps"]
        .as_u64()
        .ok_or("missing parent substeps")?;
    let start = if version == 1 {
        0
    } else {
        let start = result["segment_start_substeps"]
            .as_u64()
            .ok_or("missing parent start")?;
        if manifest["segment_start_substeps"].as_u64() != Some(start) {
            return Err("parent manifest/result start mismatch".into());
        }
        start
    };
    let end = start
        .checked_add(steps)
        .filter(|&n| n <= u64::from(u32::MAX))
        .ok_or("settling resume clock overflow")?;
    let status = result["progress"]["status"]
        .as_str()
        .ok_or("missing parent status")?;
    if manifest["fixture"] != "gravity-bed-8x8x8-v1"
        || manifest["particle_count"] != 512
        || manifest["policy"] != policy_json(policy)
        || result["workflow_completed"] != true
        || result["identity_mass_volume_material_preserved"] != true
        || !result["progress"]["failure"].is_null()
        || result["activity_converged"] != (status == "ActivityConverged")
        || !matches!(status, "ActivityConverged" | "StepLimit" | "WallLimit")
        || steps > policy.max_substeps
        || (status == "StepLimit" && steps != policy.max_substeps)
        || (status == "ActivityConverged"
            && (steps == 0
                || result["progress"]["quiet_seconds"]
                    .as_f64()
                    .is_none_or(|t| t < policy.quiet_seconds || !t.is_finite())
                || result["progress"]["quiet_observations"]
                    .as_u64()
                    .is_none_or(|n| n < u64::from(policy.quiet_observations))))
        || result["progress"]["observed_seconds"].as_f64()
            != Some(f64::from(policy.dt) * steps as f64)
        || result["progress"]["observations"].as_u64()
            != Some(steps.div_ceil(u64::from(policy.substeps_per_observation)))
        || (version == 2
            && (manifest["lineage"] != result["lineage"]
                || result["cumulative_substeps"].as_u64() != Some(end)
                || result["cumulative_seconds"].as_f64()
                    != Some(f64::from(policy.dt) * end as f64)))
    {
        return Err("incompatible, incomplete or inconsistent settling parent".into());
    }
    Ok(end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parent() -> (Value, Value, MpmSoilSettlingConfig) {
        let policy = workload().2;
        (
            json!({"schema":"molla-soil-settling-v1","fixture":"gravity-bed-8x8x8-v1",
            "particle_count":512,"policy":policy_json(policy)}),
            json!({"schema":"molla-soil-settling-result-v1","workflow_completed":true,
            "identity_mass_volume_material_preserved":true,"activity_converged":false,
            "progress":{"submitted_substeps":2000,"observed_seconds":2000.0*f64::from(policy.dt),
                "observations":125,"status":"StepLimit","failure":null}}),
            policy,
        )
    }

    #[test]
    fn resume_accepts_completed_v1_and_chained_v2_intervals() {
        let (mut m, mut r, p) = parent();
        assert_eq!(metadata(&m, &r, p).unwrap(), 2000);
        m["schema"] = json!("molla-soil-settling-v2");
        r["schema"] = json!("molla-soil-settling-result-v2");
        m["segment_start_substeps"] = json!(2000);
        r["segment_start_substeps"] = json!(2000);
        m["lineage"] = json!({"kind":"continued"});
        r["lineage"] = m["lineage"].clone();
        r["cumulative_substeps"] = json!(4000);
        r["cumulative_seconds"] = json!(4000.0 * f64::from(p.dt));
        assert_eq!(metadata(&m, &r, p).unwrap(), 4000);
        r["cumulative_substeps"] = json!(4001);
        assert!(metadata(&m, &r, p).is_err());
    }

    #[test]
    fn resume_rejects_changed_policy_failed_work_and_false_convergence() {
        let (m, r, p) = parent();
        for status in [
            "Running",
            "DrainingAfterWallLimit",
            "Failed",
            "Unknown",
            "ActivityConverged",
        ] {
            let mut bad = r.clone();
            bad["progress"]["status"] = json!(status);
            bad["activity_converged"] = json!(status == "ActivityConverged");
            assert!(metadata(&m, &bad, p).is_err());
        }
        let mut bad = m.clone();
        bad["policy"]["limits"]["speed_m_s"] = json!(0.1);
        assert!(metadata(&bad, &r, p).is_err());
        let mut bad = r.clone();
        bad["progress"]["observed_seconds"] = json!(1.0);
        assert!(metadata(&m, &bad, p).is_err());
        let mut bad = r.clone();
        bad["workflow_completed"] = json!(false);
        assert!(metadata(&m, &bad, p).is_err());
        let mut bad = r.clone();
        bad["progress"]["observations"] = json!(124);
        assert!(metadata(&m, &bad, p).is_err());
    }

    #[test]
    fn resume_reads_are_bounded_and_digest_mismatch_rejects_before_decode() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "molla-resume-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let (m, r, p) = parent();
        fs::write(dir.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
        fs::write(dir.join("result.json"), serde_json::to_vec(&r).unwrap()).unwrap();
        for file in [
            "final-archive.bin",
            "final-checkpoint.bin",
            "initial-checkpoint.bin",
        ] {
            fs::write(dir.join(file), [0_u8; 16]).unwrap();
        }
        assert_eq!(
            bounded(&dir.join("final-archive.bin"), 16).unwrap().len(),
            16
        );
        assert!(bounded(&dir.join("final-archive.bin"), 15).is_err());
        assert!(
            load(&dir, workload().0, p)
                .err()
                .unwrap()
                .to_string()
                .contains("digest mismatch")
        );
        fs::remove_dir_all(dir).unwrap();
    }
}

pub(super) fn load(path: &Path, config: MpmConfig, policy: MpmSoilSettlingConfig) -> Result<Start> {
    let manifest_bytes = bounded(&path.join("manifest.json"), 65536)?;
    let result_bytes = bounded(&path.join("result.json"), 131072)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let result: Value = serde_json::from_slice(&result_bytes)?;
    let preceding_steps = metadata(&manifest, &result, policy)?;
    let archive_bytes = bounded(&path.join("final-archive.bin"), 1048576)?;
    let checkpoint_bytes = bounded(&path.join("final-checkpoint.bin"), 1048576)?;
    let initial_bytes = bounded(&path.join("initial-checkpoint.bin"), 1048576)?;
    for (name, bytes) in [
        ("final_archive_blake3", &archive_bytes),
        ("final_checkpoint_blake3", &checkpoint_bytes),
        ("initial_checkpoint_blake3", &initial_bytes),
    ] {
        if result[name].as_str() != Some(blake3::hash(bytes).to_hex().as_str()) {
            return Err(format!("settling parent digest mismatch: {name}").into());
        }
    }
    let archive = MpmSoilTileArchive::from_bytes(&archive_bytes, 512)?;
    let selected = archive.select(&archive.tiles().collect::<Vec<_>>(), 512)?;
    let ids: Vec<_> = (0..512).map(SoilParticleId).collect();
    if archive.layout() != SoilTileLayout::new([0.0; 3], 0.125)?
        || selected.particle_ids() != ids
        || selected.material_ids() != [SoilMaterialId(1)]
        || selected.checkpoint().config() != config
        || !selected.checkpoint().materials().is_empty()
        || selected.checkpoint().to_bytes() != checkpoint_bytes
        || archive.to_bytes() != archive_bytes
    {
        return Err("settling parent changes fixture state/layout/identity".into());
    }
    let initial = MpmSoilCheckpoint::from_bytes(&initial_bytes, 512)?;
    selected
        .checkpoint()
        .activity_since(&initial, &ids, &ids, 1.0, 1)?;
    let lineage = json!({"kind":"continued","parent_result_blake3":blake3::hash(&result_bytes).to_hex().to_string(),
        "parent_archive_blake3":blake3::hash(&archive_bytes).to_hex().to_string(),
        "parent_manifest_blake3":blake3::hash(&manifest_bytes).to_hex().to_string(),
        "quiet_window":"reset at segment start; not carried across runs"});
    let checkpoint = selected.checkpoint().clone();
    Ok(Start {
        archive,
        checkpoint,
        preceding_steps,
        lineage,
        source_files: vec![
            ("parent-manifest.json", manifest_bytes),
            ("parent-result.json", result_bytes),
            ("parent-final-archive.bin", archive_bytes),
            ("parent-final-checkpoint.bin", checkpoint_bytes),
            ("parent-initial-checkpoint.bin", initial_bytes),
        ],
    })
}
