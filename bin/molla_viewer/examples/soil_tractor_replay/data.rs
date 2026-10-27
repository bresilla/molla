use serde_json::{Value, json};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

#[derive(Clone, Debug)]
pub enum Shape {
    Box([f32; 3]),
    Cylinder { radius: f32, half_width: f32 },
}

#[derive(Clone, Debug)]
pub struct Body {
    pub role: String,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub shape: Shape,
}

pub struct Snapshot {
    pub bodies: Vec<Body>,
    pub particles: Vec<[f32; 3]>,
    pub seconds: f64,
    pub frame: u64,
    pub hashes: BTreeMap<String, String>,
}

fn read(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > max {
        return Err(format!("{} exceeds input bound", path.display()));
    }
    Ok(bytes)
}

fn number(value: &Value) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| "expected finite number".into())
}

fn array<const N: usize>(value: &Value) -> Result<[f32; N], String> {
    let values = value
        .as_array()
        .filter(|a| a.len() == N)
        .ok_or("invalid vector width")?;
    let mut output = [0.0; N];
    for (dst, value) in output.iter_mut().zip(values) {
        let value = number(value)?;
        if value.abs() > 1e6 {
            return Err("coordinate exceeds replay range".into());
        }
        *dst = value as f32;
    }
    Ok(output)
}

fn integer(value: &Value) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| "expected nonnegative integer".into())
}

impl Snapshot {
    #[cfg(test)]
    pub fn load(run: &Path) -> Result<Self, String> {
        Self::load_frame(run, None)
    }

    pub fn load_frame(run: &Path, selected_frame: Option<u64>) -> Result<Self, String> {
        let mut hashes = BTreeMap::new();
        let mut input = |name: &str, limit| -> Result<Vec<u8>, String> {
            let bytes = read(&run.join(name), limit)?;
            hashes.insert(name.into(), blake3::hash(&bytes).to_hex().to_string());
            Ok(bytes)
        };
        for name in ["run.exit", "source-check.exit"] {
            if input(name, 32)?.as_slice() != b"0\n" {
                return Err(format!("{name} is not a successful run"));
            }
        }
        let manifest: Value = serde_json::from_slice(&input("data/manifest.json", 65536)?)
            .map_err(|e| e.to_string())?;
        let result: Value = serde_json::from_slice(&input("data/result.json", 262144)?)
            .map_err(|e| e.to_string())?;
        let geometry: Value = serde_json::from_slice(&input("data/render-geometry.json", 65536)?)
            .map_err(|e| e.to_string())?;
        if manifest["schema"] != 1
            || result["schema"] != 1
            || geometry["schema"] != 1
            || manifest["case"] != "native-tractor-50k"
            || result["status"] != "diagnostic_complete"
            || manifest["physical_calibration"] != "uncalibrated"
            || result["physical_calibration"] != "uncalibrated"
        {
            return Err("unsupported or incomplete uncalibrated tractor run".into());
        }
        let count = integer(&manifest["particles"])?;
        if count != 50000 || integer(&result["final_particles"])? != count {
            return Err("particle count mismatch or capacity exceeded".into());
        }
        let final_frame = integer(&result["final_committed_frame"])?;
        let expected_frames = integer(&manifest["warmup_frames"])?
            .checked_add(integer(&manifest["measured_frames"])?)
            .ok_or("frame count overflow")?;
        if final_frame == 0 || final_frame > 10000 || final_frame != expected_frames {
            return Err("inconsistent frame count".into());
        }
        let final_seconds = number(&result["total_simulated_seconds"])?;
        let dt = number(&manifest["frame_simulated_seconds"])?;
        let expected_seconds = final_frame as f64 * dt;
        if final_seconds <= 0.0
            || (final_seconds - expected_seconds).abs() > 1e-9 * final_seconds.max(1.0)
        {
            return Err("inconsistent simulated time".into());
        }
        let states = input("data/machine-states.jsonl", 32 * 1024 * 1024)?;
        let states = std::str::from_utf8(&states).map_err(|e| e.to_string())?;
        let frame = selected_frame.unwrap_or(final_frame);
        if frame > final_frame {
            return Err("requested frame exceeds completed run".into());
        }
        let mut final_state = Value::Null;
        let mut rows = 0;
        for (index, line) in states.lines().enumerate() {
            let state: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
            if state["frame"] != index as u64 || state["committed_frame"] != index as u64 + 1 {
                return Err("machine frame sequence mismatch".into());
            }
            rows += 1;
            if index as u64 + 1 == frame {
                final_state = state;
            }
        }
        if rows != final_frame {
            return Err("missing machine frames".into());
        }
        let mut particles_file = "data/final-particles.bin".to_owned();
        let seconds = frame as f64 * dt;
        if selected_frame.is_some() {
            let captures: Value =
                serde_json::from_slice(&input("data/visual-captures.json", 65536)?)
                    .map_err(|e| e.to_string())?;
            if captures["schema"] != 1 {
                return Err("unsupported visual capture schema".into());
            }
            let records = captures["frames"]
                .as_array()
                .filter(|v| !v.is_empty() && v.len() <= 64)
                .ok_or("invalid visual capture count")?;
            let planned = manifest["visual_capture_frames"]
                .as_array()
                .ok_or("visual capture schedule missing")?;
            if planned.len() != records.len()
                || result["visual_capture_frames"] != manifest["visual_capture_frames"]
            {
                return Err("visual capture schedule mismatch".into());
            }
            let mut previous = None;
            let mut selected = None;
            for (record, plan) in records.iter().zip(planned) {
                let committed = integer(&record["committed_frame"])?;
                if committed != integer(plan)?
                    || committed > final_frame
                    || previous.is_some_and(|v| committed <= v)
                    || integer(&record["particles"])? != count
                    || integer(&record["particle_file_bytes"])? != count * 12
                    || (number(&record["simulated_seconds"])? - committed as f64 * dt).abs() > 1e-9
                {
                    return Err("invalid visual capture identity/time/count".into());
                }
                previous = Some(committed);
                if record["particles_file"] != format!("visual-particles-{committed:06}.bin")
                    || record["machine_file"] != format!("visual-machine-{committed:06}.json")
                {
                    return Err("invalid visual capture file name".into());
                }
                if committed == frame {
                    selected = Some(record);
                }
            }
            if records[0]["committed_frame"] != 0
                || records.last().unwrap()["committed_frame"] != final_frame
            {
                return Err("visual capture endpoints missing".into());
            }
            let record = selected.ok_or("requested frame was not captured")?;
            particles_file = format!("data/visual-particles-{frame:06}.bin");
            let machine_file = format!("data/visual-machine-{frame:06}.json");
            let captured: Value =
                serde_json::from_slice(&input(&machine_file, 65536)?).map_err(|e| e.to_string())?;
            if captured["committed_frame"] != frame
                || (number(&captured["simulated_seconds"])? - number(&record["simulated_seconds"])?)
                    .abs()
                    > 1e-9
            {
                return Err("visual machine frame/time mismatch".into());
            }
            if frame != 0 && captured != final_state {
                return Err("visual machine differs from committed state".into());
            }
            final_state = captured;
        }
        let poses = final_state["bodies"]
            .as_array()
            .filter(|a| a.len() == 6)
            .ok_or("expected six machine bodies")?;
        let shapes = geometry["bodies"]
            .as_array()
            .filter(|a| a.len() == 6)
            .ok_or("expected six body shapes")?;
        let mut identities = std::collections::BTreeSet::new();
        let mut bodies = Vec::new();
        for record in shapes {
            let id = record["body"]
                .as_str()
                .ok_or("missing geometry body identity")?;
            if !identities.insert(id) {
                return Err("duplicate geometry body".into());
            }
            let matches: Vec<_> = poses.iter().filter(|p| p["body"] == id).collect();
            if matches.len() != 1 {
                return Err("geometry/pose identity mismatch".into());
            }
            let pose = matches[0];
            let rotation: [f32; 4] = array(&pose["rotation"])?;
            if (rotation.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() > 1e-3 {
                return Err("invalid body quaternion".into());
            }
            let role = record["role"].as_str().ok_or("missing body role")?;
            let shape = match record["shape"].as_str() {
                Some("box") if role == "chassis" || role == "tine" => {
                    let half = array(&record["half_extents"])?;
                    if half.iter().any(|&v| v <= 0.0 || v > 10.0) {
                        return Err("invalid box extents".into());
                    }
                    Shape::Box(half)
                }
                Some("cylinder-y") if role == "wheel" => {
                    let radius = number(&record["radius"])?;
                    let half_width = number(&record["half_width"])?;
                    if !(0.0 < radius && radius <= 10.0 && 0.0 < half_width && half_width <= 10.0) {
                        return Err("invalid wheel dimensions".into());
                    }
                    Shape::Cylinder {
                        radius: radius as f32,
                        half_width: half_width as f32,
                    }
                }
                _ => return Err("unsupported body shape or role".into()),
            };
            bodies.push(Body {
                role: role.into(),
                position: array(&pose["position"])?,
                rotation,
                shape,
            });
        }
        for (role, required) in [("chassis", 1), ("wheel", 4), ("tine", 1)] {
            if bodies.iter().filter(|b| b.role == role).count() != required {
                return Err("invalid tractor roles".into());
            }
        }
        let raw = input(&particles_file, count * 12)?;
        if raw.len() as u64 != count * 12 {
            return Err("truncated particle snapshot".into());
        }
        let mut particles = Vec::with_capacity(count as usize);
        for point in raw.chunks_exact(12) {
            let xyz = std::array::from_fn(|axis| {
                f32::from_le_bytes(point[axis * 4..axis * 4 + 4].try_into().unwrap())
            });
            if xyz.iter().any(|v| !v.is_finite() || v.abs() > 1e6) {
                return Err("invalid particle position".into());
            }
            particles.push(xyz);
        }
        Ok(Self {
            bodies,
            particles,
            seconds,
            frame,
            hashes,
        })
    }

    pub fn metadata(&self) -> Value {
        json!({"schema": 1, "scope": "offline recorded-state replay; no solver stepping or performance acceptance",
            "physical_calibration": "uncalibrated", "committed_frame": self.frame,
            "simulated_seconds": self.seconds, "particles": self.particles.len(), "bodies": self.bodies.len(),
            "geometry": "recorded analytic fixture shapes, not asset meshes",
            "material_points": "tetrahedral display markers, not physical grain radii or reconstructed terrain",
            "marker_radius_m": 0.018, "input_blake3": self.hashes})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "molla-soil-replay-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            std::fs::create_dir(path.join("data")).unwrap();
            let fixture = Self(path);
            for name in ["run.exit", "source-check.exit"] {
                std::fs::write(fixture.0.join(name), b"0\n").unwrap();
            }
            fixture.json("manifest.json", &json!({"schema": 1, "case": "native-tractor-50k", "physical_calibration": "uncalibrated", "particles": 50000, "warmup_frames": 0, "measured_frames": 1, "frame_simulated_seconds": 0.01}));
            fixture.json("result.json", &json!({"schema": 1, "status": "diagnostic_complete", "physical_calibration": "uncalibrated", "final_particles": 50000, "final_committed_frame": 1, "total_simulated_seconds": 0.01}));
            let roles = ["chassis", "wheel", "wheel", "wheel", "wheel", "tine"];
            let bodies: Vec<_> = roles.iter().enumerate().map(|(id, role)| {
                if *role == "wheel" { json!({"body": format!("b{id}"), "role": role, "shape": "cylinder-y", "radius": 0.45, "half_width": 0.14}) }
                else { json!({"body": format!("b{id}"), "role": role, "shape": "box", "half_extents": [0.1, 0.2, 0.3]}) }
            }).collect();
            fixture.json(
                "render-geometry.json",
                &json!({"schema": 1, "bodies": bodies}),
            );
            let bodies: Vec<_> = (0..6).map(|id| json!({"body": format!("b{id}"), "position": [0, 0, 0], "rotation": [0, 0, 0, 1]})).collect();
            fixture.json(
                "machine-states.jsonl",
                &json!({"frame": 0, "committed_frame": 1, "bodies": bodies}),
            );
            std::fs::write(
                fixture.0.join("data/final-particles.bin"),
                vec![0u8; 50000 * 12],
            )
            .unwrap();
            fixture
        }
        fn json(&self, name: &str, value: &Value) {
            std::fs::write(
                self.0.join("data").join(name),
                serde_json::to_vec(value).unwrap(),
            )
            .unwrap();
        }
        fn mutate(&self, name: &str, edit: impl FnOnce(&mut Value)) {
            let path = self.0.join("data").join(name);
            let mut value: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            edit(&mut value);
            self.json(name, &value);
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn accepts_exact_snapshot_and_records_input_identity() {
        let fixture = Fixture::new();
        let snapshot = Snapshot::load(&fixture.0).unwrap();
        assert_eq!(snapshot.particles.len(), 50000);
        assert_eq!(snapshot.bodies.len(), 6);
        assert_eq!(snapshot.frame, 1);
        assert_eq!(snapshot.seconds, 0.01);
        assert_eq!(snapshot.hashes.len(), 7);
        assert_eq!(snapshot.metadata()["physical_calibration"], "uncalibrated");
    }

    #[test]
    fn rejects_failed_run_bad_time_and_misaligned_frames() {
        for fault in 0..5 {
            let fixture = Fixture::new();
            match fault {
                0 => std::fs::write(fixture.0.join("run.exit"), "124\n").unwrap(),
                1 => fixture.mutate("result.json", |v| {
                    v["total_simulated_seconds"] = json!(0.02)
                }),
                2 => fixture.mutate("machine-states.jsonl", |v| v["committed_frame"] = json!(2)),
                3 => fixture.mutate("result.json", |v| v["final_particles"] = json!(49999)),
                _ => fixture.mutate("manifest.json", |v| {
                    v["warmup_frames"] = json!(u64::MAX);
                    v["measured_frames"] = json!(2);
                }),
            }
            assert!(Snapshot::load(&fixture.0).is_err(), "fault {fault}");
        }
    }

    #[test]
    fn rejects_ambiguous_geometry_and_invalid_particle_payload() {
        for fault in 0..5 {
            let fixture = Fixture::new();
            match fault {
                0 => fixture.mutate("render-geometry.json", |v| {
                    v["bodies"][1]["body"] = json!("b0")
                }),
                1 => fixture.mutate("machine-states.jsonl", |v| {
                    v["bodies"][0]["rotation"] = json!([0, 0, 0, 0])
                }),
                2 => fixture.mutate("render-geometry.json", |v| {
                    v["bodies"][1]["radius"] = json!(-1)
                }),
                3 => std::fs::write(fixture.0.join("data/final-particles.bin"), [0; 12]).unwrap(),
                _ => {
                    let mut bytes = vec![0u8; 50000 * 12];
                    bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
                    std::fs::write(fixture.0.join("data/final-particles.bin"), bytes).unwrap();
                }
            }
            assert!(Snapshot::load(&fixture.0).is_err(), "fault {fault}");
        }
    }

    fn temporal_fixture() -> Fixture {
        let fixture = Fixture::new();
        fixture.mutate("machine-states.jsonl", |v| {
            v["simulated_seconds"] = json!(0.01)
        });
        let state: Value = serde_json::from_slice(
            &std::fs::read(fixture.0.join("data/machine-states.jsonl")).unwrap(),
        )
        .unwrap();
        let mut initial = state.clone();
        initial["frame"] = Value::Null;
        initial["committed_frame"] = json!(0);
        initial["simulated_seconds"] = json!(0.0);
        fixture.json("visual-machine-000000.json", &initial);
        fixture.json("visual-machine-000001.json", &state);
        for frame in 0..=1 {
            std::fs::write(
                fixture
                    .0
                    .join(format!("data/visual-particles-{frame:06}.bin")),
                vec![0u8; 50000 * 12],
            )
            .unwrap();
        }
        for name in ["manifest.json", "result.json"] {
            fixture.mutate(name, |v| v["visual_capture_frames"] = json!([0, 1]));
        }
        let records: Vec<_> = (0..=1).map(|frame| json!({"committed_frame": frame, "simulated_seconds": frame as f64 * 0.01,
            "particles": 50000, "particle_file_bytes": 600000,
            "particles_file": format!("visual-particles-{frame:06}.bin"), "machine_file": format!("visual-machine-{frame:06}.json")})).collect();
        fixture.json(
            "visual-captures.json",
            &json!({"schema": 1, "frames": records}),
        );
        fixture
    }

    #[test]
    fn temporal_snapshots_preserve_initial_and_committed_identity() {
        let fixture = temporal_fixture();
        for frame in 0..=1 {
            let snapshot = Snapshot::load_frame(&fixture.0, Some(frame)).unwrap();
            assert_eq!(snapshot.frame, frame);
            assert_eq!(snapshot.seconds, frame as f64 * 0.01);
            assert_eq!(snapshot.particles.len(), 50000);
            assert!(
                snapshot
                    .hashes
                    .contains_key(&format!("data/visual-particles-{frame:06}.bin"))
            );
        }
        assert!(Snapshot::load_frame(&fixture.0, Some(2)).is_err());
        assert!(Snapshot::load_frame(&Fixture::new().0, Some(0)).is_err());
    }

    #[test]
    fn temporal_snapshots_reject_mixed_time_state_and_path_traversal() {
        for fault in 0..5 {
            let fixture = temporal_fixture();
            match fault {
                0 => fixture.mutate("visual-captures.json", |v| {
                    v["frames"][1]["particles_file"] = json!("../final-particles.bin")
                }),
                1 => fixture.mutate("visual-captures.json", |v| {
                    v["frames"][1]["simulated_seconds"] = json!(0.02)
                }),
                2 => fixture.mutate("visual-machine-000001.json", |v| {
                    v["bodies"][0]["position"] = json!([1, 0, 0])
                }),
                3 => fixture.mutate("result.json", |v| v["visual_capture_frames"] = json!([0])),
                _ => fixture.mutate("visual-captures.json", |v| {
                    v["frames"][0]["committed_frame"] = json!(1)
                }),
            }
            assert!(
                Snapshot::load_frame(&fixture.0, Some(1)).is_err(),
                "fault {fault}"
            );
        }
    }
}
