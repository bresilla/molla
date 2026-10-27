use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub type Result<T> = std::result::Result<T, String>;
const MODES: [&str; 3] = ["dense", "active-tiles", "compact-tiles"];
const CASES: [(u64, &str, u64); 8] = [
    (32, "patch", 0),
    (64, "patch", 0),
    (96, "patch", 0),
    (32, "patch", 3),
    (64, "patch", 3),
    (96, "patch", 3),
    (32, "filled", 0),
    (32, "filled", 3),
];
type Key = (u64, String, u64);

#[derive(Default)]
struct Workload {
    manifest: Option<Value>,
    samples: BTreeMap<(usize, u64), Value>,
    summaries: BTreeMap<usize, Value>,
}

#[derive(Debug)]
pub struct Checked {
    pub report: Value,
    pub csv: String,
    pub failures: Vec<String>,
}

pub fn invalid(error: String) -> Checked {
    let timeout =
        error.starts_with("benchmark exit 124:") || error.starts_with("benchmark exit 137:");
    Checked {
        report: json!({"schema":1,"checker":"active-domain-v1",
            "status":if timeout {"timeout"} else {"invalid-or-incomplete"},
            "performance_accepted":false,"complete_records":false,"error":error}),
        csv: String::new(),
        failures: vec![error],
    }
}

pub fn invalid_for_directory(error: String, dir: &Path) -> Checked {
    let mut checked = invalid(error);
    if dir.join("smoke.log").exists() && !dir.join("active-domain.log").exists() {
        checked.report["checker"] = json!("smoke-v1");
        checked.report["case"] = json!("smoke");
    }
    checked
}

pub fn write_outputs(checked: &Checked, output: &Path) -> Result<()> {
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let encoded = serde_json::to_string_pretty(&checked.report).map_err(|e| e.to_string())?;
    for (name, contents) in [
        ("checked-results.csv", checked.csv.as_str()),
        ("checked-results.json", encoded.as_str()),
    ] {
        let temporary = output.join(format!(".{name}.{}.tmp", std::process::id()));
        fs::write(&temporary, contents).map_err(|e| e.to_string())?;
        fs::rename(&temporary, output.join(name)).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}

fn uint(v: &Value, key: &str) -> Result<u64> {
    v[key]
        .as_u64()
        .ok_or_else(|| format!("{key}: expected unsigned integer"))
}

fn number(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .filter(|n| n.is_finite())
        .ok_or_else(|| format!("{key}: expected finite number"))
}

fn positive(v: &Value, key: &str) -> Result<f64> {
    let n = number(v, key)?;
    require(n > 0., format!("{key}: must be positive"))?;
    Ok(n)
}

fn flag(v: &Value, key: &str) -> Result<bool> {
    v[key]
        .as_bool()
        .ok_or_else(|| format!("{key}: expected boolean"))
}

fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{key}: expected nonempty string"))
}

fn near(a: f64, b: f64, label: &str) -> Result<()> {
    require(
        a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * b.abs().max(1e-12),
        format!("{label}: {a} != recomputed {b}"),
    )
}

fn key(v: &Value) -> Result<Key> {
    require(string(v, "case")? == "active-domain", "unsupported case")?;
    let key = (
        uint(v, "grid")?,
        string(v, "layout")?.to_owned(),
        uint(v, "jacobi_iters")?,
    );
    require(
        CASES.iter().any(|&(g, l, i)| key == (g, l.to_owned(), i)),
        format!("unknown workload {key:?}"),
    )?;
    Ok(key)
}

fn fields(v: &Value, expected: &str) -> Result<()> {
    let object = v.as_object().ok_or("record is not an object")?;
    let names: BTreeSet<_> = expected.split_whitespace().collect();
    require(
        object.len() == names.len() && object.keys().all(|k| names.contains(k.as_str())),
        format!("unexpected or missing fields in {} record", v["record"]),
    )
}

fn vector(v: &Value, key: &str) -> Result<()> {
    let a = v[key]
        .as_array()
        .ok_or_else(|| format!("{key}: expected vec3"))?;
    require(
        a.len() == 3 && a.iter().all(|n| n.as_f64().is_some_and(f64::is_finite)),
        format!("{key}: expected finite vec3"),
    )
}

fn validate_manifest(v: &Value) -> Result<()> {
    fields(
        v,
        "schema record case scope grid layout particles jacobi_iters dt substeps simulated_seconds cell_size origin particle_spacing particle_mass particle_volume youngs poisson gravity_y friction_angle cohesion boundary_friction initial_velocity rigids two_pass_p2g warmup_rounds sample_rounds reset_each_sample mode_count order compact_tile_capacity absolute_particle_tolerance instrumented",
    )?;
    require(uint(v, "schema")? == 1, "unsupported manifest schema")?;
    require(
        string(v, "scope")?
            == "solver-only diagnostic; reset, readback, raster, renderer and machine excluded",
        "unsupported timing scope",
    )?;
    let (_, layout, _) = key(v)?;
    require(
        uint(v, "particles")? == if layout == "filled" { 32768 } else { 50000 },
        "particle count differs from frozen workload",
    )?;
    require(
        uint(v, "rigids")? == if layout == "filled" { 0 } else { 2 },
        "rigid count differs from frozen workload",
    )?;
    require(
        uint(v, "mode_count")? == 3
            && uint(v, "sample_rounds")? == 12
            && uint(v, "warmup_rounds")? == 2,
        "expected two warmups and twelve rounds in three modes",
    )?;
    require(uint(v, "substeps")? == 4, "expected four physical substeps")?;
    require(
        uint(v, "compact_tile_capacity")? == if layout == "filled" { 512 } else { 256 },
        "unexpected compact capacity",
    )?;
    require(
        string(v, "order")? == "cyclic-rotation",
        "unsupported execution order",
    )?;
    for name in ["two_pass_p2g", "reset_each_sample", "instrumented"] {
        require(flag(v, name)?, format!("{name} must be enabled"))?;
    }
    for name in [
        "cell_size",
        "particle_spacing",
        "particle_mass",
        "particle_volume",
        "youngs",
    ] {
        positive(v, name)?;
    }
    for name in [
        "poisson",
        "gravity_y",
        "friction_angle",
        "cohesion",
        "boundary_friction",
    ] {
        number(v, name)?;
    }
    vector(v, "origin")?;
    vector(v, "initial_velocity")?;
    let dt = positive(v, "dt")? as f32;
    require(
        dt.is_finite() && dt > 0.,
        "dt cannot be represented by the solver",
    )?;
    near(
        positive(v, "simulated_seconds")?,
        f64::from(dt) * uint(v, "substeps")? as f64,
        "physical duration",
    )?;
    near(
        positive(v, "absolute_particle_tolerance")?,
        2e-5,
        "frozen particle tolerance",
    )
}

fn canonical(v: &Value) -> String {
    match v {
        Value::Object(object) => {
            let mut names: Vec<_> = object.keys().collect();
            names.sort();
            format!(
                "{{{}}}",
                names
                    .into_iter()
                    .map(|k| format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap(),
                        canonical(&object[k])
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(canonical).collect::<Vec<_>>().join(",")
        ),
        _ => v.to_string(),
    }
}

pub fn manifest_hash(v: &Value) -> String {
    format!("blake3:{}", blake3::hash(canonical(v).as_bytes()).to_hex())
}

pub fn check_log(log: &str, exit_code: i32) -> Result<Checked> {
    let mut workloads: BTreeMap<Key, Workload> = BTreeMap::new();
    let mut ordered = Vec::new();
    let mut failures = Vec::new();
    let mut failure_lines = 0;
    for (line_index, line) in log.lines().enumerate() {
        let line = line.trim();
        failure_lines += usize::from(line.starts_with("parity_failure:"));
        if !line.starts_with('{') {
            continue;
        }
        let v: Value =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", line_index + 1))?;
        let k = key(&v)?;
        let w = workloads.entry(k.clone()).or_default();
        match string(&v, "record")? {
            "manifest" => {
                validate_manifest(&v)?;
                require(w.manifest.is_none(), format!("duplicate manifest {k:?}"))?;
                w.manifest = Some(v);
            }
            "sample" => {
                fields(
                    &v,
                    "record case grid layout jacobi_iters mode sample wall_ms encode_submit_ms simulated_seconds allocation_observations readback_bytes upload_bytes",
                )?;
                require(w.manifest.is_some(), "sample precedes manifest")?;
                let mode = uint(&v, "mode")?;
                let index = uint(&v, "sample")?;
                require(mode < 3 && index < 12, "out-of-range sample identity")?;
                require(w.summaries.is_empty(), "sample follows summary")?;
                let wall = positive(&v, "wall_ms")?;
                let encode = number(&v, "encode_submit_ms")?;
                require(
                    encode >= 0. && encode <= wall,
                    "encode time exceeds completed wall time",
                )?;
                near(
                    positive(&v, "simulated_seconds")?,
                    positive(w.manifest.as_ref().unwrap(), "simulated_seconds")?,
                    "sample physical duration",
                )?;
                require(
                    uint(&v, "allocation_observations")? == 0 && uint(&v, "readback_bytes")? == 0,
                    "hot-path allocation/readback gate failed",
                )?;
                require(
                    uint(&v, "upload_bytes")? == if mode == 0 { 352 } else { 400 },
                    "uniform upload counter changed",
                )?;
                require(
                    w.samples.insert((mode as usize, index), v).is_none(),
                    "duplicate sample",
                )?;
                ordered.push((k, mode, index));
            }
            "summary" => {
                fields(
                    &v,
                    "schema record case grid layout particles jacobi_iters mode samples p50_ms p95_ms p99_ms max_ms simulated_seconds p50_wall_over_simulated active_tiles tile_capacity scheduled_cell_slots parity_passed dense_backing_storage grid_storage_bytes",
                )?;
                require(uint(&v, "schema")? == 1, "unsupported summary schema")?;
                let name = string(&v, "mode")?;
                let mode = MODES
                    .iter()
                    .position(|&s| s == name)
                    .ok_or("unknown summary mode")?;
                require(
                    w.manifest.is_some() && w.samples.len() == 36,
                    "summary precedes complete samples",
                )?;
                require(w.summaries.insert(mode, v).is_none(), "duplicate summary")?;
            }
            _ => return Err("unknown record type".into()),
        }
    }
    require(
        workloads.len() == CASES.len(),
        "incomplete workload set: expected all eight cases",
    )?;
    let expected_order: Vec<_> = CASES
        .iter()
        .flat_map(|&(g, l, i)| {
            (0..12).flat_map(move |sample| {
                (0..3).map(move |offset| ((g, l.to_owned(), i), (sample + 2 + offset) % 3, sample))
            })
        })
        .collect();
    require(
        ordered == expected_order,
        "sample order does not match balanced cyclic schedule",
    )?;
    let mut rows = Vec::new();
    let mut manifests = Vec::new();
    let mut csv = String::from(
        "grid,layout,jacobi_iters,mode,manifest_hash,samples,p50_ms,p95_ms,p99_ms,max_ms,simulated_seconds_per_sample,p50_wall_over_simulated,parity_passed\n",
    );
    for (k, w) in workloads {
        let manifest = w.manifest.ok_or("missing manifest")?;
        require(
            w.samples.len() == 36 && w.summaries.len() == 3,
            format!("incomplete samples/summaries {k:?}"),
        )?;
        let hash = manifest_hash(&manifest);
        let duration = positive(&manifest, "simulated_seconds")?;
        for (mode, summary) in w.summaries {
            require(
                uint(&summary, "samples")? == 12
                    && uint(&summary, "particles")? == uint(&manifest, "particles")?,
                "summary count mismatch",
            )?;
            let mut times: Vec<_> = (0..12)
                .map(|i| number(&w.samples[&(mode, i)], "wall_ms").unwrap())
                .collect();
            times.sort_by(f64::total_cmp);
            let p50 = times[5];
            let p95 = times[11];
            let p99 = times[11];
            for (label, computed) in [
                ("p50_ms", p50),
                ("p95_ms", p95),
                ("p99_ms", p99),
                ("max_ms", times[11]),
                ("simulated_seconds", duration),
                ("p50_wall_over_simulated", p50 / 1000. / duration),
            ] {
                near(positive(&summary, label)?, computed, label)?;
            }
            let active = uint(&summary, "active_tiles")?;
            let capacity = uint(&summary, "tile_capacity")?;
            let world_tiles = k.0.div_ceil(4).pow(3);
            let expected_capacity = if mode == 2 {
                uint(&manifest, "compact_tile_capacity")?
            } else {
                world_tiles
            };
            require(
                capacity == expected_capacity && active > 0 && active <= capacity,
                "tile capacity/overflow gate failed",
            )?;
            require(
                k.1 != "filled" || active == capacity,
                "filled workload missing active tiles",
            )?;
            require(
                uint(&summary, "scheduled_cell_slots")?
                    == if mode == 0 { k.0.pow(3) } else { active * 64 },
                "scheduled cell count mismatch",
            )?;
            require(
                flag(&summary, "dense_backing_storage")? == (mode != 2),
                "storage layout mismatch",
            )?;
            let expected_bytes = if mode == 2 {
                capacity * 64 * 56 + 16 + 8 * ((2 * capacity).next_power_of_two() + capacity)
            } else {
                k.0.pow(3) * 56 + 16 + 8 * world_tiles
            };
            require(
                uint(&summary, "grid_storage_bytes")? == expected_bytes,
                "grid storage byte count mismatch",
            )?;
            let parity = flag(&summary, "parity_passed")?;
            if !parity {
                failures.push(format!("physics parity failed: {k:?}/{}", MODES[mode]));
            }
            csv.push_str(&format!(
                "{},{},{},{},{},12,{p50},{p95},{p99},{p99},{duration},{},{}\n",
                k.0,
                k.1,
                k.2,
                MODES[mode],
                hash,
                p50 / 1000. / duration,
                parity
            ));
            let completed_wall_ms: f64 = times.iter().sum();
            require(
                completed_wall_ms.is_finite(),
                "completed wall time overflow",
            )?;
            rows.push(json!({"manifest_hash":hash,"summary":summary,
                "measured_simulated_seconds":duration*12.,
                "completed_wall_ms":completed_wall_ms,
                "samples":(0..12).map(|i| &w.samples[&(mode,i)]).collect::<Vec<_>>()}));
        }
        manifests.push(json!({"hash":hash,"manifest":manifest}));
    }
    if exit_code != 0 {
        failures.push(format!("benchmark exit {exit_code}"));
    }
    if failure_lines > 0 {
        failures.push(format!("{failure_lines} logged parity failures"));
    }
    let success = log
        .lines()
        .any(|l| l.starts_with("test result: ok. 1 passed; 0 failed;"));
    if !success {
        failures.push("missing successful one-test completion marker".into());
    }
    let status = if matches!(exit_code, 124 | 137) {
        "timeout"
    } else if failures.is_empty() {
        "diagnostic-pass"
    } else {
        "failed"
    };
    let report = json!({
        "schema":1,"checker":"active-domain-v1","status":status,"complete_records":true,
        "benchmark_exit":exit_code,"workloads":8,"sample_count":288,"summary_count":24,
        "performance_accepted":false,
        "acceptance_limits":["solver-only diagnostic", "unconfirmed absolute workload budget",
            "no in-run contention telemetry", "no uninstrumented control",
            "legacy manifest omits structured boundary/contact state and some physical diagnostics"],
        "failures":failures,"manifests":manifests,"results":rows,
    });
    Ok(Checked {
        report,
        csv,
        failures,
    })
}

fn read(path: &Path) -> Result<String> {
    let size = fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    require(
        size <= 32 * 1024 * 1024,
        format!("{} exceeds 32 MiB input cap", path.display()),
    )?;
    fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn hardware(csv: &str) -> Result<Value> {
    let mut lines = csv.lines();
    require(
        lines.next()
            == Some(
                "name, driver_version, utilization.gpu [%], memory.used [MiB], temperature.gpu, power.draw [W]",
            ),
        "unknown hardware snapshot schema",
    )?;
    let mut devices = Vec::new();
    for line in lines {
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        require(
            fields.len() == 6 && !fields[0].is_empty() && !fields[1].is_empty(),
            "invalid hardware identity",
        )?;
        let numeric = |index: usize, suffix: &str| -> Result<f64> {
            let text = fields[index]
                .strip_suffix(suffix)
                .ok_or("missing hardware unit")?
                .trim();
            let n: f64 = text.parse().map_err(|_| "invalid hardware metric")?;
            require(n.is_finite() && n >= 0., "invalid hardware metric")?;
            Ok(n)
        };
        let utilization = numeric(2, "%")?;
        require(utilization <= 100., "invalid GPU utilization")?;
        devices.push(json!({"name":fields[0],"driver":fields[1],"utilization_percent":utilization,
            "memory_mib":numeric(3,"MiB")?,"temperature_c":numeric(4,"")?,"power_w":numeric(5,"W")?}));
    }
    require(!devices.is_empty(), "empty hardware snapshot")?;
    Ok(Value::Array(devices))
}

pub fn check_directory(dir: &Path) -> Result<Checked> {
    let smoke = dir.join("smoke.log").exists();
    require(
        !(smoke && dir.join("active-domain.log").exists()),
        "ambiguous benchmark case",
    )?;
    let case = if smoke { "smoke" } else { "active-domain" };
    let log = read(&dir.join(format!("{case}.log")))?;
    let code: i32 = read(&dir.join(format!("{case}.exit")))?
        .trim()
        .parse()
        .map_err(|_| "invalid exit status")?;
    require((0..=255).contains(&code), "invalid exit status")?;
    let mut checked = if smoke {
        super::smoke::check_log(&log, code)
    } else {
        check_log(&log, code)
    }
    .map_err(|e| format!("benchmark exit {code}: {e}"))?;
    if smoke {
        require(
            read(&dir.join("source-check.exit"))?.trim() == "0",
            "source changed during smoke run",
        )?;
    }
    let revision = read(&dir.join("revision.txt"))?.trim().to_owned();
    require(
        revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid source revision",
    )?;
    let hashes = read(&dir.join("source-sha256.txt"))?;
    let mut sources = BTreeMap::new();
    for line in hashes.lines() {
        let (hash, path) = line.split_once("  ").ok_or("invalid source hash record")?;
        require(
            hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()) && !path.is_empty(),
            "invalid source hash",
        )?;
        require(
            sources.insert(path, hash).is_none(),
            "duplicate source hash path",
        )?;
    }
    for required in [
        "Makefile",
        "Cargo.lock",
        "crates/molla-solvers/src/mpm.rs",
        "crates/molla-solvers/tests/mpm_active_tiles_gpu.rs",
        "crates/molla-solvers/shaders/mpm.wgsl",
        "crates/molla-solvers/src/mpm_active_tiles.rs",
        "crates/molla-solvers/shaders/mpm_active_tiles.wgsl",
    ] {
        require(
            sources.contains_key(required),
            format!("missing source provenance: {required}"),
        )?;
    }
    if smoke {
        if checked.report["manifests"][0]["backend"] == "resident-dem" {
            for required in [
                "crates/molla-solvers/examples/soil_support/smoke_dem_gpu.rs",
                "crates/molla-kernels/src/tire_dem.rs",
                "crates/molla-kernels/src/soil_dem_resident.rs",
                "crates/molla-kernels/shaders/tire_dem.wgsl",
            ] {
                require(
                    sources.contains_key(required),
                    format!("missing DEM source provenance: {required}"),
                )?;
            }
        }
        for required in [
            "crates/molla-solvers/examples/soil_benchmark.rs",
            "crates/molla-solvers/examples/soil_support/smoke.rs",
            "crates/molla-solvers/examples/soil_support/smoke_gpu.rs",
        ] {
            require(
                sources.contains_key(required),
                format!("missing smoke source provenance: {required}"),
            )?;
        }
    }
    let before = hardware(&read(&dir.join("hardware-before.csv"))?)?;
    let after = hardware(&read(&dir.join("hardware-after.csv"))?)?;
    let identities = |v: &Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|d| (d["name"].clone(), d["driver"].clone()))
            .collect::<Vec<_>>()
    };
    require(
        identities(&before) == identities(&after),
        "hardware changed during run",
    )?;
    let adapters: Vec<_> = log
        .lines()
        .filter_map(|l| l.split_once("adapter=").map(|(_, a)| a))
        .collect();
    require(
        adapters.len() == 1
            && adapters[0].starts_with("Some(AdapterInfo {")
            && adapters[0].contains("is_software: false")
            && adapters[0].contains("backend: \"Vulkan\""),
        "missing real Vulkan adapter identity",
    )?;
    require(
        log.contains("Finished `release` profile [optimized]"),
        "missing release profile evidence",
    )?;
    let matching = before.as_array().unwrap().iter().any(|d| {
        adapters[0].contains(&format!("name: {}", d["name"]))
            && adapters[0].contains(&format!("driver_info: {}", d["driver"]))
    });
    require(matching, "adapter does not match hardware snapshot")?;
    checked.report["provenance"] = json!({"revision":revision,"source_hashes":sources,
        "source_hash_manifest":manifest_hash(&json!(sources)),
        "log_hash":format!("blake3:{}",blake3::hash(log.as_bytes()).to_hex()),
        "adapter":adapters[0],"profile":"release","hardware_before":before,"hardware_after":after});
    if smoke && dir.join("soil_benchmark").exists() {
        let executable = fs::read(dir.join("soil_benchmark")).map_err(|e| e.to_string())?;
        require(!executable.is_empty(), "empty archived smoke executable")?;
        checked.report["provenance"]["executable_blake3"] =
            json!(format!("blake3:{}", blake3::hash(&executable).to_hex()));
    }
    Ok(checked)
}

pub fn compatible(a: &Checked, b: &Checked) -> Result<()> {
    require(
        a.report["manifests"] == b.report["manifests"],
        "baseline physical manifests differ",
    )?;
    require(
        a.report["provenance"]["adapter"].is_string()
            && a.report["provenance"]["adapter"] == b.report["provenance"]["adapter"]
            && a.report["provenance"]["profile"] == b.report["provenance"]["profile"],
        "baseline adapter/profile mismatch",
    )
}
