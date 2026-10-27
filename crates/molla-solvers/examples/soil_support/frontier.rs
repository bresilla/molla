use serde_json::{Value, json};
use std::io::Read;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, String>;
const DT: f64 = 1.0 / 60.0;
const BOUND: f64 = 2e-5;
const STAGES: [&str; 8] = [
    "grid-reset",
    "active-tiles",
    "rigid-reset",
    "p2g",
    "grid-update",
    "contact-projection",
    "g2p",
    "plasticity",
];

fn require(ok: bool, label: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(label.into()) }
}
fn number(v: &Value) -> Result<f64> {
    v.as_f64()
        .filter(|n| n.is_finite())
        .ok_or_else(|| "expected finite number".into())
}
fn array(v: &Value) -> Result<&Vec<Value>> {
    v.as_array().ok_or_else(|| "expected array".into())
}
fn near(a: f64, b: f64) -> Result<()> {
    require(
        (a - b).abs() <= 1e-8 * b.abs().max(1.0),
        "aggregate or physical-time mismatch",
    )
}
fn distribution(values: &[f64]) -> Result<Value> {
    require(
        !values.is_empty() && values.iter().all(|v| v.is_finite() && *v >= 0.0),
        "invalid samples",
    )?;
    let mut sorted = values.to_vec();
    let sum = values.iter().sum::<f64>();
    require(sum.is_finite(), "sample sum overflow")?;
    sorted.sort_by(f64::total_cmp);
    let rank = |p: usize| sorted[(p * sorted.len()).div_ceil(100) - 1];
    Ok(
        json!({"samples":values.len(),"median":rank(50),"p95":rank(95),"p99":rank(99),"max":sorted.last().unwrap(),"mean":sum/values.len() as f64}),
    )
}
fn check_distribution(values: &[f64], claimed: &Value) -> Result<Value> {
    let computed = distribution(values)?;
    for field in ["median", "p95", "p99", "max"] {
        near(number(&computed[field])?, number(&claimed[field])?)?;
    }
    Ok(computed)
}

fn profile_totals(profile: &Value, substeps: usize) -> Result<(f64, f64)> {
    let stages = array(&profile["stages"])?;
    let contacts = array(&profile["contact_stages"])?;
    let initialization = usize::from(
        stages
            .first()
            .is_some_and(|s| s["stage"] == "initialization"),
    );
    require(
        stages.len() == initialization + 8 * substeps && contacts.len() == 3 * substeps,
        "incomplete profile stages",
    )?;
    require(
        initialization == 0 || stages[0]["substep"].is_null(),
        "invalid initialization stage",
    )?;
    let mut total = 0.0;
    let mut previous_end = 0.0;
    for (index, stage) in stages.iter().enumerate() {
        if index >= initialization {
            require(
                stage["stage"] == STAGES[(index - initialization) % 8]
                    && stage["substep"] == (index - initialization) / 8,
                "invalid profile order or substep",
            )?;
        }
        let start = number(&stage["start_ms"])?;
        let duration = number(&stage["gpu_ms"])?;
        require(
            duration >= 0.0 && start + 1e-8 >= previous_end,
            "negative or overlapping outer profile",
        )?;
        previous_end = start + duration;
        require(previous_end.is_finite(), "profile interval overflow")?;
        total += duration;
    }
    let mut solve = 0.0;
    for step in 0..substeps {
        let parent = &stages[initialization + step * 8 + 5];
        let mut end = number(&parent["start_ms"])?;
        for (offset, name) in [
            "contact-preparation",
            "contact-solve",
            "contact-publication",
        ]
        .into_iter()
        .enumerate()
        {
            let child = &contacts[3 * step + offset];
            require(
                child["stage"] == name && child["substep"] == step,
                "invalid nested contact identity",
            )?;
            let start = number(&child["start_ms"])?;
            let duration = number(&child["gpu_ms"])?;
            require(
                duration >= 0.0 && start + 1e-8 >= end,
                "negative or overlapping contact profile",
            )?;
            end = start + duration;
            require(end.is_finite(), "contact interval overflow")?;
            if offset == 1 {
                solve += duration;
            }
        }
        require(
            end <= number(&parent["start_ms"])? + number(&parent["gpu_ms"])? + 1e-8,
            "contact timings escape parent stage",
        )?;
    }
    near(total, number(&profile["mpm_kernel_gpu_ms"])?)?;
    near(
        solve,
        number(&profile["contact_stage_gpu_ms"]["contact-solve"])?,
    )?;
    Ok((total, total - solve))
}

fn state_values(state: &Value) -> Result<(Vec<String>, Vec<f64>)> {
    let mut values = Vec::new();
    for (key, length) in [("joint_q", 11), ("joint_qd", 10)] {
        let items = array(&state[key])?;
        require(items.len() == length, "invalid joint state dimensions")?;
        for item in items {
            values.push(number(item)?);
        }
    }
    let bodies = array(&state["bodies"])?;
    require(bodies.len() == 6, "incomplete body state")?;
    let mut names = Vec::new();
    for body in bodies {
        names.push(
            body["body"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or("invalid body identity")?
                .to_owned(),
        );
        for (field, length) in [
            ("position", 3),
            ("rotation", 4),
            ("linear_velocity", 3),
            ("angular_velocity", 3),
        ] {
            let items = array(&body[field])?;
            require(items.len() == length, "invalid body state dimensions")?;
            for item in items {
                values.push(number(item)?);
            }
        }
    }
    require(
        names.iter().collect::<BTreeSet<_>>().len() == 6,
        "duplicate body identity",
    )?;
    Ok((names, values))
}

struct Run {
    path: PathBuf,
    manifest: Value,
    point: Value,
    states: Vec<(Vec<String>, Vec<f64>)>,
    particles: Vec<f64>,
}

fn read_run(path: &Path) -> Result<Run> {
    let path = fs::canonicalize(path).map_err(|e| e.to_string())?;
    let mut hashes = BTreeMap::new();
    let mut read = |name: &str| -> Result<Vec<u8>> {
        let file = path.join(name);
        require(
            fs::metadata(&file).map_err(|e| e.to_string())?.len() <= 16 * 1024 * 1024,
            "frontier input exceeds 16 MiB per file",
        )?;
        let mut bytes = Vec::new();
        fs::File::open(&file)
            .map_err(|e| e.to_string())?
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        require(
            bytes.len() <= 16 * 1024 * 1024,
            "frontier input grew beyond its bound",
        )?;
        hashes.insert(name.to_owned(), blake3::hash(&bytes).to_hex().to_string());
        Ok(bytes)
    };
    let parse = |bytes: &[u8]| serde_json::from_slice::<Value>(bytes).map_err(|e| e.to_string());
    let manifest = parse(&read("manifest.json")?)?;
    let result = parse(&read("result.json")?)?;
    require(
        manifest
            .as_object()
            .is_some_and(|m| m.contains_key("newton_diagnostic_inner_limit"))
            && manifest["newton_diagnostic_inner_limit"].is_null()
            && manifest["unilateral_config"]["max_iterations"] == 64,
        "unsupported diagnostic iteration limit",
    )?;
    for (key, value) in [
        ("schema", json!(1)),
        ("case", json!("native-tractor-50k")),
        ("particles", json!(50000)),
        ("bodies", json!(6)),
        ("soil_bindings", json!(5)),
        ("warmup_frames", json!(3)),
        ("measured_frames", json!(30)),
        ("renderer_enabled", json!(false)),
        ("terrain_export_enabled", json!(false)),
        ("mpm_kernel_profile_enabled", json!(true)),
        ("acceptance", json!(false)),
        ("physical_calibration", json!("uncalibrated")),
    ] {
        require(
            manifest[key] == value,
            &format!("unsupported diagnostic manifest: {key}"),
        )?;
    }
    require(
        manifest["adapter"]["name"] == "NVIDIA GeForce RTX 4080"
            && manifest["adapter"]["is_software"] == false,
        "not the target RTX 4080",
    )?;
    near(number(&manifest["frame_simulated_seconds"])?, DT)?;
    let substeps = manifest["substeps_per_frame"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 1024)
        .ok_or("invalid substep count")? as usize;
    near(number(&manifest["substep_seconds"])? * substeps as f64, DT)?;
    require(
        result["status"] == "diagnostic_complete"
            && result["acceptance"] == false
            && result["samples"] == 30
            && result["final_committed_frame"] == 33
            && result["final_particles"] == 50000,
        "incomplete diagnostic result",
    )?;
    require(
        result["soil_only_ms"].is_null() && manifest["soil_only_ms"].is_null(),
        "unsupported soil-only timing schema",
    )?;
    let csv = String::from_utf8(read("frames.csv")?).map_err(|e| e.to_string())?;
    let mut lines = csv.lines();
    require(
        lines.next()
            == Some(
                "frame,warmup,wall_ms,frame_simulated_seconds,total_simulated_seconds,committed_frames,dispatches,upload_bytes,readback_bytes,allocation_observations,chassis_x,chassis_y,chassis_z,machine_kinetic_j,machine_gravity_potential_j,commanded_motor_work_j,machine_mechanical_change_minus_commanded_work_j",
            ),
        "unknown frame CSV schema",
    )?;
    let mut wall = Vec::new();
    let mut counters = [0.0; 4];
    let mut frame_count = 0;
    for (index, line) in lines.enumerate() {
        let fields: Vec<_> = line.split(',').collect();
        require(
            fields.len() == 17 && fields[1] == if index < 3 { "true" } else { "false" },
            "invalid CSV record",
        )?;
        let values = fields
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if i == 1 {
                    Ok(0.0)
                } else {
                    s.parse::<f64>().map_err(|e| e.to_string())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        require(
            values.iter().all(|v| v.is_finite())
                && values[0] == index as f64
                && values[5] == (index + 1) as f64
                && values[2] > 0.0,
            "invalid frame sequence",
        )?;
        near(values[3], DT)?;
        near(values[4], (index + 1) as f64 * DT)?;
        require(
            values[6..10].iter().all(|n| *n >= 0.0 && n.fract() == 0.0),
            "invalid resource counters",
        )?;
        if index >= 3 {
            wall.push(values[2]);
            for i in 0..4 {
                counters[i] += values[6 + i];
            }
        }
        frame_count += 1;
    }
    require(frame_count == 33, "incomplete frame CSV")?;
    let wall_stats = check_distribution(&wall, &result["wall_ms"])?;
    near(wall.iter().sum(), number(&result["completed_wall_ms_sum"])?)?;
    near(number(&result["measured_simulated_seconds"])?, 0.5)?;
    near(number(&result["total_simulated_seconds"])?, 0.55)?;
    near(
        number(&result["wall_over_simulated_ratio"])?,
        wall.iter().sum::<f64>() / 500.0,
    )?;
    for (index, key) in [
        "dispatches",
        "upload_bytes",
        "readback_bytes",
        "allocation_observations",
    ]
    .into_iter()
    .enumerate()
    {
        near(counters[index], number(&result[key])?)?;
    }
    let missed = wall.iter().filter(|&&v| v > 1000.0 * DT).count();
    require(
        result["missed_60hz_deadlines"] == missed,
        "deadline count mismatch",
    )?;
    let json_lines = |bytes: Vec<u8>| -> Result<Vec<Value>> {
        String::from_utf8(bytes)
            .map_err(|e| e.to_string())?
            .lines()
            .map(|line| parse(line.as_bytes()))
            .collect()
    };
    let profiles = json_lines(read("mpm-profiles.jsonl")?)?;
    require(profiles.len() == 33, "incomplete profile sequence")?;
    let mut mpm = Vec::new();
    let mut non_newton = Vec::new();
    for (index, p) in profiles.iter().enumerate() {
        require(
            p["committed_frame"] == index + 1
                && p["warmup"] == (index < 3)
                && p["requested_substeps"] == substeps,
            "profile/frame mismatch",
        )?;
        near(number(&p["requested_simulated_seconds"])?, DT)?;
        let (total, other) = profile_totals(p, substeps)?;
        if index >= 3 {
            mpm.push(total);
            non_newton.push(other);
        }
    }
    let mpm_stats = check_distribution(&mpm, &result["mpm_kernel_gpu_ms"])?;
    let raw_states = json_lines(read("machine-states.jsonl")?)?;
    require(raw_states.len() == 33, "incomplete machine trajectory")?;
    let mut states = Vec::new();
    for (index, state) in raw_states.iter().enumerate() {
        require(
            state["frame"] == index && state["committed_frame"] == index + 1,
            "invalid machine frame",
        )?;
        near(
            number(&state["simulated_seconds"])?,
            (index + 1) as f64 * DT,
        )?;
        let decoded = state_values(state)?;
        if let Some((names, _)) = states.first() {
            require(*names == decoded.0, "changing body identities")?;
        }
        states.push(decoded);
    }
    let bytes = read("final-particles.bin")?;
    require(
        bytes.len() == 50000 * 12,
        "incomplete final particle observation",
    )?;
    let particles = bytes
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
        .collect::<Vec<_>>();
    require(
        particles.iter().all(|x| x.is_finite()),
        "nonfinite final particles",
    )?;
    require(
        manifest["live_soil_newton_backend"].is_string()
            && manifest["live_soil_newton_backend"]
                == manifest["unilateral_config"]["newton_method"],
        "solver identity mismatch",
    )?;
    let wall_p95 = number(&wall_stats["p95"])?;
    let mpm_p95 = number(&mpm_stats["p95"])?;
    let point = json!({"input":path,"method":manifest["live_soil_newton_backend"],"wall_ms":wall_stats,"wall_over_simulated_ratio":wall.iter().sum::<f64>()/500.0,
        "mpm_kernel_gpu_ms":mpm_stats,"non_newton_mpm_gpu_ms":distribution(&non_newton)?,"missed_60hz_deadlines":missed,
        "diagnostic_wall_p95_budget_ratio":wall_p95/(1000.0*DT),"measured_mpm_p95_to_soil_budget_ratio":mpm_p95/4.0,
        "soil_work_gate":if mpm_p95>4.0 {"failed_by_measured_subset"} else {"unmeasured_full_scope"},
        "physical_accuracy":"uncalibrated","full_soil_wall_time_ms":null,"acceptance":false,"source_provenance_verified":false,"input_blake3":hashes});
    Ok(Run {
        path,
        manifest,
        point,
        states,
        particles,
    })
}

fn compare(reference: &Run, candidate: &Run) -> Result<(f64, f64)> {
    for field in [
        "adapter",
        "particles",
        "seed",
        "bodies",
        "soil_bindings",
        "machine_mass_kg",
        "cell_size_m",
        "grid_size",
        "grid_origin_m",
        "soil_bed_size_m",
        "youngs_pa",
        "density_kg_m3",
        "wheel_torque_nm",
        "frame_simulated_seconds",
        "substeps_per_frame",
        "substep_seconds",
        "friction_config",
        "friction_fixed_walls",
        "backend",
        "live_soil_grid",
        "machine_rate_encoding",
    ] {
        require(
            !reference.manifest[field].is_null()
                && reference.manifest[field] == candidate.manifest[field],
            &format!("declared fixture mismatch: {field}"),
        )?;
    }
    for field in [
        "max_iterations",
        "node_capacity_total",
        "search_distance_m",
        "newton_start",
    ] {
        require(
            !reference.manifest["unilateral_config"][field].is_null()
                && reference.manifest["unilateral_config"][field]
                    == candidate.manifest["unilateral_config"][field],
            "solver budget or contact fixture mismatch",
        )?;
    }
    let mut machine = 0.0f64;
    for ((a_names, a), (b_names, b)) in reference.states.iter().zip(&candidate.states) {
        require(
            a_names == b_names && a.len() == b.len(),
            "trajectory identity mismatch",
        )?;
        for (a, b) in a.iter().zip(b) {
            let delta = (a - b).abs();
            require(delta.is_finite(), "trajectory difference overflow")?;
            machine = machine.max(delta);
        }
    }
    let particles = reference
        .particles
        .iter()
        .zip(&candidate.particles)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    Ok((machine, particles))
}

pub fn report(paths: &[PathBuf]) -> Result<Value> {
    require(
        paths.len() >= 2 && paths.len() <= 16,
        "frontier requires a reference and 1..=15 observations",
    )?;
    let runs = paths
        .iter()
        .map(|p| read_run(p))
        .collect::<Result<Vec<_>>>()?;
    require(
        runs.iter().map(|r| &r.path).collect::<BTreeSet<_>>().len() == runs.len(),
        "duplicate input directory",
    )?;
    let mut points = Vec::new();
    for (index, run) in runs.iter().enumerate() {
        let (machine, particles) = compare(&runs[0], run)?;
        let mut point = run.point.clone();
        point["reference"] = json!(index == 0);
        point["machine_max_absolute_component_delta"] = json!(machine);
        point["final_particle_max_slotwise_position_delta_m"] = json!(particles);
        let ratio = machine.max(particles) / BOUND;
        require(ratio.is_finite(), "normalized discrepancy overflow")?;
        point["numerical_discrepancy_ratio"] = json!(ratio);
        point["observed_numerical_gate"] = json!(if index == 0 {
            "reference_only"
        } else if machine <= BOUND && particles <= BOUND {
            "within_observed_bound"
        } else {
            "failed"
        });
        points.push(point);
    }
    let observed_nondominated: Vec<_> = (0..points.len())
        .filter(|&i| {
            !(0..points.len()).any(|j| {
                let p = &points[i];
                let q = &points[j];
                let a = p["wall_ms"]["p95"].as_f64().unwrap();
                let b = q["wall_ms"]["p95"].as_f64().unwrap();
                let x = p["numerical_discrepancy_ratio"].as_f64().unwrap();
                let y = q["numerical_discrepancy_ratio"].as_f64().unwrap();
                b <= a && y <= x && (b < a || y < x)
            })
        })
        .collect();
    Ok(
        json!({"schema":1,"report":"tractor-frontier-v1","complete":true,"acceptance":false,
        "scope":"frozen short-run numerical-agreement/performance observations, not calibrated accuracy, paired speedup or a complete quality/resolution frontier",
        "compatibility":"declared manifest fields only; omitted physical settings and historical source provenance are not certified",
        "targets":{"adapter":"RTX 4080","particles":50000,"frame_seconds":DT,"soil_work_ms":4.0,"absolute_component_bound":BOUND},
        "limits":["separate evolving atomic trajectories; no causal attribution","reference agreement is not physical accuracy","only final particle positions, not full soil trajectories","renderer/export and full soil wall-time scope absent","nested contact timings are not added to parent MPM totals","non-Newton work is measured work, not a predicted speedup after deleting Newton","source provenance requires separate archive audits"],
        "accepted_frontier":[],"observed_nondominated_point_indices":observed_nondominated,"points":points}),
    )
}

pub fn run(args: impl Iterator<Item = String>) -> Result<u8> {
    let mut args = args;
    let mut reference = None;
    let mut inputs = Vec::new();
    let mut output = None;
    while let Some(key) = args.next() {
        let value = PathBuf::from(
            args.next()
                .ok_or_else(|| format!("missing value for {key}"))?,
        );
        match key.as_str() {
            "--reference" if reference.is_none() => reference = Some(value),
            "--input" => inputs.push(value),
            "--output" if output.is_none() => output = Some(value),
            _ => return Err(format!("unknown or duplicate option {key}")),
        }
    }
    inputs.insert(0, reference.ok_or("--reference required")?);
    let output = output.ok_or("--output requires a fresh directory")?;
    let report = report(&inputs)?;
    let encoded = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    fs::create_dir(&output).map_err(|e| e.to_string())?;
    fs::write(output.join("frontier.json"), &encoded).map_err(|e| e.to_string())?;
    println!("{encoded}");
    Ok(0)
}

#[cfg(test)]
#[path = "frontier_tests.rs"]
mod tests;
