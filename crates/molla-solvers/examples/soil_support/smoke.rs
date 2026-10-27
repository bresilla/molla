use super::results::{Checked, Result, manifest_hash};
use serde_json::{Value, json};

pub const DT: f32 = 0.0005;
pub const STEPS: u32 = 8;
pub const SAMPLES: usize = 5;
pub const PARTICLES: usize = 512;
pub const VELOCITY: [f32; 3] = [0.03, -0.02, 0.01];
pub const GRAVITY: f32 = -9.81;
pub const MASS: f32 = 0.025;
pub const TOLERANCE: f64 = 2e-5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Mpm,
    Dem,
}

pub fn parse_backend(args: impl IntoIterator<Item = String>) -> Result<Backend> {
    let args: Vec<_> = args.into_iter().collect();
    match args.as_slice() {
        [] => Ok(Backend::Mpm),
        [option, value] if option == "--backend" => match value.as_str() {
            "mpm" => Ok(Backend::Mpm),
            "dem" => Ok(Backend::Dem),
            _ => Err("smoke backend must be mpm or dem".into()),
        },
        _ => Err("usage: soil_benchmark smoke [--backend mpm|dem]".into()),
    }
}

pub fn manifest_for(backend: Backend) -> Value {
    let mut value = manifest();
    if backend == Backend::Dem {
        let object = value.as_object_mut().unwrap();
        for key in [
            "grid",
            "cell_size",
            "origin",
            "particle_volume",
            "youngs",
            "poisson",
            "friction_angle",
            "cohesion",
            "boundary_friction",
            "rigids",
            "max_rigid",
            "rheology",
            "time_step",
            "two_pass_p2g",
        ] {
            object.remove(key);
        }
        object.extend(json!({"backend":"resident-dem","scope":"solver+terrain-export completed wall time; owner reset, preparation, observations, rendering and machine excluded",
            "fixture":"lattice-freefall-dem-v1","grain_radius":0.01,"normal_stiffness":10000.,"damping":80.,"friction":0.65,
            "floor_y":-1.,"tire_position_push_scale":0.2,"contacts":[],"leaf_size":8,"rebuild_interval":1,
            "terrain_origin":[-1.,-1.],"terrain_spacing":0.125,"terrain_nx":16,"terrain_nz":16,
            "preparation_substeps_per_reset":1,"oracle_total_substeps":STEPS+1,"terrain_height_tolerance":1e-6,
            "mass_oracle":"fixed grain count times uniform grain mass; no MPM grid-mass claim",
            "oracle":"uniform freefall after one preparation plus eight measured steps; symplectic Euler, grain count and CPU terrain height reference"}).as_object().unwrap().clone());
    }
    value
}

pub fn manifest_with_instrumentation(backend: Backend, enabled: bool) -> Value {
    let mut value = manifest_for(backend);
    value["instrumented"] = json!(enabled);
    value
}

pub fn manifest() -> Value {
    let mut value = json!({"schema":1,"record":"manifest","case":"smoke","mode":"baseline",
        "backend":"dense-mpm","scope":"solver-only completed wall time; reset, observations, terrain export, rendering and machine excluded",
        "calibration":"physically-uncalibrated","performance_accepted":false,"units":"SI",
        "fixture":"lattice-freefall-v1","random_seed":null,"particles":PARTICLES,
        "grid":32,"cell_size":0.0625,"origin":[-1.,-1.,-1.],
        "lattice_origin":[-0.15,-0.15,-0.15],"particle_spacing":0.04,"scalar_type":"f32"});
    let physical = json!({
        "initial_velocity":VELOCITY,"particle_mass":MASS,"particle_volume":0.000015625,
        "youngs":3000.,"poisson":0.3,"friction_angle":0.5,"cohesion":0.,
        "boundary_friction":0.5,"gravity_y":GRAVITY,"rigids":[],"max_rigid":1,
        "rheology":"sand","time_step":"explicit","two_pass_p2g":true,
        "dt":DT,"substeps":STEPS,"simulated_seconds":f64::from(DT)*f64::from(STEPS),
        "warmup_rounds":1,"sample_rounds":SAMPLES,"reset_each_sample":true,
        "instrumented":true,"position_absolute_tolerance":TOLERANCE,
        "velocity_absolute_tolerance":TOLERANCE,"mass_relative_tolerance":TOLERANCE,
        "oracle":"uniform freefall; symplectic Euler positions, gravity impulse and total grid mass"});
    value
        .as_object_mut()
        .unwrap()
        .extend(physical.as_object().unwrap().clone());
    value
}

pub fn points() -> Vec<[f32; 3]> {
    (0..PARTICLES)
        .map(|i| [i % 8, i / 8 % 8, i / 64].map(|v| -0.15 + v as f32 * 0.04))
        .collect()
}

pub fn physical_errors(
    positions: &[[f32; 3]],
    velocities: &[[f32; 3]],
    grid_mass: &[f32],
) -> Result<[f64; 3]> {
    if positions.len() != PARTICLES
        || velocities.len() != PARTICLES
        || grid_mass.len() != 32usize.pow(3)
        || positions
            .iter()
            .flatten()
            .chain(velocities.iter().flatten())
            .chain(grid_mass)
            .any(|v| !v.is_finite())
        || grid_mass.iter().any(|v| *v < 0.)
    {
        return Err("invalid physical snapshot".into());
    }
    let mut errors = kinematic_errors(positions, velocities, STEPS)?;
    let expected_mass = PARTICLES as f64 * f64::from(MASS);
    errors[2] = (grid_mass.iter().map(|v| f64::from(*v)).sum::<f64>() - expected_mass).abs()
        / expected_mass;
    Ok(errors)
}

pub fn kinematic_errors(
    positions: &[[f32; 3]],
    velocities: &[[f32; 3]],
    steps: u32,
) -> Result<[f64; 3]> {
    if positions.len() != PARTICLES
        || velocities.len() != PARTICLES
        || positions
            .iter()
            .flatten()
            .chain(velocities.iter().flatten())
            .any(|v| !v.is_finite())
    {
        return Err("invalid grain/particle snapshot".into());
    }
    let duration = f64::from(DT) * f64::from(steps);
    let mut errors = [0f64; 3];
    for ((initial, position), velocity) in points().iter().zip(positions).zip(velocities) {
        for axis in 0..3 {
            let gravity = if axis == 1 { f64::from(GRAVITY) } else { 0. };
            let expected_position = f64::from(initial[axis])
                + f64::from(VELOCITY[axis]) * duration
                + gravity * f64::from(DT).powi(2) * f64::from(steps) * (f64::from(steps) + 1.) / 2.;
            errors[0] = errors[0].max((f64::from(position[axis]) - expected_position).abs());
            errors[1] = errors[1].max(
                (f64::from(velocity[axis]) - f64::from(VELOCITY[axis]) - gravity * duration).abs(),
            );
        }
    }
    Ok(errors)
}

pub fn check_log(log: &str, exit: i32) -> Result<Checked> {
    if exit != 0 {
        return Err(format!("smoke process failed with exit {exit}"));
    }
    let records = log
        .lines()
        .filter(|line| line.starts_with('{'))
        .map(|line| serde_json::from_str::<Value>(line).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>>>()?;
    let backend = match records.first().and_then(|v| v["backend"].as_str()) {
        Some("dense-mpm") => Backend::Mpm,
        Some("resident-dem") => Backend::Dem,
        _ => return Err("unknown or missing smoke backend".into()),
    };
    let instrumented = records[0]["instrumented"]
        .as_bool()
        .ok_or("missing instrumentation mode")?;
    if !instrumented
        && log
            .lines()
            .filter(|v| *v == "instrumentation_features core=false wgpu=false")
            .count()
            != 1
    {
        return Err("missing counters-disabled build evidence".into());
    }
    let manifest = manifest_with_instrumentation(backend, instrumented);
    if records.len() != SAMPLES + 2
        || records[0] != manifest
        || records.last().unwrap() != &json!({"record":"complete","case":"smoke","samples":SAMPLES})
    {
        return Err("missing, changed or incomplete smoke records".into());
    }
    let mut times = Vec::new();
    let mut failures = Vec::new();
    let mut csv = String::from(
        "backend,scope,sample,wall_ms,encode_submit_ms,simulated_seconds,position_error,velocity_error,mass_relative_error,allocation_observations,readback_bytes,upload_bytes,dispatches,terrain_height_error,preparation_substeps,completed_substeps,final_revision\n",
    );
    let number = |v: &Value, key: &str| {
        v[key]
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 0.)
            .ok_or_else(|| format!("invalid {key}"))
    };
    let duration = f64::from(DT) * f64::from(STEPS);
    for (i, v) in records[1..=SAMPLES].iter().enumerate() {
        if v["record"] != "sample"
            || v["sample"] != i
            || v["case"] != "smoke"
            || v["simulated_seconds"] != duration
        {
            return Err("invalid smoke sample identity/duration".into());
        }
        let wall = number(v, "wall_ms")?;
        let encode = number(v, "encode_submit_ms")?;
        if wall <= 0. || encode > wall {
            return Err("invalid completed timing".into());
        }
        for key in [
            "allocation_observations",
            "readback_bytes",
            "upload_bytes",
            "dispatches",
        ] {
            if (instrumented && v[key].as_u64().is_none())
                || (!instrumented && (!v.get(key).is_some_and(Value::is_null)))
            {
                return Err(format!("invalid {key}"));
            }
        }
        if instrumented
            && (v["allocation_observations"] != 0
                || v["readback_bytes"] != 0
                || v["dispatches"] == 0)
        {
            failures.push(format!("sample {i}: resident work/resource gate failed"));
        }
        for key in ["position_error", "velocity_error", "mass_relative_error"] {
            if number(v, key)? > TOLERANCE {
                failures.push(format!("sample {i}: {key} exceeded frozen tolerance"));
            }
        }
        if backend == Backend::Dem
            && (number(v, "terrain_height_error")? > 1e-6
                || v["terrain_cells"] != 256
                || v["completed_substeps"] != STEPS
                || v["preparation_substeps"] != 1
                || v["final_revision"] != STEPS + 1)
        {
            failures.push(format!("sample {i}: DEM preparation/export gate failed"));
        }
        times.push(wall);
        csv.push_str(&format!(
            "{},{},{i},{wall},{encode},{duration},{},{},{},{},{},{},{},{},{},{},{}\n",
            manifest["backend"].as_str().unwrap(),
            if backend == Backend::Dem {
                "solver+terrain-export"
            } else {
                "solver-only"
            },
            v["position_error"],
            v["velocity_error"],
            v["mass_relative_error"],
            v["allocation_observations"],
            v["readback_bytes"],
            v["upload_bytes"],
            v["dispatches"],
            v["terrain_height_error"],
            v["preparation_substeps"],
            v["completed_substeps"],
            v["final_revision"]
        ));
    }
    let total: f64 = times.iter().sum();
    times.sort_by(f64::total_cmp);
    Ok(Checked {
        report: json!({"schema":1,"checker":"smoke-v1","case":"smoke",
        "status":if failures.is_empty() {"diagnostic-pass"} else {"failed"},"complete_records":true,
        "performance_accepted":false,"calibration":"physically-uncalibrated","failures":failures,
        "resource_counters_available":instrumented,
        "manifests":[manifest],"manifest_hash":manifest_hash(&manifest),"sample_count":SAMPLES,
        "p50_ms":times[2],"p95_ms":times[4],"p99_ms":times[4],"quantiles":"nearest-rank",
        "completed_wall_ms":total,"measured_simulated_seconds":duration*SAMPLES as f64,
        "p50_wall_over_simulated":times[2]/1000./duration,"samples":&records[1..=SAMPLES]}),
        csv,
        failures,
    })
}

pub fn compare_instrumentation(a: &Checked, b: &Checked) -> Result<Value> {
    for result in [a, b] {
        if result.report["checker"] != "smoke-v1"
            || result.report["complete_records"] != true
            || !result.failures.is_empty()
        {
            return Err(
                "instrumentation comparison requires two complete passing smoke results".into(),
            );
        }
        if !result.report["provenance"]["executable_blake3"].is_string() {
            return Err("instrumentation comparison requires archived executables".into());
        }
    }
    let enabled = a.report["manifests"][0]["instrumented"]
        .as_bool()
        .ok_or("missing instrumentation mode")?;
    if b.report["manifests"][0]["instrumented"].as_bool() != Some(!enabled) {
        return Err("choose opposite instrumentation modes".into());
    }
    let mut left = a.report["manifests"][0].clone();
    let mut right = b.report["manifests"][0].clone();
    left["instrumented"] = json!(false);
    right["instrumented"] = json!(false);
    if left != right {
        return Err("instrumentation comparison physical/scope mismatch".into());
    }
    for key in ["adapter", "profile", "source_hashes"] {
        if a.report["provenance"][key].is_null()
            || a.report["provenance"][key] != b.report["provenance"][key]
        {
            return Err(format!("instrumentation comparison {key} mismatch"));
        }
    }
    let (on, off) = if enabled {
        (&a.report, &b.report)
    } else {
        (&b.report, &a.report)
    };
    let number = |v: &Value| {
        v["p50_ms"]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or("invalid comparison median")
    };
    let on_ms = number(on)?;
    let off_ms = number(off)?;
    Ok(
        json!({"scope":"separate reset runs with identical physical manifest and source; observed timing delta, not causal overhead or full-workload acceptance",
        "counters_on_p50_ms":on_ms,"counters_off_p50_ms":off_ms,
        "on_minus_off_ms":on_ms-off_ms,"on_over_off_percent":100.*(on_ms/off_ms-1.),
        "on_executable":on["provenance"]["executable_blake3"],"off_executable":off["provenance"]["executable_blake3"],"performance_accepted":false}),
    )
}
