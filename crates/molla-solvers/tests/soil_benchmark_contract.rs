#[path = "../examples/soil_support/results.rs"]
mod results;
#[path = "../examples/soil_support/smoke.rs"]
mod smoke;

use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn fixture() -> String {
    let mut log = String::from(
        "Finished `release` profile [optimized]\nadapter=Some(AdapterInfo { name: \"test GPU\", backend: \"Vulkan\", is_software: false, driver_info: \"1.0\" })\n",
    );
    for (grid, layout, iters) in [
        (32u64, "patch", 0),
        (64, "patch", 0),
        (96, "patch", 0),
        (32, "patch", 3),
        (64, "patch", 3),
        (96, "patch", 3),
        (32, "filled", 0),
        (32, "filled", 3),
    ] {
        let filled = layout == "filled";
        let particles = if filled { 32768 } else { 50000 };
        let duration = f64::from(0.0001_f32) * 4.;
        let manifest = json!({
            "schema":1,"record":"manifest","case":"active-domain",
            "scope":"solver-only diagnostic; reset, readback, raster, renderer and machine excluded",
            "grid":grid,"layout":layout,"particles":particles,"jacobi_iters":iters,
            "dt":0.0001,"substeps":4,"simulated_seconds":duration,"cell_size":0.0625,
            "origin":[-1,-1,-1],"particle_spacing":0.025,"particle_mass":0.025,
            "particle_volume":0.000015625,"youngs":3000,"poisson":0.3,"gravity_y":-9.81,
            "friction_angle":0.5235988,"cohesion":0,"boundary_friction":0.5,
            "initial_velocity":[0.03,-0.02,0.01],"rigids":if filled {0} else {2},
            "two_pass_p2g":true,"warmup_rounds":2,"sample_rounds":12,"reset_each_sample":true,
            "mode_count":3,"order":"cyclic-rotation","compact_tile_capacity":if filled {512} else {256},
            "absolute_particle_tolerance":0.00002,"instrumented":true
        });
        log += &format!("{manifest}\n");
        for sample in 0..12 {
            for offset in 0..3 {
                let mode = (sample + 2 + offset) % 3;
                log += &format!(
                    "{}\n",
                    json!({
                        "record":"sample","case":"active-domain","grid":grid,"layout":layout,"jacobi_iters":iters,
                        "mode":mode,"sample":sample,"wall_ms":1.+sample as f64,
                        "encode_submit_ms":0.2,"simulated_seconds":duration,"allocation_observations":0,
                        "readback_bytes":0,"upload_bytes":if mode == 0 {352} else {400}
                    })
                );
            }
        }
        for (mode, name) in ["dense", "active-tiles", "compact-tiles"]
            .iter()
            .enumerate()
        {
            let world_tiles = grid.div_ceil(4).pow(3);
            let capacity = if mode == 2 {
                if filled { 512 } else { 256 }
            } else {
                world_tiles
            };
            let active = if filled { capacity } else { 8 };
            let bytes = if mode == 2 {
                capacity * 64 * 56 + 16 + 8 * ((2 * capacity).next_power_of_two() + capacity)
            } else {
                grid.pow(3) * 56 + 16 + 8 * world_tiles
            };
            log += &format!(
                "{}\n",
                json!({
                    "schema":1,"record":"summary","case":"active-domain","grid":grid,"layout":layout,
                    "particles":particles,"jacobi_iters":iters,"mode":name,"samples":12,
                    "p50_ms":6.,"p95_ms":12.,"p99_ms":12.,"max_ms":12.,
                    "simulated_seconds":duration,"p50_wall_over_simulated":0.006/duration,
                    "active_tiles":active,"tile_capacity":capacity,
                    "scheduled_cell_slots":if mode == 0 {grid.pow(3)} else {active*64},
                    "parity_passed":true,"dense_backing_storage":mode != 2,"grid_storage_bytes":bytes
                })
            );
        }
    }
    log += "test result: ok. 1 passed; 0 failed; 0 ignored;\n";
    log
}

fn mutate(log: &str, record: &str, mut change: impl FnMut(&mut Value)) -> String {
    let mut changed = false;
    log.lines()
        .map(|line| {
            if !changed && line.starts_with('{') {
                let mut v: Value = serde_json::from_str(line).unwrap();
                if v["record"] == record {
                    change(&mut v);
                    changed = true;
                    return v.to_string();
                }
            }
            line.to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static ID: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "molla-soil-check-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("active-domain.log"), fixture()).unwrap();
        fs::write(dir.join("active-domain.exit"), "0\n").unwrap();
        fs::write(dir.join("revision.txt"), format!("{}\n", "1".repeat(40))).unwrap();
        let sources = [
            "Makefile",
            "Cargo.lock",
            "crates/molla-solvers/src/mpm.rs",
            "crates/molla-solvers/tests/mpm_active_tiles_gpu.rs",
            "crates/molla-solvers/shaders/mpm.wgsl",
            "crates/molla-solvers/src/mpm_active_tiles.rs",
            "crates/molla-solvers/shaders/mpm_active_tiles.wgsl",
        ];
        fs::write(
            dir.join("source-sha256.txt"),
            sources
                .iter()
                .map(|s| format!("{}  {s}\n", "a".repeat(64)))
                .collect::<String>(),
        )
        .unwrap();
        for name in ["hardware-before.csv", "hardware-after.csv"] {
            fs::write(dir.join(name),"name, driver_version, utilization.gpu [%], memory.used [MiB], temperature.gpu, power.draw [W]\ntest GPU, 1.0, 90 %, 8192 MiB, 72, 280 W\n").unwrap();
        }
        Self(dir)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn complete_records_recompute_quantiles_rtf_and_never_claim_acceptance() {
    let checked = results::check_log(&fixture(), 0).unwrap();
    assert!(checked.failures.is_empty());
    assert_eq!(checked.report["sample_count"], 288);
    assert_eq!(checked.report["summary_count"], 24);
    assert_eq!(checked.report["status"], "diagnostic-pass");
    assert_eq!(checked.report["performance_accepted"], false);
    assert_eq!(checked.csv.lines().count(), 25);
    assert_eq!(checked.report["results"][0]["completed_wall_ms"], 78.);
}

fn smoke_fixture() -> String {
    let mut log = format!("{}\n", smoke::manifest());
    for i in 0..smoke::SAMPLES {
        log += &format!(
            "{}\n",
            json!({"record":"sample","case":"smoke","sample":i,
            "wall_ms":i as f64+1.,"encode_submit_ms":0.1,
            "simulated_seconds":f64::from(smoke::DT)*f64::from(smoke::STEPS),
            "position_error":0.,"velocity_error":0.,"mass_relative_error":0.,
            "allocation_observations":0,"readback_bytes":0,"upload_bytes":88,"dispatches":32})
        );
    }
    log + &format!(
        "{}\n",
        json!({"record":"complete","case":"smoke","samples":smoke::SAMPLES})
    )
}

#[test]
fn smoke_recomputes_metrics_and_rejects_incomplete_or_changed_work() {
    let log = smoke_fixture();
    let checked = smoke::check_log(&log, 0).unwrap();
    assert!(checked.failures.is_empty());
    assert_eq!(checked.report["p50_ms"], 3.);
    assert_eq!(checked.report["p99_ms"], 5.);
    assert_eq!(checked.report["completed_wall_ms"], 15.);
    assert_eq!(checked.report["performance_accepted"], false);
    assert_eq!(checked.csv.lines().count(), smoke::SAMPLES + 1);
    for code in [1, 2, 124, 137] {
        assert!(smoke::check_log(&log, code).is_err());
    }
    assert!(smoke::check_log(&log.lines().skip(1).collect::<Vec<_>>().join("\n"), 0).is_err());
    assert!(
        smoke::check_log(
            &mutate(&log, "manifest", |v| v["particles"] = json!(50000)),
            0
        )
        .is_err()
    );
    assert!(smoke::check_log(&mutate(&log, "sample", |v| v["sample"] = json!(1)), 0).is_err());
    assert!(smoke::check_log(&mutate(&log, "sample", |v| v["wall_ms"] = json!(0)), 0).is_err());
    assert!(
        smoke::check_log(
            &mutate(&log, "sample", |v| v["simulated_seconds"] = json!(1.)),
            0
        )
        .is_err()
    );
    for key in [
        "position_error",
        "velocity_error",
        "mass_relative_error",
        "allocation_observations",
        "readback_bytes",
    ] {
        assert!(
            !smoke::check_log(&mutate(&log, "sample", |v| v[key] = json!(1)), 0)
                .unwrap()
                .failures
                .is_empty()
        );
    }
}

#[test]
fn smoke_disabled_counters_are_unavailable_not_zero() {
    let make_log = || {
        "instrumentation_features core=false wgpu=false\n".to_string()
            + &smoke_fixture()
                .lines()
                .map(|line| {
                    let mut v: Value = serde_json::from_str(line).unwrap();
                    if v["record"] == "manifest" {
                        v["instrumented"] = json!(false);
                    }
                    if v["record"] == "sample" {
                        for key in [
                            "allocation_observations",
                            "readback_bytes",
                            "upload_bytes",
                            "dispatches",
                        ] {
                            v[key] = Value::Null;
                        }
                    }
                    v.to_string()
                })
                .collect::<Vec<_>>()
                .join("\n")
    };
    let log = make_log();
    let mut off = smoke::check_log(&log, 0).unwrap();
    assert!(off.failures.is_empty());
    assert_eq!(off.report["resource_counters_available"], false);
    assert_eq!(off.report["performance_accepted"], false);
    for key in [
        "allocation_observations",
        "readback_bytes",
        "upload_bytes",
        "dispatches",
    ] {
        assert!(smoke::check_log(&mutate(&log, "sample", |v| v[key] = json!(0)), 0).is_err());
        assert!(
            smoke::check_log(
                &mutate(&log, "sample", |v| {
                    v.as_object_mut().unwrap().remove(key);
                }),
                0
            )
            .is_err()
        );
    }
    assert!(smoke::check_log(&log.replace("core=false", "core=true"), 0).is_err());
    assert!(
        smoke::check_log(
            &mutate(&log, "manifest", |v| v["instrumented"] = json!(true)),
            0
        )
        .is_err()
    );
    assert!(
        !smoke::check_log(
            &mutate(&log, "sample", |v| v["velocity_error"] = json!(1.)),
            0
        )
        .unwrap()
        .failures
        .is_empty()
    );
    let mut on = smoke::check_log(&smoke_fixture(), 0).unwrap();
    assert!(smoke::compare_instrumentation(&on, &off).is_err());
    let provenance = json!({"adapter":"fixture","profile":"release","source_hashes":{"source":"hash"},"executable_blake3":format!("blake3:{}","a".repeat(64))});
    on.report["provenance"] = provenance.clone();
    off.report["provenance"] = provenance;
    let compared = smoke::compare_instrumentation(&on, &off).unwrap();
    assert_eq!(compared["on_over_off_percent"], 0.);
    assert_eq!(compared["performance_accepted"], false);
    assert_eq!(compared, smoke::compare_instrumentation(&off, &on).unwrap());
    assert!(smoke::compare_instrumentation(&on, &on).is_err());
    off.report["provenance"]["source_hashes"]["source"] = json!("changed");
    assert!(smoke::compare_instrumentation(&on, &off).is_err());
    off.report["provenance"] = on.report["provenance"].clone();
    off.report["manifests"][0]["particles"] = json!(513);
    assert!(smoke::compare_instrumentation(&on, &off).is_err());
}

#[test]
fn smoke_oracle_detects_noop_nonfinite_and_mass_loss() {
    let initial = smoke::points();
    let mut positions = initial.clone();
    let mut velocities = vec![smoke::VELOCITY; smoke::PARTICLES];
    let mut mass = vec![0.; 32usize.pow(3)];
    mass[0] = smoke::PARTICLES as f32 * smoke::MASS;
    assert!(smoke::physical_errors(&positions, &velocities, &mass).unwrap()[1] > smoke::TOLERANCE);
    for _ in 0..smoke::STEPS {
        for (p, v) in positions.iter_mut().zip(&mut velocities) {
            v[1] += smoke::GRAVITY * smoke::DT;
            for axis in 0..3 {
                p[axis] += v[axis] * smoke::DT;
            }
        }
    }
    assert!(
        smoke::physical_errors(&positions, &velocities, &mass)
            .unwrap()
            .iter()
            .all(|e| *e < smoke::TOLERANCE)
    );
    mass[0] *= 0.5;
    assert!(smoke::physical_errors(&positions, &velocities, &mass).unwrap()[2] > 0.49);
    positions[0][0] = f32::NAN;
    assert!(smoke::physical_errors(&positions, &velocities, &mass).is_err());
}

#[test]
fn dem_smoke_freezes_backend_scope_preparation_and_export() {
    use smoke::Backend;
    assert_eq!(
        smoke::parse_backend(Vec::<String>::new()).unwrap(),
        Backend::Mpm
    );
    assert_eq!(
        smoke::parse_backend(["--backend".into(), "dem".into()]).unwrap(),
        Backend::Dem
    );
    for args in [
        vec!["--backend"],
        vec!["--backend", "bad"],
        vec!["--backend", "dem", "--backend", "mpm"],
        vec!["--particles", "50000"],
    ] {
        assert!(smoke::parse_backend(args.into_iter().map(String::from)).is_err());
    }
    let log = smoke_fixture()
        .lines()
        .map(|line| {
            let mut value: Value = serde_json::from_str(line).unwrap();
            if value["record"] == "manifest" {
                value = smoke::manifest_for(Backend::Dem);
            }
            if value["record"] == "sample" {
                value["terrain_height_error"] = json!(0.);
                value["terrain_cells"] = json!(256);
                value["preparation_substeps"] = json!(1);
                value["completed_substeps"] = json!(smoke::STEPS);
                value["final_revision"] = json!(smoke::STEPS + 1);
            }
            value.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let checked = smoke::check_log(&log, 0).unwrap();
    assert!(checked.failures.is_empty());
    assert_eq!(checked.report["manifests"][0]["backend"], "resident-dem");
    assert!(
        checked.report["manifests"][0]["scope"]
            .as_str()
            .unwrap()
            .starts_with("solver+terrain-export")
    );
    assert!(checked.report["manifests"][0]["grid"].is_null());
    assert!(
        checked
            .csv
            .lines()
            .nth(1)
            .unwrap()
            .starts_with("resident-dem,solver+terrain-export,")
    );
    assert_ne!(
        checked.report["manifest_hash"],
        smoke::check_log(&smoke_fixture(), 0).unwrap().report["manifest_hash"]
    );
    for (key, value) in [
        ("terrain_height_error", json!(2e-6)),
        ("terrain_cells", json!(0)),
        ("completed_substeps", json!(7)),
        ("preparation_substeps", json!(0)),
        ("final_revision", json!(8)),
    ] {
        assert!(
            !smoke::check_log(&mutate(&log, "sample", |v| v[key] = value.clone()), 0)
                .unwrap()
                .failures
                .is_empty()
        );
    }
    assert!(
        smoke::check_log(
            &mutate(&log, "sample", |v| {
                v.as_object_mut().unwrap().remove("terrain_height_error");
            }),
            0
        )
        .is_err()
    );
    assert!(
        smoke::check_log(
            &mutate(&log, "manifest", |v| v["rebuild_interval"] = json!(8)),
            0
        )
        .is_err()
    );
    let mut positions = smoke::points();
    let mut velocities = vec![smoke::VELOCITY; smoke::PARTICLES];
    for _ in 0..smoke::STEPS + 1 {
        for (p, v) in positions.iter_mut().zip(&mut velocities) {
            v[1] += smoke::GRAVITY * smoke::DT;
            for axis in 0..3 {
                p[axis] += v[axis] * smoke::DT;
            }
        }
    }
    assert!(
        smoke::kinematic_errors(&positions, &velocities, smoke::STEPS + 1)
            .unwrap()
            .iter()
            .all(|e| *e < smoke::TOLERANCE)
    );
    assert!(
        smoke::kinematic_errors(&positions, &velocities, smoke::STEPS).unwrap()[1]
            > smoke::TOLERANCE
    );
    positions.pop();
    assert!(smoke::kinematic_errors(&positions, &velocities, smoke::STEPS + 1).is_err());
}

#[test]
fn smoke_directory_requires_source_integrity_and_preserves_old_checker() {
    let dir = Directory::new();
    fs::remove_file(dir.0.join("active-domain.log")).unwrap();
    let log = format!(
        "Finished `release` profile [optimized]\nadapter=Some(AdapterInfo {{ name: \"test GPU\", backend: \"Vulkan\", is_software: false, driver_info: \"1.0\" }})\n{}",
        smoke_fixture()
    );
    fs::write(dir.0.join("smoke.log"), log).unwrap();
    fs::write(dir.0.join("smoke.exit"), "0\n").unwrap();
    fs::write(dir.0.join("source-check.exit"), "0\n").unwrap();
    let mut hashes = fs::read_to_string(dir.0.join("source-sha256.txt")).unwrap();
    for file in [
        "soil_benchmark.rs",
        "soil_support/smoke.rs",
        "soil_support/smoke_gpu.rs",
    ] {
        hashes += &format!("{}  crates/molla-solvers/examples/{file}\n", "b".repeat(64));
    }
    fs::write(dir.0.join("source-sha256.txt"), hashes).unwrap();
    let checked = results::check_directory(&dir.0).unwrap();
    assert_eq!(checked.report["checker"], "smoke-v1");
    assert!(
        results::compatible(
            &checked,
            &results::check_directory(&Directory::new().0).unwrap()
        )
        .is_err()
    );
    fs::write(dir.0.join("source-check.exit"), "1\n").unwrap();
    assert!(results::check_directory(&dir.0).is_err());
    fs::write(dir.0.join("smoke.exit"), "124\n").unwrap();
    let invalid =
        results::invalid_for_directory(results::check_directory(&dir.0).unwrap_err(), &dir.0);
    assert_eq!(invalid.report["status"], "timeout");
    assert_eq!(invalid.report["case"], "smoke");
    assert_eq!(invalid.report["checker"], "smoke-v1");
}

#[test]
fn rejects_empty_truncated_duplicate_and_unexpected_records() {
    assert!(results::check_log("", 0).is_err());
    let log = fixture();
    for record in ["manifest", "sample", "summary"] {
        let line = log
            .lines()
            .find(|l| l.contains(&format!("\"record\":\"{record}\"")))
            .unwrap();
        assert!(
            results::check_log(&log.replacen(&format!("{line}\n"), "", 1), 0).is_err(),
            "{record} missing"
        );
        assert!(
            results::check_log(&log.replacen(line, &format!("{line}\n{line}"), 1), 0).is_err(),
            "{record} duplicated"
        );
    }
    assert!(results::check_log(&(log.clone() + "{\n"), 0).is_err());
    assert!(results::check_log(&mutate(&log, "manifest", |v| v["schema"] = json!(2)), 0).is_err());
    assert!(
        results::check_log(
            &mutate(&log, "manifest", |v| v["case"] = json!("unknown")),
            0
        )
        .is_err()
    );
    assert!(
        results::check_log(
            &mutate(&log, "sample", |v| v["ignored-field"] = json!(1)),
            0
        )
        .is_err()
    );
}

#[test]
fn rejects_nonfinite_metrics_bad_counters_and_shortened_physical_time() {
    let log = fixture();
    for (record, field, value) in [
        ("manifest", "dt", json!(0)),
        ("manifest", "dt", json!(1e100)),
        ("manifest", "substeps", json!(0)),
        ("manifest", "particles", json!(0)),
        ("manifest", "absolute_particle_tolerance", json!(0.00003)),
        ("manifest", "simulated_seconds", json!(0.0004)),
        ("sample", "wall_ms", Value::Null),
        ("sample", "wall_ms", json!(-1)),
        ("sample", "encode_submit_ms", json!(100)),
        ("sample", "simulated_seconds", json!(0.0166667)),
        ("sample", "allocation_observations", json!(1)),
        ("sample", "readback_bytes", json!(1)),
        ("sample", "upload_bytes", json!(0)),
        ("sample", "mode", json!(3)),
        ("summary", "p50_ms", json!(5)),
        ("summary", "p95_ms", json!(11)),
        ("summary", "p99_ms", json!(11)),
        ("summary", "max_ms", json!(13)),
        ("summary", "p50_wall_over_simulated", json!(0.1)),
        ("summary", "active_tiles", json!(100000)),
        ("summary", "grid_storage_bytes", json!(1)),
    ] {
        let changed = mutate(&log, record, |v| v[field] = value.clone());
        assert!(results::check_log(&changed, 0).is_err(), "{record}.{field}");
    }
    assert!(results::check_log(&log.replace("\"wall_ms\":1.0", "\"wall_ms\":NaN"), 0).is_err());
}

#[test]
fn failures_timeout_and_missing_completion_never_pass() {
    let log = fixture();
    for exit in [2, 101, 124, 137] {
        let result = results::check_log(&log, exit).unwrap();
        assert!(!result.failures.is_empty());
        assert_ne!(result.report["status"], "diagnostic-pass");
    }
    let bad = mutate(&log, "summary", |v| v["parity_passed"] = json!(false));
    assert!(!results::check_log(&bad, 0).unwrap().failures.is_empty());
    let no_terminal = log.replace("test result: ok. 1 passed; 0 failed; 0 ignored;", "");
    assert!(
        !results::check_log(&no_terminal, 0)
            .unwrap()
            .failures
            .is_empty()
    );
    let hidden_failure = log + "parity_failure: deliberately inconsistent successful summary\n";
    assert!(
        !results::check_log(&hidden_failure, 0)
            .unwrap()
            .failures
            .is_empty()
    );
}

#[test]
fn hash_is_canonical_and_changes_with_physics() {
    let a: Value = serde_json::from_str("{\"a\":1,\"b\":{\"c\":2,\"d\":3}}").unwrap();
    let b: Value = serde_json::from_str("{\"b\":{\"d\":3,\"c\":2},\"a\":1}").unwrap();
    assert_eq!(results::manifest_hash(&a), results::manifest_hash(&b));
    let mut c = b;
    c["b"]["c"] = json!(4);
    assert_ne!(results::manifest_hash(&a), results::manifest_hash(&c));
}

#[test]
fn checks_provenance_compatibility_and_retains_contention_without_accepting_it() {
    let a = Directory::new();
    let b = Directory::new();
    let first = results::check_directory(&a.0).unwrap();
    let second = results::check_directory(&b.0).unwrap();
    results::compatible(&first, &second).unwrap();
    assert_eq!(
        first.report["provenance"]["hardware_before"][0]["utilization_percent"],
        90.
    );
    assert_eq!(first.report["performance_accepted"], false);
    let log = fixture().replace("\"youngs\":3000", "\"youngs\":3001");
    fs::write(b.0.join("active-domain.log"), log).unwrap();
    let different_physics = results::check_directory(&b.0).unwrap();
    assert!(results::compatible(&first, &different_physics).is_err());
    fs::write(b.0.join("hardware-after.csv"), "").unwrap();
    assert!(results::check_directory(&b.0).is_err());
    fs::remove_file(a.0.join("active-domain.exit")).unwrap();
    assert!(results::check_directory(&a.0).is_err());
}

#[test]
fn rejects_adapter_failure_software_fallback_and_nonrelease_results() {
    let dir = Directory::new();
    for log in [
        "GPU device unavailable\n".into(),
        fixture().replace("is_software: false", "is_software: true"),
        fixture().replace("test GPU", "foreign GPU"),
        fixture().replace(
            "Finished `release` profile [optimized]",
            "Finished `dev` profile",
        ),
    ] {
        fs::write(dir.0.join("active-domain.log"), log).unwrap();
        assert!(results::check_directory(&dir.0).is_err());
    }
}

#[test]
fn invalid_recheck_replaces_stale_success_and_clears_csv() {
    let dir = Directory::new();
    let valid = results::check_directory(&dir.0).unwrap();
    results::write_outputs(&valid, &dir.0).unwrap();
    assert!(
        !fs::read_to_string(dir.0.join("checked-results.csv"))
            .unwrap()
            .is_empty()
    );
    let invalid = results::invalid("benchmark exit 124: truncated sample set".into());
    results::write_outputs(&invalid, &dir.0).unwrap();
    let report: Value =
        serde_json::from_str(&fs::read_to_string(dir.0.join("checked-results.json")).unwrap())
            .unwrap();
    assert_eq!(report["status"], "timeout");
    assert_eq!(report["complete_records"], false);
    assert!(
        fs::read_to_string(dir.0.join("checked-results.csv"))
            .unwrap()
            .is_empty()
    );
}

#[test]
#[cfg(unix)]
fn make_runner_propagates_checker_failure_timeout_and_preserves_evidence() {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    let dir = Directory::new();
    let cargo = dir.0.join("fake-cargo");
    let smi = dir.0.join("fake-smi");
    fs::write(
        &cargo,
        r#"#!/bin/bash
case "$1" in
  build) exit 0 ;;
  run) echo checker-invoked; exit "$CHECK_EXIT" ;;
  test)
    if [[ " $* " == *' --no-run '* ]]; then exit 0; fi
    if [[ "$STALL" == 1 ]]; then sleep 10; fi
    echo benchmark-invoked; exit "$BENCH_EXIT" ;;
  *) exit 90 ;;
esac
"#,
    )
    .unwrap();
    fs::write(
        &smi,
        "#!/bin/bash\nprintf 'synthetic hardware fixture\\n'\n",
    )
    .unwrap();
    for path in [&cargo, &smi] {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (name, bench, checker, stall, expected_raw) in [
        ("success", "0", "0", "0", "0"),
        ("checker-failed", "0", "1", "0", "0"),
        ("benchmark-failed", "101", "0", "0", "2"),
        ("timeout", "0", "2", "1", "124"),
    ] {
        let output = dir.0.join(name);
        let invoke = || {
            Command::new("make")
                .current_dir(&root)
                .args([
                    "--no-print-directory",
                    "bench-soil",
                    "CASE=active-domain",
                    "MODE=compare",
                    "RUN_WITH=",
                    "SOIL_BENCH_SECONDS=1",
                ])
                .arg(format!("CARGO={}", cargo.display()))
                .arg(format!("NVIDIA_SMI={}", smi.display()))
                .arg(format!("SOIL_RUN_DIR={}", output.display()))
                .env("BENCH_EXIT", bench)
                .env("CHECK_EXIT", checker)
                .env("STALL", stall)
                .output()
                .unwrap()
        };
        let result = invoke();
        assert_eq!(
            result.status.success(),
            name == "success",
            "{name}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            fs::read_to_string(output.join("active-domain.exit"))
                .unwrap()
                .trim(),
            expected_raw
        );
        assert!(
            fs::read_to_string(output.join("checker.log"))
                .unwrap()
                .contains("checker-invoked")
        );
        assert_eq!(
            fs::read_to_string(output.join("checker.exit"))
                .unwrap()
                .trim(),
            if checker == "0" { "0" } else { "2" }
        );
        let log = fs::read(output.join("active-domain.log")).unwrap();
        assert!(!invoke().status.success());
        assert_eq!(fs::read(output.join("active-domain.log")).unwrap(), log);
    }
}
