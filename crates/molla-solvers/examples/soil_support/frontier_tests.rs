use super::*;

fn profile(steps: usize) -> Value {
    let mut stages =
        vec![json!({"stage":"initialization","substep":null,"start_ms":0.0,"gpu_ms":1.0})];
    let mut contacts = Vec::new();
    let mut time = 1.0;
    for step in 0..steps {
        for name in STAGES {
            let duration = if name == "contact-projection" {
                3.0
            } else {
                1.0
            };
            stages.push(json!({"stage":name,"substep":step,"start_ms":time,"gpu_ms":duration}));
            if name == "contact-projection" {
                for (index, child) in [
                    "contact-preparation",
                    "contact-solve",
                    "contact-publication",
                ]
                .into_iter()
                .enumerate()
                {
                    contacts.push(json!({"stage":child,"substep":step,"start_ms":time+index as f64,"gpu_ms":1.0}));
                }
            }
            time += duration;
        }
    }
    json!({"stages":stages,"contact_stages":contacts,"mpm_kernel_gpu_ms":time,"contact_stage_gpu_ms":{"contact-solve":steps as f64}})
}

#[test]
fn quantiles_use_nearest_rank_and_reject_changed_aggregates() {
    let values = (1..=30).rev().map(f64::from).collect::<Vec<_>>();
    let stats = distribution(&values).unwrap();
    assert_eq!(stats["median"], 15.0);
    assert_eq!(stats["p95"], 29.0);
    assert_eq!(stats["p99"], 30.0);
    assert_eq!(stats["mean"], 15.5);
    assert!(distribution(&[f64::MAX, f64::MAX]).is_err());
    check_distribution(&values, &stats).unwrap();
    let mut bad = stats;
    bad["median"] = json!(15.5);
    assert!(check_distribution(&values, &bad).is_err());
    for bad in [vec![], vec![f64::NAN], vec![-1.0], vec![f64::INFINITY]] {
        assert!(distribution(&bad).is_err());
    }
}

#[test]
fn nested_profiles_are_not_double_counted_and_require_full_coverage() {
    let p = profile(2);
    assert_eq!(profile_totals(&p, 2).unwrap(), (21.0, 19.0));
    let mut overflow = p.clone();
    overflow["stages"][0]["start_ms"] = json!(f64::MAX);
    overflow["stages"][0]["gpu_ms"] = json!(f64::MAX);
    assert!(profile_totals(&overflow, 2).is_err());
    let mut continued = p.clone();
    continued["stages"].as_array_mut().unwrap().remove(0);
    continued["mpm_kernel_gpu_ms"] = json!(20.0);
    assert_eq!(profile_totals(&continued, 2).unwrap(), (20.0, 18.0));
    let mut bad = p.clone();
    bad["stages"].as_array_mut().unwrap().pop();
    assert!(profile_totals(&bad, 2).is_err());
    let mut bad = p.clone();
    bad["stages"][2]["substep"] = json!(1);
    assert!(profile_totals(&bad, 2).is_err());
    let mut bad = p.clone();
    bad["stages"][1]["gpu_ms"] = json!(-1.0);
    assert!(profile_totals(&bad, 2).is_err());
    let mut bad = p.clone();
    bad["contact_stages"][1]["gpu_ms"] = json!(2.0);
    assert!(profile_totals(&bad, 2).is_err());
    let mut bad = p.clone();
    bad["mpm_kernel_gpu_ms"] = json!(23.0);
    assert!(profile_totals(&bad, 2).is_err());
    let mut bad = p;
    bad["contact_stages"][0]["stage"] = json!("contact-solve");
    assert!(profile_totals(&bad, 2).is_err());
}

#[test]
#[ignore = "CPU-only saved actual-tractor artifact mutation checks"]
fn saved_frontier_inputs_fail_closed_and_outputs_do_not_overwrite() {
    let reference = PathBuf::from(std::env::var_os("MOLLA_FRONTIER_REFERENCE").unwrap());
    let source = PathBuf::from(std::env::var_os("MOLLA_FRONTIER_INPUT").unwrap());
    let root = PathBuf::from(std::env::var_os("MOLLA_FRONTIER_MUTATIONS").unwrap());
    fs::create_dir(&root).unwrap();
    let copy = root.join("data");
    fs::create_dir(&copy).unwrap();
    for name in [
        "manifest.json",
        "result.json",
        "frames.csv",
        "mpm-profiles.jsonl",
        "machine-states.jsonl",
        "final-particles.bin",
    ] {
        fs::copy(source.join(name), copy.join(name)).unwrap();
    }
    let good = report(&[reference.clone(), copy.clone()]).unwrap();
    assert_eq!(good["acceptance"], false);
    assert_eq!(good["accepted_frontier"], json!([]));
    assert_eq!(good["points"][1]["observed_numerical_gate"], "failed");
    assert!(report(&[source.clone(), source.clone()]).is_err());
    for case in 0..10 {
        let name = match case {
            0 | 1 => "manifest.json",
            2 => "result.json",
            3 => "frames.csv",
            4 | 5 => "mpm-profiles.jsonl",
            6 | 7 => "machine-states.jsonl",
            _ => "final-particles.bin",
        };
        let original = fs::read(copy.join(name)).unwrap();
        let changed = if case <= 2 {
            let mut value: Value = serde_json::from_slice(&original).unwrap();
            match case {
                0 => value["particles"] = json!(49999),
                1 => value["youngs_pa"] = json!(3001),
                _ => value["wall_ms"]["median"] = json!(1.0),
            }
            serde_json::to_vec(&value).unwrap()
        } else if case == 3 {
            String::from_utf8(original.clone())
                .unwrap()
                .lines()
                .take(33)
                .collect::<Vec<_>>()
                .join("\n")
                .into_bytes()
        } else if case <= 7 {
            let mut values: Vec<Value> = String::from_utf8(original.clone())
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            match case {
                4 => {
                    values.pop();
                }
                5 => {
                    values[0]["contact_stages"].as_array_mut().unwrap().pop();
                }
                6 => values[0]["bodies"][1]["body"] = values[0]["bodies"][0]["body"].clone(),
                _ => values[0]["joint_q"][0] = json!(f64::MAX),
            }
            values
                .iter()
                .map(|v| serde_json::to_string(v).unwrap())
                .collect::<Vec<_>>()
                .join("\n")
                .into_bytes()
        } else {
            let mut bytes = original.clone();
            if case == 8 {
                bytes.truncate(bytes.len() - 4);
            } else {
                bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
            bytes
        };
        fs::write(copy.join(name), changed).unwrap();
        let error = report(&[reference.clone(), copy.clone()]).unwrap_err();
        println!("case={case} file={name} rejected: {error}");
        fs::write(copy.join(name), original).unwrap();
    }
    let output = root.join("report");
    let args = vec![
        "--reference".to_owned(),
        reference.to_string_lossy().into_owned(),
        "--input".into(),
        copy.to_string_lossy().into_owned(),
        "--output".into(),
        output.to_string_lossy().into_owned(),
    ];
    assert_eq!(run(args.clone().into_iter()).unwrap(), 0);
    let bytes = fs::read(output.join("frontier.json")).unwrap();
    assert!(run(args.into_iter()).is_err());
    assert_eq!(fs::read(output.join("frontier.json")).unwrap(), bytes);
}

#[test]
fn trajectory_comparison_requires_complete_unique_body_records() {
    let mut state = json!({"joint_q":vec![0.0;11],"joint_qd":vec![0.0;10],"bodies":(0..6).map(|id| json!({"body":format!("body-{id}"),"position":[0,0,0],"rotation":[0,0,0,1],"linear_velocity":[0,0,0],"angular_velocity":[0,0,0]})).collect::<Vec<_>>()});
    let (_, values) = state_values(&state).unwrap();
    assert_eq!(values.len(), 99);
    state["bodies"][1]["body"] = state["bodies"][0]["body"].clone();
    assert!(state_values(&state).is_err());
    state["bodies"][1]["body"] = json!("body-1");
    state["joint_qd"] = json!([0.0]);
    assert!(state_values(&state).is_err());
}

#[test]
fn frontier_cli_rejects_missing_and_duplicate_options() {
    for args in [
        vec![],
        vec!["--reference"],
        vec!["--reference", "a", "--reference", "b"],
        vec!["--unknown", "a"],
        vec!["--reference", "a", "--output", "b"],
    ] {
        assert!(run(args.into_iter().map(str::to_owned)).is_err());
    }
}
