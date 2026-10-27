use super::*;

#[derive(Debug)]
struct Prefix {
    limit: u64,
    control: Option<(f64, Vec<usize>)>,
}

fn metadata(rows: &[serde_json::Value], slots: usize) -> Vec<Prefix> {
    assert!(!rows.is_empty() && rows.len() <= 65);
    let mut previous = None;
    rows.iter()
        .map(|row| {
            let limit = row["limit"].as_u64().unwrap();
            assert!(limit <= 64 && previous.is_none_or(|old| old < limit));
            previous = Some(limit);
            let report = row["report"].as_array().unwrap();
            assert_eq!(report.len(), 8);
            let words: Vec<_> = report
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
            assert!(u64::from(words[1]) <= limit);
            let control = if words[0] == 0 {
                assert!(row["control"].is_null());
                None
            } else {
                assert_eq!(words[0], 8);
                assert_eq!(u64::from(words[1]), limit);
                let control = row["control"].as_object().unwrap();
                let smoothing = control["smoothing"].as_f64().unwrap();
                assert!(smoothing.is_finite() && (0.0..=0.01).contains(&smoothing));
                let visited: Vec<_> = control["visited"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
                    .collect();
                assert!(visited.len() <= 8);
                let releases = control["releases"].as_f64().unwrap();
                assert_eq!(releases, visited.len() as f64);
                for (i, slot) in visited.iter().enumerate() {
                    assert!(*slot < slots && !visited[..i].contains(slot));
                }
                Some((smoothing, visited))
            };
            Prefix { limit, control }
        })
        .collect()
}

pub(super) fn run(
    reference: &Reference,
    slots: usize,
    input: &std::path::Path,
    output: &std::path::Path,
) {
    let captured = match std::env::var("MOLLA_NEWTON_FAILURE_CPU_PREFIX_SMOOTHING") {
        Ok(v) if v == "captured" => true,
        Ok(v) if v == "zero" => false,
        Err(std::env::VarError::NotPresent) => false,
        _ => panic!("prefix smoothing must be zero or captured"),
    };
    let path = input.join("prefixes.json");
    assert!(path.metadata().unwrap().len() <= 1024 * 1024);
    let rows: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let prefixes = metadata(&rows, slots);
    let mut results = Vec::new();
    for prefix in prefixes {
        let limit = prefix.limit;
        let path = input.join(format!("states-{limit}.bin"));
        assert_eq!(path.metadata().unwrap().len(), (slots.max(1) * 64) as u64);
        let bytes = std::fs::read(path).unwrap();
        let word =
            |i: usize| f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()) as f64;
        let x: Vec<_> = reference
            .contacts
            .iter()
            .flat_map(|c| {
                (0..6)
                    .map(|i| {
                        let at = 16 * c.index + if i < 3 { i } else { i + 1 };
                        word(at) + word(at + 8)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(x.iter().all(|v| v.is_finite()));
        let Some((smoothing, ids)) = prefix.control else {
            results.push(
                serde_json::json!({"limit":limit,"completed_without_control":true,
                "smoothing":null,"visited":null,"cases":[]}),
            );
            continue;
        };
        let visited: Vec<_> = reference
            .contacts
            .iter()
            .map(|c| ids.contains(&c.index))
            .collect();
        assert_eq!(visited.iter().filter(|&&v| v).count(), ids.len());
        let mut reference = reference.clone();
        for c in &mut reference.contacts {
            c.smoothing = if captured { smoothing } else { 0. };
        }
        let base = norm(&reference.residual(&x));
        assert!(base.is_finite());
        let mut cases = Vec::new();
        for conditioned in [false, true] {
            let mut problem = reference.clone();
            problem.conditioned_pivot = conditioned;
            let modes = [
                local_pivot::PivotMode::Release,
                local_pivot::PivotMode::AllBranches,
                local_pivot::PivotMode::Settle,
            ];
            for mode in modes.into_iter().take(if captured { 3 } else { 2 }) {
                let all_branches = mode != local_pivot::PivotMode::Release;
                let mut next_visited = visited.clone();
                let (trial, candidates) =
                    local_pivot::pivot_diagnostic(&problem, &x, base, &mut next_visited, mode);
                let selected = next_visited
                    .iter()
                    .zip(&visited)
                    .position(|(next, old)| *next && !*old);
                let mut row = serde_json::json!({"conditioned":conditioned,"all_branches":all_branches,
                    "residual":trial.as_ref().map(|t| norm(&problem.residual(t))),
                    "physical":trial.as_ref().map(|t| problem.physical_error(t)),
                    "selected":selected.map(|i| problem.contacts[i].index),
                    "old_local":selected.map(|i| x[6*i..6*i+6].to_vec()),
                    "new_local":selected.and_then(|i| trial.as_ref().map(|t| t[6*i..6*i+6].to_vec()))});
                if captured {
                    row["policy"] = serde_json::json!(match mode {
                        local_pivot::PivotMode::Release => "release",
                        local_pivot::PivotMode::AllBranches => "all_branches",
                        local_pivot::PivotMode::Settle => "settle_top",
                    });
                    row["candidates"] = serde_json::json!(
                        candidates.into_iter().map(|c| c.json()).collect::<Vec<_>>()
                    );
                    row["decreases_merit"] = serde_json::json!(
                        trial
                            .as_ref()
                            .is_some_and(|t| norm(&problem.residual(t)) < base)
                    );
                }
                cases.push(row);
            }
        }
        let mut row = serde_json::json!({"limit":limit,"base_residual":base,
            "base_physical":reference.physical_error(&x),"visited":ids,"cases":cases});
        if captured {
            row["smoothing"] = serde_json::json!(smoothing);
            row["scope"] = serde_json::json!(
                "independent f64 candidates using captured smoothing and prepared response; not a GPU solve or acceptance"
            );
        }
        results.push(row);
    }
    assert!(!output.exists());
    std::fs::write(output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

#[test]
fn newton_prefix_oracle_metadata_contract() {
    let row = |limit: u64| {
        serde_json::json!({"limit":limit,"report":[8,limit,0,0,0,0,0,0],
        "control":{"smoothing":0.001,"releases":1.0,"visited":[115]}})
    };
    let legacy: Vec<_> = (48..=64).map(row).collect();
    assert_eq!(metadata(&legacy, 4096).len(), 17);
    let mut current: Vec<_> = [0, 1, 2, 4, 8, 16, 24, 32, 40]
        .into_iter()
        .map(row)
        .collect();
    for limit in [48, 56, 64] {
        current.push(serde_json::json!({"limit":limit,"report":[0,42,0,0,0,0,0,0],"control":null}));
    }
    let parsed = metadata(&current, 4096);
    assert_eq!(parsed.len(), 12);
    assert_eq!(parsed[5].control.as_ref().unwrap().0, 0.001);
    assert!(parsed[9..].iter().all(|p| p.control.is_none()));
    for input in [
        vec![],
        vec![row(65)],
        vec![row(2), row(1)],
        vec![row(1), row(1)],
        vec![row(1); 66],
    ] {
        assert!(std::panic::catch_unwind(|| metadata(&input, 4096)).is_err());
    }
    for (pointer, value) in [
        ("/control", serde_json::Value::Null),
        ("/control/smoothing", serde_json::json!(-0.1)),
        ("/control/visited", serde_json::json!([115, 115])),
        ("/control/visited", serde_json::json!([4096])),
        ("/control/releases", serde_json::json!(0.5)),
        ("/report/0", serde_json::json!(0)),
        ("/report/1", serde_json::json!(0)),
    ] {
        let mut invalid = row(1);
        *invalid.pointer_mut(pointer).unwrap() = value;
        assert!(std::panic::catch_unwind(|| metadata(&[invalid], 4096)).is_err());
    }
}
