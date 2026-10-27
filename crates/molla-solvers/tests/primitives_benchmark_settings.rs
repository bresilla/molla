#[path = "../benches/primitives/settings.rs"]
mod settings;

fn parse(args: &[&str]) -> settings::Settings {
    settings::Settings::parse(
        args.iter().map(|arg| arg.to_string()),
        &["distance", "mixed"],
    )
    .unwrap()
    .unwrap()
}

#[test]
fn cargo_bench_preserves_full_performance_defaults() {
    let settings = parse(&["--bench"]);
    assert!(!settings.worker && !settings.smoke && !settings.trace_frames);
    assert_eq!(settings.primitive, "all");
    assert_eq!(settings.backend, "both");
    assert_eq!(settings.counts, [1, 100, 1000, 10000]);
    assert_eq!(settings.warmup, 20);
    assert_eq!(settings.steps, 1000);
    assert_eq!(settings.substeps, 4);
    assert_eq!(settings.dt, 1. / 240.);
    assert_eq!(settings.case_timeout_seconds, 3600);
}

#[test]
fn cargo_test_and_explicit_smoke_keep_all_cases_and_both_backends() {
    for args in [
        &[][..],
        &["--test"][..],
        &["--bench", "--smoke"][..],
        &["--test", "--bench"][..],
    ] {
        let settings = parse(args);
        assert!(settings.smoke && !settings.worker);
        assert_eq!(settings.primitive, "all");
        assert_eq!(settings.backend, "both");
        assert_eq!(settings.counts, [1]);
        assert_eq!(settings.warmup, 1);
        assert_eq!(settings.steps, 2);
        assert_eq!(settings.substeps, 4);
    }
}

#[test]
fn explicit_overrides_survive_mode_selection_and_flag_order() {
    let settings = parse(&[
        "--counts",
        "100,1000",
        "--backend",
        "gpu",
        "--primitive",
        "mixed",
        "--warmup",
        "0",
        "--steps",
        "3",
        "--substeps",
        "2",
        "--dt",
        "0.01",
        "--case-timeout-seconds",
        "30",
        "--trace-frames",
        "--smoke",
        "--bench",
    ]);
    assert!(settings.smoke && settings.trace_frames);
    assert_eq!(settings.primitive, "mixed");
    assert_eq!(settings.backend, "gpu");
    assert_eq!(settings.counts, [100, 1000]);
    assert_eq!(
        (settings.warmup, settings.steps, settings.substeps),
        (0, 3, 2)
    );
    assert_eq!(settings.dt, 0.01);
    assert_eq!(settings.case_timeout_seconds, 30);
}

#[test]
fn worker_keeps_one_case_and_inherits_smoke_mode_explicitly() {
    for smoke in [false, true] {
        let mut args = vec![
            "--worker",
            "--primitive",
            "distance",
            "--backend",
            "gpu",
            "--counts",
            "1000",
        ];
        if smoke {
            args.push("--smoke");
        }
        let settings = parse(&args);
        assert!(settings.worker);
        assert_eq!(settings.smoke, smoke);
        assert_eq!(settings.steps, if smoke { 2 } else { 1000 });
        assert_eq!(settings.counts, [1000]);
    }
    for args in [
        &["--worker"][..],
        &["--worker", "--smoke"][..],
        &[
            "--worker",
            "--primitive",
            "distance",
            "--counts",
            "1",
            "--backend",
            "both",
        ][..],
        &[
            "--worker",
            "--primitive",
            "distance",
            "--counts",
            "1,100",
            "--backend",
            "gpu",
        ][..],
    ] {
        assert!(
            settings::Settings::parse(args.iter().map(|arg| arg.to_string()), &["distance"])
                .is_err()
        );
    }
}

#[test]
fn invalid_values_and_missing_arguments_fail_closed() {
    for args in [
        ["--counts", "0"],
        ["--counts", ""],
        ["--counts", "-1"],
        ["--backend", "auto"],
        ["--primitive", "unknown"],
        ["--steps", "0"],
        ["--substeps", "0"],
        ["--case-timeout-seconds", "0"],
        ["--dt", "0"],
        ["--dt", "-1"],
        ["--dt", "NaN"],
        ["--dt", "inf"],
        ["--unknown", "1"],
    ] {
        assert!(
            settings::Settings::parse(args.map(str::to_owned), &["distance"]).is_err(),
            "{args:?}"
        );
    }
    assert!(settings::Settings::parse(["--counts".to_owned()], &["distance"]).is_err());
}

#[test]
fn help_exits_without_selecting_or_executing_cases() {
    assert!(
        settings::Settings::parse(["--help".to_owned()], &[])
            .unwrap()
            .is_none()
    );
    assert!(settings::HELP.contains("--bench"));
    assert!(settings::HELP.contains("--smoke"));
}

#[test]
fn total_frame_count_overflow_is_rejected_before_worker_start() {
    assert!(
        settings::Settings::parse(
            [
                "--warmup".into(),
                usize::MAX.to_string(),
                "--steps".into(),
                "1".into()
            ],
            &["distance"],
        )
        .is_err()
    );
}
