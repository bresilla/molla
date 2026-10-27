#[path = "../benches/support/camera_latency_smoke.rs"]
mod camera_latency_smoke;
#[path = "../benches/support/cloud_benchmark_sizes.rs"]
mod cloud_benchmark_sizes;
#[path = "../benches/support/run_mode.rs"]
mod run_mode;

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn cargo_test_and_explicit_smoke_use_bounded_sampling() {
    for values in [
        &[][..],
        &["--test"],
        &["--smoke"],
        &["--bench", "--test"],
        &["--bench", "--smoke"],
    ] {
        let values = args(values);
        assert!(run_mode::is_smoke(&values));
        assert_eq!(run_mode::defaults(&values), (2, 1, 1));
        assert_eq!(
            camera_latency_smoke::dimensions(1280, 720, run_mode::is_smoke(&values)),
            (128, 72)
        );
    }
}

#[test]
fn cargo_bench_preserves_original_performance_defaults() {
    for values in [
        &["--bench"][..],
        &["--bench", "--physics-matrix"],
        &["--bench", "--camera-matrix"],
        &["--bench", "--lidar-matrix"],
    ] {
        let values = args(values);
        assert!(!run_mode::is_smoke(&values));
        assert_eq!(run_mode::defaults(&values), (64, 8, 3));
        for (width, height) in [(640, 480), (1280, 720)] {
            assert_eq!(
                camera_latency_smoke::dimensions(width, height, run_mode::is_smoke(&values)),
                (width, height)
            );
        }
    }
}

#[test]
fn camera_smoke_dimensions_keep_nonzero_images_and_bound_aperture_rays() {
    assert_eq!(camera_latency_smoke::dimensions(1, 1, true), (1, 1));
    let (width, height) = camera_latency_smoke::dimensions(1280, 720, true);
    assert_eq!(width * height * 4 * 4, 147_456);
    assert_eq!(camera_latency_smoke::dimensions(640, 480, true), (64, 48));
}

#[test]
fn cloud_smoke_bounds_storage_and_preserves_performance_sizes() {
    for values in [
        &[][..],
        &["--test"],
        &["sensor_cloud/gpu_completed"],
        &["--bench", "--test"],
    ] {
        let values = args(values);
        assert!(cloud_benchmark_sizes::is_smoke(&values));
        assert_eq!(cloud_benchmark_sizes::sizes(&values), &[1024, 65_536]);
    }
    for values in [&["--bench"][..], &["--bench", "--sample-size", "10"]] {
        let values = args(values);
        assert!(!cloud_benchmark_sizes::is_smoke(&values));
        assert_eq!(
            cloud_benchmark_sizes::sizes(&values),
            &[1024, 65_536, 1_000_003]
        );
    }
}
