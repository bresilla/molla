use super::*;

#[test]
#[ignore = "requires GPU compiler caching and a successful protected Newton packet"]
fn resident_newton_compiler_cache_roundtrip() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_NEWTON_FAILURE_REPLAY").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("MOLLA_NEWTON_CACHE_RESULT").unwrap());
    assert!(!output.exists());
    assert!(std::env::var_os("MOLLA_NEWTON_FAILURE_OUTPUT").is_none());
    assert!(std::env::var_os("MOLLA_NEWTON_SUCCESS_EXACT").is_some());
    assert_eq!(std::env::var("MOLLA_NEWTON_FAILURE_REPEATS").unwrap(), "3");
    let features = wgpu::Features::PIPELINE_CACHE | wgpu::Features::PASSTHROUGH_SHADERS;
    let mut engine = Engine::new_gpu_with_features(features).unwrap();
    engine.enable_pipeline_cache(None).unwrap();
    let mut results = Vec::new();
    let mut replay = |phase: &str, engine: &Engine| {
        let start = std::time::Instant::now();
        replay_packet(&root, Some(engine));
        let seconds = start.elapsed().as_secs_f64();
        eprintln!("Newton compiler cache phase={phase} replay_seconds={seconds}");
        results.push(serde_json::json!({"phase": phase, "replay_seconds": seconds}));
    };
    replay("empty_cache", &engine);
    engine.clear_gpu_program_cache();
    replay("same_device_rebuild", &engine);
    let snapshot = engine.pipeline_cache_snapshot().unwrap().unwrap();
    assert!(!snapshot.as_bytes().is_empty());
    let cache_bytes = snapshot.as_bytes().len();
    drop(engine);
    let mut engine = Engine::new_gpu_with_features(features).unwrap();
    engine.enable_pipeline_cache(Some(&snapshot)).unwrap();
    replay("new_device_snapshot", &engine);
    let result = serde_json::json!({
        "acceptance": false,
        "scope": "compiler cache lifecycle; not frame performance or isolated cache-hit proof",
        "snapshot_bytes": cache_bytes,
        "phases": results,
    });
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
}
