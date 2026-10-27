use super::*;

pub(super) fn method20_source(input: String) -> String {
    use crate::soil_newton_parallel_release as parallel;
    parallel::norm_tree::source(parallel::schur_inverse::source(parallel::core_source(
        parallel::source(ExperimentalSoilNewtonGpu::cached_contact_source(
            ExperimentalSoilNewtonGpu::cached_delta_source(
                ExperimentalSoilNewtonGpu::stalled_release_source(input),
            ),
        )),
    )))
}

fn source(input: String, step: u32) -> String {
    assert!(step < 64);
    let mut s = method20_source(input);
    replace(
        &mut s,
        "state_norm=scratch_norm();prepare_linear();hybrid_direction(lane);",
        &format!(
            "state_norm=scratch_norm();prepare_linear();if step=={step}u {{if lane==0u {{report.diagnostic_limit=2147483648u+step;report.steps=step;}}return;}}hybrid_direction(lane);"
        ),
    );
    marked(s)
}

pub(crate) fn capture_linearization(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    original: &std::path::Path,
    output: &std::path::Path,
) {
    use crate::soil_shared_wall_friction_gpu::capture;
    let step = std::env::var("MOLLA_NEWTON_LINEAR_CAPTURE_STEP")
        .map(|s| s.parse::<u32>().unwrap())
        .unwrap_or(0);
    let packet = std::fs::read(original.join("newton.bin")).unwrap();
    let hash = blake3::hash(&packet).to_hex().to_string();
    std::fs::create_dir_all(output).unwrap();
    let root = output.join(format!("{hash}-step-{step}"));
    std::fs::create_dir(&root).expect("fresh linearization directory");
    let borrowed = [
        ("mass", input.mass),
        ("free", input.free_rates),
        ("rows", input.point_rows),
        ("response", input.point_response),
        ("contacts", input.contacts),
        ("upstream", input.upstream_status),
    ];
    let solver = ExperimentalSoilNewtonGpu::new_with_storage(
        engine,
        input,
        cap,
        NewtonVariant::Fischer,
        |s| source(s, step),
        "newton",
        true,
    )
    .unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    solver.encode_reset(&mut encoder);
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    let report = capture::read(engine, solver.status_buffer());
    assert_eq!(
        u32::from_le_bytes(report[28..32].try_into().unwrap()),
        0x80000000 | step,
        "requested linearization reached"
    );
    let status = capture::read(engine, &solver.scratch[6]);
    assert_eq!(
        u32::from_le_bytes(status[..4].try_into().unwrap()),
        0,
        "valid captured linear preparation"
    );
    let active = u32::from_le_bytes(status[8..12].try_into().unwrap());
    assert!(active <= cap.contacts);
    let mut files = serde_json::Map::new();
    for (name, buffer) in borrowed.into_iter().chain([
        ("blocks", &solver.scratch[0]),
        ("inverse", &solver.scratch[1]),
        ("linear-status", &solver.scratch[6]),
        ("active", &solver.scratch[8]),
        ("state", solver.state_buffer()),
        ("report", solver.status_buffer()),
    ]) {
        let bytes = capture::read(engine, buffer);
        std::fs::write(root.join(format!("{name}.bin")), &bytes).unwrap();
        files.insert(name.into(),serde_json::json!({"bytes":bytes.len(),"blake3":blake3::hash(&bytes).to_hex().to_string()}));
    }
    std::fs::write(root.join("original-newton.bin"), packet).unwrap();
    std::fs::write(root.join("manifest.json"),serde_json::to_vec_pretty(&serde_json::json!({"schema":1,"width":cap.dofs,"contacts":cap.contacts,"active":active,"step":step,"iterations":cap.krylov_iterations,"original":original,"original_blake3":hash,"profile":"method20-pre-hybrid","files":files})).unwrap()).unwrap();
    println!(
        "distributed-linear-capture path={} width={} contacts={} active={active} step={step}",
        root.display(),
        cap.dofs,
        cap.contacts
    );
}

#[test]
fn newton_distributed_capture_source_contract() {
    let s = source(shader_source_for(NewtonVariant::Fischer), 0);
    assert!(s.contains("prepare_linear();if step==0u"));
    crate::soil_newton_protected::compile(&s, "newton").unwrap();
}
