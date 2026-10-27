use super::*;

pub(crate) struct Trace {
    report: wgpu::Buffer,
    packets: Vec<(usize, wgpu::Buffer)>,
    next: usize,
    capacity: usize,
}

impl Trace {
    fn validate(capacity: usize, selected: &[usize], packet_bytes: u64) -> Result<()> {
        if capacity == 0
            || capacity > 1122
            || selected.len() > 34
            || selected.iter().any(|&i| i >= capacity)
            || selected.windows(2).any(|w| w[0] >= w[1])
            || packet_bytes < 128
            || packet_bytes % 4 != 0
            || packet_bytes
                .checked_mul(selected.len() as u64)
                .and_then(|bytes| bytes.checked_add(capacity as u64 * 32))
                .is_none_or(|bytes| bytes > 64 * 1024 * 1024)
        {
            return Err(invalid(
                "invalid Newton work trace capacity or 64 MiB diagnostic budget",
            ));
        }
        Ok(())
    }

    fn new(
        device: &wgpu::Device,
        capacity: usize,
        selected: &[usize],
        packet_bytes: u64,
    ) -> Result<Self> {
        Self::validate(capacity, selected, packet_bytes)?;
        let buffer = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("diagnostic Newton work trace"),
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Ok(Self {
            report: buffer(capacity as u64 * 32),
            packets: selected
                .iter()
                .map(|&i| (i, buffer(packet_bytes)))
                .collect(),
            next: 0,
            capacity,
        })
    }

    pub(crate) fn validate_next(&self) -> Result<()> {
        if self.next >= self.capacity {
            return Err(invalid("Newton work trace exhausted"));
        }
        Ok(())
    }

    pub(crate) fn encode(&mut self, engine: &Engine, status: &wgpu::Buffer, packet: &wgpu::Buffer) {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(status, 0, &self.report, self.next as u64 * 32, 32);
        if let Some((_, output)) = self.packets.iter().find(|(i, _)| *i == self.next) {
            assert_eq!(output.size(), packet.size());
            encoder.copy_buffer_to_buffer(packet, 0, output, 0, output.size());
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        self.next += 1;
    }
}

fn capture_frame(value: Option<&str>) -> Result<usize> {
    let frame = value
        .unwrap_or("1")
        .parse::<usize>()
        .map_err(|_| invalid("capture frame must be 1..33"))?;
    if !(1..=33).contains(&frame) {
        return Err(invalid("capture frame must be 1..33"));
    }
    Ok(frame)
}

fn native_capture(value: Option<&str>) -> Result<bool> {
    match value.unwrap_or("featherstone") {
        "featherstone" => Ok(false),
        "native" => Ok(true),
        _ => Err(invalid("capture prediction must be featherstone or native")),
    }
}

fn capture_substeps(value: Option<&str>, frame: usize) -> Result<Vec<usize>> {
    let selected = match value {
        None => (0..if frame == 1 { 8 } else { 34 }).collect(),
        Some(value) => value
            .split(',')
            .map(|v| {
                v.parse::<usize>()
                    .map_err(|_| invalid("capture substeps must be comma-separated indices"))
            })
            .collect::<Result<Vec<_>>>()?,
    };
    if !selected.starts_with(&[0, 1]) {
        return Err(invalid("capture substeps must include initial anchors 0,1"));
    }
    Trace::validate(34, &selected, 128)?;
    Ok(selected)
}

#[test]
fn newton_work_trace_selected_substeps() {
    assert_eq!(
        capture_substeps(None, 1).unwrap(),
        (0..8).collect::<Vec<_>>()
    );
    assert_eq!(
        capture_substeps(None, 22).unwrap(),
        (0..34).collect::<Vec<_>>()
    );
    assert_eq!(
        capture_substeps(Some("0,1,12,17"), 22).unwrap(),
        [0, 1, 12, 17]
    );
    for value in [
        "",
        "12",
        "0",
        "0,1,1",
        "0,1,34",
        "0,1,17,12",
        "0,1,",
        "0,1, 2",
        "0,1,-1",
    ] {
        assert!(capture_substeps(Some(value), 22).is_err(), "{value}");
    }
}

#[test]
fn newton_work_trace_prediction_is_explicit() {
    assert!(!native_capture(None).unwrap());
    assert!(!native_capture(Some("featherstone")).unwrap());
    assert!(native_capture(Some("native")).unwrap());
    for value in ["", "unknown", "Native", "native "] {
        assert!(native_capture(Some(value)).is_err());
    }
}

#[test]
fn newton_work_trace_bounds_are_checked_before_gpu_allocation() {
    let full: Vec<_> = (0..34).collect();
    Trace::validate(34, &full, 1_835_616).unwrap();
    Trace::validate(1122, &[0, 1121], 1_835_616).unwrap();
    Trace::validate(34, &[], 128).unwrap();
    for (capacity, selected, bytes) in [
        (0, vec![], 128),
        (1123, vec![], 128),
        (34, vec![34], 128),
        (34, vec![2, 1], 128),
        (34, vec![1, 1], 128),
        (1122, (0..35).collect(), 128),
        (34, vec![0], 124),
        (34, vec![0], 129),
        (34, vec![0], u64::MAX - 3),
        (34, full, 2 * 1024 * 1024),
    ] {
        assert!(Trace::validate(capacity, &selected, bytes).is_err());
    }
    assert_eq!(capture_frame(None).unwrap(), 1);
    for frame in [1, 14, 33] {
        assert_eq!(capture_frame(Some(&frame.to_string())).unwrap(), frame);
    }
    for invalid in ["0", "34", "", "-1", "1.5", "18446744073709551616"] {
        assert!(capture_frame(Some(invalid)).is_err());
    }
}

fn write_trace_poses(
    file: &mut std::fs::File,
    tractor: &crate::soil_tractor::SoilTractor,
    frame: usize,
) {
    use std::io::Write;
    let handles = tractor.handles();
    let bodies: Vec<_> = [
        handles.chassis,
        handles.wheels[0],
        handles.wheels[1],
        handles.wheels[2],
        handles.wheels[3],
        handles.tine,
    ]
    .into_iter()
    .map(|body| {
        let pose = tractor.world().scene.body_pose(body).unwrap();
        serde_json::json!({"position":pose.position.to_array().map(|v| v as f32),
            "rotation":pose.rotation.to_array().map(|v| v as f32)})
    })
    .collect();
    writeln!(
        file,
        "{}",
        serde_json::json!({"committed_frame":frame,"bodies":bodies})
    )
    .unwrap();
    file.flush().unwrap();
}

#[test]
#[ignore = "requires native/protected GPU; selected-frame cold profile inputs, not performance acceptance"]
fn newton_actual_cold_profile_trace() {
    assert!(!cfg!(feature = "warm-newton"));
    assert_eq!(
        std::env::var("MOLLA_NEWTON_CAPTURE_SUCCESS").as_deref(),
        Ok("1")
    );
    let profile = std::env::var("MOLLA_NEWTON_PROFILE_CAPTURE").unwrap();
    let frame = capture_frame(std::env::var("MOLLA_NEWTON_PROFILE_FRAME").ok().as_deref()).unwrap();
    let selected = capture_substeps(
        std::env::var("MOLLA_NEWTON_PROFILE_SUBSTEPS")
            .ok()
            .as_deref(),
        frame,
    )
    .unwrap();
    let native = native_capture(
        std::env::var("MOLLA_NEWTON_PROFILE_PREDICTION")
            .ok()
            .as_deref(),
    )
    .unwrap();
    use crate::soil_newton_failure::NewtonFailureMethod as Method;
    let method = match profile.as_str() {
        "volatile" => Method::FischerConditionedProtectedVolatile256,
        "soa" => Method::FischerConditionedProtectedSoaScratch256,
        _ => panic!("unknown cold capture profile"),
    };
    let root = std::path::PathBuf::from(std::env::var_os("SOIL_NEWTON_WORK_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let _setup_trace = crate::soil_setup_trace::begin(&root.join("setup.jsonl")).unwrap();
    let start = std::time::Instant::now();
    let engine = {
        let _stage = crate::soil_setup_trace::stage("engine");
        Arc::new(
            Engine::new_gpu_with_features(
                wgpu::Features::SHADER_FLOAT32_ATOMIC
                    | wgpu::Features::PASSTHROUGH_SHADERS
                    | if native {
                        wgpu::Features::SHADER_F64
                    } else {
                        wgpu::Features::empty()
                    },
            )
            .unwrap(),
        )
    };
    assert!(!engine.adapter_info().unwrap().is_software);
    use crate::soil_tractor::SoilTractorTransfer;
    let transfer_name =
        std::env::var("MOLLA_NEWTON_PROFILE_TRANSFER").unwrap_or_else(|_| "atomic".into());
    let transfer = match transfer_name.as_str() {
        "atomic" => SoilTractorTransfer::Atomic,
        "fixed-order" => SoilTractorTransfer::FixedOrder {
            max_gather_bytes: 3_200_000,
        },
        _ => panic!("unknown cold capture transfer"),
    };
    let mut tractor = {
        let _stage = crate::soil_setup_trace::stage("tractor");
        let coupling = crate::mpm::MpmUnilateralConfig {
            node_capacity: 4096,
            search_distance: 0.05,
            max_iterations: 64,
        };
        if native {
            crate::soil_tractor::SoilTractor::new_native_prediction_with_newton_transfer(
                engine.clone(),
                crate::soil_tractor::SoilTractorConfig::default(),
                coupling,
                0.3,
                crate::mpm::MpmGridStorage::Dense,
                1.0 / 60.0,
                transfer,
            )
            .unwrap()
        } else {
            crate::soil_tractor::SoilTractor::new_native_explicit_with_paired_newton_transfer(
                engine.clone(),
                crate::soil_tractor::SoilTractorConfig::default(),
                coupling,
                0.3,
                molla_kernels::FeatherstoneExecution::Generic,
                crate::mpm::MpmGridStorage::Dense,
                transfer,
            )
            .unwrap()
        }
    };
    assert_eq!(tractor.world().gpu_uses_native_prediction(), native);
    assert_eq!(
        tractor.world().mpm_soil_newton_backend(),
        Some(method.name())
    );
    let transfer_info = tractor.world().mpm_soil_transfer_info().unwrap();
    let fixed = transfer_name == "fixed-order";
    assert_eq!(transfer_info.fixed_order, fixed);
    assert_eq!(transfer_info.particle_ordering, fixed);
    assert_eq!(transfer_info.parallel_sort, fixed);
    assert!(!transfer_info.shared_chunks);
    assert_eq!(
        transfer_info.gather_bytes,
        if fixed { 3_200_000 } else { 0 }
    );
    assert_eq!(transfer_info.ordering_bytes > 0, fixed);
    let construction_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut poses = std::fs::File::create_new(root.join("poses.jsonl")).unwrap();
    write_trace_poses(&mut poses, &tractor, 0);
    for prefix in 1..frame {
        let _stage = crate::soil_setup_trace::stage("frame.prefix.step");
        tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
        write_trace_poses(&mut poses, &tractor, prefix);
    }
    let projection = tractor
        .world_mut_for_test()
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .projection
        .as_mut()
        .unwrap();
    assert!(
        projection
            .newton_success_seeds()
            .iter()
            .all(Option::is_none)
    );
    let packet_bytes = projection.newton_success_buffers()[0].size();
    projection.newton_work_trace =
        Some(Trace::new(engine.device().unwrap(), 34, &selected, packet_bytes).unwrap());
    {
        let _stage = crate::soil_setup_trace::stage("frame.step");
        tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
    }
    write_trace_poses(&mut poses, &tractor, frame);
    let stats = tractor.world().mpm_soil_stats().unwrap();
    assert_eq!(stats.committed_frames, frame as u64);
    assert!(!stats.faulted);
    let projection = tractor
        .world_mut_for_test()
        .gpu
        .as_ref()
        .unwrap()
        .soil
        .as_ref()
        .unwrap()
        .projection
        .as_ref()
        .unwrap();
    let trace = projection.newton_work_trace.as_ref().unwrap();
    assert_eq!(trace.next, 34);
    let raw = {
        let _stage = crate::soil_setup_trace::stage("reports.readback");
        wall_runtime_read(&engine, &trace.report)
    };
    std::fs::write(root.join("reports.bin"), &raw).unwrap();
    let word = |i: usize| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
    assert!((0..34).all(|i| (0..4).all(|j| word(i * 8 + j) == 0)));
    let mut captured = Vec::new();
    for (i, buffer) in &trace.packets {
        let _stage = crate::soil_setup_trace::stage("packet.export");
        if word(i * 8 + 4) == 0 {
            continue;
        }
        let bytes = wall_runtime_read(&engine, buffer);
        let packet_word =
            |j: usize| u32::from_le_bytes(bytes[j * 4..j * 4 + 4].try_into().unwrap());
        assert_eq!(packet_word(7), method as u32);
        assert_eq!((packet_word(8), packet_word(9)), (1, 0));
        for (p, r) in [(17, 4), (18, 5), (19, 6)] {
            assert_eq!(packet_word(p), word(i * 8 + r));
        }
        crate::soil_newton_failure::capture_success(
            &engine,
            buffer,
            &root.join(format!("packet-{i}")),
            |b| map_read_with_completion(engine.device().unwrap(), b, None),
        )
        .unwrap();
        captured.push(*i);
    }
    std::fs::write(root.join("scope.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "acceptance":false,"profile":profile,"backend":method.name(),"initial_guess":"cold-zero",
        "transfer":transfer_name,"gather_bytes":transfer_info.gather_bytes,"ordering_bytes":transfer_info.ordering_bytes,
        "machine_prediction":if native {"native"} else {"featherstone"},"native_prediction":tractor.world().gpu_uses_native_prediction(),
        "preparation_backend":crate::soil_newton_prepare_gpu::ExperimentalSoilNewtonPrepareGpu::backend_label(),
        "composed_submissions":tractor.world().mpm_soil_composes_submissions(),
        "native_drift_deferred":tractor.world().mpm_soil_defers_native_prediction_drift(),
        "unrestricted_drift_backend":tractor.world().solver.gpu_unrestricted_drift_backend(),
        "particles":50000,"committed_frames":frame,"captured_frame":frame,"prefix_frames":frame-1,"substeps":34,"selected_indices":selected,"captured_indices":captured,
        "construction_ms":construction_ms,"elapsed_seconds":start.elapsed().as_secs_f64(),
        "diagnostic_packet_bytes":selected.len() as u64 * packet_bytes,"diagnostic_extra_submissions":34,
        "diagnostic_storage_budget_bytes":64*1024*1024,
        "scope":"selected frame of paired tractor with explicit transfer; prefix frames have no work trace; only nontrivial published solves captured; instrumented, not performance acceptance",
        "physical_calibration":"uncalibrated"
    })).unwrap()).unwrap();
    assert!(
        captured.starts_with(&[0, 1]),
        "first two substeps required for profile isolation"
    );
}

#[test]
#[ignore = "requires native/protected/f64 GPU; bounded 50k workload attribution, not acceptance"]
fn newton_actual_workload_trace() {
    assert!(cfg!(feature = "shared-conditioned-core-newton"));
    assert!(!cfg!(feature = "warm-newton"));
    assert_eq!(
        std::env::var("MOLLA_NEWTON_CAPTURE_SUCCESS").as_deref(),
        Ok("1")
    );
    let root = std::path::PathBuf::from(std::env::var_os("SOIL_NEWTON_WORK_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let start = std::time::Instant::now();
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::SHADER_F64
                | wgpu::Features::PASSTHROUGH_SHADERS
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_prediction_with_newton_grid(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
        crate::mpm::MpmUnilateralConfig {
            node_capacity: 4096,
            search_distance: 0.05,
            max_iterations: 64,
        },
        0.3,
        crate::mpm::MpmGridStorage::Compact {
            tile_capacity: 1024,
        },
        1.0 / 60.0,
    )
    .unwrap();
    assert_eq!(
        tractor.world().mpm_soil_newton_backend(),
        Some(
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-256-pairs"
        )
    );
    tractor
        .prepare_soil_frame_profiling_with_preparation(34)
        .unwrap();
    let selected = [
        3 * 34,
        16 * 34 + 5,
        16 * 34 + 7,
        16 * 34 + 10,
        21 * 34 + 4,
        21 * 34 + 5,
        21 * 34 + 11,
        26 * 34 + 7,
    ];
    let projection = tractor
        .world_mut_for_test()
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .projection
        .as_mut()
        .unwrap();
    let packet_bytes = projection.newton_success_buffers()[0].size();
    projection.newton_work_trace =
        Some(Trace::new(engine.device().unwrap(), 1122, &selected, packet_bytes).unwrap());
    let construction_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut rows = Vec::new();
    for frame in 0..33usize {
        tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
        assert_eq!(
            tractor.world().mpm_soil_stats().unwrap().committed_frames,
            (frame + 1) as u64
        );
        let profile = tractor.read_soil_frame_profile().unwrap().unwrap();
        assert_eq!(profile.committed_frame, (frame + 1) as u64);
        let timings: Vec<_> = profile
            .profile
            .contact_stages
            .iter()
            .filter(|t| t.stage == crate::mpm::MpmProfileStage::ContactSolve)
            .collect();
        assert_eq!(timings.len(), 34);
        for (substep, timing) in timings.into_iter().enumerate() {
            assert_eq!(timing.substep, Some(substep));
            assert!(timing.elapsed_ms.is_finite() && timing.elapsed_ms > 0.0);
            rows.push(serde_json::json!({"frame":frame,"substep":substep,"warmup":frame<3,"gpu_ms":timing.elapsed_ms}));
        }
    }
    let projection = tractor
        .world_mut_for_test()
        .gpu
        .as_ref()
        .unwrap()
        .soil
        .as_ref()
        .unwrap()
        .projection
        .as_ref()
        .unwrap();
    let trace = projection.newton_work_trace.as_ref().unwrap();
    assert_eq!(trace.next, 1122);
    assert!(trace.validate_next().is_err());
    let raw = wall_runtime_read(&engine, &trace.report);
    std::fs::write(root.join("reports.bin"), &raw).unwrap();
    let word = |i: usize| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
    for (i, row) in rows.iter_mut().enumerate() {
        assert!((0..4).all(|j| word(i * 8 + j) == 0));
        row["steps"] = serde_json::json!(word(i * 8 + 4));
        row["linear_iterations"] = serde_json::json!(word(i * 8 + 5));
        row["trials"] = serde_json::json!(word(i * 8 + 6));
    }
    std::fs::write(
        root.join("solves.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
    for (i, buffer) in &trace.packets {
        let bytes = wall_runtime_read(&engine, buffer);
        let packet_word =
            |j: usize| u32::from_le_bytes(bytes[j * 4..j * 4 + 4].try_into().unwrap());
        assert_eq!(packet_word(7), 25);
        assert!(
            word(i * 8 + 4) > 0,
            "selected packet must be the current nontrivial solve"
        );
        for (p, r) in [(17, 4), (18, 5), (19, 6)] {
            assert_eq!(packet_word(p), word(i * 8 + r));
        }
        crate::soil_newton_failure::capture_success(
            &engine,
            buffer,
            &root.join(format!("packet-{i}")),
            |b| map_read_with_completion(engine.device().unwrap(), b, None),
        )
        .unwrap();
    }
    std::fs::write(root.join("scope.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "acceptance":false,"particles":50000,"frames":33,"substeps":34,"reports":1122,
        "method":25,"selected_indices":selected,"construction_ms":construction_ms,
        "diagnostic_report_bytes":raw.len(),"diagnostic_packet_bytes":8*packet_bytes,
        "diagnostic_extra_submissions":1122,"elapsed_seconds":start.elapsed().as_secs_f64(),
        "scope":"instrumented actual compact native-prediction tractor; report copy and success-capture overhead; not timing acceptance; captures selected from previous run, not necessarily current slowest",
        "physical_calibration":"uncalibrated"
    })).unwrap()).unwrap();
}
