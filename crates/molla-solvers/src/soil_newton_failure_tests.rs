use super::*;

#[test]
fn newton_seed_replay_requires_hash_shape_and_provenance() {
    let root = std::env::temp_dir().join(format!(
        "newton-seed-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let legacy = serde_json::json!({});
    assert!(replay_seed(&root, &legacy, 1).unwrap().is_none());
    let mut bytes = vec![0u8; 64];
    bytes[..4].copy_from_slice(&1.25f32.to_le_bytes());
    let metadata = |b: &[u8]| serde_json::json!({"initial_guess":"retained-seed","seed_bytes":b.len(),"seed_blake3":blake3::hash(b).to_hex().as_str()});
    let meta = metadata(&bytes);
    assert!(replay_seed(&root, &meta, 1).is_err());
    std::fs::write(root.join("seed.bin"), &bytes).unwrap();
    assert_eq!(replay_seed(&root, &meta, 1).unwrap(), Some(bytes.clone()));
    assert!(replay_seed(&root, &legacy, 1).is_err());
    assert!(replay_seed(&root, &meta, 2).is_err());
    assert!(replay_seed(&root, &meta, 16385).is_err());
    for field in ["initial_guess", "seed_bytes", "seed_blake3"] {
        let mut missing = meta.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(replay_seed(&root, &missing, 1).is_err());
    }
    bytes[0] ^= 1;
    std::fs::write(root.join("seed.bin"), &bytes).unwrap();
    assert!(replay_seed(&root, &meta, 1).is_err());
    for (offset, value) in [(0, f32::NAN), (0, f32::INFINITY), (12, 1.0)] {
        let mut bad = vec![0u8; 64];
        bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        std::fs::write(root.join("seed.bin"), &bad).unwrap();
        assert!(replay_seed(&root, &metadata(&bad), 1).is_err());
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn newton_success_shader() {
    for source in [
        success_source(NewtonFailureMethod::FischerContinuationRelease),
        success_seed_source(NewtonFailureMethod::FischerContinuationRelease),
    ] {
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
#[ignore = "requires GPU and MOLLA_NEWTON_CAPTURE_SUCCESS=1; success/failure provenance"]
fn newton_success_capture_lifecycle() {
    assert_eq!(
        std::env::var("MOLLA_NEWTON_CAPTURE_SUCCESS").as_deref(),
        Ok("1")
    );
    let engine = Engine::new_gpu().unwrap();
    let sizes = [1, 2, 4, 4, 16, 16, 8, 8, 1, 1];
    let buffers: Vec<_> = sizes
        .iter()
        .map(|&n| storage(&engine, &vec![0u32; n]))
        .collect();
    let sources = std::array::from_fn(|i| &buffers[i]);
    let capture = NewtonFailureCapture::new_with_method(
        &engine,
        sources,
        [1, 1, 64, 64],
        NewtonFailureMethod::FischerContinuationRelease,
    )
    .unwrap();
    let set = |buffer: usize, index: u64, value: u32| {
        engine.queue().unwrap().write_buffer(
            &buffers[buffer],
            index * 4,
            bytemuck::bytes_of(&value),
        )
    };
    assert_eq!(&run(&engine, &capture)[..4], &[0; 4]);
    set(6, 1, 1);
    set(7, 2, 1);
    let first = run(&engine, &capture);
    let word =
        |bytes: &[u8], i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!(
        (
            word(&first, 0),
            word(&first, 8),
            word(&first, 16),
            word(&first, 17)
        ),
        (1, 1, 0, 1)
    );
    let root = std::env::temp_dir().join(format!(
        "newton-success-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(
        capture_failure(&engine, capture.buffer(), &root, |b| map_snapshot(
            &engine, b
        ))
        .is_err()
    );
    assert!(!root.exists());
    capture_success_with_seed(
        &engine,
        capture.buffer(),
        capture.success_seed_buffer(),
        &root,
        |b| map_snapshot(&engine, b),
    )
    .unwrap();
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("capture.json")).unwrap()).unwrap();
    assert_eq!(meta["stage"], "newton-success");
    assert_eq!(
        certificate::validate_replay(&first, &meta, true).unwrap(),
        certificate::Certificate::CURRENT
    );
    assert_eq!(meta["fault"], 0);
    assert_eq!(meta["blake3"], blake3::hash(&first).to_hex().as_str());
    set(0, 0, 99);
    set(6, 0, 16);
    assert_eq!(run(&engine, &capture), first);
    set(6, 0, 0);
    set(6, 1, 0);
    assert_eq!(run(&engine, &capture), first);
    set(6, 1, 2);
    set(7, 0, 1);
    assert_eq!(run(&engine, &capture), first);
    set(7, 0, 0);
    let second = run(&engine, &capture);
    assert_eq!(word(&second, 17), 2);
    assert_eq!(word(&second, 32), 99);
    assert!(
        capture_success_with_seed(
            &engine,
            capture.buffer(),
            capture.success_seed_buffer(),
            &root,
            |b| map_snapshot(&engine, b)
        )
        .is_err()
    );
    assert_eq!(std::fs::read(root.join("newton.bin")).unwrap(), first);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires GPU, warm-newton and MOLLA_NEWTON_CAPTURE_SUCCESS=1"]
fn newton_success_seed_latches_same_solve() {
    assert!(cfg!(feature = "warm-newton"));
    assert_eq!(
        std::env::var("MOLLA_NEWTON_CAPTURE_SUCCESS").as_deref(),
        Ok("1")
    );
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    for contacts in [0usize, 1, 65] {
        let slots = contacts.max(1);
        let sizes = [
            1,
            2,
            4 * slots,
            4 * slots,
            16 * slots,
            16 * slots,
            8,
            8,
            1,
            1,
        ];
        let buffers: Vec<_> = sizes
            .iter()
            .map(|&n| storage(&engine, &vec![0; n]))
            .collect();
        let capture = NewtonFailureCapture::new_with_method(
            &engine,
            std::array::from_fn(|i| &buffers[i]),
            [1, contacts as u32, 64, 64],
            NewtonFailureMethod::FischerContinuationRelease,
        )
        .unwrap();
        let set = |buffer: usize, index: u64, value: u32| {
            engine.queue().unwrap().write_buffer(
                &buffers[buffer],
                index * 4,
                bytemuck::bytes_of(&value),
            )
        };
        let seed = |attempt: u32| {
            let mut words = vec![0u32; 16 * slots];
            for c in 0..slots {
                words[16 * c] = (attempt as f32 + c as f32 / 128.0).to_bits();
                words[16 * c + 8] = (-1e-9f32).to_bits();
            }
            words
        };
        let attempt = |value: u32| {
            engine.queue().unwrap().write_buffer(
                &buffers[5],
                0,
                bytemuck::cast_slice(&seed(value)),
            );
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            capture.encode_initial(&mut encoder);
            encoder.clear_buffer(&buffers[5], 0, None);
            engine.queue().unwrap().submit([encoder.finish()]);
            run(&engine, &capture)
        };
        set(6, 1, 1);
        set(7, 2, 1);
        let first = attempt(1);
        let saved = capture.success_seed_buffer().unwrap();
        assert_eq!(
            read(&engine, saved),
            bytemuck::cast_slice::<u32, u8>(&seed(1))
        );
        assert_eq!(read(&engine, &buffers[5]), vec![0; 64 * slots]);
        for (buffer, index, fault) in [
            (6, 0, 16),
            (7, 0, 1),
            (8, 0, 1),
            (9, 0, 1),
            (6, 1, 0),
            (7, 2, 0),
        ] {
            set(buffer, index, fault);
            assert_eq!(attempt(2), first);
            assert_eq!(
                read(&engine, saved),
                bytemuck::cast_slice::<u32, u8>(&seed(1))
            );
            set(
                buffer,
                index,
                if (buffer, index) == (6, 1) || (buffer, index) == (7, 2) {
                    1
                } else {
                    0
                },
            );
        }
        set(0, 0, 99);
        set(6, 1, 2);
        let second = attempt(3);
        assert_ne!(first, second);
        assert_eq!(
            read(&engine, saved),
            bytemuck::cast_slice::<u32, u8>(&seed(3))
        );
        let root = std::env::temp_dir().join(format!(
            "newton-success-seed-{}-{contacts}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(
            capture_success(&engine, capture.buffer(), &root, |b| map_snapshot(
                &engine, b
            ))
            .is_err()
        );
        assert!(!root.exists());
        capture_success_with_seed(&engine, capture.buffer(), Some(saved), &root, |b| {
            map_snapshot(&engine, b)
        })
        .unwrap();
        let meta =
            serde_json::from_slice(&std::fs::read(root.join("capture.json")).unwrap()).unwrap();
        assert_eq!(
            replay_seed(&root, &meta, contacts).unwrap().unwrap(),
            bytemuck::cast_slice::<u32, u8>(&seed(3))
        );
        assert_eq!(std::fs::read(root.join("newton.bin")).unwrap(), second);
        assert_eq!(
            certificate::validate_replay(&second, &meta, true).unwrap(),
            certificate::Certificate::CURRENT
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}

use crate::soil_shared_wall_friction_gpu::capture::read;

fn map_snapshot(engine: &Engine, buffer: &wgpu::Buffer) -> Result<()> {
    let (send, receive) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(15)),
        })
        .map_err(|e| Error::Solver(e.to_string()))?;
    receive
        .recv_timeout(std::time::Duration::from_secs(15))
        .map_err(|e| Error::Solver(e.to_string()))?
        .map_err(|e| Error::Solver(e.to_string()))
}

fn storage(engine: &Engine, data: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    engine
        .device()
        .unwrap()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Newton packet fixture"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

fn run(engine: &Engine, capture: &NewtonFailureCapture) -> Vec<u8> {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    capture.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatch.finish().encoded_dispatches, 1);
    read(engine, capture.buffer())
}

#[test]
fn newton_failure_shader_and_budget() {
    assert_eq!(
        NewtonFailureCapture::required_bytes(10, 4096).unwrap(),
        1_835_632
    );
    for (n, c) in [(0, 1), (33, 1), (3, 16385)] {
        assert!(NewtonFailureCapture::required_bytes(n, c).is_err());
    }
    let source = include_str!("../shaders/soil_newton_failure.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
}

#[test]
fn newton_failure_method_tags() {
    for (tag, name) in [
        (0, "unspecified-pairs"),
        (1, "stagnation-pairs"),
        (2, "fischer-pairs"),
        (3, "fischer-continuation-pairs"),
        (4, "fischer-continuation-release-pairs"),
        (5, "fischer-condensed-pairs"),
        (6, "fischer-direct-pairs"),
        (7, "fischer-hybrid-pairs"),
        (8, "fischer-adaptive-direct-pairs"),
        (9, "fischer-adaptive-hybrid-pairs"),
        (10, "fischer-conditioned-pairs"),
        (11, "fischer-conditioned-protected-pairs"),
        (12, "fischer-conditioned-protected-low-pairs"),
        (13, "fischer-conditioned-stalled-pairs"),
        (14, "fischer-conditioned-protected-stalled-pairs"),
        (15, "fischer-conditioned-protected-stalled-256-pairs"),
        (16, "fischer-conditioned-protected-stalled-cached-256-pairs"),
        (
            17,
            "fischer-conditioned-protected-stalled-parallel-256-pairs",
        ),
        (
            18,
            "fischer-conditioned-protected-stalled-volatile-256-pairs",
        ),
        (19, "fischer-conditioned-protected-stalled-schur-256-pairs"),
        (
            20,
            "fischer-conditioned-protected-stalled-schur-norm-256-pairs",
        ),
        (
            21,
            "fischer-conditioned-protected-stalled-schur-norm-active-publication-256-pairs",
        ),
        (
            22,
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-256-pairs",
        ),
        (
            23,
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-256-pairs",
        ),
        (
            24,
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-projection-cache-256-pairs",
        ),
        (
            25,
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-256-pairs",
        ),
        (
            26,
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-256-pairs",
        ),
        (
            28,
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-sum-256-pairs",
        ),
        (
            29,
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-256-pairs",
        ),
        (
            30,
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-256-pairs",
        ),
        (
            27,
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-256-pairs",
        ),
    ] {
        let method = NewtonFailureMethod::from_tag(tag).unwrap();
        assert_eq!(method as u32, tag);
        assert_eq!(method.name(), name);
        let source = shader_source(method);
        assert!(source.contains(&format!("const METHOD_TAG:u32={tag}u;")));
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
    let method = NewtonFailureMethod::from_tag(31).unwrap();
    assert_eq!(
        method,
        NewtonFailureMethod::FischerConditionedProtectedSoaScratch256
    );
    assert!(
        method
            .name()
            .ends_with("parallel-indices-soa-scratch-256-pairs")
    );
    assert!(shader_source(method).contains("const METHOD_TAG:u32=31u;"));
    for tag in [32, 255, u32::MAX] {
        assert!(NewtonFailureMethod::from_tag(tag).is_err());
    }
}

#[test]
#[ignore = "requires GPU; bounded first-failure packet and overwrite recovery"]
fn newton_failure_gpu_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    for (n, c, method) in [
        (1, 0, NewtonFailureMethod::Unspecified),
        (3, 1, NewtonFailureMethod::StagnationPairs),
        (32, 65, NewtonFailureMethod::FischerPairs),
        (3, 1, NewtonFailureMethod::FischerContinuation),
        (3, 1, NewtonFailureMethod::FischerContinuationRelease),
        (3, 1, NewtonFailureMethod::FischerCondensed),
        (3, 1, NewtonFailureMethod::FischerDirect),
        (3, 1, NewtonFailureMethod::FischerHybrid),
        (3, 1, NewtonFailureMethod::FischerAdaptiveDirect),
        (3, 1, NewtonFailureMethod::FischerAdaptiveHybrid),
        (3, 1, NewtonFailureMethod::FischerConditioned),
        (3, 1, NewtonFailureMethod::FischerConditionedProtected),
        (3, 1, NewtonFailureMethod::FischerConditionedProtectedLow),
        (3, 1, NewtonFailureMethod::FischerConditionedStalled),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedStalled,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedWide256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedCached256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedParallel256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedVolatile256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedSchur256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedNorm256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedActivePublication256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedPublicationProducts256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedRetainedLocal256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedProjectionCache256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedSharedCore256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedStreamBlock256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedStagnantBranch256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedGroupedSum256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedGroupedEight256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedParallelIndices256,
        ),
        (
            3,
            1,
            NewtonFailureMethod::FischerConditionedProtectedSoaScratch256,
        ),
    ] {
        let slots = c.max(1);
        let sizes = [
            n * n,
            2 * n,
            4 * n * slots,
            4 * n * slots,
            16 * slots,
            16 * slots,
            8,
            8,
            1,
            1,
        ];
        let data: Vec<Vec<u32>> = sizes
            .iter()
            .enumerate()
            .map(|(b, &s)| (0..s).map(|i| (b as u32 + 1) * 10000 + i as u32).collect())
            .collect();
        let buffers: Vec<_> = data.iter().map(|v| storage(&engine, v)).collect();
        let sources: [&wgpu::Buffer; 10] = std::array::from_fn(|i| &buffers[i]);
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(NewtonFailureCapture::new(&engine, sources, [n as u32, c as u32, 0, 64]).is_err());
        assert_eq!(allocations.finish().observations, 0);
        let capture = NewtonFailureCapture::new_with_method(
            &engine,
            sources,
            [n as u32, c as u32, 64, 64],
            method,
        )
        .unwrap();
        for i in [6, 8, 9] {
            engine
                .queue()
                .unwrap()
                .write_buffer(&buffers[i], 0, bytemuck::bytes_of(&0u32));
        }
        assert_eq!(&run(&engine, &capture)[..4], &[0; 4]);
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffers[6], 0, bytemuck::bytes_of(&16u32));
        let packet = run(&engine, &capture);
        let words: Vec<u32> = packet
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(&words[..7], &[1, n as u32, c as u32, 0, 0, 64, 64]);
        assert_eq!(words[7], method as u32);
        assert_eq!(&words[8..16], &[0, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(words[16], 16);
        assert_eq!(&words[17..24], &data[6][1..]);
        assert_eq!(&words[24..32], &data[7]);
        assert_eq!(&words[32..32 + n * n], &data[0]);
        assert_eq!(&words[32 + n * n..32 + n * n + 2 * n], &data[1]);
        let mut offset = 32 + n * n + 2 * n;
        for slot in 0..slots {
            for (i, size) in [(2, 4 * n), (3, 4 * n), (4, 16), (5, 16)] {
                assert_eq!(
                    &words[offset..offset + size],
                    &data[i][slot * size..(slot + 1) * size]
                );
                offset += size;
            }
        }
        assert_eq!(offset * 4, packet.len());
        let directory = std::env::temp_dir().join(format!(
            "molla-newton-method-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        capture_failure(&engine, capture.buffer(), &directory, |b| {
            map_snapshot(&engine, b)
        })
        .unwrap();
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("capture.json")).unwrap())
                .unwrap();
        assert_eq!(metadata["method"], method.name());
        assert_eq!(metadata["certificate_version"], 1);
        assert_eq!(metadata["certificate"], "gap-independent-velocity-v1");
        assert_eq!(
            certificate::validate_replay(&packet, &metadata, true).unwrap(),
            certificate::Certificate::CURRENT
        );
        assert_eq!(metadata["blake3"], blake3::hash(&packet).to_hex().as_str());
        assert_eq!(std::fs::read(directory.join("newton.bin")).unwrap(), packet);
        let seeded_directory = directory.join("seeded");
        let mut seed = vec![0u8; 64 * c.max(1) as usize];
        seed[..4].copy_from_slice(&1.25f32.to_le_bytes());
        let seed_buffer = storage(&engine, bytemuck::cast_slice::<u8, u32>(&seed));
        capture_failure_with_seed(
            &engine,
            capture.buffer(),
            Some(&seed_buffer),
            &seeded_directory,
            |b| map_snapshot(&engine, b),
        )
        .unwrap();
        let seeded_meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(seeded_directory.join("capture.json")).unwrap())
                .unwrap();
        assert_eq!(
            replay_seed(&seeded_directory, &seeded_meta, c as usize).unwrap(),
            Some(seed)
        );
        assert_eq!(
            std::fs::read(seeded_directory.join("newton.bin")).unwrap(),
            packet
        );
        assert!(
            capture_failure(&engine, capture.buffer(), &directory, |b| map_snapshot(
                &engine, b
            ))
            .is_err()
        );
        assert_eq!(std::fs::read(directory.join("newton.bin")).unwrap(), packet);
        if c == 0 {
            engine
                .queue()
                .unwrap()
                .write_buffer(capture.buffer(), 40, &0u32.to_le_bytes());
            let legacy_directory = directory.join("legacy");
            capture_failure(&engine, capture.buffer(), &legacy_directory, |b| {
                map_snapshot(&engine, b)
            })
            .unwrap();
            let legacy_bytes = std::fs::read(legacy_directory.join("newton.bin")).unwrap();
            let legacy_meta = serde_json::from_slice(
                &std::fs::read(legacy_directory.join("capture.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(
                certificate::validate_replay(&legacy_bytes, &legacy_meta, false).unwrap(),
                certificate::Certificate::Unspecified
            );
            assert!(certificate::validate_replay(&legacy_bytes, &legacy_meta, true).is_err());
            engine
                .queue()
                .unwrap()
                .write_buffer(capture.buffer(), 40, &2u32.to_le_bytes());
            let unknown = directory.join("unknown");
            assert!(
                capture_failure(&engine, capture.buffer(), &unknown, |b| map_snapshot(
                    &engine, b
                ))
                .is_err()
            );
            assert!(!unknown.exists());
            engine
                .queue()
                .unwrap()
                .write_buffer(capture.buffer(), 40, &1u32.to_le_bytes());
        }
        std::fs::remove_dir_all(directory).unwrap();
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffers[0], 0, bytemuck::bytes_of(&99u32));
        for i in [8, 9] {
            engine
                .queue()
                .unwrap()
                .write_buffer(&buffers[i], 0, bytemuck::bytes_of(&1u32));
        }
        assert_eq!(run(&engine, &capture), packet);
        for i in [6, 8, 9] {
            engine
                .queue()
                .unwrap()
                .write_buffer(&buffers[i], 0, bytemuck::bytes_of(&0u32));
        }
        assert_eq!(&run(&engine, &capture)[..4], &[0; 4]);
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffers[6], 0, bytemuck::bytes_of(&8u32));
        let next = run(&engine, &capture);
        assert_eq!(u32::from_le_bytes(next[128..132].try_into().unwrap()), 99);
        assert_eq!(u32::from_le_bytes(next[64..68].try_into().unwrap()), 8);
    }
}
