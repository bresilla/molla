use super::*;

fn packet(tag: u32) -> Vec<u8> {
    let mut words = vec![0u32; 75];
    words[..11].copy_from_slice(&[1, 1, 1, 0, 0, 64, 64, 31, 1, 0, tag]);
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn meta(bytes: &[u8], certificate: Option<Certificate>) -> serde_json::Value {
    let mut meta = serde_json::json!({"blake3":blake3::hash(bytes).to_hex().as_str()});
    if let Some(c) = certificate {
        meta["certificate_version"] = c.tag().into();
        meta["certificate"] = c.name().into();
    }
    meta
}

#[test]
fn newton_certificate_legacy_current_and_mismatched_contracts() {
    let legacy = packet(0);
    let current = packet(1);
    for declaration in [None, Some(Certificate::Unspecified)] {
        let metadata = meta(&legacy, declaration);
        assert_eq!(
            validate_replay(&legacy, &metadata, false).unwrap(),
            Certificate::Unspecified
        );
        assert!(
            validate_replay(&legacy, &metadata, true)
                .unwrap_err()
                .to_string()
                .contains("capture-time")
        );
    }
    let good = meta(&current, Some(Certificate::CURRENT));
    for exact in [false, true] {
        assert_eq!(
            validate_replay(&current, &good, exact).unwrap(),
            Certificate::CURRENT
        );
        assert!(validate_replay(&legacy, &good, exact).is_err());
        assert!(validate_replay(&current, &meta(&current, None), exact).is_err());
        for (key, value) in [
            ("certificate_version", serde_json::json!(0)),
            ("certificate_version", serde_json::json!(2)),
            ("certificate_version", serde_json::json!(-1)),
            ("certificate_version", serde_json::json!(1.5)),
            ("certificate_version", serde_json::json!("1")),
            ("certificate", serde_json::json!("unspecified")),
            ("certificate", serde_json::Value::Null),
        ] {
            let mut bad = good.clone();
            bad[key] = value;
            assert!(validate_replay(&current, &bad, exact).is_err());
        }
        for key in ["certificate", "certificate_version"] {
            let mut bad = good.clone();
            bad.as_object_mut().unwrap().remove(key);
            assert!(validate_replay(&current, &bad, exact).is_err());
        }
    }
    for tag in [2, u32::MAX] {
        assert!(Certificate::from_packet(&packet(tag)).is_err());
    }
    for len in 0..64 {
        assert!(Certificate::from_packet(&current[..len]).is_err());
    }
    for index in 11..16 {
        let mut bad = current.clone();
        bad[4 * index] = 1;
        assert!(Certificate::from_packet(&bad).is_err());
    }
    assert!(validate_replay(&current[..64], &good, true).is_err());
    for (index, value) in [(1, 0u32), (1, 33), (2, 16385), (5, 0), (6, 65)] {
        let mut bad = current.clone();
        bad[4 * index..4 * index + 4].copy_from_slice(&value.to_le_bytes());
        assert!(validate_replay(&bad, &good, true).is_err());
    }
}

#[test]
fn newton_certificate_bounded_disk_preflight() {
    let root = std::env::temp_dir().join(format!(
        "newton-certificate-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    assert!(preflight(&root, true).is_err());
    let bytes = packet(1);
    let metadata = meta(&bytes, Some(Certificate::CURRENT));
    std::fs::write(root.join("newton.bin"), &bytes).unwrap();
    std::fs::write(
        root.join("capture.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    assert_eq!(preflight(&root, true).unwrap(), Certificate::CURRENT);
    let mut corrupt = bytes.clone();
    corrupt[70] ^= 1;
    std::fs::write(root.join("newton.bin"), &corrupt).unwrap();
    assert!(
        preflight(&root, true)
            .unwrap_err()
            .to_string()
            .contains("digest")
    );
    let legacy = packet(0);
    std::fs::write(root.join("newton.bin"), &legacy).unwrap();
    std::fs::write(
        root.join("capture.json"),
        serde_json::to_vec(&meta(&legacy, None)).unwrap(),
    )
    .unwrap();
    assert_eq!(preflight(&root, false).unwrap(), Certificate::Unspecified);
    assert!(
        preflight(&root, true)
            .unwrap_err()
            .to_string()
            .contains("capture-time")
    );
    std::fs::write(root.join("capture.json"), vec![b' '; 65537]).unwrap();
    assert!(
        preflight(&root, false)
            .unwrap_err()
            .to_string()
            .contains("bound")
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn newton_certificate_capture_shader_and_metric_contract() {
    let source = include_str!("../shaders/soil_newton_failure.wgsl");
    assert_eq!(
        source
            .matches(&format!(
                "const CERTIFICATE_TAG:u32={}u;",
                Certificate::CURRENT.tag()
            ))
            .count(),
        1
    );
    assert_eq!(source.matches("packet[10]=CERTIFICATE_TAG;").count(), 1);
    for generated in [
        shader_source(NewtonFailureMethod::Unspecified),
        success_source(NewtonFailureMethod::Unspecified),
        success_seed_source(NewtonFailureMethod::Unspecified),
    ] {
        assert_eq!(generated.matches("packet[10]=CERTIFICATE_TAG;").count(), 1);
    }
    let metric = include_str!("../shaders/soil_newton.wgsl");
    assert!(metric.contains("let vs=1.0+length(a.soil)+length(machine);"));
    assert!(!metric.contains("let vs=1.0+length(a.soil)+length(machine)+a.gap;"));
}
