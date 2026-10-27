use super::*;
use std::io::Read;
use std::path::Path;

#[test]
fn newton_preparation_capture_contract() {
    let f = pair(false, 0.0);
    let n = f.free.len();
    let c = f.contacts.len();
    let mut header = [0u32; 32];
    for (i, value) in [
        (0, 1),
        (1, n as u32),
        (2, c as u32),
        (5, 64),
        (6, 64),
        (8, 1),
        (26, c as u32),
    ] {
        header[i] = value;
    }
    let mut bytes = bytemuck::cast_slice(&header).to_vec();
    bytes.extend_from_slice(bytemuck::cast_slice(&f.mass));
    for q in &f.free {
        bytes.extend_from_slice(bytemuck::bytes_of(&[*q, 0.0]));
    }
    for slot in 0..c {
        bytes.extend_from_slice(bytemuck::cast_slice(&f.rows[slot * n..(slot + 1) * n]));
        bytes.resize(bytes.len() + 16 * n, 0);
        bytes.extend_from_slice(bytemuck::bytes_of(&f.contacts[slot]));
        bytes.resize(bytes.len() + 64, 0);
    }
    let metadata = |bytes: &[u8]| {
        serde_json::json!({
            "schema":1, "stage":"newton-success", "byte_order":"little",
            "bytes":bytes.len(), "blake3":blake3::hash(bytes).to_hex().as_str(),
            "dofs":n, "contacts":c, "newton_iterations":64, "krylov_iterations":64,
            "diagnostic_inner_limit":0, "fault":0,
        })
    };
    let meta = metadata(&bytes);
    let decoded = packet_fixture(&bytes, &meta);
    assert_eq!(decoded.mass, f.mass);
    assert_eq!(decoded.free, f.free);
    assert_eq!(decoded.rows, f.rows);
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&decoded.contacts),
        bytemuck::cast_slice::<_, u8>(&f.contacts)
    );
    for (index, value) in [(1, 33), (5, 63), (16, 1), (26, 0)] {
        let mut bad = bytes.clone();
        bad[4 * index..4 * index + 4].copy_from_slice(&u32::to_le_bytes(value));
        assert!(std::panic::catch_unwind(|| packet_fixture(&bad, &metadata(&bad))).is_err());
    }
    let mut corrupt = bytes.clone();
    corrupt[128] ^= 1;
    assert!(std::panic::catch_unwind(|| packet_fixture(&corrupt, &meta)).is_err());
    let truncated = &bytes[..bytes.len() - 4];
    assert!(std::panic::catch_unwind(|| packet_fixture(truncated, &metadata(truncated))).is_err());
}

fn bounded_read(path: &Path, limit: usize) -> Vec<u8> {
    let file = std::fs::File::open(path).unwrap();
    assert!(file.metadata().unwrap().len() <= limit as u64);
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes).unwrap();
    assert!(bytes.len() <= limit);
    bytes
}

fn packet_fixture(bytes: &[u8], meta: &serde_json::Value) -> Fixture {
    packet_fixture_with_fault(bytes, meta, false)
}

fn packet_fixture_with_fault(bytes: &[u8], meta: &serde_json::Value, failed: bool) -> Fixture {
    assert!(bytes.len() >= 128);
    assert_eq!(meta["schema"], 1);
    assert_eq!(
        meta["stage"],
        if failed {
            "newton-solve"
        } else {
            "newton-success"
        }
    );
    assert_eq!(meta["byte_order"], "little");
    assert_eq!(meta["bytes"], bytes.len());
    assert_eq!(meta["blake3"], blake3::hash(bytes).to_hex().as_str());
    let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    let n = word(1) as usize;
    let c = word(2) as usize;
    assert_eq!(word(0), 1);
    let fault = u32::try_from(meta["fault"].as_u64().unwrap()).unwrap();
    assert_eq!(failed, fault != 0);
    assert_eq!(
        (word(3), word(4), word(8), word(16)),
        (0, 0, u32::from(!failed), fault)
    );
    assert!((1..=32).contains(&n) && (1..=16384).contains(&c));
    assert_eq!((word(5), word(6)), (64, 64));
    assert_eq!(meta["dofs"], n);
    assert_eq!(meta["contacts"], c);
    assert_eq!(meta["newton_iterations"], 64);
    assert_eq!(meta["krylov_iterations"], 64);
    assert_eq!(meta["diagnostic_inner_limit"], 0);
    assert_eq!(bytes.len(), 128 + 4 * n * n + 8 * n + (32 * n + 128) * c);
    let mass_end = 128 + 4 * n * n;
    let mut f = Fixture {
        mass: bytes[128..mass_end]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        free: bytes[mass_end..mass_end + 8 * n]
            .chunks_exact(8)
            .map(|b| f32::from_le_bytes(b[..4].try_into().unwrap()))
            .collect(),
        rows: Vec::with_capacity(c * n),
        contacts: Vec::with_capacity(c),
    };
    for block in bytes[mass_end + 8 * n..].chunks_exact(32 * n + 128) {
        f.rows.extend(
            block[..16 * n]
                .chunks_exact(16)
                .map(bytemuck::pod_read_unaligned::<[f32; 4]>),
        );
        f.contacts
            .push(bytemuck::pod_read_unaligned(&block[32 * n..32 * n + 64]));
    }
    assert_eq!(
        f.contacts.iter().filter(|c| c.soil_mass[3] > 0.0).count(),
        word(26) as usize
    );
    f
}

#[test]
#[ignore = "requires GPU and a failed Newton packet; preparation and archived-response parity"]
fn newton_preparation_fault_packet_axes() {
    let root =
        std::path::PathBuf::from(std::env::var_os("MOLLA_PREPARATION_FAILURE_REPLAY").unwrap());
    let bytes = bounded_read(&root.join("newton.bin"), 20 * 1024 * 1024);
    let meta = serde_json::from_slice(&bounded_read(&root.join("capture.json"), 65536)).unwrap();
    let fixture = packet_fixture_with_fault(&bytes, &meta, true);
    let n = fixture.free.len();
    let mut expected = Vec::new();
    for block in bytes[128 + 4 * n * n + 8 * n..].chunks_exact(32 * n + 128) {
        expected.extend_from_slice(&block[16 * n..32 * n]);
    }
    preparation_fixtures(8, vec![fixture]);
    let capture =
        std::path::PathBuf::from(std::env::var_os("MOLLA_NATIVE_PREPARE_CAPTURE").unwrap());
    std::fs::write(capture.join("archived-response.bin"), &expected).unwrap();
    for arm in ["reference", "candidate"] {
        for pass in 0..2 {
            let actual =
                std::fs::read(capture.join(format!("fixture-0-pass-{pass}-{arm}-response.bin")))
                    .unwrap();
            assert_eq!(
                actual, expected,
                "{arm} response differs from the failing solve input"
            );
        }
    }
    eprintln!(
        "failed Newton input: serial and axis-parallel responses match all {} archived bytes exactly; downstream solve not rerun",
        expected.len()
    );
}

#[test]
#[ignore = "requires GPU and successful Newton packets; preparation exact oracle and paired cost"]
fn newton_preparation_captured_certificates() {
    captured_preparation(7);
}

#[test]
#[ignore = "requires GPU and successful Newton packets; axis-parallel preparation"]
fn newton_preparation_captured_axes() {
    captured_preparation(8);
}

#[test]
#[ignore = "requires GPU; archived preparation shared-axis response comparison"]
fn newton_preparation_captured_axis_shared() {
    captured_preparation(9);
}

#[test]
#[ignore = "requires GPU; archived axis-parallel phase attribution with matched controls"]
fn newton_preparation_captured_phases() {
    captured_preparation(10);
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires GPU; archived atomic preparation compiler-route control"]
fn newton_preparation_captured_atomic() {
    captured_preparation(11);
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires GPU; archived volatile preparation rounding"]
fn newton_preparation_captured_volatile() {
    captured_preparation(12);
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires GPU; archived strict volatile preparation rounding"]
fn newton_preparation_captured_strict() {
    captured_preparation(13);
}

#[cfg(all(
    feature = "strict-rounding-newton-preparation",
    not(target_arch = "wasm32")
))]
#[test]
#[ignore = "requires GPU; archived runtime strict rounding versus ordinary axis"]
fn newton_preparation_captured_runtime_rounding() {
    captured_preparation(14);
}

#[test]
#[ignore = "requires GPU and retained packets; cooperative-row preparation"]
fn newton_preparation_captured_rows() {
    captured_preparation(15);
}

#[test]
#[ignore = "requires GPU and retained packets; cooperative Cholesky factors"]
fn newton_preparation_captured_row_factor() {
    captured_preparation(17);
}

#[test]
#[ignore = "requires GPU and retained packets; width-sized response workgroups"]
fn newton_preparation_captured_row_sized() {
    captured_preparation(19);
}

#[cfg(all(
    feature = "row-parallel-newton-preparation",
    not(feature = "strict-rounding-newton-preparation")
))]
#[test]
#[ignore = "requires GPU and retained packets; public cooperative-factor selection"]
fn newton_preparation_captured_runtime_row_factor() {
    captured_preparation(18);
}

#[cfg(all(
    feature = "row-parallel-newton-preparation",
    not(feature = "strict-rounding-newton-preparation")
))]
#[test]
#[ignore = "requires GPU and retained packets; runtime cooperative-row selection"]
fn newton_preparation_captured_runtime_rows() {
    captured_preparation(16);
}

fn captured_preparation(variant: u32) {
    let paths = std::env::var_os("MOLLA_PREPARATION_REPLAY_SET").unwrap();
    let paths: Vec<_> = std::env::split_paths(&paths).collect();
    assert!((1..=8).contains(&paths.len()));
    let fixtures = paths
        .iter()
        .enumerate()
        .map(|(index, root)| {
            let bytes = bounded_read(&root.join("newton.bin"), 20 * 1024 * 1024);
            let meta =
                serde_json::from_slice(&bounded_read(&root.join("capture.json"), 65536)).unwrap();
            let f = packet_fixture(&bytes, &meta);
            eprintln!(
                "preparation input fixture={index} path={} blake3={}",
                root.display(),
                blake3::hash(&bytes)
            );
            f
        })
        .collect();
    preparation_fixtures(variant, fixtures);
}
