use super::*;
use std::path::Path;

pub(super) fn capture(
    world: &mut RigidWorld,
    engine: &Engine,
    root: &Path,
    name: &str,
    handles: &crate::soil_tractor::SoilTractorHandles,
    loads: Option<&MpmSoilFrameLoads>,
    separate: bool,
) {
    let path = root.join(name);
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("soil.bin"), checkpoint(world, engine)).unwrap();
    let machine = super::super::super::motion_tests::retry_machine_words(world);
    assert_eq!(machine.len(), 9);
    for (i, bytes) in machine.iter().enumerate() {
        std::fs::write(path.join(format!("machine-{i}.bin")), bytes).unwrap();
    }
    let cpu = std::iter::once(handles.chassis)
        .chain(handles.wheels)
        .chain([handles.tine])
        .map(|body| {
            (
                world.scene.body_pose(body).unwrap(),
                world.scene.body_velocity(body).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    std::fs::write(path.join("cpu.txt"), format!("{cpu:?}")).unwrap();
    if let Some(loads) = loads {
        assert_eq!(loads.bodies.len(), 5);
        let words: Vec<u32> = loads
            .bodies
            .iter()
            .flat_map(|b| {
                b.linear
                    .into_iter()
                    .chain(b.angular)
                    .chain(b.world_origin_angular)
                    .map(f32::to_bits)
            })
            .collect();
        std::fs::write(path.join("loads.bin"), bytemuck::cast_slice(&words)).unwrap();
        std::fs::write(
            path.join("load-frame.json"),
            serde_json::to_vec(&serde_json::json!({
                "frame": loads.committed_frame, "seconds_bits": loads.simulated_seconds.to_bits(),
            }))
            .unwrap(),
        )
        .unwrap();
    }
    let runtime = world.gpu.as_ref().unwrap().soil.as_ref().unwrap();
    assert_eq!(runtime.force_separate_submissions, separate);
    if let Some(projection) = &runtime.projection {
        let [steps, submissions] = projection.submission_counts;
        assert!(steps > 0);
        assert_eq!(submissions, steps * if separate { 3 } else { 1 });
        std::fs::write(path.join("submission.json"), serde_json::to_vec(&serde_json::json!({
            "separate": separate, "steps": steps, "projected_queue_submissions": submissions,
            "scope": "projected guard/solve/publication sequence, not whole-frame queue traffic",
        })).unwrap()).unwrap();
    } else {
        assert_eq!(name, "rejected");
    }
}

#[cfg(feature = "batched-soil-projection")]
#[test]
#[ignore = "requires native f64 GPU; exact two-arm six-body Newton submissions and recovery"]
fn native_world_composed_submissions_exact() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_NATIVE_WORLD_CAPTURE").unwrap());
    std::fs::create_dir(&root).unwrap();
    let engine = native_world_engine();
    for (name, separate) in [("separate", true), ("composed", false)] {
        native_prediction_world_case(
            engine.clone(),
            root.join(name),
            NativeWorldFixture::AllContacts,
            Some(separate),
        );
    }
    let fields: Vec<_> = ["soil.bin".to_owned(), "cpu.txt".to_owned()]
        .into_iter()
        .chain((0..9).map(|i| format!("machine-{i}.bin")))
        .collect();
    for name in [
        "frame-1",
        "frame-2",
        "frame-3",
        "frame-4",
        "rejected",
        "recovered",
    ] {
        for field in &fields {
            let a = std::fs::read(root.join("separate").join(name).join(field)).unwrap();
            let b = std::fs::read(root.join("composed").join(name).join(field)).unwrap();
            assert!(
                a == b,
                "submission parity {name}/{field}: {} != {}",
                blake3::hash(&a),
                blake3::hash(&b)
            );
            if name == "rejected" {
                for arm in ["separate", "composed"] {
                    let before = std::fs::read(root.join(arm).join("frame-4").join(field)).unwrap();
                    let after = std::fs::read(root.join(arm).join("rejected").join(field)).unwrap();
                    assert!(before == after, "control rejection changed {arm}/{field}");
                }
            }
        }
        if name != "rejected" {
            for field in ["loads.bin", "load-frame.json"] {
                assert_eq!(
                    std::fs::read(root.join("separate").join(name).join(field)).unwrap(),
                    std::fs::read(root.join("composed").join(name).join(field)).unwrap(),
                    "load parity {name}/{field}"
                );
            }
        }
        println!(
            "native submission {name}: complete soil, nine machine streams, six-body CPU state and available loads exact"
        );
    }
}
