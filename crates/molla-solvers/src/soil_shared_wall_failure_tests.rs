use super::*;

pub(super) fn load_paired(
    root: &std::path::Path,
) -> std::result::Result<(Fixture, Vec<[f32; 2]>), String> {
    let fail = |e: std::io::Error| e.to_string();
    if std::fs::metadata(root.join("capture.json"))
        .map_err(fail)?
        .len()
        > 65536
    {
        return Err("metadata too large".into());
    }
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("capture.json")).map_err(fail)?)
            .map_err(|e| e.to_string())?;
    let schema = meta["schema"].as_u64().ok_or("schema")?;
    if ![2, 3].contains(&schema) || meta["rate_encoding"] != "f32x2" {
        return Err("paired schema required".into());
    }
    let n = usize::try_from(meta["dofs"].as_u64().ok_or("dofs")?).map_err(|_| "dofs overflow")?;
    let c = usize::try_from(meta["contacts"].as_u64().ok_or("contacts")?)
        .map_err(|_| "contacts overflow")?;
    if !(1..=32).contains(&n) || c > 16384 {
        return Err("capacity".into());
    }
    let read = |name: &str, size: usize| -> std::result::Result<Vec<u8>, String> {
        let path = root.join(format!("{name}.bin"));
        if std::fs::metadata(&path).map_err(fail)?.len() != size as u64 {
            return Err(format!("{name} length"));
        }
        let bytes = std::fs::read(path).map_err(fail)?;
        if schema == 3
            && (meta["files"][name]["bytes"] != size as u64
                || meta["files"][name]["blake3"] != blake3::hash(&bytes).to_hex().as_str())
        {
            return Err(format!("{name} integrity"));
        }
        Ok(bytes)
    };
    if schema == 3 {
        if meta["byte_order"] != "little" || meta["stage"] != "failed-shared-solve" {
            return Err("snapshot format".into());
        }
        let mut total = 0u64;
        for name in [
            "mass",
            "free",
            "rows",
            "contacts",
            "status",
            "rates",
            "outputs",
            "conditional",
            "lower",
            "response",
            "params",
            "history",
        ] {
            let size = meta["files"][name]["bytes"].as_u64().ok_or("file length")?;
            total = total.checked_add(size).ok_or("size overflow")?;
            if total > 32 * 1024 * 1024 {
                return Err("snapshot limit".into());
            }
            read(name, size as usize)?;
        }
        if meta["staging_bytes"] != total {
            return Err("snapshot total".into());
        }
        if meta["max_sweeps"] != 64 || meta["local_iterations"] != 64 {
            return Err("replay iteration budget mismatch".into());
        }
        if meta["files"]["status"]["bytes"].as_u64().unwrap_or(0) < 432
            || meta["files"]["history"]["bytes"].as_u64().unwrap_or(0) < 64 * c.max(1) as u64
        {
            return Err("snapshot state length".into());
        }
        for (name, size) in [
            ("rates", 8 * n),
            ("outputs", 80 * c.max(1)),
            ("conditional", 112 * c.max(1)),
            ("lower", 4 * n * n),
            ("response", 16 * c.max(1) * n),
            ("params", 16),
        ] {
            read(name, size)?;
        }
    }
    let pairs: Vec<[f32; 2]> = read("free", 8 * n)?
        .chunks_exact(8)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let f = Fixture {
        mass: read("mass", 4 * n * n)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        free: pairs.iter().map(|p| p[0]).collect(),
        rows: read("rows", 16 * c.max(1) * n)?
            .chunks_exact(16)
            .take(c * n)
            .map(bytemuck::pod_read_unaligned)
            .collect(),
        contacts: read("contacts", 64 * c.max(1))?
            .chunks_exact(64)
            .take(c)
            .map(bytemuck::pod_read_unaligned)
            .collect(),
    };
    Ok((f, pairs))
}

fn map_snapshot(engine: &Engine, buffer: &wgpu::Buffer) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .map_err(|e| Error::Solver(e.to_string()))?;
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| Error::Solver(e.to_string()))?
        .map_err(|e| Error::Solver(e.to_string()))
}

#[test]
#[ignore = "requires GPU; bounded lossless paired failure snapshot"]
fn paired_failure_snapshot_roundtrip() {
    let engine = Engine::new_gpu().unwrap();
    let f = pair(true, 0.3);
    let pairs: Vec<_> = f
        .free
        .iter()
        .enumerate()
        .map(|(i, q)| [*q, (i + 1) as f32 * 2f32.powi(-30)])
        .collect();
    let free = storage(&engine, &pairs);
    let solver = create(&engine, &f, &free);
    engine
        .queue()
        .unwrap()
        .write_buffer(solver.status_buffer(), 0, bytemuck::bytes_of(&16u32));
    let root = std::env::temp_dir().join(format!(
        "molla-shared-failure-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    solver
        .inner
        .capture_failure(&engine, &root, |b| map_snapshot(&engine, b))
        .unwrap();
    let (loaded, loaded_pairs) = load_paired(&root).unwrap();
    assert_eq!(loaded_pairs, pairs);
    assert_eq!(loaded.mass, f.mass);
    assert_eq!(loaded.rows, f.rows);
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&loaded.contacts),
        bytemuck::cast_slice::<_, u8>(&f.contacts)
    );
    assert!(
        solver
            .inner
            .capture_failure(&engine, &root, |_| panic!(
                "must refuse overwrite before mapping"
            ))
            .is_err()
    );
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("capture.json")).unwrap()).unwrap();
    assert_eq!(meta["schema"], 3);
    let mut data = std::fs::read(root.join("free.bin")).unwrap();
    data[4] ^= 1;
    std::fs::write(root.join("free.bin"), &data).unwrap();
    assert!(
        load_paired(&root)
            .err()
            .expect("corrupt snapshot rejected")
            .contains("free integrity")
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires native GPU and MOLLA_SHARED_FAILURE_REPLAY input"]
fn captured_shared_failure_replay() {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_SHARED_FAILURE_REPLAY").expect("snapshot path"),
    );
    let (f, pairs) = load_paired(&root).unwrap();
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let free = storage(&engine, &pairs);
    let solver = create(&engine, &f, &free);
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    let bytes = capture::read(&engine, solver.status_buffer());
    let status: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let rates: Vec<f64> = if status[0] == 0 {
        combined(&read_pairs(&engine, solver.rates_buffer(), pairs.len()))
    } else {
        (0..pairs.len())
            .map(|i| {
                f32::from_bits(status[8 + 2 * i]) as f64 + f32::from_bits(status[9 + 2 * i]) as f64
            })
            .collect()
    };
    let outputs: Vec<SoilWallFrictionOutput> = capture::read(&engine, solver.output_buffer())
        .chunks_exact(80)
        .take(f.contacts.len())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let mut worst = (0usize, 0f64);
    let mut worst_slip = (0usize, 0f64, 0f64);
    for (c, contact) in f.contacts.iter().enumerate() {
        if contact.soil_mass[3] == 0.0 {
            continue;
        }
        let normal = vec(contact.normal_gap);
        let soil = normal.dot(vec(outputs[c].soil_velocity_energy));
        let machine: f64 = rates
            .iter()
            .enumerate()
            .map(|(i, q)| normal.dot(vec(f.rows[c * rates.len() + i])) * q)
            .sum();
        let gap = contact.normal_gap[3] as f64;
        let residual = (soil - machine + gap) / 1f64.max(soil.abs()).max(machine.abs()).max(gap);
        if residual < worst.1 {
            worst = (c, residual);
        }
        let machine_velocity: DVec3 = rates
            .iter()
            .enumerate()
            .map(|(i, q)| vec(f.rows[c * rates.len() + i]) * q)
            .sum();
        let soil_velocity = vec(outputs[c].soil_velocity_energy);
        let relative = soil_velocity - machine_velocity;
        let unit = normal.normalize();
        let slip = relative - unit * unit.dot(relative);
        let k = -vec(outputs[c].machine_impulse_normal);
        let w = vec(outputs[c].wall_impulse);
        let lambda = unit.dot(k);
        let tangent = k - unit * lambda;
        let vs = 1.0 + soil_velocity.length() + machine_velocity.length() + gap;
        let ps = 1.0 + k.length() + w.length();
        if slip.length() > 1e-6 * vs {
            let defect =
                (tangent + contact.friction[0] as f64 * lambda * slip.normalize()).length() / ps;
            if defect > worst_slip.1 {
                worst_slip = (c, defect, slip.length() / vs);
            }
        }
    }
    println!(
        "shared replay status={:?} worst_normal={worst:?} worst_slip={worst_slip:?}",
        &status[..8]
    );
    if std::env::var_os("MOLLA_SHARED_REPLAY_BASELINE").is_none() {
        assert_eq!(status[0], 0, "captured coupled solve must now converge");
        verify_precise_free_tolerance(&f, &combined(&pairs), &rates, &outputs, 1e-6);
        assert!(worst.1 >= -1e-6);
        assert!(worst_slip.1 <= 1e-6);
        return;
    }
    let captured = std::fs::read(root.join("status.bin")).unwrap();
    let captured_words: Vec<u32> = captured
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!(
        "captured status={:?} final_rate_words_equal={}",
        &captured_words[..8],
        captured_words[8..8 + 2 * pairs.len()] == status[8..8 + 2 * pairs.len()]
    );
    assert_eq!(
        status[0] & 16,
        16,
        "diagnostic must reproduce shared sweep failure"
    );
    assert_eq!(
        status[7], captured_words[7],
        "first failed certificate must reproduce"
    );
    assert_eq!(
        &status[8..8 + 2 * pairs.len()],
        &captured_words[8..8 + 2 * pairs.len()],
        "final paired rates must reproduce bit for bit"
    );
    assert_eq!(&status[1..8], &captured_words[1..8]);
    assert_eq!(
        capture::read(&engine, solver.output_buffer()),
        std::fs::read(root.join("outputs.bin")).unwrap(),
        "contact outputs must reproduce bit for bit"
    );
}

#[test]
#[ignore = "requires native GPU and MOLLA_SHARED_FAILURE_REPLAY input"]
fn captured_shared_local_accuracy() {
    use crate::soil_wall_friction_gpu::{ExperimentalSoilWallFrictionGpu, SoilWallFrictionInput};
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_SHARED_FAILURE_REPLAY").expect("snapshot path"),
    );
    load_paired(&root).unwrap();
    let bytes = std::fs::read(root.join("conditional.bin")).unwrap();
    let index: usize = std::env::var("MOLLA_SHARED_LOCAL_CONTACT")
        .map(|s| s.parse().unwrap())
        .unwrap_or(100);
    let input: SoilWallFrictionInput =
        bytemuck::pod_read_unaligned(&bytes[index * 112..(index + 1) * 112]);
    let engine = Engine::new_gpu().unwrap();
    let buffer = storage(&engine, &[input]);
    let solver = ExperimentalSoilWallFrictionGpu::with_source(
        &engine,
        &buffer,
        1,
        64,
        u64::MAX,
        &shared_friction_accuracy_source(&ExperimentalSoilWallFrictionGpu::shader_source()).replace(
            "separation>=-1e-5*velocity_scale",
            "separation>=-1e-6*max(1.0,max(abs(dot(axes[0],soil)),max(abs(dot(axes[0],machine)),input.normal_gap.w)))"),
    )
    .unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    let output: SoilWallFrictionOutput =
        bytemuck::pod_read_unaligned(&capture::read(&engine, solver.output_buffer())[..80]);
    let k = -vec(output.machine_impulse_normal);
    let w = vec(output.wall_impulse);
    let a = glam::DMat3::from_cols(
        vec(input.inverse_machine[0]),
        vec(input.inverse_machine[1]),
        vec(input.inverse_machine[2]),
    );
    let machine = vec(input.machine_velocity_friction) - a * k;
    let relative = vec(output.soil_velocity_energy) - machine;
    let normal = vec(input.normal_gap).normalize();
    let slip = relative - normal * normal.dot(relative);
    let lambda = k.dot(normal);
    let tangent = k - normal * lambda;
    let defect = (tangent + input.machine_velocity_friction[3] as f64 * lambda * slip.normalize())
        .length()
        / (1.0 + k.length() + w.length());
    let soil_normal = vec(input.normal_gap).dot(vec(output.soil_velocity_energy));
    let machine_normal = vec(input.normal_gap).dot(machine);
    let gap = input.normal_gap[3] as f64;
    let normal_residual = (soil_normal - machine_normal + gap)
        / 1f64
            .max(soil_normal.abs())
            .max(machine_normal.abs())
            .max(gap);
    println!(
        "conditional contact {index} status={:?} normal_residual={normal_residual} slip_defect={defect} slip_speed={} input={input:?} output={output:?}",
        output.status,
        slip.length()
    );
    assert_eq!(output.status[0], 0);
    assert!(
        defect <= 1e-6,
        "local publication must meet shared friction tolerance"
    );
}

#[test]
#[ignore = "requires MOLLA_SHARED_FAILURE_REPLAY input"]
fn captured_conditional_residuals() {
    use crate::soil_wall_friction_gpu::SoilWallFrictionInput;
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_SHARED_FAILURE_REPLAY").expect("snapshot path"),
    );
    let (f, _) = load_paired(&root).unwrap();
    let inputs = std::fs::read(root.join("conditional.bin")).unwrap();
    let outputs = std::fs::read(root.join("outputs.bin")).unwrap();
    let mut worst = [(0usize, 0f64); 4];
    for c in 0..f.contacts.len() {
        if f.contacts[c].soil_mass[3] == 0.0 {
            continue;
        }
        let input: SoilWallFrictionInput =
            bytemuck::pod_read_unaligned(&inputs[c * 112..(c + 1) * 112]);
        let output: SoilWallFrictionOutput =
            bytemuck::pod_read_unaligned(&outputs[c * 80..(c + 1) * 80]);
        let k = -vec(output.machine_impulse_normal);
        let w = vec(output.wall_impulse);
        let a = glam::DMat3::from_cols(
            vec(input.inverse_machine[0]),
            vec(input.inverse_machine[1]),
            vec(input.inverse_machine[2]),
        );
        let machine = vec(input.machine_velocity_friction) - a * k;
        let soil = vec(output.soil_velocity_energy);
        let n = vec(input.normal_gap).normalize();
        let separation = n.dot(soil - machine) + input.normal_gap[3] as f64;
        let ps = 1.0 + k.length() + w.length();
        let vs = 1.0 + soil.length() + machine.length() + input.normal_gap[3] as f64;
        let normal_scale = 1f64
            .max(n.dot(soil).abs())
            .max(n.dot(machine).abs())
            .max(input.normal_gap[3] as f64);
        let lambda = n.dot(k);
        let tangent = k - n * lambda;
        let slip = soil - machine - n * n.dot(soil - machine);
        let defect = if slip.length() > 1e-6 * vs {
            (tangent + input.machine_velocity_friction[3] as f64 * lambda * slip.normalize())
                .length()
                / ps
        } else {
            0.0
        };
        let errors = [
            -separation / normal_scale,
            (lambda * separation).abs() / (ps * vs),
            defect,
            separation.abs() / normal_scale,
        ];
        for i in 0..4 {
            if errors[i] > worst[i].1 {
                worst[i] = (c, errors[i]);
            }
        }
    }
    println!(
        "conditional worst [normal infeasibility, complementarity, slip, absolute separation]={worst:?}"
    );
}

#[test]
#[ignore = "requires native GPU and MOLLA_SHARED_FAILURE_REPLAY input"]
fn captured_shared_sweep_trace() {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_SHARED_FAILURE_REPLAY").expect("snapshot path"),
    );
    let (f, pairs) = load_paired(&root).unwrap();
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let free = storage(&engine, &pairs);
    let solver = create(&engine, &f, &free);
    let mut previous = combined(&pairs);
    println!(
        "sweep,fault,certificate,accelerated,rejected,worst_contact,worst_normal,max_rate_change,normal96_rate_change"
    );
    for (pipeline, groups) in solver.inner.passes() {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &solver.inner.group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        let bytes = capture::read(&engine, solver.status_buffer());
        let status: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if status[1] == 0 {
            continue;
        }
        let rates: Vec<f64> = if status[104] != 0 {
            combined(&read_pairs(&engine, solver.rates_buffer(), pairs.len()))
        } else {
            (0..pairs.len())
                .map(|i| {
                    f32::from_bits(status[8 + 2 * i]) as f64
                        + f32::from_bits(status[9 + 2 * i]) as f64
                })
                .collect()
        };
        let outputs: Vec<SoilWallFrictionOutput> = capture::read(&engine, solver.output_buffer())
            .chunks_exact(80)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let mut worst = (0usize, 0f64);
        for (c, contact) in f.contacts.iter().enumerate() {
            if contact.soil_mass[3] == 0.0 {
                continue;
            }
            let normal = vec(contact.normal_gap);
            let soil = normal.dot(vec(outputs[c].soil_velocity_energy));
            let machine: f64 = rates
                .iter()
                .enumerate()
                .map(|(i, q)| normal.dot(vec(f.rows[c * rates.len() + i])) * q)
                .sum();
            let gap = contact.normal_gap[3] as f64;
            let residual =
                (soil - machine + gap) / 1f64.max(soil.abs()).max(machine.abs()).max(gap);
            if residual < worst.1 {
                worst = (c, residual);
            }
        }
        let change = rates
            .iter()
            .zip(&previous)
            .map(|(q, p)| (q - p).abs())
            .fold(0f64, f64::max);
        let normal = vec(f.contacts[96].normal_gap);
        let normal_change: f64 = rates
            .iter()
            .zip(&previous)
            .enumerate()
            .map(|(i, (q, p))| normal.dot(vec(f.rows[96 * rates.len() + i])) * (q - p))
            .sum();
        println!(
            "{},{},{},{},{},{},{:.12e},{:.12e},{:.12e}",
            status[1],
            status[0],
            status[7],
            status[4],
            status[5],
            worst.0,
            worst.1,
            change,
            normal_change
        );
        previous = rates;
        if status[0] != 0 || status[104] != 0 {
            break;
        }
    }
    assert_eq!(
        capture::read(&engine, solver.output_buffer()),
        std::fs::read(root.join("outputs.bin")).unwrap()
    );
}

#[test]
fn rate_only_acceleration_misses_balanced_soil_mode() {
    let previous = [-1.0_f64, 1.0];
    let current = [-0.5_f64, 0.5];
    let projected_previous: f64 = previous.iter().sum();
    let projected_current: f64 = current.iter().sum();
    assert_eq!((projected_previous, projected_current), (0.0, 0.0));
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (f, old) in current.iter().zip(previous) {
        let difference = f - old;
        numerator += difference * f;
        denominator += difference * difference;
    }
    let coefficient = -numerator / denominator;
    assert_eq!(coefficient, 1.0);
    let previous_image = [1.0, -1.0];
    let current_image = [0.5, -0.5];
    let trial: Vec<_> = current_image
        .iter()
        .zip(previous_image)
        .map(|(g, old)| g + coefficient * (g - old))
        .collect();
    assert_eq!(trial, [0.0, 0.0]);
    assert!(current_image.iter().map(|x| x * x).sum::<f64>() > 0.0);
}
