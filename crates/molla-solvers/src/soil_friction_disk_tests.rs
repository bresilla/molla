use super::*;

fn row(w: [f32; 3], v: [f32; 2], cap: f32) -> SoilFrictionDiskInput {
    SoilFrictionDiskInput {
        inverse_mass: w,
        max_impulse: cap,
        relative_velocity: v,
        padding: [0.0; 2],
    }
}

fn input(engine: &Engine, rows: &[SoilFrictionDiskInput]) -> wgpu::Buffer {
    let buffer = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("friction fixture input"),
            size: 32 * rows.len().max(1) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    if !rows.is_empty() {
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(rows));
    }
    buffer
}

fn run(engine: &Engine, solver: &ExperimentalSoilFrictionDiskGpu) -> Vec<SoilFrictionDiskOutput> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("friction fixture readback"),
        size: solver.output.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    assert_eq!(
        dispatches.finish().encoded_dispatches,
        u64::from(solver.count != 0)
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&solver.output, 0, &staging, 0, staging.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let values =
        bytemuck::cast_slice::<u8, SoilFrictionDiskOutput>(&staging.slice(..).get_mapped_range())
            .to_vec();
    staging.unmap();
    values.into_iter().take(solver.count as usize).collect()
}

fn verify(input: SoilFrictionDiskInput, output: SoilFrictionDiskOutput) {
    assert_eq!(output.fault, 0, "input={input:?}, output={output:?}");
    let w = input.inverse_mass.map(f64::from);
    let v = input.relative_velocity.map(f64::from);
    let cap = f64::from(input.max_impulse);
    let p = output.impulse.map(f64::from);
    let scale = w[0].max(w[2]).max(v[0].abs()).max(v[1].abs());
    let (reference, _) =
        crate::contact_friction::impulse(w.map(|x| x / scale), v.map(|x| x / scale), cap, cap);
    for axis in 0..2 {
        assert!(
            (p[axis] - reference[axis]).abs() <= 5e-6 * cap.max(1e-6),
            "{input:?}: {output:?}, reference={reference:?}"
        );
    }
    assert!(p[0].hypot(p[1]) <= cap * (1.0 + 2e-6));
    let wp = [w[0] * p[0] + w[1] * p[1], w[1] * p[0] + w[2] * p[1]];
    let after = [v[0] + wp[0], v[1] + wp[1]];
    let energy = p[0] * (v[0] + 0.5 * wp[0]) + p[1] * (v[1] + 0.5 * wp[1]);
    let energy_scale = cap * (v[0].hypot(v[1]) + cap * (w[0] + w[2]));
    assert!(energy <= 2e-6 * energy_scale);
    assert!((f64::from(output.energy_change) - energy).abs() <= 2e-6 * energy_scale.max(1e-30));
    for axis in 0..2 {
        assert!(
            (f64::from(output.relative_velocity[axis]) - after[axis]).abs()
                <= 2e-6 * scale * (1.0 + cap)
        );
    }
    if cap > 0.0 {
        let lambda = if output.mode == 2 {
            -(p[0] * after[0] + p[1] * after[1]) / (p[0] * p[0] + p[1] * p[1])
        } else {
            0.0
        };
        assert!(lambda >= -1e-6 * scale);
        assert!(
            (after[0] + lambda * p[0]).hypot(after[1] + lambda * p[1])
                <= 5e-6 * scale * (1.0 + cap)
        );
        if output.mode == 2 {
            assert!((p[0].hypot(p[1]) / cap - 1.0).abs() <= 2e-6);
        }
    }
}

#[test]
fn friction_disk_layout() {
    assert_eq!(std::mem::size_of::<SoilFrictionDiskInput>(), 32);
    assert_eq!(std::mem::size_of::<SoilFrictionDiskOutput>(), 48);
    assert_eq!(
        std::mem::offset_of!(SoilFrictionDiskInput, relative_velocity),
        16
    );
    assert_eq!(std::mem::offset_of!(SoilFrictionDiskOutput, fault), 32);
}

#[test]
#[ignore = "requires GPU"]
fn friction_disk_reference_kkt_scaling_and_rotation() {
    let engine = Engine::new_gpu().unwrap();
    let mut rows = vec![
        row([2.0, 0.0, 2.0], [3.0, 4.0], 0.0),
        row([2.0, 0.5, 3.0], [0.0; 2], 2.0),
        row([2.0, 0.0, 2.0], [0.3, -0.4], 1.0),
        row([2.0, 0.0, 2.0], [3.0, 4.0], 1.0),
        row([2.0, 0.0, 2.0], [2.0, 0.0], 1.0),
        row([1.0, 0.5, 5.0], [3.0, -4.0], 0.7),
    ];
    let mut seed = 1827u32;
    let mut sample = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16777216.0
    };
    for _ in 0..129 {
        let angle = sample() * 6.0;
        let (s, c) = angle.sin_cos();
        let eigen = [0.2 + 4.0 * sample(), 0.2 + 4.0 * sample()];
        let w = [
            c * c * eigen[0] + s * s * eigen[1],
            c * s * (eigen[0] - eigen[1]),
            s * s * eigen[0] + c * c * eigen[1],
        ];
        let v = [8.0 * sample() - 4.0, 8.0 * sample() - 4.0];
        let cap = 0.05 + 2.0 * sample();
        for scale in [1e-8, 1.0, 1e8] {
            rows.push(row(w.map(|x| x * scale), v.map(|x| x * scale), cap));
        }
    }
    let base = rows[5];
    let [a, b, d] = base.inverse_mass;
    let [x, y] = base.relative_velocity;
    rows.push(row([d, -b, a], [-y, x], base.max_impulse));
    let buffer = input(&engine, &rows);
    let solver = ExperimentalSoilFrictionDiskGpu::new(
        &engine,
        &buffer,
        rows.len() as u32,
        64,
        48 * rows.len() as u64 + 16,
    )
    .unwrap();
    let outputs = run(&engine, &solver);
    for (&i, &o) in rows.iter().zip(&outputs) {
        verify(i, o);
    }
    for indices in (6..rows.len() - 1).step_by(3) {
        for offset in [1, 2] {
            for axis in 0..2 {
                assert!(
                    (outputs[indices].impulse[axis] - outputs[indices + offset].impulse[axis])
                        .abs()
                        < 5e-6 * rows[indices].max_impulse
                );
            }
        }
    }
    let rotated = outputs.last().unwrap();
    assert!((rotated.impulse[0] + outputs[5].impulse[1]).abs() < 5e-6);
    assert!((rotated.impulse[1] - outputs[5].impulse[0]).abs() < 5e-6);
    let det = a * d - b * b;
    let unconstrained = [(-d * x + b * y) / det, (b * x - a * y) / det];
    let norm = unconstrained[0].hypot(unconstrained[1]);
    let radial = unconstrained.map(|p| p * base.max_impulse / norm);
    assert!((radial[0] - outputs[5].impulse[0]).hypot(radial[1] - outputs[5].impulse[1]) > 0.05);
    println!(
        "friction disk: {} rows passed f64, KKT, energy, scaling and rotation checks",
        rows.len()
    );
}

#[test]
#[ignore = "requires GPU"]
fn friction_disk_failures_reuse_and_budget() {
    let engine = Engine::new_gpu().unwrap();
    let valid = row([2.0, 0.5, 4.0], [3.0, -2.0], 0.3);
    let mut rows = vec![valid; 65];
    rows[0].max_impulse = -1.0;
    rows[1].relative_velocity[0] = f32::NAN;
    rows[2].inverse_mass = [1.0, 2.0, 1.0];
    rows[3].inverse_mass = [1.0, 1.0, 1.0];
    rows[4].padding[1] = 1.0;
    rows[5].max_impulse = f32::INFINITY;
    rows[6].inverse_mass = [1e20, 0.0, 1e20];
    rows[6].max_impulse = 1e20;
    let buffer = input(&engine, &rows);
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    for (count, iterations, budget) in [
        (65, 64, 3135),
        (65, 0, 3136),
        (65, 65, 3136),
        (66, 64, 3184),
    ] {
        assert!(
            ExperimentalSoilFrictionDiskGpu::new(&engine, &buffer, count, iterations, budget)
                .is_err()
        );
    }
    assert_eq!(scope.finish().observations, 0);
    let solver = ExperimentalSoilFrictionDiskGpu::new(&engine, &buffer, 65, 64, 3136).unwrap();
    assert_eq!(solver.owned_buffer_bytes(), 3136);
    let outputs = run(&engine, &solver);
    for (index, fault) in [
        FRICTION_DISK_INPUT,
        FRICTION_DISK_INPUT,
        FRICTION_DISK_METRIC,
        FRICTION_DISK_METRIC,
        FRICTION_DISK_INPUT,
        FRICTION_DISK_INPUT,
        FRICTION_DISK_NONFINITE,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            outputs[index].fault, fault,
            "row {index}: {:?}",
            outputs[index]
        );
        assert_eq!(outputs[index].impulse, [0.0; 2]);
    }
    rows.fill(valid);
    engine
        .queue()
        .unwrap()
        .write_buffer(&buffer, 0, bytemuck::cast_slice(&rows));
    for output in run(&engine, &solver) {
        verify(valid, output);
    }
    let short = ExperimentalSoilFrictionDiskGpu::new(&engine, &buffer, 65, 1, 3136).unwrap();
    for output in run(&engine, &short) {
        assert_eq!(output.fault, FRICTION_DISK_ITERATIONS);
        assert_eq!(output.impulse, [0.0; 2]);
    }
    let empty = input(&engine, &[]);
    let empty = ExperimentalSoilFrictionDiskGpu::new(&engine, &empty, 0, 64, 64).unwrap();
    assert_eq!(empty.owned_buffer_bytes(), 64);
    assert!(run(&engine, &empty).is_empty());
}

#[test]
#[ignore = "requires GPU"]
fn friction_disk_finite_mass_reaction_and_angular_energy() {
    let engine = Engine::new_gpu().unwrap();
    let (machine_mass, soil_mass, inertia) = (7.0f64, 2.0f64, 3.0f64);
    let arm = [0.6f64, -0.8f64];
    let machine = [2.0f64, -1.0f64];
    let soil = [-0.5f64, 0.75f64];
    let omega = 1.2f64;
    let inv = 1.0 / machine_mass + 1.0 / soil_mass;
    let w = [
        inv + arm[1] * arm[1] / inertia,
        -arm[0] * arm[1] / inertia,
        inv + arm[0] * arm[0] / inertia,
    ];
    let v = [
        machine[0] - omega * arm[1] - soil[0],
        machine[1] + omega * arm[0] - soil[1],
    ];
    let rows = [0.0, 0.4, 10.0].map(|cap| row(w.map(|x| x as f32), v.map(|x| x as f32), cap));
    let buffer = input(&engine, &rows);
    let solver = ExperimentalSoilFrictionDiskGpu::new(&engine, &buffer, 3, 64, 160).unwrap();
    let kinetic = |m: [f64; 2], s: [f64; 2], o: f64| {
        0.5 * (machine_mass * (m[0] * m[0] + m[1] * m[1])
            + soil_mass * (s[0] * s[0] + s[1] * s[1])
            + inertia * o * o)
    };
    for (input, output) in rows.into_iter().zip(run(&engine, &solver)) {
        verify(input, output);
        let p = output.impulse.map(f64::from);
        let m = [
            machine[0] + p[0] / machine_mass,
            machine[1] + p[1] / machine_mass,
        ];
        let s = [soil[0] - p[0] / soil_mass, soil[1] - p[1] / soil_mass];
        let o = omega + (arm[0] * p[1] - arm[1] * p[0]) / inertia;
        for axis in 0..2 {
            assert!(
                (machine_mass * (m[axis] - machine[axis]) + soil_mass * (s[axis] - soil[axis]))
                    .abs()
                    < 1e-12
            );
        }
        let angular_change = inertia * (o - omega)
            + soil_mass * (arm[0] * (s[1] - soil[1]) - arm[1] * (s[0] - soil[0]));
        assert!(angular_change.abs() < 1e-12);
        let energy = kinetic(m, s, o) - kinetic(machine, soil, omega);
        assert!(energy <= 1e-12);
        assert!((energy - f64::from(output.energy_change)).abs() < 5e-6);
    }
}
