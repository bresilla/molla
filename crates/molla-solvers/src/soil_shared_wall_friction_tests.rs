use super::*;
use crate::soil_wall_friction_gpu::SoilWallFrictionOutput;
use glam::DVec3;

#[path = "soil_shared_wall_rounding_tests.rs"]
mod rounding;

#[path = "soil_shared_wall_paired_tests.rs"]
mod paired;

fn vec(v: [f32; 4]) -> DVec3 {
    DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
}
fn pack(v: DVec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}
fn buffer<T: bytemuck::Pod>(engine: &Engine, data: &[T], minimum: u64) -> wgpu::Buffer {
    let bytes = bytemuck::cast_slice(data);
    let result = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared wall fixture"),
            size: (bytes.len() as u64).max(minimum),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
    if !bytes.is_empty() {
        engine.queue().unwrap().write_buffer(&result, 0, bytes);
    }
    result
}
struct Fixture {
    mass: Vec<f32>,
    free: Vec<f32>,
    rows: Vec<[f32; 4]>,
    contacts: Vec<SoilSharedWallContact>,
}
fn create(engine: &Engine, f: &Fixture, sweeps: u32) -> ExperimentalSoilSharedWallFrictionGpu {
    create_mode(engine, f, sweeps, true)
}
fn create_mode(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    accelerate: bool,
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_ordered(
        engine,
        f,
        sweeps,
        [accelerate, true, true, false, false],
        [true, true, true, false, true],
    )
}
fn create_scheduled(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    accelerate: bool,
    segmented: bool,
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_rounding(engine, f, sweeps, accelerate, segmented, false)
}
fn create_rounding(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    accelerate: bool,
    segmented: bool,
    register_rounding: bool,
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_wall_axes(
        engine,
        f,
        sweeps,
        accelerate,
        segmented,
        register_rounding,
        false,
    )
}
fn create_wall_axes(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    accelerate: bool,
    segmented: bool,
    register_rounding: bool,
    direct_wall_axes: bool,
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_variant(
        engine,
        f,
        sweeps,
        accelerate,
        segmented,
        register_rounding,
        direct_wall_axes,
        false,
    )
}
fn create_variant(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    accelerate: bool,
    segmented: bool,
    register_rounding: bool,
    direct_wall_axes: bool,
    warm_start: bool,
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_cached(
        engine,
        f,
        sweeps,
        [
            accelerate,
            segmented,
            register_rounding,
            direct_wall_axes,
            warm_start,
        ],
        false,
    )
}
fn create_cached(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    options: [bool; 5],
    cache_setup: bool,
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_linear(engine, f, sweeps, options, cache_setup, false)
}
fn create_linear(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    options: [bool; 5],
    cache_setup: bool,
    unroll_linear: bool,
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_cooperative(
        engine,
        f,
        sweeps,
        options,
        [cache_setup, unroll_linear, false],
    )
}
fn create_cooperative(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    options: [bool; 5],
    variants: [bool; 3],
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_local_cooperative(
        engine,
        f,
        sweeps,
        options,
        [variants[0], variants[1], variants[2], false],
    )
}
fn create_local_cooperative(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    options: [bool; 5],
    variants: [bool; 4],
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_ordered(
        engine,
        f,
        sweeps,
        options,
        [variants[0], variants[1], variants[2], variants[3], false],
    )
}
fn create_ordered(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    options: [bool; 5],
    variants: [bool; 5],
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_chunks(engine, f, sweeps, options, variants, 0)
}
fn create_chunks(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    options: [bool; 5],
    variants: [bool; 5],
    chunk: u32,
) -> ExperimentalSoilSharedWallFrictionGpu {
    create_preparation(engine, f, sweeps, options, variants, chunk, false)
}

fn create_preparation(
    engine: &Engine,
    f: &Fixture,
    sweeps: u32,
    options: [bool; 5],
    variants: [bool; 5],
    chunk: u32,
    split: bool,
) -> ExperimentalSoilSharedWallFrictionGpu {
    let n = f.free.len() as u32;
    let mass = buffer(engine, &f.mass, 4);
    let free = buffer(engine, &f.free, 4);
    let rows = buffer(engine, &f.rows, 16 * u64::from(n));
    let contacts = buffer(engine, &f.contacts, 64);
    ExperimentalSoilSharedWallFrictionGpu::with_preparation(
        engine,
        SoilSharedWallInputs {
            mass: &mass,
            free_rates: &free,
            point_rows: &rows,
            contacts: &contacts,
        },
        SoilSharedWallCapacity {
            dofs: n,
            contacts: f.contacts.len() as u32,
            max_sweeps: sweeps,
            local_iterations: 64,
            max_owned_bytes: u64::MAX,
        },
        options,
        variants,
        chunk,
        split,
    )
    .unwrap()
}
fn run(
    engine: &Engine,
    solver: &ExperimentalSoilSharedWallFrictionGpu,
    n: usize,
    c: usize,
    clear: bool,
) -> (Vec<f32>, Vec<SoilWallFrictionOutput>, [u32; 8]) {
    let device = engine.device().unwrap();
    let size = 4 * n as u64 + 80 * c as u64 + 32;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfer = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    if clear {
        encoder.clear_buffer(&solver.status, 0, None);
    }
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocation.finish().observations, 0);
    let traffic = transfer.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(
        dispatch.finish().encoded_dispatches,
        (if c == 0 {
            1
        } else {
            2 + u64::from(solver.prepare_cache.is_some())
        }) + u64::from(solver.solve_dispatches)
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&solver.rates, 0, &staging, 0, 4 * n as u64);
    if c > 0 {
        encoder.copy_buffer_to_buffer(&solver.outputs, 0, &staging, 4 * n as u64, 80 * c as u64);
    }
    encoder.copy_buffer_to_buffer(&solver.status, 0, &staging, size - 32, 32);
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(15)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(15))
        .unwrap()
        .unwrap();
    let data = staging.slice(..).get_mapped_range();
    let rates = bytemuck::cast_slice::<u8, f32>(&data[..4 * n]).to_vec();
    let outputs =
        bytemuck::cast_slice::<u8, SoilWallFrictionOutput>(&data[4 * n..4 * n + 80 * c]).to_vec();
    let status = *bytemuck::from_bytes::<[u32; 8]>(&data[size as usize - 32..]);
    drop(data);
    staging.unmap();
    (rates, outputs, status)
}
fn verify(f: &Fixture, rates: &[f32], outputs: &[SoilWallFrictionOutput]) {
    verify_tolerance(f, rates, outputs, 2e-5);
}

fn verify_tolerance(
    f: &Fixture,
    rates: &[f32],
    outputs: &[SoilWallFrictionOutput],
    tolerance: f64,
) {
    verify_precise_tolerance(
        f,
        &rates.iter().map(|q| *q as f64).collect::<Vec<_>>(),
        outputs,
        tolerance,
    );
}

fn verify_precise_tolerance(
    f: &Fixture,
    rates: &[f64],
    outputs: &[SoilWallFrictionOutput],
    tolerance: f64,
) {
    verify_precise_free_tolerance(
        f,
        &f.free.iter().map(|q| *q as f64).collect::<Vec<_>>(),
        rates,
        outputs,
        tolerance,
    );
}

fn verify_precise_free_tolerance(
    f: &Fixture,
    free: &[f64],
    rates: &[f64],
    outputs: &[SoilWallFrictionOutput],
    tolerance: f64,
) {
    let n = f.free.len();
    let mut energy = 0.0;
    let mut energy_scale = 1.0;
    for i in 0..n {
        let mut reaction = 0.0;
        let mut scale = 1.0;
        for j in 0..n {
            let p = f.mass[i * n + j] as f64 * (rates[j] - free[j]);
            reaction += p;
            scale += p.abs();
            let e = 0.5 * f.mass[i * n + j] as f64 * (rates[i] * rates[j] - free[i] * free[j]);
            energy += e;
            energy_scale += e.abs();
        }
        for (c, output) in outputs.iter().enumerate() {
            let p = vec(f.rows[c * n + i]).dot(-vec(output.machine_impulse_normal));
            reaction += p;
            scale += p.abs();
        }
        assert!(
            reaction.abs() <= tolerance * scale,
            "generalized momentum {i}: {reaction}, gate={}",
            tolerance * scale
        );
    }
    for (c, output) in outputs.iter().enumerate() {
        let node = f.contacts[c];
        if node.soil_mass[3] == 0.0 {
            assert_eq!(*output, SoilWallFrictionOutput::default());
            continue;
        }
        let k = -vec(output.machine_impulse_normal);
        let wall = vec(output.wall_impulse);
        let soil = vec(output.soil_velocity_energy);
        let mass = node.soil_mass[3] as f64;
        let machine = (0..n)
            .map(|i| vec(f.rows[c * n + i]) * rates[i])
            .sum::<DVec3>();
        let normal = vec(node.normal_gap).normalize();
        let relative = soil - machine;
        let lambda = k.dot(normal);
        let tangent = k - lambda * normal;
        let slip = relative - relative.dot(normal) * normal;
        let cap = node.friction[0] as f64 * lambda;
        let separation = relative.dot(normal) + node.normal_gap[3] as f64;
        let ps = 1.0 + k.length() + wall.length();
        let vs = 1.0 + soil.length() + machine.length();
        assert_eq!(output.status[0], 0);
        assert!((output.machine_impulse_normal[3] as f64 - lambda).abs() <= tolerance * ps);
        assert!((mass * (soil - vec(node.soil_mass)) - k - wall).length() <= tolerance * ps);
        assert!((machine - vec(output.machine_velocity_residual)).length() <= tolerance * vs);
        assert!(lambda >= -tolerance * ps && separation >= -tolerance * vs);
        assert!(
            (lambda * separation).abs() <= tolerance * ps * vs,
            "normal complementarity contact={c} lambda={lambda:e} separation={separation:e} ps={ps:e} vs={vs:e} gap={:e} error={:e} limit={tolerance:e}",
            node.normal_gap[3],
            (lambda * separation).abs() / (ps * vs)
        );
        assert!(tangent.length() <= cap + tolerance * ps);
        if slip.length() > tolerance * vs {
            assert!((tangent + cap * slip.normalize()).length() <= tolerance * ps);
        }
        let mut wall_bits = 0;
        for axis in 0..3 {
            let flag = (node.boundary[0] >> (2 * axis)) & 3;
            if flag == 0 {
                assert_eq!(wall[axis], 0.0);
            } else {
                let sign = if flag == 1 { 1.0 } else { -1.0 };
                if sign * wall[axis] > 0.0 {
                    wall_bits |= 1 << axis;
                }
                assert!(
                    sign * soil[axis] >= -tolerance * vs && sign * wall[axis] >= -tolerance * ps
                );
                assert!((soil[axis] * wall[axis]).abs() <= tolerance * ps * vs);
            }
        }
        assert_eq!(output.status[3], wall_bits);
        let e = 0.5 * mass * (soil.length_squared() - vec(node.soil_mass).length_squared());
        assert!(
            (e - output.soil_velocity_energy[3] as f64).abs()
                <= tolerance
                    * (1.0 + mass * (soil.length_squared() + vec(node.soil_mass).length_squared()))
        );
        energy += e;
        energy_scale += e.abs();
    }
    assert!(energy <= tolerance * energy_scale, "global energy {energy}");
}
fn pair(walls: bool, mu: f64) -> Fixture {
    let normal = if walls {
        DVec3::new(-1.0, 1.0, 0.0).normalize()
    } else {
        DVec3::Y
    };
    let tangent = if walls {
        DVec3::new(1.0, 1.0, 0.0).normalize()
    } else {
        DVec3::X
    };
    let mut f = Fixture {
        mass: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        free: vec![0.0; 3],
        rows: Vec::new(),
        contacts: Vec::new(),
    };
    if mu == 0.0 {
        f.free = if walls {
            vec![-1.0, 1.0, 0.0]
        } else {
            vec![0.0, 1.0, 0.0]
        };
    } else {
        let k = normal + mu * tangent;
        let free = 0.5 * tangent + 2.0 * k;
        f.free = vec![free.x as f32, free.y as f32, free.z as f32];
    }
    for _ in 0..2 {
        let k = normal + mu * tangent;
        let wall = if walls {
            DVec3::new(-k.x + 0.25, 0.0, 0.0)
        } else {
            DVec3::ZERO
        };
        let soil = if mu == 0.0 { DVec3::ZERO } else { -(k + wall) };
        f.contacts.push(SoilSharedWallContact {
            soil_mass: pack(soil, 1.0),
            normal_gap: pack(normal, 0.0),
            friction: [mu as f32, 0.0, 0.0, 0.0],
            boundary: [u32::from(walls), 0, 0, 0],
        });
        f.rows.extend([
            pack(DVec3::X, 0.0),
            pack(DVec3::Y, 0.0),
            pack(DVec3::Z, 0.0),
        ]);
    }
    f
}
#[test]
fn shared_wall_layout_and_capacity() {
    assert_eq!(std::mem::size_of::<SoilSharedWallContact>(), 64);
    let cap = SoilSharedWallCapacity {
        dofs: 3,
        contacts: 2,
        max_sweeps: 128,
        local_iterations: 64,
        max_owned_bytes: 1648,
    };
    assert_eq!(
        ExperimentalSoilSharedWallFrictionGpu::required_bytes(cap).unwrap(),
        1648
    );
    assert!(
        ExperimentalSoilSharedWallFrictionGpu::required_bytes(SoilSharedWallCapacity {
            dofs: 33,
            ..cap
        })
        .is_err()
    );
}

#[test]
fn shared_wall_ordered_history_layout() {
    #[repr(C, align(16))]
    struct History {
        impulses: [[f32; 4]; 4],
        basis: [[f32; 4]; 6],
        metric: [[f32; 6]; 6],
        scale: [f32; 6],
        active_index: u32,
        padding: u32,
    }
    assert_eq!(std::mem::size_of::<History>(), 336);
    assert_eq!(std::mem::offset_of!(History, basis), 64);
    assert_eq!(std::mem::offset_of!(History, metric), 160);
    assert_eq!(std::mem::offset_of!(History, scale), 304);
    assert_eq!(std::mem::offset_of!(History, active_index), 328);
    #[repr(C, align(16))]
    struct RedistributedHistory {
        history: History,
        correction: [f32; 4],
    }
    assert_eq!(std::mem::size_of::<RedistributedHistory>(), 352);
    assert_eq!(std::mem::offset_of!(RedistributedHistory, correction), 336);
}

#[test]
fn shared_wall_split_preparation_shader_validation() {
    let source = [
        include_str!("../shaders/soil_wall_friction_block.wgsl"),
        include_str!("../shaders/soil_shared_wall_friction.wgsl"),
    ]
    .join("\n");
    for source in [source.clone(), stable_merit_source(&source)] {
        let source = cached_setup_source(&source);
        let source = register_rounding_source(&source, "shared_params.width >> 16u");
        let source = unrolled_direction_source(&source);
        let source = cooperative_source(&source);
        let source = ordered_contact_source(&source);
        let candidate = split_preparation_source(&source);
        validate_split_preparation(&candidate);
        let storage = storage_response_source(&candidate);
        assert!(!storage.contains("var response:array<vec3<f32>,32>"));
        validate_split_preparation(&storage);
        validate_split_preparation(&storage_setup_source(&candidate));
        validate_split_preparation(&storage_setup_source(&storage));
        let paired = paired_rate_source(&coverage_certificate_source(&storage_setup_source(
            &storage,
        )));
        validate_split_preparation(&paired);
        let traced = certificate_trace_source(&paired);
        assert_eq!(certificate_trace_source(&traced), traced);
        validate_split_preparation(&traced);
        validate_split_preparation(&certificate_trace_source(&storage_setup_source(&storage)));
        validate_split_preparation(&represented_rate_source(&certificate_trace_source(
            &storage_setup_source(&storage),
        )));
        validate_split_preparation(&coupled_represented_source(&certificate_trace_source(
            &storage_setup_source(&storage),
        )));
        validate_split_preparation(&backtracked_acceleration_source(
            &coupled_represented_source(&certificate_trace_source(&storage_setup_source(&storage))),
        ));
        validate_split_preparation(&redistributed_source(&coupled_represented_source(
            &certificate_trace_source(&storage_setup_source(&storage)),
        )));
        validate_split_preparation(&bounded_dual_source(&redistributed_source(
            &coupled_represented_source(&certificate_trace_source(&storage_setup_source(&storage))),
        )));
    }
}

#[test]
fn shared_wall_projected_merit_cancellation() {
    for impulse in [0.125_f32, 1.0, 128.0] {
        for sign in [-1.0_f32, 1.0] {
            let velocity_step = sign * impulse * (0.125 * f32::EPSILON);
            let projected = std::hint::black_box(impulse - velocity_step).max(0.0);
            assert_eq!(impulse - projected, 0.0);
            assert_eq!(impulse.min(velocity_step), velocity_step);
            assert_ne!(velocity_step, 0.0);
        }
    }
    for impulse in [-1.0_f32, 0.0, 0.5, 1.0] {
        for step in [-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
            let reference = impulse as f64 - (impulse as f64 - step as f64).max(0.0);
            assert_eq!(impulse.min(step) as f64, reference);
        }
    }
}

fn validate_split_preparation(candidate: &str) {
    let module = wgpu::naga::front::wgsl::parse_str(&candidate)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&candidate)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(
        module
            .entry_points
            .iter()
            .any(|e| e.name == "shared_prepare_cache")
    );
    let status = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("SharedStatus"))
        .unwrap()
        .1;
    let wgpu::naga::TypeInner::Struct { span, .. } = &status.inner else {
        panic!("status must be a struct")
    };
    assert_eq!(*span, 432);
    assert_eq!(
        candidate
            .matches("shared_cache_prepare(shared_inputs[c],c);")
            .count(),
        1
    );
}

#[test]
fn shared_wall_chunked_shader_validation() {
    let source = [
        include_str!("../shaders/soil_wall_friction_block.wgsl"),
        include_str!("../shaders/soil_shared_wall_friction.wgsl"),
    ]
    .join("\n");
    let source = cached_setup_source(&source);
    let source = register_rounding_source(&source, "shared_params.width >> 16u");
    let source = unrolled_direction_source(&source);
    let source = cooperative_source(&source);
    let source = ordered_contact_source(&source);
    for chunk in [1, 16, 64] {
        let candidate = chunked_contact_source(&source, chunk);
        let module = wgpu::naga::front::wgsl::parse_str(&candidate)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&candidate)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let status = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("SharedStatus"))
            .unwrap()
            .1;
        let wgpu::naga::TypeInner::Struct { members, span } = &status.inner else {
            panic!("status must be a struct")
        };
        assert_eq!(*span, 688);
        assert_eq!(members.last().unwrap().offset, 432);
        let local = candidate
            .split("fn coop_local_sweep()")
            .nth(1)
            .unwrap()
            .split("fn coop_finish()")
            .next()
            .unwrap();
        assert_eq!(
            local
                .matches(&format!("shared_status.control.z+{chunk}u"))
                .count(),
            1
        );
        assert!(!local.contains("var slot=0u"));
    }
}

#[test]
#[ignore = "requires GPU; split preparation candidate, not runtime acceptance"]
fn shared_wall_split_matches_ordered() {
    compare_preparation_flags(
        [true, true, true, false, true],
        [true, true, true, false, true],
        true,
        0,
        true,
    );
}

#[test]
#[ignore = "requires GPU; split preparation candidate"]
fn shared_wall_split_sparse_and_empty() {
    let engine = Engine::new_gpu().unwrap();
    let flags = [true, true, true, false, true];
    let options = [true, true, true, false, false];
    let sparse = spread_contacts(&articulated(65), 4096, 63);
    let mut empty = pair(false, 0.3);
    empty.contacts.clear();
    empty.rows.clear();
    for f in [sparse, empty] {
        let reference = create_ordered(&engine, &f, 64, options, flags);
        let candidate = create_preparation(&engine, &f, 64, options, flags, 0, true);
        assert_eq!(
            candidate.owned_buffer_bytes(),
            reference.owned_buffer_bytes()
                + if redistribution_selected() {
                    16 * f.contacts.len().max(1) as u64
                } else {
                    0
                }
        );
        assert_eq!(candidate.status.size(), 432);
        let a = run(&engine, &reference, f.free.len(), f.contacts.len(), true);
        let b = run(&engine, &candidate, f.free.len(), f.contacts.len(), true);
        assert_eq!(a, b);
        assert_eq!(b.2[0], 0);
        assert_eq!(
            run(&engine, &candidate, f.free.len(), f.contacts.len(), true),
            b
        );
        verify(&f, &b.0, &b.1);
    }
}

#[test]
#[ignore = "requires GPU; split preparation candidate"]
fn shared_wall_split_refresh_and_budget() {
    refresh_preparation_and_budget(true, false, true, 0, true);
}

#[test]
#[ignore = "requires GPU; bounded contact-dispatch candidate, not runtime acceptance"]
fn shared_wall_chunked_matches_ordered() {
    for chunk in [1, 16, 64] {
        compare_chunked_flags(
            [true, true, true, false, true],
            [true, true, true, false, true],
            true,
            chunk,
        );
    }
}

#[test]
#[ignore = "requires GPU; bounded contact-dispatch candidate"]
fn shared_wall_chunked_sparse_and_empty() {
    let engine = Engine::new_gpu().unwrap();
    let flags = [true, true, true, false, true];
    let options = [true, true, true, false, false];
    let compact = articulated(65);
    let sparse = spread_contacts(&compact, 4096, 63);
    let mut empty = pair(false, 0.3);
    empty.contacts.clear();
    empty.rows.clear();
    for f in [sparse, empty] {
        let reference = create_ordered(&engine, &f, 64, options, flags);
        let candidate = create_chunks(&engine, &f, 64, options, flags, 64);
        assert_eq!(
            candidate.owned_buffer_bytes(),
            reference.owned_buffer_bytes() + 256
        );
        assert_eq!(candidate.status.size(), 688);
        let a = run(&engine, &reference, f.free.len(), f.contacts.len(), true);
        let b = run(&engine, &candidate, f.free.len(), f.contacts.len(), true);
        assert_eq!(a, b);
        assert_eq!(b.2[0], 0);
        assert_eq!(
            run(&engine, &candidate, f.free.len(), f.contacts.len(), true),
            b
        );
        verify(&f, &b.0, &b.1);
    }
}

#[test]
#[ignore = "requires GPU; bounded contact-dispatch candidate"]
fn shared_wall_chunked_refresh_and_budget() {
    refresh_chunked_and_budget(true, false, true, 1);
}

#[test]
#[ignore = "requires GPU; test-only input capture and deliberate stop"]
fn shared_wall_capture_first_input_and_sticky_stop() {
    let engine = Engine::new_gpu().unwrap();
    let f = pair(true, 0.3);
    let mut solver = create(&engine, &f, 64);
    let snapshot = solver.capture_first_input(&engine);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let submit = |clear| {
        let mut encoder = device.create_command_encoder(&Default::default());
        if clear {
            encoder.clear_buffer(&solver.status, 0, None);
        }
        solver.encode(&mut encoder);
        queue.submit([encoder.finish()]);
    };
    submit(true);
    assert_eq!(
        u32::from_ne_bytes(
            capture::read(&engine, &solver.status)[..4]
                .try_into()
                .unwrap()
        ),
        capture::CAPTURE_STOP
    );
    let expected: [&[u8]; 4] = [
        bytemuck::cast_slice(&f.mass),
        bytemuck::cast_slice(&f.free),
        bytemuck::cast_slice(&f.rows),
        bytemuck::cast_slice(&f.contacts),
    ];
    for (b, data) in snapshot.inputs.iter().zip(expected) {
        assert_eq!(capture::read(&engine, b), data);
    }
    for b in &solver.capture_inputs {
        queue.write_buffer(b, 0, &vec![0; b.size() as usize]);
    }
    submit(true);
    for (b, data) in snapshot.inputs.iter().zip(expected) {
        assert_eq!(capture::read(&engine, b), data);
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.clear_buffer(&snapshot.control, 0, None);
    queue.submit([encoder.finish()]);
    queue.write_buffer(
        &solver.status,
        0,
        bytemuck::cast_slice(&[SHARED_WALL_INPUT]),
    );
    submit(false);
    assert_eq!(capture::read(&engine, &snapshot.control), vec![0; 16]);
    assert_eq!(
        u32::from_ne_bytes(
            capture::read(&engine, &solver.status)[..4]
                .try_into()
                .unwrap()
        ),
        SHARED_WALL_INPUT
    );
    for (b, data) in snapshot.inputs.iter().zip(expected) {
        assert_eq!(capture::read(&engine, b), data);
    }
    submit(true);
    assert_eq!(
        u32::from_ne_bytes(
            capture::read(&engine, &solver.status)[..4]
                .try_into()
                .unwrap()
        ),
        capture::CAPTURE_STOP
    );
    for b in &snapshot.inputs {
        assert_eq!(capture::read(&engine, b), vec![0; b.size() as usize]);
    }
}

#[test]
#[ignore = "requires GPU and fresh SOIL_WALL_STAGE_TRACE; diagnostic equivalence"]
fn shared_wall_stage_trace_preserves_result() {
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_WALL_STAGE_TRACE").expect("fresh trace directory required"),
    );
    let engine = std::sync::Arc::new(Engine::new_gpu().unwrap());
    let f = pair(true, 0.3);
    let reference = create(&engine, &f, 4);
    let expected = run(&engine, &reference, 3, 2, true);
    let mut traced = create(&engine, &f, 4);
    traced.trace_stages(engine.clone(), root.clone());
    let device = engine.device().unwrap();
    for index in 0..2 {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&traced.status, 0, None);
        traced.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        let path = root.join(format!("solve-{index:03}"));
        assert_eq!(
            std::fs::read(path.join("rates.bin")).unwrap(),
            bytemuck::cast_slice::<_, u8>(&expected.0)
        );
        assert_eq!(
            std::fs::read(path.join("outputs.bin")).unwrap(),
            bytemuck::cast_slice::<_, u8>(&expected.1)
        );
        assert_eq!(
            &std::fs::read(path.join("status.bin")).unwrap()[..32],
            bytemuck::cast_slice::<_, u8>(&expected.2)
        );
        assert_eq!(
            std::fs::read(path.join("mass.bin")).unwrap(),
            bytemuck::cast_slice::<_, u8>(&f.mass)
        );
        assert!(path.join("sweep-004.json").exists());
        assert_eq!(
            std::fs::read(path.join("factor-params.bin")).unwrap(),
            bytemuck::cast_slice::<_, u8>(&[3u32, 2, 4, 64])
        );
        assert_eq!(
            std::fs::metadata(path.join("factor-lower.bin"))
                .unwrap()
                .len(),
            36
        );
    }
}

#[test]
#[ignore = "requires GPU, SOIL_WALL_INPUT_REPLAY and fresh SOIL_WALL_PROBE_OUTPUT; one-sweep diagnostic"]
fn shared_wall_capture_one_sweep_probe() {
    captured_input_probe(1, 64);
}

#[test]
#[ignore = "requires GPU, SOIL_WALL_INPUT_REPLAY and fresh SOIL_WALL_PROBE_OUTPUT; captured full-solve diagnostic"]
fn shared_wall_capture_full_solve_probe() {
    captured_input_probe(64, 64);
}

#[test]
#[ignore = "requires GPU, SOIL_WALL_INPUT_REPLAY and fresh SOIL_WALL_PROBE_OUTPUT; captured ordered control"]
fn shared_wall_capture_ordered_probe() {
    captured_input_probe(64, 0);
}

fn captured_input_probe(sweeps: u32, chunk: u32) {
    captured_probe(sweeps, chunk, false, false, false);
}

#[test]
#[ignore = "requires native GPU, SOIL_WALL_INPUT_REPLAY and fresh SOIL_WALL_PROBE_OUTPUT; preparation only"]
fn shared_wall_capture_native_prepare_probe() {
    captured_probe(64, 0, true, true, false);
}

#[test]
#[ignore = "requires GPU, SOIL_WALL_INPUT_REPLAY and fresh SOIL_WALL_PROBE_OUTPUT; preparation control"]
fn shared_wall_capture_baseline_prepare_probe() {
    captured_probe(64, 0, true, false, false);
}

#[test]
#[ignore = "requires native GPU, captured input and fresh output; split preparation diagnostic"]
fn shared_wall_capture_split_prepare_probe() {
    captured_probe(64, 0, true, true, true);
}

#[test]
#[ignore = "requires native GPU, captured input and fresh output; split full-solve diagnostic"]
fn shared_wall_capture_split_solve_probe() {
    captured_probe(64, 0, false, true, true);
}

fn captured_fixture(root: &std::path::Path) -> Fixture {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("capture.json")).unwrap()).unwrap();
    assert_eq!(meta["schema"], 1);
    let n = meta["dofs"].as_u64().unwrap() as usize;
    let c = meta["contacts"].as_u64().unwrap() as usize;
    assert!((1..=32).contains(&n) && c <= 16384);
    let words = |name: &str, count: usize| {
        let bytes = std::fs::read(root.join(format!("{name}.bin"))).unwrap();
        assert_eq!(bytes.len(), 4 * count);
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let row_bytes = std::fs::read(root.join("rows.bin")).unwrap();
    let contact_bytes = std::fs::read(root.join("contacts.bin")).unwrap();
    assert_eq!(row_bytes.len(), 16 * c.max(1) * n);
    assert_eq!(contact_bytes.len(), 64 * c.max(1));
    let f = Fixture {
        mass: words("mass", n * n),
        free: words("free", n),
        rows: row_bytes
            .chunks_exact(16)
            .take(c * n)
            .map(bytemuck::pod_read_unaligned)
            .collect(),
        contacts: contact_bytes
            .chunks_exact(64)
            .take(c)
            .map(bytemuck::pod_read_unaligned)
            .collect(),
    };
    f
}

fn captured_probe(sweeps: u32, chunk: u32, prepare_only: bool, native: bool, split: bool) {
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_WALL_INPUT_REPLAY").expect("captured parity directory required"),
    );
    let output = std::path::PathBuf::from(
        std::env::var_os("SOIL_WALL_PROBE_OUTPUT").expect("fresh probe output required"),
    );
    std::fs::create_dir(&output).unwrap();
    let f = captured_fixture(&root);
    let n = f.free.len();
    let c = f.contacts.len();
    let active = f.contacts.iter().filter(|v| v.soil_mass[3] > 0.0).count();
    eprintln!(
        "actual input probe: dofs={n} capacity={c} active={active}; sweeps={sweeps}, chunk={chunk}"
    );
    let engine = std::sync::Arc::new(if native {
        Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap()
    } else {
        Engine::new_gpu().unwrap()
    });
    let solver = create_preparation(
        &engine,
        &f,
        sweeps,
        [true, true, true, false, false],
        [true, true, true, false, true],
        chunk,
        split,
    );
    if prepare_only {
        let trace = capture::SharedWallTrace::new(
            engine.clone(),
            output.join("trace"),
            &solver.capture_inputs,
            &solver.status,
            solver.capture_capacity,
            solver.paired_rates,
        );
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.clear_buffer(&solver.status, 0, None);
        let path = trace.before(
            &mut encoder,
            &solver.capture_inputs,
            solver.capture_capacity,
        );
        for (index, (pipeline, groups)) in solver.passes().take(2 + usize::from(split)).enumerate()
        {
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &solver.group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            trace.checkpoint(&mut encoder, &path, index, &solver.status, split);
            if index == 0 {
                trace.factor_state(&path, &solver.capture_storage);
            }
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        trace.finish(&path, &solver);
        let bytes = capture::read(&engine, solver.status_buffer());
        let status: Vec<_> = bytes[..32]
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        std::fs::write(output.join("probe.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "schema": 1, "native_device_features": native, "prepare_only": true,
            "split_preparation": split,
            "storage_response": split && std::env::var_os("SOIL_WALL_TEST_STORAGE_RESPONSE").is_some(),
            "storage_setup": split && std::env::var_os("SOIL_WALL_TEST_STORAGE_SETUP").is_some(),
            "status": status, "input": root, "active": active, "acceptance": false,
            "scope": "Captured input factor/preparation only; no shared sweep, physical result or trajectory acceptance"
        })).unwrap()).unwrap();
        assert_eq!(status, vec![0; 8]);
        return;
    }
    let start = std::time::Instant::now();
    let result = run(&engine, &solver, n, c, true);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let full_status = capture::read(&engine, solver.status_buffer());
    let rates: Vec<_> = full_status[32..32 + 8 * n]
        .chunks_exact(8)
        .map(|b| {
            f32::from_ne_bytes(b[..4].try_into().unwrap()) as f64
                + f32::from_ne_bytes(b[4..].try_into().unwrap()) as f64
        })
        .collect();
    let diagnostic_rates = if result.2[0] == 0 {
        result.0.iter().map(|q| *q as f64).collect()
    } else {
        rates
    };
    let mut momentum = captured_momentum_diagnostics(&f, &diagnostic_rates, &result.1);
    momentum["rate_source"] = serde_json::json!(if result.2[0] == 0 {
        "published"
    } else {
        "continuation"
    });
    std::fs::write(
        output.join("momentum-diagnostics.json"),
        serde_json::to_vec_pretty(&momentum).unwrap(),
    )
    .unwrap();
    for (name, buffer) in [
        ("status", solver.status_buffer()),
        ("rates", solver.rates_buffer()),
        ("outputs", solver.output_buffer()),
        ("conditional", solver.conditional_inputs_buffer()),
    ] {
        std::fs::write(
            output.join(format!("{name}.bin")),
            capture::read(&engine, buffer),
        )
        .unwrap();
    }
    std::fs::write(output.join("probe.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema": 1, "input": root, "dofs": n, "capacity": c, "active": active,
        "status": result.2, "elapsed_ms_including_diagnostic_readback": elapsed_ms,
        "contacts_per_dispatch": chunk, "max_sweeps": sweeps,
        "split_preparation": split,
        "storage_response": split && std::env::var_os("SOIL_WALL_TEST_STORAGE_RESPONSE").is_some(),
        "storage_setup": split && std::env::var_os("SOIL_WALL_TEST_STORAGE_SETUP").is_some(),
        "certificate_trace": std::env::var_os("SOIL_WALL_TEST_CERTIFICATE_TRACE").is_some(),
        "represented_rate_projection": split && std::env::var_os("SOIL_WALL_TEST_REPRESENTED_RATES").is_some(),
        "coupled_rounding": split && std::env::var_os("SOIL_WALL_TEST_REPRESENTED_RATES").is_some() && std::env::var_os("SOIL_WALL_TEST_COUPLED_ROUNDING").is_some(),
        "bounded_dual": split && redistribution_selected() && std::env::var_os("SOIL_WALL_TEST_BOUNDED_DUAL").is_some(),
        "redistribution": split && redistribution_selected(),
        "owned_buffer_bytes": solver.owned_buffer_bytes(),
        "certificate_reason": result.2[7] & 255, "certificate_index_one_based": result.2[7] >> 8,
        "scope": "Actual captured input, shared solve diagnostic; not trajectory or performance acceptance",
        "acceptance": false
    })).unwrap()).unwrap();
    eprintln!(
        "actual captured-solve status={:?}, diagnostic elapsed_ms={elapsed_ms}",
        result.2
    );
    assert!(matches!(
        result.2[0],
        0 | SHARED_WALL_SWEEPS | SHARED_WALL_LOCAL
    ));
    if result.2[0] == 0 {
        verify_tolerance(&f, &result.0, &result.1, 1e-6);
    }
}

fn captured_momentum_diagnostics(
    f: &Fixture,
    rates: &[f64],
    outputs: &[SoilWallFrictionOutput],
) -> serde_json::Value {
    let n = f.free.len();
    let rows: Vec<_> = (0..n)
        .map(|i| {
            let mut compensated = 0.0;
            let mut rounded = 0.0;
            let mut scale = 1.0;
            let mut rounding_shift = 0.0;
            for j in 0..n {
                let mass = f.mass[i * n + j] as f64;
                let q = rates[j] as f32 as f64;
                compensated += mass * (rates[j] - f.free[j] as f64);
                rounded += mass * (q - f.free[j] as f64);
                scale += mass.abs() * (q - f.free[j] as f64).abs();
                rounding_shift += mass * (q - rates[j]);
            }
            for (c, output) in outputs.iter().enumerate() {
                let row = vec(f.rows[c * n + i]);
                let impulse = -vec(output.machine_impulse_normal);
                compensated += row.dot(impulse);
                rounded += row.dot(impulse);
                scale += row.abs().dot(impulse.abs());
            }
            serde_json::json!({"dof": i, "compensated_residual_f64": compensated,
            "rounded_residual_f64": rounded, "rounding_shift_f64": rounding_shift,
            "gate": 1e-6 * scale, "rounded_gate_ratio": rounded.abs()/(1e-6*scale)})
        })
        .collect();
    serde_json::json!({"scope": "CPU f64 momentum diagnosis of supplied rates, not a GPU certificate; continuation rates may precede the last solve sweep", "acceptance": false, "rows": rows})
}

fn articulated(count: usize) -> Fixture {
    let n = 8;
    let mut mass = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            mass[i * n + j] =
                0.005 * ((i + 1) * (j + 1)) as f32 + if i == j { 8.0 + i as f32 } else { 0.0 };
        }
    }
    let mut f = Fixture {
        mass,
        free: vec![0.0; n],
        rows: vec![],
        contacts: vec![],
    };
    let target = [0.3, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut reaction = vec![0.0; n];
    for c in 0..count {
        let arm = DVec3::new((c % 5) as f64 * 0.2 - 0.4, -0.5, (c % 3) as f64 * 0.3 - 0.3);
        let columns = [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::X.cross(arm),
            DVec3::Y.cross(arm),
            DVec3::Z.cross(arm),
            if c % 3 == 0 {
                DVec3::X.cross(arm - DVec3::new(0.0, -0.3, 0.0))
            } else {
                DVec3::ZERO
            },
            if c % 3 == 1 {
                DVec3::Z.cross(arm - DVec3::new(0.1, -0.3, 0.0))
            } else {
                DVec3::ZERO
            },
        ];
        let normal = -DVec3::Y;
        let mu = 0.3;
        let lambda = 0.5 + (c % 4) as f64 * 0.1;
        let sliding = c % 2 == 0;
        let k = lambda * normal
            + if sliding {
                mu * lambda * DVec3::X
            } else {
                0.1 * mu * lambda * DVec3::Z
            };
        let soil = DVec3::new(if sliding { -0.2 } else { 0.3 }, 0.0, 0.0);
        let mask = if c % 3 == 0 { 4 | 32 } else { 4 };
        let wall = DVec3::new(0.0, lambda + 0.25, if mask & 32 != 0 { -0.25 } else { 0.0 });
        let node_mass = 0.5 + (c % 3) as f64 * 0.2;
        f.contacts.push(SoilSharedWallContact {
            soil_mass: pack(soil - (k + wall) / node_mass, node_mass),
            normal_gap: pack(normal, 0.025),
            friction: [mu as f32, 0.0, 0.0, 0.0],
            boundary: [mask, 0, 0, 0],
        });
        for i in 0..n {
            f.rows.push(pack(columns[i], 0.0));
            reaction[i] += columns[i].dot(k);
        }
    }
    let mut matrix = f.mass.iter().map(|&v| v as f64).collect::<Vec<_>>();
    for k in 0..n {
        for i in k + 1..n {
            let ratio = matrix[i * n + k] / matrix[k * n + k];
            for j in k..n {
                matrix[i * n + j] -= ratio * matrix[k * n + j];
            }
            reaction[i] -= ratio * reaction[k];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            reaction[i] -= matrix[i * n + j] * reaction[j];
        }
        reaction[i] /= matrix[i * n + i];
        f.free[i] = (target[i] + reaction[i]) as f32;
    }
    f
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_segmented_paired_timing() {
    paired_schedule_timing(0);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_register_rounding_paired_timing() {
    paired_schedule_timing(1);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_direct_axes_paired_timing() {
    paired_schedule_timing(2);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_warm_start_sweep_limit_rejection() {
    paired_schedule_timing(3);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_cached_setup_paired_timing() {
    paired_schedule_timing(4);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_unrolled_linear_paired_timing() {
    paired_schedule_timing(5);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_cooperative_paired_timing() {
    paired_schedule_timing(6);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_local_cooperative_paired_timing() {
    paired_schedule_timing(7);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_local_cooperative_sparse_timing() {
    paired_schedule_timing(8);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_ordered_sparse_timing() {
    paired_schedule_timing(9);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_ordered_compact_timing() {
    paired_schedule_timing(10);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_chunked_sparse_timing() {
    paired_schedule_timing(11);
}

#[test]
#[ignore = "requires GPU timestamps"]
fn shared_wall_chunked_compact_timing() {
    paired_schedule_timing(12);
}

fn conditional_input(
    engine: &Engine,
    program: &ExperimentalSoilSharedWallFrictionGpu,
    contact: u32,
) -> crate::soil_wall_friction_gpu::SoilWallFrictionInput {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("conditional failure input"),
        size: 112,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(
        program.conditional_inputs_buffer(),
        u64::from(contact) * 112,
        &staging,
        0,
        112,
    );
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(15)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(15))
        .unwrap()
        .unwrap();
    let input = bytemuck::pod_read_unaligned(&staging.slice(..).get_mapped_range());
    staging.unmap();
    input
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_register_rounding_matches_compensated_control() {
    compare_compensated_variants(false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_cached_setup_matches_compensated_control() {
    compare_compensated_variants(true);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_cached_setup_sparse_capacity() {
    sparse_capacity(false, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_cooperative_sparse_capacity() {
    sparse_capacity(true, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_local_cooperative_sparse_capacity() {
    sparse_capacity(true, true);
}

fn spread_contacts(f: &Fixture, capacity: usize, stride: usize) -> Fixture {
    let n = f.free.len();
    let mut sparse = Fixture {
        mass: f.mass.clone(),
        free: f.free.clone(),
        rows: vec![[0.0; 4]; capacity * n],
        contacts: vec![SoilSharedWallContact::default(); capacity],
    };
    for i in 0..f.contacts.len() {
        let slot = i * stride;
        sparse.contacts[slot] = f.contacts[i];
        sparse.rows[slot * n..(slot + 1) * n].copy_from_slice(&f.rows[i * n..(i + 1) * n]);
    }
    sparse
}

fn sparse_capacity(cooperative: bool, local_parallel: bool) {
    let engine = Engine::new_gpu().unwrap();
    let f = articulated(65);
    let reference = create_cached(&engine, &f, 64, [true, true, true, false, false], false);
    let expected = run(&engine, &reference, 8, 65, true);
    assert_eq!(expected.2[0], 0);
    verify(&f, &expected.0, &expected.1);
    let sparse = spread_contacts(&f, 4096, 63);
    let solver = if cooperative {
        create_local_cooperative(
            &engine,
            &sparse,
            64,
            [true, true, true, false, false],
            [true, true, true, local_parallel],
        )
    } else {
        create(&engine, &sparse, 64)
    };
    assert_eq!(solver.owned_buffer_bytes(), 2_687_712);
    let actual = run(&engine, &solver, 8, 4096, true);
    assert_eq!(actual.2, expected.2);
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&actual.0),
        bytemuck::cast_slice::<_, u32>(&expected.0)
    );
    for slot in 0..4096 {
        if slot % 63 == 0 && slot / 63 < 65 {
            assert_eq!(
                bytemuck::bytes_of(&actual.1[slot]),
                bytemuck::bytes_of(&expected.1[slot / 63])
            );
        } else {
            assert_eq!(actual.1[slot], SoilWallFrictionOutput::default());
        }
    }
    println!(
        "4096 slots, 65 noncontiguous contacts, exact outputs/status; status={:?}",
        actual.2
    );
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_cached_setup_refresh_and_budget() {
    refresh_and_budget(false, false, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_cooperative_refresh_and_budget() {
    refresh_and_budget(true, false, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_local_cooperative_refresh_and_budget() {
    refresh_and_budget(true, true, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_ordered_capacity_extremes_and_inactive_validation() {
    let engine = Engine::new_gpu().unwrap();
    let compact = pair(true, 0.3);
    let expected = run(&engine, &create(&engine, &compact, 64), 3, 2, true);
    assert_eq!(expected.2[0], 0);
    verify(&compact, &expected.0, &expected.1);
    let initial = spread_contacts(&compact, 16384, 16383);
    let mass = buffer(&engine, &initial.mass, 4);
    let free = buffer(&engine, &initial.free, 4);
    let rows = buffer(&engine, &initial.rows, 16);
    let contacts = buffer(&engine, &initial.contacts, 64);
    let cap = SoilSharedWallCapacity {
        dofs: 3,
        contacts: 16384,
        max_sweeps: 64,
        local_iterations: 64,
        max_owned_bytes: 9_437_680,
    };
    let make = |budget| {
        ExperimentalSoilSharedWallFrictionGpu::with_ordered(
            &engine,
            SoilSharedWallInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                contacts: &contacts,
            },
            SoilSharedWallCapacity {
                max_owned_bytes: budget,
                ..cap
            },
            [true, true, true, false, false],
            [true, true, true, false, true],
        )
    };
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(make(cap.max_owned_bytes - 1).is_err());
    assert_eq!(scope.finish().observations, 0);
    let solver = make(cap.max_owned_bytes).unwrap();
    assert_eq!(solver.owned_buffer_bytes(), cap.max_owned_bytes);
    let upload = |f: &Fixture| {
        engine
            .queue()
            .unwrap()
            .write_buffer(&rows, 0, bytemuck::cast_slice(&f.rows));
        engine
            .queue()
            .unwrap()
            .write_buffer(&contacts, 0, bytemuck::cast_slice(&f.contacts));
    };
    for phase in 0..4 {
        let mut f = spread_contacts(&compact, 16384, 16383);
        let positions = if phase == 2 {
            [16382, 16383]
        } else {
            [0, 16383]
        };
        if phase == 1 {
            f.contacts.fill(SoilSharedWallContact::default());
        } else if phase == 2 {
            f.contacts[16382] = f.contacts[0];
            f.contacts[0] = SoilSharedWallContact::default();
            f.rows.copy_within(0..3, 16382 * 3);
            f.rows[..3].fill([0.0; 4]);
        }
        upload(&f);
        let result = run(&engine, &solver, 3, 16384, true);
        assert_eq!(result.2[0], 0);
        if phase == 1 {
            assert_eq!(result.0, compact.free);
            assert!(
                result
                    .1
                    .iter()
                    .all(|v| *v == SoilWallFrictionOutput::default())
            );
        } else {
            assert_eq!(result.2, expected.2);
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&result.0),
                bytemuck::cast_slice::<_, u32>(&expected.0)
            );
            for (slot, output) in result.1.iter().enumerate() {
                if let Some(index) = positions.iter().position(|&i| i == slot) {
                    assert_eq!(
                        bytemuck::bytes_of(output),
                        bytemuck::bytes_of(&expected.1[index]),
                        "phase={phase} slot={slot}"
                    );
                } else {
                    assert_eq!(
                        *output,
                        SoilWallFrictionOutput::default(),
                        "phase={phase} slot={slot}"
                    );
                }
            }
        }
        println!("ordered capacity=16384 phase={phase} status={:?}", result.2);
    }
    for kind in 0..3 {
        let mut bad = spread_contacts(&compact, 16384, 16383);
        match kind {
            0 => bad.contacts[8192].friction[0] = f32::NAN,
            1 => bad.rows[8192 * 3][0] = f32::NAN,
            _ => bad.contacts[8192].soil_mass[3] = -1.0,
        }
        upload(&bad);
        let rejected = run(&engine, &solver, 3, 16384, true);
        assert_eq!(rejected.2[0], SHARED_WALL_INPUT, "inactive kind={kind}");
        assert_eq!(rejected.2[3], 8193);
        upload(&initial);
        assert_eq!(run(&engine, &solver, 3, 16384, false), rejected);
        assert_eq!(run(&engine, &solver, 3, 16384, true).2, expected.2);
    }
    let empty = Fixture {
        mass: compact.mass,
        free: compact.free,
        rows: vec![],
        contacts: vec![],
    };
    let solver = create_ordered(
        &engine,
        &empty,
        64,
        [true, true, true, false, false],
        [true, true, true, false, true],
    );
    let result = run(&engine, &solver, 3, 0, true);
    assert_eq!(result.2[0], 0);
    assert_eq!(result.0, empty.free);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_ordered_refresh_and_budget() {
    refresh_and_budget(true, false, true);
}

fn refresh_and_budget(cooperative: bool, local_parallel: bool, ordered: bool) {
    refresh_chunked_and_budget(cooperative, local_parallel, ordered, 0);
}

fn refresh_chunked_and_budget(cooperative: bool, local_parallel: bool, ordered: bool, chunk: u32) {
    refresh_preparation_and_budget(cooperative, local_parallel, ordered, chunk, false);
}

fn refresh_preparation_and_budget(
    cooperative: bool,
    local_parallel: bool,
    ordered: bool,
    chunk: u32,
    split: bool,
) {
    let engine = Engine::new_gpu().unwrap();
    let original = pair(true, 0.3);
    let mass = buffer(&engine, &original.mass, 4);
    let free = buffer(&engine, &original.free, 4);
    let rows = buffer(&engine, &original.rows, 16);
    let contacts = buffer(&engine, &original.contacts, 64);
    let make = |cached, budget| {
        ExperimentalSoilSharedWallFrictionGpu::with_preparation(
            &engine,
            SoilSharedWallInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                contacts: &contacts,
            },
            SoilSharedWallCapacity {
                dofs: 3,
                contacts: 2,
                max_sweeps: 64,
                local_iterations: 64,
                max_owned_bytes: budget,
            },
            [true, true, true, false, false],
            [
                cached,
                cached && cooperative,
                cached && cooperative,
                cached && local_parallel,
                cached && ordered,
            ],
            if cached { chunk } else { 0 },
            cached && split,
        )
    };
    let budget = 1648
        + if chunk == 0 { 0 } else { 256 }
        + if split && redistribution_selected() {
            32
        } else {
            0
        };
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(make(true, budget - 1).is_err());
    assert_eq!(allocations.finish().observations, 0);
    let cached = make(true, budget).unwrap();
    assert_eq!(cached.owned_buffer_bytes(), budget);
    let reference = make(false, 1104).unwrap();
    let queue = engine.queue().unwrap();
    let upload = |f: &Fixture| {
        queue.write_buffer(&mass, 0, bytemuck::cast_slice(&f.mass));
        queue.write_buffer(&free, 0, bytemuck::cast_slice(&f.free));
        queue.write_buffer(&rows, 0, bytemuck::cast_slice(&f.rows));
        queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&f.contacts));
    };
    for phase in 0..6 {
        let mut f = pair(phase == 0 || phase == 5, if phase == 1 { 0.0 } else { 0.3 });
        if phase == 2 || phase == 3 {
            let rotation = glam::DQuat::from_rotation_z(0.37);
            for row in &mut f.rows {
                *row = pack(rotation * vec(*row), 0.0);
            }
            for node in &mut f.contacts {
                node.soil_mass = pack(rotation * vec(node.soil_mass), node.soil_mass[3] as f64);
                node.normal_gap = pack(rotation * vec(node.normal_gap), node.normal_gap[3] as f64);
            }
        }
        if phase == 3 {
            for i in 0..3 {
                f.mass[i * 3 + i] *= 2.0;
            }
        }
        if phase == 4 {
            f.contacts.fill(SoilSharedWallContact::default());
        }
        upload(&f);
        let a = run(&engine, &reference, 3, 2, true);
        let b = run(&engine, &cached, 3, 2, true);
        assert_eq!(a.2, b.2, "phase={phase}");
        assert_eq!(b.2[0], 0);
        assert_eq!(
            bytemuck::cast_slice::<_, u32>(&a.0),
            bytemuck::cast_slice::<_, u32>(&b.0)
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u32>(&a.1),
            bytemuck::cast_slice::<_, u32>(&b.1)
        );
        if phase == 4 {
            assert_eq!(b.0, f.free);
        } else {
            verify(&f, &b.0, &b.1);
        }
        println!("cached setup refresh phase={phase} status={:?}", b.2);
    }
    let mut bad = pair(true, 0.3);
    bad.contacts[0].boundary[0] = 3;
    upload(&bad);
    let rejected = run(&engine, &cached, 3, 2, true);
    assert_eq!(rejected.2[0], SHARED_WALL_LOCAL);
    assert_eq!(rejected.2[2], 1);
    assert_eq!(rejected.2[1], 0);
    upload(&original);
    assert_eq!(run(&engine, &cached, 3, 2, false), rejected);
    let restored = run(&engine, &cached, 3, 2, true);
    assert_eq!(restored.2[0], 0);
    verify(&original, &restored.0, &restored.1);
}

fn compare_compensated_variants(cache_setup: bool) {
    compare_linear_variants(cache_setup, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_unrolled_linear_matches_reference() {
    compare_linear_variants(true, true);
}

fn compare_linear_variants(cache_setup: bool, unroll_linear: bool) {
    compare_cooperative_variants(cache_setup, unroll_linear, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_cooperative_matches_reference() {
    compare_cooperative_variants(true, true, true);
}

fn compare_cooperative_variants(cache_setup: bool, unroll_linear: bool, cooperative: bool) {
    compare_local_variants(cache_setup, unroll_linear, cooperative, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_local_cooperative_matches_reference() {
    compare_local_variants(true, true, true, true);
}

fn compare_local_variants(
    cache_setup: bool,
    unroll_linear: bool,
    cooperative: bool,
    local_parallel: bool,
) {
    compare_variant_flags(
        [unroll_linear, cooperative, local_parallel, false, false],
        [
            cache_setup,
            unroll_linear,
            cooperative,
            local_parallel,
            false,
        ],
        cache_setup,
    );
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_ordered_matches_reference() {
    compare_variant_flags(
        [true, true, true, false, false],
        [true, true, true, false, true],
        true,
    );
}

fn compare_variant_flags(
    reference_flags: [bool; 5],
    candidate_flags: [bool; 5],
    reference_rounding: bool,
) {
    compare_chunked_flags(reference_flags, candidate_flags, reference_rounding, 0);
}

fn compare_chunked_flags(
    reference_flags: [bool; 5],
    candidate_flags: [bool; 5],
    reference_rounding: bool,
    chunk: u32,
) {
    compare_preparation_flags(
        reference_flags,
        candidate_flags,
        reference_rounding,
        chunk,
        false,
    );
}

fn wide_fixture() -> Fixture {
    let mut wide = articulated(7);
    let mut mass = vec![0.0; 32 * 32];
    for i in 0..32 {
        mass[i * 32 + i] = (i + 2) as f32;
        for j in 0..8 {
            if i < 8 {
                mass[i * 32 + j] = wide.mass[i * 8 + j];
            }
        }
    }
    let mut rows = Vec::new();
    for c in 0..7 {
        rows.extend_from_slice(&wide.rows[c * 8..(c + 1) * 8]);
        for i in 8..32 {
            let mut row = [0.0; 4];
            row[(i + c) % 3] = 0.0001 * (i + 1) as f32;
            rows.push(row);
        }
    }
    wide.mass = mass;
    wide.rows = rows;
    wide.free.resize(32, 0.0);
    wide
}

#[test]
#[ignore = "requires GPU and fresh SOIL_WALL_STAGE_TRACE; 32-DOF convergence diagnostic"]
fn shared_wall_wide_convergence_probe() {
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_WALL_STAGE_TRACE").expect("fresh trace directory required"),
    );
    let engine = std::sync::Arc::new(Engine::new_gpu().unwrap());
    let f = wide_fixture();
    let mut solver = create_preparation(
        &engine,
        &f,
        64,
        [true, true, true, false, false],
        [true, true, true, false, true],
        0,
        true,
    );
    solver.trace_stages(engine.clone(), root.clone());
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    encoder.clear_buffer(&solver.status, 0, None);
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    let status = std::fs::read(root.join("solve-000/status.bin")).unwrap();
    let fault = u32::from_ne_bytes(status[..4].try_into().unwrap());
    println!("32-DOF probe fault={fault}; diagnostic capture, not acceptance");
    assert!(fault == 0 || fault == SHARED_WALL_SWEEPS);
}

#[test]
#[ignore = "requires GPU; local early-exit and bridge normal feasibility"]
fn shared_wall_split_small_normal_violation() {
    let engine = Engine::new_gpu().unwrap();
    let f = Fixture {
        mass: vec![1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1000.0],
        free: vec![-0.13, -0.086, 1e-7],
        rows: vec![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        contacts: vec![SoilSharedWallContact {
            soil_mass: [-0.002, -0.068, 1.2e-6, 0.03],
            normal_gap: [0.0, 0.0, -1.0, 0.0],
            friction: [0.3, 0.0, 0.0, 0.0],
            boundary: [0; 4],
        }],
    };
    let solver = create_preparation(
        &engine,
        &f,
        4,
        [true, true, true, false, false],
        [true, true, true, false, true],
        0,
        true,
    );
    let result = run(&engine, &solver, 3, 1, true);
    assert_eq!(result.2[0], 0);
    assert!(result.1[0].machine_impulse_normal[2] > 0.0);
    assert!(result.0[2] - result.1[0].soil_velocity_energy[2] >= -1e-6);
    verify_tolerance(&f, &result.0, &result.1, 1e-6);
    assert_eq!(run(&engine, &solver, 3, 1, true), result);
    println!(
        "small normal violation receives nonzero reaction: {:?}",
        result.2
    );
}

#[test]
#[ignore = "requires GPU; 32-DOF shared and bridge-normal certification"]
fn shared_wall_split_wide_certification() {
    let engine = Engine::new_gpu().unwrap();
    let f = wide_fixture();
    let solver = create_preparation(
        &engine,
        &f,
        64,
        [true, true, true, false, false],
        [true, true, true, false, true],
        0,
        true,
    );
    let result = run(&engine, &solver, 32, 7, true);
    assert_eq!(result.2[0], 0);
    verify_tolerance(&f, &result.0, &result.1, 1e-6);
    for (c, output) in result.1.iter().enumerate() {
        let node = f.contacts[c];
        let normal = vec(node.normal_gap).normalize();
        let soil = normal.dot(vec(output.soil_velocity_energy));
        let machine = (0..32)
            .map(|i| normal.dot(vec(f.rows[c * 32 + i])) * result.0[i] as f64)
            .sum::<f64>();
        let gap = node.normal_gap[3] as f64;
        let gate = 1e-6 * 1.0_f64.max(soil.abs()).max(machine.abs()).max(gap);
        assert!(soil - machine + gap >= -gate, "normal separation slot {c}");
    }
    assert_eq!(run(&engine, &solver, 32, 7, true), result);
    println!(
        "32-DOF strict f64 certificate and repeatability: {:?}",
        result.2
    );
}

fn compare_preparation_flags(
    reference_flags: [bool; 5],
    candidate_flags: [bool; 5],
    reference_rounding: bool,
    chunk: u32,
    split: bool,
) {
    let engine = Engine::new_gpu().unwrap();
    for f in [
        pair(false, 0.0),
        pair(false, 0.3),
        pair(true, 0.0),
        pair(true, 0.3),
        articulated(7),
        articulated(65),
        wide_fixture(),
    ] {
        let n = f.free.len();
        let c = f.contacts.len();
        for sweeps in [1, 64] {
            let reference = create_ordered(
                &engine,
                &f,
                sweeps,
                [true, true, reference_rounding, false, false],
                reference_flags,
            );
            let candidate = create_preparation(
                &engine,
                &f,
                sweeps,
                [true, true, true, false, false],
                candidate_flags,
                chunk,
                split,
            );
            let a = run(&engine, &reference, n, c, true);
            let b = run(&engine, &candidate, n, c, true);
            assert_eq!(a.2, b.2, "dofs={n} contacts={c} limit={sweeps}");
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&a.0),
                bytemuck::cast_slice::<_, u32>(&b.0)
            );
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&a.1),
                bytemuck::cast_slice::<_, u32>(&b.1)
            );
            if sweeps == 64 {
                assert_eq!(b.2[0], 0, "dofs={n} contacts={c}");
                verify(&f, &b.0, &b.1);
            } else {
                assert_eq!(b.2[0], SHARED_WALL_SWEEPS);
            }
            println!(
                "reference={reference_flags:?} candidate={candidate_flags:?} exact dofs={n} contacts={c} limit={sweeps} status={:?}",
                b.2
            );
        }
    }
}

fn paired_schedule_timing(mode: u32) {
    let engine = Engine::new_gpu_with_features(wgpu::Features::TIMESTAMP_QUERY).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    assert!(device.features().contains(wgpu::Features::TIMESTAMP_QUERY));
    let f = articulated(65);
    let f = if mode == 8 || mode == 9 || mode == 11 {
        spread_contacts(&f, 4096, 63)
    } else {
        f
    };
    println!("paired mode={mode} capacity={} dofs=8", f.contacts.len());
    let programs = [
        create_ordered(
            &engine,
            &f,
            64,
            [true, mode != 0, mode >= 2, false, false],
            [mode >= 5, mode >= 6, mode >= 7, false, mode >= 11],
        ),
        create_chunks(
            &engine,
            &f,
            64,
            [true, true, mode != 0, mode == 2, mode == 3],
            [
                mode >= 4,
                mode >= 5,
                mode >= 6,
                mode == 7 || mode == 8,
                mode >= 9,
            ],
            if mode >= 11 { 64 } else { 0 },
        ),
    ];
    let expected = programs
        .each_ref()
        .map(|program| run(&engine, program, 8, f.contacts.len(), true));
    if mode == 3 && expected[1].2[0] != 0 && expected[1].2[3] > 0 {
        let index = expected[1].2[3] - 1;
        let input = conditional_input(&engine, &programs[1], index);
        let seed = expected[1].1[index as usize];
        println!("warm failure conditional input={input:?}; prior seed={seed:?}");
        println!(
            "warm failure input words={:?}; seed words={:?}",
            bytemuck::cast_slice::<_, u32>(std::slice::from_ref(&input)),
            bytemuck::cast_slice::<_, u32>(std::slice::from_ref(&seed))
        );
    }
    if mode == 3 {
        assert_eq!(expected[0].2[0], 0);
        verify(&f, &expected[0].0, &expected[0].1);
        assert_eq!(expected[1].2[0], SHARED_WALL_SWEEPS);
        assert_eq!(expected[1].2[1], 64);
        println!(
            "warm-start rejected: cold={:?}; warm={:?}",
            expected[0].2, expected[1].2
        );
        return;
    }
    for result in &expected {
        assert_eq!(result.2[0], 0, "mode={mode} status={:?}", result.2);
        verify(&f, &result.0, &result.1);
    }
    assert_eq!(expected[0], expected[1]);
    let count = 2 * programs[1].passes().count() as u32;
    let bytes = 8 * u64::from(count);
    let queries: Vec<_> = (0..count)
        .step_by(wgpu::QUERY_SET_MAX_QUERIES as usize)
        .map(|start| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("shared sweep paired timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: (count - start).min(wgpu::QUERY_SET_MAX_QUERIES),
            })
        })
        .collect();
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes + 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut samples = [Vec::new(), Vec::new()];
    for pair in 0..8 {
        for variant in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
            let program = &programs[variant];
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.clear_buffer(&program.status, 0, None);
            let mut used = 0;
            for (pipeline, groups) in program.passes() {
                let set = &queries[(used / wgpu::QUERY_SET_MAX_QUERIES) as usize];
                let index = used % wgpu::QUERY_SET_MAX_QUERIES;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: set,
                        beginning_of_pass_write_index: Some(index),
                        end_of_pass_write_index: Some(index + 1),
                    }),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &program.group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
                used += 2;
            }
            for (set, start) in queries
                .iter()
                .zip((0..used).step_by(wgpu::QUERY_SET_MAX_QUERIES as usize))
            {
                encoder.resolve_query_set(
                    set,
                    0..(used - start).min(wgpu::QUERY_SET_MAX_QUERIES),
                    &resolved,
                    8 * u64::from(start),
                );
            }
            encoder.copy_buffer_to_buffer(&resolved, 0, &staging, 0, 8 * u64::from(used));
            encoder.copy_buffer_to_buffer(&program.status, 0, &staging, bytes, 32);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(15)),
                })
                .unwrap();
            rx.recv_timeout(std::time::Duration::from_secs(15))
                .unwrap()
                .unwrap();
            let mapped = staging.slice(..).get_mapped_range();
            let status = bytemuck::pod_read_unaligned::<[u32; 8]>(&mapped[bytes as usize..]);
            assert_eq!(status, expected[variant].2);
            let ticks: &[u64] = bytemuck::cast_slice(&mapped[..8 * used as usize]);
            let scale = f64::from(queue.get_timestamp_period()) / 1e6;
            let span = ticks[ticks.len() - 1].checked_sub(ticks[0]).unwrap() as f64 * scale;
            let longest = ticks
                .chunks_exact(2)
                .map(|v| v[1].checked_sub(v[0]).unwrap() as f64 * scale)
                .fold(0.0_f64, f64::max);
            drop(mapped);
            staging.unmap();
            println!(
                "sweep schedule pair={pair} variant={variant} span_ms={span:.6} longest_pass_ms={longest:.6}"
            );
            if pair >= 2 {
                samples[variant].push((span, longest));
            }
        }
    }
    for (variant, values) in samples.iter().enumerate() {
        let mut spans: Vec<_> = values.iter().map(|v| v.0).collect();
        let mut longest: Vec<_> = values.iter().map(|v| v.1).collect();
        spans.sort_by(f64::total_cmp);
        longest.sort_by(f64::total_cmp);
        println!(
            "sweep schedule variant={variant} measured_pairs=6 median_span_ms={:.6} median_longest_pass_ms={:.6}",
            spans[2], longest[2]
        );
        assert_eq!(
            run(&engine, &programs[variant], 8, f.contacts.len(), true),
            expected[variant]
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_segmented_matches_monolithic() {
    let engine = Engine::new_gpu().unwrap();
    for (f, sweeps, accelerate) in [
        (pair(true, 0.3), 64, true),
        (articulated(7), 64, true),
        (articulated(65), 64, true),
        (articulated(7), 64, false),
        (pair(true, 0.3), 1, true),
    ] {
        let n = f.free.len();
        let c = f.contacts.len();
        let segmented = create_scheduled(&engine, &f, sweeps, accelerate, true);
        let monolithic = create_scheduled(&engine, &f, sweeps, accelerate, false);
        let expected = run(&engine, &monolithic, n, c, true);
        let actual = run(&engine, &segmented, n, c, true);
        assert_eq!(
            actual, expected,
            "contacts={c} sweeps={sweeps} accelerate={accelerate}"
        );
        if actual.2[0] == 0 {
            verify(&f, &actual.0, &actual.1);
        }
        assert_eq!(run(&engine, &segmented, n, c, true), actual);
        println!(
            "exact segmented control contacts={c} max_sweeps={sweeps} acceleration={accelerate}: {:?}",
            actual.2
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_articulated_generalized_response() {
    let engine = Engine::new_gpu().unwrap();
    for count in [7, 65] {
        let f = articulated(count);
        let solver = create(&engine, &f, 256);
        let result = run(&engine, &solver, 8, count, true);
        assert_eq!(result.2[0], 0, "count={count}: {:?}", result.2);
        verify(&f, &result.0, &result.1);
        assert_eq!(run(&engine, &solver, 8, count, true), result);
        println!("eight-DOF contacts={count}, sweeps={}", result.2[1]);
    }
}

#[test]
#[ignore = "requires GPU; focused convergence diagnostic"]
fn shared_wall_convergence_diagnostic() {
    let engine = Engine::new_gpu().unwrap();
    let f = articulated(7);
    let solver = create(&engine, &f, 256);
    let (rates, output, status) = run(&engine, &solver, 8, 7, true);
    let n = 8;
    let mut a = f.mass.iter().map(|&v| v as f64).collect::<Vec<_>>();
    let mut reaction = (0..n)
        .map(|i| {
            output
                .iter()
                .enumerate()
                .map(|(c, o)| vec(f.rows[c * n + i]).dot(vec(o.machine_impulse_normal)))
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    for k in 0..n {
        for i in k + 1..n {
            let ratio = a[i * n + k] / a[k * n + k];
            for j in k..n {
                a[i * n + j] -= ratio * a[k * n + j];
            }
            reaction[i] -= ratio * reaction[k];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            reaction[i] -= a[i * n + j] * reaction[j];
        }
        reaction[i] /= a[i * n + i];
    }
    let q = (0..n)
        .map(|i| f.free[i] as f64 + reaction[i])
        .collect::<Vec<_>>();
    println!("status={status:?}; reconstructed q={q:?}");
    for (c, o) in output.iter().enumerate() {
        let node = f.contacts[c];
        let k = -vec(o.machine_impulse_normal);
        let v = (0..n).map(|i| vec(f.rows[c * n + i]) * q[i]).sum::<DVec3>();
        let u = vec(node.soil_mass) + (k + vec(o.wall_impulse)) / node.soil_mass[3] as f64;
        let normal = vec(node.normal_gap).normalize();
        let lambda = k.dot(normal);
        let slip = u - v - (u - v).dot(normal) * normal;
        let tangent = k - lambda * normal;
        println!(
            "c={c}: k={k:?} u={u:?} v={v:?} separation={} slip={} friction_residual={} local={:?}",
            (u - v).dot(normal) + node.normal_gap[3] as f64,
            slip.length(),
            (tangent + node.friction[0] as f64 * lambda * slip.normalize_or_zero()).length(),
            o.status
        );
    }
    assert_eq!(status[0], 0);
    verify(&f, &rates, &output);
    assert!(status[4] > 0);
    assert!(status[5] > 0);
    assert_eq!(status[4] + status[5], status[6]);
    let control = create_mode(&engine, &f, 256, false);
    let old = run(&engine, &control, 8, 7, true);
    assert_eq!(old.2[0], SHARED_WALL_SWEEPS);
    assert_eq!(old.2[1], 256);
    println!(
        "matched control status={:?}; accelerated status={status:?}",
        old.2
    );
}
#[test]
#[ignore = "requires GPU"]
fn shared_wall_pair_conservation_and_analytic_solution() {
    let engine = Engine::new_gpu().unwrap();
    for walls in [false, true] {
        for mu in [0.0, 0.3] {
            let f = pair(walls, mu);
            let solver = create(&engine, &f, 128);
            let result = run(&engine, &solver, 3, 2, true);
            assert_eq!(result.2[0], 0, "walls={walls} mu={mu}: {:?}", result.2);
            verify(&f, &result.0, &result.1);
            let expected = if mu > 0.0 {
                if walls {
                    0.5 * DVec3::new(1.0, 1.0, 0.0).normalize()
                } else {
                    0.5 * DVec3::X
                }
            } else if walls {
                DVec3::new(-0.2, 0.2, 0.0)
            } else {
                DVec3::Y / 3.0
            };
            assert!(
                (DVec3::new(result.0[0] as f64, result.0[1] as f64, result.0[2] as f64) - expected)
                    .length()
                    <= 2e-5
            );
            assert_eq!(run(&engine, &solver, 3, 2, true), result);
            println!(
                "walls={walls} mu={mu}: sweeps={} rates={:?}",
                result.2[1], result.0
            );
        }
    }
}
#[test]
#[ignore = "requires GPU"]
fn shared_wall_faults_and_empty() {
    faults_and_empty(false, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_cooperative_faults_and_empty() {
    faults_and_empty(true, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_local_cooperative_faults_and_empty() {
    faults_and_empty(true, true);
}

fn faults_and_empty(cooperative: bool, local_parallel: bool) {
    let engine = Engine::new_gpu().unwrap();
    let make = |f: &Fixture, sweeps| {
        if cooperative {
            create_local_cooperative(
                &engine,
                f,
                sweeps,
                [true, true, true, false, false],
                [true, true, true, local_parallel],
            )
        } else {
            create(&engine, f, sweeps)
        }
    };
    let f = pair(true, 0.3);
    let short = make(&f, 1);
    let failed = run(&engine, &short, 3, 2, true);
    assert_eq!(failed.2[0], 16);
    assert_eq!(run(&engine, &short, 3, 2, false), failed);
    for kind in 0..4 {
        let mut bad = pair(true, 0.3);
        match kind {
            0 => bad.mass[0] = -1.0,
            1 => bad.mass[1] = 0.1,
            2 => bad.rows[0][0] = f32::NAN,
            _ => bad.contacts[0].boundary[0] = 3,
        }
        let solver = make(&bad, 128);
        let output = run(&engine, &solver, 3, 2, true);
        assert_ne!(output.2[0], 0, "kind={kind}");
        assert_eq!(run(&engine, &solver, 3, 2, false), output);
    }
    let empty = Fixture {
        mass: f.mass,
        free: f.free,
        rows: vec![],
        contacts: vec![],
    };
    let solver = make(&empty, 128);
    let output = run(&engine, &solver, 3, 0, true);
    assert_eq!(output.2[0], 0);
    assert_eq!(output.0, empty.free);
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_acceleration_reuse_and_budget() {
    let engine = Engine::new_gpu().unwrap();
    let f = articulated(65);
    let mass = buffer(&engine, &f.mass, 4);
    let free = buffer(&engine, &f.free, 4);
    let rows = buffer(&engine, &f.rows, 16);
    let contacts = buffer(&engine, &f.contacts, 64);
    let cap = SoilSharedWallCapacity {
        dofs: 8,
        contacts: 65,
        max_sweeps: 256,
        local_iterations: 64,
        max_owned_bytes: u64::MAX,
    };
    let required = ExperimentalSoilSharedWallFrictionGpu::required_bytes(cap).unwrap();
    let create = |capacity| {
        ExperimentalSoilSharedWallFrictionGpu::new(
            &engine,
            SoilSharedWallInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                contacts: &contacts,
            },
            capacity,
        )
    };
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        create(SoilSharedWallCapacity {
            max_owned_bytes: required - 1,
            ..cap
        })
        .is_err()
    );
    assert!(create(SoilSharedWallCapacity { dofs: 33, ..cap }).is_err());
    assert_eq!(scope.finish().observations, 0);
    let solver = create(SoilSharedWallCapacity {
        max_owned_bytes: required,
        ..cap
    })
    .unwrap();
    assert_eq!(solver.owned_buffer_bytes(), required);
    let reference = run(&engine, &solver, 8, 65, true);
    assert_eq!(reference.2[0], 0);
    verify(&f, &reference.0, &reference.1);
    let queue = engine.queue().unwrap();
    let mut changed = vec![SoilSharedWallContact::default(); 65];
    queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&changed));
    let empty = run(&engine, &solver, 8, 65, true);
    assert_eq!(empty.2[0], 0);
    assert_eq!(empty.0, f.free);
    assert!(
        empty
            .1
            .iter()
            .all(|v| *v == SoilWallFrictionOutput::default())
    );
    let small = articulated(7);
    changed[..7].copy_from_slice(&small.contacts);
    queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&changed));
    queue.write_buffer(&free, 0, bytemuck::cast_slice(&small.free));
    let output = run(&engine, &solver, 8, 65, true);
    assert_eq!(output.2[0], 0);
    verify(&small, &output.0, &output.1[..7]);
    assert!(
        output.1[7..]
            .iter()
            .all(|v| *v == SoilWallFrictionOutput::default())
    );
    queue.write_buffer(&free, 0, bytemuck::cast_slice(&f.free));
    queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&f.contacts));
    assert_eq!(run(&engine, &solver, 8, 65, true), reference);
    let mut invalid = f.contacts[0];
    invalid.friction[0] = f32::NAN;
    queue.write_buffer(&contacts, 0, bytemuck::bytes_of(&invalid));
    assert_ne!(run(&engine, &solver, 8, 65, true).2[0], 0);
    queue.write_buffer(&contacts, 0, bytemuck::bytes_of(&f.contacts[0]));
    assert_ne!(run(&engine, &solver, 8, 65, false).2[0], 0);
    assert_eq!(run(&engine, &solver, 8, 65, true), reference);
    println!("65/0/7/65 contact reuse and sticky/reset pass; retained bytes={required}");
}
