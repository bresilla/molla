use super::*;
use crate::soil_floor_projection_gpu::{
    ExperimentalSoilFloorProjectionGpu, SoilFloorCapacity, SoilFloorInputs,
};

fn buffer(engine: &Engine, bytes: &[u8], minimum: u64) -> wgpu::Buffer {
    let result = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact geometry fixture"),
            size: (bytes.len() as u64).max(minimum),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    if !bytes.is_empty() {
        engine.queue().unwrap().write_buffer(&result, 0, bytes);
    }
    result
}

fn read(engine: &Engine, source: &wgpu::Buffer) -> Vec<f32> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("geometry observation"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
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
    let result = bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    result
}

struct Fixture {
    buffers: [wgpu::Buffer; 6],
    geometry: ExperimentalSoilContactGeometryGpu,
    floor: ExperimentalSoilFloorProjectionGpu,
}
impl Fixture {
    fn new(
        engine: &Engine,
        centers: &[[f32; 4]],
        shapes: &[[f32; 4]],
        jacobians: &[f32],
        queries: &[SoilContactQuery],
        free: &[f32],
    ) -> Self {
        let n = free.len();
        let count = queries.len().max(1);
        let buffers = [
            buffer(engine, bytemuck::cast_slice(centers), 16),
            buffer(engine, bytemuck::cast_slice(shapes), 64),
            buffer(engine, bytemuck::cast_slice(jacobians), 24 * n as u64),
            buffer(engine, bytemuck::cast_slice(queries), 48),
            buffer(engine, &[], (48 * count) as u64),
            buffer(engine, &[], (4 * count * n) as u64),
        ];
        let mut mass = vec![0.0f32; n * n];
        for i in 0..n {
            mass[i * n + i] = 860.0;
        }
        let mass = buffer(engine, bytemuck::cast_slice(&mass), 4);
        let rates = buffer(engine, bytemuck::cast_slice(free), 4);
        let floor = ExperimentalSoilFloorProjectionGpu::new(
            engine,
            SoilFloorInputs {
                mass: &mass,
                free_rates: &rates,
                rows: &buffers[5],
                nodes: &buffers[4],
            },
            SoilFloorCapacity {
                dofs: n as u32,
                nodes: queries.len() as u32,
                max_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
        )
        .unwrap();
        let geometry = ExperimentalSoilContactGeometryGpu::new(
            engine,
            SoilContactGeometryInputs {
                centers: &buffers[0],
                shapes: &buffers[1],
                jacobians: &buffers[2],
                queries: &buffers[3],
            },
            SoilContactGeometryOutputs {
                nodes: &buffers[4],
                rows: &buffers[5],
                status: floor.status_buffer(),
            },
            SoilContactGeometryCapacity {
                dofs: n as u32,
                bodies: centers.len() as u32,
                queries: queries.len() as u32,
                max_owned_bytes: u64::MAX,
            },
        )
        .unwrap();
        Self {
            buffers,
            geometry,
            floor,
        }
    }
    fn run(&self, engine: &Engine, coupled: bool, reset: bool) -> u32 {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        if reset {
            encoder.clear_buffer(self.floor.status_buffer(), 0, None);
        }
        self.geometry.encode(&mut encoder);
        if coupled {
            self.floor.encode(&mut encoder);
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = transfers.finish();
        assert_eq!(traffic.upload_bytes, 0);
        assert_eq!(traffic.readback_bytes, 0);
        read(engine, self.floor.status_buffer())[0].to_bits()
    }
}

fn rotate(q: [f64; 4], p: [f64; 3]) -> [f64; 3] {
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let v = [q[0], q[1], q[2]];
    let first = cross(v, p);
    let second = cross(v, std::array::from_fn(|i| first[i] + q[3] * p[i]));
    std::array::from_fn(|i| p[i] + 2.0 * second[i])
}

fn nearest(p: [f64; 3], s: [f64; 4]) -> ([f64; 3], f64) {
    let length = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let sign = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
    if s[3] == 1.0 {
        let r = length(p);
        return (
            if r == 0.0 {
                [1.0, 0.0, 0.0]
            } else {
                p.map(|v| v / r)
            },
            r - s[0],
        );
    }
    let inside;
    let mut candidates = Vec::new();
    if s[3] == 2.0 {
        inside = (0..3).all(|i| p[i].abs() <= s[i]);
        for axis in 0..3 {
            let mut point = std::array::from_fn(|i| p[i].clamp(-s[i], s[i]));
            point[axis] = sign(p[axis]) * s[axis];
            let mut normal = [0.0; 3];
            normal[axis] = sign(p[axis]);
            candidates.push((point, normal));
        }
    } else {
        let r = p[0].hypot(p[2]);
        let radial = if r == 0.0 {
            [1.0, 0.0, 0.0]
        } else {
            [p[0] / r, 0.0, p[2] / r]
        };
        inside = r <= s[0] && p[1].abs() <= s[1];
        candidates.push((
            [radial[0] * s[0], p[1].clamp(-s[1], s[1]), radial[2] * s[0]],
            radial,
        ));
        candidates.push((
            [
                radial[0] * r.min(s[0]),
                sign(p[1]) * s[1],
                radial[2] * r.min(s[0]),
            ],
            [0.0, sign(p[1]), 0.0],
        ));
    }
    let mut best = ([0.0; 3], f64::INFINITY);
    for (point, normal) in candidates {
        let delta: [f64; 3] = std::array::from_fn(|i| p[i] - point[i]);
        let distance = length(delta);
        if distance < best.1.abs() {
            let signed = if inside { -distance } else { distance };
            best = (
                if distance == 0.0 {
                    normal
                } else {
                    delta.map(|x| x / signed)
                },
                signed,
            );
        }
    }
    best
}

fn query(p: [f32; 3], body: u32, mask: u32) -> SoilContactQuery {
    SoilContactQuery {
        position_mass: [p[0], p[1], p[2], 0.47464246],
        velocity_dt: [0.0, 0.0, 0.0, 0.01],
        body_boundary: [body, mask, 0, 0],
    }
}

#[test]
fn contact_query_layout_is_three_vec4s() {
    assert_eq!(std::mem::size_of::<SoilContactQuery>(), 48);
}

#[test]
#[ignore = "requires GPU; shared geometry programs with independent state and cache lifetime"]
fn geometry_program_cache_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let shapes = [
        [0.0; 4],
        [0.0; 4],
        [0.0, 0.0, 0.0, 1.0],
        [0.45, 0.45, 0.45, 1.0],
    ];
    let a = Fixture::new(
        &engine,
        &[[0.0; 4]],
        &shapes,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[query([0.5, 0.0, 0.0], 0, 0)],
        &[0.0],
    );
    let b = Fixture::new(
        &engine,
        &[[0.0; 4]],
        &shapes,
        &[0.0; 18],
        &[query([0.0, 0.6, 0.0], 0, 0); 65],
        &[0.0; 3],
    );
    assert!(std::sync::Arc::ptr_eq(
        &a.geometry.program,
        &b.geometry.program
    ));
    let retained = std::sync::Arc::downgrade(&a.geometry.program);
    assert_eq!(a.run(&engine, false, true), 0);
    assert_eq!(b.run(&engine, false, true), 0);
    let before = read(&engine, b.geometry.geometry_buffer());
    assert_eq!(read(&engine, a.geometry.geometry_buffer())[0], 1.0);
    assert_eq!(before[1], 1.0);
    let mut bad = query([0.5, 0.0, 0.0], 0, 0);
    bad.position_mass[0] = f32::NAN;
    engine
        .queue()
        .unwrap()
        .write_buffer(&a.buffers[3], 0, bytemuck::bytes_of(&bad));
    assert_eq!(a.run(&engine, false, true), GEOMETRY_INPUT);
    assert_eq!(b.run(&engine, false, true), 0);
    assert_eq!(
        bytemuck::cast_slice::<f32, u8>(&read(&engine, b.geometry.geometry_buffer())),
        bytemuck::cast_slice::<f32, u8>(&before)
    );
    engine.clear_gpu_program_cache();
    let rebuilt = geometry_program(&engine).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&rebuilt, &b.geometry.program));
    assert_eq!(b.run(&engine, false, true), 0);
    assert_eq!(
        bytemuck::cast_slice::<f32, u8>(&read(&engine, b.geometry.geometry_buffer())),
        bytemuck::cast_slice::<f32, u8>(&before)
    );
    drop(a);
    assert!(retained.upgrade().is_some());
    drop(b);
    assert!(retained.upgrade().is_none());
    let foreign = Engine::new_gpu().unwrap();
    assert!(!std::sync::Arc::ptr_eq(
        &rebuilt,
        &geometry_program(&foreign).unwrap()
    ));
}

#[test]
#[ignore = "requires GPU"]
fn analytic_shapes_match_nearest_boundary_and_world_jacobians() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let rotation = [0.0f32, 0.0, (0.37f32).sin(), (0.37f32).cos()];
    let dims = [
        [0.45, 0.45, 0.45, 1.0],
        [0.1, 0.35, 0.1, 2.0],
        [0.45, 0.14, 0.45, 3.0],
    ];
    let centers = [[1.0, -0.2, 0.3, 1.0]; 3];
    let mut shapes = Vec::new();
    let mut jacobians = Vec::new();
    let mut queries = Vec::new();
    for (body, dim) in dims.iter().enumerate() {
        shapes.extend([[0.0; 4], [0.0; 4], rotation, *dim]);
        for i in 0..6 {
            for j in 0..6 {
                jacobians.push(if i == j { 1.0 } else { 0.0 });
            }
        }
        for p in [
            [0.0, 0.0, 0.0],
            [0.6, 0.2, 0.3],
            [-0.7, -0.4, 0.2],
            [0.02, 0.03, 0.01],
            [0.0, 0.6, 0.0],
            [0.55, 0.25, 0.0],
            [0.3, 0.05, -0.3],
        ] {
            let world = rotate(rotation.map(f64::from), p);
            queries.push(query(
                std::array::from_fn(|i| (world[i] + centers[body][i] as f64) as f32),
                body as u32,
                0,
            ));
        }
    }
    let fixture = Fixture::new(&engine, &centers, &shapes, &jacobians, &queries, &[0.0; 6]);
    assert_eq!(fixture.run(&engine, false, true), 0);
    let geometry = read(&engine, fixture.geometry.geometry_buffer());
    let rows = read(&engine, &fixture.buffers[5]);
    let nodes = read(&engine, &fixture.buffers[4]);
    let norm = rotation
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let q = rotation.map(|x| x as f64 / norm);
    let inverse = [-q[0], -q[1], -q[2], q[3]];
    for (c, input) in queries.iter().enumerate() {
        let body = input.body_boundary[0] as usize;
        let arm = std::array::from_fn(|i| input.position_mass[i] as f64 - centers[body][i] as f64);
        let (local_normal, gap) = nearest(rotate(inverse, arm), dims[body].map(f64::from));
        let normal = rotate(q, local_normal);
        let torque = [
            arm[1] * normal[2] - arm[2] * normal[1],
            arm[2] * normal[0] - arm[0] * normal[2],
            arm[0] * normal[1] - arm[1] * normal[0],
        ];
        assert!(
            (geometry[4 * c + 3] as f64 - gap).abs() < 2e-6,
            "signed gap case {c}"
        );
        assert!(
            (nodes[12 * c + 7] as f64 - gap.max(0.0) / input.velocity_dt[3] as f64).abs() < 2e-5
        );
        for i in 0..3 {
            assert!(
                (geometry[4 * c + i] as f64 - normal[i]).abs() < 2e-5,
                "normal case {c} axis {i}"
            );
            assert!((rows[6 * c + i] as f64 - normal[i]).abs() < 2e-5);
            assert!((rows[6 * c + i + 3] as f64 - torque[i]).abs() < 2e-5);
        }
        if c % 7 != 0 {
            for axis in 0..6 {
                let mut gaps = [0.0; 2];
                for (slot, step) in [-1e-5f64, 1e-5].into_iter().enumerate() {
                    let mut displaced = arm;
                    if axis < 3 {
                        displaced[axis] -= step;
                    } else {
                        let mut incremental = [0.0, 0.0, 0.0, (step / 2.0).cos()];
                        incremental[axis - 3] = (-step / 2.0).sin();
                        displaced = rotate(incremental, arm);
                    }
                    gaps[slot] = nearest(rotate(inverse, displaced), dims[body].map(f64::from)).1;
                }
                let derivative = (gaps[1] - gaps[0]) / 2e-5;
                assert!(
                    (rows[6 * c + axis] as f64 + derivative).abs() < 2e-5,
                    "virtual work case {c} axis {axis}"
                );
            }
        }
    }
    println!(
        "21 translated/rotated sphere, tine-box and wheel-cylinder closest-boundary queries pass"
    );
}

#[test]
#[ignore = "requires GPU"]
fn generated_tine_floor_contacts_cross_surface_and_release_continuously() {
    let engine = Engine::new_gpu().unwrap();
    let shapes = [
        [0.0; 4],
        [0.0; 4],
        [0.0, 0.0, 0.0, 1.0],
        [0.1, 0.35, 0.1, 2.0],
    ];
    let mut input = query([0.0, -0.35, 0.0], 0, 4);
    let fixture = Fixture::new(
        &engine,
        &[[0.0; 4]],
        &shapes,
        &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        &[input],
        &[-0.002],
    );
    let mut previous: Option<f32> = None;
    for offset in [-2e-6f32, -1e-6, 0.0, 1e-6, 2e-6, 20e-6, 30e-6] {
        input.position_mass[1] = -0.35 - offset;
        engine
            .queue()
            .unwrap()
            .write_buffer(&fixture.buffers[3], 0, bytemuck::bytes_of(&input));
        assert_eq!(fixture.run(&engine, true, true), 0);
        let q = read(&engine, fixture.floor.rates_buffer())[0];
        let impulse = read(&engine, fixture.floor.contact_impulses_buffer())[3];
        let gap = -(input.position_mass[1] as f64) - shapes[3][1] as f64;
        let expected = (-0.002f32 as f64).max(-gap.max(0.0) / input.velocity_dt[3] as f64);
        assert!((q as f64 - expected).abs() < 2e-6);
        assert!((impulse as f64 - 860.0 * (expected - (-0.002f32 as f64))).abs() < 2e-5);
        assert_eq!(read(&engine, fixture.floor.velocities_buffer())[1], 0.0);
        assert!((read(&engine, fixture.floor.floor_impulses_buffer())[1] - impulse).abs() < 2e-5);
        if offset.abs() <= 2e-6 {
            if let Some(last) = previous {
                assert!((q - last).abs() < 0.00011);
            }
            previous = Some(q);
        }
    }
    assert_eq!(
        read(&engine, fixture.floor.contact_impulses_buffer())[3],
        0.0
    );
    println!(
        "GPU geometry-to-floor encode crosses the tine surface and releases at positive gap without interior-membership switching"
    );
}

#[test]
#[ignore = "requires GPU"]
fn geometry_faults_block_composed_publication_and_reset() {
    let engine = Engine::new_gpu().unwrap();
    let shapes = [
        [0.0; 4],
        [0.0; 4],
        [0.0, 0.0, 0.0, 1.0],
        [0.1, 0.35, 0.1, 2.0],
    ];
    let mut input = query([0.0, -0.35, 0.0], 0, 4);
    let fixture = Fixture::new(
        &engine,
        &[[0.0; 4]],
        &shapes,
        &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        &[input],
        &[-0.002],
    );
    engine.queue().unwrap().write_buffer(
        fixture.floor.rates_buffer(),
        0,
        bytemuck::cast_slice(&[99.0f32]),
    );
    for bad in [0u32, 1, 2, 3, 4] {
        let mut invalid = input;
        match bad {
            0 => invalid.body_boundary[0] = 1,
            1 => invalid.velocity_dt[3] = 0.0,
            2 => invalid.position_mass[0] = f32::NAN,
            3 => invalid.body_boundary[1] = 3,
            _ => invalid.velocity_dt[1] = -0.1,
        }
        engine
            .queue()
            .unwrap()
            .write_buffer(&fixture.buffers[3], 0, bytemuck::bytes_of(&invalid));
        assert_eq!(fixture.run(&engine, true, true), GEOMETRY_INPUT);
        assert_eq!(read(&engine, fixture.floor.rates_buffer()), [99.0]);
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[3], 0, bytemuck::bytes_of(&input));
    assert_eq!(fixture.run(&engine, true, false), GEOMETRY_INPUT);
    assert_eq!(fixture.run(&engine, true, true), 0);
    input.position_mass[3] = 0.0;
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[3], 0, bytemuck::bytes_of(&input));
    assert_eq!(fixture.run(&engine, true, true), 0);
    assert_eq!(read(&engine, fixture.floor.rates_buffer()), [-0.002]);
    assert_eq!(fixture.geometry.owned_buffer_bytes(), 32);
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let rejected = ExperimentalSoilContactGeometryGpu::new(
        &engine,
        SoilContactGeometryInputs {
            centers: &fixture.buffers[0],
            shapes: &fixture.buffers[1],
            jacobians: &fixture.buffers[2],
            queries: &fixture.buffers[3],
        },
        SoilContactGeometryOutputs {
            nodes: &fixture.buffers[4],
            rows: &fixture.buffers[5],
            status: fixture.floor.status_buffer(),
        },
        SoilContactGeometryCapacity {
            dofs: 1,
            bodies: 1,
            queries: 1,
            max_owned_bytes: 31,
        },
    );
    assert!(rejected.is_err());
    assert_eq!(scope.finish().observations, 0);
}

#[test]
#[ignore = "requires GPU"]
fn captured_tine_poses_preserve_signed_surface_crossing() {
    let engine = Engine::new_gpu().unwrap();
    let centers = [
        [-1.3554472, -0.04923303, 0.0000074236746, 0.3774917],
        [-1.3554425, -0.049238823, 0.0000058639894, 0.3774917],
    ];
    let rotations = [
        [0.0005375419, 0.00024168838, -0.006627643, 0.9999779],
        [0.00053580635, 0.00024075335, -0.006626232, 0.9999779],
    ];
    let mut gaps = Vec::new();
    for (center, rotation) in centers.into_iter().zip(rotations) {
        let shapes = [[0.0; 4], [0.0; 4], rotation, [0.1, 0.35, 0.1, 2.0]];
        let input = query([-1.3, -0.4, 4.4703484e-8], 0, 4);
        let fixture = Fixture::new(
            &engine,
            &[center],
            &shapes,
            &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            &[input],
            &[-0.002],
        );
        assert_eq!(fixture.run(&engine, false, true), 0);
        let sample = read(&engine, fixture.geometry.geometry_buffer());
        let norm = rotation
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let q = rotation.map(|v| v as f64 / norm);
        let arm = std::array::from_fn(|i| input.position_mass[i] as f64 - center[i] as f64);
        let inverse = [-q[0], -q[1], -q[2], q[3]];
        let (normal, gap) = nearest(rotate(inverse, arm), shapes[3].map(f64::from));
        let normal = rotate(q, normal);
        assert!((sample[3] as f64 - gap).abs() < 1e-7);
        for i in 0..3 {
            assert!((sample[i] as f64 - normal[i]).abs() < 2e-6);
        }
        gaps.push(sample[3]);
    }
    assert!(gaps[0] > 0.0 && gaps[1] < 0.0);
    assert!((gaps[0] - gaps[1]).abs() < 6e-6);
    println!("captured frame 25/substep 15 tine poses: signed gaps {gaps:?}");
}
