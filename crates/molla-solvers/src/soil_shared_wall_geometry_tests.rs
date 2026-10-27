use super::*;
use glam::{DVec3, Quat, Vec3};

#[test]
#[ignore = "requires real GPU"]
fn paired_soil_consumer_geometry_preserves_generalized_momentum() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut fixture = Fixture::new(&engine, Quat::IDENTITY, Vec3::ZERO, 3, [0.6, 0.2]);
    let free: Vec<[f32; 2]> = fixture
        .free
        .iter()
        .enumerate()
        .map(|(i, &v)| [v, (i + 1) as f32 * 2f32.powi(-32)])
        .collect();
    fixture.buffers[1] = buffer(&engine, &free, 56);
    let capacity = SoilSharedWallGeometryCapacity {
        dofs: 7,
        bodies: 2,
        contacts: 65,
        max_sweeps: 256,
        local_iterations: 64,
        max_owned_bytes: u64::MAX,
    };
    let bytes = ExperimentalSoilSharedWallGeometryGpu::required_bytes_for_rates(
        capacity,
        RateEncoding::Paired,
    )
    .unwrap();
    let extra_history = ExperimentalSoilPairedWallFrictionGpu::history_bytes(65) - 336 * 65;
    let extra_status = ExperimentalSoilPairedWallFrictionGpu::status_bytes() - 432;
    assert_eq!(bytes, 62872 + extra_history + extra_status);
    let input = || SoilSharedWallGeometryInputs {
        mass: &fixture.buffers[0],
        free_rates: &fixture.buffers[1],
        centers: &fixture.buffers[2],
        shapes: &fixture.buffers[3],
        jacobians: &fixture.buffers[4],
        queries: &fixture.buffers[5],
        coefficients: &fixture.buffers[6],
    };
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ExperimentalSoilSharedWallGeometryGpu::with_rate_encoding(
            &engine,
            input(),
            SoilSharedWallGeometryCapacity {
                max_owned_bytes: bytes - 1,
                ..capacity
            },
            RateEncoding::Paired
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let mut owner = ExperimentalSoilSharedWallGeometryGpu::with_rate_encoding(
        &engine,
        input(),
        SoilSharedWallGeometryCapacity {
            max_owned_bytes: bytes,
            ..capacity
        },
        RateEncoding::Paired,
    )
    .unwrap();
    assert_eq!(owner.owned_buffer_bytes(), bytes);
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    owner.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    assert_eq!(read(&engine, owner.status_buffer())[0].to_bits(), 0);
    let raw = read(&engine, owner.rates_buffer());
    let rates: Vec<f64> = raw
        .chunks_exact(2)
        .map(|q| q[0] as f64 + q[1] as f64)
        .collect();
    let impulses = read(&engine, owner.contact_impulses_buffer());
    let velocities = read(&engine, owner.velocities_buffer());
    let walls = read(&engine, owner.wall_impulses_buffer());
    let mut reaction = [0f64; 7];
    for (slot, query) in fixture.queries.iter().enumerate() {
        if query.position_mass[3] == 0. {
            continue;
        }
        let b = query.body_boundary[0] as usize;
        let impulse = DVec3::new(
            impulses[slot * 4] as f64,
            impulses[slot * 4 + 1] as f64,
            impulses[slot * 4 + 2] as f64,
        );
        let arm = Vec3::from_slice(&query.position_mass).as_dvec3() - fixture.centers[b].as_dvec3();
        for i in 0..7 {
            let j = |axis| fixture.jacobians[(6 * b + axis) * 7 + i] as f64;
            reaction[i] -= (DVec3::new(j(0), j(1), j(2)) + DVec3::new(j(3), j(4), j(5)).cross(arm))
                .dot(impulse);
        }
        for axis in 0..3 {
            let change = query.position_mass[3] as f64
                * (velocities[slot * 4 + axis] as f64 - query.velocity_dt[axis] as f64);
            let applied = impulses[slot * 4 + axis] as f64 + walls[slot * 4 + axis] as f64;
            assert!((change - applied).abs() < 1e-6 * change.abs().max(applied.abs()).max(1.));
        }
    }
    let mut worst = 0f64;
    for i in 0..7 {
        let change: f64 = (0..7)
            .map(|j| {
                fixture.mass[i * 7 + j] as f64 * (rates[j] - free[j][0] as f64 - free[j][1] as f64)
            })
            .sum();
        let error = (change - reaction[i]).abs() / change.abs().max(reaction[i].abs()).max(1.);
        worst = worst.max(error);
        assert!(error < 1e-6, "momentum {i}: {change} vs {}", reaction[i]);
    }
    eprintln!("paired geometry normalized generalized momentum error={worst}");
    let capture = owner.capture_first_input(&engine);
    assert!(capture.paired_rates);
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    encoder.clear_buffer(owner.status_buffer(), 0, None);
    owner.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(
        read(&engine, owner.status_buffer())[0].to_bits(),
        crate::soil_shared_wall_friction_gpu::capture::CAPTURE_STOP
    );
    assert_eq!(
        read(&engine, &capture.inputs[1]),
        free.iter().flatten().copied().collect::<Vec<_>>()
    );
}

fn selected_budget(base: u64, contacts: u32) -> u64 {
    if std::env::var_os("SOIL_WALL_TEST_SPLIT_PREPARE").is_some() && redistribution_selected() {
        base + 16 * u64::from(contacts.max(1))
    } else {
        base
    }
}

fn buffer<T: bytemuck::Pod>(engine: &Engine, values: &[T], minimum: u64) -> wgpu::Buffer {
    let bytes = bytemuck::cast_slice(values);
    let result = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("friction geometry fixture"),
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
fn read(engine: &Engine, source: &wgpu::Buffer) -> Vec<f32> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("friction geometry observation"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
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
fn run(engine: &Engine, owner: &ExperimentalSoilSharedWallGeometryGpu, reset: bool) -> u32 {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        encoder.clear_buffer(owner.status_buffer(), 0, None);
    }
    owner.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = transfers.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(
        dispatches.finish().encoded_dispatches,
        if owner.contacts == 0 {
            2
        } else {
            261 + u64::from(std::env::var_os("SOIL_WALL_TEST_SPLIT_PREPARE").is_some())
        }
    );
    read(engine, owner.status_buffer())[0].to_bits()
}
struct Fixture {
    buffers: [wgpu::Buffer; 7],
    queries: Vec<SoilContactQuery>,
    centers: [Vec3; 2],
    mass: Vec<f32>,
    free: Vec<f32>,
    jacobians: Vec<f32>,
    coefficients: [f32; 2],
    rotation: Quat,
    translation: Vec3,
}
impl Fixture {
    fn new(
        engine: &Engine,
        rotation: Quat,
        translation: Vec3,
        kind: u32,
        coefficients: [f32; 2],
    ) -> Self {
        let mut mass = vec![0.0; 49];
        for i in 0..7 {
            mass[i * 7 + i] = if i < 3 {
                7.0
            } else if i < 6 {
                3.0
            } else {
                0.05
            };
        }
        mass[4 * 7 + 4] += 0.05;
        mass[4 * 7 + 6] = 0.05;
        mass[6 * 7 + 4] = 0.05;
        let free = vec![2.0, 0.25, -1.0, 0.1, -0.3, 0.2, -2.0];
        let offsets = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let centers = offsets.map(|v| rotation * v + translation);
        let centers_data = centers.map(|v| v.extend(0.0).to_array());
        let mut shapes = vec![[0.0f32; 4]; 8];
        shapes[2] = rotation.to_array();
        shapes[6] = rotation.to_array();
        shapes[3] = match kind {
            1 => [0.45, 0.45, 0.45, 1.0],
            2 => [0.3, 0.2, 0.45, 2.0],
            _ => [0.45, 0.14, 0.14, 3.0],
        };
        shapes[7] = [0.2, 0.2, 0.2, 2.0];
        let mut jacobians = vec![0.0; 2 * 6 * 7];
        for b in 0..2 {
            for dof in 0..7 {
                let mut linear = Vec3::ZERO;
                let mut angular = Vec3::ZERO;
                if dof < 3 {
                    linear[dof] = 1.0;
                } else if dof < 6 {
                    angular[dof - 3] = 1.0;
                    linear = angular.cross(offsets[b]);
                } else if b == 0 {
                    angular = Vec3::Y;
                }
                let linear = rotation * linear;
                let angular = rotation * angular;
                for axis in 0..3 {
                    jacobians[(6 * b + axis) * 7 + dof] = linear[axis];
                    jacobians[(6 * b + axis + 3) * 7 + dof] = angular[axis];
                }
            }
        }
        let mut queries = vec![SoilContactQuery::default(); 65];
        for (slot, b, arm, m) in [
            (0, 0, Vec3::new(0.0, 0.0, -0.45), 2.0),
            (64, 1, Vec3::new(0.0, 0.0, -0.2), 1.5),
        ] {
            let position = rotation * (offsets[b] + arm) + translation;
            let velocity = rotation * Vec3::new(0.25, -0.1, 0.0);
            queries[slot] = SoilContactQuery {
                position_mass: position.extend(m).to_array(),
                velocity_dt: velocity.extend(1.0).to_array(),
                body_boundary: [b as u32, 0, 0, 0],
            };
        }
        let buffers = [
            buffer(engine, &mass, 4),
            buffer(engine, &free, 4),
            buffer(engine, &centers_data, 16),
            buffer(engine, &shapes, 64),
            buffer(engine, &jacobians, 24),
            buffer(engine, &queries, 48),
            buffer(engine, &coefficients, 4),
        ];
        Self {
            buffers,
            queries,
            centers,
            mass,
            free,
            jacobians,
            coefficients,
            rotation,
            translation,
        }
    }
    fn owner(
        &self,
        engine: &Engine,
        budget: u64,
        count: u32,
    ) -> Result<ExperimentalSoilSharedWallGeometryGpu> {
        let b = &self.buffers;
        ExperimentalSoilSharedWallGeometryGpu::new(
            engine,
            SoilSharedWallGeometryInputs {
                mass: &b[0],
                free_rates: &b[1],
                centers: &b[2],
                shapes: &b[3],
                jacobians: &b[4],
                queries: &b[5],
                coefficients: &b[6],
            },
            SoilSharedWallGeometryCapacity {
                dofs: 7,
                bodies: 2,
                contacts: count,
                max_sweeps: 256,
                local_iterations: 64,
                max_owned_bytes: budget,
            },
        )
    }
    fn verify(
        &self,
        engine: &Engine,
        owner: &ExperimentalSoilSharedWallGeometryGpu,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(run(engine, owner, true), 0);
        let q = read(engine, owner.rates_buffer());
        let velocities = read(engine, owner.velocities_buffer());
        let impulses = read(engine, owner.contact_impulses_buffer());
        let walls = read(engine, owner.wall_impulses_buffer());
        let geometry = read(engine, owner.geometry_buffer());
        let mut reaction = vec![0.0f64; 7];
        let mut linear = DVec3::ZERO;
        let mut angular = DVec3::ZERO;
        let mut energy = 0.0;
        let normal = (self.rotation * (-Vec3::Z)).as_dvec3();
        for i in 0..7 {
            for j in 0..7 {
                energy += 0.5
                    * f64::from(self.mass[i * 7 + j])
                    * (f64::from(q[i]) * f64::from(q[j])
                        - f64::from(self.free[i]) * f64::from(self.free[j]));
            }
        }
        for (slot, query) in self.queries.iter().enumerate() {
            let b = query.body_boundary[0] as usize;
            let m = f64::from(query.position_mass[3]);
            let impulse = Vec3::from_slice(&impulses[slot * 4..]).as_dvec3();
            let velocity = Vec3::from_slice(&velocities[slot * 4..]).as_dvec3();
            let wall = Vec3::from_slice(&walls[slot * 4..]).as_dvec3();
            if m == 0.0 {
                assert_eq!(wall, DVec3::ZERO);
                assert_eq!(&impulses[slot * 4..slot * 4 + 4], &[0.0; 4]);
                assert_eq!(&velocities[slot * 4..slot * 4 + 4], &[0.0; 4]);
                continue;
            }
            let free = Vec3::from_slice(&query.velocity_dt).as_dvec3();
            assert!((m * (velocity - free) - impulse - wall).length() < 2e-5);
            let position = Vec3::from_slice(&query.position_mass).as_dvec3();
            let arm = position - self.centers[b].as_dvec3();
            linear -= impulse;
            angular -= (position - self.translation.as_dvec3()).cross(impulse);
            let mut point = DVec3::ZERO;
            for i in 0..7 {
                let jv = DVec3::new(
                    self.jacobians[(6 * b) * 7 + i] as f64,
                    self.jacobians[(6 * b + 1) * 7 + i] as f64,
                    self.jacobians[(6 * b + 2) * 7 + i] as f64,
                );
                let jw = DVec3::new(
                    self.jacobians[(6 * b + 3) * 7 + i] as f64,
                    self.jacobians[(6 * b + 4) * 7 + i] as f64,
                    self.jacobians[(6 * b + 5) * 7 + i] as f64,
                );
                let j = jv + jw.cross(arm);
                reaction[i] -= j.dot(impulse);
                point += j * f64::from(q[i]);
            }
            let lambda = f64::from(impulses[slot * 4 + 3]);
            assert!(lambda >= -2e-5);
            assert!((normal.dot(impulse) - lambda).abs() < 2e-5);
            let slip = point - velocity;
            let separation = -slip.dot(normal)
                + f64::from(geometry[slot * 4 + 3]).max(0.0) / query.velocity_dt[3] as f64;
            assert!(separation >= -2e-5);
            assert!(
                (lambda * separation).abs()
                    < 2e-5 * (1.0 + impulse.length()) * (1.0 + velocity.length() + point.length())
            );
            for axis in 0..3 {
                let flags = (query.body_boundary[1] >> (2 * axis)) & 3;
                if flags == 0 {
                    assert_eq!(wall[axis], 0.0);
                } else {
                    let sign = if flags == 1 { 1.0 } else { -1.0 };
                    assert!(sign * wall[axis] >= -2e-5);
                    assert!(sign * velocity[axis] >= -2e-5);
                    assert!(
                        (wall[axis] * velocity[axis]).abs()
                            <= 2e-5 * (1.0 + wall.length()) * (1.0 + velocity.length())
                    );
                }
            }
            let tangent = impulse - lambda * normal;
            let cap = f64::from(self.coefficients[b]) * lambda;
            assert!(tangent.length() <= cap + 2e-5);
            let slip_tangent = slip - normal * slip.dot(normal);
            if slip_tangent.length() > 1e-5 {
                assert!((tangent - cap * slip_tangent.normalize()).length() < 2e-5);
            }
            energy += 0.5 * m * (velocity.length_squared() - free.length_squared());
        }
        let mut momentum = vec![0.0f64; 7];
        for i in 0..7 {
            for j in 0..7 {
                momentum[i] +=
                    f64::from(self.mass[i * 7 + j]) * (f64::from(q[j]) - f64::from(self.free[j]));
            }
            assert!(
                (momentum[i] - reaction[i]).abs() < 2e-5,
                "generalized reaction {i}: {} vs {}",
                momentum[i],
                reaction[i]
            );
        }
        let rotate = |v: DVec3| (self.rotation * v.as_vec3()).as_dvec3();
        assert!((rotate(DVec3::from_slice(&momentum[..3])) - linear).length() < 2e-5);
        assert!((rotate(DVec3::from_slice(&momentum[3..6])) - angular).length() < 2e-5);
        assert!(energy <= 1e-5, "energy increase {energy}");
        (q, velocities, impulses)
    }
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_geometry_world_reactions_and_frame_covariance() {
    let engine = Engine::new_gpu().unwrap();
    for kind in [1, 2, 3] {
        for coefficients in [[0.0, 0.0], [0.6, 0.2]] {
            let base = Fixture::new(&engine, Quat::IDENTITY, Vec3::ZERO, kind, coefficients);
            let owner = base.owner(&engine, selected_budget(62844, 65), 65).unwrap();
            assert_eq!(owner.owned_buffer_bytes(), selected_budget(62844, 65));
            let expected = base.verify(&engine, &owner);
            let rotation = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.4);
            let f = Fixture::new(
                &engine,
                rotation,
                Vec3::new(8.0, -4.0, 2.0),
                kind,
                coefficients,
            );
            let owner = f.owner(&engine, selected_budget(62844, 65), 65).unwrap();
            let actual = f.verify(&engine, &owner);
            for i in 0..7 {
                assert!(
                    (expected.0[i] - actual.0[i]).abs() < 2e-5,
                    "rotated rate {i}"
                );
            }
            for slot in [0, 64] {
                assert!(
                    (rotation * Vec3::from_slice(&expected.1[4 * slot..])
                        - Vec3::from_slice(&actual.1[4 * slot..]))
                    .length()
                        < 2e-5
                );
                assert!(
                    (rotation * Vec3::from_slice(&expected.2[4 * slot..])
                        - Vec3::from_slice(&actual.2[4 * slot..]))
                    .length()
                        < 2e-5
                );
                assert!((expected.2[4 * slot + 3] - actual.2[4 * slot + 3]).abs() < 2e-5);
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn shared_wall_geometry_boundaries_faults_reuse_and_budget() {
    let engine = Engine::new_gpu().unwrap();
    let mut f = Fixture::new(&engine, Quat::IDENTITY, Vec3::ZERO, 3, [0.6, 0.2]);
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        f.owner(&engine, selected_budget(62844, 65) - 1, 65)
            .is_err()
    );
    assert_eq!(scope.finish().observations, 0);
    let owner = f.owner(&engine, selected_budget(62844, 65), 65).unwrap();
    f.verify(&engine, &owner);
    let masks =
        (0..3).flat_map(|x| (0..3).flat_map(move |y| (0..3).map(move |z| x | (y << 2) | (z << 4))));
    for mask in masks {
        f.queries[0].body_boundary[1] = mask;
        let original = f.queries[0].velocity_dt;
        f.queries[0].velocity_dt = [0.0, 0.0, 0.0, 1.0];
        engine
            .queue()
            .unwrap()
            .write_buffer(&f.buffers[5], 0, bytemuck::cast_slice(&f.queries));
        f.verify(&engine, &owner);
        if mask == 16 {
            let contact = read(&engine, owner.contact_impulses_buffer());
            let wall = read(&engine, owner.wall_impulses_buffer());
            assert!(contact[2] < -0.1 && wall[2] > 0.1);
            assert!((contact[2] + wall[2]).abs() < 2e-5);
            println!(
                "active lower-Z contact={:?}, external wall={:?}",
                &contact[..4],
                &wall[..4]
            );
        }
        println!("wall mask {mask} passes");
        f.queries[0].body_boundary[1] = 0;
        f.queries[0].velocity_dt = original;
        engine
            .queue()
            .unwrap()
            .write_buffer(&f.buffers[5], 0, bytemuck::cast_slice(&f.queries));
        assert_eq!(run(&engine, &owner, false), 0);
        f.verify(&engine, &owner);
    }
    for coefficient in [-1.0f32, f32::NAN, f32::INFINITY] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&f.buffers[6], 0, bytemuck::bytes_of(&coefficient));
        assert_eq!(run(&engine, &owner, true), GEOMETRY_INPUT);
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&f.buffers[6], 0, bytemuck::cast_slice(&f.coefficients));
    assert_eq!(run(&engine, &owner, false), GEOMETRY_INPUT);
    f.verify(&engine, &owner);
    let published = [
        read(&engine, owner.velocities_buffer()),
        read(&engine, owner.contact_impulses_buffer()),
        read(&engine, owner.wall_impulses_buffer()),
    ];
    let valid = f.queries[0];
    for kind in 0..5 {
        let mut invalid = valid;
        match kind {
            0 => invalid.body_boundary[1] = 3,
            1 => {
                invalid.body_boundary[1] = 1;
                invalid.velocity_dt[0] = -1.0;
            }
            2 => invalid.velocity_dt[3] = 0.0,
            3 => invalid.position_mass[0] = f32::NAN,
            _ => invalid.body_boundary[0] = 2,
        }
        f.queries[0] = invalid;
        engine
            .queue()
            .unwrap()
            .write_buffer(&f.buffers[5], 0, bytemuck::cast_slice(&f.queries));
        assert_eq!(run(&engine, &owner, true), GEOMETRY_INPUT);
        assert_eq!(
            [
                read(&engine, owner.velocities_buffer()),
                read(&engine, owner.contact_impulses_buffer()),
                read(&engine, owner.wall_impulses_buffer())
            ],
            published
        );
        f.queries[0] = valid;
        engine
            .queue()
            .unwrap()
            .write_buffer(&f.buffers[5], 0, bytemuck::cast_slice(&f.queries));
        assert_eq!(run(&engine, &owner, false), GEOMETRY_INPUT);
        f.verify(&engine, &owner);
    }
    f.queries.fill(SoilContactQuery::default());
    engine
        .queue()
        .unwrap()
        .write_buffer(&f.buffers[5], 0, bytemuck::cast_slice(&f.queries));
    assert_eq!(run(&engine, &owner, true), 0);
    assert_eq!(read(&engine, owner.rates_buffer()), f.free);
    assert!(
        read(&engine, owner.contact_impulses_buffer())
            .iter()
            .all(|&v| v == 0.0)
    );
    let empty = f.owner(&engine, selected_budget(1660, 0), 0).unwrap();
    assert_eq!(run(&engine, &empty, true), 0);
    assert_eq!(read(&engine, empty.rates_buffer()), f.free);
}

#[test]
fn shared_wall_geometry_budget_preflight() {
    let cap = SoilSharedWallGeometryCapacity {
        dofs: 7,
        bodies: 2,
        contacts: 65,
        max_sweeps: 256,
        local_iterations: 64,
        max_owned_bytes: 62844,
    };
    assert_eq!(
        ExperimentalSoilSharedWallGeometryGpu::required_bytes(cap).unwrap(),
        selected_budget(62844, 65)
    );
    assert_eq!(
        ExperimentalSoilSharedWallGeometryGpu::required_bytes(SoilSharedWallGeometryCapacity {
            contacts: 0,
            ..cap
        })
        .unwrap(),
        selected_budget(1660, 0)
    );
    assert!(
        ExperimentalSoilSharedWallGeometryGpu::required_bytes(SoilSharedWallGeometryCapacity {
            dofs: 33,
            ..cap
        })
        .is_err()
    );
    assert!(
        ExperimentalSoilSharedWallGeometryGpu::required_bytes(SoilSharedWallGeometryCapacity {
            max_sweeps: 257,
            ..cap
        })
        .is_err()
    );
}
