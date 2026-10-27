use super::*;

fn buffer(engine: &Engine, bytes: &[u8], minimum: u64) -> wgpu::Buffer {
    let result = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("soil floor fixture"),
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
        label: Some("soil floor observation"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
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
    buffers: [wgpu::Buffer; 4],
    projection: ExperimentalSoilFloorProjectionGpu,
}
struct Output {
    q: Vec<f32>,
    v: Vec<f32>,
    contact: Vec<f32>,
    floor: Vec<f32>,
    status: u32,
    working_set_events: u32,
    iterations: u32,
}
impl Fixture {
    fn new(
        engine: &Engine,
        mass: &[f32],
        free: &[f32],
        rows: &[f32],
        nodes: &[SoilFloorNode],
    ) -> Self {
        Self::with_iterations(engine, mass, free, rows, nodes, 64)
    }
    fn with_iterations(
        engine: &Engine,
        mass: &[f32],
        free: &[f32],
        rows: &[f32],
        nodes: &[SoilFloorNode],
        iterations: u32,
    ) -> Self {
        let buffers = [
            buffer(engine, bytemuck::cast_slice(mass), 4),
            buffer(engine, bytemuck::cast_slice(free), 4),
            buffer(engine, bytemuck::cast_slice(rows), 4 * free.len() as u64),
            buffer(engine, bytemuck::cast_slice(nodes), 48),
        ];
        let projection = ExperimentalSoilFloorProjectionGpu::new(
            engine,
            SoilFloorInputs {
                mass: &buffers[0],
                free_rates: &buffers[1],
                rows: &buffers[2],
                nodes: &buffers[3],
            },
            SoilFloorCapacity {
                dofs: free.len() as u32,
                nodes: nodes.len() as u32,
                max_iterations: iterations,
                max_owned_bytes: u64::MAX,
            },
        )
        .unwrap();
        Self {
            buffers,
            projection,
        }
    }
    fn run(&self, engine: &Engine, reset: bool) -> Output {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        if reset {
            encoder.clear_buffer(self.projection.status_buffer(), 0, None);
        }
        self.projection.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = transfers.finish();
        assert_eq!(traffic.upload_bytes, 0);
        assert_eq!(traffic.readback_bytes, 0);
        let status = read(engine, self.projection.status_buffer());
        Output {
            q: read(engine, self.projection.rates_buffer()),
            v: read(engine, self.projection.velocities_buffer()),
            contact: read(engine, self.projection.contact_impulses_buffer()),
            floor: read(engine, self.projection.floor_impulses_buffer()),
            status: status[0].to_bits(),
            working_set_events: status[3].to_bits(),
            iterations: status[1].to_bits(),
        }
    }
}

fn solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .unwrap();
        if a[p * n + k].abs() < 1e-12 {
            return None;
        }
        for j in 0..n {
            a.swap(k * n + j, p * n + j);
        }
        b.swap(k, p);
        for i in k + 1..n {
            let f = a[i * n + k] / a[k * n + k];
            for j in k..n {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i * n + j] * x[j]).sum::<f64>()) / a[i * n + i];
    }
    Some(x)
}

fn oracle(
    mass: &[f32],
    free: &[f32],
    rows: &[f32],
    nodes: &[SoilFloorNode],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = free.len();
    let size = n + 3 * nodes.len();
    let mut h = vec![0.0; size * size];
    for i in 0..n {
        for j in 0..n {
            h[i * size + j] = mass[i * n + j] as f64;
        }
    }
    let mut initial: Vec<f64> = free.iter().map(|&v| v as f64).collect();
    let mut constraints = Vec::new();
    let mut bounds = Vec::new();
    let mut owners = Vec::new();
    for (node_index, node) in nodes.iter().enumerate() {
        assert!(node.normal_mass[3] > 0.0);
        let offset = n + 3 * node_index;
        for axis in 0..3 {
            h[(offset + axis) * size + offset + axis] = node.normal_mass[3] as f64;
            initial.push(node.velocity_gap[axis] as f64);
        }
        let mut row = vec![0.0; size];
        for i in 0..n {
            row[i] = -rows[node_index * n + i] as f64;
        }
        for axis in 0..3 {
            row[offset + axis] = node.normal_mass[axis] as f64;
        }
        constraints.push(row);
        bounds.push(-node.velocity_gap[3] as f64);
        owners.push((node_index, None));
        for face in 0..6 {
            if node.boundary[0] & (1 << face) != 0 {
                let mut row = vec![0.0; size];
                row[offset + face / 2] = if face % 2 == 0 { 1.0 } else { -1.0 };
                constraints.push(row);
                bounds.push(0.0);
                owners.push((node_index, Some(face)));
            }
        }
    }
    let count = constraints.len();
    assert!(count <= 10);
    let response: Vec<_> = constraints
        .iter()
        .map(|row| solve(h.clone(), row.clone()).unwrap())
        .collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    let residual: Vec<_> = constraints
        .iter()
        .zip(&bounds)
        .map(|(row, b)| dot(row, &initial) - b)
        .collect();
    for mask in 0..1usize << count {
        let active: Vec<_> = (0..count).filter(|i| mask & (1 << i) != 0).collect();
        let k = active.len();
        let mut matrix = Vec::with_capacity(k * k);
        for &i in &active {
            for &j in &active {
                matrix.push(dot(&constraints[i], &response[j]));
            }
        }
        let Some(multipliers) = solve(matrix, active.iter().map(|&i| -residual[i]).collect())
        else {
            continue;
        };
        if multipliers.iter().any(|&v| v < -1e-9) {
            continue;
        }
        let mut x = initial.clone();
        for (&i, &lambda) in active.iter().zip(&multipliers) {
            for j in 0..size {
                x[j] += lambda * response[i][j];
            }
        }
        if constraints
            .iter()
            .zip(&bounds)
            .any(|(row, &b)| dot(row, &x) - b < -1e-9)
        {
            continue;
        }
        let mut contact = vec![0.0; nodes.len()];
        let mut floor = vec![0.0; 3 * nodes.len()];
        for (&i, &lambda) in active.iter().zip(&multipliers) {
            let (node, face) = owners[i];
            if let Some(face) = face {
                floor[3 * node + face / 2] += lambda * if face % 2 == 0 { 1.0 } else { -1.0 };
            } else {
                contact[node] += lambda;
            }
        }
        return (x, contact, floor);
    }
    panic!("no feasible full-space active set");
}

fn verify(mass: &[f32], free: &[f32], rows: &[f32], nodes: &[SoilFloorNode], output: &Output) {
    verify_duals(mass, free, rows, nodes, output, true);
}

fn verify_duals(
    mass: &[f32],
    free: &[f32],
    rows: &[f32],
    nodes: &[SoilFloorNode],
    output: &Output,
    unique: bool,
) {
    assert_eq!(output.status, 0, "floor solver fault");
    let n = free.len();
    let (expected, contact, floor) = oracle(mass, free, rows, nodes);
    for i in 0..n {
        assert!(
            (output.q[i] as f64 - expected[i]).abs() < 2e-5,
            "machine rate mismatch"
        );
    }
    for (c, node) in nodes.iter().enumerate() {
        assert!(
            !unique || (output.contact[4 * c + 3] as f64 - contact[c]).abs() < 2e-5,
            "contact impulse {} vs {}",
            output.contact[4 * c + 3],
            contact[c]
        );
        let lambda = output.contact[4 * c + 3] as f64;
        assert!(lambda >= 0.0);
        let mut slack = node.velocity_gap[3] as f64;
        for i in 0..n {
            slack -= rows[c * n + i] as f64 * output.q[i] as f64;
        }
        for axis in 0..3 {
            assert!(
                (output.v[4 * c + axis] as f64 - expected[n + 3 * c + axis]).abs() < 2e-5,
                "soil velocity mismatch"
            );
            assert!(
                !unique || (output.floor[4 * c + axis] as f64 - floor[3 * c + axis]).abs() < 2e-5,
                "floor impulse mismatch"
            );
            let error = node.normal_mass[3] as f64
                * (output.v[4 * c + axis] as f64 - node.velocity_gap[axis] as f64)
                - output.contact[4 * c + axis] as f64
                - output.floor[4 * c + axis] as f64;
            assert!(error.abs() < 2e-5, "node momentum {error}");
            slack += node.normal_mass[axis] as f64 * output.v[4 * c + axis] as f64;
            let flags = (node.boundary[0] >> (2 * axis)) & 3;
            if flags == 1 {
                assert!(output.v[4 * c + axis] >= 0.0);
                assert!(output.floor[4 * c + axis] >= 0.0);
            }
            if flags == 2 {
                assert!(output.v[4 * c + axis] <= 0.0);
                assert!(output.floor[4 * c + axis] <= 0.0);
            }
            assert!(
                (output.floor[4 * c + axis] as f64 * output.v[4 * c + axis] as f64).abs() < 2e-5
            );
        }
        assert!(slack >= -2e-5);
        assert!(
            (slack * lambda).abs() < 2e-5,
            "contact complementarity {slack}"
        );
    }
    for i in 0..n {
        let balance = (0..n)
            .map(|j| mass[i * n + j] as f64 * (output.q[j] as f64 - free[j] as f64))
            .sum::<f64>()
            + nodes
                .iter()
                .enumerate()
                .map(|(c, _)| rows[c * n + i] as f64 * output.contact[4 * c + 3] as f64)
                .sum::<f64>();
        assert!(balance.abs() < 2e-5, "machine momentum {balance}");
    }
    let energy = |q: &[f32], v: &[f32]| {
        0.5 * (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| q[i] as f64 * mass[i * n + j] as f64 * q[j] as f64)
                    .sum::<f64>()
            })
            .sum::<f64>()
            + 0.5
                * nodes
                    .iter()
                    .enumerate()
                    .map(|(c, node)| {
                        node.normal_mass[3] as f64
                            * (0..3).map(|k| (v[4 * c + k] as f64).powi(2)).sum::<f64>()
                    })
                    .sum::<f64>()
    };
    let initial: Vec<_> = nodes.iter().flat_map(|node| node.velocity_gap).collect();
    let before = energy(free, &initial);
    assert!(
        energy(&output.q, &output.v) <= before + 1e-6 * before.max(1.0),
        "energy increase"
    );
}

fn node(normal: [f32; 3], velocity: [f32; 3], mask: u32, gap: f32) -> SoilFloorNode {
    let norm = normal.iter().map(|v| v * v).sum::<f32>().sqrt();
    SoilFloorNode {
        normal_mass: [
            normal[0] / norm,
            normal[1] / norm,
            normal[2] / norm,
            0.47464246,
        ],
        velocity_gap: [velocity[0], velocity[1], velocity[2], gap],
        boundary: [mask, 0, 0, 0],
    }
}

#[test]
fn floor_node_layout_is_three_vec4s() {
    assert_eq!(std::mem::size_of::<SoilFloorNode>(), 48);
}

#[test]
#[ignore = "requires GPU"]
fn floor_condensation_matches_full_space_qp() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut cases = 0;
    for velocity in [[0.0f32, 0.0, 0.0], [0.1, 0.2, -0.1], [0.2, 0.05, 0.3]] {
        for mask in [0, 4, 5, 36] {
            let mut v = velocity;
            if mask & 32 != 0 {
                v[2] = -v[2].abs();
            }
            let input = node([-0.6, -0.8, 0.3], v, mask, 0.02);
            for free in [-0.5, 0.1, 0.8] {
                let fixture = Fixture::new(&engine, &[2.0], &[free], &[1.0], &[input]);
                let output = fixture.run(&engine, true);
                verify(&[2.0], &[free], &[1.0], &[input], &output);
                cases += 1;
            }
        }
    }
    let nodes = [
        node([-0.4, -0.8, 0.2], [0.1, 0.3, -0.2], 4, 0.0),
        node([0.2, -0.7, 0.4], [-0.1, 0.0, 0.1], 4, 0.03),
    ];
    let mass = [3.0, 0.2, 0.2, 2.0];
    let free = [0.5, -0.3];
    let rows = [1.0, 0.4, -0.2, 1.0];
    let output = Fixture::new(&engine, &mass, &free, &rows, &nodes).run(&engine, true);
    verify(&mass, &free, &rows, &nodes, &output);
    println!("{cases} single-node cases plus a shared-machine two-node full-space QP pass");
}

#[test]
#[ignore = "requires GPU"]
fn floor_nearly_normal_contact_and_wall_transition_are_continuous() {
    let engine = Engine::new_gpu().unwrap();
    for normal in [[0.01325, -0.9999, 0.001], [0.3, -0.9, 0.2]] {
        let input = node(normal, [0.0, 0.0, 0.0], 4, 0.0);
        let fixture = Fixture::new(&engine, &[860.0], &[0.002], &[1.0], &[input]);
        let output = fixture.run(&engine, true);
        verify(&[860.0], &[0.002], &[1.0], &[input], &output);
    }
    let input = node([-0.6, -0.8, 0.0], [0.0, 0.2, 0.0], 4, 0.0);
    let fixture = Fixture::new(&engine, &[2.0], &[0.2], &[1.0], &[input]);
    let mut previous: Option<f32> = None;
    let mut reactions = Vec::new();
    for free in [0.14932f32, 0.149325, 0.14933, 0.149335, 0.14934] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&fixture.buffers[1], 0, bytemuck::cast_slice(&[free]));
        let output = fixture.run(&engine, true);
        verify(&[2.0], &[free], &[1.0], &[input], &output);
        if let Some(last) = previous {
            assert!((output.contact[3] - last).abs() < 2e-5);
        }
        previous = Some(output.contact[3]);
        reactions.push(output.floor[1]);
    }
    assert_eq!(reactions[0], 0.0);
    assert!(reactions.last().unwrap() > &0.0);
    println!("near-normal conditioning and wall transition pass");
}

#[test]
#[ignore = "requires GPU"]
fn floor_hard_rows_solve_and_invalid_nodes_fail_closed() {
    let engine = Engine::new_gpu().unwrap();
    let mut input = node([0.0, -1.0, 0.0], [0.0, 0.0, 0.0], 4, 0.0);
    let fixture = Fixture::new(&engine, &[2.0], &[1.0], &[1.0], &[input]);
    engine.queue().unwrap().write_buffer(
        fixture.projection.rates_buffer(),
        0,
        bytemuck::cast_slice(&[99.0f32]),
    );
    let output = fixture.run(&engine, true);
    verify(&[2.0], &[1.0], &[1.0], &[input], &output);
    assert_eq!(output.q, [0.0]);
    let exact = oracle(&[2.0], &[1.0], &[1.0], &[input]);
    assert!(exact.0[0].abs() < 1e-12);
    let capped = Fixture::with_iterations(&engine, &[2.0], &[1.0], &[1.0], &[input], 1);
    engine.queue().unwrap().write_buffer(
        capped.projection.rates_buffer(),
        0,
        bytemuck::cast_slice(&[99.0f32]),
    );
    let failed = capped.run(&engine, true);
    assert_eq!(
        failed.status,
        crate::soil_normal_projection_gpu::NORMAL_ITERATIONS
    );
    assert_eq!(failed.q, [99.0]);
    let overflow = Fixture::new(
        &engine,
        &[1.0, 0.0, 0.0, 1.0],
        &[1e20, -1e20],
        &[1e30, 1e30],
        &[node([0.0, -1.0, 0.0], [0.0; 3], 4, 0.0)],
    );
    assert_eq!(
        overflow.run(&engine, true).status,
        crate::soil_normal_projection_gpu::NORMAL_NONFINITE
    );
    input = node([0.5, -1.0, 0.0], [0.0, -0.1, 0.0], 4, 0.0);
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[3], 0, bytemuck::bytes_of(&input));
    assert_eq!(fixture.run(&engine, true).status, FLOOR_INPUT);
    assert_eq!(fixture.run(&engine, false).status, FLOOR_INPUT);
    input.velocity_gap[1] = 0.0;
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[3], 0, bytemuck::bytes_of(&input));
    verify(
        &[2.0],
        &[1.0],
        &[1.0],
        &[input],
        &fixture.run(&engine, true),
    );
    assert_eq!(fixture.projection.owned_buffer_bytes(), 212);
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ExperimentalSoilFloorProjectionGpu::new(
            &engine,
            SoilFloorInputs {
                mass: &fixture.buffers[0],
                free_rates: &fixture.buffers[1],
                rows: &fixture.buffers[2],
                nodes: &fixture.buffers[3],
            },
            SoilFloorCapacity {
                dofs: 1,
                nodes: 1,
                max_iterations: 32,
                max_owned_bytes: 211
            }
        )
        .is_err()
    );
    assert_eq!(scope.finish().observations, 0);
    let empty = Fixture::new(&engine, &[2.0], &[-0.0], &[], &[]).run(&engine, true);
    assert_eq!(empty.status, 0);
    assert_eq!(empty.q[0].to_bits(), (-0.0f32).to_bits());
    println!(
        "hard-row solve, invalid free velocity, sticky reset, empty and preflight checks pass"
    );
}

#[test]
#[ignore = "requires GPU"]
fn floor_hard_rows_match_full_space_qp() {
    let engine = Engine::new_gpu().unwrap();
    let mass = [3.0, 0.2, 0.2, 2.0];
    let cases = [
        ([1.0, 0.4, -0.2, 1.0], [0.5, 0.3]),
        ([1.0, 0.0, 1.0, 1.0], [0.5, -0.8]),
        ([1.0, 1.0, 2.0, 2.0], [0.5, 0.2]),
        ([1.0, 0.0, -1.0, 0.0], [0.5, -0.3]),
        ([0.0, 0.0, -0.2, 1.0], [0.5, 0.3]),
    ];
    for (index, (rows, free)) in cases.into_iter().enumerate() {
        for gap in [0.0, 0.03, 0.7] {
            let nodes = [
                node([0.0, -1.0, 0.0], [0.1, 0.2, -0.1], 4, gap),
                node([0.0, -1.0, 0.0], [-0.1, 0.3, 0.1], 4, gap),
            ];
            let output = Fixture::new(&engine, &mass, &free, &rows, &nodes).run(&engine, true);
            assert_eq!(
                output.status, 0,
                "case={index} gap={gap} iterations={} events={}",
                output.iterations, output.working_set_events
            );
            verify_duals(
                &mass,
                &free,
                &rows,
                &nodes,
                &output,
                index != 2 && index != 3,
            );
        }
    }
    for normal in [[-0.6, -0.8, 0.0], [-0.3, -0.8, -0.5]] {
        for free in [-1.0, 0.0, 0.1, 1.0] {
            let input = node(normal, [0.3, 0.2, 0.1], 21, 0.02);
            let output =
                Fixture::new(&engine, &[2.0], &[free], &[1.0], &[input]).run(&engine, true);
            verify(&[2.0], &[free], &[1.0], &[input], &output);
        }
    }
    let nodes = [
        node([0.0, -1.0, 0.0], [0.0, 0.1, 0.0], 4, 0.01),
        node([0.3, -0.8, 0.2], [0.0, 0.0, 0.0], 4, 0.0),
    ];
    let rows = [1.0, 0.4, -0.2, 1.0];
    let free = [0.5, 0.3];
    let output = Fixture::new(&engine, &mass, &free, &rows, &nodes).run(&engine, true);
    verify(&mass, &free, &rows, &nodes, &output);
    println!("24 hard/mixed full-space QPs pass");
    let nodes = [node([0.0, -1.0, 0.0], [0.0; 3], 4, 0.0); 2];
    let rows = [1.0, 0.0, 1.0, 1.0];
    let free = [0.5, 1.5];
    let output = Fixture::new(&engine, &mass, &free, &rows, &nodes).run(&engine, true);
    verify(&mass, &free, &rows, &nodes, &output);
    assert!(
        output.working_set_events >> 16 > 0,
        "negative multiplier must remove a row"
    );
    println!("negative-multiplier removal exercised");
}

#[test]
#[ignore = "requires GPU"]
fn floor_full_rank_hard_corner_uses_all_32_machine_dofs() {
    let engine = Engine::new_gpu().unwrap();
    let n = 32;
    let mut mass = vec![0.0; n * n];
    let mut rows = vec![0.0; n * n];
    for i in 0..n {
        mass[i * n + i] = 2.0;
        rows[i * n + i] = 1.0;
    }
    let nodes = vec![node([0.0, -1.0, 0.0], [0.0; 3], 4, 0.0); n];
    let fixture = Fixture::new(&engine, &mass, &vec![1.0; n], &rows, &nodes);
    let output = fixture.run(&engine, true);
    assert_eq!(output.status, 0);
    assert!(output.q.iter().all(|q| q.abs() < 1e-6));
    for i in 0..n {
        assert!((output.contact[4 * i + 3] - 2.0).abs() < 2e-5);
        assert!((output.floor[4 * i + 1] - 2.0).abs() < 2e-5);
        assert_eq!(output.v[4 * i + 1], 0.0);
    }
}
