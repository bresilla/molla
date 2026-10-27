use super::*;
use glam::{DMat3, DVec3};

fn vector(v: [f32; 4]) -> DVec3 {
    DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
}
fn packed(v: DVec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}
fn matrix(input: &SoilWallFrictionInput) -> DMat3 {
    DMat3::from_cols(
        vector(input.inverse_machine[0]),
        vector(input.inverse_machine[1]),
        vector(input.inverse_machine[2]),
    )
}
fn fixture(
    mask: u32,
    mode: u32,
    mass: f64,
    mu: f64,
    anisotropic: bool,
) -> (SoilWallFrictionInput, DVec3, DVec3, DVec3, DVec3) {
    let wall_mode = mode / 3;
    let mode = mode % 3;
    let a = if anisotropic {
        DMat3::from_cols(
            DVec3::new(0.3, 0.025, -0.015),
            DVec3::new(0.025, 0.2, 0.02),
            DVec3::new(-0.015, 0.02, 0.4),
        )
    } else {
        DMat3::IDENTITY / 7.0
    };
    let mut n = DVec3::ZERO;
    let mut signs = DVec3::ZERO;
    for axis in 0..3 {
        let flags = (mask >> (2 * axis)) & 3;
        if flags != 0 {
            signs[axis] = if flags == 1 { 1.0 } else { -1.0 };
            n[axis] = -signs[axis];
        }
    }
    if mask == 0 {
        n = -DVec3::Y;
    }
    n = n.normalize();
    let abs = n.abs();
    let axis = if abs.y < abs.x && abs.y <= abs.z {
        DVec3::Y
    } else if abs.z < abs.x && abs.z < abs.y {
        DVec3::Z
    } else {
        DVec3::X
    };
    let t0 = axis.cross(n).normalize();
    let t1 = n.cross(t0);
    let direction = 0.6 * t0 + 0.8 * t1;
    let lambda = if mode == 0 { 0.0 } else { 2.0 };
    let tangent = if mode == 1 {
        0.25 * mu * lambda * t0
    } else {
        mu * lambda * direction
    };
    let k = lambda * n + tangent;
    let mut wall = DVec3::ZERO;
    let mut soil = DVec3::new(0.2, -0.15, 0.1);
    for axis in 0..3 {
        if signs[axis] != 0.0 {
            if wall_mode == 0 || (wall_mode == 2 && axis == 0) {
                soil[axis] = 0.0;
                wall[axis] = signs[axis] * (0.25 + k[axis].abs());
            } else {
                soil[axis] = signs[axis] * (1.0 + k[axis].abs() / mass);
            }
        }
    }
    let gap = 0.025;
    let machine = soil
        + gap * n
        + if mode == 0 {
            -0.5 * n + direction
        } else if mode == 1 {
            DVec3::ZERO
        } else {
            1.5 * direction
        };
    let input = SoilWallFrictionInput {
        inverse_machine: [
            packed(a.x_axis, 0.0),
            packed(a.y_axis, 0.0),
            packed(a.z_axis, 0.0),
        ],
        machine_velocity_friction: packed(machine + a * k, mu),
        soil_velocity_mass: packed(soil - (k + wall) / mass, mass),
        normal_gap: packed(n, gap),
        boundary: [mask, 0, 0, 0],
    };
    (input, k, wall, soil, machine)
}
fn input_buffer(engine: &Engine, rows: &[SoilWallFrictionInput]) -> wgpu::Buffer {
    let result = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("wall friction fixture"),
            size: 112 * rows.len().max(1) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    if !rows.is_empty() {
        engine
            .queue()
            .unwrap()
            .write_buffer(&result, 0, bytemuck::cast_slice(rows));
    }
    result
}
fn run(engine: &Engine, solver: &ExperimentalSoilWallFrictionGpu) -> Vec<SoilWallFrictionOutput> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wall friction fixture readback"),
        size: solver.output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = transfers.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(
        dispatches.finish().encoded_dispatches,
        u64::from(solver.count > 0)
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&solver.output, 0, &staging, 0, staging.size());
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
    let result =
        bytemuck::cast_slice::<u8, SoilWallFrictionOutput>(&staging.slice(..).get_mapped_range())
            .to_vec();
    staging.unmap();
    result
}
fn verify(input: &SoilWallFrictionInput, output: &SoilWallFrictionOutput) {
    let k = -vector(output.machine_impulse_normal);
    let wall = vector(output.wall_impulse);
    let soil = vector(output.soil_velocity_energy);
    let machine = vector(output.machine_velocity_residual);
    let n = vector(input.normal_gap).normalize();
    let mass = input.soil_velocity_mass[3] as f64;
    let a = matrix(input);
    let lambda = k.dot(n);
    let gap = input.normal_gap[3] as f64;
    let mu = input.machine_velocity_friction[3] as f64;
    let tangent = k - lambda * n;
    let relative = soil - machine;
    let slip = relative - relative.dot(n) * n;
    let scale = 1.0 + k.length() + wall.length();
    let vscale = 1.0 + soil.length() + machine.length();
    assert!((mass * (soil - vector(input.soil_velocity_mass)) - k - wall).length() <= 2e-5 * scale);
    assert!((machine - vector(input.machine_velocity_friction) + a * k).length() <= 2e-5 * vscale);
    assert!(lambda >= -2e-5 * scale);
    let separation = relative.dot(n) + gap;
    assert!(separation >= -2e-5 * vscale);
    assert!((lambda * separation).abs() <= 2e-5 * scale * vscale);
    assert!(tangent.length() <= mu * lambda + 2e-5 * scale);
    if slip.length() > 2e-5 * vscale {
        assert!((tangent + mu * lambda * slip.normalize()).length() <= 2e-5 * scale);
    }
    for axis in 0..3 {
        let flag = (input.boundary[0] >> (2 * axis)) & 3;
        if flag == 0 {
            assert_eq!(wall[axis], 0.0);
        } else {
            let sign = if flag == 1 { 1.0 } else { -1.0 };
            assert!(sign * wall[axis] >= -2e-5 * scale);
            assert!(sign * soil[axis] >= -2e-5 * vscale);
            assert!((wall[axis] * soil[axis]).abs() <= 2e-5 * scale * vscale);
        }
    }
    let inverse = a.inverse();
    let free_machine = vector(input.machine_velocity_friction);
    let before = 0.5 * free_machine.dot(inverse * free_machine)
        + 0.5 * mass * vector(input.soil_velocity_mass).length_squared();
    let after = 0.5 * machine.dot(inverse * machine) + 0.5 * mass * soil.length_squared();
    assert!(after - before <= 2e-5 * (1.0 + before));
    assert!(
        (after - before - output.soil_velocity_energy[3] as f64).abs()
            <= 2e-5 * (1.0 + before + after)
    );
}
#[test]
fn wall_friction_layout() {
    assert_eq!(std::mem::size_of::<SoilWallFrictionInput>(), 112);
    assert_eq!(std::mem::size_of::<SoilWallFrictionOutput>(), 80);
}

#[test]
fn wall_friction_armijo_rounding_requires_strict_progress() {
    let old = std::hint::black_box(7.446153e-7f32);
    let step = std::hint::black_box(2f32.powi(-14));
    let bound = (1.0 - 2e-4 * step) * old;
    assert_eq!(bound.to_bits(), old.to_bits());
    assert!(old <= bound);
    let accepted = |candidate: f32| candidate.is_finite() && candidate < old && candidate <= bound;
    assert!(!accepted(old));
    assert!(accepted(f32::from_bits(old.to_bits() - 1)));
    assert!(!accepted(f32::NAN));
    assert!(
        include_str!("../shaders/soil_wall_friction_block.wgsl")
            .contains("candidate<old_merit && candidate<=")
    );
}

#[test]
#[ignore = "requires GPU; captured tractor local friction input"]
fn wall_friction_captured_tractor_floor_contact() {
    let engine = Engine::new_gpu().unwrap();
    let words = [
        997681579u32,
        3145275839,
        3114211206,
        0,
        3145275839,
        1004444746,
        3121838838,
        0,
        3114211206,
        3121838838,
        1010303649,
        0,
        3198734199,
        3214030784,
        3166063017,
        1050253722,
        3180273706,
        1068886511,
        3116967767,
        1049629882,
        1019386244,
        3212832130,
        3005412428,
        1062406179,
        4,
        0,
        0,
        0,
    ];
    for words in [
        words,
        [
            997582675, 3145261218, 788494941, 0, 3145261218, 1004313114, 766710861, 0, 788494941,
            766710861, 1010303649, 0, 3204578252, 3209959748, 3174730566, 1050253722, 3187503437,
            1068503348, 3036048863, 1051447286, 1019386285, 3212832130, 3008965943, 1062401079, 4,
            0, 0, 0,
        ],
        [
            997681579, 3145275840, 966727572, 0, 3145275840, 1004444747, 974355194, 0, 966727572,
            974355194, 1010303651, 0, 3204485221, 3210022948, 3165089434, 1050253722, 3180273394,
            1068886668, 969274352, 1049629885, 1019386215, 3212832130, 3002475066, 1062407199, 4,
            0, 0, 0,
        ],
    ] {
        let input =
            bytemuck::pod_read_unaligned::<SoilWallFrictionInput>(bytemuck::cast_slice(&words));
        let buffer = input_buffer(&engine, &[input]);
        let source = crate::soil_shared_wall_friction_gpu::register_rounding_source(
            &ExperimentalSoilWallFrictionGpu::shader_source(),
            "params.pad0",
        );
        let source = crate::soil_shared_wall_friction_gpu::unrolled_direction_source(&source);
        let source = source.replace("separation>=-1e-5*velocity_scale",
        "separation>=-1e-6*max(1.0,max(abs(dot(axes[0],soil)),max(abs(dot(axes[0],machine)),input.normal_gap.w)))");
        let solver =
            ExperimentalSoilWallFrictionGpu::with_source(&engine, &buffer, 1, 64, 96, &source)
                .unwrap();
        let result = run(&engine, &solver);
        println!("captured tractor local status={:?}", result[0].status);
        if result[0].status[0] != 0 {
            let dump = r#"
    block_output.machine_impulse_normal=vec4<f32>(x[0],x[1],x[2],0.0);
    block_output.soil_velocity_energy=vec4<f32>(x[3],x[4],x[5],0.0);
    block_output.machine_velocity_residual=vec4<f32>(tail[0],tail[1],tail[2],0.0);
    block_output.wall_impulse=vec4<f32>(tail[3],tail[4],tail[5],merit(residual(x,tail,false)));
"#;
            let diagnostic = source
                .replace("x=trial; tail=trial_tail; accepted=true;",
                    "if candidate==old_merit { block_output.status.w+=1u; } x=trial; tail=trial_tail; accepted=true;")
                .replace(
                    "block_output.status.x=16u;",
                    &format!("block_output.status.x=16u;{dump}"),
                )
                .replace(
                    "block_output.status.x=8u; return;",
                    &format!("block_output.status.x=8u;{dump} return;"),
                );
            let probe = ExperimentalSoilWallFrictionGpu::with_source(
                &engine,
                &buffer,
                1,
                64,
                96,
                &diagnostic,
            )
            .unwrap();
            let raw = run(&engine, &probe)[0];
            let x = std::array::from_fn(|i| {
                if i < 3 {
                    raw.machine_impulse_normal[i] as f64 + raw.machine_velocity_residual[i] as f64
                } else {
                    raw.soil_velocity_energy[i - 3] as f64 + raw.wall_impulse[i - 3] as f64
                }
            });
            println!(
                "local final {raw:?}; x={x:?}; residual={:?}; refined={:?}",
                f64_residual(&input, x),
                refine_f64(&input, x)
            );
        }
        assert_eq!(result[0].status[0], 0);
        assert!(result[0].status[1] < 64);
        verify(&input, &result[0]);
        assert_eq!(result[0].soil_velocity_energy[1], 0.0);
        assert_eq!(result[0].status[3], 2);
        let impulse = -vector(result[0].machine_impulse_normal);
        let wall = vector(result[0].wall_impulse);
        let free = vector(input.soil_velocity_mass);
        let mass = input.soil_velocity_mass[3] as f64;
        let residual = mass * (vector(result[0].soil_velocity_energy) - free) - impulse - wall;
        let scale = 1.0 + impulse.length() + wall.length() + (mass * free).length();
        assert!(residual.length() <= 1e-6 * scale);
        println!(
            "active wall normalized momentum error={:e}",
            residual.length() / scale
        );
        let n = vector(input.normal_gap).normalize();
        let soil = n.dot(vector(result[0].soil_velocity_energy));
        let machine = n.dot(vector(result[0].machine_velocity_residual));
        let gap = input.normal_gap[3] as f64;
        assert!(soil - machine + gap >= -1e-6 * 1f64.max(soil.abs()).max(machine.abs()).max(gap));
    }
}

#[test]
fn wall_friction_captured_floor_f64_roots() {
    let words = [
        997582675u32,
        3145261218,
        788494941,
        0,
        3145261218,
        1004313114,
        766710861,
        0,
        788494941,
        766710861,
        1010303649,
        0,
        3204578252,
        3209959748,
        3174730566,
        1050253722,
        3187503437,
        1068503348,
        3036048863,
        1051447286,
        1019386285,
        3212832130,
        3008965943,
        1062401079,
        4,
        0,
        0,
        0,
    ];
    let input = bytemuck::pod_read_unaligned::<SoilWallFrictionInput>(bytemuck::cast_slice(&words));
    let mut roots: Vec<[f64; 6]> = Vec::new();
    for lambda in [0.0, 0.5, 1.0, 5.0, 50.0] {
        for tangent in [-0.3, 0.0, 0.3] {
            for wall in [0.0, 0.5, 1.0] {
                let seed = [lambda, tangent * lambda, 0.0, 0.0, wall * lambda, 0.0];
                if let Some(root) = refine_f64(&input, seed) {
                    if roots.iter().all(|other| {
                        other
                            .iter()
                            .zip(root)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0, f64::max)
                            > 1e-6
                    }) {
                        println!(
                            "seed={seed:?} root={root:?} residual={:?}",
                            f64_residual(&input, root)
                        );
                        roots.push(root);
                    }
                }
            }
        }
    }
    assert!(!roots.is_empty(), "no root found in bounded f64 seed probe");
}

fn basis(input: &SoilWallFrictionInput) -> [DVec3; 6] {
    let n = vector(input.normal_gap).normalize();
    let abs = n.abs();
    let axis = if abs.y < abs.x && abs.y <= abs.z {
        DVec3::Y
    } else if abs.z < abs.x && abs.z < abs.y {
        DVec3::Z
    } else {
        DVec3::X
    };
    let t = axis.cross(n).normalize();
    let mut result = [n, t, n.cross(t), DVec3::ZERO, DVec3::ZERO, DVec3::ZERO];
    for axis in 0..3 {
        let flag = (input.boundary[0] >> (2 * axis)) & 3;
        if flag != 0 {
            result[3 + axis][axis] = if flag == 1 { 1.0 } else { -1.0 };
        }
    }
    result
}
fn world_solution(input: &SoilWallFrictionInput, x: [f64; 6]) -> [DVec3; 4] {
    let c = basis(input);
    let k = c[0] * x[0] + c[1] * x[1] + c[2] * x[2];
    let w = c[3] * x[3] + c[4] * x[4] + c[5] * x[5];
    [
        k,
        w,
        vector(input.soil_velocity_mass) + (k + w) / input.soil_velocity_mass[3] as f64,
        vector(input.machine_velocity_friction) - matrix(input) * k,
    ]
}
fn coordinates(input: &SoilWallFrictionInput, k: DVec3, w: DVec3) -> [f64; 6] {
    let c = basis(input);
    [
        c[0].dot(k),
        c[1].dot(k),
        c[2].dot(k),
        c[3].dot(w),
        c[4].dot(w),
        c[5].dot(w),
    ]
}
fn f64_residual(input: &SoilWallFrictionInput, x: [f64; 6]) -> [f64; 6] {
    let c = basis(input);
    let [_, _, soil, machine] = world_solution(input, x);
    let relative = soil - machine;
    let mut result = [0.0; 6];
    let normal = c[0].dot(relative) + input.normal_gap[3] as f64;
    result[0] = x[0] - (x[0] - normal).max(0.0);
    let y = [x[1] - c[1].dot(relative), x[2] - c[2].dot(relative)];
    let radius = y[0].hypot(y[1]);
    let cap = input.machine_velocity_friction[3] as f64 * x[0].max(0.0);
    let scale = if radius > cap { cap / radius } else { 1.0 };
    result[1] = x[1] - scale * y[0];
    result[2] = x[2] - scale * y[1];
    for i in 3..6 {
        result[i] = if c[i] == DVec3::ZERO {
            x[i]
        } else {
            x[i] - (x[i] - c[i].dot(soil)).max(0.0)
        };
    }
    result
}
fn refine_f64(input: &SoilWallFrictionInput, mut x: [f64; 6]) -> Option<[f64; 6]> {
    for _ in 0..24 {
        let f = f64_residual(input, x);
        let norm = f.iter().map(|v| v * v).sum::<f64>();
        if norm.sqrt() <= 1e-11 * (1.0 + x.iter().map(|v| v.abs()).fold(0.0, f64::max)) {
            return Some(x);
        }
        let mut jac = [[0.0; 6]; 6];
        for j in 0..6 {
            let h = 1e-6 * (1.0 + x[j].abs());
            let mut plus = x;
            let mut minus = x;
            plus[j] += h;
            minus[j] -= h;
            let plus = f64_residual(input, plus);
            let minus = f64_residual(input, minus);
            for i in 0..6 {
                jac[i][j] = (plus[i] - minus[i]) / (2.0 * h);
            }
        }
        let mut delta = f.map(|v| -v);
        for k in 0..6 {
            let pivot = (k..6).max_by(|&i, &j| jac[i][k].abs().total_cmp(&jac[j][k].abs()))?;
            if !jac[pivot][k].is_finite() || jac[pivot][k].abs() < 1e-14 {
                return None;
            }
            jac.swap(k, pivot);
            delta.swap(k, pivot);
            for i in k + 1..6 {
                let factor = jac[i][k] / jac[k][k];
                for j in k + 1..6 {
                    jac[i][j] -= factor * jac[k][j];
                }
                delta[i] -= factor * delta[k];
            }
        }
        for i in (0..6).rev() {
            for j in i + 1..6 {
                delta[i] -= jac[i][j] * delta[j];
            }
            delta[i] /= jac[i][i];
        }
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..24 {
            let trial = std::array::from_fn(|i| x[i] + step * delta[i]);
            let next = f64_residual(input, trial)
                .iter()
                .map(|v| v * v)
                .sum::<f64>();
            if next.is_finite() && next <= (1.0 - 1e-4 * step) * norm {
                x = trial;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            return None;
        }
    }
    None
}

#[test]
#[ignore = "requires GPU; independent multiple-root diagnostic"]
fn wall_friction_multiple_roots_diagnostic() {
    let engine = Engine::new_gpu().unwrap();
    let (row, k, wall, _, _) = fixture(17, 8, 0.1, 1.0, false);
    let buffer = input_buffer(&engine, &[row]);
    let solver = ExperimentalSoilWallFrictionGpu::new(&engine, &buffer, 1, 64, 96).unwrap();
    let output = run(&engine, &solver);
    assert_eq!(output[0].status[0], 0);
    verify(&row, &output[0]);
    let first = refine_f64(&row, coordinates(&row, k, wall)).unwrap();
    let second = refine_f64(
        &row,
        coordinates(
            &row,
            -vector(output[0].machine_impulse_normal),
            vector(output[0].wall_impulse),
        ),
    )
    .unwrap();
    let separation = (world_solution(&row, first)[0] - world_solution(&row, second)[0]).length();
    println!(
        "same rounded input={row:?}; roots={first:?} / {second:?}; residuals={:?} / {:?}; contact impulse separation={separation}",
        f64_residual(&row, first),
        f64_residual(&row, second)
    );
    assert!(separation > 0.1);
}

#[test]
#[ignore = "requires GPU; repaired stagnation regression"]
fn wall_friction_stagnation_diagnostic() {
    let engine = Engine::new_gpu().unwrap();
    let row = fixture(1, 1, 1.0, 0.3, false).0;
    let buffer = input_buffer(&engine, &[row]);
    let source = ExperimentalSoilWallFrictionGpu::shader_source();
    assert_eq!(source.matches("block_output.status.x=16u;").count(), 1);
    let source = source.replace(
        "block_output.status.x=16u;",
        r#"
        let f=residual(x,tail,false);
        let reference=array<f32,6>(2.0,0.15,0.0,2.25,0.0,0.0);
        var reference_tail:array<f32,6>;
        let rf=residual(reference,reference_tail,false);
        block_output.machine_impulse_normal=vec4<f32>(x[0],x[1],x[2],merit(f));
        block_output.soil_velocity_energy=vec4<f32>(x[3],x[4],x[5],merit(rf));
        block_output.machine_velocity_residual=vec4<f32>(f[0],f[1],f[2],f[3]);
        block_output.wall_impulse=vec4<f32>(rf[0],rf[1],rf[2],rf[3]);
        block_output.status.x=16u;
    "#,
    );
    let solver =
        ExperimentalSoilWallFrictionGpu::with_source(&engine, &buffer, 1, 64, 96, &source).unwrap();
    let result = run(&engine, &solver);
    println!(
        "stagnation diagnostic input={row:?}; output={:?}",
        result[0]
    );
    assert_eq!(result[0].status[0], 0);
    if result[0].status[0] == 0 {
        verify(&row, &result[0]);
        let (_, k, wall, soil, machine) = fixture(1, 1, 1.0, 0.3, false);
        for (actual, expected) in [
            (-vector(result[0].machine_impulse_normal), k),
            (vector(result[0].wall_impulse), wall),
            (vector(result[0].soil_velocity_energy), soil),
            (vector(result[0].machine_velocity_residual), machine),
        ] {
            assert!((actual - expected).length() <= 2e-5 * (1.0 + expected.length()));
        }
    }
    let (row, k, wall, soil, machine) = fixture(20, 1, 0.1, 1.0, true);
    let buffer = input_buffer(&engine, &[row]);
    let source = source.replace(
        "array<f32,6>(2.0,0.15,0.0,2.25,0.0,0.0)",
        "array<f32,6>(2.0,0.5,0.0,0.0,1.3106602,2.017767)",
    );
    let solver =
        ExperimentalSoilWallFrictionGpu::with_source(&engine, &buffer, 1, 64, 96, &source).unwrap();
    let result = run(&engine, &solver);
    println!("corner stagnation input={row:?}; output={:?}", result[0]);
    assert_eq!(result[0].status[0], 0);
    verify(&row, &result[0]);
    for (actual, expected) in [
        (-vector(result[0].machine_impulse_normal), k),
        (vector(result[0].wall_impulse), wall),
        (vector(result[0].soil_velocity_energy), soil),
        (vector(result[0].machine_velocity_residual), machine),
    ] {
        assert!((actual - expected).length() <= 2e-5 * (1.0 + expected.length()));
    }
}
#[test]
#[ignore = "requires GPU"]
fn wall_friction_warm_seed_replay() {
    let engine = Engine::new_gpu().unwrap();
    let input = bytemuck::pod_read_unaligned::<SoilWallFrictionInput>(bytemuck::cast_slice(&[
        1041964683_u32,
        1005055991,
        1000742958,
        0,
        1005055991,
        1040339853,
        3163645366,
        0,
        1000742958,
        3163645366,
        1040448986,
        0,
        1058167891,
        3187196580,
        1025700917,
        1050253722,
        1050253722,
        3204448256,
        1055353995,
        1056964608,
        2147483648,
        3212836864,
        2147483648,
        1020054733,
        36,
        0,
        0,
        0,
    ]));
    let seed = bytemuck::pod_read_unaligned::<SoilWallFrictionOutput>(bytemuck::cast_slice(&[
        3183813669_u32,
        1051825733,
        3173112197,
        1051825733,
        1056713644,
        3011510272,
        0,
        3188998320,
        1057552149,
        3167538382,
        1015999039,
        842767549,
        0,
        1058589474,
        3196580214,
        0,
        0,
        6,
        0,
        6,
    ]));
    let seed_input = SoilWallFrictionInput {
        inverse_machine: [
            seed.machine_impulse_normal,
            seed.soil_velocity_energy,
            seed.machine_velocity_residual,
        ],
        machine_velocity_friction: seed.wall_impulse,
        boundary: seed.status,
        ..Default::default()
    };
    let buffer = input_buffer(&engine, &[input, seed_input]);
    let source = crate::soil_shared_wall_friction_gpu::register_rounding_source(
        &ExperimentalSoilWallFrictionGpu::shader_source(),
        "params.pad0",
    );
    let make = |source: &str| {
        ExperimentalSoilWallFrictionGpu::with_source(&engine, &buffer, 1, 64, 96, source).unwrap()
    };
    let cold = run(&engine, &make(&source));
    println!("captured conditional cold status={:?}", cold[0].status);
    assert_eq!(cold[0].status[0], 0);
    verify(&input, &cold[0]);
    let unrolled = crate::soil_shared_wall_friction_gpu::unrolled_direction_source(&source);
    let unrolled_output = run(&engine, &make(&unrolled));
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&unrolled_output),
        bytemuck::cast_slice::<_, u32>(&cold)
    );
    let anchor = "    var x:array<f32,6>; var tail:array<f32,6>;";
    let warm = format!("{}\nvar<private> warm_seed:Output;\n", source
        .replace(anchor, &format!("{anchor}\n{}", include_str!("../shaders/soil_wall_warm_seed.wgsl")))
        .replace("    solve_block(inputs[id.x],params.iterations);", "    let seed=inputs[1];\n    warm_seed=Output(seed.a0,seed.a1,seed.a2,seed.machine_mu,seed.boundary);\n    solve_block(inputs[id.x],params.iterations);"));
    let diagnose = |source: &str| {
        source.replace("    if !valid { block_output.status.x=32u; return; }", r#"
    if !valid {
        block_output.status.x=32u;
        var reasons=select(0u,1u,x[0]<0.0)
            |select(0u,2u,separation< -1e-5*velocity_scale)
            |select(0u,4u,abs(x[0]*separation)>1e-5*impulse_scale*velocity_scale)
            |select(0u,8u,norm(tangent)>cap+1e-5*impulse_scale)
            |select(0u,16u,energy>1e-5*impulse_scale*velocity_scale);
        if norm(slip)>1e-5*velocity_scale && norm(tangent+cap*slip/norm(slip))>1e-5*impulse_scale { reasons|=32u; }
        for (var i=3u;i<6u;i++) {
            let speed=dot(axes[i],soil);
            reasons|=select(0u,64u,x[i]<0.0)|select(0u,128u,speed< -1e-5*velocity_scale)
                |select(0u,256u,abs(x[i]*speed)>1e-5*impulse_scale*velocity_scale);
        }
        block_output.machine_impulse_normal=vec4<f32>(-k,x[0]);
        block_output.soil_velocity_energy=vec4<f32>(soil,energy);
        block_output.machine_velocity_residual=vec4<f32>(machine,error);
        block_output.wall_impulse=vec4<f32>(w,0.0);
        block_output.status.w=reasons;
        return;
    }"#)
    };
    let old = "if old_merit<=1e-12 && small_physical_correction(x,delta) { publish(input,x,tail,a,sqrt(old_merit)); return; }";
    let current = "if old_merit<=1e-12 && small_physical_correction(x,delta) { publish(input,x,tail,a,sqrt(old_merit)); if block_output.status.x!=32u { return; } }";
    assert_eq!(warm.matches(current).count(), 1);
    let failed = run(&engine, &make(&diagnose(&warm.replace(current, old))));
    let cold = run(&engine, &make(&diagnose(&source)));
    println!(
        "isolated warm publication: cold={:?}; warm={:?}",
        cold[0], failed[0]
    );
    assert_eq!(failed[0].status[0], 32);
    assert_eq!(failed[0].status[3], 32);
    let corrected = run(&engine, &make(&warm));
    assert_eq!(corrected[0].status[0], 0);
    verify(&input, &corrected[0]);
    println!("corrected warm publication={:?}", corrected[0]);
}

#[test]
#[ignore = "requires GPU"]
fn wall_friction_register_rounding_arithmetic_matches_reference() {
    let engine = Engine::new_gpu().unwrap();
    let mut values = vec![
        (0.0, -0.0),
        (-0.0, -0.0),
        (1.0, f32::EPSILON / 2.0),
        (1.0, -1.0),
        (16777216.0, 1.0),
        (-16777216.0, -1.0),
        (f32::MIN_POSITIVE, -f32::MIN_POSITIVE),
    ];
    let mut seed = 0x31415926_u32;
    let mut sample = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        f32::from_bits((seed & 0x807fffff) | ((67 + (seed >> 24) % 121) << 23))
    };
    for _ in 0..4096 {
        values.push((sample(), sample()));
    }
    let inputs: Vec<_> = values
        .iter()
        .map(|&(a, b)| {
            let mut input = SoilWallFrictionInput::default();
            input.inverse_machine[0] = [a, b, sample(), 0.0];
            input.inverse_machine[1] = [sample(), sample() * 1e-8, sample(), sample() * 1e-8];
            input
        })
        .collect();
    let source = ExperimentalSoilWallFrictionGpu::shader_source().replace(
        "    solve_block(inputs[id.x],params.iterations);\n    outputs[id.x]=block_output;",
        "    let input=inputs[id.x];\n    let sum=two_sum(input.a0.x,input.a0.y);\n    let product=add_product(input.a1.xy,input.a0.z,input.a1.z,input.a1.w);\n    var result=Output();\n    result.machine_impulse_normal=vec4<f32>(sum,product);\n    outputs[id.x]=result;",
    );
    let candidate =
        crate::soil_shared_wall_friction_gpu::register_rounding_source(&source, "params.pad0");
    let buffer = input_buffer(&engine, &inputs);
    let make = |source: &str| {
        ExperimentalSoilWallFrictionGpu::with_source(
            &engine,
            &buffer,
            inputs.len() as u32,
            64,
            u64::MAX,
            source,
        )
        .unwrap()
    };
    let expected = run(&engine, &make(&source));
    let actual = run(&engine, &make(&candidate));
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&actual),
        bytemuck::cast_slice::<_, u32>(&expected)
    );
    for (i, &(a, b)) in values.iter().enumerate() {
        let high = (f64::from(a) + f64::from(b)) as f32;
        let (large, small) = if a.abs() >= b.abs() { (a, b) } else { (b, a) };
        let low = ((f64::from(large) - f64::from(high)) + f64::from(small)) as f32;
        assert_eq!(actual[i].machine_impulse_normal[0], high, "high {i}");
        assert_eq!(actual[i].machine_impulse_normal[1], low, "low {i}");
    }
    println!(
        "{} two-sum/product cases match all output words; independent f64 high/residual checks pass",
        values.len()
    );
}

#[test]
#[ignore = "requires GPU"]
fn wall_friction_manufactured_solutions_and_physical_laws() {
    manufactured_solutions(false);
}

#[test]
#[ignore = "requires GPU"]
fn wall_friction_unrolled_linear_physical_laws() {
    manufactured_solutions(true);
}

fn manufactured_solutions(unroll_linear: bool) {
    let engine = Engine::new_gpu().unwrap();
    let mut fixtures = Vec::new();
    let masks =
        (0..3).flat_map(|x| (0..3).flat_map(move |y| (0..3).map(move |z| x | (y << 2) | (z << 4))));
    for mask in masks {
        for mode in 0..9 {
            for mass in [0.1, 1.0, 100.0] {
                for mu in [0.0, 0.3, 1.0] {
                    for anisotropic in [false, true] {
                        fixtures.push(fixture(mask, mode, mass, mu, anisotropic));
                    }
                }
            }
        }
    }
    let inputs = fixtures.iter().map(|f| f.0).collect::<Vec<_>>();
    let buffer = input_buffer(&engine, &inputs);
    let solver =
        ExperimentalSoilWallFrictionGpu::new(&engine, &buffer, inputs.len() as u32, 64, u64::MAX)
            .unwrap();
    assert_eq!(solver.owned_buffer_bytes(), 80 * inputs.len() as u64 + 16);
    let output = run(&engine, &solver);
    let source = crate::soil_shared_wall_friction_gpu::register_rounding_source(
        &ExperimentalSoilWallFrictionGpu::shader_source(),
        "params.pad0",
    );
    let source = if unroll_linear {
        crate::soil_shared_wall_friction_gpu::unrolled_direction_source(&source)
    } else {
        source
    };
    let candidate = ExperimentalSoilWallFrictionGpu::with_source(
        &engine,
        &buffer,
        inputs.len() as u32,
        64,
        u64::MAX,
        &source,
    )
    .unwrap();
    let candidate_output = run(&engine, &candidate);
    for (index, (actual, expected)) in candidate_output.iter().zip(&output).enumerate() {
        assert_eq!(
            bytemuck::bytes_of(actual),
            bytemuck::bytes_of(expected),
            "case={index} input={:?} actual={actual:?} expected={expected:?}",
            inputs[index]
        );
    }
    let mut failed = Vec::new();
    for (index, row) in output.iter().enumerate() {
        if row.status[0] != 0 {
            failed.push((index, row.status, inputs[index]));
        }
    }
    assert!(
        failed.is_empty(),
        "{} failed blocks; first={:?}",
        failed.len(),
        failed.first()
    );
    let mut max_iterations = 0;
    let mut alternate_branches = 0;
    for (index, (input, k, wall, soil, machine)) in fixtures.iter().enumerate() {
        let result = &output[index];
        verify(input, result);
        let manufactured = refine_f64(input, coordinates(input, *k, *wall))
            .unwrap_or_else(|| panic!("case {index}: manufactured f64 refinement failed"));
        let reference = world_solution(input, manufactured);
        for (actual, expected) in reference.into_iter().zip([*k, *wall, *soil, *machine]) {
            assert!(
                (actual - expected).length() <= 2e-5 * (1.0 + expected.length()),
                "case {index}: manufactured root {actual:?} vs {expected:?}"
            );
        }
        let actual = [
            -vector(result.machine_impulse_normal),
            vector(result.wall_impulse),
            vector(result.soil_velocity_energy),
            vector(result.machine_velocity_residual),
        ];
        let branch = refine_f64(input, coordinates(input, actual[0], actual[1]))
            .unwrap_or_else(|| panic!("case {index}: GPU-branch f64 refinement failed"));
        for (actual, expected) in actual.into_iter().zip(world_solution(input, branch)) {
            assert!(
                (actual - expected).length() <= 2e-5 * (1.0 + expected.length()),
                "case {index}: GPU {actual:?} vs f64 branch {expected:?}"
            );
        }
        if actual.into_iter().zip(reference).any(|(actual, expected)| {
            (actual - expected).length() > 2e-5 * (1.0 + expected.length())
        }) {
            alternate_branches += 1;
        }
        max_iterations = max_iterations.max(result.status[1]);
    }
    assert_eq!(run(&engine, &solver), output);
    println!(
        "{} manufactured wall/contact blocks passed independent f64 physical and branch checks; maximum iterations={max_iterations}; alternate branches={alternate_branches}",
        fixtures.len()
    );
}
#[test]
#[ignore = "requires GPU"]
fn wall_friction_faults_reuse_budget_and_iteration_bound() {
    let engine = Engine::new_gpu().unwrap();
    let valid = fixture(4, 2, 1.0, 0.3, true).0;
    let buffer = input_buffer(&engine, &[valid]);
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(ExperimentalSoilWallFrictionGpu::new(&engine, &buffer, 1, 64, 95).is_err());
    assert!(ExperimentalSoilWallFrictionGpu::new(&engine, &buffer, 1, 0, 96).is_err());
    assert!(ExperimentalSoilWallFrictionGpu::new(&engine, &buffer, 16385, 64, u64::MAX).is_err());
    assert_eq!(scope.finish().observations, 0);
    let solver = ExperimentalSoilWallFrictionGpu::new(&engine, &buffer, 1, 64, 96).unwrap();
    let reference = run(&engine, &solver);
    assert_eq!(reference[0].status[0], 0);
    for kind in 0..6 {
        let mut invalid = valid;
        match kind {
            0 => invalid.machine_velocity_friction[3] = f32::NAN,
            1 => invalid.boundary[0] = 3,
            2 => invalid.soil_velocity_mass[3] = 0.0,
            3 => invalid.inverse_machine[0][1] += 0.1,
            4 => invalid.inverse_machine[0][0] = -1.0,
            _ => invalid.normal_gap[3] = -1.0,
        }
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&invalid));
        let output = run(&engine, &solver);
        assert_ne!(output[0].status[0], 0);
        assert_eq!(output[0].machine_impulse_normal, [0.0; 4]);
        assert_eq!(output[0].wall_impulse, [0.0; 4]);
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&valid));
        assert_eq!(run(&engine, &solver), reference);
    }
    let short = ExperimentalSoilWallFrictionGpu::new(&engine, &buffer, 1, 1, 96).unwrap();
    assert_eq!(run(&engine, &short)[0].status[0], 16);
    let empty = ExperimentalSoilWallFrictionGpu::new(&engine, &buffer, 0, 64, 96).unwrap();
    run(&engine, &empty);
}
