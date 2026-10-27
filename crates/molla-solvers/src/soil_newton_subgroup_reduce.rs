use super::distributed::mixed::complementarity;
use super::*;

const MARKER: &str = "// NEWTON_SUBGROUP_REDUCTION_TEST\n";
const ENTRY: &str = "fn newton(@builtin(local_invocation_index) id:u32) {";
const PROBE: &str = "\n@compute @workgroup_size(256)\nfn subgroup_probe(@builtin(local_invocation_index) id:u32,@builtin(subgroup_invocation_id) sg:u32,@builtin(subgroup_size) width:u32) {\n    lane=(id*73u+19u)%256u;rate_lane=id;reduction_subgroup_lane=sg;reduction_subgroup_width=width;parts[lane]=free_rates[lane];\n    let total=reduce_parts();condensed[lane]=total;condensed[256u+lane]=vec2<f32>(f32(width),select(0.0,1.0,subgroupAny(lane==0u)));condensed[512u+lane]=fischer_div(free_rates[lane],vec2<f32>(contacts[0].soil_mass.x,0.0));\n}\n";

pub(crate) fn source(input: String) -> String {
    let mut input = active_indices::source(complementarity::baseline(input), false);
    assert!(input.contains("// NEWTON_EXECUTION_LANES:256\n"));
    replace(
        &mut input,
        "fn reduce_parts()->vec2<f32>",
        "fn reduce_parts_reference()->vec2<f32>",
    );
    replace(
        &mut input,
        ENTRY,
        "fn newton(@builtin(local_invocation_index) id:u32,@builtin(subgroup_invocation_id) sg:u32,@builtin(subgroup_size) width:u32) {\n    reduction_subgroup_lane=sg;reduction_subgroup_width=width;",
    );
    let prefix = crate::soil_newton_protected::PREFIX;
    assert!(input.starts_with(prefix));
    input = input.replacen(prefix, &format!("{prefix}{MARKER}"), 1);
    input.push_str(include_str!("../shaders/soil_newton_subgroup_reduce.wgsl"));
    input
}

#[test]
fn newton_subgroup_reduction_source_contract() {
    let program = source(shader_source_for(NewtonVariant::Fischer));
    crate::soil_newton_protected::compile(&program, "newton").unwrap();
    crate::soil_newton_protected::compile(&format!("{program}{PROBE}"), "subgroup_probe").unwrap();
    let mut reference: Vec<String> = (0..256).map(|n| n.to_string()).collect();
    for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
        for i in 0..stride {
            reference[i] = format!("({}+{})", reference[i], reference[i + stride]);
        }
    }
    let mut candidate: Vec<String> = (0..32)
        .map(|k| {
            format!(
                "((({k}+{})+({}+{}))+(({}+{})+({}+{})))",
                k + 128,
                k + 64,
                k + 192,
                k + 32,
                k + 160,
                k + 96,
                k + 224
            )
        })
        .collect();
    for stride in [16, 8, 4, 2, 1] {
        for i in 0..stride {
            candidate[i] = format!("({}+{})", candidate[i], candidate[i + stride]);
        }
    }
    assert_eq!(reference[0], candidate[0]);
}

#[test]
#[ignore = "requires protected/subgroup GPU; exact Newton reduction, 32-lane subgroup gate"]
fn newton_subgroup_reduction_gpu_edges() {
    use crate::soil_shared_wall_friction_gpu::capture;
    use wgpu::util::DeviceExt;
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::PASSTHROUGH_SHADERS | wgpu::Features::SUBGROUP,
    )
    .unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let make = |n| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("subgroup reduction input"),
            contents: &vec![0u8; n],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let input = make(2048);
    let empty = make(64);
    queue.write_buffer(&empty, 0, &1f32.to_le_bytes());
    let cap = SoilNewtonCapacity {
        dofs: 1,
        contacts: 1,
        newton_iterations: 64,
        krylov_iterations: 64,
        max_owned_bytes: u64::MAX,
    };
    let solvers: Vec<_> = [false, true]
        .into_iter()
        .map(|candidate| {
            let solver = ExperimentalSoilNewtonGpu::new_with_storage(
                &engine,
                SoilNewtonInputs {
                    mass: &empty,
                    free_rates: &input,
                    point_rows: &empty,
                    point_response: &empty,
                    contacts: &empty,
                    upstream_status: &empty,
                },
                cap,
                NewtonVariant::Fischer,
                |s| format!("{}{PROBE}", source(s)),
                "subgroup_probe",
                true,
            )
            .unwrap();
            queue.write_buffer(&solver.scratch[7], 20, &u32::from(candidate).to_le_bytes());
            solver
        })
        .collect();
    for case in 0..32u32 {
        let mut values = vec![0f32; 512];
        for lane in 0..256 {
            let sign = if (lane + case as usize) % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            values[2 * lane] = match case {
                0 => 0.0,
                1 => sign,
                2 => {
                    if lane % 4 == 0 {
                        1e15
                    } else if lane % 4 == 2 {
                        -1e15
                    } else {
                        0.125
                    }
                }
                3 => f32::from_bits((lane + 1) as u32),
                4 => -0.0,
                _ => {
                    sign * ((lane * 17 + case as usize * 13) % 257) as f32
                        * 2f32.powi((case as i32 % 13) - 6)
                }
            };
            if case >= 5 {
                values[2 * lane + 1] =
                    sign * ((lane * 7) % 31) as f32 * 2f32.powi((case as i32 % 13) - 31);
            }
        }
        queue.write_buffer(&input, 0, bytemuck::cast_slice(&values));
        let reference: f64 = values.iter().map(|&v| f64::from(v)).sum();
        let scale: f64 = values.iter().map(|&v| f64::from(v).abs()).sum();
        let mut expected = None;
        for (arm, solver) in solvers.iter().enumerate() {
            for _ in 0..2 {
                let mut encoder = device.create_command_encoder(&Default::default());
                solver.encode_reset(&mut encoder);
                solver.encode(&mut encoder);
                queue.submit([encoder.finish()]);
                let bytes = capture::read(&engine, solver.precision_local_buffer());
                let sums = &bytes[..2048];
                if let Some(expected) = &expected {
                    assert_eq!(sums, expected, "case={case} arm={arm}");
                } else {
                    expected = Some(sums.to_vec());
                }
                let mut selected = 0;
                for record in bytes[2048..4096].chunks_exact(8) {
                    assert_eq!(
                        f32::from_le_bytes(record[..4].try_into().unwrap()),
                        32.0,
                        "requires actual subgroup width 32; candidate not admitted"
                    );
                    selected += f32::from_le_bytes(record[4..].try_into().unwrap()) as u32;
                }
                assert_eq!(selected, 32);
                for pair in sums.chunks_exact(8) {
                    let actual = f64::from(f32::from_le_bytes(pair[..4].try_into().unwrap()))
                        + f64::from(f32::from_le_bytes(pair[4..].try_into().unwrap()));
                    assert!(actual.is_finite());
                    assert!(
                        (case == 3 && actual == 0.0) || (actual - reference).abs() <= scale * 1e-12,
                        "case={case} actual={actual} reference={reference}"
                    );
                }
            }
        }
    }
    println!(
        "subgroup reduction: 32 cases x 256 outputs x 2 arms x 2 repeats exact; permuted logical lanes; subgroup width 32; f64 oracle passed"
    );
}
