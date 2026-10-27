#[cfg(all(test, feature = "stagnant-branch-newton"))]
use super::distributed::mixed::adjugate;
use super::*;

const ANCHOR: &str = "fn reduce_parts()->vec2<f32> {\n    workgroupBarrier();";
#[cfg(test)]
const PROBE: &str = "\n@compute @workgroup_size(256)\nfn reduction_probe(@builtin(local_invocation_index) id:u32) {\n    lane=id;rate_lane=id;parts[lane]=free_rates[lane];\n    let total=reduce_parts();condensed[lane]=total;\n    condensed[256u+lane]=fischer_div(free_rates[lane],vec2<f32>(contacts[0].soil_mass.x,0.0));\n}\n";
const GROUPED: &str = "
        for(var grouped_stride=64u;grouped_stride>0u;grouped_stride/=4u) {
            if lane<grouped_stride {
                let low=rate_add(parts[lane],parts[lane+2u*grouped_stride]);
                let high=rate_add(parts[lane+grouped_stride],parts[lane+3u*grouped_stride]);
                parts[lane]=rate_add(low,high);
            }
            workgroupBarrier();
        }
        return parts[0];
";

pub(crate) fn source(mut input: String, shared: bool) -> String {
    assert!(input.contains("// NEWTON_EXECUTION_LANES:256\n"));
    assert!(!input.contains("params.pad0"));
    let policy = if shared { "params.pad0==1u" } else { "true" };
    replace(
        &mut input,
        ANCHOR,
        &format!("{ANCHOR}\n    if {policy} {{{GROUPED}    }}"),
    );
    input
}

#[cfg(any(test, feature = "grouped-eight-newton"))]
const EIGHT: &str = "
        for(var eight_stride=32u;eight_stride>0u;eight_stride/=8u) {
            if lane<eight_stride {
                let a=rate_add(parts[lane],parts[lane+4u*eight_stride]);
                let b=rate_add(parts[lane+eight_stride],parts[lane+5u*eight_stride]);
                let c=rate_add(parts[lane+2u*eight_stride],parts[lane+6u*eight_stride]);
                let d=rate_add(parts[lane+3u*eight_stride],parts[lane+7u*eight_stride]);
                parts[lane]=rate_add(rate_add(a,c),rate_add(b,d));
            }
            workgroupBarrier();
        }
        if lane==0u {
            parts[0]=rate_add(rate_add(parts[0],parts[2]),rate_add(parts[1],parts[3]));
        }
        workgroupBarrier();
        return parts[0];
";

#[cfg(any(test, feature = "grouped-eight-newton"))]
pub(crate) fn source_eight(input: String, shared: bool) -> String {
    let mut input = source(input, false);
    let policy = if shared { "params.pad0==1u" } else { "true" };
    replace(
        &mut input,
        ANCHOR,
        &format!("{ANCHOR}\n    if {policy} {{{EIGHT}    }}"),
    );
    input
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_grouped_eight_source_contract() {
    let baseline = adjugate::baseline(shader_source_for(NewtonVariant::Fischer));
    #[cfg(all(
        feature = "grouped-eight-newton",
        not(feature = "parallel-indices-newton")
    ))]
    assert_eq!(
        ExperimentalSoilNewtonGpu::conditioned_runtime_program(
            shader_source_for(NewtonVariant::Fischer),
            true,
            256,
            SoilNewtonCapacity {
                dofs: 10,
                contacts: 4096,
                newton_iterations: 64,
                krylov_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
        ),
        source_eight(baseline.clone(), false)
    );
    let control = source(baseline.clone(), false);
    let shared = source_eight(baseline.clone(), true);
    assert_eq!(
        shared.replace(&format!("\n    if params.pad0==1u {{{EIGHT}    }}"), ""),
        control
    );
    let mut binary: Vec<_> = (0..256).map(|i| i.to_string()).collect();
    let mut eight = binary.clone();
    let mut stride = 128;
    while stride != 0 {
        for i in 0..stride {
            binary[i] = format!("({}+{})", binary[i], binary[i + stride]);
        }
        stride /= 2;
    }
    let mut stride = 32;
    while stride != 0 {
        for i in 0..stride {
            let pair = |j: usize| {
                format!(
                    "({}+{})",
                    eight[i + j * stride],
                    eight[i + (j + 4) * stride]
                )
            };
            eight[i] = format!("(({}+{})+({}+{}))", pair(0), pair(2), pair(1), pair(3));
        }
        stride /= 8;
    }
    let total = format!("(({}+{})+({}+{}))", eight[0], eight[2], eight[1], eight[3]);
    assert_eq!(binary[0], total);
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            shared.matches(declaration).count()
        );
    }
    crate::soil_newton_protected::compile(&format!("{shared}{PROBE}"), "reduction_probe").unwrap();
    for (name, input) in [
        ("control", control),
        ("candidate", source_eight(baseline, false)),
        ("shared", shared),
    ] {
        let binary = crate::soil_newton_protected::compile(&input, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_GROUPED_SUM_SHADERS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), input).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&binary),
            )
            .unwrap();
        }
    }
}

#[cfg(all(feature = "grouped-sum-newton", feature = "resource-counters"))]
#[test]
#[ignore = "requires GPU; unsupported device rejects before owned allocations"]
fn newton_grouped_sum_runtime_device() {
    use wgpu::util::DeviceExt;
    let engine = Engine::new_gpu().unwrap();
    assert!(!protected_enabled(&engine));
    let input = engine
        .device()
        .unwrap()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unsupported grouped sum input"),
            contents: &[0; 64],
            usage: wgpu::BufferUsages::STORAGE,
        });
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let result = ExperimentalSoilNewtonGpu::new_fischer_conditioned(
        &engine,
        SoilNewtonInputs {
            mass: &input,
            free_rates: &input,
            point_rows: &input,
            point_response: &input,
            contacts: &input,
            upstream_status: &input,
        },
        SoilNewtonCapacity {
            dofs: 1,
            contacts: 1,
            newton_iterations: 64,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        },
    );
    assert!(result.is_err());
    assert_eq!(scope.finish().observations, 0);
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_grouped_sum_source_contract() {
    let control = adjugate::baseline(shader_source_for(NewtonVariant::Fischer));
    #[cfg(all(feature = "grouped-sum-newton", not(feature = "grouped-eight-newton")))]
    assert_eq!(
        ExperimentalSoilNewtonGpu::conditioned_runtime_program(
            shader_source_for(NewtonVariant::Fischer),
            true,
            256,
            SoilNewtonCapacity {
                dofs: 10,
                contacts: 4096,
                newton_iterations: 64,
                krylov_iterations: 64,
                max_owned_bytes: u64::MAX
            },
        ),
        source(control.clone(), false)
    );
    let shared = source(control.clone(), true);
    crate::soil_newton_protected::compile(&format!("{shared}{PROBE}"), "reduction_probe").unwrap();
    assert_eq!(
        shared.replace(&format!("\n    if params.pad0==1u {{{GROUPED}    }}"), ""),
        control
    );
    assert_eq!(
        source(control.clone(), false),
        shared.replace("params.pad0==1u", "true")
    );
    let mut binary: Vec<_> = (0..256).map(|i| i.to_string()).collect();
    let mut grouped = binary.clone();
    let mut stride = 128;
    while stride != 0 {
        for i in 0..stride {
            binary[i] = format!("({}+{})", binary[i], binary[i + stride]);
        }
        stride /= 2;
    }
    let mut stride = 64;
    while stride != 0 {
        for i in 0..stride {
            grouped[i] = format!(
                "(({}+{})+({}+{}))",
                grouped[i],
                grouped[i + 2 * stride],
                grouped[i + stride],
                grouped[i + 3 * stride]
            );
        }
        stride /= 4;
    }
    assert_eq!(binary[0], grouped[0]);
    for declaration in ["var<storage", "var<workgroup>", "@binding("] {
        assert_eq!(
            control.matches(declaration).count(),
            shared.matches(declaration).count()
        );
    }
    for (name, input) in [
        ("control", control.clone()),
        ("candidate", source(control, false)),
        ("shared", shared),
    ] {
        let binary = crate::soil_newton_protected::compile(&input, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_GROUPED_SUM_SHADERS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), input).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&binary),
            )
            .unwrap();
        }
    }
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
#[ignore = "requires protected GPU; exact paired reduction and independent f64 sums"]
fn newton_grouped_sum_gpu_edges() {
    gpu_edges(false);
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
#[ignore = "requires protected GPU; exact paired reduction and independent f64 sums"]
fn newton_grouped_eight_gpu_edges() {
    gpu_edges(true);
}

#[cfg(all(test, feature = "stagnant-branch-newton"))]
fn gpu_edges(eight: bool) {
    use crate::soil_shared_wall_friction_gpu::capture;
    use wgpu::util::DeviceExt;
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let buffer = |count: usize| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grouped sum input"),
            contents: &vec![0; count * 4],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let input = buffer(512);
    let empty = buffer(16);
    queue.write_buffer(&empty, 0, &1f32.to_le_bytes());
    let capacity = SoilNewtonCapacity {
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
                capacity,
                NewtonVariant::Fischer,
                |s| {
                    let mut s = if eight {
                        source_eight(adjugate::baseline(s), true)
                    } else {
                        source(adjugate::baseline(s), true)
                    };
                    s.push_str(PROBE);
                    s
                },
                "reduction_probe",
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
            values[2 * lane + 1] = if case >= 5 {
                sign * ((lane * 7) % 31) as f32 * 2f32.powi((case as i32 % 13) - 31)
            } else {
                0.0
            };
        }
        queue.write_buffer(&input, 0, bytemuck::cast_slice(&values));
        let reference: f64 = values.iter().map(|&v| f64::from(v)).sum();
        let scale: f64 = values.iter().map(|&v| f64::from(v).abs()).sum();
        let mut expected = None;
        for solver in &solvers {
            for _ in 0..2 {
                let mut encoder = device.create_command_encoder(&Default::default());
                solver.encode_reset(&mut encoder);
                solver.encode(&mut encoder);
                queue.submit([encoder.finish()]);
                let bytes = capture::read(&engine, solver.precision_local_buffer());
                let sums = &bytes[..256 * 8];
                if let Some(expected) = &expected {
                    assert_eq!(sums, expected, "case={case}");
                } else {
                    expected = Some(sums.to_vec());
                }
                for pair in sums.chunks_exact(8) {
                    let high = f32::from_le_bytes(pair[..4].try_into().unwrap());
                    let low = f32::from_le_bytes(pair[4..].try_into().unwrap());
                    let actual = f64::from(high) + f64::from(low);
                    assert!(actual.is_finite());
                    if case == 3 {
                        assert!(actual == 0.0 || (actual - reference).abs() <= scale * 1e-12);
                    } else {
                        assert!(
                            (actual - reference).abs() <= scale * 1e-12,
                            "case={case} actual={actual} reference={reference}"
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "grouped sum: 32 paired fixtures, 256 output lanes, two exact repeats; f64 oracle and explicit all-subnormal flush contract pass"
    );
}
