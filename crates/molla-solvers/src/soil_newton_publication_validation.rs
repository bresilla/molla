use super::replace;

const INPUTS: &str = "!finite4(node.soil_mass) || !finite4(node.normal_gap) || abs(dot(node.normal_gap.xyz,node.normal_gap.xyz)-1.0)>1e-4 || node.normal_gap.w<0.0\n            || !finite4(node.friction) || node.friction.x<0.0 || any(node.friction.yzw!=vec3<f32>(0.0)) || (node.boundary.x&~63u)!=0u || any(node.boundary.yzw!=vec3<u32>(0u))";
const ROWS: &str = "        for (var i=0u;i<params.width;i++) {\n            let a=rows[c*params.width+i];let b=response[c*params.width+i];\n            if !finite4(a) || !finite4(b) || a.w!=0.0 || b.w!=0.0 {publication_fault(1u,c+1u);}\n        }";
const MACHINE: &str = "        if !all(abs(free_rates[lane])<=vec2<f32>(3.402823e38)) || !finite(value(free_rates[lane])) {publication_fault(1u,0u);}\n        for (var j=0u;j<params.width;j++) {\n            let m=mass[lane*params.width+j];\n            if !finite(m) || (lane==j && m<=0.0) || abs(m-mass[j*params.width+lane])>1e-6*max(1.0,abs(m)) {publication_fault(1u,0u);}\n        }";

fn changes() -> Vec<(String, String)> {
    let mut changes = vec![(
        format!("if {INPUTS} || !valid_six(x.high)"),
        format!("if (!publication_inputs_checked && ({INPUTS})) || !valid_six(x.high)"),
    )];
    for region in [ROWS, MACHINE] {
        changes.push((
            region.to_owned(),
            format!("        if !publication_inputs_checked {{\n{region}\n        }}"),
        ));
    }
    changes.push((
        "return workgroupUniformLoad(&publication_stop)==0u;".to_owned(),
        "let publication_result=workgroupUniformLoad(&publication_stop);publication_inputs_checked=(publication_result&3u)==0u;return publication_result==0u;".to_owned(),
    ));
    changes
}

fn source(mut input: String) -> String {
    for (from, to) in changes() {
        replace(&mut input, &from, &to);
    }
    input.push_str("\nvar<private> publication_inputs_checked:bool=false;\n");
    input
}

pub(super) fn probe(input: String, candidate: bool, edge: bool, poison: bool) -> String {
    use crate::soil_newton_parallel_release::active_publication;
    let mut input =
        super::retained_conditioned::source(super::retained_conditioned::baseline(input));
    if candidate {
        input = source(input);
    }
    if edge {
        input = active_publication::probe_source(input, false);
        let mutate = if poison {
            "if lane==0u && previous && params.contacts>0u {states[0].high.k.w=1.0;}storageBarrier();"
        } else {
            ""
        };
        replace(
            &mut input,
            "storageBarrier();let accepted=certify_candidate(lane);",
            &format!(
                "storageBarrier();let previous=certify_candidate(lane);{mutate}let accepted=certify_candidate(lane);"
            ),
        );
    }
    format!(
        "{}{}{}256\n{input}",
        crate::soil_newton_protected::PREFIX,
        crate::soil_newton_protected::VOLATILE_MARKER,
        crate::soil_newton_width::MARKER
    )
}

#[test]
fn newton_publication_validation_source_contract() {
    let control = super::retained_conditioned::source(super::retained_conditioned::baseline(
        super::shader_source_for(super::NewtonVariant::Fischer),
    ));
    let candidate = source(control.clone());
    let mut restored = candidate.clone();
    replace(
        &mut restored,
        "\nvar<private> publication_inputs_checked:bool=false;\n",
        "",
    );
    for (from, to) in changes().into_iter().rev() {
        replace(&mut restored, &to, &from);
    }
    assert_eq!(restored, control);
    for marker in [
        "var<workgroup>",
        "var<storage",
        "backtrack<16u",
        "attempt==8u",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    assert!(candidate.contains(
        "|| !valid_six(x.high) || !valid_six(x.low) {publication_fault(1u,c+1u);continue;}"
    ));
    for (edge, poison) in [(false, false), (true, false), (true, true)] {
        let shader = probe(
            super::shader_source_for(super::NewtonVariant::Fischer),
            true,
            edge,
            poison,
        );
        crate::soil_newton_protected::compile(&shader, if edge { "norm_probe" } else { "newton" })
            .unwrap();
    }
}

#[test]
#[ignore = "requires protected GPU"]
fn newton_publication_validation_state_and_dispatch_reset() {
    use super::{ExperimentalSoilNewtonGpu as Solver, SoilNewtonCapacity, SoilNewtonInputs};
    use crate::soil_shared_wall_friction_gpu::capture;
    use wgpu::util::DeviceExt;
    let engine =
        molla_compute::Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let buffer = |values: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("publication validation lifecycle"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let mass = buffer(&[1.0]);
    let rates = buffer(&[0.0; 2]);
    let rows = buffer(&[0.0; 4]);
    let upstream = buffer(&[0.0]);
    let mut node = [0.0; 16];
    node[1] = 1.0;
    node[3] = 1.0;
    node[5] = 1.0;
    let contacts = buffer(&node);
    for poison in [false, true] {
        let solvers = [false, true].map(|candidate| {
            Solver::new_with_storage(
                &engine,
                SoilNewtonInputs {
                    mass: &mass,
                    free_rates: &rates,
                    point_rows: &rows,
                    point_response: &rows,
                    contacts: &contacts,
                    upstream_status: &upstream,
                },
                SoilNewtonCapacity {
                    dofs: 1,
                    contacts: 1,
                    newton_iterations: 64,
                    krylov_iterations: 64,
                    max_owned_bytes: u64::MAX,
                },
                super::NewtonVariant::Fischer,
                |input| probe(input, candidate, true, poison),
                "norm_probe",
                true,
            )
            .unwrap()
        });
        for bad in [false, true, false] {
            queue.write_buffer(
                &mass,
                0,
                bytemuck::bytes_of(if bad { &-1.0f32 } else { &1.0f32 }),
            );
            let mut expected = None;
            for solver in &solvers {
                queue.write_buffer(
                    solver.state_buffer(),
                    0,
                    bytemuck::cast_slice(&[0.0f32; 16]),
                );
                let mut encoder = device.create_command_encoder(&Default::default());
                solver.encode(&mut encoder);
                queue.submit([encoder.finish()]);
                let output = [
                    solver.outputs_buffer(),
                    solver.rates_buffer(),
                    solver.publication_status_buffer(),
                    solver.state_buffer(),
                    solver.status_buffer(),
                ]
                .map(|b| capture::read(&engine, b));
                let status: [u32; 8] = bytemuck::pod_read_unaligned(&output[2]);
                assert_eq!(status[0] & 1 != 0, bad || poison);
                if bad || poison {
                    assert!(output[0].iter().chain(&output[1]).all(|&v| v == 0));
                }
                if let Some(expected) = &expected {
                    assert_eq!(&output, expected);
                } else {
                    expected = Some(output);
                }
            }
        }
    }
}
