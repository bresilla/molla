use super::*;
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires protected GPU; direct contact residual reduction oracle"]
fn newton_direct_quality_gpu_ratios() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    assert!(
        device.limits().max_compute_workgroup_storage_size
            >= execution_limits(&engine).1 + SHARED_BYTES
    );
    let buffer = |values: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("direct quality oracle"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let mass = buffer(&[1.0, 0.0, 0.0, 1.0]);
    let rates = buffer(&[0.0, 0.0, 1.0, 0.0]);
    let rows = buffer(&[0.0; 24]);
    let response = buffer(&[0.0; 24]);
    let nodes = buffer(&[0.0; 48]);
    let upstream = buffer(&[0.0]);
    let solver=ExperimentalSoilNewtonGpu::new_with_storage(&engine,
        SoilNewtonInputs {mass:&mass,free_rates:&rates,point_rows:&rows,point_response:&response,contacts:&nodes,upstream_status:&upstream},
        SoilNewtonCapacity {dofs:2,contacts:3,newton_iterations:64,krylov_iterations:64,max_owned_bytes:u64::MAX},
        NewtonVariant::Fischer,|input|format!("{}\n{}",program(input,true),r#"
@compute @workgroup_size(256)
fn quality_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {direct_quality_count=0u;direct_quality_overflow=0u;report.steps=12u;status.active_count=upstream[0];}
    for(var i=lane;i<128u;i+=256u) {direct_quality_records[i]=vec4<f32>(0.0);}
    for(var c=lane;c<3u;c+=256u) {
        active_indices[c]=c;scratch[c]=Six(response[2u*c],response[2u*c+1u]);
        for(var i=0u;i<6u;i++) {blocks[c].residual[i]=rows[2u*c+i/3u][i%3u];}
    }
    storageBarrier();record_direct_quality(value(fischer_product(free_rates[1],rate_add(free_rates[0],vec2<f32>(1e-6,0.0)))));export_direct_quality();
}
"#),"quality_probe",true).unwrap();
    let cases: [([f32; 3], [f32; 3]); 7] = [
        ([3.0, 4.0, 0.0], [0.003, 0.004, 0.0]),
        ([1e30, 0.0, 0.0], [1e24, 0.0, 0.0]),
        ([1e-20, 0.0, 0.0], [1e-21, 0.0, 0.0]),
        ([0.0; 3], [0.0; 3]),
        ([0.0; 3], [1e-30, 0.0, 0.0]),
        ([f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([1e-30, 0.0, 0.0], [1e30, 0.0, 0.0]),
    ];
    for (rhs, error) in cases {
        let denominator = rhs.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        let numerator = error
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let expected = if !denominator.is_finite() || (denominator == 0.0 && numerator > 0.0) {
            (3.402823e38f32) as f64
        } else if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator).min(3.402823e38f32 as f64)
        };
        let mut r = [0.0f32; 24];
        let mut e = [0.0f32; 24];
        r[0] = 1.0;
        for c in [1, 2] {
            r[8 * c..8 * c + 3].copy_from_slice(&rhs);
            e[8 * c..8 * c + 3].copy_from_slice(&error);
        }
        queue.write_buffer(&rows, 0, bytemuck::cast_slice(&r));
        queue.write_buffer(&response, 0, bytemuck::cast_slice(&e));
        for count in [0u32, 3] {
            queue.write_buffer(&upstream, 0, bytemuck::bytes_of(&count));
            let mut previous = None;
            for _ in 0..2 {
                let mut encoder = device.create_command_encoder(&Default::default());
                solver.encode(&mut encoder);
                queue.submit([encoder.finish()]);
                let bytes = capture::read(&engine, solver.precision_local_buffer());
                let trace = decode(&bytes);
                assert_eq!(trace.len(), 1);
                let record = trace[0];
                assert_eq!(record[0], 1e-6);
                assert_eq!(record[3], 12.0);
                let expected = if count == 0 { 0.0 } else { expected };
                assert!((record[1] as f64 - expected).abs() <= 2e-5 * expected.max(1e-30));
                assert_eq!(
                    record[2],
                    if count == 0 {
                        -1.0
                    } else if expected == 0.0 {
                        0.0
                    } else {
                        1.0
                    }
                );
                if let Some(previous) = &previous {
                    assert_eq!(previous, &trace);
                }
                previous = Some(trace);
            }
        }
    }
    println!(
        "28 direct quality GPU reductions pass independent f64 ratios, saturation, ties, empty and exact repeat checks"
    );
}
