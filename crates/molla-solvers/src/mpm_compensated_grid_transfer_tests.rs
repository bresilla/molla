use super::*;
use crate::mpm::replay_tests::read_words;

#[test]
fn compensated_grid_source_contract() {
    let source = source().unwrap();
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(source.matches("transfer_error(old,val,next)").count(), 8);
    assert_eq!(source.matches("@group(2) @binding(6)").count(), 1);
}

const PROBE: &str = "
override TEST_LOCAL:bool=false;
@compute @workgroup_size(64)
fn transfer_probe(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lid:u32) {
    transfer_lane=lid;
    atomicStore(&wg_mass[lid],0u);atomicStore(&wg_mx[lid],0u);atomicStore(&wg_my[lid],0u);atomicStore(&wg_mz[lid],0u);atomicStore(&wg_kdiag[lid],0u);
    for(var k=0u;k<5u;k++) {atomicStore(&transfer_local_residual[5u*lid+k],0u);}
    workgroupBarrier();
    if gid.x<u.particle_count {
        let v=p_pos[gid.x];let k=p_volume[gid.x];
        if TEST_LOCAL {wg_atomic_add_mass(0u,v.x);wg_atomic_add_mx(0u,v.y);wg_atomic_add_my(0u,v.z);wg_atomic_add_mz(0u,v.w);wg_atomic_add_kdiag(0u,k);}
        else {atomic_add_mass(0u,v.x);atomic_add_mom(0u,v.y);atomic_add_mom(1u,v.z);atomic_add_mom(2u,v.w);atomic_add_kdiag(0u,k);}
    }
    workgroupBarrier();
    if TEST_LOCAL && lid==0u {
        atomic_add_mass(0u,bitcast<f32>(atomicLoad(&wg_mass[0])));atomic_add_mom(0u,bitcast<f32>(atomicLoad(&wg_mx[0])));atomic_add_mom(1u,bitcast<f32>(atomicLoad(&wg_my[0])));atomic_add_mom(2u,bitcast<f32>(atomicLoad(&wg_mz[0])));atomic_add_kdiag(0u,bitcast<f32>(atomicLoad(&wg_kdiag[0])));
        for(var k=0u;k<5u;k++) {transfer_global_tail(k,bitcast<f32>(atomicLoad(&transfer_local_residual[k])));}
    }
}
";

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; signed all-field residuals and explicit retained-budget lifecycle"]
fn compensated_grid_gpu_atomic_oracle_and_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let config = MpmConfig {
        max_particles: 257,
        max_rigid: 1,
        grid_size: 4,
        ..MpmConfig::sand()
    };
    let mut sim = MpmSimulation::new(&engine, config).unwrap();
    sim.set_particles(queue, &vec![[0.0; 3]; 257]).unwrap();
    for two_pass in [false, true] {
        sim.set_p2g_mode(two_pass);
        let budget = sim.g_mass.size() * 5;
        let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(sim.prepare_compensated_grid_transfer(budget - 1).is_err());
        assert_eq!(resources.finish().observations, 0);
        sim.prepare_compensated_grid_transfer(budget).unwrap();
        assert_eq!(sim.compensated_stiffness_storage_bytes(), budget);
        assert!(!sim.compensated_grid_transfer_enabled());
        assert!(sim.set_compensated_stiffness(true).is_err());
        sim.set_compensated_grid_transfer(true).unwrap();
        assert!(sim.prepare_compensated_stiffness(u64::MAX).is_err());
        assert!(sim.release_compensated_stiffness().is_err());
        let transfer = sim.compensated_stiffness.as_ref().unwrap();
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&sim.layout_uniforms),
                Some(&sim.layout_particles),
                Some(&transfer.pipelines[0].get_bind_group_layout(2)),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(format!("{}{PROBE}", source().unwrap()).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("transfer_probe"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("TEST_LOCAL", f64::from(two_pass))],
                ..Default::default()
            },
            cache: None,
        });
        for case in 0..8 {
            let n = [63, 64, 65, 129, 257, 257, 257, 257][case];
            let mut values = Vec::new();
            let mut stiffness = Vec::new();
            let mut expected = [0.0_f64; 5];
            for i in 0..n {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                let v = [
                    1.0 + (i % 17) as f32 * 0.000001,
                    sign * 100000.0 + (i % 3) as f32 * 0.125,
                    sign * (i % 29) as f32 * 0.00003125,
                    (i % 11) as f32 * 0.01,
                ];
                let k = 0.25 + (i % 13) as f32 * 0.0001;
                for (sum, value) in expected.iter_mut().zip(v.into_iter().chain([k])) {
                    *sum += f64::from(value);
                }
                values.push(v);
                stiffness.push(k);
            }
            queue.write_buffer(&sim.p_pos, 0, bytemuck::cast_slice(&values));
            queue.write_buffer(&sim.p_volume, 0, bytemuck::cast_slice(&stiffness));
            let mut u = sim.step_uniforms(0.0001);
            u.particle_count = n as u32;
            queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&u));
            for repeat in 0..4 {
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    sim.bind_step(&mut pass);
                    pass.set_pipeline(&sim.pipe_reset);
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.set_bind_group(2, &transfer.bindings, &[]);
                    pass.set_pipeline(&transfer.pipelines[0]);
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.set_pipeline(&pipeline);
                    pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
                    pass.set_pipeline(&transfer.pipelines[2]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
                let mass = read_words(device, queue, &sim.g_mass);
                let momentum = read_words(device, queue, &sim.g_momentum);
                let k = read_words(device, queue, &sim.g_kdiag);
                let actual = [mass[0], momentum[0], momentum[1], momentum[2], k[0]];
                assert_eq!(
                    actual,
                    expected.map(|v| (v as f32).to_bits()),
                    "two_pass={two_pass} case={case} repeat={repeat}"
                );
            }
        }
        sim.set_compensated_grid_transfer(false).unwrap();
        sim.release_compensated_stiffness().unwrap();
        assert_eq!(sim.compensated_stiffness_storage_bytes(), 0);
    }
    println!(
        "compensated grid: 16 signed oracle cases x4 repeats x5 fields exact rounded f64; budget and scope lifecycle pass"
    );
}
