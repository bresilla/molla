use super::*;

#[cfg(feature = "resource-counters")]
#[path = "mpm_native_compensated_bench.rs"]
mod bench;

fn source() -> String {
    let mut source = native_compensated::shader_source();
    let start = source.find("fn grid_update(").unwrap();
    let begin = start
        + source[start..]
            .find("    atomicStore(&g_mass[cidx],fma(")
            .unwrap();
    let end = start
        + source[start..]
            .find("    let m_bits = atomicLoad(&g_mass[cidx]);")
            .unwrap();
    let correction = source[begin..end]
        .replace(
            "atomicLoad(&transfer_error[4u*cidx])",
            "atomicExchange(&transfer_error[4u*cidx],0.0)",
        )
        .replace(
            "atomicLoad(&transfer_error[4u*cidx+1u+component])",
            "atomicExchange(&transfer_error[4u*cidx+1u+component],0.0)",
        );
    source.replace_range(begin..end, "");
    source.push_str(&format!("\n@compute @workgroup_size(64) fn compensated_before_contact(@builtin(global_invocation_id) gid:vec3<u32>) {{\nif !grid_valid() {{return;}}\nlet world=grid_cell_id(gid);if world>=u.grid_total {{return;}}\nlet cidx=dispatched_grid_index(world,gid);\n{correction}\n}}\n"));
    source
}

pub(super) struct Transfer {
    error: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 3],
    epoch: Arc<()>,
}

impl Transfer {
    fn new(sim: &MpmSimulation) -> Self {
        let device = &sim.device;
        assert!(sim.native_float_atomics && device.features().contains(wgpu::Features::SHADER_F64));
        assert!(
            matches!(sim.cfg.time_step, MpmTimeStep::Explicit)
                && !sim.use_two_pass_p2g
                && !sim.use_particle_order
                && !sim.use_deterministic_p2g
        );
        let bytes = sim.g_mass.size() * 4;
        let limits = device.limits();
        assert!(
            bytes <= limits.max_storage_buffer_binding_size
                && bytes <= limits.max_buffer_size
                && limits.max_storage_buffers_per_shader_stage >= 19
        );
        let error = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("test pre-contact transfer compensation"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            }
        );
        let grid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &storage_entries(7, false),
        });
        let uniform = sim.pipe_reset.get_bind_group_layout(0);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&uniform),
                Some(&sim.layout_particles),
                Some(&grid),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test pre-contact compensation"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipelines = [
            ("p2g", false),
            ("compensated_before_contact", false),
            ("compensated_before_contact", true),
        ]
        .map(|(entry, active)| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("SPARSE_GRID", f64::from(sim.sparse_grid)),
                        ("ACTIVE_GRID", f64::from(active)),
                    ],
                    ..Default::default()
                },
                cache: None,
            })
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &grid,
            entries: &[
                buffer_entry(0, &sim.g_mass),
                buffer_entry(1, &sim.g_momentum),
                buffer_entry(2, &sim.g_velocity),
                buffer_entry(3, &sim.g_rigid_mask),
                buffer_entry(4, &sim.g_kdiag),
                buffer_entry(5, &sim.g_velocity_prev),
                buffer_entry(6, &error),
            ],
        });
        Self {
            error,
            bindings,
            pipelines,
            epoch: sim.step_epoch.clone(),
        }
    }

    pub(super) fn validate(&self, sim: &MpmSimulation) -> Result<()> {
        if !Arc::ptr_eq(&self.epoch, &sim.step_epoch)
            || sim.use_two_pass_p2g
            || sim.use_particle_order
            || sim.use_deterministic_p2g
        {
            return Err(Error::Solver(
                "Pre-contact compensation is stale or requires single-pass unordered transfer"
                    .into(),
            ));
        }
        Ok(())
    }

    pub(super) fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.error, 0, None);
    }

    pub(super) fn encode(&self, pass: &mut wgpu::ComputePass<'_>, sim: &MpmSimulation) {
        pass.set_bind_group(2, &self.bindings, &[]);
        pass.set_pipeline(&self.pipelines[0]);
        pass.dispatch_workgroups(sim.particle_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
        if sim.use_active_grid {
            pass.set_pipeline(&self.pipelines[2]);
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
        } else {
            pass.set_pipeline(&self.pipelines[1]);
            pass.dispatch_workgroups(sim.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        molla_core::gpu_metrics::record_dispatch();
        pass.set_bind_group(2, &sim.bg_grid, &[]);
    }
}

impl MpmSimulation {
    pub(super) fn install_compensated_active_transfer(&mut self) {
        self.invalidate_prepared_steps();
        self.compensated_active_transfer = Some(Transfer::new(self));
    }
}

#[test]
fn native_precontact_compensation_source_contract() {
    let candidate = source();
    let original = native_compensated::shader_source();
    assert_eq!(
        candidate.matches("atomicExchange(&transfer_error").count(),
        2
    );
    let grid = candidate
        .split_once("fn grid_update(")
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;
    assert!(!grid.contains("transfer_error"));
    let body = |s: &str| {
        let start = s.find("fn p2g(@builtin").unwrap();
        let end = start + s[start..].find("\n}").unwrap() + 2;
        s[start..end].to_owned()
    };
    assert_eq!(body(&candidate), body(&original));
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires native float-atomic/f64 GPU; retained correction lifecycle and conservation"]
fn native_precontact_compensation_gpu_lifecycle() {
    use super::replay_tests::read_words;
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::SHADER_F64,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 32;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.0625;
    cfg.max_particles = 2;
    cfg.max_rigid = 1;
    cfg.youngs = 1000.0;
    cfg.particle_mass = 0.02;
    cfg.particle_volume = 0.0000125;
    for compact in [false, true] {
        let mut sim = if compact {
            MpmSimulation::new_native_sparse(&engine, cfg, 1).unwrap()
        } else {
            MpmSimulation::new_native_explicit(&engine, cfg).unwrap()
        };
        sim.set_particles(queue, &[[-0.2, 0.0, 0.0], [-0.15, 0.01, 0.0]])
            .unwrap();
        let old = sim.prepare_step(0.0001, 1).unwrap();
        sim.install_compensated_active_transfer();
        let mut canceled =
            sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(canceled.encode_step(&old).is_err());
        drop(canceled);
        if compact {
            let before = sim
                .checkpoint_buffers()
                .map(|b| read_words(device, queue, b));
            sim.step_n_with(device, queue, 0.0001, 2).unwrap();
            let status = sim.read_active_grid_status_with(device, queue).unwrap();
            assert!(status.overflow);
            assert_eq!(status.completed_substeps, 0);
            for (before, b) in before.iter().zip(sim.checkpoint_buffers()) {
                assert_eq!(before, &read_words(device, queue, b));
            }
            sim.reserve_sparse_tiles(64).unwrap();
            assert!(sim.prepare_step(0.0001, 1).is_err());
            assert!(sim.step(&engine, 0.0001).is_err());
            sim.install_compensated_active_transfer();
        }
        let bytes = sim
            .compensated_active_transfer
            .as_ref()
            .unwrap()
            .error
            .size();
        assert_eq!(bytes, sim.g_mass.size() * 4);
        let record = sim.prepare_step(0.0001, 3).unwrap();
        let encoder = device.create_command_encoder(&Default::default());
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut batch = sim.begin_command_batch(encoder);
        batch.encode_step(&record).unwrap();
        batch.encode_step(&record).unwrap();
        batch.submit(queue);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = traffic.finish();
        assert_eq!(traffic.readback_bytes, 0);
        assert_eq!(traffic.upload_bytes, 0);
        let status = sim.read_active_grid_status_with(device, queue).unwrap();
        assert!(!status.faulted());
        assert_eq!(status.completed_substeps, 6);
        assert!(
            read_words(
                device,
                queue,
                &sim.compensated_active_transfer.as_ref().unwrap().error
            )
            .iter()
            .all(|&word| word == 0)
        );
        let check = |sim: &mut MpmSimulation, steps: u32| {
            let (mass, _) = sim.read_grid_storage_mass_momentum(&engine).unwrap();
            let total = mass.iter().map(|&v| v as f64).sum::<f64>();
            assert!((total - 2.0 * cfg.particle_mass as f64).abs() <= 2e-5);
            for velocity in sim.read_velocities_with(device, queue).unwrap() {
                assert!(velocity[0].abs() <= 2e-5 && velocity[2].abs() <= 2e-5);
                assert!((velocity[1] - cfg.gravity_y * 0.0001 * steps as f32).abs() <= 2e-5);
            }
        };
        check(&mut sim, 6);
        let poisoned = vec![1e20f32; bytes as usize / 4];
        queue.write_buffer(
            &sim.compensated_active_transfer.as_ref().unwrap().error,
            0,
            bytemuck::cast_slice(&poisoned),
        );
        sim.reset_active_grid();
        sim.step(&engine, 0.0001).unwrap();
        check(&mut sim, 7);
        assert!(
            read_words(
                device,
                queue,
                &sim.compensated_active_transfer.as_ref().unwrap().error
            )
            .iter()
            .all(|&word| word == 0)
        );
        let status = sim.read_active_grid_status_with(device, queue).unwrap();
        assert!(!status.faulted());
        assert_eq!(status.completed_substeps, 1);
        sim.set_p2g_mode(true);
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        assert!(sim.step(&engine, 0.0001).is_err());
        let traffic = traffic.finish();
        assert_eq!(traffic.upload_bytes, 0);
        assert_eq!(traffic.readback_bytes, 0);
        sim.set_p2g_mode(false);
        sim.install_compensated_active_transfer();
        sim.set_time_step(MpmTimeStep::Implicit { jacobi_iters: 1 });
        assert!(sim.prepare_step(0.0001, 1).is_err());
        eprintln!(
            "precontact compact={compact} residual_bytes={bytes}: overflow/stale rejection, six retained substeps, mass/gravity and poisoned-reset recovery pass"
        );
    }
}
