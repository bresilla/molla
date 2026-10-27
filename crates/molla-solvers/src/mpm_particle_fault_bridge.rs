use super::*;

/// Coupled publication rejection after a latched particle/grid fault.
pub const MPM_COUPLING_PARTICLE_FAULT: u32 = 1 << 31;

pub(super) struct ParticleFaultBridge {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

impl ParticleFaultBridge {
    pub fn new(soil: &MpmSimulation, status: &wgpu::Buffer) -> Self {
        Self::with_fault_bit(soil, status, MPM_COUPLING_PARTICLE_FAULT)
    }

    fn with_fault_bit(soil: &MpmSimulation, status: &wgpu::Buffer, fault_bit: u32) -> Self {
        let device = &soil.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM particle fault publication guard"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mpm_particle_fault_bridge.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MPM particle fault publication guard"),
            layout: None,
            module: &shader,
            entry_point: Some("propagate"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    (
                        "METADATA_WORD",
                        (soil.active_tiles.metadata_offset / 4) as f64,
                    ),
                    ("FAULT_BIT", f64::from(fault_bit)),
                ],
                ..Default::default()
            },
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MPM particle fault publication guard"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: soil.g_kdiag.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: status.as_entire_binding(),
                },
            ],
        });
        Self { pipeline, group }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}

pub(crate) struct MpmMachineFaultBridge {
    bridge: ParticleFaultBridge,
    grid: wgpu::Buffer,
    status: wgpu::Buffer,
    epoch: Arc<()>,
}

impl MpmMachineFaultBridge {
    pub(crate) fn new(soil: &MpmSimulation, status: &wgpu::Buffer) -> Result<Self> {
        if status.size() < 4 || !status.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(Error::Solver(
                "MPM machine fault bridge requires a storage status word".into(),
            ));
        }
        Ok(Self {
            bridge: ParticleFaultBridge::with_fault_bit(soil, status, 32),
            grid: soil.g_kdiag.clone(),
            status: status.clone(),
            epoch: soil.step_epoch.clone(),
        })
    }

    pub(crate) fn matches(&self, soil: &MpmSimulation, status: &wgpu::Buffer) -> bool {
        self.grid == soil.g_kdiag
            && self.status == *status
            && Arc::ptr_eq(&self.epoch, &soil.step_epoch)
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        soil: &MpmSimulation,
    ) -> Result<()> {
        if !self.matches(soil, &self.status) {
            return Err(Error::Solver("stale MPM machine fault bridge".into()));
        }
        if !soil.active_grid_needs_reset {
            self.bridge.encode(encoder);
        }
        Ok(())
    }
}
