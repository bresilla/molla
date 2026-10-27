use super::*;

impl ResidentStepper {
    pub(super) fn prepare_grouped_refit(
        &mut self,
        device: &wgpu::Device,
        kernel: &TireDemGpuKernel,
    ) {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grouped DEM refit"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_dem_grouped_refit.wgsl").into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grouped DEM refit"),
            bind_group_layouts: &[
                Some(&kernel.layout_sparse_lbvh_build_io),
                Some(&kernel.layout_step_uniforms),
            ],
            immediate_size: 0,
        });
        self.grouped_refit = Some(["reset_top", "refit_groups"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        }));
    }

    pub(super) fn encode_grouped_refit(&self, encoder: &mut wgpu::CommandEncoder) -> bool {
        let Some(pipelines) = &self.grouped_refit else {
            return false;
        };
        let padded = (self.tree.node_count as u32 + 1) / 2;
        let groups = padded / padded.min(64);
        for (index, dispatch) in [(0, (groups - 1).div_ceil(64)), (1, groups)] {
            if dispatch == 0 {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines[index]);
            pass.set_bind_group(0, &self.build_io[self.active], &[]);
            pass.set_bind_group(1, &self.build_params[0].0, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        true
    }
}
