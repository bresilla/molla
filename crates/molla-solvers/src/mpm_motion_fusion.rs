use super::*;

fn fused_source() -> String {
    let mut source = super::source();
    for entry in ["g2p", "soil_motion_plasticity"] {
        let old = format!(
            "@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)\nfn {entry}(@builtin(global_invocation_id) gid: vec3<u32>)"
        );
        assert_eq!(source.matches(&old).count(), 1);
        source = source.replace(&old, &format!("fn fused_{entry}(gid: vec3<u32>)"));
    }
    source.push_str("\n@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)\nfn soil_motion_gather_projection(@builtin(global_invocation_id) gid: vec3<u32>) {\n    fused_g2p(gid);\n    fused_soil_motion_plasticity(gid);\n}\n");
    source
}

impl MpmSimulation {
    #[cfg(test)]
    pub(crate) fn inject_fused_motion_fault_at(&mut self, counter: u32) -> wgpu::ComputePipeline {
        let guard = self.motion_guard.as_ref().expect("prepared guard");
        assert!(guard.fuse_projection);
        let original = guard.fusion.as_ref().unwrap().clone();
        let uniforms = guard.pipeline.get_bind_group_layout(0);
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("test delayed fused motion fault"),
                bind_group_layouts: &[
                    Some(&uniforms),
                    Some(&self.layout_particles),
                    Some(&self.layout_grid),
                    Some(&self.layout_rigid),
                ],
                immediate_size: 0,
            });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("test delayed fused motion fault"),
                source: wgpu::ShaderSource::Wgsl(delayed_fault_source(counter).into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("test delayed fused motion fault"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("soil_motion_gather_projection"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("SPARSE_GRID", f64::from(self.sparse_grid))],
                    ..Default::default()
                },
                cache: None,
            });
        self.motion_guard.as_mut().unwrap().fusion = Some(pipeline);
        original
    }

    #[cfg(test)]
    pub(crate) fn restore_fused_motion_pipeline(&mut self, pipeline: wgpu::ComputePipeline) {
        self.motion_guard.as_mut().unwrap().fusion = Some(pipeline);
    }

    /// Selects fused guarded G2P/plasticity proposals with a separate commit dispatch.
    /// Requires prepared motion bounds and rejects the projection cache. Preparation
    /// retains a pipeline only; guard storage and its byte budget are unchanged.
    /// Disabled motion bounds bypass this selection without dropping the pipeline.
    pub fn set_soil_motion_projection_fused(&mut self, enabled: bool) -> Result<()> {
        if enabled {
            let guard = self.motion_guard.as_ref().ok_or_else(|| {
                Error::Solver("prepare motion bounds before proposal fusion".into())
            })?;
            if guard.bounds.is_none()
                || guard
                    .quiescence
                    .as_ref()
                    .is_some_and(|q| q.projection_cache.is_some())
            {
                return Err(Error::Solver(
                    "proposal fusion requires active bounds and no projection cache".into(),
                ));
            }
            if guard.fusion.is_none() {
                let uniforms = guard.pipeline.get_bind_group_layout(0);
                let layout = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("soil fused guarded proposals"),
                        bind_group_layouts: &[
                            Some(&uniforms),
                            Some(&self.layout_particles),
                            Some(&self.layout_grid),
                            Some(&self.layout_rigid),
                        ],
                        immediate_size: 0,
                    });
                let shader = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("soil fused guarded proposals"),
                        source: wgpu::ShaderSource::Wgsl(fused_source().into()),
                    });
                let pipeline =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("soil fused guarded proposals"),
                            layout: Some(&layout),
                            module: &shader,
                            entry_point: Some("soil_motion_gather_projection"),
                            compilation_options: wgpu::PipelineCompilationOptions {
                                constants: &[("SPARSE_GRID", f64::from(self.sparse_grid))],
                                ..Default::default()
                            },
                            cache: None,
                        });
                self.motion_guard.as_mut().unwrap().fusion = Some(pipeline);
            }
        }
        if let Some(guard) = &mut self.motion_guard {
            if guard.fuse_projection != enabled {
                guard.fuse_projection = enabled;
                self.invalidate_prepared_steps();
            }
        }
        Ok(())
    }

    /// Retained proposal-fusion selection; inactive motion bounds bypass it.
    pub fn soil_motion_projection_fused(&self) -> bool {
        self.motion_guard
            .as_ref()
            .is_some_and(|g| g.fuse_projection)
    }

    pub(in crate::mpm) fn particle_publication_stages(&self) -> [MpmProfileStage; 2] {
        if self.soil_motion_bound_enabled() && self.soil_motion_projection_fused() {
            [
                MpmProfileStage::GuardedGatherProjection,
                MpmProfileStage::GuardedParticleCommit,
            ]
        } else {
            [MpmProfileStage::GridToParticle, MpmProfileStage::Plasticity]
        }
    }
}

#[cfg(test)]
fn delayed_fault_source(counter: u32) -> String {
    let input = fused_source();
    let original = "if pid == 0u { atomicAdd(&g_kdiag[grid_metadata() + 3u], 1u); }";
    assert_eq!(input.matches(original).count(), 1);
    input.replace(original, &format!("if pid == 0u {{ let previous = atomicAdd(&g_kdiag[grid_metadata() + 3u], 1u); if previous == {counter}u {{ atomicOr(&g_kdiag[grid_metadata() + 2u], 4u); return; }} }}"))
}

#[test]
fn fused_motion_delayed_fault_source_contract() {
    for counter in [0, 17, u32::MAX] {
        let source = delayed_fault_source(counter);
        let original = "if pid == 0u { atomicAdd(&g_kdiag[grid_metadata() + 3u], 1u); }";
        let injected = format!(
            "if pid == 0u {{ let previous = atomicAdd(&g_kdiag[grid_metadata() + 3u], 1u); if previous == {counter}u {{ atomicOr(&g_kdiag[grid_metadata() + 2u], 4u); return; }} }}"
        );
        assert_eq!(source.matches(&injected).count(), 1);
        assert_eq!(source.replace(&injected, original), fused_source());
    }
}

#[cfg(test)]
#[path = "mpm_motion_fusion_tests.rs"]
mod tests;
