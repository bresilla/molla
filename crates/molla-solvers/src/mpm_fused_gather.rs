//! Test-only single-dispatch particle gather and constitutive projection.

use super::*;

fn fused_source(source: &str) -> Result<String> {
    let mut output = source.to_owned();
    for entry in ["g2p", "plasticity"] {
        let declaration = format!(
            "@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)\nfn {entry}(@builtin(global_invocation_id) gid: vec3<u32>)"
        );
        if output.matches(&declaration).count() != 1 {
            return Err(Error::Solver(format!(
                "MPM fusion source has no unique {entry} declaration"
            )));
        }
        output = output.replace(&declaration, &format!("fn fused_{entry}(gid: vec3<u32>)"));
    }
    output.push_str(
        "\n@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)\nfn gather_projection(@builtin(global_invocation_id) gid: vec3<u32>) {\n    fused_g2p(gid);\n    fused_plasticity(gid);\n}\n"
    );
    Ok(output)
}

impl MpmSimulation {
    /// Prepare a test-only fused gather/projection pipeline outside stepping.
    /// Reuses the reference function bodies, bindings and specialization; allocates
    /// no particle scratch. The separate dispatch route remains the default.
    pub fn prepare_fused_gather_projection(&mut self) -> Result<()> {
        if self.pipe_fused_gather.is_some() {
            return Ok(());
        }
        let source = fused_source(include_str!("../shaders/mpm.wgsl"))?;
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MPM fused gather/projection"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MPM fused gather/projection"),
                bind_group_layouts: &[
                    Some(&self.layout_uniforms),
                    Some(&self.layout_particles),
                    Some(&self.layout_grid),
                    Some(&self.layout_rigid),
                ],
                immediate_size: 0,
            });
        self.pipe_fused_gather = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("MPM fused gather/projection"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("gather_projection"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("SPARSE_GRID", f64::from(self.sparse_grid))],
                    ..Default::default()
                },
                cache: None,
            },
        ));
        Ok(())
    }

    /// Select an already prepared experimental gather/projection path. Switching
    /// invalidates prepared command records without resetting physical/grid state.
    pub fn set_fused_gather_projection(&mut self, enabled: bool) -> Result<()> {
        if enabled
            && self
                .motion_guard
                .as_ref()
                .is_some_and(|g| g.bounds.is_some())
        {
            return Err(Error::Solver(
                "fused gather cannot bypass swept motion guard".into(),
            ));
        }
        if enabled && self.pipe_fused_gather.is_none() {
            return Err(Error::Solver(
                "MPM fused gather/projection is not prepared".into(),
            ));
        }
        if self.use_fused_gather != enabled {
            self.use_fused_gather = enabled;
            self.invalidate_prepared_steps();
        }
        Ok(())
    }

    pub fn fused_gather_projection_enabled(&self) -> bool {
        self.use_fused_gather
    }

    pub(super) fn encode_gather_projection(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        groups: u32,
        profiled: bool,
        step: usize,
    ) {
        if self.use_fused_gather {
            self.profile_begin_pass(profiled, pass, MpmProfileStage::GatherProjection, step);
            pass.set_pipeline(
                self.pipe_fused_gather
                    .as_ref()
                    .expect("prepared fused gather"),
            );
            pass.dispatch_workgroups(groups, 1, 1);
            self.profile_end_pass(profiled, pass);
        } else {
            self.profile_begin_pass(profiled, pass, MpmProfileStage::GridToParticle, step);
            pass.set_pipeline(&self.pipe_g2p);
            pass.dispatch_workgroups(groups, 1, 1);
            self.profile_end_pass(profiled, pass);
            self.profile_begin_pass(profiled, pass, MpmProfileStage::Plasticity, step);
            pass.set_pipeline(&self.pipe_plastic);
            pass.dispatch_workgroups(groups, 1, 1);
            self.profile_end_pass(profiled, pass);
        }
    }
}

#[cfg(test)]
#[path = "mpm_fused_gather_tests.rs"]
mod tests;
