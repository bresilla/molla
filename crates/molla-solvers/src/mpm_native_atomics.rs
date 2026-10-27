use super::*;

#[cfg(test)]
#[path = "mpm_native_active_tests.rs"]
mod tests;

impl MpmSimulation {
    /// Dense explicit MPM using native float32 buffer atomics and global P2G.
    /// Requires an Engine created with `SHADER_FLOAT32_ATOMIC`. Implicit stepping
    /// is rejected; active-tile dispatch is optional and Jacobi stiffness stays zero.
    pub fn new_native_explicit(engine: &Engine, cfg: MpmConfig) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("Native MPM requires a GPU engine".into()))?;
        Self::new_grid_storage(device, cfg, None, true)
    }

    /// Explicit native-atomic MPM with a fixed compact tile pool and active dispatch.
    /// Query active-grid status before accepting advancement; overflow stops particle updates.
    pub fn new_native_sparse(engine: &Engine, cfg: MpmConfig, tile_capacity: u32) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("Native MPM requires a GPU engine".into()))?;
        Self::new_grid_storage(device, cfg, Some(tile_capacity), true)
    }

    pub fn uses_native_float_atomics(&self) -> bool {
        self.native_float_atomics
    }

    /// Selects experimental fixed-order per-body legacy shell-force reduction.
    /// Native explicit grids only; P2G and imprint reactions retain their existing arithmetic.
    #[cfg(test)]
    pub fn set_ordered_native_shell_forces(&mut self, enabled: bool) -> Result<()> {
        if enabled {
            if !self.native_float_atomics {
                return Err(Error::Solver(
                    "Ordered native shell forces require native MPM".into(),
                ));
            }
            self.validate_native_mode()?;
            if self.pipe_ordered_native_shell.is_none() {
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("ordered native shell force"),
                        source: wgpu::ShaderSource::Wgsl(
                            format!(
                                "{}\n{}",
                                shader_source(),
                                include_str!("../shaders/mpm_ordered_native_shell.wgsl")
                            )
                            .into(),
                        ),
                    });
                let uniform = self.pipe_reset.get_bind_group_layout(0);
                let layout = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[
                            Some(&uniform),
                            Some(&self.layout_particles),
                            Some(&self.layout_grid),
                            Some(&self.layout_rigid),
                        ],
                        immediate_size: 0,
                    });
                self.pipe_ordered_native_shell = Some(self.device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("ordered native shell force"),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: Some("ordered_native_shell"),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &[("SPARSE_GRID", f64::from(self.sparse_grid))],
                            ..Default::default()
                        },
                        cache: None,
                    },
                ));
            }
        }
        if self.ordered_native_shell_forces != enabled {
            self.ordered_native_shell_forces = enabled;
            self.invalidate_prepared_steps();
        }
        Ok(())
    }

    pub(super) fn encode_rigid_shell(&self, pass: &mut wgpu::ComputePass<'_>, active: bool) {
        if let Some(reduction) = self.rigid_wrench_reduction() {
            reduction.shell(pass, self, active);
            return;
        }
        #[cfg(test)]
        if self.ordered_native_shell_forces {
            let x = self
                .rigid_count
                .min(self.device.limits().max_compute_workgroups_per_dimension);
            pass.set_pipeline(self.pipe_ordered_native_shell.as_ref().unwrap());
            pass.dispatch_workgroups(x, self.rigid_count.div_ceil(x), 1);
            molla_core::gpu_metrics::record_dispatch();
            return;
        }
        if active {
            pass.set_pipeline(&self.active_pipelines[5]);
            pass.dispatch_workgroups_indirect(&self.active_tiles.indirect, 0);
        } else {
            pass.set_pipeline(&self.pipe_rigid_force);
            pass.dispatch_workgroups(
                self.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE).max(1),
                1,
                1,
            );
        }
        molla_core::gpu_metrics::record_dispatch();
    }

    pub(super) fn validate_native_mode(&self) -> Result<()> {
        self.validate_compensated_stiffness()?;
        #[cfg(test)]
        if let Some(transfer) = &self.compensated_active_transfer {
            transfer.validate(self)?;
        }
        if self.native_float_atomics && !matches!(self.cfg.time_step, MpmTimeStep::Explicit) {
            return Err(Error::Solver(
                "Native float-atomic MPM requires explicit stepping".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn shader_source() -> String {
    let source = include_str!("../shaders/mpm.wgsl");
    assert_eq!(source.matches("alias GridAtomicScalar = u32;").count(), 1);
    let mut source = source.replace(
        "alias GridAtomicScalar = u32;",
        "alias GridAtomicScalar = f32;",
    );
    for (name, body) in [
        ("atomic_add_mass", "atomicAdd(&g_mass[idx], val);"),
        ("atomic_add_mom", "atomicAdd(&g_momentum[idx], val);"),
        (
            "atomic_add_rigid_wrench",
            "atomicAdd(&rigid_wrench[idx], val);",
        ),
        ("atomic_add_kdiag", ""),
        ("wg_atomic_add_kdiag", ""),
    ] {
        let signature = format!("fn {name}(");
        assert_eq!(source.matches(&signature).count(), 1);
        let start = source.find(&signature).unwrap();
        let end = start + source[start..].find("\n}").unwrap() + 2;
        source.replace_range(
            start..end,
            &format!("fn {name}(idx: u32, val: f32) {{\n    {body}\n}}"),
        );
    }
    source
}
