use super::*;

#[path = "mpm_compensated_grid_transfer.rs"]
mod grid_transfer;

fn invalid(message: &str) -> Error {
    Error::Solver(format!("Compensated stiffness: {message}"))
}

fn replace(source: &mut String, old: &str, new: &str) -> Result<()> {
    if source.matches(old).count() != 1 {
        return Err(invalid("shader source contract changed"));
    }
    *source = source.replace(old, new);
    Ok(())
}

fn replace_function(source: &mut String, name: &str, replacement: &str) -> Result<()> {
    let start = source
        .find(&format!("fn {name}("))
        .ok_or_else(|| invalid("missing atomic helper"))?;
    let end = start
        + source[start..]
            .find("\n}\n")
            .ok_or_else(|| invalid("unterminated atomic helper"))?
        + 2;
    source.replace_range(start..end, replacement.trim_end());
    Ok(())
}

pub(super) fn source() -> Result<String> {
    let mut source = include_str!("../shaders/mpm.wgsl").to_owned();
    replace_function(
        &mut source,
        "wg_atomic_add_kdiag",
        include_str!("../shaders/mpm_stiffness_workgroup.wgsl"),
    )?;
    let global = include_str!("../shaders/mpm_stiffness_global.wgsl")
        .replace("@group(1) @binding(12)", "@group(2) @binding(6)");
    replace_function(&mut source, "atomic_add_kdiag", &global)?;
    replace(
        &mut source,
        "    atomicStore(&wg_kdiag[lid.x], 0u);",
        "    atomicStore(&wg_kdiag[lid.x], 0u);\n    stiffness_lane = lid.x;\n    atomicStore(&wg_stiffness_tail[lid.x], 0u);",
    )?;
    replace(
        &mut source,
        "atomic_add_kdiag(cidx, bitcast<f32>(atomicLoad(&wg_kdiag[slot])));",
        "atomic_add_kdiag(cidx, bitcast<f32>(atomicLoad(&wg_kdiag[slot])));\n    scatter_stiffness_global_tail(cidx, bitcast<f32>(atomicLoad(&wg_stiffness_tail[slot])));",
    )?;
    source.push_str("\n@compute @workgroup_size(64)\nfn clear_stiffness(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let world = grid_cell_id(gid);\n    if world >= u.grid_total || !grid_valid() { return; }\n    atomicStore(&stiffness_global_tail[dispatched_grid_index(world, gid)], 0u);\n}\n");
    Ok(source)
}

pub(super) struct Transfer {
    tail: wgpu::Buffer,
    mass: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 3],
    active: bool,
    deferred: bool,
    all_fields: bool,
    two_pass: bool,
}

impl Transfer {
    fn new(sim: &MpmSimulation, all_fields: bool) -> Result<Self> {
        let device = &sim.device;
        let limits = device.limits();
        let bytes = sim.g_mass.size() * if all_fields { 5 } else { 1 };
        if bytes > limits.max_storage_buffer_binding_size
            || bytes > limits.max_buffer_size
            || limits.max_storage_buffers_per_shader_stage < 19
            || limits.max_compute_workgroup_storage_size < if all_fields { 3072 } else { 2048 }
            || (!sim.use_active_grid
                && sim.cfg.grid_size.pow(3).div_ceil(64)
                    > limits.max_compute_workgroups_per_dimension)
        {
            return Err(invalid("device resource limits exceeded"));
        }
        let source = if all_fields {
            grid_transfer::source()?
        } else {
            source()?
        };
        let two_pass = !all_fields || sim.use_two_pass_p2g;
        let tail = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("MPM compensated stiffness tail"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false
            }
        );
        let grid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MPM compensated stiffness grid"),
            entries: &storage_entries(7, false),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&sim.layout_uniforms),
                Some(&sim.layout_particles),
                Some(&grid),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM compensated stiffness"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let constants = [
            ("SPARSE_GRID", f64::from(sim.sparse_grid)),
            ("ACTIVE_GRID", f64::from(sim.use_active_grid)),
            ("DEFER_P2G_LOOKUP", f64::from(sim.use_deferred_p2g_lookup)),
        ];
        let pipelines = [
            "clear_stiffness",
            if two_pass { "p2g_two_pass" } else { "p2g" },
            "resolve_stiffness",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
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
                buffer_entry(6, &tail),
            ],
        });
        Ok(Self {
            tail,
            mass: sim.g_mass.clone(),
            bindings,
            pipelines,
            active: sim.use_active_grid,
            deferred: sim.use_deferred_p2g_lookup,
            all_fields,
            two_pass,
        })
    }

    fn matches(&self, sim: &MpmSimulation) -> bool {
        self.mass == sim.g_mass
            && self.active == sim.use_active_grid
            && self.deferred == sim.use_deferred_p2g_lookup
            && (!self.all_fields || self.two_pass == sim.use_two_pass_p2g)
    }

    pub(super) fn encode(&self, pass: &mut wgpu::ComputePass<'_>, sim: &MpmSimulation) {
        pass.set_bind_group(2, &self.bindings, &[]);
        for (stage, pipeline) in self.pipelines.iter().enumerate() {
            pass.set_pipeline(pipeline);
            if stage == 1 {
                pass.dispatch_workgroups(sim.particle_count.div_ceil(64), 1, 1);
            } else if self.active {
                pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
            } else {
                pass.dispatch_workgroups(sim.cfg.grid_size.pow(3).div_ceil(64), 1, 1);
            }
            molla_core::gpu_metrics::record_dispatch();
        }
        pass.set_bind_group(2, &sim.bg_grid, &[]);
    }
}

impl MpmSimulation {
    /// Prepares non-native two-pass stiffness compensation within a tail-buffer budget.
    /// Retains four bytes per physical grid cell; does not enable the transfer.
    pub fn prepare_compensated_stiffness(&mut self, max_bytes: u64) -> Result<()> {
        self.prepare_compensated_scatter(max_bytes, false)
    }

    /// Prepares mass, momentum and stiffness residuals, 20 bytes per physical cell.
    /// Supports non-native single/two-pass scatter; preparation does not enable it.
    pub fn prepare_compensated_grid_transfer(&mut self, max_bytes: u64) -> Result<()> {
        self.prepare_compensated_scatter(max_bytes, true)
    }

    fn prepare_compensated_scatter(&mut self, max_bytes: u64, all_fields: bool) -> Result<()> {
        if self.native_float_atomics {
            return Err(invalid("native stiffness is unsupported"));
        }
        if self.g_mass.size() * if all_fields { 5 } else { 1 } > max_bytes {
            return Err(invalid("tail-buffer budget exceeded"));
        }
        if self.use_compensated_stiffness
            && self
                .compensated_stiffness
                .as_ref()
                .is_some_and(|t| t.all_fields != all_fields)
        {
            return Err(invalid("disable before changing compensation scope"));
        }
        if self
            .compensated_stiffness
            .as_ref()
            .is_none_or(|t| !t.matches(self) || t.all_fields != all_fields)
        {
            self.compensated_stiffness = Some(Transfer::new(self, all_fields)?);
            self.invalidate_prepared_steps();
        }
        Ok(())
    }

    /// Selects prepared non-native two-pass scatter with clear/scatter/resolve dispatches.
    /// Grid dispatch, deferred lookup and storage growth require disabling it first.
    pub fn set_compensated_stiffness(&mut self, enabled: bool) -> Result<()> {
        self.set_compensated_scatter(enabled, false)
    }

    /// Selects prepared all-field clear/scatter/resolve; grid/routing edits require disable.
    pub fn set_compensated_grid_transfer(&mut self, enabled: bool) -> Result<()> {
        self.set_compensated_scatter(enabled, true)
    }

    fn set_compensated_scatter(&mut self, enabled: bool, all_fields: bool) -> Result<()> {
        if enabled
            && (self.native_float_atomics
                || (!all_fields && !self.use_two_pass_p2g)
                || self.use_deterministic_p2g
                || self
                    .compensated_stiffness
                    .as_ref()
                    .is_none_or(|t| !t.matches(self) || t.all_fields != all_fields))
        {
            return Err(invalid(
                "requires matching prepared non-native scatter and compensation scope",
            ));
        }
        self.use_compensated_stiffness = enabled;
        self.invalidate_prepared_steps();
        Ok(())
    }

    pub fn compensated_stiffness_enabled(&self) -> bool {
        self.use_compensated_stiffness
    }
    pub fn compensated_grid_transfer_enabled(&self) -> bool {
        self.use_compensated_stiffness
            && self
                .compensated_stiffness
                .as_ref()
                .is_some_and(|t| t.all_fields)
    }
    pub fn compensated_stiffness_storage_bytes(&self) -> u64 {
        self.compensated_stiffness
            .as_ref()
            .map_or(0, |t| t.tail.size())
    }

    pub fn compensated_grid_transfer_storage_bytes(&self) -> u64 {
        self.compensated_stiffness
            .as_ref()
            .filter(|t| t.all_fields)
            .map_or(0, |t| t.tail.size())
    }

    /// Releases disabled stiffness tail storage, bindings and pipelines.
    pub fn release_compensated_stiffness(&mut self) -> Result<()> {
        if self.use_compensated_stiffness {
            return Err(invalid("disable before release"));
        }
        if self.compensated_stiffness.take().is_some() {
            self.invalidate_prepared_steps();
        }
        Ok(())
    }

    pub(super) fn validate_compensated_stiffness(&self) -> Result<()> {
        if self.use_compensated_stiffness
            && (self.native_float_atomics
                || (!self.use_two_pass_p2g
                    && self
                        .compensated_stiffness
                        .as_ref()
                        .is_none_or(|t| !t.all_fields))
                || self.use_deterministic_p2g
                || self
                    .compensated_stiffness
                    .as_ref()
                    .is_none_or(|t| !t.matches(self)))
        {
            return Err(invalid("stale transfer configuration"));
        }
        Ok(())
    }
}
