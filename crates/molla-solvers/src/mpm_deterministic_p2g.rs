//! Opt-in fixed-order dense and active-tile particle-to-grid gather.

use super::*;

#[path = "mpm_gather_compensated_stiffness.rs"]
mod compensated_gather;

#[path = "mpm_gather_shared_chunks.rs"]
mod shared_chunks;

#[path = "mpm_gather_stress_cache.rs"]
mod stress_cache;

#[path = "mpm_gather_grid_reuse.rs"]
mod grid_reuse;

#[cfg(test)]
#[path = "mpm_deterministic_p2g_tests.rs"]
mod tests;

#[path = "mpm_dense_gather_tests.rs"]
#[cfg(test)]
mod dense_tests;

#[cfg(test)]
#[path = "mpm_gather_pruning_tests.rs"]
mod pruning_tests;

#[cfg(test)]
#[path = "mpm_gather_mask_tests.rs"]
mod mask_tests;

#[cfg(test)]
#[path = "mpm_gather_chunk_tests.rs"]
mod chunk_tests;

#[cfg(test)]
#[path = "mpm_gather_stiffness_tests.rs"]
mod stiffness_tests;

#[cfg(test)]
#[path = "mpm_stiffness_contributions_tests.rs"]
mod stiffness_contributions_tests;

#[cfg(test)]
#[path = "mpm_compensated_stiffness_tests.rs"]
mod compensated_tests;

#[cfg(test)]
#[path = "mpm_corrected_transfer_tests.rs"]
mod corrected_transfer_tests;

pub(super) struct Gather {
    cache: wgpu::Buffer,
    stress_keys: Option<wgpu::Buffer>,
    grid_reuse: Option<grid_reuse::Buffers>,
    reuse_pipelines: Option<[wgpu::ComputePipeline; 2]>,
    bindings: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 2],
    dense_groups: Option<[u32; 2]>,
    compensated_stiffness: bool,
    shared_chunks: bool,
}

fn dense_groups(grid: u32, limit: u32) -> Result<[u32; 2]> {
    let count = grid.div_ceil(4).checked_pow(3).filter(|&n| n > 0);
    let Some(count) = count.filter(|_| limit > 0) else {
        return Err(Error::Solver(
            "Invalid dense gather dispatch capacity".into(),
        ));
    };
    let x = count.min(limit);
    let y = count.div_ceil(x);
    if y > limit {
        return Err(Error::Solver("Dense gather exceeds dispatch limits".into()));
    }
    Ok([x, y])
}

impl Gather {
    fn new_shared(sim: &MpmSimulation, compensated: bool) -> Result<Self> {
        if sim.device.limits().max_compute_workgroup_storage_size < shared_chunks::WORKGROUP_BYTES
            || (compensated && sim.native_float_atomics)
        {
            return Err(Error::Solver(
                "Shared gather requires 16 KiB workgroup storage and supported stiffness policy"
                    .into(),
            ));
        }
        let mut gather =
            Self::new_variant_source(sim, true, false, true, &shared_chunks::source(compensated)?)?;
        gather.compensated_stiffness = compensated;
        gather.shared_chunks = true;
        Ok(gather)
    }

    fn new(sim: &MpmSimulation) -> Result<Self> {
        Self::new_variant(sim, true, false, true)
    }

    fn new_compensated(sim: &MpmSimulation) -> Result<Self> {
        if sim.native_float_atomics || sim.device.limits().max_compute_workgroup_storage_size < 992
        {
            return Err(Error::Solver(
                "Compensated gather requires non-native storage and workgroup capacity".into(),
            ));
        }
        let mut gather =
            Self::new_variant_source(sim, true, false, true, &compensated_gather::source()?)?;
        gather.compensated_stiffness = true;
        Ok(gather)
    }

    #[cfg(test)]
    fn new_pruned(sim: &MpmSimulation, prune: bool) -> Result<Self> {
        Self::new_cached(sim, prune, false)
    }

    #[cfg(test)]
    fn new_cached(sim: &MpmSimulation, prune: bool, _cache_base: bool) -> Result<Self> {
        Self::new_variant(sim, prune, _cache_base, false)
    }

    #[cfg(test)]
    fn new_masked(sim: &MpmSimulation) -> Result<Self> {
        Self::new_variant(sim, true, false, true)
    }

    fn new_variant(
        sim: &MpmSimulation,
        prune: bool,
        _cache_base: bool,
        _masked: bool,
    ) -> Result<Self> {
        Self::new_variant_source(
            sim,
            prune,
            _cache_base,
            _masked,
            include_str!("../shaders/mpm_gather_masks.wgsl"),
        )
    }

    fn new_variant_source(
        sim: &MpmSimulation,
        prune: bool,
        _cache_base: bool,
        _masked: bool,
        mask_source: &str,
    ) -> Result<Self> {
        Self::new_variant_cached_source(sim, prune, _cache_base, _masked, mask_source, false, false)
    }

    fn new_retained(
        sim: &MpmSimulation,
        compensated: bool,
        shared: bool,
        cached: bool,
        reuse: bool,
    ) -> Result<Self> {
        if (compensated
            && (sim.native_float_atomics
                || sim.device.limits().max_compute_workgroup_storage_size < 992))
            || (shared
                && sim.device.limits().max_compute_workgroup_storage_size
                    < shared_chunks::WORKGROUP_BYTES)
        {
            return Err(Error::Solver(
                "Cached gather exceeds stiffness/workgroup capabilities".into(),
            ));
        }
        let mask = if shared {
            shared_chunks::source(compensated)?
        } else if compensated {
            compensated_gather::source()?
        } else {
            include_str!("../shaders/mpm_gather_masks.wgsl").to_owned()
        };
        let mask = if reuse {
            grid_reuse::mask_source(mask)?
        } else {
            mask
        };
        let mut gather =
            Self::new_variant_cached_source(sim, true, false, true, &mask, cached, reuse)?;
        gather.compensated_stiffness = compensated;
        gather.shared_chunks = shared;
        Ok(gather)
    }

    fn new_variant_cached_source(
        sim: &MpmSimulation,
        prune: bool,
        _cache_base: bool,
        _masked: bool,
        mask_source: &str,
        cached_stress: bool,
        reuse_grid: bool,
    ) -> Result<Self> {
        let order = sim.particle_order.as_ref().ok_or_else(|| {
            Error::Solver("Deterministic P2G requires prepared particle ordering".into())
        })?;
        let device = &sim.device;
        let bytes = u64::from(sim.cfg.max_particles.max(1)) * 64;
        let key_bytes = u64::from(sim.cfg.max_particles.max(1)) * stress_cache::KEY_BYTES;
        let limits = device.limits();
        if sim.native_float_atomics {
            sim.validate_native_mode()?;
        }
        let dense_groups = if sim.native_float_atomics && !sim.use_active_grid {
            Some(dense_groups(
                sim.cfg.grid_size,
                limits.max_compute_workgroups_per_dimension,
            )?)
        } else {
            None
        };
        if bytes > limits.max_storage_buffer_binding_size
            || bytes > limits.max_buffer_size
            || limits.max_storage_buffers_per_shader_stage
                < 20 + u32::from(cached_stress) + 4 * u32::from(reuse_grid)
            || (cached_stress
                && (key_bytes > limits.max_storage_buffer_binding_size
                    || key_bytes > limits.max_buffer_size))
        {
            return Err(Error::Solver(
                "Deterministic P2G exceeds device buffer/binding limits".into(),
            ));
        }
        if reuse_grid {
            grid_reuse::sizes(sim.cfg.max_particles, sim.cfg.grid_size, &limits)?;
        }
        let cache = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("MPM deterministic stress cache"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false
            }
        );
        let mut entries = storage_entries(10, false);
        entries.extend(storage_entries(11, true).into_iter().skip(10));
        if cached_stress {
            entries.extend(storage_entries(12, false).into_iter().skip(11));
        }
        if reuse_grid {
            entries.extend(storage_entries(16, false).into_iter().skip(12));
        }
        let grid_reuse = if reuse_grid {
            Some(grid_reuse::Buffers::new(sim)?)
        } else {
            None
        };
        let stress_keys = cached_stress.then(|| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("MPM deterministic stress keys"),
                    size: key_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }
            )
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MPM deterministic P2G particles"),
            entries: &entries,
        });
        let uniforms = sim.pipe_reset.get_bind_group_layout(0);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MPM deterministic P2G"),
            bind_group_layouts: &[
                Some(&uniforms),
                Some(&layout),
                Some(&sim.layout_grid),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        let base = if sim.native_float_atomics {
            native_atomics::shader_source()
        } else {
            include_str!("../shaders/mpm.wgsl").to_owned()
        };
        let gather_source = include_str!("../shaders/mpm_deterministic_p2g.wgsl");
        #[cfg(test)]
        let gather_source = if _cache_base {
            include_str!("../shaders/mpm_gather_base_cache.wgsl")
        } else {
            gather_source
        };
        let constants = [
            ("ACTIVE_GRID", f64::from(dense_groups.is_none())),
            ("SPARSE_GRID", f64::from(sim.sparse_grid)),
            ("DENSE_GATHER", f64::from(dense_groups.is_some())),
            ("ZERO_GATHER_STIFFNESS", f64::from(sim.native_float_atomics)),
            ("PRUNE_GATHER_RANGES", f64::from(prune)),
        ];
        #[cfg(test)]
        let constants = {
            let mut constants = constants.to_vec();
            if _cache_base {
                constants.push(("CACHE_GATHER_BASE", 1.0));
            }
            constants
        };
        let source = format!("{base}\n{gather_source}");
        let source = if cached_stress {
            stress_cache::source(source)?
        } else {
            source
        };
        let source = if _masked {
            format!("{source}\n{mask_source}")
        } else {
            source
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM deterministic P2G"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let grid_entry = if _masked {
            "gather_grid_masked"
        } else {
            "gather_grid"
        };
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let pipelines = ["gather_prepare", grid_entry].map(pipeline);
        let reuse_pipelines = reuse_grid.then(|| ["reuse_prepare", "reuse_mark"].map(pipeline));
        let mut entries = vec![
            buffer_entry(0, &sim.p_pos),
            buffer_entry(1, &sim.p_vel),
            buffer_entry(2, &sim.p_c),
            buffer_entry(3, &sim.p_f),
            buffer_entry(4, &sim.p_volume),
            buffer_entry(5, &sim.p_mass),
            buffer_entry(6, &sim.p_jp),
            buffer_entry(7, &sim.p_material),
            buffer_entry(8, &sim.material_table),
            buffer_entry(9, &cache),
            buffer_entry(10, &order.keys),
        ];
        if let Some(keys) = &stress_keys {
            entries.push(buffer_entry(11, keys));
        }
        if let Some(reuse) = &grid_reuse {
            entries.extend([
                buffer_entry(12, &reuse.keys),
                buffer_entry(13, &reuse.nodes),
                buffer_entry(14, &reuse.tiles),
                buffer_entry(15, &reuse.header),
            ]);
        }
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MPM deterministic P2G particles"),
            layout: &layout,
            entries: &entries,
        });
        Ok(Self {
            cache,
            stress_keys,
            grid_reuse,
            reuse_pipelines,
            bindings,
            pipelines,
            dense_groups,
            compensated_stiffness: false,
            shared_chunks: false,
        })
    }

    fn matches_mode(&self, sim: &MpmSimulation) -> bool {
        self.dense_groups.is_some() == (sim.native_float_atomics && !sim.use_active_grid)
    }

    pub(super) fn encode(&self, pass: &mut wgpu::ComputePass<'_>, sim: &MpmSimulation) {
        pass.set_bind_group(1, &self.bindings, &[]);
        pass.set_pipeline(&self.pipelines[0]);
        pass.dispatch_workgroups(sim.particle_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
        if let (Some(reuse), Some(pipelines)) = (&self.grid_reuse, &self.reuse_pipelines) {
            for (pipeline, count) in pipelines
                .iter()
                .zip([reuse.tile_groups, sim.cfg.max_particles.max(1).div_ceil(64)])
            {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(count, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
        }
        pass.set_pipeline(&self.pipelines[1]);
        if let Some([x, y]) = self.dense_groups {
            pass.dispatch_workgroups(x, y, 1);
        } else {
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
        }
        molla_core::gpu_metrics::record_dispatch();
        pass.set_bind_group(1, &sim.bg_particles, &[]);
    }
}

impl MpmSimulation {
    /// Prepares opt-in fixed-order gather over the existing sorted particle index.
    /// Adds 64 bytes per maximum particle for stress scratch; ordering is separate.
    /// Does not enable gather or certify whole-solver reaction repeatability.
    pub fn prepare_deterministic_p2g(&mut self) -> Result<()> {
        self.prepare_deterministic_p2g_with_budget(u64::MAX)
    }

    /// Prepares retained gather scratch under a logical GPU byte budget, excluding
    /// particle ordering and driver/pipeline storage. Budget rejection allocates
    /// nothing and preserves existing prepared state. Default stepping is unchanged.
    pub fn prepare_deterministic_p2g_with_budget(&mut self, max_bytes: u64) -> Result<()> {
        self.prepare_deterministic_p2g_policy(max_bytes, false, false, false, false)
    }

    /// Prepares non-native fixed-order gather with compensated stiffness accumulation.
    /// Retains 64 bytes per maximum particle and adds 256 workgroup bytes.
    pub fn prepare_deterministic_p2g_with_compensated_stiffness(
        &mut self,
        max_bytes: u64,
    ) -> Result<()> {
        self.prepare_deterministic_p2g_policy(max_bytes, true, false, false, false)
    }

    /// Prepares fixed-order gather with shared immutable 64-particle chunks.
    /// Retains 64 bytes per maximum particle and requires 16 KiB workgroup storage.
    /// Compensated stiffness is non-native only. Preparation does not enable gather;
    /// ordinary preparation selects direct loads again. Default stepping is unchanged.
    pub fn prepare_deterministic_p2g_shared_chunks(
        &mut self,
        max_bytes: u64,
        compensated_stiffness: bool,
    ) -> Result<()> {
        self.prepare_deterministic_p2g_policy(max_bytes, compensated_stiffness, true, false, false)
    }

    /// Prepares exact-input constitutive memoization for fixed-order transfer.
    /// Retains 160 bytes per capacity slot; every particle still contributes to
    /// grid mass, momentum and stiffness. Ordinary preparation removes the cache.
    /// Preparation does not enable gather or authorize particle sleep/eviction.
    pub fn prepare_deterministic_p2g_stress_cache(
        &mut self,
        max_bytes: u64,
        compensated_stiffness: bool,
        shared_chunks: bool,
    ) -> Result<()> {
        self.prepare_deterministic_p2g_policy(
            max_bytes,
            compensated_stiffness,
            shared_chunks,
            true,
            false,
        )
    }

    /// Prepares exact-input transfer-tile reuse with all particle contributions.
    /// Retains 176 bytes per capacity slot plus 20 bytes per logical node,
    /// eight bytes per logical 4³ tile and 64 header bytes. Two extra dispatches
    /// invalidate changed old/new footprints. No sleeping or eviction is enabled.
    pub fn prepare_deterministic_p2g_grid_reuse(
        &mut self,
        max_bytes: u64,
        compensated_stiffness: bool,
        shared_chunks: bool,
    ) -> Result<()> {
        self.prepare_deterministic_p2g_policy(
            max_bytes,
            compensated_stiffness,
            shared_chunks,
            false,
            true,
        )
    }

    fn prepare_deterministic_p2g_policy(
        &mut self,
        max_bytes: u64,
        compensated: bool,
        shared: bool,
        cached: bool,
        reuse: bool,
    ) -> Result<()> {
        if compensated && self.native_float_atomics {
            return Err(Error::Solver(
                "Compensated gather requires non-native storage".into(),
            ));
        }
        let required = u64::from(self.cfg.max_particles.max(1))
            * (64 + if cached { stress_cache::KEY_BYTES } else { 0 })
            + if reuse {
                grid_reuse::sizes(
                    self.cfg.max_particles,
                    self.cfg.grid_size,
                    &self.device.limits(),
                )?
                .iter()
                .sum()
            } else {
                0
            };
        if required > max_bytes {
            return Err(Error::Solver(
                "Deterministic P2G scratch budget exceeded".into(),
            ));
        }
        if self.deterministic_p2g.as_ref().is_none_or(|g| {
            !g.matches_mode(self)
                || g.compensated_stiffness != compensated
                || g.shared_chunks != shared
                || g.stress_keys.is_some() != cached
                || g.grid_reuse.is_some() != reuse
        }) {
            self.deterministic_p2g = Some(if cached || reuse {
                Gather::new_retained(self, compensated, shared, cached, reuse)?
            } else if shared {
                Gather::new_shared(self, compensated)?
            } else if compensated {
                Gather::new_compensated(self)?
            } else {
                Gather::new(self)?
            });
            self.invalidate_prepared_steps();
        }
        Ok(())
    }

    pub fn deterministic_p2g_compensated_stiffness_enabled(&self) -> bool {
        self.use_deterministic_p2g
            && self
                .deterministic_p2g
                .as_ref()
                .is_some_and(|g| g.compensated_stiffness)
    }

    /// Selects fixed-order transfer with prepared ordering and matching grid dispatch.
    pub fn set_deterministic_p2g(&mut self, enabled: bool) -> Result<()> {
        if enabled
            && ((!self.use_active_grid && !self.native_float_atomics)
                || !self.use_particle_order
                || self.use_deferred_p2g_lookup
                || self
                    .deterministic_p2g
                    .as_ref()
                    .is_none_or(|g| !g.matches_mode(self)))
        {
            return Err(Error::Solver("Deterministic P2G requires prepared gather, active or native dense grid, ordering and no deferred scatter".into()));
        }
        self.validate_native_mode()?;
        self.use_deterministic_p2g = enabled;
        if enabled {
            self.use_compensated_stiffness = false;
        }
        self.invalidate_prepared_steps();
        Ok(())
    }

    pub fn deterministic_p2g_enabled(&self) -> bool {
        self.use_deterministic_p2g
    }

    pub fn deterministic_p2g_shared_chunks_enabled(&self) -> bool {
        self.use_deterministic_p2g
            && self
                .deterministic_p2g
                .as_ref()
                .is_some_and(|g| g.shared_chunks)
    }
    pub fn deterministic_p2g_storage_bytes(&self) -> u64 {
        self.deterministic_p2g.as_ref().map_or(0, |g| {
            g.cache.size()
                + g.stress_keys.as_ref().map_or(0, wgpu::Buffer::size)
                + g.grid_reuse.as_ref().map_or(0, grid_reuse::Buffers::bytes)
        })
    }

    pub fn deterministic_p2g_stress_cache_enabled(&self) -> bool {
        self.use_deterministic_p2g
            && self
                .deterministic_p2g
                .as_ref()
                .is_some_and(|g| g.stress_keys.is_some())
    }

    pub fn deterministic_p2g_grid_reuse_enabled(&self) -> bool {
        self.use_deterministic_p2g
            && self
                .deterministic_p2g
                .as_ref()
                .is_some_and(|g| g.grid_reuse.is_some())
    }

    /// Releases disabled gather scratch and pipelines; particle ordering remains.
    pub fn release_deterministic_p2g(&mut self) -> Result<()> {
        if self.use_deterministic_p2g {
            return Err(Error::Solver(
                "Disable deterministic P2G before releasing it".into(),
            ));
        }
        if self.deterministic_p2g.take().is_some() {
            self.invalidate_prepared_steps();
        }
        Ok(())
    }
}
