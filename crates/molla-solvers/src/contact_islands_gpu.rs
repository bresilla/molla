//! Deterministic device contact islands over whole articulated response groups.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{RadixSortDispatch, RadixSortKernel};

pub const CONTACT_ISLAND_FAULT_INPUT: u32 = 1 << 20;
pub const CONTACT_ISLAND_FAULT_CAPACITY: u32 = 1 << 21;
pub const NO_CONTACT_ISLAND: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct ContactIslandConfig {
    pub articulation_count: u32,
    /// Original normal-contact row capacity, excluding optional scalar rows.
    pub row_capacity: u32,
    pub island_capacity: u32,
    /// Combined disk/scalar block capacity per island.
    pub contacts_per_island: u32,
}

/// Whole-articulation unions; immovable endpoints use `NO_CONTACT_ISLAND`.
/// Islands are ordered by minimum articulation, then boundary-only scalar row.
/// Packed blocks retain original normal-then-scalar row order.
pub struct ResidentContactIslands {
    pipelines: [wgpu::ComputePipeline; 6],
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    sort: RadixSortDispatch,
    config: ContactIslandConfig,
    work: u32,
    onsets_enabled: bool,
    island_count: wgpu::Buffer,
    contact_counts: wgpu::Buffer,
    contact_indices: wgpu::Buffer,
    row_slots: wgpu::Buffer,
    articulation_islands: wgpu::Buffer,
    roots: wgpu::Buffer,
    articulation_offsets: wgpu::Buffer,
    articulation_indices: wgpu::Buffer,
}

impl ResidentContactIslands {
    /// Rebuild after body/articulation ownership, dependency or capacity changes.
    pub fn new(
        engine: &Engine,
        body_articulations: &[u32],
        dependencies: &[[u32; 2]],
        normal_rows: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ContactIslandConfig,
    ) -> Result<Self> {
        Self::with_scalar_rows(
            engine,
            body_articulations,
            dependencies,
            normal_rows,
            None,
            status,
            config,
        )
    }

    /// Scalar rows follow normal rows in packed indices and retain joint-only islands.
    #[allow(clippy::too_many_arguments)]
    pub fn with_scalar_rows(
        engine: &Engine,
        body_articulations: &[u32],
        dependencies: &[[u32; 2]],
        normal_rows: &wgpu::Buffer,
        scalar_rows: Option<(&wgpu::Buffer, u32)>,
        status: &wgpu::Buffer,
        config: ContactIslandConfig,
    ) -> Result<Self> {
        Self::with_contact_limit(
            engine,
            body_articulations,
            dependencies,
            normal_rows,
            scalar_rows,
            status,
            config,
            config.contacts_per_island,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_contact_limit(
        engine: &Engine,
        body_articulations: &[u32],
        dependencies: &[[u32; 2]],
        normal_rows: &wgpu::Buffer,
        scalar_rows: Option<(&wgpu::Buffer, u32)>,
        status: &wgpu::Buffer,
        config: ContactIslandConfig,
        contact_limit: u32,
    ) -> Result<Self> {
        Self::with_contact_mode(
            engine,
            body_articulations,
            dependencies,
            normal_rows,
            scalar_rows,
            status,
            config,
            contact_limit,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_contact_mode(
        engine: &Engine,
        body_articulations: &[u32],
        dependencies: &[[u32; 2]],
        normal_rows: &wgpu::Buffer,
        scalar_rows: Option<(&wgpu::Buffer, u32)>,
        status: &wgpu::Buffer,
        config: ContactIslandConfig,
        contact_limit: u32,
        impact: bool,
    ) -> Result<Self> {
        Self::build(
            engine,
            body_articulations,
            dependencies,
            normal_rows,
            scalar_rows,
            status,
            config,
            contact_limit,
            impact,
            None,
            None,
            true,
            None,
        )
    }

    /// Retain only whole dynamic groups containing an onset; fixed endpoints do not join groups.
    #[allow(clippy::too_many_arguments)]
    pub fn with_impact_onsets(
        engine: &Engine,
        body_articulations: &[u32],
        dependencies: &[[u32; 2]],
        normal_rows: &wgpu::Buffer,
        scalar_rows: Option<(&wgpu::Buffer, u32)>,
        onsets: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ContactIslandConfig,
        contact_limit: u32,
    ) -> Result<Self> {
        Self::build(
            engine,
            body_articulations,
            dependencies,
            normal_rows,
            scalar_rows,
            status,
            config,
            contact_limit,
            true,
            Some(onsets),
            None,
            true,
            None,
        )
    }

    /// Predicted contact records contain an inclusion flag and normal-speed bits per row.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_predicted_impacts(
        engine: &Engine,
        body_articulations: &[u32],
        dependencies: &[[u32; 2]],
        normal_rows: &wgpu::Buffer,
        scalar_rows: Option<(&wgpu::Buffer, u32)>,
        onsets: &wgpu::Buffer,
        predicted: &wgpu::Buffer,
        require_contact: bool,
        status: &wgpu::Buffer,
        config: ContactIslandConfig,
        contact_limit: u32,
    ) -> Result<Self> {
        Self::build(
            engine,
            body_articulations,
            dependencies,
            normal_rows,
            scalar_rows,
            status,
            config,
            contact_limit,
            true,
            Some(onsets),
            Some(predicted),
            require_contact,
            None,
        )
    }

    /// Exclude whole impact groups before packing friction and scalar rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_excluded_impacts(
        engine: &Engine,
        body_articulations: &[u32],
        dependencies: &[[u32; 2]],
        normal_rows: &wgpu::Buffer,
        scalar_rows: Option<(&wgpu::Buffer, u32)>,
        impact_articulations: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ContactIslandConfig,
        contact_limit: u32,
    ) -> Result<Self> {
        Self::build(
            engine,
            body_articulations,
            dependencies,
            normal_rows,
            scalar_rows,
            status,
            config,
            contact_limit,
            false,
            None,
            None,
            true,
            Some(impact_articulations),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        engine: &Engine,
        body_articulations: &[u32],
        dependencies: &[[u32; 2]],
        normal_rows: &wgpu::Buffer,
        scalar_rows: Option<(&wgpu::Buffer, u32)>,
        status: &wgpu::Buffer,
        config: ContactIslandConfig,
        contact_limit: u32,
        impact: bool,
        onsets: Option<&wgpu::Buffer>,
        predicted: Option<&wgpu::Buffer>,
        require_contact: bool,
        excluded_impacts: Option<&wgpu::Buffer>,
    ) -> Result<Self> {
        let invalid = || Error::Gpu("invalid contact-island capacity, ownership or buffers".into());
        let device = engine.device().ok_or_else(invalid)?;
        let queue = engine.queue().ok_or_else(invalid)?;
        let limits = device.limits();
        let slots = u64::from(config.island_capacity) * u64::from(config.contacts_per_island);
        let scalar_count = scalar_rows.map_or(0, |(_, count)| count);
        let rows = config
            .row_capacity
            .checked_add(scalar_count)
            .ok_or_else(invalid)?;
        let nodes = config
            .articulation_count
            .checked_add(scalar_count)
            .ok_or_else(invalid)?;
        let arts = config.articulation_count.max(1);
        let work = u64::from(rows)
            .max(u64::from(nodes.max(1)))
            .max(slots)
            .max(dependencies.len() as u64);
        if config.row_capacity == 0
            || (scalar_rows.is_some() && scalar_count == 0)
            || config.island_capacity == 0
            || config.contacts_per_island == 0
            || contact_limit == 0
            || contact_limit > config.contacts_per_island
            || nodes > u32::MAX / 2
            || body_articulations.len() > u32::MAX as usize
            || dependencies.len() > u32::MAX as usize
            || work > u32::MAX as u64
            || work.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
            || limits.max_storage_buffers_per_shader_stage < 20
            || body_articulations
                .iter()
                .any(|&a| a != NO_CONTACT_ISLAND && a >= config.articulation_count)
            || dependencies
                .iter()
                .flatten()
                .any(|&a| a >= config.articulation_count)
        {
            return Err(invalid());
        }
        for bytes in [
            u64::from(config.row_capacity) * 128,
            slots * 4,
            u64::from(rows) * 8,
            u64::from(nodes.max(1)) * 8,
            u64::from(arts) * 4,
            body_articulations.len().max(1) as u64 * 4,
            dependencies.len().max(1) as u64 * 8,
            (u64::from(config.island_capacity) + 1) * 4,
        ] {
            if bytes > limits.max_storage_buffer_binding_size {
                return Err(invalid());
            }
        }
        for (buffer, size) in [
            (normal_rows, u64::from(config.row_capacity) * 128),
            (
                scalar_rows.map_or(normal_rows, |(buffer, _)| buffer),
                u64::from(scalar_count) * 32,
            ),
            (status, 4),
            (
                excluded_impacts.or(onsets).unwrap_or(normal_rows),
                u64::from(if excluded_impacts.is_some() {
                    config.articulation_count.max(1)
                } else {
                    config.row_capacity
                }) * 4,
            ),
            (
                predicted.unwrap_or(normal_rows),
                u64::from(config.row_capacity) * 8,
            ),
        ] {
            if buffer.size() < size
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid());
            }
        }
        let alloc = |label, bytes: u64| storage(device, label, bytes.max(4));
        let owners = alloc(
            "contact body articulations",
            body_articulations.len() as u64 * 4,
        );
        if !body_articulations.is_empty() {
            molla_core::gpu_write_buffer!(
                queue,
                &owners,
                0,
                bytemuck::cast_slice(body_articulations)
            );
        }
        let edges = alloc(
            "contact static dependencies",
            (dependencies.len() as u64 * 8).max(8),
        );
        if !dependencies.is_empty() {
            molla_core::gpu_write_buffer!(queue, &edges, 0, bytemuck::cast_slice(dependencies));
        }
        let parents = alloc("contact union parents", u64::from(nodes.max(1)) * 4);
        let selected = alloc("impact selected roots", u64::from(nodes.max(1)) * 4);
        let root_counts = alloc("contact root counts", u64::from(nodes.max(1)) * 8);
        let keys = alloc("contact sort keys", u64::from(rows) * 4);
        let values = alloc("contact sort values", u64::from(rows) * 4);
        let island_count = alloc("contact island count", 4);
        let contact_counts = alloc(
            "contact island row counts",
            u64::from(config.island_capacity) * 4,
        );
        let contact_indices = alloc("contact island row indices", slots * 4);
        let row_slots = alloc("contact row slots", u64::from(rows) * 8);
        let articulation_islands =
            alloc("contact articulation islands", u64::from(nodes.max(1)) * 4);
        let roots = alloc(
            "contact island roots",
            u64::from(config.island_capacity) * 4,
        );
        let articulation_offsets = alloc(
            "contact island articulation offsets",
            (u64::from(config.island_capacity) + 1) * 4,
        );
        let articulation_indices =
            alloc("contact island articulation indices", u64::from(arts) * 4);
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("contact island parameters"),
                size: 48,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            }
        );
        molla_core::gpu_mapped_upload!(params.slice(..)).copy_from_slice(bytemuck::cast_slice(&[
            config.articulation_count,
            body_articulations.len() as u32,
            dependencies.len() as u32,
            config.row_capacity,
            config.island_capacity,
            config.contacts_per_island,
            scalar_count,
            contact_limit,
            u32::from(impact),
            if excluded_impacts.is_some() {
                2
            } else {
                u32::from(onsets.is_some())
            },
            if predicted.is_some() {
                if require_contact { 1 } else { 2 }
            } else {
                0
            },
            0,
        ]));
        params.unmap();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("contact island bindings"),
            entries: &(0..21)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 14 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 3
                                    || binding == 17
                                    || binding == 18
                                    || binding == 20,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect::<Vec<_>>(),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact islands"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/contact_islands.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("contact islands"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = [
            "clear",
            "join",
            "select_onsets",
            "label",
            "compact",
            "scatter",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let buffers = [
            normal_rows,
            &owners,
            &edges,
            status,
            &parents,
            &root_counts,
            &keys,
            &values,
            &island_count,
            &contact_counts,
            &contact_indices,
            &row_slots,
            &articulation_islands,
            &roots,
            &params,
            &articulation_offsets,
            &articulation_indices,
            scalar_rows.map_or(normal_rows, |(buffer, _)| buffer),
            excluded_impacts.or(onsets).unwrap_or(normal_rows),
            &selected,
            predicted.unwrap_or(normal_rows),
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact islands"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let sort = RadixSortKernel::new(device.clone(), queue.clone(), 1)?.prepare(
            &keys,
            &values,
            rows as usize,
        )?;
        Ok(Self {
            pipelines,
            group,
            params,
            sort,
            config,
            work: work as u32,
            onsets_enabled: onsets.is_some(),
            island_count,
            contact_counts,
            contact_indices,
            row_slots,
            articulation_islands,
            roots,
            articulation_offsets,
            articulation_indices,
        })
    }

    pub(crate) fn set_contact_limit(&self, queue: &wgpu::Queue, limit: u32) -> Result<()> {
        if limit > self.config.contacts_per_island {
            return Err(Error::Gpu("invalid contact island normal-row limit".into()));
        }
        molla_core::gpu_write_buffer!(queue, &self.params, 7 * 4, bytemuck::bytes_of(&limit));
        Ok(())
    }

    /// Record ordered graph, stable-sort and packing passes; status stays sticky.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for stage in 0..6 {
            if stage == 2 && !self.onsets_enabled {
                continue;
            }
            if stage == 5 {
                self.sort.encode(encoder);
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines[stage]);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(
                if stage == 4 {
                    1
                } else {
                    self.work.div_ceil(64)
                },
                1,
                1,
            );
            molla_core::gpu_metrics::record_dispatch();
        }
    }

    pub fn config(&self) -> ContactIslandConfig {
        self.config
    }
    pub fn island_count_buffer(&self) -> &wgpu::Buffer {
        &self.island_count
    }
    pub fn contact_counts_buffer(&self) -> &wgpu::Buffer {
        &self.contact_counts
    }
    pub fn contact_indices_buffer(&self) -> &wgpu::Buffer {
        &self.contact_indices
    }
    /// Per original normal-then-scalar row `(island, local_block)`; inactive slots are MAX.
    pub fn row_slots_buffer(&self) -> &wgpu::Buffer {
        &self.row_slots
    }
    pub fn articulation_islands_buffer(&self) -> &wgpu::Buffer {
        &self.articulation_islands
    }
    pub fn roots_buffer(&self) -> &wgpu::Buffer {
        &self.roots
    }
    pub fn articulation_offsets_buffer(&self) -> &wgpu::Buffer {
        &self.articulation_offsets
    }
    pub fn articulation_indices_buffer(&self) -> &wgpu::Buffer {
        &self.articulation_indices
    }
    /// Buffers, bind groups and pipelines, including the retained stable sort.
    pub fn resource_counts(&self) -> [usize; 3] {
        [23, 7, 9]
    }
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }
    )
}
