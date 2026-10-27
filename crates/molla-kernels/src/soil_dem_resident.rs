//! Retained DEM radix/LBVH rebuild, ping-pong step and terrain export.

use super::*;
use molla_compute_primitives::{RadixSortDispatch, RadixSortKernel};

#[cfg(test)]
#[path = "soil_dem_refit_diagnostic.rs"]
mod refit_diagnostic;

#[cfg(test)]
#[path = "soil_dem_grouped_refit_tests.rs"]
mod grouped_refit;

#[path = "soil_dem_grouped.rs"]
mod grouped;

#[path = "soil_dem_ordered.rs"]
mod ordered;

pub(super) use crate::soil_gpu_identity::{NativeIdentity, device_identity, queue_identity};

pub(super) struct ResidentStepper {
    states: [TireDemResidentState; 2],
    active: usize,
    tree: TireDemResidentLbvh,
    sort: RadixSortDispatch,
    build_io: [wgpu::BindGroup; 2],
    build_params: Vec<(wgpu::BindGroup, u32)>,
    step_io: [wgpu::BindGroup; 2],
    step_params_group: wgpu::BindGroup,
    step_params: wgpu::Buffer,
    contacts: wgpu::Buffer,
    exports: [crate::PreparedTireDemRaster; 2],
    pipelines: [wgpu::ComputePipeline; 4],
    contact_capacity: usize,
    grain_groups: u32,
    leaf_groups: u32,
    steps_since_rebuild: u32,
    force_rebuild: bool,
    ordered_pipeline: Option<wgpu::ComputePipeline>,
    scene_bounds: ([f32; 3], [f32; 3]),
    #[cfg(test)]
    legacy_refit: bool,
    grouped_refit: Option<[wgpu::ComputePipeline; 2]>,
    #[cfg(test)]
    legacy_tree_refit: bool,
}

pub(super) fn state_view(state: &TireDemResidentState) -> TireDemResidentState {
    TireDemResidentState {
        positions: state.positions.clone(),
        velocities: state.velocities.clone(),
        grain_count: state.grain_count,
    }
}

fn allocate(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer> {
    let limits = device.limits();
    if bytes == 0
        || bytes > limits.max_buffer_size
        || (usage.contains(wgpu::BufferUsages::STORAGE)
            && bytes > limits.max_storage_buffer_binding_size)
    {
        return Err(Error::Solver(
            "resident DEM: buffer capacity exceeds device limits".into(),
        ));
    }
    Ok(molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        }
    ))
}

fn immutable_params(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    value: TireDemStepUniforms,
) -> wgpu::BindGroup {
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("resident DEM immutable build parameters"),
            size: core::mem::size_of::<TireDemStepUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        }
    );
    molla_core::gpu_mapped_upload!(buffer.slice(..)).copy_from_slice(bytemuck::bytes_of(&value));
    buffer.unmap();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resident DEM build parameters"),
        layout,
        entries: &[entry(0, &buffer)],
    })
}

impl TireDemGpuKernel {
    /// Downloads candidate/contact pair sets for at most 256 grains.
    /// The supplied tree must describe the supplied particle positions.
    pub fn audit_resident_lbvh_contacts(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &TireDemResidentState,
        tree: &TireDemResidentLbvh,
        grain_radius: f32,
    ) -> Result<TireDemContactAudit> {
        let n = state.grain_count;
        if n == 0
            || n > 256
            || tree.grain_count != n
            || tree.node_count == 0
            || tree.node_row_count != tree.node_count * 3
            || !grain_radius.is_finite()
            || grain_radius <= 0.0
        {
            return Err(Error::Solver(
                "DEM contact audit: invalid bounded fixture".into(),
            ));
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DEM traversal audit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_dem.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DEM audit layout"),
            bind_group_layouts: &[Some(&self.layout_lbvh_io), Some(&self.layout_step_uniforms)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DEM contact audit"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("step_dem_lbvh"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("AUDIT_CONTACTS", 1.0)],
                ..Default::default()
            },
            cache: None,
        });
        let output = allocate(
            device,
            "DEM audit pair matrix",
            (n * n * 16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let unused = allocate(
            device,
            "DEM audit unused output",
            32,
            wgpu::BufferUsages::STORAGE,
        )?;
        let contacts = allocate(
            device,
            "DEM audit contacts",
            32,
            wgpu::BufferUsages::STORAGE,
        )?;
        let io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DEM contact audit io"),
            layout: &self.layout_lbvh_io,
            entries: &[
                entry(0, &state.positions),
                entry(1, &state.velocities),
                entry(2, &contacts),
                entry(3, &output),
                entry(4, &unused),
                entry(10, &tree.node_rows),
                entry(11, &tree.sorted_indices),
            ],
        });
        let params = immutable_params(
            device,
            &self.layout_step_uniforms,
            TireDemStepUniforms {
                dims: [
                    n as u32,
                    0,
                    tree.node_count as u32,
                    tree.sorted_index_count as u32,
                ],
                params0: [1.0, grain_radius, 0.0, 0.0],
                params1: [0.0; 4],
                gravity: [0.0; 4],
                grid_origin_cell_size: [0.0; 4],
                grid_shape: [0; 4],
            },
        );
        let stage = stage_vec4(device, "DEM audit readback", n * n);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &io, &[]);
            pass.set_bind_group(1, &params, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &stage, 0, (n * n * 16) as u64);
        queue.submit([encoder.finish()]);
        let rows = read_vec4(device, &stage, "DEM contact pairs")?;
        let mut result = TireDemContactAudit {
            candidates: Vec::new(),
            contacts: Vec::new(),
        };
        for (i, row) in rows.into_iter().enumerate() {
            let pair = [(i / n) as u32, (i % n) as u32];
            if row[0] > 0.0 {
                result.candidates.push(pair);
            }
            if row[1] > 0.0 {
                result.contacts.push(pair);
            }
        }
        Ok(result)
    }
}

impl ResidentStepper {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireDemGpuKernel,
        initial: &TireDemResidentState,
        terrain: &TireDemResidentTerrainOutput,
        reference_heights: &[f32],
        scene_min: [f32; 3],
        scene_extent: [f32; 3],
        leaf_size: usize,
        contact_capacity: usize,
    ) -> Result<Self> {
        let n = initial.grain_count;
        let leaf_size = leaf_size.clamp(1, 32);
        let leaves = n.div_ceil(leaf_size);
        let padded = leaves
            .checked_next_power_of_two()
            .ok_or_else(|| Error::Solver("resident DEM: tree capacity overflow".into()))?;
        let nodes = padded
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| Error::Solver("resident DEM: node count overflow".into()))?;
        let max_groups = device.limits().max_compute_workgroups_per_dimension as usize;
        if n == 0
            || n > (1 << 24)
            || nodes > (1 << 24)
            || n.div_ceil(64) > max_groups
            || padded.div_ceil(64) > max_groups
            || contact_capacity == 0
            || contact_capacity > u32::MAX as usize
            || scene_min.iter().any(|v| !v.is_finite())
            || scene_extent.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::Solver(
                "resident DEM: invalid dimensions or bounds".into(),
            ));
        }
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let other = TireDemResidentState {
            positions: allocate(
                device,
                "resident DEM ping-pong positions",
                n as u64 * 16,
                storage,
            )?,
            velocities: allocate(
                device,
                "resident DEM ping-pong velocities",
                n as u64 * 16,
                storage,
            )?,
            grain_count: n,
        };
        let states = [state_view(initial), other];
        let keys = allocate(device, "resident DEM Morton keys", n as u64 * 4, storage)?;
        let sorted_indices =
            allocate(device, "resident DEM sorted indices", n as u64 * 4, storage)?;
        let node_rows = allocate(device, "resident DEM tree rows", nodes as u64 * 48, storage)?;
        let sort_kernel =
            RadixSortKernel::new(Arc::new(device.clone()), Arc::new(queue.clone()), 1)?;
        let sort = sort_kernel.prepare(&keys, &sorted_indices, n)?;
        let build_io = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resident DEM build io"),
                layout: &kernel.layout_sparse_lbvh_build_io,
                entries: &[
                    entry(0, &states[i].positions),
                    entry(3, &node_rows),
                    entry(5, &sorted_indices),
                    entry(9, &keys),
                ],
            })
        });
        let base = TireDemStepUniforms {
            dims: [n as u32, padded as u32, leaf_size as u32, 0],
            params0: [0.0; 4],
            params1: [scene_extent[0], scene_extent[1], scene_extent[2], 0.0],
            gravity: [0.0; 4],
            grid_origin_cell_size: [scene_min[0], scene_min[1], scene_min[2], 0.0],
            grid_shape: [(padded - 1) as u32, 0, nodes as u32, leaves as u32],
        };
        let mut build_params = vec![(
            immutable_params(device, &kernel.layout_step_uniforms, base),
            padded.div_ceil(64) as u32,
        )];
        let mut child_start = padded - 1;
        let mut child_count = padded;
        while child_count > 1 {
            let count = child_count / 2;
            let start = child_start - count;
            let params = TireDemStepUniforms {
                dims: [n as u32, count as u32, 0, 0],
                grid_shape: [start as u32, child_start as u32, nodes as u32, 0],
                ..base
            };
            build_params.push((
                immutable_params(device, &kernel.layout_step_uniforms, params),
                count.div_ceil(64) as u32,
            ));
            child_start = start;
            child_count = count;
        }
        let contacts = allocate(
            device,
            "resident DEM contacts",
            contact_capacity as u64 * 32,
            storage,
        )?;
        let step_params = allocate(
            device,
            "resident DEM step parameters",
            core::mem::size_of::<TireDemStepUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        let step_params_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident DEM step parameters"),
            layout: &kernel.layout_step_uniforms,
            entries: &[entry(0, &step_params)],
        });
        let step_io = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resident DEM step io"),
                layout: &kernel.layout_lbvh_io,
                entries: &[
                    entry(0, &states[i].positions),
                    entry(1, &states[i].velocities),
                    entry(2, &contacts),
                    entry(3, &states[1 - i].positions),
                    entry(4, &states[1 - i].velocities),
                    entry(10, &node_rows),
                    entry(11, &sorted_indices),
                ],
            })
        });
        let reference = allocate(
            device,
            "resident DEM retained reference heights",
            reference_heights.len() as u64 * 4,
            storage,
        )?;
        molla_core::gpu_write_buffer!(
            queue,
            &reference,
            0,
            bytemuck::cast_slice(reference_heights)
        );
        let exports = [
            kernel.prepare_resident_terrain_raster(device, &states[0], &reference, terrain)?,
            kernel.prepare_resident_terrain_raster(device, &states[1], &reference, terrain)?,
        ];
        Ok(Self {
            states,
            active: 0,
            sort,
            build_io,
            build_params,
            step_io,
            step_params_group,
            step_params,
            contacts,
            exports,
            contact_capacity,
            steps_since_rebuild: 0,
            force_rebuild: true,
            ordered_pipeline: None,
            scene_bounds: (scene_min, scene_extent),
            #[cfg(test)]
            legacy_refit: false,
            grouped_refit: None,
            #[cfg(test)]
            legacy_tree_refit: false,
            pipelines: [
                kernel.sparse_lbvh_morton_pipeline.clone(),
                kernel.sparse_lbvh_leaf_pipeline.clone(),
                kernel.sparse_lbvh_internal_pipeline.clone(),
                kernel.lbvh_step_pipeline.clone(),
            ],
            grain_groups: n.div_ceil(64) as u32,
            leaf_groups: padded.div_ceil(64) as u32,
            tree: TireDemResidentLbvh {
                node_rows,
                sorted_indices,
                grain_count: n,
                node_count: nodes,
                node_row_count: nodes * 3,
                sorted_index_count: n,
                node_row_capacity: nodes * 3,
                sorted_index_capacity: n,
                leaf_size: leaf_size as u32,
                topology_revision: 0,
            },
        })
    }

    pub(super) fn supports(&self, leaf_size: usize, contacts: usize) -> bool {
        self.tree.leaf_size as usize == leaf_size.clamp(1, 32) && contacts <= self.contact_capacity
    }

    pub(super) fn invalidate_topology(&mut self) {
        self.force_rebuild = true;
    }

    pub(super) fn prepare_ordered(&mut self, device: &wgpu::Device, kernel: &TireDemGpuKernel) {
        if self.ordered_pipeline.is_none() {
            self.ordered_pipeline = Some(ordered::pipeline(device, kernel));
        }
        if self.grouped_refit.is_none() {
            self.prepare_grouped_refit(device, kernel);
        }
    }

    pub(super) fn validate_step(
        contacts: &[TireDemContactCell],
        p: TireDemStepParams,
    ) -> Result<()> {
        let values = [
            p.grain_mass,
            p.grain_radius,
            p.normal_stiffness,
            p.damping,
            p.friction,
            p.floor_y,
            p.gravity[0],
            p.gravity[1],
            p.gravity[2],
            p.dt,
            p.tire_position_push_scale,
        ];
        if values.iter().any(|v| !v.is_finite())
            || p.grain_mass <= 0.0
            || p.grain_radius <= 0.0
            || p.dt < 0.0
            || p.normal_stiffness < 0.0
            || p.damping < 0.0
            || p.friction < 0.0
            || p.tire_position_push_scale < 0.0
            || contacts.len() > u32::MAX as usize
            || contacts.iter().any(|c| {
                c.center_radius
                    .iter()
                    .chain(&c.force)
                    .any(|v| !v.is_finite())
                    || c.center_radius[3] < 0.0
            })
        {
            return Err(Error::Solver(
                "resident DEM: invalid contact or step parameters".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        contacts: &[TireDemContactCell],
        p: TireDemStepParams,
        rebuild_interval: u32,
    ) -> bool {
        if !contacts.is_empty() {
            molla_core::gpu_write_buffer!(queue, &self.contacts, 0, bytemuck::cast_slice(contacts));
        }
        let params = TireDemStepUniforms {
            dims: [
                self.tree.grain_count as u32,
                contacts.len() as u32,
                self.tree.node_count as u32,
                self.tree.sorted_index_count as u32,
            ],
            params0: [
                p.grain_mass,
                p.grain_radius,
                p.dt,
                p.tire_position_push_scale,
            ],
            params1: [p.normal_stiffness, p.damping, p.friction, p.floor_y],
            gravity: [p.gravity[0], p.gravity[1], p.gravity[2], 0.0],
            grid_origin_cell_size: [0.0; 4],
            grid_shape: [0; 4],
        };
        let (minimum, extent) = self.scene_bounds;
        let params = TireDemStepUniforms {
            grid_origin_cell_size: [minimum[0], minimum[1], minimum[2], 0.0],
            grid_shape: [
                extent[0].to_bits(),
                extent[1].to_bits(),
                extent[2].to_bits(),
                0,
            ],
            ..params
        };
        molla_core::gpu_write_buffer!(queue, &self.step_params, 0, bytemuck::bytes_of(&params));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resident DEM complete step"),
        });
        let rebuild = self.force_rebuild || self.steps_since_rebuild >= rebuild_interval;
        if rebuild {
            self.build_pass(&mut encoder, 0, 0, self.grain_groups);
            self.sort.encode(&mut encoder);
        }
        let use_grouped = rebuild_interval > 1;
        #[cfg(test)]
        let use_grouped = use_grouped && !self.legacy_tree_refit;
        let grouped = use_grouped && self.encode_grouped_refit(&mut encoder);
        if !grouped {
            self.build_pass(&mut encoder, 1, 0, self.leaf_groups);
            for i in 1..self.build_params.len() {
                self.build_pass(&mut encoder, 2, i, self.build_params[i].1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let use_ordered = !rebuild;
            #[cfg(test)]
            let use_ordered = use_ordered && !self.legacy_refit;
            pass.set_pipeline(if use_ordered {
                self.ordered_pipeline
                    .as_ref()
                    .expect("prepared ordered DEM refit")
            } else {
                &self.pipelines[3]
            });
            pass.set_bind_group(0, &self.step_io[self.active], &[]);
            pass.set_bind_group(1, &self.step_params_group, &[]);
            pass.dispatch_workgroups(self.grain_groups, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        self.exports[1 - self.active].encode(&mut encoder);
        queue.submit([encoder.finish()]);
        self.active = 1 - self.active;
        self.steps_since_rebuild = if rebuild {
            1
        } else {
            self.steps_since_rebuild + 1
        };
        self.force_rebuild = false;
        rebuild
    }

    fn build_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: usize,
        params: usize,
        groups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipelines[pipeline]);
        pass.set_bind_group(0, &self.build_io[self.active], &[]);
        pass.set_bind_group(1, &self.build_params[params].0, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    pub(super) fn state(&self) -> TireDemResidentState {
        state_view(&self.states[self.active])
    }

    pub(super) fn tree(&self) -> TireDemResidentLbvh {
        TireDemResidentLbvh {
            node_rows: self.tree.node_rows.clone(),
            sorted_indices: self.tree.sorted_indices.clone(),
            grain_count: self.tree.grain_count,
            node_count: self.tree.node_count,
            node_row_count: self.tree.node_row_count,
            sorted_index_count: self.tree.sorted_index_count,
            node_row_capacity: self.tree.node_row_capacity,
            sorted_index_capacity: self.tree.sorted_index_capacity,
            leaf_size: self.tree.leaf_size,
            topology_revision: self.tree.topology_revision,
        }
    }
}
