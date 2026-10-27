//! Experimental one-normal-per-node condensation with fixed axis-aligned walls.

use crate::soil_normal_projection_gpu::{
    ExperimentalSoilNormalProjectionGpu, SoilNormalCapacity, SoilNormalInputs, hard_storage_bytes,
};
use molla_compute::Engine;
use molla_core::{Error, Result};

pub const FLOOR_INPUT: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilFloorNode {
    /// Unit outward contact normal and finite soil mass; zero mass is inactive.
    pub normal_mass: [f32; 4],
    /// Boundary-feasible free velocity and nonnegative gap divided by timestep.
    pub velocity_gap: [f32; 4],
    /// +X,-X,+Y,-Y,+Z,-Z wall mask, followed by three zero words.
    pub boundary: [u32; 4],
}

pub struct SoilFloorInputs<'a> {
    pub mass: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    /// One outward-normal machine Jacobian row per distinct physical node.
    pub rows: &'a wgpu::Buffer,
    pub nodes: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilFloorCapacity {
    pub dofs: u32,
    pub nodes: u32,
    pub max_iterations: u32,
    pub max_owned_bytes: u64,
}

/// Condenses shared 3D soil velocity and wall constraints into hinge potentials.
/// Outputs require zero shared fault status. Fully blocked contact normals impose
/// hard affine machine inequalities alongside the finite hinge potentials.
/// Geometry, friction, state publication and rollback remain caller responsibilities.
pub struct ExperimentalSoilFloorProjectionGpu {
    normal: ExperimentalSoilNormalProjectionGpu,
    pipelines: [wgpu::ComputePipeline; 2],
    group: wgpu::BindGroup,
    velocities: wgpu::Buffer,
    contact_impulses: wgpu::Buffer,
    floor_impulses: wgpu::Buffer,
    nodes: u32,
    bytes: u64,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil floor projection: {message}"))
}

impl ExperimentalSoilFloorProjectionGpu {
    pub fn new(
        engine: &Engine,
        input: SoilFloorInputs<'_>,
        capacity: SoilFloorCapacity,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        if !(1..=32).contains(&capacity.dofs)
            || capacity.nodes > 16384
            || !(1..=64).contains(&capacity.max_iterations)
        {
            return Err(invalid("capacity or iteration limit exceeded"));
        }
        let n = u64::from(capacity.dofs);
        let count = u64::from(capacity.nodes.max(1));
        let extra = 112 * count + 16 * count * n + 16;
        let normal_bytes = 4 * n + 8 * u64::from((4 * capacity.nodes).max(1)) + 32;
        let limits = device.limits();
        if extra + normal_bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 9
            || 16 * count * n > limits.max_storage_buffer_binding_size
            || 64 * count > limits.max_storage_buffer_binding_size
            || hard_storage_bytes(capacity.dofs) > limits.max_compute_workgroup_storage_size
            || (4 * capacity.nodes).div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(invalid("device or owned-buffer limit exceeded"));
        }
        for (buffer, required) in [input.mass, input.free_rates, input.rows, input.nodes]
            .into_iter()
            .zip([4 * n * n, 4 * n, 4 * count * n, 48 * count])
        {
            if buffer.size() < required
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid("input capacity or usage mismatch"));
            }
        }
        let buffer = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        };
        let terms = buffer("floor condensed potential terms", 64 * count);
        let term_rows = buffer("floor condensed machine rows", 16 * count * n);
        let velocities = buffer("floor corrected node velocities", 16 * count);
        let contact_impulses = buffer("soil outward contact impulses", 16 * count);
        let floor_impulses = buffer("soil fixed-wall impulses", 16 * count);
        let normal = ExperimentalSoilNormalProjectionGpu::new_with_hard_rows(
            engine,
            SoilNormalInputs {
                mass: input.mass,
                free_rates: input.free_rates,
                rows: &term_rows,
                contacts: &terms,
            },
            SoilNormalCapacity {
                dofs: capacity.dofs,
                contacts: 4 * capacity.nodes,
                max_iterations: capacity.max_iterations,
                max_owned_bytes: normal_bytes,
            },
        )?;
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("floor condensation parameters"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[capacity.dofs, capacity.nodes, 0u32, 0u32])
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil fixed-wall condensation"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_floor_projection.wgsl").into(),
            ),
        });
        let entries: Vec<_> = (0..10)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 9 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: [0, 1, 4].contains(&binding),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil floor projection"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil floor projection"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ["prepare", "recover"].map(|entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let buffers = [
            input.nodes,
            input.rows,
            &terms,
            &term_rows,
            normal.impulses_buffer(),
            &velocities,
            &contact_impulses,
            &floor_impulses,
            normal.status_buffer(),
            &params,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil floor projection"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let bytes = extra + normal.owned_buffer_bytes();
        Ok(Self {
            normal,
            pipelines,
            group,
            velocities,
            contact_impulses,
            floor_impulses,
            nodes: capacity.nodes,
            bytes,
        })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for stage in 0..2 {
            if stage == 1 {
                self.normal.encode(encoder);
            }
            if self.nodes == 0 {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[stage]);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.nodes.div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        self.normal.rates_buffer()
    }
    /// Fault, iterations, backtracks, then packed u16 working-set additions/removals.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        self.normal.status_buffer()
    }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer {
        &self.velocities
    }
    /// XYZ outward impulse on soil; W total normal multiplier. Machine impulse is opposite.
    pub fn contact_impulses_buffer(&self) -> &wgpu::Buffer {
        &self.contact_impulses
    }
    pub fn floor_impulses_buffer(&self) -> &wgpu::Buffer {
        &self.floor_impulses
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }

    #[cfg(test)]
    pub(crate) fn prepare_reference_check(
        &mut self,
        engine: &Engine,
        source: &str,
    ) -> wgpu::Buffer {
        self.normal.prepare_reference_check(engine, source)
    }

    #[cfg(test)]
    pub(crate) fn prepare_failure_capture(
        &mut self,
        engine: &Engine,
    ) -> crate::soil_normal_projection_gpu::replay::NormalReplay {
        self.normal.prepare_failure_capture(engine)
    }
}

#[cfg(test)]
#[path = "soil_floor_projection_tests.rs"]
mod tests;
