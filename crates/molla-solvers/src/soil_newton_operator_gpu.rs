//! Retained contact residual, linearization and matrix-free derivative products.

use molla_compute::Engine;
use molla_core::{Error, Result};

/// World-space soil contact and external wall impulses; W lanes must be zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilNewtonVector {
    pub contact: [f32; 4],
    pub wall: [f32; 4],
}

/// High/low impulse pairs for the nonlinear evaluation point.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilNewtonState {
    pub high: SoilNewtonVector,
    pub low: SoilNewtonVector,
}

pub struct SoilNewtonOperatorInputs<'a> {
    /// Eight-byte high/low generalized free-rate pairs.
    pub free_rates: &'a wgpu::Buffer,
    /// Contact-major world point Jacobian columns, 16 bytes per column.
    pub point_rows: &'a wgpu::Buffer,
    /// Verified M^-1 J^T columns, with the same layout as point_rows.
    pub point_response: &'a wgpu::Buffer,
    /// Existing SoilSharedWallContact records; zero mass disables a slot.
    pub contacts: &'a wgpu::Buffer,
    pub state: &'a wgpu::Buffer,
    pub vector: &'a wgpu::Buffer,
}

/// Computes F(x), its local derivative blocks, Jv and J^T v. This is not a
/// contact solve or a physical publication certificate. Response/mass consistency
/// belongs to the caller's validated preparation. Only zero-fault results are valid.
pub struct ExperimentalSoilNewtonOperatorGpu {
    pipelines: [wgpu::ComputePipeline; 6],
    group: wgpu::BindGroup,
    blocks: wgpu::Buffer,
    products: wgpu::Buffer,
    work: wgpu::Buffer,
    contacts: u32,
    bytes: u64,
    #[cfg(test)]
    point_inputs: [wgpu::Buffer; 3],
}

fn invalid() -> Error {
    Error::Solver("soil Newton operator: invalid capacity, usage or device limit".into())
}

impl ExperimentalSoilNewtonOperatorGpu {
    pub fn required_bytes(dofs: u32, contacts: u32) -> Result<u64> {
        if !(1..=32).contains(&dofs) {
            return Err(invalid());
        }
        Ok(304 * u64::from(contacts.max(1)) + 544)
    }

    pub fn new(
        engine: &Engine,
        input: SoilNewtonOperatorInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let c = u64::from(contacts.max(1));
        let n = u64::from(dofs);
        let bytes = Self::required_bytes(dofs, contacts)?;
        let borrowed = [
            input.free_rates,
            input.point_rows,
            input.point_response,
            input.contacts,
            input.state,
            input.vector,
        ];
        let required = [8 * n, 16 * c * n, 16 * c * n, 64 * c, 64 * c, 32 * c];
        if bytes > max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 9
            || limits.max_bindings_per_bind_group < 10
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 256
            || contacts.max(1).div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || borrowed.iter().zip(required).any(|(b, size)| {
                b.size() < size
                    || b.size() > limits.max_storage_buffer_binding_size
                    || !b.usage().contains(wgpu::BufferUsages::STORAGE)
            })
            || [240 * c, 64 * c, 528].iter().any(|&size| {
                size > limits.max_buffer_size || size > limits.max_storage_buffer_binding_size
            })
        {
            return Err(invalid());
        }
        let create = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false
                }
            )
        };
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let blocks = create("soil Newton local derivative blocks", 240 * c, usage);
        let products = create("soil Newton forward and transpose products", 64 * c, usage);
        let work = create("soil Newton status and retained rate pairs", 528, usage);
        let params = create(
            "soil Newton dimensions",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[dofs, contacts, 0, 0])
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil Newton operator layout"),
            entries: &(0..10)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 9 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 6,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil Newton operator pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let source = format!(
            "{}\n{}",
            crate::soil_rate_encoding::arithmetic(64),
            include_str!("../shaders/soil_newton_operator.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil Newton operator"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = [
            "state_rates",
            "linearize",
            "vector_rates",
            "forward_product",
            "transpose_rates",
            "transpose_product",
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
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil Newton operator bindings"),
            layout: &layout,
            entries: &borrowed
                .into_iter()
                .chain([&blocks, &products, &work, &params])
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            pipelines,
            group,
            blocks,
            products,
            work,
            contacts,
            bytes,
            #[cfg(test)]
            point_inputs: [
                input.point_rows.clone(),
                input.point_response.clone(),
                input.contacts.clone(),
            ],
        })
    }

    /// Clears sticky fault and scratch rates before evaluating a new state.
    pub fn encode_reset(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.work, 0, None);
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_linearize(encoder);
        self.encode_forward(encoder);
        self.encode_transpose(encoder);
    }

    /// Rebuilds F(x) and derivative blocks for the current borrowed state.
    pub fn encode_linearize(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_range(encoder, 0..2);
    }

    /// Applies the last linearization to the current borrowed vector.
    pub fn encode_forward(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_range(encoder, 2..4);
    }

    /// Applies the last transposed linearization to the current borrowed vector.
    pub fn encode_transpose(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_range(encoder, 4..6);
    }

    fn encode_range(&self, encoder: &mut wgpu::CommandEncoder, range: std::ops::Range<usize>) {
        for i in range {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[i]);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(
                if i % 2 == 0 {
                    1
                } else {
                    self.contacts.max(1).div_ceil(64)
                },
                1,
                1,
            );
            molla_core::gpu_metrics::record_dispatch();
        }
    }

    /// 240-byte rows: six residuals, 36 row-major direct and 18 machine entries.
    pub fn blocks_buffer(&self) -> &wgpu::Buffer {
        &self.blocks
    }
    /// 64-byte rows: forward SoilNewtonVector followed by transpose SoilNewtonVector.
    pub fn products_buffer(&self) -> &wgpu::Buffer {
        &self.products
    }
    /// First u32: sticky input fault 1 or nonfinite arithmetic fault 2.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.work
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
    #[cfg(test)]
    pub(crate) fn point_inputs(&self) -> [&wgpu::Buffer; 3] {
        std::array::from_fn(|i| &self.point_inputs[i])
    }
}
