//! Bounded workgroup-resident Krylov solves for frozen contact linearizations.

use molla_compute::Engine;
use molla_core::{Error, Result};

pub struct SoilNewtonKrylovInputs<'a> {
    pub blocks: &'a wgpu::Buffer,
    pub point_rows: &'a wgpu::Buffer,
    pub point_response: &'a wgpu::Buffer,
    pub contacts: &'a wgpu::Buffer,
    pub operator_status: &'a wgpu::Buffer,
}

#[derive(Clone, Copy)]
pub struct SoilNewtonKrylovCapacity {
    pub dofs: u32,
    pub contacts: u32,
    pub iterations: u32,
    /// Relative linear residual bound; this is not a physical contact tolerance.
    pub relative_tolerance: f32,
    pub max_owned_bytes: u64,
}

/// Solves J d = -F with right-preconditioned flexible GMRES or undamped LSQR.
/// Inputs are a frozen, zero-fault SoilNewtonOperator linearization. Only a
/// zero-fault direction meets the requested true linear residual bound.
pub struct ExperimentalSoilNewtonKrylovGpu {
    pipelines: [wgpu::ComputePipeline; 3],
    group: wgpu::BindGroup,
    direction: wgpu::Buffer,
    status: wgpu::Buffer,
    contacts: u32,
    bytes: u64,
    #[cfg(test)]
    scratch: [wgpu::Buffer; 5],
    #[cfg(test)]
    product_pipelines: [wgpu::ComputePipeline; 2],
}

fn invalid() -> Error {
    Error::Solver("soil Newton Krylov: invalid capacity, usage or device limit".into())
}

impl ExperimentalSoilNewtonKrylovGpu {
    pub fn required_bytes(c: SoilNewtonKrylovCapacity) -> Result<u64> {
        if !(1..=32).contains(&c.dofs)
            || !(1..=64).contains(&c.iterations)
            || !c.relative_tolerance.is_finite()
            || c.relative_tolerance <= 0.0
            || c.relative_tolerance >= 1.0
        {
            return Err(invalid());
        }
        Ok((308 + 64 * u64::from(c.iterations)) * u64::from(c.contacts.max(1)) + 48)
    }

    pub fn new(
        engine: &Engine,
        input: SoilNewtonKrylovInputs<'_>,
        capacity: SoilNewtonKrylovCapacity,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let bytes = Self::required_bytes(capacity)?;
        let c = u64::from(capacity.contacts.max(1));
        let n = u64::from(capacity.dofs);
        let borrowed = [
            input.blocks,
            input.point_rows,
            input.point_response,
            input.contacts,
            input.operator_status,
        ];
        let sizes = [
            144 * c,
            64 * c * u64::from(capacity.iterations + 1),
            32 * c,
            32 * c,
            32 * c,
            32,
            16,
            4 * c,
        ];
        if bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 12
            || limits.max_bindings_per_bind_group < 13
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 19200
            || capacity.contacts.max(1).div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || borrowed
                .iter()
                .zip([240 * c, 16 * c * n, 16 * c * n, 64 * c, 4])
                .any(|(b, size)| {
                    b.size() < size
                        || b.size() > limits.max_storage_buffer_binding_size
                        || !b.usage().contains(wgpu::BufferUsages::STORAGE)
                })
            || sizes
                .iter()
                .any(|&s| s > limits.max_buffer_size || s > limits.max_storage_buffer_binding_size)
        {
            return Err(invalid());
        }
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
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
        let pre = create("Newton block inverse", sizes[0], usage);
        let basis = create("Newton Arnoldi basis", sizes[1], usage);
        let vector = create("Newton preconditioned vector", sizes[2], usage);
        let scratch = create("Newton Arnoldi scratch", sizes[3], usage);
        let direction = create("Newton linear direction", sizes[4], usage);
        let status = create("Newton linear status", sizes[5], usage);
        let params = create(
            "Newton linear dimensions",
            sizes[6],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let active = create("Newton compact contact indices", sizes[7], usage);
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[
                capacity.dofs,
                capacity.contacts,
                capacity.iterations,
                capacity.relative_tolerance.to_bits()
            ])
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Newton resident Krylov layout"),
            entries: &(0..13)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 10 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 4 || binding == 12,
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
            label: Some("Newton resident Krylov pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let source = format!(
            "{}\n{}",
            crate::soil_rate_encoding::arithmetic(64),
            include_str!("../shaders/soil_newton_krylov.wgsl")
        );
        #[cfg(test)]
        let source = format!(
            "{source}\n@compute @workgroup_size(64) fn resident_forward(@builtin(local_invocation_index) id:u32) {{lane=id;rate_lane=id;apply_operator();}}\n@compute @workgroup_size(64) fn resident_transpose(@builtin(local_invocation_index) id:u32) {{lane=id;rate_lane=id;apply_transpose();}}"
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Newton resident Krylov"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = ["prepare", "solve", "least_squares"].map(pipeline);
        #[cfg(test)]
        let product_pipelines = ["resident_forward", "resident_transpose"].map(pipeline);
        let buffers = [
            input.blocks,
            input.point_rows,
            input.point_response,
            input.contacts,
            &pre,
            &basis,
            &vector,
            &scratch,
            &direction,
            &status,
            &params,
            &active,
            input.operator_status,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Newton resident Krylov bindings"),
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
        Ok(Self {
            pipelines,
            group,
            direction,
            status,
            contacts: capacity.contacts,
            bytes,
            #[cfg(test)]
            scratch: [pre, basis, vector, scratch, active],
            #[cfg(test)]
            product_pipelines,
        })
    }

    pub fn encode_reset(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.status, 0, None);
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_method(encoder, 1);
    }
    /// Runs undamped, unpreconditioned LSQR from zero with the same iteration
    /// and true residual bounds. This is an independent solve, not an automatic retry.
    pub fn encode_least_squares(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_method(encoder, 2);
    }
    fn encode_method(&self, encoder: &mut wgpu::CommandEncoder, method: usize) {
        encoder.clear_buffer(&self.status, 4, Some(28));
        for i in [0, method] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[i]);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(
                if i == 0 {
                    self.contacts.max(1).div_ceil(64)
                } else {
                    1
                },
                1,
                1,
            );
            molla_core::gpu_metrics::record_dispatch();
        }
    }
    pub fn direction_buffer(&self) -> &wgpu::Buffer {
        &self.direction
    }
    /// Fault, iterations, active count, local inverse fallbacks, relative error,
    /// RHS norm, true residual norm and last recurrence estimate. Faults: input=1, nonfinite=2, residual=8.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
    #[cfg(test)]
    pub(crate) fn scratch_buffers(&self) -> [&wgpu::Buffer; 5] {
        std::array::from_fn(|i| &self.scratch[i])
    }
    #[cfg(test)]
    pub(crate) fn encode_product(&self, encoder: &mut wgpu::CommandEncoder, transpose: bool) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.product_pipelines[usize::from(transpose)]);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
