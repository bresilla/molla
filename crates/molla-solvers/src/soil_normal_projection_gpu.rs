//! Experimental finite-mass unilateral normal contact for one articulated island.

use molla_compute::Engine;
use molla_core::{Error, Result};

pub const NORMAL_INPUT: u32 = 1;
pub const NORMAL_MASS: u32 = 2;
pub const NORMAL_NONFINITE: u32 = 4;
pub const NORMAL_LINE_SEARCH: u32 = 8;
pub const NORMAL_ITERATIONS: u32 = 16;
pub const NORMAL_HARD_SYSTEM: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilNormalContact {
    /// Finite soil mass; zero marks an inactive slot.
    pub mass: f32,
    pub free_normal_velocity: f32,
    /// Nonnegative gap divided by timestep; excludes penetration-repair bias.
    pub gap_speed: f32,
    pub padding: f32,
}

pub struct SoilNormalInputs<'a> {
    /// Symmetric positive-definite generalized machine mass, row-major.
    pub mass: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    /// Contact-major outward-normal machine Jacobian rows.
    pub rows: &'a wgpu::Buffer,
    /// Distinct soil normal DOFs; shared grid DOFs cannot be duplicated.
    pub contacts: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilNormalCapacity {
    pub dofs: u32,
    pub contacts: u32,
    pub max_iterations: u32,
    pub max_owned_bytes: u64,
}

/// Encodes a reduced frictionless solve and outward soil impulse recovery.
/// All inputs share the Engine device and a common configuration. Outputs are
/// consumable only with zero fault status. This component does not drift states,
/// generate geometry, repair penetration, or publish impulses to a runtime.
pub struct ExperimentalSoilNormalProjectionGpu {
    #[cfg(test)]
    pass_profile: Option<(wgpu::QuerySet, u32)>,
    #[cfg(test)]
    replay_inputs: [wgpu::Buffer; 4],
    #[cfg(test)]
    replay_config: (SoilNormalCapacity, bool),
    #[cfg(test)]
    replay_capture: Option<replay::FailureCapture>,
    #[cfg(test)]
    reference_check: Option<replay::ReferenceCheck>,
    pipelines: [wgpu::ComputePipeline; 2],
    group: wgpu::BindGroup,
    rates: wgpu::Buffer,
    impulses: wgpu::Buffer,
    normal_velocities: wgpu::Buffer,
    status: wgpu::Buffer,
    contacts: u32,
    bytes: u64,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil normal projection: {message}"))
}

fn storage_bytes(n: u32) -> u32 {
    2 * (4 * n * n).div_ceil(16) * 16 + 4 * (4 * n).div_ceil(16) * 16 + 16 * n + 416
}

pub(crate) fn hard_storage_bytes(n: u32) -> u32 {
    4 * (4 * n * n).div_ceil(16) * 16 + 7 * (4 * n).div_ceil(16) * 16 + 16 * n + 336
}

impl ExperimentalSoilNormalProjectionGpu {
    pub fn new(
        engine: &Engine,
        input: SoilNormalInputs<'_>,
        capacity: SoilNormalCapacity,
    ) -> Result<Self> {
        Self::new_internal(engine, input, capacity, false)
    }

    pub(crate) fn new_with_hard_rows(
        engine: &Engine,
        input: SoilNormalInputs<'_>,
        capacity: SoilNormalCapacity,
    ) -> Result<Self> {
        Self::new_internal(engine, input, capacity, true)
    }

    fn new_internal(
        engine: &Engine,
        input: SoilNormalInputs<'_>,
        capacity: SoilNormalCapacity,
        hard_rows: bool,
    ) -> Result<Self> {
        Self::new_shader(
            engine,
            input,
            capacity,
            hard_rows,
            include_str!("../shaders/soil_normal_projection.wgsl"),
        )
    }

    fn new_shader(
        engine: &Engine,
        input: SoilNormalInputs<'_>,
        capacity: SoilNormalCapacity,
        hard_rows: bool,
        source: &str,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        let n = u64::from(capacity.dofs);
        let count = u64::from(capacity.contacts.max(1));
        let bytes = 4 * n + 8 * count + 32;
        let limits = device.limits();
        #[cfg(not(test))]
        let candidate_storage_bytes = 0;
        #[cfg(test)]
        let candidate_storage_bytes = if source.contains("var<workgroup> cache_offsets:") {
            1296
        } else if source.contains("var<workgroup> cached_contacts:") {
            1040
        } else {
            0
        };
        if !(1..=32).contains(&capacity.dofs)
            || capacity.contacts > 65536
            || !(1..=64).contains(&capacity.max_iterations)
            || bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || (if hard_rows {
                hard_storage_bytes(capacity.dofs)
            } else {
                storage_bytes(capacity.dofs)
            }) + candidate_storage_bytes
                > limits.max_compute_workgroup_storage_size
            || capacity.contacts.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(invalid("capacity, iteration or storage limit exceeded"));
        }
        let inputs = [input.mass, input.free_rates, input.rows, input.contacts];
        for (buffer, required) in
            inputs
                .into_iter()
                .zip([4 * n * n, 4 * n, 4 * count * n, 16 * count])
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
        let rates = buffer("normal projected rates", 4 * n);
        let impulses = buffer("outward soil normal impulses", 4 * count);
        let normal_velocities = buffer("corrected soil normal velocities", 4 * count);
        let status = buffer("normal projection status", 16);
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("normal projection parameters"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[
                capacity.dofs,
                capacity.contacts,
                capacity.max_iterations,
                0u32,
            ])
        );
        let source = source
            .replace(
                "const WIDTH: u32 = 32u;",
                &format!("const WIDTH: u32 = {}u;", capacity.dofs),
            )
            .replace(
                "const HARD_WIDTH: u32 = 1u;",
                &format!(
                    "const HARD_WIDTH: u32 = {}u;",
                    if hard_rows { capacity.dofs } else { 1 }
                ),
            )
            .replace(
                "const HARD_ROWS: bool = false;",
                &format!("const HARD_ROWS: bool = {hard_rows};"),
            );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("finite-mass unilateral soil normals"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 8 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("normal projection"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("normal projection"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ["solve", "recover"].map(|entry_point| {
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
            input.mass,
            input.free_rates,
            input.rows,
            input.contacts,
            &rates,
            &impulses,
            &normal_velocities,
            &status,
            &params,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normal projection"),
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
        Ok(Self {
            #[cfg(test)]
            pass_profile: None,
            #[cfg(test)]
            replay_inputs: [
                input.mass.clone(),
                input.free_rates.clone(),
                input.rows.clone(),
                input.contacts.clone(),
            ],
            #[cfg(test)]
            replay_config: (capacity, hard_rows),
            #[cfg(test)]
            replay_capture: None,
            #[cfg(test)]
            reference_check: None,
            pipelines,
            group,
            rates,
            impulses,
            normal_velocities,
            status,
            contacts: capacity.contacts,
            bytes,
        })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(test)]
        if let Some(check) = &self.reference_check {
            encoder.copy_buffer_to_buffer(&self.status, 0, check.reference.status_buffer(), 0, 16);
        }
        #[cfg(test)]
        if let Some(capture) = &self.replay_capture {
            capture.arm(encoder);
        }
        for (_stage, (pipeline, groups)) in self
            .pipelines
            .iter()
            .zip([1, self.contacts.div_ceil(64)])
            .enumerate()
        {
            if groups == 0 {
                continue;
            }
            let descriptor = wgpu::ComputePassDescriptor::default();
            #[cfg(test)]
            let descriptor = wgpu::ComputePassDescriptor {
                timestamp_writes: self.pass_profile.as_ref().map(|(queries, offset)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: queries,
                        beginning_of_pass_write_index: Some(offset + 2 * _stage as u32),
                        end_of_pass_write_index: Some(offset + 2 * _stage as u32 + 1),
                    }
                }),
                ..descriptor
            };
            let mut pass = encoder.begin_compute_pass(&descriptor);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        #[cfg(test)]
        if let Some(capture) = &self.replay_capture {
            capture.capture(encoder);
        }
        #[cfg(test)]
        if let Some(check) = &self.reference_check {
            check.encode(encoder);
        }
    }

    #[cfg(test)]
    pub(crate) fn prepare_reference_check(
        &mut self,
        engine: &Engine,
        source: &str,
    ) -> wgpu::Buffer {
        let check = replay::ReferenceCheck::new(engine, self, source);
        let counts = check.counts.clone();
        self.reference_check = Some(check);
        counts
    }

    #[cfg(test)]
    pub(crate) fn prepare_failure_capture(&mut self, engine: &Engine) -> replay::NormalReplay {
        let capture = replay::FailureCapture::new(
            engine,
            &self.replay_inputs,
            &self.status,
            self.replay_config,
        );
        let result = capture.snapshot.clone();
        self.replay_capture = Some(capture);
        result
    }

    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        &self.rates
    }
    pub fn impulses_buffer(&self) -> &wgpu::Buffer {
        &self.impulses
    }
    pub fn normal_velocities_buffer(&self) -> &wgpu::Buffer {
        &self.normal_velocities
    }
    /// Four u32 words: sticky fault bits, iterations, backtracks and padding.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
#[path = "soil_normal_projection_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "soil_normal_replay.rs"]
pub(crate) mod replay;
