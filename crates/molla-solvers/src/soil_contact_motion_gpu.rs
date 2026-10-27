//! Explicit last-substep observation of impulse-weighted moving-soil contact slip.

use crate::soil_rate_encoding::{RateEncoding, motion_source};
use molla_compute::Engine;
use molla_core::{Error, Result};

#[cfg(test)]
#[path = "soil_contact_motion_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug)]
pub struct SoilContactMotion {
    /// Number of nodes carrying a positive machine normal impulse.
    pub supporting_nodes: u32,
    /// Sum of normal impulses on soil, in N s.
    pub normal_impulse: f32,
    /// Impulse-weighted corrected world soil velocity, in m/s; absent without support.
    pub soil_velocity: Option<[f32; 3]>,
    /// Impulse-weighted world tangential (machine point minus soil) velocity, in m/s.
    pub tangential_slip_velocity: Option<[f32; 3]>,
    /// Impulse-weighted mean of local tangential speed magnitudes, in m/s.
    pub mean_tangential_slip_speed: Option<f32>,
}

pub(crate) struct ContactMotionGpu {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    bodies: u32,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil contact motion: {message}"))
}

impl ContactMotionGpu {
    pub(crate) fn bytes(bodies: u32) -> u64 {
        96 * u64::from(bodies) + 16
    }

    #[cfg(test)]
    pub(crate) fn new(
        engine: &Engine,
        inputs: [&wgpu::Buffer; 8],
        dofs: u32,
        bodies: u32,
        contacts: u32,
        budget: u64,
    ) -> Result<Self> {
        Self::with_rate_encoding(
            engine,
            inputs,
            dofs,
            bodies,
            contacts,
            budget,
            RateEncoding::Scalar,
        )
    }

    pub(crate) fn with_rate_encoding(
        engine: &Engine,
        inputs: [&wgpu::Buffer; 8],
        dofs: u32,
        bodies: u32,
        contacts: u32,
        budget: u64,
        encoding: RateEncoding,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        let limits = device.limits();
        if !(1..=32).contains(&dofs)
            || !(1..=64).contains(&bodies)
            || !(1..=16384).contains(&contacts)
            || Self::bytes(bodies) > budget
            || limits.max_storage_buffers_per_shader_stage < 9
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_storage_size
                < if encoding == RateEncoding::Paired {
                    2116
                } else {
                    2112
                }
            || bodies > limits.max_compute_workgroups_per_dimension
        {
            return Err(invalid("capacity or resource limit exceeded"));
        }
        let (n, b, c) = (u64::from(dofs), u64::from(bodies), u64::from(contacts));
        for (buffer, required) in inputs.iter().zip([
            48 * c,
            16 * c,
            16 * c,
            16 * c,
            24 * b * n,
            encoding.bytes(n),
            16 * b,
            4,
        ]) {
            if buffer.size() < required
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid("input capacity or usage mismatch"));
            }
        }
        let create = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let output = create(
            "soil contact motion",
            48 * b,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = create(
            "soil contact motion observation",
            48 * b,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let params = create(
            "soil contact motion dimensions",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[dofs, bodies, contacts, 0])
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil contact motion"),
            source: wgpu::ShaderSource::Wgsl(motion_source(encoding).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil contact motion"),
            layout: None,
            module: &module,
            entry_point: Some("sample"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut buffers = inputs.to_vec();
        buffers.extend([&output, &params]);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil contact motion"),
            layout: &pipeline.get_bind_group_layout(0),
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
            pipeline,
            group,
            output,
            staging,
            bodies,
        })
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.bodies, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, self.staging.size());
    }
    pub(crate) fn staging(&self) -> &wgpu::Buffer {
        &self.staging
    }

    pub(crate) fn decode(&self, bytes: &[u8]) -> Result<Vec<SoilContactMotion>> {
        if bytes.len() != self.bodies as usize * 48 {
            return Err(invalid("readback length mismatch"));
        }
        bytes
            .chunks_exact(48)
            .map(|row| {
                let words = bytemuck::pod_read_unaligned::<[u32; 12]>(row);
                let values = words[..8]
                    .iter()
                    .map(|&v| f32::from_bits(v))
                    .collect::<Vec<_>>();
                let weight = values[3];
                let count = words[9];
                if words[8] != 0
                    || values.iter().any(|v| !v.is_finite())
                    || weight < 0.0
                    || values[7] < 0.0
                    || (weight == 0.0) != (count == 0)
                {
                    return Err(invalid("faulted or invalid contact sample"));
                }
                Ok(SoilContactMotion {
                    supporting_nodes: count,
                    normal_impulse: weight,
                    soil_velocity: (count > 0).then_some([values[0], values[1], values[2]]),
                    tangential_slip_velocity: (count > 0)
                        .then_some([values[4], values[5], values[6]]),
                    mean_tangential_slip_speed: (count > 0).then_some(values[7]),
                })
            })
            .collect()
    }
}
