//! Retained in-place tire feedback with immutable caller-encoded records.

use super::{
    TireMpmContactCell, TireMpmGpuInPlaceParams, TireMpmGpuKernel, TireMpmUniforms, WORKGROUP_SIZE,
    entry,
};
use crate::soil_gpu_identity::{device_identity, queue_identity};
use molla_core::{Error, Result};

struct Bindings {
    pipeline: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 3],
}

impl Bindings {
    fn new(
        kernel: &TireMpmGpuKernel,
        device: &wgpu::Device,
        particles: [&wgpu::Buffer; 3],
        contacts: &wgpu::Buffer,
        uniforms: &wgpu::Buffer,
    ) -> Self {
        Self {
            pipeline: kernel.pipeline_in_place.clone(),
            groups: [
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MPM feedback empty"),
                    layout: &kernel.layout_empty,
                    entries: &[],
                }),
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MPM feedback parameters"),
                    layout: &kernel.layout_uniforms,
                    entries: &[entry(0, uniforms)],
                }),
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MPM feedback particles and contacts"),
                    layout: &kernel.layout_in_place,
                    entries: &[
                        entry(0, particles[0]),
                        entry(1, particles[1]),
                        entry(2, particles[2]),
                        entry(3, contacts),
                    ],
                }),
            ],
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        if count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MPM retained tire feedback"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (i, group) in self.groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }
        pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}

/// Fixed counts and coefficients; contacts may be produced earlier in the same encoder.
pub struct PreparedTireMpmFeedback {
    bindings: Bindings,
    count: u32,
}

impl PreparedTireMpmFeedback {
    /// Records without creating GPU buffers/bindings, transfers or submission; encoder must use the preparing device.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.bindings.encode(encoder, self.count);
    }
}

/// Fixed-capacity CPU-contact upload bridge with retained GPU bindings.
pub struct TireMpmResidentFeedback {
    device: wgpu::Device,
    queue: wgpu::Queue,
    contacts: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    bindings: Bindings,
    particle_capacity: u32,
    contact_capacity: u32,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("MPM retained feedback: {message}"))
}

fn validate_params(params: &TireMpmGpuInPlaceParams) -> Result<()> {
    if !params.particle_mass.is_finite() || params.particle_mass <= 0.0 {
        return Err(invalid("mass must be finite and positive"));
    }
    for value in [
        params.dt,
        params.plastic_compaction_gain,
        params.max_compaction,
        params.damping,
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(invalid("coefficients must be finite and nonnegative"));
        }
    }
    Ok(())
}

fn uniform(params: &TireMpmGpuInPlaceParams, contacts: u32) -> TireMpmUniforms {
    TireMpmUniforms {
        dims: [params.particle_count, contacts, 0, 0],
        params: [
            params.particle_mass,
            params.dt,
            params.plastic_compaction_gain,
            params.max_compaction,
        ],
        params2: [params.damping, 0.0, 0.0, 0.0],
    }
}

fn validate_storage(device: &wgpu::Device, buffer: &wgpu::Buffer, bytes: u64) -> Result<()> {
    if !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        || buffer.size() < bytes
        || buffer.size() > u64::from(device.limits().max_storage_buffer_binding_size)
    {
        return Err(invalid("storage usage, size or binding limit mismatch"));
    }
    Ok(())
}

fn validate_particles(
    device: &wgpu::Device,
    particles: [&wgpu::Buffer; 3],
    count: u32,
) -> Result<()> {
    if count.div_ceil(WORKGROUP_SIZE) > device.limits().max_compute_workgroups_per_dimension {
        return Err(invalid("particle dispatch exceeds device limits"));
    }
    for (i, buffer) in particles.iter().enumerate() {
        validate_storage(device, buffer, u64::from(count.max(1)) * 16)?;
        if particles[..i].contains(buffer) {
            return Err(invalid("particle buffers must not alias"));
        }
    }
    Ok(())
}

fn make_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
    mapped_at_creation: bool,
) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation
        }
    )
}

impl TireMpmGpuKernel {
    /// Checks that feedback resources and kernel share a supported native device.
    pub fn validate_feedback_device(&self, device: &wgpu::Device) -> Result<()> {
        let identity = device_identity(&self.device)
            .ok_or_else(|| invalid("native device identity unavailable"))?;
        if device_identity(device) != Some(identity) {
            return Err(invalid("kernel/device mismatch"));
        }
        Ok(())
    }

    /// Buffers must belong to device; GPU contact rows must be finite with nonnegative radii.
    pub fn prepare_in_place_from_buffers(
        &self,
        device: &wgpu::Device,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        compaction: &wgpu::Buffer,
        contacts: &wgpu::Buffer,
        contact_count: u32,
        params: &TireMpmGpuInPlaceParams,
    ) -> Result<PreparedTireMpmFeedback> {
        self.validate_feedback_device(device)?;
        validate_params(params)?;
        let particles = [positions, velocities, compaction];
        validate_particles(device, particles, params.particle_count)?;
        validate_storage(device, contacts, u64::from(contact_count.max(1)) * 32)?;
        if particles.contains(&contacts) {
            return Err(invalid("contacts must not alias particle buffers"));
        }
        let uniforms = make_buffer(
            device,
            "MPM immutable feedback record",
            48,
            wgpu::BufferUsages::UNIFORM,
            true,
        );
        uniforms
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::bytes_of(&uniform(params, contact_count)));
        uniforms.unmap();
        Ok(PreparedTireMpmFeedback {
            bindings: Bindings::new(self, device, particles, contacts, &uniforms),
            count: params.particle_count,
        })
    }

    /// Queue and particle buffers must belong to device. Call outside the simulation loop.
    pub fn prepare_resident_in_place(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        compaction: &wgpu::Buffer,
        particle_capacity: u32,
        contact_capacity: u32,
    ) -> Result<TireMpmResidentFeedback> {
        self.validate_feedback_device(device)?;
        if queue_identity(queue).is_none() {
            return Err(invalid("native queue identity unavailable"));
        }
        let particles = [positions, velocities, compaction];
        validate_particles(device, particles, particle_capacity)?;
        let contact_bytes = u64::from(contact_capacity.max(1)) * 32;
        if contact_bytes > u64::from(device.limits().max_storage_buffer_binding_size)
            || contact_bytes > device.limits().max_buffer_size
        {
            return Err(invalid("contact capacity exceeds device limits"));
        }
        let contacts = make_buffer(
            device,
            "MPM retained contacts",
            contact_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            false,
        );
        let uniforms = make_buffer(
            device,
            "MPM retained feedback parameters",
            48,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            false,
        );
        let bindings = Bindings::new(self, device, particles, &contacts, &uniforms);
        Ok(TireMpmResidentFeedback {
            device: device.clone(),
            queue: queue.clone(),
            contacts,
            uniforms,
            bindings,
            particle_capacity,
            contact_capacity,
        })
    }
}

impl TireMpmResidentFeedback {
    pub fn particle_capacity(&self) -> u32 {
        self.particle_capacity
    }
    pub fn contact_capacity(&self) -> u32 {
        self.contact_capacity
    }

    /// Rejects invalid CPU inputs without allocating or uploading.
    pub fn validate_inputs(
        contacts: &[TireMpmContactCell],
        params: &TireMpmGpuInPlaceParams,
    ) -> Result<()> {
        validate_params(params)?;
        if contacts.len() > u32::MAX as usize {
            return Err(invalid("contact count exceeds u32"));
        }
        if contacts.iter().any(|c| {
            c.center_radius
                .iter()
                .chain(&c.force_pressure)
                .any(|v| !v.is_finite())
                || c.center_radius[3] < 0.0
        }) {
            return Err(invalid("contacts must be finite with nonnegative radii"));
        }
        Ok(())
    }

    /// Checks retained device/queue ownership without enqueuing work.
    pub fn validate_context(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireMpmGpuKernel,
    ) -> Result<()> {
        kernel.validate_feedback_device(device)?;
        if device_identity(device) != device_identity(&self.device)
            || queue_identity(queue) != queue_identity(&self.queue)
        {
            return Err(invalid("retained device/queue mismatch"));
        }
        Ok(())
    }

    /// Uploads and immediately submits one step. Empty contacts still advect nonempty particles.
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireMpmGpuKernel,
        contacts: &[TireMpmContactCell],
        params: &TireMpmGpuInPlaceParams,
    ) -> Result<()> {
        self.validate_context(device, queue, kernel)?;
        Self::validate_inputs(contacts, params)?;
        if params.particle_count > self.particle_capacity
            || contacts.len() > self.contact_capacity as usize
        {
            return Err(invalid("prepared capacity exceeded"));
        }
        if params.particle_count == 0 {
            return Ok(());
        }
        if !contacts.is_empty() {
            molla_core::gpu_write_buffer!(queue, &self.contacts, 0, bytemuck::cast_slice(contacts));
        }
        molla_core::gpu_write_buffer!(
            queue,
            &self.uniforms,
            0,
            bytemuck::bytes_of(&uniform(params, contacts.len() as u32))
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MPM retained feedback submission"),
        });
        self.bindings.encode(&mut encoder, params.particle_count);
        queue.submit([encoder.finish()]);
        Ok(())
    }
}
