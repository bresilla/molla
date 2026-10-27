//! Status-gated publication of projected rates and momentum-transport deltas.

use super::*;

pub struct SoilRatePublicationInputs<'a> {
    /// Free predicted rates, replaced in-place by corrected rates.
    pub rates: &'a wgpu::Buffer,
    /// Featherstone's resident velocity delta for momentum transport.
    pub velocity_delta: &'a wgpu::Buffer,
    /// Shared Featherstone stop status; failures set the solver input-fault bit 32.
    pub machine_status: &'a wgpu::Buffer,
    pub max_owned_bytes: u64,
}

/// Persistent generalized rates and transport deltas, each interleaved [high, low].
pub struct SoilPairedRatePublicationInputs<'a> {
    pub rates: &'a wgpu::Buffer,
    pub velocity_delta: &'a wgpu::Buffer,
    pub machine_status: &'a wgpu::Buffer,
    pub max_owned_bytes: u64,
}

#[derive(Clone, Copy)]
enum RateLayout {
    Scalar,
    Paired,
    PairedIncrement,
}

/// Bound to one output state and projection. Encode after projection and before
/// corrected drift/momentum transport. All buffers must belong to the engine;
/// rates must still contain the free prediction used by that projection.
/// Caller owns state freshness, frame rollback, and status reset boundaries.
pub struct SoilRatePublicationGpu {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

impl SoilRatePublicationGpu {
    pub(super) fn new(
        engine: &Engine,
        projection: &SoilContactProjectionGpu,
        input: SoilRatePublicationInputs<'_>,
    ) -> Result<Self> {
        Self::from_buffers(
            engine,
            projection.rates_buffer(),
            projection.status_buffer(),
            projection.capacity.dofs,
            input,
        )
    }

    pub(crate) fn from_buffers(
        engine: &Engine,
        projected_rates: &wgpu::Buffer,
        status: &wgpu::Buffer,
        dofs: u32,
        input: SoilRatePublicationInputs<'_>,
    ) -> Result<Self> {
        Self::build(
            engine,
            projected_rates,
            status,
            dofs,
            input,
            RateLayout::Scalar,
        )
    }

    /// Publishes eight-byte rate pairs without converting to scalar rate storage.
    pub fn from_paired_buffers(
        engine: &Engine,
        projected_rates: &wgpu::Buffer,
        status: &wgpu::Buffer,
        dofs: u32,
        input: SoilPairedRatePublicationInputs<'_>,
    ) -> Result<Self> {
        Self::build(
            engine,
            projected_rates,
            status,
            dofs,
            SoilRatePublicationInputs {
                rates: input.rates,
                velocity_delta: input.velocity_delta,
                machine_status: input.machine_status,
                max_owned_bytes: input.max_owned_bytes,
            },
            RateLayout::Paired,
        )
    }

    /// Advances persistent rate pairs and starts the transport delta for a new step.
    /// Increment and rate buffers both contain interleaved high/low pairs.
    pub fn from_paired_increments(
        engine: &Engine,
        increments: &wgpu::Buffer,
        status: &wgpu::Buffer,
        dofs: u32,
        input: SoilPairedRatePublicationInputs<'_>,
    ) -> Result<Self> {
        Self::build(
            engine,
            increments,
            status,
            dofs,
            SoilRatePublicationInputs {
                rates: input.rates,
                velocity_delta: input.velocity_delta,
                machine_status: input.machine_status,
                max_owned_bytes: input.max_owned_bytes,
            },
            RateLayout::PairedIncrement,
        )
    }

    fn build(
        engine: &Engine,
        projected_rates: &wgpu::Buffer,
        status: &wgpu::Buffer,
        dofs: u32,
        input: SoilRatePublicationInputs<'_>,
        layout: RateLayout,
    ) -> Result<Self> {
        let paired = !matches!(layout, RateLayout::Scalar);
        engine.check_device()?;
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        let buffers = [
            projected_rates,
            status,
            input.rates,
            input.velocity_delta,
            input.machine_status,
        ];
        let bytes = u64::from(dofs) * if paired { 8 } else { 4 };
        if !(1..=32).contains(&dofs) || input.max_owned_bytes < 16 {
            return Err(invalid("rate publication budget exceeded"));
        }
        if paired && device.limits().max_compute_workgroup_storage_size < 516 {
            return Err(invalid("paired publication workgroup capacity exceeded"));
        }
        for (i, (buffer, required)) in buffers.iter().zip([bytes, 4, bytes, bytes, 4]).enumerate() {
            if buffer.size() < required
                || buffer.size() > device.limits().max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                || buffers[..i].contains(buffer)
            {
                return Err(invalid(
                    "rate publication buffer capacity, usage or alias mismatch",
                ));
            }
        }
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil rate publication parameters"),
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
                dofs,
                u32::from(matches!(layout, RateLayout::PairedIncrement)),
                0,
                0
            ])
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil rate publication"),
            source: wgpu::ShaderSource::Wgsl(
                if paired {
                    include_str!("../shaders/soil_paired_rate_publication.wgsl")
                } else {
                    include_str!("../shaders/soil_rate_publication.wgsl")
                }
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil rate publication"),
            layout: None,
            module: &shader,
            entry_point: Some("publish"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil rate publication"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &buffers
                .into_iter()
                .chain([&params])
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self { pipeline, group })
    }

    pub fn owned_buffer_bytes(&self) -> u64 {
        16
    }

    /// Validates the complete island before publishing any rate or delta.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}
