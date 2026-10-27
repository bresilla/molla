use super::*;

/// Live read-only rendering input on the simulation's wgpu device.
/// Positions are WGSL array<vec4<f32>> with xyz in world metres and 16-byte stride.
/// Only the counted prefix is live. Bind as read-only storage and draw that count.
/// Submit rendering on the same queue after physics; this is not a frozen snapshot
/// or a numerical-success certificate. Do not mutate these buffers. Retained bind
/// groups must validate source identity and refresh the count before every draw.
pub struct MpmParticleRenderView<'a> {
    positions: &'a wgpu::Buffer,
    count: u32,
    source: &'a Arc<()>,
}

/// Binding identity only; does not freeze positions, count or physical time.
#[derive(Clone, Debug)]
pub struct MpmParticleRenderSource(Arc<()>);

impl<'a> MpmParticleRenderView<'a> {
    pub fn positions(&self) -> &'a wgpu::Buffer {
        self.positions
    }

    pub fn particle_count(&self) -> u32 {
        self.count
    }

    pub fn live_position_bytes(&self) -> u64 {
        u64::from(self.count) * 16
    }

    pub fn source(&self) -> MpmParticleRenderSource {
        MpmParticleRenderSource(self.source.clone())
    }

    pub fn matches_source(&self, source: &MpmParticleRenderSource) -> bool {
        Arc::ptr_eq(self.source, &source.0)
    }
}

impl MpmSimulation {
    /// Borrows rendering inputs without allocation, submission, upload or readback.
    /// Device health/identity are checked; queued numerical faults are not polled.
    pub fn particle_render_view(&self, engine: &Engine) -> Result<MpmParticleRenderView<'_>> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("particle render view requires GPU".into()))?;
        molla_kernels::validate_soil_device(device, &self.device)?;
        Ok(MpmParticleRenderView {
            positions: &self.p_pos,
            count: self.particle_count,
            source: &self.soil_owner,
        })
    }
}
