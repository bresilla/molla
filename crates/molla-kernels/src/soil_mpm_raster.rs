//! Prepared particle-binned MPM terrain export.

use crate::soil_particle_raster::{ParticleRasterKernel, PreparedParticleRaster, RasterParams};
use crate::tire_mpm::TireMpmTerrainRasterParams;
use molla_core::Result;

pub(crate) struct MpmRasterKernel(ParticleRasterKernel);

/// Fixed-buffer MPM export with reusable bins and immutable parameters.
/// Encode on one queue in order; prepare again when buffer identities or parameters change.
pub struct PreparedTireMpmRaster(PreparedParticleRaster);

impl MpmRasterKernel {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self(ParticleRasterKernel::new(
            device,
            include_str!("../shaders/soil_mpm_raster.wgsl"),
        ))
    }

    pub(crate) fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 6],
        params: &TireMpmTerrainRasterParams,
        fallback: [f32; 3],
    ) -> Result<PreparedTireMpmRaster> {
        params.validate()?;
        self.0
            .prepare(
                device,
                buffers,
                &RasterParams {
                    particles: params.particle_count,
                    nx: params.nx,
                    nz: params.nz,
                    terrain: [
                        params.origin[0],
                        params.origin[1],
                        params.spacing,
                        params.rest_height,
                    ],
                    extra: [fallback[0], fallback[1], fallback[2], 0.0],
                    attribute_bytes: u64::from(params.particle_count.max(1)) * 16,
                },
            )
            .map(PreparedTireMpmRaster)
    }
}

impl PreparedTireMpmRaster {
    /// Records exhaustive reference export without allocation or submission.
    pub fn encode_reference(&self, encoder: &mut wgpu::CommandEncoder) {
        self.0.encode_reference(encoder);
    }

    /// Records binned export without allocation, submission, or readback.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.0.encode(encoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soil_raster_dimensions_are_checked_before_allocation() {
        let p = TireMpmTerrainRasterParams {
            particle_count: 0,
            origin: [0.0; 2],
            spacing: 1.0,
            nx: 65,
            nz: 65,
            rest_height: 0.0,
        };
        assert_eq!(p.validate().unwrap(), 4225);
        for (nx, nz) in [(0, 1), (1, 0), (u32::MAX, 2), (65536, 65536)] {
            assert!(
                TireMpmTerrainRasterParams { nx, nz, ..p }
                    .validate()
                    .is_err()
            );
        }
        for spacing in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                TireMpmTerrainRasterParams { spacing, ..p }
                    .validate()
                    .is_err()
            );
        }
    }
}
