//! Prepared particle-binned DEM terrain export.

use crate::soil_particle_raster::{ParticleRasterKernel, PreparedParticleRaster, RasterParams};
use crate::tire_dem::{TireDemResidentState, TireDemResidentTerrainOutput};
use molla_core::{Error, Result};

pub(crate) struct DemRasterKernel(ParticleRasterKernel);

/// Fixed-buffer DEM export with immutable parameters and reusable bins.
/// Input values may change between ordered encodes on the same queue.
/// Reprepare when buffer identities, grain count or terrain parameters change.
pub struct PreparedTireDemRaster(PreparedParticleRaster);

impl DemRasterKernel {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self(ParticleRasterKernel::new(
            device,
            include_str!("../shaders/soil_dem_raster.wgsl"),
        ))
    }

    pub(crate) fn prepare(
        &self,
        device: &wgpu::Device,
        state: &TireDemResidentState,
        reference: &wgpu::Buffer,
        terrain: &TireDemResidentTerrainOutput,
    ) -> Result<PreparedTireDemRaster> {
        let p = terrain.params;
        let cells =
            p.nx.checked_mul(p.nz)
                .ok_or_else(|| Error::Solver("soil DEM raster: cell count overflow".into()))?;
        let particles = u32::try_from(state.grain_count)
            .map_err(|_| Error::Solver("soil DEM raster: grain count overflow".into()))?;
        if !p.grain_radius.is_finite() || p.grain_radius <= 0.0 {
            return Err(Error::Solver(
                "soil DEM raster: invalid grain radius".into(),
            ));
        }
        self.0
            .prepare(
                device,
                [
                    &state.positions,
                    reference,
                    &state.velocities,
                    &terrain.heights,
                    &terrain.sinkage,
                    &terrain.surface_velocity_rows,
                ],
                &RasterParams {
                    particles,
                    nx: p.nx,
                    nz: p.nz,
                    terrain: [p.origin[0], p.origin[1], p.spacing, p.floor_y],
                    extra: [p.grain_radius, 0.0, 0.0, 0.0],
                    attribute_bytes: u64::from(cells) * 4,
                },
            )
            .map(PreparedTireDemRaster)
    }
}

impl PreparedTireDemRaster {
    /// Records exhaustive reference export without allocation or submission.
    pub fn encode_reference(&self, encoder: &mut wgpu::CommandEncoder) {
        self.0.encode_reference(encoder);
    }

    /// Records binned export without allocation, submission, or readback.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.0.encode(encoder);
    }
}
