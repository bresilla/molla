//! Shared impulse minimization for circular friction and scalar joint constraints.

use crate::coulomb_disk_gpu::{CoulombDiskConfig, ResidentCoulombDisks};
use molla_core::Result;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintBlockKind {
    /// Circular static/dynamic Coulomb impulse disk.
    Disk = 0,
    /// Unbounded scalar impulse with zero final constraint velocity.
    Bilateral = 1,
    /// Nonnegative impulse complementary to nonnegative final constraint velocity.
    Nonnegative = 2,
    /// Scalar impulse in a closed interval with optional implicit feedback.
    Interval = 3,
}

#[derive(Clone, Copy, Debug)]
pub struct CoupledConstraintConfig {
    pub island_capacity: u32,
    pub blocks_per_island: u32,
    pub max_iterations: u32,
    pub velocity_tolerance: f32,
}

/// Minimizes `0.5 * impulse^T * gram * impulse + free_velocity^T * impulse`.
/// Each island uses a fixed square Gram slot with stride `2 * blocks_per_island`.
/// Two rows per block: disk tangents, or a scalar followed by a zero row.
/// Interval metadata is `[lower, upper, response_diagonal, 0]`; other scalar metadata is zero.
/// Unused free velocity and unused Gram rows/columns must be zero.
/// Mixed islands minimize the SI row-impulse Euclidean norm among primary minimizers.
pub struct ResidentCoupledConstraints {
    solver: ResidentCoulombDisks,
}

impl ResidentCoupledConstraints {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        gram: &wgpu::Buffer,
        free_velocity: &wgpu::Buffer,
        disk_data: &wgpu::Buffer,
        block_kinds: &wgpu::Buffer,
        block_counts: &wgpu::Buffer,
        island_count: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: CoupledConstraintConfig,
    ) -> Result<Self> {
        let solver = ResidentCoulombDisks::with_block_kinds(
            device,
            gram,
            free_velocity,
            disk_data,
            block_counts,
            island_count,
            status,
            Some(block_kinds),
            CoulombDiskConfig {
                island_capacity: config.island_capacity,
                contacts_per_island: config.blocks_per_island,
                max_iterations: config.max_iterations,
                velocity_tolerance: config.velocity_tolerance,
            },
        )?;
        Ok(Self { solver })
    }

    /// Records one retained solve; counts, domains and solve inputs remain device-owned.
    /// Status uses the Coulomb-disk fault bits and remains sticky until caller reset.
    /// Every output must be ignored while status is nonzero.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.solver.encode(encoder);
    }

    /// Scalar regimes are 4 (bilateral), 5 (nonnegative) and 6 (interval).
    /// Scalar impulse uses a high/low pair; slip uses x and the applied cap is zero.
    pub fn solution_buffer(&self) -> &wgpu::Buffer {
        self.solver.solution_buffer()
    }

    /// Disk high/low vectors use the primary impulse plus two residue floats per slot.
    pub fn impulse_residue_offset(&self) -> u64 {
        self.solver.impulse_residue_offset()
    }

    pub fn diagnostic_buffer(&self) -> &wgpu::Buffer {
        self.solver.diagnostic_buffer()
    }

    pub fn resource_counts(&self) -> [usize; 3] {
        self.solver.resource_counts()
    }
}

#[cfg(test)]
#[path = "coupled_constraint_gpu_tests.rs"]
mod tests;
