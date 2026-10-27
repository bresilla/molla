//! Position-dependent effort clamp via shared lookup table.
//!
//! All actuators in one stage share the same `(position →
//! max_effort)` table; the table is interpolated per-DOF using the
//! current `joint_q[dof]` value. Useful for modelling actuators
//! whose available effort drops near hard joint limits.

use super::Clamping;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::ClampingPositionBasedKernel;
use molla_math::Real;
use molla_sim::{Control, State};
use molla_storage::{Buffer, BufferUsage};

pub struct ClampingPositionBased {
    kernel: ClampingPositionBasedKernel,
    actuator_count: u32,
    lookup_size: u32,
    dof_indices: Buffer<u32>,
    lookup_positions: Buffer<Real>,
    lookup_efforts: Buffer<Real>,
}

impl ClampingPositionBased {
    /// Build a position-based clamp. `dof_indices` is the per-actuator
    /// index into `joint_q / joint_force`; `(positions, efforts)` is
    /// the shared lookup table. `positions` must be sorted ascending
    /// (no two adjacent equal) and `efforts` must match in length.
    pub fn new(
        engine: &Engine,
        dof_indices: &[u32],
        positions: &[Real],
        efforts: &[Real],
    ) -> Result<Self> {
        if dof_indices.is_empty() {
            return Err(Error::Solver(
                "ClampingPositionBased::new: empty dof list".into(),
            ));
        }
        if positions.is_empty() || positions.len() != efforts.len() {
            return Err(Error::Solver(
                "ClampingPositionBased::new: positions/efforts length mismatch or empty".into(),
            ));
        }
        for w in positions.windows(2) {
            if w[0] > w[1] {
                return Err(Error::Solver(
                    "ClampingPositionBased::new: positions must be monotonically non-decreasing"
                        .into(),
                ));
            }
        }
        for (i, &e) in efforts.iter().enumerate() {
            if e < 0.0 {
                return Err(Error::Solver(format!(
                    "ClampingPositionBased::new: efforts[{i}] must be ≥ 0, got {e}"
                )));
            }
        }

        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ClampingPositionBased: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ClampingPositionBased: no GPU queue".into()))?
            .clone();

        let mut dof_b = Buffer::from_vec(dof_indices.to_vec(), BufferUsage::Storage);
        let mut pos_b = Buffer::from_vec(positions.to_vec(), BufferUsage::Storage);
        let mut eff_b = Buffer::from_vec(efforts.to_vec(), BufferUsage::Storage);
        dof_b.ensure_device(&device)?;
        pos_b.ensure_device(&device)?;
        eff_b.ensure_device(&device)?;
        dof_b.upload(&queue)?;
        pos_b.upload(&queue)?;
        eff_b.upload(&queue)?;

        Ok(Self {
            kernel: ClampingPositionBasedKernel::new(&device),
            actuator_count: dof_indices.len() as u32,
            lookup_size: positions.len() as u32,
            dof_indices: dof_b,
            lookup_positions: pos_b,
            lookup_efforts: eff_b,
        })
    }
}

impl Clamping for ClampingPositionBased {
    fn step(&mut self, engine: &Engine, state: &State, control: &mut Control) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ClampingPositionBased::step: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ClampingPositionBased::step: no GPU queue".into()))?
            .clone();
        self.kernel.dispatch(
            &device,
            &queue,
            state,
            control,
            &self.dof_indices,
            &self.lookup_positions,
            &self.lookup_efforts,
            self.actuator_count,
            self.lookup_size,
        )
    }
}
