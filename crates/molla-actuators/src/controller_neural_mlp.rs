//! Stage-18.1 fixed-architecture 3-layer MLP policy controller.
//!
//! Use case: drop a pre-trained policy weight slab into the actuator pipeline
//! so inference happens entirely on-device — no host round-trip per substep.
//!
//! Architecture (hard-coded in WGSL): `input → hidden → hidden → output`,
//! with `tanh` non-linearities on both hidden layers and a linear
//! output. All three sizes are configurable at construction time via
//! [`ControllerNeuralMlpConfig`]; the kernel doesn't need to be
//! recompiled to switch dimensions.
//!
//! Inputs are gathered host-side (read whatever observations the
//! policy needs out of `state` / sensors / external code) and pushed
//! through [`ControllerNeuralMlp::set_inputs`]. The forward pass + a
//! scatter pass write the network's outputs straight into
//! `Control.joint_force[dof_indices[i]]` so the rest of the pipeline
//! (clamping, etc.) sees the policy's effort just like any other
//! controller.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::ActuatorNeuralMlpKernel;
use molla_math::Real;
use molla_sim::Control;
use molla_storage::{Buffer, BufferUsage};

/// Construction-time configuration for [`ControllerNeuralMlp`].
#[derive(Debug, Clone, Copy)]
pub struct ControllerNeuralMlpConfig {
    pub input_dim: u32,
    pub hidden_dim: u32,
    pub output_dim: u32,
}

/// Trained MLP weights, in row-major layout. The arity of each slab
/// must equal what [`ControllerNeuralMlp::expected_weight_dims`]
/// reports for the configured `(input_dim, hidden_dim, output_dim)`.
pub struct MlpWeights<'a> {
    pub w1: &'a [Real],
    pub b1: &'a [Real],
    pub w2: &'a [Real],
    pub b2: &'a [Real],
    pub w3: &'a [Real],
    pub b3: &'a [Real],
}

pub struct ControllerNeuralMlp {
    kernel: ActuatorNeuralMlpKernel,
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
    /// Per-output DOF index — `outputs[i]` lands in
    /// `joint_force[dof_indices[i]]`.
    dof_indices: Buffer<u32>,
    /// Network input vector. Updated host-side every substep via
    /// [`Self::set_inputs`].
    inputs: Buffer<Real>,
    /// Network output vector. Read back via [`Self::download_outputs`]
    /// for diagnostic UIs (the pipeline's scatter pass also copies
    /// into `joint_force` device-side).
    outputs: Buffer<Real>,
    w1: Buffer<Real>,
    b1: Buffer<Real>,
    w2: Buffer<Real>,
    b2: Buffer<Real>,
    w3: Buffer<Real>,
    b3: Buffer<Real>,
}

impl ControllerNeuralMlp {
    /// Allocate buffers + GPU resources for a fresh MLP. Weights start
    /// at zero — call [`Self::set_weights`] before stepping the
    /// pipeline if you have pre-trained parameters.
    ///
    /// `dof_indices` maps each network output index to the DOF that
    /// should receive the output. Length must equal `output_dim`.
    pub fn new(
        engine: &Engine,
        cfg: ControllerNeuralMlpConfig,
        dof_indices: &[u32],
    ) -> Result<Self> {
        if cfg.input_dim == 0 || cfg.hidden_dim == 0 || cfg.output_dim == 0 {
            return Err(Error::Solver(
                "ControllerNeuralMlp::new: all dims must be > 0".into(),
            ));
        }
        if cfg.hidden_dim > 256 {
            return Err(Error::Solver(
                "ControllerNeuralMlp::new: hidden_dim must be ≤ 256 (bumped via the WGSL MAX_HIDDEN constant)".into(),
            ));
        }
        if dof_indices.len() != cfg.output_dim as usize {
            return Err(Error::Solver(format!(
                "ControllerNeuralMlp::new: dof_indices.len()={} expected {}",
                dof_indices.len(),
                cfg.output_dim
            )));
        }
        // Duplicate-DOF check (concurrent writes would race in the
        // scatter pass).
        for i in 0..dof_indices.len() {
            for j in (i + 1)..dof_indices.len() {
                if dof_indices[i] == dof_indices[j] {
                    return Err(Error::Solver(format!(
                        "ControllerNeuralMlp::new: duplicate dof {} at outputs {i} and {j}",
                        dof_indices[i]
                    )));
                }
            }
        }

        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ControllerNeuralMlp: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerNeuralMlp: no GPU queue".into()))?
            .clone();

        let (w1_n, b1_n, w2_n, b2_n, w3_n, b3_n) =
            expected_weight_dims(cfg.input_dim, cfg.hidden_dim, cfg.output_dim);
        let mut dof_b = Buffer::from_vec(dof_indices.to_vec(), BufferUsage::Storage);
        let mut inputs = Buffer::from_vec(
            vec![0.0 as Real; cfg.input_dim as usize],
            BufferUsage::Storage,
        );
        let mut outputs = Buffer::from_vec(
            vec![0.0 as Real; cfg.output_dim as usize],
            BufferUsage::StorageReadWrite,
        );
        let mut w1 = Buffer::from_vec(vec![0.0 as Real; w1_n], BufferUsage::Storage);
        let mut b1 = Buffer::from_vec(vec![0.0 as Real; b1_n], BufferUsage::Storage);
        let mut w2 = Buffer::from_vec(vec![0.0 as Real; w2_n], BufferUsage::Storage);
        let mut b2 = Buffer::from_vec(vec![0.0 as Real; b2_n], BufferUsage::Storage);
        let mut w3 = Buffer::from_vec(vec![0.0 as Real; w3_n], BufferUsage::Storage);
        let mut b3 = Buffer::from_vec(vec![0.0 as Real; b3_n], BufferUsage::Storage);

        {
            let buf = &dof_b.device_buffer().is_some();
            let _ = buf;
        }
        dof_b.ensure_device(&device)?;
        inputs.ensure_device(&device)?;
        outputs.ensure_device(&device)?;
        w1.ensure_device(&device)?;
        b1.ensure_device(&device)?;
        w2.ensure_device(&device)?;
        b2.ensure_device(&device)?;
        w3.ensure_device(&device)?;
        b3.ensure_device(&device)?;

        dof_b.upload(&queue)?;
        inputs.upload(&queue)?;
        outputs.upload(&queue)?;
        w1.upload(&queue)?;
        b1.upload(&queue)?;
        w2.upload(&queue)?;
        b2.upload(&queue)?;
        w3.upload(&queue)?;
        b3.upload(&queue)?;

        Ok(Self {
            kernel: ActuatorNeuralMlpKernel::new(&device),
            input_dim: cfg.input_dim,
            hidden_dim: cfg.hidden_dim,
            output_dim: cfg.output_dim,
            dof_indices: dof_b,
            inputs,
            outputs,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        })
    }

    pub fn input_dim(&self) -> u32 {
        self.input_dim
    }
    pub fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub fn output_dim(&self) -> u32 {
        self.output_dim
    }

    /// Expected weight slab sizes for the configured architecture, as
    /// `(w1, b1, w2, b2, w3, b3)` lengths in elements.
    pub fn expected_weight_dims(&self) -> (usize, usize, usize, usize, usize, usize) {
        expected_weight_dims(self.input_dim, self.hidden_dim, self.output_dim)
    }

    /// Upload trained weights. All six slabs must match the lengths
    /// returned by [`Self::expected_weight_dims`].
    pub fn set_weights(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        weights: &MlpWeights,
    ) -> Result<()> {
        let (w1_n, b1_n, w2_n, b2_n, w3_n, b3_n) = self.expected_weight_dims();
        if weights.w1.len() != w1_n
            || weights.b1.len() != b1_n
            || weights.w2.len() != w2_n
            || weights.b2.len() != b2_n
            || weights.w3.len() != w3_n
            || weights.b3.len() != b3_n
        {
            return Err(Error::Solver(format!(
                "ControllerNeuralMlp::set_weights: expected ({w1_n},{b1_n},{w2_n},{b2_n},{w3_n},{b3_n}), \
                 got ({},{},{},{},{},{})",
                weights.w1.len(),
                weights.b1.len(),
                weights.w2.len(),
                weights.b2.len(),
                weights.w3.len(),
                weights.b3.len(),
            )));
        }
        self.w1.host_mut()?.copy_from_slice(weights.w1);
        self.b1.host_mut()?.copy_from_slice(weights.b1);
        self.w2.host_mut()?.copy_from_slice(weights.w2);
        self.b2.host_mut()?.copy_from_slice(weights.b2);
        self.w3.host_mut()?.copy_from_slice(weights.w3);
        self.b3.host_mut()?.copy_from_slice(weights.b3);
        self.w1.upload(queue)?;
        self.b1.upload(queue)?;
        self.w2.upload(queue)?;
        self.b2.upload(queue)?;
        self.w3.upload(queue)?;
        self.b3.upload(queue)?;
        Ok(())
    }

    /// Push a fresh observation vector through to the device. Length
    /// must equal `input_dim`.
    pub fn set_inputs(&mut self, queue: &wgpu::Queue, values: &[Real]) -> Result<()> {
        if values.len() != self.input_dim as usize {
            return Err(Error::Solver(format!(
                "ControllerNeuralMlp::set_inputs: expected {} values, got {}",
                self.input_dim,
                values.len()
            )));
        }
        self.inputs.host_mut()?.copy_from_slice(values);
        self.inputs.upload(queue)?;
        Ok(())
    }

    /// Read the most recent network outputs back to the host. Useful
    /// for UI readouts; the pipeline's scatter pass already lands the
    /// outputs in `joint_force` device-side, so this is purely
    /// diagnostic.
    pub fn download_outputs(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Real>> {
        self.outputs.download(device, queue)?;
        Ok(self.outputs.host()?.to_vec())
    }

    /// Pure forward pass — runs the MLP without scattering into
    /// `joint_force`. Returns once the GPU work is in flight; call
    /// [`Self::download_outputs`] (or `outputs.download(...)`) to
    /// observe the result host-side.
    pub fn forward(&mut self, engine: &Engine) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ControllerNeuralMlp::forward: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerNeuralMlp::forward: no queue".into()))?
            .clone();
        self.kernel.dispatch_forward(
            &device,
            &queue,
            &self.inputs,
            &mut self.outputs,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.w3,
            &self.b3,
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
        )
    }

    /// Pipeline-flavoured step: forward pass, then scatter outputs
    /// into `Control.joint_force` via the registered DOF indices.
    pub fn step(&mut self, engine: &Engine, control: &mut Control) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ControllerNeuralMlp::step: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerNeuralMlp::step: no queue".into()))?
            .clone();
        self.kernel.dispatch_forward(
            &device,
            &queue,
            &self.inputs,
            &mut self.outputs,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.w3,
            &self.b3,
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
        )?;
        self.kernel.dispatch_scatter(
            &device,
            &queue,
            &self.outputs,
            &self.dof_indices,
            &mut control.joint_force,
            self.output_dim,
        )?;
        Ok(())
    }
}

fn expected_weight_dims(
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
) -> (usize, usize, usize, usize, usize, usize) {
    let i = input_dim as usize;
    let h = hidden_dim as usize;
    let o = output_dim as usize;
    (h * i, h, h * h, h, o * h, o)
}
