//! Stage-31 fixed-architecture single-cell LSTM policy controller.
//!
//! Mirrors [`super::ControllerNeuralMlp`] one-for-one in shape, but
//! the cell carries hidden state (`h`) + cell state (`c`) across
//! substeps. The LSTM recurrence is:
//!
//! ```text
//! i_t = σ(W_i · x_t + U_i · h_{t-1} + b_i)
//! f_t = σ(W_f · x_t + U_f · h_{t-1} + b_f)
//! g_t = tanh(W_g · x_t + U_g · h_{t-1} + b_g)
//! o_t = σ(W_o · x_t + U_o · h_{t-1} + b_o)
//! c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
//! h_t = o_t ⊙ tanh(c_t)
//! y_t = W_y · h_t + b_y
//! ```
//!
//! The hidden + cell buffers are owned by this struct and live on the
//! device for the lifetime of the controller; no host round-trip per
//! substep. [`ControllerNeuralLstm::reset_hidden_state`] zeros both
//! when the caller wants to start a fresh trajectory (scene switch /
//! episode boundary).
//!
//! Use case: stateful policies — recurrent balance / locomotion
//! controllers, observation-history aggregators, etc. — that need to
//! remember context across timesteps but should still inference
//! entirely on-device alongside the rest of the actuator pipeline.

use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::kernels::ActuatorNeuralLstmKernel;
use crate::math::Real;
use crate::sim::Control;
use crate::storage::{Buffer, BufferUsage};

/// Construction-time configuration for [`ControllerNeuralLstm`].
#[derive(Debug, Clone, Copy)]
pub struct ControllerNeuralLstmConfig {
    pub input_dim: u32,
    pub hidden_dim: u32,
    pub output_dim: u32,
}

/// Trained LSTM weights, in row-major layout.
///
/// Per gate (`*_i` input, `*_f` forget, `*_g` cell-candidate,
/// `*_o` output):
///   - `w_*` :  `[hidden_dim × input_dim]`
///   - `u_*` :  `[hidden_dim × hidden_dim]`
///   - `b_*` :  `[hidden_dim]`
///
/// Linear projection (`*_y`):
///   - `w_y` :  `[output_dim × hidden_dim]`
///   - `b_y` :  `[output_dim]`
///
/// Use [`ControllerNeuralLstm::expected_weight_dims`] to query the
/// expected slab lengths for a given configuration.
pub struct LstmWeights<'a> {
    pub w_i: &'a [Real],
    pub u_i: &'a [Real],
    pub b_i: &'a [Real],
    pub w_f: &'a [Real],
    pub u_f: &'a [Real],
    pub b_f: &'a [Real],
    pub w_g: &'a [Real],
    pub u_g: &'a [Real],
    pub b_g: &'a [Real],
    pub w_o: &'a [Real],
    pub u_o: &'a [Real],
    pub b_o: &'a [Real],
    pub w_y: &'a [Real],
    pub b_y: &'a [Real],
}

/// Expected weight slab lengths returned by
/// [`ControllerNeuralLstm::expected_weight_dims`]. The fields are in
/// the same order as [`LstmWeights`] so users can iterate / zip.
#[derive(Debug, Clone, Copy)]
pub struct LstmWeightDims {
    pub w_i: usize,
    pub u_i: usize,
    pub b_i: usize,
    pub w_f: usize,
    pub u_f: usize,
    pub b_f: usize,
    pub w_g: usize,
    pub u_g: usize,
    pub b_g: usize,
    pub w_o: usize,
    pub u_o: usize,
    pub b_o: usize,
    pub w_y: usize,
    pub b_y: usize,
}

pub struct ControllerNeuralLstm {
    kernel: ActuatorNeuralLstmKernel,
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
    /// Per-output DOF index — `outputs[i]` lands in
    /// `joint_force[dof_indices[i]]` after the scatter pass.
    dof_indices: Buffer<u32>,
    /// Network input vector. Updated host-side every substep via
    /// [`Self::set_inputs`].
    inputs: Buffer<Real>,
    /// Network output vector. Read back via [`Self::download_outputs`]
    /// for diagnostic UIs.
    outputs: Buffer<Real>,
    /// Hidden state `h_t`. Persists across substeps; only zeroed by
    /// [`Self::reset_hidden_state`].
    h_state: Buffer<Real>,
    /// Cell state `c_t`. Persists across substeps; only zeroed by
    /// [`Self::reset_hidden_state`].
    c_state: Buffer<Real>,
    // Gate weights — input, forget, candidate, output.
    w_i: Buffer<Real>,
    u_i: Buffer<Real>,
    b_i: Buffer<Real>,
    w_f: Buffer<Real>,
    u_f: Buffer<Real>,
    b_f: Buffer<Real>,
    w_g: Buffer<Real>,
    u_g: Buffer<Real>,
    b_g: Buffer<Real>,
    w_o: Buffer<Real>,
    u_o: Buffer<Real>,
    b_o: Buffer<Real>,
    // Linear projection.
    w_y: Buffer<Real>,
    b_y: Buffer<Real>,
}

impl ControllerNeuralLstm {
    /// Allocate buffers + GPU resources for a fresh LSTM. Weights and
    /// hidden / cell states all start at zero — call
    /// [`Self::set_weights`] before stepping the pipeline if you have
    /// pre-trained parameters.
    ///
    /// `dof_indices` maps each network output index to the DOF that
    /// receives the output. Length must equal `output_dim`.
    pub fn new(
        engine: &Engine,
        cfg: ControllerNeuralLstmConfig,
        dof_indices: &[u32],
    ) -> Result<Self> {
        if cfg.input_dim == 0 || cfg.hidden_dim == 0 || cfg.output_dim == 0 {
            return Err(Error::Solver(
                "ControllerNeuralLstm::new: all dims must be > 0".into(),
            ));
        }
        if cfg.hidden_dim > 256 {
            return Err(Error::Solver(
                "ControllerNeuralLstm::new: hidden_dim must be ≤ 256 \
                 (bumped via the WGSL MAX_HIDDEN constant)"
                    .into(),
            ));
        }
        if dof_indices.len() != cfg.output_dim as usize {
            return Err(Error::Solver(format!(
                "ControllerNeuralLstm::new: dof_indices.len()={} expected {}",
                dof_indices.len(),
                cfg.output_dim
            )));
        }
        for i in 0..dof_indices.len() {
            for j in (i + 1)..dof_indices.len() {
                if dof_indices[i] == dof_indices[j] {
                    return Err(Error::Solver(format!(
                        "ControllerNeuralLstm::new: duplicate dof {} at outputs {i} and {j}",
                        dof_indices[i]
                    )));
                }
            }
        }

        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ControllerNeuralLstm: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerNeuralLstm: no GPU queue".into()))?
            .clone();

        let dims = expected_weight_dims(cfg.input_dim, cfg.hidden_dim, cfg.output_dim);

        let mut dof_b = Buffer::from_vec(dof_indices.to_vec(), BufferUsage::Storage);
        let mut inputs =
            Buffer::from_vec(vec![0.0 as Real; cfg.input_dim as usize], BufferUsage::Storage);
        let mut outputs = Buffer::from_vec(
            vec![0.0 as Real; cfg.output_dim as usize],
            BufferUsage::StorageReadWrite,
        );
        // Hidden + cell state — read-write so the kernel can
        // round-trip them in place each step.
        let mut h_state = Buffer::from_vec(
            vec![0.0 as Real; cfg.hidden_dim as usize],
            BufferUsage::StorageReadWrite,
        );
        let mut c_state = Buffer::from_vec(
            vec![0.0 as Real; cfg.hidden_dim as usize],
            BufferUsage::StorageReadWrite,
        );
        let mut w_i = Buffer::from_vec(vec![0.0 as Real; dims.w_i], BufferUsage::Storage);
        let mut u_i = Buffer::from_vec(vec![0.0 as Real; dims.u_i], BufferUsage::Storage);
        let mut b_i = Buffer::from_vec(vec![0.0 as Real; dims.b_i], BufferUsage::Storage);
        let mut w_f = Buffer::from_vec(vec![0.0 as Real; dims.w_f], BufferUsage::Storage);
        let mut u_f = Buffer::from_vec(vec![0.0 as Real; dims.u_f], BufferUsage::Storage);
        let mut b_f = Buffer::from_vec(vec![0.0 as Real; dims.b_f], BufferUsage::Storage);
        let mut w_g = Buffer::from_vec(vec![0.0 as Real; dims.w_g], BufferUsage::Storage);
        let mut u_g = Buffer::from_vec(vec![0.0 as Real; dims.u_g], BufferUsage::Storage);
        let mut b_g = Buffer::from_vec(vec![0.0 as Real; dims.b_g], BufferUsage::Storage);
        let mut w_o = Buffer::from_vec(vec![0.0 as Real; dims.w_o], BufferUsage::Storage);
        let mut u_o = Buffer::from_vec(vec![0.0 as Real; dims.u_o], BufferUsage::Storage);
        let mut b_o = Buffer::from_vec(vec![0.0 as Real; dims.b_o], BufferUsage::Storage);
        let mut w_y = Buffer::from_vec(vec![0.0 as Real; dims.w_y], BufferUsage::Storage);
        let mut b_y = Buffer::from_vec(vec![0.0 as Real; dims.b_y], BufferUsage::Storage);

        dof_b.ensure_device(&device)?;
        inputs.ensure_device(&device)?;
        outputs.ensure_device(&device)?;
        h_state.ensure_device(&device)?;
        c_state.ensure_device(&device)?;
        w_i.ensure_device(&device)?;
        u_i.ensure_device(&device)?;
        b_i.ensure_device(&device)?;
        w_f.ensure_device(&device)?;
        u_f.ensure_device(&device)?;
        b_f.ensure_device(&device)?;
        w_g.ensure_device(&device)?;
        u_g.ensure_device(&device)?;
        b_g.ensure_device(&device)?;
        w_o.ensure_device(&device)?;
        u_o.ensure_device(&device)?;
        b_o.ensure_device(&device)?;
        w_y.ensure_device(&device)?;
        b_y.ensure_device(&device)?;

        dof_b.upload(&queue)?;
        inputs.upload(&queue)?;
        outputs.upload(&queue)?;
        h_state.upload(&queue)?;
        c_state.upload(&queue)?;
        w_i.upload(&queue)?;
        u_i.upload(&queue)?;
        b_i.upload(&queue)?;
        w_f.upload(&queue)?;
        u_f.upload(&queue)?;
        b_f.upload(&queue)?;
        w_g.upload(&queue)?;
        u_g.upload(&queue)?;
        b_g.upload(&queue)?;
        w_o.upload(&queue)?;
        u_o.upload(&queue)?;
        b_o.upload(&queue)?;
        w_y.upload(&queue)?;
        b_y.upload(&queue)?;

        Ok(Self {
            kernel: ActuatorNeuralLstmKernel::new(&device),
            input_dim: cfg.input_dim,
            hidden_dim: cfg.hidden_dim,
            output_dim: cfg.output_dim,
            dof_indices: dof_b,
            inputs,
            outputs,
            h_state,
            c_state,
            w_i,
            u_i,
            b_i,
            w_f,
            u_f,
            b_f,
            w_g,
            u_g,
            b_g,
            w_o,
            u_o,
            b_o,
            w_y,
            b_y,
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

    /// Expected weight slab sizes for the configured architecture.
    pub fn expected_weight_dims(&self) -> LstmWeightDims {
        expected_weight_dims(self.input_dim, self.hidden_dim, self.output_dim)
    }

    /// Upload trained weights. Every slab in `weights` must match the
    /// lengths returned by [`Self::expected_weight_dims`].
    pub fn set_weights(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        weights: &LstmWeights,
    ) -> Result<()> {
        let d = self.expected_weight_dims();
        if weights.w_i.len() != d.w_i
            || weights.u_i.len() != d.u_i
            || weights.b_i.len() != d.b_i
            || weights.w_f.len() != d.w_f
            || weights.u_f.len() != d.u_f
            || weights.b_f.len() != d.b_f
            || weights.w_g.len() != d.w_g
            || weights.u_g.len() != d.u_g
            || weights.b_g.len() != d.b_g
            || weights.w_o.len() != d.w_o
            || weights.u_o.len() != d.u_o
            || weights.b_o.len() != d.b_o
            || weights.w_y.len() != d.w_y
            || weights.b_y.len() != d.b_y
        {
            return Err(Error::Solver(format!(
                "ControllerNeuralLstm::set_weights: dim mismatch — \
                 expected w_i={} u_i={} b_i={} w_f={} u_f={} b_f={} \
                 w_g={} u_g={} b_g={} w_o={} u_o={} b_o={} w_y={} b_y={}; \
                 got {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                d.w_i, d.u_i, d.b_i, d.w_f, d.u_f, d.b_f,
                d.w_g, d.u_g, d.b_g, d.w_o, d.u_o, d.b_o, d.w_y, d.b_y,
                weights.w_i.len(), weights.u_i.len(), weights.b_i.len(),
                weights.w_f.len(), weights.u_f.len(), weights.b_f.len(),
                weights.w_g.len(), weights.u_g.len(), weights.b_g.len(),
                weights.w_o.len(), weights.u_o.len(), weights.b_o.len(),
                weights.w_y.len(), weights.b_y.len(),
            )));
        }
        self.w_i.host_mut()?.copy_from_slice(weights.w_i);
        self.u_i.host_mut()?.copy_from_slice(weights.u_i);
        self.b_i.host_mut()?.copy_from_slice(weights.b_i);
        self.w_f.host_mut()?.copy_from_slice(weights.w_f);
        self.u_f.host_mut()?.copy_from_slice(weights.u_f);
        self.b_f.host_mut()?.copy_from_slice(weights.b_f);
        self.w_g.host_mut()?.copy_from_slice(weights.w_g);
        self.u_g.host_mut()?.copy_from_slice(weights.u_g);
        self.b_g.host_mut()?.copy_from_slice(weights.b_g);
        self.w_o.host_mut()?.copy_from_slice(weights.w_o);
        self.u_o.host_mut()?.copy_from_slice(weights.u_o);
        self.b_o.host_mut()?.copy_from_slice(weights.b_o);
        self.w_y.host_mut()?.copy_from_slice(weights.w_y);
        self.b_y.host_mut()?.copy_from_slice(weights.b_y);
        self.w_i.upload(queue)?;
        self.u_i.upload(queue)?;
        self.b_i.upload(queue)?;
        self.w_f.upload(queue)?;
        self.u_f.upload(queue)?;
        self.b_f.upload(queue)?;
        self.w_g.upload(queue)?;
        self.u_g.upload(queue)?;
        self.b_g.upload(queue)?;
        self.w_o.upload(queue)?;
        self.u_o.upload(queue)?;
        self.b_o.upload(queue)?;
        self.w_y.upload(queue)?;
        self.b_y.upload(queue)?;
        Ok(())
    }

    /// Push a fresh observation vector through to the device. Length
    /// must equal `input_dim`.
    pub fn set_inputs(&mut self, queue: &wgpu::Queue, values: &[Real]) -> Result<()> {
        if values.len() != self.input_dim as usize {
            return Err(Error::Solver(format!(
                "ControllerNeuralLstm::set_inputs: expected {} values, got {}",
                self.input_dim,
                values.len()
            )));
        }
        self.inputs.host_mut()?.copy_from_slice(values);
        self.inputs.upload(queue)?;
        Ok(())
    }

    /// Reset both the hidden state `h` and the cell state `c` to zero
    /// — call on scene switch / episode boundary.
    pub fn reset_hidden_state(&mut self, queue: &wgpu::Queue) -> Result<()> {
        for v in self.h_state.host_mut()? {
            *v = 0.0;
        }
        for v in self.c_state.host_mut()? {
            *v = 0.0;
        }
        self.h_state.upload(queue)?;
        self.c_state.upload(queue)?;
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

    /// Read the current hidden state vector `h_t` back to the host.
    /// Diagnostic / UI only.
    pub fn download_hidden_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Real>> {
        self.h_state.download(device, queue)?;
        Ok(self.h_state.host()?.to_vec())
    }

    /// Read the current cell state vector `c_t` back to the host.
    /// Diagnostic / UI only.
    pub fn download_cell_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Real>> {
        self.c_state.download(device, queue)?;
        Ok(self.c_state.host()?.to_vec())
    }

    /// Pure forward pass — runs the LSTM cell + linear projection and
    /// updates `h`, `c`, `outputs` on-device. Does NOT scatter into
    /// `joint_force`. Call [`Self::download_outputs`] (or the
    /// hidden-state download helpers) to inspect the result host-side.
    pub fn forward(&mut self, engine: &Engine) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ControllerNeuralLstm::forward: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerNeuralLstm::forward: no queue".into()))?
            .clone();
        self.kernel.dispatch_forward(
            &device,
            &queue,
            &self.inputs,
            &mut self.outputs,
            &mut self.h_state,
            &mut self.c_state,
            &self.w_i,
            &self.u_i,
            &self.b_i,
            &self.w_f,
            &self.u_f,
            &self.b_f,
            &self.w_g,
            &self.u_g,
            &self.b_g,
            &self.w_o,
            &self.u_o,
            &self.b_o,
            &self.w_y,
            &self.b_y,
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
            .ok_or_else(|| Error::Gpu("ControllerNeuralLstm::step: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerNeuralLstm::step: no queue".into()))?
            .clone();
        self.kernel.dispatch_forward(
            &device,
            &queue,
            &self.inputs,
            &mut self.outputs,
            &mut self.h_state,
            &mut self.c_state,
            &self.w_i,
            &self.u_i,
            &self.b_i,
            &self.w_f,
            &self.u_f,
            &self.b_f,
            &self.w_g,
            &self.u_g,
            &self.b_g,
            &self.w_o,
            &self.u_o,
            &self.b_o,
            &self.w_y,
            &self.b_y,
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

fn expected_weight_dims(input_dim: u32, hidden_dim: u32, output_dim: u32) -> LstmWeightDims {
    let i = input_dim as usize;
    let h = hidden_dim as usize;
    let o = output_dim as usize;
    LstmWeightDims {
        w_i: h * i,
        u_i: h * h,
        b_i: h,
        w_f: h * i,
        u_f: h * h,
        b_f: h,
        w_g: h * i,
        u_g: h * h,
        b_g: h,
        w_o: h * i,
        u_o: h * h,
        b_o: h,
        w_y: o * h,
        b_y: o,
    }
}
