//! Stage 43 v4 — GPU dispatcher for the per-face SWE numerical flux
//! (HLL / HLLC / Roe), porting the CPU library in
//! [`crate::fem::dg_element::compute_swe_numerical_flux_*`] to WGSL.
//!
//! One pipeline per flux variant share a common bind-group layout:
//!
//! - **group 0** (read-only):
//!     - `state`  — `[n_cells × 3 × 3]` f32 (3 P1 nodes × 3 SWE
//!       components `(h, hu, hv)` per cell).
//!     - `faces`  — `[n_faces × 10]` f32 (packed face record; see
//!       `shaders/dg_swe_flux.wgsl` for the layout).
//! - **group 1** (read-write):
//!     - `residual` — `[n_cells × 3 × 3]` f32 (atomic-add target).
//!     - `uniforms` — `(n_faces: u32, g: f32, ...)`.
//!
//! The kernel uses a CAS-loop atomic add on `atomic<u32>` reinterpreted
//! as `f32` (WGSL has no `atomic<f32>` — same emulation as `mpm.wgsl`).
//! Per-face parallelism, one thread per face, workgroup size 64.
//!
//! See `crates/molla/tests/fem_dg_swe_gpu.rs` for the CPU↔GPU parity
//! test driver.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::core::{Error, Result};

/// SWE numerical flux variants. Matches the three CPU oracles
/// (`compute_swe_numerical_flux_hll` / `_hllc` / `_roe`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DgSweFluxVariant {
    /// HLL — two-wave Riemann (Davis 1988 wave-speed estimate).
    Hll,
    /// HLLC — three-wave with star-region contact (Toro 1994, SWE form).
    Hllc,
    /// Roe — linearised solver with Harten-Hyman entropy fix.
    Roe,
}

/// Number of nodes per P1 triangle cell.
pub const DG_SWE_NODES_PER_CELL: usize = 3;
/// Number of conservative components per SWE node (`h, hu, hv`).
pub const DG_SWE_COMPONENTS: usize = 3;
/// Number of f32 lanes per packed face record (see WGSL shader for the
/// layout — 6 u32 indices reinterpreted as f32 + 3 f32 + 1 pad).
pub const DG_SWE_FACE_STRIDE: usize = 10;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DgSweUniforms {
    n_faces: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    g: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

const WG_SIZE: u32 = 64;

/// Packed per-face record matching `dg_swe_flux.wgsl`. Hosts construct
/// one of these per interior face from the mesh + face geometry, then
/// pass `bytemuck::cast_slice` of a `Vec<DgSweFace>` to
/// [`DgSweFluxKernel::dispatch`].
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DgSweFace {
    /// Cell index on the left side.
    pub cell_l: u32,
    /// Cell index on the right side.
    pub cell_r: u32,
    /// Local node index on the left cell of the first face vertex.
    pub local_l0: u32,
    /// Local node index on the left cell of the second face vertex.
    pub local_l1: u32,
    /// Local node index on the right cell of the first face vertex.
    pub local_r0: u32,
    /// Local node index on the right cell of the second face vertex.
    pub local_r1: u32,
    /// Outward face normal `n_LR.x` (left → right).
    pub nx: f32,
    /// Outward face normal `n_LR.y`.
    pub ny: f32,
    /// Edge length.
    pub length: f32,
    /// Pad lane to round the record up to 40 B (no host meaning).
    pub _pad: f32,
}

const _: () = {
    // Compile-time assertion that the packed face matches the WGSL stride.
    assert!(std::mem::size_of::<DgSweFace>() == DG_SWE_FACE_STRIDE * 4);
};

/// GPU dispatcher for the three SWE flux variants. One instance is
/// reusable across many residual computations — owns one shader module
/// and three pipelines.
pub struct DgSweFluxKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_hll: wgpu::ComputePipeline,
    pipe_hllc: wgpu::ComputePipeline,
    pipe_roe: wgpu::ComputePipeline,

    layout_src: wgpu::BindGroupLayout,
    layout_dst: wgpu::BindGroupLayout,
}

impl DgSweFluxKernel {
    /// Compile the shader and build all three pipelines.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::dg_swe_flux"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dg_swe_flux.wgsl").into()),
        });

        let layout_src = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::dg_swe_flux src"),
            entries: &[ro_storage(0), ro_storage(1)],
        });
        let layout_dst = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::dg_swe_flux dst"),
            entries: &[rw_storage(0), uniform_entry(1)],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::dg_swe_flux layout"),
            bind_group_layouts: &[Some(&layout_src), Some(&layout_dst)],
            immediate_size: 0,
        });

        let mk = |entry: &str, label: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipe_hll = mk("compute_swe_hll_flux_per_face", "molla::dg_swe_flux::hll");
        let pipe_hllc = mk("compute_swe_hllc_flux_per_face", "molla::dg_swe_flux::hllc");
        let pipe_roe = mk("compute_swe_roe_flux_per_face", "molla::dg_swe_flux::roe");

        Self {
            device,
            queue,
            pipe_hll,
            pipe_hllc,
            pipe_roe,
            layout_src,
            layout_dst,
        }
    }

    /// Borrow the device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Borrow the queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Upload `state_f32` to a fresh storage buffer. Convenience helper
    /// for tests + scratch paths; production code should keep state
    /// resident on the device.
    pub fn upload_state(&self, state_f32: &[f32]) -> wgpu::Buffer {
        let bytes = bytemuck::cast_slice::<f32, u8>(state_f32);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dg_swe_flux::state"),
            size: bytes.len().max(4) as u64,
            usage,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.queue.write_buffer(&buf, 0, bytes);
        }
        buf
    }

    /// Upload a packed face list to a fresh storage buffer.
    pub fn upload_faces(&self, faces: &[DgSweFace]) -> wgpu::Buffer {
        let bytes = bytemuck::cast_slice::<DgSweFace, u8>(faces);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dg_swe_flux::faces"),
            size: bytes.len().max(4) as u64,
            usage,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.queue.write_buffer(&buf, 0, bytes);
        }
        buf
    }

    /// Allocate a zero-initialised residual buffer sized for `n_cells`
    /// cells. The buffer is `n_cells * 9 * 4` bytes (3 nodes × 3
    /// components × f32) and is suitable for atomic-add accumulation.
    pub fn alloc_residual(&self, n_cells: usize) -> wgpu::Buffer {
        let bytes = (n_cells * DG_SWE_NODES_PER_CELL * DG_SWE_COMPONENTS).max(1) * 4;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dg_swe_flux::residual"),
            size: bytes as u64,
            usage,
            mapped_at_creation: false,
        });
        // Zero-init via queue.write_buffer (small, ~9N bytes; cheaper
        // than dispatching a separate clear kernel for the test path).
        let zeros = vec![0u8; bytes];
        self.queue.write_buffer(&buf, 0, &zeros);
        buf
    }

    /// Dispatch the chosen flux variant. The caller owns the three
    /// storage buffers — `state` and `faces` are read-only, `residual`
    /// is the atomic-add target (expected to be pre-cleared to zero).
    ///
    /// `n_faces` must equal the number of `DgSweFace` records in
    /// `faces`. `g` is the gravitational acceleration (typically 9.81).
    pub fn dispatch(
        &self,
        variant: DgSweFluxVariant,
        state: &wgpu::Buffer,
        faces: &wgpu::Buffer,
        residual: &wgpu::Buffer,
        n_faces: u32,
        g: f32,
    ) -> Result<()> {
        if n_faces == 0 {
            return Ok(());
        }

        let uniforms = DgSweUniforms {
            n_faces,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            g,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dg_swe_flux uniforms"),
            size: core::mem::size_of::<DgSweUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_src = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::dg_swe_flux src bg"),
            layout: &self.layout_src,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: faces.as_entire_binding(),
                },
            ],
        });
        let bg_dst = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::dg_swe_flux dst bg"),
            layout: &self.layout_dst,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: residual.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline = match variant {
            DgSweFluxVariant::Hll => &self.pipe_hll,
            DgSweFluxVariant::Hllc => &self.pipe_hllc,
            DgSweFluxVariant::Roe => &self.pipe_roe,
        };

        let groups = n_faces.div_ceil(WG_SIZE).max(1);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::dg_swe_flux encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::dg_swe_flux pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg_src, &[]);
            pass.set_bind_group(1, &bg_dst, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Read an `n`-element f32 storage buffer back into a host vec.
    /// Test helper — the production path keeps residuals resident.
    pub fn read_buffer_f32(&self, buf: &wgpu::Buffer, n: u32) -> Result<Vec<f32>> {
        read_storage_buffer_f32(&self.device, &self.queue, buf, n)
    }
}

fn read_storage_buffer_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    n: u32,
) -> Result<Vec<f32>> {
    let bytes = (n as u64).max(1) * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::dg_swe_flux readback staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::dg_swe_flux readback encoder"),
    });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("dg_swe_flux readback poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("dg_swe_flux readback recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("dg_swe_flux readback map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let mut out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    out.truncate(n as usize);
    Ok(out)
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
