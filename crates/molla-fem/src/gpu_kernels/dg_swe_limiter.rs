//! Stage 43 v5 — GPU dispatchers for DG-P1 SWE limiters.
//!
//! These kernels keep limiter stages resident on the device after a GPU
//! flux sweep. They operate on the same flat P1 triangle SWE state
//! layout as [`super::dg_swe_flux`]:
//!
//! `state[cell * 9 + local_node * 3 + component]`, with components
//! `(h, hu, hv)`.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::{FemCell, FemMesh};

/// Fixed-width per-cell metadata for GPU limiter stencils.
///
/// `neighbor*` is either a cell index or `u32::MAX` for a boundary /
/// absent neighbour. `nx, ny` is the precomputed principal normal used
/// by the characteristic limiter; scalar component-wise limiting ignores
/// it.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DgSweLimiterCell {
    pub neighbor0: u32,
    pub neighbor1: u32,
    pub neighbor2: u32,
    pub _pad0: u32,
    pub nx: f32,
    pub ny: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

const _: () = {
    assert!(std::mem::size_of::<DgSweLimiterCell>() == 32);
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DgSweLimiterUniforms {
    n_cells: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    g: f32,
    physical_h_min: f32,
    _pad3: f32,
    _pad4: f32,
}

const WG_SIZE: u32 = 64;

/// GPU dispatcher for component-wise Barth-Jespersen and SWE
/// characteristic limiters.
pub struct DgSweLimiterKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_scalar: wgpu::ComputePipeline,
    pipe_characteristic: wgpu::ComputePipeline,

    layout_src: wgpu::BindGroupLayout,
    layout_dst: wgpu::BindGroupLayout,
}

impl DgSweLimiterKernel {
    /// Compile the WGSL limiter shader and build both pipelines.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::dg_swe_limiter"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/dg_swe_limiter.wgsl").into(),
            ),
        });

        let layout_src = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::dg_swe_limiter src"),
            entries: &[ro_storage(0), ro_storage(1)],
        });
        let layout_dst = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::dg_swe_limiter dst"),
            entries: &[rw_storage(0), uniform_entry(1)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::dg_swe_limiter layout"),
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
        let pipe_scalar = mk(
            "limit_swe_scalar_componentwise",
            "molla::dg_swe_limiter::scalar",
        );
        let pipe_characteristic = mk(
            "limit_swe_characteristic",
            "molla::dg_swe_limiter::characteristic",
        );

        Self {
            device,
            queue,
            pipe_scalar,
            pipe_characteristic,
            layout_src,
            layout_dst,
        }
    }

    /// Upload a flat f32 state buffer.
    pub fn upload_state(&self, state_f32: &[f32]) -> wgpu::Buffer {
        let bytes = bytemuck::cast_slice::<f32, u8>(state_f32);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dg_swe_limiter::state"),
            size: bytes.len().max(4) as u64,
            usage,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.queue.write_buffer(&buf, 0, bytes);
        }
        buf
    }

    /// Upload fixed-width limiter cell metadata.
    pub fn upload_cells(&self, cells: &[DgSweLimiterCell]) -> wgpu::Buffer {
        let bytes = bytemuck::cast_slice::<DgSweLimiterCell, u8>(cells);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dg_swe_limiter::cells"),
            size: bytes.len().max(4) as u64,
            usage,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.queue.write_buffer(&buf, 0, bytes);
        }
        buf
    }

    /// Allocate an output state buffer for `n_cells`.
    pub fn alloc_state(&self, n_cells: usize) -> wgpu::Buffer {
        let bytes = (n_cells * 3 * 3).max(1) * 4;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dg_swe_limiter::state_out"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Dispatch component-wise scalar Barth-Jespersen limiting for the
    /// SWE state. `state_in` and `state_out` must be different buffers:
    /// all cell averages are read from the input snapshot before the
    /// limited state is written to the output.
    pub fn dispatch_scalar(
        &self,
        state_in: &wgpu::Buffer,
        cells: &wgpu::Buffer,
        state_out: &wgpu::Buffer,
        n_cells: u32,
        physical_h_min: f32,
    ) -> Result<()> {
        self.dispatch(
            &self.pipe_scalar,
            state_in,
            cells,
            state_out,
            n_cells,
            9.81,
            physical_h_min,
        )
    }

    /// Dispatch the SWE characteristic limiter. `g` is the gravity
    /// constant used for the shallow-water eigenbasis.
    pub fn dispatch_characteristic(
        &self,
        state_in: &wgpu::Buffer,
        cells: &wgpu::Buffer,
        state_out: &wgpu::Buffer,
        n_cells: u32,
        g: f32,
        physical_h_min: f32,
    ) -> Result<()> {
        self.dispatch(
            &self.pipe_characteristic,
            state_in,
            cells,
            state_out,
            n_cells,
            g,
            physical_h_min,
        )
    }

    /// Read an `n`-element f32 storage buffer back into a host vec.
    pub fn read_buffer_f32(&self, buf: &wgpu::Buffer, n: u32) -> Result<Vec<f32>> {
        read_storage_buffer_f32(&self.device, &self.queue, buf, n)
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        state_in: &wgpu::Buffer,
        cells: &wgpu::Buffer,
        state_out: &wgpu::Buffer,
        n_cells: u32,
        g: f32,
        physical_h_min: f32,
    ) -> Result<()> {
        if n_cells == 0 {
            return Ok(());
        }
        let uniforms = DgSweLimiterUniforms {
            n_cells,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            g,
            physical_h_min,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dg_swe_limiter uniforms"),
            size: core::mem::size_of::<DgSweLimiterUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_src = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::dg_swe_limiter src bg"),
            layout: &self.layout_src,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cells.as_entire_binding(),
                },
            ],
        });
        let bg_dst = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::dg_swe_limiter dst bg"),
            layout: &self.layout_dst,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let groups = n_cells.div_ceil(WG_SIZE).max(1);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::dg_swe_limiter encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::dg_swe_limiter pass"),
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
}

/// Pack limiter metadata from a triangle DG mesh and a face-neighbour
/// map (for example from [`crate::build_cell_neighbours`]).
#[must_use]
pub fn pack_dg_swe_limiter_cells(
    mesh: &FemMesh,
    cell_neighbours: &[Vec<u32>],
) -> Vec<DgSweLimiterCell> {
    assert_eq!(
        mesh.cells.len(),
        cell_neighbours.len(),
        "limiter neighbour count must match mesh cell count"
    );
    mesh.cells
        .iter()
        .enumerate()
        .map(|(ci, cell)| {
            let mut nbs = [u32::MAX; 3];
            for (slot, &nb) in cell_neighbours[ci].iter().take(3).enumerate() {
                nbs[slot] = nb;
            }
            let n = principal_normal_2d(mesh, cell);
            DgSweLimiterCell {
                neighbor0: nbs[0],
                neighbor1: nbs[1],
                neighbor2: nbs[2],
                _pad0: 0,
                nx: n.x as f32,
                ny: n.y as f32,
                _pad1: 0.0,
                _pad2: 0.0,
            }
        })
        .collect()
}

fn principal_normal_2d(mesh: &FemMesh, cell: &FemCell) -> molla_math::Vec3 {
    let FemCell::Tri(v) = cell else {
        return molla_math::Vec3::X;
    };
    let centroid = (mesh.vertices[v[0] as usize]
        + mesh.vertices[v[1] as usize]
        + mesh.vertices[v[2] as usize])
        / 3.0;
    let mut n_avg = molla_math::Vec3::ZERO;
    for &[a, b] in &[[v[0], v[1]], [v[1], v[2]], [v[2], v[0]]] {
        let pa = mesh.vertices[a as usize];
        let pb = mesh.vertices[b as usize];
        let mid = (pa + pb) * 0.5;
        let raw = edge_normal_2d(pa, pb);
        let out = if raw.dot(mid - centroid) >= 0.0 {
            raw
        } else {
            -raw
        };
        n_avg += out;
    }
    let len = (n_avg.x * n_avg.x + n_avg.y * n_avg.y).sqrt();
    if len > 1.0e-12 {
        molla_math::Vec3::new(n_avg.x / len, n_avg.y / len, 0.0)
    } else {
        molla_math::Vec3::X
    }
}

fn edge_normal_2d(a: molla_math::Vec3, b: molla_math::Vec3) -> molla_math::Vec3 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0e-30 {
        return molla_math::Vec3::ZERO;
    }
    molla_math::Vec3::new(dy / len, -dx / len, 0.0)
}

fn read_storage_buffer_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    n: u32,
) -> Result<Vec<f32>> {
    let bytes = (n as u64).max(1) * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::dg_swe_limiter readback staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::dg_swe_limiter readback encoder"),
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
        .map_err(|e| Error::Gpu(format!("dg_swe_limiter readback poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("dg_swe_limiter readback recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("dg_swe_limiter readback map: {e:?}")))?;
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
