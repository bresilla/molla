//! Stage 40 — Rust dispatcher for the implicit-MPM colored
//! Gauss-Seidel sweep on the GPU. Pairs with
//! `shaders/implicit_mpm.wgsl`.
//!
//! Scope: the host pre-assembles `g_mass`, `k_stencil`, `v_pred`,
//! and the color-group cell-index lists on the CPU (Stage 39A's P2G
//! code path), uploads them, and dispatches the inner solver loop on
//! the GPU. `v_grid` is read back at the end. This is intentionally
//! a "lift the inner loop only" port — the full P2G/G2P scatter on
//! GPU is deferred to a later stage.

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::math::Real;

const WG_SIZE: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    grid_size: u32,
    color_n: u32,
    use_off_diag: u32,
    _pad: u32,
    dt: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

/// Compute pipeline + bind-group layouts for the implicit-MPM GS sweep.
/// Build once per device; reuse across many timesteps.
pub struct ImplicitMpmKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ImplicitMpmKernel {
    /// Build the kernel against `device` + `queue`.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::implicit_mpm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/implicit_mpm.wgsl").into(),
            ),
        });

        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::implicit_mpm io"),
            entries: &[
                ro_storage(0), // g_mass
                ro_storage(1), // k_stencil
                ro_storage(2), // v_pred
                ro_storage(3), // color_cells
                rw_storage(4), // v_grid
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::implicit_mpm uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::implicit_mpm layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::implicit_mpm::gs_sweep"),
            layout: Some(&pipe_layout),
            module: &shader,
            entry_point: Some("gs_sweep"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            layout_io,
            layout_uniforms,
        }
    }

    /// Owned `device` borrow — kept so the parity test can build extra
    /// buffers without re-passing it down.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Owned `queue` borrow.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Run a single colored-GS sweep on the GPU.
    ///
    /// `color_groups[c]` lists the cell indices in color group `c`.
    /// One dispatch per color per iteration; the loop runs
    /// `max_iters` outer iterations (no inner residual check — the
    /// parity test compares final velocities, not iteration count).
    ///
    /// All buffer sizes must match the input shapes documented in the
    /// shader header. `v_grid` is read-write and must already contain
    /// the initial guess (`= v_pred` on the host).
    #[allow(clippy::too_many_arguments)]
    pub fn run_gs_sweeps(
        &self,
        grid_size: u32,
        dt: f32,
        use_off_diag: bool,
        max_iters: u32,
        color_groups: &[Vec<u32>],
        g_mass: &wgpu::Buffer,
        k_stencil: &wgpu::Buffer,
        v_pred: &wgpu::Buffer,
        v_grid: &wgpu::Buffer,
    ) -> Result<()> {
        if color_groups.is_empty() || max_iters == 0 {
            return Ok(());
        }
        // Upload each color group once into its own GPU buffer so the
        // shader can index it without a second copy per iteration.
        let mut color_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(color_groups.len());
        for (c, group) in color_groups.iter().enumerate() {
            let label = format!("molla::implicit_mpm color {c}");
            let bytes: Vec<u8> = if group.is_empty() {
                bytemuck::cast_slice::<u32, u8>(&[0u32]).to_vec()
            } else {
                bytemuck::cast_slice::<u32, u8>(group).to_vec()
            };
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, &bytes);
            color_bufs.push(buf);
        }
        // Per-color uniform buffers (color_n varies); one per color.
        let mut uniform_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(color_groups.len());
        for (c, group) in color_groups.iter().enumerate() {
            let label = format!("molla::implicit_mpm uniforms {c}");
            let uniforms = Uniforms {
                grid_size,
                color_n: group.len() as u32,
                use_off_diag: u32::from(use_off_diag),
                _pad: 0,
                dt,
                _pad2: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            };
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(&uniforms));
            uniform_bufs.push(buf);
        }
        // One outer iteration → one dispatch per color (in order). The
        // colored-GS structure guarantees within a color group all
        // cells are independent, so the per-color dispatch can run all
        // its threads in parallel; the *inter-color* dependency is
        // captured by serialising the dispatches via the encoder.
        for _ in 0..max_iters {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::implicit_mpm enc"),
            });
            for (c, group) in color_groups.iter().enumerate() {
                if group.is_empty() {
                    continue;
                }
                let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("molla::implicit_mpm io bg"),
                    layout: &self.layout_io,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: g_mass.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: k_stencil.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: v_pred.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: color_bufs[c].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: v_grid.as_entire_binding(),
                        },
                    ],
                });
                let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("molla::implicit_mpm u bg"),
                    layout: &self.layout_uniforms,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_bufs[c].as_entire_binding(),
                    }],
                });
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::implicit_mpm pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg_io, &[]);
                pass.set_bind_group(1, &bg_u, &[]);
                let n_groups = (group.len() as u32).div_ceil(WG_SIZE).max(1);
                pass.dispatch_workgroups(n_groups, 1, 1);
            }
            self.queue.submit(std::iter::once(enc.finish()));
        }
        Ok(())
    }
}

/// Upload a host `Vec<f32>` (or `Vec<u32>` via bytemuck cast) as a
/// read-only storage buffer. Convenience helper for the parity test.
pub fn upload_storage_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[f32],
) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    buf
}

/// Upload a host `Vec<vec3<f32>>` (laid out as `[x, y, z, 0]` to match
/// WGSL's `vec3<f32>` storage stride of 16 bytes).
pub fn upload_storage_vec3_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[[f32; 3]],
) -> wgpu::Buffer {
    // WGSL `vec3<f32>` storage buffers use 16-byte stride (`vec3<f32>`
    // is aligned and sized like `vec4<f32>` in storage). Pad each
    // element with a trailing zero so the device sees a tight array.
    let mut padded: Vec<f32> = Vec::with_capacity(data.len() * 4);
    for v in data {
        padded.push(v[0]);
        padded.push(v[1]);
        padded.push(v[2]);
        padded.push(0.0);
    }
    let bytes: &[u8] = bytemuck::cast_slice(&padded);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    buf
}

/// Read back a vec3<f32> storage buffer (with the same 16-byte stride
/// as [`upload_storage_vec3_f32`]) and strip the padding.
pub fn readback_vec3_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    n: usize,
) -> Result<Vec<[Real; 3]>> {
    let bytes = (n * 4 * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::implicit_mpm staging"),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::implicit_mpm readback enc"),
    });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes.max(16));
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("implicit_mpm read poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("implicit_mpm read recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("implicit_mpm read map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let raw: &[f32] = bytemuck::cast_slice(&mapped);
    let mut out: Vec<[Real; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        out.push([
            raw[i * 4] as Real,
            raw[i * 4 + 1] as Real,
            raw[i * 4 + 2] as Real,
        ]);
    }
    drop(mapped);
    staging.unmap();
    Ok(out)
}

// ─── Bind-group entry helpers ───────────────────────────────────────

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
