//! Stage 24c — host-side dispatcher for the Stockham radix-2 FFT
//! (`shaders/fft.wgsl`). 1D / 2D / 3D wrappers; forward and inverse
//! share the same compute pipeline and differ only in a uniform flag.
//!
//! ## Public API
//!
//! - [`FftKernel::new`] — build the pipelines once per [`wgpu::Device`].
//! - [`FftKernel::dispatch_1d`] — N-point complex 1D FFT, host slice
//!   in / host vector out.
//! - [`FftKernel::dispatch_2d`] — H × W complex 2D FFT, row-major.
//! - [`FftKernel::dispatch_3d`] — D × H × W complex 3D FFT, row-major.
//!
//! All sizes must be powers of two (the kernel is radix-2 only). The
//! `inverse: bool` flag flips the twiddle-phase sign; **no
//! normalisation** is applied inside the kernel (so
//! `forward * inverse = N`, matching `rustfft`'s convention).
//!
//! ## Storage layout
//!
//! Inputs and outputs are `[Complex32]` with `Complex32 = (re, im)` as
//! a packed `[f32; 2]`. 2D and 3D buffers are row-major flat slices
//! (e.g. a 256 × 256 image is a single 65536-element slice indexed as
//! `data[y * 256 + x]`).

use crate::core::{Error, Result};
use bytemuck::{Pod, Zeroable};

/// Packed complex single-precision element. Layout matches WGSL's
/// `vec2<f32>` so we can ship raw `&[Complex32]` slices to the GPU
/// without re-packing.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    n: u32,
    half_n: u32,
    pass_idx: u32,
    row_count: u32,
    row_stride: u32,
    inverse: u32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 64;
const TRANSPOSE_TILE: u32 = 8;

/// Built once, dispatched many times. Holds two pipelines:
/// the per-pass Stockham butterfly (`fft_pass`) and a transpose
/// helper (`transpose_2d`) used by the multi-dim wrappers.
pub struct FftKernel {
    pipeline_pass: wgpu::ComputePipeline,
    pipeline_transpose: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl FftKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fft"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fft.wgsl").into()),
        });

        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fft io"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fft uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::fft layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let pipeline_pass = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fft::pass"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fft_pass"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_transpose = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fft::transpose_2d"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("transpose_2d"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline_pass,
            pipeline_transpose,
            layout_io,
            layout_uniforms,
        }
    }

    /// 1D FFT of `n` complex elements. Synchronous: uploads the input,
    /// runs `log2(n)` passes ping-ponging two on-device buffers, reads
    /// the result back into `output`. `output.len()` must equal `n`.
    pub fn dispatch_1d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Complex32],
        output: &mut [Complex32],
        n: usize,
        inverse: bool,
    ) -> Result<()> {
        check_pow2(n, "dispatch_1d")?;
        if input.len() != n {
            return Err(Error::Gpu(format!(
                "fft.dispatch_1d: input.len() = {} but n = {}",
                input.len(),
                n
            )));
        }
        if output.len() != n {
            return Err(Error::Gpu(format!(
                "fft.dispatch_1d: output.len() = {} but n = {}",
                output.len(),
                n
            )));
        }
        if n <= 1 {
            // 0- and 1-point FFTs are identity.
            output.copy_from_slice(input);
            return Ok(());
        }

        let bytes = (n * core::mem::size_of::<Complex32>()) as u64;
        let buf_a = mk_storage(device, "molla::fft buf a", bytes);
        let buf_b = mk_storage(device, "molla::fft buf b", bytes);
        queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(input));

        let final_buf = self.run_1d_passes(device, queue, &buf_a, &buf_b, n, 1, n, inverse)?;
        readback_into(device, queue, &final_buf, output, bytes)?;
        Ok(())
    }

    /// 2D FFT of an `height × width` complex array stored row-major.
    /// `input.len() == output.len() == height * width`. Both
    /// dimensions must be powers of two.
    ///
    /// Implementation: row-FFT (independent 1D-of-`width` per row) →
    /// transpose → row-FFT (independent 1D-of-`height` per row, which
    /// is now the original columns) → transpose back. Works in
    /// f32 throughout; matches `rustfft` row-then-column reference to
    /// 1e−5 on 256×256.
    pub fn dispatch_2d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Complex32],
        output: &mut [Complex32],
        height: usize,
        width: usize,
        inverse: bool,
    ) -> Result<()> {
        check_pow2(width, "dispatch_2d width")?;
        check_pow2(height, "dispatch_2d height")?;
        let total = height * width;
        if input.len() != total || output.len() != total {
            return Err(Error::Gpu(format!(
                "fft.dispatch_2d: buffer sizes {}/{} != {}",
                input.len(),
                output.len(),
                total
            )));
        }
        if total == 0 {
            return Ok(());
        }

        let bytes = (total * core::mem::size_of::<Complex32>()) as u64;
        let buf_a = mk_storage(device, "molla::fft 2d buf a", bytes);
        let buf_b = mk_storage(device, "molla::fft 2d buf b", bytes);
        queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(input));

        // Pass 1: 1D FFT along rows (one independent FFT per row).
        let after_rows: &wgpu::Buffer = if width > 1 {
            self.run_1d_passes(device, queue, &buf_a, &buf_b, width, height, width, inverse)?
        } else {
            &buf_a
        };

        // Transpose to expose columns as contiguous rows. Pick the
        // ping-pong slot that doesn't currently hold the data.
        let scratch: &wgpu::Buffer = if std::ptr::eq(after_rows, &buf_a) {
            &buf_b
        } else {
            &buf_a
        };
        self.run_transpose(device, queue, after_rows, scratch, height, width)?;

        // Pass 2: 1D FFT along the (now transposed) rows = original columns.
        // Layout is now [width][height], so each "row" has `height`
        // complex elements, and there are `width` rows.
        let other: &wgpu::Buffer = if std::ptr::eq(scratch, &buf_a) {
            &buf_b
        } else {
            &buf_a
        };
        let after_cols: &wgpu::Buffer = if height > 1 {
            self.run_1d_passes(device, queue, scratch, other, height, width, height, inverse)?
        } else {
            scratch
        };

        // Transpose back to natural [height][width] layout.
        let final_scratch: &wgpu::Buffer = if std::ptr::eq(after_cols, &buf_a) {
            &buf_b
        } else {
            &buf_a
        };
        self.run_transpose(device, queue, after_cols, final_scratch, width, height)?;

        readback_into(device, queue, final_scratch, output, bytes)?;
        Ok(())
    }

    /// 3D FFT of a `depth × height × width` row-major complex array.
    /// Implementation: 2D FFT on each XY-slice, then 1D FFT along the
    /// Z axis for each (y, x) column. All three dimensions must be
    /// powers of two.
    pub fn dispatch_3d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Complex32],
        output: &mut [Complex32],
        depth: usize,
        height: usize,
        width: usize,
        inverse: bool,
    ) -> Result<()> {
        check_pow2(width, "dispatch_3d width")?;
        check_pow2(height, "dispatch_3d height")?;
        check_pow2(depth, "dispatch_3d depth")?;
        let plane = height * width;
        let total = depth * plane;
        if input.len() != total || output.len() != total {
            return Err(Error::Gpu(format!(
                "fft.dispatch_3d: buffer sizes {}/{} != {}",
                input.len(),
                output.len(),
                total
            )));
        }
        if total == 0 {
            return Ok(());
        }

        // Step A: 2D FFT on each Z-slice. Done independently on the host
        // side by chunking through `dispatch_2d`. Cheap for the sizes
        // this kernel targets (32³ in the tests).
        let mut intermediate: Vec<Complex32> = vec![Complex32::ZERO; total];
        for z in 0..depth {
            let off = z * plane;
            self.dispatch_2d(
                device,
                queue,
                &input[off..off + plane],
                &mut intermediate[off..off + plane],
                height,
                width,
                inverse,
            )?;
        }

        // Step B: 1D FFT along Z. We need to transpose [Z][Y][X] →
        // [Y][X][Z] so each Z-column is contiguous, run a 1D-of-Z FFT
        // per (y,x), then transpose back. Done on host: 3D is small.
        if depth <= 1 {
            output.copy_from_slice(&intermediate);
            return Ok(());
        }

        let mut z_pencils: Vec<Complex32> = vec![Complex32::ZERO; total];
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let src = z * plane + y * width + x;
                    let dst = (y * width + x) * depth + z;
                    z_pencils[dst] = intermediate[src];
                }
            }
        }

        let mut z_pencils_out: Vec<Complex32> = vec![Complex32::ZERO; total];
        // Batch all (y,x) Z-FFTs in one call.
        let pencil_count = height * width;
        let bytes = (total * core::mem::size_of::<Complex32>()) as u64;
        let buf_a = mk_storage(device, "molla::fft 3d z buf a", bytes);
        let buf_b = mk_storage(device, "molla::fft 3d z buf b", bytes);
        queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&z_pencils));
        let after = self.run_1d_passes(
            device,
            queue,
            &buf_a,
            &buf_b,
            depth,
            pencil_count,
            depth,
            inverse,
        )?;
        readback_into(device, queue, &after, &mut z_pencils_out, bytes)?;

        // Transpose [Y][X][Z] back to [Z][Y][X].
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let dst = z * plane + y * width + x;
                    let src = (y * width + x) * depth + z;
                    output[dst] = z_pencils_out[src];
                }
            }
        }
        Ok(())
    }

    // ── Internals ───────────────────────────────────────────────────

    /// Run `log2(n)` butterfly passes, ping-ponging between
    /// `buf_a` (initial input) and `buf_b`. Returns whichever buffer
    /// holds the final result. `row_count` independent 1D FFTs are
    /// run in parallel; they occupy adjacent rows of length `row_stride`
    /// in both buffers (so a 2D row-FFT pass uses `row_stride = width`,
    /// `row_count = height`).
    #[allow(clippy::too_many_arguments)]
    fn run_1d_passes<'a>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buf_a: &'a wgpu::Buffer,
        buf_b: &'a wgpu::Buffer,
        n: usize,
        row_count: usize,
        row_stride: usize,
        inverse: bool,
    ) -> Result<&'a wgpu::Buffer> {
        let l = (n as u32).trailing_zeros();

        // Pre-build uniform buffers per pass — small (32 bytes each)
        // and lets us submit a single command encoder for all passes.
        let mut uniform_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(l as usize);
        let mut bind_groups_io: Vec<wgpu::BindGroup> = Vec::with_capacity(l as usize);
        let mut bind_groups_u: Vec<wgpu::BindGroup> = Vec::with_capacity(l as usize);

        for s in 0..l {
            let (in_buf, out_buf) = if s % 2 == 0 {
                (buf_a, buf_b)
            } else {
                (buf_b, buf_a)
            };

            let uniforms = Uniforms {
                n: n as u32,
                half_n: (n / 2) as u32,
                pass_idx: s,
                row_count: row_count as u32,
                row_stride: row_stride as u32,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
                _pad1: 0,
            };
            let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::fft uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&uniforms));

            let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::fft io bg"),
                layout: &self.layout_io,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
                ],
            });
            let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::fft uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                }],
            });
            uniform_bufs.push(ubuf);
            bind_groups_io.push(bg_io);
            bind_groups_u.push(bg_u);
        }

        let groups_x = ((n / 2) as u32).div_ceil(WORKGROUP_SIZE);
        let groups_y = row_count as u32;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fft encoder"),
        });
        for s in 0..l as usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fft pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_pass);
            pass.set_bind_group(0, &bind_groups_io[s], &[]);
            pass.set_bind_group(1, &bind_groups_u[s], &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Result is in buf_a if l is even, buf_b if odd.
        Ok(if l % 2 == 0 { buf_a } else { buf_b })
    }

    fn run_transpose(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        height: usize, // input rows
        width: usize,  // input cols
    ) -> Result<()> {
        // Reuse the same uniform layout: width → row_count, height → row_stride.
        let uniforms = Uniforms {
            n: 0,
            half_n: 0,
            pass_idx: 0,
            row_count: width as u32,
            row_stride: height as u32,
            inverse: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fft transpose uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fft transpose io"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fft transpose uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
        });

        let groups_x = (width as u32).div_ceil(TRANSPOSE_TILE);
        let groups_y = (height as u32).div_ceil(TRANSPOSE_TILE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fft transpose encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fft transpose"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_transpose);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

// ── helpers ────────────────────────────────────────────────────────

fn check_pow2(n: usize, ctx: &str) -> Result<()> {
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(Error::Gpu(format!(
            "fft: {ctx} requires a power-of-two length; got {n}"
        )));
    }
    Ok(())
}

fn mk_storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn readback_into(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    dst: &mut [Complex32],
    bytes: u64,
) -> Result<()> {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fft readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::fft readback encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &stage, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = stage.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r.map_err(|e| Error::Gpu(format!("fft readback: {e:?}"))))
            .ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("fft readback poll: {e:?}")))?;
    receiver
        .recv()
        .map_err(|e| Error::Gpu(format!("fft readback recv: {e:?}")))??;
    let data = slice.get_mapped_range();
    let bytes_read: &[Complex32] = bytemuck::cast_slice(&data);
    dst.copy_from_slice(bytes_read);
    drop(data);
    stage.unmap();
    Ok(())
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
