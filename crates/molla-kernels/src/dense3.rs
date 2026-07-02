//! Stage 24a — batched 3×3 SVD / QR / symmetric-eig kernels.
//!
//! Wraps `shaders/dense3_svd.wgsl`. One [`Dense3Kernel`] holds the
//! pipelines for all three entry points; pick which one to dispatch
//! at call time via [`Dense3Op`]. The shader operates on row-major
//! 9-float matrices; this wrapper packs/unpacks `Mat3` for callers.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};
use molla_math::{Mat3, Real, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    matrix_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Which decomposition the kernel should run.
#[derive(Copy, Clone, Debug)]
pub enum Dense3Op {
    /// SVD: outputs `(U, σ, V)` packed across the three output buffers
    /// (σ is the first three floats of the middle buffer).
    Svd,
    /// QR: outputs `(Q, R, _identity)`.
    Qr,
    /// Symmetric eigendecomposition: outputs `(V, λ, _identity)`.
    Eig,
}

/// Result of a batched 3×3 decomposition.
#[derive(Clone, Debug)]
pub struct Dense3Result {
    /// Left factor — `U` for SVD, `Q` for QR, eigenvectors `V` for eig.
    pub left: Vec<Mat3>,
    /// Middle factor:
    ///   * SVD: singular values per matrix (`Vec3`).
    ///   * QR: full upper-triangular `R`.
    ///   * Eig: eigenvalues per matrix (`Vec3`).
    pub middle_mats: Vec<Mat3>,
    /// Convenience view of `middle_mats`'s diagonal-or-vector slot
    /// (only meaningful for SVD + Eig — for QR all entries are zero).
    pub diag: Vec<Vec3>,
    /// Right factor — `V` for SVD; identity for QR + eig.
    pub right: Vec<Mat3>,
}

pub struct Dense3Kernel {
    pipeline_svd: wgpu::ComputePipeline,
    pipeline_qr: wgpu::ComputePipeline,
    pipeline_eig: wgpu::ComputePipeline,
    pipeline_adj_svd: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    layout_adj_io: wgpu::BindGroupLayout,
}

impl Dense3Kernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::dense3"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dense3_svd.wgsl").into()),
        });

        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::dense3 io"),
            entries: &[
                storage_entry(0, true),  // input
                storage_entry(1, false), // out left
                storage_entry(2, false), // out middle
                storage_entry(3, false), // out right
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::dense3 uniforms"),
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
            label: Some("molla::dense3 layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let mk = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        // ── Adjoint SVD pipeline (separate module + bind-group). ───
        let adj_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::dense3 adj"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dense3_adj.wgsl").into()),
        });
        let layout_adj_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::dense3 adj io"),
            entries: &[
                storage_entry(0, true),  // A
                storage_entry(1, true),  // U
                storage_entry(2, true),  // sigma
                storage_entry(3, true),  // V
                storage_entry(4, true),  // adj_U
                storage_entry(5, true),  // adj_sigma
                storage_entry(6, true),  // adj_V
                storage_entry(7, false), // adj_A (output)
            ],
        });
        let adj_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::dense3 adj layout"),
            bind_group_layouts: &[&layout_adj_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline_adj_svd = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::dense3::adj_svd"),
            layout: Some(&adj_pipeline_layout),
            module: &adj_shader,
            entry_point: Some("adj_svd_kernel"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline_svd: mk("svd_kernel", "molla::dense3::svd"),
            pipeline_qr: mk("qr_kernel", "molla::dense3::qr"),
            pipeline_eig: mk("eig_kernel", "molla::dense3::eig"),
            pipeline_adj_svd,
            layout_io,
            layout_uniforms,
            layout_adj_io,
        }
    }

    /// Run the chosen decomposition over `input` and read the result
    /// back to host. Synchronous — submits + maps in one call. Suited
    /// for parity tests + small batches; high-throughput callers
    /// should drive the pipelines directly with persistent buffers.
    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        op: Dense3Op,
        input: &[Mat3],
    ) -> Result<Dense3Result> {
        let n = input.len();
        if n == 0 {
            return Ok(Dense3Result {
                left: vec![],
                middle_mats: vec![],
                diag: vec![],
                right: vec![],
            });
        }
        let floats_per_mat = 9usize;
        let total_floats = n * floats_per_mat;
        let bytes = (total_floats * 4) as u64;

        // Pack input as f32 row-major.
        let mut packed: Vec<f32> = Vec::with_capacity(total_floats);
        for m in input {
            for i in 0..3 {
                for j in 0..3 {
                    packed.push(m.col(j)[i] as f32);
                }
            }
        }

        let usage_storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk_buf = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: usage_storage,
                mapped_at_creation: false,
            })
        };

        let in_buf = mk_buf("molla::dense3 input");
        let out_l = mk_buf("molla::dense3 out left");
        let out_m = mk_buf("molla::dense3 out middle");
        let out_r = mk_buf("molla::dense3 out right");
        queue.write_buffer(&in_buf, 0, bytemuck::cast_slice(&packed));

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::dense3 io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_m.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_r.as_entire_binding(),
                },
            ],
        });

        let uniforms = Uniforms {
            matrix_count: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dense3 uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::dense3 uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline = match op {
            Dense3Op::Svd => &self.pipeline_svd,
            Dense3Op::Qr => &self.pipeline_qr,
            Dense3Op::Eig => &self.pipeline_eig,
        };
        let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::dense3 encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::dense3 pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // Stage outputs into mappable buffers.
        let mk_stage = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let stage_l = mk_stage("molla::dense3 stage left");
        let stage_m = mk_stage("molla::dense3 stage middle");
        let stage_r = mk_stage("molla::dense3 stage right");
        encoder.copy_buffer_to_buffer(&out_l, 0, &stage_l, 0, bytes);
        encoder.copy_buffer_to_buffer(&out_m, 0, &stage_m, 0, bytes);
        encoder.copy_buffer_to_buffer(&out_r, 0, &stage_r, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let read = |buf: &wgpu::Buffer| -> Result<Vec<f32>> {
            let slice = buf.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                sender
                    .send(r.map_err(|e| Error::Gpu(format!("dense3 readback: {e:?}"))))
                    .ok();
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("dense3 readback poll: {e:?}")))?;
            receiver
                .recv()
                .map_err(|e| Error::Gpu(format!("dense3 readback recv: {e:?}")))??;
            let data = slice.get_mapped_range();
            let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            buf.unmap();
            Ok(v)
        };
        let raw_l = read(&stage_l)?;
        let raw_m = read(&stage_m)?;
        let raw_r = read(&stage_r)?;

        // Unpack — row-major 9 floats each.
        let unpack_mat = |raw: &[f32], idx: usize| -> Mat3 {
            let base = idx * 9;
            Mat3::from_cols(
                Vec3::new(
                    raw[base] as Real,
                    raw[base + 3] as Real,
                    raw[base + 6] as Real,
                ),
                Vec3::new(
                    raw[base + 1] as Real,
                    raw[base + 4] as Real,
                    raw[base + 7] as Real,
                ),
                Vec3::new(
                    raw[base + 2] as Real,
                    raw[base + 5] as Real,
                    raw[base + 8] as Real,
                ),
            )
        };
        let unpack_diag = |raw: &[f32], idx: usize| -> Vec3 {
            let base = idx * 9;
            Vec3::new(
                raw[base] as Real,
                raw[base + 1] as Real,
                raw[base + 2] as Real,
            )
        };

        let mut left = Vec::with_capacity(n);
        let mut middle_mats = Vec::with_capacity(n);
        let mut diag = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);
        for i in 0..n {
            left.push(unpack_mat(&raw_l, i));
            middle_mats.push(unpack_mat(&raw_m, i));
            diag.push(unpack_diag(&raw_m, i));
            right.push(unpack_mat(&raw_r, i));
        }
        Ok(Dense3Result {
            left,
            middle_mats,
            diag,
            right,
        })
    }
}

impl Dense3Kernel {
    /// Reverse-mode adjoint SVD over a batch of matrices, operating on
    /// pre-allocated wgpu buffers. Caller owns the buffers — this just
    /// records the dispatch onto the supplied encoder.
    ///
    /// All buffers must hold `n * 9` `f32` values in row-major layout
    /// (matching the forward `Dense3Op::Svd` output convention), except
    /// `sigma_buf` and `adj_sigma_buf` which use the first 3 of every 9
    /// slot (see `dense3_svd.wgsl::store_vec`). The remaining 6 floats
    /// of those buffers are unread.
    ///
    /// `adj_a_buf` is written; the other seven inputs are read-only.
    /// Dispatches one workgroup of size 64 per `WORKGROUP_SIZE` matrices.
    #[allow(clippy::too_many_arguments)]
    pub fn adj_svd3_batch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: &wgpu::Buffer,
        u_buf: &wgpu::Buffer,
        sigma_buf: &wgpu::Buffer,
        v_buf: &wgpu::Buffer,
        adj_u_buf: &wgpu::Buffer,
        adj_sigma_buf: &wgpu::Buffer,
        adj_v_buf: &wgpu::Buffer,
        adj_a_buf: &wgpu::Buffer,
        uniform_buf: &wgpu::Buffer,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::dense3 adj io bg"),
            layout: &self.layout_adj_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sigma_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adj_u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adj_sigma_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: adj_v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: adj_a_buf.as_entire_binding(),
                },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::dense3 adj uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let groups = n.div_ceil(WORKGROUP_SIZE);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::dense3 adj pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_adj_svd);
        pass.set_bind_group(0, &bg_io, &[]);
        pass.set_bind_group(1, &bg_u, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        Ok(())
    }

    /// Blocking convenience wrapper around [`Self::adj_svd3_batch`].
    /// Allocates all buffers, dispatches once, and reads `adj_A` back
    /// to host as `Vec<Mat3>`. Used by tests + small-batch callers.
    #[allow(clippy::too_many_arguments)]
    pub fn adj_svd3_dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        a: &[Mat3],
        u: &[Mat3],
        sigma: &[Vec3],
        v: &[Mat3],
        adj_u: &[Mat3],
        adj_sigma: &[Vec3],
        adj_v: &[Mat3],
    ) -> Result<Vec<Mat3>> {
        let n = a.len();
        assert_eq!(u.len(), n);
        assert_eq!(sigma.len(), n);
        assert_eq!(v.len(), n);
        assert_eq!(adj_u.len(), n);
        assert_eq!(adj_sigma.len(), n);
        assert_eq!(adj_v.len(), n);
        if n == 0 {
            return Ok(vec![]);
        }

        let floats_per_mat = 9usize;
        let bytes = (n * floats_per_mat * 4) as u64;
        let usage_storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk_buf = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: usage_storage,
                mapped_at_creation: false,
            })
        };

        let pack_mats = |slice: &[Mat3]| -> Vec<f32> {
            let mut packed: Vec<f32> = Vec::with_capacity(n * floats_per_mat);
            for m in slice {
                for i in 0..3 {
                    for j in 0..3 {
                        packed.push(m.col(j)[i] as f32);
                    }
                }
            }
            packed
        };
        let pack_vecs = |slice: &[Vec3]| -> Vec<f32> {
            // First 3 floats per slot = (x, y, z), pad to 9.
            let mut packed: Vec<f32> = vec![0.0; n * floats_per_mat];
            for (i, vv) in slice.iter().enumerate() {
                packed[i * 9] = vv.x as f32;
                packed[i * 9 + 1] = vv.y as f32;
                packed[i * 9 + 2] = vv.z as f32;
            }
            packed
        };

        let a_buf = mk_buf("adj_svd a");
        let u_buf = mk_buf("adj_svd u");
        let sigma_buf = mk_buf("adj_svd sigma");
        let v_buf = mk_buf("adj_svd v");
        let adj_u_buf = mk_buf("adj_svd adj_u");
        let adj_sigma_buf = mk_buf("adj_svd adj_sigma");
        let adj_v_buf = mk_buf("adj_svd adj_v");
        let adj_a_buf = mk_buf("adj_svd adj_a");

        queue.write_buffer(&a_buf, 0, bytemuck::cast_slice(&pack_mats(a)));
        queue.write_buffer(&u_buf, 0, bytemuck::cast_slice(&pack_mats(u)));
        queue.write_buffer(&sigma_buf, 0, bytemuck::cast_slice(&pack_vecs(sigma)));
        queue.write_buffer(&v_buf, 0, bytemuck::cast_slice(&pack_mats(v)));
        queue.write_buffer(&adj_u_buf, 0, bytemuck::cast_slice(&pack_mats(adj_u)));
        queue.write_buffer(
            &adj_sigma_buf,
            0,
            bytemuck::cast_slice(&pack_vecs(adj_sigma)),
        );
        queue.write_buffer(&adj_v_buf, 0, bytemuck::cast_slice(&pack_mats(adj_v)));

        let uniforms = Uniforms {
            matrix_count: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adj_svd uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adj_svd encoder"),
        });
        self.adj_svd3_batch(
            device,
            &mut encoder,
            &a_buf,
            &u_buf,
            &sigma_buf,
            &v_buf,
            &adj_u_buf,
            &adj_sigma_buf,
            &adj_v_buf,
            &adj_a_buf,
            &uniform_buf,
            n as u32,
        )?;

        // Stage adj_a for readback.
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adj_svd stage"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&adj_a_buf, 0, &stage, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = stage.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("adj_svd readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("adj_svd readback poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("adj_svd readback recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let raw: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        stage.unmap();

        // Unpack 9-float row-major.
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 9;
            out.push(Mat3::from_cols(
                Vec3::new(
                    raw[base] as Real,
                    raw[base + 3] as Real,
                    raw[base + 6] as Real,
                ),
                Vec3::new(
                    raw[base + 1] as Real,
                    raw[base + 4] as Real,
                    raw[base + 7] as Real,
                ),
                Vec3::new(
                    raw[base + 2] as Real,
                    raw[base + 5] as Real,
                    raw[base + 8] as Real,
                ),
            ));
        }
        Ok(out)
    }
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
