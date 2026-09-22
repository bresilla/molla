//! Differentiable mass-spring cloth on the GPU (INSPECTION.md F15 /
//! DiffsimCloth adjoint) — host wrapper around
//! `shaders/cloth_diff.wgsl`.
//!
//! Forward: semi-implicit Euler mass-spring cloth, gather-form (per
//! particle over a spring CSR) — bit-deterministic, so finite
//! differences of the GPU forward are a valid oracle for the analytic
//! reverse pass. Reverse: three passes per step (â_v staging →
//! per-spring stiffness gradient → adjoint step), walking the stored
//! trajectory backwards. Gradients exposed: ∂J/∂k (per spring) and
//! ∂J/∂v₀, ∂J/∂p₀ (per particle).
//!
//! Scope note: this is the Warp-style force-based diffsim cloth (the
//! parallel-friendly formulation Newton's `example_diffsim_cloth`
//! family uses). The CPU `molla-autograd::xpbd_adjoint` remains the
//! oracle for the XPBD Gauss-Seidel formulation, whose sequential
//! reverse is GPU-hostile by construction.

use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    particle_count: u32,
    spring_count: u32,
    step: u32,
    tet_count: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
}

const WG: u32 = 64;

/// One spring: endpoints, rest length, stiffness.
#[derive(Copy, Clone, Debug)]
pub struct ClothSpring {
    pub a: u32,
    pub b: u32,
    pub rest: f32,
    pub k: f32,
}

/// One softbody tet for the volume-preservation term
/// `½·k_v·(V − V₀)²` (V₀ measured from the initial positions).
#[derive(Copy, Clone, Debug)]
pub struct SoftTet {
    pub ids: [u32; 4],
    pub kv: f32,
}

pub struct ClothDiffGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipe_forward: wgpu::ComputePipeline,
    pipe_hat: wgpu::ComputePipeline,
    pipe_stiffness: wgpu::ComputePipeline,
    pipe_volume_stiffness: wgpu::ComputePipeline,
    pipe_backward: wgpu::ComputePipeline,
    bg_uniforms: wgpu::BindGroup,
    bg_topology: wgpu::BindGroup,
    bg_traj: wgpu::BindGroup,
    bg_adj: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    traj_p: wgpu::Buffer,
    /// Held for readback symmetry / future ∂J/∂trajectory probes; the
    /// bind group owns its GPU lifetime.
    #[allow(dead_code)]
    traj_v: wgpu::Buffer,
    adj_p: wgpu::Buffer,
    adj_v: wgpu::Buffer,
    adj_k: wgpu::Buffer,
    adj_kv: wgpu::Buffer,
    spring_k: wgpu::Buffer,
    particle_count: u32,
    spring_count: u32,
    tet_count: u32,
    max_steps: u32,
    dt: f32,
    gravity: [f32; 3],
}

impl ClothDiffGpu {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        inv_mass: &[f32],
        springs: &[ClothSpring],
        max_steps: u32,
        dt: f32,
        gravity: [f32; 3],
    ) -> Result<Self> {
        Self::new_with_tets(
            device,
            queue,
            positions,
            velocities,
            inv_mass,
            springs,
            &[],
            max_steps,
            dt,
            gravity,
        )
    }

    /// Like [`Self::new`] with an additional per-tet volume term
    /// (softbody diffsim). Rest volumes are measured from `positions`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_tets(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        inv_mass: &[f32],
        springs: &[ClothSpring],
        tets: &[SoftTet],
        max_steps: u32,
        dt: f32,
        gravity: [f32; 3],
    ) -> Result<Self> {
        let n = positions.len();
        if velocities.len() != n || inv_mass.len() != n {
            return Err(Error::Solver(
                "cloth_diff: positions/velocities/inv_mass lengths must match".into(),
            ));
        }
        let ns = springs.len();

        // CSR: particle → incident spring ids.
        let mut counts = vec![0u32; n + 1];
        for s in springs {
            counts[s.a as usize + 1] += 1;
            counts[s.b as usize + 1] += 1;
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        let offsets = counts.clone();
        let mut cursor = counts;
        let mut entries = vec![0u32; offsets[n] as usize];
        for (sid, s) in springs.iter().enumerate() {
            for node in [s.a as usize, s.b as usize] {
                entries[cursor[node] as usize] = sid as u32;
                cursor[node] += 1;
            }
        }
        // Tet CSR + rest volumes.
        let nt = tets.len();
        let mut t_counts = vec![0u32; n + 1];
        for t in tets {
            for &id in &t.ids {
                t_counts[id as usize + 1] += 1;
            }
        }
        for i in 1..t_counts.len() {
            t_counts[i] += t_counts[i - 1];
        }
        let t_offsets = t_counts.clone();
        let mut t_cursor = t_counts;
        let mut t_entries = vec![0u32; t_offsets[n] as usize];
        for (tid, t) in tets.iter().enumerate() {
            for &id in &t.ids {
                t_entries[t_cursor[id as usize] as usize] = tid as u32;
                t_cursor[id as usize] += 1;
            }
        }
        let vol = |t: &SoftTet| -> f32 {
            let p = |k: usize| positions[t.ids[k] as usize];
            let e = |a: [f32; 3], b: [f32; 3]| [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let u = e(p(0), p(1));
            let v = e(p(0), p(2));
            let w = e(p(0), p(3));
            (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                + u[2] * (v[0] * w[1] - v[1] * w[0]))
                / 6.0
        };
        let tet_v0: Vec<f32> = tets.iter().map(vol).collect();
        let tet_kvs: Vec<f32> = tets.iter().map(|t| t.kv).collect();
        let tet_id_rows: Vec<[u32; 4]> = tets.iter().map(|t| t.ids).collect();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::cloth_diff"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cloth_diff.wgsl").into()),
        });
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::cloth_diff uniforms"),
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
        let layout_topology = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::cloth_diff topology"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false), // spring_k (optimised parameter)
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, true),
                storage_entry(9, false), // tet_kv (optimised parameter)
                storage_entry(10, true),
                storage_entry(11, true),
            ],
        });
        let layout_traj = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::cloth_diff trajectory"),
            entries: &[storage_entry(0, false), storage_entry(1, false)],
        });
        let layout_adj = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::cloth_diff adjoint"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false), // adj_kv
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::cloth_diff pipeline layout"),
            bind_group_layouts: &[
                Some(&layout_uniforms),
                Some(&layout_topology),
                Some(&layout_traj),
                Some(&layout_adj),
            ],
            immediate_size: 0,
        });
        let mk_pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("molla::cloth_diff::{entry}")),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk_init_u32 = |label: &str, data: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let mk_init_f32 = |label: &str, data: &[f32], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            })
        };
        let a_ids: Vec<u32> = springs.iter().map(|s| s.a).collect();
        let b_ids: Vec<u32> = springs.iter().map(|s| s.b).collect();
        let rests: Vec<f32> = springs.iter().map(|s| s.rest).collect();
        let ks: Vec<f32> = springs.iter().map(|s| s.k).collect();
        let spring_a = mk_init_u32("cloth_diff.spring_a", &a_ids);
        let spring_b = mk_init_u32("cloth_diff.spring_b", &b_ids);
        let spring_rest = mk_init_f32("cloth_diff.rest", &rests, wgpu::BufferUsages::STORAGE);
        let spring_k = mk_init_f32("cloth_diff.k", &ks, usage_rw);
        let inv_mass_buf =
            mk_init_f32("cloth_diff.inv_mass", inv_mass, wgpu::BufferUsages::STORAGE);
        let csr_off = mk_init_u32("cloth_diff.csr_offsets", &offsets);
        let csr_ent = mk_init_u32("cloth_diff.csr_entries", &entries);
        let tet_ids_buf = if nt > 0 {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cloth_diff.tet_ids"),
                contents: bytemuck::cast_slice(&tet_id_rows),
                usage: wgpu::BufferUsages::STORAGE,
            })
        } else {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cloth_diff.tet_ids dummy"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let tet_v0_buf = mk_init_f32(
            "cloth_diff.tet_v0",
            if tet_v0.is_empty() {
                &[0.0][..]
            } else {
                &tet_v0
            },
            wgpu::BufferUsages::STORAGE,
        );
        let tet_kv_buf = mk_init_f32(
            "cloth_diff.tet_kv",
            if tet_kvs.is_empty() {
                &[0.0][..]
            } else {
                &tet_kvs
            },
            usage_rw,
        );
        let tet_csr_off = mk_init_u32("cloth_diff.tet_csr_offsets", &t_offsets);
        let tet_csr_ent = mk_init_u32(
            "cloth_diff.tet_csr_entries",
            if t_entries.is_empty() {
                &[0][..]
            } else {
                &t_entries
            },
        );

        let slab = (n.max(1) * 16) as u64;
        let traj_bytes = slab * (max_steps as u64 + 1);
        let mk_buf = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: usage_rw,
                mapped_at_creation: false,
            })
        };
        let traj_p = mk_buf("cloth_diff.traj_p", traj_bytes);
        let traj_v = mk_buf("cloth_diff.traj_v", traj_bytes);
        // Seed slab 0.
        let p0: Vec<[f32; 4]> = positions.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let v0: Vec<[f32; 4]> = velocities.iter().map(|v| [v[0], v[1], v[2], 0.0]).collect();
        queue.write_buffer(&traj_p, 0, bytemuck::cast_slice(&p0));
        queue.write_buffer(&traj_v, 0, bytemuck::cast_slice(&v0));

        let adj_p = mk_buf("cloth_diff.adj_p", slab);
        let adj_v = mk_buf("cloth_diff.adj_v", slab);
        let adj_vh = mk_buf("cloth_diff.adj_vh", slab);
        let adj_k = mk_buf("cloth_diff.adj_k", (ns.max(1) * 4) as u64);
        let adj_kv = mk_buf("cloth_diff.adj_kv", (nt.max(1) * 4) as u64);

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloth_diff.uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            }
        }
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloth_diff.bg_uniforms"),
            layout: &layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });
        let bg_topology = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloth_diff.bg_topology"),
            layout: &layout_topology,
            entries: &[
                entry(0, &spring_a),
                entry(1, &spring_b),
                entry(2, &spring_rest),
                entry(3, &spring_k),
                entry(4, &inv_mass_buf),
                entry(5, &csr_off),
                entry(6, &csr_ent),
                entry(7, &tet_ids_buf),
                entry(8, &tet_v0_buf),
                entry(9, &tet_kv_buf),
                entry(10, &tet_csr_off),
                entry(11, &tet_csr_ent),
            ],
        });
        let bg_traj = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloth_diff.bg_traj"),
            layout: &layout_traj,
            entries: &[entry(0, &traj_p), entry(1, &traj_v)],
        });
        let bg_adj = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloth_diff.bg_adj"),
            layout: &layout_adj,
            entries: &[
                entry(0, &adj_p),
                entry(1, &adj_v),
                entry(2, &adj_k),
                entry(3, &adj_vh),
                entry(4, &adj_kv),
            ],
        });

        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipe_forward: mk_pipe("forward_step"),
            pipe_hat: mk_pipe("backward_hat"),
            pipe_stiffness: mk_pipe("backward_stiffness"),
            pipe_volume_stiffness: mk_pipe("backward_volume_stiffness"),
            pipe_backward: mk_pipe("backward_step"),
            bg_uniforms,
            bg_topology,
            bg_traj,
            bg_adj,
            uniform_buf,
            traj_p,
            traj_v,
            adj_p,
            adj_v,
            adj_k,
            adj_kv,
            spring_k,
            particle_count: n as u32,
            spring_count: ns as u32,
            tet_count: nt as u32,
            max_steps,
            dt,
            gravity,
        })
    }

    fn write_uniforms(&self, step: u32) {
        let u = Uniforms {
            particle_count: self.particle_count,
            spring_count: self.spring_count,
            step,
            tet_count: self.tet_count,
            dt: self.dt,
            gravity_x: self.gravity[0],
            gravity_y: self.gravity[1],
            gravity_z: self.gravity[2],
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    fn dispatch(&self, pipe: &wgpu::ComputePipeline, groups: u32, label: &str) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bg_uniforms, &[]);
            pass.set_bind_group(1, &self.bg_topology, &[]);
            pass.set_bind_group(2, &self.bg_traj, &[]);
            pass.set_bind_group(3, &self.bg_adj, &[]);
            pass.set_pipeline(pipe);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run `steps` forward steps from slab 0, storing the trajectory.
    pub fn forward(&self, steps: u32) -> Result<()> {
        if steps > self.max_steps {
            return Err(Error::Solver(format!(
                "cloth_diff: steps {steps} exceeds capacity {}",
                self.max_steps
            )));
        }
        let groups = self.particle_count.div_ceil(WG).max(1);
        for t in 0..steps {
            self.write_uniforms(t);
            self.dispatch(&self.pipe_forward, groups, "cloth_diff forward");
        }
        Ok(())
    }

    /// Reverse pass over `steps` stored forward steps. Seeds are
    /// (∂J/∂p_T, ∂J/∂v_T); on return the device buffers hold ∂J/∂p₀,
    /// ∂J/∂v₀ and the accumulated ∂J/∂k.
    pub fn backward(
        &self,
        steps: u32,
        adj_p_final: &[[f32; 3]],
        adj_v_final: &[[f32; 3]],
    ) -> Result<()> {
        if adj_p_final.len() != self.particle_count as usize
            || adj_v_final.len() != self.particle_count as usize
        {
            return Err(Error::Solver("cloth_diff: adjoint seed length".into()));
        }
        let ap: Vec<[f32; 4]> = adj_p_final
            .iter()
            .map(|p| [p[0], p[1], p[2], 0.0])
            .collect();
        let av: Vec<[f32; 4]> = adj_v_final
            .iter()
            .map(|v| [v[0], v[1], v[2], 0.0])
            .collect();
        self.queue
            .write_buffer(&self.adj_p, 0, bytemuck::cast_slice(&ap));
        self.queue
            .write_buffer(&self.adj_v, 0, bytemuck::cast_slice(&av));
        self.queue.write_buffer(
            &self.adj_k,
            0,
            bytemuck::cast_slice(&vec![0.0_f32; self.spring_count.max(1) as usize]),
        );
        self.queue.write_buffer(
            &self.adj_kv,
            0,
            bytemuck::cast_slice(&vec![0.0_f32; self.tet_count.max(1) as usize]),
        );
        let groups_p = self.particle_count.div_ceil(WG).max(1);
        let groups_s = self.spring_count.div_ceil(WG).max(1);
        let groups_t = self.tet_count.div_ceil(WG).max(1);
        for t in (0..steps).rev() {
            self.write_uniforms(t);
            self.dispatch(&self.pipe_hat, groups_p, "cloth_diff hat");
            self.dispatch(&self.pipe_stiffness, groups_s, "cloth_diff dk");
            if self.tet_count > 0 {
                self.dispatch(&self.pipe_volume_stiffness, groups_t, "cloth_diff dkv");
            }
            self.dispatch(&self.pipe_backward, groups_p, "cloth_diff backward");
        }
        Ok(())
    }

    /// Overwrite the spring stiffnesses (optimiser update path).
    pub fn set_stiffness(&self, k: &[f32]) -> Result<()> {
        if k.len() != self.spring_count as usize {
            return Err(Error::Solver("cloth_diff: k length".into()));
        }
        self.queue
            .write_buffer(&self.spring_k, 0, bytemuck::cast_slice(k));
        Ok(())
    }

    fn read_vec4(&self, buf: &wgpu::Buffer, count: usize) -> Result<Vec<[f32; 3]>> {
        let bytes = (count.max(1) * 16) as u64;
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloth_diff.stage"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buf, 0, &stage, 0, bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r.map_err(|e| Error::Gpu(format!("cloth_diff map: {e:?}"))))
                .ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("cloth_diff poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("cloth_diff recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let v: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        stage.unmap();
        Ok(v.into_iter()
            .take(count)
            .map(|r| [r[0], r[1], r[2]])
            .collect())
    }

    /// Positions at trajectory slab `step` (0..=steps).
    pub fn read_positions(&self, step: u32) -> Result<Vec<[f32; 3]>> {
        let n = self.particle_count as usize;
        let offset = (step as u64) * (n as u64) * 16;
        // Copy the slab into a fresh staging read via a sub-slice copy.
        let stage_src = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloth_diff.slab"),
            size: (n.max(1) * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.traj_p, offset, &stage_src, 0, (n.max(1) * 16) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));
        self.read_vec4(&stage_src, n)
    }

    /// ∂J/∂p₀ after [`Self::backward`].
    pub fn read_grad_p0(&self) -> Result<Vec<[f32; 3]>> {
        self.read_vec4(&self.adj_p, self.particle_count as usize)
    }

    /// ∂J/∂v₀ after [`Self::backward`].
    pub fn read_grad_v0(&self) -> Result<Vec<[f32; 3]>> {
        self.read_vec4(&self.adj_v, self.particle_count as usize)
    }

    /// ∂J/∂k_v (per tet) after [`Self::backward`].
    pub fn read_grad_kv(&self) -> Result<Vec<f32>> {
        self.read_scalars(&self.adj_kv, self.tet_count as usize)
    }

    /// ∂J/∂k after [`Self::backward`].
    pub fn read_grad_k(&self) -> Result<Vec<f32>> {
        self.read_scalars(&self.adj_k, self.spring_count as usize)
    }

    fn read_scalars(&self, src: &wgpu::Buffer, ns: usize) -> Result<Vec<f32>> {
        let bytes = (ns.max(1) * 4) as u64;
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloth_diff.stage_k"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(src, 0, &stage, 0, bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r.map_err(|e| Error::Gpu(format!("cloth_diff map k: {e:?}"))))
                .ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("cloth_diff poll k: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("cloth_diff recv k: {e:?}")))??;
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        stage.unmap();
        Ok(v.into_iter().take(ns).collect())
    }
}
