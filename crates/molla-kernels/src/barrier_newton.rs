//! C0 phase 2 — GPU barrier-Newton contact (host driver around
//! `shaders/barrier_newton.wgsl`).
//!
//! f32 port of `molla-solvers::barrier_contact` (the f64 oracle and
//! parity reference). The host drives the outer Newton loop and the
//! CCD-truncated Armijo backtracking with 4-byte scalar readbacks;
//! gradient, Hessian-vector products, the full PCG (with the f32
//! convergence freeze), the conservative TOI min-reduction, trial
//! energies, and the min-gap invariant scan all run on device.

use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    vertex_count: u32,
    spring_count: u32,
    obstacle_count: u32,
    tri_count: u32,
    h: f32,
    dhat: f32,
    kappa_scale: f32,
    line_alpha: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    mu_k: f32,
    edge_count: u32,
    pair_capacity: u32,
    tet_count: u32,
    _pad3: u32,
    mu_s_scale: f32,
    epsv: f32,
    strain_limit: f32,
    strain_scale: f32,
}

const WG: u32 = 64;

/// Analytic obstacle (mirror of the CPU oracle's enum).
#[derive(Copy, Clone, Debug)]
pub enum BarrierGpuObstacle {
    HalfSpace { n: [f32; 3], offset: f32 },
    Sphere { center: [f32; 3], radius: f32 },
}

/// One stretch spring (mirror of the CPU oracle).
#[derive(Copy, Clone, Debug)]
pub struct BarrierGpuSpring {
    pub a: u32,
    pub b: u32,
    pub rest: f32,
    pub k: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct BarrierGpuParams {
    pub dhat: f32,
    pub kappa_scale: f32,
    pub newton_tol: f32,
    pub max_newton_iters: usize,
    pub cg_iters: usize,
    pub gravity: [f32; 3],
    /// Kinetic Coulomb μ for obstacle contacts (0 = off).
    pub friction_mu: f32,
    /// μ_s = scale·μ_k, scale ≥ 1.
    pub friction_mu_s_scale: f32,
    /// Stick-well width as a velocity (e = ε_v·h).
    pub friction_epsv: f32,
    /// Hard strain limit (len/rest); 0 disables.
    pub strain_limit: f32,
    pub strain_barrier_scale: f32,
}

impl Default for BarrierGpuParams {
    fn default() -> Self {
        Self {
            dhat: 1.0e-3,
            kappa_scale: 1.0e3,
            newton_tol: 1.0e-2,
            max_newton_iters: 12,
            cg_iters: 60,
            gravity: [0.0, -9.81, 0.0],
            friction_mu: 0.0,
            friction_mu_s_scale: 1.0,
            friction_epsv: 1.0e-3,
            strain_limit: 0.0,
            strain_barrier_scale: 10.0,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BarrierGpuStats {
    pub newton_iters: usize,
    /// Smallest gap at any ACCEPTED iterate this step.
    pub min_gap: f32,
    pub energy: f32,
    pub hit_iter_cap: bool,
}

pub struct BarrierNewtonGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
    bg_uniforms: wgpu::BindGroup,
    bg_topology: wgpu::BindGroup,
    bg_state: wgpu::BindGroup,
    bg_work: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    x_pos: wgpu::Buffer,
    vel: wgpu::Buffer,
    scalars: wgpu::Buffer,
    vertex_count: u32,
    spring_count: u32,
    obstacle_count: u32,
    tri_count: u32,
    tet_count: u32,
    edge_count: u32,
    pair_capacity: u32,
    pair_count_buf: wgpu::Buffer,
    mass_max: f32,
    pub params: BarrierGpuParams,
}

const ENTRIES: &[&str] = &[
    "find_pairs_vt",
    "find_pairs_ee",
    "energy_pairs",
    "friction_lag",
    "predict",
    "gradient",
    "hessian_mul",
    "cg_init",
    "cg_dot_init",
    "cg_dot_dhd",
    "cg_update_pr",
    "cg_dot_rz",
    "cg_update_d",
    "dot_grad_p",
    "ccd_alpha",
    "make_trial",
    "energy_vertex",
    "energy_spring",
    "reduce_energy_gap",
    "accept_trial",
    "commit",
];

impl BarrierNewtonGpu {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[[f32; 3]],
        masses: &[f32],
        springs: &[BarrierGpuSpring],
        obstacles: &[BarrierGpuObstacle],
        params: BarrierGpuParams,
    ) -> Result<Self> {
        Self::new_with_mesh(
            device,
            queue,
            positions,
            masses,
            springs,
            obstacles,
            &[],
            &[],
            params,
        )
    }

    /// Phase 3b — self-collision: `triangles` become VT/EE barrier
    /// pairs discovered ON DEVICE each Newton iterate; `pinned_idx`
    /// vertices are true DoF removals (see the CPU oracle).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_mesh(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[[f32; 3]],
        masses: &[f32],
        springs: &[BarrierGpuSpring],
        obstacles: &[BarrierGpuObstacle],
        triangles: &[[u32; 3]],
        pinned_idx: &[u32],
        params: BarrierGpuParams,
    ) -> Result<Self> {
        Self::new_full(
            device,
            queue,
            positions,
            masses,
            springs,
            obstacles,
            triangles,
            &[],
            pinned_idx,
            params,
        )
    }

    /// Full constructor — adds volumetric tets `(ids, k_v)`; rest
    /// volumes are measured from `positions`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[[f32; 3]],
        masses: &[f32],
        springs: &[BarrierGpuSpring],
        obstacles: &[BarrierGpuObstacle],
        triangles: &[[u32; 3]],
        tets: &[([u32; 4], f32)],
        pinned_idx: &[u32],
        params: BarrierGpuParams,
    ) -> Result<Self> {
        let n = positions.len();
        if masses.len() != n {
            return Err(Error::Solver("barrier_newton: mass length".into()));
        }
        // Spring CSR.
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
        // Obstacle rows.
        let mut ob_rows: Vec<[f32; 4]> = Vec::with_capacity(obstacles.len() * 2);
        for ob in obstacles {
            match *ob {
                BarrierGpuObstacle::HalfSpace { n, offset } => {
                    ob_rows.push([0.0, n[0], n[1], n[2]]);
                    ob_rows.push([offset, 0.0, 0.0, 0.0]);
                }
                BarrierGpuObstacle::Sphere { center, radius } => {
                    ob_rows.push([1.0, center[0], center[1], center[2]]);
                    ob_rows.push([radius, 0.0, 0.0, 0.0]);
                }
            }
        }
        if ob_rows.is_empty() {
            ob_rows.push([0.0; 4]);
            ob_rows.push([0.0; 4]);
        }
        // Unique edges from the triangle set (same as the CPU oracle).
        let mut edge_set = std::collections::BTreeSet::new();
        for t in triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edge_set.insert((a.min(b), a.max(b)));
            }
        }
        let edge_rows: Vec<[u32; 4]> = edge_set.iter().map(|&(a, b)| [a, b, 0, 0]).collect();
        let tri_rows: Vec<[u32; 4]> = triangles.iter().map(|t| [t[0], t[1], t[2], 0]).collect();
        let mut pinned_mask = vec![0u32; n];
        for &i in pinned_idx {
            pinned_mask[i as usize] = 1;
        }
        let pair_capacity = (32 * (n + edge_rows.len())).max(64) as u32;
        // Tet rows + rest volumes measured from the initial positions.
        let tet_id_rows: Vec<[u32; 4]> = tets.iter().map(|(ids, _)| *ids).collect();
        let tet_kv_v0: Vec<[f32; 2]> = tets
            .iter()
            .map(|(ids, kv)| {
                let p = |k: usize| positions[ids[k] as usize];
                let sub3 = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                let e1 = sub3(p(1), p(0));
                let e2 = sub3(p(2), p(0));
                let e3 = sub3(p(3), p(0));
                let cx = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let v0 = (cx[0] * e3[0] + cx[1] * e3[1] + cx[2] * e3[2]) / 6.0;
                [*kv, v0]
            })
            .collect();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::barrier_newton"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/barrier_newton.wgsl").into()),
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
            label: Some("molla::barrier_newton uniforms"),
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
            label: Some("molla::barrier_newton topology"),
            entries: &(0..14).map(|i| storage_entry(i, true)).collect::<Vec<_>>(),
        });
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::barrier_newton state"),
            entries: &(0..5).map(|i| storage_entry(i, false)).collect::<Vec<_>>(),
        });
        let layout_work = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::barrier_newton work"),
            entries: &(0..11).map(|i| storage_entry(i, false)).collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::barrier_newton pl"),
            bind_group_layouts: &[
                &layout_uniforms,
                &layout_topology,
                &layout_state,
                &layout_work,
            ],
            push_constant_ranges: &[],
        });
        let mut pipes = std::collections::HashMap::new();
        for &entry in ENTRIES {
            pipes.insert(
                entry,
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("molla::barrier_newton::{entry}")),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            );
        }

        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let init_u32 = |label: &str, data: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let init_f32 = |label: &str, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let a_ids: Vec<u32> = springs.iter().map(|s| s.a).collect();
        let b_ids: Vec<u32> = springs.iter().map(|s| s.b).collect();
        let rests: Vec<f32> = springs.iter().map(|s| s.rest).collect();
        let ks: Vec<f32> = springs.iter().map(|s| s.k).collect();
        let one = [0.0f32];
        let spring_a = init_u32(
            "bn.spring_a",
            if a_ids.is_empty() { &[0][..] } else { &a_ids },
        );
        let spring_b = init_u32(
            "bn.spring_b",
            if b_ids.is_empty() { &[0][..] } else { &b_ids },
        );
        let rest_buf = init_f32("bn.rest", if rests.is_empty() { &one[..] } else { &rests });
        let k_buf = init_f32("bn.k", if ks.is_empty() { &one[..] } else { &ks });
        let inv_mass_buf = init_f32("bn.inv_mass_reserved", &vec![0.0; n.max(1)]);
        let mass_buf = init_f32("bn.mass", masses);
        let csr_off = init_u32("bn.csr_off", &offsets);
        let csr_ent = init_u32(
            "bn.csr_ent",
            if entries.is_empty() {
                &[0][..]
            } else {
                &entries
            },
        );
        let ob_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bn.obstacles"),
            contents: bytemuck::cast_slice(&ob_rows),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_u4 = [[0u32; 4]];
        let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bn.tris"),
            contents: bytemuck::cast_slice(if tri_rows.is_empty() {
                &dummy_u4[..]
            } else {
                &tri_rows
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let edge_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bn.edges"),
            contents: bytemuck::cast_slice(if edge_rows.is_empty() {
                &dummy_u4[..]
            } else {
                &edge_rows
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pinned_buf = init_u32("bn.pinned", &pinned_mask);
        let tet_ids_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bn.tet_ids"),
            contents: bytemuck::cast_slice(if tet_id_rows.is_empty() {
                &dummy_u4[..]
            } else {
                &tet_id_rows
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_f2 = [[0.0f32; 2]];
        let tet_kv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bn.tet_kv_v0"),
            contents: bytemuck::cast_slice(if tet_kv_v0.is_empty() {
                &dummy_f2[..]
            } else {
                &tet_kv_v0
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let slab = (n.max(1) * 16) as u64;
        let mk = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: usage_rw,
                mapped_at_creation: false,
            })
        };
        let x_pos = mk("bn.x", slab);
        let vel = mk("bn.v", slab);
        let x_tilde = mk("bn.x_tilde", slab);
        let x_iter = mk("bn.x_iter", slab);
        let x_try = mk("bn.x_try", slab);
        let grad = mk("bn.grad", slab);
        let p_dir = mk("bn.p", slab);
        let cg_r = mk("bn.r", slab);
        let cg_d = mk("bn.d", slab);
        let cg_hd = mk("bn.hd", slab);
        let part = mk(
            "bn.part",
            ((n + springs.len() + pair_capacity as usize).max(1) * 4) as u64,
        );
        let scalars = mk("bn.scalars", 64);
        let pair_meta = mk("bn.pair_meta", u64::from(pair_capacity) * 16);
        let pair_wts = mk("bn.pair_wts", u64::from(pair_capacity) * 16);
        let pair_count_buf = mk("bn.pair_count", 16);
        let fric_lag = mk(
            "bn.fric_lag",
            (n.max(1) * obstacles.len().max(1) * 16) as u64,
        );
        let p0: Vec<[f32; 4]> = positions.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        queue.write_buffer(&x_pos, 0, bytemuck::cast_slice(&p0));
        queue.write_buffer(&vel, 0, &vec![0u8; slab as usize]);

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn.uniforms"),
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
            label: Some("bn.bg_uniforms"),
            layout: &layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });
        let bg_topology = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bn.bg_topology"),
            layout: &layout_topology,
            entries: &[
                entry(0, &spring_a),
                entry(1, &spring_b),
                entry(2, &rest_buf),
                entry(3, &k_buf),
                entry(4, &inv_mass_buf),
                entry(5, &mass_buf),
                entry(6, &csr_off),
                entry(7, &csr_ent),
                entry(8, &ob_buf),
                entry(9, &tri_buf),
                entry(10, &edge_buf),
                entry(11, &pinned_buf),
                entry(12, &tet_ids_buf),
                entry(13, &tet_kv_buf),
            ],
        });
        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bn.bg_state"),
            layout: &layout_state,
            entries: &[
                entry(0, &x_pos),
                entry(1, &vel),
                entry(2, &x_tilde),
                entry(3, &x_iter),
                entry(4, &x_try),
            ],
        });
        let bg_work = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bn.bg_work"),
            layout: &layout_work,
            entries: &[
                entry(0, &grad),
                entry(1, &p_dir),
                entry(2, &cg_r),
                entry(3, &cg_d),
                entry(4, &cg_hd),
                entry(5, &part),
                entry(6, &scalars),
                entry(7, &pair_meta),
                entry(8, &pair_wts),
                entry(9, &pair_count_buf),
                entry(10, &fric_lag),
            ],
        });

        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipes,
            bg_uniforms,
            bg_topology,
            bg_state,
            bg_work,
            uniform_buf,
            x_pos,
            vel,
            scalars,
            vertex_count: n as u32,
            spring_count: springs.len() as u32,
            obstacle_count: obstacles.len() as u32,
            tri_count: triangles.len() as u32,
            tet_count: tets.len() as u32,
            edge_count: edge_rows.len() as u32,
            pair_capacity,
            pair_count_buf,
            mass_max: masses.iter().cloned().fold(0.0, f32::max),
            params,
        })
    }

    fn write_uniforms(&self, h: f32, line_alpha: f32) {
        let u = Uniforms {
            vertex_count: self.vertex_count,
            spring_count: self.spring_count,
            obstacle_count: self.obstacle_count,
            tri_count: self.tri_count,
            h,
            dhat: self.params.dhat,
            kappa_scale: self.params.kappa_scale,
            line_alpha,
            gravity_x: self.params.gravity[0],
            gravity_y: self.params.gravity[1],
            gravity_z: self.params.gravity[2],
            mu_k: self.params.friction_mu,
            edge_count: self.edge_count,
            pair_capacity: self.pair_capacity,
            tet_count: self.tet_count,
            _pad3: 0,
            mu_s_scale: self.params.friction_mu_s_scale,
            epsv: self.params.friction_epsv,
            strain_limit: self.params.strain_limit,
            strain_scale: self.params.strain_barrier_scale,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    fn run(&self, names: &[(&str, u32)]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bn.encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bn.pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bg_uniforms, &[]);
            pass.set_bind_group(1, &self.bg_topology, &[]);
            pass.set_bind_group(2, &self.bg_state, &[]);
            pass.set_bind_group(3, &self.bg_work, &[]);
            for &(name, groups) in names {
                pass.set_pipeline(&self.pipes[name]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn read_scalars(&self) -> Result<[f32; 16]> {
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn.stage"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.scalars, 0, &stage, 0, 64);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r.map_err(|e| Error::Gpu(format!("bn map: {e:?}"))))
                .ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("bn poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("bn recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        stage.unmap();
        let mut out = [0.0f32; 16];
        out.copy_from_slice(&v[..16]);
        Ok(out)
    }

    /// Evaluate energy + min-gap + grad_inf of `x_iter + alpha·p` into
    /// the scalars buffer.
    fn eval_trial(&self, h: f32, alpha: f32) {
        self.write_uniforms(h, alpha);
        let gv = self.vertex_count.div_ceil(WG).max(1);
        let gs = self.spring_count.div_ceil(WG).max(1);
        let mut plan: Vec<(&str, u32)> = vec![
            ("make_trial", gv),
            ("energy_vertex", gv),
            ("energy_spring", gs),
        ];
        if self.tri_count > 0 {
            plan.push(("energy_pairs", self.pair_capacity.div_ceil(WG).max(1)));
        }
        plan.push(("reduce_energy_gap", 1));
        self.run(&plan);
    }

    /// One implicit step. Mirrors `BarrierShell::step`.
    pub fn step(&self, h: f32) -> Result<BarrierGpuStats> {
        let gv = self.vertex_count.div_ceil(WG).max(1);
        self.write_uniforms(h, 0.0);
        self.run(&[("predict", gv)]);
        if self.params.friction_mu > 0.0 && self.obstacle_count > 0 {
            let gf = (self.vertex_count * self.obstacle_count)
                .div_ceil(WG)
                .max(1);
            self.run(&[("friction_lag", gf)]);
        }

        let mut stats = BarrierGpuStats {
            min_gap: f32::INFINITY,
            ..Default::default()
        };
        // Same characteristic scale as the CPU oracle: the inertia
        // term of moving the heaviest vertex by one dhat. newton_tol
        // is much looser than the f64 oracle's — f32 gradient noise
        // makes 1e-6-relative unreachable (measured: the solve stalls
        // at ~1e-2 relative and burns the full iteration budget).
        let tol = self.params.newton_tol * (self.mass_max * self.params.dhat).max(1.0e-20);

        for it in 0..self.params.max_newton_iters {
            // Rebuild the self-pair active set ON DEVICE, then
            // gradient + baseline energy/gap/grad_inf (α = 0 trial).
            self.write_uniforms(h, 0.0);
            if self.tri_count > 0 {
                self.queue.write_buffer(&self.pair_count_buf, 0, &[0u8; 16]);
                let g_vt = (self.vertex_count * self.tri_count).div_ceil(WG).max(1);
                let g_ee = (self.edge_count * self.edge_count).div_ceil(WG).max(1);
                self.run(&[("find_pairs_vt", g_vt), ("find_pairs_ee", g_ee)]);
            }
            self.run(&[("gradient", gv)]);
            self.eval_trial(h, 0.0);
            let s = self.read_scalars()?;
            let (e0, gap0, g_inf) = (s[7], s[8], s[9]);
            if s[10] != 0.0 {
                return Err(Error::Solver(
                    "barrier_newton: self-pair buffer overflow — raise pair capacity".into(),
                ));
            }
            stats.min_gap = stats.min_gap.min(gap0);
            if g_inf <= tol {
                stats.newton_iters = it;
                stats.energy = e0;
                return self.finish_step(h, stats);
            }
            stats.newton_iters = it + 1;

            // Device PCG (fixed iterations; the freeze makes overshoot
            // a no-op — the APIC lesson).
            // One submission for the entire PCG: dispatches in a
            // compute pass are ordered, and the freeze turns
            // past-convergence iterations into no-ops.
            let mut cg_plan: Vec<(&str, u32)> = vec![("cg_init", gv), ("cg_dot_init", 1)];
            for _ in 0..self.params.cg_iters {
                cg_plan.extend_from_slice(&[
                    ("hessian_mul", gv),
                    ("cg_dot_dhd", 1),
                    ("cg_update_pr", gv),
                    ("cg_dot_rz", 1),
                    ("cg_update_d", gv),
                ]);
            }
            self.run(&cg_plan);

            // CCD cap + slope, then Armijo backtracking.
            self.run(&[("ccd_alpha", 1), ("dot_grad_p", 1)]);
            let s2 = self.read_scalars()?;
            let alpha_max = s2[6];
            let slope = s2[1];
            let mut alpha = alpha_max;
            let mut accepted = false;
            for _ in 0..30 {
                self.eval_trial(h, alpha);
                let st = self.read_scalars()?;
                if st[7] <= e0 + 1.0e-4 * alpha * slope {
                    stats.min_gap = stats.min_gap.min(st[8]);
                    self.run(&[("accept_trial", gv)]);
                    accepted = true;
                    stats.energy = st[7];
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                stats.hit_iter_cap = true;
                break;
            }
            if it + 1 == self.params.max_newton_iters {
                stats.hit_iter_cap = true;
            }
        }
        self.finish_step(h, stats)
    }

    fn finish_step(&self, h: f32, stats: BarrierGpuStats) -> Result<BarrierGpuStats> {
        self.write_uniforms(h, 0.0);
        let gv = self.vertex_count.div_ceil(WG).max(1);
        self.run(&[("commit", gv)]);
        Ok(stats)
    }

    fn read_vec3(&self, buf: &wgpu::Buffer) -> Result<Vec<[f32; 3]>> {
        let n = self.vertex_count as usize;
        let bytes = (n.max(1) * 16) as u64;
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn.stage_v"),
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
            tx.send(r.map_err(|e| Error::Gpu(format!("bn map v: {e:?}"))))
                .ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("bn poll v: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("bn recv v: {e:?}")))??;
        let data = slice.get_mapped_range();
        let v: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        stage.unmap();
        Ok(v.into_iter().take(n).map(|r| [r[0], r[1], r[2]]).collect())
    }

    pub fn read_positions(&self) -> Result<Vec<[f32; 3]>> {
        self.read_vec3(&self.x_pos)
    }

    pub fn read_velocities(&self) -> Result<Vec<[f32; 3]>> {
        self.read_vec3(&self.vel)
    }

    pub fn set_velocities(&self, v: &[[f32; 3]]) -> Result<()> {
        if v.len() != self.vertex_count as usize {
            return Err(Error::Solver("barrier_newton: velocity length".into()));
        }
        let rows: Vec<[f32; 4]> = v.iter().map(|v| [v[0], v[1], v[2], 0.0]).collect();
        self.queue
            .write_buffer(&self.vel, 0, bytemuck::cast_slice(&rows));
        Ok(())
    }
}
