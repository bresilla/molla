use super::{CutReport, StepConfig, StrandModel, invalid};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_geometry::convex_cut::CutPlane;
use molla_kernels::strands::{
    KRYLOV_BYTES,
    BondRecord, BondStateRecord, FiberRecord, MotionRecord, StrandDispatch, StrandParams,
    StrandProgram,
};
use molla_math::{Mat3, Quat, Vec3};
use std::sync::Arc;

fn v4(v: Vec3) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, 0.0]
}
fn m4(m: Mat3) -> [[f32; 4]; 3] {
    [v4(m.x_axis), v4(m.y_axis), v4(m.z_axis)]
}
fn vec3(v: [f32; 4]) -> Vec3 {
    Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
}
fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("GPU input exceeds f32 range"));
    }
    Ok(())
}

struct Packed {
    fibers: Vec<FiberRecord>,
    bonds: Vec<BondRecord>,
    motions: Vec<MotionRecord>,
    damage: Vec<BondStateRecord>,
    vertices: Vec<[f32; 4]>,
    colors: Vec<u32>,
    groups: Vec<(u32, u32)>,
    topology: Vec<u32>,
    islands: [u32; 4],
    island_max: u32,
    tree: [u32; 4],
}

impl Packed {
    fn new(model: &StrandModel, contacts: bool) -> Result<Self> {
        let mut packed = Self {
            fibers: Vec::new(),
            bonds: Vec::new(),
            motions: Vec::new(),
            damage: Vec::new(),
            vertices: Vec::new(),
            colors: Vec::new(),
            groups: Vec::new(),
            topology: Vec::new(),
            islands: [0; 4],
            island_max: 0,
            tree: [0; 4],
        };
        for fiber in model.fibers() {
            fiber.validate()?;
            let mut com_mass = v4(fiber.mass.local_com);
            com_mass[3] = fiber.inverse_mass() as f32;
            let inverse = if fiber.pin.is_some() {
                Mat3::ZERO
            } else {
                fiber.mass.inverse_inertia()?
            };
            let mut inertia = m4(fiber.mass.inertia);
            inertia[0][3] = fiber
                .shape
                .vertices()
                .iter()
                .map(|v| v.distance(fiber.mass.local_com))
                .fold(0.0, f64::max) as f32;
            let mut record = FiberRecord {
                com_mass,
                inertia,
                inverse: m4(inverse),
                vertices: [
                    u32::try_from(packed.vertices.len())
                        .map_err(|_| invalid("GPU vertex index overflow"))?,
                    fiber.shape.vertices().len() as u32,
                    0,
                    0,
                ],
                surface: [0; 4],
            };
            if contacts {
                if fiber.shape.vertices().len() > 64
                    || fiber.shape.faces().len() > 64
                    || fiber.shape.faces().iter().any(|f| f.vertices.len() > 32)
                {
                    return Err(invalid(
                        "GPU contact cell exceeds 64 vertices/faces or 32 vertices per face",
                    ));
                }
                record.surface[0] = u32::try_from(packed.topology.len())
                    .map_err(|_| invalid("contact topology overflow"))?;
                record.surface[1] = fiber.shape.faces().len() as u32;
                for (face_index, face) in fiber.shape.faces().iter().enumerate() {
                    let normal = v4(molla_geometry::cell_contact::face_normal(
                        &fiber.shape,
                        face_index,
                    ));
                    packed.topology.extend([
                        face.vertices.len() as u32,
                        normal[0].to_bits(),
                        normal[1].to_bits(),
                        normal[2].to_bits(),
                    ]);
                    packed.topology.extend(&face.vertices);
                }
                let edges = molla_geometry::cell_contact::edges(&fiber.shape);
                record.surface[2] = u32::try_from(packed.topology.len())
                    .map_err(|_| invalid("contact topology overflow"))?;
                record.surface[3] = edges.len() as u32;
                for edge in edges {
                    packed.topology.extend(edge);
                }
            }
            finite(&record.com_mass)?;
            finite(bytemuck::cast_slice(&record.inertia))?;
            finite(bytemuck::cast_slice(&record.inverse))?;
            if fiber.pin.is_none() && !record.com_mass[3].is_normal() {
                return Err(invalid("GPU inverse mass underflow"));
            }
            for &vertex in fiber.shape.vertices() {
                let v = v4(vertex);
                finite(&v)?;
                packed.vertices.push(v);
            }
            let motion = MotionRecord {
                center: v4(fiber.center()),
                rotation: fiber.motion.pose.rotation.to_array().map(|v| v as f32),
                velocity: v4(fiber.motion.linear_velocity),
                omega: v4(fiber.motion.angular_velocity),
            };
            finite(bytemuck::cast_slice(std::slice::from_ref(&motion)))?;
            packed.fibers.push(record);
            packed.motions.push(motion);
        }
        for bond in model.bonds() {
            let ids = bond.fibers.map(|id| model.indices[&id] as u32);
            let record = BondRecord {
                fibers: [ids[0], ids[1], 0, 0],
                anchor0: v4(bond.anchors[0]),
                anchor1: v4(bond.anchors[1]),
                rest_offset: v4(bond.rest_offset),
                rest_rotation: bond.rest_rotation.to_array().map(|v| v as f32),
                translation: v4(bond.material.translation),
                rotation: v4(bond.material.rotation),
                force_limit: v4(bond.material.force_limit),
                torque_limit: v4(bond.material.torque_limit),
                material: [
                    bond.material.damage_rate as f32,
                    bond.material.damping as f32,
                    0.0,
                    0.0,
                ],
            };
            for values in [
                record.anchor0,
                record.anchor1,
                record.rest_offset,
                record.rest_rotation,
                record.translation,
                record.rotation,
                record.force_limit,
                record.torque_limit,
                record.material,
            ] {
                finite(&values)?;
            }
            for values in [
                record.translation,
                record.rotation,
                record.force_limit,
                record.torque_limit,
            ] {
                if values[..3].iter().any(|v| !v.is_normal() || *v <= 0.0) {
                    return Err(invalid("GPU material underflow"));
                }
            }
            packed.bonds.push(record);
            packed.damage.push(BondStateRecord {
                lambda0: [0.0; 4],
                lambda1: [0.0; 4],
                damage: [bond.damage as f32, 0.0, 0.0, 0.0],
            });
        }
        for group in model.colors() {
            packed
                .groups
                .push((packed.colors.len() as u32, group.len() as u32));
            packed.colors.extend(group.iter().map(|&j| j as u32));
        }
        let mut adjacent = vec![Vec::new(); packed.fibers.len()];
        for (j, bond) in packed.bonds.iter().enumerate() {
            for side in 0..2 {
                adjacent[bond.fibers[side] as usize].push((2 * j + side) as u32);
            }
        }
        for (fiber, edges) in packed.fibers.iter_mut().zip(adjacent) {
            fiber.vertices[2] = u32::try_from(packed.colors.len())
                .map_err(|_| invalid("GPU adjacency overflow"))?;
            fiber.vertices[3] = edges.len() as u32;
            packed.colors.extend(edges);
        }
        let mut visited = vec![false; packed.fibers.len()];
        let mut local = vec![0; packed.fibers.len()];
        let mut ranges = vec![0];
        let mut members = Vec::new();
        for root in 0..packed.fibers.len() {
            if visited[root] {
                continue;
            }
            let mut pending = vec![root];
            visited[root] = true;
            let mut component = Vec::new();
            if packed.fibers.len() <= 128 {
                component.extend(0..packed.fibers.len());
                visited.fill(true);
                pending.clear();
            }
            while let Some(i) = pending.pop() {
                component.push(i);
                let f = packed.fibers[i];
                for j in 0..f.vertices[3] {
                    let b =
                        packed.bonds[(packed.colors[(f.vertices[2] + j) as usize] / 2) as usize];
                    for neighbor in b.fibers[..2].iter().copied().map(|v| v as usize) {
                        if !visited[neighbor] {
                            visited[neighbor] = true;
                            pending.push(neighbor);
                        }
                    }
                }
            }
            component.sort_unstable();
            packed.island_max = packed.island_max.max(component.len() as u32);
            for (slot, i) in component.into_iter().enumerate() {
                local[i] = slot as u32;
                members.push(i as u32);
            }
            ranges.push(members.len() as u32);
        }
        packed.islands[0] =
            u32::try_from(packed.colors.len()).map_err(|_| invalid("island index overflow"))?;
        packed.colors.extend(local);
        packed.islands[1] =
            u32::try_from(packed.colors.len()).map_err(|_| invalid("island index overflow"))?;
        packed.islands[3] = ranges.len() as u32 - 1;
        packed.colors.extend(ranges);
        packed.islands[2] =
            u32::try_from(packed.colors.len()).map_err(|_| invalid("island index overflow"))?;
        packed.colors.extend(members);
        packed.plan_trees()?;
        Ok(packed)
    }

    /// Spanning forest of each island's bond graph for the exact tree preconditioner:
    /// parent per fiber, per-island `(first level, level count)`, level boundaries into
    /// the elimination order, and the order itself (deepest level first). Pinned fibers
    /// are decoupled in the coupled system and stay roots without tree edges.
    fn plan_trees(&mut self) -> Result<()> {
        let n = self.fibers.len();
        let islands = self.islands[3] as usize;
        let ranges = self.colors[self.islands[1] as usize..][..=islands].to_vec();
        let members = self.colors[self.islands[2] as usize..][..n].to_vec();
        let pinned = |i: usize| self.fibers[i].com_mass[3] == 0.0;
        let mut parent = vec![u32::MAX; n];
        let mut depth = vec![0usize; n];
        let mut seen = vec![false; n];
        let mut table = Vec::with_capacity(2 * islands);
        let mut levels = Vec::new();
        let mut order = Vec::with_capacity(n);
        for island in 0..islands {
            let cells = &members[ranges[island] as usize..ranges[island + 1] as usize];
            let mut visit = Vec::with_capacity(cells.len());
            for &root in cells {
                let root = root as usize;
                if seen[root] {
                    continue;
                }
                seen[root] = true;
                let start = visit.len();
                visit.push(root);
                let mut head = start;
                while head < visit.len() {
                    let i = visit[head];
                    head += 1;
                    if pinned(i) {
                        continue;
                    }
                    let f = self.fibers[i];
                    for j in 0..f.vertices[3] {
                        let edge = self.colors[(f.vertices[2] + j) as usize] as usize;
                        let bond = self.bonds[edge / 2];
                        let other = bond.fibers[1 - edge % 2] as usize;
                        if !seen[other] && !pinned(other) {
                            seen[other] = true;
                            parent[other] = i as u32;
                            depth[other] = depth[i] + 1;
                            visit.push(other);
                        }
                    }
                }
            }
            let deepest = visit.iter().map(|&i| depth[i]).max().unwrap_or(0);
            let first = u32::try_from(levels.len()).map_err(|_| invalid("tree plan overflow"))?;
            for level in (0..=deepest).rev() {
                levels.push(order.len() as u32);
                order.extend(visit.iter().filter(|&&i| depth[i] == level).map(|&i| i as u32));
            }
            levels.push(order.len() as u32);
            table.extend([first, deepest as u32 + 1]);
        }
        let mut append = |values: Vec<u32>| -> Result<u32> {
            let offset =
                u32::try_from(self.colors.len()).map_err(|_| invalid("tree plan overflow"))?;
            self.colors.extend(values);
            Ok(offset)
        };
        self.tree = [append(parent)?, append(table)?, append(levels)?, append(order)?];
        Ok(())
    }
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    minimum: u64,
    usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer> {
    let size = (data.len() as u64).max(minimum).max(16).next_multiple_of(4);
    let limits = device.limits();
    if size > limits.max_buffer_size
        || (usage.contains(wgpu::BufferUsages::STORAGE)
            && size > limits.max_storage_buffer_binding_size)
    {
        return Err(invalid("GPU strand buffer exceeds device capacity"));
    }
    let result = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: !data.is_empty()
        }
    );
    if !data.is_empty() {
        result
            .slice(0..data.len() as u64)
            .get_mapped_range_mut()
            .map_err(|e| Error::Gpu(format!("strand setup mapping: {e}")))?
            .copy_from_slice(molla_core::gpu_metrics::upload_slice(data));
        result.unmap();
    }
    Ok(result)
}

/// GPU-resident fiber state; host synchronization occurs only on snapshots or edits.
pub struct GpuStrands {
    owner: Arc<wgpu::Device>,
    template: StrandModel,
    dispatch: StrandDispatch,
    buffers: Vec<wgpu::Buffer>,
    staging: wgpu::Buffer,
    state_bytes: u64,
    bond_bytes: u64,
    limits: GpuStrandLimits,
    last_coupled: Option<CoupledDiagnostics>,
    pair_count: u32,
    uploads: wgpu::util::StagingBelt,
    islands: [u32; 4],
    island_max: u32,
    tree: [u32; 4],
    published_revision: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct CoupledDiagnostics {
    /// Maximum iteration count across independent elastic islands.
    pub iterations: u32,
    /// Maximum relative residual across independent elastic islands.
    pub relative_residual: f32,
    /// Maximum initial norm squared across independent elastic islands.
    pub initial_preconditioned_norm_squared: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuStrandLimits {
    pub fibers: usize,
    pub bonds: usize,
    pub vertices: usize,
    /// Maximum all-pairs contact slots; zero leaves convex pair contact disabled.
    pub contact_pairs: usize,
}

impl Default for GpuStrandLimits {
    fn default() -> Self {
        Self {
            fibers: 1_000_000,
            bonds: 4_000_000,
            vertices: 32_000_000,
            contact_pairs: 0,
        }
    }
}

impl GpuStrands {
    pub fn new(engine: &Engine, model: StrandModel) -> Result<Self> {
        Self::with_limits(engine, model, GpuStrandLimits::default())
    }

    pub fn with_limits(
        engine: &Engine,
        model: StrandModel,
        limits: GpuStrandLimits,
    ) -> Result<Self> {
        engine.check_device()?;
        let vertices = model
            .fibers()
            .iter()
            .try_fold(0usize, |sum, f| sum.checked_add(f.shape.vertices().len()))
            .ok_or_else(|| invalid("GPU vertex count overflow"))?;
        if limits.fibers == 0
            || limits.vertices == 0
            || model.fibers().len() > limits.fibers
            || model.bonds().len() > limits.bonds
            || model.bonds().len() > u32::MAX as usize / 3
            || vertices > limits.vertices
        {
            return Err(invalid("GPU strand topology exceeds configured capacity"));
        }
        let pair_count = if limits.contact_pairs == 0 {
            0
        } else {
            let n = model.fibers().len();
            if n > 65535 {
                return Err(invalid("GPU contact pair indexing exceeds 65535 cells"));
            }
            let pairs = n
                .checked_mul(n - 1)
                .map(|n| n / 2)
                .ok_or_else(|| invalid("contact pair count overflow"))?;
            if pairs > limits.contact_pairs || pairs > u32::MAX as usize {
                return Err(invalid("GPU contact pair capacity exceeded"));
            }
            if pairs.div_ceil(64)
                > engine
                    .device()
                    .unwrap()
                    .limits()
                    .max_compute_workgroups_per_dimension as usize
            {
                return Err(invalid("GPU contact dispatch capacity exceeded"));
            }
            pairs as u32
        };
        let packed = Packed::new(&model, limits.contact_pairs > 0)?;
        let program = engine.cached_gpu_program("strands-v22", StrandProgram::new)?;
        engine.checked_gpu_setup(|device| {
            let storage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
            let state_bytes = (packed.motions.len() * std::mem::size_of::<MotionRecord>()) as u64;
            let bond_bytes = (packed.damage.len() * std::mem::size_of::<BondStateRecord>()) as u64;
            let stride = device.limits().min_uniform_buffer_offset_alignment.max(16);
            let batch_count = (packed.groups.len() * 2).max(1);
            let batch_bytes = (batch_count as u64)
                .checked_mul(stride as u64)
                .ok_or_else(|| invalid("GPU batch size overflow"))?;
            if batch_bytes > u32::MAX as u64 {
                return Err(invalid("GPU dynamic batch offset overflow"));
            }
            let mut batches = vec![0u8; batch_bytes as usize];
            for (i, &(offset, count)) in packed.groups.iter().enumerate() {
                for reverse in 0..2 {
                    let start = (i * 2 + reverse) * stride as usize;
                    batches[start..start + 16].copy_from_slice(bytemuck::bytes_of(&[
                        offset,
                        count,
                        reverse as u32,
                        0,
                    ]));
                }
            }
            let buffers = vec![
                buffer(
                    device,
                    "strand fibers",
                    bytemuck::cast_slice(&packed.fibers),
                    144,
                    storage,
                )?,
                buffer(
                    device,
                    "strand bonds",
                    bytemuck::cast_slice(&packed.bonds),
                    160,
                    storage,
                )?,
                buffer(
                    device,
                    "strand vertices",
                    bytemuck::cast_slice(&packed.vertices),
                    16,
                    storage,
                )?,
                buffer(
                    device,
                    "strand graph indices",
                    bytemuck::cast_slice(&packed.colors),
                    16,
                    storage,
                )?,
                buffer(device, "strand working", &[], state_bytes, storage)?,
                buffer(device, "strand predicted", &[], state_bytes, storage)?,
                buffer(
                    device,
                    "strand bond working",
                    &[],
                    bond_bytes.max(48),
                    storage,
                )?,
                buffer(
                    device,
                    "strand ground multipliers",
                    &[],
                    packed.vertices.len() as u64 * 4,
                    storage,
                )?,
                buffer(device, "strand faults", &[], 32, storage)?,
                buffer(
                    device,
                    "strand live",
                    bytemuck::cast_slice(&packed.motions),
                    64,
                    storage,
                )?,
                buffer(
                    device,
                    "strand live bonds",
                    bytemuck::cast_slice(&packed.damage),
                    48,
                    storage,
                )?,
                buffer(
                    device,
                    "strand params",
                    &[],
                    std::mem::size_of::<StrandParams>() as u64,
                    uniform,
                )?,
                buffer(device, "strand batches", &batches, 16, uniform)?,
                buffer(
                    device,
                    "strand frozen rows",
                    &[],
                    (packed.bonds.len() as u64 * 6 + 3) * 80,
                    storage,
                )?,
                buffer(
                    device,
                    "strand coupled scratch",
                    &[],
                    packed.fibers.len() as u64 * KRYLOV_BYTES,
                    storage,
                )?,
                buffer(
                    device,
                    "strand contact topology",
                    bytemuck::cast_slice(&packed.topology),
                    16,
                    storage,
                )?,
                buffer(
                    device,
                    "strand contact pairs",
                    &[],
                    (pair_count as u64 * 144).max(144),
                    storage,
                )?,
                buffer(
                    device,
                    "strand contact graph",
                    &[],
                    if pair_count == 0 {
                        16
                    } else {
                        let n = packed.fibers.len() as u64;
                        (3 + 4 * n
                            + n * (2 * n).div_ceil(32)
                            + pair_count as u64
                            + (pair_count as u64).div_ceil(32)
                            + (pair_count as u64).div_ceil(32)
                            + 1)
                            * 4
                    },
                    storage,
                )?,
            ];
            let dispatch = StrandDispatch::new(
                device,
                program,
                std::array::from_fn(|i| &buffers[i]),
                packed.groups,
                [packed.fibers.len() as u32, packed.bonds.len() as u32],
                stride,
                packed.islands[3],
            )?;
            let staging = buffer(
                device,
                "strand readback",
                &[],
                state_bytes + bond_bytes + 16,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )?;
            Ok(Self {
                owner: engine.device().unwrap().clone(),
                template: model,
                dispatch,
                buffers,
                staging,
                state_bytes,
                bond_bytes,
                limits,
                last_coupled: None,
                pair_count,
                uploads: wgpu::util::StagingBelt::new(device.clone(), state_bytes.max(65536)),
                islands: packed.islands,
                island_max: packed.island_max,
                tree: packed.tree,
                published_revision: None,
            })
        })
    }

    fn check_engine(&self, engine: &Engine) -> Result<()> {
        engine.check_device()?;
        if !Arc::ptr_eq(&self.owner, engine.device().unwrap()) {
            return Err(invalid("strand solver belongs to another device"));
        }
        Ok(())
    }

    /// Enqueue one step; shader faults retain the last valid state and are sticky.
    pub fn step(&mut self, engine: &Engine, config: StepConfig) -> Result<()> {
        self.check_engine(engine)?;
        config.validate()?;
        if config.contact_margin.is_some() && self.limits.contact_pairs == 0 {
            return Err(invalid(
                "convex pair contact requires a nonzero contact_pairs capacity",
            ));
        }
        let mut grab_anchor = [0.0; 4];
        let mut grab_target = [0.0, 0.0, 0.0, -1.0];
        let mut grab_velocity = [0.0; 4];
        if let Some(grab) = config.grab {
            if grab.compliance as f32 <= 0.0 || grab.max_force.is_some_and(|f| f as f32 <= 0.0) {
                return Err(invalid("point grab coefficients outside GPU range"));
            }
            let index = self
                .template
                .indices
                .get(&grab.fiber)
                .copied()
                .ok_or_else(|| invalid("unknown grabbed fiber"))?;
            let fiber = &self.template.fibers[index];
            if fiber.pin.is_some() || !fiber.contains_local(grab.local_point) {
                return Err(invalid("invalid grabbed fiber"));
            }
            grab_anchor = v4(grab.local_point);
            grab_anchor[3] = grab.compliance as f32;
            grab_target = v4(grab.target);
            grab_target[3] = index as f32;
            grab_velocity = v4(grab.target_velocity);
            grab_velocity[3] = grab.damping as f32;
        }
        let params = StrandParams {
            step: [
                config.dt as f32,
                config.air_damping as f32,
                config.ground_height.unwrap_or(0.0) as f32,
                config.friction as f32,
            ],
            gravity: v4(config.gravity),
            counts: [
                self.template.fibers.len() as u32,
                self.template.bonds.len() as u32,
                u32::from(config.ground_height.is_some()),
                if config.contact_margin.is_some() {
                    self.pair_count
                } else {
                    0
                },
            ],
            coupled: [
                config.coupled.map_or(0.0, |c| c.linear_iterations as f32),
                config.coupled.map_or(0.0, |c| c.relative_tolerance as f32),
                config.contact_margin.unwrap_or(0.0) as f32,
                config.pair_restitution as f32,
            ],
            numerics: [
                config.coupled.map_or(0.0, |c| c.absolute_tolerance as f32),
                config.contact_exclusion_hops as f32,
                config.grab.and_then(|g| g.max_force).unwrap_or(0.0) as f32,
                self.island_max as f32,
            ],
            grab_anchor,
            grab_target,
            grab_velocity,
            islands: self.islands,
            tree: self.tree,
        };
        finite(&params.step)?;
        finite(&params.gravity)?;
        finite(&params.coupled)?;
        finite(&params.numerics)?;
        finite(&params.grab_anchor)?;
        finite(&params.grab_target)?;
        finite(&params.grab_velocity)?;
        if !(params.step[0] * params.step[0]).is_normal() {
            return Err(invalid("GPU timestep underflow"));
        }
        let queue = engine.queue().unwrap();
        self.published_revision = None;
        molla_core::gpu_write_buffer!(queue, &self.buffers[11], 0, bytemuck::bytes_of(&params));
        let mut encoder = self
            .owner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("strand step"),
            });
        self.dispatch.encode(
            &mut encoder,
            config.iterations,
            config.ground_height.is_some(),
            config.coupled.map(|c| c.nonlinear_iterations),
            params.counts[3],
            config.grab.is_some(),
        );
        queue.submit([encoder.finish()]);
        Ok(())
    }

    pub fn snapshot(&mut self, engine: &Engine) -> Result<StrandModel> {
        let (model, status) = self.last_valid_snapshot(engine)?;
        if status != 0 {
            return Err(invalid(&format!(
                "GPU strand fault bits {status:#x}; previous state retained; coupled={:?}",
                self.last_coupled
            )));
        }
        Ok(model)
    }

    /// Diagnostics from the latest readback, not from queued unsynchronized steps.
    pub fn last_coupled_diagnostics(&self) -> Option<CoupledDiagnostics> {
        self.last_coupled
    }

    /// Read published state and sticky faults: motion=1, bonds=2, PCG=4, contact=8, rotation=16.
    pub fn last_valid_snapshot(&mut self, engine: &Engine) -> Result<(StrandModel, u32)> {
        self.check_engine(engine)?;
        let mut encoder = self
            .owner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("strand snapshot"),
            });
        encoder.copy_buffer_to_buffer(&self.buffers[9], 0, &self.staging, 0, self.state_bytes);
        if self.bond_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &self.buffers[10],
                0,
                &self.staging,
                self.state_bytes,
                self.bond_bytes,
            );
        }
        encoder.copy_buffer_to_buffer(
            &self.buffers[8],
            0,
            &self.staging,
            self.state_bytes + self.bond_bytes,
            16,
        );
        engine.queue().unwrap().submit([encoder.finish()]);
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.owner
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("strand readback poll: {e}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(e.to_string()))?
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let result = {
            let bytes = molla_core::gpu_readback!(slice);
            let offset = (self.state_bytes + self.bond_bytes) as usize;
            let words: &[u32] = bytemuck::cast_slice(&bytes[offset..offset + 16]);
            self.last_coupled = Some(CoupledDiagnostics {
                iterations: words[1],
                relative_residual: f32::from_bits(words[2]),
                initial_preconditioned_norm_squared: f32::from_bits(words[3]),
            });
            self.decode_snapshot(&bytes)
        };
        self.staging.unmap();
        engine.check_device()?;
        self.published_revision = result
            .as_ref()
            .ok()
            .and_then(|(model, status)| (*status == 0).then_some(model.revision()));
        result
    }

    fn decode_snapshot(&self, bytes: &[u8]) -> Result<(StrandModel, u32)> {
        let state_end = self.state_bytes as usize;
        let bond_end = state_end + self.bond_bytes as usize;
        let status = u32::from_le_bytes(bytes[bond_end..bond_end + 4].try_into().unwrap());
        let motions: &[MotionRecord] = bytemuck::cast_slice(&bytes[..state_end]);
        let bonds: &[BondStateRecord] = bytemuck::cast_slice(&bytes[state_end..bond_end]);
        let mut model = self.template.clone();
        for (fiber, motion) in model.fibers.iter_mut().zip(motions) {
            finite(bytemuck::cast_slice(std::slice::from_ref(motion)))?;
            if fiber.pin.is_none() {
                let q = Quat::from_array(motion.rotation.map(f64::from));
                if (q.length_squared() - 1.0).abs() > 1e-3 {
                    return Err(invalid("invalid GPU orientation"));
                }
                fiber.motion.pose.rotation = q.normalize();
                fiber.motion.pose.position =
                    vec3(motion.center) - fiber.motion.pose.rotation * fiber.mass.local_com;
                fiber.motion.linear_velocity = vec3(motion.velocity);
                fiber.motion.angular_velocity = vec3(motion.omega);
            } else {
                let initial = MotionRecord {
                    center: v4(fiber.center()),
                    rotation: fiber.motion.pose.rotation.to_array().map(|v| v as f32),
                    velocity: [0.0; 4],
                    omega: [0.0; 4],
                };
                if bytemuck::bytes_of(motion) != bytemuck::bytes_of(&initial) {
                    return Err(invalid("GPU pin moved"));
                }
            }
            fiber.validate_motion()?;
        }
        for (bond, state) in model.bonds.iter_mut().zip(bonds) {
            finite(bytemuck::cast_slice(std::slice::from_ref(state)))?;
            if !(0.0..=1.0).contains(&state.damage[0]) {
                return Err(invalid("invalid GPU bond damage"));
            }
            bond.damage = state.damage[0] as f64;
        }
        model.advance_revision()?;
        Ok((model, status))
    }

    /// Rebuild changed geometry and GPU resources before publishing either.
    pub fn cut(
        &mut self,
        engine: &Engine,
        plane: CutPlane,
        selected: &[u64],
        cut_id: u64,
    ) -> Result<CutReport> {
        let snapshot = self.snapshot(engine)?;
        let transaction = snapshot.prepare_cut(plane, selected, cut_id)?;
        let report = transaction.report().clone();
        let replacement = Self::with_limits(engine, transaction.candidate().clone(), self.limits)?;
        *self = replacement;
        Ok(report)
    }

    pub fn impulse(&mut self, engine: &Engine, id: u64, point: Vec3, impulse: Vec3) -> Result<()> {
        self.impulses(engine, &[(id, point, impulse)])
    }

    /// Publish nonplanar fragment assemblies after geometry and GPU capacity validation.
    pub fn splinter(
        &mut self,
        engine: &Engine,
        id: u64,
        plane: CutPlane,
        event: u64,
    ) -> Result<super::SplinterReport> {
        let snapshot = self.snapshot(engine)?;
        let transaction = snapshot.prepare_splinter(id, plane, event)?;
        let report = transaction.report().clone();
        let replacement = Self::with_limits(engine, transaction.candidate().clone(), self.limits)?;
        *self = replacement;
        Ok(report)
    }

    /// Publish a prescribed noncoplanar fracture only after replacement resources succeed.
    pub fn fracture(
        &mut self,
        engine: &Engine,
        reference: CutPlane,
        surfaces: &[(u64, CutPlane)],
        event_id: u64,
    ) -> Result<CutReport> {
        let snapshot = self.snapshot(engine)?;
        let transaction = snapshot.prepare_fracture(reference, surfaces, event_id)?;
        let report = transaction.report().clone();
        let replacement = Self::with_limits(engine, transaction.candidate().clone(), self.limits)?;
        *self = replacement;
        Ok(report)
    }

    /// Apply an impulse batch atomically at a step boundary.
    pub fn impulses(&mut self, engine: &Engine, impulses: &[(u64, Vec3, Vec3)]) -> Result<()> {
        self.impulses_snapshot(engine, impulses).map(|_| ())
    }

    /// Apply an atomic impulse batch without rebuilding geometry, returning the updated state.
    pub fn impulses_snapshot(
        &mut self,
        engine: &Engine,
        impulses: &[(u64, Vec3, Vec3)],
    ) -> Result<StrandModel> {
        let snapshot = self.snapshot(engine)?;
        self.apply_impulses_snapshot(engine, snapshot, impulses)
    }

    /// Reuse a matching last readback; stale cache hints synchronize against the live GPU state.
    pub fn impulses_snapshot_cached(
        &mut self,
        engine: &Engine,
        cached: &StrandModel,
        impulses: &[(u64, Vec3, Vec3)],
    ) -> Result<StrandModel> {
        self.check_engine(engine)?;
        let snapshot = if self.published_revision == Some(cached.revision()) {
            cached.clone()
        } else {
            self.snapshot(engine)?
        };
        self.apply_impulses_snapshot(engine, snapshot, impulses)
    }

    fn apply_impulses_snapshot(
        &mut self,
        engine: &Engine,
        mut snapshot: StrandModel,
        impulses: &[(u64, Vec3, Vec3)],
    ) -> Result<StrandModel> {
        for &(id, point, impulse) in impulses {
            snapshot.impulse(id, point, impulse)?;
        }
        let motions: Vec<_> = snapshot
            .fibers()
            .iter()
            .map(|f| MotionRecord {
                center: v4(f.center()),
                rotation: f.motion.pose.rotation.to_array().map(|v| v as f32),
                velocity: v4(f.motion.linear_velocity),
                omega: v4(f.motion.angular_velocity),
            })
            .collect();
        finite(bytemuck::cast_slice(&motions))?;
        let bytes = bytemuck::cast_slice(&motions);
        let mut commands = self.owner.create_command_encoder(&Default::default());
        let mut upload = self.uploads.write_buffer(
            &mut commands,
            &self.buffers[9],
            0,
            wgpu::BufferSize::new(bytes.len() as u64).unwrap(),
        );
        molla_compute::copy_to_mapped(&mut upload, bytes);
        drop(upload);
        molla_core::gpu_metrics::record_upload(bytes.len() as u64);
        self.uploads.finish_and_recall_on_submit(&commands);
        engine.queue().unwrap().submit([commands.finish()]);
        self.last_coupled = None;
        self.published_revision = None;
        Ok(snapshot)
    }
}
