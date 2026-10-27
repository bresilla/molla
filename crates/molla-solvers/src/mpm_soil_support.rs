//! State-bound quadratic transfer-stencil occupancy queries.

use super::*;

#[path = "mpm_soil_retained_support.rs"]
mod retained;
pub use retained::{MpmSoilRetainedSupportParticle, MpmSoilRetainedSupportSnapshot};

#[path = "mpm_soil_support_regions.rs"]
mod region_seeds;

/// Conservative direct transfer dependencies in physical-slot order.
/// All 27 stencil nodes count, including zero-weight endpoints and the particle
/// itself. This is not a transitive component or a sleep/eviction certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmSoilStencilSupport {
    pub quiet: bool,
    pub touches_quiet: bool,
    pub touches_nonquiet: bool,
}

/// Immutable dependencies at the exact-rest snapshot's state boundary.
/// Gravity, future loads, rigid contacts and omitted particles are not certified.
#[derive(Clone, Debug)]
pub struct MpmSoilSupportSnapshot {
    quiescence: MpmSoilQuiescenceSnapshot,
    minimum_quiet_steps: u32,
    particles: Vec<MpmSoilStencilSupport>,
}

impl MpmSoilSupportSnapshot {
    pub fn quiescence(&self) -> &MpmSoilQuiescenceSnapshot {
        &self.quiescence
    }
    pub fn minimum_quiet_steps(&self) -> u32 {
        self.minimum_quiet_steps
    }
    pub fn particles(&self) -> &[MpmSoilStencilSupport] {
        &self.particles
    }
}

/// Membership in a stencil-connected component containing nonquiet soil.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmSoilSupportClosureParticle {
    pub quiet: bool,
    pub connected_to_nonquiet: bool,
}

/// Converged conservative stencil connectivity at one exact-rest boundary.
/// All 27 nodes count. This does not certify force equilibrium, future loads,
/// omitted neighbors, sleep or eviction. Failed convergence returns no snapshot.
#[derive(Clone, Debug)]
pub struct MpmSoilSupportClosureSnapshot {
    quiescence: MpmSoilQuiescenceSnapshot,
    minimum_quiet_steps: u32,
    rounds: u32,
    particles: Vec<MpmSoilSupportClosureParticle>,
}

impl MpmSoilSupportClosureSnapshot {
    pub fn quiescence(&self) -> &MpmSoilQuiescenceSnapshot {
        &self.quiescence
    }
    pub fn minimum_quiet_steps(&self) -> u32 {
        self.minimum_quiet_steps
    }
    pub fn rounds(&self) -> u32 {
        self.rounds
    }
    pub fn particles(&self) -> &[MpmSoilSupportClosureParticle] {
        &self.particles
    }
}

pub(super) struct SupportQuery {
    pending: Option<PendingSupport>,
    failure: Option<String>,
    regions: Option<region_seeds::RegionSeeds>,
    params: wgpu::Buffer,
    nodes: wgpu::Buffer,
    result: wgpu::Buffer,
    staging: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    mark: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    closure_gather: wgpu::ComputePipeline,
    closure_mark: wgpu::ComputePipeline,
    closure_check: wgpu::ComputePipeline,
}

struct PendingSupport {
    receive: std::sync::Mutex<
        std::sync::mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
    >,
    snapshot: Option<MpmSoilQuiescenceSnapshot>,
    request: Option<SupportRequest>,
    minimum_quiet_steps: u32,
    bytes: u64,
    cancelled: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SupportKind {
    Direct,
    Closure,
    Retained,
    Region,
}

enum SupportRequest {
    Direct,
    Closure(u32),
    Retained(u32, Vec<molla_vehicle::SoilParticleId>),
    Region(u32),
}

impl SupportRequest {
    fn kind(&self) -> SupportKind {
        match self {
            Self::Direct => SupportKind::Direct,
            Self::Closure(_) => SupportKind::Closure,
            Self::Retained(..) => SupportKind::Retained,
            Self::Region(_) => SupportKind::Region,
        }
    }
}

fn sizes(capacity: u32, grid_size: u32, limits: &wgpu::Limits, budget: u64) -> Result<[u64; 4]> {
    let nodes = u64::from(grid_size)
        .checked_pow(3)
        .filter(|&n| n <= u64::from(u32::MAX));
    let Some(nodes) = nodes else {
        return Err(Error::Solver(
            "soil support grid exceeds index range".into(),
        ));
    };
    let particles = u64::from(capacity.max(1)) * 4;
    let sizes = [64, nodes * 4, particles, 16 + particles];
    if grid_size < 3
        || sizes.iter().sum::<u64>() > budget
        || sizes.iter().any(|&n| n > limits.max_buffer_size)
        || sizes[1..3]
            .iter()
            .any(|&n| n > u64::from(limits.max_storage_buffer_binding_size))
        || capacity.div_ceil(64) > limits.max_compute_workgroups_per_dimension
    {
        return Err(Error::Solver(
            "soil support exceeds budget/device limits".into(),
        ));
    }
    Ok(sizes)
}

impl SupportQuery {
    fn new(sim: &MpmSimulation, budget: u64) -> Result<Self> {
        let observer = sim
            .motion_guard
            .as_ref()
            .and_then(|g| g.quiescence.as_ref())
            .ok_or_else(|| Error::Solver("prepare swept quiescence before support query".into()))?;
        if sim.cfg.time_step != MpmTimeStep::Explicit || sim.rigid_count != 0 {
            return Err(Error::Solver(
                "soil support requires detached explicit soil".into(),
            ));
        }
        let sizes = sizes(
            sim.cfg.max_particles,
            sim.cfg.grid_size,
            &sim.device.limits(),
            budget,
        )?;
        let device = &sim.device;
        let buffer = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let params = buffer(
            "soil support parameters",
            sizes[0],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let nodes = buffer(
            "soil support nodes",
            sizes[1],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let result = buffer(
            "soil support result",
            sizes[2],
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let staging = buffer(
            "soil support readback",
            sizes[3],
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil support"),
            entries: &[0, 1, 2, 3, 4].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 3,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil support"),
            layout: &layout,
            entries: &[
                buffer_entry(0, &params),
                buffer_entry(1, &sim.p_pos),
                buffer_entry(2, &observer.records),
                buffer_entry(3, &nodes),
                buffer_entry(4, &result),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil support"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil support"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mpm_soil_support.wgsl").into(),
            ),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            pending: None,
            failure: None,
            regions: None,
            params,
            nodes,
            result,
            staging,
            bindings,
            mark: pipeline("support_mark"),
            gather: pipeline("support_gather"),
            closure_gather: pipeline("closure_gather"),
            closure_mark: pipeline("closure_mark"),
            closure_check: pipeline("closure_check"),
        })
    }

    fn bytes(&self) -> u64 {
        self.params.size() + self.nodes.size() + self.result.size() + self.staging.size()
    }
}

fn decode(words: &[u32]) -> Result<Vec<MpmSoilStencilSupport>> {
    words
        .iter()
        .map(|&word| {
            if !matches!(word, 9 | 18 | 25 | 26) {
                return Err(Error::Solver(
                    "soil support has invalid or clipped transfer stencil".into(),
                ));
            }
            Ok(MpmSoilStencilSupport {
                quiet: word & 1 != 0,
                touches_quiet: word & 8 != 0,
                touches_nonquiet: word & 16 != 0,
            })
        })
        .collect()
}

fn decode_closure(words: &[u32]) -> Result<Vec<MpmSoilSupportClosureParticle>> {
    if words.iter().any(|&word| word & 128 != 0) {
        return Err(Error::Solver(
            "soil support closure did not converge within round budget".into(),
        ));
    }
    words
        .iter()
        .map(|&word| {
            if !matches!(word, 1 | 65 | 66) {
                return Err(Error::Solver(
                    "soil support closure has invalid or clipped transfer stencil".into(),
                ));
            }
            Ok(MpmSoilSupportClosureParticle {
                quiet: word & 1 != 0,
                connected_to_nonquiet: word & 64 != 0,
            })
        })
        .collect()
}

impl MpmSimulation {
    pub(in crate::mpm) fn prepare_soil_support(&mut self, budget: u64) -> Result<()> {
        sizes(
            self.cfg.max_particles,
            self.cfg.grid_size,
            &self.device.limits(),
            budget,
        )?;
        if self.soil_support_bytes() != 0 {
            return Ok(());
        }
        let query = SupportQuery::new(self, budget)?;
        self.motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .support = Some(query);
        Ok(())
    }

    pub(in crate::mpm) fn soil_support_bytes(&self) -> u64 {
        self.motion_guard
            .as_ref()
            .and_then(|g| g.quiescence.as_ref())
            .and_then(|o| o.support.as_ref())
            .map_or(0, SupportQuery::bytes)
    }

    pub(in crate::mpm) fn disable_soil_support(&mut self) {
        if let Some(observer) = self
            .motion_guard
            .as_mut()
            .and_then(|g| g.quiescence.as_mut())
        {
            observer.support = None;
        }
    }

    pub(in crate::mpm) fn read_soil_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
    ) -> Result<MpmSoilSupportSnapshot> {
        let words = self.read_soil_support_words(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            None,
            None,
            None,
        )?;
        Ok(MpmSoilSupportSnapshot {
            quiescence: snapshot.clone(),
            minimum_quiet_steps,
            particles: decode(&words)?,
        })
    }

    pub(in crate::mpm) fn read_soil_support_closure(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
    ) -> Result<MpmSoilSupportClosureSnapshot> {
        if !(1..=64).contains(&rounds) {
            return Err(Error::Solver(
                "soil support closure requires 1..=64 rounds".into(),
            ));
        }
        let words = self.read_soil_support_words(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            Some(rounds),
            None,
            None,
        )?;
        Ok(MpmSoilSupportClosureSnapshot {
            quiescence: snapshot.clone(),
            minimum_quiet_steps,
            rounds,
            particles: decode_closure(&words)?,
        })
    }

    fn read_soil_support_words(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: Option<u32>,
        seeds: Option<&[u32]>,
        boxes: Option<&[[f32; 8]]>,
    ) -> Result<Vec<u32>> {
        if !self.request_soil_support_words(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            rounds,
            seeds,
            boxes,
            None,
        )? {
            return Err(Error::Solver("soil support readback is pending".into()));
        }
        self.poll_soil_support_words(engine, true, None)?
            .map(|(_, words)| words)
            .ok_or_else(|| Error::Gpu("soil support wait did not complete".into()))
    }

    pub(in crate::mpm) fn try_request_soil_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
    ) -> Result<bool> {
        self.request_soil_support_words(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            None,
            None,
            None,
            Some(SupportRequest::Direct),
        )
    }

    pub(in crate::mpm) fn poll_soil_support(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilSupportSnapshot>> {
        self.poll_soil_support_words(engine, false, Some(SupportKind::Direct))?
            .map(|(pending, words)| {
                Ok(MpmSoilSupportSnapshot {
                    quiescence: pending.snapshot.expect("asynchronous support provenance"),
                    minimum_quiet_steps: pending.minimum_quiet_steps,
                    particles: decode(&words)?,
                })
            })
            .transpose()
    }

    pub(in crate::mpm) fn try_request_soil_support_closure(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
    ) -> Result<bool> {
        if !(1..=64).contains(&rounds) {
            return Err(Error::Solver(
                "soil support closure requires 1..=64 rounds".into(),
            ));
        }
        self.request_soil_support_words(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            Some(rounds),
            None,
            None,
            Some(SupportRequest::Closure(rounds)),
        )
    }

    pub(in crate::mpm) fn poll_soil_support_closure(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilSupportClosureSnapshot>> {
        self.poll_soil_support_words(engine, false, Some(SupportKind::Closure))?
            .map(|(pending, words)| {
                let Some(SupportRequest::Closure(rounds)) = pending.request else {
                    unreachable!("checked support request kind");
                };
                Ok(MpmSoilSupportClosureSnapshot {
                    quiescence: pending.snapshot.expect("asynchronous support provenance"),
                    minimum_quiet_steps: pending.minimum_quiet_steps,
                    rounds,
                    particles: decode_closure(&words)?,
                })
            })
            .transpose()
    }

    pub(in crate::mpm) fn cancel_soil_support(&mut self) -> bool {
        let Some(query) = self
            .motion_guard
            .as_mut()
            .and_then(|g| g.quiescence.as_mut())
            .and_then(|o| o.support.as_mut())
        else {
            return false;
        };
        let Some(pending) = query.pending.as_mut() else {
            return false;
        };
        if pending.cancelled {
            return false;
        }
        pending.cancelled = true;
        pending.snapshot = None;
        pending.request = None;
        true
    }

    fn soil_support_request_available(
        &self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
    ) -> Result<bool> {
        engine.check_device()?;
        molla_kernels::validate_soil_device(engine.device().unwrap(), &self.device)?;
        self.validate_soil_quiescence(snapshot)?;
        if minimum_quiet_steps == 0
            || self.particle_count > max_particles
            || self.cfg.time_step != MpmTimeStep::Explicit
            || self.rigid_count != 0
            || self
                .grid_window_cells
                .iter()
                .any(|c| c.unsigned_abs() > 1 << 29)
        {
            return Err(Error::Solver("unsupported soil support request".into()));
        }
        let query = self
            .motion_guard
            .as_ref()
            .and_then(|g| g.quiescence.as_ref())
            .and_then(|o| o.support.as_ref())
            .ok_or_else(|| Error::Solver("soil support is not prepared".into()))?;
        if let Some(error) = &query.failure {
            return Err(Error::Gpu(error.clone()));
        }
        Ok(query.pending.is_none())
    }

    fn request_soil_support_words(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: Option<u32>,
        seeds: Option<&[u32]>,
        boxes: Option<&[[f32; 8]]>,
        request: Option<SupportRequest>,
    ) -> Result<bool> {
        if !self.soil_support_request_available(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
        )? {
            return Ok(false);
        }
        let query = self
            .motion_guard
            .as_ref()
            .unwrap()
            .quiescence
            .as_ref()
            .unwrap()
            .support
            .as_ref()
            .unwrap();
        let mut params = [0u32; 16];
        for axis in 0..3 {
            params[axis] = self.cfg.grid_origin[axis].to_bits();
            params[4 + axis] = self.grid_window_cells[axis] as u32;
        }
        params[8] = self.particle_count;
        params[9] = self.cfg.grid_size;
        params[10] = minimum_quiet_steps;
        params[11] = u32::from(seeds.is_some());
        params[12] = (1.0 / self.cfg.cell_size).to_bits();
        let queue = engine.queue().unwrap();
        let regions = if let Some(boxes) = boxes {
            let regions = query.regions.as_ref().ok_or_else(|| {
                Error::Solver("soil support region seeds are not prepared".into())
            })?;
            if seeds.is_some()
                || rounds.is_none()
                || boxes.len() > regions.capacity as usize
                || boxes
                    .iter()
                    .any(|b| b.iter().any(|v| !v.is_finite()) || (0..3).any(|a| b[a] > b[4 + a]))
            {
                return Err(Error::Solver("invalid soil support region boxes".into()));
            }
            params[11] = boxes.len() as u32;
            if !boxes.is_empty() {
                molla_core::gpu_metrics::record_upload(boxes.len() as u64 * 32);
                queue.write_buffer(&regions.boxes, 0, bytemuck::cast_slice(boxes));
            }
            Some(regions)
        } else {
            None
        };
        if let Some(seeds) = seeds {
            if seeds.len() != self.particle_count as usize || rounds.is_none() {
                return Err(Error::Solver("invalid soil support seed mask".into()));
            }
            if !seeds.is_empty() {
                molla_core::gpu_metrics::record_upload(seeds.len() as u64 * 4);
                queue.write_buffer(&query.result, 0, bytemuck::cast_slice(seeds));
            }
        }
        molla_core::gpu_metrics::record_upload(64);
        queue.write_buffer(&query.params, 0, bytemuck::cast_slice(&params));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&query.nodes, 0, None);
        if self.particle_count != 0 {
            if let Some(regions) = regions.filter(|_| params[11] != 0) {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&regions.pipeline);
                pass.set_bind_group(0, &regions.bindings, &[]);
                molla_core::gpu_metrics::record_dispatch();
                pass.dispatch_workgroups(self.particle_count.div_ceil(64), 1, 1);
            }
            let mut dispatch = |pipeline: &wgpu::ComputePipeline| {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &query.bindings, &[]);
                molla_core::gpu_metrics::record_dispatch();
                pass.dispatch_workgroups(self.particle_count.div_ceil(64), 1, 1);
            };
            dispatch(&query.mark);
            if let Some(rounds) = rounds {
                for _ in 0..rounds {
                    dispatch(&query.closure_gather);
                    dispatch(&query.closure_mark);
                }
                dispatch(&query.closure_check);
            } else {
                dispatch(&query.gather);
            }
            encoder.copy_buffer_to_buffer(
                &query.result,
                0,
                &query.staging,
                16,
                u64::from(self.particle_count) * 4,
            );
        }
        encoder.copy_buffer_to_buffer(
            &self.active_tiles.list,
            self.active_tiles.metadata_offset,
            &query.staging,
            0,
            16,
        );
        queue.submit([encoder.finish()]);
        let bytes = 16 + u64::from(self.particle_count) * 4;
        molla_core::gpu_metrics::record_readback(bytes);
        let slice = query.staging.slice(..bytes);
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = send.send(r);
        });
        self.motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .support
            .as_mut()
            .unwrap()
            .pending = Some(PendingSupport {
            receive: std::sync::Mutex::new(receive),
            snapshot: request.as_ref().map(|_| snapshot.clone()),
            request,
            minimum_quiet_steps,
            bytes,
            cancelled: false,
        });
        Ok(true)
    }

    fn poll_soil_support_words(
        &mut self,
        engine: &Engine,
        wait: bool,
        expected: Option<SupportKind>,
    ) -> Result<Option<(PendingSupport, Vec<u32>)>> {
        engine.check_device()?;
        molla_kernels::validate_soil_device(engine.device().unwrap(), &self.device)?;
        let query = self
            .motion_guard
            .as_mut()
            .and_then(|g| g.quiescence.as_mut())
            .and_then(|o| o.support.as_mut())
            .ok_or_else(|| Error::Solver("soil support is not prepared".into()))?;
        if let Some(error) = &query.failure {
            return Err(Error::Gpu(error.clone()));
        }
        let Some(pending) = query.pending.as_mut() else {
            return Ok(None);
        };
        if !pending.cancelled
            && expected.is_some_and(|kind| {
                pending.request.as_ref().map(SupportRequest::kind) != Some(kind)
            })
        {
            return Err(Error::Solver(
                "different soil support query is pending".into(),
            ));
        }
        let poll = self.device.poll(if wait {
            wgpu::PollType::wait_indefinitely()
        } else {
            wgpu::PollType::Poll
        });
        let mapped = match poll {
            Err(error) => Err(Error::Gpu(error.to_string())),
            Ok(_) => match pending.receive.get_mut().unwrap().try_recv() {
                Ok(result) => result.map_err(|e| Error::Gpu(e.to_string())),
                Err(std::sync::mpsc::TryRecvError::Empty) if !wait => return Ok(None),
                Err(error) => Err(Error::Gpu(error.to_string())),
            },
        };
        let pending = query.pending.take().unwrap();
        if let Err(error) = mapped {
            query.staging.unmap();
            query.failure = Some(error.to_string());
            return Err(error);
        }
        if pending.cancelled {
            query.staging.unmap();
            return Ok(None);
        }
        let mut words: Vec<u32> =
            bytemuck::cast_slice(&query.staging.slice(..pending.bytes).get_mapped_range()).to_vec();
        query.staging.unmap();
        if words[2] != 0 {
            return Err(Error::Solver("faulted soil support snapshot".into()));
        }
        words.drain(..4);
        Ok(Some((pending, words)))
    }
}

#[cfg(test)]
#[path = "mpm_soil_support_tests.rs"]
mod tests;

#[cfg(all(test, feature = "resource-counters"))]
#[path = "mpm_soil_support_async_tests.rs"]
mod async_tests;

#[cfg(all(test, feature = "resource-counters"))]
#[path = "mpm_soil_support_frontier_tests.rs"]
mod frontier_tests;
