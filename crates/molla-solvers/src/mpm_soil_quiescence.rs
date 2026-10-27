//! Per-substep exact-rest observations on the guarded particle publication path.

use super::*;

#[path = "mpm_soil_quiescence_async.rs"]
mod asynchronous;

#[path = "mpm_soil_projection_cache.rs"]
mod projection_cache;

#[path = "mpm_soil_support.rs"]
mod support;
pub use support::{MpmSoilRetainedSupportParticle, MpmSoilRetainedSupportSnapshot};
pub use support::{MpmSoilStencilSupport, MpmSoilSupportSnapshot};
pub use support::{MpmSoilSupportClosureParticle, MpmSoilSupportClosureSnapshot};

/// Exact-rest streak in physical-slot order at one completed GPU read boundary.
/// A dt change restarts the streak; the count saturates at u32::MAX.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpmSoilQuiescence {
    pub consecutive_steps: u32,
    pub dt: f32,
}

/// Historical exact-rest observations, not a sleep/support or eviction certificate.
/// Zero velocity/affine motion and unchanged position/F/plastic history are checked
/// at every guarded substep. External loads, omitted neighbors and future motion
/// need separate admission. Restore/configuration/identity changes reset streaks.
/// Payloads are immutable. Owner validation rejects foreign or stale snapshots;
/// any encoded nonempty step, including cancelled work, invalidates old snapshots.
#[derive(Clone, Debug)]
pub struct MpmSoilQuiescenceSnapshot {
    identity: Option<MpmSoilIdentitySnapshot>,
    particles: Vec<MpmSoilQuiescence>,
    epoch: Arc<()>,
    attempt: u64,
}

impl MpmSoilQuiescenceSnapshot {
    pub fn identity(&self) -> Option<&MpmSoilIdentitySnapshot> {
        self.identity.as_ref()
    }

    pub fn particles(&self) -> &[MpmSoilQuiescence] {
        &self.particles
    }
}

pub(super) struct Observer {
    pending: Option<asynchronous::PendingRead>,
    read_failure: Option<String>,
    support: Option<support::SupportQuery>,
    pub(super) projection_cache: Option<projection_cache::ProjectionCache>,
    records: wgpu::Buffer,
    staging: wgpu::Buffer,
    pub(super) bindings: wgpu::BindGroup,
    pub(super) pipeline: wgpu::ComputePipeline,
    needs_reset: bool,
    attempt: Option<u64>,
}

fn source() -> String {
    let original = super::source();
    let marker = "    let proposal = soil_motion_proposals[pid];\n";
    assert_eq!(original.matches(marker).count(), 1);
    let mut source = original.replace(
        marker,
        &format!("{marker}    soil_quiet_observe(pid, proposal);\n"),
    );
    source.push_str(include_str!("../shaders/mpm_soil_quiescence.wgsl"));
    source
}

fn sizes(capacity: u32, limits: &wgpu::Limits, budget: u64) -> Result<[u64; 2]> {
    let records = u64::from(capacity.max(1)) * 8;
    let sizes = [records, 16 + records];
    if sizes.iter().sum::<u64>() > budget
        || sizes.iter().any(|&n| n > limits.max_buffer_size)
        || records > u64::from(limits.max_storage_buffer_binding_size)
    {
        return Err(Error::Solver(
            "soil quiescence exceeds budget/device limits".into(),
        ));
    }
    Ok(sizes)
}

impl Observer {
    fn new(sim: &MpmSimulation, max_bytes: u64) -> Result<Self> {
        let guard = sim
            .motion_guard
            .as_ref()
            .ok_or_else(|| Error::Solver("prepare swept motion guard before quiescence".into()))?;
        let sizes = sizes(sim.cfg.max_particles, &sim.device.limits(), max_bytes)?;
        let device = &sim.device;
        let buffer = |name, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(name),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let records = buffer(
            "soil quiescence records",
            sizes[0],
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let staging = buffer(
            "soil quiescence readback",
            sizes[1],
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let entries = [0, 1, 2, 3].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding < 2 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil quiescence publication"),
            entries: &entries,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil quiescence publication"),
            layout: &uniforms,
            entries: &[
                buffer_entry(0, &sim.uniform_buf),
                buffer_entry(1, &guard.params),
                buffer_entry(2, &guard.proposals),
                buffer_entry(3, &records),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil quiescence publication"),
            bind_group_layouts: &[
                Some(&uniforms),
                Some(&sim.layout_particles),
                Some(&sim.layout_grid),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil quiescence publication"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil quiescence publication"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("soil_motion_commit"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SPARSE_GRID", f64::from(sim.sparse_grid))],
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self {
            pending: None,
            read_failure: None,
            support: None,
            projection_cache: None,
            records,
            staging,
            bindings,
            pipeline,
            needs_reset: true,
            attempt: Some(0),
        })
    }
}

impl MpmSimulation {
    pub(in crate::mpm) fn prepare_soil_quiescence(&mut self, max_bytes: u64) -> Result<()> {
        if self.rigid_count != 0 {
            return Err(Error::Solver("quiescence requires detached soil".into()));
        }
        if self.soil_quiescence_bytes() > max_bytes {
            return Err(Error::Solver(
                "retained quiescence exceeds byte budget".into(),
            ));
        }
        if self
            .motion_guard
            .as_ref()
            .is_some_and(|g| g.quiescence.is_some())
        {
            return Ok(());
        }
        let observer = Observer::new(self, max_bytes)?;
        self.motion_guard.as_mut().unwrap().quiescence = Some(observer);
        self.invalidate_prepared_steps();
        Ok(())
    }

    pub(in crate::mpm) fn soil_quiescence_bytes(&self) -> u64 {
        self.motion_guard
            .as_ref()
            .and_then(|g| g.quiescence.as_ref())
            .map_or(0, |o| o.records.size() + o.staging.size())
    }

    pub(in crate::mpm) fn quiescence_needs_reset(&self) -> bool {
        self.motion_guard
            .as_ref()
            .and_then(|g| g.quiescence.as_ref())
            .is_some_and(|o| o.needs_reset)
    }

    pub(in crate::mpm) fn invalidate_quiescence(&mut self) {
        if let Some(observer) = self
            .motion_guard
            .as_mut()
            .and_then(|g| g.quiescence.as_mut())
        {
            observer.needs_reset = true;
        }
    }

    pub(in crate::mpm) fn encode_quiescence_reset(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(observer) = self
            .motion_guard
            .as_mut()
            .and_then(|g| g.quiescence.as_mut())
        {
            observer.attempt = observer.attempt.and_then(|n| n.checked_add(1));
            if observer.needs_reset {
                encoder.clear_buffer(&observer.records, 0, None);
                if let Some(cache) = &observer.projection_cache {
                    encoder.clear_buffer(&cache.records, 0, None);
                }
                observer.needs_reset = false;
            }
        }
    }

    pub(in crate::mpm) fn disable_soil_quiescence(&mut self) {
        if let Some(guard) = &mut self.motion_guard {
            guard.quiescence = None;
        }
        self.invalidate_prepared_steps();
    }

    pub(in crate::mpm) fn validate_soil_quiescence(
        &self,
        snapshot: &MpmSoilQuiescenceSnapshot,
    ) -> Result<()> {
        let current = self.motion_guard.as_ref().is_some_and(|guard| {
            guard.bounds.is_some()
                && guard.quiescence.as_ref().is_some_and(|observer| {
                    !observer.needs_reset && observer.attempt == Some(snapshot.attempt)
                })
        });
        let identity_current = match &snapshot.identity {
            Some(identity) => self.is_soil_identity_current(identity),
            None => self.soil_identity_snapshot().is_none(),
        };
        if !current
            || !identity_current
            || self.active_grid_needs_reset
            || !Arc::ptr_eq(&self.step_epoch, &snapshot.epoch)
            || snapshot.particles.len() != self.particle_count as usize
        {
            return Err(Error::Solver(
                "stale or foreign soil quiescence snapshot".into(),
            ));
        }
        Ok(())
    }

    pub(in crate::mpm) fn read_soil_quiescence(
        &mut self,
        engine: &Engine,
        max_particles: u32,
    ) -> Result<MpmSoilQuiescenceSnapshot> {
        if !self.try_request_soil_quiescence(engine, max_particles)? {
            return Err(Error::Solver("soil quiescence readback is pending".into()));
        }
        self.poll_soil_quiescence_inner(engine, true)?
            .ok_or_else(|| Error::Gpu("soil quiescence wait did not complete".into()))
    }
}

#[cfg(test)]
#[path = "mpm_soil_quiescence_tests.rs"]
mod tests;
