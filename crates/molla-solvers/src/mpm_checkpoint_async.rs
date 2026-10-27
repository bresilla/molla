//! Bounded FIFO asynchronous soil archive capture.

use super::*;
use molla_compute::DeviceHealth;
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct MpmSoilCheckpointTicket {
    owner: Arc<()>,
    generation: Arc<()>,
    source_epoch: Arc<()>,
    revision: u64,
}

impl MpmSoilCheckpointTicket {
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

/// One requested historical state, including a per-snapshot numerical/fault error.
#[derive(Debug)]
pub struct MpmSoilCheckpointCompletion {
    ticket: MpmSoilCheckpointTicket,
    result: Result<MpmSoilCheckpoint>,
    identity: Option<MpmSoilIdentitySnapshot>,
}

impl MpmSoilCheckpointCompletion {
    pub fn ticket(&self) -> &MpmSoilCheckpointTicket {
        &self.ticket
    }
    pub fn into_result(self) -> Result<MpmSoilCheckpoint> {
        self.result
    }

    pub fn identity(&self) -> Option<&MpmSoilIdentitySnapshot> {
        self.identity.as_ref()
    }

    /// Build a globally identified tile archive from the request-time mapping.
    /// Captures made before an identified restore return an error rather than
    /// inventing IDs. The particle-only `into_result` discards this mapping.
    pub fn into_tile_archive(
        self,
        layout: molla_vehicle::SoilTileLayout,
    ) -> Result<MpmSoilTileArchive> {
        let checkpoint = self.result?;
        let identity = self
            .identity
            .ok_or_else(|| invalid("capture has no global soil identity mapping"))?;
        checkpoint.partition_tiles(layout, identity.particle_ids(), identity.material_ids())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpmSoilReadbackStats {
    pub requests: u64,
    pub backpressure: u64,
    pub completed: u64,
    pub rejected: u64,
    pub cancelled: u64,
    pub aborted: u64,
    pub readback_bytes: u64,
}

struct Request {
    ticket: MpmSoilCheckpointTicket,
    metadata: CheckpointMetadata,
    completion: Arc<OnceLock<std::result::Result<(), wgpu::BufferAsyncError>>>,
}

struct Slot {
    buffer: wgpu::Buffer,
    request: Option<Request>,
}

/// Fixed-capacity archive readback, not optional/coalesced telemetry. All accepted
/// captures are returned FIFO unless explicitly cancelled or a sticky device/map
/// failure aborts the ring. Capture/poll never wait for GPU completion. Consuming a
/// ready snapshot still allocates/copies/validates host particle data.
pub struct MpmSoilCheckpointReadback {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    health: Arc<DeviceHealth>,
    source: Arc<()>,
    owner: Arc<()>,
    generation: Arc<()>,
    slots: Vec<Slot>,
    capacity: u32,
    staging_bytes: u64,
    last_revision: Option<u64>,
    stats: MpmSoilReadbackStats,
    failure: Option<String>,
}

impl MpmSimulation {
    /// Prepare retained MAP_READ slots for this simulation's lifetime. Engine
    /// device must match. The logical staging budget excludes driver overhead and
    /// returned host snapshots. Capacity is fixed; no growth occurs during capture.
    pub fn prepare_soil_checkpoint_readback(
        &self,
        engine: &Engine,
        particle_capacity: u32,
        slots: usize,
        max_staging_bytes: u64,
    ) -> Result<MpmSoilCheckpointReadback> {
        engine.check_device()?;
        let device = engine.device().expect("checked GPU engine");
        let queue = engine.queue().expect("checked GPU engine");
        molla_kernels::validate_soil_device(device, &self.device)?;
        let bytes = u64::from(particle_capacity) * WORDS_PER_PARTICLE as u64 * 4 + 16;
        let total = bytes
            .checked_mul(slots as u64)
            .ok_or_else(|| invalid("readback staging size overflow"))?;
        if slots == 0
            || slots > 64
            || particle_capacity > self.cfg.max_particles
            || bytes > device.limits().max_buffer_size
            || total > max_staging_bytes
        {
            return Err(invalid(
                "invalid readback capacity, slot count or staging budget",
            ));
        }
        let slots = (0..slots)
            .map(|_| Slot {
                buffer: molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("MPM retained soil checkpoint slot"),
                        size: bytes,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }
                ),
                request: None,
            })
            .collect();
        Ok(MpmSoilCheckpointReadback {
            device: device.clone(),
            queue: queue.clone(),
            health: engine.device_health().unwrap().clone(),
            source: self.soil_owner.clone(),
            owner: Arc::new(()),
            generation: Arc::new(()),
            slots,
            capacity: particle_capacity,
            staging_bytes: total,
            last_revision: None,
            stats: Default::default(),
            failure: None,
        })
    }
}

impl MpmSoilCheckpointReadback {
    pub fn particle_capacity(&self) -> u32 {
        self.capacity
    }
    pub fn slots(&self) -> usize {
        self.slots.len()
    }
    pub fn staging_bytes(&self) -> u64 {
        self.staging_bytes
    }
    pub fn pending(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.request.is_some())
            .count()
    }
    pub fn stats(&self) -> MpmSoilReadbackStats {
        self.stats
    }

    /// Generation identity only; true does not imply newest physical state.
    pub fn is_current(&self, ticket: &MpmSoilCheckpointTicket) -> bool {
        self.failure.is_none()
            && self.health.check().is_ok()
            && Arc::ptr_eq(&self.owner, &ticket.owner)
            && Arc::ptr_eq(&self.generation, &ticket.generation)
    }

    /// Also checks source/configuration identity. Physics steps do not invalidate
    /// compatibility; callers must use revision/time to reason about freshness.
    pub fn is_compatible(
        &self,
        simulation: &MpmSimulation,
        ticket: &MpmSoilCheckpointTicket,
    ) -> bool {
        self.is_current(ticket)
            && Arc::ptr_eq(&self.source, &simulation.soil_owner)
            && Arc::ptr_eq(&ticket.source_epoch, &simulation.step_epoch)
    }

    fn fail(&mut self, message: String) -> Error {
        for slot in &mut self.slots {
            if slot.request.take().is_some() {
                slot.buffer.unmap();
                self.stats.aborted = self.stats.aborted.saturating_add(1);
            }
        }
        self.failure = Some(message.clone());
        Error::Gpu(message)
    }

    fn check(&mut self) -> Result<()> {
        if let Some(message) = &self.failure {
            return Err(Error::Gpu(message.clone()));
        }
        if let Err(error) = self.health.check() {
            return Err(self.fail(error.to_string()));
        }
        Ok(())
    }

    /// Explicitly discard all unpublished captures without waiting or reclaiming
    /// in-flight slots. Polling reclaims cancelled slots after map completion.
    /// New revisions may restart at zero in the new generation.
    pub fn cancel_all(&mut self) -> Result<()> {
        self.check()?;
        self.generation = Arc::new(());
        self.last_revision = None;
        Ok(())
    }

    /// Submit copies after prior queue work; None means backpressure with no GPU
    /// commands, allocations or consumed revision. Material/config/count metadata
    /// is captured now, including pending material edits. Revisions must increase.
    /// A retry captures the state at retry time, not the refused boundary.
    pub fn try_capture(
        &mut self,
        simulation: &MpmSimulation,
        revision: u64,
    ) -> Result<Option<MpmSoilCheckpointTicket>> {
        self.check()?;
        if !Arc::ptr_eq(&self.source, &simulation.soil_owner) {
            return Err(invalid("readback source belongs to another simulation"));
        }
        if simulation.particle_count > self.capacity {
            return Err(invalid("particle count exceeds prepared readback capacity"));
        }
        if self.last_revision.is_some_and(|last| revision <= last) {
            return Err(invalid(
                "checkpoint revisions must increase within a generation",
            ));
        }
        let Some(index) = self.slots.iter().position(|slot| slot.request.is_none()) else {
            self.stats.backpressure = self.stats.backpressure.saturating_add(1);
            return Ok(None);
        };
        let metadata = simulation.checkpoint_metadata()?;
        let ticket = MpmSoilCheckpointTicket {
            owner: self.owner.clone(),
            generation: self.generation.clone(),
            source_epoch: simulation.step_epoch.clone(),
            revision,
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MPM asynchronous soil checkpoint"),
            });
        simulation.encode_checkpoint_copy(&mut encoder, &self.slots[index].buffer);
        self.queue.submit([encoder.finish()]);
        let completion = Arc::new(OnceLock::new());
        let signal = completion.clone();
        self.slots[index]
            .buffer
            .slice(0..metadata.bytes())
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = signal.set(result);
            });
        self.slots[index].request = Some(Request {
            ticket: ticket.clone(),
            metadata,
            completion,
        });
        self.last_revision = Some(revision);
        self.stats.requests = self.stats.requests.saturating_add(1);
        Ok(Some(ticket))
    }

    /// Nonblocking FIFO poll. Invalid physical snapshots are returned individually;
    /// device/map failures poison the ring and abort all outstanding requests.
    /// Never coalesces, substitutes or silently drops an accepted live capture.
    pub fn poll_next(&mut self) -> Result<Option<MpmSoilCheckpointCompletion>> {
        self.check()?;
        if self.pending() == 0 {
            return Ok(None);
        }
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Err(self.fail(format!("MPM checkpoint device poll: {error}")));
        }
        self.check()?;
        for slot in &mut self.slots {
            if slot.request.as_ref().is_some_and(|request| {
                !Arc::ptr_eq(&request.ticket.generation, &self.generation)
                    && request.completion.get().is_some()
            }) {
                slot.buffer.unmap();
                slot.request = None;
                self.stats.cancelled = self.stats.cancelled.saturating_add(1);
            }
        }
        let failure = self
            .slots
            .iter()
            .filter_map(|slot| slot.request.as_ref())
            .find_map(|request| match request.completion.get() {
                Some(Err(error)) => Some(format!("MPM checkpoint map: {error}")),
                _ => None,
            });
        if let Some(message) = failure {
            return Err(self.fail(message));
        }
        let index = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                slot.request
                    .as_ref()
                    .filter(|request| Arc::ptr_eq(&request.ticket.generation, &self.generation))
                    .map(|request| (index, request.ticket.revision))
            })
            .min_by_key(|&(_, revision)| revision)
            .map(|(index, _)| index);
        let Some(index) = index else {
            return Ok(None);
        };
        let slot = &mut self.slots[index];
        if slot.request.as_ref().unwrap().completion.get().is_none() {
            return Ok(None);
        }
        let request = slot.request.take().unwrap();
        let bytes = request.metadata.bytes();
        let identity = request.metadata.identity.clone();
        let slice = slot.buffer.slice(0..bytes);
        let data = molla_core::gpu_readback!(slice);
        let result = request.metadata.finish(bytemuck::cast_slice(&data));
        drop(data);
        slot.buffer.unmap();
        self.stats.completed = self.stats.completed.saturating_add(1);
        self.stats.rejected = self
            .stats
            .rejected
            .saturating_add(u64::from(result.is_err()));
        self.stats.readback_bytes = self.stats.readback_bytes.saturating_add(bytes);
        Ok(Some(MpmSoilCheckpointCompletion {
            ticket: request.ticket,
            result,
            identity,
        }))
    }
}

impl Drop for MpmSoilCheckpointReadback {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if slot.request.is_some() {
                slot.buffer.unmap();
            }
        }
    }
}

#[cfg(test)]
#[path = "mpm_checkpoint_async_tests.rs"]
mod tests;
