//! Bounded asynchronous observations of GPU MPM compaction.

use crate::TireMpmCompactionSummary;
use crate::soil_gpu_identity::device_identity;
use crate::soil_mpm_summary::CompactionSummaryPass;
use molla_compute::{DeviceHealth, Engine};
use molla_core::{Error, Result};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Debug)]
pub struct TireMpmCompactionObservation {
    owner: Arc<()>,
    generation: u64,
    revision: u64,
    summary: TireMpmCompactionSummary,
}

impl TireMpmCompactionObservation {
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn summary(&self) -> TireMpmCompactionSummary {
        self.summary
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TireMpmCompactionReadbackStats {
    pub requests: u64,
    pub backpressure: u64,
    pub completed: u64,
    pub discarded: u64,
    pub readback_bytes: u64,
}

enum SlotState {
    Free,
    Pending {
        generation: u64,
        revision: u64,
        completion: Arc<OnceLock<std::result::Result<(), wgpu::BufferAsyncError>>>,
    },
    Ready {
        generation: u64,
        revision: u64,
    },
}

struct Slot {
    buffer: wgpu::Buffer,
    state: SlotState,
}

/// Fixed-source optional telemetry. Source must belong to source_device; physical consumers must not use delayed samples.
pub struct PreparedTireMpmCompactionReadback {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    health: Arc<DeviceHealth>,
    owner: Arc<()>,
    summary: CompactionSummaryPass,
    slots: Vec<Slot>,
    count: u32,
    generation: u64,
    last_requested: Option<u64>,
    last_published: Option<u64>,
    stats: TireMpmCompactionReadbackStats,
    failure: Option<String>,
}

impl PreparedTireMpmCompactionReadback {
    pub fn new(
        engine: &Engine,
        source_device: &wgpu::Device,
        source: &wgpu::Buffer,
        count: u32,
        threshold: f32,
        slots: usize,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().expect("checked GPU engine");
        let queue = engine.queue().expect("checked GPU engine");
        if slots == 0 || slots > 64 {
            return Err(Error::Solver(
                "MPM observation ring requires 1 through 64 slots".into(),
            ));
        }
        let identity = device_identity(device).ok_or_else(|| {
            Error::Gpu("MPM observations require a native device identity".into())
        })?;
        if device_identity(source_device) != Some(identity) {
            return Err(Error::Gpu(
                "MPM observation source belongs to another device".into(),
            ));
        }
        CompactionSummaryPass::validate(device, source, count, threshold)?;
        let summary = CompactionSummaryPass::new(device, queue, source, count, threshold)?;
        let slots = (0..slots)
            .map(|_| Slot {
                buffer: molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("MPM optional summary ring slot"),
                        size: 16,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }
                ),
                state: SlotState::Free,
            })
            .collect();
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            health: engine.device_health().unwrap().clone(),
            owner: Arc::new(()),
            summary,
            slots,
            count,
            generation: 0,
            last_requested: None,
            last_published: None,
            stats: Default::default(),
            failure: None,
        })
    }

    pub fn particle_count(&self) -> u32 {
        self.count
    }
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
    pub fn pending(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !matches!(s.state, SlotState::Free))
            .count()
    }
    pub fn stats(&self) -> TireMpmCompactionReadbackStats {
        self.stats
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn check(&self) -> Result<()> {
        self.health.check()?;
        match &self.failure {
            Some(message) => Err(Error::Gpu(message.clone())),
            None => Ok(()),
        }
    }

    fn fail(&mut self, message: String) -> Error {
        for slot in &mut self.slots {
            if !matches!(slot.state, SlotState::Free) {
                slot.buffer.unmap();
            }
            slot.state = SlotState::Free;
        }
        self.failure = Some(message.clone());
        Error::Gpu(message)
    }

    /// Invalidate observations without reclaiming in-flight slots or waiting for the GPU.
    pub fn reset(&mut self) -> Result<()> {
        self.check()?;
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::Solver("MPM observation generation exhausted".into()))?;
        self.last_requested = None;
        self.last_published = None;
        Ok(())
    }

    /// Check live owner/generation identity; a valid sample may have an older revision.
    pub fn is_current(&self, sample: &TireMpmCompactionObservation) -> bool {
        self.check().is_ok()
            && Arc::ptr_eq(&self.owner, &sample.owner)
            && self.generation == sample.generation
    }

    /// Submit a reduction/copy after preceding queue work. False skips telemetry when the ring is full.
    pub fn try_request(&mut self, revision: u64) -> Result<bool> {
        self.check()?;
        if self.last_requested.is_some_and(|last| revision <= last) {
            return Err(Error::Solver(
                "MPM observation revisions must increase within a generation".into(),
            ));
        }
        let Some(index) = self
            .slots
            .iter()
            .position(|s| matches!(s.state, SlotState::Free))
        else {
            self.stats.backpressure = self.stats.backpressure.saturating_add(1);
            return Ok(false);
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MPM optional compaction observation"),
            });
        self.summary.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&self.summary.output, 0, &self.slots[index].buffer, 0, 16);
        self.queue.submit([encoder.finish()]);
        let completion = Arc::new(OnceLock::new());
        let signal = completion.clone();
        self.slots[index]
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = signal.set(result);
            });
        self.slots[index].state = SlotState::Pending {
            generation: self.generation,
            revision,
            completion,
        };
        self.last_requested = Some(revision);
        self.stats.requests = self.stats.requests.saturating_add(1);
        Ok(true)
    }

    /// Poll without waiting and return only the newest ready observation of the current generation.
    pub fn poll_latest(&mut self) -> Result<Option<TireMpmCompactionObservation>> {
        self.check()?;
        if self.pending() == 0 {
            return Ok(None);
        }
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Err(self.fail(format!("MPM observation device poll: {error}")));
        }
        self.check()?;
        let mut failure = None;
        for slot in &mut self.slots {
            if let SlotState::Pending {
                generation,
                revision,
                completion,
            } = &slot.state
            {
                match completion.get() {
                    Some(Ok(())) => {
                        slot.state = SlotState::Ready {
                            generation: *generation,
                            revision: *revision,
                        };
                        self.stats.completed = self.stats.completed.saturating_add(1);
                    }
                    Some(Err(error)) => {
                        failure = Some(format!("MPM observation map: {error}"));
                        break;
                    }
                    None => {}
                }
            }
        }
        if let Some(message) = failure {
            return Err(self.fail(message));
        }
        let selected = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match slot.state {
                SlotState::Ready {
                    generation,
                    revision,
                } if generation == self.generation
                    && self.last_published.is_none_or(|last| revision > last) =>
                {
                    Some((index, revision))
                }
                _ => None,
            })
            .max_by_key(|&(_, revision)| revision);
        let mut observation = None;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if let SlotState::Ready {
                generation,
                revision,
            } = slot.state
            {
                if selected.is_some_and(|(selected, _)| selected == index) {
                    let slice = slot.buffer.slice(..);
                    let data = molla_core::gpu_readback!(slice);
                    let words: &[u32] = bytemuck::cast_slice(&data);
                    self.stats.readback_bytes = self.stats.readback_bytes.saturating_add(16);
                    if !f32::from_bits(words[0]).is_finite() || words[1] > self.count {
                        drop(data);
                        slot.buffer.unmap();
                        slot.state = SlotState::Free;
                        failure = Some("MPM observation summary is invalid".into());
                        break;
                    }
                    observation = Some(TireMpmCompactionObservation {
                        owner: self.owner.clone(),
                        generation,
                        revision,
                        summary: TireMpmCompactionSummary {
                            maximum: f32::from_bits(words[0]),
                            active_particles: words[1],
                        },
                    });
                    self.last_published = Some(revision);
                    drop(data);
                } else {
                    self.stats.discarded = self.stats.discarded.saturating_add(1);
                }
                slot.buffer.unmap();
                slot.state = SlotState::Free;
            }
        }
        if let Some(message) = failure {
            return Err(self.fail(message));
        }
        Ok(observation)
    }
}
