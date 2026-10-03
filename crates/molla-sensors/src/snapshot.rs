//! Bounded device result snapshots and completion-tracked consumer batches.

use crate::readback::{BufferLease, Channel, ReadbackSource, ReadbackStamp};
use molla_compute::{DeviceHealth, Engine};
use molla_core::{Error, Result};
use molla_storage::BufferElement;
use std::{any::TypeId, marker::PhantomData, num::NonZeroU64, ops::Range, sync::Arc};

#[derive(Debug, Clone, Copy)]
pub struct DeviceSnapshotConfig {
    pub slots: usize,
    pub bytes_per_slot: u64,
    pub max_total_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceSnapshotStats {
    pub buffer_allocations: u64,
    pub captures: u64,
    pub submissions: u64,
    pub copied_bytes: u64,
    pub backpressure: u64,
}

struct SnapshotSlot {
    device: Arc<wgpu::Device>,
    buffer: wgpu::Buffer,
    in_flight: Arc<()>,
    owner: Arc<()>,
    generation: u64,
    capture_id: u64,
    stamp: ReadbackStamp,
    channels: Vec<Channel>,
    health: Arc<DeviceHealth>,
}

/// Cloneable immutable device snapshot. The slot remains occupied while any
/// clone, tracked consumer or GPU copy still uses it, even after pool reset/drop.
#[derive(Clone)]
pub struct DeviceSnapshot {
    slot: Arc<SnapshotSlot>,
}

impl DeviceSnapshot {
    pub fn stamp(&self) -> ReadbackStamp {
        self.slot.stamp
    }
    pub fn capture_id(&self) -> u64 {
        self.slot.capture_id
    }
    pub fn channel_count(&self) -> usize {
        self.slot.channels.len()
    }

    pub fn channel<T: BufferElement>(&self, index: usize) -> Result<SnapshotChannel<'_, T>> {
        let channel = self
            .slot
            .channels
            .get(index)
            .ok_or_else(|| Error::Storage("snapshot: channel index out of range".into()))?;
        if channel.kind != TypeId::of::<T>() {
            return Err(Error::Storage("snapshot: channel type mismatch".into()));
        }
        Ok(SnapshotChannel {
            snapshot: self,
            channel,
            marker: PhantomData,
        })
    }
}

/// Typed channel at a device-compatible storage-binding offset.
pub struct SnapshotChannel<'a, T: BufferElement> {
    snapshot: &'a DeviceSnapshot,
    channel: &'a Channel,
    marker: PhantomData<T>,
}

impl<'a, T: BufferElement> SnapshotChannel<'a, T> {
    pub fn len(&self) -> usize {
        self.channel.bytes / self.channel.element_size
    }
    pub fn is_empty(&self) -> bool {
        self.channel.bytes == 0
    }

    /// Read-only consumer binding. Track this snapshot in DeviceCommandBatch;
    /// raw commands outside it require retaining the snapshot through completion.
    pub fn binding(&self) -> Result<wgpu::BufferBinding<'a>> {
        self.snapshot.slot.health.check()?;
        let size = NonZeroU64::new(self.channel.bytes as u64).ok_or_else(|| {
            Error::Storage("snapshot: empty channel has no storage binding".into())
        })?;
        Ok(wgpu::BufferBinding {
            buffer: &self.snapshot.slot.buffer,
            offset: self.channel.offset as u64,
            size: Some(size),
        })
    }

    pub fn readback_source(&self, range: Range<usize>) -> Result<ReadbackSource<'a>> {
        self.snapshot.slot.health.check()?;
        if range.start > range.end || range.end > self.len() {
            return Err(Error::Storage(
                "snapshot: readback range out of bounds".into(),
            ));
        }
        let bytes = range.len() * self.channel.element_size;
        Ok(ReadbackSource {
            device: self.snapshot.slot.device.clone(),
            buffer: (bytes != 0).then_some(&self.snapshot.slot.buffer),
            offset: (self.channel.offset + range.start * self.channel.element_size) as u64,
            channel: Channel {
                kind: self.channel.kind,
                offset: 0,
                bytes,
                element_size: self.channel.element_size,
            },
            lease: Some(self.snapshot.slot.in_flight.clone()),
        })
    }
}

/// Fixed-budget GPU-to-GPU snapshots. No host collection or blocking waits.
pub struct DeviceSnapshotPool {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    owner: Arc<()>,
    slots: Vec<Arc<SnapshotSlot>>,
    generation: u64,
    bytes_per_slot: u64,
    stats: DeviceSnapshotStats,
    failure: Option<String>,
    health: Arc<DeviceHealth>,
}

impl DeviceSnapshotPool {
    pub fn new(engine: &Engine, config: DeviceSnapshotConfig) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("snapshot requires GPU".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("snapshot requires queue".into()))?
            .clone();
        let bytes = align(config.bytes_per_slot, 4)?;
        let total = bytes
            .checked_mul(config.slots as u64)
            .ok_or_else(|| Error::Storage("snapshot: memory budget overflow".into()))?;
        if config.slots == 0
            || config.slots > 64
            || bytes == 0
            || bytes > device.limits().max_buffer_size
            || bytes > usize::MAX as u64
            || total > config.max_total_bytes
        {
            return Err(Error::Storage(
                "snapshot: slot/device/memory budget exceeded".into(),
            ));
        }
        let owner = Arc::new(());
        let slots = (0..config.slots)
            .map(|_| {
                Arc::new(SnapshotSlot {
                    device: device.clone(),
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("molla::sensor snapshot slot"),
                        size: bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    in_flight: Arc::new(()),
                    owner: owner.clone(),
                    generation: 0,
                    capture_id: 0,
                    stamp: ReadbackStamp::default(),
                    channels: Vec::with_capacity(64),
                    health: engine.device_health().expect("checked GPU engine").clone(),
                })
            })
            .collect();
        Ok(Self {
            device,
            queue,
            owner,
            slots,
            generation: 0,
            bytes_per_slot: bytes,
            stats: DeviceSnapshotStats {
                buffer_allocations: config.slots as u64,
                ..Default::default()
            },
            failure: None,
            health: engine.device_health().expect("checked GPU engine").clone(),
        })
    }

    pub fn stats(&self) -> DeviceSnapshotStats {
        self.stats
    }

    /// Poll completion callbacks without waiting; CPU leases keep slots occupied.
    pub fn available_slots(&mut self) -> Result<usize> {
        self.check()?;
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            let error = format!("snapshot device poll: {error}");
            self.failure = Some(error.clone());
            return Err(Error::Gpu(error));
        }
        self.health.check()?;
        Ok(self
            .slots
            .iter()
            .filter(|slot| slot_available(slot))
            .count())
    }

    /// Storage-aligned slot size needed for these active typed channels.
    pub fn required_bytes(&self, sources: &[ReadbackSource<'_>]) -> Result<u64> {
        self.health.check()?;
        if sources.len() > 64 {
            return Err(Error::Storage("snapshot: at most 64 channels".into()));
        }
        let mut bytes = 0u64;
        for source in sources {
            if !Arc::ptr_eq(&self.device, &source.device) {
                return Err(Error::Gpu("snapshot: foreign source device".into()));
            }
            if source.channel.bytes as u64 > self.device.limits().max_storage_buffer_binding_size {
                return Err(Error::Storage(
                    "snapshot: channel exceeds storage binding limit".into(),
                ));
            }
            bytes = align(
                bytes,
                self.device.limits().min_storage_buffer_offset_alignment as u64,
            )?
            .checked_add(source.channel.bytes as u64)
            .ok_or_else(|| Error::Storage("snapshot: channel size overflow".into()))?;
        }
        Ok(bytes)
    }

    /// Copy channels after prior work on this queue. None reports occupied slots.
    /// Sources must not be overwritten before this immediate copy submission.
    pub fn try_capture(
        &mut self,
        sources: &[ReadbackSource<'_>],
        stamp: ReadbackStamp,
    ) -> Result<Option<DeviceSnapshot>> {
        if stamp
            .simulation_time
            .is_some_and(|t| !t.is_finite() || t < 0.0)
        {
            return Err(Error::Storage(
                "snapshot: invalid simulation timestamp".into(),
            ));
        }
        if self.required_bytes(sources)? > self.bytes_per_slot {
            return Err(Error::Storage(
                "snapshot: channels exceed slot capacity".into(),
            ));
        }
        self.available_slots()?;
        let Some(index) = self.slots.iter().position(slot_available) else {
            self.stats.backpressure += 1;
            return Ok(None);
        };
        let capture_id = self
            .stats
            .captures
            .checked_add(1)
            .ok_or_else(|| Error::Storage("snapshot: capture identity exhausted".into()))?;
        let slot = Arc::get_mut(&mut self.slots[index]).expect("unleased snapshot slot");
        slot.generation = self.generation;
        slot.capture_id = capture_id;
        slot.stamp = stamp;
        slot.channels.clear();
        let alignment = self.device.limits().min_storage_buffer_offset_alignment as u64;
        let mut end = 0u64;
        for source in sources {
            let offset = align(end, alignment)?;
            let mut channel = source.channel.clone();
            channel.offset = offset as usize;
            end = offset + channel.bytes as u64;
            slot.channels.push(channel);
        }
        let bytes: u64 = sources.iter().map(|s| s.channel.bytes as u64).sum();
        let snapshot = DeviceSnapshot {
            slot: self.slots[index].clone(),
        };
        if bytes != 0 {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::sensor snapshot copy"),
                });
            for (source, destination) in sources.iter().zip(&snapshot.slot.channels) {
                if let Some(buffer) = source.buffer {
                    encoder.copy_buffer_to_buffer(
                        buffer,
                        source.offset,
                        &snapshot.slot.buffer,
                        destination.offset as u64,
                        source.channel.bytes as u64,
                    );
                }
            }
            self.queue.submit([encoder.finish()]);
            let destination = snapshot.slot.in_flight.clone();
            let sources: Vec<BufferLease> =
                sources.iter().filter_map(|s| s.lease.clone()).collect();
            self.queue
                .on_submitted_work_done(move || drop((destination, sources)));
            self.stats.submissions += 1;
            self.stats.copied_bytes += bytes;
        }
        self.stats.captures = capture_id;
        Ok(Some(snapshot))
    }

    /// Invalidate prior generations without reusing their occupied slots.
    pub fn reset(&mut self) -> Result<()> {
        self.check()?;
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::Storage("snapshot: generation exhausted".into()))?;
        Ok(())
    }

    pub fn is_current(&self, snapshot: &DeviceSnapshot) -> bool {
        self.check().is_ok()
            && Arc::ptr_eq(&self.owner, &snapshot.slot.owner)
            && self.generation == snapshot.slot.generation
    }

    /// Device loss and a failed completion poll are both terminal.
    fn check(&self) -> Result<()> {
        self.health.check()?;
        if let Some(error) = &self.failure {
            return Err(Error::Gpu(error.clone()));
        }
        Ok(())
    }
}

/// Owns unsubmitted commands and pins up to 64 distinct snapshots through GPU
/// completion. Dropping without submission releases the commands and their leases.
pub struct DeviceCommandBatch {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    encoder: wgpu::CommandEncoder,
    snapshots: Vec<Arc<SnapshotSlot>>,
    failed: bool,
    health: Arc<DeviceHealth>,
}

impl DeviceCommandBatch {
    pub fn new(engine: &Engine) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("device batch requires GPU".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("device batch requires queue".into()))?
            .clone();
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::leased sensor consumers"),
        });
        Ok(Self {
            device,
            queue,
            encoder,
            snapshots: Vec::with_capacity(64),
            failed: false,
            health: engine.device_health().expect("checked GPU engine").clone(),
        })
    }

    /// Encode consumers of read-only snapshots. Do not move out this encoder or
    /// mutate snapshot buffers. On closure failure/panic, submission is rejected.
    pub fn encode_with_snapshots<F>(
        &mut self,
        snapshots: &[&DeviceSnapshot],
        encode: F,
    ) -> Result<()>
    where
        F: FnOnce(&mut wgpu::CommandEncoder) -> Result<()>,
    {
        self.health.check()?;
        if self.failed {
            return Err(Error::Gpu("device batch: previous encoding failed".into()));
        }
        if snapshots.len() > 64 {
            return Err(Error::Storage(
                "device batch: too many snapshot references".into(),
            ));
        }
        for snapshot in snapshots {
            if !Arc::ptr_eq(&self.device, &snapshot.slot.device) {
                return Err(Error::Gpu("device batch: foreign snapshot".into()));
            }
        }
        for snapshot in snapshots {
            if !self
                .snapshots
                .iter()
                .any(|slot| Arc::ptr_eq(slot, &snapshot.slot))
            {
                if self.snapshots.len() == 64 {
                    self.failed = true;
                    return Err(Error::Storage(
                        "device batch: at most 64 distinct snapshots".into(),
                    ));
                }
                self.snapshots.push(snapshot.slot.clone());
            }
        }
        self.failed = true;
        encode(&mut self.encoder)?;
        self.failed = false;
        Ok(())
    }

    pub fn submit(self) -> Result<wgpu::SubmissionIndex> {
        self.health.check()?;
        if self.failed {
            return Err(Error::Gpu(
                "device batch: cannot submit failed encoding".into(),
            ));
        }
        let index = self.queue.submit([self.encoder.finish()]);
        let snapshots: Vec<_> = self
            .snapshots
            .iter()
            .map(|slot| slot.in_flight.clone())
            .collect();
        if !snapshots.is_empty() {
            self.queue.on_submitted_work_done(move || drop(snapshots));
        }
        Ok(index)
    }
}

fn slot_available(slot: &Arc<SnapshotSlot>) -> bool {
    Arc::strong_count(slot) == 1
        && Arc::weak_count(slot) == 0
        && Arc::strong_count(&slot.in_flight) == 1
}

fn align(bytes: u64, alignment: u64) -> Result<u64> {
    bytes
        .checked_add(alignment - 1)
        .map(|n| n / alignment * alignment)
        .ok_or_else(|| Error::Storage("snapshot: aligned size overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_alignment_is_checked() {
        assert_eq!(align(0, 256).unwrap(), 0);
        assert_eq!(align(4, 256).unwrap(), 256);
        assert_eq!(align(256, 256).unwrap(), 256);
        assert_eq!(align(257, 256).unwrap(), 512);
        assert!(align(u64::MAX, 256).is_err());
    }

    #[test]
    #[ignore = "needs a real GPU"]
    fn completion_callbacks_retain_tokens_not_device_resource_owners() {
        let engine = Engine::new_gpu().expect("GPU required");
        let source = crate::readback::DeviceBuffer::from_host(
            &engine,
            vec![7u32],
            molla_storage::BufferUsage::StorageReadWrite,
        )
        .unwrap();
        let mut pool = DeviceSnapshotPool::new(
            &engine,
            DeviceSnapshotConfig {
                slots: 1,
                bytes_per_slot: 4,
                max_total_bytes: 4,
            },
        )
        .unwrap();
        let snapshot = pool
            .try_capture(
                &[source.readback_source(0..1).unwrap()],
                ReadbackStamp::default(),
            )
            .unwrap()
            .unwrap();
        let owner = Arc::downgrade(&snapshot.slot);
        let mut batch = DeviceCommandBatch::new(&engine).unwrap();
        batch
            .encode_with_snapshots(&[&snapshot], |_| Ok(()))
            .unwrap();
        drop(snapshot);
        batch.submit().unwrap();
        drop(pool);
        assert!(owner.upgrade().is_none());
        engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
    }

    #[test]
    #[ignore = "needs a real GPU"]
    fn released_snapshots_wait_for_copy_completion_and_poll_failure_is_terminal() {
        let engine = Engine::new_gpu().expect("GPU required");
        let wait = || {
            engine
                .device()
                .unwrap()
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
        };
        let source = crate::readback::DeviceBuffer::from_host(
            &engine,
            vec![7u32],
            molla_storage::BufferUsage::StorageReadWrite,
        )
        .unwrap();
        let mut pool = DeviceSnapshotPool::new(
            &engine,
            DeviceSnapshotConfig {
                slots: 1,
                bytes_per_slot: 4,
                max_total_bytes: 4,
            },
        )
        .unwrap();
        let capture = |pool: &mut DeviceSnapshotPool| {
            pool.try_capture(
                &[source.readback_source(0..1).unwrap()],
                ReadbackStamp::default(),
            )
        };

        // Released by the CPU, but no poll has observed the copy completing.
        drop(capture(&mut pool).unwrap().unwrap());
        assert!(Arc::strong_count(&pool.slots[0].in_flight) > 1);
        assert!(!slot_available(&pool.slots[0]));
        wait();
        assert!(slot_available(&pool.slots[0]));

        let held = capture(&mut pool).unwrap().unwrap();
        wait();
        assert!(pool.is_current(&held));
        pool.failure = Some("injected poll failure".into());
        assert!(pool.available_slots().is_err());
        assert!(capture(&mut pool).is_err());
        assert!(pool.reset().is_err());
        assert!(!pool.is_current(&held));
        assert_eq!(pool.stats().captures, 2);
    }
}
