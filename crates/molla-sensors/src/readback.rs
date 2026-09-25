//! Bounded asynchronous collection of mixed sensor output channels.

use std::{
    any::{Any, TypeId},
    ops::Range,
    sync::{Arc, Mutex},
    time::Duration,
};

use molla_compute::{DeviceHealth, Engine};
use molla_core::{Error, Result};
use molla_storage::{Buffer, BufferElement, BufferUsage};

/// Sensor storage with an engine identity that is retained across resource clones.
pub struct DeviceBuffer<T: BufferElement> {
    pub(crate) buffer: Buffer<T>,
    pub(crate) device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    health: Arc<DeviceHealth>,
}

impl<T: BufferElement> DeviceBuffer<T> {
    pub fn from_host(engine: &Engine, values: Vec<T>, usage: BufferUsage) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("sensor buffer requires a GPU".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("sensor buffer requires a GPU queue".into()))?
            .clone();
        let bytes = values
            .len()
            .checked_mul(core::mem::size_of::<T::Gpu>())
            .ok_or_else(|| Error::Storage("sensor buffer size overflow".into()))?
            as u64;
        if bytes > device.limits().max_buffer_size
            || bytes > device.limits().max_storage_buffer_binding_size as u64
        {
            return Err(Error::Storage("sensor buffer exceeds GPU limits".into()));
        }
        let mut buffer = Buffer::from_vec(values, usage);
        buffer.ensure_device(&device)?;
        buffer.upload(&queue)?;
        Ok(Self {
            buffer,
            device,
            queue,
            health: engine.device_health().expect("checked GPU engine").clone(),
        })
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub(crate) fn check_device(&self) -> Result<()> {
        self.health.check()
    }
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    pub fn host(&self) -> Result<&[T]> {
        self.buffer.host()
    }
    pub fn host_mut(&mut self) -> Result<&mut [T]> {
        self.buffer.host_mut()
    }
    pub fn upload(&mut self) -> Result<()> {
        self.health.check()?;
        self.buffer.upload(&self.queue)
    }
    pub fn device_buffer(&self) -> Result<&wgpu::Buffer> {
        self.health.check()?;
        self.buffer.device_buffer_current()
    }
    pub fn mark_device_dirty(&mut self) {
        self.buffer.mark_device_dirty();
    }
    pub fn readback_source(&self, range: Range<usize>) -> Result<ReadbackSource<'_>> {
        self.health.check()?;
        ReadbackSource::from_owned(&self.device, &self.buffer, range)
    }
}

/// Fixed staging allocation and total GPU memory budget.
#[derive(Debug, Clone, Copy)]
pub struct ReadbackConfig {
    pub slots: usize,
    pub bytes_per_slot: u64,
    pub max_total_bytes: u64,
}

/// Caller-provided identity carried unchanged through collection.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ReadbackStamp {
    pub sensor_id: u64,
    pub sample_id: u64,
    pub scene_revision: u64,
    pub simulation_time: Option<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct Channel {
    pub(crate) kind: TypeId,
    pub(crate) offset: usize,
    pub(crate) bytes: usize,
    pub(crate) element_size: usize,
}

pub(crate) type BufferLease = Arc<dyn Any + Send + Sync>;

/// A checked active range of a device-authoritative or synchronized buffer.
pub struct ReadbackSource<'a> {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) buffer: Option<&'a wgpu::Buffer>,
    pub(crate) offset: u64,
    pub(crate) channel: Channel,
    pub(crate) lease: Option<BufferLease>,
}

impl<'a> ReadbackSource<'a> {
    pub(crate) fn from_owned<T: BufferElement>(
        device: &Arc<wgpu::Device>,
        buffer: &'a Buffer<T>,
        range: Range<usize>,
    ) -> Result<Self> {
        if range.start > range.end || range.end > buffer.len() {
            return Err(Error::Storage(
                "readback range exceeds buffer length".into(),
            ));
        }
        let size = core::mem::size_of::<T::Gpu>();
        let offset = range
            .start
            .checked_mul(size)
            .ok_or_else(|| Error::Storage("readback source offset overflow".into()))?;
        let bytes = range
            .len()
            .checked_mul(size)
            .ok_or_else(|| Error::Storage("readback source size overflow".into()))?;
        if size == 0 || offset % 4 != 0 || bytes % 4 != 0 {
            return Err(Error::Storage(
                "readback requires four-byte copy alignment".into(),
            ));
        }
        let source = if range.is_empty() {
            None
        } else {
            Some(buffer.device_buffer_current()?)
        };
        Ok(Self {
            device: device.clone(),
            lease: None,
            buffer: source,
            offset: offset as u64,
            channel: Channel {
                kind: TypeId::of::<T>(),
                offset: 0,
                bytes,
                element_size: size,
            },
        })
    }
}

/// An outstanding request, valid only in its originating ring until consumed/cancelled.
#[derive(Debug, Clone)]
pub struct ReadbackTicket {
    owner: Arc<()>,
    serial: u64,
    slot: usize,
}

/// Collected active data, independent of staging slot reuse.
#[derive(Debug)]
pub struct ReadbackFrame {
    pub stamp: ReadbackStamp,
    bytes: Vec<u8>,
    channels: Vec<Channel>,
}

impl ReadbackFrame {
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Decode one channel using its original host element type.
    pub fn channel<T: BufferElement>(&self, index: usize) -> Result<Vec<T>> {
        Ok(self.channel_iter::<T>(index)?.collect())
    }

    /// Decode elements without allocating an intermediate host array.
    pub fn channel_iter<T: BufferElement>(
        &self,
        index: usize,
    ) -> Result<impl ExactSizeIterator<Item = T> + '_> {
        let channel = self
            .channels
            .get(index)
            .ok_or_else(|| Error::Storage("readback channel index out of range".into()))?;
        if channel.kind != TypeId::of::<T>() {
            return Err(Error::Storage(
                "readback channel element type mismatch".into(),
            ));
        }
        Ok(self.bytes[channel.offset..channel.offset + channel.bytes]
            .chunks_exact(channel.element_size)
            .map(|bytes| T::from_gpu(&bytemuck::pod_read_unaligned::<T::Gpu>(bytes))))
    }
}

/// Instrumentation for allocation, transfer and collection behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadbackStats {
    pub staging_allocations: u64,
    pub submissions: u64,
    pub copied_bytes: u64,
    pub blocking_waits: u64,
    pub cancellations: u64,
}

type Completion = Arc<Mutex<(u64, Option<core::result::Result<(), String>>)>>;

struct Pending {
    serial: u64,
    cancelled: bool,
    bytes: usize,
    channels: Vec<Channel>,
    stamp: ReadbackStamp,
}

struct Slot {
    staging: wgpu::Buffer,
    completion: Completion,
    pending: Option<Pending>,
}

/// Fixed-capacity staging pool. `try_enqueue` returns `None` under backpressure.
pub struct ReadbackRing {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    owner: Arc<()>,
    slots: Vec<Slot>,
    bytes_per_slot: u64,
    serial: u64,
    stats: ReadbackStats,
    failure: Option<String>,
    health: Arc<DeviceHealth>,
}

impl ReadbackRing {
    pub fn new(engine: &Engine, config: ReadbackConfig) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("readback requires a GPU".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("readback requires a GPU queue".into()))?
            .clone();
        let bytes = config
            .bytes_per_slot
            .checked_add(7)
            .map(|n| n / 8 * 8)
            .ok_or_else(|| Error::Storage("readback capacity overflow".into()))?;
        let total = bytes
            .checked_mul(config.slots as u64)
            .ok_or_else(|| Error::Storage("readback total capacity overflow".into()))?;
        if config.slots == 0
            || config.slots > 64
            || bytes == 0
            || bytes > device.limits().max_buffer_size
            || total > config.max_total_bytes
        {
            return Err(Error::Storage(
                "readback allocation exceeds slot/device/memory budget".into(),
            ));
        }
        let slots = (0..config.slots)
            .map(|_| Slot {
                staging: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("molla::sensor readback slot"),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                completion: Arc::new(Mutex::new((0, None))),
                pending: None,
            })
            .collect();
        Ok(Self {
            device,
            queue,
            owner: Arc::new(()),
            slots,
            bytes_per_slot: bytes,
            serial: 0,
            stats: ReadbackStats {
                staging_allocations: config.slots as u64,
                ..Default::default()
            },
            failure: None,
            health: engine.device_health().expect("checked GPU engine").clone(),
        })
    }

    pub fn stats(&self) -> ReadbackStats {
        self.stats
    }
    pub fn available_slots(&self) -> usize {
        if self.failure.is_some() || self.health.is_lost() {
            return 0;
        }
        self.slots.iter().filter(|s| s.pending.is_none()).count()
    }

    pub fn is_failed(&self) -> bool {
        self.failure.is_some() || self.health.is_lost()
    }

    fn fail(&mut self, reason: String) -> Error {
        self.failure = Some(reason.clone());
        for slot in &mut self.slots {
            if let Some(pending) = slot.pending.take() {
                if !pending.cancelled {
                    self.stats.cancellations += 1;
                }
            }
            if let Ok(mut completion) = slot.completion.lock() {
                *completion = (0, None);
            }
        }
        Error::Gpu(reason)
    }

    /// Reclaim cancelled slots after mapping completes, without waiting or reallocating.
    pub fn poll_releases(&mut self) -> Result<()> {
        if let Some(reason) = &self.failure {
            return Err(Error::Gpu(reason.clone()));
        }
        if let Err(error) = self.health.check() {
            return Err(self.fail(error.to_string()));
        }
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Err(self.fail(format!("readback device poll: {error}")));
        }
        if let Err(error) = self.health.check() {
            return Err(self.fail(error.to_string()));
        }
        for index in 0..self.slots.len() {
            let slot = &mut self.slots[index];
            if !slot.pending.as_ref().is_some_and(|p| p.cancelled) {
                continue;
            }
            let completion = slot.completion.lock().map_err(poisoned)?.1.take();
            match completion {
                None => {}
                Some(Err(error)) => return Err(self.fail(format!("readback mapping: {error}"))),
                Some(Ok(())) => {
                    let pending = slot.pending.take().expect("cancelled request");
                    if pending.bytes > 0 {
                        slot.staging.unmap();
                    }
                }
            }
        }
        Ok(())
    }

    /// Copy requested ranges after preceding work on this engine's queue; never wait for completion.
    pub fn try_enqueue(
        &mut self,
        sources: &[ReadbackSource<'_>],
        stamp: ReadbackStamp,
    ) -> Result<Option<ReadbackTicket>> {
        self.poll_releases()?;
        if sources.len() > 64 {
            return Err(Error::Storage(
                "readback supports at most 64 channels per request".into(),
            ));
        }
        if stamp
            .simulation_time
            .is_some_and(|time| !time.is_finite() || time < 0.0)
        {
            return Err(Error::Storage(
                "readback simulation time must be finite and nonnegative".into(),
            ));
        }
        let mut bytes = 0usize;
        let mut channels = Vec::with_capacity(sources.len());
        for source in sources {
            if !Arc::ptr_eq(&source.device, &self.device) {
                return Err(Error::Storage(
                    "readback source belongs to a different device".into(),
                ));
            }
            let mut channel = source.channel.clone();
            channel.offset = bytes;
            bytes = bytes
                .checked_add(channel.bytes)
                .ok_or_else(|| Error::Storage("readback batch size overflow".into()))?;
            channels.push(channel);
        }
        let mapped_bytes = (bytes as u64)
            .checked_add(7)
            .map(|n| n / 8 * 8)
            .ok_or_else(|| Error::Storage("readback batch size overflow".into()))?;
        if mapped_bytes > self.bytes_per_slot {
            return Err(Error::Storage(
                "readback batch exceeds staging capacity".into(),
            ));
        }
        let Some(index) = self.slots.iter().position(|slot| slot.pending.is_none()) else {
            return Ok(None);
        };
        let serial = self
            .serial
            .checked_add(1)
            .ok_or_else(|| Error::Storage("readback request counter exhausted".into()))?;
        let slot = &mut self.slots[index];
        *slot.completion.lock().map_err(poisoned)? = (serial, None);
        self.serial = serial;
        slot.pending = Some(Pending {
            serial,
            cancelled: false,
            bytes,
            channels,
            stamp,
        });
        if bytes == 0 {
            slot.completion.lock().map_err(poisoned)?.1 = Some(Ok(()));
        } else {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::sensor grouped readback"),
                });
            let mut destination = 0;
            for source in sources {
                if let Some(buffer) = source.buffer {
                    encoder.copy_buffer_to_buffer(
                        buffer,
                        source.offset,
                        &slot.staging,
                        destination,
                        source.channel.bytes as u64,
                    );
                }
                destination += source.channel.bytes as u64;
            }
            self.queue.submit([encoder.finish()]);
            let leases: Vec<_> = sources
                .iter()
                .filter_map(|source| source.lease.clone())
                .collect();
            if !leases.is_empty() {
                self.queue.on_submitted_work_done(move || drop(leases));
            }
            self.stats.submissions += 1;
            self.stats.copied_bytes += bytes as u64;
            let completion = slot.completion.clone();
            slot.staging
                .slice(0..mapped_bytes)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if let Ok(mut state) = completion.lock() {
                        if state.0 == serial {
                            state.1 = Some(result.map_err(|e| e.to_string()));
                        }
                    }
                });
        }
        Ok(Some(ReadbackTicket {
            owner: self.owner.clone(),
            serial,
            slot: index,
        }))
    }

    fn validate_ticket(&self, ticket: &ReadbackTicket) -> Result<()> {
        if !Arc::ptr_eq(&self.owner, &ticket.owner)
            || self
                .slots
                .get(ticket.slot)
                .and_then(|s| s.pending.as_ref())
                .is_none_or(|p| p.serial != ticket.serial || p.cancelled)
        {
            return Err(Error::Storage(
                "readback ticket is foreign, cancelled or already consumed".into(),
            ));
        }
        Ok(())
    }

    /// Nonblocking collection; `None` means the requested transfer is still pending.
    pub fn poll(&mut self, ticket: &ReadbackTicket) -> Result<Option<ReadbackFrame>> {
        self.validate_ticket(ticket)?;
        self.poll_releases()?;
        let slot = &mut self.slots[ticket.slot];
        let Some(completion) = slot.completion.lock().map_err(poisoned)?.1.take() else {
            return Ok(None);
        };
        if let Err(error) = completion {
            return Err(self.fail(format!("readback mapping: {error}")));
        }
        let pending = slot.pending.take().expect("validated ticket");
        let bytes = if pending.bytes == 0 {
            Vec::new()
        } else {
            let mapped_bytes = (pending.bytes as u64 + 7) / 8 * 8;
            let view = slot.staging.slice(0..mapped_bytes).get_mapped_range();
            let bytes = view[..pending.bytes].to_vec();
            drop(view);
            slot.staging.unmap();
            bytes
        };
        Ok(Some(ReadbackFrame {
            stamp: pending.stamp,
            bytes,
            channels: pending.channels,
        }))
    }

    /// Explicit synchronous convenience with a finite timeout; timeout leaves the ticket retryable.
    pub fn wait(&mut self, ticket: &ReadbackTicket, timeout: Duration) -> Result<ReadbackFrame> {
        if let Some(frame) = self.poll(ticket)? {
            return Ok(frame);
        }
        self.stats.blocking_waits += 1;
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(timeout),
            })
            .map_err(|error| Error::Gpu(format!("readback wait: {error}")))?;
        self.poll(ticket)?
            .ok_or_else(|| Error::Gpu("readback completion still pending".into()))
    }

    /// Invalidate a ticket; its staging slot is reclaimed after pending GPU work completes.
    pub fn cancel(&mut self, ticket: &ReadbackTicket) -> Result<()> {
        self.validate_ticket(ticket)?;
        let slot = &mut self.slots[ticket.slot];
        slot.pending.as_mut().expect("validated request").cancelled = true;
        self.stats.cancellations += 1;
        Ok(())
    }

    /// Cancel all outstanding requests, including completed but uncollected mappings.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            if let Some(pending) = &mut slot.pending {
                if !pending.cancelled {
                    pending.cancelled = true;
                    self.stats.cancellations += 1;
                }
            }
        }
    }
}

impl Drop for ReadbackRing {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if let Ok(mut completion) = slot.completion.lock() {
                *completion = (0, None);
            }
        }
    }
}

fn poisoned<T>(_: std::sync::PoisonError<T>) -> Error {
    Error::Storage("readback completion state poisoned".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "needs a real GPU"]
    fn readback_mapping_failure_is_terminal_and_pending_wait_is_retryable() {
        let engine = Engine::new_gpu().unwrap();
        let mut ring = ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: 8,
                max_total_bytes: 8,
            },
        )
        .unwrap();
        let ticket = ring
            .try_enqueue(&[], ReadbackStamp::default())
            .unwrap()
            .unwrap();
        ring.slots[0].completion.lock().unwrap().1 = Some(Err("injected map failure".into()));
        assert!(
            ring.poll(&ticket)
                .unwrap_err()
                .to_string()
                .contains("injected map failure")
        );
        assert!(ring.is_failed());
        assert!(ring.try_enqueue(&[], ReadbackStamp::default()).is_err());
        let mut ring = ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: 8,
                max_total_bytes: 8,
            },
        )
        .unwrap();
        let ticket = ring
            .try_enqueue(&[], ReadbackStamp::default())
            .unwrap()
            .unwrap();
        ring.slots[0].completion.lock().unwrap().1 = None;
        assert!(ring.wait(&ticket, Duration::ZERO).is_err());
        ring.slots[0].completion.lock().unwrap().1 = Some(Ok(()));
        assert_eq!(ring.poll(&ticket).unwrap().unwrap().byte_len(), 0);
    }

    #[test]
    #[ignore = "needs a real GPU"]
    fn cancelled_slots_are_reclaimed_only_by_poll_releases_after_mapping() {
        let engine = Engine::new_gpu().unwrap();
        let source = DeviceBuffer::from_host(
            &engine,
            vec![7u32, 8],
            molla_storage::BufferUsage::StorageReadWrite,
        )
        .unwrap();
        let mut ring = ReadbackRing::new(
            &engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: 8,
                max_total_bytes: 8,
            },
        )
        .unwrap();
        let sources = [source.readback_source(0..2).unwrap()];
        let ticket = ring
            .try_enqueue(&sources, ReadbackStamp::default())
            .unwrap()
            .unwrap();
        ring.cancel(&ticket).unwrap();
        assert_eq!(ring.available_slots(), 0);
        assert!(ring.poll(&ticket).is_err());
        engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(10)),
            })
            .unwrap();
        assert_eq!(ring.available_slots(), 0);
        ring.poll_releases().unwrap();
        assert_eq!(ring.available_slots(), 1);
        let ticket = ring
            .try_enqueue(&sources, ReadbackStamp::default())
            .unwrap()
            .unwrap();
        let frame = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
        assert_eq!(frame.channel::<u32>(0).unwrap(), vec![7, 8]);
        assert_eq!(ring.stats().staging_allocations, 1);
    }
}
