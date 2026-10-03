//! Direct producer outputs with bounded slot leases and explicit submission.

use crate::readback::{BufferLease, Channel, ReadbackSource, ReadbackStamp};
use arrayvec::ArrayVec;
use molla_compute::{DeviceHealth, Engine};
use molla_core::{Error, Result};
use molla_storage::BufferElement;
use std::{
    any::TypeId,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SensorChannelSpec {
    kind: TypeId,
    element_size: usize,
    capacity: usize,
}

impl SensorChannelSpec {
    pub fn of<T: BufferElement>(capacity: usize) -> Self {
        Self {
            kind: TypeId::of::<T>(),
            element_size: size_of::<T::Gpu>(),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn bytes(&self) -> Result<u64> {
        if self.element_size == 0 || !self.element_size.is_multiple_of(4) {
            return Err(Error::Storage(
                "sensor output: invalid element width".into(),
            ));
        }
        self.capacity
            .checked_mul(self.element_size)
            .and_then(|bytes| u64::try_from(bytes.max(self.element_size)).ok())
            .ok_or_else(|| Error::Storage("sensor output: size overflow".into()))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SensorFramePoolConfig {
    pub slots: usize,
    pub max_total_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SensorFramePoolStats {
    pub buffer_allocations: u64,
    pub reserved_frames: u64,
    pub backpressure: u64,
}

struct OutputSlot {
    device: Arc<wgpu::Device>,
    health: Arc<DeviceHealth>,
    owner: Arc<()>,
    in_flight: Arc<()>,
    buffers: Vec<wgpu::Buffer>,
    specs: Vec<SensorChannelSpec>,
    ranges: Vec<Range<usize>>,
    generation: u64,
    stamp: ReadbackStamp,
}

impl OutputSlot {
    fn available(slot: &Arc<Self>) -> bool {
        Arc::strong_count(slot) == 1
            && Arc::weak_count(slot) == 0
            && Arc::strong_count(&slot.in_flight) == 1
    }
}

/// Fixed channel capacities; each slot has separate storage buffers per channel.
pub struct SensorFramePool {
    device: Arc<wgpu::Device>,
    health: Arc<DeviceHealth>,
    owner: Arc<()>,
    slots: Vec<Arc<OutputSlot>>,
    specs: Vec<SensorChannelSpec>,
    generation: u64,
    stats: SensorFramePoolStats,
    failure: Option<String>,
}

impl SensorFramePool {
    pub fn new(
        engine: &Engine,
        config: SensorFramePoolConfig,
        specs: &[SensorChannelSpec],
    ) -> Result<Self> {
        engine.check_device()?;
        if !(1..=64).contains(&config.slots) || !(1..=64).contains(&specs.len()) {
            return Err(Error::Storage(
                "sensor output: expected 1..=64 slots/channels".into(),
            ));
        }
        let device = engine.device().expect("checked engine").clone();
        let health = engine.device_health().expect("checked engine").clone();
        let mut bytes_per_slot = 0u64;
        for spec in specs {
            let bytes = spec.bytes()?;
            if bytes > device.limits().max_buffer_size
                || bytes > device.limits().max_storage_buffer_binding_size
            {
                return Err(Error::Storage(
                    "sensor output: device capacity exceeded".into(),
                ));
            }
            bytes_per_slot = bytes_per_slot
                .checked_add(bytes)
                .ok_or_else(|| Error::Storage("sensor output: budget overflow".into()))?;
        }
        if bytes_per_slot
            .checked_mul(config.slots as u64)
            .is_none_or(|total| total > config.max_total_bytes)
        {
            return Err(Error::Storage(
                "sensor output: memory budget exceeded".into(),
            ));
        }
        let owner = Arc::new(());
        let slots = (0..config.slots)
            .map(|_| {
                Arc::new(OutputSlot {
                    device: device.clone(),
                    health: health.clone(),
                    owner: owner.clone(),
                    in_flight: Arc::new(()),
                    buffers: specs
                        .iter()
                        .map(|spec| {
                            device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("molla::direct sensor output"),
                                size: spec.bytes().expect("validated channel size"),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_SRC
                                    | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            })
                        })
                        .collect(),
                    specs: specs.to_vec(),
                    ranges: vec![0..0; specs.len()],
                    generation: 0,
                    stamp: ReadbackStamp::default(),
                })
            })
            .collect();
        Ok(Self {
            device,
            health,
            owner,
            slots,
            specs: specs.to_vec(),
            generation: 0,
            stats: SensorFramePoolStats {
                buffer_allocations: (config.slots * specs.len()) as u64,
                ..Default::default()
            },
            failure: None,
        })
    }

    pub fn stats(&self) -> SensorFramePoolStats {
        self.stats
    }

    /// Poll completion callbacks without waiting. Live CPU frames still occupy slots.
    pub fn available_slots(&mut self) -> Result<usize> {
        self.check()?;
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            let message = format!("sensor output poll: {error}");
            self.failure = Some(message.clone());
            return Err(Error::Gpu(message));
        }
        self.check()?;
        Ok(self
            .slots
            .iter()
            .filter(|slot| OutputSlot::available(slot))
            .count())
    }

    /// Advance identity without reclaiming pending or externally retained frames.
    pub fn reset(&mut self) -> Result<()> {
        self.check()?;
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::Storage("sensor output: generation exhausted".into()))?;
        Ok(())
    }

    pub fn is_current(&self, frame: &SensorFrame) -> bool {
        self.check().is_ok()
            && Arc::ptr_eq(&self.owner, &frame.slot.owner)
            && self.generation == frame.slot.generation
    }

    fn check(&self) -> Result<()> {
        self.health.check()?;
        if let Some(error) = &self.failure {
            return Err(Error::Gpu(error.clone()));
        }
        Ok(())
    }

    fn reserve(
        &mut self,
        device: &Arc<wgpu::Device>,
        specs: &[SensorChannelSpec],
        ranges: &[Range<usize>],
        stamp: ReadbackStamp,
    ) -> Result<Option<Arc<OutputSlot>>> {
        self.validate_layout(device, specs)?;
        if ranges.len() != specs.len()
            || ranges
                .iter()
                .zip(specs)
                .any(|(range, spec)| range.start > range.end || range.end > spec.capacity)
        {
            return Err(Error::Storage(
                "sensor output: channel layout/range mismatch".into(),
            ));
        }
        let Some(slot) = self
            .slots
            .iter_mut()
            .find(|slot| OutputSlot::available(slot))
        else {
            self.stats.backpressure += 1;
            return Ok(None);
        };
        let output = Arc::get_mut(slot).expect("exclusive available slot");
        output.ranges.clone_from_slice(ranges);
        output.stamp = stamp;
        output.generation = self.generation;
        self.stats.reserved_frames += 1;
        Ok(Some(slot.clone()))
    }

    pub(crate) fn validate_layout(
        &self,
        device: &Arc<wgpu::Device>,
        specs: &[SensorChannelSpec],
    ) -> Result<()> {
        self.check()?;
        if !Arc::ptr_eq(device, &self.device) {
            return Err(Error::Gpu("sensor output: foreign pool".into()));
        }
        if specs != self.specs {
            return Err(Error::Storage(
                "sensor output: channel layout/range mismatch".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn preflight_available(&mut self) -> Result<bool> {
        self.check()?;
        let available = self.slots.iter().any(OutputSlot::available);
        if !available {
            self.stats.backpressure += 1;
        }
        Ok(available)
    }
}

/// Submitted immutable result. Clones keep the slot occupied through CPU release.
#[derive(Clone)]
pub struct SensorFrame {
    slot: Arc<OutputSlot>,
}

impl SensorFrame {
    pub fn stamp(&self) -> ReadbackStamp {
        self.slot.stamp
    }
    pub fn channel_count(&self) -> usize {
        self.slot.specs.len()
    }
    pub fn channel<T: BufferElement>(&self, index: usize) -> Result<SensorFrameChannel<'_, T>> {
        self.slot.health.check()?;
        if self
            .slot
            .specs
            .get(index)
            .is_none_or(|spec| spec.kind != TypeId::of::<T>())
        {
            return Err(Error::Storage(
                "sensor output: channel index/type mismatch".into(),
            ));
        }
        Ok(SensorFrameChannel {
            slot: &self.slot,
            index,
            marker: PhantomData,
        })
    }
}

pub struct SensorFrameChannel<'a, T: BufferElement> {
    slot: &'a OutputSlot,
    index: usize,
    marker: PhantomData<T>,
}

impl<'a, T: BufferElement> SensorFrameChannel<'a, T> {
    pub fn len(&self) -> usize {
        self.slot.ranges[self.index].len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Active indices within the full-capacity storage binding.
    pub fn active_range(&self) -> Range<usize> {
        self.slot.ranges[self.index].clone()
    }

    /// Full-capacity read-only binding; consumers must use `active_range()`.
    /// Retain the frame until raw consumer commands complete.
    pub fn binding(&self) -> Result<wgpu::BufferBinding<'a>> {
        self.slot.health.check()?;
        Ok(wgpu::BufferBinding {
            buffer: &self.slot.buffers[self.index],
            offset: 0,
            size: None,
        })
    }

    /// Range relative to the active channel, excluding internal producer slots.
    pub fn readback_source(&self, range: Range<usize>) -> Result<ReadbackSource<'a>> {
        self.slot.health.check()?;
        if range.start > range.end || range.end > self.len() {
            return Err(Error::Storage(
                "sensor output: readback range out of bounds".into(),
            ));
        }
        let spec = self.slot.specs[self.index];
        let bytes = range.len() * spec.element_size;
        Ok(ReadbackSource {
            device: self.slot.device.clone(),
            buffer: (bytes != 0).then_some(&self.slot.buffers[self.index]),
            offset: ((self.slot.ranges[self.index].start + range.start) * spec.element_size) as u64,
            channel: Channel {
                kind: spec.kind,
                offset: 0,
                bytes,
                element_size: spec.element_size,
            },
            lease: Some(self.slot.in_flight.clone()),
        })
    }
}

/// Identity of one command batch; process-unique and never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchId(u64);

static NEXT_BATCH: AtomicU64 = AtomicU64::new(0);

impl BatchId {
    fn next() -> Self {
        Self(NEXT_BATCH.fetch_add(1, Ordering::Relaxed))
    }
}

/// Opaque receipt; it has no readable channels until its batch submits.
#[derive(Clone)]
pub struct EncodedSensorFrame {
    owner: BatchId,
    index: usize,
}

impl EncodedSensorFrame {
    pub(crate) fn same_receipt(&self, other: &Self) -> bool {
        self.owner == other.owner && self.index == other.index
    }
}

/// Scoped inputs from earlier producers in the same command batch.
pub struct PendingSensorInputs<'a> {
    slots: &'a [Option<Arc<OutputSlot>>],
}

impl PendingSensorInputs<'_> {
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    fn slot(&self, input: usize) -> Result<&OutputSlot> {
        let slot = self
            .slots
            .get(input)
            .and_then(Option::as_deref)
            .ok_or_else(|| {
                Error::Storage("sensor batch: pending input index out of bounds".into())
            })?;
        slot.health.check()?;
        Ok(slot)
    }

    pub fn stamp(&self, input: usize) -> Result<ReadbackStamp> {
        Ok(self.slot(input)?.stamp)
    }

    pub fn channel<T: BufferElement>(
        &self,
        input: usize,
        channel: usize,
    ) -> Result<PendingSensorChannel<'_, T>> {
        let slot = self.slot(input)?;
        if slot
            .specs
            .get(channel)
            .is_none_or(|spec| spec.kind != TypeId::of::<T>())
        {
            return Err(Error::Storage(
                "sensor batch: pending channel index/type mismatch".into(),
            ));
        }
        Ok(PendingSensorChannel {
            slot,
            index: channel,
            range: slot.ranges[channel].clone(),
            marker: PhantomData,
        })
    }
}

/// A typed read-only binding, available only inside a pending-input callback.
pub struct PendingSensorChannel<'a, T: BufferElement> {
    slot: &'a OutputSlot,
    index: usize,
    range: Range<usize>,
    marker: PhantomData<T>,
}

impl<T: BufferElement> PendingSensorChannel<'_, T> {
    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    pub fn active_range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Select a range relative to this view, not the full storage buffer.
    pub fn slice(&self, range: Range<usize>) -> Result<PendingSensorChannel<'_, T>> {
        self.slot.health.check()?;
        if range.start > range.end || range.end > self.len() {
            return Err(Error::Storage(
                "sensor batch: pending channel range out of bounds".into(),
            ));
        }
        Ok(PendingSensorChannel {
            slot: self.slot,
            index: self.index,
            range: self.range.start + range.start..self.range.start + range.end,
            marker: PhantomData,
        })
    }

    /// Full-capacity binding; shaders must apply `active_range()` themselves.
    pub fn binding(&self) -> Result<wgpu::BufferBinding<'_>> {
        self.slot.health.check()?;
        Ok(wgpu::BufferBinding {
            buffer: &self.slot.buffers[self.index],
            offset: 0,
            size: None,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SensorCommandStats {
    pub encoded_frames: u64,
    pub kernel_dispatches: u64,
    pub output_clears: u64,
    pub submissions: u64,
    /// Encoded copies; temporal producer state commits only at submission.
    pub history_copies: u64,
    /// Host payload bytes queued by producers, excluding uniform parameters.
    pub input_upload_bytes: u64,
    /// Successful pending-input callbacks, including custom output producers.
    pub downstream_encodes: u64,
    /// Successful auxiliary callbacks; not producer kernel dispatches.
    pub auxiliary_encodes: u64,
}

pub(crate) trait SubmitAction: Send {
    fn commit(&mut self);
    fn cancel(&mut self);
}

struct PendingCommit<'a> {
    action: &'a mut dyn SubmitAction,
    applied: bool,
}

impl Drop for PendingCommit<'_> {
    fn drop(&mut self) {
        if !self.applied {
            self.action.cancel();
        }
    }
}

pub struct SensorSubmission {
    owner: BatchId,
    frames: ArrayVec<SensorFrame, 64>,
    pub index: Option<wgpu::SubmissionIndex>,
    pub stats: SensorCommandStats,
}

impl SensorSubmission {
    pub fn frame(&self, receipt: &EncodedSensorFrame) -> Result<&SensorFrame> {
        if self.owner != receipt.owner {
            return Err(Error::Storage(
                "sensor output: receipt belongs to another batch".into(),
            ));
        }
        self.frames
            .get(receipt.index)
            .ok_or_else(|| Error::Storage("sensor output: invalid receipt".into()))
    }
    pub fn into_frames(self) -> Vec<SensorFrame> {
        self.into_frames_iter().collect()
    }

    /// Consume submitted frames in encoding order without allocating a result vector.
    pub fn into_frames_iter(self) -> impl ExactSizeIterator<Item = SensorFrame> {
        self.frames.into_iter()
    }
}

/// Producer/state borrows last until submission or drop. At most 64 outputs.
/// Raw wgpu handles must not mutate leased inputs/outputs behind these borrows.
pub struct SensorCommandBatch<'a> {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    health: Arc<DeviceHealth>,
    owner: BatchId,
    encoder: wgpu::CommandEncoder,
    frames: ArrayVec<SensorFrame, 64>,
    failed: bool,
    inputs: ArrayVec<Arc<OutputSlot>, 64>,
    source_leases: ArrayVec<BufferLease, 64>,
    source_count: usize,
    commits: ArrayVec<PendingCommit<'a>, 64>,
    stats: SensorCommandStats,
    marker: PhantomData<&'a mut ()>,
}

impl<'a> SensorCommandBatch<'a> {
    pub fn new(engine: &Engine) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().expect("checked engine").clone();
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sensor producers"),
        });
        Ok(Self {
            device,
            queue: engine.queue().expect("checked engine").clone(),
            health: engine.device_health().expect("checked engine").clone(),
            owner: BatchId::next(),
            encoder,
            frames: ArrayVec::new(),
            inputs: ArrayVec::new(),
            source_leases: ArrayVec::new(),
            source_count: 0,
            commits: ArrayVec::new(),
            failed: false,
            stats: Default::default(),
            marker: PhantomData,
        })
    }

    pub fn stats(&self) -> SensorCommandStats {
        self.stats
    }

    pub(crate) fn check_output_capacity(&self, count: usize) -> Result<()> {
        self.health.check()?;
        if self.failed || self.frames.len().checked_add(count).is_none_or(|n| n > 64) {
            return Err(Error::Storage(
                "sensor batch: poisoned or output limit exceeded".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn poison(&mut self) {
        self.failed = true;
    }

    pub(crate) fn check_commit_capacity(&self) -> Result<()> {
        self.health.check()?;
        if self.failed || self.commits.len() == 64 {
            return Err(Error::Storage(
                "sensor batch: poisoned or commit limit exceeded".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn on_submit(&mut self, action: &'a mut dyn SubmitAction) -> Result<()> {
        if let Err(error) = self.check_commit_capacity() {
            self.failed = true;
            action.cancel();
            return Err(error);
        }
        self.commits.push(PendingCommit {
            action,
            applied: false,
        });
        Ok(())
    }

    pub(crate) fn record_history_copy(&mut self) {
        self.stats.history_copies += 1;
    }

    pub(crate) fn record_input_upload(&mut self, bytes: u64) {
        self.stats.input_upload_bytes += bytes;
    }

    pub(crate) fn check_source_capacity(&self, count: usize) -> Result<()> {
        self.health.check()?;
        if self.failed
            || self
                .inputs
                .len()
                .checked_add(self.source_count)
                .and_then(|total| total.checked_add(count))
                .is_none_or(|total| total > 64)
        {
            return Err(Error::Storage(
                "sensor batch: poisoned or input limit exceeded".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn retain_source(&mut self, source: &ReadbackSource<'_>) -> Result<()> {
        if let Err(error) = self.check_source_capacity(1) {
            self.failed = true;
            return Err(error);
        }
        if !Arc::ptr_eq(&self.device, &source.device) {
            self.failed = true;
            return Err(Error::Gpu("sensor batch: foreign retained source".into()));
        }
        if let Some(lease) = &source.lease {
            self.source_leases.push(lease.clone());
        }
        self.source_count += 1;
        Ok(())
    }

    pub(crate) fn pending_range<T: BufferElement>(
        &self,
        receipt: &EncodedSensorFrame,
        channel: usize,
        range: Range<usize>,
    ) -> Result<Range<usize>> {
        let slots = self.pending_slots(&[receipt])?;
        let inputs = PendingSensorInputs { slots: &slots[..1] };
        Ok(inputs
            .channel::<T>(0, channel)?
            .slice(range)?
            .active_range())
    }

    /// Encode auxiliary commands such as timestamp markers and query resolves.
    ///
    /// This does not retain sensor-frame leases. Frame consumers must use
    /// `encode_with_frames` or `encode_with_pending` instead. The callback must
    /// use this encoder and this device's resources, not queue writes or submissions.
    /// At most 128 callbacks per batch. Callback failure poisons the batch.
    pub fn encode_auxiliary(
        &mut self,
        encode: impl FnOnce(&mut wgpu::CommandEncoder) -> Result<()>,
    ) -> Result<()> {
        self.health.check()?;
        if self.failed || self.stats.auxiliary_encodes == 128 {
            return Err(Error::Storage(
                "sensor batch: poisoned or auxiliary limit exceeded".into(),
            ));
        }
        self.failed = true;
        encode(&mut self.encoder)?;
        self.failed = false;
        self.stats.auxiliary_encodes += 1;
        Ok(())
    }

    /// Encode raw read-only consumers of submitted frames, retaining their slots
    /// through GPU completion. The callback must use only this device's resources
    /// and must not write input bindings or submit separate commands using them.
    pub fn encode_with_frames(
        &mut self,
        frames: &[&SensorFrame],
        encode: impl FnOnce(&mut wgpu::CommandEncoder) -> Result<()>,
    ) -> Result<()> {
        self.health.check()?;
        self.check_source_capacity(frames.len())?;
        if frames.is_empty() {
            return Err(Error::Storage(
                "sensor batch: poisoned or invalid input count".into(),
            ));
        }
        for frame in frames {
            frame.slot.health.check()?;
            if !Arc::ptr_eq(&self.device, &frame.slot.device) {
                return Err(Error::Gpu("sensor batch: foreign input frame".into()));
            }
        }
        self.failed = true;
        self.inputs
            .extend(frames.iter().map(|frame| frame.slot.clone()));
        encode(&mut self.encoder)?;
        self.failed = false;
        Ok(())
    }

    fn pending_slots(
        &self,
        receipts: &[&EncodedSensorFrame],
    ) -> Result<[Option<Arc<OutputSlot>>; 64]> {
        self.health.check()?;
        if self.failed || !(1..=64).contains(&receipts.len()) {
            return Err(Error::Storage(
                "sensor batch: poisoned or invalid pending input count".into(),
            ));
        }
        for receipt in receipts {
            if self.owner != receipt.owner || receipt.index >= self.frames.len() {
                return Err(Error::Storage(
                    "sensor batch: pending receipt belongs to another batch".into(),
                ));
            }
        }
        let mut slots = std::array::from_fn(|_| None);
        for (slot, receipt) in slots.iter_mut().zip(receipts) {
            *slot = Some(self.frames[receipt.index].slot.clone());
        }
        Ok(slots)
    }

    /// Encode a read-only GPU consumer after earlier producers, before submission.
    /// Return the number of dispatched kernels. Raw handles must stay on this
    /// device and must not mutate inputs or escape into independently submitted work.
    ///
    /// ```compile_fail
    /// use molla_sensors::{EncodedSensorFrame, SensorCommandBatch};
    /// fn escape(batch: &mut SensorCommandBatch<'_>, receipt: &EncodedSensorFrame) {
    ///     let mut channel = None;
    ///     batch.encode_with_pending(&[receipt], |_, inputs| {
    ///         channel = Some(inputs.channel::<u32>(0, 0)?);
    ///         Ok(0)
    ///     }).unwrap();
    ///     channel.unwrap().binding().unwrap();
    /// }
    /// ```
    pub fn encode_with_pending(
        &mut self,
        receipts: &[&EncodedSensorFrame],
        encode: impl FnOnce(&mut wgpu::CommandEncoder, PendingSensorInputs<'_>) -> Result<u64>,
    ) -> Result<()> {
        let slots = self.pending_slots(receipts)?;
        self.failed = true;
        let dispatches = encode(
            &mut self.encoder,
            PendingSensorInputs {
                slots: &slots[..receipts.len()],
            },
        )?;
        self.stats.kernel_dispatches = self
            .stats
            .kernel_dispatches
            .checked_add(dispatches)
            .ok_or_else(|| Error::Storage("sensor batch: dispatch count overflow".into()))?;
        self.stats.downstream_encodes += 1;
        self.failed = false;
        Ok(())
    }

    /// Encode a custom producer consuming earlier outputs into a new leased slot.
    /// All output channels are cleared first. Input bindings are read-only;
    /// output writes must use this encoder, not queue writes or separate submissions.
    /// Return the number of dispatched kernels. Backpressure skips the callback.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_from_pending(
        &mut self,
        receipts: &[&EncodedSensorFrame],
        pool: &mut SensorFramePool,
        specs: &[SensorChannelSpec],
        ranges: &[Range<usize>],
        stamp: ReadbackStamp,
        encode: impl FnOnce(
            &mut wgpu::CommandEncoder,
            PendingSensorInputs<'_>,
            &[wgpu::Buffer],
        ) -> Result<u64>,
    ) -> Result<Option<EncodedSensorFrame>> {
        let slots = self.pending_slots(receipts)?;
        let result = self.encode_output(pool, specs, ranges, stamp, |_, _, encoder, outputs| {
            encode(
                encoder,
                PendingSensorInputs {
                    slots: &slots[..receipts.len()],
                },
                outputs,
            )
        })?;
        if result.is_some() {
            self.stats.downstream_encodes += 1;
        }
        Ok(result)
    }

    pub(crate) fn check_engine(&self, engine: &Engine) -> Result<()> {
        self.health.check()?;
        engine.check_device()?;
        if self.failed {
            return Err(Error::Gpu("sensor batch: poisoned".into()));
        }
        if !Arc::ptr_eq(&self.device, engine.device().expect("checked engine")) {
            return Err(Error::Gpu("sensor batch: foreign engine".into()));
        }
        Ok(())
    }

    pub(crate) fn check_device(&self, device: &Arc<wgpu::Device>) -> Result<()> {
        self.health.check()?;
        if self.failed || !Arc::ptr_eq(&self.device, device) {
            return Err(Error::Gpu(
                "sensor batch: poisoned or foreign device".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn encode_output(
        &mut self,
        pool: &mut SensorFramePool,
        specs: &[SensorChannelSpec],
        ranges: &[Range<usize>],
        stamp: ReadbackStamp,
        encode: impl FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            &[wgpu::Buffer],
        ) -> Result<u64>,
    ) -> Result<Option<EncodedSensorFrame>> {
        self.encode_output_initialized(pool, specs, ranges, stamp, u64::MAX, encode)
    }

    /// Channels excluded from `clear_mask` are initialized by the producer's queue writes.
    pub(crate) fn encode_output_initialized(
        &mut self,
        pool: &mut SensorFramePool,
        specs: &[SensorChannelSpec],
        ranges: &[Range<usize>],
        stamp: ReadbackStamp,
        clear_mask: u64,
        encode: impl FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            &[wgpu::Buffer],
        ) -> Result<u64>,
    ) -> Result<Option<EncodedSensorFrame>> {
        self.health.check()?;
        if self.failed || self.frames.len() == 64 {
            return Err(Error::Storage(
                "sensor batch: poisoned or output limit exceeded".into(),
            ));
        }
        let Some(slot) = pool.reserve(&self.device, specs, ranges, stamp)? else {
            return Ok(None);
        };
        self.failed = true;
        let mut clears = 0;
        for (index, buffer) in slot.buffers.iter().enumerate() {
            if clear_mask & (1u64 << index) != 0 {
                self.encoder.clear_buffer(buffer, 0, None);
                clears += 1;
            }
        }
        let dispatches = encode(&self.device, &self.queue, &mut self.encoder, &slot.buffers)?;
        let dispatches = self
            .stats
            .kernel_dispatches
            .checked_add(dispatches)
            .ok_or_else(|| Error::Storage("sensor batch: dispatch count overflow".into()))?;
        let index = self.frames.len();
        self.stats.output_clears += clears;
        self.frames.push(SensorFrame { slot });
        self.stats.encoded_frames += 1;
        self.stats.kernel_dispatches = dispatches;
        self.failed = false;
        Ok(Some(EncodedSensorFrame {
            owner: self.owner,
            index,
        }))
    }

    /// Submit once; only successful submission publishes readable frame handles.
    pub fn submit(mut self) -> Result<SensorSubmission> {
        self.health.check()?;
        if self.failed {
            return Err(Error::Gpu("sensor batch: poisoned".into()));
        }
        if self.frames.is_empty() && self.inputs.is_empty() && self.stats.auxiliary_encodes == 0 {
            return Ok(SensorSubmission {
                owner: self.owner,
                frames: self.frames,
                index: None,
                stats: self.stats,
            });
        }
        let tokens: ArrayVec<_, 128> = self
            .frames
            .iter()
            .map(|frame| frame.slot.in_flight.clone())
            .chain(self.inputs.iter().map(|slot| slot.in_flight.clone()))
            .collect();
        let index = self.queue.submit([self.encoder.finish()]);
        let sources = self.source_leases;
        self.queue
            .on_submitted_work_done(move || drop((tokens, sources)));
        self.stats.submissions = 1;
        for commit in &mut self.commits {
            commit.applied = true;
            commit.action.commit();
        }
        Ok(SensorSubmission {
            owner: self.owner,
            frames: self.frames,
            index: Some(index),
            stats: self.stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_deduplication_uses_owner_identity_and_output_index() {
        let first = EncodedSensorFrame {
            owner: BatchId::next(),
            index: 0,
        };
        let next = EncodedSensorFrame {
            owner: first.owner,
            index: 1,
        };
        let foreign = EncodedSensorFrame {
            owner: BatchId::next(),
            index: 0,
        };
        assert!(first.same_receipt(&first.clone()));
        assert!(!first.same_receipt(&next));
        assert!(!first.same_receipt(&foreign));
    }

    #[test]
    #[ignore = "requires GPU"]
    fn bounded_device_source_leases_release_on_completion_cancellation_error_and_loss() {
        let engine = Engine::new_gpu().unwrap();
        let specs = [SensorChannelSpec::of::<u32>(0)];
        let mut pool = SensorFramePool::new(
            &engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 4,
            },
            &specs,
        )
        .unwrap();
        for mode in 0..4 {
            let tokens: Vec<_> = (0..64).map(|_| Arc::new(())).collect();
            let weak: Vec<_> = tokens.iter().map(Arc::downgrade).collect();
            let sources: Vec<_> = tokens
                .iter()
                .map(|token| ReadbackSource {
                    device: engine.device().unwrap().clone(),
                    buffer: None,
                    offset: 0,
                    channel: Channel {
                        kind: TypeId::of::<u32>(),
                        offset: 0,
                        bytes: 0,
                        element_size: 4,
                    },
                    lease: Some(token.clone()),
                })
                .collect();
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let receipt = batch
                .encode_output(
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..0)),
                    Default::default(),
                    |_, _, _, _| Ok(0),
                )
                .unwrap()
                .unwrap();
            let storage = batch.source_leases.as_ptr();
            batch.check_source_capacity(64).unwrap();
            for source in &sources {
                batch.retain_source(source).unwrap();
            }
            assert_eq!(batch.source_leases.as_ptr(), storage);
            assert_eq!(batch.source_leases.capacity(), 64);
            assert!(batch.check_source_capacity(1).is_err());
            drop(sources);
            drop(tokens);
            assert!(weak.iter().all(|token| token.strong_count() == 1));
            match mode {
                0 => drop(batch),
                1 => {
                    assert!(
                        batch
                            .encode_with_pending(&[&receipt], |_, _| Err(Error::Gpu(
                                "injected".into()
                            )))
                            .is_err()
                    );
                    assert!(batch.submit().is_err());
                }
                2 => {
                    drop(batch.submit().unwrap());
                    engine
                        .device()
                        .unwrap()
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(std::time::Duration::from_secs(30)),
                        })
                        .unwrap();
                }
                _ => {
                    engine.destroy_device().unwrap();
                    assert!(batch.submit().is_err());
                }
            }
            assert!(weak.iter().all(|token| token.upgrade().is_none()));
        }
    }

    #[test]
    #[ignore = "requires GPU"]
    fn pending_custom_producers_validate_before_work_and_poison_on_error_or_panic() {
        let engine = Engine::new_gpu().unwrap();
        let specs = [SensorChannelSpec::of::<u32>(0)];
        let config = SensorFramePoolConfig {
            slots: 2,
            max_total_bytes: 8,
        };
        let mut pool = SensorFramePool::new(&engine, config, &specs).unwrap();
        let foreign = Engine::new_gpu().unwrap();
        let mut foreign_pool = SensorFramePool::new(&foreign, config, &specs).unwrap();
        let mut abandoned = None;
        for failure in 0..4 {
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let receipt = batch
                .encode_output(
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..0)),
                    Default::default(),
                    |_, _, _, _| Ok(0),
                )
                .unwrap()
                .unwrap();
            if let Some(old) = abandoned.as_ref() {
                assert!(
                    batch
                        .encode_with_pending(&[old], |_, _| panic!("abandoned receipt"))
                        .is_err()
                );
            }
            assert!(
                batch
                    .encode_from_pending(
                        &[&receipt],
                        &mut foreign_pool,
                        &specs,
                        std::slice::from_ref(&(0..0)),
                        Default::default(),
                        |_, _, _| panic!("foreign pool")
                    )
                    .is_err()
            );
            assert_eq!(foreign_pool.stats().reserved_frames, 0);
            assert!(
                batch
                    .encode_from_pending(
                        &[&receipt],
                        &mut pool,
                        &[SensorChannelSpec::of::<i32>(0)],
                        std::slice::from_ref(&(0..0)),
                        Default::default(),
                        |_, _, _| panic!("wrong layout")
                    )
                    .is_err()
            );
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                batch.encode_from_pending(
                    &[&receipt],
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..0)),
                    Default::default(),
                    |encoder, inputs, outputs| {
                        let channel = inputs.channel::<u32>(0, 0)?;
                        assert!(channel.is_empty());
                        assert_eq!(channel.active_range(), 0..0);
                        assert_eq!(channel.binding()?.buffer.size(), 4);
                        assert!(channel.slice(0..1).is_err());
                        assert_eq!(channel.slice(0..0)?.len(), 0);
                        encoder.clear_buffer(&outputs[0], 0, None);
                        match failure {
                            0 => Err(Error::Gpu("injected output error".into())),
                            1 => panic!("injected output panic"),
                            _ => Ok(u64::MAX),
                        }
                    },
                )
            }));
            match failure {
                0 => assert!(result.unwrap().is_err()),
                1 => assert!(result.is_err()),
                _ => {
                    let stage = result.unwrap().unwrap().unwrap();
                    if failure == 2 {
                        assert!(batch.encode_with_pending(&[&stage], |_, _| Ok(1)).is_err());
                    } else {
                        let mut more = SensorFramePool::new(&engine, config, &specs).unwrap();
                        assert!(
                            batch
                                .encode_from_pending(
                                    &[&stage],
                                    &mut more,
                                    &specs,
                                    std::slice::from_ref(&(0..0)),
                                    Default::default(),
                                    |_, _, _| Ok(1)
                                )
                                .is_err()
                        );
                        assert_eq!(more.available_slots().unwrap(), 2);
                    }
                }
            }
            assert!(batch.submit().is_err());
            assert_eq!(pool.available_slots().unwrap(), 2);
            abandoned = Some(receipt);
        }
        assert_eq!(pool.stats().buffer_allocations, 2);
        assert_eq!(pool.stats().reserved_frames, 8);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn pending_custom_chain_obeys_output_bound_and_does_not_pin_cancelled_slots() {
        let engine = Engine::new_gpu().unwrap();
        let specs = [SensorChannelSpec::of::<u32>(0)];
        let mut pool = SensorFramePool::new(
            &engine,
            SensorFramePoolConfig {
                slots: 64,
                max_total_bytes: 256,
            },
            &specs,
        )
        .unwrap();
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let mut last = batch
            .encode_output(
                &mut pool,
                &specs,
                std::slice::from_ref(&(0..0)),
                Default::default(),
                |_, _, _, _| Ok(0),
            )
            .unwrap()
            .unwrap();
        for _ in 1..64 {
            last = batch
                .encode_from_pending(
                    &[&last],
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..0)),
                    Default::default(),
                    |_, inputs, _| {
                        assert!(inputs.channel::<u32>(0, 0)?.is_empty());
                        Ok(0)
                    },
                )
                .unwrap()
                .unwrap();
        }
        assert_eq!(batch.stats().downstream_encodes, 63);
        assert_eq!(batch.stats().encoded_frames, 64);
        assert!(
            batch
                .encode_from_pending(
                    &[&last],
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..0)),
                    Default::default(),
                    |_, _, _| panic!("output limit")
                )
                .is_err()
        );
        assert_eq!(pool.available_slots().unwrap(), 0);
        drop(batch);
        assert_eq!(pool.available_slots().unwrap(), 64);
    }

    #[derive(Default)]
    struct CountAction {
        commits: usize,
        cancels: usize,
    }
    impl SubmitAction for CountAction {
        fn commit(&mut self) {
            self.commits += 1;
        }
        fn cancel(&mut self) {
            self.cancels += 1;
        }
    }

    #[test]
    #[ignore = "requires GPU"]
    fn bounded_submission_actions_commit_once_or_cancel_on_drop_error_and_loss() {
        let engine = Engine::new_gpu().unwrap();
        let specs = [SensorChannelSpec::of::<u32>(1)];
        let mut pool = SensorFramePool::new(
            &engine,
            SensorFramePoolConfig {
                slots: 64,
                max_total_bytes: 256,
            },
            &specs,
        )
        .unwrap();
        let mut actions: Vec<_> = (0..64).map(|_| CountAction::default()).collect();
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let storage = batch.commits.as_ptr();
        for action in &mut actions {
            batch.check_commit_capacity().unwrap();
            batch
                .encode_output(
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..1)),
                    Default::default(),
                    |_, _, _, _| Ok(0),
                )
                .unwrap()
                .unwrap();
            batch.on_submit(action).unwrap();
        }
        assert_eq!(batch.commits.as_ptr(), storage);
        assert_eq!(batch.commits.capacity(), 64);
        assert!(batch.check_commit_capacity().is_err());
        drop(batch.submit().unwrap());
        assert!(
            actions
                .iter()
                .all(|action| action.commits == 1 && action.cancels == 0)
        );
        engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .unwrap();
        assert_eq!(pool.available_slots().unwrap(), 64);
        for failure in 0..3 {
            let mut action = CountAction::default();
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            batch
                .encode_output(
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..1)),
                    Default::default(),
                    |_, _, _, _| Ok(0),
                )
                .unwrap()
                .unwrap();
            batch.on_submit(&mut action).unwrap();
            if failure == 0 {
                drop(batch);
            } else {
                if failure == 1 {
                    assert!(
                        batch
                            .encode_output(
                                &mut pool,
                                &specs,
                                std::slice::from_ref(&(0..1)),
                                Default::default(),
                                |_, _, _, _| Err(Error::Gpu("injected producer failure".into()))
                            )
                            .is_err()
                    );
                } else {
                    engine.destroy_device().unwrap();
                }
                assert!(batch.submit().is_err());
            }
            assert_eq!(action.commits, 0);
            assert_eq!(action.cancels, 1);
        }
    }

    #[derive(Clone, Copy)]
    struct Empty;
    impl BufferElement for Empty {
        type Gpu = [u32; 0];
        fn to_gpu(&self) -> Self::Gpu {
            []
        }
        fn from_gpu(_: &Self::Gpu) -> Self {
            Self
        }
    }

    #[test]
    fn channel_sizes_reject_overflow_and_zero_width() {
        assert!(SensorChannelSpec::of::<Empty>(1).bytes().is_err());
        assert!(
            SensorChannelSpec::of::<molla_math::Vec3>(usize::MAX)
                .bytes()
                .is_err()
        );
        assert_eq!(SensorChannelSpec::of::<u32>(0).bytes().unwrap(), 4);
        assert_eq!(
            SensorChannelSpec::of::<molla_math::Vec3>(0)
                .bytes()
                .unwrap(),
            16
        );
        assert_eq!(
            SensorChannelSpec::of::<molla_math::Vec3>(3)
                .bytes()
                .unwrap(),
            48
        );
    }

    #[test]
    #[ignore = "requires GPU"]
    fn zero_channels_and_failed_or_panicking_encoding_never_publish_or_leak_slots() {
        let engine = Engine::new_gpu().unwrap();
        let specs = [SensorChannelSpec::of::<u32>(0)];
        let mut pool = SensorFramePool::new(
            &engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 4,
            },
            &specs,
        )
        .unwrap();
        let mut failed = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            failed
                .encode_output(
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..0)),
                    Default::default(),
                    |_, _, _, _| { Err(Error::Gpu("injected encode error".into())) }
                )
                .is_err()
        );
        assert!(failed.submit().is_err());
        assert_eq!(pool.available_slots().unwrap(), 1);
        let mut panicked = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = panicked.encode_output(
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..0)),
                    Default::default(),
                    |_, _, _, _| panic!("injected"),
                );
            }))
            .is_err()
        );
        assert!(panicked.submit().is_err());
        assert_eq!(pool.available_slots().unwrap(), 1);
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let receipt = batch
            .encode_output(
                &mut pool,
                &specs,
                std::slice::from_ref(&(0..0)),
                Default::default(),
                |_, _, _, _| Ok(0),
            )
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        let frame = submitted.frame(&receipt).unwrap();
        let channel = frame.channel::<u32>(0).unwrap();
        assert!(channel.is_empty());
        assert_eq!(channel.active_range(), 0..0);
        assert_eq!(channel.binding().unwrap().buffer.size(), 4);
        assert!(channel.readback_source(0..1).is_err());
        let source = channel.readback_source(0..0).unwrap();
        assert_eq!(source.channel.bytes, 0);
        assert!(source.buffer.is_none());
        drop(source);
        drop(submitted);
        engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .unwrap();
        assert_eq!(pool.available_slots().unwrap(), 1);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn maximum_pending_outputs_are_bounded_and_receipts_do_not_hold_slots() {
        let engine = Engine::new_gpu().unwrap();
        let specs = [SensorChannelSpec::of::<u32>(1)];
        let config = SensorFramePoolConfig {
            slots: 64,
            max_total_bytes: 256,
        };
        let mut pool = SensorFramePool::new(&engine, config, &specs).unwrap();
        let mut other = SensorFramePool::new(&engine, config, &specs).unwrap();
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        let receipts: Vec<_> = (0..64)
            .map(|_| {
                batch
                    .encode_output(
                        &mut pool,
                        &specs,
                        std::slice::from_ref(&(0..1)),
                        Default::default(),
                        |_, _, _, _| Ok(0),
                    )
                    .unwrap()
                    .unwrap()
            })
            .collect();
        assert_eq!(pool.available_slots().unwrap(), 0);
        assert!(
            batch
                .encode_output(
                    &mut other,
                    &specs,
                    std::slice::from_ref(&(0..1)),
                    Default::default(),
                    |_, _, _, _| panic!("must not encode")
                )
                .is_err()
        );
        assert_eq!(other.stats().reserved_frames, 0);
        drop(batch);
        assert_eq!(receipts.len(), 64);
        assert_eq!(pool.available_slots().unwrap(), 64);
        assert_eq!(pool.stats().buffer_allocations, 64);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn released_frames_wait_for_gpu_completion_and_poll_failure_is_terminal() {
        let engine = Engine::new_gpu().unwrap();
        let specs = [SensorChannelSpec::of::<u32>(4)];
        let mut pool = SensorFramePool::new(
            &engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 16,
            },
            &specs,
        )
        .unwrap();
        let wait = || {
            engine
                .device()
                .unwrap()
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(30)),
                })
                .unwrap();
        };
        let encode = |pool: &mut SensorFramePool| {
            let mut batch = SensorCommandBatch::new(&engine).unwrap();
            let receipt = batch
                .encode_output(
                    pool,
                    &specs,
                    std::slice::from_ref(&(0..4)),
                    Default::default(),
                    |_, _, encoder, outputs| {
                        encoder.clear_buffer(&outputs[0], 0, None);
                        Ok(1)
                    },
                )
                .unwrap()
                .unwrap();
            let submitted = batch.submit().unwrap();
            assert_eq!(submitted.stats.submissions, 1);
            submitted.frame(&receipt).unwrap().clone()
        };

        // Released by the CPU, but no poll has observed the submission completing.
        drop(encode(&mut pool));
        assert!(Arc::strong_count(&pool.slots[0].in_flight) > 1);
        assert!(!OutputSlot::available(&pool.slots[0]));
        wait();
        assert!(OutputSlot::available(&pool.slots[0]));

        let held = encode(&mut pool);
        wait();
        assert!(pool.is_current(&held));
        pool.failure = Some("injected poll failure".into());
        assert!(pool.available_slots().is_err());
        assert!(pool.reset().is_err());
        assert!(!pool.is_current(&held));
        let mut batch = SensorCommandBatch::new(&engine).unwrap();
        assert!(
            batch
                .encode_output(
                    &mut pool,
                    &specs,
                    std::slice::from_ref(&(0..4)),
                    Default::default(),
                    |_, _, _, _| { panic!("must not encode") }
                )
                .is_err()
        );
        assert_eq!(pool.stats().reserved_frames, 2);
    }
}
