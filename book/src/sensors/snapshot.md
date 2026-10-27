# Leased GPU snapshots

`DeviceSnapshotPool` preserves selected sensor output channels entirely on the
GPU. A capture performs a device-to-device copy into a fixed-budget pool; it
does not download data or wait for GPU completion. `DeviceCommandBatch` retains
captured versions for consumers which may be encoded now and submitted later.
Source: `crates/molla-sensors/src/snapshot.rs`; requires `gpu`.

## Capture before reusing producer outputs

```rust,ignore
use molla::sensors::{DeviceSnapshotConfig, DeviceSnapshotPool};
use molla::sensors::readback::ReadbackStamp;

let mut pool = DeviceSnapshotPool::new(&engine, DeviceSnapshotConfig {
    slots: 3,
    bytes_per_slot: 1024 * 1024,
    max_total_bytes: 3 * 1024 * 1024,
})?;
let hits = raycaster.dispatch_scene(
    &engine, &scene, &state, &origins, &directions, max_distance,
)?;
let channels = [
    hits.distance.readback_source(0..hits.ray_count)?,
    hits.shape.readback_source(0..hits.ray_count)?,
];
let needed = pool.required_bytes(&channels)?;
let Some(snapshot) = pool.try_capture(&channels, ReadbackStamp {
    sensor_id, sample_id, scene_revision,
    simulation_time: Some(simulation_time),
})? else {
    report_backpressure();
    return Ok(());
};
```

`Some(snapshot)` means the copy is submitted after preceding work on that engine's
queue; the producer can now reuse its own outputs without changing the captured
version. `None` is backpressure: no version was accepted. If the result must be
preserved, do not overwrite the producer until retry succeeds. Otherwise record
an explicit rejected sample according to application policy. A capture is optional
when an immediately submitted consumer or host readback already provides the
required ordering.

There are 1–64 slots and at most 64 channels per capture. A channel starts at the
device's storage-buffer binding alignment; sizes therefore include padding
between channels. `required_bytes` computes this layout and validates provenance.
Each active channel must fit the storage binding limit, each slot the device
buffer limit, and all preallocated slots the explicit total budget. Constructor
sizes round up to four bytes. Empty channels are allowed but cannot be bound as
storage; all-empty captures submit no GPU copy.

## Tracked consumer commands

Snapshots are immutable and cloneable. `channel::<T>(index)` checks the original
host element type; its `binding()` supplies an offset and active size suitable
for a **read-only** storage binding. Build the consumer bind group using those
bindings, then encode through a tracked batch:

```rust,ignore
use molla::sensors::DeviceCommandBatch;

let mut batch = DeviceCommandBatch::new(&engine)?;
batch.encode_with_snapshots(&[&snapshot], |encoder| {
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&consumer_pipeline);
    pass.set_bind_group(0, &consumer_bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
    Ok(())
})?;
drop(snapshot);
// The batch still retains its input version, even before submission.
let submission = batch.submit()?;
```

Track every snapshot the commands consume, including snapshots from different
pools on the same engine. A batch accepts at most 64 distinct versions. The
versions remain unavailable for pool reuse until all CPU snapshot clones are
released **and** tracked GPU work completes. Dropping an unsubmitted batch
discards its commands and releases its leases. A failing/panicking encoding
closure poisons the batch; it cannot subsequently submit partial commands.
Foreign-engine snapshots are rejected before the encoding closure runs.

The encoder is borrowed for recording only: do not move it out, manually submit
its contents, or write into snapshot buffers. Raw wgpu buffers/bind groups used
outside `DeviceCommandBatch` are an explicit interop boundary: the caller must
retain the snapshot until those raw commands finish. Merely keeping a cloned
wgpu handle or unsubmitted command buffer is not a snapshot lease.

GPU-completion callbacks retain lightweight occupancy tokens, not owning
device/buffer objects. Polling the device processes release callbacks. The pool's
`available_slots()` does this nonblockingly; it never waits for a slot or
allocates replacement storage.

## Optional host collection and forwarding

A typed snapshot channel exposes `readback_source(range)` with the same interface
as `DeviceBuffer`. `ReadbackRing::try_enqueue` automatically retains its occupancy
token through GPU copy completion, including cancellation/reset of the readback
ticket. The ring transfers only requested payload bytes, not storage-alignment
padding. A host `ReadbackFrame` owns its own bytes and no longer needs the GPU slot.

Snapshot channels can also be captured into another pool. These forwarding copies
retain source occupancy until completion. Keeping a `DeviceSnapshot` after dropping
its pool remains valid; the pool owns allocation/reuse, not the only resource handle.

## Reset, failure and sizing policy

`pool.reset()` invalidates the current generation without reclaiming occupied
slots. `is_current(snapshot)` checks pool identity and generation; historical
snapshots remain readable but should not be treated as current-scene data.
Their `stamp()` and monotonically increasing `capture_id()` do not change.
Use scheduler/sample generation checks as well when scene freshness matters.
For owned ray scenes, retain the result's `SceneStamp` alongside the snapshot;
the generic readback stamp does not replace scene identity and shape resolution.

The pool is fixed-capacity. For growth, create a new explicitly budgeted pool and
retire the old one; outstanding old snapshots retain their resources until released.
Both pools count toward application memory while they coexist. There is no global
process-wide budget or implicit eviction of accepted versions.

Observed poll errors or engine device-loss notifications make the pool unusable;
reset does not recover the device. Lost-device snapshots are not current and
cannot expose GPU bindings/readback sources. Recreate against a new engine.
Logical-device shutdown/callback and fresh-engine reconstruction are verified;
physical driver/device recovery is not. See [device lifecycle](./device_lifecycle.md).
Ordinary budget/type/provenance errors remain recoverable without consuming a slot.

## Cost and current limits

`stats()` reports slot-buffer allocations, accepted captures, copy submissions,
copied payload bytes and backpressure. Slot buffers and channel metadata capacity
are preallocated. Command encoders, callback bookkeeping and consumer-created
bind groups are not claimed allocation-free.

This is a **copy-based** lifetime option: nonempty capture adds one device-to-device
copy submission and extra result storage. It does not make existing borrowed
sensor views persistent automatically, and is not direct sensor dispatch into
leased output slots. Consumer batches can combine downstream work, but producer
dispatches still manage their own submissions. Retained bindings for remaining
sensors, public cross-sensor producer encoding and full performance acceptance
remain in `SENSORS.md`.

```sh
make test-gpu P=molla-sensors T=sensor_snapshot_gpu A='--test-threads=1'
make test-sensors
```
