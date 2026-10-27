# Asynchronous sensor readback

`molla::sensors::readback` provides `DeviceBuffer`, `ReadbackRing`, typed channel
decoding, request stamps and collection statistics. Source:
`crates/molla-sensors/src/readback.rs`. The `gpu` feature is required.

## Ownership and queue ordering

`DeviceBuffer::from_host` uploads sensor-owned storage and records its engine's
identity. Sensor device-result views expose the same typed storage without a
mutable buffer replacement API. `readback_source(range)` validates active ranges
and rejects stale GPU mirrors. A ring rejects a source from another engine before
submitting work, even when the two wgpu instances have equal numeric device IDs.

Submit a producer or GPU consumer on the originating engine queue before reusing
its output. `try_enqueue` submits grouped copies after earlier work on that queue;
later writes cannot overwrite the captured staging snapshot. It does not wait
for GPU completion. Raw wgpu handles still require the caller to honor queue
ordering; merely encoding an unsubmitted consumer does not protect its inputs.

For deferred GPU consumers, [leased snapshots](./snapshot.md) provide fixed-budget
device copies and completion-tracked batches. Snapshot readback sources carry
occupancy tokens which `ReadbackRing` retains through GPU copy completion, even
if the ticket is cancelled or its ring is reset. Ordinary borrowed sensor views
retain the original immediate-submission contract above.

Collection does not update a source's host mirror. Device-authoritative buffers
remain host-stale; the returned `ReadbackFrame` is an independent host snapshot.

## Configuration and requests

```rust,ignore
use molla::sensors::readback::{ReadbackConfig, ReadbackRing, ReadbackStamp};

let bytes = imu.sensor_count() as u64 * 32; // Two padded vec3 channels.
let mut ring = ReadbackRing::new(&engine, ReadbackConfig {
    slots: 3,
    bytes_per_slot: bytes,
    max_total_bytes: bytes * 3,
})?;

if let Some(readings) = imu.dispatch(&engine, &model, &state, sample_dt)? {
    let n = readings.accelerometer.len();
    let channels = [
        readings.accelerometer.readback_source(0..n)?,
        readings.gyroscope.readback_source(0..n)?,
    ];
    let stamp = ReadbackStamp {
        sensor_id: 7,
        sample_id,
        scene_revision,
        simulation_time: Some(simulation_time),
    };
    match ring.try_enqueue(&channels, stamp)? {
        Some(ticket) => pending.push(ticket),
        None => report_backpressure(), // Explicit caller policy; no request was queued.
    }
}

// Poll later; keep the ticket when the result is still None.
if let Some(frame) = ring.poll(&pending[0])? {
    let acceleration = frame.channel::<molla::math::Vec3>(0)?;
    let gyro = frame.channel::<molla::math::Vec3>(1)?;
    consume(frame.stamp, acceleration, gyro);
}
```

Staging buffers are allocated once at construction. Sizes round up to eight
bytes and must fit both device and explicit total-memory budgets. There are
1–64 slots and at most 64 channels per request. Source copy offsets/sizes must
be four-byte aligned. Empty channels/batches complete without a GPU copy.

Each request may mix element types. Decoding uses the original host element
type (`Real`, `Vec3`, `u32`, etc.); incorrect types and channel indices return
errors. `channel_iter` avoids allocating intermediate decoded arrays. The raw
frame contains only requested bytes, not staging padding or unused buffer capacity.

`ReadbackStamp` carries caller-supplied sensor/sample/revision identity and an
optional simulation timestamp unchanged. It is not a scheduler or a source of
physical time. Legacy synchronous paths leave time unspecified rather than
inventing a timestamp.

## Completion, cancellation and limits

- `poll` uses nonblocking device polling and returns `None` while mapping is pending.
- `wait(ticket, timeout)` is an explicit synchronous convenience with a finite
  wait. A timeout/pending result leaves the ticket available for retry or cancellation.
- `cancel` invalidates one ticket; `reset` invalidates every outstanding ticket.
  Cancelled slots remain reserved until their GPU mapping finishes. `poll_releases`
  and subsequent enqueue/poll calls reclaim them without waiting or allocating.
  Late callbacks carry generation numbers and cannot complete a reused slot.
- Drop releases handles without waiting for outstanding GPU work. Foreign, cancelled or consumed tickets are
  rejected. A completed frame remains valid after its slot is reused.
- Mapping/poll failures return errors, not fabricated data, and make the ring
  terminal (`is_failed`). Recreate the ring after recovering the device/context;
  reset does not revive failed GPU resources. Engine-observed loss also rejects
  zero-byte requests. See [device loss and recovery boundaries](./device_lifecycle.md)
  for logical shutdown, backend callback delivery and explicit reconstruction.

The pool is fixed-capacity: larger requests fail rather than growing allocations
inside a frame. For changed sensor capacity, cancel/drain requests and construct
a ring with an explicit new memory budget. CPU channel metadata and collected
frame bytes still allocate; zero staging reallocations is not a claim of zero
CPU allocations.

`stats()` records staging allocations, copy submissions, requested bytes,
explicit blocking waits and cancellations. It does not instrument every shader,
queue or allocation in the application.

## Current integration status

IMU/contact/frame and ray-query paths expose device channels with optional
grouped collection. Cameras and image/cloud/grid/annotation processors provide
checked active-range sources for this ring. Legacy camera `download_*` methods
remain synchronous buffer conveniences; use selected sources for grouped
collection. GPU LiDAR, moving scene geometry and simulation-time scheduling
are separate APIs, not capabilities implemented by the staging pool itself.
Public cross-sensor producer composition and direct leased dispatch remain open.

Verification: `make test-gpu P=molla-sensors T=sensor_readback_gpu` and
`make test-gpu P=molla-sensors T=sensor_device_pipeline_gpu`. The full sensor gate
is `make test-sensors`.
