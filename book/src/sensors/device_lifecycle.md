# Sensor device loss and recovery boundaries

Each GPU `Engine` owns a shared `DeviceHealth` signal. Its wgpu loss callback
records the first `DeviceLoss { reason, message }`; subsequent notifications do
not overwrite that diagnostic. `is_lost`, `loss` and `check` expose the signal.
The healthy path uses an atomic check, not a GPU wait. The signal/callback owns
no GPU resources, so retaining a diagnostic does not retain the device.

```rust
if let Some(health) = engine.device_health() {
    if let Some(loss) = health.loss() {
        eprintln!("sensor device unavailable: {:?}: {}", loss.reason, loss.message);
        // Stop scheduling GPU work and rebuild against a new engine.
    }
}
```

`Engine::check_device()` fails for a CPU engine or an observed lost GPU device.
A CPU engine has no `DeviceHealth`. Raw device/queue getters keep their existing
handle-returning semantics; a returned handle alone is not proof of health.

## Explicit shutdown versus backend loss

`Engine::destroy_device()` marks its own logical device lost immediately and
then calls wgpu destruction. It is terminal and idempotent. It is not a reset
of the physical GPU, driver, or other applications. Do not call it merely to
pause a sensor or reset a scene: use the respective scheduler/sensor reset APIs.

Backend loss uses wgpu's callback. Callback delivery can require device polling;
successful polling by itself is not a health assertion. Sensor readback and
snapshot pools check health both before and after their nonblocking polls.
External code using raw `device.destroy()` should poll to observe callback
delivery, or use the engine method for immediate application-side rejection.

The engine owns the single wgpu device-loss callback. Do not replace it through
the raw device handle; observe `DeviceHealth` instead. Concurrent/raw wgpu work
can still fail between a health check and submission. These checks do not
provide a transaction across arbitrary external API calls, intercept all wgpu
validation errors, detect a driver hang before notification, or retrofit loss
handling into every physics/renderer kernel in the workspace.

## Sensor behavior after observed loss

- `ReadbackRing` becomes terminal: no available slots or new requests. Polling
  an outstanding ticket returns an error rather than reading destroyed GPU
  storage—even a zero-byte request does not masquerade as a successful sample.
  Reset/cancellation cannot revive the device. Mapping failure remains a
  separately reported terminal condition; ordinary finite wait timeout remains
  retryable/cancellable.
- `DeviceSnapshotPool` refuses capture, availability checks and reset; old
  snapshots are not current. Their identity/shape metadata remains inspectable,
  but `SnapshotChannel::binding` and `readback_source` reject GPU access.
- `DeviceCommandBatch` rejects creation, encoding and submission on a known
  lost device. An already encoded but unsubmitted batch cannot be submitted
  after loss. Dropping it releases its unsubmitted commands/leases.
- `DeviceBuffer` refuses creation, upload and GPU access. Synchronized host
  values and previously collected `ReadbackFrame` data remain CPU-readable.
- Guarded sensor entry points reject known loss before GPU use: ray queries,
  LiDAR batches, tiled/grouped cameras, IMU/contact/frame dispatch, scene
  construction/replacement, and image/cloud/grid/annotation processing.
  Camera legacy download helpers also check engine health. Metadata-only
  operations are not automatic recovery operations.

Failure handling does not unmap a pending transfer prematurely. Dropping a ring
invalidates callback serials and releases its handles without a blocking wait;
callbacks own completion state, not the entire ring. Snapshot/batch completion
callbacks retain occupancy tokens, not owning device references.

## Recovery is explicit reconstruction

1. Stop accepting work for the failed engine and invalidate application sample
   generations. Record the first-loss diagnostic.
2. Drop/retire old producers, pending tickets, snapshot leases and command
   batches. They cannot be rebound to another logical device.
3. Create a fresh `Engine` when the backend is available, then recreate scenes,
   sensors, processing and readback/snapshot resources from authoritative
   configuration/data. GPU-only history/state must be regenerated; it is not
   recovered from a loss flag.
4. Prime history-based sensors such as IMU and restart sample generations before
   accepting fresh results. Previously collected host data can remain archival.

Resetting an existing failed pool is not step 3. Old sources/handles are rejected
by fresh pools because exact device identity differs. There is no automatic
adapter failover, GPU-state recovery or physical driver-restart mechanism here.

## Evidence and limits

`sensor_device_loss_gpu` uses isolated test engines. It verifies engine-owned
destruction with outstanding/cancelled readbacks and an unsubmitted leased
batch; raw wgpu destruction delivered through the actual callback; rejection by
all guarded producer families; old-handle rejection and successful collection
on a newly created engine; and a retained diagnostic not retaining the device.
Already collected CPU values remain unchanged. Host tests cover CPU-engine
behavior and sticky first-loss diagnostics across threads. Existing readback
tests also inject mapping failure and exercise timeout retryability.

```sh
make test-gpu P=molla-sensors T=sensor_device_loss_gpu A='--test-threads=1'
make test APP_TARGET='-p molla-compute --features gpu --lib'
make test-sensors
```

These are real logical-device destruction/callback tests, not physical GPU
removal, forced driver faults or GPU-hang recovery tests. No user application or
physical device is reset by the fixtures. Public producer composition, direct
dispatch into leased output slots and full warmed-allocation/performance gates
are separate unfinished S02 work.
