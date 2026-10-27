# Simulation-time sensor scheduling

`SensorScheduler` issues bounded sample requests for independently configured
sensor periods and phases. It is available without the `gpu` feature: scheduling
is small host-side control work, while sensor evaluation remains on the GPU.
Source: `crates/molla-sensors/src/scheduler.rs`.

## Time and sampling contract

- Pass the **simulation snapshot timestamp** after every physics step. Render
  frame count and wall-clock time are not scheduler inputs. A renderer may read
  cached results at any rate without changing sensor cadence.
- Times, periods and phases use `std::time::Duration`, with nanosecond resolution.
  Choose representable periods explicitly (100 Hz = 10 ms). For a fractional
  physics timestep, convert the cumulative simulation timestamp, not a rounded
  per-step duration repeatedly added to itself. `Duration::try_from_secs_f64`
  can validate/round external floating-point timestamps; such conversion is
  separate from the scheduler's exact integer arithmetic.
- The first deadline is `epoch + phase`; zero phase requests a sample at the
  epoch. Later deadlines remain anchored to that epoch, not to late arrivals.
- A late call requests **one sample of the current state**, with both the
  actual `simulation_time` and latest elapsed `scheduled_time`. Earlier missed
  deadlines are counted in `skipped_samples`, not replayed against today's state.
  No interpolation or historical physics state is invented.
- Repeated calls at the same timestamp do not duplicate delivered requests.
  Backward time returns an error until an explicit reset. Pausing suppresses
  delivery without shifting deadlines: resuming at unchanged simulation time
  has no extra elapsed interval; if time advanced while delivery was paused,
  the next request reports skipped deadlines.

## Independent rates

```rust
use std::time::Duration;
use molla_sensors::{SensorSchedule, SensorScheduler};

let schedules = [
    SensorSchedule {
        sensor_id: 10,
        period: Duration::from_millis(10),
        phase: Duration::ZERO,
    },
    SensorSchedule {
        sensor_id: 20,
        period: Duration::from_millis(50),
        phase: Duration::from_millis(5),
    },
];
let mut scheduler = SensorScheduler::new(Duration::ZERO, 1, &schedules)?;
for sample in scheduler.advance(Duration::from_millis(100))? {
    // Dispatch this sensor against the snapshot at 100 ms.
    println!("{}: skipped {} deadlines", sample.sensor_id, sample.skipped_samples);
}
# Ok::<(), molla_core::Error>(())
```

Sensor IDs must be unique; zero periods and overflowing epochs are rejected.
There are at most 4096 configured sensors and at most one request per sensor per
advance. Request order matches configuration order. Warm advances reuse their
batch allocation, even after large time jumps; there is no growing catch-up queue.
Counter exhaustion returns an error without consuming any sensors' cursors.

## Evaluation and optional collection

The scheduler emits requests, not readings. The caller maps each ID to its
sensor configuration/batch and calls its device-only dispatch method. Keep
separate batches for sensors with different rates; do not collapse independent
mount settings to fit one batch. IMU priming still yields no reading on its
first dispatch. With successful dispatch on each request, use
`sample.elapsed.unwrap_or(sample.period).as_secs_f64()` as its elapsed interval.

`elapsed` measures time since the preceding **issued request**. If a dispatch
fails or is deliberately skipped, maintain the time of the last successful IMU
history update yourself, or reset/re-prime the IMU before resuming. Collection
backpressure does not undo a successful history update. `skipped_samples` counts
missed deadlines only; dispatch failures and rejected collections need separate
application counters. Issued requests are not retried automatically.

With GPU support, `sample.readback_stamp()` supplies a `ReadbackRing` stamp.
Keep a clone of the `ScheduledSample` beside each pending ticket. On completion:

```text
scheduler.is_current(sample) && sample.matches_readback(frame)
```

Both checks are necessary: `matches_readback` identifies the collected sample;
`is_current` rejects other schedulers and pre-reset/replacement generations.
It does not assert that a result is the most recent sample. Sample IDs never
repeat during a scheduler's lifetime, even across resets with the same scene
revision. The readback stamp has a floating-point time for compatibility; the
retained sample carries its exact integer time. Submit consumers/copies before
reusing sensor output buffers, and honor readback-ring backpressure explicitly.

## Reset, replacement and snapshot consistency

`reset(epoch, scene_revision)` resets cadence and request elapsed times and
invalidates all prior-generation samples. `replace(...)` also atomically replaces
the schedule set, allowing rate/mount registration changes. Neither method
resets sensors or cancels GPU work automatically. On a scene reset/replacement:

1. Stop issuing work for the old snapshot.
2. Reset/replace the scheduler and reset IMU history. Recreate fixed-index
   sensors if bodies were replaced/remapped; replace owned geometry as needed.
3. Cancel obsolete readback tickets or collect and reject them using the
   generation checks. Resetting a ring does not immediately free mapping slots;
   poll its release progress before reuse.
4. Resume with a complete new snapshot. Paused status is preserved by reset
   and replacement until explicitly changed.

For **CPU physics**, copy a coherent model/state/contact snapshot at the sample
timestamp under the producer's ownership boundary, then upload changed inputs
once. Do not let physics mutate only half that snapshot while sensors read it;
release producer locks before any GPU wait. CPU snapshot creation/thread locking
belongs to the caller, not the scheduler.

For **GPU physics**, submit all sensors for a timestamp after that physics step
and before submitting a step which overwrites its buffers, on the same engine's
ordered queue. The scheduler neither downloads state nor makes an extra copy of
physics buffers. These ordering rules apply equally to all due sensors.

## Verification

```sh
make test APP_TARGET='-p molla-sensors --no-default-features'
make test-gpu P=molla-sensors T=sensor_scheduler_gpu A='--test-threads=1'
```

Host tests cover phase/rate arithmetic, late samples, pause, resets, atomic
errors, large epochs, counter exhaustion and retained batch storage. GPU tests
schedule IMU, frames and contacts at independent rates after real GPU integration,
compare readings across render cadences, and reject pre-reset pending results
without hidden sensor readback. This is not viewer integration or a physics
snapshot adapter for Gearbox.
