# Contact sensor

`encode` supports [direct leased outputs](device_results.md) and mixed sensor
command batches without collection. Nonempty input arrays and CSR must have
canonical device ownership. Empty-contact frames clear all active body readings;
zero-body frames contain empty channels. Existing `dispatch`/`update` remain available.

`ContactSensor` reduces the per-frame `Contacts` buffer into one
per-body summary: count, max depth, summed normals, mean point. It
mirrors Newton's `SensorContact` and is the primitive that "foot is
on the ground", "robot tipped over", and proximity-shielding
controllers consume.

Source: `crates/molla-sensors/src/contact.rs`. Shader:
`crates/molla-kernels/shaders/contact_aggregate.wgsl` (one GPU thread per
body, walks the contact CSR).

## Reading

```rust
pub struct BodyContactReading {
    pub count: u32,            // number of contacts the body is in
    pub max_depth: Real,       // largest penetration depth (m)
    pub normal_sum: Vec3,      // Σ (sign · normal), points along net push
    pub centroid: Vec3,        // mean contact point, world frame
}
```

The sign in `normal_sum` is flipped when the body is the *a*-side
of a contact, and kept positive on the b-side. This is an unweighted
geometric normal sum, not the direction or magnitude of net force. Zero `count`
contacts ⇒ `max_depth = 0`, both vectors are zero.

## What it does not measure

This sensor reports geometric / kinematic summaries only — it
**does not surface contact forces or λ impulses**. Wiring the
per-contact lambdas from the XPBD velocity-projection pass into a
sensor channel is its own piece of solver work; for the v1 contract
("is body touching anything? where?") the geometric summaries cover
the common cases:

- *Foot in contact?* — `count > 0`.
- *How deep is the overlap?* — `max_depth`, in meters; it does not
  determine force without solver-specific impulse/force information.
- *Which geometric directions oppose overlap?* — `normal_sum`,
  unweighted by actual contact forces.
- *Where are the contacts?* — `centroid`, the mean world-space point.

## API

```rust
use molla::compute::Engine;
use molla::sensors::ContactSensor;

let engine = Engine::new_gpu()?;
let mut sensor = ContactSensor::new(&engine, &model)?;

// After the solver has populated `contacts`:
let readings = sensor.update(&engine, &contacts)?;
for (body_idx, r) in readings.iter().enumerate() {
    if r.count > 0 {
        println!(
            "body {body_idx}: {} contacts, depth={:.4} m, push direction={:?}",
            r.count, r.max_depth, r.normal_sum,
        );
    }
}
```

`ContactSensor::new(engine, model)` allocates one slot per body in
the model — i.e. the output vector always has length
`model.body_count`. Bodies with no contacts this frame land at a
zeroed reading; callers typically filter by `count > 0`.

`dispatch(engine, contacts)` returns `ContactDeviceReadings`: borrowed
`DeviceBuffer` channels named `count`, `max_depth`, `normal_sum` and
`centroid`. It submits GPU aggregation without downloads or CPU waits.
Use each channel's `readback_source(range)` with a caller-owned
[`ReadbackRing`](readback.md) for selected asynchronous collection, or
submit a GPU consumer directly. Submit consumers/copies before the next
dispatch; merely recording an unsubmitted command does not lease outputs.

The synchronous `update()` wraps dispatch and one grouped collection of
40 bytes per body, reusing one staging slot after first use. Its wait is
bounded to 30 seconds; errors are returned rather than hidden. An empty
model returns empty readings without collection. `readback_stats()` counts
only the sensor's own synchronous collection, not external rings.

Inputs must belong to the same engine and have current GPU mirrors. The
contact producer must supply a valid per-body CSR consistent with active
contacts; buffer lengths and authority are checked, but device CSR contents
are not downloaded for CPU validation. `Contacts::clear()` resets outputs
to zero even if old CSR data remains allocated. Recreate the sensor after
body remapping/count changes; raw indices are not stable application handles.
`dispatch_stats()` records kernel dispatches, submissions, clear batches and
binding preparations. The sensor retains two least-recently-used input buffer
variants, including CSR/output identities: alternating two stable producers
does not recreate uniforms or bind groups after warm-up. Replacing a bound
buffer prepares a new set; changing contents in place does not. A clear needs
no bindings. Each preparation creates one uniform and three bind groups.
Command encoders and driver upload staging are not counted as preparations.

`ContactAggregateKernel::prepare` / `encode_prepared` support lower-level
shared command streams without submission or authority bookkeeping. Validate
the CSR/active ranges and mark output buffers device-authoritative yourself.
Submit before changing a reused parameter buffer; distinct pending parameter
sets require separate prepared bindings. Public sensor-level producer batching
and direct dispatch into leased output slots remain pending.

## Performance

The kernel is launched with one thread per body. Each thread walks
that body's slice of the contact CSR
(`contacts.body_contact_offsets[body]..body_contact_offsets[body+1]`) — typically
very short (4–8 contacts for an articulation foot, often 0). The
CSR layout matches the rest of the contact pipeline so no
permutation step is needed.

## Cross-references

- [`Geometry / Narrow phase`](../geometry/narrow_phase.md) — where
  the per-contact tuples come from (GJK / EPA / MPR).
- [`Geometry / Contact reduction`](../geometry/contact_reduction.md)
  — how the raw contacts are deduplicated before reaching the
  sensor.
- [`Sensors / IMU`](./imu.md) — common partner: an accelerometer
  combined with a foot-contact flag is enough to close a basic
  ZMP-style balance loop.
