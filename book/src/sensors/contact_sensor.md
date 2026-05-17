# Contact sensor

`ContactSensor` reduces the per-frame `Contacts` buffer into one
per-body summary: count, max depth, summed normals, mean point. It
mirrors Newton's `SensorContact` and is the primitive that "foot is
on the ground", "robot tipped over", and proximity-shielding
controllers consume.

Source: `crates/molla/src/sensors/contact.rs`. Shader:
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

The sign in `normal_sum` is flipped when the body is the *b*-side
of a contact, so the resulting vector always points along the
direction the contact force is pushing *this* body. Zero `count`
contacts ⇒ `max_depth = 0`, both vectors are zero.

## What it does not measure

This sensor reports geometric / kinematic summaries only — it
**does not surface contact forces or λ impulses**. Wiring the
per-contact lambdas from the XPBD velocity-projection pass into a
sensor channel is its own piece of solver work; for the v1 contract
("is body touching anything? where?") the geometric summaries cover
the common cases:

- *Foot in contact?* — `count > 0`.
- *Foot pressed how hard?* — `max_depth` is a proxy for normal
  force, since XPBD penetration depth scales linearly with reaction
  impulse via the compliance term `α = 1/k·dt²`.
- *Which direction is the body being pushed?* — `normal_sum`
  (un-normalised, so its magnitude reflects the spread + count).
- *Where, on the body, is the push concentrated?* — `centroid`.

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

The sensor is reusable across many steps. The output buffers stay
on the device between calls and only get downloaded inside
`update()`; if you want to consume the readings device-side, pull
the buffers off `ContactSensor` (Stage 18.1 internal API) instead
of paying the readback cost.

## Performance

The kernel is launched with one thread per body. Each thread walks
that body's slice of the contact CSR
(`contacts.body_offsets[body]..body_offsets[body+1]`) — typically
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
