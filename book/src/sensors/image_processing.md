# GPU image processing

`ImageProcessor` executes independent packed RGBA8 pipelines on the existing
sensor GPU. Camera color/albedo slices, copied snapshot channels and other
`DeviceBuffer<u32>` sources can feed it directly. No image download, host pixel
array or blocking wait occurs inside `process`.

## Operations and numerical contract

`ImageOp` supports exposure, exponent gamma, linear/sRGB encoding and decoding,
Reinhard tone mapping with or without a white point, brightness/contrast,
saturation, gray-world white balance and white-patch white balance.
Exposure is a linear multiplier, not exposure stops: `Exposure(2.0)` doubles RGB.
The corresponding `camera_post` CPU functions remain reference/convenience APIs,
including `apply_srgb_to_linear` for the reverse sRGB conversion.

Pixels are `0xAABBGGRR`. Each stage unpacks normalized RGB, evaluates its formula,
clamps and rounds back to RGBA8. Alpha is preserved exactly, including transparent
pixels; alpha does not mask RGB adjustments or exclude pixels from reductions.
Color space is caller-controlled: camera output is linear, and conversions occur
only when explicitly listed. This does not create HDR values from already-clipped
8-bit camera color. Operation order and intermediate quantization are observable.

Gamma and white point use the CPU API's `max(value, 1e-6)` semantics. Nonfinite
parameters are rejected, as are brightness/contrast combinations whose RGB bound
can overflow. WGSL uses f32 arithmetic and hardware transcendental functions;
bit-identical CPU f64 white-balance sums are not promised. Tests compare individual
operations within one RGB code value and the tested seven-stage chain within two;
other chains can amplify rounding. Packing uses half-up rounding rather than WGSL
ties-to-even. Explicit gamma endpoints preserve black/white even when an extreme
finite gamma's reciprocal becomes subnormal on the GPU.

White balance runs a parallel 256-thread partial reduction, a second parallel
reduction of those partials, then a per-pixel application pass. Each image and
each white-balance stage gets independent gains; no CPU reduction or global
cross-camera color average is used. Empty gray-world input returns `[1,1,1]`;
empty white-patch input returns `[1e6,1e6,1e6]`. A black nonempty image follows
the CPU formulas: gray-world gains are zero, white-patch gains are `1e6`.

## Camera-to-device pipeline

```rust
use molla_sensors::{ImageJob, ImageOp, ImageProcessor, ImageProcessorConfig};

let mut processor = ImageProcessor::new(&engine, ImageProcessorConfig::default())?;
let frame = cameras.dispatch_scene(&engine, &scene, &state)?;
let camera = frame.image(sensor_id).expect("configured camera");
let scene_stamp = camera.scene_stamp(CameraChannels::COLOR)?;
let receipt = frame.stamp().clone();
let operations = [ImageOp::Exposure(0.75), ImageOp::GrayWorld, ImageOp::LinearToSrgb];
let jobs = [ImageJob {
    source: camera.readback_source(CameraChannels::COLOR)?,
    operations: &operations,
    stamp: camera.readback_stamp(CameraChannels::COLOR, Some(simulation_time))?,
}];
let processed = processor.process(&jobs)?;
let image = processed.image(sensor_id).expect("submitted image");
let pending = ring.try_enqueue(
    &[image.readback_source()?, image.white_balance_source(1)?],
    image.layout().stamp,
)?;
```

The source method's name refers to the checked device range, not to a performed
download. Only the optional ring performs GPU-to-host collection. Pass processed
sources to another `ImageProcessor`, or capture a `DeviceSnapshotPool` copy for
completion-tracked GPU consumers. Retain the original full scene stamp and camera
receipt alongside asynchronous tickets when resolving IDs against a live scene.
`ReadbackStamp` preserves sensor/sample/revision/time metadata but does not contain
full scene identity or camera generation.

Every job has an independent source range, operation list and stamp. Sensor IDs
must be unique in a submitted batch. The processor does not invent sample IDs,
timestamps or geometry revisions. Flat output pixel ranges preserve job order;
width/height remain in the caller's camera metadata. White-balance stage indices
are local to that image's operation list. Requesting gains for a different stage
is an error, not a read of old contents.

## Ownership, batching and budgets

The entire job list is validated before encoding. Foreign-device or non-u32
sources, excess images/operations/pixels, duplicate IDs and invalid timestamps
fail without submitting a partial batch. Every active input range is copied
GPU-to-GPU into retained working storage; the source is never modified. Copies
and all image stages share one submission in `process`. The direct APIs below
instead append those copies and stages to an external sensor batch.

Processed views borrow the processor, preventing mutable reuse while those views
are used. Enqueue collection or capture a snapshot before releasing them. Queued
copies survive later frames. Raw external consumers must submit before reuse or
hold copied snapshot leases. Snapshot input leases are retained through submitted
work completion even if their pool is reset and their public handles are dropped.
The direct APIs instead return independently leased output frames.

With `process`, empty batches and empty images without white-balance operations submit no GPU
work. Empty white-balance operations submit gain calculations but no pixel pass.
A nonempty image with no operations performs an exact device copy. Unused capacity
is not exposed as output; skipped/non-white-balance stages cannot expose old gains.

Defaults: 4,000,000 aggregate pixels, 16 images, 16 operations per image and
32 MiB of application GPU buffers. Configurable hard limits are 64 images and
64 operations per image. Device storage/workgroup limits and checked total bytes
are enforced before allocations. Four retained buffers are counted:

- RGBA pixels: `4 * max_pixels` bytes.
- Reduction partials: `16 * ceil(max_pixels / 256)` bytes.
- Per-stage gains: `16 * max_images * max_operations_per_image` bytes.
- Parameters: one 32-byte uniform per stage, rounded up to device uniform alignment.

One retained bind group and three pipelines serve all operations. The budget and
statistics exclude CPU metadata, pipeline/driver memory and upload staging. Job
metadata and lease vectors are bounded but can allocate per frame. Dimensions or
budgets are fixed at construction; create another processor to change capacity.

## Direct leased output and pending-input composition

Create a fixed-budget pool with `processor.output_pool(&engine, config)`, then use
one of these APIs with a `SensorCommandBatch`:

- `encode(batch, jobs, pool, stamp)` takes the existing `ImageJob` device sources:
  camera channels, snapshots, direct frames or `DeviceBuffer<u32>` ranges.
- `encode_pending(batch, jobs, pool, stamp)` takes `PendingImageJob` descriptions
  pointing to earlier receipts in that same batch. Each specifies the receipt,
  channel index, pixel range **relative to that channel's active range**, operation
  list and independent image stamp. No source submission or readback is needed.

Both return `Result<Option<EncodedProcessedImages>>`: `None` means pool
backpressure, with no clears, copies, parameter writes or binding preparation.
Successful encoding returns an opaque `frame` receipt and `layouts` in job order.
The frame-level stamp is caller metadata independent of each image's stamp.
Channels are:

| Channel | Type | Active range |
|---|---|---|
| 0 | `u32` packed RGBA8 | `0..total_pixels`; layouts select each image |
| 1 | `Vec3` stage gains | `0..total_operations`; layouts select each stage |

Only the stages listed in `white_balance_stages` contain white-balance gains.
Other active gain entries are zeroed, not stale values from a previous frame.
Each output slot allocates `4 * max_pixels + 16 * max_images * max_operations_per_image`
bytes, separately budgeted from the processor's four retained buffers. Pixels
are copied directly into the leased working output, then adjusted in place;
there is no legacy-output-to-snapshot copy. Sources are not modified.

Two distinct processors can form a complete image chain before submission:

```rust
let first = a.encode(&mut batch, &device_jobs, &mut pool_a, frame_stamp)?;
if let Some(first) = first {
    let jobs: Vec<_> = first.layouts.iter().map(|layout| PendingImageJob {
        frame: &first.frame,
        channel: 0,
        pixels: layout.pixels.clone(),
        operations: &adjustments,
        stamp: layout.stamp,
    }).collect();
    let second = b.encode_pending(&mut batch, &jobs, &mut pool_b, frame_stamp)?;
    let submitted = batch.submit()?;
    if let Some(second) = second {
        let pixels = submitted.frame(&second.frame)?.channel::<u32>(0)?;
        // Enqueue only the image ranges needed by the caller.
    }
}
```

If the second pool is full, the caller chooses whether to submit the first stage
or drop the entire batch. A pending empty job list still reserves a valid frame
with two empty active channels and encodes output clears, unlike legacy `process`'s
no-submission empty case. Empty white-balance jobs preserve the numerical behavior
above. Nonempty images without operations perform only an exact GPU copy.

The processor is mutably borrowed until batch submission/drop, protecting its
shared uniform-stage bank and reduction scratch against pending reuse. Distinct
pending processors have distinct parameter storage. Device-source jobs and their
input buffers are also borrowed until submission; compile-fail tests cover both
processor reuse and input mutation. Source occupancy tokens are retained through
GPU completion even after the jobs and snapshots are dropped. External device
sources share the batch's 64-input-reference budget with submitted-frame consumers;
pending jobs refer to already-owned outputs and do not consume that budget.

Whole-batch validation precedes reservation: source types/devices, receipt identity,
pixel ranges, unique IDs, operations, capacities and finite nonnegative image times
are checked. Invalid jobs leave earlier batch work usable. Once encoding starts,
errors poison the enclosing batch. Dropped batches publish nothing; queued uniform
writes may flush later, but ordered replacement parameters prevent stale-stage use.
Pool reset preserves historical output leases. Known device loss rejects encoding
and collection; it does not silently recreate a device.

Direct bindings share the existing scratch, uniform bank and three pipelines and
use a bounded 128-entry output-binding cache. `binding_preparations` includes the
initial legacy binding plus direct cache misses. `encoded_frames` counts successful
direct encodes even if their enclosing batch is later dropped; `frames` and
`submissions` count legacy `process` calls/submissions. Copy and kernel counters
include direct encoded work, not necessarily completed GPU work. Batch statistics
report the shared submission separately. No image input payload is uploaded by
these APIs; uniform parameter writes are not counted as image uploads.
Cached groups retain their output buffers until eviction or processor drop, even
if the corresponding pool has been dropped; pool budgets are not a global cache
memory budget across arbitrarily many replaced pools.

The four-slot warm fixture cycles changing counts and exposure settings over 24
frames with four direct preparations, eight fixed pool buffers and the processor's
original four buffers. This is not a proof of whole-pipeline zero allocation:
layout vectors, validation sets, batch metadata, wgpu and driver resources remain
outside that claim. LiDAR/camera and cloud construction/filtering direct migrations are
still separate unfinished S02 work.

## Verification and performance

```bash
make test-gpu P=molla-sensors T=sensor_processing_gpu A='--test-threads=1'
make test-gpu P=molla-sensors T=sensor_image_direct_gpu A='--test-threads=1'
make test-sensors
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_images -- --test'
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_images'
```

Fixtures cover scalar CPU references, exact alpha and no-op copies, unaligned
input subranges, independent image reductions, empty/black/zero-channel inputs,
256-thread and second-reduction boundaries, the final slot at 4,096-slot capacity, pending
frame reuse, validation, and heterogeneous camera → processing → snapshot →
processing chains. Host tests cover parameter semantics and exact buffer budgets.

The benchmark matrix covers all ten operation kinds at 1,024, 65,536 and 1,048,576
pixels, with separate CPU, completed-GPU and GPU-plus-host-collection lanes. CPU
timing includes copying into retained working memory; GPU timing includes the
device input copy, host orchestration, dispatch and completion wait. Initial GPU
upload is outside timed iterations. The host-collection lane additionally includes
staging, map and host decoding. It is not an upload-every-frame comparison.
`--test` is a smoke run, not performance evidence. Uncontended release measurements
and numerical/performance acceptance across other adapters remain outstanding.

GPU point-cloud operations and annotation reductions are separate, unfinished S07
work. File serialization remains on the CPU.
