# GPU class and instance annotation

`AnnotationProcessor` consumes per-pixel shape IDs directly from device camera
outputs or copied snapshots. It produces full-width class masks, shape-instance
masks, counts and compacted bounding boxes without downloading input images.
`Dataset` remains the CPU reference and label-building convenience API.
Direct `encode`/`encode_pending` entry points also support leased outputs and
composition with earlier sensor results in one `SensorCommandBatch`.

```rust
use molla_sensors::{AnnotationConfig, AnnotationJob, AnnotationProcessor, Dataset};

let mut labels = Dataset::new();
labels.label_shapes(&[sphere_shape_id], 10);
let mut processor = AnnotationProcessor::new(&engine, AnnotationConfig::default())?;
processor.set_labels(&labels.shape_labels())?;

let cameras_frame = cameras.dispatch_scene(&engine, &scene, &state)?;
let image = cameras_frame.image(camera_id).unwrap();
let jobs = [AnnotationJob::from_camera(&image, Some(simulation_time))?];
let annotations = processor.process(&jobs)?;
let pending_count = ring.try_enqueue(
    &[annotations.box_count_source(0)?],
    annotations.layouts()[0].stamp,
)?;
```

`AnnotationJob::from_camera` requires a rendered `SHAPE_INDEX` channel and
preserves the camera's u64 sensor/sample identity and geometry revision. Raw
jobs accept a checked `ReadbackSource` containing u32 elements, dimensions and
caller-provided stamp; sources are runtime type/device checked. Merely creating
a source does not read it back. Different image sizes and source offsets are
supported in one batch. Sensor IDs must be unique within that batch.

## Labels and instance meaning

- A `ShapeLabel` assigns one u32 class to one u32 shape ID. The sorted lookup
  table is sparse: a large shape ID does not require a correspondingly large
  dense allocation. `u32::MAX` is reserved for missed rays and cannot be labelled.
- `set_labels` validates a complete replacement before queuing the upload.
  Duplicate shape IDs are rejected rather than choosing an arbitrary winner.
  Its checked monotonic revision advances on every successful replacement,
  including an empty table. Invalid replacements leave the old table/revision
  intact. `Dataset::shape_labels` copies its existing mapping in sorted order;
  a reserved ID authored through legacy `Dataset` is not silently discarded.
- One shape is one instance, matching the existing CPU dataset contract. Shapes
  sharing a class still have separate boxes. There is no automatic grouping of
  multiple shapes into a machine/body-level instance.
- Misses and unlabelled shapes have class 0. An explicitly labelled class 0
  still produces a box and contributes to labelled counts. Class-mask value 0
  alone therefore does not distinguish these cases.
- `shape_source` is the full-width instance mask: original shape IDs, including
  unlabelled ones, and `u32::MAX` for misses. No lossy `shape + 1` remapping occurs
  on the GPU. Keep scene identity with these model-local IDs; label revisions
  do not establish cross-scene or application-stable identity.

## Bounds, counts and active ranges

`DeviceBoundingBox` is a 32-byte record containing inclusive pixel bounds
`[x_min, y_min, x_max, y_max]`, shape ID, class ID, pixel count and reserved zero.
Boxes are compacted independently for each image and sorted by shape ID.
`boxes_source(image, range)` is relative to that image's label-capacity extent;
only the visible-count prefix is valid. Unused active slots are zeroed. There
is capacity for every labelled instance, not a smaller silently truncating cap.

`counts_source(image)` returns four u32 values decoded by
`AnnotationCounts::from_words`: labelled pixels, unlabelled non-miss pixels,
missed pixels and visible instances. The first three partition image pixels.
`box_count_source` selects only the final word. `class_source`, `shape_source`
and `boxes_source` accept image-relative checked ranges. Inactive images and
out-of-range tails are inaccessible through a frame.

A 0×0 job is allowed and produces zero counts/boxes. Exactly one zero dimension
is rejected. An empty job list submits a count reset but has no image outputs.
An empty label table still maps image pixels to class 0 and reports misses and
unlabelled counts; its box extent is empty. Mixed empty/nonempty image jobs do
not change other cameras' pixel coordinates or reduction slots.

The three retained passes initialize per-image label slots, map pixels and
atomically reduce bounds, then compact visible instances with a parallel
integer prefix. Count/bounds/class output is exact and ordering deterministic.
One submission covers the whole batch. There is no implicit CPU image array,
blocking wait or readback inside `process`.

## Ownership and selected export

Legacy `process` frames borrow the processor. Enqueue copies or capture snapshots before its
next mutable use. Snapshot inputs pin their occupancy through submission
completion even after public snapshot drop or pool reset. Queued historical
readbacks preserve their original outputs across relabelling and processing.
Keep `label_revision()` beside each request/snapshot as well as its image stamp;
the generic readback stamp does not contain the label revision.

After explicitly collecting a box count, collect only its prefix:

```rust
use molla_sensors::DeviceBoundingBox;

let ticket = ring.try_enqueue(
    &[annotations.boxes_source(0, 0..visible_count)?],
    annotations.layouts()[0].stamp,
)?;
// After explicitly polling/waiting for this ticket:
let packed = collected.channel::<DeviceBoundingBox>(0)?;
let boxes: Vec<_> = packed.into_iter().filter_map(DeviceBoundingBox::to_cpu).collect();
molla_sensors::dataset::write_bbox_json(&boxes, &mut json_writer)?;
```

This path needs no shape/class image collection. Conversely, mask-only export
can collect only `class_source` or `shape_source` and call the CPU helpers
`dataset::write_class_image_pgm` / `write_instance_image_pgm`. All serialization
and `Write`/filesystem operations stay on CPU; no file path or I/O is embedded
in the GPU processor. Helpers validate dimensions/bounds before writing and
propagate I/O errors. Existing `AnnotationFrame` writers delegate to them.

PGM helpers preserve the legacy **lossy 8-bit** format: class IDs clamp to 255;
instance masks use miss=0 and `min(shape_id + 1, 255)`. Do not use that format
to preserve arbitrary instance/class identity. Device outputs and typed
readback preserve all 32 bits, and bounding-box JSON preserves integer IDs.
The processor does not truncate u64 camera IDs to legacy `AnnotationFrame`'s
u32 `camera_id`; selected-output writers do not require that legacy frame.

## Direct outputs and pending images

Create a pool once with `processor.output_pool(&engine, config)`. Its four
channels are:

| Channel | Type | Active range |
|---|---|---|
| 0 | `u32` shape IDs | `0..total_pixels` |
| 1 | `u32` class IDs | `0..total_pixels` |
| 2 | `DeviceBoundingBox` | `0..image_count * current_label_count` |
| 3 | `u32` counters | `0..image_count * 4` |

`encode(batch, jobs, pool, stamp)` accepts existing `AnnotationJob` device
sources. `encode_pending` accepts `PendingAnnotationJob` entries with an earlier
receipt, channel index, pixel range, width/height and independent image stamp.
Pixel ranges are **relative to the source channel's active range**, not raw buffer
indices. The selected u32 element count must equal checked `width * height`.

Both return `Result<Option<EncodedAnnotations>>`. `None` means backpressure,
with no clears, copies, descriptor uploads or binding preparation. A successful
receipt carries `frame`, per-image `layouts`, and the actual `label_revision`
used for that encoding. Label revisions are processor-local: different or recreated
processors can have the same revision number. The frame-level `ReadbackStamp` is
separate caller metadata, not a substitute for label or scene identity. Earlier encoded frames
can feed new annotation stages with independent label tables:

```rust
let pending: Vec<_> = earlier.layouts.iter().map(|layout| PendingAnnotationJob {
    frame: &earlier.frame,
    channel: 0,
    pixels: layout.pixels.clone(),
    width: layout.width,
    height: layout.height,
    stamp: layout.stamp,
}).collect();
let result = processor.encode_pending(&mut batch, &pending, &mut pool, frame_stamp)?;
let submitted = batch.submit()?;
if let Some(result) = result {
    let frame = submitted.frame(&result.frame)?;
    let counts = frame.channel::<u32>(3)?;
    // Queue selected image counts before collecting box prefixes.
}
```

Input shape ranges copy directly into the leased shape channel; the same three
annotation kernels write classes, boxes and counters directly into that slot.
All output channels are cleared before use. There is no copy of legacy outputs
into a snapshot and no internal image download or submission. An empty direct
job list still reserves a frame, clears its channels and exposes empty active
ranges. Nonempty 0×0 jobs have zero counters and zeroed box extents; with no labels
they dispatch no kernels. Nonempty pixels with no labels still run the mapping
pass to report misses and unlabelled counts.

The processor is borrowed mutably until batch submission/drop, preventing both
relabeling and descriptor/uniform reuse while commands are pending. Source jobs
passed to `encode` are borrowed through submission; their occupancy tokens are
retained through completion. Compile-fail tests protect processor reuse and
source mutation. Raw cloned wgpu handles remain caller-synchronized resources.
After submission, labels may change while historical frame leases remain valid:
their output ranges, data and recorded label revisions do not change.

External source jobs share the batch's **64-input-reference** budget with other
device-source producers and submitted-frame consumers. `encode` therefore accepts
at most the remaining budget, even when the processor supports 1,024 images.
This is an explicit error, not truncation or implicit extra submissions. The
legacy `process` API retains its configured image capacity. For a large composed
pipeline, use aggregate pending shape channels and describe their per-image ranges.

`encode_pending` preserves the full configured **1,024-image** capacity. Repeated
dependencies are deduplicated by batch-owner identity and output index, including
cloned receipts. Logical image count is not treated as dependency count: a test
processes 1,024 independent images from one pending channel with a nonzero active
offset. Distinct dependencies are bounded by the batch's existing 64-output limit;
annotation output itself also needs a free output entry.

Invalid sources, receipts, dimensions, capacities, IDs and times fail before
reservation and leave earlier valid commands usable. After encoding begins,
errors poison the enclosing batch. Dropping it publishes no result. Queued
descriptors from cancelled work may flush later, but subsequent ordered parameter
writes and checked active counts prevent stale descriptors from being used.
Pool reset changes generation without reclaiming pending/historical slots.
Known device loss rejects further encoding and readable channels.

Each slot's byte budget is
`8 * max_pixels + 32 * max_images * max_labels + 16 * max_images`, separate from
the processor's original eight buffers. Direct bindings reuse the label table,
descriptors, scratch boxes and uniform buffer, with a 128-entry output-binding
cache. Cached groups retain their buffers until eviction or processor drop;
pool budgets are not a global memory limit across replaced pools.

`encoded_frames` counts successful direct encodes, even if their batches are later
cancelled. `frames`/`submissions` describe legacy `process`; copy/kernel counters
include direct encoded work, not necessarily completed work. Batch statistics
record the shared submission. Descriptor and uniform writes are control metadata,
not host-uploaded shape images.

The four-slot warm fixture changes label values and active pixel counts over 24
frames with four direct binding preparations, 16 fixed pool buffers and the
processor's original eight buffers. CPU descriptor/layout/dependency vectors,
validation sets, command encoders and driver memory still have separate allocation
behavior. This does not close S02's whole-pipeline allocation/performance gate.

## Budgets and measurement

Defaults are 4,000,000 total pixels, 16 images, 1,024 labels and 64 MiB. At most
1,024 images are accepted, with additional checked device/dispatch limits.
Eight retained GPU buffers use exactly

`8 * max_pixels + 64 * max_images * max_labels + 8 * max_labels + 32 * max_images + 16`

bytes. This includes shape/class pixels, working/output boxes, labels, image
descriptors, counts and uniforms, but excludes host mirrors, per-call metadata,
driver staging and pipelines. The image×label product and all budgets are
validated before allocation. Statistics track retained resources, submissions,
dispatches, copied bytes and label uploads; no driver allocation-free claim is
implied. Label updates upload metadata; they do not rebuild pipelines/bindings.

```sh
make test-gpu P=molla-sensors T=sensor_annotation_gpu A='--test-threads=1'
make test-gpu P=molla-sensors T=dataset_export_gpu A='--test-threads=1'
make bench BENCH_ARGS='-p molla-sensors --features gpu --bench bench_sensor_annotations'
```

The benchmark covers 1,024 / 65,536 / 1,000,000 total pixels, one/four cameras,
and one dense or 257 distributed instances. CPU `Dataset::annotate`, completed
GPU work and GPU plus selected-box collection are separate lanes. GPU work
includes the input copy and three passes; label upload/initial allocation are
outside timing. The CPU API allocates host outputs; selected-box collection
explicitly collects count then prefix for each camera. Binary label lookup and
contended dense-instance atomics remain costs to measure. Smoke success is not
performance parity or a completed workload/crossover report.
