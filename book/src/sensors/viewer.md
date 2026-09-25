# Viewer sensor integration

## Current migration status

| Example | Current path | Rendered inspection on software Vulkan |
|---|---|---|
| 05 LiDAR | GPU configurable layout, shared scene, bounded asynchronous ray display | Ray fan, paused heading edit and reset/single-step inspected |
| 06 IMU | Direct leased outputs, bounded asynchronous display | Priming, single-step measurement and scrolled accelerometer/gyro readouts inspected |
| 07 Contact | Direct leased outputs, bounded asynchronous display | Per-body readouts, visible normal-overlay arrows, reset clearing and first-step empty contacts inspected |
| 12 Camera | Shared scene, leased camera-to-image GPU chain, bounded asynchronous thumbnails | All five channels, FOV, physical optics, aperture samples, post-processing, reset and pane reopening inspected |
| 105 LiDAR v2 | GPU layout, shared scene, direct leased outputs and asynchronous selected collection | First-step point cloud, fixed settings and reset clearing inspected |
| 106 Aerial LiDAR | GPU aerial layout, shared scene, one-shot asynchronous selected collection | Origins/cloud at entry, no rescan on Step and reset rescan inspected |
| 108 Synthetic Annotation | Device-only camera/annotation chain, asynchronous box-only display | Three detections, pixel-bound readouts and world markers inspected |
| 144 Shared-scene sensor lab | Shared moving-mesh queries, independently scheduled LiDAR/camera/IMU, asynchronous displays | Moving occlusion, readouts, scrolling, pause/reset/step, settings and pane reopening inspected |

These are basic isolated llvmpipe/Xvfb inspections, not complete interaction or
hardware-renderer acceptance. Neither the library regression count nor the
headless viewer tests establish visual acceptance of these examples. The remaining
checks are listed at the end of this chapter.

## Playback and pane controls

Play/Pause, Step and Reset occupy three equal-width columns. Step is visible but
disabled while playing. A paused single-step advances simulation time; periodic
sensors submit only when their own sampling rules require it. Reset and scene
switch clear ordinary physics counters even when paused; pausing alone retains
the previous counters. This changes telemetry, not solver inputs or stepping.

All eight sensor panes have bounded vertical scrolling and scene-local scroll
positions. Their content has a 12-pixel inset to keep checkboxes and slider ends
outside Frost's 10-pixel interior resize hit region. Pure egui regressions exercise
button click delivery, paused/running states, short windows and a checkbox-square
click in the actual resizable pane.

## IMU and contact preview lifecycle

`sensor_preview.rs` owns a three-slot result pool and a three-slot asynchronous
readback ring per active preview. The examples submit `SensorCommandBatch` outputs,
enqueue only the channels displayed by their pane/gizmos, and poll without waiting
on subsequent updates. Completed frames are drained in order; only the latest
collected frame is displayed. No intermediate sensor result is synchronously
downloaded. The simulation and renderer still have their own blocking readbacks;
this is not a claim that the entire viewer is nonblocking.

The one-mount IMU copies 32 bytes per measured sample (accelerometer and gyroscope).
Priming copies device history but does not request display readback. Contact copies
40 bytes per body: count, maximum depth, signed normal sum and centroid. The current
three-body scene therefore copies 120 bytes. Output and staging budgets each cover
three full frames, with checked budget multiplication. They exclude sensor history,
kernel resources, host display data and driver memory.

Closing the IMU pane stops new display readbacks, not device sampling/history.
One latest GPU frame is retained within the same three-slot pool. Reopening while
paused can collect that frame without redispatching the IMU or advancing its
history. Requests already enqueued before closing may still finish. Contact keeps
collecting while its pane is closed because all four channels drive viewport
gizmos as well as the pane.

`Sim::sensor_clock` advances once per completed ordinary rigid-physics substep.
The static 105/106/108 scenes and prescribed-motion scene 144 explicitly advance
the same clock on run/single-step without running a physics solver.
The current examples sample at most once per rendered physics frame: ten substeps
at 600 Hz, nominally 60 Hz of **simulation time**, not wall time. Repeated updates
while paused may collect pending display data but do not submit periodic measurements.
Examples 05 and 12 also accept explicit configuration requests while paused, as described below.
A single-step advances the clock and allows one new sample. Host-driven examples
that bypass the ordinary stepper still need their own explicit sample-time wiring
when migrated; this clock is not asserted to cover every existing viewer scene.
The contact stamp is the end-of-frame aggregation time. Its input geometry comes
from the last collision pass, before the final integrator substep (one `FIXED_DT`
earlier); the preview does not recompute collisions at the final body poses.

When a pool/ring is full, the preview counts that state's sample as deferred and
does not replay it later. The IMU's next accepted finite difference spans the time
since the last submitted velocity snapshot, including skipped steps. The pending
queue is bounded independently of render speed; it never grows a backlog of
unavailable intermediate simulation states.

Both reset controls and scene rebuilding advance a revision. The next sensor update
discards the previous sensor, pools, pending tickets and display data. The IMU primes
new history. Contact waits for the first post-reset physics step rather than sampling
the previous collision buffer. Reset reads back only device-authoritative body and
particle buffers before host mutation; it never disables storage dirty-state checks.

The panes expose nominal simulation cadence, submitted/collected counts, pending/
deferred counts, sample time and simulation age, CPU sensor-submit duration and an
optional GPU batch duration. CPU submission does not include the separate readback
enqueue or claim GPU execution time. Default devices show **GPU batch not measured**.
Errors stay visible until
reset or scene switch retries initialization; resetting cannot revive a lost GPU
device. Contact arrows and text describe geometry, never inferred net force.
Contact-normal arrows use a dedicated overlay group and remain visible through
bodies. Their tails are the sampled contact centroids, directions are normalized
signed normal sums, and lengths are 0.15 m plus capped penetration-depth scaling.
Cancelled normal sums display a small sphere instead of a direction. These
markers are neither force vectors nor depth-tested surface decorations; other
scene gizmos retain their ordinary depth behavior.

## Verification

Use the repository development environment and wrapper:

```sh
make check APP_TARGET='-p molla_viewer --all-targets'
make test-viewer-sensors
make test CARGO='nixVulkan cargo' APP_TARGET='-p molla_viewer --bin molla_viewer -- viewer_readback_tests --ignored --test-threads=1'
```

The sensor lane includes pure clock/budget checks and explicitly executes ignored
GPU tests. Coverage includes ring saturation/latest-frame draining, zero sensor
blocking waits, analytical IMU acceleration, GPU-only velocity input, skipped-sample
intervals, pause/single-step, reset with pending results, visible producer faults
and retry, closed-pane device-only sampling and paused reopen, scene switching,
contact parity against CPU aggregate references, empty
contact clearing, and host/device-authority reset combinations for bodies/particles.
GPU waits in these tests drain fixtures, not the preview update path.

## GPU LiDAR previews

Examples 05/105/106 share `lidar_preview.rs`. Each compiles one owned `SensorScene`,
retains a `LidarBatch` and a three-slot output/readback pool, and sends only scan
descriptors to generate rays on device. Pose-only changes use the current GPU body
buffer without rebuilding the scene; reset/switch recreates revision-bound state.
Sampling is ordered after the viewer step system, and gizmos after the preview update.

05 preserves the interactive origin, yaw/pitch, FOV, resolution and range controls.
Rows run from low to high pitch, columns from low to high yaw; singleton axes use
their central angle. The GPU spherical layout uses forward -Z and theta =
pitch - pi/2, preserving the former +Z-forward, +X-positive-yaw convention without
constructing ray arrays on the host. Auto-spin advances with simulation time and
freezes on pause.

Paused configuration edits request a new scan at the unchanged simulation time,
with a separate monotonic request ID. The latest request retries after readback
backpressure clears; superseded unsent requests are not replayed. Only three
in-flight configuration snapshots are retained. Displayed rays use their sampled
origin/range, not newer slider values, including while a replacement scan is pending.
Origin, direction, shape ID and point are copied: 52 bytes per ray. These channels
serve the viewport even when the pane is closed.

Resolution growth allocates replacement sensor/output/staging resources before
swapping them, retains the owned scene, and discards old pending readbacks and their
configuration snapshots. Existing displayed rays remain valid until replacement.
Preview counters restart on growth. Shrinking uses existing capacity. The maximum
UI resolution is 512 × 64 rays; checked multiplication rejects overflow.

105 preserves its fixed 24 × 64 grid with three divergent sub-rays per pulse,
0.006 rad beam half-angle and 25 m range. It requires Play or Step for the first
scan after entry/reset, then submits once per new simulation frame, not on
duplicate paused updates. It copies distance, shape ID and point only:
24 bytes per ray, or 110,592 bytes for 4,608 rays. The pane displays these settings;
they are not interactive scan-grid controls. Shape IDs select valid hits on CPU,
and distances color the point cloud. No host ray-vector construction is used.

106 preserves its one-shot pass on entry and now also refreshes after reset. It
uses an 8 × 8 m footprint at altitude 8 m and density two pulses per square metre.
GPU layout generates the Halton origins and nadir rays. It copies origin, shape
ID and point only: 36 bytes per ray. Origins are displayed once per pulse, even
if a scan uses multiple divergent sub-rays. It continues polling completion while
paused, but does not redispatch merely because another render frame arrives.

Both retain all nine device output channels required by `LidarBatch`; unused
directions, normals and pulse reductions are not downloaded for these displays.
Capacity rounds up to 64 rays with checked arithmetic. Output payload is 84 bytes
per capacity ray per slot; staging uses only the selected channels. The three-slot
budgets exclude owned geometry, query scratch, pipelines and driver memory.

The headless LiDAR test compares both displays with legacy CPU-layout/synchronous
results, checks GPU-only moving poses, retained hierarchy and allocation counters,
exact selected byte counts, zero preview blocking waits, pause/single-step,
reset with pending results, scene switching and aerial one-shot behavior. This
is parity/lifecycle evidence, not a timing benchmark or visual inspection.

The configurable LiDAR fixture independently constructs the former yaw/pitch rays,
compares GPU direction ordering and hits, and covers single-row/column scans, angle
wrapping and pole crossing, paused edits, historical display settings, simulation
rotation, capacity growth/shrink and pending-growth/reset/switch behavior. It checks
the full 512 × 64 UI resolution, exact selected transfer bytes and zero sensor
blocking waits.

## GPU camera preview

Example 12 retains its 1280 × 720 output, five channel thumbnails, FOV and physical
optics controls. `camera_preview.rs` owns a shared scene, camera, image processor,
bounded leased outputs and asynchronous display ring. Camera pose comes from
simulation time, so pause freezes the orbit. Explicit FOV/optics/post-process
changes while paused request a fresh image at the same simulation time.

Camera and photographic processing are encoded into one command batch. Exposure,
Reinhard tone mapping and linear-to-sRGB run on the GPU in that order, preserving
the previous RGBA8 quantization between operations. Identity processing uses the
camera's color output directly without an extra image-processing copy. Display
pose is retained with its submitted frame rather than taken from newer settings.

When the pane is open, all five configured channels are collected: depth, processed
color, normal, albedo and shape index, totaling 32 bytes/pixel (29,491,200 bytes per
720p frame). Raw color is not additionally downloaded when processed color is used.
Closing the pane stops new readbacks while GPU sampling continues. One latest
device frame is retained; reopening while paused queues that frame without a new
camera dispatch. Already queued readbacks may complete after the pane closes.
Per-thumbnail collapse is not a separate channel-selection control.

Camera output and staging each have three full-frame slots (88,473,600 payload
bytes each). Processed outputs have three slots of 4 bytes/pixel plus 48 bytes
for processing gains. These budgets exclude scene geometry, camera/query/optics
scratch, image-processor scratch, host image storage and driver memory.

Five Bevy image handles and egui texture IDs are allocated once and reused across
reset/switch/re-entry. Reset clears the valid-display marker and pending device
state while retaining controls; old thumbnails are not shown until a new frame
arrives. CPU code converts collected depth/normals/shape IDs to thumbnail colors
and computes the finite positive depth range. This is not renderer-device texture
interop, PBR camera rendering, or an entirely GPU-resident display.

Headless tests compare all five channels to the previous synchronous camera and
CPU photographic path, including aperture sampling. RGB comparisons allow one
8-bit level for GPU arithmetic; depth/normal tolerance is 1e-5 and albedo/shape IDs
are exact. Full-resolution ECS tests check five populated images, stable asset
handles/counts, pane hiding/reopening, simulation-time orbit, paused edits, reset
with pending work, scene switching and visible invalid-config errors/recovery.
No sensor preview blocking wait is used; tests explicitly drain their GPU fixtures.

## GPU annotation preview

Example 108 keeps its fixed 192 × 192 camera and labels: shape/class 1 is the box,
2 the sphere and 3 the capsule; the ground is unlabelled. Only the shape-index
camera channel is active. A pending camera frame feeds AnnotationProcessor in the
same command batch, with no intermediate camera image on the CPU.

The display reads three 32-byte compacted box records: 96 bytes per sample rather
than the 147,456-byte shape-index image. Zero-pixel tail records are ignored and an
empty completed frame clears previous detections. Class masks, shape masks and
annotation counts remain on device. Box collection continues with the pane closed
because detections also drive the viewport markers.

The right pane shows inclusive pixel bounds, shape/class IDs, pixel counts and
preview lifecycle/status. Viewport wireframes remain approximate fixed-size
world-space markers around currently detected bodies; they are not projections
of the image-space rectangles. This static scene advances a sensor-only clock on
play/single-step. It samples once at entry/reset even if paused, then once per new
simulation tick; duplicate paused updates only poll completed output.

Three camera-output slots budget 32 bytes/pixel each even though only shape ID is
active; the current library pool still allocates all five camera channels. Three
annotation-output slots budget 8 bytes/pixel plus 112 bytes each. Staging has three
96-byte slots. These payload budgets exclude constructor/query/work buffers,
geometry, host display records and driver memory.

The headless fixture compares every bound, class/shape ID and pixel count exactly
against the old camera plus Dataset::annotate path. It covers all three detections,
GPU-authoritative moved bodies producing an empty result, one retained scene build,
single label upload, exact selected transfers, pause/single-step, pending reset,
missing-channel fault/recovery, and switch/re-entry. This is not visual acceptance.

## Optional GPU timestamp measurements

Start the viewer with `MOLLA_SENSOR_TIMING=1 make run` inside the native development
environment to request timestamp-query and encoder-timestamp features when supported.
Default startup still uses the ordinary Engine constructor with no extra features.
The profiled viewer owns its device and adopts it through Engine::from_device,
forwarding its loss callback to that engine. Scene switches reuse the device.
Unsupported adapters continue without timing and display the unmeasured label.

All seven migrated previews and the three producers in scene 144 bracket their commands with two
timestamps inside the same command batch. The interval includes GPU scene refits
and sensor/post-processing passes but excludes prior host uploads, queue waiting
before the first marker, timestamp resolve/copy and display readback/rendering.
It is elapsed GPU timestamp time, not a sum of independently timed kernels or a
benchmark under exclusive device use. External GPU contention can affect it.

Each enabled timer has two queries, a 16-byte resolve buffer, a 16-byte transfer
buffer and three 16-byte staging slots. These are additional to the documented
sensor payload budgets. Timestamp collection is nonblocking and bounded; a full
timing ring skips timing without suppressing the sensor sample. The pane shows
timing deferrals, the timed sample ID and simulation age separately from the most
recent CPU submission measurement. Tiny intervals can round to zero.

Camera and IMU do not start timing readbacks while their panes are closed.
Already queued measurements may finish after closure. Their latest GPU timing can
therefore be older than a newly displayed frame on paused reopen; the shown timing
age/identity makes this explicit. Reset/revision changes invalidate old timings.

## Shared-scene sensor lab

Select **144 Shared-scene sensor lab** from the Example menu. A triangulated box
translates and rotates in front of a stationary sphere. Both query sensors use
one retained `SensorScene` with current body poses; the IMU sits at the moving
body's origin. Motion/velocity are prescribed analytic functions of simulation
time, not a rigid-body/contact solver. The matching rendered box is a primitive
visual, but the actual sensor collider has eight vertices and twelve triangles.
This demonstrates rigid moving-mesh occlusion, not deforming geometry.

The eight sensor panes scroll within the viewport; each scene retains its own
scroll position. Scroll down for camera/IMU data in the combined lab. Sensor-only
scenes label physics as **sensor-only (no solve)** instead of displaying stale
contact/solver timing from the previous scene.

`SensorScheduler` drives default LiDAR/camera/IMU rates of **10/5/30 Hz** in
simulation time. Each rate is adjustable from 1 to 60 Hz. LiDAR has five rows and
9–129 columns (65 initially), plus FOV/range controls. The fixed 160 × 120 camera
has an adjustable vertical FOV. Its simple traced color is not the Bevy renderer's
PBR image. The viewport displays LiDAR hit points; the pane shows the camera,
body-local IMU specific force/gyro, sample identity/age, nominal rate, submitted/
collected counts, skipped deadlines and backpressure. Camera and LiDAR ray counts
are shown separately from CPU submission and optional GPU timestamp durations.

Sampling sees the current end-of-render-frame state. Missed deadlines are counted,
not replayed with invented historical poses. Editing settings replaces the three
schedules with an epoch at the current simulation time; it does not rebuild the
scene or sensor pools. LiDAR/camera can refresh while paused. If IMU history is
already at that tick, its next request is delayed by one new period to avoid a
zero-time finite difference. Accepted IMU differences span the elapsed time since
the last successful submission, including any deferred requests.

Camera/IMU keep sampling when the pane closes but stop new display/timing readbacks.
One latest device frame per producer is retained for paused reopening. LiDAR
collection continues because it drives viewport points. Every producer uses three
result slots and three readback slots. The LiDAR capacity is 768 rays: 84 bytes per
capacity ray per output slot, 20 bytes per ray collected (position and shape ID).
Camera output slots budget 32 bytes/pixel even though only color is active;
collection is 4 bytes/pixel. IMU result/collection slots hold 32 bytes each;
priming has no displayed readback. These budgets exclude query/scene resources,
history, host display storage and driver memory. Reset discards pending epochs
while preserving controls and the one reusable camera image asset.

`viewer_sensor_lab_shared_mesh_cadence_occlusion_imu_and_lifecycle` checks default
cadence, mesh-to-sphere central-ray occlusion, camera parity with the synchronous
API, analytic IMU specific force using the model's gravity, exact selected byte
counts and zero preview blocking waits. It also covers hiding/reopening, settings
changes, skipped deadlines, pool saturation without replay, pending reset, visible
invalid-setting errors/recovery and switch/re-entry. Default and timestamp-enabled
viewer sensor lanes both pass; these headless checks do not establish rendered
acceptance or benchmark speedups.

## Remaining acceptance

Basic rendered inspection now covers all seven migrated examples and scene 144
on Mesa llvmpipe 26.1.3 Vulkan with an isolated 1600×1000 Xvfb display. Scene 144
also had a 1000×650 layout check. This exposed inaccessible playback buttons,
stale reset/switch counters and resize strips intercepting sensor controls; those
now have focused regressions. It also exposed occluded contact-normal segments;
the dedicated overlay arrows were inspected at default and closer viewing scales,
with reset removing old markers. The latest viewer sensor lane passes 17/17 both
with and without GPU timing. Test success is not a performance result.

## Hardware presentation (2026-09-25)

The viewer takes environment hooks for unattended runs: `MOLLA_VIEWER_SCENE`
starts on a scene by full label or, when unique, by its leading number (`105`,
`106` and `108` are shared with non-sensor examples and need the full label);
`MOLLA_VIEWER_ACTIONS` is a comma-separated `frame:action` script with the
actions `pause`, `play`, `step`, `reset`, `shot=<file.png>` and `exit`, applied
through the same run state the Space/Step/R controls use; and
`MOLLA_VIEWER_SCREENSHOT=<file.png>` is shorthand for one shot at
`MOLLA_VIEWER_SCREENSHOT_FRAME` (default 240) followed by `exit` 90 frames later. Run them through the GPU wrapper, for example
`MOLLA_VIEWER_SCENE="105 LiDAR v2 (scan grid + cone)" MOLLA_VIEWER_SCREENSHOT=/tmp/105.png nixVulkan target/release/molla_viewer`.

All eight examples were captured this way on an NVIDIA RTX 4080 (Vulkan, driver
595.84, native Wayland, release build) at frame 300: 05 LiDAR fan with 216/512
hits at 157 fps, 06 IMU at 152 fps, 07 contact arrows at 145 fps, 12 camera at
33 fps (the frame is bound by the XPBD download of about 6 ms, not the sensor),
105 LiDAR v2 with 4,170 displayed hits at 131 fps, 106 aerial pass with 128/128
hits at 564 fps, 108 annotation with three boxes at 703 fps and 144 sensor lab at
531 fps. Every pane reported its asynchronous collection status
(`Submitted / collected 300 / 298–299`, one pending). Captures:
`/tmp/molla-viewer-{05,06,07,12,105,106,108,144}.png`. The earlier NVIDIA/Xvfb
crash was not reproduced on the native display.

Launching viewers back to back can panic in Bevy's `prepare_windows` with
"Couldn't get swap chain texture … timeout … acquire the next frame" before the
first frame; five of eight first launches hit it and all passed on retry with a
five-second pause. That is a presentation race at window creation, not a sensor
fault, and it is not handled by the viewer.

Playback controls were also exercised on the RTX 4080 through `MOLLA_VIEWER_ACTIONS`
for all eight examples (running, pause, single step, play, reset, one capture each;
`/tmp/molla-viewer-interact/*.png`): paused panes report `Pending 0 / 0` with a
zero sample age and the Play/Step/Reset buttons, a step advances one sample, and
reset restarts the counters and simulation time with the displays rebuilt.

Still required: settings-pane edits and layout/resize interaction on hardware, and
rendered comparison against the software-renderer inspections. Software-renderer
FPS/timestamps are not hardware performance evidence, and the frame rates above
are single captures, not benchmarks. Do not close user preview windows to
perform these checks. Performance comparisons require monitoring external GPU
activity during the run; a competing application can start after an idle
initial sample.
