# Controlled-clock repeatability and legacy comparison — 2026-09-25

The rerun of the three instrumentation-off latency matrices under locked GPU
clocks, plus a matched comparison against the pre-plan source (`d2369a9`).
These are recorded results and limitations, not regression budgets.

## Environment

- **Hardware:** RTX 4080 / Vulkan, NVIDIA 595.91.07, Linux 7.0.0-34.
- **Build:** release profile, from the repository Makefile inside the Nix shell.
  `info-before.log` and `info-after.log` hold toolchain, HEAD, dirty count, the
  crate fingerprint and driver.
- **Locked clocks for the matrix sessions:** graphics clocks were locked at
  2505 MHz and memory at 10501 MHz with `sudo nvidia-smi -lgc/-lmc`. The lock
  was taken at 12:29:02 CEST and reset (`-rgc/-rmc`) at 12:51:28 by an exit
  trap; see `clock-lock.log` and `clock-reset.log`. `gpu.csv.gz` (500 ms) shows
  every sample at 2505 MHz; the only throttle reason reported is GPU idle.
- **Other GPU users:** the desktop was a GNOME Remote Desktop session.
  `gnome-remote-desktop-daemon` held a GPU context (≈0.9–1.2 GB) throughout,
  and its encoding load varies with screen activity.

## Artifacts

- `{lidar,camera,cloud}-{a,b}.{tsv,log}.gz`: two independent sessions, each run
  with `--no-timestamps --samples 64 --warmup 8 --repeats 3`. The matrices hold
  120, 56 and 96 cases.
- `transfers.tsv`: per case (session a), the mean upload bytes, result-download
  bytes, submissions and waits per measured frame.
- `legacy/`: the interleaved legacy/current criterion runs (see below).
- `memory/`: one repeat per matrix, with per-process VRAM sampled every 200 ms
  (`vram.csv`) and peak RSS (`/usr/bin/time -v` in each log).

## Repeatability

Each count is the number of cases whose p50 or p95 (nearest rank, 64 samples)
varied by more than 10% across the three repeats. The archive rows are the
earlier unlocked instrumentation-off runs.

| Matrix | Unlocked archive p50 / p95 | Locked a p50 / p95 | Locked b p50 / p95 |
|---|---|---|---|
| LiDAR (120) | 88 / 93 | 7 / 46 | 40 / 60 |
| Camera (56) | 10 / 29 | 12 / 23 | 15 / 26 |
| Cloud (96) | 84 / 83 | 63 / 77 | 35 / 50 |

The variation has three causes, each shown in the data.

1. **Interference windows.** In run order, every block of 30 case/repeat groups
   in LiDAR session a sits at 1.00–1.07× each case's fastest median. Session b
   has three blocks where almost every case ran 1.3–2.3× slower, with normal
   blocks between. Cloud session a has three such blocks. Clocks stayed locked
   throughout, so these are episodes of outside load on the machine (the remote
   desktop encoder is the one other GPU client), not frequency changes.
   `position-drift` analysis:
   - LiDAR: 0 of 12 windows above 1.2× in session a, 3 of 12 in session b;
   - camera: 0 of 6 in both sessions;
   - cloud: 3 of 10 in session a, 0 of 10 in session b.
2. **Too few warm-ups on a case's first repeat.** In cloud session b (no
   interference windows), 25 of the 35 unstable cases are terrain or
   static-mesh cases whose first repeat is 3–5× slower than the other two.
   Example: `terrain_r100000_m4_s1_warm_cloud_keep_device` at 2,121 / 419 /
   415 µs. Eight warm-up frames do not absorb their first-pass cost. The other
   15 are moving-mesh `pose` cases, whose medians also wander between repeats.
3. **p95 from 64 samples is a tail estimate.** It is roughly the fourth-largest
   sample, so single scheduling outliers move it. Even quiet session a has 46
   LiDAR cases above 10% at p95 but only 7 at p50.

Under locked clocks and without interference (LiDAR session a), 113 of 120
LiDAR medians repeat within 10%. No hard p95 budget is set from these runs:
the interference cannot be controlled on this shared desktop, and the cloud
first-repeat effect needs a longer warm-up in the harness.

## Transfers

From `transfers.tsv`, all 136 device-only cases download zero result bytes per
frame. Host cases download only the selected outputs. The largest are:

- 117,964,800 B per frame: four 1280×720 aperture cameras (`1280x720_c4_aperture4_heterogeneous_warm_camera_host`);
- 116,314,460 B: one million cloud-cropped rays (`mesh100_r1000000_m16_s1_pose_cloud_crop_host`);
- 52,023,296 B: one million LiDAR rays (`mesh10_r1000000_m16_s1_pose_host`).

## Legacy comparison

The pre-plan source (`git archive d2369a9`) and the current tree ran the same
criterion groups, interleaved over three rounds. Clocks were not locked.
Interleaving exposes both trees to the same drift, and `legacy/gpu.csv.gz` and
`legacy/pmon.log.gz` record board and per-process load.
`sensor_raycast_gpu` is unchanged between the trees. `sensor_camera_gpu`
in the legacy copy was given the current loop (GPU completion, or completion
plus all five host downloads), because the original timed CPU submission only.
Medians of the three rounds' criterion estimates:

| Benchmark | Legacy | Current | Current / legacy |
|---|---|---|---|
| `Raycaster::cast`, 1,000 rays / 100 shapes | 3.99 ms | 0.29 ms | 0.07 |
| `Raycaster::cast`, 10,000 rays / 1,000 shapes | 4.92 ms | 1.07 ms | 0.22 |
| Camera + five host channels, 256² | 5.98 ms | 0.47 ms | 0.08 |
| Camera + five host channels, 512² | 8.56 ms | 1.24 ms | 0.15 |
| Camera + five host channels, 720p | 29.3 ms | 4.30 ms | 0.15 |
| Camera dispatch to GPU completion, 256² | 86.8 µs | 145 µs | **1.67** |
| Camera dispatch to GPU completion, 512² | 193 µs | 262 µs | **1.35** |
| Camera dispatch to GPU completion, 720p | 424 µs | 590 µs | **1.39** |

The synchronous paths with host results are 4.6–14× faster. The bare camera
dispatch to completion is a **regression of 1.35–1.67×**: the current default
camera always traces all channels through the separate ray/trace/shade
kernels. It is recorded, not waived.

## Peak memory

One instrumentation-off repeat per matrix. The benchmark process's VRAM was
sampled every 200 ms (`memory/vram.csv.gz`), and peak RSS comes from
`/usr/bin/time -v` (`memory/*.log`).

| Matrix | Peak process VRAM | Peak RSS |
|---|---|---|
| LiDAR (120 cases) | 491 MiB | 541 MiB |
| Camera/image/annotation (56) | 1,067 MiB | 1,660 MiB |
| LiDAR/cloud/grid (96) | 1,323 MiB | 1,195 MiB |
