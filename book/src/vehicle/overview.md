# molla-vehicle overview

`crates/molla-vehicle` is the wheeled-vehicle simulation crate: terrain,
tires, suspension, driveline, a lightweight chassis, and the
differentiable system-identification demo. It is deliberately
**dependency-free** (no other workspace crates), and is re-exported
through the facade as `molla::vehicle::*`. It is also the home of the
`real_tire/` tree — the deformable tetra-FEM tire that the tire viewer
scenes (119–123) and the 40-wheel GPU scaling effort build on
(`PLAN_REALTIRE.md` / `PROGRESS.md` at the repo root track that work).

## Subsystems

| Module | What it provides |
|--------|------------------|
| `terrain` | `FlatTerrain`, `HeightfieldTerrain`, SDF and triangle-mesh terrain, plus the Bekker-pressure-sinkage `DeformableTerrain` and the DEM `GranularBed` (`deformable.rs` / `granular.rs`) |
| `tire` | Force-element tire models: `RigidTire`, `TmSimpleTire`, `Pac89Tire` (diagnostic shape only — molla ships no fitted Magic-Formula source), and the reusable `RealBrushTireModel` |
| `real_tire/` | The deformable tire stack: tetra tire FEM (`TetraTireFem`), brush/bristle traction, cavity pressure, thermal/wear state, parity ledger against the GPU kernels in `molla-kernels`, and the embedded validation rig/report harness |
| `suspension` | Per-corner spring/damper (`force()` from stiffness, damping, max compression). The `SuspensionKind` variants are descriptive labels by decision (INSPECTION.md F9): kind-driven camber/toe kinematics needs a chassis roll DOF the 3-DOF `Vehicle` deliberately excludes |
| `powertrain` | `Engine` (torque curve + redline), `Transmission` (gear ratios + RPM-triggered auto-shift with hysteresis), `Driveline` (FWD/RWD/AWD split), Ackermann steering, and `Vehicle::apply_driver` |
| `vehicle` | The 3-DOF chassis (`pos`/`vel`/`yaw`/`yaw_rate`, semi-implicit Euler) with four `Corner`s carrying wheel spin, steer, and tire state |
| `sysid` | Differentiable-simulation demo: recover tyre-ground friction from an observed trajectory by Gauss-Newton over finite-difference rollouts (`identify_friction`). Viewer scene 124 drives it live |

## GPU relationship

The crate itself is host-side; its GPU twins live in `molla-kernels`
(`tire_brush.wgsl`, `tire_fem*.wgsl`, bekker/DEM/thermal/transient tire
kernels) and are validated by the `tire_*` hardware lanes:

```bash
make test-gpu P=molla-kernels T=tire_fem_gpu
make test-gpu P=molla-kernels T=tire_brush_gpu
```

The CPU↔GPU status of each tire lane is recorded in
`real_tire/parity_ledger.rs` and the `TireFemTorqueGpuStatus` /
`TireFemMatrixFreeNewtonGpuStatus` contracts in `molla-kernels`.

## Viewer scenes

Scenes 103 (flat-ground car), 110 (offroad heightfield), 118–122 (Bekker
sinkage, soil compaction, tire FEM dynamics/traction/test-bench), 123
(terrain lab), and 124 (sysid friction recovery) exercise the crate
interactively.

## Testing

The crate is CPU-testable with plain `cargo test -p molla-vehicle`.
The suite also hosts the tire validation harness (`real_tire/rig`,
`real_tire/reports`) — structural truth gates (loaded radius, footprint,
pressure distribution, bead drift) and force-shape checks that guard the
tire physics without external measured-data calibration (that scope is
deliberately excluded — see the "Calibration stance" section of
`PLAN_REALTIRE.md`).


## Fully autonomous GPU substep loop (F2)

The tire FEM substep loop runs with zero per-substep host values:
`step_persistent → dispatch_resident(resident-spin) →
step_hub_persistent_with_tread_forces_spinning`. The wheel spin DOF lives
in `hub_state[1].zw` (integrated on-device with the exact
`integrate_spin_dof` formula, soft-sign brake saturator included), and the
feedback shader reads ω from the hub rows. Proven bit-identical to a
host-bridged reference over 96 rolling substeps.

The fused-atlas path scales this to fleets: one dispatch runs the bristle
physics for every tire (`step_resident_atlas_feedback`), the hub-param
bridge folds per-node force rows into each tire's `extf`, and
`hub_reduce_atlas` integrates all hubs (spin DOF included). Measured:
**1.53× realtime for 40 wheels** running the complete loop. A shared fleet
heightfield (`TireFemAtlasTerrain`) adds bilinear terrain sampling — an
all-zero heightfield is bitwise identical to the flat path, and an incline
makes the wheel drift downhill (the tilted contact normal redirects the
bristle reaction, which a flat-ground model cannot produce).
