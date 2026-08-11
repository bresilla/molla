# molla — real-tire program

_Condensed from the 262 KB PLAN_REALTIRE.md reverse-chronological log (2026-08). The
May-2026 consolidation (Streams A/B/C, torque G1-G4) is effectively **complete** — G4 rolling
CPU↔GPU parity is CLOSED, `TireFemTorqueGpuStatus = LiveValidated`, molla-vehicle baseline
225/0 since 2026-07-02. What remains below is the deeper-physics tail, not the streams._

For per-module status see [`STATUS.md`](STATUS.md). For the general roadmap see [`ROADMAP.md`](ROADMAP.md).

## Scope invariant (do not violate)

NO external calibration / dataset fitting. NO Pacejka/TMeasy/Magic-Formula as a *force source*.
NO MJCF/USD/URDF/FFI glue. NO hidden fitted constants. All hand-tuned values stay named,
unit-labelled, range-audited, provenance-tagged. Reference curves, if any, are diagnostic
shape/sign/saturation comparators only. Falsification is via deterministic replay rigs with
broad qualitative acceptance envelopes, **not** fitted target curves.

**Core premise:** a tire is a *force element*, not a collider. Each step the wheel queries the
terrain under the hub, derives the patch, computes slip ratio κ and slip angle α, and maps
(κ, α, Fz, μ) → (Fx, Fy, Mz). No GJK against the ground.

## Architecture (current, validated)

- **Brush/bristle contact** — `RealTireParams` + `evaluate_brush_tire*`: combined
  longitudinal/lateral slip, load sensitivity, transient relaxation length, camber thrust,
  pneumatic trail, aligning moment; state-derived dry/wet/ice/dirt/soil + surface-film +
  thermal/wear response. Leading-edge bristle profile landed
  (`brush_bristle_profile` + `brush_pressure_patch_gain`; GPU mirror `tire_brush.wgsl`).
- **Tetra FEM tire (`TetraTireFem`)** — compressible Neo-Hookean tets + Holzapfel-style fibre
  stress across four explicit ply families (crossed ± belt/sidewall, cap/hoop, radial/axial
  helpers); gas-law cavity pressure via named `cavity_carcass_coupling` preset; XPBD
  constraints for volume, bead mount, structural edges, sidewall dihedral, sidewall ply-strain.
- **GPU path (`molla-kernels`)** — `tire_fem.wgsl` (one invocation/node, CSR adjacency, no
  float atomics), `tire_fem_hub.wgsl` (1-DoF hub reduction/integration), `tire_brush.wgsl`
  (batched combined-slip on flat/heightfield terrain), `tire_fem_atlas.wgsl` (fused multi-tire
  atlas). Resident owners: `TireFemGpuAtlasBatch`, `TireResidentVehicleFemAtlasOwner`. Viewer
  scenes 119-123 consume these; scene 120 defaults to the combined resident vehicle+FEM owner.
- **Terrain coupling** — heightfield + triangle-mesh + simple-SDF queries (CPU FEM projection
  uses true SDF/triangle signed distance); Bekker soil sinkage; first-pass CPU+GPU DEM
  tire-cell-to-grain impulse transfer with GPU sparse-LBVH neighbour search; CPU+GPU MPM
  compaction-to-terrain feedback. Barrier-Newton (C0) contact backend landed (CPU+GPU,
  self-collision, tets, hard strain limits, μ_s/μ_k, hash pruning).

## Key decisions worth preserving

1. **Body-frame canonical storage** is the settled answer for rolling FEM tires: physics
   (forces/constraints) run on `body_pos`/`body_vel`; contact runs in world and transforms
   Δp/Δv back via R⁻¹; `pos`/`vel` are a derived world view rebuilt each substep. Spin
   `v_spin = ω×r` falls out of the transform. Mirrored host↔GPU so the atlas path drops in.
   (The earlier per-substep rigid-rotation kinematic pass was reverted for jelly-wobble.)
2. **GPU-first is the production path** for the 40-wheel target (~10 machines × 4 wheels). One
   FEM wheel saturates a CPU core (~200 ms CPU / simulated-second at 8000 substeps/s), so CPU
   `TetraTireFem` is reference/oracle only; the fused resident GPU atlas is production.
3. **Structural truth gates precede traction trust** — loaded radius, sidewall bulge, footprint
   width/length, pressure distribution, belt support, bead/rim drift, cavity response must be
   inspectable before believing any force number.
4. **Every tire scene carries a truth label + approximation ledger** (physics path, expected
   behaviour, controls/state, known approximations, CPU-vs-GPU deltas). The default
   non-calibration runner fails if a report has no replay context.

## Open roadmap (the deeper-physics tail)

- **[high] Implicit/Newton rubber solve** — replace explicit Neo-Hookean stepping with a true
  implicit tire solve; promote the matrix-free Newton debug lane
  (`step_on_terrain_matrix_free_newton_debug`) from CPU correctness gate to production,
  mirroring the body-frame layout. It does not yet integrate the spin DOF.
- **[med] True laminated sidewall shell/carcass constraint** — beyond the current ply-strain
  XPBD + dihedral projection proxy (`sidewall_bend_proxy`).
- **[med] Validated rim/bead-seat contact solve** — retire the `cavity_carcass_coupling` proxy
  fraction (`cavity_bead_sealing`).
- **[med] Barrier-Newton (C0) follow-ons** — elasticity-inclusive stiffness κ, LBVH
  acceleration, warm-start PCG.
- **[med] Fully GPU-resident FEM terrain feedback** — close the Bekker / DEM / MPM bridge into
  full residency (bridges exist; end-to-end residency does not).
- **[med] Broader live-vehicle-scene migration** onto `TireResidentVehicleFemAtlasOwner` +
  deeper full-vehicle GPU chassis ownership (scene 120 defaults to it; richer scenes pending).
- **[low] GPU sparse-LBVH for DEM beds** — broader multi-scene adoption + larger-scale scaling
  evidence (Morton/radix/leaf/tree/no-readback-resident-builder stages already landed).
- **[low] Close/annotate remaining planned-GPU parity-ledger rows** (e.g.
  `tetra_fem_sidewall_laminate_coupling_report`) — each keeps an explicit reason + follow-up.
- **[low] Deeper seam-specific triangle-contact visualization** if fixtures need it
  (`TireTriangleSeamManifoldReport` is the current first-pass seam sanity report).

## Test / validation infrastructure (already built — keep)

Reusable non-calibration runner emits deterministic rigs: vertical load/deflection sweep;
longitudinal/combined/camber slip sweeps across asphalt/dirt/wet/soil/ice; torque/brake/steer
step response; heightfield bump/curb + rolling-obstacle sequence; thermal/wear heat-peak +
cooling-recovery. Sanity gates: no NaN/inf; no terrain tunnelling; lower pressure → larger
patch; ice caps force below dry asphalt; adhesion↔sliding continuity; non-negative dissipated
energy; dimensionless ratios finite. Presets (`RealTireParams`, `SurfaceFilmState`,
`TireThermalLayers`, `TetraTireFemParams`, `DeformableTerrain`) carry unit/range/provenance
metadata; viewer sliders are preset `_override` inputs, not the source of truth. The GPU parity
checklist tracks CpuAndGpu / CpuOnlyDebug / PlannedGpu rows.

## Module layout (`real_tire/` — reference if re-verifying)

`surface_material.rs`, `brush.rs`, `transient.rs`, `evaluate.rs`, `contact.rs`, `telemetry.rs`,
`realism.rs`, `triangle.rs`, `parity_ledger.rs`, `rig/`, `reports/`, `carcass.rs`, and `fem/`
(`params`, `mesh`, `tetra`, `xpbd`, `contact`, `solver`, `matrix_free`, `diagnostics`,
`gpu_snapshot`, `torque`). All public items re-exported via `pub use` from `real_tire/mod.rs`
(243 lines); shared helpers (`finite_or`, `normalize3`, `smoothstep01`) are `pub(crate)`.
