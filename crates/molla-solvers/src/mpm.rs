//! Stage 24d.1 — Material-Point-Method (MPM) for granular materials.
//!
//! Builds on Stage 24d v1 (explicit MPM) with five 24d.1 deliverables:
//!
//! 1. **Semi-implicit time-step** ([`MpmTimeStep::Implicit`]). Per the
//!    brief's scope-realism note ("if you hit a wall, ship semi-
//!    implicit"), we ship the diagonal-preconditioned Jacobi
//!    relaxation flavour rather than full matrix-free CG over the
//!    grid Lagrangian — see [`MpmTimeStep::Implicit::jacobi_iters`].
//!    Each iteration applies
//!    v_new = (m·v_explicit + dt² · k_diag · v_avg) /
//!    (m + dt² · k_diag)
//!    where `k_diag` is a particle-scattered diagonal estimate of the
//!    stress Hessian and `v_avg` is the 6-neighbour grid average.
//!    The convergence numbers are weaker than Klár-style Newton CG
//!    but the stability win — larger dt without explicit-CFL blow-up
//!    — is the same; we measure 4×-6× larger dt allowed in practice.
//!
//! 2. **Calibrated angle of repose**. Stage 24d v1's pile test was
//!    qualitative; 24d.1 ships [`MpmConfig::sand_calibrated`] tuned
//!    so a 30° friction-angle parameter produces a measured pile
//!    slope of 30° ± 5° in a column-collapse test (see
//!    `tests/mpm_v3_gpu.rs::mpm_calibrated_aor_30deg`).
//!    Calibration knobs that mattered most:
//!      * particle volume / cell volume ≈ 1 / (4-8) so each cell sees
//!        a healthy mass deposit (under-resolved cells produce a
//!        soft pile that flattens too far).
//!      * Young's modulus ≥ 5e4 Pa so the elastic stress can hold
//!        the pile against gravity (1 kPa is too soft).
//!      * dt small enough that the elastic step doesn't oscillate;
//!        1/480 s with E = 5e4 Pa is comfortable.
//!
//! 3. **Snow rheology** ([`MpmRheology::Snow`]). Stomakhin 2013 §6.2
//!    multiplicative split: F = F_e · F_p with elastic singular
//!    values clamped to `[1 - θ_c, 1 + θ_s]` and `log det F_p`
//!    rolling into `jp` for exp-hardening of μ and λ. No friction
//!    angle.
//!
//! 4. **Viscous rheology** ([`MpmRheology::Viscous`]). A GPU-friendly
//!    damped material path for fluid-like demos: after grid gather it
//!    applies exponential velocity and affine-velocity damping, while
//!    keeping the elastic deformation gradient unprojected.
//!
//! 5. **Two-way rigid coupling**. New kernel
//!    `mpm_rigid_force_integrate` aggregates per-cell stress
//!    impulses (`m · w · (v_rigid - v_grid)`) for cells inside or
//!    near each rigid sphere into a single 6-DoF wrench `(force,
//!    torque)` per rigid body. Host reads back via
//!    [`MpmSimulation::read_rigid_wrenches`].
//!
//! ## Pipeline (one substep, implicit case)
//!
//! 1. `reset_grid` — clear mass + momentum + k_diag + v_prev atomics.
//! 2. `p2g` — particle → grid scatter (mass, momentum, **k_diag**).
//! 3. `rigid_imprint` — stamp rigid-body velocities into overlapped
//!    grid cells.
//! 4. `grid_update` — momentum / mass → velocity, gravity, boundary
//!    friction. Output goes into `g_velocity`.
//! 5. **Implicit loop** (only when [`MpmTimeStep::Implicit`]):
//!    `jacobi_iters` × (`mpm_implicit_apply`, `mpm_implicit_swap`).
//! 6. `mpm_rigid_wrench_reset` then `mpm_rigid_force_integrate` —
//!    aggregate two-way coupling wrenches.
//! 7. `g2p` — grid → particle gather; APIC affine matrix C; F update.
//! 8. `plasticity` — SVD return mapping (sand DP **or** snow
//!    Stomakhin).
//!
//! ## Public API
//!
//! ```rust,ignore
//! use molla::compute::{Engine, MpmSimulation, MpmConfig, MpmRheology, MpmTimeStep};
//!
//! let engine = Engine::new_gpu()?;
//! let mut cfg = MpmConfig::sand_calibrated();
//! cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 3 };
//! cfg.rheology = MpmRheology::Sand { friction_angle: 30f32.to_radians() };
//! let mut sim = MpmSimulation::new(&engine, cfg)?;
//! sim.set_particles(queue, &positions)?;
//! for _ in 0..240 {
//!     sim.step(&engine, 1.0 / 240.0)?;
//! }
//! let wrenches = sim.read_rigid_wrenches(&engine)?;
//! ```
//!
//! ## Performance (Stage 24d.3 atomic-contention reduction)
//!
//! Measured on the dev laptop (debug build, single-threaded `cargo test`):
//!   * 10k particles + 32³ grid via `step_n(100)`: ~ 45 Hz (unchanged
//!     by 24d.3 — at this scale submit cost dominates over per-particle
//!     atomics, so dedup doesn't help much).
//!   * 50k particles + 64³ grid: legacy single-pass `p2g` **7.9 Hz** →
//!     Stage 24d.3 workgroup-local two-pass `p2g_two_pass` **36.9 Hz**
//!     (**4.66× speedup**, ≥ 30 Hz DoD). The remaining 60 Hz gap
//!     requires implicit CG (Stage 24d.4).
//!
//! Mechanism: legacy `p2g` does ~ 5 global atomic-CAS adds per cell
//! per particle (1 mass + 3 momentum + 1 kdiag = 5 × 27 cells × 50 000
//! particles = 6.75 M global CAS / substep). `p2g_two_pass` aggregates
//! those into a workgroup-local 4×4×4 = 64-slot scratch
//! (`var<workgroup>` atomic CAS — workgroup-shared memory, no L2 round-
//! trip), then commits each non-empty slot once at the end with one
//! global CAS per accumulator. For spatially-clustered particles
//! (the common MPM granular case) the slot is direct-mapped from the
//! lower 2 bits of the cell index per axis so consecutive particles
//! in a workgroup share slots → ~ 5–10× reduction in global-atomic
//! traffic. Hash collisions (different cells mapping to the same
//! slot) fall back to the legacy global-atomic path; correctness is
//! preserved either way. See `set_p2g_mode` to toggle for the parity
//! test in `tests/mpm_perf.rs`.
//!
//! [`MpmSimulation::step_n`] batches `N` substeps into a single
//! encoder + single `queue.submit`, sharing pre-allocated uniform +
//! readback staging buffers. `step()` is retained as a back-compat
//! wrapper that delegates to `step_n(_, _, 1)`. The viewer + perf
//! tests use the batched path; see
//! `tests/mpm_perf.rs::mpm_step_n_beats_individual_steps` for the
//! empirical proof (9.1× speedup at 60 substeps, well above the 2× DoD).

use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::kernels::{
    PreparedTireMpmRaster, PreparedTireMpmTerrainReadback, TireFemContactTerrain,
    TireMpmContactCell, TireMpmFemTerrainOwner, TireMpmGpuInPlaceParams, TireMpmGpuInput,
    TireMpmGpuKernel, TireMpmGpuOutput, TireMpmResidentFeedback, TireMpmTerrainRasterParams,
    tire_mpm_reference_feedback,
};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[path = "mpm_active_tiles.rs"]
mod active_tiles;
#[path = "mpm_native_atomics.rs"]
mod native_atomics;

#[path = "mpm_particle_order.rs"]
mod particle_order;

#[path = "mpm_deterministic_p2g.rs"]
mod deterministic_p2g;

#[path = "mpm_compensated_stiffness.rs"]
mod compensated_stiffness;

#[path = "mpm_compensated_wrench.rs"]
mod compensated_wrench;

#[path = "mpm_command_batch.rs"]
mod command_batch;
pub use command_batch::{MpmCommandBatch, PreparedMpmStep};

#[path = "mpm_checkpoint.rs"]
mod checkpoint;
#[path = "mpm_frame_backup.rs"]
mod frame_backup;
#[path = "mpm_motion_guard.rs"]
mod motion_guard;
pub use checkpoint::{
    MpmSoilActivity, MpmSoilActivityReport, MpmSoilCanonicalSelection, MpmSoilCanonicalStore,
    MpmSoilCanonicalStoreRead, MpmSoilCanonicalStoreReadCompletion, MpmSoilCanonicalStoreReadTask,
    MpmSoilCanonicalStoreWritebackCompletion, MpmSoilCanonicalStoreWritebackTask,
    MpmSoilCanonicalTileArchive, MpmSoilCheckpoint, MpmSoilCheckpointCompletion,
    MpmSoilCheckpointReadback, MpmSoilCheckpointTicket, MpmSoilDiskWorld, MpmSoilDiskWorldStatus,
    MpmSoilEnergy, MpmSoilFrame, MpmSoilFrameCheckpoint, MpmSoilFrameSelection, MpmSoilFrameStore,
    MpmSoilFrameStoreRead, MpmSoilFrameStoreReadCompletion, MpmSoilFrameStoreReadTask,
    MpmSoilFrameStoreWritebackCompletion, MpmSoilFrameStoreWritebackTask, MpmSoilFrameTileArchive,
    MpmSoilMomenta, MpmSoilPinConfig, MpmSoilReadbackStats, MpmSoilRegionEnvelope,
    MpmSoilSelection, MpmSoilSettling, MpmSoilSettlingConfig, MpmSoilSettlingLimits,
    MpmSoilSettlingProgress, MpmSoilSettlingStatus, MpmSoilStore, MpmSoilStoreLimits,
    MpmSoilStoreRead, MpmSoilStoreReadCompletion, MpmSoilStoreReadLimits, MpmSoilStoreReadStats,
    MpmSoilStoreReadTask, MpmSoilStoreStats, MpmSoilStoreStorageStats,
    MpmSoilStoreWritebackCompletion, MpmSoilStoreWritebackTask, MpmSoilTileActivity,
    MpmSoilTileArchive, MpmSoilTransferPath, MpmSoilTransferStats, MpmSoilWorld,
    MpmSoilWorldStatus,
};
pub(crate) use frame_backup::MpmFrameBackup;
pub use motion_guard::{MpmSoilQuiescence, MpmSoilQuiescenceSnapshot};
pub use motion_guard::{MpmSoilRetainedSupportParticle, MpmSoilRetainedSupportSnapshot};
pub use motion_guard::{MpmSoilStencilSupport, MpmSoilSupportSnapshot};
pub use motion_guard::{MpmSoilSupportClosureParticle, MpmSoilSupportClosureSnapshot};

#[cfg(test)]
#[path = "mpm_fused_gather.rs"]
mod fused_gather;
#[path = "mpm_profile.rs"]
mod profiling;
#[path = "mpm_soil_identity.rs"]
mod soil_identity;
pub use profiling::{MpmProfileStage, MpmStageTiming, MpmStepProfile};
pub use soil_identity::{MpmSoilIdentityBuffers, MpmSoilIdentitySnapshot};

#[path = "mpm_prescribed_solids.rs"]
mod prescribed_solids;
pub use prescribed_solids::{MpmPrescribedSolid, MpmSolidShape};
#[path = "mpm_rigid_device.rs"]
mod rigid_device;
pub use rigid_device::{MpmRigidDeviceBinding, PreparedMpmRigidDeviceBridge};

#[path = "mpm_mass_coupling.rs"]
mod mass_coupling;
#[path = "mpm_particle_fault_bridge.rs"]
mod particle_fault_bridge;
pub use mass_coupling::{
    MPM_COUPLING_BOUNDARY_FAULT, MPM_COUPLING_CAPACITY_FAULT, MpmMassCouplingInputs,
    PreparedMpmMassCoupling,
};
pub use particle_fault_bridge::MPM_COUPLING_PARTICLE_FAULT;
pub(crate) use particle_fault_bridge::MpmMachineFaultBridge;
#[path = "mpm_unilateral_coupling.rs"]
mod unilateral_coupling;
pub use unilateral_coupling::{
    MpmUnilateralConfig, MpmUnilateralInputs, PreparedMpmUnilateralCoupling,
};

#[cfg(test)]
#[path = "mpm_replay_tests.rs"]
mod replay_tests;

#[cfg(test)]
#[path = "mpm_numerical_fault_tests.rs"]
mod numerical_fault_tests;

#[cfg(test)]
#[path = "mpm_native_replay.rs"]
pub(crate) mod native_replay;

#[cfg(test)]
#[path = "mpm_native_stencil.rs"]
mod native_stencil;

#[cfg(test)]
#[path = "mpm_native_compensated.rs"]
mod native_compensated;

#[cfg(test)]
#[path = "mpm_native_compensated_active.rs"]
mod native_compensated_active;

#[cfg(test)]
#[path = "mpm_local_transfer.rs"]
mod local_transfer;

#[cfg(test)]
#[path = "mpm_scatter_wrench_replay.rs"]
mod scatter_wrench_replay;

#[cfg(test)]
#[path = "mpm_relative_transfer.rs"]
mod relative_transfer;

#[cfg(test)]
#[path = "mpm_wrench_oracle.rs"]
mod wrench_oracle;

#[cfg(test)]
#[path = "mpm_bounded_reference.rs"]
mod bounded_reference;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmActiveGridStats {
    pub active_tiles: u32,
    pub tile_capacity: u32,
    pub overflow: bool,
    pub numerical_fault: bool,
    /// A swept-residency displacement bound was exceeded during gather.
    pub motion_bound_exceeded: bool,
    /// Gather count; a faulted batch may include a partially updated substep.
    pub completed_substeps: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MpmGridStorage {
    #[default]
    Dense,
    ActiveTiles,
    Compact {
        tile_capacity: u32,
    },
}

impl MpmActiveGridStats {
    pub fn faulted(&self) -> bool {
        self.overflow || self.numerical_fault || self.motion_bound_exceeded
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    particle_count: u32,
    rigid_count: u32,
    grid_size: u32,
    grid_total: u32,

    grid_origin: [f32; 3],
    cell_size: f32,

    dt: f32,
    gravity_y: f32,
    youngs: f32,
    poisson: f32,

    dp_alpha: f32,
    cohesion: f32,
    boundary_friction: f32,
    inv_cell_size: f32,

    rheology: u32, // 0 = sand, 1 = snow
    snow_theta_c: f32,
    snow_theta_s: f32,
    snow_hardening: f32,
    grid_layout: [u32; 4],
    grid_window_cells: [i32; 4],
}

const WORKGROUP_SIZE: u32 = 64;

/// Stage 24d.1 — rheology selection. Sand uses Drucker-Prager; snow
/// uses Stomakhin elastic-envelope clamping with hardening on
/// `log det Fp`; viscous uses GPU-side exponential velocity/affine
/// damping for fluid-like material smoke tests.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MpmRheology {
    /// Drucker-Prager granular sand. `friction_angle` is in **radians**.
    Sand {
        /// Friction angle φ in radians (sand: 25°-40° = 0.44-0.70 rad).
        friction_angle: f32,
    },
    /// Stomakhin 2013 snow. Elastic envelope `[1 - θ_c, 1 + θ_s]`;
    /// hardening exponent on `log det Fp`.
    Snow {
        /// Critical compression θ_c (Stomakhin: 2.5e-2).
        theta_c: f32,
        /// Critical stretch θ_s (Stomakhin: 7.5e-3).
        theta_s: f32,
        /// Hardening exponent (Stomakhin: 10).
        hardening: f32,
    },
    /// Viscous material path. This is intentionally simple and local:
    /// it damps particle velocity and APIC affine velocity after G2P
    /// with `exp(-damping * dt)`. It gives the Warp-fluid parity plan
    /// a real GPU-owned viscous knob without pretending to be a full
    /// incompressible Navier-Stokes solve.
    Viscous {
        /// Linear velocity damping coefficient in 1/s.
        velocity_damping: f32,
        /// APIC affine-velocity damping coefficient in 1/s.
        affine_damping: f32,
    },
}

impl Default for MpmRheology {
    fn default() -> Self {
        Self::Sand {
            friction_angle: 30.0_f32.to_radians(),
        }
    }
}

/// Stage 24d.1 — time-step selection.
///
/// `Explicit` reproduces the Stage 24d v1 path bit-exactly. `Implicit`
/// runs a fixed number of diagonal-preconditioned Jacobi relaxation
/// sweeps on the grid velocity — cheap (≈ 1 grid sweep per
/// iteration), unconditionally stable, and lets the caller raise dt
/// 4×-6× without blow-up.
///
/// **Caveat (Jacobi vs CG):** this is **not** Conjugate Gradient — no
/// Krylov subspace, no residual-based early exit. The Jacobi
/// preconditioner is the diagonal of the stress-Hessian estimate
/// scattered from the particles. On highly anisotropic strain states
/// (e.g. very stiff snow under unilateral compression) it
/// under-corrects and a few extra iterations are needed; if the user
/// reports oscillations, raise `jacobi_iters` to 8-20 or fall back to
/// `Explicit` with a smaller dt. The "20-iter cap with residual exit"
/// in the original brief was for a full-CG plan; with Jacobi we
/// simply run a fixed number of sweeps.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum MpmTimeStep {
    /// Stage 24d v1 explicit symplectic step.
    #[default]
    Explicit,
    /// Semi-implicit step with `jacobi_iters` Jacobi sweeps. Typical
    /// 1-3 for sand at 1/240 s; 5-10 for snow / stiff materials.
    Implicit {
        /// Number of diagonal-preconditioned Jacobi sweeps per
        /// substep. The original brief named this `cg_iters` for a
        /// full-CG plan that never landed; the actual implementation
        /// is fixed-iteration Jacobi (no CG, no residual exit).
        jacobi_iters: u32,
    },
}

/// Material configuration. Defaults are tuned for dry sand
/// (Klár 2016 §6).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MpmConfig {
    /// Cubic grid edge length (cells per side).
    pub grid_size: u32,
    /// World-space size of one grid cell.
    pub cell_size: f32,
    /// World-space lattice anchor. The active window is displaced by
    /// [`MpmSimulation::grid_window_cells`] integer cells from this point.
    pub grid_origin: [f32; 3],
    /// Gravity along `+Y`. Set to `-9.81` for terrestrial sims.
    pub gravity_y: f32,
    /// Young's modulus E (Pa). Sand: `1e5` is typical for "sandbox" toy
    /// rheology; `1e7` for cohesive packings.
    pub youngs: f32,
    /// Poisson's ratio ν.
    pub poisson: f32,
    /// Drucker-Prager friction angle φ (radians). Used to compute the
    /// dimensionless cone constant α host-side. **Deprecated**: prefer
    /// [`MpmConfig::rheology`]; this field shadows it for source-compat
    /// when the rheology is `Sand`.
    pub friction_angle: f32,
    /// Cohesion c (Pa). 0 for granular sand.
    pub cohesion: f32,
    /// Velocity multiplier on the boundary tangent (Coulomb-style).
    /// 0 = sticking, 1 = frictionless slip.
    pub boundary_friction: f32,
    /// Per-particle mass (kg). For uniform fills only — extension
    /// hooks live on `set_particles`.
    pub particle_mass: f32,
    /// Per-particle reference volume (m³).
    pub particle_volume: f32,
    /// Maximum particles the simulation will accept.
    pub max_particles: u32,
    /// Maximum rigid-body imprints per substep.
    pub max_rigid: u32,
    /// Stage 24d.1 — rheology selector.
    pub rheology: MpmRheology,
    /// Stage 24d.1 — time-step selector.
    pub time_step: MpmTimeStep,
}

/// Tire-specific material-point feedback parameters for composing the
/// tire-contact pass with [`MpmSimulation`].
///
/// This is deliberately not calibration data. These are numerical/physical
/// coupling knobs that decide how tire contact-cell force and pressure are
/// injected into the MPM particles before the full particle/grid solver keeps
/// advancing the continuum.
#[derive(Clone, Copy, Debug)]
pub struct MpmTireFeedbackParams {
    /// Tire feedback integration step in seconds.
    pub dt: f32,
    /// Plastic compaction increment per pressure·second.
    pub plastic_compaction_gain: f32,
    /// Clamp for the accumulated plastic sinkage/compaction diagnostic.
    pub max_compaction: f32,
    /// Local velocity damping around active tire contact cells.
    pub damping: f32,
}

impl MpmTireFeedbackParams {
    pub fn new(dt: f32) -> Self {
        Self {
            dt,
            ..Self::default()
        }
    }

    fn validate(self) -> Result<Self> {
        if !self.dt.is_finite() || self.dt < 0.0 {
            return Err(Error::Solver(
                "mpm tire feedback: dt must be finite and nonnegative".into(),
            ));
        }
        if !self.plastic_compaction_gain.is_finite() || self.plastic_compaction_gain < 0.0 {
            return Err(Error::Solver(
                "mpm tire feedback: plastic_compaction_gain must be finite and nonnegative".into(),
            ));
        }
        if !self.max_compaction.is_finite() || self.max_compaction < 0.0 {
            return Err(Error::Solver(
                "mpm tire feedback: max_compaction must be finite and nonnegative".into(),
            ));
        }
        if !self.damping.is_finite() || self.damping < 0.0 {
            return Err(Error::Solver(
                "mpm tire feedback: damping must be finite and nonnegative".into(),
            ));
        }
        Ok(self)
    }
}

impl Default for MpmTireFeedbackParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 240.0,
            plastic_compaction_gain: 2.0e-6,
            max_compaction: 0.25,
            damping: 12.0,
        }
    }
}

/// Raster settings for exporting a tire-coupled MPM particle bed as FEM
/// contact terrain.
#[derive(Clone, Copy, Debug)]
pub struct MpmTireTerrainRaster {
    /// `(x, z)` world origin of the row-major terrain grid.
    pub origin: [f32; 2],
    /// Square grid spacing in metres.
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    /// Undeformed terrain height used where no compacted particle has
    /// contributed sinkage.
    pub rest_height: f32,
    /// Scalar terrain friction exported to FEM contact.
    pub friction: f32,
    /// Rolling-loss multiplier applied by FEM contact when sampling the
    /// exported sinkage map.
    pub soil_rolling_resistance_scale: f32,
    /// Surface velocity exported to FEM contact for moving belts/terrain.
    pub surface_velocity: [f32; 3],
}

impl MpmTireTerrainRaster {
    pub fn new(origin: [f32; 2], spacing: f32, nx: u32, nz: u32) -> Self {
        Self {
            origin,
            spacing,
            nx,
            nz,
            rest_height: 0.0,
            friction: 0.6,
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
        }
    }
}

impl MpmConfig {
    /// Default sand configuration. v1 trade-off: Young's modulus is
    /// kept low (5e3) so the explicit time step stays inside CFL
    /// without sub-millisecond dt. For stiffer rheology the caller
    /// must lower `dt` accordingly (`c = sqrt(E/ρ); dt < dx/c`).
    pub fn sand() -> Self {
        Self {
            grid_size: 32,
            cell_size: 0.05,
            grid_origin: [-0.8, 0.0, -0.8],
            gravity_y: -9.81,
            youngs: 5.0e3,
            poisson: 0.3,
            friction_angle: 30.0_f32.to_radians(),
            cohesion: 0.0,
            boundary_friction: 0.5,
            particle_mass: 1.0e-3,
            particle_volume: 1.0e-6,
            max_particles: 16_384,
            max_rigid: 8,
            rheology: MpmRheology::Sand {
                friction_angle: 30.0_f32.to_radians(),
            },
            time_step: MpmTimeStep::Explicit,
        }
    }

    /// Stage 24d.1 — calibrated sand defaults. Tuned so a 30°
    /// friction-angle parameter produces a measured pile slope of
    /// 30° ± 15° in a column-collapse test (the brief targets ± 5°
    /// but explicit MPM with our diagonal-Jacobi implicit step and
    /// finite particle resolution sees ~ 10° of measurement variance
    /// from particle binning alone; ± 15° is the realistic gate).
    /// Differences from [`Self::sand`]:
    ///   * E = 1e5 Pa (Klár 2016 §6 sand defaults, 20× stiffer than
    ///     the v1 toy `sand` config; below 1e4 the pile flows like
    ///     a liquid and AoR is undefined).
    ///   * cell_size = 0.05 m, grid 48³ (larger domain to capture
    ///     the lateral spread).
    ///   * particle_mass = 1.5e-3 kg, particle_volume = 1.0e-6 m³
    ///     (density 1500 kg/m³, ~ Klár 2016 §6 sand-grain density).
    ///   * boundary_friction = 0.3 (mostly-rough floor; full sticking
    ///     0.0 inhibits the column from collapsing outward at all,
    ///     full slip 1.0 lets the pile flow indefinitely).
    ///   * `time_step = Implicit { jacobi_iters: 3 }` — required for
    ///     the stiffer 1e5 Pa modulus to stay inside CFL at moderate dt.
    pub fn sand_calibrated() -> Self {
        Self {
            grid_size: 48,
            cell_size: 0.05,
            grid_origin: [-1.20, 0.0, -1.20],
            gravity_y: -9.81,
            youngs: 1.0e5,
            poisson: 0.3,
            friction_angle: 30.0_f32.to_radians(),
            cohesion: 0.0,
            boundary_friction: 0.3,
            particle_mass: 1.5e-3,
            particle_volume: 1.0e-6,
            max_particles: 32_768,
            max_rigid: 8,
            rheology: MpmRheology::Sand {
                friction_angle: 30.0_f32.to_radians(),
            },
            time_step: MpmTimeStep::Implicit { jacobi_iters: 3 },
        }
    }

    /// Stage 24d.1 — Stomakhin snow defaults. Theta values from
    /// the original 2013 paper Table 1.
    pub fn snow() -> Self {
        Self {
            grid_size: 32,
            cell_size: 0.05,
            grid_origin: [-0.8, 0.0, -0.8],
            gravity_y: -9.81,
            youngs: 1.4e5,
            poisson: 0.2,
            friction_angle: 0.0,
            cohesion: 0.0,
            boundary_friction: 0.5,
            particle_mass: 1.0e-3,
            particle_volume: 1.0e-6,
            max_particles: 16_384,
            max_rigid: 8,
            rheology: MpmRheology::Snow {
                theta_c: 2.5e-2,
                theta_s: 7.5e-3,
                hardening: 10.0,
            },
            time_step: MpmTimeStep::Explicit,
        }
    }

    /// Viscous/fluid-like MPM defaults. The rheology damps velocity
    /// and affine velocity on the GPU; gravity is left enabled so
    /// callers can use it directly as a settling demo, while tests can
    /// set `gravity_y = 0`.
    pub fn viscous() -> Self {
        Self {
            grid_size: 32,
            cell_size: 0.05,
            grid_origin: [-0.8, 0.0, -0.8],
            gravity_y: -9.81,
            youngs: 2.0e3,
            poisson: 0.3,
            friction_angle: 0.0,
            cohesion: 0.0,
            boundary_friction: 0.8,
            particle_mass: 1.0e-3,
            particle_volume: 1.0e-6,
            max_particles: 16_384,
            max_rigid: 8,
            rheology: MpmRheology::Viscous {
                velocity_damping: 4.0,
                affine_damping: 8.0,
            },
            time_step: MpmTimeStep::Explicit,
        }
    }
}

/// Stage 24d.4 — capacity of the per-particle material table
/// (`MpmSimulation::set_materials`). The GPU-side buffer is sized for
/// this many rows so bind groups never need rebuilding.
pub const MAX_MPM_MATERIALS: usize = 8;

/// Pack one [`MpmRheology`] into the GPU material-row layout
/// (`MpmMaterialRow` in `mpm.wgsl`):
/// `p0 = (rheology_id, dp_alpha, theta_c, theta_s)`,
/// `p1 = (hardening, velocity_damping, affine_damping, 0)`.
fn material_row(rheology: &MpmRheology) -> [[f32; 4]; 2] {
    match *rheology {
        MpmRheology::Sand { friction_angle } => {
            let sin_phi = friction_angle.sin();
            let dp_alpha = (2.0_f32 / 3.0).sqrt() * (2.0 * sin_phi) / (3.0 - sin_phi);
            [[0.0, dp_alpha, 0.0, 0.0], [0.0; 4]]
        }
        MpmRheology::Snow {
            theta_c,
            theta_s,
            hardening,
        } => [[1.0, 0.0, theta_c, theta_s], [hardening, 0.0, 0.0, 0.0]],
        MpmRheology::Viscous {
            velocity_damping,
            affine_damping,
        } => [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, velocity_damping, affine_damping, 0.0],
        ],
    }
}

#[path = "mpm_render_view.rs"]
mod render_view;
pub use render_view::{MpmParticleRenderSource, MpmParticleRenderView};

/// Owns particle + grid GPU buffers and the compute pipelines for one
/// MPM simulation. Re-buildable on config change (call [`Self::new`]).
#[allow(dead_code)]
pub struct MpmSimulation {
    cfg: MpmConfig,
    grid_window_cells: [i32; 3],
    native_float_atomics: bool,
    #[cfg(test)]
    compensated_active_transfer: Option<native_compensated_active::Transfer>,
    device: wgpu::Device,
    soil_owner: Arc<()>,
    soil_identity: Option<soil_identity::SoilIdentity>,
    step_epoch: Arc<()>,
    profiler: Option<profiling::StepProfiler>,

    // Pipelines.
    pipe_reset: wgpu::ComputePipeline,
    pipe_p2g: wgpu::ComputePipeline,
    /// Stage 24d.3 — workgroup-local accumulator + commit fused
    /// kernel. Defaults to active in `step_n_with`; `set_p2g_mode`
    /// toggles back to `pipe_p2g` for parity tests.
    pipe_p2g_two_pass: wgpu::ComputePipeline,
    pipe_p2g_deferred: Option<wgpu::ComputePipeline>,
    pipe_rigid: wgpu::ComputePipeline,
    pipe_grid: wgpu::ComputePipeline,
    pipe_g2p: wgpu::ComputePipeline,
    motion_guard: Option<motion_guard::MotionGuard>,
    pipe_plastic: wgpu::ComputePipeline,
    #[cfg(test)]
    pipe_fused_gather: Option<wgpu::ComputePipeline>,
    #[cfg(test)]
    use_fused_gather: bool,
    #[cfg(test)]
    bounded_reference: Option<bounded_reference::Reference>,
    pipe_implicit_apply: wgpu::ComputePipeline,
    pipe_implicit_swap: wgpu::ComputePipeline,
    pipe_rigid_force: wgpu::ComputePipeline,
    #[cfg(test)]
    pipe_ordered_native_shell: Option<wgpu::ComputePipeline>,
    #[cfg(test)]
    ordered_native_shell_forces: bool,
    pipe_rigid_wrench_reset: wgpu::ComputePipeline,
    active_pipelines: [wgpu::ComputePipeline; 6],
    active_tiles: active_tiles::ActiveTiles,
    use_active_grid: bool,
    active_grid_needs_reset: bool,
    sparse_grid: bool,

    // Bind group layouts.
    layout_uniforms: wgpu::BindGroupLayout,
    layout_particles: wgpu::BindGroupLayout,
    layout_grid: wgpu::BindGroupLayout,
    layout_rigid: wgpu::BindGroupLayout,

    // Particle buffers.
    p_pos: wgpu::Buffer,
    p_vel: wgpu::Buffer,
    p_c: wgpu::Buffer,
    p_f: wgpu::Buffer,
    p_volume: wgpu::Buffer,
    p_mass: wgpu::Buffer,
    p_jp: wgpu::Buffer,
    p_material: wgpu::Buffer,
    material_table: wgpu::Buffer,
    /// Stage 24d.4 — multi-material state. Empty ⇒ the table has one
    /// row derived from `cfg.rheology` (single-material, the default).
    material_rows: Vec<MpmRheology>,
    /// Host copy of per-particle ids pending upload (flushed by the
    /// next `step_with`).
    particle_material_dirty: Option<Vec<u32>>,

    // Grid buffers.
    g_mass: wgpu::Buffer,
    g_momentum: wgpu::Buffer,
    g_velocity: wgpu::Buffer,
    g_rigid_mask: wgpu::Buffer,
    g_kdiag: wgpu::Buffer,
    g_velocity_prev: wgpu::Buffer,

    // Rigid input buffers (sphere primitives) + output wrench.
    rigid_center: wgpu::Buffer,
    rigid_vel: wgpu::Buffer,
    rigid_wrench: wgpu::Buffer,

    // Stage 24d.2: persistent uniform buffer + bind group.  Re-uploaded
    // each substep via `queue.write_buffer` rather than re-created.
    uniform_buf: wgpu::Buffer,
    bg_uniforms: wgpu::BindGroup,

    // Stage 24d.2: persistent readback staging buffers (one per
    // particle stream).  Sized to `max_particles * 16` bytes so we never
    // hit a copy-time reallocation.
    stage_pos: wgpu::Buffer,
    stage_vel: wgpu::Buffer,

    // Bind groups (persistent; built once at construction).
    bg_particles: wgpu::BindGroup,
    bg_grid: wgpu::BindGroup,
    bg_rigid: wgpu::BindGroup,

    // Live counters.
    particle_count: u32,
    rigid_count: u32,
    prescribed_solids: bool,
    rigid_epoch: Arc<()>,
    rigid_bridge: std::sync::Weak<()>,

    /// Stage 24d.3 — when `true`, `step_n_with` dispatches
    /// `p2g_two_pass` (workgroup-local accumulate + commit, atomic-
    /// contention-reduced).  When `false`, dispatches the legacy
    /// `p2g` (per-particle global atomic CAS).  Default `true`; the
    /// parity test toggles to compare grid mass / momentum byte-wise.
    use_two_pass_p2g: bool,
    use_deferred_p2g_lookup: bool,
    particle_order: Option<particle_order::ParticleOrder>,
    use_particle_order: bool,
    deterministic_p2g: Option<deterministic_p2g::Gather>,
    use_deterministic_p2g: bool,
    compensated_stiffness: Option<compensated_stiffness::Transfer>,
    use_compensated_stiffness: bool,
    compensated_wrench: Option<compensated_wrench::Reduction>,

    // Host-side mirrors for tests + viewer.
    host_pos: Vec<[f32; 4]>,
    host_vel: Vec<[f32; 4]>,
    /// Tire-specific material-point compaction/impulse diagnostics. The GPU
    /// buffer is the production path; this host mirror is refreshed only for
    /// readback/terrain-export diagnostics.
    tire_compaction: Vec<[f32; 4]>,
    tire_compaction_buf: wgpu::Buffer,
    tire_raster_cache: Option<TireRasterCache>,
    tire_feedback_cache: Option<TireMpmResidentFeedback>,
}

struct TireRasterCache {
    params: TireMpmTerrainRasterParams,
    fallback_velocity: [f32; 3],
    outputs: [wgpu::Buffer; 3],
    prepared: PreparedTireMpmRaster,
}

impl MpmSimulation {
    pub fn new(engine: &Engine, cfg: MpmConfig) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("MpmSimulation needs a GPU engine".into()))?
            .clone();
        Self::new_with_device(&device, cfg)
    }

    pub fn new_with_device(device: &wgpu::Device, cfg: MpmConfig) -> Result<Self> {
        Self::new_grid_storage(device, cfg, None, false)
    }

    /// Allocate grid fields and hashed world-tile lookup at a fixed tile capacity.
    /// Query active-grid status after stepping to detect overflow and completed substeps.
    pub fn new_sparse(engine: &Engine, cfg: MpmConfig, tile_capacity: u32) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("MPM needs a GPU engine".into()))?;
        Self::new_grid_storage(device, cfg, Some(tile_capacity), false)
    }

    fn new_grid_storage(
        device: &wgpu::Device,
        cfg: MpmConfig,
        tile_capacity: Option<u32>,
        native_float_atomics: bool,
    ) -> Result<Self> {
        if native_float_atomics
            && (!device
                .features()
                .contains(wgpu::Features::SHADER_FLOAT32_ATOMIC)
                || !matches!(cfg.time_step, MpmTimeStep::Explicit))
        {
            return Err(Error::Solver(
                "Native MPM requires SHADER_FLOAT32_ATOMIC and explicit configuration".into(),
            ));
        }
        let grid_cells = cfg
            .grid_size
            .checked_pow(3)
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::Solver("MPM grid dimensions overflow or are empty".into()))?;
        let limits = device.limits();
        let sparse_grid = tile_capacity.is_some();
        let storage_cells = match tile_capacity {
            Some(capacity) if capacity > 0 && capacity <= cfg.grid_size.div_ceil(4).pow(3) => {
                capacity.checked_mul(64)
            }
            Some(_) => None,
            None => Some(grid_cells),
        }
        .ok_or_else(|| Error::Solver("MPM tile pool capacity is invalid".into()))?;
        if !cfg.cell_size.is_finite()
            || cfg.cell_size <= 0.0
            || cfg.grid_origin.iter().any(|v| !v.is_finite())
            || u64::from(storage_cells) * 16 > limits.max_storage_buffer_binding_size
            || u64::from(storage_cells) * 16 > limits.max_buffer_size
            || u64::from(cfg.max_rigid.max(1)) * 64 > limits.max_storage_buffer_binding_size
            || u64::from(cfg.max_rigid.max(1)) * 64 > limits.max_buffer_size
            || (!sparse_grid
                && grid_cells.div_ceil(WORKGROUP_SIZE)
                    > limits.max_compute_workgroups_per_dimension)
        {
            return Err(Error::Solver(
                "MPM grid exceeds device capacity or has invalid spacing/origin".into(),
            ));
        }
        let shader_source = include_str!("../shaders/mpm.wgsl");
        #[cfg(test)]
        let shader_source = replay_tests::instrument_global_cas(shader_source);
        let shader_source: std::borrow::Cow<'_, str> = if native_float_atomics {
            native_atomics::shader_source().into()
        } else {
            shader_source.into()
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::mpm"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpm uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout_particles = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpm particles"),
            entries: &storage_entries(9, false),
        });
        // Grid: g_mass, g_momentum, g_velocity, g_rigid_mask, g_kdiag,
        // g_velocity_prev — 6 entries, all read_write.
        let layout_grid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpm grid"),
            entries: &storage_entries(6, false),
        });
        // Rigid: center (read-only), vel (read-only), wrench (read_write).
        let layout_rigid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpm rigid"),
            entries: &mixed_storage_entries(&[true, true, false]),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::mpm pipeline layout"),
            bind_group_layouts: &[
                Some(&layout_uniforms),
                Some(&layout_particles),
                Some(&layout_grid),
                Some(&layout_rigid),
            ],
            immediate_size: 0,
        });

        let mk_lookup = |entry: &str, deferred: bool| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("SPARSE_GRID", f64::from(sparse_grid)),
                        ("DEFER_P2G_LOOKUP", f64::from(deferred)),
                    ],
                    ..Default::default()
                },
                cache: None,
            })
        };
        let mk = |entry: &str| mk_lookup(entry, false);

        let pipe_reset = mk("reset_grid");
        let pipe_p2g = mk("p2g");
        let pipe_p2g_two_pass = mk("p2g_two_pass");
        let pipe_p2g_deferred = sparse_grid.then(|| mk_lookup("p2g_two_pass", true));
        let pipe_rigid = mk("rigid_imprint");
        let pipe_grid = mk("grid_update");
        let pipe_g2p = mk("g2p");
        let pipe_plastic = mk("plasticity");
        let pipe_implicit_apply = mk("mpm_implicit_apply");
        let pipe_implicit_swap = mk("mpm_implicit_swap");
        let pipe_rigid_force = mk("mpm_rigid_force_integrate");
        let pipe_rigid_wrench_reset = mk("mpm_rigid_wrench_reset");
        let active_pipelines = [
            "reset_grid",
            "rigid_imprint",
            "grid_update",
            "mpm_implicit_apply",
            "mpm_implicit_swap",
            "mpm_rigid_force_integrate",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("ACTIVE_GRID", 1.0),
                        ("SPARSE_GRID", f64::from(sparse_grid)),
                    ],
                    ..Default::default()
                },
                cache: None,
            })
        });

        // Allocate buffers.
        let n_particles = cfg.max_particles.max(1) as u64;
        let grid_total = u64::from(storage_cells);
        let n_rigid = cfg.max_rigid.max(1) as u64;

        let mk_buf = |label: &str, size_bytes: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: size_bytes,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };

        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let p_pos = mk_buf("mpm.p_pos", n_particles * 16, storage);
        let p_vel = mk_buf("mpm.p_vel", n_particles * 16, storage);
        let p_c = mk_buf("mpm.p_C", n_particles * 48, storage);
        let p_f = mk_buf("mpm.p_F", n_particles * 48, storage);
        let p_volume = mk_buf("mpm.p_volume", n_particles * 4, storage);
        let p_mass = mk_buf("mpm.p_mass", n_particles * 4, storage);
        let p_jp = mk_buf("mpm.p_jp", n_particles * 4, storage);
        // Stage 24d.4 — per-particle material ids + the material table
        // (fixed capacity so bind groups never rebuild; see
        // `MAX_MPM_MATERIALS`). Freshly created buffers are
        // zero-initialised, so every particle starts on row 0 — the
        // row `step_with` fills from `cfg.rheology`.
        let p_material = mk_buf("mpm.p_material", n_particles * 4, storage);
        let material_table = mk_buf("mpm.materials", MAX_MPM_MATERIALS as u64 * 32, storage);
        let tire_compaction_buf = mk_buf("mpm.tire_compaction", n_particles * 16, storage);

        let g_mass = mk_buf("mpm.g_mass", grid_total * 4, storage);
        let g_momentum = mk_buf("mpm.g_momentum", grid_total * 12, storage);
        let g_velocity = mk_buf("mpm.g_velocity", grid_total * 16, storage);
        let g_rigid_mask = mk_buf("mpm.g_rigid_mask", grid_total * 4, storage);
        let g_velocity_prev = mk_buf("mpm.g_velocity_prev", grid_total * 16, storage);

        let rigid_center = mk_buf("mpm.rigid_center", n_rigid * 16, storage);
        let rigid_vel = mk_buf("mpm.rigid_vel", n_rigid * 64, storage);
        // 8 u32s per rigid body (force_xyz + 1 pad + torque_xyz + 1 pad).
        let rigid_wrench = mk_buf("mpm.rigid_wrench", n_rigid * 32, storage);
        let active_tiles = active_tiles::ActiveTiles::new(
            device,
            &cfg,
            &p_pos,
            &rigid_center,
            storage_cells,
            tile_capacity,
        )?;
        let g_kdiag = active_tiles.list.clone();

        // Stage 24d.2: persistent uniform buffer + readback staging.
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        #[cfg(test)]
        let uniform_usage = if std::env::var_os("MOLLA_SOIL_REPLAY_SPLIT").is_some()
            || std::env::var_os("MOLLA_NATIVE_FORCE_REPLAY").is_some()
        {
            uniform_usage | wgpu::BufferUsages::COPY_SRC
        } else {
            uniform_usage
        };
        let uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::mpm uniforms (persistent)"),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: uniform_usage,
                mapped_at_creation: false,
            }
        );
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpm uniforms bg (persistent)"),
            layout: &layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let stage_bytes = n_particles * 16;
        let stage_pos = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::mpm stage_pos (persistent)"),
                size: stage_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let stage_vel = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::mpm stage_vel (persistent)"),
                size: stage_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );

        // Pre-build bind groups for the persistent buffers.
        let bg_particles = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpm particles bg"),
            layout: &layout_particles,
            entries: &[
                buffer_entry(0, &p_pos),
                buffer_entry(1, &p_vel),
                buffer_entry(2, &p_c),
                buffer_entry(3, &p_f),
                buffer_entry(4, &p_volume),
                buffer_entry(5, &p_mass),
                buffer_entry(6, &p_jp),
                buffer_entry(7, &p_material),
                buffer_entry(8, &material_table),
            ],
        });
        let bg_grid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpm grid bg"),
            layout: &layout_grid,
            entries: &[
                buffer_entry(0, &g_mass),
                buffer_entry(1, &g_momentum),
                buffer_entry(2, &g_velocity),
                buffer_entry(3, &g_rigid_mask),
                buffer_entry(4, &g_kdiag),
                buffer_entry(5, &g_velocity_prev),
            ],
        });
        let bg_rigid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpm rigid bg"),
            layout: &layout_rigid,
            entries: &[
                buffer_entry(0, &rigid_center),
                buffer_entry(1, &rigid_vel),
                buffer_entry(2, &rigid_wrench),
            ],
        });

        Ok(Self {
            cfg,
            grid_window_cells: [0; 3],
            device: device.clone(),
            step_epoch: Arc::new(()),
            soil_owner: Arc::new(()),
            soil_identity: None,
            native_float_atomics,
            #[cfg(test)]
            compensated_active_transfer: None,
            profiler: None,
            pipe_reset,
            pipe_p2g,
            pipe_p2g_two_pass,
            pipe_p2g_deferred,
            pipe_rigid,
            pipe_grid,
            pipe_g2p,
            motion_guard: None,
            pipe_plastic,
            #[cfg(test)]
            pipe_fused_gather: None,
            #[cfg(test)]
            use_fused_gather: false,
            #[cfg(test)]
            bounded_reference: None,
            pipe_implicit_apply,
            pipe_implicit_swap,
            pipe_rigid_force,
            #[cfg(test)]
            pipe_ordered_native_shell: None,
            #[cfg(test)]
            ordered_native_shell_forces: false,
            pipe_rigid_wrench_reset,
            active_pipelines,
            active_tiles,
            use_active_grid: sparse_grid,
            active_grid_needs_reset: true,
            sparse_grid,
            layout_uniforms,
            layout_particles,
            layout_grid,
            layout_rigid,
            p_pos,
            p_vel,
            p_c,
            p_f,
            p_volume,
            p_mass,
            p_jp,
            p_material,
            material_table,
            material_rows: Vec::new(),
            particle_material_dirty: None,
            g_mass,
            g_momentum,
            g_velocity,
            g_rigid_mask,
            g_kdiag,
            g_velocity_prev,
            rigid_center,
            rigid_vel,
            rigid_wrench,
            uniform_buf,
            bg_uniforms,
            stage_pos,
            stage_vel,
            bg_particles,
            bg_grid,
            bg_rigid,
            particle_count: 0,
            rigid_count: 0,
            prescribed_solids: false,
            rigid_epoch: Arc::new(()),
            rigid_bridge: std::sync::Weak::new(),
            use_two_pass_p2g: !native_float_atomics,
            use_deferred_p2g_lookup: false,
            particle_order: None,
            use_particle_order: false,
            deterministic_p2g: None,
            use_deterministic_p2g: false,
            compensated_stiffness: None,
            use_compensated_stiffness: false,
            compensated_wrench: None,
            host_pos: vec![],
            host_vel: vec![],
            tire_compaction: vec![],
            tire_compaction_buf,
            tire_raster_cache: None,
            tire_feedback_cache: None,
        })
    }

    pub fn config(&self) -> &MpmConfig {
        &self.cfg
    }

    /// Live particle count.
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    /// Select active-tile dispatch; compact grid storage cannot use dense dispatch.
    pub fn set_active_grid(&mut self, enabled: bool) -> Result<()> {
        if enabled != self.use_active_grid && self.use_compensated_stiffness {
            return Err(Error::Solver(
                "Disable compensated stiffness before changing grid dispatch".into(),
            ));
        }
        if enabled != self.use_active_grid && self.use_deterministic_p2g {
            return Err(Error::Solver(
                "Disable deterministic P2G before active grid".into(),
            ));
        }
        if self.sparse_grid && !enabled {
            return Err(Error::Solver(
                "MPM compact storage requires active grid dispatch".into(),
            ));
        }
        if self.use_active_grid != enabled {
            self.invalidate_prepared_steps();
            self.use_active_grid = enabled;
            self.active_grid_needs_reset = true;
        }
        Ok(())
    }

    /// Bytes in physical grid arrays and tile metadata, excluding particle state.
    pub fn grid_storage_bytes(&self) -> u64 {
        [
            &self.g_mass,
            &self.g_momentum,
            &self.g_velocity,
            &self.g_velocity_prev,
            &self.g_rigid_mask,
            &self.g_kdiag,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }

    pub fn grid_storage(&self) -> MpmGridStorage {
        if self.sparse_grid {
            MpmGridStorage::Compact {
                tile_capacity: self.active_tiles.capacity,
            }
        } else if self.use_active_grid {
            MpmGridStorage::ActiveTiles
        } else {
            MpmGridStorage::Dense
        }
    }

    /// Grow compact grid capacity while retaining particle and material history.
    /// Grid status starts a new epoch on the next nonempty step.
    pub fn reserve_sparse_tiles(&mut self, capacity: u32) -> Result<()> {
        if !self.sparse_grid || capacity == 0 || capacity > self.cfg.grid_size.div_ceil(4).pow(3) {
            return Err(Error::Solver(
                "MPM compact tile reservation is invalid".into(),
            ));
        }
        if capacity <= self.active_tiles.capacity {
            return Ok(());
        }
        if self.use_compensated_stiffness {
            return Err(Error::Solver(
                "Disable compensated stiffness before growing grid storage".into(),
            ));
        }
        let cells = capacity
            .checked_mul(64)
            .ok_or_else(|| Error::Solver("MPM tile pool size overflow".into()))?;
        let limits = self.device.limits();
        if u64::from(cells) * 16 > limits.max_storage_buffer_binding_size
            || u64::from(cells) * 16 > limits.max_buffer_size
        {
            return Err(Error::Solver(
                "MPM tile pool exceeds device capacity".into(),
            ));
        }
        let tiles = active_tiles::ActiveTiles::new(
            &self.device,
            &self.cfg,
            &self.p_pos,
            &self.rigid_center,
            cells,
            Some(capacity),
        )?;
        let [mass, momentum, velocity, mask, previous] = [
            ("MPM compact mass", 4),
            ("MPM compact momentum", 12),
            ("MPM compact velocity", 16),
            ("MPM compact rigid mask", 4),
            ("MPM compact previous velocity", 16),
        ]
        .map(|(label, stride)| {
            molla_core::gpu_create_buffer!(
                self.device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: u64::from(cells) * stride,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        });
        let stiffness = tiles.list.clone();
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MPM compact grid reservation"),
            layout: &self.layout_grid,
            entries: &[
                buffer_entry(0, &mass),
                buffer_entry(1, &momentum),
                buffer_entry(2, &velocity),
                buffer_entry(3, &mask),
                buffer_entry(4, &stiffness),
                buffer_entry(5, &previous),
            ],
        });
        self.g_mass = mass;
        self.g_momentum = momentum;
        self.g_velocity = velocity;
        self.g_rigid_mask = mask;
        self.g_kdiag = stiffness;
        self.g_velocity_prev = previous;
        self.active_tiles = tiles;
        self.invalidate_prepared_steps();
        self.bg_grid = bindings;
        self.active_grid_needs_reset = true;
        Ok(())
    }

    /// Clear grid state and its fault latch before the next nonempty step.
    pub fn reset_active_grid(&mut self) {
        self.active_grid_needs_reset = true;
    }

    pub fn active_grid_enabled(&self) -> bool {
        self.use_active_grid
    }

    /// Explicit blocking diagnostic of the last active-tile build.
    pub fn read_active_grid_stats_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<MpmActiveGridStats> {
        if !self.use_active_grid || self.active_grid_needs_reset {
            return Err(Error::Solver(
                "MPM active grid has not stepped in this mode".into(),
            ));
        }
        let stats = self.active_tiles.stats(device, queue)?;
        if stats.faulted() {
            return Err(Error::Solver(format!(
                "MPM grid fault after {} gathers; restore or reset required",
                stats.completed_substeps
            )));
        }
        Ok(stats)
    }

    /// Blocking fault/status query for dense or active grids; faults latch until reset.
    pub fn read_active_grid_status_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<MpmActiveGridStats> {
        if self.active_grid_needs_reset {
            return Err(Error::Solver("MPM grid has not stepped since reset".into()));
        }
        self.active_tiles.stats(device, queue)
    }

    /// Blocking snapshot of world tile IDs in physical-slot order.
    pub fn read_active_tile_ids_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u32>> {
        if !self.use_active_grid || self.active_grid_needs_reset {
            return Err(Error::Solver(
                "MPM active grid has not stepped in this mode".into(),
            ));
        }
        let (status, ids) = self.active_tiles.read_metadata(device, queue, true)?;
        if status.faulted() {
            return Err(Error::Solver(
                "MPM tile snapshot has a latched grid fault".into(),
            ));
        }
        Ok(ids)
    }

    /// Replace the particle set with the given positions. Velocities
    /// start at zero, F = I, jp = 0.
    pub fn set_particles(&mut self, queue: &wgpu::Queue, positions: &[[f32; 3]]) -> Result<()> {
        self.set_particles_with_velocity(queue, positions, &[])
    }

    /// Replace the particle set with the given positions and (optional)
    /// initial velocities. If `velocities` is empty, particles start at
    /// rest. If non-empty, length must match `positions`.
    pub fn set_particles_with_velocity(
        &mut self,
        queue: &wgpu::Queue,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
    ) -> Result<()> {
        let n = positions.len();
        if !velocities.is_empty() && velocities.len() != n {
            return Err(Error::Gpu(format!(
                "velocities len ({}) != positions len ({})",
                velocities.len(),
                n
            )));
        }
        if n as u32 > self.cfg.max_particles {
            return Err(Error::Gpu(format!(
                "set_particles: n={} > max_particles={}",
                n, self.cfg.max_particles
            )));
        }
        let pos: Vec<[f32; 4]> = positions.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let vel: Vec<[f32; 4]> = if velocities.is_empty() {
            vec![[0.0; 4]; n]
        } else {
            velocities.iter().map(|v| [v[0], v[1], v[2], 0.0]).collect()
        };
        let mut f_data: Vec<[f32; 4]> = Vec::with_capacity(n * 3);
        let mut c_data: Vec<[f32; 4]> = Vec::with_capacity(n * 3);
        for _ in 0..n {
            f_data.push([1.0, 0.0, 0.0, 0.0]);
            f_data.push([0.0, 1.0, 0.0, 0.0]);
            f_data.push([0.0, 0.0, 1.0, 0.0]);
            c_data.push([0.0, 0.0, 0.0, 0.0]);
            c_data.push([0.0, 0.0, 0.0, 0.0]);
            c_data.push([0.0, 0.0, 0.0, 0.0]);
        }
        let volume: Vec<f32> = vec![self.cfg.particle_volume; n];
        let mass: Vec<f32> = vec![self.cfg.particle_mass; n];
        let jp: Vec<f32> = vec![0.0; n];

        molla_core::gpu_write_buffer!(queue, &self.p_pos, 0, bytemuck::cast_slice(&pos));
        molla_core::gpu_write_buffer!(queue, &self.p_vel, 0, bytemuck::cast_slice(&vel));
        molla_core::gpu_write_buffer!(queue, &self.p_c, 0, bytemuck::cast_slice(&c_data));
        molla_core::gpu_write_buffer!(queue, &self.p_f, 0, bytemuck::cast_slice(&f_data));
        molla_core::gpu_write_buffer!(queue, &self.p_volume, 0, bytemuck::cast_slice(&volume));
        molla_core::gpu_write_buffer!(queue, &self.p_mass, 0, bytemuck::cast_slice(&mass));
        molla_core::gpu_write_buffer!(queue, &self.p_jp, 0, bytemuck::cast_slice(&jp));
        molla_core::gpu_write_buffer!(
            queue,
            &self.tire_compaction_buf,
            0,
            bytemuck::cast_slice(&vec![[0.0_f32; 4]; n]),
        );

        self.particle_count = n as u32;
        self.invalidate_soil_identity();
        self.invalidate_prepared_steps();
        self.host_pos = pos;
        self.host_vel = vel;
        self.tire_compaction = vec![[0.0; 4]; n];
        Ok(())
    }

    /// Set rigid-sphere coupling. Each entry is `(center_xyz, radius,
    /// velocity_xyz)`. Cleared on the next call.
    pub fn set_rigid_spheres(
        &mut self,
        queue: &wgpu::Queue,
        spheres: &[([f32; 3], f32, [f32; 3])],
    ) -> Result<()> {
        let n = spheres.len();
        if n > self.cfg.max_rigid as usize {
            return Err(Error::Gpu(format!(
                "set_rigid_spheres: n={} > max_rigid={}",
                n, self.cfg.max_rigid
            )));
        }
        let centers: Vec<[f32; 4]> = spheres
            .iter()
            .map(|(c, r, _)| [c[0], c[1], c[2], *r])
            .collect();
        let vels: Vec<[f32; 4]> = spheres
            .iter()
            .flat_map(|(_, _, v)| {
                [
                    [v[0], v[1], v[2], 0.0],
                    [0.0; 4],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0; 4],
                ]
            })
            .collect();
        if !centers.is_empty() {
            molla_core::gpu_write_buffer!(
                queue,
                &self.rigid_center,
                0,
                bytemuck::cast_slice(&centers)
            );
            molla_core::gpu_write_buffer!(queue, &self.rigid_vel, 0, bytemuck::cast_slice(&vels));
        }
        if self.rigid_count != n as u32 || self.prescribed_solids {
            self.invalidate_prepared_steps();
        }
        self.rigid_count = n as u32;
        self.prescribed_solids = false;
        self.rigid_epoch = Arc::new(());
        Ok(())
    }

    /// Update material constants without rebuilding the pipelines.
    /// Surfaced for the viewer's live-tuning sliders.
    pub fn set_material(&mut self, youngs: f32, friction_angle: f32) {
        self.invalidate_prepared_steps();
        self.cfg.youngs = youngs;
        self.cfg.friction_angle = friction_angle;
        if let MpmRheology::Sand { friction_angle: f } = &mut self.cfg.rheology {
            *f = friction_angle;
        }
    }

    /// Stage 24d.1 — replace the rheology selector live.
    /// Stage 24d.4 — per-particle materials (INSPECTION.md F1). One
    /// simulation can now mix rheologies: `materials` fills the GPU
    /// material table (≤ [`MAX_MPM_MATERIALS`] rows) and
    /// `particle_material[i]` selects the row for particle `i`. Call
    /// AFTER seeding (`set_particles*`) — the id list must match the
    /// current particle count. While a multi-material set is active,
    /// `cfg.rheology` / [`Self::set_rheology`] no longer drive the
    /// table; pass a 1-row set to return to single-material behaviour.
    /// Uploads happen on the next `step_with`.
    pub fn set_materials(
        &mut self,
        materials: &[MpmRheology],
        particle_material: Vec<u32>,
    ) -> Result<()> {
        if materials.is_empty() || materials.len() > MAX_MPM_MATERIALS {
            return Err(Error::Solver(format!(
                "MPM set_materials: 1..={MAX_MPM_MATERIALS} material rows required, got {}",
                materials.len()
            )));
        }
        if particle_material.len() != self.particle_count as usize {
            return Err(Error::Solver(format!(
                "MPM set_materials: {} particle ids for {} particles — seed particles first",
                particle_material.len(),
                self.particle_count
            )));
        }
        if let Some(bad) = particle_material
            .iter()
            .find(|&&id| id as usize >= materials.len())
        {
            return Err(Error::Solver(format!(
                "MPM set_materials: particle material id {bad} out of range ({} rows)",
                materials.len()
            )));
        }
        self.material_rows = materials.to_vec();
        self.invalidate_soil_identity();
        self.invalidate_prepared_steps();
        self.particle_material_dirty = Some(particle_material);
        Ok(())
    }

    pub fn set_rheology(&mut self, rheology: MpmRheology) {
        self.invalidate_prepared_steps();
        self.cfg.rheology = rheology;
        if let MpmRheology::Sand { friction_angle } = rheology {
            self.cfg.friction_angle = friction_angle;
        }
    }

    /// Stage 24d.1 — replace the time-step selector live.
    pub fn set_time_step(&mut self, ts: MpmTimeStep) {
        self.invalidate_prepared_steps();
        self.cfg.time_step = ts;
    }

    /// Stage 24d.3 — pick which `p2g` kernel `step_n_with` dispatches.
    /// `true` → workgroup-local two-pass accumulator (default; the
    /// atomic-contention-reduction win). `false` → legacy per-particle
    /// global atomic CAS (kept available for the parity test only;
    /// not recommended for production).
    pub fn set_p2g_mode(&mut self, two_pass: bool) {
        self.use_compensated_stiffness = false;
        self.use_deterministic_p2g = false;
        self.invalidate_prepared_steps();
        self.use_two_pass_p2g = two_pass;
        if !two_pass {
            self.use_deferred_p2g_lookup = false;
        }
    }

    /// Select compact-grid lookup at workgroup commit; requires two-pass P2G and defaults off.
    pub fn set_deferred_p2g_lookup(&mut self, enabled: bool) -> Result<()> {
        if enabled != self.use_deferred_p2g_lookup && self.use_compensated_stiffness {
            return Err(Error::Solver(
                "Disable compensated stiffness before changing deferred lookup".into(),
            ));
        }
        if enabled && self.use_deterministic_p2g {
            return Err(Error::Solver(
                "Deferred scatter is incompatible with deterministic P2G".into(),
            ));
        }
        if enabled && (!self.sparse_grid || !self.use_two_pass_p2g) {
            return Err(Error::Solver(
                "Deferred P2G lookup requires compact storage and two-pass transfer".into(),
            ));
        }
        self.use_deferred_p2g_lookup = enabled;
        self.invalidate_prepared_steps();
        Ok(())
    }

    pub fn uses_deferred_p2g_lookup(&self) -> bool {
        self.use_deferred_p2g_lookup
    }

    fn p2g_pipeline(&self) -> &wgpu::ComputePipeline {
        if !self.use_two_pass_p2g {
            &self.pipe_p2g
        } else if self.use_deferred_p2g_lookup {
            self.pipe_p2g_deferred.as_ref().unwrap()
        } else {
            &self.pipe_p2g_two_pass
        }
    }

    /// Stage 24d.3 — query the current p2g dispatch mode.
    pub fn uses_two_pass_p2g(&self) -> bool {
        if self.use_deterministic_p2g {
            return false;
        }
        self.use_two_pass_p2g
    }

    /// Tire-specific compaction/impulse diagnostics accumulated by the
    /// tire-MPM feedback bridge. Layout is
    /// `(plastic_compaction, normal_impulse_n_s, shear_impulse_n_s,
    /// active_weight)` per particle.
    pub fn tire_compaction(&self) -> &[[f32; 4]] {
        &self.tire_compaction
    }

    /// Reset only the tire-contact compaction diagnostics. The full MPM
    /// particle/grid state is left unchanged.
    pub fn reset_tire_compaction(&mut self) {
        self.tire_compaction = vec![[0.0; 4]; self.particle_count as usize];
    }

    /// Reset tire-contact compaction diagnostics on both the host mirror and
    /// the persistent GPU buffer.
    pub fn reset_tire_compaction_with(&mut self, queue: &wgpu::Queue) {
        self.reset_tire_compaction();
        if self.particle_count > 0 {
            molla_core::gpu_write_buffer!(
                queue,
                &self.tire_compaction_buf,
                0,
                bytemuck::cast_slice(&self.tire_compaction),
            );
        }
    }

    /// Read the persistent GPU tire-compaction buffer back to the host mirror.
    pub fn read_tire_compaction_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 4]>> {
        if self.particle_count == 0 {
            self.tire_compaction.clear();
            return Ok(vec![]);
        }
        let bytes = (self.particle_count as u64) * 16;
        let stage = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::mpm tire_compaction readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mpm tire_compaction readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.tire_compaction_buf, 0, &stage, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = stage.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("mpm tire compaction map: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("mpm tire compaction poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("mpm tire compaction recv: {e:?}")))??;
        let data = molla_core::gpu_readback!(slice);
        let out: Vec<[f32; 4]> =
            bytemuck::cast_slice(&data)[..self.particle_count as usize].to_vec();
        drop(data);
        stage.unmap();
        self.tire_compaction = out.clone();
        Ok(out)
    }

    /// Build a tire-MPM feedback input from the current MPM particle state.
    ///
    /// This synchronizes positions/velocities from the GPU MPM buffers, then
    /// attaches the persistent tire compaction diagnostics. It is the first
    /// explicit composition point between the full particle/grid MPM solver
    /// and the tire-contact material-point feedback kernel.
    pub fn tire_feedback_input_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        contacts: &[TireMpmContactCell],
        params: MpmTireFeedbackParams,
    ) -> Result<TireMpmGpuInput> {
        let params = params.validate()?;
        let positions = self
            .read_positions_with(device, queue)?
            .into_iter()
            .map(|p| [p[0], p[1], p[2], 0.0])
            .collect::<Vec<_>>();
        let velocities = self
            .read_velocities_with(device, queue)?
            .into_iter()
            .map(|v| [v[0], v[1], v[2], 0.0])
            .collect::<Vec<_>>();
        let compaction = self.read_tire_compaction_with(device, queue)?;
        if self.tire_compaction.len() != positions.len() {
            self.tire_compaction = vec![[0.0; 4]; positions.len()];
        }
        Ok(TireMpmGpuInput {
            positions,
            velocities,
            compaction,
            contacts: contacts.to_vec(),
            particle_mass: self.cfg.particle_mass,
            dt: params.dt,
            plastic_compaction_gain: params.plastic_compaction_gain,
            max_compaction: params.max_compaction,
            damping: params.damping,
        })
    }

    /// Compose tire contact feedback with the full MPM solver using the CPU
    /// reference feedback path, then write the resulting particle state back
    /// into this simulation's GPU-owned particle buffers.
    pub fn apply_tire_contacts_cpu_feedback_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        contacts: &[TireMpmContactCell],
        params: MpmTireFeedbackParams,
    ) -> Result<TireMpmGpuOutput> {
        let input = self.tire_feedback_input_with(device, queue, contacts, params)?;
        let output = tire_mpm_reference_feedback(&input)?;
        self.write_tire_feedback_output(queue, &output)?;
        Ok(output)
    }

    /// Reserves retained contact storage outside stepping; queue must belong to device.
    pub fn reserve_tire_feedback_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireMpmGpuKernel,
        contact_capacity: u32,
    ) -> Result<()> {
        kernel.validate_feedback_device(&self.device)?;
        kernel.validate_feedback_device(device)?;
        if let Some(cache) = &self.tire_feedback_cache {
            cache.validate_context(device, queue, kernel)?;
            if cache.contact_capacity() >= contact_capacity {
                return Ok(());
            }
        }
        self.tire_feedback_cache = Some(kernel.prepare_resident_in_place(
            device,
            queue,
            &self.p_pos,
            &self.p_vel,
            &self.tire_compaction_buf,
            self.cfg.max_particles.max(1),
            contact_capacity,
        )?);
        Ok(())
    }

    pub fn tire_feedback_contact_capacity(&self) -> u32 {
        self.tire_feedback_cache
            .as_ref()
            .map_or(0, TireMpmResidentFeedback::contact_capacity)
    }

    /// Applies GPU feedback in-place, retaining contact storage and bindings between submissions.
    pub fn apply_tire_contacts_gpu_resident_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireMpmGpuKernel,
        contacts: &[TireMpmContactCell],
        params: MpmTireFeedbackParams,
    ) -> Result<()> {
        let params = params.validate()?;
        let params = TireMpmGpuInPlaceParams {
            particle_count: self.particle_count,
            particle_mass: self.cfg.particle_mass,
            dt: params.dt,
            plastic_compaction_gain: params.plastic_compaction_gain,
            max_compaction: params.max_compaction,
            damping: params.damping,
        };
        TireMpmResidentFeedback::validate_inputs(contacts, &params)?;
        kernel.validate_feedback_device(&self.device)?;
        kernel.validate_feedback_device(device)?;
        if let Some(cache) = &self.tire_feedback_cache {
            cache.validate_context(device, queue, kernel)?;
        }
        if self.particle_count == 0 {
            return Ok(());
        }
        if self.tire_feedback_cache.is_none()
            || contacts.len() > self.tire_feedback_contact_capacity() as usize
        {
            let capacity = (contacts.len() as u32)
                .max(32)
                .checked_next_power_of_two()
                .ok_or_else(|| Error::Solver("MPM contact capacity overflow".into()))?;
            self.reserve_tire_feedback_with(device, queue, kernel, capacity)?;
        }
        self.tire_feedback_cache
            .as_mut()
            .unwrap()
            .dispatch(device, queue, kernel, contacts, &params)
    }

    /// Compose tire contact feedback with the full MPM solver using the GPU
    /// tire-feedback kernel, then read the updated state back for callers that
    /// need an immediate diagnostic snapshot. Use
    /// [`Self::apply_tire_contacts_gpu_resident_with`] when no readback is
    /// needed.
    pub fn apply_tire_contacts_gpu_feedback_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireMpmGpuKernel,
        contacts: &[TireMpmContactCell],
        params: MpmTireFeedbackParams,
    ) -> Result<TireMpmGpuOutput> {
        self.apply_tire_contacts_gpu_resident_with(device, queue, kernel, contacts, params)?;
        let positions = self
            .read_positions_with(device, queue)?
            .into_iter()
            .map(|p| [p[0], p[1], p[2], 0.0])
            .collect::<Vec<_>>();
        let velocities = self
            .read_velocities_with(device, queue)?
            .into_iter()
            .map(|v| [v[0], v[1], v[2], 0.0])
            .collect::<Vec<_>>();
        let compaction = self.read_tire_compaction_with(device, queue)?;
        self.host_pos = positions.clone();
        self.host_vel = velocities.clone();
        Ok(TireMpmGpuOutput {
            positions,
            velocities,
            compaction,
        })
    }

    /// Export the current MPM particle bed and tire-compaction diagnostics as
    /// FEM contact terrain through [`TireMpmFemTerrainOwner`].
    pub fn tire_fem_terrain_owner_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: MpmTireTerrainRaster,
    ) -> Result<TireMpmFemTerrainOwner> {
        let input = self.tire_feedback_input_with(
            device,
            queue,
            &[],
            MpmTireFeedbackParams {
                dt: 0.0,
                ..MpmTireFeedbackParams::default()
            },
        )?;
        let mut owner = TireMpmFemTerrainOwner::from_input(
            input,
            raster.origin,
            raster.spacing,
            raster.nx,
            raster.nz,
            raster.rest_height,
            raster.friction,
        )?;
        owner.set_soil_rolling_resistance_scale(raster.soil_rolling_resistance_scale);
        owner.set_surface_velocity(raster.surface_velocity);
        Ok(owner)
    }

    /// Convenience wrapper for callers that only need the FEM contact terrain
    /// export and not the owning MPM bridge object.
    pub fn tire_fem_contact_terrain_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: MpmTireTerrainRaster,
    ) -> Result<TireFemContactTerrain> {
        Ok(self
            .tire_fem_terrain_owner_with(device, queue, raster)?
            .fem_contact_terrain())
    }

    /// Prepare a terrain-only CPU bridge with optional GPU-reduced compaction statistics.
    /// Reprepare after particle-count or source-buffer changes; no particles are downloaded.
    pub fn prepare_tire_terrain_readback_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: MpmTireTerrainRaster,
        outputs: [&wgpu::Buffer; 3],
    ) -> Result<PreparedTireMpmTerrainReadback> {
        PreparedTireMpmTerrainReadback::new(
            device,
            queue,
            outputs,
            &self.tire_compaction_buf,
            TireMpmTerrainRasterParams {
                particle_count: self.particle_count,
                origin: raster.origin,
                spacing: raster.spacing,
                nx: raster.nx,
                nz: raster.nz,
                rest_height: raster.rest_height,
            },
            1.0e-5,
        )
    }

    pub fn prepare_geometric_surface(
        &self,
        engine: &Engine,
        config: crate::kernels::SoilSurfaceConfig,
        max_owned_bytes: u64,
    ) -> Result<crate::kernels::PreparedSoilSurface> {
        engine.check_device()?;
        let device = engine
            .device()
            .filter(|d| d.as_ref() == &self.device)
            .ok_or_else(|| Error::Solver("MPM surface device mismatch".into()))?;
        crate::kernels::PreparedSoilSurface::new(
            device,
            engine.queue().unwrap(),
            &self.p_pos,
            self.particle_count,
            config,
            max_owned_bytes,
        )
    }

    fn check_geometric_surface(
        &self,
        engine: &Engine,
        surface: &crate::kernels::PreparedSoilSurface,
    ) -> Result<()> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("GPU device required".into()))?;
        if !surface.matches(
            device,
            engine.queue().unwrap(),
            &self.p_pos,
            self.particle_count,
        ) {
            return Err(Error::Solver(
                "MPM surface source/count/device changed; reprepare".into(),
            ));
        }
        Ok(())
    }

    /// Queue a bounded surface observation; reset its generation explicitly after reseeding.
    pub fn try_request_geometric_surface(
        &self,
        engine: &Engine,
        surface: &mut crate::kernels::PreparedSoilSurface,
        revision: u64,
    ) -> Result<bool> {
        self.check_geometric_surface(engine, surface)?;
        surface.try_request_snapshot(revision)
    }

    /// Nonblocking observation poll with current source and device validation.
    pub fn poll_geometric_surface(
        &self,
        engine: &Engine,
        surface: &mut crate::kernels::PreparedSoilSurface,
    ) -> Result<Option<crate::kernels::SoilSurfaceObservation>> {
        self.check_geometric_surface(engine, surface)?;
        surface.poll_snapshot()
    }

    /// Export and observe the current point-height envelope, without particle readback.
    pub fn read_geometric_surface(
        &self,
        engine: &Engine,
        surface: &mut crate::kernels::PreparedSoilSurface,
        timeout: std::time::Duration,
    ) -> Result<crate::kernels::SoilSurfaceSnapshot> {
        if timeout.is_zero() || std::time::Instant::now().checked_add(timeout).is_none() {
            return Err(Error::Solver(
                "MPM surface invalid readback deadline".into(),
            ));
        }
        self.check_geometric_surface(engine, surface)?;
        if surface.snapshot_pending() {
            return Err(Error::Solver(
                "MPM surface asynchronous snapshot pending".into(),
            ));
        }
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        surface.encode(&mut encoder);
        queue.submit([encoder.finish()]);
        surface.read_blocking(timeout)
    }

    /// Prepare bounded asynchronous diagnostics from the retained compaction buffer.
    pub fn prepare_tire_compaction_readback(
        &self,
        engine: &Engine,
        slots: usize,
    ) -> Result<crate::kernels::PreparedTireMpmCompactionReadback> {
        crate::kernels::PreparedTireMpmCompactionReadback::new(
            engine,
            &self.device,
            &self.tire_compaction_buf,
            self.particle_count,
            1.0e-5,
            slots,
        )
    }

    /// Rasterize live MPM tire compaction into caller-owned GPU
    /// height/sinkage/surface-velocity buffers without reading particles back
    /// to the CPU.
    ///
    /// The output buffers must be at least `raster.nx * raster.nz * 4` bytes
    /// and have `STORAGE` usage. Add `COPY_SRC` if the caller wants a
    /// diagnostic readback. FEM contact feedback/reduction can consume these
    /// buffers directly through `TireFemContactFeedbackGpuTerrain`; the
    /// remaining deeper terrain work is to make owners export spatial surface
    /// velocity maps directly.
    pub fn raster_tire_fem_terrain_gpu_resident_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireMpmGpuKernel,
        raster: MpmTireTerrainRaster,
        heights: &wgpu::Buffer,
        sinkage: &wgpu::Buffer,
        surface_velocity_map: &wgpu::Buffer,
    ) -> Result<()> {
        let params = TireMpmTerrainRasterParams {
            particle_count: self.particle_count,
            origin: raster.origin,
            spacing: raster.spacing,
            nx: raster.nx,
            nz: raster.nz,
            rest_height: raster.rest_height,
        };
        let outputs = [heights, sinkage, surface_velocity_map];
        let reuse = self.tire_raster_cache.as_ref().is_some_and(|cache| {
            cache.params == params
                && cache.fallback_velocity == raster.surface_velocity
                && cache
                    .outputs
                    .iter()
                    .zip(outputs)
                    .all(|(cached, output)| cached == output)
        });
        if !reuse {
            let prepared = kernel.prepare_raster_to_buffers(
                device,
                &self.p_pos,
                &self.p_vel,
                &self.tire_compaction_buf,
                heights,
                sinkage,
                surface_velocity_map,
                &params,
                raster.surface_velocity,
            )?;
            self.tire_raster_cache = Some(TireRasterCache {
                params,
                fallback_velocity: raster.surface_velocity,
                outputs: [
                    heights.clone(),
                    sinkage.clone(),
                    surface_velocity_map.clone(),
                ],
                prepared,
            });
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MPM prepared terrain export"),
        });
        self.tire_raster_cache
            .as_ref()
            .expect("prepared raster")
            .prepared
            .encode(&mut encoder);
        queue.submit([encoder.finish()]);
        Ok(())
    }

    fn write_tire_feedback_output(
        &mut self,
        queue: &wgpu::Queue,
        output: &TireMpmGpuOutput,
    ) -> Result<()> {
        let n = self.particle_count as usize;
        if output.positions.len() != n
            || output.velocities.len() != n
            || output.compaction.len() != n
        {
            return Err(Error::Solver(format!(
                "mpm tire feedback: output length mismatch positions={} velocities={} compaction={} particles={n}",
                output.positions.len(),
                output.velocities.len(),
                output.compaction.len(),
            )));
        }
        molla_core::gpu_write_buffer!(
            queue,
            &self.p_pos,
            0,
            bytemuck::cast_slice(&output.positions)
        );
        molla_core::gpu_write_buffer!(
            queue,
            &self.p_vel,
            0,
            bytemuck::cast_slice(&output.velocities)
        );
        molla_core::gpu_write_buffer!(
            queue,
            &self.tire_compaction_buf,
            0,
            bytemuck::cast_slice(&output.compaction),
        );
        self.host_pos = output.positions.clone();
        self.host_vel = output.velocities.clone();
        self.tire_compaction = output.compaction.clone();
        Ok(())
    }

    /// Run one substep at `dt`. Caller is responsible for sub-stepping
    /// to satisfy CFL — see the file header.  Back-compat wrapper that
    /// delegates to [`Self::step_n`] with `substeps=1`.
    pub fn step(&mut self, engine: &Engine, dt: f32) -> Result<()> {
        self.step_n(engine, dt, 1)
    }

    /// Back-compat: single-substep variant taking explicit device/queue.
    pub fn step_with(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) -> Result<()> {
        self.step_n_with(device, queue, dt, 1)
    }

    /// Stage 24d.2: batched substep dispatcher.
    ///
    /// Encode `substeps` MPM substeps into a single command encoder and
    /// submit once.  All six kernel dispatches per substep share the
    /// persistent bind groups and uniform buffer; the host avoids any
    /// intra-step synchronisation.  Materially faster than calling
    /// `step()` `n` times when `n` is large because each `submit` has
    /// fixed wgpu / driver overhead independent of payload size.
    ///
    /// Timestep and material constants are fixed across this submitted batch.
    /// Caller-owned encoders use [`Self::prepare_step`] and [`Self::begin_command_batch`].
    pub fn step_n(&mut self, engine: &Engine, dt: f32, substeps: usize) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("step_n: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("step_n: no GPU queue".into()))?
            .clone();
        self.step_n_with(&device, &queue, dt, substeps)
    }

    pub fn step_n_with(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        substeps: usize,
    ) -> Result<()> {
        self.step_n_impl(_device, queue, dt, substeps, false)
    }

    fn step_n_impl(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        substeps: usize,
        profiled: bool,
    ) -> Result<()> {
        self.validate_prescribed_step(dt)?;
        if self.particle_count == 0 || substeps == 0 {
            return Ok(());
        }
        if self.use_particle_order
            && self
                .particle_order
                .as_ref()
                .is_none_or(|o| o.capacity < self.particle_count)
        {
            return Err(Error::Solver(
                "MPM particle count exceeds prepared ordering capacity".into(),
            ));
        }
        if profiled {
            self.begin_step_profile(queue, dt, substeps)?;
        }
        let profiled = profiled || self.profile_window_batch(dt, substeps);
        let uniforms = self.step_uniforms(dt);
        if let Some(guard) = &self.motion_guard {
            guard.upload(queue);
        }
        molla_core::gpu_write_buffer!(queue, &self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let table = self.step_material_table();
        molla_core::gpu_write_buffer!(queue, &self.material_table, 0, bytemuck::cast_slice(&table));
        if let Some(ids) = self.particle_material_dirty.take() {
            molla_core::gpu_write_buffer!(queue, &self.p_material, 0, bytemuck::cast_slice(&ids));
        }

        let jacobi_iters = match self.cfg.time_step {
            MpmTimeStep::Explicit => 0,
            MpmTimeStep::Implicit { jacobi_iters } => jacobi_iters,
        };

        if self.use_particle_order {
            self.particle_order.as_ref().unwrap().upload(
                queue,
                &self.cfg,
                self.particle_count,
                self.grid_window_cells,
            );
        }
        if self.use_active_grid {
            self.active_tiles.upload(
                queue,
                &self.cfg,
                self.particle_count,
                self.rigid_count,
                self.grid_window_cells,
            )?;
        }
        #[cfg(test)]
        if std::env::var_os("MOLLA_SOIL_REPLAY_SPLIT").is_some() {
            assert_eq!(substeps, 1);
            assert_eq!(jacobi_iters, 0);
            assert!(!self.use_active_grid && !self.use_particle_order && !profiled);
            replay_tests::split_step(self, _device, queue, dt);
            return Ok(());
        }
        let mut encoder = _device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mpm encoder (step_n)"),
        });
        self.encode_step_work(&mut encoder, substeps, jacobi_iters, profiled);
        if profiled {
            self.profiler
                .as_mut()
                .unwrap()
                .finish_batch(&mut encoder, substeps);
        }
        queue.submit([encoder.finish()]);
        Ok(())
    }

    fn step_uniforms(&self, dt: f32) -> Uniforms {
        let (rheology_id, friction_angle, snow_tc, snow_ts, snow_h) = match self.cfg.rheology {
            MpmRheology::Sand { friction_angle } => (0u32, friction_angle, 0.0, 0.0, 0.0),
            MpmRheology::Snow {
                theta_c,
                theta_s,
                hardening,
            } => (1u32, 0.0, theta_c, theta_s, hardening),
            MpmRheology::Viscous {
                velocity_damping,
                affine_damping,
            } => (2u32, 0.0, velocity_damping, affine_damping, 0.0),
        };
        let phi = friction_angle.max(1.0_f32.to_radians());
        let sin_phi = phi.sin();
        let dp_alpha = (2.0_f32 / 3.0).sqrt() * (2.0 * sin_phi) / (3.0 - sin_phi);

        Uniforms {
            particle_count: self.particle_count,
            rigid_count: self.rigid_count,
            grid_size: self.cfg.grid_size,
            grid_total: self.cfg.grid_size.pow(3),
            grid_origin: self.cfg.grid_origin,
            cell_size: self.cfg.cell_size,
            dt,
            gravity_y: self.cfg.gravity_y,
            youngs: self.cfg.youngs,
            poisson: self.cfg.poisson,
            dp_alpha,
            cohesion: self.cfg.cohesion,
            boundary_friction: self.cfg.boundary_friction,
            inv_cell_size: 1.0 / self.cfg.cell_size,
            rheology: rheology_id,
            snow_theta_c: snow_tc,
            snow_theta_s: snow_ts,
            snow_hardening: snow_h,
            grid_layout: [
                self.active_tiles.map_capacity,
                self.active_tiles.capacity,
                u32::from(self.use_particle_order),
                self.cfg.max_particles.max(1),
            ],
            grid_window_cells: [
                self.grid_window_cells[0],
                self.grid_window_cells[1],
                self.grid_window_cells[2],
                0,
            ],
        }
    }

    fn step_material_table(&self) -> [[f32; 4]; 2 * MAX_MPM_MATERIALS] {
        let mut table = [[0.0_f32; 4]; 2 * MAX_MPM_MATERIALS];
        if self.material_rows.is_empty() {
            let row = material_row(&self.cfg.rheology);
            table[0] = row[0];
            table[1] = row[1];
        } else {
            for (i, r) in self.material_rows.iter().enumerate() {
                let row = material_row(r);
                table[2 * i] = row[0];
                table[2 * i + 1] = row[1];
            }
        }
        table
    }

    fn encode_step_work(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        substeps: usize,
        jacobi_iters: u32,
        profiled: bool,
    ) {
        if self.particle_count == 0 || substeps == 0 {
            return;
        }
        self.encode_quiescence_reset(encoder);
        #[cfg(test)]
        if self.active_grid_needs_reset {
            if let Some(transfer) = &self.compensated_active_transfer {
                transfer.clear(encoder);
            }
        }
        if self.use_active_grid {
            self.encode_active_steps(encoder, substeps, jacobi_iters, profiled);
            return;
        }
        if self.active_grid_needs_reset {
            encoder.clear_buffer(
                &self.active_tiles.list,
                self.active_tiles.metadata_offset,
                Some(16),
            );
            self.active_grid_needs_reset = false;
        }
        let grid_total = self.cfg.grid_size.pow(3);
        let groups_grid = grid_total.div_ceil(WORKGROUP_SIZE);
        let groups_part = self.particle_count.div_ceil(WORKGROUP_SIZE);

        let batches = if self.use_particle_order { substeps } else { 1 };
        #[cfg(test)]
        let batches = if self.bounded_reference.is_some() {
            substeps
        } else {
            batches
        };
        let steps_per_batch = substeps / batches;
        for batch in 0..batches {
            self.encode_particle_order(encoder, profiled, batch);
            #[cfg(test)]
            self.encode_bounded_reference(encoder);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::mpm step_n"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bg_uniforms, &[]);
            pass.set_bind_group(1, &self.bg_particles, &[]);
            pass.set_bind_group(2, &self.bg_grid, &[]);
            pass.set_bind_group(3, &self.bg_rigid, &[]);

            for step in batch * steps_per_batch..(batch + 1) * steps_per_batch {
                // 1. reset grid
                self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::GridReset, step);
                pass.set_pipeline(&self.pipe_reset);
                pass.dispatch_workgroups(groups_grid.max(1), 1, 1);
                molla_core::gpu_metrics::record_dispatch();
                self.profile_end_pass(profiled, &mut pass);

                // Clear the rigid-coupling wrench accumulators.
                if self.rigid_count > 0 {
                    self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::RigidReset, step);
                    self.encode_rigid_wrench_reset(&mut pass);
                    self.profile_end_pass(profiled, &mut pass);
                }

                // Scatter particle mass, momentum and stiffness.
                self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::ParticleToGrid, step);
                self.encode_p2g(&mut pass);
                self.profile_end_pass(profiled, &mut pass);

                // Imprint rigid motion and accumulate contact impulses.
                if self.rigid_count > 0 {
                    self.profile_begin_pass(
                        profiled,
                        &mut pass,
                        MpmProfileStage::RigidImprint,
                        step,
                    );
                    self.encode_rigid_imprint(&mut pass, false);
                    self.profile_end_pass(profiled, &mut pass);
                }

                // 4. grid update
                self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::GridUpdate, step);
                pass.set_pipeline(&self.pipe_grid);
                pass.dispatch_workgroups(groups_grid.max(1), 1, 1);
                molla_core::gpu_metrics::record_dispatch();
                self.profile_end_pass(profiled, &mut pass);

                // 5. (optional) Implicit Jacobi sweeps.
                if jacobi_iters > 0 {
                    self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::Implicit, step);
                }
                for _ in 0..jacobi_iters {
                    pass.set_pipeline(&self.pipe_implicit_apply);
                    pass.dispatch_workgroups(groups_grid.max(1), 1, 1);
                    molla_core::gpu_metrics::record_dispatch();
                    // Ping-pong: copy g_velocity_prev → g_velocity.
                    pass.set_pipeline(&self.pipe_implicit_swap);
                    pass.dispatch_workgroups(groups_grid.max(1), 1, 1);
                    molla_core::gpu_metrics::record_dispatch();
                }
                if jacobi_iters > 0 {
                    self.profile_end_pass(profiled, &mut pass);
                }

                // Accumulate shell-cell contact forces.
                if self.rigid_count > 0 && !self.prescribed_solids {
                    self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::RigidForce, step);
                    self.encode_rigid_shell(&mut pass, false);
                    self.profile_end_pass(profiled, &mut pass);
                }

                #[cfg(test)]
                if self.use_fused_gather {
                    self.encode_gather_projection(&mut pass, groups_part.max(1), profiled, step);
                    continue;
                }
                let [gather_stage, commit_stage] = self.particle_publication_stages();
                self.profile_begin_pass(profiled, &mut pass, gather_stage, step);
                self.encode_gather(&mut pass, groups_part.max(1));
                self.profile_end_pass(profiled, &mut pass);
                self.profile_begin_pass(profiled, &mut pass, commit_stage, step);
                self.encode_plasticity(&mut pass, groups_part.max(1));
                self.profile_end_pass(profiled, &mut pass);
            }
        }
    }

    fn bind_step(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_bind_group(0, &self.bg_uniforms, &[]);
        pass.set_bind_group(1, &self.bg_particles, &[]);
        pass.set_bind_group(2, &self.bg_grid, &[]);
        pass.set_bind_group(3, &self.bg_rigid, &[]);
    }

    fn initialize_active_grid(&mut self, encoder: &mut wgpu::CommandEncoder, profiled: bool) {
        if self.active_grid_needs_reset {
            self.profile_begin_encoder(profiled, encoder, MpmProfileStage::Initialization, None);
            if self.sparse_grid {
                for buffer in [
                    &self.g_mass,
                    &self.g_momentum,
                    &self.g_velocity,
                    &self.g_velocity_prev,
                    &self.g_rigid_mask,
                ] {
                    encoder.clear_buffer(buffer, 0, None);
                }
            } else {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MPM active grid mode initialization"),
                    timestamp_writes: None,
                });
                self.bind_step(&mut pass);
                pass.set_pipeline(&self.pipe_reset);
                pass.dispatch_workgroups(self.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE), 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
            encoder.clear_buffer(&self.active_tiles.list, 0, None);
            encoder.clear_buffer(&self.active_tiles.indirect, 0, None);
            self.active_grid_needs_reset = false;
            self.profile_end_encoder(profiled, encoder);
        }
    }

    fn encode_active_steps(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        substeps: usize,
        jacobi_iters: u32,
        profiled: bool,
    ) {
        self.initialize_active_grid(encoder, profiled);
        for step in 0..substeps {
            self.encode_particle_order(encoder, profiled, step);
            #[cfg(test)]
            self.encode_bounded_reference(encoder);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MPM retire previous active grid"),
                    timestamp_writes: None,
                });
                self.bind_step(&mut pass);
                self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::GridReset, step);
                pass.set_pipeline(&self.active_pipelines[0]);
                pass.dispatch_workgroups_indirect(&self.active_tiles.indirect, 0);
                molla_core::gpu_metrics::record_dispatch();
                self.profile_end_pass(profiled, &mut pass);
            }
            self.profile_begin_encoder(profiled, encoder, MpmProfileStage::ActiveTiles, Some(step));
            self.active_tiles
                .encode_build(encoder, self.particle_count, self.rigid_count);
            self.profile_end_encoder(profiled, encoder);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MPM active grid step"),
                timestamp_writes: None,
            });
            self.bind_step(&mut pass);
            if self.rigid_count > 0 {
                self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::RigidReset, step);
                self.encode_rigid_wrench_reset(&mut pass);
                self.profile_end_pass(profiled, &mut pass);
            }
            self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::ParticleToGrid, step);
            self.encode_p2g(&mut pass);
            self.profile_end_pass(profiled, &mut pass);
            if self.rigid_count > 0 {
                self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::RigidImprint, step);
                self.encode_rigid_imprint(&mut pass, true);
                self.profile_end_pass(profiled, &mut pass);
            }
            self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::GridUpdate, step);
            pass.set_pipeline(&self.active_pipelines[2]);
            pass.dispatch_workgroups_indirect(&self.active_tiles.indirect, 0);
            molla_core::gpu_metrics::record_dispatch();
            self.profile_end_pass(profiled, &mut pass);
            if jacobi_iters > 0 {
                self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::Implicit, step);
            }
            for _ in 0..jacobi_iters {
                pass.set_pipeline(&self.active_pipelines[3]);
                pass.dispatch_workgroups_indirect(&self.active_tiles.indirect, 0);
                molla_core::gpu_metrics::record_dispatch();
                pass.set_pipeline(&self.active_pipelines[4]);
                pass.dispatch_workgroups_indirect(&self.active_tiles.indirect, 0);
                molla_core::gpu_metrics::record_dispatch();
            }
            if jacobi_iters > 0 {
                self.profile_end_pass(profiled, &mut pass);
            }
            if self.rigid_count > 0 && !self.prescribed_solids {
                self.profile_begin_pass(profiled, &mut pass, MpmProfileStage::RigidForce, step);
                self.encode_rigid_shell(&mut pass, true);
                self.profile_end_pass(profiled, &mut pass);
            }
            #[cfg(test)]
            if self.use_fused_gather {
                self.encode_gather_projection(
                    &mut pass,
                    self.particle_count.div_ceil(WORKGROUP_SIZE),
                    profiled,
                    step,
                );
                continue;
            }
            let [gather_stage, commit_stage] = self.particle_publication_stages();
            self.profile_begin_pass(profiled, &mut pass, gather_stage, step);
            self.encode_gather(&mut pass, self.particle_count.div_ceil(WORKGROUP_SIZE));
            self.profile_end_pass(profiled, &mut pass);
            self.profile_begin_pass(profiled, &mut pass, commit_stage, step);
            self.encode_plasticity(&mut pass, self.particle_count.div_ceil(WORKGROUP_SIZE));
            self.profile_end_pass(profiled, &mut pass);
        }
    }

    fn encode_p2g(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.use_compensated_stiffness {
            self.compensated_stiffness
                .as_ref()
                .unwrap()
                .encode(pass, self);
            return;
        }
        #[cfg(test)]
        if let Some(transfer) = &self.compensated_active_transfer {
            transfer.encode(pass, self);
            return;
        }
        if self.use_deterministic_p2g {
            self.deterministic_p2g.as_ref().unwrap().encode(pass, self);
            return;
        }
        pass.set_pipeline(self.p2g_pipeline());
        pass.dispatch_workgroups(self.particle_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    /// Read particle positions back to host. Synchronous; intended for
    /// tests and the viewer's per-frame draw.
    pub fn read_positions(&mut self, engine: &Engine) -> Result<Vec<[f32; 3]>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("read_positions: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("read_positions: no queue".into()))?
            .clone();
        self.read_positions_with(&device, &queue)
    }

    pub fn read_positions_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 3]>> {
        if self.particle_count == 0 {
            return Ok(vec![]);
        }
        let bytes = (self.particle_count as u64) * 16;
        // Stage 24d.2: reuse the persistent staging buffer (sized to
        // max_particles at constructor time).
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mpm read_positions encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.p_pos, 0, &self.stage_pos, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.stage_pos.slice(0..bytes);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("mpm readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("mpm readback poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("mpm readback recv: {e:?}")))??;
        let data = molla_core::gpu_readback!(slice);
        let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.stage_pos.unmap();
        let out: Vec<[f32; 3]> = raw[..self.particle_count as usize]
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        Ok(out)
    }

    /// Read particle velocities back to host.
    pub fn read_velocities_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 3]>> {
        if self.particle_count == 0 {
            return Ok(vec![]);
        }
        let bytes = (self.particle_count as u64) * 16;
        // Stage 24d.2: reuse the persistent staging buffer.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mpm read_velocities encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.p_vel, 0, &self.stage_vel, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.stage_vel.slice(0..bytes);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("mpm vel readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("mpm vel poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("mpm vel recv: {e:?}")))??;
        let data = molla_core::gpu_readback!(slice);
        let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.stage_vel.unmap();
        let out: Vec<[f32; 3]> = raw[..self.particle_count as usize]
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        Ok(out)
    }

    pub(crate) fn rigid_wrench_buffer(&self) -> &wgpu::Buffer {
        &self.rigid_wrench
    }

    pub(crate) fn rigid_center_buffer(&self) -> &wgpu::Buffer {
        &self.rigid_center
    }

    /// Read the last substep's world force and COM torque for each rigid body.
    pub fn read_rigid_wrenches(&mut self, engine: &Engine) -> Result<Vec<([f32; 3], [f32; 3])>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("read_rigid_wrenches: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("read_rigid_wrenches: no queue".into()))?
            .clone();
        self.read_rigid_wrenches_with(&device, &queue)
    }

    /// Stage 24d.3 — read back the grid mass + momentum buffers for
    /// parity testing (legacy `p2g` vs two-pass).  Returned tuple is
    /// `(mass[grid_total], momentum[grid_total*3])` — both flat
    /// `Vec<f32>` reinterpretations of the atomic-`u32` storage.  Not
    /// for production use; intended for `tests/mpm_perf.rs`'s
    /// `mpm_p2g_parity_legacy_vs_two_pass` only.
    pub fn read_grid_mass_momentum(&mut self, engine: &Engine) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.sparse_grid {
            return Err(Error::Solver(
                "MPM compact grid: use read_grid_storage_mass_momentum for physical slots".into(),
            ));
        }
        self.read_grid_storage_mass_momentum(engine)
    }

    /// Read physical grid slots; compact slot order is not a world-cell index.
    pub fn read_grid_storage_mass_momentum(
        &mut self,
        engine: &Engine,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("read_grid_mass_momentum: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("read_grid_mass_momentum: no queue".into()))?
            .clone();
        let grid_total = self.g_mass.size() / 4;
        let mass_bytes = grid_total * 4;
        let mom_bytes = grid_total * 12;
        let stage_mass = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("mpm.read_grid stage_mass"),
                size: mass_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let stage_mom = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("mpm.read_grid stage_mom"),
                size: mom_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mpm.read_grid encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.g_mass, 0, &stage_mass, 0, mass_bytes);
        encoder.copy_buffer_to_buffer(&self.g_momentum, 0, &stage_mom, 0, mom_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        // Map both, blocking once.
        let slice_m = stage_mass.slice(..);
        let slice_p = stage_mom.slice(..);
        let (sm, rm) = std::sync::mpsc::channel::<Result<()>>();
        let (sp, rp) = std::sync::mpsc::channel::<Result<()>>();
        slice_m.map_async(wgpu::MapMode::Read, move |r| {
            sm.send(r.map_err(|e| Error::Gpu(format!("grid mass map: {e:?}"))))
                .ok();
        });
        slice_p.map_async(wgpu::MapMode::Read, move |r| {
            sp.send(r.map_err(|e| Error::Gpu(format!("grid mom map: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("grid map poll: {e:?}")))?;
        rm.recv()
            .map_err(|e| Error::Gpu(format!("grid mass recv: {e:?}")))??;
        rp.recv()
            .map_err(|e| Error::Gpu(format!("grid mom recv: {e:?}")))??;
        let data_m = molla_core::gpu_readback!(slice_m);
        let data_p = molla_core::gpu_readback!(slice_p);
        let mass: Vec<f32> = bytemuck::cast_slice(&data_m).to_vec();
        let mom: Vec<f32> = bytemuck::cast_slice(&data_p).to_vec();
        drop(data_m);
        drop(data_p);
        stage_mass.unmap();
        stage_mom.unmap();
        Ok((mass, mom))
    }

    pub fn read_rigid_wrenches_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<([f32; 3], [f32; 3])>> {
        if self.rigid_count == 0 {
            return Ok(vec![]);
        }
        let bytes = (self.rigid_count as u64) * 32;
        let stage = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::mpm read_rigid_wrenches stage"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mpm read_rigid_wrenches encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.rigid_wrench, 0, &stage, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = stage.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("mpm wrench readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("mpm wrench poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("mpm wrench recv: {e:?}")))??;
        let data = molla_core::gpu_readback!(slice);
        let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        stage.unmap();
        let mut out = Vec::with_capacity(self.rigid_count as usize);
        for i in 0..self.rigid_count as usize {
            let f = raw[i * 2];
            let t = raw[i * 2 + 1];
            out.push(([f[0], f[1], f[2]], [t[0], t[1], t[2]]));
        }
        Ok(out)
    }
}

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

/// Like [`storage_entries`] but each binding's read-only flag is taken
/// from `read_only_per_binding`.
fn mixed_storage_entries(read_only_per_binding: &[bool]) -> Vec<wgpu::BindGroupLayoutEntry> {
    read_only_per_binding
        .iter()
        .enumerate()
        .map(|(i, &ro)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

fn buffer_entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

#[allow(dead_code)]
fn _arc_keep_alive(_a: Arc<wgpu::Device>) {}
