//! `TetraTireFemParams` + solver-mode enum and presets.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::super::SurfaceMaterial;

/// Parameters for a true volumetric tetrahedral FEM tire.
///
/// The mesh is an annular tire volume: circumferential segments × radial
/// layers × width layers. Every parametric hexahedral cell is split into
/// tetrahedra and solved with a finite-deformation compressible Neo-Hookean
/// constitutive law. The inner radius is bead/rim-coupled to a one-DOF hub;
/// the outer tread nodes receive floor contact. Inflation pressure is modeled
/// primarily as membrane pre-stress / tangent stiffening; optional direct
/// boundary traction exists but defaults to zero on this coarse exterior mesh.
#[derive(Clone, Copy, Debug)]
pub struct TetraTireFemParams {
    pub outer_radius: f64,
    pub inner_radius: f64,
    pub width: f64,
    pub segments: usize,
    pub radial_layers: usize,
    pub width_layers: usize,
    pub pressure_pa: f64,
    pub density: f64,
    /// Fallback/body Young's modulus for carcass rubber (Pa).
    pub young_modulus: f64,
    /// Softer sidewall material (Pa).
    pub sidewall_young_modulus: f64,
    /// Stiffer tread/belt material (Pa).
    pub tread_young_modulus: f64,
    /// Stiff bead/rim-adjacent material (Pa).
    pub bead_young_modulus: f64,
    pub poisson_ratio: f64,
    pub node_damping: f64,
    pub hub_mass: f64,
    pub downforce: f64,
    pub bead_stiffness: f64,
    pub bead_damping: f64,
    pub floor_stiffness: f64,
    pub floor_damping: f64,
    /// Legacy: scale for applying pressure directly to the tread surface.
    /// Defaults to `0.0` — the proper model (stage (b) in `PLAN_REALTIRE.md`)
    /// is the cavity gas-law in `cavity_polytropic_gamma` below, which acts
    /// on the inner bead surface, not the tread. Kept here so existing
    /// callers (e.g. the viewer scene) compile.
    pub pressure_load_scale: f64,
    /// Polytropic exponent γ for the cavity gas law `P · V^γ = const`.
    /// `1.0` (default) is isothermal — appropriate for slow tire cycles;
    /// real tires under fast deformation are closer to adiabatic (γ≈1.4)
    /// because heat can't conduct out fast enough.
    pub cavity_polytropic_gamma: f64,
    /// Named pressure-to-carcass coupling factor for the simplified sealed
    /// cavity.
    ///
    /// A full bead-seat + laminated carcass solve would distribute pressure
    /// through belts/plies and rim contact. Until that lands, only this
    /// audited fraction of the raw gas-law pressure is applied as bulk cavity
    /// traction. Keeping it in the preset makes the approximation visible and
    /// range-governed instead of hiding a hard-coded "leak" constant inside
    /// the solver.
    pub cavity_carcass_coupling: f64,
    /// Stage (d) — surface/ground material driving the brush contact
    /// friction at every FEM tread node. Use `SurfaceMaterial::ice()`
    /// for low-grip, `dry_asphalt()` for high, etc.
    pub surface_material: SurfaceMaterial,
    /// Stage (d) — bristle shear stiffness per unit contact area
    /// (N/m³). The brush force at a tread node is
    /// `K_brush · A_node · bristle_displacement`, capped by the
    /// Coulomb friction cone from `surface_material`.
    pub tread_shear_stiffness: f64,
    /// Stage (d) — first-order bristle relaxation rate (1/s). When a
    /// tread node leaves contact its bristle decays toward zero at
    /// this rate so re-contact starts from a fresh slip estimate.
    pub bristle_relax_rate: f64,
    /// Stage (f) — XPBD compliance for per-tetra volume preservation.
    ///
    /// This is not a replacement for an implicit Newton solve; it is a
    /// constraint stabilization layer over the explicit Neo-Hookean forces so
    /// high loads / low pressures do not silently crush elements toward zero
    /// volume. `0.0` is rigid/incompressible for the projection pass; larger
    /// values make the constraint softer.
    pub xpbd_volume_compliance: f64,
    /// Stage (f) — number of XPBD volume-projection sweeps per explicit
    /// substep. Keep this small in the real-time viewer; tests can raise it
    /// to probe the constraint in isolation.
    pub xpbd_volume_iterations: usize,
    /// Stage (f) — XPBD compliance for rim/bead positional attachment.
    ///
    /// The explicit bead spring still transfers load to the hub; this
    /// projection is the geometric no-detachment guardrail that keeps bead
    /// nodes near their mounted rim offsets under high carcass load.
    pub xpbd_bead_compliance: f64,
    /// Stage (f) — XPBD compliance for tread-belt edge inextensibility.
    ///
    /// Applied to circumferential, axial, and diagonal edges on the outer
    /// tread layer. This is a first structural belt constraint, not a full
    /// multi-ply belt/sidewall bending model.
    pub xpbd_tread_belt_compliance: f64,
    /// Stage (f) — XPBD compliance for sidewall bending-surrogate hinge
    /// constraints. Implemented as rest-distance constraints between
    /// opposite vertices across sidewall surface quads, which is the standard
    /// cheap cloth/FEM bending proxy before a full dihedral-angle solve.
    pub xpbd_sidewall_bend_compliance: f64,
    /// Stage (f3) — XPBD compliance for true sidewall quad dihedral-angle
    /// preservation on the CPU path.
    ///
    /// This complements the older edge-distance bend proxy by constraining
    /// the measured fold angle between the two triangles of each sidewall
    /// quad. GPU parity is tracked separately; use a larger compliance for
    /// realtime and lower values in structural tests.
    pub xpbd_sidewall_dihedral_compliance: f64,
    /// Stage (f) — number of bead/belt structural projection sweeps per
    /// explicit substep.
    pub xpbd_structural_iterations: usize,
    /// Stage (f2) — diagonal semi-implicit stiffness scale for the
    /// Neo-Hookean/fibre force update.
    ///
    /// The main constitutive law still evaluates explicit internal forces, but
    /// the velocity solve can include a lumped tangent-stiffness denominator
    /// `m + c·dt + scale·k_diag·dt²`. `0.0` gives the old explicit-elasticity
    /// behavior; `1.0` enables the first CPU semi-implicit stiff-rubber
    /// stabilizer for high pressure/high modulus tires.
    pub semi_implicit_stiffness_scale: f64,
    /// Solver-mode preset selector for [`TetraTireFem::step_on_terrain`].
    ///
    /// `ExplicitDefault` keeps the original explicit/semi-implicit production
    /// stepper. `MatrixFreeNewtonDebug` opts this tire into the CPU/debug
    /// matrix-free Newton timestep (the same path that
    /// `step_on_terrain_matrix_free_newton_debug` runs directly). The CPU/debug
    /// path is intentionally not the default — see the GPU parity ledger entry
    /// `tetra_fem_newton_solver_debug` for the GPU-parity follow-up.
    pub solver_mode: TetraTireFemSolverMode,
}

/// Solver-mode preset for [`TetraTireFem::step_on_terrain`].
///
/// This is the production-promotion seam for the matrix-free Newton work.
/// Defaulting to [`Self::ExplicitDefault`] keeps the existing live viewer /
/// vehicle path untouched; callers that want to exercise the CPU/debug
/// implicit timestep opt in via [`Self::MatrixFreeNewtonDebug`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TetraTireFemSolverMode {
    /// The existing explicit / semi-implicit Neo-Hookean stepper with XPBD
    /// constraint projections. Used by all live viewer / vehicle scenes today.
    ExplicitDefault,
    /// CPU/debug matrix-free Newton timestep that delegates to
    /// [`TetraTireFem::step_on_terrain_matrix_free_newton_debug`]. Still
    /// labelled CPU/debug — production tire stepping continues to use
    /// `ExplicitDefault` until the GPU sparse/matrix-free parity lane lands.
    MatrixFreeNewtonDebug,
}

impl Default for TetraTireFemSolverMode {
    fn default() -> Self {
        Self::ExplicitDefault
    }
}

impl Default for TetraTireFemParams {
    fn default() -> Self {
        Self::realtime_viewer()
    }
}

impl TetraTireFemParams {
    /// Hand-tuned realtime tire-FEM preset used by viewer/debug simulations.
    ///
    /// It keeps the explicit Neo-Hookean update stable with modest resolution
    /// while enabling first-pass XPBD volume/bead/belt/sidewall constraints.
    pub const fn realtime_viewer() -> Self {
        Self {
            outer_radius: 0.32,
            inner_radius: 0.12,
            width: 0.185,
            segments: 32,
            radial_layers: 3,
            width_layers: 3,
            pressure_pa: 180_000.0,
            density: 1_100.0,
            young_modulus: 4.0e6,
            sidewall_young_modulus: 2.0e6,
            tread_young_modulus: 8.0e6,
            bead_young_modulus: 2.5e7,
            poisson_ratio: 0.46,
            node_damping: 18.0,
            hub_mass: 22.0,
            downforce: 2_600.0,
            bead_stiffness: 5.5e4,
            bead_damping: 650.0,
            floor_stiffness: 2.8e5,
            floor_damping: 1_200.0,
            pressure_load_scale: 0.0,
            cavity_polytropic_gamma: 1.0,
            cavity_carcass_coupling: 0.18,
            surface_material: SurfaceMaterial::dry_asphalt(),
            tread_shear_stiffness: 2.6e6,
            bristle_relax_rate: 25.0,
            xpbd_volume_compliance: 5.0e-10,
            xpbd_volume_iterations: 1,
            xpbd_bead_compliance: 5.0e-5,
            xpbd_tread_belt_compliance: 5.0e-5,
            xpbd_sidewall_bend_compliance: 1.0e-3,
            xpbd_sidewall_dihedral_compliance: 5.0e-3,
            xpbd_structural_iterations: 1,
            semi_implicit_stiffness_scale: 0.0,
            solver_mode: TetraTireFemSolverMode::ExplicitDefault,
        }
    }

    /// Softer aired-down off-road FEM preset for terrain envelope/rut tests.
    pub const fn offroad_soft() -> Self {
        Self {
            outer_radius: 0.38,
            inner_radius: 0.14,
            width: 0.24,
            segments: 32,
            radial_layers: 3,
            width_layers: 3,
            pressure_pa: 95_000.0,
            young_modulus: 3.0e6,
            sidewall_young_modulus: 1.35e6,
            tread_young_modulus: 6.5e6,
            bead_young_modulus: 2.2e7,
            downforce: 3_800.0,
            bead_stiffness: 4.8e4,
            xpbd_tread_belt_compliance: 8.0e-5,
            xpbd_sidewall_bend_compliance: 1.6e-3,
            xpbd_sidewall_dihedral_compliance: 8.0e-3,
            ..Self::realtime_viewer()
        }
    }

    /// Higher-pressure/stiffer reference preset for envelope/contact
    /// comparisons.
    pub const fn high_pressure_reference() -> Self {
        Self {
            pressure_pa: 320_000.0,
            young_modulus: 5.0e6,
            sidewall_young_modulus: 2.8e6,
            tread_young_modulus: 9.5e6,
            bead_young_modulus: 2.8e7,
            bead_stiffness: 6.8e4,
            xpbd_tread_belt_compliance: 3.5e-5,
            xpbd_sidewall_bend_compliance: 7.5e-4,
            xpbd_sidewall_dihedral_compliance: 2.5e-3,
            semi_implicit_stiffness_scale: 1.0,
            ..Self::realtime_viewer()
        }
    }

    /// High-pressure preset that defaults to the matrix-free Newton implicit
    /// timestep.
    ///
    /// Numerically equivalent to [`Self::high_pressure_reference`] except for
    /// the `solver_mode`, which is set to
    /// [`TetraTireFemSolverMode::MatrixFreeNewtonDebug`]. This is the first
    /// production-promotion preset for the matrix-free Newton work: the
    /// GPU parity row remains live (`tire_fem_matrix_free_newton.wgsl` +
    /// `TireFemMatrixFreeNewtonGpuKernel`), and the CPU/debug timestep
    /// still drives the live tire state. Use this preset when the
    /// explicit/semi-implicit CFL becomes the dominant cost.
    pub const fn high_pressure_implicit() -> Self {
        Self {
            solver_mode: TetraTireFemSolverMode::MatrixFreeNewtonDebug,
            ..Self::high_pressure_reference()
        }
    }

    /// Return a copy with mesh resolution overridden by a test/viewer
    /// scenario while keeping the named preset's material/solver constants.
    pub fn with_resolution(
        mut self,
        segments: usize,
        radial_layers: usize,
        width_layers: usize,
    ) -> Self {
        self.segments = segments;
        self.radial_layers = radial_layers;
        self.width_layers = width_layers;
        self
    }

    /// Return a copy with inflation pressure and vertical load overridden.
    pub fn with_pressure_and_load(mut self, pressure_pa: f64, downforce: f64) -> Self {
        self.pressure_pa = pressure_pa;
        self.downforce = downforce;
        self
    }

    /// Return a copy with the live solver mode overridden.
    ///
    /// Use [`TetraTireFemSolverMode::MatrixFreeNewtonDebug`] to opt this tire
    /// into the CPU/debug matrix-free Newton timestep via
    /// [`TetraTireFem::step_on_terrain`]. The default
    /// [`TetraTireFemSolverMode::ExplicitDefault`] keeps the live viewer /
    /// vehicle stepper unchanged.
    pub fn with_solver_mode(mut self, solver_mode: TetraTireFemSolverMode) -> Self {
        self.solver_mode = solver_mode;
        self
    }
}
