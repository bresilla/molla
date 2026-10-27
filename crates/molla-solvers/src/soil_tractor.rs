//! GPU articulated tractor and single-tine soil workload, physically uncalibrated.

use crate::mpm::{MpmConfig, MpmGridStorage, MpmRigidDeviceBinding, MpmSimulation, MpmSolidShape};
use crate::rigid_world::{
    MpmSoilFrameLoadObservation, MpmSoilFrameLoads, MpmSoilFrameProfile, MpmSoilWrenchFrame,
    RigidWorld,
};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{Mat3, Quat, Real, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use std::sync::Arc;

pub const WHEEL_RADIUS: Real = 0.45;
pub const WHEEL_HALF_WIDTH: Real = 0.14;
pub const CHASSIS_HALF_EXTENTS: Vec3 = Vec3::new(0.95, 0.22, 0.42);
pub const TINE_HALF_EXTENTS: Vec3 = Vec3::new(0.10, 0.35, 0.10);

#[derive(Clone, Copy)]
enum PairedSoilContact {
    WallFriction,
    Newton,
    Admm,
}

impl PairedSoilContact {
    fn enable(
        self,
        world: &mut RigidWorld,
        config: crate::mpm::MpmUnilateralConfig,
        coefficients: &[(BodyHandle, f32)],
    ) -> Result<()> {
        match self {
            Self::WallFriction => {
                world.enable_mpm_soil_wall_friction_coupling(config, coefficients)
            }
            Self::Newton => world.enable_mpm_soil_newton_coupling(config, coefficients),
            Self::Admm => world.enable_mpm_soil_admm_coupling(config, coefficients),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SoilTractorTransfer {
    #[default]
    Atomic,
    /// Fixed-order gather with a stress-cache budget; ordering and driver storage are separate.
    FixedOrder { max_gather_bytes: u64 },
    /// Fixed-order gather with shared immutable particle chunks and the same cache budget.
    SharedChunk { max_gather_bytes: u64 },
    /// Fixed-order gather with exact-input grid reuse and bounded retained storage.
    GridReuse { max_gather_bytes: u64 },
}

impl SoilTractorTransfer {
    /// Check particle storage only; grid reuse also requires `validate_with_grid`.
    pub fn validate(self, particles: u32) -> Result<()> {
        let (budget, stride) = match self {
            Self::Atomic => return Ok(()),
            Self::FixedOrder { max_gather_bytes } | Self::SharedChunk { max_gather_bytes } => {
                (max_gather_bytes, 64)
            }
            Self::GridReuse { max_gather_bytes } => (max_gather_bytes, 176),
        };
        if budget < u64::from(particles.max(1)) * stride {
            return Err(invalid("fixed-order gather scratch budget exceeded"));
        }
        Ok(())
    }

    pub fn required_gather_bytes(self, particles: u32, grid_size: u32) -> Result<u64> {
        match self {
            Self::Atomic => Ok(0),
            Self::FixedOrder { .. } | Self::SharedChunk { .. } => {
                Ok(u64::from(particles.max(1)) * 64)
            }
            Self::GridReuse { .. } => {
                let nodes = u64::from(grid_size)
                    .checked_pow(3)
                    .filter(|&n| n > 0 && n <= u64::from(u32::MAX / 5))
                    .ok_or_else(|| invalid("grid reuse index overflow"))?;
                let tiles = u64::from(grid_size.div_ceil(4)).pow(3);
                Ok(u64::from(particles.max(1)) * 176 + nodes * 20 + tiles * 8 + 64)
            }
        }
    }

    pub fn validate_with_grid(self, particles: u32, grid_size: u32) -> Result<()> {
        self.validate(particles)?;
        if let Self::GridReuse { max_gather_bytes } = self {
            if max_gather_bytes < self.required_gather_bytes(particles, grid_size)? {
                return Err(invalid("grid reuse gather scratch budget exceeded"));
            }
        }
        Ok(())
    }

    fn prepare(
        self,
        engine: &Engine,
        soil: &mut MpmSimulation,
        particles: u32,
        grid_size: u32,
        mut trace: impl FnMut(&str),
    ) -> Result<()> {
        #[cfg(test)]
        let _setup = crate::soil_setup_trace::stage("tractor.transfer");
        self.validate_with_grid(particles, grid_size)?;
        if let Self::FixedOrder { max_gather_bytes }
        | Self::SharedChunk { max_gather_bytes }
        | Self::GridReuse { max_gather_bytes } = self
        {
            trace("ordering begin");
            soil.prepare_parallel_particle_tile_ordering(
                engine
                    .queue()
                    .ok_or_else(|| invalid("GPU queue required"))?,
                particles,
            )?;
            soil.set_particle_tile_ordering(true)?;
            trace("ordering ready; gather begin");
            if matches!(self, Self::GridReuse { .. }) {
                soil.prepare_deterministic_p2g_grid_reuse(max_gather_bytes, false, false)?;
            } else if matches!(self, Self::SharedChunk { .. }) {
                soil.prepare_deterministic_p2g_shared_chunks(max_gather_bytes, false)?;
            } else {
                soil.prepare_deterministic_p2g_with_budget(max_gather_bytes)?;
            }
            soil.set_deterministic_p2g(true)?;
            trace("gather ready");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SoilTractorConfig {
    /// X/Y/Z lattice counts over a 5.6 x 0.25 x 3.2 metre soil bed.
    pub seed: [u32; 3],
    pub grid_size: u32,
    pub cell_size: f32,
    pub youngs_pa: f32,
    pub density_kg_m3: f32,
    pub max_substep_s: Real,
    pub coupling_budget_bytes: u64,
}

impl Default for SoilTractorConfig {
    fn default() -> Self {
        Self {
            seed: [100, 10, 50],
            grid_size: 64,
            cell_size: 0.1,
            youngs_pa: 3000.0,
            density_kg_m3: 1600.0,
            max_substep_s: 0.0005,
            coupling_budget_bytes: 128 * 1024 * 1024,
        }
    }
}

impl SoilTractorConfig {
    pub fn validate_grid_storage(self, storage: MpmGridStorage) -> Result<()> {
        self.validate()?;
        if let MpmGridStorage::Compact { tile_capacity } = storage {
            if tile_capacity == 0 || tile_capacity > self.grid_size.div_ceil(4).pow(3) {
                return Err(invalid(
                    "compact tile capacity must fit the configured grid",
                ));
            }
        }
        Ok(())
    }

    pub fn particle_count(self) -> Result<u32> {
        self.seed
            .into_iter()
            .try_fold(1u32, |n, count| {
                if count == 0 {
                    None
                } else {
                    n.checked_mul(count)
                }
            })
            .ok_or_else(|| invalid("invalid particle lattice"))
    }

    pub fn validate(self) -> Result<()> {
        let count = self.particle_count()?;
        if count > 1_000_000
            || self.grid_size < 8
            || self.grid_size > 256
            || !self.cell_size.is_finite()
            || !(0.025..=0.125).contains(&self.cell_size)
            || self.grid_size as f32 * self.cell_size < 6.0
            || !self.youngs_pa.is_finite()
            || self.youngs_pa <= 0.0
            || !self.density_kg_m3.is_finite()
            || self.density_kg_m3 <= 0.0
            || !self.max_substep_s.is_finite()
            || !(0.00001..=0.0005).contains(&self.max_substep_s)
            || self.coupling_budget_bytes == 0
        {
            return Err(invalid("invalid bounded workload configuration"));
        }
        Ok(())
    }

    #[cfg(test)]
    fn soil(self, engine: &Engine) -> Result<MpmSimulation> {
        self.soil_backend(engine, false)
    }

    #[cfg(test)]
    fn soil_backend(self, engine: &Engine, native: bool) -> Result<MpmSimulation> {
        self.soil_backend_with_grid(engine, native, MpmGridStorage::Dense)
    }

    fn soil_backend_with_grid(
        self,
        engine: &Engine,
        native: bool,
        storage: MpmGridStorage,
    ) -> Result<MpmSimulation> {
        #[cfg(test)]
        let _setup = crate::soil_setup_trace::stage("tractor.soil");
        self.validate_grid_storage(storage)?;
        let count = self.particle_count()?;
        let spacing = [
            5.6 / self.seed[0] as f32,
            0.25 / self.seed[1] as f32,
            3.2 / self.seed[2] as f32,
        ];
        let volume = spacing.into_iter().product();
        let mut cfg = MpmConfig::sand();
        cfg.grid_size = self.grid_size;
        cfg.grid_origin = [-3.0, -0.5, -3.0];
        cfg.cell_size = self.cell_size;
        cfg.max_particles = count;
        cfg.max_rigid = 5;
        cfg.youngs = self.youngs_pa;
        cfg.particle_volume = volume;
        cfg.particle_mass = self.density_kg_m3 * volume;
        let mut soil = match (native, storage) {
            (true, MpmGridStorage::Compact { tile_capacity }) => {
                MpmSimulation::new_native_sparse(engine, cfg, tile_capacity)
            }
            (false, MpmGridStorage::Compact { tile_capacity }) => {
                MpmSimulation::new_sparse(engine, cfg, tile_capacity)
            }
            (true, _) => MpmSimulation::new_native_explicit(engine, cfg),
            (false, _) => MpmSimulation::new(engine, cfg),
        }?;
        if storage == MpmGridStorage::ActiveTiles {
            soil.set_active_grid(true)?;
        }
        #[cfg(test)]
        if std::env::var_os("MOLLA_SOIL_DIAGNOSTIC_GLOBAL_P2G").is_some() {
            soil.set_p2g_mode(false);
        }
        let mut points = Vec::with_capacity(count as usize);
        for y in 0..self.seed[1] {
            for z in 0..self.seed[2] {
                for x in 0..self.seed[0] {
                    points.push([
                        -2.8 + (x as f32 + 0.5) * spacing[0],
                        -0.25 + (y as f32 + 0.5) * spacing[1],
                        -1.6 + (z as f32 + 0.5) * spacing[2],
                    ]);
                }
            }
        }
        soil.set_particles(
            engine
                .queue()
                .ok_or_else(|| invalid("GPU queue required"))?,
            &points,
        )?;
        Ok(soil)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SoilTractorHandles {
    pub chassis: BodyHandle,
    pub wheels: [BodyHandle; 4],
    pub axles: [JointHandle; 4],
    pub tine: BodyHandle,
    pub hitch: JointHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilTractorTelemetry {
    pub simulated_seconds: Real,
    pub chassis_pose: Transform,
    pub wheel_surface_speed_m_s: [Real; 4],
    pub wheel_forward_speed_m_s: [Real; 4],
    /// Slip ratio relative to stationary ground, not local moving soil velocity.
    pub nominal_slip: [Real; 4],
}

pub struct SoilTractor {
    world: RigidWorld,
    handles: SoilTractorHandles,
    config: SoilTractorConfig,
    simulated_seconds: Real,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil tractor: {message}"))
}

fn box_inertia(mass: Real, half: Vec3) -> Mat3 {
    Mat3::from_diagonal(
        Vec3::new(
            half.y * half.y + half.z * half.z,
            half.x * half.x + half.z * half.z,
            half.x * half.x + half.y * half.y,
        ) * (mass / 3.0),
    )
}

fn body(scene: &mut RigidScene, pose: Transform, mass: Real, inertia: Mat3) -> Result<BodyHandle> {
    scene.insert_body(BodyDesc {
        params: BodyParams {
            pose,
            ..Default::default()
        },
        additional_mass: Some(MassProperties {
            mass,
            local_com: Vec3::ZERO,
            inertia,
        }),
        ..Default::default()
    })
}

/// Six real dynamic bodies, four revolute axles and one fixed tine hitch.
/// +X is forward, +Y is up; cylinder local Y maps to the world Z axle.
pub fn build_soil_tractor_scene() -> Result<(RigidScene, SoilTractorHandles)> {
    let mut scene = RigidScene::new()?;
    scene.set_gravity(Vec3::new(0.0, -9.81, 0.0))?;
    let origin = Vec3::new(0.0, 0.95, 0.0);
    let chassis = body(
        &mut scene,
        Transform::from_translation(origin),
        700.0,
        box_inertia(700.0, CHASSIS_HALF_EXTENTS),
    )?;
    let wheel_rotation = Quat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    let radial = 35.0 * (3.0 * WHEEL_RADIUS.powi(2) + (2.0 * WHEEL_HALF_WIDTH).powi(2)) / 12.0;
    let axial = 35.0 * WHEEL_RADIUS.powi(2) / 2.0;
    let mut wheels = Vec::new();
    let mut axles = Vec::new();
    for (x, z) in [(-0.75, -0.65), (-0.75, 0.65), (0.75, -0.65), (0.75, 0.65)] {
        let position = Vec3::new(x, 0.40, z);
        let wheel = body(
            &mut scene,
            Transform::new(position, wheel_rotation),
            35.0,
            Mat3::from_diagonal(Vec3::new(radial, axial, radial)),
        )?;
        let axle = scene.insert_joint(JointDesc {
            parent: Some(chassis),
            child: wheel,
            kind: JointType::Revolute,
            axis: Vec3::Z,
            frame_parent: Transform::from_translation(position - origin),
            frame_child: Transform::new(Vec3::ZERO, wheel_rotation.inverse()),
        })?;
        wheels.push(wheel);
        axles.push(axle);
    }
    let tine_position = Vec3::new(-1.45, 0.23, 0.0);
    let tine = body(
        &mut scene,
        Transform::from_translation(tine_position),
        20.0,
        box_inertia(20.0, TINE_HALF_EXTENTS),
    )?;
    let hitch = scene.insert_joint(JointDesc {
        parent: Some(chassis),
        child: tine,
        kind: JointType::Fixed,
        axis: Vec3::X,
        frame_parent: Transform::from_translation(tine_position - origin),
        frame_child: Transform::IDENTITY,
    })?;
    Ok((
        scene,
        SoilTractorHandles {
            chassis,
            wheels: wheels.try_into().unwrap(),
            axles: axles.try_into().unwrap(),
            tine,
            hitch,
        },
    ))
}

impl SoilTractor {
    #[cfg(test)]
    pub(crate) fn world_mut_for_test(&mut self) -> &mut RigidWorld {
        &mut self.world
    }

    pub fn world(&self) -> &RigidWorld {
        &self.world
    }

    pub fn handles(&self) -> SoilTractorHandles {
        self.handles
    }

    pub fn read_soil_positions(&mut self) -> Result<Vec<[f32; 3]>> {
        self.world.read_mpm_soil_positions()
    }

    /// Prepares changed coupling without advancing the tractor or soil.
    pub fn prepare_soil_coupling(&mut self) -> Result<()> {
        self.world.prepare_mpm_soil_coupling()
    }

    /// Sets per-axis soil displacement bounds in metres per substep.
    /// Guard storage has a separate byte budget; None disables without freeing it.
    pub fn set_soil_motion_bound(
        &mut self,
        bound: Option<[f64; 3]>,
        max_bytes: u64,
    ) -> Result<u64> {
        self.world.set_mpm_soil_motion_bound(bound, max_bytes)
    }

    /// Selects fused guarded G2P/plasticity proposals with separate publication.
    pub fn set_soil_motion_projection_fused(&mut self, enabled: bool) -> Result<()> {
        self.world.set_mpm_soil_motion_projection_fused(enabled)
    }

    /// Selects portable direct reaction reduction with a separate cold-storage budget.
    pub fn set_soil_compensated_reactions(&mut self, enabled: bool, max_bytes: u64) -> Result<u64> {
        self.world
            .set_mpm_soil_compensated_reactions(enabled, max_bytes)
    }

    /// Blocking, bounded particle-state observation, outside the physics hot path.
    /// Returns the committed-frame ordinal and soil checkpoint, not a world save.
    pub fn read_soil_checkpoint(
        &mut self,
        max_particles: u32,
    ) -> Result<(u64, crate::mpm::MpmSoilCheckpoint)> {
        self.world.read_mpm_soil_checkpoint(max_particles)
    }

    /// Moves soil-grid bounds without translating the tractor or soil coordinates.
    /// Requires a committed frame and unchanged support on every moved boundary.
    pub fn move_soil_grid_window(&mut self, origin: [f32; 3]) -> Result<u64> {
        self.world
            .move_mpm_soil_grid_window(origin, self.config.particle_count()?)
    }

    /// Moves physical soil-grid bounds by absolute cells against the fixed lattice anchor.
    /// Requires a committed frame and preserves tractor and particle coordinates.
    pub fn move_soil_grid_window_cells(&mut self, cells: [i32; 3]) -> Result<u64> {
        self.world
            .move_mpm_soil_grid_window_cells(cells, self.config.particle_count()?)
    }

    pub fn soil_particle_render_view(
        &self,
    ) -> Result<(u64, crate::mpm::MpmParticleRenderView<'_>)> {
        self.world.mpm_soil_particle_render_view()
    }

    pub fn prepare_soil_surface(
        &self,
        config: crate::kernels::SoilSurfaceConfig,
        max_owned_bytes: u64,
    ) -> Result<crate::kernels::PreparedSoilSurface> {
        self.world.prepare_mpm_soil_surface(config, max_owned_bytes)
    }

    pub fn try_request_soil_surface(
        &self,
        surface: &mut crate::kernels::PreparedSoilSurface,
    ) -> Result<bool> {
        self.world.try_request_mpm_soil_surface(surface)
    }

    pub fn poll_soil_surface(
        &self,
        surface: &mut crate::kernels::PreparedSoilSurface,
    ) -> Result<Option<crate::kernels::SoilSurfaceObservation>> {
        self.world.poll_mpm_soil_surface(surface)
    }

    pub fn read_soil_surface(
        &self,
        surface: &mut crate::kernels::PreparedSoilSurface,
    ) -> Result<crate::rigid_world::MpmSoilSurfaceFrame> {
        self.world.read_mpm_soil_surface(surface)
    }

    pub fn prepare_soil_frame_profiling(&mut self, max_substeps: usize) -> Result<()> {
        self.world.prepare_mpm_soil_frame_profiling(max_substeps)
    }

    /// Retain whole-frame timestamps including nested Newton preparation stages.
    pub fn prepare_soil_frame_profiling_with_preparation(
        &mut self,
        max_substeps: usize,
    ) -> Result<()> {
        self.world
            .prepare_mpm_soil_frame_profiling_with_preparation(max_substeps)
    }

    pub fn read_soil_frame_profile(&mut self) -> Result<Option<MpmSoilFrameProfile>> {
        self.world.read_mpm_soil_frame_profile()
    }

    pub fn prepare_soil_frame_loads(&mut self) -> Result<()> {
        self.world.prepare_mpm_soil_frame_loads()
    }

    pub fn enable_mass_coupling(&mut self, nodes_per_body: u32) -> Result<()> {
        self.world.enable_mpm_soil_mass_coupling(nodes_per_body)
    }

    pub fn enable_mass_coupling_with_boundary_scan(
        &mut self,
        nodes_per_body: u32,
        boundary_scan: crate::soil_contact_projection_gpu::SoilBoundaryScan,
    ) -> Result<()> {
        self.world
            .enable_mpm_soil_mass_coupling_with_boundary_scan(nodes_per_body, boundary_scan)
    }
    pub fn enable_unilateral_coupling(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
    ) -> Result<()> {
        self.world.enable_mpm_soil_unilateral_coupling(config)
    }

    pub fn enable_friction_coupling(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
        coefficients: &[(BodyHandle, f32)],
    ) -> Result<()> {
        self.world
            .enable_mpm_soil_friction_coupling(config, coefficients)
    }

    pub fn read_soil_frame_loads(&mut self) -> Result<MpmSoilFrameLoads> {
        self.world.read_mpm_soil_frame_loads()
    }

    pub fn try_request_soil_frame_loads(&mut self) -> Result<bool> {
        self.world.try_request_mpm_soil_frame_loads()
    }

    pub fn poll_soil_frame_loads(&mut self) -> Result<Option<MpmSoilFrameLoadObservation>> {
        self.world.poll_mpm_soil_frame_loads()
    }

    pub fn reset_soil_frame_load_observations(&mut self) -> Result<()> {
        self.world.reset_mpm_soil_frame_load_observations()
    }

    pub fn is_current_soil_frame_load_observation(
        &self,
        sample: &MpmSoilFrameLoadObservation,
    ) -> bool {
        self.world
            .is_current_mpm_soil_frame_load_observation(sample)
    }

    pub fn enable_wall_friction_coupling(
        &mut self,
        config: crate::mpm::MpmUnilateralConfig,
        coefficients: &[(BodyHandle, f32)],
    ) -> Result<()> {
        self.world
            .enable_mpm_soil_wall_friction_coupling(config, coefficients)
    }

    pub fn read_soil_contact_motion(
        &mut self,
    ) -> Result<crate::rigid_world::MpmSoilContactMotionFrame> {
        self.world.read_mpm_soil_contact_motion()
    }

    pub fn new(engine: Arc<Engine>, config: SoilTractorConfig) -> Result<Self> {
        Self::new_backend(
            engine,
            config,
            false,
            molla_kernels::FeatherstoneRounding::Storage,
            molla_kernels::FeatherstoneExecution::Generic,
            false,
        )
    }

    /// Select portable soil storage; active grids require explicit coupling before stepping.
    pub fn new_with_grid_storage(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        storage: MpmGridStorage,
    ) -> Result<Self> {
        Self::new_backend_with_grid(
            engine,
            config,
            false,
            molla_kernels::FeatherstoneRounding::Storage,
            molla_kernels::FeatherstoneExecution::Generic,
            false,
            storage,
        )
    }

    /// Use native float32 buffer atomics for dense explicit soil on a capable Engine.
    pub fn new_native_explicit(engine: Arc<Engine>, config: SoilTractorConfig) -> Result<Self> {
        Self::new_native_explicit_with_rounding(
            engine,
            config,
            molla_kernels::FeatherstoneRounding::Storage,
        )
    }

    pub fn new_native_explicit_with_rounding(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        rounding: molla_kernels::FeatherstoneRounding,
    ) -> Result<Self> {
        Self::new_native_explicit_with_machine_options(
            engine,
            config,
            rounding,
            molla_kernels::FeatherstoneExecution::Generic,
        )
    }

    pub fn new_native_explicit_with_machine_options(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        rounding: molla_kernels::FeatherstoneRounding,
        execution: molla_kernels::FeatherstoneExecution,
    ) -> Result<Self> {
        Self::new_backend(engine, config, true, rounding, execution, false)
    }

    /// Select native soil storage and machine options; active grids require coupling before stepping.
    pub fn new_native_explicit_with_grid_options(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        storage: MpmGridStorage,
        rounding: molla_kernels::FeatherstoneRounding,
        execution: molla_kernels::FeatherstoneExecution,
    ) -> Result<Self> {
        Self::new_backend_with_grid(engine, config, true, rounding, execution, false, storage)
    }

    /// Native float32-atomic soil with paired machine rates and shared-wall friction.
    pub fn new_native_explicit_with_paired_wall_friction(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
    ) -> Result<Self> {
        Self::new_native_explicit_with_paired_wall_friction_execution(
            engine,
            config,
            coupling,
            friction,
            molla_kernels::FeatherstoneExecution::Generic,
        )
    }

    /// Native float32-atomic soil with paired machine rates and prepared Newton contact.
    pub fn new_native_explicit_with_paired_newton(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
    ) -> Result<Self> {
        Self::new_native_explicit_with_paired_newton_execution(
            engine,
            config,
            coupling,
            friction,
            molla_kernels::FeatherstoneExecution::Generic,
        )
    }

    /// Paired shared-wall friction with an explicit supported machine execution profile.
    pub fn new_native_explicit_with_paired_wall_friction_execution(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        execution: molla_kernels::FeatherstoneExecution,
    ) -> Result<Self> {
        Self::new_native_explicit_with_paired_friction(
            engine,
            config,
            coupling,
            friction,
            PairedSoilContact::WallFriction,
            execution,
            MpmGridStorage::Dense,
            SoilTractorTransfer::Atomic,
        )
    }

    /// Paired Newton contact with an explicit supported machine execution profile.
    pub fn new_native_explicit_with_paired_newton_execution(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        execution: molla_kernels::FeatherstoneExecution,
    ) -> Result<Self> {
        Self::new_native_explicit_with_paired_friction(
            engine,
            config,
            coupling,
            friction,
            PairedSoilContact::Newton,
            execution,
            MpmGridStorage::Dense,
            SoilTractorTransfer::Atomic,
        )
    }

    pub fn new_native_explicit_with_paired_wall_friction_grid(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        execution: molla_kernels::FeatherstoneExecution,
        storage: MpmGridStorage,
    ) -> Result<Self> {
        Self::new_native_explicit_with_paired_friction(
            engine,
            config,
            coupling,
            friction,
            PairedSoilContact::WallFriction,
            execution,
            storage,
            SoilTractorTransfer::Atomic,
        )
    }

    pub fn new_native_explicit_with_paired_newton_grid(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        execution: molla_kernels::FeatherstoneExecution,
        storage: MpmGridStorage,
    ) -> Result<Self> {
        Self::new_native_explicit_with_paired_friction(
            engine,
            config,
            coupling,
            friction,
            PairedSoilContact::Newton,
            execution,
            storage,
            SoilTractorTransfer::Atomic,
        )
    }

    /// Selects native P2G before constructing paired machine and Newton contact owners.
    /// Fixed-order transfer controls P2G summation, not whole-trajectory determinism.
    pub fn new_native_explicit_with_paired_newton_transfer(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        execution: molla_kernels::FeatherstoneExecution,
        storage: MpmGridStorage,
        transfer: SoilTractorTransfer,
    ) -> Result<Self> {
        Self::new_native_explicit_with_paired_friction(
            engine,
            config,
            coupling,
            friction,
            PairedSoilContact::Newton,
            execution,
            storage,
            transfer,
        )
    }

    /// Experimental guarded ADMM with explicit paired-machine and grid/transfer choices.
    pub fn new_native_explicit_with_paired_admm_transfer(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        execution: molla_kernels::FeatherstoneExecution,
        storage: MpmGridStorage,
        transfer: SoilTractorTransfer,
    ) -> Result<Self> {
        Self::new_native_explicit_with_paired_friction(
            engine,
            config,
            coupling,
            friction,
            PairedSoilContact::Admm,
            execution,
            storage,
            transfer,
        )
    }

    /// Native force prediction and shared mass with paired Newton soil at a fixed frame interval.
    pub fn new_native_prediction_with_newton_grid(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        storage: MpmGridStorage,
        frame_dt: Real,
    ) -> Result<Self> {
        Self::new_native_prediction_with_newton_transfer(
            engine,
            config,
            coupling,
            friction,
            storage,
            frame_dt,
            SoilTractorTransfer::Atomic,
        )
    }

    /// Prepares the selected transfer before attaching soil and its retained coupling owners.
    pub fn new_native_prediction_with_newton_transfer(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        storage: MpmGridStorage,
        frame_dt: Real,
        transfer: SoilTractorTransfer,
    ) -> Result<Self> {
        Self::new_native_prediction_with_contact_transfer(
            engine,
            config,
            coupling,
            friction,
            storage,
            frame_dt,
            transfer,
            PairedSoilContact::Newton,
        )
    }

    /// Native force prediction and shared mass with experimental guarded ADMM contact.
    pub fn new_native_prediction_with_admm_transfer(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        storage: MpmGridStorage,
        frame_dt: Real,
        transfer: SoilTractorTransfer,
    ) -> Result<Self> {
        Self::new_native_prediction_with_contact_transfer(
            engine,
            config,
            coupling,
            friction,
            storage,
            frame_dt,
            transfer,
            PairedSoilContact::Admm,
        )
    }

    fn new_native_prediction_with_contact_transfer(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        storage: MpmGridStorage,
        frame_dt: Real,
        transfer: SoilTractorTransfer,
        contact: PairedSoilContact,
    ) -> Result<Self> {
        let started =
            std::env::var_os("MOLLA_GPU_CONSTRUCTION_TRACE").map(|_| std::time::Instant::now());
        let trace = |phase: &str| {
            if let Some(started) = started {
                eprintln!(
                    "GPU construction soil-tractor t={:.6}s {phase}",
                    started.elapsed().as_secs_f64()
                );
            }
        };
        trace("validation begin");
        config.validate()?;
        coupling.validate()?;
        transfer.validate_with_grid(config.particle_count()?, config.grid_size)?;
        let substeps = (frame_dt / config.max_substep_s).ceil() as usize;
        if !frame_dt.is_finite()
            || frame_dt <= 0.0
            || frame_dt > 0.1
            || !(1..=1024).contains(&substeps)
            || !friction.is_finite()
            || friction < 0.0
            || engine
                .device()
                .is_none_or(|d| !d.features().contains(wgpu::Features::SHADER_F64))
        {
            return Err(invalid(
                "invalid native prediction frame, friction or f64 device",
            ));
        }
        trace("soil begin");
        let mut soil = config.soil_backend_with_grid(&engine, true, storage)?;
        trace("soil ready");
        transfer.prepare(
            &engine,
            &mut soil,
            config.particle_count()?,
            config.grid_size,
            trace,
        )?;
        trace("rigid attachment begin");
        let mut tractor = Self::from_soil_backend(
            engine,
            config,
            soil,
            molla_kernels::FeatherstoneRounding::Storage,
            molla_kernels::FeatherstoneExecution::Generic,
            true,
            Some(frame_dt / substeps as Real),
        )?;
        let coefficients = tractor
            .handles
            .wheels
            .into_iter()
            .chain([tractor.handles.tine])
            .map(|body| (body, friction))
            .collect::<Vec<_>>();
        trace("rigid attachment ready; coupling begin");
        contact.enable(&mut tractor.world, coupling, &coefficients)?;
        trace("coupling ready");
        Ok(tractor)
    }

    fn new_native_explicit_with_paired_friction(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        coupling: crate::mpm::MpmUnilateralConfig,
        friction: f32,
        contact: PairedSoilContact,
        execution: molla_kernels::FeatherstoneExecution,
        storage: MpmGridStorage,
        transfer: SoilTractorTransfer,
    ) -> Result<Self> {
        #[cfg(test)]
        let _setup = crate::soil_setup_trace::stage("tractor.coupled");
        if !matches!(
            execution,
            molla_kernels::FeatherstoneExecution::Generic
                | molla_kernels::FeatherstoneExecution::ExperimentalJointTypes
        ) {
            return Err(invalid(
                "paired soil tractor requires generic or joint-types execution",
            ));
        }
        coupling.validate()?;
        if !friction.is_finite() || friction < 0.0 {
            return Err(invalid("finite nonnegative friction coefficient required"));
        }
        config.validate_grid_storage(storage)?;
        transfer.validate_with_grid(config.particle_count()?, config.grid_size)?;
        let mut soil = config.soil_backend_with_grid(&engine, true, storage)?;
        transfer.prepare(
            &engine,
            &mut soil,
            config.particle_count()?,
            config.grid_size,
            |_| {},
        )?;
        let mut tractor = Self::from_soil_backend(
            engine,
            config,
            soil,
            molla_kernels::FeatherstoneRounding::Storage,
            execution,
            true,
            None,
        )?;
        let coefficients = tractor
            .handles
            .wheels
            .into_iter()
            .chain([tractor.handles.tine])
            .map(|body| (body, friction))
            .collect::<Vec<_>>();
        contact.enable(&mut tractor.world, coupling, &coefficients)?;
        Ok(tractor)
    }

    fn new_backend(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        native: bool,
        rounding: molla_kernels::FeatherstoneRounding,
        execution: molla_kernels::FeatherstoneExecution,
        paired_rates: bool,
    ) -> Result<Self> {
        Self::new_backend_with_grid(
            engine,
            config,
            native,
            rounding,
            execution,
            paired_rates,
            MpmGridStorage::Dense,
        )
    }

    fn new_backend_with_grid(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        native: bool,
        rounding: molla_kernels::FeatherstoneRounding,
        execution: molla_kernels::FeatherstoneExecution,
        paired_rates: bool,
        storage: MpmGridStorage,
    ) -> Result<Self> {
        let soil = config.soil_backend_with_grid(&engine, native, storage)?;
        Self::from_soil_backend(
            engine,
            config,
            soil,
            rounding,
            execution,
            paired_rates,
            None,
        )
    }

    fn from_soil_backend(
        engine: Arc<Engine>,
        config: SoilTractorConfig,
        soil: MpmSimulation,
        rounding: molla_kernels::FeatherstoneRounding,
        execution: molla_kernels::FeatherstoneExecution,
        paired_rates: bool,
        native_dt: Option<Real>,
    ) -> Result<Self> {
        #[cfg(test)]
        let _setup = crate::soil_setup_trace::stage("tractor.machine");
        let (scene, handles) = build_soil_tractor_scene()?;
        let construct = if paired_rates {
            RigidWorld::from_scene_gpu_with_paired_machine_options
        } else {
            RigidWorld::from_scene_gpu_with_machine_options
        };
        let mut world = if let Some(dt) = native_dt {
            let bytes = 2 * crate::native_prediction_gpu::PreparedNativePrediction::required_bytes(
                scene.model(),
            )?;
            RigidWorld::from_scene_gpu_with_native_prediction(scene, engine, dt, bytes)?
        } else {
            construct(
                scene,
                engine,
                crate::rigid_world::GpuContactCapacity::default(),
                rounding,
                execution,
            )?
        };
        let mut bindings: Vec<_> = handles
            .wheels
            .into_iter()
            .map(|body| MpmRigidDeviceBinding {
                body,
                shape: MpmSolidShape::Cylinder {
                    radius: WHEEL_RADIUS as f32,
                    half_width: WHEEL_HALF_WIDTH as f32,
                },
            })
            .collect();
        bindings.push(MpmRigidDeviceBinding {
            body: handles.tine,
            shape: MpmSolidShape::Box {
                half_extents: TINE_HALF_EXTENTS.as_vec3().to_array(),
            },
        });
        world.attach_mpm_soil(soil, bindings, config.coupling_budget_bytes)?;
        Ok(Self {
            world,
            handles,
            config,
            simulated_seconds: 0.0,
        })
    }

    #[cfg(all(test, feature = "publication-products-newton"))]
    pub(crate) fn new_ordered_newton_test(engine: Arc<Engine>) -> Result<Self> {
        let config = SoilTractorConfig::default();
        Self::new_native_explicit_with_paired_newton_transfer(
            engine,
            config,
            crate::mpm::MpmUnilateralConfig {
                node_capacity: 4096,
                search_distance: 0.05,
                max_iterations: 64,
            },
            0.3,
            molla_kernels::FeatherstoneExecution::Generic,
            MpmGridStorage::Dense,
            SoilTractorTransfer::FixedOrder {
                max_gather_bytes: u64::from(config.particle_count()?) * 64,
            },
        )
    }

    /// Apply per-wheel drive torque; positive torque drives toward chassis +X.
    /// Completed simulated time advances only after a successful whole frame.
    pub fn step(&mut self, dt: Real, drive_torque_nm: [Real; 4]) -> Result<()> {
        if !dt.is_finite()
            || dt <= 0.0
            || dt > 0.1
            || drive_torque_nm
                .iter()
                .any(|t| !t.is_finite() || t.abs() > 10_000.0)
        {
            return Err(invalid("invalid frame interval or bounded wheel torque"));
        }
        let substeps = (dt / self.config.max_substep_s).ceil() as usize;
        if substeps == 0 || substeps > 1024 {
            return Err(invalid("substep capacity exceeded"));
        }
        for axle in self.handles.axles {
            if self.world.scene.joint(axle).is_none() {
                return Err(invalid("wheel articulation removed"));
            }
        }
        for (axle, torque) in self.handles.axles.into_iter().zip(drive_torque_nm) {
            self.world.scene.set_joint_motor(
                axle,
                JointAxis::AngX,
                JointMotor {
                    mode: JointTargetMode::Effort,
                    target_position: -torque,
                    ..Default::default()
                },
            )?;
        }
        self.world.solver.substeps = substeps;
        self.world.step(dt)?;
        self.simulated_seconds += dt;
        Ok(())
    }

    pub fn telemetry(&self) -> Result<SoilTractorTelemetry> {
        let pose = self.world.scene.body_pose(self.handles.chassis)?;
        let forward = pose.rotation * Vec3::X;
        let mut sample = SoilTractorTelemetry {
            simulated_seconds: self.simulated_seconds,
            chassis_pose: pose,
            wheel_surface_speed_m_s: [0.0; 4],
            wheel_forward_speed_m_s: [0.0; 4],
            nominal_slip: [0.0; 4],
        };
        for (i, wheel) in self.handles.wheels.into_iter().enumerate() {
            let velocity = self.world.scene.body_velocity(wheel)?;
            let axle = self.world.scene.body_pose(wheel)?.rotation * Vec3::Y;
            let rolling = -velocity.angular.dot(axle) * WHEEL_RADIUS;
            let travel = velocity.linear.dot(forward);
            sample.wheel_surface_speed_m_s[i] = rolling;
            sample.wheel_forward_speed_m_s[i] = travel;
            sample.nominal_slip[i] = (rolling - travel) / rolling.abs().max(travel.abs()).max(0.1);
        }
        Ok(sample)
    }

    /// Explicit blocking last-substep soil force/COM-torque observation.
    pub fn read_soil_loads(&mut self) -> Result<MpmSoilWrenchFrame> {
        self.world.read_mpm_soil_wrenches()
    }
}

#[cfg(test)]
#[path = "soil_tractor_tests.rs"]
mod tests;
