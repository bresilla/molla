// T5 — Realistic tire test bench.
//
// One large volumetric tetra-FEM tire sits stationary at the world
// origin while a textured "rolling road" moves under it. The user
// dials in three independent slip inputs:
//
//   * belt speed v_belt  (how fast the road moves under the tire)
//   * slip ratio  κ      (drives wheel ω = (1+κ)·v_belt/R; +κ → spin,
//                         −κ → lock)
//   * slip angle  α      (the road's motion direction is rotated by α
//                         relative to the tire's heading axis)
//
// The brush model is evaluated each substep with these inputs and
// returns the cornering forces (Fx, Fy, Mz) and dissipated power.
// Multi-layer surface/tread/carcass/rim heat + wear integrate; their
// μ-factor feeds back into the friction cap so a cold, hot or worn tire
// visibly loses grip. The FEM carcass (`TetraTireFem`) runs underneath
// with the kinematic hub, so the mesh visibly deforms under the
// prescribed vertical load.
//
// Everything is staged like a real tire-test rig:
//
//   * The tire stays put — the wheel mesh never wanders sideways.
//   * A 3-D road strip beneath it scrolls in the (−belt) direction,
//     showing rolling. When slip angle is non-zero the stripes also
//     drift sideways so the angle is immediately visible.
//   * Three big force arrows at the contact patch (red Fx along
//     heading, cyan Fy perpendicular, white Fn vertical).
//   * A “heading” arrow and a “relative-velocity” arrow above the
//     tire make the slip angle unmistakable.
//   * The tire body tints cold-blue → optimal-green → overheated-red
//     with temperature and darkens with wear.
//   * Live charts: Fy versus slip angle (the canonical cornering
//     curve) and Fx versus slip ratio (the canonical traction curve).

const TB_TIRE_RADIUS: f64 = 0.32;
const TB_INNER_RADIUS: f64 = 0.12;
const TB_TIRE_WIDTH: f64 = 0.185;
const TB_SEGMENTS: usize = 24;
const TB_FEM_FEEDBACK_MAX_BYTES: u64 = 64 * 1024 * 1024;

fn tb_fem_terrain_feedback(
    prepared: &mut Option<molla::kernels::PreparedTireFemTerrainFeedback>,
    retained: bool,
    engine: &molla::compute::Engine,
    kernel: &molla::kernels::TireFemContactFeedbackKernel,
    input: &molla::kernels::TireFemContactFeedbackInput,
    terrain: &molla::kernels::TireFemContactFeedbackGpuTerrain<'_>,
) -> molla::core::Result<molla::kernels::TireFemContactFeedbackOutput> {
    if !retained {
        *prepared = None;
        engine.check_device()?;
        let device = engine.device().ok_or_else(|| molla::core::Error::Gpu("GPU device required".into()))?;
        let queue = engine.queue().ok_or_else(|| molla::core::Error::Gpu("GPU queue required".into()))?;
        let output = kernel.dispatch_resident_with_gpu_terrain(device, queue, input, terrain)?;
        return kernel.readback_resident_feedback_output(device, queue, &output);
    }
    if prepared.as_ref().is_none_or(|p| {
        !p.matches(engine, kernel, input.positions.len(), input.tread_ids.len(), terrain)
    }) {
        *prepared = Some(molla::kernels::PreparedTireFemTerrainFeedback::new(
            engine,
            kernel,
            input.positions.len(),
            input.tread_ids.len(),
            terrain,
            TB_FEM_FEEDBACK_MAX_BYTES,
        )?);
    }
    prepared.as_mut().unwrap().run_blocking(input, terrain)
}
// Defaults tuned so the FEM tire shows a *visible* deflection out of
// the gate. ~5 kN load on a 150-kPa tire gives roughly 25–35 mm hub
// drop on the default mesh resolution.
const TB_PRESSURE_DEFAULT: f32 = 150_000.0;
const TB_VERTICAL_LOAD_DEFAULT: f32 = 5_000.0;
const TB_BELT_SPEED_DEFAULT: f32 = 8.0;
const TB_SLIP_ANGLE_DEFAULT: f32 = 0.0;
const TB_SLIP_RATIO_DEFAULT: f32 = 0.0;

const TB_CURVE_SAMPLES: usize = 91;
const TB_BEKKER_NX: u32 = 61;
const TB_BEKKER_NZ: u32 = 31;
const TB_BEKKER_SPACING: f32 = 0.04;
const TB_BEKKER_ORIGIN: [f32; 2] = [
    -(TB_BEKKER_NX as f32 - 1.0) * TB_BEKKER_SPACING * 0.5,
    -(TB_BEKKER_NZ as f32 - 1.0) * TB_BEKKER_SPACING * 0.5,
];
const TB_MPM_NX: u32 = 41;
const TB_MPM_NZ: u32 = 25;
const TB_MPM_SPACING: f32 = 0.04;
const TB_MPM_ORIGIN: [f32; 2] = [
    -(TB_MPM_NX as f32 - 1.0) * TB_MPM_SPACING * 0.5,
    -(TB_MPM_NZ as f32 - 1.0) * TB_MPM_SPACING * 0.5,
];
const TB_DEM_NX: u32 = 41;
const TB_DEM_NZ: u32 = 25;
const TB_DEM_SPACING: f32 = 0.045;
const TB_DEM_ORIGIN: [f32; 2] = [
    -(TB_DEM_NX as f32 - 1.0) * TB_DEM_SPACING * 0.5,
    -(TB_DEM_NZ as f32 - 1.0) * TB_DEM_SPACING * 0.5,
];
const TB_DEM_GRAIN_RADIUS: f32 = 0.025;
const TB_DEM_GRAIN_MASS: f32 = 0.18;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchMaterial {
    Asphalt,
    Dirt,
    Sand,
    Ice,
}

impl BenchMaterial {
    fn material(self) -> molla_vehicle::SurfaceMaterial {
        match self {
            BenchMaterial::Asphalt => molla_vehicle::SurfaceMaterial::dry_asphalt(),
            BenchMaterial::Dirt => molla_vehicle::SurfaceMaterial::loose_dirt(),
            BenchMaterial::Sand => molla_vehicle::SurfaceMaterial {
                mu_static: 0.65,
                mu_dynamic: 0.48,
                bristle_damping: 8.0e4,
                rolling_resistance: 0.085,
                cohesion: 600.0,
            },
            BenchMaterial::Ice => molla_vehicle::SurfaceMaterial::ice(),
        }
    }
    fn label(self) -> &'static str {
        match self {
            BenchMaterial::Asphalt => "asphalt",
            BenchMaterial::Dirt => "dirt",
            BenchMaterial::Sand => "sand",
            BenchMaterial::Ice => "ice",
        }
    }
    fn color(self) -> Color {
        match self {
            BenchMaterial::Asphalt => Color::srgb(0.32, 0.34, 0.38),
            BenchMaterial::Dirt => Color::srgb(0.55, 0.42, 0.28),
            BenchMaterial::Sand => Color::srgb(0.80, 0.68, 0.42),
            BenchMaterial::Ice => Color::srgb(0.50, 0.80, 0.95),
        }
    }
    fn road_temperature_c(self) -> f64 {
        match self {
            BenchMaterial::Asphalt => 25.0,
            BenchMaterial::Dirt => 22.0,
            BenchMaterial::Sand => 32.0,
            BenchMaterial::Ice => -5.0,
        }
    }
}

fn build_tire_realistic_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("tire bench floor");
    (b.build().expect("tire bench model"), Vec::new())
}

fn tb_new_bekker_owner() -> molla::kernels::TireBekkerFemTerrainOwner {
    let mut owner = molla::kernels::TireBekkerFemTerrainOwner::flat(
        TB_BEKKER_ORIGIN,
        TB_BEKKER_SPACING,
        TB_BEKKER_NX,
        TB_BEKKER_NZ,
        0.0,
        0.8,
        820_000.0,
        0.16,
        11_000.0,
        0.08,
        0.95,
    )
    .expect("tire bench Bekker terrain owner");
    owner.set_soil_rolling_resistance_scale(1.4);
    owner
}

fn tb_vehicle_heightfield_from_bekker(
    owner: &molla::kernels::TireBekkerFemTerrainOwner,
    friction: f64,
    water_film_m: f64,
) -> molla_vehicle::HeightfieldTerrain {
    let input = owner.bekker_input();
    let heights = owner.heights().iter().map(|&h| h as f64).collect::<Vec<_>>();
    let mut terrain = molla_vehicle::HeightfieldTerrain::new(
        [input.origin[0] as f64, input.origin[1] as f64],
        input.spacing as f64,
        input.nx as usize,
        input.nz as usize,
        heights,
        friction,
    );
    if water_film_m > 0.0 {
        terrain = terrain.with_water_film_map(vec![
            water_film_m;
            input.nx as usize * input.nz as usize
        ]);
    }
    terrain
}

fn tb_vehicle_heightfield_from_mpm(
    terrain: &molla::kernels::TireFemContactTerrain,
    friction: f64,
    water_film_m: f64,
) -> molla_vehicle::HeightfieldTerrain {
    let heights = terrain.heights.iter().map(|&h| h as f64).collect::<Vec<_>>();
    let mut heightfield = molla_vehicle::HeightfieldTerrain::new(
        [terrain.origin[0] as f64, terrain.origin[1] as f64],
        terrain.spacing as f64,
        terrain.nx as usize,
        terrain.nz as usize,
        heights,
        friction,
    );
    if water_film_m > 0.0 {
        heightfield = heightfield.with_water_film_map(vec![
            water_film_m;
            terrain.nx as usize * terrain.nz as usize
        ]);
    }
    heightfield
}

fn tb_vehicle_heightfield_from_dem(
    owner: &molla::kernels::TireDemFemTerrainOwner,
    friction: f64,
    water_film_m: f64,
) -> Option<molla_vehicle::HeightfieldTerrain> {
    let terrain = owner.fem_contact_terrain().ok()?;
    let heights = terrain.heights.iter().map(|&h| h as f64).collect::<Vec<_>>();
    let mut heightfield = molla_vehicle::HeightfieldTerrain::new(
        [terrain.origin[0] as f64, terrain.origin[1] as f64],
        terrain.spacing as f64,
        terrain.nx as usize,
        terrain.nz as usize,
        heights,
        friction,
    );
    if water_film_m > 0.0 {
        heightfield = heightfield.with_water_film_map(vec![
            water_film_m;
            terrain.nx as usize * terrain.nz as usize
        ]);
    }
    Some(heightfield)
}

fn tb_bekker_contacts_from_tire_contact(
    contact: &molla_vehicle::RealTireContact,
    hub_pos: [f64; 3],
) -> Vec<molla::kernels::TireBekkerContactCell> {
    contact
        .cells
        .iter()
        .map(|cell| {
            let area = cell.area.max(1.0e-6);
            let shear = (cell.tangential_force * cell.tangential_force
                + cell.lateral_force * cell.lateral_force)
                .sqrt()
                / area;
            molla::kernels::TireBekkerContactCell {
                xz_pressure: [
                    (hub_pos[0] + cell.local_x) as f32,
                    (hub_pos[2] + cell.local_z) as f32,
                    (cell.normal_load / area).max(0.0) as f32,
                    shear.max(0.0) as f32,
                ],
            }
        })
        .collect()
}

fn tb_dem_contacts_from_tire_contact(
    contact: &molla_vehicle::RealTireContact,
    wheel: &molla_vehicle::RealWheelState,
) -> Vec<molla::kernels::TireDemContactCell> {
    let heading = {
        let len = (wheel.heading[0] * wheel.heading[0] + wheel.heading[2] * wheel.heading[2])
            .sqrt()
            .max(1.0e-9);
        [wheel.heading[0] / len, 0.0, wheel.heading[2] / len]
    };
    let side = [-heading[2], 0.0, heading[0]];
    contact
        .cells
        .iter()
        .map(|cell| {
            let area = cell.area.max(1.0e-6);
            let radius = (area / std::f64::consts::PI)
                .sqrt()
                .max(TB_DEM_GRAIN_RADIUS as f64 * 1.5)
                .min(TB_TIRE_WIDTH * 0.55) as f32;
            let x = wheel.hub_pos[0] + heading[0] * cell.local_x + side[0] * cell.local_z;
            let z = wheel.hub_pos[2] + heading[2] * cell.local_x + side[2] * cell.local_z;
            let normal = cell.normal;
            let fx = -(normal[0] * cell.normal_load
                + heading[0] * cell.tangential_force
                + side[0] * cell.lateral_force);
            let fy = -(normal[1] * cell.normal_load
                + heading[1] * cell.tangential_force
                + side[1] * cell.lateral_force);
            let fz = -(normal[2] * cell.normal_load
                + heading[2] * cell.tangential_force
                + side[2] * cell.lateral_force);
            molla::kernels::TireDemContactCell {
                center_radius: [x as f32, 0.0, z as f32, radius],
                force: [fx as f32, fy as f32, fz as f32, 0.0],
            }
        })
        .collect()
}

fn tb_mpm_contacts_from_tire_contact(
    contact: &molla_vehicle::RealTireContact,
    hub_pos: [f64; 3],
) -> Vec<molla::kernels::TireMpmContactCell> {
    contact
        .cells
        .iter()
        .map(|cell| {
            let area = cell.area.max(1.0e-6);
            let pressure = (cell.normal_load / area).max(0.0) as f32;
            let radius = (cell.area.sqrt() * 1.6).clamp(0.035, 0.16) as f32;
            molla::kernels::TireMpmContactCell {
                center_radius: [
                    (hub_pos[0] + cell.local_x) as f32,
                    0.0,
                    (hub_pos[2] + cell.local_z) as f32,
                    radius,
                ],
                force_pressure: [
                    -(cell.tangential_force as f32),
                    -(cell.normal_load.max(0.0) as f32),
                    -(cell.lateral_force as f32),
                    pressure,
                ],
            }
        })
        .collect()
}

fn tb_mpm_particle_bed() -> Vec<[f32; 3]> {
    let mut particles = Vec::new();
    let nx = 22;
    let nz = 14;
    let layers = 2;
    let spacing = 0.035_f32;
    for y in 0..layers {
        for z in 0..nz {
            for x in 0..nx {
                particles.push([
                    (x as f32 - (nx as f32 - 1.0) * 0.5) * spacing,
                    0.018 + y as f32 * spacing * 0.80,
                    (z as f32 - (nz as f32 - 1.0) * 0.5) * spacing,
                ]);
            }
        }
    }
    particles
}

fn tb_dem_particle_bed() -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let mut positions = Vec::new();
    let nx = 24;
    let nz = 14;
    let layers = 2;
    let spacing = TB_DEM_GRAIN_RADIUS * 2.05;
    for y in 0..layers {
        for z in 0..nz {
            for x in 0..nx {
                let jitter = (((x * 17 + z * 31 + y * 13) % 11) as f32 - 5.0) * 0.0012;
                positions.push([
                    (x as f32 - (nx as f32 - 1.0) * 0.5) * spacing + jitter,
                    TB_DEM_GRAIN_RADIUS + y as f32 * spacing * 0.86,
                    (z as f32 - (nz as f32 - 1.0) * 0.5) * spacing - jitter,
                    0.0,
                ]);
            }
        }
    }
    let velocities = vec![[0.0; 4]; positions.len()];
    (positions, velocities)
}


fn tb_new_mpm_sim(engine: &molla::compute::Engine, queue: &wgpu::Queue) -> molla::core::Result<molla::compute::MpmSimulation> {
    let cfg = molla::compute::MpmConfig {
        grid_size: 32,
        cell_size: 0.04,
        grid_origin: [-0.70, -0.02, -0.70],
        gravity_y: -5.0,
        youngs: 3.5e4,
        poisson: 0.30,
        friction_angle: 33.0_f32.to_radians(),
        cohesion: 0.0,
        boundary_friction: 0.55,
        particle_mass: 1.5e-3,
        particle_volume: 1.0e-6,
        max_particles: 4096,
        max_rigid: 0,
        rheology: molla::compute::MpmRheology::Sand {
            friction_angle: 33.0_f32.to_radians(),
        },
        time_step: molla::compute::MpmTimeStep::Explicit,
    };
    let mut sim = molla::compute::MpmSimulation::new(engine, cfg)?;
    sim.set_particles(queue, &tb_mpm_particle_bed())?;
    Ok(sim)
}

fn tb_mpm_raster(friction: f32, surface_velocity: [f32; 3]) -> molla::compute::MpmTireTerrainRaster {
    let mut raster =
        molla::compute::MpmTireTerrainRaster::new(TB_MPM_ORIGIN, TB_MPM_SPACING, TB_MPM_NX, TB_MPM_NZ);
    raster.rest_height = 0.0;
    raster.friction = friction.max(0.05);
    raster.soil_rolling_resistance_scale = 1.25;
    raster.surface_velocity = surface_velocity;
    raster
}

fn tb_gpu_thermal_state_from_vehicle(
    thermal: &molla_vehicle::TireThermalLayers,
    wear: &molla_vehicle::TireWearState,
) -> molla::kernels::TireThermalGpuState {
    molla::kernels::TireThermalGpuState {
        temperatures: [
            thermal.surface_c as f32,
            thermal.tread_c as f32,
            thermal.carcass_c as f32,
            thermal.rim_c as f32,
        ],
        wear: [
            wear.depth_m as f32,
            wear.max_depth_m as f32,
            wear.wear_rate_m3_per_j as f32,
            wear.ref_patch_area_m2 as f32,
        ],
        capacities: [
            thermal.surface_heat_capacity_j_per_k as f32,
            thermal.tread_heat_capacity_j_per_k as f32,
            thermal.carcass_heat_capacity_j_per_k as f32,
            thermal.rim_heat_capacity_j_per_k as f32,
        ],
        coeffs0: [
            thermal.ambient_c as f32,
            thermal.road_conductance_w_per_k as f32,
            thermal.air_conductance_w_per_k as f32,
            thermal.surface_tread_conductance_w_per_k as f32,
        ],
        coeffs1: [
            thermal.tread_carcass_conductance_w_per_k as f32,
            thermal.carcass_rim_conductance_w_per_k as f32,
            0.0,
            0.0,
        ],
    }
}

fn tb_apply_gpu_thermal_state(
    state: &mut TireRealisticState,
    gpu_state: molla::kernels::TireThermalGpuState,
) {
    state.thermal.surface_c = gpu_state.temperatures[0] as f64;
    state.thermal.tread_c = gpu_state.temperatures[1] as f64;
    state.thermal.carcass_c = gpu_state.temperatures[2] as f64;
    state.thermal.rim_c = gpu_state.temperatures[3] as f64;
    state.thermal.surface_heat_capacity_j_per_k = gpu_state.capacities[0] as f64;
    state.thermal.tread_heat_capacity_j_per_k = gpu_state.capacities[1] as f64;
    state.thermal.carcass_heat_capacity_j_per_k = gpu_state.capacities[2] as f64;
    state.thermal.rim_heat_capacity_j_per_k = gpu_state.capacities[3] as f64;
    state.thermal.ambient_c = gpu_state.coeffs0[0] as f64;
    state.thermal.road_conductance_w_per_k = gpu_state.coeffs0[1] as f64;
    state.thermal.air_conductance_w_per_k = gpu_state.coeffs0[2] as f64;
    state.thermal.surface_tread_conductance_w_per_k = gpu_state.coeffs0[3] as f64;
    state.thermal.tread_carcass_conductance_w_per_k = gpu_state.coeffs1[0] as f64;
    state.thermal.carcass_rim_conductance_w_per_k = gpu_state.coeffs1[1] as f64;
    state.wear.depth_m = gpu_state.wear[0] as f64;
    state.wear.max_depth_m = gpu_state.wear[1] as f64;
    state.wear.wear_rate_m3_per_j = gpu_state.wear[2] as f64;
    state.wear.ref_patch_area_m2 = gpu_state.wear[3] as f64;
}

fn tb_pressure_bin_loads(bins: &[molla_vehicle::TirePressureDistributionBin]) -> Vec<f32> {
    bins.iter().map(|bin| bin.normal_load_n as f32).collect()
}

fn tb_format_pressure_bins(loads: &[f32]) -> String {
    if loads.is_empty() {
        return "none".to_owned();
    }
    let peak = loads
        .iter()
        .copied()
        .fold(0.0_f32, |a, b| a.max(b.abs()))
        .max(1.0);
    loads
        .iter()
        .map(|load| {
            let bars = ((*load / peak).clamp(0.0, 1.0) * 8.0).round() as usize;
            format!("{:>4.0}N {}", load, "▉".repeat(bars.max(1)))
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn tb_transient_gpu_input(
    raw_kappa: f32,
    raw_alpha_rad: f32,
    belt_speed_mps: f32,
    dt: f32,
) -> molla::kernels::TireTransientGpuInput {
    let ds = belt_speed_mps.abs().max(0.05) * dt.max(0.0);
    molla::kernels::TireTransientGpuInput {
        command: [
            raw_kappa.clamp(-4.0, 4.0),
            raw_alpha_rad.clamp(-1.2, 1.2),
            ds,
            TB_TIRE_RADIUS as f32,
        ],
        tread_lengths: [0.28, 0.34, 0.42, 1.0],
        belt_lengths: [0.55, 0.68, 0.78, 0.0],
        carcass_lengths: [0.95, 1.10, 1.25, 0.0],
    }
}

fn tb_default_feedback_terrain(
    friction: f32,
    surface_velocity: [f32; 3],
) -> molla::kernels::TireFemContactTerrain {
    molla::kernels::TireFemContactTerrain {
        origin: [0.0, 0.0],
        spacing: 1.0,
        nx: 1,
        nz: 1,
        heights: vec![0.0],
        friction: friction.max(0.05),
        friction_map: None,
        water_film_map: None,
        sinkage_map: None,
        soil_rolling_resistance_scale: 1.0,
        surface_velocity,
        surface_velocity_map: None,
        sdf_primitives: Vec::new(),
        triangles: Vec::new(),
    }
}

#[derive(Resource)]
struct TireRealisticState {
    tire: Option<molla_vehicle::TetraTireFem>,
    thermal: molla_vehicle::TireThermalLayers,
    wear: molla_vehicle::TireWearState,
    pressure_pa: f32,
    vertical_load_n: f32,
    /// Rolling road speed (m/s). Positive moves the road in (−heading,
    /// rotated by slip angle).
    belt_speed_mps: f32,
    /// Slip ratio κ. ω_wheel = (1+κ)·v_belt/R.
    slip_ratio: f32,
    /// Slip angle α (rad). Road velocity vector is rotated by α
    /// relative to the tire heading.
    slip_angle_rad: f32,
    material: BenchMaterial,
    /// Water/film depth on the road surface. This feeds the reusable
    /// `SurfaceFilmState`, not a viewer-only force multiplier.
    water_film_mm: f32,
    /// Prefer the GPU batch path for four-node thermal + wear stepping.
    use_gpu_thermal: bool,
    /// Feed transient-relaxed slip into the FEM tread feedback path.
    use_gpu_transient_fem_feedback: bool,
    /// Cumulative road offset for visualisation.
    road_phase_x: f32,
    road_phase_z: f32,
    /// Visual wheel angle so the single rig tire visibly spins.
    wheel_angle: f32,
    /// Last brush snapshot for visualisation + readouts.
    last_fx_n: f32,
    last_fy_n: f32,
    last_fn_n: f32,
    last_mz_nm: f32,
    last_dissipated_w: f32,
    last_mu_factor: f32,
    last_pressure_factor: f32,
    last_contact_count: u32,
    last_pressure_longitudinal_loads_n: Vec<f32>,
    last_pressure_lateral_loads_n: Vec<f32>,
    last_pressure_longitudinal_span_m: f32,
    last_pressure_lateral_span_m: f32,
    last_pressure_peak_pa: f32,
    last_pressure_total_load_n: f32,
    last_raw_kappa: f32,
    last_raw_alpha_rad: f32,
    last_effective_kappa: f32,
    last_effective_alpha_rad: f32,
    /// Recorded `(slip_ratio, Fx)` and `(slip_angle, Fy)` traces shown
    /// as the canonical Fy(α) and Fx(κ) sweep charts.
    fx_kappa_curve: Vec<f32>,
    fy_alpha_curve: Vec<f32>,
    auto_sweep_alpha: bool,
    sweep_phase: f32,
    reset_requested: bool,
    replay_panel: TireRigReplayPanelState,
    compare_soil_backends: bool,
    soil_summary_diagnostics: bool,
    enable_bekker_terrain: bool,
    use_gpu_bekker: bool,
    bekker_owner: Option<molla::kernels::TireBekkerFemTerrainOwner>,
    bekker_kernel: Option<molla::kernels::TireBekkerGpuKernel>,
    bekker_resident_terrain: Option<molla::kernels::TireBekkerGpuResidentTerrain>,
    last_bekker_backend: &'static str,
    last_bekker_error: Option<String>,
    last_rut_depth_m: f32,
    last_rut_mean_m: f32,
    last_rut_cells: usize,
    enable_mpm_terrain: bool,
    use_gpu_mpm: bool,
    mpm_sim: Option<molla::compute::MpmSimulation>,
    mpm_kernel: Option<molla::kernels::TireMpmGpuKernel>,
    mpm_terrain_mirror: Option<molla::kernels::TireFemContactTerrain>,
    mpm_terrain_readback: Option<molla::kernels::PreparedTireMpmTerrainReadback>,
    mpm_compaction_readback: Option<molla::kernels::PreparedTireMpmCompactionReadback>,
    mpm_terrain_revision: u64,
    last_mpm_observation: Option<(u64, u64)>,
    last_mpm_observation_error: Option<String>,
    mpm_observation_backpressure: u64,
    mpm_height_buf: Option<wgpu::Buffer>,
    mpm_sinkage_buf: Option<wgpu::Buffer>,
    mpm_surface_velocity_buf: Option<wgpu::Buffer>,
    last_mpm_backend: &'static str,
    last_mpm_error: Option<String>,
    last_mpm_compaction_m: f32,
    last_mpm_active_particles: usize,
    enable_dem_terrain: bool,
    use_gpu_dem: bool,
    dem_kernel: Option<molla::kernels::TireDemGpuKernel>,
    dem_positions: Vec<[f32; 4]>,
    dem_velocities: Vec<[f32; 4]>,
    dem_owner: Option<molla::kernels::TireDemFemTerrainOwner>,
    dem_resident_owner: Option<molla::kernels::TireDemResidentFemTerrainOwner>,
    last_dem_backend: &'static str,
    last_dem_error: Option<String>,
    last_dem_sinkage_m: f32,
    last_dem_velocity_cells: u32,
    last_dem_active_grains: usize,
    last_dem_resident_reseeds: u64,
    thermal_kernel: Option<molla::kernels::TireThermalGpuKernel>,
    thermal_batch: Option<molla::kernels::TireThermalGpuBatch>,
    transient_kernel: Option<molla::kernels::TireTransientGpuKernel>,
    transient_batch: Option<molla::kernels::TireTransientGpuBatch>,
    transient_state: molla::kernels::TireTransientGpuState,
    last_transient_backend: &'static str,
    last_transient_error: Option<String>,
    fem_feedback_kernel: Option<molla::kernels::TireFemContactFeedbackKernel>,
    fem_feedback_prepared: Option<molla::kernels::PreparedTireFemTerrainFeedback>,
    retain_fem_feedback_buffers: bool,
    fem_feedback_bristles: Vec<[f32; 4]>,
    last_fem_feedback_backend: &'static str,
    last_fem_feedback_error: Option<String>,
    last_fem_feedback_force_n: f32,
    last_fem_feedback_bristle_m: f32,
    last_fem_feedback_force_rows: usize,
    last_fem_feedback_bristle_rows: usize,
    last_fem_feedback_total_force_n: [f32; 3],
    last_fem_feedback_mean_force_n: f32,
    last_fem_feedback_mean_bristle_m: f32,
    last_thermal_backend: &'static str,
    last_thermal_error: Option<String>,
    last_replay_thermal_backend: &'static str,
    last_replay_thermal_count: usize,
    last_replay_thermal_error: Option<String>,
}

impl Default for TireRealisticState {
    fn default() -> Self {
        Self {
            tire: None,
            thermal: molla_vehicle::TireThermalLayers::default(),
            wear: molla_vehicle::TireWearState::default(),
            pressure_pa: TB_PRESSURE_DEFAULT,
            vertical_load_n: TB_VERTICAL_LOAD_DEFAULT,
            belt_speed_mps: TB_BELT_SPEED_DEFAULT,
            slip_ratio: TB_SLIP_RATIO_DEFAULT,
            slip_angle_rad: TB_SLIP_ANGLE_DEFAULT,
            material: BenchMaterial::Asphalt,
            water_film_mm: 0.0,
            use_gpu_thermal: true,
            use_gpu_transient_fem_feedback: true,
            road_phase_x: 0.0,
            road_phase_z: 0.0,
            wheel_angle: 0.0,
            last_fx_n: 0.0,
            last_fy_n: 0.0,
            last_fn_n: 0.0,
            last_mz_nm: 0.0,
            last_dissipated_w: 0.0,
            last_mu_factor: 1.0,
            last_pressure_factor: 1.0,
            last_contact_count: 0,
            last_pressure_longitudinal_loads_n: Vec::new(),
            last_pressure_lateral_loads_n: Vec::new(),
            last_pressure_longitudinal_span_m: 0.0,
            last_pressure_lateral_span_m: 0.0,
            last_pressure_peak_pa: 0.0,
            last_pressure_total_load_n: 0.0,
            last_raw_kappa: 0.0,
            last_raw_alpha_rad: 0.0,
            last_effective_kappa: 0.0,
            last_effective_alpha_rad: 0.0,
            fx_kappa_curve: vec![0.0; TB_CURVE_SAMPLES],
            fy_alpha_curve: vec![0.0; TB_CURVE_SAMPLES],
            auto_sweep_alpha: false,
            sweep_phase: 0.0,
            reset_requested: false,
            replay_panel: Default::default(),
            compare_soil_backends: false,
            soil_summary_diagnostics: false,
            enable_bekker_terrain: false,
            use_gpu_bekker: true,
            bekker_owner: None,
            bekker_kernel: None,
            bekker_resident_terrain: None,
            last_bekker_backend: "not started",
            last_bekker_error: None,
            last_rut_depth_m: 0.0,
            last_rut_mean_m: 0.0,
            last_rut_cells: 0,
            enable_mpm_terrain: true,
            use_gpu_mpm: true,
            mpm_sim: None,
            mpm_kernel: None,
            mpm_terrain_mirror: None,
            mpm_terrain_readback: None,
            mpm_compaction_readback: None,
            mpm_terrain_revision: 0,
            last_mpm_observation: None,
            last_mpm_observation_error: None,
            mpm_observation_backpressure: 0,
            mpm_height_buf: None,
            mpm_sinkage_buf: None,
            mpm_surface_velocity_buf: None,
            last_mpm_backend: "not started",
            last_mpm_error: None,
            last_mpm_compaction_m: 0.0,
            last_mpm_active_particles: 0,
            enable_dem_terrain: false,
            use_gpu_dem: true,
            dem_kernel: None,
            dem_positions: Vec::new(),
            dem_velocities: Vec::new(),
            dem_owner: None,
            dem_resident_owner: None,
            last_dem_backend: "not started",
            last_dem_error: None,
            last_dem_sinkage_m: 0.0,
            last_dem_velocity_cells: 0,
            last_dem_active_grains: 0,
            last_dem_resident_reseeds: 0,
            thermal_kernel: None,
            thermal_batch: None,
            transient_kernel: None,
            transient_batch: None,
            transient_state: Default::default(),
            last_transient_backend: "not started",
            last_transient_error: None,
            fem_feedback_kernel: None,
            fem_feedback_prepared: None,
            retain_fem_feedback_buffers: false,
            fem_feedback_bristles: Vec::new(),
            last_fem_feedback_backend: "not started",
            last_fem_feedback_error: None,
            last_fem_feedback_force_n: 0.0,
            last_fem_feedback_bristle_m: 0.0,
            last_fem_feedback_force_rows: 0,
            last_fem_feedback_bristle_rows: 0,
            last_fem_feedback_total_force_n: [0.0; 3],
            last_fem_feedback_mean_force_n: 0.0,
            last_fem_feedback_mean_bristle_m: 0.0,
            last_thermal_backend: "not started",
            last_thermal_error: None,
            last_replay_thermal_backend: "not started",
            last_replay_thermal_count: 0,
            last_replay_thermal_error: None,
        }
    }
}

impl TireRealisticState {
    fn clear_mpm_observations(&mut self) {
        self.mpm_compaction_readback = None;
        self.last_mpm_observation = None;
        self.last_mpm_observation_error = None;
        self.mpm_observation_backpressure = 0;
    }

    fn observe_mpm_compaction(&mut self, engine: &molla::compute::Engine, mpm: &molla::compute::MpmSimulation) {
        self.mpm_terrain_revision = self.mpm_terrain_revision.saturating_add(1);
        if !self.soil_summary_diagnostics {
            self.clear_mpm_observations();
            return;
        }
        let result = (|| -> molla::core::Result<_> {
            if self.mpm_compaction_readback.as_ref().is_none_or(|r| r.particle_count() != mpm.particle_count()) {
                self.clear_mpm_observations();
                self.mpm_compaction_readback = Some(mpm.prepare_tire_compaction_readback(engine, 3)?);
            }
            let reader = self.mpm_compaction_readback.as_mut().expect("MPM observation ring");
            let latest = reader.poll_latest()?.filter(|sample| reader.is_current(sample));
            let _queued = reader.try_request(self.mpm_terrain_revision)?;
            Ok((latest, reader.stats().backpressure))
        })();
        match result {
            Ok((latest, backpressure)) => {
                self.last_mpm_observation_error = None;
                self.mpm_observation_backpressure = backpressure;
                if let Some(sample) = latest {
                    self.last_mpm_observation = Some((sample.generation(), sample.revision()));
                    self.last_mpm_compaction_m = sample.summary().maximum;
                    self.last_mpm_active_particles = sample.summary().active_particles as usize;
                }
            }
            Err(error) => self.last_mpm_observation_error = Some(error.to_string()),
        }
    }
}

fn tb_tire_params(pressure_pa: f32, vertical_load_n: f32) -> molla_vehicle::TetraTireFemParams {
    molla_vehicle::TetraTireFemParams {
        outer_radius: TB_TIRE_RADIUS,
        inner_radius: TB_INNER_RADIUS,
        width: TB_TIRE_WIDTH,
        segments: TB_SEGMENTS,
        radial_layers: 3,
        width_layers: 3,
        pressure_pa: pressure_pa as f64,
        downforce: vertical_load_n as f64,
        young_modulus: 4.0e6,
        sidewall_young_modulus: 2.0e6,
        tread_young_modulus: 8.0e6,
        bead_young_modulus: 2.5e7,
        density: 1_100.0,
        node_damping: 55.0,
        bead_stiffness: 5.8e4,
        ..Default::default()
    }
}

fn tb_exclusive_soil_backends(enabled: [bool; 3]) -> [bool; 3] {
    [
        enabled[0],
        !enabled[0] && enabled[1],
        !enabled[0] && !enabled[1] && enabled[2],
    ]
}

fn tb_soil_summary(
    enabled: bool,
    values: impl Iterator<Item = f32>,
    threshold: f32,
) -> Option<(f32, usize, f32)> {
    if !enabled {
        return None;
    }
    let (mut maximum, mut active, mut sum, mut count) = (0.0_f32, 0, 0.0, 0);
    for value in values {
        maximum = maximum.max(value);
        active += usize::from(value > threshold);
        sum += value;
        count += 1;
    }
    let mean = if active > 0 { sum / count as f32 } else { 0.0 };
    Some((maximum, active, mean))
}

#[cfg(test)]
mod tire_testbench_soil_selection_tests {
    use super::*;

    #[test]
    #[ignore = "requires a real GPU adapter"]
    fn soil_retained_fem_bridge_uses_real_mesh_and_rebuilds() {
        let engine = molla::compute::Engine::new_gpu().unwrap();
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut kernel = molla::kernels::TireFemContactFeedbackKernel::new(device);
        let params = tb_tire_params(TB_PRESSURE_DEFAULT, TB_VERTICAL_LOAD_DEFAULT);
        let mut tire = molla_vehicle::TetraTireFem::new(0.0, params.pressure_pa, params);
        let mut prepared = None;
        let mut bristles = vec![];
        for generation in 0..3 {
            let maps = [36, 36, 144].map(|size| device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewer FEM bridge terrain fixture"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let terrain = molla::kernels::TireFemContactFeedbackGpuTerrain {
                origin: [-1.0, -1.0], spacing: 1.0, nx: 3, nz: 3,
                heights: &maps[0], friction: 0.7,
                friction_map: None, water_film_map: None,
                sinkage_map: Some(&maps[1]), surface_velocity_map: Some(&maps[2]),
                soil_rolling_resistance_scale: 1.0, surface_velocity: [0.0; 3],
            };
            if generation == 1 {
                kernel = molla::kernels::TireFemContactFeedbackKernel::new(device);
            }
            if generation == 2 {
                prepared = None;
                tire = molla_vehicle::TetraTireFem::new(0.0, params.pressure_pa, params);
                bristles.clear();
            }
            for step in 0..4 {
                let snapshot = tire.gpu_snapshot();
                bristles.resize(snapshot.tread_ids.len(), [0.0; 4]);
                let heights: Vec<u8> = [0.005_f32 * step as f32; 9]
                    .into_iter().flat_map(f32::to_le_bytes).collect();
                queue.write_buffer(&maps[0], 0, &heights);
                let input = molla::kernels::TireFemContactFeedbackInput {
                    positions: snapshot.positions,
                    velocities: snapshot.velocities,
                    masses: snapshot.masses,
                    tread_ids: snapshot.tread_ids,
                    bristles: bristles.clone(), terrain: None,
                    params: molla::kernels::TireFemContactFeedbackParams {
                        dt: 1.0 / 240.0, omega: 3.0, contact_band: 0.055,
                        normal_load_per_tread: 50.0,
                        ..Default::default()
                    },
                };
                let old = kernel.dispatch_resident_with_gpu_terrain(device, queue, &input, &terrain).unwrap();
                let old = kernel.readback_resident_feedback_output(device, queue, &old).unwrap();
                let retained = step != 2;
                let result = tb_fem_terrain_feedback(&mut prepared, retained, &engine, &kernel, &input, &terrain).unwrap();
                assert_eq!(result.velocities, old.velocities);
                assert_eq!(result.bristles, old.bristles);
                assert_eq!(result.forces, old.forces);
                assert!(result.forces.iter().flatten().any(|v| v.abs() > 0.01));
                let hub = tire.hub();
                tire.apply_gpu_node_state(&input.positions, &result.velocities,
                    [hub[0] as f32, hub[1] as f32, hub[2] as f32, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
                assert_eq!(tire.gpu_snapshot().velocities, result.velocities);
                bristles = result.bristles;
                if retained {
                    assert!(prepared.as_ref().unwrap().matches(&engine, &kernel,
                        input.positions.len(), input.tread_ids.len(), &terrain));
                } else {
                    assert!(prepared.is_none());
                }
                println!("viewer FEM bridge generation={generation} step={step} nodes={} treads={} exact=true", input.positions.len(), input.tread_ids.len());
            }
        }
    }

    #[test]
    fn soil_mpm_observation_reset_clears_labels_without_changing_terrain_revision() {
        let mut state = TireRealisticState::default();
        state.mpm_terrain_revision = 43;
        state.last_mpm_observation = Some((1, 42));
        state.last_mpm_observation_error = Some("old telemetry failure".into());
        state.mpm_observation_backpressure = 3;
        state.clear_mpm_observations();
        assert_eq!(state.mpm_terrain_revision, 43);
        assert!(state.last_mpm_observation.is_none());
        assert!(state.last_mpm_observation_error.is_none());
        assert_eq!(state.mpm_observation_backpressure, 0);
    }

    #[test]
    fn soil_comparison_exit_preserves_contact_backend_priority() {
        let defaults = TireRealisticState::default();
        assert!(!defaults.retain_fem_feedback_buffers);
        assert!(!defaults.compare_soil_backends);
        assert!(!defaults.soil_summary_diagnostics);
        assert_eq!([
            defaults.enable_mpm_terrain, defaults.enable_bekker_terrain, defaults.enable_dem_terrain,
        ], [true, false, false]);
        for mask in 0..8 {
            let enabled = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
            let selected = tb_exclusive_soil_backends(enabled);
            assert_eq!(
                selected.iter().filter(|&&on| on).count(),
                usize::from(mask != 0)
            );
            assert_eq!(
                enabled.iter().position(|&on| on),
                selected.iter().position(|&on| on)
            );
            assert_eq!(tb_exclusive_soil_backends(selected), selected);
        }
    }

    #[test]
    fn soil_disabled_summary_does_not_visit_samples_or_change_physics() {
        let unused = std::iter::from_fn(|| -> Option<f32> {
            panic!("disabled summary must not read samples")
        });
        assert_eq!(tb_soil_summary(false, unused, 1e-4), None);
        assert_eq!(tb_soil_summary(true, [].into_iter(), 1e-4), Some((0.0, 0, 0.0)));
        assert_eq!(tb_soil_summary(true, [0.0, 0.1, 0.2, 0.1].into_iter(), 0.1), Some((0.2, 1, 0.1)));
        let mut observed = tb_new_bekker_owner();
        let mut unobserved = observed.clone();
        for step in 0..8 {
            let contacts = [molla::kernels::TireBekkerContactCell {
                xz_pressure: [step as f32 * 0.04, 0.0, 20_000.0 + step as f32 * 4_000.0, 15_000.0],
            }];
            observed.apply_contacts_cpu(&contacts).unwrap();
            unobserved.apply_contacts_cpu(&contacts).unwrap();
            let _ = tb_soil_summary(true, observed.sinkage().iter().copied(), 1e-4);
            assert_eq!(tb_soil_summary(false, unobserved.sinkage().iter().copied(), 1e-4), None);
            assert_eq!(observed.heights(), unobserved.heights());
            assert_eq!(observed.sinkage(), unobserved.sinkage());
        }
    }

    #[test]
    fn soil_mpm_terrain_mirror_preserves_cpu_contact_queries() {
        use molla_vehicle::Terrain;
        let params = molla::kernels::TireMpmTerrainRasterParams {
            particle_count: 0, origin: [-1.0, -2.0], spacing: 0.5, nx: 2, nz: 2, rest_height: 0.0,
        };
        let rows = molla::kernels::TireMpmTerrainRasterGpuOutput {
            heights: vec![0.0, 0.125, 0.25, 0.375], sinkage: vec![0.0; 4],
            surface_velocity_map: vec![[0.0; 4]; 4],
        };
        let terrain = rows.into_fem_contact_terrain(params, 0.6, 1.4, [0.0; 3]).unwrap();
        let mirror = tb_vehicle_heightfield_from_mpm(&terrain, 0.8, 0.002);
        assert_eq!(mirror.height(-1.0, -2.0), 0.0);
        assert_eq!(mirror.height(-0.5, -1.5), 0.375);
        assert_eq!(mirror.height(-0.75, -1.75), 0.1875);
        let reference = molla_vehicle::HeightfieldTerrain::new(
            [-1.0, -2.0], 0.5, 2, 2, vec![0.0, 0.125, 0.25, 0.375], 0.8,
        ).with_water_film_map(vec![0.002; 4]);
        for (x, z) in [(-1.0, -2.0), (-0.75, -1.75), (-0.5, -1.5), (4.0, -5.0)] {
            assert_eq!(mirror.normal(x, z), reference.normal(x, z));
            assert_eq!(mirror.friction_at(x, z), 0.8);
            assert_eq!(mirror.water_film_depth(x, z), 0.002);
        }
    }
}

fn update_tire_realistic_example(
    _time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut state: ResMut<TireRealisticState>,
) {
    if sim.scene != Scene::TireRealistic {
        return;
    }
    if state.reset_requested {
        state.fem_feedback_prepared = None;
        state.tire = None;
        state.thermal = molla_vehicle::TireThermalLayers::default();
        state.wear = molla_vehicle::TireWearState::default();
        state.road_phase_x = 0.0;
        state.road_phase_z = 0.0;
        state.wheel_angle = 0.0;
        state.bekker_owner = None;
        state.bekker_resident_terrain = None;
        state.mpm_sim = None;
        state.mpm_terrain_mirror = None;
        state.mpm_terrain_readback = None;
        state.clear_mpm_observations();
        state.mpm_terrain_revision = 0;
        state.mpm_height_buf = None;
        state.mpm_sinkage_buf = None;
        state.mpm_surface_velocity_buf = None;
        state.dem_positions.clear();
        state.dem_velocities.clear();
        state.dem_owner = None;
        state.dem_resident_owner = None;
        state.dem_kernel = None;
        state.last_rut_depth_m = 0.0;
        state.last_rut_mean_m = 0.0;
        state.last_rut_cells = 0;
        state.last_mpm_compaction_m = 0.0;
        state.last_mpm_active_particles = 0;
        state.last_dem_sinkage_m = 0.0;
        state.last_dem_velocity_cells = 0;
        state.last_dem_active_grains = 0;
        state.last_dem_resident_reseeds = 0;
        state.last_dem_backend = "reset";
        state.last_dem_error = None;
        state.thermal_batch = None;
        state.transient_batch = None;
        state.transient_state = Default::default();
        state.last_transient_backend = "reset";
        state.last_transient_error = None;
        state.fem_feedback_bristles.clear();
        state.last_fem_feedback_backend = "reset";
        state.last_fem_feedback_error = None;
        state.last_fem_feedback_force_n = 0.0;
        state.last_fem_feedback_bristle_m = 0.0;
        state.last_fem_feedback_force_rows = 0;
        state.last_fem_feedback_bristle_rows = 0;
        state.last_fem_feedback_total_force_n = [0.0; 3];
        state.last_fem_feedback_mean_force_n = 0.0;
        state.last_fem_feedback_mean_bristle_m = 0.0;
        state.last_pressure_longitudinal_loads_n.clear();
        state.last_pressure_lateral_loads_n.clear();
        state.last_pressure_longitudinal_span_m = 0.0;
        state.last_pressure_lateral_span_m = 0.0;
        state.last_pressure_peak_pa = 0.0;
        state.last_pressure_total_load_n = 0.0;
        state.last_thermal_backend = "reset";
        state.last_thermal_error = None;
        state.last_replay_thermal_backend = "reset";
        state.last_replay_thermal_count = 0;
        state.last_replay_thermal_error = None;
        state.reset_requested = false;
    }
    if state.tire.is_none() {
        let params = tb_tire_params(state.pressure_pa, state.vertical_load_n);
        state.tire = Some(molla_vehicle::TetraTireFem::new(
            0.0,
            params.pressure_pa,
            params,
        ));
    }
    if state.enable_bekker_terrain && state.bekker_owner.is_none() {
        state.bekker_owner = Some(tb_new_bekker_owner());
    }
    if state.enable_dem_terrain && state.dem_owner.is_none() {
        let (positions, velocities) = tb_dem_particle_bed();
        match molla::kernels::TireDemFemTerrainOwner::from_grain_state(
            TB_DEM_ORIGIN,
            TB_DEM_SPACING,
            TB_DEM_NX,
            TB_DEM_NZ,
            0.0,
            state.material.material().mu_static as f32,
            TB_DEM_GRAIN_RADIUS,
            &positions,
            &velocities,
        ) {
            Ok(owner) => {
                state.last_dem_active_grains = positions.len();
                state.dem_positions = positions;
                state.dem_velocities = velocities;
                state.dem_owner = Some(owner);
                state.last_dem_backend = "DEM FEM terrain owner initialized";
                state.last_dem_error = None;
            }
            Err(err) => {
                state.last_dem_backend = "DEM owner init failed";
                state.last_dem_error = Some(format!("{err}"));
            }
        }
    }
    if run.paused {
        return;
    }
    // Fixed physics dt (2026-07-05 tire audit; matches scene 120).
    let dt = 1.0 / 120.0;
    let pressure = state.pressure_pa;
    let load = state.vertical_load_n;
    if let Some(t) = state.tire.as_mut() {
        if (t.pressure() as f32 - pressure).abs() > 1.0 {
            t.set_pressure(pressure as f64);
        }
        if (t.downforce() as f32 - load).abs() > 1.0 {
            t.set_downforce(load as f64);
        }
    }

    // Optional auto-sweep: ramps the slip angle through ±20° so the
    // user sees the classic cornering curve trace itself out.
    if state.auto_sweep_alpha {
        state.sweep_phase += dt;
        let omega = 0.6_f32;
        state.slip_angle_rad = (state.sweep_phase * omega).sin() * 20.0_f32.to_radians();
    }

    // ── Run the FEM tire under its own downforce ────────────────────
    // The hub falls under load and the carcass deforms visibly; no
    // kinematic pinning so what you see on screen is the actual
    // equilibrium shape (and it changes live when you move the load
    // or pressure sliders).
    let base_material_for_step = state.material.material();
    let mpm_step_terrain = state
        .mpm_terrain_mirror
        .as_ref()
        .filter(|_| state.enable_mpm_terrain)
        .map(|owner| {
            tb_vehicle_heightfield_from_mpm(
                owner,
                base_material_for_step.mu_static,
                state.water_film_mm as f64 / 1000.0,
            )
        });
    let bekker_step_terrain = state
        .bekker_owner
        .as_ref()
        .filter(|_| state.enable_bekker_terrain)
        .map(|owner| {
            tb_vehicle_heightfield_from_bekker(
                owner,
                base_material_for_step.mu_static,
                state.water_film_mm as f64 / 1000.0,
            )
        });
    let dem_step_terrain = state
        .dem_owner
        .as_ref()
        .filter(|_| state.enable_dem_terrain)
        .and_then(|owner| {
            tb_vehicle_heightfield_from_dem(
                owner,
                base_material_for_step.mu_static,
                state.water_film_mm as f64 / 1000.0,
            )
        });
    if let Some(t) = state.tire.as_mut() {
        if let Some(terrain) = mpm_step_terrain
            .as_ref()
            .or(bekker_step_terrain.as_ref())
            .or(dem_step_terrain.as_ref())
        {
            t.step_on_terrain(terrain, dt as f64);
        } else {
            t.step(dt as f64);
        }
    }
    let raw_alpha = state.slip_angle_rad;
    let raw_kappa = state.slip_ratio;
    let v = state.belt_speed_mps;
    state.last_raw_kappa = raw_kappa;
    state.last_raw_alpha_rad = raw_alpha;
    let mut transient_output = molla::kernels::TireTransientGpuOutput {
        effective: [raw_kappa, raw_alpha, 0.0, 1.0],
        deflection: [0.0; 4],
    };
    state.last_transient_error = None;
    if state.use_gpu_transient_fem_feedback {
        match (sim.engine.device(), sim.engine.queue()) {
            (Some(device), Some(queue)) => {
                if state.transient_kernel.is_none() {
                    state.transient_kernel = Some(molla::kernels::TireTransientGpuKernel::new(device));
                }
                if state.transient_batch.is_none()
                    && let Some(kernel) = state.transient_kernel.as_ref() {
                        match kernel.create_batch(device, 1) {
                            Ok(batch) => {
                                state.transient_batch = Some(batch);
                            }
                            Err(err) => {
                                state.last_transient_error = Some(format!("{err}"));
                            }
                        }
                    }
                if let (Some(kernel), Some(batch)) =
                    (state.transient_kernel.as_ref(), state.transient_batch.as_ref())
                {
                    let input = tb_transient_gpu_input(raw_kappa, raw_alpha, v, dt);
                    match kernel.dispatch_persistent_blocking(
                        device,
                        queue,
                        batch,
                        &[input],
                        &[state.transient_state],
                    ) {
                        Ok((states, outputs)) if states.len() == 1 && outputs.len() == 1 => {
                            state.transient_state = states[0];
                            transient_output = outputs[0];
                            state.last_transient_backend = "GPU transient → FEM feedback";
                        }
                        Ok((states, outputs)) => {
                            state.last_transient_error = Some(format!(
                                "GPU transient returned {} states / {} outputs, expected 1",
                                states.len(),
                                outputs.len()
                            ));
                            state.last_transient_backend = "raw slip fallback";
                        }
                        Err(err) => {
                            state.last_transient_error = Some(format!("{err}"));
                            state.last_transient_backend = "raw slip fallback";
                        }
                    }
                }
            }
            _ => {
                state.last_transient_error =
                    Some("GPU transient requested but no GPU device is available".into());
                state.last_transient_backend = "raw slip fallback";
            }
        }
    } else {
        state.transient_state = molla::kernels::TireTransientGpuState {
            tread: [raw_kappa, raw_alpha, 0.0, 0.0],
            belt: [raw_kappa, raw_alpha, 0.0, 0.0],
            carcass: [raw_kappa, raw_alpha, 0.0, 0.0],
        };
        state.last_transient_backend = "raw slip controls";
    }
    let transient_command = molla::kernels::tire_fem_transient_feedback_command(
        &transient_output,
        v,
        TB_TIRE_RADIUS as f32,
    );
    let alpha = transient_command.effective_alpha_rad;
    let kappa = transient_command.effective_kappa;
    state.last_effective_kappa = kappa;
    state.last_effective_alpha_rad = alpha;
    // The brush model gets the tire's current hub pose (so the rolling
    // road appears at the tire's actual contact height) and a
    // synthetic relative velocity that encodes the test bench's
    // (belt_speed, slip_angle) inputs.
    let hub_pos = state.tire.as_ref().map(|t| t.hub()).unwrap_or([0.0; 3]);
    let hub_vel = [
        -transient_command.surface_velocity_m_per_s[0] as f64,
        0.0,
        -transient_command.surface_velocity_m_per_s[2] as f64,
    ];
    let omega = transient_command.wheel_omega_rad_per_s;
    state.road_phase_x += transient_command.surface_velocity_m_per_s[0] * dt;
    state.road_phase_z += transient_command.surface_velocity_m_per_s[2] * dt;
    state.wheel_angle -= omega * dt;

    // ── Brush model with thermal/wear-modulated friction ────────────
    let mut material = state.material.material();
    let film = molla_vehicle::SurfaceFilmState::wet(state.water_film_mm as f64 / 1000.0);
    material = film.apply_to(material, v.abs() as f64, &state.wear);
    let condition = state.thermal.condition_report(&state.wear);
    let pressure_factor = condition.carcass_stiffness_factor;
    let mu_factor = condition.grip_factor.clamp(0.10, 1.5);
    material.mu_static *= mu_factor;
    material.mu_dynamic *= mu_factor;
    let params = molla_vehicle::RealTireParams::passenger_radial()
        .with_geometry_and_pressure(TB_TIRE_RADIUS, TB_TIRE_WIDTH, pressure as f64 * pressure_factor)
        .with_segments(96);
    let wheel_state = molla_vehicle::RealWheelState {
        hub_pos,
        hub_vel,
        heading: [1.0, 0.0, 0.0],
        omega: omega as f64,
    };
    let contact = if let Some(tire) = state.tire.as_ref() {
        if let Some(terrain) = state
            .mpm_terrain_mirror
            .as_ref()
            .filter(|_| state.enable_mpm_terrain)
            .map(|owner| {
                tb_vehicle_heightfield_from_mpm(
                    owner,
                    material.mu_static,
                    state.water_film_mm as f64 / 1000.0,
                )
            })
            .or_else(|| {
                state
                    .bekker_owner
                    .as_ref()
                    .filter(|_| state.enable_bekker_terrain)
                    .map(|owner| {
                        tb_vehicle_heightfield_from_bekker(
                            owner,
                            material.mu_static,
                            state.water_film_mm as f64 / 1000.0,
                        )
                    })
            })
            .or_else(|| {
                state
                    .dem_owner
                    .as_ref()
                    .filter(|_| state.enable_dem_terrain)
                    .and_then(|owner| {
                        tb_vehicle_heightfield_from_dem(
                            owner,
                            material.mu_static,
                            state.water_film_mm as f64 / 1000.0,
                        )
                    })
            })
        {
            let fem_contact = tire.evaluate_tread_contact(&wheel_state, &terrain, &material);
            if fem_contact.in_contact {
                fem_contact
            } else {
                molla_vehicle::evaluate_brush_tire(&params, &wheel_state, &terrain, &material)
            }
        } else {
            let terrain = molla_vehicle::FlatTerrain::new(0.0, material.mu_static);
            let fem_contact = tire.evaluate_tread_contact(&wheel_state, &terrain, &material);
            if fem_contact.in_contact {
                fem_contact
            } else {
                molla_vehicle::evaluate_brush_tire(&params, &wheel_state, &terrain, &material)
            }
        }
    } else {
        let terrain = molla_vehicle::FlatTerrain::new(0.0, material.mu_static);
        molla_vehicle::evaluate_brush_tire(&params, &wheel_state, &terrain, &material)
    };
    state.last_fx_n = contact.longitudinal_force as f32;
    state.last_fy_n = contact.lateral_force as f32;
    state.last_fn_n = contact.normal_load as f32;
    state.last_mz_nm = contact.aligning_moment_z as f32;
    state.last_dissipated_w = contact.dissipated_power as f32;
    state.last_mu_factor = mu_factor as f32;
    state.last_pressure_factor = pressure_factor as f32;
    state.last_contact_count = contact.contact_count as u32;
    let pressure_distribution = contact.pressure_distribution(8, 5);
    state.last_pressure_longitudinal_loads_n =
        tb_pressure_bin_loads(&pressure_distribution.longitudinal_bins);
    state.last_pressure_lateral_loads_n = tb_pressure_bin_loads(&pressure_distribution.lateral_bins);
    state.last_pressure_longitudinal_span_m = pressure_distribution.longitudinal_span_m as f32;
    state.last_pressure_lateral_span_m = pressure_distribution.lateral_span_m as f32;
    state.last_pressure_peak_pa = pressure_distribution.peak_normal_pressure_pa as f32;
    state.last_pressure_total_load_n = pressure_distribution.total_normal_load_n as f32;
    state.last_fem_feedback_error = None;
    state.last_fem_feedback_force_n = 0.0;
    state.last_fem_feedback_bristle_m = 0.0;
    state.last_fem_feedback_force_rows = 0;
    state.last_fem_feedback_bristle_rows = 0;
    state.last_fem_feedback_total_force_n = [0.0; 3];
    state.last_fem_feedback_mean_force_n = 0.0;
    state.last_fem_feedback_mean_bristle_m = 0.0;
    if state.use_gpu_transient_fem_feedback {
        let feedback_terrain_base = state
            .mpm_terrain_mirror
            .as_ref()
            .filter(|_| state.enable_mpm_terrain)
            .cloned()
            .or_else(|| {
                state
                    .bekker_owner
                    .as_ref()
                    .filter(|_| state.enable_bekker_terrain)
                    .map(|owner| owner.fem_contact_terrain())
            })
            .or_else(|| {
                state
                    .dem_owner
                    .as_ref()
                    .filter(|_| state.enable_dem_terrain)
                    .and_then(|owner| owner.fem_contact_terrain().ok())
            })
            .unwrap_or_else(|| {
                tb_default_feedback_terrain(
                    material.mu_static as f32,
                    transient_command.surface_velocity_m_per_s,
                )
            });
        let feedback_terrain = molla::kernels::tire_fem_contact_terrain_with_transient_surface_velocity(
            feedback_terrain_base,
            &transient_output,
            v,
            TB_TIRE_RADIUS as f32,
        );
        match (sim.engine.device(), sim.engine.queue()) {
            (Some(device), Some(queue)) => {
                if state.fem_feedback_kernel.is_none() {
                    state.fem_feedback_kernel =
                        Some(molla::kernels::TireFemContactFeedbackKernel::new(device));
                }
                if let Some(snapshot) = state.tire.as_ref().map(|tire| tire.gpu_snapshot()) {
                    if state.fem_feedback_bristles.len() != snapshot.tread_ids.len() {
                        state.fem_feedback_bristles = vec![[0.0; 4]; snapshot.tread_ids.len()];
                    }
                    let normal_load_per_tread =
                        (load / snapshot.tread_ids.len().max(1) as f32).max(0.0);
                    let area_per_tread = ((std::f32::consts::TAU * TB_TIRE_RADIUS as f32
                        * TB_TIRE_WIDTH as f32)
                        / snapshot.tread_ids.len().max(1) as f32)
                        .max(1.0e-5);
                    let feedback_params = molla::kernels::tire_fem_feedback_params_with_transient(
                        molla::kernels::TireFemContactFeedbackParams {
                            dt,
                            contact_band: 0.055,
                            ground_y: 0.0,
                            omega,
                            hub: [hub_pos[0] as f32, hub_pos[1] as f32, hub_pos[2] as f32],
                            hub_vel: [hub_vel[0] as f32, hub_vel[1] as f32, hub_vel[2] as f32],
                            heading: [1.0, 0.0, 0.0],
                            rolling_radius: TB_TIRE_RADIUS as f32,
                            normal_load_per_tread,
                            area_per_tread,
                            tread_shear_stiffness: params.tread_shear_stiffness as f32,
                            mu_static: material.mu_static as f32,
                            mu_dynamic: material.mu_dynamic as f32,
                            bristle_damping: material.bristle_damping as f32,
                            cohesion: material.cohesion as f32,
                            bristle_relax_rate: 25.0,
                        },
                        &transient_output,
                        v,
                    );
                    let use_resident_mpm_terrain = state.enable_mpm_terrain
                        && state.mpm_height_buf.is_some()
                        && state.mpm_sinkage_buf.is_some()
                        && state.mpm_surface_velocity_buf.is_some();
                    let use_resident_bekker_terrain = !use_resident_mpm_terrain
                        && state.enable_bekker_terrain
                        && state.bekker_resident_terrain.is_some();
                    let use_resident_dem_terrain = !use_resident_mpm_terrain
                        && !use_resident_bekker_terrain
                        && state.enable_dem_terrain
                        && state.dem_resident_owner.is_some();
                    let use_dem_terrain = !use_resident_mpm_terrain
                        && !use_resident_bekker_terrain
                        && state.enable_dem_terrain
                        && feedback_terrain.surface_velocity_map.is_some()
                        && feedback_terrain.sinkage_map.is_some();
                    let feedback_input = molla::kernels::TireFemContactFeedbackInput {
                        positions: snapshot.positions.clone(),
                        velocities: snapshot.velocities.clone(),
                        masses: snapshot.masses.clone(),
                        tread_ids: snapshot.tread_ids.clone(),
                        bristles: state.fem_feedback_bristles.clone(),
                        terrain: if use_resident_mpm_terrain
                            || use_resident_bekker_terrain
                            || use_resident_dem_terrain
                        {
                            None
                        } else {
                            Some(feedback_terrain.clone())
                        },
                        params: feedback_params,
                    };
                    let state = &mut *state;
                    if let Some(kernel) = state.fem_feedback_kernel.as_ref() {
                        let feedback_result =
                            if let (true, Some(heights), Some(sinkage), Some(surface_velocity)) = (
                                use_resident_mpm_terrain,
                                state.mpm_height_buf.as_ref(),
                                state.mpm_sinkage_buf.as_ref(),
                                state.mpm_surface_velocity_buf.as_ref(),
                            ) {
                                let gpu_terrain =
                                    molla::kernels::TireFemContactFeedbackGpuTerrain {
                                        origin: feedback_terrain.origin,
                                        spacing: feedback_terrain.spacing,
                                        nx: feedback_terrain.nx,
                                        nz: feedback_terrain.nz,
                                        heights,
                                        friction: feedback_terrain.friction,
                                        friction_map: None,
                                        water_film_map: None,
                                        sinkage_map: Some(sinkage),
                                        surface_velocity_map: Some(surface_velocity),
                                        soil_rolling_resistance_scale: feedback_terrain
                                            .soil_rolling_resistance_scale,
                                        surface_velocity: feedback_terrain.surface_velocity,
                                    };
                                tb_fem_terrain_feedback(
                                    &mut state.fem_feedback_prepared, state.retain_fem_feedback_buffers, &sim.engine, kernel,
                                    &feedback_input, &gpu_terrain,
                                )
                            } else if let (true, Some(bekker)) =
                                (use_resident_bekker_terrain, state.bekker_resident_terrain.as_ref())
                            {
                                let gpu_terrain =
                                    molla::kernels::TireFemContactFeedbackGpuTerrain {
                                        origin: bekker.origin,
                                        spacing: bekker.spacing,
                                        nx: bekker.nx,
                                        nz: bekker.nz,
                                        heights: &bekker.heights,
                                        friction: bekker.friction,
                                        friction_map: bekker.friction_map.as_ref(),
                                        water_film_map: bekker.water_film_map.as_ref(),
                                        sinkage_map: Some(&bekker.sinkage),
                                        surface_velocity_map: bekker.surface_velocity_map.as_ref(),
                                        soil_rolling_resistance_scale: bekker
                                            .soil_rolling_resistance_scale,
                                        surface_velocity: bekker.surface_velocity,
                                    };
                                tb_fem_terrain_feedback(
                                    &mut state.fem_feedback_prepared, state.retain_fem_feedback_buffers, &sim.engine, kernel,
                                    &feedback_input, &gpu_terrain,
                                )
                            } else if let (true, Some(dem)) =
                                (use_resident_dem_terrain, state.dem_resident_owner.as_ref())
                            {
                                dem.resident_terrain()
                                    .fem_contact_feedback_gpu_terrain(
                                    feedback_terrain.friction,
                                    feedback_terrain.soil_rolling_resistance_scale,
                                )
                                .and_then(|gpu_terrain| {
                                    tb_fem_terrain_feedback(
                                        &mut state.fem_feedback_prepared, state.retain_fem_feedback_buffers, &sim.engine, kernel,
                                        &feedback_input, &gpu_terrain,
                                    )
                                })
                            } else {
                                kernel.dispatch_blocking(device, queue, &feedback_input)
                            };
                        match feedback_result {
                            Ok(out) => {
                                let accounting = out.patch_accounting();
                                state.last_fem_feedback_force_n = accounting.max_force_n;
                                state.last_fem_feedback_bristle_m =
                                    accounting.max_bristle_deflection_m;
                                state.last_fem_feedback_force_rows = accounting.force_row_count;
                                state.last_fem_feedback_bristle_rows =
                                    accounting.bristle_row_count;
                                state.last_fem_feedback_total_force_n = accounting.total_force;
                                state.last_fem_feedback_mean_force_n = accounting.mean_force_n;
                                state.last_fem_feedback_mean_bristle_m =
                                    accounting.mean_bristle_deflection_m;
                                state.fem_feedback_bristles = out.bristles;
                                if let Some(tire) = state.tire.as_mut() {
                                    let hub_lateral = tire.hub_lateral_velocity();
                                    let _ = tire.apply_gpu_node_state(
                                        &snapshot.positions,
                                        &out.velocities,
                                        [
                                            hub_pos[0] as f32,
                                            hub_pos[1] as f32,
                                            hub_pos[2] as f32,
                                            0.0,
                                            hub_lateral[0] as f32,
                                            hub_lateral[1] as f32,
                                            0.0,
                                            0.0,
                                        ],
                                    );
                                }
                                state.last_fem_feedback_backend = if use_resident_mpm_terrain {
                                    if state.retain_fem_feedback_buffers { "GPU MPM terrain / retained CPU-tire feedback bridge" }
                                    else { "GPU MPM terrain / legacy CPU-tire feedback bridge" }
                                } else if use_resident_bekker_terrain {
                                    if state.retain_fem_feedback_buffers { "GPU Bekker terrain / retained CPU-tire feedback bridge" }
                                    else { "GPU Bekker terrain / legacy CPU-tire feedback bridge" }
                                } else if use_resident_dem_terrain {
                                    if state.retain_fem_feedback_buffers { "GPU DEM terrain / retained CPU-tire feedback bridge" }
                                    else { "GPU DEM terrain / legacy CPU-tire feedback bridge" }
                                } else if use_dem_terrain {
                                    "GPU FEM feedback + TireDemFemTerrainOwner terrain"
                                } else {
                                    "GPU transient-relaxed FEM contact feedback"
                                };
                            }
                            Err(err) => {
                                state.last_fem_feedback_error = Some(format!("{err}"));
                                state.last_fem_feedback_backend = "GPU FEM feedback failed";
                            }
                        }
                    }
                }
            }
            _ => {
                state.last_fem_feedback_error =
                    Some("GPU FEM contact feedback requested but no GPU device is available".into());
                state.last_fem_feedback_backend = "GPU FEM feedback unavailable";
            }
        }
    } else {
        state.last_fem_feedback_backend = "disabled";
    }

    if state.enable_mpm_terrain {
        state.last_mpm_error = None;
        match (sim.engine.device(), sim.engine.queue()) {
            (Some(device), Some(queue)) => {
                if state.mpm_kernel.is_none() {
                    state.mpm_kernel = Some(molla::kernels::TireMpmGpuKernel::new(device));
                }
                if state.mpm_sim.is_none() {
                    match tb_new_mpm_sim(&sim.engine, queue) {
                        Ok(mpm) => {
                            state.clear_mpm_observations();
                            state.mpm_terrain_revision = 0;
                            state.mpm_sim = Some(mpm);
                        }
                        Err(err) => {
                            state.last_mpm_error = Some(format!("{err}"));
                        }
                    }
                }
                let mpm_contacts = tb_mpm_contacts_from_tire_contact(&contact, hub_pos);
                let surface_velocity = [-hub_vel[0] as f32, 0.0, -hub_vel[2] as f32];
                let raster = tb_mpm_raster(material.mu_static as f32, surface_velocity);
                if state.mpm_sim.is_some() && state.mpm_kernel.is_some() {
                    let mut mpm = state.mpm_sim.take().expect("checked mpm sim");
                    let kernel = state.mpm_kernel.take().expect("checked mpm kernel");
                    let feedback = molla::compute::MpmTireFeedbackParams {
                        dt,
                        plastic_compaction_gain: 3.5e-6,
                        max_compaction: 0.20,
                        damping: 10.0,
                    };
                    let use_gpu_mpm = state.use_gpu_mpm;
                    let gpu_result = if use_gpu_mpm {
                        mpm.apply_tire_contacts_gpu_resident_with(
                            device,
                            queue,
                            &kernel,
                            &mpm_contacts,
                            feedback,
                        )
                    } else {
                        mpm.apply_tire_contacts_cpu_feedback_with(
                            device,
                            queue,
                            &mpm_contacts,
                            feedback,
                        )
                        .map(|_| ())
                    };
                    if let Err(err) = gpu_result {
                        state.last_mpm_error = Some(format!("{err}"));
                    } else {
                        let snapshot_result = (|| -> molla::core::Result<_> {
                            mpm.step_n_with(device, queue, dt.min(1.0 / 120.0), 1)?;
                            let cell_count = (raster.nx as usize * raster.nz as usize).max(1);
                            let f32_bytes = (cell_count * core::mem::size_of::<f32>()) as u64;
                            let usage = wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST;
                            let need_buffers = state.mpm_height_buf.as_ref().is_none_or(|b| b.size() != f32_bytes)
                                || state.mpm_sinkage_buf.as_ref().is_none_or(|b| b.size() != f32_bytes)
                                || state.mpm_surface_velocity_buf.as_ref().is_none_or(|b| b.size() != f32_bytes * 4);
                            if need_buffers {
                                state.mpm_terrain_readback = None;
                                state.mpm_height_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some("tire realistic mpm terrain heights"),
                                    size: f32_bytes.max(4),
                                    usage,
                                    mapped_at_creation: false,
                                }));
                                state.mpm_sinkage_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some("tire realistic mpm terrain sinkage"),
                                    size: f32_bytes.max(4),
                                    usage,
                                    mapped_at_creation: false,
                                }));
                                state.mpm_surface_velocity_buf = Some(device.create_buffer(
                                    &wgpu::BufferDescriptor {
                                        label: Some("tire realistic mpm terrain surface velocity"),
                                        size: (cell_count * core::mem::size_of::<[f32; 4]>()) as u64,
                                        usage,
                                        mapped_at_creation: false,
                                    },
                                ));
                            }
                            let heights = state.mpm_height_buf.as_ref().expect("MPM height buffer").clone();
                            let sinkage = state.mpm_sinkage_buf.as_ref().expect("MPM sinkage buffer").clone();
                            let surface_velocity = state.mpm_surface_velocity_buf.as_ref().expect("MPM velocity buffer").clone();
                            mpm.raster_tire_fem_terrain_gpu_resident_with(
                                device, queue, &kernel, raster, &heights, &sinkage, &surface_velocity,
                            )?;
                            let params = molla::kernels::TireMpmTerrainRasterParams {
                                particle_count: mpm.particle_count(), origin: raster.origin,
                                spacing: raster.spacing, nx: raster.nx, nz: raster.nz, rest_height: raster.rest_height,
                            };
                            if state.mpm_terrain_readback.as_ref().is_none_or(|r| r.params() != params) {
                                state.mpm_terrain_readback = Some(mpm.prepare_tire_terrain_readback_with(
                                    device, queue, raster, [&heights, &sinkage, &surface_velocity],
                                )?);
                            }
                            let snapshot = state.mpm_terrain_readback.as_mut().expect("MPM terrain readback")
                                .read_blocking(false)?;
                            let terrain = snapshot.terrain.into_fem_contact_terrain(
                                params, raster.friction, raster.soil_rolling_resistance_scale, raster.surface_velocity,
                            )?;
                            Ok(terrain)
                        })();
                        match snapshot_result {
                            Ok(terrain) => {
                                state.mpm_terrain_mirror = Some(terrain);
                                state.observe_mpm_compaction(&sim.engine, &mpm);
                                state.last_mpm_backend = if use_gpu_mpm {
                                    "GPU MPM + blocking terrain-only CPU bridge"
                                } else {
                                    "CPU MPM feedback + blocking terrain bridge"
                                };
                            }
                            Err(err) => {
                                state.last_mpm_error = Some(format!("{err}"));
                                state.last_mpm_backend = "MPM step/export failed";
                                state.clear_mpm_observations();
                                state.mpm_terrain_mirror = None;
                                state.mpm_terrain_readback = None;
                                state.mpm_height_buf = None;
                                state.mpm_sinkage_buf = None;
                                state.mpm_surface_velocity_buf = None;
                            }
                        }
                    }
                    state.mpm_sim = Some(mpm);
                    state.mpm_kernel = Some(kernel);
                }
            }
            _ => {
                state.last_mpm_error =
                    Some("MPM terrain bridge requested but no GPU device is available".into());
            }
        }
    }

    if state.enable_bekker_terrain {
        let contact_cells = tb_bekker_contacts_from_tire_contact(&contact, hub_pos);
        let water_film_m = state.water_film_mm.max(0.0) / 1000.0;
        state.last_bekker_error = None;
        if let Some(owner) = state.bekker_owner.as_mut() {
            let material_result = owner
                .set_uniform_friction_map(material.mu_static as f32)
                .and_then(|_| owner.set_uniform_water_film_map(water_film_m));
            owner.set_surface_velocity([-hub_vel[0] as f32, 0.0, -hub_vel[2] as f32]);
            if let Err(err) = material_result {
                state.last_bekker_error = Some(format!("{err}"));
            }
        }
        let mut used_gpu = false;
        if state.use_gpu_bekker {
            let gpu = match (sim.engine.device(), sim.engine.queue()) {
                (Some(device), Some(queue)) => {
                    if state.bekker_kernel.is_none() {
                        state.bekker_kernel = Some(molla::kernels::TireBekkerGpuKernel::new(device));
                    }
                    Some((device.clone(), queue.clone()))
                }
                _ => {
                    state.last_bekker_error =
                        Some("GPU Bekker requested but no GPU device is available".into());
                    None
                }
            };
            if let Some((device, queue)) = gpu
                && let Some(kernel) = state.bekker_kernel.take() {
                    if let Some(owner) = state.bekker_owner.as_mut() {
                        match owner.apply_contacts_gpu_resident(
                            &device,
                            &queue,
                            &kernel,
                            &contact_cells,
                        ) {
                            Ok(resident_terrain) => {
                                state.bekker_resident_terrain = Some(resident_terrain);
                                used_gpu = true;
                                state.last_bekker_backend = "GPU Bekker resident terrain owner";
                            }
                            Err(err) => {
                                state.last_bekker_error = Some(format!("{err}"));
                            }
                        }
                    }
                    state.bekker_kernel = Some(kernel);
                }
        }
        if !used_gpu
            && let Some(owner) = state.bekker_owner.as_mut() {
                if let Err(err) = owner.apply_contacts_cpu(&contact_cells) {
                    state.last_bekker_error = Some(format!("{err}"));
                } else {
                    state.bekker_resident_terrain = None;
                    state.last_bekker_backend = "CPU Bekker owner fallback";
                }
            }
        if let Some((rut_depth, rut_cells, rut_mean)) = state.bekker_owner.as_ref().and_then(|owner| {
            tb_soil_summary(state.soil_summary_diagnostics, owner.sinkage().iter().copied(), 1.0e-4)
        }) {
            state.last_rut_depth_m = rut_depth;
            state.last_rut_cells = rut_cells;
            state.last_rut_mean_m = rut_mean;
        }
    }

    if state.enable_dem_terrain {
        state.last_dem_error = None;
        if state.dem_owner.is_none() {
            let (positions, velocities) = tb_dem_particle_bed();
            match molla::kernels::TireDemFemTerrainOwner::from_grain_state(
                TB_DEM_ORIGIN,
                TB_DEM_SPACING,
                TB_DEM_NX,
                TB_DEM_NZ,
                0.0,
                material.mu_static as f32,
                TB_DEM_GRAIN_RADIUS,
                &positions,
                &velocities,
            ) {
                Ok(owner) => {
                    state.dem_positions = positions;
                    state.dem_velocities = velocities;
                    state.dem_owner = Some(owner);
                }
                Err(err) => {
                    state.last_dem_backend = "DEM owner init failed";
                    state.last_dem_error = Some(format!("{err}"));
                }
            }
        }
        let dem_contacts = tb_dem_contacts_from_tire_contact(&contact, &wheel_state);
        if let Some(mut owner) = state.dem_owner.take() {
            let params = molla::kernels::TireDemStepParams {
                grain_mass: TB_DEM_GRAIN_MASS,
                grain_radius: TB_DEM_GRAIN_RADIUS,
                normal_stiffness: 1.4e4,
                damping: 90.0,
                friction: material.mu_static as f32,
                floor_y: 0.0,
                gravity: [0.0, -9.81, 0.0],
                dt,
                tire_position_push_scale: 0.18,
            };
            let mut gpu_unavailable = None;
            let gpu_step_result = if state.use_gpu_dem {
                match (sim.engine.device(), sim.engine.queue()) {
                    (Some(device), Some(queue)) => {
                        let kernel = state
                            .dem_kernel
                            .take()
                            .unwrap_or_else(|| molla::kernels::TireDemGpuKernel::new(device));
                        let result = (|| -> Result<(usize, u32, f32, u64), String> {
                            if state.dem_resident_owner.as_ref().is_none_or(|resident| {
                                resident.resident_state().grain_count != owner.positions().len()
                            }) {
                                state.dem_resident_owner = Some(
                                    molla::kernels::TireDemResidentFemTerrainOwner::from_grain_state(
                                        device,
                                        queue,
                                        &kernel,
                                        molla::kernels::TireDemResidentTerrainParams {
                                            origin: TB_DEM_ORIGIN,
                                            spacing: TB_DEM_SPACING,
                                            nx: TB_DEM_NX,
                                            nz: TB_DEM_NZ,
                                            floor_y: 0.0,
                                            grain_radius: TB_DEM_GRAIN_RADIUS,
                                        },
                                        owner.positions(),
                                        owner.velocities(),
                                    )
                                    .map_err(|err| format!("{err}"))?,
                                );
                            }
                            let resident_owner = state
                                .dem_resident_owner
                                .as_mut()
                                .ok_or_else(|| "resident DEM FEM terrain owner missing".to_owned())?;
                            if resident_owner.positions() != owner.positions()
                                || resident_owner.velocities() != owner.velocities()
                            {
                                resident_owner
                                    .reseed_grain_state(
                                        device,
                                        queue,
                                        &kernel,
                                        owner.positions(),
                                        owner.velocities(),
                                    )
                                    .map_err(|err| format!("{err}"))?;
                            }
                            let report = resident_owner
                                .step_sparse_lbvh_gpu_mirrored(
                                    device,
                                    queue,
                                    &kernel,
                                    &dem_contacts,
                                    params,
                                    8,
                                )
                                .map_err(|err| format!("{err}"))?;
                            owner
                                .update_grain_state(
                                    resident_owner.positions(),
                                    resident_owner.velocities(),
                                )
                                .map_err(|err| format!("{err}"))?;
                            Ok((
                                report.grain_count,
                                report.active_velocity_cells,
                                report.max_sinkage_m,
                                resident_owner.reseed_count(),
                            ))
                        })();
                        state.dem_kernel = Some(kernel);
                        Some(result)
                    }
                    _ => {
                        gpu_unavailable =
                            Some("GPU DEM requested but no GPU device is available".to_owned());
                        None
                    }
                }
            } else {
                None
            };
            let used_gpu_dem = gpu_step_result.is_some();
            let step_result = gpu_step_result.unwrap_or_else(|| {
                state.dem_resident_owner = None;
                owner
                    .step_sparse_lbvh_reference(&dem_contacts, params)
                    .map(|report| {
                        (
                            report.grain_count,
                            report.active_velocity_cells,
                            report.max_sinkage_m,
                            0,
                        )
                    })
                    .map_err(|err| format!("{err}"))
            });
            match step_result {
                Ok((grain_count, active_velocity_cells, max_sinkage_m, resident_reseeds)) => {
                    let positions = owner.positions().to_vec();
                    let velocities = owner.velocities().to_vec();
                    state.last_dem_sinkage_m = max_sinkage_m;
                    state.last_dem_velocity_cells = active_velocity_cells;
                    state.last_dem_active_grains = grain_count;
                    state.last_dem_resident_reseeds = resident_reseeds;
                    state.last_dem_backend = if used_gpu_dem {
                        "GPU resident sparse-LBVH DEM → resident terrain/FEM handoff"
                    } else {
                        "CPU sparse-LBVH DEM → TireDemFemTerrainOwner"
                    };
                    if let Some(reason) = gpu_unavailable {
                        state.last_dem_error = Some(reason);
                    }
                    state.dem_positions = positions;
                    state.dem_velocities = velocities;
                }
                Err(err) => {
                    state.last_dem_backend = "owned DEM sparse-LBVH step failed";
                    state.last_dem_error = Some(err.to_string());
                }
            }
            state.dem_owner = Some(owner);
        }
    } else {
        state.last_dem_sinkage_m = 0.0;
        state.last_dem_velocity_cells = 0;
        state.last_dem_active_grains = 0;
        state.last_dem_resident_reseeds = 0;
        state.dem_resident_owner = None;
    }

    // Multi-layer thermal + temperature-aware wear integration.
    let speed = v.abs() as f64;
    let rolling_power = contact.rolling_resistance_force.abs() * speed;
    let carcass_flex_power = contact.normal_load.max(0.0)
        * speed
        * 0.0025
        * (1.0 + 0.50 * (kappa as f64).abs() + 0.30 * (alpha as f64).abs());
    let brake_rim_power = if kappa < 0.0 {
        contact.longitudinal_force.abs() * speed * (-(kappa as f64)).min(1.0) * 0.12
    } else {
        0.0
    };
    let road_temperature_c = state.material.road_temperature_c();
    state.last_thermal_error = None;
    let mut used_gpu_thermal = false;
    if state.use_gpu_thermal {
        match (sim.engine.device(), sim.engine.queue()) {
            (Some(device), Some(queue)) => {
                if state.thermal_kernel.is_none() {
                    state.thermal_kernel = Some(molla::kernels::TireThermalGpuKernel::new(device));
                }
                if state.thermal_batch.is_none()
                    && let Some(kernel) = state.thermal_kernel.as_ref() {
                        match kernel.create_batch(device, 64) {
                            Ok(batch) => {
                                state.thermal_batch = Some(batch);
                            }
                            Err(err) => {
                                state.last_thermal_error = Some(format!("{err}"));
                            }
                        }
                    }
                if let Some(kernel) = state.thermal_kernel.take() {
                    let input = molla::kernels::TireThermalGpuInput {
                        powers: [
                            contact.dissipated_power as f32,
                            carcass_flex_power as f32,
                            rolling_power as f32,
                            brake_rim_power as f32,
                        ],
                        environment: [road_temperature_c as f32, dt, 0.0, 0.0],
                    };
                    let gpu_state = tb_gpu_thermal_state_from_vehicle(&state.thermal, &state.wear);
                    let dispatch = if let Some(batch) = state.thermal_batch.as_ref() {
                        kernel.dispatch_persistent_blocking(
                            device,
                            queue,
                            batch,
                            &[input],
                            &[gpu_state],
                        )
                    } else {
                        kernel.dispatch_blocking(device, queue, &[input], &[gpu_state])
                    };
                    match dispatch {
                        Ok((states, _outputs)) => {
                            if let Some(next) = states.into_iter().next() {
                                tb_apply_gpu_thermal_state(&mut state, next);
                                state.last_thermal_backend = if state.thermal_batch.is_some() {
                                    "GPU persistent thermal/wear batch"
                                } else {
                                    "GPU thermal/wear batch"
                                };
                                used_gpu_thermal = true;
                            } else {
                                state.last_thermal_error =
                                    Some("GPU thermal kernel returned no state".into());
                            }
                        }
                        Err(err) => {
                            state.last_thermal_error = Some(format!("{err}"));
                        }
                    }
                    state.thermal_kernel = Some(kernel);
                }
            }
            _ => {
                state.last_thermal_error =
                    Some("GPU thermal requested but no GPU device is available".into());
            }
        }
    }
    if !used_gpu_thermal {
        state.thermal.step(
            contact.dissipated_power,
            carcass_flex_power,
            rolling_power,
            brake_rim_power,
            road_temperature_c,
            dt as f64,
        );
        let condition = state.thermal.condition_report(&state.wear);
        state
            .wear
            .step_with_condition_report(contact.dissipated_power, condition, dt as f64);
        state.last_thermal_backend = if state.use_gpu_thermal {
            "CPU thermal/wear fallback"
        } else {
            "CPU thermal/wear"
        };
    }

    state.last_replay_thermal_count = 0;
    state.last_replay_thermal_error = None;
    if state.use_gpu_thermal {
        let replay_thermal_work = state
            .replay_panel
            .run
            .as_ref()
            .map(|run| {
                molla::kernels::tire_thermal_build_rig_replay_batch(&run.snapshots, dt, 20.0)
            })
            .filter(|replay| !replay.is_empty());
        if let Some(replay) = replay_thermal_work {
            match (sim.engine.device(), sim.engine.queue()) {
                (Some(device), Some(queue)) => {
                    if state.thermal_kernel.is_none() {
                        state.thermal_kernel =
                            Some(molla::kernels::TireThermalGpuKernel::new(device));
                    }
                    let needed = replay.capacity_hint();
                    if state
                        .thermal_batch
                        .as_ref()
                        .is_none_or(|batch| batch.capacity() < needed)
                        && let Some(kernel) = state.thermal_kernel.as_ref() {
                            match kernel.create_batch(device, needed) {
                                Ok(batch) => {
                                        state.thermal_batch = Some(batch);
                                    }
                                    Err(err) => {
                                        state.last_replay_thermal_error = Some(format!("{err}"));
                                    }
                                }
                            }
                    if let (Some(kernel), Some(batch)) =
                        (state.thermal_kernel.as_ref(), state.thermal_batch.as_ref())
                    {
                        match kernel.dispatch_persistent_blocking(
                            device,
                            queue,
                            batch,
                            &replay.inputs,
                            &replay.states,
                        ) {
                                Ok((next_states, outputs)) => {
                                    state.last_replay_thermal_count =
                                        next_states.len().min(outputs.len());
                                    state.last_replay_thermal_backend =
                                        "GPU replay persistent thermal/wear batch";
                                }
                                Err(err) => {
                                    state.last_replay_thermal_error = Some(format!("{err}"));
                                    state.last_replay_thermal_backend =
                                        "GPU replay thermal/wear failed";
                                }
                            }
                        }
                    }
                    _ => {
                        state.last_replay_thermal_error =
                            Some("GPU replay thermal requested but no GPU device is available".into());
                        state.last_replay_thermal_backend = "GPU replay unavailable";
                    }
                }
        }
    } else {
        state.last_replay_thermal_backend = "CPU replay snapshots";
    }

    // ── Pre-computed Fy(α) and Fx(κ) sweep curves (visualised) ──────
    //
    // Recompute the canonical curves fresh each frame against the
    // current (pressure, load, material, μ-factor) state so the chart
    // tracks the running tire condition. The brush model is cheap
    // enough that 91 + 91 evaluations per frame are well in the
    // budget for a viewer demo.
    let curve_terrain = molla_vehicle::FlatTerrain::new(0.0, material.mu_static);
    for i in 0..TB_CURVE_SAMPLES {
        let frac = i as f32 / (TB_CURVE_SAMPLES - 1) as f32;
        // α sweep over ±25°
        let alpha = (-25.0 + 50.0 * frac).to_radians();
        let hub_vel = [
            (v * alpha.cos()) as f64,
            0.0,
            (v * alpha.sin()) as f64,
        ];
        let omega = v / TB_TIRE_RADIUS as f32;
        let c = molla_vehicle::evaluate_brush_tire(
            &params,
            &molla_vehicle::RealWheelState {
                hub_pos,
                hub_vel,
                heading: [1.0, 0.0, 0.0],
                omega: omega as f64,
            },
            &curve_terrain,
            &material,
        );
        state.fy_alpha_curve[i] = c.lateral_force as f32;

        // κ sweep over ±0.5
        let kappa = -0.5 + frac;
        let omega_k = (1.0 + kappa) * v / TB_TIRE_RADIUS as f32;
        let c = molla_vehicle::evaluate_brush_tire(
            &params,
            &molla_vehicle::RealWheelState {
                hub_pos,
                hub_vel: [v as f64, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: omega_k as f64,
            },
            &curve_terrain,
            &material,
        );
        state.fx_kappa_curve[i] = c.longitudinal_force as f32;
    }
}

fn tb_temperature_color(temp_c: f32) -> Color {
    if temp_c < 80.0 {
        let t = (temp_c / 80.0).clamp(0.0, 1.0);
        Color::srgb(0.20 + 0.20 * t, 0.45 + 0.45 * t, 0.95 - 0.55 * t)
    } else {
        let t = ((temp_c - 80.0) / 80.0).clamp(0.0, 1.0);
        Color::srgb(0.40 + 0.60 * t, 0.90 - 0.70 * t, 0.40 - 0.30 * t)
    }
}

fn draw_tire_realistic_gizmos(
    sim: Res<Sim>,
    state: Res<TireRealisticState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::TireRealistic {
        return;
    }
    let tire = match state.tire.as_ref() {
        Some(t) => t,
        None => return,
    };

    // ── Rolling road ────────────────────────────────────────────────
    // A 6 m × 1.5 m strip centred at the origin, with painted stripes
    // that advance with `road_phase_*`. Stripes are normal to belt
    // motion, so when slip angle is non-zero you see them diagonally.
    let road_color = state.material.color();
    let stripe_color = Color::srgb(1.0, 0.97, 0.85);
    let half_l = 3.5_f32;
    let half_w = 1.0_f32;
    // Outline.
    let road_corners = [
        Vec3::new(-half_l, 0.001, -half_w),
        Vec3::new(half_l, 0.001, -half_w),
        Vec3::new(half_l, 0.001, half_w),
        Vec3::new(-half_l, 0.001, half_w),
    ];
    for i in 0..4 {
        gizmos.line(road_corners[i], road_corners[(i + 1) % 4], road_color);
    }
    // Subtle fill via cross lines (gives the road material colour
    // some surface area in gizmo space).
    for k in 1..=8 {
        let t = k as f32 / 9.0;
        let z = -half_w + 2.0 * half_w * t;
        gizmos.line(
            Vec3::new(-half_l, 0.001, z),
            Vec3::new(half_l, 0.001, z),
            road_color,
        );
    }
    // Moving stripes: spaced every 1 m, animated by the phase. Stripe
    // direction is perpendicular to the belt's motion direction so the
    // user can see WHERE the road is going.
    let dir_x = state.slip_angle_rad.cos();
    let dir_z = state.slip_angle_rad.sin();
    let perp_x = -dir_z;
    let perp_z = dir_x;
    let stripe_spacing = 1.0_f32;
    let phase_along = state.road_phase_x * dir_x + state.road_phase_z * dir_z;
    let phase_mod = phase_along.rem_euclid(stripe_spacing);
    for i in -8..=8 {
        let s = i as f32 * stripe_spacing - phase_mod;
        let cx = dir_x * s;
        let cz = dir_z * s;
        // Clip stripes that fall outside the road outline.
        if cx.abs() > half_l + 1.0 || cz.abs() > half_w + 1.0 {
            continue;
        }
        let len = (half_l.min(half_w)) * 0.7;
        gizmos.line(
            Vec3::new(cx - perp_x * len, 0.0015, cz - perp_z * len),
            Vec3::new(cx + perp_x * len, 0.0015, cz + perp_z * len),
            stripe_color,
        );
    }

    // ── Live Bekker rut map ────────────────────────────────────────
    // The height/sinkage map is owned by `TireBekkerFemTerrainOwner`.
    // It is updated from the current FEM contact-cell pressure/shear and
    // exported as `TireFemContactTerrain` for the GPU FEM contact path.
    if let Some(owner) = state.bekker_owner.as_ref() {
        let input = owner.bekker_input();
        let heights = owner.heights();
        let sinkage = owner.sinkage();
        let rut_color = Color::srgb(0.36, 0.25, 0.16);
        let deep_rut = Color::srgb(0.20, 0.12, 0.08);
        let stride_z = 4_usize;
        let stride_x = 4_usize;
        for iz in (0..input.nz as usize).step_by(stride_z) {
            for ix in 0..input.nx as usize - 1 {
                let i0 = iz * input.nx as usize + ix;
                let i1 = i0 + 1;
                let x0 = input.origin[0] + ix as f32 * input.spacing;
                let x1 = input.origin[0] + (ix + 1) as f32 * input.spacing;
                let z = input.origin[1] + iz as f32 * input.spacing;
                let c = if sinkage[i0].max(sinkage[i1]) > 0.015 {
                    deep_rut
                } else {
                    rut_color
                };
                gizmos.line(
                    Vec3::new(x0, heights[i0] + 0.002, z),
                    Vec3::new(x1, heights[i1] + 0.002, z),
                    c,
                );
            }
        }
        for ix in (0..input.nx as usize).step_by(stride_x) {
            for iz in 0..input.nz as usize - 1 {
                let i0 = iz * input.nx as usize + ix;
                let i1 = i0 + input.nx as usize;
                let x = input.origin[0] + ix as f32 * input.spacing;
                let z0 = input.origin[1] + iz as f32 * input.spacing;
                let z1 = input.origin[1] + (iz + 1) as f32 * input.spacing;
                let c = if sinkage[i0].max(sinkage[i1]) > 0.015 {
                    deep_rut
                } else {
                    rut_color
                };
                gizmos.line(
                    Vec3::new(x, heights[i0] + 0.002, z0),
                    Vec3::new(x, heights[i1] + 0.002, z1),
                    c,
                );
            }
        }
    }

    // ── Optional live DEM grain bridge ─────────────────────────────
    // Original per-grain render restored — discussion in progress about
    // what the DEM bridge visualization should look like.
    if state.enable_dem_terrain {
        let grain_color = Color::srgb(0.62, 0.50, 0.34);
        let moving_color = Color::srgb(0.95, 0.70, 0.28);
        for (idx, (p, v)) in state
            .dem_positions
            .iter()
            .zip(state.dem_velocities.iter())
            .enumerate()
        {
            if idx % 2 != 0 {
                continue;
            }
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            gizmos.sphere(
                Vec3::new(p[0], p[1], p[2]),
                TB_DEM_GRAIN_RADIUS * 0.45,
                if speed > 0.05 { moving_color } else { grain_color },
            );
        }
    }

    // ── The tire (FEM mesh) ─────────────────────────────────────────
    let temp_color = tb_temperature_color(state.thermal.surface_c as f32);
    let wear_ratio = (state.wear.depth_m / state.wear.max_depth_m).clamp(0.0, 1.0) as f32;
    let condition = state.thermal.condition_report(&state.wear);
    let abuse_dim = (1.0 - 0.35 * condition.abuse_index as f32).clamp(0.55, 1.0);
    let wear_dim = 1.0 - 0.55 * wear_ratio;
    let body = Color::srgb(
        temp_color.to_srgba().red * wear_dim * abuse_dim,
        temp_color.to_srgba().green * wear_dim * abuse_dim,
        temp_color.to_srgba().blue * wear_dim * abuse_dim,
    );
    let contact_color = Color::srgb(1.0, 0.92, 0.30);
    // Primary visible tire: same pressure/spin silhouette as 120/121, so
    // pressure and rolling are obvious instead of hidden in a subtle FEM mesh.
    let hub = tire.hub();
    let visual_hub_y =
        tire_visual_hub_height_for_load(state.pressure_pa as f64, state.vertical_load_n as f64)
            as f32;
    draw_traction_wheel(
        &mut gizmos,
        Vec3::new(hub[0] as f32, visual_hub_y, hub[2] as f32),
        state.pressure_pa,
        state.wheel_angle,
        body,
        contact_color,
    );

    // ── Heading + velocity arrows above the tire ────────────────────
    let above = Vec3::new(hub[0] as f32, hub[1] as f32 + 0.55, hub[2] as f32);
    let heading_arrow = Vec3::new(1.0, 0.0, 0.0) * 0.80;
    let vel_dir = Vec3::new(dir_x, 0.0, dir_z) * 0.80;
    gizmos.line(above, above + heading_arrow, Color::srgb(0.95, 0.95, 0.97));
    gizmos.line(above, above + vel_dir, Color::srgb(0.30, 0.85, 0.95));
    // Slip-angle wedge between them (orange).
    let steps = 12;
    let r_wedge = 0.40_f32;
    for k in 0..steps {
        let t0 = k as f32 / steps as f32;
        let t1 = (k + 1) as f32 / steps as f32;
        let a0 = state.slip_angle_rad * t0;
        let a1 = state.slip_angle_rad * t1;
        let p0 = above + Vec3::new(r_wedge * a0.cos(), 0.0, r_wedge * a0.sin());
        let p1 = above + Vec3::new(r_wedge * a1.cos(), 0.0, r_wedge * a1.sin());
        gizmos.line(p0, p1, Color::srgb(0.98, 0.62, 0.20));
    }

    // ── FEM contact patch outline on the floor ──────────────────────
    //
    // Tread nodes that are within ~5 mm of the floor are highlighted.
    // The footprint visibly grows when the user lowers pressure or
    // raises the load — this is THE thing the FEM physics buys us.
    let tread_ids = tire.tread_nodes();
    let nodes = tire.nodes();
    let mut patch_points: Vec<Vec3> = tread_ids
        .iter()
        .map(|&id| {
            Vec3::new(
                nodes[id][0] as f32,
                nodes[id][1] as f32,
                nodes[id][2] as f32,
            )
        })
        .filter(|p| p.y < 5.0e-3)
        .collect();
    // Order them by angle around the hub centre so the polyline
    // outline reads cleanly.
    let hub_xz = Vec2::new(hub[0] as f32, hub[2] as f32);
    patch_points.sort_by(|a, b| {
        let aa = (a.z - hub_xz.y).atan2(a.x - hub_xz.x);
        let ab = (b.z - hub_xz.y).atan2(b.x - hub_xz.x);
        aa.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
    });
    let patch_color = Color::srgb(0.98, 0.82, 0.15);
    let n_patch = patch_points.len();
    if n_patch >= 2 {
        for i in 0..n_patch {
            let a = patch_points[i];
            let b = patch_points[(i + 1) % n_patch];
            gizmos.line(
                Vec3::new(a.x, 0.003, a.z),
                Vec3::new(b.x, 0.003, b.z),
                patch_color,
            );
        }
    }
    for p in &patch_points {
        gizmos.sphere(
            Vec3::new(p.x, 0.005, p.z),
            0.018,
            Color::srgb(1.0, 0.95, 0.30),
        );
    }

    // ── Force arrows at the contact patch ───────────────────────────
    let contact_anchor = Vec3::new(0.0, 0.02, 0.0);
    let fx_arrow = Vec3::new(1.0, 0.0, 0.0)
        * (state.last_fx_n * 0.00040).clamp(-2.2, 2.2);
    let fy_arrow = Vec3::new(0.0, 0.0, 1.0)
        * (state.last_fy_n * 0.00040).clamp(-2.2, 2.2);
    let fn_arrow = Vec3::new(0.0, 1.0, 0.0)
        * (state.last_fn_n * 0.00040).clamp(0.0, 2.0);
    let arrow_x = Color::srgb(0.98, 0.34, 0.30);
    let arrow_y = Color::srgb(0.30, 0.85, 0.95);
    let arrow_n = Color::srgb(0.95, 0.95, 0.97);
    gizmos.line(contact_anchor, contact_anchor + fx_arrow, arrow_x);
    gizmos.line(contact_anchor, contact_anchor + fy_arrow, arrow_y);
    gizmos.line(contact_anchor, contact_anchor + fn_arrow, arrow_n);
    let tip_tick = |g: &mut Gizmos, base: Vec3, tip: Vec3, perp: Vec3, c: Color| {
        let dir = (tip - base).normalize_or_zero();
        let p = tip - dir * 0.10;
        g.line(p + perp * 0.06, tip, c);
        g.line(p - perp * 0.06, tip, c);
    };
    tip_tick(&mut gizmos, contact_anchor, contact_anchor + fx_arrow, Vec3::Z, arrow_x);
    tip_tick(&mut gizmos, contact_anchor, contact_anchor + fy_arrow, Vec3::X, arrow_y);
    tip_tick(&mut gizmos, contact_anchor, contact_anchor + fn_arrow, Vec3::X, arrow_n);

    // ── Cornering and traction curves ───────────────────────────────
    // Two 3-m × 1-m charts behind the tire, slightly above the ground.
    let fy_chart = Vec3::new(-2.5, 0.0, -2.2);
    let fx_chart = Vec3::new(0.5, 0.0, -2.2);
    let chart_w = 1.8_f32;
    let chart_h = 0.85_f32;
    let axis = Color::srgb(0.55, 0.55, 0.60);
    for origin in [fy_chart, fx_chart] {
        gizmos.line(origin, origin + Vec3::new(chart_w, 0.0, 0.0), axis);
        gizmos.line(
            origin + Vec3::new(0.0, 0.0, 0.0),
            origin + Vec3::new(0.0, chart_h, 0.0),
            axis,
        );
        gizmos.line(
            origin + Vec3::new(0.0, chart_h * 0.5, 0.0),
            origin + Vec3::new(chart_w, chart_h * 0.5, 0.0),
            axis,
        );
    }
    let scale_fy = chart_h * 0.5 / state.vertical_load_n.max(1.0);
    let scale_fx = scale_fy;
    let n = TB_CURVE_SAMPLES;
    let plot_curve = |g: &mut Gizmos, origin: Vec3, samples: &[f32], scale: f32, color: Color| {
        if samples.len() < 2 {
            return;
        }
        let mut prev = origin
            + Vec3::new(
                0.0,
                chart_h * 0.5 + samples[0] * scale,
                0.0,
            );
        for i in 1..samples.len() {
            let x = i as f32 / (samples.len() - 1) as f32 * chart_w;
            let y = chart_h * 0.5 + samples[i] * scale;
            let here = origin + Vec3::new(x, y.clamp(0.0, chart_h), 0.0);
            g.line(prev, here, color);
            prev = here;
        }
    };
    plot_curve(
        &mut gizmos,
        fy_chart,
        &state.fy_alpha_curve,
        scale_fy,
        Color::srgb(0.30, 0.85, 0.95),
    );
    plot_curve(
        &mut gizmos,
        fx_chart,
        &state.fx_kappa_curve,
        scale_fx,
        Color::srgb(0.98, 0.34, 0.30),
    );
    // Current operating points on each curve.
    let fy_frac = ((state.slip_angle_rad.to_degrees() + 25.0) / 50.0).clamp(0.0, 1.0);
    let fx_frac = ((state.slip_ratio + 0.5) / 1.0).clamp(0.0, 1.0);
    let fy_idx = (fy_frac * (n - 1) as f32) as usize;
    let fx_idx = (fx_frac * (n - 1) as f32) as usize;
    let dot_fy = fy_chart
        + Vec3::new(
            fy_frac * chart_w,
            (chart_h * 0.5 + state.fy_alpha_curve[fy_idx] * scale_fy).clamp(0.0, chart_h),
            0.0,
        );
    let dot_fx = fx_chart
        + Vec3::new(
            fx_frac * chart_w,
            (chart_h * 0.5 + state.fx_kappa_curve[fx_idx] * scale_fx).clamp(0.0, chart_h),
            0.0,
        );
    gizmos.sphere(dot_fy, 0.04, Color::srgb(1.0, 1.0, 0.4));
    gizmos.sphere(dot_fx, 0.04, Color::srgb(1.0, 1.0, 0.4));
    // Axis labels via tick marks at zero / mid.
    let _ = (above, n);

    // ── Compact T / wear / μ stack ──────────────────────────────────
    // Three vertical bars to the left of the tire so they're always
    // legible alongside the gizmo arrows.
    let bar_anchor = Vec3::new(-1.0, 0.0, 1.4);
    let bar = |g: &mut Gizmos, x: f32, ratio: f32, color: Color| {
        let base = bar_anchor + Vec3::new(x, 0.0, 0.0);
        g.line(base, base + Vec3::new(0.0, 0.85 * ratio, 0.0), color);
        g.line(
            base - Vec3::new(0.05, 0.0, 0.0),
            base + Vec3::new(0.05, 0.0, 0.0),
            Color::srgb(0.7, 0.7, 0.75),
        );
    };
    let temp_ratio = ((state.thermal.rubber_peak_c() - 10.0) / 150.0).clamp(0.0, 1.0) as f32;
    let wear_ratio_bar = wear_ratio;
    let mu_ratio = state.last_mu_factor.clamp(0.0, 1.2) / 1.2;
    bar(&mut gizmos, 0.00, temp_ratio, Color::srgb(0.98, 0.32, 0.25));
    bar(&mut gizmos, 0.12, wear_ratio_bar, Color::srgb(0.98, 0.62, 0.18));
    bar(&mut gizmos, 0.24, mu_ratio, Color::srgb(0.40, 0.85, 0.50));

    // ── Layered thermal stack ───────────────────────────────────────
    // Four separate bars expose the thermal model's actual state instead
    // of a single "tire temp" number: surface reacts fastest, tread follows,
    // carcass is slower, and rim is the sink/source for brake-side heat.
    let layer_anchor = Vec3::new(-1.0, 0.0, 1.72);
    let layer_bar = |g: &mut Gizmos, x: f32, temp_c: f64, color: Color| {
        let base = layer_anchor + Vec3::new(x, 0.0, 0.0);
        let ratio = ((temp_c as f32 + 20.0) / 180.0).clamp(0.0, 1.2);
        g.line(base, base + Vec3::Y * (0.70 * ratio), color);
        g.sphere(base + Vec3::Y * (0.70 * ratio), 0.018, color);
        g.line(
            base - Vec3::X * 0.035,
            base + Vec3::X * 0.035,
            Color::srgb(0.62, 0.62, 0.68),
        );
    };
    layer_bar(
        &mut gizmos,
        0.00,
        state.thermal.surface_c,
        Color::srgb(1.0, 0.20, 0.12),
    );
    layer_bar(
        &mut gizmos,
        0.10,
        state.thermal.tread_c,
        Color::srgb(1.0, 0.75, 0.20),
    );
    layer_bar(
        &mut gizmos,
        0.20,
        state.thermal.carcass_c,
        Color::srgb(0.65, 0.45, 1.0),
    );
    layer_bar(
        &mut gizmos,
        0.30,
        state.thermal.rim_c,
        Color::srgb(0.28, 0.65, 1.0),
    );
}

fn tire_realistic_pane(
    ui: &mut egui::Ui,
    state: &mut TireRealisticState,
    accent: egui::Color32,
) {
    ui.label("Tire test bench (FEM + brush + layered heat + wear)");
    ui.separator();
    ui.label("Belt speed (road moves under the tire)");
    let _ = ui.add(egui::Slider::new(&mut state.belt_speed_mps, 0.5..=22.0).suffix(" m/s"));
    ui.label("Slip ratio κ (drive torque proxy)");
    let _ = ui.add(egui::Slider::new(&mut state.slip_ratio, -0.5..=0.5));
    ui.label("Slip angle α");
    let mut alpha_deg = state.slip_angle_rad.to_degrees();
    if ui
        .add(egui::Slider::new(&mut alpha_deg, -25.0..=25.0).suffix("°"))
        .changed()
    {
        state.auto_sweep_alpha = false;
        state.slip_angle_rad = alpha_deg.to_radians();
    }
    ui.checkbox(&mut state.auto_sweep_alpha, "Auto-sweep α (±20°)");
    ui.separator();
    ui.label("Inflation pressure");
    let _ = ui.add(
        egui::Slider::new(&mut state.pressure_pa, 40_000.0..=400_000.0)
            .suffix(" Pa")
            .logarithmic(true),
    );
    ui.label("Vertical wheel load");
    let _ = ui.add(
        egui::Slider::new(&mut state.vertical_load_n, 800.0..=8_000.0).suffix(" N"),
    );
    ui.separator();
    ui.label("Road material");
    ui.horizontal(|ui| {
        for opt in [
            BenchMaterial::Asphalt,
            BenchMaterial::Dirt,
            BenchMaterial::Sand,
            BenchMaterial::Ice,
        ] {
            if ui
                .selectable_label(state.material == opt, opt.label())
                .clicked()
            {
                state.material = opt;
            }
        }
    });
    ui.label("Water film");
    let _ = ui.add(egui::Slider::new(&mut state.water_film_mm, 0.0..=6.0).suffix(" mm"));
    ui.separator();
    if ui
        .checkbox(
            &mut state.compare_soil_backends,
            "Compare multiple soil backends",
        )
        .changed()
        && !state.compare_soil_backends
    {
        let enabled = tb_exclusive_soil_backends([
            state.enable_mpm_terrain,
            state.enable_bekker_terrain,
            state.enable_dem_terrain,
        ]);
        state.enable_mpm_terrain = enabled[0];
        state.enable_bekker_terrain = enabled[1];
        state.enable_dem_terrain = enabled[2];
    }
    if state.compare_soil_backends {
        ui.label("Comparison steps every selected backend; FEM uses MPM, then Bekker, then DEM.");
        ui.checkbox(&mut state.enable_mpm_terrain, "Live MPM offroad bridge");
        ui.checkbox(&mut state.enable_bekker_terrain, "Live Bekker rut terrain");
        ui.checkbox(&mut state.enable_dem_terrain, "Live DEM→FEM terrain bridge");
    } else {
        ui.horizontal(|ui| {
            for (label, selected) in [
                ("MPM", [true, false, false]),
                ("Bekker", [false, true, false]),
                ("DEM", [false, false, true]),
                ("No soil", [false, false, false]),
            ] {
                let current = [
                    state.enable_mpm_terrain,
                    state.enable_bekker_terrain,
                    state.enable_dem_terrain,
                ];
                if ui.selectable_label(current == selected, label).clicked() {
                    state.enable_mpm_terrain = selected[0];
                    state.enable_bekker_terrain = selected[1];
                    state.enable_dem_terrain = selected[2];
                }
            }
        });
    }
    ui.checkbox(&mut state.use_gpu_bekker, "Use GPU Bekker writeback");
    ui.checkbox(&mut state.use_gpu_mpm, "Use GPU MPM resident feedback");
    ui.checkbox(&mut state.use_gpu_dem, "Use GPU DEM sparse-LBVH step");
    ui.checkbox(
        &mut state.use_gpu_transient_fem_feedback,
        "Use GPU transient → FEM contact feedback",
    );
    ui.checkbox(
        &mut state.retain_fem_feedback_buffers,
        "Retain terrain feedback buffers (experimental timing)",
    );
    ui.checkbox(&mut state.use_gpu_thermal, "Use GPU thermal/wear batch");
    ui.separator();
    if button(ui, "Reset wear + heat", accent).clicked() {
        state.reset_requested = true;
    }
    ui.separator();
    ui.label(format!(
        "Slip ratio {:+.3}  slip angle {:+.2}°",
        state.slip_ratio,
        state.slip_angle_rad.to_degrees()
    ));
    ui.label(format!(
        "Transient effective κ {:+.3} / α {:+.2}°   raw κ {:+.3} / α {:+.2}°",
        state.last_effective_kappa,
        state.last_effective_alpha_rad.to_degrees(),
        state.last_raw_kappa,
        state.last_raw_alpha_rad.to_degrees()
    ));
    ui.label(format!(
        "Fx {:+.0} N   Fy {:+.0} N   Fn {:.0} N",
        state.last_fx_n, state.last_fy_n, state.last_fn_n
    ));
    ui.label(format!("Aligning moment Mz {:+.1} N·m", state.last_mz_nm));
    ui.label(format!(
        "Dissipated power {:.1} kW",
        state.last_dissipated_w / 1000.0
    ));
    ui.label(format!("Contact cells: {}", state.last_contact_count));
    ui.label("Contact source: deformed FEM tread nodes");
    ui.label(format!(
        "Transient/FEM feedback: {}",
        state.last_fem_feedback_backend
    ));
    ui.label(format!(
        "FEM feedback max force {:.0} N  max bristle {:.2} mm",
        state.last_fem_feedback_force_n,
        state.last_fem_feedback_bristle_m * 1000.0
    ));
    ui.label(format!(
        "FEM patch accounting: force rows {}  bristle rows {}",
        state.last_fem_feedback_force_rows, state.last_fem_feedback_bristle_rows
    ));
    ui.label(format!(
        "FEM patch total force [{:+.0}, {:+.0}, {:+.0}] N",
        state.last_fem_feedback_total_force_n[0],
        state.last_fem_feedback_total_force_n[1],
        state.last_fem_feedback_total_force_n[2]
    ));
    ui.label(format!(
        "FEM patch means: force {:.0} N  bristle {:.2} mm",
        state.last_fem_feedback_mean_force_n,
        state.last_fem_feedback_mean_bristle_m * 1000.0
    ));
    ui.small(
        "Patch accounting comes from TireFemContactFeedbackOutput::patch_accounting(), \
         so seam/curb regressions can be spotted when force or bristle support collapses \
         even if a single max-force number still looks plausible.",
    );
    ui.label(format!("Transient backend: {}", state.last_transient_backend));
    if let Some(err) = &state.last_transient_error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 80), err);
    }
    if let Some(err) = &state.last_fem_feedback_error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 80), err);
    }
    let deflection_mm = state
        .tire
        .as_ref()
        .map(|t| t.hub_deflection_m() * 1000.0)
        .unwrap_or(0.0);
    ui.label(format!(
        "FEM hub deflection: {:.1} mm  (load↑ → more)",
        deflection_mm
    ));
    // Patch outline derived from FEM tread nodes touching the floor.
    let patch_nodes = state
        .tire
        .as_ref()
        .map(|t| {
            let nodes = t.nodes();
            t.tread_nodes()
                .iter()
                .filter(|&&id| nodes[id][1] < 5.0e-3)
                .count()
        })
        .unwrap_or(0);
    ui.label(format!(
        "FEM contact-patch nodes: {}  (pressure↓ or load↑ → grows)",
        patch_nodes
    ));
    ui.label(format!(
        "Pressure bins span long/width: {:.1}/{:.1} mm, peak {:.0} kPa, load {:.0} N",
        state.last_pressure_longitudinal_span_m * 1000.0,
        state.last_pressure_lateral_span_m * 1000.0,
        state.last_pressure_peak_pa / 1000.0,
        state.last_pressure_total_load_n
    ));
    ui.small(format!(
        "Longitudinal pressure-bin load: {}",
        tb_format_pressure_bins(&state.last_pressure_longitudinal_loads_n)
    ));
    ui.small(format!(
        "Widthwise pressure-bin load: {}",
        tb_format_pressure_bins(&state.last_pressure_lateral_loads_n)
    ));
    ui.label(format!("Bekker terrain backend: {}", state.last_bekker_backend));
    if let Some(err) = &state.last_bekker_error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 80), err);
    }
    ui.checkbox(&mut state.soil_summary_diagnostics, "Collect MPM/Bekker soil summaries");
    if state.soil_summary_diagnostics {
        ui.label(format!(
            "Last Bekker sample — rut max/mean: {:.1}/{:.1} mm  active cells {}",
            state.last_rut_depth_m * 1000.0,
            state.last_rut_mean_m * 1000.0,
            state.last_rut_cells
        ));
    } else {
        ui.small("MPM/Bekker summaries disabled; physical terrain snapshots remain enabled.");
    }
    ui.small(
        "The rut map is owned by TireBekkerFemTerrainOwner: contact-cell pressure/shear writes \
         Bekker height/sinkage, then exports TireFemContactTerrain for GPU FEM sinkage/rolling-loss sampling.",
    );
    ui.label(format!("MPM terrain backend: {}", state.last_mpm_backend));
    if let Some(err) = &state.last_mpm_error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 80), err);
    }
    if state.soil_summary_diagnostics {
        if let Some((generation, revision)) = state.last_mpm_observation {
            ui.label(format!(
                "MPM sample g{generation}/r{revision}, {} terrain revisions old — max {:.1} mm, active {}, skipped {}",
                state.mpm_terrain_revision.saturating_sub(revision), state.last_mpm_compaction_m * 1000.0,
                state.last_mpm_active_particles, state.mpm_observation_backpressure,
            ));
        } else {
            ui.label("MPM optional compaction sample pending");
        }
        if let Some(error) = &state.last_mpm_observation_error {
            ui.colored_label(egui::Color32::from_rgb(255, 180, 80), format!("MPM telemetry: {error}"));
        }
    }
    ui.small(
        "The MPM bridge owns a persistent MpmSimulation: tire contact cells write directly into \
         live MPM particle/compaction buffers, then rasterize GPU terrain buffers. CPU contact queries \
         use one blocking terrain-only snapshot; optional statistics use a three-slot asynchronous 16-byte readback ring.",
    );
    ui.label(format!("DEM terrain backend: {}", state.last_dem_backend));
    if let Some(err) = &state.last_dem_error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 80), err);
    }
    ui.label(format!(
        "DEM→FEM sinkage {:.1} mm  velocity cells {}  grains {}",
        state.last_dem_sinkage_m * 1000.0,
        state.last_dem_velocity_cells,
        state.last_dem_active_grains
    ));
    ui.label(format!(
        "Resident DEM reseeds: {}",
        state.last_dem_resident_reseeds
    ));
    ui.small(
        "The DEM bridge steps a live sparse-LBVH grain bed from tire contact cells. In GPU mode \
         it keeps the next DEM state resident, rasterizes resident height/sinkage/surface-velocity \
         rows, and hands those buffers directly to FEM contact feedback.",
    );
    ui.separator();
    ui.label(format!(
        "Thermal layers °C  surface {:.1}  tread {:.1}",
        state.thermal.surface_c, state.thermal.tread_c
    ));
    ui.label(format!("Thermal/wear backend: {}", state.last_thermal_backend));
    if let Some(err) = &state.last_thermal_error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 80), err);
    }
    ui.label(format!(
        "                 carcass {:.1}  rim {:.1}",
        state.thermal.carcass_c, state.thermal.rim_c
    ));
    ui.label(format!(
        "Thermal μ-fac {:.2}   carcass pressure/stiffness fac {:.2}",
        state.thermal.mu_factor(),
        state.last_pressure_factor
    ));
    ui.label(format!(
        "Tread wear {:.2} mm of {:.1} mm   wear μ-fac {:.2}",
        state.wear.depth_m * 1000.0,
        state.wear.max_depth_m * 1000.0,
        state.wear.mu_factor()
    ));
    let condition = state.thermal.condition_report(&state.wear);
    ui.label(format!(
        "Condition: {}   abuse {:.0}%   wear-rate ×{:.1}",
        condition.level.label(),
        condition.abuse_index * 100.0,
        condition.wear_rate_factor
    ));
    ui.label(format!(
        "Combined μ-factor on friction cap: {:.2}",
        state.last_mu_factor
    ));
    ui.separator();
    tire_rig_replay_panel(ui, &mut state.replay_panel);
    ui.label(format!(
        "Replay thermal batch: {} ({} snapshots)",
        state.last_replay_thermal_backend, state.last_replay_thermal_count
    ));
    if let Some(err) = &state.last_replay_thermal_error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 80), err);
    }
    ui.separator();
    ui.small(
        "Heading arrow (white) points along the tire's natural rolling \
         direction; relative-velocity arrow (cyan) shows where the road \
         is moving. The orange wedge between them is the slip angle. \
         Red arrow at the contact = Fx, cyan = Fy, white = Fn. \
         The two charts behind the tire are the canonical Fy(α) and \
         Fx(κ) sweep curves — yellow dots mark the current operating \
         point. The current force readout comes from actual deformed FEM \
         tread nodes, while the charts remain ideal brush sweeps for \
         comparison. The first three vertical bars left of the tire are \
         peak rubber temperature (red), wear (orange) and current \
         μ-factor (green); the four bars behind them are thermal layers: \
         surface, tread, carcass and rim.",
    );
}
