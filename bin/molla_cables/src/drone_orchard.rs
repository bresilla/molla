#[allow(dead_code)]
mod adaptive_step;
mod ag_drone;
mod ag_mission;
mod ag_model_lab;
#[allow(dead_code)]
mod apple_tree_model;
#[allow(dead_code)]
mod apple_tree_sim;
#[allow(dead_code)]
mod branch;
#[allow(dead_code)]
mod branch_draw;
mod crosswind;
#[allow(dead_code)]
mod draw;
mod drone_asset;
mod drone_flight_lab;
#[allow(dead_code)]
mod grab_handle;
mod leaf_flight_lab;
mod leaf_focus;
#[allow(dead_code)]
mod leafy_draw;
#[allow(dead_code)]
mod leafy_model;
mod orchard_model;
mod orchard_sim;
mod orchard_spray;
#[allow(dead_code)]
mod scene;
mod foliage_layout;
mod studio;
mod video_capture;

use eframe::egui::{self, Color32};
use molla_compute::Engine;
use molla_math::Vec3;
use orchard_sim::{AIR_DT, FLIGHT_SECONDS, Orchard};
use std::{path::Path, time::Instant};

const SPRAY_MIST_OPACITY: f64 = 0.012;
/// Second-row station whose tree is inspected after spraying in the crosswind.
const DRIFT_STATION: usize = orchard_model::TREES_PER_ROW + 1;

fn camera() -> draw::Camera {
    draw::Camera {
        yaw: 0.85,
        pitch: 0.34,
        distance: 22.0,
        target: Vec3::new(0.0, 1.35, 0.0),
    }
}

fn paint(ui: &mut egui::Ui, world: &Orchard, camera: &mut draw::Camera, waves: bool) {
    let rect = ui.available_rect_before_wrap();
    let leaves = leafy_draw::instances(&world.leaves, &world.particles, &world.tree.model);
    let mut foliage = Vec::new();
    if world.spray.is_some() {
        foliage.extend(ag_drone::mesh(&world.drone));
        foliage.extend(leaf_focus::beads(world, camera));
    } else {
        foliage.extend(drone_asset::mesh(&world.drone, world.time()));
    }
    branch_draw::paint_cinematic_leaves(ui, &world.tree.model, camera, foliage, leaves);
    if let Some(spray) = &world.spray {
        let points: Vec<_> = spray
            .parcels
            .iter()
            .flat_map(|p| {
                let radius = 0.025 + 0.025 * (p.age / 0.4).clamp(0.0, 1.0);
                [
                    (
                        p.drop.position,
                        radius,
                        (SPRAY_MIST_OPACITY * (0.04 / radius).powi(2)) as f32,
                    ),
                    (
                        p.drop.position,
                        0.0008 + 0.0005 * (p.drop.diameter() / orchard_spray::DIAMETER_RANGE_M[1]),
                        0.10,
                    ),
                ]
            })
            .collect();
        studio::paint_spray(ui.painter(), camera, rect, &points);
    }
    if waves {
        studio::paint_vapor(ui.painter(), camera, rect, &world.air_paths, world.time());
    }
}

fn report(
    world: &mut Orchard,
    engine: &Engine,
    elapsed: f64,
) -> molla_core::Result<serde_json::Value> {
    world.collect_air_boundary(engine)?;
    let air = world.diagnose(engine)?;
    let root_error = world
        .tree
        .model
        .fibers()
        .iter()
        .filter_map(|f| {
            f.pin
                .map(|pin| f.motion.pose.position.distance(pin.pose.position))
        })
        .fold(0.0, f64::max);
    let mut result = serde_json::json!({
        "trees": orchard_model::TREE_COUNT, "wood_cells": world.tree.model.fibers().len(),
        "leaves": world.particles.len(), "attached": world.attached(), "shed": world.particles.len() - world.attached(),
        "simulation_seconds": world.time(), "wall_seconds": elapsed,
        "mean_air_frame_ms": elapsed * 1000.0 / world.frames.max(1) as f64,
        "stage_mean_ms": (["prepare", "sample_air", "flight", "leaves", "loads", "wood_impulses", "wood_step", "air_reactions", "air_step", "air_enqueue", "air_wait_and_tracers"]
            .into_iter().zip(world.stage_seconds).map(|(name, seconds)| (name.to_string(), serde_json::json!(seconds * 1000.0 / world.frames.max(1) as f64))).collect::<serde_json::Map<_, _>>()),
        "coupling": world.coupling, "motors": world.motors,
        "leaf_strength_scale": world.leaf_strength_scale,
        "represented_foliage_mass_kg": world.coefficients.iter().map(|a| a.mass).sum::<f64>(),
        "physics_hz": world.physics_substeps * 60,
        "ambient_wind": world.ambient_wind,
        "escaped_detached_leaves": world.escaped_leaves(),
        "maximum_detached_horizontal_travel_m": world.maximum_leaf_travel,
        "peak_leaf_spin_rad_s": world.peak_leaf_spin,
        "external_boundary_angular_impulse": world.boundary_angular_impulse.to_array(),
        "leaf_track_columns": ["time_s","x_m","y_m","z_m","normal_x","normal_y","normal_z"],
        "leaf_tracks": world.leaf_tracks.iter().map(|t|serde_json::json!({"index":t.index,"samples":t.samples})).collect::<Vec<_>>(),
        "settling_seconds": world.settling_seconds, "settling_speed_m_s": world.settling_speed,
        "drone_position": world.drone.pose()?.position.to_array(),
        "drone_velocity": world.drone.velocity()?.to_array(),
        "peak_leaf_air_force_n": world.peak_air_force,
        "nearest_rotor_leaf_m": world.nearest_rotor_leaf,
        "peak_shedding_wind_m_s": world.peak_shedding_wind,
        "shed_per_tree": world.particles.chunks(world.particles.len() / orchard_model::TREE_COUNT).map(|p| p.iter().filter(|p| !p.attached).count()).collect::<Vec<_>>(),
        "peak_wood_displacement_m": world.peak_branch_displacement,
        "root_error_m": root_error, "wood_failures": world.tree.stress_breaks,
        "air_divergence_rms": air.divergence_rms, "air_divergence_max": air.divergence_max,
        "air_face_velocity_max": air.speed_max, "ground_air_flux_max": air.ground_flux_max,
        "external_boundary_impulse": world.boundary_impulse.to_array(),
        "leaf_flight_model": "APW circulation with reduced 3D added-mass extension",
        "sources": ["AppleQSM Row13 tree1 / Appleskel", "Bridson and Muller-Fischer 2007", "Bangura et al. 2016", "Andersen, Pesavento and Wang 2005b, doi:10.1017/S0022112005005847"],
    });
    result["drone_mass_kg"] = serde_json::json!(world.drone.config.mass);
    result["exterior_wake_decay_length_m"] = serde_json::json!(orchard_sim::EXTERIOR_WAKE_LENGTH);
    result["subgrid_curl_strength"] = serde_json::json!(if world.ambient_wind {
        orchard_sim::TURBULENCE_STRENGTH
    } else {
        0.0
    });
    result["leaf_camber_fraction"] = serde_json::json!([0.03, 0.08]);
    result["leaf_area_per_mass_range"] = serde_json::json!([8.0, 13.0]);
    result["leaf_relaunches"] = serde_json::json!(world.leaf_relaunches);
    result["unique_relaunched_leaves"] = serde_json::json!(world.relaunched_leaves.len());
    result["airborne_rising_steps"] = serde_json::json!(world.airborne_rising_steps);
    result["ground_rolling_steps"] = serde_json::json!(world.ground_rolling_steps);
    result["downwash_level"] = serde_json::json!(world.downwash_level);
    result["wood_flexibility"] = serde_json::json!(world.wood_flexibility);
    result["peak_trunk_displacement_m"] = serde_json::json!(world.peak_trunk_displacement);
    result["peak_limb_displacement_m"] = serde_json::json!(world.peak_limb_displacement);
    result["visited_canopy_sides"] = serde_json::json!(world.visited_canopy_sides);
    result["trees_per_row"] = serde_json::json!(orchard_model::TREES_PER_ROW);
    result["row_length_m"] = serde_json::json!(2.0 * orchard_model::ROW_HALF_LENGTH);
    result["flight_seconds"] = serde_json::json!(FLIGHT_SECONDS);
    result["row_centers_x_m"] = serde_json::json!(orchard_model::ROW_X);
    result["tree_overflight_horizontal_gaps_m"] =
        serde_json::json!(world.tree_overflight_distances);
    result["flight_trace_txyz"] = serde_json::json!(world.flight_trace);
    result["flight_controller"] = serde_json::json!(
        "bounded-jerk references, acceleration feed-forward, thrust-priority quaternion control, measured-load observer"
    );
    result["flight_dynamics_columns"] = serde_json::json!([
        "time_s",
        "vx",
        "vy",
        "vz",
        "shaft_x",
        "shaft_y",
        "shaft_z",
        "air_force_x",
        "air_force_y",
        "air_force_z",
        "heading_rad",
        "tilt_rad"
    ]);
    result["flight_dynamics"] = serde_json::json!(world.flight_dynamics);
    if let Some(spray) = &world.spray {
        result["spray"] = serde_json::json!({
            "pressure_ratio":orchard_spray::PRESSURE_RATIO,
            "nozzle_flow_coefficient_ratio":orchard_spray::NOZZLE_FLOW_RATIO,
            "flow_l_s":orchard_spray::flow_l_s(),"exit_speed_m_s":orchard_spray::exit_speed_m_s()*orchard_spray::DOWNWARD_EXIT_GAIN,
            "base_exit_speed_m_s":orchard_spray::exit_speed_m_s(),
            "diameter_um_range":orchard_spray::DIAMETER_RANGE_M.map(|d|d*1e6),
            "parcel_rate_per_s":orchard_spray::PARCELS_S,
            "cone_full_angle_deg":orchard_spray::effective_cone_angle_deg(),
            "base_cone_full_angle_deg":2.0*orchard_spray::CONE_HALF_ANGLE_DEG,
            "downward_exit_gain":orchard_spray::DOWNWARD_EXIT_GAIN,
            "axial_exit_speed_m_s":orchard_spray::exit_speed_m_s()*orchard_spray::DOWNWARD_EXIT_GAIN,
            "minimum_downward_exit_speed_m_s":orchard_spray::exit_speed_m_s()*orchard_spray::DOWNWARD_EXIT_GAIN*orchard_spray::CONE_HALF_ANGLE_DEG.to_radians().cos(),
            "dispersion_intensity":spray.dispersion_intensity,
            "dispersion_correlation_s":orchard_spray::DISPERSION_CORRELATION_S,
            "spread_depths_m":orchard_spray::SPREAD_DEPTHS_M,
            "spread_crossings":spray.spread_crossings,
            "spread_rms_m":spray.spread_rms(),
            "parcel_capacity":orchard_spray::CAPACITY,
            "wind_sample_count":spray.wind_samples,
            "max_sampled_air_speed_m_s":spray.max_sampled_air_speed,
            "max_horizontal_drift_m":spray.max_horizontal_drift,
            "model":"original compact agricultural quadrotor", "liquid":"illustrative water",
            "phase":format!("{:?}",spray.mission.phase), "station":spray.mission.station,
            "dose_seconds_per_tree":spray.mission.dose, "max_spray_speed_m_s":spray.mission.max_spray_speed,
            "max_spray_position_error_m":spray.mission.max_spray_error,
            "tank_litres":spray.tank_litres,"emitted_litres":spray.emitted_litres,
            "airborne_litres":spray.airborne_litres(),"canopy_litres":spray.canopy_litres,"ground_litres":spray.ground_litres,
            "peak_parcels":spray.peak_parcels,"capacity_stops":spray.capacity_stops,
            "wet_leaves":spray.wet_litres.iter().filter(|v|**v>0.0).count(),
            "deposition_per_tree_litres":spray.wet_litres.chunks(spray.wet_litres.len()/orchard_model::TREE_COUNT).map(|a|a.iter().sum::<f64>()).collect::<Vec<_>>(),
            "finite":spray.parcels.iter().all(|p|p.drop.position.is_finite()&&p.drop.velocity.is_finite()),
            "coupling":"one-way water parcels in GPU air plus stochastic subgrid dispersion; fixed aircraft mass; no evaporation or liquid film"
        });
        result["spray"]["settling"] = serde_json::json!({
            "gravity_m_s2":9.81,
            "airborne_volume_time_l_s":spray.airborne_volume_time,
            "upward_volume_time_l_s":spray.upward_volume_time,
            "upward_volume_time_fraction":spray.upward_volume_time/spray.airborne_volume_time.max(1e-30),
            "max_rise_above_nozzle_m":spray.max_rise_above_nozzle
        });
        result["spray"]["appearance"] = serde_json::json!({
            "color":"neutral white", "density_kernel_opacity":SPRAY_MIST_OPACITY
        });
        result["spray"]["boom"] = serde_json::json!({
            "shape":"square", "side_m":2.0*orchard_spray::BOOM_HALF_SIDE_M,
            "nozzle_positions_local_m":([0,1,2,3].map(|i| orchard_spray::nozzle_local(i).to_array()))
        });
        let per_tree = spray.wet_litres.len() / orchard_model::TREE_COUNT;
        result["spray"]["wet_leaves_per_tree"] = serde_json::json!(spray
            .wet_litres
            .chunks(per_tree)
            .map(|a| a.iter().filter(|v| **v > 0.0).count())
            .collect::<Vec<_>>());
        result["spray"]["parcels_per_tree"] = serde_json::json!(spray.deposits.iter().fold(
            vec![0usize; orchard_model::TREE_COUNT],
            |mut n, d| {
                n[d.leaf / per_tree] += 1;
                n
            }
        ));
    }
    result["tree_yaw_rad"] = serde_json::json!(
        (0..orchard_model::TREE_COUNT)
            .map(orchard_model::tree_yaw)
            .collect::<Vec<_>>()
    );
    if let Some(wind) = &world.crosswind {
        let (ambient, gusts) = wind.background(world.time());
        result["crosswind"] = serde_json::json!({
            "speed_m_s": wind.speed, "announced_s": wind.announced, "onset_s": wind.onset(),
            "full_s": wind.onset().map(|t| t + crosswind::RAMP_S),
            "trigger": "spray mission reaches the first station of the second row",
            "final_ambient_m_s": ambient.to_array(), "final_gust_strength": gusts,
            "direction": "+x, across the rows"
        });
    }
    Ok(result)
}

fn full_row_verified(
    world: &Orchard,
    report: &serde_json::Value,
    furthest_z: f64,
    initial: Vec3,
) -> molla_core::Result<bool> {
    Ok(furthest_z > orchard_model::ROW_HALF_LENGTH - 0.1
        && world.drone.pose()?.position.distance(initial) < 0.1
        && world.drone.velocity()?.length() < 0.1
        && world.attached() < world.particles.len()
        && world.attached() > world.particles.len() * 3 / 4
        && (0.15..0.65).contains(&world.nearest_rotor_leaf)
        && world.peak_shedding_wind > 2.0
        && (!world.ambient_wind
            || (world.escaped_leaves() > 0
                && world.maximum_leaf_travel > 3.0
                && world.peak_leaf_spin > 1.0))
        && world.visited_canopy_sides.iter().all(|visited| *visited)
        && world.tree_overflight_distances.iter().all(|gap| *gap < 0.3)
        && world.tree.stress_breaks == 0
        && report["root_error_m"] == 0.0
        && world.air_diagnostics.unwrap().divergence_rms < 0.01
        && world.air_diagnostics.unwrap().ground_flux_max == 0.0
        && world
            .particles
            .iter()
            .all(|p| p.position.is_finite() && p.velocity.is_finite() && p.position.y >= 0.0))
}

fn write_state(world: &Orchard, path: &str) -> video_capture::Result<()> {
    let data = serde_json::json!({
        "wood": world.tree.model.fibers().iter().map(|f| serde_json::json!({"id":f.id,"center":f.center().to_array(),"velocity":f.motion.linear_velocity.to_array()})).collect::<Vec<_>>(),
        "leaves": world.particles.iter().map(|p| serde_json::json!({"position":p.position.to_array(),"velocity":p.velocity.to_array(),"attached":p.attached})).collect::<Vec<_>>()
    });
    std::fs::write(path, serde_json::to_string(&data)?)?;
    Ok(())
}

struct App {
    engine: Engine,
    world: Orchard,
    camera: draw::Camera,
    paused: bool,
    waves: bool,
    error: Option<String>,
    step_ms: f64,
    last: Instant,
    accumulator: f64,
    settled: bool,
    follow_sprayer: bool,
}
impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = Instant::now();
        let elapsed = now.duration_since(self.last).as_secs_f64().min(0.1);
        self.last = now;
        ui.horizontal(|ui| {
            ui.heading("Molla · drone orchard");
            ui.checkbox(&mut self.paused, "Pause");
            if ui.button("Reset").clicked() {
                let reset = (|| -> molla_core::Result<_> {
                    let mut world = Orchard::configured(
                        &self.engine,
                        self.world.ambient_wind,
                        self.world.downwash_level,
                        self.world.wood_flexibility,
                    )?;
                    world.set_physics_hz(self.world.physics_substeps * 60)?;
                    if self.settled {
                        world.settle(&self.engine)?;
                    }
                    if self.world.spray.is_some() {
                        world.enable_spraying()?;
                    }
                    Ok(world)
                })();
                match reset {
                    Ok(world) => {
                        self.world = world;
                        self.error = None;
                        self.accumulator = 0.0;
                    }
                    Err(e) => self.error = Some(e.to_string()),
                }
            }
            if ui.checkbox(&mut self.world.autopilot, "Fly row").changed()
                && !self.world.autopilot
                && let Ok(pose) = self.world.drone.pose()
            {
                self.world.target = pose.position;
            }
            ui.checkbox(&mut self.world.coupling, "Air coupling");
            ui.checkbox(&mut self.waves, "Air vapor");
            if self.world.spray.is_some() {
                ui.checkbox(&mut self.follow_sprayer, "Follow sprayer");
            }
            ui.add(
                egui::Slider::new(&mut self.world.leaf_strength_scale, 0.25..=2.0)
                    .text("Leaf strength"),
            );
        });
        ui.horizontal(|ui| {
            if let Some(spray) = &self.world.spray {
                ui.label(format!(
                    "AG sprayer · tree {}/{} · {:?} · pump {} · {:.0} mL applied",
                    spray.mission.station + 1,
                    orchard_model::TREE_COUNT,
                    spray.mission.phase,
                    if spray.on { "ON" } else { "off" },
                    spray.emitted_litres * 1000.0
                ));
            }
            ui.label(format!(
                "{:.1} s · {} / {} leaves attached · {:.1} ms / air frame",
                self.world.time(),
                self.world.attached(),
                self.world.particles.len(),
                self.step_ms
            ));
            ui.label("Right-drag: orbit · wheel: zoom");
        });
        if let Some(error) = &self.error {
            ui.colored_label(Color32::LIGHT_RED, error);
        }
        if !self.paused && self.error.is_none() {
            self.accumulator = (self.accumulator + elapsed).min(AIR_DT);
            let start = Instant::now();
            let mut steps = 0;
            while self.accumulator >= AIR_DT && steps < 1 {
                match self.world.advance(&self.engine) {
                    Ok(()) => {
                        self.accumulator -= AIR_DT;
                        steps += 1;
                    }
                    Err(e) => {
                        self.error = Some(e.to_string());
                        break;
                    }
                }
            }
            if steps > 0 {
                self.step_ms = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
            }
        }
        if self.follow_sprayer
            && let Ok(pose) = self.world.drone.pose()
        {
            self.camera.target = self
                .camera
                .target
                .lerp(pose.position - Vec3::Y * 0.95, 0.15);
        }
        paint(ui, &self.world, &mut self.camera, self.waves);
        ctx.request_repaint();
    }
}

fn airflow_visible(args: &[String]) -> bool {
    args.iter().any(|a| a == "--show-airflow") && !args.iter().any(|a| a == "--hide-airflow")
}

#[test]
fn airflow_visualization_requires_explicit_opt_in() {
    assert!(!airflow_visible(&[]));
    assert!(airflow_visible(&["--show-airflow".into()]));
    assert!(!airflow_visible(&[
        "--show-airflow".into(),
        "--hide-airflow".into()
    ]));
}

fn main() -> video_capture::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|a| a == "--ag-model-lab") {
        return ag_model_lab::capture(&args);
    }
    if args.iter().any(|a| a == "--flight-lab") {
        return drone_flight_lab::capture(&args);
    }
    if args.iter().any(|a| a == "--leaf-lab") {
        return leaf_flight_lab::capture(&args);
    }
    let show_airflow = airflow_visible(&args);
    let value = |name: &str| {
        args.windows(2)
            .find(|p| p[0] == name)
            .map(|p| p[1].as_str())
    };
    let seconds: f64 = value("--seconds").unwrap_or("12").parse()?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("seconds must be finite and positive".into());
    }
    let engine = Engine::new_gpu()?;
    eprintln!("Orchard physics adapter: {:?}", engine.adapter_info());
    let mut world = Orchard::configured(
        &engine,
        !args.iter().any(|a| a == "--still-air"),
        value("--downwash")
            .map(str::parse)
            .transpose()?
            .unwrap_or(orchard_sim::DOWNWASH_LEVEL),
        value("--wood-flex")
            .map(str::parse)
            .transpose()?
            .unwrap_or(orchard_model::WOOD_FLEXIBILITY),
    )?;
    if let Some(hz) = value("--physics-hz") {
        world.set_physics_hz(hz.parse()?)?;
    }
    if let Some(strength) = value("--leaf-strength") {
        world.leaf_strength_scale = strength.parse()?;
    }
    if !world.leaf_strength_scale.is_finite() || world.leaf_strength_scale <= 0.0 {
        return Err("leaf strength must be finite and positive".into());
    }
    let settled = !args.iter().any(|a| a == "--cold-start");
    if settled {
        world.settle(&engine)?;
    }
    if args.iter().any(|a| a == "--ag-spray") {
        world.enable_spraying()?;
    }
    if let Some(speed) = value("--crosswind") {
        if world.spray.is_none() {
            return Err("--crosswind needs the --ag-spray mission".into());
        }
        world.enable_crosswind(speed.parse()?)?;
    }
    world.coupling = !args.iter().any(|a| a == "--no-coupling");
    world.motors = !args.iter().any(|a| a == "--motors-off");
    if let Some(path) = value("--verify") {
        if seconds < 4.0 {
            return Err("verification needs at least four seconds".into());
        }
        let mut control = Orchard::configured(
            &engine,
            world.ambient_wind,
            world.downwash_level,
            world.wood_flexibility,
        )?;
        control.set_physics_hz(world.physics_substeps * 60)?;
        control.leaf_strength_scale = world.leaf_strength_scale;
        if settled {
            control.settle(&engine)?;
        }
        control.coupling = false;
        let start = Instant::now();
        let mut elapsed = [0.0; 2];
        for _ in 0..(seconds / AIR_DT).ceil() as usize {
            let stamp = Instant::now();
            world.advance(&engine)?;
            elapsed[0] += stamp.elapsed().as_secs_f64();
            let stamp = Instant::now();
            control.advance(&engine)?;
            elapsed[1] += stamp.elapsed().as_secs_f64();
        }
        let differences: Vec<_> = world
            .tree
            .model
            .fibers()
            .iter()
            .map(|f| {
                control
                    .tree
                    .model
                    .fiber(f.id)
                    .map_or(f64::INFINITY, |other| f.center().distance(other.center()))
            })
            .collect();
        let maximum = differences.iter().copied().fold(0.0, f64::max);
        let rms =
            (differences.iter().map(|v| v * v).sum::<f64>() / differences.len() as f64).sqrt();
        let coupled = report(&mut world, &engine, elapsed[0])?;
        let quiet = report(&mut control, &engine, elapsed[1])?;
        let shed = world.particles.len() - world.attached();
        let verified =
            maximum.is_finite()
                && maximum > 0.002
                && shed < world.particles.len() / 4
                && control.attached() == control.particles.len()
                && world.drone.pose()?.position.z > -orchard_model::ROW_HALF_LENGTH - 0.3
                && world.tree.stress_breaks == 0
                && control.tree.stress_breaks == 0
                && world
                    .tree
                    .model
                    .fibers()
                    .iter()
                    .chain(control.tree.model.fibers())
                    .all(|f| {
                        f.pin.is_none_or(|pin| {
                            f.motion.pose == pin.pose
                                && f.motion.linear_velocity == Vec3::ZERO
                                && f.motion.angular_velocity == Vec3::ZERO
                        })
                    })
                && [
                    world.air_diagnostics.unwrap(),
                    control.air_diagnostics.unwrap(),
                ]
                .iter()
                .all(|d| d.divergence_rms < 0.01 && d.ground_flux_max == 0.0)
                && world.particles.iter().chain(&control.particles).all(|p| {
                    p.position.is_finite() && p.velocity.is_finite() && p.position.y >= 0.0
                });
        let evidence = serde_json::json!({"verified": verified, "verification_wall_seconds": start.elapsed().as_secs_f64(), "wood_response_max_m": maximum,
            "wood_response_rms_m": rms, "coupled": coupled, "control": quiet});
        if let Some(parent) = Path::new(path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&evidence)?)?;
        println!("{evidence}");
        if !verified {
            return Err("orchard causal verification failed; inspect report".into());
        }
        return Ok(());
    }
    if let Some(path) = value("--probe").or_else(|| value("--verify-row")) {
        let verify_row = value("--verify-row").is_some();
        if verify_row && (seconds - FLIGHT_SECONDS).abs() > 1e-9 {
            return Err(
                format!("full-row verification requires --seconds {FLIGHT_SECONDS}").into(),
            );
        }
        let start = Instant::now();
        let initial = world.drone.pose()?.position;
        let mut furthest_z = initial.z;
        for frame in 0..(seconds / AIR_DT).ceil() as usize {
            world.advance(&engine)?;
            furthest_z = furthest_z.max(world.drone.pose()?.position.z);
            if frame % 60 == 59 {
                eprintln!(
                    "ORCHARD t={:.2} shed={} displacement={:.4}",
                    world.time(),
                    world.particles.len() - world.attached(),
                    world.peak_branch_displacement
                );
            }
        }
        let mut report = report(&mut world, &engine, start.elapsed().as_secs_f64())?;
        report["furthest_drone_z_m"] = serde_json::json!(furthest_z);
        let verified = !verify_row || full_row_verified(&world, &report, furthest_z, initial)?;
        if verify_row {
            report["verified_full_row"] = serde_json::json!(verified);
        }
        if let Some(state) = value("--state") {
            write_state(&world, state)?;
        }
        if let Some(parent) = Path::new(path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
        println!("{report}");
        if !verified {
            return Err("full-row verification failed; inspect report".into());
        }
        return Ok(());
    }
    if let Some(path) = value("--video") {
        let verify_flight = args.iter().any(|a| a == "--verify-flight");
        if verify_flight && (seconds - FLIGHT_SECONDS).abs() > 1e-9 {
            return Err(format!("verified capture requires {FLIGHT_SECONDS} seconds").into());
        }
        let initial = world.drone.pose()?.position;
        let mut furthest_z = initial.z;
        let mut capture =
            video_capture::Capture::new_with_title(Path::new(path), "Molla | Drone orchard")?;
        let mut camera = camera();
        let close_up = args.iter().any(|a| a == "--camera-drone");
        let frames = (seconds * video_capture::FPS as f64).ceil() as usize;
        let keyframes = value("--keyframes").map(Path::new);
        if let Some(directory) = keyframes {
            std::fs::create_dir_all(directory)?;
        }
        let leaf_shot = world.spray.is_some() && !args.iter().any(|a| a == "--no-leaf-focus");
        let shots: Vec<usize> = if world.crosswind.is_some() {
            vec![0, DRIFT_STATION]
        } else {
            vec![0]
        };
        let mut departed = vec![None; shots.len()];
        let mut focuses: Vec<leaf_focus::Focus> = Vec::new();
        let mut shown = vec![None; shots.len()];
        let mut wind_shown = false;
        let start = Instant::now();
        for frame in 0..frames {
            world.advance(&engine)?;
            furthest_z = furthest_z.max(world.drone.pose()?.position.z);
            world.advance(&engine)?;
            furthest_z = furthest_z.max(world.drone.pose()?.position.z);
            let position = world.drone.pose()?.position;
            let mut focus_still = None;
            if let Some(spray) = &world.spray {
                let blend = ((world.time() - 1.0) / 3.0).clamp(0.0, 1.0);
                let blend = blend * blend * (3.0 - 2.0 * blend);
                camera = self::camera();
                camera.target = Vec3::new(0.0, 1.5, 0.0).lerp(position - Vec3::Y * 0.95, blend);
                camera.distance = 22.0 - 16.2 * blend;
                camera.pitch = 0.28;
                for (k, &station) in shots.iter().enumerate() {
                    if leaf_shot && spray.mission.station > station {
                        departed[k].get_or_insert(world.time());
                    }
                    if !(leaf_shot
                        && focuses.len() == k
                        && departed[k].is_some_and(|t| world.time() >= t + leaf_focus::LEAD_IN_S))
                    {
                        continue;
                    }
                    let tree = ag_mission::tree_index(station);
                    let twin = focuses
                        .first()
                        .map(|first| leaf_focus::twin(&world, first.leaf, first.tree, tree))
                        .filter(|&leaf| world.particles[leaf].attached);
                    let mut f = twin
                        .or_else(|| leaf_focus::select(&world, tree))
                        .and_then(|leaf| {
                            leaf_focus::Focus::new(&world, leaf, tree, world.time(), camera.yaw)
                        })
                        .ok_or_else(|| format!("no leaf on tree {} to inspect", tree + 1))?;
                    if let (Some(first), Some(wind)) = (focuses.first(), &world.crosswind) {
                        let (title, label) = if twin.is_some() {
                            ("THE SAME LEAF", "same leaf")
                        } else {
                            ("ONE LEAF UP CLOSE", "best leaf")
                        };
                        f.title = format!("{title} · {} CROSSWIND", crosswind::label(wind.speed));
                        f.reference = Some((
                            leaf_focus::summary(&world, first),
                            format!("{label} on tree {} in the light breeze", first.tree + 1),
                        ));
                    }
                    focuses.push(f);
                }
                for (k, f) in focuses.iter().enumerate() {
                    camera = f.camera(&world, camera, world.time());
                    let middle = f.start + leaf_focus::ZOOM_S + leaf_focus::HOLD_S * 0.5;
                    if shown[k].is_none() && world.time() >= middle {
                        shown[k] = Some(leaf_focus::tally(&spray.deposits, f.leaf));
                        let still = if k == 0 { "leaf.png" } else { "drift-leaf.png" };
                        focus_still = Some(Path::new(path).with_extension(still));
                    }
                }
            } else if close_up {
                camera.target = position;
                camera.distance = 2.0;
                camera.pitch = 0.45;
            } else {
                let blend = ((world.time() - 2.0) / 4.0)
                    .clamp(0.0, 1.0)
                    .min(((FLIGHT_SECONDS - world.time()) / 4.0).clamp(0.0, 1.0));
                let blend = blend * blend * (3.0 - 2.0 * blend);
                camera.target = Vec3::new(0.0, 1.35 + 0.35 * blend, position.z * blend);
                camera.distance = 22.0 - 12.5 * blend;
            }
            let still = if frame + 1 == frames {
                Some(Path::new(path).with_extension("png"))
            } else if focus_still.is_some() {
                focus_still
            } else if !wind_shown
                && world.crosswind.and_then(|w| w.onset()).is_some_and(|t| {
                    world.time() >= t + crosswind::RAMP_S * 0.5
                })
            {
                wind_shown = true;
                Some(Path::new(path).with_extension("wind.png"))
            } else if [59, 179, 359].contains(&frame) || frame % 300 == 299 {
                keyframes.map(|directory| directory.join(format!("frame-{:04}.png", frame + 1)))
            } else {
                None
            };
            capture.frame(
                |ui| {
                    let rect = ui.available_rect_before_wrap();
                    paint(ui, &world, &mut camera, show_airflow);
                    for f in &focuses {
                        leaf_focus::overlay(ui.painter(), rect, &camera, &world, f);
                    }
                    if let Some(wind) = &world.crosswind {
                        crosswind::overlay(ui.painter(), rect, &camera, wind, world.time());
                    }
                },
                still.as_deref(),
            )?;
            if frame % 30 == 29 {
                eprintln!(
                    "ORCHARD VIDEO {}/{} frames, shed={}",
                    frame + 1,
                    frames,
                    world.particles.len() - world.attached()
                );
            }
        }
        capture.finish()?;
        if let Some(state) = value("--state") {
            write_state(&world, state)?;
        }
        let mut report = report(&mut world, &engine, start.elapsed().as_secs_f64())?;
        let verified = !verify_flight || full_row_verified(&world, &report, furthest_z, initial)?;
        report["airflow_visible"] = serde_json::json!(show_airflow);
        for (k, f) in focuses.iter().enumerate() {
            let Some(spray) = &world.spray else { break };
            let last = leaf_focus::summary(&world, f);
            let landed: Vec<_> = spray.deposits.iter().filter(|d| d.leaf == f.leaf).collect();
            let shown = shown[k].unwrap_or(last.leaf);
            report[if k == 0 { "focus_leaf" } else { "drift_leaf" }] = serde_json::json!({
                "tree": f.tree + 1, "leaf_index": f.leaf, "leaf_length_m": f.length,
                "window_s": [f.start, f.end()], "attached_at_end": world.particles[f.leaf].attached,
                "shown_parcels": shown.parcels, "shown_litres": shown.litres,
                "shown_equivalent_droplets": shown.droplets,
                "final_parcels": last.leaf.parcels, "final_litres": last.leaf.litres,
                "final_equivalent_droplets": last.leaf.droplets,
                "first_landing_s": landed.first().map(|d| d.time),
                "last_landing_s": landed.last().map(|d| d.time),
                "tree_parcels": last.tree_parcels, "tree_wet_leaves": last.wet_leaves,
                "tree_mean_parcels_per_wet_leaf": last.tree_parcels as f64 / last.wet_leaves.max(1) as f64,
                "selection": "attached leaf of the inspected tree with |normal_y| > 0.5 and the most landed parcels when the camera leaves the drone",
                "droplet_count": "sum over landed parcels of parcel volume / (pi/6 d^3)"
            });
            if k > 0 {
                let twin = leaf_focus::twin(&world, focuses[0].leaf, focuses[0].tree, f.tree);
                report["drift_leaf"]["same_leaf_as_focus"] = serde_json::json!(twin == f.leaf);
                report["drift_leaf"]["selection"] = serde_json::json!(
                    "the first close-up's leaf on this tree when still attached, else the best-covered leaf"
                );
            }
        }
        report["camera_mode"] = serde_json::json!(if focuses.len() > 1 {
            "AG overview, canopy follow, first-tree leaf close-up and crosswind leaf close-up"
        } else if !focuses.is_empty() {
            "AG overview, canopy follow and first-tree leaf close-up"
        } else if world.spray.is_some() {
            "AG overview and canopy follow"
        } else if close_up {
            "drone close-up"
        } else {
            "overview and follow"
        });
        if verify_flight {
            report["verified_full_row"] = serde_json::json!(verified);
            report["furthest_drone_z_m"] = serde_json::json!(furthest_z);
        }
        std::fs::write(
            Path::new(path).with_extension("json"),
            serde_json::to_string_pretty(&report)?,
        )?;
        if !verified {
            return Err("captured full flight failed verification; inspect report".into());
        }
        return Ok(());
    }
    let follow_sprayer = world.spray.is_some();
    let mut initial_camera = camera();
    if follow_sprayer {
        initial_camera.distance = 5.8;
        initial_camera.target = world.drone.pose()?.position - Vec3::Y * 0.95;
        initial_camera.pitch = 0.28;
    }
    let app = App {
        engine,
        world,
        camera: initial_camera,
        paused: false,
        waves: show_airflow,
        error: None,
        step_ms: 0.0,
        last: Instant::now(),
        accumulator: 0.0,
        settled,
        follow_sprayer,
    };
    eframe::run_native(
        "Molla · drone orchard",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 960.0]),
            renderer: eframe::Renderer::Wgpu,
            multisampling: 4,
            depth_buffer: 24,
            ..Default::default()
        },
        Box::new(|cc| {
            studio::initialize(cc.wgpu_render_state.as_ref().unwrap());
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )?;
    Ok(())
}
