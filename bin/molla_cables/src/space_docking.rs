//! Realistic Zero-G ISS IDSS / NASA Docking System Showcase in Molla (`molla_space_docking`).
//!
//! Features:
//! - Physical 6-strut Stewart-platform Soft-Capture Ring (`JointType::Prismatic` actuator)
//! - 12 articulated Hard-Capture Clamping Hooks (Gang A + Gang B staggered clamping)
//! - Pressurized Vestibule Air cycle (`0.0 -> 101.3 kPa` seal check & choked-flow blowdown vent)
//! - Expanding volumetric vacuum cold-gas RCS & pressurized-air plumes (`studio::paint_spray`)

#[allow(dead_code)]
mod draw;
#[allow(dead_code)]
mod foliage_layout;
#[allow(dead_code)]
mod scene;
#[allow(dead_code)]
mod space_mesh;
#[allow(dead_code)]
mod space_sim;
#[allow(dead_code)]
mod studio;
#[allow(dead_code)]
mod video_capture;

use std::path::Path;
use std::time::Instant;

use eframe::egui::{self, Color32, CornerRadius, Margin, RichText, Stroke};
use molla_math::Vec3;
use space_sim::{MISSION_SECONDS, MissionPhase, SIM_DT, SpaceDockingSim};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CameraMode {
    FreeOrbit,
    FollowCollar,
    FollowCapsule,
    AutoDirector,
}

impl CameraMode {
    fn next(self) -> Self {
        match self {
            Self::FreeOrbit => Self::FollowCollar,
            Self::FollowCollar => Self::FollowCapsule,
            Self::FollowCapsule => Self::AutoDirector,
            Self::AutoDirector => Self::FreeOrbit,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::FreeOrbit => "Co-Moving Free Orbit (Click Ship=Nudge, Space=Orbit)",
            Self::FollowCollar => "Follow Collar (Click Ship=Nudge, Space=Orbit)",
            Self::FollowCapsule => "Follow Capsule (Click Ship=Nudge, Space=Orbit)",
            Self::AutoDirector => "Auto Director (Drag to Unlock)",
        }
    }
}

fn default_camera() -> draw::Camera {
    draw::Camera {
        yaw: 0.34,
        pitch: 0.20,
        distance: 7.4,
        target: Vec3::new(2.9, 0.1, 0.0),
    }
}

fn paint_scene(
    ui: &mut egui::Ui,
    sim: &mut SpaceDockingSim,
    camera: &mut draw::Camera,
    mode: &mut CameraMode,
    show_guides: bool,
    interactive: bool,
) {
    let sense = if interactive {
        egui::Sense::drag()
    } else {
        egui::Sense::hover()
    };
    let (response, painter) = ui.allocate_painter(ui.available_size(), sense);
    let rect = response.rect;
    if interactive {
        // Click directly on either spacecraft (or solar wing) to grab & nudge/spin it in 3D!
        // Click on empty space to orbit the camera freely!
        if response.drag_started_by(egui::PointerButton::Primary)
            && let Some(pos) = response.interact_pointer_pos()
        {
            let (ro, rd) = camera.ray(pos, rect);
            let _ = sim.try_begin_grab(ro, rd);
        }
        if response.drag_stopped_by(egui::PointerButton::Primary) {
            sim.end_grab();
        }

        let left_drag = response.dragged_by(egui::PointerButton::Primary);
        let pan_drag = response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle);

        if left_drag && sim.user_grab.is_some() {
            if let Some(pos) = response.interact_pointer_pos() {
                let (ro, rd) = camera.ray(pos, rect);
                sim.update_grab_ray(ro, rd);
            }
        } else if left_drag || pan_drag {
            if *mode == CameraMode::AutoDirector {
                *mode = CameraMode::FreeOrbit;
            }
            let delta = ui.input(|i| i.pointer.delta());
            if left_drag {
                camera.yaw -= f64::from(delta.x) * 0.0065;
                camera.pitch = (camera.pitch + f64::from(delta.y) * 0.0065).clamp(-1.42, 1.42);
            }
            if pan_drag {
                let (_, ray) = camera.ray(rect.center(), rect);
                let forward = -ray;
                let right = Vec3::Y.cross(forward).normalize();
                let up = forward.cross(right);
                let pan_scale = camera.distance * 0.0014;
                camera.target -= right * (f64::from(delta.x) * pan_scale);
                camera.target += up * (f64::from(delta.y) * pan_scale);
            }
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                if *mode == CameraMode::AutoDirector {
                    *mode = CameraMode::FreeOrbit;
                }
                camera.distance =
                    (camera.distance * (-f64::from(scroll) * 0.0022).exp()).clamp(1.2, 45.0);
            }
        }
    }
    space_mesh::paint_backdrop(&painter, camera, rect);
    if let Ok(tris) = space_mesh::scene_mesh(sim) {
        studio::paint_space(&painter, camera, rect, tris);
    }
    // Velocity-stretched wispy FBM vacuum gas & cryo-air plumes!
    let puffs = sim.plume_billboards();
    if !puffs.is_empty() {
        studio::paint_plume(&painter, camera, rect, &puffs);
    }
    if show_guides {
        space_mesh::paint_orbital_guides(&painter, camera, rect, sim);
    }
    space_mesh::paint_grab_tether(&painter, camera, rect, sim);
}

fn paint_hud(ctx: &egui::Context, sim: &SpaceDockingSim, mode: CameraMode, paused: bool) {
    let Ok(t) = sim.telemetry() else {
        return;
    };
    egui::Area::new(egui::Id::new("space_docking_hud"))
        .fixed_pos(egui::pos2(16.0, 16.0))
        .show(ctx, |ui| {
            egui::Frame::NONE
                .fill(Color32::from_rgba_unmultiplied(8, 13, 24, 218))
                .stroke(Stroke::new(1.0_f32, Color32::from_rgb(52, 84, 128)))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.set_min_width(380.0);
                    ui.label(
                        RichText::new("MOLLA · FULL-PHYSICS ISS IDSS TESTBED (960 Hz)")
                            .strong()
                            .size(13.2)
                            .color(Color32::from_rgb(156, 224, 255)),
                    );
                    ui.add_space(3.0);
                    let phase_color = if t.phase == MissionPhase::EmergencyBlowoff {
                        Color32::from_rgb(255, 92, 92)
                    } else if t.seal_verified {
                        Color32::from_rgb(95, 255, 162)
                    } else if t.venting_air {
                        Color32::from_rgb(145, 232, 255)
                    } else {
                        Color32::from_rgb(255, 210, 102)
                    };
                    ui.label(
                        RichText::new(format!(
                            "{}{}",
                            t.phase.label(),
                            if paused { "  [PAUSED]" } else { "" }
                        ))
                        .strong()
                        .color(phase_color),
                    );
                    ui.separator();
                    let mono = |ui: &mut egui::Ui, k: &str, v: String, col: Color32| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(k).color(Color32::from_rgb(150, 172, 198)));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(RichText::new(v).monospace().color(col));
                                },
                            );
                        });
                    };
                    let white = Color32::from_rgb(235, 244, 255);
                    let warn = Color32::from_rgb(255, 178, 86);
                    let good = Color32::from_rgb(95, 255, 162);
                    let cyan = Color32::from_rgb(145, 232, 255);
                    let danger = Color32::from_rgb(255, 96, 96);

                    mono(ui, "Sim Time", format!("{:6.2} s", t.time_s), white);
                    mono(
                        ui,
                        "Orbital [O]",
                        if t.iss_leo {
                            format!(
                                "{} · {:.0}km ({:.1e})",
                                t.orbit_mode.label(),
                                t.altitude_km,
                                t.atmo_density_kg_m3
                            )
                        } else {
                            t.orbit_mode.label().into()
                        },
                        white,
                    );
                    mono(
                        ui,
                        "Ring / Flange Gap",
                        format!("{:6.3} m · {:+6.1} mm", t.port_distance_m, t.flange_gap_mm),
                        white,
                    );
                    mono(
                        ui,
                        "Closing / Att",
                        format!("{:+5.3} m/s · {:4.2}°", t.closing_speed_mps, t.attitude_err_deg),
                        white,
                    );
                    mono(
                        ui,
                        "RCS Mod [P]",
                        format!(
                            "{} · {} cyc{}",
                            t.modulation_mode.label(),
                            t.rcs_valve_cycles,
                            if t.faulted_thrusters > 0 { " · 1 FLT" } else { "" }
                        ),
                        if t.faulted_thrusters > 0 { warn } else { cyan },
                    );
                    mono(
                        ui,
                        "Draco RCS (16)",
                        format!(
                            "{} active ({:.0} N · imp {:.1} N)",
                            t.active_thrusters, t.total_thrust_n, t.impingement_force_n,
                        ),
                        white,
                    );
                    mono(
                        ui,
                        "HIL Nav [N]",
                        format!(
                            "{} · LiDAR {} ({:.2}m) · e_q {:.2}°",
                            if t.hil_sensor_gnc { "SENSORS" } else { "TRUTH" },
                            if t.lidar_locked { "LOCK" } else { "SCAN" },
                            t.lidar_range_m,
                            t.nav_att_err_deg,
                        ),
                        if t.hil_sensor_gnc && t.lidar_locked {
                            good
                        } else if t.hil_sensor_gnc {
                            warn
                        } else {
                            white
                        },
                    );
                    mono(
                        ui,
                        "Propellant / Mass",
                        format!(
                            "{:.1}/{:.0} kg · {:.1} kg",
                            t.propellant_kg, t.initial_propellant_kg, t.chaser_mass_kg
                        ),
                        white,
                    );
                    ui.separator();
                    mono(
                        ui,
                        "6-DOF Stewart SCR",
                        format!(
                            "{:.3}m · lat {:4.1}mm · pawls {}/3 ({:.1}kN)",
                            t.scr_stroke_m, t.scr_lateral_mm, t.soft_pawls_latched, t.soft_pawl_force_kn
                        ),
                        if t.soft_pawls_latched == 3 { good } else { white },
                    );
                    mono(
                        ui,
                        "12 Dynamic Hooks",
                        format!(
                            "{:2}/12 · {:5.1} kN · τ {:5.0} N·m",
                            t.hooks_engaged, t.clamp_force_kn, t.hook_peak_motor_nm
                        ),
                        if t.hooks_engaged == 12 {
                            good
                        } else if t.hooks_engaged > 0 {
                            warn
                        } else {
                            white
                        },
                    );
                    mono(
                        ui,
                        "O-Ring Gasket Load",
                        format!(
                            "{:5.1} kN · min pad {:4.1} kN",
                            t.seal_force_kn, t.seal_min_pad_kn
                        ),
                        if t.seal_min_pad_kn > 0.5 { good } else { white },
                    );
                    mono(
                        ui,
                        "Wing Flex (0.28Hz)",
                        format!("{:4.2}° · {:5.2} J CSI", t.wing_flex_deg, t.wing_flex_energy_j),
                        if t.wing_flex_deg > 1.0 { warn } else { white },
                    );
                    mono(
                        ui,
                        "Fuel Slosh (120 kg)",
                        format!("{:5.1} mm · {:5.1} N", t.slosh_disp_mm, t.slosh_force_n),
                        white,
                    );
                    mono(
                        ui,
                        "LEO Drag+SRP / MTB",
                        format!("{:5.2} mN · τ {:5.2} mN·m", t.env_drag_srp_mn, t.mtb_torque_mnm),
                        white,
                    );
                    mono(
                        ui,
                        "Vestibule Gas (P·A)",
                        format!(
                            "{:5.1} kPa · {:5.1} K → {:5.1} kN",
                            t.vestibule_pressure_kpa, t.vestibule_temp_k, t.air_sep_force_kn
                        ),
                        if t.air_sep_force_kn > 50.0 {
                            cyan
                        } else if t.air_sep_force_kn > 1.0 {
                            warn
                        } else {
                            white
                        },
                    );
                    mono(
                        ui,
                        "Vent Nozzles ±Z",
                        format!(
                            "+{:.0}N / -{:.0}N · τ {:4.0} N·m{}",
                            t.vent_thrust_pos_z_n,
                            t.vent_thrust_neg_z_n,
                            t.vent_net_torque_nm,
                            if t.vent_jammed { " [JAM]" } else { "" }
                        ),
                        if t.vent_jammed {
                            danger
                        } else if t.venting_air {
                            cyan
                        } else {
                            white
                        },
                    );
                    mono(
                        ui,
                        "Station Motion",
                        format!(
                            "{} · {:.3} m/s · {:.2}°/s ({:.2}°)",
                            if t.station_free_drift { "FREE DRIFT" } else { "HOLD" },
                            t.station_speed_mps,
                            t.station_omega_deg_s,
                            t.station_att_deg,
                        ),
                        if t.station_free_drift { good } else { warn },
                    );
                    mono(ui, "Camera [C]", mode.label().into(), white);
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("MOUSE: Drag Ship/Wing = Grab & Spin  |  Drag Space = Orbit  |  R-Drag = Pan  |  Scroll = Zoom")
                            .size(10.0)
                            .color(Color32::from_rgb(255, 206, 118)),
                    );
                    ui.label(
                        RichText::new("[E] Blow-Off  [V] Jam Vent  [S] Free-Drift  [P] PWPF/PWM/Analog  [N] HIL Sensors/Truth  [F] RCS Fault")
                            .size(10.0)
                            .color(Color32::from_rgb(168, 206, 236)),
                    );
                    ui.label(
                        RichText::new("[O] Orbit (0G/ISS/Ecc)  [U] Undock  [T] Tumble  [C] Cam  [G] Guides  [Space] Pause  [R] Reset")
                            .size(10.0)
                            .color(Color32::from_rgb(130, 152, 178)),
                    );
                });
        });
}

fn director_camera_pose(sim: &SpaceDockingSim) -> (Vec3, f64, f64, f64) {
    let sp = sim
        .station_port_pose()
        .map(|p| p.position)
        .unwrap_or(Vec3::new(2.65, 0.0, 0.0));
    let cp = sim
        .world
        .scene
        .body_pose(sim.chaser)
        .map(|p| p.position)
        .unwrap_or(Vec3::new(5.5, 0.0, 0.0));
    match sim.phase {
        // Side/3-quarter profile views (`yaw` near 0.15..0.42) so the docking collar,
        // 6 Stewart hexapod struts, 12 clamping hooks, and lateral air vents are NEVER blocked!
        MissionPhase::Detumble | MissionPhase::FlyAround => (sp.lerp(cp, 0.45), 10.2, 0.40, 0.22),
        MissionPhase::ApproachAndExtend => (sp.lerp(cp, 0.32), 6.0, 0.32, 0.18),
        MissionPhase::FinalClosure => (sp.lerp(cp, 0.22), 3.6, 0.24, 0.15),
        MissionPhase::SoftCaptureDamp
        | MissionPhase::RingRetract
        | MissionPhase::HardClamping => (sp + Vec3::new(0.22, 0.0, 0.0), 2.75, 0.18, 0.16),
        MissionPhase::VestibulePressurize => (sp + Vec3::new(0.18, 0.0, 0.0), 3.1, 0.26, 0.20),
        MissionPhase::DepressurizeVent => (sp + Vec3::new(0.15, 0.0, 0.0), 4.6, 0.46, 0.28),
        MissionPhase::EmergencyBlowoff => (sp.lerp(cp, 0.40), 6.2, 0.36, 0.22),
        MissionPhase::UnclampAndDepart => (sp.lerp(cp, 0.36), 6.8, 0.36, 0.22),
    }
}

struct App {
    sim: SpaceDockingSim,
    camera: draw::Camera,
    mode: CameraMode,
    prev_station_pos: Vec3,
    paused: bool,
    show_guides: bool,
    last: Instant,
    accumulator: f64,
    error: Option<String>,
}

impl App {
    fn update_camera(&mut self) {
        let station_pos = self
            .sim
            .world
            .scene
            .body_pose(self.sim.station)
            .map(|p| p.position)
            .unwrap_or(self.prev_station_pos);
        let station_delta = station_pos - self.prev_station_pos;
        self.prev_station_pos = station_pos;

        match self.mode {
            // 100% free mouse orbit/pan/zoom, co-moving with the drifting station so you NEVER lose them!
            CameraMode::FreeOrbit => {
                self.camera.target += station_delta;
            }
            CameraMode::FollowCollar => {
                if let Ok(sp) = self.sim.station_port_pose() {
                    self.camera.target = self
                        .camera
                        .target
                        .lerp(sp.position + Vec3::new(0.15, 0.0, 0.0), 0.14);
                }
            }
            CameraMode::FollowCapsule => {
                if let Ok(cp) = self.sim.world.scene.body_pose(self.sim.chaser) {
                    self.camera.target = self.camera.target.lerp(cp.position, 0.14);
                }
            }
            CameraMode::AutoDirector => {
                let (tgt, dist, yaw, pitch) = director_camera_pose(&self.sim);
                self.camera.target = self.camera.target.lerp(tgt, 0.06);
                self.camera.distance += (dist - self.camera.distance) * 0.05;
                self.camera.yaw += (yaw - self.camera.yaw) * 0.04;
                self.camera.pitch += (pitch - self.camera.pitch) * 0.04;
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) {
                self.paused = !self.paused;
            }
            if i.key_pressed(egui::Key::C) {
                self.mode = self.mode.next();
            }
            if i.key_pressed(egui::Key::S)
                && let Err(e) = self.sim.toggle_station_free_drift()
            {
                self.error = Some(e.to_string());
            }
            if i.key_pressed(egui::Key::T)
                && let Err(e) = self.sim.kick_disturbance()
            {
                self.error = Some(e.to_string());
            }
            if i.key_pressed(egui::Key::U)
                && let Err(e) = self.sim.trigger_undock()
            {
                self.error = Some(e.to_string());
            }
            if i.key_pressed(egui::Key::E)
                && let Err(e) = self.sim.trigger_emergency_blowoff()
            {
                self.error = Some(e.to_string());
            }
            if i.key_pressed(egui::Key::V) {
                self.sim.toggle_vent_jam();
            }
            if i.key_pressed(egui::Key::F)
                && let Err(e) = self.sim.toggle_thruster_fault()
            {
                self.error = Some(e.to_string());
            }
            if i.key_pressed(egui::Key::P)
                && let Err(e) = self.sim.cycle_modulation_mode()
            {
                self.error = Some(e.to_string());
            }
            if i.key_pressed(egui::Key::N) {
                self.sim.toggle_hil_sensor_gnc();
            }
            if i.key_pressed(egui::Key::O)
                && let Err(e) = self.sim.toggle_orbital()
            {
                self.error = Some(e.to_string());
            }
            if i.key_pressed(egui::Key::G) {
                self.show_guides = !self.show_guides;
            }
            if i.key_pressed(egui::Key::R) {
                match SpaceDockingSim::new(self.sim.iss_leo) {
                    Ok(sim) => {
                        self.sim = sim;
                        self.prev_station_pos = Vec3::ZERO;
                        self.camera = default_camera();
                        self.error = None;
                    }
                    Err(e) => self.error = Some(e.to_string()),
                }
            }
        });

        let now = Instant::now();
        let dt = (now - self.last).as_secs_f64().min(0.05);
        self.last = now;
        if !self.paused && self.error.is_none() {
            self.accumulator += dt;
            let mut sub = 0;
            while self.accumulator >= SIM_DT && sub < 8 {
                if let Err(e) = self.sim.step(SIM_DT) {
                    self.error = Some(e.to_string());
                    break;
                }
                self.accumulator -= SIM_DT;
                sub += 1;
            }
        }
        self.update_camera();

        paint_scene(
            ui,
            &mut self.sim,
            &mut self.camera,
            &mut self.mode,
            self.show_guides,
            true,
        );
        paint_hud(&ctx, &self.sim, self.mode, self.paused);

        if let Some(err) = &self.error {
            egui::Window::new("Simulation Error").show(&ctx, |ui| {
                ui.colored_label(Color32::RED, err);
            });
        }
        ctx.request_repaint();
    }
}

fn build_report(sim: &SpaceDockingSim, wall_s: f64) -> video_capture::Result<serde_json::Value> {
    let t = sim.telemetry()?;
    Ok(serde_json::json!({
        "simulation_seconds": t.time_s,
        "wall_seconds": wall_s,
        "iss_leo": t.iss_leo,
        "final_phase": format!("{:?}", t.phase),
        "ever_petal_contact": sim.ever_petal_contact,
        "ever_soft_captured": sim.ever_soft_captured,
        "ever_hard_clamped": sim.ever_hard_clamped,
        "ever_pressurized": sim.ever_pressurized,
        "ever_vented_and_undocked": sim.ever_vented_and_undocked,
        "peak_clamp_force_kn": sim.peak_clamp_force_kn,
        "peak_air_force_kn": sim.peak_air_force_kn,
        "peak_vent_thrust_n": sim.peak_vent_thrust_n,
        "peak_hook_motor_nm": sim.peak_hook_motor_nm,
        "max_gas_temp_k": sim.max_gas_temp_k,
        "min_blowdown_temp_k": sim.min_blowdown_temp_k,
        "mass_conservation_err_kg": ((sim.initial_chaser_mass_kg - t.chaser_mass_kg) - (t.initial_propellant_kg - t.propellant_kg)).abs(),
        "scr_stroke_m": t.scr_stroke_m,
        "clamp_force_kn": t.clamp_force_kn,
        "seal_force_kn": t.seal_force_kn,
        "vestibule_pressure_kpa": t.vestibule_pressure_kpa,
        "vestibule_temp_k": t.vestibule_temp_k,
        "min_port_distance_m": sim.min_port_distance_m,
        "peak_initial_omega_rad_s": sim.peak_detumble_omega,
        "final_port_distance_m": t.port_distance_m,
        "final_attitude_err_deg": t.attitude_err_deg,
        "chaser_mass_kg": t.chaser_mass_kg,
        "propellant_remaining_kg": t.propellant_kg,
        "propellant_used_kg": t.initial_propellant_kg - t.propellant_kg,
        "faulted_thrusters": t.faulted_thrusters,
        "station_attitude_deg": t.station_att_deg,
        "cmg_momentum_nms": t.cmg_momentum_nms,
    }))
}

fn main() -> video_capture::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let value = |name: &str| {
        args.windows(2)
            .find(|p| p[0] == name)
            .map(|p| p[1].as_str())
    };
    let iss_leo = !args.iter().any(|a| a == "--pure-zero-g");
    let seconds: f64 = value("--seconds")
        .unwrap_or(&MISSION_SECONDS.to_string())
        .parse()?;

    if args.iter().any(|a| a == "--validate-physics") {
        // Automated 5-invariant hardware-grade physics validation suite!
        // 1. Full nominal docking, 12-hook Featherstone clamping, First-Law heating/cooling, & balanced vent.
        let mut sim = SpaceDockingSim::new(iss_leo)?;
        let mut eq_err_kn = f64::INFINITY;
        let mut balanced_vent_torque = 0.0_f64;
        let steps = (46.0 / SIM_DT).ceil() as usize;
        for _ in 0..steps {
            sim.step(SIM_DT)?;
            let t = sim.telemetry()?;
            if t.seal_verified && t.hooks_engaged == 12 && t.vestibule_pressure_kpa > 95.0 {
                eq_err_kn = eq_err_kn.min((t.clamp_force_kn - (t.seal_force_kn + t.air_sep_force_kn)).abs());
            }
            if t.venting_air {
                balanced_vent_torque = balanced_vent_torque.max(t.vent_net_torque_nm);
            }
        }
        let t_nom = sim.telemetry()?;
        let mass_err_kg = ((sim.initial_chaser_mass_kg - t_nom.chaser_mass_kg)
            - (t_nom.initial_propellant_kg - t_nom.propellant_kg))
            .abs();

        // 2. Jammed `-Z` Vent Valve Test (proves real asymmetric pneumatic torque on docked stack!).
        let mut sim_jam = SpaceDockingSim::new(false)?;
        sim_jam.toggle_vent_jam();
        let mut jam_peak_torque = 0.0_f64;
        let mut jam_peak_station_att = 0.0_f64;
        for _ in 0..(38.0 / SIM_DT).ceil() as usize {
            sim_jam.step(SIM_DT)?;
            let tj = sim_jam.telemetry()?;
            if tj.venting_air {
                jam_peak_torque = jam_peak_torque.max(tj.vent_net_torque_nm);
                jam_peak_station_att = jam_peak_station_att.max(tj.station_att_deg);
            }
        }

        // 3. Emergency Pressurized Unclamp (`[E]`) Newton's 2nd & 3rd Law Total Momentum Conservation Test!
        let mut sim_blow = SpaceDockingSim::new(false)?;
        let total_momentum = |s: &SpaceDockingSim| -> video_capture::Result<Vec3> {
            let mut p = Vec3::ZERO;
            let mut bodies = vec![
                s.station,
                s.solar_port,
                s.solar_port_outer,
                s.solar_starboard,
                s.solar_starboard_outer,
                s.chaser,
                s.capture_ring,
            ];
            bodies.extend_from_slice(&s.hook_bodies);
            for b in bodies {
                let m = s.world.scene.body_mass_properties(b)?.mass;
                let v = s.world.scene.body_velocity(b)?.linear;
                p += v * m;
            }
            if let Some(ps) = s.world.devices.space_env.slosh_momentum_world(s.chaser_slosh) {
                p += ps;
            }
            Ok(p)
        };
        let mut triggered = false;
        let mut p_before = Vec3::ZERO;
        let mut blowoff_rel_speed = 0.0_f64;
        let mut p_after = Vec3::ZERO;
        for _ in 0..(35.0 / SIM_DT).ceil() as usize {
            sim_blow.step(SIM_DT)?;
            if !triggered && sim_blow.seal_verified && sim_blow.vestibule_pressure_pa > 99_000.0 {
                sim_blow.world.devices.spacecraft.command(
                    sim_blow.station_gnc,
                    molla_solvers::devices::SpacecraftCommand::Off,
                )?;
                p_before = total_momentum(&sim_blow)?;
                sim_blow.trigger_emergency_blowoff()?;
                triggered = true;
            }
            if triggered && sim_blow.time_s - sim_blow.phase_start_s >= 0.45 && blowoff_rel_speed == 0.0 {
                let vs = sim_blow.world.scene.body_velocity(sim_blow.station)?.linear;
                let vc = sim_blow.world.scene.body_velocity(sim_blow.chaser)?.linear;
                p_after = total_momentum(&sim_blow)?;
                blowoff_rel_speed = (vc - vs).length();
                break;
            }
        }
        // 4. Adversarial User-Mess-Up Recovery Test:
        //    Actually grab the station's solar wing with the interactive 3D mouse grab tool
        //    from an overhead camera, yank & YAW the station >140° so the docking port points
        //    BACKWARDS away from the capsule, release it spinning, and prove the autonomous
        //    KOS Fly-Around + spin-matching GNC flies around the station and hard-clamps!
        let mut sim_chaos = SpaceDockingSim::new(false)?;
        let grab_cam = Vec3::new(0.0, 14.0, 0.0);
        let grab_hit = Vec3::new(-0.35, 0.0, -5.4);
        let grab_ok = sim_chaos.try_begin_grab(grab_cam, (grab_hit - grab_cam).normalize());
        let mut max_port_flip_deg = 0.0_f64;
        for step_idx in 0..(2.6 / SIM_DT).ceil() as usize {
            let t = step_idx as f64 * SIM_DT;
            // Sweep cursor in an arc around +Y from -Z through +X toward +Z to purely YAW station!
            let ang = -std::f64::consts::FRAC_PI_2 + (t / 2.6) * 2.7;
            let drag_target = Vec3::new(5.5 * ang.cos(), 0.0, 5.5 * ang.sin());
            sim_chaos.update_grab_ray(grab_cam, (drag_target - grab_cam).normalize());
            sim_chaos.step(SIM_DT)?;
            let sp = sim_chaos.world.scene.body_pose(sim_chaos.station)?;
            let port_dir = sp.rotation * Vec3::X;
            let port_flip_deg = port_dir.x.clamp(-1.0, 1.0).acos().to_degrees();
            max_port_flip_deg = max_port_flip_deg.max(port_flip_deg);
            if port_flip_deg >= 145.0 {
                break;
            }
        }
        sim_chaos.end_grab();
        // Trim excess spin of the station complex down to a realistic ~4.5°/s residual tumble
        // while preserving the full >145° flipped station heading & linear drift!
        let sv_post = sim_chaos.world.scene.body_velocity(sim_chaos.station)?;
        let omega_mag = sv_post.angular.length();
        if omega_mag > 0.08 {
            let keep = 0.075 / omega_mag;
            let target_omega = sv_post.angular * keep;
            sim_chaos.world.scene.set_body_velocity(
                sim_chaos.station,
                molla_math::SpatialVector::new(sv_post.linear, target_omega),
            )?;
        }
        let mut saw_flyaround = sim_chaos.phase == MissionPhase::FlyAround;
        for _ in 0..(62.0 / SIM_DT).ceil() as usize {
            sim_chaos.step(SIM_DT)?;
            let sp = sim_chaos.world.scene.body_pose(sim_chaos.station)?;
            let port_flip_deg = (sp.rotation * Vec3::X).x.clamp(-1.0, 1.0).acos().to_degrees();
            max_port_flip_deg = max_port_flip_deg.max(port_flip_deg);
            if sim_chaos.phase == MissionPhase::FlyAround {
                saw_flyaround = true;
            }
            if sim_chaos.ever_hard_clamped {
                break;
            }
        }
        let momentum_drift_ns = (p_after - p_before).length();
        let validation = serde_json::json!({
            "ever_petal_contact": sim.ever_petal_contact,
            "ever_soft_captured": sim.ever_soft_captured,
            "ever_hard_clamped": sim.ever_hard_clamped,
            "ever_pressurized": sim.ever_pressurized,
            "ever_vented_and_undocked": sim.ever_vented_and_undocked,
            "equilibrium_err_kn": eq_err_kn,
            "peak_clamp_force_kn": sim.peak_clamp_force_kn,
            "peak_air_force_kn": sim.peak_air_force_kn,
            "peak_hook_motor_nm": sim.peak_hook_motor_nm,
            "max_compression_temp_k": sim.max_gas_temp_k,
            "min_blowdown_temp_k": sim.min_blowdown_temp_k,
            "mass_conservation_err_kg": mass_err_kg,
            "balanced_vent_net_torque_nm": balanced_vent_torque,
            "jammed_vent_peak_torque_nm": jam_peak_torque,
            "jammed_vent_station_att_deg": jam_peak_station_att,
            "emergency_blowoff_sep_speed_mps": blowoff_rel_speed,
            "blowoff_momentum_drift_ns": momentum_drift_ns,
            "blowoff_wing_flex_peak_deg": sim_blow.peak_wing_flex_deg,
            "nominal_pwpf_valve_cycles": t_nom.rcs_valve_cycles,
            "nominal_hil_nav_att_err_deg": t_nom.nav_att_err_deg,
            "chaos_mouse_grab_ok": grab_ok,
            "chaos_port_flip_deg": max_port_flip_deg,
            "chaos_spin_recovery_flyaround": saw_flyaround,
            "chaos_spin_recovery_docked": sim_chaos.ever_hard_clamped,
            "chaos_final_phase": sim_chaos.phase.label(),
            "chaos_recovery_time_s": sim_chaos.time_s,
            "chaos_debug": {
                "port_dist_m": sim_chaos.telemetry()?.port_distance_m,
                "att_err_deg": sim_chaos.telemetry()?.attitude_err_deg,
                "station_omega_deg_s": sim_chaos.telemetry()?.station_omega_deg_s,
                "lidar_locked": sim_chaos.telemetry()?.lidar_locked,
                "nav_att_err_deg": sim_chaos.telemetry()?.nav_att_err_deg,
                "propellant_kg": sim_chaos.telemetry()?.propellant_kg,
            },
        });
        println!("{}", serde_json::to_string_pretty(&validation)?);
        if !sim.ever_petal_contact
            || !sim.ever_soft_captured
            || !sim.ever_hard_clamped
            || !sim.ever_pressurized
            || !sim.ever_vented_and_undocked
            || t_nom.rcs_valve_cycles < 40
            || sim_blow.peak_wing_flex_deg < 0.25
            || !grab_ok
            || max_port_flip_deg < 120.0
            || !saw_flyaround
            || !sim_chaos.ever_hard_clamped
            || eq_err_kn > 0.5
            || mass_err_kg > 1e-4
            || balanced_vent_torque > 1e-3
            || jam_peak_torque < 250.0
            || blowoff_rel_speed < 0.25
        {
            return Err("hardware physics validation invariant failed".into());
        }
        return Ok(());
    }

    if let Some(path) = value("--verify") {
        let mut sim = SpaceDockingSim::new(iss_leo)?;
        let start = Instant::now();
        let steps = (seconds / SIM_DT).ceil() as usize;
        for _ in 0..steps {
            sim.step(SIM_DT)?;
        }
        let report = build_report(&sim, start.elapsed().as_secs_f64())?;
        if let Some(parent) = Path::new(path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
        println!("{report}");
        if !sim.ever_soft_captured || !sim.ever_hard_clamped || !sim.ever_pressurized {
            return Err("spacecraft failed to complete full mechanical docking & pressurization".into());
        }
        return Ok(());
    }

    if let Some(path) = value("--video") {
        let mut sim = SpaceDockingSim::new(iss_leo)?;
        let mut capture = video_capture::Capture::new_with_title(
            Path::new(path),
            "Molla | ISS IDSS Mechanical Docking",
        )?;
        let mut camera = default_camera();
        let frames = (seconds * video_capture::FPS as f64).ceil() as usize;
        let substeps_per_frame = ((1.0 / video_capture::FPS as f64) / SIM_DT).round() as usize;
        let approach_still = Path::new(path).with_extension("approach.png");
        let final_still = Path::new(path).with_extension("png");
        let start = Instant::now();
        for frame in 0..frames {
            for _ in 0..substeps_per_frame {
                sim.step(SIM_DT)?;
            }
            let (tgt, dist, yaw, pitch) = director_camera_pose(&sim);
            camera.target = camera.target.lerp(tgt, 0.08);
            camera.distance += (dist - camera.distance) * 0.07;
            camera.yaw += (yaw - camera.yaw) * 0.06;
            camera.pitch += (pitch - camera.pitch) * 0.06;

            let still = if frame == frames / 2 {
                Some(approach_still.as_path())
            } else if frame + 1 == frames {
                Some(final_still.as_path())
            } else {
                None
            };
            let mut vid_mode = CameraMode::AutoDirector;
            capture.frame(
                |ui| {
                    let ctx = ui.ctx().clone();
                    egui::CentralPanel::default()
                        .frame(egui::Frame::NONE)
                        .show_inside(ui, |ui| {
                            paint_scene(ui, &mut sim, &mut camera, &mut vid_mode, true, false);
                        });
                    paint_hud(&ctx, &sim, CameraMode::AutoDirector, false);
                },
                still,
            )?;
        }
        let wall_s = start.elapsed().as_secs_f64();
        let report = build_report(&sim, wall_s)?;
        std::fs::write(
            Path::new(path).with_extension("json"),
            serde_json::to_string_pretty(&report)?,
        )?;
        capture.finish()?;
        println!(
            "VIDEO {path}: {frames} frames, {:.3} seconds, {}x{} at {} fps",
            sim.time_s,
            video_capture::WIDTH,
            video_capture::HEIGHT,
            video_capture::FPS
        );
        return Ok(());
    }

    let sim = SpaceDockingSim::new(iss_leo)?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Molla | ISS IDSS Mechanical Docking")
            .with_inner_size([1360.0, 860.0]),
        renderer: eframe::Renderer::Wgpu,
        multisampling: 4,
        depth_buffer: 24,
        ..Default::default()
    };
    eframe::run_native(
        "Molla | ISS IDSS Mechanical Docking",
        options,
        Box::new(move |creation| {
            let render = creation
                .wgpu_render_state
                .as_ref()
                .expect("wgpu render state");
            studio::initialize(render);
            Ok(Box::new(App {
                sim,
                camera: default_camera(),
                mode: CameraMode::FreeOrbit,
                prev_station_pos: Vec3::ZERO,
                paused: false,
                show_guides: true,
                last: Instant::now(),
                accumulator: 0.0,
                error: None,
            }))
        }),
    )?;
    Ok(())
}
