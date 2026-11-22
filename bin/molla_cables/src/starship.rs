//! Full-Scale 2-Stage SpaceX Starship + Super Heavy + Mechazilla Physics Testbed (`molla_starship`).
//!
//! 100% First-Principles Multi-Engine Physics:
//! - 33-Raptor Super Heavy Booster (13 TVC ±15° + TWD + 20 fixed) + 4 Titanium Grid Fins + SRP
//! - 6-Raptor Starship Upper Stage (3 SL TVC + 3 RVac Schmucker separation) + 4 Body Flaps
//! - Multi-Tank Shifting COM / 3×3 Inertia + NASA SP-8009 Acceleration-Driven Pendulum Slosh
//! - Articulated Featherstone Mechazilla Chopsticks Catch + Hot-Staging Blast Impingement!

#[allow(dead_code)]
mod draw;
#[allow(dead_code)]
mod foliage_layout;
#[allow(dead_code)]
mod scene;
#[allow(dead_code)]
mod starship_mesh;
#[allow(dead_code)]
mod starship_sim;
#[allow(dead_code)]
mod studio;
#[allow(dead_code)]
mod video_capture;

use std::time::Instant;

use eframe::egui::{self, Color32, CornerRadius, Margin, RichText, Stroke};
use molla_math::Vec3;
use starship_mesh::RENDER_SCALE;
use starship_sim::{
    BoosterPhase, SIM_DT, ShipPhase, StarshipScenario, StarshipSim,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum FocusTarget {
    ActiveVehicle,
    Booster,
    Ship,
    MechazillaTower,
}

impl FocusTarget {
    fn next(self) -> Self {
        match self {
            Self::ActiveVehicle => Self::Booster,
            Self::Booster => Self::Ship,
            Self::Ship => Self::MechazillaTower,
            Self::MechazillaTower => Self::ActiveVehicle,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ActiveVehicle => "Auto Active Vehicle",
            Self::Booster => "Super Heavy Booster",
            Self::Ship => "Starship Upper Stage",
            Self::MechazillaTower => "Mechazilla Tower & Pad",
        }
    }
}

fn default_camera(scenario: StarshipScenario) -> draw::Camera {
    match scenario {
        StarshipScenario::FullStackMission => draw::Camera {
            target: Vec3::ZERO,
            distance: 9.8, // ~245 m in SI
            yaw: 0.55,
            pitch: 0.16,
        },
        StarshipScenario::HotStaging => draw::Camera {
            target: Vec3::ZERO,
            distance: 7.2,
            yaw: 0.42,
            pitch: 0.12,
        },
        StarshipScenario::BoosterCatch => draw::Camera {
            target: Vec3::ZERO,
            distance: 9.2,
            yaw: 0.52,
            pitch: 0.14,
        },
        StarshipScenario::ShipFlipLanding
        | StarshipScenario::MarsSrpLanding
        | StarshipScenario::LunarHlsDescent => draw::Camera {
            target: Vec3::ZERO,
            distance: 7.5,
            yaw: 0.35,
            pitch: 0.12,
        },
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SplitMode {
    Auto,
    Split,
    Single,
}

impl SplitMode {
    fn next(self) -> Self {
        match self {
            Self::Auto => Self::Split,
            Self::Split => Self::Single,
            Self::Single => Self::Auto,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto Split",
            Self::Split => "Split On",
            Self::Single => "Single View",
        }
    }
}

struct App {
    sim: StarshipSim,
    camera: draw::Camera,
    camera_ship: draw::Camera,
    focus: FocusTarget,
    split_mode: SplitMode,
    was_split: bool,
    render_origin: Vec3,
    paused: bool,
    time_scale: f64,
    last: Instant,
    accumulator: f64,
    error: Option<String>,
}

impl App {
    fn booster_world_pos(&self) -> Vec3 {
        self.sim
            .world
            .scene
            .body_pose(self.sim.booster)
            .map(|p| p.position)
            .unwrap_or(Vec3::new(0.0, 55.0, 0.0))
    }

    fn ship_world_pos(&self) -> Vec3 {
        self.sim
            .world
            .scene
            .body_pose(self.sim.ship)
            .map(|p| p.position)
            .unwrap_or(Vec3::new(0.0, 115.0, 0.0))
    }

    fn is_split_active(&self) -> bool {
        match self.split_mode {
            SplitMode::Split => true,
            SplitMode::Single => false,
            SplitMode::Auto => {
                matches!(
                    self.sim.scenario,
                    StarshipScenario::FullStackMission | StarshipScenario::HotStaging
                ) && !self.sim.interstage_latched
            }
        }
    }

    fn focus_world_pos(&self) -> Vec3 {
        let bp = self.booster_world_pos();
        let sp = self.ship_world_pos();
        match self.focus {
            FocusTarget::ActiveVehicle => match self.sim.scenario {
                StarshipScenario::ShipFlipLanding
                | StarshipScenario::MarsSrpLanding
                | StarshipScenario::LunarHlsDescent => sp,
                StarshipScenario::BoosterCatch => bp,
                StarshipScenario::FullStackMission | StarshipScenario::HotStaging => {
                    if self.sim.interstage_latched {
                        bp.lerp(sp, 0.45)
                    } else if self.sim.booster_caught && !self.sim.ship_landed {
                        sp
                    } else {
                        bp
                    }
                }
            },
            FocusTarget::Booster => bp,
            FocusTarget::Ship => sp,
            FocusTarget::MechazillaTower => Vec3::new(0.0, 72.0, 0.0),
        }
    }

    fn load_scenario(&mut self, scenario: StarshipScenario) {
        match StarshipSim::new(scenario) {
            Ok(sim) => {
                self.sim = sim;
                self.focus = FocusTarget::ActiveVehicle;
                self.camera = default_camera(scenario);
                self.camera_ship = draw::Camera {
                    target: Vec3::ZERO,
                    distance: 7.4,
                    yaw: 0.48,
                    pitch: 0.14,
                };
                self.was_split = false;
                self.render_origin = self.focus_world_pos();
                self.accumulator = 0.0;
                self.error = None;
            }
            Err(e) => self.error = Some(e.to_string()),
        }
    }
}

fn paint_viewport(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    id_salt: &str,
    sim: &mut StarshipSim,
    camera: &mut draw::Camera,
    render_origin: Vec3,
    slot: usize,
    banner: Option<(&str, Color32)>,
) {
    let sense = egui::Sense::click_and_drag();
    let response = ui.interact(rect, ui.id().with(id_salt), sense);
    let painter = ui.painter().with_clip_rect(rect);

    // Interactive 3D Mouse Nudge on Booster or Ship (left-drag body) or Free Orbit (left-drag sky)!
    if response.drag_started_by(egui::PointerButton::Primary)
        && let Some(pos) = response.interact_pointer_pos()
    {
        let (ro_r, rd) = camera.ray(pos, rect);
        let ro_w = render_origin + ro_r / RENDER_SCALE;
        let _ = sim.try_begin_nudge(ro_w, rd);
    }
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        sim.end_nudge();
    }

    let left_drag = response.dragged_by(egui::PointerButton::Primary);
    let pan_drag = response.dragged_by(egui::PointerButton::Secondary)
        || response.dragged_by(egui::PointerButton::Middle);

    if left_drag && sim.user_nudge.is_some() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (ro_r, rd) = camera.ray(pos, rect);
            let ro_w = render_origin + ro_r / RENDER_SCALE;
            sim.update_nudge_ray(ro_w, rd);
        }
    } else if left_drag || pan_drag {
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
            camera.distance =
                (camera.distance * (-f64::from(scroll) * 0.0022).exp()).clamp(2.0, 38.0);
        }
    }

    starship_mesh::paint_sky_backdrop(&painter, rect, camera, render_origin.y, sim.atmosphere);
    if let Ok(tris) = starship_mesh::build_scene_mesh(sim, render_origin) {
        studio::paint_space_slot(&painter, camera, rect, tris, slot);
    }
    let puffs = starship_mesh::build_plume_billboards(sim, render_origin);
    if !puffs.is_empty() {
        studio::paint_plume_slot(&painter, camera, rect, &puffs, slot);
    }
    starship_mesh::paint_overlays(&painter, camera, rect, sim, render_origin);

    if let Some((text, col)) = banner {
        let badge_w = (rect.width() - 28.0).min(460.0);
        let badge = egui::Rect::from_center_size(
            egui::pos2(rect.center().x, rect.top() + 22.0),
            egui::vec2(badge_w, 26.0),
        );
        painter.rect_filled(badge, 6.0, Color32::from_rgba_unmultiplied(8, 12, 22, 215));
        painter.rect_stroke(
            badge,
            6.0,
            Stroke::new(1.0_f32, col),
            egui::StrokeKind::Inside,
        );
        painter.text(
            badge.center(),
            egui::Align2::CENTER_CENTER,
            text,
            egui::FontId::monospace(12.0),
            col,
        );
    }
}

fn paint_hud(
    ctx: &egui::Context,
    sim: &StarshipSim,
    focus: FocusTarget,
    split_mode: SplitMode,
    split_active: bool,
    paused: bool,
    time_scale: f64,
) {
    let Ok(t) = sim.telemetry() else {
        return;
    };
    egui::Area::new(egui::Id::new("starship_hud"))
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(14.0, -14.0))
        .show(ctx, |ui| {
            egui::Frame::NONE
                .fill(Color32::from_rgba_unmultiplied(8, 12, 22, 222))
                .stroke(Stroke::new(1.0_f32, Color32::from_rgb(56, 88, 132)))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.set_min_width(410.0);
                    ui.label(
                        RichText::new("MOLLA · FULL-SCALE STARSHIP + SUPER HEAVY (39 RAPTORS)")
                            .strong()
                            .size(13.0)
                            .color(Color32::from_rgb(160, 224, 255)),
                    );
                    let mono = |ui: &mut egui::Ui, k: &str, v: String, col: Color32| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(k).color(Color32::from_rgb(152, 174, 200)));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(RichText::new(v).monospace().color(col));
                                },
                            );
                        });
                    };
                    let white = Color32::from_rgb(236, 244, 255);
                    let amber = Color32::from_rgb(255, 198, 92);
                    let good = Color32::from_rgb(96, 255, 164);
                    let cyan = Color32::from_rgb(142, 228, 255);
                    let red = Color32::from_rgb(255, 98, 98);

                    mono(
                        ui,
                        "Scenario",
                        format!(
                            "{} · T+{:5.1}s ({:.1}x){}",
                            t.scenario.label(),
                            t.time_s,
                            time_scale,
                            if paused { " [PAUSED]" } else { "" }
                        ),
                        amber,
                    );
                    ui.separator();
                    ui.label(
                        RichText::new(format!("SUPER HEAVY BOOSTER · {}", t.booster_phase.label()))
                            .strong()
                            .color(if t.booster_phase == BoosterPhase::CaughtInChopsticks {
                                good
                            } else {
                                amber
                            }),
                    );
                    mono(
                        ui,
                        "Alt / Vy / Mach",
                        format!("{:6.0} m · {:+6.1} m/s · M {:4.2} (q {:4.1} kPa)", t.booster_alt_m, t.booster_vy_mps, t.booster_mach, t.booster_q_kpa),
                        white,
                    );
                    mono(
                        ui,
                        "Raptors (33) / Thrust",
                        format!("{:2}/33 lit · {:5.2} MN · TWD {:5.0} kN·m", t.booster_active_engines, t.booster_thrust_mn, t.booster_twd_knm),
                        if t.booster_active_engines > 0 { cyan } else { white },
                    );
                    mono(
                        ui,
                        "Mass / COM / Slosh",
                        format!("{:6.1} t ({:5.1}t prop) · COM {:+5.1}m · {:4.2}Hz", t.booster_mass_t, t.booster_prop_t, t.booster_com_y_m, t.booster_slosh_hz),
                        white,
                    );
                    mono(
                        ui,
                        "Tower / Tilt / SRP",
                        format!("dx {:5.1}m · {:4.1}° · SRP {:4.2} · Arms {:3.0}%", t.booster_dist_tower_m, t.booster_tilt_deg, t.booster_srp_factor, t.chopsticks_closed_frac * 100.0),
                        if t.booster_phase == BoosterPhase::CaughtInChopsticks { good } else { white },
                    );
                    mono(
                        ui,
                        "Flex / Pogo / PIFS",
                        format!("{:4.2}Hz ({:4.1}cm) · Pogo {:4.2}g · PIFS {:+4.1}m · {:3.1}bar", t.booster_mode1_hz, t.booster_tip_cm, t.booster_pogo_g, t.booster_pifs_m, t.booster_ullage_bar),
                        cyan,
                    );
                    ui.separator();
                    ui.label(
                        RichText::new(format!("STARSHIP UPPER STAGE · {}", t.ship_phase.label()))
                            .strong()
                            .color(if t.ship_phase == ShipPhase::LandedOnPad {
                                good
                            } else {
                                cyan
                            }),
                    );
                    mono(
                        ui,
                        "Alt / Vy / AoA",
                        format!("{:6.0} m · {:+6.1} m/s · AoA {:4.1}° (tilt {:4.1}°)", t.ship_alt_m, t.ship_vy_mps, t.ship_aoa_deg, t.ship_tilt_deg),
                        white,
                    );
                    mono(
                        ui,
                        "Raptors (3SL+3RVac)",
                        format!("{}/6 lit · {:5.2} MN · Hdr {:4.1}t ({:3.1}bar){}", t.ship_active_engines, t.ship_thrust_mn, t.ship_header_t, t.ship_ullage_bar, if t.ship_rvac_separated { " [SEP]" } else { "" }),
                        if t.ship_rvac_separated { red } else if t.ship_active_engines > 0 { cyan } else { white },
                    );
                    mono(
                        ui,
                        "4 Flaps / Flex / Pad",
                        format!("F[{:2.0},{:2.0}] A[{:2.0},{:2.0}]° · {:3.1}Hz({:3.1}cm) · dx {:4.1}m", t.ship_flap_deg[0], t.ship_flap_deg[1], t.ship_flap_deg[2], t.ship_flap_deg[3], t.ship_mode1_hz, t.ship_tip_cm, t.ship_dist_pad_m),
                        white,
                    );
                    mono(
                        ui,
                        "TVC / Gimbal / View",
                        format!(
                            "{} · B:{:4.1}° S:{:4.1}° · {} · {} [{}]",
                            if t.tvc_enabled { "TVC ON" } else { "TVC LOCKED 0°!" },
                            t.booster_gimbal_deg,
                            t.ship_gimbal_deg,
                            t.wind_preset.label(),
                            if split_active { "SPLIT" } else { focus.label() },
                            split_mode.label()
                        ),
                        if t.tvc_enabled { white } else { red },
                    );
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("[1] Stack  [2] Hot-Stage  [3] Catch  [4] Earth Flip  [5] Mars SRP  [6] Lunar HLS")
                            .size(10.0)
                            .color(Color32::from_rgb(255, 210, 120)),
                    );
                    ui.label(
                        RichText::new("[T] TVC On/Off (0° Open-Loop!)  [F] Engine-Out  [S] Split  [V] Focus  [W] Wind  [+/-] Speed  [R] Reset")
                            .size(10.0)
                            .color(Color32::from_rgb(162, 200, 232)),
                    );
                    ui.label(
                        RichText::new("MOUSE: Drag Rocket = 3D Nudge  |  Drag Sky = Free Orbit  |  R-Drag = Pan  |  Scroll = Zoom")
                            .size(10.0)
                            .color(Color32::from_rgb(132, 156, 184)),
                    );
                });
        });
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) {
                self.paused = !self.paused;
            }
            if i.key_pressed(egui::Key::Num1) {
                self.load_scenario(StarshipScenario::FullStackMission);
            }
            if i.key_pressed(egui::Key::Num2) {
                self.load_scenario(StarshipScenario::HotStaging);
            }
            if i.key_pressed(egui::Key::Num3) {
                self.load_scenario(StarshipScenario::BoosterCatch);
            }
            if i.key_pressed(egui::Key::Num4) {
                self.load_scenario(StarshipScenario::ShipFlipLanding);
            }
            if i.key_pressed(egui::Key::Num5) {
                self.load_scenario(StarshipScenario::MarsSrpLanding);
            }
            if i.key_pressed(egui::Key::Num6) {
                self.load_scenario(StarshipScenario::LunarHlsDescent);
            }
            if i.key_pressed(egui::Key::V) {
                self.focus = self.focus.next();
                self.camera.target = Vec3::ZERO;
            }
            if i.key_pressed(egui::Key::S) {
                self.split_mode = self.split_mode.next();
            }
            if i.key_pressed(egui::Key::W) {
                self.sim.cycle_wind();
            }
            if i.key_pressed(egui::Key::T) {
                self.sim.tvc_enabled = !self.sim.tvc_enabled;
            }
            if i.key_pressed(egui::Key::F)
                && let Err(e) = self.sim.inject_engine_out()
            {
                self.error = Some(e.to_string());
            }
            if i.key_pressed(egui::Key::Equals) || i.key_pressed(egui::Key::Plus) {
                self.time_scale = (self.time_scale * 1.5).min(4.0);
            }
            if i.key_pressed(egui::Key::Minus) {
                self.time_scale = (self.time_scale / 1.5).max(0.25);
            }
            if i.key_pressed(egui::Key::R) {
                let s = self.sim.scenario;
                self.load_scenario(s);
            }
        });

        let now = Instant::now();
        let dt = (now - self.last).as_secs_f64().min(0.05) * self.time_scale;
        self.last = now;
        if !self.paused && self.error.is_none() {
            self.accumulator += dt;
            let mut sub = 0;
            while self.accumulator >= SIM_DT && sub < 10 {
                if let Err(e) = self.sim.step(SIM_DT) {
                    self.error = Some(e.to_string());
                    break;
                }
                self.accumulator -= SIM_DT;
                sub += 1;
            }
        }

        // Auto split-screen the instant Booster & Ship separate at Hot-Staging (or when forced via [S])!
        let split_active = self.is_split_active();
        if split_active && !self.was_split {
            self.camera_ship.yaw = self.camera.yaw;
            self.camera_ship.pitch = self.camera.pitch;
            self.camera_ship.target = Vec3::ZERO;
            self.camera_ship.distance = (self.camera.distance * 0.85).clamp(5.0, 10.5);
        }
        self.was_split = split_active;

        let full_rect = ui.available_rect_before_wrap();
        ui.allocate_rect(full_rect, egui::Sense::hover());

        if split_active {
            let mid_x = full_rect.center().x;
            let left_rect = egui::Rect::from_min_max(full_rect.min, egui::pos2(mid_x - 1.0, full_rect.max.y));
            let right_rect = egui::Rect::from_min_max(egui::pos2(mid_x + 1.0, full_rect.min.y), full_rect.max);
            let bp = self.booster_world_pos();
            let sp = self.ship_world_pos();
            let (b_label, s_label) = match self.sim.telemetry() {
                Ok(t) => (
                    format!(
                        "SUPER HEAVY · {} · {:.0}m · {}/33",
                        t.booster_phase.label(),
                        t.booster_alt_m,
                        t.booster_active_engines
                    ),
                    format!(
                        "STARSHIP · {} · {:.0}m · {}/6",
                        t.ship_phase.label(),
                        t.ship_alt_m,
                        t.ship_active_engines
                    ),
                ),
                Err(_) => ("SUPER HEAVY BOOSTER".into(), "STARSHIP UPPER STAGE".into()),
            };
            paint_viewport(
                ui,
                left_rect,
                "vp_booster",
                &mut self.sim,
                &mut self.camera,
                bp,
                0,
                Some((&b_label, Color32::from_rgb(255, 198, 92))),
            );
            paint_viewport(
                ui,
                right_rect,
                "vp_ship",
                &mut self.sim,
                &mut self.camera_ship,
                sp,
                1,
                Some((&s_label, Color32::from_rgb(130, 224, 255))),
            );
            // Crisp vertical broadcast split divider between Booster (left) and Ship (right)!
            ui.painter().line_segment(
                [egui::pos2(mid_x, full_rect.top()), egui::pos2(mid_x, full_rect.bottom())],
                Stroke::new(2.0_f32, Color32::from_rgb(68, 98, 140)),
            );
        } else {
            self.render_origin = self.focus_world_pos();
            paint_viewport(
                ui,
                full_rect,
                "vp_single",
                &mut self.sim,
                &mut self.camera,
                self.render_origin,
                0,
                None,
            );
        }
        paint_hud(
            &ctx,
            &self.sim,
            self.focus,
            self.split_mode,
            split_active,
            self.paused,
            self.time_scale,
        );

        if let Some(err) = &self.error {
            egui::Window::new("Simulation Error").show(&ctx, |ui| {
                ui.colored_label(Color32::RED, err);
            });
        }
        ctx.request_repaint();
    }
}

fn run_validation() -> video_capture::Result<()> {
    // 1. Full-Stack 33-Raptor Liftoff -> Max-Q -> Hot-Staging Separation!
    let mut sim_stack = StarshipSim::new(StarshipScenario::FullStackMission)?;
    for _ in 0..(46.0 / SIM_DT).ceil() as usize {
        sim_stack.step(SIM_DT)?;
        if sim_stack.ever_hot_staged && sim_stack.booster_phase == BoosterPhase::FlipAndBoostback {
            break;
        }
    }
    let t_stack = sim_stack.telemetry()?;

    // 2. Super Heavy Booster Grid-Fin Descent -> 13->3 Raptor Mechazilla Chopstick Catch!
    let mut sim_catch = StarshipSim::new(StarshipScenario::BoosterCatch)?;
    for _ in 0..(45.0 / SIM_DT).ceil() as usize {
        sim_catch.step(SIM_DT)?;
        if sim_catch.ever_booster_caught {
            break;
        }
    }
    let t_catch = sim_catch.telemetry()?;

    // 3. Starship 90° Belly-Flop Skydiving (4 Flaps) -> 3->2->1 Raptor Flip-and-Burn Landing!
    let mut sim_ship = StarshipSim::new(StarshipScenario::ShipFlipLanding)?;
    for _ in 0..(80.0 / SIM_DT).ceil() as usize {
        sim_ship.step(SIM_DT)?;
        if sim_ship.ever_ship_landed {
            break;
        }
    }
    let t_ship = sim_ship.telemetry()?;

    // 4. Mars Supersonic Entry (699 Pa CO2, 3.72 m/s²) -> SRP Flip-and-Burn Touchdown!
    let mut sim_mars = StarshipSim::new(StarshipScenario::MarsSrpLanding)?;
    let mut mars_peak_mach = 0.0_f64;
    let mut mars_min_srp = 1.0_f64;
    for _ in 0..(105.0 / SIM_DT).ceil() as usize {
        sim_mars.step(SIM_DT)?;
        if let Some(ar) = sim_mars.world.devices.rocket_aero.body_reading(sim_mars.ship_aero) {
            mars_peak_mach = mars_peak_mach.max(ar.mach);
            mars_min_srp = mars_min_srp.min(ar.srp_drag_factor);
        }
        if sim_mars.ever_ship_landed {
            break;
        }
    }
    let t_mars = sim_mars.telemetry()?;

    // 5. Lunar HLS Powered Descent (0 Pa Vacuum, 1.625 m/s²): Main Vacuum Raptors -> 4 Mid-Body HLS Thrusters!
    let mut sim_lunar = StarshipSim::new(StarshipScenario::LunarHlsDescent)?;
    for _ in 0..(95.0 / SIM_DT).ceil() as usize {
        sim_lunar.step(SIM_DT)?;
        if sim_lunar.ever_ship_landed {
            break;
        }
    }
    let t_lunar = sim_lunar.telemetry()?;

    let report = serde_json::json!({
        "earth_stack": {
            "lifted_off": sim_stack.ever_lifted_off,
            "hot_staged": sim_stack.ever_hot_staged,
            "max_q_kpa": sim_stack.peak_max_q_kpa,
            "interstage_blast_mn": sim_stack.peak_interstage_blast_mn,
            "peak_twd_knm": sim_stack.peak_twd_knm,
            "booster_alt_m": t_stack.booster_alt_m,
            "ship_alt_m": t_stack.ship_alt_m,
        },
        "earth_catch": {
            "caught": sim_catch.ever_booster_caught,
            "time_s": sim_catch.time_s,
            "speed_mps": sim_catch.booster_catch_speed_mps,
            "offset_m": sim_catch.booster_catch_offset_m,
            "tilt_deg": t_catch.booster_tilt_deg,
            "phase": t_catch.booster_phase.label(),
        },
        "earth_ship": {
            "bellyflop": sim_ship.ever_ship_bellyflop,
            "flipped": sim_ship.ever_ship_flipped,
            "landed": sim_ship.ever_ship_landed,
            "time_s": sim_ship.time_s,
            "touchdown_speed_mps": sim_ship.ship_touchdown_speed_mps,
            "touchdown_tilt_deg": sim_ship.ship_touchdown_tilt_deg,
            "pad_dx_m": t_ship.ship_dist_pad_m,
            "phase": t_ship.ship_phase.label(),
        },
        "mars_srp": {
            "landed": sim_mars.ever_ship_landed,
            "time_s": sim_mars.time_s,
            "peak_mach": mars_peak_mach,
            "min_srp_factor": mars_min_srp,
            "touchdown_speed_mps": sim_mars.ship_touchdown_speed_mps,
            "touchdown_tilt_deg": sim_mars.ship_touchdown_tilt_deg,
            "pad_dx_m": t_mars.ship_dist_pad_m,
            "alt_m": t_mars.ship_alt_m,
            "phase": t_mars.ship_phase.label(),
        },
        "lunar_hls": {
            "landed": sim_lunar.ever_ship_landed,
            "time_s": sim_lunar.time_s,
            "touchdown_speed_mps": sim_lunar.ship_touchdown_speed_mps,
            "touchdown_tilt_deg": sim_lunar.ship_touchdown_tilt_deg,
            "pad_dx_m": t_lunar.ship_dist_pad_m,
            "alt_m": t_lunar.ship_alt_m,
            "phase": t_lunar.ship_phase.label(),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);

    if !sim_stack.ever_lifted_off
        || !sim_stack.ever_hot_staged
        || sim_stack.peak_interstage_blast_mn < 1.0
        || !sim_catch.ever_booster_caught
        || sim_catch.booster_catch_offset_m > 4.5
        || !sim_ship.ever_ship_bellyflop
        || !sim_ship.ever_ship_flipped
        || !sim_ship.ever_ship_landed
        || sim_ship.ship_touchdown_speed_mps > 4.5
        || sim_ship.ship_touchdown_tilt_deg > 8.0
        || t_ship.ship_dist_pad_m > 25.0
        || !sim_mars.ever_ship_landed
        || mars_peak_mach < 1.15
        || sim_mars.ship_touchdown_speed_mps > 4.5
        || sim_mars.ship_touchdown_tilt_deg > 8.0
        || t_mars.ship_dist_pad_m > 30.0
        || !sim_lunar.ever_ship_landed
        || sim_lunar.ship_touchdown_speed_mps > 3.5
        || sim_lunar.ship_touchdown_tilt_deg > 8.0
        || t_lunar.ship_dist_pad_m > 25.0
    {
        return Err("Starship full-stack physics validation failed".into());
    }
    Ok(())
}

fn main() -> video_capture::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--validate-physics") {
        return run_validation();
    }
    let scenario = if args.iter().any(|a| a == "--catch") {
        StarshipScenario::BoosterCatch
    } else if args.iter().any(|a| a == "--flip") {
        StarshipScenario::ShipFlipLanding
    } else if args.iter().any(|a| a == "--hotstage") {
        StarshipScenario::HotStaging
    } else if args.iter().any(|a| a == "--mars") {
        StarshipScenario::MarsSrpLanding
    } else if args.iter().any(|a| a == "--lunar") {
        StarshipScenario::LunarHlsDescent
    } else {
        StarshipScenario::FullStackMission
    };

    let sim = StarshipSim::new(scenario)?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Molla | Full-Scale Starship + Super Heavy + Mechazilla")
            .with_inner_size([1380.0, 880.0]),
        renderer: eframe::Renderer::Wgpu,
        multisampling: 4,
        depth_buffer: 24,
        ..Default::default()
    };
    eframe::run_native(
        "Molla | Full-Scale Starship + Super Heavy + Mechazilla",
        options,
        Box::new(move |creation| {
            let render = creation.wgpu_render_state.as_ref().expect("wgpu render state");
            studio::initialize(render);
            let mut app = App {
                sim,
                camera: default_camera(scenario),
                camera_ship: draw::Camera {
                    target: Vec3::ZERO,
                    distance: 7.4,
                    yaw: 0.48,
                    pitch: 0.14,
                },
                focus: FocusTarget::ActiveVehicle,
                split_mode: SplitMode::Auto,
                was_split: false,
                render_origin: Vec3::ZERO,
                paused: false,
                time_scale: 1.0,
                last: Instant::now(),
                accumulator: 0.0,
                error: None,
            };
            app.render_origin = app.focus_world_pos();
            Ok(Box::new(app))
        }),
    )?;
    Ok(())
}
