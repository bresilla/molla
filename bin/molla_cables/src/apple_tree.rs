mod adaptive_step;
mod apple_showcase;
mod apple_showcase_probe;
mod apple_tree_model;
mod apple_tree_sim;
mod apple_tree_verify;
#[allow(dead_code)]
mod branch;
#[allow(dead_code)]
mod branch_draw;
#[allow(dead_code)]
mod draw;
mod grab_handle;
#[allow(dead_code)]
mod scene;
mod foliage_layout;
mod studio;
mod video_capture;

use branch_draw::{Action, Interaction, Tool};
use eframe::egui;
use molla_compute::Engine;
use molla_core::Result;
use molla_geometry::convex_cut::{CutPlane, FaceOrigin};
use molla_math::Vec3;
use std::time::Instant;

use apple_tree_sim::{DT, Tree, gpu_limits, step_config};

struct App {
    engine: Engine,
    tree: Tree,
    source: String,
    budget: usize,
    camera: draw::Camera,
    interaction: Interaction,
    paused: bool,
    bonds: bool,
    error: Option<String>,
    last_tick: Instant,
    accumulated: f64,
    step_ms: f64,
    playback: f64,
    frame: usize,
    screenshot: Option<String>,
    verify_ui: bool,
    verify_grab_ui: bool,
    verify_grab_pick: usize,
    verify_grab_direction: egui::Vec2,
    verify_grab_gentle: bool,
    verify_grab_allow_intact: bool,
    grab_test_pixel: Option<egui::Pos2>,
    captured: bool,
    initial_cells: usize,
    capture_frame: usize,
    started: Instant,
    grab_plane: Option<(Vec3, Vec3)>,
}

impl App {
    fn fail(&mut self, error: impl std::fmt::Display) {
        eprintln!("Apple tree error at frame {}: {error}", self.frame);
        self.error = Some(error.to_string());
        self.paused = true;
    }

    fn reset(&mut self) {
        match Tree::new(&self.engine, &self.source, self.budget) {
            Ok(mut tree) => {
                tree.grip_force = self.tree.grip_force;
                self.tree = tree;
                self.error = None;
                self.accumulated = 0.0;
                self.interaction.start = None;
                self.grab_plane = None;
            }
            Err(e) => self.fail(e),
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(12.0);
        ui.colored_label(draw::COLORS[0], "MOLLA / APPLE ORCHARD");
        ui.heading("A real apple skeleton.");
        ui.label("Bend it. Prune it. Let it fall.");
        ui.add_space(8.0);
        ui.small(format!(
            "AppleQSM Row13 · {} · Appleskel reconstruction",
            self.tree.name
        ));
        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .button(if self.paused { "Play" } else { "Pause" })
                .clicked()
            {
                self.paused = !self.paused;
            }
            if ui.button("Reset [R]").clicked() {
                self.reset();
            }
        });
        if ui.button("Breeze / sway").clicked()
            && let Err(e) = self.tree.sway(&self.engine)
        {
            self.fail(e);
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.interaction.tool, Tool::Cut, "Cut [C]");
            ui.selectable_value(&mut self.interaction.tool, Tool::Break, "Break [B]");
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.interaction.tool, Tool::Push, "Push [P]");
            ui.selectable_value(&mut self.interaction.tool, Tool::Grab, "Grab / bend [G]");
            ui.selectable_value(&mut self.interaction.tool, Tool::Orbit, "Orbit [O]");
        });
        ui.label(match self.interaction.tool {
            Tool::Cut => "Click for a smooth cut, or drag a blade across the wood.",
            Tool::Break => "Click for a rough break at that position, or drag across the wood.",
            Tool::Grab => {
                "Hold the left mouse button on wood and drag to bend it. Release to let go."
            }
            Tool::Push => "Click wood to give it a push.",
            Tool::Orbit => "Drag to orbit the tree. Scroll to zoom.",
        });
        ui.small("Cut: smooth face. Break: splintered wood at the click. Grab: hold and drag until the wood fails; release to drop it.");
        ui.checkbox(&mut self.bonds, "Show elastic joints");
        ui.add(egui::Slider::new(&mut self.tree.grip_force, 1.0..=20.0).text("Grip force (N)"));
        ui.small("Higher grip force bends thicker wood but stores more recoil energy.");
        ui.add(egui::Slider::new(&mut self.budget, 4..=60).text("branch budget"));
        if ui.button("Rebuild with this budget").clicked() {
            self.reset();
        }
        if ui.button("Reset camera").clicked() {
            self.camera = camera();
        }
        ui.separator();
        ui.monospace(format!(
            "Limbs         {:3} / {}",
            self.tree.limbs, self.tree.source_limbs
        ));
        ui.monospace(format!(
            "Solid cells   {:3}",
            self.tree.model.fibers().len()
        ));
        ui.monospace(format!(
            "Pieces        {:3}",
            self.tree.model.components().len()
        ));
        ui.monospace(format!(
            "Cuts / breaks {:3} / {}",
            self.tree.cuts, self.tree.breaks
        ));
        ui.monospace(format!("Physics       {:5.1} ms", self.step_ms));
        ui.monospace(format!("Playback      {:5.2} ×", self.playback));
        ui.monospace(format!("Time          {:5.2} s", self.tree.time));
        ui.monospace(format!("Stress snaps  {:3}", self.tree.stress_breaks));
        if self.tree.recovered_batches > 0 {
            ui.small(format!(
                "Adaptive recoveries: {}",
                self.tree.recovered_batches
            ));
        }
        ui.add_space(8.0);
        ui.label(&self.tree.message);
        ui.small("GPU elastic joints + convex piece contacts + floor. Discrete collision, not CCD. Radius taper and wood material are illustrative, not calibrated measurements.");
        ui.small("Small twigs omitted for speed. SPACE pauses. Right-drag orbits; scroll zooms.");
        ui.hyperlink_to(
            "Tian Qiu · AppleQSM-Row13 · CC BY 4.0",
            "https://doi.org/10.6084/m9.figshare.26215094.v1",
        );
        ui.hyperlink_to("Appleskel source", "https://github.com/bresilla/appleskel");
        ui.hyperlink_to(
            "All acknowledgements",
            concat!(
                "file://",
                env!("CARGO_MANIFEST_DIR"),
                "/../../ACKNOWLEDGEMENT.md"
            ),
        );
        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
    }
}

impl eframe::App for App {
    fn raw_input_hook(&mut self, _: &egui::Context, input: &mut egui::RawInput) {
        if !self.verify_ui {
            return;
        }
        input
            .events
            .retain(|event| matches!(event, egui::Event::Screenshot { .. }));
        input.focused = true;
        let Some(rect) = self.interaction.viewport else {
            return;
        };
        if self.verify_grab_ui {
            if self.frame == 20 {
                let (pixel, id) =
                    self.tree
                        .visible_limb_at(&self.camera, rect, self.verify_grab_pick);
                eprintln!(
                    "Native grab case: cell {id}, direction {:?}, gentle={}",
                    self.verify_grab_direction, self.verify_grab_gentle
                );
                self.grab_test_pixel = Some(pixel);
            }
            if self.frame == 20 || self.frame == 360 {
                let key = if self.frame == 20 {
                    egui::Key::G
                } else {
                    egui::Key::R
                };
                input.events.push(egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
            }
            if let Some(start) = self.grab_test_pixel
                && (21..=225).contains(&self.frame)
            {
                let x = if self.frame < 23 {
                    0.0
                } else if self.frame <= 60 {
                    8.0 * (((self.frame - 23) as f32 / 37.0) * std::f32::consts::TAU).sin()
                } else if self.verify_grab_gentle {
                    12.0 * (((self.frame - 60) as f32 / 160.0).min(1.0) * std::f32::consts::TAU)
                        .sin()
                } else {
                    600.0 * ((self.frame - 60) as f32 / 160.0).min(1.0)
                };
                let pos = start + self.verify_grab_direction * x;
                input.events.push(egui::Event::PointerMoved(pos));
                if self.frame == 22 || self.frame == 225 {
                    input.events.push(egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed: self.frame == 22,
                        modifiers: Default::default(),
                    });
                }
            }
            return;
        }
        if matches!(self.frame, 20 | 21 | 22 | 29 | 30 | 31) {
            let pixel = self.tree.visible_limb(&self.camera, rect);
            input.events.push(egui::Event::PointerMoved(pixel));
            if matches!(self.frame, 21 | 22 | 30 | 31) {
                input.events.push(egui::Event::PointerButton {
                    pos: pixel,
                    button: egui::PointerButton::Primary,
                    pressed: self.frame == 21 || self.frame == 30,
                    modifiers: Default::default(),
                });
            }
        }
        let key = match self.frame {
            28 => Some(egui::Key::B),
            40 => Some(egui::Key::Space),
            170 => Some(egui::Key::R),
            _ => None,
        };
        if let Some(key) = key {
            for pressed in [true, false] {
                input.events.push(egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed,
                    repeat: false,
                    modifiers: Default::default(),
                });
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        for event in ctx.input(|i| i.events.clone()) {
            if let egui::Event::Screenshot { image, .. } = event
                && let Some(path) = self.screenshot.take()
            {
                let bytes: Vec<_> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                image::save_buffer(
                    &path,
                    &bytes,
                    image.width() as u32,
                    image.height() as u32,
                    image::ColorType::Rgba8,
                )
                .expect("save framebuffer");
                self.captured = true;
                eprintln!("Captured {path}");
                eprintln!(
                    "Apple viewport: {} frames, {:.3} wall seconds, {:.3} simulated seconds, {:.2} recent playback",
                    self.frame,
                    self.started.elapsed().as_secs_f64(),
                    self.tree.time,
                    self.playback
                );
                if !self.verify_ui {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.paused = !self.paused;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.reset();
        }
        for (key, tool) in [
            (egui::Key::C, Tool::Cut),
            (egui::Key::B, Tool::Break),
            (egui::Key::P, Tool::Push),
            (egui::Key::G, Tool::Grab),
            (egui::Key::O, Tool::Orbit),
        ] {
            if ctx.input(|i| i.key_pressed(key)) {
                self.interaction.tool = tool;
            }
        }
        if (self.interaction.tool != Tool::Grab
            && !(self.interaction.tool == Tool::Push && self.tree.push_remaining.is_some()))
            || ctx.input(|i| {
                i.key_pressed(egui::Key::Escape)
                    || !i.focused
                    || (self.interaction.tool == Tool::Grab && i.pointer.primary_released())
            })
        {
            self.tree.release_grab();
            self.grab_plane = None;
        }
        let elapsed = self.last_tick.elapsed().as_secs_f64().min(0.1);
        self.last_tick = Instant::now();
        if !self.paused && self.interaction.start.is_none() {
            let batch_limit = if self.tree.grab.is_some() { 4.0 } else { 8.0 };
            self.accumulated = (self.accumulated + elapsed).min(DT * batch_limit);
            let steps = if self.verify_ui {
                4
            } else {
                (self.accumulated / DT).floor() as usize
            };
            if steps > 0 {
                let now = Instant::now();
                if let Err(e) = self.tree.advance(&self.engine, steps) {
                    self.fail(e);
                }
                self.step_ms = now.elapsed().as_secs_f64() * 1000.0;
                self.accumulated = (self.accumulated - steps as f64 * DT).max(0.0);
            }
            self.playback = self.playback * 0.9 + 0.1 * (steps as f64 * DT / elapsed.max(1e-6));
        } else {
            self.accumulated = 0.0;
        }
        egui::Panel::left("apple controls")
            .exact_size(290.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        let mut action = None;
        egui::CentralPanel::default().show_inside(ui, |ui| {
            action = branch_draw::paint_model(
                ui,
                &self.tree.model,
                &mut self.camera,
                &mut self.interaction,
                self.bonds,
                false,
            );
            if let (Some(grab), Some(handle), Some(rect)) =
                (self.tree.grab, self.tree.handle, self.interaction.viewport)
            {
                let point = self
                    .tree
                    .model
                    .fiber(grab.fiber)
                    .unwrap()
                    .motion
                    .pose
                    .transform_point(grab.local_point);
                let (a, za) = self.camera.project(point, rect);
                let (b, zb) = self.camera.project(handle.point, rect);
                if za > 0.0 && zb > 0.0 {
                    let color = egui::Color32::from_rgb(255, 190, 90);
                    ui.painter()
                        .line_segment([a, b], egui::Stroke::new(1.5, color));
                    ui.painter().circle_filled(a, 3.5, color);
                    ui.painter()
                        .circle_stroke(b, 6.0, egui::Stroke::new(1.5, color));
                }
            }
        });
        if let Some(action) = action {
            let result = match action {
                Action::Click(p, r, fracture) => {
                    self.tree
                        .click(&self.engine, &self.camera, p, r, fracture, false)
                }
                Action::Push(p, r) => {
                    self.paused = false;
                    self.tree
                        .click(&self.engine, &self.camera, p, r, false, true)
                }
                Action::Section(a, b, r, fracture) => {
                    branch::blade_selection(&self.tree.model, &self.camera, r, a, b).and_then(
                        |(plane, ids)| self.tree.section(&self.engine, plane, &ids, fracture),
                    )
                }
                Action::GrabStart(p, r) => {
                    let (origin, direction) = self.camera.ray(p, r);
                    if let Some((id, hit)) = branch::ray_hit(&self.tree.model, origin, direction) {
                        let f = self.tree.model.fiber(id).unwrap();
                        if f.pin.is_none() {
                            self.tree.start_grab(id, hit);
                            self.grab_plane = Some((hit, self.camera.ray(r.center(), r).1));
                            self.paused = false;
                            self.tree.message = "Holding the wood. Drag to bend; enough load breaks it. Release to let go.".into();
                        }
                    }
                    Ok(())
                }
                Action::GrabMove(p, r) => {
                    if let (Some(grab), Some((point, normal))) =
                        (&mut self.tree.grab, self.grab_plane)
                    {
                        let (origin, direction) = self.camera.ray(p, r);
                        let denominator = normal.dot(direction);
                        if denominator.abs() > 1e-6 {
                            let t = normal.dot(point - origin) / denominator;
                            if t > 0.0 {
                                grab.target = origin + direction * t;
                            }
                        }
                    }
                    Ok(())
                }
                Action::GrabEnd => {
                    self.tree.release_grab();
                    self.grab_plane = None;
                    Ok(())
                }
            };
            if let Err(e) = result {
                self.fail(e);
            }
        }
        if self.verify_ui && !self.verify_grab_ui && self.frame == 159 {
            assert!(self.error.is_none(), "tree UI error: {:?}", self.error);
            assert_eq!(self.tree.cuts, 1);
            assert_eq!(self.tree.breaks, 1);
            assert_eq!(self.tree.model.components().len(), 3);
            assert!(
                self.tree.clearance() > -0.01,
                "tree penetrated the floor: {}",
                self.tree.clearance()
            );
            let caps = self
                .tree
                .model
                .fibers()
                .iter()
                .flat_map(|f| f.shape.faces())
                .filter(|f| {
                    matches!(
                        f.origin,
                        FaceOrigin::Cut { .. } | FaceOrigin::Fracture { .. }
                    )
                })
                .count();
            assert!(
                caps >= 10,
                "rough break must have multiple cap faces: {caps}"
            );
            self.paused = true;
            eprintln!(
                "PASS apple UI: click cut + splintered break, three pieces, multiple actual caps, finite GPU motion, floor"
            );
        }
        if self.verify_grab_ui && self.frame == 60 {
            assert!(self.tree.grab.is_some(), "native drag did not grab wood");
            assert_eq!(self.tree.stress_breaks, 0, "gentle native drag broke wood");
        }
        if self.verify_grab_ui && self.frame == 349 {
            assert!(self.error.is_none(), "grab UI error: {:?}", self.error);
            if self.verify_grab_gentle {
                assert_eq!(self.tree.stress_breaks, 0, "gentle native drag broke wood");
            } else if !self.verify_grab_allow_intact {
                assert!(
                    self.tree.stress_breaks > 0,
                    "native bend did not break wood"
                );
            }
            assert!(
                self.tree.grab.is_none(),
                "mouse release must release the wood"
            );
            assert_eq!(self.tree.cuts + self.tree.breaks, 0);
            if self.tree.stress_breaks == 0 {
                assert_eq!(self.tree.model.components().len(), 1);
            } else {
                assert!(self.tree.model.components().len() > 1);
            }
            for fiber in self.tree.model.fibers() {
                if let Some(pin) = fiber.pin {
                    assert_eq!(fiber.motion.pose, pin.pose);
                }
            }
            assert!(self.tree.clearance() > -0.01);
            self.paused = true;
            eprintln!(
                "PASS native grab: actual mouse picking, direction reversal, gentle={}, release, fixed root and floor; {} stress snaps; {:.3} simulated / {:.3} wall seconds, {} recoveries, {} N grip",
                self.verify_grab_gentle,
                self.tree.stress_breaks,
                self.tree.time,
                self.started.elapsed().as_secs_f64(),
                self.tree.recovered_batches,
                self.tree.grip_force
            );
        }
        if self.frame == self.capture_frame && self.screenshot.is_some() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        if self.verify_ui && self.frame == if self.verify_grab_ui { 364 } else { 174 } {
            assert!(self.captured, "framebuffer capture did not complete");
            assert_eq!(self.tree.model.fibers().len(), self.initial_cells);
            assert_eq!(self.tree.model.components().len(), 1);
            assert_eq!(self.tree.cuts + self.tree.breaks, 0);
            eprintln!("PASS apple UI: reset restored the measured tree");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        self.frame += 1;
        ctx.request_repaint_after(std::time::Duration::from_millis(1));
    }
}

fn camera() -> draw::Camera {
    draw::Camera {
        yaw: 0.6,
        pitch: 0.16,
        distance: 4.4,
        ..Default::default()
    }
}

fn verify(engine: &Engine, source: &str, budget: usize, bench: bool) -> Result<()> {
    let mut tree = Tree::new(engine, source, budget)?;
    let root = tree.model.fibers()[0].motion.pose;
    let initial: Vec<_> = tree.model.fibers().iter().map(|f| f.center()).collect();
    tree.sway(engine)?;
    if !bench {
        let mut cpu = tree.model.clone();
        molla_solvers::strands::step_cpu(&mut cpu, step_config())?;
        tree.advance(engine, 1)?;
        let error = tree
            .model
            .fibers()
            .iter()
            .zip(cpu.fibers())
            .map(|(a, b)| a.center().distance(b.center()))
            .fold(0.0, f64::max);
        assert!(error < 1e-4, "apple-tree CPU/GPU position error {error}");
        eprintln!("PASS apple CPU/GPU one-step position oracle: {error:.9} m");
    }
    for _ in 0..30 {
        tree.advance(engine, 2)?;
    }
    assert_eq!(tree.model.components().len(), 1);
    assert_eq!(tree.model.fibers()[0].motion.pose, root);
    let moved = tree
        .model
        .fibers()
        .iter()
        .zip(&initial)
        .map(|(f, p)| f.center().distance(*p))
        .fold(0.0, f64::max);
    eprintln!(
        "Apple motion: cells={}, peak deflection={moved:.4} m, clearance={:.5}",
        tree.model.fibers().len(),
        tree.clearance()
    );
    assert!(
        moved > 0.001 && moved < 0.5,
        "unexpected tree deflection {moved}"
    );
    let mut timing = Vec::new();
    for _ in 0..20 {
        let now = Instant::now();
        tree.advance(engine, 2)?;
        timing.push(now.elapsed().as_secs_f64() * 1000.0);
    }
    timing.sort_by(f64::total_cmp);
    eprintln!(
        "APPLE_BENCH,limbs={},cells={},two_steps_ms={:.3},worst_ms={:.3},simulated_ms={:.3},deflection_m={moved:.4}",
        tree.limbs,
        tree.model.fibers().len(),
        timing[10],
        timing[19],
        2000.0 * DT
    );
    if bench {
        return Ok(());
    }
    let f = tree.model.fibers().last().unwrap();
    let id = f.id;
    let initial_y = f.center().y;
    let plane = CutPlane::new(f.center(), f.motion.pose.rotation * Vec3::Z)?;
    let mass = tree.model.fibers().iter().map(|f| f.mass.mass).sum::<f64>();
    tree.section(engine, plane, &[id], false)?;
    assert_eq!(tree.model.components().len(), 2);
    assert!((tree.model.fibers().iter().map(|f| f.mass.mass).sum::<f64>() - mass).abs() < 1e-8);
    for _ in 0..240 {
        tree.advance(engine, 2)?;
    }
    let detached = tree
        .model
        .components()
        .into_iter()
        .find(|ids| {
            ids.iter()
                .all(|&id| tree.model.fiber(id).unwrap().pin.is_none())
        })
        .unwrap();
    assert!(
        detached
            .iter()
            .any(|&id| tree.model.fiber(id).unwrap().center().y < initial_y - 0.1),
        "cut subtree did not fall"
    );
    assert!(
        tree.clearance() > -0.01,
        "tree floor penetration {}",
        tree.clearance()
    );
    eprintln!(
        "PASS apple tree: pinned root, elastic sway, no spontaneous break, exact cut, mass conservation, falling detached wood, floor"
    );
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let value = |key: &str| {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let source = value("--skeleton")
        .map(std::fs::read_to_string)
        .transpose()?
        .unwrap_or_else(|| apple_tree_model::SOURCE.into());
    let budget = value("--branches").map_or(Ok(apple_tree_model::DEFAULT_LIMBS), |s| s.parse())?;
    let grip_force = value("--grip-force").map_or(Ok(4.0), |s| s.parse::<f64>())?;
    if !grip_force.is_finite() || !(1.0..=20.0).contains(&grip_force) {
        return Err("grip force must be 1..20 N".into());
    }
    let engine = Engine::new_gpu()?;
    eprintln!("Apple tree physics adapter: {:?}", engine.adapter_info());
    if args.iter().any(|a| a == "--probe-release-gap") {
        apple_showcase_probe::released_sequence(&engine)?;
        return Ok(());
    }
    if let Some(path) = value("--capture-smoke") {
        apple_showcase::smoke(&engine, std::path::Path::new(&path))?;
        return Ok(());
    }
    if let Some(path) = value("--showcase-video").or_else(|| value("--verify-showcase")) {
        apple_showcase::run(
            &engine,
            std::path::Path::new(&path),
            args.iter().any(|a| a == "--showcase-video"),
        )?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--probe-showcase") {
        apple_showcase_probe::run(
            &engine,
            value("--grab-cell").map(|s| s.parse()).transpose()?,
            value("--grab-axis").map(|s| s.parse()).transpose()?,
        )?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--verify-grab-matrix") {
        apple_tree_verify::grab_matrix(
            &engine,
            &source,
            budget,
            args.iter().any(|a| a == "--strong"),
            value("--grab-cell").map(|s| s.parse::<u64>()).transpose()?,
            value("--grab-axis")
                .map(|s| s.parse::<usize>())
                .transpose()?,
            grip_force,
        )?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--verify-grab") {
        apple_tree_verify::grab_motion(&engine, &source, budget)?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--verify-tools") {
        apple_tree_verify::tools(&engine, &source, budget)?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--verify-recovery") {
        let seconds = value("--stress-seconds").map_or(Ok(30.0), |s| s.parse::<f64>())?;
        if !seconds.is_finite() || !(0.0..=1800.0).contains(&seconds) {
            return Err("stress duration must be 0..1800 seconds".into());
        }
        apple_tree_verify::recovery(&engine, &source, budget, seconds)?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--verify" || a == "--bench") {
        verify(
            &engine,
            &source,
            budget,
            args.iter().any(|a| a == "--bench"),
        )?;
        return Ok(());
    }
    let verify_grab_ui = args.iter().any(|a| a == "--verify-grab-ui");
    let verify_ui = verify_grab_ui || args.iter().any(|a| a == "--verify-ui");
    let live = args.iter().any(|a| a == "--live");
    let grab_angle = value("--grab-angle")
        .map_or(Ok(90.0), |s| s.parse::<f32>())?
        .to_radians();
    if !grab_angle.is_finite() {
        return Err("invalid grab angle".into());
    }
    let screenshot =
        value("--screenshot").or_else(|| verify_ui.then(|| "/tmp/molla-apple-tree-ui.png".into()));
    let mut tree = Tree::new(&engine, &source, budget)?;
    tree.grip_force = grip_force;
    let initial_cells = tree.model.fibers().len();
    eprintln!(
        "Apple tree: {} of {} limbs, {} simulated convex cells",
        tree.limbs, tree.source_limbs, initial_cells
    );
    let app = App {
        engine,
        tree,
        source,
        budget,
        camera: draw::Camera {
            distance: value("--camera-distance")
                .map_or(Ok(camera().distance), |s| s.parse::<f64>())?
                .clamp(0.6, 12.0),
            ..camera()
        },
        interaction: Interaction::default(),
        paused: verify_ui || screenshot.is_some() && !live,
        bonds: false,
        error: None,
        last_tick: Instant::now(),
        accumulated: 0.0,
        step_ms: 0.0,
        playback: 1.0,
        frame: 0,
        screenshot,
        verify_ui,
        verify_grab_ui,
        verify_grab_pick: value("--grab-pick").map_or(Ok(8), |s| s.parse::<usize>())?,
        verify_grab_direction: egui::vec2(grab_angle.cos(), grab_angle.sin()),
        verify_grab_gentle: args.iter().any(|a| a == "--gentle"),
        verify_grab_allow_intact: args.iter().any(|a| a == "--allow-intact"),
        grab_test_pixel: None,
        captured: false,
        initial_cells,
        capture_frame: value("--capture-frame")
            .map(|s| s.parse::<usize>())
            .transpose()?
            .unwrap_or(if verify_grab_ui {
                350
            } else if verify_ui {
                160
            } else if live {
                300
            } else {
                20
            }),
        started: Instant::now(),
        grab_plane: None,
    };
    let options = eframe::NativeOptions {
        depth_buffer: 24,
        multisampling: 4,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 900.0])
            .with_app_id("molla-apple-tree"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "Molla — Apple Tree",
        options,
        Box::new(move |cc| {
            studio::initialize(cc.wgpu_render_state.as_ref().expect("wgpu renderer"));
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )?;
    Ok(())
}
