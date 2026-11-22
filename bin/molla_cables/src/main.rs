mod branch;
mod branch_draw;
mod draw;
mod scene;
mod foliage_layout;
mod studio;
mod verify;

use branch_draw::{Action, Interaction, Tool};
use eframe::egui;
use molla_compute::Engine;
use molla_core::Result;
use molla_math::Vec3;
use std::time::Instant;

const NAMES: [&str; 5] = [
    "Suspended cable",
    "Bending comparison",
    "Drape over a sphere",
    "Elastic branch — Cut",
    "Elastic branch — Break",
];

enum ActiveScene {
    Cable(Box<scene::Scene>),
    Branch(Box<branch::Branch>),
}
impl ActiveScene {
    fn new(engine: &Engine, preset: usize) -> Result<Self> {
        if preset < 3 {
            Ok(Self::Cable(Box::new(scene::Scene::new(engine, preset)?)))
        } else {
            Ok(Self::Branch(Box::new(branch::Branch::new(engine)?)))
        }
    }
    fn advance(&mut self, engine: &Engine, steps: usize) -> Result<()> {
        match self {
            Self::Cable(s) => s.advance(engine, steps),
            Self::Branch(s) => s.advance(engine, steps),
        }
    }
    fn pluck(&mut self, engine: &Engine, strength: f64) -> Result<()> {
        match self {
            Self::Cable(s) => s.pluck(engine),
            Self::Branch(s) => s.pluck(engine, strength),
        }
    }
    fn time(&self) -> f64 {
        match self {
            Self::Cable(s) => s.time,
            Self::Branch(s) => s.time,
        }
    }
    fn clearance(&self) -> f64 {
        match self {
            Self::Cable(s) => s.clearance(),
            Self::Branch(s) => s.clearance(),
        }
    }
}

struct App {
    engine: Engine,
    scene: ActiveScene,
    preset: usize,
    camera: draw::Camera,
    interaction: Interaction,
    paused: bool,
    nodes: bool,
    steps: usize,
    step_ms: f64,
    strength: f64,
    error: Option<String>,
    screenshot: Option<String>,
    frame: usize,
    verify_ui: bool,
    verify_click: bool,
    capture_saved: bool,
    last_tick: Instant,
    accumulated_time: f64,
    simulation_rate: f64,
}

impl App {
    fn reset(&mut self, preset: usize) {
        match ActiveScene::new(&self.engine, preset) {
            Ok(scene) => {
                self.scene = scene;
                self.preset = preset;
                self.error = None;
                self.accumulated_time = 0.0;
                self.simulation_rate = 1.0;
                self.interaction.start = None;
                self.interaction.tool = if preset == 4 { Tool::Break } else { Tool::Cut };
            }
            Err(error) => self.fail(error),
        }
    }
    fn fail(&mut self, error: impl std::fmt::Display) {
        self.error = Some(error.to_string());
        self.paused = true;
    }
    fn advance(&mut self, steps: usize) {
        let now = Instant::now();
        if let Err(error) = self.scene.advance(&self.engine, steps) {
            self.fail(error);
        }
        self.step_ms = now.elapsed().as_secs_f64() * 1000.0;
    }
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.label(
            egui::RichText::new("MOLLA / CABLE LAB")
                .color(draw::COLORS[0])
                .size(13.0),
        );
        ui.heading(if self.preset < 3 {
            "Bend. Swing. Drape."
        } else {
            "Bend. Cut. Break."
        });
        ui.small(if self.preset < 3 {
            "GPU VBD · stretch + discrete bending"
        } else {
            "GPU coupled XPBD · oriented fiber volumes"
        });
        ui.separator();
        for (i, name) in NAMES.iter().enumerate() {
            if ui
                .selectable_label(self.preset == i, format!("{}   {name}", i + 1))
                .clicked()
            {
                self.reset(i);
            }
        }
        ui.add_space(8.0);
        if self.preset < 3 {
            ui.label(scene::DESCRIPTIONS[self.preset]);
        } else {
            ui.label("Select Cut or Break, then click the branch. Cut makes a clean face; Break splits fibers unevenly around that point.");
        }
        ui.horizontal(|ui| {
            if ui
                .button(if self.paused { "Play" } else { "Pause" })
                .clicked()
            {
                self.paused = !self.paused;
            }
            if ui.button("Reset").clicked() {
                self.reset(self.preset);
            }
            if ui.button("Step").clicked() {
                self.paused = true;
                self.advance(1);
            }
        });
        if ui
            .button(if self.preset < 3 {
                "Pluck cables"
            } else {
                "Bend / pluck branch"
            })
            .clicked()
            && let Err(error) = self.scene.pluck(&self.engine, self.strength)
        {
            self.fail(error);
        }
        if self.preset >= 3 {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.interaction.tool, Tool::Cut, "Cut [C]");
                ui.selectable_value(&mut self.interaction.tool, Tool::Break, "Break [B]");
                ui.selectable_value(&mut self.interaction.tool, Tool::Push, "Push [P]");
                ui.selectable_value(&mut self.interaction.tool, Tool::Orbit, "Orbit [O]");
            });
            ui.add(egui::Slider::new(&mut self.strength, 0.1..=8.0).text("push / pluck"));
            if let ActiveScene::Branch(branch) = &mut self.scene {
                let mut rigidity = branch.rigidity;
                let response = ui.add(
                    egui::Slider::new(&mut rigidity, 0.2..=5.0)
                        .logarithmic(true)
                        .text("rigidity ×"),
                );
                if response.changed()
                    && let Err(error) = branch.set_rigidity(&self.engine, rigidity)
                {
                    self.fail(error);
                }
            }
        }
        ui.add(egui::Slider::new(&mut self.steps, 1..=8).text("max steps / frame"));
        ui.checkbox(
            &mut self.nodes,
            if self.preset < 3 {
                "Show simulation nodes"
            } else {
                "Show internal bonds"
            },
        );
        if ui.button("Reset camera").clicked() {
            self.camera = draw::Camera::default();
        }
        ui.separator();
        ui.monospace(format!("Simulation     {:6.2} s", self.scene.time()));
        ui.monospace(format!("Playback       {:6.2} ×", self.simulation_rate));
        ui.monospace(format!("GPU + readback {:6.1} ms", self.step_ms));
        match &self.scene {
            ActiveScene::Cable(s) => {
                ui.monospace(format!("Particles      {:6}", s.points.len()));
                ui.monospace(format!("Max stretch    {:6.3} x", s.max_stretch));
                if self.preset == 1 {
                    for (i, label) in [
                        "Flexible · k_b = 5",
                        "Medium · k_b = 2,000",
                        "Stiff · k_b = 80,000",
                    ]
                    .iter()
                    .enumerate()
                    {
                        ui.colored_label(draw::COLORS[i], *label);
                    }
                }
                ui.small("VBD demo: no torsional frame or continuous cable self-contact.");
            }
            ActiveScene::Branch(s) => {
                let (damaged, broken) = s.damage_counts();
                ui.monospace(format!("Long fibers    {:3}", branch::STRANDS));
                ui.monospace(format!(
                    "Cells / pieces {:3} / {}",
                    s.model.fibers().len(),
                    s.model.components().len()
                ));
                ui.monospace(format!("Damaged/broken {:3} / {broken}", damaged));
                ui.monospace(format!("Cuts / breaks  {:3} / {}", s.cuts, s.breaks));
                ui.monospace(format!("Exposed caps   {:3}", s.cap_count()));
                ui.small(&s.message);
                ui.colored_label(
                    draw::COLORS[1],
                    "Same elasticity in Cut and Break. Only the separation surface changes.",
                );
                ui.small("Convex piece collision + floor. Discrete contacts: high-speed tunnelling is not yet prevented. Simulation speed is being qualified.");
                ui.small("Click or drag with either tool. Cut is planar; Break staggers the fiber ends. Both keep the same material and elasticity.");
            }
        }
        ui.add_space(8.0);
        ui.small("SPACE pause · R reset · 1–5 scenes");
        ui.hyperlink_to(
            "Sources and attribution",
            concat!(
                "file://",
                env!("CARGO_MANIFEST_DIR"),
                "/../../ACKNOWLEDGEMENT.md"
            ),
        );
        if let Some(info) = self.engine.adapter_info() {
            ui.small(format!("{} / {}", info.name, info.backend));
        }
        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
    }
}

impl eframe::App for App {
    fn raw_input_hook(&mut self, _ctx: &egui::Context, input: &mut egui::RawInput) {
        if !self.verify_ui {
            return;
        }
        input
            .events
            .retain(|event| matches!(event, egui::Event::Screenshot { .. }));
        input.focused = true;
        let Some(frame) = self.frame.checked_sub(20) else {
            return;
        };
        let Some(rect) = self.interaction.viewport else {
            return;
        };
        let stroke = match frame {
            3..=5 if !self.verify_click => Some((-0.413, 3)),
            9..=11 if !self.verify_click => Some((0.537, 9)),
            13..=15 => Some((0.0, 13)),
            _ => None,
        };
        if let Some((x, start)) = stroke {
            let a = self.camera.project(Vec3::new(x, 1.3, 0.0), rect).0;
            let b = self.camera.project(Vec3::new(x + 0.12, 2.4, 0.0), rect).0;
            let point = if frame == start { a } else { b };
            input.events.push(egui::Event::PointerMoved(point));
            if frame != start + 1 {
                input.events.push(egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed: frame == start,
                    modifiers: Default::default(),
                });
            }
        }
        if self.verify_click {
            let click = match frame {
                3..=4 => Some((-0.413, 3)),
                9..=10 => Some((0.537, 9)),
                _ => None,
            };
            if let Some((x, start)) = click {
                let point = self
                    .camera
                    .project(Vec3::new(x, 1.8 + (x + 1.15) / 2.3 * 0.15, 0.0), rect)
                    .0;
                input.events.push(egui::Event::PointerMoved(point));
                input.events.push(egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed: frame == start,
                    modifiers: Default::default(),
                });
            }
        }
        let key = match frame {
            8 if self.verify_click => Some(if self.preset == 4 {
                egui::Key::C
            } else {
                egui::Key::B
            }),
            12 if self.verify_click => Some(egui::Key::C),
            15 => Some(egui::Key::Escape),
            16 => Some(egui::Key::P),
            20 => Some(egui::Key::C),
            21 => Some(egui::Key::Space),
            124 => Some(egui::Key::R),
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
        if (18..=19).contains(&frame)
            && let ActiveScene::Branch(branch) = &self.scene
        {
            let tip = branch
                .model
                .fibers()
                .iter()
                .max_by(|a, b| a.center().x.total_cmp(&b.center().x))
                .unwrap()
                .center();
            let point = self.camera.project(tip, rect).0;
            input.events.push(egui::Event::PointerMoved(point));
            input.events.push(egui::Event::PointerButton {
                pos: point,
                button: egui::PointerButton::Primary,
                pressed: frame == 18,
                modifiers: Default::default(),
            });
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        for event in ctx.input(|i| i.events.clone()) {
            if let egui::Event::Screenshot { image, .. } = event
                && let Some(path) = self.screenshot.take()
            {
                let bytes: Vec<_> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                match image::save_buffer(
                    &path,
                    &bytes,
                    image.width() as u32,
                    image.height() as u32,
                    image::ColorType::Rgba8,
                ) {
                    Ok(()) => {
                        eprintln!(
                            "Captured {path}; scene={} time={:.3} clearance={:.6}",
                            NAMES[self.preset],
                            self.scene.time(),
                            self.scene.clearance()
                        );
                        self.capture_saved = true;
                        if !self.verify_ui {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                    Err(error) => self.fail(error),
                }
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.paused = !self.paused;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.reset(self.preset);
        }
        for (i, key) in [
            egui::Key::Num1,
            egui::Key::Num2,
            egui::Key::Num3,
            egui::Key::Num4,
            egui::Key::Num5,
        ]
        .into_iter()
        .enumerate()
        {
            if ctx.input(|input| input.key_pressed(key)) {
                self.reset(i);
            }
        }
        for (key, tool) in [
            (egui::Key::C, Tool::Cut),
            (egui::Key::B, Tool::Break),
            (egui::Key::P, Tool::Push),
            (egui::Key::O, Tool::Orbit),
        ] {
            if ctx.input(|i| i.key_pressed(key)) {
                self.interaction.tool = tool;
            }
        }
        let elapsed = self.last_tick.elapsed().as_secs_f64().min(0.1);
        self.last_tick = Instant::now();
        if !self.paused && self.interaction.start.is_none() {
            if self.verify_ui || self.preset < 3 {
                self.advance(self.steps);
            } else {
                let dt = branch::DT;
                self.accumulated_time = (self.accumulated_time + elapsed).min(8.0 * dt);
                let count = (self.accumulated_time / dt).floor() as usize;
                let count = count.min(self.steps);
                if count > 0 {
                    self.advance(count);
                    self.accumulated_time -= count as f64 * dt;
                }
                self.simulation_rate =
                    0.9 * self.simulation_rate + 0.1 * (count as f64 * dt / elapsed.max(1e-6));
            }
        } else {
            self.accumulated_time = 0.0;
        }
        egui::Panel::left("controls")
            .exact_size(310.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        let mut action = None;
        egui::CentralPanel::default().show_inside(ui, |ui| match &self.scene {
            ActiveScene::Cable(s) => draw::paint(ui, s, &mut self.camera, self.nodes),
            ActiveScene::Branch(s) => {
                action =
                    branch_draw::paint(ui, s, &mut self.camera, &mut self.interaction, self.nodes)
            }
        });
        if let Some(action) = action
            && let ActiveScene::Branch(branch) = &mut self.scene
        {
            let result = match action {
                Action::Section(a, b, r, fracture) => {
                    branch.section_stroke(&self.engine, &self.camera, r, a, b, fracture)
                }
                Action::Click(p, r, fracture) => {
                    branch.click_section(&self.engine, &self.camera, r, p, fracture)
                }
                Action::Push(p, r) => {
                    branch.push_at(&self.engine, &self.camera, r, p, self.strength)
                }
                Action::GrabStart(..) | Action::GrabMove(..) | Action::GrabEnd => Ok(()),
            };
            if let Err(error) = result {
                self.fail(error);
            }
        }
        if self.verify_ui && self.frame == 40 {
            let ActiveScene::Branch(branch) = &self.scene else {
                panic!("wrong scene");
            };
            assert!(
                branch.message.starts_with("Pushed fiber"),
                "pointer push failed: {}",
                branch.message
            );
        }
        if self.verify_ui && self.frame == 139 {
            assert!(self.error.is_none(), "UI physics error: {:?}", self.error);
            let ActiveScene::Branch(branch) = &self.scene else {
                panic!("wrong scene");
            };
            assert_eq!(branch.cuts, if self.verify_click { 1 } else { 2 });
            assert_eq!(branch.breaks, usize::from(self.verify_click));
            assert_eq!(branch.model.components().len(), 3);
            assert!(branch.cap_count() >= 4 * branch::STRANDS);
            assert!(
                branch.clearance() > -0.005,
                "clipped cells penetrated the floor"
            );
            assert!(
                branch
                    .model
                    .fibers()
                    .iter()
                    .any(|f| f.pin.is_none() && f.center().y < 0.5),
                "detached pieces did not fall"
            );
            eprintln!(
                "PASS UI: preset={}, click_mode={}, two section operations, Escape cancellation, surface push, capped geometry and falling pieces",
                self.preset, self.verify_click
            );
        }
        if self.frame == if self.verify_ui { 140 } else { 120 } && self.screenshot.is_some() {
            self.paused = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        if self.verify_ui && self.frame == 148 {
            assert!(self.capture_saved, "framebuffer capture did not complete");
            let ActiveScene::Branch(branch) = &self.scene else {
                panic!("wrong scene");
            };
            assert_eq!(branch.cuts, 0);
            assert_eq!(branch.breaks, 0);
            assert_eq!(branch.cap_count(), 0);
            assert_eq!(
                branch.model.fibers().len(),
                branch::STRANDS * branch::SEGMENTS
            );
            assert_eq!(branch.model.components().len(), 1);
            eprintln!("PASS UI: keyboard reset restored original topology");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        self.frame += 1;
        ctx.request_repaint_after(std::time::Duration::from_millis(1));
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let engine = Engine::new_gpu()?;
    eprintln!("Cable Lab physics adapter: {:?}", engine.adapter_info());
    if args.iter().any(|a| a == "--bench-branches") {
        verify::latency(&engine)?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--verify") {
        verify::cables(&engine)?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--verify-branches") {
        verify::branches(&engine)?;
        return Ok(());
    }
    let value = |key: &str| {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let verify_click = args.iter().any(|a| a == "--verify-click-ui");
    let verify_ui = verify_click || args.iter().any(|a| a == "--verify-ui");
    let preset = if verify_ui {
        value("--scene")
            .map_or(Ok(3), |s| s.parse::<usize>())?
            .clamp(3, 4)
    } else {
        value("--scene")
            .map_or(Ok(0), |s| s.parse::<usize>())?
            .min(4)
    };
    let app = App {
        scene: ActiveScene::new(&engine, preset)?,
        engine,
        preset,
        camera: draw::Camera {
            distance: if preset >= 3 { 5.4 } else { 6.8 },
            ..Default::default()
        },
        interaction: Interaction {
            tool: if preset == 4 { Tool::Break } else { Tool::Cut },
            ..Default::default()
        },
        paused: verify_ui,
        nodes: false,
        steps: if verify_ui {
            2
        } else if preset >= 3 {
            8
        } else {
            4
        },
        step_ms: 0.0,
        strength: 1.5,
        error: None,
        screenshot: value("--screenshot")
            .or_else(|| verify_ui.then(|| "/tmp/molla-branch-ui.png".into())),
        frame: 0,
        verify_ui,
        verify_click,
        capture_saved: false,
        last_tick: Instant::now(),
        accumulated_time: 0.0,
        simulation_rate: 1.0,
    };
    let options = eframe::NativeOptions {
        depth_buffer: 24,
        multisampling: 4,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 840.0])
            .with_app_id("molla-cables"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "Molla — Cable Lab",
        options,
        Box::new(move |cc| {
            studio::initialize(cc.wgpu_render_state.as_ref().expect("wgpu renderer"));
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )?;
    Ok(())
}
