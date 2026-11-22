#[allow(dead_code)]
mod adaptive_step;
mod apple_tree_model;
#[allow(dead_code)]
mod apple_tree_sim;
#[allow(dead_code)]
mod branch;
#[allow(dead_code)]
mod branch_draw;
#[allow(dead_code)]
mod draw;
mod grab_handle;
mod leafy_draw;
mod leafy_model;
mod leafy_probe;
mod leafy_showcase;
#[allow(dead_code)]
mod scene;
mod foliage_layout;
mod studio;
mod video_capture;

use apple_tree_sim::{DT, Tree};
use branch_draw::{Action, Interaction, Tool};
use eframe::egui;
use molla_compute::Engine;
use molla_core::Result;
use molla_math::Vec3;
use molla_solvers::{
    attachments::{Anchor, Config, GpuAttachments, Particle},
    strands::GpuStrands,
};
use std::time::Instant;

struct Foliage {
    tree: Tree,
    leaves: Vec<leafy_model::Leaf>,
    coefficients: Vec<molla_solvers::attachments::Attachment>,
    particles: Vec<Particle>,
    gpu: GpuAttachments,
    twigs: Vec<u64>,
    strength: f64,
    wind: f64,
    selected: u64,
    shake: Option<(f64, Vec3)>,
    shake_angle: f64,
    shake_gain: f64,
}

impl Foliage {
    fn new(engine: &Engine) -> Result<Self> {
        let canopy = leafy_model::build()?;
        let gpu = GpuAttachments::new(engine, canopy.attachments.clone(), &canopy.particles)?;
        let mut tree = Tree::from_built(engine, canopy.tree)?;
        tree.retained_subdivisions = 64;
        tree.coarsen_batches = 120;
        Ok(Self {
            tree,
            leaves: canopy.leaves,
            coefficients: canopy.attachments,
            particles: canopy.particles,
            gpu,
            twigs: canopy.twigs,
            strength: 1.0,
            wind: 0.0,
            selected: 21,
            shake: None,
            shake_angle: 0.0,
            shake_gain: 1.0,
        })
    }

    fn attached(&self) -> usize {
        self.particles.iter().filter(|p| p.attached).count()
    }

    fn anchors(&mut self) -> Vec<Anchor> {
        self.leaves
            .iter_mut()
            .zip(&self.particles)
            .map(|(leaf, p)| {
                if self.tree.model.fiber(leaf.owner).is_none()
                    && let Some(f) = self
                        .tree
                        .model
                        .fibers()
                        .iter()
                        .find(|f| f.rest == leaf.rest && f.contains_local(leaf.anchor))
                {
                    leaf.owner = f.id;
                }
                if let Some(f) = self.tree.model.fiber(leaf.owner) {
                    let position = f.motion.pose.transform_point(leaf.anchor + leaf.offset);
                    if p.attached {
                        leaf.released_direction = f.motion.pose.rotation * leaf.direction;
                        leaf.released_normal = f.motion.pose.rotation * leaf.normal;
                    }
                    Anchor {
                        position,
                        velocity: f.motion.linear_velocity
                            + f.motion.angular_velocity.cross(position - f.center()),
                        valid: true,
                    }
                } else {
                    Anchor {
                        position: p.position,
                        velocity: p.velocity,
                        valid: false,
                    }
                }
            })
            .collect()
    }

    fn start_shake(&mut self) {
        let selected = self
            .tree
            .model
            .fiber(self.selected)
            .filter(|f| f.pin.is_none())
            .or_else(|| self.tree.model.fibers().iter().find(|f| f.pin.is_none()));
        if let Some(f) = selected {
            let id = f.id;
            let point = f.center();
            self.tree.start_grab(id, point);
            self.shake = Some((self.tree.time, point));
        }
    }

    fn advance(&mut self, engine: &Engine, count: usize) -> Result<()> {
        for _ in 0..count {
            if let Some((start, point)) = self.shake {
                let t = self.tree.time - start;
                if t >= 1.5 || self.tree.grab.is_none() {
                    self.shake = None;
                    self.tree.release_grab();
                } else {
                    let envelope = (std::f64::consts::PI * t / 1.5).sin();
                    self.tree.grab.as_mut().unwrap().target = point
                        + molla_math::Quat::from_rotation_y(self.shake_angle)
                            * Vec3::new(
                                0.17 * (t * 18.0).sin(),
                                0.17 * (t * 18.0).cos(),
                                0.08 * (t * 18.0).sin(),
                            )
                            * (envelope * self.shake_gain);
                }
            }
            let before = self.tree.model.clone();
            let old_particles = self.particles.clone();
            let anchors = self.anchors();
            self.gpu.step(
                engine,
                &anchors,
                Config {
                    dt: DT,
                    wind: Vec3::new(self.wind, 0.0, self.wind * 0.25),
                    strength_scale: self.strength,
                    ..Default::default()
                },
            )?;
            let next = self.gpu.snapshot(engine)?;
            let impulses: Vec<_> = next
                .iter()
                .zip(&self.leaves)
                .filter(|(p, _)| p.attached && p.force.length_squared() > 1e-16)
                .map(|(p, leaf)| {
                    (
                        leaf.owner,
                        self.tree
                            .model
                            .fiber(leaf.owner)
                            .unwrap()
                            .motion
                            .pose
                            .transform_point(leaf.anchor),
                        -p.force * DT,
                    )
                })
                .collect();
            let result = (|| {
                if !impulses.is_empty() {
                    self.tree.model = self.tree.gpu.impulses_snapshot(engine, &impulses)?;
                }
                self.tree.advance(engine, 1)
            })();
            if let Err(error) = result {
                self.gpu.restore(engine, &old_particles)?;
                self.tree.gpu =
                    GpuStrands::with_limits(engine, before.clone(), apple_tree_sim::gpu_limits())?;
                self.tree.model = before;
                return Err(error);
            }
            self.particles = next;
        }
        Ok(())
    }
}

struct App {
    engine: Engine,
    world: Foliage,
    camera: draw::Camera,
    interaction: Interaction,
    paused: bool,
    error: Option<String>,
    last: Instant,
    step_ms: f64,
    playback: f64,
    grab_plane: Option<(Vec3, Vec3)>,
    stress: bool,
    frame: usize,
    screenshot: Option<String>,
    verify_ui: bool,
    verify_pointer: Option<egui::Pos2>,
    captured: bool,
    close_on_capture: bool,
}

impl App {
    fn fail(&mut self, e: impl std::fmt::Display) {
        self.error = Some(e.to_string());
        self.paused = true;
        self.world.shake = None;
        self.world.tree.release_grab();
    }
    fn reset(&mut self) {
        match Foliage::new(&self.engine) {
            Ok(world) => {
                self.world = world;
                self.error = None;
                self.paused = false;
                self.grab_plane = None;
            }
            Err(e) => self.fail(e),
        }
    }
    fn action(&mut self, action: Action) -> Result<()> {
        match action {
            Action::Click(p, r, rough) => {
                self.world
                    .tree
                    .click(&self.engine, &self.camera, p, r, rough, false)?
            }
            Action::Push(p, r) => {
                self.paused = false;
                self.world
                    .tree
                    .click(&self.engine, &self.camera, p, r, false, true)?;
            }
            Action::Section(a, b, r, rough) => {
                let (plane, ids) =
                    branch::blade_selection(&self.world.tree.model, &self.camera, r, a, b)?;
                self.world.tree.section(&self.engine, plane, &ids, rough)?;
            }
            Action::GrabStart(p, r) => {
                let (o, d) = self.camera.ray(p, r);
                if let Some((id, point)) = branch::ray_hit(&self.world.tree.model, o, d)
                    && self.world.tree.model.fiber(id).unwrap().pin.is_none()
                {
                    self.world.selected = id;
                    self.world.shake = None;
                    self.world.tree.start_grab(id, point);
                    self.paused = false;
                    self.grab_plane = Some((point, self.camera.ray(r.center(), r).1));
                }
            }
            Action::GrabMove(p, r) => {
                if let (Some(grab), Some((point, n))) = (&mut self.world.tree.grab, self.grab_plane)
                {
                    let (o, d) = self.camera.ray(p, r);
                    let denom = n.dot(d);
                    if denom.abs() > 1e-6 {
                        let t = n.dot(point - o) / denom;
                        if t > 0.0 {
                            grab.target = o + d * t;
                        }
                    }
                }
            }
            Action::GrabEnd => {
                if self.world.shake.is_none() {
                    self.world.tree.release_grab();
                    self.grab_plane = None;
                }
            }
        }
        self.world.anchors();
        Ok(())
    }
}

impl eframe::App for App {
    fn raw_input_hook(&mut self, _: &egui::Context, input: &mut egui::RawInput) {
        if !self.verify_ui {
            return;
        }
        input.events.retain(|event| {
            !matches!(
                event,
                egui::Event::PointerMoved(_)
                    | egui::Event::PointerButton { .. }
                    | egui::Event::PointerGone
                    | egui::Event::Key { .. }
                    | egui::Event::MouseWheel { .. }
            )
        });
        input.focused = true;
        if self.frame == 25 || self.frame == 260 {
            let key = if self.frame == 25 {
                egui::Key::S
            } else {
                egui::Key::C
            };
            input.events.push(egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Default::default(),
            });
            input.events.push(egui::Event::Key {
                key,
                physical_key: None,
                pressed: false,
                repeat: false,
                modifiers: Default::default(),
            });
        }
        if matches!(self.frame, 270 | 271 | 272)
            && let Some(rect) = self.interaction.viewport
        {
            let p = *self
                .verify_pointer
                .get_or_insert_with(|| self.world.tree.visible_limb(&self.camera, rect));
            input.events.push(egui::Event::PointerMoved(p));
            if self.frame != 270 {
                input.events.push(egui::Event::PointerButton {
                    pos: p,
                    button: egui::PointerButton::Primary,
                    pressed: self.frame == 271,
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
                .unwrap();
                self.captured = true;
                eprintln!("Captured {path}");
                if self.verify_ui {
                    assert!(self.error.is_none(), "leafy UI error: {:?}", self.error);
                    assert!(
                        self.world.attached() < self.world.leaves.len(),
                        "shake shed no leaves"
                    );
                    assert_eq!(self.world.tree.cuts, 1, "native cut missed");
                    eprintln!(
                        "PASS leafy native UI: shake, shed {}, exact cut, framebuffer",
                        self.world.leaves.len() - self.world.attached()
                    );
                }
                if self.close_on_capture {
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
        if ctx.input(|i| i.key_pressed(egui::Key::S)) {
            self.world.start_shake();
            self.paused = false;
        }
        for (key, tool) in [
            (egui::Key::G, Tool::Grab),
            (egui::Key::C, Tool::Cut),
            (egui::Key::B, Tool::Break),
            (egui::Key::O, Tool::Orbit),
        ] {
            if ctx.input(|i| i.key_pressed(key)) {
                self.interaction.tool = tool;
                self.world.shake = None;
                self.world.tree.release_grab();
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape) || !i.focused) && !self.verify_ui {
            self.world.tree.release_grab();
            self.world.shake = None;
            self.grab_plane = None;
        }
        let elapsed = self.last.elapsed().as_secs_f64().min(0.1);
        self.last = Instant::now();
        if !self.paused && self.interaction.start.is_none() {
            let steps = if self.verify_ui {
                4
            } else {
                (elapsed / DT).ceil().clamp(1.0, 4.0) as usize
            };
            let now = Instant::now();
            let mut completed = 0;
            for _ in 0..steps {
                if let Err(e) = self.world.advance(&self.engine, 1) {
                    self.fail(e);
                    break;
                }
                completed += 1;
                if !self.verify_ui && now.elapsed().as_secs_f64() >= 0.008 {
                    break;
                }
            }
            self.step_ms = now.elapsed().as_secs_f64() * 1000.0;
            self.playback = 0.9 * self.playback + 0.1 * (completed as f64 * DT / elapsed.max(DT));
        }
        egui::Panel::left("leafy controls").exact_size(260.0).show_inside(ui,|ui|{
            ui.add_space(14.0);ui.heading("MOLLA / CANOPY");ui.label("alpha 0.0.1 · apple tree");ui.separator();
            ui.label(format!("{} cuttable twigs · {} leaves",self.world.twigs.len(),self.world.leaves.len()));
            ui.label(format!("Attached: {}   Fallen: {}",self.world.attached(),self.world.leaves.len()-self.world.attached()));
            ui.add_space(10.0);
            for (tool,label) in [(Tool::Grab,"Grab / bend [G]"),(Tool::Cut,"Clean cut [C]"),(Tool::Break,"Rough break [B]"),(Tool::Orbit,"Orbit [O]")] {
                if ui.selectable_label(self.interaction.tool==tool,label).clicked(){self.interaction.tool=tool;self.world.shake=None;self.world.tree.release_grab();}
            }
            ui.separator();
            if ui.button("Shake selected branch [S]").clicked(){self.world.start_shake();self.paused=false;}
            ui.label("Grab wood to select a branch.");
            ui.add(egui::Slider::new(&mut self.world.tree.grip_force,1.0..=12.0).text("Grip / N"));
            ui.add(egui::Slider::new(&mut self.world.strength,0.3..=3.0).text("Leaf strength"));
            ui.add(egui::Slider::new(&mut self.world.wind,0.0..=6.0).text("Wind / m/s"));
            ui.checkbox(&mut self.stress,"Show attachment load");
            let minimum=self.world.coefficients.iter().map(|a|a.strength).fold(f64::INFINITY,f64::min)*self.world.strength*1000.0;
            let maximum=self.world.coefficients.iter().map(|a|a.strength).fold(0.0,f64::max)*self.world.strength*1000.0;
            ui.label(format!("Individual strengths: {minimum:.1}–{maximum:.1} mN"));
            ui.separator();
            if ui.button(if self.paused {"Play [Space]"}else{"Pause [Space]"}).clicked(){self.paused=!self.paused;}
            if ui.button("Reset tree + leaves [R]").clicked(){self.reset();}
            ui.label("Right-drag: orbit · wheel: zoom");
            ui.add_space(10.0);ui.small(format!("GPU wood + leaves: {:.1} ms\nSimulation speed: {:.2}×",self.step_ms,self.playback));
            ui.small("Illustrative strengths; not calibrated wood.\nGPU point-mass leaves, drag + floor.");
            if let Some(error)=&self.error {ui.colored_label(egui::Color32::LIGHT_RED,error);}
            ui.add_space(8.0);ui.small("Tree: AppleQSM / Appleskel\nLeaf geometry: adapted from Appleskel\nFull citations: ACKNOWLEDGEMENT.md");
        });
        let mut action = None;
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let mesh = leafy_draw::mesh(
                &self.world.leaves,
                &self.world.particles,
                &self.world.tree.model,
                self.stress,
                &self.world.coefficients,
                self.world.strength,
            );
            action = branch_draw::paint_foliage(
                ui,
                &self.world.tree.model,
                &mut self.camera,
                &mut self.interaction,
                mesh,
            );
            if let (Some(grab), Some(handle), Some(rect)) = (
                self.world.tree.grab,
                self.world.tree.handle,
                self.interaction.viewport,
            ) && let Some(f) = self.world.tree.model.fiber(grab.fiber)
            {
                let a = self
                    .camera
                    .project(f.motion.pose.transform_point(grab.local_point), rect)
                    .0;
                let b = self.camera.project(handle.point, rect).0;
                ui.painter()
                    .line_segment([a, b], egui::Stroke::new(2.0, egui::Color32::GOLD));
                ui.painter()
                    .circle_stroke(b, 7.0, egui::Stroke::new(2.0, egui::Color32::GOLD));
            }
        });
        if let Some(action) = action
            && let Err(e) = self.action(action)
        {
            self.fail(e);
        }
        let capture_frame = if self.verify_ui { 430 } else { 15 };
        if self.frame == capture_frame && self.screenshot.is_some() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        self.frame += 1;
        ctx.request_repaint_after(std::time::Duration::from_millis(8));
    }
}

fn verify(engine: &Engine) -> Result<()> {
    let mut world = Foliage::new(engine)?;
    let count = world.leaves.len();
    let root = world.tree.model.fibers()[0].motion.pose;
    eprintln!(
        "Leafy tree: {} wood cells, {} twigs, {} leaves",
        world.tree.model.fibers().len(),
        world.twigs.len(),
        count
    );
    let start = Instant::now();
    for _ in 0..120 {
        world.advance(engine, 1)?;
    }
    for (i, ((leaf, p), a)) in world
        .leaves
        .iter()
        .zip(&world.particles)
        .zip(&world.coefficients)
        .enumerate()
    {
        if !p.attached {
            eprintln!(
                "IDLE DETACH {i}: twig={} force={} strength={}",
                leaf.owner, p.peak_force, a.strength
            );
        }
    }
    assert_eq!(world.attached(), count, "idle leaves detached");
    assert_eq!(world.tree.stress_breaks, 0, "idle wood broke");
    eprintln!(
        "LEAFY_BENCH step_ms={:.3}, cells={}, leaves={}",
        start.elapsed().as_secs_f64() * 1000.0 / 120.0,
        world.tree.model.fibers().len(),
        count
    );
    world.start_shake();
    for _ in 0..480 {
        world.advance(engine, 1)?;
    }
    let detached = count - world.attached();
    let mut peaks: Vec<_> = world
        .particles
        .iter()
        .zip(&world.coefficients)
        .enumerate()
        .map(|(i, (p, a))| (p.peak_force / a.strength, i, p.peak_force, a.strength))
        .collect();
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!("Leaf load peaks: {:?}", &peaks[..8]);
    eprintln!(
        "Leafy shake: detached={detached}, wood failures={}",
        world.tree.stress_breaks
    );
    assert!(
        detached > 0 && detached * 20 < count,
        "strong attachments must retain over 95% of leaves during the standard shake"
    );
    assert_eq!(world.tree.model.fibers()[0].motion.pose, root);
    let id = world
        .leaves
        .iter()
        .zip(&world.particles)
        .find(|(_, p)| p.attached)
        .unwrap()
        .0
        .owner;
    let f = world.tree.model.fiber(id).unwrap();
    let plane =
        molla_geometry::convex_cut::CutPlane::new(f.center(), f.motion.pose.rotation * Vec3::Z)?;
    let states = world.particles.clone();
    world.tree.section(engine, plane, &[id], false)?;
    world.anchors();
    assert_eq!(states, world.particles, "cut teleported leaves");
    assert!(
        world
            .leaves
            .iter()
            .zip(&world.particles)
            .filter(|(_, p)| p.attached)
            .all(|(leaf, _)| world.tree.model.fiber(leaf.owner).is_some())
    );
    for _ in 0..480 {
        world.advance(engine, 1)?;
    }
    let id = world
        .leaves
        .iter()
        .zip(&world.particles)
        .find(|(leaf, p)| p.attached && world.tree.model.fiber(leaf.owner).is_some())
        .unwrap()
        .0
        .owner;
    let f = world.tree.model.fiber(id).unwrap();
    let plane =
        molla_geometry::convex_cut::CutPlane::new(f.center(), f.motion.pose.rotation * Vec3::Z)?;
    let states = world.particles.clone();
    world.tree.section(engine, plane, &[id], true)?;
    world.anchors();
    assert_eq!(states, world.particles, "break teleported leaves");
    for _ in 0..180 {
        world.advance(engine, 1)?;
    }
    assert!(
        world
            .particles
            .iter()
            .all(|p| p.position.is_finite() && p.position.y >= 0.0019)
    );
    eprintln!(
        "PASS leafy tree: rest, physical shake, heterogeneous shedding, cut/break/remap continuity, finite state, floor"
    );
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let value = |key: &str| args.windows(2).find(|v| v[0] == key).map(|v| v[1].clone());
    let engine = Engine::new_gpu()?;
    eprintln!("Leafy physics adapter: {:?}", engine.adapter_info());
    if args.iter().any(|a| a == "--verify") {
        verify(&engine)?;
        return Ok(());
    }
    for (flag, record) in [("--showcase-video", true), ("--verify-showcase", false)] {
        if let Some(path) = value(flag) {
            leafy_showcase::run(&engine, std::path::Path::new(&path), record, false)?;
            return Ok(());
        }
    }
    for (flag, record) in [("--detail-video", true), ("--verify-detail", false)] {
        if let Some(path) = value(flag) {
            leafy_showcase::run_detail(&engine, std::path::Path::new(&path), record)?;
            return Ok(());
        }
    }
    if let Some(path) = value("--capture-smoke") {
        leafy_showcase::smoke(&engine, std::path::Path::new(&path))?;
        return Ok(());
    }
    if let Some(path) = value("--fracture-preview") {
        leafy_showcase::fracture_preview(&engine, std::path::Path::new(&path))?;
        return Ok(());
    }
    if let Some(path) = value("--verify-intro") {
        leafy_showcase::run(&engine, std::path::Path::new(&path), false, true)?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--probe-release") {
        leafy_probe::run(&engine)?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--verify-matrix") {
        for (id, angle) in [(11, 0.0), (21, 1.7), (37, 3.1), (54, 4.7)] {
            let mut world = Foliage::new(&engine)?;
            world.selected = id;
            world.shake_angle = angle;
            for _ in 0..120 {
                world.advance(&engine, 1)?;
            }
            assert_eq!(world.attached(), world.leaves.len());
            world.start_shake();
            for _ in 0..480 {
                world.advance(&engine, 1)?;
            }
            assert!(
                world
                    .particles
                    .iter()
                    .all(|p| p.position.is_finite() && p.position.y >= 0.0019)
            );
            eprintln!(
                "PASS leafy shake branch={id}, angle={angle}, detached={}, wood failures={}",
                world.leaves.len() - world.attached(),
                world.tree.stress_breaks
            );
        }
        return Ok(());
    }
    let verify_ui = args.iter().any(|a| a == "--verify-ui");
    let capture_live = value("--capture-live");
    let close_on_capture = capture_live.is_none();
    let screenshot = capture_live
        .or_else(|| value("--screenshot"))
        .or_else(|| verify_ui.then(|| "/tmp/molla-leafy-interaction.png".into()));
    let world = Foliage::new(&engine)?;
    let app = App {
        engine,
        world,
        camera: draw::Camera {
            yaw: 0.65,
            pitch: 0.16,
            distance: 4.5,
            ..Default::default()
        },
        interaction: Interaction {
            tool: Tool::Grab,
            ..Default::default()
        },
        paused: screenshot.is_some() && !verify_ui && close_on_capture,
        error: None,
        last: Instant::now(),
        step_ms: 0.0,
        playback: 1.0,
        grab_plane: None,
        stress: false,
        frame: 0,
        screenshot,
        verify_ui,
        verify_pointer: None,
        captured: false,
        close_on_capture,
    };
    eframe::run_native(
        "Molla · Leafy apple tree — full canopy",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 960.0]),
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
