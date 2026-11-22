use crate::{Foliage, branch_draw, draw, leafy_draw, leafy_model, video_capture};
use eframe::egui::{self, Align2, Color32, FontId};
use molla_compute::Engine;
use molla_geometry::convex_cut::CutPlane;
use molla_math::{Transform, Vec3};
use molla_solvers::strands::{Fiber, GpuStrands, StrandModel};
use std::path::Path;

#[path = "leafy_detail.rs"]
mod detail;

fn camera() -> draw::Camera {
    draw::Camera {
        yaw: 0.65,
        pitch: 0.16,
        distance: 4.7,
        ..Default::default()
    }
}

#[derive(Clone)]
struct Mark {
    rest: Transform,
    local: Vec3,
    point: Vec3,
    rough: bool,
}

fn paint(ui: &mut egui::Ui, world: &Foliage, camera: &mut draw::Camera, marks: &[Mark]) {
    let tree = &world.tree;
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(Color32::from_rgb(15, 21, 31)))
        .show_inside(ui, |ui| {
            let scene = ui.max_rect();
            let foliage = leafy_draw::mesh(
                &world.leaves,
                &world.particles,
                &tree.model,
                false,
                &world.coefficients,
                world.strength,
            );
            branch_draw::paint_cinematic_foliage(ui, &tree.model, camera, foliage);
            let painter = ui.painter().clone();
            for mark in marks {
                let point = tree
                    .model
                    .fibers()
                    .iter()
                    .find(|f| f.rest == mark.rest && f.contains_local(mark.local))
                    .map_or(mark.point, |f| f.motion.pose.transform_point(mark.local));
                let (p, depth) = camera.project(point, scene);
                if depth <= 0.0 || !scene.contains(p) {
                    continue;
                }
                let color = if mark.rough {
                    Color32::from_rgb(255, 119, 99)
                } else {
                    Color32::from_rgb(85, 225, 205)
                };
                painter.circle_stroke(p, 12.0, egui::Stroke::new(2.0, color));
                let line = if mark.rough {
                    vec![
                        p + egui::vec2(-28.0, 10.0),
                        p + egui::vec2(-6.0, -8.0),
                        p + egui::vec2(4.0, 8.0),
                        p + egui::vec2(28.0, -10.0),
                    ]
                } else {
                    vec![p + egui::vec2(-28.0, 10.0), p + egui::vec2(28.0, -10.0)]
                };
                painter.add(egui::Shape::line(line, egui::Stroke::new(2.5, color)));
                painter.text(
                    p + egui::vec2(20.0, -25.0),
                    Align2::LEFT_BOTTOM,
                    if mark.rough { "BREAK" } else { "CUT" },
                    FontId::monospace(17.0),
                    color,
                );
            }
            if let (Some(grab), Some(handle)) = (tree.grab, tree.handle) {
                let point = tree
                    .model
                    .fiber(grab.fiber)
                    .unwrap()
                    .motion
                    .pose
                    .transform_point(grab.local_point);
                let (a, _) = camera.project(point, scene);
                let (b, _) = camera.project(handle.point, scene);
                let color = Color32::from_rgb(255, 191, 100);
                painter.line_segment([a, b], egui::Stroke::new(1.5, color));
                painter.circle_stroke(b, 8.0, egui::Stroke::new(2.0, color));
            }
            painter.text(
                scene.right_top() + egui::vec2(-32.0, 26.0),
                Align2::RIGHT_TOP,
                format!("molla  alpha {}", env!("CARGO_PKG_VERSION")),
                FontId::monospace(22.0),
                Color32::from_white_alpha(185),
            );
        });
}

pub fn smoke(engine: &Engine, path: &Path) -> video_capture::Result<()> {
    let tree = Foliage::new(engine)?;
    let mut capture = video_capture::Capture::new(path)?;
    let mut camera = camera();
    camera.distance = 4.7;
    for i in 0..30 {
        capture.frame(
            |ui| paint(ui, &tree, &mut camera, &[]),
            (i == 15).then(|| path.with_extension("png")).as_deref(),
        )?;
    }
    capture.finish()
}

struct Movie {
    world: Foliage,
    camera: draw::Camera,
    capture: Option<video_capture::Capture>,
    preview: Option<video_capture::Capture>,
    path: std::path::PathBuf,
    frame: u64,
    marks: Vec<Mark>,
    events: Vec<serde_json::Value>,
    initial_mass: f64,
}

impl Movie {
    fn event(&mut self, kind: &str, data: serde_json::Value) {
        eprintln!(
            "SHOWCASE {kind} at frame {} / t={:.6}: {data}",
            self.frame, self.world.tree.time
        );
        self.events
            .push(serde_json::json!({"kind":kind, "frame":self.frame,
            "video_seconds":self.frame as f64 / 30.0,"simulation_seconds":self.world.tree.time,
            "attached_leaves":self.world.attached(),"data":data}));
    }

    fn frame(
        &mut self,
        engine: &Engine,
        steps: usize,
        still: Option<&str>,
    ) -> video_capture::Result<()> {
        if steps > 0 {
            self.world.advance(engine, steps)?;
        }
        if let Some(capture) = &mut self.capture {
            let still = still.map(|name| self.path.with_file_name(format!("{name}.png")));
            capture.frame(
                |ui| paint(ui, &self.world, &mut self.camera, &self.marks),
                still.as_deref(),
            )?;
        } else if let (Some(capture), Some(name)) = (&mut self.preview, still) {
            let still = self.path.with_file_name(format!("{name}.png"));
            capture.frame(
                |ui| paint(ui, &self.world, &mut self.camera, &self.marks),
                Some(&still),
            )?;
        }
        self.frame += 1;
        if self.frame % 150 == 0 {
            eprintln!(
                "Showcase frame {} / t={:.2} / {} cells",
                self.frame,
                self.world.tree.time,
                self.world.tree.model.fibers().len()
            );
        }
        Ok(())
    }

    fn frames(&mut self, engine: &Engine, count: usize) -> video_capture::Result<()> {
        for _ in 0..count {
            self.frame(engine, 4, None)?;
        }
        Ok(())
    }

    fn camera_to(
        &mut self,
        engine: &Engine,
        target: Vec3,
        distance: f64,
        yaw: f64,
        frames: usize,
    ) -> video_capture::Result<()> {
        let start = self.camera;
        for i in 0..frames {
            let u = (i + 1) as f64 / frames as f64;
            let t = u * u * (3.0 - 2.0 * u);
            self.camera.target = start.target.lerp(target, t);
            self.camera.distance = start.distance + (distance - start.distance) * t;
            self.camera.yaw = start.yaw + (yaw - start.yaw) * t;
            self.frame(engine, 4, None)?;
        }
        Ok(())
    }

    fn check(&self) -> video_capture::Result<()> {
        if !self
            .world
            .particles
            .iter()
            .all(|p| p.position.is_finite() && p.velocity.is_finite() && p.position.y >= 0.0019)
        {
            return Err("leafy showcase particle/floor invariant".into());
        }
        let mass: f64 = self
            .world
            .tree
            .model
            .fibers()
            .iter()
            .map(|f| f.mass.mass)
            .sum();
        if (mass - self.initial_mass).abs() > 1e-8 || self.world.tree.clearance() < -0.01 {
            return Err("showcase mass/floor invariant".into());
        }
        for f in self.world.tree.model.fibers() {
            f.validate()?;
        }
        Ok(())
    }

    fn cue(
        &mut self,
        engine: &Engine,
        rest: Transform,
        local: Vec3,
        rough: bool,
        frames: usize,
    ) -> video_capture::Result<()> {
        let f = material_span(&self.world.tree.model, rest, local)
            .ok_or("tool target is not attached")?;
        self.marks = vec![Mark {
            rest,
            local,
            point: f.motion.pose.transform_point(local),
            rough,
        }];
        self.frames(engine, frames)
    }

    fn rough_break(&mut self, engine: &Engine, id: u64, cue: bool) -> video_capture::Result<()> {
        let f = self
            .world
            .tree
            .model
            .fiber(id)
            .ok_or("rough-break span missing")?;
        let rest = f.rest;
        let local = f.mass.local_com;
        if cue {
            self.cue(engine, rest, local, true, 24)?;
        }
        let f = material_span(&self.world.tree.model, rest, local)
            .ok_or("rough-break span detached")?;
        let point = f.motion.pose.transform_point(local);
        let plane = CutPlane::new(point, f.motion.pose.rotation * Vec3::Z)?;
        let id = f.id;
        self.world.tree.section(engine, plane, &[id], true)?;
        self.event(
            "rough_break",
            serde_json::json!({"span":id,"point":point.to_array()}),
        );
        self.marks.clear();
        Ok(())
    }
}

fn extent(fiber: &Fiber) -> (f64, f64) {
    fiber
        .shape
        .vertices()
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
            (a.min(p.z), b.max(p.z))
        })
}

fn rooted(model: &StrandModel) -> Vec<u64> {
    model
        .components()
        .into_iter()
        .find(|c| c.iter().any(|&id| model.fiber(id).unwrap().pin.is_some()))
        .unwrap()
}

fn material_span(model: &StrandModel, rest: Transform, local: Vec3) -> Option<&Fiber> {
    let root = rooted(model);
    model
        .fibers()
        .iter()
        .find(|f| f.rest == rest && root.contains(&f.id) && f.contains_local(local))
}

fn trunk_components(model: &StrandModel, original: &[Transform]) -> usize {
    model
        .components()
        .iter()
        .filter(|c| {
            c.iter()
                .any(|&id| original.contains(&model.fiber(id).unwrap().rest))
        })
        .count()
}

fn collar_z(original: &Fiber, trunk: &[&Fiber]) -> f64 {
    let (lo, hi) = extent(original);
    let inside = |z| {
        let p = original.rest.transform_point(Vec3::Z * z);
        trunk
            .iter()
            .any(|f| f.contains_local(f.rest.inverse_transform_point(p)))
    };
    let mut low = lo;
    let mut high = hi;
    for _ in 0..40 {
        let middle = (low + high) * 0.5;
        if inside(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    (high + 0.001).max(lo + 0.008).min(hi - 0.03)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_collars_cross_only_their_own_root_material() {
        let built = leafy_model::build().unwrap().tree;
        let trunk: Vec<_> = built
            .trunk_cells
            .iter()
            .map(|&id| built.model.fiber(id).unwrap())
            .collect();
        assert!(trunk.len() > 5);
        for b in built.model.bonds() {
            let [a, c] = b.fibers;
            if built.trunk_cells.contains(&a) && !built.trunk_cells.contains(&c) {
                let f = built.model.fiber(c).unwrap();
                let z = collar_z(f, &trunk);
                let (lo, hi) = extent(f);
                assert!(z > lo && z < hi);
                assert!(f.contains_local(Vec3::Z * z));
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan and FFmpeg"]
    fn watermark_is_present_in_captured_frame() {
        let engine = Engine::new_gpu().unwrap();
        let directory =
            std::env::temp_dir().join(format!("molla-leafy-overlay-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("overlay.mp4");
        smoke(&engine, &path).unwrap();
        let pixels = image::open(path.with_extension("png")).unwrap().to_rgba8();
        let bright = pixels
            .enumerate_pixels()
            .filter(|(x, y, p)| {
                *x > 1600
                    && *x < 1900
                    && *y > 20
                    && *y < 65
                    && p[0] > 120
                    && p[1] > 120
                    && p[2] > 120
            })
            .count();
        assert!(
            bright > 250,
            "version watermark is missing: {bright} bright pixels"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}

pub fn run(
    engine: &Engine,
    path: &Path,
    record: bool,
    intro_only: bool,
) -> video_capture::Result<()> {
    run_variant(engine, path, record, intro_only, false)
}

pub fn run_detail(engine: &Engine, path: &Path, record: bool) -> video_capture::Result<()> {
    run_variant(engine, path, record, false, true)
}

pub fn fracture_preview(engine: &Engine, path: &Path) -> video_capture::Result<()> {
    let mut world = Foliage::new(engine)?;
    world.advance(engine, 180)?;
    let f = world.tree.model.fiber(32).unwrap();
    let point = f.center();
    let axis = f.motion.pose.rotation * Vec3::Z;
    let mut camera = camera();
    camera.target = point;
    camera.distance = 0.32;
    camera.yaw = axis.x.atan2(axis.z) + 0.7;
    world
        .tree
        .section(engine, CutPlane::new(point, axis)?, &[32], true)?;
    let mut capture = video_capture::Capture::new(path)?;
    for frame in 0..30 {
        world.advance(engine, 4)?;
        let still = path.with_file_name(format!("fracture-preview-{frame}.png"));
        capture.frame(
            |ui| paint(ui, &world, &mut camera, &[]),
            [0, 8, 15, 29].contains(&frame).then_some(still.as_path()),
        )?;
    }
    capture.finish()
}

fn run_variant(
    engine: &Engine,
    path: &Path,
    record: bool,
    intro_only: bool,
    detailed: bool,
) -> video_capture::Result<()> {
    let source = leafy_model::build()?.tree;
    let trunk_ids = source.trunk_cells;
    let initial = source.model;
    let trunk_rest: Vec<_> = trunk_ids
        .iter()
        .map(|&id| initial.fiber(id).unwrap().rest)
        .collect();
    let initial_mass = initial.fibers().iter().map(|f| f.mass.mass).sum();
    let mut movie = Movie {
        world: Foliage::new(engine)?,
        camera: camera(),
        capture: if record {
            Some(video_capture::Capture::new(path)?)
        } else {
            None
        },
        preview: if !record {
            std::env::var_os("MOLLA_SHOWCASE_PREVIEW")
                .map(|p| video_capture::Capture::new(Path::new(&p)))
                .transpose()?
        } else {
            None
        },
        path: path.into(),
        frame: 0,
        marks: Vec::new(),
        events: Vec::new(),
        initial_mass,
    };
    movie.camera.distance = 4.7;
    for frame in 0..60 {
        movie.frame(engine, 0, (frame == 15).then_some("cover"))?;
    }

    movie.camera_to(engine, Vec3::new(0.0, 1.0, 0.0), 3.6, 0.65, 45)?;
    let before_shaking = movie.world.attached();
    movie.world.shake_gain = 1.5;
    for (index, (branch, angle)) in [(21, 0.0), (11, std::f64::consts::PI)]
        .into_iter()
        .enumerate()
    {
        movie.world.selected = branch;
        movie.world.shake_angle = angle;
        movie.world.start_shake();
        movie.event(
            "shake_started",
            serde_json::json!({"branch":branch,"angle":angle,"target_amplitude_scale":movie.world.shake_gain}),
        );
        for frame in 0..150 {
            movie.frame(
                engine,
                4,
                (index == 1 && frame == 100).then_some("00-shaking-leaves"),
            )?;
        }
        movie.event("shake_finished", serde_json::json!({"branch":branch}));
        movie.check()?;
    }
    let shed = before_shaking - movie.world.attached();
    movie.world.shake_gain = 1.0;
    if !(5..=60).contains(&shed)
        || movie.world.tree.stress_breaks != 0
        || movie.world.tree.grab.is_some()
    {
        return Err(format!(
            "opening shake must shed a few leaves without breaking wood: shed={shed}, wood={}",
            movie.world.tree.stress_breaks
        )
        .into());
    }
    movie.event(
        "opening_shake_complete",
        serde_json::json!({"shed":shed,"wood_failures":0}),
    );
    if intro_only {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&movie.events)?)?;
        if let Some(capture) = movie.preview.take() {
            capture.finish()?;
        }
        eprintln!("PASS leafy opening: {shed} leaves shed, no wood failure, both handles released");
        return Ok(());
    }

    let load_steps = 86;
    let release_batches = 30;
    let break_span = 44;
    let hit = movie.world.tree.model.fiber(49).unwrap().center();
    movie.world.tree.start_grab(49, hit);
    let start = movie.camera;
    for step in 0..load_steps {
        let u = (step as f64 / 65.0).min(1.0);
        let t = u * u * (3.0 - 2.0 * u);
        movie.camera.target = start.target.lerp(Vec3::new(0.1, 0.85, 0.3), t);
        movie.camera.distance = start.distance + (2.9 - start.distance) * t;
        movie.camera.yaw = start.yaw + (0.85 - start.yaw) * t;
        movie.world.tree.grab.as_mut().unwrap().target = hit - Vec3::Z * (step as f64 / 240.0);
        movie.frame(engine, 4, None)?;
    }
    movie.world.tree.release_grab();
    let release_time = movie.world.tree.time;
    let damage = movie
        .world
        .tree
        .model
        .bonds()
        .iter()
        .find(|b| b.id == 48)
        .unwrap()
        .damage;
    movie.event(
        "grab_released",
        serde_json::json!({"span":49,"damage":damage}),
    );
    if detailed {
        let f = movie.world.tree.model.fiber(break_span).unwrap();
        let point = f.center();
        movie.marks = vec![Mark {
            rest: f.rest,
            local: f.mass.local_com,
            point,
            rough: true,
        }];
        movie.camera_to(engine, point, 0.55, 0.85, release_batches)?;
        movie.event(
            "fracture_closeup",
            serde_json::json!({"span":break_span,"distance_m":0.55}),
        );
    } else {
        movie.frames(engine, release_batches - 24)?;
        let f = movie.world.tree.model.fiber(break_span).unwrap();
        movie.cue(engine, f.rest, f.mass.local_com, true, 24)?;
    }
    if movie.world.tree.stress_breaks != 0 {
        return Err("weakened branch broke before the separate operation".into());
    }
    let mut control = Foliage::new(engine)?;
    control.tree.model = movie.world.tree.model.clone();
    control.tree.gpu = GpuStrands::with_limits(
        engine,
        control.tree.model.clone(),
        crate::apple_tree_sim::gpu_limits(),
    )?;
    control.leaves = movie.world.leaves.clone();
    control.particles = movie.world.particles.clone();
    control.gpu.restore(engine, &control.particles)?;
    for _ in 0..180 {
        control.advance(engine, 4)?;
    }
    if control.tree.stress_breaks != 0 {
        return Err("unedited post-release control failed".into());
    }
    movie.event("released_control",serde_json::json!({"release_to_break_seconds":movie.world.tree.time-release_time,"control_seconds":3,"failures":0}));
    movie.rough_break(engine, break_span, false)?;
    let break_time = movie.world.tree.time;
    let mut delayed = None;
    let close = movie.camera;
    for tick in 0..180 {
        if detailed && tick >= 12 {
            let u = ((tick - 12) as f64 / 32.0).min(1.0);
            let t = u * u * (3.0 - 2.0 * u);
            movie.camera.target = close.target.lerp(Vec3::new(0.1, 0.85, 0.3), t);
            movie.camera.distance = close.distance + (2.9 - close.distance) * t;
        }
        let still = if detailed && tick == 8 {
            Some("break-44-fibers")
        } else {
            (tick == 70).then_some("01-first-break")
        };
        movie.frame(engine, 4, still)?;
        if delayed.is_none()
            && movie
                .world
                .tree
                .model
                .bonds()
                .iter()
                .find(|b| b.id == 48)
                .is_some_and(|b| !b.intact())
        {
            delayed = Some(movie.world.tree.time - break_time);
            movie.event(
                "delayed_load_failure",
                serde_json::json!({"bond":48,"delay_seconds":delayed}),
            );
        }
    }
    movie.event(
        "delayed_failure_observation",
        serde_json::json!({"occurred":delayed.is_some()}),
    );
    if delayed.is_none() {
        return Err("delayed leafy failure did not reproduce after release".into());
    }
    if detailed {
        return detail::finish(movie, engine, &initial, &trunk_ids, shed);
    }
    movie.camera_to(engine, Vec3::new(0.0, 1.2, 0.0), 4.7, 0.6, 45)?;
    for id in [32, 41, 56] {
        movie.rough_break(engine, id, true)?;
        movie.frames(engine, 120)?;
        movie.check()?;
    }
    movie.frame(engine, 4, Some("02-four-breaks"))?;

    let trunk: Vec<_> = trunk_ids
        .iter()
        .map(|&id| initial.fiber(id).unwrap())
        .collect();
    let mut roots: Vec<_> = initial
        .bonds()
        .iter()
        .filter_map(|b| {
            let [a, c] = b.fibers;
            if trunk_ids.contains(&a) && !trunk_ids.contains(&c) {
                Some(c)
            } else if trunk_ids.contains(&c) && !trunk_ids.contains(&a) {
                Some(a)
            } else {
                None
            }
        })
        .collect();
    roots.sort_by(|&a, &b| {
        initial
            .fiber(b)
            .unwrap()
            .center()
            .y
            .total_cmp(&initial.fiber(a).unwrap().center().y)
    });
    roots.dedup();
    let mut collars = Vec::new();
    for source_id in roots {
        let original = initial.fiber(source_id).unwrap();
        let z = collar_z(original, &trunk);
        let local = Vec3::Z * z;
        let rest = original.rest;
        let Some(_) = material_span(&movie.world.tree.model, rest, local) else {
            continue;
        };
        let small_shoot = movie.world.twigs.contains(&source_id);
        movie.cue(
            engine,
            rest,
            local,
            false,
            if small_shoot { 18 } else { 24 },
        )?;
        let f = material_span(&movie.world.tree.model, rest, local)
            .ok_or("branch collar detached during cue")?;
        let plane = CutPlane::new(
            f.motion.pose.transform_point(local),
            f.motion.pose.rotation * Vec3::Z,
        )?;
        let root = rooted(&movie.world.tree.model);
        let ids: Vec<_> = movie
            .world
            .tree
            .model
            .fibers()
            .iter()
            .filter(|f| {
                let (lo, hi) = extent(f);
                f.rest == rest && root.contains(&f.id) && lo < z - 1e-8 && hi > z + 1e-8
            })
            .map(|f| f.id)
            .collect();
        if ids.is_empty() {
            return Err("no collar geometry to cut".into());
        }
        movie.world.tree.section(engine, plane, &ids, false)?;
        movie.event("whole_branch_cut",serde_json::json!({"source_span":source_id,"selected_spans":ids,"local_z":z,"retained_collar_m":z-extent(original).0}));
        collars.push((rest, z));
        movie.marks.clear();
        movie.frames(engine, if small_shoot { 60 } else { 105 })?;
        movie.check()?;
    }
    let root = rooted(&movie.world.tree.model);
    for &id in &root {
        let f = movie.world.tree.model.fiber(id).unwrap();
        if !trunk_rest.contains(&f.rest)
            && !collars
                .iter()
                .any(|(r, z)| *r == f.rest && extent(f).1 <= *z + 1e-6)
        {
            return Err(format!("a branch remains on the trunk: {id}").into());
        }
    }
    movie.event(
        "only_trunk",
        serde_json::json!({"root_cells":root,"retained_branch_collars":collars.len()}),
    );
    movie.frame(engine, 4, Some("03-only-trunk"))?;
    movie.frames(engine, 90)?;
    let lengths: Vec<_> = trunk
        .iter()
        .map(|f| {
            let (a, b) = extent(f);
            b - a
        })
        .collect();
    let total: f64 = lengths.iter().sum();
    for (cut_index, fraction) in [2.0 / 3.0, 1.0 / 3.0].into_iter().enumerate() {
        let mut distance = total * fraction;
        let mut selected = None;
        for (f, &len) in trunk.iter().zip(&lengths) {
            if distance < len {
                selected = Some((f.rest, extent(f).0 + distance));
                break;
            }
            distance -= len;
        }
        let (rest, z) = selected.ok_or("trunk section outside centerline")?;
        let local = Vec3::Z * z;
        movie.cue(engine, rest, local, false, 30)?;
        let f =
            material_span(&movie.world.tree.model, rest, local).ok_or("trunk section missing")?;
        let side = Vec3::new(if cut_index == 0 { 0.8 } else { -0.8 }, 0.0, 0.6);
        let plane = CutPlane::new(
            f.motion.pose.transform_point(local),
            (f.motion.pose.rotation * Vec3::Z + side).normalize(),
        )?;
        let id = f.id;
        movie.world.tree.section(engine, plane, &[id], false)?;
        movie.event("trunk_cut",serde_json::json!({"fraction_from_base":fraction,"selected_span":id,"normal":plane.normal().to_array()}));
        movie.marks.clear();
        movie.frames(engine, 240)?;
        movie.check()?;
        if trunk_components(&movie.world.tree.model, &trunk_rest) != cut_index + 2 {
            return Err("unexpected number of trunk pieces".into());
        }
    }
    movie.camera_to(engine, Vec3::new(0.0, 0.9, 0.0), 4.2, 0.9, 60)?;
    movie.frame(engine, 4, Some("04-trunk-three-parts"))?;
    movie.frames(engine, 120)?;
    movie.check()?;
    if movie.world.tree.breaks != 4 || trunk_components(&movie.world.tree.model, &trunk_rest) != 3 {
        return Err("incomplete final trunk sequence".into());
    }
    let report = serde_json::json!({"width":1920,"height":1080,"fps":30,"frames":movie.frame,
        "duration_seconds":movie.frame as f64/30.0,"simulation_seconds":movie.world.tree.time,
        "rough_breaks":movie.world.tree.breaks,"clean_cuts":movie.world.tree.cuts,"stress_breaks":movie.world.tree.stress_breaks,
        "standing_branches_removed":true,"trunk_components":3,"mass_before":movie.initial_mass,
        "mass_after":movie.world.tree.model.fibers().iter().map(|f|f.mass.mass).sum::<f64>(),
        "floor_clearance_m":movie.world.tree.clearance(),"events":movie.events,
        "leaves_total":movie.world.leaves.len(), "leaves_attached":movie.world.attached(),
        "opening_leaves_shed":shed,
        "leaf_strength_multiplier_vs_original":leafy_model::ATTACHMENT_STRENGTH,"leaf_color_palette":"forest, pine, apple green, sage, lime, yellow green",
        "rendering":"full-frame offscreen Vulkan; 0.5x simulation time, two-second introductory hold; no performance claim",
        "watermark":format!("molla alpha {}",env!("CARGO_PKG_VERSION"))});
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        path.with_extension("json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    if let Some(capture) = movie.capture.take() {
        capture.finish()?;
    }
    if let Some(capture) = movie.preview.take() {
        capture.finish()?;
    }
    eprintln!(
        "PASS leafy showcase: {} frames, four rough breaks, {} cuts, three trunk pieces",
        movie.frame, movie.world.tree.cuts
    );
    Ok(())
}
