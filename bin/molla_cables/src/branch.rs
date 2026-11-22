use crate::draw::Camera;
use eframe::egui::{Pos2, Rect};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_geometry::convex_cut::{CutPlane, FaceOrigin};
use molla_math::Vec3;
use molla_solvers::strands::{
    BundleDesc, CoupledConfig, GpuStrandLimits, GpuStrands, StepConfig, StrandModel,
};

pub struct Branch {
    pub model: StrandModel,
    gpu: GpuStrands,
    pub time: f64,
    pub cuts: usize,
    pub breaks: usize,
    pub message: String,
    pub rigidity: f64,
    pub last_cut_ms: f64,
    pub last_enqueue_ms: f64,
    pub last_snapshot_ms: f64,
}

pub const STRANDS: usize = 7;
pub const SEGMENTS: usize = 4;
pub const DT: f64 = 1.0 / 120.0;

fn gpu_scene(engine: &Engine, model: StrandModel) -> Result<GpuStrands> {
    GpuStrands::with_limits(
        engine,
        model,
        GpuStrandLimits {
            contact_pairs: 262_144,
            ..Default::default()
        },
    )
}

pub fn description() -> BundleDesc {
    BundleDesc {
        start: Vec3::new(-1.15, 1.8, 0.0),
        end: Vec3::new(1.15, 1.95, 0.0),
        strand_radius: 0.025 * (7.0 / STRANDS as f64).sqrt(),
        rings: 1,
        segments: SEGMENTS,
        density: 650.0,
        young_modulus: 1.5e7,
        shear_modulus: 6e6,
        tensile_strength: 2e7,
        shear_strength: 1e7,
        interface_ratio: 1.0,
        damage_rate: 0.0,
        damping: 0.35,
        pin_start: true,
    }
}

impl Branch {
    pub fn new(engine: &Engine) -> Result<Self> {
        Self::with_description(engine, description())
    }

    pub fn with_description(engine: &Engine, desc: BundleDesc) -> Result<Self> {
        let model = desc.build()?.model;
        Ok(Self {
            gpu: gpu_scene(engine, model.clone())?,
            model,
            time: 0.0,
            cuts: 0,
            breaks: 0,
            message: "Select Cut or Break, then click the branch. Right-drag to orbit.".into(),
            rigidity: 1.0,
            last_cut_ms: 0.0,
            last_enqueue_ms: 0.0,
            last_snapshot_ms: 0.0,
        })
    }

    pub fn advance(&mut self, engine: &Engine, steps: usize) -> Result<()> {
        let config = StepConfig {
            dt: DT,
            coupled: Some(CoupledConfig {
                linear_iterations: 512,
                relative_tolerance: 2e-5,
                ..Default::default()
            }),
            contact_margin: Some(1e-5),
            pair_restitution: 0.15,
            ..Default::default()
        };
        let submit = std::time::Instant::now();
        for _ in 0..steps {
            self.gpu.step(engine, config)?;
        }
        self.last_enqueue_ms = submit.elapsed().as_secs_f64() * 1000.0;
        let read = std::time::Instant::now();
        self.model = self.gpu.snapshot(engine).map_err(|error| {
            Error::Solver(format!(
                "branch at t={:.4}, rigidity={:.3}: {error}",
                self.time, self.rigidity
            ))
        })?;
        self.last_snapshot_ms = read.elapsed().as_secs_f64() * 1000.0;
        self.time += config.dt * steps as f64;
        Ok(())
    }

    pub fn pluck(&mut self, engine: &Engine, strength: f64) -> Result<()> {
        let impulses: Vec<_> = self
            .model
            .fibers()
            .iter()
            .filter(|f| f.pin.is_none())
            .map(|f| {
                let amount = ((f.rest.position.x + 1.15) / 2.3).clamp(0.0, 1.0);
                (
                    f.id,
                    f.center(),
                    Vec3::new(0.0, -0.5, 1.0) * strength * amount * f.mass.mass,
                )
            })
            .collect();
        self.gpu.impulses(engine, &impulses)?;
        self.model = self.gpu.snapshot(engine)?;
        self.message =
            format!("Applied a {strength:.1} m/s weighted velocity impulse; no bonds deleted.");
        Ok(())
    }

    pub fn set_rigidity(&mut self, engine: &Engine, value: f64) -> Result<()> {
        let mut bonds = self.model.bonds().to_vec();
        for bond in &mut bonds {
            bond.material.translation *= self.rigidity / value;
            bond.material.rotation *= self.rigidity / value;
        }
        let model = StrandModel::new(self.model.fibers().to_vec(), bonds)?;
        let gpu = gpu_scene(engine, model.clone())?;
        self.gpu = gpu;
        self.model = model;
        self.rigidity = value;
        Ok(())
    }

    pub fn cut_stroke(
        &mut self,
        engine: &Engine,
        camera: &Camera,
        rect: Rect,
        a: Pos2,
        b: Pos2,
    ) -> Result<()> {
        self.section_stroke(engine, camera, rect, a, b, false)
    }

    pub fn section_stroke(
        &mut self,
        engine: &Engine,
        camera: &Camera,
        rect: Rect,
        a: Pos2,
        b: Pos2,
        fracture: bool,
    ) -> Result<()> {
        if a.distance(b) < 6.0 {
            self.message = "Drag farther to define a blade stroke.".into();
            return Ok(());
        }
        let now = std::time::Instant::now();
        let (plane, selected) = blade_selection(&self.model, camera, rect, a, b)?;
        if selected.is_empty() {
            self.message =
                "No complete fiber cross section crossed. Extend the stroke through the branch."
                    .into();
            return Ok(());
        }
        let event = (self.cuts + self.breaks) as u64 + 1;
        let report = if fracture {
            let picked = self.model.fiber(selected[0]).unwrap();
            let surfaces = fracture_surfaces(&self.model, &selected, picked.id, plane)?;
            self.gpu.fracture(engine, plane, &surfaces, event)?
        } else {
            self.gpu.cut(engine, plane, &selected, event)?
        };
        self.model = self.gpu.snapshot(engine)?;
        if fracture {
            self.breaks += 1;
        } else {
            self.cuts += 1;
        }
        self.last_cut_ms = now.elapsed().as_secs_f64() * 1000.0;
        self.message = format!(
            "{}: {} cells split, {} bonds severed, {} pieces. {:.1} ms",
            if fracture {
                "Uneven break"
            } else {
                "Exact cut"
            },
            report.replacements.len(),
            report.severed_bonds.len(),
            self.model.components().len(),
            self.last_cut_ms
        );
        eprintln!("{}", self.message);
        Ok(())
    }

    pub fn click_section(
        &mut self,
        engine: &Engine,
        camera: &Camera,
        rect: Rect,
        pixel: Pos2,
        fracture: bool,
    ) -> Result<()> {
        let (origin, direction) = camera.ray(pixel, rect);
        let Some((id, hit)) = ray_hit(&self.model, origin, direction) else {
            self.message = "Click on the actual branch volume.".into();
            return Ok(());
        };
        let now = std::time::Instant::now();
        let (plane, surfaces) = click_surfaces(&self.model, id, hit, fracture)?;
        let event_id = (self.cuts + self.breaks) as u64 + 1;
        let report = if fracture {
            self.gpu.fracture(engine, plane, &surfaces, event_id)?
        } else {
            self.gpu.cut(
                engine,
                plane,
                &surfaces.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
                event_id,
            )?
        };
        self.model = self.gpu.snapshot(engine)?;
        if fracture {
            self.breaks += 1;
        } else {
            self.cuts += 1;
        }
        self.last_cut_ms = now.elapsed().as_secs_f64() * 1000.0;
        self.message = format!(
            "{} at the clicked surface: {} cells split, {} pieces ({:.1} ms).",
            if fracture {
                "Uneven break"
            } else {
                "Clean cut"
            },
            report.replacements.len(),
            self.model.components().len(),
            self.last_cut_ms
        );
        eprintln!("{} hit={hit:?}", self.message);
        Ok(())
    }

    pub fn push_at(
        &mut self,
        engine: &Engine,
        camera: &Camera,
        rect: Rect,
        point: Pos2,
        strength: f64,
    ) -> Result<()> {
        let (origin, direction) = camera.ray(point, rect);
        let Some((id, hit)) = ray_hit(&self.model, origin, direction) else {
            self.message = "Push missed the actual branch volume.".into();
            return Ok(());
        };
        let fiber = self.model.fiber(id).unwrap();
        if fiber.pin.is_some() {
            self.message = "The root is clamped; push farther along the branch.".into();
            return Ok(());
        }
        self.gpu
            .impulse(engine, id, hit, direction * strength * fiber.mass.mass)?;
        self.model = self.gpu.snapshot(engine)?;
        self.message = format!("Pushed fiber {id} at the picked surface point.");
        Ok(())
    }

    pub fn clearance(&self) -> f64 {
        self.model
            .fibers()
            .iter()
            .flat_map(|f| {
                f.shape
                    .vertices()
                    .iter()
                    .map(|&v| f.motion.pose.transform_point(v).y)
            })
            .fold(f64::INFINITY, f64::min)
    }
    pub fn damage_counts(&self) -> (usize, usize) {
        (
            self.model
                .bonds()
                .iter()
                .filter(|b| b.damage > 0.0 && b.damage < 1.0)
                .count(),
            self.model.bonds().iter().filter(|b| !b.intact()).count(),
        )
    }
    pub fn cap_count(&self) -> usize {
        self.model
            .fibers()
            .iter()
            .flat_map(|f| f.shape.faces())
            .filter(|f| {
                matches!(
                    f.origin,
                    FaceOrigin::Cut { .. } | FaceOrigin::Fracture { .. }
                )
            })
            .count()
    }
}

pub fn click_surfaces(
    model: &StrandModel,
    id: u64,
    hit: Vec3,
    fracture: bool,
) -> Result<(CutPlane, Vec<(u64, CutPlane)>)> {
    let fiber = model
        .fiber(id)
        .ok_or_else(|| Error::Build("unknown picked fiber".into()))?;
    let n = fiber.motion.pose.rotation * Vec3::Z;
    let reference = CutPlane::new(hit, n)?;
    let component = model
        .components()
        .into_iter()
        .find(|ids| ids.contains(&id))
        .unwrap();
    let mut selected = Vec::new();
    for candidate in component {
        let f = model.fiber(candidate).unwrap();
        let d = f.center() - hit;
        if (d - n * d.dot(n)).length() > 0.18 {
            continue;
        }
        selected.push(candidate);
    }
    let surfaces = if fracture {
        fracture_surfaces(model, &selected, id, reference)?
    } else {
        selected.into_iter().map(|id| (id, reference)).collect()
    };
    Ok((reference, surfaces))
}

fn fracture_surfaces(
    model: &StrandModel,
    selected: &[u64],
    picked: u64,
    plane: CutPlane,
) -> Result<Vec<(u64, CutPlane)>> {
    let fiber = model.fiber(picked).unwrap();
    let u = fiber.rest.rotation * Vec3::X;
    let v = fiber.rest.rotation * Vec3::Y;
    let rough = |p: Vec3| {
        let phase = p.dot(u) * 1231.0 + p.dot(v) * 2753.0;
        let hash = phase.sin() * 43758.5453;
        0.04 * (2.0 * (hash - hash.floor()) - 1.0)
    };
    let offset = rough(fiber.rest.position);
    selected
        .iter()
        .map(|&id| {
            let shift = rough(model.fiber(id).unwrap().rest.position) - offset;
            Ok((
                id,
                CutPlane::new(plane.point() + plane.normal() * shift, plane.normal())?,
            ))
        })
        .collect()
}

pub fn blade_selection(
    model: &StrandModel,
    camera: &Camera,
    rect: Rect,
    a: Pos2,
    b: Pos2,
) -> Result<(CutPlane, Vec<u64>)> {
    if !a.is_finite() || !b.is_finite() || a.distance(b) < 6.0 {
        return Err(Error::Build(
            "blade stroke must span at least six pixels".into(),
        ));
    }
    let (origin, ra) = camera.ray(a, rect);
    let (_, rb) = camera.ray(b, rect);
    let plane = CutPlane::new(origin, ra.cross(rb))?;
    let direction = b - a;
    let length2 = direction.length_sq();
    let mut selected = Vec::new();
    for fiber in model.fibers() {
        let points: Vec<_> = fiber
            .shape
            .vertices()
            .iter()
            .map(|&v| fiber.motion.pose.transform_point(v))
            .collect();
        let mut section = Vec::new();
        for face in fiber.shape.faces() {
            for j in 0..face.vertices.len() {
                let p = points[face.vertices[j] as usize];
                let q = points[face.vertices[(j + 1) % face.vertices.len()] as usize];
                let dp = plane.distance(p);
                let dq = plane.distance(q);
                if dp.abs() < 1e-10 {
                    section.push(p);
                }
                if dp * dq < 0.0 {
                    section.push(p + (q - p) * (dp / (dp - dq)));
                }
            }
        }
        section.dedup_by(|a, b| a.distance(*b) < 1e-10);
        let nonzero = section.len() >= 3
            && section
                .iter()
                .skip(2)
                .any(|p| (section[1] - section[0]).cross(*p - section[0]).length() > 1e-12);
        if nonzero
            && section.iter().all(|&p| {
                let (screen, depth) = camera.project(p, rect);
                let t = (screen - a).dot(direction) / length2;
                depth > 0.2 && (-1e-5..=1.0 + 1e-5).contains(&t)
            })
        {
            selected.push(fiber.id);
        }
    }
    Ok((plane, selected))
}

pub fn ray_hit(model: &StrandModel, origin: Vec3, direction: Vec3) -> Option<(u64, Vec3)> {
    let mut nearest = f64::INFINITY;
    let mut hit = None;
    for fiber in model.fibers() {
        let pose = fiber.motion.pose;
        let p = pose.inverse_transform_point(origin);
        let d = pose.rotation.conjugate() * direction;
        let vertices = fiber.shape.vertices();
        let mut enter: f64 = 0.0;
        let mut exit = f64::INFINITY;
        for face in fiber.shape.faces() {
            let a = vertices[face.vertices[0] as usize];
            let b = vertices[face.vertices[1] as usize];
            let c = vertices[face.vertices[2] as usize];
            let n = (b - a).cross(c - a).normalize();
            let distance = n.dot(p - a);
            let denominator = n.dot(d);
            if denominator.abs() < 1e-12 {
                if distance > 0.0 {
                    exit = -1.0;
                    break;
                }
                continue;
            }
            let t = -distance / denominator;
            if denominator < 0.0 {
                enter = enter.max(t);
            } else {
                exit = exit.min(t);
            }
        }
        if enter <= exit && enter < nearest {
            nearest = enter;
            hit = Some((fiber.id, origin + direction * enter));
        }
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fast_bundle_preserves_mass_and_disables_spontaneous_failure() {
        let desc = description();
        let dense = desc.build().unwrap();
        let previous = BundleDesc {
            rings: 1,
            strand_radius: 0.025,
            ..desc
        }
        .build()
        .unwrap();
        assert_eq!(dense.strands.len(), STRANDS);
        assert_eq!(dense.model.fibers().len(), STRANDS * SEGMENTS);
        let mass = |model: &StrandModel| model.fibers().iter().map(|f| f.mass.mass).sum::<f64>();
        assert!((mass(&dense.model) / mass(&previous.model) - 1.0).abs() < 1e-12);
        assert!(
            dense
                .model
                .bonds()
                .iter()
                .all(|b| b.material.damage_rate == 0.0)
        );
    }

    #[test]
    fn clicked_cut_and_break_use_the_picked_position_and_distinct_surfaces() {
        let model = description().build().unwrap().model;
        let id = 4;
        let fiber = model.fiber(id).unwrap();
        let hit =
            fiber
                .motion
                .pose
                .transform_point(Vec3::new(description().strand_radius, 0.0, 0.0317));
        for fracture in [false, true] {
            let (reference, surfaces) = click_surfaces(&model, id, hit, fracture).unwrap();
            assert!(reference.distance(hit).abs() < 1e-15);
            assert!(
                surfaces
                    .iter()
                    .find(|(fid, _)| *fid == id)
                    .unwrap()
                    .1
                    .distance(hit)
                    .abs()
                    < 1e-15
            );
            let edit = if fracture {
                model.prepare_fracture(reference, &surfaces, 1).unwrap()
            } else {
                model
                    .prepare_cut(
                        reference,
                        &surfaces.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
                        1,
                    )
                    .unwrap()
            };
            assert_eq!(edit.candidate().components().len(), 2);
            assert!(edit.report().replacements.len() >= STRANDS);
            if fracture {
                assert!(
                    surfaces
                        .iter()
                        .any(|(_, p)| reference.distance(p.point()).abs() > 0.005)
                );
            }
        }
    }
    #[test]
    fn finite_screen_blade_splits_at_the_plane_not_at_segment_nodes() {
        let model = description().build().unwrap().model;
        let camera = Camera::default();
        let rect = Rect::from_min_max(Pos2::ZERO, Pos2::new(900.0, 700.0));
        for x in [-0.487, 0.137, 0.763] {
            let a = camera.project(Vec3::new(x, 1.4, 0.0), rect).0;
            let b = camera.project(Vec3::new(x + 0.12, 2.3, 0.0), rect).0;
            let (plane, ids) = blade_selection(&model, &camera, rect, a, b).unwrap();
            let cut = model.prepare_cut(plane, &ids, 3).unwrap();
            assert!(cut.report().replacements.len() >= STRANDS);
            assert_eq!(cut.candidate().components().len(), 2);
            for f in cut.candidate().fibers() {
                for face in f.shape.faces() {
                    if matches!(face.origin, FaceOrigin::Cut { .. }) {
                        for &v in &face.vertices {
                            assert!(
                                plane
                                    .distance(
                                        f.motion
                                            .pose
                                            .transform_point(f.shape.vertices()[v as usize])
                                    )
                                    .abs()
                                    < 1e-10
                            );
                        }
                    }
                }
            }
        }
        let a = camera.project(Vec3::new(0.137, 1.0, 0.0), rect).0;
        let b = camera.project(Vec3::new(0.137, 1.3, 0.0), rect).0;
        assert!(
            blade_selection(&model, &camera, rect, a, b)
                .unwrap()
                .1
                .is_empty()
        );
    }
    #[test]
    fn camera_ray_and_convex_pick_agree_with_visible_volume() {
        let model = description().build().unwrap().model;
        let camera = Camera::default();
        let rect = Rect::from_min_max(Pos2::ZERO, Pos2::new(900.0, 700.0));
        let p = model.fibers()[4].center();
        let pixel = camera.project(p, rect).0;
        let (origin, direction) = camera.ray(pixel, rect);
        assert!(direction.cross((p - origin).normalize()).length() < 1e-6);
        let (id, hit) = ray_hit(&model, origin, direction).unwrap();
        let fiber = model.fiber(id).unwrap();
        assert!(fiber.contains_local(fiber.motion.pose.inverse_transform_point(hit)));
        assert!(ray_hit(&model, origin, Vec3::Y).is_none());
    }
}
