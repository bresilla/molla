use crate::{adaptive_step, apple_tree_model, branch, draw, grab_handle};
use eframe::egui;
use molla_compute::Engine;
use molla_core::Result;
use molla_geometry::convex_cut::CutPlane;
use molla_math::Vec3;
use molla_solvers::strands::{
    CoupledConfig, GpuStrandLimits, GpuStrands, PointGrab, StepConfig, StrandModel,
};

pub(crate) const DT: f64 = 1.0 / 240.0;

pub(crate) fn gpu_limits() -> GpuStrandLimits {
    GpuStrandLimits {
        contact_pairs: 262_144,
        ..Default::default()
    }
}

pub(crate) fn step_config() -> StepConfig {
    StepConfig {
        dt: DT,
        iterations: 8,
        air_damping: 0.25,
        coupled: Some(CoupledConfig {
            nonlinear_iterations: 2,
            linear_iterations: 512,
            relative_tolerance: 2e-5,
            absolute_tolerance: 1e-10,
        }),
        contact_margin: Some(1e-5),
        contact_exclusion_hops: 2,
        pair_restitution: 0.1,
        ..Default::default()
    }
}

pub(crate) struct Tree {
    pub(crate) model: StrandModel,
    pub(crate) gpu: GpuStrands,
    pub(crate) name: String,
    pub(crate) source_limbs: usize,
    pub(crate) limbs: usize,
    pub(crate) time: f64,
    pub(crate) cuts: usize,
    pub(crate) breaks: usize,
    pub(crate) message: String,
    pub(crate) recovered_batches: usize,
    pub(crate) last_subdivisions: usize,
    pub(crate) grab: Option<PointGrab>,
    pub(crate) handle: Option<grab_handle::GrabHandle>,
    pub(crate) push_remaining: Option<f64>,
    pub(crate) grip_force: f64,
    pub(crate) stress_breaks: usize,
    pub(crate) step_substeps: usize,
    pub(crate) quiet_batches: usize,
    pub(crate) retained_subdivisions: usize,
    pub(crate) coarsen_batches: usize,
    pub(crate) limits: GpuStrandLimits,
    pub(crate) config: StepConfig,
}

impl Tree {
    pub(crate) fn new(engine: &Engine, source: &str, limbs: usize) -> Result<Self> {
        let tree = apple_tree_model::build(source, limbs)?;
        Self::from_built(engine, tree)
    }

    pub(crate) fn from_built(engine: &Engine, tree: apple_tree_model::TreeModel) -> Result<Self> {
        Self::from_built_with_limits(engine, tree, gpu_limits())
    }

    pub(crate) fn from_built_with_limits(
        engine: &Engine,
        tree: apple_tree_model::TreeModel,
        limits: GpuStrandLimits,
    ) -> Result<Self> {
        let gpu = GpuStrands::with_limits(engine, tree.model.clone(), limits)?;
        Ok(Self {
            model: tree.model,
            gpu,
            name: tree.tree_id,
            source_limbs: tree.source_limbs,
            limbs: tree.simulated_limbs,
            time: 0.0,
            cuts: 0,
            breaks: 0,
            message: "Click a limb to cut at the actual surface. Right-drag to orbit.".into(),
            recovered_batches: 0,
            last_subdivisions: 1,
            grab: None,
            handle: None,
            push_remaining: None,
            grip_force: 4.0,
            stress_breaks: 0,
            step_substeps: 1,
            quiet_batches: 0,
            retained_subdivisions: 8,
            coarsen_batches: 12,
            limits,
            config: step_config(),
        })
    }

    pub(crate) fn start_grab(&mut self, id: u64, hit: Vec3) {
        self.grab = Some(PointGrab {
            fiber: id,
            local_point: self
                .model
                .fiber(id)
                .unwrap()
                .motion
                .pose
                .inverse_transform_point(hit),
            target: hit,
            compliance: 2e-3,
            target_velocity: Vec3::ZERO,
            damping: 10.0,
            max_force: Some(self.grip_force),
        });
        self.handle = Some(grab_handle::GrabHandle::new(hit));
        self.push_remaining = None;
    }

    pub(crate) fn release_grab(&mut self) {
        self.grab = None;
        self.handle = None;
        self.push_remaining = None;
    }

    pub(crate) fn next_grab(
        &self,
        count: usize,
    ) -> Option<(grab_handle::GrabHandle, PointGrab, Vec3)> {
        self.grab.map(|mut grab| {
            let mut handle = self.handle.unwrap_or_else(|| {
                grab_handle::GrabHandle::new(
                    self.model
                        .fiber(grab.fiber)
                        .unwrap()
                        .motion
                        .pose
                        .transform_point(grab.local_point),
                )
            });
            let start = handle.point;
            for _ in 0..count {
                handle.advance(grab.target, self.config.dt);
            }
            grab.target = handle.point;
            grab.target_velocity = (handle.point - start) / (count as f64 * self.config.dt);
            (handle, grab, start)
        })
    }

    pub(crate) fn advance(&mut self, engine: &Engine, count: usize) -> Result<()> {
        let substeps = self.step_substeps;
        let motion = self.next_grab(count);
        let result = adaptive_step::advance_with_grab_start(
            engine,
            &mut self.gpu,
            &self.model,
            self.limits,
            StepConfig {
                dt: self.config.dt / substeps as f64,
                grab: motion.map(|m| m.1),
                ..self.config
            },
            count * substeps,
            motion.map(|m| m.2),
        )?;
        self.handle = motion.map(|m| m.0);
        if let Some(remaining) = &mut self.push_remaining {
            *remaining -= count as f64 * self.config.dt;
            if *remaining <= 0.0 {
                self.release_grab();
            }
        }
        if result.halvings > 0 {
            self.recovered_batches += 1;
            eprintln!(
                "Apple solver recovered at t={:.4}: replayed {} steps with {} subdivisions; edits retained",
                self.time,
                count,
                substeps << result.halvings
            );
        }
        let failed: Vec<_> = result
            .model
            .bonds()
            .iter()
            .filter(|b| {
                b.damage >= 1.0
                    && self
                        .model
                        .bonds()
                        .iter()
                        .any(|old| old.id == b.id && old.intact())
            })
            .map(|b| (b.fibers, b.anchors))
            .collect();
        self.model = result.model;
        self.last_subdivisions = substeps << result.halvings;
        self.step_substeps = self.last_subdivisions.clamp(1, self.retained_subdivisions);
        self.quiet_batches = if result.halvings == 0 {
            self.quiet_batches + 1
        } else {
            0
        };
        if self.grab.is_none() && self.quiet_batches >= self.coarsen_batches {
            self.step_substeps = (self.step_substeps / 2).max(1);
            self.quiet_batches = 0;
        }
        self.time += count as f64 * self.config.dt;
        if !failed.is_empty() {
            self.stress_breaks += failed.len();
            self.roughen_failed_ends(engine, &failed)?;
            self.message = format!(
                "Wood failed under bending load: {} stressed joints. Release the mouse to drop it.",
                self.stress_breaks
            );
            eprintln!("{}", self.message);
        }
        Ok(())
    }

    pub(crate) fn roughen_failed_ends(
        &mut self,
        engine: &Engine,
        failed: &[([u64; 2], [Vec3; 2])],
    ) -> Result<()> {
        let mut candidate = self.model.clone();
        let mut new_grab = self.grab;
        let mut used = std::collections::BTreeSet::new();
        for (ids, anchors) in failed {
            for side in 0..2 {
                let id = ids[side];
                let Some(f) = candidate.fiber(id) else {
                    continue;
                };
                if f.pin.is_some() || !used.insert(id) {
                    continue;
                }
                let toward = f.mass.local_com - anchors[side];
                let radius = f
                    .shape
                    .vertices()
                    .iter()
                    .map(|p| (p.truncate() - f.mass.local_com.truncate()).length())
                    .fold(0.0, f64::max);
                if toward.length() < radius * 2.0 {
                    continue;
                }
                let point =
                    anchors[side] + toward.normalize() * (radius * 1.4).min(toward.length() * 0.25);
                let plane = CutPlane::new(
                    f.motion.pose.transform_point(point),
                    f.motion.pose.rotation * toward.normalize(),
                )?;
                let edit = candidate.prepare_splinter(
                    id,
                    plane,
                    (self.cuts + self.breaks + self.stress_breaks) as u64 + 1,
                )?;
                let report = candidate.commit_splinter(edit)?;
                if let Some(grab) = &mut new_grab
                    && grab.fiber == id
                {
                    grab.fiber = report
                        .fragments
                        .iter()
                        .flatten()
                        .copied()
                        .find(|&id| {
                            candidate
                                .fiber(id)
                                .unwrap()
                                .contains_local(grab.local_point)
                        })
                        .ok_or_else(|| {
                            molla_core::Error::Solver("lost grab point during fracture".into())
                        })?;
                }
            }
        }
        let replacement = GpuStrands::with_limits(engine, candidate.clone(), self.limits)?;
        self.gpu = replacement;
        self.model = candidate;
        self.grab = new_grab;
        Ok(())
    }

    pub(crate) fn section(
        &mut self,
        engine: &Engine,
        plane: CutPlane,
        ids: &[u64],
        fracture: bool,
    ) -> Result<()> {
        let event = (self.cuts + self.breaks + self.stress_breaks + 1) as u64;
        let split = if fracture {
            let mut candidate = self.model.clone();
            for &id in ids {
                let edit = candidate.prepare_splinter(id, plane, event)?;
                candidate.commit_splinter(edit)?;
            }
            self.gpu = GpuStrands::with_limits(engine, candidate, self.limits)?;
            ids.len()
        } else {
            self.gpu.cut(engine, plane, ids, event)?.replacements.len()
        };
        self.model = self.gpu.snapshot(engine)?;
        self.release_grab();
        if split > 0 {
            if fracture {
                self.breaks += 1;
            } else {
                self.cuts += 1;
            }
        }
        self.message = format!(
            "{}: {} cells split · {} pieces",
            if fracture {
                "Splintered break"
            } else {
                "Exact cut"
            },
            split,
            self.model.components().len()
        );
        eprintln!("{}", self.message);
        Ok(())
    }

    pub(crate) fn click(
        &mut self,
        engine: &Engine,
        camera: &draw::Camera,
        pixel: egui::Pos2,
        rect: egui::Rect,
        fracture: bool,
        push: bool,
    ) -> Result<()> {
        let (origin, direction) = camera.ray(pixel, rect);
        let Some((id, hit)) = branch::ray_hit(&self.model, origin, direction) else {
            self.message = "Missed the wood. Zoom in to select thin twigs.".into();
            return Ok(());
        };
        let f = self.model.fiber(id).unwrap();
        eprintln!(
            "Apple pick: id={id} local_hit={:?} mass={:.6}",
            f.motion.pose.inverse_transform_point(hit),
            f.mass.mass
        );
        if push {
            if f.pin.is_some() {
                self.message = "The trunk base is clamped.".into();
                return Ok(());
            }
            self.start_grab(id, hit);
            self.grab.as_mut().unwrap().target += direction * 0.18;
            self.push_remaining = Some(0.32);
            self.message = "Applying a short, damped push to the picked wood.".into();
            return Ok(());
        }
        let axis = Vec3::Z;
        let plane = CutPlane::new(hit, f.motion.pose.rotation * axis)?;
        self.section(engine, plane, &[id], fracture)
    }

    pub(crate) fn sway(&mut self, engine: &Engine) -> Result<()> {
        let impulses: Vec<_> = self
            .model
            .fibers()
            .iter()
            .filter(|f| f.pin.is_none())
            .map(|f| {
                let amount = (f.center().y / 2.5).clamp(0.0, 1.0);
                (
                    f.id,
                    f.center(),
                    Vec3::new(0.22, 0.0, 0.35) * amount * f.mass.mass,
                )
            })
            .collect();
        self.gpu.impulses(engine, &impulses)?;
        self.model = self.gpu.snapshot(engine)?;
        self.message = "Applied one breeze impulse; elastic joints restore the tree.".into();
        Ok(())
    }

    pub(crate) fn clearance(&self) -> f64 {
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

    pub(crate) fn visible_limb(&self, camera: &draw::Camera, rect: egui::Rect) -> egui::Pos2 {
        self.visible_limb_at(camera, rect, 0).0
    }

    pub(crate) fn visible_limb_at(
        &self,
        camera: &draw::Camera,
        rect: egui::Rect,
        pick: usize,
    ) -> (egui::Pos2, u64) {
        let root = self
            .model
            .components()
            .into_iter()
            .find(|ids| {
                ids.iter()
                    .any(|&id| self.model.fiber(id).unwrap().pin.is_some())
            })
            .unwrap();
        self.model
            .fibers()
            .iter()
            .rev()
            .filter(|f| f.pin.is_none() && root.contains(&f.id))
            .filter_map(|f| {
                let pixel = camera.project(f.center(), rect).0;
                if !rect.contains(pixel) {
                    return None;
                }
                let (origin, direction) = camera.ray(pixel, rect);
                branch::ray_hit(&self.model, origin, direction)
                    .filter(|(id, _)| *id == f.id)
                    .map(|_| (pixel, f.id))
            })
            .nth(pick)
            .expect("a visible connected limb")
    }
}
