use super::{FeatherstoneSolver, cpu_step};
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::{BodyFlags, Contacts, Control, Model, ShapeFlags, State};

pub(super) struct ActiveSet {
    pub(super) articulations: Vec<bool>,
    pub(super) joints: Vec<bool>,
    bodies: Vec<bool>,
}

impl ActiveSet {
    fn new(model: &Model, contacts: &Contacts, mask: &[bool]) -> Result<Self> {
        let count = model
            .articulation_count
            .max(usize::from(model.joint_count > 0));
        if mask.len() != count {
            return Err(invalid("invalid articulation activation count"));
        }
        let mut result = Self {
            articulations: mask.to_vec(),
            joints: vec![true; model.joint_count],
            bodies: vec![true; model.body_count],
        };
        let starts = model.articulation_start.host()?;
        let children = model.joint_child.host()?;
        for (art, &active) in mask.iter().enumerate() {
            let (begin, end) = if model.articulation_count == 0 {
                (0, model.joint_count)
            } else {
                let begin = *starts
                    .get(art)
                    .ok_or_else(|| invalid("missing articulation start"))?
                    as usize;
                let end = *starts
                    .get(art + 1)
                    .ok_or_else(|| invalid("missing articulation end"))?
                    as usize;
                (begin, end)
            };
            if begin > end || end > model.joint_count {
                return Err(invalid("invalid articulation range"));
            }
            result.joints[begin..end].fill(active);
            for &body in children
                .get(begin..end)
                .ok_or_else(|| invalid("missing articulation children"))?
            {
                *result
                    .bodies
                    .get_mut(body as usize)
                    .ok_or_else(|| invalid("invalid articulation child"))? = active;
            }
        }
        let a = model.mimic_joint_a.host()?;
        let b = model.mimic_joint_b.host()?;
        let enabled = model.mimic_enabled.host()?;
        if [
            a.len(),
            b.len(),
            enabled.len(),
            model.mimic_c0.len(),
            model.mimic_c1.len(),
        ]
        .into_iter()
        .any(|len| len < model.mimic_count)
        {
            return Err(invalid("invalid activation mimic arrays"));
        }
        for i in 0..model.mimic_count {
            if enabled[i] == 0 {
                continue;
            }
            if let (Some(a), Some(b)) = (
                result.joints.get(a[i] as usize),
                result.joints.get(b[i] as usize),
            ) && a != b
            {
                return Err(invalid("activation splits a mimic dependency"));
            }
        }
        result.validate_contacts(model, contacts)?;
        Ok(result)
    }

    fn validate_contacts(&self, model: &Model, contacts: &Contacts) -> Result<()> {
        let count = contacts.count as usize;
        let a = contacts.a_index.host()?;
        let b = contacts.b_index.host()?;
        let sa = contacts.a_shape.host()?;
        let sb = contacts.b_shape.host()?;
        let depth = contacts.depth.host()?;
        if count > contacts.capacity
            || [a.len(), b.len(), sa.len(), sb.len(), depth.len()]
                .into_iter()
                .any(|len| len < count)
        {
            return Err(invalid("invalid activation contact arrays"));
        }
        let flags = model.body_flags.host()?;
        let shapes = model.shape_flags.host()?;
        for i in 0..count {
            let mut sensor = false;
            for shape in [sa[i], sb[i]] {
                if shape != u32::MAX {
                    let flag = *shapes
                        .get(shape as usize)
                        .ok_or_else(|| invalid("invalid activation contact shape"))?;
                    sensor |= flag & ShapeFlags::SENSOR != 0 || flag & ShapeFlags::COLLIDES == 0;
                }
            }
            let mut active = [None; 2];
            for (slot, body) in active.iter_mut().zip([a[i], b[i]]) {
                if body != u32::MAX {
                    let flag = *flags
                        .get(body as usize)
                        .ok_or_else(|| invalid("invalid activation contact body"))?;
                    let selected = *self
                        .bodies
                        .get(body as usize)
                        .ok_or_else(|| invalid("invalid activation contact body"))?;
                    if flag & BodyFlags::IMMOVABLE == 0 {
                        *slot = Some(selected);
                    }
                }
            }
            if !sensor && depth[i] > 0.0 && matches!(active, [Some(a), Some(b)] if a != b) {
                return Err(invalid("activation splits a dynamic contact"));
            }
        }
        Ok(())
    }

    pub(super) fn filter_readings(
        &self,
        model: &Model,
        readings: &mut [crate::penalty_contact::ContactForce],
    ) -> Result<()> {
        let flags = model.body_flags.host()?;
        for reading in readings {
            if !reading.bodies.iter().flatten().any(|body| {
                self.bodies[body.index()] && flags[body.index()] & BodyFlags::IMMOVABLE == 0
            }) {
                reading.acted = false;
                reading.normal_force = 0.0;
                reading.force_on_b = Vec3::ZERO;
            }
        }
        Ok(())
    }

    pub(super) fn restore(&self, model: &Model, input: &State, output: &mut State) -> Result<()> {
        let coords = model.joint_coord_offset.host()?;
        let dofs = model.joint_dof_offset.host()?;
        for (j, &active) in self.joints.iter().enumerate() {
            if active {
                continue;
            }
            let q = coords[j] as usize..coords[j + 1] as usize;
            let qd = dofs[j] as usize..dofs[j + 1] as usize;
            output.joint_q.host_mut()?[q.clone()].copy_from_slice(&input.joint_q.host()?[q]);
            output.joint_qd.host_mut()?[qd.clone()].copy_from_slice(&input.joint_qd.host()?[qd]);
        }
        for (body, &active) in self.bodies.iter().enumerate() {
            if !active {
                output.body_q.host_mut()?[body] = input.body_q.host()?[body];
                output.body_qd.host_mut()?[body] = input.body_qd.host()?[body];
            }
        }
        Ok(())
    }
}

impl FeatherstoneSolver {
    /// Refresh contacts, select complete articulations and apply transient wrenches each substep.
    /// The force callback returns the activation mask after inspecting fresh contacts.
    pub fn step_with_collision_active(
        &mut self,
        model: &Model,
        input: &State,
        control: &Control,
        pipeline: &mut molla_geometry::CollisionPipeline,
        contacts: &mut Contacts,
        dt: Real,
        output: &mut State,
        excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
        forces: impl FnMut(
            &State,
            &Contacts,
            Real,
            &mut [molla_math::SpatialVector],
        ) -> Result<Vec<bool>>,
    ) -> Result<()> {
        self.step_with_active_bounds(
            model, input, control, pipeline, contacts, dt, output, excluded, forces, None,
        )
    }

    pub(crate) fn step_with_active_bounds(
        &mut self,
        model: &Model,
        input: &State,
        control: &Control,
        pipeline: &mut molla_geometry::CollisionPipeline,
        contacts: &mut Contacts,
        dt: Real,
        output: &mut State,
        excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
        mut forces: impl FnMut(
            &State,
            &Contacts,
            Real,
            &mut [molla_math::SpatialVector],
        ) -> Result<Vec<bool>>,
        bounds: Option<&crate::geometry::shape_world::PreparedShapeBounds<'_>>,
    ) -> Result<()> {
        self.step_with_active_constraints(
            model,
            input,
            control,
            pipeline,
            contacts,
            dt,
            output,
            excluded,
            |state, contacts, dt, wrenches| forces(state, contacts, dt, wrenches),
            bounds,
            false,
            |_, _, _, _, _| Ok(()),
        )
    }

    pub(crate) fn step_with_active_constraints(
        &mut self,
        model: &Model,
        input: &State,
        control: &Control,
        pipeline: &mut molla_geometry::CollisionPipeline,
        contacts: &mut Contacts,
        dt: Real,
        output: &mut State,
        mut excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
        mut forces: impl FnMut(
            &State,
            &mut Contacts,
            Real,
            &mut [molla_math::SpatialVector],
        ) -> Result<Vec<bool>>,
        bounds: Option<&crate::geometry::shape_world::PreparedShapeBounds<'_>>,
        defer_constraints: bool,
        mut constraints: impl FnMut(&mut Self, &State, &mut State, Real, &[bool]) -> Result<()>,
    ) -> Result<()> {
        self.contact_forces.clear();
        if self.is_gpu() || !dt.is_finite() || dt <= 0.0 {
            return Err(invalid(
                "active collision stepping requires CPU and positive finite dt",
            ));
        }
        let count = self.substeps.max(1);
        let sub_dt = dt / count as Real;
        if sub_dt == 0.0 {
            return Err(invalid("activation substep dt underflows"));
        }
        let mut scratch = input.clone_host()?;
        molla_sim::eval_fk(model, &mut scratch)?;
        molla_sim::eval_body_velocities(model, &mut scratch)?;
        let persistent = input.body_f.host()?.to_vec();
        let mut applied = persistent.clone();
        let mut samples = Vec::new();
        let mut efforts = Vec::new();
        let owned_bounds;
        let bounds = match bounds {
            Some(bounds) if std::ptr::eq(bounds.model(), model) => bounds,
            Some(_) => return Err(invalid("bounds model mismatch")),
            None => {
                owned_bounds = crate::geometry::shape_world::PreparedShapeBounds::new(model);
                &owned_bounds
            }
        };
        for substep in 0..count {
            pipeline.collide_prepared(bounds, &scratch, contacts, &mut excluded)?;
            applied.copy_from_slice(&persistent);
            let mask = forces(&scratch, contacts, sub_dt, &mut applied)?;
            let active = ActiveSet::new(model, contacts, &mask)?;
            if applied
                .iter()
                .any(|force| !force.linear.is_finite() || !force.angular.is_finite())
            {
                return Err(invalid("non-finite transient body force"));
            }
            scratch.body_f.host_mut()?.copy_from_slice(&applied);
            self.contact_forces.clear();
            cpu_step(
                model,
                &scratch,
                control,
                contacts,
                sub_dt,
                output,
                self.joint_damping,
                self.stable_motor_feedback,
                self.rigorous_d_solve,
                self.rigorous_distance,
                &mut self.contact_forces,
                &mut self.substep_efforts,
                Some(&active),
                None,
                defer_constraints,
            )?;
            constraints(self, &scratch, output, sub_dt, &mask)?;
            for reading in &mut self.contact_forces {
                reading.sample_dt = sub_dt;
                reading.normal_impulse = reading.normal_force * sub_dt;
            }
            output.body_f.host_mut()?.copy_from_slice(&persistent);
            samples.append(&mut self.contact_forces);
            self.accumulate_efforts(&mut efforts, 1.0 / count as Real);
            if substep + 1 < count {
                std::mem::swap(&mut scratch, output);
            }
        }
        self.contact_forces = samples;
        self.joint_efforts = efforts;
        Ok(())
    }

    /// CPU advancement of selected whole articulations; inactive rigid state is retained.
    /// Mixed active/inactive dynamic contacts and mimic links are rejected.
    /// The caller supplies force-coupling closure and wake policy; this does not detect sleep.
    /// Contacts are held constant across substeps. Inactive velocities are not zeroed.
    pub fn step_active_articulations(
        &mut self,
        model: &Model,
        input: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        output: &mut State,
        active: &[bool],
    ) -> Result<()> {
        self.contact_forces.clear();
        if self.is_gpu() {
            return Err(invalid("articulation activation requires CPU Featherstone"));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid("step dt must be positive and finite"));
        }
        let active = ActiveSet::new(model, contacts, active)?;
        let count = self.substeps.max(1);
        let sub_dt = dt / count as Real;
        if sub_dt == 0.0 {
            return Err(invalid("activation substep dt underflows"));
        }
        let mut scratch = input.clone_host()?;
        let mut samples = Vec::new();
        let mut efforts = Vec::new();
        for step in 0..count {
            self.contact_forces.clear();
            cpu_step(
                model,
                &scratch,
                control,
                contacts,
                sub_dt,
                output,
                self.joint_damping,
                self.stable_motor_feedback,
                self.rigorous_d_solve,
                self.rigorous_distance,
                &mut self.contact_forces,
                &mut self.substep_efforts,
                Some(&active),
                None,
                false,
            )?;
            for reading in &mut self.contact_forces {
                reading.sample_dt = sub_dt;
                reading.normal_impulse = reading.normal_force * sub_dt;
            }
            samples.append(&mut self.contact_forces);
            self.accumulate_efforts(&mut efforts, 1.0 / count as Real);
            if step + 1 < count {
                std::mem::swap(&mut scratch, output);
            }
        }
        self.contact_forces = samples;
        self.joint_efforts = efforts;
        Ok(())
    }
}

fn invalid(message: &str) -> Error {
    Error::Solver(message.into())
}
