use super::*;

/// Initial scene authoring with one final compilation and no intermediate solver snapshots.
/// Geometry and topology are validated by `build`; returned handles belong to the built scene.
pub struct RigidSceneBuilder {
    description: Description,
    gravity: Vec3,
}

impl Default for RigidSceneBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RigidSceneBuilder {
    pub fn new() -> Self {
        Self {
            description: Description::default(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }

    pub fn set_gravity(&mut self, gravity: Vec3) -> Result<()> {
        if !gravity.is_finite() {
            return Err(Error::Build("non-finite gravity".into()));
        }
        self.gravity = gravity;
        Ok(())
    }

    pub fn insert_body(&mut self, desc: BodyDesc) -> Result<BodyHandle> {
        self.description.insert_body(desc)
    }

    pub fn insert_collider(&mut self, desc: ColliderDesc) -> Result<ColliderHandle> {
        self.description.insert_collider(desc)
    }

    pub fn insert_joint(&mut self, desc: JointDesc) -> Result<JointHandle> {
        self.description.insert_joint(desc)
    }

    pub fn set_distance_limits(&mut self, id: JointHandle, limits: DistanceLimits) -> Result<()> {
        limits.validate()?;
        if self.joint(id)?.kind != JointType::Distance {
            return Err(Error::Build(
                "distance limits require a distance joint".into(),
            ));
        }
        self.description.distance_limits.insert(id, limits);
        Ok(())
    }

    pub fn set_cone_limits(&mut self, id: JointHandle, limits: SphericalConeLimits) -> Result<()> {
        limits.validate()?;
        let joint = self.joint(id)?;
        if joint.kind != JointType::Ball {
            return Err(Error::Build("cone limits require a ball joint".into()));
        }
        if !limits.is_unrestricted() {
            crate::spherical_cone::basis(joint.axis)?;
        }
        if limits == SphericalConeLimits::default() {
            self.description.cone_limits.remove(&id);
        } else {
            self.description.cone_limits.insert(id, limits);
        }
        Ok(())
    }

    pub fn set_joint_limits(
        &mut self,
        id: JointHandle,
        axis: JointAxis,
        limits: Option<[Real; 2]>,
    ) -> Result<()> {
        axis.dof(self.joint(id)?.kind)?;
        if limits.is_some_and(|[lo, hi]| !lo.is_finite() || !hi.is_finite() || lo > hi) {
            return Err(Error::Build("invalid joint axis limits".into()));
        }
        self.description.joint_axes.get_mut(&id).unwrap()[axis as usize].limits = limits;
        Ok(())
    }

    pub fn set_joint_motor(
        &mut self,
        id: JointHandle,
        axis: JointAxis,
        motor: JointMotor,
    ) -> Result<()> {
        motor.validate()?;
        axis.dof(self.joint(id)?.kind)?;
        self.description.joint_axes.get_mut(&id).unwrap()[axis as usize].motor = Some(motor);
        Ok(())
    }

    /// Compile all staged authoring; failure returns no partially usable scene.
    pub fn build(self) -> Result<RigidScene> {
        RigidScene::from_description(self.description, self.gravity)
    }

    fn joint(&self, id: JointHandle) -> Result<&JointDesc> {
        self.description
            .joints
            .get(&id)
            .ok_or_else(|| missing("joint"))
    }
}
