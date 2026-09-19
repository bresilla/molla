//! Axis-addressed runtime limits and motor authoring.

use crate::{Control, JointTargetMode, JointType};
use molla_core::{Error, Result};
use molla_math::Real;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum JointAxis {
    LinX,
    LinY,
    LinZ,
    AngX,
    AngY,
    AngZ,
}

impl JointAxis {
    pub const ALL: [Self; 6] = [
        Self::LinX,
        Self::LinY,
        Self::LinZ,
        Self::AngX,
        Self::AngY,
        Self::AngZ,
    ];

    /// Revolute and prismatic freedoms use AngX and LinX regardless of authored axis.
    pub fn dof(self, kind: JointType) -> Result<usize> {
        match (kind, self) {
            (JointType::Revolute, Self::AngX) | (JointType::Prismatic, Self::LinX) => Ok(0),
            (JointType::Ball, Self::AngX | Self::AngY | Self::AngZ) => Ok(self as usize - 3),
            (JointType::D6 | JointType::Free, _) => Ok(self as usize),
            _ => Err(Error::Build(
                "axis is not a freedom of this joint kind".into(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JointAxes(pub u8);

impl JointAxes {
    pub const NONE: Self = Self(0);
    pub const LIN: Self = Self(7);
    pub const ANG: Self = Self(56);
    pub const ALL: Self = Self(63);

    pub const fn with(self, axis: JointAxis) -> Self {
        Self(self.0 | (1 << axis as u8))
    }
    pub const fn contains(self, axis: JointAxis) -> bool {
        self.0 & (1 << axis as u8) != 0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum MotorModel {
    #[default]
    Force,
    Acceleration,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointMotor {
    pub mode: JointTargetMode,
    pub target_position: Real,
    pub target_velocity: Real,
    pub stiffness: Real,
    pub damping: Real,
    pub max_force: Real,
    pub model: MotorModel,
}

impl Default for JointMotor {
    fn default() -> Self {
        Self {
            mode: JointTargetMode::None,
            target_position: 0.0,
            target_velocity: 0.0,
            stiffness: 0.0,
            damping: 0.0,
            max_force: Real::INFINITY,
            model: MotorModel::Force,
        }
    }
}

impl JointMotor {
    pub fn validate(self) -> Result<()> {
        if !self.target_position.is_finite()
            || !self.target_velocity.is_finite()
            || !self.stiffness.is_finite()
            || self.stiffness < 0.0
            || !self.damping.is_finite()
            || self.damping < 0.0
            || self.max_force.is_nan()
            || self.max_force < 0.0
        {
            return Err(Error::Build("invalid joint motor parameters".into()));
        }
        Ok(())
    }

    pub(crate) fn apply(self, control: &mut Control, dof: usize) -> Result<()> {
        control.joint_target_mode.host_mut()?[dof] = self.mode as u32;
        control.joint_target_pos.host_mut()?[dof] = self.target_position;
        control.joint_target_vel.host_mut()?[dof] = self.target_velocity;
        control.joint_target_kp.host_mut()?[dof] = self.stiffness;
        control.joint_target_kd.host_mut()?[dof] = self.damping;
        control.joint_motor_model.host_mut()?[dof] = self.model as u32;
        control.joint_motor_max_force.host_mut()?[dof] = self.max_force;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JointAxisSettings {
    pub limits: Option<[Real; 2]>,
    pub locked: bool,
    pub motor: Option<JointMotor>,
}
