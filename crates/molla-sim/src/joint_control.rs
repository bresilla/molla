//! Axis-addressed runtime limits and motor authoring.

use crate::{Control, JointTargetMode, JointType};
use molla_core::{Error, Result};
use molla_math::Real;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointSoftness {
    pub natural_frequency: Real,
    pub damping_ratio: Real,
}

impl JointSoftness {
    pub fn validate(self) -> Result<()> {
        let omega = std::f64::consts::TAU * self.natural_frequency;
        if !self.natural_frequency.is_finite()
            || self.natural_frequency < 0.0
            || !self.damping_ratio.is_finite()
            || self.damping_ratio < 0.0
            || !(omega * omega).is_finite()
            || !(2.0 * self.damping_ratio * omega).is_finite()
        {
            return Err(Error::Build("invalid joint softness".into()));
        }
        Ok(())
    }
}

/// Per-DOF spring interval; inactive records apply no constraint force.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JointCompliance {
    pub enabled: u32,
    pub padding: u32,
    pub lower: Real,
    pub upper: Real,
    pub natural_frequency: Real,
    pub damping_ratio: Real,
}

impl JointCompliance {
    pub fn validate(self) -> Result<()> {
        if self.enabled > 1
            || self.lower.is_nan()
            || self.upper.is_nan()
            || self.lower == Real::INFINITY
            || self.upper == Real::NEG_INFINITY
            || self.lower > self.upper
        {
            return Err(Error::Build("invalid compliant joint interval".into()));
        }
        JointSoftness {
            natural_frequency: self.natural_frequency,
            damping_ratio: self.damping_ratio,
        }
        .validate()
    }

    pub const DISABLED: Self = Self {
        enabled: 0,
        padding: 0,
        lower: Real::NEG_INFINITY,
        upper: Real::INFINITY,
        natural_frequency: 0.0,
        damping_ratio: 0.0,
    };
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuJointCompliance {
    pub enabled: u32,
    pub padding: u32,
    pub lower: f32,
    pub upper: f32,
    pub natural_frequency: f32,
    pub damping_ratio: f32,
}

impl molla_storage::BufferElement for JointCompliance {
    type Gpu = GpuJointCompliance;

    fn to_gpu(&self) -> Self::Gpu {
        GpuJointCompliance {
            enabled: self.enabled,
            padding: 0,
            lower: self.lower as f32,
            upper: self.upper as f32,
            natural_frequency: self.natural_frequency as f32,
            damping_ratio: self.damping_ratio as f32,
        }
    }

    fn from_gpu(gpu: &Self::Gpu) -> Self {
        Self {
            enabled: gpu.enabled,
            padding: 0,
            lower: gpu.lower as Real,
            upper: gpu.upper as Real,
            natural_frequency: gpu.natural_frequency as Real,
            damping_ratio: gpu.damping_ratio as Real,
        }
    }
}

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
    /// Brake damping (N·m·s/rad or N·s/m); zero releases the brake.
    pub brake: Real,
}
