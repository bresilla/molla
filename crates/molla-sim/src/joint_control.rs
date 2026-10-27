//! Axis-addressed runtime limits and motor authoring.

use crate::{Control, JointTargetMode, JointType};
use molla_core::{Error, Result};
use molla_math::Real;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JointBreakLimits {
    pub force: Option<Real>,
    pub torque: Option<Real>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum JointBreakTrigger {
    Force = 1,
    Torque = 2,
    Both = 3,
}

impl JointBreakLimits {
    pub fn validate(self) -> Result<()> {
        if [self.force, self.torque]
            .into_iter()
            .flatten()
            .any(|v| v.is_nan() || v < 0.0)
        {
            return Err(Error::Build(
                "joint break thresholds must be nonnegative and not NaN".into(),
            ));
        }
        Ok(())
    }

    pub fn is_enabled(self) -> bool {
        [self.force, self.torque]
            .into_iter()
            .flatten()
            .any(Real::is_finite)
    }

    /// Inclusive thresholds; zero triggers on any valid sample, including zero load.
    pub fn trigger(self, force: Real, torque: Real) -> Result<Option<JointBreakTrigger>> {
        self.validate()?;
        if !force.is_finite() || force < 0.0 || !torque.is_finite() || torque < 0.0 {
            return Err(Error::Solver(
                "invalid joint break sample magnitudes".into(),
            ));
        }
        Ok(
            match (
                self.force.is_some_and(|limit| force >= limit),
                self.torque.is_some_and(|limit| torque >= limit),
            ) {
                (true, false) => Some(JointBreakTrigger::Force),
                (false, true) => Some(JointBreakTrigger::Torque),
                (true, true) => Some(JointBreakTrigger::Both),
                (false, false) => None,
            },
        )
    }
}

/// Geodesic swing half-extents toward the two cyclic transverse joint-frame axes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SphericalConeLimits {
    pub angle0: Option<Real>,
    pub angle1: Option<Real>,
}

impl SphericalConeLimits {
    pub fn validate(self) -> Result<()> {
        if self
            .angle0
            .into_iter()
            .chain(self.angle1)
            .any(|angle| !angle.is_finite() || !(0.0..=std::f64::consts::PI).contains(&angle))
        {
            return Err(Error::Build("invalid spherical cone limits".into()));
        }
        Ok(())
    }

    pub fn is_unrestricted(self) -> bool {
        [self.angle0, self.angle1]
            .into_iter()
            .all(|angle| angle.is_none_or(|angle| angle == std::f64::consts::PI))
    }
}

/// Independent nonnegative radial bounds between joint anchors, in metres.
/// At coincident anchors, a positive radial bound uses world X oriented from the
/// lower stable body handle to the higher; a world parent uses positive X.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DistanceLimits {
    pub min: Option<Real>,
    pub max: Option<Real>,
}

impl DistanceLimits {
    pub fn validate(self) -> Result<()> {
        if self
            .min
            .into_iter()
            .chain(self.max)
            .any(|v| !v.is_finite() || v < 0.0)
            || matches!((self.min, self.max), (Some(min), Some(max)) if min > max)
        {
            return Err(Error::Build("invalid distance limits".into()));
        }
        Ok(())
    }
}

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

/// Whether Euler point limits constrain only part of the angular motion.
pub fn partial_angular_lock(lower: &[Real], upper: &[Real]) -> bool {
    let count = lower
        .iter()
        .zip(upper)
        .take(3)
        .filter(|(lo, hi)| lo.is_finite() && hi.is_finite() && (**hi - **lo).abs() < 1e-12)
        .count();
    count > 0 && count < 3
}

/// Whether six-axis limits or velocity caps require nonstructural constraint rows.
pub fn d6_coupled_stops(lower: &[Real], upper: &[Real], caps: &[Real]) -> bool {
    if lower.len() != 6 || upper.len() != 6 || caps.len() != 6 {
        return false;
    }
    let point = |axis: usize| {
        lower[axis].is_finite()
            && upper[axis].is_finite()
            && (upper[axis] - lower[axis]).abs() < 1e-12
    };
    let angular_fixed = (3..6).all(point);
    (0..6).any(|axis| {
        let structural = point(axis) && (axis < 3 || angular_fixed);
        !structural
            && (lower[axis].is_finite() || upper[axis].is_finite() || caps[axis].is_finite())
    })
}

/// Whether partial angular bounds or Cartesian velocity caps require constraint rows.
pub fn ball_coupled_stops(lower: &[Real], upper: &[Real], caps: &[Real]) -> bool {
    if lower.len() != 3 || upper.len() != 3 || caps.len() != 3 {
        return false;
    }
    let fixed = (0..3).all(|axis| {
        lower[axis].is_finite()
            && upper[axis].is_finite()
            && (upper[axis] - lower[axis]).abs() < 1e-12
    });
    !fixed
        && (0..3).any(|axis| {
            lower[axis].is_finite() || upper[axis].is_finite() || caps[axis].is_finite()
        })
}

/// Whether radial bounds or Cartesian velocity caps require constraint rows.
pub fn distance_coupled_stops(lower: &[Real], upper: &[Real], caps: &[Real]) -> bool {
    lower.len() == 6
        && upper.len() == 6
        && caps.len() == 6
        && (lower[0].is_finite() || upper[0].is_finite() || caps.iter().any(|cap| cap.is_finite()))
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
    /// Absolute generalized speed cap; None leaves the axis uncapped.
    pub velocity_limit: Option<Real>,
    pub locked: bool,
    pub motor: Option<JointMotor>,
    /// Brake damping (N·m·s/rad or N·s/m); zero releases the brake.
    pub brake: Real,
}

#[cfg(test)]
mod coupled_stop_tests {
    use super::*;

    #[test]
    fn ball_activation_keeps_partial_points_and_caps_nonstructural() {
        let lower = [Real::NEG_INFINITY; 3];
        let upper = [Real::INFINITY; 3];
        let caps = [Real::INFINITY; 3];
        assert!(!ball_coupled_stops(&lower, &upper, &caps));
        for axis in 0..3 {
            let mut lo = lower;
            let mut hi = upper;
            lo[axis] = 0.3;
            assert!(ball_coupled_stops(&lo, &hi, &caps));
            hi[axis] = lo[axis];
            assert!(ball_coupled_stops(&lo, &hi, &caps));
            hi[axis] += 1e-9;
            assert_eq!(lo[axis] as f32, hi[axis] as f32);
            assert!(ball_coupled_stops(&lo, &hi, &caps));
            let mut speed = caps;
            speed[axis] = 0.;
            assert!(ball_coupled_stops(&lower, &upper, &speed));
        }
        assert!(ball_coupled_stops(
            &[0., 0., Real::NEG_INFINITY],
            &[0., 0., Real::INFINITY],
            &caps
        ));
        assert!(!ball_coupled_stops(&[0.; 3], &[0.; 3], &[0.; 3]));
        assert!(!ball_coupled_stops(&[Real::NAN; 3], &[Real::NAN; 3], &caps));
        assert!(!ball_coupled_stops(&[], &[], &[]));
    }

    #[test]
    fn distance_activation_uses_radial_bounds_and_independent_caps() {
        let lower = [Real::NEG_INFINITY; 6];
        let upper = [Real::INFINITY; 6];
        let caps = [Real::INFINITY; 6];
        assert!(!distance_coupled_stops(&lower, &upper, &caps));
        for axis in 0..6 {
            let mut bounds = lower;
            bounds[axis] = 0.;
            assert_eq!(distance_coupled_stops(&bounds, &upper, &caps), axis == 0);
            let mut speed = caps;
            speed[axis] = 0.;
            assert!(distance_coupled_stops(&lower, &upper, &speed));
        }
        assert!(distance_coupled_stops(&[0.; 6], &[0.; 6], &caps));
        assert!(!distance_coupled_stops(
            &[Real::NAN; 6],
            &[Real::NAN; 6],
            &caps
        ));
        assert!(!distance_coupled_stops(&[], &[], &[]));
    }

    #[test]
    fn d6_activation_distinguishes_structural_points_intervals_and_caps() {
        let free_lower = [Real::NEG_INFINITY; 6];
        let free_upper = [Real::INFINITY; 6];
        let free_caps = [Real::INFINITY; 6];
        assert!(!d6_coupled_stops(&free_lower, &free_upper, &free_caps));
        for axis in 0..6 {
            let mut lower = free_lower;
            let mut upper = free_upper;
            lower[axis] = 1.5;
            upper[axis] = 1.5 + 1e-9;
            assert_eq!(lower[axis] as f32, upper[axis] as f32);
            assert!(d6_coupled_stops(&lower, &upper, &free_caps));
            upper[axis] = lower[axis];
            assert_eq!(d6_coupled_stops(&lower, &upper, &free_caps), axis >= 3);
            let mut caps = free_caps;
            caps[axis] = 0.;
            assert!(d6_coupled_stops(&free_lower, &free_upper, &caps));
            assert_eq!(d6_coupled_stops(&lower, &upper, &caps), axis >= 3);
        }
        let points = [0.; 6];
        assert!(!d6_coupled_stops(&points, &points, &points));
        assert!(!d6_coupled_stops(&[], &[], &[]));
    }
}
