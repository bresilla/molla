//! Per-wheel kinematic state types.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

/// Kinematic state of one wheel.
#[derive(Clone, Copy, Debug)]
pub struct RealWheelState {
    pub hub_pos: [f64; 3],
    pub hub_vel: [f64; 3],
    pub heading: [f64; 3],
    /// Wheel angular velocity (rad/s). Positive means rolling forward along
    /// `heading`.
    pub omega: f64,
}

/// Optional wheel orientation extension for camber-aware tire contact.
///
/// `RealWheelState::heading` still defines the rolling direction projected
/// onto the ground. `wheel_up` is the wheel plane's local up/radial direction;
/// when it tilts away from the terrain normal the brush contact adds camber
/// thrust and an asymmetric normal-pressure bias across the patch.
#[derive(Clone, Copy, Debug)]
pub struct RealWheelOrientation {
    pub wheel_up: [f64; 3],
}

impl Default for RealWheelOrientation {
    fn default() -> Self {
        Self {
            wheel_up: [0.0, 1.0, 0.0],
        }
    }
}
