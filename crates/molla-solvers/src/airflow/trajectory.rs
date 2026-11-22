use super::invalid;
use molla_core::Result;
use molla_math::Vec3;

/// Straight-line rest-to-rest motion bounds in m/s, m/s² and m/s³.
#[derive(Clone, Copy, Debug)]
pub struct MotionLimits {
    pub speed: f64,
    pub acceleration: f64,
    pub jerk: f64,
}

impl Default for MotionLimits {
    fn default() -> Self {
        Self {
            speed: 1.5,
            acceleration: 2.0,
            jerk: 5.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlightReference {
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub jerk: Vec3,
}

impl FlightReference {
    pub fn hold(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            jerk: Vec3::ZERO,
        }
    }
}

/// Seven-phase symmetric S-curve; endpoints have zero velocity and acceleration.
#[derive(Clone, Copy, Debug)]
pub struct FlightSegment {
    start: Vec3,
    end: Vec3,
    direction: Vec3,
    phases: [f64; 7],
    jerk: f64,
    duration: f64,
    stretch: f64,
}

impl FlightSegment {
    pub fn new(start: Vec3, end: Vec3, limits: MotionLimits) -> Result<Self> {
        if !start.is_finite()
            || !end.is_finite()
            || [limits.speed, limits.acceleration, limits.jerk]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid(
                "flight endpoints and positive motion bounds must be finite",
            ));
        }
        let distance = start.distance(end);
        if !distance.is_finite() {
            return Err(invalid("flight distance overflow"));
        }
        let j = limits.jerk;
        let mut tj = (limits.acceleration / j).min((limits.speed / j).sqrt());
        let mut ta = (limits.speed / (j * tj) - tj).max(0.0);
        let ramp_distance = j * tj * (tj + ta) * (2.0 * tj + ta);
        if distance < ramp_distance {
            if distance >= 2.0 * j * tj.powi(3) {
                ta = ((tj * tj + 4.0 * distance / (j * tj)).sqrt() - 3.0 * tj) * 0.5;
            } else {
                tj = (distance / (2.0 * j)).cbrt();
                ta = 0.0;
            }
        }
        let peak_speed = j * tj * (tj + ta);
        let tv = if peak_speed > 0.0 {
            (distance / peak_speed - 2.0 * tj - ta).max(0.0)
        } else {
            0.0
        };
        let phases = [tj, ta, tj, tv, tj, ta, tj];
        let duration = phases.iter().sum();
        if !phases.iter().all(|v| v.is_finite() && *v >= 0.0) || !f64::is_finite(duration) {
            return Err(invalid("flight profile overflow"));
        }
        Ok(Self {
            start,
            end,
            direction: (end - start).normalize_or_zero(),
            phases,
            jerk: j,
            duration,
            stretch: 1.0,
        })
    }

    pub fn duration(&self) -> f64 {
        self.duration * self.stretch
    }

    /// Slows a profile without exceeding its original motion bounds.
    pub fn with_duration(mut self, seconds: f64) -> Result<Self> {
        if !seconds.is_finite() || seconds < self.duration() || seconds <= 0.0 {
            return Err(invalid(
                "flight duration must be finite and no shorter than the bounded profile",
            ));
        }
        if self.duration > 0.0 {
            self.stretch = seconds / self.duration;
        } else {
            self.duration = seconds;
        }
        Ok(self)
    }

    pub fn sample(&self, seconds: f64) -> Result<FlightReference> {
        if !seconds.is_finite() {
            return Err(invalid("flight sample time must be finite"));
        }
        if seconds <= 0.0 {
            return Ok(FlightReference::hold(self.start));
        }
        if seconds >= self.duration() {
            return Ok(FlightReference::hold(self.end));
        }
        let mut remaining = seconds / self.stretch;
        let (mut position, mut velocity, mut acceleration) = (0.0, 0.0, 0.0);
        let mut jerk = 0.0;
        for (phase, sign) in self
            .phases
            .into_iter()
            .zip([1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        {
            let dt = remaining.min(phase);
            jerk = sign * self.jerk;
            position += velocity * dt + acceleration * dt * dt * 0.5 + jerk * dt.powi(3) / 6.0;
            velocity += acceleration * dt + jerk * dt * dt * 0.5;
            acceleration += jerk * dt;
            remaining -= dt;
            if remaining <= 0.0 {
                break;
            }
        }
        Ok(FlightReference {
            position: self.start + self.direction * position,
            velocity: self.direction * (velocity / self.stretch),
            acceleration: self.direction * (acceleration / self.stretch.powi(2)),
            jerk: self.direction * (jerk / self.stretch.powi(3)),
        })
    }
}
