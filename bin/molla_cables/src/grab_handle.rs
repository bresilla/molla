use molla_math::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct GrabHandle {
    pub point: Vec3,
    pub velocity: Vec3,
}

impl GrabHandle {
    pub fn new(point: Vec3) -> Self {
        Self {
            point,
            velocity: Vec3::ZERO,
        }
    }

    pub fn advance(&mut self, target: Vec3, dt: f64) {
        let acceleration =
            ((target - self.point) * 400.0 - self.velocity * 40.0).clamp_length_max(12.0);
        self.velocity = (self.velocity + acceleration * dt).clamp_length_max(1.5);
        self.point += self.velocity * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jumps_reversals_and_hold_have_bounded_speed_and_acceleration() {
        let mut handle = GrabHandle::new(Vec3::ZERO);
        let dt = 1.0 / 240.0;
        for step in 0..1440 {
            let target = if step < 120 {
                Vec3::X * 10.0
            } else if step < 240 {
                -Vec3::X * 10.0
            } else {
                Vec3::ZERO
            };
            let previous = handle;
            handle.advance(target, dt);
            assert!(handle.velocity.length() <= 1.5 + 1e-12);
            assert!(handle.velocity.distance(previous.velocity) <= 12.0 * dt + 1e-12);
            assert!(handle.point.distance(previous.point) <= 1.5 * dt + 1e-12);
        }
        assert!(handle.point.length() < 1e-8);
        assert!(handle.velocity.length() < 1e-8);
    }
}
