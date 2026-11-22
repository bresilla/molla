use super::AirConfig;
use molla_math::Vec3;

impl AirConfig {
    /// Curl of six time-varying potentials, with zero floor-normal velocity.
    pub fn turbulence(self, point: Vec3) -> Vec3 {
        if self.turbulence_strength == 0.0 {
            return Vec3::ZERO;
        }
        let waves = [
            Vec3::new(0.7, 1.7, -1.1),
            Vec3::new(-1.9, 3.9, 1.3),
            Vec3::new(3.7, 7.0, 2.9),
            Vec3::new(-7.1, 13.0, -4.3),
            Vec3::new(12.9, 22.0, -9.7),
            Vec3::new(-18.3, 34.0, 21.1),
        ];
        let axes = [
            Vec3::new(0.8, 0.6, -0.7),
            Vec3::new(-0.6, 0.9, 0.8),
            Vec3::new(0.9, -0.7, 0.5),
            Vec3::new(0.7, 0.8, -0.9),
            Vec3::new(-0.8, 0.6, 0.9),
            Vec3::new(0.6, -0.9, -0.7),
        ];
        let signed_height = point.y - self.grid.origin().y;
        let height = signed_height.abs();
        let envelope = (-height / 1.4).exp();
        let mut velocity = Vec3::ZERO;
        for (i, (k, a)) in waves.into_iter().zip(axes).enumerate() {
            let index = i as f64;
            let phase = k.x * point.x + k.z * point.z - (0.6 + 0.73 * index) * self.time
                + 1.71 * index
                + 0.6 * (0.17 * self.time + index).sin();
            let (s, c) = phase.sin_cos();
            let (sy, cy) = (k.y * height).sin_cos();
            let dy = k.y * cy - sy / 1.4;
            let scale = envelope * 0.9 / ((1.0 + 0.4 * index) * k.length());
            velocity += Vec3::new(
                a.z * c * dy + a.y * k.z * s * cy,
                sy * (a.x * k.z * c + a.z * k.x * s),
                -a.y * k.x * s * cy - a.x * s * dy,
            ) * scale;
        }
        velocity.y *= if signed_height < 0.0 { -1.0 } else { 1.0 };
        velocity * self.turbulence_strength
    }
}
