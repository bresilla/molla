use molla_core::{Error, Result};
use molla_math::Real;
use molla_storage::BufferElement;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JointLimitReference {
    pub bounds: [f32; 4],
    pub full: [f32; 4],
    pub full_low: [f32; 4],
    pub half: [f32; 4],
    pub half_low: [f32; 4],
}

impl JointLimitReference {
    pub fn new(lower: Real, upper: Real) -> Result<Self> {
        let split = |value: Real| -> Result<[f32; 2]> {
            let high = value as f32;
            if value.is_finite() && !high.is_finite() {
                return Err(Error::Build(
                    "authored joint limit exceeds f32 range".into(),
                ));
            }
            Ok([
                high,
                if value.is_finite() {
                    (value - high as Real) as f32
                } else {
                    0.
                },
            ])
        };
        let lo = split(lower)?;
        let hi = split(upper)?;
        let mut reference = Self {
            bounds: [lo[0], lo[1], hi[0], hi[1]],
            ..Self::default()
        };
        for (side, angle) in [lower, upper].into_iter().enumerate() {
            for (factor, high, low) in [
                (1., &mut reference.full, &mut reference.full_low),
                (0.5, &mut reference.half, &mut reference.half_low),
            ] {
                let (sine, cosine) = if angle.is_finite() {
                    (angle * factor).sin_cos()
                } else {
                    (0., 1.)
                };
                for (lane, value) in [sine, cosine].into_iter().enumerate() {
                    let pair = split(value)?;
                    high[2 * side + lane] = pair[0];
                    low[2 * side + lane] = pair[1];
                }
            }
        }
        Ok(reference)
    }
}

impl BufferElement for JointLimitReference {
    type Gpu = Self;
    fn to_gpu(&self) -> Self {
        *self
    }
    fn from_gpu(value: &Self) -> Self {
        *value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_endpoints_and_angle_references_retain_low_bits() {
        let lower = 1.5;
        let upper = lower + 1e-9;
        let reference = JointLimitReference::new(lower, upper).unwrap();
        assert_eq!(std::mem::size_of::<JointLimitReference>(), 80);
        assert_eq!(reference.bounds[0], reference.bounds[2]);
        assert!(reference.bounds[3] > reference.bounds[1]);
        for (side, angle) in [lower, upper].into_iter().enumerate() {
            for (factor, high, low) in [
                (1., reference.full, reference.full_low),
                (0.5, reference.half, reference.half_low),
            ] {
                let values = (angle * factor).sin_cos();
                for (lane, expected) in [values.0, values.1].into_iter().enumerate() {
                    assert!(
                        (high[2 * side + lane] as f64 + low[2 * side + lane] as f64 - expected)
                            .abs()
                            < 2e-15
                    );
                }
            }
        }
        let unlimited = JointLimitReference::new(f64::NAN, f64::INFINITY).unwrap();
        assert!(unlimited.bounds[0].is_nan() && unlimited.bounds[2].is_infinite());
        assert_eq!(unlimited.full, [0., 1., 0., 1.]);
        assert!(JointLimitReference::new(f64::MAX, f64::MAX).is_err());
    }
}
