use molla_core::{Error, Result};
use molla_math::SpatialVector;
use molla_sim::Control;

pub(crate) fn add_control(control: &Control, forces: &mut [SpatialVector]) -> Result<()> {
    if control.body_external_f.is_empty() {
        return Ok(());
    }
    let supplied = control.body_external_f.host()?;
    if supplied.len() != forces.len() {
        return Err(Error::Solver(
            "control body wrench count does not match bodies".into(),
        ));
    }
    for (total, applied) in forces.iter_mut().zip(supplied) {
        *total = *total + *applied;
        if !total.linear.is_finite() || !total.angular.is_finite() {
            return Err(Error::Solver("non-finite total body wrench".into()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::Vec3;
    use molla_storage::{Buffer, BufferUsage};

    #[test]
    fn body_control_addition_is_optional_and_keeps_authored_inputs() {
        let persistent = SpatialVector::new(Vec3::X, Vec3::Z);
        let mut forces = [persistent];
        let mut control = Control::empty();
        add_control(&control, &mut forces).unwrap();
        assert_eq!(forces[0], persistent);
        let supplied = SpatialVector::new(Vec3::Y * 2., Vec3::X * 3.);
        control.body_external_f = Buffer::from_vec(vec![supplied], BufferUsage::Storage);
        add_control(&control, &mut forces).unwrap();
        assert_eq!(forces[0], persistent + supplied);
        assert_eq!(control.body_external_f.host().unwrap(), &[supplied]);
    }

    #[test]
    fn body_control_addition_rejects_mismatched_or_nonfinite_inputs() {
        let mut control = Control::empty();
        control.body_external_f =
            Buffer::from_vec(vec![SpatialVector::ZERO; 2], BufferUsage::Storage);
        assert!(add_control(&control, &mut [SpatialVector::ZERO]).is_err());
        for value in [f64::NAN, f64::INFINITY, f64::MAX] {
            control.body_external_f = Buffer::from_vec(
                vec![SpatialVector::new(Vec3::X * value, Vec3::ZERO)],
                BufferUsage::Storage,
            );
            assert!(
                add_control(
                    &control,
                    &mut [SpatialVector::new(Vec3::X * f64::MAX, Vec3::ZERO)]
                )
                .is_err()
            );
        }
    }
}
