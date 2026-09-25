//! Fallible fixed-capacity host arrays for sensor construction.

use molla_core::{Error, Result};

pub(crate) fn reserved<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Storage("sensor host array allocation failed".into()))?;
    Ok(values)
}

pub(crate) fn filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
    let mut values = reserved(count)?;
    values.resize(count, value);
    Ok(values)
}

pub(crate) fn from_fn<T>(count: usize, f: impl FnMut(usize) -> T) -> Result<Vec<T>> {
    let mut values = reserved(count)?;
    values.extend((0..count).map(f));
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_arrays_preserve_values_capacity_and_empty_inputs() {
        assert!(reserved::<u32>(0).unwrap().is_empty());
        assert_eq!(reserved::<u32>(7).unwrap().capacity(), 7);
        assert_eq!(filled(3, 7u32).unwrap(), [7, 7, 7]);
        assert_eq!(from_fn(4, |i| i * 3).unwrap(), [0, 3, 6, 9]);
        assert!(
            from_fn::<u32>(0, |_| panic!("empty array callback"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn capacity_errors_precede_element_initialization() {
        assert!(reserved::<u32>(usize::MAX).is_err());
        assert!(filled(usize::MAX, 0u32).is_err());
        assert!(from_fn::<u32>(usize::MAX, |_| panic!("overflow callback")).is_err());
    }
}
