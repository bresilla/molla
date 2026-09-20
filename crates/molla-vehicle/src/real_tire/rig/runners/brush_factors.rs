use super::{brush_load_sensitivity, brush_pressure_stiffness_factor};

pub(super) struct BrushFactors<const CACHED: bool> {
    pressure_pa: f64,
    inflation: Option<f64>,
    previous: Option<(u64, (f64, f64))>,
}

impl<const CACHED: bool> BrushFactors<CACHED> {
    pub(super) fn new(pressure_pa: f64) -> Self {
        Self {
            pressure_pa,
            inflation: None,
            previous: None,
        }
    }

    pub(super) fn inflation(&mut self) -> f64 {
        if CACHED {
            *self
                .inflation
                .get_or_insert_with(|| brush_pressure_stiffness_factor(self.pressure_pa))
        } else {
            brush_pressure_stiffness_factor(self.pressure_pa)
        }
    }

    pub(super) fn load(&mut self, normal_pressure: f64) -> (f64, f64) {
        let key = normal_pressure.to_bits();
        if CACHED
            && let Some((previous, factors)) = self.previous
            && previous == key
        {
            return factors;
        }
        let factors = brush_load_sensitivity(normal_pressure);
        if CACHED {
            self.previous = Some((key, factors));
        }
        factors
    }
}
