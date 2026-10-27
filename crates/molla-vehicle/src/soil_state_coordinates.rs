use super::SoilStateError;

#[derive(Clone, Copy)]
struct Expansion {
    words: [f64; 8],
    len: usize,
}

impl Expansion {
    fn add(&mut self, mut value: f64) -> Result<(), SoilStateError> {
        if !value.is_finite() {
            return Err(SoilStateError::UnrepresentablePosition);
        }
        let mut used = 0;
        for i in 0..self.len {
            let term = self.words[i];
            let sum = value + term;
            let split = sum - value;
            let error = (value - (sum - split)) + (term - split);
            if !sum.is_finite() || !error.is_finite() {
                return Err(SoilStateError::UnrepresentablePosition);
            }
            if error != 0.0 {
                self.words[used] = error;
                used += 1;
            }
            value = sum;
        }
        if used == self.words.len() {
            return Err(SoilStateError::UnrepresentablePosition);
        }
        self.words[used] = value;
        self.len = used + 1;
        Ok(())
    }

    fn leading(&self) -> f64 {
        self.words[..self.len]
            .iter()
            .rev()
            .copied()
            .find(|&x| x != 0.0)
            .unwrap_or(0.0)
    }

    fn boundary_sign(mut self, tile: f64, edge: f64) -> Result<i8, SoilStateError> {
        let product = tile * edge;
        let error = tile.mul_add(edge, -product);
        self.add(-product)?;
        self.add(-error)?;
        let leading = self.leading();
        Ok(if leading < 0.0 {
            -1
        } else if leading > 0.0 {
            1
        } else {
            0
        })
    }
}

pub(super) fn coordinate(
    local: f64,
    frame: f64,
    origin: f64,
    edge: f64,
) -> Result<i64, SoilStateError> {
    let mut distance = Expansion {
        words: [0.0; 8],
        len: 0,
    };
    distance.add(frame)?;
    distance.add(-origin)?;
    distance.add(local)?;
    let mut tile = (distance.leading() / edge).floor();
    let limit = (1u64 << 52) as f64;
    if !tile.is_finite() || tile.abs() > limit + 1.0 {
        return Err(SoilStateError::UnrepresentablePosition);
    }
    for _ in 0..3 {
        if distance.boundary_sign(tile, edge)? < 0 {
            tile -= 1.0;
        } else if distance.boundary_sign(tile + 1.0, edge)? >= 0 {
            tile += 1.0;
        } else if tile.abs() < limit {
            return Ok(tile as i64);
        } else {
            break;
        }
    }
    Err(SoilStateError::UnrepresentablePosition)
}

#[cfg(test)]
#[path = "soil_state_coordinates_tests.rs"]
mod tests;
