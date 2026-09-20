use super::{Error, MassDecomposition, MassFactor, Real, Result};

impl MassFactor {
    pub(crate) fn solve_batch(&self, rhs: &[Vec<Real>]) -> Result<Vec<Vec<Real>>> {
        let n = self.n;
        if rhs
            .iter()
            .any(|row| row.len() != n || row.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::Solver("invalid impulse mass right-hand side".into()));
        }
        let MassDecomposition::Cholesky(factor) = &self.decomposition else {
            return rhs.iter().map(|row| self.solve(row)).collect();
        };
        const LANES: usize = 4;
        let full_count = rhs.len() / LANES * LANES;
        let mut result = Vec::with_capacity(rhs.len());
        if full_count != 0 {
            result.extend((0..full_count).map(|_| vec![0.0; n]));
            let mut work = vec![[0.0; LANES]; n];
            for start in (0..full_count).step_by(LANES) {
                for i in 0..n {
                    let mut values: [Real; LANES] =
                        std::array::from_fn(|lane| rhs[start + lane][i] / self.scale);
                    for k in 0..i {
                        let coefficient = factor[i * n + k];
                        for lane in 0..LANES {
                            values[lane] -= coefficient * work[k][lane];
                        }
                    }
                    let diagonal = factor[i * n + i];
                    if diagonal.abs() < 1e-30 {
                        return Err(Error::Build(format!(
                            "forward_sub: zero diagonal at row {i}"
                        )));
                    }
                    work[i] = values.map(|v| v / diagonal);
                }
                for i in (0..n).rev() {
                    let mut values = work[i];
                    for k in i + 1..n {
                        let coefficient = factor[k * n + i];
                        for lane in 0..LANES {
                            values[lane] -= coefficient * work[k][lane];
                        }
                    }
                    let diagonal = factor[i * n + i];
                    if diagonal.abs() < 1e-30 {
                        return Err(Error::Build(format!("back_sub: zero diagonal at row {i}")));
                    }
                    work[i] = values.map(|v| v / diagonal);
                    for lane in 0..LANES {
                        result[start + lane][i] = work[i][lane];
                    }
                }
            }
        }
        for row in &rhs[full_count..] {
            result.push(self.solve(row)?);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
