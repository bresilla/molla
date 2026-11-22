use molla_core::{Error, Result};
use std::collections::BTreeSet;

/// Unique elastic-stencil indices touching each queried triangle.
pub fn elastic_stencil_neighborhoods(
    vertex_count: usize,
    stencils: &[Vec<u32>],
    faces: &[[u32; 3]],
) -> Result<Vec<Vec<u32>>> {
    let invalid = || Error::Solver("invalid elastic stencil adjacency".into());
    let mut incident = vec![Vec::new(); vertex_count];
    for (index, ids) in stencils.iter().enumerate() {
        let index = u32::try_from(index).map_err(|_| invalid())?;
        for &v in ids {
            incident.get_mut(v as usize).ok_or_else(invalid)?.push(index);
        }
    }
    faces.iter().map(|face| {
        let mut neighbors = BTreeSet::new();
        for &v in face {
            neighbors.extend(incident.get(v as usize).ok_or_else(invalid)?.iter().copied());
        }
        Ok(neighbors.into_iter().collect())
    }).collect()
}
