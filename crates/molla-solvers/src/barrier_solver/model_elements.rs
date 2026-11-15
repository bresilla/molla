use molla_core::{Error, Result};
use molla_math::{
    Mat3,
    constitutive::{ElasticMaterial, ElasticModel},
};
use molla_sim::{ElasticElement, ElasticElementSet, Model, SoftSurfaceKind};

fn invalid(detail: &str) -> Error {
    Error::Solver(format!("barrier solver: {detail}"))
}

fn legacy_elements(m: &Model) -> Result<Vec<ElasticElement>> {
    let count = m.tet_count;
    if m.tet_indices.len()
        != count
            .checked_mul(4)
            .ok_or_else(|| invalid("tet count overflow"))?
    {
        return Err(invalid("invalid tet index length"));
    }
    for length in [
        m.tet_dm_inv.len(),
        m.tet_rest_volume.len(),
        m.tet_mu.len(),
        m.tet_lambda.len(),
        m.tet_viscosity.len(),
        m.tet_yield_stress.len(),
        m.tet_hardening.len(),
        m.tet_kinematic_hardening.len(),
        m.tet_f_plastic.len(),
        m.tet_plastic_strain.len(),
    ] {
        if length != count {
            return Err(invalid("invalid tet material array length"));
        }
    }
    let ids = m.tet_indices.host()?;
    let inverse = m.tet_dm_inv.host()?;
    let volume = m.tet_rest_volume.host()?;
    let mu = m.tet_mu.host()?;
    let lambda = m.tet_lambda.host()?;
    let viscosity = m.tet_viscosity.host()?;
    let yield_stress = m.tet_yield_stress.host()?;
    let hardening = m.tet_hardening.host()?;
    let kinematic = m.tet_kinematic_hardening.host()?;
    let plastic = m.tet_f_plastic.host()?;
    let strain = m.tet_plastic_strain.host()?;
    let mut elements = Vec::with_capacity(count);
    for t in 0..count {
        viscosity[t].validate()?;
        if viscosity[t].shear != 0.0 || viscosity[t].bulk != 0.0 {
            return Err(invalid("legacy tet viscosity is not integrated"));
        }
        if yield_stress[t] != f64::INFINITY
            || hardening[t] != 0.0
            || kinematic[t] != 0.0
            || plastic[t] != Mat3::IDENTITY
            || strain[t] != 0.0
        {
            return Err(invalid("legacy tet plasticity is not integrated"));
        }
        if !lambda[t].is_finite() {
            return Err(invalid("nonfinite tet lambda"));
        }
        elements.push(ElasticElement::from_rest_coordinates(
            &ids[4 * t..4 * t + 4],
            inverse[t],
            volume[t],
            ElasticMaterial {
                model: ElasticModel::StableNeoHookean,
                mu: mu[t],
                lambda: lambda[t] + mu[t],
            },
        )?);
    }
    Ok(elements)
}

pub(super) fn extract(m: &Model) -> Result<(ElasticElementSet, Vec<[u32; 3]>)> {
    let mut legacy = legacy_elements(m)?;
    let elements = if legacy.is_empty() {
        m.elastic_elements().clone()
    } else {
        legacy.extend_from_slice(m.elastic_elements().as_slice());
        let initial = m.initial_particle_q.host()?;
        if initial.len() != m.particle_count {
            return Err(invalid("invalid initial particle length"));
        }
        ElasticElementSet::new(initial, legacy)?
    };
    let mut from_tets = vec![false; m.soft_surface_tris.len()];
    for surface in &m.soft_surfaces {
        if surface.kind == SoftSurfaceKind::TetMesh && m.tet_count > 0 {
            let start = surface.tri_offset as usize;
            let end = start
                .checked_add(surface.tri_count as usize)
                .ok_or_else(|| invalid("surface range overflow"))?;
            if end > from_tets.len() {
                return Err(invalid("invalid tet surface range"));
            }
            from_tets[start..end].fill(true);
        }
    }
    let mut triangles: Vec<_> = m
        .soft_surface_tris
        .iter()
        .zip(from_tets)
        .filter_map(|(t, skip)| (!skip).then_some(*t))
        .collect();
    triangles.extend_from_slice(elements.triangles());
    let mut seen = std::collections::BTreeSet::new();
    triangles.retain(|t| {
        let mut key = *t;
        key.sort_unstable();
        seen.insert(key)
    });
    Ok((elements, triangles))
}
