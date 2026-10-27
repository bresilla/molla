//! Indexed-mesh convex decomposition for runtime collider authoring.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::runtime::{ColliderGeometry, validate_convex_points};
use parry3d_f64::{
    math::Vector,
    transformation::vhacd::{VHACD, VHACDParameters},
};

#[derive(Clone, Copy, Debug)]
pub struct DecompositionOptions {
    pub resolution: u32,
    pub concavity: Real,
    pub max_parts: u32,
}

impl Default for DecompositionOptions {
    fn default() -> Self {
        let p = VHACDParameters::default();
        Self {
            resolution: p.resolution,
            concavity: p.concavity,
            max_parts: p.max_convex_hulls,
        }
    }
}

/// Run VHACD and reconstruct convex hulls from the indexed surface, not voxel boundaries.
pub fn convex_decomposition(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    options: DecompositionOptions,
) -> Result<ColliderGeometry> {
    if !(8..=256).contains(&options.resolution)
        || !(1..=1024).contains(&options.max_parts)
        || !options.concavity.is_finite()
        || !(0.0..=1.0).contains(&options.concavity)
        || indices.len() < 4
        || vertices.len() > u32::MAX as usize
        || vertices.iter().any(|p| !p.is_finite())
        || indices
            .iter()
            .flatten()
            .any(|&i| i as usize >= vertices.len())
    {
        return Err(Error::Build("invalid decomposition mesh or options".into()));
    }
    let mut used = BTreeMap::new();
    for &i in indices.iter().flatten() {
        used.insert(i, 0);
    }
    let mut points = Vec::with_capacity(used.len());
    for (&i, remapped) in &mut used {
        *remapped = points.len() as u32;
        points.push(vertices[i as usize]);
    }
    let indices: Vec<[u32; 3]> = indices.iter().map(|face| face.map(|i| used[&i])).collect();
    let lo = points
        .iter()
        .copied()
        .fold(Vec3::splat(Real::INFINITY), Vec3::min);
    let hi = points
        .iter()
        .copied()
        .fold(Vec3::splat(Real::NEG_INFINITY), Vec3::max);
    let scale = (hi - lo).max_element();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Build("degenerate decomposition mesh".into()));
    }
    let center = lo * 0.5 + hi * 0.5;
    let normalized: Vec<Vec3> = points.iter().map(|p| (*p - center) / scale).collect();
    validate_convex_points(&normalized)?;
    if indices.iter().any(|[a, b, c]| {
        (normalized[*b as usize] - normalized[*a as usize])
            .cross(normalized[*c as usize] - normalized[*a as usize])
            .length_squared()
            == 0.0
    }) {
        return Err(Error::Build("degenerate decomposition triangles".into()));
    }
    let points: Vec<Vector> = normalized
        .iter()
        .map(|p| Vector::new(p.x, p.y, p.z))
        .collect();
    let params = VHACDParameters {
        resolution: options.resolution,
        concavity: options.concavity,
        max_convex_hulls: options.max_parts,
        ..VHACDParameters::default()
    };
    let hulls = std::panic::catch_unwind(|| {
        VHACD::decompose(&params, &points, &indices, true)
            .compute_exact_convex_hulls(&points, &indices)
    })
    .map_err(|_| Error::Build("convex decomposition failed".into()))?;
    let mut parts = Vec::new();
    for (points, triangles) in hulls {
        if points.len() < 4 || triangles.len() < 4 {
            continue;
        }
        let mut points: Vec<Vec3> = points
            .into_iter()
            .map(|p| Vec3::new(p.x, p.y, p.z) * scale + center)
            .collect();
        if points.iter().any(|p| !p.is_finite()) {
            return Err(Error::Build("non-finite decomposition output".into()));
        }
        points.sort_by(point_order);
        points.dedup();
        if validate_convex_points(&points).is_ok() {
            parts.push(points);
        }
    }
    parts.sort_by(|a, b| {
        a.iter()
            .zip(b)
            .map(|(a, b)| point_order(a, b))
            .find(|c| !c.is_eq())
            .unwrap_or_else(|| a.len().cmp(&b.len()))
    });
    if parts.is_empty() {
        return Err(Error::Build(
            "decomposition produced no solid pieces".into(),
        ));
    }
    Ok(ColliderGeometry::ConvexParts(parts))
}

fn point_order(a: &Vec3, b: &Vec3) -> std::cmp::Ordering {
    a.x.total_cmp(&b.x)
        .then(a.y.total_cmp(&b.y))
        .then(a.z.total_cmp(&b.z))
}
