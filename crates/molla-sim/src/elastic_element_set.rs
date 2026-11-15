use crate::{ElasticElement, compute_tet_surface_triangles};
use molla_core::Result;
use molla_math::Vec3;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Debug, Default)]
struct Data {
    elements: Vec<ElasticElement>,
    triangles: Vec<[u32; 3]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::constitutive::{ElasticMaterial, ElasticModel};

    #[test]
    fn shared_tet_faces_and_duplicate_elements_do_not_duplicate_boundaries() {
        let x = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z, -Vec3::Z];
        let m = ElasticMaterial {
            model: ElasticModel::StableNeoHookean,
            mu: 10.0,
            lambda: 10.0,
        };
        let a = ElasticElement::tetrahedron(&x, [0, 1, 2, 3], m).unwrap();
        let b = ElasticElement::tetrahedron(&x, [0, 1, 2, 4], m).unwrap();
        let set = ElasticElementSet::new(&x, vec![a.clone(), a, b]).unwrap();
        assert_eq!(set.as_slice().len(), 3);
        assert_eq!(set.triangles().len(), 6);
        assert!(set.same_revision(&set.clone()));
        let rebuilt = ElasticElementSet::new(&x, set.as_slice().to_vec()).unwrap();
        assert!(!set.same_revision(&rebuilt));
        assert_eq!(set.triangles(), rebuilt.triangles());
    }
}

/// Immutable material elements and their membrane/tet boundary triangles.
#[derive(Clone, Debug, Default)]
pub struct ElasticElementSet(Arc<Data>);

impl ElasticElementSet {
    pub fn new(positions: &[Vec3], elements: Vec<ElasticElement>) -> Result<Self> {
        let mut triangles = Vec::new();
        let mut tets = BTreeMap::new();
        for e in &elements {
            e.validate(positions)?;
            let ids = e.vertices();
            if e.dimensions() == 2 {
                triangles.push([ids[0], ids[1], ids[2]]);
            } else {
                let mut tet = [ids[0], ids[1], ids[2], ids[3]];
                let mut key = tet;
                key.sort_unstable();
                if e.inverse_rest().determinant() < 0.0 {
                    tet.swap(0, 1);
                }
                tets.entry(key).or_insert(tet);
            }
        }
        triangles.extend(compute_tet_surface_triangles(
            &tets.into_values().collect::<Vec<_>>(),
        ));
        let mut seen = BTreeSet::new();
        triangles.retain(|t| {
            let mut key = *t;
            key.sort_unstable();
            seen.insert(key)
        });
        Ok(Self(Arc::new(Data {
            elements,
            triangles,
        })))
    }

    pub fn as_slice(&self) -> &[ElasticElement] {
        &self.0.elements
    }
    pub fn is_empty(&self) -> bool {
        self.0.elements.is_empty()
    }
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.0.triangles
    }

    /// Compare immutable snapshots without traversing element data.
    pub fn same_revision(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
