use molla_sim::TriMesh;
use molla_sim::runtime::SceneStamp;

use crate::mesh_bvh::MeshBvh;

struct CachedMesh {
    source: TriMesh,
    bvh: MeshBvh,
    checked: bool,
}

#[derive(Default)]
pub(crate) struct MeshBvhCache {
    entries: Vec<Option<CachedMesh>>,
    builds: usize,
    /// The compiled model the entries were last checked against.
    stamp: Option<SceneStamp>,
}

impl MeshBvhCache {
    /// Start a query. Under the model the entries were checked against,
    /// they stand without comparing their geometry again.
    pub(crate) fn begin_query(&mut self, mesh_count: usize, stamp: Option<&SceneStamp>) {
        self.entries.resize_with(mesh_count, || None);
        if stamp.is_some() && stamp == self.stamp.as_ref() {
            return;
        }
        self.stamp = stamp.cloned();
        for entry in self.entries.iter_mut().flatten() {
            entry.checked = false;
        }
    }

    pub(crate) fn builds(&self) -> usize {
        self.builds
    }

    pub(crate) fn get(&mut self, id: usize, mesh: &TriMesh) -> &MeshBvh {
        let entry = &mut self.entries[id];
        let unchanged = entry.as_ref().is_some_and(|cached| {
            cached.checked
                || (cached.source.vertices == mesh.vertices
                    && cached.source.indices == mesh.indices)
        });
        if !unchanged {
            *entry = Some(CachedMesh {
                source: mesh.clone(),
                bvh: MeshBvh::build(&mesh.vertices, &mesh.indices, 4),
                checked: true,
            });
            self.builds += 1;
        }
        let cached = entry.as_mut().unwrap();
        cached.checked = true;
        &cached.bvh
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::Vec3;
    use molla_sim::runtime::RigidScene;

    fn mesh(x: f64) -> TriMesh {
        TriMesh {
            vertices: vec![Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 0.0, 0.0), Vec3::new(x, 0.0, 1.0)],
            indices: vec![[0, 1, 2]],
        }
    }

    #[test]
    fn a_model_s_meshes_are_compared_once_and_again_when_the_model_changes() {
        let (one, other) = (RigidScene::new().unwrap(), RigidScene::new().unwrap());
        let mut cache = MeshBvhCache::default();
        cache.begin_query(1, Some(&one.model_stamp()));
        cache.get(0, &mesh(0.0));
        cache.begin_query(1, Some(&one.model_stamp()));
        cache.get(0, &mesh(5.0));
        assert_eq!(cache.builds(), 1);
        cache.begin_query(1, Some(&other.model_stamp()));
        cache.get(0, &mesh(5.0));
        assert_eq!(cache.builds(), 2);
        cache.begin_query(1, None);
        cache.get(0, &mesh(9.0));
        assert_eq!(cache.builds(), 3);
    }
}
