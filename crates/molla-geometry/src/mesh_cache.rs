use molla_sim::TriMesh;

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
}

impl MeshBvhCache {
    pub(crate) fn begin_query(&mut self, mesh_count: usize) {
        self.entries.resize_with(mesh_count, || None);
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
