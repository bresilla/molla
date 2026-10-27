//! Exact-input reuse of fixed-order transfer tiles, retaining every contribution.

use super::*;

pub(super) struct Buffers {
    pub(super) keys: wgpu::Buffer,
    pub(super) nodes: wgpu::Buffer,
    pub(super) tiles: wgpu::Buffer,
    pub(super) header: wgpu::Buffer,
    pub(super) tile_groups: u32,
}

pub(super) fn sizes(capacity: u32, grid: u32, limits: &wgpu::Limits) -> Result<[u64; 4]> {
    let nodes = u64::from(grid)
        .checked_pow(3)
        .filter(|&n| n > 0 && n <= u64::from(u32::MAX / 5));
    let tiles = u64::from(grid.div_ceil(4)).checked_pow(3);
    let (Some(nodes), Some(tiles)) = (nodes, tiles) else {
        return Err(Error::Solver("Grid reuse index overflow".into()));
    };
    let sizes = [u64::from(capacity.max(1)) * 112, nodes * 20, tiles * 8, 64];
    if sizes
        .iter()
        .any(|&s| s > limits.max_buffer_size || s > limits.max_storage_buffer_binding_size)
        || tiles.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        || capacity.max(1).div_ceil(64) > limits.max_compute_workgroups_per_dimension
    {
        return Err(Error::Solver(
            "Grid reuse exceeds storage/dispatch limits".into(),
        ));
    }
    Ok(sizes)
}

impl Buffers {
    pub(super) fn new(sim: &MpmSimulation) -> Result<Self> {
        let sizes = sizes(
            sim.cfg.max_particles,
            sim.cfg.grid_size,
            &sim.device.limits(),
        )?;
        let buffer = |label, size| {
            molla_core::gpu_create_buffer!(
                &sim.device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }
            )
        };
        Ok(Self {
            keys: buffer("gather reuse particle keys", sizes[0]),
            nodes: buffer("gather reuse node values", sizes[1]),
            tiles: buffer("gather reuse tile validity", sizes[2]),
            header: buffer("gather reuse signature", sizes[3]),
            tile_groups: sim.cfg.grid_size.div_ceil(4).pow(3).div_ceil(64),
        })
    }
    pub(super) fn bytes(&self) -> u64 {
        self.keys.size() + self.nodes.size() + self.tiles.size() + self.header.size()
    }
}

pub(super) fn mask_source(mut source: String) -> Result<String> {
    let marker = "    let tile_id = shared_tile.x;";
    if source.matches(marker).count() != 1 {
        return Err(Error::Solver(
            "Grid reuse tile source contract changed".into(),
        ));
    }
    source = source.replace(marker, &format!("{marker}
    if lane == 0u {{ reuse_tile_hit = select(0u, 1u, atomicLoad(&reuse_header[15]) == 0u && atomicLoad(&reuse_tiles[tile_id].dirty) == 0u); }}
    let reuse_hit = workgroupUniformLoad(&reuse_tile_hit);
    if reuse_hit != 0u {{ reuse_restore_node(tile_id, lane, gid); return; }}"));
    let marker = "    atomicStore(&g_kdiag[index],";
    if source.matches(marker).count() != 1 {
        return Err(Error::Solver(
            "Grid reuse output source contract changed".into(),
        ));
    }
    let at = source.find(marker).unwrap();
    let end = at + source[at..].find(';').unwrap() + 1;
    source.insert_str(end, "
    reuse_nodes[world * 5u] = bitcast<u32>(atomicLoad(&g_mass[index]));
    for (var a = 0u; a < 3u; a++) { reuse_nodes[world * 5u + 1u + a] = bitcast<u32>(atomicLoad(&g_momentum[index * 3u + a])); }
    reuse_nodes[world * 5u + 4u] = atomicLoad(&g_kdiag[index]);
    if lane == 0u { atomicStore(&reuse_tiles[tile_id].valid, 1u); }");
    source.push_str(include_str!("../shaders/mpm_gather_grid_reuse.wgsl"));
    Ok(source)
}

#[cfg(test)]
#[path = "mpm_gather_grid_reuse_tests.rs"]
mod tests;
