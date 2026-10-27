use super::{Error, Result, TireFemGpuInput};

fn invalid(kind: &str) -> Error {
    Error::Solver(format!("tire_fem: invalid {kind} topology"))
}

fn adjacency(
    nodes: usize,
    width: usize,
    count: usize,
    vertex: impl Fn(usize, usize) -> u32,
    offsets: &[u32],
    entries: &[u32],
    kind: &str,
) -> Result<()> {
    let expected = count.checked_mul(width).ok_or_else(|| invalid(kind))?;
    if expected > u32::MAX as usize || nodes >= u32::MAX as usize {
        return Err(invalid(kind));
    }
    if count == 0 && offsets.is_empty() && entries.is_empty() {
        return Ok(());
    }
    if offsets.len() != nodes + 1
        || offsets.first() != Some(&0)
        || offsets.last().copied().map(|n| n as usize) != Some(expected)
        || entries.len() != expected
        || offsets.windows(2).any(|p| p[0] > p[1])
    {
        return Err(invalid(kind));
    }
    for constraint in 0..count {
        for local in 0..width {
            let node = vertex(constraint, local);
            if node as usize >= nodes || (0..local).any(|i| vertex(constraint, i) == node) {
                return Err(invalid(kind));
            }
        }
    }
    let mut seen = vec![false; expected];
    for (node, range) in offsets.windows(2).enumerate() {
        for &entry in &entries[range[0] as usize..range[1] as usize] {
            let entry = entry as usize;
            if entry >= expected
                || seen[entry]
                || vertex(entry / width, entry % width) as usize != node
            {
                return Err(invalid(kind));
            }
            seen[entry] = true;
        }
    }
    Ok(())
}

fn node_set(nodes: usize, ids: &[u32], kind: &str) -> Result<()> {
    let mut seen = vec![false; nodes];
    for &id in ids {
        let id = id as usize;
        if id >= nodes || seen[id] {
            return Err(invalid(kind));
        }
        seen[id] = true;
    }
    Ok(())
}

pub(super) fn validate(input: &TireFemGpuInput) -> Result<()> {
    let n = input.positions.len();
    adjacency(
        n,
        4,
        input.tet_indices.len() / 4,
        |tet, local| input.tet_indices[4 * tet + local],
        &input.node_tet_offsets,
        &input.node_tet_entries,
        "tetrahedron",
    )?;
    adjacency(
        n,
        2,
        input.xpbd_edge_indices.len(),
        |edge, local| input.xpbd_edge_indices[edge][local],
        &input.node_edge_offsets,
        &input.node_edge_entries,
        "edge",
    )?;
    adjacency(
        n,
        4,
        input.xpbd_dihedral_indices.len(),
        |hinge, local| input.xpbd_dihedral_indices[hinge][local],
        &input.node_dihedral_offsets,
        &input.node_dihedral_entries,
        "dihedral",
    )?;
    node_set(n, &input.bead_ids, "bead")?;
    node_set(n, &input.tread_ids, "tread")
}
