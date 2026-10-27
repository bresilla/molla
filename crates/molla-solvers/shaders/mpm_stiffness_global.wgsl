@group(1) @binding(12) var<storage, read_write> stiffness_global_tail: array<atomic<u32>>;

fn scatter_stiffness_global_tail(index: u32, value: f32) {
    var old_bits = atomicLoad(&stiffness_global_tail[index]);
    loop {
        let next = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let result = atomicCompareExchangeWeak(&stiffness_global_tail[index], old_bits, next);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

fn atomic_add_kdiag(idx: u32, val: f32) {
    var old_bits = atomicLoad(&g_kdiag[idx]);
    loop {
        let old = bitcast<f32>(old_bits);
        let next = scatter_stiffness_round(old + val);
        let result = atomicCompareExchangeWeak(&g_kdiag[idx], old_bits, bitcast<u32>(next));
        if result.exchanged {
            let large = max(old, val);
            let small = min(old, val);
            let error = scatter_stiffness_round(scatter_stiffness_round(large - next) + small);
            scatter_stiffness_global_tail(idx, error);
            break;
        }
        old_bits = result.old_value;
    }
}

@compute @workgroup_size(64)
fn resolve_stiffness(@builtin(global_invocation_id) gid: vec3<u32>) {
    let world = grid_cell_id(gid);
    if world >= u.grid_total || !grid_valid() { return; }
    let index = dispatched_grid_index(world, gid);
    let high = bitcast<f32>(atomicLoad(&g_kdiag[index]));
    let low = bitcast<f32>(atomicLoad(&stiffness_global_tail[index]));
    atomicStore(&g_kdiag[index], bitcast<u32>(high + low));
}
