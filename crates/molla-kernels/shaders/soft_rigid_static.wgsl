override SEPARATE_STATIC: bool = false;

fn schedule_base() -> u32 { return params.rows + 1u + (params.rows+63u)/64u; }
fn schedule_nodes() -> u32 { return schedule_base() + 132u; }
fn schedule_levels() -> u32 { return schedule_nodes() + params.particles; }
fn schedule_dynamic() -> u32 { return schedule_levels() + params.rows; }
fn schedule_static() -> u32 { return schedule_dynamic() + params.rows; }

@compute @workgroup_size(1)
fn schedule_static_contacts() {
    let base = schedule_base();
    active_rows[base] = 0u;
    active_rows[base+1u] = 0u;
    active_rows[base+2u] = 0u;
    if atomicLoad(&status) != 0u { return; }
    for (var node = 0u; node < params.particles; node += 1u) {
        active_rows[schedule_nodes()+node] = 0u;
    }
    for (var level = 0u; level < 64u; level += 1u) {
        active_rows[base+68u+level] = 0u;
    }
    var dynamic_count = 0u;
    for (var rank = 0u; rank < active_rows[0]; rank += 1u) {
        let r = active_rows[rank+1u];
        let contact = contacts[r];
        if contact.ids.y == 0xffffffffu { continue; }
        active_rows[schedule_dynamic()+dynamic_count] = r;
        dynamic_count += 1u;
        for (var i = 0u; i < contact.support.z; i += 1u) {
            active_rows[schedule_nodes()+support_node(contact,i)] = 0xffffffffu;
        }
    }
    var levels = 0u;
    for (var rank = 0u; rank < active_rows[0]; rank += 1u) {
        let r = active_rows[rank+1u];
        let contact = contacts[r];
        if contact.ids.y != 0xffffffffu { continue; }
        var level = 0u;
        for (var i = 0u; i < contact.support.z; i += 1u) {
            level = max(level, active_rows[schedule_nodes()+support_node(contact,i)]);
        }
        if level >= 64u { return; }
        active_rows[schedule_levels()+r] = level;
        active_rows[base+68u+level] += 1u;
        for (var i = 0u; i < contact.support.z; i += 1u) {
            active_rows[schedule_nodes()+support_node(contact,i)] = level+1u;
        }
        levels = max(levels, level+1u);
    }
    var offset = 0u;
    for (var level = 0u; level < 64u; level += 1u) {
        let count = active_rows[base+68u+level];
        active_rows[base+3u+level] = offset;
        active_rows[base+68u+level] = offset;
        offset += count;
    }
    active_rows[base+67u] = offset;
    for (var rank = 0u; rank < active_rows[0]; rank += 1u) {
        let r = active_rows[rank+1u];
        if contacts[r].ids.y != 0xffffffffu { continue; }
        let level = active_rows[schedule_levels()+r];
        let cursor = base+68u+level;
        active_rows[schedule_static()+active_rows[cursor]] = r;
        active_rows[cursor] += 1u;
    }
    active_rows[base+1u] = dynamic_count;
    active_rows[base+2u] = levels;
    active_rows[base] = 1u;
}

@compute @workgroup_size(64)
fn solve_static_contacts(@builtin(local_invocation_index) lane: u32) {
    let base = schedule_base();
    if lane == 0u {
        sweep_count = active_rows[base+2u];
        sweep_fault = atomicLoad(&status) | (1u-active_rows[base]);
    }
    let levels = workgroupUniformLoad(&sweep_count);
    let fault = workgroupUniformLoad(&sweep_fault);
    if fault != 0u { return; }
    for (var iteration = 0u; iteration < params.iterations; iteration += 1u) {
        for (var level = 0u; level < levels; level += 1u) {
            let start = active_rows[base+3u+level];
            let end = active_rows[base+4u+level];
            for (var index = start+lane; index < end; index += 64u) {
                let r = active_rows[schedule_static()+index];
                let failure = cooperative_row(r);
                if failure != 0u { atomicOr(&status, failure); }
            }
            storageBarrier();
        }
    }
}
