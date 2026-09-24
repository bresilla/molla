override BLOCK_CONTACTS: bool = false;

fn blocks_base() -> u32 { return schedule_static() + params.rows; }
fn points_base() -> u32 { return blocks_base() + 2u*params.rows; }
fn point_hash(node: u32, contact: Contact) -> u32 {
    return ((node*1664525u) ^ (contact.ids.y*1013904223u) ^ (contact.ids.z*747796405u)) % params.rows;
}

@compute @workgroup_size(64)
fn reset_contact_blocks(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.rows { return; }
    active_rows[blocks_base()+2u*gid.x] = 0xffffffffu;
    active_rows[blocks_base()+2u*gid.x+1u] = 0xffffffffu;
    active_rows[points_base()+gid.x] = 0xffffffffu;
}

@compute @workgroup_size(1)
fn map_contact_points() {
    if atomicLoad(&status) != 0u { return; }
    for (var rank = 0u; rank < active_rows[0]; rank += 1u) {
        let r = active_rows[rank+1u];
        let contact = contacts[r];
        if contact.ids.w != 1u || contact.support.z != 1u || contact.weights.x != 1.0 { continue; }
        var slot = point_hash(contact.ids.x, contact);
        for (var probe = 0u; probe < params.rows; probe += 1u) {
            let previous = active_rows[points_base()+slot];
            if previous == 0xffffffffu { active_rows[points_base()+slot] = r; break; }
            let other = contacts[previous];
            if all(other.ids.xyz == contact.ids.xyz) { break; }
            slot = (slot+1u) % params.rows;
        }
    }
}

fn find_contact_point(node: u32, contact: Contact) -> u32 {
    var slot = point_hash(node, contact);
    for (var probe = 0u; probe < params.rows; probe += 1u) {
        let r = active_rows[points_base()+slot];
        if r == 0xffffffffu { return r; }
        let other = contacts[r];
        if other.ids.x == node && all(other.ids.yz == contact.ids.yz) { return r; }
        slot = (slot+1u) % params.rows;
    }
    return 0xffffffffu;
}

@compute @workgroup_size(64)
fn group_contact_blocks(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= params.rows || atomicLoad(&status) != 0u { return; }
    let contact = contacts[r];
    if contact.ids.w != 1u || contact.support.z != 2u || any(contact.weights.xy <= vec2<f32>(0.0)) { return; }
    let a = find_contact_point(contact.ids.x, contact);
    let b = find_contact_point(contact.support.x, contact);
    if a == 0xffffffffu || b == 0xffffffffu || a == b { return; }
    if any(contacts[a].normal_gap.xyz != contact.normal_gap.xyz)
        || any(contacts[b].normal_gap.xyz != contact.normal_gap.xyz) { return; }
    let weighted_target = dot(contact.weights.xy, vec2<f32>(rows[a].target_speed, rows[b].target_speed));
    if rows[r].target_speed <= weighted_target + 1e-6 { return; }
    active_rows[blocks_base()+2u*r] = a;
    active_rows[blocks_base()+2u*r+1u] = b;
}

fn block_relative(r: u32, direction: vec3<f32>, k: u32) -> f32 {
    let contact = contacts[r];
    var value = 0.0;
    for (var i = 0u; i < contact.support.z; i += 1u) {
        value += contact.weights[i]*dot(velocities[support_node(contact,i)].xyz, direction);
    }
    if contact.ids.y != 0xffffffffu {
        for (var d = 0u; d < SWEEP_DOFS; d += 1u) {
            value -= projected[(r*3u+k)*SWEEP_DOFS+d]*(sweep_velocity[d] + sweep_delta_velocity[d]);
        }
    }
    return value;
}

fn block_delta(r: u32, direction: vec3<f32>, k: u32, impulse: f32) {
    if impulse == 0.0 { return; }
    let contact = contacts[r];
    for (var i = 0u; i < contact.support.z; i += 1u) {
        let node = support_node(contact, i);
        velocities[node] = vec4<f32>(velocities[node].xyz + direction*(impulse*contact.weights[i]*inverse_mass[node]), 0.0);
    }
    if contact.ids.y != 0xffffffffu {
        for (var d = 0u; d < SWEEP_DOFS; d += 1u) {
            accumulate_rigid_delta(d, -response[(r*3u+k)*SWEEP_DOFS+d]*impulse);
        }
    }
}

fn block_normal_mass(a: u32, b: u32) -> f32 {
    let ca = contacts[a];
    let cb = contacts[b];
    var value = 0.0;
    for (var i = 0u; i < ca.support.z; i += 1u) {
        for (var j = 0u; j < cb.support.z; j += 1u) {
            let node = support_node(ca,i);
            if node == support_node(cb,j) { value += ca.weights[i]*cb.weights[j]*inverse_mass[node]; }
        }
    }
    value *= dot(ca.normal_gap.xyz, cb.normal_gap.xyz);
    if ca.ids.y != 0xffffffffu {
        for (var d = 0u; d < SWEEP_DOFS; d += 1u) {
            value += projected[a*3u*SWEEP_DOFS+d]*response[b*3u*SWEEP_DOFS+d];
        }
    }
    return value;
}

fn solve_contact_block(r: u32) -> bool {
    let a = active_rows[blocks_base()+2u*r];
    let b = active_rows[blocks_base()+2u*r+1u];
    if a == 0xffffffffu || b == 0xffffffffu { return false; }
    let ids = vec3<u32>(a,b,r);
    var matrix: array<vec3<f32>,3>;
    var rhs = vec3<f32>(0.0);
    for (var i = 0u; i < 3u; i += 1u) {
        rhs[i] = rows[ids[i]].target_speed - block_relative(ids[i], contacts[r].normal_gap.xyz, 0u);
        for (var j = 0u; j < 3u; j += 1u) {
            matrix[i][j] = block_normal_mass(ids[i],ids[j]);
            rhs[i] += matrix[i][j]*rows[ids[j]].lambda.x;
        }
        if !finite(dot(matrix[i],matrix[i])) || matrix[i][i] <= 0.0 { return false; }
    }
    if !finite(dot(rhs,rhs)) { return false; }
    var normal = vec3<f32>(0.0);
    var found = false;
    for (var mask = 0u; mask < 7u; mask += 1u) {
        var candidate = vec3<f32>(0.0);
        let count = countOneBits(mask);
        let x = select(select(2u, 1u, (mask & 2u) != 0u), 0u, (mask & 1u) != 0u);
        let y = select(1u, 2u, (mask & 4u) != 0u);
        if count == 1u { candidate[x] = rhs[x]/matrix[x][x]; }
        if count == 2u {
            let determinant = matrix[x][x]*matrix[y][y]-matrix[x][y]*matrix[y][x];
            if determinant <= 1e-6*matrix[x][x]*matrix[y][y] { continue; }
            candidate[x] = (matrix[y][y]*rhs[x]-matrix[x][y]*rhs[y])/determinant;
            candidate[y] = (matrix[x][x]*rhs[y]-matrix[y][x]*rhs[x])/determinant;
        }
        if !finite(dot(candidate,candidate)) || any(candidate < vec3<f32>(0.0)) { continue; }
        var error = 0.0;
        for (var i = 0u; i < 3u; i += 1u) {
            let slack = dot(matrix[i],candidate)-rhs[i];
            error = max(error, select(max(0.0,-slack), abs(slack), candidate[i] > 0.0));
        }
        if error <= 1e-5 { normal = candidate; found = true; break; }
    }
    if !found { return false; }
    for (var i = 0u; i < 3u; i += 1u) {
        block_delta(ids[i], contacts[r].normal_gap.xyz, 0u, normal[i]-rows[ids[i]].lambda.x);
        rows[ids[i]].lambda.x = normal[i];
    }
    for (var i = 0u; i < 3u; i += 1u) {
        let row = rows[ids[i]];
        let directions = basis(contacts[ids[i]].normal_gap.xyz);
        var tangent = row.lambda.yz;
        if row.tangent_lipschitz > 0.0 {
            tangent -= vec2<f32>(block_relative(ids[i],directions[1],1u), block_relative(ids[i],directions[2],2u))/row.tangent_lipschitz;
        }
        let magnitude = length(tangent);
        let limit = contacts[ids[i]].material.x*normal[i];
        if magnitude > limit { tangent *= limit/magnitude; }
        if !finite(dot(tangent,tangent)) { atomicOr(&status,4u); return true; }
        block_delta(ids[i], directions[1], 1u, tangent.x-row.lambda.y);
        block_delta(ids[i], directions[2], 2u, tangent.y-row.lambda.z);
        rows[ids[i]].lambda = vec4<f32>(normal[i],tangent,0.0);
    }
    return true;
}
