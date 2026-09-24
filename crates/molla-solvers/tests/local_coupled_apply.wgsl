struct PatchRow {
    normal_mass: f32, tangent_mass: f32, target_speed: f32, residual: f32,
    lambda: vec4<f32>,
}
struct PatchContact {
    normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32>,
    support: vec4<u32>, weights: vec4<f32>,
}
@group(6) @binding(0) var<storage, read_write> patch_rows: array<PatchRow>;
@group(6) @binding(1) var<storage, read> patch_projected: array<f32>;
@group(6) @binding(2) var<storage, read> patch_response: array<f32>;
@group(6) @binding(3) var<storage, read_write> patch_rigid: array<f32>;
@group(6) @binding(4) var<storage, read_write> patch_contacts: array<PatchContact>;
@group(6) @binding(5) var<storage, read> patch_schedule: array<u32>;

@compute @workgroup_size(1)
fn patch_select() {
    let rows = arrayLength(&patch_contacts);
    let blocks = rows+1u+(rows+63u)/64u+132u+u.particle_count+3u*rows;
    var best_error = -1.0;
    patch_input[191] = 0.0;
    for (var rank = 0u; rank < patch_schedule[0]; rank += 1u) {
        let row = patch_schedule[rank+1u];
        let a = patch_schedule[blocks+2u*row];
        let b = patch_schedule[blocks+2u*row+1u];
        if a == 0xffffffffu || b == 0xffffffffu { continue; }
        let error = max(patch_rows[row].residual,max(patch_rows[a].residual,patch_rows[b].residual));
        if error <= best_error { continue; }
        let first = patch_contacts[a].ids.x;
        let second = patch_contacts[b].ids.x;
        var overlap = false;
        for (var other_rank = 0u; other_rank < patch_schedule[0]; other_rank += 1u) {
            let other = patch_schedule[other_rank+1u];
            if other == row { continue; }
            let contact = patch_contacts[other];
            if contact.ids.w != 1u || contact.support.z != 2u
                || any(contact.ids.yz != patch_contacts[row].ids.yz) { continue; }
            overlap = overlap || contact.ids.x == first || contact.ids.x == second
                || contact.support.x == first || contact.support.x == second;
        }
        if overlap { continue; }
        best_error = error;
        patch_input[0] = f32(first);
        patch_input[1] = f32(second);
        patch_input[187] = f32(a);
        patch_input[188] = f32(b);
        patch_input[189] = f32(row);
        patch_input[191] = 1.0;
    }
}

@compute @workgroup_size(1)
fn patch_live_solve() {
    if patch_input[191] == 0.0 { return; }
    patch_solve_impl();
}

@compute @workgroup_size(1)
fn patch_disable_split_contacts() {
    if patch_input[191] == 0.0 { return; }
    for (var row = 0u; row < 3u; row += 1u) {
        patch_contacts[u32(patch_input[187u+row])].ids.w = 0u;
    }
}

@compute @workgroup_size(1)
fn patch_restore_contacts() {
    if patch_input[191] == 0.0 { return; }
    for (var row = 0u; row < 3u; row += 1u) {
        patch_contacts[u32(patch_input[187u+row])].ids.w = 1u;
    }
}

@compute @workgroup_size(64)
fn patch_polish_neighbors(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.work_count { return; }
    let v = particle_indices_by_color[particle_color_offsets[u.color_index]+gid.x];
    if v >= u.particle_count { return; }
    if patch_input[191] != 0.0 && (v == u32(patch_input[0]) || v == u32(patch_input[1])) { return; }
    solve_vertex(v,false,true);
}

fn patch_row_index(local_axis: u32) -> u32 {
    return u32(patch_input[187u+local_axis/3u])*3u+local_axis%3u;
}

@compute @workgroup_size(1)
fn patch_refresh() {
    if patch_input[191] == 0.0 { return; }
    let dofs = arrayLength(&patch_rigid);
    patch_input[2] = 1.0/max(particle_inv_mass[u32(patch_input[0])],particle_inv_mass[u32(patch_input[1])]);
    for (var i = 0u; i < 2u; i += 1u) {
        let node = u32(patch_input[i]);
        for (var k = 0u; k < 3u; k += 1u) {
            patch_input[4u+3u*i+k] = particle_qd[node][k];
            patch_input[10u+3u*i+k] = particle_delta[node][k];
            patch_input[16u+3u*i+k] = inertia_y[node][k];
            patch_input[192u+3u*i+k] = 0.0;
        }
    }
    for (var row = 0u; row < 3u; row += 1u) {
        let global = u32(patch_input[187u+row]);
        let contact = patch_contacts[global];
        let normal = contact.normal_gap.xyz;
        let axis = select(vec3<f32>(0.0,1.0,0.0),vec3<f32>(1.0,0.0,0.0),abs(normal.x)<0.8);
        let tangent = normalize(cross(normal,axis));
        let basis = mat3x3<f32>(normal,tangent,cross(normal,tangent));
        patch_input[175u+row*2u] = patch_rows[global].normal_mass;
        patch_input[176u+row*2u] = patch_rows[global].tangent_mass;
        patch_input[181u+row] = patch_rows[global].target_speed;
        patch_input[184u+row] = contact.material.x;
        for (var component = 0u; component < 3u; component += 1u) {
            let local_axis = row*3u+component;
            let impulse = patch_rows[global].lambda[component];
            patch_input[22u+local_axis] = impulse;
            patch_input[198u+local_axis] = impulse/patch_input[2];
            var relative = 0.0;
            for (var i = 0u; i < 2u; i += 1u) {
                let node = u32(patch_input[i]);
                var weight = 0.0;
                for (var s = 0u; s < contact.support.z; s += 1u) {
                    var support = contact.ids.x;
                    if s > 0u { support = contact.support[s-1u]; }
                    if support == node { weight += contact.weights[s]; }
                }
                for (var k = 0u; k < 3u; k += 1u) {
                    let value = weight*basis[component][k];
                    patch_input[40u+local_axis*6u+i*3u+k] = value;
                    relative += value*particle_qd[node][k];
                }
            }
            for (var d = 0u; d < dofs; d += 1u) {
                relative -= patch_projected[(global*3u+component)*dofs+d]*patch_rigid[d];
            }
            patch_input[31u+local_axis] = relative;
            for (var j = 0u; j < 9u; j += 1u) {
                var response = 0.0;
                for (var d = 0u; d < dofs; d += 1u) {
                    response += patch_projected[(global*3u+component)*dofs+d]
                        *patch_response[patch_row_index(j)*dofs+d];
                }
                patch_input[94u+local_axis*9u+j] = response;
            }
        }
    }
}

@compute @workgroup_size(1)
fn patch_apply() {
    if patch_input[191] == 0.0 { return; }
    var accepted = patch_output[32] == 1.0 && patch_output[33] <= 1e-5;
    for (var i = 0u; i < 34u; i += 1u) {
        accepted = accepted && abs(patch_output[i]) <= 3.402823e38;
    }
    patch_output[34] += 1.0;
    patch_output[35] += select(0.0,1.0,accepted);
    var change: array<f32,9>;
    for (var j = 0u; j < 9u; j += 1u) {
        change[j] = patch_input[190]*(patch_output[6u+j]*patch_input[2]-patch_input[22u+j]);
    }
    for (var i = 0u; i < 2u; i += 1u) {
        let node = u32(patch_input[i]);
        for (var k = 0u; k < 3u; k += 1u) {
            let coordinate = i*3u+k;
            particle_delta[node][k] = patch_input[10u+coordinate];
            if accepted {
                var impulse = 0.0;
                for (var j = 0u; j < 9u; j += 1u) {
                    impulse += patch_input[40u+j*6u+coordinate]*change[j];
                }
                inertia_y[node][k] += u.dt*particle_inv_mass[node]*impulse;
                particle_qd[node][k] = patch_input[4u+coordinate]+patch_input[190]*patch_output[coordinate];
                particle_delta[node][k] += u.dt*patch_input[190]*patch_output[coordinate];
            }
        }
    }
    if !accepted { return; }
    let dofs = arrayLength(&patch_rigid);
    for (var d = 0u; d < dofs; d += 1u) {
        var delta = 0.0;
        for (var j = 0u; j < 9u; j += 1u) {
            delta -= patch_response[patch_row_index(j)*dofs+d]*change[j];
        }
        patch_rigid[d] += delta;
    }
    for (var row = 0u; row < 3u; row += 1u) {
        let global = u32(patch_input[187u+row]);
        for (var k = 0u; k < 3u; k += 1u) {
            patch_rows[global].lambda[k] = patch_input[22u+row*3u+k]+change[row*3u+k];
        }
    }
}
