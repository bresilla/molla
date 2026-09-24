@group(6) @binding(6) var<storage, read_write> patch_input: array<f32>;
@group(6) @binding(7) var<storage, read_write> patch_output: array<f32>;

fn patch_force(v: u32) -> vec3<f32> {
    var force = vec3<f32>(0.0);
    for (var s = vertex_adj_offsets[v]; s < vertex_adj_offsets[v+1u]; s += 1u) {
        var f: vec3<f32>;
        var h: mat3x3<f32>;
        spring_force_hessian(v, vertex_adj_springs[s], true, &f, &h);
        force += f;
    }
    for (var t = vertex_adj_tets_offsets[v]; t < vertex_adj_tets_offsets[v+1u]; t += 2u) {
        var f: vec3<f32>;
        var h: mat3x3<f32>;
        tet_neo_hookean_block(vertex_adj_tets[t], vertex_adj_tets[t+1u], true, &f, &h);
        force += f;
    }
    return force;
}

fn patch_evaluate(x: array<f32,15>, mask: i32) -> array<f32,15> {
        var result: array<f32,15>;
        var lambda: array<f32,9>;
        var change: array<f32,9>;
        for (var i = 0u; i < 9u; i += 1u) {
            lambda[i] = x[6u+i] * patch_input[2];
            change[i] = lambda[i] - patch_input[22u+i];
        }
        for (var i = 0u; i < 2u; i += 1u) {
            let node = u32(patch_input[i]);
            var delta: vec3<f32>;
            for (var k = 0u; k < 3u; k += 1u) {
                delta[k] = patch_input[10u+3u*i+k] + u.dt*x[3u*i+k];
            }
            material_trial_nodes[i] = node;
            material_trial_delta[i] = delta;
        }
        material_trial_enabled = true;
        for (var i = 0u; i < 2u; i += 1u) {
            let node = u32(patch_input[i]);
            let force = patch_force(node);
            for (var k = 0u; k < 3u; k += 1u) {
                let coordinate = 3u*i+k;
                var impulse = 0.0;
                for (var j = 0u; j < 9u; j += 1u) {
                    impulse += patch_input[40u+j*6u+coordinate]*change[j];
                }
                let inertia_target = patch_input[16u+coordinate] + u.dt*particle_inv_mass[node]*impulse;
                result[coordinate] = u.dt*particle_inv_mass[node]*force[k]
                    + (inertia_target-material_trial_delta[i][k])/u.dt;
            }
        }
        var relative: array<f32,9>;
        for (var i = 0u; i < 9u; i += 1u) {
            var value = patch_input[31u+i];
            for (var j = 0u; j < 6u; j += 1u) {
                value += patch_input[40u+i*6u+j]*x[j];
            }
            for (var j = 0u; j < 9u; j += 1u) {
                value += patch_input[94u+i*9u+j]*change[j];
            }
            relative[i] = value;
        }
        for (var row = 0u; row < 3u; row += 1u) {
            let normal_mass = patch_input[175u+row*2u];
            let tangent_mass = patch_input[176u+row*2u];
            let ln = lambda[row*3u];
            result[6u+row*3u] =
                (max(0.0,ln+(patch_input[181u+row]-relative[row*3u])/normal_mass)-ln)*normal_mass;
            let lt = vec2<f32>(lambda[row*3u+1u],lambda[row*3u+2u]);
            var trial = lt-vec2<f32>(relative[row*3u+1u],relative[row*3u+2u])/tangent_mass;
            let limit = patch_input[184u+row]*max(0.0,ln);
            let magnitude = length(trial);
            if magnitude > limit { trial *= limit/max(magnitude,1e-30); }
            let tangent = (trial-lt)*tangent_mass;
            result[7u+row*3u] = tangent.x;
            result[8u+row*3u] = tangent.y;
            if mask >= 0 {
                result[6u+row*3u] = select(-ln*normal_mass,
                    patch_input[181u+row]-relative[row*3u], (u32(mask) & (1u<<row)) != 0u);
            }
        }
        material_trial_enabled = false;
        return result;
}

fn patch_merit(value: array<f32,15>) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < 15u; i += 1u) { sum += value[i]*value[i]; }
    return sum;
}

fn patch_maximum(value: array<f32,15>) -> f32 {
    var result = 0.0;
    for (var i = 0u; i < 15u; i += 1u) { result = max(result,abs(value[i])); }
    return result;
}

fn patch_newton_delta(jacobian: array<f32,225>, residual: array<f32,15>) -> array<f32,16> {
    var matrix: array<f32,240>;
    var answer: array<f32,16>;
    for (var i = 0u; i < 15u; i += 1u) {
        for (var j = 0u; j < 15u; j += 1u) { matrix[i*16u+j] = jacobian[i*15u+j]; }
        matrix[i*16u+15u] = -residual[i];
    }
    for (var col = 0u; col < 15u; col += 1u) {
        var pivot = col;
        var magnitude = abs(matrix[col*16u+col]);
        for (var row = col+1u; row < 15u; row += 1u) {
            let candidate = abs(matrix[row*16u+col]);
            if candidate > magnitude { pivot = row; magnitude = candidate; }
        }
        if !(magnitude > 1e-7) { return answer; }
        for (var j = col; j < 16u; j += 1u) {
            let saved = matrix[col*16u+j];
            matrix[col*16u+j] = matrix[pivot*16u+j];
            matrix[pivot*16u+j] = saved;
        }
        let diagonal = matrix[col*16u+col];
        for (var j = col; j < 16u; j += 1u) { matrix[col*16u+j] /= diagonal; }
        for (var row = 0u; row < 15u; row += 1u) {
            if row == col { continue; }
            let factor = matrix[row*16u+col];
            for (var j = col; j < 16u; j += 1u) {
                matrix[row*16u+j] -= factor*matrix[col*16u+j];
            }
        }
    }
    for (var i = 0u; i < 15u; i += 1u) { answer[i] = matrix[i*16u+15u]; }
    answer[15] = 1.0;
    return answer;
}

fn patch_solve_impl() {
    var initial: array<f32,15>;
    for (var i = 0u; i < 15u; i += 1u) { initial[i] = patch_input[192u+i]; }
    var best = initial;
    var best_merit = 1e30;
    var best_mask = -1.0;
    var best_iteration = 0.0;
    var converged = false;
    for (var mask = 0; mask < 7; mask += 1) {
        var x = initial;
        for (var iteration = 0u; iteration < 48u; iteration += 1u) {
            let full = patch_evaluate(x,-1);
            let full_merit = patch_merit(full);
            if full_merit < best_merit {
                best = x;
                best_merit = full_merit;
                best_mask = f32(mask);
                best_iteration = f32(iteration);
            }
            if patch_maximum(full) < 1e-5 { converged = true; break; }
            let value = patch_evaluate(x,mask);
            let merit = patch_merit(value);
            if merit < 1e-10 { break; }
            var jacobian: array<f32,225>;
            for (var column = 0u; column < 15u; column += 1u) {
                let epsilon = 1e-3*max(1.0,abs(x[column]));
                var plus = x;
                var minus = x;
                plus[column] += epsilon;
                minus[column] -= epsilon;
                let a = patch_evaluate(plus,mask);
                let b = patch_evaluate(minus,mask);
                for (var row = 0u; row < 15u; row += 1u) {
                    jacobian[row*15u+column] = (a[row]-b[row])/(2.0*epsilon);
                }
            }
            let delta = patch_newton_delta(jacobian,value);
            if delta[15] == 0.0 { break; }
            var scale = 1.0;
            for (var i = 0u; i < 6u; i += 1u) { scale = min(scale,1.0/max(1.0,abs(delta[i]))); }
            var accepted = false;
            for (var trial = 0u; trial < 12u; trial += 1u) {
                var candidate = x;
                for (var i = 0u; i < 15u; i += 1u) { candidate[i] += scale*delta[i]; }
                if patch_merit(patch_evaluate(candidate,mask)) < merit {
                    x = candidate;
                    accepted = true;
                    break;
                }
                scale *= 0.5;
            }
            if !accepted { break; }
        }
        if converged { break; }
    }
    let final_residual = patch_evaluate(best,-1);
    for (var i = 0u; i < 15u; i += 1u) {
        patch_output[i] = best[i];
        patch_output[15u+i] = final_residual[i];
    }
    patch_output[30] = best_mask;
    patch_output[31] = best_iteration;
    patch_output[32] = select(0.0,1.0,converged);
    patch_output[33] = patch_maximum(final_residual);
}

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
        if particle_inv_mass[first] <= 0.0 || particle_inv_mass[second] <= 0.0 { continue; }
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

var<workgroup> patch_iterate: array<f32,15>;
var<workgroup> patch_jacobian: array<f32,225>;
var<workgroup> patch_running: u32;
var<workgroup> patch_converged: u32;
var<workgroup> patch_present: u32;

@compute @workgroup_size(32)
fn patch_live_solve(@builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        patch_present = u32(patch_input[191] != 0.0);
        patch_converged = 0u;
    }
    if workgroupUniformLoad(&patch_present) == 0u { return; }
    var best: array<f32,15>;
    var best_merit = 1e30;
    var best_mask = -1.0;
    var best_iteration = 0.0;
    for (var mask = 0; mask < 7; mask += 1) {
        if lane == 0u {
            for (var i = 0u; i < 15u; i += 1u) {
                patch_iterate[i] = patch_input[192u+i];
                if mask == 0 { best[i] = patch_iterate[i]; }
            }
        }
        workgroupBarrier();
        for (var iteration = 0u; iteration < 48u; iteration += 1u) {
            var value: array<f32,15>;
            var merit = 0.0;
            if lane == 0u {
                let full = patch_evaluate(patch_iterate,-1);
                let full_merit = patch_merit(full);
                if full_merit < best_merit {
                    best = patch_iterate;
                    best_merit = full_merit;
                    best_mask = f32(mask);
                    best_iteration = f32(iteration);
                }
                patch_converged = u32(patch_maximum(full) < 1e-5);
                patch_running = 1u-patch_converged;
                if patch_running != 0u {
                    value = patch_evaluate(patch_iterate,mask);
                    merit = patch_merit(value);
                    if merit < 1e-10 { patch_running = 0u; }
                }
            }
            if workgroupUniformLoad(&patch_running) == 0u { break; }
            if lane < 15u {
                let epsilon = 1e-3*max(1.0,abs(patch_iterate[lane]));
                var plus = patch_iterate;
                var minus = patch_iterate;
                plus[lane] += epsilon;
                minus[lane] -= epsilon;
                let a = patch_evaluate(plus,mask);
                let b = patch_evaluate(minus,mask);
                for (var row = 0u; row < 15u; row += 1u) {
                    patch_jacobian[row*15u+lane] = (a[row]-b[row])/(2.0*epsilon);
                }
            }
            workgroupBarrier();
            if lane == 0u {
                let delta = patch_newton_delta(patch_jacobian,value);
                patch_running = 0u;
                if delta[15] != 0.0 {
                    var scale = 1.0;
                    for (var i = 0u; i < 6u; i += 1u) { scale = min(scale,1.0/max(1.0,abs(delta[i]))); }
                    for (var trial = 0u; trial < 12u; trial += 1u) {
                        var candidate = patch_iterate;
                        for (var i = 0u; i < 15u; i += 1u) { candidate[i] += scale*delta[i]; }
                        if patch_merit(patch_evaluate(candidate,mask)) < merit {
                            patch_iterate = candidate;
                            patch_running = 1u;
                            break;
                        }
                        scale *= 0.5;
                    }
                }
            }
            if workgroupUniformLoad(&patch_running) == 0u { break; }
        }
        if workgroupUniformLoad(&patch_converged) != 0u { break; }
    }
    if lane == 0u {
        let residual = patch_evaluate(best,-1);
        for (var i = 0u; i < 15u; i += 1u) {
            patch_output[i] = best[i];
            patch_output[15u+i] = residual[i];
        }
        patch_output[30] = best_mask;
        patch_output[31] = best_iteration;
        patch_output[32] = f32(patch_converged);
        patch_output[33] = patch_maximum(residual);
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
    patch_output[36] = 0.0;
    if patch_input[191] == 0.0 { return; }
    var accepted = patch_output[32] == 1.0 && patch_output[33] <= 1e-5;
    for (var i = 0u; i < 34u; i += 1u) {
        accepted = accepted && abs(patch_output[i]) <= 3.402823e38;
    }
    patch_output[36] = select(0.0,1.0,accepted);
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
    for (var i = 0u; i < 2u; i += 1u) {
        let node = u32(patch_input[i]);
        particle_q[node] = vec4<f32>(particle_q_in[node].xyz+particle_delta[node].xyz,0.0);
    }
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
