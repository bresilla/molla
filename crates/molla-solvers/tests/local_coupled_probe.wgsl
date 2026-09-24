@group(2) @binding(5) var<storage, read_write> patch_input: array<f32>;
@group(2) @binding(6) var<storage, read_write> patch_output: array<f32>;

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
            particle_delta[node] = vec4<f32>(delta,0.0);
        }
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
                    + (inertia_target-particle_delta[node][k])/u.dt;
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
        return result;
}

@compute @workgroup_size(1)
fn patch_residuals() {
    for (var sample = 0u; sample < u32(patch_input[3]); sample += 1u) {
        var x: array<f32,15>;
        for (var i = 0u; i < 15u; i += 1u) { x[i] = patch_input[192u+sample*15u+i]; }
        let result = patch_evaluate(x,-1);
        for (var i = 0u; i < 15u; i += 1u) { patch_output[sample*15u+i] = result[i]; }
    }
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

@compute @workgroup_size(1)
fn patch_solve() {
    patch_solve_impl();
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
