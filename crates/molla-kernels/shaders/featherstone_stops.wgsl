alias StopScalars = array<array<f32, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
alias StopVectors = array<array<vec3<f32>, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
alias StopBlocks = array<array<f32, 36>, MAX_JOINTS_PER_ART>;
alias StopCounts = array<u32, MAX_JOINTS_PER_ART>;
alias StopParents = array<i32, MAX_JOINTS_PER_ART>;

// Apply the cached articulated inverse mass to a generalized impulse.
fn stop_mass_response(
    n: u32,
    parents: ptr<function, StopParents>,
    counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>,
    ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>,
    force: ptr<function, StopScalars>,
) -> StopScalars {
    var pl: array<vec3<f32>, 32>;
    var pa: array<vec3<f32>, 32>;
    var rhs: StopScalars;
    for (var back = 0u; back < n; back += 1u) {
        let k = n - 1u - back;
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            rhs[k][d] = (*force)[k][d] - dot((*sl)[k][d], pl[k]) - dot((*sa)[k][d], pa[k]);
        }
        var propagated_l = pl[k];
        var propagated_a = pa[k];
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            var scale = 0.0;
            for (var c = 0u; c < (*counts)[k]; c += 1u) {
                scale += (*inverse)[k][d*6u+c] * rhs[k][c];
            }
            propagated_l += (*ul)[k][d] * scale;
            propagated_a += (*ua)[k][d] * scale;
        }
        let parent = (*parents)[k];
        if parent >= 0 {
            pl[u32(parent)] += propagated_l;
            pa[u32(parent)] += propagated_a;
        }
    }
    var al: array<vec3<f32>, 32>;
    var aa: array<vec3<f32>, 32>;
    var result: StopScalars;
    for (var k = 0u; k < n; k += 1u) {
        var parent_l = vec3<f32>(0.0);
        var parent_a = vec3<f32>(0.0);
        let parent = (*parents)[k];
        if parent >= 0 {
            parent_l = al[u32(parent)];
            parent_a = aa[u32(parent)];
        }
        al[k] = parent_l;
        aa[k] = parent_a;
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            for (var c = 0u; c < (*counts)[k]; c += 1u) {
                let value = rhs[k][c] - dot((*ul)[k][c], parent_l) - dot((*ua)[k][c], parent_a);
                result[k][d] += (*inverse)[k][d*6u+c] * value;
            }
            al[k] += (*sl)[k][d] * result[k][d];
            aa[k] += (*sa)[k][d] * result[k][d];
        }
    }
    return result;
}

// Solve bounded scalar velocities and apply their coupled articulation reactions.
fn react_scalar_stops(
    begin: u32, n: u32,
    counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>,
    ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>,
    acceleration: ptr<function, StopScalars>,
) -> bool {
    let advance = uniforms.dt * max(1.0 - uniforms.joint_damping * uniforms.dt, 0.0);
    if advance == 0.0 { return true; }
    var parents: StopParents;
    var joints: array<u32, 32>;
    var lo: array<f32, 32>;
    var hi: array<f32, 32>;
    var velocity: array<f32, 32>;
    var rows = 0u;
    var violated = false;
    for (var k = 0u; k < n; k += 1u) {
        let j = begin + k;
        parents[k] = -1;
        for (var p = 0u; p < k; p += 1u) {
            if joint_child[begin+p] == joint_parent[j] { parents[k] = i32(p); }
        }
        let kind = joint_type[j];
        if kind != JOINT_REVOLUTE && kind != JOINT_PRISMATIC { continue; }
        let dof = joint_dof_offset[j];
        let lower = joint_limit_lower[dof];
        let upper = joint_limit_upper[dof];
        if abs(upper-lower) < 1e-12 || (lower < -1e30 && upper > 1e30) { continue; }
        let q = joint_q[joint_coord_offset[j]];
        joints[rows] = k;
        lo[rows] = (lower-q) / advance;
        hi[rows] = (upper-q) / advance;
        velocity[rows] = joint_qd[dof] + uniforms.dt * (*acceleration)[k][0];
        if !(abs(q) <= 3.402823e38 && abs(velocity[rows]) <= 3.402823e38) { return false; }
        violated = violated || velocity[rows] < lo[rows] || velocity[rows] > hi[rows];
        rows += 1u;
    }
    if !violated { return true; }

    var gram: array<array<f32, 32>, 32>;
    var unit: StopScalars;
    for (var c = 0u; c < rows; c += 1u) {
        if c > 0u { unit[joints[c-1u]][0] = 0.0; }
        unit[joints[c]][0] = 1.0;
        let response = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &unit);
        for (var r = 0u; r < rows; r += 1u) { gram[r][c] = response[joints[r]][0]; }
    }
    for (var r = 0u; r < rows; r += 1u) {
        if !(gram[r][r] > 0.0) { return false; }
        for (var c = 0u; c < r; c += 1u) {
            let value = 0.5 * (gram[r][c] + gram[c][r]);
            gram[r][c] = value;
            gram[c][r] = value;
        }
    }

    var signs: array<f32, 32>;
    var impulses: array<f32, 32>;
    var active_rows: array<u32, 32>;
    var active_count = 0u;
    var factor: array<array<f32, 32>, 32>;
    var converged = false;
    for (var iteration = 0u; iteration < 4u*rows*rows+1u; iteration += 1u) {
        var entering = rows;
        var entering_sign = 0.0;
        var worst = 0.0;
        for (var r = 0u; r < rows; r += 1u) {
            if signs[r] != 0.0 { continue; }
            var v = velocity[r];
            for (var c = 0u; c < rows; c += 1u) { v += gram[r][c] * signs[c] * impulses[c]; }
            let residual = max(lo[r]-v, v-hi[r]);
            let tolerance = 1e-7 * (1.0 + abs(velocity[r]));
            if residual > tolerance && residual / sqrt(gram[r][r]) > worst {
                entering = r;
                entering_sign = select(-1.0, 1.0, v < lo[r]);
                worst = residual / sqrt(gram[r][r]);
            }
        }
        if entering == rows { converged = true; break; }
        signs[entering] = entering_sign;
        active_rows[active_count] = entering;
        active_count += 1u;
        var solved = false;
        for (var retry = 0u; retry <= rows; retry += 1u) {
            var solution: array<f32, 32>;
            for (var r = 0u; r < active_count; r += 1u) {
                let row = active_rows[r];
                for (var c = 0u; c <= r; c += 1u) {
                    let col = active_rows[c];
                    var value = gram[row][col] * signs[row] * signs[col];
                    for (var d = 0u; d < c; d += 1u) { value -= factor[r][d] * factor[c][d]; }
                    if r == c {
                        if !(value > 0.0) { return false; }
                        factor[r][c] = sqrt(value);
                    } else { factor[r][c] = value / factor[c][c]; }
                }
                let bound = select(hi[row], lo[row], signs[row] > 0.0);
                var rhs = signs[row] * (bound - velocity[row]);
                for (var c = 0u; c < r; c += 1u) { rhs -= factor[r][c] * solution[c]; }
                solution[r] = rhs / factor[r][r];
            }
            var trial: array<f32, 32>;
            for (var r = 0u; r < rows; r += 1u) { trial[r] = 0.0; }
            var positive = true;
            for (var back = 0u; back < active_count; back += 1u) {
                let r = active_count - 1u - back;
                var value = solution[r];
                for (var c = r+1u; c < active_count; c += 1u) { value -= factor[c][r] * solution[c]; }
                solution[r] = value / factor[r][r];
                trial[active_rows[r]] = solution[r];
                positive = positive && solution[r] > 0.0;
            }
            if positive { impulses = trial; solved = true; break; }
            var alpha = 1.0;
            for (var r = 0u; r < active_count; r += 1u) {
                let row = active_rows[r];
                if trial[row] <= 0.0 {
                    if impulses[row] == 0.0 { alpha = 0.0; }
                    else { alpha = min(alpha, impulses[row] / (impulses[row]-trial[row])); }
                }
            }
            var retained = 0u;
            for (var r = 0u; r < active_count; r += 1u) {
                let row = active_rows[r];
                impulses[row] += alpha * (trial[row]-impulses[row]);
                if impulses[row] > 0.0 { active_rows[retained] = row; retained += 1u; }
                else { impulses[row] = 0.0; signs[row] = 0.0; }
            }
            active_count = retained;
        }
        if !solved { return false; }
    }
    if !converged { return false; }
    var force: StopScalars;
    for (var r = 0u; r < rows; r += 1u) { force[joints[r]][0] = signs[r] * impulses[r]; }
    var response = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &force);
    for (var refinement = 0u; refinement < 4u; refinement += 1u) {
        var residual: array<f32, 32>;
        var needs_refinement = false;
        for (var r = 0u; r < active_count; r += 1u) {
            let row = active_rows[r];
            let bound = select(hi[row], lo[row], signs[row] > 0.0);
            let error = bound - (velocity[row] + response[joints[row]][0]);
            needs_refinement = needs_refinement || abs(error) > 1e-7 * (1.0 + abs(velocity[row]));
            var value = signs[row] * error;
            for (var c = 0u; c < r; c += 1u) { value -= factor[r][c] * residual[c]; }
            residual[r] = value / factor[r][r];
        }
        if !needs_refinement { break; }
        var correction: StopScalars;
        for (var back = 0u; back < active_count; back += 1u) {
            let r = active_count - 1u - back;
            let row = active_rows[r];
            var value = residual[r];
            for (var c = r+1u; c < active_count; c += 1u) { value -= factor[c][r] * residual[c]; }
            residual[r] = value / factor[r][r];
            if impulses[row] + residual[r] < 0.0 { return false; }
            impulses[row] += residual[r];
            correction[joints[row]][0] = signs[row] * residual[r];
        }
        let delta = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &correction);
        for (var k = 0u; k < n; k += 1u) {
            for (var d = 0u; d < (*counts)[k]; d += 1u) { response[k][d] += delta[k][d]; }
        }
    }
    for (var r = 0u; r < rows; r += 1u) {
        let v = velocity[r] + response[joints[r]][0];
        let tolerance = 2e-5 * (1.0 + abs(velocity[r]));
        if !(v >= lo[r]-tolerance && v <= hi[r]+tolerance) { return false; }
        if impulses[r] > 0.0 {
            let bound = select(hi[r], lo[r], signs[r] > 0.0);
            if abs(v-bound) > tolerance { return false; }
        }
    }
    for (var k = 0u; k < n; k += 1u) {
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            (*acceleration)[k][d] += response[k][d] / uniforms.dt;
        }
    }
    return true;
}
